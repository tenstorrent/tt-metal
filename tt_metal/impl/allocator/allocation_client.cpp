// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "allocation_client.hpp"
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>
#include <cstring>
#include <cstdlib>
#include <chrono>
#include <iostream>
#include <cerrno>

#define TT_ALLOC_SERVER_SOCKET "/tmp/tt_allocation_server.sock"

namespace tt::tt_metal {

// Message protocol matching allocation_server_poc.cpp
struct __attribute__((packed)) AllocMessage {
    enum Type : uint8_t {
        ALLOC = 1,
        FREE = 2,
        QUERY = 3,
        RESPONSE = 4,
        DUMP_REMAINING = 5,
        DEVICE_INFO_QUERY = 6,
        DEVICE_INFO_RESPONSE = 7,
        CB_ALLOC = 8,       // Circular buffer allocation
        CB_FREE = 9,        // Circular buffer free
        KERNEL_LOAD = 10,   // Kernel loaded
        KERNEL_UNLOAD = 11  // Kernel unloaded
    };

    Type type;            // 1 byte
    uint8_t pad1[3];      // 3 bytes padding
    int32_t device_id;    // 4 bytes
    uint64_t size;        // 8 bytes
    uint8_t buffer_type;  // 1 byte
    uint8_t pad2[3];      // 3 bytes padding
    int32_t process_id;   // 4 bytes
    uint64_t buffer_id;   // 8 bytes
    uint64_t timestamp;   // 8 bytes

    // Response fields (unused by client for alloc/free)
    uint64_t dram_allocated;
    uint64_t l1_allocated;
    uint64_t l1_small_allocated;
    uint64_t trace_allocated;
    uint64_t cb_allocated;      // NEW: Circular buffer memory
    uint64_t kernel_allocated;  // NEW: Kernel code memory

    // Device info fields (unused by client for alloc/free, used by server for device queries)
    uint64_t total_dram_size;
    uint64_t total_l1_size;
    uint32_t arch_type;  // 0=Invalid, 1=Grayskull, 2=Wormhole_B0, 3=Blackhole, 4=Quasar
    uint32_t num_dram_channels;
    uint32_t dram_size_per_channel;
    uint32_t l1_size_per_core;
    uint32_t is_available;
    uint32_t num_devices;

    // Ring buffer stats (for real-time kernel L1 usage)
    uint32_t ringbuffer_total_size;    // Total ring buffer capacity (~67KB)
    uint32_t ringbuffer_used_bytes;    // Currently used by cached kernels
    uint32_t ringbuffer_num_programs;  // Number of programs cached
    uint32_t pad3;                     // Padding for alignment
    // Total: 1+3+4+8+1+3+4+8+8+48+16+16 = 128 bytes
};

AllocationClient::AllocationClient() : socket_fd_(-1), enabled_(false), connected_(false) {
    // Check environment variable
    const char* env_enabled = std::getenv("TT_ALLOC_TRACKING_ENABLED");
    if (env_enabled && std::string(env_enabled) == "1") {
        enabled_ = true;

        // Attempt to connect (lazy, will retry on first use)
        connect_to_server();
    }
}

AllocationClient::~AllocationClient() {
    if (socket_fd_ >= 0) {
        close(socket_fd_);
        socket_fd_ = -1;
    }
}

AllocationClient& AllocationClient::instance() {
    static AllocationClient inst;
    return inst;
}

bool AllocationClient::connect_to_server() {
    if (connected_) {
        return true;
    }

    socket_fd_ = socket(AF_UNIX, SOCK_STREAM, 0);
    if (socket_fd_ < 0) {
        return false;
    }

    struct sockaddr_un addr;
    memset(&addr, 0, sizeof(addr));
    addr.sun_family = AF_UNIX;
    strncpy(addr.sun_path, TT_ALLOC_SERVER_SOCKET, sizeof(addr.sun_path) - 1);

    if (connect(socket_fd_, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
        close(socket_fd_);
        socket_fd_ = -1;

        // Only warn once
        static bool warned = false;
        if (!warned) {
            std::cerr << "[TT-Metal] Warning: Allocation tracking enabled but server not available at "
                      << TT_ALLOC_SERVER_SOCKET << std::endl;
            warned = true;
        }
        return false;
    }

    // Increase socket buffer sizes to handle bursts from multiple devices
    // This significantly reduces blocking frequency during rapid allocations
    int sndbuf_size = 1024 * 1024;  // 1MB send buffer
    if (setsockopt(socket_fd_, SOL_SOCKET, SO_SNDBUF, &sndbuf_size, sizeof(sndbuf_size)) < 0) {
        // Non-fatal, just log
        std::cerr << "[TT-Metal] Warning: Could not set socket send buffer size" << std::endl;
    }

    connected_ = true;
    return true;
}

void AllocationClient::send_allocation_message(int device_id, uint64_t size, uint8_t buffer_type, uint64_t buffer_id) {
    if (!connect_to_server()) {
        return;
    }

    AllocMessage msg;
    memset(&msg, 0, sizeof(msg));
    msg.type = AllocMessage::ALLOC;
    msg.device_id = device_id;
    msg.size = size;
    msg.buffer_type = buffer_type;
    msg.process_id = getpid();
    msg.buffer_id = buffer_id;
    msg.timestamp = std::chrono::system_clock::now().time_since_epoch().count();

    // Serialize socket writes to prevent message interleaving from multiple threads
    std::lock_guard<std::mutex> lock(socket_mutex_);

    // CRITICAL: Use blocking send to ensure message is fully delivered
    // MSG_DONTWAIT was causing dropped messages when socket buffer filled up
    size_t total_sent = 0;
    while (total_sent < sizeof(msg)) {
        ssize_t sent = send(
            socket_fd_,
            reinterpret_cast<const char*>(&msg) + total_sent,
            sizeof(msg) - total_sent,
            0);  // Blocking send
        if (sent < 0) {
            if (errno == EINTR) {
                // Interrupted by signal, retry
                continue;
            }
            // Connection lost, mark as disconnected
            connected_ = false;
            return;
        }
        total_sent += sent;
    }
}

void AllocationClient::send_deallocation_message(int device_id, uint64_t buffer_id) {
    if (!connect_to_server()) {
        return;
    }

    AllocMessage msg;
    memset(&msg, 0, sizeof(msg));
    msg.type = AllocMessage::FREE;
    msg.device_id = device_id;
    msg.buffer_id = buffer_id;
    msg.process_id = getpid();

    // Serialize socket writes to prevent message interleaving from multiple threads
    std::lock_guard<std::mutex> lock(socket_mutex_);

    // CRITICAL: Use blocking send to ensure message is fully delivered
    // MSG_DONTWAIT was causing dropped messages when socket buffer filled up
    size_t total_sent = 0;
    while (total_sent < sizeof(msg)) {
        ssize_t sent = send(
            socket_fd_,
            reinterpret_cast<const char*>(&msg) + total_sent,
            sizeof(msg) - total_sent,
            0);  // Blocking send
        if (sent < 0) {
            if (errno == EINTR) {
                // Interrupted by signal, retry
                continue;
            }
            // Connection lost, mark as disconnected
            connected_ = false;
            return;
        }
        total_sent += sent;
    }
}

// Circular buffer allocation/deallocation
void AllocationClient::send_cb_allocation_message(int device_id, uint64_t size, uint64_t cb_id) {
    if (!connect_to_server()) {
        return;
    }

    AllocMessage msg;
    memset(&msg, 0, sizeof(msg));
    msg.type = AllocMessage::CB_ALLOC;
    msg.device_id = device_id;
    msg.size = size;
    msg.buffer_id = cb_id;
    msg.process_id = getpid();
    msg.timestamp = std::chrono::system_clock::now().time_since_epoch().count();

    std::lock_guard<std::mutex> lock(socket_mutex_);
    size_t total_sent = 0;
    while (total_sent < sizeof(msg)) {
        ssize_t sent = send(socket_fd_, reinterpret_cast<const char*>(&msg) + total_sent, sizeof(msg) - total_sent, 0);
        if (sent < 0) {
            if (errno == EINTR) {
                continue;
            }
            connected_ = false;
            return;
        }
        total_sent += sent;
    }
}

void AllocationClient::send_cb_deallocation_message(int device_id, uint64_t size, uint64_t cb_id) {
    if (!connect_to_server()) {
        return;
    }

    AllocMessage msg;
    memset(&msg, 0, sizeof(msg));
    msg.type = AllocMessage::CB_FREE;
    msg.device_id = device_id;
    msg.size = size;  // Include size for proper deallocation tracking
    msg.buffer_id = cb_id;
    msg.process_id = getpid();

    std::lock_guard<std::mutex> lock(socket_mutex_);
    size_t total_sent = 0;
    while (total_sent < sizeof(msg)) {
        ssize_t sent = send(socket_fd_, reinterpret_cast<const char*>(&msg) + total_sent, sizeof(msg) - total_sent, 0);
        if (sent < 0) {
            if (errno == EINTR) {
                continue;
            }
            connected_ = false;
            return;
        }
        total_sent += sent;
    }
}

// Kernel load/unload
void AllocationClient::send_kernel_load_message(int device_id, uint64_t size, uint64_t kernel_id, uint8_t kernel_type) {
    if (!connect_to_server()) {
        return;
    }

    AllocMessage msg;
    memset(&msg, 0, sizeof(msg));
    msg.type = AllocMessage::KERNEL_LOAD;
    msg.device_id = device_id;
    msg.size = size;
    msg.buffer_id = kernel_id;
    msg.buffer_type = kernel_type;  // 0=App, 1=Fabric, 2=Dispatch
    msg.process_id = getpid();
    msg.timestamp = std::chrono::system_clock::now().time_since_epoch().count();

    std::lock_guard<std::mutex> lock(socket_mutex_);
    size_t total_sent = 0;
    while (total_sent < sizeof(msg)) {
        ssize_t sent = send(socket_fd_, reinterpret_cast<const char*>(&msg) + total_sent, sizeof(msg) - total_sent, 0);
        if (sent < 0) {
            if (errno == EINTR) {
                continue;
            }
            connected_ = false;
            return;
        }
        total_sent += sent;
    }
}

void AllocationClient::send_kernel_unload_message(int device_id, uint64_t size, uint64_t kernel_id) {
    if (!connect_to_server()) {
        return;
    }

    AllocMessage msg;
    memset(&msg, 0, sizeof(msg));
    msg.type = AllocMessage::KERNEL_UNLOAD;
    msg.device_id = device_id;
    msg.size = size;  // CRITICAL: Pass size for proper deallocation tracking
    msg.buffer_id = kernel_id;
    msg.process_id = getpid();

    std::lock_guard<std::mutex> lock(socket_mutex_);
    size_t total_sent = 0;
    while (total_sent < sizeof(msg)) {
        ssize_t sent = send(socket_fd_, reinterpret_cast<const char*>(&msg) + total_sent, sizeof(msg) - total_sent, 0);
        if (sent < 0) {
            if (errno == EINTR) {
                continue;
            }
            connected_ = false;
            return;
        }
        total_sent += sent;
    }
}

// Static interface methods
void AllocationClient::report_allocation(int device_id, uint64_t size, uint8_t buffer_type, uint64_t buffer_id) {
    auto& inst = instance();
    if (inst.enabled_) {
        inst.send_allocation_message(device_id, size, buffer_type, buffer_id);
    }
}

void AllocationClient::report_deallocation(int device_id, uint64_t buffer_id) {
    auto& inst = instance();
    if (inst.enabled_) {
        inst.send_deallocation_message(device_id, buffer_id);
    }
}

void AllocationClient::report_cb_allocation(int device_id, uint64_t size, uint64_t cb_id) {
    auto& inst = instance();
    if (inst.enabled_) {
        inst.send_cb_allocation_message(device_id, size, cb_id);
    }
}

void AllocationClient::report_cb_deallocation(int device_id, uint64_t size, uint64_t cb_id) {
    auto& inst = instance();
    if (inst.enabled_) {
        inst.send_cb_deallocation_message(device_id, size, cb_id);
    }
}

void AllocationClient::report_kernel_load(int device_id, uint64_t size, uint64_t kernel_id, uint8_t kernel_type) {
    auto& inst = instance();
    if (inst.enabled_) {
        inst.send_kernel_load_message(device_id, size, kernel_id, kernel_type);
    }
}

void AllocationClient::report_kernel_unload(int device_id, uint64_t size, uint64_t kernel_id) {
    auto& inst = instance();
    if (inst.enabled_) {
        inst.send_kernel_unload_message(device_id, size, kernel_id);
    }
}

bool AllocationClient::is_enabled() { return instance().enabled_; }

}  // namespace tt::tt_metal
