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
    enum Type : uint8_t { ALLOC = 1, FREE = 2, QUERY = 3, RESPONSE = 4 };

    Type type;            // 1 byte
    uint8_t pad1[3];      // 3 bytes padding
    int32_t device_id;    // 4 bytes
    uint64_t size;        // 8 bytes
    uint8_t buffer_type;  // 1 byte
    uint8_t pad2[3];      // 3 bytes padding
    int32_t process_id;   // 4 bytes
    uint64_t buffer_id;   // 8 bytes
    uint64_t timestamp;   // 8 bytes

    // Response fields (unused by client)
    uint64_t dram_allocated;
    uint64_t l1_allocated;
    uint64_t l1_small_allocated;
    uint64_t trace_allocated;
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

bool AllocationClient::is_enabled() { return instance().enabled_; }

}  // namespace tt::tt_metal
