// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Instrumented helper functions for tracking CB and Kernel memory
// Use these instead of direct TT-Metal APIs to automatically report to allocation server

#pragma once

#include <tt-metalium/host_api.hpp>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>
#include <cstring>

namespace tt::tt_metal::instrumented {

// Message types for CB and Kernel tracking (extend AllocMessage protocol)
enum class MemoryEventType : uint8_t { CB_ALLOC = 8, CB_FREE = 9, KERNEL_LOAD = 10, KERNEL_UNLOAD = 11 };

struct __attribute__((packed)) CBKernelMessage {
    MemoryEventType type;
    uint8_t pad1[3];
    int32_t device_id;
    uint64_t size;
    uint64_t id;  // CB index or kernel ID
    int32_t process_id;
    uint8_t pad2[4];
};

// Helper to send CB/Kernel events to allocation server
inline bool send_memory_event(MemoryEventType type, int device_id, uint64_t size, uint64_t id = 0) {
    static int socket_fd = -1;
    const char* SOCKET_PATH = "/tmp/tt_allocation_server.sock";

    // Try to connect if not connected
    if (socket_fd < 0) {
        socket_fd = socket(AF_UNIX, SOCK_STREAM, 0);
        if (socket_fd < 0) {
            return false;
        }

        struct sockaddr_un addr;
        memset(&addr, 0, sizeof(addr));
        addr.sun_family = AF_UNIX;
        strncpy(addr.sun_path, SOCKET_PATH, sizeof(addr.sun_path) - 1);

        if (connect(socket_fd, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
            close(socket_fd);
            socket_fd = -1;
            return false;  // Server not running
        }
    }

    CBKernelMessage msg;
    memset(&msg, 0, sizeof(msg));
    msg.type = type;
    msg.device_id = device_id;
    msg.size = size;
    msg.id = id;
    msg.process_id = getpid();

    send(socket_fd, &msg, sizeof(msg), 0);
    return true;
}

// Instrumented CreateCircularBuffer that reports to allocation server
inline std::shared_ptr<CircularBuffer> CreateCircularBufferWithTracking(
    Device* device, const std::map<uint8_t, tt::DataFormat>& data_format_spec, const CircularBufferConfig& config) {
    // Create the actual CB
    auto cb = CreateCircularBuffer(device, data_format_spec, config);

    // Report to allocation server
    uint32_t total_size = config.total_size();
    send_memory_event(MemoryEventType::CB_ALLOC, device->id(), total_size, config.buffer_index());

    return cb;
}

// Instrumented Program::add_kernel that reports to allocation server
inline KernelHandle AddKernelWithTracking(
    Program& program,
    const std::variant<DataMovementConfig, ComputeConfig, EthernetConfig>& kernel_config,
    const CoreRangeSet& core_ranges) {
    // Get kernel size before adding
    // Note: This is an approximation since we don't have direct access to binary size
    uint64_t estimated_kernel_size = 50 * 1024;  // Conservative 50KB estimate per kernel

    // Add the kernel
    auto kernel_handle = program.add_kernel(kernel_config, core_ranges);

    // Report to allocation server
    // Use program ID + kernel handle as unique ID
    uint64_t kernel_id = (static_cast<uint64_t>(program.id()) << 32) | kernel_handle;

    send_memory_event(
        MemoryEventType::KERNEL_LOAD,
        program.device()->id(),
        estimated_kernel_size * core_ranges.size(),  // Size × num cores
        kernel_id);

    return kernel_handle;
}

}  // namespace tt::tt_metal::instrumented
