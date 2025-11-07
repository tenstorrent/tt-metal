// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <atomic>
#include <mutex>

namespace tt::tt_metal {

/**
 * @brief Client for reporting allocations to the allocation tracking server
 *
 * This class provides a singleton interface for TT-Metal's allocator to report
 * memory allocations to a centralized tracking server for cross-process monitoring.
 *
 * Usage:
 *   In allocator.cpp:
 *     if (AllocationClient::is_enabled()) {
 *         AllocationClient::report_allocation(device_id, size, buffer_type, address);
 *     }
 *
 * Enable tracking via environment variable:
 *   export TT_ALLOC_TRACKING_ENABLED=1
 *
 * The tracking server must be running:
 *   ./allocation_server_poc &
 */
class AllocationClient {
public:
    /**
     * @brief Report an allocation to the tracking server
     *
     * @param device_id Device ID (0-7)
     * @param size Size of allocation in bytes
     * @param buffer_type Buffer type (0=DRAM, 1=L1, 2=L1_SMALL, 3=TRACE)
     * @param buffer_id Unique identifier (typically the address)
     */
    static void report_allocation(int device_id, uint64_t size, uint8_t buffer_type, uint64_t buffer_id);

    /**
     * @brief Report a deallocation to the tracking server
     *
     * @param buffer_id The buffer ID that was reported in report_allocation()
     */
    static void report_deallocation(int device_id, uint64_t buffer_id);

    /**
     * @brief Report a circular buffer allocation to the tracking server
     *
     * @param device_id Device ID (0-7)
     * @param size Size of CB allocation in bytes
     * @param cb_id Unique identifier (typically the CB address)
     */
    static void report_cb_allocation(int device_id, uint64_t size, uint64_t cb_id);

    /**
     * @brief Report a circular buffer deallocation to the tracking server
     *
     * @param device_id Device ID
     * @param size Size of the CB being freed (for accurate tracking)
     * @param cb_id The CB ID that was reported in report_cb_allocation()
     */
    static void report_cb_deallocation(int device_id, uint64_t size, uint64_t cb_id);

    /**
     * @brief Report a kernel load to the tracking server
     *
     * @param device_id Device ID (0-7)
     * @param size Size of kernel binary in bytes
     * @param kernel_id Unique identifier for the kernel
     */
    static void report_kernel_load(int device_id, uint64_t size, uint64_t kernel_id, uint8_t kernel_type = 0);

    /**
     * @brief Report a kernel unload to the tracking server
     *
     * @param device_id Device ID
     * @param size The size of the kernel to unload (must match report_kernel_load size)
     * @param kernel_id The kernel ID that was reported in report_kernel_load()
     */
    static void report_kernel_unload(int device_id, uint64_t size, uint64_t kernel_id);

    /**
     * @brief Check if allocation tracking is enabled
     *
     * @return true if TT_ALLOC_TRACKING_ENABLED=1, false otherwise
     */
    static bool is_enabled();

private:
    AllocationClient();
    ~AllocationClient();

    // Singleton instance
    static AllocationClient& instance();

    // Internal state
    int socket_fd_;
    std::atomic<bool> enabled_;
    std::atomic<bool> connected_;
    std::mutex socket_mutex_;  // Protects socket operations from concurrent threads

    // Connect to server (lazy initialization)
    bool connect_to_server();

    // Internal send methods
    void send_allocation_message(int device_id, uint64_t size, uint8_t buffer_type, uint64_t buffer_id);
    void send_deallocation_message(int device_id, uint64_t buffer_id);
    void send_cb_allocation_message(int device_id, uint64_t size, uint64_t cb_id);
    void send_cb_deallocation_message(int device_id, uint64_t size, uint64_t cb_id);
    void send_kernel_load_message(int device_id, uint64_t size, uint64_t kernel_id, uint8_t kernel_type = 0);
    void send_kernel_unload_message(int device_id, uint64_t size, uint64_t kernel_id);

    // Disable copy/move
    AllocationClient(const AllocationClient&) = delete;
    AllocationClient& operator=(const AllocationClient&) = delete;
    AllocationClient(AllocationClient&&) = delete;
    AllocationClient& operator=(AllocationClient&&) = delete;
};

}  // namespace tt::tt_metal
