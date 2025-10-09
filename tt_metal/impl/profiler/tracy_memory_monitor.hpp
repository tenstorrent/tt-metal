// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <atomic>
#include <array>
#include <mutex>
#include <unordered_map>
#include <vector>
#include <chrono>
#include <cstdint>

namespace tt::tt_metal {

/**
 * @brief Real-time memory monitor that integrates with Tracy profiling
 *
 * This class provides:
 * - Real-time queryable memory statistics per device
 * - Integration with Tracy profiler for detailed analysis
 * - Low-overhead atomic counters for concurrent access
 * - Cross-process visibility when used with Tracy
 *
 * Unlike the allocation_server_poc which requires a separate server process,
 * this monitor is embedded in the application and leverages Tracy's infrastructure.
 */
class TracyMemoryMonitor {
public:
    static constexpr int MAX_DEVICES = 8;

    enum class BufferType : uint8_t { DRAM = 0, L1 = 1, SYSTEM_MEMORY = 2, L1_SMALL = 3, TRACE = 4 };

    // Snapshot of memory statistics (copyable/moveable)
    struct DeviceMemoryStats {
        uint64_t dram_allocated = 0;
        uint64_t l1_allocated = 0;
        uint64_t system_memory_allocated = 0;
        uint64_t l1_small_allocated = 0;
        uint64_t trace_allocated = 0;
        uint64_t num_buffers = 0;
        uint64_t total_allocs = 0;
        uint64_t total_frees = 0;

        // Get total allocated across all buffer types
        uint64_t get_total_allocated() const {
            return dram_allocated + l1_allocated + system_memory_allocated + l1_small_allocated + trace_allocated;
        }

        // Get allocation for specific buffer type
        uint64_t get_allocated(BufferType type) const {
            switch (type) {
                case BufferType::DRAM: return dram_allocated;
                case BufferType::L1: return l1_allocated;
                case BufferType::SYSTEM_MEMORY: return system_memory_allocated;
                case BufferType::L1_SMALL: return l1_small_allocated;
                case BufferType::TRACE: return trace_allocated;
                default: return 0;
            }
        }
    };

    struct BufferInfo {
        uint64_t buffer_id = 0;
        int device_id = 0;
        uint64_t size = 0;
        BufferType buffer_type = BufferType::DRAM;
        std::chrono::steady_clock::time_point alloc_time{};
    };

private:
    // Internal atomic statistics (for thread-safe updates)
    struct AtomicDeviceStats {
        std::atomic<uint64_t> dram_allocated{0};
        std::atomic<uint64_t> l1_allocated{0};
        std::atomic<uint64_t> system_memory_allocated{0};
        std::atomic<uint64_t> l1_small_allocated{0};
        std::atomic<uint64_t> trace_allocated{0};
        std::atomic<uint64_t> num_buffers{0};
        std::atomic<uint64_t> total_allocs{0};
        std::atomic<uint64_t> total_frees{0};
    };

    // Per-device memory statistics (lock-free for queries)
    std::array<AtomicDeviceStats, MAX_DEVICES> device_stats_;

    // Active buffer tracking (for leak detection and detailed queries)
    mutable std::mutex registry_mutex_;

    struct BufferKey {
        int device_id;
        uint64_t buffer_id;

        bool operator==(const BufferKey& other) const {
            return device_id == other.device_id && buffer_id == other.buffer_id;
        }
    };

    struct BufferKeyHash {
        std::size_t operator()(const BufferKey& k) const {
            return std::hash<int>()(k.device_id) ^ (std::hash<uint64_t>()(k.buffer_id) << 1);
        }
    };

    std::unordered_map<BufferKey, BufferInfo, BufferKeyHash> active_buffers_;

    // Singleton instance
    static TracyMemoryMonitor& instance_impl() {
        static TracyMemoryMonitor instance;
        return instance;
    }

    TracyMemoryMonitor() = default;

    // Get Tracy pool name for buffer type
    static const char* get_tracy_pool_name(BufferType type, int device_id);

public:
    // Singleton access
    static TracyMemoryMonitor& instance() { return instance_impl(); }

    // Disable copy/move
    TracyMemoryMonitor(const TracyMemoryMonitor&) = delete;
    TracyMemoryMonitor& operator=(const TracyMemoryMonitor&) = delete;
    TracyMemoryMonitor(TracyMemoryMonitor&&) = delete;
    TracyMemoryMonitor& operator=(TracyMemoryMonitor&&) = delete;

    /**
     * @brief Track a buffer allocation
     *
     * This function:
     * 1. Updates local atomic counters for real-time queries
     * 2. Sends allocation event to Tracy profiler (if enabled)
     * 3. Records buffer info for leak detection
     *
     * @param device_id Device ID (0-7)
     * @param buffer_id Unique buffer address/ID
     * @param size Buffer size in bytes
     * @param buffer_type Type of buffer (DRAM, L1, etc)
     */
    void track_allocation(int device_id, uint64_t buffer_id, uint64_t size, BufferType buffer_type);

    /**
     * @brief Track a buffer deallocation
     *
     * @param device_id Device ID (0-7)
     * @param buffer_id Unique buffer address/ID
     */
    void track_deallocation(int device_id, uint64_t buffer_id);

    /**
     * @brief Query memory statistics for a specific device
     *
     * This is a lock-free operation for real-time monitoring.
     *
     * @param device_id Device ID (0-7)
     * @return Current memory statistics snapshot
     */
    DeviceMemoryStats query_device(int device_id) const {
        if (device_id < 0 || device_id >= MAX_DEVICES) {
            return DeviceMemoryStats{};
        }
        // Create a snapshot by loading each atomic value
        DeviceMemoryStats result;
        result.dram_allocated = device_stats_[device_id].dram_allocated.load(std::memory_order_relaxed);
        result.l1_allocated = device_stats_[device_id].l1_allocated.load(std::memory_order_relaxed);
        result.system_memory_allocated =
            device_stats_[device_id].system_memory_allocated.load(std::memory_order_relaxed);
        result.l1_small_allocated = device_stats_[device_id].l1_small_allocated.load(std::memory_order_relaxed);
        result.trace_allocated = device_stats_[device_id].trace_allocated.load(std::memory_order_relaxed);
        result.num_buffers = device_stats_[device_id].num_buffers.load(std::memory_order_relaxed);
        result.total_allocs = device_stats_[device_id].total_allocs.load(std::memory_order_relaxed);
        result.total_frees = device_stats_[device_id].total_frees.load(std::memory_order_relaxed);
        return result;
    }

    /**
     * @brief Query memory statistics for all devices
     *
     * @return Array of statistics for all devices
     */
    std::array<DeviceMemoryStats, MAX_DEVICES> query_all_devices() const {
        std::array<DeviceMemoryStats, MAX_DEVICES> result;
        for (int i = 0; i < MAX_DEVICES; i++) {
            result[i].dram_allocated = device_stats_[i].dram_allocated.load(std::memory_order_relaxed);
            result[i].l1_allocated = device_stats_[i].l1_allocated.load(std::memory_order_relaxed);
            result[i].system_memory_allocated =
                device_stats_[i].system_memory_allocated.load(std::memory_order_relaxed);
            result[i].l1_small_allocated = device_stats_[i].l1_small_allocated.load(std::memory_order_relaxed);
            result[i].trace_allocated = device_stats_[i].trace_allocated.load(std::memory_order_relaxed);
            result[i].num_buffers = device_stats_[i].num_buffers.load(std::memory_order_relaxed);
            result[i].total_allocs = device_stats_[i].total_allocs.load(std::memory_order_relaxed);
            result[i].total_frees = device_stats_[i].total_frees.load(std::memory_order_relaxed);
        }
        return result;
    }

    /**
     * @brief Get count of active (unfreed) buffers on a device
     *
     * This requires a lock and is slightly more expensive than query_device.
     *
     * @param device_id Device ID (0-7)
     * @return Number of active buffers
     */
    size_t get_active_buffer_count(int device_id) const;

    /**
     * @brief Get list of active buffers on a device (for leak detection)
     *
     * @param device_id Device ID (0-7)
     * @return Vector of active buffer info
     */
    std::vector<BufferInfo> get_active_buffers(int device_id) const;

    /**
     * @brief Check if Tracy profiler is enabled at compile time
     */
    static constexpr bool is_tracy_enabled() {
#ifdef TRACY_ENABLE
        return true;
#else
        return false;
#endif
    }

    /**
     * @brief Reset all statistics (for testing)
     */
    void reset();
};

}  // namespace tt::tt_metal
