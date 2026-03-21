// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "device/device_impl.hpp"
#include "profiler/memory_stats_shm.hpp"
#include <tt-logger/tt-logger.hpp>
#include <chrono>
#include <mutex>
#include <unordered_map>

namespace tt::tt_metal {

// Implementation of SharedMemoryStatsProvider::update_from_allocator()
// Query ONLY locally-allocated CBs (device->get_total_cb_allocated() only counts local CBs)
// Globally-allocated CBs create L1 Buffers and are already tracked in L1 column
void SharedMemoryStatsProvider::update_from_allocator(const Device* device, pid_t pid) {
    if (!region_ || !device) {
        return;
    }

    // Rate limiting: max 10 updates/sec per device to reduce overhead
    static std::unordered_map<uint32_t, std::chrono::steady_clock::time_point> last_updates;
    static std::mutex rate_limit_mutex;

    {
        std::lock_guard<std::mutex> lock(rate_limit_mutex);
        auto now = std::chrono::steady_clock::now();
        auto& last = last_updates[device->id()];

        if (now - last < std::chrono::milliseconds(100)) {
            return;  // Skip update - too soon since last update
        }
        last = now;
    }

    try {
        // Query actual LOCALLY-allocated CB usage (globally-allocated CBs are in L1 already)
        uint64_t cb_allocated = device->get_total_cb_allocated();

        // Update device-wide CB total (query-based, accurate, no accumulation)
        region_->total_cb_allocated.store(cb_allocated, std::memory_order_relaxed);

        // Update timestamp
        auto now = std::chrono::system_clock::now();
        region_->last_update_timestamp =
            std::chrono::duration_cast<std::chrono::nanoseconds>(now.time_since_epoch()).count();

        // Update per-chip CB stats for this device
        uint32_t chip_id = device->id();
        for (size_t i = 0; i < MAX_CHIPS_PER_DEVICE; i++) {
            if (region_->chip_stats[i].chip_id == chip_id || region_->chip_stats[i].chip_id == 0) {
                region_->chip_stats[i].chip_id = chip_id;
                region_->chip_stats[i].cb_allocated.store(cb_allocated, std::memory_order_relaxed);
                break;
            }
        }

        // Update per-process CB stats for this PID
        for (size_t i = 0; i < MAX_PROCESSES; i++) {
            if (region_->processes[i].pid == pid) {
                // Update only locally-allocated CBs (query-based, accurate even with caching)
                region_->processes[i].cb_allocated = cb_allocated;
                region_->processes[i].last_update_timestamp =
                    std::chrono::duration_cast<std::chrono::nanoseconds>(now.time_since_epoch()).count();
                break;
            }
        }
    } catch (const std::exception& e) {
        log_warning(LogMetal, "Failed to query locally-allocated CB stats: {}", e.what());
    }
}

}  // namespace tt::tt_metal
