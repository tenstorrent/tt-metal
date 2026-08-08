// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "impl/device/device_impl.hpp"
#include "impl/memory_tracking/memory_stats_shm.hpp"
#include <tt-logger/tt-logger.hpp>
#include <cassert>
#include <chrono>
#include <mutex>
#include <optional>
#include <unordered_map>

namespace tt::tt_metal {

// Implementation of SharedMemoryStatsProvider::update_from_allocator()
// Query ONLY locally-allocated CBs (device->get_total_cb_allocated() only counts local CBs)
// Globally-allocated CBs create L1 Buffers and are already tracked in L1 column
void SharedMemoryStatsProvider::update_from_allocator(const Device* device, pid_t pid) {
    std::optional<uint64_t> uncached;
    update_from_allocator(device, pid, uncached);
}

void SharedMemoryStatsProvider::update_from_allocator(
    const Device* device, pid_t pid, std::optional<uint64_t>& cached_cb_allocated) {
    if (!region_ || !device) {
        return;
    }

    // Rate limiting: max 10 updates/sec per device to reduce overhead
    static std::unordered_map<uint32_t, std::chrono::steady_clock::time_point> last_updates;
    static std::mutex rate_limit_mutex;

    {
        std::lock_guard<std::mutex> lock(rate_limit_mutex);
        auto now = std::chrono::steady_clock::now();
        auto& last = last_updates[static_cast<uint32_t>(device->id())];

        if (now - last < std::chrono::milliseconds(100)) {
            return;  // Skip update - too soon since last update
        }
        last = now;
    }

    try {
        // Query actual LOCALLY-allocated CB usage (globally-allocated CBs are in L1 already).
        // On a homogeneous mesh this value is identical across sub-devices, so a caller updating
        // every sub-device in a loop can compute it once and pass it via cached_cb_allocated.
        uint64_t cb_allocated;
        if (cached_cb_allocated.has_value()) {
            cb_allocated = *cached_cb_allocated;
            assert(
                cb_allocated == device->get_total_cb_allocated() &&
                "SHM CB-allocated differs across mesh sub-devices; a per-device CB layout must not "
                "share a cached value (see ProgramImpl::kCbL1LayoutIsDeviceIndependent)");
        } else {
            cb_allocated = device->get_total_cb_allocated();
            cached_cb_allocated = cb_allocated;
        }

        // Update device-wide CB total (query-based, accurate, no accumulation)
        region_->total_cb_allocated.store(cb_allocated, std::memory_order_relaxed);

        // Update timestamp
        region_->last_update_timestamp.store(current_timestamp_ns(), std::memory_order_relaxed);

        // Update per-chip CB stats for this device
        uint32_t chip_id = static_cast<uint32_t>(device->id());
        for (auto & chip_stat : region_->chip_stats) {
            uint32_t slot_id = chip_stat.chip_id.load(std::memory_order_relaxed);
            if (slot_id == chip_id || slot_id == CHIP_STATS_UNUSED) {
                chip_stat.chip_id.store(chip_id, std::memory_order_relaxed);
                chip_stat.cb_allocated.store(cb_allocated, std::memory_order_relaxed);
                break;
            }
        }

        // Update per-process CB stats for this PID
        for (auto& processe : region_->processes) {
            if (processe.pid == pid) {
                // Update only locally-allocated CBs (query-based, accurate even with caching)
                processe.cb_allocated = cb_allocated;
                processe.last_update_timestamp.store(current_timestamp_ns(), std::memory_order_relaxed);
                break;
            }
        }
    } catch (const std::exception& e) {
        log_warning(LogMetal, "Failed to query locally-allocated CB stats: {}", e.what());
    }
}

}  // namespace tt::tt_metal
