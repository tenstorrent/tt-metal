// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "impl/device/device_impl.hpp"
#include "impl/memory_tracking/memory_stats_shm.hpp"
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

    // No rate limiting. It existed to throttle get_total_cb_allocated(), which used to walk
    // every live program; that walk is gone and this is now an atomic load plus a scan of the
    // 64 process slots. Throttling here was also incorrect: a dropped update is never retried,
    // so a workload that dispatched and then went idle left a stale figure in shared memory
    // forever -- exactly what an external monitor would display.
    try {
        // Query actual LOCALLY-allocated CB usage (globally-allocated CBs are in L1 already)
        uint64_t cb_allocated = device->get_total_cb_allocated();

        // Record OUR contribution in our own process slot first. This value is
        // per-process by construction: get_total_cb_allocated() only sees the programs
        // registered by this process.
        for (auto& slot : region_->processes) {
            if (slot.pid.load(std::memory_order_relaxed) == pid) {
                slot.cb_allocated.store(cb_allocated, std::memory_order_relaxed);
                slot.last_update_timestamp.store(current_timestamp_ns(), std::memory_order_relaxed);
                break;
            }
        }

        // Then republish the device-wide totals as the SUM over live process slots.
        // This previously store()d our own value straight into total_cb_allocated, which
        // made the device-wide figure last-writer-wins instead of a sum: with two
        // processes on one device it reported one process's CB usage, not the total.
        // recompute_aggregates() also reaps nothing but re-derives reference_count, so
        // the rate-limited cadence here doubles as a periodic self-correction.
        recompute_aggregates();

        // Per-chip CB stats for this device. Use the CAS-based claim helper rather than
        // grabbing the first matching-or-unused slot with a plain store, which raced
        // find_or_create_chip_entry() and could let two processes claim the same slot.
        const uint32_t chip_id = static_cast<uint32_t>(device->id());
        if (auto* chip_entry = find_or_create_chip_entry(chip_id)) {
            // Local (gateway) chip: its CB total is the device-wide total we just derived.
            chip_entry->cb_allocated.store(
                region_->total_cb_allocated.load(std::memory_order_relaxed), std::memory_order_relaxed);
        }
    } catch (const std::exception& e) {
        log_warning(LogMetal, "Failed to query locally-allocated CB stats: {}", e.what());
    }
}

}  // namespace tt::tt_metal
