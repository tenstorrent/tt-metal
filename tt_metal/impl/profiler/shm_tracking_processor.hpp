// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/graph_tracking.hpp>
#include <mutex>

namespace tt::tt_metal {

// Forward declarations
class Device;

// Processor that tracks buffer allocations/deallocations to shared memory (SHM)
// for real-time monitoring by external tools (e.g. tt-smi-ui)
class ShmTrackingProcessor : public IGraphProcessor {
public:
    ShmTrackingProcessor();
    ~ShmTrackingProcessor() override = default;

    void track_allocate(const Buffer* buffer) override;
    void track_deallocate(Buffer* buffer) override;

    // Note: CB tracking is handled separately via update_from_allocator() in device code
    // These are no-ops for SHM tracking
    void track_allocate_cb(
        const CoreRangeSet& /*core_range_set*/,
        uint64_t /*addr*/,
        uint64_t /*size*/,
        bool /*is_globally_allocated*/,
        const IDevice* /*device*/) override {}

    void track_deallocate_cb(const IDevice* /*device*/) override {}

private:
    // Global mutex to serialize all buffer tracking calls
    // Prevents race conditions where concurrent allocations/deallocations
    // at the same address send out-of-order updates to the SHM tracking
    std::mutex tracking_mutex_;

    // Verbose logging flag (set from TT_METAL_SHM_VERBOSE env var)
    bool verbose_enabled_;
};

}  // namespace tt::tt_metal
