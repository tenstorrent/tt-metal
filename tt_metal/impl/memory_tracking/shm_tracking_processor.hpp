// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/graph_tracking.hpp>

namespace tt::tt_metal {

// Forward declarations
class Device;
class Buffer;

// Direct, thread-agnostic SHM recording entry points, invoked from
// Buffer::allocate_impl() / Buffer::deallocate_impl().
//
// These deliberately do NOT route through GraphTracker. Since #44668 made
// GraphTracker::processors thread_local, a processor pushed once on the
// device-init thread is invisible to allocations dispatched on any other
// thread, which silently disabled SHM DRAM/L1/trace tracking. Calling the
// SHM provider directly here records allocations regardless of which thread
// performs them, while leaving the thread_local capture semantics (used by
// transient graph-capture processors) untouched.
void shm_record_buffer_allocation(const Buffer* buffer);
void shm_record_buffer_deallocation(Buffer* buffer);

// Processor that tracks buffer allocations/deallocations to shared memory (SHM)
// for real-time monitoring by external tools (e.g. tt-smi-ui)
class ShmTrackingProcessor : public IGraphProcessor {
public:
    // verbose: process-wide TT_METAL_SHM_VERBOSE flag, captured at construction time from
    // the owning Device's MetalContext rtoptions so the processor does not need to walk
    // MetalContext slots later. The flag is process-wide, so capturing once is correct.
    explicit ShmTrackingProcessor(bool verbose);
    ~ShmTrackingProcessor() override = default;

    // ShmTrackingProcessor is a permanent background processor; it must not
    // cause is_graph_capture_active() to return true when no capture is in progress.
    bool is_capture_processor() const override { return false; }

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
};

}  // namespace tt::tt_metal
