// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <mutex>

#include "impl/memory_tracking/buffer_allocation_observer.hpp"

namespace tt::tt_metal {

// Forward declarations
class Device;

// Records buffer allocations/deallocations into shared memory (SHM) for real-time monitoring
// by external tools (e.g. tt-smi-ui).
//
// A BufferAllocationObserver rather than an IGraphProcessor: the two notifications it wants are
// emitted from impl/buffers/buffer.cpp and consumed here in impl, so there is no reason for any
// of it to reach the Metalium public API.
class ShmTrackingProcessor : public BufferAllocationObserver {
public:
    // verbose: process-wide TT_METAL_SHM_VERBOSE flag, captured at construction time from
    // the owning Device's MetalContext rtoptions so the processor does not need to walk
    // MetalContext slots later. The flag is process-wide, so capturing once is correct.
    explicit ShmTrackingProcessor(bool verbose);
    ~ShmTrackingProcessor() override = default;

    // CB tracking is not here: it is recorded at dispatch by
    // Device::record_dispatched_program_cbs() and published through update_from_allocator().
    void track_allocate(const Buffer* buffer) override;
    void track_deallocate(Buffer* buffer) override;

private:
    // Global mutex to serialize all buffer tracking calls
    // Prevents race conditions where concurrent allocations/deallocations
    // at the same address send out-of-order updates to the SHM tracking
    std::mutex tracking_mutex_;

    // Verbose logging flag (set from TT_METAL_SHM_VERBOSE env var)
    bool verbose_enabled_;
};

}  // namespace tt::tt_metal
