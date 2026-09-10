// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <mutex>

namespace tt::tt_metal {

// Forward declarations
class Device;

class Buffer;

// Records buffer allocations/deallocations into shared memory (SHM) for real-time monitoring
// by external tools (e.g. tt-smi-ui).
//
// Reached through the free functions below rather than by implementing IGraphProcessor. The two
// notifications it wants are emitted from impl/buffers/buffer.cpp and consumed here in impl, so
// none of it belongs on the Metalium public API surface.
class ShmTrackingProcessor {
public:
    // verbose: process-wide TT_METAL_SHM_VERBOSE flag, captured at construction time from
    // the owning Device's MetalContext rtoptions so the processor does not need to walk
    // MetalContext slots later. The flag is process-wide, so capturing once is correct.
    explicit ShmTrackingProcessor(bool verbose);

    // CB tracking is not here: it is recorded at dispatch by
    // Device::record_dispatched_program_cbs() and published through update_from_allocator().
    void track_allocate(const Buffer* buffer);
    void track_deallocate(Buffer* buffer);

private:
    // Global mutex to serialize all buffer tracking calls
    // Prevents race conditions where concurrent allocations/deallocations
    // at the same address send out-of-order updates to the SHM tracking
    std::mutex tracking_mutex_;

    // Verbose logging flag (set from TT_METAL_SHM_VERBOSE env var)
    bool verbose_enabled_;
};

// Turn on SHM buffer tracking for this process. Called from Device::initialize() by every device
// that has a stats provider; the tracker is constructed on the first such call and shared, since
// the verbose flag and the SHM regions are both process-wide.
void enable_shm_buffer_tracking(bool verbose);

// Notified from Buffer::allocate_impl() and Buffer::deallocate(). One atomic load and a return
// when tracking is off, which is the default.
void record_buffer_allocation(const Buffer* buffer);
void record_buffer_deallocation(Buffer* buffer);

}  // namespace tt::tt_metal
