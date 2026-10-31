// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "internal/firmware_common.h"

namespace experimental {

// Signal the Tensor prefetcher running on this kernel's device: atomically increments by one the 32-bit
// counter at `signal_addr` that each DRAM bank keeps for the signal. `signal_addr` is the value the host
// gets from GetTensorPrefetcherSignalAddress for the signal to raise, which is the same on every device
// and every bank. One call is one signal, releasing one wait queued on that signal with
// QueueTensorPrefetcherWaitForSignal.
//
// The increments go out on `noc`, by default this kernel's own NoC; either NoC reaches the counters. A
// dedicated-NoC kernel can only use its own; a dynamic-NoC kernel may pick either.
//
// Like noc_semaphore_inc, the increments are in flight on return: noc_async_atomic_barrier(noc) waits for
// them to land. They are not ordered after this kernel's earlier writes, so a kernel that signals that
// it has written data the prefetcher will read must wait for those writes (noc_async_write_barrier)
// first. Tensix data-movement kernels only, on a device whose Tensor prefetcher is running.
template <uint8_t noc = noc_index>
FORCE_INLINE void tensor_prefetcher_signal(uint32_t signal_addr) {
    static_assert(noc < NUM_NOCS, "tensor_prefetcher_signal: noc must be 0 or 1");
    static_assert(
        noc_mode == DM_DYNAMIC_NOC || noc == noc_index,
        "tensor_prefetcher_signal: a dedicated-NoC kernel must signal on its own NoC, whose command buffers it owns");
    WAYPOINT("TPSW");
    for (uint32_t bank = 0; bank < NUM_DRAM_BANKS; ++bank) {
        // noc_semaphore_inc without its watcher NoC check, as for the PrefetcherPipe credit atomics that reach
        // a DRAM sender: the watcher knows DRAM cores only as GDDR endpoints and does not list the bank's free
        // subchannel that holds the counter.
        noc_fast_atomic_increment<noc_mode>(
            noc,
            write_at_cmd_buf,
            noc_address_backend::packed_worker_address(tensor_prefetcher_signal_noc_xy[bank], signal_addr),
            NOC_UNICAST_WRITE_VC,
            1 /*incr*/,
            31 /*wrap*/,
            false /*linked*/,
            false /*posted*/,
            MEM_NOC_ATOMIC_RET_VAL_ADDR);
    }
    WAYPOINT("TPSD");
}

}  // namespace experimental
