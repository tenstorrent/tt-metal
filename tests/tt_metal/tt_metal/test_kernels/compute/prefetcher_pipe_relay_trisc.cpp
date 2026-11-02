// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// TRISC consumer of a PrefetcherPipe relay DFB.
//
// Bindings:
//   dfb::relay             — CONSUMER accessor of the relay DFB. Because the DFB relays a
//                            PrefetcherPipe, the generated token is a RelayDFBBindingToken that
//                            carries the pipe slot, so constructing the DataflowBuffer snaps the
//                            borrowed local iface to the durable fifo_ptr checkpoint (O(1)
//                            launch-msg slot lookup, no JIT-emitted trisck align).
// Args (named CTAs):
//   args::entries_this_thread - relay entries each consumer thread pops
//                               STRIDED: total_entries / num_consumers
//                               ALL:     total_entries (each Neo consumes the full stream)
//   args::batch_size          - entries per wait_front / pop_front
//   args::delay_iterations    - busy-loop iterations per batch (backpressure tests)
// Args (named RTAs):
//   args::result_addr         - per-thread [entries_consumed, checksum] at
//                               result_addr + 2 * get_my_thread_id() words

#include "api/compute/compute_kernel_api.h"
#include "api/compute/common.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/kernel_thread_globals.h"
#include "experimental/kernel_args.h"
#ifdef ARCH_QUASAR
#include "api/compute/tile_move_copy.h"
#include "internal/tt-2xx/quasar/dev_mem_map.h"
#endif

void kernel_main() {
#ifdef UCK_CHLKC_UNPACK
    constexpr uint32_t entries_this_thread = get_arg(args::entries_this_thread);
    constexpr uint16_t batch_size = get_arg(args::batch_size);
    constexpr uint32_t delay_iterations = get_arg(args::delay_iterations);

    DataflowBuffer relay(dfb::relay);
    volatile tt_l1_ptr uint32_t* result = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_arg(args::result_addr));
    const uint32_t tid = get_my_thread_id();

    uint32_t checksum = 0;
    for (uint32_t offset = 0; offset < entries_this_thread; offset += batch_size) {
        relay.wait_front(batch_size);
#ifdef ARCH_QUASAR
        // Quasar: NOC->TL1 fills are not L2-coherent; hand-read via the uncached alias.
        // (#55938: wait_front blocks until WAIT_TILES resolves — no tensix_sync needed.)
        const uint32_t read_ptr = (relay.get_read_ptr() << cb_addr_shift) + MEM_L1_UNCACHED_BASE;
#else
        const uint32_t read_ptr = relay.get_read_ptr() << cb_addr_shift;
#endif
        // STRIDED batches are stride_size apart (not contiguous entry_size).
        const uint32_t stride_size = relay.get_stride_size();
        for (uint32_t i = 0; i < batch_size; ++i) {
            checksum += *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(read_ptr + i * stride_size);
        }
        for (volatile uint32_t delay = 0; delay < delay_iterations; ++delay) {
        }
#ifdef ARCH_QUASAR
        // TEN-4746: hand-read L1 (no UNPACR) since wait_front; dummy unpack orders pop.
        ckernel::dummy_unpack(static_cast<uint32_t>(dfb::relay));
#endif
        relay.pop_front(batch_size);
    }
    result[tid * 2 + 0] = entries_this_thread;
    result[tid * 2 + 1] = checksum;
#endif
}
