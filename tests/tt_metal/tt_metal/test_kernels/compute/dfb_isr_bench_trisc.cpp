// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// dfb_isr_timing_bench: TRISC side. No timing instrumentation (measured from waveforms).
//
// Binds 1..12 DFBs ("d00".."d11"). The DM side of each DFB uses implicit sync, so DM0's ISR updates
// this Neo's tile counters: it posts credits to a consumer here and acks space to a producer here.
//
// role 0  CONSUMER (unpacker): wait_front / dummy_unpack / pop_front, as fast as possible.
// role 1  PRODUCER (packer):   reserve_back / dummy_pack / push_back, as fast as possible.
//
// No data is unpacked or packed: dummy_unpack / dummy_pack only supply the LLK op TEN-4746 requires
// between the two DFB calls, so the run isolates the synchronisation cost.

#include "api/dataflow/dataflow_buffer.h"
#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "experimental/kernel_args.h"
#include "../dataflow/dfb_isr_bench_common.h"

void kernel_main() {
    constexpr uint32_t role = get_arg(args::role);
    constexpr uint32_t entries_per_thread = get_arg(args::entries_per_thread);  // per DFB

    constexpr std::size_t num_dfbs = isr_bench::count_bound_dfbs();
    static_assert(num_dfbs >= 1, "bind at least d00");

    auto dfbs = isr_bench::make_dfbs(std::make_index_sequence<num_dfbs>{});
    compute_kernel_hw_startup(isr_bench::token<0>(), isr_bench::token<0>());

    if constexpr (role == 0) {
        const uint32_t share = std::get<0>(dfbs).get_consumer_share();
        for (uint32_t e = 0; e < entries_per_thread; e += share) {
            isr_bench::for_each_dfb(dfbs, [&](auto& d, uint32_t) {
                d.wait_front(share);
                dummy_unpack(d.get_id());
                d.pop_front(share);
            });
        }
    } else {
        const uint32_t share = std::get<0>(dfbs).get_producer_share();
        for (uint32_t e = 0; e < entries_per_thread; e += share) {
            isr_bench::for_each_dfb(dfbs, [&](auto& d, uint32_t) {
                d.reserve_back(share);
                ckernel::dummy_pack(d.get_id());
                d.push_back(share);
            });
        }
    }
}
