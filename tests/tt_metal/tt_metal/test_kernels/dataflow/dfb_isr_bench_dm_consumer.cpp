// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// dfb_isr_timing_bench: DM consumer side. No timing instrumentation (measured from waveforms).
//
// Binds 1..12 DFBs as consumer ("d00".."d11") and the DRAM tensor "dst", whose page size is one NoC
// transaction (one entry: consumers are STRIDED or ALL).
//
// mode 0  WRITE_IMPLICIT  noc.async_write<TXN_ID>(DFB -> dst): the ISR acks the freed space.
// mode 1  SINK_EXPLICIT   wait_front / pop_front with no data movement (fast consumer for read-side runs).
// mode 2  IDLE            binds the DFBs and does nothing (pairs with the producer's RAW_READ mode).
//
// Page layout mirrors dfb_isr_bench_dm_producer.cpp.

#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"
#include "api/kernel_thread_globals.h"
#include "experimental/kernel_args.h"
#include "dfb_isr_bench_common.h"

void kernel_main() {
    constexpr uint32_t mode = get_arg(args::mode);
    constexpr uint32_t entries_per_thread = get_arg(args::entries_per_thread);  // per DFB
    constexpr uint32_t pages_per_dfb = get_arg(args::pages_per_dfb);
    constexpr uint32_t dfb_base = get_arg(args::dfb_base);
    constexpr uint32_t dfbs_per_core = get_arg(args::dfbs_per_core);
    const uint32_t core_idx = get_arg(args::core_idx);

    constexpr std::size_t num_dfbs = isr_bench::count_bound_dfbs();
    static_assert(num_dfbs >= 1, "bind at least d00");

    const uint32_t t = get_my_thread_id();
    const uint32_t num_threads = get_num_threads();
    const uint32_t first_page = (core_idx * dfbs_per_core + dfb_base) * pages_per_dfb;

    Noc noc;
    const auto dst = TensorAccessor(tensor::dst);
    auto dfbs = isr_bench::make_dfbs(std::make_index_sequence<num_dfbs>{});

    if constexpr (mode == 0) {
        // One entry, one NoC transaction, per call.
        for (uint32_t c = 0; c < entries_per_thread; c++) {
            const uint32_t call_idx = c * num_threads + t;
            isr_bench::for_each_dfb(dfbs, [&](auto& d, uint32_t j) {
                noc.async_write<NocOptions::TXN_ID>(d, dst, {}, {.page_id = first_page + j * pages_per_dfb + call_idx});
            });
        }
    } else if constexpr (mode == 1) {
        const uint32_t share = std::get<0>(dfbs).get_consumer_share();
        const uint32_t ops = entries_per_thread / share;
        for (uint32_t o = 0; o < ops; o++) {
            isr_bench::for_each_dfb(dfbs, [&](auto& d, uint32_t) {
                d.wait_front(share);
                d.pop_front(share);
            });
        }
    }
    // mode 2: nothing. The DataflowBuffer destructors drain (posted == acked) and wait for outbound writes.
}
