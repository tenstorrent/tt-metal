// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// dfb_isr_timing_bench: DM producer side. No timing instrumentation (measured from waveforms).
//
// Binds 1..12 DFBs as producer ("d00".."d11") and the DRAM tensor "src", whose page size is one NoC
// transaction (one entry: producers are STRIDED).
//
// mode 0  READ_IMPLICIT  noc.async_read<TXN_ID>(src -> DFB): the ISR posts the credits.
// mode 1  FILL_EXPLICIT  reserve_back / push_back with no data movement (fast producer for write-side runs).
// mode 2  RAW_READ       plain NoC reads into d00's L1 ring with no tile counters and no ISR: the ring is
//                        split into raw_sections trid-tagged sections, and a section waits for its own
//                        previous batch before reuse (the same in-flight limit implicit sync imposes).
//
// Each call round-robins over the bound DFBs, so one thread can feed several DFBs.
// Pages: DFB j of this kernel on core c reads pages [(c*dfbs_per_core + dfb_base + j) * pages_per_dfb, +pages_per_dfb);
// thread t takes calls t, t+T, ... of that range (STRIDED interleave).

#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"
#include "api/core_local_mem.h"
#include "api/kernel_thread_globals.h"
#include "experimental/kernel_args.h"
#include "dfb_isr_bench_common.h"

void kernel_main() {
    constexpr uint32_t mode = get_arg(args::mode);
    constexpr uint32_t entries_per_thread = get_arg(args::entries_per_thread);  // per DFB
    constexpr uint32_t pages_per_dfb = get_arg(args::pages_per_dfb);
    constexpr uint32_t dfb_base = get_arg(args::dfb_base);
    constexpr uint32_t dfbs_per_core = get_arg(args::dfbs_per_core);
    constexpr uint32_t raw_sections = get_arg(args::raw_sections);
    constexpr uint32_t raw_ops_per_section = get_arg(args::raw_ops_per_section);
    constexpr uint32_t raw_txn_bytes = get_arg(args::raw_txn_bytes);
    const uint32_t core_idx = get_arg(args::core_idx);

    constexpr std::size_t num_dfbs = isr_bench::count_bound_dfbs();
    static_assert(num_dfbs >= 1, "bind at least d00");

    const uint32_t t = get_my_thread_id();
    const uint32_t num_threads = get_num_threads();
    const uint32_t first_page = (core_idx * dfbs_per_core + dfb_base) * pages_per_dfb;

    Noc noc;
    const auto src = TensorAccessor(tensor::src);
    auto dfbs = isr_bench::make_dfbs(std::make_index_sequence<num_dfbs>{});

    if constexpr (mode == 0) {
        // STRIDED producer: one entry, one NoC transaction, per call.
        for (uint32_t c = 0; c < entries_per_thread; c++) {
            const uint32_t call_idx = c * num_threads + t;
            isr_bench::for_each_dfb(dfbs, [&](auto& d, uint32_t j) {
                noc.async_read<NocOptions::TXN_ID>(src, d, {.page_id = first_page + j * pages_per_dfb + call_idx}, {});
            });
        }
    } else if constexpr (mode == 1) {
        const uint32_t share = std::get<0>(dfbs).get_producer_share();
        const uint32_t ops = entries_per_thread / share;
        for (uint32_t o = 0; o < ops; o++) {
            isr_bench::for_each_dfb(dfbs, [&](auto& d, uint32_t) {
                d.reserve_back(share);
                d.push_back(share);
            });
        }
    } else {
        // RAW_READ: d00 only lends its L1 ring. get_write_ptr() is the uncached alias on Quasar DM.
        auto& d = std::get<0>(dfbs);
        const uint32_t ring_base = d.get_write_ptr() - MEM_L1_UNCACHED_BASE;
        const uint32_t slice_bytes = d.get_total_size_bytes() / num_threads;
        const uint32_t my_base = ring_base + t * slice_bytes;
        const uint32_t ops = entries_per_thread;
        uint32_t slice_off = 0;
        for (uint32_t o = 0; o < ops; o++) {
            const uint32_t batch = o / raw_ops_per_section;
            const uint32_t trid = 1 + batch % raw_sections;
            if (o % raw_ops_per_section == 0 && batch >= raw_sections) {
                noc.async_read_barrier<NocOptions::TXN_ID>({.trid = trid});  // section reuse: previous batch landed
            }
            noc.async_read<NocOptions::TXN_ID>(
                src,
                CoreLocalMem<uint32_t>(my_base + slice_off),
                raw_txn_bytes,
                {.page_id = first_page + o * num_threads + t},
                {},
                {.trid = trid});
            slice_off += raw_txn_bytes;
            if (slice_off + raw_txn_bytes > slice_bytes) {
                slice_off = 0;
            }
        }
        for (uint32_t s = 0; s < raw_sections; s++) {
            noc.async_read_barrier<NocOptions::TXN_ID>({.trid = 1 + s});
        }
    }
}
