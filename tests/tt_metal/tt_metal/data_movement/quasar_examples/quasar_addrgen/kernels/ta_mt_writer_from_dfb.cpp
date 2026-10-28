// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Multi-threaded DFB -> TensorAccessor writer, the consumer side of ta_mt_reader_to_dfb.cpp (whose DFB holds the
// tensor's pages in page-id order), or of its shards mode.
//
// Named CTAs:
//   num_pages:     tensor pages (interleaved accessors don't know their volume)
//   shards:        1 = thread t writes every page of its strided_shard_pages() (shards t, t+T, ...) in shard order,
//                  the order the reader's thread t filled its entries in. Needs as many consumer threads as producer
//                  threads and a STRIDED consumer, so consumer t gets exactly producer t's entries. Sharded only.
//   all_consumer:  0 = STRIDED consumer pattern: thread t of T gets entries t, t+T, ... and writes its strided_pages();
//                  1 = ALL consumer pattern: every thread gets every entry and writes all pages() (the same data to
//                  the same pages, once per thread)
//   implicit_sync: 1 = DFB implicit sync (async_write<TXN_ID>), 0 = explicit wait_front / async_write / barrier /
//                  pop_front. An ALL consumer must sync explicitly.
// Named RTAs:
//   report_addr: per thread (at + thread * 64 bytes), 4 words {hw, pushes, transfers, done marker} (hw and pushes in
//                TT_TA_ADDRGEN_STATS builds; see internal/tensor/generated_noc_addr.h)

#include <type_traits>

#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc.h"
#include "api/kernel_thread_globals.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

namespace {
constexpr uint32_t kReportStride = 64;
constexpr uint32_t kDoneMarker = 0x600DD00Du;
}  // namespace

void kernel_main() {
    constexpr uint32_t num_pages = get_arg(args::num_pages);
    constexpr uint32_t all_consumer = get_arg(args::all_consumer);
    constexpr uint32_t implicit_sync = get_arg(args::implicit_sync);
    constexpr uint32_t shards = get_arg(args::shards);
    static_assert(!(all_consumer && implicit_sync), "an ALL DM consumer must sync explicitly");
    static_assert(!(all_consumer && shards), "shards mode needs a STRIDED consumer");
    const uint32_t report_addr = get_arg(args::report_addr);

    Noc noc;
    DataflowBuffer dfb(dfb::in);
    const auto dst = TensorAccessor(tensor::dst);
    const uint32_t entry_size = dfb.get_entry_size();

    uint32_t transfers = 0;
    auto run = [&](const auto& ta) {
        using TA = std::decay_t<decltype(ta)>;
        auto my_pages = [&] {
            if constexpr (TA::DSpec::is_interleaved) {
                return all_consumer ? ta.pages(0, num_pages) : ta.strided_pages(num_pages);
            } else {
                return all_consumer ? ta.pages() : ta.strided_pages();
            }
        };
        auto write = [&](const auto& page) {
            using Traits = noc_traits_t<std::decay_t<decltype(page)>>;
            if constexpr (implicit_sync) {
                noc.async_write<NocOptions::TXN_ID>(dfb, page, {}, typename Traits::dst_args_type{});
            } else {
                dfb.wait_front(1);
                noc.async_write(dfb, page, entry_size, {}, typename Traits::dst_args_type{});
                noc.async_write_barrier();
                dfb.pop_front(1);
            }
            ++transfers;
        };
        if constexpr (shards) {
            static_assert(!TA::DSpec::is_interleaved, "strided_shard_pages() needs a sharded tensor");
            for (const auto& shard : ta.strided_shard_pages()) {
                for (const auto& page : shard) {
                    write(page);
                }
            }
        } else {
            for (const auto& page : my_pages()) {
                write(page);
            }
        }
    };
    run(dst);
    dfb.finish();
    dfb.write_barrier(noc);

    // Uncached: the host reads this back, and the DM data cache would otherwise hold the stores.
    volatile tt_l1_ptr uint32_t* report = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
        report_addr + (get_my_thread_id() * kReportStride) + MEM_L1_UNCACHED_BASE);
#if defined(TT_TA_ADDRGEN_STATS)
    report[0] = tensor_accessor::detail::transfer_stats.hw;
    report[1] = tensor_accessor::detail::transfer_stats.pushes;
#else
    report[0] = 0;
    report[1] = 0;
#endif
    report[2] = transfers;
    report[3] = kDoneMarker;
}
