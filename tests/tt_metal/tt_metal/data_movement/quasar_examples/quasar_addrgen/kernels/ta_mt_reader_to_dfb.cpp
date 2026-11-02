// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Multi-threaded TensorAccessor -> DFB reader. Each producer thread t of T reads its strided_pages() (pages t, t+T,
// ...) of tensor::src into dfb::out, one page per entry. With a STRIDED producer pattern, thread t fills DFB entries
// t, t+T, ..., so the DFB holds the tensor's pages in page-id order. With shards = 1, thread t instead reads every page
// of its strided_shard_pages() (shards t, t+T, ...) in shard order.
//
// Named CTAs:
//   num_pages:     tensor pages (interleaved accessors don't know their volume)
//   shards:        1 = walk strided_shard_pages() (sharded only), 0 = strided_pages()
//   implicit_sync: 1 = DFB implicit sync (async_read<TXN_ID>), 0 = explicit reserve_back / async_read / barrier /
//                  push_back
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
    constexpr uint32_t implicit_sync = get_arg(args::implicit_sync);
    constexpr uint32_t shards = get_arg(args::shards);
    const uint32_t report_addr = get_arg(args::report_addr);

    Noc noc;
    DataflowBuffer dfb(dfb::out);
    const auto src = TensorAccessor(tensor::src);
    const uint32_t entry_size = dfb.get_entry_size();

    uint32_t transfers = 0;
    auto run = [&](const auto& ta) {
        using TA = std::decay_t<decltype(ta)>;
        auto my_pages = [&] {
            if constexpr (TA::DSpec::is_interleaved) {
                return ta.strided_pages(num_pages);
            } else {
                return ta.strided_pages();
            }
        };
        auto read = [&](const auto& page) {
            using Traits = noc_traits_t<std::decay_t<decltype(page)>>;
            if constexpr (implicit_sync) {
                noc.async_read<NocOptions::TXN_ID>(page, dfb, typename Traits::src_args_type{}, {});
            } else {
                dfb.reserve_back(1);
                noc.async_read(page, dfb, entry_size, typename Traits::src_args_type{}, {});
                noc.async_read_barrier();
                dfb.push_back(1);
            }
            ++transfers;
        };
        if constexpr (shards) {
            static_assert(!TA::DSpec::is_interleaved, "strided_shard_pages() needs a sharded tensor");
            for (const auto& shard : ta.strided_shard_pages()) {
                for (const auto& page : shard) {
                    read(page);
                }
            }
        } else {
            for (const auto& page : my_pages()) {
                read(page);
            }
        }
    };
    run(src);
    dfb.finish();

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
