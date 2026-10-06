// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// DFB -> TensorAccessor writer for the TensorAccessor AddrGen pattern matrix.
// Writes entries of dfb::in to pages [start_page, start_page + num_pages) of tensor::dst.
//
// Named CTAs:
//   iter_mode:     0 = explicit page_id loop (async_write(dfb, ta, {.page_id})), 1 = pages() iterator (Page endpoint),
//                  2 = PageView(ta) with {.page_id}, 3 = AbstractTensorAccessorWrapper(ta) with {.page_id},
//                  4 = ShardView(ta): every page slot of every shard as {.shard_id, .offset_bytes} (sharded only),
//                  5 = shard_pages(shard) for every shard (sharded only), 6 = page ids even then odd (stride 2).
//                  4/5/6 don't go in page-id order, so the other side must use the same mode.
//   implicit_sync: 1 = DFB implicit sync (async_write<TXN_ID>, the default Quasar path),
//                  0 = explicit wait_front / async_write / barrier / pop_front
// Named RTAs:
//   start_page, num_pages,
//   report_addr: L1 address for 13 words {hw, sw_ineligible, sw_unsupported, seeks,
//                transfers issued, skips, restores, write seeks, write restores,
//                (word 9 unused), fallbacks, write fallbacks, pushes} -- how each transfer address was
//                produced (TT_TA_ADDRGEN_STATS builds only; see api/tensor/transfer_noc_addr.h)

#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    constexpr uint32_t iter_mode = get_arg(args::iter_mode);
    constexpr uint32_t implicit_sync = get_arg(args::implicit_sync);
    const uint32_t start_page = get_arg(args::start_page);
    const uint32_t num_pages = get_arg(args::num_pages);
    const uint32_t report_addr = get_arg(args::report_addr);

    Noc noc;
    DataflowBuffer dfb(dfb::in);
    const auto dst = TensorAccessor(tensor::dst);
    const uint32_t entry_size = dfb.get_entry_size();

    auto write_page = [&](const auto& endpoint, const auto& dst_args) {
        if constexpr (implicit_sync) {
            noc.async_write<NocOptions::TXN_ID>(dfb, endpoint, {}, dst_args);
        } else {
            dfb.wait_front(1);
            noc.async_write(dfb, endpoint, entry_size, {}, dst_args);
            noc.async_write_barrier();
            dfb.pop_front(1);
        }
    };

    uint32_t transfers = 0;
    // Generic lambda: the shard modes' branches only compile for a sharded accessor, and a discarded `if constexpr`
    // branch is only skipped inside a template.
    auto run = [&](const auto& ta) {
        using TA = std::decay_t<decltype(ta)>;
        auto xfer = [&](const auto& endpoint, const auto& args) {
            write_page(endpoint, args);
            ++transfers;
        };
        if constexpr (iter_mode == 0) {
            for (uint32_t page_id = start_page; page_id < start_page + num_pages; ++page_id) {
                xfer(ta, typename noc_traits_t<TA>::dst_args_type{.page_id = page_id});
            }
        } else if constexpr (iter_mode == 1) {
            for (const auto& page : ta.pages(start_page, start_page + num_pages)) {
                xfer(page, typename noc_traits_t<std::decay_t<decltype(page)>>::dst_args_type{});
            }
        } else if constexpr (iter_mode == 2) {
            const PageView view(ta);
            for (uint32_t page_id = start_page; page_id < start_page + num_pages; ++page_id) {
                xfer(view, typename noc_traits_t<PageView<TA>>::dst_args_type{.page_id = page_id});
            }
        } else if constexpr (iter_mode == 3) {
            const AbstractTensorAccessorWrapper wrapper(ta);
            for (uint32_t page_id = start_page; page_id < start_page + num_pages; ++page_id) {
                xfer(wrapper, typename noc_traits_t<AbstractTensorAccessorWrapper>::dst_args_type{.page_id = page_id});
            }
        } else if constexpr (iter_mode == 6) {
            // Every other page, then the ones in between: a steady stride of 2 (skipped in hardware), one jump back.
            for (uint32_t pass = 0; pass < 2; ++pass) {
                for (uint32_t page_id = start_page + pass; page_id < start_page + num_pages; page_id += 2) {
                    xfer(ta, typename noc_traits_t<TA>::dst_args_type{.page_id = page_id});
                }
            }
        } else if constexpr (iter_mode == 4) {
            // Every page slot of every shard (padding included), through the shard's base + offset.
            const ShardView view(ta);
            const uint32_t page_size = ta.get_aligned_page_size();
            for (uint32_t shard = 0; shard < ta.dspec().num_shards(); ++shard) {
                for (uint32_t p = 0; p < ta.dspec().shard_volume(); ++p) {
                    xfer(
                        view,
                        typename noc_traits_t<ShardView<TA>>::dst_args_type{
                            .shard_id = shard, .offset_bytes = p * page_size});
                }
            }
        } else {
            // The tensor's pages shard by shard (padding skipped by the iterator).
            for (uint32_t shard = 0; shard < ta.dspec().num_shards(); ++shard) {
                for (const auto& page : ta.shard_pages(shard)) {
                    xfer(page, typename noc_traits_t<std::decay_t<decltype(page)>>::dst_args_type{});
                }
            }
        }
    };
    run(dst);
    dfb.finish();
    dfb.write_barrier(noc);

#if defined(TT_TA_ADDRGEN_STATS)
    // Uncached: the host reads this back, and the DM data cache would otherwise hold the stores.
    volatile tt_l1_ptr uint32_t* report =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(report_addr + MEM_L1_UNCACHED_BASE);
    report[0] = tensor_accessor::detail::transfer_stats.hw;
    report[1] = tensor_accessor::detail::transfer_stats.sw_ineligible;
    report[2] = tensor_accessor::detail::transfer_stats.sw_unsupported;
    report[3] = tensor_accessor::detail::transfer_stats.seeks;
    report[4] = transfers;
    report[5] = tensor_accessor::detail::transfer_stats.skips;
    report[6] = tensor_accessor::detail::transfer_stats.restores;
    report[7] = tensor_accessor::detail::transfer_stats.write_seeks;
    report[8] = tensor_accessor::detail::transfer_stats.write_restores;
    report[10] = tensor_accessor::detail::transfer_stats.fallbacks;
    report[11] = tensor_accessor::detail::transfer_stats.write_fallbacks;
    report[12] = tensor_accessor::detail::transfer_stats.pushes;
#else
    (void)report_addr;
    (void)transfers;
#endif
}
