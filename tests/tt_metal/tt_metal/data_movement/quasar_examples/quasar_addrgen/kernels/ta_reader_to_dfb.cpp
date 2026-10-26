// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// TensorAccessor -> DFB reader for the TensorAccessor AddrGen pattern matrix.
// Reads pages [start_page, start_page + num_pages) of tensor::src into dfb::out, one page per entry.
//
// Named CTAs:
//   iter_mode:     0 = explicit page_id loop (async_read(ta, dfb, {.page_id})), 1 = pages() iterator (Page endpoint),
//                  2 = PageView(ta) with {.page_id}, 3 = AbstractTensorAccessorWrapper(ta) with {.page_id},
//                  4 = ShardView(ta): every page slot of every shard as {.shard_id, .offset_bytes} (sharded only),
//                  5 = shard_pages(shard) for every shard (sharded only), 6 = page ids even then odd (stride 2).
//                  4/5/6 don't go in page-id order, so the other side must use the same mode.
//   implicit_sync: 1 = DFB implicit sync (async_read<TXN_ID>, the default Quasar path),
//                  0 = explicit reserve_back / async_read / barrier / push_back
// Named RTAs:
//   start_page, num_pages,
//   report_addr: L1 address for 4 words {hw, pushes, transfers issued, unused stack bytes} (TT_TA_ADDRGEN_STATS
//                builds only; see internal/tensor/generated_noc_addr.h)

#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

#if defined(TT_TA_ADDRGEN_STATS) && defined(ARCH_QUASAR)
// Stack high-water mark, for the sequencer-state budget (DM cores share 8 KB between thread-local storage and stack):
// paint the free stack with a pattern at entry, count the untouched words at exit. Same scheme as
// internal/debug/stack_usage.h, which only exists with the watcher on. Reported as bytes never used.
extern thread_local uint32_t __stack_base_lwm[];
extern uint32_t __stack_base_offset[];
static inline void paint_stack() {
    uint32_t* base = __stack_base_lwm + reinterpret_cast<uintptr_t>(__stack_base_offset);
    uint32_t* sp;
    asm volatile("mv %0,sp" : "=r"(sp));
    for (uint32_t* p = sp - 8; p != base;) {  // leave a few words for this function's own frame
        *--p = 0xBABABABAu;
    }
}
static inline uint32_t unused_stack_bytes() {
    uint32_t* base = __stack_base_lwm + reinterpret_cast<uintptr_t>(__stack_base_offset);
    uint32_t* p = base;
    while (*p == 0xBABABABAu) {
        ++p;
    }
    return static_cast<uint32_t>(reinterpret_cast<uintptr_t>(p) - reinterpret_cast<uintptr_t>(base));
}
#endif

void kernel_main() {
#if defined(TT_TA_ADDRGEN_STATS) && defined(ARCH_QUASAR)
    paint_stack();
#endif
    constexpr uint32_t iter_mode = get_arg(args::iter_mode);
    constexpr uint32_t implicit_sync = get_arg(args::implicit_sync);
    const uint32_t start_page = get_arg(args::start_page);
    const uint32_t num_pages = get_arg(args::num_pages);
    const uint32_t report_addr = get_arg(args::report_addr);

    Noc noc;
    DataflowBuffer dfb(dfb::out);
    const auto src = TensorAccessor(tensor::src);
    const uint32_t entry_size = dfb.get_entry_size();

    auto read_page = [&](const auto& endpoint, const auto& src_args) {
        if constexpr (implicit_sync) {
            noc.async_read<NocOptions::TXN_ID>(endpoint, dfb, src_args, {});
        } else {
            dfb.reserve_back(1);
            noc.async_read(endpoint, dfb, entry_size, src_args, {});
            noc.async_read_barrier();
            dfb.push_back(1);
        }
    };

    uint32_t transfers = 0;
    // Generic lambda: the shard modes' branches only compile for a sharded accessor, and a discarded `if constexpr`
    // branch is only skipped inside a template.
    auto run = [&](const auto& ta) {
        using TA = std::decay_t<decltype(ta)>;
        auto xfer = [&](const auto& endpoint, const auto& args) {
            read_page(endpoint, args);
            ++transfers;
        };
        if constexpr (iter_mode == 0) {
            for (uint32_t page_id = start_page; page_id < start_page + num_pages; ++page_id) {
                xfer(ta, typename noc_traits_t<TA>::src_args_type{.page_id = page_id});
            }
        } else if constexpr (iter_mode == 1) {
            for (const auto& page : ta.pages(start_page, start_page + num_pages)) {
                xfer(page, typename noc_traits_t<std::decay_t<decltype(page)>>::src_args_type{});
            }
        } else if constexpr (iter_mode == 2) {
            const PageView view(ta);
            for (uint32_t page_id = start_page; page_id < start_page + num_pages; ++page_id) {
                xfer(view, typename noc_traits_t<PageView<TA>>::src_args_type{.page_id = page_id});
            }
        } else if constexpr (iter_mode == 3) {
            const AbstractTensorAccessorWrapper wrapper(ta);
            for (uint32_t page_id = start_page; page_id < start_page + num_pages; ++page_id) {
                xfer(wrapper, typename noc_traits_t<AbstractTensorAccessorWrapper>::src_args_type{.page_id = page_id});
            }
        } else if constexpr (iter_mode == 6) {
            // Every other page, then the ones in between: a steady stride of 2 (skipped in hardware), one jump back.
            for (uint32_t pass = 0; pass < 2; ++pass) {
                for (uint32_t page_id = start_page + pass; page_id < start_page + num_pages; page_id += 2) {
                    xfer(ta, typename noc_traits_t<TA>::src_args_type{.page_id = page_id});
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
                        typename noc_traits_t<ShardView<TA>>::src_args_type{
                            .shard_id = shard, .offset_bytes = p * page_size});
                }
            }
        } else {
            // The tensor's pages shard by shard (padding skipped by the iterator).
            for (uint32_t shard = 0; shard < ta.dspec().num_shards(); ++shard) {
                for (const auto& page : ta.shard_pages(shard)) {
                    xfer(page, typename noc_traits_t<std::decay_t<decltype(page)>>::src_args_type{});
                }
            }
        }
    };
    run(src);
    dfb.finish();

#if defined(TT_TA_ADDRGEN_STATS)
    // Uncached: the host reads this back, and the DM data cache would otherwise hold the stores.
    volatile tt_l1_ptr uint32_t* report =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(report_addr + MEM_L1_UNCACHED_BASE);
    report[0] = tensor_accessor::detail::transfer_stats.hw;
    report[1] = tensor_accessor::detail::transfer_stats.pushes;
    report[2] = transfers;
    report[3] = unused_stack_bytes();
#else
    (void)report_addr;
    (void)transfers;
#endif
}
