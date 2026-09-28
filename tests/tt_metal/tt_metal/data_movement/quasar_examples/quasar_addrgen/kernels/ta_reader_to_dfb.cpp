// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// TensorAccessor -> DFB reader for the TensorAccessor AddrGen pattern matrix.
// Reads pages [start_page, start_page + num_pages) of tensor::src into dfb::out, one page per entry.
//
// Named CTAs:
//   iter_mode:     0 = explicit page_id loop (async_read(ta, dfb, {.page_id})), 1 = pages() iterator (Page endpoint)
//   implicit_sync: 1 = DFB implicit sync (async_read<TXN_ID>, the default Quasar path),
//                  0 = explicit reserve_back / async_read / barrier / push_back
// Named RTAs:
//   start_page, num_pages

#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    constexpr uint32_t iter_mode = get_arg(args::iter_mode);
    constexpr uint32_t implicit_sync = get_arg(args::implicit_sync);
    const uint32_t start_page = get_arg(args::start_page);
    const uint32_t num_pages = get_arg(args::num_pages);

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

    if constexpr (iter_mode == 0) {
        for (uint32_t page_id = start_page; page_id < start_page + num_pages; ++page_id) {
            read_page(src, typename noc_traits_t<std::decay_t<decltype(src)>>::src_args_type{.page_id = page_id});
        }
    } else {
        for (const auto& page : src.pages(start_page, start_page + num_pages)) {
            read_page(page, typename noc_traits_t<std::decay_t<decltype(page)>>::src_args_type{});
        }
    }
    dfb.finish();
}
