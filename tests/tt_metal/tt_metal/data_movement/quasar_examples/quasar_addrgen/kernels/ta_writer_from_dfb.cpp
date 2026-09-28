// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// DFB -> TensorAccessor writer for the TensorAccessor AddrGen pattern matrix.
// Writes entries of dfb::in to pages [start_page, start_page + num_pages) of tensor::dst.
//
// Named CTAs:
//   iter_mode:     0 = explicit page_id loop (async_write(dfb, ta, {.page_id})), 1 = pages() iterator (Page endpoint)
//   implicit_sync: 1 = DFB implicit sync (async_write<TXN_ID>, the default Quasar path),
//                  0 = explicit wait_front / async_write / barrier / pop_front
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

    if constexpr (iter_mode == 0) {
        for (uint32_t page_id = start_page; page_id < start_page + num_pages; ++page_id) {
            write_page(dst, typename noc_traits_t<std::decay_t<decltype(dst)>>::dst_args_type{.page_id = page_id});
        }
    } else {
        for (const auto& page : dst.pages(start_page, start_page + num_pages)) {
            write_page(page, typename noc_traits_t<std::decay_t<decltype(page)>>::dst_args_type{});
        }
    }
    dfb.finish();
    dfb.write_barrier(noc);
}
