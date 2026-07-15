// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"
#include "api/kernel_thread_globals.h"

void kernel_main() {
    constexpr uint32_t num_entries_per_producer_a = get_arg(args::num_entries_per_producer_a);
    constexpr uint32_t num_entries_per_producer_b = get_arg(args::num_entries_per_producer_b);
    constexpr uint32_t num_producers              = get_arg(args::num_producers);

    const uint32_t chunk_offset_a    = get_arg(args::chunk_offset_a);
    const uint32_t chunk_offset_b    = get_arg(args::chunk_offset_b);
    const uint32_t entries_per_core_a = get_arg(args::entries_per_core_a);
    const uint32_t entries_per_core_b = get_arg(args::entries_per_core_b);

    const uint32_t producer_idx = get_my_thread_id();

    DataflowBuffer dfb_a(dfb::out_a);
    DataflowBuffer dfb_b(dfb::out_b);
    Noc noc;

    const uint32_t entry_size_a = dfb_a.get_entry_size();
    const uint32_t entry_size_b = dfb_b.get_entry_size();

    const auto src_a = TensorAccessor(tensor::src_a);
    const auto src_b = TensorAccessor(tensor::src_b);

    for (uint32_t tile = 0; tile < num_entries_per_producer_a; tile++) {
        const uint32_t page_id = chunk_offset_a + tile * num_producers + producer_idx;
        if (page_id >= chunk_offset_a + entries_per_core_a) {
            break;
        }
        dfb_a.reserve_back(1);
        noc.async_read(src_a, dfb_a, entry_size_a, {.page_id = page_id}, {});
        noc.async_read_barrier();
        dfb_a.push_back(1);
    }
    dfb_a.finish();

    for (uint32_t tile = 0; tile < num_entries_per_producer_b; tile++) {
        const uint32_t page_id = chunk_offset_b + tile * num_producers + producer_idx;
        if (page_id >= chunk_offset_b + entries_per_core_b) {
            break;
        }
        dfb_b.reserve_back(1);
        noc.async_read(src_b, dfb_b, entry_size_b, {.page_id = page_id}, {});
        noc.async_read_barrier();
        dfb_b.push_back(1);
    }
    dfb_b.finish();
}
