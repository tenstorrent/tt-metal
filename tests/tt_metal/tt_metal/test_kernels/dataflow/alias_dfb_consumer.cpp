// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"
#include "api/kernel_thread_globals.h"

void kernel_main() {
    constexpr uint32_t num_entries_per_consumer_a = get_arg(args::num_entries_per_consumer_a);
    constexpr uint32_t num_entries_per_consumer_b = get_arg(args::num_entries_per_consumer_b);
    constexpr uint32_t num_consumers              = get_arg(args::num_consumers);

    const uint32_t chunk_offset_a     = get_arg(args::chunk_offset_a);
    const uint32_t chunk_offset_b     = get_arg(args::chunk_offset_b);
    const uint32_t entries_per_core_a = get_arg(args::entries_per_core_a);
    const uint32_t entries_per_core_b = get_arg(args::entries_per_core_b);

    const uint32_t consumer_idx = get_my_thread_id();

    DataflowBuffer dfb_a(dfb::in_a);
    DataflowBuffer dfb_b(dfb::in_b);
    Noc noc;

    const uint32_t entry_size_a = dfb_a.get_entry_size();
    const uint32_t entry_size_b = dfb_b.get_entry_size();

    const auto dst_a = TensorAccessor(tensor::dst_a);
    const auto dst_b = TensorAccessor(tensor::dst_b);

    for (uint32_t tile = 0; tile < num_entries_per_consumer_a; tile++) {
        const uint32_t page_id = chunk_offset_a + tile * num_consumers + consumer_idx;
        if (page_id >= chunk_offset_a + entries_per_core_a) {
            break;
        }
        dfb_a.wait_front(1);
        noc.async_write(dfb_a, dst_a, entry_size_a, {}, {.page_id = page_id});
        noc.async_write_barrier();
        dfb_a.pop_front(1);
    }
    dfb_a.finish();
    dfb_a.write_barrier(noc);

    for (uint32_t tile = 0; tile < num_entries_per_consumer_b; tile++) {
        const uint32_t page_id = chunk_offset_b + tile * num_consumers + consumer_idx;
        if (page_id >= chunk_offset_b + entries_per_core_b) {
            break;
        }
        dfb_b.wait_front(1);
        noc.async_write(dfb_b, dst_b, entry_size_b, {}, {.page_id = page_id});
        noc.async_write_barrier();
        dfb_b.pop_front(1);
    }
    dfb_b.finish();
    dfb_b.write_barrier(noc);
}
