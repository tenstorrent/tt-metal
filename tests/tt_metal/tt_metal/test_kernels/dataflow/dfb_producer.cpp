// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "experimental/dataflow_buffer.h"
#include "experimental/noc.h"
#include "experimental/tensor.h"
#include "api/debug/dprint.h"

void kernel_main() {
    const uint32_t src_addr_base = get_compile_time_arg_val(0);
    const uint32_t num_entries_per_producer = get_compile_time_arg_val(1);
    constexpr auto src_args = TensorAccessorArgs<2>();

    uint32_t producer_mask = get_arg_val<uint32_t>(0);

    experimental::DataflowBuffer dfb(0);
    experimental::Noc noc;

    // kinda weird to do this to get the producer idx
    std::uint64_t hartid;
    asm volatile("csrr %0, mhartid" : "=r"(hartid));
    uint32_t producer_idx = static_cast<uint32_t>(__builtin_popcount(producer_mask & ((1u << hartid) - 1u)));

    uint32_t entry_size = dfb.get_entry_size();
    const auto tensor_accessor = TensorAccessor(src_args, src_addr_base, entry_size);

    for (uint32_t tile_id = 0; tile_id < num_entries_per_producer; tile_id++) {
        dfb.reserve_back(1);
        noc.async_read(
            tensor_accessor, dfb, entry_size, {.page_id = producer_idx * num_entries_per_producer + tile_id}, {});
        noc.async_read_barrier();
        dfb.push_back(1);
    }
    dfb.finish();
}
