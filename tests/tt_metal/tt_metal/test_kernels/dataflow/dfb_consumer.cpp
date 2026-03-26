// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "experimental/dataflow_buffer.h"
#include "experimental/noc.h"
#include "experimental/tensor.h"
#include "api/debug/dprint.h"

void kernel_main() {
    const uint32_t dst_addr_base = get_compile_time_arg_val(0);
    const uint32_t num_entries_per_consumer = get_compile_time_arg_val(1);
    const uint32_t blocked_consumer = get_compile_time_arg_val(2);
    constexpr uint32_t implicit_sync = get_compile_time_arg_val(3);
    constexpr auto dst_args = TensorAccessorArgs<4>();

    uint32_t consumer_mask = get_arg_val<uint32_t>(0);
    uint32_t logical_dfb_id = get_arg_val<uint32_t>(1);
    // Base page offset for this core's slice of the global buffer.
    // Single-core callers pass 0; multi-core callers pass core_idx * entries_per_core.
    const uint32_t chunk_offset = get_arg_val<uint32_t>(2);
    const uint32_t num_consumers = static_cast<uint32_t>(__builtin_popcount(consumer_mask));

    experimental::DataflowBuffer dfb(logical_dfb_id);
    experimental::Noc noc;

    // TODO: Replace with get_thread_idx() kernel API when available
    std::uint64_t hartid;
    asm volatile("csrr %0, mhartid" : "=r"(hartid));
    uint32_t consumer_idx = static_cast<uint32_t>(__builtin_popcount(consumer_mask & ((1u << hartid) - 1u)));

    // DPRINT << "consumer_idx: " << consumer_idx << " num_entries_per_consumer: " << num_entries_per_consumer <<
    // ENDL();

    uint32_t entry_size = dfb.get_entry_size();
    const auto tensor_accessor = TensorAccessor(dst_args, dst_addr_base, entry_size);

    for (uint32_t tile_id = 0; tile_id < num_entries_per_consumer; tile_id++) {
        uint32_t page_id = 0;
        if constexpr (blocked_consumer) {
            page_id = chunk_offset + tile_id;
        } else {
            page_id = chunk_offset + tile_id * num_consumers + consumer_idx;
        }
        DPRINT << "consumer tile id " << tile_id << " page id " << page_id << ENDL();
        if constexpr (implicit_sync) {
            dfb.write_out(noc, tensor_accessor, {.page_id = page_id});
        } else {
            dfb.wait_front(1);
            noc.async_write(dfb, tensor_accessor, entry_size, {}, {.page_id = page_id});
            noc.async_write_barrier();
            dfb.pop_front(1);
        }
    }
    DPRINT << "CBW" << ENDL();
    noc.async_write_barrier();
    DPRINT << "CBWD" << ENDL();
}
