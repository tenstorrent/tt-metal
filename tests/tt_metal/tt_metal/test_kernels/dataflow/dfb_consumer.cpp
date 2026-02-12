// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "experimental/dataflow_buffer.h"
#ifndef COMPILE_FOR_TRISC
#include "experimental/noc.h"
#include "experimental/tensor.h"
#endif
#include "api/debug/dprint.h"

void kernel_main() {
    DPRINT << "dfb_consumer kernel started" << ENDL();
    const uint32_t dst_addr_base = get_compile_time_arg_val(0);
    const uint32_t num_entries_per_consumer = get_compile_time_arg_val(1);
    const uint32_t blocked_consumer = get_compile_time_arg_val(2);
#ifndef COMPILE_FOR_TRISC
    constexpr auto dst_args = TensorAccessorArgs<3>();
#endif

    // uint32_t consumer_mask = get_arg_val<uint32_t>(0);
    const uint32_t num_consumers = 1;  // static_cast<uint32_t>(__builtin_popcount(consumer_mask));

    experimental::DataflowBuffer dfb(0);
#ifndef COMPILE_FOR_TRISC
    experimental::Noc noc;

    std::uint64_t hartid;
    asm volatile("csrr %0, mhartid" : "=r"(hartid));
    uint32_t consumer_idx = static_cast<uint32_t>(__builtin_popcount(consumer_mask & ((1u << hartid) - 1u)));
#else
    uint32_t consumer_idx = 0;
#endif

    DPRINT << "consumer_idx: " << consumer_idx << " num_entries_per_consumer: " << num_entries_per_consumer << ENDL();

    // uint32_t dst_addr_base = get_arg_val<uint32_t>(0);
#ifndef COMPILE_FOR_TRISC
    uint32_t entry_size = dfb.get_entry_size();
    const auto tensor_accessor = TensorAccessor(dst_args, dst_addr_base, entry_size);
#endif

    for (uint32_t tile_id = 0; tile_id < num_entries_per_consumer; tile_id++) {
        // DPRINT << "wfw" << ENDL();
        dfb.wait_front(1);
        // in blocked case maybe each consumer can modify the data so host knows that each have consumed it
        // DPRINT << "wfd" << ENDL();
#ifndef COMPILE_FOR_TRISC
        uint32_t page_id = 0;
        if constexpr (blocked_consumer) {
            page_id = tile_id;
        } else {
            page_id = tile_id * num_consumers + consumer_idx;
        }
        DPRINT << "consumer tile id " << tile_id << " page id " << page_id << ENDL();
        // for blocked consumer each consumer reads each tile .. user kernel shouldn't have to think about this (tensor
        // accessor will abstract)
        noc.async_write(dfb, tensor_accessor, entry_size, {}, {.page_id = page_id});
#endif
        DPRINT << "consumer tile id " << tile_id << ENDL();
        // DPRINT << "pfw" << ENDL();
        dfb.pop_front(1);
        // DPRINT << "pfd" << ENDL();
    }

#ifndef COMPILE_FOR_TRISC
    DPRINT << "CBW" << ENDL();
    noc.async_write_barrier();
    DPRINT << "CBWD" << ENDL();
#endif
    DPRINT << "dfb_consumer kernel finished" << ENDL();
}
