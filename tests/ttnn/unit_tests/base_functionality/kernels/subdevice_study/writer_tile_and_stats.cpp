// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Waits for the compute result tile, writes it to page `page` of the output tensor, and writes this core's
// [t_start, t_end] wall-clock ticks to page `page` of the stats tensor (16 x uint32 per page).
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint64_t t_start = get_timestamp();
    const uint32_t out_addr = get_arg_val<uint32_t>(0);
    const uint32_t stats_addr = get_arg_val<uint32_t>(1);
    const uint32_t page = get_arg_val<uint32_t>(2);
    constexpr uint32_t cb_out = get_named_compile_time_arg_val("cb_out");
    constexpr uint32_t cb_stats = get_named_compile_time_arg_val("cb_stats");
    constexpr auto out_args = TensorAccessorArgs<0>();
    constexpr auto st_args = TensorAccessorArgs<out_args.next_compile_time_args_offset()>();
    const auto out = TensorAccessor(out_args, out_addr);
    const auto st = TensorAccessor(st_args, stats_addr);

    cb_wait_front(cb_out, 1);
    const uint64_t t_end = get_timestamp();
    noc_async_write(get_read_ptr(cb_out), out.get_noc_addr(page), get_tile_size(cb_out));

    volatile tt_l1_ptr uint32_t* w = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_stats));
    for (uint32_t i = 0; i < 16; ++i) {
        w[i] = 0;
    }
    w[0] = static_cast<uint32_t>(t_start);
    w[1] = static_cast<uint32_t>(t_start >> 32);
    w[2] = static_cast<uint32_t>(t_end);
    w[3] = static_cast<uint32_t>(t_end >> 32);
    noc_async_write(get_write_ptr(cb_stats), st.get_noc_addr(page), 64);
    noc_async_write_barrier();
    cb_pop_front(cb_out, 1);
}
