// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Microbench reader (BRISC): copies in0 [N0 tiles] and in1 [N1 tiles] from DRAM into CB 0 / CB 1 once and pushes
// them; the compute kernel then loops over the resident tiles.
// RT: 0 in0 addr, 1 in1 addr, 2 N0, 3 N1. CT: TensorAccessorArgs(in0), TensorAccessorArgs(in1).
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t a0 = get_arg_val<uint32_t>(0);
    const uint32_t a1 = get_arg_val<uint32_t>(1);
    const uint32_t n0 = get_arg_val<uint32_t>(2);
    const uint32_t n1 = get_arg_val<uint32_t>(3);
    constexpr auto args0 = TensorAccessorArgs<0>();
    constexpr auto args1 = TensorAccessorArgs<args0.next_compile_time_args_offset()>();
    const auto s0 = TensorAccessor(args0, a0, get_tile_size(0));
    const auto s1 = TensorAccessor(args1, a1, get_tile_size(1));
    cb_reserve_back(0, n0);
    uint32_t l1 = get_write_ptr(0);
    for (uint32_t i = 0; i < n0; ++i) {
        noc_async_read_page(i, s0, l1);
        l1 += get_tile_size(0);
    }
    cb_reserve_back(1, n1);
    l1 = get_write_ptr(1);
    for (uint32_t i = 0; i < n1; ++i) {
        noc_async_read_page(i, s1, l1);
        l1 += get_tile_size(1);
    }
    noc_async_read_barrier();
    cb_push_back(0, n0);
    cb_push_back(1, n1);
}
