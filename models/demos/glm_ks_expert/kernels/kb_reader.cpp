// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Stage 1 bench reader (BRISC): copies four DRAM tensors into CB 0 / 1 / 2 / 3 once and pushes them (a count of 0
// skips one); the compute kernel then loops over the resident tiles.
// RT: 0-3 addresses, 4-7 tile counts. CT: TensorAccessorArgs of the four tensors.
#include "api/dataflow/dataflow_api.h"

template <typename S>
FORCE_INLINE void load(uint32_t cb, const S& s, uint32_t n) {
    if (n == 0) {
        return;
    }
    cb_reserve_back(cb, n);
    uint32_t l1 = get_write_ptr(cb);
    for (uint32_t i = 0; i < n; ++i) {
        noc_async_read_page(i, s, l1);
        l1 += get_tile_size(cb);
    }
    noc_async_read_barrier();
    cb_push_back(cb, n);
}

void kernel_main() {
    constexpr auto a0 = TensorAccessorArgs<0>();
    constexpr auto a1 = TensorAccessorArgs<a0.next_compile_time_args_offset()>();
    constexpr auto a2 = TensorAccessorArgs<a1.next_compile_time_args_offset()>();
    constexpr auto a3 = TensorAccessorArgs<a2.next_compile_time_args_offset()>();
    load(0, TensorAccessor(a0, get_arg_val<uint32_t>(0), get_tile_size(0)), get_arg_val<uint32_t>(4));
    load(1, TensorAccessor(a1, get_arg_val<uint32_t>(1), get_tile_size(1)), get_arg_val<uint32_t>(5));
    load(2, TensorAccessor(a2, get_arg_val<uint32_t>(2), get_tile_size(2)), get_arg_val<uint32_t>(6));
    load(3, TensorAccessor(a3, get_arg_val<uint32_t>(3), get_tile_size(3)), get_arg_val<uint32_t>(7));
}
