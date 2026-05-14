// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    uint32_t src_addr = get_arg_val<uint32_t>(0);
    uint32_t n_tiles = get_arg_val<uint32_t>(1);

    constexpr uint32_t cb_in = 0;

    constexpr auto s_args = TensorAccessorArgs<0>();
    const auto s = TensorAccessor(s_args, src_addr);

    for (uint32_t i = 0; i < n_tiles; i++) {
        cb_reserve_back(cb_in, 1);
        uint32_t l1_addr = get_write_ptr(cb_in);
        noc_async_read_tile(i, s, l1_addr);
        noc_async_read_barrier();
        cb_push_back(cb_in, 1);
    }
}
