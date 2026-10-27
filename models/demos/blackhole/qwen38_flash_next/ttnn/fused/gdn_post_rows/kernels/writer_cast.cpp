// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// ``post_cast`` writer: the bf16 tiles back to ``o16`` [12, T, 128] at the input's own page order
// (page (h * NC + c) * 4 + d), four pages per group.
// CBs: cb_out (16, bf16).
// Compile-time args: TensorAccessorArgs of o16 from 0.  Runtime args: 0 o16 address, 1 groups on this core,
// 2 first group.

#include "api/dataflow/dataflow_api.h"
#include "api/tensor/noc_traits.h"

namespace {
constexpr uint32_t cb_out = 16;
constexpr uint32_t group_tiles = 4;  // 128 / 32 column tiles of one head
constexpr uint32_t bf16_tile_bytes = 2048;
}  // namespace

void kernel_main() {
    constexpr auto out_args = TensorAccessorArgs<0>();
    uint32_t arg = 0;
    const uint32_t out_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t groups = get_arg_val<uint32_t>(arg++);
    const uint32_t first_group = get_arg_val<uint32_t>(arg++);
    const auto out = TensorAccessor(out_args, out_addr);

    for (uint32_t g = 0; g < groups; ++g) {
        const uint32_t page = (first_group + g) * group_tiles;
        cb_wait_front(cb_out, group_tiles);
        const uint32_t l1 = get_read_ptr(cb_out);
        for (uint32_t d = 0; d < group_tiles; ++d) {
            noc_async_write_page(page + d, out, l1 + d * bf16_tile_bytes);
        }
        noc_async_write_barrier();
        cb_pop_front(cb_out, group_tiles);
    }
}
