// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// ``post_cast`` writer: the bf16 tiles back to ``o16`` [12, T, 128] at the input's own page order
// (page (h * NC + c) * 4 + d), four pages per group.
// CBs: cb_out (16, bf16).
// Compile-time args: TensorAccessorArgs of o16 from 0.  Runtime args: 0 o16 address, 1 groups on this core,
// 2 first group.

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

namespace {
constexpr uint32_t cb_out = 16;
constexpr uint32_t group_tiles = 4;  // 128 / 32 column tiles of one head
constexpr uint32_t bf16_tile_bytes = 2048;
}  // namespace

void kernel_main() {
    constexpr auto o16_args = TensorAccessorArgs<0>();
    uint32_t arg = 0;
    const uint32_t o16_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t groups = get_arg_val<uint32_t>(arg++);
    const uint32_t first_group = get_arg_val<uint32_t>(arg++);
    const auto o16 = TensorAccessor(o16_args, o16_addr);
    const Noc noc;

    for (uint32_t g = 0; g < groups; ++g) {
        const uint32_t page = (first_group + g) * group_tiles;
        CircularBuffer(cb_out).wait_front(group_tiles);
        for (uint32_t d = 0; d < group_tiles; ++d) {
            noc.async_write(
                CircularBuffer(cb_out),
                o16,
                o16.get_aligned_page_size(),
                {.offset_bytes = d * bf16_tile_bytes},
                {.page_id = page + d});
        }
        noc.async_write_barrier();
        CircularBuffer(cb_out).pop_front(group_tiles);
    }
}
