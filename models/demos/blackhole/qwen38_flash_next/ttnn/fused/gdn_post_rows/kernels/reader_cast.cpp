// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// ``post_cast`` reader: the scan's fp32 output ``o`` [12, T, 128] in 4-tile groups.  Group g of this core covers the
// four column tiles of one (value head, tile row): pages (start + g) * 4 .. + 3 of the head-major page order
// (page (h * NC + c) * 4 + d), so a group is four consecutive pages and a core's run is one contiguous read.
// CBs: cb_in (0, fp32, 4-tile groups).
// Compile-time args: TensorAccessorArgs of o from 0.  Runtime args: 0 o address, 1 groups on this core, 2 first group.

#include "api/dataflow/dataflow_api.h"
#include "api/tensor/noc_traits.h"

namespace {
constexpr uint32_t cb_in = 0;
constexpr uint32_t group_tiles = 4;  // 128 / 32 column tiles of one head
constexpr uint32_t fp32_tile_bytes = 4096;
}  // namespace

void kernel_main() {
    constexpr auto scan_output_args = TensorAccessorArgs<0>();
    uint32_t arg = 0;
    const uint32_t scan_output_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t groups = get_arg_val<uint32_t>(arg++);
    const uint32_t first_group = get_arg_val<uint32_t>(arg++);
    const auto scan_output = TensorAccessor(scan_output_args, scan_output_addr);

    for (uint32_t g = 0; g < groups; ++g) {
        const uint32_t page = (first_group + g) * group_tiles;
        cb_reserve_back(cb_in, group_tiles);
        const uint32_t l1 = get_write_ptr(cb_in);
        for (uint32_t d = 0; d < group_tiles; ++d) {
            noc_async_read_page(page + d, scan_output, l1 + d * fp32_tile_bytes);
        }
        noc_async_read_barrier();
        cb_push_back(cb_in, group_tiles);
    }
}
