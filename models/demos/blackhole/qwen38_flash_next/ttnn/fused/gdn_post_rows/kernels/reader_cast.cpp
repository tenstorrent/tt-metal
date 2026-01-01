// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// ``post_cast`` reader: the scan's fp32 output ``o`` [12, T, 128] in 4-tile groups.  Group g of this core covers the
// four column tiles of one (value head, tile row): pages (first_group + g) * 4 .. + 3 of the head-major page order
// (page (h * NC + c) * 4 + d), so a group is four consecutive pages and a core's run is one contiguous read.
// CBs: cb_in (0, fp32, 4-tile groups).
// Compile-time args: TensorAccessorArgs of o from 0.  Runtime args: 0 o address, 1 groups on this core, 2 first group.

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
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
    const Noc noc;

    for (uint32_t g = 0; g < groups; ++g) {
        const uint32_t page = (first_group + g) * group_tiles;
        CircularBuffer(cb_in).reserve_back(group_tiles);
        for (uint32_t d = 0; d < group_tiles; ++d) {
            noc.async_read(
                scan_output,
                CircularBuffer(cb_in),
                scan_output.get_aligned_page_size(),
                {.page_id = page + d},
                {.offset_bytes = d * fp32_tile_bytes});
        }
        noc.async_read_barrier();
        CircularBuffer(cb_in).push_back(group_tiles);
    }
}
