// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// BGE-M3 head-split writer for nlp_concat_heads.
//
// Output layout is [B, 1, S, num_heads * head_dim] in TILE_LAYOUT, with tiles
// ordered along the inner (num_heads*head_dim) axis first, then S, then B.
// For a single (batch, s_tile, head_group), the destination tile range is:
//     out_base = batch * (in0_h_tiles * per_tensor_tiles)
//              + s_tile * per_tensor_tiles
//              + group * group_tiles
// and we write group_tiles consecutive output tiles into it.
//
// Compile-time args:
//   0: head_groups          (e.g. 16)
//   1: heads_per_group      (e.g. 1)
//   2: in0_w_tiles          (= head_dim / TILE_W, e.g. 2)
//   3: per_tensor_tiles     (= num_heads * in0_w_tiles, e.g. 32)
//   4: in0_h_tiles          (= seq_len / TILE_H, e.g. 16)
//   5+: TensorAccessorArgs for the concatenated output tensor
//
// Runtime args:
//   0: out_tensor_addr
//   1: num_work_units
//   2: work_unit_start

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    const uint32_t out_tensor_addr = get_arg_val<uint32_t>(0);
    const uint32_t num_work_units = get_arg_val<uint32_t>(1);
    const uint32_t work_unit_start = get_arg_val<uint32_t>(2);

    constexpr uint32_t head_groups = get_compile_time_arg_val(0);
    constexpr uint32_t heads_per_group = get_compile_time_arg_val(1);
    constexpr uint32_t in0_w_tiles = get_compile_time_arg_val(2);
    constexpr uint32_t per_tensor_tiles = get_compile_time_arg_val(3);
    constexpr uint32_t in0_h_tiles = get_compile_time_arg_val(4);
    constexpr auto out_args = TensorAccessorArgs<5>();

    constexpr uint32_t cb_id = 0;
    constexpr uint32_t group_tiles = heads_per_group * in0_w_tiles;
    constexpr uint32_t s_block_tiles = in0_h_tiles * per_tensor_tiles;

    const uint32_t tile_size_bytes = get_tile_size(cb_id);
    const auto sout = TensorAccessor(out_args, out_tensor_addr);

    // Device 2.0 data-movement API (see device_api_migration_guide.md).
    Noc noc;
    CircularBuffer cb(cb_id);

    for (uint32_t w = 0; w < num_work_units; ++w) {
        const uint32_t work_unit = work_unit_start + w;
        const uint32_t block = work_unit / head_groups;
        const uint32_t group = work_unit - block * head_groups;
        const uint32_t batch = block / in0_h_tiles;
        const uint32_t h_tile = block - batch * in0_h_tiles;

        const uint32_t out_base = batch * s_block_tiles + h_tile * per_tensor_tiles + group * group_tiles;

        cb.wait_front(group_tiles);
        uint32_t l1_read_offset = 0;
        for (uint32_t i = 0; i < group_tiles; ++i) {
            noc.async_write(cb, sout, tile_size_bytes, {.offset_bytes = l1_read_offset}, {.page_id = out_base + i});
            l1_read_offset += tile_size_bytes;
        }
        noc.async_write_barrier();
        cb.pop_front(group_tiles);
    }
}
