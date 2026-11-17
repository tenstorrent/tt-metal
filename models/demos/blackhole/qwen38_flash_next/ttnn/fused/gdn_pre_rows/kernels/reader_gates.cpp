// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Group B reader. Once per core: the two full fp32 constant tiles (the dt_bias and neg_exp_A rows repeated in all 32
// rows, host-built with the same bits as binary_ng's row-broadcast reader). Per unit of tile row c: kind 0 reads the
// a and b column tiles (pages c * 130 + 128 and + 129) into cb_ab; kind 1..12 reads value head kind - 1 of z (pages
// c * 130 + 80 + 4 hv + d) into cb_z.
//
// Compile-time args: TensorAccessorArgs of projected, constants (no chunk count).  Runtime args: 0 projected,
// 1 constants addresses, 2 units, then (tile row, kind) pairs.

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

namespace {
constexpr uint32_t cb_ab = 20, cb_z = 21, cb_const = 22;
constexpr uint32_t head_tiles = 4, projection_tiles = 130, z_tile0 = 80, a_tile = 128, b_tile = 129;
constexpr uint32_t constant_tiles = 2, bf16_tile_bytes = 2048, fp32_tile_bytes = 4096;
}  // namespace

void kernel_main() {
    constexpr auto projected_args = TensorAccessorArgs<0>();
    constexpr auto constants_args = TensorAccessorArgs<projected_args.next_compile_time_args_offset()>();

    const uint32_t projected_addr = get_arg_val<uint32_t>(0);
    const uint32_t constants_addr = get_arg_val<uint32_t>(1);
    const uint32_t units = get_arg_val<uint32_t>(2);
    uint32_t arg = 3;

    const auto projected = TensorAccessor(projected_args, projected_addr);
    const auto constants = TensorAccessor(constants_args, constants_addr);
    const Noc noc;

    {
        CircularBuffer(cb_const).reserve_back(constant_tiles);
        for (uint32_t t = 0; t < constant_tiles; ++t) {
            noc.async_read(
                constants,
                CircularBuffer(cb_const),
                constants.get_aligned_page_size(),
                {.page_id = t},
                {.offset_bytes = t * fp32_tile_bytes});
        }
        noc.async_read_barrier();
        CircularBuffer(cb_const).push_back(constant_tiles);
    }

    for (uint32_t unit = 0; unit < units; ++unit) {
        const uint32_t chunk = get_arg_val<uint32_t>(arg++);
        const uint32_t kind = get_arg_val<uint32_t>(arg++);
        const uint32_t base = chunk * projection_tiles;
        if (kind == 0) {
            CircularBuffer(cb_ab).reserve_back(2);
            noc.async_read(
                projected,
                CircularBuffer(cb_ab),
                projected.get_aligned_page_size(),
                {.page_id = base + a_tile},
                {.offset_bytes = 0});
            noc.async_read(
                projected,
                CircularBuffer(cb_ab),
                projected.get_aligned_page_size(),
                {.page_id = base + b_tile},
                {.offset_bytes = bf16_tile_bytes});
            noc.async_read_barrier();
            CircularBuffer(cb_ab).push_back(2);
        } else {
            const uint32_t head = kind - 1;
            CircularBuffer(cb_z).reserve_back(head_tiles);
            for (uint32_t d = 0; d < head_tiles; ++d) {
                noc.async_read(
                    projected,
                    CircularBuffer(cb_z),
                    projected.get_aligned_page_size(),
                    {.page_id = base + z_tile0 + head * head_tiles + d},
                    {.offset_bytes = d * bf16_tile_bytes});
            }
            noc.async_read_barrier();
            CircularBuffer(cb_z).push_back(head_tiles);
        }
    }
}
