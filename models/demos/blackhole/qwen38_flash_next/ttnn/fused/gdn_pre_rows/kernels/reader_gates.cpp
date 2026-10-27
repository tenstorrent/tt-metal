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

    {
        cb_reserve_back(cb_const, constant_tiles);
        const uint32_t l1 = get_write_ptr(cb_const);
        for (uint32_t t = 0; t < constant_tiles; ++t) {
            noc_async_read_page(t, constants, l1 + t * fp32_tile_bytes);
        }
        noc_async_read_barrier();
        cb_push_back(cb_const, constant_tiles);
    }

    for (uint32_t unit = 0; unit < units; ++unit) {
        const uint32_t chunk = get_arg_val<uint32_t>(arg++);
        const uint32_t kind = get_arg_val<uint32_t>(arg++);
        const uint32_t base = chunk * projection_tiles;
        if (kind == 0) {
            cb_reserve_back(cb_ab, 2);
            const uint32_t l1 = get_write_ptr(cb_ab);
            noc_async_read_page(base + a_tile, projected, l1);
            noc_async_read_page(base + b_tile, projected, l1 + bf16_tile_bytes);
            noc_async_read_barrier();
            cb_push_back(cb_ab, 2);
        } else {
            const uint32_t head = kind - 1;
            cb_reserve_back(cb_z, head_tiles);
            const uint32_t l1 = get_write_ptr(cb_z);
            for (uint32_t d = 0; d < head_tiles; ++d) {
                noc_async_read_page(base + z_tile0 + head * head_tiles + d, projected, l1 + d * bf16_tile_bytes);
            }
            noc_async_read_barrier();
            cb_push_back(cb_z, head_tiles);
        }
    }
}
