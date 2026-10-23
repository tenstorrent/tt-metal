// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Group B reader.  Once per core: the two full fp32 constant tiles (dt_bias and neg_exp_A, each the row
// [c[0..11], 0 x 20] repeated in all 32 rows -- the bits the Blackhole binary_ng ROW-broadcast reader would build
// with fill_tile_with_first_row, built on the host instead so the RISC stays off the compute's critical path).
// Per unit of tile row c: kind 0 reads the a and b column tiles (projection pages c * 130 + 128 and + 129) into
// CB_AB; kind 1..12 reads value head kind - 1 of z (pages c * 130 + 80 + 4 hv + d) into CB_Z.
//
// Compile-time args: TensorAccessorArgs of projected, constants (no chunk count).  Runtime args: 0 projected,
// 1 constants addresses, 2 units, then (tile row, kind) pairs.

#include "api/dataflow/dataflow_api.h"
#include "api/tensor/noc_traits.h"
#include "../../kernels/zones.h"

namespace {
constexpr uint32_t CB_AB = 20, CB_Z = 21, CB_CONST = 22;
constexpr uint32_t HEAD_TILES = 4, PROJECTION_TILES = 130, Z_TILE0 = 80, A_TILE = 128, B_TILE = 129;
constexpr uint32_t CONST_TILES = 2, BF16_TILE = 2048, FP32_TILE = 4096;
}  // namespace

void kernel_main() {
    constexpr auto p_args = TensorAccessorArgs<0>();
    constexpr auto c_args = TensorAccessorArgs<p_args.next_compile_time_args_offset()>();

    const uint32_t p_addr = get_arg_val<uint32_t>(0);
    const uint32_t c_addr = get_arg_val<uint32_t>(1);
    const uint32_t units = get_arg_val<uint32_t>(2);
    uint32_t arg = 3;

    const auto p = TensorAccessor(p_args, p_addr);
    const auto k = TensorAccessor(c_args, c_addr);

    {
        FUSED_ZONE("fz_gpr_rg_setup");
        cb_reserve_back(CB_CONST, CONST_TILES);
        const uint32_t l1 = get_write_ptr(CB_CONST);
        for (uint32_t t = 0; t < CONST_TILES; ++t) {
            noc_async_read_page(t, k, l1 + t * FP32_TILE);
        }
        noc_async_read_barrier();
        cb_push_back(CB_CONST, CONST_TILES);
    }

    for (uint32_t unit = 0; unit < units; ++unit) {
        const uint32_t chunk = get_arg_val<uint32_t>(arg++);
        const uint32_t kind = get_arg_val<uint32_t>(arg++);
        const uint32_t base = chunk * PROJECTION_TILES;
        if (kind == 0) {
            FUSED_ZONE("fz_gpr_rg_ab");
            cb_reserve_back(CB_AB, 2);
            const uint32_t l1 = get_write_ptr(CB_AB);
            noc_async_read_page(base + A_TILE, p, l1);
            noc_async_read_page(base + B_TILE, p, l1 + BF16_TILE);
            noc_async_read_barrier();
            cb_push_back(CB_AB, 2);
        } else {
            FUSED_ZONE("fz_gpr_rg_z");
            const uint32_t head = kind - 1;
            cb_reserve_back(CB_Z, HEAD_TILES);
            const uint32_t l1 = get_write_ptr(CB_Z);
            for (uint32_t d = 0; d < HEAD_TILES; ++d) {
                noc_async_read_page(base + Z_TILE0 + head * HEAD_TILES + d, p, l1 + d * BF16_TILE);
            }
            noc_async_read_barrier();
            cb_push_back(CB_Z, HEAD_TILES);
        }
    }
}
