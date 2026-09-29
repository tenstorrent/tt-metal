// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// ``post_norm`` reader.  Once per core: the four gamma tiles of ``norm`` [1, 1, 1, 128] (the layernorm reader reads
// them as they are; ``mul_tiles_bcast_rows`` takes row 0), the SUM / REDUCE_ROW scaler tile (page 0 of ``scalars``)
// and the eps tile (page 1), both built host side exactly as the layernorm reader's
// ``calculate_and_prepare_reduce_scaler`` and ``generate_bcast_col_scalar`` fill them.  Per work unit
// u = head * NC + tile row: the four ``o16`` tiles (pages 4u .. 4u + 3 of the head-major order) and the four
// ``sig`` tiles (pages c * 48 + 4h + d of the token-major order).  For the history units (the last tile row,
// heads 0..9) also the eight projection tiles of that tile row's q|k|v columns 8h .. 8h + 7.
// CBs: CB_X (0), CB_SIG (1), CB_SCALER (2), CB_EPS (3), CB_GAMMA (4), CB_PROJ (5); all bf16.
// Compile-time args: TensorAccessorArgs of o16, sig, norm, scalars, projected, chained from 0.
// Runtime args: 0-4 those addresses, 5 chunks (NC), 6 history, 7 units on this core, 8 first unit.

#include "api/dataflow/dataflow_api.h"
#include "api/tensor/noc_traits.h"
#include "../../kernels/zones.h"

namespace {
constexpr uint32_t CB_X = 0, CB_SIG = 1, CB_SCALER = 2, CB_EPS = 3, CB_GAMMA = 4, CB_PROJ = 5;
constexpr uint32_t HEAD_TILES = 4;          // 128 / 32
constexpr uint32_t VALUE_TILES = 48;        // 1536 / 32
constexpr uint32_t PROJECTION_TILES = 130;  // 4160 / 32
constexpr uint32_t HISTORY_HEADS = 10;      // 80 q|k|v column tiles over 8 tiles per head
constexpr uint32_t HISTORY_COLUMNS = 8;
constexpr uint32_t BF16_TILE = 2048;
}  // namespace

void kernel_main() {
    constexpr auto x_args = TensorAccessorArgs<0>();
    constexpr auto sig_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
    constexpr auto gamma_args = TensorAccessorArgs<sig_args.next_compile_time_args_offset()>();
    constexpr auto scalars_args = TensorAccessorArgs<gamma_args.next_compile_time_args_offset()>();
    constexpr auto proj_args = TensorAccessorArgs<scalars_args.next_compile_time_args_offset()>();

    uint32_t arg = 0;
    const uint32_t x_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t sig_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t gamma_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t scalars_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t proj_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t chunks = get_arg_val<uint32_t>(arg++);
    const uint32_t history = get_arg_val<uint32_t>(arg++);
    const uint32_t units = get_arg_val<uint32_t>(arg++);
    const uint32_t first = get_arg_val<uint32_t>(arg++);

    const auto x = TensorAccessor(x_args, x_addr);
    const auto sig = TensorAccessor(sig_args, sig_addr);
    const auto gamma = TensorAccessor(gamma_args, gamma_addr);
    const auto scalars = TensorAccessor(scalars_args, scalars_addr);
    const auto proj = TensorAccessor(proj_args, proj_addr);

    {
        FUSED_ZONE("fz_gpo_rn_setup");
        cb_reserve_back(CB_GAMMA, HEAD_TILES);
        const uint32_t g_l1 = get_write_ptr(CB_GAMMA);
        for (uint32_t d = 0; d < HEAD_TILES; ++d) {
            noc_async_read_page(d, gamma, g_l1 + d * BF16_TILE);
        }
        cb_reserve_back(CB_SCALER, 1);
        noc_async_read_page(0, scalars, get_write_ptr(CB_SCALER));
        cb_reserve_back(CB_EPS, 1);
        noc_async_read_page(1, scalars, get_write_ptr(CB_EPS));
        noc_async_read_barrier();
        cb_push_back(CB_GAMMA, HEAD_TILES);
        cb_push_back(CB_SCALER, 1);
        cb_push_back(CB_EPS, 1);
    }

    for (uint32_t i = 0; i < units; ++i) {
        const uint32_t unit = first + i;
        const uint32_t head = unit / chunks;
        const uint32_t chunk = unit - head * chunks;

        {
            FUSED_ZONE("fz_gpo_rn_tiles");
            cb_reserve_back(CB_X, HEAD_TILES);
            cb_reserve_back(CB_SIG, HEAD_TILES);
            const uint32_t x_l1 = get_write_ptr(CB_X);
            const uint32_t s_l1 = get_write_ptr(CB_SIG);
            for (uint32_t d = 0; d < HEAD_TILES; ++d) {
                noc_async_read_page(unit * HEAD_TILES + d, x, x_l1 + d * BF16_TILE);
                noc_async_read_page(chunk * VALUE_TILES + head * HEAD_TILES + d, sig, s_l1 + d * BF16_TILE);
            }
            noc_async_read_barrier();
            cb_push_back(CB_X, HEAD_TILES);
            cb_push_back(CB_SIG, HEAD_TILES);
        }

        if (history != 0 && chunk + 1 == chunks && head < HISTORY_HEADS) {
            FUSED_ZONE("fz_gpo_rn_history");
            cb_reserve_back(CB_PROJ, HISTORY_COLUMNS);
            const uint32_t p_l1 = get_write_ptr(CB_PROJ);
            for (uint32_t j = 0; j < HISTORY_COLUMNS; ++j) {
                noc_async_read_page(chunk * PROJECTION_TILES + head * HISTORY_COLUMNS + j, proj, p_l1 + j * BF16_TILE);
            }
            noc_async_read_barrier();
            cb_push_back(CB_PROJ, HISTORY_COLUMNS);
        }
    }
}
