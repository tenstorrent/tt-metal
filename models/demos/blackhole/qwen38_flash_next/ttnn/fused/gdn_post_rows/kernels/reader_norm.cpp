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
// CBs: cb_x (0), cb_sig (1), cb_scaler (2), cb_eps (3), cb_gamma (4), cb_proj (5); all bf16.
// Compile-time args: TensorAccessorArgs of o16, sig, norm, scalars, projected, chained from 0.
// Runtime args: 0-4 those addresses, 5 chunks (NC), 6 history, 7 units on this core, 8 first unit.

#include "api/dataflow/dataflow_api.h"
#include "api/tensor/noc_traits.h"

namespace {
constexpr uint32_t cb_x = 0, cb_sig = 1, cb_scaler = 2, cb_eps = 3, cb_gamma = 4, cb_proj = 5;
constexpr uint32_t head_tiles = 4;          // 128 / 32
constexpr uint32_t value_tiles = 48;        // 1536 / 32
constexpr uint32_t projection_tiles = 130;  // 4160 / 32
constexpr uint32_t history_heads = 10;      // 80 q|k|v column tiles over 8 tiles per head
constexpr uint32_t history_columns = 8;
constexpr uint32_t bf16_tile_bytes = 2048;
}  // namespace

void kernel_main() {
    constexpr auto o16_args = TensorAccessorArgs<0>();
    constexpr auto sig_args = TensorAccessorArgs<o16_args.next_compile_time_args_offset()>();
    constexpr auto gamma_args = TensorAccessorArgs<sig_args.next_compile_time_args_offset()>();
    constexpr auto scalars_args = TensorAccessorArgs<gamma_args.next_compile_time_args_offset()>();
    constexpr auto projected_args = TensorAccessorArgs<scalars_args.next_compile_time_args_offset()>();

    uint32_t arg = 0;
    const uint32_t o16_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t sig_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t gamma_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t scalars_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t projected_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t chunks = get_arg_val<uint32_t>(arg++);
    const uint32_t history = get_arg_val<uint32_t>(arg++);
    const uint32_t units = get_arg_val<uint32_t>(arg++);
    const uint32_t first_unit = get_arg_val<uint32_t>(arg++);

    const auto o16 = TensorAccessor(o16_args, o16_addr);
    const auto sig = TensorAccessor(sig_args, sig_addr);
    const auto gamma = TensorAccessor(gamma_args, gamma_addr);
    const auto scalars = TensorAccessor(scalars_args, scalars_addr);
    const auto projected = TensorAccessor(projected_args, projected_addr);

    {
        cb_reserve_back(cb_gamma, head_tiles);
        const uint32_t g_l1 = get_write_ptr(cb_gamma);
        for (uint32_t d = 0; d < head_tiles; ++d) {
            noc_async_read_page(d, gamma, g_l1 + d * bf16_tile_bytes);
        }
        cb_reserve_back(cb_scaler, 1);
        noc_async_read_page(0, scalars, get_write_ptr(cb_scaler));
        cb_reserve_back(cb_eps, 1);
        noc_async_read_page(1, scalars, get_write_ptr(cb_eps));
        noc_async_read_barrier();
        cb_push_back(cb_gamma, head_tiles);
        cb_push_back(cb_scaler, 1);
        cb_push_back(cb_eps, 1);
    }

    for (uint32_t i = 0; i < units; ++i) {
        const uint32_t unit = first_unit + i;
        const uint32_t head = unit / chunks;
        const uint32_t chunk = unit - head * chunks;

        {
            cb_reserve_back(cb_x, head_tiles);
            cb_reserve_back(cb_sig, head_tiles);
            const uint32_t x_l1 = get_write_ptr(cb_x);
            const uint32_t s_l1 = get_write_ptr(cb_sig);
            for (uint32_t d = 0; d < head_tiles; ++d) {
                noc_async_read_page(unit * head_tiles + d, o16, x_l1 + d * bf16_tile_bytes);
                noc_async_read_page(chunk * value_tiles + head * head_tiles + d, sig, s_l1 + d * bf16_tile_bytes);
            }
            noc_async_read_barrier();
            cb_push_back(cb_x, head_tiles);
            cb_push_back(cb_sig, head_tiles);
        }

        if (history != 0 && chunk + 1 == chunks && head < history_heads) {
            cb_reserve_back(cb_proj, history_columns);
            const uint32_t p_l1 = get_write_ptr(cb_proj);
            for (uint32_t j = 0; j < history_columns; ++j) {
                noc_async_read_page(
                    chunk * projection_tiles + head * history_columns + j, projected, p_l1 + j * bf16_tile_bytes);
            }
            noc_async_read_barrier();
            cb_push_back(cb_proj, history_columns);
        }
    }
}
