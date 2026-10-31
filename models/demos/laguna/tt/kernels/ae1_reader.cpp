// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// Decode attention epilogue (1..32 rows; Laguna), reader: core L (row-major index) = (token row b, head_dim tile
// group) takes row b's SDPA output tiles (4b + j0 .. : [heads as rows, 128] bf16) and its per-head gate logits g[h],
// read from row s = row_in + b of the fused qkv(+gate) output ([R, Wt tiles]) at columns g_col .. g_col + H - 1 (one
// tile) and placed in column 0 of row h of a zeroed tile, the column-broadcast operand of the gate multiply.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t H = get_compile_time_arg_val(0);
    constexpr uint32_t g_tile = get_compile_time_arg_val(1);  // qkv tile holding the gate columns
    constexpr uint32_t TJ = get_compile_time_arg_val(2);      // head_dim tiles per core (4, 2 or 1)
    constexpr uint32_t grid_x = get_compile_time_arg_val(3);
    constexpr uint32_t row_in = get_compile_time_arg_val(4);  // qkv row of token row 0
    constexpr uint32_t Wt = get_compile_time_arg_val(5);      // qkv width in tiles
    constexpr uint32_t n_attn = get_compile_time_arg_val(7);  // SDPA outputs, one per 32 rows (common args 2 ..)
    constexpr auto a_args = TensorAccessorArgs<8>();
    constexpr auto x_args = TensorAccessorArgs<a_args.next_compile_time_args_offset()>();
    constexpr uint32_t T = 2048, cb_attn = 0, cb_g = 1, cb_tmp = 2;
    // more than 32 rows: token row b's SDPA output is row b % 32 of chunk b / 32's (common arg 2 + chunk)
    const auto x = TensorAccessor(x_args, get_common_arg_val<uint32_t>(1), T);

    const uint32_t L = get_absolute_logical_y() * grid_x + get_absolute_logical_x();
    const uint32_t b = L / (4 / TJ);
    const uint32_t j0 = (L % (4 / TJ)) * TJ;
    const uint32_t s = row_in + b, r = s % 32;
    const auto attn = TensorAccessor(a_args, get_common_arg_val<uint32_t>(n_attn > 1 ? 2 + b / 32 : 0), T);
    const uint32_t ab = n_attn > 1 ? b % 32 : b;
    cb_reserve_back(cb_attn, TJ);
    for (uint32_t j = 0; j < TJ; ++j) {
        noc_async_read(attn.get_noc_addr(ab * 4 + j0 + j), get_write_ptr(cb_attn) + j * T, T);
    }
    cb_reserve_back(cb_g, 1);
    const uint32_t gt = get_write_ptr(cb_g), tmp = get_write_ptr(cb_tmp);
    const uint64_t zeros = get_noc_addr(MEM_ZEROS_BASE);
    for (uint32_t off = 0; off < T; off += MEM_ZEROS_SIZE) {
        noc_async_read(zeros, gt + off, MEM_ZEROS_SIZE < T - off ? MEM_ZEROS_SIZE : T - off);
    }
    // row r of the gate tile: 64-byte aligned reads around it in face (r / 16) * 2 and the next face (a DRAM source
    // needs the same alignment), then the row's 16 + 16 values at its 32-byte offset in each
    const uint32_t roff = ((r / 16) * 2) * 512 + (r % 16) * 32;
    const uint64_t src = x.get_noc_addr((s / 32) * Wt + g_tile) + (roff & ~63u);
    noc_async_read(src, tmp, 64);
    noc_async_read(src + 512, tmp + 64, 64);
    noc_async_read_barrier();
    invalidate_l1_cache();
    volatile tt_l1_ptr uint16_t* row = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(tmp + (roff & 63u));
    volatile tt_l1_ptr uint16_t* g = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(gt);
    for (uint32_t h = 0; h < H; ++h) {
        const uint16_t v = h < 16 ? row[h] : row[32 + (h - 16)];  // face 1 row 0 starts at byte 64 of tmp
        g[(h < 16 ? 0 : 512) + (h % 16) * 16] = v;               // (row h, col 0): face 0 or face 2, 16 values per row
    }
    asm volatile("fence" ::: "memory");
    cb_push_back(cb_attn, TJ);
    cb_push_back(cb_g, 1);
}
