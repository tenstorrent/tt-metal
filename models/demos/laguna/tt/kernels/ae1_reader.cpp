// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// Decode attention epilogue (batch 1, or up to 8 rows; Laguna), reader: core b (logical x) takes token row b's SDPA
// output (tiles 4b .. 4b + 3: [heads as rows, 128] bf16) and its per-head gate logits g[h], read from row b of the
// fused qkv(+gate) output at columns g_col .. g_col + H - 1 (one tile) and placed in column 0 of row h of a zeroed
// tile, the column-broadcast operand of the gate multiply.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t H = get_compile_time_arg_val(0);
    constexpr uint32_t g_tile = get_compile_time_arg_val(1);  // qkv tile holding the gate columns
    constexpr auto a_args = TensorAccessorArgs<2>();
    constexpr auto x_args = TensorAccessorArgs<a_args.next_compile_time_args_offset()>();
    constexpr uint32_t T = 2048, cb_attn = 0, cb_g = 1, cb_tmp = 2;
    const auto attn = TensorAccessor(a_args, get_common_arg_val<uint32_t>(0), T);
    const auto x = TensorAccessor(x_args, get_common_arg_val<uint32_t>(1), T);

    const uint32_t b = get_absolute_logical_x();
    cb_reserve_back(cb_attn, 4);
    for (uint32_t j = 0; j < 4; ++j) {
        noc_async_read(attn.get_noc_addr(b * 4 + j), get_write_ptr(cb_attn) + j * T, T);
    }
    cb_reserve_back(cb_g, 1);
    const uint32_t gt = get_write_ptr(cb_g), tmp = get_write_ptr(cb_tmp);
    const uint64_t zeros = get_noc_addr(MEM_ZEROS_BASE);
    for (uint32_t off = 0; off < T; off += MEM_ZEROS_SIZE) {
        noc_async_read(zeros, gt + off, MEM_ZEROS_SIZE < T - off ? MEM_ZEROS_SIZE : T - off);
    }
    // row b of the gate tile: 64-byte aligned reads around it in face (b / 16) * 2 and the next face (a DRAM source
    // needs the same alignment), then the row's 16 + 16 values at its 32-byte offset in each
    const uint32_t roff = ((b / 16) * 2) * 512 + (b % 16) * 32;
    const uint64_t src = x.get_noc_addr(g_tile) + (roff & ~63u);
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
    cb_push_back(cb_attn, 4);
    cb_push_back(cb_g, 1);
}
