// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// Decode attention prologue (batch 1, or up to 8 rows), reader (Laguna). Role 0 = q, role 1 = k (+ v); the core's
// logical x is its token row b. The fused qkv matmul output is [32, W] bf16 TILE with rows 0..B-1 real; head h's
// 128 values are row b of tiles off + 4h .. off + 4h + 3 (16 values in face (b / 16) * 2 at byte (b % 16) * 32, 16 in
// the next face). They become row h of a [32, 128] head-major block of 4 tiles (rows 0-15 in faces 0/1, rows 16-31
// in faces 2/3), zero elsewhere. Also pushes the norm weight row (4 tiles), row b's cos / sin (tiles b * rd / 32 ..,
// value row 0; the tables are [1, B, 1, rd]) and a reduce scaler tile (all 1 / head_dim).

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

constexpr uint32_t cb_x = 0, cb_w = 1, cb_cos = 2, cb_sin = 3, cb_scaler = 4, cb_v = 5;

template <typename Acc>
void gather_heads(const Acc& src, uint32_t off_tiles, uint32_t nheads, uint32_t dst, uint32_t row) {
    const uint32_t src_off = ((row / 16) * 2) * 512 + (row % 16) * 32;
    const uint64_t zeros = get_noc_addr(MEM_ZEROS_BASE);
    for (uint32_t b = 0; b < 4 * 2048; b += MEM_ZEROS_SIZE) {
        noc_async_read(zeros, dst + b, MEM_ZEROS_SIZE < 4 * 2048 - b ? MEM_ZEROS_SIZE : 4 * 2048 - b);
    }
    noc_async_read_barrier();
    for (uint32_t h = 0; h < nheads; ++h) {
        const uint32_t fbase = h < 16 ? 0 : 2;  // faces of row h
        const uint32_t roff = (h % 16) * 32;
        for (uint32_t j = 0; j < 4; ++j) {
            const uint64_t a = src.get_noc_addr(off_tiles + h * 4 + j) + src_off;
            noc_async_read(a, dst + j * 2048 + fbase * 512 + roff, 32);
            noc_async_read(a + 512, dst + j * 2048 + (fbase + 1) * 512 + roff, 32);
        }
    }
    noc_async_read_barrier();
}

void kernel_main() {
    constexpr uint32_t role = get_compile_time_arg_val(0);
    constexpr uint32_t nheads = get_compile_time_arg_val(1);
    constexpr uint32_t off_tiles = get_compile_time_arg_val(2);
    constexpr uint32_t v_off_tiles = get_compile_time_arg_val(3);
    constexpr uint32_t rd_tiles = get_compile_time_arg_val(4);
    constexpr uint32_t w_page = get_compile_time_arg_val(5);
    constexpr uint32_t cs_page = get_compile_time_arg_val(6);
    constexpr auto qkv_args = TensorAccessorArgs<7>();
    constexpr auto w_args = TensorAccessorArgs<qkv_args.next_compile_time_args_offset()>();
    constexpr auto cos_args = TensorAccessorArgs<w_args.next_compile_time_args_offset()>();
    constexpr auto sin_args = TensorAccessorArgs<cos_args.next_compile_time_args_offset()>();
    constexpr auto sc_args = TensorAccessorArgs<sin_args.next_compile_time_args_offset()>();
    const auto qkv = TensorAccessor(qkv_args, get_common_arg_val<uint32_t>(0), 2048);
    const auto wt = TensorAccessor(w_args, get_common_arg_val<uint32_t>(4), w_page);
    const auto cs = TensorAccessor(cos_args, get_common_arg_val<uint32_t>(1), cs_page);
    const auto sn = TensorAccessor(sin_args, get_common_arg_val<uint32_t>(2), cs_page);
    const auto sc = TensorAccessor(sc_args, get_common_arg_val<uint32_t>(3), 2048);
    const uint32_t row = get_absolute_logical_x();

    cb_reserve_back(cb_w, 4);
    cb_reserve_back(cb_cos, rd_tiles);
    cb_reserve_back(cb_sin, rd_tiles);
    cb_reserve_back(cb_scaler, 1);
    for (uint32_t j = 0; j < 4; ++j) {
        noc_async_read_page(j, wt, get_write_ptr(cb_w) + j * w_page);
    }
    for (uint32_t j = 0; j < rd_tiles; ++j) {
        noc_async_read_page(row * rd_tiles + j, cs, get_write_ptr(cb_cos) + j * cs_page);
        noc_async_read_page(row * rd_tiles + j, sn, get_write_ptr(cb_sin) + j * cs_page);
    }
    noc_async_read_page(0, sc, get_write_ptr(cb_scaler));
    cb_reserve_back(cb_x, 4);
    gather_heads(qkv, off_tiles, nheads, get_write_ptr(cb_x), row);  // its barriers also cover the reads above
    cb_push_back(cb_w, 4);
    cb_push_back(cb_cos, rd_tiles);
    cb_push_back(cb_sin, rd_tiles);
    cb_push_back(cb_scaler, 1);
    cb_push_back(cb_x, 4);
    if constexpr (role == 1) {
        cb_reserve_back(cb_v, 4);
        gather_heads(qkv, v_off_tiles, nheads, get_write_ptr(cb_v), row);
        cb_push_back(cb_v, 4);
    }
}
