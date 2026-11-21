// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// Decode attention prologue (1..96 rows), reader (Laguna). Role 0 = q, role 1 = k (+ v); each core handles rpc
// consecutive token rows b (and, for q, one part of their heads). The fused qkv matmul output is [R, W] bf16 TILE (Wt tiles wide);
// this call's rows are row0 .. row0 + B - 1 (the batched verify: one 32-row chunk of all its rows). Head h's 128 values
// of row s = row0 + b are row s % 32 of tiles (s / 32) * Wt + off + 4h .. + 3 (16 values in face ((s % 32) / 16) * 2
// at byte (s % 16) * 32, 16 in the next face). They become row h of a [32, 128] head-major block of 4 tiles (rows 0-15 in faces 0/1, rows 16-31
// in faces 2/3), zero elsewhere. Also pushes the norm weight row (4 tiles), row b's cos / sin (tiles b * rd / 32 ..,
// value row 0; the tables are [1, B, 1, rd]) and a reduce scaler tile (all 1 / head_dim).

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

constexpr uint32_t cb_x = 0, cb_w = 1, cb_cos = 2, cb_sin = 3, cb_scaler = 4, cb_v = 5;

template <typename Acc>
void gather_heads(const Acc& src, uint32_t off_tiles, uint32_t h0, uint32_t nheads, uint32_t dst, uint32_t row) {
    // row: the source row's index within its tile row; off_tiles includes that tile row's first tile
    const uint32_t src_off = ((row / 16) * 2) * 512 + (row % 16) * 32;
    const uint64_t zeros = get_noc_addr(MEM_ZEROS_BASE);
    for (uint32_t b = 0; b < 4 * 2048; b += MEM_ZEROS_SIZE) {
        noc_async_read(zeros, dst + b, MEM_ZEROS_SIZE < 4 * 2048 - b ? MEM_ZEROS_SIZE : 4 * 2048 - b);
    }
    noc_async_read_barrier();
    for (uint32_t h = h0; h < h0 + nheads; ++h) {
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
    constexpr uint32_t total_heads = get_compile_time_arg_val(7);  // q heads in all (role 0); nheads = per part
    // this role's cores are the row-major core indices base .. base + rows * parts - 1: row = L / parts, part = L % parts
    constexpr uint32_t grid_x = get_compile_time_arg_val(8);
    constexpr uint32_t base = get_compile_time_arg_val(9);
    constexpr uint32_t parts = get_compile_time_arg_val(10);
    constexpr uint32_t row0 = get_compile_time_arg_val(11);  // first source row of this call
    constexpr uint32_t Wt = get_compile_time_arg_val(12);    // qkv width in tiles
    constexpr uint32_t rpc = get_compile_time_arg_val(13);   // rows per core (consecutive)
    constexpr auto qkv_args = TensorAccessorArgs<14>();
    constexpr auto w_args = TensorAccessorArgs<qkv_args.next_compile_time_args_offset()>();
    constexpr auto cos_args = TensorAccessorArgs<w_args.next_compile_time_args_offset()>();
    constexpr auto sin_args = TensorAccessorArgs<cos_args.next_compile_time_args_offset()>();
    constexpr auto sc_args = TensorAccessorArgs<sin_args.next_compile_time_args_offset()>();
    const auto qkv = TensorAccessor(qkv_args, get_common_arg_val<uint32_t>(0), 2048);
    const auto wt = TensorAccessor(w_args, get_common_arg_val<uint32_t>(4), w_page);
    const auto cs = TensorAccessor(cos_args, get_common_arg_val<uint32_t>(1), cs_page);
    const auto sn = TensorAccessor(sin_args, get_common_arg_val<uint32_t>(2), cs_page);
    const auto sc = TensorAccessor(sc_args, get_common_arg_val<uint32_t>(3), 2048);
    const uint32_t L = get_absolute_logical_y() * grid_x + get_absolute_logical_x() - base;
    const uint32_t row_first = (L / parts) * rpc;
    // q split over parts: this core's heads [h0, h0 + n) keep their rows of the block
    const uint32_t part = role == 0 ? L % parts : 0;
    const uint32_t h0 = part * nheads;
    const uint32_t n = role == 0 ? (h0 >= total_heads ? 0 : (total_heads - h0 < nheads ? total_heads - h0 : nheads)) : nheads;

    // once: the norm weight row and the reduce scaler (compute keeps them for every row)
    cb_reserve_back(cb_w, 4);
    cb_reserve_back(cb_scaler, 1);
    for (uint32_t j = 0; j < 4; ++j) {
        noc_async_read_page(j, wt, get_write_ptr(cb_w) + j * w_page);
    }
    noc_async_read_page(0, sc, get_write_ptr(cb_scaler));
    for (uint32_t i = 0; i < rpc; ++i) {
        const uint32_t row = row_first + i;
        cb_reserve_back(cb_cos, rd_tiles);
        cb_reserve_back(cb_sin, rd_tiles);
        for (uint32_t j = 0; j < rd_tiles; ++j) {
            noc_async_read_page(row * rd_tiles + j, cs, get_write_ptr(cb_cos) + j * cs_page);
            noc_async_read_page(row * rd_tiles + j, sn, get_write_ptr(cb_sin) + j * cs_page);
        }
        cb_reserve_back(cb_x, 4);
        const uint32_t s = row0 + row, s_tiles = (s / 32) * Wt;
        gather_heads(qkv, s_tiles + off_tiles, h0, n, get_write_ptr(cb_x), s % 32);  // its barriers cover the reads above
        if (i == 0) {
            cb_push_back(cb_w, 4);
            cb_push_back(cb_scaler, 1);
        }
        cb_push_back(cb_cos, rd_tiles);
        cb_push_back(cb_sin, rd_tiles);
        cb_push_back(cb_x, 4);
        if constexpr (role == 1) {
            cb_reserve_back(cb_v, 4);
            gather_heads(qkv, s_tiles + v_off_tiles, 0, nheads, get_write_ptr(cb_v), s % 32);
            cb_push_back(cb_v, 4);
        }
    }
}
