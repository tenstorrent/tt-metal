// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Region scatter of the regions MoE (tt-blaze G53F-0193): buf[pos[t, j]] = x[t] for every LOCAL pick (pos != dummy),
// read straight from pos's fp32 TILE layout. Replaces pos -> uint32 ROW_MAJOR (typecast + untilize), the slot -> token
// scatter and the ttnn.embedding gather of every max_buf row: only the local picks' rows are written (the expert kernel
// reads region rows only, row by row, so the rows it uses are identical; padding / dummy rows are never combined).
// One data-movement kernel: per 32-token block, read the block's slot tile, then per batch of tokens read the x rows of
// the tokens with a local pick and write each to its slots.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"

constexpr uint32_t K = get_compile_time_arg_val(0);
constexpr uint32_t row_bytes = get_compile_time_arg_val(1);
constexpr uint32_t dummy_slot = get_compile_time_arg_val(2);
constexpr uint32_t cb_rows_id = get_compile_time_arg_val(3);
constexpr uint32_t cb_pos_id = get_compile_time_arg_val(4);
constexpr uint32_t NB = get_compile_time_arg_val(5);  // tokens staged per batch (divides 32)
constexpr uint32_t pos_tile_bytes = get_compile_time_arg_val(6);
constexpr auto x_args = TensorAccessorArgs<7>();
constexpr auto pos_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
constexpr auto buf_args = TensorAccessorArgs<pos_args.next_compile_time_args_offset()>();
constexpr uint32_t TILE_ROWS = 32;

// the fp32 bit pattern of a non-negative integer value below 2^24 -> that integer (exact; no FPU on the data movers)
FORCE_INLINE uint32_t f32_to_u32(uint32_t bits) {
    const uint32_t e = (bits >> 23) & 0xFFu;
    if (e < 127u) {
        return 0u;
    }
    const uint32_t m = (bits & 0x7FFFFFu) | 0x800000u;
    return e >= 150u ? (m << (e - 150u)) : (m >> (150u - e));
}

// slot of (row r, pick j < 16) in a 32x32 fp32 tile: the left faces, 0 for rows 0-15 and 2 for rows 16-31
FORCE_INLINE uint32_t slot_at(const CoreLocalMem<volatile uint32_t>& tile, uint32_t r, uint32_t j) {
    const uint32_t face = r < 16u ? 0u : 2u;
    return f32_to_u32(tile[face * 256u + (r & 15u) * 16u + j]);
}

void kernel_main() {
    const uint32_t x_addr = get_arg_val<uint32_t>(0);
    const uint32_t pos_addr = get_arg_val<uint32_t>(1);
    const uint32_t buf_addr = get_arg_val<uint32_t>(2);
    const uint32_t block0 = get_arg_val<uint32_t>(3);
    const uint32_t nblocks = get_arg_val<uint32_t>(4);

    const auto s_x = TensorAccessor(x_args, x_addr, row_bytes);
    const auto s_pos = TensorAccessor(pos_args, pos_addr, pos_tile_bytes);
    const auto s_buf = TensorAccessor(buf_args, buf_addr, row_bytes);

    Noc noc;
    CircularBuffer cb_rows(cb_rows_id), cb_pos(cb_pos_id);
    // private scratch, reserved once and never pushed: its read and write pointers stay at its base
    cb_rows.reserve_back(NB);
    cb_pos.reserve_back(1);
    const uint32_t pos_base = get_write_ptr(cb_pos_id);
    const CoreLocalMem<volatile uint32_t> tile(pos_base);

    for (uint32_t b = block0; b < block0 + nblocks; ++b) {
        noc.async_read(s_pos, CoreLocalMem<uint32_t>(pos_base), pos_tile_bytes, {.page_id = b}, {});
        noc.async_read_barrier();
        for (uint32_t t0 = 0; t0 < TILE_ROWS; t0 += NB) {
            bool any = false;
            for (uint32_t i = 0; i < NB; ++i) {
                bool local = false;
                for (uint32_t j = 0; j < K; ++j) {
                    local = local || slot_at(tile, t0 + i, j) != dummy_slot;
                }
                if (local) {
                    noc.async_read(
                        s_x, cb_rows, row_bytes, {.page_id = b * TILE_ROWS + t0 + i}, {.offset_bytes = i * row_bytes});
                    any = true;
                }
            }
            if (!any) {
                continue;
            }
            noc.async_read_barrier();
            for (uint32_t i = 0; i < NB; ++i) {
                for (uint32_t j = 0; j < K; ++j) {
                    const uint32_t slot = slot_at(tile, t0 + i, j);
                    if (slot != dummy_slot) {
                        noc.async_write(cb_rows, s_buf, row_bytes, {.offset_bytes = i * row_bytes}, {.page_id = slot});
                    }
                }
            }
            noc.async_write_barrier();
        }
    }
}
