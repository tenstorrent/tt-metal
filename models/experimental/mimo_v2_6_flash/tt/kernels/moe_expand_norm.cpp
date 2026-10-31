// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// BRISC (NOC 1) of the fused reduce + residual + RMSNorm: writes the residual sum back into rows 0-3 of ``out`` with
// the residual's rows 4-31 (as moe_expand_residual.cpp), sends rows 0-3 of this core's partial sum of squares into
// rows 4 * index.. of every core's gather tile and increments their semaphores, then writes the normalized tiles into
// rows 0-3 of ``norm_out`` (a persistent buffer whose rows 4-31 stay zero).
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "moe_stream_common.hpp"

void kernel_main() {
    constexpr uint32_t cb_sum = get_compile_time_arg_val(0);
    constexpr uint32_t cb_x = get_compile_time_arg_val(1);
    constexpr uint32_t tiles = get_compile_time_arg_val(2);
    constexpr uint32_t compact_tiles = get_compile_time_arg_val(3);
    constexpr uint32_t row_blocks = get_compile_time_arg_val(4);
    constexpr uint32_t cb_part = get_compile_time_arg_val(5);
    constexpr uint32_t cb_gather = get_compile_time_arg_val(6);
    constexpr uint32_t cb_h = get_compile_time_arg_val(7);
    constexpr uint32_t cores = get_compile_time_arg_val(8);
    constexpr uint32_t sem_id = get_compile_time_arg_val(9);
    constexpr uint32_t rows = get_compile_time_arg_val(10);  // user rows per block: 4, 8 or 16
    constexpr uint32_t tile_bytes = 2048;
    constexpr uint32_t fp32_tile_bytes = 4096;
    constexpr uint32_t face_bytes = mimo_moe::kBf16FaceBytes;
    constexpr uint32_t rows_bytes = rows * mimo_moe::kFaceDim * 2;  // the user rows of one face
    constexpr uint32_t blocks_per_face = face_bytes / rows_bytes;
    constexpr uint32_t gap_bytes = face_bytes - rows_bytes;                // the other rows of face 0
    constexpr uint32_t tail_bytes = tile_bytes - face_bytes - rows_bytes;  // the other rows of face 1, faces 2-3

    const uint32_t out_addr = get_arg_val<uint32_t>(0);
    const uint32_t first = get_arg_val<uint32_t>(1);
    const uint32_t res_addr = get_arg_val<uint32_t>(2);
    const uint32_t norm_addr = get_arg_val<uint32_t>(3);
    const uint32_t index = get_arg_val<uint32_t>(4);
    // args 5.. : NoC x, y of every core of the program, in core-index order

    // The residual tiles of every output tile column this core writes (their rows 4-31 pass through).
    const InterleavedAddrGen<true> res_gen = {.bank_base_address = res_addr, .page_size = tile_bytes};
    cb_reserve_back(cb_x, tiles * row_blocks);
    const uint32_t x_l1 = get_write_ptr(cb_x);
    for (uint32_t i = 0; i < tiles; i++) {
        for (uint32_t c = 0; c < row_blocks; c++) {
            const uint32_t page = c * compact_tiles + first + i;
            noc_async_read(res_gen.get_noc_addr(page, 0), x_l1 + (i * row_blocks + c) * tile_bytes, tile_bytes);
        }
    }
    noc_async_read_barrier();

    const InterleavedAddrGen<true> out_gen = {.bank_base_address = out_addr, .page_size = tile_bytes};
    for (uint32_t i = 0; i < tiles; i++) {
        cb_wait_front(cb_sum, i + 1);
        const uint32_t l1 = get_read_ptr(cb_sum) + i * tile_bytes;
        const uint32_t tc = first + i;
        for (uint32_t c = 0; c < row_blocks; c++) {
            const uint32_t page = c * compact_tiles + tc;
            const uint32_t src = l1 + (c / blocks_per_face) * 2 * face_bytes + (c % blocks_per_face) * rows_bytes;
            const uint32_t xt = x_l1 + (i * row_blocks + c) * tile_bytes;
            noc_async_write(src, out_gen.get_noc_addr(page, 0), rows_bytes);
            if constexpr (gap_bytes > 0) {  // 16 user rows fill face 0
                noc_async_write(xt + rows_bytes, out_gen.get_noc_addr(page, rows_bytes), gap_bytes);
            }
            noc_async_write(src + face_bytes, out_gen.get_noc_addr(page, face_bytes), rows_bytes);
            noc_async_write(
                xt + face_bytes + rows_bytes, out_gen.get_noc_addr(page, face_bytes + rows_bytes), tail_bytes);
        }
    }

    // The user rows of the partial (fp32: rows x 16 x 4 B in faces 0 and 1) go to rows rows * index.. of the gather
    // tiles on every core (32 / rows cores per tile; tile index / (32 / rows), rows (rows * index) % 32 of it), then
    // one increment each.
    constexpr uint32_t fp32_face_bytes = 1024;
    constexpr uint32_t fp32_tile_bytes_g = 4096;
    constexpr uint32_t part_rows_bytes = rows * mimo_moe::kFaceDim * 4;
    constexpr uint32_t cores_per_tile = 32 / rows;
    const uint32_t sem_addr = get_semaphore(sem_id);
    const uint32_t row0 = rows * (index % cores_per_tile);  // within its gather tile
    const uint32_t slot = get_write_ptr(cb_gather) + (index / cores_per_tile) * fp32_tile_bytes_g +
                          (row0 / 16) * 2 * fp32_face_bytes + (row0 % 16) * mimo_moe::kFaceDim * 4;
    cb_wait_front(cb_part, 1);
    const uint32_t part_l1 = get_read_ptr(cb_part);
    for (uint32_t p = 0; p < cores; p++) {
        const uint32_t px = get_arg_val<uint32_t>(5 + 2 * p);
        const uint32_t py = get_arg_val<uint32_t>(6 + 2 * p);
        noc_async_write(part_l1, get_noc_addr(px, py, slot), part_rows_bytes);
        noc_async_write(part_l1 + fp32_face_bytes, get_noc_addr(px, py, slot + fp32_face_bytes), part_rows_bytes);
    }
    noc_async_write_barrier();
    for (uint32_t p = 0; p < cores; p++) {
        const uint32_t px = get_arg_val<uint32_t>(5 + 2 * p);
        const uint32_t py = get_arg_val<uint32_t>(6 + 2 * p);
        noc_semaphore_inc(get_noc_addr(px, py, sem_addr), 1);
    }
    cb_pop_front(cb_part, 1);

    const InterleavedAddrGen<true> norm_gen = {.bank_base_address = norm_addr, .page_size = tile_bytes};
    cb_wait_front(cb_h, tiles);
    const uint32_t h_l1 = get_read_ptr(cb_h);
    for (uint32_t i = 0; i < tiles; i++) {
        const uint32_t tc = first + i;
        for (uint32_t c = 0; c < row_blocks; c++) {
            const uint32_t page = c * compact_tiles + tc;
            const uint32_t src =
                h_l1 + i * tile_bytes + (c / blocks_per_face) * 2 * face_bytes + (c % blocks_per_face) * rows_bytes;
            noc_async_write(src, norm_gen.get_noc_addr(page, 0), rows_bytes);
            noc_async_write(src + face_bytes, norm_gen.get_noc_addr(page, face_bytes), rows_bytes);
        }
    }
    noc_async_write_barrier();
    cb_pop_front(cb_h, tiles);
    cb_pop_front(cb_sum, tiles);
    noc_async_atomic_barrier();
}
