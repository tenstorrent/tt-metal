// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// BRISC (NOC 1): the reduced compact tiles plus residual (moe_reduce_residual_compute.cpp) back into rows 0-3 of a
// [1, 1, 32, hidden] bf16 tile tensor; rows 4-31 are the residual's, unchanged. Rows 4c..4c+3 of compact tile tc hold
// rows 0-3 of output tile column c * compact_tiles + tc. This core writes every row block of its compact tiles
// [first, first + tiles).
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "moe_stream_common.hpp"

void kernel_main() {
    constexpr uint32_t cb_sum = get_compile_time_arg_val(0);
    constexpr uint32_t cb_x = get_compile_time_arg_val(1);
    constexpr uint32_t tiles = get_compile_time_arg_val(2);
    constexpr uint32_t compact_tiles = get_compile_time_arg_val(3);
    constexpr uint32_t row_blocks = get_compile_time_arg_val(4);
    constexpr uint32_t rows = get_compile_time_arg_val(5);  // user rows per block: 4, 8 or 16
    constexpr uint32_t tile_bytes = 2048;
    constexpr uint32_t face_bytes = mimo_moe::kBf16FaceBytes;
    constexpr uint32_t rows_bytes = rows * mimo_moe::kFaceDim * 2;  // the user rows of one face
    constexpr uint32_t blocks_per_face = face_bytes / rows_bytes;
    constexpr uint32_t gap_bytes = face_bytes - rows_bytes;                // the other rows of face 0
    constexpr uint32_t tail_bytes = tile_bytes - face_bytes - rows_bytes;  // the other rows of face 1, faces 2-3

    const uint32_t out_addr = get_arg_val<uint32_t>(0);
    const uint32_t first = get_arg_val<uint32_t>(1);
    const uint32_t res_addr = get_arg_val<uint32_t>(2);

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
    noc_async_write_barrier();
    cb_pop_front(cb_sum, tiles);
}
