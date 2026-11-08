// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// BRISC (NOC 1): the all-reduced compact [1, 1, 32, hidden / 8] bf16 tiles (moe_reduce_compute.cpp) back into rows
// 0-3 of a [1, 1, 32, hidden] bf16 tile tensor, zeros in rows 4-31. Rows 4c..4c+3 of compact tile tc hold rows 0-3
// of output tile column c * compact_tiles + tc (moe_down_writer.cpp). This core writes every row block of its
// compact tiles [first, first + tiles).
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "moe_stream_common.hpp"

void kernel_main() {
    constexpr uint32_t cb_sum = get_compile_time_arg_val(0);
    constexpr uint32_t cb_zero = get_compile_time_arg_val(1);
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

    // Scratch: tail_bytes of zeros.
    cb_reserve_back(cb_zero, 1);
    const uint32_t zeros = get_write_ptr(cb_zero);
    volatile tt_l1_ptr uint32_t* z = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(zeros);
    for (uint32_t w = 0; w < tail_bytes / 4; w++) {
        z[w] = 0;
    }

    const InterleavedAddrGen<true> out_gen = {.bank_base_address = out_addr, .page_size = tile_bytes};
    for (uint32_t i = 0; i < tiles; i++) {
        cb_wait_front(cb_sum, i + 1);
        const uint32_t l1 = get_read_ptr(cb_sum) + i * tile_bytes;
        const uint32_t tc = first + i;
        for (uint32_t c = 0; c < row_blocks; c++) {
            const uint32_t page = c * compact_tiles + tc;
            const uint32_t src = l1 + (c / blocks_per_face) * 2 * face_bytes + (c % blocks_per_face) * rows_bytes;
            noc_async_write(src, out_gen.get_noc_addr(page, 0), rows_bytes);
            if constexpr (gap_bytes > 0) {  // 16 user rows fill face 0
                noc_async_write(zeros, out_gen.get_noc_addr(page, rows_bytes), gap_bytes);
            }
            noc_async_write(src + face_bytes, out_gen.get_noc_addr(page, face_bytes), rows_bytes);
            noc_async_write(zeros, out_gen.get_noc_addr(page, face_bytes + rows_bytes), tail_bytes);
        }
    }
    noc_async_write_barrier();
    cb_pop_front(cb_sum, tiles);
}
