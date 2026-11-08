// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// NCRISC (NOC 0): rows 0-3 of the residual's tile columns packed like this core's compact tiles (rows 4c..4c+3 of
// compact tile tc = rows 0-3 of tile column c * compact_tiles + tc), then the chips' copies of the compact tiles from
// the row all-gather ([1, 1, 32, chips * hidden / 8]: chip c's compact tile tc is page c * compact_tiles + tc), one
// tile's chips copies per push, for moe_reduce_residual_compute.cpp.
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "moe_stream_common.hpp"

void kernel_main() {
    constexpr uint32_t cb_in = get_compile_time_arg_val(0);
    constexpr uint32_t cb_res = get_compile_time_arg_val(1);
    constexpr uint32_t tiles = get_compile_time_arg_val(2);
    constexpr uint32_t compact_tiles = get_compile_time_arg_val(3);
    constexpr uint32_t chips = get_compile_time_arg_val(4);
    constexpr uint32_t row_blocks = get_compile_time_arg_val(5);
    constexpr uint32_t rows = get_compile_time_arg_val(6);  // user rows per block: 4, 8 or 16
    constexpr uint32_t tile_bytes = 2048;
    constexpr uint32_t face_bytes = mimo_moe::kBf16FaceBytes;
    constexpr uint32_t rows_bytes = rows * mimo_moe::kFaceDim * 2;  // the user rows of one face
    constexpr uint32_t blocks_per_face = face_bytes / rows_bytes;

    const uint32_t in_addr = get_arg_val<uint32_t>(0);
    const uint32_t first = get_arg_val<uint32_t>(1);
    const uint32_t res_addr = get_arg_val<uint32_t>(2);

    // Issued first; the first tile's barrier below covers them.
    const InterleavedAddrGen<true> res_gen = {.bank_base_address = res_addr, .page_size = tile_bytes};
    cb_reserve_back(cb_res, tiles);
    const uint32_t res_l1 = get_write_ptr(cb_res);
    for (uint32_t i = 0; i < tiles; i++) {
        for (uint32_t c = 0; c < row_blocks; c++) {
            const uint32_t page = c * compact_tiles + first + i;
            const uint32_t dst =
                res_l1 + i * tile_bytes + (c / blocks_per_face) * 2 * face_bytes + (c % blocks_per_face) * rows_bytes;
            noc_async_read(res_gen.get_noc_addr(page, 0), dst, rows_bytes);
            noc_async_read(res_gen.get_noc_addr(page, face_bytes), dst + face_bytes, rows_bytes);
        }
    }

    const InterleavedAddrGen<true> in_gen = {.bank_base_address = in_addr, .page_size = tile_bytes};
    for (uint32_t i = 0; i < tiles; i++) {
        cb_reserve_back(cb_in, chips);
        const uint32_t l1 = get_write_ptr(cb_in);
        for (uint32_t c = 0; c < chips; c++) {
            noc_async_read(in_gen.get_noc_addr(c * compact_tiles + first + i, 0), l1 + c * tile_bytes, tile_bytes);
        }
        noc_async_read_barrier();
        cb_push_back(cb_in, chips);
        if (i == 0) {
            cb_push_back(cb_res, tiles);
        }
    }
}
