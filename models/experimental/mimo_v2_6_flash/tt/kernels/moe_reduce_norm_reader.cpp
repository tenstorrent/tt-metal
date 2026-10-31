// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// NCRISC (NOC 0) of the fused reduce + residual + RMSNorm (ReduceExpand with ``norm``): everything
// moe_reduce_residual_reader.cpp reads, plus this core's compact gamma tiles and the two fp32 constant tiles of the
// norm's row sums (J: all ones; M: M[r][c] = 1 when r % 4 == c % 4). Then it waits until every core of the program
// has written its partial sum of squares into this core's cb_gather tile (semaphore count == cores) and hands it to
// compute.
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
    constexpr uint32_t cb_gamma = get_compile_time_arg_val(6);
    constexpr uint32_t cb_const = get_compile_time_arg_val(7);
    constexpr uint32_t cb_gather = get_compile_time_arg_val(8);
    constexpr uint32_t cores = get_compile_time_arg_val(9);
    constexpr uint32_t sem_id = get_compile_time_arg_val(10);
    constexpr uint32_t rows = get_compile_time_arg_val(11);          // user rows per block: 4, 8 or 16
    constexpr uint32_t gather_tiles = get_compile_time_arg_val(12);  // tiles of every core's partial rows
    constexpr uint32_t tile_bytes = 2048;
    constexpr uint32_t fp32_tile_bytes = 4096;
    constexpr uint32_t face_bytes = mimo_moe::kBf16FaceBytes;
    constexpr uint32_t rows_bytes = rows * mimo_moe::kFaceDim * 2;  // the user rows of one face
    constexpr uint32_t blocks_per_face = face_bytes / rows_bytes;

    const uint32_t in_addr = get_arg_val<uint32_t>(0);
    const uint32_t first = get_arg_val<uint32_t>(1);
    const uint32_t res_addr = get_arg_val<uint32_t>(2);
    const uint32_t gamma_addr = get_arg_val<uint32_t>(3);
    const uint32_t const_addr = get_arg_val<uint32_t>(4);

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
    const InterleavedAddrGen<true> gamma_gen = {.bank_base_address = gamma_addr, .page_size = tile_bytes};
    cb_reserve_back(cb_gamma, tiles);
    const uint32_t gamma_l1 = get_write_ptr(cb_gamma);
    for (uint32_t i = 0; i < tiles; i++) {
        noc_async_read(gamma_gen.get_noc_addr(first + i, 0), gamma_l1 + i * tile_bytes, tile_bytes);
    }
    const InterleavedAddrGen<true> const_gen = {.bank_base_address = const_addr, .page_size = fp32_tile_bytes};
    cb_reserve_back(cb_const, 2);
    const uint32_t const_l1 = get_write_ptr(cb_const);
    for (uint32_t i = 0; i < 2; i++) {
        noc_async_read(const_gen.get_noc_addr(i, 0), const_l1 + i * fp32_tile_bytes, fp32_tile_bytes);
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
            cb_push_back(cb_gamma, tiles);
            cb_push_back(cb_const, 2);
        }
    }

    // Every core's writer puts the user rows of its partial into rows rows * index.. of this core's cb_gather tiles
    // (32 / rows cores per tile), then increments this core's semaphore.
    cb_reserve_back(cb_gather, gather_tiles);
    volatile tt_l1_ptr uint32_t* sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_id));
    noc_semaphore_wait(sem, cores);
    noc_semaphore_set(sem, 0);
    cb_push_back(cb_gather, gather_tiles);
}
