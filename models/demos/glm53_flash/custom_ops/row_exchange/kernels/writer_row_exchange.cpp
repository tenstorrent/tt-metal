// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Row exchange, worker writer (BRISC): cuts this chunk's pieces out of the untilized local block and hands them to
// the exchange core's send staging (gather: the T local user rows, one piece; reduce: the T rows of every other mesh
// row's users, one piece per peer), signals the exchange core, waits for its GO (every peer's packets landed in the
// receive staging), assembles the [32, CT * 32] row-major block from the peers' pieces (row r * T + t = mesh row r's
// user t) plus the local rows, and writes the finished tiles to the output.
//
// compile-time: [MODE, CB_SEND, CB_RM, CB_RM2, CB_SRC, CB_ZERO, CT, T, ROWS, NCHUNKS, PS, CPP, NP, SEM_GO, SEM_STAGED,
//                <TensorAccessorArgs out>, <TensorAccessorArgs recv>]
// runtime: [out_addr, col_tile0, chunk k, my_row, xchg_noc (x << 16 | y), recv_addr]
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t MODE = get_compile_time_arg_val(0);
    constexpr uint32_t CB_SEND = get_compile_time_arg_val(1);
    constexpr uint32_t CB_RM = get_compile_time_arg_val(2);
    constexpr uint32_t CB_RM2 = get_compile_time_arg_val(3);
    constexpr uint32_t CB_SRC = get_compile_time_arg_val(4);
    constexpr uint32_t CB_ZERO = get_compile_time_arg_val(5);
    constexpr uint32_t CT = get_compile_time_arg_val(6);
    constexpr uint32_t T = get_compile_time_arg_val(7);
    constexpr uint32_t ROWS = get_compile_time_arg_val(8);
    constexpr uint32_t NCHUNKS = get_compile_time_arg_val(9);
    constexpr uint32_t PS = get_compile_time_arg_val(10);
    constexpr uint32_t CPP = get_compile_time_arg_val(11);
    constexpr uint32_t NP = get_compile_time_arg_val(12);
    constexpr uint32_t SEM_GO = get_compile_time_arg_val(13);
    constexpr uint32_t SEM_STAGED = get_compile_time_arg_val(14);
    constexpr auto out_args = TensorAccessorArgs<15>();
    constexpr auto recv_args = TensorAccessorArgs<out_args.next_compile_time_args_offset()>();
    constexpr uint32_t TILE_BYTES = 2048;
    constexpr uint32_t ROW_B = CT * 64;      // one row of the chunk (bf16)
    constexpr uint32_t PIECE = T * ROW_B;    // one mesh row's users
    constexpr uint32_t SLOT = NCHUNKS * PS;  // one source row's pieces in the staging
    constexpr uint32_t PKT = CPP * PS;
    static_assert(PIECE <= PS, "piece larger than the staging slot");
    static_assert(ROWS * T <= 32, "the exchange assembles one tile row");

    const uint32_t out_addr = get_arg_val<uint32_t>(0);
    const uint32_t col0 = get_arg_val<uint32_t>(1);
    const uint32_t k = get_arg_val<uint32_t>(2);
    const uint32_t my_row = get_arg_val<uint32_t>(3);
    const uint32_t xchg_noc = get_arg_val<uint32_t>(4);
    const uint32_t recv_addr = get_arg_val<uint32_t>(5);
    const uint32_t xx = xchg_noc >> 16, xy = xchg_noc & 0xffff;

    const auto s_out = TensorAccessor(out_args, out_addr, TILE_BYTES);
    const auto s_recv = TensorAccessor(recv_args, recv_addr, PKT);
    const uint32_t send_base = get_write_ptr(CB_SEND);  // the same address on the exchange core (first CB everywhere)
    volatile tt_l1_ptr uint32_t* go = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SEM_GO));
    const uint64_t staged_noc = get_noc_addr(xx, xy, get_semaphore(SEM_STAGED));

    // 1. the pieces -> the exchange core's send staging
    cb_wait_front(CB_RM, CT);
    const uint32_t rm = get_read_ptr(CB_RM);
    if constexpr (MODE == 0) {
        noc_async_write(rm, get_noc_addr(xx, xy, send_base + k * PS), PIECE);
    } else {
        for (uint32_t r = 0; r < ROWS; ++r) {
            if (r != my_row) {
                noc_async_write(rm + r * PIECE, get_noc_addr(xx, xy, send_base + r * SLOT + k * PS), PIECE);
            }
        }
    }
    noc_async_write_barrier();
    noc_semaphore_inc(staged_noc, 1);

    // 2. wait for every peer's pieces, assemble the block
    noc_semaphore_wait(go, 1);
    noc_semaphore_set(go, 0);
    cb_reserve_back(CB_RM2, CT);
    const uint32_t rm2 = get_write_ptr(CB_RM2);
    for (uint32_t r = 0; r < ROWS; ++r) {
        const uint32_t dst = rm2 + r * PIECE;
        if (r == my_row) {
            const uint32_t src = (MODE == 0) ? rm : rm + my_row * PIECE;
            noc_async_read(get_noc_addr(src), dst, PIECE);
        } else {
            noc_async_read(s_recv.get_noc_addr(r * NP + k / CPP) + (k % CPP) * PS, dst, PIECE);
        }
    }
    if constexpr (MODE == 0 && ROWS * T < 32) {
        cb_wait_front(CB_ZERO, 1);
        const uint64_t zero = get_noc_addr(get_read_ptr(CB_ZERO));
        for (uint32_t i = ROWS * T; i < 32; ++i) {
            noc_async_read(zero, rm2 + i * ROW_B, ROW_B);
        }
    }
    noc_async_read_barrier();
    cb_pop_front(CB_RM, CT);
    cb_push_back(CB_RM2, CT);

    // 3. the finished tiles -> the output
    cb_wait_front(CB_SRC, CT);
    const uint32_t src = get_read_ptr(CB_SRC);
    for (uint32_t j = 0; j < CT; ++j) {
        noc_async_write(src + j * TILE_BYTES, s_out.get_noc_addr(col0 + j), TILE_BYTES);
    }
    noc_async_write_barrier();
    cb_pop_front(CB_SRC, CT);
}
