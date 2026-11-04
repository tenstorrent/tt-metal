// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Persistent MoE program, exchange worker writer (BRISC): phase A = the gather writer flow of
// custom_ops/row_exchange (stage the T local rows for every peer, GO, assemble the sources' rows, write the gathered
// tiles), then a second STAGED increment telling the exchange core the tiles are written (it releases the experts);
// phase B (workers with a reduce chunk) = the reduce writer flow (stage every peer's users' rows, GO, assemble the
// sources' pieces for this row's users, write the summed tiles).
//
// compile-time: [CB_SEND_A, CB_SEND_B, CB_RM, CB_RM2, CB_T, CB_OUT, CB_ZERO, CT_A, CT_B, T, NSRC, NCHUNKS_A,
//                NCHUNKS_B, PS, CPP, NP_A, NP_B, SEM_GO, SEM_STAGED, HAS_B,
//                <TensorAccessorArgs outA>, <TensorAccessorArgs recvA>, <TensorAccessorArgs outB>, <TensorAccessorArgs
//                recvB>]
// runtime: [outA_addr, colA0, kA, my_idx, xchg_noc, recvA_addr, own_noc, outB_addr, colB0, kB, recvB_addr]
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t CB_SEND_A = get_compile_time_arg_val(0);
    constexpr uint32_t CB_SEND_B = get_compile_time_arg_val(1);
    constexpr uint32_t CB_RM = get_compile_time_arg_val(2);
    constexpr uint32_t CB_RM2 = get_compile_time_arg_val(3);
    constexpr uint32_t CB_T = get_compile_time_arg_val(4);
    constexpr uint32_t CB_OUT = get_compile_time_arg_val(5);
    constexpr uint32_t CB_ZERO = get_compile_time_arg_val(6);
    constexpr uint32_t CT_A = get_compile_time_arg_val(7);
    constexpr uint32_t CT_B = get_compile_time_arg_val(8);
    constexpr uint32_t T = get_compile_time_arg_val(9);
    constexpr uint32_t NSRC = get_compile_time_arg_val(10);
    constexpr uint32_t NCHUNKS_A = get_compile_time_arg_val(11);
    constexpr uint32_t NCHUNKS_B = get_compile_time_arg_val(12);
    constexpr uint32_t PS = get_compile_time_arg_val(13);
    constexpr uint32_t CPP = get_compile_time_arg_val(14);
    constexpr uint32_t NP_A = get_compile_time_arg_val(15);
    constexpr uint32_t NP_B = get_compile_time_arg_val(16);
    constexpr uint32_t SEM_GO = get_compile_time_arg_val(17);
    constexpr uint32_t SEM_STAGED = get_compile_time_arg_val(18);
    constexpr uint32_t HAS_B = get_compile_time_arg_val(19);
    constexpr auto outA_args = TensorAccessorArgs<20>();
    constexpr auto recvA_args = TensorAccessorArgs<outA_args.next_compile_time_args_offset()>();
    constexpr auto outB_args = TensorAccessorArgs<recvA_args.next_compile_time_args_offset()>();
    constexpr auto recvB_args = TensorAccessorArgs<outB_args.next_compile_time_args_offset()>();
    constexpr uint32_t TILE_BYTES = 2048;
    constexpr uint32_t PKT = CPP * PS;
    constexpr uint32_t SLOT_B = NCHUNKS_B * PS;
    static_assert(NSRC * T <= 32, "one tile row");

    const uint32_t outA_addr = get_arg_val<uint32_t>(0);
    const uint32_t colA0 = get_arg_val<uint32_t>(1);
    const uint32_t kA = get_arg_val<uint32_t>(2);
    const uint32_t my_idx = get_arg_val<uint32_t>(3);
    const uint32_t xchg_noc = get_arg_val<uint32_t>(4);
    const uint32_t recvA_addr = get_arg_val<uint32_t>(5);
    const uint32_t own_noc = get_arg_val<uint32_t>(6);
    const uint32_t outB_addr = get_arg_val<uint32_t>(7);
    const uint32_t colB0 = get_arg_val<uint32_t>(8);
    const uint32_t kB = get_arg_val<uint32_t>(9);
    const uint32_t recvB_addr = get_arg_val<uint32_t>(10);
    const uint32_t xx = xchg_noc >> 16, xy = xchg_noc & 0xffff;
    const uint32_t ox = own_noc >> 16, oy = own_noc & 0xffff;

    volatile tt_l1_ptr uint32_t* go = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SEM_GO));
    const uint64_t staged_noc = get_noc_addr(xx, xy, get_semaphore(SEM_STAGED));
    const uint32_t sendA = get_write_ptr(CB_SEND_A);
    const uint32_t sendB = get_write_ptr(CB_SEND_B);

    // ---------------------------------------------------------------- phase A: the gather of this chunk
    {
        constexpr uint32_t ROW_B = CT_A * 64, PIECE = T * ROW_B;
        const auto s_out = TensorAccessor(outA_args, outA_addr, TILE_BYTES);
        const auto s_recv = TensorAccessor(recvA_args, recvA_addr, PKT);
        cb_wait_front(CB_RM, CT_A);
        const uint32_t rm = get_read_ptr(CB_RM);
        noc_async_write(rm, get_noc_addr(xx, xy, sendA + kA * PS), PIECE);  // the T local rows, for every peer
        noc_async_write_barrier();
        noc_semaphore_inc(staged_noc, 1);
        noc_semaphore_wait_min(go, 1);
        noc_semaphore_set(go, 0);
        cb_reserve_back(CB_RM2, CT_A);
        const uint32_t rm2 = get_write_ptr(CB_RM2);
        for (uint32_t r = 0; r < NSRC; ++r) {
            const uint32_t dst = rm2 + r * PIECE;
            if (r == my_idx) {
                noc_async_read(get_noc_addr(ox, oy, rm), dst, PIECE);
            } else {
                noc_async_read(s_recv.get_noc_addr(r * NP_A + kA / CPP, (kA % CPP) * PS), dst, PIECE);
            }
        }
        if constexpr (NSRC * T < 32) {
            cb_wait_front(CB_ZERO, 1);
            const uint64_t zero = get_noc_addr(ox, oy, get_read_ptr(CB_ZERO));
            for (uint32_t i = NSRC * T; i < 32; ++i) {
                noc_async_read(zero, rm2 + i * ROW_B, ROW_B);
            }
        }
        noc_async_read_barrier();
        cb_pop_front(CB_RM, CT_A);
        cb_push_back(CB_RM2, CT_A);
        cb_wait_front(CB_T, CT_A);
        const uint32_t src = get_read_ptr(CB_T);
        for (uint32_t j = 0; j < CT_A; ++j) {
            noc_async_write(src + j * TILE_BYTES, s_out.get_noc_addr(colA0 + j), TILE_BYTES);
        }
        noc_async_write_barrier();
        cb_pop_front(CB_T, CT_A);
        noc_semaphore_inc(staged_noc, 1);  // "written": the exchange core releases the experts after NW of these
    }

    if constexpr (HAS_B) {
        // ------------------------------------------------------------ phase B: the reduce of this chunk
        constexpr uint32_t ROW_B = CT_B * 64, PIECE = T * ROW_B;
        const auto s_out = TensorAccessor(outB_args, outB_addr, TILE_BYTES);
        const auto s_recv = TensorAccessor(recvB_args, recvB_addr, PKT);
        cb_wait_front(CB_RM, CT_B);
        const uint32_t rm = get_read_ptr(CB_RM);
        for (uint32_t r = 0; r < NSRC; ++r) {
            if (r != my_idx) {
                noc_async_write(rm + r * PIECE, get_noc_addr(xx, xy, sendB + r * SLOT_B + kB * PS), PIECE);
            }
        }
        noc_async_write_barrier();
        noc_semaphore_inc(staged_noc, 1);
        noc_semaphore_wait_min(go, 1);
        noc_semaphore_set(go, 0);
        cb_reserve_back(CB_RM2, CT_B);
        const uint32_t rm2 = get_write_ptr(CB_RM2);
        for (uint32_t r = 0; r < NSRC; ++r) {
            const uint32_t dst = rm2 + r * PIECE;
            if (r == my_idx) {
                noc_async_read(get_noc_addr(ox, oy, rm + my_idx * PIECE), dst, PIECE);
            } else {
                noc_async_read(s_recv.get_noc_addr(r * NP_B + kB / CPP, (kB % CPP) * PS), dst, PIECE);
            }
        }
        if constexpr (NSRC * T < 32) {
            cb_wait_front(CB_ZERO, 1);
            const uint64_t zero = get_noc_addr(ox, oy, get_read_ptr(CB_ZERO));
            for (uint32_t i = NSRC * T; i < 32; ++i) {
                noc_async_read(zero, rm2 + i * ROW_B, ROW_B);
            }
        }
        noc_async_read_barrier();
        cb_pop_front(CB_RM, CT_B);
        cb_push_back(CB_RM2, CT_B);
        cb_wait_front(CB_OUT, CT_B);
        const uint32_t src = get_read_ptr(CB_OUT);
        for (uint32_t j = 0; j < CT_B; ++j) {
            noc_async_write(src + j * TILE_BYTES, s_out.get_noc_addr(colB0 + j), TILE_BYTES);
        }
        noc_async_write_barrier();
        cb_pop_front(CB_OUT, CT_B);
    }
    noc_async_full_barrier();
}
