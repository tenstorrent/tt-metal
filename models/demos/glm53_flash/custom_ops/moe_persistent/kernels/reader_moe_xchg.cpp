// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Persistent MoE program, exchange worker reader (NCRISC): phase A reads this worker's chunk of the local x /
// routing tile row for the gather (CT_A tiles into CB_TIN) and the zero block; phase B (workers with a reduce chunk)
// waits until every down core wrote its output tiles (PSEM_DONE counted PDONE times), then reads this worker's
// chunk of the expert output (CT_B tiles into CB_TIN) and builds the reduce selector tile.
//
// compile-time: [CB_TIN, CB_SEL, CB_ZERO, CT_A, CT_B, T, NSRC, PS, HAS_B, PSEM_DONE, PDONE,
//                <TensorAccessorArgs inA>, <TensorAccessorArgs inB>]
// runtime: [inA_addr, colA0, inB_addr, colB0]
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t CB_TIN = get_compile_time_arg_val(0);
    constexpr uint32_t CB_SEL = get_compile_time_arg_val(1);
    constexpr uint32_t CB_ZERO = get_compile_time_arg_val(2);
    constexpr uint32_t CT_A = get_compile_time_arg_val(3);
    constexpr uint32_t CT_B = get_compile_time_arg_val(4);
    constexpr uint32_t T = get_compile_time_arg_val(5);
    constexpr uint32_t NSRC = get_compile_time_arg_val(6);
    constexpr uint32_t PS = get_compile_time_arg_val(7);
    constexpr uint32_t HAS_B = get_compile_time_arg_val(8);
    constexpr uint32_t PSEM_DONE = get_compile_time_arg_val(9);
    constexpr uint32_t PDONE = get_compile_time_arg_val(10);
    constexpr auto inA_args = TensorAccessorArgs<11>();
    constexpr auto inB_args = TensorAccessorArgs<inA_args.next_compile_time_args_offset()>();
    constexpr uint32_t TILE_BYTES = 2048;

    const uint32_t inA_addr = get_arg_val<uint32_t>(0);
    const uint32_t colA0 = get_arg_val<uint32_t>(1);
    const uint32_t inB_addr = get_arg_val<uint32_t>(2);
    const uint32_t colB0 = get_arg_val<uint32_t>(3);

    // ---- phase A: the local chunk of the gather input
    {
        const auto s_in = TensorAccessor(inA_args, inA_addr, TILE_BYTES);
        cb_reserve_back(CB_TIN, CT_A);
        const uint32_t tin = get_write_ptr(CB_TIN);
        for (uint32_t j = 0; j < CT_A; ++j) {
            noc_async_read(s_in.get_noc_addr(colA0 + j), tin + j * TILE_BYTES, TILE_BYTES);
        }
        if constexpr (NSRC * T < 32) {
            cb_reserve_back(CB_ZERO, 1);
            volatile tt_l1_ptr uint32_t* z = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(CB_ZERO));
            for (uint32_t i = 0; i < PS / 4; ++i) {
                z[i] = 0;
            }
            cb_push_back(CB_ZERO, 1);
        }
        noc_async_read_barrier();
        cb_push_back(CB_TIN, CT_A);
    }

    if constexpr (HAS_B) {
        // ---- phase B: the expert output is written when every down core counted itself
        volatile tt_l1_ptr uint32_t* done = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(PSEM_DONE));
        noc_semaphore_wait_min(done, PDONE);
        noc_semaphore_set(done, 0);
        const auto s_in = TensorAccessor(inB_args, inB_addr, TILE_BYTES);
        cb_reserve_back(CB_TIN, CT_B);
        const uint32_t tin = get_write_ptr(CB_TIN);
        for (uint32_t j = 0; j < CT_B; ++j) {
            noc_async_read(s_in.get_noc_addr(colB0 + j), tin + j * TILE_BYTES, TILE_BYTES);
        }
        // the selector tile S[t, s*T + t] = 1 (bf16 TILE: face f = (row / 16) * 2 + col / 16)
        cb_reserve_back(CB_SEL, 1);
        volatile tt_l1_ptr uint32_t* sel32 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(CB_SEL));
        for (uint32_t i = 0; i < TILE_BYTES / 4; ++i) {
            sel32[i] = 0;
        }
        volatile tt_l1_ptr uint16_t* sel = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(get_write_ptr(CB_SEL));
        for (uint32_t r = 0; r < NSRC; ++r) {
            for (uint32_t t = 0; t < T; ++t) {
                const uint32_t row = t, col = r * T + t;
                const uint32_t face = (row >> 4) * 2 + (col >> 4);
                sel[face * 256 + (row & 15) * 16 + (col & 15)] = 0x3F80;
            }
        }
        cb_push_back(CB_SEL, 1);
        noc_async_read_barrier();
        cb_push_back(CB_TIN, CT_B);
    }
}
