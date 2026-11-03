// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Row exchange, worker reader (NCRISC): reads this worker's chunk of the local input (CT tiles of one tile row) into
// CB_TIN for the untilize, and prepares the constant tiles the writer / compute need: the reduce selector tile
// S[t, r*T + t] = 1 (every row's user t summed into output row t) and, for a gather whose rows*T < 32, a zero block
// for the padding rows.
//
// compile-time: [MODE (0 gather / 1 reduce), CB_TIN, CB_SEL, CB_ZERO, CT, T, ROWS, PS, <TensorAccessorArgs in>]
// runtime: [in_addr, col_tile0]
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t MODE = get_compile_time_arg_val(0);
    constexpr uint32_t CB_TIN = get_compile_time_arg_val(1);
    constexpr uint32_t CB_SEL = get_compile_time_arg_val(2);
    constexpr uint32_t CB_ZERO = get_compile_time_arg_val(3);
    constexpr uint32_t CT = get_compile_time_arg_val(4);
    constexpr uint32_t T = get_compile_time_arg_val(5);
    constexpr uint32_t ROWS = get_compile_time_arg_val(6);
    constexpr uint32_t PS = get_compile_time_arg_val(7);
    constexpr auto in_args = TensorAccessorArgs<8>();
    constexpr uint32_t TILE_BYTES = 2048;

    const uint32_t in_addr = get_arg_val<uint32_t>(0);
    const uint32_t col0 = get_arg_val<uint32_t>(1);

    const auto s_in = TensorAccessor(in_args, in_addr, TILE_BYTES);
    cb_reserve_back(CB_TIN, CT);
    const uint32_t tin = get_write_ptr(CB_TIN);
    for (uint32_t j = 0; j < CT; ++j) {
        noc_async_read(s_in.get_noc_addr(col0 + j), tin + j * TILE_BYTES, TILE_BYTES);
    }

    if constexpr (MODE == 1) {
        // the selector tile (bf16 TILE layout: face f = (row / 16) * 2 + col / 16, element (row % 16) * 16 + col % 16)
        cb_reserve_back(CB_SEL, 1);
        volatile tt_l1_ptr uint32_t* sel32 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(CB_SEL));
        for (uint32_t i = 0; i < TILE_BYTES / 4; ++i) {
            sel32[i] = 0;
        }
        volatile tt_l1_ptr uint16_t* sel = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(get_write_ptr(CB_SEL));
        for (uint32_t r = 0; r < ROWS; ++r) {
            for (uint32_t t = 0; t < T; ++t) {
                const uint32_t row = t, col = r * T + t;
                const uint32_t face = (row >> 4) * 2 + (col >> 4);
                sel[face * 256 + (row & 15) * 16 + (col & 15)] = 0x3F80;  // bf16 1.0
            }
        }
        cb_push_back(CB_SEL, 1);
    } else if constexpr (ROWS * T < 32) {
        cb_reserve_back(CB_ZERO, 1);
        volatile tt_l1_ptr uint32_t* z = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(CB_ZERO));
        for (uint32_t i = 0; i < PS / 4; ++i) {
            z[i] = 0;
        }
        cb_push_back(CB_ZERO, 1);
    }

    noc_async_read_barrier();
    cb_push_back(CB_TIN, CT);
}
