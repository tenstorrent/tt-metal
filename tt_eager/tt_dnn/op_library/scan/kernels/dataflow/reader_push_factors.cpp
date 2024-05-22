
// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "dataflow_api.h"

constexpr uint32_t TILE_WIDTH = 32;
constexpr uint32_t TILE_HEIGHT = 32;
constexpr uint32_t TILE_HW = 1024;

constexpr auto cb_src = get_compile_time_arg_val(0);
constexpr auto cb_factors = get_compile_time_arg_val(2);
constexpr uint32_t bf16_one_u32 = get_compile_time_arg_val(4);

#define ALWI inline __attribute__((always_inline))

ALWI void fill_with_val(uint32_t begin_addr, uint32_t n_tiles, uint16_t val) {
    volatile tt_l1_ptr uint16_t* ptr = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(begin_addr);

    for (uint32_t i = 0; i < n_tiles; ++i) {
        for (uint32_t j = 0; j < TILE_HW; ++j) {
            ptr[j] = val;
        }
        ptr += TILE_HW;
    }
}

void kernel_main() {
    uint32_t tiles_per_col = get_arg_val<uint32_t>(0);
    uint32_t reshapes_per_row = get_arg_val<uint32_t>(1);
    uint32_t total_tiles = get_arg_val<uint32_t>(2);

    uint16_t bf16_one_u16 = bf16_one_u32 >> 16;
    fill_with_val(get_write_ptr(cb_factors), reshapes_per_row, bf16_one_u16);

    cb_push_back(cb_src, total_tiles);  // signal to compute kernel that the src CB is ready
}
