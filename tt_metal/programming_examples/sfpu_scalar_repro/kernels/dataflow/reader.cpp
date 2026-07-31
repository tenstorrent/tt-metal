// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "tt-metalium/constants.hpp"
#include "api/numeric/bfloat16.h"

#include <cstdint>

void kernel_main() {
    // Compile time args
    constexpr uint32_t cb_in0 = get_compile_time_arg_val(0);
    constexpr uint32_t cb_in1 = get_compile_time_arg_val(1);

    constexpr uint32_t one_tile = 1;

    // Fabricate two tiles filled with 1.0f directly in L1 -- no DRAM input needed, this repro
    // only cares about known, exact values (1.0 + 1.0 = 2.0 going into the scalar op under test).
    cb_reserve_back(cb_in0, one_tile);
    volatile tt_l1_ptr uint16_t* in0_ptr = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(get_write_ptr(cb_in0));
    for (uint32_t i = 0; i < tt::constants::TILE_HW; i++) {
        in0_ptr[i] = fp32_to_bf16_truncate(1.0f);
    }
    cb_push_back(cb_in0, one_tile);

    cb_reserve_back(cb_in1, one_tile);
    volatile tt_l1_ptr uint16_t* in1_ptr = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(get_write_ptr(cb_in1));
    for (uint32_t i = 0; i < tt::constants::TILE_HW; i++) {
        in1_ptr[i] = fp32_to_bf16_truncate(1.0f);
    }
    cb_push_back(cb_in1, one_tile);
}
