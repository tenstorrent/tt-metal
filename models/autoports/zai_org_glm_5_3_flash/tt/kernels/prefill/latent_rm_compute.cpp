// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Stage 16 GLM53_PF_LAT_BOUND: untilize the visible local blocks' tile rows (TPR tiles wide) into ROW_MAJOR rows. The
// block count comes from the reader through CB_META (read_tile_value: one value on all three compute threads).
// Compile-time: TPR, TROWS.
#include <cstdint>

#include "api/compute/cb_api.h"
#include "api/compute/common.h"
#include "ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp"

void kernel_main() {
    constexpr uint32_t TPR = get_compile_time_arg_val(0);
    constexpr uint32_t TROWS = get_compile_time_arg_val(1);
    constexpr uint32_t CB_IN = 0, CB_OUT = 1, CB_META = 2;
    compute_kernel_hw_startup(CB_IN, CB_OUT);
    cb_wait_front(CB_META, 1);
    const uint32_t n = read_tile_value(CB_META, 0, 0);
    cb_pop_front(CB_META, 1);
    if (n == 0) {
        return;
    }
    compute_kernel_lib::untilize<
        TPR,
        CB_IN,
        CB_OUT,
        compute_kernel_lib::untilize_config::InitUninitMode::InitAndUninit,
        compute_kernel_lib::untilize_config::WaitMode::WaitBlock,
        compute_kernel_lib::untilize_config::ReconfigureRegisterDatatypeMode::NoReconfigure>(n * TROWS);
}
