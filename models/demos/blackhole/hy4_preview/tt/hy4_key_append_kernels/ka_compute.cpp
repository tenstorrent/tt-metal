// SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
// Hy4 B32 key append, compute: untilize the input tiles once, then per row: untilize the 4 cache tiles (cb_cache ->
// cb_ut), wait for the reader's patch (cb_ut2 aliases cb_ut), tilize back (-> cb_out). Same helper calls as the stock
// update_cache compute kernel. CT: 0 cb_kin, 1 cb_cache, 2 cb_ut, 3 cb_ut2, 4 cb_kin_rm, 5 cb_out, 6 S.
#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/pack_untilize.h"
#include "api/compute/tilize.h"
#include "ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"
#include "api/compute/pack.h"

void kernel_main() {
    constexpr uint32_t cb_kin = get_compile_time_arg_val(0);
    constexpr uint32_t cb_cache = get_compile_time_arg_val(1);
    constexpr uint32_t cb_ut = get_compile_time_arg_val(2);
    constexpr uint32_t cb_ut2 = get_compile_time_arg_val(3);
    constexpr uint32_t cb_kin_rm = get_compile_time_arg_val(4);
    constexpr uint32_t cb_out = get_compile_time_arg_val(5);
    constexpr uint32_t S = get_compile_time_arg_val(6);
    constexpr uint32_t Wt = 4;
    compute_kernel_hw_startup(cb_kin, cb_kin_rm);
    compute_kernel_lib::untilize<
        Wt,
        cb_kin,
        cb_kin_rm,
        compute_kernel_lib::untilize_config::InitUninitMode::InitAndUninit,
        compute_kernel_lib::untilize_config::WaitMode::WaitBlock,
        compute_kernel_lib::untilize_config::ReconfigureRegisterDatatypeMode::NoReconfigure>(1);
    for (uint32_t j = 0; j < S; ++j) {
        compute_kernel_lib::untilize<Wt, cb_cache, cb_ut>(1);
        compute_kernel_lib::tilize<Wt, cb_ut2, cb_out>(1);
    }
}
