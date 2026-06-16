// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#if defined(UCK_CHLKC_UNPACK) || defined(UCK_CHLKC_MATH) || defined(UCK_CHLKC_PACK)
#include "api/compute/common.h"
#endif
#include "api/debug/device_print.h"

/*
 * Same as print_iterations.cpp but takes the iteration count as a compile-time arg.
 * Used by tests that need to run on Quasar where runtime args via SetRuntimeArgs/get_arg_val
 * are not exposed for the QuasarDataMovementConfig/QuasarComputeConfig kernel path yet.
 */
void kernel_main() {
    constexpr uint32_t count = get_compile_time_arg_val(0);
    for (uint32_t i = 0; i < count; i++) {
        DEVICE_PRINT("Test iteration: {}\n", i);
    }
}
