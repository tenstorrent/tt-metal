// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Prefill q / o layout moves, compute: MODE 0 untilize 16 tiles -> 32 rows, MODE 1 tilize 32 rows -> 16 tiles (bf16).

#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/dataflow/circular_buffer.h"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp"

void kernel_main() {
    constexpr uint32_t cb_in = get_compile_time_arg_val(0);
    constexpr uint32_t cb_out = get_compile_time_arg_val(1);
    constexpr uint32_t MODE = get_compile_time_arg_val(2);
    const uint32_t nu = get_arg_val<uint32_t>(0);
    compute_kernel_hw_startup(cb_in, cb_in, cb_out);
    for (uint32_t u = 0; u < nu; ++u) {
        if constexpr (MODE == 0) {
            compute_kernel_lib::untilize<16, cb_in, cb_out>(1);
        } else {
            compute_kernel_lib::tilize<16, cb_in, cb_out>(1, 32);
        }
    }
}
