// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/untilize.h"
#include "api/compute/pack_untilize.h"
#include "ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp"

using namespace compute_kernel_lib::untilize_config;

#ifndef UNTILIZE_INIT_UNINIT_MODE
#define UNTILIZE_INIT_UNINIT_MODE InitUninitMode::InitAndUninit
#endif

#ifndef UNTILIZE_WAIT_MODE
#define UNTILIZE_WAIT_MODE WaitMode::WaitBlock
#endif

#ifndef UNTILIZE_RECONFIG_MODE
#define UNTILIZE_RECONFIG_MODE ReconfigureRegisterDatatypeMode::UnpackAndPackReconfigure
#endif

void kernel_main() {
    constexpr uint32_t block_width = get_compile_time_arg_val(0);
    constexpr uint32_t num_blocks = get_compile_time_arg_val(1);
    constexpr uint32_t input_cb_id = get_compile_time_arg_val(2);
    constexpr uint32_t output_cb_id = get_compile_time_arg_val(3);

    // Runtime arg: total_output_pages (rows) for asymmetric CB page mode.
    // 0 means symmetric (std::nullopt).
    uint32_t rows = get_arg_val<uint32_t>(0);
    std::optional<uint32_t> total_output_pages = rows > 0 ? std::optional<uint32_t>(rows) : std::nullopt;

    compute_kernel_hw_startup(input_cb_id, output_cb_id);

    compute_kernel_lib::untilize<
        block_width,
        input_cb_id,
        output_cb_id,
        UNTILIZE_INIT_UNINIT_MODE,
        UNTILIZE_WAIT_MODE,
        UNTILIZE_RECONFIG_MODE>(num_blocks, total_output_pages);
}
