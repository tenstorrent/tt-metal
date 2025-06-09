// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_fabric_test_kernels_utils.hpp"

using ReceiverKernelConfig = tt::tt_fabric::fabric_tests::SenderKernelConfig;

constexpr uint8_t NUM_TRAFFIC_CONFIGS = get_compile_time_arg_val(0);
constexpr bool BENCHMARK_MODE = get_compile_time_arg_val(1);

void kernel_main() {
    size_t rt_args_idx = 0;
    auto receiver_config = ReceiverKernelConfig::build_from_args<IS_2D_FABRIC, USE_DYNAMIC_ROUTING>(rt_args_idx);

    // clear out test results area

    for (uint8_t i = 0; i < NUM_TRAFFIC_CONFIGS; i++) {
        auto& traffic_config = receiver_config.traffic_configs[i];
    }

    // dump results
}
