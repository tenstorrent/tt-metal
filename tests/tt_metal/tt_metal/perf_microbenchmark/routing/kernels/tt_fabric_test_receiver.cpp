// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "debug/dprint.h"
#include "tt_fabric_test_kernels_utils.hpp"

constexpr uint8_t NUM_TRAFFIC_CONFIGS = get_compile_time_arg_val(0);
constexpr bool BENCHMARK_MODE = get_compile_time_arg_val(1);

void kernel_main() {
    using ReceiverKernelConfig = tt::tt_fabric::fabric_tests::ReceiverKernelConfig<NUM_TRAFFIC_CONFIGS>;

    size_t rt_args_idx = 0;
    auto receiver_config = ReceiverKernelConfig::build_from_args(rt_args_idx);

    // clear out test results area

    bool failed = false;

    bool packets_left_to_validate = true;
    while (packets_left_to_validate) {
        packets_left_to_validate = false;
        for (uint8_t i = 0; i < NUM_TRAFFIC_CONFIGS; i++) {
            auto* traffic_config = receiver_config.traffic_configs[i];
            if (!traffic_config->has_packets_to_validate()) {
                continue;
            }

            // if we are here, this means that we have atleast 1 packet left to validate
            packets_left_to_validate = true;
            bool got_new_data = traffic_config->poll();
            if (!got_new_data) {
                continue;
            }

            bool data_valid = traffic_config->validate();
            if (!data_valid) {
                failed = true;
                break;
            }

            traffic_config->advance();
            packets_left_to_validate |= traffic_config->has_packets_to_validate();
        }

        if (failed) {
            break;
        }
    }

    // dump results
}
