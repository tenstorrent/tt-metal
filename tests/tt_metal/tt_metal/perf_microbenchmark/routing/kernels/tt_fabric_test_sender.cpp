// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_fabric_test_kernels_utils.hpp"

constexpr uint8_t IS_2D_FABRIC = get_compile_time_arg_val(0);
constexpr uint8_t USE_DYNAMIC_ROUTING = get_compile_time_arg_val(1);
constexpr uint8_t NUM_FABRIC_CONNECTIONS = get_compile_time_arg_val(2);
constexpr uint8_t NUM_TRAFFIC_CONFIGS = get_compile_time_arg_val(3);
constexpr bool BENCHMARK_MODE = get_compile_time_arg_val(5);

using SenderKernelConfig = tt::tt_fabric::fabric_tests::SenderKernelConfig;

void kernel_main() {
    size_t rt_args_idx = 0;
    auto sender_config = SenderKernelConfig::build_from_args<IS_2D_FABRIC, USE_DYNAMIC_ROUTING>(rt_args_idx);

    std::array<uint64_t, NUM_TRAFFIC_CONFIGS> elapsed_cycles = {};

    // clear out test results area

    sender_config.open_connections();

    bool packets_left = true;

    while (packets_left) {
        packets_left = false;
        for (uint8_t i = 0; i < NUM_TRAFFIC_CONFIGS; i++) {
            auto& traffic_config = sender_config.traffic_configs[i];
            if (!traffic_config.has_packets_to_send()) {
                continue;
            }

            uint64_t start_timestamp = get_timestamp();
            packets_left |= traffic_config.send_packets<BENCHMARK_MODE>();
            elapsed_cycles[i] += get_timestamp() - start_timestamp;
        }
    }

    sender_config.close_connections();

    // dump results per traffic config
}
