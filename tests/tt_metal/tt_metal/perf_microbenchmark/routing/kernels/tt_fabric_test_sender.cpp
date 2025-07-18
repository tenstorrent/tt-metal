// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_fabric_test_kernels_utils.hpp"

constexpr uint8_t IS_2D_FABRIC = get_compile_time_arg_val(0);
constexpr uint8_t USE_DYNAMIC_ROUTING = get_compile_time_arg_val(1);
constexpr uint8_t NUM_FABRIC_CONNECTIONS = get_compile_time_arg_val(2);
constexpr uint8_t NUM_TRAFFIC_CONFIGS = get_compile_time_arg_val(3);
constexpr bool BENCHMARK_MODE = get_compile_time_arg_val(4);
constexpr bool LINE_SYNC = get_compile_time_arg_val(5);
constexpr uint8_t NUM_LOCAL_SYNC_CORES = get_compile_time_arg_val(6);

using SenderKernelConfig = tt::tt_fabric::fabric_tests::SenderKernelConfig<
    NUM_FABRIC_CONNECTIONS,
    NUM_TRAFFIC_CONFIGS,
    IS_2D_FABRIC,
    USE_DYNAMIC_ROUTING,
    LINE_SYNC,
    NUM_LOCAL_SYNC_CORES>;

void kernel_main() {
    size_t rt_args_idx = 0;
    auto sender_config = SenderKernelConfig::build_from_args(rt_args_idx);

    // Clear test results area and mark as started
    tt::tt_fabric::fabric_tests::clear_test_results(
        sender_config.get_result_buffer_address(), sender_config.get_result_buffer_size());
    tt::tt_fabric::fabric_tests::write_test_status(sender_config.get_result_buffer_address(), TT_FABRIC_STATUS_STARTED);

    // Local sync (as participant, not master)
    if constexpr (LINE_SYNC) {
        sender_config.local_sync();
    }

    sender_config.open_connections();

    bool packets_left_to_send = true;
    uint64_t total_packets_sent = 0;
    uint64_t total_elapsed_cycles = 0;

    // Round-robin packet sending: send one packet from each config per iteration
    uint64_t start_timestamp = get_timestamp();
    while (packets_left_to_send) {
        packets_left_to_send = false;
        for (uint8_t i = 0; i < NUM_TRAFFIC_CONFIGS; i++) {
            auto* traffic_config = sender_config.traffic_config_ptrs[i];
            if (!traffic_config->has_packets_to_send()) {
                continue;
            }

            // TODO: might want to check if the buffer has wrapped or not
            // if wrapped, then wait for credits from the receiver

            // Always send exactly one packet per config per round
            traffic_config->send_one_packet<BENCHMARK_MODE>();
            packets_left_to_send |= traffic_config->has_packets_to_send();
        }
    }
    uint64_t total_elapsed_cycles_outer_loop = get_timestamp() - start_timestamp;

    sender_config.close_connections();

    // Collect results from all traffic configs
    for (uint8_t i = 0; i < NUM_TRAFFIC_CONFIGS; i++) {
        auto* traffic_config = sender_config.traffic_config_ptrs[i];
        total_packets_sent += traffic_config->num_packets_processed;
    }

    // Write test results
    tt::tt_fabric::fabric_tests::write_test_cycles(
        sender_config.get_result_buffer_address(), total_elapsed_cycles_outer_loop);
    tt::tt_fabric::fabric_tests::write_test_packets(sender_config.get_result_buffer_address(), total_packets_sent);

    // Mark test as passed
    tt::tt_fabric::fabric_tests::write_test_status(sender_config.get_result_buffer_address(), TT_FABRIC_STATUS_PASS);
}
