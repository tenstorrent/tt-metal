// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_fabric_test_kernels_utils.hpp"

using namespace tt::tt_fabric::fabric_tests;

constexpr uint8_t IS_2D_FABRIC = get_compile_time_arg_val(0);
constexpr uint8_t USE_DYNAMIC_ROUTING = get_compile_time_arg_val(1);
constexpr uint8_t NUM_FABRIC_CONNECTIONS = get_compile_time_arg_val(2);
constexpr uint8_t NUM_TRAFFIC_CONFIGS = get_compile_time_arg_val(3);
constexpr bool BENCHMARK_MODE = get_compile_time_arg_val(4);
constexpr bool LINE_SYNC = get_compile_time_arg_val(5);
constexpr uint8_t NUM_LOCAL_SYNC_CORES = get_compile_time_arg_val(6);
constexpr uint32_t KERNEL_CONFIG_BUFFER_SIZE = get_compile_time_arg_val(7);

using SenderKernelConfigType = SenderKernelConfig<
    NUM_FABRIC_CONNECTIONS,
    NUM_TRAFFIC_CONFIGS,
    IS_2D_FABRIC,
    USE_DYNAMIC_ROUTING,
    LINE_SYNC,
    NUM_LOCAL_SYNC_CORES>;

// Static assertion to ensure this config fits within the allocated kernel config region
static_assert(
    sizeof(SenderKernelConfigType) <= KERNEL_CONFIG_BUFFER_SIZE,
    "SenderKernelConfig size exceeds allocated kernel config buffer size");

void kernel_main() {
    size_t rt_args_idx = 0;

    // Get kernel config address from runtime args
    CommonMemoryMap common_memory_map = CommonMemoryMap::build_from_args(rt_args_idx);
    uint32_t kernel_config_address = common_memory_map.kernel_config_base;

    // Use placement new to construct config in L1 memory
    auto* sender_config = new (reinterpret_cast<void*>(kernel_config_address))
        SenderKernelConfigType(SenderKernelConfigType::build_from_args(common_memory_map, rt_args_idx));

    // Clear test results area and mark as started
    clear_test_results(sender_config->get_result_buffer_address(), sender_config->get_result_buffer_size());
    write_test_status(sender_config->get_result_buffer_address(), TT_FABRIC_STATUS_STARTED);

    // Local sync (as participant, not master)
    uint8_t sync_iter = 0;
    if constexpr (LINE_SYNC) {
        sender_config->local_sync(sync_iter++);
    }

    sender_config->open_connections();

    bool packets_left_to_send = true;
    uint64_t total_packets_sent = 0;
    uint64_t total_elapsed_cycles = 0;

    // Round-robin packet sending: send one packet from each config per iteration
    uint64_t start_timestamp = get_timestamp();
    while (packets_left_to_send) {
        packets_left_to_send = false;
        for (uint8_t i = 0; i < NUM_TRAFFIC_CONFIGS; i++) {
            auto* traffic_config = sender_config->traffic_config_ptrs[i];
            if (!traffic_config->has_packets_to_send()) {
                continue;
            }

            // for (int c = 0; c < 10000; ++c) {
            //     asm volatile("nop");
            // }

            // TODO: might want to check if the buffer has wrapped or not
            // if wrapped, then wait for credits from the receiver

            // Always send exactly one packet per config per round
            traffic_config->send_one_packet<BENCHMARK_MODE>();
            packets_left_to_send |= traffic_config->has_packets_to_send();
        }
    }

    sender_config->close_connections();

    // Local sync (as participant, not master) for end of sync, first sync tells sync core to start global sync, second
    // sync is waiting for global sync done
    if constexpr (LINE_SYNC) {
        sender_config->local_sync(sync_iter++);
        sender_config->local_sync(sync_iter++);
    }

    uint64_t total_elapsed_cycles_outer_loop = get_timestamp() - start_timestamp;

    // Collect results from all traffic configs
    for (uint8_t i = 0; i < NUM_TRAFFIC_CONFIGS; i++) {
        auto* traffic_config = sender_config->traffic_config_ptrs[i];
        total_packets_sent += traffic_config->num_packets_processed;
    }

    // Write test results
    write_test_cycles(sender_config->get_result_buffer_address(), total_elapsed_cycles_outer_loop);
    write_test_packets(sender_config->get_result_buffer_address(), total_packets_sent);

    // Mark test as passed
    write_test_status(sender_config->get_result_buffer_address(), TT_FABRIC_STATUS_PASS);

    // Make sure all the noc txns are done
    noc_async_full_barrier();
}
