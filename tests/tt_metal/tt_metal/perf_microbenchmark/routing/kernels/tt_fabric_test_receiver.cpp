// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "debug/dprint.h"
#include "tt_fabric_test_kernels_utils.hpp"

using namespace tt::tt_fabric::fabric_tests;

constexpr uint8_t NUM_TRAFFIC_CONFIGS = get_compile_time_arg_val(0);
constexpr bool BENCHMARK_MODE = get_compile_time_arg_val(1);
constexpr uint32_t KERNEL_CONFIG_BUFFER_SIZE = get_compile_time_arg_val(2);

using ReceiverKernelConfigType = ReceiverKernelConfig<NUM_TRAFFIC_CONFIGS>;

// Static assertion to ensure this config fits within the allocated kernel config region
static_assert(
    sizeof(ReceiverKernelConfigType) <= KERNEL_CONFIG_BUFFER_SIZE,
    "ReceiverKernelConfig size exceeds allocated kernel config buffer size");

void kernel_main() {
    size_t rt_args_idx = 0;

    // Get kernel config address from runtime args
    CommonMemoryMap common_memory_map = CommonMemoryMap::build_from_args(rt_args_idx);
    uint32_t kernel_config_address = common_memory_map.kernel_config_base;

    // Use placement new to construct config in L1 memory
    auto* receiver_config = new (reinterpret_cast<void*>(kernel_config_address))
        ReceiverKernelConfigType(ReceiverKernelConfigType::build_from_args(common_memory_map, rt_args_idx));

    // Clear test results area and mark as started
    clear_test_results(receiver_config->get_result_buffer_address(), receiver_config->get_result_buffer_size());
    write_test_status(receiver_config->get_result_buffer_address(), TT_FABRIC_STATUS_STARTED);

    bool failed = false;
    uint64_t total_packets_received = 0;

    bool packets_left_to_validate = true;
    while (packets_left_to_validate) {
        packets_left_to_validate = false;
        for (uint8_t i = 0; i < NUM_TRAFFIC_CONFIGS; i++) {
            auto* traffic_config = receiver_config->traffic_configs[i];
            if constexpr (!BENCHMARK_MODE) {
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
                total_packets_received++;
                packets_left_to_validate |= traffic_config->has_packets_to_validate();
            } else {
                total_packets_received += traffic_config->metadata.num_packets;
            }
        }

        if (failed) {
            break;
        }
    }

    // Write test results
    write_test_packets(receiver_config->get_result_buffer_address(), total_packets_received);

    // Mark test as passed or failed
    uint32_t final_status = failed ? TT_FABRIC_STATUS_DATA_MISMATCH : TT_FABRIC_STATUS_PASS;
    write_test_status(receiver_config->get_result_buffer_address(), final_status);

    // Make sure all the noc txns are done
    noc_async_full_barrier();
}
