// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_fabric_test_kernels_utils.hpp"

constexpr uint8_t IS_2D_FABRIC = get_compile_time_arg_val(0);
constexpr uint8_t USE_DYNAMIC_ROUTING = get_compile_time_arg_val(1);
constexpr uint8_t NUM_SYNC_FABRIC_CONNECTIONS = get_compile_time_arg_val(2);
constexpr uint8_t NUM_LOCAL_SYNC_CORES = get_compile_time_arg_val(3);

using SyncKernelConfig = tt::tt_fabric::fabric_tests::
    SyncKernelConfig<NUM_SYNC_FABRIC_CONNECTIONS, IS_2D_FABRIC, USE_DYNAMIC_ROUTING, NUM_LOCAL_SYNC_CORES>;

void kernel_main() {
    size_t rt_args_idx = 0;
    auto sync_config = SyncKernelConfig::build_from_args(rt_args_idx);

    // Clear test results area and mark as started
    tt::tt_fabric::fabric_tests::clear_test_results(
        sync_config.get_result_buffer_address(), sync_config.get_result_buffer_size());
    tt::tt_fabric::fabric_tests::write_test_status(sync_config.get_result_buffer_address(), TT_FABRIC_STATUS_STARTED);

    // Perform global sync (master sync core)
    sync_config.global_sync();

    // Perform local sync
    sync_config.local_sync();

    // Mark test as passed. TODO: might need a local sync after all test done (TBD).
    tt::tt_fabric::fabric_tests::write_test_status(sync_config.get_result_buffer_address(), TT_FABRIC_STATUS_PASS);
}
