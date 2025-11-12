// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_fabric_test_kernels_utils.hpp"

using namespace tt::tt_fabric::fabric_tests;

constexpr uint8_t IS_2D_FABRIC = get_compile_time_arg_val(0);
constexpr uint8_t USE_DYNAMIC_ROUTING = get_compile_time_arg_val(1);
constexpr uint8_t NUM_SYNC_FABRIC_CONNECTIONS = get_compile_time_arg_val(2);
constexpr uint8_t NUM_LOCAL_SYNC_CORES = get_compile_time_arg_val(3);
constexpr uint32_t KERNEL_CONFIG_BUFFER_SIZE = get_compile_time_arg_val(4);
constexpr bool HAS_MUX_CONNECTIONS = get_compile_time_arg_val(5);
constexpr uint8_t NUM_MUXES_TO_TERMINATE = get_compile_time_arg_val(6);

using SyncKernelConfigType =
    SyncKernelConfig<NUM_SYNC_FABRIC_CONNECTIONS, IS_2D_FABRIC, USE_DYNAMIC_ROUTING, NUM_LOCAL_SYNC_CORES>;

// Static assertion to ensure this config fits within the allocated kernel config region
static_assert(
    sizeof(SyncKernelConfigType) <= KERNEL_CONFIG_BUFFER_SIZE,
    "SyncKernelConfig size exceeds allocated kernel config buffer size");

void kernel_main() {
    size_t rt_args_idx = 0;
    size_t local_args_idx = 0;

    // Get kernel config address from runtime args
    CommonMemoryMap common_memory_map = CommonMemoryMap::build_from_args(rt_args_idx);
    uint32_t kernel_config_address = common_memory_map.kernel_config_base;

    // Use placement new to construct config in L1 memory
    auto* sync_config = new (reinterpret_cast<void*>(kernel_config_address))
        SyncKernelConfigType(SyncKernelConfigType::build_from_args(common_memory_map, rt_args_idx, local_args_idx));

    // Build mux termination manager from local args (uses advanced local_args_idx)
    MuxTerminationManager<HAS_MUX_CONNECTIONS, NUM_MUXES_TO_TERMINATE> mux_termination_manager(
        local_args_idx, common_memory_map.mux_termination_sync_address);

    // Clear test results area and mark as started
    clear_test_results(sync_config->get_result_buffer_address(), sync_config->get_result_buffer_size());
    write_test_status(sync_config->get_result_buffer_address(), TT_FABRIC_STATUS_STARTED);

    // Perform global sync (master sync core) for start of sync
    uint8_t local_sync_iter = 0, global_sync_iter = 0;
    sync_config->global_sync(global_sync_iter++);

    // Perform local sync for start of sync
    sync_config->local_sync(local_sync_iter++);

    // Perform local sync for end of sync
    // first sync tells sync core to start global sync, second sync is waiting for global sync done
    sync_config->local_sync(local_sync_iter++);

    // Perform global sync (master sync core) for end of sync
    sync_config->global_sync(global_sync_iter++);

    // Perform local sync for end of sync
    sync_config->local_sync(local_sync_iter++);

    // Terminate muxes if this core uses mux connections
    mux_termination_manager.terminate_muxes();

    // Mark test as passed. TODO: might need a local sync after all test done (TBD).
    write_test_status(sync_config->get_result_buffer_address(), TT_FABRIC_STATUS_PASS);

    // Make sure all the noc txns are done
    noc_async_full_barrier();
}
