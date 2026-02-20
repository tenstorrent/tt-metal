// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "api/debug/dprint.h"
#include "tt_fabric_test_kernels_utils.hpp"
#include "tt_metal/fabric/hw/inc/tt_fabric_mux_interface.hpp"

using namespace tt::tt_fabric::fabric_tests;

constexpr uint8_t IS_2D_FABRIC = get_compile_time_arg_val(0);
constexpr uint8_t NUM_FABRIC_CONNECTIONS = get_compile_time_arg_val(1);
constexpr uint8_t NUM_TRAFFIC_CONFIGS = get_compile_time_arg_val(2);
constexpr bool BENCHMARK_MODE = get_compile_time_arg_val(3);
constexpr bool LINE_SYNC = get_compile_time_arg_val(4);
constexpr uint8_t NUM_LOCAL_SYNC_CORES = get_compile_time_arg_val(5);
constexpr uint32_t KERNEL_CONFIG_BUFFER_SIZE = get_compile_time_arg_val(6);
constexpr bool HAS_MUX_CONNECTIONS = get_compile_time_arg_val(7);
constexpr uint8_t NUM_MUXES_TO_TERMINATE = get_compile_time_arg_val(8);

using SenderKernelConfigType =
    SenderKernelConfig<NUM_TRAFFIC_CONFIGS, IS_2D_FABRIC, LINE_SYNC, NUM_LOCAL_SYNC_CORES>;

// Static assertion to ensure this config fits within the allocated kernel config region
static_assert(
    sizeof(SenderKernelConfigType) <= KERNEL_CONFIG_BUFFER_SIZE,
    "SenderKernelConfig size exceeds allocated kernel config buffer size");

// Static assertion to ensure we don't exceed max fabric connections
static_assert(
    NUM_FABRIC_CONNECTIONS <= MAX_NUM_FABRIC_CONNECTIONS, "NUM_FABRIC_CONNECTIONS exceeds MAX_NUM_FABRIC_CONNECTIONS");

void kernel_main() {
    size_t rt_args_idx = 0;
    size_t local_args_idx = 0;  // Initialize local args index

    // Get kernel config address from runtime args
    CommonMemoryMap common_memory_map = CommonMemoryMap::build_from_args(rt_args_idx);
    uint32_t kernel_config_address = common_memory_map.kernel_config_base;

    // Use placement new to construct config in L1 memory (advances local_args_idx)
    auto* sender_config = new (reinterpret_cast<void*>(kernel_config_address))
        SenderKernelConfigType(SenderKernelConfigType::build_from_args(
            common_memory_map, rt_args_idx, local_args_idx, NUM_FABRIC_CONNECTIONS));

    // Build mux termination manager from local args (uses advanced local_args_idx)
    MuxTerminationManager<HAS_MUX_CONNECTIONS, NUM_MUXES_TO_TERMINATE> mux_termination_manager(
        local_args_idx, common_memory_map.mux_termination_sync_address);

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
    uint32_t loop_count = 0;

    // Round-robin packet sending: send one packet from each config per iteration
    uint64_t start_timestamp = get_timestamp();
    constexpr uint32_t PROGRESS_UPDATE_INTERVAL = 1000;  // Write progress every 1000 loops

    // Per-phase profiling accumulators (local vars → stay in registers)
    uint32_t wait_accum = 0;                // time in wait_for_empty_write_slot
    uint32_t advance_accum = 0;             // time in advance_buffer_slot_write_index
    uint32_t noc_accum = 0;                 // time in update_edm_buffer_free_slots (noc inline write)
    uint32_t loop_accum = 0;                // time in outer loop overhead
    uint32_t prev_t = get_timestamp_32b();  // dummy initial timestamp

    if constexpr (NUM_TRAFFIC_CONFIGS == 1 && BENCHMARK_MODE) {
        // Optimized single-config path: simple counted loop, no flag logic
        auto* traffic_config = sender_config->traffic_config_ptrs[0];
        auto* conn = static_cast<WorkerToFabricEdmSender*>(traffic_config->connection_ptr_);
        const uint32_t num_packets = traffic_config->metadata.num_packets;
        const uint32_t num_warmup = conn->num_buffers_per_channel;

        // Phase 1: Warmup — send actual headers to fill all buffer slots
        const uint32_t warmup_end = (num_packets < num_warmup) ? num_packets : num_warmup;
        for (uint32_t pkt = 0; pkt < warmup_end; pkt++) {
            traffic_config->template send_one_packet<BENCHMARK_MODE, false>(
                wait_accum, advance_accum, noc_accum, loop_accum, prev_t);
        }

        // Set up stateful NOC cmd buf for credit updates (one-time cost)
        conn->setup_credit_update_noc_state();

        // Phase 2: Steady state — credit-only sends with stateful NOC
        for (uint32_t pkt = warmup_end; pkt < num_packets; pkt++) {
            traffic_config->template send_one_packet<BENCHMARK_MODE, true>(
                wait_accum, advance_accum, noc_accum, loop_accum, prev_t);
        }
    } else {
        while (packets_left_to_send) {
            packets_left_to_send = false;

            for (uint8_t i = 0; i < NUM_TRAFFIC_CONFIGS; i++) {
                auto* traffic_config = sender_config->traffic_config_ptrs[i];
                if (!traffic_config->has_packets_to_send()) {
                    continue;
                }

                // Send one packet — accumulators passed by ref, stays in registers
                bool sent = traffic_config->template send_one_packet<BENCHMARK_MODE>(
                    wait_accum, advance_accum, noc_accum, loop_accum, prev_t);

                if (!sent) {
                    // Packet blocked (no credits) - keep trying
                    packets_left_to_send = true;
                    continue;
                }

                // Check if more packets remain
                packets_left_to_send |= traffic_config->has_packets_to_send();
            }

            loop_count++;

            // Periodically write progress updates (skip in BENCHMARK_MODE for performance)
            if constexpr (!BENCHMARK_MODE) {
                if (loop_count % PROGRESS_UPDATE_INTERVAL == 0) {
                    // Calculate total packets sent across all traffic configs
                    uint64_t progress_packets_sent = 0;
                    for (uint8_t i = 0; i < NUM_TRAFFIC_CONFIGS; i++) {
                        progress_packets_sent += sender_config->traffic_config_ptrs[i]->num_packets_processed;
                    }
                    write_test_packets(sender_config->get_result_buffer_address(), progress_packets_sent);
                }
            }
        }
    }

    sender_config->close_connections();

    // Local sync (as participant, not master) for end of sync first sync tells sync core to start global sync, second
    // sync is waiting for global sync done
    if constexpr (LINE_SYNC) {
        sender_config->local_sync(sync_iter++);
        sender_config->local_sync(sync_iter++);
    }

    uint64_t total_elapsed_cycles_outer_loop = get_timestamp() - start_timestamp;

    // Terminate muxes and wait for completion
    mux_termination_manager.terminate_muxes();

    // Collect results from all traffic configs
    for (uint8_t i = 0; i < NUM_TRAFFIC_CONFIGS; i++) {
        auto* traffic_config = sender_config->traffic_config_ptrs[i];
        total_packets_sent += traffic_config->num_packets_processed;
    }

    // Write test results
    write_test_cycles(sender_config->get_result_buffer_address(), total_elapsed_cycles_outer_loop);
    write_test_packets(sender_config->get_result_buffer_address(), total_packets_sent);

    // Write worker sender profiling to MISC region
    // Layout at MISC_INDEX: [wait_accum, advance_accum, noc_accum, loop_accum, num_packets]
    {
        auto* result_buffer = reinterpret_cast<tt_l1_ptr uint32_t*>(sender_config->get_result_buffer_address());
        uint32_t idx = TT_FABRIC_MISC_INDEX;
        result_buffer[idx++] = wait_accum;
        result_buffer[idx++] = advance_accum;
        result_buffer[idx++] = noc_accum;
        result_buffer[idx++] = loop_accum;
        result_buffer[idx++] = static_cast<uint32_t>(total_packets_sent);
    }

    write_test_status(sender_config->get_result_buffer_address(), TT_FABRIC_STATUS_PASS);

    noc_async_full_barrier();
}
