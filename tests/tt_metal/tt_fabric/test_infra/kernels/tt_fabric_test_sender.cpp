// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
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
constexpr uint8_t VC_ID = get_compile_time_arg_val(9);

// Select the EDM sender adapter type based on VC_ID:
// VC0/VC1 use WorkerToFabricEdmSender (stream ID 22), VC2 uses WorkerToFabricEdmSenderVC2 (stream ID 30)
using EdmSenderType = std::conditional_t<VC_ID == 2, WorkerToFabricEdmSenderVC2, WorkerToFabricEdmSender>;

using SenderKernelConfigType =
    SenderKernelConfig<NUM_TRAFFIC_CONFIGS, IS_2D_FABRIC, LINE_SYNC, NUM_LOCAL_SYNC_CORES, EdmSenderType>;

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

    auto send_packets_stateful = [&]() {
        auto* traffic_config = sender_config->traffic_config_ptrs[0];
        auto* conn = static_cast<EdmSenderType*>(traffic_config->connection_ptr_);
        const uint32_t num_packets = traffic_config->metadata.num_packets;
        const uint32_t num_warmup = conn->num_buffers_per_channel;

        traffic_config->template send_packets_stateful<BENCHMARK_MODE>(num_packets, num_warmup);
    };

    auto send_one_packet = [&](uint8_t idx) {
        auto* traffic_config = sender_config->traffic_config_ptrs[idx];
        if (!traffic_config->has_packets_to_send()) {
            return;
        }

        // Send one packet (credit management is automatic, inside send_one_packet)
        bool sent = traffic_config->template send_one_packet<BENCHMARK_MODE>();

        if (!sent) {
            // Packet blocked (no credits) - keep trying
            packets_left_to_send = true;
            return;
        }

        // Check if more packets remain
        packets_left_to_send |= traffic_config->has_packets_to_send();
    };

    // Round-robin packet sending: send one packet from each config per iteration
    uint64_t start_timestamp = get_timestamp();
    constexpr uint32_t PROGRESS_UPDATE_INTERVAL = 1000;  // Write progress every 1000 loops

    if constexpr (NUM_TRAFFIC_CONFIGS == 1 && BENCHMARK_MODE) {
        {
            // CRAQ #52 diag: the credit window as seeded at connection open (before any send).
            auto* conn0 = static_cast<EdmSenderType*>(sender_config->traffic_config_ptrs[0]->connection_ptr_);
            auto* rb0 = reinterpret_cast<tt_l1_ptr uint32_t*>(sender_config->get_result_buffer_address());
            invalidate_l1_cache();
            rb0[TT_FABRIC_MISC_INDEX + 18] = conn0->buffer_slot_write_counter.counter;
            rb0[TT_FABRIC_MISC_INDEX + 19] = *conn0->edm_buffer_local_free_slots_read_ptr;
        }
        // Stateful sends optimized for single traffic config tests
        send_packets_stateful();
    } else {
        while (packets_left_to_send) {
            packets_left_to_send = false;

            // Round robin sends
            for (uint8_t i = 0; i < NUM_TRAFFIC_CONFIGS; i++) {
                send_one_packet(i);
            }

            loop_count++;

            // Periodically write progress updates (skip in BENCHMARK_MODE for performance)
            if constexpr (!BENCHMARK_MODE) {
                if (loop_count % PROGRESS_UPDATE_INTERVAL == 0) {
                    auto* per_config_results = get_per_config_results(sender_config->get_result_buffer_address());
                    uint64_t progress_packets_sent = 0;
                    for (uint8_t i = 0; i < NUM_TRAFFIC_CONFIGS; i++) {
                        uint64_t config_packets = sender_config->traffic_config_ptrs[i]->num_packets_processed;
                        progress_packets_sent += config_packets;
                        write_per_config_result(&per_config_results[i], config_packets);
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

    // Aggregate the per-config send counts for the final test_packets write.
    // The per-config flush itself is skipped in BENCHMARK_MODE to avoid extra
    // L1 writes on the exit path (per-config progress monitoring is disabled
    // in benchmark mode anyway).
    for (uint8_t i = 0; i < NUM_TRAFFIC_CONFIGS; i++) {
        total_packets_sent += sender_config->traffic_config_ptrs[i]->num_packets_processed;
    }
    if constexpr (!BENCHMARK_MODE) {
        auto* per_config_results = get_per_config_results(sender_config->get_result_buffer_address());
        for (uint8_t i = 0; i < NUM_TRAFFIC_CONFIGS; i++) {
            uint64_t config_packets = sender_config->traffic_config_ptrs[i]->num_packets_processed;
            write_per_config_result(&per_config_results[i], config_packets);
        }
    }

    // Worker diagnostics (CRAQ #52): MISC+9/+10 wait cycles, +11/+12 signal cycles, +13 packets that waited.
    {
        uint64_t wait_cycles = 0;
        uint64_t signal_cycles = 0;
        uint32_t waited = 0;
        for (uint8_t i = 0; i < NUM_TRAFFIC_CONFIGS; i++) {
            wait_cycles += sender_config->traffic_config_ptrs[i]->diag_wait_cycles;
            signal_cycles += sender_config->traffic_config_ptrs[i]->diag_signal_cycles;
            waited += sender_config->traffic_config_ptrs[i]->diag_waited;
        }
        auto* rb = reinterpret_cast<tt_l1_ptr uint32_t*>(sender_config->get_result_buffer_address());
        rb[TT_FABRIC_MISC_INDEX + 9] = static_cast<uint32_t>(wait_cycles);
        rb[TT_FABRIC_MISC_INDEX + 10] = static_cast<uint32_t>(wait_cycles >> 32);
        rb[TT_FABRIC_MISC_INDEX + 11] = static_cast<uint32_t>(signal_cycles);
        rb[TT_FABRIC_MISC_INDEX + 12] = static_cast<uint32_t>(signal_cycles >> 32);
        rb[TT_FABRIC_MISC_INDEX + 13] = waited;
        // Credit window as the worker sees it at the end: its write counter, the router's last read
        // counter, and the slot count (effective window = num_buffers - (wr - rd)).
        if constexpr (NUM_TRAFFIC_CONFIGS == 1 && BENCHMARK_MODE) {
            auto* conn = static_cast<EdmSenderType*>(sender_config->traffic_config_ptrs[0]->connection_ptr_);
            rb[TT_FABRIC_MISC_INDEX + 14] = conn->buffer_slot_write_counter.counter;
            rb[TT_FABRIC_MISC_INDEX + 15] = *conn->edm_buffer_local_free_slots_read_ptr;
            rb[TT_FABRIC_MISC_INDEX + 16] = conn->num_buffers_per_channel;
            // Sample the read counter again after a long settle (~50 us) to separate in-flight from leaked credits.
            {
                const uint64_t t0 = get_timestamp();
                while (get_timestamp() - t0 < 50000) {
                }
                invalidate_l1_cache();
                rb[TT_FABRIC_MISC_INDEX + 17] = *conn->edm_buffer_local_free_slots_read_ptr;
            }
        }
    }
    // Write test results
    write_test_cycles(sender_config->get_result_buffer_address(), total_elapsed_cycles_outer_loop);
    write_test_packets(sender_config->get_result_buffer_address(), total_packets_sent);
    write_test_status(sender_config->get_result_buffer_address(), TT_FABRIC_STATUS_PASS);

    noc_async_full_barrier();
}
