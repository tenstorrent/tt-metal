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
    [[maybe_unused]] uint64_t perf_t0 = PERF_BD_NOW();
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
        // Payloads arrive from the compute kernel in strict round-robin order, so a config that takes them is waited on
        // rather than skipped when it has no credits; skipping would hand the next such config this config's page.
        while (!sent && traffic_config->metadata.sfpu_payload) {
            sent = traffic_config->template send_one_packet<BENCHMARK_MODE>();
        }

        if (!sent) {
            // Packet blocked (no credits) - keep trying
            packets_left_to_send = true;
            return;
        }

        // Check if more packets remain
        packets_left_to_send |= traffic_config->has_packets_to_send();
    };

    PERF_HOT(perf_hot_calibrate();)
    // Round-robin packet sending: send one packet from each config per iteration
    uint64_t start_timestamp = get_timestamp();
    PERF_BD_SUMMARY_SET_SINCE(0, perf_t0);  // S_SETUP
    constexpr uint32_t PROGRESS_UPDATE_INTERVAL = 1000;  // Write progress every 1000 loops

#if defined(FABRIC_TEST_SENDER_SKIP) && !defined(FABRIC_TEST_SENDER_SLOT_CACHE)
#error "FABRIC_TEST_SENDER_SKIP needs FABRIC_TEST_SENDER_SLOT_CACHE (skips are decided per connection)"
#endif
#ifdef FABRIC_TEST_SENDER_HOT_STATE
    // Per-config hot state lives on the stack, which shares the core's 8 KB local memory with firmware data.
    constexpr bool USE_HOT_STATE =
        !BENCHMARK_MODE && NUM_TRAFFIC_CONFIGS * (sizeof(SenderHotState) + sizeof(void*)) <= 2048;
#else
    constexpr bool USE_HOT_STATE = false;
#endif

    if constexpr (NUM_TRAFFIC_CONFIGS == 1 && BENCHMARK_MODE) {
        // Stateful sends optimized for single traffic config tests
        send_packets_stateful();
    } else if constexpr (USE_HOT_STATE) {
        // Send from local memory: a copy of the connection array (the adapters' slot and credit state) and each
        // config's SenderHotState. Both are written back to L1 before the connections are closed.
        using TrafficConfigType = SenderKernelTrafficConfig<EdmSenderType>;
        FabricConnectionArray<EdmSenderType> local_connections = sender_config->connections;
        TrafficConfigType* configs[NUM_TRAFFIC_CONFIGS];
        SenderHotState hot[NUM_TRAFFIC_CONFIGS];
        for (uint8_t i = 0; i < NUM_TRAFFIC_CONFIGS; i++) {
            configs[i] = sender_config->traffic_config_ptrs[i];
            configs[i]->rebind_connections(&local_connections);
            hot[i] = configs[i]->make_hot_state();
        }
#ifdef FABRIC_TEST_SENDER_BATCH
        sender_batch_sync();
#endif
#ifdef FABRIC_TEST_SENDER_CMDBUFS
        sender_cmdbuf_init();
#endif

        // Free slots per connection, cached across packets (FABRIC_TEST_SENDER_SLOT_CACHE), and the connections found
        // full in the current round, whose skippable configs are passed over without another check
        // (FABRIC_TEST_SENDER_SKIP).
        uint32_t slot_budget[MAX_NUM_FABRIC_CONNECTIONS] = {};
        PERF_SCHED(
            uint64_t sched[PERF_SCHED_NUM_COUNTERS] = {}; uint32_t sched_finish[NUM_TRAFFIC_CONFIGS] = {};
            uint32_t sched_misses[NUM_TRAFFIC_CONFIGS] = {};
            const uint32_t sched_t0 = get_timestamp_32b();)
        auto write_hot_progress = [&]() {
            auto* per_config_results = get_per_config_results(sender_config->get_result_buffer_address());
            uint64_t progress_packets_sent = 0;
            for (uint8_t i = 0; i < NUM_TRAFFIC_CONFIGS; i++) {
                const uint64_t config_packets = configs[i]->metadata.num_packets - hot[i].packets_left;
                progress_packets_sent += config_packets;
                write_per_config_result(&per_config_results[i], config_packets);
            }
            write_test_packets(sender_config->get_result_buffer_address(), progress_packets_sent);
        };

#ifdef FABRIC_TEST_SENDER_SKIP
        // When every config may be skipped, send connection by connection: each connection's free slots go to its
        // configs in rotation, from a cursor kept across rounds, so no config on a busy connection is starved by the
        // configs ahead of it. A full connection is passed over until its next turn.
        bool rotate = true;
        for (uint8_t i = 0; i < NUM_TRAFFIC_CONFIGS; i++) {
            rotate &= hot[i].mode != SenderHotState::HOT_FALLBACK && TrafficConfigType::can_skip_when_full(hot[i]);
        }
#else
        constexpr bool rotate = false;
#endif
        if (rotate) {
#ifdef FABRIC_TEST_SENDER_SKIP
            // Configs grouped by connection: connection c's configs are order[conn_start[c] .. conn_start[c + 1]).
            const uint8_t num_connections = local_connections.num_connections;
            uint8_t order[NUM_TRAFFIC_CONFIGS];
            uint8_t conn_start[MAX_NUM_FABRIC_CONNECTIONS + 1] = {};
            uint8_t conn_fill[MAX_NUM_FABRIC_CONNECTIONS];
            uint8_t conn_cursor[MAX_NUM_FABRIC_CONNECTIONS] = {};
            uint8_t conn_active[MAX_NUM_FABRIC_CONNECTIONS] = {};
            uint32_t configs_active = 0;
            for (uint8_t i = 0; i < NUM_TRAFFIC_CONFIGS; i++) {
                conn_start[hot[i].conn_idx + 1]++;
            }
            for (uint8_t c = 0; c < num_connections; c++) {
                conn_start[c + 1] += conn_start[c];
                conn_fill[c] = conn_start[c];
            }
            for (uint8_t i = 0; i < NUM_TRAFFIC_CONFIGS; i++) {
                order[conn_fill[hot[i].conn_idx]++] = i;
                if (hot[i].packets_left > 0) {
                    conn_active[hot[i].conn_idx]++;
                    configs_active++;
                }
            }

            // Diagnostics: FABRIC_TEST_SENDER_ROT_WAIT waits on a full connection instead of passing it over, and
            // FABRIC_TEST_SENDER_ROT_BURST caps the packets sent per connection per turn.
#ifdef FABRIC_TEST_SENDER_ROT_BURST
            constexpr uint32_t ROT_BURST = FABRIC_TEST_SENDER_ROT_BURST;
#else
            constexpr uint32_t ROT_BURST = UINT32_MAX;
#endif
            while (configs_active > 0) {
                PERF_SCHED(const uint32_t pass_t0 = get_timestamp_32b(); const uint64_t pass_packets = sched[SC_PACKETS];)
                for (uint8_t c = 0; c < num_connections; c++) {
                    if (conn_active[c] == 0) {
                        continue;
                    }
                    const uint8_t base = conn_start[c];
                    const uint8_t count = conn_start[c + 1] - base;
                    if (slot_budget[c] == 0) {
                        slot_budget[c] = TrafficConfigType::connection_free_slots(hot[order[base]]);
                        PERF_SCHED(sched[SC_RELOADS]++; sched[SC_BUDGET_SUM] += slot_budget[c];)
                        if (slot_budget[c] == 0) {
                            PERF_SCHED(sched[SC_RELOADS_FULL]++;)
#ifdef FABRIC_TEST_SENDER_ROT_WAIT
                            PERF_SCHED(const uint32_t wait_t0 = get_timestamp_32b();)
                            do {
                                slot_budget[c] = TrafficConfigType::connection_free_slots(hot[order[base]]);
                            } while (slot_budget[c] == 0);
                            PERF_SCHED(sched[SC_IDLE_CYCLES] += get_timestamp_32b() - wait_t0;)
#else
                            PERF_BD_SUMMARY_INC(5);  // counts connection skips with the credit-blocked attempts
                            continue;
#endif
                        }
                    }
                    // Hand out the free slots in rotation; stop after a full lap without a send (no credits).
                    uint8_t misses = 0;
                    uint32_t burst = ROT_BURST;
                    while (slot_budget[c] > 0 && conn_active[c] > 0 && misses < count && burst > 0) {
                        const uint8_t k = conn_cursor[c];
                        conn_cursor[c] = (k + 1 == count) ? 0 : k + 1;
                        const uint8_t i = order[base + k];
                        SenderHotState& h = hot[i];
                        if (h.packets_left == 0) {
                            misses++;
                            continue;
                        }
                        if (configs[i]->send_one_packet_hot(h, slot_budget[c]) != HotSendResult::SENT) {
                            PERF_SCHED(sched_misses[i]++; sched[SC_CREDIT_MISSES]++;)
                            misses++;
                            continue;
                        }
                        misses = 0;
                        burst--;
                        PERF_SCHED(sched[SC_PACKETS]++;)
                        if (h.packets_left == 0) {
                            PERF_SCHED(sched_finish[i] = get_timestamp_32b() - sched_t0;)
                            conn_active[c]--;
                            configs_active--;
                        }
                    }
                }

                PERF_SCHED(sched[SC_PASSES]++; if (sched[SC_PACKETS] == pass_packets) {
                    sched[SC_IDLE_PASSES]++;
                    sched[SC_IDLE_CYCLES] += get_timestamp_32b() - pass_t0;
                })
                loop_count++;
                if (loop_count % PROGRESS_UPDATE_INTERVAL == 0) {
                    write_hot_progress();
                }
            }
#endif
        } else {
            while (packets_left_to_send) {
                packets_left_to_send = false;
                [[maybe_unused]] uint32_t full_connections = 0;
                PERF_SCHED(const uint32_t pass_t0 = get_timestamp_32b(); const uint64_t pass_packets = sched[SC_PACKETS];)
                for (uint8_t i = 0; i < NUM_TRAFFIC_CONFIGS; i++) {
                    SenderHotState& h = hot[i];
                    if (h.packets_left == 0) {
                        continue;
                    }
                    bool sent;
                    if (h.mode == SenderHotState::HOT_FALLBACK) {
#ifdef FABRIC_TEST_SENDER_CMDBUFS
                        sender_credits_flush();
#endif
#ifdef FABRIC_TEST_SENDER_BATCH
                        // The fallback path pops the payload CB itself: return the batched pages first.
                        sender_batch_flush();
#endif
                        sent = configs[i]->template send_one_packet<BENCHMARK_MODE>();
                        while (!sent && h.sfpu_payload) {
                            sent = configs[i]->template send_one_packet<BENCHMARK_MODE>();
                        }
#ifdef FABRIC_TEST_SENDER_BATCH
                        sender_batch_sync();
#endif
#ifdef FABRIC_TEST_SENDER_CMDBUFS
                        sender_cmdbuf_init();  // the fallback path programs the command buffers itself
#endif
                        h.packets_left -= sent;
                        // The fallback path takes slots without counting them: re-read its connections next time.
                        for (uint8_t j = 0; j < configs[i]->num_connections_; j++) {
                            slot_budget[configs[i]->connection_indices_[j]] = 0;
                        }
                    } else {
#ifdef FABRIC_TEST_SENDER_SKIP
                        if ((full_connections >> h.conn_idx) & 1 && TrafficConfigType::can_skip_when_full(h)) {
                            packets_left_to_send = true;
                            continue;
                        }
#endif
                        HotSendResult result = configs[i]->send_one_packet_hot(h, slot_budget[h.conn_idx]);
                        // Strict round-robin for configs that take SFPU pages, as in send_one_packet() above.
                        while (result != HotSendResult::SENT && h.sfpu_payload) {
                            result = configs[i]->send_one_packet_hot(h, slot_budget[h.conn_idx]);
                        }
                        if (result == HotSendResult::CONNECTION_FULL) {
                            full_connections |= 1u << h.conn_idx;
                        }
                        PERF_SCHED(if (result == HotSendResult::NO_CREDITS) {
                            sched_misses[i]++;
                            sched[SC_CREDIT_MISSES]++;
                        })
                        sent = result == HotSendResult::SENT;
                    }
                    PERF_SCHED(if (sent) {
                        sched[SC_PACKETS]++;
                        if (h.packets_left == 0) {
                            sched_finish[i] = get_timestamp_32b() - sched_t0;
                        }
                    })
                    packets_left_to_send |= !sent || h.packets_left > 0;
                }

                PERF_SCHED(sched[SC_PASSES]++; if (sched[SC_PACKETS] == pass_packets) {
                    sched[SC_IDLE_PASSES]++;
                    sched[SC_IDLE_CYCLES] += get_timestamp_32b() - pass_t0;
                })
                loop_count++;
                if (loop_count % PROGRESS_UPDATE_INTERVAL == 0) {
                    write_hot_progress();
                }
            }
        }

#ifdef FABRIC_TEST_SENDER_CMDBUFS
        sender_credits_flush();
#endif
#ifdef FABRIC_TEST_SENDER_BATCH
        sender_batch_flush();
#endif
        sender_config->connections = local_connections;
        for (uint8_t i = 0; i < NUM_TRAFFIC_CONFIGS; i++) {
            configs[i]->write_back_hot_state(hot[i]);
            configs[i]->rebind_connections(&sender_config->connections);
        }
        PERF_SCHED(write_sender_sched(
                       sender_config->get_result_buffer_address(), sched, sched_finish, sched_misses, NUM_TRAFFIC_CONFIGS);)
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

    PERF_BD_SET_SINCE64(1, start_timestamp);  // S_LOOP
    perf_t0 = PERF_BD_NOW();
    sender_config->close_connections();

    // Local sync (as participant, not master) for end of sync first sync tells sync core to start global sync, second
    // sync is waiting for global sync done
    if constexpr (LINE_SYNC) {
        sender_config->local_sync(sync_iter++);
        sender_config->local_sync(sync_iter++);
    }

    uint64_t total_elapsed_cycles_outer_loop = get_timestamp() - start_timestamp;
    PERF_BD_SUMMARY_SET_SINCE(6, perf_t0);  // S_CLOSE

    // Terminate muxes and wait for completion
    perf_t0 = PERF_BD_NOW();
    mux_termination_manager.terminate_muxes();
    PERF_BD_SUMMARY_SET_SINCE(7, perf_t0);  // S_MUX_TERM

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

    // Write test results
    write_test_cycles(sender_config->get_result_buffer_address(), total_elapsed_cycles_outer_loop);
    write_perf_breakdown(sender_config->get_result_buffer_address());
    PERF_HOT(write_perf_hot_detail(sender_config->get_result_buffer_address());)
    write_test_packets(sender_config->get_result_buffer_address(), total_packets_sent);
    write_test_status(sender_config->get_result_buffer_address(), TT_FABRIC_STATUS_PASS);

    noc_async_full_barrier();
}
