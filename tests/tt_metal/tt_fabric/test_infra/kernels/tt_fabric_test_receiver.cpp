// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/debug/dprint.h"
#include "tt_fabric_test_kernels_utils.hpp"
#include "tt_metal/fabric/hw/inc/tt_fabric_mux_interface.hpp"

using namespace tt::tt_fabric::fabric_tests;

constexpr uint8_t IS_2D_FABRIC = get_compile_time_arg_val(0);
constexpr uint8_t NUM_TRAFFIC_CONFIGS = get_compile_time_arg_val(1);
constexpr bool BENCHMARK_MODE = get_compile_time_arg_val(2);
constexpr uint32_t KERNEL_CONFIG_BUFFER_SIZE = get_compile_time_arg_val(3);
constexpr uint8_t NUM_CREDIT_CONNECTIONS = get_compile_time_arg_val(4);
constexpr bool HAS_MUX_CONNECTIONS = get_compile_time_arg_val(5);
constexpr uint8_t NUM_MUXES_TO_TERMINATE = get_compile_time_arg_val(6);

using ReceiverKernelConfigType = ReceiverKernelConfig<NUM_TRAFFIC_CONFIGS, NUM_CREDIT_CONNECTIONS, IS_2D_FABRIC>;

// Static assertion to ensure this config fits within the allocated kernel config region
static_assert(
    sizeof(ReceiverKernelConfigType) <= KERNEL_CONFIG_BUFFER_SIZE,
    "ReceiverKernelConfig size exceeds allocated kernel config buffer size");

// Static assertion to ensure we don't exceed max credit connections
static_assert(
    NUM_CREDIT_CONNECTIONS <= MAX_NUM_FABRIC_CONNECTIONS, "NUM_CREDIT_CONNECTIONS exceeds MAX_NUM_FABRIC_CONNECTIONS");

// Packets handed to the compute verifier (tt_fabric_test_payload_verify.cpp) whose verdicts are still outstanding.
constexpr uint32_t SFPU_VERIFY_DEPTH = FABRIC_TEST_SFPU_VERIFY_DEPTH;

// Hands the packet at config_idx's current ring slot to the core's compute engine without waiting for its verdict.
inline void sfpu_verify_issue(uint8_t config_idx, const TrafficValidationConfigBase& config) {
    cb_reserve_back(SFPU_VERIFY_REQ_CB_INDEX, 1);
    auto* request = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(SFPU_VERIFY_REQ_CB_INDEX));
    request[SFPU_VERIFY_REQ_OP] = SFPU_VERIFY_OP_CHECK;
    request[SFPU_VERIFY_REQ_CONFIG] = config_idx;
    request[SFPU_VERIFY_REQ_SEED] = config.metadata.seed;
    request[SFPU_VERIFY_REQ_WORDS] = config.payload_words;
    request[SFPU_VERIFY_REQ_SEGMENTS] = config.num_payload_segments;
    for (uint32_t s = 0; s < config.num_payload_segments; s++) {
        request[SFPU_VERIFY_REQ_SEGMENT_END + 2 * s] = config.payload_segment_end[s];
        request[SFPU_VERIFY_REQ_SEGMENT_DELTA + 2 * s] = config.payload_segment_delta[s];
    }
    cb_push_back(SFPU_VERIFY_REQ_CB_INDEX, 1);

    // The data CB lies over the config's receive ring; pushing a page just publishes the slot the packet is in.
    const uint32_t data_cb = SFPU_VERIFY_DATA_CB_BASE + config_idx;
    cb_reserve_back(data_cb, 1);
    cb_push_back(data_cb, 1);
}

inline bool sfpu_verdict_ready() { return cb_pages_available_at_front(SFPU_VERIFY_FLAG_CB_INDEX, 1); }

// The oldest outstanding verdict (verdicts come back in issue order): true when every word matched.
inline bool sfpu_take_verdict() {
    cb_wait_front(SFPU_VERIFY_FLAG_CB_INDEX, 1);
    invalidate_l1_cache();
    auto* flag = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_read_ptr(SFPU_VERIFY_FLAG_CB_INDEX));
    uint32_t mismatch = 0;
    for (uint32_t lane = 0; lane < SFPU_VERIFY_FLAG_LANES; lane++) {
        mismatch |= flag[2 * lane];
    }
    cb_pop_front(SFPU_VERIFY_FLAG_CB_INDEX, 1);
    return mismatch == 0;
}

inline void sfpu_verify_exit() {
    cb_reserve_back(SFPU_VERIFY_REQ_CB_INDEX, 1);
    auto* request = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(SFPU_VERIFY_REQ_CB_INDEX));
    request[SFPU_VERIFY_REQ_OP] = SFPU_VERIFY_OP_EXIT;
    cb_push_back(SFPU_VERIFY_REQ_CB_INDEX, 1);
}

void kernel_main() {
    [[maybe_unused]] uint64_t perf_t0 = PERF_BD_NOW();
    size_t rt_args_idx = 0;
    size_t local_args_idx = 0;

    // Get kernel config address from runtime args
    CommonMemoryMap common_memory_map = CommonMemoryMap::build_from_args(rt_args_idx);
    uint32_t kernel_config_address = common_memory_map.kernel_config_base;

    // Use placement new to construct config in L1 memory (advances local_args_idx)
    auto* receiver_config = new (reinterpret_cast<void*>(kernel_config_address)) ReceiverKernelConfigType(
        ReceiverKernelConfigType::build_from_args(common_memory_map, rt_args_idx, local_args_idx));

    // Build mux termination manager from local args (uses advanced local_args_idx)
    MuxTerminationManager<HAS_MUX_CONNECTIONS, NUM_MUXES_TO_TERMINATE> mux_termination_manager(
        local_args_idx, common_memory_map.mux_termination_sync_address);

    // Clear test results area and mark as started
    clear_test_results(receiver_config->get_result_buffer_address(), receiver_config->get_result_buffer_size());
    write_test_status(receiver_config->get_result_buffer_address(), TT_FABRIC_STATUS_STARTED);

    // Open credit connections (no-op if NUM_CREDIT_CONNECTIONS == 0)
    receiver_config->open_credit_connections();

    bool failed = false;
    uint64_t total_packets_received = 0;

    constexpr uint32_t PROGRESS_UPDATE_INTERVAL = 1000;

    PERF_BD_SET_SINCE(0, perf_t0);  // R_SETUP
    [[maybe_unused]] const uint64_t loop_t0 = PERF_BD_NOW64();
    [[maybe_unused]] const uint32_t loop_t0_32 = PERF_BD_NOW();
    [[maybe_unused]] bool seen_first_packet = false;

    auto record_progress = [&]() {
        if (total_packets_received % PROGRESS_UPDATE_INTERVAL == 0) {
            auto* per_config_results = get_per_config_results(receiver_config->get_result_buffer_address());
            uint64_t progress_packets_received = 0;
            for (uint8_t j = 0; j < NUM_TRAFFIC_CONFIGS; j++) {
                uint64_t config_packets = receiver_config->traffic_configs()[j]->num_packets_processed;
                progress_packets_received += config_packets;
                write_per_config_result(&per_config_results[j], config_packets);
            }
            write_test_packets(receiver_config->get_result_buffer_address(), progress_packets_received);
        }
    };

    // After a mismatch the receiver keeps consuming packets without checking them, so it keeps returning credits
    // and its senders finish; the mismatch is reported in the final status instead of the test hanging on credits.
    // The host starts a compute verifier on this core when any of its configs is checked there.
    bool has_verifier = false;
    for (uint8_t i = 0; i < NUM_TRAFFIC_CONFIGS; i++) {
        has_verifier |= receiver_config->traffic_configs()[i]->metadata.sfpu_verify;
    }
    if (!BENCHMARK_MODE && has_verifier) {
        // Pipelined: a packet is issued to the compute verifier as soon as it lands and its slot cursor moves on, so
        // the next packet can be polled while earlier ones are checked. Its credit is returned only when its verdict is
        // taken (retire), so the sender cannot reuse the slot before it has been checked.
        uint8_t in_flight[SFPU_VERIFY_DEPTH];
        uint32_t in_flight_head = 0;
        uint32_t in_flight_count = 0;
        auto retire_oldest = [&]() {
            [[maybe_unused]] uint64_t verdict_t0 = PERF_BD_NOW();
            failed |= !sfpu_take_verdict();
            PERF_BD_ADD_SINCE(2, verdict_t0);  // R_VALIDATE (waiting for verdicts)
            auto* config = receiver_config->traffic_configs()[in_flight[in_flight_head]];
            in_flight_head = (in_flight_head + 1) % SFPU_VERIFY_DEPTH;
            in_flight_count--;
            config->retire();
            total_packets_received++;
            record_progress();
        };

        bool packets_left_to_issue = true;
        while (packets_left_to_issue) {
            packets_left_to_issue = false;

            for (uint8_t i = 0; i < NUM_TRAFFIC_CONFIGS; i++) {
                auto* traffic_config = receiver_config->traffic_configs()[i];
                if (!traffic_config->has_packets_to_issue()) {
                    continue;
                }
                packets_left_to_issue = true;

                while (in_flight_count > 0 && sfpu_verdict_ready()) {
                    retire_oldest();
                }

                if (!traffic_config->poll()) {
                    PERF_BD_INC(5);  // R_POLL_MISS
                    continue;
                }
#ifdef FABRIC_TEST_PERF_BREAKDOWN
                if (!seen_first_packet) {
                    seen_first_packet = true;
                    PERF_BD_SET_SINCE(4, loop_t0_32);  // R_FIRST_PKT
                }
#endif
                if (!traffic_config->metadata.sfpu_verify) {
                    // A config the verifier does not take is checked here, as on a core without one.
                    [[maybe_unused]] uint64_t validate_t0 = PERF_BD_NOW();
                    if (!failed && !traffic_config->validate()) {
                        failed = true;
                    }
                    PERF_BD_ADD_SINCE(2, validate_t0);  // R_VALIDATE
                    traffic_config->advance();
                    total_packets_received++;
                    record_progress();
                    continue;
                }
                if (!failed && in_flight_count == SFPU_VERIFY_DEPTH) {
                    retire_oldest();
                }

                [[maybe_unused]] uint64_t advance_t0 = PERF_BD_NOW();
                if (failed) {
                    traffic_config->advance();
                    total_packets_received++;
                    record_progress();
                } else {
                    sfpu_verify_issue(i, *traffic_config);
                    traffic_config->advance_cursor();
                    in_flight[(in_flight_head + in_flight_count) % SFPU_VERIFY_DEPTH] = i;
                    in_flight_count++;
                }
                PERF_BD_ADD_SINCE(3, advance_t0);  // R_ADVANCE (issue + cursor, or drain)
            }
        }
        while (in_flight_count > 0) {
            retire_oldest();
        }
    } else {
        bool packets_left_to_validate = true;
        while (packets_left_to_validate) {
            packets_left_to_validate = false;

            for (uint8_t i = 0; i < NUM_TRAFFIC_CONFIGS; i++) {
                auto* traffic_config = receiver_config->traffic_configs()[i];

                if constexpr (!BENCHMARK_MODE) {
                    if (!traffic_config->has_packets_to_validate()) {
                        continue;
                    }

                    // if we are here, this means that we have atleast 1 packet left to validate
                    packets_left_to_validate = true;

                    bool got_new_data = traffic_config->poll();
                    if (!got_new_data) {
                        PERF_BD_INC(5);  // R_POLL_MISS
                        continue;
                    }
#ifdef FABRIC_TEST_PERF_BREAKDOWN
                    if (!seen_first_packet) {
                        seen_first_packet = true;
                        PERF_BD_SET_SINCE(4, loop_t0_32);  // R_FIRST_PKT
                    }
#endif

                    [[maybe_unused]] uint64_t validate_t0 = PERF_BD_NOW();
                    if (!failed && !traffic_config->validate()) {
                        failed = true;
                    }
                    PERF_BD_ADD_SINCE(2, validate_t0);  // R_VALIDATE

                    [[maybe_unused]] uint64_t advance_t0 = PERF_BD_NOW();
                    traffic_config->advance();         // Automatically handles credit return
                    PERF_BD_ADD_SINCE(3, advance_t0);  // R_ADVANCE
                    total_packets_received++;
                    record_progress();

                    packets_left_to_validate |= traffic_config->has_packets_to_validate();
                } else {
                    total_packets_received += traffic_config->metadata.num_packets;
                }
            }
        }
    }

    PERF_BD_SET_SINCE64(1, loop_t0);  // R_LOOP

    if (has_verifier) {
        // Sent on every exit path, including a mismatch, so the compute verifier never waits forever.
        sfpu_verify_exit();
    }

    // Close credit connections (automatically flushes remaining credits, no-op if NUM_CREDIT_CONNECTIONS == 0)
    perf_t0 = PERF_BD_NOW();
    receiver_config->close_credit_connections();
    PERF_BD_SET_SINCE(6, perf_t0);  // R_CLOSE

    // Terminate muxes and wait for completion
    perf_t0 = PERF_BD_NOW();
    mux_termination_manager.terminate_muxes();
    PERF_BD_SET_SINCE(7, perf_t0);  // R_MUX_TERM

    // Final per-config flush — write per-config counters for progress monitoring.
    // Skipped in BENCHMARK_MODE: num_packets_processed is never advanced in the
    // skip-validation path so the values would be zero (causing host-side
    // granular monitoring to falsely report stalls), and we want to keep the
    // exit path free of extra L1 writes for accurate perf measurement.
    if constexpr (!BENCHMARK_MODE) {
        auto* per_config_results = get_per_config_results(receiver_config->get_result_buffer_address());
        for (uint8_t i = 0; i < NUM_TRAFFIC_CONFIGS; i++) {
            uint64_t config_packets = receiver_config->traffic_configs()[i]->num_packets_processed;
            write_per_config_result(&per_config_results[i], config_packets);
        }
    }

    // Write test results
    write_test_packets(receiver_config->get_result_buffer_address(), total_packets_received);
    write_perf_breakdown(receiver_config->get_result_buffer_address());
    uint32_t final_status = failed ? TT_FABRIC_STATUS_DATA_MISMATCH : TT_FABRIC_STATUS_PASS;
    write_test_status(receiver_config->get_result_buffer_address(), final_status);

    noc_async_full_barrier();
}
