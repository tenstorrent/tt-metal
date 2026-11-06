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
// 0: no stream; 1: a stream; 2: a stream with idle segments (after a segment's packets the worker may wait a number
// of cycles, modelling compute gaps). One argument for both, so senders without idle segments build as before.
constexpr uint32_t STREAM_ARG = get_compile_time_arg_val(10);
constexpr bool STREAM_MODE = STREAM_ARG != 0;
constexpr bool STREAM_IDLE = STREAM_ARG == 2;

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

// A stream feeds one router.
static_assert(!STREAM_MODE || NUM_FABRIC_CONNECTIONS == 1, "A stream sender has exactly one fabric connection");
static_assert(!STREAM_IDLE || STREAM_MODE, "Idle segments belong to a stream");

// A single-pattern benchmark sender runs the stream loop as a one-segment stream.
constexpr bool RUNS_STREAM_LOOP = STREAM_MODE || (BENCHMARK_MODE && NUM_TRAFFIC_CONFIGS == 1);

// Benchmark stream path: the segment whose header each router slot holds, so a slot is rewritten
// only when its next packet belongs to a different segment.
constexpr uint32_t MAX_STREAM_SLOTS = 256;
constexpr uint8_t STREAM_SLOT_EMPTY = 0xFF;
static_assert(
    !RUNS_STREAM_LOOP || NUM_TRAFFIC_CONFIGS < STREAM_SLOT_EMPTY, "Too many stream segments for the slot table");
uint8_t stream_slot_segment[RUNS_STREAM_LOOP ? MAX_STREAM_SLOTS : 1];

// Benchmark stream path: one lap's packets in send order (segment and header address), so the send
// loop is a single loop instead of nested segment/count loops. Laps longer than this use the nested
// loops, where long runs of one segment make the per-segment overhead rare.
constexpr uint32_t MAX_FLAT_LAP = 64;
uint8_t stream_lap_segment[RUNS_STREAM_LOOP ? MAX_FLAT_LAP : 1];
uint32_t stream_lap_header[RUNS_STREAM_LOOP ? MAX_FLAT_LAP : 1];
// Idle segments: cycles to wait after each segment's packets, and after each position of a flat lap (non-zero only
// on a segment's last packet).
uint32_t stream_segment_idle[STREAM_IDLE ? NUM_TRAFFIC_CONFIGS : 1];
uint32_t stream_lap_idle[STREAM_IDLE ? MAX_FLAT_LAP : 1];

// An idle segment: the worker sends nothing for `cycles` cycles on the clock the run is timed with, so the wait
// counts in the run's elapsed time. Returns the cycles actually waited.
FORCE_INLINE uint64_t stream_idle_wait(uint32_t cycles) {
    const uint64_t start = get_timestamp();
    uint64_t now = start;
    while (now - start < cycles) {
        now = get_timestamp();
    }
    return now - start;
}

// Sends the header already in the current slot again by returning the slot to the router, without
// copying it. Only the VC0 adapter exposes the credit-only update, so other adapters (VC2's) return
// false and the caller copies the header instead.
template <bool STATEFUL_NOC = true, typename EdmSenderT>
FORCE_INLINE bool replay_current_slot(EdmSenderT* conn) {
    if constexpr (std::is_same_v<EdmSenderT, WorkerToFabricEdmSender>) {
        tt::tt_fabric::fabric_detail::update_credits_and_slots<STATEFUL_NOC>(conn);
        return true;
    } else {
        return false;
    }
}

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

    // Stream program, after the traffic configs in local args: lap count, then packets per lap for
    // each traffic config in order.
    uint32_t stream_repeat = 0;
    uint32_t stream_counts[NUM_TRAFFIC_CONFIGS];
    if constexpr (STREAM_MODE) {
        stream_repeat = get_local_arg_val<uint32_t>(local_args_idx++);
        for (uint8_t i = 0; i < NUM_TRAFFIC_CONFIGS; i++) {
            stream_counts[i] = get_local_arg_val<uint32_t>(local_args_idx++);
        }
        if constexpr (STREAM_IDLE) {
            for (uint8_t i = 0; i < NUM_TRAFFIC_CONFIGS; i++) {
                stream_segment_idle[i] = get_local_arg_val<uint32_t>(local_args_idx++);
            }
        }
    }

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

    constexpr uint32_t PROGRESS_UPDATE_INTERVAL = 1000;  // Write progress every 1000 loops

    auto write_progress = [&]() {
        auto* per_config_results = get_per_config_results(sender_config->get_result_buffer_address());
        uint64_t progress_packets_sent = 0;
        for (uint8_t i = 0; i < NUM_TRAFFIC_CONFIGS; i++) {
            uint64_t config_packets = sender_config->traffic_config_ptrs[i]->num_packets_processed;
            progress_packets_sent += config_packets;
            write_per_config_result(&per_config_results[i], config_packets);
        }
        write_test_packets(sender_config->get_result_buffer_address(), progress_packets_sent);
    };

    // Benchmark stream: every segment goes to the same router. Each segment's header is built once;
    // a packet copies it into the router slot only when that slot last held another segment's header,
    // and otherwise just returns the slot so the router sends the header already there.
    auto send_stream_benchmark = [&]() {
        // The connection object lives in L1, and every free-slot check invalidates Blackhole's small
        // L1 cache, so reading its fields per packet costs several L1 misses. The loop works on a copy
        // in local memory and writes it back afterwards for close_connections.
        auto* conn_l1 = static_cast<EdmSenderType*>(sender_config->traffic_config_ptrs[0]->connection_ptr_);
        EdmSenderType local_conn = *conn_l1;
        auto* conn = &local_conn;
        ASSERT(conn->num_buffers_per_channel <= MAX_STREAM_SLOTS);
        for (uint32_t slot = 0; slot < conn->num_buffers_per_channel; slot++) {
            stream_slot_segment[slot] = STREAM_SLOT_EMPTY;
        }

        // Arms the header-copy and credit command buffers for this router. The loop below issues no
        // other NoC writes, since under DM_DYNAMIC_NOC they would share these command buffers.
        conn->setup_stateful_send_cmd_bufs();
        uint32_t header_copies = 0;
        // Packets that found every router slot full. A router that keeps up with the worker never
        // fills up, so a run with almost no waits measured the worker rather than the router.
        uint32_t slot_waits = 0;
        // Cycles spent waiting for a slot, timed only when a wait happens. The rest of the loop's time
        // is the worker's own cost per packet, which a router-bound run otherwise hides.
        uint64_t wait_cycles = 0;
        // Cycles spent in idle segments, which are part of the run but not the worker's cost per packet.
        uint64_t idle_cycles = 0;
        // Slots known to be free since the last check. Checking costs a fence and an L1 read of the
        // router's counter, so the loop checks only once it has used up the slots it knows about.
        uint32_t free_slots = 0;

        auto send_packet = [&](uint8_t segment, uint32_t header_address) __attribute__((always_inline)) {
            if (free_slots == 0) {
                free_slots = conn->get_num_free_write_slots();
                if (free_slots == 0) {
                    slot_waits++;
                    const uint64_t wait_start = get_timestamp();
                    do {
                        free_slots = conn->get_num_free_write_slots();
                    } while (free_slots == 0);
                    wait_cycles += get_timestamp() - wait_start;
                }
            }
            free_slots--;
            const uint8_t slot = conn->buffer_slot_index.get();
            if (stream_slot_segment[slot] == segment && replay_current_slot(conn)) {
                return;
            }
            conn->send_current_slot_stateful_non_blocking_from_address(header_address, sizeof(PACKET_HEADER_TYPE));
            stream_slot_segment[slot] = segment;
            header_copies++;
        };

        uint32_t lap_length = 0;
        for (uint8_t segment = 0; segment < NUM_TRAFFIC_CONFIGS; segment++) {
            lap_length += stream_counts[segment];
        }

        const uint64_t loop_start = get_timestamp();
        if (lap_length <= MAX_FLAT_LAP) {
            // One lap's packets in order, so each packet is one local-memory lookup.
            uint32_t position = 0;
            for (uint8_t segment = 0; segment < NUM_TRAFFIC_CONFIGS; segment++) {
                const uint32_t header_address = (uint32_t)sender_config->traffic_config_ptrs[segment]->packet_header;
                for (uint32_t i = 0; i < stream_counts[segment]; i++) {
                    stream_lap_segment[position] = segment;
                    stream_lap_header[position] = header_address;
                    if constexpr (STREAM_IDLE) {
                        stream_lap_idle[position] = i + 1 == stream_counts[segment] ? stream_segment_idle[segment] : 0;
                    }
                    position++;
                }
            }
            // The host keeps repeat x lap length within 32 bits.
            const uint32_t total_packets = stream_repeat * lap_length;
            position = 0;
            for (uint32_t packet = 0; packet < total_packets; packet++) {
                send_packet(stream_lap_segment[position], stream_lap_header[position]);
                if constexpr (STREAM_IDLE) {
                    if (stream_lap_idle[position] != 0) {
                        idle_cycles += stream_idle_wait(stream_lap_idle[position]);
                    }
                }
                position = position + 1 == lap_length ? 0 : position + 1;
            }
        } else {
            for (uint32_t lap = 0; lap < stream_repeat; lap++) {
                for (uint8_t segment = 0; segment < NUM_TRAFFIC_CONFIGS; segment++) {
                    const uint32_t header_address =
                        (uint32_t)sender_config->traffic_config_ptrs[segment]->packet_header;
                    for (uint32_t i = 0; i < stream_counts[segment]; i++) {
                        send_packet(segment, header_address);
                    }
                    if constexpr (STREAM_IDLE) {
                        if (stream_segment_idle[segment] != 0) {
                            idle_cycles += stream_idle_wait(stream_segment_idle[segment]);
                        }
                    }
                }
            }
        }
        uint64_t busy_cycles = get_timestamp() - loop_start - wait_cycles;
        if constexpr (STREAM_IDLE) {
            busy_cycles -= idle_cycles;
        }
        *conn_l1 = local_conn;

        for (uint8_t segment = 0; segment < NUM_TRAFFIC_CONFIGS; segment++) {
            sender_config->traffic_config_ptrs[segment]->num_packets_processed = stream_repeat * stream_counts[segment];
        }
        write_stream_counters(sender_config->get_result_buffer_address(), header_copies, slot_waits, busy_cycles);
    };

    // Benchmark round robin over several patterns: each pattern writes its header into the slots of its
    // first packets, then sends credit-only updates so the router resends what the slots hold. Like the
    // stream loop, it works on local-memory copies of the connections and checks a connection for
    // free slots only after using up the ones it already knows about.
    auto send_round_robin_benchmark = [&]() {
        EdmSenderType local_conns[NUM_FABRIC_CONNECTIONS];
        uint32_t free_slots[NUM_FABRIC_CONNECTIONS];
        for (uint8_t c = 0; c < NUM_FABRIC_CONNECTIONS; c++) {
            local_conns[c] = sender_config->connections.get_fabric_connection(c);
            free_slots[c] = 0;
        }
        uint8_t connection_of[NUM_TRAFFIC_CONFIGS];
        uint32_t header_of[NUM_TRAFFIC_CONFIGS];
        uint32_t packets_of[NUM_TRAFFIC_CONFIGS];
        uint32_t sent[NUM_TRAFFIC_CONFIGS];
        for (uint8_t i = 0; i < NUM_TRAFFIC_CONFIGS; i++) {
            auto* traffic_config = sender_config->traffic_config_ptrs[i];
            connection_of[i] = traffic_config->connection_idx_;
            header_of[i] = (uint32_t)traffic_config->packet_header;
            packets_of[i] = traffic_config->metadata.num_packets;
            sent[i] = 0;
        }

        uint32_t header_writes = 0;
        uint32_t slot_waits = 0;
        uint64_t wait_cycles = 0;
        const uint64_t loop_start = get_timestamp();
        bool packets_left = true;
        while (packets_left) {
            packets_left = false;
            for (uint8_t i = 0; i < NUM_TRAFFIC_CONFIGS; i++) {
                if (sent[i] == packets_of[i]) {
                    continue;
                }
                const uint8_t c = connection_of[i];
                auto& conn = local_conns[c];
                if (free_slots[c] == 0) {
                    free_slots[c] = conn.get_num_free_write_slots();
                    if (free_slots[c] == 0) {
                        slot_waits++;
                        const uint64_t wait_start = get_timestamp();
                        do {
                            free_slots[c] = conn.get_num_free_write_slots();
                        } while (free_slots[c] == 0);
                        wait_cycles += get_timestamp() - wait_start;
                    }
                }
                free_slots[c]--;
                if (sent[i] < conn.num_buffers_per_channel || !replay_current_slot</*STATEFUL_NOC=*/false>(&conn)) {
                    conn.send_payload_flush_non_blocking_from_address(header_of[i], sizeof(PACKET_HEADER_TYPE));
                    header_writes++;
                }
                sent[i]++;
                packets_left |= sent[i] < packets_of[i];
            }
        }
        const uint64_t busy_cycles = get_timestamp() - loop_start - wait_cycles;

        for (uint8_t c = 0; c < NUM_FABRIC_CONNECTIONS; c++) {
            sender_config->connections.get_fabric_connection(c) = local_conns[c];
        }
        for (uint8_t i = 0; i < NUM_TRAFFIC_CONFIGS; i++) {
            sender_config->traffic_config_ptrs[i]->num_packets_processed = sent[i];
        }
        write_stream_counters(sender_config->get_result_buffer_address(), header_writes, slot_waits, busy_cycles);
    };

    // Validated stream: the normal per-packet send in program order. A segment that is out of
    // flow-control credits holds up the stream rather than letting another segment go first.
    auto send_stream_validated = [&]() {
        uint32_t packets_since_progress = 0;
        for (uint32_t lap = 0; lap < stream_repeat; lap++) {
            for (uint8_t segment = 0; segment < NUM_TRAFFIC_CONFIGS; segment++) {
                auto* traffic_config = sender_config->traffic_config_ptrs[segment];
                for (uint32_t i = 0; i < stream_counts[segment]; i++) {
                    while (!traffic_config->template send_one_packet<false>()) {
                    }
                    if (++packets_since_progress == PROGRESS_UPDATE_INTERVAL) {
                        packets_since_progress = 0;
                        write_progress();
                    }
                }
                if constexpr (STREAM_IDLE) {
                    if (stream_segment_idle[segment] != 0) {
                        stream_idle_wait(stream_segment_idle[segment]);
                    }
                }
            }
        }
    };

    // Round-robin packet sending: send one packet from each config per iteration
    uint64_t start_timestamp = get_timestamp();

    if constexpr (STREAM_MODE) {
        if constexpr (BENCHMARK_MODE) {
            send_stream_benchmark();
        } else {
            send_stream_validated();
        }
    } else if constexpr (RUNS_STREAM_LOOP) {
        // A single benchmark pattern is a one-segment stream of its packets.
        stream_repeat = sender_config->traffic_config_ptrs[0]->metadata.num_packets;
        stream_counts[0] = 1;
        send_stream_benchmark();
    } else if constexpr (BENCHMARK_MODE) {
        send_round_robin_benchmark();
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
                    write_progress();
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

    // Write test results
    write_test_cycles(sender_config->get_result_buffer_address(), total_elapsed_cycles_outer_loop);
    write_test_packets(sender_config->get_result_buffer_address(), total_packets_sent);
    write_test_status(sender_config->get_result_buffer_address(), TT_FABRIC_STATUS_PASS);

    noc_async_full_barrier();
}
