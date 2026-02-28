// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// This header provides amortized credit-passing "speedy" step functions for the
// fabric erisc router.  They are only used when super_speedy_mode == true
// (single sender channel with non-zero amortization frequencies).
//
// It also provides line-topology speedy step functions (line_speedy_mode) for
// 2 sender channels with forwarding support and packed credit epoch array
// disambiguation.
//
// Must be included AFTER fabric_erisc_router_ct_args.hpp and all other router
// headers that define helpers such as send_next_data, send_credits_to_upstream_workers,
// receiver_send_completion_ack, receiver_send_received_ack,
// can_forward_packet_completely, receiver_forward_packet, etc.

#include "api/debug/ring_buffer.h"

// Watcher ring buffer event tags (upper 4 bits of the uint32_t)
// Encoding: [tag:4][payload:28]
//
// TX events:
//   0x1 = TX_SENT:       [tag:4][ch:1][epc:3][we:8][re:8][epoch_val_lo:8]
//   0x2 = TX_EPOCH_ADV:  [tag:4][0:4][we:8][re:8][0:8]
//   0x3 = TX_COMPL:      [tag:4][0:4][new_completions:12][total_unproc:12]
//   0x4 = TX_DECOMPOSE:  [tag:4][re:4][ch0:8][ch1:8][free_slots_lo:8]
//   0x5 = TX_UPSTREAM:   [tag:4][ch:4][credits:12][0:12]
//   0x6 = TX_EPOCH_BP:   [tag:4][0:4][we:8][re:8][epc:8]
//
// RX events:
//   0xA = RX_FWD:        [tag:4][trid:4][unacked:12][0:12]
//   0xB = RX_TRID_FLIP:  [tag:4][pending_trid:4][batch:12][new_wr_trid:12]
//   0xC = RX_FLUSH_DONE: [tag:4][trid:4][batch:12][remaining:12]

static size_t completion_count = 0;

/*
 * A fast neighbour exchange only sender channel step impl
 */
template <
    uint8_t sender_channel_index,
    uint8_t to_receiver_pkts_sent_id,
    bool enable_first_level_ack,
    typename SenderChannelT,
    typename WorkerInterfaceT,
    typename ReceiverPointersT,
    typename ReceiverChannelT,
    typename LocalTelemetryT>
FORCE_INLINE bool run_sender_channel_step_speedy(
    SenderChannelT& local_sender_channel,
    WorkerInterfaceT& local_sender_channel_worker_interface,
    ReceiverPointersT& outbound_to_receiver_channel_pointers,
    ReceiverChannelT& remote_receiver_channel,
    bool& channel_connection_established,
    uint32_t sender_channel_free_slots_stream_id,
    SenderChannelFromReceiverCredits& sender_channel_from_receiver_credits,
    PerfTelemetryRecorder& perf_telemetry_recorder,
    LocalTelemetryT& local_fabric_telemetry,
    uint32_t& sender_amort_counter) {
    bool progress = false;

    bool receiver_has_space_for_packet = outbound_to_receiver_channel_pointers.has_space_for_packet();
    uint32_t free_slots = get_ptr_val(sender_channel_free_slots_stream_id);
    bool has_unsent_packet = free_slots != WorkerInterfaceT::num_buffers;
    bool can_send = receiver_has_space_for_packet && has_unsent_packet;

    if constexpr (!ETH_TXQ_SPIN_WAIT_SEND_NEXT_DATA) {
        can_send = can_send && !internal_::eth_txq_is_busy(sender_txq_id);
    }

    if (can_send) {
        progress = true;

        auto* pkt_header = reinterpret_cast<volatile tt_l1_ptr PACKET_HEADER_TYPE*>(
            local_sender_channel.get_cached_next_buffer_slot_addr());
        // explicit inline of send next data shaves about 9 cycles off the send time due to more efficient code-gen
        {
            auto& remote_receiver_num_free_slots = outbound_to_receiver_channel_pointers.num_free_slots;
            uint32_t src_addr = local_sender_channel.get_cached_next_buffer_slot_addr();

            const size_t payload_size_bytes = pkt_header->get_payload_size_including_header();

            bool busy = internal_::eth_txq_is_busy(sender_txq_id);

            const auto dest_addr = outbound_to_receiver_channel_pointers.remote_receiver_channel_address_ptr;

            if constexpr (ETH_TXQ_SPIN_WAIT_SEND_NEXT_DATA) {
                while (busy) {
                    busy = internal_::eth_txq_is_busy(sender_txq_id);
                }
            }
            internal_::eth_send_packet_bytes_unsafe(sender_txq_id, src_addr, dest_addr, payload_size_bytes);

            // Note: We can only advance to the next buffer index if we have fully completed the send (both the payload
            // and sync messages)
            local_sender_channel_worker_interface.template update_write_counter_for_send<false /*SKIP_LIVENESS*/>();

            // Advance receiver buffer pointers
            busy = internal_::eth_txq_is_busy(sender_txq_id);
            outbound_to_receiver_channel_pointers.advance_remote_receiver_buffer_pointer();
            local_sender_channel.advance_to_next_cached_buffer_slot_addr();
            remote_receiver_num_free_slots--;

            record_packet_send(perf_telemetry_recorder, sender_channel_index, payload_size_bytes);

            while (busy) {
                busy = internal_::eth_txq_is_busy(sender_txq_id);
            };
            remote_update_ptr_val<to_receiver_pkts_sent_id, sender_txq_id>(1U);
        }
        sender_amort_counter++;
        if constexpr (FABRIC_TELEMETRY_BANDWIDTH) {
            update_bw_counters(pkt_header, local_fabric_telemetry);
        }
        increment_local_update_ptr_val(sender_channel_free_slots_stream_id, 1);
    }

    // We only want to actually bother checking for completions after a certain number of sent packets are outstanding
    // since the instructions to actually process each inbound completion from receiver is somewhat costly
    bool check_completions = sender_amort_counter > SENDER_CREDIT_AMORTIZATION_FREQUENCY;
    if (check_completions) {
        int32_t completions = sender_channel_from_receiver_credits
                                  .template get_num_unprocessed_completions_from_receiver<ENABLE_RISC_CPU_DATA_CACHE>();
        if (completions) {
            outbound_to_receiver_channel_pointers.num_free_slots += completions;
            sender_channel_from_receiver_credits.increment_num_processed_completions(completions);

            completion_count += completions;
        }
    }

    // Similarly only send back the credit to the worker very infrequently since it's a very
    // expensive operation.
    bool send_credits = completion_count >= SENDER_CREDIT_AMORTIZATION_FREQUENCY;
    if (send_credits) {
        send_credits_to_upstream_workers<false /*deadlock_avoidance*/, false /*SKIP_LIVENESS*/>(
            local_sender_channel_worker_interface, completion_count, channel_connection_established);
        sender_amort_counter -= completion_count;
        completion_count = 0;
    }
    return progress;
}

/*
 * A fast neighbour exchange only receiver channel step impl
 */
static uint32_t unacked_sends = 0;
static constexpr uint8_t pingpong_trid_a = RX_CH_TRID_STARTS[0];
static constexpr uint8_t pingpong_trid_b = RX_CH_TRID_STARTS[0] + 1;
static_assert(
    !super_speedy_mode || NUM_TRANSACTION_IDS >= 2,
    "Ping-pong TRID requires at least 2 transaction IDs per receiver channel");
static_assert(!super_speedy_mode || pingpong_trid_a == 0, "Ping-pong TRID flip uses '1 - trid', requires trid_a == 0");
static uint8_t current_write_trid = pingpong_trid_a;
static uint8_t pending_flush_trid = pingpong_trid_b;
static uint32_t pending_flush_batch_count = 0;
static bool has_pending_flush = false;

template <
    uint8_t receiver_channel,
    uint8_t to_receiver_pkts_sent_id,
    bool enable_first_level_ack,
    size_t DOWNSTREAM_EDM_SIZE,
    typename WriteTridTracker,
    typename ReceiverChannelBufferT,
    typename ReceiverChannelPointersT,
    typename DownstreamSenderT,
    typename LocalRelayInterfaceT,
    typename LocalTelemetryT>
FORCE_INLINE bool run_receiver_channel_step_speedy(
    ReceiverChannelBufferT& local_receiver_channel,
    std::array<DownstreamSenderT, DOWNSTREAM_EDM_SIZE>& downstream_edm_interfaces,
    LocalRelayInterfaceT& local_relay_interface,
    ReceiverChannelPointersT& receiver_channel_pointers,
    WriteTridTracker& receiver_channel_trid_tracker,
    std::array<uint8_t, num_eth_ports>& port_direction_table,
    ReceiverChannelResponseCreditSender& receiver_channel_response_credit_sender,
    const tt::tt_fabric::routing_l1_info_t& routing_table,
    LocalTelemetryT& local_fabric_telemetry) {
    bool progress = false;
    auto& wr_sent_counter = receiver_channel_pointers.wr_sent_counter;
    auto pkts_received = get_ptr_val<to_receiver_pkts_sent_id>();
    bool unwritten_packets = pkts_received != 0;

    if (unwritten_packets) {
        static_assert(!ENABLE_RISC_CPU_DATA_CACHE, "ENABLE_RISC_CPU_DATA_CACHE must be disabled for speedy path");
        // TODO: hoist me between get_ptr_val and the mask for its value.
        auto receiver_buffer_index = wr_sent_counter.get_buffer_index();
        tt_l1_ptr PACKET_HEADER_TYPE* packet_header = const_cast<PACKET_HEADER_TYPE*>(
            local_receiver_channel.template get_packet_header<PACKET_HEADER_TYPE>(receiver_buffer_index));

        // Single 4B aligned load at offset 40 to get payload_size_bytes + noc_send_type
        // instead of two separate uncached L1 reads.
        // (adjacency of payload_size_bytes and noc_send_type validated by PackedPayloadAndSendType)
        auto packed = PACKET_HEADER_TYPE::PackedPayloadAndSendType::load(packet_header);

        execute_chip_unicast_to_local_chip_impl(
            packet_header, packed.payload_size_bytes, packed.noc_send_type, current_write_trid, receiver_channel);

        did_something = true;
        progress = true;
        if constexpr (FABRIC_TELEMETRY_BANDWIDTH) {
            update_bw_counters(packet_header, local_fabric_telemetry);
        }
        channel_trimming_usage_recorder.set_receiver_channel_data_forwarded(receiver_channel);

        wr_sent_counter.increment();
        increment_local_update_ptr_val<to_receiver_pkts_sent_id>(-1);
        unacked_sends++;
    }

    // --- Ping-pong TRID flush ---
    // All packets in a batch share a single TRID (current_write_trid). When the batch
    // threshold is hit, we flip to the other TRID and check the previous batch's single
    // TRID for completion — replacing the per-slot loop with a single register read.
    //
    // The pending TRID is checked eagerly (every call) to minimize credit return latency.
    // Only the batch flip requires reaching the threshold.
    // when we pass the threshold of unacked messages,
    if ((unacked_sends >= RECEIVER_CREDIT_AMORTIZATION_FREQUENCY) && !has_pending_flush) {
        pending_flush_trid = current_write_trid;
        pending_flush_batch_count = unacked_sends;
        current_write_trid = 1 - current_write_trid;
        has_pending_flush = true;
    }
    if (has_pending_flush) {
        bool flushed = ncrisc_noc_nonposted_write_with_transaction_id_sent(
            tt::tt_fabric::edm_to_local_chip_noc, pending_flush_trid);

        if (flushed) {
            auto& completion_counter = receiver_channel_pointers.completion_counter;
            completion_counter.increment_n(pending_flush_batch_count);
            receiver_send_completion_ack<false /*CHECK_BUSY*/>(
                receiver_channel_response_credit_sender, 0, pending_flush_batch_count);

            unacked_sends -= pending_flush_batch_count;
            has_pending_flush = false;
        }
    }

    return progress;
}

constexpr uint32_t next_power_of_2(uint32_t v) {
    v--;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    return v + 1;
}

// ===========================================================================
// ===========================================================================
//
//  LINE-TOPOLOGY SPEEDY PATH (2 sender channels + forwarding)
//
// ===========================================================================
// ===========================================================================

// ---------------------------------------------------------------------------
// Credit epoch array configuration
// ---------------------------------------------------------------------------
// NUM_CREDIT_EPOCHS must be a power of 2 for branchless index masking.
// Derived from the remote receiver's buffer slot count so we can never overflow the epoch ring.
//
// We size the array to 2x the minimum needed to cover REMOTE_RECEIVER_NUM_BUFFERS.
// This eliminates the need for a runtime back-pressure check (write_epoch catching
// read_epoch) in line_speedy_send_one_packet.  The sender can have at most
// ceil(REMOTE_RECEIVER_NUM_BUFFERS / EPOCH_SIZE) epochs in-flight (it cannot send
// more packets than the receiver has buffer slots).  With 2x headroom the write_epoch
// can never wrap into read_epoch, so the epoch array is always safe to write without
// checking for overlap.
static constexpr uint32_t EPOCH_SIZE = SENDER_CREDIT_AMORTIZATION_FREQUENCY;
static constexpr uint32_t REMOTE_RECEIVER_NUM_BUFFERS = REMOTE_RECEIVER_NUM_BUFFERS_ARRAY[VC0_RECEIVER_CHANNEL];
static constexpr uint32_t MIN_CREDIT_EPOCHS =
    next_power_of_2((REMOTE_RECEIVER_NUM_BUFFERS + EPOCH_SIZE - 1) / EPOCH_SIZE);
static constexpr uint32_t NUM_CREDIT_EPOCHS = MIN_CREDIT_EPOCHS * 2;
static_assert((NUM_CREDIT_EPOCHS & (NUM_CREDIT_EPOCHS - 1)) == 0, "NUM_CREDIT_EPOCHS must be a power of 2");
static_assert(
    NUM_CREDIT_EPOCHS > (REMOTE_RECEIVER_NUM_BUFFERS + EPOCH_SIZE - 1) / EPOCH_SIZE,
    "Epoch array must be strictly larger than max in-flight epochs to avoid write/read overlap");
static constexpr uint32_t EPOCH_MASK = NUM_CREDIT_EPOCHS - 1;

// ---------------------------------------------------------------------------
// Epoch advance strategy selection (constexpr toggle for benchmarking)
// ---------------------------------------------------------------------------
// EPOCH_ADVANCE_BRANCHING:  original if (epoch_pkt_count == EPOCH_SIZE) branch
// EPOCH_ADVANCE_BRANCHLESS: branchless epoch advance — always pre-clears next slot
// EPOCH_ADVANCE_POINTER:    raw pointer traversal instead of array indexing
enum class EpochAdvanceStrategy { BRANCHING, BRANCHLESS, POINTER };
static constexpr EpochAdvanceStrategy epoch_advance_strategy = EpochAdvanceStrategy::BRANCHING;

// ---------------------------------------------------------------------------
// Sender-side state for line speedy mode
// ---------------------------------------------------------------------------
static uint32_t credit_epoch_array[NUM_CREDIT_EPOCHS] = {};
static uint32_t write_epoch = 0;
static uint32_t read_epoch = 0;
static uint32_t epoch_pkt_count = 0;

// Pointer-mode state (only used when epoch_advance_strategy == POINTER)
static uint32_t* credit_epoch_write_ptr = &credit_epoch_array[0];
static uint32_t* credit_epoch_read_ptr = &credit_epoch_array[0];
// Byte mask for wrapping pointer within the array: (NUM_CREDIT_EPOCHS * 4) - 1
// Works because the array is power-of-2 entries of 4 bytes each.
static constexpr uintptr_t EPOCH_PTR_BYTE_MASK = (NUM_CREDIT_EPOCHS * sizeof(uint32_t)) - 1;

// Accumulated per-channel upstream credits not yet sent, packed as bytes:
//   byte 0 = ch0 count, byte 1 = ch1 count (matches credit_epoch_array packing).
// Kept packed so epoch decompose is a single add (no per-channel unpacking).
// Byte extraction only happens in the credit-send path which fires infrequently.
static uint32_t unsent_upstream_credits_packed = 0;

// Total bulk completions received but not yet decomposed into per-channel credits.
static uint32_t total_unprocessed_completions = 0;

// Amortization counter for completion checks: incremented per successful send,
// completion check only fires when this exceeds EPOCH_SIZE (matching neighbor exchange pattern).
static uint32_t sender_completion_amort_counter = 0;

// ---------------------------------------------------------------------------
// Receiver-side state for line speedy mode (ping-pong TRID for amortized flush)
// ---------------------------------------------------------------------------
static uint32_t line_unacked_sends = 0;
static constexpr uint8_t line_pingpong_trid_a = RX_CH_TRID_STARTS[0];
static constexpr uint8_t line_pingpong_trid_b = RX_CH_TRID_STARTS[0] + 1;
static_assert(
    !line_speedy_mode || NUM_TRANSACTION_IDS >= 2,
    "Ping-pong TRID requires at least 2 transaction IDs per receiver channel");
static_assert(
    !line_speedy_mode || line_pingpong_trid_a == 0, "Ping-pong TRID flip uses '1 - trid', requires trid_a == 0");
static uint8_t line_current_write_trid = line_pingpong_trid_a;
static uint8_t line_pending_flush_trid = line_pingpong_trid_b;
static uint32_t line_pending_flush_batch_count = 0;
static bool line_has_pending_flush = false;

// ===========================================================================
//  LINE SENDER: Inline send helper (shared between both channels)
// ===========================================================================

// Performs the ethernet send for a single sender channel and records the send
// into the credit epoch array.  Returns true if a packet was sent.
template <
    uint8_t sender_channel_index,
    uint8_t to_receiver_pkts_sent_id,
    typename SenderChannelT,
    typename WorkerInterfaceT,
    typename ReceiverPointersT,
    typename LocalTelemetryT>
FORCE_INLINE bool line_speedy_send_one_packet(
    SenderChannelT& local_sender_channel,
    WorkerInterfaceT& local_sender_channel_worker_interface,
    ReceiverPointersT& outbound_to_receiver_channel_pointers,
    uint32_t sender_channel_free_slots_stream_id,
    PerfTelemetryRecorder& perf_telemetry_recorder,
    LocalTelemetryT& local_fabric_telemetry) {
    NamedProfiler<
        CodeProfilingTimerType::SPEEDY_SENDER_SEND_ONE_PACKET,
        code_profiling_enabled_timers_bitfield,
        code_profiling_buffer_base_addr>
        send_one_packet_timer;
    send_one_packet_timer.open();

    bool receiver_has_space = outbound_to_receiver_channel_pointers.has_space_for_packet();
    uint32_t free_slots = get_ptr_val(sender_channel_free_slots_stream_id);
    bool has_unsent_packet = free_slots != WorkerInterfaceT::num_buffers;
    bool can_send = receiver_has_space && has_unsent_packet;

    if constexpr (!ETH_TXQ_SPIN_WAIT_SEND_NEXT_DATA) {
        can_send = can_send && !internal_::eth_txq_is_busy(sender_txq_id);
    }

    // No epoch back-pressure check needed: the epoch array is sized to 2x the
    // max in-flight epochs (bounded by REMOTE_RECEIVER_NUM_BUFFERS / EPOCH_SIZE),
    // so write_epoch can never wrap into read_epoch.

    if (can_send) {
        // Timer only dumps when we actually send a packet
        send_one_packet_timer.set_should_dump(true);

        // --- Send the packet (inlined for codegen efficiency) ---
        auto* pkt_header = reinterpret_cast<volatile tt_l1_ptr PACKET_HEADER_TYPE*>(
            local_sender_channel.get_cached_next_buffer_slot_addr());
        {
            auto& remote_receiver_num_free_slots = outbound_to_receiver_channel_pointers.num_free_slots;
            uint32_t src_addr = local_sender_channel.get_cached_next_buffer_slot_addr();
            const size_t payload_size_bytes = pkt_header->get_payload_size_including_header();

            bool busy = internal_::eth_txq_is_busy(sender_txq_id);
            const auto dest_addr = outbound_to_receiver_channel_pointers.remote_receiver_channel_address_ptr;
            WATCHER_RING_BUFFER_PUSH(0x55555555);
            WATCHER_RING_BUFFER_PUSH(src_addr);
            WATCHER_RING_BUFFER_PUSH(dest_addr);

            if constexpr (ETH_TXQ_SPIN_WAIT_SEND_NEXT_DATA) {
                while (busy) {
                    busy = internal_::eth_txq_is_busy(sender_txq_id);
                }
            }
            internal_::eth_send_packet_bytes_unsafe(sender_txq_id, src_addr, dest_addr, payload_size_bytes);

            local_sender_channel_worker_interface
                .template update_write_counter_for_send<sender_channel_index != 0 /*SKIP_LIVENESS*/>();

            busy = internal_::eth_txq_is_busy(sender_txq_id);
            outbound_to_receiver_channel_pointers.advance_remote_receiver_buffer_pointer();
            local_sender_channel.advance_to_next_cached_buffer_slot_addr();
            remote_receiver_num_free_slots--;

            record_packet_send(perf_telemetry_recorder, sender_channel_index, payload_size_bytes);

            while (busy) {
                busy = internal_::eth_txq_is_busy(sender_txq_id);
            };
            remote_update_ptr_val<to_receiver_pkts_sent_id, sender_txq_id>(1U);

            if constexpr (FABRIC_TELEMETRY_BANDWIDTH) {
                update_bw_counters(pkt_header, local_fabric_telemetry);
            }
        }

        increment_local_update_ptr_val(sender_channel_free_slots_stream_id, 1);

        // --- Record this send in the credit epoch array ---
        //
        // Byte (sender_channel_index) of epoch_array[write_epoch] gets incremented.
        // The shift is resolved at compile time since sender_channel_index is a template param.
        static constexpr uint32_t credit_inc = 1u << (sender_channel_index * 8);
        credit_epoch_array[write_epoch] += credit_inc;
        epoch_pkt_count++;
        sender_completion_amort_counter++;
        if (epoch_pkt_count == EPOCH_SIZE) {
            write_epoch = (write_epoch + 1) & EPOCH_MASK;
            epoch_pkt_count = 0;
            credit_epoch_array[write_epoch] = 0;
        }

        send_one_packet_timer.close();
    }

    return can_send;
}

// ===========================================================================
//  LINE SENDER: Main 2-channel step function
// ===========================================================================

template <
    uint8_t to_receiver_pkts_sent_id,
    bool enable_first_level_ack,
    typename SenderChannel0T,
    typename SenderChannel1T,
    typename WorkerInterface0T,
    typename WorkerInterface1T,
    typename ReceiverPointersT,
    typename ReceiverChannelT,
    typename LocalTelemetryT>
FORCE_INLINE bool run_sender_channels_step_line_speedy(
    SenderChannel0T& local_sender_channel_0,
    WorkerInterface0T& local_sender_channel_worker_interface_0,
    SenderChannel1T& local_sender_channel_1,
    WorkerInterface1T& local_sender_channel_worker_interface_1,
    ReceiverPointersT& outbound_to_receiver_channel_pointers,
    ReceiverChannelT& remote_receiver_channel,
    std::array<bool, NUM_SENDER_CHANNELS>& channel_connection_established,
    std::array<uint32_t, NUM_SENDER_CHANNELS>& local_sender_channel_free_slots_stream_ids,
    // Single shared bulk completion credit (replaces per-channel SenderChannelFromReceiverCredits)
    SenderChannelFromReceiverCredits& shared_completion_credits,
    PerfTelemetryRecorder& perf_telemetry_recorder,
    LocalTelemetryT& local_fabric_telemetry) {
    // --- Profiling timers (compile-time disabled when bitfield == 0) ---
    NamedProfiler<
        CodeProfilingTimerType::SPEEDY_SENDER_FULL_BOTH,
        code_profiling_enabled_timers_bitfield,
        code_profiling_buffer_base_addr>
        sender_full_timer;
    NamedProfiler<
        CodeProfilingTimerType::SPEEDY_SENDER_FULL_SINGLE,
        code_profiling_enabled_timers_bitfield,
        code_profiling_buffer_base_addr>
        sender_full_with_send_timer;
    NamedProfiler<
        CodeProfilingTimerType::SPEEDY_SENDER_SEND_DATA_BOTH,
        code_profiling_enabled_timers_bitfield,
        code_profiling_buffer_base_addr>
        sender_send_data_timer;
    NamedProfiler<
        CodeProfilingTimerType::SPEEDY_SENDER_SEND_DATA_SINGLE,
        code_profiling_enabled_timers_bitfield,
        code_profiling_buffer_base_addr>
        sender_send_data_single_timer;
    NamedProfiler<
        CodeProfilingTimerType::SPEEDY_SENDER_CHECK_COMPLETIONS,
        code_profiling_enabled_timers_bitfield,
        code_profiling_buffer_base_addr>
        sender_check_completions_timer;
    NamedProfiler<
        CodeProfilingTimerType::SPEEDY_SENDER_CREDITS_UPSTREAM,
        code_profiling_enabled_timers_bitfield,
        code_profiling_buffer_base_addr>
        sender_credits_upstream_timer;

    bool progress = false;

    // Full function timers: opened unconditionally, closed with should_dump at end
    sender_full_timer.open();
    sender_full_with_send_timer.open();

    // -----------------------------------------------------------------------
    // Phase 1 & 2: Send data from both sender channels
    //
    // Both channels share the same outbound receiver channel pointers (they
    // send to the same downstream receiver). Send ch0 first, then ch1.
    // -----------------------------------------------------------------------
    sender_send_data_timer.open();
    sender_send_data_single_timer.open();

    bool ch0_sent = line_speedy_send_one_packet<0, to_receiver_pkts_sent_id>(
        local_sender_channel_0,
        local_sender_channel_worker_interface_0,
        outbound_to_receiver_channel_pointers,
        local_sender_channel_free_slots_stream_ids[0],
        perf_telemetry_recorder,
        local_fabric_telemetry);

    bool ch1_sent = line_speedy_send_one_packet<1, to_receiver_pkts_sent_id>(
        local_sender_channel_1,
        local_sender_channel_worker_interface_1,
        outbound_to_receiver_channel_pointers,
        local_sender_channel_free_slots_stream_ids[1],
        perf_telemetry_recorder,
        local_fabric_telemetry);

    progress = ch0_sent || ch1_sent;
    // SEND_DATA: both packets sent
    sender_send_data_timer.set_should_dump(ch0_sent && ch1_sent);
    sender_send_data_timer.close();
    // SEND_DATA_SINGLE: exactly one packet sent (XOR)
    sender_send_data_single_timer.set_should_dump(ch0_sent != ch1_sent);
    sender_send_data_single_timer.close();

    // -----------------------------------------------------------------------
    // Phase 3: Check completions and decompose via credit epoch array
    //
    // === Credit Epoch Array Book-keeping ===
    //
    // The receiver sends BULK completion credits — a single count with no
    // per-sender-channel breakdown.  The sender must decompose these bulk
    // credits back into per-channel counts to know how many upstream credits
    // to propagate for each sender channel.
    //
    // We do this via a circular array of "epochs", where each epoch represents
    // EPOCH_SIZE consecutive packets sent (across all channels).  Each epoch
    // slot is a uint32_t with per-channel counts packed as bytes:
    //
    //          read_epoch                          write_epoch
    //               ↓                                   ↓
    //           epoch[0]   epoch[1]   epoch[2]   epoch[3]  ...
    //          ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐
    //   byte 3 │   0    │ │   0    │ │   0    │ │   0    │  (future ch3)
    //          ├────────┤ ├────────┤ ├────────┤ ├────────┤
    //   byte 2 │   0    │ │   0    │ │   0    │ │   0    │  (future ch2)
    //          ├────────┤ ├────────┤ ├────────┤ ├────────┤
    //   byte 1 │   1    │ │   3    │ │   0    │ │        │  ← ch1 count
    //          ├────────┤ ├────────┤ ├────────┤ ├────────┤
    //   byte 0 │   3    │ │   1    │ │   0    │ │        │  ← ch0 count
    //          └────────┘ └────────┘ └────────┘ └────────┘
    //           sum=4      sum=4      sum=0      in progress
    //           (full)     (full)     (cleared)
    //
    // Recording a send (compile-time constant shift):
    //     credit_epoch_array[write_epoch] += (1u << (sender_ch * 8));
    //
    // When epoch_pkt_count == EPOCH_SIZE, write_epoch advances (branchless).
    //
    // Processing completions (at most ONE epoch per pass to avoid starving
    // the send path):
    //   1. Read bulk completion count from shared credit register
    //   2. If total_unprocessed >= EPOCH_SIZE:
    //      - Load epoch_array[read_epoch] (single uint32_t — both channels)
    //      - Extract per-channel byte counts
    //      - Accumulate into unsent_upstream_credits_chX
    //      - Advance read_epoch
    //      - Reclaim EPOCH_SIZE receiver buffer slots
    //
    // Upstream credit propagation (at most ONE noc write per pass):
    //   - When unsent_upstream_credits_chX >= threshold, send and reset
    //   - Only one channel sends per pass to keep the send path responsive
    //
    // Invariants:
    //   - write_epoch is always ahead of or equal to read_epoch (mod NUM_CREDIT_EPOCHS)
    //   - Completions are consumed in EPOCH_SIZE chunks (receiver also sends
    //     in batches of RECEIVER_CREDIT_AMORTIZATION_FREQUENCY which should
    //     be a multiple of EPOCH_SIZE or equal to it)
    //   - credit_epoch_array[next_epoch] is always safe to clear because it is
    //     strictly ahead of read_epoch
    // -----------------------------------------------------------------------

    // Amortize completion checks: only check after enough sends have occurred,
    // matching the neighbor exchange pattern. This avoids the ~25 cycle cost of
    // get_num_unprocessed_completions_from_receiver on iterations with no progress.
    bool check_completions = sender_completion_amort_counter > SENDER_CREDIT_AMORTIZATION_FREQUENCY;

    sender_check_completions_timer.set_should_dump(check_completions);
    sender_check_completions_timer.open();

    bool had_new_completions = false;
    bool did_epoch_decompose = false;
    if (check_completions) {
        int32_t new_completions =
            shared_completion_credits
                .template get_num_unprocessed_completions_from_receiver<ENABLE_RISC_CPU_DATA_CACHE>();
        if (new_completions) {
            shared_completion_credits.increment_num_processed_completions(new_completions);
            total_unprocessed_completions += new_completions;
            sender_completion_amort_counter -= new_completions;
            had_new_completions = true;
        }

        // Process at most one epoch worth of completions per pass.
        // unsent_upstream_credits_packed uses the same byte packing as credit_epoch_array
        // (byte 0 = ch0, byte 1 = ch1), so decompose is a single add — no per-channel unpacking.
        if (total_unprocessed_completions >= EPOCH_SIZE) {
            unsent_upstream_credits_packed += credit_epoch_array[read_epoch];

            read_epoch = (read_epoch + 1) & EPOCH_MASK;
            total_unprocessed_completions -= EPOCH_SIZE;
            outbound_to_receiver_channel_pointers.num_free_slots += EPOCH_SIZE;
            did_epoch_decompose = true;
        }
    }

    sender_check_completions_timer.close();

    // -----------------------------------------------------------------------
    // Phase 4: Send upstream credits (at most 1 NOC write per pass)
    //
    // We check ch0 first, then ch1.  Only one channel sends per pass to
    // avoid starving the forward/send path with expensive NOC writes.
    // -----------------------------------------------------------------------
    uint32_t ch0_credits = unsent_upstream_credits_packed & 0xFF;
    uint32_t ch1_credits = (unsent_upstream_credits_packed >> 8) & 0xFF;
    sender_credits_upstream_timer.set_should_dump(
        ch0_credits >= SENDER_CREDIT_AMORTIZATION_FREQUENCY || ch1_credits >= SENDER_CREDIT_AMORTIZATION_FREQUENCY);
    sender_credits_upstream_timer.open();

    bool did_send_upstream_credits = false;
    if (ch0_credits >= SENDER_CREDIT_AMORTIZATION_FREQUENCY) {
        // 0x5 = TX_UPSTREAM: [tag:4][ch:4][credits:12][0:12]
        // WATCHER_RING_BUFFER_PUSH(0x50000000u | (0u << 24) | ((ch0_credits & 0xFFF) << 12));
        send_credits_to_upstream_workers<false /*deadlock_avoidance*/, false /*SKIP_LIVENESS*/>(
            local_sender_channel_worker_interface_0, ch0_credits, channel_connection_established[0]);
        unsent_upstream_credits_packed &= ~0xFFu;  // clear ch0 byte
        did_send_upstream_credits = true;
    } else if (ch1_credits >= SENDER_CREDIT_AMORTIZATION_FREQUENCY) {
        // 0x5 = TX_UPSTREAM: [tag:4][ch:4][credits:12][0:12]
        // WATCHER_RING_BUFFER_PUSH(0x50000000u | (1u << 24) | ((ch1_credits & 0xFFF) << 12));
        send_credits_to_upstream_workers<false /*deadlock_avoidance*/, true /*SKIP_LIVENESS*/>(
            local_sender_channel_worker_interface_1, ch1_credits, channel_connection_established[1]);
        unsent_upstream_credits_packed &= ~0xFF00u;  // clear ch1 byte
        did_send_upstream_credits = true;
    }

    sender_credits_upstream_timer.close();

    // Common gate: all non-send codepaths must have fired
    bool all_other_codepaths = had_new_completions && did_epoch_decompose && did_send_upstream_credits;

    // FULL: both packets sent AND all other codepaths
    bool both_sent = ch0_sent && ch1_sent;
    sender_full_timer.set_should_dump(both_sent && all_other_codepaths);
    sender_full_timer.close();

    // FULL_SINGLE: exactly one packet sent (XOR) AND all other codepaths
    bool exactly_one_sent = ch0_sent != ch1_sent;
    sender_full_with_send_timer.set_should_dump(exactly_one_sent && all_other_codepaths);
    sender_full_with_send_timer.close();

    return progress;
}

// ===========================================================================
//  LINE RECEIVER: Speedy step with forwarding support
// ===========================================================================

template <
    uint8_t receiver_channel,
    uint8_t to_receiver_pkts_sent_id,
    bool enable_first_level_ack,
    size_t DOWNSTREAM_EDM_SIZE,
    typename WriteTridTracker,
    typename ReceiverChannelBufferT,
    typename ReceiverChannelPointersT,
    typename DownstreamSenderT,
    typename LocalRelayInterfaceT,
    typename LocalTelemetryT>
FORCE_INLINE bool run_receiver_channel_step_line_speedy(
    ReceiverChannelBufferT& local_receiver_channel,
    std::array<DownstreamSenderT, DOWNSTREAM_EDM_SIZE>& downstream_edm_interfaces,
    LocalRelayInterfaceT& local_relay_interface,
    ReceiverChannelPointersT& receiver_channel_pointers,
    WriteTridTracker& receiver_channel_trid_tracker,
    std::array<uint8_t, num_eth_ports>& port_direction_table,
    ReceiverChannelResponseCreditSender& receiver_channel_response_credit_sender,
    const tt::tt_fabric::routing_l1_info_t& routing_table,
    LocalTelemetryT& local_fabric_telemetry) {
    // --- Profiling timers (compile-time disabled when bitfield == 0) ---
    NamedProfiler<
        CodeProfilingTimerType::SPEEDY_RECEIVER_FULL,
        code_profiling_enabled_timers_bitfield,
        code_profiling_buffer_base_addr>
        receiver_full_timer;
    NamedProfiler<
        CodeProfilingTimerType::SPEEDY_RECEIVER_FORWARD,
        code_profiling_enabled_timers_bitfield,
        code_profiling_buffer_base_addr>
        receiver_forward_timer;
    NamedProfiler<
        CodeProfilingTimerType::SPEEDY_RECEIVER_FLUSH,
        code_profiling_enabled_timers_bitfield,
        code_profiling_buffer_base_addr>
        receiver_flush_timer;

    bool progress = false;
    auto& wr_sent_counter = receiver_channel_pointers.wr_sent_counter;
    auto pkts_received = get_ptr_val<to_receiver_pkts_sent_id>();
    bool unwritten_packets = pkts_received != 0;

    // Full function timer: opened unconditionally, closed with should_dump at end
    receiver_full_timer.open();

    // -----------------------------------------------------------------------
    // Phase 1: Receive + forward/deliver one packet
    //
    // Unlike the neighbor-exchange speedy path, we support forwarding here.
    // We do NOT inspect src_ch_id — the sender-side epoch array handles
    // credit disambiguation.
    // -----------------------------------------------------------------------
    receiver_forward_timer.open();

    if (unwritten_packets) {
        static_assert(!ENABLE_RISC_CPU_DATA_CACHE, "ENABLE_RISC_CPU_DATA_CACHE must be disabled for speedy path");

        auto receiver_buffer_index = wr_sent_counter.get_buffer_index();
        tt_l1_ptr PACKET_HEADER_TYPE* packet_header = const_cast<PACKET_HEADER_TYPE*>(
            local_receiver_channel.template get_packet_header<PACKET_HEADER_TYPE>(receiver_buffer_index));

        // Check if we can forward (downstream has space or local-only delivery)
        ROUTING_FIELDS_TYPE cached_routing_fields;
        cached_routing_fields = packet_header->routing_fields;
        WATCHER_RING_BUFFER_PUSH((uint32_t)packet_header);
        WATCHER_RING_BUFFER_PUSH((uint32_t)&(packet_header->routing_fields));

        bool can_send_to_all_local_chip_receivers =
            can_forward_packet_completely(cached_routing_fields, downstream_edm_interfaces[0]);

        if (can_send_to_all_local_chip_receivers) {
            receiver_forward_timer.set_should_dump(unwritten_packets);

            // Single 4B aligned load to get payload_size_bytes + noc_send_type
            // instead of two separate uncached L1 reads inside receiver_forward_packet.
            auto packed = PACKET_HEADER_TYPE::PackedPayloadAndSendType::load(packet_header);

            receiver_forward_packet_impl<receiver_channel>(
                packet_header, cached_routing_fields, packed, downstream_edm_interfaces[0], line_current_write_trid);

            did_something = true;
            progress = true;
            if constexpr (FABRIC_TELEMETRY_BANDWIDTH) {
                update_bw_counters(packet_header, local_fabric_telemetry);
            }
            // channel_trimming_usage_recorder.set_receiver_channel_data_forwarded(receiver_channel);

            wr_sent_counter.increment();
            increment_local_update_ptr_val<to_receiver_pkts_sent_id>(-1);
            line_unacked_sends++;
            // 0xA = RX_FWD: [tag:4][trid:4][unacked:12][0:12]
            WATCHER_RING_BUFFER_PUSH(
                0xA0000000u | ((uint32_t)line_current_write_trid << 24) | ((line_unacked_sends & 0xFFF) << 12));
        }
    }

    receiver_forward_timer.close();

    // -----------------------------------------------------------------------
    // Phase 2: Ping-pong TRID flush + bulk completion
    //
    // All packets in a batch share a single TRID (line_current_write_trid).
    // When the batch threshold is hit, we flip to the other TRID and check
    // the previous batch's single TRID for completion — replacing the
    // per-slot flush loop with a single register read.
    //
    // The pending TRID is checked eagerly (every call) to minimize credit
    // return latency. Only the batch flip requires reaching the threshold.
    //
    // Completion credits are sent as a BULK count (no src_ch_id). The sender
    // side's epoch array decomposes them into per-channel credits.
    // -----------------------------------------------------------------------
    receiver_flush_timer.set_should_dump(
        line_has_pending_flush || (line_unacked_sends >= RECEIVER_CREDIT_AMORTIZATION_FREQUENCY));
    receiver_flush_timer.open();

    bool did_flush = false;
    if ((line_unacked_sends >= RECEIVER_CREDIT_AMORTIZATION_FREQUENCY) && !line_has_pending_flush) {
        line_pending_flush_trid = line_current_write_trid;
        line_pending_flush_batch_count = line_unacked_sends;
        line_current_write_trid = 1 - line_current_write_trid;
        line_has_pending_flush = true;
        // 0xB = RX_TRID_FLIP: [tag:4][pending_trid:4][batch:12][new_wr_trid:12]
        WATCHER_RING_BUFFER_PUSH(
            0xB0000000u | ((uint32_t)line_pending_flush_trid << 24) | ((line_pending_flush_batch_count & 0xFFF) << 12) |
            ((uint32_t)line_current_write_trid & 0xFFF));
    }
    if (line_has_pending_flush) {
        // Must check both NOCs: local delivery uses edm_to_local_chip_noc,
        // forwarding uses edm_to_downstream_noc. If they differ, both must be flushed
        // before we can safely release the receiver buffer slots.
        bool flushed = ncrisc_noc_nonposted_write_with_transaction_id_sent(
            tt::tt_fabric::edm_to_local_chip_noc, line_pending_flush_trid);
        if constexpr (tt::tt_fabric::edm_to_local_chip_noc != tt::tt_fabric::edm_to_downstream_noc) {
            flushed = flushed && ncrisc_noc_nonposted_write_with_transaction_id_sent(
                                     tt::tt_fabric::edm_to_downstream_noc, line_pending_flush_trid);
        }

        if (flushed) {
            auto& completion_counter = receiver_channel_pointers.completion_counter;
            completion_counter.increment_n(line_pending_flush_batch_count);
            // Send bulk completion — src_id=0 is used as the single shared credit channel.
            // The sender side does not use src_id for disambiguation; it uses the epoch array.
            receiver_send_completion_ack<true /*CHECK_BUSY*/>(
                receiver_channel_response_credit_sender, 0, line_pending_flush_batch_count);

            // 0xC = RX_FLUSH_DONE: [tag:4][trid:4][batch:12][remaining:12]
            WATCHER_RING_BUFFER_PUSH(
                0xC0000000u | ((uint32_t)line_pending_flush_trid << 24) |
                ((line_pending_flush_batch_count & 0xFFF) << 12) |
                ((line_unacked_sends - line_pending_flush_batch_count) & 0xFFF));
            line_unacked_sends -= line_pending_flush_batch_count;
            line_has_pending_flush = false;
            did_flush = true;
        }
    }

    receiver_flush_timer.close();

    // Full function timer: only dump when both packet forwarded AND flush completed
    receiver_full_timer.set_should_dump(progress && did_flush);
    receiver_full_timer.close();

    return progress;
}
