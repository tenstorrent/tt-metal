// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// This header provides amortized credit-passing "speedy" step functions for the
// fabric erisc router.  They are only used when super_speedy_mode == true
// (single sender channel with non-zero amortization frequencies).
//
// Must be included AFTER fabric_erisc_router_ct_args.hpp and all other router
// headers that define helpers such as send_next_data, send_credits_to_upstream_workers,
// receiver_send_completion_ack, receiver_send_received_ack, etc.

static_assert(
    !super_speedy_mode || !enable_deadlock_avoidance,
    "super_speedy_mode is incompatible with deadlock avoidance (bubble flow control)");
static_assert(!super_speedy_mode || NUM_SENDER_CHANNELS == 1, "super_speedy_mode requires exactly 1 sender channel");

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
