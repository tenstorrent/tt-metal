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
// ---------------------------------------------------------------------------
// Sender channel speedy step
// ---------------------------------------------------------------------------
// Differences from run_sender_channel_step_impl:
//   - Skips bubble flow control (static_assert ensures it's disabled)
//   - Always skips connection liveness check in inner loop
//   - Amortizes credits sent to upstream workers: accumulates completions/acks
//     in sender_amort_counter and only calls send_credits_to_upstream_workers
//     when counter >= SENDER_CREDIT_AMORTIZATION_FREQUENCY
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

    // --- Send packet if possible (unchanged from normal path) ---
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

        send_next_data<sender_channel_index, to_receiver_pkts_sent_id, false /*SKIP_CONNECTION_LIVENESS_CHECK*/>(
            local_sender_channel,
            local_sender_channel_worker_interface,
            outbound_to_receiver_channel_pointers,
            perf_telemetry_recorder);
        sender_amort_counter++;
        if constexpr (FABRIC_TELEMETRY_BANDWIDTH) {
            update_bw_counters(pkt_header, local_fabric_telemetry);
        }
        increment_local_update_ptr_val(sender_channel_free_slots_stream_id, 1);
    }

    if (sender_amort_counter > SENDER_CREDIT_AMORTIZATION_FREQUENCY) {
        // --- Always check for new completions from receiver (cheap read) ---
        int32_t completions = sender_channel_from_receiver_credits
                                  .template get_num_unprocessed_completions_from_receiver<ENABLE_RISC_CPU_DATA_CACHE>();
        if (completions) {
            outbound_to_receiver_channel_pointers.num_free_slots += completions;
            sender_channel_from_receiver_credits.increment_num_processed_completions(completions);

            completion_count += completions;

            // send_credits_to_upstream_workers<false /*deadlock_avoidance*/, false /*SKIP_LIVENESS*/>(
            //     local_sender_channel_worker_interface, sender_amort_counter, channel_connection_established);
            // sender_amort_counter = 0;
        }
    }

    // --- Amortized: only send credits to upstream workers every N completions/acks ---
    if (completion_count >= SENDER_CREDIT_AMORTIZATION_FREQUENCY) {
        // likely we are seeing an issue due to L1
        send_credits_to_upstream_workers<false /*deadlock_avoidance*/, false /*SKIP_LIVENESS*/>(
            local_sender_channel_worker_interface, completion_count, channel_connection_established);
        sender_amort_counter -= completion_count;
        completion_count = 0;
    }
    // if (!channel_connection_established) {
    //     auto check_connection_status =
    //         !channel_connection_established || local_sender_channel_worker_interface.has_worker_teardown_request();
    //     if (check_connection_status) {
    //         check_worker_connections<MY_ETH_CHANNEL, ENABLE_RISC_CPU_DATA_CACHE>(
    //             local_sender_channel_worker_interface, channel_connection_established,
    //             sender_channel_free_slots_stream_id);
    //     }
    // }

    // NO connection liveness check in inner loop (checked in outer loop between context switches)
    return progress;
}

// ---------------------------------------------------------------------------
// Receiver channel speedy step
// ---------------------------------------------------------------------------
// Differences from run_receiver_channel_step_impl:
//   - Contains an inner loop that processes up to RECEIVER_CREDIT_AMORTIZATION_FREQUENCY
//     packets in a single call before batch-flushing completions
//   - Leverages skip_src_ch_id_update (single sender channel) to cache src_ch_id
//   - Fused flush + completion model (static_assert ensures fuse_receiver_flush_and_completion_ptr)
//   - After inner loop, batch-flushes all outstanding completions
static uint32_t unacked_sends = 0;
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
    uint8_t src_ch_id = 0;  // receiver_channel_pointers.get_src_chan_id();

    // Inner loop: process up to RECEIVER_CREDIT_AMORTIZATION_FREQUENCY packets
    for (uint32_t pkt = 0; pkt < RECEIVER_CREDIT_AMORTIZATION_FREQUENCY; pkt++) {
        auto pkts_received = get_ptr_val<to_receiver_pkts_sent_id>();

        // --- ACK phase (if first-level ack enabled) ---
        bool unwritten_packets = pkts_received != 0;

        if (!unwritten_packets) {
            break;  // No more packets to process
        }

        // --- Forward packet ---
        router_invalidate_l1_cache<ENABLE_RISC_CPU_DATA_CACHE>();
        auto receiver_buffer_index = wr_sent_counter.get_buffer_index();
        tt_l1_ptr PACKET_HEADER_TYPE* packet_header = const_cast<PACKET_HEADER_TYPE*>(
            local_receiver_channel.template get_packet_header<PACKET_HEADER_TYPE>(receiver_buffer_index));

        // Single 4B aligned load at offset 40 to get payload_size_bytes + noc_send_type
        // instead of two separate uncached L1 reads
        auto packed = PACKET_HEADER_TYPE::PackedPayloadAndSendType::load(packet_header);

        did_something = true;
        progress = true;
        if constexpr (FABRIC_TELEMETRY_BANDWIDTH) {
            update_bw_counters(packet_header, local_fabric_telemetry);
        }
        channel_trimming_usage_recorder.set_receiver_channel_data_forwarded(receiver_channel);

        execute_chip_unicast_to_local_chip_impl(
            packet_header, packed.payload_size_bytes, packed.noc_send_type, receiver_buffer_index, receiver_channel);
        wr_sent_counter.increment();
        if constexpr (!enable_first_level_ack) {
            increment_local_update_ptr_val<to_receiver_pkts_sent_id>(-1);
        }
        unacked_sends++;
    }  // end inner loop

    // --- Batched completion: flush all outstanding completions ---
    // Uses the fused flush+completion model (fuse_receiver_flush_and_completion_ptr == true)
    if (unacked_sends >= RECEIVER_CREDIT_AMORTIZATION_FREQUENCY) {
        auto& completion_counter = receiver_channel_pointers.completion_counter;
        uint32_t num_completions = 0;
        // note that short-circuiting this loop does not result in any speedup.
        // the bottlneck is likely on sender side.
        while (!completion_counter.is_caught_up_to(wr_sent_counter)) {
            auto buf_idx = completion_counter.get_buffer_index();
            bool flushed = receiver_channel_trid_tracker.transaction_flushed(buf_idx);
            if (!flushed) {
                break;
            }
            receiver_channel_trid_tracker.clear_trid_at_buffer_slot(buf_idx);
            completion_counter.increment();
            num_completions++;
        }
        if (num_completions > 0) {
            receiver_send_completion_ack<ETH_TXQ_SPIN_WAIT_RECEIVER_SEND_COMPLETION_ACK>(
                receiver_channel_response_credit_sender, src_ch_id, num_completions);
            unacked_sends -= num_completions;
        }
    }

    return progress;
}
