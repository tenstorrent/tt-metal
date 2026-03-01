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

// Ping-pong TRID constants (must precede struct default initializers)
static constexpr uint8_t pingpong_trid_a = RX_CH_TRID_STARTS[0];
static constexpr uint8_t pingpong_trid_b = RX_CH_TRID_STARTS[0] + 1;
static_assert(
    !super_speedy_mode || NUM_TRANSACTION_IDS >= 2,
    "Ping-pong TRID requires at least 2 transaction IDs per receiver channel");
static_assert(!super_speedy_mode || pingpong_trid_a == 0, "Ping-pong TRID flip uses '1 - trid', requires trid_a == 0");

struct NeighborSenderState {
    size_t completion_count = 0;
    uint32_t sender_amort_counter = 0;
};

struct NeighborReceiverState {
    uint32_t unacked_sends = 0;
    uint8_t current_write_trid = pingpong_trid_a;
    uint8_t pending_flush_trid = pingpong_trid_b;
    uint32_t pending_flush_batch_count = 0;
    bool has_pending_flush = false;
};

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
    NeighborSenderState& ns,
    PerfTelemetryRecorder& perf_telemetry_recorder,
    LocalTelemetryT& local_fabric_telemetry) {
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
        ns.sender_amort_counter++;
        if constexpr (FABRIC_TELEMETRY_BANDWIDTH) {
            update_bw_counters(pkt_header, local_fabric_telemetry);
        }
        increment_local_update_ptr_val(sender_channel_free_slots_stream_id, 1);
    }

    // We only want to actually bother checking for completions after a certain number of sent packets are outstanding
    // since the instructions to actually process each inbound completion from receiver is somewhat costly
    bool check_completions = ns.sender_amort_counter > SENDER_CREDIT_AMORTIZATION_FREQUENCY;
    if (check_completions) {
        int32_t completions = sender_channel_from_receiver_credits
                                  .template get_num_unprocessed_completions_from_receiver<ENABLE_RISC_CPU_DATA_CACHE>();
        if (completions) {
            outbound_to_receiver_channel_pointers.num_free_slots += completions;
            sender_channel_from_receiver_credits.increment_num_processed_completions(completions);

            ns.completion_count += completions;
        }
    }

    // Similarly only send back the credit to the worker very infrequently since it's a very
    // expensive operation.
    bool send_credits = ns.completion_count >= SENDER_CREDIT_AMORTIZATION_FREQUENCY;
    if (send_credits) {
        send_credits_to_upstream_workers<false /*deadlock_avoidance*/, false /*SKIP_LIVENESS*/>(
            local_sender_channel_worker_interface, ns.completion_count, channel_connection_established);
        ns.sender_amort_counter -= ns.completion_count;
        ns.completion_count = 0;
    }
    return progress;
}

/*
 * A fast neighbour exchange only receiver channel step impl
 */
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
    NeighborReceiverState& nr,
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
            packet_header, packed.payload_size_bytes, packed.noc_send_type, nr.current_write_trid, receiver_channel);

        did_something = true;
        progress = true;
        if constexpr (FABRIC_TELEMETRY_BANDWIDTH) {
            update_bw_counters(packet_header, local_fabric_telemetry);
        }
        channel_trimming_usage_recorder.set_receiver_channel_data_forwarded(receiver_channel);

        wr_sent_counter.increment();
        increment_local_update_ptr_val<to_receiver_pkts_sent_id>(-1);
        nr.unacked_sends++;
    }

    // --- Ping-pong TRID flush ---
    // All packets in a batch share a single TRID (nr.current_write_trid). When the batch
    // threshold is hit, we flip to the other TRID and check the previous batch's single
    // TRID for completion — replacing the per-slot loop with a single register read.
    //
    // The pending TRID is checked eagerly (every call) to minimize credit return latency.
    // Only the batch flip requires reaching the threshold.
    // when we pass the threshold of unacked messages,
    if ((nr.unacked_sends >= RECEIVER_CREDIT_AMORTIZATION_FREQUENCY) && !nr.has_pending_flush) {
        nr.pending_flush_trid = nr.current_write_trid;
        nr.pending_flush_batch_count = nr.unacked_sends;
        nr.current_write_trid = 1 - nr.current_write_trid;
        nr.has_pending_flush = true;
    }
    if (nr.has_pending_flush) {
        bool flushed = ncrisc_noc_nonposted_write_with_transaction_id_sent(
            tt::tt_fabric::edm_to_local_chip_noc, nr.pending_flush_trid);

        if (flushed) {
            auto& completion_counter = receiver_channel_pointers.completion_counter;
            completion_counter.increment_n(nr.pending_flush_batch_count);
            receiver_send_completion_ack<false /*CHECK_BUSY*/>(
                receiver_channel_response_credit_sender, 0, nr.pending_flush_batch_count);

            nr.unacked_sends -= nr.pending_flush_batch_count;
            nr.has_pending_flush = false;
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
    line_speedy_mode ? next_power_of_2((REMOTE_RECEIVER_NUM_BUFFERS + EPOCH_SIZE - 1) / EPOCH_SIZE) : 1;
static constexpr uint32_t NUM_CREDIT_EPOCHS = MIN_CREDIT_EPOCHS * 2;
static_assert((NUM_CREDIT_EPOCHS & (NUM_CREDIT_EPOCHS - 1)) == 0, "NUM_CREDIT_EPOCHS must be a power of 2");
static_assert(
    !line_speedy_mode || (NUM_CREDIT_EPOCHS > (REMOTE_RECEIVER_NUM_BUFFERS + EPOCH_SIZE - 1) / EPOCH_SIZE),
    "Epoch array must be strictly larger than max in-flight epochs to avoid write/read overlap");
static constexpr uint32_t EPOCH_MASK = NUM_CREDIT_EPOCHS - 1;

enum class EpochAdvanceStrategy { BRANCHING, BRANCHLESS, POINTER };
static constexpr EpochAdvanceStrategy epoch_advance_strategy = EpochAdvanceStrategy::BRANCHING;

struct LineSenderState {
    uint32_t write_epoch = 0;
    uint32_t read_epoch = 0;
    uint32_t epoch_pkt_count = 0;
    uint32_t epoch_accumulator = 0;
    uint32_t unsent_upstream_credits_packed = 0;
    uint32_t total_unprocessed_completions = 0;
    uint32_t sender_completion_amort_counter = 0;
};

static constexpr uint8_t line_pingpong_trid_a = RX_CH_TRID_STARTS[0];
static constexpr uint8_t line_pingpong_trid_b = RX_CH_TRID_STARTS[0] + 1;
static_assert(
    !line_speedy_mode || NUM_TRANSACTION_IDS >= 2,
    "Ping-pong TRID requires at least 2 transaction IDs per receiver channel");
static_assert(
    !line_speedy_mode || line_pingpong_trid_a == 0, "Ping-pong TRID flip uses '1 - trid', requires trid_a == 0");

struct LineReceiverState {
    uint32_t line_unacked_sends = 0;
    uint32_t line_pending_flush_batch_count = 0;
    uint8_t line_current_write_trid = line_pingpong_trid_a;
    uint8_t line_pending_flush_trid = line_pingpong_trid_b;
    bool line_has_pending_flush = false;
};

// Empty state for non-speedy mode — zero-size, no stack cost.
struct NoOpSpeedyState {};

// Compile-time type aliases that resolve to the active speedy mode's state types.
// Only one of super_speedy_mode / line_speedy_mode / neither can be active.
using SpeedySenderState = std::conditional_t<
    super_speedy_mode,
    NeighborSenderState,
    std::conditional_t<line_speedy_mode, LineSenderState, NoOpSpeedyState>>;

using SpeedyReceiverState = std::conditional_t<
    super_speedy_mode,
    NeighborReceiverState,
    std::conditional_t<line_speedy_mode, LineReceiverState, NoOpSpeedyState>>;

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
    LineSenderState& ss,
    uint32_t* credit_epoch_array,
    PerfTelemetryRecorder& perf_telemetry_recorder,
    LocalTelemetryT& local_fabric_telemetry) {
    bool busy;
    if constexpr (!ETH_TXQ_SPIN_WAIT_SEND_NEXT_DATA) {
        busy = internal_::eth_txq_is_busy(sender_txq_id);
    }
    bool receiver_has_space = outbound_to_receiver_channel_pointers.has_space_for_packet();
    uint32_t free_slots = get_ptr_val(sender_channel_free_slots_stream_id);
    bool has_unsent_packet = free_slots != WorkerInterfaceT::num_buffers;
    bool can_send = receiver_has_space && has_unsent_packet;

    if constexpr (!ETH_TXQ_SPIN_WAIT_SEND_NEXT_DATA) {
        can_send = can_send && !busy;
    }

    // No epoch back-pressure check needed: the epoch array is sized to 2x the
    // max in-flight epochs (bounded by REMOTE_RECEIVER_NUM_BUFFERS / EPOCH_SIZE),
    // so write_epoch can never wrap into read_epoch.
    if (can_send) {
        // --- Send the packet (inlined for codegen efficiency) ---
        auto* pkt_header = reinterpret_cast<volatile tt_l1_ptr PACKET_HEADER_TYPE*>(
            local_sender_channel.get_cached_next_buffer_slot_addr());
        {
            auto& remote_receiver_num_free_slots = outbound_to_receiver_channel_pointers.num_free_slots;
            uint32_t src_addr = local_sender_channel.get_cached_next_buffer_slot_addr();
            bool busy = internal_::eth_txq_is_busy(sender_txq_id);
            const size_t payload_size_bytes = pkt_header->get_payload_size_including_header();

            const auto dest_addr = outbound_to_receiver_channel_pointers.remote_receiver_channel_address_ptr;

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
        // Accumulate into a scalar (register) instead of indexing into the array
        // every packet. Only flush to the array on epoch advance.
        static constexpr uint32_t credit_inc = 1u << (sender_channel_index * 8);
        ss.epoch_accumulator += credit_inc;
        ss.epoch_pkt_count++;
        ss.sender_completion_amort_counter++;
        if (ss.epoch_pkt_count == EPOCH_SIZE) {
            credit_epoch_array[ss.write_epoch] = ss.epoch_accumulator;
            ss.epoch_accumulator = 0;
            ss.write_epoch = (ss.write_epoch + 1) & EPOCH_MASK;
            ss.epoch_pkt_count = 0;
            credit_epoch_array[ss.write_epoch] = 0;
        }
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
    LineSenderState& ss,
    uint32_t* credit_epoch_array,
    PerfTelemetryRecorder& perf_telemetry_recorder,
    LocalTelemetryT& local_fabric_telemetry) {
    bool progress = false;

    bool ch0_sent = line_speedy_send_one_packet<0, to_receiver_pkts_sent_id>(
        local_sender_channel_0,
        local_sender_channel_worker_interface_0,
        outbound_to_receiver_channel_pointers,
        local_sender_channel_free_slots_stream_ids[0],
        ss,
        credit_epoch_array,
        perf_telemetry_recorder,
        local_fabric_telemetry);

    bool ch1_sent = line_speedy_send_one_packet<1, to_receiver_pkts_sent_id>(
        local_sender_channel_1,
        local_sender_channel_worker_interface_1,
        outbound_to_receiver_channel_pointers,
        local_sender_channel_free_slots_stream_ids[1],
        ss,
        credit_epoch_array,
        perf_telemetry_recorder,
        local_fabric_telemetry);

    progress = ch0_sent || ch1_sent;

    // -----------------------------------------------------------------------
    // Check completions and decompose via credit epoch array
    //
    // === Credit Epoch Array Book-keeping ===
    //
    // The receiver sends BULK completion credits — a single count with no
    // per-sender-channel breakdown.  The sender must decompose these bulk
    // credits back into per-channel counts to know how many upstream credits
    // to propagate for each sender channel.
    //
    // We do this via a circular buffers of "epochs", where each epoch represents
    // EPOCH_SIZE consecutive packets sent (across all channels). An epoch is the
    // amortization granularity of credit handling. Each epoch slot is a uint32_t
    // with per-channel counts packed as bytes:
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
    // The high level approach is as follows:
    //   1. Each epoch holds a fixed number of packet/credit counts
    //      a. These counts are tracked per channel separately (see diagram above)
    //   2. For each packet send, we accumulate a count into the current epoch in the array
    //   3. When the count for that epoch reaches the epoch size (amortization granularity),
    //      we advance the epoch. This epoch we just "closed" can now be checked against for
    //      completions
    //   4. Critical: We only check for completions after we have >= epoch size worth of
    //      outstanding sends
    //   5. We have additional levels of amortization for the credit response to the producer(s)
    //      since noc requests are relatively expensive.
    //
    // Additional algorithm notes:
    //   - At most only one channel sends acks per pass to keep the send path responsive
    //     (this is not a functional requirement, but a performance one)
    //   - Write_epoch is always ahead of or equal to read_epoch (mod NUM_CREDIT_EPOCHS)
    //   - Completions are consumed in EPOCH_SIZE chunks (receiver also sends
    //     in batches of RECEIVER_CREDIT_AMORTIZATION_FREQUENCY which should
    //     be a multiple of EPOCH_SIZE or equal to it)
    //   - credit_epoch_array[next_epoch] is always safe to clear because it is
    //     strictly ahead of read_epoch
    //   - The epoch array is always bigger than the total number of epochs for outstanding
    //     sends to avoid a race where sender fills the epoch array before we've cleared any
    //     entries
    // -----------------------------------------------------------------------

    // Amortize completion checks: only check after enough sends have occurred,
    // matching the neighbor exchange pattern. This avoids the ~25 cycle cost of
    // get_num_unprocessed_completions_from_receiver on iterations with no progress.
    bool check_completions = ss.sender_completion_amort_counter > SENDER_CREDIT_AMORTIZATION_FREQUENCY;

    if (check_completions) {
        int32_t new_completions =
            shared_completion_credits
                .template get_num_unprocessed_completions_from_receiver<ENABLE_RISC_CPU_DATA_CACHE>();
        if (new_completions) {
            shared_completion_credits.increment_num_processed_completions(new_completions);
            ss.total_unprocessed_completions += new_completions;
            ss.sender_completion_amort_counter -= new_completions;
        }

        // Process at most one epoch worth of completions per pass.
        // unsent_upstream_credits_packed uses the same byte packing as credit_epoch_array
        // (byte 0 = ch0, byte 1 = ch1), so decompose is a single add — no per-channel unpacking.
        if (ss.total_unprocessed_completions >= EPOCH_SIZE) {
            ss.unsent_upstream_credits_packed += credit_epoch_array[ss.read_epoch];

            ss.read_epoch = (ss.read_epoch + 1) & EPOCH_MASK;
            ss.total_unprocessed_completions -= EPOCH_SIZE;
            outbound_to_receiver_channel_pointers.num_free_slots += EPOCH_SIZE;
        }
    }

    // -----------------------------------------------------------------------
    // Send upstream credits (at most 1 NOC write per pass)
    //
    // Only one channel sends per pass to avoid starving the forward/send path
    // with expensive NOC writes.
    // -----------------------------------------------------------------------

    uint32_t ch0_credits = ss.unsent_upstream_credits_packed & 0xFF;
    if (ch0_credits >= SENDER_CREDIT_AMORTIZATION_FREQUENCY) {
        send_credits_to_upstream_workers<false /*deadlock_avoidance*/, false /*SKIP_LIVENESS*/>(
            local_sender_channel_worker_interface_0, ch0_credits, channel_connection_established[0]);
        ss.unsent_upstream_credits_packed &= ~0xFFu;  // clear ch0 byte
    } else {
        uint32_t ch1_credits = (ss.unsent_upstream_credits_packed >> 8) & 0xFF;
        if (ch1_credits >= SENDER_CREDIT_AMORTIZATION_FREQUENCY) {
            send_credits_to_upstream_workers<false /*deadlock_avoidance*/, true /*SKIP_LIVENESS*/>(
                local_sender_channel_worker_interface_1, ch1_credits, channel_connection_established[1]);
            ss.unsent_upstream_credits_packed &= ~0xFF00u;  // clear ch1 byte
        }
    }

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
    LineReceiverState& rs,
    LocalTelemetryT& local_fabric_telemetry) {
    bool progress = false;
    auto& wr_sent_counter = receiver_channel_pointers.wr_sent_counter;
    auto pkts_received = get_ptr_val<to_receiver_pkts_sent_id>();
    bool unwritten_packets = pkts_received != 0;
    // -----------------------------------------------------------------------
    // Receive + forward/deliver one packet
    //
    // Unlike the neighbor-exchange speedy path, we support forwarding here.
    // We do NOT inspect src_ch_id — the sender-side epoch array handles
    // credit disambiguation.
    // -----------------------------------------------------------------------
    if (unwritten_packets) {
        static_assert(!ENABLE_RISC_CPU_DATA_CACHE, "ENABLE_RISC_CPU_DATA_CACHE must be disabled for speedy path");

        auto receiver_buffer_index = wr_sent_counter.get_buffer_index();
        tt_l1_ptr PACKET_HEADER_TYPE* packet_header = const_cast<PACKET_HEADER_TYPE*>(
            local_receiver_channel.template get_packet_header<PACKET_HEADER_TYPE>(receiver_buffer_index));

        // Check if we can forward (downstream has space or local-only delivery)
        ROUTING_FIELDS_TYPE cached_routing_fields;
        cached_routing_fields = packet_header->routing_fields;

        bool can_send_to_all_local_chip_receivers =
            can_forward_packet_completely(cached_routing_fields, downstream_edm_interfaces[0]);

        if (can_send_to_all_local_chip_receivers) {
            // Single 4B aligned load to get payload_size_bytes + noc_send_type
            // instead of two separate uncached L1 reads inside receiver_forward_packet.
            auto packed = PACKET_HEADER_TYPE::PackedPayloadAndSendType::load(packet_header);

            receiver_forward_packet_impl<receiver_channel>(
                packet_header, cached_routing_fields, packed, downstream_edm_interfaces[0], rs.line_current_write_trid);

            did_something = true;
            progress = true;
            if constexpr (FABRIC_TELEMETRY_BANDWIDTH) {
                update_bw_counters(packet_header, local_fabric_telemetry);
            }

            wr_sent_counter.increment();
            increment_local_update_ptr_val<to_receiver_pkts_sent_id>(-1);
            rs.line_unacked_sends++;
        }
    }

    // -----------------------------------------------------------------------
    // Ping-pong TRID flush + bulk completion
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
    bool did_flush = false;
    if ((rs.line_unacked_sends >= RECEIVER_CREDIT_AMORTIZATION_FREQUENCY) && !rs.line_has_pending_flush) {
        rs.line_pending_flush_trid = rs.line_current_write_trid;
        rs.line_pending_flush_batch_count = rs.line_unacked_sends;
        rs.line_current_write_trid = 1 - rs.line_current_write_trid;
        rs.line_has_pending_flush = true;
    }
    if (rs.line_has_pending_flush) {
        // Must check both NOCs: local delivery uses edm_to_local_chip_noc,
        // forwarding uses edm_to_downstream_noc. If they differ, both must be flushed
        // before we can safely release the receiver buffer slots.
        bool flushed = ncrisc_noc_nonposted_write_with_transaction_id_sent(
            tt::tt_fabric::edm_to_local_chip_noc, rs.line_pending_flush_trid);
        if constexpr (tt::tt_fabric::edm_to_local_chip_noc != tt::tt_fabric::edm_to_downstream_noc) {
            flushed = flushed && ncrisc_noc_nonposted_write_with_transaction_id_sent(
                                     tt::tt_fabric::edm_to_downstream_noc, rs.line_pending_flush_trid);
        }

        if (flushed) {
            auto& completion_counter = receiver_channel_pointers.completion_counter;
            completion_counter.increment_n(rs.line_pending_flush_batch_count);
            // Send bulk completion — src_id=0 is currently hardcoded to enable re-use of the
            // receiver_send_completion_ack function until all receiver channel steps use this
            // style of implementation. The sender side does not use src_id for disambiguation;
            // it uses the epoch array.
            receiver_send_completion_ack<true /*CHECK_BUSY*/>(
                receiver_channel_response_credit_sender, 0, rs.line_pending_flush_batch_count);

            rs.line_unacked_sends -= rs.line_pending_flush_batch_count;
            rs.line_has_pending_flush = false;
            did_flush = true;
        }
    }

    return progress;
}
