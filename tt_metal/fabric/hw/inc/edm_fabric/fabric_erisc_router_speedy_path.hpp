// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// This header provides amortized credit-passing "speedy" step functions for the
// fabric erisc router. They are only used when super_speedy_mode == true
// for the explicit VC0 worker-only fast path.
//
// Must be included AFTER fabric_erisc_router_ct_args.hpp and all other router
// headers that define helpers such as send_next_data, send_credits_to_upstream_workers,
// receiver_send_completion_ack, receiver_send_received_ack, etc.

static constexpr bool speedy_mode_has_non_worker_vc0_sender = []() constexpr {
    for (size_t ch = 1; ch < ACTUAL_VC0_SENDER_CHANNELS; ++ch) {
        if (is_sender_channel_serviced[ch]) {
            return true;
        }
    }
    return false;
}();

static_assert(
    !super_speedy_mode || !enable_deadlock_avoidance,
    "super_speedy_mode is incompatible with deadlock avoidance (bubble flow control)");
static_assert(
    !super_speedy_mode || !ENABLE_FIRST_LEVEL_ACK_VC0, "super_speedy_mode is incompatible with first-level ack on VC0");
static_assert(
    !super_speedy_mode || !speedy_mode_has_non_worker_vc0_sender,
    "super_speedy_mode requires all serviced VC0 senders beyond channel 0 to be trimmed");
static_assert(
    !super_speedy_mode || !is_receiver_channel_serviced[VC0_RECEIVER_CHANNEL] || disable_rx_ch0_forwarding,
    "super_speedy_mode requires RX channel 0 forwarding to be disabled when RX0 is serviced");

struct SpeedySenderState {
    size_t completion_count = 0;
    uint32_t sender_amort_counter = 0;
    uint32_t connection_liveness_check_counter = 0;
};

template <size_t VC_ID>
struct SpeedyReceiverState {
    uint32_t unacked_sends = 0;
    uint8_t current_write_trid = RX_CH_TRID_STARTS[VC_ID];
    uint8_t pending_flush_trid = RX_CH_TRID_STARTS[VC_ID] + 1;
    uint32_t pending_flush_batch_count = 0;
    bool has_pending_flush = false;
};

template <size_t VC_ID>
struct NoOpSpeedyReceiverState {};

struct NoOpSpeedySenderState {};

template <bool SuperSpeedyModeEnabled>
using ActualSpeedySenderState = std::conditional_t<SuperSpeedyModeEnabled, SpeedySenderState, NoOpSpeedySenderState>;

template <bool SuperSpeedyModeEnabled, size_t VC_ID>
using ActualSpeedyReceiverState =
    std::conditional_t<SuperSpeedyModeEnabled, SpeedyReceiverState<VC_ID>, NoOpSpeedyReceiverState<VC_ID>>;

template <bool SuperSpeedyModeEnabled, size_t VC_ID>
FORCE_INLINE void speedy_state_copy_in(
    ActualSpeedySenderState<SuperSpeedyModeEnabled>& local_sender,
    ActualSpeedyReceiverState<SuperSpeedyModeEnabled, VC_ID>& local_receiver,
    const ActualSpeedySenderState<SuperSpeedyModeEnabled>& persistent_sender,
    const ActualSpeedyReceiverState<SuperSpeedyModeEnabled, VC_ID>& persistent_receiver) {
    if constexpr (SuperSpeedyModeEnabled) {
        local_sender = persistent_sender;
        local_receiver = persistent_receiver;
    }
}

template <bool SuperSpeedyModeEnabled, size_t VC_ID>
FORCE_INLINE void speedy_state_copy_out(
    ActualSpeedySenderState<SuperSpeedyModeEnabled>& persistent_sender,
    ActualSpeedyReceiverState<SuperSpeedyModeEnabled, VC_ID>& persistent_receiver,
    const ActualSpeedySenderState<SuperSpeedyModeEnabled>& local_sender,
    const ActualSpeedyReceiverState<SuperSpeedyModeEnabled, VC_ID>& local_receiver) {
    if constexpr (SuperSpeedyModeEnabled) {
        persistent_sender = local_sender;
        persistent_receiver = local_receiver;
    }
}

/*
 * A fast neighbour exchange only sender channel step impl
 */
template <
    uint8_t sender_channel_index,
    uint8_t to_receiver_pkts_sent_id,
    size_t SENDER_CREDIT_AMORTIZATION_FREQUENCY_LOCAL,
    bool MANAGE_CONNECTION_LIVENESS_IN_SPEEDY_HELPER,
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
    SpeedySenderState& sender_state) {
    bool progress = false;

    bool receiver_has_space_for_packet = outbound_to_receiver_channel_pointers.has_space_for_packet();
    uint32_t free_slots = get_ptr_val(sender_channel_free_slots_stream_id);
    bool has_unsent_packet = free_slots != WorkerInterfaceT::num_buffers;
    bool can_send = receiver_has_space_for_packet && has_unsent_packet;

#if defined(ARCH_BLACKHOLE)
    // [#45872 LOST-vs-RESET PROBE] word[23] = min free_slots since last TX. At the sync-barrier hang (no more
    // TX resets it) this isolates the sync window: ==num_buffers -> decrement NEVER landed (LOST);
    // <num_buffers -> it landed (dipped) then got reset (RESET).
    fabric_dbg_track_min_free_since_tx(free_slots);
    // [#45872 OCCUPANCY COMPARE] Channel-0 (stream 22) only. Once the sender has quiesced (ACK set), its write
    // count is frozen, so occupancy changes only as WE forward. Compute the counter-based true occupancy from the
    // occupancy latched at STOP minus our own forward delta, and the register-based occupancy from get_ptr_val --
    // both read locally in THIS iteration (true simultaneity). At the settled end-of-test they should match; a gap
    // means the register isn't tracking the drain. word[7]=occ_true (counter), word[15]=occ_reg (register).
    if constexpr (sender_channel_index == 0) {
        if (*reinterpret_cast<volatile uint32_t*>(MEM_AERISC_HANDSHAKE_ACK_ADDR) != 0u) {
            static bool occ_latched = false;
            static bool rp_finish_latched = false;
            static uint32_t occ_at_stop = 0, f0 = 0, rc_at_begin = 0;
            const uint32_t fwd = local_sender_channel_worker_interface.get_local_read_counter();  // router read pointer
            const uint32_t rc = fabric_get_retrain_count();
            if (!occ_latched) {
                const uint32_t w8_free =
                    *reinterpret_cast<volatile uint32_t*>(MEM_AERISC_DBELL_RECV_CUM_ADDR);  // sender free-slots @STOP
                occ_at_stop =
                    (w8_free <= WorkerInterfaceT::num_buffers) ? (WorkerInterfaceT::num_buffers - w8_free) : 0u;
                f0 = fwd;
                rc_at_begin = rc;
                occ_latched = true;
                *reinterpret_cast<volatile uint32_t*>(MEM_AERISC_RP_BEGIN_ADDR) =
                    fwd;  // word[13] read ptr @ retrain-begin
            }
            // [#45872 READ-POINTER] retrain-finish = first retrain_count change after ACK; settled = live last value.
            if (occ_latched && !rp_finish_latched && rc != rc_at_begin) {
                rp_finish_latched = true;
                *reinterpret_cast<volatile uint32_t*>(MEM_AERISC_RP_FINISH_ADDR) =
                    fwd;  // word[14] read ptr @ retrain-finish
            }
            *reinterpret_cast<volatile uint32_t*>(MEM_AERISC_RP_SETTLED_ADDR) = fwd;  // word[6] read ptr live (settled)
            const uint32_t fwd_since = fwd - f0;
            const uint32_t occ_true = (fwd_since <= occ_at_stop) ? (occ_at_stop - fwd_since) : 0u;
            const uint32_t occ_reg =
                (free_slots <= WorkerInterfaceT::num_buffers) ? (WorkerInterfaceT::num_buffers - free_slots) : 0u;
            *reinterpret_cast<volatile uint32_t*>(MEM_AERISC_OCC_TRUE_ADDR) = occ_true;  // word[7]
            *reinterpret_cast<volatile uint32_t*>(MEM_AERISC_OCC_REG_ADDR) = occ_reg;    // word[15]

            // [#45872 DRAIN TIME-SERIES] Time-resolved trace of the post-retrain drain. Sampled HERE, outside
            // if(can_send), so the trace keeps running after forwarding stops and the settle is visible -- the
            // 0xE4 probe further down is deliberately left alone and still marks when the drain reaches
            // can_send. Pushed only when reg or occ MOVES, so the static link-down window costs a single entry
            // and the 32-entry ring is spent on the drain. Packing/tags: FABRIC_DBG_DRAIN_TS_TAG, eth_fw_api.h.
            // All state lives inside the (noinline) helper -- keeping it out of this frame is mandatory, not
            // stylistic: inlining it here took the router main loop to 3680B of stack against a hard 1912B
            // cap and failed the build. Self-check needs no extra state: word[6]/[7]/[15] above are written
            // every iteration, so a live word[6] with an EMPTY ring means the (guaranteed) first push was
            // lost rather than never attempted. See the FABRIC_DBG_DRAIN notes in eth_fw_api.h.
            fabric_dbg_drain_sample(free_slots, occ_true, rc);
        }
    }
#endif

    if constexpr (!ETH_TXQ_SPIN_WAIT_SEND_NEXT_DATA) {
        can_send = can_send && !internal_::eth_txq_is_busy(sender_txq_id);
    }

#if defined(ARCH_BLACKHOLE)
    // [SLOT-CONTENT PROBE] Read the bytes in the slot we would transmit next, unconditionally --
    // including when the gate says the channel is empty. That is the whole point: if the gate claims
    // nothing is queued but the slot contains an atomic-inc header, the packet demonstrably arrived
    // and the router just isn't picking it up, proven from memory contents rather than a NoC ack.
    // [DOORBELL CROSS-CHECK] Record the stream id we poll and the raw uncached value we read from it,
    // so both can be compared against an independent host-side read of the same hardware register.
    fabric_dbg_set_polled_doorbell(sender_channel_free_slots_stream_id, free_slots);

    fabric_dbg_set_next_slot_content(
        local_sender_channel.get_cached_next_buffer_slot_addr(), sender_channel_free_slots_stream_id);

    // [SEND-GATE PROBE] Record why this channel will/won't transmit. At end of run the only packet
    // left is the sync packet, so a frozen gate state here is the reason the barrier wedged.
    fabric_dbg_set_sender_gate(
        sender_channel_index,
        outbound_to_receiver_channel_pointers.num_free_slots,
        free_slots,
        has_unsent_packet,
        receiver_has_space_for_packet,
        can_send);

    // [DECREMENT-LOST vs DECREMENT-RESET PROBE] Min free_slots seen while the sync packet occupies the
    // slot (payload demonstrably landed via L1 scan, but free_slots reads 32). ==32 => the worker's
    // doorbell decrement never landed; <32 => it landed then got reset. See eth_fw_api.h.
    // Min free_slots since the last transmitted packet. Reset on every TX (see fabric_dbg_inc_tx_pkt_count),
    // so after traffic stops it isolates the barrier window, where only the sync packet can ring the
    // doorbell. Reads no L1 -> immune to the cache staleness that broke the header-gated version.
    // [SELF-LOOPBACK PROBE] Read this core's OWN free-slots register back through the NOC and latch it
    // beside the local read (word[25]) taken in this same iteration. See fabric_dbg_latch_loopback().
    //
    // Rate limiting matters here: this sits in the router's hot path, and a NOC read plus barrier on
    // every iteration would both wreck throughput and perturb the timing we are trying to observe. It is
    // therefore gated on a long run of consecutive EMPTY reads -- any packet at all resets the streak, so
    // it cannot fire during normal traffic, only once the channel has been idle far longer than any gap
    // in the 100M-packet stream. It then re-samples periodically so we can see whether the value ever
    // changes while wedged.
    //
    // [TRAFFIC-VOLUME GATED] With this probe ungated the router wedges in the SENDER STEP
    // (RESUME_PHASE_TX_STEP_ENTER 0x11) when a link is down -- the noc_async_read_barrier never completes on
    // the recovering NOC. A link-up gate is NOT enough: the workers do a round-0 barrier BEFORE traffic, so
    // the channel is idle AND the link is still up at startup, the probe fires, then the injected down wedges
    // it -- so recovery never runs. The reliable discriminator is TRAFFIC VOLUME: only the END-of-run barrier
    // hang has ~100M packets already sent, whereas the startup window has ~0. So fire only once TX_PKT_COUNT
    // is large (post-recovery, post-traffic) AND the link is up. During the down/retrain window TX is tiny ->
    // we skip -> recovery proceeds; at the end-of-run hang it fires and gives the real reading.
    {
        static uint32_t empty_streak = 0;
        static uint32_t loopback_samples = 0;
        if (free_slots == WorkerInterfaceT::num_buffers) {
            empty_streak++;
            constexpr uint32_t LOOPBACK_FIRST = 1u << 20;
            constexpr uint32_t LOOPBACK_PERIOD_MASK = (1u << 18) - 1;
            constexpr uint32_t LOOPBACK_MIN_TX = 50000000u;  // only after most of the 100M-packet stream
            if (empty_streak >= LOOPBACK_FIRST && (empty_streak & LOOPBACK_PERIOD_MASK) == 0 &&
                *reinterpret_cast<volatile uint32_t*>(MEM_AERISC_TX_PKT_COUNT_ADDR) > LOOPBACK_MIN_TX &&
                fabric_dbg_link_is_up()) {
                // Destination must be real L1. A `static` here lands in ERISC local data memory
                // (observed at 0xFFB014E0), which is not NOC-addressable -- the watcher rejects it as
                // "Local L1 address overflow". The debug slot itself is L1 (the host reads it over NOC),
                // so land the value directly in word[23], then repack it locally with the sample count.
                const uint64_t self_addr = get_noc_addr(static_cast<uint32_t>(STREAM_REG_ADDR(
                    sender_channel_free_slots_stream_id, STREAM_REMOTE_DEST_BUF_SPACE_AVAILABLE_REG_INDEX)));
                noc_async_read(self_addr, MEM_AERISC_SYNC_MIN_FREE_ADDR, sizeof(uint32_t));
                noc_async_read_barrier();
                invalidate_l1_cache();
                const uint32_t loopback_val = *reinterpret_cast<volatile uint32_t*>(MEM_AERISC_SYNC_MIN_FREE_ADDR);
                loopback_samples++;
                // [#45872] loopback DISABLED: word[23] is now owned by the LOST-vs-RESET min-free probe.
                (void)loopback_val;
                // fabric_dbg_latch_loopback(loopback_val, loopback_samples);
            }
        } else {
            empty_streak = 0;
        }
    }
#endif

    if (can_send) {
        progress = true;

#if defined(ARCH_BLACKHOLE)
        // [#45872 DRAIN-TRACE] Channel-0, post-retrain: push the register (pre-forward free_slots) on each of the
        // first ~24 forwards after a retrain. Ring: tag 0xE4 [31:24], forward index [23:16], free_slots [15:0].
        // A ramp = honest drain. Now the ONLY thing filling the ring, so it won't be evicted by recover stages.
        if constexpr (sender_channel_index == 0) {
            static uint32_t drain_budget = 24;
            if (fabric_get_retrain_count() != 0u && drain_budget != 0u) {
                WATCHER_RING_BUFFER_PUSH(
                    (0xE4u << 24) | (((24u - drain_budget) & 0xFFu) << 16) | (free_slots & 0xFFFFu));
                drain_budget--;
            }
        }
#endif

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
#if defined(ARCH_BLACKHOLE)
            fabric_dbg_inc_tx_pkt_count();  // [TX-COUNT] one successful eth-link send by ERISC0
#endif
        }
        sender_state.sender_amort_counter++;
        if constexpr (FABRIC_TELEMETRY_BANDWIDTH) {
            update_bw_counters(pkt_header, local_fabric_telemetry);
        }
        increment_local_update_ptr_val(sender_channel_free_slots_stream_id, 1);
    }

    // We only want to actually bother checking for completions after a certain number of sent packets are outstanding
    // since the instructions to actually process each inbound completion from receiver is somewhat costly
    //
    // [TAIL-STALL FIX 2B - TEMPORARILY REVERTED for A/B confirmation] The force-poll-when-out-of-slots
    // clause below is removed to confirm it was what eliminated the tail-stalls. If tail-stalls reappear
    // with it gone, the fix is confirmed. Restore by re-adding:
    //     || !outbound_to_receiver_channel_pointers.has_space_for_packet()
    bool check_completions = sender_state.sender_amort_counter >= SENDER_CREDIT_AMORTIZATION_FREQUENCY_LOCAL;
    if (check_completions) {
        int32_t completions = sender_channel_from_receiver_credits
                                  .template get_num_unprocessed_completions_from_receiver<ENABLE_RISC_CPU_DATA_CACHE>();
        if (completions) {
            outbound_to_receiver_channel_pointers.num_free_slots += completions;
            sender_channel_from_receiver_credits.increment_num_processed_completions(completions);

            sender_state.completion_count += completions;
        }
    }

    // Similarly only send back the credit to the worker very infrequently since it's a very
    // expensive operation.
    bool send_credits = sender_state.completion_count >= SENDER_CREDIT_AMORTIZATION_FREQUENCY_LOCAL;
    if (send_credits) {
        send_credits_to_upstream_workers<false /*deadlock_avoidance*/, false /*SKIP_LIVENESS*/>(
            local_sender_channel_worker_interface, sender_state.completion_count, channel_connection_established);
        sender_state.sender_amort_counter -= sender_state.completion_count;
        sender_state.completion_count = 0;
    }

    if constexpr (MANAGE_CONNECTION_LIVENESS_IN_SPEEDY_HELPER) {
        // Speedy sender service cadence is still decoupled from the credit
        // amortization cadence. A period of N gives a 1:N liveness polling
        // ratio for the helper-managed VC0 path.
        static constexpr uint32_t speedy_connection_liveness_check_period = 1;
        static_assert(speedy_connection_liveness_check_period > 0);

        sender_state.connection_liveness_check_counter++;
        const bool reached_liveness_poll_period =
            sender_state.connection_liveness_check_counter >= speedy_connection_liveness_check_period;
        if (reached_liveness_poll_period) {
            // Reset on the scheduled poll boundary so the cadence stays exact
            // even when there is no open/close transition to process.
            sender_state.connection_liveness_check_counter = 0;

            auto check_connection_status =
                !channel_connection_established || local_sender_channel_worker_interface.has_worker_teardown_request();
            if (check_connection_status) {
                check_worker_connections<MY_ETH_CHANNEL, ENABLE_RISC_CPU_DATA_CACHE>(
                    local_sender_channel_worker_interface,
                    channel_connection_established,
                    sender_channel_free_slots_stream_id);
            }
        }
    }
    return progress;
}

/*
 * A fast neighbour exchange only receiver channel step impl
 */
static_assert(
    !super_speedy_mode || NUM_TRANSACTION_IDS >= 2,
    "Ping-pong TRID requires at least 2 transaction IDs per receiver channel");

template <
    uint8_t receiver_channel,
    uint8_t producer_sender_channel_index,
    uint8_t to_receiver_pkts_sent_id,
    size_t RECEIVER_CREDIT_AMORTIZATION_FREQUENCY_LOCAL,
    typename WriteTridTracker,
    typename ReceiverChannelBufferT,
    typename ReceiverChannelPointersT,
    typename LocalRelayInterfaceT,
    typename LocalTelemetryT,
    size_t VC_ID>
FORCE_INLINE bool run_receiver_channel_step_speedy(
    ReceiverChannelBufferT& local_receiver_channel,
    LocalRelayInterfaceT& local_relay_interface,
    ReceiverChannelPointersT& receiver_channel_pointers,
    WriteTridTracker& receiver_channel_trid_tracker,
    std::array<uint8_t, num_eth_ports>& port_direction_table,
    ReceiverChannelResponseCreditSender& receiver_channel_response_credit_sender,
    const tt::tt_fabric::routing_l1_info_t& routing_table,
    LocalTelemetryT& local_fabric_telemetry,
    SpeedyReceiverState<VC_ID>& receiver_state) {
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
            packet_header,
            packed.payload_size_bytes,
            packed.noc_send_type,
            receiver_state.current_write_trid,
            receiver_channel);

        did_something = true;
        progress = true;
#if defined(ARCH_BLACKHOLE)
        fabric_dbg_inc_rx_pkt_count();  // [RX-COUNT] one packet received off eth + delivered locally
#endif
        if constexpr (FABRIC_TELEMETRY_BANDWIDTH) {
            update_bw_counters(packet_header, local_fabric_telemetry);
        }
        channel_trimming_usage_recorder.set_receiver_channel_data_forwarded(receiver_channel);

        wr_sent_counter.increment();
        increment_local_update_ptr_val<to_receiver_pkts_sent_id>(-1);
        receiver_state.unacked_sends++;
    }

    // --- Ping-pong TRID flush ---
    // All packets in a batch share a single TRID (receiver_state.current_write_trid). When the batch
    // threshold is hit, we flip to the other TRID and check the previous batch's single
    // TRID for completion — replacing the per-slot loop with a single register read.
    //
    // The pending TRID is checked eagerly (every call) to minimize credit return latency.
    // Only the batch flip requires reaching the threshold.
    // when we pass the threshold of unacked messages,
    if ((receiver_state.unacked_sends >= RECEIVER_CREDIT_AMORTIZATION_FREQUENCY_LOCAL) &&
        !receiver_state.has_pending_flush) {
        receiver_state.pending_flush_trid = receiver_state.current_write_trid;
        receiver_state.pending_flush_batch_count = receiver_state.unacked_sends;
        receiver_state.current_write_trid = 1 - receiver_state.current_write_trid;
        receiver_state.has_pending_flush = true;
    }
    if (receiver_state.has_pending_flush) {
        bool flushed = ncrisc_noc_nonposted_write_with_transaction_id_sent(
            tt::tt_fabric::edm_to_local_chip_noc, receiver_state.pending_flush_trid);

        if (flushed) {
            auto& completion_counter = receiver_channel_pointers.completion_counter;
            completion_counter.increment_n(receiver_state.pending_flush_batch_count);
            receiver_send_completion_ack<false /*CHECK_BUSY*/>(
                receiver_channel_response_credit_sender,
                producer_sender_channel_index,
                receiver_state.pending_flush_batch_count);

            receiver_state.unacked_sends -= receiver_state.pending_flush_batch_count;
            receiver_state.has_pending_flush = false;
        }
    }

    return progress;
}
