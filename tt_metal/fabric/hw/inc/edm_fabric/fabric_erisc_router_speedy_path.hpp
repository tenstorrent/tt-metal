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
    // --- Code profiling timers ---
    NamedProfiler<
        CodeProfilingTimerType::SPEEDY_SENDER_FULL,
        code_profiling_enabled_timers_bitfield,
        code_profiling_buffer_base_addr>
        sender_full_timer;
    NamedProfiler<
        CodeProfilingTimerType::SPEEDY_SENDER_SEND_DATA,
        code_profiling_enabled_timers_bitfield,
        code_profiling_buffer_base_addr>
        sender_send_data_timer;
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
    // Sub-timers for SEND_DATA breakdown
    NamedProfiler<
        CodeProfilingTimerType::SPEEDY_SENDER_SEND_ETH,
        code_profiling_enabled_timers_bitfield,
        code_profiling_buffer_base_addr>
        sender_send_eth_timer;
    NamedProfiler<
        CodeProfilingTimerType::SPEEDY_SENDER_SEND_ADV,
        code_profiling_enabled_timers_bitfield,
        code_profiling_buffer_base_addr>
        sender_send_adv_timer;
    NamedProfiler<
        CodeProfilingTimerType::SPEEDY_SENDER_SEND_NOTIFY,
        code_profiling_enabled_timers_bitfield,
        code_profiling_buffer_base_addr>
        sender_send_notify_timer;
    // Spin iteration counters
    SpinCounter<
        CodeProfilingTimerType::SPEEDY_SENDER_ETH_TXQ_SPIN_1,
        code_profiling_enabled_timers_bitfield,
        code_profiling_buffer_base_addr>
        eth_txq_spin_1;
    SpinCounter<
        CodeProfilingTimerType::SPEEDY_SENDER_ETH_TXQ_SPIN_2,
        code_profiling_enabled_timers_bitfield,
        code_profiling_buffer_base_addr>
        eth_txq_spin_2;
    SpinCounter<
        CodeProfilingTimerType::SPEEDY_SENDER_NOC_FLUSH_SPIN,
        code_profiling_enabled_timers_bitfield,
        code_profiling_buffer_base_addr>
        noc_flush_spin;
    SpinCounter<
        CodeProfilingTimerType::SPEEDY_SENDER_NOC_CMD_BUF_SPIN,
        code_profiling_enabled_timers_bitfield,
        code_profiling_buffer_base_addr>
        noc_cmd_buf_spin;

    bool progress = false;

    bool capture_full_timer = true;
    sender_full_timer.open();

    // --- Prefetch packet header payload_size and ETH TXQ status via inline asm ---
    // Issue loads early so their multi-cycle latencies (8c L1, 7c ETH) are hidden
    // behind the flow control stream register reads and branch computation.
    uint32_t src_addr = local_sender_channel.get_cached_next_buffer_slot_addr();
    uint32_t prefetched_payload_size_raw;
    asm volatile("lhu %0, 40(%1)" : "=r"(prefetched_payload_size_raw) : "r"(src_addr));

    // Prefetch ETH TXQ busy status: CMD dummy read (BH-55 workaround) + STATUS read.
    // By the time we reach the spin-wait ~20 instructions later, the value is ready.
    constexpr uint32_t txq_base = ETH_TXQ0_REGS_START + (sender_txq_id * ETH_TXQ_REGS_SIZE);
    uint32_t prefetched_txq_cmd_dummy, prefetched_txq_status;
    asm volatile(
        "lw %0, %2(%3)\n\t"
        "lw %1, %4(%3)"
        : "=r"(prefetched_txq_cmd_dummy), "=r"(prefetched_txq_status)
        : "i"(ETH_TXQ_CMD), "r"(txq_base), "i"(ETH_TXQ_STATUS));

    // --- Send packet if possible ---
    bool receiver_has_space_for_packet = outbound_to_receiver_channel_pointers.has_space_for_packet();
    uint32_t free_slots = get_ptr_val(sender_channel_free_slots_stream_id);
    bool has_unsent_packet = free_slots != WorkerInterfaceT::num_buffers;
    bool can_send = receiver_has_space_for_packet && has_unsent_packet;

    if constexpr (!ETH_TXQ_SPIN_WAIT_SEND_NEXT_DATA) {
        can_send = can_send && !internal_::eth_txq_is_busy(sender_txq_id);
    }

    sender_send_data_timer.set_should_dump(can_send);
    sender_send_data_timer.open();
    capture_full_timer = can_send && capture_full_timer;
    if (can_send) {
        progress = true;

        // --- Inlined send_next_data with sub-timers and spin counters ---
        auto& remote_receiver_num_free_slots = outbound_to_receiver_channel_pointers.num_free_slots;
        // payload_size_bytes = raw payload + sizeof(packet header) = prefetched + 48
        const size_t payload_size_bytes = static_cast<size_t>(prefetched_payload_size_raw) + sizeof(PACKET_HEADER_TYPE);
        const auto dest_addr = outbound_to_receiver_channel_pointers.remote_receiver_channel_address_ptr;

        // SUB-TIMER: ETH_SEND (L1 reads + spin-wait + eth_send)
        sender_send_eth_timer.set_should_dump(true);
        sender_send_eth_timer.open();

        channel_trimming_usage_recorder.set_sender_channel_used(sender_channel_index);
        channel_trimming_usage_recorder.update_sender_channel_packet_size(
            sender_channel_index, static_cast<uint16_t>(prefetched_payload_size_raw));

        if constexpr (ETH_TXQ_SPIN_WAIT_SEND_NEXT_DATA) {
            // Use prefetched TXQ status for first check — the reads were issued ~20 instructions
            // ago so the 7c ETH latency is fully hidden. Only enter the spin-wait if busy.
            if ((prefetched_txq_status >> ETH_TXQ_STATUS_CMD_ONGOING_BIT) & 0x1) {
                while (internal_::eth_txq_is_busy(sender_txq_id)) {
                    eth_txq_spin_1.increment();
                }
            }
            eth_txq_spin_1.flush();
        }
        internal_::eth_send_packet_bytes_unsafe(sender_txq_id, src_addr, dest_addr, payload_size_bytes);

        // Prefetch TXQ status for second spin-wait. Issue right after eth_send so the
        // 7c ETH latency is hidden behind the ~30 instructions of bookkeeping below.
        uint32_t prefetched_txq_status_2;
        asm volatile(
            "lw %0, %1(%2)\n\t"
            "lw %0, %3(%2)"
            : "=r"(prefetched_txq_status_2)
            : "i"(ETH_TXQ_CMD), "r"(txq_base), "i"(ETH_TXQ_STATUS));

        sender_send_eth_timer.close();

        // SUB-TIMER: ADVANCE (pointer bookkeeping)
        sender_send_adv_timer.set_should_dump(true);
        sender_send_adv_timer.open();

        local_sender_channel_worker_interface.template update_write_counter_for_send<false /*SKIP_LIVENESS*/>();
        outbound_to_receiver_channel_pointers.advance_remote_receiver_buffer_pointer();
        local_sender_channel.advance_to_next_cached_buffer_slot_addr();
        remote_receiver_num_free_slots--;
        record_packet_send(perf_telemetry_recorder, sender_channel_index, payload_size_bytes);

        sender_send_adv_timer.close();

        // SUB-TIMER: NOTIFY (spin-wait + remote reg update)
        sender_send_notify_timer.set_should_dump(true);
        sender_send_notify_timer.open();

        // Use prefetched TXQ status — if the ETH transfer completed during bookkeeping,
        // this skips the spin-wait entirely. If still busy, fall into the normal loop.
        if ((prefetched_txq_status_2 >> ETH_TXQ_STATUS_CMD_ONGOING_BIT) & 0x1) {
            while (internal_::eth_txq_is_busy(sender_txq_id)) {
                eth_txq_spin_2.increment();
            }
        }
        eth_txq_spin_2.flush();
        remote_update_ptr_val<to_receiver_pkts_sent_id, sender_txq_id>(1U);

        sender_send_notify_timer.close();

        sender_amort_counter++;
        if constexpr (FABRIC_TELEMETRY_BANDWIDTH) {
            auto* pkt_header_v = reinterpret_cast<volatile tt_l1_ptr PACKET_HEADER_TYPE*>(src_addr);
            update_bw_counters(pkt_header_v, local_fabric_telemetry);
        }
        increment_local_update_ptr_val(sender_channel_free_slots_stream_id, 1);
    }
    sender_send_data_timer.close();

    bool check_completions = sender_amort_counter > SENDER_CREDIT_AMORTIZATION_FREQUENCY;
    sender_check_completions_timer.set_should_dump(check_completions);
    sender_check_completions_timer.open();
    capture_full_timer = check_completions && capture_full_timer;
    if (check_completions) {
        // --- Always check for new completions from receiver (cheap read) ---
        int32_t completions = sender_channel_from_receiver_credits
                                  .template get_num_unprocessed_completions_from_receiver<ENABLE_RISC_CPU_DATA_CACHE>();
        if (completions) {
            outbound_to_receiver_channel_pointers.num_free_slots += completions;
            sender_channel_from_receiver_credits.increment_num_processed_completions(completions);

            completion_count += completions;
        }
    }
    sender_check_completions_timer.close();

    // --- Amortized: only send credits to upstream workers every N completions/acks ---
    bool send_credits = completion_count >= SENDER_CREDIT_AMORTIZATION_FREQUENCY;
    sender_credits_upstream_timer.set_should_dump(send_credits);
    sender_credits_upstream_timer.open();
    capture_full_timer = send_credits && capture_full_timer;
    if (send_credits) {
        // Pre-spin on NOC flush and CMD_BUF with counters, pulled out of noc_fast_spoof_write_dw_inline
        // to avoid modifying the shared NOC header. The function's internal spin-waits will be no-ops
        // since we've already ensured the conditions are met.
        send_credits_to_upstream_workers<false /*deadlock_avoidance*/, false /*SKIP_LIVENESS*/>(
            local_sender_channel_worker_interface, completion_count, channel_connection_established);
        sender_amort_counter -= completion_count;
        completion_count = 0;
    }
    sender_credits_upstream_timer.close();

    sender_full_timer.set_should_dump(capture_full_timer);
    sender_full_timer.close();

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

// Ping-pong TRID state for amortized flush.
// Instead of checking each buffer slot's TRID individually (~3.7 iterations × 2 register reads),
// all packets in a batch share a single TRID. When the batch threshold is hit, we flip to the
// other TRID. Flush checks only the previous batch's single TRID (1 check vs ~7.4 register reads).
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
    // --- Code profiling timers ---
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
    // Sub-timers for RECEIVER_FORWARD breakdown
    NamedProfiler<
        CodeProfilingTimerType::SPEEDY_RECEIVER_FWD_HDR,
        code_profiling_enabled_timers_bitfield,
        code_profiling_buffer_base_addr>
        receiver_fwd_hdr_timer;
    NamedProfiler<
        CodeProfilingTimerType::SPEEDY_RECEIVER_FWD_NOC,
        code_profiling_enabled_timers_bitfield,
        code_profiling_buffer_base_addr>
        receiver_fwd_noc_timer;
    NamedProfiler<
        CodeProfilingTimerType::SPEEDY_RECEIVER_FWD_BOOK,
        code_profiling_enabled_timers_bitfield,
        code_profiling_buffer_base_addr>
        receiver_fwd_book_timer;
    // Sub-timers for RECEIVER_FLUSH breakdown
    NamedProfiler<
        CodeProfilingTimerType::SPEEDY_RECEIVER_FLUSH_TRID,
        code_profiling_enabled_timers_bitfield,
        code_profiling_buffer_base_addr>
        receiver_flush_trid_timer;
    NamedProfiler<
        CodeProfilingTimerType::SPEEDY_RECEIVER_FLUSH_SEND,
        code_profiling_enabled_timers_bitfield,
        code_profiling_buffer_base_addr>
        receiver_flush_send_timer;
    // Spin iteration counters
    SpinCounter<
        CodeProfilingTimerType::SPEEDY_RECEIVER_NOC_CMD_BUF_SPIN,
        code_profiling_enabled_timers_bitfield,
        code_profiling_buffer_base_addr>
        noc_cmd_buf_spin;
    SpinCounter<
        CodeProfilingTimerType::SPEEDY_RECEIVER_FLUSH_ETH_TXQ_SPIN,
        code_profiling_enabled_timers_bitfield,
        code_profiling_buffer_base_addr>
        flush_eth_txq_spin;

    bool progress = false;
    auto& wr_sent_counter = receiver_channel_pointers.wr_sent_counter;
    uint8_t src_ch_id = 0;  // receiver_channel_pointers.get_src_chan_id();

    bool capture_full_timer = true;
    receiver_full_timer.open();

    // Inner loop: process up to RECEIVER_CREDIT_AMORTIZATION_FREQUENCY packets
    // for (uint32_t pkt = 0; pkt < RECEIVER_CREDIT_AMORTIZATION_FREQUENCY; pkt++) {
    receiver_forward_timer.open();
    auto pkts_received = get_ptr_val<to_receiver_pkts_sent_id>();

    // --- ACK phase (if first-level ack enabled) ---
    bool unwritten_packets = pkts_received != 0;

    if (unwritten_packets) {
        receiver_forward_timer.set_should_dump(true);

        // SUB-TIMER: HDR (cache invalidate + header load + packed load)
        receiver_fwd_hdr_timer.set_should_dump(true);
        receiver_fwd_hdr_timer.open();

        router_invalidate_l1_cache<ENABLE_RISC_CPU_DATA_CACHE>();
        auto receiver_buffer_index = wr_sent_counter.get_buffer_index();
        tt_l1_ptr PACKET_HEADER_TYPE* packet_header = const_cast<PACKET_HEADER_TYPE*>(
            local_receiver_channel.template get_packet_header<PACKET_HEADER_TYPE>(receiver_buffer_index));

        // Single 4B aligned load at offset 40 to get payload_size_bytes + noc_send_type
        // instead of two separate uncached L1 reads
        auto packed = PACKET_HEADER_TYPE::PackedPayloadAndSendType::load(packet_header);

        receiver_fwd_hdr_timer.close();

        did_something = true;
        progress = true;
        if constexpr (FABRIC_TELEMETRY_BANDWIDTH) {
            update_bw_counters(packet_header, local_fabric_telemetry);
        }
        channel_trimming_usage_recorder.set_receiver_channel_data_forwarded(receiver_channel);

        // SUB-TIMER: NOC (execute_chip_unicast_to_local_chip_impl)
        receiver_fwd_noc_timer.set_should_dump(true);
        receiver_fwd_noc_timer.open();

        execute_chip_unicast_to_local_chip_impl(
            packet_header, packed.payload_size_bytes, packed.noc_send_type, current_write_trid, receiver_channel);

        receiver_fwd_noc_timer.close();

        // SUB-TIMER: BOOK (counter increment + decrement pkts)
        receiver_fwd_book_timer.set_should_dump(true);
        receiver_fwd_book_timer.open();

        wr_sent_counter.increment();
        if constexpr (!enable_first_level_ack) {
            increment_local_update_ptr_val<to_receiver_pkts_sent_id>(-1);
        }
        unacked_sends++;

        receiver_fwd_book_timer.close();
    } else {
        capture_full_timer = false;
    }

    receiver_forward_timer.close();

    // --- Ping-pong TRID flush ---
    // All packets in a batch share a single TRID (current_write_trid). When the batch
    // threshold is hit, we flip to the other TRID and check the previous batch's single
    // TRID for completion — replacing the per-slot loop with a single register read.
    //
    // The pending TRID is checked eagerly (every call) to minimize credit return latency.
    // Only the batch flip requires reaching the threshold.
    bool did_flush = false;
    receiver_flush_timer.open();

    // Step 1: Eagerly check pending batch's TRID (cheap: single register read)
    if (has_pending_flush) {
        receiver_flush_trid_timer.set_should_dump(true);
        receiver_flush_trid_timer.open();

        bool flushed = ncrisc_noc_nonposted_write_with_transaction_id_sent(
            tt::tt_fabric::edm_to_local_chip_noc, pending_flush_trid);
        if constexpr (!tt::tt_fabric::local_chip_noc_equals_downstream_noc) {
            flushed = flushed &&
                      ncrisc_noc_nonposted_write_with_transaction_id_sent(edm_to_downstream_noc, pending_flush_trid);
        }

        receiver_flush_trid_timer.close();

        if (flushed) {
            // Credit back entire previous batch
            receiver_flush_send_timer.set_should_dump(true);
            receiver_flush_send_timer.open();

            auto& completion_counter = receiver_channel_pointers.completion_counter;
            completion_counter.increment_n(pending_flush_batch_count);
            receiver_send_completion_ack<false /*CHECK_BUSY*/>(
                receiver_channel_response_credit_sender, src_ch_id, pending_flush_batch_count);

            receiver_flush_send_timer.close();

            unacked_sends -= pending_flush_batch_count;
            has_pending_flush = false;
            did_flush = true;
        }
    }

    // Step 2: Flip batch when threshold hit and no pending flush
    if (!has_pending_flush && unacked_sends >= RECEIVER_CREDIT_AMORTIZATION_FREQUENCY) {
        pending_flush_trid = current_write_trid;
        pending_flush_batch_count = unacked_sends;
        current_write_trid = 1 - current_write_trid;
        has_pending_flush = true;
    }

    receiver_flush_timer.set_should_dump(did_flush);
    if (!did_flush) {
        capture_full_timer = false;
    }
    receiver_flush_timer.close();

    receiver_full_timer.close();

    return progress;
}
