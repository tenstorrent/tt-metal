#!/usr/bin/env python3
"""mk_v3y.py SENDER_HPP: v3v6's sender_erisc -> v3y (per-round and rare work out of line; TX-queue and stream registers
through the provided API helpers). Every anchor must match exactly once."""
import sys
p = sys.argv[1]
API_ONLY = "--api-only" in sys.argv
s = open(p).read()


def sub(a, b, n=1, structure=False):
    global s
    if structure and API_ONLY:
        return
    c = s.count(a)
    assert c == n, (c, n, a[:100])
    s = s.replace(a, b)


if True:
    pass
# --- header comment -------------------------------------------------------------------------------------------------
sub("""// The pass makes no calls while it sends, takes completions and returns credits (as in the generic loop). Out of line
// and off the pass's path: worker connection open and close, and the static-connection wait at init.
""", """// The pass makes no calls while it sends, takes completions and returns credits (as in the generic loop). Out of line
// and off the pass's path: worker connection open and close, and the static-connection wait at init.
//
// run_main_loop is a function of its own and holds only per-pass work. What it does once per round or rarely is in small
// functions of its own (termination_requested, next_round, end_round, run_router_command): the termination and
// router-command tests, the heartbeat, the idle context switch with its handshake, and the pause handshake. So edits to
// that code do not reach the pass loop's code (as on ERISC 1, RESULTS §97).
""", structure=True)
sub("""// Grouped reads (see "Grouped reads"): the grouped free-slots read, and send_packet's pre-send and post-send groups,
// which are sender_channel_try_send's.
""", """// Grouped reads (see "Grouped reads"): the grouped free-slots read, and send_packet's pre-send and post-send groups,
// which are sender_channel_try_send's. Registers are accessed through the provided helpers: the TX queue through
// eth_txq_reg_read and eth_send_packet_bytes_unsafe, the stream credit registers through increment_local_update_ptr_val
// and remote_update_ptr_val, and the NoC through noc_inline_dw_write.
""")

# --- free-slots stream ids: compile-time per channel -------------------------------------------------------------------
sub("""// A VC's free-slots stream registers are contiguous, so a channel's update register is the VC's first plus an offset.
template <size_t VC_BEGIN>
constexpr uint32_t first_free_slots_stream =
    sender_channel_free_slots_stream_ids[fabric_position_for_compact_sender(VC_BEGIN)];
template <size_t VC_BEGIN>
FORCE_INLINE constexpr uint32_t free_slots_update_addr(uint32_t ch) {
    constexpr uint32_t first_stream = first_free_slots_stream<VC_BEGIN>;
    return STREAM_REG_ADDR(first_stream, STREAM_REMOTE_DEST_BUF_SPACE_AVAILABLE_UPDATE_REG_INDEX) +
           (ch - VC_BEGIN) * NOC_STREAM_REG_SPACE_SIZE;
}
template <size_t VC_BEGIN>
constexpr bool free_slots_streams_contiguous(uint32_t channels) {
    for (size_t ch = 0; ch < NUM_SENDER_CHANNELS; ch++) {
        if (((channels >> ch) & 1) == 0) {
            continue;
        }
        const uint32_t stream = sender_channel_free_slots_stream_ids[fabric_position_for_compact_sender(ch)];
        if (ch < VC_BEGIN || stream != first_free_slots_stream<VC_BEGIN> + (ch - VC_BEGIN)) {
            return false;
        }
    }
    return true;
}
""", """// Channel CH's free-slots stream register: the producer (worker or upstream router) adds each packet it writes, and this
// ERISC adds each slot it sends from.
template <size_t CH>
constexpr uint32_t free_slots_stream = sender_channel_free_slots_stream_ids[fabric_position_for_compact_sender(CH)];
""")

# --- far ring: the packets-received stream id instead of a raw register address --------------------------------------
sub("""    static constexpr uint32_t packets_sent_update_addr =
        STREAM_REG_ADDR(to_receiver_packets_sent_streams[RX], STREAM_REMOTE_DEST_BUF_SPACE_AVAILABLE_UPDATE_REG_INDEX);
""", """    static constexpr uint32_t packets_received_stream = to_receiver_packets_sent_streams[RX];  // on the far receiver
""")

# --- send_packet ------------------------------------------------------------------------------------------------------
sub("""// receiver's packets-received register on the same queue. `completions` gets the completions newly reported. A send
// attempt that yields sets `attempted`; one that sends counts in `sent` (the idle context switch test reads both).
template <size_t RX, typename TelemetryT>
FORCE_INLINE bool send_packet(
    uint32_t ch,
    uint32_t& slot,
    uint32_t first_slot,
    uint32_t last_slot,
    FarRing<RX>& far,
    uint32_t free_slots_update,
    uint32_t completion_counter,""", """// receiver's packets-received register on the same queue. `completions` gets the completions newly reported. A send
// attempt that yields sets `attempted`; one that sends counts in `sent` (the idle context switch test reads both).
// FREE_SLOTS_STREAM is the channel's free-slots stream register.
template <uint32_t FREE_SLOTS_STREAM, size_t RX, typename TelemetryT>
FORCE_INLINE bool send_packet(
    uint32_t ch,
    uint32_t& slot,
    uint32_t first_slot,
    uint32_t last_slot,
    FarRing<RX>& far,
    uint32_t completion_counter,""")
sub("""    auto* pkt_header = reinterpret_cast<volatile tt_l1_ptr PACKET_HEADER_TYPE*>(src_addr);
    constexpr uint32_t txq_base = ETH_TXQ0_REGS_START + sender_txq_id * ETH_TXQ_REGS_SIZE;
    volatile uint32_t* const txq_regs = reinterpret_cast<volatile uint32_t*>(txq_base);

    std::atomic_signal_fence(std::memory_order_seq_cst);
    const uint32_t raw_payload_size = pkt_header->get_payload_size_excluding_header();
    (void)txq_regs[ETH_TXQ_CMD >> 2];
    const uint32_t pre_txq_status = txq_regs[ETH_TXQ_STATUS >> 2];
    std::atomic_signal_fence(std::memory_order_seq_cst);
""", """    auto* pkt_header = reinterpret_cast<volatile tt_l1_ptr PACKET_HEADER_TYPE*>(src_addr);

    // The TX-queue status is read as eth_txq_is_busy reads it (a CMD read first, BH-55), but inside the group.
    std::atomic_signal_fence(std::memory_order_seq_cst);
    const uint32_t raw_payload_size = pkt_header->get_payload_size_excluding_header();
    (void)eth_txq_reg_read(sender_txq_id, ETH_TXQ_CMD);
    const uint32_t pre_txq_status = eth_txq_reg_read(sender_txq_id, ETH_TXQ_STATUS);
    std::atomic_signal_fence(std::memory_order_seq_cst);
""")
sub("""    std::atomic_signal_fence(std::memory_order_seq_cst);
    (void)txq_regs[ETH_TXQ_CMD >> 2];
    const uint32_t txq_status = txq_regs[ETH_TXQ_STATUS >> 2];
    const uint32_t completions_raw = *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(completion_counter);
    std::atomic_signal_fence(std::memory_order_seq_cst);

    far.free_slots--;
    *reinterpret_cast<volatile uint32_t*>(free_slots_update) = 1u << REMOTE_DEST_BUF_WORDS_FREE_INC;
    std::atomic_signal_fence(std::memory_order_seq_cst);

    if (static_cast<int32_t>(txq_status << (31 - ETH_TXQ_STATUS_CMD_ONGOING_BIT)) < 0) {
        wait_for_sender_txq_idle();
    }
    txq_regs[ETH_TXQ_DEST_ADDR >> 2] = FarRing<RX>::packets_sent_update_addr;
    txq_regs[ETH_TXQ_REMOTE_REG_DATA >> 2] = 1u << REMOTE_DEST_BUF_WORDS_FREE_INC;
    txq_regs[ETH_TXQ_CMD >> 2] = ETH_TXQ_CMD_START_REG;
    sent++;
""", """    std::atomic_signal_fence(std::memory_order_seq_cst);
    (void)eth_txq_reg_read(sender_txq_id, ETH_TXQ_CMD);
    const uint32_t txq_status = eth_txq_reg_read(sender_txq_id, ETH_TXQ_STATUS);
    const uint32_t completions_raw = *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(completion_counter);
    std::atomic_signal_fence(std::memory_order_seq_cst);

    far.free_slots--;
    increment_local_update_ptr_val<FREE_SLOTS_STREAM>(1);
    std::atomic_signal_fence(std::memory_order_seq_cst);

    if (static_cast<int32_t>(txq_status << (31 - ETH_TXQ_STATUS_CMD_ONGOING_BIT)) < 0) {
        wait_for_sender_txq_idle();
    }
    remote_update_ptr_val<FarRing<RX>::packets_received_stream, sender_txq_id>(1);
    sent++;
""")

# --- forwarding visits: no VC_BEGIN (each channel's stream id is its own) ------------------------------------------------
sub("""template <size_t CH, size_t RX, size_t VC_BEGIN, bool FIRST_LEVEL_ACK>
FORCE_INLINE void visit_forwarding_channel(""", """template <size_t CH, size_t RX, bool FIRST_LEVEL_ACK>
FORCE_INLINE void visit_forwarding_channel(""")
sub("""        sent_packet = send_packet<RX>(
            CH,
            slot,
            ChannelSlots<CH>::first,
            ChannelSlots<CH>::last,
            far,
            free_slots_update_addr<VC_BEGIN>(CH),
            completion_counter_addr(CH),""", """        sent_packet = send_packet<free_slots_stream<CH>, RX>(
            CH,
            slot,
            ChannelSlots<CH>::first,
            ChannelSlots<CH>::last,
            far,
            completion_counter_addr(CH),""")
sub("""template <size_t RX, size_t VC_BEGIN, uint32_t CHANNELS, size_t BASE, bool FIRST_LEVEL_ACK, bool SKIP_IF_ALL_IDLE>
FORCE_INLINE void visit_forwarding_channels(const uint32_t (&raw)[4], FarRing<RX>& far, ForwardingState& state) {
    static_assert(
        (CHANNELS >> BASE) < 16 && (CHANNELS & ((1u << BASE) - 1)) == 0,
        "a VC's forwarding channels must be in its group of four reads");
    static_assert(free_slots_streams_contiguous<VC_BEGIN>(CHANNELS));
""", """template <size_t RX, uint32_t CHANNELS, size_t BASE, bool FIRST_LEVEL_ACK, bool SKIP_IF_ALL_IDLE>
FORCE_INLINE void visit_forwarding_channels(const uint32_t (&raw)[4], FarRing<RX>& far, ForwardingState& state) {
    static_assert(
        (CHANNELS >> BASE) < 16 && (CHANNELS & ((1u << BASE) - 1)) == 0,
        "a VC's forwarding channels must be in its group of four reads");
""")
sub("""                visit_forwarding_channel<CH, RX, VC_BEGIN, FIRST_LEVEL_ACK>(has_unsent_packet, far, state);""",
    """                visit_forwarding_channel<CH, RX, FIRST_LEVEL_ACK>(has_unsent_packet, far, state);""")

# --- once-per-round and rare work, out of line --------------------------------------------------------------------------
sub("""// The sending ERISC's main loop: router commands, the outer loop (termination, telemetry, heartbeat, idle context
// switch) and the inner loop of passes over the sender channels.
template <typename EdmChannelWorkerIFs>
__attribute__((noinline)) void run_main_loop(""", """// The main loop's once-per-round and rare work, out of line (see run_main_loop).
// The termination test: the termination signal, or the launch mailbox's exit flag.
__attribute__((noinline)) bool termination_requested() {
    return got_immediate_termination_signal<ENABLE_RISC_CPU_DATA_CACHE>(
        reinterpret_cast<volatile tt::tt_fabric::TerminationSignal*>(termination_signal_addr));
}

// At the top of a round: run it unless termination is requested or, on the context-switch master, a router command
// other than RUN is pending.
__attribute__((noinline)) bool next_round(tt_l1_ptr RouterStateManager* state_manager_l1) {
    return continue_running_main_run_loop<ENABLE_RISC_CPU_DATA_CACHE>(
        reinterpret_cast<volatile tt::tt_fabric::TerminationSignal*>(termination_signal_addr), state_manager_l1);
}

// At the end of a round: the heartbeat, then the context switch. As the context-switch master this ERISC hands the core
// to the base firmware once more than SWITCH_INTERVAL rounds in a row passed without a send attempt
// (IDLE_CONTEXT_SWITCHING; otherwise every SWITCH_INTERVAL rounds); otherwise it follows the master's handshake.
__attribute__((noinline)) void end_round(
    uint16_t& fabric_heartbeat_counter, size_t& did_nothing_count, bool attempted_send) {
    auto termination_signal_ptr = reinterpret_cast<volatile tt::tt_fabric::TerminationSignal*>(termination_signal_addr);
    if ((++fabric_heartbeat_counter & 0x3F) == 0) {
        *reinterpret_cast<volatile uint32_t*>(FABRIC_KERNEL_HEARTBEAT_ADDR) = 0xDCBA0000 | fabric_heartbeat_counter;
    }
    if constexpr (enable_context_switch) {
        // shouldn't do noc counter sync since we are not incrementing them
        if constexpr (IDLE_CONTEXT_SWITCHING) {
            if (attempted_send) {
                did_nothing_count = 0;
            } else if (did_nothing_count++ > SWITCH_INTERVAL) {
                did_nothing_count = 0;
                run_coordinated_context_switch_to_base_firmware(termination_signal_ptr);
            }
        } else {
            if (did_nothing_count++ > SWITCH_INTERVAL) {
                did_nothing_count = 0;
                run_coordinated_context_switch_to_base_firmware(termination_signal_ptr);
            }
        }
    } else {
        run_routing_without_noc_sync_coordinated_as_non_master(termination_signal_ptr);
    }
}

// A router command other than RUN, as read by the caller: PAUSE runs the pause handshake as the context-switch master.
__attribute__((noinline)) void run_router_command(tt_l1_ptr RouterStateManager* state_manager_l1, RouterCommand command) {
    auto termination_signal_ptr = reinterpret_cast<volatile tt::tt_fabric::TerminationSignal*>(termination_signal_addr);
    switch (command) {
        case RouterCommand::PAUSE:
            coordinated_context_switch_start_as_master(termination_signal_ptr);
            execute_pause_command(state_manager_l1, termination_signal_ptr);
            coordinated_context_switch_finish_as_master(termination_signal_ptr);
            break;
        default: ASSERT(false); break;
    }
}

// The sending ERISC's main loop: router commands, rounds (each followed by end_round) and, in each round,
// iterations_between_ctx_switch_and_teardown_checks passes over the sender channels.
//
// A function of its own, which owns the channel state: its code depends only on this body, not on kernel_main's init
// and teardown. The work it does once per round or rarely is in the small functions above, so that this function holds
// only per-pass code; the worker connection check stays in every pass, with only its rare open and close out of line
// (worker_connection_event).
template <typename EdmChannelWorkerIFs>
__attribute__((noinline)) void run_main_loop(""", structure=True)

sub("""    auto termination_signal_ptr = reinterpret_cast<volatile tt::tt_fabric::TerminationSignal*>(termination_signal_addr);
    const auto& routing_table =
        *reinterpret_cast<const tt_l1_ptr tt::tt_fabric::routing_l1_info_t*>(ROUTING_TABLE_BASE);
    auto* state_manager_l1 = const_cast<tt_l1_ptr RouterStateManager*>(&routing_table.state_manager);
""", """    const auto& routing_table =
        *reinterpret_cast<const tt_l1_ptr tt::tt_fabric::routing_l1_info_t*>(ROUTING_TABLE_BASE);
    auto* state_manager_l1 = const_cast<tt_l1_ptr RouterStateManager*>(&routing_table.state_manager);
""", structure=True)
sub("""    size_t did_nothing_count = 0;
    uint16_t fabric_heartbeat_counter = 0;
    volatile uint32_t* fabric_heartbeat_ptr = reinterpret_cast<volatile uint32_t*>(FABRIC_KERNEL_HEARTBEAT_ADDR);
""", """    size_t did_nothing_count = 0;
    uint16_t fabric_heartbeat_counter = 0;
""", structure=True)
sub("""        // STABILITY_PROBE_PRELOOP
        while (continue_running_main_run_loop<ENABLE_RISC_CPU_DATA_CACHE>(termination_signal_ptr, state_manager_l1)) {
""", """        // STABILITY_PROBE_PRELOOP
        while (next_round(state_manager_l1)) {
""", structure=True)
sub("""                    if (far_has_room<VC0_RECEIVER_CHANNEL, 1u>(far_vc0, 0) && has_unsent_packet) {
                        sent_packet = send_packet<VC0_RECEIVER_CHANNEL>(
                            0,
                            worker_slot,
                            ChannelSlots<0>::first,
                            ChannelSlots<0>::last,
                            far_vc0,
                            free_slots_update_addr<0>(0),
                            completion_counter_addr(0),""", """                    if (far_has_room<VC0_RECEIVER_CHANNEL, 1u>(far_vc0, 0) && has_unsent_packet) {
                        sent_packet = send_packet<free_slots_stream<0>, VC0_RECEIVER_CHANNEL>(
                            0,
                            worker_slot,
                            ChannelSlots<0>::first,
                            ChannelSlots<0>::last,
                            far_vc0,
                            completion_counter_addr(0),""")
sub("""                    visit_forwarding_channels<
                        VC0_RECEIVER_CHANNEL,
                        0,
                        vc0_forwarding,
                        0,""", """                    visit_forwarding_channels<
                        VC0_RECEIVER_CHANNEL,
                        vc0_forwarding,
                        0,""")
sub("""                    visit_forwarding_channels<
                        vc1_far_ring,
                        vc1_begin,
                        vc1_forwarding,
                        vc1_begin,""", """                    visit_forwarding_channels<
                        vc1_far_ring,
                        vc1_forwarding,
                        vc1_begin,""")
sub("""            if ((++fabric_heartbeat_counter & 0x3F) == 0) {
                *fabric_heartbeat_ptr = 0xDCBA0000 | fabric_heartbeat_counter;
            }

            if constexpr (enable_context_switch) {
                // shouldn't do noc counter sync since we are not incrementing them
                if constexpr (IDLE_CONTEXT_SWITCHING) {
                    if (state.attempted || packets_sent() != round_packets_sent) {
                        did_nothing_count = 0;
                    } else {
                        if (did_nothing_count++ > SWITCH_INTERVAL) {
                            did_nothing_count = 0;
                            run_coordinated_context_switch_to_base_firmware(termination_signal_ptr);
                        }
                    }
                } else {
                    if (did_nothing_count++ > SWITCH_INTERVAL) {
                        did_nothing_count = 0;
                        run_coordinated_context_switch_to_base_firmware(termination_signal_ptr);
                    }
                }
            } else {
                run_routing_without_noc_sync_coordinated_as_non_master(termination_signal_ptr);
            }
""", """            // A round attempted a send if a send attempt yielded or any channel's sent count changed (read only by
            // the idle context switch).
            end_round(
                fabric_heartbeat_counter,
                did_nothing_count,
                !IDLE_CONTEXT_SWITCHING || state.attempted || packets_sent() != round_packets_sent);
""", structure=True)
sub("""    while (!got_immediate_termination_signal<ENABLE_RISC_CPU_DATA_CACHE>(termination_signal_ptr)) {
        switch (reinterpret_cast<volatile tt_l1_ptr RouterStateManager*>(state_manager_l1)->command) {
            case RouterCommand::RUN:
                state_manager_l1->state = RouterState::RUNNING;
                execute_main_loop();
                break;
            case RouterCommand::PAUSE:
                coordinated_context_switch_start_as_master(termination_signal_ptr);
                execute_pause_command(state_manager_l1, termination_signal_ptr);
                coordinated_context_switch_finish_as_master(termination_signal_ptr);
                break;
            default: ASSERT(false); break;
        }
    }
""", """    while (!termination_requested()) {
        const RouterCommand command = reinterpret_cast<volatile tt_l1_ptr RouterStateManager*>(state_manager_l1)->command;
        if (command == RouterCommand::RUN) {
            state_manager_l1->state = RouterState::RUNNING;
            execute_main_loop();
        } else {
            run_router_command(state_manager_l1, command);
        }
    }
""", structure=True)
open(p, "w").write(s)
print("ok")
