// ---------------------------------------------------------------------------------------------------------------------
// Visits
// ---------------------------------------------------------------------------------------------------------------------

// A worker channel's connection, checked every pass (by rule) from the semaphore the pass's read group read: a change
// (open or close request) is handled out of line. Returns whether it changed.
template <size_t CH, typename Interface>
FORCE_INLINE bool check_connection(SenderChannelState<CH>& state, Interface& interface, uint32_t semaphore) {
    if (semaphore != state.connection_state) {
        const uint32_t returned =
            SenderChannel<CH>::first_level_ack ? state.progress.first_level_acked : state.progress.completed;
        state.connection_state = worker_connection_event<CH>(interface, semaphore, state.connection_state, returned);
        return true;
    }
    return false;
}

// Slots back to channel CH's producer, `n` more for `returned` in all: to a worker, the absolute count (out of line);
// to an upstream router, +n on its free-slots register.
template <size_t CH, typename Connections>
FORCE_INLINE void return_slots(
    const SenderChannelState<CH>& state, Connections& connections, uint32_t n, uint32_t returned) {
    if constexpr (SenderChannel<CH>::worker) {
        const bool connected = state.connection_state != tt::tt_fabric::connection_interface::unused_connection_value;
        credit_worker<CH>(connections.template get<CH>(), returned, connected);
    } else {
        credit_upstream(state.upstream_free_slots_addr, n);
    }
}

// A worker channel, its state a loop local (visited every pass): send if it has a packet and the remote receiver channel
// has room, take completions (and first-level acks), return the slots to the worker. (Its connection is checked at the
// top of the pass: check_worker_connection.)
template <size_t CH, typename Connections>
FORCE_INLINE void visit_worker(
    SenderChannelState<CH>& state,
    Connections& connections,
    bool has_packet,
    RemoteReceiverChannelProgress& remote,
    bool& send_yielded) {
    SenderChannelProgress& progress = state.progress;
    uint32_t completions = 0;
    uint32_t sent_acks = 0;  // the ack count the send read (first-level-ack VCs)
    bool sent = false;
    if (has_packet && remote.free_slots >= SenderChannel<CH>::remote_free_slots_to_send) {
        sent = send<CH>(
            progress.next_slot_addr, remote, progress.sent, progress.completed, completions, sent_acks, send_yielded);
    }
    // The visit's counters, unless the send read them, as one read group: completions, and acks (first-level-ack VCs).
    // Completions before acks, so the acks seen are never fewer than the completions seen. Whether to read each is
    // known before any is read.
    const bool read_completions = !sent && progress.sent != progress.completed;
    const bool read_acks = SenderChannel<CH>::first_level_ack && !sent && progress.sent != progress.first_level_acked;
    uint32_t reported_completions = 0;
    uint32_t reported_acks = sent_acks;
    begin_read_group();
    if (read_completions) {
        reported_completions =
            *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(SenderChannel<CH>::completion_counter_addr);
    }
    if (read_acks) {
        reported_acks =
            *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(SenderChannel<CH>::first_level_ack_counter_addr);
    }
    end_read_group();
    if (read_completions) {
        completions = reported_completions - progress.completed;
    }
    if (completions != 0) {
        remote.free_slots += completions;
        progress.completed += completions;
        if constexpr (!SenderChannel<CH>::first_level_ack) {
            return_slots<CH>(state, connections, completions, progress.completed);
        }
    }
    if constexpr (SenderChannel<CH>::first_level_ack) {
        if ((sent || read_acks) && reported_acks != progress.first_level_acked) {
            const uint32_t acks = reported_acks - progress.first_level_acked;
            progress.first_level_acked = reported_acks;
            return_slots<CH>(state, connections, acks, progress.first_level_acked);
        }
    }
}

// A forwarding channel with work, its state in memory: send if it has a packet and the remote receiver channel has room,
// take completions (and acks), return the slots to the upstream router, and keep the channel's bit in
// `forwarding_in_flight` equal to "packets not yet completed (or acked)". The state's fields are taken into locals where
// first needed and written back where they change (the send's read groups would make the compiler store and reload
// memory state around them).
template <size_t CH, typename Connections>
FORCE_INLINE void visit_forwarding(
    SenderChannelState<CH>& state,
    Connections& connections,
    uint32_t& forwarding_in_flight,
    bool has_packet,
    RemoteReceiverChannelProgress& remote,
    bool& send_yielded) {
    uint32_t sent = state.progress.sent;
    uint32_t completed = state.progress.completed;
    uint32_t completions = 0;
    uint32_t sent_acks = 0;  // the ack count the send read (first-level-ack VCs)
    bool did_send = false;
    if (has_packet && remote.free_slots >= SenderChannel<CH>::remote_free_slots_to_send) {
        uint32_t next_slot_addr = state.progress.next_slot_addr;
        did_send = send<CH>(next_slot_addr, remote, sent, completed, completions, sent_acks, send_yielded);
        if (did_send) {
            state.progress.next_slot_addr = next_slot_addr;
            state.progress.sent = sent;
        }
    }
    if (!did_send && sent != completed) {
        completions =
            *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(SenderChannel<CH>::completion_counter_addr) - completed;
    }
    if (completions != 0) {
        remote.free_slots += completions;
        completed += completions;
        state.progress.completed = completed;
        if constexpr (!SenderChannel<CH>::first_level_ack) {
            return_slots<CH>(state, connections, completions, completed);
        }
    }
    bool busy = sent != completed;
    if constexpr (SenderChannel<CH>::first_level_ack) {
        if (sent != state.progress.first_level_acked) {
            const uint32_t reported_acks =
                did_send ? sent_acks
                         : *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                               SenderChannel<CH>::first_level_ack_counter_addr);
            const uint32_t acks = reported_acks - state.progress.first_level_acked;
            if (acks != 0) {
                state.progress.first_level_acked += acks;
                return_slots<CH>(state, connections, acks, state.progress.first_level_acked);
            }
        }
        busy = busy || sent != state.progress.first_level_acked;
    }
    forwarding_in_flight = busy ? (forwarding_in_flight | (1u << CH)) : (forwarding_in_flight & ~(1u << CH));
    keep_in_memory(state);
}

// ---------------------------------------------------------------------------------------------------------------------
// Where the channel states live, and the loop
// ---------------------------------------------------------------------------------------------------------------------

// Every serviced VC's send state (vcs.get<VC>()) and every forwarding channel's state (forwarding.get<CH>()), and where
// they live. The loop-local send state is copied into a local of run()'s loop, so the compiler can keep it in registers
// (loopvars.py --headroom checks); every forwarding channel's state stays in memory, kept there after each of its
// visits (keep_in_memory), so it does not take registers from the send state every pass uses (V4_SENDER_OWNERSHIP.md
// §8). Today's rule: every VC's send state is loop-local. A one-ERISC router keeps every state in memory and visits on
// it in place (router_single.hpp). The accessors are templates on the VC so that an absent VC's state is named only in
// template-dependent code.
struct SenderChannelStates {
    Each<sender_vcs, SenderVcState> vcs;
    Each<forwarding_channels, SenderChannelState> forwarding;

    // The loop-local send state in place: what run() copies at its start; a one-ERISC router visits on it in place.
    FORCE_INLINE auto& loop_local_progress() { return vcs; }

    // VC's send state for its visit: the loop's copy.
    template <size_t VC, typename LoopCopy>
    FORCE_INLINE SenderVcState<VC>& progress_for(LoopCopy& loop_copy) {
        return loop_copy.template get<VC>();
    }

    // After VC's visit: nothing to keep in memory (every VC's send state is loop-local).
    template <size_t VC>
    FORCE_INLINE void after_visit() {}
};

// The forwarding channels in MASK that have a packet (`holding`) or packets in flight (`forwarding_in_flight`), lowest
// channel first.
template <uint32_t MASK, typename Connections>
FORCE_INLINE void visit_forwarding_channels(
    SenderChannelStates& states,
    Connections& connections,
    uint32_t holding,
    uint32_t& forwarding_in_flight,
    RemoteReceiverChannelProgress& remote,
    bool& send_yielded) {
    if constexpr (MASK != 0) {
        constexpr size_t CH = lowest_bit(MASK);
        const bool has_packet = ((holding >> CH) & 1u) != 0;
        if (has_packet || ((forwarding_in_flight >> CH) & 1u) != 0) {
            visit_forwarding<CH>(
                states.forwarding.template get<CH>(), connections, forwarding_in_flight, has_packet, remote,
                send_yielded);
        }
        visit_forwarding_channels<without_lowest_bit(MASK)>(
            states, connections, holding, forwarding_in_flight, remote, send_yielded);
    }
}

// VC's visit, on its send state where SenderChannelStates places it: its free-slots registers (when not read with the
// pass's first read group), its worker channel, then its forwarding channels, skipped with one test when none holds a
// packet or has packets in flight. None where VC has no serviced sender channel.
template <size_t VC, typename LoopCopy, typename Connections>
FORCE_INLINE void visit_vc(
    LoopCopy& loop_copy,
    SenderChannelStates& states,
    Connections& connections,
    uint32_t holding_at_pass_start,
    uint32_t& forwarding_in_flight,
    bool& send_yielded) {
    if constexpr (SenderVc<VC>::sender_channels != 0) {
        uint32_t holding = holding_at_pass_start;
        if constexpr (!SenderVc<VC>::reads_free_slots_at_pass_start) {
            holding = channels_with_packets<SenderVc<VC>::sender_channels>();
        }
        SenderVcState<VC>& send_state = states.template progress_for<VC>(loop_copy);
        if constexpr (SenderVc<VC>::worker_channels != 0) {
            constexpr size_t CH = lowest_bit(SenderVc<VC>::worker_channels);
            visit_worker<CH>(send_state.worker, connections, ((holding >> CH) & 1u) != 0, send_state.remote, send_yielded);
        }
        if constexpr (SenderVc<VC>::forwarding_channels != 0) {
            if (((holding | forwarding_in_flight) & SenderVc<VC>::forwarding_channels) != 0) {
                visit_forwarding_channels<SenderVc<VC>::forwarding_channels>(
                    states, connections, holding, forwarding_in_flight, send_state.remote, send_yielded);
            }
        }
        states.template after_visit<VC>();
    }
}

// The producers' addresses of the forwarding channels, cached when each upstream router connected.
struct UpstreamCredits {
    std::array<uint64_t, popcount(forwarding_channels)> addr;  // channel CH's at forwarding_index<CH>()
};

// The forwarding channels in MASK: each one's upstream router's free-slots register address.
template <uint32_t MASK>
FORCE_INLINE void init_forwarding(SenderChannelStates& states, const UpstreamCredits& upstream) {
    if constexpr (MASK != 0) {
        constexpr size_t CH = lowest_bit(MASK);
        states.forwarding.template get<CH>().upstream_free_slots_addr = upstream.addr[forwarding_index<CH>()];
        init_forwarding<without_lowest_bit(MASK)>(states, upstream);
    }
}

// Packets sent so far on the forwarding channels in MASK.
template <uint32_t MASK>
FORCE_INLINE uint32_t forwarded(SenderChannelStates& states) {
    if constexpr (MASK != 0) {
        return states.forwarding.template get<lowest_bit(MASK)>().progress.sent +
               forwarded<without_lowest_bit(MASK)>(states);
    } else {
        return 0;
    }
}

// Packets sent so far on VC's worker channel (0 where it has none).
template <size_t VC, typename LoopCopy>
FORCE_INLINE uint32_t sent_by_worker(LoopCopy& loop_copy, SenderChannelStates& states) {
    if constexpr (SenderVc<VC>::worker_channels != 0) {
        return states.template progress_for<VC>(loop_copy).worker.progress.sent;
    } else {
        return 0;
    }
}

// Packets sent so far on every channel (a round attempted a send if this changed or a send yielded; read by the idle
// context switch only).
template <typename LoopCopy>
FORCE_INLINE uint32_t total_sent(LoopCopy& loop_copy, SenderChannelStates& states) {
    return forwarded<forwarding_channels>(states) + sent_by_worker<0>(loop_copy, states) +
           sent_by_worker<1>(loop_copy, states) + sent_by_worker<2>(loop_copy, states);
}

// Whether a sender channel holds a packet (telemetry, once per round; the pass's register groups).
FORCE_INLINE bool any_channel_holds_packet() {
    return (channels_with_packets<channels_read_at_pass_start>() |
            channels_with_packets<sender_channels & ~channels_read_at_pass_start>()) != 0;
}

// The worker channels' interfaces, the loop's own copies (connection events and the workers' credit addresses): passed
// to out-of-line calls, so in memory. After init nothing else uses them, and copies keep the loop from holding a
// pointer to the caller's for its whole run.
template <size_t CH>
using WorkerInterface = tt::tt_fabric::StaticSizedSenderChannelWorkerInterface<WORKER_CREDIT_NOC, SENDER_NUM_BUFFERS_ARRAY[CH]>;
template <uint32_t MASK, typename WorkerInterfaces>
FORCE_INLINE Each<MASK, WorkerInterface> worker_connections(WorkerInterfaces& interfaces) {
    if constexpr (MASK == 0) {
        return {};
    } else {
        return {interfaces.template get<lowest_bit(MASK)>(), worker_connections<without_lowest_bit(MASK)>(interfaces)};
    }
}

// Something else used the write command buffer (router commands may have run the base firmware; another worker
// channel had a connection event): program the credit command of the worker channel that presets it again while its
// worker is connected (a no-op unless it owns the preset command buffer).
template <typename LoopCopy, typename Connections>
FORCE_INLINE void restore_worker_credit_preset(
    LoopCopy& loop_copy, SenderChannelStates& states, Connections& connections) {
    if constexpr ((SenderVc<0>::worker_channels & 1u) != 0) {
        if (states.template progress_for<0>(loop_copy).worker.connection_state !=
            tt::tt_fabric::connection_interface::unused_connection_value) {
            preset_worker_credit(connections.template get<0>());
        }
    }
}

// VC's worker channel's connection (every pass, by rule), from the semaphores the pass's read group read
// (`connection_semaphores[CH]`). A worker channel that does not own the credit preset may use the write command buffer
// on a connection event (its library connect), so the preset is restored after one.
template <size_t VC, typename LoopCopy, typename Connections>
FORCE_INLINE void check_worker_connection(
    LoopCopy& loop_copy, SenderChannelStates& states, Connections& connections, const uint32_t* connection_semaphores) {
    if constexpr (SenderVc<VC>::worker_channels != 0) {
        constexpr size_t CH = lowest_bit(SenderVc<VC>::worker_channels);
        if (check_connection<CH>(
                states.template progress_for<VC>(loop_copy).worker,
                connections.template get<CH>(),
                connection_semaphores[CH])) {
            if constexpr (!WorkerCredit<CH>::owns_write_buffer) {
                restore_worker_credit_preset(loop_copy, states, connections);
            }
        }
    }
}

// One pass over the sender channels: the first read group, each worker channel's connection, then one visit per VC.
// The body of run()'s loop; a single-ERISC router runs it as the sender half of each pass over its state in memory
// (router_single.hpp).
template <typename LoopCopy, typename Connections>
FORCE_INLINE void pass(
    LoopCopy& loop_copy,
    SenderChannelStates& states,
    Connections& connections,
    uint32_t& forwarding_in_flight,
    bool& send_yielded) {
    uint32_t connection_semaphores[worker_channels == 0 ? 1 : 32 - __builtin_clz(worker_channels)];
    const uint32_t holding =
        channels_with_packets<channels_read_at_pass_start, worker_channels>(connection_semaphores);
    check_worker_connection<0>(loop_copy, states, connections, connection_semaphores);
    check_worker_connection<1>(loop_copy, states, connections, connection_semaphores);
    check_worker_connection<2>(loop_copy, states, connections, connection_semaphores);
    visit_vc<0>(loop_copy, states, connections, holding, forwarding_in_flight, send_yielded);
    visit_vc<1>(loop_copy, states, connections, holding, forwarding_in_flight, send_yielded);
    visit_vc<2>(loop_copy, states, connections, holding, forwarding_in_flight, send_yielded);
}

// The sender loop: rounds of passes over the sender channels, with router commands between runs of rounds. A function
// of its own: its code depends only on this body, and once-per-round work is out of line (router_round.hpp).
template <typename WorkerInterfaces>
__attribute__((noinline)) void run(WorkerInterfaces& interfaces, const UpstreamCredits& upstream) {
    auto connections = worker_connections<worker_channels>(interfaces);
    SenderChannelStates states;
    init_forwarding<forwarding_channels>(states, upstream);
    auto loop_copy = states.loop_local_progress();  // carried through the loop (SenderChannelStates)
    uint32_t forwarding_in_flight = 0;  // the forwarding channels with packets not yet completed (or acked)

    RoundCounters round;
    while (!terminating()) {
        if (!start_running()) {
            continue;
        }
        restore_worker_credit_preset(loop_copy, states, connections);
        while (next_round()) {
            round.send_yielded = false;
            round.sent_at_round_start = total_sent(loop_copy, states);
            if constexpr (telemetry::any_option) {
                telemetry::round_start();
            }
            for (size_t n = 0; n < iterations_between_ctx_switch_and_teardown_checks; n++) {
                pass(loop_copy, states, connections, forwarding_in_flight, round.send_yielded);
            }
            const bool firmware_ran = end_round(
                round,
                !IDLE_CONTEXT_SWITCHING || round.send_yielded ||
                    total_sent(loop_copy, states) != round.sent_at_round_start);
            if constexpr (telemetry::any_option) {
                telemetry::round_end<true, false>([] { return any_channel_holds_packet(); }, [] { return false; });
            }
            if (firmware_ran) {
                restore_worker_credit_preset(loop_copy, states, connections);
            }
        }
    }
}

}  // namespace fabric_router::sender
