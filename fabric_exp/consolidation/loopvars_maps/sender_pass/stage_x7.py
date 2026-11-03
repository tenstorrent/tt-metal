"""stage_x7.py: v4txo7 = v4txo6 + X7 (V4_SENDER_OWNERSHIP.md §3 #5/#10): the sender's round work in two helpers,
before_round and after_round, used by run() and by the single-ERISC loop, which no longer repeats them. Identity
expected."""
import os, shutil, sys
V = "/home/aagarwal/tt-metal/fabric_exp/variants"
src, dst = f"{V}/v4txo6", f"{V}/v4txo7"
if os.path.exists(dst):
    sys.exit(f"{dst} exists")
shutil.copytree(src, dst, symlinks=True)
R = f"{dst}/tt_metal/fabric/hw/inc/edm_fabric/router"

def edit(path, pairs):
    s = open(path).read()
    for o, n in pairs:
        assert s.count(o) == 1, (path, o[:90])
        s = s.replace(o, n)
    open(path, "w").write(s)

edit(f"{R}/router_sender.hpp", [
    ("""// The sender loop: rounds of passes over the sender channels, with router commands between runs of rounds.""",
     """// The sender's work before a round's passes: nothing yielded yet, the packets sent so far (for the idle context switch
// and telemetry), telemetry's round start. (run() and the single-ERISC loop.)
template <typename LoopCopy>
FORCE_INLINE void before_round(RoundCounters& round, LoopCopy& loop_copy, SenderChannelStates& states) {
    round.send_yielded = false;
    round.sent_at_round_start = total_sent(loop_copy, states);
    if constexpr (telemetry::any_option) {
        telemetry::round_start();
    }
}

// The sender's work after a round's passes: end_round (router commands; the idle context switch when no send was
// attempted), telemetry's round end (RECEIVER: the loop runs the receiver too; rx_register_empty for its heartbeat), and
// the credit presets again where the base firmware ran. (run() and the single-ERISC loop.)
template <bool RECEIVER, typename LoopCopy, typename RxRegisterEmpty>
FORCE_INLINE void after_round(
    RoundCounters& round, LoopCopy& loop_copy, SenderChannelStates& states, RxRegisterEmpty rx_register_empty) {
    const bool firmware_ran = end_round(
        round,
        !IDLE_CONTEXT_SWITCHING || round.send_yielded || total_sent(loop_copy, states) != round.sent_at_round_start);
    if constexpr (telemetry::any_option) {
        telemetry::round_end<true, RECEIVER>(
            [] { return any_channel_holds_packet(); },
            rx_register_empty,
            total_sent(loop_copy, states) - round.sent_at_round_start);
    }
    if (firmware_ran) {
        restore_worker_credit_presets(loop_copy, states);
    }
}

// The sender loop: rounds of passes over the sender channels, with router commands between runs of rounds."""),
    ("""        while (next_round()) {
            round.send_yielded = false;
            round.sent_at_round_start = total_sent(loop_copy, states);
            if constexpr (telemetry::any_option) {
                telemetry::round_start();
            }
            for (size_t n = 0; n < iterations_between_ctx_switch_and_teardown_checks; n++) {
                pass(loop_copy, states, channels_in_flight, round.send_yielded);
            }
            const bool firmware_ran = end_round(
                round,
                !IDLE_CONTEXT_SWITCHING || round.send_yielded ||
                    total_sent(loop_copy, states) != round.sent_at_round_start);
            if constexpr (telemetry::any_option) {
                telemetry::round_end<true, false>(
                    [] { return any_channel_holds_packet(); },
                    [] { return false; },
                    total_sent(loop_copy, states) - round.sent_at_round_start);
            }
            if (firmware_ran) {
                restore_worker_credit_presets(loop_copy, states);
            }
        }""", """        while (next_round()) {
            before_round(round, loop_copy, states);
            for (size_t n = 0; n < iterations_between_ctx_switch_and_teardown_checks; n++) {
                pass(loop_copy, states, channels_in_flight, round.send_yielded);
            }
            after_round<false>(round, loop_copy, states, [] { return false; });
        }"""),
])
edit(f"{R}/router_single.hpp", [
    ("""        while (next_round()) {
            round.send_yielded = false;
            round.sent_at_round_start = sender::total_sent(s.states.loop_local_progress(), s.states);
            if constexpr (telemetry::any_option) {
                telemetry::round_start();
            }
            for (size_t n = 0; n < iterations_between_ctx_switch_and_teardown_checks; n++) {
                receiver_pass(r, downstreams, my_chip);
                sender_pass(s, round.send_yielded);
            }
            const bool firmware_ran = end_round(
                round,
                !IDLE_CONTEXT_SWITCHING || round.send_yielded ||
                    sender::total_sent(s.states.loop_local_progress(), s.states) != round.sent_at_round_start);
            if constexpr (telemetry::any_option) {
                telemetry::round_end<true, true>(
                    [] { return sender::any_channel_holds_packet(); },
                    [] { return receiver::vc0_register_empty(); },
                    sender::total_sent(s.states.loop_local_progress(), s.states) - round.sent_at_round_start);
            }
            if (firmware_ran) {
                sender::restore_worker_credit_presets(s.states.loop_local_progress(), s.states);
            }
        }""", """        while (next_round()) {
            sender::before_round(round, s.states.loop_local_progress(), s.states);
            for (size_t n = 0; n < iterations_between_ctx_switch_and_teardown_checks; n++) {
                receiver_pass(r, downstreams, my_chip);
                sender_pass(s, round.send_yielded);
            }
            sender::after_round<true>(
                round, s.states.loop_local_progress(), s.states, [] { return receiver::vc0_register_empty(); });
        }"""),
])
print("made", dst)
