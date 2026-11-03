"""stage_x6.py: v4txo6 = v4txo5b + C (V4_SENDER_OWNERSHIP.md §4 X6): telemetry's sent packet count from its owner, the
channels' `sent` (the round's count, from total_sent, folded into the 64-bit total once per round); the per-packet hook
(note_sent) adds only the words, into 32-bit round counts folded once per round. Identical with telemetry and capture
off; telemetry-on images change (functional check)."""
import os, shutil, sys
V = "/home/aagarwal/tt-metal/fabric_exp/variants"
src, dst = f"{V}/v4txo5b", f"{V}/v4txo6"
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

edit(f"{R}/router_telemetry.hpp", [
    ("""// The running counts of this ERISC, and where the current round started.
struct Counters {
    uint64_t tx_packets;
    uint64_t tx_words;      // 4 B words (dynamic bandwidth)
    uint64_t tx_eth_words;  // 16 B words, truncated per packet (perf recorder)""",
     """// The running counts of this ERISC, and where the current round started. The sender's totals are folded in once per
// round: its packets from its channels' sent counts (round_end's tx_packets), its words from this round's counts.
struct Counters {
    uint64_t tx_packets;
    uint64_t tx_words;      // 4 B words (dynamic bandwidth)
    uint64_t tx_eth_words;  // 16 B words, truncated per packet (perf recorder)
    uint32_t round_tx_words;      // this round's (a round sends far fewer than 2^32 words)
    uint32_t round_tx_eth_words;  // this round's"""),
    ("""    uint64_t round_tx_packets;
    uint64_t round_tx_eth_words;
    uint64_t round_rx_packets;""", """    uint64_t round_rx_packets;"""),
    ("""// A packet sent over the link, `bytes` with its header.
FORCE_INLINE void count_sent(uint32_t bytes) {
    if constexpr (counts_sent) {
        counters.tx_packets++;
        if constexpr (perf_recorder) {
            counters.tx_eth_words += bytes_to_eth_words_truncated(bytes);
        }
        if constexpr (FABRIC_TELEMETRY_BANDWIDTH) {
            counters.tx_words += (bytes + 3) >> 2;
        }
    }
}""", """// A packet sent over the link, `bytes` with its header: its words, into this round's counts. (Its count comes from the
// channel's sent count, once per round: round_end.)
FORCE_INLINE void note_sent(uint32_t bytes) {
    if constexpr (counts_sent) {
        if constexpr (perf_recorder) {
            counters.round_tx_eth_words += bytes_to_eth_words_truncated(bytes);
        }
        if constexpr (FABRIC_TELEMETRY_BANDWIDTH) {
            counters.round_tx_words += (bytes + 3) >> 2;
        }
    }
}"""),
    ("""    counters.round_tx_packets = counters.tx_packets;
    counters.round_tx_eth_words = counters.tx_eth_words;
    counters.round_rx_packets = counters.rx_packets;""", """    counters.round_tx_words = 0;
    counters.round_tx_eth_words = 0;
    counters.round_rx_packets = counters.rx_packets;"""),
    ("""// After a round (its passes and end_round), for the sides this loop runs (SENDER, RECEIVER). any_held(): whether a
// sender channel holds a packet; rx_register_empty(): whether VC0's packets-received register reads 0; each is read only
// when needed.
template <bool SENDER, bool RECEIVER, typename AnyHeld, typename RxRegisterEmpty>
__attribute__((noinline)) void round_end_impl(AnyHeld any_held, RxRegisterEmpty rx_register_empty) {
    Counters& c = counters;
    const bool tx_progress = c.tx_packets != c.round_tx_packets;""", """// After a round (its passes and end_round), for the sides this loop runs (SENDER, RECEIVER). any_held(): whether a
// sender channel holds a packet; rx_register_empty(): whether VC0's packets-received register reads 0; each is read only
// when needed. tx_packets: the packets the sender's channels sent this round (their sent counts).
template <bool SENDER, bool RECEIVER, typename AnyHeld, typename RxRegisterEmpty>
__attribute__((noinline)) void round_end_impl(AnyHeld any_held, RxRegisterEmpty rx_register_empty, uint32_t tx_packets) {
    Counters& c = counters;
    const bool tx_progress = tx_packets != 0;
    if constexpr (SENDER) {
        c.tx_packets += tx_packets;
        c.tx_words += c.round_tx_words;
        c.tx_eth_words += c.round_tx_eth_words;
    }"""),
    ("""            buffer->num_words_sent = buffer->num_words_sent + (c.tx_eth_words - c.round_tx_eth_words);
            buffer->num_packets_sent = buffer->num_packets_sent + (c.tx_packets - c.round_tx_packets);""",
     """            buffer->num_words_sent = buffer->num_words_sent + c.round_tx_eth_words;
            buffer->num_packets_sent = buffer->num_packets_sent + tx_packets;"""),
    ("""template <bool SENDER, bool RECEIVER, typename AnyHeld, typename RxRegisterEmpty>
FORCE_INLINE void round_end(AnyHeld any_held, RxRegisterEmpty rx_register_empty) {
    if constexpr (any_option) {
        round_end_impl<SENDER, RECEIVER>(any_held, rx_register_empty);
    }
}""", """template <bool SENDER, bool RECEIVER, typename AnyHeld, typename RxRegisterEmpty>
FORCE_INLINE void round_end(AnyHeld any_held, RxRegisterEmpty rx_register_empty, uint32_t tx_packets = 0) {
    if constexpr (any_option) {
        round_end_impl<SENDER, RECEIVER>(any_held, rx_register_empty, tx_packets);
    }
}"""),
])
edit(f"{R}/router_sender.hpp", [
    ("""    if constexpr (telemetry::counts_sent) {
        telemetry::count_sent(size);
    }""", """    if constexpr (telemetry::counts_sent) {
        telemetry::note_sent(size);
    }"""),
    ("""            const bool firmware_ran = end_round(
                round,
                !IDLE_CONTEXT_SWITCHING || round.send_yielded ||
                    total_sent(loop_copy, states) != round.sent_at_round_start);
            if constexpr (telemetry::any_option) {
                telemetry::round_end<true, false>([] { return any_channel_holds_packet(); }, [] { return false; });
            }""", """            const bool firmware_ran = end_round(
                round,
                !IDLE_CONTEXT_SWITCHING || round.send_yielded ||
                    total_sent(loop_copy, states) != round.sent_at_round_start);
            if constexpr (telemetry::any_option) {
                telemetry::round_end<true, false>(
                    [] { return any_channel_holds_packet(); },
                    [] { return false; },
                    total_sent(loop_copy, states) - round.sent_at_round_start);
            }"""),
])
edit(f"{R}/router_single.hpp", [
    ("""            const bool firmware_ran = end_round(
                round,
                !IDLE_CONTEXT_SWITCHING || round.send_yielded ||
                    sender::total_sent(s.states.loop_local_progress(), s.states) != round.sent_at_round_start);
            if constexpr (telemetry::any_option) {
                telemetry::round_end<true, true>(
                    [] { return sender::any_channel_holds_packet(); }, [] { return receiver::vc0_register_empty(); });
            }""", """            const bool firmware_ran = end_round(
                round,
                !IDLE_CONTEXT_SWITCHING || round.send_yielded ||
                    sender::total_sent(s.states.loop_local_progress(), s.states) != round.sent_at_round_start);
            if constexpr (telemetry::any_option) {
                telemetry::round_end<true, true>(
                    [] { return sender::any_channel_holds_packet(); },
                    [] { return receiver::vc0_register_empty(); },
                    sender::total_sent(s.states.loop_local_progress(), s.states) - round.sent_at_round_start);
            }"""),
])
print("made", dst)
