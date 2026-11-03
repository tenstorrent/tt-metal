"""stage_x3.py: v4txo3 = v4txo2 + the placement rule P-B (V4_SENDER_OWNERSHIP.md §8, the user's pick): the loop-local VC
(the lowest VC this router sends on, derived) keeps its send state (remote receiver progress, worker channel state) in
run()'s loop copy; every later VC's send state stays in memory, visited only when its channels show work, its fields
taken where first needed and written back where they change. SenderChannelStates owns the rule."""
import os, re, shutil, sys
V = "/home/aagarwal/tt-metal/fabric_exp/variants"
src, dst = f"{V}/v4txo2", f"{V}/v4txo3"
if os.path.exists(dst):
    sys.exit(f"{dst} exists")
shutil.copytree(src, dst, symlinks=True)
R = f"{dst}/tt_metal/fabric/hw/inc/edm_fabric/router"
p = f"{R}/router_sender.hpp"
s = open(p).read()

def rep(o, n):
    global s
    assert s.count(o) == 1, o[:90]
    s = s.replace(o, n)

# 1. the rule
a = s.index("// Every serviced VC's send state (vcs.get<VC>())")
b = s.index("};\n", s.index("struct SenderChannelStates {")) + 3
s = s[:a] + """// Every serviced VC's send state (vcs.get<VC>()) and every forwarding channel's state (forwarding.get<CH>()), and where
// they live. The loop-local VC's send state (its remote receiver progress and its worker channel's state) is copied into
// a local of run()'s loop, so the compiler can keep it in registers (loopvars.py --headroom checks). Every other VC's
// send state and every forwarding channel's state stays in memory: such a channel is visited only when the pass's reads
// show it work (it holds a packet, or has packets in flight: channels_in_flight), its fields are taken into locals where
// first needed and written back where they change, and its state is kept in memory after the visit (keep_in_memory),
// so it does not take the loop-local VC's registers on its packet path (V4_SENDER_OWNERSHIP.md §8). A one-ERISC router
// keeps every state in memory and visits the loop-local VC's in place (router_single.hpp). The accessors are templates
// on the VC so that an absent VC's state is named only in template-dependent code.
struct SenderChannelStates {
    Each<sender_vcs, SenderVcState> vcs;
    Each<forwarding_channels, SenderChannelState> forwarding;

    // The VC whose send state run() carries as a loop local: the lowest VC this router sends on (channel trimming can
    // leave a router sending on VC2 and not VC0, V4_TRIM.md). 0 where it sends on none: then nothing is visited.
    static constexpr size_t loop_local_vc = sender_vcs != 0 ? lowest_bit(sender_vcs) : 0;

    // Whether VC's channels are visited every pass (the loop-local VC's), or only when they show work.
    template <size_t VC>
    static constexpr bool visited_every_pass = VC == loop_local_vc;

    // The loop-local VC's send state in place: what run() copies at its start; a one-ERISC router visits on it in place.
    // Without a sender VC (sender_vcs == 0), the states themselves: no visit uses it.
    template <size_t VC = loop_local_vc>
    FORCE_INLINE auto& loop_local_progress() {
        if constexpr (((sender_vcs >> VC) & 1u) != 0) {
            return vcs.template get<VC>();
        } else {
            return *this;
        }
    }

    // VC's send state for its visit: the loop's copy for the loop-local VC, the state in memory otherwise.
    template <size_t VC, typename LoopCopy>
    FORCE_INLINE SenderVcState<VC>& progress_for(LoopCopy& loop_copy) {
        if constexpr (VC == loop_local_vc) {
            return loop_copy;
        } else {
            return vcs.template get<VC>();
        }
    }

    // After VC's visit: its remote receiver progress put back (`remote`, taken as `taken` at the visit's start): on the
    // loop's copy, or in memory where it changed, and a later VC's state kept in memory.
    template <size_t VC>
    FORCE_INLINE void after_visit(
        SenderVcState<VC>& send_state,
        const RemoteReceiverChannelProgress& taken,
        const RemoteReceiverChannelProgress& remote) {
        if constexpr (VC == loop_local_vc) {
            send_state.remote = remote;
        } else {
            if (remote.free_slots != taken.free_slots) {
                send_state.remote.free_slots = remote.free_slots;
            }
            if (remote.next_slot_addr != taken.next_slot_addr) {
                send_state.remote.next_slot_addr = remote.next_slot_addr;
            }
            keep_in_memory(send_state);
        }
    }
};
""" + s[b:]

# 2. the visit of a channel kept in memory serves a later VC's worker channel too
rep("""// A forwarding channel with work, its state in memory: send if it has a packet and the remote receiver channel has room,
// take completions (and acks), return the slots to the upstream router, and keep the channel's bit in
// `forwarding_in_flight` equal to "packets not yet completed (or acked)". The state's fields are taken into locals where
// first needed and written back where they change (the send's read groups would make the compiler store and reload
// memory state around them).
template <size_t CH, typename Connections>
FORCE_INLINE void visit_forwarding(""", """// A channel with work whose state stays in memory (a forwarding channel, or a later VC's worker channel:
// SenderChannelStates): send if it has a packet and the remote receiver channel has room, take completions (and acks),
// return the slots to the producer, and keep the channel's bit in `channels_in_flight` equal to "packets not yet
// completed (or acked)". The state's fields are taken into locals where first needed and written back where they change
// (the send's read groups would make the compiler store and reload memory state around them).
template <size_t CH, typename Connections>
FORCE_INLINE void visit_channel_with_work(""")
s = s.replace("visit_forwarding<CH>(", "visit_channel_with_work<CH>(")
s = s.replace("forwarding_in_flight", "channels_in_flight")

# 3. the VC's visit: a later VC only with work, its remote receiver progress taken and put back
rep("""// VC's visit, on its send state where SenderChannelStates places it: its free-slots registers (when not read with the
// pass's first read group), its worker channel, then its forwarding channels, skipped with one test when none holds a
// packet or has packets in flight. None where VC has no serviced sender channel.""",
"""// VC's visit, on its send state where SenderChannelStates places it: its free-slots registers (when not read with the
// pass's first read group), its worker channel, then its forwarding channels, skipped with one test when none holds a
// packet or has packets in flight. A VC visited only with work is skipped with one test too. Its remote receiver
// progress is taken into a local where the visit starts and put back where it ends (after_visit). None where VC has no
// serviced sender channel.""")
rep("""        SenderVcState<VC>& send_state = states.template progress_for<VC>(loop_copy);
        if constexpr (SenderVc<VC>::worker_channels != 0) {
            constexpr size_t CH = lowest_bit(SenderVc<VC>::worker_channels);
            visit_worker<CH>(send_state.worker, connections, ((holding >> CH) & 1u) != 0, send_state.remote, send_yielded);
        }
        if constexpr (SenderVc<VC>::forwarding_channels != 0) {
            if (((holding | channels_in_flight) & SenderVc<VC>::forwarding_channels) != 0) {
                visit_forwarding_channels<SenderVc<VC>::forwarding_channels>(
                    states, connections, holding, channels_in_flight, send_state.remote, send_yielded);
            }
        }
        states.template after_visit<VC>();""",
"""        if constexpr (!SenderChannelStates::visited_every_pass<VC>) {
            if (((holding | channels_in_flight) & SenderVc<VC>::sender_channels) == 0) {
                return;
            }
        }
        SenderVcState<VC>& send_state = states.template progress_for<VC>(loop_copy);
        const RemoteReceiverChannelProgress taken = send_state.remote;
        RemoteReceiverChannelProgress remote = taken;
        if constexpr (SenderVc<VC>::worker_channels != 0) {
            constexpr size_t CH = lowest_bit(SenderVc<VC>::worker_channels);
            const bool has_packet = ((holding >> CH) & 1u) != 0;
            if constexpr (SenderChannelStates::visited_every_pass<VC>) {
                visit_worker<CH>(send_state.worker, connections, has_packet, remote, send_yielded);
            } else if (has_packet || ((channels_in_flight >> CH) & 1u) != 0) {
                visit_channel_with_work<CH>(
                    send_state.worker, connections, channels_in_flight, has_packet, remote, send_yielded);
            }
        }
        if constexpr (SenderVc<VC>::forwarding_channels != 0) {
            if (((holding | channels_in_flight) & SenderVc<VC>::forwarding_channels) != 0) {
                visit_forwarding_channels<SenderVc<VC>::forwarding_channels>(
                    states, connections, holding, channels_in_flight, remote, send_yielded);
            }
        }
        states.template after_visit<VC>(send_state, taken, remote);""")
rep("""    uint32_t channels_in_flight = 0;  // the forwarding channels with packets not yet completed (or acked)""",
    """    uint32_t channels_in_flight = 0;  // the channels kept in memory with packets not yet completed (or acked)""")
rep("""// A worker channel, its state a loop local (visited every pass): send if it has a packet and the remote receiver channel
// has room,""", """// The loop-local VC's worker channel, its state on the loop's copy (visited every pass): send if it has a packet and the
// remote receiver channel has room,""")
open(p, "w").write(s)

p = f"{R}/router_single.hpp"
t = open(p).read()
t = t.replace("forwarding_in_flight", "channels_in_flight")
open(p, "w").write(t)
# 4. the file header's placement paragraph
p = f"{R}/router_sender.hpp"
s = open(p).read()
a = """// State, by placement class: the worker channels' progress and connection state, each VC's remote receiver progress
// and which forwarding channels have packets in flight, loop locals; each forwarding channel's progress and its
// producer's address, in memory between its visits; the worker interfaces, in memory (they escape to out-of-line
// calls); ring bounds, registers and counters, compile-time. The block before pass() states where each lives and why."""
b = """// State, by placement class: the loop-local VC's send state (its remote receiver progress and its worker channel's
// progress and connection state) and which channels kept in memory have packets in flight, loop locals; every other
// VC's send state and each forwarding channel's state, in memory between its visits; the worker interfaces, in memory
// (they escape to out-of-line calls); slot bounds, registers and counters, compile-time. The block before pass()
// (SenderChannelStates) states the rule."""
if s.count(a) == 1:
    s = s.replace(a, b)
    open(p, "w").write(s)
print("made", dst)
