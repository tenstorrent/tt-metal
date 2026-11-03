"""stage_x3b.py: v4txo3b = v4txo3a + the loop-local VC visited on its send state in place (no take and put-back of its
remote receiver progress); a later VC's progress is still taken into a local and put back where it changed. On two
ERISCs the loop-local state is the loop's register copy either way; on one ERISC it is the state record in memory,
where the take and put-back added a load and a store per pass (ew4352-1e, stage A/B)."""
import os, shutil, sys
V = "/home/aagarwal/tt-metal/fabric_exp/variants"
src, dst = f"{V}/v4txo3a", f"{V}/v4txo3b"
if os.path.exists(dst):
    sys.exit(f"{dst} exists")
shutil.copytree(src, dst, symlinks=True)
p = f"{dst}/tt_metal/fabric/hw/inc/edm_fabric/router/router_sender.hpp"
s = open(p).read()
a = s.index("// VC's visit, on its send state where SenderChannelStates places it:")
b = s.index("\n}\n", a) + 3
new = '''// VC's channels against its remote receiver progress (`remote`): its worker channel (visited every pass where VC is the
// loop-local VC, otherwise when it holds a packet or has packets in flight), then its forwarding channels, skipped with
// one test when none holds a packet or has packets in flight.
template <size_t VC, typename Remote>
FORCE_INLINE void visit_channels(
    SenderChannelStates& states,
    SenderVcState<VC>& send_state,
    Remote& remote,
    uint32_t holding,
    uint32_t& channels_in_flight,
    bool& send_yielded) {
    if constexpr (SenderVc<VC>::worker_channels != 0) {
        constexpr size_t CH = lowest_bit(SenderVc<VC>::worker_channels);
        const bool has_packet = ((holding >> CH) & 1u) != 0;
        if constexpr (SenderChannelStates::visited_every_pass<VC>) {
            visit_worker<CH>(send_state.worker, has_packet, remote, send_yielded);
        } else if (has_packet || ((channels_in_flight >> CH) & 1u) != 0) {
            visit_channel_with_work<CH>(send_state.worker, channels_in_flight, has_packet, remote, send_yielded);
        }
    }
    if constexpr (SenderVc<VC>::forwarding_channels != 0) {
        if (((holding | channels_in_flight) & SenderVc<VC>::forwarding_channels) != 0) {
            visit_forwarding_channels<SenderVc<VC>::forwarding_channels>(
                states, holding, channels_in_flight, remote, send_yielded);
        }
    }
}

// VC's visit, on its send state where SenderChannelStates places it: its free-slots registers (when not read with the
// pass's first read group), then its channels. The loop-local VC is visited every pass on its state in place. A later
// VC is skipped with one test when none of its channels holds a packet or has packets in flight; otherwise its remote
// receiver progress is taken into a local and put back where it changed (after_visit). None where VC has no serviced
// sender channel.
template <size_t VC, typename LoopCopy>
FORCE_INLINE void visit_vc(
    LoopCopy& loop_copy,
    SenderChannelStates& states,
    uint32_t holding_at_pass_start,
    uint32_t& channels_in_flight,
    bool& send_yielded) {
    if constexpr (SenderVc<VC>::sender_channels != 0) {
        uint32_t holding = holding_at_pass_start;
        if constexpr (!SenderVc<VC>::reads_free_slots_at_pass_start) {
            holding = channels_with_packets<SenderVc<VC>::sender_channels>();
        }
        SenderVcState<VC>& send_state = states.template progress_for<VC>(loop_copy);
        if constexpr (SenderChannelStates::visited_every_pass<VC>) {
            visit_channels<VC>(states, send_state, send_state.remote, holding, channels_in_flight, send_yielded);
        } else {
            if (((holding | channels_in_flight) & SenderVc<VC>::sender_channels) == 0) {
                return;
            }
            const RemoteReceiverChannelProgress<VC> taken = send_state.remote;
            RemoteReceiverChannelProgress<VC> remote = taken;
            visit_channels<VC>(states, send_state, remote, holding, channels_in_flight, send_yielded);
            states.template after_visit<VC>(send_state, taken, remote);
        }
    }
}
'''
s = s[:a] + new + s[b:]
o = '''    // After VC's visit: its remote receiver progress put back (`remote`, taken as `taken` at the visit's start): on the
    // loop's copy, or in memory where it changed, and a later VC's state kept in memory.
    template <size_t VC>
    FORCE_INLINE void after_visit(
        SenderVcState<VC>& send_state,
        const RemoteReceiverChannelProgress<VC>& taken,
        const RemoteReceiverChannelProgress<VC>& remote) {
        if constexpr (VC == loop_local_vc) {
            send_state.remote = remote;
        } else {
            if (remote != taken) {
                send_state.remote = remote;
            }
            keep_in_memory(send_state);
        }
    }'''
n = '''    // After a later VC's visit: its remote receiver progress put back where it changed (`remote`, taken as `taken` at the
    // visit's start), and its state kept in memory. (The loop-local VC is visited on its state in place.)
    template <size_t VC>
    FORCE_INLINE void after_visit(
        SenderVcState<VC>& send_state,
        const RemoteReceiverChannelProgress<VC>& taken,
        const RemoteReceiverChannelProgress<VC>& remote) {
        static_assert(VC != loop_local_vc);
        if (remote != taken) {
            send_state.remote = remote;
        }
        keep_in_memory(send_state);
    }'''
assert s.count(o) == 1
s = s.replace(o, n)
open(p, "w").write(s)
print("made", dst)
