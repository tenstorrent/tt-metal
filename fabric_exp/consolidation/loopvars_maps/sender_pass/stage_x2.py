"""stage_x2.py: v4txo2 = v4txo1 + host facts (V4_SENDER_OWNERSHIP.md §2 #1-#5, #11, #12): the worker role from the host's
SENDER_CH_<i>_IS_WORKER_CHANNEL, per-VC ct arrays (first sender channel, sender channel count, receiver channel), the
counter address helper in fabric_router_flow_control.hpp, and the static_asserts (counter credits, the host's
connection check on every worker channel, one worker channel per VC). (#6, the shared bubble-flow-control constant,
goes with the host patch of X4.)"""
import os, shutil, sys
V = "/home/aagarwal/tt-metal/fabric_exp/variants"
src, dst = f"{V}/v4txo1", f"{V}/v4txo2"
if os.path.exists(dst):
    sys.exit(f"{dst} exists")
shutil.copytree(src, dst, symlinks=True)
R = f"{dst}/tt_metal/fabric/hw/inc/edm_fabric/router"

def edit(path, pairs):
    s = open(path).read()
    for o, n in pairs:
        assert s.count(o) == 1, (path, o[:80])
        s = s.replace(o, n)
    open(path, "w").write(s)

edit(f"{dst}/fabric_erisc_router_ct_args.hpp", [
    ("""constexpr std::array<bool, MAX_NUM_VCS> is_vc_crossover_to_vc1 = {""",
     """// Each VC's sender channels, as the host lays them out (VC0's, then VC1's, then VC2's): the first one's index and their
// count; and the VC's receiver channel's index in the host's per-receiver-channel arrays (on both routers of a link).
constexpr std::array<size_t, MAX_NUM_VCS> first_sender_channel_by_vc = {0, VC1_LOCAL_CHANNEL_START, VC2_LOCAL_CHANNEL_START};
constexpr std::array<size_t, MAX_NUM_VCS> sender_channel_count_by_vc = {
    ACTUAL_VC0_SENDER_CHANNELS, ACTUAL_VC1_SENDER_CHANNELS, ACTUAL_VC2_SENDER_CHANNELS};
constexpr std::array<size_t, MAX_NUM_VCS> receiver_channel_by_vc = {
    VC0_RECEIVER_CHANNEL, VC1_RECEIVER_CHANNEL, VC2_RECEIVER_CHANNEL};
constexpr std::array<bool, MAX_NUM_VCS> is_vc_crossover_to_vc1 = {"""),
    ("""// ============================================================================
// Channel servicing flags
// ============================================================================
""", """// ============================================================================
// Channel servicing flags
// ============================================================================
// Whether a sender channel position takes a worker: the first sender channel of each VC whose routing mode has a worker
// (SENDER_CH_<i>_IS_WORKER_CHANNEL; erisc_datamover_builder.cpp), whether or not trimming services it.
constexpr std::array<bool, MAX_NUM_SENDER_CHANNELS> is_sender_channel_worker = {
    static_cast<bool>(NAMED_CT_ARG("SENDER_CH_0_IS_WORKER_CHANNEL")),
    static_cast<bool>(NAMED_CT_ARG("SENDER_CH_1_IS_WORKER_CHANNEL")),
    static_cast<bool>(NAMED_CT_ARG("SENDER_CH_2_IS_WORKER_CHANNEL")),
    static_cast<bool>(NAMED_CT_ARG("SENDER_CH_3_IS_WORKER_CHANNEL")),
    static_cast<bool>(NAMED_CT_ARG("SENDER_CH_4_IS_WORKER_CHANNEL")),
    static_cast<bool>(NAMED_CT_ARG("SENDER_CH_5_IS_WORKER_CHANNEL")),
    static_cast<bool>(NAMED_CT_ARG("SENDER_CH_6_IS_WORKER_CHANNEL")),
    static_cast<bool>(NAMED_CT_ARG("SENDER_CH_7_IS_WORKER_CHANNEL")),
    static_cast<bool>(NAMED_CT_ARG("SENDER_CH_8_IS_WORKER_CHANNEL")),
    static_cast<bool>(NAMED_CT_ARG("SENDER_CH_9_IS_WORKER_CHANNEL")),
};
"""),
])

edit(f"{dst}/fabric_router_flow_control.hpp", [
    ("""struct SenderChannelFromReceiverCounterBasedCreditsReceiver {""",
     """// Sender channel `ch`'s counter in a block of per-sender-channel counters at `base`, one word each: where the far
// router's receiver writes that channel's completions and first-level acks (the receiver's credit sender indexes its
// block by the packet's source channel; the receiver below reads them).
constexpr uint32_t sender_channel_counter_addr(uint32_t base, size_t ch) {
    return base + static_cast<uint32_t>(ch * sizeof(uint32_t));
}

struct SenderChannelFromReceiverCounterBasedCreditsReceiver {"""),
])

edit(f"{R}/router_sender.hpp", [
    ("""// The host's layout of the sender channels: VC0's, then VC1's, then VC2's (ct args).
constexpr size_t first_sender_channel_of(size_t vc) {
    return vc == 0 ? 0 : vc == 1 ? VC1_LOCAL_CHANNEL_START : VC2_LOCAL_CHANNEL_START;
}
constexpr size_t sender_channel_count_of(size_t vc) {
    return vc == 0 ? ACTUAL_VC0_SENDER_CHANNELS : vc == 1 ? ACTUAL_VC1_SENDER_CHANNELS : ACTUAL_VC2_SENDER_CHANNELS;
}
constexpr size_t vc_of_sender_channel(size_t ch) {
    return ch < VC1_LOCAL_CHANNEL_START ? 0 : ch < VC2_LOCAL_CHANNEL_START ? 1 : 2;
}

// A worker channel: the first sender channel of a VC whose routing mode has a worker (VC0's; VC2's, neighbour exchange;
// VC1 is router to router), as the host marks it (erisc_datamover_builder.cpp). For every channel position, serviced or
// not; the others are forwarding channels, which the routers on this chip that face other directions write into.
constexpr bool is_worker_channel(size_t ch) {
    return vc_of_sender_channel(ch) != 1 && ch == first_sender_channel_of(vc_of_sender_channel(ch));
}

""", ""),
    ("""    static constexpr bool worker = is_worker_channel(CH);""",
     """    // A worker channel takes a worker (the host's role); the others are forwarding channels, which the routers on this
    // chip that face other directions write into.
    static constexpr bool worker = is_sender_channel_worker[CH];"""),
    ("""    static constexpr uint32_t completion_counter_addr = to_sender_remote_completion_counters_base_address + CH * 4;
    static constexpr uint32_t first_level_ack_counter_addr = to_sender_remote_ack_counters_base_address + CH * 4;""",
     """    static constexpr uint32_t completion_counter_addr =
        sender_channel_counter_addr(to_sender_remote_completion_counters_base_address, CH);
    static constexpr uint32_t first_level_ack_counter_addr =
        sender_channel_counter_addr(to_sender_remote_ack_counters_base_address, CH);"""),
    ("""    static_assert(num_slots > 0 && num_slots <= stream_count_mask);
};

// VC's sender features""", """    static_assert(num_slots > 0 && num_slots <= stream_count_mask);
    static_assert(sender_channel_uses_counter_credits(CH), "the sender reads its credits from L1 counters");
    static_assert(!worker || !sender_ch_live_check_skip[CH], "the host checks every worker channel's connection");
};

// VC's sender features"""),
    ("""    static constexpr uint32_t sender_channels = channels_in(first_sender_channel_of(VC), sender_channel_count_of(VC));
    static constexpr uint32_t worker_channels =
        is_worker_channel(first_sender_channel_of(VC)) ? sender_channels & (1u << first_sender_channel_of(VC)) : 0u;
    static constexpr uint32_t forwarding_channels = sender_channels & ~worker_channels;
    static constexpr bool first_level_ack = is_vc_first_level_ack_enabled[VC];
    static constexpr bool reads_free_slots_at_pass_start = is_worker_channel(first_sender_channel_of(VC));
};""", """    static constexpr uint32_t sender_channels =
        channels_in(first_sender_channel_by_vc[VC], sender_channel_count_by_vc[VC]);
    static constexpr uint32_t worker_channels = sender_channels & worker_channel_positions(VC);
    static constexpr uint32_t forwarding_channels = sender_channels & ~worker_channels;
    static constexpr bool first_level_ack = is_vc_first_level_ack_enabled[VC];
    static constexpr bool reads_free_slots_at_pass_start = worker_channel_positions(VC) != 0;
    static_assert(popcount(worker_channel_positions(VC)) <= 1, "a VC takes at most one worker");
};"""),
    ("""// The serviced sender channels among the `count` channels from `first`, one bit each.""",
     """// The worker channel positions of VC (the host's role, serviced or not), one bit each.
constexpr uint32_t worker_channel_positions(size_t vc) {
    uint32_t mask = 0;
    for (size_t ch = first_sender_channel_by_vc[vc];
         ch < first_sender_channel_by_vc[vc] + sender_channel_count_by_vc[vc] && ch < MAX_NUM_SENDER_CHANNELS;
         ch++) {
        mask |= is_sender_channel_worker[ch] ? (1u << ch) : 0u;
    }
    return mask;
}

// The serviced sender channels among the `count` channels from `first`, one bit each."""),
    ("""    static constexpr size_t rx =  // the host's index of the channel
        VC == 0 ? VC0_RECEIVER_CHANNEL : VC == 1 ? VC1_RECEIVER_CHANNEL : VC2_RECEIVER_CHANNEL;""",
     """    static constexpr size_t rx = receiver_channel_by_vc[VC];  // the host's index of the channel"""),
    ("""    static constexpr size_t vc = vc_of_sender_channel(CH);""", """    static constexpr size_t vc = vc_of_sender_channel(CH);  // ct args"""),
])
s = open(f"{R}/router_sender.hpp").read()
assert "first_sender_channel_of(" in s
s = s.replace("first_sender_channel_of(VC)", "first_sender_channel_by_vc[VC]")
assert "first_sender_channel_of" not in s and "is_worker_channel(" not in s, "leftover"
open(f"{R}/router_sender.hpp", "w").write(s)

edit(f"{dst}/fabric_erisc_router_v4.cpp", [
    ("""        constexpr bool vc2_worker = CH == VC2_LOCAL_CHANNEL_START;""",
     """        constexpr bool worker = is_sender_channel_worker[CH];  // the host's role, serviced or not"""),
    ("""        ASSERT(!(CH == 0 || vc2_worker) || args.connection_semaphores[CH] == sender_connection_semaphore_addrs[CH]);""",
     """        ASSERT(!worker || args.connection_semaphores[CH] == sender_connection_semaphore_addrs[CH]);"""),
    ("""                CH == 0 || vc2_worker ? 0 : SENDER_NUM_BUFFERS_ARRAY[CH],
                vc2_worker ? sender::WorkerCredit<CH>::source : notify_worker_of_read_counter_update_src_address);""",
     """                worker ? 0 : SENDER_NUM_BUFFERS_ARRAY[CH],
                worker && CH != 0 ? sender::WorkerCredit<CH>::source : notify_worker_of_read_counter_update_src_address);"""),
])
print("made", dst)
