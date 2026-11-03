"""stage_g2.py: v4rxg2 = v4rxg1 + ReceiverChannelStates owning the placement rule (loop_local_vc still 0), the rule's
one block above pass() and run(), and DownstreamSenderChannelProgress's field order {free_slots, next_slot_addr}."""
import os, shutil, sys
V = "/home/aagarwal/tt-metal/fabric_exp/variants"
src, dst = f"{V}/v4rxg1", f"{V}/v4rxg2"
if os.path.exists(dst):
    sys.exit(f"{dst} exists")
shutil.copytree(src, dst, symlinks=True)
R = f"{dst}/tt_metal/fabric/hw/inc/edm_fabric/router"

def edit(name, subs):
    p = f"{R}/{name}"
    s = open(p).read()
    for o, n in subs:
        assert s.count(o) == 1, (name, o[:90])
        s = s.replace(o, n)
    open(p, "w").write(s)

edit("router_receiver.hpp", [
# The state section: the old placement block goes (it moves above pass() and run()).
("""// The VC whose ring progress the receiver loop carries as a local (ReceiverChannelStates).
constexpr size_t FIRST_VC = 0;

// Every serviced VC's receiver channel state (get<VC>()).
// Placement: the first VC's ring progress is carried through the receiver loop as a local copy (run); every later VC's
// state stays in memory between its steps (keep_in_memory, step_channel), so it does not take the first VC's registers
// on its packet path (V4_RECEIVER_OWNERSHIP.md §14; loopvars.py's headroom check guards it).
using ReceiverChannelStates = Each<receiver_vcs, ReceiverChannelState>;

// The first VC's ring progress where this router runs its channel (Present, router_common.hpp).
using FirstVcProgress = Present<runs_vc(FIRST_VC), ReceiverChannelProgress, FIRST_VC>;

// The first VC's ring progress in `states` (Absent where its channel does not run: no step uses it). A template, so
// that an absent channel's state is named only in template-dependent code.
template <typename States>
FORCE_INLINE FirstVcProgress& first_vc_progress_in(States& states) {
    if constexpr (runs_vc(FIRST_VC)) {
        return states.template get<FIRST_VC>().progress;
    } else {
        static FirstVcProgress absent;
        return absent;
    }
}

""", ""),
# step_channel, pass and run: the rule's one block, then its users.
("""// VC's step on its state where the placement puts it (ReceiverChannelStates): the first VC on the loop's local copy
// of its ring progress, a later VC on its state in memory, kept there after the step. None where VC's channel does
// not run.
template <size_t VC, typename FirstProgress, typename States, typename Connections>
FORCE_INLINE void step_channel(
    FirstProgress& first_vc_progress,
    States& states,
    Connections& downstreams,
    Credits& credits,
    bool& credit_frame_due,
    uint32_t arrived_raw,
    const ChipCoordinates& my_chip) {
    if constexpr (runs_vc(VC)) {
        auto& state = states.template get<VC>();
        if constexpr (VC == FIRST_VC) {
            step<VC>(first_vc_progress, state.sources, downstreams, credits, credit_frame_due, arrived_raw, my_chip);
        } else {
            step<VC>(state.progress, state.sources, downstreams, credits, credit_frame_due, arrived_raw, my_chip);
            keep_in_memory(state);
        }
    }
}

// One pass over the serviced VCs' receiver channels: the pass's grouped reads, one step per channel, and the credit
// frame when the counts changed. The group (a read group, begin_read_group): every channel's packets-received register
// and, with first-level acks (a frame goes out most passes), the TX queue's status, raw, each tested where it is used.
// The body of run()'s loop; a single-ERISC router runs it as the receiver half of each pass over its state in memory
// (router_single.hpp). The state types are template parameters so that an absent channel's state is named only in
// template-dependent code.
static_assert(MAX_NUM_VCS == 3, "pass() steps the receiver channels of VCs 0, 1 and 2");
template <typename FirstProgress, typename States, typename Connections>
FORCE_INLINE void pass(
    FirstProgress& first_vc_progress,
    bool& credit_frame_due,
    States& states,
    Credits& credits,
    Connections& downstreams,
    const ChipCoordinates& my_chip) {""",
"""// ---------------------------------------------------------------------------------------------------------------------
// Where the channel states live, and the loop
// ---------------------------------------------------------------------------------------------------------------------

// Every serviced VC's receiver channel state (get<VC>()) and where it lives. The loop-local VC's progress is copied
// into a local of run()'s loop, so the compiler can keep it in registers (loopvars.py --headroom checks); every other
// VC's state stays in memory, kept there after each of its steps (keep_in_memory), so it does not take the loop-local
// VC's registers on its packet path (V4_RECEIVER_OWNERSHIP.md §14). A one-ERISC router keeps every state in memory and
// steps the loop-local VC's progress in place (router_single.hpp). The accessors are templates on the VC so that an
// absent channel's state is named only in template-dependent code.
struct ReceiverChannelStates : Each<receiver_vcs, ReceiverChannelState> {
    // The VC whose progress run() carries as a loop local.
    static constexpr size_t loop_local_vc = 0;

    // The loop-local VC's progress in place: what run() copies at its start; a one-ERISC router steps on it in place.
    // Where that VC's channel does not run, the states themselves: no step uses it.
    template <size_t VC = loop_local_vc>
    FORCE_INLINE auto& loop_local_progress() {
        if constexpr (runs_vc(VC)) {
            return this->template get<VC>().progress;
        } else {
            return *this;
        }
    }

    // VC's progress for its step: the loop's copy for the loop-local VC, the state in memory otherwise.
    template <size_t VC, typename LoopCopy>
    FORCE_INLINE ReceiverChannelProgress& progress_for(LoopCopy& loop_copy) {
        if constexpr (VC == loop_local_vc) {
            return loop_copy;
        } else {
            return this->template get<VC>().progress;
        }
    }

    // After VC's step: a later VC's state is kept in memory; nothing for the loop-local VC.
    template <size_t VC>
    FORCE_INLINE void after_step() {
        if constexpr (VC != loop_local_vc) {
            keep_in_memory(this->template get<VC>());
        }
    }
};

// VC's step, on its progress where ReceiverChannelStates places it. None where VC's channel does not run.
template <size_t VC, typename LoopCopy, typename Connections>
FORCE_INLINE void step_channel(
    LoopCopy& loop_copy,
    ReceiverChannelStates& states,
    Connections& downstreams,
    Credits& credits,
    bool& credit_frame_due,
    uint32_t arrived_raw,
    const ChipCoordinates& my_chip) {
    if constexpr (runs_vc(VC)) {
        step<VC>(
            states.template progress_for<VC>(loop_copy),
            states.template get<VC>().sources,
            downstreams,
            credits,
            credit_frame_due,
            arrived_raw,
            my_chip);
        states.template after_step<VC>();
    }
}

// One pass over the serviced VCs' receiver channels: the pass's grouped reads, one step per channel, and the credit
// frame when the counts changed. The group (a read group, begin_read_group): every channel's packets-received register
// and, with first-level acks (a frame goes out most passes), the TX queue's status, raw, each tested where it is used.
// The body of run()'s loop; a single-ERISC router runs it as the receiver half of each pass over its state in memory
// (router_single.hpp).
static_assert(MAX_NUM_VCS == 3, "pass() steps the receiver channels of VCs 0, 1 and 2");
template <typename LoopCopy, typename Connections>
FORCE_INLINE void pass(
    LoopCopy& loop_copy,
    bool& credit_frame_due,
    ReceiverChannelStates& states,
    Credits& credits,
    Connections& downstreams,
    const ChipCoordinates& my_chip) {"""),
("""    step_channel<0>(first_vc_progress, states, downstreams, credits, credit_frame_due, arrived[0], my_chip);
    step_channel<1>(first_vc_progress, states, downstreams, credits, credit_frame_due, arrived[1], my_chip);
    step_channel<2>(first_vc_progress, states, downstreams, credits, credit_frame_due, arrived[2], my_chip);""",
"""    step_channel<0>(loop_copy, states, downstreams, credits, credit_frame_due, arrived[0], my_chip);
    step_channel<1>(loop_copy, states, downstreams, credits, credit_frame_due, arrived[1], my_chip);
    step_channel<2>(loop_copy, states, downstreams, credits, credit_frame_due, arrived[2], my_chip);"""),
("""    ReceiverChannelStates states;
    FirstVcProgress first_vc_progress = first_vc_progress_in(states);  // a copy, carried through the loop
""", """    ReceiverChannelStates states;
    auto loop_copy = states.loop_local_progress();  // carried through the loop (ReceiverChannelStates)
"""),
("            pass(first_vc_progress, credit_frame_due, states, credits, downstreams, my_chip);",
 "            pass(loop_copy, credit_frame_due, states, credits, downstreams, my_chip);"),
])

edit("router_single.hpp", [
("""// The receiver's state between passes (receiver::run keeps the same objects; there the first VC's ring progress is a
// loop local, here it stays in the state like every VC's).""",
"""// The receiver's state between passes (receiver::run keeps the same objects; there the loop-local VC's progress is
// copied into a loop local, here every VC's stays in the state: receiver::ReceiverChannelStates)."""),
("""        receiver::first_vc_progress_in(r.states), r.credit_frame_due, r.states, r.credits, downstreams, my_chip);""",
 """        r.states.loop_local_progress(), r.credit_frame_due, r.states, r.credits, downstreams, my_chip);"""),
])

edit("router_downstream.hpp", [
("""// Changes with every forward.
struct DownstreamSenderChannelProgress {
    uint32_t next_slot_addr;  // the downstream sender channel's next slot
    uint32_t free_slots;      // slots known free there (a lower bound of the local credit register)
""", """// A downstream sender channel's progress: changes with every forward.
struct DownstreamSenderChannelProgress {
    uint32_t free_slots;      // slots known free there (a lower bound of the local credit register)
    uint32_t next_slot_addr;  // its next slot
"""),
("        downstreams.progress[entry] = {links[entry].first_slot_addr, 0};",
 "        downstreams.progress[entry] = {/*free_slots=*/0, /*next_slot_addr=*/links[entry].first_slot_addr};"),
("        relay.progress = {relay.link.first_slot_addr, 0};",
 "        relay.progress = {/*free_slots=*/0, /*next_slot_addr=*/relay.link.first_slot_addr};"),
])
print("made", dst)
