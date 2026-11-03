"""stage_x1.py: v4txo1 = v4txo0 + the structure (V4_SENDER_OWNERSHIP.md §4 X1): SenderVc<VC> / SenderChannel<CH>
features from the serviced set, SenderChannelState<CH> {progress; producer link}, SenderVcState<VC>, Each containers,
SenderChannelStates owning today's placement rule (every VC's send state loop-local), visit_vc<0..2>, one producer
credit path (return_slots). The new sections are in loopvars_maps/sender_pass/x1/."""
import os, shutil, sys
V = "/home/aagarwal/tt-metal/fabric_exp/variants"
X = os.path.join(os.path.dirname(os.path.abspath(__file__)), "x1")
src, dst = f"{V}/v4txo0", f"{V}/v4txo1"
if os.path.exists(dst):
    sys.exit(f"{dst} exists")
shutil.copytree(src, dst, symlinks=True)
R = f"{dst}/tt_metal/fabric/hw/inc/edm_fabric/router"
p = f"{R}/router_sender.hpp"
s = open(p).read()
SEP = "// ---------------------------------------------------------------------------------------------------------------------\n"
a = s.index(SEP + "// Channels and their features\n")
b = s.index(SEP + "// Free-slots registers\n")
c = s.index(SEP + "// Visits\n")
s = s[:a] + open(f"{X}/features_state.hpp").read() + s[b:c] + open(f"{X}/visits_loop.hpp").read()
for o, n in [
    ("template <typename WorkerInterface>\n__attribute__((noinline)) void preset_worker_credit(const WorkerInterface& interface) {",
     "template <typename Interface>\n__attribute__((noinline)) void preset_worker_credit(const Interface& interface) {"),
    ("template <size_t CH, typename WorkerInterface>\n__attribute__((noinline)) void credit_worker(const WorkerInterface& interface,",
     "template <size_t CH, typename Interface>\n__attribute__((noinline)) void credit_worker(const Interface& interface,"),
    ("template <size_t CH, typename WorkerInterface>\n__attribute__((noinline)) uint32_t worker_connection_event(\n    WorkerInterface& interface,",
     "template <size_t CH, typename Interface>\n__attribute__((noinline)) uint32_t worker_connection_event(\n    Interface& interface,"),
]:
    assert s.count(o) == 1, o[:80]
    s = s.replace(o, n)
open(p, "w").write(s)

# router_common.hpp: Each's const get
p = f"{R}/router_common.hpp"
s = open(p).read()
o = """    template <size_t I>
    FORCE_INLINE T<I>& get() {
        if constexpr (I == lowest_bit(MASK)) {
            return first;
        } else {
            return rest.template get<I>();
        }
    }
};"""
n = """    template <size_t I>
    FORCE_INLINE T<I>& get() {
        if constexpr (I == lowest_bit(MASK)) {
            return first;
        } else {
            return rest.template get<I>();
        }
    }
    template <size_t I>
    FORCE_INLINE const T<I>& get() const {
        if constexpr (I == lowest_bit(MASK)) {
            return first;
        } else {
            return rest.template get<I>();
        }
    }
};"""
assert s.count(o) == 1
s = s.replace(o, n)
o = """    template <size_t I>
    T<I>& get() {
        static_assert(NotHeld<I>::value, "no member for this index");
    }
};"""
n = """    template <size_t I>
    T<I>& get() {
        static_assert(NotHeld<I>::value, "no member for this index");
    }
    template <size_t I>
    const T<I>& get() const {
        static_assert(NotHeld<I>::value, "no member for this index");
    }
};"""
assert s.count(o) == 1
s = s.replace(o, n)
open(p, "w").write(s)

# router_single.hpp: the sender's state record
p = f"{R}/router_single.hpp"
s = open(p).read()
for o, n in [
    ("""// The sender's state between passes (sender::run keeps the same groups as loop locals).
template <typename Connections>
struct SenderState {
    Connections connections;
    sender::Workers workers;
    sender::ForwardingTable forwarding;
    uint32_t forwarding_in_flight = 0;
    sender::RemoteReceiverChannelProgresses remote;
};""", """// The sender's state between passes (sender::run keeps the same objects; there the loop-local send state is copied
// into a loop local, here it stays in the state: sender::SenderChannelStates).
template <typename Connections>
struct SenderState {
    Connections connections;
    sender::SenderChannelStates states;
    uint32_t forwarding_in_flight = 0;
};"""),
    ("""    sender::pass(s.workers, s.connections, s.forwarding, s.forwarding_in_flight, s.remote, send_yielded);""",
     """    sender::pass(s.states.loop_local_progress(), s.states, s.connections, s.forwarding_in_flight, send_yielded);"""),
    ("""    SenderState<decltype(sender::worker_connections(interfaces))> s{
        sender::worker_connections(interfaces), {}, {}, 0, sender::remote_progresses_start()};
    s.workers = sender::workers_start(s.connections);
    sender::init_forwarding<sender::all_forwarding>(s.forwarding, upstream);""",
     """    SenderState<decltype(sender::worker_connections<sender::worker_channels>(interfaces))> s{
        sender::worker_connections<sender::worker_channels>(interfaces)};
    sender::init_forwarding<sender::forwarding_channels>(s.states, upstream);"""),
]:
    assert s.count(o) == 1, o[:80]
    s = s.replace(o, n)
s = s.replace("sender::restore_worker_credit_preset(s.workers, s.connections);",
              "sender::restore_worker_credit_preset(s.states.loop_local_progress(), s.states, s.connections);")
s = s.replace("sender::total_sent(s.workers, s.forwarding)", "sender::total_sent(s.states.loop_local_progress(), s.states)")
open(p, "w").write(s)

# fabric_erisc_router_v4.cpp: the forwarding channels' names
p = f"{dst}/fabric_erisc_router_v4.cpp"
s = open(p).read()
for o, n in [("sender::all_forwarding", "sender::forwarding_channels"), ("sender::record_index<CH>()", "sender::forwarding_index<CH>()")]:
    assert s.count(o) >= 1, o
    s = s.replace(o, n)
open(p, "w").write(s)
print("made", dst)
