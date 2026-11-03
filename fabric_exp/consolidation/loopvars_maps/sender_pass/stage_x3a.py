"""stage_x3a.py: v4txo3a = v4txo3k + A (V4_SENDER_OWNERSHIP.md §4 X3): the sender channels' interfaces at a fixed
location (worker_interface<CH>(), built once by the kernel), not the loop's copies; UpstreamCredits gone (a forwarding
channel's upstream address read from its interface, where the connection cached it)."""
import os, re, shutil, sys
V = "/home/aagarwal/tt-metal/fabric_exp/variants"
src, dst = f"{V}/v4txo3k", f"{V}/v4txo3a"
if os.path.exists(dst):
    sys.exit(f"{dst} exists")
shutil.copytree(src, dst, symlinks=True)
R = f"{dst}/tt_metal/fabric/hw/inc/edm_fabric/router"
p = f"{R}/router_sender.hpp"
s = open(p).read()

def rep(o, n, count=1):
    global s
    assert s.count(o) == count, (s.count(o), o[:90])
    s = s.replace(o, n)

# the interfaces at their fixed location, defined before the credits (which use them)
rep("""// ---------------------------------------------------------------------------------------------------------------------
// Free-slots registers
// ---------------------------------------------------------------------------------------------------------------------
""", """// The sender channels' interfaces (the library's connection state, the producer's credit address), at a fixed
// location: built by the kernel at init, then reached by channel (worker_interface<CH>()). Storage and an explicit
// construction: the kernel runs no global constructors.
using WorkerInterfaces = decltype(tt::tt_fabric::EdmChannelWorkerInterfaces<WORKER_CREDIT_NOC, SENDER_NUM_BUFFERS_ARRAY>::make(
    std::make_index_sequence<NUM_SENDER_CHANNELS>{}));
struct WorkerInterfaceStorage {
    alignas(WorkerInterfaces) unsigned char bytes[sizeof(WorkerInterfaces)];
};
inline WorkerInterfaceStorage worker_interface_storage;
// Builds the interfaces at their location (once, at init); the kernel then sets each one up.
FORCE_INLINE WorkerInterfaces& start_worker_interfaces() {
    return *new (worker_interface_storage.bytes) WorkerInterfaces{};
}
template <size_t CH>
FORCE_INLINE auto& worker_interface() {
    return std::launder(reinterpret_cast<WorkerInterfaces*>(worker_interface_storage.bytes))->template get<CH>();
}

// ---------------------------------------------------------------------------------------------------------------------
// Free-slots registers
// ---------------------------------------------------------------------------------------------------------------------
""")
# no more connection threading
s = s.replace("template <size_t CH, typename Connections>\n", "template <size_t CH>\n")
s = s.replace("template <uint32_t MASK, typename Connections, typename Remote>\n", "template <uint32_t MASK, typename Remote>\n")
s = s.replace("template <size_t VC, typename LoopCopy, typename Connections>\n", "template <size_t VC, typename LoopCopy>\n")
s = s.replace("template <typename LoopCopy, typename Connections>\n", "template <typename LoopCopy>\n")
s = s.replace("    Connections& connections,\n", "")
s = re.sub(r", Connections& connections(?=[,)])", "", s)
s = s.replace("connections.template get<CH>()", "worker_interface<CH>()")
s = s.replace("connections.template get<0>()", "worker_interface<0>()")
s = re.sub(r"\bconnections, ", "", s)
s = re.sub(r", connections\)", ")", s)
assert "Connections" not in s, [l for l in s.split("\n") if "Connections" in l][:5]

rep("""// The producers' addresses of the forwarding channels, cached when each upstream router connected.
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
}""", """// The forwarding channels in MASK: each one's upstream router's free-slots register address, as its interface cached it
// when the upstream router connected.
template <uint32_t MASK>
FORCE_INLINE void init_forwarding(SenderChannelStates& states) {
    if constexpr (MASK != 0) {
        constexpr size_t CH = lowest_bit(MASK);
        states.forwarding.template get<CH>().upstream_free_slots_addr = worker_interface<CH>().cached_worker_semaphore_address;
        init_forwarding<without_lowest_bit(MASK)>(states);
    }
}""")
i = s.index("// The worker channels' interfaces, the loop's own copies")
j = s.index("// Something else used the write command buffer")
s = s[:i] + s[j:]
rep(
"""template <typename WorkerInterfaces>
__attribute__((noinline)) void run(WorkerInterfaces& interfaces, const UpstreamCredits& upstream) {
    auto connections = worker_connections<worker_channels>(interfaces);
    SenderChannelStates states;
    init_forwarding<forwarding_channels>(states, upstream);""", """__attribute__((noinline)) void run() {
    SenderChannelStates states;
    init_forwarding<forwarding_channels>(states);""")
assert not re.search(r"\bconnections\b", s), [l for l in s.split("\n") if re.search(r"\bconnections\b", l)][:8]
open(p, "w").write(s)

p = f"{R}/router_single.hpp"
t = open(p).read()
for o, n in [
    ("""template <typename Connections>
struct SenderState {
    Connections connections;
    sender::SenderChannelStates states;""", """struct SenderState {
    sender::SenderChannelStates states;"""),
    ("""template <typename State>
__attribute__((noinline)) void sender_pass(State& s, bool& send_yielded) {
    sender::pass(s.states.loop_local_progress(), s.states, s.connections, s.channels_in_flight, send_yielded);""",
     """__attribute__((noinline)) void sender_pass(SenderState& s, bool& send_yielded) {
    sender::pass(s.states.loop_local_progress(), s.states, s.channels_in_flight, send_yielded);"""),
    ("""template <typename WorkerInterfaces>
__attribute__((noinline)) void run(
    WorkerInterfaces& interfaces, const sender::UpstreamCredits& upstream, DownstreamConnections& downstreams) {""",
     """__attribute__((noinline)) void run(DownstreamConnections& downstreams) {"""),
    ("""    SenderState<decltype(sender::worker_connections<sender::worker_channels>(interfaces))> s{
        sender::worker_connections<sender::worker_channels>(interfaces)};
    sender::init_forwarding<sender::forwarding_channels>(s.states, upstream);""",
     """    SenderState s;
    sender::init_forwarding<sender::forwarding_channels>(s.states);"""),
]:
    assert t.count(o) == 1, o[:80]
    t = t.replace(o, n)
t = t.replace("s.states, s.connections)", "s.states)")
assert "connections" not in t.replace("DownstreamConnections", "").replace("downstreams", ""), "single leftover"
open(p, "w").write(t)

p = f"{dst}/fabric_erisc_router_v4.cpp"
t = open(p).read()
for o, n in [
    ("""using WorkerInterfaces = decltype(tt::tt_fabric::EdmChannelWorkerInterfaces<tt::tt_fabric::worker_handshake_noc, SENDER_NUM_BUFFERS_ARRAY>::make(
    std::make_index_sequence<NUM_SENDER_CHANNELS>{}));
""", """using sender::WorkerInterfaces;
"""),
    ("""__attribute__((noinline)) void wait_for_upstream_routers(WorkerInterfaces& interfaces, sender::UpstreamCredits& upstream) {""",
     """__attribute__((noinline)) void wait_for_upstream_routers(WorkerInterfaces& interfaces) {"""),
    ("""            static_assert(((sender::forwarding_channels >> CH) & 1u) != 0, "a static upstream connection is a forwarding channel's");
            upstream.addr[sender::forwarding_index<CH>()] = interface.cached_worker_semaphore_address;
""", """            static_assert(((sender::forwarding_channels >> CH) & 1u) != 0, "a static upstream connection is a forwarding channel's");
"""),
    ("""    WorkerInterfaces interfaces{};""", """    WorkerInterfaces& interfaces = sender::start_worker_interfaces();"""),
    ("""    sender::UpstreamCredits upstream{};
    if constexpr (is_sender_role) {
        wait_for_upstream_routers(interfaces, upstream);
    }""", """    if constexpr (is_sender_role) {
        wait_for_upstream_routers(interfaces);
    }"""),
    ("""        single::run(interfaces, upstream, downstreams);""", """        single::run(downstreams);"""),
    ("""        sender::run(interfaces, upstream);""", """        sender::run();"""),
]:
    assert t.count(o) == 1, o[:80]
    t = t.replace(o, n)
open(p, "w").write(t)
print("made", dst)
