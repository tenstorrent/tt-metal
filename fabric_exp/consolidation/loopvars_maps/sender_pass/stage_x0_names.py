"""stage_x0_names.py: v4txo0 = v4rxg3 + the sender's renames of V4_SENDER_OWNERSHIP.md §1.0 (names only; the regroup
and the comments are edited by hand after this script)."""
import os, re, shutil, sys
V = "/home/aagarwal/tt-metal/fabric_exp/variants"
src, dst = f"{V}/v4rxg3", f"{V}/v4txo0"
if os.path.exists(dst):
    sys.exit(f"{dst} exists")
shutil.copytree(src, dst, symlinks=True)
R = f"{dst}/tt_metal/fabric/hw/inc/edm_fabric/router"

def edit(path, pairs, regex=()):
    s = open(path).read()
    for o, n in pairs:
        assert s.count(o) >= 1, (path, o[:70])
        s = s.replace(o, n)
    for o, n in regex:
        s2 = re.sub(o, n, s)
        assert s2 != s, (path, o)
        s = s2
    open(path, "w").write(s)

# router_sender.hpp
edit(f"{R}/router_sender.hpp", [
    ("constexpr uint8_t txq = sender_txq_id;",
     "constexpr uint8_t SENDER_TXQ = sender_txq_id;\n"
     "// The NoC of the credits this router writes to its producers (workers and upstream routers).\n"
     "constexpr uint8_t WORKER_CREDIT_NOC = tt::tt_fabric::worker_handshake_noc;"),
    ("    using C = Channel<CH>;\n    using FarT = FarRingTraits<C::far_receiver>;\n",
     "    constexpr size_t VC = SenderChannel<CH>::vc;\n"),
    ("    using C = Channel<CH>;\n    using namespace tt::tt_fabric::connection_interface;\n",
     "    using namespace tt::tt_fabric::connection_interface;\n"),
    ("    ForwardingRecord& r = table.records[record_index<CH>()];\n    using C = Channel<CH>;\n",
     "    ForwardingRecord& state = table.records[record_index<CH>()];\n"),
    ("    using W = WorkerCredit<CH>;\n", ""),
    # the far receiver ring traits, keyed by VC
    ("""// The far router's receiver ring of a VC: its slots and its packets-received register (on the far router).
template <size_t RX>
struct FarRingTraits {
    static constexpr uint32_t num_slots = REMOTE_RECEIVER_NUM_BUFFERS_ARRAY[RX];
    static constexpr uint32_t first_slot = eth_remote_channel_allocs::Entry<REMOTE_RECEIVER_TO_ENTRY_IDX[RX]>::base_address;
    static constexpr uint32_t last_slot = first_slot + (num_slots - 1) * channel_buffer_size;
    static constexpr uint32_t packets_received_stream = to_receiver_packets_sent_streams[RX];
};""", """// The far router's receiver channel of VC: its slots and its packets-received register (on the far router).
template <size_t VC>
struct RemoteReceiverChannel {
    static constexpr size_t rx =  // the host's index of the channel
        VC == 0 ? VC0_RECEIVER_CHANNEL : VC == 1 ? VC1_RECEIVER_CHANNEL : VC2_RECEIVER_CHANNEL;
    static constexpr uint32_t num_slots = REMOTE_RECEIVER_NUM_BUFFERS_ARRAY[rx];
    static constexpr uint32_t first_slot_addr =
        eth_remote_channel_allocs::Entry<REMOTE_RECEIVER_TO_ENTRY_IDX[rx]>::base_address;
    static constexpr uint32_t last_slot_addr = first_slot_addr + (num_slots - 1) * channel_buffer_size;
    static constexpr uint32_t packets_received_stream = to_receiver_packets_sent_streams[rx];
};"""),
    ("""    static constexpr size_t far_receiver =
        vc == 0 ? VC0_RECEIVER_CHANNEL : vc == 1 ? VC1_RECEIVER_CHANNEL : VC2_RECEIVER_CHANNEL;
""", ""),
    ("""template <size_t RX>
constexpr FarRing far_ring_start() {
    return {FarRingTraits<RX>::num_slots, FarRingTraits<RX>::first_slot};
}""", """template <size_t VC>
constexpr RemoteReceiverChannelProgress remote_receiver_channel_progress_start() {
    return {RemoteReceiverChannel<VC>::num_slots, RemoteReceiverChannel<VC>::first_slot_addr};
}"""),
    ("""template <size_t RX0 = VC0_RECEIVER_CHANNEL, size_t RX1 = VC1_RECEIVER_CHANNEL, size_t RX2 = VC2_RECEIVER_CHANNEL>
FORCE_INLINE FarRings far_rings_start() {""", """template <size_t VC0 = 0, size_t VC1 = 1, size_t VC2 = 2>
FORCE_INLINE RemoteReceiverChannelProgresses remote_progresses_start() {"""),
    ("far_ring_start<RX0>()", "remote_receiver_channel_progress_start<VC0>()"),
    ("far_ring_start<RX1>()", "remote_receiver_channel_progress_start<VC1>()"),
    ("far_ring_start<RX2>()", "remote_receiver_channel_progress_start<VC2>()"),
    ("struct FarRing {\n    uint32_t free_slots;\n    uint32_t next_slot;\n};",
     "struct RemoteReceiverChannelProgress {\n    uint32_t free_slots;\n    uint32_t next_slot_addr;\n};"),
    # alias uses
    ("FarT::", "RemoteReceiverChannel<VC>::"),
    ("W::", "WorkerCredit<CH>::"),
    # inline-write byte enables
    ("                0xF,\n", "                INLINE_WRITE_ALL_BYTES,\n"),
    ("pack_value_for_inc_on_write_stream_reg_write(n), 0xf,", "pack_value_for_inc_on_write_stream_reg_write(n), INLINE_WRITE_ALL_BYTES,"),
    ("tt::tt_fabric::worker_handshake_noc", "WORKER_CREDIT_NOC"),
], regex=[
    (r"\bC::", "SenderChannel<CH>::"),
    (r"\bChannel<", "SenderChannel<"),
    (r"struct Channel \{", "struct SenderChannel {"),
    (r"\bFarRings\b", "RemoteReceiverChannelProgresses"),
    (r"\bFarRing\b", "RemoteReceiverChannelProgress"),
    (r"\bfar_rings_start\b", "remote_progresses_start"),
    (r"(?<=::)first_slot\b", "first_slot_addr"),
    (r"(?<=::)last_slot\b", "last_slot_addr"),
    (r"static constexpr uint32_t first_slot = ", "static constexpr uint32_t first_slot_addr = "),
    (r"static constexpr uint32_t last_slot = first_slot \+", "static constexpr uint32_t last_slot_addr = first_slot_addr +"),
    (r"\bcompletion_counter\b", "completion_counter_addr"),
    (r"\back_counter\b", "first_level_ack_counter_addr"),
    (r"\bconnection_semaphore\b(?!s)", "connection_semaphore_addr"),
    (r"\bfar_slots_needed_by\b", "remote_free_slots_to_send_by"),
    (r"\bfar_slots_needed\b", "remote_free_slots_to_send"),
    (r"\btxq\b", "SENDER_TXQ"),
    (r"\.next_slot\b", ".next_slot_addr"),
    (r"\bexpected_connection\b", "connection_state"),
    (r"\.acked\b", ".first_level_acked"),
    (r"uint32_t acked;", "uint32_t first_level_acked;"),
    (r"\bupstream_credit_addr\b", "upstream_free_slots_addr"),
    (r"\bin_flight\b", "forwarding_in_flight"),
    (r"\bpending1\b", "holding_vc1"),
    (r"\bpending\b", "holding"),
    (r"\bsemaphores\b", "connection_semaphores"),
    (r"\battempted\b", "send_yielded"),
    (r"\bw\.", "worker."),
    (r"Worker& w\b", "Worker& worker"),
    (r"\br\.", "state."),
    (r"\bfar\b", "remote"),
    (r"\bFar\b", "Remote"),
    (r"\bsrc\b", "slot_addr"),
])
s = open(f"{R}/router_sender.hpp").read()
s = s.replace("constexpr uint8_t WORKER_CREDIT_NOC = WORKER_CREDIT_NOC;",
              "constexpr uint8_t WORKER_CREDIT_NOC = tt::tt_fabric::worker_handshake_noc;")
# send's and the forwarding visit's slot locals
for o, n in [
    ("    uint32_t& slot,\n", "    uint32_t& next_slot_addr,\n"),
    ("    const uint32_t slot_addr = slot;\n", "    const uint32_t slot_addr = next_slot_addr;\n"),
    ("    slot = slot_addr == SenderChannel<CH>::last_slot_addr", "    next_slot_addr = slot_addr == SenderChannel<CH>::last_slot_addr"),
    ("        uint32_t slot = state.slot;\n", "        uint32_t next_slot_addr = state.next_slot_addr;\n"),
    ("did_send = send<CH>(slot, remote,", "did_send = send<CH>(next_slot_addr, remote,"),
    ("            state.slot = slot;\n", "            state.next_slot_addr = next_slot_addr;\n"),
    ("    keep_in_memory(r);\n", "    keep_in_memory(state);\n"),
    ("sent = send<CH>(worker.slot, remote,", "sent = send<CH>(worker.next_slot_addr, remote,"),
    ("    uint32_t slot;       // next slot to send from\n", "    uint32_t next_slot_addr;  // next slot to send from\n"),
    ("struct Worker {\n    uint32_t slot;\n", "struct Worker {\n    uint32_t next_slot_addr;\n"),
]:
    assert s.count(o) == 1, o[:70]
    s = s.replace(o, n)
open(f"{R}/router_sender.hpp", "w").write(s)

# the read groups (four): channels_with_packets, send's two, visit_worker's
s = open(f"{R}/router_sender.hpp").read()
for o, n in [
    ("""    std::atomic_signal_fence(std::memory_order_seq_cst);
    read_free_slots_registers<MASK>(registers);
    read_connection_semaphores<WORKERS>(connection_semaphores);
    std::atomic_signal_fence(std::memory_order_seq_cst);""", """    begin_read_group();
    read_free_slots_registers<MASK>(registers);
    read_connection_semaphores<WORKERS>(connection_semaphores);
    end_read_group();"""),
    ("""    std::atomic_signal_fence(std::memory_order_seq_cst);
    const uint32_t payload_size = header->get_payload_size_excluding_header();
    const uint32_t status = read_txq_status(SENDER_TXQ);
    std::atomic_signal_fence(std::memory_order_seq_cst);""", """    begin_read_group();
    const uint32_t payload_size = header->get_payload_size_excluding_header();
    const uint32_t status = read_txq_status(SENDER_TXQ);
    end_read_group();"""),
    ("""    std::atomic_signal_fence(std::memory_order_seq_cst);
    const uint32_t status_after = read_txq_status(SENDER_TXQ);""", """    begin_read_group();
    const uint32_t status_after = read_txq_status(SENDER_TXQ);"""),
    ("""        reported_acks = *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(SenderChannel<CH>::first_level_ack_counter_addr);
    }
    std::atomic_signal_fence(std::memory_order_seq_cst);

""", """        reported_acks = *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(SenderChannel<CH>::first_level_ack_counter_addr);
    }
    end_read_group();

"""),
    ("""    uint32_t reported_acks = sent_acks;
    std::atomic_signal_fence(std::memory_order_seq_cst);
    if (read_completions) {""", """    uint32_t reported_acks = sent_acks;
    begin_read_group();
    if (read_completions) {"""),
    ("""    }
    std::atomic_signal_fence(std::memory_order_seq_cst);
    if (read_completions) {
        completions = reported_completions""", """    }
    end_read_group();
    if (read_completions) {
        completions = reported_completions"""),
]:
    assert s.count(o) == 1, o[:90]
    s = s.replace(o, n)
open(f"{R}/router_sender.hpp", "w").write(s)

# INLINE_WRITE_ALL_BYTES to router_common.hpp
edit(f"{R}/router_downstream.hpp", [("""// Byte enables of a whole-word inline write (the credit register).
constexpr uint32_t INLINE_WRITE_ALL_BYTES = 0xF;

""", "")])
edit(f"{R}/router_common.hpp", [("""// Whether NoC command buffers stay programmed between commands""", """// Byte enables of a whole-word inline write (credit registers and words).
constexpr uint32_t INLINE_WRITE_ALL_BYTES = 0xF;

// Whether NoC command buffers stay programmed between commands""")])
# the round's flag
edit(f"{R}/router_round.hpp", [
    ("    bool attempted = false;         // sender: a send yielded (TX queue busy) this round",
     "    bool send_yielded = false;      // sender: a send yielded (TX queue busy) this round"),
    ("bool end_round(RoundCounters& counters, bool attempted_send) {", "bool end_round(RoundCounters& counters, bool send_yielded) {"),
    ("        if (IDLE_CONTEXT_SWITCHING && attempted_send) {", "        if (IDLE_CONTEXT_SWITCHING && send_yielded) {"),
])
# router_single.hpp
edit(f"{R}/router_single.hpp", [
    ("    uint32_t in_flight = 0;\n    sender::FarRings far;", "    uint32_t forwarding_in_flight = 0;\n    sender::RemoteReceiverChannelProgresses remote;"),
    ("__attribute__((noinline)) void sender_pass(State& s, bool& attempted) {\n    sender::pass(s.workers, s.connections, s.forwarding, s.in_flight, s.far, attempted);",
     "__attribute__((noinline)) void sender_pass(State& s, bool& send_yielded) {\n    sender::pass(s.workers, s.connections, s.forwarding, s.forwarding_in_flight, s.remote, send_yielded);"),
    ("sender::far_rings_start()", "sender::remote_progresses_start()"),
], regex=[(r"round\.attempted", "round.send_yielded")])
print("made", dst)
