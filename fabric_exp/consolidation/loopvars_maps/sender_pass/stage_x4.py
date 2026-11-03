"""stage_x4.py: v4txo4 = v4txo3f + B (V4_SENDER_OWNERSHIP.md §4 X4): each worker channel's credits on the command buffer
the host gives it (SENDER_CH_<i>_ACK_CMD_BUF_ID: VC0's worker channel WR, VC2's WR_REG; host_rebuild_20261102b), from
its own read counter, preset on two ERISCs (every worker channel, VC2's too) and posted; the connection's opening
credit on that buffer; the preset restored after the base firmware ran, not after another channel's connection event."""
import os, shutil, sys
V = "/home/aagarwal/tt-metal/fabric_exp/variants"
src, dst = f"{V}/v4txo3f", f"{V}/v4txo4"
if os.path.exists(dst):
    sys.exit(f"{dst} exists")
shutil.copytree(src, dst, symlinks=True)
p = f"{dst}/tt_metal/fabric/hw/inc/edm_fabric/router/router_sender.hpp"
s = open(p).read()

def rep(o, n):
    global s
    assert s.count(o) == 1, o[:90]
    s = s.replace(o, n)

# the channel's credit facts
rep("""    static constexpr uint32_t remote_free_slots_to_send = remote_free_slots_to_send_by(CH);""",
    """    static constexpr uint32_t remote_free_slots_to_send = remote_free_slots_to_send_by(CH);
    // A worker channel's credits: the command buffer they go out on (the host's), and its read counter, which holds the
    // credits returned so far and is the source word of each credit write.
    static constexpr uint8_t producer_credit_cmd_buf = sender_channel_ack_cmd_buf_ids[CH];
    static constexpr uint32_t read_counter_addr = worker_read_counter_addr(CH);""")
rep("""// The serviced sender channels among the `count` channels from `first`, one bit each.""",
    """// Channel `ch`'s read counter in its connection info: a worker reads the credits returned so far there when it connects.
constexpr uint32_t worker_read_counter_addr(size_t ch) {
    return sender_connection_info_addrs[ch] + offsetof(tt::tt_fabric::EDMChannelWorkerLocationInfo, edm_read_counter);
}

// The serviced sender channels among the `count` channels from `first`, one bit each.""")
rep("""// In Tensix mux mode the mux is the only producer: no router forwards into this one.""",
    """// Each serviced worker channel's credits on a command buffer of its own, none the atomics buffer (the upstream credits'
// and the teardowns').
constexpr bool worker_credit_cmd_bufs_distinct() {
    uint32_t used = 1u << write_at_cmd_buf;
    for (size_t ch = 0; ch < NUM_SENDER_CHANNELS; ch++) {
        if (((worker_channels >> ch) & 1u) != 0) {
            if (((used >> sender_channel_ack_cmd_buf_ids[ch]) & 1u) != 0) {
                return false;
            }
            used |= 1u << sender_channel_ack_cmd_buf_ids[ch];
        }
    }
    return true;
}
static_assert(worker_credit_cmd_bufs_distinct(), "each worker channel's credits need a command buffer of their own");
// In Tensix mux mode the mux is the only producer: no router forwards into this one.""")

# the credit path
a = s.index("// The credit write. VC0's worker: Blackhole emulates an inline write to L1")
b = s.index("// To an upstream router: +n on its free-slots register.")
s = s[:a] + """// The worker credit: the count goes into the channel's read counter (SenderChannel<CH>::read_counter_addr), which
// therefore always holds it, and while a worker is connected a 4-byte posted NoC write copies that word into the
// worker's L1 on the channel's own command buffer (SenderChannel<CH>::producer_credit_cmd_buf). With two ERISCs that
// buffer stays programmed for the whole connection (preset_worker_credit: at the connection, and again after the base
// firmware ran), so a credit only issues the write; with one ERISC both roles share the buffers and every credit is a
// whole command. The sender's upstream credits and teardowns use the atomics buffer.
constexpr bool worker_credit_preset = command_buffer_presets;
static_assert(!enable_read_counter_update_noc_flush);  // the read counter may be rewritten before an earlier write read
                                                       // it: the worker then sees the newer, larger count

// Programs worker channel CH's credit command buffer for its connected worker (two ERISCs).
template <size_t CH>
__attribute__((noinline)) void preset_worker_credit() {
    if constexpr (worker_credit_preset) {
        ncrisc_noc_write_set_state</*posted=*/true, /*one_packet=*/true>(
            WORKER_CREDIT_NOC,
            SenderChannel<CH>::producer_credit_cmd_buf,
            worker_interface<CH>().cached_worker_semaphore_address,
            sizeof(uint32_t),
            NOC_UNICAST_WRITE_VC);
    }
}

// To worker channel CH's worker: the absolute count of slots returned, into the channel's read counter and, while a
// worker is connected, from there into the worker's L1 (unflushed: a later, larger count may overwrite the word before
// this write reads it, which the worker accepts).
template <size_t CH>
__attribute__((noinline)) void credit_worker(uint32_t returned, bool connected) {
    constexpr uint32_t source = SenderChannel<CH>::read_counter_addr;
    constexpr uint8_t cmd_buf = SenderChannel<CH>::producer_credit_cmd_buf;
    *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(source) = returned;
    if (connected) {
        if constexpr (worker_credit_preset) {
            ncrisc_noc_write_with_state<noc_mode, /*posted=*/true, /*update_counter=*/true, /*one_packet=*/true>(
                WORKER_CREDIT_NOC,
                cmd_buf,
                source,
                static_cast<uint32_t>(worker_interface<CH>().cached_worker_semaphore_address));
        } else {
            while (!noc_cmd_buf_ready(WORKER_CREDIT_NOC, cmd_buf)) {
            }
            ncrisc_noc_fast_write<noc_mode>(
                WORKER_CREDIT_NOC,
                cmd_buf,
                source,
                worker_interface<CH>().cached_worker_semaphore_address,
                sizeof(uint32_t),
                NOC_UNICAST_WRITE_VC,
                /*mcast=*/false,
                /*linked=*/false,
                /*num_dests=*/1,
                /*multicast_path_reserve=*/true,
                /*posted=*/true);
        }
    }
}

""" + s[b:]

rep("""// Worker channel CH's connection semaphore changed (open or close request): handle it out of line and return the value
// it holds while nothing changes. The interface's read counter is set to the credits returned so far. VC0's worker
// presets its credit command at connection (a no-op unless it owns the preset command buffer).
template <size_t CH, typename Interface>
__attribute__((noinline)) uint32_t worker_connection_event(
    Interface& interface, uint32_t semaphore, uint32_t expected, uint32_t returned) {
    using namespace tt::tt_fabric::connection_interface;
    interface.local_read_counter.counter = returned;
    if (expected == unused_connection_value) {
        if (connect_is_requested(semaphore)) {
            establish_worker_connection<MY_ETH_CHANNEL, ENABLE_RISC_CPU_DATA_CACHE>(interface);
            if constexpr (CH == 0) {
                preset_worker_credit(interface);
            }
            return open_connection_value;
        }""", """// Worker channel CH's connection semaphore changed (open or close request): handle it out of line and return the value
// it holds while nothing changes. The interface's read counter is set to the credits returned so far. Opening caches
// the worker's credit address, programs the channel's credit command buffer for it and sends the worker the credits
// returned so far, on the channel's own buffer (the library's connect would send them on the write buffer).
template <size_t CH>
__attribute__((noinline)) uint32_t worker_connection_event(uint32_t semaphore, uint32_t expected, uint32_t returned) {
    using namespace tt::tt_fabric::connection_interface;
    auto& interface = worker_interface<CH>();
    interface.local_read_counter.counter = returned;
    if (expected == unused_connection_value) {
        if (connect_is_requested(semaphore)) {
            interface.template cache_producer_noc_addr<ENABLE_RISC_CPU_DATA_CACHE, MY_ETH_CHANNEL>();
            preset_worker_credit<CH>();
            credit_worker<CH>(returned, true);
            return open_connection_value;
        }""")
rep("""template <size_t CH, typename Interface>
FORCE_INLINE bool check_connection(SenderChannelState<CH>& state, Interface& interface, uint32_t semaphore) {""",
    """template <size_t CH>
FORCE_INLINE bool check_connection(SenderChannelState<CH>& state, uint32_t semaphore) {""")
rep("""        state.connection_state = worker_connection_event<CH>(interface, semaphore, state.connection_state, returned);""",
    """        state.connection_state = worker_connection_event<CH>(semaphore, state.connection_state, returned);""")
rep("""        credit_worker<CH>(worker_interface<CH>(), returned, connected);""", """        credit_worker<CH>(returned, connected);""")

# the presets: every connected worker channel's, after the base firmware ran (at start and after a round's commands)
a = s.index("// Something else used the write command buffer")
b = s.index("\n}\n", a) + 3
s = s[:a] + """// The base firmware may have used the command buffers (router commands, the context switch): program each connected
// worker channel's credit command buffer again (a no-op where command buffers are not preset).
template <size_t VC, typename LoopCopy>
FORCE_INLINE void restore_worker_credit_preset(LoopCopy& loop_copy, SenderChannelStates& states) {
    if constexpr (worker_credit_preset && SenderVc<VC>::worker_channels != 0) {
        constexpr size_t CH = lowest_bit(SenderVc<VC>::worker_channels);
        if (states.template progress_for<VC>(loop_copy).worker.connection_state !=
            tt::tt_fabric::connection_interface::unused_connection_value) {
            preset_worker_credit<CH>();
        }
    }
}
template <typename LoopCopy>
FORCE_INLINE void restore_worker_credit_presets(LoopCopy& loop_copy, SenderChannelStates& states) {
    restore_worker_credit_preset<0>(loop_copy, states);
    restore_worker_credit_preset<1>(loop_copy, states);
    restore_worker_credit_preset<2>(loop_copy, states);
}
""" + s[b:]
rep("""// VC's worker channel's connection (every pass, by rule), from the semaphores the pass's read group read
// (`connection_semaphores[CH]`). A worker channel that does not own the credit preset may use the write command buffer
// on a connection event (its library connect), so the preset is restored after one.""",
    """// VC's worker channel's connection (every pass, by rule), from the semaphores the pass's read group read
// (`connection_semaphores[CH]`).""")
rep("""        if (check_connection<CH>(
                states.template progress_for<VC>(loop_copy).worker,
                worker_interface<CH>(),
                connection_semaphores[CH])) {
            if constexpr (!WorkerCredit<CH>::owns_write_buffer) {
                restore_worker_credit_preset(loop_copy, states);
            }
        }""", """        check_connection<CH>(states.template progress_for<VC>(loop_copy).worker, connection_semaphores[CH]);""")
s = s.replace("restore_worker_credit_preset(loop_copy, states);", "restore_worker_credit_presets(loop_copy, states);")
assert "WorkerCredit<" not in s
open(p, "w").write(s)

p = f"{dst}/tt_metal/fabric/hw/inc/edm_fabric/router/router_single.hpp"
t = open(p).read()
assert t.count("sender::restore_worker_credit_preset(") == 2
t = t.replace("sender::restore_worker_credit_preset(", "sender::restore_worker_credit_presets(")
open(p, "w").write(t)

p = f"{dst}/fabric_erisc_router_v4.cpp"
t = open(p).read()
o = """                worker && CH != 0 ? sender::WorkerCredit<CH>::source : notify_worker_of_read_counter_update_src_address);"""
assert t.count(o) == 1
t = t.replace(o, """                worker ? sender::worker_read_counter_addr(CH) : notify_worker_of_read_counter_update_src_address);""")
o = "// credits (a worker channel's: router_sender.hpp, WorkerCredit)."
if o in t:
    t = t.replace(o, "// credits (a worker channel's read counter: router_sender.hpp, credit_worker).")
open(p, "w").write(t)
print("made", dst)
