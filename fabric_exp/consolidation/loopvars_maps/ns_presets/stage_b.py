"""stage_b.py: v4rxnsb = v4rxg3 + partial presets on every forward (option b; V4_NS_PRESETS.md). Draft, offline only."""
import os, shutil, subprocess, sys
V = "/home/aagarwal/tt-metal/fabric_exp/variants"
dst = f"{V}/v4rxnsb"
if os.path.exists(dst):
    sys.exit(f"{dst} exists")
shutil.copytree(f"{V}/v4rxg3", dst, symlinks=True)
subprocess.run([sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "api_patch.py"), "v4rxnsb"], check=True)
p = f"{dst}/tt_metal/fabric/hw/inc/edm_fabric/router/router_downstream.hpp"
s = open(p).read()

def sub(o, n):
    global s
    assert s.count(o) == 1, o[:90]
    s = s.replace(o, n)

def cut(start, end_marker, new):
    """replace s[start : end of end_marker] with new"""
    global s
    a = s.index(start)
    b = s.index(end_marker, a) + len(end_marker)
    s = s[:a] + new + s[b:]

sub("""// Forwards go through the NoC API. Where VC0 has a single live downstream (E/W-facing 2D routers inside a mesh, 1D),
// its forwarding command buffers stay programmed with that downstream's coordinate for the whole run (stateful API),
// and another VC whose forwards use the same buffers and go to the same downstream forwards on that preset too; every
// other forward programs the destination per write (stateless API).""",
"""// Forwards go through the NoC API, on command buffers partially preset at init: the fields no forward changes stay
// programmed, and each forward writes only its own (forwards_use_presets).""")

cut("// Forward command-buffer presets. VC0's forwards keep their command buffers programmed",
    "constexpr bool forwards_set_credit_register = enable_deadlock_avoidance || another_vc_rides_vc0_preset();",
"""// Forward command-buffer presets (partial). Every forward to a router is two commands on its VC's forwarding command
// buffers (the host's RX_CH_<n>_FWD_DATA_ / FWD_SYNC_CMD_BUF_ID): the packet on the data buffer, then -1 on the
// destination's free-slots register (its doorbell) on the credit buffer. The fields no forward changes are preset at
// init (preset_downstream_cmd_bufs): the data write's control (posted, static VC) and high address word, the credit's
// control, value (-1), byte enables and high address word. Each forward writes only its own: the data write's
// transaction ID, source, destination node and address, and length; the credit's register and node. One shape serves
// one downstream or several, a downstream picked at run time (intermesh, edge) included, and VCs whose forwards share
// buffers share the preset (their invariant fields are the same).
//
// Not with DPRINT or the watcher, whose traffic shares the command buffers, nor on a one-ERISC router
// (command_buffer_presets), nor on a UDM router, whose relay forwards are stateless on the same data buffer: there
// every forward programs its whole commands (forward_stateless).

// The VCs whose forwards go to downstream routers here (DownstreamConnections).
constexpr bool vc_forwards_to_routers(size_t vc) { return vc == 0 || (vc == 1 && has_vc1); }

// A buffer is preset for one kind of command: VC's data and credit buffers differ, neither is VC0's other one, and
// neither is the local-write buffer or the atomics buffer (write_at_cmd_buf), whose commands program those fields.
template <size_t VC>
constexpr bool forward_cmd_bufs_consistent() {
    if constexpr (!vc_forwards_to_routers(VC)) {
        return true;
    } else {
        using T = DownstreamTraits<VC>;
        constexpr uint8_t local = tt::tt_fabric::local_chip_data_cmd_buf;
        return T::data_cmd_buf != T::sync_cmd_buf && T::data_cmd_buf != DownstreamTraits<0>::sync_cmd_buf &&
               T::sync_cmd_buf != DownstreamTraits<0>::data_cmd_buf && T::data_cmd_buf != local &&
               T::sync_cmd_buf != local && T::data_cmd_buf != write_at_cmd_buf && T::sync_cmd_buf != write_at_cmd_buf;
    }
}

constexpr bool forwards_use_presets =
#if !defined(DEBUG_PRINT_ENABLED) && !defined(WATCHER_ENABLED)
    command_buffer_presets && !FORCE_ALL_PATHS_TO_USE_SAME_NOC && !udm_mode && forward_cmd_bufs_consistent<0>() &&
    forward_cmd_bufs_consistent<1>();
#else
    false;
#endif""")

sub("""// UDM's relay: a ring of LOCAL_RELAY_NUM_BUFFERS slots in the relay's L1 that takes the whole packet, and a free-slots
// register the relay increments as it frees slots, of which this router keeps a local copy. Its forwards are stateless,
// as main's, on VC0's forward command buffer (UDM routers have VC0 only).""",
"""// A forward on the partially preset buffers (forwards_use_presets): the packet into `dst` on the node at `link`, tagged
// with `trid`, then -1 on that node's free-slots register `credit_register`. The two buffers' ready state is read as a
// pair, so the two reads overlap (RESULTS §17), and tested together: when either is busy (rare), both are waited for
// before either command, so no ready state is held across the data command. Only this ERISC issues on them, so a buffer
// seen ready stays ready until its next command.
template <uint8_t NOC, uint8_t DATA_CMD_BUF, uint8_t CREDIT_CMD_BUF>
FORCE_INLINE void forward_partially_preset(
    const DownstreamLink& link,
    uint32_t credit_register,
    uint32_t dst,
    uint32_t src_addr,
    uint32_t size_bytes,
    uint8_t trid) {
    // Read once: the register writes below store through uint32_t pointers, which may alias the field.
    const uint32_t node = link.noc_coord;
    const bool data_ready = noc_cmd_buf_ready(NOC, DATA_CMD_BUF);
    const bool credit_ready = noc_cmd_buf_ready(NOC, CREDIT_CMD_BUF);
    if (!(data_ready & credit_ready)) {
        while (!noc_cmd_buf_ready(NOC, DATA_CMD_BUF));
        while (!noc_cmd_buf_ready(NOC, CREDIT_CMD_BUF));
    }
    ncrisc_noc_set_transaction_id</*skip_cmdbuf_chk=*/true>(NOC, DATA_CMD_BUF, trid);
    noc_write_to_node_with_state<
        DM_DEDICATED_NOC,
        DATA_CMD_BUF,
        CQ_NOC_SNDL,
        CQ_NOC_SEND,
        CQ_NOC_wait,
        /*update_counter=*/is_blackhole_arch,
        /*posted=*/true>(NOC, src_addr, node, dst, size_bytes);
    noc_inline_dw_write_to_node_with_state</*update_counter=*/false, /*posted=*/true, /*skip_cmdbuf_chk=*/true>(
        node, credit_register, CREDIT_CMD_BUF, NOC);
}

// UDM's relay: a ring of LOCAL_RELAY_NUM_BUFFERS slots in the relay's L1 that takes the whole packet, and a free-slots
// register the relay increments as it frees slots, of which this router keeps a local copy. Its forwards are stateless,
// as main's, on VC0's forward command buffer (UDM routers have VC0 only).""")

cut("        if constexpr (forwards_are_stateful<VC>()) {\n#if defined(ARCH_BLACKHOLE)",
    "        } else {\n            forward_stateless<Traits::noc, Traits::data_cmd_buf>(links[entry], dst, src_addr, size_bytes, trid);\n        }",
"""        if constexpr (forwards_use_presets) {
            forward_partially_preset<Traits::noc, Traits::data_cmd_buf, Traits::sync_cmd_buf>(
                links[entry],
                get_stream_reg_write_addr<Traits::template remote_credit_stream<entry>>(),
                dst,
                src_addr,
                size_bytes,
                trid);
        } else {
            forward_stateless<Traits::noc, Traits::data_cmd_buf>(links[entry], dst, src_addr, size_bytes, trid);
        }""")

cut("// After the open: program VC0's forwarding command buffers with its one downstream, where it keeps them programmed",
    "\n}\n\n}  // namespace fabric_router",
"""// After the opens: preset the forwarding command buffers' invariant fields (forwards_use_presets), VC0's and, where its
// buffers differ, VC1's. The destination node in the presets is a placeholder: every forward writes its own.
template <uint8_t NOC, uint8_t DATA_CMD_BUF, uint8_t CREDIT_CMD_BUF>
__attribute__((noinline)) void preset_forward_cmd_bufs() {
    constexpr uint8_t noc_vc = tt::tt_fabric::forward_and_local_write_noc_vc;
    noc_async_write_one_packet_with_trid_set_state</*posted=*/true>(0, DATA_CMD_BUF, NOC, noc_vc);
    noc_inline_dw_write_set_state</*posted=*/true, /*set_val=*/true>(
        0, pack_value_for_inc_on_write_stream_reg_write(-1), INLINE_WRITE_ALL_BYTES, CREDIT_CMD_BUF, NOC, noc_vc);
}
// VC's buffers, where it forwards and they are not VC0's (already preset).
template <size_t VC>
FORCE_INLINE void preset_vc_forward_cmd_bufs() {
    if constexpr (vc_forwards_to_routers(VC)) {
        using T = DownstreamTraits<VC>;
        if constexpr (
            VC == 0 || T::data_cmd_buf != DownstreamTraits<0>::data_cmd_buf ||
            T::sync_cmd_buf != DownstreamTraits<0>::sync_cmd_buf) {
            preset_forward_cmd_bufs<T::noc, T::data_cmd_buf, T::sync_cmd_buf>();
        }
    }
}
template <size_t VC>
__attribute__((noinline)) void preset_downstream_cmd_bufs(const DownstreamArgs&) {
    static_assert(VC == 0, "presets every forwarding VC's buffers from VC0's call");
    if constexpr (forwards_use_presets) {
        preset_vc_forward_cmd_bufs<0>();
        preset_vc_forward_cmd_bufs<1>();
    }
}

}  // namespace fabric_router""")
sub("""    uint32_t noc_coord;        // the downstream's NoC coordinate (stateless forwards)
    uint64_t credit_noc_addr;  // the downstream's free-slots register (stateless credits)""",
"""    uint32_t noc_coord;        // the downstream's NoC coordinate (every forward but a full preset's)
    uint64_t credit_noc_addr;  // the downstream's free-slots register (stateless credits)""")
open(p, "w").write(s)
print("made", dst)
