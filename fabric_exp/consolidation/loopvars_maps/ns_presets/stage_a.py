"""stage_a.py: v4rxnsa = v4rxg3 + partial presets only where the lead keeps no destination preset (option a;
V4_NS_PRESETS.md). Single-downstream routers keep the lead's full presets verbatim. Draft, offline only."""
import os, shutil, subprocess, sys
V = "/home/aagarwal/tt-metal/fabric_exp/variants"
dst = f"{V}/v4rxnsa"
if os.path.exists(dst):
    sys.exit(f"{dst} exists")
shutil.copytree(f"{V}/v4rxg3", dst, symlinks=True)
subprocess.run([sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "api_patch.py"), "v4rxnsa"], check=True)
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
"""// Forwards go through the NoC API, on preset command buffers. Where VC0 has a single live downstream (E/W-facing 2D
// routers inside a mesh, 1D), its forwarding command buffers stay programmed with that downstream's coordinate for the
// whole run (full presets), and another VC whose forwards use the same buffers and go to the same downstream forwards
// on that preset too. Elsewhere the buffers keep only the fields no forward changes, and each forward writes its
// destination (partial presets).""")

sub("""// whose exits pick a downstream at run time.
//
// Another VC's forwards may share VC0's command buffers""",
"""// whose exits pick a downstream at run time: those keep partial presets (forwards_partially_preset).
//
// Another VC's forwards may share VC0's command buffers""")

sub("""// the preset would overwrite it, so VC0 then keeps none: every forward is stateless.
constexpr bool command_buffers_can_keep_presets =
#if !defined(DEBUG_PRINT_ENABLED) && !defined(WATCHER_ENABLED)
    command_buffer_presets && !FORCE_ALL_PATHS_TO_USE_SAME_NOC && !udm_mode && !is_intermesh_router &&
    !is_intermesh_router_on_edge && !is_intramesh_router_on_edge;
#else
    false;
#endif""",
"""// the preset would overwrite it, so VC0 then keeps none: every forward is partially preset.
//
// Forwards preset their command buffers at all only without DPRINT or the watcher, whose traffic shares the command
// buffers, on two-ERISC routers (command_buffer_presets), and not on UDM routers, whose relay forwards are stateless on
// the same data buffer. Elsewhere every forward programs its whole commands (forward_stateless).
constexpr bool forward_cmd_bufs_can_be_preset =
#if !defined(DEBUG_PRINT_ENABLED) && !defined(WATCHER_ENABLED)
    command_buffer_presets && !FORCE_ALL_PATHS_TO_USE_SAME_NOC && !udm_mode;
#else
    false;
#endif
constexpr bool command_buffers_can_keep_presets = forward_cmd_bufs_can_be_preset && !is_intermesh_router &&
                                                  !is_intermesh_router_on_edge && !is_intramesh_router_on_edge;""")

sub("""constexpr bool forwards_set_credit_register = enable_deadlock_avoidance || another_vc_rides_vc0_preset();""",
"""constexpr bool forwards_set_credit_register = enable_deadlock_avoidance || another_vc_rides_vc0_preset();

// Partial presets, where VC0 keeps no destination preset. Every forward to a router is two commands on its VC's
// forwarding command buffers: the packet on the data buffer, then -1 on the destination's free-slots register (its
// doorbell) on the credit buffer. The fields no forward changes are preset at init (preset_downstream_cmd_bufs): the
// data write's control (posted, static VC) and high address word, the credit's control, value (-1), byte enables and
// high address word. Each forward writes only its own: the data write's transaction ID, source, destination node and
// address, and length; the credit's register and node. VCs whose forwards share buffers share the preset (their
// invariant fields are the same).
//
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
constexpr bool forwards_partially_preset = forward_cmd_bufs_can_be_preset && !vc0_keeps_forward_preset() &&
                                           forward_cmd_bufs_consistent<0>() && forward_cmd_bufs_consistent<1>();""")

sub("""// UDM's relay: a ring of LOCAL_RELAY_NUM_BUFFERS slots in the relay's L1 that takes the whole packet, and a free-slots
// register the relay increments as it frees slots, of which this router keeps a local copy. Its forwards are stateless,
// as main's, on VC0's forward command buffer (UDM routers have VC0 only).""",
"""// A forward on the partially preset buffers (forwards_partially_preset): the packet into `dst` on the node at `link`,
// tagged with `trid`, then -1 on that node's free-slots register `credit_register`. The two buffers' ready state is
// read as a pair, so the two reads overlap (RESULTS §17), and tested together: when either is busy (rare), both are
// waited for before either command, so no ready state is held across the data command. Only this ERISC issues on them,
// so a buffer seen ready stays ready until its next command.
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

sub("""                0, get_stream_reg_write_addr<Traits::template remote_credit_stream<entry>>(), Traits::sync_cmd_buf, Traits::noc);
        } else {
            forward_stateless<Traits::noc, Traits::data_cmd_buf>(links[entry], dst, src_addr, size_bytes, trid);
        }""",
"""                0, get_stream_reg_write_addr<Traits::template remote_credit_stream<entry>>(), Traits::sync_cmd_buf, Traits::noc);
        } else if constexpr (forwards_partially_preset) {
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

sub("""// After the open: program VC0's forwarding command buffers with its one downstream, where it keeps them programmed
// (vc0_keeps_forward_preset; VCs that forward on the preset set their own ring slot and credit register).
template <size_t VC>""",
"""// Partial presets (forwards_partially_preset): the invariant fields of a VC's forwarding command buffers. The destination
// node in them is a placeholder: every forward writes its own.
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

// After the opens: program VC0's forwarding command buffers with its one downstream, where it keeps them programmed
// (vc0_keeps_forward_preset; VCs that forward on the preset set their own ring slot and credit register); elsewhere preset
// every forwarding VC's buffers partially, from VC0's call.
template <size_t VC>""")

sub("""                Traits::sync_cmd_buf,
                Traits::noc,
                noc_vc);
        }
    }
}""",
"""                Traits::sync_cmd_buf,
                Traits::noc,
                noc_vc);
        }
    } else if constexpr (forwards_partially_preset) {
        static_assert(VC == 0, "presets every forwarding VC's buffers from VC0's call");
        preset_vc_forward_cmd_bufs<0>();
        preset_vc_forward_cmd_bufs<1>();
    }
}""")
sub("""    uint32_t noc_coord;        // the downstream's NoC coordinate (stateless forwards)
    uint64_t credit_noc_addr;  // the downstream's free-slots register (stateless credits)""",
"""    uint32_t noc_coord;        // the downstream's NoC coordinate (every forward but a full preset's)
    uint64_t credit_noc_addr;  // the downstream's free-slots register (stateless credits)""")
open(p, "w").write(s)
print("made", dst)
