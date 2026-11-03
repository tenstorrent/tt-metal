"""stage_f.py: v4rxnsf = v4rxg3 + the compile-time forward forms (V4_NS_PRESETS.md §9; user approval 2026-11-03):
ForwardCommands<VC> = StatelessForwards / FullPresetForwards / PartialPresetForwards, one interface (preset, forward),
chosen from forwards_can_preset and forwards_have_one_destination. The lead's nine preset predicates go."""
import os, shutil, subprocess, sys
V = "/home/aagarwal/tt-metal/fabric_exp/variants"
dst = f"{V}/v4rxnsf"
if os.path.exists(dst):
    sys.exit(f"{dst} exists")
shutil.copytree(f"{V}/v4rxg3", dst, symlinks=True)
subprocess.run([sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "api_patch.py"), "v4rxnsf"], check=True)
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


def take(start, end_marker):
    """remove s[start : end of end_marker] and return it"""
    global s
    a = s.index(start)
    b = s.index(end_marker, a) + len(end_marker)
    t = s[a:b]
    s = s[:a] + s[b:]
    return t


sub("""// Forwards go through the NoC API. Where VC0 has a single live downstream (E/W-facing 2D routers inside a mesh, 1D),
// its forwarding command buffers stay programmed with that downstream's coordinate for the whole run (stateful API),
// and another VC whose forwards use the same buffers and go to the same downstream forwards on that preset too; every
// other forward programs the destination per write (stateless API).""",
"""// Forwards go through the NoC API. How a VC's forwards use their command buffers is fixed at compile time
// (ForwardCommands): with one destination for the whole run the buffers keep it, with several they keep only what no
// forward changes, and without presets every forward writes its whole commands.""")

# The lead's nine preset predicates go; the selection facts take their place.
cut("// Forward command-buffer presets. VC0's forwards keep their command buffers programmed",
    "constexpr bool forwards_set_credit_register = enable_deadlock_avoidance || another_vc_rides_vc0_preset();",
"""// Forward command buffers. Every forward to a router is two commands on its VC's forwarding command buffers (the host's
// RX_CH_<n>_FWD_DATA_ / FWD_SYNC_CMD_BUF_ID): the packet on the data buffer, then -1 on the destination's free-slots
// register (its doorbell) on the credit buffer. Which fields stay programmed between forwards is decided at compile
// time, per VC (ForwardCommands, below):
//   - one destination for the whole run (forwards_have_one_destination): the buffers keep it (FullPresetForwards);
//   - several destinations: the buffers keep only what no forward changes (PartialPresetForwards);
//   - no presets (forwards_can_preset false): every forward writes its whole commands (StatelessForwards).

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

// Whether forwards may keep anything in their command buffers between forwards: not with DPRINT or the watcher, whose
// traffic shares the command buffers; only on two-ERISC routers (on one ERISC the forward data shares buffer 0 with the
// local writes); not on UDM routers, whose relay forwards are stateless on the same data buffer.
constexpr bool forwards_can_preset =
#if !defined(DEBUG_PRINT_ENABLED) && !defined(WATCHER_ENABLED)
    command_buffer_presets && !FORCE_ALL_PATHS_TO_USE_SAME_NOC && !udm_mode && forward_cmd_bufs_consistent<0>() &&
    forward_cmd_bufs_consistent<1>();
#else
    false;
#endif

// Whether VC's forwards can use VC0's destination preset: VC forwards to no router here, or it uses VC0's two buffers
// and goes to VC0's downstreams.
template <size_t VC>
constexpr bool can_share_vc0_destination() {
    if constexpr (!vc_forwards_to_routers(VC)) {
        return true;
    } else {
        using T = DownstreamTraits<VC>;
        return T::data_cmd_buf == DownstreamTraits<0>::data_cmd_buf &&
               T::sync_cmd_buf == DownstreamTraits<0>::sync_cmd_buf && live_downstreams<VC>() == live_downstreams<0>();
    }
}

// Whether every forward here goes to one destination for the whole run: VC0 has one live downstream (every 1D router;
// an E/W-facing 2D router inside a mesh), the router picks no downstream at run time (not intermesh, not on a mesh's
// edge), and VC1 shares VC0's destination. Otherwise every forwarding VC is partially preset, so no VC's commands
// overwrite another's preset.
constexpr bool forwards_have_one_destination = !is_intermesh_router && !is_intermesh_router_on_edge &&
                                               !is_intramesh_router_on_edge && popcount(live_downstreams<0>()) == 1 &&
                                               can_share_vc0_destination<1>();""")

# DownstreamArgs (and its coordinate helpers) move up: the forms' presets read them.
args_block = take("// The downstream connections of one VC as the runtime args describe them: which entries have a downstream (one bit",
                  """FORCE_INLINE uint32_t dense_noc_y(const DownstreamArgs& args, uint32_t dense) {
    return (args.noc_y_packed >> (dense * 8)) & 0xFF;
}
""")

full_block = '''
// ---------------------------------------------------------------------------------------------------------------------
// Forward command forms (one interface: preset(args) once after the opens, forward<ENTRY>(...) per packet)
// ---------------------------------------------------------------------------------------------------------------------

''' + args_block + '''
// No presets: every forward writes its whole commands.
template <size_t VC>
struct StatelessForwards {
    using Traits = DownstreamTraits<VC>;

    __attribute__((noinline)) static void preset(const DownstreamArgs&) {}

    template <size_t ENTRY>
    FORCE_INLINE static void forward(
        const DownstreamLink& link, uint32_t dst, uint32_t src_addr, uint32_t size_bytes, uint8_t trid) {
        forward_stateless<Traits::noc, Traits::data_cmd_buf>(link, dst, src_addr, size_bytes, trid);
    }
};

// One destination for the whole run: the buffers keep the downstream's coordinate and credit register, programmed
// once at init; each forward sets its ring slot, length and go (and VC1's forwards, sharing the preset, their credit
// register).
template <size_t VC>
struct FullPresetForwards {
    using Traits = DownstreamTraits<VC>;

    // After the opens: program the buffers with VC0's one live downstream.
    __attribute__((noinline)) static void preset(const DownstreamArgs& args) {
        constexpr size_t entry = downstream_entry(lowest_bit(live_downstreams<VC>()));
        if ((args.entries >> entry) & 1u) {
            constexpr uint8_t noc_vc = tt::tt_fabric::forward_and_local_write_noc_vc;
            const uint32_t dense = popcount(args.entries & ((1u << entry) - 1));  // entries below it with a downstream
            const uint32_t x = dense_noc_x(args, dense);
            const uint32_t y = dense_noc_y(args, dense);
            noc_async_write_one_packet_with_trid_set_state</*posted=*/true>(
                get_noc_addr(x, y, 0, Traits::noc), Traits::data_cmd_buf, Traits::noc, noc_vc);
            noc_inline_dw_write_set_state</*posted=*/true, /*set_val=*/true>(
                get_noc_addr(x, y, get_stream_reg_write_addr(Traits::template remote_credit_stream<entry>), Traits::noc),
                pack_value_for_inc_on_write_stream_reg_write(-1),
                INLINE_WRITE_ALL_BYTES,
                Traits::sync_cmd_buf,
                Traits::noc,
                noc_vc);
        }
    }

    template <size_t ENTRY>
    FORCE_INLINE static void forward(
        const DownstreamLink&, uint32_t dst, uint32_t src_addr, uint32_t size_bytes, uint8_t trid) {
#if defined(ARCH_BLACKHOLE)
        // The forward's two commands go on the data and credit command buffers: their ready state is read as a pair, so
        // the two reads overlap (RESULTS §17) instead of each being a lone read inside the NoC API, and a buffer not
        // ready yet is waited for just before its command, as the API would. Only this ERISC issues on them, so a
        // buffer seen ready stays ready until its next command.
        const bool data_ready = noc_cmd_buf_ready(Traits::noc, Traits::data_cmd_buf);
        const bool sync_ready = noc_cmd_buf_ready(Traits::noc, Traits::sync_cmd_buf);
        if (!data_ready) {
            while (!noc_cmd_buf_ready(Traits::noc, Traits::data_cmd_buf));
        }
        ncrisc_noc_set_transaction_id</*skip_cmdbuf_chk=*/true>(Traits::noc, Traits::data_cmd_buf, trid);
        ncrisc_noc_write_with_state<
            DM_DEDICATED_NOC,
            /*posted=*/true,
            /*update_counter=*/true,
            /*one_packet=*/false,
            /*skip_cmdbuf_chk=*/true>(Traits::noc, Traits::data_cmd_buf, src_addr, dst, size_bytes);
#else
        noc_async_write_one_packet_with_trid_with_state</*update_counter=*/false, /*posted=*/true>(
            src_addr, dst, size_bytes, trid, Traits::data_cmd_buf, Traits::noc);
#endif
        // The credit register, where it is set on every forward (credit_register_varies): one command-buffer register
        // write.
        if (!sync_ready) {
            while (!noc_cmd_buf_ready(Traits::noc, Traits::sync_cmd_buf));
        }
        noc_inline_dw_write_with_state<
            /*update_addr_lo=*/credit_register_varies,
            /*update_counter=*/false,
            /*posted=*/true,
            /*update_addr_hi=*/false,
            /*update_val=*/false,
            InlineWriteDst::REG,
            /*skip_cmdbuf_chk=*/true>(
            0, get_stream_reg_write_addr<Traits::template remote_credit_stream<ENTRY>>(), Traits::sync_cmd_buf, Traits::noc);
    }

private:
    // Whether a forward sets its credit register: where VC1 shares the preset (its credit registers differ), and on
    // deadlock-avoiding routers, where main re-programs it on every forward (the same address as preset; main gives no
    // reason).
    static constexpr bool credit_register_varies = enable_deadlock_avoidance || vc_forwards_to_routers(1);
};

// Several destinations: the buffers keep the fields no forward changes, preset at init: the data write's control
// (posted, static VC) and high address word, the credit's control, value (-1), byte enables and high address word. Each
// forward writes only its own: the data write's transaction ID, source, destination node and address, and length; the
// credit's register and node. VCs whose forwards share buffers share the preset (their invariant fields are the same).
template <size_t VC>
struct PartialPresetForwards {
    using Traits = DownstreamTraits<VC>;

    // After the opens: preset every forwarding VC's buffers, VC0's and, where its buffers differ, VC1's. The
    // destination node in the presets is a placeholder: every forward writes its own.
    __attribute__((noinline)) static void preset(const DownstreamArgs&) {
        static_assert(VC == 0, "presets every forwarding VC's buffers from VC0's call");
        preset_vc<0>();
        preset_vc<1>();
    }

    // The packet into `dst` on the node at `link`, tagged with `trid`, then -1 on that node's free-slots register. The
    // two buffers' ready state is read as a pair, so the two reads overlap (RESULTS §17), and tested together: when
    // either is busy (rare), both are waited for before either command, so no ready state is held across the data
    // command. Only this ERISC issues on them, so a buffer seen ready stays ready until its next command.
    template <size_t ENTRY>
    FORCE_INLINE static void forward(
        const DownstreamLink& link, uint32_t dst, uint32_t src_addr, uint32_t size_bytes, uint8_t trid) {
        // Read once: the register writes below store through uint32_t pointers, which may alias the field.
        const uint32_t node = link.noc_coord;
        const bool data_ready = noc_cmd_buf_ready(Traits::noc, Traits::data_cmd_buf);
        const bool credit_ready = noc_cmd_buf_ready(Traits::noc, Traits::sync_cmd_buf);
        if (!(data_ready & credit_ready)) {
            while (!noc_cmd_buf_ready(Traits::noc, Traits::data_cmd_buf));
            while (!noc_cmd_buf_ready(Traits::noc, Traits::sync_cmd_buf));
        }
        ncrisc_noc_set_transaction_id</*skip_cmdbuf_chk=*/true>(Traits::noc, Traits::data_cmd_buf, trid);
        noc_write_to_node_with_state<
            DM_DEDICATED_NOC,
            Traits::data_cmd_buf,
            CQ_NOC_SNDL,
            CQ_NOC_SEND,
            CQ_NOC_wait,
            /*update_counter=*/is_blackhole_arch,
            /*posted=*/true>(Traits::noc, src_addr, node, dst, size_bytes);
        noc_inline_dw_write_to_node_with_state</*update_counter=*/false, /*posted=*/true, /*skip_cmdbuf_chk=*/true>(
            node,
            get_stream_reg_write_addr<Traits::template remote_credit_stream<ENTRY>>(),
            Traits::sync_cmd_buf,
            Traits::noc);
    }

private:
    template <uint8_t NOC, uint8_t DATA_CMD_BUF, uint8_t CREDIT_CMD_BUF>
    __attribute__((noinline)) static void preset_cmd_bufs() {
        constexpr uint8_t noc_vc = tt::tt_fabric::forward_and_local_write_noc_vc;
        noc_async_write_one_packet_with_trid_set_state</*posted=*/true>(0, DATA_CMD_BUF, NOC, noc_vc);
        noc_inline_dw_write_set_state</*posted=*/true, /*set_val=*/true>(
            0, pack_value_for_inc_on_write_stream_reg_write(-1), INLINE_WRITE_ALL_BYTES, CREDIT_CMD_BUF, NOC, noc_vc);
    }
    // FV's buffers, where FV forwards and they are not VC0's (already preset).
    template <size_t FV>
    FORCE_INLINE static void preset_vc() {
        if constexpr (vc_forwards_to_routers(FV)) {
            using T = DownstreamTraits<FV>;
            if constexpr (
                FV == 0 || T::data_cmd_buf != DownstreamTraits<0>::data_cmd_buf ||
                T::sync_cmd_buf != DownstreamTraits<0>::sync_cmd_buf) {
                preset_cmd_bufs<T::noc, T::data_cmd_buf, T::sync_cmd_buf>();
            }
        }
    }
};

// How VC's forwards use their command buffers, chosen at compile time.
template <size_t VC>
using ForwardCommands = std::conditional_t<
    !forwards_can_preset,
    StatelessForwards<VC>,
    std::conditional_t<forwards_have_one_destination, FullPresetForwards<VC>, PartialPresetForwards<VC>>>;
'''

# The forms go right after the stateless forward's helper (before UDM's relay).
sub("""// UDM's relay: a ring of LOCAL_RELAY_NUM_BUFFERS slots in the relay's L1""",
    full_block.lstrip("\n") + """
// UDM's relay: a ring of LOCAL_RELAY_NUM_BUFFERS slots in the relay's L1""")

# Downstreams<VC>::forward: one call.
cut("        if constexpr (forwards_are_stateful<VC>()) {\n#if defined(ARCH_BLACKHOLE)",
    "        } else {\n            forward_stateless<Traits::noc, Traits::data_cmd_buf>(links[entry], dst, src_addr, size_bytes, trid);\n        }",
    "        ForwardCommands<VC>::template forward<entry>(links[entry], dst, src_addr, size_bytes, trid);")

# The lead's preset function goes (FullPresetForwards::preset holds its body).
cut("// After the open: program VC0's forwarding command buffers with its one downstream, where it keeps them programmed",
    "\n}\n\n}  // namespace fabric_router",
    "}  // namespace fabric_router")
sub("""    uint32_t noc_coord;        // the downstream's NoC coordinate (stateless forwards)""",
    """    uint32_t noc_coord;        // the downstream's NoC coordinate (stateless and partially preset forwards)""")
open(p, "w").write(s)

# The init call site.
q = f"{dst}/fabric_erisc_router_v4.cpp"
t = open(q).read()
o = "        preset_downstream_cmd_bufs<0>(args.vc0());"
assert t.count(o) == 1
open(q, "w").write(t.replace(o, "        ForwardCommands<0>::preset(args.vc0());"))
print("made", dst)
