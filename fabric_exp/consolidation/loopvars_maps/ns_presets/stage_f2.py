"""stage_f2.py: v4rxnsf2 = v4rxnsf with trim 1 (user, 2026-11-03): PartialPresetForwards presets VC0's pair only, with a
static_assert that VC1 is absent or on VC0's buffers (the host's buffer ids own that), and a helper
uses_vc0_forward_cmd_bufs<VC> shared with can_share_vc0_destination."""
import os, shutil, sys
V = "/home/aagarwal/tt-metal/fabric_exp/variants"
dst = f"{V}/v4rxnsf2"
if os.path.exists(dst):
    sys.exit(f"{dst} exists")
shutil.copytree(f"{V}/v4rxnsf", dst, symlinks=True)
p = f"{dst}/tt_metal/fabric/hw/inc/edm_fabric/router/router_downstream.hpp"
s = open(p).read()


def sub(o, n):
    global s
    assert s.count(o) == 1, o[:90]
    s = s.replace(o, n)


sub("""// Whether VC's forwards can use VC0's destination preset: VC forwards to no router here, or it uses VC0's two buffers
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
}""",
"""// Whether VC's forwards use VC0's two forwarding buffers (true where VC forwards to no router here).
template <size_t VC>
constexpr bool uses_vc0_forward_cmd_bufs() {
    if constexpr (!vc_forwards_to_routers(VC)) {
        return true;
    } else {
        return DownstreamTraits<VC>::data_cmd_buf == DownstreamTraits<0>::data_cmd_buf &&
               DownstreamTraits<VC>::sync_cmd_buf == DownstreamTraits<0>::sync_cmd_buf;
    }
}

// Whether VC's forwards can use VC0's destination preset: VC forwards to no router here, or it uses VC0's two buffers
// and goes to VC0's downstreams.
template <size_t VC>
constexpr bool can_share_vc0_destination() {
    if constexpr (!vc_forwards_to_routers(VC)) {
        return true;
    } else {
        return uses_vc0_forward_cmd_bufs<VC>() && live_downstreams<VC>() == live_downstreams<0>();
    }
}""")

sub("""    // After the opens: preset every forwarding VC's buffers, VC0's and, where its buffers differ, VC1's. The
    // destination node in the presets is a placeholder: every forward writes its own.
    __attribute__((noinline)) static void preset(const DownstreamArgs&) {
        static_assert(VC == 0, "presets every forwarding VC's buffers from VC0's call");
        preset_vc<0>();
        preset_vc<1>();
    }""",
"""    // After the opens: preset the forwarding buffers' invariant fields. The destination node in the preset is a
    // placeholder: every forward writes its own.
    __attribute__((noinline)) static void preset(const DownstreamArgs&) {
        static_assert(VC == 0, "presets every forwarding VC's buffers from VC0's call");
        // The host's buffer ids (RX_CH_<n>_FWD_DATA_ / FWD_SYNC_CMD_BUF_ID) put every VC's forwards on VC0's two
        // buffers, so one preset serves them all; a VC given buffers of its own would be preset here.
        static_assert(uses_vc0_forward_cmd_bufs<1>(), "VC1's forwards share VC0's forwarding command buffers");
        preset_cmd_bufs();
    }""")

sub("""private:
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
};""",
"""private:
    __attribute__((noinline)) static void preset_cmd_bufs() {
        constexpr uint8_t noc_vc = tt::tt_fabric::forward_and_local_write_noc_vc;
        noc_async_write_one_packet_with_trid_set_state</*posted=*/true>(0, Traits::data_cmd_buf, Traits::noc, noc_vc);
        noc_inline_dw_write_set_state</*posted=*/true, /*set_val=*/true>(
            0,
            pack_value_for_inc_on_write_stream_reg_write(-1),
            INLINE_WRITE_ALL_BYTES,
            Traits::sync_cmd_buf,
            Traits::noc,
            noc_vc);
    }
};""")
open(p, "w").write(s)
print("made", dst)
