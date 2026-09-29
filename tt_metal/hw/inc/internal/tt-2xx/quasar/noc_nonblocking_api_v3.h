// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Quasar NOC non-blocking API V3: the ATT-native transport.
//
// V3 exists only for the ATT (Address Translation Table) address backend.
// Under ATT every NoC operand is one opaque translated 64-bit address written
// whole into the command buffer's address register; the coordinate registers
// of the XY encoding are never programmed, and no function here decomposes an
// address.
//
// Local addresses stay 32-bit in the API (same signatures as V2): because the
// active ATT map is required to provide a local (self) window - source
// translation is hardwired on, so a map without one cannot support an
// initiator at all - V3 assembles the full local operand itself:
//
//   local operand = NOC_ATT_LOCAL_WINDOW_BASE | local_address
//
// NOC_ATT_LOCAL_WINDOW_BASE comes from the selected ATT configuration (0x0 on
// maps whose first mask entry is a pass-through local window; the per-tile
// config-window base on the QSR1 boot map). The window's per-tile endpoint is
// boot-patched to the initiating tile, so the same operand value is correct on
// every core.
//
// The stateful pairs keep V2's signatures and per-issue register count:
// set_state latches a software-held 64-bit base whose offset field must be
// zero (get_noc_addr(x, y, 0) produces one); each with_state issue writes
// base | local as one full-operand register write.
//
// Not provided (compile-time rejected or absent):
//  - the CQ flag-based stateful family (dispatch kernels stay on V2 until
//    their dedicated conversion);
//  - the coordinate-patching inline-write variant (update_addr_hi): its
//    contract writes the XY coordinate register and cannot be expressed over
//    ATT operands;
//  - every multicast (write, inline write, atomic increment) decodes its
//    worker rectangle through the active map below. Regular writes and
//    atomics program the start operand in DEST_ADDR plus the width/height
//    extent in DEST_COORD: their destination travels in the packet's
//    return-address field, which the NOC translation stage
//    (tt_noc_att_and_routing.sv) turns into start = translated tile,
//    end = start + extent - 1. Inline writes carry their destination in the
//    target-address field instead, which that stage translates WITHOUT the
//    rectangle math (the end coordinate becomes the translated tile, the
//    start coordinate fields pass through), so they program the END tile's
//    operand in DEST_ADDR and the rectangle's NOC coordinates in DEST_COORD.
//    MCAST_DESTS carries the caller's destination count on both forms.
//
// Shared RoCC command-buffer definitions (register wrappers, MISC/VC values,
// counters, init, barriers) come from noc_cmd_buf_common.h.

#if !defined(NOC_ATT_ENABLED)
#error "NOC API V3 is the ATT-native transport and requires the ATT address backend (NOC_ATT_ENABLED)"
#endif

// Selects the active map configuration and defines NOC_ATT_LOCAL_WINDOW_BASE.
#include "internal/tt-2xx/quasar/noc/att/att_config.h"

#if !defined(NOC_ATT_LOCAL_WINDOW_BASE)
#error "The selected ATT configuration must define NOC_ATT_LOCAL_WINDOW_BASE (the self window base address)"
#endif

#include "internal/tt-2xx/quasar/noc_cmd_buf_common.h"
#if defined(ATT_PROGRAM_FOR_TEST)
#include "internal/tt-2xx/quasar/noc/att/temporary_programming/att_program.h"
#include "internal/tt-2xx/quasar/noc/att/temporary_programming/att_program_data.h"
#endif

// The full ATT operand for an address in this initiator's own L1.
inline __attribute__((always_inline)) constexpr uint64_t noc_v3_local_operand(uint32_t local_address) {
    return (uint64_t{NOC_ATT_LOCAL_WINDOW_BASE}) | local_address;
}

// DEST_COORD for an inline multicast: the rectangle's NOC coordinates in the
// XY register layout (the start fields survive translation, the end fields are
// replaced by the translated DEST_ADDR tile); see the header note.
inline __attribute__((always_inline)) uint32_t noc_v3_inline_mcast_coord(const noc_att::NocMulticastAddress& target) {
    return NOC_MULTICAST_COORD(
        target.start_node_xy & 0x3f, target.start_node_xy >> 6, target.end_node_xy & 0x3f, target.end_node_xy >> 6);
}

// with_state issue operand: the set_state base (offset field zero by contract)
// combined with the per-issue local address.
inline __attribute__((always_inline)) constexpr uint64_t noc_v3_state_operand(
    uint64_t state_base, uint32_t local_address) {
    return state_base | local_address;
}

// Software-held stateful operand bases (set_state / with_state pairs), one per
// command buffer to match V2's per-buffer hardware coordinate latch: states on
// different buffers are independent.
//
// Contract differences from V2's latch, enforced below:
//  - The state base's local-address field must be ZERO (e.g. from
//    get_noc_addr(x, y, 0)). V2 silently dropped the local bits of the state
//    address; V3 folds base | local per issue, so stale base bits would merge
//    into every operand. Passing a nonzero offset in the state address is a
//    rejected legacy pattern - re-supply the full local address per issue,
//    exactly as V2 callers already do.
//  - The state lives in this program's memory, not in a hardware register:
//    set_state and its with_state issues must run in the same binary
//    (firmware state is invisible to kernels and vice versa). No current
//    caller crosses that line; V2 could, V3 cannot.
inline constexpr uint32_t NOC_V3_STATE_CMD_BUFS = 3;
inline uint64_t noc_v3_read_state_base[NOC_V3_STATE_CMD_BUFS] = {};
inline uint64_t noc_v3_write_state_base[NOC_V3_STATE_CMD_BUFS] = {};
// The inline-write pair runs on the simple command buffer only, so one base.
inline uint64_t noc_v3_inline_write_state_base = 0;
// The value a set_state<set_val> call latches for with_state<update_val =
// false> issues: the RoCC inline write takes its data from the issue
// instruction, not from a sticky register, so the reuse contract of the
// stateful pair is kept in software.
inline uint32_t noc_v3_inline_write_state_val = 0;

// Reduce a state address to its base, reproducing V2's latch semantics
// exactly: V2 kept only the coordinate bits of the state address and dropped
// its local bits, with every with_state issue re-supplying the full local
// address. The active map tells us the matched window's local-field width, so
// callers that pass a state address with a nonzero offset (the existing
// dataflow wrappers do) behave identically to V2 instead of leaking stale
// base bits into base | local.
inline __attribute__((always_inline)) uint64_t noc_v3_state_base_of(uint64_t noc_addr) {
    const noc_att::WindowClass window_class = noc_att::matching_window_class(ACTIVE_ATT_MAP, noc_addr);
    if (window_class != noc_att::WindowClass::Invalid) {
        const noc_att::Window& window = noc_att::map_window(ACTIVE_ATT_MAP, window_class);
        return noc_addr & ~noc_att::low_mask(window.local_address_bits());
    }
    // Matches no window: keep it whole; the issue will fault like any other
    // invalid operand.
    return noc_addr;
}

inline __attribute__((always_inline)) void noc_init(uint32_t atomic_ret_val) {
    // The command buffers are programmed by overlay_cmd_buff_init.
#if defined(ATT_PROGRAM_FOR_TEST)
    // Emulator bring-up only: boot has not programmed the ATT tables, so
    // firmware replays the generated image once before any traffic.
    // Production boot/UMD owns this before DM startup.
    noc_att::program_for_test(active_att_program::PROGRAM_IMAGE);
    ASSERT(noc_att::check_no_faults());
#endif
}

// Point atomic return values back at the default slot after a caller redirected
// them (noc_fast_atomic_cas4 below). The slot is in this initiator's own L1, so
// it is issued as the local-window operand like every other V3 source address.
inline __attribute__((always_inline)) void noc_restore_default_atomic_ret_addr(uint32_t atomic_ret_val) {
    __builtin_riscv_ttrocc_scmdbuf_wr_reg(
        TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_SRC_ADDR_REG_OFFSET / 8, noc_v3_local_operand(atomic_ret_val));
}

// ============================================================================
// Stateless issues
// ============================================================================

// Expects overlay_cmd_buff_init to have set on OVERLAY_RD_CMD_BUF:
//   MISC = CMD_BUF_MISC_READ
//   TR_ID / WR_SENT_TR_ID / TR_ACK_TR_ID = NOC_OVERLAY_TRID_STATIC (0)
template <uint8_t noc_mode = DM_DEDICATED_NOC>
inline __attribute__((always_inline)) void ncrisc_noc_fast_read(
    uint32_t noc,
    uint32_t cmd_buf,
    uint64_t src_addr,
    uint32_t dest_addr,
    uint32_t len_bytes,
    uint32_t read_req_vc = NOC_OVERLAY_RD_REQ_VC) {
    static_assert(noc_mode != DM_DYNAMIC_NOC, "Quasar does not support DYNAMIC_NOC as it has only 1 NOC");

    __builtin_riscv_ttrocc_cmdbuf_wr_reg(
        cmd_buf, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_REQ_VC_REG_OFFSET / 8, read_req_vc);
    __builtin_riscv_ttrocc_cmdbuf_wr_reg(
        cmd_buf, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_RESP_VC_REG_OFFSET / 8, NOC_OVERLAY_RD_RESP_VC);
    __builtin_riscv_ttrocc_cmdbuf_wr_reg(
        cmd_buf, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_SRC_ADDR_REG_OFFSET / 8, src_addr);
    __builtin_riscv_ttrocc_cmdbuf_wr_reg(
        cmd_buf, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_DEST_ADDR_REG_OFFSET / 8, noc_v3_local_operand(dest_addr));
    __builtin_riscv_ttrocc_cmdbuf_wr_reg(
        cmd_buf, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_LEN_BYTES_REG_OFFSET / 8, len_bytes);
    __builtin_riscv_ttrocc_cmdbuf_issue_trans(cmd_buf);

    uint32_t num_packets =
        len_bytes / NOC_OVERLAY_MAX_BYTES_IN_PACKET + ((len_bytes % NOC_OVERLAY_MAX_BYTES_IN_PACKET) ? 1 : 0);
    noc_reads_num_issued[noc] += num_packets;
}

template <uint8_t noc_mode = DM_DEDICATED_NOC, bool use_vc = false>
inline __attribute__((always_inline)) void ncrisc_noc_fast_read_any_len(
    uint32_t noc,
    uint32_t cmd_buf,
    uint64_t src_addr,
    uint32_t dest_addr,
    uint32_t len_bytes,
    uint32_t read_req_vc = NOC_OVERLAY_RD_REQ_VC) {
    static_assert(noc_mode != DM_DYNAMIC_NOC, "Quasar does not support DYNAMIC_NOC as it has only 1 NOC");
    // Overlay handles packetization via MAX_BYTES_IN_PACKET register; no software chunking needed.
    ncrisc_noc_fast_read<noc_mode>(noc, cmd_buf, src_addr, dest_addr, len_bytes, read_req_vc);
}

template <uint8_t noc_mode = DM_DEDICATED_NOC, bool use_trid = false, bool update_counter = true>
inline __attribute__((always_inline)) void ncrisc_noc_fast_write(
    uint32_t noc,
    uint32_t cmd_buf,
    uint32_t src_addr,
    uint64_t dest_addr,
    uint32_t len_bytes,
    uint32_t vc,
    bool mcast,
    bool linked,
    uint32_t num_dests,
    bool multicast_path_reserve,
    bool posted = false,
    uint32_t trid = 0) {
    static_assert(noc_mode != DM_DYNAMIC_NOC, "Quasar does not support DYNAMIC_NOC as it has only 1 NOC");
    // A multicast dest_addr is the packed software rectangle descriptor;
    // decode it through the active map into the flat start operand plus the
    // width/height extent the DEST_COORD register carries.
    noc_att::NocMulticastAddress mcast_target{};
    if (mcast) {
        mcast_target = noc_att::resolve_worker_multicast<ACTIVE_ATT_MAP>(dest_addr, len_bytes);
        if (mcast_target.rectangle_count == 0) {
            // Invalid descriptor: trap unconditionally (ASSERT is a no-op
            // outside watcher/lightweight-assert builds) rather than issue a
            // transaction with an unresolved operand.
            __builtin_trap();
        }
        // The sender is not included on this path, so a rectangle containing it
        // has one destination fewer than its area; catch gross mismatches.
        ASSERT(num_dests <= mcast_target.rectangle_count);
        dest_addr = mcast_target.start_address;
    }

    // Rebuild MISC per-transaction since mcast/linked/posted can change.
    uint64_t misc = CMD_BUF_MISC_WRITE_TRANS | (linked ? CMD_BUF_MISC_LINKED : 0) |
                    (mcast ? CMD_BUF_MISC_MULTICAST : 0) | (posted ? CMD_BUF_MISC_POSTED : 0);
    __builtin_riscv_ttrocc_cmdbuf_wr_reg(cmd_buf, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_MISC_REG_OFFSET / 8, misc);

    __builtin_riscv_ttrocc_cmdbuf_wr_reg(cmd_buf, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_REQ_VC_REG_OFFSET / 8, vc);
    __builtin_riscv_ttrocc_cmdbuf_wr_reg(
        cmd_buf,
        TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_RESP_VC_REG_OFFSET / 8,
        mcast ? NOC_OVERLAY_MCAST_RESP_VC : NOC_OVERLAY_WR_RESP_VC);

    if constexpr (use_trid) {
        __builtin_riscv_ttrocc_cmdbuf_wr_reg(cmd_buf, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_TR_ID_REG_OFFSET / 8, trid);
    }

    __builtin_riscv_ttrocc_cmdbuf_wr_reg(
        cmd_buf, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_SRC_ADDR_REG_OFFSET / 8, noc_v3_local_operand(src_addr));
    __builtin_riscv_ttrocc_cmdbuf_wr_reg(
        cmd_buf, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_DEST_ADDR_REG_OFFSET / 8, dest_addr);
    __builtin_riscv_ttrocc_cmdbuf_wr_reg(
        cmd_buf, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_LEN_BYTES_REG_OFFSET / 8, len_bytes);
    if (mcast) {
        __builtin_riscv_ttrocc_cmdbuf_wr_reg(
            cmd_buf, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_DEST_COORD_REG_OFFSET / 8, mcast_target.extent_xy);
        // HW needs MCAST_DESTS to match the number of cores in the rectangle
        // so it can track per-destination acks for the multicast.
        __builtin_riscv_ttrocc_cmdbuf_wr_reg(
            cmd_buf, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_MCAST_DESTS_REG_OFFSET / 8, num_dests);
    }
    __builtin_riscv_ttrocc_cmdbuf_issue_trans(cmd_buf);

    if constexpr (update_counter) {
        uint32_t num_packets =
            len_bytes / NOC_OVERLAY_MAX_BYTES_IN_PACKET + ((len_bytes % NOC_OVERLAY_MAX_BYTES_IN_PACKET) ? 1 : 0);
        if (posted) {
            noc_posted_writes_num_issued[noc] += num_packets;
        } else {
            noc_nonposted_writes_num_issued[noc] += num_packets;
            noc_nonposted_writes_acked[noc] += num_dests * num_packets;
        }
    }
}

template <uint8_t noc_mode = DM_DEDICATED_NOC>
inline __attribute__((always_inline)) void ncrisc_noc_fast_write_loopback_src(
    uint32_t noc,
    uint32_t cmd_buf,
    uint32_t src_addr,
    uint64_t dest_addr,
    uint32_t len_bytes,
    uint32_t vc,
    bool mcast,
    bool linked,
    uint32_t num_dests,
    bool multicast_path_reserve) {
    static_assert(noc_mode != DM_DYNAMIC_NOC, "Quasar does not support DYNAMIC_NOC as it has only 1 NOC");
    // A multicast dest_addr is the packed software rectangle descriptor; see
    // ncrisc_noc_fast_write.
    noc_att::NocMulticastAddress mcast_target{};
    if (mcast) {
        mcast_target = noc_att::resolve_worker_multicast<ACTIVE_ATT_MAP>(dest_addr, len_bytes);
        if (mcast_target.rectangle_count == 0) {
            // Invalid descriptor: trap unconditionally (ASSERT is a no-op
            // outside watcher/lightweight-assert builds) rather than issue a
            // transaction with an unresolved operand.
            __builtin_trap();
        }
        // HW needs MCAST_DESTS to match the rectangle; catch caller/descriptor
        // mismatches in checked builds.
        ASSERT(num_dests == mcast_target.rectangle_count);
        dest_addr = mcast_target.start_address;
    }

    // Always nonposted, always src_include (loopback)
    uint64_t misc = CMD_BUF_MISC_WRITE_TRANS | CMD_BUF_MISC_SRC_INCLUDE | (linked ? CMD_BUF_MISC_LINKED : 0) |
                    (mcast ? CMD_BUF_MISC_MULTICAST : 0);
    __builtin_riscv_ttrocc_cmdbuf_wr_reg(cmd_buf, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_MISC_REG_OFFSET / 8, misc);

    __builtin_riscv_ttrocc_cmdbuf_wr_reg(cmd_buf, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_REQ_VC_REG_OFFSET / 8, vc);
    __builtin_riscv_ttrocc_cmdbuf_wr_reg(
        cmd_buf,
        TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_RESP_VC_REG_OFFSET / 8,
        mcast ? NOC_OVERLAY_MCAST_RESP_VC : NOC_OVERLAY_WR_RESP_VC);

    __builtin_riscv_ttrocc_cmdbuf_wr_reg(
        cmd_buf, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_SRC_ADDR_REG_OFFSET / 8, noc_v3_local_operand(src_addr));
    __builtin_riscv_ttrocc_cmdbuf_wr_reg(
        cmd_buf, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_DEST_ADDR_REG_OFFSET / 8, dest_addr);
    __builtin_riscv_ttrocc_cmdbuf_wr_reg(
        cmd_buf, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_LEN_BYTES_REG_OFFSET / 8, len_bytes);
    if (mcast) {
        __builtin_riscv_ttrocc_cmdbuf_wr_reg(
            cmd_buf, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_DEST_COORD_REG_OFFSET / 8, mcast_target.extent_xy);
        // HW needs MCAST_DESTS to match the number of cores in the rectangle
        // so it can track per-destination acks for the multicast.
        __builtin_riscv_ttrocc_cmdbuf_wr_reg(
            cmd_buf, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_MCAST_DESTS_REG_OFFSET / 8, num_dests);
    }
    __builtin_riscv_ttrocc_cmdbuf_issue_trans(cmd_buf);

    if constexpr (noc_mode == DM_DEDICATED_NOC) {
        uint32_t num_packets =
            len_bytes / NOC_OVERLAY_MAX_BYTES_IN_PACKET + ((len_bytes % NOC_OVERLAY_MAX_BYTES_IN_PACKET) ? 1 : 0);
        noc_nonposted_writes_num_issued[noc] += num_packets;
        noc_nonposted_writes_acked[noc] += num_dests * num_packets;
    }
}

template <uint8_t noc_mode = DM_DEDICATED_NOC, bool use_trid = false, bool one_packet = false>
inline __attribute__((always_inline)) void ncrisc_noc_fast_write_any_len(
    uint32_t noc,
    uint32_t cmd_buf,
    uint32_t src_addr,
    uint64_t dest_addr,
    uint32_t len_bytes,
    uint32_t vc,
    bool mcast,
    bool linked,
    uint32_t num_dests,
    bool multicast_path_reserve,
    bool posted = false,
    uint32_t trid = 0) {
    static_assert(noc_mode != DM_DYNAMIC_NOC, "Quasar does not support DYNAMIC_NOC as it has only 1 NOC");
    // Overlay handles packetization via MAX_BYTES_IN_PACKET register; no software chunking needed.
    ncrisc_noc_fast_write<noc_mode, use_trid>(
        noc,
        cmd_buf,
        src_addr,
        dest_addr,
        len_bytes,
        vc,
        mcast,
        linked,
        num_dests,
        multicast_path_reserve,
        posted,
        trid);
}

template <uint8_t noc_mode = DM_DEDICATED_NOC>
inline __attribute__((always_inline)) void ncrisc_noc_fast_write_any_len_loopback_src(
    uint32_t noc,
    uint32_t cmd_buf,
    uint32_t src_addr,
    uint64_t dest_addr,
    uint32_t len_bytes,
    uint32_t vc,
    bool mcast,
    bool linked,
    uint32_t num_dests,
    bool multicast_path_reserve) {
    static_assert(noc_mode != DM_DYNAMIC_NOC, "Quasar does not support DYNAMIC_NOC as it has only 1 NOC");
    // Overlay handles packetization via MAX_BYTES_IN_PACKET register; no software chunking needed.
    ncrisc_noc_fast_write_loopback_src<noc_mode>(
        noc, cmd_buf, src_addr, dest_addr, len_bytes, vc, mcast, linked, num_dests, multicast_path_reserve);
}

template <uint8_t noc_mode = DM_DEDICATED_NOC, InlineWriteDst dst_type = InlineWriteDst::DEFAULT, bool flush = true>
inline __attribute__((always_inline)) void noc_fast_write_dw_inline(
    uint32_t noc,
    uint32_t cmd_buf,
    uint32_t val,
    uint64_t dest_addr,
    uint32_t be,
    uint32_t static_vc,
    bool mcast,
    bool posted = false,
    uint32_t customized_src_addr = 0) {
    static_assert(noc_mode != DM_DYNAMIC_NOC, "Quasar does not support DYNAMIC_NOC as it has only 1 NOC");
    // Register recipe per cmdbuff_api.hpp (the reference): an inline write is a
    // plain write transaction whose data arrives with the inline-issue
    // instruction. LEN is the transfer size. The INLINE_WR/BYTE_ENABLE MISC bits
    // with a byte-enable mask in LEN never complete on this NIU (no ack, no
    // data), and one such issue wedges every later barrier on the core.
    ASSERT(be == 0xF);  // the Quasar RoCC path exposes no byte-enable mask for inline writes
    uint32_t num_dests = 1;
    uint32_t mcast_coord = 0;
    if (mcast) {
        // A multicast dest_addr is the packed software rectangle descriptor;
        // decode it through the active map exactly as ncrisc_noc_fast_write does.
        const noc_att::NocMulticastAddress mcast_target =
            noc_att::resolve_worker_multicast<ACTIVE_ATT_MAP>(dest_addr, sizeof(uint32_t));
        if (mcast_target.rectangle_count == 0) {
            __builtin_trap();
        }
        // Inline-write form: end operand + rectangle coordinates (header note).
        dest_addr = mcast_target.end_address;
        num_dests = mcast_target.rectangle_count;
        mcast_coord = noc_v3_inline_mcast_coord(mcast_target);
    }

    // linked follows the reference (linked = multicast); the sender is not
    // included, matching the Blackhole inline-multicast semantics.
    uint64_t misc = CMD_BUF_MISC_WRITE_TRANS | (posted ? CMD_BUF_MISC_POSTED : 0) |
                    (mcast ? (CMD_BUF_MISC_MULTICAST | CMD_BUF_MISC_LINKED) : 0);
    __builtin_riscv_ttrocc_scmdbuf_wr_reg(TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_MISC_REG_OFFSET / 8, misc);
    __builtin_riscv_ttrocc_scmdbuf_wr_reg(TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_REQ_VC_REG_OFFSET / 8, static_vc);
    __builtin_riscv_ttrocc_scmdbuf_wr_reg(
        TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_RESP_VC_REG_OFFSET / 8,
        mcast ? NOC_OVERLAY_MCAST_RESP_VC : NOC_OVERLAY_WR_RESP_VC);
    __builtin_riscv_ttrocc_scmdbuf_wr_reg(
        TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_LEN_BYTES_REG_OFFSET / 8, sizeof(uint32_t));
    __builtin_riscv_ttrocc_scmdbuf_wr_reg(TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_DEST_ADDR_REG_OFFSET / 8, dest_addr);
    if (mcast) {
        __builtin_riscv_ttrocc_scmdbuf_wr_reg(
            TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_DEST_COORD_REG_OFFSET / 8, mcast_coord);
        // HW needs MCAST_DESTS to match the number of cores in the rectangle
        // so it can track per-destination acks for the multicast.
        __builtin_riscv_ttrocc_scmdbuf_wr_reg(
            TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_MCAST_DESTS_REG_OFFSET / 8, num_dests);
    }
    __builtin_riscv_ttrocc_scmdbuf_issue_inline_trans(val);

    if constexpr (noc_mode == DM_DEDICATED_NOC) {
        if (posted) {
            noc_posted_writes_num_issued[noc] += 1;
        } else {
            noc_nonposted_writes_num_issued[noc] += 1;
            noc_nonposted_writes_acked[noc] += num_dests;
        }
    }
}

template <uint8_t noc_mode = DM_DEDICATED_NOC, InlineWriteDst dst_type = InlineWriteDst::DEFAULT, bool flush = true>
inline __attribute__((always_inline)) void noc_fast_write_dw_inline_multicast(
    uint32_t noc,
    uint32_t cmd_buf,
    uint32_t val,
    uint64_t dest_addr,
    uint32_t be,
    uint32_t static_vc,
    bool mcast,
    bool posted = false,
    uint32_t customized_src_addr = 0,
    uint32_t num_dests = 1) {
    static_assert(noc_mode != DM_DYNAMIC_NOC, "Quasar does not support DYNAMIC_NOC as it has only 1 NOC");
    // Same recipe as noc_fast_write_dw_inline (see there); the caller supplies
    // the destination count. The sender is not included, so a rectangle that
    // contains it has one destination fewer than its area.
    ASSERT(be == 0xF);  // the Quasar RoCC path exposes no byte-enable mask for inline writes
    uint32_t mcast_coord = 0;
    if (mcast) {
        const noc_att::NocMulticastAddress mcast_target =
            noc_att::resolve_worker_multicast<ACTIVE_ATT_MAP>(dest_addr, sizeof(uint32_t));
        if (mcast_target.rectangle_count == 0) {
            __builtin_trap();
        }
        ASSERT(num_dests <= mcast_target.rectangle_count);
        // Inline-write form: end operand + rectangle coordinates (header note).
        dest_addr = mcast_target.end_address;
        mcast_coord = noc_v3_inline_mcast_coord(mcast_target);
    }

    uint64_t misc = CMD_BUF_MISC_WRITE_TRANS | (posted ? CMD_BUF_MISC_POSTED : 0) |
                    (mcast ? (CMD_BUF_MISC_MULTICAST | CMD_BUF_MISC_LINKED) : 0);
    __builtin_riscv_ttrocc_scmdbuf_wr_reg(TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_MISC_REG_OFFSET / 8, misc);
    __builtin_riscv_ttrocc_scmdbuf_wr_reg(TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_REQ_VC_REG_OFFSET / 8, static_vc);
    __builtin_riscv_ttrocc_scmdbuf_wr_reg(
        TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_RESP_VC_REG_OFFSET / 8,
        mcast ? NOC_OVERLAY_MCAST_RESP_VC : NOC_OVERLAY_WR_RESP_VC);
    __builtin_riscv_ttrocc_scmdbuf_wr_reg(
        TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_LEN_BYTES_REG_OFFSET / 8, sizeof(uint32_t));
    __builtin_riscv_ttrocc_scmdbuf_wr_reg(TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_DEST_ADDR_REG_OFFSET / 8, dest_addr);
    if (mcast) {
        __builtin_riscv_ttrocc_scmdbuf_wr_reg(
            TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_DEST_COORD_REG_OFFSET / 8, mcast_coord);
        __builtin_riscv_ttrocc_scmdbuf_wr_reg(
            TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_MCAST_DESTS_REG_OFFSET / 8, num_dests);
    }
    __builtin_riscv_ttrocc_scmdbuf_issue_inline_trans(val);

    if (posted) {
        noc_posted_writes_num_issued[noc] += 1;
    } else {
        noc_nonposted_writes_num_issued[noc] += 1;
        noc_nonposted_writes_acked[noc] += num_dests;
    }
}

template <uint8_t noc_mode = DM_DEDICATED_NOC, bool program_ret_addr = false>
inline __attribute__((always_inline)) void noc_fast_atomic_increment(
    uint32_t noc,
    uint32_t cmd_buf,
    uint64_t addr,
    uint32_t vc,
    uint32_t incr,
    uint32_t wrap,
    bool linked,
    bool posted = false,
    uint32_t atomic_ret_val = 0) {
    static_assert(noc_mode != DM_DYNAMIC_NOC, "Quasar does not support DYNAMIC_NOC as it has only 1 NOC");
    uint64_t misc = CMD_BUF_MISC_ATOMIC_TRANS | CMD_BUF_MISC_SRC_INCLUDE | (posted ? CMD_BUF_MISC_POSTED : 0) |
                    (linked ? CMD_BUF_MISC_LINKED : 0);
    __builtin_riscv_ttrocc_scmdbuf_wr_reg(TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_MISC_REG_OFFSET / 8, misc);
    __builtin_riscv_ttrocc_scmdbuf_wr_reg(TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_REQ_VC_REG_OFFSET / 8, vc);
    __builtin_riscv_ttrocc_scmdbuf_wr_reg(
        TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_RESP_VC_REG_OFFSET / 8, NOC_OVERLAY_WR_RESP_VC);
    if constexpr (program_ret_addr) {
        // The atomic return value lands in this initiator's own L1.
        __builtin_riscv_ttrocc_scmdbuf_wr_reg(
            TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_SRC_ADDR_REG_OFFSET / 8, noc_v3_local_operand(atomic_ret_val));
    }
    __builtin_riscv_ttrocc_scmdbuf_wr_reg(TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_DEST_ADDR_REG_OFFSET / 8, addr);
    uint64_t at_len =
        NOC_AT_INS(NOC_AT_INS_INCR_GET) | NOC_AT_WRAP(wrap) | NOC_AT_IND_32((addr >> 2) & 0x3) | NOC_AT_IND_32_SRC(0);
    __builtin_riscv_ttrocc_scmdbuf_wr_reg(TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_LEN_BYTES_REG_OFFSET / 8, at_len);
    __builtin_riscv_ttrocc_scmdbuf_wr_reg(
        TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_INLINE_DATA_REG_OFFSET / 8, (uint64_t)incr);
    __builtin_riscv_ttrocc_scmdbuf_issue_trans();

    if (!posted) {
        noc_nonposted_atomics_acked[noc] += 1;
    }
}

// Quasar NoC CAS: a 4-BIT compare-and-swap on one 4-byte word of a 16B L1 atom. Succeeds
// iff word == {28'b0, cmp4}, then word <- {28'b0, swap4}; the PRE-OP word is returned to
// this hart's R_SRC_ADDR slot. Usable ONLY for words whose value stays in [0, 15].
// Same contract as V2; addr is a complete ATT operand and the return slot is this
// initiator's local address, issued as the local-window operand. The redirected
// return address stays latched on the simple command buffer until
// noc_restore_default_atomic_ret_addr.
template <uint8_t noc_mode = DM_DEDICATED_NOC>
inline __attribute__((always_inline)) void noc_fast_atomic_cas4(
    uint32_t noc, uint64_t addr, uint32_t vc, uint32_t cmp4, uint32_t swap4, uint32_t atomic_ret_val) {
    static_assert(noc_mode != DM_DYNAMIC_NOC, "Quasar does not support DYNAMIC_NOC as it has only 1 NOC");
    // Masked to 4 bits below; an out-of-range cmp could ACQUIRE a free lock.
    ASSERT(cmp4 <= 0xF && swap4 <= 0xF);
    uint64_t misc = CMD_BUF_MISC_ATOMIC_TRANS | CMD_BUF_MISC_SRC_INCLUDE;
    __builtin_riscv_ttrocc_scmdbuf_wr_reg(TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_MISC_REG_OFFSET / 8, misc);
    __builtin_riscv_ttrocc_scmdbuf_wr_reg(TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_REQ_VC_REG_OFFSET / 8, vc);
    __builtin_riscv_ttrocc_scmdbuf_wr_reg(
        TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_RESP_VC_REG_OFFSET / 8, NOC_OVERLAY_WR_RESP_VC);
    // The pre-op word lands in this initiator's own L1.
    __builtin_riscv_ttrocc_scmdbuf_wr_reg(
        TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_SRC_ADDR_REG_OFFSET / 8, noc_v3_local_operand(atomic_ret_val));
    __builtin_riscv_ttrocc_scmdbuf_wr_reg(TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_DEST_ADDR_REG_OFFSET / 8, addr);
    uint64_t at_len = NOC_AT_INS(NOC_AT_INS_CAS) | ((uint64_t)(swap4 & 0xF) << 6) | ((uint64_t)(cmp4 & 0xF) << 2) |
                      NOC_AT_IND_32((addr >> 2) & 0x3);
    __builtin_riscv_ttrocc_scmdbuf_wr_reg(TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_LEN_BYTES_REG_OFFSET / 8, at_len);
    __builtin_riscv_ttrocc_scmdbuf_wr_reg(TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_INLINE_DATA_REG_OFFSET / 8, (uint64_t)0);
    __builtin_riscv_ttrocc_scmdbuf_issue_trans();

    noc_nonposted_atomics_acked[noc] += 1;
}

template <uint8_t noc_mode = DM_DEDICATED_NOC>
inline __attribute__((always_inline)) void noc_fast_multicast_atomic_increment(
    uint32_t noc,
    uint32_t cmd_buf,
    uint64_t addr,
    uint32_t vc,
    uint32_t incr,
    uint32_t wrap,
    bool linked,
    uint32_t num_dests,
    bool multicast_path_reserve,
    bool posted = false,
    uint32_t atomic_ret_val = 0) {
    static_assert(noc_mode != DM_DYNAMIC_NOC, "Quasar does not support DYNAMIC_NOC as it has only 1 NOC");
    // Rectangle decode as for the multicast writes; the atomic return goes to
    // the default return slot, which the ATT-aware init_at_cmd_buf programs as
    // this initiator's local-window operand.
    const noc_att::NocMulticastAddress mcast_target =
        noc_att::resolve_worker_multicast<ACTIVE_ATT_MAP>(addr, sizeof(uint32_t));
    if (mcast_target.rectangle_count == 0) {
        __builtin_trap();
    }
    ASSERT(num_dests == mcast_target.rectangle_count);
    addr = mcast_target.start_address;

    uint64_t misc = CMD_BUF_MISC_ATOMIC_TRANS | CMD_BUF_MISC_SRC_INCLUDE | CMD_BUF_MISC_MULTICAST |
                    (posted ? CMD_BUF_MISC_POSTED : 0) | (linked ? CMD_BUF_MISC_LINKED : 0);
    __builtin_riscv_ttrocc_scmdbuf_wr_reg(TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_MISC_REG_OFFSET / 8, misc);
    __builtin_riscv_ttrocc_scmdbuf_wr_reg(TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_REQ_VC_REG_OFFSET / 8, vc);
    __builtin_riscv_ttrocc_scmdbuf_wr_reg(
        TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_RESP_VC_REG_OFFSET / 8, NOC_OVERLAY_MCAST_RESP_VC);
    __builtin_riscv_ttrocc_scmdbuf_wr_reg(TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_DEST_ADDR_REG_OFFSET / 8, addr);
    __builtin_riscv_ttrocc_scmdbuf_wr_reg(
        TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_DEST_COORD_REG_OFFSET / 8, mcast_target.extent_xy);
    uint64_t at_len =
        NOC_AT_INS(NOC_AT_INS_INCR_GET) | NOC_AT_WRAP(wrap) | NOC_AT_IND_32((addr >> 2) & 0x3) | NOC_AT_IND_32_SRC(0);
    __builtin_riscv_ttrocc_scmdbuf_wr_reg(TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_LEN_BYTES_REG_OFFSET / 8, at_len);
    __builtin_riscv_ttrocc_scmdbuf_wr_reg(
        TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_INLINE_DATA_REG_OFFSET / 8, (uint64_t)incr);
    // HW needs MCAST_DESTS to match the number of cores in the rectangle so
    // it can track per-destination acks for the multicast atomic.
    __builtin_riscv_ttrocc_scmdbuf_wr_reg(TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_MCAST_DESTS_REG_OFFSET / 8, num_dests);
    __builtin_riscv_ttrocc_scmdbuf_issue_trans();

    if (!posted) {
        noc_nonposted_atomics_acked[noc] += num_dests;
    }
}

// Transaction-id read against the latched read state: the remote base comes
// from the preceding read_set_state, the transaction id from
// ncrisc_noc_set_transaction_id; both src arguments are offsets within the
// target's window (V2 keeps the same contract, with its base latched in the
// coordinate register instead).
template <uint8_t noc_mode = DM_DEDICATED_NOC, bool skip_ptr_update = false, bool skip_cmdbuf_chk = false>
inline __attribute__((always_inline)) void ncrisc_noc_fast_read_with_transaction_id(
    uint32_t noc, uint32_t cmd_buf, uint32_t src_base_addr, uint32_t src_addr, uint32_t dest_addr, uint32_t trid) {
    static_assert(noc_mode != DM_DYNAMIC_NOC, "Quasar does not support DYNAMIC_NOC as it has only 1 NOC");
    uint32_t src_local_addr = src_base_addr + src_addr;

    while (NOC_STATUS_READ_REG(noc, NIU_MST_REQS_OUTSTANDING_ID(trid)) > ((NOC_MAX_TRANSACTION_ID + 1) / 2));

    __builtin_riscv_ttrocc_cmdbuf_wr_reg(
        cmd_buf, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_DEST_ADDR_REG_OFFSET / 8, noc_v3_local_operand(dest_addr));
    __builtin_riscv_ttrocc_cmdbuf_wr_reg(
        cmd_buf,
        TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_SRC_ADDR_REG_OFFSET / 8,
        noc_v3_state_operand(noc_v3_read_state_base[cmd_buf], src_local_addr));
    __builtin_riscv_ttrocc_cmdbuf_issue_trans(cmd_buf);
    if constexpr (!skip_ptr_update) {
        noc_reads_num_issued[noc] += 1;
    }
}

// ============================================================================
// Stateful pairs (set_state / with_state)
//
// set_state latches the target's base address in software; each with_state
// issue writes base | local as one full-operand register write - the same
// per-issue register count V2 achieves with its hardware coordinate latch.
// Contract: the base's offset field must be zero (get_noc_addr(x, y, 0)).
// ============================================================================

template <uint8_t noc_mode = DM_DEDICATED_NOC, bool one_packet = false, bool use_vc = false>
inline __attribute__((always_inline)) void ncrisc_noc_read_set_state(
    uint32_t noc, uint32_t cmd_buf, uint64_t src_noc_addr, uint32_t len_bytes = 0, const uint32_t vc = 0) {
    static_assert(noc_mode != DM_DYNAMIC_NOC, "Quasar does not support DYNAMIC_NOC as it has only 1 NOC");

    if constexpr (use_vc) {
        __builtin_riscv_ttrocc_cmdbuf_wr_reg(cmd_buf, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_REQ_VC_REG_OFFSET / 8, vc);
    }

    noc_v3_read_state_base[cmd_buf] = noc_v3_state_base_of(src_noc_addr);

    if constexpr (one_packet) {
        __builtin_riscv_ttrocc_cmdbuf_wr_reg(
            cmd_buf, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_LEN_BYTES_REG_OFFSET / 8, len_bytes);
    }
}

template <uint8_t noc_mode = DM_DEDICATED_NOC, bool inc_num_issued = true, bool one_packet = false>
inline __attribute__((always_inline)) void ncrisc_noc_read_with_state(
    uint32_t noc, uint32_t cmd_buf, uint32_t src_local_addr, uint32_t dst_local_addr, uint32_t len_bytes = 0) {
    static_assert(noc_mode != DM_DYNAMIC_NOC, "Quasar does not support DYNAMIC_NOC as it has only 1 NOC");

    __builtin_riscv_ttrocc_cmdbuf_wr_reg(
        cmd_buf,
        TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_SRC_ADDR_REG_OFFSET / 8,
        noc_v3_state_operand(noc_v3_read_state_base[cmd_buf], src_local_addr));
    __builtin_riscv_ttrocc_cmdbuf_wr_reg(
        cmd_buf, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_DEST_ADDR_REG_OFFSET / 8, noc_v3_local_operand(dst_local_addr));
    if constexpr (!one_packet) {
        __builtin_riscv_ttrocc_cmdbuf_wr_reg(
            cmd_buf, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_LEN_BYTES_REG_OFFSET / 8, len_bytes);
    }
    __builtin_riscv_ttrocc_cmdbuf_issue_trans(cmd_buf);

    if constexpr (inc_num_issued) {
        if constexpr (one_packet) {
            noc_reads_num_issued[noc] += 1;
        } else {
            uint32_t num_packets =
                len_bytes / NOC_OVERLAY_MAX_BYTES_IN_PACKET + ((len_bytes % NOC_OVERLAY_MAX_BYTES_IN_PACKET) ? 1 : 0);
            noc_reads_num_issued[noc] += num_packets;
        }
    }
}

template <uint8_t noc_mode = DM_DEDICATED_NOC, bool inc_num_issued = true>
inline __attribute__((always_inline)) void ncrisc_noc_read_any_len_with_state(
    uint32_t noc, uint32_t cmd_buf, uint32_t src_local_addr, uint32_t dst_local_addr, uint32_t len_bytes) {
    static_assert(noc_mode != DM_DYNAMIC_NOC, "Quasar does not support DYNAMIC_NOC as it has only 1 NOC");
    // Overlay handles packetization via MAX_BYTES_IN_PACKET register; no software chunking needed.
    ncrisc_noc_read_with_state<noc_mode, inc_num_issued>(noc, cmd_buf, src_local_addr, dst_local_addr, len_bytes);
}

template <bool posted = false, bool one_packet = false>
inline __attribute__((always_inline)) void ncrisc_noc_write_set_state(
    uint32_t noc, uint32_t cmd_buf, uint64_t dst_noc_addr, uint32_t len_bytes = 0, const uint32_t vc = 0) {
    // MISC: write, posted flag from template param.
    uint64_t misc = CMD_BUF_MISC_WRITE_TRANS | CMD_BUF_MISC_SRC_INCLUDE | (posted ? CMD_BUF_MISC_POSTED : 0);
    __builtin_riscv_ttrocc_cmdbuf_wr_reg(cmd_buf, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_MISC_REG_OFFSET / 8, misc);
    __builtin_riscv_ttrocc_cmdbuf_wr_reg(cmd_buf, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_REQ_VC_REG_OFFSET / 8, vc);
    __builtin_riscv_ttrocc_cmdbuf_wr_reg(
        cmd_buf, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_RESP_VC_REG_OFFSET / 8, NOC_OVERLAY_WR_RESP_VC);

    noc_v3_write_state_base[cmd_buf] = noc_v3_state_base_of(dst_noc_addr);

    if constexpr (one_packet) {
        __builtin_riscv_ttrocc_cmdbuf_wr_reg(
            cmd_buf, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_LEN_BYTES_REG_OFFSET / 8, len_bytes);
    }
}

template <uint8_t noc_mode = DM_DEDICATED_NOC, bool posted = false, bool update_counter = true, bool one_packet = false>
inline __attribute__((always_inline)) void ncrisc_noc_write_with_state(
    uint32_t noc, uint32_t cmd_buf, uint32_t src_local_addr, uint32_t dst_local_addr, uint32_t len_bytes = 0) {
    static_assert(noc_mode != DM_DYNAMIC_NOC, "Quasar does not support DYNAMIC_NOC as it has only 1 NOC");

    __builtin_riscv_ttrocc_cmdbuf_wr_reg(
        cmd_buf, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_SRC_ADDR_REG_OFFSET / 8, noc_v3_local_operand(src_local_addr));
    __builtin_riscv_ttrocc_cmdbuf_wr_reg(
        cmd_buf,
        TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_DEST_ADDR_REG_OFFSET / 8,
        noc_v3_state_operand(noc_v3_write_state_base[cmd_buf], dst_local_addr));
    if constexpr (!one_packet) {
        __builtin_riscv_ttrocc_cmdbuf_wr_reg(
            cmd_buf, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_LEN_BYTES_REG_OFFSET / 8, len_bytes);
    }
    __builtin_riscv_ttrocc_cmdbuf_issue_trans(cmd_buf);

    if constexpr (update_counter) {
        if constexpr (one_packet) {
            if constexpr (posted) {
                noc_posted_writes_num_issued[noc] += 1;
            } else {
                noc_nonposted_writes_num_issued[noc] += 1;
                noc_nonposted_writes_acked[noc] += 1;
            }
        } else {
            uint32_t num_packets =
                len_bytes / NOC_OVERLAY_MAX_BYTES_IN_PACKET + ((len_bytes % NOC_OVERLAY_MAX_BYTES_IN_PACKET) ? 1 : 0);
            if constexpr (posted) {
                noc_posted_writes_num_issued[noc] += num_packets;
            } else {
                noc_nonposted_writes_num_issued[noc] += num_packets;
                noc_nonposted_writes_acked[noc] += num_packets;
            }
        }
    }
}

template <uint8_t noc_mode = DM_DEDICATED_NOC, bool posted = false, bool update_counter = true>
inline __attribute__((always_inline)) void ncrisc_noc_write_any_len_with_state(
    uint32_t noc, uint32_t cmd_buf, uint32_t src_local_addr, uint32_t dst_local_addr, uint32_t len_bytes) {
    static_assert(noc_mode != DM_DYNAMIC_NOC, "Quasar does not support DYNAMIC_NOC as it has only 1 NOC");
    // Overlay handles packetization via MAX_BYTES_IN_PACKET register; no software chunking needed.
    ncrisc_noc_write_with_state<noc_mode, posted, update_counter>(
        noc, cmd_buf, src_local_addr, dst_local_addr, len_bytes);
}

template <bool posted = false, bool set_val = false>
inline __attribute__((always_inline)) void noc_fast_write_dw_inline_set_state(
    uint32_t noc, uint32_t cmd_buf, uint64_t dest_addr, uint32_t be, uint32_t static_vc, uint32_t val = 0) {
    // Reference recipe (cmdbuff_api.hpp): plain write, LEN = dword; see
    // noc_fast_write_dw_inline for why the INLINE_WR/BYTE_ENABLE form is wrong.
    ASSERT(be == 0xF);  // the Quasar RoCC path exposes no byte-enable mask for inline writes
    if constexpr (set_val) {
        noc_v3_inline_write_state_val = val;
    }
    uint64_t misc = CMD_BUF_MISC_WRITE_TRANS | (posted ? CMD_BUF_MISC_POSTED : 0);
    __builtin_riscv_ttrocc_scmdbuf_wr_reg(TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_MISC_REG_OFFSET / 8, misc);
    __builtin_riscv_ttrocc_scmdbuf_wr_reg(TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_REQ_VC_REG_OFFSET / 8, static_vc);
    __builtin_riscv_ttrocc_scmdbuf_wr_reg(
        TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_RESP_VC_REG_OFFSET / 8, NOC_OVERLAY_WR_RESP_VC);
    __builtin_riscv_ttrocc_scmdbuf_wr_reg(TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_DEST_ADDR_REG_OFFSET / 8, dest_addr);
    // Held for update_addr_lo issues: the sticky DEST_ADDR register carries the
    // full operand, so a per-issue local address must be folded into the state
    // base rather than written whole. V2 semantics: the state address's own
    // local bits never survive into updated issues.
    noc_v3_inline_write_state_base = noc_v3_state_base_of(dest_addr);

    __builtin_riscv_ttrocc_scmdbuf_wr_reg(
        TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_LEN_BYTES_REG_OFFSET / 8, sizeof(uint32_t));
}

// The V2 update_addr_hi variant patches the destination coordinate register -
// an XY-only contract with no ATT equivalent.
template <
    uint8_t noc_mode = DM_DEDICATED_NOC,
    bool update_addr_lo = false,
    bool update_addr_hi = false,
    bool update_val = false,
    bool posted = false,
    bool update_counter = true,
    InlineWriteDst dst_type = InlineWriteDst::DEFAULT>
inline __attribute__((always_inline)) void noc_fast_write_dw_inline_with_state(
    uint32_t noc, uint32_t cmd_buf, uint32_t val = 0, uint64_t dest_addr = 0) {
    static_assert(noc_mode != DM_DYNAMIC_NOC, "Quasar does not support DYNAMIC_NOC as it has only 1 NOC");
    static_assert("Error: Only High or Low address update is supported" && (update_addr_lo && update_addr_hi) == 0);
    static_assert(
        !update_addr_hi,
        "update_addr_hi patches the XY coordinate register; ATT operands have no coordinate half - update the full "
        "address with update_addr_lo");

    if constexpr (update_addr_lo) {
        // dest_addr is the per-issue LOCAL address (V2 replaced only the low
        // register half); the sticky DEST_ADDR register holds a full ATT
        // operand, so fold the local into the state base - writing the bare
        // local would discard the window and selector bits.
        __builtin_riscv_ttrocc_scmdbuf_wr_reg(
            TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_DEST_ADDR_REG_OFFSET / 8,
            noc_v3_state_operand(noc_v3_inline_write_state_base, static_cast<uint32_t>(dest_addr)));
    }
    // update_val == false reuses the value set_state<set_val> latched.
    __builtin_riscv_ttrocc_scmdbuf_issue_inline_trans(update_val ? val : noc_v3_inline_write_state_val);

    if constexpr (update_counter) {
        if constexpr (posted) {
            noc_posted_writes_num_issued[noc] += 1;
        } else {
            noc_nonposted_writes_num_issued[noc] += 1;
            noc_nonposted_writes_acked[noc] += 1;
        }
    }
}

// ============================================================================
// Stateful NoC calls for the command-queue (dispatch) kernels.
//
// The dispatch kernels program a transfer in pieces and reuse what they set
// before: the CQ_NOC_* flags on each call say which fields (source,
// coordinate, destination offset, length) that call updates. Under ATT the
// destination is a single 64-bit address and another register carries multicast
// extents, so the pieces are kept in software: for every command buffer this
// file remembers the coordinate half and the offset half, and writes the
// combined address to the hardware whenever either half changes.
//
// Multicast destinations are remembered as a rectangle descriptor and resolved
// through the ATT map when the transfer is issued. Reads mirror all of this on
// the source side.
// ============================================================================

constexpr uint32_t CQ_NOC_FLAG_SRC = 0x01;
constexpr uint32_t CQ_NOC_FLAG_NOC = 0x02;
constexpr uint32_t CQ_NOC_FLAG_DST = 0x04;
constexpr uint32_t CQ_NOC_FLAG_LEN = 0x08;

constexpr uint32_t CQ_NOC_INLINE_FLAG_VAL = 0x10;
constexpr uint32_t CQ_NOC_INLINE_FLAG_BE = 0x20;

constexpr uint32_t CQ_NOC_CMD_FLAG_MCAST = 0x01;
constexpr uint32_t CQ_NOC_CMD_FLAG_LINKED = 0x02;
constexpr uint32_t CQ_NOC_CMD_FLAG_POSTED = 0x04;

enum CQNocFlags {
    CQ_NOC_sndl = 0,
    CQ_NOC_sndL = CQ_NOC_FLAG_LEN,
    CQ_NOC_snDl = CQ_NOC_FLAG_DST,
    CQ_NOC_snDL = CQ_NOC_FLAG_DST | CQ_NOC_FLAG_LEN,
    CQ_NOC_sNdl = CQ_NOC_FLAG_NOC,
    CQ_NOC_sNdL = CQ_NOC_FLAG_NOC | CQ_NOC_FLAG_LEN,
    CQ_NOC_sNDl = CQ_NOC_FLAG_NOC | CQ_NOC_FLAG_DST,
    CQ_NOC_sNDL = CQ_NOC_FLAG_NOC | CQ_NOC_FLAG_DST | CQ_NOC_FLAG_LEN,
    CQ_NOC_Sndl = CQ_NOC_FLAG_SRC,
    CQ_NOC_SndL = CQ_NOC_FLAG_SRC | CQ_NOC_FLAG_LEN,
    CQ_NOC_SnDl = CQ_NOC_FLAG_SRC | CQ_NOC_FLAG_DST,
    CQ_NOC_SnDL = CQ_NOC_FLAG_SRC | CQ_NOC_FLAG_DST | CQ_NOC_FLAG_LEN,
    CQ_NOC_SNdl = CQ_NOC_FLAG_SRC | CQ_NOC_FLAG_NOC,
    CQ_NOC_SNdL = CQ_NOC_FLAG_SRC | CQ_NOC_FLAG_NOC | CQ_NOC_FLAG_LEN,
    CQ_NOC_SNDl = CQ_NOC_FLAG_SRC | CQ_NOC_FLAG_NOC | CQ_NOC_FLAG_DST,
    CQ_NOC_SNDL = CQ_NOC_FLAG_SRC | CQ_NOC_FLAG_NOC | CQ_NOC_FLAG_DST | CQ_NOC_FLAG_LEN,
};

enum CQNocCmdFlags {
    CQ_NOC_mkp = 0,
    CQ_NOC_mkP = CQ_NOC_CMD_FLAG_POSTED,
    CQ_NOC_mKp = CQ_NOC_CMD_FLAG_LINKED,
    CQ_NOC_mKP = CQ_NOC_CMD_FLAG_LINKED | CQ_NOC_CMD_FLAG_POSTED,
    CQ_NOC_Mkp = CQ_NOC_CMD_FLAG_MCAST,
    CQ_NOC_MkP = CQ_NOC_CMD_FLAG_MCAST | CQ_NOC_CMD_FLAG_POSTED,
    CQ_NOC_MKp = CQ_NOC_CMD_FLAG_MCAST | CQ_NOC_CMD_FLAG_LINKED,
    CQ_NOC_MKP = CQ_NOC_CMD_FLAG_MCAST | CQ_NOC_CMD_FLAG_LINKED | CQ_NOC_CMD_FLAG_POSTED,
};

enum CQNocInlineFlags {
    CQ_NOC_INLINE_ndvb = 0,
    CQ_NOC_INLINE_ndvB = CQ_NOC_INLINE_FLAG_BE,
    CQ_NOC_INLINE_ndVb = CQ_NOC_INLINE_FLAG_VAL,
    CQ_NOC_INLINE_ndVB = CQ_NOC_INLINE_FLAG_VAL | CQ_NOC_INLINE_FLAG_BE,
    CQ_NOC_INLINE_nDvb = CQ_NOC_FLAG_DST,
    CQ_NOC_INLINE_nDvB = CQ_NOC_FLAG_DST | CQ_NOC_INLINE_FLAG_BE,
    CQ_NOC_INLINE_nDVb = CQ_NOC_FLAG_DST | CQ_NOC_INLINE_FLAG_VAL,
    CQ_NOC_INLINE_nDVB = CQ_NOC_FLAG_DST | CQ_NOC_INLINE_FLAG_VAL | CQ_NOC_INLINE_FLAG_BE,
    CQ_NOC_INLINE_Ndvb = CQ_NOC_FLAG_NOC,
    CQ_NOC_INLINE_NdvB = CQ_NOC_FLAG_NOC | CQ_NOC_INLINE_FLAG_BE,
    CQ_NOC_INLINE_NdVb = CQ_NOC_FLAG_NOC | CQ_NOC_INLINE_FLAG_VAL,
    CQ_NOC_INLINE_NdVB = CQ_NOC_FLAG_NOC | CQ_NOC_INLINE_FLAG_VAL | CQ_NOC_INLINE_FLAG_BE,
    CQ_NOC_INLINE_NDvb = CQ_NOC_FLAG_NOC | CQ_NOC_FLAG_DST,
    CQ_NOC_INLINE_NDvB = CQ_NOC_FLAG_NOC | CQ_NOC_FLAG_DST | CQ_NOC_INLINE_FLAG_BE,
    CQ_NOC_INLINE_NDVb = CQ_NOC_FLAG_NOC | CQ_NOC_FLAG_DST | CQ_NOC_INLINE_FLAG_VAL,
    CQ_NOC_INLINE_NDVB = CQ_NOC_FLAG_NOC | CQ_NOC_FLAG_DST | CQ_NOC_INLINE_FLAG_VAL | CQ_NOC_INLINE_FLAG_BE,
};

enum CQNocWait {
    CQ_NOC_wait = 0,
    CQ_NOC_WAIT = 1,
};
enum CQNocSend {
    CQ_NOC_send = 0,
    CQ_NOC_SEND = 1,
};

// The remembered state of one command buffer
struct alignas(64) NocV3CqState {
    // The two halves of the destination and of the source. "base" is the
    // coordinate half: a full address with a zero offset, or for multicast a
    // rectangle descriptor with a zero offset. "local" is the offset half. The
    // hardware receives base | local.
    uint64_t dest_base;
    uint64_t dest_local;
    uint64_t src_base;
    uint64_t src_local;
    // The value the last CQ_NOC_INLINE_FLAG_VAL call set. Inline data travels
    // with the inline-issue instruction, not through a register, so a send
    // without that flag issues this value again.
    uint32_t inline_val;
    // Set by the write init call from its multicast flag. When true, the
    // destination halves form a multicast rectangle that is resolved through
    // the map when the transfer is issued.
    bool dest_mcast;
};
inline NocV3CqState noc_v3_cq_state[NOC_V3_STATE_CMD_BUFS] = {};

// This state is a kernel global in L1, and the DM kernel runtime zeroes only
// its local-memory bss, so it starts with whatever the previous kernel left
// behind. Every CQ kernel calls this once at entry.
inline __attribute__((always_inline)) void noc_v3_cq_state_reset() {
    for (uint32_t i = 0; i < NOC_V3_STATE_CMD_BUFS; ++i) {
        noc_v3_cq_state[i] = {};
    }
}

// A multicast descriptor holds the offset in its low bits and the rectangle
// corners above them, so its two halves combine with OR just like a unicast
// address. This mask separates them.
inline constexpr uint64_t NOC_V3_CQ_MCAST_LOCAL_MASK = noc_att::DESCRIPTOR_LOCAL_LIMIT - 1;

// Extract the offset half of an address.
inline __attribute__((always_inline)) uint64_t noc_v3_cq_local_of(uint64_t noc_addr) {
    const noc_att::WindowClass window_class = noc_att::matching_window_class(ACTIVE_ATT_MAP, noc_addr);
    if (window_class == noc_att::WindowClass::Invalid) {
        return noc_addr;
    }
    return noc_att::map_window(ACTIVE_ATT_MAP, window_class).local_address(noc_addr);
}

// Turn a host-packed coordinate word ((y << NOC_ADDR_NODE_ID_BITS) | x) into
// the coordinate half of an address: the tile's window and selector with a
// zero offset.
inline __attribute__((always_inline)) uint64_t noc_v3_cq_packed_base(uint32_t packed_xy) {
    constexpr uint32_t node_mask = (1u << NOC_ADDR_NODE_ID_BITS) - 1;
    const noc_att::ResolvedTile tile = noc_att::resolve_current(
        ACTIVE_ATT_MAP, packed_xy & node_mask, (packed_xy >> NOC_ADDR_NODE_ID_BITS) & node_mask);
    if (!tile.valid) {
        __builtin_trap();
    }
    return noc_att::map_window(ACTIVE_ATT_MAP, tile.window).make_address(tile.selector, 0);
}

// Turn a host-packed multicast rectangle (NOC_MULTICAST_ENCODING: x_end,
// y_end, x_start, y_start from low to high, each NOC_ADDR_NODE_ID_BITS wide)
// into a rectangle descriptor with a zero offset.
inline __attribute__((always_inline)) uint64_t noc_v3_cq_packed_mcast_base(uint32_t packed_rect) {
    constexpr uint32_t node_mask = (1u << NOC_ADDR_NODE_ID_BITS) - 1;
    const uint32_t x_end = packed_rect & node_mask;
    const uint32_t y_end = (packed_rect >> NOC_ADDR_NODE_ID_BITS) & node_mask;
    const uint32_t x_start = (packed_rect >> (2 * NOC_ADDR_NODE_ID_BITS)) & node_mask;
    const uint32_t y_start = (packed_rect >> (3 * NOC_ADDR_NODE_ID_BITS)) & node_mask;
    return noc_att::make_multicast_descriptor(x_start, y_start, x_end, y_end, 0);
}

// Write the remembered destination to the hardware. Unicast: one register,
// base | local. Multicast: resolve the descriptor through the map; the first
// tile's address goes to DEST_ADDR and the rectangle extents to DEST_COORD.
template <uint32_t cmd_buf, enum CQNocFlags flags, bool check_count = true>
inline __attribute__((always_inline)) void noc_v3_cq_program_dest(uint32_t size, uint32_t ndests) {
    if (noc_v3_cq_state[cmd_buf].dest_mcast) {
        const uint64_t descriptor = noc_v3_cq_state[cmd_buf].dest_base | noc_v3_cq_state[cmd_buf].dest_local;
        const noc_att::NocMulticastAddress target =
            noc_att::resolve_worker_multicast<ACTIVE_ATT_MAP>(descriptor, size != 0 ? size : 1);
        if (target.rectangle_count == 0) {
            __builtin_trap();  // a corner outside the map; ASSERT does nothing outside checked builds
        }
        if constexpr (check_count) {
            ASSERT(ndests == target.rectangle_count);
        }
        __builtin_riscv_ttrocc_cmdbuf_wr_reg(
            cmd_buf, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_DEST_ADDR_REG_OFFSET / 8, target.start_address);
        if constexpr (flags & CQ_NOC_FLAG_NOC) {
            __builtin_riscv_ttrocc_cmdbuf_wr_reg(
                cmd_buf, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_DEST_COORD_REG_OFFSET / 8, target.extent_xy);
        }
    } else {
        __builtin_riscv_ttrocc_cmdbuf_wr_reg(
            cmd_buf,
            TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_DEST_ADDR_REG_OFFSET / 8,
            noc_v3_cq_state[cmd_buf].dest_base | noc_v3_cq_state[cmd_buf].dest_local);
    }
}

// --------------------------------------------------------------------------
// Read family. On reads the NOC flag updates the source coordinate.
// --------------------------------------------------------------------------

template <uint32_t cmd_buf>
inline __attribute__((always_inline)) void noc_read_init_state(uint32_t noc) {
    __builtin_riscv_ttrocc_cmdbuf_wr_reg(
        cmd_buf, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_MISC_REG_OFFSET / 8, CMD_BUF_MISC_READ);
}

// src_addr carries the whole source: a full ATT address, or, when only the
// SRC flag is set, a bare offset combined with the remembered source
// coordinate.
template <
    uint8_t noc_mode = DM_DEDICATED_NOC,
    uint32_t cmd_buf,
    enum CQNocFlags flags,
    enum CQNocSend send = CQ_NOC_SEND,
    enum CQNocWait wait = CQ_NOC_WAIT>
inline __attribute__((always_inline)) void noc_read_with_state(
    uint32_t noc, uint64_t src_addr, uint32_t dst_addr, uint32_t size) {
    static_assert(noc_mode != DM_DYNAMIC_NOC, "Quasar does not support DYNAMIC_NOC as it has only 1 NOC");

    if constexpr (flags & CQ_NOC_FLAG_SRC) {
        noc_v3_cq_state[cmd_buf].src_local = noc_v3_cq_local_of(src_addr);
    }
    if constexpr (flags & CQ_NOC_FLAG_NOC) {
        noc_v3_cq_state[cmd_buf].src_base = noc_v3_state_base_of(src_addr);
    }
    if constexpr (flags & (CQ_NOC_FLAG_SRC | CQ_NOC_FLAG_NOC)) {
        __builtin_riscv_ttrocc_cmdbuf_wr_reg(
            cmd_buf,
            TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_SRC_ADDR_REG_OFFSET / 8,
            noc_v3_cq_state[cmd_buf].src_base | noc_v3_cq_state[cmd_buf].src_local);
    }
    if constexpr (flags & CQ_NOC_FLAG_DST) {
        __builtin_riscv_ttrocc_cmdbuf_wr_reg(
            cmd_buf, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_DEST_ADDR_REG_OFFSET / 8, noc_v3_local_operand(dst_addr));
    }
    if constexpr (flags & CQ_NOC_FLAG_LEN) {
        __builtin_riscv_ttrocc_cmdbuf_wr_reg(
            cmd_buf, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_LEN_BYTES_REG_OFFSET / 8, size);
    }
    if constexpr (send) {
        __builtin_riscv_ttrocc_cmdbuf_issue_trans(cmd_buf);
        // Count one read per issued call. The barrier waits on the hardware's
        // outstanding count, so the exact number of packets does not matter,
        // only that calls which just program state are not counted.
        noc_reads_num_issued[noc] += 1;
    }
}

// The source arrives in two parts, a host-packed coordinate word and a plain
// offset, so the offset needs no window lookup.
template <
    uint8_t noc_mode = DM_DEDICATED_NOC,
    uint32_t cmd_buf,
    enum CQNocFlags flags,
    enum CQNocSend send = CQ_NOC_SEND,
    enum CQNocWait wait = CQ_NOC_WAIT>
inline __attribute__((always_inline)) void noc_read_with_state(
    uint32_t noc, uint32_t src_noc_addr, uint64_t src_addr, uint32_t dst_addr, uint32_t size) {
    static_assert(noc_mode != DM_DYNAMIC_NOC, "Quasar does not support DYNAMIC_NOC as it has only 1 NOC");

    if constexpr (flags & CQ_NOC_FLAG_SRC) {
        noc_v3_cq_state[cmd_buf].src_local = src_addr;
    }
    if constexpr (flags & CQ_NOC_FLAG_NOC) {
        noc_v3_cq_state[cmd_buf].src_base = noc_v3_cq_packed_base(src_noc_addr);
    }
    if constexpr (flags & (CQ_NOC_FLAG_SRC | CQ_NOC_FLAG_NOC)) {
        __builtin_riscv_ttrocc_cmdbuf_wr_reg(
            cmd_buf,
            TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_SRC_ADDR_REG_OFFSET / 8,
            noc_v3_cq_state[cmd_buf].src_base | noc_v3_cq_state[cmd_buf].src_local);
    }
    if constexpr (flags & CQ_NOC_FLAG_DST) {
        __builtin_riscv_ttrocc_cmdbuf_wr_reg(
            cmd_buf, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_DEST_ADDR_REG_OFFSET / 8, noc_v3_local_operand(dst_addr));
    }
    if constexpr (flags & CQ_NOC_FLAG_LEN) {
        __builtin_riscv_ttrocc_cmdbuf_wr_reg(
            cmd_buf, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_LEN_BYTES_REG_OFFSET / 8, size);
    }
    if constexpr (send) {
        __builtin_riscv_ttrocc_cmdbuf_issue_trans(cmd_buf);
        noc_reads_num_issued[noc] += 1;
    }
}

// --------------------------------------------------------------------------
// Write family.
// --------------------------------------------------------------------------

template <uint32_t cmd_buf, enum CQNocCmdFlags cmd_flags = CQ_NOC_mkp>
inline __attribute__((always_inline)) void noc_write_init_state(uint32_t noc, uint32_t vc) {
    uint64_t misc = CMD_BUF_MISC_WRITE_TRANS | CMD_BUF_MISC_SRC_INCLUDE |
                    ((cmd_flags & CQ_NOC_CMD_FLAG_LINKED) ? CMD_BUF_MISC_LINKED : 0) |
                    ((cmd_flags & CQ_NOC_CMD_FLAG_MCAST) ? CMD_BUF_MISC_MULTICAST : 0) |
                    ((cmd_flags & CQ_NOC_CMD_FLAG_POSTED) ? CMD_BUF_MISC_POSTED : 0);
    __builtin_riscv_ttrocc_cmdbuf_wr_reg(cmd_buf, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_MISC_REG_OFFSET / 8, misc);
    __builtin_riscv_ttrocc_cmdbuf_wr_reg(cmd_buf, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_REQ_VC_REG_OFFSET / 8, vc);
    __builtin_riscv_ttrocc_cmdbuf_wr_reg(
        cmd_buf,
        TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_RESP_VC_REG_OFFSET / 8,
        (cmd_flags & CQ_NOC_CMD_FLAG_MCAST) ? NOC_OVERLAY_MCAST_RESP_VC : NOC_OVERLAY_WR_RESP_VC);
    // Remember whether this command buffer's destination is a multicast rectangle.
    noc_v3_cq_state[cmd_buf].dest_mcast = (cmd_flags & CQ_NOC_CMD_FLAG_MCAST) != 0;
}

template <
    uint8_t noc_mode = DM_DEDICATED_NOC,
    uint32_t cmd_buf,
    enum CQNocFlags flags,
    enum CQNocSend send = CQ_NOC_SEND,
    enum CQNocWait wait = CQ_NOC_WAIT,
    bool update_counter = true,
    bool posted = false>
inline __attribute__((always_inline)) void noc_write_with_state(
    uint32_t noc, uint32_t src_addr, uint64_t dst_addr, uint32_t size = 0, uint32_t ndests = 1) {
    static_assert(noc_mode != DM_DYNAMIC_NOC, "Quasar does not support DYNAMIC_NOC as it has only 1 NOC");

    if constexpr (flags & CQ_NOC_FLAG_SRC) {
        __builtin_riscv_ttrocc_cmdbuf_wr_reg(
            cmd_buf, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_SRC_ADDR_REG_OFFSET / 8, noc_v3_local_operand(src_addr));
    }
    if constexpr (flags & CQ_NOC_FLAG_NOC) {
        noc_v3_cq_state[cmd_buf].dest_base = noc_v3_cq_state[cmd_buf].dest_mcast
                                                 ? (dst_addr & ~NOC_V3_CQ_MCAST_LOCAL_MASK)
                                                 : noc_v3_state_base_of(dst_addr);
    }
    if constexpr (flags & CQ_NOC_FLAG_DST) {
        noc_v3_cq_state[cmd_buf].dest_local = noc_v3_cq_state[cmd_buf].dest_mcast
                                                  ? (dst_addr & NOC_V3_CQ_MCAST_LOCAL_MASK)
                                                  : noc_v3_cq_local_of(dst_addr);
    }
    if constexpr (flags & (CQ_NOC_FLAG_NOC | CQ_NOC_FLAG_DST)) {
        noc_v3_cq_program_dest<cmd_buf, flags, send == CQ_NOC_SEND>(size, ndests);
    }
    if constexpr (flags & CQ_NOC_FLAG_LEN) {
        __builtin_riscv_ttrocc_cmdbuf_wr_reg(
            cmd_buf, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_LEN_BYTES_REG_OFFSET / 8, size);
    }
    __builtin_riscv_ttrocc_cmdbuf_wr_reg(
        cmd_buf, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_MCAST_DESTS_REG_OFFSET / 8, ndests);
    if constexpr (send) {
        __builtin_riscv_ttrocc_cmdbuf_issue_trans(cmd_buf);
    }

    if constexpr (update_counter) {
        if constexpr (posted) {
            noc_posted_writes_num_issued[noc] += 1;
        } else {
            noc_nonposted_writes_num_issued[noc] += 1;
            noc_nonposted_writes_acked[noc] += ndests;
        }
    }
}

// The destination arrives in two parts: a host-packed coordinate word (or a
// packed multicast rectangle) and a plain offset.
template <
    uint8_t noc_mode = DM_DEDICATED_NOC,
    uint32_t cmd_buf,
    enum CQNocFlags flags,
    enum CQNocSend send = CQ_NOC_SEND,
    enum CQNocWait wait = CQ_NOC_WAIT,
    bool update_counter = true,
    bool posted = false>
inline __attribute__((always_inline)) void noc_wwrite_with_state(
    uint32_t noc, uint32_t src_addr, uint32_t dst_noc_addr, uint64_t dst_addr, uint32_t size = 0, uint32_t ndests = 1) {
    static_assert(noc_mode != DM_DYNAMIC_NOC, "Quasar does not support DYNAMIC_NOC as it has only 1 NOC");

    if constexpr (flags & CQ_NOC_FLAG_SRC) {
        __builtin_riscv_ttrocc_cmdbuf_wr_reg(
            cmd_buf, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_SRC_ADDR_REG_OFFSET / 8, noc_v3_local_operand(src_addr));
    }
    if constexpr (flags & CQ_NOC_FLAG_NOC) {
        noc_v3_cq_state[cmd_buf].dest_base = noc_v3_cq_state[cmd_buf].dest_mcast
                                                 ? noc_v3_cq_packed_mcast_base(dst_noc_addr)
                                                 : noc_v3_cq_packed_base(dst_noc_addr);
    }
    if constexpr (flags & CQ_NOC_FLAG_DST) {
        if (noc_v3_cq_state[cmd_buf].dest_mcast) {
            ASSERT(dst_addr < noc_att::DESCRIPTOR_LOCAL_LIMIT);
            noc_v3_cq_state[cmd_buf].dest_local = dst_addr & NOC_V3_CQ_MCAST_LOCAL_MASK;
        } else {
            noc_v3_cq_state[cmd_buf].dest_local = dst_addr;
        }
    }
    if constexpr (flags & (CQ_NOC_FLAG_NOC | CQ_NOC_FLAG_DST)) {
        noc_v3_cq_program_dest<cmd_buf, flags, send == CQ_NOC_SEND>(size, ndests);
    }
    if constexpr (flags & CQ_NOC_FLAG_LEN) {
        __builtin_riscv_ttrocc_cmdbuf_wr_reg(
            cmd_buf, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_LEN_BYTES_REG_OFFSET / 8, size);
    }
    __builtin_riscv_ttrocc_cmdbuf_wr_reg(
        cmd_buf, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_MCAST_DESTS_REG_OFFSET / 8, ndests);
    if constexpr (send) {
        __builtin_riscv_ttrocc_cmdbuf_issue_trans(cmd_buf);
    }

    if constexpr (update_counter) {
        if constexpr (posted) {
            noc_posted_writes_num_issued[noc] += 1;
        } else {
            noc_nonposted_writes_num_issued[noc] += 1;
            noc_nonposted_writes_acked[noc] += ndests;
        }
    }
}

// --------------------------------------------------------------------------
// Inline-write family: a single 32-bit value carried in the command itself.
// --------------------------------------------------------------------------

template <uint32_t cmd_buf, enum CQNocCmdFlags cmd_flags = CQ_NOC_mkp>
inline __attribute__((always_inline)) void noc_inline_dw_write_init_state(uint32_t noc, uint32_t vc) {
    static_assert(cmd_buf <= 2, "Qsr has 2 complex cmd buffers (0,1) and one simple (2) command buffer");
    (void)noc;
    static_assert(
        (cmd_flags & CQ_NOC_CMD_FLAG_MCAST) == 0, "inline-write multicast is not implemented under ATT (V3 traps)");
    uint64_t misc = CMD_BUF_MISC_WRITE_TRANS | ((cmd_flags & CQ_NOC_CMD_FLAG_POSTED) ? CMD_BUF_MISC_POSTED : 0);

    if constexpr (cmd_buf == 2) {
        __builtin_riscv_ttrocc_scmdbuf_wr_reg(TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_MISC_REG_OFFSET / 8, misc);
        __builtin_riscv_ttrocc_scmdbuf_wr_reg(TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_REQ_VC_REG_OFFSET / 8, vc);
        __builtin_riscv_ttrocc_scmdbuf_wr_reg(
            TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_RESP_VC_REG_OFFSET / 8, NOC_OVERLAY_WR_RESP_VC);
        __builtin_riscv_ttrocc_scmdbuf_wr_reg(
            TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_LEN_BYTES_REG_OFFSET / 8, sizeof(uint32_t));
    } else {
        __builtin_riscv_ttrocc_cmdbuf_wr_reg(cmd_buf, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_MISC_REG_OFFSET / 8, misc);
        __builtin_riscv_ttrocc_cmdbuf_wr_reg(cmd_buf, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_REQ_VC_REG_OFFSET / 8, vc);
        __builtin_riscv_ttrocc_cmdbuf_wr_reg(
            cmd_buf, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_RESP_VC_REG_OFFSET / 8, NOC_OVERLAY_WR_RESP_VC);
        __builtin_riscv_ttrocc_cmdbuf_wr_reg(
            cmd_buf, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_LEN_BYTES_REG_OFFSET / 8, sizeof(uint32_t));
    }
    noc_v3_cq_state[cmd_buf].dest_mcast = false;
}

template <
    uint32_t cmd_buf,
    enum CQNocInlineFlags flags,
    enum CQNocWait wait = CQ_NOC_WAIT,
    enum CQNocSend send = CQ_NOC_SEND>
inline __attribute__((always_inline)) void noc_inline_dw_write_with_state(
    uint32_t noc, uint64_t dst_addr, uint32_t val = 0, uint8_t be = 0xF) {
    static_assert(cmd_buf <= 2, "noc_inline_dw_write_* only supports cmd_buf 0, 1, or 2");
    (void)noc;

    if constexpr (flags & CQ_NOC_INLINE_FLAG_VAL) {
        noc_v3_cq_state[cmd_buf].inline_val = val;
    }
    if constexpr (flags & CQ_NOC_FLAG_NOC) {
        noc_v3_cq_state[cmd_buf].dest_base = noc_v3_state_base_of(dst_addr);
    }
    if constexpr (flags & CQ_NOC_FLAG_DST) {
        noc_v3_cq_state[cmd_buf].dest_local = noc_v3_cq_local_of(dst_addr);
    }
    if constexpr (flags & (CQ_NOC_FLAG_NOC | CQ_NOC_FLAG_DST)) {
        const uint64_t operand = noc_v3_cq_state[cmd_buf].dest_base | noc_v3_cq_state[cmd_buf].dest_local;
        if constexpr (cmd_buf == 2) {
            __builtin_riscv_ttrocc_scmdbuf_wr_reg(
                TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_DEST_ADDR_REG_OFFSET / 8, operand);
        } else {
            __builtin_riscv_ttrocc_cmdbuf_wr_reg(
                cmd_buf, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_DEST_ADDR_REG_OFFSET / 8, operand);
        }
    }
    if constexpr (flags & CQ_NOC_INLINE_FLAG_BE) {
        ASSERT(be == 0xF);
    }
    if constexpr (send) {
        const uint32_t data = (flags & CQ_NOC_INLINE_FLAG_VAL) ? val : noc_v3_cq_state[cmd_buf].inline_val;
        if constexpr (cmd_buf == 2) {
            __builtin_riscv_ttrocc_scmdbuf_issue_inline_trans(data);
        } else {
            __builtin_riscv_ttrocc_cmdbuf_issue_inline_trans(cmd_buf, data);
        }
    }
}
