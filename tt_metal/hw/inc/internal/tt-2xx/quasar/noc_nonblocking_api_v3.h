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
//  - multicast issues trap at runtime until the map-aware rectangle decode
//    lands with the ATT maps.
//
// Shared RoCC command-buffer definitions (register wrappers, MISC/VC values,
// counters, init, barriers) come from noc_cmd_buf_common.h.

#if !defined(NOC_ATT_ENABLED)
#error "NOC API V3 is the ATT-native transport and requires the ATT address backend (NOC_ATT_ENABLED)"
#endif

#if !defined(NOC_ATT_LOCAL_WINDOW_BASE)
#error "The selected ATT configuration must define NOC_ATT_LOCAL_WINDOW_BASE (the self window base address)"
#endif

#include "internal/tt-2xx/quasar/noc_cmd_buf_common.h"

// The full ATT operand for an address in this initiator's own L1.
inline __attribute__((always_inline)) constexpr uint64_t noc_v3_local_operand(uint32_t local_address) {
    return (uint64_t{NOC_ATT_LOCAL_WINDOW_BASE}) | local_address;
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
inline constexpr uint32_t NOC_V3_STATE_CMD_BUFS = 4;
inline uint64_t noc_v3_read_state_base[NOC_V3_STATE_CMD_BUFS] = {};
inline uint64_t noc_v3_write_state_base[NOC_V3_STATE_CMD_BUFS] = {};
// The inline-write pair runs on the simple command buffer only, so one base.
inline uint64_t noc_v3_inline_write_state_base = 0;

// Tripwire for the zero-local-field contract above. The exact local-field
// width is a property of the window the address resolves through, which the
// transport does not know; 20 bits is the smallest local field of any window
// on the configured maps, so this never rejects a legitimate base while
// catching typical nonzero offsets. Map-aware validation belongs to the
// address backend.
inline __attribute__((always_inline)) void noc_v3_check_state_base(uint64_t noc_addr) {
    ASSERT((noc_addr & 0xFFFFFull) == 0);
}

inline __attribute__((always_inline)) void noc_init(uint32_t atomic_ret_val) {
    // The command buffers are programmed by overlay_cmd_buff_init. The ATT
    // enablement adds its bring-up table replay here.
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
    // Multicast needs the map-aware rectangle decode (start address + DEST_COORD
    // extent); it lands with the ATT maps.
    ASSERT(!mcast);

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
    // Multicast needs the map-aware rectangle decode; it lands with the ATT maps.
    ASSERT(!mcast);

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
    // Multicast needs the map-aware rectangle decode; it lands with the ATT maps.
    ASSERT(!mcast);

    uint64_t misc = CMD_BUF_MISC_INLINE_WRITE | CMD_BUF_MISC_BYTE_ENABLE | CMD_BUF_MISC_SRC_INCLUDE |
                    (mcast ? (CMD_BUF_MISC_MULTICAST | CMD_BUF_MISC_LINKED) : 0) | (posted ? CMD_BUF_MISC_POSTED : 0);
    __builtin_riscv_ttrocc_scmdbuf_wr_reg(TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_MISC_REG_OFFSET / 8, misc);

    __builtin_riscv_ttrocc_scmdbuf_wr_reg(TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_REQ_VC_REG_OFFSET / 8, static_vc);
    __builtin_riscv_ttrocc_scmdbuf_wr_reg(
        TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_RESP_VC_REG_OFFSET / 8,
        mcast ? NOC_OVERLAY_MCAST_RESP_VC : NOC_OVERLAY_WR_RESP_VC);

    uint32_t be32 = be << (dest_addr & (NOC_WORD_BYTES - 1));
    __builtin_riscv_ttrocc_scmdbuf_wr_reg(TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_LEN_BYTES_REG_OFFSET / 8, be32);
    __builtin_riscv_ttrocc_scmdbuf_wr_reg(TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_DEST_ADDR_REG_OFFSET / 8, dest_addr);
    __builtin_riscv_ttrocc_scmdbuf_issue_inline_trans(val);

    if constexpr (noc_mode == DM_DEDICATED_NOC) {
        if (posted) {
            noc_posted_writes_num_issued[noc] += 1;
        } else {
            noc_nonposted_writes_num_issued[noc] += 1;
            noc_nonposted_writes_acked[noc] += 1;
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
    // Multicast needs the map-aware rectangle decode; it lands with the ATT maps.
    ASSERT(false);
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
    // Multicast needs the map-aware rectangle decode; it lands with the ATT maps.
    ASSERT(false);
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

    noc_v3_check_state_base(src_noc_addr);
    noc_v3_read_state_base[cmd_buf] = src_noc_addr;

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

    noc_v3_check_state_base(dst_noc_addr);
    noc_v3_write_state_base[cmd_buf] = dst_noc_addr;

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
    uint64_t misc = CMD_BUF_MISC_INLINE_WRITE | CMD_BUF_MISC_BYTE_ENABLE | CMD_BUF_MISC_SRC_INCLUDE |
                    (posted ? CMD_BUF_MISC_POSTED : 0);
    __builtin_riscv_ttrocc_scmdbuf_wr_reg(TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_MISC_REG_OFFSET / 8, misc);
    __builtin_riscv_ttrocc_scmdbuf_wr_reg(TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_REQ_VC_REG_OFFSET / 8, static_vc);
    __builtin_riscv_ttrocc_scmdbuf_wr_reg(
        TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_RESP_VC_REG_OFFSET / 8, NOC_OVERLAY_WR_RESP_VC);
    __builtin_riscv_ttrocc_scmdbuf_wr_reg(TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_DEST_ADDR_REG_OFFSET / 8, dest_addr);
    // Held for update_addr_lo issues: the sticky DEST_ADDR register carries the
    // full operand, so a per-issue local address must be folded into the state
    // base rather than written whole (state-base contract as for the other
    // stateful pairs: zero local field, checked below).
    noc_v3_check_state_base(dest_addr);
    noc_v3_inline_write_state_base = dest_addr;

    uint32_t be32 = be << (dest_addr & (NOC_WORD_BYTES - 1));
    __builtin_riscv_ttrocc_scmdbuf_wr_reg(TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_LEN_BYTES_REG_OFFSET / 8, be32);
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
    __builtin_riscv_ttrocc_scmdbuf_issue_inline_trans(val);

    if constexpr (update_counter) {
        if constexpr (posted) {
            noc_posted_writes_num_issued[noc] += 1;
        } else {
            noc_nonposted_writes_num_issued[noc] += 1;
            noc_nonposted_writes_acked[noc] += 1;
        }
    }
}
