// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include "internal/risc_attribs.h"
#include "noc_parameters.h"
#include "hostdev/dev_msgs.h"
#include "noc_overlay_parameters.h"
#include "api/debug/assert.h"
#include "internal/tt-2xx/quasar/overlay/rocc_instructions.hpp"
#include "internal/tt-2xx/quasar/noc_cmd_buf_common.h"

// ============================================================================

inline __attribute__((always_inline)) void noc_init(uint32_t atomic_ret_val) {
    // TODO: Add ATT configuration here
}

// Expects noc_init to have set on OVERLAY_RD_CMD_BUF:
//   MISC      = CMD_BUF_MISC_READ
//   DEST_COORD = my_xy (local core, read return destination)
//   TR_ID / WR_SENT_TR_ID / TR_ACK_TR_ID = NOC_V2_TRID_STATIC (0)
template <uint8_t noc_mode = DM_DEDICATED_NOC>
inline __attribute__((always_inline)) void ncrisc_noc_fast_read(
    uint32_t noc,
    uint32_t cmd_buf,
    uint64_t src_addr,
    uint32_t dest_addr,
    uint32_t len_bytes,
    uint32_t read_req_vc = NOC_V2_RD_REQ_VC) {
    static_assert(noc_mode != DM_DYNAMIC_NOC, "Quasar does not support DYNAMIC_NOC as it has only 1 NOC");

    __builtin_riscv_ttrocc_cmdbuf_wr_reg(
        cmd_buf, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_REQ_VC_REG_OFFSET / 8, read_req_vc);
    __builtin_riscv_ttrocc_cmdbuf_wr_reg(
        cmd_buf, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_RESP_VC_REG_OFFSET / 8, NOC_V2_RD_RESP_VC);
    __builtin_riscv_ttrocc_cmdbuf_wr_reg(
        cmd_buf, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_SRC_ADDR_REG_OFFSET / 8, (uint32_t)src_addr);
    __builtin_riscv_ttrocc_cmdbuf_wr_reg(
        cmd_buf,
        TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_SRC_COORD_REG_OFFSET / 8,
        (uint32_t)(src_addr >> NOC_ADDR_COORD_SHIFT) & NOC_COORDINATE_MASK);
    __builtin_riscv_ttrocc_cmdbuf_wr_reg(
        cmd_buf, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_DEST_ADDR_REG_OFFSET / 8, dest_addr);
    __builtin_riscv_ttrocc_cmdbuf_wr_reg(
        cmd_buf, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_LEN_BYTES_REG_OFFSET / 8, len_bytes);
    __builtin_riscv_ttrocc_cmdbuf_issue_trans(cmd_buf);

    uint32_t num_packets = len_bytes / NOC_V2_MAX_BYTES_IN_PACKET + ((len_bytes % NOC_V2_MAX_BYTES_IN_PACKET) ? 1 : 0);
    noc_reads_num_issued[noc] += num_packets;
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

    // SRC_COORD (local coordinate) is set once in noc_init for the write cmd buffer.
    // Rebuild MISC per-transaction since mcast/linked/posted can change.
    uint64_t misc = CMD_BUF_MISC_WRITE_TRANS | (linked ? CMD_BUF_MISC_LINKED : 0) |
                    (mcast ? CMD_BUF_MISC_MULTICAST : 0) | (posted ? CMD_BUF_MISC_POSTED : 0);
    __builtin_riscv_ttrocc_cmdbuf_wr_reg(cmd_buf, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_MISC_REG_OFFSET / 8, misc);

    __builtin_riscv_ttrocc_cmdbuf_wr_reg(cmd_buf, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_REQ_VC_REG_OFFSET / 8, vc);
    __builtin_riscv_ttrocc_cmdbuf_wr_reg(
        cmd_buf,
        TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_RESP_VC_REG_OFFSET / 8,
        mcast ? NOC_V2_MCAST_RESP_VC : NOC_V2_WR_RESP_VC);

    if constexpr (use_trid) {
        __builtin_riscv_ttrocc_cmdbuf_wr_reg(cmd_buf, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_TR_ID_REG_OFFSET / 8, trid);
    }

    __builtin_riscv_ttrocc_cmdbuf_wr_reg(
        cmd_buf, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_SRC_ADDR_REG_OFFSET / 8, src_addr);
    __builtin_riscv_ttrocc_cmdbuf_wr_reg(
        cmd_buf, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_DEST_ADDR_REG_OFFSET / 8, (uint32_t)dest_addr);
    __builtin_riscv_ttrocc_cmdbuf_wr_reg(
        cmd_buf,
        TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_DEST_COORD_REG_OFFSET / 8,
        (uint32_t)(dest_addr >> NOC_ADDR_COORD_SHIFT) & NOC_COORDINATE_MASK);
    __builtin_riscv_ttrocc_cmdbuf_wr_reg(
        cmd_buf, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_LEN_BYTES_REG_OFFSET / 8, len_bytes);
    if (mcast) {
        // HW needs MCAST_DESTS to match the number of cores in the (start,end) rectangle
        // so it can track per-destination acks for the multicast.
        __builtin_riscv_ttrocc_cmdbuf_wr_reg(
            cmd_buf, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_MCAST_DESTS_REG_OFFSET / 8, num_dests);
    }
    __builtin_riscv_ttrocc_cmdbuf_issue_trans(cmd_buf);

    if constexpr (update_counter) {
        uint32_t num_packets =
            len_bytes / NOC_V2_MAX_BYTES_IN_PACKET + ((len_bytes % NOC_V2_MAX_BYTES_IN_PACKET) ? 1 : 0);
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

    // Always nonposted, always src_include (loopback)
    uint64_t misc = CMD_BUF_MISC_WRITE_TRANS | CMD_BUF_MISC_SRC_INCLUDE | (linked ? CMD_BUF_MISC_LINKED : 0) |
                    (mcast ? CMD_BUF_MISC_MULTICAST : 0);
    __builtin_riscv_ttrocc_cmdbuf_wr_reg(cmd_buf, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_MISC_REG_OFFSET / 8, misc);

    __builtin_riscv_ttrocc_cmdbuf_wr_reg(cmd_buf, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_REQ_VC_REG_OFFSET / 8, vc);
    __builtin_riscv_ttrocc_cmdbuf_wr_reg(
        cmd_buf,
        TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_RESP_VC_REG_OFFSET / 8,
        mcast ? NOC_V2_MCAST_RESP_VC : NOC_V2_WR_RESP_VC);

    __builtin_riscv_ttrocc_cmdbuf_wr_reg(
        cmd_buf, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_SRC_ADDR_REG_OFFSET / 8, src_addr);
    __builtin_riscv_ttrocc_cmdbuf_wr_reg(
        cmd_buf, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_DEST_ADDR_REG_OFFSET / 8, (uint32_t)dest_addr);
    __builtin_riscv_ttrocc_cmdbuf_wr_reg(
        cmd_buf,
        TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_DEST_COORD_REG_OFFSET / 8,
        (uint32_t)(dest_addr >> NOC_ADDR_COORD_SHIFT) & NOC_COORDINATE_MASK);
    __builtin_riscv_ttrocc_cmdbuf_wr_reg(
        cmd_buf, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_LEN_BYTES_REG_OFFSET / 8, len_bytes);
    if (mcast) {
        // HW needs MCAST_DESTS to match the number of cores in the (start,end) rectangle
        // so it can track per-destination acks for the multicast.
        __builtin_riscv_ttrocc_cmdbuf_wr_reg(
            cmd_buf, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_MCAST_DESTS_REG_OFFSET / 8, num_dests);
    }
    __builtin_riscv_ttrocc_cmdbuf_issue_trans(cmd_buf);

    if constexpr (noc_mode == DM_DEDICATED_NOC) {
        uint32_t num_packets =
            len_bytes / NOC_V2_MAX_BYTES_IN_PACKET + ((len_bytes % NOC_V2_MAX_BYTES_IN_PACKET) ? 1 : 0);
        noc_nonposted_writes_num_issued[noc] += num_packets;
        noc_nonposted_writes_acked[noc] += num_dests * num_packets;
    }
}

template <uint8_t noc_mode = DM_DEDICATED_NOC, bool use_vc = false>
inline __attribute__((always_inline)) void ncrisc_noc_fast_read_any_len(
    uint32_t noc,
    uint32_t cmd_buf,
    uint64_t src_addr,
    uint32_t dest_addr,
    uint32_t len_bytes,
    uint32_t read_req_vc = NOC_V2_RD_REQ_VC) {
    static_assert(noc_mode != DM_DYNAMIC_NOC, "Quasar does not support DYNAMIC_NOC as it has only 1 NOC");
    // Overlay handles packetization via MAX_BYTES_IN_PACKET register; no software chunking needed.
    ncrisc_noc_fast_read<noc_mode>(noc, cmd_buf, src_addr, dest_addr, len_bytes, read_req_vc);
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

    uint64_t misc = CMD_BUF_MISC_INLINE_WRITE | CMD_BUF_MISC_BYTE_ENABLE | CMD_BUF_MISC_SRC_INCLUDE |
                    (mcast ? (CMD_BUF_MISC_MULTICAST | CMD_BUF_MISC_LINKED) : 0) | (posted ? CMD_BUF_MISC_POSTED : 0);
    __builtin_riscv_ttrocc_scmdbuf_wr_reg(TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_MISC_REG_OFFSET / 8, misc);

    __builtin_riscv_ttrocc_scmdbuf_wr_reg(TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_REQ_VC_REG_OFFSET / 8, static_vc);
    __builtin_riscv_ttrocc_scmdbuf_wr_reg(
        TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_RESP_VC_REG_OFFSET / 8, mcast ? NOC_V2_MCAST_RESP_VC : NOC_V2_WR_RESP_VC);

    uint32_t be32 = be << (dest_addr & (NOC_WORD_BYTES - 1));
    __builtin_riscv_ttrocc_scmdbuf_wr_reg(TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_LEN_BYTES_REG_OFFSET / 8, be32);
    __builtin_riscv_ttrocc_scmdbuf_wr_reg(
        TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_DEST_ADDR_REG_OFFSET / 8, (uint32_t)dest_addr);
    __builtin_riscv_ttrocc_scmdbuf_wr_reg(
        TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_DEST_COORD_REG_OFFSET / 8,
        (uint32_t)(dest_addr >> NOC_ADDR_COORD_SHIFT) & NOC_COORDINATE_MASK);
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

    uint64_t misc = CMD_BUF_MISC_INLINE_WRITE | CMD_BUF_MISC_BYTE_ENABLE | CMD_BUF_MISC_SRC_INCLUDE |
                    (mcast ? (CMD_BUF_MISC_MULTICAST | CMD_BUF_MISC_LINKED) : 0) | (posted ? CMD_BUF_MISC_POSTED : 0);
    __builtin_riscv_ttrocc_scmdbuf_wr_reg(TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_MISC_REG_OFFSET / 8, misc);

    __builtin_riscv_ttrocc_scmdbuf_wr_reg(TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_REQ_VC_REG_OFFSET / 8, static_vc);
    __builtin_riscv_ttrocc_scmdbuf_wr_reg(
        TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_RESP_VC_REG_OFFSET / 8, mcast ? NOC_V2_MCAST_RESP_VC : NOC_V2_WR_RESP_VC);

    uint32_t be32 = be << (dest_addr & (NOC_WORD_BYTES - 1));
    __builtin_riscv_ttrocc_scmdbuf_wr_reg(TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_LEN_BYTES_REG_OFFSET / 8, be32);
    __builtin_riscv_ttrocc_scmdbuf_wr_reg(
        TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_DEST_ADDR_REG_OFFSET / 8, (uint32_t)dest_addr);
    __builtin_riscv_ttrocc_scmdbuf_wr_reg(
        TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_DEST_COORD_REG_OFFSET / 8,
        (uint32_t)(dest_addr >> NOC_ADDR_COORD_SHIFT) & NOC_COORDINATE_MASK);
    if (mcast) {
        // HW needs MCAST_DESTS to match the number of cores in the (start,end) rectangle
        // so it can track per-destination acks for the multicast.
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
        TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_RESP_VC_REG_OFFSET / 8, NOC_V2_WR_RESP_VC);
    if constexpr (program_ret_addr) {
        __builtin_riscv_ttrocc_scmdbuf_wr_reg(
            TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_SRC_ADDR_REG_OFFSET / 8, atomic_ret_val);
    }
    __builtin_riscv_ttrocc_scmdbuf_wr_reg(
        TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_DEST_ADDR_REG_OFFSET / 8, (uint32_t)(addr & 0xFFFFFFFF));
    __builtin_riscv_ttrocc_scmdbuf_wr_reg(
        TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_DEST_COORD_REG_OFFSET / 8,
        (uint32_t)(addr >> NOC_ADDR_COORD_SHIFT) & NOC_COORDINATE_MASK);
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
    uint64_t misc = CMD_BUF_MISC_ATOMIC_TRANS | CMD_BUF_MISC_SRC_INCLUDE | CMD_BUF_MISC_MULTICAST |
                    (posted ? CMD_BUF_MISC_POSTED : 0) | (linked ? CMD_BUF_MISC_LINKED : 0);
    __builtin_riscv_ttrocc_scmdbuf_wr_reg(TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_MISC_REG_OFFSET / 8, misc);
    __builtin_riscv_ttrocc_scmdbuf_wr_reg(TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_REQ_VC_REG_OFFSET / 8, vc);
    __builtin_riscv_ttrocc_scmdbuf_wr_reg(
        TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_RESP_VC_REG_OFFSET / 8, NOC_V2_MCAST_RESP_VC);
    __builtin_riscv_ttrocc_scmdbuf_wr_reg(
        TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_DEST_ADDR_REG_OFFSET / 8, (uint32_t)(addr & 0xFFFFFFFF));
    __builtin_riscv_ttrocc_scmdbuf_wr_reg(
        TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_DEST_COORD_REG_OFFSET / 8,
        (uint32_t)(addr >> NOC_ADDR_COORD_SHIFT) & NOC_COORDINATE_MASK);
    uint64_t at_len =
        NOC_AT_INS(NOC_AT_INS_INCR_GET) | NOC_AT_WRAP(wrap) | NOC_AT_IND_32((addr >> 2) & 0x3) | NOC_AT_IND_32_SRC(0);
    __builtin_riscv_ttrocc_scmdbuf_wr_reg(TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_LEN_BYTES_REG_OFFSET / 8, at_len);
    __builtin_riscv_ttrocc_scmdbuf_wr_reg(
        TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_INLINE_DATA_REG_OFFSET / 8, (uint64_t)incr);
    // HW needs MCAST_DESTS to match the number of cores in the (start,end) rectangle
    // so it can track per-destination acks for the multicast atomic.
    __builtin_riscv_ttrocc_scmdbuf_wr_reg(TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_MCAST_DESTS_REG_OFFSET / 8, num_dests);
    __builtin_riscv_ttrocc_scmdbuf_issue_trans();

    if (!posted) {
        noc_nonposted_atomics_acked[noc] += num_dests;
    }
}

// issue noc reads while wait for outstanding transactions done
template <uint8_t noc_mode = DM_DEDICATED_NOC, bool skip_ptr_update = false, bool skip_cmdbuf_chk = false>
inline __attribute__((always_inline)) void ncrisc_noc_fast_read_with_transaction_id(
    uint32_t noc, uint32_t cmd_buf, uint32_t src_base_addr, uint32_t src_addr, uint32_t dest_addr, uint32_t trid) {
    static_assert(noc_mode != DM_DYNAMIC_NOC, "Quasar does not support DYNAMIC_NOC as it has only 1 NOC");
    uint32_t src_addr_;
    src_addr_ = src_base_addr + src_addr;

    while (NOC_STATUS_READ_REG(noc, NIU_MST_REQS_OUTSTANDING_ID(trid)) > ((NOC_MAX_TRANSACTION_ID + 1) / 2));

    __builtin_riscv_ttrocc_cmdbuf_wr_reg(
        cmd_buf, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_DEST_ADDR_REG_OFFSET / 8, dest_addr);
    __builtin_riscv_ttrocc_cmdbuf_wr_reg(
        cmd_buf, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_SRC_ADDR_REG_OFFSET / 8, src_addr_);
    __builtin_riscv_ttrocc_cmdbuf_issue_trans(cmd_buf);
    if constexpr (!skip_ptr_update) {
        noc_reads_num_issued[noc] += 1;
    }
}

// clang-format off
/**
 * Sets the stateful registers for an asynchronous read from a specified source node located at NOC
 * coordinates (x,y) at a local address (encoded as a uint64_t using \a
 * get_noc_addr function). This function is used to set up the state for
 * \a ncrisc_noc_read_with_state, which will issue the actual read request.
 *
 * The source node can be either a DRAM bank, a Tensix core or a PCIe controller.
 *
 * Return value: None
 *
 * | Argument                        | Description                                        | Data type | Valid range                                              | required |
 * |---------------------------------|----------------------------------------------------|-----------|----------------------------------------------------------|----------|
 * | noc                             | Which NOC to use for the transaction               | uint32_t  | 0 or 1                                                   | True     |
 * | cmd_buf                         | Which command buffer to use for the transaction    | uint32_t  | 0 - 3                                                    | True     |
 * | src_noc_addr                    | Encoding of the source NOC location (x,y)+address  | uint64_t  | Results of \a get_noc_addr calls                         | True     |
 * | len_bytes                       | Size of the transaction in bytes.                  | uint32_t  | 0..1 MB                                                  | False    |
 * | vc                              | Which VC to use for the transaction                | uint32_t  | 0 - 3                                                    | False    |
 * | noc_mode (template parameter)   | NOC mode for the transaction                       | uint8_t   | DM_DEDICATED_NOC, DM_DYNAMIC_NOC or DM_INVALID_NOC (0-2) | False    |
 * | one_packet (template parameter) | Whether transaction size is <= NOC_V2_MAX_BYTES_IN_PACKET  | bool      | true or false                                            | False    |
 * | use_vc (template parameter)     | Use custom VC, enables vc parameter                | bool      | true or false                                            | False    |
 */
// clang-format on
template <uint8_t noc_mode = DM_DEDICATED_NOC, bool one_packet = false, bool use_vc = false>
inline __attribute__((always_inline)) void ncrisc_noc_read_set_state(
    uint32_t noc, uint32_t cmd_buf, uint64_t src_noc_addr, uint32_t len_bytes = 0, const uint32_t vc = 0) {
    static_assert(noc_mode != DM_DYNAMIC_NOC, "Quasar does not support DYNAMIC_NOC as it has only 1 NOC");

    if constexpr (use_vc) {
        __builtin_riscv_ttrocc_cmdbuf_wr_reg(cmd_buf, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_REQ_VC_REG_OFFSET / 8, vc);
    }

    __builtin_riscv_ttrocc_cmdbuf_wr_reg(
        cmd_buf,
        TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_SRC_COORD_REG_OFFSET / 8,
        (uint32_t)(src_noc_addr >> NOC_ADDR_COORD_SHIFT) & NOC_COORDINATE_MASK);

    if constexpr (one_packet) {
        __builtin_riscv_ttrocc_cmdbuf_wr_reg(
            cmd_buf, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_LEN_BYTES_REG_OFFSET / 8, len_bytes);
    }
}

// clang-format off
/**
 * Initiates an asynchronous read from a specified source node located at NOC
 * coordinates (x,y) at a local address (encoded as a uint64_t using \a
 * get_noc_addr function) for a single packet with size <= NOC_V2_MAX_BYTES_IN_PACKET (i.e. maximum packet size).
 * This function must be preceded by a call to \a ncrisc_noc_read_set_state.
 * This function is used to issue the actual read request after the state has been set up.
 *
 * Return value: None
 *
 * | Argument                            | Description                                        | Data type | Valid range                                              | required |
 * |-------------------------------------|----------------------------------------------------|-----------|----------------------------------------------------------|----------|
 * | noc                                 | Which NOC to use for the transaction               | uint32_t  | 0 or 1                                                   | True     |
 * | cmd_buf                             | Which command buffer to use for the transaction    | uint32_t  | 0 - 3                                                    | True     |
 * | src_local_addr                      | Address in local L1 memory on source core          | uint32_t  | 0..1 MB                                                  | True     |
 * | dst_local_addr                      | Address in local L1 memory on destination core     | uint32_t  | 0..1 MB                                                  | True     |
 * | len_bytes                           | Size of transaction in bytes                       | uint32_t  | 0..1 MB                                                  | False    |
 * | noc_mode (template parameter)       | NOC mode for the transaction                       | uint8_t   | DM_DEDICATED_NOC, DM_DYNAMIC_NOC or DM_INVALID_NOC (0-2) | False    |
 * | inc_num_issued (template parameter) | Increment enable for transaction issued counters   | bool      | true or false                                            | False    |
 * | one_packet (template parameter)     | Whether transaction size is <= NOC_V2_MAX_BYTES_IN_PACKET  | bool      | true or false                                            | False    |
 */
// clang-format on
template <uint8_t noc_mode = DM_DEDICATED_NOC, bool inc_num_issued = true, bool one_packet = false>
inline __attribute__((always_inline)) void ncrisc_noc_read_with_state(
    uint32_t noc, uint32_t cmd_buf, uint32_t src_local_addr, uint32_t dst_local_addr, uint32_t len_bytes = 0) {
    static_assert(noc_mode != DM_DYNAMIC_NOC, "Quasar does not support DYNAMIC_NOC as it has only 1 NOC");

    __builtin_riscv_ttrocc_cmdbuf_wr_reg(
        cmd_buf, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_SRC_ADDR_REG_OFFSET / 8, src_local_addr);
    __builtin_riscv_ttrocc_cmdbuf_wr_reg(
        cmd_buf, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_DEST_ADDR_REG_OFFSET / 8, dst_local_addr);
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
                len_bytes / NOC_V2_MAX_BYTES_IN_PACKET + ((len_bytes % NOC_V2_MAX_BYTES_IN_PACKET) ? 1 : 0);
            noc_reads_num_issued[noc] += num_packets;
        }
    }
}

// clang-format off
/**
 * Initiates an asynchronous read for all transaction sizes.
 * Refer to \a ncrisc_noc_read_with_state for more details.
 *
 * Return value: None
 *
 * | Argument                            | Description                                        | Data type | Valid range                                              | required |
 * |-------------------------------------|----------------------------------------------------|-----------|----------------------------------------------------------|----------|
 * | noc                                 | Which NOC to use for the transaction               | uint32_t  | 0 or 1                                                   | True     |
 * | cmd_buf                             | Which command buffer to use for the transaction    | uint32_t  | 0 - 3                                                    | True     |
 * | src_local_addr                      | Address in local L1 memory on source core          | uint32_t  | 0..1 MB                                                  | True     |
 * | dst_local_addr                      | Address in local L1 memory on destination core     | uint32_t  | 0..1 MB                                                  | True     |
 * | len_bytes                           | Size of transaction in bytes                       | uint32_t  | 0..1 MB                                                  | True     |
 * | noc_mode (template parameter)       | NOC mode for the transaction                       | uint8_t   | DM_DEDICATED_NOC, DM_DYNAMIC_NOC or DM_INVALID_NOC (0-2) | False    |
 * | inc_num_issued (template parameter) | Increment enable for transaction issued counters   | bool      | true or false                                            | False    |
 */
// clang-format on
template <uint8_t noc_mode = DM_DEDICATED_NOC, bool inc_num_issued = true>
inline __attribute__((always_inline)) void ncrisc_noc_read_any_len_with_state(
    uint32_t noc, uint32_t cmd_buf, uint32_t src_local_addr, uint32_t dst_local_addr, uint32_t len_bytes) {
    static_assert(noc_mode != DM_DYNAMIC_NOC, "Quasar does not support DYNAMIC_NOC as it has only 1 NOC");
    // Overlay handles packetization via MAX_BYTES_IN_PACKET register; no software chunking needed.
    ncrisc_noc_read_with_state<noc_mode, inc_num_issued>(noc, cmd_buf, src_local_addr, dst_local_addr, len_bytes);
}

// clang-format off
/**
 * Sets the stateful registers for an asynchronous write to a specified destination node located at
 * NOC coordinates (x,y) at a local address (encoded as a uint64_t using \a
 * get_noc_addr function). This function is used to set up the state for
 * \a ncrisc_noc_write_with_state, which will issue the actual
 * write request.
 *
 * The destination node can be either a DRAM bank, a Tensix core or a PCIe controller.
 *
 * Return value: None
 *
 * | Argument                        | Description                                              | Data type | Valid range                      | required |
 * |---------------------------------|----------------------------------------------------------|-----------|----------------------------------|----------|
 * | noc                             | NOC to use for the transaction                           | uint32_t  | 0 or 1                           | True     |
 * | cmd_buf                         | Command buffer to use for the transaction                | uint32_t  | 0 - 3                            | True     |
 * | dst_noc_addr                    | Encoding of the destination NOC location (x,y)+address   | uint64_t  | Results of \a get_noc_addr calls | True     |
 * | len_bytes                       | Size of the transaction in bytes.                        | uint32_t  | 0..1 MB                          | False    |
 * | vc                              | Which VC to use for the transaction                      | uint32_t  | 0 - 3                            | False    |
 * | posted (template parameter)     | Whether the transaction is posted (i.e. no ack required) | bool      | true or false                    | False    |
 * | one_packet (template parameter) | Whether transaction size is <= NOC_V2_MAX_BYTES_IN_PACKET        | bool      | true or false                    | False    |
 */
// clang-format on
template <bool posted = false, bool one_packet = false>
inline __attribute__((always_inline)) void ncrisc_noc_write_set_state(
    uint32_t noc, uint32_t cmd_buf, uint64_t dst_noc_addr, uint32_t len_bytes = 0, const uint32_t vc = 0) {
    // MISC: write, posted flag from template param. SRC_COORD set in noc_init.
    uint64_t misc = CMD_BUF_MISC_WRITE_TRANS | CMD_BUF_MISC_SRC_INCLUDE | (posted ? CMD_BUF_MISC_POSTED : 0);
    __builtin_riscv_ttrocc_cmdbuf_wr_reg(cmd_buf, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_MISC_REG_OFFSET / 8, misc);
    __builtin_riscv_ttrocc_cmdbuf_wr_reg(cmd_buf, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_REQ_VC_REG_OFFSET / 8, vc);
    __builtin_riscv_ttrocc_cmdbuf_wr_reg(
        cmd_buf, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_RESP_VC_REG_OFFSET / 8, NOC_V2_WR_RESP_VC);

    // Set remote destination coordinate
    __builtin_riscv_ttrocc_cmdbuf_wr_reg(
        cmd_buf,
        TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_DEST_COORD_REG_OFFSET / 8,
        (uint32_t)(dst_noc_addr >> NOC_ADDR_COORD_SHIFT) & NOC_COORDINATE_MASK);

    if constexpr (one_packet) {
        __builtin_riscv_ttrocc_cmdbuf_wr_reg(
            cmd_buf, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_LEN_BYTES_REG_OFFSET / 8, len_bytes);
    }
}

// clang-format off
/**
 * Initiates an asynchronous write to a specified destination node located at
 * NOC coordinates (x,y) at a local address (encoded as a uint64_t using \a
 * get_noc_addr function). This function must be preceded by a call to
 * \a ncrisc_noc_write_set_state. This function is used to issue the actual
 * write request after the state has been set up.
 *
 * Return value: None
 *
 * | Argument                            | Description                                              | Data type | Valid range                                              | required |
 * |-------------------------------------|----------------------------------------------------------|-----------|----------------------------------------------------------|----------|
 * | noc                                 | NOC to use for the transaction                           | uint32_t  | 0 or 1                                                   | True     |
 * | cmd_buf                             | Command buffer to use for the transaction                | uint32_t  | 0 - 3                                                    | True     |
 * | src_local_addr                      | Address in local L1 memory on source core                | uint32_t  | 0..1 MB                                                  | True     |
 * | dst_local_addr                      | Address in local L1 memory on destination core           | uint32_t  | 0..1 MB                                                  | True     |
 * | len_bytes                           | Size of transaction in bytes                             | uint32_t  | 0..1 MB                                                  | False    |
 * | noc_mode (template parameter)       | NOC mode for the transaction                             | uint8_t   | DM_DEDICATED_NOC, DM_DYNAMIC_NOC or DM_INVALID_NOC (0-2) | False    |
 * | posted (template parameter)         | Whether the transaction is posted (i.e. no ack required) | bool      | true or false                                            | False    |
 * | update_counter (template parameter) | Whether to increment write counters                      | bool      | true or false                                            | False    |
 * | one_packet (template parameter)     | Whether transaction size is <= NOC_V2_MAX_BYTES_IN_PACKET        | bool      | true or false                                            | False    |
 */
// clang-format on
template <uint8_t noc_mode = DM_DEDICATED_NOC, bool posted = false, bool update_counter = true, bool one_packet = false>
inline __attribute__((always_inline)) void ncrisc_noc_write_with_state(
    uint32_t noc, uint32_t cmd_buf, uint32_t src_local_addr, uint32_t dst_local_addr, uint32_t len_bytes = 0) {
    static_assert(noc_mode != DM_DYNAMIC_NOC, "Quasar does not support DYNAMIC_NOC as it has only 1 NOC");

    __builtin_riscv_ttrocc_cmdbuf_wr_reg(
        cmd_buf, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_SRC_ADDR_REG_OFFSET / 8, src_local_addr);
    __builtin_riscv_ttrocc_cmdbuf_wr_reg(
        cmd_buf, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_DEST_ADDR_REG_OFFSET / 8, dst_local_addr);
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
                len_bytes / NOC_V2_MAX_BYTES_IN_PACKET + ((len_bytes % NOC_V2_MAX_BYTES_IN_PACKET) ? 1 : 0);
            if constexpr (posted) {
                noc_posted_writes_num_issued[noc] += num_packets;
            } else {
                noc_nonposted_writes_num_issued[noc] += num_packets;
                noc_nonposted_writes_acked[noc] += num_packets;
            }
        }
    }
}

// clang-format off
/**
 * Initiates an asynchronous write for all transaction sizes.
 * Refer to \a ncrisc_noc_write_with_state for more details.
 *
 * Return value: None
 *
 * | Argument                            | Description                                              | Data type | Valid range                                              | required |
 * |-------------------------------------|----------------------------------------------------------|-----------|----------------------------------------------------------|----------|
 * | noc                                 | NOC to use for the transaction                           | uint32_t  | 0 or 1                                                   | True     |
 * | cmd_buf                             | Command buffer to use for the transaction                | uint32_t  | 0 - 3                                                    | True     |
 * | src_local_addr                      | Address in local L1 memory on source core                | uint32_t  | 0..1 MB                                                  | True     |
 * | dst_local_addr                      | Address in local L1 memory on destination core           | uint32_t  | 0..1 MB                                                  | True     |
 * | len_bytes                           | Size of transaction in bytes                             | uint32_t  | 0..1 MB                                                  | True     |
 * | noc_mode (template parameter)       | NOC mode for the transaction                             | uint8_t   | DM_DEDICATED_NOC, DM_DYNAMIC_NOC or DM_INVALID_NOC (0-2) | False    |
 * | posted (template parameter)         | Whether the transaction is posted (i.e. no ack required) | bool      | true or false                                            | False    |
 * | update_counter (template parameter) | Whether to increment write counters                      | bool      | true or false                                            | False    |
 */
// clang-format on
template <uint8_t noc_mode = DM_DEDICATED_NOC, bool posted = false, bool update_counter = true>
inline __attribute__((always_inline)) void ncrisc_noc_write_any_len_with_state(
    uint32_t noc, uint32_t cmd_buf, uint32_t src_local_addr, uint32_t dst_local_addr, uint32_t len_bytes) {
    static_assert(noc_mode != DM_DYNAMIC_NOC, "Quasar does not support DYNAMIC_NOC as it has only 1 NOC");
    // Overlay handles packetization via MAX_BYTES_IN_PACKET register; no software chunking needed.
    ncrisc_noc_write_with_state<noc_mode, posted, update_counter>(
        noc, cmd_buf, src_local_addr, dst_local_addr, len_bytes);
}

// clang-format off
/**
 * Sets the stateful registers for an inline write of a 32-bit value to a NOC destination.
 * This function is used to set up the state for \a noc_fast_write_dw_inline_with_state, which will issue the actual
 * write request. The 32-bit value and part of the destination address can be set later in \a noc_fast_write_dw_inline_with_state.
 *
 * The destination node can be either a Tensix core+L1 memory
 * address or a PCIe controller; This API does not support DRAM addresses.
 *
 * Note: On Blackhole, this API can only write to stream registers, writing to L1 will cause hangs!
 *
 * Return value: None
 *
 * | Argument                     | Description                                            | Type     | Valid Range                      | Required |
 * |------------------------------|--------------------------------------------------------|----------|----------------------------------|----------|
 * | noc                          | NOC to use for the transaction                         | uint32_t | 0 or 1                           | True     |
 * | cmd_buf                      | Command buffer to use for the transaction              | uint32_t | 0 - 3                            | True     |
 * | dest_addr                    | Encoding of the destination NOC location (x,y)+address | uint64_t | Results of \a get_noc_addr calls | True     |
 * | be                           | Byte-enable                                            | uint32_t | 0x1-0xF                          | True     |
 * | static_vc                    | VC to use for the transaction                          | uint32_t | 0 - 3 (Unicast VCs)              | True     |
 * | val                          | The value to be written                                | uint32_t | Any uint32_t value               | False    |
 * | posted (template parameter)  | Whether the call is posted (i.e. ack requirement)      | bool     | true or false                    | False    |
 * | set_val (template parameter) | Whether to set the value for the write here            | bool     | true or false                    | False    |
 */
// clang-format on
template <bool posted = false, bool set_val = false>
inline __attribute__((always_inline)) void noc_fast_write_dw_inline_set_state(
    uint32_t noc, uint32_t cmd_buf, uint64_t dest_addr, uint32_t be, uint32_t static_vc, uint32_t val = 0) {
    uint64_t misc = CMD_BUF_MISC_INLINE_WRITE | CMD_BUF_MISC_BYTE_ENABLE | CMD_BUF_MISC_SRC_INCLUDE |
                    (posted ? CMD_BUF_MISC_POSTED : 0);
    __builtin_riscv_ttrocc_scmdbuf_wr_reg(TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_MISC_REG_OFFSET / 8, misc);
    __builtin_riscv_ttrocc_scmdbuf_wr_reg(TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_REQ_VC_REG_OFFSET / 8, static_vc);
    __builtin_riscv_ttrocc_scmdbuf_wr_reg(
        TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_RESP_VC_REG_OFFSET / 8, NOC_V2_WR_RESP_VC);
    __builtin_riscv_ttrocc_scmdbuf_wr_reg(
        TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_DEST_ADDR_REG_OFFSET / 8, (uint32_t)dest_addr);
    __builtin_riscv_ttrocc_scmdbuf_wr_reg(
        TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_DEST_COORD_REG_OFFSET / 8,
        (uint32_t)(dest_addr >> NOC_ADDR_COORD_SHIFT) & NOC_COORDINATE_MASK);

    uint32_t be32 = be << (dest_addr & (NOC_WORD_BYTES - 1));
    __builtin_riscv_ttrocc_scmdbuf_wr_reg(TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_LEN_BYTES_REG_OFFSET / 8, be32);
}

// clang-format off
/**
 * Initiates an inline write of a 32-bit value to a NOC destination.
 * This function must be preceded by a call to \a noc_fast_write_dw_inline_set_state.
 * This function is used to issue the actual write request after the state has been set up.
 * The 32-bit value and part of the destination address can also be set in this API
 * (Either hi or lo address should be getting updated).
 *
 * The destination node can be either a Tensix core+L1 memory
 * address or a PCIe controller; This API does not support DRAM addresses.
 *
 * Note: On Blackhole, this API can only write to stream registers, writing to L1 will cause hangs!
 *
 * Return value: None
 *
 * | Argument                                   | Description                                            | Type     | Valid Range                      | Required |
 * |--------------------------------------------|--------------------------------------------------------|----------|----------------------------------|----------|
 * | noc                                        | NOC to use for the transaction                         | uint32_t | 0 or 1                           | True     |
 * | cmd_buf                                    | Command buffer to use for the transaction              | uint32_t | 0 - 3                            | True     |
 * | val                                        | The value to be written                                | uint32_t | Any uint32_t value               | False    |
 * | dest_addr                                  | Encoding of the destination NOC location (x,y)+address | uint64_t | Results of \a get_noc_addr calls | False    |
 * | update_addr_lo (template parameter)        | Whether to update the lower 32 bits of the address     | bool     | true or false                    | False    |
 * | update_addr_hi (template parameter)        | Whether to update the upper 32 bits of the address     | bool     | true or false                    | False    |
 * | update_val (template parameter)            | Whether to set the value to be written                 | bool     | true or false                    | False    |
 * | posted (template parameter)                | Whether the call is posted (i.e. ack requirement)      | bool     | true or false                    | False    |
 * | update_counter (template parameter)        | Whether to update the write counters                   | bool     | true or false                    | False    |
 */
// clang-format on
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

    if constexpr (update_addr_lo) {
        __builtin_riscv_ttrocc_scmdbuf_wr_reg(TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_DEST_ADDR_REG_OFFSET / 8, dest_addr);
    } else if constexpr (update_addr_hi) {
        __builtin_riscv_ttrocc_scmdbuf_wr_reg(
            TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_DEST_COORD_REG_OFFSET / 8, dest_addr);
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

// clang-format off
/**
 * The stateful NOC commands provide granular control over NOC register programming by writing
 * only a subset of registers for each transaction. This approach leverages the fact that many
 * transactions re-use certain values (e.g. length, coordinates) while varying others.
 *
 * This design provides significant advantages over previous stateful APIs:
 * - Fine-grained control: Users can specify exactly which registers to update per transaction
 * - Better optimization: Avoid unnecessary register writes for unchanged values
 * - Flexible transaction patterns: Support complex sequences with selective updates
 * - Performance benefits: Reduce NOC register write overhead for repetitive operations
 *
 * The flags parameter uses a bitmask approach to specify which registers to program.
 * Making template functions with a long list of booleans makes understanding what registers
 * are being set tedious. This is an attempt to pack that data in a way thats ~easy to visually parse.
 *
 * S/s: write, do not write to src address register (NOC_TARG_ADDR_LO)
 * N/n: write, do not write to noc coordinates register (NOC_RET_ADDR_COORDINATE)
 * D/d: write, do not write to dst address register (NOC_RET_ADDR_LO)
 * L/l: write, do not write to length register (NOC_AT_LEN)
 *
 * M/m: write, do not write to multicast register (NOC_CMD_BRCST_PACKET)
 * K/k: write, do not write to linked register (NOC_CMD_VC_LINKED)
 * P/p: write, do not write to posted register (NOC_CMD_RESP_MARKED)
 *
 * V/v: write, do not write to value register (NOC_AT_DATA)
 * B/b: write, do not write to byte-enable register (NOC_AT_LEN)
 *
 * WAIT/wait: wait, do not wait for command buffer readiness (NOC_CMD_CTRL)
 * SEND/send: send, do not send the transaction immediately (NOC_CTRL_SEND_REQ)
 */
// clang-format on
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

// Wormhole API compatibility wrapper for stateful inline direct writes.
template <uint32_t cmd_buf, enum CQNocCmdFlags cmd_flags = CQ_NOC_mkp>
inline __attribute__((always_inline)) void noc_inline_dw_write_init_state(uint32_t noc, uint32_t vc) {
    static_assert(cmd_buf <= 2, "Qsr has 2 complex cmd buffers (0,1) and one simple (2) command buffer");
    (void)noc;
    uint64_t misc = CMD_BUF_MISC_INLINE_WRITE | CMD_BUF_MISC_BYTE_ENABLE | CMD_BUF_MISC_SRC_INCLUDE |
                    ((cmd_flags & CQ_NOC_CMD_FLAG_MCAST) ? (CMD_BUF_MISC_MULTICAST | CMD_BUF_MISC_LINKED) : 0) |
                    ((cmd_flags & CQ_NOC_CMD_FLAG_POSTED) ? CMD_BUF_MISC_POSTED : 0);

    if constexpr (cmd_buf == 2) {
        __builtin_riscv_ttrocc_scmdbuf_wr_reg(TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_MISC_REG_OFFSET / 8, misc);
        __builtin_riscv_ttrocc_scmdbuf_wr_reg(TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_REQ_VC_REG_OFFSET / 8, vc);
        __builtin_riscv_ttrocc_scmdbuf_wr_reg(
            TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_RESP_VC_REG_OFFSET / 8,
            (cmd_flags & CQ_NOC_CMD_FLAG_MCAST) ? NOC_V2_MCAST_RESP_VC : NOC_V2_WR_RESP_VC);
    } else {
        static_assert(cmd_buf <= 1, "normal cmdbuf operations are only valid for cmd_buf 0 or 1");
        __builtin_riscv_ttrocc_cmdbuf_wr_reg(cmd_buf, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_MISC_REG_OFFSET / 8, misc);
        __builtin_riscv_ttrocc_cmdbuf_wr_reg(cmd_buf, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_REQ_VC_REG_OFFSET / 8, vc);
        __builtin_riscv_ttrocc_cmdbuf_wr_reg(
            cmd_buf,
            TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_RESP_VC_REG_OFFSET / 8,
            (cmd_flags & CQ_NOC_CMD_FLAG_MCAST) ? NOC_V2_MCAST_RESP_VC : NOC_V2_WR_RESP_VC);
    }
}

// Wormhole API compatibility wrapper for stateful inline direct writes.
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
        if constexpr (cmd_buf == 2) {
            __builtin_riscv_ttrocc_scmdbuf_wr_reg(TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_INLINE_DATA_REG_OFFSET / 8, val);
        } else {
            static_assert(cmd_buf <= 1, "normal cmdbuf operations are only valid for cmd_buf 0 or 1");
            __builtin_riscv_ttrocc_cmdbuf_wr_reg(
                cmd_buf, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_INLINE_DATA_REG_OFFSET / 8, val);
        }
    }
    if constexpr (flags & CQ_NOC_FLAG_DST) {
        if constexpr (cmd_buf == 2) {
            __builtin_riscv_ttrocc_scmdbuf_wr_reg(
                TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_DEST_ADDR_REG_OFFSET / 8, static_cast<uint32_t>(dst_addr));
        } else {
            static_assert(cmd_buf <= 1, "normal cmdbuf operations are only valid for cmd_buf 0 or 1");
            __builtin_riscv_ttrocc_cmdbuf_wr_reg(
                cmd_buf,
                TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_DEST_ADDR_REG_OFFSET / 8,
                static_cast<uint32_t>(dst_addr));
        }
    }
    if constexpr (flags & CQ_NOC_FLAG_NOC) {
        if constexpr (cmd_buf == 2) {
            __builtin_riscv_ttrocc_scmdbuf_wr_reg(
                TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_DEST_COORD_REG_OFFSET / 8,
                (uint32_t)(dst_addr >> NOC_ADDR_COORD_SHIFT) & NOC_COORDINATE_MASK);
        } else {
            static_assert(cmd_buf <= 1, "normal cmdbuf operations are only valid for cmd_buf 0 or 1");
            __builtin_riscv_ttrocc_cmdbuf_wr_reg(
                cmd_buf,
                TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_DEST_COORD_REG_OFFSET / 8,
                (uint32_t)(dst_addr >> NOC_ADDR_COORD_SHIFT) & NOC_COORDINATE_MASK);
        }
    }
    if constexpr (flags & CQ_NOC_INLINE_FLAG_BE) {
        uint32_t be32 = be << (dst_addr & (NOC_WORD_BYTES - 1));
        if constexpr (cmd_buf == 2) {
            __builtin_riscv_ttrocc_scmdbuf_wr_reg(TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_LEN_BYTES_REG_OFFSET / 8, be32);
        } else {
            static_assert(cmd_buf <= 1, "normal cmdbuf operations are only valid for cmd_buf 0 or 1");
            __builtin_riscv_ttrocc_cmdbuf_wr_reg(
                cmd_buf, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_LEN_BYTES_REG_OFFSET / 8, be32);
        }
    }
    if constexpr (send) {
        if constexpr (cmd_buf == 2) {
            if constexpr (flags & CQ_NOC_INLINE_FLAG_VAL) {
                __builtin_riscv_ttrocc_scmdbuf_issue_inline_trans(val);
            } else {
                __builtin_riscv_ttrocc_scmdbuf_issue_trans();
            }
        } else {
            __builtin_riscv_ttrocc_cmdbuf_issue_trans(cmd_buf);
        }
    }
}

// clang-format off
/**
 * Initializes the stateful registers for NOC read operations using a specific command buffer.
 * This function sets up the basic NOC read command configuration that will be reused across
 * multiple read transactions using the same command buffer.
 *
 * Return value: None
 *
 * | Argument                     | Description                                     | Data type | Valid range | Required |
 * |------------------------------|-------------------------------------------------|-----------|-------------|----------|
 * | noc                          | Which NOC to use for the transaction            | uint32_t  | 0 or 1      | True     |
 * | cmd_buf (template parameter) | Which command buffer to initialize              | uint32_t  | 0 - 3       | True     |
 */
// clang-format on
template <uint32_t cmd_buf>
inline __attribute__((always_inline)) void noc_read_init_state(uint32_t noc) {
    __builtin_riscv_ttrocc_cmdbuf_wr_reg(
        cmd_buf, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_MISC_REG_OFFSET / 8, CMD_BUF_MISC_READ);
}

// clang-format off
/**
 * Initiates an asynchronous read transaction using previously initialized stateful registers.
 * This function must be preceded by a call to \a noc_read_init_state for the same command buffer.
 * The function leverages stateful NOC registers to minimize register writes for repeated transactions
 * with similar characteristics.
 *
 * This function provides more granular control compared to previous stateful NOC APIs by allowing
 * selective register updates via the flags parameter. Users can specify exactly which NOC registers
 * (source address, destination address, coordinates, length) should be programmed on each call,
 * enabling fine-tuned optimization for specific transaction patterns.
 *
 * Return value: None
 *
 * | Argument                      | Description                                              | Data type        | Valid range                                              | Required |
 * |-------------------------------|----------------------------------------------------------|------------------|----------------------------------------------------------|----------|
 * | noc                           | Which NOC to use for the transaction                     | uint32_t         | 0 or 1                                                   | True     |
 * | src_addr                      | Source NOC address (x,y)+local address                   | uint64_t         | Results of \a get_noc_addr calls                         | True     |
 * | dst_addr                      | Destination address in local L1 memory                   | uint32_t         | 0..1 MB                                                  | True     |
 * | size                          | Size of transaction in bytes                             | uint32_t         | 0..NOC_V2_MAX_BYTES_IN_PACKET for single packet                  | True     |
 * | noc_mode (template parameter) | NOC mode for the transaction                             | uint8_t          | DM_DEDICATED_NOC, DM_DYNAMIC_NOC or DM_INVALID_NOC (0-2) | False    |
 * | cmd_buf (template parameter)  | Which command buffer to use for the transaction          | uint32_t         | 0 - 3                                                    | True     |
 * | flags (template parameter)    | Which NOC registers to update in this call               | enum CQNocFlags  | Combination of CQ_NOC_FLAG_* flags                       | True     |
 * | send (template parameter)     | Whether to send the transaction immediately              | enum CQNocSend   | CQ_NOC_SEND or CQ_NOC_send                               | False    |
 * | wait (template parameter)     | Whether to wait for command buffer readiness             | enum CQNocWait   | CQ_NOC_WAIT or CQ_NOC_wait                               | False    |
 */
// clang-format on
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
        __builtin_riscv_ttrocc_cmdbuf_wr_reg(
            cmd_buf,
            TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_SRC_ADDR_REG_OFFSET / 8,
            src_addr & ((1ULL << NOC_ADDR_COORD_SHIFT) - 1));
    }
    if constexpr (flags & CQ_NOC_FLAG_DST) {
        __builtin_riscv_ttrocc_cmdbuf_wr_reg(
            cmd_buf, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_DEST_ADDR_REG_OFFSET / 8, dst_addr);
    }
    if constexpr (flags & CQ_NOC_FLAG_NOC) {
        __builtin_riscv_ttrocc_cmdbuf_wr_reg(
            cmd_buf,
            TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_SRC_COORD_REG_OFFSET / 8,
            (uint32_t)(src_addr >> NOC_ADDR_COORD_SHIFT) & NOC_COORDINATE_MASK);
    }
    if constexpr (flags & CQ_NOC_FLAG_LEN) {
        __builtin_riscv_ttrocc_cmdbuf_wr_reg(
            cmd_buf, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_LEN_BYTES_REG_OFFSET / 8, size);
    }
    if constexpr (send) {
        __builtin_riscv_ttrocc_cmdbuf_issue_trans(cmd_buf);
        // Only a call that issues a transaction has a response to account for. Counting one that merely programs
        // state would overstate the reads in flight, which is what noc_common_read_with_state avoids on tt-1xx.
        noc_reads_num_issued[noc] += 1;
    }
}

// Same as above, but with src_noc_addr giving the source NOC address separately.
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
        __builtin_riscv_ttrocc_cmdbuf_wr_reg(
            cmd_buf, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_SRC_ADDR_REG_OFFSET / 8, src_addr);
    }
    if constexpr (flags & CQ_NOC_FLAG_DST) {
        __builtin_riscv_ttrocc_cmdbuf_wr_reg(
            cmd_buf, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_DEST_ADDR_REG_OFFSET / 8, dst_addr);
    }
    if constexpr (flags & CQ_NOC_FLAG_NOC) {
        __builtin_riscv_ttrocc_cmdbuf_wr_reg(
            cmd_buf, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_SRC_COORD_REG_OFFSET / 8, src_noc_addr);
    }
    if constexpr (flags & CQ_NOC_FLAG_LEN) {
        __builtin_riscv_ttrocc_cmdbuf_wr_reg(
            cmd_buf, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_LEN_BYTES_REG_OFFSET / 8, size);
    }
    if constexpr (send) {
        __builtin_riscv_ttrocc_cmdbuf_issue_trans(cmd_buf);
        // Only a call that issues a transaction has a response to account for. Counting one that merely programs
        // state would overstate the reads in flight, which is what noc_common_read_with_state avoids on tt-1xx.
        noc_reads_num_issued[noc] += 1;
    }
}

// clang-format off
/**
 * Initializes the stateful registers for NOC write operations using a specific command buffer.
 * This function sets up the basic NOC write command configuration including VC, multicast,
 * linked, and posted flags that will be reused across multiple write transactions using
 * the same command buffer.
 *
 * Return value: None
 *
 * | Argument                       | Description                                        | Data type           | Valid range         | Required |
 * |--------------------------------|----------------------------------------------------|---------------------|---------------------|----------|
 * | noc                            | Which NOC to use for the transaction               | uint32_t            | 0 or 1              | True     |
 * | vc                             | Virtual channel to use for the transactions        | uint32_t            | 0 - 3               | True     |
 * | cmd_buf (template parameter)   | Which command buffer to initialize                 | uint32_t            | 0 - 3               | True     |
 * | cmd_flags (template parameter) | Command flags for multicast/linked/posted options  | enum CQNocCmdFlags  | CQ_NOC_mkp variants | False    |
 */
// clang-format on
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
        (cmd_flags & CQ_NOC_CMD_FLAG_MCAST) ? NOC_V2_MCAST_RESP_VC : NOC_V2_WR_RESP_VC);
}

// clang-format off
/**
 * Initiates an asynchronous write transaction using previously initialized stateful registers.
 * This function must be preceded by a call to \a noc_write_init_state for the same command buffer.
 * The function leverages stateful NOC registers to minimize register writes for repeated transactions
 * with similar characteristics.
 *
 * This function provides more granular control compared to previous stateful NOC APIs by allowing
 * selective register updates via the flags parameter. Users can specify exactly which NOC registers
 * (source address, destination address, coordinates, length) should be programmed on each call,
 * enabling fine-tuned optimization for specific transaction patterns.
 *
 * Return value: None
 *
 * | Argument                            | Description                                              | Data type       | Valid range                                              | Required |
 * |-------------------------------------|----------------------------------------------------------|-----------------|----------------------------------------------------------|----------|
 * | noc                                 | Which NOC to use for the transaction                     | uint32_t        | 0 or 1                                                   | True     |
 * | src_addr                            | Source address in local L1 memory                        | uint32_t        | 0..1 MB                                                  | True     |
 * | dst_addr                            | Destination NOC address (x,y)+local address              | uint64_t        | Results of \a get_noc_addr calls                         | True     |
 * | size                                | Size of transaction in bytes                             | uint32_t        | 0..NOC_V2_MAX_BYTES_IN_PACKET for single packet                  | False    |
 * | ndests                              | Number of destinations for multicast operations          | uint32_t        | 1 or more                                                | False    |
 * | noc_mode (template parameter)       | NOC mode for the transaction                             | uint8_t         | DM_DEDICATED_NOC, DM_DYNAMIC_NOC or DM_INVALID_NOC (0-2) | False    |
 * | cmd_buf (template parameter)        | Which command buffer to use for the transaction          | uint32_t        | 0 - 3                                                    | True     |
 * | flags (template parameter)          | Which NOC registers to update in this call               | enum CQNocFlags | Combination of CQ_NOC_FLAG_* flags                       | True     |
 * | send (template parameter)           | Whether to send the transaction immediately              | enum CQNocSend  | CQ_NOC_SEND or CQ_NOC_send                               | False    |
 * | wait (template parameter)           | Whether to wait for command buffer readiness             | enum CQNocWait  | CQ_NOC_WAIT or CQ_NOC_wait                               | False    |
 * | update_counter (template parameter) | Whether to increment write counters                      | bool            | true or false                                            | False    |
 * | posted (template parameter)         | Whether the transaction is posted (no ack required)      | bool            | true or false                                            | False    |
 */
// clang-format on
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
            cmd_buf, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_SRC_ADDR_REG_OFFSET / 8, src_addr);
    }
    if constexpr (flags & CQ_NOC_FLAG_DST) {
        __builtin_riscv_ttrocc_cmdbuf_wr_reg(
            cmd_buf,
            TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_DEST_ADDR_REG_OFFSET / 8,
            dst_addr & ((1ULL << NOC_ADDR_COORD_SHIFT) - 1));
    }
    if constexpr (flags & CQ_NOC_FLAG_NOC) {
        __builtin_riscv_ttrocc_cmdbuf_wr_reg(
            cmd_buf,
            TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_DEST_COORD_REG_OFFSET / 8,
            (uint32_t)(dst_addr >> NOC_ADDR_COORD_SHIFT) & NOC_COORDINATE_MASK);
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

// Similar to above except takes additional argument, dst_noc_addr, to free up dst_addr to be 64 bits.
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
            cmd_buf, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_SRC_ADDR_REG_OFFSET / 8, src_addr);
    }
    if constexpr (flags & CQ_NOC_FLAG_DST) {
        __builtin_riscv_ttrocc_cmdbuf_wr_reg(
            cmd_buf, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_DEST_ADDR_REG_OFFSET / 8, dst_addr);
    }
    if constexpr (flags & CQ_NOC_FLAG_NOC) {
        __builtin_riscv_ttrocc_cmdbuf_wr_reg(
            cmd_buf, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_DEST_COORD_REG_OFFSET / 8, dst_noc_addr);
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
