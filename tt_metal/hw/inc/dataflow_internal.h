// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The routines in this file are not supported APIs and may change
// These routines are split out reads/wrties into multiple calls for optimized
// transfers

#pragma once

#include "dataflow_api.h"

FORCE_INLINE
void noc_fast_read_wait_ready() {
    while (!noc_cmd_buf_ready(noc_index, NCRISC_RD_CMD_BUF));
}

FORCE_INLINE
void noc_fast_read_set_src_xy(uint64_t src_addr) {
#ifdef ARCH_BLACKHOLE
    NOC_CMD_BUF_WRITE_REG(noc_index, NCRISC_RD_CMD_BUF, NOC_TARG_ADDR_MID, (uint32_t)(src_addr >> 32) & 0x1000000F);
#endif
    NOC_CMD_BUF_WRITE_REG(noc_index, NCRISC_RD_CMD_BUF, NOC_TARG_ADDR_COORDINATE, uint32_t(src_addr >> NOC_ADDR_COORD_SHIFT) & NOC_COORDINATE_MASK);
}

FORCE_INLINE
void noc_fast_read_set_len(uint32_t len_bytes) {
    NOC_CMD_BUF_WRITE_REG(noc_index, NCRISC_RD_CMD_BUF, NOC_AT_LEN_BE, len_bytes);
}

FORCE_INLINE
void noc_fast_read(uint32_t src_addr, uint32_t dest_addr) {
    DEBUG_STATUS("NFRW");
    DEBUG_SANITIZE_NOC_READ_TRANSACTION(
        noc_index, (uint64_t)(src_addr) | (uint64_t)NOC_CMD_BUF_READ_REG(noc_index, NCRISC_RD_CMD_BUF, NOC_TARG_ADDR_COORDINATE) << 32,
        dest_addr,
        NOC_CMD_BUF_READ_REG(noc_index, NCRISC_RD_CMD_BUF, NOC_AT_LEN_BE)
    );
    while (!noc_cmd_buf_ready(noc_index, NCRISC_RD_CMD_BUF));
    DEBUG_STATUS("NFRD");

    NOC_CMD_BUF_WRITE_REG(noc_index, NCRISC_RD_CMD_BUF, NOC_RET_ADDR_LO, dest_addr);
    NOC_CMD_BUF_WRITE_REG(noc_index, NCRISC_RD_CMD_BUF, NOC_TARG_ADDR_LO, src_addr);
    NOC_CMD_BUF_WRITE_REG(noc_index, NCRISC_RD_CMD_BUF, NOC_CMD_CTRL, NOC_CTRL_SEND_REQ);
}

FORCE_INLINE
void noc_fast_read_inc_num_issued(uint32_t num_issued) {
    // while (!noc_cmd_buf_ready(noc_index, NCRISC_RD_CMD_BUF));
    noc_reads_num_issued[noc_index] += num_issued;
}

FORCE_INLINE
void noc_fast_write_wait_ready() {
    while (!noc_cmd_buf_ready(noc_index, NCRISC_WR_CMD_BUF));
}

FORCE_INLINE
void noc_fast_write_set_cmd_field(uint32_t vc, bool mcast, bool linked) {
    uint32_t noc_cmd_field = NOC_CMD_CPY | NOC_CMD_WR | NOC_CMD_VC_STATIC | NOC_CMD_STATIC_VC(vc) |
                             (linked ? NOC_CMD_VC_LINKED : 0x0) |
                             (mcast ? (NOC_CMD_PATH_RESERVE | NOC_CMD_BRCST_PACKET) : 0x0) | NOC_CMD_RESP_MARKED;

    NOC_CMD_BUF_WRITE_REG(noc_index, NCRISC_WR_CMD_BUF, NOC_CTRL, noc_cmd_field);
}

FORCE_INLINE
void noc_fast_write_set_dst_xy(uint64_t dest_addr) {
#ifdef ARCH_BLACKHOLE
    NOC_CMD_BUF_WRITE_REG(noc_index, NCRISC_WR_CMD_BUF, NOC_RET_ADDR_MID, (uint32_t)(dest_addr >> 32) & 0x1000000F);
#endif
    NOC_CMD_BUF_WRITE_REG(noc_index, NCRISC_WR_CMD_BUF, NOC_RET_ADDR_COORDINATE, (uint32_t)(dest_addr >> NOC_ADDR_COORD_SHIFT) & NOC_COORDINATE_MASK);
}

FORCE_INLINE
void noc_fast_write_set_len(uint32_t len_bytes) {
    NOC_CMD_BUF_WRITE_REG(noc_index, NCRISC_WR_CMD_BUF, NOC_AT_LEN_BE, len_bytes);
}

// a fast write that assumes a single-dest (ie unicast)
FORCE_INLINE
void noc_fast_write(uint32_t src_addr, uint64_t dest_addr) {
    DEBUG_SANITIZE_NOC_WRITE_TRANSACTION(
        noc_index, dest_addr | (uint64_t)NOC_CMD_BUF_READ_REG(noc_index, NCRISC_WR_CMD_BUF, NOC_RET_ADDR_COORDINATE) << 32,
        dest_addr,
        NOC_CMD_BUF_READ_REG(noc_index, NCRISC_WR_CMD_BUF, NOC_AT_LEN_BE)
    );
    NOC_CMD_BUF_WRITE_REG(noc_index, NCRISC_WR_CMD_BUF, NOC_TARG_ADDR_LO, src_addr);
    NOC_CMD_BUF_WRITE_REG(noc_index, NCRISC_WR_CMD_BUF, NOC_RET_ADDR_LO, (uint32_t)dest_addr);
    NOC_CMD_BUF_WRITE_REG(noc_index, NCRISC_WR_CMD_BUF, NOC_CMD_CTRL, NOC_CTRL_SEND_REQ);
}

FORCE_INLINE
void noc_fast_write_inc_num_dests(uint32_t num_issued) {
    noc_nonposted_writes_num_issued[noc_index] += num_issued;
    noc_nonposted_writes_acked[noc_index] += num_issued;
}
