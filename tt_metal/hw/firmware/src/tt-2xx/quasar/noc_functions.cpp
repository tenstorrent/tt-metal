// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <noc_functions.hpp>

// All functions here can be accelerated by hardware
// Legacy functions that should NOT be used.

bool cmd_buf_ok(uint32_t noc, uint32_t cmd_buf) {
    return (NOC_CMD_BUF_READ_REG(noc, cmd_buf, NOC_CMD_CTRL) == NOC_CTRL_STATUS_READY);
}

void noc_read(
    uint32_t noc,
    uint64_t src_coordinate,
    uint64_t src_addr,
    uint64_t dest_coordinate,
    uint64_t dest_addr,
    uint64_t len_bytes,
    uint32_t transaction_id,
    uint32_t static_vc,
    uint32_t cmd_buf) {
    if (len_bytes > 0) {
        while (!cmd_buf_ok(noc, cmd_buf));

        // word offset noc cmd interface
        uint32_t noc_rd_cmd_field =
            NOC_CMD_CPY | NOC_CMD_RD | NOC_CMD_RESP_MARKED | NOC_CMD_STATIC_VC(static_vc) | NOC_RESP_STATIC_VC(14);
        uint64_t offset = (cmd_buf << NOC_CMD_BUF_OFFSET_BIT) + (noc << NOC_INSTANCE_OFFSET_BIT);
        volatile uint32_t* ptr = (volatile uint32_t*)offset;
        volatile uint64_t* ptr64 = (volatile uint64_t*)offset;

        ptr64[NOC_TARG_ADDR_LO >> 3] = src_addr;
        ptr64[NOC_TARG_ADDR_HI >> 3] = (((dest_addr & 0xFFFFFFFF) << 32) | src_coordinate);
        ptr64[NOC_RET_ADDR_MID >> 3] = ((dest_coordinate << 32) | (dest_addr >> 32));
        ptr64[NOC_AT_LEN >> 3] = len_bytes;
        ptr[NOC_CTRL_LO >> 2] = noc_rd_cmd_field;
        ptr[NOC_CTRL_HI >> 2] = NOC_CMD_PKT_TAG_ID(transaction_id);
        ptr[NOC_CMD_CTRL >> 2] = NOC_CTRL_SEND_REQ;
    }
}

bool noc_reads_flushed(uint32_t noc, uint32_t transaction_id) {
    return (NOC_STATUS_READ_REG(noc, NIU_MST_REQS_OUTSTANDING_ID(transaction_id)) == 0);
}

bool all_noc_reads_flushed(uint32_t noc) {
    bool all_flushed = true;
    for (uint32_t id = NCRISC_RD_DEF_TRID; id <= NCRISC_RD_END_TRID; id++) {
        all_flushed &= NOC_STATUS_READ_REG(noc, NIU_MST_REQS_OUTSTANDING_ID(id)) == 0;
    }
    return all_flushed;
}

void noc_write(
    uint32_t noc,
    uint64_t src_coordinate,
    uint64_t src_addr,
    uint64_t dest_coordinate,
    uint64_t dest_addr,
    uint64_t len_bytes,
    uint32_t transaction_id,
    uint32_t vc,
    bool mcast,
    bool linked,
    uint32_t cmd_buf) {
    if (len_bytes > 0) {
        while (!cmd_buf_ok(noc, cmd_buf));

        uint32_t noc_cmd_field = NOC_CMD_CPY | NOC_CMD_WR | NOC_CMD_STATIC_VC(vc) | NOC_RESP_STATIC_VC(14) |
                                 (linked ? NOC_CMD_VC_LINKED : 0x0) |
                                 (mcast ? (NOC_CMD_PATH_RESERVE | NOC_CMD_BRCST_PACKET) : 0x0) | NOC_CMD_RESP_MARKED;

        // word offset noc cmd interface
        uint64_t offset = (cmd_buf << NOC_CMD_BUF_OFFSET_BIT) + (noc << NOC_INSTANCE_OFFSET_BIT);
        volatile uint32_t* ptr = (volatile uint32_t*)offset;
        volatile uint64_t* ptr64 = (volatile uint64_t*)offset;

        ptr64[NOC_TARG_ADDR_LO >> 3] = src_addr;
        ptr64[NOC_TARG_ADDR_HI >> 3] = (((dest_addr & 0xFFFFFFFF) << 32) | src_coordinate);
        ptr64[NOC_RET_ADDR_MID >> 3] = ((dest_coordinate << 32) | (dest_addr >> 32));
        ptr64[NOC_AT_LEN >> 3] = len_bytes;
        ptr[NOC_CTRL_LO >> 2] = noc_cmd_field;
        ptr[NOC_CTRL_HI >> 2] = NOC_CMD_PKT_TAG_ID(transaction_id);
        ptr[NOC_CMD_CTRL >> 2] = NOC_CTRL_SEND_REQ;
    }
}

bool noc_writes_sent(uint32_t noc, uint32_t transaction_id) {
    return (NOC_STATUS_READ_REG(noc, NIU_MST_WRITE_REQS_OUTGOING_ID(transaction_id)) == 0);
}

bool all_noc_writes_sent(uint32_t noc) {
    bool all_sent = true;
    for (uint32_t id = NCRISC_WR_DEF_TRID; id <= NCRISC_WR_LOCAL_TRID; id++) {
        all_sent &= NOC_STATUS_READ_REG(noc, NIU_MST_WRITE_REQS_OUTGOING_ID(id)) == 0;
    }
    return all_sent;
}

bool noc_nonposted_writes_flushed(uint32_t noc, uint32_t transaction_id) {
    return (NOC_STATUS_READ_REG(noc, NIU_MST_REQS_OUTSTANDING_ID(transaction_id)) == 0);
}

bool all_noc_nonposted_writes_flushed(uint32_t noc) {
    bool all_flushed = true;
    for (uint32_t id = NCRISC_WR_DEF_TRID; id <= NCRISC_WR_LOCAL_TRID; id++) {
        all_flushed &= NOC_STATUS_READ_REG(noc, NIU_MST_REQS_OUTSTANDING_ID(id)) == 0;
    }
    return all_flushed;
}

void noc_atomic_increment(
    uint32_t noc, uint64_t noc_coordinate, uint64_t addr, uint32_t incr, uint32_t wrap, bool linked, uint32_t cmd_buf) {
    while (!cmd_buf_ok(noc, cmd_buf));

    uint32_t noc_cmd_field = (linked ? NOC_CMD_VC_LINKED : 0x0) | NOC_CMD_AT;
    uint64_t at_len =
        NOC_AT_INS(NOC_AT_INS_INCR_GET) | NOC_AT_WRAP(wrap) | NOC_AT_IND_32((addr >> 2) & 0x3) | NOC_AT_IND_32_SRC(0);

    uint64_t offset = (cmd_buf << NOC_CMD_BUF_OFFSET_BIT) + (noc << NOC_INSTANCE_OFFSET_BIT);
    volatile uint32_t* ptr = (volatile uint32_t*)offset;
    volatile uint64_t* ptr64 = (volatile uint64_t*)offset;

    ptr64[NOC_TARG_ADDR_LO >> 3] = addr;
    ptr64[NOC_TARG_ADDR_HI >> 3] = noc_coordinate;
    ptr64[NOC_AT_LEN >> 3] = at_len;
    ptr[NOC_CTRL_LO >> 2] = noc_cmd_field;
    ptr[NOC_AT_DATA >> 2] = incr;
    ptr[NOC_CMD_CTRL >> 2] = NOC_CTRL_SEND_REQ;
}
