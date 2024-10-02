// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stdint.h>

#include "noc_parameters.h"

////
/*TODO: RT review this file, currently using wormhole b0 copy, check if any changes needed for BH*/
constexpr uint32_t DYNAMIC_NOC_NCRISC_WR_CMD_BUF = 2; // all writes share cmd buf
constexpr uint32_t DYNAMIC_NOC_NCRISC_WR_REG_CMD_BUF = 2;
constexpr uint32_t DYNAMIC_NOC_NCRISC_AT_CMD_BUF = 2;
constexpr uint32_t DYNAMIC_NOC_NCRISC_RD_CMD_BUF = 3;

constexpr uint32_t DYNAMIC_NOC_BRISC_WR_CMD_BUF = 0; // all writes share cmd buf
constexpr uint32_t DYNAMIC_NOC_BRISC_WR_REG_CMD_BUF = 0;
constexpr uint32_t DYNAMIC_NOC_BRISC_AT_CMD_BUF = 0;
constexpr uint32_t DYNAMIC_NOC_BRISC_RD_CMD_BUF = 1;

constexpr uint32_t NCRISC_WR_CMD_BUF = 0;  // for large writes
constexpr uint32_t NCRISC_RD_CMD_BUF = 1;  // for all reads
constexpr uint32_t NCRISC_WR_REG_CMD_BUF = 2;  // for small writes (e.g., registers, semaphores)
constexpr uint32_t NCRISC_AT_CMD_BUF = 3; // for atomics

constexpr uint32_t BRISC_WR_CMD_BUF = 0;  // for large writes
constexpr uint32_t BRISC_RD_CMD_BUF = 1;  // for all reads
constexpr uint32_t BRISC_WR_REG_CMD_BUF = 2;  // for small writes (e.g., registers, semaphores)
constexpr uint32_t BRISC_AT_CMD_BUF = 3; // for atomics

// BH has 64 bit address space but pipegen was not updated to support this so WH scheme of encoding addresses is used (36 bits of address followed by coordinates)
// This means that lo and mid registers need to have the address portion while the coordinates go into hi register
// Metal does not need to use more than 32 bits for addresses but the 60th bit needs to be set to enable NoC transactions through PCIe (see get_pcie_base_addr_from_device)
constexpr uint32_t NOC_ADDR_COORD_SHIFT = 36;
const uint32_t NOC_TARG_ADDR_COORDINATE = NOC_TARG_ADDR_HI;
const uint32_t NOC_RET_ADDR_COORDINATE = NOC_RET_ADDR_HI;
const uint32_t NOC_COORDINATE_MASK = 0xFFFFFF;

extern uint32_t noc_reads_num_issued[NUM_NOCS];
extern uint32_t noc_nonposted_writes_num_issued[NUM_NOCS];
extern uint32_t noc_nonposted_writes_acked[NUM_NOCS];
extern uint32_t noc_nonposted_atomics_acked[NUM_NOCS];
extern uint32_t noc_posted_writes_num_issued[NUM_NOCS];

inline __attribute__((always_inline)) void NOC_CMD_BUF_WRITE_REG(
    uint32_t noc, uint32_t buf, uint32_t addr, uint32_t val) {
    uint32_t offset = (buf << NOC_CMD_BUF_OFFSET_BIT) + (noc << NOC_INSTANCE_OFFSET_BIT) + addr;
    volatile uint32_t* ptr = (volatile uint32_t*)offset;
    *ptr = val;
}

inline __attribute__((always_inline)) uint32_t NOC_CMD_BUF_READ_REG(uint32_t noc, uint32_t buf, uint32_t addr) {
    uint32_t offset = (buf << NOC_CMD_BUF_OFFSET_BIT) + (noc << NOC_INSTANCE_OFFSET_BIT) + addr;
    volatile uint32_t* ptr = (volatile uint32_t*)offset;
    return *ptr;
}

inline __attribute__((always_inline)) uint32_t NOC_STATUS_READ_REG(uint32_t noc, uint32_t reg_id) {
    uint32_t offset = (noc << NOC_INSTANCE_OFFSET_BIT) + NOC_STATUS(reg_id);
    volatile uint32_t* ptr = (volatile uint32_t*)offset;
    return *ptr;
}

inline __attribute__((always_inline)) uint32_t NOC_CFG_READ_REG(uint32_t noc, uint32_t reg_id) {
    uint32_t offset = (noc << NOC_INSTANCE_OFFSET_BIT) + NOC_CFG(reg_id);
    volatile uint32_t* ptr = (volatile uint32_t*)offset;
    return *ptr;
}

inline __attribute__((always_inline)) bool noc_cmd_buf_ready(uint32_t noc, uint32_t cmd_buf) {
    return (NOC_CMD_BUF_READ_REG(noc, cmd_buf, NOC_CMD_CTRL) == NOC_CTRL_STATUS_READY);
}

inline __attribute__((always_inline)) void ncrisc_noc_fast_read(
    uint32_t noc, uint32_t cmd_buf, uint64_t src_addr, uint32_t dest_addr, uint32_t len_bytes) {
    NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_RET_ADDR_LO, dest_addr);
    NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_TARG_ADDR_LO, (uint32_t)src_addr);
    NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_TARG_ADDR_MID, (uint32_t)(src_addr >> 32) & 0x1000000F);
    NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_TARG_ADDR_COORDINATE, (uint32_t)(src_addr >> NOC_ADDR_COORD_SHIFT) & NOC_COORDINATE_MASK);
    NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_AT_LEN_BE, len_bytes);
    NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_CMD_CTRL, NOC_CTRL_SEND_REQ);
    noc_reads_num_issued[noc] += 1;
}

inline __attribute__((always_inline)) bool ncrisc_noc_reads_flushed(uint32_t noc) {
    return (NOC_STATUS_READ_REG(noc, NIU_MST_RD_RESP_RECEIVED) == noc_reads_num_issued[noc]);
}

inline __attribute__((always_inline)) bool ncrisc_noc_read_with_transaction_id_flushed(uint32_t noc, uint32_t transcation_id) {
  return (NOC_STATUS_READ_REG(noc, NIU_MST_REQS_OUTSTANDING_ID(transcation_id)) == 0);
}

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
    bool posted = false) {
    uint32_t noc_cmd_field =
        NOC_CMD_CPY | NOC_CMD_WR | NOC_CMD_VC_STATIC | NOC_CMD_STATIC_VC(vc) | (linked ? NOC_CMD_VC_LINKED : 0x0) |
        (mcast ? ((multicast_path_reserve ? NOC_CMD_PATH_RESERVE : 0) | NOC_CMD_BRCST_PACKET) : 0x0) |
        (posted ? 0 : NOC_CMD_RESP_MARKED);

    NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_CTRL, noc_cmd_field);
    NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_TARG_ADDR_LO, src_addr);
    NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_RET_ADDR_LO, (uint32_t)dest_addr);
    NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_RET_ADDR_MID, (uint32_t)(dest_addr >> 32) & 0x1000000F);
    NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_RET_ADDR_COORDINATE, (uint32_t)(dest_addr >> NOC_ADDR_COORD_SHIFT) & NOC_COORDINATE_MASK);
    NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_AT_LEN_BE, len_bytes);
    NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_CMD_CTRL, NOC_CTRL_SEND_REQ);
    if (posted) {
        noc_posted_writes_num_issued[noc] += 1;
    } else {
        noc_nonposted_writes_num_issued[noc] += 1;
        noc_nonposted_writes_acked[noc] += num_dests;
    }
}

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
    uint32_t noc_cmd_field =
        NOC_CMD_CPY | NOC_CMD_WR | NOC_CMD_VC_STATIC | NOC_CMD_STATIC_VC(vc) | (linked ? NOC_CMD_VC_LINKED : 0x0) |
        (mcast ? ((multicast_path_reserve ? NOC_CMD_PATH_RESERVE : 0) | NOC_CMD_BRCST_PACKET) : 0x0) |
        NOC_CMD_BRCST_SRC_INCLUDE | NOC_CMD_RESP_MARKED;

    NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_CTRL, noc_cmd_field);
    NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_TARG_ADDR_LO, src_addr);
    NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_RET_ADDR_LO, (uint32_t)dest_addr);
    NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_RET_ADDR_MID, (uint32_t)(dest_addr >> 32) & 0x1000000F);
    NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_RET_ADDR_COORDINATE, (uint32_t)(dest_addr >> NOC_ADDR_COORD_SHIFT) & NOC_COORDINATE_MASK);
    NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_AT_LEN_BE, len_bytes);
    NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_CMD_CTRL, NOC_CTRL_SEND_REQ);
    noc_nonposted_writes_num_issued[noc] += 1;
    noc_nonposted_writes_acked[noc] += num_dests;
}

inline __attribute__((always_inline)) void ncrisc_noc_blitz_write_setup(
    uint32_t noc, uint32_t cmd_buf, uint64_t dest_addr, uint32_t len_bytes, uint32_t vc, uint32_t num_times_to_write) {
    uint32_t noc_cmd_field = NOC_CMD_CPY | NOC_CMD_WR | NOC_CMD_VC_STATIC | NOC_CMD_STATIC_VC(vc) | NOC_CMD_RESP_MARKED;

    while (!noc_cmd_buf_ready(noc, cmd_buf));
    NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_CTRL, noc_cmd_field);
    NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_AT_LEN_BE, len_bytes);
    NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_RET_ADDR_MID, (uint32_t)(dest_addr >> 32) & 0x1000000F);
    NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_RET_ADDR_COORDINATE, (uint32_t)(dest_addr >> NOC_ADDR_COORD_SHIFT) & NOC_COORDINATE_MASK);
    noc_nonposted_writes_num_issued[noc] += num_times_to_write;
    noc_nonposted_writes_acked[noc] += num_times_to_write;
}

inline __attribute__((always_inline)) bool ncrisc_noc_nonposted_writes_sent(uint32_t noc) {
    return (NOC_STATUS_READ_REG(noc, NIU_MST_NONPOSTED_WR_REQ_SENT) == noc_nonposted_writes_num_issued[noc]);
}

inline __attribute__((always_inline)) bool ncrisc_noc_posted_writes_sent(uint32_t noc) {
    return (NOC_STATUS_READ_REG(noc, NIU_MST_POSTED_WR_REQ_SENT) == noc_posted_writes_num_issued[noc]);
}

inline __attribute__((always_inline)) bool ncrisc_noc_nonposted_writes_flushed(uint32_t noc) {
    return (NOC_STATUS_READ_REG(noc, NIU_MST_WR_ACK_RECEIVED) == noc_nonposted_writes_acked[noc]);
}

inline __attribute__((always_inline)) bool ncrisc_noc_nonposted_atomics_flushed(uint32_t noc) {
    return (NOC_STATUS_READ_REG(noc, NIU_MST_ATOMIC_RESP_RECEIVED) == noc_nonposted_atomics_acked[noc]);
}

inline __attribute__((always_inline)) void noc_init(uint32_t atomic_ret_val) {
#pragma GCC unroll 0
    for (int noc = 0; noc < NUM_NOCS; noc++) {
        uint32_t noc_id_reg = NOC_CMD_BUF_READ_REG(noc, 0, NOC_NODE_ID);
        uint32_t my_x = noc_id_reg & NOC_NODE_ID_MASK;
        uint32_t my_y = (noc_id_reg >> NOC_ADDR_NODE_ID_BITS) & NOC_NODE_ID_MASK;
        uint64_t xy_local_addr = NOC_XY_ADDR(my_x, my_y, 0);

        NOC_CMD_BUF_WRITE_REG(noc, NCRISC_WR_CMD_BUF, NOC_TARG_ADDR_MID, 0x0);
        NOC_CMD_BUF_WRITE_REG(noc, NCRISC_WR_CMD_BUF, NOC_TARG_ADDR_COORDINATE, (uint32_t)(xy_local_addr >> NOC_ADDR_COORD_SHIFT) & NOC_COORDINATE_MASK);
        NOC_CMD_BUF_WRITE_REG(noc, NCRISC_WR_REG_CMD_BUF, NOC_TARG_ADDR_MID, 0x0);
        NOC_CMD_BUF_WRITE_REG(noc, NCRISC_WR_REG_CMD_BUF, NOC_TARG_ADDR_COORDINATE, (uint32_t)(xy_local_addr >> NOC_ADDR_COORD_SHIFT) & NOC_COORDINATE_MASK);

        uint64_t atomic_ret_addr = NOC_XY_ADDR(my_x, my_y, atomic_ret_val);
        NOC_CMD_BUF_WRITE_REG(noc, NCRISC_AT_CMD_BUF, NOC_RET_ADDR_LO, (uint32_t)(atomic_ret_addr & 0xFFFFFFFF));
        NOC_CMD_BUF_WRITE_REG(noc, NCRISC_AT_CMD_BUF, NOC_RET_ADDR_MID, 0x0);
        NOC_CMD_BUF_WRITE_REG(noc, NCRISC_AT_CMD_BUF, NOC_RET_ADDR_COORDINATE, (uint32_t)(atomic_ret_addr >> NOC_ADDR_COORD_SHIFT) & NOC_COORDINATE_MASK);

        uint32_t noc_rd_cmd_field =
            NOC_CMD_CPY | NOC_CMD_RD | NOC_CMD_RESP_MARKED | NOC_CMD_VC_STATIC | NOC_CMD_STATIC_VC(1);
        NOC_CMD_BUF_WRITE_REG(noc, NCRISC_RD_CMD_BUF, NOC_CTRL, noc_rd_cmd_field);
        NOC_CMD_BUF_WRITE_REG(noc, NCRISC_RD_CMD_BUF, NOC_RET_ADDR_MID, 0x0); // get rid of this?
        NOC_CMD_BUF_WRITE_REG(noc, NCRISC_RD_CMD_BUF, NOC_RET_ADDR_COORDINATE, (uint32_t)(xy_local_addr >> NOC_ADDR_COORD_SHIFT) & NOC_COORDINATE_MASK);
    }
}

inline __attribute__((always_inline)) void noc_init_dynamic_noc() {
#pragma GCC unroll 0
  for (int noc = 0; noc < NUM_NOCS; noc++) {
    uint32_t noc_id_reg = NOC_CMD_BUF_READ_REG(noc, 0, NOC_NODE_ID);
    uint32_t my_x = noc_id_reg & NOC_NODE_ID_MASK;
    uint32_t my_y = (noc_id_reg >> NOC_ADDR_NODE_ID_BITS) & NOC_NODE_ID_MASK;
    uint64_t xy_local_addr = NOC_XY_ADDR(my_x, my_y, 0);

    uint32_t noc_rd_cmd_field = NOC_CMD_CPY | NOC_CMD_RD | NOC_CMD_RESP_MARKED | NOC_CMD_VC_STATIC | NOC_CMD_STATIC_VC(1);

    // program brisc cmd_buf 0
    NOC_CMD_BUF_WRITE_REG(noc, DYNAMIC_NOC_BRISC_RD_CMD_BUF, NOC_CTRL, noc_rd_cmd_field);
    NOC_CMD_BUF_WRITE_REG(noc, DYNAMIC_NOC_BRISC_RD_CMD_BUF, NOC_RET_ADDR_COORDINATE, (uint32_t)(xy_local_addr >> NOC_ADDR_COORD_SHIFT));

    // program brisc cmd_buf 1
    NOC_CMD_BUF_WRITE_REG(noc, DYNAMIC_NOC_BRISC_WR_CMD_BUF, NOC_TARG_ADDR_COORDINATE, (uint32_t)(xy_local_addr >> NOC_ADDR_COORD_SHIFT));

    // program ncrisc cmd_buf 2
    NOC_CMD_BUF_WRITE_REG(noc, DYNAMIC_NOC_NCRISC_RD_CMD_BUF, NOC_CTRL, noc_rd_cmd_field);
    NOC_CMD_BUF_WRITE_REG(noc, DYNAMIC_NOC_NCRISC_RD_CMD_BUF, NOC_RET_ADDR_COORDINATE, (uint32_t)(xy_local_addr >> NOC_ADDR_COORD_SHIFT));

    // program ncrisc cmd_buf 3
    NOC_CMD_BUF_WRITE_REG(noc, DYNAMIC_NOC_NCRISC_WR_CMD_BUF, NOC_TARG_ADDR_COORDINATE, (uint32_t)(xy_local_addr >> NOC_ADDR_COORD_SHIFT));
  }
}

// set noc local memory state for a single kernel from the global state
inline __attribute__((always_inline)) void noc_local_state_init(int noc) {
    noc_reads_num_issued[noc] = NOC_STATUS_READ_REG(noc, NIU_MST_RD_RESP_RECEIVED);
    noc_nonposted_writes_num_issued[noc] = NOC_STATUS_READ_REG(noc, NIU_MST_NONPOSTED_WR_REQ_SENT);
    noc_nonposted_writes_acked[noc] = NOC_STATUS_READ_REG(noc, NIU_MST_WR_ACK_RECEIVED);
    noc_nonposted_atomics_acked[noc] = NOC_STATUS_READ_REG(noc, NIU_MST_ATOMIC_RESP_RECEIVED);
    noc_posted_writes_num_issued[noc] = NOC_STATUS_READ_REG(noc, NIU_MST_POSTED_WR_REQ_SENT);
}

inline __attribute__((always_inline)) void ncrisc_noc_counters_init() {
    for (int noc = 0; noc < NUM_NOCS; noc++) {
        noc_reads_num_issued[noc] = NOC_STATUS_READ_REG(noc, NIU_MST_RD_RESP_RECEIVED);
        noc_nonposted_writes_num_issued[noc] = NOC_STATUS_READ_REG(noc, NIU_MST_NONPOSTED_WR_REQ_SENT);
        noc_nonposted_writes_acked[noc] = NOC_STATUS_READ_REG(noc, NIU_MST_WR_ACK_RECEIVED);
        noc_nonposted_atomics_acked[noc] = NOC_STATUS_READ_REG(noc, NIU_MST_ATOMIC_RESP_RECEIVED);
        noc_posted_writes_num_issued[noc] = NOC_STATUS_READ_REG(noc, NIU_MST_POSTED_WR_REQ_SENT);
    }
}

inline __attribute__((always_inline)) void ncrisc_noc_full_sync() {
    for (uint32_t n = 0; n < NUM_NOCS; n++) {
        while (!ncrisc_noc_reads_flushed(n));
        while (!ncrisc_noc_nonposted_writes_sent(n));
        while (!ncrisc_noc_nonposted_writes_flushed(n));
        while (!ncrisc_noc_nonposted_atomics_flushed(n));
        while (!ncrisc_noc_posted_writes_sent(n));
    }
}

inline __attribute__((always_inline)) void ncrisc_noc_fast_read_any_len(
    uint32_t noc, uint32_t cmd_buf, uint64_t src_addr, uint32_t dest_addr, uint32_t len_bytes) {
    while (len_bytes > NOC_MAX_BURST_SIZE) {
        while (!noc_cmd_buf_ready(noc, cmd_buf));
        ncrisc_noc_fast_read(noc, cmd_buf, src_addr, dest_addr, NOC_MAX_BURST_SIZE);
        src_addr += NOC_MAX_BURST_SIZE;
        dest_addr += NOC_MAX_BURST_SIZE;
        len_bytes -= NOC_MAX_BURST_SIZE;
    }
    while (!noc_cmd_buf_ready(noc, cmd_buf));
    ncrisc_noc_fast_read(noc, cmd_buf, src_addr, dest_addr, len_bytes);
}

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
    bool posted = false) {
    while (len_bytes > NOC_MAX_BURST_SIZE) {
        while (!noc_cmd_buf_ready(noc, cmd_buf));
        ncrisc_noc_fast_write(
            noc,
            cmd_buf,
            src_addr,
            dest_addr,
            NOC_MAX_BURST_SIZE,
            vc,
            mcast,
            linked,
            num_dests,
            multicast_path_reserve,
            posted);
        src_addr += NOC_MAX_BURST_SIZE;
        dest_addr += NOC_MAX_BURST_SIZE;
        len_bytes -= NOC_MAX_BURST_SIZE;
    }
    while (!noc_cmd_buf_ready(noc, cmd_buf));
    ncrisc_noc_fast_write(
        noc, cmd_buf, src_addr, dest_addr, len_bytes, vc, mcast, linked, num_dests, multicast_path_reserve, posted);
}

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
    while (len_bytes > NOC_MAX_BURST_SIZE) {
        while (!noc_cmd_buf_ready(noc, cmd_buf));
        ncrisc_noc_fast_write_loopback_src(
            noc,
            cmd_buf,
            src_addr,
            dest_addr,
            NOC_MAX_BURST_SIZE,
            vc,
            mcast,
            linked,
            num_dests,
            multicast_path_reserve);
        src_addr += NOC_MAX_BURST_SIZE;
        dest_addr += NOC_MAX_BURST_SIZE;
        len_bytes -= NOC_MAX_BURST_SIZE;
    }
    while (!noc_cmd_buf_ready(noc, cmd_buf));
    ncrisc_noc_fast_write_loopback_src(
        noc, cmd_buf, src_addr, dest_addr, len_bytes, vc, mcast, linked, num_dests, multicast_path_reserve);
}

inline __attribute__((always_inline)) void noc_fast_write_dw_inline(
    uint32_t noc,
    uint32_t cmd_buf,
    uint32_t val,
    uint64_t dest_addr,
    uint32_t be,
    uint32_t static_vc,
    bool mcast,
    bool posted = false) {
    bool static_vc_alloc = true;
    uint32_t noc_cmd_field = (static_vc_alloc ? NOC_CMD_VC_STATIC : 0x0) | NOC_CMD_STATIC_VC(static_vc) | NOC_CMD_CPY |
                             NOC_CMD_WR | NOC_CMD_WR_INLINE |
                             (mcast ? (NOC_CMD_PATH_RESERVE | NOC_CMD_BRCST_PACKET) : 0x0) |
                             (posted ? 0x0 : NOC_CMD_RESP_MARKED);

    uint32_t be32 = be;
    uint32_t be_shift = (dest_addr & (NOC_WORD_BYTES - 1));
    be32 = (be32 << be_shift);

    while (!noc_cmd_buf_ready(noc, cmd_buf));
    NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_AT_DATA, val);
    NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_CTRL, noc_cmd_field);
    NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_TARG_ADDR_LO, (uint32_t)(dest_addr));
    NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_TARG_ADDR_MID, (uint32_t)(dest_addr >> 32) & 0x1000000F);
    NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_TARG_ADDR_COORDINATE, (uint32_t)(dest_addr >> NOC_ADDR_COORD_SHIFT) & NOC_COORDINATE_MASK);
    NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_AT_LEN_BE, be32);
    NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_CMD_CTRL, NOC_CTRL_SEND_REQ);
    if (posted) {
        noc_posted_writes_num_issued[noc] += 1;
    } else {
        noc_nonposted_writes_num_issued[noc] += 1;
        noc_nonposted_writes_acked[noc] += 1;
    }
}

inline __attribute__((always_inline)) void noc_fast_atomic_increment(
    uint32_t noc,
    uint32_t cmd_buf,
    uint64_t addr,
    uint32_t vc,
    uint32_t incr,
    uint32_t wrap,
    bool linked,
    bool posted = false) {
    while (!noc_cmd_buf_ready(noc, cmd_buf));
    if constexpr (NOC_MODE == DM_DYNAMIC_NOC) {
        uint32_t noc_id_reg = NOC_CMD_BUF_READ_REG(noc, 0, NOC_NODE_ID);
        uint32_t my_x = noc_id_reg & NOC_NODE_ID_MASK;
        uint32_t my_y = (noc_id_reg >> NOC_ADDR_NODE_ID_BITS) & NOC_NODE_ID_MASK;
        uint64_t atomic_ret_addr = NOC_XY_ADDR(my_x, my_y, (uint32_t)(&atomic_ret_val));
        NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_RET_ADDR_LO, (uint32_t)(atomic_ret_addr & 0xFFFFFFFF));
        NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_RET_ADDR_COORDINATE, (uint32_t)(atomic_ret_addr >> NOC_ADDR_COORD_SHIFT));
    }
    NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_TARG_ADDR_LO, (uint32_t)(addr & 0xFFFFFFFF));
    NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_TARG_ADDR_MID, (uint32_t)(addr >> 32) & 0x1000000F);
    NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_TARG_ADDR_COORDINATE, (uint32_t)(addr >> NOC_ADDR_COORD_SHIFT) & NOC_COORDINATE_MASK);
    NOC_CMD_BUF_WRITE_REG(
        noc,
        cmd_buf,
        NOC_CTRL,
        NOC_CMD_VC_STATIC | NOC_CMD_STATIC_VC(vc) | (linked ? NOC_CMD_VC_LINKED : 0x0) |
            (posted ? 0 : NOC_CMD_RESP_MARKED) | NOC_CMD_AT);
    NOC_CMD_BUF_WRITE_REG(
        noc,
        cmd_buf,
        NOC_AT_LEN_BE,
        NOC_AT_INS(NOC_AT_INS_INCR_GET) | NOC_AT_WRAP(wrap) | NOC_AT_IND_32((addr >> 2) & 0x3) | NOC_AT_IND_32_SRC(0));
    NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_AT_DATA, incr);
    NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_CMD_CTRL, 0x1);
    if (!posted) {
        noc_nonposted_atomics_acked[noc] += 1;
    }
}

// issue noc reads while wait for outstanding transactions done
inline __attribute__((always_inline)) void ncrisc_noc_fast_read_with_transaction_id(uint32_t noc, uint32_t cmd_buf, uint32_t src_base_addr, uint32_t src_addr, uint32_t dest_addr, uint32_t trid) {
    uint32_t src_addr_;
    src_addr_ = src_base_addr + src_addr;

    while (!noc_cmd_buf_ready(noc, cmd_buf));
    while (NOC_STATUS_READ_REG(noc, NIU_MST_REQS_OUTSTANDING_ID(trid)) > ((NOC_MAX_TRANSACTION_ID_COUNT+1)/2));

    NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_RET_ADDR_LO, dest_addr);
    NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_TARG_ADDR_LO, src_addr_);      // (uint32_t)src_addr
    NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_CMD_CTRL, NOC_CTRL_SEND_REQ);
    noc_reads_num_issued[noc] += 1;
}

// set transaction id for a noc read
inline __attribute__((always_inline)) void ncrisc_noc_set_transaction_id(uint32_t noc, uint32_t cmd_buf, uint32_t trid) {
    NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_PACKET_TAG, NOC_PACKET_TAG_TRANSACTION_ID(trid));
}
