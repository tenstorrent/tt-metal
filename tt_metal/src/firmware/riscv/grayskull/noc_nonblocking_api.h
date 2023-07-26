#pragma once

#include <stdint.h>

#include "noc_parameters.h"

////

const uint32_t NCRISC_WR_CMD_BUF = 0;
const uint32_t NCRISC_RD_CMD_BUF = 1;
const uint32_t NCRISC_WR_REG_CMD_BUF = 2;
const uint32_t NCRISC_AT_CMD_BUF = 3;

extern uint32_t noc_reads_num_issued[NUM_NOCS];
extern uint32_t noc_nonposted_writes_num_issued[NUM_NOCS];
extern uint32_t noc_nonposted_writes_acked[NUM_NOCS];

inline void NOC_CMD_BUF_WRITE_REG(uint32_t noc, uint32_t buf, uint32_t addr, uint32_t val) {
  uint32_t offset = (buf << NOC_CMD_BUF_OFFSET_BIT) + (noc << NOC_INSTANCE_OFFSET_BIT) + addr;
  volatile tt_reg_ptr uint32_t* ptr = (volatile tt_reg_ptr uint32_t*)offset;
  *ptr = val;
}


inline uint32_t NOC_CMD_BUF_READ_REG(uint32_t noc, uint32_t buf, uint32_t addr) {
  uint32_t offset = (buf << NOC_CMD_BUF_OFFSET_BIT) + (noc << NOC_INSTANCE_OFFSET_BIT) + addr;
  volatile tt_reg_ptr uint32_t* ptr = (volatile tt_reg_ptr uint32_t*)offset;
  return *ptr;
}


inline uint32_t NOC_STATUS_READ_REG(uint32_t noc, uint32_t reg_id) {
  uint32_t offset = (noc << NOC_INSTANCE_OFFSET_BIT) + NOC_STATUS(reg_id);
  volatile tt_reg_ptr uint32_t* ptr = (volatile tt_reg_ptr uint32_t*)offset;
  return *ptr;
}


inline void NOC_CMD_BUF_WRITE_REG_L1(uint32_t noc, uint32_t buf, uint32_t addr, uint32_t val) {
  uint32_t offset = (buf << NOC_CMD_BUF_OFFSET_BIT) + (noc << NOC_INSTANCE_OFFSET_BIT) + addr;
  volatile tt_reg_ptr uint32_t* ptr = (volatile tt_reg_ptr uint32_t*)offset;
  *ptr = val;
}


inline uint32_t NOC_CMD_BUF_READ_REG_L1(uint32_t noc, uint32_t buf, uint32_t addr) {
  uint32_t offset = (buf << NOC_CMD_BUF_OFFSET_BIT) + (noc << NOC_INSTANCE_OFFSET_BIT) + addr;
  volatile tt_reg_ptr uint32_t* ptr = (volatile tt_reg_ptr uint32_t*)offset;
  return *ptr;
}


inline uint32_t NOC_STATUS_READ_REG_L1(uint32_t noc, uint32_t reg_id) {
  uint32_t offset = (noc << NOC_INSTANCE_OFFSET_BIT) + NOC_STATUS(reg_id);
  volatile tt_reg_ptr uint32_t* ptr = (volatile tt_reg_ptr uint32_t*)offset;
  return *ptr;
}


inline void ncrisc_noc_fast_read(uint32_t noc, uint32_t cmd_buf, uint64_t src_addr, uint32_t dest_addr, uint32_t len_bytes) {
  if (len_bytes > 0) {
    NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_RET_ADDR_LO, dest_addr);
    NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_TARG_ADDR_LO, (uint32_t)src_addr);
    NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_TARG_ADDR_MID, src_addr >> 32);
    NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_AT_LEN_BE, len_bytes);
    NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_CMD_CTRL, NOC_CTRL_SEND_REQ);
    noc_reads_num_issued[noc] += 1;
  }
}

inline __attribute__((always_inline)) void ncrisc_noc_fast_read_l1(uint32_t noc, uint32_t cmd_buf, uint64_t src_addr, uint32_t dest_addr, uint32_t len_bytes) {
  if (len_bytes > 0) {
    NOC_CMD_BUF_WRITE_REG_L1(noc, cmd_buf, NOC_RET_ADDR_LO, dest_addr);
    NOC_CMD_BUF_WRITE_REG_L1(noc, cmd_buf, NOC_TARG_ADDR_LO, (uint32_t)src_addr);
    NOC_CMD_BUF_WRITE_REG_L1(noc, cmd_buf, NOC_TARG_ADDR_MID, src_addr >> 32);
    NOC_CMD_BUF_WRITE_REG_L1(noc, cmd_buf, NOC_AT_LEN_BE, len_bytes);
    NOC_CMD_BUF_WRITE_REG_L1(noc, cmd_buf, NOC_CMD_CTRL, NOC_CTRL_SEND_REQ);
    noc_reads_num_issued[noc] += 1;
  }
}

inline bool ncrisc_noc_reads_flushed(uint32_t noc) {
  return (NOC_STATUS_READ_REG(noc, NIU_MST_RD_RESP_RECEIVED) == noc_reads_num_issued[noc]);
}

inline __attribute__((always_inline)) bool ncrisc_noc_reads_flushed_l1(uint32_t noc) {
  return (NOC_STATUS_READ_REG_L1(noc, NIU_MST_RD_RESP_RECEIVED) == noc_reads_num_issued[noc]);
}

inline bool ncrisc_noc_fast_read_ok(uint32_t noc, uint32_t cmd_buf) {
  return (NOC_CMD_BUF_READ_REG(noc, cmd_buf, NOC_CMD_CTRL) == NOC_CTRL_STATUS_READY);
}

inline __attribute__((always_inline)) bool ncrisc_noc_fast_read_ok_l1(uint32_t noc, uint32_t cmd_buf) {
  return (NOC_CMD_BUF_READ_REG_L1(noc, cmd_buf, NOC_CMD_CTRL) == NOC_CTRL_STATUS_READY);
}

inline void ncrisc_noc_fast_write(uint32_t noc, uint32_t cmd_buf, uint32_t src_addr, uint64_t dest_addr, uint32_t len_bytes, uint32_t vc, bool mcast, bool linked, uint32_t num_dests) {
  if (len_bytes > 0) {
    uint32_t noc_cmd_field =
      NOC_CMD_CPY | NOC_CMD_WR |
      NOC_CMD_VC_STATIC  |
      NOC_CMD_STATIC_VC(vc) |
      (linked ? NOC_CMD_VC_LINKED : 0x0) |
      (mcast ? (NOC_CMD_PATH_RESERVE | NOC_CMD_BRCST_PACKET) : 0x0) |
      NOC_CMD_RESP_MARKED;

    NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_CTRL, noc_cmd_field);
    NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_TARG_ADDR_LO, src_addr);
    NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_RET_ADDR_LO, (uint32_t)dest_addr);
    NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_RET_ADDR_MID, dest_addr >> 32);
    NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_AT_LEN_BE, len_bytes);
    NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_CMD_CTRL, NOC_CTRL_SEND_REQ);
    noc_nonposted_writes_num_issued[noc] += 1;
    noc_nonposted_writes_acked[noc] += num_dests;
  }
}

inline void ncrisc_noc_fast_write_loopback_src(uint32_t noc, uint32_t cmd_buf, uint32_t src_addr, uint64_t dest_addr, uint32_t len_bytes, uint32_t vc, bool mcast, bool linked, uint32_t num_dests) {
  if (len_bytes > 0) {
    uint32_t noc_cmd_field =
      NOC_CMD_CPY | NOC_CMD_WR |
      NOC_CMD_VC_STATIC  |
      NOC_CMD_STATIC_VC(vc) |
      (linked ? NOC_CMD_VC_LINKED : 0x0) |
      (mcast ? (NOC_CMD_PATH_RESERVE | NOC_CMD_BRCST_PACKET) : 0x0) |
      NOC_CMD_BRCST_SRC_INCLUDE |
      NOC_CMD_RESP_MARKED;

    NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_CTRL, noc_cmd_field);
    NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_TARG_ADDR_LO, src_addr);
    NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_RET_ADDR_LO, (uint32_t)dest_addr);
    NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_RET_ADDR_MID, dest_addr >> 32);
    NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_AT_LEN_BE, len_bytes);
    NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_CMD_CTRL, NOC_CTRL_SEND_REQ);
    noc_nonposted_writes_num_issued[noc] += 1;
    noc_nonposted_writes_acked[noc] += num_dests;
  }
}

inline __attribute__((always_inline)) void ncrisc_noc_fast_write_l1(uint32_t noc, uint32_t cmd_buf, uint32_t src_addr, uint64_t dest_addr, uint32_t len_bytes, uint32_t vc, bool mcast, bool linked, uint32_t num_dests) {
  if (len_bytes > 0) {
    uint32_t noc_cmd_field =
      NOC_CMD_CPY | NOC_CMD_WR |
      NOC_CMD_VC_STATIC  |
      NOC_CMD_STATIC_VC(vc) |
      (linked ? NOC_CMD_VC_LINKED : 0x0) |
      (mcast ? (NOC_CMD_PATH_RESERVE | NOC_CMD_BRCST_PACKET) : 0x0) |
      NOC_CMD_RESP_MARKED;

    NOC_CMD_BUF_WRITE_REG_L1(noc, cmd_buf, NOC_CTRL, noc_cmd_field);
    NOC_CMD_BUF_WRITE_REG_L1(noc, cmd_buf, NOC_TARG_ADDR_LO, src_addr);
    NOC_CMD_BUF_WRITE_REG_L1(noc, cmd_buf, NOC_RET_ADDR_LO, (uint32_t)dest_addr);
    NOC_CMD_BUF_WRITE_REG_L1(noc, cmd_buf, NOC_RET_ADDR_MID, dest_addr >> 32);
    NOC_CMD_BUF_WRITE_REG_L1(noc, cmd_buf, NOC_AT_LEN_BE, len_bytes);
    NOC_CMD_BUF_WRITE_REG_L1(noc, cmd_buf, NOC_CMD_CTRL, NOC_CTRL_SEND_REQ);
    noc_nonposted_writes_num_issued[noc] += 1;
    noc_nonposted_writes_acked[noc] += num_dests;
  }
}

inline bool ncrisc_noc_fast_write_ok(uint32_t noc, uint32_t cmd_buf) {
  return (NOC_CMD_BUF_READ_REG(noc, cmd_buf, NOC_CMD_CTRL) == NOC_CTRL_STATUS_READY);
}

inline __attribute__((always_inline)) bool ncrisc_noc_fast_write_ok_l1(uint32_t noc, uint32_t cmd_buf) {
  return (NOC_CMD_BUF_READ_REG_L1(noc, cmd_buf, NOC_CMD_CTRL) == NOC_CTRL_STATUS_READY);
}

inline void ncrisc_noc_blitz_write_setup(uint32_t noc, uint32_t cmd_buf, uint64_t dest_addr, uint32_t len_bytes, uint32_t vc, uint32_t num_times_to_write) {
  uint32_t noc_cmd_field =
    NOC_CMD_CPY | NOC_CMD_WR |
    NOC_CMD_VC_STATIC  |
    NOC_CMD_STATIC_VC(vc) |
    NOC_CMD_RESP_MARKED;

  while (!ncrisc_noc_fast_write_ok(noc, cmd_buf));
  NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_CTRL, noc_cmd_field);
  NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_AT_LEN_BE, len_bytes);
  NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_RET_ADDR_MID, dest_addr >> 32);
  noc_nonposted_writes_num_issued[noc] += num_times_to_write;
  noc_nonposted_writes_acked[noc] += num_times_to_write;
}

inline bool ncrisc_noc_nonposted_writes_sent(uint32_t noc) {
  return (NOC_STATUS_READ_REG(noc, NIU_MST_NONPOSTED_WR_REQ_SENT) == noc_nonposted_writes_num_issued[noc]);
}

inline __attribute__((always_inline)) bool ncrisc_noc_nonposted_writes_sent_l1(uint32_t noc) {
  return (NOC_STATUS_READ_REG_L1(noc, NIU_MST_NONPOSTED_WR_REQ_SENT) == noc_nonposted_writes_num_issued[noc]);
}

inline bool ncrisc_noc_nonposted_writes_flushed(uint32_t noc) {
  return (NOC_STATUS_READ_REG(noc, NIU_MST_WR_ACK_RECEIVED) == noc_nonposted_writes_acked[noc]);
}

inline __attribute__((always_inline)) bool ncrisc_noc_nonposted_writes_flushed_l1(uint32_t noc) {
  return (NOC_STATUS_READ_REG_L1(noc, NIU_MST_WR_ACK_RECEIVED) == noc_nonposted_writes_acked[noc]);
}


// modified to initialize just the specified NOC
// this way BRISC can initialize NOC-0, and NCRISC NOC-1
// renamed to noc_init
inline void noc_init(int noc) {
//  for (int noc = 0; noc < NUM_NOCS; noc++) {
    uint32_t noc_id_reg = NOC_CMD_BUF_READ_REG(noc, 0, NOC_NODE_ID);
    uint32_t my_x = noc_id_reg & NOC_NODE_ID_MASK;
    uint32_t my_y = (noc_id_reg >> NOC_ADDR_NODE_ID_BITS) & NOC_NODE_ID_MASK;
    uint64_t xy_local_addr = NOC_XY_ADDR(my_x, my_y, 0);

    NOC_CMD_BUF_WRITE_REG(noc, NCRISC_WR_CMD_BUF, NOC_TARG_ADDR_MID, (uint32_t)(xy_local_addr >> 32));
    NOC_CMD_BUF_WRITE_REG(noc, NCRISC_WR_REG_CMD_BUF, NOC_TARG_ADDR_MID, (uint32_t)(xy_local_addr >> 32));

    uint32_t noc_rd_cmd_field = NOC_CMD_CPY | NOC_CMD_RD | NOC_CMD_RESP_MARKED | NOC_CMD_VC_STATIC | NOC_CMD_STATIC_VC(1);

    NOC_CMD_BUF_WRITE_REG(noc, NCRISC_RD_CMD_BUF, NOC_CTRL, noc_rd_cmd_field);
    NOC_CMD_BUF_WRITE_REG(noc, NCRISC_RD_CMD_BUF, NOC_RET_ADDR_MID, (uint32_t)(xy_local_addr >> 32));

    noc_reads_num_issued[noc] = NOC_STATUS_READ_REG(noc, NIU_MST_RD_RESP_RECEIVED);
    noc_nonposted_writes_num_issued[noc] = NOC_STATUS_READ_REG(noc, NIU_MST_NONPOSTED_WR_REQ_SENT);
    noc_nonposted_writes_acked[noc] = NOC_STATUS_READ_REG(noc, NIU_MST_WR_ACK_RECEIVED);
//  }
}


inline void ncrisc_noc_fast_read_any_len(uint32_t noc, uint32_t cmd_buf, uint64_t src_addr, uint32_t dest_addr, uint32_t len_bytes) {
  while (len_bytes > NOC_MAX_BURST_SIZE) {
    while (!ncrisc_noc_fast_read_ok(noc, cmd_buf));
    ncrisc_noc_fast_read(noc, cmd_buf, src_addr, dest_addr, NOC_MAX_BURST_SIZE);
    src_addr += NOC_MAX_BURST_SIZE;
    dest_addr += NOC_MAX_BURST_SIZE;
    len_bytes -= NOC_MAX_BURST_SIZE;
  }
  while (!ncrisc_noc_fast_read_ok(noc, cmd_buf));
  ncrisc_noc_fast_read(noc, cmd_buf, src_addr, dest_addr, len_bytes);
}


inline __attribute__((always_inline)) void ncrisc_noc_fast_read_any_len_l1(uint32_t noc, uint32_t cmd_buf, uint64_t src_addr, uint32_t dest_addr, uint32_t len_bytes) {
  while (len_bytes > NOC_MAX_BURST_SIZE) {
    while (!ncrisc_noc_fast_read_ok_l1(noc, cmd_buf));
    ncrisc_noc_fast_read_l1(noc, cmd_buf, src_addr, dest_addr, NOC_MAX_BURST_SIZE);
    src_addr += NOC_MAX_BURST_SIZE;
    dest_addr += NOC_MAX_BURST_SIZE;
    len_bytes -= NOC_MAX_BURST_SIZE;
  }
  while (!ncrisc_noc_fast_read_ok_l1(noc, cmd_buf));
  ncrisc_noc_fast_read_l1(noc, cmd_buf, src_addr, dest_addr, len_bytes);
}


inline void ncrisc_noc_fast_write_any_len(uint32_t noc, uint32_t cmd_buf, uint32_t src_addr, uint64_t dest_addr, uint32_t len_bytes, uint32_t vc, bool mcast, bool linked, uint32_t num_dests) {
  while (len_bytes > NOC_MAX_BURST_SIZE) {
    while (!ncrisc_noc_fast_write_ok(noc, cmd_buf));
    ncrisc_noc_fast_write(noc, cmd_buf, src_addr, dest_addr, NOC_MAX_BURST_SIZE, vc, mcast, linked, num_dests);
    src_addr += NOC_MAX_BURST_SIZE;
    dest_addr += NOC_MAX_BURST_SIZE;
    len_bytes -= NOC_MAX_BURST_SIZE;
  }
  while (!ncrisc_noc_fast_write_ok(noc, cmd_buf));
  ncrisc_noc_fast_write(noc, cmd_buf, src_addr, dest_addr, len_bytes, vc, mcast, linked, num_dests);
}

inline void ncrisc_noc_fast_write_any_len_loopback_src(uint32_t noc, uint32_t cmd_buf, uint32_t src_addr, uint64_t dest_addr, uint32_t len_bytes, uint32_t vc, bool mcast, bool linked, uint32_t num_dests) {
  while (len_bytes > NOC_MAX_BURST_SIZE) {
    while (!ncrisc_noc_fast_write_ok(noc, cmd_buf));
    ncrisc_noc_fast_write_loopback_src(noc, cmd_buf, src_addr, dest_addr, NOC_MAX_BURST_SIZE, vc, mcast, linked, num_dests);
    src_addr += NOC_MAX_BURST_SIZE;
    dest_addr += NOC_MAX_BURST_SIZE;
    len_bytes -= NOC_MAX_BURST_SIZE;
  }
  while (!ncrisc_noc_fast_write_ok(noc, cmd_buf));
  ncrisc_noc_fast_write_loopback_src(noc, cmd_buf, src_addr, dest_addr, len_bytes, vc, mcast, linked, num_dests);
}

inline __attribute__((always_inline)) void ncrisc_noc_fast_write_any_len_l1(uint32_t noc, uint32_t cmd_buf, uint32_t src_addr, uint64_t dest_addr, uint32_t len_bytes, uint32_t vc, bool mcast, bool linked, uint32_t num_dests) {
  while (len_bytes > NOC_MAX_BURST_SIZE) {
    while (!ncrisc_noc_fast_write_ok_l1(noc, cmd_buf));
    ncrisc_noc_fast_write_l1(noc, cmd_buf, src_addr, dest_addr, NOC_MAX_BURST_SIZE, vc, mcast, linked, num_dests);
    src_addr += NOC_MAX_BURST_SIZE;
    dest_addr += NOC_MAX_BURST_SIZE;
    len_bytes -= NOC_MAX_BURST_SIZE;
  }
  while (!ncrisc_noc_fast_write_ok_l1(noc, cmd_buf));
  ncrisc_noc_fast_write_l1(noc, cmd_buf, src_addr, dest_addr, len_bytes, vc, mcast, linked, num_dests);
}

inline void noc_fast_posted_write_dw_inline(uint32_t noc, uint32_t cmd_buf, uint32_t val, uint64_t dest_addr, uint32_t be, uint32_t static_vc, bool mcast) {
  bool posted = true;
  bool static_vc_alloc = true;
  uint32_t noc_cmd_field =
    (static_vc_alloc ? NOC_CMD_VC_STATIC : 0x0) |
    NOC_CMD_STATIC_VC(static_vc) |
    NOC_CMD_CPY | NOC_CMD_WR |
    NOC_CMD_WR_INLINE |
    (mcast ? (NOC_CMD_PATH_RESERVE | NOC_CMD_BRCST_PACKET) : 0x0) |
    (posted ? 0x0 : NOC_CMD_RESP_MARKED);

  uint32_t be32 = be;
  uint32_t be_shift = (dest_addr & (NOC_WORD_BYTES-1));
  be32 = (be32 << be_shift);

  while (NOC_CMD_BUF_READ_REG(noc, cmd_buf, NOC_CMD_CTRL) != NOC_CTRL_STATUS_READY);
  NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_AT_DATA, val);
  NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_CTRL, noc_cmd_field);
  NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_TARG_ADDR_LO, (uint32_t)(dest_addr & ~(NOC_WORD_BYTES-1)));
  NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_TARG_ADDR_MID, dest_addr >> 32);
  NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_AT_LEN_BE, be32);
  NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_CMD_CTRL, NOC_CTRL_SEND_REQ);
}

inline void noc_fast_atomic_increment(uint32_t noc, uint32_t cmd_buf, uint64_t addr, uint32_t incr, uint32_t wrap, bool linked) {
  while (!ncrisc_noc_fast_write_ok(noc, cmd_buf));
  NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_TARG_ADDR_LO, (uint32_t)(addr & 0xFFFFFFFF));
  NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_TARG_ADDR_MID, (uint32_t)(addr >> 32));
  NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_CTRL, (linked ? NOC_CMD_VC_LINKED : 0x0) | NOC_CMD_AT);
  NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_AT_LEN_BE, NOC_AT_INS(NOC_AT_INS_INCR_GET) | NOC_AT_WRAP(wrap) | NOC_AT_IND_32((addr>>2) & 0x3) | NOC_AT_IND_32_SRC(0));
  NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_AT_DATA, incr);
  NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_CMD_CTRL, 0x1);
}

inline void noc_fast_atomic_increment_l1(uint32_t noc, uint32_t cmd_buf, uint64_t addr, uint32_t incr, uint32_t wrap, bool linked) {
  NOC_CMD_BUF_WRITE_REG_L1(noc, cmd_buf, NOC_TARG_ADDR_LO, (uint32_t)(addr & 0xFFFFFFFF));
  NOC_CMD_BUF_WRITE_REG_L1(noc, cmd_buf, NOC_TARG_ADDR_MID, (uint32_t)(addr >> 32));
  NOC_CMD_BUF_WRITE_REG_L1(noc, cmd_buf, NOC_CTRL, (linked ? NOC_CMD_VC_LINKED : 0x0) | NOC_CMD_AT);
  NOC_CMD_BUF_WRITE_REG_L1(noc, cmd_buf, NOC_AT_LEN_BE, NOC_AT_INS(NOC_AT_INS_INCR_GET) | NOC_AT_WRAP(wrap) | NOC_AT_IND_32((addr>>2) & 0x3) | NOC_AT_IND_32_SRC(0));
  NOC_CMD_BUF_WRITE_REG_L1(noc, cmd_buf, NOC_AT_DATA, incr);
  NOC_CMD_BUF_WRITE_REG_L1(noc, cmd_buf, NOC_CMD_CTRL, 0x1);
}
