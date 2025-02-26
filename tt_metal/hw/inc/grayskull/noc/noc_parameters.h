// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#ifndef _NOC_PARAMETERS_H_
#define _NOC_PARAMETERS_H_

#ifndef NOC_X_SIZE
#define NOC_X_SIZE 1
#endif

#ifndef NOC_Y_SIZE
#define NOC_Y_SIZE 1
#endif

// Coordinate Virtualization is not supported on GS (feature does not exist in NOC Hardware).
#define VIRTUAL_TENSIX_START_X 0
#define VIRTUAL_TENSIX_START_Y 0
#define COORDINATE_VIRTUALIZATION_ENABLED 0

#define NUM_NOCS 2
#define NUM_TENSIXES 120

#define NOC_REG_SPACE_START_ADDR 0xFF000000
#define NOC_REGS_START_ADDR 0xFFB20000
#define NOC_CMD_BUF_OFFSET 0x00000400
#define NOC_CMD_BUF_OFFSET_BIT 10
#define NOC_INSTANCE_OFFSET 0x00010000
#define NOC_INSTANCE_OFFSET_BIT 16
#define NOC_CMD_BUF_INSTANCE_OFFSET(noc, buf) ((buf << NOC_CMD_BUF_OFFSET_BIT) + (noc << NOC_INSTANCE_OFFSET_BIT))

////
// NIU master IF control registers:

#define NOC_TARG_ADDR_LO (NOC_REGS_START_ADDR + 0x0)
#define NOC_TARG_ADDR_MID (NOC_REGS_START_ADDR + 0x4)
#define NOC_TARG_ADDR_HI (NOC_REGS_START_ADDR + 0x8)

#define NOC_RET_ADDR_LO (NOC_REGS_START_ADDR + 0xC)
#define NOC_RET_ADDR_MID (NOC_REGS_START_ADDR + 0x10)
#define NOC_RET_ADDR_HI (NOC_REGS_START_ADDR + 0x14)

#define NOC_PACKET_TAG_OFFSET (NOC_REGS_START_ADDR + 0x18)

#define NOC_CTRL (NOC_REGS_START_ADDR + 0x1C)
#define NOC_AT_LEN_BE (NOC_REGS_START_ADDR + 0x20)
#define NOC_AT_DATA (NOC_REGS_START_ADDR + 0x24)

#define NOC_CMD_CTRL (NOC_REGS_START_ADDR + 0x28)
#define NOC_NODE_ID (NOC_REGS_START_ADDR + 0x2C)
#define NOC_CHIP_ID (NOC_REGS_START_ADDR + 0x30)

#define NUM_MEM_PARITY_ERR (NOC_REGS_START_ADDR + 0x40)
#define NUM_HEADER_1B_ERR (NOC_REGS_START_ADDR + 0x44)
#define NUM_HEADER_2B_ERR (NOC_REGS_START_ADDR + 0x48)
#define ECC_CTRL (NOC_REGS_START_ADDR + 0x4C)  // [2:0] = clear ECC interrupts, [5:3] = force ECC error
////

#define NOC_STATUS(cnt) (NOC_REGS_START_ADDR + 0x200 + ((cnt) * 4))
// status/performance counter registers
// IMPROVE: add offsets for misc. debug status regiters

// NIU DBG registers start at 0x40

#define ROUTER_PORT_X_VC14_VC15_DBG 0x3d
#define ROUTER_PORT_X_VC12_VC13_DBG 0x3c
#define ROUTER_PORT_X_VC10_VC11_DBG 0x3b
#define ROUTER_PORT_X_VC8_VC9_DBG 0x3a
#define ROUTER_PORT_X_VC6_VC7_DBG 0x39
#define ROUTER_PORT_X_VC4_VC5_DBG 0x38
#define ROUTER_PORT_X_VC2_VC3_DBG 0x37
#define ROUTER_PORT_X_VC0_VC1_DBG 0x36

#define ROUTER_PORT_Y_VC14_VC15_DBG 0x35
#define ROUTER_PORT_Y_VC12_VC13_DBG 0x34
#define ROUTER_PORT_Y_VC10_VC11_DBG 0x33
#define ROUTER_PORT_Y_VC8_VC9_DBG 0x32
#define ROUTER_PORT_Y_VC6_VC7_DBG 0x31
#define ROUTER_PORT_Y_VC4_VC5_DBG 0x30
#define ROUTER_PORT_Y_VC2_VC3_DBG 0x2f
#define ROUTER_PORT_Y_VC0_VC1_DBG 0x2e

// 0x2a to 0x2d are 0
#define ROUTER_PORT_NIU_VC6_VC7_DBG 0x29
#define ROUTER_PORT_NIU_VC4_VC5_DBG 0x28
#define ROUTER_PORT_NIU_VC2_VC3_DBG 0x27
#define ROUTER_PORT_NIU_VC0_VC1_DBG 0x26

#define ROUTER_PORT_X_OUT_VC_FULL_CREDIT_OUT_VC_STALL 0x24
#define ROUTER_PORT_Y_OUT_VC_FULL_CREDIT_OUT_VC_STALL 0x22
#define ROUTER_PORT_NIU_OUT_VC_FULL_CREDIT_OUT_VC_STALL 0x20

#define NIU_SLV_POSTED_WR_REQ_STARTED 0x1D
#define NIU_SLV_NONPOSTED_WR_REQ_STARTED 0x1C
#define NIU_SLV_POSTED_WR_REQ_RECEIVED 0x1B
#define NIU_SLV_NONPOSTED_WR_REQ_RECEIVED 0x1A
#define NIU_SLV_POSTED_WR_DATA_WORD_RECEIVED 0x19
#define NIU_SLV_NONPOSTED_WR_DATA_WORD_RECEIVED 0x18
#define NIU_SLV_POSTED_ATOMIC_RECEIVED 0x17
#define NIU_SLV_NONPOSTED_ATOMIC_RECEIVED 0x16
#define NIU_SLV_RD_REQ_RECEIVED 0x15

#define NIU_SLV_REQ_ACCEPTED 0x14
#define NIU_SLV_RD_DATA_WORD_SENT 0x13
#define NIU_SLV_RD_RESP_SENT 0x12
#define NIU_SLV_WR_ACK_SENT 0x11
#define NIU_SLV_ATOMIC_RESP_SENT 0x10

#define NIU_MST_NONPOSTED_ATOMIC_STARTED 0xF
#define NIU_MST_RD_REQ_STARTED 0xE
#define NIU_MST_POSTED_WR_REQ_STARTED 0xD
#define NIU_MST_NONPOSTED_WR_REQ_STARTED 0xC
#define NIU_MST_POSTED_WR_REQ_SENT 0xB
#define NIU_MST_NONPOSTED_WR_REQ_SENT 0xA
#define NIU_MST_POSTED_WR_DATA_WORD_SENT 0x9
#define NIU_MST_NONPOSTED_WR_DATA_WORD_SENT 0x8
#define NIU_MST_POSTED_ATOMIC_SENT 0x7
#define NIU_MST_NONPOSTED_ATOMIC_SENT 0x6
#define NIU_MST_RD_REQ_SENT 0x5

#define NIU_MST_CMD_ACCEPTED 0x4
#define NIU_MST_RD_DATA_WORD_RECEIVED 0x3
#define NIU_MST_RD_RESP_RECEIVED 0x2
#define NIU_MST_WR_ACK_RECEIVED 0x1
#define NIU_MST_ATOMIC_RESP_RECEIVED 0x0

/////

#define NOC_CFG(cnt) (NOC_REGS_START_ADDR + 0x100 + ((cnt) * 4))

#define NIU_CFG_0 \
    0x0  // 0 = clk gt enable, [7:1] = clk gt hysteresis, [8] = NIU mem parity enable, [11:9] = ECC interrupts enable,
         // [12] = tile clock disable
#define NIU_CFG_0_ECC_NIU_MEM_PARITY_EN 8
#define NIU_CFG_0_ECC_MEM_PARITY_INT_EN 9
#define NIU_CFG_0_ECC_HEADER_1B_INT_EN 10
#define NIU_CFG_0_ECC_HEADER_2B_INT_EN 11
#define NIU_CFG_0_TILE_CLK_OFF 12

#define ROUTER_CFG_0 \
    0x1  // 0 = clk gt enable, [7:1] = clk gt hysteresis, [11:8] = max_backoff_exp, [15:12] = log2_basic_timeout_delay,
         // [16] = router mem parity enable, [17] = packet header chk bits enable, [18] = packet header SECDED enable
#define ROUTER_CFG_0_ECC_ROUTER_MEM_PARITY_EN 16
#define ROUTER_CFG_0_ECC_HEADER_CHKBITS_EN 17
#define ROUTER_CFG_0_ECC_HEADER_SECDED_EN 18
#define ROUTER_CFG_1 0x2  // broadcast disable row
#define ROUTER_CFG_2 0x3
#define ROUTER_CFG_3 0x4  // broadcast disable column
#define ROUTER_CFG_4 0x5

/////

// Flit types
#define NOC_HEAD_FLIT 0x1
#define NOC_BODY_FLIT 0x2
#define NOC_TAIL_FLIT 0x4
#define NOC_FLIT_TYPE_WIDTH 3

// addr fields
#define NOC_ADDR_LOCAL_BITS 32
#define NOC_ADDR_NODE_ID_BITS 6

// NOC CMD fields
#define NOC_CMD_AT (0x1 << 0)
#define NOC_CMD_CPY (0x0 << 0)
#define NOC_CMD_RD (0x0 << 1)
#define NOC_CMD_WR (0x1 << 1)
#define NOC_CMD_WR_BE (0x1 << 2)
#define NOC_CMD_WR_INLINE (0x1 << 3)
#define NOC_CMD_RESP_MARKED (0x1 << 4)
#define NOC_CMD_BRCST_PACKET (0x1 << 5)
#define NOC_CMD_VC_LINKED (0x1 << 6)
#define NOC_CMD_VC_STATIC (0x1 << 7)
#define NOC_CMD_PATH_RESERVE (0x1 << 8)
#define NOC_CMD_MEM_RD_DROP_ACK (0x1 << 9)
#define NOC_CMD_STATIC_VC(vc) (((uint32_t)(vc)) << 13)

#define NOC_CMD_BRCST_XY(y) (((uint32_t)(y)) << 16)
#define NOC_CMD_BRCST_SRC_INCLUDE (0x1 << 17)
#define NOC_CMD_ARB_PRIORITY(p) (((uint32_t)(p)) << 27)

//
// NOC CTRL fields
#define NOC_CTRL_SEND_REQ (0x1 << 0)
//
#define NOC_CTRL_STATUS_READY 0x0
// Atomic command codes
#define NOC_AT_INS_NOP 0x0
#define NOC_AT_INS_INCR_GET 0x1
#define NOC_AT_INS_INCR_GET_PTR 0x2
#define NOC_AT_INS_SWAP 0x3
#define NOC_AT_INS_CAS 0x4
#define NOC_AT_INS_GET_TILE_MAP 0x5
#define NOC_AT_INS_STORE_IND 0x6
#define NOC_AT_INS_SWAP_4B 0x7

#define NOC_AT_IND_32(index) ((index) << 0)
#define NOC_AT_IND_32_SRC(index) ((index) << 10)
#define NOC_AT_WRAP(wrap) ((wrap) << 2)
// #define NOC_AT_INCR(incr)        ((incr) << 6)
#define NOC_AT_INS(ins) ((ins) << 12)
#define NOC_AT_TILE_MAP_IND(ind) ((ind) << 2)

///

#define NOC_DATA_WIDTH 259
#define NOC_PAYLOAD_WIDTH 256
#define NOC_WORD_BYTES (NOC_PAYLOAD_WIDTH / 8)
#define NOC_MAX_BURST_WORDS 256
#define NOC_MAX_BURST_SIZE (NOC_MAX_BURST_WORDS * NOC_WORD_BYTES)
// #define MEM_WORD_BYTES      16
#define NOC_WORD_OFFSET_MASK (NOC_WORD_BYTES - 1)

#define MEM_DATA_WIDTH 128
#define MEM_WORD_BYTES (MEM_DATA_WIDTH / 8)
#define MEM_WORD_OFFSET_MASK (MEM_WORD_BYTES - 1)

#define NOC_VCS 16

#define NOC_BCAST_VC_START 4

#define NOC_ROUTER_PORTS 3
#define NOC_PORT_NIU 0
#define NOC_PORT_X 1
#define NOC_PORT_Y 2

////

#define NOC_NODE_ID_MASK ((((uint64_t)0x1) << NOC_ADDR_NODE_ID_BITS) - 1)
#define NOC_LOCAL_ADDR_MASK ((((uint64_t)0x1) << NOC_ADDR_LOCAL_BITS) - 1)

#define NOC_LOCAL_ADDR_OFFSET(addr) ((addr) & NOC_LOCAL_ADDR_MASK)

#define NOC_UNICAST_ADDR_X(addr) (((addr) >> NOC_ADDR_LOCAL_BITS) & NOC_NODE_ID_MASK)
#define NOC_UNICAST_ADDR_Y(addr) (((addr) >> (NOC_ADDR_LOCAL_BITS + NOC_ADDR_NODE_ID_BITS)) & NOC_NODE_ID_MASK)

#define NOC_MCAST_ADDR_END_X(addr) (((addr) >> NOC_ADDR_LOCAL_BITS) & NOC_NODE_ID_MASK)
#define NOC_MCAST_ADDR_END_Y(addr) (((addr) >> (NOC_ADDR_LOCAL_BITS + NOC_ADDR_NODE_ID_BITS)) & NOC_NODE_ID_MASK)
#define NOC_MCAST_ADDR_START_X(addr) (((addr) >> (NOC_ADDR_LOCAL_BITS + 2 * NOC_ADDR_NODE_ID_BITS)) & NOC_NODE_ID_MASK)
#define NOC_MCAST_ADDR_START_Y(addr) (((addr) >> (NOC_ADDR_LOCAL_BITS + 3 * NOC_ADDR_NODE_ID_BITS)) & NOC_NODE_ID_MASK)

// Addres formats

#define NOC_XY_ADDR(x, y, addr)                                                                                      \
    ((((uint64_t)(y)) << (NOC_ADDR_LOCAL_BITS + NOC_ADDR_NODE_ID_BITS)) | (((uint64_t)(x)) << NOC_ADDR_LOCAL_BITS) | \
     ((uint64_t)(addr)))

#define NOC_MULTICAST_ADDR(x_start, y_start, x_end, y_end, addr)                    \
    ((((uint64_t)(x_start)) << (NOC_ADDR_LOCAL_BITS + 2 * NOC_ADDR_NODE_ID_BITS)) | \
     (((uint64_t)(y_start)) << (NOC_ADDR_LOCAL_BITS + 3 * NOC_ADDR_NODE_ID_BITS)) | \
     (((uint64_t)(x_end)) << NOC_ADDR_LOCAL_BITS) |                                 \
     (((uint64_t)(y_end)) << (NOC_ADDR_LOCAL_BITS + NOC_ADDR_NODE_ID_BITS)) | ((uint64_t)(addr)))

// GS address encoding is 32 bits of address followed by coordinate. First address goes into lo register, coordinates
// are in the mid register
#define NOC_COORD_REG_OFFSET 0  // offset (from LSB) in register holding x-y coordinate

// Address formats
#define NOC_XY_ENCODING(x, y) ((((uint32_t)(y)) << (NOC_ADDR_NODE_ID_BITS)) | (((uint32_t)(x))))

#define NOC_XY_PCIE_ENCODING(x, y) ((uint64_t(NOC_XY_ENCODING(x, y)) << (NOC_ADDR_LOCAL_BITS - NOC_COORD_REG_OFFSET)))

#define NOC_MULTICAST_ENCODING(x_start, y_start, x_end, y_end)                                          \
    ((x_start) << (2 * NOC_ADDR_NODE_ID_BITS)) | ((y_start) << (3 * NOC_ADDR_NODE_ID_BITS)) | (x_end) | \
        ((y_end) << (NOC_ADDR_NODE_ID_BITS))

#define NOC_XY_ADDR2(xy, addr) ((((uint64_t)(xy)) << NOC_ADDR_LOCAL_BITS) | ((uint64_t)(addr)))

// Pass-through for WH and GS, special cased for BH
#define NOC_LOCAL_ADDR(addr) NOC_LOCAL_ADDR_OFFSET(addr)

// Alignment restrictions
#define NOC_L1_READ_ALIGNMENT_BYTES 16
#define NOC_L1_WRITE_ALIGNMENT_BYTES 16
#define NOC_PCIE_READ_ALIGNMENT_BYTES 32
#define NOC_PCIE_WRITE_ALIGNMENT_BYTES 16
#define NOC_DRAM_READ_ALIGNMENT_BYTES 32
#define NOC_DRAM_WRITE_ALIGNMENT_BYTES 16

#define L1_ALIGNMENT                                                                              \
    (static_cast<uint32_t>(                                                                       \
        NOC_L1_READ_ALIGNMENT_BYTES >= NOC_L1_WRITE_ALIGNMENT_BYTES ? NOC_L1_READ_ALIGNMENT_BYTES \
                                                                    : NOC_L1_WRITE_ALIGNMENT_BYTES))
#define PCIE_ALIGNMENT                                                                                  \
    (static_cast<uint32_t>(                                                                             \
        NOC_PCIE_READ_ALIGNMENT_BYTES >= NOC_PCIE_WRITE_ALIGNMENT_BYTES ? NOC_PCIE_READ_ALIGNMENT_BYTES \
                                                                        : NOC_PCIE_WRITE_ALIGNMENT_BYTES))
#define DRAM_ALIGNMENT                                                                                  \
    (static_cast<uint32_t>(                                                                             \
        NOC_DRAM_READ_ALIGNMENT_BYTES >= NOC_DRAM_WRITE_ALIGNMENT_BYTES ? NOC_DRAM_READ_ALIGNMENT_BYTES \
                                                                        : NOC_DRAM_WRITE_ALIGNMENT_BYTES))

#endif
