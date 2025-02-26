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

// Coordinate Virtualization is fully supported by WH NOC Hardware and Firmware.
// Tensix cores start at coorddinate <x = 18, y = 18> in Virtual Space and are contiguous.
#define VIRTUAL_TENSIX_START_X 18
#define VIRTUAL_TENSIX_START_Y 18
#define COORDINATE_VIRTUALIZATION_ENABLED 1

#define NUM_NOCS 2
#define NUM_TENSIXES 80

#define NOC_MAX_TRANSACTION_ID 0xF
#define NOC_MAX_TRANSACTION_ID_COUNT 255

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

#define NOC_PACKET_TAG (NOC_REGS_START_ADDR + 0x18)

#define NOC_CTRL (NOC_REGS_START_ADDR + 0x1C)
#define NOC_AT_LEN_BE (NOC_REGS_START_ADDR + 0x20)
#define NOC_AT_DATA (NOC_REGS_START_ADDR + 0x24)

#define NOC_CMD_CTRL (NOC_REGS_START_ADDR + 0x28)
#define NOC_NODE_ID (NOC_REGS_START_ADDR + 0x2C)
#define NOC_ENDPOINT_ID (NOC_REGS_START_ADDR + 0x30)

#define NUM_MEM_PARITY_ERR (NOC_REGS_START_ADDR + 0x40)
#define NUM_HEADER_1B_ERR (NOC_REGS_START_ADDR + 0x44)
#define NUM_HEADER_2B_ERR (NOC_REGS_START_ADDR + 0x48)
#define ECC_CTRL (NOC_REGS_START_ADDR + 0x4C)  // [2:0] = clear ECC interrupts, [5:3] = force ECC error

#define NOC_CLEAR_OUTSTANDING_REQ_CNT (NOC_REGS_START_ADDR + 0x50)

////

#define NOC_STATUS(cnt) (NOC_REGS_START_ADDR + 0x200 + ((cnt) * 4))
// status/performance counter registers
// IMPROVE: add offsets for misc. debug status regiters

#define NIU_SLV_POSTED_WR_REQ_STARTED 0x3D
#define NIU_SLV_NONPOSTED_WR_REQ_STARTED 0x3C
#define NIU_SLV_POSTED_WR_REQ_RECEIVED 0x3B
#define NIU_SLV_NONPOSTED_WR_REQ_RECEIVED 0x3A
#define NIU_SLV_POSTED_WR_DATA_WORD_RECEIVED 0x39
#define NIU_SLV_NONPOSTED_WR_DATA_WORD_RECEIVED 0x38
#define NIU_SLV_POSTED_ATOMIC_RECEIVED 0x37
#define NIU_SLV_NONPOSTED_ATOMIC_RECEIVED 0x36
#define NIU_SLV_RD_REQ_RECEIVED 0x35

#define NIU_SLV_REQ_ACCEPTED 0x34
#define NIU_SLV_RD_DATA_WORD_SENT 0x33
#define NIU_SLV_RD_RESP_SENT 0x32
#define NIU_SLV_WR_ACK_SENT 0x31
#define NIU_SLV_ATOMIC_RESP_SENT 0x30

#define NIU_MST_WRITE_REQS_OUTGOING_ID(id) (0x20 + (id))
#define NIU_MST_REQS_OUTSTANDING_ID(id) (0x10 + (id))

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

// 0 = clk gt enable
// [7:1] = clk gt hysteresis
// [8] = NIU mem parity enable
// [11:9] = ECC interrupts enable
// [12] = tile clock disable
// [13] = (noc2axi only) header double store disable
// [14] = enable coordinate translation
#define NIU_CFG_0 0x0
#define NIU_CFG_0_ECC_NIU_MEM_PARITY_EN 8
#define NIU_CFG_0_ECC_MEM_PARITY_INT_EN 9
#define NIU_CFG_0_ECC_HEADER_1B_INT_EN 10
#define NIU_CFG_0_ECC_HEADER_2B_INT_EN 11
#define NIU_CFG_0_TILE_CLK_OFF 12
#define NIU_CFG_0_TILE_HEADER_STORE_OFF 13  // NOC2AXI only
#define NIU_CFG_0_NOC_ID_TRANSLATE_EN 14
// NCRISC is using NIU_CFG_0[31:24] to store debug postcodes, if you need these bits for hardware move ncrisc postcode
// write location in ncrisc.cc.

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

#define NOC_X_ID_TRANSLATE_TABLE_0 0x6  // entries 0-7 in the X ID translation table (total 32 x 4 bit entries)
#define NOC_X_ID_TRANSLATE_TABLE_1 0x7  // entries 8-15 in the X ID translation table (total 32 x 4 bit entries)
#define NOC_X_ID_TRANSLATE_TABLE_2 0x8  // entries 16-23 in the X ID translation table (total 32 x 4 bit entries)
#define NOC_X_ID_TRANSLATE_TABLE_3 0x9  // entries 24-31 in the X ID translation table (total 32 x 4 bit entries)

#define NOC_Y_ID_TRANSLATE_TABLE_0 0xA  // entries 0-7 in the Y ID translation table (total 32 x 4 bit entries)
#define NOC_Y_ID_TRANSLATE_TABLE_1 0xB  // entries 8-15 in the Y ID translation table (total 32 x 4 bit entries)
#define NOC_Y_ID_TRANSLATE_TABLE_2 0xC  // entries 16-23 in the Y ID translation table (total 32 x 4 bit entries)
#define NOC_Y_ID_TRANSLATE_TABLE_3 0xD  // entries 24-31 in the Y ID translation table (total 32 x 4 bit entries)

#define NOC_ID_LOGICAL \
    0xE  // logical coordinates of the local NOC NIU if ID translation is enabled (format = {logical_y[5:0],
         // logical_x[5:0]})

/////

// Flit types
#define NOC_HEAD_FLIT 0x1
#define NOC_BODY_FLIT 0x2
#define NOC_TAIL_FLIT 0x4
#define NOC_FLIT_TYPE_WIDTH 3

// addr fields
#define NOC_ADDR_LOCAL_BITS 36
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

#define NOC_PACKET_TAG_TRANSACTION_ID(id) ((id) << 10)
#define NOC_PACKET_TAG_HEADER_STORE (0x1 << 9)

///

#define NOC_DATA_WIDTH 259
#define NOC_PAYLOAD_WIDTH 256
#define NOC_WORD_BYTES (NOC_PAYLOAD_WIDTH / 8)
#define NOC_MAX_BURST_WORDS 256
#ifdef RISC_A0_HW
#define NOC_MAX_BURST_SIZE 512
#else
#define NOC_MAX_BURST_SIZE (NOC_MAX_BURST_WORDS * NOC_WORD_BYTES)
#endif
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

// 36 bits of address followed by coordinate. First 32 bits of address go into lo register, remaining address bits and
// coordinates are in the mid register
#define NOC_COORD_REG_OFFSET 4  // offset (from LSB) in register holding x-y coordinate

// Address formats
#define NOC_XY_ENCODING(x, y)                                                   \
    (((uint32_t)(y)) << ((NOC_ADDR_LOCAL_BITS % 32) + NOC_ADDR_NODE_ID_BITS)) | \
        (((uint32_t)(x)) << (NOC_ADDR_LOCAL_BITS % 32))

// Address formats
#define NOC_XY_PCIE_ENCODING(x, y) \
    ((uint64_t(NOC_XY_ENCODING(x, y)) << (NOC_ADDR_LOCAL_BITS - NOC_COORD_REG_OFFSET)) | 0x800000000)

#define NOC_MULTICAST_ENCODING(x_start, y_start, x_end, y_end)                                \
    (((uint32_t)(x_start)) << ((NOC_ADDR_LOCAL_BITS % 32) + 2 * NOC_ADDR_NODE_ID_BITS)) |     \
        (((uint32_t)(y_start)) << ((NOC_ADDR_LOCAL_BITS % 32) + 3 * NOC_ADDR_NODE_ID_BITS)) | \
        (((uint32_t)(x_end)) << (NOC_ADDR_LOCAL_BITS % 32)) |                                 \
        (((uint32_t)(y_end)) << ((NOC_ADDR_LOCAL_BITS % 32) + NOC_ADDR_NODE_ID_BITS))

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
