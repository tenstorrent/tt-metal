// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#ifndef _NOC_PARAMETERS_H_
#define _NOC_PARAMETERS_H_

#define VIRTUAL_TENSIX_START_X 1
#define VIRTUAL_TENSIX_START_Y 2
#define COORDINATE_VIRTUALIZATION_ENABLED 1

#define NUM_NOCS 2
#define NUM_TENSIXES 140

#define NOC_MAX_TRANSACTION_ID 0xF
#define NOC_MAX_TRANSACTION_ID_COUNT 255

#define NOC_REG_SPACE_START_ADDR 0xFF000000
#define NOC_REGS_START_ADDR 0xFFB20000
#define NOC_CMD_BUF_OFFSET 0x00000800
#define NOC_CMD_BUF_OFFSET_BIT 11
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
#define NOC_AT_LEN_BE_1 (NOC_REGS_START_ADDR + 0x24)
#define NOC_AT_DATA (NOC_REGS_START_ADDR + 0x28)
#define NOC_BRCST_EXCLUDE (NOC_REGS_START_ADDR + 0x2C)
#define NOC_L1_ACC_AT_INSTRN (NOC_REGS_START_ADDR + 0x30)
#define NOC_SEC_CTRL (NOC_REGS_START_ADDR + 0x34)

#define NOC_CMD_CTRL (NOC_REGS_START_ADDR + 0x40)
#define NOC_NODE_ID (NOC_REGS_START_ADDR + 0x44)
#define NOC_ENDPOINT_ID (NOC_REGS_START_ADDR + 0x48)

#define NUM_MEM_PARITY_ERR (NOC_REGS_START_ADDR + 0x50)
#define NUM_HEADER_1B_ERR (NOC_REGS_START_ADDR + 0x54)
#define NUM_HEADER_2B_ERR (NOC_REGS_START_ADDR + 0x58)
#define ECC_CTRL (NOC_REGS_START_ADDR + 0x5C)  // [2:0] = clear ECC interrupts, [5:3] = force ECC error

#define NOC_CLEAR_OUTSTANDING_REQ_CNT (NOC_REGS_START_ADDR + 0x60)
#define CMD_BUF_AVAIL (NOC_REGS_START_ADDR + 0x64)  // [28:24], [20:16], [12:8], [4:0]
#define CMD_BUF_OVFL (NOC_REGS_START_ADDR + 0x68)

#define NOC_SEC_FENCE_RANGE(cnt) (NOC_REGS_START_ADDR + 0x400 + ((cnt) * 4))      // 32 inst
#define NOC_SEC_FENCE_ATTRIBUTE(cnt) (NOC_REGS_START_ADDR + 0x480 + ((cnt) * 4))  // 8 inst
#define NOC_SEC_FENCE_MASTER_LEVEL (NOC_REGS_START_ADDR + 0x4A0)
#define NOC_SEC_FENCE_FIFO_STATUS (NOC_REGS_START_ADDR + 0x4A4)
#define NOC_SEC_FENCE_FIFO_RDDATA (NOC_REGS_START_ADDR + 0x4A8)

// 16 VC, 64 bit registers, 2 ports
#define PORT1_FLIT_COUNTER_LOWER(vc) (NOC_REGS_START_ADDR + 0x500 + ((vc) * 8))
#define PORT1_FLIT_COUNTER_UPPER(vc) (NOC_REGS_START_ADDR + 0x504 + ((vc) * 8))

#define PORT2_FLIT_COUNTER_LOWER(vc) (NOC_REGS_START_ADDR + 0x580 + ((vc) * 8))
#define PORT2_FLIT_COUNTER_UPPER(vc) (NOC_REGS_START_ADDR + 0x584 + ((vc) * 8))

////

#define NOC_STATUS(cnt) (NOC_REGS_START_ADDR + 0x200 + ((cnt) * 4))

// status/performance counter registers
// IMPROVE: add offsets for misc. debug status regiters

// from noc/rtl/tt_noc_params.svh
// parameter TOTAL_STATUS_REGS = NIU_STATUS_REGS + MST_IF_INTP_STATUS_REGS + ROUTER_STATUS_REGS + SLV_IF_STATUS_REGS +
// MST_IF_STATUS_REGS; // 32+2+30+16+48=128
// NIU_STATUS        : 0x60-0x7F
// MST_IF_INTP_STATUS: 0x5E-0x5F
// ROUTER_STATUS     : 0x40-0x5D
// SLV_IF_STATUS     : 0x30-0x3F
// MST_IF_STATUS     : 0x 0-0x2F

#define NIU_TRANS_COUNT_RTZ_NUM 0x5E
#define NIU_TRANS_COUNT_RTZ_SOURCE 0x5F

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
#define NIU_CFG_0_AXI_SUBORDINATE_ENABLE 15
#define NIU_CFG_0_CMD_BUFFER_FIFO_EN 16
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

#define NOC_TRANSLATE_ID_WIDTH 5
#define NOC_TRANSLATE_TABLE_XY_SIZE (32 / NOC_TRANSLATE_ID_WIDTH)

#define NOC_X_ID_TRANSLATE_TABLE_0 0x6  // entries 0-5 in the X ID translation table (total 32 x 5 bit entries)
#define NOC_X_ID_TRANSLATE_TABLE_1 0x7  // entries 6-11 in the X ID translation table (total 32 x 5 bit entries)
#define NOC_X_ID_TRANSLATE_TABLE_2 0x8  // entries 12-17 in the X ID translation table (total 32 x 5 bit entries)
#define NOC_X_ID_TRANSLATE_TABLE_3 0x9  // entries 18-23 in the X ID translation table (total 32 x 5 bit entries)
#define NOC_X_ID_TRANSLATE_TABLE_4 0xA  // entries 24-29 in the X ID translation table (total 32 x 5 bit entries)
#define NOC_X_ID_TRANSLATE_TABLE_5 0xB  // entries 30-31 in the X ID translation table (total 32 x 5 bit entries)

#define NOC_Y_ID_TRANSLATE_TABLE_0 0xC   // entries 0-5 in the Y ID translation table (total 32 x 5 bit entries)
#define NOC_Y_ID_TRANSLATE_TABLE_1 0xD   // entries 6-11 in the Y ID translation table (total 32 x 5 bit entries)
#define NOC_Y_ID_TRANSLATE_TABLE_2 0xE   // entries 12-17 in the Y ID translation table (total 32 x 5 bit entries)
#define NOC_Y_ID_TRANSLATE_TABLE_3 0xF   // entries 18-23 in the Y ID translation table (total 32 x 5 bit entries)
#define NOC_Y_ID_TRANSLATE_TABLE_4 0x10  // entries 24-29 in the X ID translation table (total 32 x 5 bit entries)
#define NOC_Y_ID_TRANSLATE_TABLE_5 0x11  // entries 30-31 in the X ID translation table (total 32 x 5 bit entries)

#define NOC_ID_LOGICAL \
    0x12  // logical coordinates of the local NOC NIU if ID translation is enabled (format = {logical_y[5:0],
          // logical_x[5:0]})
#define MEMORY_SHUTDOWN_CONTROL \
    0x13  // controls Shutdown (bit0), Deepsleep (bit1), Retention Disable for Deepsleep (bit2)
#define MEMORY_SD_BIT 0
#define MEMORY_DSLP_BIT 1
#define MEMORY_DSLPLV_BIT 2
#define NOC_ID_TRANSLATE_COL_MASK 0x14       // Mask to indication with column would ignore ID translation
#define NOC_ID_TRANSLATE_ROW_MASK 0x15       // Mask to indication with row    would ignore ID translation
#define DDR_COORD_TRANSLATE_TABLE_0 0x16     // entries  0- 5 in the DDR translation table (total 32 x 5 bit entries)
#define DDR_COORD_TRANSLATE_TABLE_1 0x17     // entries  6-11 in the DDR translation table (total 32 x 5 bit entries)
#define DDR_COORD_TRANSLATE_TABLE_2 0x18     // entries 12-17 in the DDR translation table (total 32 x 5 bit entries)
#define DDR_COORD_TRANSLATE_TABLE_3 0x19     // entries 18-23 in the DDR translation table (total 32 x 5 bit entries)
#define DDR_COORD_TRANSLATE_TABLE_4 0x1A     // entries 24-29 in the DDR translation table (total 32 x 5 bit entries)
#define DDR_COORD_TRANSLATE_TABLE_5 0x1B     // entries 30-31 in the DDR translation table (total 32 x 5 bit entries)
#define DDR_COORD_TRANSLATE_COL_SEL_WIDTH 2  //
#define DDR_COORD_TRANSLATE_COL_SEL_EAST 10  // if bit is set, ddr translation applies to column 0.
#define DDR_COORD_TRANSLATE_COL_SEL_WEST 11  // if bit is set, ddr translation applies to column 9.
#define DDR_COORD_TRANSLATE_COL_SWAP 0x1C    // entries 30-31 in the DDR translation table (total 32 x 5 bit entries)

#define DEBUG_COUNTER_RESET \
    0x1D  // write 1 to reset counter; self-clearing, as a reset pulse is generated when written.
          // bit 0 - resets ROUTER_OUTGOING_FLIT_COUNTER
          // bit 4 - clears CMD_BUFFER_FIFO_OVFL_FLAG
#define ROUTER_OUTGOING_FLIT_COUNTER_BIT 0
#define CMD_BUFFER_FIFO_OVFL_CLEAR_BIT 4
#define NIU_TRANS_COUNT_RTZ_CFG 0x1E
#define NIU_TRANS_COUNT_RTZ_CLR 0x1F

/////

// Flit types
#define NOC_HEAD_FLIT 0x1
#define NOC_BODY_FLIT 0x2
#define NOC_TAIL_FLIT 0x4
#define NOC_FLIT_TYPE_WIDTH 3

// addr fields
// MM Jul 21 2022: For backwards compatibility, all the BH NoC API functions
// will accept a 36-bit address and left-pad it to 64-bits within the function
#define NOC_ADDR_LOCAL_BITS /*64*/ 36
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
#define NOC_CMD_L1_ACC_AT_EN (0x1 << 31)

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
#define NOC_AT_INS_ACC 0x9

#define NOC_AT_IND_32(index) ((index) << 0)
#define NOC_AT_IND_32_SRC(index) ((index) << 10)
#define NOC_AT_WRAP(wrap) ((wrap) << 2)
// #define NOC_AT_INCR(incr)         ((incr) << 6)
#define NOC_AT_INS(ins) ((ins) << 12)
#define NOC_AT_TILE_MAP_IND(ind) ((ind) << 2)
#define NOC_AT_ACC_FORMAT(format) (((format) << 0) & 0x7)
#define NOC_AT_ACC_SAT_DIS(dis) ((dis) << 3)

///

#define NOC_AT_ACC_FP32 0x0
#define NOC_AT_ACC_FP16_A 0x1
#define NOC_AT_ACC_FP16_B 0x2
#define NOC_AT_ACC_INT32 0x3
#define NOC_AT_ACC_INT32_COMPL 0x4
#define NOC_AT_ACC_INT32_UNS 0x5
#define NOC_AT_ACC_INT8 0x6

///

#define NOC_PACKET_TAG_TRANSACTION_ID(id) ((id) << 10)
#define NOC_PACKET_TAG_HEADER_STORE (0x1 << 9)

///

#define NOC_DATA_WIDTH 512 + 3
#define NOC_PAYLOAD_WIDTH 512
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

#define NOC_UNICAST_COORDINATE_Y(noc_coordinate) (((noc_coordinate) >> (1 * NOC_ADDR_NODE_ID_BITS)) & NOC_NODE_ID_MASK)
#define NOC_UNICAST_COORDINATE_X(noc_coordinate) (((noc_coordinate) >> (0 * NOC_ADDR_NODE_ID_BITS)) & NOC_NODE_ID_MASK)

#define NOC_MCAST_COORDINATE_START_Y(noc_coordinate) \
    (((noc_coordinate) >> (3 * NOC_ADDR_NODE_ID_BITS)) & NOC_NODE_ID_MASK)
#define NOC_MCAST_COORDINATE_START_X(noc_coordinate) \
    (((noc_coordinate) >> (2 * NOC_ADDR_NODE_ID_BITS)) & NOC_NODE_ID_MASK)
#define NOC_MCAST_COORDINATE_END_Y(noc_coordinate) \
    (((noc_coordinate) >> (1 * NOC_ADDR_NODE_ID_BITS)) & NOC_NODE_ID_MASK)
#define NOC_MCAST_COORDINATE_END_X(noc_coordinate) \
    (((noc_coordinate) >> (0 * NOC_ADDR_NODE_ID_BITS)) & NOC_NODE_ID_MASK)

// Addres formats

#define NOC_XY_ADDR(x, y, addr)                                                                                      \
    ((((uint64_t)(y)) << (NOC_ADDR_LOCAL_BITS + NOC_ADDR_NODE_ID_BITS)) | (((uint64_t)(x)) << NOC_ADDR_LOCAL_BITS) | \
     ((uint64_t)(addr)))

#define NOC_MULTICAST_ADDR(x_start, y_start, x_end, y_end, addr)                    \
    ((((uint64_t)(x_start)) << (NOC_ADDR_LOCAL_BITS + 2 * NOC_ADDR_NODE_ID_BITS)) | \
     (((uint64_t)(y_start)) << (NOC_ADDR_LOCAL_BITS + 3 * NOC_ADDR_NODE_ID_BITS)) | \
     (((uint64_t)(x_end)) << NOC_ADDR_LOCAL_BITS) |                                 \
     (((uint64_t)(y_end)) << (NOC_ADDR_LOCAL_BITS + NOC_ADDR_NODE_ID_BITS)) | ((uint64_t)(addr)))

#define NOC_XY_COORD(x, y) ((((uint32_t)(y)) << NOC_ADDR_NODE_ID_BITS) | ((uint32_t)(x)))

#define NOC_MULTICAST_COORD(x_start, y_start, x_end, y_end)                                                            \
    ((((uint32_t)(y_start)) << (3 * NOC_ADDR_NODE_ID_BITS)) | (((uint32_t)(x_start)) << (2 * NOC_ADDR_NODE_ID_BITS)) | \
     (((uint32_t)(y_end)) << (1 * NOC_ADDR_NODE_ID_BITS)) | ((uint32_t)(x_end)))

// BH has 64 bit address space but pipegen was not updated to support this so WH scheme of encoding addresses is used
// (36 bits of address followed by coordinates) This means that lo and mid registers need to have the address portion
// while the coordinates go into hi register
#define NOC_COORD_REG_OFFSET 0  // offset (from LSB) in register holding x-y coordinate

// Addres formats

#define NOC_XY_ENCODING(x, y) ((((uint32_t)(y)) << (NOC_ADDR_NODE_ID_BITS)) | (((uint32_t)(x))))

// Base address pulled from tt::umd::Cluster::get_pcie_base_addr_from_device
#define NOC_XY_PCIE_ENCODING(x, y) \
    ((uint64_t(NOC_XY_ENCODING(x, y)) << (NOC_ADDR_LOCAL_BITS - NOC_COORD_REG_OFFSET)) | 0x1000000000000000)

#define NOC_MULTICAST_ENCODING(x_start, y_start, x_end, y_end)                                                         \
    ((((uint32_t)(x_start)) << (2 * NOC_ADDR_NODE_ID_BITS)) | (((uint32_t)(y_start)) << (3 * NOC_ADDR_NODE_ID_BITS)) | \
     (((uint32_t)(x_end))) | (((uint32_t)(y_end)) << (NOC_ADDR_NODE_ID_BITS)))

// Because BH uses WH style address encoding (36 bits followed by coordinates) but PCIe transactions require bit 60 to
// be set, we need to mask out the xy-coordinate When NOC_ADDR_LOCAL_BITS is 64 then NOC_LOCAL_ADDR_OFFSET can be used
// and the below define can be deprecated
#define NOC_LOCAL_ADDR(addr) ((addr) & 0x1000000FFFFFFFFF)

// Alignment restrictions
#define NOC_L1_READ_ALIGNMENT_BYTES 16
#define NOC_L1_WRITE_ALIGNMENT_BYTES 16
#define NOC_PCIE_READ_ALIGNMENT_BYTES 64
#define NOC_PCIE_WRITE_ALIGNMENT_BYTES 16
#define NOC_DRAM_READ_ALIGNMENT_BYTES 64
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
