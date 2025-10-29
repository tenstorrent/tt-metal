// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_tensix_noc_overlay_reg.h"

#ifndef _NOC_PARAMETERS_H_
#define _NOC_PARAMETERS_H_

// TODO: review these values
#define VIRTUAL_TENSIX_START_X 1
#define VIRTUAL_TENSIX_START_Y 2
#define COORDINATE_VIRTUALIZATION_ENABLED 1

#ifndef NOC_X_SIZE
#define NOC_X_SIZE 4
#endif

#ifndef NOC_Y_SIZE
#ifdef NOC_OVERLAY_TOP_LEVEL
#define NOC_Y_SIZE 8
#else
#define NOC_Y_SIZE 4
#endif
#endif

#define NUM_NOCS 1
#define NOC_ID_WIDTH 6

#define NOC_MAX_TRANSACTION_ID 0xF

// FLEX Port defnes
#define FLEX_PORT_ADDR_HASH_SIZE 8
#define FLEX_PORT_ADDR_HASH_BIT_WIDTH 18
#define FLEX_PORT_ADDR_MATCH_BIT_WIDTH 10

#define NOC_REGS_START_ADDR TT_NOC_REG_MAP_BASE_ADDR
#define NOC_REG_SPACE_START_ADDR NOC_REGS_START_ADDR

#define NOC_CMD_BUF_OFFSET 0x00000800
#define NOC_CMD_BUF_OFFSET_BIT 11
//! USE_4_NOCS--#define NOC_INSTANCE_OFFSET       0x00008000
//! USE_4_NOCS--#define NOC_INSTANCE_OFFSET_BIT   15
#define NOC_INSTANCE_OFFSET 0x00010000
#define NOC_INSTANCE_OFFSET_BIT 16
#define NOC_CMD_BUF_INSTANCE_OFFSET(noc, buf) (((buf) << NOC_CMD_BUF_OFFSET_BIT) + ((noc) << NOC_INSTANCE_OFFSET_BIT))

////
// NIU master IF control registers:

#define NOC_TARG_ADDR_LO (NOC_REGS_START_ADDR + 0x0)
#define NOC_TARG_ADDR_MID (NOC_REGS_START_ADDR + 0x4)
#define NOC_TARG_ADDR_HI (NOC_REGS_START_ADDR + 0x8)

#define NOC_RET_ADDR_LO (NOC_REGS_START_ADDR + 0xC)
#define NOC_RET_ADDR_MID (NOC_REGS_START_ADDR + 0x10)
#define NOC_RET_ADDR_HI (NOC_REGS_START_ADDR + 0x14)

// #define NOC_PACKET_TAG           (NOC_REGS_START_ADDR+0x18)
#define NOC_CTRL_LO (NOC_REGS_START_ADDR + 0x18)
#define NOC_CTRL_HI (NOC_REGS_START_ADDR + 0x1C)
#define NOC_BRCST_LO (NOC_REGS_START_ADDR + 0x20)
#define NOC_BRCST_HI (NOC_REGS_START_ADDR + 0x24)
#define NOC_AT_LEN (NOC_REGS_START_ADDR + 0x28)
#define NOC_L1_ACC_AT_INSTRN (NOC_REGS_START_ADDR + 0x2C)
#define NOC_SEC_CTRL (NOC_REGS_START_ADDR + 0x30)
#define NOC_AT_DATA (NOC_REGS_START_ADDR + 0x34)
#define NOC_INLINE_DATA_LO (NOC_REGS_START_ADDR + 0x38)
#define NOC_INLINE_DATA_HI (NOC_REGS_START_ADDR + 0x3C)
#define NOC_BYTE_ENABLE_0 (NOC_REGS_START_ADDR + 0x40)
#define NOC_BYTE_ENABLE_1 (NOC_REGS_START_ADDR + 0x44)
#define NOC_BYTE_ENABLE_2 (NOC_REGS_START_ADDR + 0x48)
#define NOC_BYTE_ENABLE_3 (NOC_REGS_START_ADDR + 0x4C)
#define NOC_BYTE_ENABLE_4 (NOC_REGS_START_ADDR + 0x50)
#define NOC_BYTE_ENABLE_5 (NOC_REGS_START_ADDR + 0x54)
#define NOC_BYTE_ENABLE_6 (NOC_REGS_START_ADDR + 0x58)
#define NOC_BYTE_ENABLE_7 (NOC_REGS_START_ADDR + 0x5C)
#define NOC_CMD_CTRL (NOC_REGS_START_ADDR + 0x60)
#define NOC_NODE_ID (NOC_REGS_START_ADDR + 0x64)
#define NOC_ENDPOINT_ID (NOC_REGS_START_ADDR + 0x68)

#define NUM_MEM_PARITY_ERR (NOC_REGS_START_ADDR + 0x6C)
#define NUM_HEADER_1B_ERR (NOC_REGS_START_ADDR + 0x70)
#define NUM_HEADER_2B_ERR (NOC_REGS_START_ADDR + 0x74)
#define ECC_CTRL (NOC_REGS_START_ADDR + 0x78)  // [2:0] = clear ECC interrupts, [5:3] = force ECC error

#define NOC_CLEAR_OUTSTANDING_REQ_CNT (NOC_REGS_START_ADDR + 0x7C)
#define CMD_BUF_AVAIL (NOC_REGS_START_ADDR + 0x80)  // [28:24], [20:16], [12:8], [4:0]
#define CMD_BUF_OVFL (NOC_REGS_START_ADDR + 0x84)

#define NOC_SENT_TARG_ADDR_LO (NOC_REGS_START_ADDR + 0x88)
#define NOC_SENT_TARG_ADDR_MID (NOC_REGS_START_ADDR + 0x8C)
#define NOC_SENT_TARG_ADDR_HI (NOC_REGS_START_ADDR + 0x90)

#define NOC_SENT_RET_ADDR_LO (NOC_REGS_START_ADDR + 0x94)
#define NOC_SENT_RET_ADDR_MID (NOC_REGS_START_ADDR + 0x98)
#define NOC_SENT_RET_ADDR_HI (NOC_REGS_START_ADDR + 0x9C)

// #define NOC_PACKET_TAG           (NOC_REGS_START_ADDR+0x18)
#define NOC_SENT_CTRL_LO (NOC_REGS_START_ADDR + 0xA0)
#define NOC_SENT_CTRL_HI (NOC_REGS_START_ADDR + 0xA4)
#define NOC_SENT_BRCST_LO (NOC_REGS_START_ADDR + 0xA8)
#define NOC_SENT_BRCST_HI (NOC_REGS_START_ADDR + 0xAC)
#define NOC_SENT_AT_LEN (NOC_REGS_START_ADDR + 0xB0)
#define NOC_SENT_L1_ACC_AT_INSTRN (NOC_REGS_START_ADDR + 0xB4)
#define NOC_SENT_SEC_CTRL (NOC_REGS_START_ADDR + 0xB8)
#define NOC_SENT_AT_DATA (NOC_REGS_START_ADDR + 0xBC)
#define NOC_SENT_INLINE_DATA_LO (NOC_REGS_START_ADDR + 0xC0)
#define NOC_SENT_INLINE_DATA_HI (NOC_REGS_START_ADDR + 0xC4)
#define NOC_SENT_BYTE_ENABLE_0 (NOC_REGS_START_ADDR + 0xC8)
#define NOC_SENT_BYTE_ENABLE_1 (NOC_REGS_START_ADDR + 0xCC)
#define NOC_SENT_BYTE_ENABLE_2 (NOC_REGS_START_ADDR + 0xD0)
#define NOC_SENT_BYTE_ENABLE_3 (NOC_REGS_START_ADDR + 0xD4)
#define NOC_SENT_BYTE_ENABLE_4 (NOC_REGS_START_ADDR + 0xD8)
#define NOC_SENT_BYTE_ENABLE_5 (NOC_REGS_START_ADDR + 0xDC)
#define NOC_SENT_BYTE_ENABLE_6 (NOC_REGS_START_ADDR + 0xE0)
#define NOC_SENT_BYTE_ENABLE_7 (NOC_REGS_START_ADDR + 0xE4)

#define NOC_SEC_FENCE_RANGE(cnt) (NOC_REGS_START_ADDR + 0x400 + ((cnt) * 4))      // 32 inst
#define NOC_SEC_FENCE_ATTRIBUTE(cnt) (NOC_REGS_START_ADDR + 0x480 + ((cnt) * 4))  // 8 inst
#define NOC_SEC_FENCE_MASTER_LEVEL (NOC_REGS_START_ADDR + 0x4A0)
#define NOC_SEC_FENCE_FIFO_STATUS (NOC_REGS_START_ADDR + 0x4A4)
#define NOC_SEC_FENCE_FIFO_RDDATA (NOC_REGS_START_ADDR + 0x4A8)

// Output router - 16 VC, 32 bit registers, 4 ports
#define PORT1_OUT_FLIT_COUNTER(vc) (NOC_REGS_START_ADDR + 0x600 + ((vc) * 4))
#define PORT2_OUT_FLIT_COUNTER(vc) (NOC_REGS_START_ADDR + 0x640 + ((vc) * 4))
#define PORT3_OUT_FLIT_COUNTER(vc) (NOC_REGS_START_ADDR + 0x680 + ((vc) * 4))
#define PORT4_OUT_FLIT_COUNTER(vc) (NOC_REGS_START_ADDR + 0x6C0 + ((vc) * 4))

// Input router - 16 VC, 32 bit registers, 4 ports
#define PORT1_IN_FLIT_COUNTER(vc) (NOC_REGS_START_ADDR + 0x700 + ((vc) * 4))
#define PORT2_IN_FLIT_COUNTER(vc) (NOC_REGS_START_ADDR + 0x740 + ((vc) * 4))
#define PORT3_IN_FLIT_COUNTER(vc) (NOC_REGS_START_ADDR + 0x780 + ((vc) * 4))
#define PORT4_IN_FLIT_COUNTER(vc) (NOC_REGS_START_ADDR + 0x7C0 + ((vc) * 4))

////

#define NOC_CFG(cnt) (NOC_REGS_START_ADDR + 0x100 + ((cnt) * 4))

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

#define NIU_TRANS_COUNT_RTZ_SOURCE 0x3F
#define NIU_TRANS_COUNT_RTZ_NUM 0x3E

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
#define NIU_CFG_0_NIU_TIMEOUT_IRQ_EN 14
#define NIU_CFG_0_AXI_SLAVE_ENABLE 15
#define NIU_CFG_0_CMD_BUFFER_FIFO_EN 16
#define NIU_CFG_0_AUTOINLINE_DISABLE 17
#define NIU_CFG_0_ALWAYS_FLUSH 18
// NCRISC is using NIU_CFG_0[31:24] to store debug postcodes, if you need these bits for hardware move ncrisc postcode
// write location in ncrisc.cc.

#define ROUTER_CFG_0 \
    0x1  // [    0] = clk gt enable,
         // [ 7: 1] = clk gt hysteresis,
         // [11: 8] = max_backoff_exp,
         // [15:12] = log2_basic_timeout_delay,
         // [   16] = router mem parity enable,
         // [   17] = packet header chk bits enable,
         // [   18] = packet header SECDED enable
#define ROUTER_CFG_0_ECC_ROUTER_MEM_PARITY_EN 16
#define ROUTER_CFG_0_ECC_HEADER_CHKBITS_EN 17
#define ROUTER_CFG_0_ECC_HEADER_SECDED_EN 18
#define ROUTER_CFG_1 0x2  // broadcast disable row
#define ROUTER_CFG_2 0x3
#define ROUTER_CFG_3 0x4  // broadcast disable column
#define ROUTER_CFG_4 0x5

#define MEMORY_SHUTDOWN_CONTROL \
    0x6  // controls Shutdown (bit0), Deepsleep (bit1), Retention Disable for Deepsleep (bit2)

#define DEBUG_COUNTER_RESET \
    0x7  // write 1 to reset counter; self-clearing, as a reset pulse is generated when written.
         // bit 0 - resets ROUTER_OUTGOING_FLIT_COUNTER
         // bit 4 - clears CMD_BUFFER_FIFO_OVFL_FLAG
#define ROUTER_OUTGOING_FLIT_COUNTER_BIT 0
#define CMD_BUFFER_FIFO_OVFL_CLEAR_BIT 4
#define NIU_TRANS_COUNT_RTZ_CFG 0x8
#define NIU_TRANS_COUNT_RTZ_CLR 0x9
#define VC_DIM_ORDER 0xa
#define THROTTLER_CYCLES_PER_WINDOW 0xb
#define THROTTLER_HANDSHAKES_PER_WINDOW_NIU 0xc
#define THROTTLER_HANDSHAKES_PER_WINDOW_NORTH 0xd
#define THROTTLER_HANDSHAKES_PER_WINDOW_EAST 0xe
#define THROTTLER_HANDSHAKES_PER_WINDOW_SOUTH 0xf
#define THROTTLER_HANDSHAKES_PER_WINDOW_WEST 0x10
#define NIU_TIMEOUT_DETECTED 0x11
#define NIU_TIMEOUT_VALUE_0 0x12
#define NIU_TIMEOUT_VALUE_1 0x13
#define INVALID_FENCE_START_ADDR_LO_0 0x14
#define INVALID_FENCE_START_ADDR_HI_0 0x15
#define INVALID_FENCE_END_ADDR_LO_0 0x16
#define INVALID_FENCE_END_ADDR_HI_0 0x17
#define INVALID_FENCE_START_ADDR_LO_1 0x18
#define INVALID_FENCE_START_ADDR_HI_1 0x19
#define INVALID_FENCE_END_ADDR_LO_1 0x1a
#define INVALID_FENCE_END_ADDR_HI_1 0x1b
#define INVALID_FENCE_START_ADDR_LO_2 0x1c
#define INVALID_FENCE_START_ADDR_HI_2 0x1d
#define INVALID_FENCE_END_ADDR_LO_2 0x1e
#define INVALID_FENCE_END_ADDR_HI_2 0x1f
#define FLEX_PORT_ADDR_FN0 0x24
#define VC_THROTTLER_HANDSHAKES_PER_WINDOW_0 0x2C
#define VC_THROTTLER_HANDSHAKES_PER_WINDOW_1 0x2D
#define VC_THROTTLER_HANDSHAKES_PER_WINDOW_2 0x2E
#define VC_THROTTLER_HANDSHAKES_PER_WINDOW_3 0x2F
#define VC_THROTTLER_CYCLES_PER_WINDOW 0x30
#define FLEX_PORT_ADDR_FN1 0x31
#define FLEX_PORT_ADDR_FN0_MASK_MATCH 0x39
#define FLEX_PORT_ADDR_FN1_MASK_MATCH 0x3A
#define FLEX_PORT_IN_ORDER_MASK 0x3B
#define FLEX_PORT_IN_ORDER_MATCH 0x3C
#define FLEX_PORT_RW_GLOBAL_ENABLE 0x3D

#define MEMORY_SD_BIT 0
#define MEMORY_DSLP_BIT 1
#define MEMORY_DSLPLV_BIT 2

/////

// Flit types
#define NOC_HEAD_FLIT 0x1
#define NOC_BODY_FLIT 0x2
#define NOC_TAIL_FLIT 0x4
#define NOC_FLIT_TYPE_WIDTH 3

// addr fields
#define NOC_ADDR_LOCAL_BITS 36 /*64*/
#define NOC_ADDR_NODE_ID_BITS 6

// NOC CMD fields
// CMD_LO
#define NOC_CMD_AT (0x1 << 0)
#define NOC_CMD_CPY (0x0 << 0)
#define NOC_CMD_RD (0x0 << 1)
#define NOC_CMD_WR (0x1 << 1)
#define NOC_CMD_WR_BE (0x1 << 2)
#define NOC_CMD_WR_INLINE (0x1 << 3)
#define NOC_CMD_WR_INLINE_64 (0x1 << 4)
#define NOC_CMD_RESP_MARKED (0x1 << 5)
#define NOC_CMD_BRCST_PACKET (0x1 << 6)
#define NOC_CMD_VC_LINKED (0x1 << 7)
#define NOC_CMD_PATH_RESERVE (0x1 << 8)
#define NOC_CMD_MEM_RD_DROP_ACK (0x1 << 9)
#define NOC_CMD_DYNA_ROUTING_EN (0x1 << 10)
#define NOC_CMD_L1_ACC_AT (0x1 << 11)
#define NOC_CMD_FLUSH (0x1 << 12)
#define NOC_CMD_SNOOP (0x1 << 13)
#define NOC_CMD_STATIC_VC(vc) (((uint32_t)(vc)) << 14)
#define NOC_RESP_STATIC_VC(vc) (((uint32_t)(vc)) << 20)
#define NOC_CMD_PORT_REQ_MASK(m) (((uint32_t)(m)) << 26)
#define NOC_CMD_VC_STATIC (0x1 << 15)  // TODO remove

// CMD_HI
#define NOC_CMD_PKT_TAG_ID(id) (((uint32_t)(id)) << 0)
#define NOC_CMD_FORCE_DIM_ROUTING (0x1 << 4)

// NOC Broadcast fields
// BRCST_LO
#define NOC_CMD_BRCST_XY(y) (((uint32_t)(y)) << 0)
#define NOC_BRCST_SRC_INCLUDE (0x1 << 1)
#define NOC_BRCST_CTRL_STATE(x) (((uint32_t)(x)) << 2)
#define NOC_BRCST_ACTIVE_NODE (0x1 << 4)
#define NOC_BRCST_CTRL_END_NODE(x) (((uint32_t)(x)) << 5)
#define NOC_BRCST_QUAD_EXCL_ENABLE (0x1 << 13)
#define NOC_BRCST_QUAD_EXCL_COORD_X(x) (((uint32_t)(x)) << 14)
#define NOC_BRCST_QUAD_EXCL_COORD_Y(y) (((uint32_t)(y)) << 22)
#define NOC_BRCST_QUAD_EXCL_DIR_X (0x1 << 30)
#define NOC_BRCST_QUAD_EXCL_DIR_Y (0x1 << 31)

// BRCST_HI
#define NOC_BRCST_STRIDED_KEEP_X(x) (((uint32_t)(x)) << 0)
#define NOC_BRCST_STRIDED_SKIP_X(x) (((uint32_t)(x)) << 2)
#define NOC_BRCST_STRIDED_KEEP_Y(y) (((uint32_t)(y)) << 4)
#define NOC_BRCST_STRIDED_SKIP_Y(y) (((uint32_t)(y)) << 6)

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
#define NOC_AT_INS_ECC_RMW 0x8
#define NOC_AT_INS_ACC 0x9
#define NOC_AT_INS_RISCV_AMO 0xA

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

#define NOC_DATA_WIDTH (2048 + 3)
#define NOC_PAYLOAD_WIDTH 2048
#define NOC_WORD_BYTES (NOC_PAYLOAD_WIDTH / 8)
#define NOC_MAX_BURST_WORDS 256
#define NOC_MAX_BURST_SIZE (NOC_MAX_BURST_WORDS * NOC_WORD_BYTES)
// #define MEM_WORD_BYTES      16
#define NOC_WORD_OFFSET_MASK (NOC_WORD_BYTES - 1)

#define MEM_DATA_WIDTH 128
#define MEM_WORD_BYTES (MEM_DATA_WIDTH / 8)
#define MEM_WORD_OFFSET_MASK (MEM_WORD_BYTES - 1)

#define NOC_VCS 16

#define NOC_BCAST_VC_START 8

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

#define NOC_COORD_REG_OFFSET 0  // offset (from LSB) in register holding x-y coordinate

#define NOC_XY_ENCODING(x, y) ((((uint32_t)(y)) << (NOC_ADDR_NODE_ID_BITS)) | (((uint32_t)(x))))

// Base address pulled from tt::umd::Cluster::get_pcie_base_addr_from_device
#define NOC_XY_PCIE_ENCODING(x, y) \
    ((uint64_t(NOC_XY_ENCODING(x, y)) << (NOC_ADDR_LOCAL_BITS - NOC_COORD_REG_OFFSET)) | 0x1000000000000000)

#define NOC_LOCAL_ADDR(addr) (addr)

// TODO review these alignment restrictions
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
