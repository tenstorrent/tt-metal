// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Quasar overlay command-buffer definitions shared by the RoCC-based NOC APIs
// (noc_nonblocking_api_v2.h and noc_nonblocking_api_v3.h): register layout
// constants and wrappers, MISC/VC values, counter bookkeeping, command-buffer
// initialization, and the flush/barrier predicates. Moved verbatim from
// noc_nonblocking_api_v2.h. The NOC_V2_* names are kept for the shared values
// (VCs, packetization limit, static transaction id): they describe the RoCC
// transport generation, which V3 shares, not the API version.

#include <cstdint>
#include "internal/risc_attribs.h"
#include "noc_parameters.h"
#include "hostdev/dev_msgs.h"
#include "noc_overlay_parameters.h"
#include "api/debug/assert.h"
#include "internal/tt-2xx/quasar/overlay/rocc_instructions.hpp"

#if !defined(COMPILE_FOR_DM)
#error "NOC API V2 requires COMPILE_FOR_DM (uses RoCC custom instructions)"
#endif

constexpr std::underlying_type_t<TensixProcessorTypes> proc_type =
    static_cast<std::underlying_type_t<TensixProcessorTypes>>(COMPILE_FOR_DM);

// Helper functions to convert NoC coordinates to NoC-0 coordinates, used in metal as "physical" coordinates.
#define NOC_0_X(noc_index, noc_size_x, x) x
#define NOC_0_Y(noc_index, noc_size_y, y) y
#define NOC_0_X_PHYS_COORD(noc_index, noc_size_x, x) x
#define NOC_0_Y_PHYS_COORD(noc_index, noc_size_y, y) y
#define MY_NOC_ENCODING(noc_index) NOC_CMD_BUF_READ_REG(noc_index, 0, NOC_NODE_ID)

// Quasar overlay command buffer indices (3 buffers: write, read, atomic)
#define OVERLAY_WR_CMD_BUF 0
#define OVERLAY_RD_CMD_BUF 1
#define OVERLAY_AT_CMD_BUF 2

/* Qsr has 64 bit addresses, use same encoding as BH and WH */
constexpr uint32_t NOC_ADDR_COORD_SHIFT = 36;
const uint32_t NOC_TARG_ADDR_COORDINATE = NOC_TARG_ADDR_HI;
const uint32_t NOC_RET_ADDR_COORDINATE = NOC_RET_ADDR_HI;
const uint32_t NOC_COORDINATE_MASK = 0xFFFFFF;

// ToDo check with Keranous if this is correct
constexpr uint32_t NOC_PCIE_MASK = 0x1000000F;

constexpr uint32_t WRITE_RESPONSE_STATIC_VC = 14;
constexpr uint32_t READ_RESPONSE_STATIC_VC = 12;

// NOC V2 command buffer VC assignments (same HW values as overlay::CMDBUF_*_VC)
constexpr uint32_t NOC_V2_RD_REQ_VC = 1;
constexpr uint32_t NOC_V2_RD_RESP_VC = 12;
constexpr uint32_t NOC_V2_WR_REQ_VC = 1;
constexpr uint32_t NOC_V2_WR_RESP_VC = 13;
constexpr uint32_t NOC_V2_MCAST_REQ_VC = 8;
constexpr uint32_t NOC_V2_MCAST_RESP_VC = 14;

// Static transaction ID used for all command buffers
constexpr uint32_t NOC_V2_TRID_STATIC = 0;

// Per-cmd-buf packetization limit programmed at boot. 8KB; lower than the 64KB HW default to avoid NOC congestion.
constexpr uint32_t NOC_V2_MAX_BYTES_IN_PACKET = 8 * 1024;

// ============================================================================
// CMD_BUF_MISC Register Bit Definitions (TT_ROCC_CMD_BUF_MISC_reg_t)
// ============================================================================
// Individual bit positions for the MISC register
constexpr uint64_t CMD_BUF_MISC_LINKED = (1 << 0);              // bit 0:  linked transaction
constexpr uint64_t CMD_BUF_MISC_POSTED = (1 << 1);              // bit 1:  posted (no ack)
constexpr uint64_t CMD_BUF_MISC_INLINE_WR = (1 << 2);           // bit 2:  inline write
constexpr uint64_t CMD_BUF_MISC_MULTICAST = (1 << 3);           // bit 3:  multicast enable
constexpr uint64_t CMD_BUF_MISC_MULTICAST_MODE = (1 << 4);      // bit 4:  multicast mode
constexpr uint64_t CMD_BUF_MISC_SRC_INCLUDE = (1 << 5);         // bit 5:  include src in mcast
constexpr uint64_t CMD_BUF_MISC_SCATTER_LIST_EN = (1 << 6);     // bit 6:  scatter list enable
constexpr uint64_t CMD_BUF_MISC_SCATTER_TO_DEST = (1 << 7);     // bit 7:  scatter to dest addr
constexpr uint64_t CMD_BUF_MISC_WRAPPING_EN = (1 << 8);         // bit 8:  address wrapping
constexpr uint64_t CMD_BUF_MISC_WRITE_TRANS = (1 << 9);         // bit 9:  write transaction
constexpr uint64_t CMD_BUF_MISC_ATOMIC_TRANS = (1 << 10);       // bit 10: atomic transaction
constexpr uint64_t CMD_BUF_MISC_BYTE_ENABLE = (1 << 11);        // bit 11: byte enable trans
constexpr uint64_t CMD_BUF_MISC_DIS_LINKED_PER_TR = (1 << 12);  // bit 12: disable linked per trans
constexpr uint64_t CMD_BUF_MISC_SCATTER_HAS_SIZE = (1 << 13);   // bit 13: scatter list has size
constexpr uint64_t CMD_BUF_MISC_SCATTER_HAS_XY = (1 << 14);     // bit 14: scatter list has xy
constexpr uint64_t CMD_BUF_MISC_L1_ACCUM_EN = (1 << 15);        // bit 15: L1 accumulation enable
constexpr uint64_t CMD_BUF_MISC_IDMA_EN = (1 << 16);            // bit 16: IDMA enable
constexpr uint64_t CMD_BUF_MISC_FORCE_DIM_ROUTING = (1 << 17);  // bit 17: force dimension routing
constexpr uint64_t CMD_BUF_MISC_PATH_RES_DISABLE = (1 << 18);   // bit 18: path reservation disable

// ============================================================================
// Pre-defined MISC register values for common transaction types
// ============================================================================
// Read transaction: write_trans=0, atomic_trans=0, posted=0 → all bits clear
constexpr uint64_t CMD_BUF_MISC_READ = TT_ROCC_CMD_BUF_MISC_REG_DEFAULT;

// Unicast write: write_trans=1
constexpr uint64_t CMD_BUF_MISC_WRITE = CMD_BUF_MISC_WRITE_TRANS;

// Unicast write (posted, no ack): write_trans=1 + posted=1
constexpr uint64_t CMD_BUF_MISC_WRITE_POSTED = CMD_BUF_MISC_WRITE_TRANS | CMD_BUF_MISC_POSTED;

// Multicast write: write_trans=1 + multicast=1 + linked=1 + src_include=1
// NOTE: linked=1 is required for multicast (per cmdbuff_api.hpp pattern)
constexpr uint64_t CMD_BUF_MISC_MCAST_WRITE =
    CMD_BUF_MISC_WRITE_TRANS | CMD_BUF_MISC_MULTICAST | CMD_BUF_MISC_LINKED | CMD_BUF_MISC_SRC_INCLUDE;

// Multicast write (posted): write_trans=1 + multicast=1 + linked=1 + src_include=1 + posted=1
constexpr uint64_t CMD_BUF_MISC_MCAST_WRITE_POSTED = CMD_BUF_MISC_MCAST_WRITE | CMD_BUF_MISC_POSTED;

// Multicast write (no src include): write_trans=1 + multicast=1 + linked=1
constexpr uint64_t CMD_BUF_MISC_MCAST_WRITE_NO_SRC =
    CMD_BUF_MISC_WRITE_TRANS | CMD_BUF_MISC_MULTICAST | CMD_BUF_MISC_LINKED;

// Atomic transaction: atomic_trans=1 + posted=1 + src_include=1
constexpr uint64_t CMD_BUF_MISC_ATOMIC = CMD_BUF_MISC_ATOMIC_TRANS | CMD_BUF_MISC_POSTED | CMD_BUF_MISC_SRC_INCLUDE;

// Inline write: write_trans=1 + inline_wr=1
constexpr uint64_t CMD_BUF_MISC_INLINE_WRITE = CMD_BUF_MISC_WRITE_TRANS | CMD_BUF_MISC_INLINE_WR;

// ============================================================================

extern uint32_t noc_reads_num_issued[NUM_NOCS];
extern uint32_t noc_nonposted_writes_num_issued[NUM_NOCS];
extern uint32_t noc_nonposted_writes_acked[NUM_NOCS];
extern uint32_t noc_nonposted_atomics_acked[NUM_NOCS];
extern uint32_t noc_posted_writes_num_issued[NUM_NOCS];

inline __attribute__((always_inline)) void NOC_CMD_BUF_WRITE_REG(
    uint32_t noc, uint32_t buf, uint32_t addr, uint32_t val) {
#if defined(WATCHER_ENABLE_NOC_SANITIZE_LINKED_TRANSACTION)
    if (addr == NOC_CTRL_LO) {
        auto* watcher_msg = GET_MAILBOX_ADDRESS_DEV(watcher);
        watcher_msg->noc_linked_status[noc] = (val & NOC_CMD_VC_LINKED) != 0;
    }
#endif
    uintptr_t offset = (buf << NOC_CMD_BUF_OFFSET_BIT) + (noc << NOC_INSTANCE_OFFSET_BIT) + addr;
    volatile uint32_t* ptr = (volatile uint32_t*)offset;
    *ptr = val;
}

inline __attribute__((always_inline)) uint64_t NOC_CMD_BUF_READ_OVERLAY_REG(uint32_t buf, uint32_t reg_offset) {
    // The AT buffer is backed by Quasar's simple command buffer; WR/RD buffers are regular indexed command buffers.
    if (buf == OVERLAY_AT_CMD_BUF) {
        return __builtin_riscv_ttrocc_scmdbuf_rd_reg(reg_offset / 8);
    }
    return __builtin_riscv_ttrocc_cmdbuf_rd_reg(buf, reg_offset / 8);
}

inline __attribute__((always_inline)) uint32_t NOC_CMD_BUF_READ_REG(uint32_t noc, uint32_t buf, uint32_t addr) {
    switch (addr) {
        // NOC_TARG_ADDR_* -> cmd-buf SRC_{ADDR,COORD} (data source: remote for reads, local for writes)
        case NOC_TARG_ADDR_LO:
            return (uint32_t)NOC_CMD_BUF_READ_OVERLAY_REG(
                buf, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_SRC_ADDR_REG_OFFSET);

        case NOC_TARG_ADDR_MID:
            return (
                uint32_t)(NOC_CMD_BUF_READ_OVERLAY_REG(buf, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_SRC_ADDR_REG_OFFSET) >>
                          32);

        case NOC_TARG_ADDR_HI:  // NOC_TARG_ADDR_COORDINATE aliases this
            return (uint32_t)NOC_CMD_BUF_READ_OVERLAY_REG(
                buf, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_SRC_COORD_REG_OFFSET);

        // NOC_RET_ADDR_* -> cmd-buf DEST_{ADDR,COORD} (data dest: local for reads, remote for writes)
        case NOC_RET_ADDR_LO:
            return (uint32_t)NOC_CMD_BUF_READ_OVERLAY_REG(
                buf, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_DEST_ADDR_REG_OFFSET);

        case NOC_RET_ADDR_MID:
            return (uint32_t)(NOC_CMD_BUF_READ_OVERLAY_REG(
                                  buf, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_DEST_ADDR_REG_OFFSET) >>
                              32);

        // NOC_RET_ADDR_COORDINATE aliases this
        case NOC_RET_ADDR_HI:
            return (uint32_t)NOC_CMD_BUF_READ_OVERLAY_REG(
                buf, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_DEST_COORD_REG_OFFSET);

        // NOC_AT_LEN_BE aliases NOC_AT_LEN on Quasar
        case NOC_AT_LEN:
            return (uint32_t)NOC_CMD_BUF_READ_OVERLAY_REG(
                buf, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_LEN_BYTES_REG_OFFSET);

        // Inline write / atomic data value
        case NOC_AT_DATA:
            return (uint32_t)NOC_CMD_BUF_READ_OVERLAY_REG(
                buf, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_INLINE_DATA_REG_OFFSET);

        // NOC_CMD_CTRL is the transaction trigger on BH/WH (written as 0x1 to fire);
        // on Quasar the trigger is cmdbuf_issue_trans() with no stored state.
        case NOC_CMD_CTRL: return 0;

        // NOC_NODE_ID and other NOC config registers are true MMIO globals - not cmd buf registers (buf-independent).
        default: {
            ASSERT(buf == 0);  // cmd-buf regs route through RoCC above; this path is globals-only
            uintptr_t offset = (noc << NOC_INSTANCE_OFFSET_BIT) + addr;
            return *((volatile uint32_t*)offset);
        }
    }
}

inline __attribute__((always_inline)) uint32_t NOC_STATUS_REG_ADDR(uint32_t noc, uint32_t reg_id) {
    return (noc << NOC_INSTANCE_OFFSET_BIT) + NOC_STATUS(reg_id);
}

inline __attribute__((always_inline)) uint32_t NOC_STATUS_READ_REG(uint32_t noc, uint32_t reg_id) {
    uintptr_t offset = NOC_STATUS_REG_ADDR(noc, reg_id);
    volatile uint32_t* ptr = (volatile uint32_t*)offset;
    return *ptr;
}

inline __attribute__((always_inline)) uint32_t NOC_CFG_READ_REG(uint32_t noc, uint32_t reg_id) {
    uintptr_t offset = (noc << NOC_INSTANCE_OFFSET_BIT) + NOC_CFG(reg_id);
    volatile uint32_t* ptr = (volatile uint32_t*)offset;
    return *ptr;
}

inline __attribute__((always_inline)) bool noc_cmd_buf_ready(uint32_t noc, uint32_t cmd_buf) {
    /* Overlay cmd buffers will stall cpu if not ready */
    return true;
}

inline __attribute__((always_inline)) void noc_clear_outstanding_req_cnt(uint32_t noc, uint32_t id_mask) {
    uintptr_t offset = (noc << NOC_INSTANCE_OFFSET_BIT) + NOC_CLEAR_OUTSTANDING_REQ_CNT;
    volatile uint32_t* ptr = (volatile uint32_t*)offset;
    *ptr = id_mask;
}

struct NocCmdBufState {
    uint32_t ctrl;
    uint32_t ret_addr_coord;
    uint32_t targ_addr_lo;
    uint32_t ret_addr_lo;
    uint32_t at_len_be;
    uint32_t targ_addr_coord;
    uint32_t targ_addr_mid;
    uint32_t packet_tag;
    uint32_t at_data;
    uint32_t ret_addr_mid;
};

inline __attribute__((always_inline)) void noc_cmd_buf_save_state(
    uint32_t noc, uint32_t cmd_buf, NocCmdBufState* state) {
    state->ctrl = NOC_CMD_BUF_READ_REG(noc, cmd_buf, NOC_CMD_CTRL);
    constexpr uint32_t noc_ctrl_reserved_bit_mask = ((1u << 27) - (1u << 18)) | (1u << 31);
    state->ctrl &= ~noc_ctrl_reserved_bit_mask;
    state->ret_addr_coord = NOC_CMD_BUF_READ_REG(noc, cmd_buf, NOC_RET_ADDR_COORDINATE);
    state->targ_addr_lo = NOC_CMD_BUF_READ_REG(noc, cmd_buf, NOC_TARG_ADDR_LO);
    state->ret_addr_lo = NOC_CMD_BUF_READ_REG(noc, cmd_buf, NOC_RET_ADDR_LO);
    state->at_len_be = NOC_CMD_BUF_READ_REG(noc, cmd_buf, NOC_AT_LEN);
    state->targ_addr_coord = NOC_CMD_BUF_READ_REG(noc, cmd_buf, NOC_TARG_ADDR_COORDINATE);
    state->targ_addr_mid = NOC_CMD_BUF_READ_REG(noc, cmd_buf, NOC_TARG_ADDR_MID);
    state->packet_tag = 0;
    state->at_data = NOC_CMD_BUF_READ_REG(noc, cmd_buf, NOC_AT_DATA);
    state->ret_addr_mid = NOC_CMD_BUF_READ_REG(noc, cmd_buf, NOC_RET_ADDR_MID);
}

inline __attribute__((always_inline)) void noc_clear_packet_tag(uint32_t /* noc */, uint32_t /* cmd_buf */) {}

inline __attribute__((always_inline)) void noc_clear_packet_tags(uint32_t /* noc */) {}

inline __attribute__((always_inline)) void noc_cmd_buf_restore_state(
    uint32_t noc, uint32_t cmd_buf, const NocCmdBufState* state) {
    while (!noc_cmd_buf_ready(noc, cmd_buf));
    NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_CMD_CTRL, state->ctrl);
    NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_RET_ADDR_COORDINATE, state->ret_addr_coord);
    NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_TARG_ADDR_LO, state->targ_addr_lo);
    NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_RET_ADDR_LO, state->ret_addr_lo);
    NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_AT_LEN, state->at_len_be);
    NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_TARG_ADDR_COORDINATE, state->targ_addr_coord);
    NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_TARG_ADDR_MID, state->targ_addr_mid);
    NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_AT_DATA, state->at_data);
    NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_RET_ADDR_MID, state->ret_addr_mid);
}

inline __attribute__((always_inline)) uint32_t noc_get_reads_issued(uint32_t noc) { return noc_reads_num_issued[noc]; }

inline __attribute__((always_inline)) uint32_t noc_get_nonposted_writes_issued(uint32_t noc) {
    return noc_nonposted_writes_num_issued[noc];
}

inline __attribute__((always_inline)) uint32_t noc_get_nonposted_writes_acked(uint32_t noc) {
    return noc_nonposted_writes_acked[noc];
}

inline __attribute__((always_inline)) uint32_t noc_get_nonposted_atomics_acked(uint32_t noc) {
    return noc_nonposted_atomics_acked[noc];
}

inline __attribute__((always_inline)) uint32_t noc_get_posted_writes_issued(uint32_t noc) {
    return noc_posted_writes_num_issued[noc];
}

inline __attribute__((always_inline)) void noc_increment_nonposted_writes_acked(uint32_t noc, uint32_t delta) {
    noc_nonposted_writes_acked[noc] += delta;
}

inline __attribute__((always_inline)) void noc_increment_nonposted_writes_issued(uint32_t noc, uint32_t delta) {
    noc_nonposted_writes_num_issued[noc] += delta;
}

inline __attribute__((always_inline)) bool noc_nonposted_writes_sent_at_count(uint32_t noc, uint32_t expected_count) {
    uint32_t sent = NOC_STATUS_READ_REG(noc, NIU_MST_NONPOSTED_WR_REQ_SENT);
    return (int32_t)(sent - expected_count) >= 0;
}

inline __attribute__((always_inline)) void noc_cmd_buf_set_targ_addr_coordinate(
    uint32_t noc, uint32_t cmd_buf, uint32_t coord) {
    NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_TARG_ADDR_COORDINATE, coord);
}

inline __attribute__((always_inline)) void noc_cmd_buf_set_targ_addr(
    uint32_t noc, uint32_t cmd_buf, uint64_t targ_addr) {
    NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_TARG_ADDR_LO, (uint32_t)(targ_addr & 0xFFFFFFFF));
    NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_TARG_ADDR_MID, (uint32_t)(targ_addr >> 32) & NOC_PCIE_MASK);
    NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_TARG_ADDR_COORDINATE, (uint32_t)(targ_addr >> NOC_ADDR_COORD_SHIFT));
}

inline __attribute__((always_inline)) void noc_cmd_buf_set_ret_addr_coordinate(
    uint32_t noc, uint32_t cmd_buf, uint32_t coord) {
    NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_RET_ADDR_COORDINATE, coord);
}

inline __attribute__((always_inline)) void noc_cmd_buf_set_ret_addr(uint32_t noc, uint32_t cmd_buf, uint64_t ret_addr) {
    NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_RET_ADDR_LO, (uint32_t)(ret_addr & 0xFFFFFFFF));
    NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_RET_ADDR_MID, (uint32_t)(ret_addr >> 32) & NOC_PCIE_MASK);
    NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_RET_ADDR_COORDINATE, (uint32_t)(ret_addr >> NOC_ADDR_COORD_SHIFT));
}

inline __attribute__((always_inline)) uint32_t noc_debug_read_at_len_be(uint32_t noc, uint32_t cmd_buf) {
    return NOC_CMD_BUF_READ_REG(noc, cmd_buf, NOC_AT_LEN);
}

inline __attribute__((always_inline)) uint64_t noc_local_xy() {
    constexpr uint32_t noc = 0;
    uint32_t noc_id_reg = NOC_CMD_BUF_READ_REG(noc, 0, NOC_NODE_ID);
    uint32_t my_x = noc_id_reg & NOC_NODE_ID_MASK;
    uint32_t my_y = (noc_id_reg >> NOC_ADDR_NODE_ID_BITS) & NOC_NODE_ID_MASK;
    return NOC_XY_COORD(my_x, my_y);
}

// snoop asks the destination NIU to raise a cache snoop on receive. flush, per quasar noc spec,
// prevents the next packet from committing until all previous packets have committed -- so tagging
// just the final transfer guarantees the whole payload is committed before any later same-VC packet
// to that destination.
//
// Persistent command-buffer state, not a per-transaction argument: set the tags, issue the transfer
// that should carry them, then call again with both false. Left set, they tag every subsequent packet
// on this buffer.
//
// Writes the whole register, so a call overwrites the other caller's bit as well as its own.
template <uint32_t cmd_buf>
inline __attribute__((always_inline)) void noc_set_packet_tags(bool snoop, bool flush) {
    static_assert(
        cmd_buf == OVERLAY_WR_CMD_BUF || cmd_buf == OVERLAY_RD_CMD_BUF,
        "packet tags are only defined for the full command buffers (0/1)");
    TT_ROCC_CMD_BUF_PACKET_TAGS_reg_u tags;
    tags.val = 0;
    tags.f.snoop_bit = snoop;
    tags.f.flush_bit = flush;
    __builtin_riscv_ttrocc_cmdbuf_wr_reg(
        cmd_buf, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_PACKET_TAGS_REG_OFFSET / 8, tags.val);
}

inline __attribute__((always_inline)) void init_wr_cmd_buf(uint64_t my_xy) {
    __builtin_riscv_ttrocc_cmdbuf_reset(OVERLAY_WR_CMD_BUF);
    __builtin_riscv_ttrocc_cmdbuf_wr_reg(
        OVERLAY_WR_CMD_BUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_MISC_REG_OFFSET / 8, CMD_BUF_MISC_WRITE_POSTED);
    __builtin_riscv_ttrocc_cmdbuf_wr_reg(
        OVERLAY_WR_CMD_BUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_SRC_COORD_REG_OFFSET / 8, my_xy);
    __builtin_riscv_ttrocc_cmdbuf_wr_reg(
        OVERLAY_WR_CMD_BUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_REQ_VC_REG_OFFSET / 8, NOC_V2_WR_REQ_VC);
    __builtin_riscv_ttrocc_cmdbuf_wr_reg(
        OVERLAY_WR_CMD_BUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_RESP_VC_REG_OFFSET / 8, NOC_V2_WR_RESP_VC);
    __builtin_riscv_ttrocc_cmdbuf_wr_reg(
        OVERLAY_WR_CMD_BUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_WR_SENT_TR_ID_REG_OFFSET / 8, NOC_V2_TRID_STATIC);
    __builtin_riscv_ttrocc_cmdbuf_wr_reg(
        OVERLAY_WR_CMD_BUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_TR_ACK_TR_ID_REG_OFFSET / 8, NOC_V2_TRID_STATIC);
    __builtin_riscv_ttrocc_cmdbuf_wr_reg(
        OVERLAY_WR_CMD_BUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_TR_ID_REG_OFFSET / 8, NOC_V2_TRID_STATIC);
    __builtin_riscv_ttrocc_cmdbuf_wr_reg(
        OVERLAY_WR_CMD_BUF,
        TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_MAX_BYTES_IN_PACKET_REG_OFFSET / 8,
        NOC_V2_MAX_BYTES_IN_PACKET);
}

inline __attribute__((always_inline)) void init_rd_cmd_buf(uint64_t my_xy) {
    __builtin_riscv_ttrocc_cmdbuf_reset(OVERLAY_RD_CMD_BUF);
    __builtin_riscv_ttrocc_cmdbuf_wr_reg(
        OVERLAY_RD_CMD_BUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_MISC_REG_OFFSET / 8, CMD_BUF_MISC_READ);
    __builtin_riscv_ttrocc_cmdbuf_wr_reg(
        OVERLAY_RD_CMD_BUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_DEST_COORD_REG_OFFSET / 8, my_xy);
    __builtin_riscv_ttrocc_cmdbuf_wr_reg(
        OVERLAY_RD_CMD_BUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_REQ_VC_REG_OFFSET / 8, NOC_V2_RD_REQ_VC);
    __builtin_riscv_ttrocc_cmdbuf_wr_reg(
        OVERLAY_RD_CMD_BUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_RESP_VC_REG_OFFSET / 8, NOC_V2_RD_RESP_VC);
    __builtin_riscv_ttrocc_cmdbuf_wr_reg(
        OVERLAY_RD_CMD_BUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_WR_SENT_TR_ID_REG_OFFSET / 8, NOC_V2_TRID_STATIC);
    __builtin_riscv_ttrocc_cmdbuf_wr_reg(
        OVERLAY_RD_CMD_BUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_TR_ACK_TR_ID_REG_OFFSET / 8, NOC_V2_TRID_STATIC);
    __builtin_riscv_ttrocc_cmdbuf_wr_reg(
        OVERLAY_RD_CMD_BUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_TR_ID_REG_OFFSET / 8, NOC_V2_TRID_STATIC);
    __builtin_riscv_ttrocc_cmdbuf_wr_reg(
        OVERLAY_RD_CMD_BUF,
        TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_MAX_BYTES_IN_PACKET_REG_OFFSET / 8,
        NOC_V2_MAX_BYTES_IN_PACKET);
}

inline __attribute__((always_inline)) void init_at_cmd_buf(uint64_t my_xy, uint32_t atomic_ret_val) {
    __builtin_riscv_ttrocc_scmdbuf_reset();
    __builtin_riscv_ttrocc_scmdbuf_wr_reg(
        TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_MISC_REG_OFFSET / 8, CMD_BUF_MISC_ATOMIC);
    __builtin_riscv_ttrocc_scmdbuf_wr_reg(TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_SRC_ADDR_REG_OFFSET / 8, atomic_ret_val);
    __builtin_riscv_ttrocc_scmdbuf_wr_reg(TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_SRC_COORD_REG_OFFSET / 8, my_xy);
    __builtin_riscv_ttrocc_scmdbuf_wr_reg(
        TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_RESP_VC_REG_OFFSET / 8, NOC_V2_WR_RESP_VC);
    __builtin_riscv_ttrocc_scmdbuf_wr_reg(
        TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_MAX_BYTES_IN_PACKET_REG_OFFSET / 8, NOC_V2_MAX_BYTES_IN_PACKET);
}

inline __attribute__((always_inline)) void overlay_cmd_buff_init(uint32_t atomic_ret_val) {
    uint64_t my_xy = noc_local_xy();
    init_wr_cmd_buf(my_xy);  // Write command buffer (CMDBUF_0): local src -> remote dest
    init_rd_cmd_buf(my_xy);  // Read command buffer (CMDBUF_1): remote src -> local dest
    init_at_cmd_buf(
        my_xy, atomic_ret_val);  // Atomic command buffer (SCMDBUF): simple buffer for atomics and inline writes
}

// set noc local memory state for a single kernel from the global state
inline __attribute__((always_inline)) void noc_local_state_init(int noc) {
    // Hide latency of NOC reg reads by reading first, writing second
    uint32_t reads_num_issued = NOC_STATUS_READ_REG(noc, NIU_MST_RD_RESP_RECEIVED);
    uint32_t nonposted_writes_num_issued = NOC_STATUS_READ_REG(noc, NIU_MST_NONPOSTED_WR_REQ_SENT);
    uint32_t nonposted_writes_acked = NOC_STATUS_READ_REG(noc, NIU_MST_WR_ACK_RECEIVED);
    uint32_t nonposted_atomics_acked = NOC_STATUS_READ_REG(noc, NIU_MST_ATOMIC_RESP_RECEIVED);
    uint32_t posted_writes_num_issued = NOC_STATUS_READ_REG(noc, NIU_MST_POSTED_WR_REQ_SENT);

    noc_reads_num_issued[noc] = reads_num_issued;
    noc_nonposted_writes_num_issued[noc] = nonposted_writes_num_issued;
    noc_nonposted_writes_acked[noc] = nonposted_writes_acked;
    noc_nonposted_atomics_acked[noc] = nonposted_atomics_acked;
    noc_posted_writes_num_issued[noc] = posted_writes_num_issued;
}

inline __attribute__((always_inline)) void ncrisc_noc_counters_init() {
    constexpr uint32_t noc = 0;
    // Hide latency of NOC reg reads by reading first, writing second
    uint32_t reads_num_issued = NOC_STATUS_READ_REG(noc, NIU_MST_RD_RESP_RECEIVED);
    uint32_t nonposted_writes_num_issued = NOC_STATUS_READ_REG(noc, NIU_MST_NONPOSTED_WR_REQ_SENT);
    uint32_t nonposted_writes_acked = NOC_STATUS_READ_REG(noc, NIU_MST_WR_ACK_RECEIVED);
    uint32_t nonposted_atomics_acked = NOC_STATUS_READ_REG(noc, NIU_MST_ATOMIC_RESP_RECEIVED);
    uint32_t posted_writes_num_issued = NOC_STATUS_READ_REG(noc, NIU_MST_POSTED_WR_REQ_SENT);

    noc_reads_num_issued[noc] = reads_num_issued;
    noc_nonposted_writes_num_issued[noc] = nonposted_writes_num_issued;
    noc_nonposted_writes_acked[noc] = nonposted_writes_acked;
    noc_nonposted_atomics_acked[noc] = nonposted_atomics_acked;
    noc_posted_writes_num_issued[noc] = posted_writes_num_issued;
}

inline __attribute__((always_inline)) bool ncrisc_noc_reads_flushed(uint32_t noc) {
    return __builtin_riscv_ttrocc_scmdbuf_tr_ack() == 0;
}

inline __attribute__((always_inline)) bool ncrisc_noc_read_with_transaction_id_flushed(
    uint32_t noc, uint32_t transcation_id) {
    return __builtin_riscv_ttrocc_scmdbuf_tr_ack_trid(transcation_id) == 0;
}

inline __attribute__((always_inline)) uint32_t noc_available_transactions(uint32_t noc, uint32_t trid) {
    return NOC_MAX_TRANSACTION_ID_COUNT - __builtin_riscv_ttrocc_scmdbuf_tr_ack_trid(trid);
}

inline __attribute__((always_inline)) bool ncrisc_noc_nonposted_writes_sent(uint32_t noc) {
    return __builtin_riscv_ttrocc_scmdbuf_wr_sent() == 0;
}

inline __attribute__((always_inline)) bool ncrisc_noc_posted_writes_sent(uint32_t noc) {
    return __builtin_riscv_ttrocc_scmdbuf_wr_sent() == 0;
}

inline __attribute__((always_inline)) bool ncrisc_noc_nonposted_writes_flushed(uint32_t noc) {
    return __builtin_riscv_ttrocc_scmdbuf_tr_ack() == 0;
}

inline __attribute__((always_inline)) bool ncrisc_noc_nonposted_write_with_transaction_id_sent(
    uint32_t noc, uint32_t transcation_id) {
    return __builtin_riscv_ttrocc_scmdbuf_wr_sent_trid(transcation_id) == 0;
}

inline __attribute__((always_inline)) bool ncrisc_noc_nonposted_write_with_transaction_id_flushed(
    uint32_t noc, uint32_t transcation_id) {
    return __builtin_riscv_ttrocc_scmdbuf_tr_ack_trid(transcation_id) == 0;
}

inline __attribute__((always_inline)) bool ncrisc_noc_nonposted_atomics_flushed(uint32_t noc) {
    return __builtin_riscv_ttrocc_scmdbuf_tr_ack() == 0;
}

// clang-format off
/**
 * Sets the transaction id for a noc transaction.
 *
 * Return value: None
 *
 * | Argument | Description                                        | Data type | Valid range | Required |
 * |----------|----------------------------------------------------|-----------|-------------|----------|
 * | noc      | Which NOC to use for the transaction               | uint32_t  | 0 or 1      | True     |
 * | cmd_buf  | Which command buffer to use for the transaction    | uint32_t  | 0 - 3       | True     |
 * | trid     | Transaction id for the transaction                 | uint32_t  | 0x0 - 0xF   | True     |
 */
// clang-format on
inline __attribute__((always_inline)) void ncrisc_noc_set_transaction_id(
    uint32_t noc, uint32_t cmd_buf, uint32_t trid) {
    __builtin_riscv_ttrocc_cmdbuf_wr_reg(cmd_buf, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_TR_ID_REG_OFFSET / 8, trid);
}

// clang-format off
/**
 * Issues a transaction on the given command buffer.
 * Dispatches to the correct hardware builtin based on the buffer index:
 *   - cmd_buf 0 (OVERLAY_WR_CMD_BUF) and 1 (OVERLAY_RD_CMD_BUF): regular command buffer
 *   - cmd_buf 2 (OVERLAY_AT_CMD_BUF): simple command buffer
 *
 * Return value: None
 *
 * | Argument                     | Description                                  | Data type | Valid range | Required |
 * |------------------------------|----------------------------------------------|-----------|-------------|----------|
 * | cmd_buf (template parameter) | Which command buffer to issue transaction on | uint32_t  | 0, 1, or 2  | True     |
 */
// clang-format on
template <uint8_t cmd_buf>
inline __attribute__((always_inline)) void noc_issue_transaction() {
    static_assert(cmd_buf <= 1, "cmd_buf must be 0 (WR), 1 (RD), or 2 (AT/simple)");
    if constexpr (cmd_buf == 2) {
        __builtin_riscv_ttrocc_scmdbuf_issue_trans();
    } else {
        __builtin_riscv_ttrocc_cmdbuf_issue_trans(cmd_buf);
    }
}

inline __attribute__((always_inline)) void ncrisc_noc_full_sync() {
    while (!ncrisc_noc_reads_flushed(0));
    while (!ncrisc_noc_nonposted_writes_sent(0));
    while (!ncrisc_noc_nonposted_writes_flushed(0));
    while (!ncrisc_noc_nonposted_atomics_flushed(0));
    while (!ncrisc_noc_posted_writes_sent(0));
}

template <bool write, bool posted>
inline __attribute__((always_inline)) uint32_t get_noc_counter_for_debug(uint32_t noc) {
    if constexpr (write) {
        if constexpr (posted) {
            return NOC_STATUS_READ_REG(noc, NIU_MST_POSTED_WR_REQ_SENT);
        } else {
            return NOC_STATUS_READ_REG(noc, NIU_MST_NONPOSTED_WR_REQ_SENT);
        }
    } else {
        // Read
        static_assert(posted == false, "There is no such thing as posted reads");
        return NOC_STATUS_READ_REG(noc, NIU_MST_RD_RESP_RECEIVED);
    }
}

// ============================================================================================================
// DEPRECATED DYNAMIC NOC TYPES AND FUNCTIONS - NOT SUPPORTED ON QUASAR
// ============================================================================================================
// The following types and functions are kept for API backward compatibility only.
// Quasar has only 1 NOC, so dynamic NOC functionality is not supported.
// Any attempt to use these functions will result in a compile-time error.
// ============================================================================================================

// Dynamic NOC barrier types and structures (unused on Quasar)
enum class NocBarrierType : uint8_t {
    READS_NUM_ISSUED,
    NONPOSTED_WRITES_NUM_ISSUED,
    NONPOSTED_WRITES_ACKED,
    NONPOSTED_ATOMICS_ACKED,
    POSTED_WRITES_NUM_ISSUED,
    COUNT
};

static constexpr uint8_t NUM_BARRIER_TYPES = static_cast<uint32_t>(NocBarrierType::COUNT);

struct BarrierCounter {
    uint32_t barrier[NUM_BARRIER_TYPES];
};

struct RiscBarrierCounter {
    BarrierCounter risc[MaxDMProcessorsPerCoreType];
};

struct NocBarrierCounter {
    RiscBarrierCounter noc[NUM_NOCS];
};

// Dynamic NOC counter helper functions (unused on Quasar)
template <uint8_t proc_t, NocBarrierType barrier_type>
inline __attribute__((always_inline)) uint32_t get_noc_counter_address(uint32_t noc) {
    static_assert(proc_t < MaxDMProcessorsPerCoreType);
    static_assert(static_cast<std::underlying_type_t<NocBarrierType>>(barrier_type) < NUM_BARRIER_TYPES);

#if defined(COMPILE_FOR_DISPATCH_ENGINE)
    constexpr uint32_t base = MEM_DISPATCH_NOC_COUNTER_BASE;
#else
    constexpr uint32_t base = MEM_NOC_COUNTER_BASE;
#endif
    constexpr uint32_t size = MEM_NOC_COUNTER_SIZE;

    // Calculate most of the offset at compile time. Only the noc is variable at runtime.
    constexpr uint32_t compile_time_offset =
        offsetof(NocBarrierCounter, noc) + proc_t * sizeof(decltype(std::declval<NocBarrierCounter>().noc[0].risc[0])) +
        static_cast<std::underlying_type_t<NocBarrierType>>(barrier_type) *
            sizeof(decltype(std::declval<NocBarrierCounter>().noc[0].risc[0].barrier[0]));

    constexpr uint32_t noc_stride = sizeof(decltype(std::declval<NocBarrierCounter>().noc[0]));

    return base + noc * noc_stride + compile_time_offset;
}

template <uint8_t proc_t, NocBarrierType barrier_type>
inline __attribute__((always_inline)) uint32_t get_noc_counter_val(uint32_t noc) {
    uint32_t counter_addr = get_noc_counter_address<proc_t, barrier_type>(noc);
    return *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(counter_addr);
}

template <uint8_t proc_t, NocBarrierType barrier_type>
inline __attribute__((always_inline)) void inc_noc_counter_val(uint32_t noc, uint32_t inc = 1) {
    uint32_t counter_addr = get_noc_counter_address<proc_t, barrier_type>(noc);
    *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(counter_addr) += inc;
}

template <uint8_t proc_t, NocBarrierType barrier_type>
inline __attribute__((always_inline)) void set_noc_counter_val(uint32_t noc, uint32_t val) {
    uint32_t counter_addr = get_noc_counter_address<proc_t, barrier_type>(noc);
    *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(counter_addr) = val;
}

// Dynamic NOC functions (will fail at compile-time if used)
template <typename T = void>
inline __attribute__((always_inline)) bool ncrisc_dynamic_noc_reads_flushed(uint32_t noc) {
    static_assert(sizeof(T*) == 0, "Quasar does not support DYNAMIC_NOC as it has only 1 NOC");
    return false;
}

template <typename T = void>
inline __attribute__((always_inline)) bool ncrisc_dynamic_noc_nonposted_writes_sent(uint32_t noc) {
    static_assert(sizeof(T*) == 0, "Quasar does not support DYNAMIC_NOC as it has only 1 NOC");
    return false;
}

template <typename T = void>
inline __attribute__((always_inline)) bool ncrisc_dynamic_noc_posted_writes_sent(uint32_t noc) {
    static_assert(sizeof(T*) == 0, "Quasar does not support DYNAMIC_NOC as it has only 1 NOC");
    return false;
}

template <typename T = void>
inline __attribute__((always_inline)) bool ncrisc_dynamic_noc_nonposted_writes_flushed(uint32_t noc) {
    static_assert(sizeof(T*) == 0, "Quasar does not support DYNAMIC_NOC as it has only 1 NOC");
    return false;
}

template <typename T = void>
inline __attribute__((always_inline)) bool ncrisc_dynamic_noc_nonposted_atomics_flushed(uint32_t noc) {
    static_assert(sizeof(T*) == 0, "Quasar does not support DYNAMIC_NOC as it has only 1 NOC");
    return false;
}

template <typename T = void>
inline __attribute__((always_inline)) void dynamic_noc_init() {
    static_assert(sizeof(T*) == 0, "Quasar does not support DYNAMIC_NOC as it has only 1 NOC");
}

template <NocBarrierType barrier_type, uint32_t status_register, typename T = void>
inline __attribute__((always_inline)) void dynamic_noc_local_barrier_init(
    uint32_t noc0_status_reg, uint32_t noc1_status_reg) {
    static_assert(sizeof(T*) == 0, "Quasar does not support DYNAMIC_NOC as it has only 1 NOC");
}

template <typename T = void>
inline __attribute__((always_inline)) void dynamic_noc_local_state_init() {
    static_assert(sizeof(T*) == 0, "Quasar does not support DYNAMIC_NOC as it has only 1 NOC");
}

template <uint8_t MAX_NOCS_TO_INIT = NUM_NOCS, typename T = void>
inline __attribute__((always_inline)) void ncrisc_dynamic_noc_full_sync() {
    static_assert(sizeof(T*) == 0, "Quasar does not support DYNAMIC_NOC as it has only 1 NOC");
}

// ============================================================================================================
// END OF DEPRECATED DYNAMIC NOC FUNCTIONS
// ============================================================================================================
