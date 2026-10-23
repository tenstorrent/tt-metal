// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
// Version: FFN1.3.0
/**
 * @file cmdbuff_api.hpp
 * @brief Command Buffer API for Overlay Data Movement Operations
 *
 * This API provides control over the overlay's two normal command buffers used for
 * complex data movement operations. Each overlay contains 2 normal command buffers
 * plus 1 simple command buffer (simple buffer API is separate).
 *
 * ## Implementation Details
 *
 * Command buffer operations use custom RISC-V ROCC instructions with unique opcodes
 * for each buffer. Since opcodes must be known at compile time, the command buffer id
 * is a template parameter:
 * - `xxx_cmdbuf<CMDBUF_0>()` - Functions for command buffer 0
 * - `xxx_cmdbuf<CMDBUF_1>()` - Functions for command buffer 1
 * `xxx_cmdbuf_0()` / `xxx_cmdbuf_1()` aliases are provided at the end of the file.
 *
 * ## Transaction ID Management
 *
 * Each command buffer maintains separate transaction ID ranges to avoid conflicts:
 * - **Buffer 0**: Static TID 1, wrapping range 2-6
 * - **Buffer 1**: Static TID 7, wrapping range 8-12
 *
 * @note Requires ROCC instruction definitions from rocc_instructions.hpp
 */
#pragma once

#include "rocc_instructions.hpp"

namespace overlay {

/* Command buffer id, the template parameter of every *_cmdbuf<CMDBUF>() function. An enum rather than
 * an integer so that an out-of-range id is a compile error: the __builtin_riscv_ttrocc_cmdbuf_* builtins
 * accept any constant and would silently encode it as a different instruction. */
enum CmdBuf : uint32_t { CMDBUF_0 = 0, CMDBUF_1 = 1 };

/* Default transaction ID for both command buffers */
constexpr uint32_t CMDBUF_DEF_TRID = 0;
/* Static (starting) transaction ID for command buffer 0 */
constexpr uint32_t CMDBUF_0_TRID_STATIC = 1;
/* Start transaction ID for command buffer 0 when using TID wrapping */
constexpr uint32_t CMDBUF_0_TRID_START = 2;
/* End transaction ID for command buffer 0 when using TID wrapping */
constexpr uint32_t CMDBUF_0_TRID_END = 6;
/* Static (starting) transaction ID for command buffer 1 */
constexpr uint32_t CMDBUF_1_TRID_STATIC = 7;
/* Start transaction ID for command buffer 1 when using TID wrapping */
constexpr uint32_t CMDBUF_1_TRID_START = 8;
/* End transaction ID for command buffer 1 when using TID wrapping */
constexpr uint32_t CMDBUF_1_TRID_END = 12;

/* Read request virtual channel - used by both command buffers */
constexpr uint32_t CMDBUF_RD_REQ_VC = 1;
/* Read response virtual channel - used by both command buffers */
constexpr uint32_t CMDBUF_RD_RESP_VC = 12;
/* Write request virtual channel - used by both command buffers */
constexpr uint32_t CMDBUF_WR_REQ_VC = 1;
/* Write response virtual channel - used by both command buffers */
constexpr uint32_t CMDBUF_WR_RESP_VC = 13;
/* Multicast request virtual channel - used by both command buffers */
constexpr uint32_t CMDBUF_MCAST_REQ_VC = 8;
/* Multicast response virtual channel - used by both command buffers */
constexpr uint32_t CMDBUF_MCAST_RESP_VC = 14;
/* First of the 8 iDMA backend request virtual channels (CMDBUF_FIRST_IDMA_VC..+7).
 * Kept independent of CMDBUF_WR_REQ_VC so that moving the unicast write VC cannot
 * walk the iDMA VC-autoincrement range into the multicast request VCs. */
constexpr uint32_t CMDBUF_FIRST_IDMA_VC = 0;
/* Number of iDMA backend engines the cmdbuf round-robins packets across */
constexpr uint32_t CMDBUF_NUM_IDMA_VCS = 8;

/* Transaction IDs owned by a command buffer: the static ID and the range used when TID wrapping is on */
struct TridRange {
    uint32_t static_trid;
    uint32_t start;
    uint32_t end;
};

constexpr TridRange cmdbuf_trid_range(CmdBuf cmdbuf) {
    return cmdbuf == CMDBUF_0 ? TridRange{CMDBUF_0_TRID_STATIC, CMDBUF_0_TRID_START, CMDBUF_0_TRID_END}
                              : TridRange{CMDBUF_1_TRID_STATIC, CMDBUF_1_TRID_START, CMDBUF_1_TRID_END};
}

/* Virtual channel set programmed by setup_vcs_cmdbuf() */
enum class NocVcs : uint8_t {
    /* CMDBUF_RD_REQ_VC / CMDBUF_RD_RESP_VC */
    READ,
    /* CMDBUF_WR_REQ_VC / CMDBUF_WR_RESP_VC */
    WRITE,
    /* CMDBUF_MCAST_REQ_VC / CMDBUF_MCAST_RESP_VC */
    MCAST_WRITE,
};

/* Packet tags (PACKET_TAGS register / fast-issue flags) */
struct PacketTags {
    /* Destination NIU snoops its cache */
    bool snoop = false;
    /* Destination NIU commits all parts of a flit before committing the next packet */
    bool flush = false;
};

/* Transfer options of a NOC copy or scatter-list transaction (MISC and MCAST_EXCLUDE registers) */
struct NocTransferConfig {
    /* Write (local -> remote) when true, read (remote -> local) otherwise */
    bool write = false;
    /* Posted write: no ack is returned. Reads are always non-posted, so this is ignored for reads */
    bool posted = true;
    /* Multicast write */
    bool mcast = false;
    /* Linked transaction: keeps the NOC path reserved for the transaction that follows */
    bool linked = false;
    /* Include the source core in the multicast; the RDL default of MISC.src_include is 1 */
    bool src_include = true;
    /* MISC.multicast_mode */
    bool mcast_mode = false;
    /* Cores excluded from the multicast */
    TT_ROCC_CMD_BUF_MCAST_EXCLUDE_reg_u mcast_exclude = {0};
    /* Wrap source/destination addresses inside [base, base + size) (see set_src_cmdbuf / set_dest_cmdbuf) */
    bool wrapping = false;
};

/* Scatter-list options (MISC register) */
struct ScatterListConfig {
    /* Scatter-list addresses replace the destination address when true, the source address otherwise */
    bool apply_to_dest = false;
    /* Each scatter-list element carries its own size */
    bool has_size = false;
    /* Each scatter-list element carries NOC XY coordinates */
    bool has_xy = false;
};

/* Per-issue auto-increment options (AUTOINC register) */
struct AutoIncConfig {
    /* Source address advances by the transfer length after every issue */
    bool src_addr = false;
    /* Destination address advances by the transfer length after every issue */
    bool dest_addr = false;
    /* Transaction ID advances after every issue (within the range set by setup_trids_cmdbuf) */
    bool trid = false;
    /* Request VC advances after every packet (within the range set by setup_wrapping_vcs_cmdbuf) */
    bool req_vc = false;
    /* Response VC advances after every packet */
    bool resp_vc = false;
    /* Request VC advances once per transaction instead of once per packet */
    bool req_vc_on_entire_trans = false;
    /* Response VC advances once per transaction instead of once per packet */
    bool resp_vc_on_entire_trans = false;
};

/* Inclusive virtual channel range used by setup_wrapping_vcs_cmdbuf() */
struct VcRange {
    /* First VC of the range */
    uint32_t start = 0;
    /* Last VC of the range, inclusive */
    uint32_t end = 0;
    /* VC the command buffer starts on, relative to start */
    uint32_t offset = 0;
};

/* Options of noc_read_cmdbuf() / noc_read_prep_cmdbuf() */
struct NocReadOptions {
    uint32_t trid = CMDBUF_DEF_TRID;
    PacketTags tags = {};
};

/* Options of noc_write_cmdbuf() / noc_write_prep_cmdbuf() */
struct NocWriteOptions {
    uint32_t trid = CMDBUF_DEF_TRID;
    /* Posted write: no ack is returned */
    bool posted = true;
    /* Multicast write; the destination coordinate is then the multicast rectangle */
    bool mcast = false;
    /* Linked transaction: keeps the NOC path reserved for the transaction that follows */
    bool linked = false;
    /* Include the source core in the multicast */
    bool src_include = true;
    /* Cores excluded from the multicast */
    TT_ROCC_CMD_BUF_MCAST_EXCLUDE_reg_u mcast_exclude = {0};
    PacketTags tags = {};
};

namespace detail {

inline __attribute__((always_inline)) TT_ROCC_CMD_BUF_MISC_reg_u noc_transfer_misc(const NocTransferConfig& cfg) {
    TT_ROCC_CMD_BUF_MISC_reg_u misc;
    misc.val = TT_ROCC_CMD_BUF_MISC_REG_DEFAULT;

    misc.f.write_trans = cfg.write;
    misc.f.posted = cfg.write && cfg.posted;
    misc.f.multicast = cfg.mcast;
    misc.f.multicast_mode = cfg.mcast_mode;
    misc.f.linked = cfg.linked;
    misc.f.src_include = cfg.src_include;
    misc.f.wrapping_en = cfg.wrapping;
    return misc;
}

/* Flag bits [63:60] of the rs2 operand of the fast-issue (read2/write2/inline-addr) instructions */
inline __attribute__((always_inline)) uint64_t fast_issue_flags(bool has_xy, bool posted, PacketTags tags) {
    return (static_cast<uint64_t>(has_xy) << 60) | (static_cast<uint64_t>(posted) << 61) |
           (static_cast<uint64_t>(tags.snoop) << 62) | (static_cast<uint64_t>(tags.flush) << 63);
}

}  // namespace detail

/*
 * @fn reset_cmdbuf<CMDBUF>()
 *
 * @brief Resets all registers of the command buffer to their RDL defaults.
 * Call before any other command buffer setup function.
 */
template <CmdBuf CMDBUF>
inline __attribute__((always_inline)) void reset_cmdbuf() {
    __builtin_riscv_ttrocc_cmdbuf_reset(CMDBUF);
}

/*
 * @fn setup_as_copy_cmdbuf<CMDBUF>
 *
 * @brief Configures the command buffer for NOC copy transactions
 *
 * @param cfg Transfer options; MCAST_EXCLUDE is written only for multicast
 */
template <CmdBuf CMDBUF>
inline __attribute__((always_inline)) void setup_as_copy_cmdbuf(const NocTransferConfig& cfg) {
    __builtin_riscv_ttrocc_cmdbuf_wr_reg(
        CMDBUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_MISC_REG_OFFSET / 8, detail::noc_transfer_misc(cfg).val);

    if (cfg.mcast) {
        __builtin_riscv_ttrocc_cmdbuf_wr_reg(
            CMDBUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_MCAST_EXCLUDE_REG_OFFSET / 8, cfg.mcast_exclude.val);
    }
}

/*
 * @fn setup_as_scatter_list_cmdbuf<CMDBUF>
 *
 * @brief Configures the command buffer for NOC scatter-list transactions
 *
 * @param cfg Transfer options; MCAST_EXCLUDE is written only for multicast
 * @param scatter Scatter-list options
 *
 * @note To be used with set_scatter_list_cmdbuf()
 */
template <CmdBuf CMDBUF>
inline __attribute__((always_inline)) void setup_as_scatter_list_cmdbuf(
    const NocTransferConfig& cfg, const ScatterListConfig& scatter) {
    TT_ROCC_CMD_BUF_MISC_reg_u misc = detail::noc_transfer_misc(cfg);
    misc.f.scatter_list_en = true;
    misc.f.scatter_list_to_dest_addr = scatter.apply_to_dest;
    misc.f.scatter_list_has_size = scatter.has_size;
    misc.f.scatter_list_has_xy = scatter.has_xy;

    __builtin_riscv_ttrocc_cmdbuf_wr_reg(CMDBUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_MISC_REG_OFFSET / 8, misc.val);

    if (cfg.mcast) {
        __builtin_riscv_ttrocc_cmdbuf_wr_reg(
            CMDBUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_MCAST_EXCLUDE_REG_OFFSET / 8, cfg.mcast_exclude.val);
    }
}

/*
 * @fn setup_as_atomic_cmdbuf<CMDBUF>
 *
 * @brief Configures the command buffer for posted NOC atomic transactions
 *
 * @note Overwrites the MISC register
 */
template <CmdBuf CMDBUF>
inline __attribute__((always_inline)) void setup_as_atomic_cmdbuf() {
    TT_ROCC_CMD_BUF_MISC_reg_u misc;
    misc.val = TT_ROCC_CMD_BUF_MISC_REG_DEFAULT;

    misc.f.posted = 1;
    misc.f.write_trans = 0;
    misc.f.atomic_trans = 1;

    __builtin_riscv_ttrocc_cmdbuf_wr_reg(CMDBUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_MISC_REG_OFFSET / 8, misc.val);
}

/*
 * @fn idma_setup_as_copy_cmdbuf<CMDBUF>
 *
 * @brief Configures the command buffer for iDMA copy operations
 *
 * @param wrapping Wrap source/destination addresses inside [base, base + size)
 */
template <CmdBuf CMDBUF>
inline __attribute__((always_inline)) void idma_setup_as_copy_cmdbuf(bool wrapping = false) {
    TT_ROCC_CMD_BUF_MISC_reg_u misc;
    misc.val = TT_ROCC_CMD_BUF_MISC_REG_DEFAULT;

    misc.f.write_trans = 1;
    misc.f.idma_en = 1;
    misc.f.wrapping_en = wrapping;

    __builtin_riscv_ttrocc_cmdbuf_wr_reg(CMDBUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_MISC_REG_OFFSET / 8, misc.val);
}

/*
 * @fn idma_setup_as_scatter_list_cmdbuf<CMDBUF>
 *
 * @brief Configures the command buffer for iDMA scatter-list operations
 *
 * @param scatter Scatter-list options
 * @param wrapping Wrap source/destination addresses inside [base, base + size)
 */
template <CmdBuf CMDBUF>
inline __attribute__((always_inline)) void idma_setup_as_scatter_list_cmdbuf(
    const ScatterListConfig& scatter, bool wrapping = false) {
    TT_ROCC_CMD_BUF_MISC_reg_u misc;
    misc.val = TT_ROCC_CMD_BUF_MISC_REG_DEFAULT;

    misc.f.scatter_list_en = true;
    misc.f.scatter_list_to_dest_addr = scatter.apply_to_dest;
    misc.f.write_trans = 1;
    misc.f.scatter_list_has_size = scatter.has_size;
    misc.f.scatter_list_has_xy = scatter.has_xy;
    misc.f.idma_en = 1;
    misc.f.wrapping_en = wrapping;

    __builtin_riscv_ttrocc_cmdbuf_wr_reg(CMDBUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_MISC_REG_OFFSET / 8, misc.val);
}

/*
 * @fn idma_setup_as_atomic_accum_cmdbuf<CMDBUF>
 *
 * @brief Configures the command buffer for iDMA L1 atomic accumulation (see l1_atomic_instr_cmdbuf)
 *
 * @param wrapping Wrap source/destination addresses inside [base, base + size)
 */
template <CmdBuf CMDBUF>
inline __attribute__((always_inline)) void idma_setup_as_atomic_accum_cmdbuf(bool wrapping = false) {
    TT_ROCC_CMD_BUF_MISC_reg_u misc;
    misc.val = TT_ROCC_CMD_BUF_MISC_REG_DEFAULT;

    misc.f.write_trans = 1;
    misc.f.idma_en = 1;
    misc.f.wrapping_en = wrapping;
    misc.f.l1_accum_en = 1;

    __builtin_riscv_ttrocc_cmdbuf_wr_reg(CMDBUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_MISC_REG_OFFSET / 8, misc.val);
}

/*
 * @fn l1_atomic_instr_cmdbuf<CMDBUF>
 *
 * @brief Programs the L1 accumulation format and operation (L1_ACCUM_CFG register)
 *
 * @param fmt L1 atomic data format
 * @param disable_saturation Disable saturation of the accumulated result
 * @param atomic_op L1 atomic operation
 */
template <CmdBuf CMDBUF>
inline __attribute__((always_inline)) void l1_atomic_instr_cmdbuf(
    uint32_t fmt, bool disable_saturation, uint32_t atomic_op) {
    TT_ROCC_CMD_BUF_L1_ACCUM_CFG_reg_u l1_atomic_instr;
    l1_atomic_instr.val = TT_ROCC_CMD_BUF_L1_ACCUM_CFG_REG_DEFAULT;

    l1_atomic_instr.f.l1_atomic_fmt = fmt;
    l1_atomic_instr.f.disable_sat = disable_saturation;
    l1_atomic_instr.f.l1_atomic_operation = atomic_op;

    __builtin_riscv_ttrocc_cmdbuf_wr_reg(
        CMDBUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_L1_ACCUM_CFG_REG_OFFSET / 8, l1_atomic_instr.val);
}

/*
 * @fn set_axi_opt_1_cmdbuf<CMDBUF>
 *
 * @brief Programs the AXI_OPT_1 cmdbuf register. All fields not exposed as parameters
 *        are written at their TT_ROCC_CMD_BUF_AXI_OPT_1_REG_DEFAULT values.
 *
 * @param src_protocol AXI source protocol selector
 * @param decouple_aw  Decouple AXI AW from W channel
 */
template <CmdBuf CMDBUF>
inline __attribute__((always_inline)) void set_axi_opt_1_cmdbuf(uint8_t src_protocol, uint8_t decouple_aw) {
    TT_ROCC_CMD_BUF_AXI_OPT_1_reg_u axi_opt_1;
    axi_opt_1.val = TT_ROCC_CMD_BUF_AXI_OPT_1_REG_DEFAULT;
    axi_opt_1.f.src_protocol = src_protocol;
    axi_opt_1.f.decouple_aw = decouple_aw;

    __builtin_riscv_ttrocc_cmdbuf_wr_reg(
        CMDBUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_AXI_OPT_1_REG_OFFSET / 8, axi_opt_1.val);
}

/*
 * @fn setup_ongoing_cmdbuf<CMDBUF>
 *
 * @brief Configures which addresses, VCs and transaction IDs advance automatically after each issue
 *
 * @param cfg Auto-increment options; all off by default
 */
template <CmdBuf CMDBUF>
inline __attribute__((always_inline)) void setup_ongoing_cmdbuf(const AutoIncConfig& cfg) {
    TT_ROCC_CMD_BUF_AUTOINC_reg_u ongoing;
    ongoing.val = TT_ROCC_CMD_BUF_AUTOINC_REG_DEFAULT;

    ongoing.f.src_addr_inc_en = cfg.src_addr;
    ongoing.f.dest_addr_inc_en = cfg.dest_addr;
    ongoing.f.trid_inc_en = cfg.trid;
    ongoing.f.req_vc_inc_en = cfg.req_vc;
    ongoing.f.resp_vc_inc_en = cfg.resp_vc;
    ongoing.f.req_vc_inc_on_entire_trans = cfg.req_vc_on_entire_trans;
    ongoing.f.resp_vc_inc_on_entire_trans = cfg.resp_vc_on_entire_trans;

    __builtin_riscv_ttrocc_cmdbuf_wr_reg(
        CMDBUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_AUTOINC_REG_OFFSET / 8, ongoing.val);
}

/*
 * @fn setup_vcs_cmdbuf<CMDBUF>
 *
 * @brief Programs the request/response virtual channels from the CMDBUF_*_VC constants in this file
 *
 * @param vcs Virtual channel set
 */
template <CmdBuf CMDBUF>
inline __attribute__((always_inline)) void setup_vcs_cmdbuf(NocVcs vcs) {
    const uint32_t req_vc = vcs == NocVcs::READ    ? CMDBUF_RD_REQ_VC
                            : vcs == NocVcs::WRITE ? CMDBUF_WR_REQ_VC
                                                   : CMDBUF_MCAST_REQ_VC;
    const uint32_t resp_vc = vcs == NocVcs::READ    ? CMDBUF_RD_RESP_VC
                             : vcs == NocVcs::WRITE ? CMDBUF_WR_RESP_VC
                                                    : CMDBUF_MCAST_RESP_VC;
    __builtin_riscv_ttrocc_cmdbuf_wr_reg(CMDBUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_REQ_VC_REG_OFFSET / 8, req_vc);
    __builtin_riscv_ttrocc_cmdbuf_wr_reg(CMDBUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_RESP_VC_REG_OFFSET / 8, resp_vc);
}

/*
 * @fn setup_wrapping_vcs_cmdbuf<CMDBUF>
 *
 * @brief Programs wrapping request and response virtual channel ranges (used with AutoIncConfig::req_vc /
 * resp_vc)
 *
 * @param req Request VC range
 * @param resp Response VC range
 */
template <CmdBuf CMDBUF>
inline __attribute__((always_inline)) void setup_wrapping_vcs_cmdbuf(const VcRange& req, const VcRange& resp = {}) {
    __builtin_riscv_ttrocc_cmdbuf_wr_reg(
        CMDBUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_REQ_VC_REG_OFFSET / 8, req.start + req.offset);
    __builtin_riscv_ttrocc_cmdbuf_wr_reg(
        CMDBUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_REQ_VC_BASE_REG_OFFSET / 8, req.start);
    __builtin_riscv_ttrocc_cmdbuf_wr_reg(
        CMDBUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_REQ_VC_SIZE_REG_OFFSET / 8, req.end - req.start + 1);
    __builtin_riscv_ttrocc_cmdbuf_wr_reg(
        CMDBUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_RESP_VC_REG_OFFSET / 8, resp.start + resp.offset);
    __builtin_riscv_ttrocc_cmdbuf_wr_reg(
        CMDBUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_RESP_VC_BASE_REG_OFFSET / 8, resp.start);
    __builtin_riscv_ttrocc_cmdbuf_wr_reg(
        CMDBUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_RESP_VC_SIZE_REG_OFFSET / 8, resp.end - resp.start + 1);
}

/*
 * @fn setup_trids_cmdbuf<CMDBUF>
 *
 * @brief Programs the transaction ID used for issue, write-sent and ack tracking
 *
 * @param trid Transaction ID; with wrapping it is an offset into the command buffer's TID range
 * @param wrapping Wrap the transaction ID inside the command buffer's range (cmdbuf_trid_range)
 */
template <CmdBuf CMDBUF>
inline __attribute__((always_inline)) void setup_trids_cmdbuf(uint32_t trid = CMDBUF_DEF_TRID, bool wrapping = false) {
    constexpr TridRange range = cmdbuf_trid_range(CMDBUF);
    const uint32_t first_trid = wrapping ? range.start + trid : trid;

    __builtin_riscv_ttrocc_cmdbuf_wr_reg(
        CMDBUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_WR_SENT_TR_ID_REG_OFFSET / 8, first_trid);
    __builtin_riscv_ttrocc_cmdbuf_wr_reg(
        CMDBUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_TR_ACK_TR_ID_REG_OFFSET / 8, first_trid);
    __builtin_riscv_ttrocc_cmdbuf_wr_reg(CMDBUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_TR_ID_REG_OFFSET / 8, first_trid);
    if (wrapping) {
        __builtin_riscv_ttrocc_cmdbuf_wr_reg(
            CMDBUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_TR_ID_BASE_REG_OFFSET / 8, range.start);
        __builtin_riscv_ttrocc_cmdbuf_wr_reg(
            CMDBUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_TR_ID_SIZE_REG_OFFSET / 8, range.end - range.start + 1);
    }
}

/*
 * @fn swap_trid_cmdbuf<CMDBUF>
 *
 * @brief Swaps the current transaction ID with a new one
 *
 * @param new_trid New transaction ID
 * @return Previous transaction ID
 */
template <CmdBuf CMDBUF>
inline __attribute__((always_inline)) uint32_t swap_trid_cmdbuf(uint32_t new_trid) {
    uint32_t prev_trid =
        __builtin_riscv_ttrocc_cmdbuf_rd_reg(CMDBUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_TR_ID_REG_OFFSET / 8);
    __builtin_riscv_ttrocc_cmdbuf_wr_reg(CMDBUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_TR_ID_REG_OFFSET / 8, new_trid);
    return prev_trid;
}

/*
 * @fn setup_max_bytes_in_packet_cmdbuf<CMDBUF>
 *
 * @brief Sets the maximum number of bytes per packet
 *
 * @param max_bytes_in_packet Maximum bytes in a single packet
 */
template <CmdBuf CMDBUF>
inline __attribute__((always_inline)) void setup_max_bytes_in_packet_cmdbuf(uint64_t max_bytes_in_packet) {
    __builtin_riscv_ttrocc_cmdbuf_wr_reg(
        CMDBUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_MAX_BYTES_IN_PACKET_REG_OFFSET / 8, max_bytes_in_packet);
}

/*
 * @fn setup_packet_tags_cmdbuf<CMDBUF>
 *
 * @brief Programs the snoop and flush packet tags
 *
 * @param tags Packet tags
 */
template <CmdBuf CMDBUF>
inline __attribute__((always_inline)) void setup_packet_tags_cmdbuf(PacketTags tags) {
    TT_ROCC_CMD_BUF_PACKET_TAGS_reg_u packet_tags;
    packet_tags.val = TT_ROCC_CMD_BUF_PACKET_TAGS_REG_DEFAULT;

    packet_tags.f.snoop_bit = tags.snoop;
    packet_tags.f.flush_bit = tags.flush;

    __builtin_riscv_ttrocc_cmdbuf_wr_reg(
        CMDBUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_PACKET_TAGS_REG_OFFSET / 8, packet_tags.val);
}

/*
 * @fn get_src_cmdbuf<CMDBUF>
 *
 * @return Source address of the next transaction
 */
template <CmdBuf CMDBUF>
inline __attribute__((always_inline)) uint64_t get_src_cmdbuf() {
    return __builtin_riscv_ttrocc_cmdbuf_rd_reg(CMDBUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_SRC_ADDR_REG_OFFSET / 8);
}

/*
 * @fn set_src_cmdbuf<CMDBUF>
 *
 * @brief Sets the source of the next transaction. Overloads with fewer parameters leave the omitted
 * registers unchanged.
 *
 * @param addr Source address without coordinates
 * @param coord Source coordinate generated with the NOC_XY_COORD macro
 * @param wrap_base Base of the wrapping window (NocTransferConfig::wrapping)
 * @param wrap_size Size of the wrapping window in bytes
 */
template <CmdBuf CMDBUF>
inline __attribute__((always_inline)) void set_src_cmdbuf(
    uint64_t addr, uint64_t coord, uint64_t wrap_base, uint64_t wrap_size) {
    __builtin_riscv_ttrocc_cmdbuf_wr_reg(CMDBUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_SRC_ADDR_REG_OFFSET / 8, addr);
    __builtin_riscv_ttrocc_cmdbuf_wr_reg(
        CMDBUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_SRC_BASE_REG_OFFSET / 8, wrap_base);
    __builtin_riscv_ttrocc_cmdbuf_wr_reg(
        CMDBUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_SRC_SIZE_REG_OFFSET / 8, wrap_size);
    __builtin_riscv_ttrocc_cmdbuf_wr_reg(CMDBUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_SRC_COORD_REG_OFFSET / 8, coord);
}
template <CmdBuf CMDBUF>
inline __attribute__((always_inline)) void set_src_cmdbuf(uint64_t addr, uint64_t coord, uint64_t wrap_base) {
    __builtin_riscv_ttrocc_cmdbuf_wr_reg(CMDBUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_SRC_ADDR_REG_OFFSET / 8, addr);
    __builtin_riscv_ttrocc_cmdbuf_wr_reg(
        CMDBUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_SRC_BASE_REG_OFFSET / 8, wrap_base);
    __builtin_riscv_ttrocc_cmdbuf_wr_reg(CMDBUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_SRC_COORD_REG_OFFSET / 8, coord);
}
template <CmdBuf CMDBUF>
inline __attribute__((always_inline)) void set_src_cmdbuf(uint64_t addr, uint64_t coord) {
    __builtin_riscv_ttrocc_cmdbuf_wr_reg(CMDBUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_SRC_ADDR_REG_OFFSET / 8, addr);
    __builtin_riscv_ttrocc_cmdbuf_wr_reg(CMDBUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_SRC_COORD_REG_OFFSET / 8, coord);
}
template <CmdBuf CMDBUF>
inline __attribute__((always_inline)) void set_src_cmdbuf(uint64_t addr) {
    __builtin_riscv_ttrocc_cmdbuf_wr_reg(CMDBUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_SRC_ADDR_REG_OFFSET / 8, addr);
}

/*
 * @fn get_dest_cmdbuf<CMDBUF>
 *
 * @return Destination address of the next transaction
 */
template <CmdBuf CMDBUF>
inline __attribute__((always_inline)) uint64_t get_dest_cmdbuf() {
    return __builtin_riscv_ttrocc_cmdbuf_rd_reg(CMDBUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_DEST_ADDR_REG_OFFSET / 8);
}

/*
 * @fn set_dest_cmdbuf<CMDBUF>
 *
 * @brief Sets the destination of the next transaction. Overloads with fewer parameters leave the omitted
 * registers unchanged.
 *
 * @param addr Destination address without coordinates
 * @param coord Destination coordinate generated with the NOC_XY_COORD macro
 * @param wrap_base Base of the wrapping window (NocTransferConfig::wrapping)
 * @param wrap_size Size of the wrapping window in bytes
 */
template <CmdBuf CMDBUF>
inline __attribute__((always_inline)) void set_dest_cmdbuf(
    uint64_t addr, uint64_t coord, uint64_t wrap_base, uint64_t wrap_size) {
    __builtin_riscv_ttrocc_cmdbuf_wr_reg(CMDBUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_DEST_ADDR_REG_OFFSET / 8, addr);
    __builtin_riscv_ttrocc_cmdbuf_wr_reg(
        CMDBUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_DEST_BASE_REG_OFFSET / 8, wrap_base);
    __builtin_riscv_ttrocc_cmdbuf_wr_reg(
        CMDBUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_DEST_SIZE_REG_OFFSET / 8, wrap_size);
    __builtin_riscv_ttrocc_cmdbuf_wr_reg(CMDBUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_DEST_COORD_REG_OFFSET / 8, coord);
}
template <CmdBuf CMDBUF>
inline __attribute__((always_inline)) void set_dest_cmdbuf(uint64_t addr, uint64_t coord, uint64_t wrap_base) {
    __builtin_riscv_ttrocc_cmdbuf_wr_reg(CMDBUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_DEST_ADDR_REG_OFFSET / 8, addr);
    __builtin_riscv_ttrocc_cmdbuf_wr_reg(
        CMDBUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_DEST_BASE_REG_OFFSET / 8, wrap_base);
    __builtin_riscv_ttrocc_cmdbuf_wr_reg(CMDBUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_DEST_COORD_REG_OFFSET / 8, coord);
}
template <CmdBuf CMDBUF>
inline __attribute__((always_inline)) void set_dest_cmdbuf(uint64_t addr, uint64_t coord) {
    __builtin_riscv_ttrocc_cmdbuf_wr_reg(CMDBUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_DEST_ADDR_REG_OFFSET / 8, addr);
    __builtin_riscv_ttrocc_cmdbuf_wr_reg(CMDBUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_DEST_COORD_REG_OFFSET / 8, coord);
}
template <CmdBuf CMDBUF>
inline __attribute__((always_inline)) void set_dest_cmdbuf(uint64_t addr) {
    __builtin_riscv_ttrocc_cmdbuf_wr_reg(CMDBUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_DEST_ADDR_REG_OFFSET / 8, addr);
}

/*
 * @fn set_scatter_list_cmdbuf<CMDBUF>
 *
 * @brief Programs the scatter list
 *
 * @param addr Scatter list address
 * @param base Base address the scatter-list entries are relative to
 * @param index Index of the first entry
 * @param times Number of entries
 */
template <CmdBuf CMDBUF>
inline __attribute__((always_inline)) void set_scatter_list_cmdbuf(
    uint64_t addr, uint64_t base, uint64_t index, uint64_t times) {
    __builtin_riscv_ttrocc_cmdbuf_wr_reg(
        CMDBUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_SCATTER_LIST_ADDR_REG_OFFSET / 8, addr);
    __builtin_riscv_ttrocc_cmdbuf_wr_reg(
        CMDBUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_SCATTER_BASE_ADDR_REG_OFFSET / 8, base);
    __builtin_riscv_ttrocc_cmdbuf_wr_reg(
        CMDBUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_SCATTER_INDEX_REG_OFFSET / 8, index);
    __builtin_riscv_ttrocc_cmdbuf_wr_reg(
        CMDBUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_SCATTER_TIMES_REG_OFFSET / 8, times);
}
template <CmdBuf CMDBUF>
inline __attribute__((always_inline)) void set_scatter_list_base_cmdbuf(uint64_t base) {
    __builtin_riscv_ttrocc_cmdbuf_wr_reg(
        CMDBUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_SCATTER_BASE_ADDR_REG_OFFSET / 8, base);
}
template <CmdBuf CMDBUF>
inline __attribute__((always_inline)) void set_scatter_list_index_cmdbuf(uint64_t index) {
    __builtin_riscv_ttrocc_cmdbuf_wr_reg(
        CMDBUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_SCATTER_INDEX_REG_OFFSET / 8, index);
}
template <CmdBuf CMDBUF>
inline __attribute__((always_inline)) void set_scatter_list_times_cmdbuf(uint64_t times) {
    __builtin_riscv_ttrocc_cmdbuf_wr_reg(
        CMDBUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_SCATTER_TIMES_REG_OFFSET / 8, times);
}
template <CmdBuf CMDBUF>
inline __attribute__((always_inline)) uint64_t get_scatter_list_addr_cmdbuf() {
    return __builtin_riscv_ttrocc_cmdbuf_rd_reg(
        CMDBUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_SCATTER_LIST_ADDR_REG_OFFSET / 8);
}
template <CmdBuf CMDBUF>
inline __attribute__((always_inline)) uint64_t get_scatter_list_base_cmdbuf() {
    return __builtin_riscv_ttrocc_cmdbuf_rd_reg(
        CMDBUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_SCATTER_BASE_ADDR_REG_OFFSET / 8);
}
template <CmdBuf CMDBUF>
inline __attribute__((always_inline)) uint64_t get_scatter_list_index_cmdbuf() {
    return __builtin_riscv_ttrocc_cmdbuf_rd_reg(
        CMDBUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_SCATTER_INDEX_REG_OFFSET / 8);
}
template <CmdBuf CMDBUF>
inline __attribute__((always_inline)) uint64_t get_scatter_list_times_cmdbuf() {
    return __builtin_riscv_ttrocc_cmdbuf_rd_reg(
        CMDBUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_SCATTER_TIMES_REG_OFFSET / 8);
}

/*
 * @fn set_len_cmdbuf<CMDBUF>
 *
 * @brief Sets the length of the next transaction
 *
 * @param len_bytes Length in bytes
 */
template <CmdBuf CMDBUF>
inline __attribute__((always_inline)) void set_len_cmdbuf(uint64_t len_bytes) {
    __builtin_riscv_ttrocc_cmdbuf_wr_reg(
        CMDBUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_LEN_BYTES_REG_OFFSET / 8, len_bytes);
}

/*
 * @fn issue_cmdbuf<CMDBUF>
 *
 * @brief Issues the transaction programmed into the command buffer
 */
template <CmdBuf CMDBUF>
inline __attribute__((always_inline)) void issue_cmdbuf() {
    __builtin_riscv_ttrocc_cmdbuf_issue_trans(CMDBUF);
}

/*
 * @fn issue_read_cmdbuf<CMDBUF>
 *
 * @brief Issues the read transaction programmed into the command buffer (same as issue_cmdbuf)
 */
template <CmdBuf CMDBUF>
inline __attribute__((always_inline)) void issue_read_cmdbuf() {
    issue_cmdbuf<CMDBUF>();
}

/*
 * @fn issue_write_cmdbuf<CMDBUF>
 *
 * @brief Issues the write transaction programmed into the command buffer (same as issue_cmdbuf)
 */
template <CmdBuf CMDBUF>
inline __attribute__((always_inline)) void issue_write_cmdbuf() {
    issue_cmdbuf<CMDBUF>();
}

/*
 * @fn issue_write_inline_cmdbuf<CMDBUF>
 *
 * @brief Issues an inline write of the given data using the programmed destination
 *
 * @param data Inline data to be written
 */
template <CmdBuf CMDBUF>
inline __attribute__((always_inline)) void issue_write_inline_cmdbuf(uint64_t data) {
    __builtin_riscv_ttrocc_cmdbuf_issue_inline_trans(CMDBUF, data);
}

/*
 * @fn issue_write_inline_len_cmdbuf<CMDBUF>
 *
 * @brief Issues an inline write with its destination, length and flags given in the instruction
 *
 * @param data Inline data to be written
 * @param dest_addr Destination address (with or without XY coordinates embedded)
 * @param len_bytes Length of the data in bytes (1-8)
 * @param has_xy Destination address contains NOC coordinates (NOC_XY_COORD)
 * @param posted Posted write: no ack is returned
 * @param tags Packet tags
 */
template <CmdBuf CMDBUF>
inline __attribute__((always_inline)) void issue_write_inline_len_cmdbuf(
    uint64_t data,
    uint64_t dest_addr,
    uint64_t len_bytes,
    bool has_xy = true,
    bool posted = true,
    PacketTags tags = {}) {
    uint64_t rs2 = dest_addr | ((len_bytes - 1) << 57) | detail::fast_issue_flags(has_xy, posted, tags);
    __builtin_riscv_ttrocc_cmdbuf_issue_inline_addr_trans(CMDBUF, data, rs2);
}

/* Interrupt enables (IE) and pending bits (IP). Pending bits are cleared by writing 0. */
template <CmdBuf CMDBUF>
inline __attribute__((always_inline)) void interrupt_enable_cmdbuf(uint32_t id) {
    TT_ROCC_CMD_BUF_IE_reg_u val;
    val.val = __builtin_riscv_ttrocc_cmdbuf_rd_reg(CMDBUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_IE_REG_OFFSET / 8);
    val.val |= (1ULL << id);
    __builtin_riscv_ttrocc_cmdbuf_wr_reg(CMDBUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_IE_REG_OFFSET / 8, val.val);
}

template <CmdBuf CMDBUF>
inline __attribute__((always_inline)) void interrupt_disable_cmdbuf(uint32_t id) {
    TT_ROCC_CMD_BUF_IE_reg_u val;
    val.val = __builtin_riscv_ttrocc_cmdbuf_rd_reg(CMDBUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_IE_REG_OFFSET / 8);
    val.val &= ~(1ULL << id);
    __builtin_riscv_ttrocc_cmdbuf_wr_reg(CMDBUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_IE_REG_OFFSET / 8, val.val);
}

template <CmdBuf CMDBUF>
inline __attribute__((always_inline)) uint64_t interrupts_pending_cmdbuf() {
    TT_ROCC_CMD_BUF_IP_reg_u val;
    val.val = __builtin_riscv_ttrocc_cmdbuf_rd_reg(CMDBUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_IP_REG_OFFSET / 8);
    return val.val;
}

template <CmdBuf CMDBUF>
inline __attribute__((always_inline)) void interrupt_clear_cmdbuf(uint32_t id) {
    TT_ROCC_CMD_BUF_IP_reg_u val;
    val.val = ~(1ULL << id);
    __builtin_riscv_ttrocc_cmdbuf_wr_reg(CMDBUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_IP_REG_OFFSET / 8, val.val);
}
/* Per-TRID Count Zero Interrupts (IE_0[31:0], IP_0[31:0]) */
template <CmdBuf CMDBUF>
inline __attribute__((always_inline)) void per_trid_count_zero_interrupt_enable_cmdbuf(uint32_t trid) {
    uint64_t val;
    val = __builtin_riscv_ttrocc_cmdbuf_rd_reg(
        CMDBUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_PER_TR_ID_IE_0_REG_OFFSET / 8);
    val |= (1ULL << trid);
    __builtin_riscv_ttrocc_cmdbuf_wr_reg(
        CMDBUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_PER_TR_ID_IE_0_REG_OFFSET / 8, val);
}

template <CmdBuf CMDBUF>
inline __attribute__((always_inline)) void per_trid_count_zero_interrupt_disable_cmdbuf(uint32_t trid) {
    uint64_t val;
    val = __builtin_riscv_ttrocc_cmdbuf_rd_reg(
        CMDBUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_PER_TR_ID_IE_0_REG_OFFSET / 8);
    val &= ~(1ULL << trid);
    __builtin_riscv_ttrocc_cmdbuf_wr_reg(
        CMDBUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_PER_TR_ID_IE_0_REG_OFFSET / 8, val);
}

template <CmdBuf CMDBUF>
inline __attribute__((always_inline)) uint32_t per_trid_count_zero_get_interrupt_enable_cmdbuf() {
    uint64_t val;
    val = __builtin_riscv_ttrocc_cmdbuf_rd_reg(
        CMDBUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_PER_TR_ID_IE_0_REG_OFFSET / 8);
    return val & 0xFFFFFFFFULL; /* Lower 32 bits */
}

template <CmdBuf CMDBUF>
inline __attribute__((always_inline)) void per_trid_count_zero_set_interrupt_enable_cmdbuf(uint32_t val) {
    uint64_t reg_val;
    reg_val = __builtin_riscv_ttrocc_cmdbuf_rd_reg(
        CMDBUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_PER_TR_ID_IE_0_REG_OFFSET / 8);
    reg_val = (reg_val & 0xFFFFFFFF00000000ULL) | (val & 0xFFFFFFFFULL);
    __builtin_riscv_ttrocc_cmdbuf_wr_reg(
        CMDBUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_PER_TR_ID_IE_0_REG_OFFSET / 8, reg_val);
}

template <CmdBuf CMDBUF>
inline __attribute__((always_inline)) uint64_t per_trid_count_zero_interrupts_pending_cmdbuf() {
    uint64_t val;
    val = __builtin_riscv_ttrocc_cmdbuf_rd_reg(
        CMDBUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_PER_TR_ID_IP_0_REG_OFFSET / 8);
    return val & 0xFFFFFFFFULL; /* Lower 32 bits */
}

template <CmdBuf CMDBUF>
inline __attribute__((always_inline)) uint32_t per_trid_count_zero_get_interrupt_pending_cmdbuf() {
    uint64_t val;
    val = __builtin_riscv_ttrocc_cmdbuf_rd_reg(
        CMDBUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_PER_TR_ID_IP_0_REG_OFFSET / 8);
    return val & 0xFFFFFFFFULL; /* Lower 32 bits */
}

template <CmdBuf CMDBUF>
inline __attribute__((always_inline)) void per_trid_count_zero_set_interrupt_pending_cmdbuf(uint32_t val) {
    uint64_t reg_val;
    reg_val = __builtin_riscv_ttrocc_cmdbuf_rd_reg(
        CMDBUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_PER_TR_ID_IP_0_REG_OFFSET / 8);
    reg_val = (reg_val & 0xFFFFFFFF00000000ULL) | (val & 0xFFFFFFFFULL);
    __builtin_riscv_ttrocc_cmdbuf_wr_reg(
        CMDBUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_PER_TR_ID_IP_0_REG_OFFSET / 8, reg_val);
}

template <CmdBuf CMDBUF>
inline __attribute__((always_inline)) void per_trid_count_zero_interrupt_clear_cmdbuf(uint32_t trid) {
    uint64_t val;
    val = ~(1ULL << trid);
    __builtin_riscv_ttrocc_cmdbuf_wr_reg(
        CMDBUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_PER_TR_ID_IP_0_REG_OFFSET / 8, val);
}

/* Per-TRID Write Count Zero Interrupts (IE_0[63:32], IP_0[63:32]) */
template <CmdBuf CMDBUF>
inline __attribute__((always_inline)) void per_trid_wr_count_zero_interrupt_enable_cmdbuf(uint32_t trid) {
    uint64_t val;
    val = __builtin_riscv_ttrocc_cmdbuf_rd_reg(
        CMDBUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_PER_TR_ID_IE_0_REG_OFFSET / 8);
    val |= (1ULL << (trid + 32));
    __builtin_riscv_ttrocc_cmdbuf_wr_reg(
        CMDBUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_PER_TR_ID_IE_0_REG_OFFSET / 8, val);
}

template <CmdBuf CMDBUF>
inline __attribute__((always_inline)) void per_trid_wr_count_zero_interrupt_disable_cmdbuf(uint32_t trid) {
    uint64_t val;
    val = __builtin_riscv_ttrocc_cmdbuf_rd_reg(
        CMDBUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_PER_TR_ID_IE_0_REG_OFFSET / 8);
    val &= ~(1ULL << (trid + 32));
    __builtin_riscv_ttrocc_cmdbuf_wr_reg(
        CMDBUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_PER_TR_ID_IE_0_REG_OFFSET / 8, val);
}

template <CmdBuf CMDBUF>
inline __attribute__((always_inline)) uint32_t per_trid_wr_count_zero_get_interrupt_enable_cmdbuf() {
    uint64_t val;
    val = __builtin_riscv_ttrocc_cmdbuf_rd_reg(
        CMDBUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_PER_TR_ID_IE_0_REG_OFFSET / 8);
    return (val >> 32) & 0xFFFFFFFFULL; /* Upper 32 bits */
}

template <CmdBuf CMDBUF>
inline __attribute__((always_inline)) void per_trid_wr_count_zero_set_interrupt_enable_cmdbuf(uint32_t val) {
    uint64_t reg_val;
    reg_val = __builtin_riscv_ttrocc_cmdbuf_rd_reg(
        CMDBUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_PER_TR_ID_IE_0_REG_OFFSET / 8);
    reg_val = (reg_val & 0x00000000FFFFFFFFULL) | ((uint64_t)(val & 0xFFFFFFFFULL) << 32);
    __builtin_riscv_ttrocc_cmdbuf_wr_reg(
        CMDBUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_PER_TR_ID_IE_0_REG_OFFSET / 8, reg_val);
}

template <CmdBuf CMDBUF>
inline __attribute__((always_inline)) uint64_t per_trid_wr_count_zero_interrupts_pending_cmdbuf() {
    uint64_t val;
    val = __builtin_riscv_ttrocc_cmdbuf_rd_reg(
        CMDBUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_PER_TR_ID_IP_0_REG_OFFSET / 8);
    return (val >> 32) & 0xFFFFFFFFULL; /* Upper 32 bits */
}

template <CmdBuf CMDBUF>
inline __attribute__((always_inline)) uint32_t per_trid_wr_count_zero_get_interrupt_pending_cmdbuf() {
    uint64_t val;
    val = __builtin_riscv_ttrocc_cmdbuf_rd_reg(
        CMDBUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_PER_TR_ID_IP_0_REG_OFFSET / 8);
    return (val >> 32) & 0xFFFFFFFFULL; /* Upper 32 bits */
}

template <CmdBuf CMDBUF>
inline __attribute__((always_inline)) void per_trid_wr_count_zero_set_interrupt_pending_cmdbuf(uint32_t val) {
    uint64_t reg_val;
    reg_val = __builtin_riscv_ttrocc_cmdbuf_rd_reg(
        CMDBUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_PER_TR_ID_IP_0_REG_OFFSET / 8);
    reg_val = (reg_val & 0x00000000FFFFFFFFULL) | ((uint64_t)(val & 0xFFFFFFFFULL) << 32);
    __builtin_riscv_ttrocc_cmdbuf_wr_reg(
        CMDBUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_PER_TR_ID_IP_0_REG_OFFSET / 8, reg_val);
}

template <CmdBuf CMDBUF>
inline __attribute__((always_inline)) void per_trid_wr_count_zero_interrupt_clear_cmdbuf(uint32_t trid) {
    uint64_t val;
    val = ~(1ULL << (trid + 32));
    __builtin_riscv_ttrocc_cmdbuf_wr_reg(
        CMDBUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_PER_TR_ID_IP_0_REG_OFFSET / 8, val);
}

/* Per-TRID iDMA Count Zero Interrupts (IE_1[31:0], IP_1[31:0]) */
template <CmdBuf CMDBUF>
inline __attribute__((always_inline)) void per_trid_idma_count_zero_interrupt_enable_cmdbuf(uint32_t trid) {
    uint64_t val;
    val = __builtin_riscv_ttrocc_cmdbuf_rd_reg(
        CMDBUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_PER_TR_ID_IE_1_REG_OFFSET / 8);
    val |= (1ULL << trid);
    __builtin_riscv_ttrocc_cmdbuf_wr_reg(
        CMDBUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_PER_TR_ID_IE_1_REG_OFFSET / 8, val);
}

template <CmdBuf CMDBUF>
inline __attribute__((always_inline)) void per_trid_idma_count_zero_interrupt_disable_cmdbuf(uint32_t trid) {
    uint64_t val;
    val = __builtin_riscv_ttrocc_cmdbuf_rd_reg(
        CMDBUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_PER_TR_ID_IE_1_REG_OFFSET / 8);
    val &= ~(1ULL << trid);
    __builtin_riscv_ttrocc_cmdbuf_wr_reg(
        CMDBUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_PER_TR_ID_IE_1_REG_OFFSET / 8, val);
}

template <CmdBuf CMDBUF>
inline __attribute__((always_inline)) uint32_t per_trid_idma_count_zero_get_interrupt_enable_cmdbuf() {
    uint64_t val;
    val = __builtin_riscv_ttrocc_cmdbuf_rd_reg(
        CMDBUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_PER_TR_ID_IE_1_REG_OFFSET / 8);
    return val & 0xFFFFFFFFULL; /* Lower 32 bits */
}

template <CmdBuf CMDBUF>
inline __attribute__((always_inline)) void per_trid_idma_count_zero_set_interrupt_enable_cmdbuf(uint32_t val) {
    uint64_t reg_val;
    reg_val = __builtin_riscv_ttrocc_cmdbuf_rd_reg(
        CMDBUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_PER_TR_ID_IE_1_REG_OFFSET / 8);
    reg_val = (reg_val & 0xFFFFFFFF00000000ULL) | (val & 0xFFFFFFFFULL);
    __builtin_riscv_ttrocc_cmdbuf_wr_reg(
        CMDBUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_PER_TR_ID_IE_1_REG_OFFSET / 8, reg_val);
}

template <CmdBuf CMDBUF>
inline __attribute__((always_inline)) uint64_t per_trid_idma_count_zero_interrupts_pending_cmdbuf() {
    uint64_t val;
    val = __builtin_riscv_ttrocc_cmdbuf_rd_reg(
        CMDBUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_PER_TR_ID_IP_1_REG_OFFSET / 8);
    return val & 0xFFFFFFFFULL;
}

template <CmdBuf CMDBUF>
inline __attribute__((always_inline)) uint32_t per_trid_idma_count_zero_get_interrupt_pending_cmdbuf() {
    uint64_t val;
    val = __builtin_riscv_ttrocc_cmdbuf_rd_reg(
        CMDBUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_PER_TR_ID_IP_1_REG_OFFSET / 8);
    return val & 0xFFFFFFFFULL; /* Lower 32 bits */
}

template <CmdBuf CMDBUF>
inline __attribute__((always_inline)) void per_trid_idma_count_zero_set_interrupt_pending_cmdbuf(uint32_t val) {
    uint64_t reg_val;
    reg_val = __builtin_riscv_ttrocc_cmdbuf_rd_reg(
        CMDBUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_PER_TR_ID_IP_1_REG_OFFSET / 8);
    reg_val = (reg_val & 0xFFFFFFFF00000000ULL) | (val & 0xFFFFFFFFULL);
    __builtin_riscv_ttrocc_cmdbuf_wr_reg(
        CMDBUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_PER_TR_ID_IP_1_REG_OFFSET / 8, reg_val);
}

template <CmdBuf CMDBUF>
inline __attribute__((always_inline)) void per_trid_idma_count_zero_interrupt_clear_cmdbuf(uint32_t trid) {
    uint64_t val;
    val = ~(1ULL << trid);
    __builtin_riscv_ttrocc_cmdbuf_wr_reg(
        CMDBUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_PER_TR_ID_IP_1_REG_OFFSET / 8, val);
}

/* Per-TRID Tiles-to-Process TR_ACK Threshold Interrupts (IE_1[63:32], IP_1[63:32]) */
template <CmdBuf CMDBUF>
inline __attribute__((always_inline)) void per_trid_tiles_to_process_interrupt_enable_cmdbuf(uint32_t trid) {
    uint64_t val;
    val = __builtin_riscv_ttrocc_cmdbuf_rd_reg(
        CMDBUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_PER_TR_ID_IE_1_REG_OFFSET / 8);
    val |= (1ULL << (trid + 32));
    __builtin_riscv_ttrocc_cmdbuf_wr_reg(
        CMDBUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_PER_TR_ID_IE_1_REG_OFFSET / 8, val);
}

template <CmdBuf CMDBUF>
inline __attribute__((always_inline)) void per_trid_tiles_to_process_interrupt_disable_cmdbuf(uint32_t trid) {
    uint64_t val;
    val = __builtin_riscv_ttrocc_cmdbuf_rd_reg(
        CMDBUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_PER_TR_ID_IE_1_REG_OFFSET / 8);
    val &= ~(1ULL << (trid + 32));
    __builtin_riscv_ttrocc_cmdbuf_wr_reg(
        CMDBUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_PER_TR_ID_IE_1_REG_OFFSET / 8, val);
}

template <CmdBuf CMDBUF>
inline __attribute__((always_inline)) uint32_t per_trid_tiles_to_process_get_interrupt_enable_cmdbuf() {
    uint64_t val;
    val = __builtin_riscv_ttrocc_cmdbuf_rd_reg(
        CMDBUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_PER_TR_ID_IE_1_REG_OFFSET / 8);
    return (val >> 32) & 0xFFFFFFFFULL; /* Upper 32 bits */
}

template <CmdBuf CMDBUF>
inline __attribute__((always_inline)) void per_trid_tiles_to_process_set_interrupt_enable_cmdbuf(uint32_t val) {
    uint64_t reg_val;
    reg_val = __builtin_riscv_ttrocc_cmdbuf_rd_reg(
        CMDBUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_PER_TR_ID_IE_1_REG_OFFSET / 8);
    reg_val = (reg_val & 0x00000000FFFFFFFFULL) | ((uint64_t)(val & 0xFFFFFFFFULL) << 32);
    __builtin_riscv_ttrocc_cmdbuf_wr_reg(
        CMDBUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_PER_TR_ID_IE_1_REG_OFFSET / 8, reg_val);
}

template <CmdBuf CMDBUF>
inline __attribute__((always_inline)) uint64_t per_trid_tiles_to_process_interrupts_pending_cmdbuf() {
    uint64_t val;
    val = __builtin_riscv_ttrocc_cmdbuf_rd_reg(
        CMDBUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_PER_TR_ID_IP_1_REG_OFFSET / 8);
    return (val >> 32) & 0xFFFFFFFFULL;
}

template <CmdBuf CMDBUF>
inline __attribute__((always_inline)) uint32_t per_trid_tiles_to_process_get_interrupt_pending_cmdbuf() {
    uint64_t val;
    val = __builtin_riscv_ttrocc_cmdbuf_rd_reg(
        CMDBUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_PER_TR_ID_IP_1_REG_OFFSET / 8);
    return (val >> 32) & 0xFFFFFFFFULL; /* Upper 32 bits */
}

template <CmdBuf CMDBUF>
inline __attribute__((always_inline)) void per_trid_tiles_to_process_set_interrupt_pending_cmdbuf(uint32_t val) {
    uint64_t reg_val;
    reg_val = __builtin_riscv_ttrocc_cmdbuf_rd_reg(
        CMDBUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_PER_TR_ID_IP_1_REG_OFFSET / 8);
    reg_val = (reg_val & 0x00000000FFFFFFFFULL) | ((uint64_t)(val & 0xFFFFFFFFULL) << 32);
    __builtin_riscv_ttrocc_cmdbuf_wr_reg(
        CMDBUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_PER_TR_ID_IP_1_REG_OFFSET / 8, reg_val);
}

template <CmdBuf CMDBUF>
inline __attribute__((always_inline)) void per_trid_tiles_to_process_interrupt_clear_cmdbuf(uint32_t trid) {
    uint64_t val;
    val = ~(1ULL << (trid + 32));
    __builtin_riscv_ttrocc_cmdbuf_wr_reg(
        CMDBUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_PER_TR_ID_IP_1_REG_OFFSET / 8, val);
}

/* Per-TRID Write Tiles-to-Process WR_SENT Threshold Interrupts (IE_2[31:0], IP_2[31:0]) */
template <CmdBuf CMDBUF>
inline __attribute__((always_inline)) void per_trid_wr_tiles_to_process_interrupt_enable_cmdbuf(uint32_t trid) {
    uint64_t val;
    val = __builtin_riscv_ttrocc_cmdbuf_rd_reg(
        CMDBUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_PER_TR_ID_IE_2_REG_OFFSET / 8);
    val |= (1ULL << trid);
    __builtin_riscv_ttrocc_cmdbuf_wr_reg(
        CMDBUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_PER_TR_ID_IE_2_REG_OFFSET / 8, val);
}

template <CmdBuf CMDBUF>
inline __attribute__((always_inline)) void per_trid_wr_tiles_to_process_interrupt_disable_cmdbuf(uint32_t trid) {
    uint64_t val;
    val = __builtin_riscv_ttrocc_cmdbuf_rd_reg(
        CMDBUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_PER_TR_ID_IE_2_REG_OFFSET / 8);
    val &= ~(1ULL << trid);
    __builtin_riscv_ttrocc_cmdbuf_wr_reg(
        CMDBUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_PER_TR_ID_IE_2_REG_OFFSET / 8, val);
}

template <CmdBuf CMDBUF>
inline __attribute__((always_inline)) uint32_t per_trid_wr_tiles_to_process_get_interrupt_enable_cmdbuf() {
    uint64_t val;
    val = __builtin_riscv_ttrocc_cmdbuf_rd_reg(
        CMDBUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_PER_TR_ID_IE_2_REG_OFFSET / 8);
    return val & 0xFFFFFFFFULL; /* Lower 32 bits */
}

template <CmdBuf CMDBUF>
inline __attribute__((always_inline)) void per_trid_wr_tiles_to_process_set_interrupt_enable_cmdbuf(uint32_t val) {
    uint64_t reg_val;
    reg_val = __builtin_riscv_ttrocc_cmdbuf_rd_reg(
        CMDBUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_PER_TR_ID_IE_2_REG_OFFSET / 8);
    reg_val = (reg_val & 0xFFFFFFFF00000000ULL) | (val & 0xFFFFFFFFULL);
    __builtin_riscv_ttrocc_cmdbuf_wr_reg(
        CMDBUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_PER_TR_ID_IE_2_REG_OFFSET / 8, reg_val);
}

template <CmdBuf CMDBUF>
inline __attribute__((always_inline)) uint64_t per_trid_wr_tiles_to_process_interrupts_pending_cmdbuf() {
    uint64_t val;
    val = __builtin_riscv_ttrocc_cmdbuf_rd_reg(
        CMDBUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_PER_TR_ID_IP_2_REG_OFFSET / 8);
    return val & 0xFFFFFFFFULL;
}

template <CmdBuf CMDBUF>
inline __attribute__((always_inline)) uint32_t per_trid_wr_tiles_to_process_get_interrupt_pending_cmdbuf() {
    uint64_t val;
    val = __builtin_riscv_ttrocc_cmdbuf_rd_reg(
        CMDBUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_PER_TR_ID_IP_2_REG_OFFSET / 8);
    return val & 0xFFFFFFFFULL; /* Lower 32 bits */
}

template <CmdBuf CMDBUF>
inline __attribute__((always_inline)) void per_trid_wr_tiles_to_process_set_interrupt_pending_cmdbuf(uint32_t val) {
    uint64_t reg_val;
    reg_val = __builtin_riscv_ttrocc_cmdbuf_rd_reg(
        CMDBUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_PER_TR_ID_IP_2_REG_OFFSET / 8);
    reg_val = (reg_val & 0xFFFFFFFF00000000ULL) | (val & 0xFFFFFFFFULL);
    __builtin_riscv_ttrocc_cmdbuf_wr_reg(
        CMDBUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_PER_TR_ID_IP_2_REG_OFFSET / 8, reg_val);
}

template <CmdBuf CMDBUF>
inline __attribute__((always_inline)) void per_trid_wr_tiles_to_process_interrupt_clear_cmdbuf(uint32_t trid) {
    uint64_t val;
    val = ~(1ULL << trid);
    __builtin_riscv_ttrocc_cmdbuf_wr_reg(
        CMDBUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_PER_TR_ID_IP_2_REG_OFFSET / 8, val);
}

/* Per-TRID iDMA Tiles-to-Process IDMA_TR_ACK Threshold Interrupts (IE_2[63:32], IP_2[63:32]) */
template <CmdBuf CMDBUF>
inline __attribute__((always_inline)) void per_trid_idma_tiles_to_process_interrupt_enable_cmdbuf(uint32_t trid) {
    uint64_t val;
    val = __builtin_riscv_ttrocc_cmdbuf_rd_reg(
        CMDBUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_PER_TR_ID_IE_2_REG_OFFSET / 8);
    val |= (1ULL << (trid + 32));
    __builtin_riscv_ttrocc_cmdbuf_wr_reg(
        CMDBUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_PER_TR_ID_IE_2_REG_OFFSET / 8, val);
}

template <CmdBuf CMDBUF>
inline __attribute__((always_inline)) void per_trid_idma_tiles_to_process_interrupt_disable_cmdbuf(uint32_t trid) {
    uint64_t val;
    val = __builtin_riscv_ttrocc_cmdbuf_rd_reg(
        CMDBUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_PER_TR_ID_IE_2_REG_OFFSET / 8);
    val &= ~(1ULL << (trid + 32));
    __builtin_riscv_ttrocc_cmdbuf_wr_reg(
        CMDBUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_PER_TR_ID_IE_2_REG_OFFSET / 8, val);
}

template <CmdBuf CMDBUF>
inline __attribute__((always_inline)) uint32_t per_trid_idma_tiles_to_process_get_interrupt_enable_cmdbuf() {
    uint64_t val;
    val = __builtin_riscv_ttrocc_cmdbuf_rd_reg(
        CMDBUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_PER_TR_ID_IE_2_REG_OFFSET / 8);
    return (val >> 32) & 0xFFFFFFFFULL; /* Upper 32 bits */
}

template <CmdBuf CMDBUF>
inline __attribute__((always_inline)) void per_trid_idma_tiles_to_process_set_interrupt_enable_cmdbuf(uint32_t val) {
    uint64_t reg_val;
    reg_val = __builtin_riscv_ttrocc_cmdbuf_rd_reg(
        CMDBUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_PER_TR_ID_IE_2_REG_OFFSET / 8);
    reg_val = (reg_val & 0x00000000FFFFFFFFULL) | ((uint64_t)(val & 0xFFFFFFFFULL) << 32);
    __builtin_riscv_ttrocc_cmdbuf_wr_reg(
        CMDBUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_PER_TR_ID_IE_2_REG_OFFSET / 8, reg_val);
}

template <CmdBuf CMDBUF>
inline __attribute__((always_inline)) uint64_t per_trid_idma_tiles_to_process_interrupts_pending_cmdbuf() {
    uint64_t val;
    val = __builtin_riscv_ttrocc_cmdbuf_rd_reg(
        CMDBUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_PER_TR_ID_IP_2_REG_OFFSET / 8);
    return (val >> 32) & 0xFFFFFFFFULL;
}

template <CmdBuf CMDBUF>
inline __attribute__((always_inline)) uint32_t per_trid_idma_tiles_to_process_get_interrupt_pending_cmdbuf() {
    uint64_t val;
    val = __builtin_riscv_ttrocc_cmdbuf_rd_reg(
        CMDBUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_PER_TR_ID_IP_2_REG_OFFSET / 8);
    return (val >> 32) & 0xFFFFFFFFULL; /* Upper 32 bits */
}

template <CmdBuf CMDBUF>
inline __attribute__((always_inline)) void per_trid_idma_tiles_to_process_set_interrupt_pending_cmdbuf(uint32_t val) {
    uint64_t reg_val;
    reg_val = __builtin_riscv_ttrocc_cmdbuf_rd_reg(
        CMDBUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_PER_TR_ID_IP_2_REG_OFFSET / 8);
    reg_val = (reg_val & 0x00000000FFFFFFFFULL) | ((uint64_t)(val & 0xFFFFFFFFULL) << 32);
    __builtin_riscv_ttrocc_cmdbuf_wr_reg(
        CMDBUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_PER_TR_ID_IP_2_REG_OFFSET / 8, reg_val);
}

template <CmdBuf CMDBUF>
inline __attribute__((always_inline)) void per_trid_idma_tiles_to_process_interrupt_clear_cmdbuf(uint32_t trid) {
    uint64_t val;
    val = ~(1ULL << (trid + 32));
    __builtin_riscv_ttrocc_cmdbuf_wr_reg(
        CMDBUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_PER_TR_ID_IP_2_REG_OFFSET / 8, val);
}

/* Per-VC Has Space Interrupts (IE[47:32], IP[47:32]) */
template <CmdBuf CMDBUF>
inline __attribute__((always_inline)) void per_vc_has_space_interrupt_enable_cmdbuf(uint32_t vc) {
    TT_ROCC_CMD_BUF_IE_reg_u val;
    val.val = __builtin_riscv_ttrocc_cmdbuf_rd_reg(CMDBUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_IE_REG_OFFSET / 8);
    val.val |= (1ULL << (vc + 32));
    __builtin_riscv_ttrocc_cmdbuf_wr_reg(CMDBUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_IE_REG_OFFSET / 8, val.val);
}

template <CmdBuf CMDBUF>
inline __attribute__((always_inline)) void per_vc_has_space_interrupt_disable_cmdbuf(uint32_t vc) {
    TT_ROCC_CMD_BUF_IE_reg_u val;
    val.val = __builtin_riscv_ttrocc_cmdbuf_rd_reg(CMDBUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_IE_REG_OFFSET / 8);
    val.val &= ~(1ULL << (vc + 32));
    __builtin_riscv_ttrocc_cmdbuf_wr_reg(CMDBUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_IE_REG_OFFSET / 8, val.val);
}

template <CmdBuf CMDBUF>
inline __attribute__((always_inline)) uint16_t per_vc_has_space_get_interrupt_enable_cmdbuf() {
    uint64_t val;
    val = __builtin_riscv_ttrocc_cmdbuf_rd_reg(CMDBUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_IE_REG_OFFSET / 8);
    return (val >> 32) & 0xFFFFULL; /* Bits [47:32] */
}

template <CmdBuf CMDBUF>
inline __attribute__((always_inline)) void per_vc_has_space_set_interrupt_enable_cmdbuf(uint16_t val) {
    uint64_t reg_val;
    reg_val = __builtin_riscv_ttrocc_cmdbuf_rd_reg(CMDBUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_IE_REG_OFFSET / 8);
    reg_val = (reg_val & 0xFFFF0000FFFFFFFFULL) | (((uint64_t)(val & 0xFFFFULL)) << 32);
    __builtin_riscv_ttrocc_cmdbuf_wr_reg(CMDBUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_IE_REG_OFFSET / 8, reg_val);
}

template <CmdBuf CMDBUF>
inline __attribute__((always_inline)) uint64_t per_vc_has_space_interrupts_pending_cmdbuf() {
    TT_ROCC_CMD_BUF_IP_reg_u val;
    val.val = __builtin_riscv_ttrocc_cmdbuf_rd_reg(CMDBUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_IP_REG_OFFSET / 8);
    return (val.val >> 32) & 0xFFFFULL; /* Bits [47:32] shifted to [15:0] */
}

template <CmdBuf CMDBUF>
inline __attribute__((always_inline)) uint16_t per_vc_has_space_get_interrupt_pending_cmdbuf() {
    uint64_t val;
    val = __builtin_riscv_ttrocc_cmdbuf_rd_reg(CMDBUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_IP_REG_OFFSET / 8);
    return (val >> 32) & 0xFFFFULL; /* Bits [47:32] */
}

template <CmdBuf CMDBUF>
inline __attribute__((always_inline)) void per_vc_has_space_set_interrupt_pending_cmdbuf(uint16_t val) {
    uint64_t reg_val;
    reg_val = __builtin_riscv_ttrocc_cmdbuf_rd_reg(CMDBUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_IP_REG_OFFSET / 8);
    reg_val = (reg_val & 0xFFFF0000FFFFFFFFULL) | (((uint64_t)(val & 0xFFFFULL)) << 32);
    __builtin_riscv_ttrocc_cmdbuf_wr_reg(CMDBUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_IP_REG_OFFSET / 8, reg_val);
}

template <CmdBuf CMDBUF>
inline __attribute__((always_inline)) void per_vc_has_space_interrupt_clear_cmdbuf(uint32_t vc) {
    TT_ROCC_CMD_BUF_IP_reg_u val;
    val.val = ~(1ULL << (vc + 32));
    __builtin_riscv_ttrocc_cmdbuf_wr_reg(CMDBUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_IP_REG_OFFSET / 8, val.val);
}

/* Per-iDMA-VC Has Space Interrupts (IE[63:48], IP[63:48]) */
template <CmdBuf CMDBUF>
inline __attribute__((always_inline)) void per_idma_vc_has_space_interrupt_enable_cmdbuf(uint32_t vc) {
    TT_ROCC_CMD_BUF_IE_reg_u val;
    val.val = __builtin_riscv_ttrocc_cmdbuf_rd_reg(CMDBUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_IE_REG_OFFSET / 8);
    val.val |= (1ULL << (vc + 48));
    __builtin_riscv_ttrocc_cmdbuf_wr_reg(CMDBUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_IE_REG_OFFSET / 8, val.val);
}

template <CmdBuf CMDBUF>
inline __attribute__((always_inline)) void per_idma_vc_has_space_interrupt_disable_cmdbuf(uint32_t vc) {
    TT_ROCC_CMD_BUF_IE_reg_u val;
    val.val = __builtin_riscv_ttrocc_cmdbuf_rd_reg(CMDBUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_IE_REG_OFFSET / 8);
    val.val &= ~(1ULL << (vc + 48));
    __builtin_riscv_ttrocc_cmdbuf_wr_reg(CMDBUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_IE_REG_OFFSET / 8, val.val);
}

template <CmdBuf CMDBUF>
inline __attribute__((always_inline)) uint16_t per_idma_vc_has_space_get_interrupt_enable_cmdbuf() {
    uint64_t val;
    val = __builtin_riscv_ttrocc_cmdbuf_rd_reg(CMDBUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_IE_REG_OFFSET / 8);
    return (val >> 48) & 0xFFFFULL; /* Bits [63:48] */
}

template <CmdBuf CMDBUF>
inline __attribute__((always_inline)) void per_idma_vc_has_space_set_interrupt_enable_cmdbuf(uint16_t val) {
    uint64_t reg_val;
    reg_val = __builtin_riscv_ttrocc_cmdbuf_rd_reg(CMDBUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_IE_REG_OFFSET / 8);
    reg_val = (reg_val & 0x0000FFFFFFFFFFFFULL) | (((uint64_t)(val & 0xFFFFULL)) << 48);
    __builtin_riscv_ttrocc_cmdbuf_wr_reg(CMDBUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_IE_REG_OFFSET / 8, reg_val);
}

template <CmdBuf CMDBUF>
inline __attribute__((always_inline)) uint64_t per_idma_vc_has_space_interrupts_pending_cmdbuf() {
    TT_ROCC_CMD_BUF_IP_reg_u val;
    val.val = __builtin_riscv_ttrocc_cmdbuf_rd_reg(CMDBUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_IP_REG_OFFSET / 8);
    return (val.val >> 48) & 0xFFFFULL; /* Bits [63:48] shifted to [15:0] */
}

template <CmdBuf CMDBUF>
inline __attribute__((always_inline)) uint16_t per_idma_vc_has_space_get_interrupt_pending_cmdbuf() {
    uint64_t val;
    val = __builtin_riscv_ttrocc_cmdbuf_rd_reg(CMDBUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_IP_REG_OFFSET / 8);
    return (val >> 48) & 0xFFFFULL; /* Bits [63:48] */
}

template <CmdBuf CMDBUF>
inline __attribute__((always_inline)) void per_idma_vc_has_space_set_interrupt_pending_cmdbuf(uint16_t val) {
    uint64_t reg_val;
    reg_val = __builtin_riscv_ttrocc_cmdbuf_rd_reg(CMDBUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_IP_REG_OFFSET / 8);
    reg_val = (reg_val & 0x0000FFFFFFFFFFFFULL) | (((uint64_t)(val & 0xFFFFULL)) << 48);
    __builtin_riscv_ttrocc_cmdbuf_wr_reg(CMDBUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_IP_REG_OFFSET / 8, reg_val);
}

template <CmdBuf CMDBUF>
inline __attribute__((always_inline)) void per_idma_vc_has_space_interrupt_clear_cmdbuf(uint32_t vc) {
    TT_ROCC_CMD_BUF_IP_reg_u val;
    val.val = ~(1ULL << (vc + 48));
    __builtin_riscv_ttrocc_cmdbuf_wr_reg(CMDBUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_IP_REG_OFFSET / 8, val.val);
}

/*
 * @fn noc_fast_read_cmdbuf<CMDBUF>
 *
 * @brief Standalone NOC read issued with a single instruction, bypassing the command buffer programming
 *
 * @param src_addr Remote source address
 * @param dest_addr Local L1 destination address (32 bits)
 * @param len_bytes Length in bytes
 * @param has_xy Source address contains NOC coordinates (NOC_XY_COORD)
 * @param tags Packet tags
 *
 * @note Reads are always non-posted
 */
template <CmdBuf CMDBUF>
inline __attribute__((always_inline)) void noc_fast_read_cmdbuf(
    uint64_t src_addr, uint32_t dest_addr, uint64_t len_bytes, bool has_xy = true, PacketTags tags = {}) {
    uint64_t rs1 = (len_bytes << 32) | dest_addr;
    uint64_t rs2 = src_addr | detail::fast_issue_flags(has_xy, /*posted=*/false, tags);
    __builtin_riscv_ttrocc_cmdbuf_issue_read2_trans(CMDBUF, rs1, rs2);
}

/*
 * @fn noc_fast_write_cmdbuf<CMDBUF>
 *
 * @brief Standalone NOC write issued with a single instruction, bypassing the command buffer programming
 * (reset the command buffer before use)
 *
 * @param src_addr Local L1 source address (32 bits)
 * @param dest_addr Remote destination address
 * @param len_bytes Length in bytes
 * @param has_xy Destination address contains NOC coordinates (NOC_XY_COORD)
 * @param posted Posted write: no ack is returned
 * @param tags Packet tags
 */
template <CmdBuf CMDBUF>
inline __attribute__((always_inline)) void noc_fast_write_cmdbuf(
    uint32_t src_addr,
    uint64_t dest_addr,
    uint64_t len_bytes,
    bool has_xy = true,
    bool posted = true,
    PacketTags tags = {}) {
    uint64_t rs1 = (len_bytes << 32) | src_addr;
    uint64_t rs2 = dest_addr | detail::fast_issue_flags(has_xy, posted, tags);
    __builtin_riscv_ttrocc_cmdbuf_issue_write2_trans(CMDBUF, rs1, rs2);
}

/*
 * @fn noc_read_prep_cmdbuf<CMDBUF>
 *
 * @brief Programs a complete NOC read without issuing it (see issue_read_cmdbuf)
 *
 * @param src_coord Source coordinate packed with the NOC_XY_COORD macro
 * @param src_addr Remote source address
 * @param dest_coord Destination coordinate packed with the NOC_XY_COORD macro
 * @param dest_addr Local L1 destination address
 * @param len_bytes Length in bytes
 * @param opts Transaction ID and packet tags
 */
template <CmdBuf CMDBUF>
inline __attribute__((always_inline)) void noc_read_prep_cmdbuf(
    uint64_t src_coord,
    uint64_t src_addr,
    uint64_t dest_coord,
    uint64_t dest_addr,
    uint64_t len_bytes,
    const NocReadOptions& opts = {}) {
    reset_cmdbuf<CMDBUF>();
    setup_as_copy_cmdbuf<CMDBUF>({.write = false});
    setup_ongoing_cmdbuf<CMDBUF>({});
    setup_vcs_cmdbuf<CMDBUF>(NocVcs::READ);
    setup_trids_cmdbuf<CMDBUF>(opts.trid);
    if (opts.tags.snoop || opts.tags.flush) {
        setup_packet_tags_cmdbuf<CMDBUF>(opts.tags);
    }
    set_src_cmdbuf<CMDBUF>(src_addr, src_coord);
    set_dest_cmdbuf<CMDBUF>(dest_addr, dest_coord);
    set_len_cmdbuf<CMDBUF>(len_bytes);
}

/*
 * @fn noc_read_cmdbuf<CMDBUF>
 *
 * @brief Programs and issues a complete NOC read (see noc_read_prep_cmdbuf for the parameters)
 */
template <CmdBuf CMDBUF>
inline __attribute__((always_inline)) void noc_read_cmdbuf(
    uint64_t src_coord,
    uint64_t src_addr,
    uint64_t dest_coord,
    uint64_t dest_addr,
    uint64_t len_bytes,
    const NocReadOptions& opts = {}) {
    noc_read_prep_cmdbuf<CMDBUF>(src_coord, src_addr, dest_coord, dest_addr, len_bytes, opts);
    issue_read_cmdbuf<CMDBUF>();
}

/*
 * @fn noc_write_prep_cmdbuf<CMDBUF>
 *
 * @brief Programs a complete NOC write without issuing it (see issue_write_cmdbuf)
 *
 * @param src_coord Source coordinate packed with the NOC_XY_COORD macro
 * @param src_addr Local L1 source address
 * @param dest_coord Destination coordinate (multicast rectangle for multicast) packed with NOC_XY_COORD
 * @param dest_addr Remote destination address
 * @param len_bytes Length in bytes
 * @param opts Transaction ID, posted/multicast/linked options and packet tags
 */
template <CmdBuf CMDBUF>
inline __attribute__((always_inline)) void noc_write_prep_cmdbuf(
    uint64_t src_coord,
    uint64_t src_addr,
    uint64_t dest_coord,
    uint64_t dest_addr,
    uint64_t len_bytes,
    const NocWriteOptions& opts = {}) {
    reset_cmdbuf<CMDBUF>();
    setup_as_copy_cmdbuf<CMDBUF>(
        {.write = true,
         .posted = opts.posted,
         .mcast = opts.mcast,
         .linked = opts.linked,
         .src_include = opts.src_include,
         .mcast_exclude = opts.mcast_exclude});
    setup_ongoing_cmdbuf<CMDBUF>({});
    setup_vcs_cmdbuf<CMDBUF>(opts.mcast ? NocVcs::MCAST_WRITE : NocVcs::WRITE);
    setup_trids_cmdbuf<CMDBUF>(opts.trid);
    if (opts.tags.snoop || opts.tags.flush) {
        setup_packet_tags_cmdbuf<CMDBUF>(opts.tags);
    }
    set_src_cmdbuf<CMDBUF>(src_addr, src_coord);
    set_dest_cmdbuf<CMDBUF>(dest_addr, dest_coord);
    set_len_cmdbuf<CMDBUF>(len_bytes);
}

/*
 * @fn noc_write_cmdbuf<CMDBUF>
 *
 * @brief Programs and issues a complete NOC write (see noc_write_prep_cmdbuf for the parameters)
 */
template <CmdBuf CMDBUF>
inline __attribute__((always_inline)) void noc_write_cmdbuf(
    uint64_t src_coord,
    uint64_t src_addr,
    uint64_t dest_coord,
    uint64_t dest_addr,
    uint64_t len_bytes,
    const NocWriteOptions& opts = {}) {
    noc_write_prep_cmdbuf<CMDBUF>(src_coord, src_addr, dest_coord, dest_addr, len_bytes, opts);
    issue_write_cmdbuf<CMDBUF>();
}

/*
 * @fn idma_copy_cmdbuf<CMDBUF>
 *
 * @brief Programs and issues a complete iDMA copy
 *
 * @param src_addr Source address
 * @param dest_addr Destination address
 * @param len_bytes Length in bytes
 * @param trid Transaction ID
 */
template <CmdBuf CMDBUF>
inline __attribute__((always_inline)) void idma_copy_cmdbuf(
    uint64_t src_addr, uint64_t dest_addr, uint64_t len_bytes, uint32_t trid = CMDBUF_DEF_TRID) {
    reset_cmdbuf<CMDBUF>();
    idma_setup_as_copy_cmdbuf<CMDBUF>();
    setup_ongoing_cmdbuf<CMDBUF>({});
    setup_vcs_cmdbuf<CMDBUF>(NocVcs::WRITE);
    setup_trids_cmdbuf<CMDBUF>(trid);
    set_src_cmdbuf<CMDBUF>(src_addr);
    set_dest_cmdbuf<CMDBUF>(dest_addr);
    set_len_cmdbuf<CMDBUF>(len_bytes);
    issue_cmdbuf<CMDBUF>();
}

/*
 * @fn idma_l1_atomic_accum_cmdbuf<CMDBUF>
 *
 * @brief Programs and issues a complete iDMA copy that accumulates into the destination L1
 *
 * @param src_addr Source address
 * @param dest_addr Destination address
 * @param len_bytes Length in bytes
 * @param trid Transaction ID
 * @param fmt L1 atomic data format
 * @param atomic_op L1 atomic operation
 */
template <CmdBuf CMDBUF>
inline __attribute__((always_inline)) void idma_l1_atomic_accum_cmdbuf(
    uint64_t src_addr,
    uint64_t dest_addr,
    uint64_t len_bytes,
    uint32_t trid = CMDBUF_DEF_TRID,
    uint32_t fmt = 0x0,
    uint32_t atomic_op = 0x9) {
    reset_cmdbuf<CMDBUF>();
    idma_setup_as_atomic_accum_cmdbuf<CMDBUF>();
    l1_atomic_instr_cmdbuf<CMDBUF>(fmt, /*disable_saturation=*/false, atomic_op);
    setup_ongoing_cmdbuf<CMDBUF>({});
    setup_vcs_cmdbuf<CMDBUF>(NocVcs::WRITE);
    setup_trids_cmdbuf<CMDBUF>(trid);
    set_src_cmdbuf<CMDBUF>(src_addr);
    set_dest_cmdbuf<CMDBUF>(dest_addr);
    set_len_cmdbuf<CMDBUF>(len_bytes);
    issue_cmdbuf<CMDBUF>();
}

/*
 * @fn noc_atomic_increment_cmdbuf<CMDBUF>
 *
 * @brief Issues a posted NOC atomic increment
 *
 * @param coord Coordinate of the target core packed with the NOC_XY_COORD macro
 * @param addr Remote destination address
 * @param incr Increment value
 * @param wrap Wrap value
 * @param tags Packet tags
 */
template <CmdBuf CMDBUF>
inline __attribute__((always_inline)) void noc_atomic_increment_cmdbuf(
    uint64_t coord, uint64_t addr, uint32_t incr = 1, uint32_t wrap = 31, PacketTags tags = {}) {
    uint64_t at_len =
        NOC_AT_INS(NOC_AT_INS_INCR_GET) | NOC_AT_WRAP(wrap) | NOC_AT_IND_32((addr >> 2) & 0x3) | NOC_AT_IND_32_SRC(0);
    setup_as_atomic_cmdbuf<CMDBUF>();
    if (tags.snoop || tags.flush) {
        setup_packet_tags_cmdbuf<CMDBUF>(tags);
    }
    set_dest_cmdbuf<CMDBUF>(addr, coord);
    set_len_cmdbuf<CMDBUF>(at_len);
    __builtin_riscv_ttrocc_cmdbuf_wr_reg(
        CMDBUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_INLINE_DATA_REG_OFFSET / 8, static_cast<uint64_t>(incr));
    issue_cmdbuf<CMDBUF>();
}

/*
 * @fn free_space_cmdbuf<CMDBUF>
 *
 * @brief Returns the free space of a request virtual channel
 *
 * @param vc Virtual channel; the programmed request VC when omitted
 */
template <CmdBuf CMDBUF>
inline __attribute__((always_inline)) uint64_t free_space_cmdbuf(uint32_t vc) {
    return __builtin_riscv_ttrocc_cmdbuf_get_vc_space_vc(CMDBUF, vc);
}
template <CmdBuf CMDBUF>
inline __attribute__((always_inline)) uint64_t free_space_cmdbuf() {
    return __builtin_riscv_ttrocc_cmdbuf_get_vc_space(CMDBUF);
}

/*
 * @fn idma_free_space_cmdbuf<CMDBUF>
 *
 * @brief Returns the free space of an iDMA request virtual channel
 *
 * @param vc Virtual channel; the programmed request VC when omitted
 */
template <CmdBuf CMDBUF>
inline __attribute__((always_inline)) uint64_t idma_free_space_cmdbuf(uint32_t vc) {
    return __builtin_riscv_ttrocc_cmdbuf_idma_get_vc_space_vc(CMDBUF, vc);
}
template <CmdBuf CMDBUF>
inline __attribute__((always_inline)) uint64_t idma_free_space_cmdbuf() {
    return __builtin_riscv_ttrocc_cmdbuf_idma_get_vc_space(CMDBUF);
}

/*
 * @fn noc_reads_acked_cmdbuf<CMDBUF>
 *
 * @return True when all reads with the given (or the programmed) transaction ID are acked
 */
template <CmdBuf CMDBUF>
inline __attribute__((always_inline)) bool noc_reads_acked_cmdbuf(uint32_t trid) {
    return __builtin_riscv_ttrocc_cmdbuf_tr_ack_trid(CMDBUF, trid) == 0;
}
template <CmdBuf CMDBUF>
inline __attribute__((always_inline)) bool noc_reads_acked_cmdbuf() {
    return __builtin_riscv_ttrocc_cmdbuf_tr_ack(CMDBUF) == 0;
}

/*
 * @fn all_noc_reads_acked_cmdbuf<CMDBUF>
 *
 * @return True when all reads of every transaction ID owned by the command buffer are acked
 */
template <CmdBuf CMDBUF>
inline __attribute__((always_inline)) bool all_noc_reads_acked_cmdbuf() {
    constexpr TridRange range = cmdbuf_trid_range(CMDBUF);
    bool all = true;
    for (uint32_t k = range.static_trid; k <= range.end; k++) {
        all = all && __builtin_riscv_ttrocc_cmdbuf_tr_ack_trid(CMDBUF, k) == 0;
    }
    return all;
}

/*
 * @fn idma_acked_cmdbuf<CMDBUF>
 *
 * @return True when all iDMA transfers with the given (or the programmed) transaction ID are acked
 */
template <CmdBuf CMDBUF>
inline __attribute__((always_inline)) bool idma_acked_cmdbuf(uint32_t trid) {
    return __builtin_riscv_ttrocc_cmdbuf_idma_tr_ack_trid(CMDBUF, trid) == 0;
}
template <CmdBuf CMDBUF>
inline __attribute__((always_inline)) bool idma_acked_cmdbuf() {
    return __builtin_riscv_ttrocc_cmdbuf_idma_tr_ack(CMDBUF) == 0;
}

/*
 * @fn all_idma_acked_cmdbuf<CMDBUF>
 *
 * @return True when all iDMA transfers of every transaction ID owned by the command buffer are acked
 */
template <CmdBuf CMDBUF>
inline __attribute__((always_inline)) bool all_idma_acked_cmdbuf() {
    constexpr TridRange range = cmdbuf_trid_range(CMDBUF);
    bool all = true;
    for (uint32_t k = range.static_trid; k <= range.end; k++) {
        all = all && __builtin_riscv_ttrocc_cmdbuf_idma_tr_ack_trid(CMDBUF, k) == 0;
    }
    return all;
}

/*
 * @fn noc_writes_sent_cmdbuf<CMDBUF>
 *
 * @return True when all writes with the given (or the programmed) transaction ID have left the core
 */
template <CmdBuf CMDBUF>
inline __attribute__((always_inline)) bool noc_writes_sent_cmdbuf(uint32_t trid) {
    return __builtin_riscv_ttrocc_cmdbuf_wr_sent_trid(CMDBUF, trid) == 0;
}
template <CmdBuf CMDBUF>
inline __attribute__((always_inline)) bool noc_writes_sent_cmdbuf() {
    return __builtin_riscv_ttrocc_cmdbuf_wr_sent(CMDBUF) == 0;
}

/*
 * @fn all_noc_writes_sent_cmdbuf<CMDBUF>
 *
 * @return True when the writes of every transaction ID owned by the command buffer have left the core
 */
template <CmdBuf CMDBUF>
inline __attribute__((always_inline)) bool all_noc_writes_sent_cmdbuf() {
    constexpr TridRange range = cmdbuf_trid_range(CMDBUF);
    bool all = true;
    for (uint32_t k = range.static_trid; k <= range.end; k++) {
        all = all && __builtin_riscv_ttrocc_cmdbuf_wr_sent_trid(CMDBUF, k) == 0;
    }
    return all;
}

/*
 * @fn noc_nonposted_writes_acked_cmdbuf<CMDBUF>
 *
 * @return True when all non-posted writes with the given (or the programmed) transaction ID are acked
 */
template <CmdBuf CMDBUF>
inline __attribute__((always_inline)) bool noc_nonposted_writes_acked_cmdbuf(uint32_t trid) {
    return __builtin_riscv_ttrocc_cmdbuf_tr_ack_trid(CMDBUF, trid) == 0;
}
template <CmdBuf CMDBUF>
inline __attribute__((always_inline)) bool noc_nonposted_writes_acked_cmdbuf() {
    return __builtin_riscv_ttrocc_cmdbuf_tr_ack(CMDBUF) == 0;
}

/*
 * @fn all_noc_nonposted_writes_acked_cmdbuf<CMDBUF>
 *
 * @return True when the non-posted writes of every transaction ID owned by the command buffer are acked
 */
template <CmdBuf CMDBUF>
inline __attribute__((always_inline)) bool all_noc_nonposted_writes_acked_cmdbuf() {
    constexpr TridRange range = cmdbuf_trid_range(CMDBUF);
    bool all = true;
    for (uint32_t k = range.static_trid; k <= range.end; k++) {
        all = all && __builtin_riscv_ttrocc_cmdbuf_tr_ack_trid(CMDBUF, k) == 0;
    }
    return all;
}

/* Per-cmdbuf aliases: cmdbuf_0/_1 spellings of the CMDBUF-templated functions above. */
inline __attribute__((always_inline)) void reset_cmdbuf_0() { return reset_cmdbuf<CMDBUF_0>(); }
inline __attribute__((always_inline)) void reset_cmdbuf_1() { return reset_cmdbuf<CMDBUF_1>(); }
inline __attribute__((always_inline)) void setup_as_copy_cmdbuf_0(const NocTransferConfig& cfg) {
    return setup_as_copy_cmdbuf<CMDBUF_0>(cfg);
}
inline __attribute__((always_inline)) void setup_as_copy_cmdbuf_1(const NocTransferConfig& cfg) {
    return setup_as_copy_cmdbuf<CMDBUF_1>(cfg);
}
inline __attribute__((always_inline)) void setup_as_scatter_list_cmdbuf_0(
    const NocTransferConfig& cfg, const ScatterListConfig& scatter) {
    return setup_as_scatter_list_cmdbuf<CMDBUF_0>(cfg, scatter);
}
inline __attribute__((always_inline)) void setup_as_scatter_list_cmdbuf_1(
    const NocTransferConfig& cfg, const ScatterListConfig& scatter) {
    return setup_as_scatter_list_cmdbuf<CMDBUF_1>(cfg, scatter);
}
inline __attribute__((always_inline)) void setup_as_atomic_cmdbuf_0() { return setup_as_atomic_cmdbuf<CMDBUF_0>(); }
inline __attribute__((always_inline)) void setup_as_atomic_cmdbuf_1() { return setup_as_atomic_cmdbuf<CMDBUF_1>(); }
inline __attribute__((always_inline)) void idma_setup_as_copy_cmdbuf_0(bool wrapping = false) {
    return idma_setup_as_copy_cmdbuf<CMDBUF_0>(wrapping);
}
inline __attribute__((always_inline)) void idma_setup_as_copy_cmdbuf_1(bool wrapping = false) {
    return idma_setup_as_copy_cmdbuf<CMDBUF_1>(wrapping);
}
inline __attribute__((always_inline)) void idma_setup_as_scatter_list_cmdbuf_0(
    const ScatterListConfig& scatter, bool wrapping = false) {
    return idma_setup_as_scatter_list_cmdbuf<CMDBUF_0>(scatter, wrapping);
}
inline __attribute__((always_inline)) void idma_setup_as_scatter_list_cmdbuf_1(
    const ScatterListConfig& scatter, bool wrapping = false) {
    return idma_setup_as_scatter_list_cmdbuf<CMDBUF_1>(scatter, wrapping);
}
inline __attribute__((always_inline)) void idma_setup_as_atomic_accum_cmdbuf_0(bool wrapping = false) {
    return idma_setup_as_atomic_accum_cmdbuf<CMDBUF_0>(wrapping);
}
inline __attribute__((always_inline)) void idma_setup_as_atomic_accum_cmdbuf_1(bool wrapping = false) {
    return idma_setup_as_atomic_accum_cmdbuf<CMDBUF_1>(wrapping);
}
inline __attribute__((always_inline)) void l1_atomic_instr_cmdbuf_0(
    uint32_t fmt, bool disable_saturation, uint32_t atomic_op) {
    return l1_atomic_instr_cmdbuf<CMDBUF_0>(fmt, disable_saturation, atomic_op);
}
inline __attribute__((always_inline)) void l1_atomic_instr_cmdbuf_1(
    uint32_t fmt, bool disable_saturation, uint32_t atomic_op) {
    return l1_atomic_instr_cmdbuf<CMDBUF_1>(fmt, disable_saturation, atomic_op);
}
inline __attribute__((always_inline)) void set_axi_opt_1_cmdbuf_0(uint8_t src_protocol, uint8_t decouple_aw) {
    return set_axi_opt_1_cmdbuf<CMDBUF_0>(src_protocol, decouple_aw);
}
inline __attribute__((always_inline)) void set_axi_opt_1_cmdbuf_1(uint8_t src_protocol, uint8_t decouple_aw) {
    return set_axi_opt_1_cmdbuf<CMDBUF_1>(src_protocol, decouple_aw);
}
inline __attribute__((always_inline)) void setup_ongoing_cmdbuf_0(const AutoIncConfig& cfg) {
    return setup_ongoing_cmdbuf<CMDBUF_0>(cfg);
}
inline __attribute__((always_inline)) void setup_ongoing_cmdbuf_1(const AutoIncConfig& cfg) {
    return setup_ongoing_cmdbuf<CMDBUF_1>(cfg);
}
inline __attribute__((always_inline)) void setup_vcs_cmdbuf_0(NocVcs vcs) { return setup_vcs_cmdbuf<CMDBUF_0>(vcs); }
inline __attribute__((always_inline)) void setup_vcs_cmdbuf_1(NocVcs vcs) { return setup_vcs_cmdbuf<CMDBUF_1>(vcs); }
inline __attribute__((always_inline)) void setup_wrapping_vcs_cmdbuf_0(const VcRange& req, const VcRange& resp = {}) {
    return setup_wrapping_vcs_cmdbuf<CMDBUF_0>(req, resp);
}
inline __attribute__((always_inline)) void setup_wrapping_vcs_cmdbuf_1(const VcRange& req, const VcRange& resp = {}) {
    return setup_wrapping_vcs_cmdbuf<CMDBUF_1>(req, resp);
}
inline __attribute__((always_inline)) void setup_trids_cmdbuf_0(
    uint32_t trid = CMDBUF_DEF_TRID, bool wrapping = false) {
    return setup_trids_cmdbuf<CMDBUF_0>(trid, wrapping);
}
inline __attribute__((always_inline)) void setup_trids_cmdbuf_1(
    uint32_t trid = CMDBUF_DEF_TRID, bool wrapping = false) {
    return setup_trids_cmdbuf<CMDBUF_1>(trid, wrapping);
}
inline __attribute__((always_inline)) uint32_t swap_trid_cmdbuf_0(uint32_t new_trid) {
    return swap_trid_cmdbuf<CMDBUF_0>(new_trid);
}
inline __attribute__((always_inline)) uint32_t swap_trid_cmdbuf_1(uint32_t new_trid) {
    return swap_trid_cmdbuf<CMDBUF_1>(new_trid);
}
inline __attribute__((always_inline)) void setup_max_bytes_in_packet_cmdbuf_0(uint64_t max_bytes_in_packet) {
    return setup_max_bytes_in_packet_cmdbuf<CMDBUF_0>(max_bytes_in_packet);
}
inline __attribute__((always_inline)) void setup_max_bytes_in_packet_cmdbuf_1(uint64_t max_bytes_in_packet) {
    return setup_max_bytes_in_packet_cmdbuf<CMDBUF_1>(max_bytes_in_packet);
}
inline __attribute__((always_inline)) void setup_packet_tags_cmdbuf_0(PacketTags tags) {
    return setup_packet_tags_cmdbuf<CMDBUF_0>(tags);
}
inline __attribute__((always_inline)) void setup_packet_tags_cmdbuf_1(PacketTags tags) {
    return setup_packet_tags_cmdbuf<CMDBUF_1>(tags);
}
inline __attribute__((always_inline)) uint64_t get_src_cmdbuf_0() { return get_src_cmdbuf<CMDBUF_0>(); }
inline __attribute__((always_inline)) uint64_t get_src_cmdbuf_1() { return get_src_cmdbuf<CMDBUF_1>(); }
inline __attribute__((always_inline)) void set_src_cmdbuf_0(
    uint64_t addr, uint64_t coord, uint64_t wrap_base, uint64_t wrap_size) {
    return set_src_cmdbuf<CMDBUF_0>(addr, coord, wrap_base, wrap_size);
}
inline __attribute__((always_inline)) void set_src_cmdbuf_1(
    uint64_t addr, uint64_t coord, uint64_t wrap_base, uint64_t wrap_size) {
    return set_src_cmdbuf<CMDBUF_1>(addr, coord, wrap_base, wrap_size);
}
inline __attribute__((always_inline)) void set_src_cmdbuf_0(uint64_t addr, uint64_t coord, uint64_t wrap_base) {
    return set_src_cmdbuf<CMDBUF_0>(addr, coord, wrap_base);
}
inline __attribute__((always_inline)) void set_src_cmdbuf_1(uint64_t addr, uint64_t coord, uint64_t wrap_base) {
    return set_src_cmdbuf<CMDBUF_1>(addr, coord, wrap_base);
}
inline __attribute__((always_inline)) void set_src_cmdbuf_0(uint64_t addr, uint64_t coord) {
    return set_src_cmdbuf<CMDBUF_0>(addr, coord);
}
inline __attribute__((always_inline)) void set_src_cmdbuf_1(uint64_t addr, uint64_t coord) {
    return set_src_cmdbuf<CMDBUF_1>(addr, coord);
}
inline __attribute__((always_inline)) void set_src_cmdbuf_0(uint64_t addr) { return set_src_cmdbuf<CMDBUF_0>(addr); }
inline __attribute__((always_inline)) void set_src_cmdbuf_1(uint64_t addr) { return set_src_cmdbuf<CMDBUF_1>(addr); }
inline __attribute__((always_inline)) uint64_t get_dest_cmdbuf_0() { return get_dest_cmdbuf<CMDBUF_0>(); }
inline __attribute__((always_inline)) uint64_t get_dest_cmdbuf_1() { return get_dest_cmdbuf<CMDBUF_1>(); }
inline __attribute__((always_inline)) void set_dest_cmdbuf_0(
    uint64_t addr, uint64_t coord, uint64_t wrap_base, uint64_t wrap_size) {
    return set_dest_cmdbuf<CMDBUF_0>(addr, coord, wrap_base, wrap_size);
}
inline __attribute__((always_inline)) void set_dest_cmdbuf_1(
    uint64_t addr, uint64_t coord, uint64_t wrap_base, uint64_t wrap_size) {
    return set_dest_cmdbuf<CMDBUF_1>(addr, coord, wrap_base, wrap_size);
}
inline __attribute__((always_inline)) void set_dest_cmdbuf_0(uint64_t addr, uint64_t coord, uint64_t wrap_base) {
    return set_dest_cmdbuf<CMDBUF_0>(addr, coord, wrap_base);
}
inline __attribute__((always_inline)) void set_dest_cmdbuf_1(uint64_t addr, uint64_t coord, uint64_t wrap_base) {
    return set_dest_cmdbuf<CMDBUF_1>(addr, coord, wrap_base);
}
inline __attribute__((always_inline)) void set_dest_cmdbuf_0(uint64_t addr, uint64_t coord) {
    return set_dest_cmdbuf<CMDBUF_0>(addr, coord);
}
inline __attribute__((always_inline)) void set_dest_cmdbuf_1(uint64_t addr, uint64_t coord) {
    return set_dest_cmdbuf<CMDBUF_1>(addr, coord);
}
inline __attribute__((always_inline)) void set_dest_cmdbuf_0(uint64_t addr) { return set_dest_cmdbuf<CMDBUF_0>(addr); }
inline __attribute__((always_inline)) void set_dest_cmdbuf_1(uint64_t addr) { return set_dest_cmdbuf<CMDBUF_1>(addr); }
inline __attribute__((always_inline)) void set_scatter_list_cmdbuf_0(
    uint64_t addr, uint64_t base, uint64_t index, uint64_t times) {
    return set_scatter_list_cmdbuf<CMDBUF_0>(addr, base, index, times);
}
inline __attribute__((always_inline)) void set_scatter_list_cmdbuf_1(
    uint64_t addr, uint64_t base, uint64_t index, uint64_t times) {
    return set_scatter_list_cmdbuf<CMDBUF_1>(addr, base, index, times);
}
inline __attribute__((always_inline)) void set_scatter_list_base_cmdbuf_0(uint64_t base) {
    return set_scatter_list_base_cmdbuf<CMDBUF_0>(base);
}
inline __attribute__((always_inline)) void set_scatter_list_base_cmdbuf_1(uint64_t base) {
    return set_scatter_list_base_cmdbuf<CMDBUF_1>(base);
}
inline __attribute__((always_inline)) void set_scatter_list_index_cmdbuf_0(uint64_t index) {
    return set_scatter_list_index_cmdbuf<CMDBUF_0>(index);
}
inline __attribute__((always_inline)) void set_scatter_list_index_cmdbuf_1(uint64_t index) {
    return set_scatter_list_index_cmdbuf<CMDBUF_1>(index);
}
inline __attribute__((always_inline)) void set_scatter_list_times_cmdbuf_0(uint64_t times) {
    return set_scatter_list_times_cmdbuf<CMDBUF_0>(times);
}
inline __attribute__((always_inline)) void set_scatter_list_times_cmdbuf_1(uint64_t times) {
    return set_scatter_list_times_cmdbuf<CMDBUF_1>(times);
}
inline __attribute__((always_inline)) uint64_t get_scatter_list_addr_cmdbuf_0() {
    return get_scatter_list_addr_cmdbuf<CMDBUF_0>();
}
inline __attribute__((always_inline)) uint64_t get_scatter_list_addr_cmdbuf_1() {
    return get_scatter_list_addr_cmdbuf<CMDBUF_1>();
}
inline __attribute__((always_inline)) uint64_t get_scatter_list_base_cmdbuf_0() {
    return get_scatter_list_base_cmdbuf<CMDBUF_0>();
}
inline __attribute__((always_inline)) uint64_t get_scatter_list_base_cmdbuf_1() {
    return get_scatter_list_base_cmdbuf<CMDBUF_1>();
}
inline __attribute__((always_inline)) uint64_t get_scatter_list_index_cmdbuf_0() {
    return get_scatter_list_index_cmdbuf<CMDBUF_0>();
}
inline __attribute__((always_inline)) uint64_t get_scatter_list_index_cmdbuf_1() {
    return get_scatter_list_index_cmdbuf<CMDBUF_1>();
}
inline __attribute__((always_inline)) uint64_t get_scatter_list_times_cmdbuf_0() {
    return get_scatter_list_times_cmdbuf<CMDBUF_0>();
}
inline __attribute__((always_inline)) uint64_t get_scatter_list_times_cmdbuf_1() {
    return get_scatter_list_times_cmdbuf<CMDBUF_1>();
}
inline __attribute__((always_inline)) void set_len_cmdbuf_0(uint64_t len_bytes) {
    return set_len_cmdbuf<CMDBUF_0>(len_bytes);
}
inline __attribute__((always_inline)) void set_len_cmdbuf_1(uint64_t len_bytes) {
    return set_len_cmdbuf<CMDBUF_1>(len_bytes);
}
inline __attribute__((always_inline)) void issue_cmdbuf_0() { return issue_cmdbuf<CMDBUF_0>(); }
inline __attribute__((always_inline)) void issue_cmdbuf_1() { return issue_cmdbuf<CMDBUF_1>(); }
inline __attribute__((always_inline)) void issue_read_cmdbuf_0() { return issue_read_cmdbuf<CMDBUF_0>(); }
inline __attribute__((always_inline)) void issue_read_cmdbuf_1() { return issue_read_cmdbuf<CMDBUF_1>(); }
inline __attribute__((always_inline)) void issue_write_cmdbuf_0() { return issue_write_cmdbuf<CMDBUF_0>(); }
inline __attribute__((always_inline)) void issue_write_cmdbuf_1() { return issue_write_cmdbuf<CMDBUF_1>(); }
inline __attribute__((always_inline)) void issue_write_inline_cmdbuf_0(uint64_t data) {
    return issue_write_inline_cmdbuf<CMDBUF_0>(data);
}
inline __attribute__((always_inline)) void issue_write_inline_cmdbuf_1(uint64_t data) {
    return issue_write_inline_cmdbuf<CMDBUF_1>(data);
}
inline __attribute__((always_inline)) void issue_write_inline_len_cmdbuf_0(
    uint64_t data,
    uint64_t dest_addr,
    uint64_t len_bytes,
    bool has_xy = true,
    bool posted = true,
    PacketTags tags = {}) {
    return issue_write_inline_len_cmdbuf<CMDBUF_0>(data, dest_addr, len_bytes, has_xy, posted, tags);
}
inline __attribute__((always_inline)) void issue_write_inline_len_cmdbuf_1(
    uint64_t data,
    uint64_t dest_addr,
    uint64_t len_bytes,
    bool has_xy = true,
    bool posted = true,
    PacketTags tags = {}) {
    return issue_write_inline_len_cmdbuf<CMDBUF_1>(data, dest_addr, len_bytes, has_xy, posted, tags);
}
inline __attribute__((always_inline)) void interrupt_enable_cmdbuf_0(uint32_t id) {
    return interrupt_enable_cmdbuf<CMDBUF_0>(id);
}
inline __attribute__((always_inline)) void interrupt_enable_cmdbuf_1(uint32_t id) {
    return interrupt_enable_cmdbuf<CMDBUF_1>(id);
}
inline __attribute__((always_inline)) void interrupt_disable_cmdbuf_0(uint32_t id) {
    return interrupt_disable_cmdbuf<CMDBUF_0>(id);
}
inline __attribute__((always_inline)) void interrupt_disable_cmdbuf_1(uint32_t id) {
    return interrupt_disable_cmdbuf<CMDBUF_1>(id);
}
inline __attribute__((always_inline)) uint64_t interrupts_pending_cmdbuf_0() {
    return interrupts_pending_cmdbuf<CMDBUF_0>();
}
inline __attribute__((always_inline)) uint64_t interrupts_pending_cmdbuf_1() {
    return interrupts_pending_cmdbuf<CMDBUF_1>();
}
inline __attribute__((always_inline)) void interrupt_clear_cmdbuf_0(uint32_t id) {
    return interrupt_clear_cmdbuf<CMDBUF_0>(id);
}
inline __attribute__((always_inline)) void interrupt_clear_cmdbuf_1(uint32_t id) {
    return interrupt_clear_cmdbuf<CMDBUF_1>(id);
}
inline __attribute__((always_inline)) void per_trid_count_zero_interrupt_enable_cmdbuf_0(uint32_t trid) {
    return per_trid_count_zero_interrupt_enable_cmdbuf<CMDBUF_0>(trid);
}
inline __attribute__((always_inline)) void per_trid_count_zero_interrupt_enable_cmdbuf_1(uint32_t trid) {
    return per_trid_count_zero_interrupt_enable_cmdbuf<CMDBUF_1>(trid);
}
inline __attribute__((always_inline)) void per_trid_count_zero_interrupt_disable_cmdbuf_0(uint32_t trid) {
    return per_trid_count_zero_interrupt_disable_cmdbuf<CMDBUF_0>(trid);
}
inline __attribute__((always_inline)) void per_trid_count_zero_interrupt_disable_cmdbuf_1(uint32_t trid) {
    return per_trid_count_zero_interrupt_disable_cmdbuf<CMDBUF_1>(trid);
}
inline __attribute__((always_inline)) uint32_t per_trid_count_zero_get_interrupt_enable_cmdbuf_0() {
    return per_trid_count_zero_get_interrupt_enable_cmdbuf<CMDBUF_0>();
}
inline __attribute__((always_inline)) uint32_t per_trid_count_zero_get_interrupt_enable_cmdbuf_1() {
    return per_trid_count_zero_get_interrupt_enable_cmdbuf<CMDBUF_1>();
}
inline __attribute__((always_inline)) void per_trid_count_zero_set_interrupt_enable_cmdbuf_0(uint32_t val) {
    return per_trid_count_zero_set_interrupt_enable_cmdbuf<CMDBUF_0>(val);
}
inline __attribute__((always_inline)) void per_trid_count_zero_set_interrupt_enable_cmdbuf_1(uint32_t val) {
    return per_trid_count_zero_set_interrupt_enable_cmdbuf<CMDBUF_1>(val);
}
inline __attribute__((always_inline)) uint64_t per_trid_count_zero_interrupts_pending_cmdbuf_0() {
    return per_trid_count_zero_interrupts_pending_cmdbuf<CMDBUF_0>();
}
inline __attribute__((always_inline)) uint64_t per_trid_count_zero_interrupts_pending_cmdbuf_1() {
    return per_trid_count_zero_interrupts_pending_cmdbuf<CMDBUF_1>();
}
inline __attribute__((always_inline)) uint32_t per_trid_count_zero_get_interrupt_pending_cmdbuf_0() {
    return per_trid_count_zero_get_interrupt_pending_cmdbuf<CMDBUF_0>();
}
inline __attribute__((always_inline)) uint32_t per_trid_count_zero_get_interrupt_pending_cmdbuf_1() {
    return per_trid_count_zero_get_interrupt_pending_cmdbuf<CMDBUF_1>();
}
inline __attribute__((always_inline)) void per_trid_count_zero_set_interrupt_pending_cmdbuf_0(uint32_t val) {
    return per_trid_count_zero_set_interrupt_pending_cmdbuf<CMDBUF_0>(val);
}
inline __attribute__((always_inline)) void per_trid_count_zero_set_interrupt_pending_cmdbuf_1(uint32_t val) {
    return per_trid_count_zero_set_interrupt_pending_cmdbuf<CMDBUF_1>(val);
}
inline __attribute__((always_inline)) void per_trid_count_zero_interrupt_clear_cmdbuf_0(uint32_t trid) {
    return per_trid_count_zero_interrupt_clear_cmdbuf<CMDBUF_0>(trid);
}
inline __attribute__((always_inline)) void per_trid_count_zero_interrupt_clear_cmdbuf_1(uint32_t trid) {
    return per_trid_count_zero_interrupt_clear_cmdbuf<CMDBUF_1>(trid);
}
inline __attribute__((always_inline)) void per_trid_wr_count_zero_interrupt_enable_cmdbuf_0(uint32_t trid) {
    return per_trid_wr_count_zero_interrupt_enable_cmdbuf<CMDBUF_0>(trid);
}
inline __attribute__((always_inline)) void per_trid_wr_count_zero_interrupt_enable_cmdbuf_1(uint32_t trid) {
    return per_trid_wr_count_zero_interrupt_enable_cmdbuf<CMDBUF_1>(trid);
}
inline __attribute__((always_inline)) void per_trid_wr_count_zero_interrupt_disable_cmdbuf_0(uint32_t trid) {
    return per_trid_wr_count_zero_interrupt_disable_cmdbuf<CMDBUF_0>(trid);
}
inline __attribute__((always_inline)) void per_trid_wr_count_zero_interrupt_disable_cmdbuf_1(uint32_t trid) {
    return per_trid_wr_count_zero_interrupt_disable_cmdbuf<CMDBUF_1>(trid);
}
inline __attribute__((always_inline)) uint32_t per_trid_wr_count_zero_get_interrupt_enable_cmdbuf_0() {
    return per_trid_wr_count_zero_get_interrupt_enable_cmdbuf<CMDBUF_0>();
}
inline __attribute__((always_inline)) uint32_t per_trid_wr_count_zero_get_interrupt_enable_cmdbuf_1() {
    return per_trid_wr_count_zero_get_interrupt_enable_cmdbuf<CMDBUF_1>();
}
inline __attribute__((always_inline)) void per_trid_wr_count_zero_set_interrupt_enable_cmdbuf_0(uint32_t val) {
    return per_trid_wr_count_zero_set_interrupt_enable_cmdbuf<CMDBUF_0>(val);
}
inline __attribute__((always_inline)) void per_trid_wr_count_zero_set_interrupt_enable_cmdbuf_1(uint32_t val) {
    return per_trid_wr_count_zero_set_interrupt_enable_cmdbuf<CMDBUF_1>(val);
}
inline __attribute__((always_inline)) uint64_t per_trid_wr_count_zero_interrupts_pending_cmdbuf_0() {
    return per_trid_wr_count_zero_interrupts_pending_cmdbuf<CMDBUF_0>();
}
inline __attribute__((always_inline)) uint64_t per_trid_wr_count_zero_interrupts_pending_cmdbuf_1() {
    return per_trid_wr_count_zero_interrupts_pending_cmdbuf<CMDBUF_1>();
}
inline __attribute__((always_inline)) uint32_t per_trid_wr_count_zero_get_interrupt_pending_cmdbuf_0() {
    return per_trid_wr_count_zero_get_interrupt_pending_cmdbuf<CMDBUF_0>();
}
inline __attribute__((always_inline)) uint32_t per_trid_wr_count_zero_get_interrupt_pending_cmdbuf_1() {
    return per_trid_wr_count_zero_get_interrupt_pending_cmdbuf<CMDBUF_1>();
}
inline __attribute__((always_inline)) void per_trid_wr_count_zero_set_interrupt_pending_cmdbuf_0(uint32_t val) {
    return per_trid_wr_count_zero_set_interrupt_pending_cmdbuf<CMDBUF_0>(val);
}
inline __attribute__((always_inline)) void per_trid_wr_count_zero_set_interrupt_pending_cmdbuf_1(uint32_t val) {
    return per_trid_wr_count_zero_set_interrupt_pending_cmdbuf<CMDBUF_1>(val);
}
inline __attribute__((always_inline)) void per_trid_wr_count_zero_interrupt_clear_cmdbuf_0(uint32_t trid) {
    return per_trid_wr_count_zero_interrupt_clear_cmdbuf<CMDBUF_0>(trid);
}
inline __attribute__((always_inline)) void per_trid_wr_count_zero_interrupt_clear_cmdbuf_1(uint32_t trid) {
    return per_trid_wr_count_zero_interrupt_clear_cmdbuf<CMDBUF_1>(trid);
}
inline __attribute__((always_inline)) void per_trid_idma_count_zero_interrupt_enable_cmdbuf_0(uint32_t trid) {
    return per_trid_idma_count_zero_interrupt_enable_cmdbuf<CMDBUF_0>(trid);
}
inline __attribute__((always_inline)) void per_trid_idma_count_zero_interrupt_enable_cmdbuf_1(uint32_t trid) {
    return per_trid_idma_count_zero_interrupt_enable_cmdbuf<CMDBUF_1>(trid);
}
inline __attribute__((always_inline)) void per_trid_idma_count_zero_interrupt_disable_cmdbuf_0(uint32_t trid) {
    return per_trid_idma_count_zero_interrupt_disable_cmdbuf<CMDBUF_0>(trid);
}
inline __attribute__((always_inline)) void per_trid_idma_count_zero_interrupt_disable_cmdbuf_1(uint32_t trid) {
    return per_trid_idma_count_zero_interrupt_disable_cmdbuf<CMDBUF_1>(trid);
}
inline __attribute__((always_inline)) uint32_t per_trid_idma_count_zero_get_interrupt_enable_cmdbuf_0() {
    return per_trid_idma_count_zero_get_interrupt_enable_cmdbuf<CMDBUF_0>();
}
inline __attribute__((always_inline)) uint32_t per_trid_idma_count_zero_get_interrupt_enable_cmdbuf_1() {
    return per_trid_idma_count_zero_get_interrupt_enable_cmdbuf<CMDBUF_1>();
}
inline __attribute__((always_inline)) void per_trid_idma_count_zero_set_interrupt_enable_cmdbuf_0(uint32_t val) {
    return per_trid_idma_count_zero_set_interrupt_enable_cmdbuf<CMDBUF_0>(val);
}
inline __attribute__((always_inline)) void per_trid_idma_count_zero_set_interrupt_enable_cmdbuf_1(uint32_t val) {
    return per_trid_idma_count_zero_set_interrupt_enable_cmdbuf<CMDBUF_1>(val);
}
inline __attribute__((always_inline)) uint64_t per_trid_idma_count_zero_interrupts_pending_cmdbuf_0() {
    return per_trid_idma_count_zero_interrupts_pending_cmdbuf<CMDBUF_0>();
}
inline __attribute__((always_inline)) uint64_t per_trid_idma_count_zero_interrupts_pending_cmdbuf_1() {
    return per_trid_idma_count_zero_interrupts_pending_cmdbuf<CMDBUF_1>();
}
inline __attribute__((always_inline)) uint32_t per_trid_idma_count_zero_get_interrupt_pending_cmdbuf_0() {
    return per_trid_idma_count_zero_get_interrupt_pending_cmdbuf<CMDBUF_0>();
}
inline __attribute__((always_inline)) uint32_t per_trid_idma_count_zero_get_interrupt_pending_cmdbuf_1() {
    return per_trid_idma_count_zero_get_interrupt_pending_cmdbuf<CMDBUF_1>();
}
inline __attribute__((always_inline)) void per_trid_idma_count_zero_set_interrupt_pending_cmdbuf_0(uint32_t val) {
    return per_trid_idma_count_zero_set_interrupt_pending_cmdbuf<CMDBUF_0>(val);
}
inline __attribute__((always_inline)) void per_trid_idma_count_zero_set_interrupt_pending_cmdbuf_1(uint32_t val) {
    return per_trid_idma_count_zero_set_interrupt_pending_cmdbuf<CMDBUF_1>(val);
}
inline __attribute__((always_inline)) void per_trid_idma_count_zero_interrupt_clear_cmdbuf_0(uint32_t trid) {
    return per_trid_idma_count_zero_interrupt_clear_cmdbuf<CMDBUF_0>(trid);
}
inline __attribute__((always_inline)) void per_trid_idma_count_zero_interrupt_clear_cmdbuf_1(uint32_t trid) {
    return per_trid_idma_count_zero_interrupt_clear_cmdbuf<CMDBUF_1>(trid);
}
inline __attribute__((always_inline)) void per_trid_tiles_to_process_interrupt_enable_cmdbuf_0(uint32_t trid) {
    return per_trid_tiles_to_process_interrupt_enable_cmdbuf<CMDBUF_0>(trid);
}
inline __attribute__((always_inline)) void per_trid_tiles_to_process_interrupt_enable_cmdbuf_1(uint32_t trid) {
    return per_trid_tiles_to_process_interrupt_enable_cmdbuf<CMDBUF_1>(trid);
}
inline __attribute__((always_inline)) void per_trid_tiles_to_process_interrupt_disable_cmdbuf_0(uint32_t trid) {
    return per_trid_tiles_to_process_interrupt_disable_cmdbuf<CMDBUF_0>(trid);
}
inline __attribute__((always_inline)) void per_trid_tiles_to_process_interrupt_disable_cmdbuf_1(uint32_t trid) {
    return per_trid_tiles_to_process_interrupt_disable_cmdbuf<CMDBUF_1>(trid);
}
inline __attribute__((always_inline)) uint32_t per_trid_tiles_to_process_get_interrupt_enable_cmdbuf_0() {
    return per_trid_tiles_to_process_get_interrupt_enable_cmdbuf<CMDBUF_0>();
}
inline __attribute__((always_inline)) uint32_t per_trid_tiles_to_process_get_interrupt_enable_cmdbuf_1() {
    return per_trid_tiles_to_process_get_interrupt_enable_cmdbuf<CMDBUF_1>();
}
inline __attribute__((always_inline)) void per_trid_tiles_to_process_set_interrupt_enable_cmdbuf_0(uint32_t val) {
    return per_trid_tiles_to_process_set_interrupt_enable_cmdbuf<CMDBUF_0>(val);
}
inline __attribute__((always_inline)) void per_trid_tiles_to_process_set_interrupt_enable_cmdbuf_1(uint32_t val) {
    return per_trid_tiles_to_process_set_interrupt_enable_cmdbuf<CMDBUF_1>(val);
}
inline __attribute__((always_inline)) uint64_t per_trid_tiles_to_process_interrupts_pending_cmdbuf_0() {
    return per_trid_tiles_to_process_interrupts_pending_cmdbuf<CMDBUF_0>();
}
inline __attribute__((always_inline)) uint64_t per_trid_tiles_to_process_interrupts_pending_cmdbuf_1() {
    return per_trid_tiles_to_process_interrupts_pending_cmdbuf<CMDBUF_1>();
}
inline __attribute__((always_inline)) uint32_t per_trid_tiles_to_process_get_interrupt_pending_cmdbuf_0() {
    return per_trid_tiles_to_process_get_interrupt_pending_cmdbuf<CMDBUF_0>();
}
inline __attribute__((always_inline)) uint32_t per_trid_tiles_to_process_get_interrupt_pending_cmdbuf_1() {
    return per_trid_tiles_to_process_get_interrupt_pending_cmdbuf<CMDBUF_1>();
}
inline __attribute__((always_inline)) void per_trid_tiles_to_process_set_interrupt_pending_cmdbuf_0(uint32_t val) {
    return per_trid_tiles_to_process_set_interrupt_pending_cmdbuf<CMDBUF_0>(val);
}
inline __attribute__((always_inline)) void per_trid_tiles_to_process_set_interrupt_pending_cmdbuf_1(uint32_t val) {
    return per_trid_tiles_to_process_set_interrupt_pending_cmdbuf<CMDBUF_1>(val);
}
inline __attribute__((always_inline)) void per_trid_tiles_to_process_interrupt_clear_cmdbuf_0(uint32_t trid) {
    return per_trid_tiles_to_process_interrupt_clear_cmdbuf<CMDBUF_0>(trid);
}
inline __attribute__((always_inline)) void per_trid_tiles_to_process_interrupt_clear_cmdbuf_1(uint32_t trid) {
    return per_trid_tiles_to_process_interrupt_clear_cmdbuf<CMDBUF_1>(trid);
}
inline __attribute__((always_inline)) void per_trid_wr_tiles_to_process_interrupt_enable_cmdbuf_0(uint32_t trid) {
    return per_trid_wr_tiles_to_process_interrupt_enable_cmdbuf<CMDBUF_0>(trid);
}
inline __attribute__((always_inline)) void per_trid_wr_tiles_to_process_interrupt_enable_cmdbuf_1(uint32_t trid) {
    return per_trid_wr_tiles_to_process_interrupt_enable_cmdbuf<CMDBUF_1>(trid);
}
inline __attribute__((always_inline)) void per_trid_wr_tiles_to_process_interrupt_disable_cmdbuf_0(uint32_t trid) {
    return per_trid_wr_tiles_to_process_interrupt_disable_cmdbuf<CMDBUF_0>(trid);
}
inline __attribute__((always_inline)) void per_trid_wr_tiles_to_process_interrupt_disable_cmdbuf_1(uint32_t trid) {
    return per_trid_wr_tiles_to_process_interrupt_disable_cmdbuf<CMDBUF_1>(trid);
}
inline __attribute__((always_inline)) uint32_t per_trid_wr_tiles_to_process_get_interrupt_enable_cmdbuf_0() {
    return per_trid_wr_tiles_to_process_get_interrupt_enable_cmdbuf<CMDBUF_0>();
}
inline __attribute__((always_inline)) uint32_t per_trid_wr_tiles_to_process_get_interrupt_enable_cmdbuf_1() {
    return per_trid_wr_tiles_to_process_get_interrupt_enable_cmdbuf<CMDBUF_1>();
}
inline __attribute__((always_inline)) void per_trid_wr_tiles_to_process_set_interrupt_enable_cmdbuf_0(uint32_t val) {
    return per_trid_wr_tiles_to_process_set_interrupt_enable_cmdbuf<CMDBUF_0>(val);
}
inline __attribute__((always_inline)) void per_trid_wr_tiles_to_process_set_interrupt_enable_cmdbuf_1(uint32_t val) {
    return per_trid_wr_tiles_to_process_set_interrupt_enable_cmdbuf<CMDBUF_1>(val);
}
inline __attribute__((always_inline)) uint64_t per_trid_wr_tiles_to_process_interrupts_pending_cmdbuf_0() {
    return per_trid_wr_tiles_to_process_interrupts_pending_cmdbuf<CMDBUF_0>();
}
inline __attribute__((always_inline)) uint64_t per_trid_wr_tiles_to_process_interrupts_pending_cmdbuf_1() {
    return per_trid_wr_tiles_to_process_interrupts_pending_cmdbuf<CMDBUF_1>();
}
inline __attribute__((always_inline)) uint32_t per_trid_wr_tiles_to_process_get_interrupt_pending_cmdbuf_0() {
    return per_trid_wr_tiles_to_process_get_interrupt_pending_cmdbuf<CMDBUF_0>();
}
inline __attribute__((always_inline)) uint32_t per_trid_wr_tiles_to_process_get_interrupt_pending_cmdbuf_1() {
    return per_trid_wr_tiles_to_process_get_interrupt_pending_cmdbuf<CMDBUF_1>();
}
inline __attribute__((always_inline)) void per_trid_wr_tiles_to_process_set_interrupt_pending_cmdbuf_0(uint32_t val) {
    return per_trid_wr_tiles_to_process_set_interrupt_pending_cmdbuf<CMDBUF_0>(val);
}
inline __attribute__((always_inline)) void per_trid_wr_tiles_to_process_set_interrupt_pending_cmdbuf_1(uint32_t val) {
    return per_trid_wr_tiles_to_process_set_interrupt_pending_cmdbuf<CMDBUF_1>(val);
}
inline __attribute__((always_inline)) void per_trid_wr_tiles_to_process_interrupt_clear_cmdbuf_0(uint32_t trid) {
    return per_trid_wr_tiles_to_process_interrupt_clear_cmdbuf<CMDBUF_0>(trid);
}
inline __attribute__((always_inline)) void per_trid_wr_tiles_to_process_interrupt_clear_cmdbuf_1(uint32_t trid) {
    return per_trid_wr_tiles_to_process_interrupt_clear_cmdbuf<CMDBUF_1>(trid);
}
inline __attribute__((always_inline)) void per_trid_idma_tiles_to_process_interrupt_enable_cmdbuf_0(uint32_t trid) {
    return per_trid_idma_tiles_to_process_interrupt_enable_cmdbuf<CMDBUF_0>(trid);
}
inline __attribute__((always_inline)) void per_trid_idma_tiles_to_process_interrupt_enable_cmdbuf_1(uint32_t trid) {
    return per_trid_idma_tiles_to_process_interrupt_enable_cmdbuf<CMDBUF_1>(trid);
}
inline __attribute__((always_inline)) void per_trid_idma_tiles_to_process_interrupt_disable_cmdbuf_0(uint32_t trid) {
    return per_trid_idma_tiles_to_process_interrupt_disable_cmdbuf<CMDBUF_0>(trid);
}
inline __attribute__((always_inline)) void per_trid_idma_tiles_to_process_interrupt_disable_cmdbuf_1(uint32_t trid) {
    return per_trid_idma_tiles_to_process_interrupt_disable_cmdbuf<CMDBUF_1>(trid);
}
inline __attribute__((always_inline)) uint32_t per_trid_idma_tiles_to_process_get_interrupt_enable_cmdbuf_0() {
    return per_trid_idma_tiles_to_process_get_interrupt_enable_cmdbuf<CMDBUF_0>();
}
inline __attribute__((always_inline)) uint32_t per_trid_idma_tiles_to_process_get_interrupt_enable_cmdbuf_1() {
    return per_trid_idma_tiles_to_process_get_interrupt_enable_cmdbuf<CMDBUF_1>();
}
inline __attribute__((always_inline)) void per_trid_idma_tiles_to_process_set_interrupt_enable_cmdbuf_0(uint32_t val) {
    return per_trid_idma_tiles_to_process_set_interrupt_enable_cmdbuf<CMDBUF_0>(val);
}
inline __attribute__((always_inline)) void per_trid_idma_tiles_to_process_set_interrupt_enable_cmdbuf_1(uint32_t val) {
    return per_trid_idma_tiles_to_process_set_interrupt_enable_cmdbuf<CMDBUF_1>(val);
}
inline __attribute__((always_inline)) uint64_t per_trid_idma_tiles_to_process_interrupts_pending_cmdbuf_0() {
    return per_trid_idma_tiles_to_process_interrupts_pending_cmdbuf<CMDBUF_0>();
}
inline __attribute__((always_inline)) uint64_t per_trid_idma_tiles_to_process_interrupts_pending_cmdbuf_1() {
    return per_trid_idma_tiles_to_process_interrupts_pending_cmdbuf<CMDBUF_1>();
}
inline __attribute__((always_inline)) uint32_t per_trid_idma_tiles_to_process_get_interrupt_pending_cmdbuf_0() {
    return per_trid_idma_tiles_to_process_get_interrupt_pending_cmdbuf<CMDBUF_0>();
}
inline __attribute__((always_inline)) uint32_t per_trid_idma_tiles_to_process_get_interrupt_pending_cmdbuf_1() {
    return per_trid_idma_tiles_to_process_get_interrupt_pending_cmdbuf<CMDBUF_1>();
}
inline __attribute__((always_inline)) void per_trid_idma_tiles_to_process_set_interrupt_pending_cmdbuf_0(uint32_t val) {
    return per_trid_idma_tiles_to_process_set_interrupt_pending_cmdbuf<CMDBUF_0>(val);
}
inline __attribute__((always_inline)) void per_trid_idma_tiles_to_process_set_interrupt_pending_cmdbuf_1(uint32_t val) {
    return per_trid_idma_tiles_to_process_set_interrupt_pending_cmdbuf<CMDBUF_1>(val);
}
inline __attribute__((always_inline)) void per_trid_idma_tiles_to_process_interrupt_clear_cmdbuf_0(uint32_t trid) {
    return per_trid_idma_tiles_to_process_interrupt_clear_cmdbuf<CMDBUF_0>(trid);
}
inline __attribute__((always_inline)) void per_trid_idma_tiles_to_process_interrupt_clear_cmdbuf_1(uint32_t trid) {
    return per_trid_idma_tiles_to_process_interrupt_clear_cmdbuf<CMDBUF_1>(trid);
}
inline __attribute__((always_inline)) void per_vc_has_space_interrupt_enable_cmdbuf_0(uint32_t vc) {
    return per_vc_has_space_interrupt_enable_cmdbuf<CMDBUF_0>(vc);
}
inline __attribute__((always_inline)) void per_vc_has_space_interrupt_enable_cmdbuf_1(uint32_t vc) {
    return per_vc_has_space_interrupt_enable_cmdbuf<CMDBUF_1>(vc);
}
inline __attribute__((always_inline)) void per_vc_has_space_interrupt_disable_cmdbuf_0(uint32_t vc) {
    return per_vc_has_space_interrupt_disable_cmdbuf<CMDBUF_0>(vc);
}
inline __attribute__((always_inline)) void per_vc_has_space_interrupt_disable_cmdbuf_1(uint32_t vc) {
    return per_vc_has_space_interrupt_disable_cmdbuf<CMDBUF_1>(vc);
}
inline __attribute__((always_inline)) uint16_t per_vc_has_space_get_interrupt_enable_cmdbuf_0() {
    return per_vc_has_space_get_interrupt_enable_cmdbuf<CMDBUF_0>();
}
inline __attribute__((always_inline)) uint16_t per_vc_has_space_get_interrupt_enable_cmdbuf_1() {
    return per_vc_has_space_get_interrupt_enable_cmdbuf<CMDBUF_1>();
}
inline __attribute__((always_inline)) void per_vc_has_space_set_interrupt_enable_cmdbuf_0(uint16_t val) {
    return per_vc_has_space_set_interrupt_enable_cmdbuf<CMDBUF_0>(val);
}
inline __attribute__((always_inline)) void per_vc_has_space_set_interrupt_enable_cmdbuf_1(uint16_t val) {
    return per_vc_has_space_set_interrupt_enable_cmdbuf<CMDBUF_1>(val);
}
inline __attribute__((always_inline)) uint64_t per_vc_has_space_interrupts_pending_cmdbuf_0() {
    return per_vc_has_space_interrupts_pending_cmdbuf<CMDBUF_0>();
}
inline __attribute__((always_inline)) uint64_t per_vc_has_space_interrupts_pending_cmdbuf_1() {
    return per_vc_has_space_interrupts_pending_cmdbuf<CMDBUF_1>();
}
inline __attribute__((always_inline)) uint16_t per_vc_has_space_get_interrupt_pending_cmdbuf_0() {
    return per_vc_has_space_get_interrupt_pending_cmdbuf<CMDBUF_0>();
}
inline __attribute__((always_inline)) uint16_t per_vc_has_space_get_interrupt_pending_cmdbuf_1() {
    return per_vc_has_space_get_interrupt_pending_cmdbuf<CMDBUF_1>();
}
inline __attribute__((always_inline)) void per_vc_has_space_set_interrupt_pending_cmdbuf_0(uint16_t val) {
    return per_vc_has_space_set_interrupt_pending_cmdbuf<CMDBUF_0>(val);
}
inline __attribute__((always_inline)) void per_vc_has_space_set_interrupt_pending_cmdbuf_1(uint16_t val) {
    return per_vc_has_space_set_interrupt_pending_cmdbuf<CMDBUF_1>(val);
}
inline __attribute__((always_inline)) void per_vc_has_space_interrupt_clear_cmdbuf_0(uint32_t vc) {
    return per_vc_has_space_interrupt_clear_cmdbuf<CMDBUF_0>(vc);
}
inline __attribute__((always_inline)) void per_vc_has_space_interrupt_clear_cmdbuf_1(uint32_t vc) {
    return per_vc_has_space_interrupt_clear_cmdbuf<CMDBUF_1>(vc);
}
inline __attribute__((always_inline)) void per_idma_vc_has_space_interrupt_enable_cmdbuf_0(uint32_t vc) {
    return per_idma_vc_has_space_interrupt_enable_cmdbuf<CMDBUF_0>(vc);
}
inline __attribute__((always_inline)) void per_idma_vc_has_space_interrupt_enable_cmdbuf_1(uint32_t vc) {
    return per_idma_vc_has_space_interrupt_enable_cmdbuf<CMDBUF_1>(vc);
}
inline __attribute__((always_inline)) void per_idma_vc_has_space_interrupt_disable_cmdbuf_0(uint32_t vc) {
    return per_idma_vc_has_space_interrupt_disable_cmdbuf<CMDBUF_0>(vc);
}
inline __attribute__((always_inline)) void per_idma_vc_has_space_interrupt_disable_cmdbuf_1(uint32_t vc) {
    return per_idma_vc_has_space_interrupt_disable_cmdbuf<CMDBUF_1>(vc);
}
inline __attribute__((always_inline)) uint16_t per_idma_vc_has_space_get_interrupt_enable_cmdbuf_0() {
    return per_idma_vc_has_space_get_interrupt_enable_cmdbuf<CMDBUF_0>();
}
inline __attribute__((always_inline)) uint16_t per_idma_vc_has_space_get_interrupt_enable_cmdbuf_1() {
    return per_idma_vc_has_space_get_interrupt_enable_cmdbuf<CMDBUF_1>();
}
inline __attribute__((always_inline)) void per_idma_vc_has_space_set_interrupt_enable_cmdbuf_0(uint16_t val) {
    return per_idma_vc_has_space_set_interrupt_enable_cmdbuf<CMDBUF_0>(val);
}
inline __attribute__((always_inline)) void per_idma_vc_has_space_set_interrupt_enable_cmdbuf_1(uint16_t val) {
    return per_idma_vc_has_space_set_interrupt_enable_cmdbuf<CMDBUF_1>(val);
}
inline __attribute__((always_inline)) uint64_t per_idma_vc_has_space_interrupts_pending_cmdbuf_0() {
    return per_idma_vc_has_space_interrupts_pending_cmdbuf<CMDBUF_0>();
}
inline __attribute__((always_inline)) uint64_t per_idma_vc_has_space_interrupts_pending_cmdbuf_1() {
    return per_idma_vc_has_space_interrupts_pending_cmdbuf<CMDBUF_1>();
}
inline __attribute__((always_inline)) uint16_t per_idma_vc_has_space_get_interrupt_pending_cmdbuf_0() {
    return per_idma_vc_has_space_get_interrupt_pending_cmdbuf<CMDBUF_0>();
}
inline __attribute__((always_inline)) uint16_t per_idma_vc_has_space_get_interrupt_pending_cmdbuf_1() {
    return per_idma_vc_has_space_get_interrupt_pending_cmdbuf<CMDBUF_1>();
}
inline __attribute__((always_inline)) void per_idma_vc_has_space_set_interrupt_pending_cmdbuf_0(uint16_t val) {
    return per_idma_vc_has_space_set_interrupt_pending_cmdbuf<CMDBUF_0>(val);
}
inline __attribute__((always_inline)) void per_idma_vc_has_space_set_interrupt_pending_cmdbuf_1(uint16_t val) {
    return per_idma_vc_has_space_set_interrupt_pending_cmdbuf<CMDBUF_1>(val);
}
inline __attribute__((always_inline)) void per_idma_vc_has_space_interrupt_clear_cmdbuf_0(uint32_t vc) {
    return per_idma_vc_has_space_interrupt_clear_cmdbuf<CMDBUF_0>(vc);
}
inline __attribute__((always_inline)) void per_idma_vc_has_space_interrupt_clear_cmdbuf_1(uint32_t vc) {
    return per_idma_vc_has_space_interrupt_clear_cmdbuf<CMDBUF_1>(vc);
}
inline __attribute__((always_inline)) void noc_fast_read_cmdbuf_0(
    uint64_t src_addr, uint32_t dest_addr, uint64_t len_bytes, bool has_xy = true, PacketTags tags = {}) {
    return noc_fast_read_cmdbuf<CMDBUF_0>(src_addr, dest_addr, len_bytes, has_xy, tags);
}
inline __attribute__((always_inline)) void noc_fast_read_cmdbuf_1(
    uint64_t src_addr, uint32_t dest_addr, uint64_t len_bytes, bool has_xy = true, PacketTags tags = {}) {
    return noc_fast_read_cmdbuf<CMDBUF_1>(src_addr, dest_addr, len_bytes, has_xy, tags);
}
inline __attribute__((always_inline)) void noc_fast_write_cmdbuf_0(
    uint32_t src_addr,
    uint64_t dest_addr,
    uint64_t len_bytes,
    bool has_xy = true,
    bool posted = true,
    PacketTags tags = {}) {
    return noc_fast_write_cmdbuf<CMDBUF_0>(src_addr, dest_addr, len_bytes, has_xy, posted, tags);
}
inline __attribute__((always_inline)) void noc_fast_write_cmdbuf_1(
    uint32_t src_addr,
    uint64_t dest_addr,
    uint64_t len_bytes,
    bool has_xy = true,
    bool posted = true,
    PacketTags tags = {}) {
    return noc_fast_write_cmdbuf<CMDBUF_1>(src_addr, dest_addr, len_bytes, has_xy, posted, tags);
}
inline __attribute__((always_inline)) void noc_read_prep_cmdbuf_0(
    uint64_t src_coord,
    uint64_t src_addr,
    uint64_t dest_coord,
    uint64_t dest_addr,
    uint64_t len_bytes,
    const NocReadOptions& opts = {}) {
    return noc_read_prep_cmdbuf<CMDBUF_0>(src_coord, src_addr, dest_coord, dest_addr, len_bytes, opts);
}
inline __attribute__((always_inline)) void noc_read_prep_cmdbuf_1(
    uint64_t src_coord,
    uint64_t src_addr,
    uint64_t dest_coord,
    uint64_t dest_addr,
    uint64_t len_bytes,
    const NocReadOptions& opts = {}) {
    return noc_read_prep_cmdbuf<CMDBUF_1>(src_coord, src_addr, dest_coord, dest_addr, len_bytes, opts);
}
inline __attribute__((always_inline)) void noc_read_cmdbuf_0(
    uint64_t src_coord,
    uint64_t src_addr,
    uint64_t dest_coord,
    uint64_t dest_addr,
    uint64_t len_bytes,
    const NocReadOptions& opts = {}) {
    return noc_read_cmdbuf<CMDBUF_0>(src_coord, src_addr, dest_coord, dest_addr, len_bytes, opts);
}
inline __attribute__((always_inline)) void noc_read_cmdbuf_1(
    uint64_t src_coord,
    uint64_t src_addr,
    uint64_t dest_coord,
    uint64_t dest_addr,
    uint64_t len_bytes,
    const NocReadOptions& opts = {}) {
    return noc_read_cmdbuf<CMDBUF_1>(src_coord, src_addr, dest_coord, dest_addr, len_bytes, opts);
}
inline __attribute__((always_inline)) void noc_write_prep_cmdbuf_0(
    uint64_t src_coord,
    uint64_t src_addr,
    uint64_t dest_coord,
    uint64_t dest_addr,
    uint64_t len_bytes,
    const NocWriteOptions& opts = {}) {
    return noc_write_prep_cmdbuf<CMDBUF_0>(src_coord, src_addr, dest_coord, dest_addr, len_bytes, opts);
}
inline __attribute__((always_inline)) void noc_write_prep_cmdbuf_1(
    uint64_t src_coord,
    uint64_t src_addr,
    uint64_t dest_coord,
    uint64_t dest_addr,
    uint64_t len_bytes,
    const NocWriteOptions& opts = {}) {
    return noc_write_prep_cmdbuf<CMDBUF_1>(src_coord, src_addr, dest_coord, dest_addr, len_bytes, opts);
}
inline __attribute__((always_inline)) void noc_write_cmdbuf_0(
    uint64_t src_coord,
    uint64_t src_addr,
    uint64_t dest_coord,
    uint64_t dest_addr,
    uint64_t len_bytes,
    const NocWriteOptions& opts = {}) {
    return noc_write_cmdbuf<CMDBUF_0>(src_coord, src_addr, dest_coord, dest_addr, len_bytes, opts);
}
inline __attribute__((always_inline)) void noc_write_cmdbuf_1(
    uint64_t src_coord,
    uint64_t src_addr,
    uint64_t dest_coord,
    uint64_t dest_addr,
    uint64_t len_bytes,
    const NocWriteOptions& opts = {}) {
    return noc_write_cmdbuf<CMDBUF_1>(src_coord, src_addr, dest_coord, dest_addr, len_bytes, opts);
}
inline __attribute__((always_inline)) void idma_copy_cmdbuf_0(
    uint64_t src_addr, uint64_t dest_addr, uint64_t len_bytes, uint32_t trid = CMDBUF_DEF_TRID) {
    return idma_copy_cmdbuf<CMDBUF_0>(src_addr, dest_addr, len_bytes, trid);
}
inline __attribute__((always_inline)) void idma_copy_cmdbuf_1(
    uint64_t src_addr, uint64_t dest_addr, uint64_t len_bytes, uint32_t trid = CMDBUF_DEF_TRID) {
    return idma_copy_cmdbuf<CMDBUF_1>(src_addr, dest_addr, len_bytes, trid);
}
inline __attribute__((always_inline)) void idma_l1_atomic_accum_cmdbuf_0(
    uint64_t src_addr,
    uint64_t dest_addr,
    uint64_t len_bytes,
    uint32_t trid = CMDBUF_DEF_TRID,
    uint32_t fmt = 0x0,
    uint32_t atomic_op = 0x9) {
    return idma_l1_atomic_accum_cmdbuf<CMDBUF_0>(src_addr, dest_addr, len_bytes, trid, fmt, atomic_op);
}
inline __attribute__((always_inline)) void idma_l1_atomic_accum_cmdbuf_1(
    uint64_t src_addr,
    uint64_t dest_addr,
    uint64_t len_bytes,
    uint32_t trid = CMDBUF_DEF_TRID,
    uint32_t fmt = 0x0,
    uint32_t atomic_op = 0x9) {
    return idma_l1_atomic_accum_cmdbuf<CMDBUF_1>(src_addr, dest_addr, len_bytes, trid, fmt, atomic_op);
}
inline __attribute__((always_inline)) void noc_atomic_increment_cmdbuf_0(
    uint64_t coord, uint64_t addr, uint32_t incr = 1, uint32_t wrap = 31, PacketTags tags = {}) {
    return noc_atomic_increment_cmdbuf<CMDBUF_0>(coord, addr, incr, wrap, tags);
}
inline __attribute__((always_inline)) void noc_atomic_increment_cmdbuf_1(
    uint64_t coord, uint64_t addr, uint32_t incr = 1, uint32_t wrap = 31, PacketTags tags = {}) {
    return noc_atomic_increment_cmdbuf<CMDBUF_1>(coord, addr, incr, wrap, tags);
}
inline __attribute__((always_inline)) uint64_t free_space_cmdbuf_0(uint32_t vc) {
    return free_space_cmdbuf<CMDBUF_0>(vc);
}
inline __attribute__((always_inline)) uint64_t free_space_cmdbuf_1(uint32_t vc) {
    return free_space_cmdbuf<CMDBUF_1>(vc);
}
inline __attribute__((always_inline)) uint64_t free_space_cmdbuf_0() { return free_space_cmdbuf<CMDBUF_0>(); }
inline __attribute__((always_inline)) uint64_t free_space_cmdbuf_1() { return free_space_cmdbuf<CMDBUF_1>(); }
inline __attribute__((always_inline)) uint64_t idma_free_space_cmdbuf_0(uint32_t vc) {
    return idma_free_space_cmdbuf<CMDBUF_0>(vc);
}
inline __attribute__((always_inline)) uint64_t idma_free_space_cmdbuf_1(uint32_t vc) {
    return idma_free_space_cmdbuf<CMDBUF_1>(vc);
}
inline __attribute__((always_inline)) uint64_t idma_free_space_cmdbuf_0() { return idma_free_space_cmdbuf<CMDBUF_0>(); }
inline __attribute__((always_inline)) uint64_t idma_free_space_cmdbuf_1() { return idma_free_space_cmdbuf<CMDBUF_1>(); }
inline __attribute__((always_inline)) bool noc_reads_acked_cmdbuf_0(uint32_t trid) {
    return noc_reads_acked_cmdbuf<CMDBUF_0>(trid);
}
inline __attribute__((always_inline)) bool noc_reads_acked_cmdbuf_1(uint32_t trid) {
    return noc_reads_acked_cmdbuf<CMDBUF_1>(trid);
}
inline __attribute__((always_inline)) bool noc_reads_acked_cmdbuf_0() { return noc_reads_acked_cmdbuf<CMDBUF_0>(); }
inline __attribute__((always_inline)) bool noc_reads_acked_cmdbuf_1() { return noc_reads_acked_cmdbuf<CMDBUF_1>(); }
inline __attribute__((always_inline)) bool all_noc_reads_acked_cmdbuf_0() {
    return all_noc_reads_acked_cmdbuf<CMDBUF_0>();
}
inline __attribute__((always_inline)) bool all_noc_reads_acked_cmdbuf_1() {
    return all_noc_reads_acked_cmdbuf<CMDBUF_1>();
}
inline __attribute__((always_inline)) bool idma_acked_cmdbuf_0(uint32_t trid) {
    return idma_acked_cmdbuf<CMDBUF_0>(trid);
}
inline __attribute__((always_inline)) bool idma_acked_cmdbuf_1(uint32_t trid) {
    return idma_acked_cmdbuf<CMDBUF_1>(trid);
}
inline __attribute__((always_inline)) bool idma_acked_cmdbuf_0() { return idma_acked_cmdbuf<CMDBUF_0>(); }
inline __attribute__((always_inline)) bool idma_acked_cmdbuf_1() { return idma_acked_cmdbuf<CMDBUF_1>(); }
inline __attribute__((always_inline)) bool all_idma_acked_cmdbuf_0() { return all_idma_acked_cmdbuf<CMDBUF_0>(); }
inline __attribute__((always_inline)) bool all_idma_acked_cmdbuf_1() { return all_idma_acked_cmdbuf<CMDBUF_1>(); }
inline __attribute__((always_inline)) bool noc_writes_sent_cmdbuf_0(uint32_t trid) {
    return noc_writes_sent_cmdbuf<CMDBUF_0>(trid);
}
inline __attribute__((always_inline)) bool noc_writes_sent_cmdbuf_1(uint32_t trid) {
    return noc_writes_sent_cmdbuf<CMDBUF_1>(trid);
}
inline __attribute__((always_inline)) bool noc_writes_sent_cmdbuf_0() { return noc_writes_sent_cmdbuf<CMDBUF_0>(); }
inline __attribute__((always_inline)) bool noc_writes_sent_cmdbuf_1() { return noc_writes_sent_cmdbuf<CMDBUF_1>(); }
inline __attribute__((always_inline)) bool all_noc_writes_sent_cmdbuf_0() {
    return all_noc_writes_sent_cmdbuf<CMDBUF_0>();
}
inline __attribute__((always_inline)) bool all_noc_writes_sent_cmdbuf_1() {
    return all_noc_writes_sent_cmdbuf<CMDBUF_1>();
}
inline __attribute__((always_inline)) bool noc_nonposted_writes_acked_cmdbuf_0(uint32_t trid) {
    return noc_nonposted_writes_acked_cmdbuf<CMDBUF_0>(trid);
}
inline __attribute__((always_inline)) bool noc_nonposted_writes_acked_cmdbuf_1(uint32_t trid) {
    return noc_nonposted_writes_acked_cmdbuf<CMDBUF_1>(trid);
}
inline __attribute__((always_inline)) bool noc_nonposted_writes_acked_cmdbuf_0() {
    return noc_nonposted_writes_acked_cmdbuf<CMDBUF_0>();
}
inline __attribute__((always_inline)) bool noc_nonposted_writes_acked_cmdbuf_1() {
    return noc_nonposted_writes_acked_cmdbuf<CMDBUF_1>();
}
inline __attribute__((always_inline)) bool all_noc_nonposted_writes_acked_cmdbuf_0() {
    return all_noc_nonposted_writes_acked_cmdbuf<CMDBUF_0>();
}
inline __attribute__((always_inline)) bool all_noc_nonposted_writes_acked_cmdbuf_1() {
    return all_noc_nonposted_writes_acked_cmdbuf<CMDBUF_1>();
}

//////////////////////
/// Simple CMD Buf ///
// Below are the functions for the simple command buffer, the third command buffer of the overlay.
// It takes the same arguments as the functions above; it has no auto-increment, VC/TID wrapping,
// scatter-list or iDMA support.
//////////////////////

inline __attribute__((always_inline)) void reset_reg_cmdbuf() { __builtin_riscv_ttrocc_scmdbuf_reset(); }

inline __attribute__((always_inline)) void setup_as_copy_reg_cmdbuf(const NocTransferConfig& cfg) {
    __builtin_riscv_ttrocc_scmdbuf_wr_reg(
        TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_MISC_REG_OFFSET / 8, detail::noc_transfer_misc(cfg).val);

    if (cfg.mcast) {
        __builtin_riscv_ttrocc_scmdbuf_wr_reg(
            TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_MCAST_EXCLUDE_REG_OFFSET / 8, cfg.mcast_exclude.val);
    }
}

inline __attribute__((always_inline)) void setup_as_atomic_reg_cmdbuf() {
    TT_ROCC_CMD_BUF_MISC_reg_u misc;
    misc.val = TT_ROCC_CMD_BUF_MISC_REG_DEFAULT;

    misc.f.posted = 1;
    misc.f.write_trans = 0;
    misc.f.atomic_trans = 1;

    __builtin_riscv_ttrocc_scmdbuf_wr_reg(TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_MISC_REG_OFFSET / 8, misc.val);
}

inline __attribute__((always_inline)) void setup_vcs_reg_cmdbuf(NocVcs vcs) {
    const uint32_t req_vc = vcs == NocVcs::READ    ? CMDBUF_RD_REQ_VC
                            : vcs == NocVcs::WRITE ? CMDBUF_WR_REQ_VC
                                                   : CMDBUF_MCAST_REQ_VC;
    const uint32_t resp_vc = vcs == NocVcs::READ    ? CMDBUF_RD_RESP_VC
                             : vcs == NocVcs::WRITE ? CMDBUF_WR_RESP_VC
                                                    : CMDBUF_MCAST_RESP_VC;
    __builtin_riscv_ttrocc_scmdbuf_wr_reg(TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_REQ_VC_REG_OFFSET / 8, req_vc);
    __builtin_riscv_ttrocc_scmdbuf_wr_reg(TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_RESP_VC_REG_OFFSET / 8, resp_vc);
}

inline __attribute__((always_inline)) void setup_trids_reg_cmdbuf(uint32_t trid = CMDBUF_DEF_TRID) {
    __builtin_riscv_ttrocc_scmdbuf_wr_reg(TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_TR_ID_REG_OFFSET / 8, trid);
}

inline __attribute__((always_inline)) uint32_t swap_trid_reg_cmdbuf(uint32_t new_trid) {
    uint32_t prev_trid =
        __builtin_riscv_ttrocc_scmdbuf_rd_reg(TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_TR_ID_REG_OFFSET / 8);
    __builtin_riscv_ttrocc_scmdbuf_wr_reg(TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_TR_ID_REG_OFFSET / 8, new_trid);
    return prev_trid;
}

inline __attribute__((always_inline)) void setup_packet_tags_reg_cmdbuf(PacketTags tags) {
    TT_ROCC_CMD_BUF_PACKET_TAGS_reg_u packet_tags;
    packet_tags.val = TT_ROCC_CMD_BUF_PACKET_TAGS_REG_DEFAULT;

    packet_tags.f.snoop_bit = tags.snoop;
    packet_tags.f.flush_bit = tags.flush;

    __builtin_riscv_ttrocc_scmdbuf_wr_reg(
        TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_PACKET_TAGS_REG_OFFSET / 8, packet_tags.val);
}

inline __attribute__((always_inline)) uint64_t get_src_reg_cmdbuf() {
    return __builtin_riscv_ttrocc_scmdbuf_rd_reg(TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_SRC_ADDR_REG_OFFSET / 8);
}

inline __attribute__((always_inline)) void set_src_reg_cmdbuf(uint64_t addr, uint64_t coord) {
    __builtin_riscv_ttrocc_scmdbuf_wr_reg(TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_SRC_ADDR_REG_OFFSET / 8, addr);
    __builtin_riscv_ttrocc_scmdbuf_wr_reg(TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_SRC_COORD_REG_OFFSET / 8, coord);
}
inline __attribute__((always_inline)) void set_src_reg_cmdbuf(uint64_t addr) {
    __builtin_riscv_ttrocc_scmdbuf_wr_reg(TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_SRC_ADDR_REG_OFFSET / 8, addr);
}

inline __attribute__((always_inline)) uint64_t get_dest_reg_cmdbuf() {
    return __builtin_riscv_ttrocc_scmdbuf_rd_reg(TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_DEST_ADDR_REG_OFFSET / 8);
}

inline __attribute__((always_inline)) void set_dest_reg_cmdbuf(uint64_t addr, uint64_t coord) {
    __builtin_riscv_ttrocc_scmdbuf_wr_reg(TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_DEST_ADDR_REG_OFFSET / 8, addr);
    __builtin_riscv_ttrocc_scmdbuf_wr_reg(TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_DEST_COORD_REG_OFFSET / 8, coord);
}
inline __attribute__((always_inline)) void set_dest_reg_cmdbuf(uint64_t addr) {
    __builtin_riscv_ttrocc_scmdbuf_wr_reg(TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_DEST_ADDR_REG_OFFSET / 8, addr);
}

inline __attribute__((always_inline)) void set_len_reg_cmdbuf(uint64_t len_bytes) {
    __builtin_riscv_ttrocc_scmdbuf_wr_reg(TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_LEN_BYTES_REG_OFFSET / 8, len_bytes);
}

inline __attribute__((always_inline)) void issue_reg_cmdbuf() { __builtin_riscv_ttrocc_scmdbuf_issue_trans(); }

inline __attribute__((always_inline)) void issue_read_reg_cmdbuf() { issue_reg_cmdbuf(); }

inline __attribute__((always_inline)) void issue_write_reg_cmdbuf() { issue_reg_cmdbuf(); }

inline __attribute__((always_inline)) void issue_write_inline_reg_cmdbuf(uint64_t data) {
    __builtin_riscv_ttrocc_scmdbuf_issue_inline_trans(data);
}

inline __attribute__((always_inline)) void issue_write_inline_len_reg_cmdbuf(
    uint64_t data,
    uint64_t dest_addr,
    uint64_t len_bytes,
    bool has_xy = true,
    bool posted = true,
    PacketTags tags = {}) {
    uint64_t rs2 = dest_addr | ((len_bytes - 1) << 57) | detail::fast_issue_flags(has_xy, posted, tags);
    __builtin_riscv_ttrocc_scmdbuf_issue_inline_addr_trans(data, rs2);
}

inline __attribute__((always_inline)) void interrupt_enable_reg_cmdbuf(uint32_t id) {
    TT_ROCC_CMD_BUF_IE_reg_u val;
    val.val = __builtin_riscv_ttrocc_scmdbuf_rd_reg(TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_IE_REG_OFFSET / 8);
    val.val |= (1ULL << id);
    __builtin_riscv_ttrocc_scmdbuf_wr_reg(TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_IE_REG_OFFSET / 8, val.val);
}

inline __attribute__((always_inline)) void interrupt_disable_reg_cmdbuf(uint32_t id) {
    TT_ROCC_CMD_BUF_IE_reg_u val;
    val.val = __builtin_riscv_ttrocc_scmdbuf_rd_reg(TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_IE_REG_OFFSET / 8);
    val.val &= ~(1ULL << id);
    __builtin_riscv_ttrocc_scmdbuf_wr_reg(TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_IE_REG_OFFSET / 8, val.val);
}

inline __attribute__((always_inline)) uint64_t interrupts_pending_reg_cmdbuf() {
    TT_ROCC_CMD_BUF_IP_reg_u val;
    val.val = __builtin_riscv_ttrocc_scmdbuf_rd_reg(TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_IP_REG_OFFSET / 8);
    return val.val;
}

inline __attribute__((always_inline)) void interrupt_clear_reg_cmdbuf(uint32_t id) {
    TT_ROCC_CMD_BUF_IP_reg_u val;
    val.val = ~(1ULL << id);
    __builtin_riscv_ttrocc_scmdbuf_wr_reg(TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_IP_REG_OFFSET / 8, val.val);
}

inline __attribute__((always_inline)) void noc_fast_read_reg_cmdbuf(
    uint64_t src_addr, uint32_t dest_addr, uint64_t len_bytes, bool has_xy = true, PacketTags tags = {}) {
    uint64_t rs1 = (len_bytes << 32) | dest_addr;
    uint64_t rs2 = src_addr | detail::fast_issue_flags(has_xy, /*posted=*/false, tags);
    __builtin_riscv_ttrocc_scmdbuf_issue_read2_trans(rs1, rs2);
}

inline __attribute__((always_inline)) void noc_read_reg_cmdbuf(
    uint64_t src_coord,
    uint64_t src_addr,
    uint64_t dest_coord,
    uint64_t dest_addr,
    uint64_t len_bytes,
    const NocReadOptions& opts = {}) {
    reset_reg_cmdbuf();
    setup_as_copy_reg_cmdbuf({.write = false});
    setup_vcs_reg_cmdbuf(NocVcs::READ);
    setup_trids_reg_cmdbuf(opts.trid);
    if (opts.tags.snoop || opts.tags.flush) {
        setup_packet_tags_reg_cmdbuf(opts.tags);
    }
    set_src_reg_cmdbuf(src_addr, src_coord);
    set_dest_reg_cmdbuf(dest_addr, dest_coord);
    set_len_reg_cmdbuf(len_bytes);
    issue_read_reg_cmdbuf();
}

inline __attribute__((always_inline)) void noc_fast_write_reg_cmdbuf(
    uint32_t src_addr,
    uint64_t dest_addr,
    uint64_t len_bytes,
    bool has_xy = true,
    bool posted = true,
    PacketTags tags = {}) {
    uint64_t rs1 = (len_bytes << 32) | src_addr;
    uint64_t rs2 = dest_addr | detail::fast_issue_flags(has_xy, posted, tags);
    __builtin_riscv_ttrocc_scmdbuf_issue_write2_trans(rs1, rs2);
}

inline __attribute__((always_inline)) void noc_write_reg_cmdbuf(
    uint64_t src_coord,
    uint64_t src_addr,
    uint64_t dest_coord,
    uint64_t dest_addr,
    uint64_t len_bytes,
    const NocWriteOptions& opts = {}) {
    reset_reg_cmdbuf();
    setup_as_copy_reg_cmdbuf(
        {.write = true,
         .posted = opts.posted,
         .mcast = opts.mcast,
         .linked = opts.linked,
         .src_include = opts.src_include,
         .mcast_exclude = opts.mcast_exclude});
    setup_vcs_reg_cmdbuf(opts.mcast ? NocVcs::MCAST_WRITE : NocVcs::WRITE);
    setup_trids_reg_cmdbuf(opts.trid);
    if (opts.tags.snoop || opts.tags.flush) {
        setup_packet_tags_reg_cmdbuf(opts.tags);
    }
    set_src_reg_cmdbuf(src_addr, src_coord);
    set_dest_reg_cmdbuf(dest_addr, dest_coord);
    set_len_reg_cmdbuf(len_bytes);
    issue_write_reg_cmdbuf();
}

inline __attribute__((always_inline)) void noc_atomic_increment_reg_cmdbuf(
    uint64_t coord, uint64_t addr, uint32_t incr = 1, uint32_t wrap = 31, PacketTags tags = {}) {
    uint64_t at_len =
        NOC_AT_INS(NOC_AT_INS_INCR_GET) | NOC_AT_WRAP(wrap) | NOC_AT_IND_32((addr >> 2) & 0x3) | NOC_AT_IND_32_SRC(0);
    setup_as_atomic_reg_cmdbuf();
    if (tags.snoop || tags.flush) {
        setup_packet_tags_reg_cmdbuf(tags);
    }
    set_dest_reg_cmdbuf(addr, coord);
    set_len_reg_cmdbuf(at_len);
    __builtin_riscv_ttrocc_scmdbuf_wr_reg(
        TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_INLINE_DATA_REG_OFFSET / 8, static_cast<uint64_t>(incr));
    issue_reg_cmdbuf();
}

inline __attribute__((always_inline)) uint64_t free_space_reg_cmdbuf(uint32_t vc) {
    return __builtin_riscv_ttrocc_scmdbuf_get_vc_space_vc(vc);
}
inline __attribute__((always_inline)) uint64_t free_space_reg_cmdbuf() {
    return __builtin_riscv_ttrocc_scmdbuf_get_vc_space();
}

inline __attribute__((always_inline)) bool noc_reads_acked_reg_cmdbuf(uint32_t trid) {
    return __builtin_riscv_ttrocc_scmdbuf_tr_ack_trid(trid) == 0;
}
inline __attribute__((always_inline)) bool noc_reads_acked_reg_cmdbuf() {
    return __builtin_riscv_ttrocc_scmdbuf_tr_ack() == 0;
}

inline __attribute__((always_inline)) bool noc_writes_sent_reg_cmdbuf(uint32_t trid) {
    return __builtin_riscv_ttrocc_scmdbuf_wr_sent_trid(trid) == 0;
}
inline __attribute__((always_inline)) bool noc_writes_sent_reg_cmdbuf() {
    return __builtin_riscv_ttrocc_scmdbuf_wr_sent() == 0;
}

inline __attribute__((always_inline)) bool noc_nonposted_writes_acked_reg_cmdbuf(uint32_t trid) {
    return __builtin_riscv_ttrocc_scmdbuf_tr_ack_trid(trid) == 0;
}
inline __attribute__((always_inline)) bool noc_nonposted_writes_acked_reg_cmdbuf() {
    return __builtin_riscv_ttrocc_scmdbuf_tr_ack() == 0;
}

}  // namespace overlay
