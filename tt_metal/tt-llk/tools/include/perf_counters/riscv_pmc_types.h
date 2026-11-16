// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>
#include <utility>

namespace llk::perf::riscv
{

// Core families that carry the RISC-V performance monitor and how their mhpmevent select is encoded.
enum class CoreKind : std::uint8_t
{
    BABY = 0, // Blackhole BRISC/NCRISC/TRISC: eight 4-bit nibbles, one event index per pipeline domain
    TRISC,    // Quasar TRISC: the BABY layout plus two IF events
    DM,       // Quasar data-movement cores: (event mask << 8) | event set, one bit per event
};

inline constexpr std::size_t NUM_CORE_KINDS = 3;

// Names only: a profiler record carries the raw mhpmevent select, not an ordinal, so the host decodes a select
// by looking it up in the arch tables. The enum exists so the host can parse names; append, never reorder.
enum class PmcEvent : std::uint16_t
{
    UNDEF = 0,
    BP_PC_STALL_ANY,
    BP_STALL_NO_ID_RTR,
    BP_STALL_NO_MEM_RTR,
    BP_STALL_PC_FIFO_FULL,
    BP_STALL_BP_FIFO_FULL,
    BP_BTB_HIT,
    BP_PRED_TYPE_JALR,
    BP_PRED_TYPE_JAL,
    BP_PRED_TYPE_BR,
    IF_STALL_ANY,
    IF_STALL_NO_ID_RTR,
    IF_STALL_LQ_FULL,
    IF_STALL_LQ_SB_FULL,
    IF_REQ_STALL_ANY,
    IF_REQ_STALL_NO_MEM_RTR,
    DE_STALL_ANY,
    DE_STALL_NO_UNIT_RTR,
    DE_STALL_IF_EMPTY,
    DE_STALL_SYNC,
    DE_STALL_HAZARD,
    DE_HAZARD_INT,
    DE_HAZARD_FP,
    DE_HAZARD_VEC,
    DE_REG_DATA_FORWARD,
    DE_MEM_OP_DISPATCH_LD,
    DE_MEM_OP_DISPATCH_ST,
    DE_MEM_INSTR_DISPATCH_VEC_LD,
    DE_MEM_INSTR_DISPATCH_VEC_ST,
    DE_EX_OP_DISPATCH_MUL,
    DE_EX_OP_DISPATCH_DIV,
    DE1_EX_OP_DISPATCH_CSR,
    DE1_EX_OP_DISPATCH_I_EXT,
    DE1_EX_OP_DISPATCH_M_EXT,
    DE1_EX_OP_DISPATCH_A_EXT,
    DE1_EX_OP_DISPATCH_ZB_EXT,
    DE1_FP_OP_DISPATCH_S,
    DE1_FP_OP_DISPATCH_ZFH,
    DE1_VEC_OP_DISPATCH_CSR,
    DE1_VEC_OP_DISPATCH_INT,
    DE1_VEC_OP_DISPATCH_FIXED,
    DE1_VEC_OP_DISPATCH_FLOAT,
    DE1_VEC_OP_DISPATCH_LDST,
    DE1_VALID_TT_INSTR,
    DE1_VALID_VEC_REPLAY,
    DE1_VALID_ILLEGAL,
    EX_STALL_ANY,
    EX_STALL_NO_INSTR,
    EX_MISPRED_ANY,
    EX_MISPRED_NEW_BR,
    EX_MISPRED_DIR,
    EX_MISPRED_TGT_COND,
    EX_MISPRED_TGT_UNCOND,
    EX_INST_RET,
    LS_STALL_ANY,
    LS_STALL_NO_MEM_RTR,
    LS_STALL_LDQ_FULL,
    LS_STALL_STORE_BUF_FULL,
    LS_STALL_SKID_BUF_FULL,
    LS_FENCE_OP,
    LS_STALL_WAIT_FOR_DATA,
    LS_OP_RET,
    LS_STORE_REQ_WINS,
    LS_WR_BUF_STLH,
    LS_WR_BUF_STSC,
    LS_LOAD_OP,
    LS_STORE_OP,
    LS_VEC_LOAD_OP,
    LS_VEC_STORE_OP,
    TTCSR_SSTATUS7_HIGH,
    TTCSR_SSTATUS6_HIGH,
    TTCSR_SSTATUS5_HIGH,
    TTCSR_SSTATUS4_HIGH,
    TTCSR_SSTATUS3_HIGH,
    TTCSR_SSTATUS2_HIGH,
    TTCSR_SSTATUS1_HIGH,
    TTCSR_SSTATUS0_HIGH,
    TTCSR_BSTATUS_HIGH,
    TTCSR_QSTATUS_HIGH,
    MEM_STALL_ANY,
    MEM_LOCAL_REG_RD_ONGOING,
    MEM_MAILBOX_RD_ONGOING,
    MEM_DATA_L1_RD_ONGOING,
    MEM_CFG_REG_RD_ONGOING,
    MEM_REG_RD_ONGOING,
    MEM_TDMA_REG_RD_ONGOING,
    MEM_PC_BUF_RD_ONGOING,
    MEM_DATA_ACC_REQ_LOCAL,
    MEM_INS_BUF_WRITE,
    MEM_INS_BUF_WRITE_MULTIPLE,
    MEM_DATA_RD_REQ_L1,
    MEM_DATA_RD_REQ_HIT_LOCACHE,
    MEM_IF_ACCESS,
    MEM_IF_ACCESS_HIT,
    IF_STALL_NO_IF_RTS,
    IF_ID_STALL_LD,
    DM_COMMIT_EXCEPTION,
    DM_COMMIT_LOAD,
    DM_COMMIT_STORE,
    DM_COMMIT_AMO,
    DM_COMMIT_SYSTEM,
    DM_COMMIT_ARITH,
    DM_COMMIT_BRANCH,
    DM_COMMIT_JAL,
    DM_COMMIT_JALR,
    DM_COMMIT_MUL,
    DM_COMMIT_DIV,
    DM_PIPE_LOAD_USE,
    DM_PIPE_LONG_LATENCY,
    DM_PIPE_CSR_INTERLOCK,
    DM_PIPE_ICACHE_BLOCKED,
    DM_PIPE_DCACHE_BLOCKED,
    DM_PIPE_BRANCH_MISPREDICT,
    DM_PIPE_TARGET_MISPREDICT,
    DM_PIPE_FLUSH,
    DM_PIPE_REPLAY,
    DM_PIPE_MULDIV_INTERLOCK,
    DM_MEM_ICACHE_MISS,
    DM_MEM_DCACHE_MISS,
    DM_MEM_DCACHE_RELEASE,
};

// {event name, mhpmevent select for that single event}. A select that ORs several events is the bitwise OR of
// entries of the same core kind, one per nibble (BABY, TRISC) or one set with several mask bits (DM).
using Entry = std::pair<PmcEvent, std::uint32_t>;

} // namespace llk::perf::riscv
