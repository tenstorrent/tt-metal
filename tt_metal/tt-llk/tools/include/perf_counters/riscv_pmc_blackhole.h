// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

#include "perf_counters/riscv_pmc_types.h"

// Blackhole baby cores (BRISC, NCRISC, TRISC0-2): select tables of the RISC-V performance monitor. Include through
// perf_counters/riscv_pmc_inventory.h. Event 0 of every domain is a tie-off and is not listed.

namespace llk::perf::riscv
{

// clang-format off
inline constexpr std::array<Entry, 93> baby_events = {
    {
    {PmcEvent::BP_PC_STALL_ANY, 0x1},
    {PmcEvent::BP_STALL_NO_ID_RTR, 0x2},
    {PmcEvent::BP_STALL_NO_MEM_RTR, 0x3},
    {PmcEvent::BP_STALL_PC_FIFO_FULL, 0x4},
    {PmcEvent::BP_STALL_BP_FIFO_FULL, 0x5},
    {PmcEvent::BP_BTB_HIT, 0x6},
    {PmcEvent::BP_PRED_TYPE_JALR, 0x7},
    {PmcEvent::BP_PRED_TYPE_JAL, 0x8},
    {PmcEvent::BP_PRED_TYPE_BR, 0x9},
    {PmcEvent::IF_STALL_ANY, 0x10},
    {PmcEvent::IF_STALL_NO_ID_RTR, 0x20},
    {PmcEvent::IF_STALL_LQ_FULL, 0x30},
    {PmcEvent::IF_STALL_LQ_SB_FULL, 0x40},
    {PmcEvent::IF_REQ_STALL_ANY, 0x50},
    {PmcEvent::IF_REQ_STALL_NO_MEM_RTR, 0x60},
    {PmcEvent::DE_STALL_ANY, 0x100},
    {PmcEvent::DE_STALL_NO_UNIT_RTR, 0x200},
    {PmcEvent::DE_STALL_IF_EMPTY, 0x300},
    {PmcEvent::DE_STALL_SYNC, 0x400},
    {PmcEvent::DE_STALL_HAZARD, 0x500},
    {PmcEvent::DE_HAZARD_INT, 0x600},
    {PmcEvent::DE_HAZARD_FP, 0x700},
    {PmcEvent::DE_HAZARD_VEC, 0x800},
    {PmcEvent::DE_REG_DATA_FORWARD, 0x900},
    {PmcEvent::DE_MEM_OP_DISPATCH_LD, 0xA00},
    {PmcEvent::DE_MEM_OP_DISPATCH_ST, 0xB00},
    {PmcEvent::DE_MEM_INSTR_DISPATCH_VEC_LD, 0xC00},
    {PmcEvent::DE_MEM_INSTR_DISPATCH_VEC_ST, 0xD00},
    {PmcEvent::DE_EX_OP_DISPATCH_MUL, 0xE00},
    {PmcEvent::DE_EX_OP_DISPATCH_DIV, 0xF00},
    {PmcEvent::DE1_EX_OP_DISPATCH_CSR, 0x1000},
    {PmcEvent::DE1_EX_OP_DISPATCH_I_EXT, 0x2000},
    {PmcEvent::DE1_EX_OP_DISPATCH_M_EXT, 0x3000},
    {PmcEvent::DE1_EX_OP_DISPATCH_A_EXT, 0x4000},
    {PmcEvent::DE1_EX_OP_DISPATCH_ZB_EXT, 0x5000},
    {PmcEvent::DE1_FP_OP_DISPATCH_S, 0x6000},
    {PmcEvent::DE1_FP_OP_DISPATCH_ZFH, 0x7000},
    {PmcEvent::DE1_VEC_OP_DISPATCH_CSR, 0x8000},
    {PmcEvent::DE1_VEC_OP_DISPATCH_INT, 0x9000},
    {PmcEvent::DE1_VEC_OP_DISPATCH_FIXED, 0xA000},
    {PmcEvent::DE1_VEC_OP_DISPATCH_FLOAT, 0xB000},
    {PmcEvent::DE1_VEC_OP_DISPATCH_LDST, 0xC000},
    {PmcEvent::DE1_VALID_TT_INSTR, 0xD000},
    {PmcEvent::DE1_VALID_VEC_REPLAY, 0xE000},
    {PmcEvent::DE1_VALID_ILLEGAL, 0xF000},
    {PmcEvent::EX_STALL_ANY, 0x10000},
    {PmcEvent::EX_STALL_NO_INSTR, 0x20000},
    {PmcEvent::EX_MISPRED_ANY, 0x30000},
    {PmcEvent::EX_MISPRED_NEW_BR, 0x40000},
    {PmcEvent::EX_MISPRED_DIR, 0x50000},
    {PmcEvent::EX_MISPRED_TGT_COND, 0x60000},
    {PmcEvent::EX_MISPRED_TGT_UNCOND, 0x70000},
    {PmcEvent::EX_INST_RET, 0x80000},
    {PmcEvent::LS_STALL_ANY, 0x100000},
    {PmcEvent::LS_STALL_NO_MEM_RTR, 0x200000},
    {PmcEvent::LS_STALL_LDQ_FULL, 0x300000},
    {PmcEvent::LS_STALL_STORE_BUF_FULL, 0x400000},
    {PmcEvent::LS_STALL_SKID_BUF_FULL, 0x500000},
    {PmcEvent::LS_FENCE_OP, 0x600000},
    {PmcEvent::LS_STALL_WAIT_FOR_DATA, 0x700000},
    {PmcEvent::LS_OP_RET, 0x800000},
    {PmcEvent::LS_STORE_REQ_WINS, 0x900000},
    {PmcEvent::LS_WR_BUF_STLH, 0xA00000},
    {PmcEvent::LS_WR_BUF_STSC, 0xB00000},
    {PmcEvent::LS_LOAD_OP, 0xC00000},
    {PmcEvent::LS_STORE_OP, 0xD00000},
    {PmcEvent::LS_VEC_LOAD_OP, 0xE00000},
    {PmcEvent::LS_VEC_STORE_OP, 0xF00000},
    {PmcEvent::TTCSR_SSTATUS7_HIGH, 0x1000000},
    {PmcEvent::TTCSR_SSTATUS6_HIGH, 0x2000000},
    {PmcEvent::TTCSR_SSTATUS5_HIGH, 0x3000000},
    {PmcEvent::TTCSR_SSTATUS4_HIGH, 0x4000000},
    {PmcEvent::TTCSR_SSTATUS3_HIGH, 0x5000000},
    {PmcEvent::TTCSR_SSTATUS2_HIGH, 0x6000000},
    {PmcEvent::TTCSR_SSTATUS1_HIGH, 0x7000000},
    {PmcEvent::TTCSR_SSTATUS0_HIGH, 0x8000000},
    {PmcEvent::TTCSR_BSTATUS_HIGH, 0x9000000},
    {PmcEvent::TTCSR_QSTATUS_HIGH, 0xA000000},
    {PmcEvent::MEM_STALL_ANY, 0x10000000},
    {PmcEvent::MEM_LOCAL_REG_RD_ONGOING, 0x20000000},
    {PmcEvent::MEM_MAILBOX_RD_ONGOING, 0x30000000},
    {PmcEvent::MEM_DATA_L1_RD_ONGOING, 0x40000000},
    {PmcEvent::MEM_CFG_REG_RD_ONGOING, 0x50000000},
    {PmcEvent::MEM_REG_RD_ONGOING, 0x60000000},
    {PmcEvent::MEM_TDMA_REG_RD_ONGOING, 0x70000000},
    {PmcEvent::MEM_PC_BUF_RD_ONGOING, 0x80000000},
    {PmcEvent::MEM_DATA_ACC_REQ_LOCAL, 0x90000000},
    {PmcEvent::MEM_INS_BUF_WRITE, 0xA0000000},
    {PmcEvent::MEM_INS_BUF_WRITE_MULTIPLE, 0xB0000000},
    {PmcEvent::MEM_DATA_RD_REQ_L1, 0xC0000000},
    {PmcEvent::MEM_DATA_RD_REQ_HIT_LOCACHE, 0xD0000000},
    {PmcEvent::MEM_IF_ACCESS, 0xE0000000},
    {PmcEvent::MEM_IF_ACCESS_HIT, 0xF0000000}}};
// clang-format on

} // namespace llk::perf::riscv
