// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

#include "perf_counters/riscv_pmc_types.h"

// Blackhole baby cores (BRISC, NCRISC, TRISC0-2, ERISC0 and ERISC1 of the ethernet tiles): select
// tables of the RISC-V performance monitor. Event 0 of every domain is a tie-off and is not listed.

namespace llk::perf::riscv
{

// clang-format off
inline constexpr std::array<Entry, 93> baby_events = {
    {
    {PmcEvent::BP_PC_STALL_ANY, 0x1},                    // fetch PC stalled for any reason
    {PmcEvent::BP_STALL_NO_ID_RTR, 0x2},                 // stalled, decode not ready
    {PmcEvent::BP_STALL_NO_MEM_RTR, 0x3},                // stalled, instruction memory not ready
    {PmcEvent::BP_STALL_PC_FIFO_FULL, 0x4},              // stalled, PC FIFO full or nearly full
    {PmcEvent::BP_STALL_BP_FIFO_FULL, 0x5},              // stalled, predictor FIFO full or nearly full
    {PmcEvent::BP_BTB_HIT, 0x6},                         // branch target buffer hit
    {PmcEvent::BP_PRED_TYPE_JALR, 0x7},                  // predicted a jalr
    {PmcEvent::BP_PRED_TYPE_JAL, 0x8},                   // predicted a jal
    {PmcEvent::BP_PRED_TYPE_BR, 0x9},                    // predicted a conditional branch
    {PmcEvent::IF_STALL_ANY, 0x10},                      // instruction FIFO empty or not popped
    {PmcEvent::IF_STALL_NO_ID_RTR, 0x20},                // fetch has an instruction, decode not ready
    {PmcEvent::IF_STALL_LQ_FULL, 0x30},                  // fetch has an instruction, load queue full
    {PmcEvent::IF_STALL_LQ_SB_FULL, 0x40},               // fetch has an instruction, load skid buffer full
    {PmcEvent::IF_REQ_STALL_ANY, 0x50},                  // instruction request stalled
    {PmcEvent::IF_REQ_STALL_NO_MEM_RTR, 0x60},           // request stalled, instruction memory not ready
    {PmcEvent::DE_STALL_ANY, 0x100},                     // decode has an instruction but cannot issue it
    {PmcEvent::DE_STALL_NO_UNIT_RTR, 0x200},             // execution unit not ready
    {PmcEvent::DE_STALL_IF_EMPTY, 0x300},                // nothing to decode
    {PmcEvent::DE_STALL_SYNC, 0x400},                    // serializing operation draining the load queue
    {PmcEvent::DE_STALL_HAZARD, 0x500},                  // register hazard
    {PmcEvent::DE_HAZARD_INT, 0x600},                    // hazard on the integer register file
    {PmcEvent::DE_HAZARD_FP, 0x700},                     // hazard on the FP register file
    {PmcEvent::DE_HAZARD_VEC, 0x800},                    // hazard on the vector register file
    {PmcEvent::DE_REG_DATA_FORWARD, 0x900},              // operand forwarding used
    {PmcEvent::DE_MEM_OP_DISPATCH_LD, 0xA00},            // scalar load dispatched
    {PmcEvent::DE_MEM_OP_DISPATCH_ST, 0xB00},            // scalar store dispatched
    {PmcEvent::DE_MEM_INSTR_DISPATCH_VEC_LD, 0xC00},     // vector load dispatched
    {PmcEvent::DE_MEM_INSTR_DISPATCH_VEC_ST, 0xD00},     // vector store dispatched
    {PmcEvent::DE_EX_OP_DISPATCH_MUL, 0xE00},            // multiply dispatched
    {PmcEvent::DE_EX_OP_DISPATCH_DIV, 0xF00},            // divide dispatched
    {PmcEvent::DE1_EX_OP_DISPATCH_CSR, 0x1000},          // CSR, ecall, ebreak dispatched
    {PmcEvent::DE1_EX_OP_DISPATCH_I_EXT, 0x2000},        // base integer instruction dispatched
    {PmcEvent::DE1_EX_OP_DISPATCH_M_EXT, 0x3000},        // M extension (multiply, divide) dispatched
    {PmcEvent::DE1_EX_OP_DISPATCH_A_EXT, 0x4000},        // A extension (atomic) dispatched
    {PmcEvent::DE1_EX_OP_DISPATCH_ZB_EXT, 0x5000},       // Zb extension (bit manipulation) dispatched
    {PmcEvent::DE1_FP_OP_DISPATCH_S, 0x6000},            // single precision FP dispatched
    {PmcEvent::DE1_FP_OP_DISPATCH_ZFH, 0x7000},          // half precision FP dispatched
    {PmcEvent::DE1_VEC_OP_DISPATCH_CSR, 0x8000},         // vector configuration dispatched
    {PmcEvent::DE1_VEC_OP_DISPATCH_INT, 0x9000},         // vector integer dispatched
    {PmcEvent::DE1_VEC_OP_DISPATCH_FIXED, 0xA000},       // vector fixed point dispatched
    {PmcEvent::DE1_VEC_OP_DISPATCH_FLOAT, 0xB000},       // vector floating point dispatched
    {PmcEvent::DE1_VEC_OP_DISPATCH_LDST, 0xC000},        // vector load or store dispatched
    {PmcEvent::DE1_VALID_TT_INSTR, 0xD000},              // Tensix instruction issued by this core
    {PmcEvent::DE1_VALID_VEC_REPLAY, 0xE000},            // vector replay
    {PmcEvent::DE1_VALID_ILLEGAL, 0xF000},               // illegal instruction
    {PmcEvent::EX_STALL_ANY, 0x10000},                   // execute back-pressure; always zero on Blackhole, hazards resolve at decode
    {PmcEvent::EX_STALL_NO_INSTR, 0x20000},              // execute starved
    {PmcEvent::EX_MISPRED_ANY, 0x30000},                 // any branch mispredict
    {PmcEvent::EX_MISPRED_NEW_BR, 0x40000},              // mispredict of a branch not in the target buffer
    {PmcEvent::EX_MISPRED_DIR, 0x50000},                 // wrong direction
    {PmcEvent::EX_MISPRED_TGT_COND, 0x60000},            // wrong target, conditional
    {PmcEvent::EX_MISPRED_TGT_UNCOND, 0x70000},          // wrong target, unconditional
    {PmcEvent::EX_INST_RET, 0x80000},                    // instruction retired; equals minstret
    {PmcEvent::LS_STALL_ANY, 0x100000},                  // memory operation valid but not accepted
    {PmcEvent::LS_STALL_NO_MEM_RTR, 0x200000},           // memory port not ready
    {PmcEvent::LS_STALL_LDQ_FULL, 0x300000},             // load queue full
    {PmcEvent::LS_STALL_STORE_BUF_FULL, 0x400000},       // store waiting on a full write buffer
    {PmcEvent::LS_STALL_SKID_BUF_FULL, 0x500000},        // skid buffer full
    {PmcEvent::LS_FENCE_OP, 0x600000},                   // fence
    {PmcEvent::LS_STALL_WAIT_FOR_DATA, 0x700000},        // oldest load waiting for data
    {PmcEvent::LS_OP_RET, 0x800000},                     // load queue retire
    {PmcEvent::LS_STORE_REQ_WINS, 0x900000},             // store won arbitration over a load
    {PmcEvent::LS_WR_BUF_STLH, 0xA00000},                // store-to-load hazard in the write buffer
    {PmcEvent::LS_WR_BUF_STSC, 0xB00000},                // store coalescing
    {PmcEvent::LS_LOAD_OP, 0xC00000},                    // 32-bit load
    {PmcEvent::LS_STORE_OP, 0xD00000},                   // 32-bit store
    {PmcEvent::LS_VEC_LOAD_OP, 0xE00000},                // vector load
    {PmcEvent::LS_VEC_STORE_OP, 0xF00000},               // vector store
    {PmcEvent::TTCSR_SSTATUS7_HIGH, 0x1000000},          // status CSR 7 non-zero
    {PmcEvent::TTCSR_SSTATUS6_HIGH, 0x2000000},          // status CSR 6 non-zero
    {PmcEvent::TTCSR_SSTATUS5_HIGH, 0x3000000},          // status CSR 5 non-zero
    {PmcEvent::TTCSR_SSTATUS4_HIGH, 0x4000000},          // status CSR 4 non-zero
    {PmcEvent::TTCSR_SSTATUS3_HIGH, 0x5000000},          // status CSR 3 non-zero
    {PmcEvent::TTCSR_SSTATUS2_HIGH, 0x6000000},          // status CSR 2 non-zero
    {PmcEvent::TTCSR_SSTATUS1_HIGH, 0x7000000},          // status CSR 1 non-zero
    {PmcEvent::TTCSR_SSTATUS0_HIGH, 0x8000000},          // status CSR 0 non-zero
    {PmcEvent::TTCSR_BSTATUS_HIGH, 0x9000000},           // Tensix busy status non-zero
    {PmcEvent::TTCSR_QSTATUS_HIGH, 0xA000000},           // Tensix queue status non-zero
    {PmcEvent::MEM_STALL_ANY, 0x10000000},               // memory request not granted
    {PmcEvent::MEM_LOCAL_REG_RD_ONGOING, 0x20000000},    // local register read in flight
    {PmcEvent::MEM_MAILBOX_RD_ONGOING, 0x30000000},      // mailbox read in flight
    {PmcEvent::MEM_DATA_L1_RD_ONGOING, 0x40000000},      // L1 read in flight
    {PmcEvent::MEM_CFG_REG_RD_ONGOING, 0x50000000},      // Tensix config register read in flight
    {PmcEvent::MEM_REG_RD_ONGOING, 0x60000000},          // Tensix register read in flight
    {PmcEvent::MEM_TDMA_REG_RD_ONGOING, 0x70000000},     // TDMA register read in flight
    {PmcEvent::MEM_PC_BUF_RD_ONGOING, 0x80000000},       // PC buffer read in flight
    {PmcEvent::MEM_DATA_ACC_REQ_LOCAL, 0x90000000},      // local memory access accepted
    {PmcEvent::MEM_INS_BUF_WRITE, 0xA0000000},           // one Tensix instruction pushed
    {PmcEvent::MEM_INS_BUF_WRITE_MULTIPLE, 0xB0000000},  // multi-word Tensix instruction push
    {PmcEvent::MEM_DATA_RD_REQ_L1, 0xC0000000},          // L1 data read request; does not track this core's L1 reads on Blackhole
    {PmcEvent::MEM_DATA_RD_REQ_HIT_LOCACHE, 0xD0000000}, // local cache hit; does not track this core's L1 reads on Blackhole
    {PmcEvent::MEM_IF_ACCESS, 0xE0000000},               // instruction cache access
    {PmcEvent::MEM_IF_ACCESS_HIT, 0xF0000000}}};         // instruction cache hit
// clang-format on

} // namespace llk::perf::riscv
