// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTNOP_TENSIX_ISA_H
#define TTNOP_TENSIX_ISA_H

// Encodings needed to recognise detour sites in a compiled kernel's .text and to
// build the filler instructions that pad the cave. Shared by the scanner; the host
// side reads the same numbers back out of the scanner's JSON.

#include <cstdint>

struct SyncOp
{
    std::uint32_t opcode;
    const char* name;
};

// Sync / handshake ops. These are the default site set: every one of them is a
// point where a thread's timing is observable by another thread or by a backend unit.
constexpr SyncOp kSyncOps[] = {
    {0xa6u, "SEMWAIT"},
    {0xa5u, "SEMGET"},
    {0xa4u, "SEMPOST"},
    {0xa2u, "STALLWAIT"},
    {0xa0u, "ATGETM"},
    {0xa1u, "ATRELM"},
    {0x57u, "SETDVALID"},
    {0x36u, "CLEARDVALID"},
};

// MOP/REPLAY opcodes to avoid
constexpr std::uint32_t TT_OP_MOP     = 0x01u;
constexpr std::uint32_t TT_OP_NOP     = 0x02u;
constexpr std::uint32_t TT_OP_MOP_CFG = 0x03u;
constexpr std::uint32_t TT_OP_REPLAY  = 0x04u;
constexpr std::uint32_t TT_OP_SEMINIT = 0xa3u;

// RV32 instructions whose meaning depends on the instruction's own PC or
// fall-through address cannot be moved into the cave without relocation.
constexpr std::uint32_t RISCV_OPCODE_MASK = 0x7fu;
constexpr std::uint32_t RISCV_AUIPC       = 0x17u;
constexpr std::uint32_t RISCV_BRANCH      = 0x63u;
constexpr std::uint32_t RISCV_JALR        = 0x67u;
constexpr std::uint32_t RISCV_JAL         = 0x6fu;
constexpr std::uint32_t RISCV_SYSTEM      = 0x73u;

// TT_OP(opcode, params) packing (ckernel_ops.h): opcode in bits[31:24], params in [23:0].
constexpr unsigned TT_OP_OPCODE_SHIFT     = 24;
constexpr std::uint32_t TT_OP_OPCODE_MASK = 0xffu;
constexpr std::uint32_t TT_OP_PARAMS_MASK = 0xffffffu;

constexpr std::uint32_t TT_OP_STALLWAIT        = 0xa2u;
constexpr unsigned TT_STALLWAIT_STALL_SHIFT    = 15;
constexpr std::uint32_t TT_STALLWAIT_WAIT_MASK = 0x7fffu;
constexpr std::uint32_t P_STALL_SFPU           = 0x100u;
constexpr std::uint32_t P_STALL_WAIT_SFPU      = 0x4000u;

// TO determine what unpacker(s) the kernel loads with we OR CntSetMask from SETADCXX words.
// LLK only SETADCXXs the unpackers that read L1
//
// SETADCXX params layout
//   params bits:  23 22 21   | 20 ...... 10 | 9 ...... 0
//                 CntSetMask |    x_end     |  x_start
//   CntSetMask: bit0 = UNP0/SrcA, bit1 = UNP1/SrcB, bit2 = packer (ignored here)
constexpr std::uint32_t TT_OP_SETADCXX          = 0x5eu;
constexpr unsigned TT_SETADCXX_CNTSETMASK_SHIFT = 21;
constexpr std::uint32_t TT_SETADCXX_UNP_MASK    = 0x3u;

// .ttinsn stores TT_OP rotated left by 2; rotate right to recover the logical TT_OP word.
constexpr unsigned TTINSN_ROTATE_BITS = 2;

static std::uint32_t rotate_right_2(std::uint32_t word)
{
    return (word >> TTINSN_ROTATE_BITS) | (word << (32u - TTINSN_ROTATE_BITS));
}

// REPLAY params: bit0 = load_mode and bits[13:4] = len. if its in load mode
// skip the next 'len' instructions
constexpr std::uint32_t TT_REPLAY_LOAD_MODE = 0x1u;
constexpr unsigned TT_REPLAY_LEN_SHIFT      = 4;
constexpr std::uint32_t TT_REPLAY_LEN_MASK  = 0x3ffu;

// A .text word is a Tensix instruction iff its low two bits are not 0b11
inline bool is_tensix_word(std::uint32_t word)
{
    return (word & 3u) != 3u;
}

// Every SFPU op lives in this range; the SFPU nop is 0x8f. Anything here is an SFPU site.
constexpr std::uint32_t SFPU_OPCODE_FIRST = 0x70u;
constexpr std::uint32_t SFPU_OPCODE_LAST  = 0x95u;
constexpr std::uint32_t TT_OP_SFPNOP      = 0x8fu;

// Filler words as .text stores them (.ttinsn = TT_OP rotated left by 2). Verified
// encodings — only pure UNP_NOP mode for the unpackers (no ZEROSRC / SET_DVALID /
// NEGINFSRC side effects). WH/BH UNP_NOP is Nop_type=2; Quasar Nop_type=2 is
// UNP_NOP_SETDAVLID (sets SrcA/B dvalid and can stall on FPU). Quasar cycle delay
// is Nop_type=1 → TT_OP(0x43, (UNP_SEL<<8)|1).
constexpr std::uint32_t FILLER_TTI_NOP = 0x08000000u; // TTI_NOP
constexpr std::uint32_t FILLER_SFPNOP  = 0x3C000002u; // SFPNOP
constexpr std::uint32_t FILLER_UNPACR0 = 0x0C000005u; // quasar UNPACR_NOP UNP_A, Nop_type=1
constexpr std::uint32_t FILLER_UNPACR1 = 0x0C000405u; // quasar UNPACR_NOP UNP_B, Nop_type=1
// constexpr std::uint32_t FILLER_UNPACR0 = 0x0C000009u; // WH UNPACR_NOP unpacker 0 / SrcA
// constexpr std::uint32_t FILLER_UNPACR1 = 0x0E000009u; // WH UNPACR_NOP unpacker 1 / SrcB

// The only filler that costs the RISC a cycle without also costing the Tensix
// front-end one, so it is the only one that can shift a RISC MMIO write against
// the backend unit that consumes it. Plain RV32 `addi x0, x0, 0` which is what
// asm volatile("nop") assembles to.
constexpr std::uint32_t FILLER_RISC_NOP = 0x00000013u;

// Two STALLWAITs that bound an SFPU block — pack sfpnop here instead of tti:
//   start: STALLWAIT(STALL_SFPU, MATH)    // _llk_math_eltwise_sfpu_start_ — stall SFPU until math is done
//   done:  STALLWAIT(STALL_CFG, WAIT_SFPU) // _llk_math_eltwise_sfpu_done_  — stall CFG until SFPU is done
inline bool stallwait_touches_sfpu(std::uint32_t params)
{
    const std::uint32_t stall = params >> TT_STALLWAIT_STALL_SHIFT;
    const std::uint32_t wait  = params & TT_STALLWAIT_WAIT_MASK;
    // STALL_SFPU or WAIT_SFPU → SFPU nop
    return (stall & P_STALL_SFPU) != 0u || wait == P_STALL_WAIT_SFPU;
}

inline bool is_sfpu_opcode(std::uint32_t opcode)
{
    return opcode >= SFPU_OPCODE_FIRST && opcode <= SFPU_OPCODE_LAST;
}

// Moving an instruction into the cave changes its PC, so anything that reads its own
// address or redirects control flow has to stay put.
inline bool is_relocatable_riscv(std::uint32_t word)
{
    const std::uint32_t opcode = word & RISCV_OPCODE_MASK;
    return opcode != RISCV_AUIPC && opcode != RISCV_BRANCH && opcode != RISCV_JALR && opcode != RISCV_JAL && opcode != RISCV_SYSTEM;
}

// MOP and REPLAY reference an instruction buffer rather than executing in place, and
// SEMINIT/NOP are not interesting sites, so none of them can host a detour.
inline bool is_detourable_tensix(std::uint32_t opcode)
{
    return opcode != TT_OP_MOP && opcode != TT_OP_MOP_CFG && opcode != TT_OP_REPLAY && opcode != TT_OP_SEMINIT && opcode != TT_OP_NOP;
}

inline const char* sync_op_name(std::uint32_t opcode)
{
    for (const SyncOp& op : kSyncOps)
    {
        if (op.opcode == opcode)
        {
            return op.name;
        }
    }
    return nullptr;
}

#endif
