// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel.h"
#include "ckernel_addrmod.h"
#include "ckernel_defs.h"
#include "ckernel_instr_params.h"
#include "ckernel_ops.h"
#include "cmath_common.h"
#include "lltt.h"
#include "sfpi.h"
#include "sfpu/ckernel_sfpu_load_config.h"

using namespace sfpi;

namespace ckernel {
namespace sfpu {

// Fast bf16 abs for Blackhole. Origin: llk-bench (LLM-agent-written kernel, claude-fable-5, 2026-08),
// validated exhaustively over all 65,536 bf16 inputs vs the exact golden: max 0 ULP (gate <= 0).
// Measured 47.1 cycles/tile vs 146.1 for the previous kernel on p150b (tt-metal v0.76.0 baseline).
// SFPU state programmed by the init: programmable constant LREG[11] = 0x00007fff, SFPLOADMACRO
// InstructionTemplate[0] (SFPAND) + LoadMacroConfig.Sequence[0] (macro 0) + Misc, and math-thread replay
// slots 0..7 (the per-face body: 8 SFPLOADMACROs). bf16 DEST only (LO16 raw loads) -- gated on
// !is_fp32_dest_acc_en below; the sfpi loop stays as the fallback for every other configuration.
//
// abs of a bf16 is a pure bit operation: clear bit 15. DST holds bf16 (16-bit rows), so each vector is
// loaded raw with InstrModLoadStore::LO16, ANDed with 0x7fff, and stored back -- exact for every input bit
// pattern (denormals, infinities; NaN inputs are don't-care).
//
// Blackhole's SFPLOADMACRO issues the load and schedules the AND (Simple sub-unit) and the SFPSTORE (Store
// sub-unit) for the following cycles, so the whole load->and->store chain costs a single issued instruction
// per vector (vs 3 for the plain SFPLOAD/SFPABS/SFPSTORE baseline):
//
//   t   | Load unit          | Simple unit             | Store unit
//   ----|--------------------|-------------------------|---------------------
//   0   | L0 = Dst[a] (LO16) |                         |
//   1   | (next vector)      | L16 = L0 & L11 (0x7fff) |
//   2   |                    |                         | Dst[a] = L16 (LO16)
//
// The AND result is staged in LReg[16] (writable only by macros), so back-to-back macros re-using L0 as the
// load target never clobber a value before its store reads it. All delays are cycle-based
// (UnitDelayKind = WaitForElapsedCycles), so trailing stores drain on their own -- no SFPNOP padding needed
// at face/tile boundaries.
#ifndef DISABLE_SFPLOADMACRO
inline void _init_abs_bf16_fast_() {
    // Programmable constant LReg[11] = 0x00007fff (bf16 sign-bit clear mask).
    TTI_SFPLOADI(p_sfpu::LREG0, sfpi::SFPLOADI_MOD0_USHORT, 0x7fff);
    TTI_SFPCONFIG(0, 11, 0);

    // InstructionTemplate[0] = SFPAND(VB, VC=L11, VD, USE_VB). Issuing with VD=12 backdoor-writes the
    // instruction bits into template slot 0 instead of executing it. The macro substitutes VB = load target
    // and VD = L16.
    TTI_SFPAND(0, 11, 12, 1);

    // Macro 0 sequence, one byte per sub-unit {store, round, mad, simple}:
    //   simple = 0x80 (VB <- load reg) | 0x40 (VD <- L16) | delay 0 | 4 (template 0) = 0xc4
    //   store  = 0x40 (VD <- L16) | delay 1 | 3 (SFPSTORE)                           = 0x4b
    TTI_SFPLOADI(p_sfpu::LREG0, sfpi::SFPLOADI_MOD0_LOWER, 0x00c4);
    TTI_SFPLOADI(p_sfpu::LREG0, sfpi::SFPLOADI_MOD0_UPPER, 0x4b00);
    TTI_SFPCONFIG(0, 4, 0);

    // Misc: UsesLoadMod0ForStore for macro 0 (store reuses the load's LO16 mod and address);
    // UnitDelayKind = 0 -> delays count elapsed cycles.
    TTI_SFPCONFIG(0x010, 8, 1);

    // Record the per-face body in the math thread's replay buffer: one face = 8 vectors of 32 lanes at Dst
    // offsets 0,2,...,14 (the per-face base advances via the caller's RWC increments). Each SFPLOADMACRO
    // triggers macro 0 with load target L0. Nothing else in the unary SFPU math thread (datacopy A2D, the
    // sfpu params wrapper) records replay slots.
    lltt::record(0, 8);
    TTI_SFPLOADMACRO(0, InstrModLoadStore::LO16, ADDR_MOD_7, 0);
    TTI_SFPLOADMACRO(0, InstrModLoadStore::LO16, ADDR_MOD_7, 2);
    TTI_SFPLOADMACRO(0, InstrModLoadStore::LO16, ADDR_MOD_7, 4);
    TTI_SFPLOADMACRO(0, InstrModLoadStore::LO16, ADDR_MOD_7, 6);
    TTI_SFPLOADMACRO(0, InstrModLoadStore::LO16, ADDR_MOD_7, 8);
    TTI_SFPLOADMACRO(0, InstrModLoadStore::LO16, ADDR_MOD_7, 10);
    TTI_SFPLOADMACRO(0, InstrModLoadStore::LO16, ADDR_MOD_7, 12);
    TTI_SFPLOADMACRO(0, InstrModLoadStore::LO16, ADDR_MOD_7, 14);
}

// One face (8 dst vectors): replay the 8 SFPLOADMACROs recorded by _init_abs_bf16_fast_().
inline void _calculate_abs_bf16_fast_() { lltt::replay(0, 8); }
#endif

template <bool is_fp32_dest_acc_en>
inline void abs_init() {
    // Common SFPU init inlined (SFPU config register + ADDR_MOD_7 + counter reset) so this init is
    // self-contained (same convention as exp_init; abs uses only ADDR_MOD_7, no op-specific ADDR_MOD_6).
    sfpu::_init_sfpu_config_reg();
    addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 0}}.set(ADDR_MOD_7);
    math::reset_counters(p_setrwc::SET_ABD_F);
#ifndef DISABLE_SFPLOADMACRO
    if constexpr (!is_fp32_dest_acc_en) {
        _init_abs_bf16_fast_();
    }
#endif
}

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en, int ITERATIONS = 8>
inline void calculate_abs() {
#ifndef DISABLE_SFPLOADMACRO
    if constexpr (!is_fp32_dest_acc_en && ITERATIONS == 8) {
        _calculate_abs_bf16_fast_();
        return;
    }
#endif
    // SFPU microcode
    for (int d = 0; d < ITERATIONS; d++) {
        vFloat v = dst_reg[0];
        dst_reg[0] = sfpi::abs(v);
        dst_reg++;
    }
}

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_abs_int32() {
    // SFPU microcode
    for (int d = 0; d < ITERATIONS; d++) {
        // sfpi::abs(vInt) lowers to the dedicated SFPABS integer instruction (mod 0),
        // matching the raw TTI sequence (SFPLOAD + SFPABS + SFPSTORE = 3 SFPU ops).
        // On Blackhole INT32_2S_COMP load/store is a no-op vs INT32, so I32 access is
        // byte-for-byte equivalent to the previous mode-12 access. abs() yields a vMag,
        // which stores through the M32 layout.
        sfpi::vInt v = sfpi::dst_reg[0].mode<sfpi::DataLayout::I32>();
        sfpi::dst_reg[0].mode<sfpi::DataLayout::M32>() = sfpi::abs(v);
        sfpi::dst_reg++;
    }
}

}  // namespace sfpu
}  // namespace ckernel
