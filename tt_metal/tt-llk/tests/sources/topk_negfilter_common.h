// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// ============================================================================
// Value-preserving SFPU threshold filter for Blackhole Top-K (negative-threshold
// fallback for the packer-resident MIN_THRESHOLD_RELU path)
// ============================================================================
//
// WHAT IT COMPUTES, per 32-lane vector of Dst:
//
//     Dst[i] = (Dst[i] > Threshold) ? Dst[i] : 0x00000000
//
// where ">" is the SIGN-MAGNITUDE TOTAL ORDER
// (-NaN < -Inf < ... < -0 < +0 < ... < +Inf < +NaN) that SFPGT implements and
// that the packer's ReLU stage was measured to use. Survivors keep their EXACT
// bit pattern (no float datapath touches them, so no denormal flush and no NaN
// canonicalisation); losers become EXACTLY +0.0, which is what the packer's
// zero-compression needs in order to elide them.
//
// WHY THIS EXISTS
// ---------------
// The zero-SFPU path -- packer MIN_THRESHOLD_RELU + zero-compression -- cannot
// express a negative threshold: Packers/ReLU.md:41 makes signbit(Threshold)
// UndefinedBehavior. Signed logits (MoE routing, vocab sampling) need one.
//
// COST ANALYSIS (why two instructions per vector, and why not one)
// ---------------------------------------------------------------
// Per vector the work is: read Dst, compare against T, apply the result, write
// Dst. SFPLOADMACRO can retire a Load + one Simple + one MAD + one Round + one
// Store per issue, so the question is how few Simple/MAD slots the "compare and
// apply" needs.
//
//   (a) LaneFlags route -- Simple = SFPLE(SET_CC), MAD = predicated zeroing.
//       SFPLE's LaneFlags write is itself gated on LaneEnabled (SFPLE.md), so a
//       lane that fails once stays disabled: an SFPENCC restore is required
//       EVERY vector. SFPENCC is a Simple, and a macro schedules at most one
//       Simple, so the restore has to be a software instruction -- and it then
//       has to execute strictly between the MAD of vector i and the SFPLE of
//       vector i+1, which the 1-Simple-per-cycle sub-unit cannot provide
//       without a bubble. 2 issues/vector at best, with a same-cycle
//       LaneFlags race.
//
//   (b) Mask route (THIS ONE) -- SFPGT(SET_VD) produces 0xFFFFFFFF/0x00000000
//       (SFPGT.md: LReg[VD].i32 = IsVcSmaller ? -1 : 0), then SFPAND applies
//       it. No LaneFlags, so nothing is sticky and nothing has to be restored.
//       But SFPGT is 2-operand (VD is both the compared value and the result),
//       so the value has to live in a second register -- and the only exact
//       32-bit copy available is a second load. Two loads + two Simple + one
//       Store per vector; the load sub-unit and the Simple sub-unit each retire
//       one per cycle, so BOTH bind at 2 issues/vector.
//
//   (c) One issue/vector is impossible. Applying the mask needs a bitwise AND,
//       which lives only on the Simple sub-unit; producing the mask needs a
//       compare, also Simple. Two Simple ops cannot come from one macro. The
//       MAD sub-unit cannot substitute: it is a float datapath (SFPMAD would
//       flush denormals and canonicalise NaNs, and 0*(-Inf) is NaN rather than
//       +0.0), and SFPMUL24 -- the one exact integer MAD -- can only produce
//       zero, not a select.
//
// So: 2 SFPU issues per 32-element vector is the floor for a bit-exact
// value-preserving filter on this hardware. This header implements that floor.
//
// SCHEDULE (issue rate 1/cycle out of the MOP expander; c = issue cycle of the
// first macro of a vector)
//
//   c+0  CMP  macro: SFPLOAD v -> L_M(p)                      [load  sub-unit]
//   c+1  AND  macro: SFPLOAD v -> L_V(p)                      [load  sub-unit]
//        CMP's scheduled Simple: L_M(p) = (v > T) ? -1 : 0    [simple, delay 0]
//   c+2  (next vector's CMP issues)
//        AND's scheduled Simple: L_V(p) &= L_M(p)             [simple, delay 0]
//   c+3  AND's scheduled Store: Dst[addr of AND's load] = L_V(p)  [store, delay 1]
//
// The two loads address the SAME vector: CMP uses an addr_mod with dest.incr=0
// and AND uses dest.incr=2, so the pair advances the Dst walk exactly once. The
// macro-scheduled SFPSTORE writes to the address ITS OWN load used
// (SFPLOADMACRO.md), which is that same vector.
//
// L_M/L_V are ping-ponged across consecutive vectors (phase 0 -> L_M0/L_V0,
// phase 1 -> L_M1/L_V1) because a scheduled instruction of vector i executes
// after vector i+1 has issued: without the ping-pong, vector i+1's load would
// clobber L_V on the very cycle vector i's Store reads it.
//
// LaneFlags are taken out of the picture entirely at init with
// SFPENCC(Imm2=0, EI): UseLaneFlagsForLaneEnable := false makes every lane
// unconditionally enabled, so the SET_VD write of SFPGT and the write of
// SFPAND (both gated on LaneEnabled) can never be silently suppressed by state
// an earlier kernel left behind.

#pragma once

#include <cstdint>

#include "ckernel.h"
#include "ckernel_addrmod.h"
#include "ckernel_defs.h"
#include "ckernel_ops.h"
#include "ckernel_template.h"
#include "llk_defs.h"
#include "lltt.h"
#include "sfpu/ckernel_sfpu_load_config.h"

namespace topk_negfilter
{

// LReg map. Everything must be < 8: SFPGT's SET_VD write is gated on
// (VD < 8 || VD == 16) and SFPAND's write on the same condition.
constexpr std::uint32_t L_M0  = 0; // mask, phase 0
constexpr std::uint32_t L_V0  = 1; // value, phase 0
constexpr std::uint32_t L_M1  = 2; // mask, phase 1
constexpr std::uint32_t L_V1  = 3; // value, phase 1
constexpr std::uint32_t L_THR = 4; // threshold, whole-tile invariant

constexpr std::uint32_t SFPGT_MOD1_SET_VD     = 8; // SFPGT.md:53
constexpr std::uint32_t SFPENCC_MOD1_EI       = 2; // SFPENCC.md
constexpr std::uint32_t SFPCFG_IMM16_IS_VALUE = 1; // SFPCONFIG.md

// ADDR_MOD_0/2/3 belong to the A2D datacopy and ADDR_MOD_6/7 to the SFPU unary
// path (llk_math_eltwise_unary_sfpu.h:33-57), so 4 and 5 are the free slots.
constexpr std::uint32_t AM_STAY = ckernel::ADDR_MOD_4; // dest.incr = 0
constexpr std::uint32_t AM_WALK = ckernel::ADDR_MOD_5; // dest.incr = 2

// Macro indices. Two AND macros exist only because the SFPAND template carries
// the mask register in its VC field, which the macro override does not touch --
// so the ping-pong needs one template (and hence one macro) per phase.
constexpr std::uint32_t MACRO_CMP  = 2;
constexpr std::uint32_t MACRO_AND0 = 3;
constexpr std::uint32_t MACRO_AND1 = 1;

// Sequence bytes (SFPLOADMACRO.md:110-150).
//   Simple: 0x80 -> Insn.VB = macroVD (so SFPGT compares the loaded value and
//                   SFPAND reads it); 0x40 clear -> Insn.VD = macroVD.
//           selector 4/5/6 = InstructionTemplate[0]/[1]/[2].
//   Store : selector 3 = SFPSTORE; 0x40 and 0x80 both clear -> Insn.VD =
//           macroVD, i.e. it stores the register the AND just wrote.
constexpr std::uint32_t SEQ_CMP  = (0x80u | (0u << 3) | 4u);
constexpr std::uint32_t SEQ_AND0 = (0x80u | (0u << 3) | 5u) | (((1u << 3) | 3u) << 24);
constexpr std::uint32_t SEQ_AND1 = (0x80u | (0u << 3) | 6u) | (((1u << 3) | 3u) << 24);

// Misc (SFPLOADMACRO.md:53-57):
//   bits 4..7  UsesLoadMod0ForStore, one bit per macro -> macros 1 and 3 (the
//              two AND macros, the only ones that store) inherit the load's
//              INT32 mode, so the store is a raw 32-bit write with no format
//              conversion.
//   bits 8..11 UnitDelayKind, one bit per sub-unit -> Simple (8) and Store (11)
//              use WaitForElapsedInstructions, which counts SFPU issues rather
//              than wall cycles so a frontend bubble cannot slide a scheduled
//              instruction off its slot.
constexpr std::uint32_t MISC_WORD = 0x900u | 0xA0u;

// SFPLOADMACRO field packing (ckernel_ops.h:683, SFPLOADMACRO.md:20-26,45):
//   lreg_ind      = (MacroIndex << 2) | (VD & 3)
//   dest_reg_addr = (Imm9 << 1) | (VD >> 2)
#define NEGFILTER_MACRO(macro_idx, vd, addr_mod) TTI_SFPLOADMACRO(((macro_idx) << 2) | ((vd) & 3u), ckernel::InstrModLoadStore::INT32, (addr_mod), ((vd) >> 2))

// Write LoadMacroConfig::Sequence[idx]. The 32-bit word does not fit SFPCONFIG's
// 16-bit immediate path, so it is staged through LReg[0] -- the idiom of
// _init_mul_int_. LReg[0] is L_M0, which is scratch at this point and is
// reloaded by the first CMP macro anyway.
template <std::uint32_t IDX, std::uint32_t VALUE>
inline void write_sequence()
{
    TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_LOWER, VALUE & 0xFFFF);
    TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_UPPER, (VALUE >> 16) & 0xFFFF);
    TTI_SFPCONFIG(0, 4 + IDX, 0);
}

// One-time setup. MUST run after _llk_math_eltwise_unary_sfpu_init_once_(),
// which clears LaneConfig and so re-enables the VD >= 12 backdoor that the
// InstructionTemplate writes below depend on (SFPCONFIG.md:45-46).
inline void configure(std::uint32_t thr_bits)
{
    ckernel::addr_mod_t {
        .srca = {.incr = 0},
        .srcb = {.incr = 0},
        .dest = {.incr = 0},
    }
        .set(AM_STAY);

    // dest.incr = 2, not 4: the addr_mod dest field is in u10 Addr units where
    // bits [9:2] pick the 4-row group and bit 1 picks even-vs-odd columns, so
    // one SFPLOAD advances by 2.
    ckernel::addr_mod_t {
        .srca = {.incr = 0},
        .srcb = {.incr = 0},
        .dest = {.incr = 2},
    }
        .set(AM_WALK);

    // Imm2 = 0 with MOD1_EI -> UseLaneFlagsForLaneEnable := false, i.e. every
    // lane is unconditionally enabled and stays that way. Nothing in the loop
    // touches LaneFlags, so predication cannot go sticky.
    TTI_SFPENCC(0, 0, 0, SFPENCC_MOD1_EI);

    ckernel::sfpu::_sfpu_load_imm32_(L_THR, thr_bits);

    // InstructionTemplates, written through the VD >= 12 backdoor.
    //   [0] SFPGT: IsVcSmaller = SignMagIsSmaller(LReg[VC] = T, LReg[VB] = v)
    //              = (v > T); SET_VD writes -1/0 into LReg[VD].
    //   [1] SFPAND with VC = L_M0, [2] with VC = L_M1: LReg[VD] = LReg[VB] & LReg[VC].
    TTI_SFPGT(0, L_THR, 12, SFPGT_MOD1_SET_VD);
    TTI_SFPAND(0, L_M0, 13, 0);
    TTI_SFPAND(0, L_M1, 14, 0);

    write_sequence<MACRO_CMP, SEQ_CMP>();
    write_sequence<MACRO_AND0, SEQ_AND0>();
    write_sequence<MACRO_AND1, SEQ_AND1>();

    TTI_SFPCONFIG(MISC_WORD, 8, SFPCFG_IMM16_IS_VALUE);
    TTI_SFPNOP;
    TTI_SFPNOP;
}

// Record the 4-instruction body (two vectors) once and program the MOP that
// replays it. One MOP issue then feeds the backend at a guaranteed one
// instruction per cycle with the RISC-V off the critical path.
inline void program_replay()
{
    ckernel::load_replay_buf<ckernel::NoExec>(
        0,
        4,
        []
        {
            NEGFILTER_MACRO(MACRO_CMP, L_M0, AM_STAY);
            NEGFILTER_MACRO(MACRO_AND0, L_V0, AM_WALK);
            NEGFILTER_MACRO(MACRO_CMP, L_M1, AM_STAY);
            NEGFILTER_MACRO(MACRO_AND1, L_V1, AM_WALK);
        });
    ckernel::ckernel_unpack_template::lA(lltt::replay_insn(0, 4), TT_OP_NOP).program();
}

// 32 SFPLOADs cover one 32x32 tile (an SFPLOAD reads 4 consecutive Dst rows and
// one AM_WALK advance is 2 u10 Addr units, so 32 walks cover the tile's 64 Dst
// rows). The recorded body handles two vectors, so a tile is 16 passes.
constexpr std::uint32_t VECTORS_PER_TILE = 32;
constexpr std::uint32_t PASSES_PER_TILE  = VECTORS_PER_TILE / 2;
static_assert(PASSES_PER_TILE <= 128, "ckernel_unpack_template::run loop_count is 7 bits");

// One tile of filtering. Caller must have re-pointed the Dst window (i.e. be
// inside _llk_math_eltwise_sfpu_start_ / _done_).
inline void run_tile()
{
    ckernel::ckernel_unpack_template::run(PASSES_PER_TILE);
    // Drain the final macro's scheduled Simple (+1) and Store (+2 issues) before
    // the Dst base moves. Each SFPNOP is an SFPU issue, so it both advances the
    // WaitForElapsedInstructions counters and occupies a cycle.
    TTI_SFPNOP;
    TTI_SFPNOP;
    TTI_SFPNOP;
    TTI_SFPNOP;
}

} // namespace topk_negfilter
