// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// ============================================================================
//  Pricing `_topk_xl_rebuild_` on Blackhole -- decomposition + candidates
// ============================================================================
//
// WHY THIS FILE EXISTS
// --------------------
// `perf_topk_merge_macro.py` established that the shipping K-reduction step is
//
//     _topk_xl_merge_<512>    91.0 cyc/call    2.844 cyc/vector (32 vectors in)
//     _topk_xl_rebuild_<512> 374.0 cyc/call   23.375 cyc/vector (16 vectors out)
//     step (merge+rebuild)   464.0 cyc/call   14.500 cyc/vector
//
// i.e. the rebuild is 81% of the step. The merge was then beaten 1.978x by
// scheduling its SFPSWAP into an SFPLOADMACRO Simple slot, which moved the step
// only 1.107x. Everything left is in the rebuild, and the rebuild had never
// been instruction-counted.
//
// WHAT THE REBUILD IS, ALGORITHMICALLY
// ------------------------------------
// The names are misleading. `_topk_xl_merge_` is not a merge: it is ONE
// compare-exchange level -- the first level of a bitonic merge of two sorted
// K-runs -- with the min half discarded. `_topk_xl_rebuild_` is levels
// 2..log2(2K) of that same bitonic merge, run on the K survivors.
//
// For K = 512: 2K = 1024, so the full merge network is log2(1024) = 10 levels.
// Level 1 is `_topk_xl_merge_`. Levels 2..10 -- NINE levels -- are the rebuild.
// Nine levels on 512 elements = 16 vectors of 32 lanes = 8 vector
// compare-exchanges per level = 72 SFPSWAP. The census below confirms exactly
// 72. So the rebuild is not doing redundant work: 9 levels is the information-
// theoretic depth of a bitonic merge at this K, and 72 SFPSWAP is the exact
// comparator count for that depth.
//
// INSTRUCTION CENSUS, `_topk_xl_rebuild_<512, false, true>` (K != 2048 routes
// to `_topk_xl_rebuild_generic_`, row_scale_factor = 1, N = 2 faces):
//
//   phase                            instrs                       cycles
//   ------------------------------------------------------------------------
//   STALLWAIT + enter cfg block      1 + 3                          4
//   transpose_N_faces<2>             2 x transpose_dest_face_32b   ~62
//   build: stride-2 + sort_16_alt    2 iters x 34                 ~100 + push
//   transpose_N_faces<2>             2 x transpose_dest_face_32b   ~62
//   leave cfg block                  3                              3
//   2 x canonical_big_block<1>       2 x (8 ld + 20 swap + 2 tr + 8 st) ~118
//   mop restore + final SETRWC       ~7                             ~7
//   ------------------------------------------------------------------------
//                                                                  ~370
//
// SFPU side, priced with the measured primitives (SFPLOAD/SFPSTORE 1 cyc,
// SFPSWAP 2 cyc with a non-fillable bubble (SFPSWAP.md:110), SFPTRANSP 1 cyc):
//
//   32 SFPLOAD    32 cyc      (16 vectors read twice -- two passes)
//   32 SFPSTORE   32 cyc      (16 vectors written twice)
//   72 SFPSWAP   144 cyc      (9 levels x 8 vector compare-exchanges)
//    8 SFPTRANSP   8 cyc
//   --------------------
//  144 instrs    216 cyc      <-- the SFPU floor for this algorithm
//
// 374 - 216 = 158 cycles are NOT compare-exchange or its load/store. ~130 of
// those are the four `transpose_dest_face_32b` calls and their CFG traffic;
// ~28 are envelope (replay pushes, two MOP template programs, SETRWC, SETC16).
//
// WHY THE Dst TRANSPOSES ARE THERE, AND WHY THEY ARE UNAVOIDABLE IN THIS SHAPE
// ---------------------------------------------------------------------------
// One LReg is 4 rows x 8 lanes. `SFPSWAP` compares two LRegs lane-for-lane;
// `SFPTRANSP` swaps the register-index axis with the row axis (both length 4,
// SFPTRANSP.md functional model -- "the data movement is purely column-wise").
// NEITHER instruction can move data along the 8-lane axis. `SFPSHFT2` rotates
// by +/-1 within a group of 8 lanes, which cannot express strides 2 or 4 in one
// issue. So of the 9 levels the rebuild owes, the 3 whose stride falls on the
// LANE axis are unreachable by the Vector Unit at all. The only mechanism that
// moves Dst columns into Dst rows is the Matrix Unit face transpose
// (MOVD2B / TRNSPSRCB / MOVB2A / MOVB2D / MOVA2D), and it has to be paid twice
// -- in and back out.
//
// ARMS
// ----
//   CtrlLoad    control -- replay+MOP-fed plain SFPLOAD.  MUST be ~1.000/vector.
//   CtrlSwap    control -- replay+MOP-fed plain SFPSWAP.  MUST be ~2.000/vector.
//               Shared verbatim with perf_topk_merge_macro.py so the two
//               harnesses are stitched. If CtrlSwap is not 2.00x CtrlLoad the
//               run is INVALID.
//   RbCall      `_topk_xl_rebuild_<512,false,true>` -- reproduces the 374.
//   RbXposeFace ONE `transpose_dest_face_32b<0>` in the rebuild's own bracket.
//   RbXposeN    the rebuild's ENTIRE transpose content: both
//               `transpose_N_faces<2>` sweeps in one shared CFG block.
//   RbBuild     the stride-2 + sort_16_alt build phase (levels reached in the
//               transposed domain), MOP template and all.
//   RbBlock     the two `canonical_big_block_with_replay<1>` columns.
//   RbXposeNFlat  RbXposeN with the per-face CFG writes hoisted: one
//               SrcA-format switch per PASS instead of per FACE, parking all
//               faces' lo16 halves in distinct SrcA row ranges.
//   RbBuildMacro  RbBuild with the FIRST level of `sort_16_alt` scheduled into
//               the loads' SFPLOADMACRO Simple slots.
//   RbBlockMacro  RbBlock with the first level (Step 5) of `bitonic_sort_len_32`
//               scheduled the same way.
//
// PREDICTED vs MEASURED (Blackhole silicon, MATH_ISOLATE, two-point slope over
// REBUILD_ITER_COUNT, 5 runs/point). Every prediction below was recorded before
// the first run.
//
//   arm            cyc/call   cyc/vec   PREDICTED
//   CtrlLoad          1.997     0.999     1.000   control, frontend floor
//   CtrlSwap          3.997     1.999     2.000   THE TRIPWIRE, 2.00x CtrlLoad
//   RbCall          374.000    23.375   374       reproduces perf_topk_merge_macro
//   RbXposeFace      44.000     2.750    39       one 16x16 32-bit face + bracket
//   RbXposeN        143.000     8.938   132       the rebuild's WHOLE transpose
//   RbBuild         102.000     6.375   111       stride-2 + sort_16_alt
//   RbBlock         120.000     7.500   120       2 x canonical_big_block<1>
//   RbXposeNFlat    197.000    12.312   122       REFUTED: +54, not -21
//   RbXposeNFill    142.996     8.937   143       24 SFPNOPs cost EXACTLY ZERO
//   RbBuildMacro     90.000     5.625    90       -12
//   RbBlockMacro    108.000     6.750   108       -12
//   RbCallMacro     367.025    22.939   350       template swap NOT hidden
//   RbCallSched     350.000    21.875   350       THE CANDIDATE
//
//     rebuild : 374 -> 350 = 1.069x
//     step    : (91 + 374) = 465 -> (46 + 350) = 396 = 1.174x
//
//   RbXposeN + RbBuild + RbBlock = 365 against RbCall's 374; the missing 9 is
//   the second MOP template program and the SETRWCs only the full call pays. So
//   the decomposition is complete:
//
//     Dst transposes   143 cyc  38%
//     lattice + ld/st  222 cyc  59%   <- at its floor, see the census above
//     envelope           9 cyc   2%
//
// CORRECTNESS IS NOT ESTABLISHED BY ANY OF THIS. A misconfigured -- or all-zero,
// i.e. "schedule nothing" -- LoadMacroConfig.Sequence degenerates an
// SFPLOADMACRO into a plain SFPLOAD, and losing a whole bitonic level makes the
// arm look FASTER. `tests/python_tests/test_topk_rebuild_macro.py` is what
// establishes it: 74/74 against the shipping torch golden at K = 512/1024/2048
// in both sort directions, plus a mutation control.
//
// AND THE FIRST VERSION OF THIS WAS WRONG IN A WAY TIMING COULD NOT SEE. With
// the four SFPLOADMACROs issued back to back, their scheduled SFPSWAPs -- two
// cycles each, and exempt from the automatic post-SFPSWAP stall
// (SFPSWAP.md CAUTION) -- overlapped and corrupted each other. It measured
// FASTER (86 and 104 instead of 90 and 108) and passed every single-merge test,
// because the rebuild only PERMUTES its K survivors: one merge+rebuild cannot
// change the returned SET no matter how broken the sort is. It took
// test_topk_xl_fused_reduce with num_chunks=4 -- three chained merge+rebuild
// pairs, so a mis-ordered run becomes the next merge's supposedly-sorted
// operand -- to turn it red. Hence the interleaved plain/macro load order and
// the two drain SFPNOPs in every macro body below.
//
// THE ~1.4x MACRO BOUND, AND WHY IT IS WRONG
// ------------------------------------------
// The standing estimate was that the merge's macro trick, applied to
// `sort_16_alt`, could hide "at most half of the 16 SFPSWAPs behind 8 loads",
// bounding at ~1.4x. It cannot hide 8. It can hide FOUR, and the reason is
// structural, not a scheduling detail:
//
//   A macro-scheduled Simple instruction must have VD == macroVD, and macroVD
//   is the register THAT MACRO'S OWN LOAD just wrote (SFPLOADMACRO.md:111-115;
//   the macro always overrides Insn.VD). So a macro-scheduled SFPSWAP can only
//   express a compare-exchange whose destination is a freshly loaded register
//   AND whose other operand (the template's fixed VC, kept by setting sequence
//   bit 0x80) is also already loaded and NOT YET MODIFIED.
//
//   Only the FIRST level of any lattice satisfies that. Level 2 reads level 1's
//   outputs, which live in registers that have already been loaded -- there is
//   no load left to attach them to, and reloading them would cost the very
//   instruction the macro was meant to save.
//
// So the ceiling is one level per load-pass, i.e. 4 SFPSWAPs per 8-register
// body, independent of how deep the lattice is:
//
//   build  (sort_16_alt,  4 levels, 16 swaps/iter): 4 hidden -> 8 cyc/iter
//   block  (sort_len_32,  5 levels, 20 swaps/col ): 4 hidden -> 8 cyc/col
//
//   4 swaps x 2 cyc - 2 drain SFPNOP = 6 cyc per body, 4 bodies per rebuild
//   = 24 cycles off 374 = 1.069x, not 1.4x.
//
// Ordering constraint the arms below respect: the swap's VC operand must be
// loaded BEFORE the macro that carries it, so the load order is permuted to
// (VC-side register first, VD-side register as the macro).
//
// DEVICE / RUN NOTES. Blackhole silicon only. Read CtrlLoad and CtrlSwap FIRST.
// The rebuild reaches Dst strides > 256 through `transpose_N_faces`, whose
// MOVD2B/MOVB2A stall MATH on SrcB valid -- the UNPACK thread below issues one
// `_llk_unpack_set_srcb_dummy_valid_()` per iteration for every arm that
// transposes. Without it the math thread hangs (TENSIX TIMED OUT).

#include <cstdint>

#include "ckernel.h"
#include "ckernel_addrmod.h"
#include "ckernel_defs.h"
#include "ckernel_instr_params.h"
#include "ckernel_ops.h"
#include "ckernel_template.h"
#include "counters.h" // START_PERF_MEASURE (counters.h:616)
#include "llk_defs.h"
#include "lltt.h"
#include "params.h"
#include "perf.h"
#include "profiler.h"

std::uint32_t unp_cfg_context          = 0;
std::uint32_t pack_sync_tile_dst_ptr   = 0;
std::uint32_t math_sync_tile_dst_index = 0;

// ---------------------------------------------------------------------------
// Parameters. Both MUST be #define (not constexpr): the kernel guards each with
// #ifndef and a constexpr does not satisfy a preprocessor guard -- every swept
// variant would compile identically while still hashing to a distinct id.
// ---------------------------------------------------------------------------
#define ARM_CTRL_LOAD_ID      0
#define ARM_CTRL_SWAP_ID      1
#define ARM_RB_CALL_ID        2
#define ARM_RB_XPOSE_FACE_ID  3
#define ARM_RB_XPOSE_N_ID     4
#define ARM_RB_BUILD_ID       5
#define ARM_RB_BLOCK_ID       6
#define ARM_RB_XPOSE_FLAT_ID  7
#define ARM_RB_BUILD_MACRO_ID 8
#define ARM_RB_BLOCK_MACRO_ID 9
#define ARM_RB_XPOSE_FILL_ID  10
#define ARM_RB_CALL_MACRO_ID  11
#define ARM_RB_CALL_SCHED_ID  12

#ifndef REBUILD_ARM
#define REBUILD_ARM 2
#endif

#ifndef REBUILD_ITER_COUNT
#define REBUILD_ITER_COUNT 32
#endif

// Arms whose body drives the Matrix Unit through SrcB and therefore needs an
// unpack-side dummy valid per iteration.
#if REBUILD_ARM == ARM_RB_CALL_ID || REBUILD_ARM == ARM_RB_XPOSE_FACE_ID || REBUILD_ARM == ARM_RB_XPOSE_N_ID || REBUILD_ARM == ARM_RB_XPOSE_FLAT_ID || \
    REBUILD_ARM == ARM_RB_XPOSE_FILL_ID || REBUILD_ARM == ARM_RB_CALL_MACRO_ID || REBUILD_ARM == ARM_RB_CALL_SCHED_ID
#define REBUILD_NEEDS_SRCB_VALID 1
#else
#define REBUILD_NEEDS_SRCB_VALID 0
#endif

namespace
{
[[maybe_unused]] constexpr std::uint32_t XL_K = 512;
[[maybe_unused]] constexpr bool XL_FUSED      = true;
[[maybe_unused]] constexpr bool TOPK_APPROX   = false;

// Control arms only.
constexpr std::uint32_t L_CTRL_A = ckernel::p_sfpu::LREG0;
constexpr std::uint32_t L_CTRL_B = ckernel::p_sfpu::LREG1;

constexpr std::uint32_t SFPENCC_MOD1_EI       = 2; // SFPENCC.md:41
constexpr std::uint32_t SFPCFG_IMM16_IS_VALUE = 1; // SFPCONFIG.md:108

// --- SFPLOADMACRO sequence words -------------------------------------------
//
// Same shape as the merge's, and for the same documented reasons:
//   Simple byte 0x80 SET   -> Insn.VB = macroVD, leaving Insn.VC at the
//                             template's value (SFPLOADMACRO.md:100). This is
//                             the ONLY way to get a software-loaded register in
//                             as the compare's second operand; with 0x80 clear
//                             the macro assigns Insn.VC = macroVD and SFPSWAP
//                             degenerates to a self-compare.
//   Simple byte 0x40 CLEAR -> Insn.VD = macroVD.
//   selector 4+m           -> InstructionTemplate[m].
//   delay 0                -> the cycle right after the SFPLOADMACRO, consuming
//                             the value that macro just loaded.
//
// UNLIKE the merge, Mod1 stays at p_sfpswap::ALL_ROWS_MAX (= 1, "VD = min and
// VC = max"). The merge needed the undocumented Mod1 = 9 because it wanted the
// MAX in macroVD so a macro-scheduled SFPSTORE could write it out. Here NOTHING
// is stored from the macro: this is an interior lattice level, both halves stay
// live in registers, and the shipping lattice's own operand order already puts
// min in VD. Preserving Mod1 = 1 is what makes the macro body bit-identical in
// semantics to the instruction it replaces.
constexpr std::uint32_t seq_simple(std::uint32_t m)
{
    return 0x80u | (0u << 3) | (4u + m);
}

// MAD byte: selector 2 = SFPNOP. Required, not decorative -- SFPLOADMACRO.md:11
// footnote (‡): "If SFPSWAP is scheduled to the Simple sub-unit, then SFPNOP
// needs to be scheduled to the MAD sub-unit for the same time".
constexpr std::uint32_t SEQ_MAD = (0u << 3) | 2u;

// Round byte 0 = schedule nothing; also discharges the other half of (‡).
constexpr std::uint32_t SEQ_ROUND = 0u;

// Store byte 0 = schedule nothing. The merge could ride a store on its macro
// because its result was final at the load. Here the register is rewritten by
// three more lattice levels before it may be stored, and the Store slot's delay
// field is 2 bits (max 3 cycles), so no store can ride these macros.
constexpr std::uint32_t SEQ_STORE = 0u;

constexpr std::uint32_t sequence_word(std::uint32_t m)
{
    return (SEQ_STORE << 24) | (SEQ_ROUND << 16) | (SEQ_MAD << 8) | seq_simple(m);
}

// Misc (SFPLOADMACRO.md:53-57): StoreMod0 [0:3], UsesLoadMod0ForStore [4:7],
// UnitDelayKind [8:11].
//   0xF0  -> stores inherit the load's Mod0 (INT32). Inert here (no store is
//            scheduled) but kept identical to the merge's so the two macro
//            configurations differ in exactly one field, the Sequence.
//   0x300 -> Simple (bit 8) and MAD (bit 9) on WaitForElapsedInstructions.
//            Instruction-counting rather than cycle-counting, so the delay-0
//            Simple cannot slide if the frontend bubbles at a MOP boundary.
constexpr std::uint32_t MISC_WORD_REBUILD = 0x300u | 0xF0u;

// SFPLOADMACRO field packing (ckernel_ops.h:689, SFPLOADMACRO.md:20-26,45):
//   lreg_ind      = (MacroIndex << 2) | (VD & 3)
//   dest_reg_addr = (Imm9 << 1) | (VD >> 2)
// VD is u3 so the split is exact; SFPLOAD.md:83 -- the address low bit is
// unused, so rotating VD across 0..7 does not perturb the address.
#define RB_LOADMACRO(macro_idx, vd, addr_mod, off) \
    TTI_SFPLOADMACRO(((macro_idx) << 2) | ((vd) & 3u), ckernel::InstrModLoadStore::INT32, (addr_mod), (off) | ((vd) >> 2))

// MOP iteration ceiling. TT_OP_MOP's loop_count field is SEVEN bits
// (ckernel_ops.h:276) while `count` is a uint8_t -- passing 256 silently
// truncates to 0 and the arm reads out as a spectacular fake result.
constexpr std::uint32_t MOP_MAX_ITERS = 128;
constexpr std::uint32_t FULL_RUNS     = REBUILD_ITER_COUNT / MOP_MAX_ITERS;
constexpr std::uint32_t REM_PASSES    = REBUILD_ITER_COUNT % MOP_MAX_ITERS;

inline void mop_run_all()
{
    for (std::uint32_t i = 0; i < FULL_RUNS; ++i)
    {
        ckernel::ckernel_unpack_template::run(MOP_MAX_ITERS);
    }
    if constexpr (REM_PASSES > 0)
    {
        ckernel::ckernel_unpack_template::run(REM_PASSES);
    }
}
} // namespace

#ifdef LLK_TRISC_MATH

#include "llk_math_common.h"
#include "llk_math_eltwise_sfpu_common.h"
#include "llk_math_eltwise_unary_sfpu.h"
#include "sfpu/experimental/ckernel_sfpu_topk_xl.h"

namespace
{
using namespace ckernel;

// ---------------------------------------------------------------------------
// Macro configuration. Four macros, four templates, one shared Misc word.
// MUST run after _llk_math_eltwise_unary_sfpu_init_once_(), which clears
// LaneConfig -- the VD >= 12 backdoor that stores an instruction into
// InstructionTemplate[] instead of executing it is gated on
// LaneConfig.DISABLE_BACKDOOR_LOAD being false (SFPCONFIG.md:45-46, :120).
//
// `vc[m]` is the template's fixed second operand for macro m. It differs
// between the build arm and the block arm (different first lattice level), so
// the caller passes it in.
// ---------------------------------------------------------------------------
inline void configure_rebuild_macros(const std::uint32_t vc0, const std::uint32_t vc1, const std::uint32_t vc2, const std::uint32_t vc3)
{
    // InstructionTemplate[m] = SFPSWAP whose VC is the m-th partner register.
    // lreg_dest = 12 + m is the backdoor index, NOT an operand: it selects which
    // template slot receives the word. The macro overrides Insn.VD with macroVD
    // at issue time, so the 12..15 written here never reaches the Vector Unit.
    TT_SFPSWAP(0, vc0, 12, p_sfpswap::ALL_ROWS_MAX);
    TT_SFPSWAP(0, vc1, 13, p_sfpswap::ALL_ROWS_MAX);
    TT_SFPSWAP(0, vc2, 14, p_sfpswap::ALL_ROWS_MAX);
    TT_SFPSWAP(0, vc3, 15, p_sfpswap::ALL_ROWS_MAX);

    // Sequence[m]. The word does not fit the 16-bit immediate path, so stage it
    // through LReg[0] and write with Mod1 = 0 -- the ckernel_sfpu_mul_int.h
    // idiom. LReg[0] is clobbered, which is harmless: every body below seeds
    // all eight LRegs from loads before touching them.
    TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_LOWER, sequence_word(0) & 0xFFFF);
    TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_UPPER, (sequence_word(0) >> 16) & 0xFFFF);
    TTI_SFPCONFIG(0, 4 + 0, 0);

    TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_LOWER, sequence_word(1) & 0xFFFF);
    TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_UPPER, (sequence_word(1) >> 16) & 0xFFFF);
    TTI_SFPCONFIG(0, 4 + 1, 0);

    TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_LOWER, sequence_word(2) & 0xFFFF);
    TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_UPPER, (sequence_word(2) >> 16) & 0xFFFF);
    TTI_SFPCONFIG(0, 4 + 2, 0);

    TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_LOWER, sequence_word(3) & 0xFFFF);
    TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_UPPER, (sequence_word(3) >> 16) & 0xFFFF);
    TTI_SFPCONFIG(0, 4 + 3, 0);

    TTI_SFPCONFIG(MISC_WORD_REBUILD, 8, SFPCFG_IMM16_IS_VALUE);
    TTI_SFPNOP;
    TTI_SFPNOP;
}

// ===========================================================================
//  Candidate 1 -- macro-scheduled first level of the build phase
// ===========================================================================
//
// Shipping body (`load16_rows_x2<2>` + `bitonic_sort_len_16_alt_swaps` x2 with
// two SFPTRANSPs), descending branch (dir = false, which is what the shipping
// rebuild is called with in the reduction step):
//
//   loads   LREG0..7 <- Dst + {0,4,8,12, 2,6,10,14}
//   Step 4  SWAP(VC=0,VD=2) (1,3) (4,6) (5,7)     <-- FIRST LEVEL, macro-able
//   Step 3  SWAP(VC=0,VD=1) (2,3) (4,5) (6,7)
//   SFPTRANSP
//   Step 4  SWAP(VC=0,VD=2) (1,3) (4,6) (5,7)
//   Step 3  SWAP(VC=0,VD=1) (2,3) (4,5) (6,7)
//   SFPTRANSP
//
// The four Step-4 swaps have VD in {2,3,6,7} and VC in {0,1,4,5}. So load
// LREG0, LREG1, LREG4, LREG5 first as plain SFPLOADs, then issue LREG2, LREG3,
// LREG6, LREG7 as SFPLOADMACROs carrying the swap. Load ADDRESSES are unchanged
// -- only the ORDER of the eight loads is permuted, and SFPLOAD has no
// inter-load ordering constraint.
//
// 4 SFPSWAPs vanish from the issue stream, and 2 drain SFPNOPs come back:
// 34 -> 32 instructions and 50 -> 44 cycles per iter, i.e. 6 not 8. The body itself is emitted inline
// by `rb_build_phase_macro` below, because it has to be split across the
// 5-slot MOP template's REPLAY ranges.

// ===========================================================================
//  Candidate 2 -- macro-scheduled first level of the big block
// ===========================================================================
//
// `bitonic_sort_len_32(false)` descending branch:
//   Step 5  SWAP(VC=0,VD=4) (1,5) (2,6) (3,7)     <-- FIRST LEVEL, macro-able
//   Step 4  SWAP(VC=0,VD=2) (1,3) (4,6) (5,7)
//   Step 3  SWAP(VC=0,VD=1) (2,3) (4,5) (6,7)
//   SFPTRANSP + Step 4 + Step 3 + SFPTRANSP
//
// Step 5's VD is {4,5,6,7} and VC is {0,1,2,3}. `load16_rows_x2<16>` already
// loads LREG0..3 before LREG4..7, so no reordering is needed at all here --
// the four LREG4..7 loads simply become macros.
inline void macro_block_body_desc()
{
    TTI_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::INT32, ADDR_MOD_7, 0);
    RB_LOADMACRO(0u, p_sfpu::LREG4, ADDR_MOD_7, 16 + 0);
    TTI_SFPLOAD(p_sfpu::LREG1, InstrModLoadStore::INT32, ADDR_MOD_7, 4);
    RB_LOADMACRO(1u, p_sfpu::LREG5, ADDR_MOD_7, 16 + 4);
    TTI_SFPLOAD(p_sfpu::LREG2, InstrModLoadStore::INT32, ADDR_MOD_7, 8);
    RB_LOADMACRO(2u, p_sfpu::LREG6, ADDR_MOD_7, 16 + 8);
    TTI_SFPLOAD(p_sfpu::LREG3, InstrModLoadStore::INT32, ADDR_MOD_7, 12);
    RB_LOADMACRO(3u, p_sfpu::LREG7, ADDR_MOD_7, 16 + 12);

    // Drain. A macro-scheduled SFPSWAP owns the Simple sub-unit for the two
    // cycles after its macro issues, and macro-scheduled SFPSWAPs are exempt
    // from the automatic one-cycle stall (SFPSWAP.md CAUTION) -- so nothing
    // stops the next software SFPSWAP from being issued into a busy sub-unit.
    // The same reason the loads above are interleaved plain/macro instead of
    // grouped: consecutive macros must be two issue slots apart.
    TTI_SFPNOP;
    TTI_SFPNOP;

    // Step 4 -- stride 8.
    TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG2, p_sfpswap::ALL_ROWS_MAX);
    TTI_SFPSWAP(0, p_sfpu::LREG1, p_sfpu::LREG3, p_sfpswap::ALL_ROWS_MAX);
    TTI_SFPSWAP(0, p_sfpu::LREG4, p_sfpu::LREG6, p_sfpswap::ALL_ROWS_MAX);
    TTI_SFPSWAP(0, p_sfpu::LREG5, p_sfpu::LREG7, p_sfpswap::ALL_ROWS_MAX);

    // Step 3 -- stride 4.
    TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG1, p_sfpswap::ALL_ROWS_MAX);
    TTI_SFPSWAP(0, p_sfpu::LREG2, p_sfpu::LREG3, p_sfpswap::ALL_ROWS_MAX);
    TTI_SFPSWAP(0, p_sfpu::LREG4, p_sfpu::LREG5, p_sfpswap::ALL_ROWS_MAX);
    TTI_SFPSWAP(0, p_sfpu::LREG6, p_sfpu::LREG7, p_sfpswap::ALL_ROWS_MAX);

    TTI_SFPTRANSP(0, 0, 0, 0);

    // Step 4 -- stride 8 (second pass).
    TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG2, p_sfpswap::ALL_ROWS_MAX);
    TTI_SFPSWAP(0, p_sfpu::LREG1, p_sfpu::LREG3, p_sfpswap::ALL_ROWS_MAX);
    TTI_SFPSWAP(0, p_sfpu::LREG4, p_sfpu::LREG6, p_sfpswap::ALL_ROWS_MAX);
    TTI_SFPSWAP(0, p_sfpu::LREG5, p_sfpu::LREG7, p_sfpswap::ALL_ROWS_MAX);

    // Step 3 -- stride 4 (second pass).
    TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG1, p_sfpswap::ALL_ROWS_MAX);
    TTI_SFPSWAP(0, p_sfpu::LREG2, p_sfpu::LREG3, p_sfpswap::ALL_ROWS_MAX);
    TTI_SFPSWAP(0, p_sfpu::LREG4, p_sfpu::LREG5, p_sfpswap::ALL_ROWS_MAX);
    TTI_SFPSWAP(0, p_sfpu::LREG6, p_sfpu::LREG7, p_sfpswap::ALL_ROWS_MAX);

    TTI_SFPTRANSP(0, 0, 0, 0);

    ckernel::sfpu::store16_rows_x2<16, 32>();
}

// ===========================================================================
//  Candidate 3 -- CFG-hoisted face transpose
// ===========================================================================
//
// `transpose_dest_face_32b` spends FIVE `cfg_reg_rmw_tensix` per face on
// SrcA-format and Fp32_enabled switches, and the shipping sweep pays them once
// per face. The three phases (lo16 shuffle, hi16 shuffle, lo16 writeback) each
// want one format, and SrcA has room for four faces' worth of rows (a bank is
// 64 rows, one face is 16), so the loop can be turned inside out: run phase 1
// for every face, then phase 2 for every face, then phase 3 for every face,
// with ONE format switch per phase instead of per face.
//
// Instruction count is identical (20 Matrix Unit ops per face either way); the
// saving is purely the 5N -> 5 CFG writes.
template <int N>
inline void transpose_N_faces_flat()
{
    static_assert(N == 2 || N == 4, "N faces must fit SrcA's 64 rows");

    // --- Phase 1: lo16 of every face -> SrcA rows [16*f, 16*f+16) ---
    cfg_reg_rmw_tensix<ALU_FORMAT_SPEC_REG0_SrcA_RMW>(to_underlying(DataFormat::Float16_b));
#pragma GCC unroll 4
    for (int f = 0; f < N; ++f)
    {
        const int d = f * 16;
        TT_MOVD2B(1, 16, ADDR_MOD_7, p_movd2b::MOV_4_ROWS, d + 0);
        TT_MOVD2B(1, 20, ADDR_MOD_7, p_movd2b::MOV_4_ROWS, d + 4);
        TT_MOVD2B(1, 24, ADDR_MOD_7, p_movd2b::MOV_4_ROWS, d + 8);
        TT_MOVD2B(1, 28, ADDR_MOD_7, p_movd2b::MOV_4_ROWS, d + 12);
        TTI_TRNSPSRCB;
        TT_MOVB2A(d + 0, ADDR_MOD_7, p_movb2a::MOV_4_ROWS, 16);
        TT_MOVB2A(d + 4, ADDR_MOD_7, p_movb2a::MOV_4_ROWS, 20);
        TT_MOVB2A(d + 8, ADDR_MOD_7, p_movb2a::MOV_4_ROWS, 24);
        TT_MOVB2A(d + 12, ADDR_MOD_7, p_movb2a::MOV_4_ROWS, 28);
    }

    // --- Phase 2: hi16 of every face, shuffled straight back into Dst ---
    cfg_reg_rmw_tensix<ALU_FORMAT_SPEC_REG0_SrcA_RMW>(to_underlying(DataFormat::Tf32));
#pragma GCC unroll 4
    for (int f = 0; f < N; ++f)
    {
        const int d = f * 16;
        TT_MOVD2B(0, 16, ADDR_MOD_7, p_movd2b::MOV_4_ROWS, d + 0);
        TT_MOVD2B(0, 20, ADDR_MOD_7, p_movd2b::MOV_4_ROWS, d + 4);
        TT_MOVD2B(0, 24, ADDR_MOD_7, p_movd2b::MOV_4_ROWS, d + 8);
        TT_MOVD2B(0, 28, ADDR_MOD_7, p_movd2b::MOV_4_ROWS, d + 12);
        TTI_TRNSPSRCB;
        TT_MOVB2D(0, 16, ADDR_MOD_7, p_movb2d::MOV_4_ROWS, d + 0);
        TT_MOVB2D(0, 20, ADDR_MOD_7, p_movb2d::MOV_4_ROWS, d + 4);
        TT_MOVB2D(0, 24, ADDR_MOD_7, p_movb2d::MOV_4_ROWS, d + 8);
        TT_MOVB2D(0, 28, ADDR_MOD_7, p_movb2d::MOV_4_ROWS, d + 12);
    }

    // --- Phase 3: lo16 writeback for every face ---
    // Fp32_enabled = 0 + SrcA = Float32 -> writes lo16, preserves hi16.
    cfg_reg_rmw_tensix<ALU_ACC_CTRL_Fp32_enabled_RMW>(0);
    cfg_reg_rmw_tensix<ALU_FORMAT_SPEC_REG0_SrcA_RMW>(to_underlying(DataFormat::Float32));
#pragma GCC unroll 4
    for (int f = 0; f < N; ++f)
    {
        const int d = f * 16;
        TT_MOVA2D(1, d + 0, ADDR_MOD_7, p_mova2d::MOV_8_ROWS, d + 0);
        TT_MOVA2D(1, d + 8, ADDR_MOD_7, p_mova2d::MOV_8_ROWS, d + 8);
    }
    cfg_reg_rmw_tensix<ALU_ACC_CTRL_Fp32_enabled_RMW>(1);
}

// ===========================================================================
//  Phase bodies, lifted verbatim from `_topk_xl_rebuild_generic_<512,_,true>`
// ===========================================================================

// The rebuild's ENTIRE transpose content: two `transpose_N_faces<2>` sweeps
// inside one shared CFG block, in the same STALLWAIT / SETRWC bracket the
// shipping call wraps them in.
inline void rb_transpose_sweeps()
{
    TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::WAIT_SFPU | p_stall::SRCA_VLD | p_stall::SRCB_VLD);
    ckernel::sfpu::enter_transpose_cfg_block();
    ckernel::sfpu::transpose_N_faces<2, true, 256, false>();
    ckernel::sfpu::transpose_N_faces<2, true, 256, false>();
    ckernel::sfpu::leave_transpose_cfg_block();
    TTI_SETRWC(p_setrwc::CLR_AB, 0, 0, 0, 0, p_setrwc::SET_ABD);
}

inline void rb_transpose_sweeps_flat()
{
    TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::WAIT_SFPU | p_stall::SRCA_VLD | p_stall::SRCB_VLD);
    ckernel::sfpu::enter_transpose_cfg_block();
    transpose_N_faces_flat<2>();
    transpose_N_faces_flat<2>();
    ckernel::sfpu::leave_transpose_cfg_block();
    TTI_SETRWC(p_setrwc::CLR_AB, 0, 0, 0, 0, p_setrwc::SET_ABD);
}

// ===========================================================================
//  Probe -- are the transpose's stall cycles fillable from the Vector Unit?
// ===========================================================================
//
// MOVD2B.md:148 (Wormhole tree, shared behaviour): "If MOVD2B is used, then
// during the next three cycles, the only instruction that the MATRIX UNIT (FPU)
// can accept is another MOVD2B. If a thread presents any other MATRIX UNIT
// (FPU) instruction, then hardware will automatically stall the thread." The
// restriction is scoped to the Matrix Unit. `TRNSPSRCB` is a Matrix Unit
// instruction and follows four `MOVD2B`s, so it eats that 3-cycle stall twice
// per face -- which is most of the gap between the 25 issue slots a face costs
// and the ~34 cycles it measures.
//
// If the stall really is Matrix-Unit-scoped, three `SFPNOP`s slotted into each
// window are FREE: they retire in cycles the thread was going to spend stalled
// anyway. That is the whole question, because if they are free then so is real
// SFPU lattice work, and a software-pipelined rebuild (transpose face f+1 while
// sorting face f) could hide ~24 cycles per rebuild inside the transposes.
//
// This arm inserts 6 SFPNOPs per face (2 windows x 3) and changes nothing else.
//   PREDICTION: 143 (unchanged, +/- 2) if the stalls are fillable.
//               167 (= 143 + 24) if SFPNOP is charged like any other issue.
template <int dst_offset>
inline void transpose_dest_face_32b_filled()
{
    cfg_reg_rmw_tensix<ALU_FORMAT_SPEC_REG0_SrcA_RMW>(to_underlying(DataFormat::Float16_b));
    TTI_MOVD2B(1, 16, ADDR_MOD_7, p_movd2b::MOV_4_ROWS, dst_offset + 0);
    TTI_MOVD2B(1, 20, ADDR_MOD_7, p_movd2b::MOV_4_ROWS, dst_offset + 4);
    TTI_MOVD2B(1, 24, ADDR_MOD_7, p_movd2b::MOV_4_ROWS, dst_offset + 8);
    TTI_MOVD2B(1, 28, ADDR_MOD_7, p_movd2b::MOV_4_ROWS, dst_offset + 12);

    TTI_SFPNOP;
    TTI_SFPNOP;
    TTI_SFPNOP;

    TTI_TRNSPSRCB;

    TTI_MOVB2A(0, ADDR_MOD_7, p_movb2a::MOV_4_ROWS, 16);
    TTI_MOVB2A(4, ADDR_MOD_7, p_movb2a::MOV_4_ROWS, 20);
    TTI_MOVB2A(8, ADDR_MOD_7, p_movb2a::MOV_4_ROWS, 24);
    TTI_MOVB2A(12, ADDR_MOD_7, p_movb2a::MOV_4_ROWS, 28);

    cfg_reg_rmw_tensix<ALU_FORMAT_SPEC_REG0_SrcA_RMW>(to_underlying(DataFormat::Tf32));

    TTI_MOVD2B(0, 16, ADDR_MOD_7, p_movd2b::MOV_4_ROWS, dst_offset + 0);
    TTI_MOVD2B(0, 20, ADDR_MOD_7, p_movd2b::MOV_4_ROWS, dst_offset + 4);
    TTI_MOVD2B(0, 24, ADDR_MOD_7, p_movd2b::MOV_4_ROWS, dst_offset + 8);
    TTI_MOVD2B(0, 28, ADDR_MOD_7, p_movd2b::MOV_4_ROWS, dst_offset + 12);

    TTI_SFPNOP;
    TTI_SFPNOP;
    TTI_SFPNOP;

    TTI_TRNSPSRCB;

    TTI_MOVB2D(0, 16, ADDR_MOD_7, p_movb2d::MOV_4_ROWS, dst_offset + 0);
    TTI_MOVB2D(0, 20, ADDR_MOD_7, p_movb2d::MOV_4_ROWS, dst_offset + 4);
    TTI_MOVB2D(0, 24, ADDR_MOD_7, p_movb2d::MOV_4_ROWS, dst_offset + 8);
    TTI_MOVB2D(0, 28, ADDR_MOD_7, p_movb2d::MOV_4_ROWS, dst_offset + 12);

    cfg_reg_rmw_tensix<ALU_ACC_CTRL_Fp32_enabled_RMW>(0);
    cfg_reg_rmw_tensix<ALU_FORMAT_SPEC_REG0_SrcA_RMW>(to_underlying(DataFormat::Float32));

    TTI_MOVA2D(1, 0, ADDR_MOD_7, p_mova2d::MOV_8_ROWS, dst_offset + 0);
    TTI_MOVA2D(1, 8, ADDR_MOD_7, p_mova2d::MOV_8_ROWS, dst_offset + 8);

    cfg_reg_rmw_tensix<ALU_ACC_CTRL_Fp32_enabled_RMW>(1);
}

inline void rb_transpose_sweeps_filled()
{
    TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::WAIT_SFPU | p_stall::SRCA_VLD | p_stall::SRCB_VLD);
    ckernel::sfpu::enter_transpose_cfg_block();
    transpose_dest_face_32b_filled<0>();
    transpose_dest_face_32b_filled<16>();
    transpose_dest_face_32b_filled<0>();
    transpose_dest_face_32b_filled<16>();
    ckernel::sfpu::leave_transpose_cfg_block();
    TTI_SETRWC(p_setrwc::CLR_AB, 0, 0, 0, 0, p_setrwc::SET_ABD);
}

// ===========================================================================
//  Candidate 4 -- put the macro template swap INSIDE the transpose's stalls
// ===========================================================================
//
// `RbXposeNFill` measured 143.000 cyc against `RbXposeN`'s 143.000 with 24 more
// SFPNOPs in the stream, so each face's two MOVD2B->TRNSPSRCB windows really do
// hand back 3 free Vector Unit issue slots. 24 free SFPU slots per rebuild.
//
// The obvious customer is the ONE thing standing between the two macro
// candidates and their combined 32-cycle saving: `RbCallMacro` has to rewrite
// InstructionTemplate[] between the build phase (Step 4, VC = LREG0,1,4,5) and
// the block phase (Step 5, VC = LREG0,1,2,3), and the backdoor write is itself
// an SFPSWAP -- 2 cycles plus its auto-stall. Measured, that reprogramming ate
// 26 of the 32 cycles the two candidates saved.
//
// Two things fix it, and both are free:
//   1. Templates 0 and 1 are IDENTICAL between the two phases (both VC = LREG0
//      and LREG1). Only templates 2 and 3 differ, so the switch is TWO backdoor
//      writes, not four.
//   2. Each sweep transposes two faces and therefore offers four free windows.
//      One backdoor write per face's first window, one SFPNOP settle per face's
//      second window, and the switch costs nothing at all.
//
// Ordering is self-consistent across calls: sweep 1 (which precedes the build)
// installs the build's templates, sweep 2 (which precedes the block) installs
// the block's, and the next call's sweep 1 puts the build's back.
//
//   PREDICTION: 342 = 374 - 32, i.e. the full phase-level saving with the
//   template switch fully absorbed.
template <int dst_offset>
inline void transpose_dest_face_32b_inject(const std::uint32_t inject_w1, const std::uint32_t inject_w2)
{
    cfg_reg_rmw_tensix<ALU_FORMAT_SPEC_REG0_SrcA_RMW>(to_underlying(DataFormat::Float16_b));
    TTI_MOVD2B(1, 16, ADDR_MOD_7, p_movd2b::MOV_4_ROWS, dst_offset + 0);
    TTI_MOVD2B(1, 20, ADDR_MOD_7, p_movd2b::MOV_4_ROWS, dst_offset + 4);
    TTI_MOVD2B(1, 24, ADDR_MOD_7, p_movd2b::MOV_4_ROWS, dst_offset + 8);
    TTI_MOVD2B(1, 28, ADDR_MOD_7, p_movd2b::MOV_4_ROWS, dst_offset + 12);

    ckernel::instrn_buffer[0] = inject_w1; // rides the 3-cycle MOVD2B shadow

    TTI_TRNSPSRCB;

    TTI_MOVB2A(0, ADDR_MOD_7, p_movb2a::MOV_4_ROWS, 16);
    TTI_MOVB2A(4, ADDR_MOD_7, p_movb2a::MOV_4_ROWS, 20);
    TTI_MOVB2A(8, ADDR_MOD_7, p_movb2a::MOV_4_ROWS, 24);
    TTI_MOVB2A(12, ADDR_MOD_7, p_movb2a::MOV_4_ROWS, 28);

    cfg_reg_rmw_tensix<ALU_FORMAT_SPEC_REG0_SrcA_RMW>(to_underlying(DataFormat::Tf32));

    TTI_MOVD2B(0, 16, ADDR_MOD_7, p_movd2b::MOV_4_ROWS, dst_offset + 0);
    TTI_MOVD2B(0, 20, ADDR_MOD_7, p_movd2b::MOV_4_ROWS, dst_offset + 4);
    TTI_MOVD2B(0, 24, ADDR_MOD_7, p_movd2b::MOV_4_ROWS, dst_offset + 8);
    TTI_MOVD2B(0, 28, ADDR_MOD_7, p_movd2b::MOV_4_ROWS, dst_offset + 12);

    ckernel::instrn_buffer[0] = inject_w2;

    TTI_TRNSPSRCB;

    TTI_MOVB2D(0, 16, ADDR_MOD_7, p_movb2d::MOV_4_ROWS, dst_offset + 0);
    TTI_MOVB2D(0, 20, ADDR_MOD_7, p_movb2d::MOV_4_ROWS, dst_offset + 4);
    TTI_MOVB2D(0, 24, ADDR_MOD_7, p_movb2d::MOV_4_ROWS, dst_offset + 8);
    TTI_MOVB2D(0, 28, ADDR_MOD_7, p_movb2d::MOV_4_ROWS, dst_offset + 12);

    cfg_reg_rmw_tensix<ALU_ACC_CTRL_Fp32_enabled_RMW>(0);
    cfg_reg_rmw_tensix<ALU_FORMAT_SPEC_REG0_SrcA_RMW>(to_underlying(DataFormat::Float32));

    TTI_MOVA2D(1, 0, ADDR_MOD_7, p_mova2d::MOV_8_ROWS, dst_offset + 0);
    TTI_MOVA2D(1, 8, ADDR_MOD_7, p_mova2d::MOV_8_ROWS, dst_offset + 8);

    cfg_reg_rmw_tensix<ALU_ACC_CTRL_Fp32_enabled_RMW>(1);
}

// One `transpose_N_faces<2>`-equivalent sweep that also installs
// InstructionTemplate[2] = SFPSWAP(VC = vc2) and [3] = SFPSWAP(VC = vc3).
inline void transpose_sweep_with_template_swap(const std::uint32_t vc2, const std::uint32_t vc3)
{
    transpose_dest_face_32b_inject<0>(TT_OP_SFPSWAP(0, vc2, 14, p_sfpswap::ALL_ROWS_MAX), TT_OP_SFPNOP);
    transpose_dest_face_32b_inject<16>(TT_OP_SFPSWAP(0, vc3, 15, p_sfpswap::ALL_ROWS_MAX), TT_OP_SFPNOP);
}

inline void rb_transpose_one_face()
{
    TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::WAIT_SFPU | p_stall::SRCA_VLD | p_stall::SRCB_VLD);
    ckernel::sfpu::enter_transpose_cfg_block();
    ckernel::sfpu::transpose_dest_face_32b<0>();
    ckernel::sfpu::leave_transpose_cfg_block();
    TTI_SETRWC(p_setrwc::CLR_AB, 0, 0, 0, 0, p_setrwc::SET_ABD);
}

// The stride-2 + sort_16_alt build phase, verbatim from the shipping rebuild
// (n_iters_stride2 = row_scale_factor * 2 = 2 at K = 512).
inline void rb_build_phase()
{
    ckernel::sfpu::topk_rebuild_build2048_mop_config();
    ckernel::load_replay_buf<ckernel::Exec>(
        0,
        16,
        []
        {
            ckernel::sfpu::load16_rows_x2<2>();
            ckernel::sfpu::bitonic_sort_len_16_alt_swaps<true>(false);
        });
    TTI_SFPTRANSP(0, 0, 0, 0);
    ckernel::load_replay_buf<ckernel::Exec>(16, 8, [] { ckernel::sfpu::bitonic_sort_len_16_alt_swaps<true>(false); });
    TTI_SFPTRANSP(0, 0, 0, 0);
    ckernel::load_replay_buf<ckernel::Exec>(24, 8, [] { ckernel::sfpu::store16_rows_x2<2, 16>(); });
    ckernel::ckernel_unpack_template::run(2 - 1);
    TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D);
}

// Same phase with the first lattice level moved into the loads' macro slots.
// The 5-slot MOP template's REPLAY ranges shift because the recorded body is
// 30 instructions rather than 34: [0..11] load + first-half Step 3,
// [12..19] second half's Step 4 + Step 3, [20..27] store.
inline void rb_build_phase_macro()
{
    constexpr std::uint32_t replay_a  = lltt::replay_insn(0, 14);
    constexpr std::uint32_t replay_b  = lltt::replay_insn(14, 8);
    constexpr std::uint32_t replay_st = lltt::replay_insn(22, 8);
    constexpr std::uint32_t transp_op = TT_OP_SFPTRANSP(0, 0, 0, 0);

    ckernel::ckernel_unpack_template tmpl(
        /*unpackB=*/true,
        /*unpack_halo=*/true,
        /*A0=*/replay_a,
        /*A1=*/transp_op,
        /*A2=*/replay_b,
        /*A3=*/transp_op,
        /*skipA=*/TT_OP_NOP,
        /*B=*/replay_st,
        /*skipB=*/TT_OP_NOP);
    tmpl.program();

    ckernel::load_replay_buf<ckernel::Exec>(
        0,
        14,
        []
        {
            // 8 loads (interleaved plain/macro so consecutive macros are two
            // issue slots apart) + 2 drain SFPNOPs + Step 3 (4). Step 4 rides
            // the four macros.
            TTI_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::INT32, ADDR_MOD_7, 0);
            RB_LOADMACRO(0u, p_sfpu::LREG2, ADDR_MOD_7, 8);
            TTI_SFPLOAD(p_sfpu::LREG1, InstrModLoadStore::INT32, ADDR_MOD_7, 4);
            RB_LOADMACRO(1u, p_sfpu::LREG3, ADDR_MOD_7, 12);
            TTI_SFPLOAD(p_sfpu::LREG4, InstrModLoadStore::INT32, ADDR_MOD_7, 2 + 0);
            RB_LOADMACRO(2u, p_sfpu::LREG6, ADDR_MOD_7, 2 + 8);
            TTI_SFPLOAD(p_sfpu::LREG5, InstrModLoadStore::INT32, ADDR_MOD_7, 2 + 4);
            RB_LOADMACRO(3u, p_sfpu::LREG7, ADDR_MOD_7, 2 + 12);
            TTI_SFPNOP;
            TTI_SFPNOP;
            TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG1, p_sfpswap::ALL_ROWS_MAX);
            TTI_SFPSWAP(0, p_sfpu::LREG2, p_sfpu::LREG3, p_sfpswap::ALL_ROWS_MAX);
            TTI_SFPSWAP(0, p_sfpu::LREG4, p_sfpu::LREG5, p_sfpswap::ALL_ROWS_MAX);
            TTI_SFPSWAP(0, p_sfpu::LREG6, p_sfpu::LREG7, p_sfpswap::ALL_ROWS_MAX);
        });
    TTI_SFPTRANSP(0, 0, 0, 0);
    ckernel::load_replay_buf<ckernel::Exec>(14, 8, [] { ckernel::sfpu::bitonic_sort_len_16_alt_swaps<true>(false); });
    TTI_SFPTRANSP(0, 0, 0, 0);
    ckernel::load_replay_buf<ckernel::Exec>(22, 8, [] { ckernel::sfpu::store16_rows_x2<2, 16>(); });
    ckernel::ckernel_unpack_template::run(2 - 1);
    TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D);
}

// The two per-column `canonical_big_block_with_replay<1>` passes, verbatim.
inline void rb_block_phase(const std::uint32_t tile_offset)
{
    for (int col = 0; col < 2; col++)
    {
        ckernel::sfpu::canonical_big_block_with_replay<1>(false);
        ckernel::sfpu::set_dst_write_addr_offset(tile_offset + (col ? 0 : 2));
    }
}

inline void rb_block_phase_macro(const std::uint32_t tile_offset)
{
    for (int col = 0; col < 2; col++)
    {
        macro_block_body_desc();
        TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D);
        ckernel::sfpu::set_dst_write_addr_offset(tile_offset + (col ? 0 : 2));
    }
}

// Rewrite the four InstructionTemplates in place. Only the templates change
// between the build phase (Step 4: VC = LREG0,1,4,5) and the block phase
// (Step 5: VC = LREG0,1,2,3); Sequence and Misc are identical, so the mid-call
// switch is four backdoor writes plus the two SFPNOP settle instructions the
// merge's config also ends with. This is a REAL cost of running both
// candidates in one call and is included in the RbCallMacro number.
inline void reprogram_macro_templates(const std::uint32_t vc0, const std::uint32_t vc1, const std::uint32_t vc2, const std::uint32_t vc3)
{
    TT_SFPSWAP(0, vc0, 12, p_sfpswap::ALL_ROWS_MAX);
    TT_SFPSWAP(0, vc1, 13, p_sfpswap::ALL_ROWS_MAX);
    TT_SFPSWAP(0, vc2, 14, p_sfpswap::ALL_ROWS_MAX);
    TT_SFPSWAP(0, vc3, 15, p_sfpswap::ALL_ROWS_MAX);
    TTI_SFPNOP;
    TTI_SFPNOP;
}

// `_topk_xl_rebuild_generic_<512, false, true>` with BOTH macro candidates
// applied, structurally line-for-line the shipping body so the two call arms
// differ by exactly the two lattice bodies (plus the one template swap).
inline void macro_rebuild_call(const std::uint32_t dst_index)
{
    const std::uint32_t tile_offset = dst_index << DstTileSizeLog2[DstTileShape::Tile32x32];

    TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::WAIT_SFPU | p_stall::SRCA_VLD | p_stall::SRCB_VLD);
    ckernel::sfpu::enter_transpose_cfg_block();
    ckernel::sfpu::transpose_N_faces<2, true, 256, false>();

    reprogram_macro_templates(p_sfpu::LREG0, p_sfpu::LREG1, p_sfpu::LREG4, p_sfpu::LREG5);
    rb_build_phase_macro();

    ckernel::sfpu::transpose_N_faces<2, true, 256, false>();
    ckernel::sfpu::leave_transpose_cfg_block();

    reprogram_macro_templates(p_sfpu::LREG0, p_sfpu::LREG1, p_sfpu::LREG2, p_sfpu::LREG3);
    rb_block_phase_macro(tile_offset);

    // The build phase clobbered the MOP Expander config; restore the merge
    // programming exactly as the shipping rebuild does.
    ckernel::sfpu::topk_mop_config<true>();
    TTI_SETRWC(p_setrwc::CLR_AB, 0, 0, 0, 0, p_setrwc::SET_ABD);
}

// Same, with the template switch folded into the transposes' free slots.
// THIS IS THE CANDIDATE. Structurally identical to
// `_topk_xl_rebuild_generic_<512, false, true>`; the only differences are the
// two lattice bodies and where the (now free) template writes live.
inline void macro_rebuild_call_sched(const std::uint32_t dst_index)
{
    const std::uint32_t tile_offset = dst_index << DstTileSizeLog2[DstTileShape::Tile32x32];

    TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::WAIT_SFPU | p_stall::SRCA_VLD | p_stall::SRCB_VLD);
    ckernel::sfpu::enter_transpose_cfg_block();

    // Sweep 1 (C -> T) also installs the BUILD phase's Step-4 templates.
    transpose_sweep_with_template_swap(p_sfpu::LREG4, p_sfpu::LREG5);
    rb_build_phase_macro();

    // Sweep 2 (T -> C) also installs the BLOCK phase's Step-5 templates.
    transpose_sweep_with_template_swap(p_sfpu::LREG2, p_sfpu::LREG3);
    ckernel::sfpu::leave_transpose_cfg_block();

    rb_block_phase_macro(tile_offset);

    ckernel::sfpu::topk_mop_config<true>();
    TTI_SETRWC(p_setrwc::CLR_AB, 0, 0, 0, 0, p_setrwc::SET_ABD);
}
} // namespace

void run_kernel(RUNTIME_PARAMETERS params)
{
    {
        START_PERF_MEASURE("INIT")

        // Kernel-invariant SFPU init: SFPCONFIG(0, 0xF, 1) plus ADDR_MOD_7.
        // Also clears LaneConfig, the precondition for the VD >= 12 backdoor
        // template writes. Must come first.
        _llk_math_eltwise_unary_sfpu_init_once_();

        // Programs ADDR_MOD_1/5/6 and the merge MOP template. Run for EVERY arm
        // so the arms differ by exactly the body under test.
        ckernel::sfpu::_topk_xl_init_<XL_K, XL_FUSED>();

        // Clear stale lane predication. SFPSWAP's writes are gated on
        // LaneEnabled (SFPSWAP.md:38) and SFPTRANSP's on the same, so a mask
        // left behind by a previously-run kernel would silently suppress work.
        TTI_SFPENCC(0, 0, 0, SFPENCC_MOD1_EI);

#if REBUILD_ARM == ARM_RB_BUILD_MACRO_ID
        // Build phase Step 4 (descending): SWAP(VC=0,VD=2) (1,3) (4,6) (5,7).
        configure_rebuild_macros(p_sfpu::LREG0, p_sfpu::LREG1, p_sfpu::LREG4, p_sfpu::LREG5);
#elif REBUILD_ARM == ARM_RB_BLOCK_MACRO_ID
        // Block phase Step 5 (descending): SWAP(VC=0,VD=4) (1,5) (2,6) (3,7).
        configure_rebuild_macros(p_sfpu::LREG0, p_sfpu::LREG1, p_sfpu::LREG2, p_sfpu::LREG3);
#elif REBUILD_ARM == ARM_RB_CALL_MACRO_ID || REBUILD_ARM == ARM_RB_CALL_SCHED_ID
        // Sequence + Misc, plus templates 0 and 1 -- which are IDENTICAL for
        // both lattice phases and therefore never rewritten. Templates 2 and 3
        // are installed per phase by the call bodies.
        configure_rebuild_macros(p_sfpu::LREG0, p_sfpu::LREG1, p_sfpu::LREG4, p_sfpu::LREG5);
#endif

        // Establish the Dst window once.
        _llk_math_eltwise_sfpu_start_(0);

        PROFILER_SYNC();
    }

    {
        START_PERF_MEASURE("TILE_LOOP")

#if REBUILD_ARM == ARM_CTRL_LOAD_ID
        {
            // CONTROL -- frontend floor. Deliberately a PLAIN load: a macro
            // issue in a control arm would run against whatever
            // LoadMacroConfig a previously executed kernel left behind.
            ckernel::load_replay_buf<ckernel::NoExec>(
                0,
                2,
                []
                {
                    TTI_SFPLOAD(L_CTRL_A, ckernel::InstrModLoadStore::INT32, ckernel::ADDR_MOD_7, 0);
                    TTI_SFPLOAD(L_CTRL_B, ckernel::InstrModLoadStore::INT32, ckernel::ADDR_MOD_7, 2);
                });
            ckernel::ckernel_unpack_template::lA(lltt::replay_insn(0, 2), TT_OP_NOP).program();
            mop_run_all();
        }
#elif REBUILD_ARM == ARM_CTRL_SWAP_ID
        {
            // CONTROL -- the tripwire. MUST come out at ~2.0x ARM_CTRL_LOAD.
            ckernel::load_replay_buf<ckernel::NoExec>(
                0,
                2,
                []
                {
                    TTI_SFPSWAP(0, L_CTRL_A, L_CTRL_B, ckernel::p_sfpswap::ALL_ROWS_MAX);
                    TTI_SFPSWAP(0, L_CTRL_B, L_CTRL_A, ckernel::p_sfpswap::ALL_ROWS_MAX);
                });
            ckernel::ckernel_unpack_template::lA(lltt::replay_insn(0, 2), TT_OP_NOP).program();
            mop_run_all();
        }
#elif REBUILD_ARM == ARM_RB_CALL_ID
        {
#pragma GCC unroll 1
            for (std::uint32_t i = 0; i < REBUILD_ITER_COUNT; ++i)
            {
                _llk_math_eltwise_sfpu_start_(0);
                ckernel::sfpu::_topk_xl_rebuild_<XL_K, TOPK_APPROX, XL_FUSED>(0 /* dst_index */, false /* ascending */);
                _llk_math_eltwise_sfpu_done_();
            }
        }
#elif REBUILD_ARM == ARM_RB_XPOSE_FACE_ID
        {
#pragma GCC unroll 1
            for (std::uint32_t i = 0; i < REBUILD_ITER_COUNT; ++i)
            {
                rb_transpose_one_face();
            }
        }
#elif REBUILD_ARM == ARM_RB_XPOSE_N_ID
        {
#pragma GCC unroll 1
            for (std::uint32_t i = 0; i < REBUILD_ITER_COUNT; ++i)
            {
                rb_transpose_sweeps();
            }
        }
#elif REBUILD_ARM == ARM_RB_XPOSE_FLAT_ID
        {
#pragma GCC unroll 1
            for (std::uint32_t i = 0; i < REBUILD_ITER_COUNT; ++i)
            {
                rb_transpose_sweeps_flat();
            }
        }
#elif REBUILD_ARM == ARM_RB_BUILD_ID
        {
#pragma GCC unroll 1
            for (std::uint32_t i = 0; i < REBUILD_ITER_COUNT; ++i)
            {
                rb_build_phase();
            }
        }
#elif REBUILD_ARM == ARM_RB_BUILD_MACRO_ID
        {
#pragma GCC unroll 1
            for (std::uint32_t i = 0; i < REBUILD_ITER_COUNT; ++i)
            {
                rb_build_phase_macro();
            }
        }
#elif REBUILD_ARM == ARM_RB_BLOCK_ID
        {
#pragma GCC unroll 1
            for (std::uint32_t i = 0; i < REBUILD_ITER_COUNT; ++i)
            {
                rb_block_phase(0);
            }
        }
#elif REBUILD_ARM == ARM_RB_XPOSE_FILL_ID
        {
#pragma GCC unroll 1
            for (std::uint32_t i = 0; i < REBUILD_ITER_COUNT; ++i)
            {
                rb_transpose_sweeps_filled();
            }
        }
#elif REBUILD_ARM == ARM_RB_CALL_MACRO_ID
        {
#pragma GCC unroll 1
            for (std::uint32_t i = 0; i < REBUILD_ITER_COUNT; ++i)
            {
                _llk_math_eltwise_sfpu_start_(0);
                macro_rebuild_call(0 /* dst_index */);
                _llk_math_eltwise_sfpu_done_();
            }
        }
#elif REBUILD_ARM == ARM_RB_CALL_SCHED_ID
        {
#pragma GCC unroll 1
            for (std::uint32_t i = 0; i < REBUILD_ITER_COUNT; ++i)
            {
                _llk_math_eltwise_sfpu_start_(0);
                macro_rebuild_call_sched(0 /* dst_index */);
                _llk_math_eltwise_sfpu_done_();
            }
        }
#else // ARM_RB_BLOCK_MACRO_ID
        {
#pragma GCC unroll 1
            for (std::uint32_t i = 0; i < REBUILD_ITER_COUNT; ++i)
            {
                rb_block_phase_macro(0);
            }
        }
#endif

        // MANDATORY. ZONE_SCOPED timestamps on the RISC-V at scope exit and the
        // RISC-V runs far ahead of the backend -- most of all on the MOP-fed
        // arms, where one push leaves the entire loop in flight.
        PROFILER_SYNC();
    }
}

#endif // LLK_TRISC_MATH

// Unpack and pack do no work but MUST declare the same zones in the same order
// as math: under --enable-perf-counters the zones form a three-thread semaphore
// barrier (counters.h:545-587) that deadlocks on a mismatched set.

#ifdef LLK_TRISC_UNPACK

#include "llk_unpack_common.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
    {
        START_PERF_MEASURE("INIT")
        PROFILER_SYNC();
    }
    {
        START_PERF_MEASURE("TILE_LOOP")

#if REBUILD_NEEDS_SRCB_VALID
        // Every transposing arm shuttles half-words through SrcB
        // (MOVD2B / TRNSPSRCB / MOVB2A / MOVB2D). Those stall MATH until the
        // unpacker marks SrcB valid, and this kernel unpacks nothing -- so
        // without one dummy valid per iteration the math thread hangs
        // (observed: TENSIX TIMED OUT, "waited 2 seconds for Math").
        // sources/topk_xl_test.cpp:151 issues the same call for the same reason.
        for (std::uint32_t i = 0; i < REBUILD_ITER_COUNT; ++i)
        {
            _llk_unpack_set_srcb_dummy_valid_();
        }
#endif

        PROFILER_SYNC();
    }
}
#endif

#ifdef LLK_TRISC_PACK

#include "llk_pack_common.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
    {
        START_PERF_MEASURE("INIT")
        PROFILER_SYNC();
    }
    {
        START_PERF_MEASURE("TILE_LOOP")
        PROFILER_SYNC();
    }
}
#endif
