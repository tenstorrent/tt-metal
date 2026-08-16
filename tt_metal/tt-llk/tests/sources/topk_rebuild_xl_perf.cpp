// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// ============================================================================
//  Pricing `_topk_xl_rebuild_` at K = 1024 and K = 2048, and attacking the ONE
//  place where the merge's FULL SFPLOADMACRO trick applies (Blackhole only)
// ============================================================================
//
// WHY THIS FILE EXISTS
// --------------------
// `topk_rebuild_perf.cpp` priced `_topk_xl_rebuild_<512>` at 374 cyc/call and
// recovered 24 of them (1.069x) by moving the FIRST level of each lattice into
// the loads' SFPLOADMACRO Simple slots. It could not do better because a
// macro-scheduled Simple must have VD == macroVD -- the register that macro's
// own load just wrote -- so only a compare-exchange with BOTH operands freshly
// loaded can ride a load. In a multi-level lattice that is level 1 and nothing
// else.
//
// `canonical_big_block_with_replay<rsf>` (ckernel_sfpu_topk_xl.h:1088) has two
// sub-blocks that are NOT multi-level:
//
//   sub-block A (rsf >= 4, i.e. K = 2048 only):
//       rsf x [ load16_rows_x2<64> + bitonic_sort_len_k(dir) + store16_rows_x2<64,16> ]
//   sub-block B (rsf >= 2, i.e. K = 1024 and K = 2048):
//       rsf x [ load16_rows_x2<32> + bitonic_sort_len_k(dir) + store16_rows_x2<32,16|48> ]
//
// `bitonic_sort_len_k` (:830) is exactly 4 TTI_SFPSWAP -- a SINGLE level -- and
// the store addresses are IDENTICAL to the load addresses. That is structurally
// the merge body, so the merge's FULL trick applies: the SFPSWAP rides the
// macro's Simple slot AND an SFPSTORE rides the macro's Store slot, whose
// address SFPLOADMACRO.md:140 forces to equal its load's.
//
// At rsf == 1 (K = 512) both sub-blocks are `if constexpr`'d away, which is
// exactly why the K=512 work saw none of this.
//
// THE BODY, SHIPPING vs MACRO (descending; ascending is the mirror)
// ------------------------------------------------------------------
// Shipping, 20 instructions / 24 cycles:
//     8 x SFPLOAD    (LREG0..3 <- {0,4,8,12}, LREG4..7 <- {D+0,D+4,D+8,D+12})
//     4 x SFPSWAP    (VC = LREG0..3, VD = LREG4..7, Mod1 = ALL_ROWS_MAX)
//                    -> LREG4..7 = min, LREG0..3 = max
//     8 x SFPSTORE   (back to the same eight addresses)
//     8*1 + 4*2 + 8*1 = 24 cycles for 8 vectors = 3.000 cyc/vector
//
// Macro, 14 instructions / 14 cycles:
//     i1  SFPLOAD      LREG0 <- 0
//     i2  SFPLOADMACRO LREG4 <- D+0   Simple: SFPSWAP(VC=LREG0, VD=macroVD)
//                                     MAD:    SFPNOP            (footnote (‡))
//                                     Store:  LREG4 -> D+0, delay 2
//     i3  SFPLOAD      LREG1 <- 4
//     i4  SFPLOADMACRO LREG5 <- D+4
//     i5  SFPLOAD      LREG2 <- 8         | macro store 0 fires
//     i6  SFPLOADMACRO LREG6 <- D+8
//     i7  SFPLOAD      LREG3 <- 12        | macro store 1 fires
//     i8  SFPLOADMACRO LREG7 <- D+12
//     i9  SFPNOP                          | macro store 2 fires
//     i10 SFPSTORE     LREG0 -> 0
//     i11 SFPNOP                          | macro store 3 fires
//     i12 SFPSTORE     LREG1 -> 4
//     i13 SFPSTORE     LREG2 -> 8
//     i14 SFPSTORE     LREG3 -> 12        (carries the ADDR_MOD fold)
//
// Three constraints the layout is built around, each learned the hard way:
//   * Consecutive SFPLOADMACROs are TWO issue slots apart. A macro-scheduled
//     SFPSWAP owns the Simple sub-unit for two cycles and is EXEMPT from the
//     automatic post-SFPSWAP stall (SFPSWAP.md CAUTION), so back-to-back macros
//     let their swaps overlap. That measures FASTER and is silently wrong.
//   * The four macro-scheduled stores fire at i5/i7/i9/i11 (delay 2 =
//     WaitForElapsedInstructions, so macro + 3). NO software SFPSTORE is placed
//     on those cycles -- the two SFPNOPs at i9/i11 exist for exactly that, and
//     they also serve as the Simple-sub-unit drain the last macro needs.
//   * The Dst advance that `store16_rows_x2<D, inc>` folds into its LREG7 store
//     moves to the LAST SOFTWARE store (i14), because LREG7's store now rides a
//     macro and a macro store inherits the LOAD's ADDR_MOD. Macro store
//     addresses are latched at SFPLOADMACRO time (SFPLOADMACRO.md:140), so all
//     four have already been issued with pre-advance addresses.
//
// WHY ONE SEQUENCE WORD SERVES ALL THREE PHASES
// ---------------------------------------------
// `topk_rebuild_macro_test.cpp` configures the macros with the Store slot
// DISABLED, because in the build phase and in sub-block C the loaded register
// is rewritten by three or four more lattice levels before it may be stored.
// Sub-blocks A and B need the Store slot ENABLED. Reprogramming Sequence[0..3]
// between sub-blocks costs 12 instructions and would have to happen four times
// per rebuild (A/B -> C and back, twice) -- 48 cycles, which is more than the
// whole K=1024 saving.
//
// It does not have to happen at all. In the build phase and in sub-block C the
// load addresses and the final store addresses are THE SAME EIGHT ADDRESSES
// (`load16_rows_x2<g>` / `store16_rows_x2<g, inc>` are mirrors). So a
// macro-scheduled store there writes a mid-lattice value to a Dst word that the
// body's own closing `store16_rows_x2` overwrites unconditionally a dozen
// instructions later. It is a wasted write on an otherwise idle sub-unit, not a
// corruption. One Sequence word, Store slot on, for every phase.
//
// PREDICTED vs MEASURED (Blackhole silicon, MATH_ISOLATE, two-point slope over
// REBUILD_ITER_COUNT, 5 runs/point). Every prediction was recorded before the
// first run; the four that matter (RbCall and RbCallFull at both K) landed
// EXACTLY.
//
//   arm            K=512            K=1024           K=2048
//                  pred / meas      pred / meas      pred / meas
//   CtrlLoad          -  / 0.999       -  / 0.999       -  / 0.999   cyc/vector
//   CtrlSwap          -  / 1.999       -  / 1.999       -  / 1.999   cyc/vector
//   RbCall          374  /  374      822  /  822     1810  / 1810
//   RbXposeN        143  /  143      275  /  275      539  /  539
//   RbBuild         102  /  102      198  /  202      390  /  402
//   RbBlock         120  /  120      340  /  336      872  /  860
//   RbSubA            -  /    -        -  /    -      200  /  198
//   RbSubB            -  /    -       96  /  100      192  /  196
//   RbSubC          120  /  120      244  /  238      488  /  470
//   RbSubABMacro      0  /    4       56  /   60      224  /  232
//   RbBlockFull     108  /  108      252  /  272      616  /  652
//   RbCallSched     350  /  350      774  /  774     1714  / 1714
//   RbCallFull      350  /  350      734  /  734     1554  / 1554
//   XlMerge          91  /   91      171  /  171      331  /  331
//   MacroMerge       46  /   46       78  /   78      142  /  142
//   XlStep            -  /  459        -  /  987        -  / 2135
//   FullStep          -  /  404        -  /  817        -  / 1701
//
// HARNESS VALIDATION. RbCall 374.000, RbCallSched 350.017, XlMerge 91.000 and
// MacroMerge 46.000 at K = 512 reproduce perf_topk_rebuild.py and
// perf_topk_merge_macro.py to the cycle, and RbSubABMacro measures 4 cycles at
// K = 512 -- the two_cols bracket with an EMPTY body, which is the direct
// evidence that sub-blocks A and B are `if constexpr`'d away at rsf == 1 and
// hence that the K = 512 work could not have seen them.
//
// THE RESULT
//
//   rebuild K=512  :  374 ->  350 = 1.069x   (unchanged -- no sub-block A or B)
//   rebuild K=1024 :  822 ->  734 = 1.120x
//   rebuild K=2048 : 1810 -> 1554 = 1.165x
//
//   of which the FULL trick on sub-blocks A and B contributes, on top of the
//   first-level-only macro (RbCallSched):
//     K=1024   774 -> 734  = 40 cycles =  4 bodies x 10
//     K=2048  1714 -> 1554 = 160 cycles = 16 bodies x 10
//   i.e. exactly the predicted 24 -> 14 cycles per sub-block A/B body.
//
//   reduction step (merge + rebuild), both halves on macros:
//     K=512    459 -> 404 = 1.136x   (28.688 -> 25.250 cyc/vector)
//     K=1024   987 -> 817 = 1.208x   (30.843 -> 25.533 cyc/vector)
//     K=2048  2135 -> 1701 = 1.255x  (33.359 -> 26.578 cyc/vector)
//
// SUB-BLOCK A/B SHARE OF THE SHIPPING REBUILD -- the question this file was
// written to answer:
//     K=512   :   0 / 374  =  0.0%   (both compiled out)
//     K=1024  : 100 / 822  = 12.2%   (sub-block B only)
//     K=2048  : 394 / 1810 = 21.8%   (sub-blocks A and B)
//
// WHAT IS LEFT IN SUB-BLOCK C, AND WHY IT IS NOTHING. Sub-block C's body is
// 8 SFPLOAD + `bitonic_sort_len_32` (20 SFPSWAP + 2 SFPTRANSP) + 8 SFPSTORE =
// 60 cycles measured, and the macro version is 54 (4 of the 20 swaps hidden,
// minus 2 drain SFPNOPs). The 16 remaining swaps are levels 2..5, which read
// level 1's outputs: there is no load left to attach them to, and reloading
// would cost the very instruction the macro saves. Its stores cannot ride
// either -- the register is rewritten four more times before it is final, and
// the Store slot's delay field is 2 bits (max 3 cycles). 54 against an SFPU
// floor of 8 + 16*2 + 2 + 8 = 50 leaves ~4 cycles per body, ~1% of the call.
// Sub-block C is done.
//
// CORRECTNESS IS NOT ESTABLISHED BY ANY OF THIS -- see
// `tests/python_tests/test_topk_rebuild_full_macro.py`, which runs the shipping
// torch golden at K = 512/1024/2048 in both directions with num_chunks=4 (three
// CHAINED merge+rebuild pairs; a single pair cannot see a broken rebuild
// because the rebuild only PERMUTES its K survivors) plus a mutation control.
//
// DEVICE / RUN NOTES. Blackhole silicon only. Read CtrlLoad and CtrlSwap FIRST;
// if CtrlSwap is not 2.00x CtrlLoad the run is INVALID. Every transposing arm
// needs one `_llk_unpack_set_srcb_dummy_valid_()` per iteration from the UNPACK
// thread or MATH hangs on SrcB valid (TENSIX TIMED OUT).

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
// Parameters. All MUST be #define (not constexpr): the kernel guards each with
// #ifndef and a constexpr does not satisfy a preprocessor guard -- every swept
// variant would compile identically while still hashing to a distinct id.
// ---------------------------------------------------------------------------
#define ARM_CTRL_LOAD_ID       0
#define ARM_CTRL_SWAP_ID       1
#define ARM_RB_CALL_ID         2
#define ARM_RB_XPOSE_N_ID      3
#define ARM_RB_BUILD_ID        4
#define ARM_RB_BLOCK_ID        5
#define ARM_RB_SUB_A_ID        6
#define ARM_RB_SUB_B_ID        7
#define ARM_RB_SUB_C_ID        8
#define ARM_RB_SUB_AB_MACRO_ID 9
#define ARM_RB_BLOCK_FULL_ID   10
#define ARM_RB_CALL_SCHED_ID   11
#define ARM_RB_CALL_FULL_ID    12
// Context arms: the rebuild is only HALF of a K-reduction step. `_topk_xl_merge_`
// leaves the survivors ordered but no longer bitonic, so every merge must be
// followed by a rebuild before the next one. These four price the rebuild win
// against the step that actually ships.
#define ARM_XL_MERGE_ID    13
#define ARM_MACRO_MERGE_ID 14
#define ARM_XL_STEP_ID     15
#define ARM_FULL_STEP_ID   16

#ifndef REBUILD_ARM
#define REBUILD_ARM 2
#endif

#ifndef REBUILD_ITER_COUNT
#define REBUILD_ITER_COUNT 16
#endif

#ifndef REBUILD_K
#define REBUILD_K 1024
#endif

// Arms whose body drives the Matrix Unit through SrcB and therefore needs an
// unpack-side dummy valid per iteration.
#if REBUILD_ARM == ARM_RB_CALL_ID || REBUILD_ARM == ARM_RB_XPOSE_N_ID || REBUILD_ARM == ARM_RB_CALL_SCHED_ID || REBUILD_ARM == ARM_RB_CALL_FULL_ID || \
    REBUILD_ARM == ARM_XL_STEP_ID || REBUILD_ARM == ARM_FULL_STEP_ID
#define REBUILD_NEEDS_SRCB_VALID 1
#else
#define REBUILD_NEEDS_SRCB_VALID 0
#endif

namespace
{
[[maybe_unused]] constexpr std::uint32_t XL_K = REBUILD_K;
[[maybe_unused]] constexpr bool XL_FUSED      = true;
[[maybe_unused]] constexpr bool TOPK_APPROX   = false;

// row_scale_factor, verbatim from ckernel_sfpu_topk_xl.h:2005.
constexpr int RSF = (REBUILD_K == 512) ? 1 : (REBUILD_K == 1024) ? 2 : 4;
constexpr int NF  = RSF * 2; // faces per transpose sweep

// `_topk_xl_merge_` fused: distance = 64 * num_tiles_per_sequence, and the
// per-column MOP trip count is row_scale_factor * 2 (ckernel_sfpu_topk_xl.h:1689).
constexpr int MERGE_DISTANCE          = 64 * ((REBUILD_K == 2048) ? 2 : 1);
constexpr std::uint32_t MERGE_N_ITERS = RSF * 2;

// SFPSWAP Mod1 = 9: "In all lanes, VD = max and VC = min" (SFPSWAP.md:31). The
// MERGE needs it (it keeps only the max and has to get it into macroVD for the
// macro store); the rebuild does not, and uses p_sfpswap::ALL_ROWS_MAX (= 1),
// the opposite assignment. `p_sfpswap` defines no enum for 9.
constexpr std::uint32_t SFPSWAP_MOD1_VD_GETS_MAX = 9;

// Control arms only.
constexpr std::uint32_t L_CTRL_A = ckernel::p_sfpu::LREG0;
constexpr std::uint32_t L_CTRL_B = ckernel::p_sfpu::LREG1;

constexpr std::uint32_t SFPENCC_MOD1_EI       = 2; // SFPENCC.md:41
constexpr std::uint32_t SFPCFG_IMM16_IS_VALUE = 1; // SFPCONFIG.md:108

// --- SFPLOADMACRO sequence words -------------------------------------------
//
// Simple byte 0x80 SET   -> Insn.VB = macroVD, leaving Insn.VC at the
//                           template's value (SFPLOADMACRO.md:100). The only
//                           way to get a software-loaded register in as the
//                           compare's second operand; with 0x80 clear the macro
//                           assigns Insn.VC = macroVD and SFPSWAP degenerates
//                           to a self-compare.
// Simple byte 0x40 CLEAR -> Insn.VD = macroVD.
// selector 4+m           -> InstructionTemplate[m].
// delay 0                -> the cycle right after the SFPLOADMACRO, consuming
//                           the value that macro just loaded.
constexpr std::uint32_t seq_simple(std::uint32_t m)
{
    return 0x80u | (0u << 3) | (4u + m);
}

// MAD byte: selector 2 = SFPNOP, same delay. Required by SFPLOADMACRO.md:11 (‡).
constexpr std::uint32_t SEQ_MAD = (0u << 3) | 2u;

// Round byte 0 = schedule nothing; discharges the other half of (‡).
constexpr std::uint32_t SEQ_ROUND = 0u;

// Store byte: selector 3 = the built-in SFPSTORE, delay 2.
//   0x40 and 0x80 both CLEAR -> Insn.VD = macroVD, i.e. store the register the
//        scheduled SFPSWAP just wrote. 0x40 would store LReg[16].
//   delay 2 -> the SFPSWAP writes on its second cycle (macro+2), so the store
//        fires at macro+3, one cycle behind its producer. Identical to the
//        merge's, which measured at the load-issue floor with it.
constexpr std::uint32_t SEQ_STORE = (2u << 3) | 3u;

constexpr std::uint32_t sequence_word(std::uint32_t m)
{
    return (SEQ_STORE << 24) | (SEQ_ROUND << 16) | (SEQ_MAD << 8) | seq_simple(m);
}

// Misc (SFPLOADMACRO.md:53-57): StoreMod0 [0:3], UsesLoadMod0ForStore [4:7],
// UnitDelayKind [8:11].
//   0xF0  -> all four macros' stores inherit the LOAD's Mod0 (INT32). The fused
//            [bf16 value | u16 index] word is an opaque sort key; a StoreMod0
//            format conversion would destroy the index in the low half.
//   0xB00 -> Simple (bit 8), MAD (bit 9) and Store (bit 11) on
//            WaitForElapsedInstructions -- instruction-counting rather than
//            cycle-counting, so the delay chain cannot slide if the frontend
//            bubbles at a MOP boundary.
constexpr std::uint32_t MISC_WORD = 0xB00u | 0xF0u;

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

// Every arm below runs the DESCENDING rebuild, matching the K=512 sheet and the
// direction the shipping reduction step uses.
constexpr bool DIR = false;

// ---------------------------------------------------------------------------
// Macro configuration. Four macros, four templates, one shared Misc word.
// MUST run after _llk_math_eltwise_unary_sfpu_init_once_(), which clears
// LaneConfig -- the VD >= 12 backdoor that stores an instruction into
// InstructionTemplate[] instead of executing it is gated on
// LaneConfig.DISABLE_BACKDOOR_LOAD being false (SFPCONFIG.md:45-46, :120).
// ---------------------------------------------------------------------------
inline void write_templates_mod(const std::uint32_t vc0, const std::uint32_t vc1, const std::uint32_t vc2, const std::uint32_t vc3, const std::uint32_t mod1)
{
    TT_SFPSWAP(0, vc0, 12, mod1);
    TT_SFPSWAP(0, vc1, 13, mod1);
    TT_SFPSWAP(0, vc2, 14, mod1);
    TT_SFPSWAP(0, vc3, 15, mod1);
    TTI_SFPNOP;
    TTI_SFPNOP;
}

inline void write_templates(const std::uint32_t vc0, const std::uint32_t vc1, const std::uint32_t vc2, const std::uint32_t vc3)
{
    // lreg_dest = 12 + m is the backdoor index, NOT an operand: it selects which
    // template slot receives the word. The macro overrides Insn.VD with macroVD
    // at issue time, so the 12..15 written here never reaches the Vector Unit.
    TT_SFPSWAP(0, vc0, 12, p_sfpswap::ALL_ROWS_MAX);
    TT_SFPSWAP(0, vc1, 13, p_sfpswap::ALL_ROWS_MAX);
    TT_SFPSWAP(0, vc2, 14, p_sfpswap::ALL_ROWS_MAX);
    TT_SFPSWAP(0, vc3, 15, p_sfpswap::ALL_ROWS_MAX);
    TTI_SFPNOP;
    TTI_SFPNOP;
}

inline void configure_macros(const std::uint32_t vc0, const std::uint32_t vc1, const std::uint32_t vc2, const std::uint32_t vc3)
{
    write_templates(vc0, vc1, vc2, vc3);

    // Sequence[m]. The word does not fit the 16-bit immediate path, so stage it
    // through LReg[0] and write with Mod1 = 0 -- the ckernel_sfpu_mul_int.h
    // idiom. LReg[0] is clobbered, which is harmless: every body below seeds all
    // eight LRegs from loads before touching them.
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
    TTI_SFPCONFIG(MISC_WORD, 8, SFPCFG_IMM16_IS_VALUE);
    TTI_SFPNOP;
    TTI_SFPNOP;
}

// ===========================================================================
//  THE CANDIDATE BODY -- sub-block A / B with SWAP *and* STORE on the macro
// ===========================================================================
//
// `D` is the group-2 offset (64 for sub-block A, 32 for sub-block B); `INC` is
// the Dst advance `store16_rows_x2<D, INC>` would have folded into its LREG7
// store, which here moves to the last SOFTWARE store.
//
// Descending: `bitonic_sort_len_k(false)` is SWAP(VC = LREG0..3, VD = LREG4..7)
// with Mod1 = ALL_ROWS_MAX = "VD = min, VC = max", so LREG4..7 (the macroVDs)
// end up holding the min and belong at the HIGH addresses -- which is exactly
// where their own loads read from, i.e. exactly where the macro's Store slot
// writes. No Mod1 = 9 needed: the shipping lattice's own operand order already
// puts the macro-reachable half where the macro can store it.
//
// Ascending is the mirror: SWAP(VC = LREG4..7, VD = LREG0..3), so the macroVDs
// are LREG0..3, they load from the LOW addresses and store back to them, and
// the plain loads/stores take the LREG4..7 half.
template <int INC>
inline void ce_last_store(const std::uint32_t lreg, const int off)
{
    static_assert(INC == 0 || INC == 16 || INC == 48, "INC must be 0, 16 or 48");
    if constexpr (INC == 48)
    {
        TT_SFPSTORE(lreg, InstrModLoadStore::INT32, ADDR_MOD_1, off);
    }
    else if constexpr (INC == 16)
    {
        TT_SFPSTORE(lreg, InstrModLoadStore::INT32, ADDR_MOD_5, off);
    }
    else
    {
        TT_SFPSTORE(lreg, InstrModLoadStore::INT32, ADDR_MOD_7, off);
    }
}

template <int D, int INC, bool ASC>
inline void macro_ce_body()
{
    if constexpr (ASC)
    {
        TTI_SFPLOAD(p_sfpu::LREG4, InstrModLoadStore::INT32, ADDR_MOD_7, D + 0);
        RB_LOADMACRO(0u, p_sfpu::LREG0, ADDR_MOD_7, 0);
        TTI_SFPLOAD(p_sfpu::LREG5, InstrModLoadStore::INT32, ADDR_MOD_7, D + 4);
        RB_LOADMACRO(1u, p_sfpu::LREG1, ADDR_MOD_7, 4);
        TTI_SFPLOAD(p_sfpu::LREG6, InstrModLoadStore::INT32, ADDR_MOD_7, D + 8);
        RB_LOADMACRO(2u, p_sfpu::LREG2, ADDR_MOD_7, 8);
        TTI_SFPLOAD(p_sfpu::LREG7, InstrModLoadStore::INT32, ADDR_MOD_7, D + 12);
        RB_LOADMACRO(3u, p_sfpu::LREG3, ADDR_MOD_7, 12);
        TTI_SFPNOP; // macro store 2 lands here -- keep it store-free
        TTI_SFPSTORE(p_sfpu::LREG4, InstrModLoadStore::INT32, ADDR_MOD_7, D + 0);
        TTI_SFPNOP; // macro store 3 lands here
        TTI_SFPSTORE(p_sfpu::LREG5, InstrModLoadStore::INT32, ADDR_MOD_7, D + 4);
        TTI_SFPSTORE(p_sfpu::LREG6, InstrModLoadStore::INT32, ADDR_MOD_7, D + 8);
        ce_last_store<INC>(p_sfpu::LREG7, D + 12);
    }
    else
    {
        TTI_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::INT32, ADDR_MOD_7, 0);
        RB_LOADMACRO(0u, p_sfpu::LREG4, ADDR_MOD_7, D + 0);
        TTI_SFPLOAD(p_sfpu::LREG1, InstrModLoadStore::INT32, ADDR_MOD_7, 4);
        RB_LOADMACRO(1u, p_sfpu::LREG5, ADDR_MOD_7, D + 4);
        TTI_SFPLOAD(p_sfpu::LREG2, InstrModLoadStore::INT32, ADDR_MOD_7, 8);
        RB_LOADMACRO(2u, p_sfpu::LREG6, ADDR_MOD_7, D + 8);
        TTI_SFPLOAD(p_sfpu::LREG3, InstrModLoadStore::INT32, ADDR_MOD_7, 12);
        RB_LOADMACRO(3u, p_sfpu::LREG7, ADDR_MOD_7, D + 12);
        TTI_SFPNOP; // macro store 2 lands here -- keep it store-free
        TTI_SFPSTORE(p_sfpu::LREG0, InstrModLoadStore::INT32, ADDR_MOD_7, 0);
        TTI_SFPNOP; // macro store 3 lands here
        TTI_SFPSTORE(p_sfpu::LREG1, InstrModLoadStore::INT32, ADDR_MOD_7, 4);
        TTI_SFPSTORE(p_sfpu::LREG2, InstrModLoadStore::INT32, ADDR_MOD_7, 8);
        ce_last_store<INC>(p_sfpu::LREG3, 12);
    }
}

// ===========================================================================
//  Sub-block bodies, lifted verbatim from `canonical_big_block_with_replay`
// ===========================================================================

// Sub-block A: rsf x (load<64> + sort_k + store<64,16>), replay-recorded.
inline void sub_block_a()
{
    if constexpr (RSF >= 4)
    {
        load_replay_buf<Exec>(0, 8, [] { ckernel::sfpu::load16_rows_x2<64>(); });
        ckernel::sfpu::bitonic_sort_len_k(DIR);
        load_replay_buf<Exec>(8, 8, [] { ckernel::sfpu::store16_rows_x2<64, 16>(); });
        for (int i = 1; i < RSF; i++)
        {
            lltt::replay(0, 8);
            ckernel::sfpu::bitonic_sort_len_k(DIR);
            lltt::replay(8, 8);
        }
        TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D);
    }
}

// Sub-block B: rsf/2 pairs of (load<32> + sort_k + store<32,16|48>), inline.
inline void sub_block_b()
{
    if constexpr (RSF >= 2)
    {
        for (int i = 0; i < (RSF >> 1); i++)
        {
            ckernel::sfpu::load16_rows_x2<32>();
            ckernel::sfpu::bitonic_sort_len_k(DIR);
            ckernel::sfpu::store16_rows_x2<32, 16>();
            ckernel::sfpu::load16_rows_x2<32>();
            ckernel::sfpu::bitonic_sort_len_k(DIR);
            ckernel::sfpu::store16_rows_x2<32, 48>();
        }
        TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D);
    }
}

// Sub-block C: rsf x (load<16> + sort_32 + store<16,32>).
inline void sub_block_c()
{
    if constexpr (RSF >= 2)
    {
        load_replay_buf<Exec>(0, 8, [] { ckernel::sfpu::load16_rows_x2<16>(); });
        ckernel::sfpu::bitonic_sort_len_32(DIR);
        load_replay_buf<Exec>(8, 8, [] { ckernel::sfpu::store16_rows_x2<16, 32>(); });
        for (int i = 1; i < RSF; i++)
        {
            lltt::replay(0, 8);
            ckernel::sfpu::bitonic_sort_len_32(DIR);
            lltt::replay(8, 8);
        }
        TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D);
    }
    else
    {
        ckernel::sfpu::load16_rows_x2<16>();
        ckernel::sfpu::bitonic_sort_len_32(DIR);
        ckernel::sfpu::store16_rows_x2<16, 32>();
        TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D);
    }
}

// Macro versions of A and B. Same loop / replay shape, 14-instruction bodies.
inline void sub_block_a_macro()
{
    if constexpr (RSF >= 4)
    {
        load_replay_buf<Exec>(0, 14, [] { macro_ce_body<64, 16, false>(); });
        for (int i = 1; i < RSF; i++)
        {
            lltt::replay(0, 14);
        }
        TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D);
    }
}

inline void sub_block_b_macro()
{
    if constexpr (RSF >= 2)
    {
        for (int i = 0; i < (RSF >> 1); i++)
        {
            macro_ce_body<32, 16, false>();
            macro_ce_body<32, 48, false>();
        }
        TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D);
    }
}

// Sub-block C with its FIRST level (Step 5 of `bitonic_sort_len_32`) on the
// macros -- the K=512 candidate, generalised. Descending Step 5 is
// SWAP(VC = LREG0..3, VD = LREG4..7), the SAME template VC set sub-blocks A and
// B need, which is why one template programming serves all three.
inline void block_loads_macro()
{
    TTI_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::INT32, ADDR_MOD_7, 0);
    RB_LOADMACRO(0u, p_sfpu::LREG4, ADDR_MOD_7, 16);
    TTI_SFPLOAD(p_sfpu::LREG1, InstrModLoadStore::INT32, ADDR_MOD_7, 4);
    RB_LOADMACRO(1u, p_sfpu::LREG5, ADDR_MOD_7, 20);
    TTI_SFPLOAD(p_sfpu::LREG2, InstrModLoadStore::INT32, ADDR_MOD_7, 8);
    RB_LOADMACRO(2u, p_sfpu::LREG6, ADDR_MOD_7, 24);
    TTI_SFPLOAD(p_sfpu::LREG3, InstrModLoadStore::INT32, ADDR_MOD_7, 12);
    RB_LOADMACRO(3u, p_sfpu::LREG7, ADDR_MOD_7, 28);
    TTI_SFPNOP; // drain: the last macro's SFPSWAP owns Simple for two cycles
    TTI_SFPNOP;
}

inline void sub_block_c_macro()
{
    if constexpr (RSF >= 2)
    {
        load_replay_buf<Exec>(0, 10, [] { block_loads_macro(); });
        ckernel::sfpu::bitonic_sort_len_16_alt<true>(DIR);
        load_replay_buf<Exec>(10, 8, [] { ckernel::sfpu::store16_rows_x2<16, 32>(); });
        for (int i = 1; i < RSF; i++)
        {
            lltt::replay(0, 10);
            ckernel::sfpu::bitonic_sort_len_16_alt<true>(DIR);
            lltt::replay(10, 8);
        }
        TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D);
    }
    else
    {
        block_loads_macro();
        ckernel::sfpu::bitonic_sort_len_16_alt<true>(DIR);
        ckernel::sfpu::store16_rows_x2<16, 32>();
        TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D);
    }
}

// ===========================================================================
//  Phase bodies
// ===========================================================================

inline void rb_transpose_sweeps()
{
    TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::WAIT_SFPU | p_stall::SRCA_VLD | p_stall::SRCB_VLD);
    ckernel::sfpu::enter_transpose_cfg_block();
    ckernel::sfpu::transpose_N_faces<NF, true, 256, false>();
    ckernel::sfpu::transpose_N_faces<NF, true, 256, false>();
    ckernel::sfpu::leave_transpose_cfg_block();
    TTI_SETRWC(p_setrwc::CLR_AB, 0, 0, 0, 0, p_setrwc::SET_ABD);
}

// The stride-2 + sort_16_alt build phase, verbatim from the shipping rebuild.
inline void rb_build_phase()
{
    ckernel::sfpu::topk_rebuild_build2048_mop_config();
    load_replay_buf<Exec>(
        0,
        16,
        []
        {
            ckernel::sfpu::load16_rows_x2<2>();
            ckernel::sfpu::bitonic_sort_len_16_alt_swaps<true>(DIR);
        });
    TTI_SFPTRANSP(0, 0, 0, 0);
    load_replay_buf<Exec>(16, 8, [] { ckernel::sfpu::bitonic_sort_len_16_alt_swaps<true>(DIR); });
    TTI_SFPTRANSP(0, 0, 0, 0);
    load_replay_buf<Exec>(24, 8, [] { ckernel::sfpu::store16_rows_x2<2, 16>(); });
    ckernel_unpack_template::run(RSF * 2 - 1);
    TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D);
}

// Build phase with the first level (Step 4) on the macros. Recorded body is 30
// instructions rather than 34, so the 5-slot MOP template's REPLAY ranges shift.
inline void rb_build_phase_macro()
{
    constexpr std::uint32_t replay_a  = lltt::replay_insn(0, 14);
    constexpr std::uint32_t replay_b  = lltt::replay_insn(14, 8);
    constexpr std::uint32_t replay_st = lltt::replay_insn(22, 8);
    constexpr std::uint32_t transp_op = TT_OP_SFPTRANSP(0, 0, 0, 0);

    ckernel_unpack_template tmpl(true, true, replay_a, transp_op, replay_b, transp_op, TT_OP_NOP, replay_st, TT_OP_NOP);
    tmpl.program();

    load_replay_buf<Exec>(
        0,
        14,
        []
        {
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
    load_replay_buf<Exec>(14, 8, [] { ckernel::sfpu::bitonic_sort_len_16_alt_swaps<true>(DIR); });
    TTI_SFPTRANSP(0, 0, 0, 0);
    load_replay_buf<Exec>(22, 8, [] { ckernel::sfpu::store16_rows_x2<2, 16>(); });
    ckernel_unpack_template::run(RSF * 2 - 1);
    TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D);
}

inline void rb_block_phase(const std::uint32_t tile_offset)
{
    for (int col = 0; col < 2; col++)
    {
        ckernel::sfpu::canonical_big_block_with_replay<RSF>(DIR);
        ckernel::sfpu::set_dst_write_addr_offset(tile_offset + (col ? 0 : 2));
    }
}

// Two-column drivers for the isolated sub-block arms. Each sub-block ends with
// the same SETRWC(SET_D) the shipping helper uses, so the Dst pointer resets
// every pass and REBUILD_ITER_COUNT iterations stay in range. The
// `set_dst_write_addr_offset` between columns is part of the shipping
// `canonical_big_block` loop and is kept so the isolated rows add up to the
// RbBlock row.
#define TWO_COLS(BODY)                                             \
    do                                                             \
    {                                                              \
        for (int col = 0; col < 2; col++)                          \
        {                                                          \
            BODY;                                                  \
            ckernel::sfpu::set_dst_write_addr_offset(col ? 0 : 2); \
        }                                                          \
    } while (0)

// ===========================================================================
//  Full-call candidates
// ===========================================================================
//
// `transpose_dest_face_32b<dst_offset>` with two instruction words injected into
// the two MOVD2B->TRNSPSRCB shadows, which `topk_rebuild_perf.cpp`'s
// RbXposeNFill arm measured as free (142.996 against 143.000 with 24 extra
// SFPNOPs). Byte-for-byte the shipping helper otherwise.
template <int dst_offset>
inline void face_inject(const std::uint32_t w1, const std::uint32_t w2)
{
    cfg_reg_rmw_tensix<ALU_FORMAT_SPEC_REG0_SrcA_RMW>(to_underlying(DataFormat::Float16_b));
    TTI_MOVD2B(1, 16, ADDR_MOD_7, p_movd2b::MOV_4_ROWS, dst_offset + 0);
    TTI_MOVD2B(1, 20, ADDR_MOD_7, p_movd2b::MOV_4_ROWS, dst_offset + 4);
    TTI_MOVD2B(1, 24, ADDR_MOD_7, p_movd2b::MOV_4_ROWS, dst_offset + 8);
    TTI_MOVD2B(1, 28, ADDR_MOD_7, p_movd2b::MOV_4_ROWS, dst_offset + 12);
    ckernel::instrn_buffer[0] = w1;
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
    ckernel::instrn_buffer[0] = w2;
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

// One `transpose_N_faces<NF>` sweep that also installs the four
// InstructionTemplates, in the first two faces' four free shadow slots.
inline void sweep_with_templates(const std::uint32_t vc0, const std::uint32_t vc1, const std::uint32_t vc2, const std::uint32_t vc3)
{
    face_inject<0>(TT_OP_SFPSWAP(0, vc0, 12, p_sfpswap::ALL_ROWS_MAX), TT_OP_SFPSWAP(0, vc1, 13, p_sfpswap::ALL_ROWS_MAX));
    face_inject<16>(TT_OP_SFPSWAP(0, vc2, 14, p_sfpswap::ALL_ROWS_MAX), TT_OP_SFPSWAP(0, vc3, 15, p_sfpswap::ALL_ROWS_MAX));
    if constexpr (NF > 2)
    {
        face_inject<32>(TT_OP_SFPNOP, TT_OP_SFPNOP);
        face_inject<48>(TT_OP_SFPNOP, TT_OP_SFPNOP);
    }
    if constexpr (NF > 4)
    {
        face_inject<64>(TT_OP_SFPNOP, TT_OP_SFPNOP);
        face_inject<80>(TT_OP_SFPNOP, TT_OP_SFPNOP);
        face_inject<96>(TT_OP_SFPNOP, TT_OP_SFPNOP);
        face_inject<112>(TT_OP_SFPNOP, TT_OP_SFPNOP);
    }
}

// `FULL_AB` = false reproduces the K=512 candidate generalised to K (first
// lattice level of the build phase and of sub-block C on the macros, sub-blocks
// A and B on the shipping helpers). `FULL_AB` = true additionally puts sub-block
// A and B's single-level compare-exchange AND their stores on the macros.
template <bool FULL_AB>
inline void macro_rebuild_call(const std::uint32_t dst_index)
{
    const std::uint32_t tile_offset = dst_index << DstTileSizeLog2[DstTileShape::Tile32x32];

    TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::WAIT_SFPU | p_stall::SRCA_VLD | p_stall::SRCB_VLD);
    ckernel::sfpu::enter_transpose_cfg_block();

    // Sweep 1 installs the BUILD phase's Step-4 templates (VC = LREG0,1,4,5).
    sweep_with_templates(p_sfpu::LREG0, p_sfpu::LREG1, p_sfpu::LREG4, p_sfpu::LREG5);
    rb_build_phase_macro();

    // Sweep 2 installs the templates the whole big block needs -- Step 5 of
    // sub-block C and `bitonic_sort_len_k` of sub-blocks A and B are the SAME
    // level shape (VC = LREG0..3 descending), so one programming covers all
    // three sub-blocks and nothing is reprogrammed mid-block.
    sweep_with_templates(p_sfpu::LREG0, p_sfpu::LREG1, p_sfpu::LREG2, p_sfpu::LREG3);
    ckernel::sfpu::leave_transpose_cfg_block();

    for (int col = 0; col < 2; col++)
    {
        if constexpr (FULL_AB)
        {
            sub_block_a_macro();
            sub_block_b_macro();
        }
        else
        {
            sub_block_a();
            sub_block_b();
        }
        sub_block_c_macro();
        ckernel::sfpu::set_dst_write_addr_offset(tile_offset + (col ? 0 : 2));
    }

    // The build phase clobbered the MOP Expander config; restore the merge
    // programming exactly as the shipping rebuild does.
    ckernel::sfpu::topk_mop_config<true>();
    TTI_SETRWC(p_setrwc::CLR_AB, 0, 0, 0, 0, p_setrwc::SET_ABD);
}

// ===========================================================================
//  The proven macro merge (topk_merge_macro_perf.cpp), K-parameterised
// ===========================================================================
//
// 8 instructions for 8 input vectors. Under Mod1 = 9 the macroVD (the run-A
// register the macro's own load just wrote) receives the MAX, and the macro's
// Store slot writes it back over run A -- which is exactly where the merged max
// belongs. The Dst advance rides the last macro's LOAD, not a store, because a
// macro-scheduled SFPSTORE skips ApplyPartialAddrMod entirely
// (SFPLOADMACRO.md:139) and its address was already resolved at issue (:140).
inline void macro_merge_body()
{
    TTI_SFPLOAD(p_sfpu::LREG4, InstrModLoadStore::INT32, ADDR_MOD_7, MERGE_DISTANCE + 0);
    RB_LOADMACRO(0u, p_sfpu::LREG0, ADDR_MOD_7, 0);
    TTI_SFPLOAD(p_sfpu::LREG5, InstrModLoadStore::INT32, ADDR_MOD_7, MERGE_DISTANCE + 4);
    RB_LOADMACRO(1u, p_sfpu::LREG1, ADDR_MOD_7, 4);
    TTI_SFPLOAD(p_sfpu::LREG6, InstrModLoadStore::INT32, ADDR_MOD_7, MERGE_DISTANCE + 8);
    RB_LOADMACRO(2u, p_sfpu::LREG2, ADDR_MOD_7, 8);
    TTI_SFPLOAD(p_sfpu::LREG7, InstrModLoadStore::INT32, ADDR_MOD_7, MERGE_DISTANCE + 12);
    RB_LOADMACRO(3u, p_sfpu::LREG3, ADDR_MOD_5, 12);
}

// Drop-in for `_topk_xl_merge_<K, false, true>`: identical envelope, Dst window
// and column split (ckernel_sfpu_topk_xl.h:1683-1739).
inline void macro_merge_call(const std::uint32_t dst_index)
{
    const std::uint32_t tile_offset = dst_index << DstTileSizeLog2[DstTileShape::Tile32x32];

    load_replay_buf<Exec>(0, 8, [] { macro_merge_body(); });
    ckernel_unpack_template::run(MERGE_N_ITERS - 1);
    TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D);
    ckernel::sfpu::set_dst_write_addr_offset(tile_offset + 2);
    ckernel_unpack_template::run(MERGE_N_ITERS);
    TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D);
    ckernel::sfpu::set_dst_write_addr_offset(tile_offset + 0);

    // Retire the last two scheduled SFPSTOREs: under WaitForElapsedInstructions
    // their counters only move when this thread issues an SFPU instruction, and
    // TTI_SETRWC / TT_SETC16 are not SFPU instructions.
    TTI_SFPNOP;
    TTI_SFPNOP;
    TTI_SFPNOP;
}
} // namespace

void run_kernel(RUNTIME_PARAMETERS params)
{
    {
        START_PERF_MEASURE("INIT")

        // Kernel-invariant SFPU init: SFPCONFIG(0, 0xF, 1) plus ADDR_MOD_7. Also
        // clears LaneConfig, the precondition for the VD >= 12 backdoor template
        // writes. Must come first.
        _llk_math_eltwise_unary_sfpu_init_once_();

        // Programs ADDR_MOD_1/5/6 and the merge MOP template. Run for EVERY arm
        // so the arms differ by exactly the body under test.
        ckernel::sfpu::_topk_xl_init_<XL_K, XL_FUSED>();

        // Clear stale lane predication. SFPSWAP's writes are gated on LaneEnabled
        // (SFPSWAP.md:38), so a mask left behind by a previously-run kernel would
        // silently suppress work.
        TTI_SFPENCC(0, 0, 0, SFPENCC_MOD1_EI);

#if REBUILD_ARM == ARM_MACRO_MERGE_ID
        // The MERGE's templates: VC = LREG4..7 (the run-B holders) and Mod1 = 9
        // so macroVD gets the max. Sequence and Misc are identical to the
        // rebuild's -- the two macro users differ in exactly the four templates.
        write_templates_mod(p_sfpu::LREG4, p_sfpu::LREG5, p_sfpu::LREG6, p_sfpu::LREG7, SFPSWAP_MOD1_VD_GETS_MAX);
        configure_macros(p_sfpu::LREG4, p_sfpu::LREG5, p_sfpu::LREG6, p_sfpu::LREG7);
        write_templates_mod(p_sfpu::LREG4, p_sfpu::LREG5, p_sfpu::LREG6, p_sfpu::LREG7, SFPSWAP_MOD1_VD_GETS_MAX);
#elif REBUILD_ARM == ARM_RB_SUB_AB_MACRO_ID || REBUILD_ARM == ARM_RB_BLOCK_FULL_ID || REBUILD_ARM == ARM_RB_CALL_SCHED_ID || \
    REBUILD_ARM == ARM_RB_CALL_FULL_ID || REBUILD_ARM == ARM_FULL_STEP_ID
        // Descending: `bitonic_sort_len_k` and `bitonic_sort_len_32`'s Step 5 are
        // both SWAP(VC = LREG0..3, VD = LREG4..7). The full-call arms overwrite
        // templates 2 and 3 per phase from inside the transposes; the isolated
        // block arms use this programming as-is.
        configure_macros(p_sfpu::LREG0, p_sfpu::LREG1, p_sfpu::LREG2, p_sfpu::LREG3);
#endif

        // Establish the Dst window once.
        _llk_math_eltwise_sfpu_start_(0);

        PROFILER_SYNC();
    }

    {
        START_PERF_MEASURE("TILE_LOOP")

#if REBUILD_ARM == ARM_CTRL_LOAD_ID
        {
            // CONTROL -- frontend floor. Deliberately a PLAIN load: a macro issue
            // in a control arm would run against whatever LoadMacroConfig a
            // previously executed kernel left behind.
            load_replay_buf<NoExec>(
                0,
                2,
                []
                {
                    TTI_SFPLOAD(L_CTRL_A, InstrModLoadStore::INT32, ADDR_MOD_7, 0);
                    TTI_SFPLOAD(L_CTRL_B, InstrModLoadStore::INT32, ADDR_MOD_7, 2);
                });
            ckernel_unpack_template::lA(lltt::replay_insn(0, 2), TT_OP_NOP).program();
            mop_run_all();
        }
#elif REBUILD_ARM == ARM_CTRL_SWAP_ID
        {
            // CONTROL -- the tripwire. MUST come out at ~2.0x ARM_CTRL_LOAD.
            load_replay_buf<NoExec>(
                0,
                2,
                []
                {
                    TTI_SFPSWAP(0, L_CTRL_A, L_CTRL_B, p_sfpswap::ALL_ROWS_MAX);
                    TTI_SFPSWAP(0, L_CTRL_B, L_CTRL_A, p_sfpswap::ALL_ROWS_MAX);
                });
            ckernel_unpack_template::lA(lltt::replay_insn(0, 2), TT_OP_NOP).program();
            mop_run_all();
        }
#elif REBUILD_ARM == ARM_RB_CALL_ID
        {
#pragma GCC unroll 1
            for (std::uint32_t i = 0; i < REBUILD_ITER_COUNT; ++i)
            {
                _llk_math_eltwise_sfpu_start_(0);
                ckernel::sfpu::_topk_xl_rebuild_<XL_K, TOPK_APPROX, XL_FUSED>(0 /* dst_index */, DIR);
                _llk_math_eltwise_sfpu_done_();
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
#elif REBUILD_ARM == ARM_RB_BUILD_ID
        {
#pragma GCC unroll 1
            for (std::uint32_t i = 0; i < REBUILD_ITER_COUNT; ++i)
            {
                rb_build_phase();
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
#elif REBUILD_ARM == ARM_RB_SUB_A_ID
        {
#pragma GCC unroll 1
            for (std::uint32_t i = 0; i < REBUILD_ITER_COUNT; ++i)
            {
                TWO_COLS(sub_block_a());
            }
        }
#elif REBUILD_ARM == ARM_RB_SUB_B_ID
        {
#pragma GCC unroll 1
            for (std::uint32_t i = 0; i < REBUILD_ITER_COUNT; ++i)
            {
                TWO_COLS(sub_block_b());
            }
        }
#elif REBUILD_ARM == ARM_RB_SUB_C_ID
        {
#pragma GCC unroll 1
            for (std::uint32_t i = 0; i < REBUILD_ITER_COUNT; ++i)
            {
                TWO_COLS(sub_block_c());
            }
        }
#elif REBUILD_ARM == ARM_RB_SUB_AB_MACRO_ID
        {
#pragma GCC unroll 1
            for (std::uint32_t i = 0; i < REBUILD_ITER_COUNT; ++i)
            {
                TWO_COLS(sub_block_a_macro(); sub_block_b_macro());
            }
        }
#elif REBUILD_ARM == ARM_RB_BLOCK_FULL_ID
        {
#pragma GCC unroll 1
            for (std::uint32_t i = 0; i < REBUILD_ITER_COUNT; ++i)
            {
                TWO_COLS(sub_block_a_macro(); sub_block_b_macro(); sub_block_c_macro());
            }
        }
#elif REBUILD_ARM == ARM_RB_CALL_SCHED_ID
        {
#pragma GCC unroll 1
            for (std::uint32_t i = 0; i < REBUILD_ITER_COUNT; ++i)
            {
                _llk_math_eltwise_sfpu_start_(0);
                macro_rebuild_call<false>(0 /* dst_index */);
                _llk_math_eltwise_sfpu_done_();
            }
        }
#elif REBUILD_ARM == ARM_RB_CALL_FULL_ID
        {
#pragma GCC unroll 1
            for (std::uint32_t i = 0; i < REBUILD_ITER_COUNT; ++i)
            {
                _llk_math_eltwise_sfpu_start_(0);
                macro_rebuild_call<true>(0 /* dst_index */);
                _llk_math_eltwise_sfpu_done_();
            }
        }
#elif REBUILD_ARM == ARM_XL_MERGE_ID
        {
#pragma GCC unroll 1
            for (std::uint32_t i = 0; i < REBUILD_ITER_COUNT; ++i)
            {
                _llk_math_eltwise_sfpu_start_(0);
                ckernel::sfpu::_topk_xl_merge_<XL_K, TOPK_APPROX, XL_FUSED>(0 /* dst_index */);
                _llk_math_eltwise_sfpu_done_();
            }
        }
#elif REBUILD_ARM == ARM_MACRO_MERGE_ID
        {
#pragma GCC unroll 1
            for (std::uint32_t i = 0; i < REBUILD_ITER_COUNT; ++i)
            {
                _llk_math_eltwise_sfpu_start_(0);
                // The macro body is 8 instructions; `_topk_xl_init_` left the MOP
                // template at REPLAY(0, 16) for the shipping 16-instruction one.
                ckernel_unpack_template::lA(lltt::replay_insn(0, 8), TT_OP_NOP).program();
                macro_merge_call(0 /* dst_index */);
                _llk_math_eltwise_sfpu_done_();
            }
        }
#elif REBUILD_ARM == ARM_XL_STEP_ID
        {
#pragma GCC unroll 1
            for (std::uint32_t i = 0; i < REBUILD_ITER_COUNT; ++i)
            {
                _llk_math_eltwise_sfpu_start_(0);
                ckernel::sfpu::_topk_xl_merge_<XL_K, TOPK_APPROX, XL_FUSED>(0 /* dst_index */);
                ckernel::sfpu::_topk_xl_rebuild_<XL_K, TOPK_APPROX, XL_FUSED>(0 /* dst_index */, DIR);
                _llk_math_eltwise_sfpu_done_();
            }
        }
#else // ARM_FULL_STEP_ID
        {
#pragma GCC unroll 1
            for (std::uint32_t i = 0; i < REBUILD_ITER_COUNT; ++i)
            {
                _llk_math_eltwise_sfpu_start_(0);
                // The merge and the rebuild want DIFFERENT InstructionTemplates
                // (merge: VC = LREG4..7 with Mod1 = 9; rebuild: VC per phase with
                // Mod1 = 1), and the merge's cannot ride a transpose shadow --
                // there is no transpose before it. So the step pays 6 real
                // instructions here, and this arm includes them.
                write_templates_mod(p_sfpu::LREG4, p_sfpu::LREG5, p_sfpu::LREG6, p_sfpu::LREG7, SFPSWAP_MOD1_VD_GETS_MAX);
                ckernel_unpack_template::lA(lltt::replay_insn(0, 8), TT_OP_NOP).program();
                macro_merge_call(0 /* dst_index */);
                // The rebuild's own two transpose sweeps reinstall its templates
                // for free, so nothing is paid on this side of the boundary.
                macro_rebuild_call<true>(0 /* dst_index */);
                _llk_math_eltwise_sfpu_done_();
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
