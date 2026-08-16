// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// ============================================================================
//  Beating `_topk_xl_merge_` with an SFPLOADMACRO-scheduled compare-exchange
//  (Blackhole only)
// ============================================================================
//
// WHAT `_topk_xl_merge_` ACTUALLY DOES
// ------------------------------------
// Despite the name, the fused K=512 merge body is NOT a bitonic lattice. It is
// ONE compare-exchange level -- the first step of a bitonic merge -- with the
// min half thrown away:
//
//     load16_rows_x2<64>()        8 x SFPLOAD   (LREG0..3 = run A, LREG4..7 = run B)
//     bitonic_sort_len_k(false)   4 x SFPSWAP   (VC = LREG0..3 gets max)
//     store4_rows_top_only<16>()  4 x SFPSTORE  (LREG0..3 only; min half is dead)
//     ------------------------------------------------------------------------
//                                16 instructions, 8 distinct 32-element input
//                                vectors, 4 output vectors.
//
// COST MODEL, from the already-measured primitives (all same harness, same
// silicon): SFPLOAD = 1 cycle IPC 1, SFPSTORE = 1 cycle IPC 1, SFPSWAP = 2
// backend cycles with a hardware-inserted non-fillable bubble (SFPSWAP.md:110,
// confirmed by the ReplaySwap control landing at exactly 2.000).
//
//     8 * 1 (load) + 4 * 2 (swap) + 4 * 1 (store) = 20 cycles / 8 vectors
//                                                 = 2.500 cyc/vector
//
// `_topk_xl_merge_<512,false,true>` measures 2.844 cyc/vector in
// perf_topk_micro_op.py, i.e. 91.0 cycles for its 32-vector call. 4 bodies x
// 20 = 80 cycles of body plus ~11 cycles of per-call envelope (the
// `load_replay_buf<Exec>` REPLAY push, 2 TTI_MOP pushes, 2 TTI_SETRWC, 2
// TT_SETC16, and the `_llk_math_eltwise_sfpu_start_/done_` pair the harness
// wraps it in). ARM_XL_BODY below isolates the 80 from the 11 by MOP-firing
// the body with no envelope at all; ARM_XL_CALL reproduces the 2.844 in THIS
// harness so the two are known to be measuring the same thing.
//
// So the merge is at its own floor FOR THAT INSTRUCTION MIX. The slack is not
// in the schedule, it is in the mix: 12 of the 20 cycles are spent on work
// that a single SFPLOADMACRO can carry for free.
//
// THE OBSERVATION
// ---------------
// Three facts, none of them new, that nothing in the shipping sort/topk tree
// has ever put together:
//
//   1. `SFPSWAP` is legal in an `SFPLOADMACRO` Simple slot -- it is in the
//      Simple column of SFPLOADMACRO.md:7, with footnote (‡) spelling out the
//      exact constraints for scheduling it.
//   2. A macro-scheduled `SFPSTORE` writes to the ADDRESS THE LOAD USED
//      (SFPLOADMACRO.md:140). For a merge that is not a limitation, it is
//      precisely the wanted behaviour: the merged max belongs where run A was.
//   3. The merge needs only the max. `SFPSWAP` computes min AND max, and
//      `store4_rows_top_only` already documents that the min half is dead. The
//      min is therefore free collateral, not a second result to be routed.
//
// Put together, one output vector costs exactly two instructions:
//
//     SFPLOAD      L_B[m]  <- B[i]                        (plain, 1 cycle)
//     SFPLOADMACRO L_A[m]  <- A[i]                        (1 cycle) and, for free:
//                  Simple  : SFPSWAP(VC = L_B[m], VD = macroVD)  -> macroVD = max
//                  MAD     : SFPNOP                        (required by (‡))
//                  Store   : SFPSTORE macroVD -> A[i]      (address latched by the load)
//
// That is 8 instructions per body against 16, and -- because the SFPSWAP no
// longer occupies a software issue slot -- 8 cycles against 20.
//
//     PREDICTION (made before any measurement): 8 cycles / 8 vectors
//                                             = 1.000 cyc/vector for the body,
//     which is the ARCHITECTURAL FLOOR: every 32-element input vector needs its
//     own load-class instruction and the frontend dequeues at most one
//     instruction per thread per cycle (PushTensixInstruction.md:19).
//
//     With the same ~11-cycle per-call envelope, a full 32-vector merge call is
//     predicted at (4 * 8 + 11 + 3) / 32 = 1.44 cyc/vector against 2.844.
//     (+3 = the SFPNOP drain at the end of the call, see DRAINING below.)
//
// WHY NOT SFPGT -- THE EXACT INSTRUCTION COUNT
// --------------------------------------------
// SFPGT/SFPLE are new in Blackhole, 1 cycle, IPC 1, and produce a -1/0 mask as
// a VALUE (SFPGT.md:29) rather than a condition code. The obvious question is
// whether a 1-cycle compare can undercut a 2-cycle SFPSWAP. It cannot, and the
// count is not close:
//
//   compare-exchange (need max only), as software instructions
//     SFPSWAP                                              1 instr, 2 cycles
//     SFPGT + blend                                        4 instr, 4 cycles
//         SFPGT   m = (a > b)          -> -1/0
//         SFPXOR  t = a ^ b
//         SFPAND  t = t & m
//         SFPXOR  r = b ^ t
//     SFPGT + arithmetic blend                             3 instr, 3+ cycles
//         SFPMOV/SFPADD d = a - b ; SFPGT mask ; SFPMAD b + mask*d
//         -- and the mask is -1/0 as an INTEGER, not 1.0f, so it needs an
//            SFPCAST (Simple) before it can multiply. 4 instructions.
//     SFPSETCC + SFPMOV (the pre-Blackhole idiom)           3 instr (setcc, mov, encc-restore)
//
// SFPGT wins only when the mask ITSELF is the answer (the MaskStore filter,
// measured at 1.003). For a compare-exchange it loses 2:1 or worse, and every
// blend variant needs at least two Simple-sub-unit instructions, so a macro
// cannot host them either (SFPLOADMACRO.md:5 -- at most ONE Simple per macro).
// The zero SFPGT sites in the 230-SFPSWAP census are not an oversight.
//
// The winning move is not to replace SFPSWAP but to STOP PAYING FOR IT: leave
// it at 2 backend cycles and hide those 2 cycles under the 2 cycles the load
// pair already costs.
//
// MACRO PLUMBING -- THE THREE THINGS THAT HAD TO BE GOT RIGHT
// -----------------------------------------------------------
// (a) WHICH OPERAND ENDS UP IN macroVD.
//     `p_sfpswap::ALL_ROWS_MAX` is Mod1 = 1 = SFPSWAP_MOD1_VEC_MIN_MAX, which
//     is "VD = min and VC = max" (SFPSWAP.md:23). The macro ALWAYS overrides
//     Insn.VD (SFPLOADMACRO.md:111-115) -- there is no sequence bit that leaves
//     VD alone for a non-Store sub-unit -- so under Mod1 = 1 macroVD would
//     receive the MIN and the max would land in the template's fixed VC, which
//     the Store slot cannot reach. Mod1 = 9 is the documented inverse ("In all
//     lanes, VD = max and VC = min", SFPSWAP.md:31); it has no `p_sfpswap`
//     enum, which is why it is spelled out here. With Mod1 = 9:
//         macroVD (just loaded with A[i]) <- max(A[i], B[i])   -- stored
//         template VC (holding B[i])      <- min                -- dead, reloaded
//
// (b) WHICH FIELD THE MACRO OVERRIDES.
//     Sequence bit 0x80 SET means `Insn.VB = macroVD` (SFPLOADMACRO.md:100).
//     SFPSWAP's functional model reads only VC and VD, so assigning VB is
//     inert and -- crucially -- VC is LEFT AT THE TEMPLATE'S VALUE. That is the
//     only way to get a software-loaded register in as the second operand.
//     With 0x80 CLEAR the macro would instead do `Insn.VC = macroVD`, giving
//     SFPSWAP(macroVD, macroVD): a degenerate self-compare.
//
// (c) TEMPLATE VC IS A CONSTANT, SO THE B REGISTER CANNOT PING-PONG PER
//     INSTRUCTION. `InstructionTemplate[k]` is an immutable instruction word,
//     so its VC is baked in. The rotation therefore has to come from the MACRO
//     INDEX: four macros (0..3) bound to four templates (0..3), whose VCs are
//     LREG4..LREG7. The A side rotates through LREG0..LREG3 via the
//     SFPLOADMACRO's own VD field. Eight registers, exactly the eight the
//     hardware allows a macro to target (VD is u3, SFPLOADMACRO.md:45).
//
// TIMING, AND WHY IT DOES NOT COLLIDE
// -----------------------------------
// One SFPU instruction issues per cycle (REPLAY/MOP-fed, so the RISC-V is out
// of the loop). Body = 8 instructions; i0..i7 below are both issue index and
// cycle index.
//
//   i0  SFPLOAD  L_B0 <- B[0]
//   i1  MACRO0   L_A0 <- A[0]    sched Simple(SWAP,d0) MAD(NOP,d0) Store(d2)
//   i2  SFPLOAD  L_B1 <- B[1]              | SWAP0 cycle 1 (reads L_B0, L_A0)
//   i3  MACRO1   L_A1 <- A[1]              | SWAP0 cycle 2 (writes both)
//   i4  SFPLOAD  L_B2 <- B[2]              | SWAP1 cycle 1   STORE0: L_A0 -> A[0]
//   i5  MACRO2   L_A2 <- A[2]              | SWAP1 cycle 2
//   i6  SFPLOAD  L_B3 <- B[3]              | SWAP2 cycle 1   STORE1: L_A1 -> A[1]
//   i7  MACRO3   L_A3 <- A[3]  (ADDR_MOD_5: Dst += 16)
//
// * SFPSWAP occupies the Simple sub-unit for two cycles (SFPSWAP.md:113) and
//   the macros are two cycles apart, so consecutive SFPSWAPs abut exactly and
//   never overlap. This is the whole trick: the swap is not free, it is
//   PERFECTLY HIDDEN behind the load pair it is sandwiched between.
// * Footnote (‡) of SFPLOADMACRO.md:11 demands SFPNOP on the MAD sub-unit at
//   the same time as the SFPSWAP, and Simple+Round idle on the following
//   cycle. MAD carries the required SFPNOP; Round is never scheduled; the next
//   SFPSWAP is two cycles out, not one. All three satisfied by construction.
// * The Store fires at delay 2, one cycle after the SFPSWAP's write cycle --
//   the same one-cycle producer/consumer margin ckernel_sfpu_mul_int.h uses.
// * Auto-stall after SFPSWAP does NOT apply to macro-scheduled ones
//   (SFPSWAP.md:113 CAUTION), which is what keeps the issue rate at 1/cycle.
// * A/B both rotate 4 deep, so a register is rewritten 8 cycles after its last
//   read. The 2-deep rotation would have left a 1-cycle margin on L_A.
//
// DRAINING
// --------
// The sub-unit delays are `WaitForElapsedInstructions` (Misc bits 8/9/11), so
// a scheduled instruction counts down only when THIS THREAD ISSUES AN SFPU
// INSTRUCTION. TTI_SETRWC and TT_SETC16 are not SFPU instructions. Two
// consequences, both handled:
//   * Between the two MOP runs of a merge call the pending stores simply drain
//     into the first two instructions of the next run. Their Dst addresses were
//     latched at SFPLOADMACRO time (SFPLOADMACRO.md:140) so the intervening
//     TT_SETC16 write of DEST_TARGET_REG_CFG_MATH_Offset cannot corrupt them,
//     and the body length is even so the store phase is preserved.
//   * At the END of a call there is no following SFPU instruction, so three
//     TTI_SFPNOPs are issued to retire the last two scheduled stores. That is
//     the +3 in the prediction, and it is a real cost a production caller pays.
//
// ARMS
// ----
//   CtrlLoad   control -- replay+MOP-fed plain SFPLOAD. MUST be ~1.000.
//   CtrlSwap   control -- replay+MOP-fed plain SFPSWAP. MUST be ~2.000.
//              Shared verbatim with perf_sfpu_count_above.py and
//              perf_topk_micro_op.py so the three harnesses are stitched.
//   XlCall     `_topk_xl_merge_<512,false,true>` in its shipping envelope.
//              Reproduces perf_topk_micro_op.py's 2.844 in this harness.
//   XlBody     the same 16-instruction body, MOP-fired with no envelope.
//              Predicted 2.500 -- the pure instruction-mix cost.
//   MacroBody  the 8-instruction macro body, MOP-fired. Predicted 1.000.
//   MacroCall  the macro body in the SAME envelope as XlCall, so the two rows
//              are the head-to-head number. Predicted ~1.44.
//
// MEASURED ON BLACKHOLE SILICON. Two-point slope, MATH_ISOLATE, 5 runs/point.
// Every one of the six predictions above was recorded before the first run and
// every one landed:
//
//   arm          cyc/body   cyc/vector   PREDICTED
//   CtrlLoad        2.000       1.000      1.000    control, frontend floor
//   CtrlSwap        4.000       2.001      2.000    control, the tripwire
//   XlCall         91.000       2.844      2.844    reproduces perf_topk_micro_op.py
//   XlBody         19.992       2.499      2.500    pure instruction mix
//   MacroBody       8.000       1.000      1.000    THE CANDIDATE, at the floor
//   MacroCall      46.000       1.438      1.438    head-to-head vs XlCall
//
//     merge body : 2.499 -> 1.000  =  2.499x
//     merge call : 2.844 -> 1.438  =  1.978x
//
// Correctness is NOT established by any of this. A misconfigured (or all-zero,
// i.e. "schedule nothing") LoadMacroConfig.Sequence degenerates an
// SFPLOADMACRO into a plain SFPLOAD, which measures the SAME 1.000 -- the
// scheduled SFPSWAP and SFPSTORE ride free sub-units either way, so the issue
// rate cannot tell a working macro from a dead one.
// `tests/python_tests/test_topk_merge_macro.py` is what establishes it: 71/71
// against the shipping torch golden at K = 512/1024/2048, and mutating
// SFPSWAP's Mod1 from 9 to 1 turns that green into a top-K value mismatch.
//
// CONTEXT ARMS, added afterwards. A merge is only half of a K-reduction step:
// `_topk_xl_merge_` leaves the survivors ordered but no longer bitonic, so
// every merge is followed by `_topk_xl_rebuild_`.
//
//   XlRebuild     374.000      23.375    (normalised by K/32 = 16 vectors)
//   XlStep        464.012      14.500     merge + rebuild, shipping
//   MacroStep     419.000      13.094     macro merge + rebuild
//
//     step: 14.500 -> 13.094 = 1.107x, and the delta is 1.407 cyc/vector
//     against the 1.406 predicted from (XlCall - MacroCall) -- i.e. the win is
//     exactly the merge win and nothing else.
//
// So the merge is 91 of the step's 464 cycles (20%) and the rebuild is 374
// (81%). Beating the merge 2x moves the shipping step 11%. If the reduction
// step is the thing that matters, the rebuild -- not the merge -- is where the
// remaining cycles are.
//
// DEVICE / RUN NOTES. Blackhole silicon only; ttsim does not implement
// SFPLOADMACRO. Run the consumer phase under `flock /tmp/tt-device.lock`. Read
// CtrlLoad and CtrlSwap FIRST: if they are not 1.0 and 2.0 the feed path is the
// limiter and nothing else on the sheet is interpretable.

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
// #ifndef, and a constexpr does not satisfy the preprocessor guard -- the
// fallback would fire and every swept variant would compile identically while
// still hashing to a distinct variant id.
// ---------------------------------------------------------------------------
#define ARM_CTRL_LOAD_ID  0
#define ARM_CTRL_SWAP_ID  1
#define ARM_XL_CALL_ID    2
#define ARM_XL_BODY_ID    3
#define ARM_MACRO_BODY_ID 4
#define ARM_MACRO_CALL_ID 5
// Context arms: the merge is only HALF of a K-reduction step. `_topk_xl_merge_`
// leaves the surviving top-K ordered but no longer bitonic, so every merge must
// be followed by `_topk_xl_rebuild_` before the next one. These three arms price
// the merge win against the step that actually ships.
#define ARM_XL_REBUILD_ID 6
#define ARM_XL_STEP_ID    7
#define ARM_MACRO_STEP_ID 8

#ifndef MERGE_ARM
#define MERGE_ARM 5
#endif

// Number of times the arm's BODY runs inside the timed region. Two values per
// arm give the slope that cancels the ~30-cycle profiler marker pair and every
// one-time cost inside the zone.
#ifndef MERGE_ITER_COUNT
#define MERGE_ITER_COUNT 32
#endif

namespace
{
// K for the merge. 512 is the smallest legal value for `_topk_xl_merge_` and
// the one perf_topk_micro_op.py's XlMerge row was taken at, so the head-to-head
// is at the same configuration.
[[maybe_unused]] constexpr std::uint32_t XL_K = 512;
[[maybe_unused]] constexpr bool XL_FUSED      = true;
[[maybe_unused]] constexpr bool TOPK_APPROX   = false;

// Fused K=512: `_topk_xl_merge_`'s `distance` = 64 * num_tiles_per_sequence
// with num_tiles_per_sequence = 1. Run B sits 64 Dst-address units past run A.
constexpr int MERGE_DISTANCE = 64;

// `_topk_xl_merge_`'s per-column MOP trip count for K=512 fused:
// row_scale_factor(=1) * 2.
constexpr std::uint32_t MERGE_N_ITERS = 2;

// LReg map for the macro merge.
//   LREG0..3 : the A rotation. These are the macroVD targets, written by the
//              SFPLOADMACRO's own load and then overwritten with the max by the
//              scheduled SFPSWAP. VD is u3 (SFPLOADMACRO.md:45), so the
//              rotation has to live in LREG0..LREG7 -- which is exactly the
//              eight registers used here, leaving nothing for constants. The
//              macro merge needs none.
//   LREG4..7 : the B holders, one per macro index. Baked into the four
//              InstructionTemplates as SFPSWAP's VC; see (c) in the header.
constexpr std::uint32_t L_A0 = ckernel::p_sfpu::LREG0;
constexpr std::uint32_t L_B0 = ckernel::p_sfpu::LREG4;

// Control arms only.
constexpr std::uint32_t L_CTRL_A = ckernel::p_sfpu::LREG0;
constexpr std::uint32_t L_CTRL_B = ckernel::p_sfpu::LREG1;

// SFPSWAP Mod1 = 9: "In all lanes, VD = max and VC = min" (SFPSWAP.md:31).
// Deliberately NOT p_sfpswap::ALL_ROWS_MAX (= 1 = SFPSWAP_MOD1_VEC_MIN_MAX),
// which is the opposite assignment -- see (a) in the file header for why the
// direction is forced rather than chosen. `p_sfpswap` defines no enum for 9
// because no shipping kernel needs the max in VD; the value is architectural,
// listed as its own `case 9:` in SFPSWAP.md's functional model rather than
// falling into the `default:` NonContractualBehavior branch.
constexpr std::uint32_t SFPSWAP_MOD1_VD_GETS_MAX = 9;

constexpr std::uint32_t SFPENCC_MOD1_EI       = 2; // SFPENCC.md:41
constexpr std::uint32_t SFPCFG_IMM16_IS_VALUE = 1; // SFPCONFIG.md:108

// --- Sequence words, one per macro index -----------------------------------
//
// Simple byte: selector 4+m -> InstructionTemplate[m] (SFPLOADMACRO.md:83-86).
//   0x80 SET   -> Insn.VB = macroVD, leaving Insn.VC at the template's LREG(4+m).
//                 MANDATORY; with it clear the macro assigns Insn.VC = macroVD
//                 and the SFPSWAP degenerates to a self-compare. See (b).
//   0x40 CLEAR -> Insn.VD = macroVD, which under Mod1 = 9 is where the max
//                 lands and is what the Store slot then writes out.
//   delay 0    -> executes on the cycle immediately after the SFPLOADMACRO,
//                 consuming the value that same macro just loaded.
constexpr std::uint32_t seq_simple(std::uint32_t m)
{
    return 0x80u | (0u << 3) | (4u + m);
}

// MAD byte: selector 2 = SFPNOP, same delay as the Simple slot. Required, not
// decorative: SFPLOADMACRO.md:11 footnote (‡) -- "If SFPSWAP is scheduled to
// the Simple sub-unit, then SFPNOP needs to be scheduled to the MAD sub-unit
// for the same time".
constexpr std::uint32_t SEQ_MAD = (0u << 3) | 2u;

// Round byte: 0 = schedule nothing. Keeping Round idle also discharges the
// other half of (‡) ("both of the Simple and Round sub-units either need to be
// idle on the next cycle or have SFPNOP scheduled for then") and sidesteps the
// (†) Simple/Round VD==16 exclusivity rule entirely.
constexpr std::uint32_t SEQ_ROUND = 0u;

// Store byte: selector 3 = the built-in SFPSTORE (no template needed).
//   0x40 and 0x80 both CLEAR -> Insn.VD = macroVD, i.e. store the register the
//        SFPSWAP just wrote the max into. 0x40 would store LReg[16], which
//        nothing here writes.
//   delay 2 -> the SFPSWAP writes on its second cycle (macro+2), so the store
//        fires at macro+3, one cycle behind its producer.
constexpr std::uint32_t SEQ_STORE = (2u << 3) | 3u;

constexpr std::uint32_t sequence_word(std::uint32_t m)
{
    return (SEQ_STORE << 24) | (SEQ_ROUND << 16) | (SEQ_MAD << 8) | seq_simple(m);
}

// Misc (SFPLOADMACRO.md:53-57): StoreMod0 [0:3], UsesLoadMod0ForStore [4:7]
// (one bit per macro), UnitDelayKind [8:11] (one bit per sub-unit).
//
//   0xF0  -> all four macros' stores inherit the LOAD's Mod0, i.e. INT32. The
//            fused [bf16 value | u16 index] word is an opaque sort key; letting
//            the store use StoreMod0 would format-convert it and destroy the
//            index in the low half. This is the same reason `_topk_xl_merge_`
//            moves fused words with InstrModLoadStore::INT32 on both sides.
//   0xB00 -> Simple (bit 8), MAD (bit 9) and Store (bit 11) on
//            WaitForElapsedInstructions. Instruction-counting, not
//            cycle-counting: the Store's delay-2 producer chain must not slide
//            if the frontend ever bubbles at a MOP chunk boundary. Round (bit
//            10) is left at 0 because nothing is ever scheduled to it.
constexpr std::uint32_t MISC_WORD_MERGE = 0xB00u | 0xF0u;

// SFPLOADMACRO field packing (ckernel_ops.h:689, SFPLOADMACRO.md:20-26,45):
//   lreg_ind      = (MacroIndex << 2) | (VD & 3)
//   dest_reg_addr = (Imm9 << 1) | (VD >> 2)
// VD is u3, so this split is exact. VD >> 2 lands in bit 0 of the 10-bit Dst
// address, and SFPLOAD.md:83 states "the low bit goes unused" -- so rotating VD
// across 0..7 does not perturb the address. Mirrors the ckernel_sfpu_mul_int.h
// idiom.
#define MERGE_LOADMACRO(macro_idx, vd, addr_mod, off) \
    TTI_SFPLOADMACRO(((macro_idx) << 2) | ((vd) & 3u), ckernel::InstrModLoadStore::INT32, (addr_mod), (off) | ((vd) >> 2))

// MOP iteration ceiling. ckernel_unpack_template::run(count) emits
// TT_MOP(0, count - 1, 0) and TT_OP_MOP's loop_count field is SEVEN bits
// (ckernel_ops.h:276), so count <= 128. The `count` parameter is a
// std::uint8_t -- wider than the field it feeds -- so passing 256 silently
// truncates to 0, the MOP runs ZERO times, and the arm reads out as a
// spectacular fake result rather than as an error.
constexpr std::uint32_t MOP_MAX_ITERS = 128;
constexpr std::uint32_t FULL_RUNS     = MERGE_ITER_COUNT / MOP_MAX_ITERS;
constexpr std::uint32_t REM_PASSES    = MERGE_ITER_COUNT % MOP_MAX_ITERS;

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
// ---------------------------------------------------------------------------
// The macro configuration. Four macros, four templates, one shared Misc word.
//
// MUST run after _llk_math_eltwise_unary_sfpu_init_once_(), which clears
// LaneConfig: the VD >= 12 backdoor that stores an instruction into
// InstructionTemplate[] instead of executing it is gated on
// LaneConfig.DISABLE_BACKDOOR_LOAD being false (SFPCONFIG.md:45-46, :120).
// ---------------------------------------------------------------------------
inline void configure_merge_macros()
{
    // InstructionTemplate[m] = SFPSWAP whose VC is the m-th B register.
    // lreg_dest = 12 + m is the backdoor index, NOT an operand: it selects
    // which template slot receives the word. The macro overrides Insn.VD with
    // macroVD at issue time, so the 12..15 written here never reaches the
    // Vector Unit as a register index.
    TTI_SFPSWAP(0, L_B0 + 0, 12, SFPSWAP_MOD1_VD_GETS_MAX);
    TTI_SFPSWAP(0, L_B0 + 1, 13, SFPSWAP_MOD1_VD_GETS_MAX);
    TTI_SFPSWAP(0, L_B0 + 2, 14, SFPSWAP_MOD1_VD_GETS_MAX);
    TTI_SFPSWAP(0, L_B0 + 3, 15, SFPSWAP_MOD1_VD_GETS_MAX);

    // Sequence[m]. The Store byte lives in bits 24..31, so these do NOT fit the
    // 16-bit immediate path -- stage the full 32-bit word through LReg[0] and
    // write with Mod1 = 0. The idiom of ckernel_sfpu_mul_int.h's
    // _init_mul_int_. LReg[0] is one of the A rotation registers, which is
    // harmless: the rotation is seeded by the loads themselves.
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

    TTI_SFPCONFIG(MISC_WORD_MERGE, 8, SFPCFG_IMM16_IS_VALUE);
    TTI_SFPNOP;
    TTI_SFPNOP;
}

// ---------------------------------------------------------------------------
// The macro merge body: 8 instructions, 8 input vectors, 4 output vectors.
//
// Address layout is byte-for-byte `_topk_xl_merge_`'s fused body:
//   run A at Dst + {0, 4, 8, 12}
//   run B at Dst + MERGE_DISTANCE + {0, 4, 8, 12}
//   merged max written back over run A
//   Dst advanced by +16 (ADDR_MOD_5) on the last instruction of the body
//
// The advance rides the last SFPLOADMACRO's own load rather than the last
// store, because a macro-scheduled SFPSTORE skips ApplyPartialAddrMod entirely
// (SFPLOADMACRO.md:139) -- it cannot advance anything. Its address was already
// resolved at SFPLOADMACRO time (:140), so putting the advance on the macro
// that produced it is safe: the store still lands at +12, not at +28.
// ---------------------------------------------------------------------------
inline void macro_merge_body()
{
    TTI_SFPLOAD(L_B0 + 0, ckernel::InstrModLoadStore::INT32, ckernel::ADDR_MOD_7, MERGE_DISTANCE + 0);
    MERGE_LOADMACRO(0u, L_A0 + 0, ckernel::ADDR_MOD_7, 0);
    TTI_SFPLOAD(L_B0 + 1, ckernel::InstrModLoadStore::INT32, ckernel::ADDR_MOD_7, MERGE_DISTANCE + 4);
    MERGE_LOADMACRO(1u, L_A0 + 1, ckernel::ADDR_MOD_7, 4);
    TTI_SFPLOAD(L_B0 + 2, ckernel::InstrModLoadStore::INT32, ckernel::ADDR_MOD_7, MERGE_DISTANCE + 8);
    MERGE_LOADMACRO(2u, L_A0 + 2, ckernel::ADDR_MOD_7, 8);
    TTI_SFPLOAD(L_B0 + 3, ckernel::InstrModLoadStore::INT32, ckernel::ADDR_MOD_7, MERGE_DISTANCE + 12);
    MERGE_LOADMACRO(3u, L_A0 + 3, ckernel::ADDR_MOD_5, 12);
}

// The 16-instruction reference body, assembled from the SHIPPING helpers so
// ARM_XL_BODY cannot drift from what `_topk_xl_merge_` actually issues.
inline void xl_merge_body()
{
    ckernel::sfpu::load16_rows_x2<MERGE_DISTANCE>();
    ckernel::sfpu::bitonic_sort_len_k(false /* ascending -> the descending branch, VC = LREG0..3 gets max */);
    ckernel::sfpu::store4_rows_top_only<16>();
}

// ---------------------------------------------------------------------------
// The macro merge as a drop-in for `_topk_xl_merge_<512, false, true>`:
// identical envelope, identical Dst window, identical column split. Only the
// body differs. Structure mirrors ckernel_sfpu_topk_xl.h:1683-1739 line for
// line so the two arms differ by exactly the thing under test.
// ---------------------------------------------------------------------------
inline void macro_merge_call(const std::uint32_t dst_index)
{
    const std::uint32_t tile_offset = dst_index << ckernel::DstTileSizeLog2[ckernel::DstTileShape::Tile32x32];

    // Recording IS iter 0 of col=0's work (Exec), so col=0 fires only
    // (n_iters - 1) more through the MOP.
    ckernel::load_replay_buf<ckernel::Exec>(0, 8, [] { macro_merge_body(); });

    ckernel::ckernel_unpack_template::run(MERGE_N_ITERS - 1);
    TTI_SETRWC(ckernel::p_setrwc::CLR_NONE, 0, 0, 0, 0, ckernel::p_setrwc::SET_D);

    // Switch the Dst write pointer to the odd column group.
    ckernel::sfpu::set_dst_write_addr_offset(tile_offset + 2);

    ckernel::ckernel_unpack_template::run(MERGE_N_ITERS);
    TTI_SETRWC(ckernel::p_setrwc::CLR_NONE, 0, 0, 0, 0, ckernel::p_setrwc::SET_D);
    ckernel::sfpu::set_dst_write_addr_offset(tile_offset + 0);

    // Retire the last two scheduled SFPSTOREs. Under
    // WaitForElapsedInstructions their counters only move when this thread
    // issues an SFPU instruction, and nothing above is one. See DRAINING in the
    // file header.
    TTI_SFPNOP;
    TTI_SFPNOP;
    TTI_SFPNOP;
}
} // namespace

void run_kernel(RUNTIME_PARAMETERS params)
{
    {
        START_PERF_MEASURE("INIT")

        // Kernel-invariant SFPU init: SFPCONFIG(0, 0xF, 1) plus the invariant
        // ADDR_MOD_7 = {srca:0, srcb:0, dest:0} that every load/store below
        // rides on. Also clears LaneConfig, the precondition for the VD >= 12
        // backdoor template writes. Must come first.
        _llk_math_eltwise_unary_sfpu_init_once_();

        // Programs ADDR_MOD_1/5/6 and the merge MOP template. Run for EVERY
        // arm, not just the XL ones: the macro body uses the same ADDR_MOD_5
        // (+16) advance, and running the identical init everywhere keeps the
        // arms differing by exactly the body under test. The macro arms
        // reprogram the MOP template below -- `_topk_xl_init_` sets it to
        // REPLAY(0, 16), and the macro body is 8 instructions long.
        ckernel::sfpu::_topk_xl_init_<XL_K, XL_FUSED>();

        // Clear stale lane predication. SFPSWAP's writes are gated on
        // LaneEnabled (SFPSWAP.md:38), so a predication mask left behind by a
        // previously-run kernel would silently suppress the compare-exchange in
        // some lanes -- a wrong answer that looks like a correct-but-different
        // one, and on a perf arm would not show up at all.
        TTI_SFPENCC(0, 0, 0, SFPENCC_MOD1_EI);

#if MERGE_ARM == ARM_MACRO_BODY_ID || MERGE_ARM == ARM_MACRO_CALL_ID || MERGE_ARM == ARM_MACRO_STEP_ID
        configure_merge_macros();
#endif

        // Establish the Dst window once. `_llk_math_eltwise_sfpu_start_` inside
        // the timed loop re-points it for the *_CALL arms, but the *_BODY and
        // control arms need a sane base before their first access.
        _llk_math_eltwise_sfpu_start_(0);

        PROFILER_SYNC();
    }

    {
        START_PERF_MEASURE("TILE_LOOP")

#if MERGE_ARM == ARM_CTRL_LOAD_ID
        {
            // CONTROL -- frontend floor. Two plain SFPLOADs recorded once and
            // replayed under one MOP issue per <= 128 passes, so the RISC-V is
            // out of the loop and the expander feeds the backend at a
            // guaranteed 1 instruction/cycle.
            //
            // Deliberately a PLAIN load and not SFPLOADMACRO: a macro issue in
            // a control arm would run against whatever LoadMacroConfig the
            // previously executed kernel left behind, which is undefined and
            // has been observed to hang the math thread non-deterministically.
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
#elif MERGE_ARM == ARM_CTRL_SWAP_ID
        {
            // CONTROL -- the tripwire. SFPSWAP is 2 backend cycles with a
            // hardware-inserted bubble that cannot be filled from this thread
            // (SFPSWAP.md:110), so this arm is backend-bound and replay buys
            // nothing. It MUST come out at ~2.0x ARM_CTRL_LOAD. It is also the
            // right control for the XL arms, whose body is an SFPSWAP lattice.
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
#elif MERGE_ARM == ARM_XL_CALL_ID
        {
#pragma GCC unroll 1
            for (std::uint32_t i = 0; i < MERGE_ITER_COUNT; ++i)
            {
                _llk_math_eltwise_sfpu_start_(0);
                ckernel::sfpu::_topk_xl_merge_<XL_K, TOPK_APPROX, XL_FUSED>(0 /* dst_index */);
                _llk_math_eltwise_sfpu_done_();
            }
        }
#elif MERGE_ARM == ARM_XL_BODY_ID
        {
            // The 16-instruction body with NO envelope: one replay recording,
            // then MOP issues only. Isolates the instruction-mix cost (predicted
            // 8*1 + 4*2 + 4*1 = 20 cycles / 8 vectors = 2.500) from the ~11
            // cycles of per-call overhead ARM_XL_CALL also carries.
            ckernel::load_replay_buf<ckernel::NoExec>(0, 16, [] { xl_merge_body(); });
            ckernel::ckernel_unpack_template::lA(lltt::replay_insn(0, 16), TT_OP_NOP).program();
            mop_run_all();
        }
#elif MERGE_ARM == ARM_MACRO_BODY_ID
        {
            // THE CANDIDATE. 8 instructions for the same 8 input vectors.
            // Predicted 1.000 cyc/vector -- the load-issue floor.
            ckernel::load_replay_buf<ckernel::NoExec>(0, 8, [] { macro_merge_body(); });
            ckernel::ckernel_unpack_template::lA(lltt::replay_insn(0, 8), TT_OP_NOP).program();
            mop_run_all();

            // Retire the trailing scheduled stores before the zone closes.
            TTI_SFPNOP;
            TTI_SFPNOP;
            TTI_SFPNOP;
        }
#elif MERGE_ARM == ARM_XL_REBUILD_ID
        {
            // CONTEXT. The other half of a K-reduction step. Normalised by
            // K/32 = 16 vectors (the K elements it rewrites), so its
            // cyc/vector is directly addable to a merge's.
#pragma GCC unroll 1
            for (std::uint32_t i = 0; i < MERGE_ITER_COUNT; ++i)
            {
                _llk_math_eltwise_sfpu_start_(0);
                ckernel::sfpu::_topk_xl_rebuild_<XL_K, TOPK_APPROX, XL_FUSED>(0 /* dst_index */, false /* ascending */);
                _llk_math_eltwise_sfpu_done_();
            }
        }
#elif MERGE_ARM == ARM_XL_STEP_ID || MERGE_ARM == ARM_MACRO_STEP_ID
        {
            // THE SHIPPING REDUCTION STEP, with and without the macro merge.
            // Normalised by 32 (the merge's input vectors) so the two rows
            // differ by exactly the merge delta and nothing else.
            //
            // The rebuild reprograms the MOP template for K=2048 and restores
            // it to REPLAY(0, 16) when done, so ARM_MACRO_STEP re-points it at
            // REPLAY(0, 8) inside `macro_merge_call` on every call rather than
            // hoisting it -- one extra push per step, already in the number.
#pragma GCC unroll 1
            for (std::uint32_t i = 0; i < MERGE_ITER_COUNT; ++i)
            {
                _llk_math_eltwise_sfpu_start_(0);
#if MERGE_ARM == ARM_MACRO_STEP_ID
                ckernel::ckernel_unpack_template::lA(lltt::replay_insn(0, 8), TT_OP_NOP).program();
                macro_merge_call(0 /* dst_index */);
#else
                ckernel::sfpu::_topk_xl_merge_<XL_K, TOPK_APPROX, XL_FUSED>(0 /* dst_index */);
#endif
                _llk_math_eltwise_sfpu_done_();

                _llk_math_eltwise_sfpu_start_(0);
                // NOTE for ARM_MACRO_STEP: the rebuild ends by restoring
                // `topk_mop_config<true>()`'s REPLAY(0, 16), and
                // `_topk_xl_init_` is not re-run per step -- which is why the
                // macro merge re-points the template at the top of every
                // iteration rather than once outside the loop.
                ckernel::sfpu::_topk_xl_rebuild_<XL_K, TOPK_APPROX, XL_FUSED>(0 /* dst_index */, false /* ascending */);
                _llk_math_eltwise_sfpu_done_();
            }
        }
#else // ARM_MACRO_CALL_ID
        {
            // THE HEAD-TO-HEAD. Same envelope as ARM_XL_CALL, same Dst window,
            // same column split; only the 8-instruction body differs from the
            // 16-instruction one.
            //
            // The MOP template is programmed once, outside the loop, because it
            // is loop-invariant -- exactly as `_topk_xl_init_` does it for
            // `_topk_xl_merge_`. It overrides the REPLAY(0, 16) that
            // `_topk_xl_init_` left behind; the macro body is 8 instructions.
            ckernel::ckernel_unpack_template::lA(lltt::replay_insn(0, 8), TT_OP_NOP).program();

#pragma GCC unroll 1
            for (std::uint32_t i = 0; i < MERGE_ITER_COUNT; ++i)
            {
                _llk_math_eltwise_sfpu_start_(0);
                macro_merge_call(0 /* dst_index */);
                _llk_math_eltwise_sfpu_done_();
            }
        }
#endif

        // MANDATORY. ZONE_SCOPED timestamps on the RISC-V at scope exit, and
        // the RISC-V runs far ahead of the SFPU backend -- most of all on the
        // MOP-fed arms, where one push leaves the entire loop in flight.
        // Without this drain the zone measures the MOP push, which is a single
        // cycle regardless of arm.
        PROFILER_SYNC();
    }
}

#endif // LLK_TRISC_MATH

// Unpack and pack do no work but MUST declare the same zones in the same order
// as math: under --enable-perf-counters the zones form a three-thread semaphore
// barrier (counters.h:545-587) that deadlocks on a mismatched set.

#ifdef LLK_TRISC_UNPACK

// Pulled in for `using namespace ckernel;` -- PROFILER_SYNC() expands to an
// unqualified tensix_sync() (profiler.h:290), which does not resolve in a TU
// that includes no LLK headers.
#include "llk_unpack_common.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
    {
        START_PERF_MEASURE("INIT")
        PROFILER_SYNC();
    }
    {
        START_PERF_MEASURE("TILE_LOOP")

#if MERGE_ARM == ARM_XL_REBUILD_ID || MERGE_ARM == ARM_XL_STEP_ID || MERGE_ARM == ARM_MACRO_STEP_ID
        // `_topk_xl_rebuild_` reaches strides > 256 through `transpose_N_faces`,
        // which shuttles half-words through SrcB (MOVD2B / TRNSPSRCB / MOVB2A /
        // MOVB2D). Those stall MATH until the unpacker marks SrcB valid, and
        // this kernel unpacks nothing -- so without one dummy valid per rebuild
        // the math thread hangs (observed: TENSIX TIMED OUT, "waited 2 seconds
        // for Math"). topk_xl_test.cpp issues the same call for the same reason
        // (sources/topk_xl_test.cpp:151).
        //
        // Only the rebuild needs it. `_topk_xl_merge_` and the macro merge are
        // pure SFPU load/swap/store and touch neither SrcA nor SrcB, which is
        // why the six merge-only arms run with an idle unpacker.
        for (std::uint32_t i = 0; i < MERGE_ITER_COUNT; ++i)
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
