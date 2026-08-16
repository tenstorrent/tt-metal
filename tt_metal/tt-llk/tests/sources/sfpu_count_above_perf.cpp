// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// ============================================================================
// Replay-fed threshold-count inner loop  (Blackhole only)
// ============================================================================
//
// WHAT THIS IS
// ------------
// The candidate inner loop for threshold-based Top-K selection: count, per
// lane, how many elements exceed a running threshold. This is the "D1"
// formulation -- the only one that both (a) computes a correct count and
// (b) can reach its issue-rate floor.
//
// WHY REPLAY IS MANDATORY, NOT AN OPTIMIZATION
// --------------------------------------------
// The Tensix frontend dequeues at most ONE instruction per thread per cycle
// (PushTensixInstruction.md:19 -- `.ttinsn` fusion queues up to four per
// cycle, "but the maximum dequeue rate is only one instruction per thread per
// cycle"). A selection loop issues 2 instructions per 32-element vector, so a
// RISC-V-driven loop is frontend-saturated at exactly 2 cycles/vector with
// ZERO margin: every loop branch, address increment, or icache miss lands
// directly on the critical path.
//
// REPLAY removes the RISC-V from the loop. On a `Load=false` REPLAY the
// expander "will ingest the REPLAY instruction and emit a different
// instruction from its ReplayBuffer. For the next Count-1 cycles it will not
// ingest anything ... but it will emit one instruction per cycle"
// (REPLAY.md:46). One push buys `Count` gapless backend cycles.
//
// The full feed path, and the reason all three stages matter:
//
//   RISC-V --1 push--> MOP/REPLAY expander --1 instr/cyc, Count deep-->
//                      SFPLOADMACRO --up to 5 sub-unit instrs/cyc--> SFPU
//
// MOP/REPLAY *sustains* 1/cycle; SFPLOADMACRO *exceeds* it at the backend.
// A kernel built only on raw TTI_ issues can never sustain the first, so its
// measured rate is the RISC-V's, not the SFPU's.
//
// RULE OF THUMB derived this session: if a sequence averages >1 backend cycle
// per instruction (e.g. an SFPSWAP lattice at 2 cyc each) the frontend has
// slack and replay buys nothing. If it averages ~1 (SFPGT, SFPLOAD, SFPIADD,
// SFPSTORE are all IPC 1) the frontend IS the limiter and replay is
// load-bearing. Selection is entirely in the second class. This is why the
// shipped SFPSWAP-based LLKs never had to care.
//
// THE SELF-COLLISION HAZARD THAT SHAPES THIS KERNEL
// -------------------------------------------------
// Naively the body is (SFPLOADMACRO carrying SFPGT, then software SFPIADD).
// At 1 instr/cycle with SFPGT at delay 0, the SFPGT fires on exactly the
// cycle the software SFPIADD issues. Both are Simple sub-unit, and
// SFPLOADMACRO.md:149 is unambiguous: "If an instruction scheduled via
// SFPLOADMACRO arrives at a sub-unit on the same cycle as software issues a
// regular Vector Unit (SFPU) instruction to that sub-unit, then the scheduled
// instruction takes priority and the regular instruction is silently
// discarded."
//
// Silently. No fault, no watcher trip -- just a low count. A random
// half-above stimulus would look entirely plausible while dropping every
// accumulate. The all-above correctness case (count == N exactly) is the only
// thing that catches it.
//
// Fix: SFPGT at delay 1 (fires at t+2), and PING-PONG the macro's load
// register so the SFPGT write never lands on the cycle the next macro loads
// the same LReg. Scheduled work then occupies even cycles, software work odd.
//
//   t | Load(sched)   | Simple                    | issued by
//   - | ------------- | ------------------------- | ---------
//   0 | [A] load ->A  |                           | macro_A
//   1 |               | acc += B   (SFPIADD)      | software
//   2 | [B] load ->B  | A = (A > thr)  (SFPGT_A)  | macro_B
//   3 |               | acc += A   (SFPIADD)      | software
//   4 | [A] load ->A  | B = (B > thr)  (SFPGT_B)  | macro_A
//
// Steady state: 2 cycles per 32-element vector, no collision, exact count.
// The accumulate is one vector behind its compare -- a 1-deep software
// pipeline -- so the loop needs a prologue/epilogue (see EPILOGUE below).
//
// WHY THE Dst WALK USES ADDR_MOD AND NOT dst_reg++
// ------------------------------------------------
// The reference integer walk (`_add_int_`, ckernel_sfpu_add_int.h:41-58) uses
// `sfpi::dst_reg++`, which is RISC-V-side address arithmetic folded into the
// emitted instruction word. **Recorded instructions are immutable words**, so
// a replayed body cannot walk Dst that way. The advance must come from
// hardware: an ADDR_MOD with a non-zero `dest.incr`, applied by the SFPLOAD
// itself. This is the same reason ckernel_sfpu_mul_int.h drives its macro
// loop off ADDR_MOD_6/ADDR_MOD_7 rather than recomputing offsets.
//
// STATUS: MEASURED ON BLACKHOLE SILICON. Results (two-point slope over
// ITER_COUNT {512, 2048}, MATH_ISOLATE, 5 runs/point), after
// test_profiler_overhead.py confirmed the marker pair at 30 +/- 5 cycles:
//
//   ReplayLoad  (Load only, floor)          1.000 cyc/vector   32.0 elem/cyc
//   ReplaySwap  (SFPSWAP control)           2.000              16.0
//   CountD1     (Load+SFPGT + sw SFPIADD)   1.998              16.0
//   MacroTriple (Load+SFPGT+SFPMAD)         1.002              31.9
//   MaskStore   (Load+SFPGT+SFPSTORE)       1.003              31.9
//   MacroExp    (Load+SFPEXEXP)             1.000              32.0
//   HistNibble  (8-bucket exp histogram)    5.000               6.4
//   MultiPass   (CountD1, blind restarts)   2.097
//   PassSync    (CountD1, synced restarts)  2.389
//   HistMacro   (Load+EXEXP+MUL24+SHFT2+ST) 1.000              32.0
//   HistSum     (Load + sw SFPIADD)         2.000              16.0
//
// HistMacro/HistSum were predicted at 1.000/2.000 from the instruction count
// BEFORE the run and landed on both. Together they are the same 8-bucket
// cumulative histogram HistNibble computes, split into a materialise pass and a
// summing pass, at 3.000 cyc/vector against HistNibble's 5.000 -- i.e. 1.00
// cycles per bit of threshold resolution, against 1.67 for HistNibble and 2.00
// for a bit-serial binary search built on CountD1. The gain is entirely the
// three free sub-unit slots: HistNibble issues four SOFTWARE instructions per
// vector, HistMacro issues none.
//
// The SFPSWAP control landing on 2.000 -- predicted from SFPSWAP.md:110 alone
// -- is what makes the rest trustworthy.
//
// WHY 2.0 IS THE FLOOR FOR ANY SFPU-ONLY COUNT
// --------------------------------------------
// Not a tuning result -- an architectural bound, and worth stating once so
// nobody spends another week on it:
//
//   * The frontend dequeues at most ONE instruction per thread per cycle
//     (PushTensixInstruction.md:19).
//   * Every 32-element vector needs its own load-class instruction; an SFPLOAD
//     moves exactly 32 datums whatever the format.
//   * A macro schedules AT MOST ONE Simple instruction (SFPLOADMACRO.md:5), so
//     the compare and the accumulate cannot share a macro.
//   * A macro-scheduled result must land in macroVD -- which the next load
//     overwrites -- or in LReg[16] (SFPLOADMACRO.md:111-115). LReg[16] is
//     "only writable via SFPLOADMACRO" and "only readable via SFPLOADMACRO"
//     (:112, :120), and the only reader is a macro-scheduled SFPSTORE. It is
//     not an ALU input, so nothing macro-resident can accumulate onto itself.
//
// Therefore a count costs >= 2 instructions/vector and CountD1's 1.998 is
// optimal. What the macro CAN do at 1.0 is an arbitrary 4-deep MAP
// (Load -> Simple -> MAD -> Round -> Store, up to five instructions per cycle,
// SFPLOADMACRO.md:13), with the answer leaving through the Store slot into Dst.
// MacroTriple and MaskStore are the two measured corners of that.
//
// Consequence for Top-K: a filter/map pass is N/32 cycles and a counting pass
// is N/16. The threshold search must therefore be held to about ONE full-width
// pass, which rules out binary search (12-24 passes) and full-width histograms
// (HistNibble buys 3 bits for 2.5 passes) and leaves a per-token prior or a
// subsample estimate as the only admissible designs.
//
// RESTART COST, from MultiPass/PassSync at VECTORS_PER_SEGMENT = 64:
//   blind restart  (drain + threshold reload + Dst rewind + MOP re-issue)
//       (2.097 - 1.997) * 64 =  6.4 cycles
//   synced restart (the above + SFPSTORE of the lane partials + tensix_sync,
//                   i.e. what a DATA-DEPENDENT next threshold actually needs)
//       (2.389 - 1.997) * 64 = 25.1 cycles, and that is a LOWER bound: it
//       excludes the RISC-V's own read of the Dst window and the 32-way sum.
// At N = 32768 (1024 vectors) 25 cycles is 1.2% of a counting pass and can be
// ignored. At N = 256 (8 vectors, 16 cycles of work) it is larger than the pass
// itself, and the fixed ~92-cycle setup (this kernel's zone intercept) larger
// still -- which is why a threshold search is the wrong shape for small N and a
// sorting network is the right one.
//
// Run the consumer phase under `flock /tmp/tt-device.lock`; do NOT use
// scripts/run_safe_pytest.sh, which cd's to the tt-metal root and activates
// the wrong venv. ttsim cannot substitute: it does not implement
// SFPLOADMACRO and is functional, not cycle-accurate.
//
// FIRST RUN ORDER (do not skip):
//   1. tests/python_tests/test_profiler_overhead.py  -- confirms the marker
//      pair costs 30 +/- 5 cycles on BH. If it fails, nothing below is
//      trustworthy.
//   2. ARM_REPLAY_LOAD, two ITER_COUNTs, two-point slope. Must be ~1.0
//      cyc/vector. If it is not, the frontend is still limiting and every
//      other arm is uninterpretable.
//   3. ARM_REPLAY_SWAP. Must be ~2.0x arm 1 (SFPSWAP is documented 2 cycles
//      with a hardware-inserted, non-fillable bubble, SFPSWAP.md:110).
//   4. Only then read ARM_COUNT_D1.

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
#include "sfpu/ckernel_sfpu_load_config.h"

std::uint32_t unp_cfg_context          = 0;
std::uint32_t pack_sync_tile_dst_ptr   = 0;
std::uint32_t math_sync_tile_dst_index = 0;

// ---------------------------------------------------------------------------
// Parameters. All must be #define (not constexpr): the kernel guards each with
// #ifndef, and a constexpr does not satisfy the preprocessor guard -- the
// fallback would fire and every variant would silently compile identically.
// ---------------------------------------------------------------------------
#ifndef COUNT_ARM
#define COUNT_ARM 2
#endif

// 32-element vectors processed inside the timed region. Must be even (the
// ping-pong body covers two vectors per replay pass).
#ifndef ITER_COUNT
#define ITER_COUNT 512
#endif

// Raw 32-bit threshold bit pattern. Passed as bits, not a float, so that
// -0.0 / +-Inf / NaN thresholds are expressible exactly -- SFPGT orders by
// the sign-magnitude total order -NaN < -Inf < ... < -0 < +0 < ... < +Inf <
// +NaN (SFPGT.md:3), which disagrees with IEEE on precisely those values.
#ifndef THR_BITS
#define THR_BITS 0x3F800000 // 1.0f
#endif

namespace
{
constexpr std::uint32_t ARM_REPLAY_LOAD  = 0; // control: frontend floor, ~1.0 cyc/vec
constexpr std::uint32_t ARM_REPLAY_SWAP  = 1; // control: known 2 cyc/vec
constexpr std::uint32_t ARM_COUNT_D1     = 2; // the real selection loop
constexpr std::uint32_t ARM_MACRO_TRIPLE = 3; // 3-sub-unit ceiling probe, see below
constexpr std::uint32_t ARM_MASK_STORE   = 4; // Load+SFPGT+Store: the D2 filter, see below
constexpr std::uint32_t ARM_MACRO_EXP    = 5; // control: is a macro-scheduled SFPEXEXP free?
constexpr std::uint32_t ARM_HIST_NIBBLE  = 6; // one-pass packed exponent histogram
constexpr std::uint32_t ARM_MULTI_PASS   = 7; // per-pass restart overhead of ARM_COUNT_D1
constexpr std::uint32_t ARM_PASS_SYNC    = 8; // ...plus the RISC-V/backend rendezvous a decision needs
constexpr std::uint32_t ARM_HIST_MACRO   = 9; // 4-sub-unit histogram materialise, see below
constexpr std::uint32_t ARM_HIST_SUM     = 10; // the summing pass HistMacro must be paired with

// LReg map. A/B ping-pong as the macro load target; SFPGT overwrites the
// loaded register with its own -1/0 mask, which the SFPIADD then consumes.
//
// Explicitly `ckernel::`-qualified. The `using namespace ckernel;` that lets
// LLK headers write these bare lives in llk_math_common.h and
// llk_math_eltwise_unary_sfpu.h, which are included *below* this block and
// only under LLK_TRISC_MATH -- so unqualified names would not resolve here,
// nor in the UNPACK/PACK translation units at all. Matches the explicit
// qualification style of sfpu_binop_scalar_perf.cpp:23-24.
constexpr std::uint32_t L_A   = ckernel::p_sfpu::LREG0;
constexpr std::uint32_t L_B   = ckernel::p_sfpu::LREG1;
constexpr std::uint32_t L_ACC = ckernel::p_sfpu::LREG2;
constexpr std::uint32_t L_THR = ckernel::p_sfpu::LREG3;

// Extra registers, used only by ARM_HIST_NIBBLE. L_ONE holds the thermometer
// seed 0x11111111 and L_TMP the per-element shifted copy of it. Both are
// distinct from L_A/L_B/L_ACC and from every VD the macro can target.
constexpr std::uint32_t L_ONE = ckernel::p_sfpu::LREG4;
constexpr std::uint32_t L_TMP = ckernel::p_sfpu::LREG5;

constexpr std::uint32_t SFPGT_MOD1_SET_VD         = 8; // SFPGT.md:53
constexpr std::uint32_t SFPIADD_MOD1_ARG_LREG_DST = 0; // SFPIADD.md:48
constexpr std::uint32_t SFPIADD_MOD1_ARG_IMM      = 1; // SFPIADD.md:49
constexpr std::uint32_t SFPIADD_MOD1_CC_NONE      = 4; // SFPIADD.md:52
constexpr std::uint32_t SFPENCC_MOD1_EI           = 2; // SFPENCC.md:41
constexpr std::uint32_t SFPCFG_IMM16_IS_VALUE     = 1; // SFPCONFIG.md:108

// SFPSHFT.md:58. Not exported by sfpi_constants.h (only SFPSHFT2_MOD1_* is),
// so it is spelled out here the same way ckernel_sfpu_binary_bcast.h:146 does.
constexpr std::uint32_t SFPSHFT_MOD1_ARG_IMM = 1;

// Dst walk. SFPLOAD reads 4 consecutive Dst rows (VectorUnit.md:80), so the
// hardware advance per load is 4 rows. ADDR_MOD_7 is the invariant
// {srca:0, srcb:0, dest:0} established by the SFPU init and is used where no
// advance is wanted.
// ADDR_MOD_6 is the slot the SFPU unary path itself uses for this walk
// (llk_math_eltwise_unary_sfpu.h:52-57). ADDR_MOD_2 was the earlier choice
// and collides with the slot the A2D datacopy programs
// (llk_math_eltwise_unary_datacopy.h:362,381,400) -- benign while this
// kernel runs no datacopy, but not worth the fragility.
constexpr std::uint32_t ADDR_MOD_WALK = ckernel::ADDR_MOD_6;

// ---------------------------------------------------------------------------
// Macro 0: Load + Simple(SFPGT). No MAD, no Round, no Store.
//
// simple_bits = 0x80 | (1 << 3) | 4
//   0x80  -> Insn.VB = macroVD, so SFPGT computes (loaded > L_THR). MANDATORY:
//            with the bit clear the macro puts macroVD in VC and resolves VB
//            to the template's own VD (12), silently comparing against a
//            programmable constant instead. Same reason
//            ckernel_sfpu_binary_max_min.h sets it on its SFPSWAP byte.
//   0x40 clear -> Insn.VD = macroVD, so the mask lands in the loaded register
//            where the SFPIADD can read it. Safe only because of the
//            ping-pong (see the timing table in the header).
//   delay 1 -> fires at t+2, dodging the software SFPIADD at t+1. This is the
//            field that prevents the silent-discard hazard.
//   selector 4 -> InstructionTemplate[0] (SFPLOADMACRO.md:83).
constexpr std::uint32_t SEQ_SIMPLE = 0x80 | (1u << 3) | 4u; // 0x8C
constexpr std::uint32_t SEQ_MAD    = 0;                     // nothing scheduled
constexpr std::uint32_t SEQUENCE_0 = (SEQ_MAD << 8) | SEQ_SIMPLE;

// Misc (SFPLOADMACRO.md:53-57): StoreMod0 [0:3], UsesLoadMod0ForStore [4:7],
// UnitDelayKind [8:11]. Bit 8 = Simple = WaitForElapsedInstructions.
//
// Instruction-counting (not cycle-counting) is required here: the delay must
// track SFPU issues, not wall cycles, or a frontend bubble would slide the
// SFPGT out from under its load. Every in-tree user sets it --
// ckernel_sfpu_mul_int.h writes 0x330, ckernel_sfpu_where.h writes 0x770.
constexpr std::uint32_t MISC_WORD = 0x100;

inline void configure_macro0()
{
    // InstructionTemplate[0] via the VD>=12 backdoor: an instruction with
    // VD >= 12 is stored rather than executed, provided
    // LaneConfig.DISABLE_BACKDOOR_LOAD is false (SFPCONFIG.md:45-46, :120).
    // The SFPU init below clears LaneConfig, which establishes that -- so
    // this call MUST come after it.
    TTI_SFPGT(0, L_THR, 12, SFPGT_MOD1_SET_VD);

    TTI_SFPCONFIG(SEQUENCE_0, 4 + 0, SFPCFG_IMM16_IS_VALUE); // Sequence[0]
    TTI_SFPCONFIG(MISC_WORD, 8, SFPCFG_IMM16_IS_VALUE);      // Misc
    TTI_SFPNOP;
    TTI_SFPNOP;
}

// ---------------------------------------------------------------------------
// ARM_MACRO_TRIPLE -- the 3-sub-unit ceiling probe.
//
// QUESTION: does one SFPLOADMACRO retire Load + Simple + MAD in a single
// cycle, as SFPLOADMACRO.md:13 claims ("up to five instructions per cycle")?
//
// WHY IT MATTERS EVEN THOUGH THE MAD CANNOT ACCUMULATE. The MAD sub-unit is
// useless for a *count* -- macro-scheduled destinations are restricted to
// macroVD or the write-only LReg[16], so no reduction is expressible (this is
// why ARM_COUNT_D1 pays 2 cycles for a software SFPIADD). But the MAD slot is
// free real estate for a *map*: in a fused MoE gate the Simple slot could
// compare while the MAD slot scales, biases, or exponentiates. If the triple
// retires at ~1.0 cyc/vector, that arithmetic is free. If it retires at ~2.0,
// every fused design pays for it. This bounds the ceiling for any future
// fused kernel and is worth one arm to settle.
//
// Timing, same ping-pong discipline as ARM_COUNT_D1 (see header table):
//   Simple = SFPGT  at delay 0 -> fires t+1, writes the mask into macroVD
//   MAD    = SFPMAD at delay 1 -> fires t+2, reads that mask as its addend
// The +1 stagger is mandatory: same-delay means same cycle, and instructions
// executing in cycle T read pre-T state, so the MAD would consume the loaded
// value rather than the mask. The MAD's result goes to LReg[16] (bit 0x40) so
// it never collides with the next macro's load into macroVD.
//
// The template's operands are deliberately non-degenerate: an earlier version
// used LCONST_0 * LCONST_0 + x, which is a copy, and would have measured the
// MAD sub-unit not being exercised at all.
constexpr std::uint32_t SEQ3_SIMPLE = 0x80 | (0u << 3) | 4u; // 0x84 template[0]=SFPGT, delay 0
constexpr std::uint32_t SEQ3_MAD    = 0x40 | (1u << 3) | 5u; // 0x4D template[1]=SFPMAD, delay 1, VD=LReg16
constexpr std::uint32_t SEQUENCE_1  = (SEQ3_MAD << 8) | SEQ3_SIMPLE;

// Simple (bit 8) and MAD (bit 9) both on WaitForElapsedInstructions. Bit 9 is
// load-bearing here in a way it is not for macro 0: the MAD's delay is 1, so a
// frontend bubble under cycle-counting would slide it off its producer.
constexpr std::uint32_t MISC_WORD_3 = 0x300;

inline void configure_macro1_triple()
{
    // template[0] = SFPGT (VD=12 backdoor), same as macro 0.
    TTI_SFPGT(0, L_THR, 12, SFPGT_MOD1_SET_VD);

    // template[1] = SFPMAD (VD=13 backdoor). Operands (VA, VB, VC) -> VD.
    // The macro overrides VC to macroVD (the mask) and VD to LReg[16], leaving
    // VA=LCONST_1 and VB=L_ACC from the template: LReg16 = 1.0*L_ACC + mask.
    TTI_SFPMAD(ckernel::p_sfpu::LCONST_1, L_ACC, ckernel::p_sfpu::LCONST_0, 13, 0);

    TTI_SFPCONFIG(SEQUENCE_1, 4 + 1, SFPCFG_IMM16_IS_VALUE); // Sequence[1] -> macro index 1
    TTI_SFPCONFIG(MISC_WORD_3, 8, SFPCFG_IMM16_IS_VALUE);    // Misc
    TTI_SFPNOP;
    TTI_SFPNOP;
}

// Macro index 1 variant of the LOADMACRO packing helper.
#define LOADMACRO1(vd, addr_mod, off) TTI_SFPLOADMACRO((1u << 2) | ((vd) & 3u), ckernel::InstrModLoadStore::INT32, (addr_mod), (off) | ((vd) >> 2))

// ---------------------------------------------------------------------------
// ARM_MASK_STORE -- the "D2" filter, and the real candidate kernel.
//
// ARM_MACRO_TRIPLE shows the Simple and MAD slots are free. The Store slot is
// free too, and unlike MAD it can write somewhere that SURVIVES: Dst. So
//
//     Load + Simple(SFPGT) + Store   ->  materialise a -1/0 mask tile in place
//
// is a MAP, not a reduction, and should therefore cost ~1.0 cyc/vector -- half
// of ARM_COUNT_D1's 2.0. The count then comes from a separate reduction over
// the mask tile (FPU, packer accumulate, or a second SFPU pass), which is
// amortised over the whole tile instead of paid per vector.
//
// This is structurally the same shape as ckernel_sfpu_mul_int.h's 1-cycle case
// (Load + MAD + Store with in == out), which the LLK documents at
// "1, 2, or 3 cycles per input row" -- 1 when input and output share a Dst
// index. Here they necessarily do: SFPLOADMACRO.md:140 forces the scheduled
// store's address to equal the load's, so the mask overwrites its own input.
// For a filter pass that is exactly what is wanted.
//
// store_bits = (1 << 3) | 3
//   selector 3 -> SFPSTORE (SFPLOADMACRO.md:82)
//   0x40 CLEAR and 0x80 CLEAR -> Insn.VD = macroVD, i.e. store the register the
//        SFPGT just wrote (the mask). Setting 0x40 would store LReg[16]
//        instead, which is not what the compare produced.
//   delay 1 -> fires at t+2, one cycle after the SFPGT at t+1, so it stores the
//        mask rather than the pre-compare loaded value.
constexpr std::uint32_t SEQ4_SIMPLE = 0x80 | (0u << 3) | 4u; // SFPGT, delay 0
constexpr std::uint32_t SEQ4_STORE  = (1u << 3) | 3u;        // SFPSTORE, delay 1, VD=macroVD
constexpr std::uint32_t SEQUENCE_2  = (SEQ4_STORE << 24) | SEQ4_SIMPLE;

// Misc for macro 2: UsesLoadMod0ForStore bit for macro 2 is bit 4+2 = 0x40, so
// the store inherits the load's INT32 mode rather than StoreMod0 (the mask is
// a raw bit pattern and must not be format-converted). UnitDelayKind bit 8
// (Simple) and bit 11 (Store) = 0x900.
constexpr std::uint32_t MISC_WORD_4 = 0x900 | 0x40;

inline void configure_macro2_maskstore()
{
    TTI_SFPGT(0, L_THR, 12, SFPGT_MOD1_SET_VD); // template[0] = SFPGT

    // Sequence[2] needs the Store byte in bits 24..31, so it does NOT fit the
    // 16-bit immediate path used by the other macros. Stage the full 32-bit
    // word through LReg[0] and write with Mod1=0 -- the idiom of
    // ckernel_sfpu_mul_int.h's _init_mul_int_.
    TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_LOWER, SEQUENCE_2 & 0xFFFF);
    TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_UPPER, (SEQUENCE_2 >> 16) & 0xFFFF);
    TTI_SFPCONFIG(0, 4 + 2, 0);

    TTI_SFPCONFIG(MISC_WORD_4, 8, SFPCFG_IMM16_IS_VALUE);
    TTI_SFPNOP;
    TTI_SFPNOP;
}

#define LOADMACRO2(vd, addr_mod, off) TTI_SFPLOADMACRO((2u << 2) | ((vd) & 3u), ckernel::InstrModLoadStore::INT32, (addr_mod), (off) | ((vd) >> 2))

// ---------------------------------------------------------------------------
// Macro 3 -- Load + Simple(SFPEXEXP). Shared by ARM_MACRO_EXP and
// ARM_HIST_NIBBLE; the two differ only in the scheduled delay.
//
// WHY SFPEXEXP AT ALL. Every threshold search that is not a plain binary search
// needs a cheap value -> bucket map, and the fp32 exponent is the only bucket
// index available in one instruction (SFPEXEXP.md, Simple sub-unit, IPC 1,
// VectorUnit.md:39). SFPLOADMACRO.md:7 lists SFPEXEXP in the Simple column, so
// the question this pair of arms settles is whether that map rides the macro at
// no cost -- i.e. whether the *bucketing* is free and only the *reduction* is
// paid for.
//
// Operand plumbing (SFPLOADMACRO.md:96-115). SFPEXEXP is `LReg[VD] =
// exp(LReg[VC]) - Bias`: it has a VC and no VB. With bit 0x80 CLEAR the macro
// does `Insn.VC = macroVD` -- which is what we want, the loaded datum as the
// input -- and then `Insn.VB = Insn.VD` because the template has no VB, which
// is harmless. With bit 0x80 SET it would instead assign VB (a field SFPEXEXP
// does not have) and leave VC pointing at whatever the template encoded, i.e.
// it would silently take the exponent of a stale register. Bit 0x40 is left
// CLEAR so the result lands in macroVD, where software can read it; LReg[16]
// is writable but readable only by a macro-scheduled SFPSTORE
// (SFPLOADMACRO.md:112,120), which is no use to a software consumer.
//
// NODEBIAS (SFPEXEXP.md:42) keeps the raw 0..255 field rather than subtracting
// 127. The histogram subtracts its own base anyway, and an unbiased value is
// non-negative, which keeps the subsequent left-shift's sign unambiguous.
constexpr std::uint32_t SEQ_EXP_SIMPLE(std::uint32_t delay)
{
    return (delay << 3) | 4u; // template[0], VD=macroVD, VC=macroVD
}

// ARM_MACRO_EXP: delay 0, exactly like ARM_MACRO_TRIPLE's Simple slot, so the
// only difference from the already-measured triple arm is which instruction
// occupies the slot.
constexpr std::uint32_t SEQUENCE_3_D0 = SEQ_EXP_SIMPLE(0);

// ARM_HIST_NIBBLE: delay 4. The histogram body is 5 instructions per vector
// (one macro + four software), and *every one of the four software instructions
// is Simple or Round*. Under WaitForElapsedInstructions the scheduled SFPEXEXP
// counts down one per SFPU issue, so delay 4 lands it on the cycle of the NEXT
// macro -- the only cycle in the body on which software is not issuing to the
// Simple sub-unit. Any smaller delay puts it on top of a software Simple
// instruction and SFPLOADMACRO.md:149 silently discards the software one.
constexpr std::uint32_t SEQUENCE_3_D4 = SEQ_EXP_SIMPLE(4);

// UnitDelayKind bit 8 = Simple = WaitForElapsedInstructions. Mandatory for the
// delay-4 form (a frontend bubble under cycle-counting would slide the SFPEXEXP
// off its slot and onto a software Simple instruction).
constexpr std::uint32_t MISC_WORD_EXP = 0x100;

template <std::uint32_t sequence_word>
inline void configure_macro3_exp()
{
    // template[0] = SFPEXEXP, written through the VD>=12 backdoor. VC is the
    // placeholder the macro overwrites with macroVD.
    TTI_SFPEXEXP(0, L_A, 12, sfpi::SFPEXEXP_MOD1_NODEBIAS);

    TTI_SFPCONFIG(sequence_word, 4 + 3, SFPCFG_IMM16_IS_VALUE); // Sequence[3]
    TTI_SFPCONFIG(MISC_WORD_EXP, 8, SFPCFG_IMM16_IS_VALUE);     // Misc
    TTI_SFPNOP;
    TTI_SFPNOP;
}

#define LOADMACRO3(vd, addr_mod, off) TTI_SFPLOADMACRO((3u << 2) | ((vd) & 3u), ckernel::InstrModLoadStore::INT32, (addr_mod), (off) | ((vd) >> 2))

// ---------------------------------------------------------------------------
// ARM_HIST_NIBBLE constants.
//
// Eight cumulative buckets, one nibble each, packed into a single 32-bit lane
// accumulator. Per element: bucket b = exp(x) - EXP_BASE, and the value added
// is 0x11111111 << (4*b), which sets nibbles b..7. Summing those gives, in
// nibble j, the number of elements with bucket index <= j -- a cumulative
// histogram, which is exactly the "count above threshold" quantity a search
// wants, for eight thresholds at once.
//
// EXP_BASE is taken from the swept threshold so the eight buckets straddle it
// rather than sitting at a fixed and possibly irrelevant part of the range.
constexpr std::uint32_t EXP_BASE = (THR_BITS >> 23) & 0xFFu;

// The SFPIADD immediate that turns 4*exp into 4*(exp - EXP_BASE). Imm12 is
// sign-extended (SFPIADD.md:55), range -2048..2047; 4*255 = 1020 is the worst
// case, so every legal EXP_BASE fits.
constexpr std::int32_t EXP_BIAS_IMM = -4 * static_cast<std::int32_t>(EXP_BASE);

// Thermometer seed. One `1` in every nibble.
constexpr std::uint32_t THERMOMETER_SEED = 0x11111111u;

// ---------------------------------------------------------------------------
// ARM_MULTI_PASS -- how much does *restarting* a pass cost?
//
// Every multi-pass threshold search (binary search, prior-plus-fixup,
// subsample-then-confirm) pays a fixed cost per pass on top of the per-vector
// rate: drain the SFPU pipeline, reload the threshold register, rewind the Dst
// walk, and re-issue the MOP. At N = 32768 that cost is noise. At N = 256 --
// eight vectors, sixteen cycles of actual work -- it is the entire budget, and
// it decides whether a multi-pass search is even admissible for small tensors.
//
// The arm runs the ARM_COUNT_D1 body but chops it into segments of
// VECTORS_PER_SEGMENT, restarting between each. The segment count scales with
// ITER_COUNT, so the restart cost survives the two-point slope instead of
// cancelling in it, and
//
//     restart_cost = (slope_multipass - slope_count_d1) * VECTORS_PER_SEGMENT
//
// recovers it directly. 64 is chosen so that one segment is 32 replay passes,
// comfortably inside the 128-iteration MOP ceiling, and so that both swept
// ITER_COUNTs divide evenly.
constexpr std::uint32_t VECTORS_PER_SEGMENT = 64;

// ---------------------------------------------------------------------------
// MOP iteration ceiling.
//
// ckernel_unpack_template::run(count) emits TT_MOP(0, count - 1, 0)
// (ckernel_template.h:386). In TT_OP_MOP the loop_count field occupies bits
// 16..22 -- SEVEN bits (ckernel_ops.h:276) -- so count - 1 <= 127, i.e.
// count <= 128. The `count` parameter is a std::uint8_t, which is a wider
// type than the field it feeds: passing 256 silently truncates to 0 and the
// MOP runs zero times, which reads out as a spectacularly fast result rather
// than as an error. Chunk the run instead.
//
// One MOP issue per 128 passes is ~0.4% RISC-V involvement at ITER_COUNT=512,
// which does not meaningfully re-enter the critical path.
constexpr std::uint32_t MOP_MAX_ITERS = 128;

// Each recorded body covers two vectors (the A/B ping-pong), so a "pass" is
// two vectors for every arm.
constexpr std::uint32_t PASSES     = ITER_COUNT / 2;
constexpr std::uint32_t FULL_RUNS  = PASSES / MOP_MAX_ITERS;
constexpr std::uint32_t REM_PASSES = PASSES % MOP_MAX_ITERS;

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

// ARM_MULTI_PASS segmentation, derived once so the static_asserts sit next to
// the constants they constrain.
constexpr std::uint32_t SEGMENTS           = ITER_COUNT / VECTORS_PER_SEGMENT;
constexpr std::uint32_t PASSES_PER_SEGMENT = VECTORS_PER_SEGMENT / 2;

// ---------------------------------------------------------------------------
// ARM_HIST_MACRO -- the FOUR-sub-unit macro, and the point of this arm.
//
// WHAT IT SETTLES. The file header proves a *count* costs >= 2 instructions per
// vector and observes that what a macro CAN do at 1.0 is "an arbitrary 4-deep
// MAP (Load -> Simple -> MAD -> Round -> Store)". No arm measured that:
// MacroTriple is Load+Simple+MAD, MaskStore is Load+Simple+Store, MacroExp is
// Load+Simple. All three are 2-deep and none touches the Round sub-unit. This
// arm is the 4-deep corner, and it is not a synthetic probe -- it is exactly
// HistNibble's body with its four SOFTWARE instructions relocated into the
// macro's free MAD/Round/Store slots:
//
//   HistNibble (measured 5.000 cyc/vector, 10 instructions per 2 vectors)
//       macro:    Load + Simple(SFPEXEXP)
//       software: SFPSHFT(<<2), SFPIADD(-4*base), SFPSHFT2(thermo), SFPIADD(acc)
//
//   HistMacro (this arm, ONE instruction per vector)
//       macro:    Load
//                 + Simple(SFPEXEXP)      exp(x)
//                 + MAD   (SFPMUL24 x4)   4*exp   <- replaces the software SFPSHFT
//                 + Round (SFPSHFT2)      seed << 4*exp   <- the thermometer
//                 + Store (SFPSTORE)      thermometer -> Dst, in place
//
// The accumulate is the one thing that cannot move into the macro (the header's
// bound), so it is dropped here and paid for by a separate summing pass. The
// eight-bucket cumulative histogram therefore costs
//     1.0 (this arm, materialise)  +  2.0 (a CountD1-shaped sum over the
//                                          thermometer tile)  =  3.0 cyc/vector
// for 3 bits of threshold resolution, i.e. 1.00 cyc/bit against HistNibble's
// 1.67 and bit-serial binary search's 2.00.
//
// PREDICTION, MADE BEFORE MEASURING: 1.00 cyc/vector. One SFPLOADMACRO per
// vector, IPC 1, replay+MOP fed; every other instruction rides a free sub-unit.
//
// WHY THE ROTATION IS EIGHT DEEP, NOT TWO. The chain is four cycles long:
//
//   t+0  load          macroVD = x
//   t+1  Simple  d=0   macroVD = exp(x)                (SFPEXEXP, latency 1)
//   t+2  MAD     d=1   macroVD = (4 * macroVD) & 0x7FFFFF   (SFPMUL24)
//   t+4  Round   d=3   LReg[16] = SEED << macroVD      (SFPSHFT2, LREG mode)
//   t+5  Store   d=4   Dst[load addr] = LReg[16]
//
// The MAD sits at delay 1 and the Round at delay 3 because SFPMUL24 has a
// 2-cycle latency (VectorUnit.md integer table) -- ckernel_sfpu_mul_int.h:126,128
// spaces its own MAD and Store by exactly 2 for the same reason. So macroVD is
// live from t to t+4 and must not be reloaded in between: the rotation period P
// needs P > 4, and it also needs P not to divide 4 so that the Round's read at
// cycle t of R[(t-4) mod P] never lands on the same register the load writes at
// cycle t. P = 8 is the smallest value satisfying both that also divides the
// swept ITER_COUNTs. VD is capped at 0..7 (SFPLOADMACRO.md:45, bit 3 hard-zero),
// so all eight rotation slots are LREG0..LREG7 and the two loop constants have
// to live in the programmable-constant file (LReg11..14, LReg.md) instead.
//
// WHY (dagger) IS SATISFIED. SFPLOADMACRO.md:5 footnote: "If a Simple
// instruction and a Round instruction execute on the same cycle, then one of
// them needs to have VD == 16 and the other needs to have VD != 16." At one
// macro per cycle the Simple (t+1) and Round (t+4) sub-units BOTH fire every
// cycle in steady state, so this is not a theoretical concern -- it is the
// binding constraint on the whole design. It is met because the Round writes
// LReg[16] (bit 0x40 set) and the Simple writes macroVD. That is also why the
// Store slot is mandatory rather than decorative: LReg[16] is readable ONLY by
// a macro-scheduled SFPSTORE (LReg.md), so the Store is the only way the
// thermometer can escape.
//
// This is also the reason the ACCUMULATE cannot be folded in even in principle.
// SFPIADD is a Simple instruction, and the Simple slot is spent on SFPEXEXP;
// moving the accumulate to Round is impossible (Round hosts only SFPSHFT2 and
// SFPSTOCHRND, neither of which adds) and moving it to MAD is impossible
// (SFPMUL24 is the only integer op on MAD and it has no addend, while SFPMAD is
// floating point and the thermometer is a bit pattern). The one instruction
// that would bridge them, SFPCAST int->float, is itself a Simple instruction.
//
// WHAT THIS ARM DOES NOT MEASURE. Correctness. Under MATH_ISOLATE the stimulus
// is whatever is in Dst, and the store is in-place (SFPLOADMACRO.md:140 forces
// the scheduled store's address to equal the load's), so each pass consumes its
// own previous output. That is fine for an issue-rate number and it is also a
// real constraint on the algorithm: the histogram pass DESTROYS its input, so a
// production kernel must either run it on a scratch copy or make it the last
// use of the tile.
constexpr std::uint32_t HIST_MACRO_ROTATION = 8;

// LREG0..LREG7 are the rotation; the constants move to the programmable file.
// LReg11..LReg14 are written only through the SFPCONFIG path
// (_sfpu_load_config32_, ckernel_sfpu_load_config.h:28-35).
constexpr std::uint32_t L_SEED_C = 11; // thermometer seed 0x11111111
constexpr std::uint32_t L_FOUR_C = 12; // integer 4, the nibble stride

// Sequence[0] for the four-sub-unit macro.
//
//   Simple: 0x80 CLEAR -> Insn.VC = macroVD, which is SFPEXEXP's input operand.
//           Setting 0x80 would assign VB (a field SFPEXEXP does not have) and
//           leave VC pointing at the template's stale register -- the same trap
//           documented for macro 3 above. VD = macroVD (0x40 clear).
//   MAD:    0x80 SET   -> Insn.VB = macroVD. This one MUST be set: with it clear
//           the macro would assign Insn.VC = macroVD, and SFPMUL24 requires
//           VC == 9 (the constant zero) or hardware performs an undocumented
//           shift/add on the product (SFPMUL24.md:3, "software is strongly
//           encouraged to turn this operation into a no-op by always setting
//           VC == 9"). VA stays LREG12 (=4) from the template.
//   Round:  0x80 CLEAR -> Insn.VC = macroVD, which in SFPSHFT2_MOD1_SHFT_LREG is
//           the SHIFT AMOUNT (SFPSHFT2.md:120-133, `LReg[VD] = LReg[VB] <<
//           (LReg[VC] & 31)`), leaving VB = LREG11 (the seed) from the template.
//           0x40 SET -> VD = LReg[16], which is what satisfies the (dagger)
//           Simple/Round exclusivity rule.
//   Store:  selector 3 is the built-in SFPSTORE, no template needed. 0x40 SET ->
//           Insn.VD = 16, i.e. store LReg[16] rather than macroVD.
constexpr std::uint32_t SEQ6_SIMPLE = 0x00 | (0u << 3) | 4u; // template[0] SFPEXEXP, delay 0
constexpr std::uint32_t SEQ6_MAD    = 0x80 | (1u << 3) | 5u; // template[1] SFPMUL24, delay 1
constexpr std::uint32_t SEQ6_ROUND  = 0x40 | (3u << 3) | 6u; // template[2] SFPSHFT2, delay 3, VD=LReg16
constexpr std::uint32_t SEQ6_STORE  = 0x40 | (4u << 3) | 3u; // SFPSTORE, delay 4, source LReg16
constexpr std::uint32_t SEQUENCE_5  = (SEQ6_STORE << 24) | (SEQ6_ROUND << 16) | (SEQ6_MAD << 8) | SEQ6_SIMPLE;

// Misc (SFPLOADMACRO.md:53-57). Bit 4 = UsesLoadMod0ForStore for macro 0, so the
// store inherits the load's INT32 mode: the thermometer is a raw bit pattern and
// must not be format-converted on the way out. Bits 8..11 put all four sub-units
// on WaitForElapsedInstructions -- mandatory here because the delays span four
// cycles and a single frontend bubble under cycle-counting would slide the Round
// off its producer.
constexpr std::uint32_t MISC_WORD_6 = 0xF00 | 0x10;

inline void configure_macro0_histmacro()
{
    // template[0] = SFPEXEXP. VC is a placeholder the macro overwrites.
    TTI_SFPEXEXP(0, L_A, 12, sfpi::SFPEXEXP_MOD1_NODEBIAS);

    // template[1] = SFPMUL24, VA = LREG12 (the constant 4), VB placeholder
    // (overwritten with macroVD), VC = LCONST_0 == LReg[9] per SFPMUL24.md:3.
    // Same shape as ckernel_sfpu_mul_int.h:121.
    TTI_SFPMUL24(L_FOUR_C, 0, ckernel::p_sfpu::LCONST_0, 13, sfpi::SFPMUL24_MOD1_LOWER);

    // template[2] = SFPSHFT2 in LREG mode. The ckernel encoding puts VB in the
    // imm12 field (see ARM_HIST_NIBBLE's use above), so LREG11 is the seed and
    // lreg_src_c is the placeholder the macro overwrites with macroVD.
    TTI_SFPSHFT2(L_SEED_C, 0, 14, sfpi::SFPSHFT2_MOD1_SHFT_LREG);

    // Sequence[0] needs all four bytes, so it does not fit the 16-bit immediate
    // path -- stage it through LReg[0] exactly as configure_macro2_maskstore does.
    TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_LOWER, SEQUENCE_5 & 0xFFFF);
    TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_UPPER, (SEQUENCE_5 >> 16) & 0xFFFF);
    TTI_SFPCONFIG(0, 4 + 0, 0);

    TTI_SFPCONFIG(MISC_WORD_6, 8, SFPCFG_IMM16_IS_VALUE);
    TTI_SFPNOP;
    TTI_SFPNOP;
}

// The rotation makes a "pass" eight vectors rather than two, so this arm needs
// its own MOP chunking. Both swept ITER_COUNTs are multiples of 8, and
// 2048/8 = 256 still exceeds the 7-bit MOP loop_count ceiling, so the chunking
// is load-bearing, not decorative.
constexpr std::uint32_t HM_PASSES = ITER_COUNT / HIST_MACRO_ROTATION;
constexpr std::uint32_t HM_FULL   = HM_PASSES / MOP_MAX_ITERS;
constexpr std::uint32_t HM_REM    = HM_PASSES % MOP_MAX_ITERS;

inline void mop_run_hist_macro()
{
    for (std::uint32_t i = 0; i < HM_FULL; ++i)
    {
        ckernel::ckernel_unpack_template::run(MOP_MAX_ITERS);
    }
    if constexpr (HM_REM > 0)
    {
        ckernel::ckernel_unpack_template::run(HM_REM);
    }
}

// SFPLOADMACRO field packing (ckernel_ops.h:683, SFPLOADMACRO.md:20-26,45):
//   lreg_ind      = (MacroIndex << 2) | (VD & 3)
//   dest_reg_addr = (Imm9 << 1) | (VD >> 2)
// VD is constrained to 0..7 (bit 3 is hard-zeroed), so this split is exact.
// Mirrors the ckernel_sfpu_mul_int.h:77 idiom.
#define LOADMACRO(vd, addr_mod, off) TTI_SFPLOADMACRO((0u << 2) | ((vd) & 3u), ckernel::InstrModLoadStore::INT32, (addr_mod), (off) | ((vd) >> 2))

} // namespace

#ifdef LLK_TRISC_MATH

#include "llk_math_common.h"
#include "llk_math_eltwise_unary_sfpu.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
    {
        START_PERF_MEASURE("INIT")

        // Establishes the SFPU config register and the invariant
        // ADDR_MOD_7 = {srca:0, srcb:0, dest:0}. Also clears LaneConfig,
        // which is the precondition for the backdoor template write in
        // configure_macro0(). Do not reorder.
        _llk_math_eltwise_unary_sfpu_init_once_();

        // Hardware Dst advance for the replayed body. Recorded instruction
        // words are immutable, so the walk cannot use sfpi::dst_reg++ the way
        // _add_int_ does -- see the header.
        ckernel::addr_mod_t {
            .srca = {.incr = 0},
            .srcb = {.incr = 0},
            // 2, NOT 4. The addr_mod `dest` field is in u10 Addr units, where
            // bits [9:2] pick the 4-row group and bit 1 picks even-vs-odd
            // columns (SFPLOAD.md:86-107) -- so one SFPLOAD advances by 2.
            // Cross-checked against llk_math_eltwise_unary_sfpu.h:52-57 and
            // _add_int_'s sfpi::dst_reg++ covering 64 Dst rows in 32 loads.
            // incr=4 read only even columns and strided two tiles per tile.
            .dest = {.incr = 2},
        }
            .set(ADDR_MOD_WALK);

        // Clear stale lane predication. SFPGT's SET_VD write is itself gated
        // on LaneEnabled (SFPGT.md:27-33), so a predication mask left behind
        // by a previous kernel would silently suppress the compare in some
        // lanes -- an undercount that looks like a correct-but-slow result.
        TTI_SFPENCC(0, 0, 0, SFPENCC_MOD1_EI);

        ckernel::sfpu::_sfpu_load_imm32_(L_THR, THR_BITS);
        ckernel::sfpu::_sfpu_load_imm32_(L_ACC, 0x00000000);

        if constexpr (COUNT_ARM == ARM_COUNT_D1)
        {
            configure_macro0();
        }
        else if constexpr (COUNT_ARM == ARM_MACRO_TRIPLE)
        {
            configure_macro1_triple();
        }
        else if constexpr (COUNT_ARM == ARM_MASK_STORE)
        {
            configure_macro2_maskstore();
        }
        else if constexpr (COUNT_ARM == ARM_MACRO_EXP)
        {
            configure_macro3_exp<SEQUENCE_3_D0>();
        }
        else if constexpr (COUNT_ARM == ARM_HIST_MACRO)
        {
            // The two loop constants must sit in LReg11..14: LREG0..LREG7 are
            // all consumed by the eight-deep macroVD rotation. _sfpu_load_config32_
            // stages through LReg[0], so it MUST run before anything that wants
            // LReg[0] to hold data -- here nothing does, since the rotation is
            // seeded by the loads themselves.
            ckernel::sfpu::_sfpu_load_config32_(L_SEED_C, THERMOMETER_SEED >> 16, THERMOMETER_SEED & 0xFFFF);
            ckernel::sfpu::_sfpu_load_config32_(L_FOUR_C, 0, 4);
            configure_macro0_histmacro();
        }
        else if constexpr (COUNT_ARM == ARM_HIST_NIBBLE)
        {
            configure_macro3_exp<SEQUENCE_3_D4>();
            ckernel::sfpu::_sfpu_load_imm32_(L_ONE, THERMOMETER_SEED);
            // L_A / L_B hold the previous vector's exponent when the body
            // starts, so seed them. Any value is arithmetically legal here
            // (SFPSHFT masks the shift amount with 31, SFPSHFT.md:45, so no
            // input can fault) but leaving them undefined would make the arm's
            // first pass depend on residual state from a previous kernel.
            ckernel::sfpu::_sfpu_load_imm32_(L_A, 0);
            ckernel::sfpu::_sfpu_load_imm32_(L_B, 0);
        }

        PROFILER_SYNC();
    }

    {
        START_PERF_MEASURE("TILE_LOOP")

        static_assert(ITER_COUNT % 2 == 0, "ITER_COUNT must be even (ping-pong body covers 2 vectors)");

        if constexpr (COUNT_ARM == ARM_REPLAY_LOAD)
        {
            // CONTROL. Two plain SFPLOADs recorded, replayed via one MOP
            // issue. Establishes that the frontend can sustain 1 instr/cycle
            // -- if this does not measure ~1.0 cyc/vector, the feed path is
            // the limiter and no other arm means anything.
            //
            // DELIBERATELY *NOT* SFPLOADMACRO. An earlier revision used the
            // macro here, and it hung the math thread with TENSIX TIMED OUT:
            // configure_macro0() runs only under ARM_COUNT_D1, so this arm
            // issued macro index 0 against whatever LoadMacroConfig the
            // hardware still held from a previously-run kernel. That is
            // undefined -- a stale Sequence word can schedule arbitrary
            // instructions into sub-units with delays that never resolve.
            //
            // It failed non-deterministically, which is the dangerous part:
            // the first run passed and reported a clean 1.000 cyc/vector
            // (a leftover all-zero Sequence means "schedule nothing", which
            // degenerates to a plain load), and only hung once a different
            // test had run first and left different SFPU state behind. A
            // control arm that silently depends on residual hardware state is
            // worse than no control at all.
            //
            // A plain load is also the better control on the merits: this arm
            // exists to measure the REPLAY/MOP feed rate, so keeping macro
            // semantics out of it removes a confound.
            load_replay_buf<NoExec>(
                0,
                2,
                []
                {
                    TTI_SFPLOAD(L_A, ckernel::InstrModLoadStore::INT32, ADDR_MOD_WALK, 0);
                    TTI_SFPLOAD(L_B, ckernel::InstrModLoadStore::INT32, ADDR_MOD_WALK, 0);
                });
            ckernel_unpack_template::lA(lltt::replay_insn(0, 2), TT_OP_NOP).program();
            mop_run_all();
        }
        else if constexpr (COUNT_ARM == ARM_REPLAY_SWAP)
        {
            // CONTROL. SFPSWAP is 2 backend cycles with a hardware-inserted
            // bubble that cannot be filled from this thread (SFPSWAP.md:110),
            // so this is backend-bound and replay should buy nothing. Expect
            // ~2.0x arm 0. This is the tripwire: it is the one arm whose
            // answer is known independently of anything being measured.
            load_replay_buf<NoExec>(
                0,
                2,
                []
                {
                    TTI_SFPSWAP(0, L_A, L_B, ckernel::p_sfpswap::ALL_ROWS_MAX);
                    TTI_SFPSWAP(0, L_B, L_A, ckernel::p_sfpswap::ALL_ROWS_MAX);
                });
            ckernel_unpack_template::lA(lltt::replay_insn(0, 2), TT_OP_NOP).program();
            mop_run_all();
        }
        else if constexpr (COUNT_ARM == ARM_MASK_STORE)
        {
            // THE D2 FILTER. One macro per vector: Load + SFPGT + SFPSTORE.
            // A map, not a reduction -- so unlike ARM_COUNT_D1 there is no
            // software-issued instruction and the expected cost is ~1.0
            // cyc/vector, matching ARM_REPLAY_LOAD.
            //
            // The mask overwrites its own input (the scheduled store's address
            // is forced equal to the load's, SFPLOADMACRO.md:140). Ping-ponged
            // A/B for the same double-write reason as the other macro arms.
            load_replay_buf<NoExec>(
                0,
                2,
                []
                {
                    LOADMACRO2(L_A, ADDR_MOD_WALK, 0);
                    LOADMACRO2(L_B, ADDR_MOD_WALK, 0);
                });

            ckernel_unpack_template::lA(lltt::replay_insn(0, 2), TT_OP_NOP).program();
            mop_run_all();

            // Drain the scheduled Simple (t+1) and Store (t+2).
            TTI_SFPNOP;
            TTI_SFPNOP;
            TTI_SFPNOP;
        }
        else if constexpr (COUNT_ARM == ARM_HIST_MACRO)
        {
            // FOUR-SUB-UNIT MACRO. One SFPLOADMACRO per vector retiring
            // Load + Simple(SFPEXEXP) + MAD(SFPMUL24) + Round(SFPSHFT2) +
            // Store(SFPSTORE) -- five instructions in one issue slot, which is
            // the maximum the Vector Unit can retire per cycle
            // (SFPLOADMACRO.md:13). See the constants block for the schedule,
            // the (dagger) argument, and the prediction of 1.000 cyc/vector.
            //
            // Eight distinct macroVDs, because macroVD stays live for four
            // cycles after its load. All eight are LREG0..LREG7; the seed and
            // the nibble stride were pushed into LReg11/LReg12 in INIT.
            static_assert(ITER_COUNT % HIST_MACRO_ROTATION == 0, "ITER_COUNT must be a whole number of 8-vector rotations");
            static_assert(HM_PASSES > 0, "ITER_COUNT too small for one rotation");

            load_replay_buf<NoExec>(
                0,
                8,
                []
                {
                    LOADMACRO(ckernel::p_sfpu::LREG0, ADDR_MOD_WALK, 0);
                    LOADMACRO(ckernel::p_sfpu::LREG1, ADDR_MOD_WALK, 0);
                    LOADMACRO(ckernel::p_sfpu::LREG2, ADDR_MOD_WALK, 0);
                    LOADMACRO(ckernel::p_sfpu::LREG3, ADDR_MOD_WALK, 0);
                    LOADMACRO(ckernel::p_sfpu::LREG4, ADDR_MOD_WALK, 0);
                    LOADMACRO(ckernel::p_sfpu::LREG5, ADDR_MOD_WALK, 0);
                    LOADMACRO(ckernel::p_sfpu::LREG6, ADDR_MOD_WALK, 0);
                    LOADMACRO(ckernel::p_sfpu::LREG7, ADDR_MOD_WALK, 0);
                });

            ckernel_unpack_template::lA(lltt::replay_insn(0, 8), TT_OP_NOP).program();
            mop_run_hist_macro();

            // Drain the five-deep tail of the final macro (Store fires at t+5).
            TTI_SFPNOP;
            TTI_SFPNOP;
            TTI_SFPNOP;
            TTI_SFPNOP;
            TTI_SFPNOP;
            TTI_SFPNOP;
        }
        else if constexpr (COUNT_ARM == ARM_HIST_SUM)
        {
            // THE OTHER HALF OF HistMacro. HistMacro materialises the
            // thermometer into Dst at 1.0 cyc/vector but cannot accumulate it
            // (the file header's bound: the accumulate is SFPIADD, a Simple
            // instruction, and the Simple slot is spent on SFPEXEXP). This arm
            // is the pass that pays for the accumulate: a plain SFPLOAD of the
            // thermometer tile plus a software SFPIADD into a per-lane running
            // total, ping-ponged A/B on the same 1-deep software pipeline as
            // ARM_COUNT_D1.
            //
            // Measured here rather than inferred so that the 8-bucket
            // cumulative histogram's cost is an END-TO-END number
            // (HistMacro + HistSum) instead of one measurement plus an argument.
            //
            // Expected 2.000 cyc/vector: two instructions per vector, both
            // IPC 1, and it is exactly ARM_REPLAY_LOAD with one software
            // instruction interleaved. ARM_COUNT_D1's measured 1.997 is the
            // same shape with a macro-scheduled SFPGT riding along for free, so
            // this arm must not come out below it.
            load_replay_buf<NoExec>(
                0,
                4,
                []
                {
                    TTI_SFPLOAD(L_A, ckernel::InstrModLoadStore::INT32, ADDR_MOD_WALK, 0);
                    TTI_SFPIADD(0, L_B, L_ACC, SFPIADD_MOD1_ARG_LREG_DST | SFPIADD_MOD1_CC_NONE);
                    TTI_SFPLOAD(L_B, ckernel::InstrModLoadStore::INT32, ADDR_MOD_WALK, 0);
                    TTI_SFPIADD(0, L_A, L_ACC, SFPIADD_MOD1_ARG_LREG_DST | SFPIADD_MOD1_CC_NONE);
                });
            ckernel_unpack_template::lA(lltt::replay_insn(0, 4), TT_OP_NOP).program();
            mop_run_all();

            TTI_SFPNOP;
            TTI_SFPNOP;
            TTI_SFPNOP;
        }
        else if constexpr (COUNT_ARM == ARM_MACRO_TRIPLE)
        {
            // 3-SUB-UNIT CEILING PROBE. One SFPLOADMACRO per vector, carrying
            // both a Simple (SFPGT, delay 0) and a MAD (SFPMAD, delay 1).
            // ONE issue per vector versus ARM_COUNT_D1's two.
            //
            // Ping-ponged A/B for the same reason as ARM_COUNT_D1: the SFPGT
            // of macro N writes macroVD at t+1, and macro N+1 loads at t+1.
            // Without the ping-pong those are a same-cycle double write to one
            // LReg, which is undefined and (as this file's history shows) can
            // hang the math thread rather than fail loudly.
            //
            // Expected: ~1.0 cyc/vector if Load+Simple+MAD co-issue.
            //           ~2.0 if the MAD serialises behind the Simple.
            // Compare directly against ARM_REPLAY_LOAD (pure load, ~1.0) --
            // if this arm matches it, the Simple and MAD slots were free.
            load_replay_buf<NoExec>(
                0,
                2,
                []
                {
                    LOADMACRO1(L_A, ADDR_MOD_WALK, 0);
                    LOADMACRO1(L_B, ADDR_MOD_WALK, 0);
                });

            ckernel_unpack_template::lA(lltt::replay_insn(0, 2), TT_OP_NOP).program();
            mop_run_all();

            // Drain the scheduled Simple (t+1) and MAD (t+2) of the final
            // macro before the zone closes.
            TTI_SFPNOP;
            TTI_SFPNOP;
            TTI_SFPNOP;
        }
        else if constexpr (COUNT_ARM == ARM_MACRO_EXP)
        {
            // CONTROL for ARM_HIST_NIBBLE. One SFPLOADMACRO per vector carrying
            // a single macro-scheduled SFPEXEXP. Structurally identical to
            // ARM_MACRO_TRIPLE with the MAD slot removed, so the delta between
            // the two arms is attributable to the instruction in the slot and
            // nothing else.
            //
            // Expected ~1.0 cyc/vector, matching ARM_REPLAY_LOAD. That would
            // establish the load-bearing fact for every bucketing scheme below:
            // the value -> bucket-index map is FREE, and any histogram's whole
            // cost is the packing and the accumulate that follow it.
            //
            // Ping-ponged A/B: the scheduled SFPEXEXP of macro N writes macroVD
            // at t+1, which is the cycle macro N+1 loads. Aiming both at one
            // LReg is a same-cycle double write.
            load_replay_buf<NoExec>(
                0,
                2,
                []
                {
                    LOADMACRO3(L_A, ADDR_MOD_WALK, 0);
                    LOADMACRO3(L_B, ADDR_MOD_WALK, 0);
                });

            ckernel_unpack_template::lA(lltt::replay_insn(0, 2), TT_OP_NOP).program();
            mop_run_all();

            TTI_SFPNOP;
            TTI_SFPNOP;
            TTI_SFPNOP;
        }
        else if constexpr (COUNT_ARM == ARM_HIST_NIBBLE)
        {
            // ONE-PASS PACKED EXPONENT HISTOGRAM. Eight cumulative buckets per
            // pass instead of the one that ARM_COUNT_D1 produces.
            //
            // WHAT IS BEING PRICED. A binary search buys one bit of threshold
            // resolution per counting pass, at ARM_COUNT_D1's 2.0 cyc/vector,
            // i.e. 2.0 cyc/vector/bit. This arm buys three bits in one pass. It
            // is a win if and only if it lands below 6.0 cyc/vector; the honest
            // expectation from the instruction count is 5.0.
            //
            // The body, ping-ponged A/B, ten instructions for two vectors:
            //
            //   0 | LOADMACRO3 -> L_A     Load; schedules SFPEXEXP(L_A) delay 4
            //   1 | SFPSHFT    L_B <<= 2                      Simple
            //   2 | SFPIADD    L_B += -4*EXP_BASE             Simple
            //   3 | SFPSHFT2   L_TMP = L_ONE << L_B           Round
            //   4 | SFPIADD    L_ACC += L_TMP                 Simple
            //   5 | LOADMACRO3 -> L_B     (and the delay-4 SFPEXEXP fires here)
            //   6..9 | the same four, on L_A
            //
            // Slots 1-4 consume the exponent the OTHER half's macro produced --
            // a 1-deep software pipeline, the same shape as ARM_COUNT_D1, and
            // it is what lets the scheduled SFPEXEXP land on slot 5's cycle
            // (a Load-sub-unit cycle) instead of colliding with a software
            // Simple instruction and being discarded.
            //
            // SFPSHFT2 rather than a second SFPSHFT for the variable shift:
            // SFPSHFT's variable form is `LReg[VD] <<= LReg[VC]`, which is
            // destructive and would need L_ONE reloaded every element.
            // SFPSHFT2_MOD1_SHFT_LREG is the three-operand form
            // `LReg[VD] = LReg[VB] << LReg[VC]` (SFPSHFT2.md:120-133), so the
            // seed survives. It also sits on the Round sub-unit rather than
            // Simple, which costs nothing here (the limit is frontend issue
            // slots, not sub-unit occupancy) but keeps the Simple sub-unit
            // clear for the scheduled SFPEXEXP.
            //
            // SELF-CHECK. The body is exactly ten instructions and every one is
            // IPC 1, so a correct run MUST measure 5.000 cyc/vector. Anything
            // materially below that means an instruction was silently discarded
            // (SFPLOADMACRO.md:149) and the arm is not measuring the sequence
            // it claims to.
            load_replay_buf<NoExec>(
                0,
                10,
                []
                {
                    LOADMACRO3(L_A, ADDR_MOD_WALK, 0);
                    TTI_SFPSHFT(2, 0, L_B, SFPSHFT_MOD1_ARG_IMM);
                    TTI_SFPIADD(EXP_BIAS_IMM & 0xFFF, L_B, L_B, SFPIADD_MOD1_ARG_IMM | SFPIADD_MOD1_CC_NONE);
                    TTI_SFPSHFT2(L_ONE, L_B, L_TMP, sfpi::SFPSHFT2_MOD1_SHFT_LREG);
                    TTI_SFPIADD(0, L_TMP, L_ACC, SFPIADD_MOD1_ARG_LREG_DST | SFPIADD_MOD1_CC_NONE);

                    LOADMACRO3(L_B, ADDR_MOD_WALK, 0);
                    TTI_SFPSHFT(2, 0, L_A, SFPSHFT_MOD1_ARG_IMM);
                    TTI_SFPIADD(EXP_BIAS_IMM & 0xFFF, L_A, L_A, SFPIADD_MOD1_ARG_IMM | SFPIADD_MOD1_CC_NONE);
                    TTI_SFPSHFT2(L_ONE, L_A, L_TMP, sfpi::SFPSHFT2_MOD1_SHFT_LREG);
                    TTI_SFPIADD(0, L_TMP, L_ACC, SFPIADD_MOD1_ARG_LREG_DST | SFPIADD_MOD1_CC_NONE);
                });

            ckernel_unpack_template::lA(lltt::replay_insn(0, 10), TT_OP_NOP).program();
            mop_run_all();

            TTI_SFPNOP;
            TTI_SFPNOP;
            TTI_SFPNOP;
        }
        else if constexpr (COUNT_ARM == ARM_MULTI_PASS || COUNT_ARM == ARM_PASS_SYNC)
        {
            // PER-PASS RESTART COST. Identical inner body to ARM_COUNT_D1,
            // chopped into segments of VECTORS_PER_SEGMENT with a realistic
            // restart in between: drain the 1-deep pipeline, reload the
            // threshold register, rewind the Dst walk, re-issue the MOP.
            //
            // Read as: restart = (this arm's slope - ARM_COUNT_D1's slope)
            //                    * VECTORS_PER_SEGMENT.
            static_assert(ITER_COUNT % VECTORS_PER_SEGMENT == 0, "ITER_COUNT must be a whole number of segments");
            static_assert(PASSES_PER_SEGMENT <= MOP_MAX_ITERS, "segment exceeds the 7-bit MOP loop_count ceiling");

            load_replay_buf<NoExec>(
                0,
                4,
                []
                {
                    LOADMACRO(L_A, ADDR_MOD_WALK, 0);
                    TTI_SFPIADD(0, L_B, L_ACC, SFPIADD_MOD1_ARG_LREG_DST | SFPIADD_MOD1_CC_NONE);
                    LOADMACRO(L_B, ADDR_MOD_WALK, 0);
                    TTI_SFPIADD(0, L_A, L_ACC, SFPIADD_MOD1_ARG_LREG_DST | SFPIADD_MOD1_CC_NONE);
                });
            ckernel_unpack_template::lA(lltt::replay_insn(0, 4), TT_OP_NOP).program();

            for (std::uint32_t seg = 0; seg < SEGMENTS; ++seg)
            {
                ckernel::ckernel_unpack_template::run(PASSES_PER_SEGMENT);

                // Drain the scheduled SFPGT of the segment's last macro before
                // anything reads L_A/L_B again.
                TTI_SFPNOP;
                TTI_SFPNOP;
                TTI_SFPNOP;

                if constexpr (COUNT_ARM == ARM_PASS_SYNC)
                {
                    // ARM_MULTI_PASS restarts a pass, but it restarts it with a
                    // threshold the compiler already knew. A real search cannot:
                    // the next threshold is a function of the count this pass
                    // just produced, so the RISC-V has to (a) get the 32 lane
                    // partials out of the Vector Unit and (b) actually WAIT for
                    // them before it can pick the next threshold. That is a
                    // control dependency, and it is the thing ARM_MULTI_PASS
                    // leaves out.
                    //
                    // The SFPSTORE lands the partials in Dst. tensix_sync() is
                    // the rendezvous: the RISC-V otherwise runs far ahead of the
                    // backend (which is exactly why every timed region in this
                    // file ends with PROFILER_SYNC), so without it the "read"
                    // would race the accumulate.
                    //
                    // This is a LOWER BOUND on the real cost, deliberately. It
                    // omits the RISC-V's own read of the Dst window
                    // (Dst.md:101-116), the 32-way host-side sum, and the
                    // branch. Those need RISC_DEST_ACCESS_CTRL configuration
                    // that this kernel does not otherwise touch, and getting
                    // them wrong is a hang rather than a wrong number.
                    TTI_SFPSTORE(L_ACC, ckernel::InstrModLoadStore::INT32, ckernel::ADDR_MOD_7, 0);
                    tensix_sync();
                }

                // A real pass restart moves the threshold. Reloading the same
                // value costs the same two SFPLOADIs as reloading a new one.
                ckernel::sfpu::_sfpu_load_imm32_(L_THR, THR_BITS);

                // ...and rewinds the Dst walk to the start of the operand.
                ckernel::math::clear_dst_reg_addr();
            }
        }
        else // ARM_COUNT_D1 -- the real selection inner loop
        {
            // Recorded body covers TWO vectors, ping-ponged A/B. The SFPIADD
            // in each half consumes the mask produced by the macro of the
            // OTHER half, one vector earlier -- a 1-deep software pipeline.
            // See the timing table in the file header.
            load_replay_buf<NoExec>(
                0,
                4,
                []
                {
                    LOADMACRO(L_A, ADDR_MOD_WALK, 0);
                    TTI_SFPIADD(0, L_B, L_ACC, SFPIADD_MOD1_ARG_LREG_DST | SFPIADD_MOD1_CC_NONE);
                    LOADMACRO(L_B, ADDR_MOD_WALK, 0);
                    TTI_SFPIADD(0, L_A, L_ACC, SFPIADD_MOD1_ARG_LREG_DST | SFPIADD_MOD1_CC_NONE);
                });

            // One TTI_MOP issue drives the whole loop: the RISC-V leaves the
            // critical path entirely and the expander feeds the backend at a
            // guaranteed 1 instruction/cycle.
            ckernel_unpack_template::lA(lltt::replay_insn(0, 4), TT_OP_NOP).program();
            mop_run_all();

            // EPILOGUE. The pipeline is 1 vector deep, so the final mask has
            // been produced but not yet accumulated, and the first SFPIADD of
            // the very first pass accumulated an uninitialised L_B. Both are
            // off-by-one errors of exactly one vector. A correctness variant
            // must prologue/epilogue around this; for a throughput arm the
            // bias is 1 vector in ITER_COUNT and is stated, not hidden.
            TTI_SFPNOP;
            TTI_SFPNOP;
            TTI_SFPNOP;

            // L_ACC holds -(count) in two's complement: SFPGT's mask is -1 per
            // hit (SFPGT.md:29). A correctness variant negates once here --
            // via SFPIADD against LCONST_0 in 2's-comp mode -- so the stored
            // value is non-negative and therefore bit-identical under two's
            // complement and sign-magnitude. Do NOT rely on
            // InstrModLoadStore::INT32_2S_COMP to convert: that mode is a
            // no-op on Blackhole (ckernel_sfpu_add_int.h:28-29).
        }

        // MANDATORY. ZONE_SCOPED timestamps on the RISC-V at scope exit, and
        // the RISC-V runs far ahead of the backend -- more so here than
        // anywhere, since one MOP issue leaves the whole loop in flight.
        // Without this drain the zone measures the MOP push, which is a
        // single cycle regardless of arm.
        PROFILER_SYNC();
    }
}

#endif // LLK_TRISC_MATH

// Unpack and pack do no work but MUST declare the same zones in the same
// order as math: under --enable-perf-counters the zones form a three-thread
// semaphore barrier (counters.h:545-587) that deadlocks on a mismatched set.

#ifdef LLK_TRISC_UNPACK

// Pulled in for `using namespace ckernel;` -- PROFILER_SYNC() expands to an
// unqualified tensix_sync() (profiler.h:290), which does not resolve in a TU
// that includes no LLK headers. Same reason the MATH block includes
// llk_math_common.h. Matches sfpu_binop_scalar_perf.cpp.
#include "llk_unpack_common.h"

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
