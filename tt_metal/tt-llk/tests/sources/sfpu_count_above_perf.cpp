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
//
// The SFPSWAP control landing on 2.000 -- predicted from SFPSWAP.md:110 alone
// -- is what makes the rest trustworthy.
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

constexpr std::uint32_t SFPGT_MOD1_SET_VD         = 8; // SFPGT.md:53
constexpr std::uint32_t SFPIADD_MOD1_ARG_LREG_DST = 0; // SFPIADD.md:48
constexpr std::uint32_t SFPIADD_MOD1_CC_NONE      = 4; // SFPIADD.md:52
constexpr std::uint32_t SFPENCC_MOD1_EI           = 2; // SFPENCC.md:41
constexpr std::uint32_t SFPCFG_IMM16_IS_VALUE     = 1; // SFPCONFIG.md:108

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
