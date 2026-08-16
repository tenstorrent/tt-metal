// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// ============================================================================
// SFPLOADMACRO issue-rate microbenchmark  (Blackhole only)
// ============================================================================
//
// QUESTION THIS ANSWERS
// ---------------------
// Does SFPLOADMACRO actually retire one load + one Simple + one MAD
// instruction per cycle, as BlackholeA0/.../SFPLOADMACRO.md:13 claims?
//
// This is the fork in the road for threshold-based Top-K selection. A
// selection inner loop is
//
//     SFPLOAD  (load sub-unit)  ->  SFPGT (simple sub-unit)  ->  SFPMAD (MAD sub-unit)
//
// i.e. three instructions on three DIFFERENT sub-units. If SFPLOADMACRO
// schedules them concurrently, selection runs at ~1 cycle per 32-element
// vector, which is exactly the unpacker bound (64 B/cycle) -- the kernel would
// sit on the roofline. If it serialises, selection costs ~3 cycles/vector and
// the achievable floor for N=32k moves from ~1,024 to ~3,072 cycles.
//
// Note the contrast with the bitonic sort kernels: SFPSWAP and SFPTRANSP are
// BOTH simple-sub-unit instructions (SFPLOADMACRO.md:7), so a bitonic inner
// loop cannot benefit from multi-issue at all. Selection is the one shape in
// this domain where the 5-IPC path is even theoretically reachable.
//
// PRIOR ART -- READ THIS BEFORE TRUSTING ANY NUMBER BELOW
// -------------------------------------------------------
// SFPLOADMACRO is already used in this tree, and part of this question is
// already answered:
//
//   * ckernel_sfpu_mul_int.h:36-38 states the achieved throughput outright:
//     "1, 2, or 3 cycles per input row" -- 1 cycle when the two inputs and the
//     output are the same Dst index. Its cycle-by-cycle sub-unit tables at
//     :40-66 are the best documentation of macro scheduling in the tree.
//     That case is Load + MAD + Store.
//   * ckernel_sfpu_where.h:108-147 schedules the SIMPLE sub-unit from a macro
//     (SFPSETCC / SFPENCC templates), so Simple-in-macro is exercised.
//   * eltwise_unary_typecast_perf.cpp + python_tests/perf_eltwise_typecast.py
//     already A/B macro vs plain loop by toggling DISABLE_SFPLOADMACRO. That
//     is the established way to measure this, and is cheaper than this file.
//
// So multi-issue demonstrably works on silicon. What is NOT covered by the
// above is the exact combination selection needs: Load + Simple(SFPGT) + MAD,
// with no store. mul_int's 1-cycle case retires Load + MAD + Store; where.h
// uses Simple but is not a throughput benchmark. This file exists only to
// close that specific gap. If you only want a yes/no on "does multi-issue
// work", read mul_int and skip this.
//
// STATUS: WRITTEN, NEVER EXECUTED. No Blackhole silicon was available on the
// machine where this was authored. Every predicted number below is a
// prediction, not a measurement. Arm 1 exists specifically so that a wrong
// prediction is self-diagnosing -- see "CONTROL" below.
//
// ARMS AND PREDICTIONS  (cycles for ITER_COUNT vectors, MATH_ISOLATE)
// -------------------------------------------------------------------
//   0 LOAD_ONLY      ~1.0 * N   Issue-rate floor. One SFPU instruction per
//                               cycle is the documented maximum absent
//                               SFPLOADMACRO (VectorUnit.md:112).
//   1 SWAP_ONLY      ~2.0 * N   CONTROL. SFPSWAP is documented at 2 cycles
//                               with a hardware-inserted bubble
//                               (SFPSWAP.md:110). If this arm does not measure
//                               ~2x arm 0, THE HARNESS IS NOT MEASURING WHAT
//                               WE THINK IT IS -- stop and fix that before
//                               reading arms 2 and 3.
//   2 SERIAL_TRIPLE  ~3.0 * N   The cost of the selection loop WITHOUT
//                               SFPLOADMACRO. This is the number to beat.
//   3 MACRO_TRIPLE   ~1.0 * N   The hypothesis. Same logical work as arm 2.
//
// DECISION RULE
// -------------
//   arm3 / arm0 ~= 1.0  -> GO.   Multi-issue works. Selection is unpack-bound;
//                                proceed to a real threshold-selection kernel.
//   arm3 ~= arm2        -> NO-GO. Macro serialises. Selection costs 3
//                                cycles/vector; re-derive the floor before
//                                committing to any selection design.
//   anything else       -> the encoding below is wrong. Suspect the Sequence
//                                word first (see SEQUENCE ENCODING).
//
// WHY MATH_ISOLATE: this is a pure instruction-issue question. Unpack and pack
// traffic would only add noise, and MATH_ISOLATE removes the software
// unpack->math handshake.
//
// NOT RUNNABLE UNDER ttsim: the simulator does not implement SFPLOADMACRO
// (hence TT_METAL_DISABLE_SFPLOADMACRO), and it is functional rather than
// cycle-accurate, so it could not answer a timing question even if it did.
// Silicon only.

#include <cstdint>

#include "ckernel.h"
#include "ckernel_addrmod.h"
#include "ckernel_defs.h"
#include "ckernel_instr_params.h"
#include "ckernel_ops.h"
#include "counters.h" // START_PERF_MEASURE (counters.h:616)
#include "llk_defs.h"
#include "params.h"
#include "perf.h"
#include "profiler.h"
#include "sfpu/ckernel_sfpu_load_config.h"

// Globals expected by the LLK test harness.
std::uint32_t unp_cfg_context          = 0;
std::uint32_t pack_sync_tile_dst_ptr   = 0;
std::uint32_t math_sync_tile_dst_index = 0;

// ---------------------------------------------------------------------------
// Benchmark parameters (override from the driver via params.h / -D)
// ---------------------------------------------------------------------------
#ifndef LOADMACRO_ARM
#define LOADMACRO_ARM 3
#endif

// Number of SFPU vector operations in the timed region. Large enough that
// loop overhead and the START_PERF_MEASURE bracket are negligible; small
// enough to stay well inside a 32-bit cycle counter.
#ifndef ITER_COUNT
#define ITER_COUNT 512
#endif

// Unrolling factor. The RISC-V math thread must sustain >= 1 Tensix
// instruction push per cycle or IT becomes the bottleneck rather than the
// SFPU, which would mask the very effect being measured. 8 is the minimum
// defensible value; raise it if arm 0 fails to reach ~1.0 cycles/vector.
//
// CAVEAT ON INSTRUCTION GATHERING -- this harness and production metal differ,
// so numbers here are NOT directly transferable to a metal kernel:
//   * tt-metal disables 4-way .ttinsn gathering on Blackhole
//     (firmware_common.h:275-276, 323 -- workaround for tt-metal#16439), so a
//     production kernel pushes at most 1 Tensix instruction per cycle.
//   * This LLK harness never configures it. tests/helpers/include/boot.h's
//     device_setup() does no CSR 0x7c0 write, and ENABLE_GATHERING appears
//     nowhere under tests/. So the harness runs at the hardware reset default,
//     which metal's workaround implies is ENABLED.
// If arm 0 measures materially below 1.0 cycles/vector, gathering is active
// and the RISC-V push rate is not the floor you think it is. Report the arm-0
// number alongside any conclusion so the reader can tell which regime the
// measurement was taken in.
#ifndef UNROLL
#define UNROLL 8
#endif

// `#pragma GCC unroll N` does not macro-expand its argument, so route it
// through _Pragma with the standard double-expansion trick. Getting this wrong
// silently leaves the loop rolled, which would put the RISC-V branch in the
// critical path and mask the effect under measurement.
#define LM_DO_PRAGMA(x)     _Pragma(#x)
#define LM_UNROLL_PRAGMA(n) LM_DO_PRAGMA(GCC unroll n)

namespace
{
constexpr std::uint32_t ARM_LOAD_ONLY     = 0;
constexpr std::uint32_t ARM_SWAP_ONLY     = 1;
constexpr std::uint32_t ARM_SERIAL_TRIPLE = 2;
constexpr std::uint32_t ARM_MACRO_TRIPLE  = 3;

// LReg allocation.
//   LREG0     - destination of the load; the value under test
//   LREG1     - comparison partner / swap partner
//   LREG2     - MAD operand
//   LREG16    - MAD result sink (writable only via SFPLOADMACRO), keeps the
//               accumulate off the critical registers
constexpr std::uint32_t L_VAL = 0;
constexpr std::uint32_t L_THR = 1;
constexpr std::uint32_t L_ACC = 2;

// SFPGT Mod1 (SFPGT.md:50-53). SET_VD writes the comparison result to VD as
// -1 / 0 (two's complement), which is the mask-as-value form. We deliberately
// do NOT set SET_CC: touching the condition codes would drag in lane-enable
// state and the SFPENCC save/restore that this benchmark exists to avoid.
constexpr std::uint32_t SFPGT_MOD1_SET_VD = 8;

// SFPCONFIG Mod1 (SFPCONFIG.md:108-112).
constexpr std::uint32_t SFPCFG_IMM16_IS_VALUE = 1;

// SFPCONFIG VD selector (SFPCONFIG.md:39-101).
constexpr std::uint32_t SFPCFG_VD_SEQUENCE0 = 4; // -> LoadMacroConfig.Sequence[0]
constexpr std::uint32_t SFPCFG_VD_MISC      = 8; // -> LoadMacroConfig.Misc

// ---------------------------------------------------------------------------
// SEQUENCE ENCODING  (SFPLOADMACRO.md:66-133)
// ---------------------------------------------------------------------------
// Sequence[MacroIndex] packs one byte per sub-unit, in this order:
//     bits  0..7  Simple
//     bits  8..15 MAD
//     bits 16..23 Round
//     bits 24..31 Store
//
// Within each byte:
//     bits 0..2  what to schedule: 0 = nothing, 2 = SFPNOP, 3 = SFPSTORE,
//                4..7 = InstructionTemplate[0..3]
//     bits 3..5  delay
//     bit  6     (0x40) force Insn.VD = LReg[16]
//     bit  7     (0x80) Insn.VB = VD  (else Insn.VC = VD)
//
// Simple byte = 0x84:
//     0x04 -> take InstructionTemplate[0] (our SFPGT)
//     0x80 -> Insn.VB = VD, i.e. the just-loaded register becomes the LHS of
//             the comparison. SFPGT computes LReg[VB] > LReg[VC]; VC stays as
//             encoded in the template (L_THR), so this is value > threshold.
//     delay 0 -> executes the cycle immediately after SFPLOADMACRO.
//
// MAD byte = 0x45:
//     0x05 -> take InstructionTemplate[1] (our SFPMAD)
//     0x40 -> Insn.VD = LReg[16]. Without this the macro would force the MAD's
//             result onto the loaded register, clobbering the value the Simple
//             sub-unit is reading in the same window. LReg[16] exists exactly
//             for this (SFPLOADMACRO.md:112).
//     delay 0.
//
// Round byte = 0x00, Store byte = 0x00 -> schedule nothing on those sub-units
// (case 0 in the functional model). Leaving Store unscheduled matters: an
// SFPSTORE here would add L1/Dst write traffic and turn an issue-rate question
// into a bandwidth question.
//
// The whole word therefore fits in 16 bits, which is what lets us write it
// with MOD1_IMM16_IS_VALUE instead of staging through LReg[0].
constexpr std::uint32_t SEQ_SIMPLE = 0x84;
constexpr std::uint32_t SEQ_MAD    = 0x45;
constexpr std::uint32_t SEQUENCE_0 = (SEQ_MAD << 8) | SEQ_SIMPLE; // 0x4584

// Misc (SFPLOADMACRO.md:53-57): StoreMod0 bits 0..3, UsesLoadMod0ForStore bits
// 4..7, UnitDelayKind bits 8..11.
//
// UnitDelayKind = 1 (WaitForElapsedInstructions) for the two sub-units we
// schedule. This matches every in-tree user -- ckernel_sfpu_mul_int.h writes
// 0x330 and ckernel_sfpu_where.h writes 0x770 -- and it is the correct kind
// here: with a cycle-based delay the scheduled instructions would count down
// on cycles the thread is not issuing, so a stall anywhere upstream would
// desynchronise the macro from its own load. Instruction-based delay ties the
// countdown to SFPU issues, which is what the timing tables in mul_int assume.
//
// bits 8..11 = 0b0011 -> Simple and MAD use WaitForElapsedInstructions.
// No store is scheduled, so StoreMod0 and UsesLoadMod0ForStore stay 0.
constexpr std::uint32_t MISC_WORD = 0x300;

// ---------------------------------------------------------------------------
// Configure LoadMacroConfig for macro index 0.
// ---------------------------------------------------------------------------
inline void configure_macro0()
{
    // InstructionTemplate[i] is written by executing an instruction with
    // VD = 12 + i, which stores the instruction WORD rather than executing it.
    // This requires LaneConfig.DISABLE_BACKDOOR_LOAD == false, which is the
    // reset default (SFPCONFIG.md:45-46, :120). It is cheaper and far less
    // error-prone than staging a 32-bit word through LReg[0] + SFPCONFIG.
    //
    // The macro overrides VD and one of VB/VC at issue time, so only the
    // opcode, Mod1, and the non-overridden operand carry over from the
    // template.

    // Template[0] (VD = 12): SFPGT, compare against L_THR, write mask to VD.
    TTI_SFPGT(0, L_THR, 12, SFPGT_MOD1_SET_VD);

    // Template[1] (VD = 13): SFPMAD. Operands are (VA, VB, VC) -> VD.
    // VD is redirected to LReg[16] by the sequence byte, and VB is left as
    // encoded; the macro rewrites VC to the loaded register.
    TTI_SFPMAD(L_ACC, L_ACC, L_ACC, 13, 0);

    // Sequence[0] and Misc.
    TTI_SFPCONFIG(SEQUENCE_0, SFPCFG_VD_SEQUENCE0, SFPCFG_IMM16_IS_VALUE);
    TTI_SFPCONFIG(MISC_WORD, SFPCFG_VD_MISC, SFPCFG_IMM16_IS_VALUE);

    // SFPCONFIG has an ordering hazard against the next SFPU instruction when
    // it changes DISABLE_BACKDOOR_LOAD (SFPCONFIG.md:139). We do not change
    // that bit, but the templates were just written via the backdoor path, so
    // pad before the timed region begins. Cost is outside the measurement.
    TTI_SFPNOP;
    TTI_SFPNOP;
}
} // namespace

#ifdef LLK_TRISC_MATH

#include "llk_math_common.h"
#include "llk_math_eltwise_unary_sfpu.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
    {
        START_PERF_MEASURE("INIT")

        // Kernel-invariant SFPU init: writes the SFPU config register
        // (SFPCONFIG(0, 0xF, 1)) and establishes the invariant
        // ADDR_MOD_7 = {srca:0, srcb:0, dest:0}. This is the init the tt-llk
        // standalone SFPU harness uses, and it is what makes ADDR_MOD_7 safe
        // to reference below.
        //
        // ORDERING MATTERS: this clears LaneConfig, so DISABLE_BACKDOOR_LOAD
        // is false afterwards -- which is the precondition for writing
        // InstructionTemplate via the VD >= 12 backdoor in configure_macro0().
        // Do not reorder these two.
        _llk_math_eltwise_unary_sfpu_init_once_();

        // Seed the registers. Values are irrelevant to issue rate; they exist
        // so the instructions are well-formed and so a future correctness
        // variant of this file has somewhere to start.
        ckernel::sfpu::_sfpu_load_imm32_(L_VAL, 0x3F800000); //  1.0f
        ckernel::sfpu::_sfpu_load_imm32_(L_THR, 0x3F000000); //  0.5f
        ckernel::sfpu::_sfpu_load_imm32_(L_ACC, 0x00000000); //  0

        if constexpr (LOADMACRO_ARM == ARM_MACRO_TRIPLE)
        {
            configure_macro0();
        }

        PROFILER_SYNC();
    }

    {
        START_PERF_MEASURE("TILE_LOOP")

        // Every arm issues exactly ITER_COUNT vector operations. Divide by
        // UNROLL for the trip count; ITER_COUNT must be a multiple of UNROLL.
        static_assert(ITER_COUNT % UNROLL == 0, "ITER_COUNT must be a multiple of UNROLL");
        constexpr std::uint32_t TRIPS = ITER_COUNT / UNROLL;

        for (std::uint32_t trip = 0; trip < TRIPS; ++trip)
        {
            LM_UNROLL_PRAGMA(UNROLL)
            for (std::uint32_t u = 0; u < UNROLL; ++u)
            {
                if constexpr (LOADMACRO_ARM == ARM_LOAD_ONLY)
                {
                    // Issue-rate floor: one load per cycle, nothing scheduled.
                    TTI_SFPLOAD(L_VAL, 0, ADDR_MOD_7, 0);
                }
                else if constexpr (LOADMACRO_ARM == ARM_SWAP_ONLY)
                {
                    // CONTROL. Documented 2 cycles each, hardware-inserted
                    // bubble, and the bubble is NOT fillable from this thread
                    // (SFPSWAP.md:110). Expect ~2x arm 0.
                    TTI_SFPSWAP(0, L_VAL, L_THR, ckernel::p_sfpswap::ALL_ROWS_MAX);
                }
                else if constexpr (LOADMACRO_ARM == ARM_SERIAL_TRIPLE)
                {
                    // The selection inner loop, issued the ordinary way: three
                    // separate pushes to three sub-units, one per cycle.
                    TTI_SFPLOAD(L_VAL, 0, ADDR_MOD_7, 0);
                    TTI_SFPGT(0, L_THR, L_VAL, SFPGT_MOD1_SET_VD);
                    TTI_SFPMAD(L_ACC, L_ACC, L_VAL, L_ACC, 0);
                }
                else // ARM_MACRO_TRIPLE
                {
                    // The same three operations behind one push. MacroIndex 0
                    // and VD = L_VAL (=0), so the lreg_ind field is
                    // (0 << 2) + 0 = 0 (SFPLOADMACRO.md:20-26, :45).
                    TTI_SFPLOADMACRO(0, 0, ADDR_MOD_7, 0);
                }
            }
        }

        // The macro leaves up to four instructions scheduled for future
        // cycles. Drain them before the timing bracket closes, or the last
        // few land outside the measured window and bias the result low.
        if constexpr (LOADMACRO_ARM == ARM_MACRO_TRIPLE)
        {
            TTI_SFPNOP;
            TTI_SFPNOP;
            TTI_SFPNOP;
            TTI_SFPNOP;
        }

        // MANDATORY. ZONE_SCOPED timestamps in its destructor, on the RISC-V,
        // at scope exit -- and the RISC-V math thread runs far ahead of the
        // Tensix backend (the instruction FIFO alone is 28 deep). Without this
        // drain the zone measures how fast instructions were PUSHED, not how
        // fast the SFPU RETIRED them, which is ~1 cycle/op for every arm
        // including SWAP_ONLY. That is a false positive that would silently
        // confirm the hypothesis this benchmark exists to test.
        //
        // PROFILER_SYNC() is tensix_sync() (profiler.h:290). Every perf kernel
        // in tests/sources closes every zone with it -- see
        // eltwise_unary_typecast_perf.cpp:92,125,172,312 and
        // sfpu_binop_scalar_perf.cpp:49,77,224.
        PROFILER_SYNC();
    }
}

#endif // LLK_TRISC_MATH

// The unpack and pack threads do no work, but they MUST declare the same zones
// in the same order as the math thread. Under the performance-counter build
// (--enable-perf-counters), the zone constructor/destructor form a three-thread
// pc_buf semaphore barrier (counters.h:545-587, PERF_NUM_SPINWAITERS = 2): the
// arming thread posts twice and the other two spin-wait. If the threads declare
// different zone sets, the barrier never balances and the test hangs with
// TENSIX TIMED OUT against a semaphore initialised with max 1 (boot.h:123-127).
//
// Declaring only TILE_LOOP here while math declares INIT + TILE_LOOP is exactly
// that bug. Every other perf source declares INIT + TILE_LOOP on all three
// threads. The default (no-counters) build has no barrier and would not hang,
// which is what makes this a landmine rather than an obvious failure.

#ifdef LLK_TRISC_UNPACK
void run_kernel(RUNTIME_PARAMETERS params)
{
    // MATH_ISOLATE: the math thread must not stall on an unpack handshake.
    // Nothing is unpacked -- the timed region reads only LRegs and one fixed
    // Dst address, never SrcA/SrcB, so none of the usual
    // _perf_unpack_loop_set_valid dvalid mocking is needed.
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
void run_kernel(RUNTIME_PARAMETERS params)
{
    // Nothing is packed. A store in the timed region would convert an
    // issue-rate measurement into a bandwidth measurement.
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
