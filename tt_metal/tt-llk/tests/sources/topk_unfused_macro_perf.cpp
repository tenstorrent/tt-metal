// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// ============================================================================
//  Pricing the UNFUSED topk_xl merge / rebuild / step, macro vs shipping
//  (Blackhole only)
// ============================================================================
//
// `ckernel_sfpu_topk_xl.h` now macro-schedules the UNFUSED merge body and the
// unfused rebuild's single-level stride-64/32/16 bodies by default
// (SFPLOADMACRO: the two SFPSWAPs ride the loads' Simple slots, the macroVD
// stores ride their Store slots; opt-out `DISABLE_TOPK_XL_SFPLOADMACRO`).
// This kernel prices BOTH sides of that flag with the SAME source: the python
// driver (perf_topk_unfused_macro.py) builds each timing arm twice, once with
// the opt-out defined and once without, so the pair differs by exactly the
// header's macro bodies.
//
// PREDICTION (recorded before the first run, per the branch's discipline):
// the unfused body goes from 18 issues + 2 software SFPSWAPs (2 cycles each,
// plus the index-tracking stall) ~= 20-22 cycles to 16 single-cycle issues.
//   merge:   every body is macro'd            -> ~1.15-1.35x per call
//   rebuild: only the stride-64/32/16 bodies  -> ~1.05-1.15x per call
//            (the stride-8 sort_16_alt phases and the transposes are
//             untouched — multi-level lattices have no load to ride)
//   step:    in between, weighted by the rebuild's dominance
//
// READ CtrlLoad AND CtrlSwap FIRST. If CtrlSwap is not ~2.00x CtrlLoad the
// run is INVALID (both are documented per-instruction constants; a broken
// ratio means the harness measured the RISC-V push rate, not the SFPU).
// A misconfigured macro degenerates into a plain SFPLOAD and measures the
// SAME issue rate — timing CANNOT distinguish "free" from "silently not
// executed". Correctness is established ONLY by
// tests/python_tests/test_topk_xl_unfused_macro.py (chained num_chunks=4
// golden + opt-out differential + schedule-nothing mutation control).
//
// Blackhole silicon only. Every rebuild/step iteration needs one
// `_llk_unpack_set_srcb_dummy_valid_()` from UNPACK (the rebuild's Dst
// transposes shuttle half-words through SrcB) or MATH hangs on SrcB valid.

#include <cstdint>

#include "ckernel.h"
#include "ckernel_addrmod.h"
#include "ckernel_defs.h"
#include "ckernel_instr_params.h"
#include "ckernel_ops.h"
#include "ckernel_template.h"
#include "counters.h" // START_PERF_MEASURE
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
// #ifndef and a constexpr does not satisfy a preprocessor guard.
//
// UNF_ARM:
//   0  CtrlLoad   plain-load issue-rate floor (per-instruction constant)
//   1  CtrlSwap   the tripwire — MUST measure ~2.00x CtrlLoad
//   2  Merge      _topk_xl_merge_<K, false, false>  per iteration
//   3  Rebuild    _topk_xl_rebuild_<K, false, false>(0, false) per iteration
//   4  Step       merge + rebuild per iteration (the unit the op actually runs)
//
// The macro-vs-shipping pairing is NOT an arm: the driver compiles each arm
// twice, toggling DISABLE_TOPK_XL_SFPLOADMACRO in the build header.
// ---------------------------------------------------------------------------
#define UNF_ARM_CTRL_LOAD_ID 0
#define UNF_ARM_CTRL_SWAP_ID 1
#define UNF_ARM_MERGE_ID     2
#define UNF_ARM_REBUILD_ID   3
#define UNF_ARM_STEP_ID      4

#ifndef UNF_ARM
#define UNF_ARM 2
#endif

#ifndef UNF_ITER_COUNT
#define UNF_ITER_COUNT 16
#endif

#ifndef UNF_K
#define UNF_K 1024
#endif

#if UNF_ARM == UNF_ARM_REBUILD_ID || UNF_ARM == UNF_ARM_STEP_ID
#define UNF_NEEDS_SRCB_VALID 1
#else
#define UNF_NEEDS_SRCB_VALID 0
#endif

namespace
{
[[maybe_unused]] constexpr std::uint32_t XL_K = UNF_K;
[[maybe_unused]] constexpr bool TOPK_APPROX   = false;
[[maybe_unused]] constexpr bool XL_DIR        = false; // the op always rebuilds descending

// Control arms only.
[[maybe_unused]] constexpr std::uint32_t L_CTRL_A = ckernel::p_sfpu::LREG0;
[[maybe_unused]] constexpr std::uint32_t L_CTRL_B = ckernel::p_sfpu::LREG1;

[[maybe_unused]] constexpr std::uint32_t SFPENCC_MOD1_EI = 2; // SFPENCC.md:41

// MOP iteration ceiling: TT_OP_MOP's loop_count field is SEVEN bits; 256
// silently truncates to 0 and the arm reads out as a spectacular fake result.
constexpr std::uint32_t MOP_MAX_ITERS = 128;
constexpr std::uint32_t FULL_RUNS     = UNF_ITER_COUNT / MOP_MAX_ITERS;
constexpr std::uint32_t REM_PASSES    = UNF_ITER_COUNT % MOP_MAX_ITERS;

[[maybe_unused]] inline void mop_run_all()
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

using namespace ckernel;

void run_kernel(RUNTIME_PARAMETERS params)
{
    {
        START_PERF_MEASURE("INIT")

        // Kernel-invariant SFPU init: SFPCONFIG(0, 0xF, 1) + ADDR_MOD_7. Also
        // clears LaneConfig, the precondition for the VD >= 12 backdoor
        // template writes inside `_topk_xl_init_<K, false>` and the per-call
        // installs in merge / rebuild. Must come first.
        _llk_math_eltwise_unary_sfpu_init_once_();

        // The UNFUSED init: ADDR_MOD_1..7, index-tracking mode (LaneConfig
        // bit [2]), the merge MOP template, and — unless
        // DISABLE_TOPK_XL_SFPLOADMACRO — the macro Sequence/Misc words. Run
        // for EVERY arm so the paired variants differ by exactly the bodies
        // the flag swaps.
        ckernel::sfpu::_topk_xl_init_<XL_K, false /* fused */>();

        // Clear stale lane predication: SFPSWAP's writes are gated on
        // LaneEnabled, so a mask left behind by a previously-run kernel would
        // silently suppress work.
        TTI_SFPENCC(0, 0, 0, SFPENCC_MOD1_EI);

        _llk_math_eltwise_sfpu_start_(0);

        PROFILER_SYNC();
    }

    {
        START_PERF_MEASURE("TILE_LOOP")

#if UNF_ARM == UNF_ARM_CTRL_LOAD_ID
        {
            // CONTROL — frontend floor. Deliberately a PLAIN load.
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
#elif UNF_ARM == UNF_ARM_CTRL_SWAP_ID
        {
            // CONTROL — the tripwire. MUST come out at ~2.0x CtrlLoad.
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
#elif UNF_ARM == UNF_ARM_MERGE_ID
        {
#pragma GCC unroll 1
            for (std::uint32_t i = 0; i < UNF_ITER_COUNT; ++i)
            {
                _llk_math_eltwise_sfpu_start_(0);
                ckernel::sfpu::_topk_xl_merge_<XL_K, TOPK_APPROX, false /* fused */>(0 /* dst_index */);
                _llk_math_eltwise_sfpu_done_();
            }
        }
#elif UNF_ARM == UNF_ARM_REBUILD_ID
        {
#pragma GCC unroll 1
            for (std::uint32_t i = 0; i < UNF_ITER_COUNT; ++i)
            {
                _llk_math_eltwise_sfpu_start_(0);
                ckernel::sfpu::_topk_xl_rebuild_<XL_K, TOPK_APPROX, false /* fused */>(0 /* dst_index */, XL_DIR);
                _llk_math_eltwise_sfpu_done_();
            }
        }
#elif UNF_ARM == UNF_ARM_STEP_ID
        {
#pragma GCC unroll 1
            for (std::uint32_t i = 0; i < UNF_ITER_COUNT; ++i)
            {
                _llk_math_eltwise_sfpu_start_(0);
                ckernel::sfpu::_topk_xl_merge_<XL_K, TOPK_APPROX, false /* fused */>(0 /* dst_index */);
                ckernel::sfpu::_topk_xl_rebuild_<XL_K, TOPK_APPROX, false /* fused */>(0 /* dst_index */, XL_DIR);
                _llk_math_eltwise_sfpu_done_();
            }
        }
#else
#error "UNF_ARM must be 0..4"
#endif

        PROFILER_SYNC();
    }
}

#endif // LLK_TRISC_MATH

// Unpack and pack do no work but MUST declare the same zones in the same order
// as math: under --enable-perf-counters the zones form a three-thread
// semaphore barrier that deadlocks on a mismatched set.

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

#if UNF_NEEDS_SRCB_VALID
        // The rebuild's Dst transposes shuttle half-words through SrcB
        // (MOVD2B / TRNSPSRCB / MOVB2A / MOVB2D); those stall MATH until the
        // unpacker marks SrcB valid, and this kernel unpacks nothing.
        for (std::uint32_t i = 0; i < UNF_ITER_COUNT; ++i)
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
