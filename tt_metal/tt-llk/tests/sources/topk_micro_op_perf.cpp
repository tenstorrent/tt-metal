// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// ============================================================================
// Issue-rate baseline for the SHIPPING Top-K SFPU micro-ops  (Blackhole only)
// ============================================================================
//
// WHY THIS FILE EXISTS
// --------------------
// A candidate threshold-selection inner loop (sources/sfpu_count_above_perf.cpp)
// measures 2.000 cycles per 32-element vector on this Blackhole. Nothing in the
// shipping Top-K path has ever been measured in the same units, so there is no
// baseline to compare it against. This file produces that baseline, in the same
// units, from the same session, using the same method.
//
// It is deliberately a *sibling* of sfpu_count_above_perf.cpp and copies its
// structure exactly: MATH_ISOLATE only, three-thread zone declaration,
// two-point ITER_COUNT slope, and -- most importantly -- it carries the SAME
// two known-cost control arms in the SAME translation unit, so the controls and
// the measurements share a build, a clock and a profiler.
//
// WHAT IS MEASURED
// ----------------
//   ARM_CTRL_LOAD   control. Replay+MOP-fed stream of plain TTI_SFPLOAD.
//                   Must land at ~1.0 cyc/vector: SFPLOAD is IPC 1 and the
//                   MOP expander sustains 1 instruction/cycle, so this is the
//                   frontend floor. If it is not ~1.0, the feed path is the
//                   limiter and nothing below is interpretable.
//   ARM_CTRL_SWAP   control. Replay+MOP-fed stream of plain TTI_SFPSWAP.
//                   MUST measure ~2.0x ARM_CTRL_LOAD. SFPSWAP is documented at
//                   2 cycles with a hardware-inserted, non-fillable bubble
//                   (SFPSWAP.md:110), so its rate is known independently of
//                   anything being measured. THIS IS THE TRIPWIRE: if the ratio
//                   is not ~2.0, the measurement is invalid -- stop.
//   ARM_LOCAL_SORT  ckernel::sfpu::_bitonic_topk_phases_steps -- the per-tile
//                   bitonic behind ttnn.topk's `topk_local_sort`
//                   (ttnn .../compute/topk.cpp calls it with end_phase = 5).
//   ARM_MERGE       ckernel::sfpu::_bitonic_topk_merge.
//   ARM_GMG_TOP8    ckernel::sfpu::bitonic_top8_ph0_to_ph3 -- the 25-instruction
//                   generalized-MoE-gate single-face micro-op. This is the one
//                   the candidate has to beat.
//   ARM_GMG_TOP8_LS ARM_GMG_TOP8 wrapped in the load16/store8 pair it is
//                   actually issued with in the gate, so the micro-op can be
//                   read both bare and in situ.
//   ARM_XL_MERGE    ckernel::sfpu::_topk_xl_merge_<512, false, true> -- the
//                   topk_xl merge body (the "optionally" item).
//
// THE UNIT: CYCLES PER 32-ELEMENT VECTOR
// --------------------------------------
// An SFPU LReg is 32 lanes, and one SFPLOAD/SFPSTORE moves one 32-element
// vector. Every arm here is therefore normalised by the number of DISTINCT
// 32-element input vectors its body consumes -- not by the number of loads it
// issues (a bitonic sort revisits its data once per phase, so loads >> data).
// The per-arm figure is a compile-time constant recorded next to each arm in
// python_tests/perf_topk_micro_op.py (VECTORS_PER_BODY) and reproduced here:
//
//   ARM_CTRL_LOAD    2   (recorded body = 2 SFPLOADs)
//   ARM_CTRL_SWAP    2   (recorded body = 2 SFPSWAPs, one 32-lane pair each)
//   ARM_LOCAL_SORT  64   2 value tiles = 2048 datums; DEST addr unit 2 == one
//                        32-lane vector, so a tile is 32 vectors
//                        (ckernel_sfpu_topk.h:44 states this outright).
//   ARM_MERGE       64   same 2-value-tile window.
//   ARM_GMG_TOP8     4   LREG0..3 only; a single face's even columns =
//                        4 vectors = 128 datums.
//   ARM_GMG_TOP8_LS  4   same data, plus its load/store envelope.
//   ARM_XL_MERGE    32   4 body iters x load16_rows_x2 (8 fused value|index
//                        vectors) = 32 vectors = 1024 datums = 2 tiles.
//
// WHY GARBAGE DEST DATA IS SOUND HERE
// -----------------------------------
// Under MATH_ISOLATE nothing is unpacked, so DEST holds whatever the previous
// kernel left. That is fine and NOT a shortcut: every one of these bodies is
// data-INDEPENDENT. Loop bounds come from (m_iter, k, end_phase), the compare
// networks are fixed lattices of SFPSWAP, and SFPSWAP/SFPTRANSP have no
// data-dependent timing. There is no branch anywhere in these kernels that a
// datum can steer. Correctness of the ops is covered by test_topk.py /
// test_generalized_moe_gate.py; this file measures only issue rate.
//
// TWO-POINT SLOPE
// ---------------
// Each arm is run at two ITER_COUNT values and the rate is taken as
//   (mean@hi - mean@lo) / (hi - lo) / VECTORS_PER_BODY
// The subtraction cancels the ~30-cycle START_PERF_MEASURE marker pair
// (test_profiler_overhead.py asserts 30 +/- 5 on Blackhole) plus every
// one-time cost inside the zone: the SFPU init, the ADDR_MOD writes, and --
// for ARM_LOCAL_SORT specifically -- the FIRST call's three
// load_replay_buf<Exec> recordings, which `topk_replay_init` suppresses on
// every subsequent call. The slope is therefore the steady-state per-call
// cost, which is exactly what a tile-loop pays.
//
// PARAMETERS MUST BE #define, NEVER constexpr
// -------------------------------------------
// Every knob below is guarded with #ifndef and has a fallback. A constexpr
// emitted by the harness would NOT satisfy the preprocessor guard, the
// fallback would fire, and all seven arms would compile to the same body while
// still hashing to distinct variant ids -- a sweep that silently measures one
// thing seven times. See helpers/test_variant_parameters.py::TOPK_MICRO_OP.

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
// Parameters (all #define -- see the header note).
// ---------------------------------------------------------------------------
#ifndef TOPK_PERF_ARM
#define TOPK_PERF_ARM 4
#endif

// Number of times the arm's BODY runs inside the timed region. For the two
// control arms a "body" is one replay pass (2 vectors); for every other arm it
// is one invocation of the micro-op.
#ifndef ITER_COUNT
#define ITER_COUNT 8
#endif

// _bitonic_topk_phases_steps(idir, i_end_phase, i_start_phase, i_end_step, i_start_step).
// ttnn's topk compute kernel passes end_phase = 5; topk_test.cpp passes
// TOPK_LOGK - 1 (= 4 for K = 32). Both are swept.
#ifndef TOPK_END_PHASE
#define TOPK_END_PHASE 5
#endif

// _bitonic_topk_merge(m_iter, k).
#ifndef TOPK_M_ITER
#define TOPK_M_ITER 0
#endif

#ifndef TOPK_KVAL
#define TOPK_KVAL 32
#endif

// Sort direction. 0 = SortDir::ArgMax (descending), 1 = SortDir::ArgMin.
// Drives `idir` for local sort / the gate micro-op, and `top_min` for merge.
#ifndef TOPK_DIR
#define TOPK_DIR 0
#endif

// STABLE_SORT template argument of the topk kernels. Changes the length of the
// recorded compare lattices (e.g. ph0 replay_count 4 -> 6), so it is a real
// perf axis, not a flag.
#ifndef TOPK_STABLE
#define TOPK_STABLE 0
#endif

// Arm ids. These are #define and not constexpr because the arm dispatch below
// is #if / #elif, not `if constexpr`. That is not a style choice: `if constexpr`
// in a NON-template function does not discard its untaken branches from name
// lookup, so an `if constexpr` dispatch would require every arm's LLK header to
// be included in every variant. The three families here
// (ckernel_sfpu_topk.h, the generalized-MoE-gate single-face header, and
// ckernel_sfpu_topk_xl.h) are independent experimental trees that are never
// included together anywhere in-tree, so pulling all three into one translation
// unit would be inventing an integration this benchmark has no business
// testing. #if keeps each variant compiling exactly the headers its arm needs.
#define ARM_CTRL_LOAD_ID   0
#define ARM_CTRL_SWAP_ID   1
#define ARM_LOCAL_SORT_ID  2
#define ARM_MERGE_ID       3
#define ARM_GMG_TOP8_ID    4
#define ARM_GMG_TOP8_LS_ID 5
#define ARM_XL_MERGE_ID    6

// Experimental issue-rate variants of ARM_LOCAL_SORT. Same call, same
// parameters, same VECTORS_PER_BODY -- the ONLY difference is which of
// ckernel_sfpu_topk.h's #if-guarded knobs the header is compiled with, so a
// difference against ARM_LOCAL_SORT in the same sweep is attributable to that
// knob and nothing else. They are separate ARM IDS rather than an extra
// parameter because the knobs are preprocessor-level: the header must be
// compiled differently, which is exactly what a distinct variant id buys.
//
//   ARM_LOCAL_SORT_HOIST     TOPK_HOIST_INIT_GUARDS -- peel d == 0 so the
//                            init branches leave the phase 0/1/2 loop.
//   ARM_LOCAL_SORT_MOP       TOPK_MOP_INNER_LOOP -- drive that loop from the
//                            math MOP expander (implies the hoist).
//   ARM_LOCAL_SORT_RV_PROBE  TOPK_PROBE_RV_NOPS -- DIAGNOSTIC. Injects N
//                            RISC-V-only instructions per iteration of the
//                            phase >= 4 compare/exchange loop. Its delta
//                            against ARM_LOCAL_SORT measures directly whether
//                            that loop is RISC-V-issue-bound (delta = N x
//                            iterations) or SFPU-backend-bound (delta = 0).
//   ARM_LOCAL_SORT_REPLAY_LD TOPK_REPLAY_STEP_LOAD -- replay the phase >= 4
//                            loop's load16 (1 RISC-V issue per iteration
//                            instead of 8). This is the arm the RV_PROBE
//                            result points at.
#define ARM_LOCAL_SORT_HOIST_ID     7
#define ARM_LOCAL_SORT_MOP_ID       8
#define ARM_LOCAL_SORT_RV_PROBE_ID  9
#define ARM_LOCAL_SORT_REPLAY_LD_ID 10
//   ARM_LOCAL_SORT_REPLAY_LS TOPK_REPLAY_STEP_STORE -- the same for the loop's
//                            store16, taking that iteration from 8 RISC-V
//                            issues to 1 as well. Only pays if the loop is
//                            STILL issue-bound after the load is replayed.
#define ARM_LOCAL_SORT_REPLAY_LS_ID 11

// Injected RISC-V instructions per phase >= 4 inner-loop iteration for
// ARM_LOCAL_SORT_RV_PROBE. At TOPK_END_PHASE 5 that loop runs 48 times per
// call (phase 4: 4 iters x 4 (face,col); phase 5: 8 x 4), so 8 injects 384
// instructions -- ~8% of the arm's measured 4876 cycles/call if, and only if,
// the loop is issue-bound. Far above the run-to-run noise, which the two-point
// slope reports as ~0.
#define TOPK_PROBE_RV_NOP_COUNT 8

// True for every arm that calls _bitonic_topk_phases_steps, whatever knob it
// was built with. Keeps the init block and the timed body written once.
#define TOPK_ARM_IS_LOCAL_SORT                                                                                                   \
    (TOPK_PERF_ARM == ARM_LOCAL_SORT_ID || TOPK_PERF_ARM == ARM_LOCAL_SORT_HOIST_ID || TOPK_PERF_ARM == ARM_LOCAL_SORT_MOP_ID || \
     TOPK_PERF_ARM == ARM_LOCAL_SORT_RV_PROBE_ID || TOPK_PERF_ARM == ARM_LOCAL_SORT_REPLAY_LD_ID || TOPK_PERF_ARM == ARM_LOCAL_SORT_REPLAY_LS_ID)

namespace
{

// Explicitly `ckernel::`-qualified: the `using namespace ckernel;` that lets
// LLK headers write these bare arrives only with the LLK includes below, which
// are inside #ifdef LLK_TRISC_MATH. Unqualified names would not resolve here at
// all in the UNPACK/PACK translation units. Same rule as
// sfpu_count_above_perf.cpp:166 and sfpu_binop_scalar_perf.cpp:23.
[[maybe_unused]] constexpr std::uint32_t L_A = ckernel::p_sfpu::LREG0;
[[maybe_unused]] constexpr std::uint32_t L_B = ckernel::p_sfpu::LREG1;

// ---------------------------------------------------------------------------
// MOP iteration ceiling for the two control arms.
//
// ckernel_unpack_template::run(count) emits TT_MOP(0, count - 1, 0)
// (ckernel_template.h). In TT_OP_MOP the loop_count field is SEVEN bits
// (ckernel_ops.h:276), so count <= 128. The `count` parameter is a
// std::uint8_t -- WIDER than the field it feeds -- so passing 256 silently
// truncates to 0, the MOP runs ZERO times, and the arm reads out as a
// spectacular fake result rather than as an error. Chunk it.
// ---------------------------------------------------------------------------
constexpr std::uint32_t MOP_MAX_ITERS = 128;
constexpr std::uint32_t FULL_RUNS     = ITER_COUNT / MOP_MAX_ITERS;
constexpr std::uint32_t REM_PASSES    = ITER_COUNT % MOP_MAX_ITERS;

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

// Needed by the two _bitonic_topk_* arms, and by the gate arms for _init_topk()
// (the gate's own _init_generalized_moe_gate_topk_ is an intentional no-op, so
// the SFPU index-tracking config write has to come from here). Deliberately NOT
// pulled in alongside ckernel_sfpu_topk_xl.h: the two are separate trees that
// are never co-included in-tree.
// The experimental knobs are preprocessor-level and MUST be defined before the
// header is included; see the ARM_LOCAL_SORT_* block above.
#if TOPK_PERF_ARM == ARM_LOCAL_SORT_HOIST_ID
#define TOPK_HOIST_INIT_GUARDS 1
#elif TOPK_PERF_ARM == ARM_LOCAL_SORT_MOP_ID
#define TOPK_MOP_INNER_LOOP 1
#elif TOPK_PERF_ARM == ARM_LOCAL_SORT_RV_PROBE_ID
#define TOPK_PROBE_RV_NOPS TOPK_PROBE_RV_NOP_COUNT
#elif TOPK_PERF_ARM == ARM_LOCAL_SORT_REPLAY_LD_ID
#define TOPK_REPLAY_STEP_LOAD 1
#elif TOPK_PERF_ARM == ARM_LOCAL_SORT_REPLAY_LS_ID
#define TOPK_REPLAY_STEP_STORE 1
#endif

#if TOPK_PERF_ARM != ARM_XL_MERGE_ID
#include "sfpu/ckernel_sfpu_topk.h"
#endif

#if TOPK_PERF_ARM == ARM_GMG_TOP8_ID || TOPK_PERF_ARM == ARM_GMG_TOP8_LS_ID
#include "sfpu/experimental/ckernel_sfpu_generalized_moe_gate_topk_single_face.h"
#endif

#if TOPK_PERF_ARM == ARM_XL_MERGE_ID
#include "sfpu/experimental/ckernel_sfpu_topk_xl.h"
#endif

namespace
{
[[maybe_unused]] constexpr bool TOPK_APPROX           = false;
[[maybe_unused]] constexpr bool TOPK_PERF_STABLE_SORT = (TOPK_STABLE != 0);
[[maybe_unused]] constexpr bool TOPK_IDIR             = (TOPK_DIR != 0); // false == SortDir::ArgMax

// K for the topk_xl merge. 512 is the smallest legal value (the kernel
// static_asserts K in {512, 1024, 2048}) and the only one whose two-tile DEST
// window matches the other arms', keeping the per-vector figures comparable.
[[maybe_unused]] constexpr std::uint32_t XL_K = 512;
[[maybe_unused]] constexpr bool XL_FUSED      = true;
} // namespace

void run_kernel(RUNTIME_PARAMETERS params)
{
    {
        START_PERF_MEASURE("INIT")

        // Kernel-invariant SFPU init: SFPCONFIG(0, 0xF, 1) plus the invariant
        // ADDR_MOD_7 = {srca:0, srcb:0, dest:0} that every topk load/store
        // below rides on. Must come first.
        _llk_math_eltwise_unary_sfpu_init_once_();

#if TOPK_ARM_IS_LOCAL_SORT || TOPK_PERF_ARM == ARM_MERGE_ID
        {
            // The topk-specific half of the shipping init: re-asserts
            // ADDR_MOD_7 and adds ADDR_MOD_6 = {dest.incr = 32}, which
            // bitonic_topk_store16<..., alt_addr_mod = true> uses on its last
            // store. Also resets the RWC counters.
            _llk_math_eltwise_unary_sfpu_init_<SfpuType::topk_local_sort>();

            // Enables SFPU index-tracking mode (bit 2 of SFPU_CONTROL_REG) and
            // clears `topk_replay_init` so the first _bitonic_topk_phases_steps
            // call records its replay buffers. Every later call replays them --
            // which is precisely why the rate must come from a SLOPE, not from
            // a single point.
            ckernel::sfpu::_init_topk();
        }
#elif TOPK_PERF_ARM == ARM_GMG_TOP8_ID || TOPK_PERF_ARM == ARM_GMG_TOP8_LS_ID
        {
            // The gate's single-face helpers address DEST through ADDR_MOD_3
            // with no advance -- mirror
            // generalized_moe_gate_transpose_dest_single_face_configure_addrmod().
            // bitonic_top8_ph0_to_ph3 itself touches no ADDR_MOD (it is pure
            // SFPSWAP/SFPTRANSP on LREG0..3), so this matters only to the
            // ARM_GMG_TOP8_LS envelope; setting it for both keeps the two arms
            // differing by exactly the load/store pair.
            ckernel::addr_mod_t {
                .srca = {.incr = 0},
                .srcb = {.incr = 0},
                .dest = {.incr = 0},
            }
                .set(ckernel::ADDR_MOD_3);

            // The gate enables index tracking through the same SFPU control
            // bit the topk init writes; _init_generalized_moe_gate_topk_ is an
            // intentional no-op (it documents that reg 14 would clobber the
            // reciprocal constants), so the config write is done here.
            ckernel::sfpu::_init_topk();
        }
#elif TOPK_PERF_ARM == ARM_XL_MERGE_ID
        {
            // Programs ADDR_MOD_5/6 and the merge MOP template. _topk_xl_merge_
            // fires that template with ckernel_unpack_template::run(), so
            // skipping this would run whatever MOP the previous kernel left --
            // undefined, and the exact failure mode that hung the math thread
            // when sfpu_count_above_perf.cpp's control arm depended on residual
            // SFPU state.
            ckernel::sfpu::_topk_xl_init_<XL_K, XL_FUSED>();
        }
#endif

        // Establish the DEST window once. The per-invocation
        // _llk_math_eltwise_sfpu_start_ inside the timed loop re-points it, but
        // the very first ARM_XL_MERGE / control-arm access needs a sane base.
        _llk_math_eltwise_sfpu_start_(0);

        PROFILER_SYNC();
    }

    {
        START_PERF_MEASURE("TILE_LOOP")

#if TOPK_PERF_ARM == ARM_CTRL_LOAD_ID
        {
            // CONTROL -- frontend floor. Two plain SFPLOADs recorded once and
            // replayed under one MOP issue per <=128 passes, so the RISC-V is
            // out of the loop and the expander feeds the backend at a
            // guaranteed 1 instruction/cycle.
            //
            // Deliberately a PLAIN load and not SFPLOADMACRO: a macro issue
            // here would run against whatever LoadMacroConfig the previously
            // executed kernel left behind, which is undefined and has been
            // observed to hang the math thread non-deterministically.
            load_replay_buf<NoExec>(
                0,
                2,
                []
                {
                    TTI_SFPLOAD(L_A, ckernel::InstrModLoadStore::INT32, ckernel::ADDR_MOD_7, 0);
                    TTI_SFPLOAD(L_B, ckernel::InstrModLoadStore::INT32, ckernel::ADDR_MOD_7, 2);
                });
            ckernel_unpack_template::lA(lltt::replay_insn(0, 2), TT_OP_NOP).program();
            mop_run_all();
        }
#elif TOPK_PERF_ARM == ARM_CTRL_SWAP_ID
        {
            // CONTROL -- the tripwire. SFPSWAP is 2 backend cycles with a
            // hardware-inserted bubble that cannot be filled from this thread
            // (SFPSWAP.md:110), so this arm is backend-bound and replay buys
            // nothing. It MUST come out at ~2.0x ARM_CTRL_LOAD. It is also the
            // right control for every topk arm below, all of which are
            // SFPSWAP lattices.
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
#elif TOPK_ARM_IS_LOCAL_SORT
        {
// `unroll 1` on purpose. _bitonic_topk_phases_steps expands to thousands of
// instructions; letting the compiler unroll ITER_COUNT copies would trade the
// measured quantity for an icache-miss study.
#pragma GCC unroll 1
            for (std::uint32_t i = 0; i < ITER_COUNT; ++i)
            {
                // start_/done_ are the shipping envelope
                // (_llk_math_eltwise_unary_sfpu_params_ wraps every SFPU
                // dispatch in them). They are load-bearing here, not padding:
                // _bitonic_topk_phases_steps leaves the DEST write pointer at
                // offset 16 when it returns, so without start_ resetting it the
                // second invocation would address a different DEST window than
                // the first and the two-point slope would compare unlike work.
                _llk_math_eltwise_sfpu_start_(0);
                ckernel::sfpu::_bitonic_topk_phases_steps<TOPK_APPROX, false /* is_fp32_dest_acc_en */, TOPK_PERF_STABLE_SORT>(
                    TOPK_IDIR, TOPK_END_PHASE, 0 /* i_start_phase */, 0 /* i_end_step */, 0 /* i_start_step */);
                _llk_math_eltwise_sfpu_done_();
            }
        }
#elif TOPK_PERF_ARM == ARM_MERGE_ID
        {
#pragma GCC unroll 1
            for (std::uint32_t i = 0; i < ITER_COUNT; ++i)
            {
                _llk_math_eltwise_sfpu_start_(0);
                ckernel::sfpu::_bitonic_topk_merge<TOPK_APPROX, false /* is_fp32_dest_acc_en */, TOPK_IDIR /* top_min */, TOPK_PERF_STABLE_SORT>(
                    TOPK_M_ITER, TOPK_KVAL);
                _llk_math_eltwise_sfpu_done_();
            }
        }
#elif TOPK_PERF_ARM == ARM_GMG_TOP8_ID
        {
            // THE MICRO-OP UNDER COMPARISON. 25 instructions -- 18 SFPSWAP and
            // 7 SFPTRANSP -- operating entirely on LREG0..3. No DEST traffic,
            // no ADDR_MOD, no replay: exactly as the gate issues it.
            //
            // No MOP/replay wrapper on purpose. The rule of thumb established
            // in sfpu_count_above_perf.cpp applies: a body averaging >1 backend
            // cycle per instruction leaves the frontend slack, so replay buys
            // nothing. An all-SFPSWAP lattice at 2 cycles each is the textbook
            // case, and this is exactly what ARM_CTRL_SWAP is there to confirm.
#pragma GCC unroll 1
            for (std::uint32_t i = 0; i < ITER_COUNT; ++i)
            {
                ckernel::sfpu::bitonic_top8_ph0_to_ph3<TOPK_APPROX, false /* is_fp32_dest_acc_en */, TOPK_IDIR>();
            }
        }
#elif TOPK_PERF_ARM == ARM_GMG_TOP8_LS_ID
        {
            // The same micro-op inside the load/store envelope the gate issues
            // it with: 8 SFPLOADs in, 6 SFPSTOREs out. Difference against
            // ARM_GMG_TOP8 is the amortised cost of feeding it.
#pragma GCC unroll 1
            for (std::uint32_t i = 0; i < ITER_COUNT; ++i)
            {
                ckernel::sfpu::bitonic_topk_load16_concat_indices_single_face<false /* is_fp32_dest_acc_en */>();
                ckernel::sfpu::bitonic_top8_ph0_to_ph3<TOPK_APPROX, false /* is_fp32_dest_acc_en */, TOPK_IDIR>();
                ckernel::sfpu::bitonic_topk_store8_even_cols_split_indices_single_face<false /* is_fp32_dest_acc_en */>();
            }
        }
#else // ARM_XL_MERGE
        {
#pragma GCC unroll 1
            for (std::uint32_t i = 0; i < ITER_COUNT; ++i)
            {
                // _topk_xl_merge_ restores the DEST write pointer to
                // tile_offset + 0 itself before returning, so unlike the
                // _bitonic_topk_* arms it is already re-entrant; start_ is
                // still issued for envelope parity with them.
                _llk_math_eltwise_sfpu_start_(0);
                ckernel::sfpu::_topk_xl_merge_<XL_K, TOPK_APPROX, XL_FUSED>(0 /* dst_index */);
                _llk_math_eltwise_sfpu_done_();
            }
        }
#endif

        // MANDATORY. ZONE_SCOPED timestamps on the RISC-V at scope exit, and
        // the RISC-V runs far ahead of the SFPU backend -- most of all on the
        // control arms, where one MOP issue leaves the entire loop in flight.
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
