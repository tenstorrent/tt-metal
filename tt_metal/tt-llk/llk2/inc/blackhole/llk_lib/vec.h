// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// llk-lib, the SFPU functions (vec_silu, vec_tanh, vec_exp, ...) on Blackhole.
//
// The rule for the vector unit: llk-lib is written in HAL calls, or in SFPI for the body of a vector function.
// Everything around a function is HAL calls (tt_llk_blackhole/llk_lib/hal/): the SFPU setup, the address modifiers,
// the Dest offset of each tile, the row counter steps between faces and the stalls. The fast exp is HAL calls
// throughout (llk_lib/vec/fast_exp.h: vec_fast_exp_init, vec_fast_exp_tiles). Every other function's arithmetic is
// SFPI code in LLK 2.0's own files, llk_lib/vec/<function>.h (llk_lib/vec/common.h has their shape):
// vec_<function>_init and vec_<function>_iterations, which this file picks by the Plan's function. Nothing here includes or calls the 1.0
// SFPU library (ckernel::sfpu). Ported from llk_math_eltwise_unary_sfpu.h, llk_math_eltwise_sfpu_common.h and
// llk_math_eltwise_unary_sfpu_params.h; the function bodies from ckernel_sfpu_*.h, instruction for instruction.
//
//   vec_params                one struct; vec_configure and vec_execute take the same one
//   vec_params_from(Plan)     fills it from the Resolver's Plan
//   vec_configure<P>()        waits for math and SFPU to drain, sets the SFPU's Dest mode, resets the lane config,
//                             programs the address modifier SFPI loads and stores use, clears the row counters,
//                             then runs the function's SFPI init (its constants and lookup tables)
//   vec_execute<P>(dest_row, tiles, scalars...)
//                             the function on every tile of the tensor at Dest row dest_row, in place. tiles is their
//                             count: a ct or, for a size known only at run time, an rt (resolver/work_counts.h). The
//                             scalars (relu_max's max, ...) are FP32 bit patterns, passed on to the SFPI function
//   vec<On, Plan>(counts, dest_row, scalars...)
//                             the per-thread call the Compute API makes: configure, then execute, only when this
//                             build is for thread On, the math or the pack thread. counts (WorkCounts) is the run-time
//                             side of the Plan's Work
//
// How one tile is done. The SFPU has 32 lanes. An SFPI load reads 32 datums from Dest: the even or the odd columns
// of four rows, at the Dest row counter plus the Dest offset. One SFPI iteration loads, computes and stores those
// 32 datums, then moves the row counter on; 8 iterations cover one 16 x 16 face. So for each tile: point the Dest
// offset at the tile, and for each of the 4 faces call the SFPI
// function once (8 iterations) and move the row counter's CR to the next face. After each tile, the row counter goes
// back to 0.
//
// Not every call walks the tile that way. vec_walk picks one of three walks, by the form's body and the region:
//   Faces      the above: the 4 faces, one call of 8 iterations each (1.0's VectorMode::RC). VecRegion::All
//   Tile       the fast exp: one call of 32 iterations per tile, the SFPI code stepping the row counter through all 4
//              faces itself (1.0's VectorMode::None with ITERATIONS = 32). VecRegion::All
//   LeftFaces  faces 0 and 2 only, the two that hold tile columns 0..15, skipping faces 1 and 3 (1.0's
//              VectorMode::C). VecRegion::LeftFaces: one call of 8 iterations on each; VecRegion::LeftFacesEvenColumns:
//              exp's half-face body, 4 iterations on the even columns of each.
//
// Which thread drives the SFPU. Either the math or the pack thread can issue SFPU instructions; each has its own
// Dest offset register, row counters and address modifiers, so configure and execute are the same on both. The
// hand-over around the call is the owners' (llk_lib/sync.h), not the op's: the Compute API takes the Dest tensor over
// for the SFPU on the driving thread before vec runs (a STALLWAIT for the FPU on math; on pack, math drains the FPU and
// posts math_pack_sync, and pack waits with every instruction class held, SFPU and Config included), and the op that
// reads the result takes it over from the SFPU (pack: a STALLWAIT on the pack thread until the SFPU is idle).
// Either way the Dest offset is the tensor's address (the absolute Dest row) plus the tile's rows: no section base.
//
// What an init programs. vec_configure writes what every function needs: the Dest mode, the lane control, the
// address modifier that moves nothing (addr_mod_hold) and the row counters. A function's init adds only its own:
// constants, lookup tables, the stepping address modifier (vec_addr_mod_step) and LOADMACRO programs.
//
// Not HAL: the functions' bodies are SFPI code (llk_lib/thread.h, "What llk-lib is written in"): sfpi:: C++, and for
// the bodies written instruction by instruction (the exact exps' replayed bodies, the reciprocals, exp's
// polynomial form, the ttsim build's plain fast exp) the SFPI compiler's instruction builtins TTI_SFP* with
// the operand names p_sfpu and InstrModLoadStore. The fast exp is HAL calls throughout (platform/sfpu.h).
//
// SFPLOADMACRO. With DISABLE_SFPLOADMACRO (TT_METAL_DISABLE_SFPLOADMACRO=1, the ttsim build; vec_has_loadmacro,
// llk_lib/vec/common.h) the exp and reciprocal
// SFPI code takes its plain SFPLOAD / SFPSTORE loops; without it (silicon), the fast exp runs 32 LOADMACROs a tile
// from the replay buffer, the bf16 / fp32 exact reciprocals and the fast reciprocal run LOADMACRO sequences, and the fp32
// exact exp runs its replayed TTI body. Same results by design; only silicon runs the macro forms. The exact bf16
// exp (exp_21f, replayed TTI) and exp's half-face bodies take one form on both.

#pragma once

#include <cstdint>

#include "hal/cfg.h"
#include "hal/replay.h"
#include "hal/sync.h"
#include "hal/tensix_counters.h"
#include "llk_lib/math_common.h" // counter_step, set_addr_mod
#include "llk_lib/platform/issue.h"
#include "llk_lib/platform/sfpu.h"
#include "llk_lib/thread.h"
#include "resolver/plan.h"
#include "resolver/vec_function.h"
#include "resolver/work_counts.h"

// The functions: each file has its init and its iterations (llk_lib/vec/common.h).
#include "llk_lib/vec/common.h"
#include "llk_lib/vec/exp.h"
#include "llk_lib/vec/fast_exp.h"
#include "llk_lib/vec/gelu.h"
#include "llk_lib/vec/hardsigmoid.h"
#include "llk_lib/vec/hardtanh.h"
#include "llk_lib/vec/recip.h"
#include "llk_lib/vec/relu_max.h"
#include "llk_lib/vec/selu.h"
#include "llk_lib/vec/sigmoid.h"
#include "llk_lib/vec/silu.h"
#include "llk_lib/vec/softplus.h"
#include "llk_lib/vec/tanh.h"

namespace llk2::llk_lib
{

namespace cfg = hal::cfg;
using hal::sync::StallCondition;
using hal::sync::StallTarget;

// ---- parameters -----------------------------------------------------------------------------------------------------

/// Parameters of an SFPU activation on the thread that drives the SFPU. configure and execute take the same struct.
struct vec_params
{
    VecFunction function;     ///< which function
    VecForm form;             ///< which of its algorithms
    VecRegion region;         ///< the datums of each tile it rewrites
    std::uint32_t scale;      ///< exp's input scale, an FP32 bit pattern
    bool dest_fp32;           ///< 32-bit Dest mode
    std::uint32_t tiles;      ///< tiles of the tensor, row-major in Dest
    std::uint32_t face_r_dim; ///< rows per face
    std::uint32_t num_faces;  ///< faces per tile
};

/// The Plan of an SFPU activation as parameters.
constexpr vec_params vec_params_from(const Plan& plan)
{
    return {
        .function   = plan.math.vec,
        .form       = plan.math.vec_form,
        .region     = plan.math.vec_region,
        .scale      = plan.math.scale,
        .dest_fp32  = plan.math.dest_fp32,
        .tiles      = plan.work.tile_rows * plan.work.tile_cols,
        .face_r_dim = plan.work.face_r_dim,
        .num_faces  = plan.work.num_faces,
    };
}

// ---- constants and helpers ------------------------------------------------------------------------------------------

inline constexpr std::uint32_t vec_max_rwc_step = 15;   ///< the largest step SETRWC takes (a 4-bit field)
inline constexpr std::uint32_t vec_max_rwc_rows = 1024; ///< Dest rows the Dest row counter runs through without wrapping (10 bits: all of a 16-bit Dest)

/// How the function walks a tile: see the file comment.
enum class VecWalk
{
    Faces,     ///< every face, one call per face
    Tile,      ///< the whole tile in one call
    LeftFaces, ///< faces 0 and 2 (tile columns 0..15), one call each
};

/// The walk of params: the fast exp's body is a whole tile; any other body runs per face, on every face or, for a
/// region of the left faces, on faces 0 and 2.
constexpr VecWalk vec_walk(const vec_params& params)
{
    if (params.function == VecFunction::Exp && params.form == VecForm::Fast)
    {
        return VecWalk::Tile;
    }
    return params.region == VecRegion::All ? VecWalk::Faces : VecWalk::LeftFaces;
}

/// The bf16 pattern of an FP32 bit pattern, truncated: what 1.0's exp takes as its scale (scale_fp32 >> 16).
constexpr std::uint16_t vec_bf16_of(std::uint32_t fp32)
{
    return static_cast<std::uint16_t>(fp32 >> 16);
}

/// Whether P's tiles run in one call, back to back: the walk is Tile, there is more than one
/// tile, and the Dest row counter can run through all of them. A tensor's tiles follow each other in Dest (row-major,
/// one tile's rows after the other's), so the SFPI code's own row steps carry it from tile to tile.
constexpr bool vec_block_walk(const vec_params& params)
{
    return vec_walk(params) == VecWalk::Tile && params.tiles > 1 && params.tiles * params.num_faces * params.face_r_dim <= vec_max_rwc_rows;
}

// ---- configure ------------------------------------------------------------------------------------------------------

/// Runs the init of function P.function (llk_lib/vec/<function>.h): its programmable constants and lookup tables.
/// flatten: the init is part of this op's configure, so its body is here and not behind a call. Left to itself the
/// compiler keeps an init that two ops of a kernel share (the same exp over blocks of two widths) as one function,
/// and a compiler that removes the writes a thread already made cannot see into a call.
template <vec_params P>
inline __attribute__((always_inline, flatten)) void vec_function_init()
{
#ifdef LLK2_SFPU_THREAD
    constexpr bool fast = P.form == VecForm::Fast;
    constexpr bool fp32 = P.dest_fp32;
    if constexpr (P.function == VecFunction::Silu)
    {
        vec_silu_init<fast, fp32>();
    }
    else if constexpr (P.function == VecFunction::Tanh)
    {
        vec_tanh_init<fast, fp32>();
    }
    else if constexpr (P.function == VecFunction::Gelu)
    {
        vec_gelu_init<fast, fp32>();
    }
    else if constexpr (P.function == VecFunction::GeluTanh)
    {
        vec_gelu_tanh_init<fast, fp32>();
    }
    else if constexpr (P.function == VecFunction::ReluMax)
    {
        vec_relu_max_init<fast, fp32>();
    }
    else if constexpr (P.function == VecFunction::Sigmoid)
    {
        vec_sigmoid_init<fast, fp32>();
    }
    else if constexpr (P.function == VecFunction::Hardsigmoid)
    {
        vec_hardsigmoid_init<fast, fp32>();
    }
    else if constexpr (P.function == VecFunction::Hardtanh)
    {
        vec_hardtanh_init<fast, fp32>();
    }
    else if constexpr (P.function == VecFunction::Selu)
    {
        vec_selu_init<fast, fp32>();
    }
    else if constexpr (P.function == VecFunction::Softplus)
    {
        vec_softplus_init<fast, fp32>();
    }
    else if constexpr (P.function == VecFunction::Reciprocal)
    {
        vec_recip_init<fast, fp32>();
    }
    else if constexpr (P.function == VecFunction::Exp)
    {
        // The exp's bodies are separate programs: the fast one folds the scale into its constants.
        if constexpr (fast)
        {
            // llk-lib's own, in HAL calls.
            vec_fast_exp_init<P.scale>();
        }
        else if constexpr (P.region != VecRegion::LeftFacesEvenColumns)
        {
            vec_exp_init<fp32>(); // the exact exp multiplies by its scale in its body
        }
        // The half-face body (LeftFacesEvenColumns): no init. It loads its own constants; it needs only the setup
        // vec_configure does.
    }
#endif
}

/// Programs the SFPU for P: its Dest mode, lane config, address modifier and row counters, then the function's init.
template <vec_params P>
inline __attribute__((always_inline)) void vec_configure()
{
    static_assert(
        P.num_faces == faces_per_full_tile && P.face_r_dim == max_face_rows, "32x32 tiles only, for now: the face loop below assumes four 16-row faces");
    static_assert(P.function != VecFunction::None, "name an SFPU function");

    // The SFPU's Dest mode is ALU config, shared with the FPU: wait for both to drain before writing it.
    // The wait is this write's alone (platform::stall_scope); nothing after it needs it.
    const auto write_dest_mode = [] __attribute__((always_inline))
    { cfg::write<cfg::Access::TensixCfgUnit, cfg::AluAccCtrl::SFPU_Fp32_enabled, cfg::Sec::S0, P.dest_fp32 ? 1u : 0u>(); };
    platform::stall_scope<StallTarget::Config, StallCondition::MathIdle | StallCondition::SfpuIdle>(write_dest_mode);

    vec_reset_lane_control();

    // The SFPI address modifier: no counter moves.
    set_hold_addr_mod();

    reset_math_counters();

    vec_function_init<P>();
}

// ---- execute --------------------------------------------------------------------------------------------------------

/// Runs function P.function on one face, from the Dest row counter on: its iterations (llk_lib/vec/<function>.h).
template <vec_params P, class... Scalars>
inline __attribute__((always_inline)) void vec_function_face([[maybe_unused]] Scalars... scalars)
{
#ifdef LLK2_SFPU_THREAD
    constexpr bool fast = P.form == VecForm::Fast;
    constexpr bool fp32 = P.dest_fp32;
    constexpr int n     = vec_iterations_per_face; // the SFPI functions take their iteration count as int
    if constexpr (P.function == VecFunction::Silu)
    {
        vec_silu_iterations<fast, fp32, n>();
    }
    else if constexpr (P.function == VecFunction::Tanh)
    {
        vec_tanh_iterations<fast, fp32, n>();
    }
    else if constexpr (P.function == VecFunction::Gelu)
    {
        vec_gelu_iterations<fast, fp32, n>();
    }
    else if constexpr (P.function == VecFunction::GeluTanh)
    {
        vec_gelu_tanh_iterations<fast, fp32, n>();
    }
    else if constexpr (P.function == VecFunction::ReluMax)
    {
        vec_relu_max_iterations<fast, fp32, n>(scalars...);
    }
    else if constexpr (P.function == VecFunction::Sigmoid)
    {
        vec_sigmoid_iterations<fast, fp32, n>();
    }
    else if constexpr (P.function == VecFunction::Hardsigmoid)
    {
        vec_hardsigmoid_iterations<fast, fp32, n>();
    }
    else if constexpr (P.function == VecFunction::Hardtanh)
    {
        vec_hardtanh_iterations<fast, fp32, n>(scalars...);
    }
    else if constexpr (P.function == VecFunction::Selu)
    {
        vec_selu_iterations<fast, fp32, n>(scalars...);
    }
    else if constexpr (P.function == VecFunction::Softplus)
    {
        vec_softplus_iterations<fast, fp32, n>(scalars...);
    }
    else if constexpr (P.function == VecFunction::Reciprocal)
    {
        vec_recip_iterations<fast, fp32, n>();
    }
    else if constexpr (P.function == VecFunction::Exp)
    {
        static_assert(!fast, "the fast exp walks the whole tile: vec_function_tile");
        if constexpr (P.region == VecRegion::LeftFacesEvenColumns)
        {
            // The half-face body: 4 iterations, the even columns; the exact exp or the polynomial.
            vec_exp_even_columns_iterations<P.form == VecForm::Exact, vec_bf16_of(P.scale), fp32>();
        }
        else
        {
            // The exact exp; its scale is a bf16 pattern, multiplied in when it is not 1.
            static_assert(P.form == VecForm::Exact, "the polynomial exp has a half-face body only");
            constexpr bool scaled = P.scale != vec_fp32_one;
            vec_exp_iterations<fp32, scaled, n>(vec_bf16_of(P.scale));
        }
    }
#endif
}

/// Runs function P.function on `tiles` whole tiles that follow each other in Dest, from the Dest row counter on: the
/// functions whose walk is Tile. tiles is a ct or an rt (vec_execute). The fast exp's walk is llk-lib's own
/// (vec_fast_exp_tiles, HAL calls); without SFPLOADMACRO (the ttsim build) it is the plain load / store loop, one
/// tile's iterations per pass: its row steps carry the Dest row counter on from tile to tile, as the macro form's do.
template <vec_params P, class Tiles>
inline __attribute__((always_inline)) void vec_function_tile([[maybe_unused]] const Tiles& tiles)
{
#ifdef LLK2_SFPU_THREAD
    static_assert(P.function == VecFunction::Exp, "only the fast exp has a whole-tile form");
    if constexpr (!vec_has_loadmacro)
    {
        constexpr int iterations_per_tile = vec_iterations_per_face * P.num_faces; // 32: the SFPI code steps through all four faces
        for (std::uint32_t tile = 0; tile < wrapper::value(tiles); tile++)
        {
            vec_fast_exp_plain_iterations<iterations_per_tile>();
        }
    }
    else
    {
        vec_fast_exp_tiles(tiles);
    }
#endif
}

/// The function on every tile of the tensor that starts at dest_row (the absolute Dest row), in place. tiles is their
/// count: a ct (P.tiles) or, for a size known only at run time, an rt of at most that
/// (resolver/work_counts.h); how the tiles are walked was chosen on P.tiles.
template <vec_params P, class Tiles, ctrt<Tiles, std::uint32_t> = 0, class... Scalars>
inline __attribute__((always_inline)) void vec_execute(std::uint32_t dest_row, const Tiles& tiles, Scalars... scalars)
{
    using namespace hal::counters::rwc;
    // The Dest row counter moves one face per face. SETRWC steps at most 15 rows, so a 16-row face is two half steps,
    // each through the CR: SFPI's own row steps inside the face do not count. LeftFaces moves two faces per call: the
    // one it ran on and the one to its right, which it skips.
    constexpr VecWalk walk                 = vec_walk(P);
    constexpr std::uint32_t faces_per_call = walk == VecWalk::LeftFaces ? 2 : 1;
    constexpr std::uint32_t calls          = P.num_faces / faces_per_call;
    constexpr std::uint32_t half_face      = P.face_r_dim / 2;   // two steps per face
    constexpr std::uint32_t half_steps     = 2 * faces_per_call; // per call
    static_assert(half_face <= vec_max_rwc_step, "a half face must fit one SETRWC step");
    constexpr MathCounter next_half_face {.dest = AdvanceAndReset(half_face)};
    if constexpr (wrapper::is_rt_v<Tiles>)
    {
        // Above the bound the block walk would run the Dest row counter past the tensor's section.
        LLK_ASSERT(wrapper::value(tiles) >= 1 && wrapper::value(tiles) <= P.tiles, "a run-time tile count is outside 1..its bound");
    }

    if constexpr (vec_block_walk(P))
    {
        // Every tile in one call: the Dest offset at the tensor's first tile, the row counter running on through all
        // of them.
        set_math_dest_tile<P>(dest_row, 0);
        vec_function_tile<P>(tiles);
        set<MathCounter {.dest = Set(0)}>();
        return;
    }
    for (std::uint32_t tile = 0; tile < wrapper::value(tiles); tile++)
    {
        set_math_dest_tile<P>(dest_row, tile);
        if constexpr (walk == VecWalk::Tile)
        {
            vec_function_tile<P>(ct<std::uint32_t {1}> {});
        }
        else
        {
// unroll 0: the loop stays a loop, with one copy of the function's iterations for all the faces of a tile.
#pragma GCC unroll 0
            for (std::uint32_t face = 0; face < calls; face++)
            {
                vec_function_face<P>(scalars...);
// unroll 4: the two or four SETRWCs are issued one after the other, no loop.
#pragma GCC unroll 4
                for (std::uint32_t step = 0; step < half_steps; step++)
                {
                    set<next_half_face>();
                }
            }
        }
        set<MathCounter {.dest = Set(0)}>(); // the next tile starts from its own Dest offset
    }
}

// ---- per-thread entry -----------------------------------------------------------------------------------------------

/// The driving thread's part of a Compute API vec_ call: configure, then the function on every tile from dest_row.
template <Thread On, Plan Pl, class... Scalars>
inline __attribute__((always_inline)) void vec(const WorkCounts<Pl.work>& counts, std::uint32_t dest_row, Scalars... scalars)
{
    static_assert(On == Thread::Math || On == Thread::Pack, "the SFPU is driven from the math or the pack thread");
    constexpr vec_params params = vec_params_from(Pl);
    if constexpr (runs_on<On>)
    {
        vec_configure<params>();
        vec_execute<params>(dest_row, plan_count<WorkCount::Tiles, Pl.work.tile_rows * Pl.work.tile_cols>(counts), scalars...);
    }
}

} // namespace llk2::llk_lib
