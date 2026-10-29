// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// llk-lib, SFPU activations (vec_silu, vec_tanh, ...) on Blackhole. Everything around the activation is HAL calls
// (tt_llk_blackhole/llk_lib/hal/): the SFPU setup, the address modifier, the Dest offset of each tile, the row
// counter steps between faces and the stalls. The activation itself is SFPI code, not HAL: the calculate_* and
// *_init functions of tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_*.h and
// tt_llk_blackhole/common/inc/sfpu/. This file only calls them. Ported from llk_math_eltwise_unary_sfpu.h,
// llk_math_eltwise_sfpu_common.h and llk_math_eltwise_unary_sfpu_params.h.
//
//   vec_params                one struct; vec_configure and vec_execute take the same one
//   vec_params_from(Plan)     fills it from the Resolver's Plan
//   vec_configure<P>()        waits for math and SFPU to drain, sets the SFPU's Dest mode, resets the lane config,
//                             programs the address modifier SFPI loads and stores use, clears the row counters,
//                             then runs the function's SFPI init (its constants and lookup tables)
//   vec_execute<P>(row, ...)  the function on every tile of the tensor at Dest row `row`, in place. The scalars
//                             (relu_max's max, ...) are FP32 bit patterns, passed on to the SFPI function
//   vec<On, Plan>(row, ...)   the per-thread call the Compute API makes: configure, then execute, only when this
//                             build is for thread On, the math or the pack thread
//
// How one tile is done. The SFPU has 32 lanes. An SFPI load reads 32 datums from Dest: the even or the odd columns
// of four rows, at the Dest row counter plus the Dest offset. One SFPI iteration loads, computes and stores those
// 32 datums, then moves the row counter on; 8 iterations cover one 16 x 16 face. So for each tile: point the Dest
// offset at the tile, and for each of the 4 faces call the SFPI
// function once (8 iterations) and move the row counter's CR to the next face. After each tile, the row counter goes
// back to 0.
//
// Which thread drives the SFPU. Either the math or the pack thread can issue SFPU instructions; each has its own
// Dest offset register, row counters and address modifiers, so configure and execute are the same on both. The
// hand-over around the call is the owners' (llk_lib/sync.h), not the op's: the Compute API takes the Dest tensor over
// for the SFPU on the driving thread before vec runs (a STALLWAIT for the FPU on math; on pack, math drains the FPU and
// posts math_pack_sync, and pack waits with every instruction class held, SFPU and Config included), and the op that
// reads the result takes it over from the SFPU (pack: a STALLWAIT on the pack thread until the SFPU is idle).
// Either way the Dest offset is the tensor's address (the absolute Dest row) plus the tile's rows: no section base.
//
// Not HAL yet:
//   - SFPCONFIG (the SFPU lane config): ckernel::sfpu::_init_sfpu_config_reg(), flagged below.
//   - The SFPI init functions also issue one SETRWC each (math::reset_counters); they are called as they are.

#pragma once

#include <cstdint>

#include "ckernel.h"
#include "hal/cfg.h"
#include "hal/sync.h"
#include "hal/tensix_counters.h"
#include "llk_lib/math_common.h" // counter_step, set_addr_mod
#include "llk_lib/thread.h"
#include "resolver/plan.h"
#include "resolver/vec_function.h"

// The SFPI code is compiled only into the threads that can drive the SFPU (math, pack); unpack plans only. The
// metal headers are named with their folder: tt-llk's common/inc/sfpu has files of the same name.
#if defined(LLK_TRISC_MATH) || defined(TRISC_MATH) || defined(LLK_TRISC_PACK) || defined(TRISC_PACK)
#define LLK2_SFPU_THREAD 1
#include "cmath_common.h"
#include "llk_sfpu/ckernel_sfpu_activations.h"
#include "llk_sfpu/ckernel_sfpu_gelu.h"
#include "llk_sfpu/ckernel_sfpu_hardtanh.h"
#include "llk_sfpu/ckernel_sfpu_relu.h"
#include "llk_sfpu/ckernel_sfpu_selu.h"
#include "llk_sfpu/ckernel_sfpu_sigmoid.h"
#include "llk_sfpu/ckernel_sfpu_silu.h"
#include "llk_sfpu/ckernel_sfpu_softplus.h"
#include "llk_sfpu/ckernel_sfpu_tanh.h"
#include "sfpu/ckernel_sfpu_load_config.h"
#endif

namespace llk2::llk_lib
{

namespace cfg = hal::cfg;
using hal::sync::StallCondition;
using hal::sync::StallTarget;

/// Parameters of an SFPU activation on the thread that drives the SFPU. configure and execute take the same struct.
struct vec_params
{
    VecFunction function;     ///< which activation
    bool fast;                ///< approximate mode
    bool dest_fp32;           ///< 32-bit Dest mode
    std::uint32_t tiles;      ///< tiles of the tensor, row-major in Dest
    std::uint32_t face_r_dim; ///< rows per face
    std::uint32_t num_faces;  ///< faces per tile
};

/// The Plan of an SFPU activation as parameters.
constexpr vec_params vec_params_from(const Plan& p)
{
    return {
        static_cast<VecFunction>(p.math.function),
        p.math.fast,
        p.math.dest_fp32,
        p.work.rt_dim * p.work.ct_dim,
        p.work.face_r_dim,
        p.work.num_faces,
    };
}

inline constexpr std::uint32_t vec_lanes               = 32; ///< SFPU lanes: datums one SFPI iteration loads, computes and stores
inline constexpr std::uint32_t vec_iterations_per_face = max_face_rows * face_cols / vec_lanes; ///< 8 iterations of 4 rows x 8 columns: one face
inline constexpr std::uint32_t vec_max_rwc_step        = 15;                                    ///< the largest step SETRWC takes (a 4-bit field)

/// The address modifier slot SFPI loads and stores name (sfpi_constants.h: SFPLOAD_ADDR_MODE_NOINC). It moves no
/// counter: SFPI steps the Dest row counter itself.
inline constexpr cfg::Sec vec_sfpi_addr_mod = cfg::Sec::S7;

/// Runs the SFPI init of function P.function: its programmable constants and lookup tables. SFPI, not HAL.
template <vec_params P>
inline __attribute__((always_inline)) void vec_function_init()
{
#ifdef LLK2_SFPU_THREAD
    using namespace ckernel::sfpu;
    constexpr bool fast = P.fast;
    constexpr bool fp32 = P.dest_fp32;
    if constexpr (P.function == VecFunction::Silu)
    {
        silu_init<fast>();
    }
    else if constexpr (P.function == VecFunction::Tanh)
    {
        tanh_init<fast, fp32>();
    }
    else if constexpr (P.function == VecFunction::Gelu)
    {
        gelu_init<fast, fp32>();
    }
    else if constexpr (P.function == VecFunction::GeluTanh)
    {
        if constexpr (fast)
        {
            gelu_tanh_fast_init();
        }
        else
        {
            gelu_tanh_init();
        }
    }
    else if constexpr (P.function == VecFunction::ReluMax)
    {
        relu_max_init();
    }
    else if constexpr (P.function == VecFunction::Sigmoid)
    {
        sigmoid_init<fast>();
    }
    else if constexpr (P.function == VecFunction::Hardsigmoid)
    {
        hardsigmoid_init<fast>();
    }
    else if constexpr (P.function == VecFunction::Hardtanh)
    {
        hardtanh_init();
    }
    else if constexpr (P.function == VecFunction::Selu)
    {
        selu_init();
    }
    else if constexpr (P.function == VecFunction::Softplus)
    {
        softplus_init();
    }
#endif
}

/// Runs function P.function on one face, from the Dest row counter on. SFPI, not HAL.
template <vec_params P, class... Scalars>
inline __attribute__((always_inline)) void vec_function_face([[maybe_unused]] Scalars... scalars)
{
#ifdef LLK2_SFPU_THREAD
    using namespace ckernel::sfpu;
    constexpr bool fast = P.fast;
    constexpr bool fp32 = P.dest_fp32;
    constexpr int n     = vec_iterations_per_face; // the SFPI functions take their iteration count as int
    if constexpr (P.function == VecFunction::Silu)
    {
        calculate_silu<fp32, n>();
    }
    else if constexpr (P.function == VecFunction::Tanh)
    {
        calculate_tanh<fast, fp32, n>();
    }
    else if constexpr (P.function == VecFunction::Gelu)
    {
        calculate_gelu<fast, fp32, n>();
    }
    else if constexpr (P.function == VecFunction::GeluTanh)
    {
        if constexpr (fast)
        {
            calculate_gelu_tanh_fast<fp32, n>();
        }
        else
        {
            calculate_gelu_tanh<fp32, n>();
        }
    }
    else if constexpr (P.function == VecFunction::ReluMax)
    {
        _relu_max_<sfpi::vFloat, fast, n, std::uint32_t>(scalars...);
    }
    else if constexpr (P.function == VecFunction::Sigmoid)
    {
        calculate_sigmoid<fast, fp32, n>();
    }
    else if constexpr (P.function == VecFunction::Hardsigmoid)
    {
        calculate_activation<fast, ckernel::ActivationType::Hardsigmoid, n>();
    }
    else if constexpr (P.function == VecFunction::Hardtanh)
    {
        calculate_hardtanh<fast, n>(scalars...);
    }
    else if constexpr (P.function == VecFunction::Selu)
    {
        calculate_selu<fast, fp32, n>(scalars...);
    }
    else if constexpr (P.function == VecFunction::Softplus)
    {
        calculate_softplus<fast, fp32, n>(scalars...);
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
    hal::sync::wait::stall<StallTarget::Config, StallCondition::MathIdle | StallCondition::SfpuIdle>();
    cfg::write<cfg::Access::TensixCfgUnit, cfg::AluAccCtrl::SFPU_Fp32_enabled, cfg::Sec::S0, P.dest_fp32 ? 1u : 0u>();

#ifdef LLK2_SFPU_THREAD
    // Not HAL: the HAL has no SFPCONFIG. This writes 0 to every lane's LaneConfig (SFPCONFIG VD = 15).
    ckernel::sfpu::_init_sfpu_config_reg();
#endif

    // The SFPI address modifier: no counter moves.
    set_addr_mod<vec_sfpi_addr_mod, counter_stay, counter_stay, counter_stay>();

    reset_math_counters();

    vec_function_init<P>();
}

/// The function on every tile of the tensor that starts at dest_row (the absolute Dest row), in place.
template <vec_params P, class... Scalars>
inline __attribute__((always_inline)) void vec_execute(std::uint32_t dest_row, Scalars... scalars)
{
    using namespace hal::counters::rwc;
    // The Dest row counter moves one face per face. SETRWC steps at most 15 rows, so a 16-row face is two half steps,
    // each through the CR: SFPI's own row steps inside the face do not count.
    constexpr std::uint32_t half_face = P.face_r_dim / 2; // two steps per face
    static_assert(half_face <= vec_max_rwc_step, "a half face must fit one SETRWC step");
    constexpr MathCounter next_half_face {.dest = AdvanceAndReset(half_face)};

    for (std::uint32_t tile = 0; tile < P.tiles; tile++)
    {
        cfg::write<cfg::Access::TensixCfgUnit, cfg::DestTargetRegCfgMath::Offset, cfg::Sec::S0>(dest_row + tile * dest_rows_per_tile(P));
#pragma GCC unroll 0
        for (std::uint32_t face = 0; face < P.num_faces; face++)
        {
            vec_function_face<P>(scalars...);
            set<next_half_face>();
            set<next_half_face>();
        }
        set<MathCounter {.dest = Set(0)}>(); // the next tile starts from its own Dest offset
    }
}

/// The driving thread's part of a Compute API vec_ call: configure, then the function on every tile from dest_row.
template <Thread On, Plan Pl, class... Scalars>
inline __attribute__((always_inline)) void vec(std::uint32_t dest_row, Scalars... scalars)
{
    static_assert(On == Thread::Math || On == Thread::Pack, "the SFPU is driven from the math or the pack thread");
    constexpr vec_params p = vec_params_from(Pl);
    if constexpr (runs_on<On>)
    {
        vec_configure<p>();
        vec_execute<p>(dest_row, scalars...);
    }
}

} // namespace llk2::llk_lib
