// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// Quasar SFPU op dispatcher — shared by sfpu_quasar_srca_test.cpp and
// sfpu_quasar_dest_test.cpp.  Mirrors tests/helpers/include/sfpu_operations.h
// for the non-Quasar architectures.
//
// Include inside #ifdef LLK_TRISC_MATH in both .cpp files.
//
// SFPU_UNARY_OPERATION is a constexpr SfpuType emitted into build.h by the
// Python harness via MATH_OP(mathop=op).  quasar_sfpu_init and quasar_sfpu_call
// are templates on this constant so the compiler resolves them at compile time
// to a direct call to the single matching _calculate_*_ function with no switch
// and no dead code.
//
// Fill, swiglu, binary int, binary float, binary max/min, and where are NOT
// dispatched here — they are handled inline in the .cpp files with #ifdef /
// if constexpr because their SFPU calling conventions differ from the standard
// _llk_math_eltwise_unary_sfpu_params_ interface.

#pragma once

// Standard unary ops
#include "experimental/ckernel_sfpu_abs.h"
#include "sfpu/ckernel_sfpu_exp.h"
#include "sfpu/ckernel_sfpu_gelu.h"
#include "sfpu/ckernel_sfpu_recip.h"
#include "sfpu/ckernel_sfpu_relu.h"
#include "sfpu/ckernel_sfpu_rsqrt.h"
#include "sfpu/ckernel_sfpu_sigmoid.h"
#include "sfpu/ckernel_sfpu_silu.h"
#include "sfpu/ckernel_sfpu_sqrt.h"
#include "sfpu/ckernel_sfpu_square.h"
#include "sfpu/ckernel_sfpu_tanh.h"
// Structural exceptions (handled inline with if constexpr / #ifdef in the .cpp)
#include "experimental/ckernel_sfpu_fill.h"
#include "experimental/ckernel_sfpu_swiglu.h"
// Binary and ternary ops (handled inline in the .cpp)
#include "experimental/ckernel_sfpu_binary_max_min.h"
#include "llk_sfpu/ckernel_sfpu_binary.h"
#include "llk_sfpu/ckernel_sfpu_where.h"
#include "sfpu/ckernel_sfpu_add.h"
#include "sfpu/ckernel_sfpu_binary_comp.h"
#include "sfpu/ckernel_sfpu_mul_int32.h"

using namespace ckernel;
using namespace ckernel::math;
using namespace ckernel::sfpu;

// sfpu_op_dispatcher: one specialisation per standard unary op.
// Each exposes static void call(int tile_idx, int n), and optionally
// static void init() for ops that need per-operation initialisation.

template <SfpuType op>
struct sfpu_op_dispatcher;

template <>
struct sfpu_op_dispatcher<SfpuType::abs>
{
    static void call(int tile_idx, int n)
    {
        _llk_math_eltwise_unary_sfpu_params_(_calculate_abs_, tile_idx, n);
    }
};

template <>
struct sfpu_op_dispatcher<SfpuType::exponential>
{
    static void call(int tile_idx, int n)
    {
        _llk_math_eltwise_unary_sfpu_params_(_calculate_exp_<true>, tile_idx, n);
    }
};

template <>
struct sfpu_op_dispatcher<SfpuType::gelu>
{
    static void init()
    {
        _init_gelu_();
    }

    static void call(int tile_idx, int n)
    {
        _llk_math_eltwise_unary_sfpu_params_(_calculate_gelu_, tile_idx, n);
    }
};

template <>
struct sfpu_op_dispatcher<SfpuType::relu>
{
    static void call(int tile_idx, int n)
    {
        _llk_math_eltwise_unary_sfpu_params_(_calculate_relu_, tile_idx, n);
    }
};

template <>
struct sfpu_op_dispatcher<SfpuType::reciprocal>
{
    static void call(int tile_idx, int n)
    {
        _llk_math_eltwise_unary_sfpu_params_(_calculate_reciprocal_<true>, tile_idx, n);
    }
};

template <>
struct sfpu_op_dispatcher<SfpuType::sqrt>
{
    static void call(int tile_idx, int n)
    {
        _llk_math_eltwise_unary_sfpu_params_(_calculate_sqrt_<true>, tile_idx, n);
    }
};

template <>
struct sfpu_op_dispatcher<SfpuType::tanh>
{
    static void call(int tile_idx, int n)
    {
        _llk_math_eltwise_unary_sfpu_params_(_calculate_tanh_<true>, tile_idx, n);
    }
};

template <>
struct sfpu_op_dispatcher<SfpuType::sigmoid>
{
    static void call(int tile_idx, int n)
    {
        _llk_math_eltwise_unary_sfpu_params_(_calculate_sigmoid_, tile_idx, n);
    }
};

template <>
struct sfpu_op_dispatcher<SfpuType::silu>
{
    static void call(int tile_idx, int n)
    {
        _llk_math_eltwise_unary_sfpu_params_(_calculate_silu_, tile_idx, n);
    }
};

template <>
struct sfpu_op_dispatcher<SfpuType::rsqrt>
{
    static void call(int tile_idx, int n)
    {
        _llk_math_eltwise_unary_sfpu_params_(_calculate_rsqrt_, tile_idx, n);
    }
};

template <>
struct sfpu_op_dispatcher<SfpuType::square>
{
    static void call(int tile_idx, int n)
    {
        _llk_math_eltwise_unary_sfpu_params_(_calculate_square_, tile_idx, n);
    }
};

// quasar_sfpu_init<OPERATION>() — call once before the tile loop.
// OPERATION is SFPU_UNARY_OPERATION, a constexpr SfpuType baked in by the harness.
template <SfpuType OPERATION>
inline void quasar_sfpu_init()
{
    if constexpr (OPERATION == SfpuType::gelu)
    {
        sfpu_op_dispatcher<SfpuType::gelu>::init();
    }
}

// quasar_sfpu_call<OPERATION>(tile_idx, n) — per-tile dispatch.
// Resolves at compile time to a direct call to the matching _calculate_*_ function.
template <SfpuType OPERATION>
inline void quasar_sfpu_call(int tile_idx, int num_sfpu_iterations)
{
    sfpu_op_dispatcher<OPERATION>::call(tile_idx, num_sfpu_iterations);
}
