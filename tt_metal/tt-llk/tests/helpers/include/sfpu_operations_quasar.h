// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel.h"
#include "llk_defs.h"
#include "llk_sfpu/llk_math_eltwise_unary_sfpu_macros.h"

// To add a new Quasar unary SFPU operation:
// 1. Include its `ckernel_sfpu_<op>.h` below.
// 2. Add the `SfpuType` enumerator to the `if constexpr` chain in
//    call_unary_sfpu_operation_quasar() (and to init_unary_sfpu_operation_quasar()
//    if the op needs an init step).
#include "experimental/ckernel_sfpu_abs.h"
#include "llk_sfpu/ckernel_sfpu_clamp.h"
#include "llk_sfpu/ckernel_sfpu_comp.h"
#include "llk_sfpu/ckernel_sfpu_exp.h"
#include "llk_sfpu/ckernel_sfpu_gelu.h"
#include "llk_sfpu/ckernel_sfpu_negative.h"
#include "llk_sfpu/ckernel_sfpu_recip.h"
#include "llk_sfpu/ckernel_sfpu_rsqrt.h"
#include "llk_sfpu/ckernel_sfpu_softplus.h"
#include "llk_sfpu/ckernel_sfpu_square.h"
#include "llk_sfpu/ckernel_sfpu_tanh.h"
#include "llk_sfpu/ckernel_sfpu_trigonometry.h"
#include "llk_sfpu/ckernel_sfpu_typecast.h"
#include "sfpu/ckernel_sfpu_relu.h"
#include "sfpu/ckernel_sfpu_sigmoid.h"
#include "sfpu/ckernel_sfpu_silu.h"
#include "sfpu/ckernel_sfpu_sqrt.h"

// ─────────────────────────────────────────────────────────────────────────────
// SFPU parity set — gated includes
//
// These 57 kernels are implemented in pure sfpi:: on Blackhole and are not yet ported to
// Quasar. Their tests are written ahead of the port, so every include and every dispatch
// branch below is guarded on the header actually existing. A guard that does not fire
// costs nothing; a guard that does fire activates the op's full sweep with no test edit.
//
// The Python side gates the same ops on the same header basenames via
// helpers/sfpu_port_quasar.py, so the two cannot disagree about what is available. Bare
// basenames are used deliberately rather than "llk_sfpu/<name>.h": all three Quasar SFPU
// trees are on the include path as bare-name roots, so the gate does not care which one a
// future port lands in.
//
// To activate a ported kernel: nothing here needs editing. If its Quasar signature differs
// from the Blackhole one mirrored below, adjust that branch only.
// ─────────────────────────────────────────────────────────────────────────────

// clang-format off
#if __has_include("ckernel_sfpu_activations.h")
#include "ckernel_sfpu_activations.h"
#define QSR_HAS_ACTIVATIONS 1
#endif
#if __has_include("ckernel_sfpu_add1.h")
#include "ckernel_sfpu_add1.h"
#define QSR_HAS_ADD1 1
#endif
#if __has_include("ckernel_sfpu_bitwise.h")
#include "ckernel_sfpu_bitwise.h"
#define QSR_HAS_BITWISE 1
#endif
#if __has_include("ckernel_sfpu_bitwise_not.h")
#include "ckernel_sfpu_bitwise_not.h"
#define QSR_HAS_BITWISE_NOT 1
#endif
#if __has_include("ckernel_sfpu_cast_fp32_to_fp16a.h")
#include "ckernel_sfpu_cast_fp32_to_fp16a.h"
#define QSR_HAS_CAST_FP32_TO_FP16A 1
#endif
#if __has_include("ckernel_sfpu_cbrt.h")
#include "ckernel_sfpu_cbrt.h"
#define QSR_HAS_CBRT 1
#endif
#if __has_include("ckernel_sfpu_celu.h")
#include "ckernel_sfpu_celu.h"
#define QSR_HAS_CELU 1
#endif
#if __has_include("ckernel_sfpu_digamma.h")
#include "ckernel_sfpu_digamma.h"
#define QSR_HAS_DIGAMMA 1
#endif
#if __has_include("ckernel_sfpu_elu.h")
#include "ckernel_sfpu_elu.h"
#define QSR_HAS_ELU 1
#endif
#if __has_include("ckernel_sfpu_erf.h")
#include "ckernel_sfpu_erf.h"
#define QSR_HAS_ERF 1
#endif
#if __has_include("ckernel_sfpu_erfc.h")
#include "ckernel_sfpu_erfc.h"
#define QSR_HAS_ERFC 1
#endif
#if __has_include("ckernel_sfpu_erfinv.h")
#include "ckernel_sfpu_erfinv.h"
#define QSR_HAS_ERFINV 1
#endif
#if __has_include("ckernel_sfpu_exp2.h")
#include "ckernel_sfpu_exp2.h"
#define QSR_HAS_EXP2 1
#endif
#if __has_include("ckernel_sfpu_expm1.h")
#include "ckernel_sfpu_expm1.h"
#define QSR_HAS_EXPM1 1
#endif
#if __has_include("ckernel_sfpu_fmod.h")
#include "ckernel_sfpu_fmod.h"
#define QSR_HAS_FMOD 1
#endif
#if __has_include("ckernel_sfpu_hardmish.h")
#include "ckernel_sfpu_hardmish.h"
#define QSR_HAS_HARDMISH 1
#endif
#if __has_include("ckernel_sfpu_hardshrink.h")
#include "ckernel_sfpu_hardshrink.h"
#define QSR_HAS_HARDSHRINK 1
#endif
#if __has_include("ckernel_sfpu_hardtanh.h")
#include "ckernel_sfpu_hardtanh.h"
#define QSR_HAS_HARDTANH 1
#endif
#if __has_include("ckernel_sfpu_heaviside.h")
#include "ckernel_sfpu_heaviside.h"
#define QSR_HAS_HEAVISIDE 1
#endif
#if __has_include("ckernel_sfpu_i0.h")
#include "ckernel_sfpu_i0.h"
#define QSR_HAS_I0 1
#endif
#if __has_include("ckernel_sfpu_i1.h")
#include "ckernel_sfpu_i1.h"
#define QSR_HAS_I1 1
#endif
#if __has_include("ckernel_sfpu_identity.h")
#include "ckernel_sfpu_identity.h"
#define QSR_HAS_IDENTITY 1
#endif
#if __has_include("ckernel_sfpu_lgamma.h")
#include "ckernel_sfpu_lgamma.h"
#define QSR_HAS_LGAMMA 1
#endif
#if __has_include("ckernel_sfpu_logical_not.h")
#include "ckernel_sfpu_logical_not.h"
#define QSR_HAS_LOGICAL_NOT 1
#endif
#if __has_include("ckernel_sfpu_polygamma.h")
#include "ckernel_sfpu_polygamma.h"
#define QSR_HAS_POLYGAMMA 1
#endif
#if __has_include("ckernel_sfpu_prelu.h")
#include "ckernel_sfpu_prelu.h"
#define QSR_HAS_PRELU 1
#endif
#if __has_include("ckernel_sfpu_rdiv.h")
#include "ckernel_sfpu_rdiv.h"
#define QSR_HAS_RDIV 1
#endif
#if __has_include("ckernel_sfpu_remainder.h")
#include "ckernel_sfpu_remainder.h"
#define QSR_HAS_REMAINDER 1
#endif
#if __has_include("ckernel_sfpu_rpow.h")
#include "ckernel_sfpu_rpow.h"
#define QSR_HAS_RPOW 1
#endif
#if __has_include("ckernel_sfpu_selu.h")
#include "ckernel_sfpu_selu.h"
#define QSR_HAS_SELU 1
#endif
#if __has_include("ckernel_sfpu_sign.h")
#include "ckernel_sfpu_sign.h"
#define QSR_HAS_SIGN 1
#endif
#if __has_include("ckernel_sfpu_softshrink.h")
#include "ckernel_sfpu_softshrink.h"
#define QSR_HAS_SOFTSHRINK 1
#endif
#if __has_include("ckernel_sfpu_softsign.h")
#include "ckernel_sfpu_softsign.h"
#define QSR_HAS_SOFTSIGN 1
#endif
#if __has_include("ckernel_sfpu_tanhshrink.h")
#include "ckernel_sfpu_tanhshrink.h"
#define QSR_HAS_TANHSHRINK 1
#endif
#if __has_include("ckernel_sfpu_unary_comp.h")
#include "ckernel_sfpu_unary_comp.h"
#define QSR_HAS_UNARY_COMP 1
#endif
#if __has_include("ckernel_sfpu_unary_power.h")
#include "ckernel_sfpu_unary_power.h"
#define QSR_HAS_UNARY_POWER 1
#endif
#if __has_include("ckernel_sfpu_unary_shift.h")
#include "ckernel_sfpu_unary_shift.h"
#define QSR_HAS_UNARY_SHIFT 1
#endif
#if __has_include("ckernel_sfpu_xielu.h")
#include "ckernel_sfpu_xielu.h"
#define QSR_HAS_XIELU 1
#endif

// Binary parity kernels
#if __has_include("ckernel_sfpu_atan2.h")
#include "ckernel_sfpu_atan2.h"
#define QSR_HAS_ATAN2 1
#endif
#if __has_include("ckernel_sfpu_binary_bitwise.h")
#include "ckernel_sfpu_binary_bitwise.h"
#define QSR_HAS_BINARY_BITWISE 1
#endif
#if __has_include("ckernel_sfpu_binary_fmod.h")
#include "ckernel_sfpu_binary_fmod.h"
#define QSR_HAS_BINARY_FMOD 1
#endif
#if __has_include("ckernel_sfpu_binary_pow.h")
#include "ckernel_sfpu_binary_pow.h"
#define QSR_HAS_BINARY_POW 1
#endif
#if __has_include("ckernel_sfpu_binary_remainder.h")
#include "ckernel_sfpu_binary_remainder.h"
#define QSR_HAS_BINARY_REMAINDER 1
#endif
#if __has_include("ckernel_sfpu_div_int32.h")
#include "ckernel_sfpu_div_int32.h"
#define QSR_HAS_DIV_INT32 1
#endif
#if __has_include("ckernel_sfpu_div_int32_floor.h")
#include "ckernel_sfpu_div_int32_floor.h"
#define QSR_HAS_DIV_INT32_FLOOR 1
#endif
#if __has_include("ckernel_sfpu_isclose.h")
#include "ckernel_sfpu_isclose.h"
#define QSR_HAS_ISCLOSE 1
#endif
#if __has_include("ckernel_sfpu_logsigmoid.h")
#include "ckernel_sfpu_logsigmoid.h"
#define QSR_HAS_LOGSIGMOID 1
#endif
#if __has_include("ckernel_sfpu_mask.h")
#include "ckernel_sfpu_mask.h"
#define QSR_HAS_MASK 1
#endif
#if __has_include("ckernel_sfpu_rsub_int32.h")
#include "ckernel_sfpu_rsub_int32.h"
#define QSR_HAS_RSUB_INT32 1
#endif

// Ternary parity kernels
#if __has_include("ckernel_sfpu_addcdiv.h")
#include "ckernel_sfpu_addcdiv.h"
#define QSR_HAS_ADDCDIV 1
#endif
#if __has_include("ckernel_sfpu_addcmul.h")
#include "ckernel_sfpu_addcmul.h"
#define QSR_HAS_ADDCMUL 1
#endif
#if __has_include("ckernel_sfpu_lerp.h")
#include "ckernel_sfpu_lerp.h"
#define QSR_HAS_LERP 1
#endif
#if __has_include("ckernel_sfpu_snake_beta.h")
#include "ckernel_sfpu_snake_beta.h"
#define QSR_HAS_SNAKE_BETA 1
#endif

// Tile-structural parity kernels
#if __has_include("ckernel_sfpu_alt_complex_rotate90.h")
#include "ckernel_sfpu_alt_complex_rotate90.h"
#define QSR_HAS_ALT_COMPLEX_ROTATE90 1
#endif
#if __has_include("ckernel_sfpu_int_sum.h")
#include "ckernel_sfpu_int_sum.h"
#define QSR_HAS_INT_SUM 1
#endif
#if __has_include("ckernel_sfpu_tiled_prod.h")
#include "ckernel_sfpu_tiled_prod.h"
#define QSR_HAS_TILED_PROD 1
#endif
// clang-format on

// Binary SFPU op headers (consumed by the binary dispatchers below). The op is
// selected via the LLK ckernel::BinaryOp enum (reused like Blackhole; the
// comparison and max/min enumerators were added to it in llk_defs.h).
//
// To add a new Quasar binary SFPU op:
// 1. Include its ckernel header below.
// 2. Add the enumerator to ckernel::BinaryOp (llk_defs.h) if it is not there.
// 3. Add the `if constexpr` branch in call_binary_sfpu_operation_quasar()
//    (and init_binary_sfpu_operation_quasar() if it needs an init step).
#include "llk_sfpu/ckernel_sfpu_binary.h"         // calculate_sfpu_binary / sfpu_binary_init (float mul/div)
#include "llk_sfpu/ckernel_sfpu_binary_max_min.h" // calculate_binary_max_min / _init_binary_max_min_
#include "llk_sfpu/ckernel_sfpu_quant.h"          // quant_family / quant_family_init (quant/requant/dequant)
#include "llk_sfpu/llk_math_eltwise_binary_sfpu_macros.h"
// Ternary plumbing, already present on Quasar for `where`; the four ternary parity ops
// (addcmul / addcdiv / lerp / snake_beta) reuse it.
#include "llk_sfpu/llk_math_eltwise_ternary_sfpu_macros.h"
#include "sfpu/ckernel_sfpu_add.h"         // _add_int_ (int add)
#include "sfpu/ckernel_sfpu_binary_comp.h" // calculate_binary_comp_int32 (int gt/lt/le/ge)
#include "sfpu/ckernel_sfpu_mul_int32.h"   // _mul_int32_ (int mul)

namespace test_utils
{
using namespace ckernel;
using namespace ckernel::math;
using namespace ckernel::sfpu;

template <auto>
inline constexpr bool unhandled_op = false;

/**
 * @brief Fixed scalar arguments the parity kernels bake in, as fp32 bit patterns.
 *
 * Mirrors tests/python_tests/helpers/sfpu_dispatch_constants.py, which the goldens and the
 * sfpu_domains edge probes read. Two independent copies of one number is a silent-drift
 * bug rather than duplication: change the threshold here without changing it there and the
 * golden keeps computing against a value the kernel no longer uses, which reads as a
 * kernel bug. Keep the two in step.
 */
namespace parity_constants
{
constexpr std::uint32_t kHalf         = 0x3F000000; // 0.5f  — comp threshold, shrink lambdas, heaviside
constexpr std::uint32_t kOne          = 0x3F800000; // 1.0f  — elu/celu alpha, polygamma n and scale, xielu alphas
constexpr std::uint32_t kTwo          = 0x40000000; // 2.0f  — fmod/remainder divisor, rpow base, unary_power exponent
constexpr std::uint32_t kQuarter      = 0x3E800000; // 0.25f — prelu slope
constexpr std::uint32_t kSeluScale    = 0x3F867D5F; // 1.0507009873554805f
constexpr std::uint32_t kSeluAlpha    = 0x3FD62D7D; // 1.6732632423543772f
constexpr std::uint32_t kBitwiseValue = 0x0F0F0F0F; // unary bitwise mask
constexpr std::uint32_t kShiftAmount  = 3;          // unary left_shift / right_shift
// hardtanh takes bf16 half-words rather than fp32 words: p0 = -min, p1 = -(max-min), p2 = max
// for bounds [-1, +1].
constexpr std::uint32_t kHardtanhP0 = 0x3F80; // 1.0 (bf16)
constexpr std::uint32_t kHardtanhP1 = 0xC000; // -2.0 (bf16)
constexpr std::uint32_t kHardtanhP2 = 0x3F80; // 1.0 (bf16)
} // namespace parity_constants

/**
 * @brief Whether OPERATION is one of the six unary comparisons against a fixed scalar.
 *
 * Distinct from @ref is_zero_comp_op: those compare against zero and read the SFPU format
 * at runtime, these compare against UNARY_COMP_THRESHOLD and are float-only.
 *
 * @param op The SFPU operation type to classify.
 */
inline constexpr bool is_unary_comp_op(SfpuType op)
{
    return op == SfpuType::unary_gt || op == SfpuType::unary_lt || op == SfpuType::unary_ge || op == SfpuType::unary_le || op == SfpuType::unary_ne ||
           op == SfpuType::unary_eq;
}

/**
 * @brief Whether OPERATION is one of the three unary bitwise-against-scalar modes.
 *
 * @param op The SFPU operation type to classify.
 */
inline constexpr bool is_unary_bitwise_op(SfpuType op)
{
    return op == SfpuType::bitwise_and || op == SfpuType::bitwise_or || op == SfpuType::bitwise_xor;
}

/**
 * @brief Whether OPERATION is one of the six comparison-to-zero modes.
 *
 * The comp family needs a runtime format switch (@ref call_zero_comp_operation_quasar)
 * to pick the integer-vs-float compare path, unlike the float-only unary ops, so the
 * dispatcher special-cases it.
 *
 * @param op The SFPU operation type to classify.
 */
inline constexpr bool is_zero_comp_op(SfpuType op)
{
    return op == SfpuType::equal_zero || op == SfpuType::not_equal_zero || op == SfpuType::less_than_zero || op == SfpuType::greater_than_zero ||
           op == SfpuType::less_than_equal_zero || op == SfpuType::greater_than_equal_zero;
}

/**
 * @brief Whether OPERATION is one of the trigonometry / inverse-hyperbolic ops.
 *
 * They share one init (@ref init_trigonometry, which programs ADDR_MOD_6 for the
 * auto-incrementing Dest store) since every trig body has the same load/compute/store shape.
 *
 * @param op The SFPU operation type to classify.
 */
inline constexpr bool is_trig_op(SfpuType op)
{
    return op == SfpuType::sine || op == SfpuType::cosine || op == SfpuType::acosh || op == SfpuType::asinh || op == SfpuType::atanh;
}

/**
 * @brief Run the per-operation init step for a Quasar unary SFPU op.
 *
 * @tparam OPERATION The SFPU operation type (compile-time `SfpuType` constant).
 * @note Pair with @ref call_unary_sfpu_operation_quasar for the calculate step.
 */
template <SfpuType OPERATION, bool is_fp32_dest_acc_en, bool APPROX = false>
void init_unary_sfpu_operation_quasar()
{
    if constexpr (OPERATION == SfpuType::gelu)
    {
        gelu_init<APPROX, is_fp32_dest_acc_en>();
    }
    else if constexpr (OPERATION == SfpuType::square)
    {
        init_square();
    }
    else if constexpr (OPERATION == SfpuType::rsqrt)
    {
        _init_rsqrt_<APPROX>();
    }
    else if constexpr (OPERATION == SfpuType::reciprocal)
    {
        _init_reciprocal_<APPROX>();
    }
    else if constexpr (is_zero_comp_op(OPERATION))
    {
        init_zero_comp();
    }
    else if constexpr (OPERATION == SfpuType::typecast)
    {
        init_typecast();
    }
    else if constexpr (is_trig_op(OPERATION))
    {
        init_trigonometry<OPERATION, is_fp32_dest_acc_en>();
    }
    // ── SFPU parity set ──────────────────────────────────────────────────────────────
    // Each branch is compiled only when its kernel header resolved above. Ops whose
    // Blackhole kernel has no init step are absent here by design, not by omission.
#ifdef QSR_HAS_ACTIVATIONS
    else if constexpr (OPERATION == SfpuType::hardsigmoid)
    {
        hardsigmoid_init<APPROX>();
    }
#endif
#ifdef QSR_HAS_BITWISE
    else if constexpr (OPERATION == SfpuType::bitwise_and)
    {
        bitwise_and_init();
    }
    else if constexpr (OPERATION == SfpuType::bitwise_or)
    {
        bitwise_or_init();
    }
    else if constexpr (OPERATION == SfpuType::bitwise_xor)
    {
        bitwise_xor_init();
    }
#endif
#ifdef QSR_HAS_BITWISE_NOT
    else if constexpr (OPERATION == SfpuType::bitwise_not)
    {
        bitwise_not_init();
    }
#endif
#ifdef QSR_HAS_CBRT
    else if constexpr (OPERATION == SfpuType::cbrt)
    {
        cube_root_init<APPROX>();
    }
#endif
#ifdef QSR_HAS_CELU
    else if constexpr (OPERATION == SfpuType::celu)
    {
        celu_init<APPROX>();
    }
#endif
#ifdef QSR_HAS_DIGAMMA
    else if constexpr (OPERATION == SfpuType::digamma)
    {
        digamma_init<APPROX>();
    }
#endif
#ifdef QSR_HAS_ELU
    else if constexpr (OPERATION == SfpuType::elu)
    {
        elu_init<APPROX>();
    }
#endif
#ifdef QSR_HAS_ERF
    else if constexpr (OPERATION == SfpuType::erf)
    {
        erf_init<APPROX>();
    }
#endif
#ifdef QSR_HAS_ERFC
    else if constexpr (OPERATION == SfpuType::erfc)
    {
        erfc_init<APPROX>();
    }
#endif
#ifdef QSR_HAS_ERFINV
    else if constexpr (OPERATION == SfpuType::erfinv)
    {
        erfinv_init<APPROX>();
    }
#endif
#ifdef QSR_HAS_EXP2
    else if constexpr (OPERATION == SfpuType::exp2)
    {
        exp2_init<APPROX, is_fp32_dest_acc_en>();
    }
#endif
#ifdef QSR_HAS_EXPM1
    else if constexpr (OPERATION == SfpuType::expm1)
    {
        expm1_init<APPROX, is_fp32_dest_acc_en>();
    }
#endif
#ifdef QSR_HAS_FMOD
    else if constexpr (OPERATION == SfpuType::fmod)
    {
        // calculate_fmod reads vConstFloatPrgm0/1, so the divisor is programmed here and
        // not passed at the call site.
        init_fmod<APPROX>(parity_constants::kTwo, parity_constants::kHalf);
    }
#endif
#ifdef QSR_HAS_HARDMISH
    else if constexpr (OPERATION == SfpuType::hardmish)
    {
        hardmish_init<APPROX>();
    }
#endif
#ifdef QSR_HAS_HARDSHRINK
    else if constexpr (OPERATION == SfpuType::hardshrink)
    {
        hardshrink_init<APPROX>();
    }
#endif
#ifdef QSR_HAS_HARDTANH
    else if constexpr (OPERATION == SfpuType::hardtanh)
    {
        hardtanh_init<APPROX>();
    }
#endif
#ifdef QSR_HAS_HEAVISIDE
    else if constexpr (OPERATION == SfpuType::heaviside)
    {
        heaviside_init<APPROX>();
    }
#endif
#ifdef QSR_HAS_I0
    else if constexpr (OPERATION == SfpuType::i0)
    {
        i0_init<APPROX>();
    }
#endif
#ifdef QSR_HAS_I1
    else if constexpr (OPERATION == SfpuType::i1)
    {
        i1_init<APPROX>();
    }
#endif
#ifdef QSR_HAS_LGAMMA
    else if constexpr (OPERATION == SfpuType::lgamma)
    {
        lgamma_stirling_init<APPROX, is_fp32_dest_acc_en>();
    }
#endif
#ifdef QSR_HAS_LOGICAL_NOT
    else if constexpr (OPERATION == SfpuType::logical_not_unary)
    {
        logical_not_unary_init<APPROX>();
    }
#endif
#ifdef QSR_HAS_POLYGAMMA
    else if constexpr (OPERATION == SfpuType::polygamma)
    {
        polygamma_init<APPROX, is_fp32_dest_acc_en>();
    }
#endif
#ifdef QSR_HAS_PRELU
    else if constexpr (OPERATION == SfpuType::prelu)
    {
        prelu_init<APPROX>();
    }
#endif
#ifdef QSR_HAS_RDIV
    else if constexpr (OPERATION == SfpuType::rdiv)
    {
        rdiv_init<APPROX>();
    }
#endif
#ifdef QSR_HAS_REMAINDER
    else if constexpr (OPERATION == SfpuType::remainder)
    {
        // As with fmod, the divisor lives in vConstFloatPrgm0/1 rather than in the call.
        init_remainder<APPROX>(parity_constants::kTwo, parity_constants::kHalf);
    }
#endif
#ifdef QSR_HAS_RPOW
    else if constexpr (OPERATION == SfpuType::rpow)
    {
        sfpu_binary_pow_init<APPROX>();
    }
#endif
#ifdef QSR_HAS_SELU
    else if constexpr (OPERATION == SfpuType::selu)
    {
        selu_init<APPROX>();
    }
#endif
#ifdef QSR_HAS_SIGN
    else if constexpr (OPERATION == SfpuType::sign)
    {
        sign_init<APPROX>();
    }
#endif
#ifdef QSR_HAS_SOFTSHRINK
    else if constexpr (OPERATION == SfpuType::softshrink)
    {
        softshrink_init<APPROX>();
    }
#endif
#ifdef QSR_HAS_SOFTSIGN
    else if constexpr (OPERATION == SfpuType::softsign)
    {
        init_softsign<APPROX>();
    }
#endif
#ifdef QSR_HAS_TANHSHRINK
    else if constexpr (OPERATION == SfpuType::tanhshrink)
    {
        tanhshrink_init<APPROX, is_fp32_dest_acc_en>();
    }
#endif
#ifdef QSR_HAS_UNARY_COMP
    else if constexpr (OPERATION == SfpuType::unary_gt)
    {
        unary_gt_init<APPROX>();
    }
    else if constexpr (OPERATION == SfpuType::unary_lt)
    {
        unary_lt_init<APPROX>();
    }
    else if constexpr (OPERATION == SfpuType::unary_ge)
    {
        unary_ge_init<APPROX>();
    }
    else if constexpr (OPERATION == SfpuType::unary_le)
    {
        unary_le_init<APPROX>();
    }
    else if constexpr (OPERATION == SfpuType::unary_ne)
    {
        unary_ne_init<APPROX>();
    }
    else if constexpr (OPERATION == SfpuType::unary_eq)
    {
        unary_eq_init<APPROX>();
    }
#endif
#ifdef QSR_HAS_UNARY_POWER
    else if constexpr (OPERATION == SfpuType::power)
    {
        sfpu_unary_pow_init<APPROX>();
    }
#endif
#ifdef QSR_HAS_UNARY_SHIFT
    else if constexpr (OPERATION == SfpuType::left_shift)
    {
        left_shift_init<APPROX>();
    }
    else if constexpr (OPERATION == SfpuType::right_shift)
    {
        right_shift_init<APPROX>();
    }
#endif
#ifdef QSR_HAS_XIELU
    else if constexpr (OPERATION == SfpuType::xielu)
    {
        xielu_init<APPROX>();
    }
#endif
#ifdef QSR_HAS_ALT_COMPLEX_ROTATE90
    else if constexpr (OPERATION == SfpuType::alt_complex_rotate90)
    {
        alt_complex_rotate90_init();
    }
#endif
#ifdef QSR_HAS_INT_SUM
    else if constexpr (OPERATION == SfpuType::sum_int_row || OPERATION == SfpuType::sum_int_col)
    {
        sum_int_init<APPROX>();
    }
#endif
#ifdef QSR_HAS_TILED_PROD
    else if constexpr (OPERATION == SfpuType::tiled_prod)
    {
        tiled_prod_init();
    }
#endif
}

/**
 * @brief Apply a comparison-to-zero SFPU op in-place on one Dest tile.
 *
 * Unlike the float-only unary ops, comp needs the SFPU math format at runtime to
 * pick the integer load/store width and the integer-vs-float compare path (see
 * `ckernel_sfpu_comp.h`). Int32/Int16/Int8/UInt16/UInt8 select their explicit
 * sfpmem width; all float widths share the width-agnostic `Float32` instantiation,
 * whose sfpi compare path resolves the actual width from the HW format config.
 *
 * @tparam OPERATION The comparison-to-zero `SfpuType` (compile-time constant).
 * @tparam DST_SYNC Destination synchronization mode used for bounds checking.
 * @tparam is_fp32_dest_acc_en Whether Dest is in FP32 mode.
 * @tparam ITERATIONS Number of SFPU loop iterations.
 * @param dst_index Destination tile index operated on (already offset by DST_INDEX).
 * @param sfpu_format SFPU math format selecting the sfpmem mode / result encoding.
 * @note Must be preceded by @ref init_unary_sfpu_operation_quasar for the same op.
 */
template <SfpuType OPERATION, DstSync DST_SYNC, bool is_fp32_dest_acc_en, int ITERATIONS = SFPU_ITERATIONS>
void call_zero_comp_operation_quasar(std::uint32_t dst_index, DataFormat sfpu_format)
{
    static_assert(is_zero_comp_op(OPERATION), "call_zero_comp_operation_quasar: OPERATION must be a comparison-to-zero SfpuType");

    switch (sfpu_format)
    {
        case DataFormat::Int32:
            SFPU_UNARY_CALL(DST_SYNC, is_fp32_dest_acc_en, calculate_zero_comp, (false, DataFormat::Int32, OPERATION, ITERATIONS), dst_index, VectorMode::RC);
            break;
        case DataFormat::Int16:
            SFPU_UNARY_CALL(DST_SYNC, is_fp32_dest_acc_en, calculate_zero_comp, (false, DataFormat::Int16, OPERATION, ITERATIONS), dst_index, VectorMode::RC);
            break;
        case DataFormat::Int8:
        {
            constexpr DataFormat sfpu_fmt = is_fp32_dest_acc_en ? DataFormat::Int32 : DataFormat::Int8;
            SFPU_UNARY_CALL(DST_SYNC, is_fp32_dest_acc_en, calculate_zero_comp, (false, sfpu_fmt, OPERATION, ITERATIONS), dst_index, VectorMode::RC);
            break;
        }
        case DataFormat::UInt16:
            SFPU_UNARY_CALL(DST_SYNC, is_fp32_dest_acc_en, calculate_zero_comp, (false, DataFormat::UInt16, OPERATION, ITERATIONS), dst_index, VectorMode::RC);
            break;
        case DataFormat::UInt8:
        {
            constexpr DataFormat sfpu_fmt = is_fp32_dest_acc_en ? DataFormat::Int32 : DataFormat::UInt8;
            SFPU_UNARY_CALL(DST_SYNC, is_fp32_dest_acc_en, calculate_zero_comp, (false, sfpu_fmt, OPERATION, ITERATIONS), dst_index, VectorMode::RC);
            break;
        }
        case DataFormat::Float16:
        case DataFormat::Float16_b:
        case DataFormat::Float32:
            // Float widths share the width-agnostic Float32 path: its sfpmem::DEFAULT access mode
            // resolves the actual width from ALU_FORMAT_SPEC_REG / ACC_CTRL.
            SFPU_UNARY_CALL(DST_SYNC, is_fp32_dest_acc_en, calculate_zero_comp, (false, DataFormat::Float32, OPERATION, ITERATIONS), dst_index, VectorMode::RC);
            break;
        default:
            LLK_ASSERT(false, "Unsupported Quasar comp-to-zero SFPU format");
            break;
    }
}

#ifdef QSR_HAS_UNARY_COMP
/**
 * @brief Apply a compare-against-scalar SFPU op in-place on one Dest tile.
 *
 * The six modes are separate functors rather than one op-templated kernel, so they need a
 * dispatch of their own; @ref call_unary_sfpu_operation_quasar forwards to it. All six
 * compare against the same fixed threshold (0.5f), which the goldens and the
 * sfpu_domains edge probes also read -- see @ref parity_constants.
 *
 * @tparam OPERATION One of the six unary-comparison `SfpuType` values.
 * @tparam DST_SYNC Destination synchronization mode used for bounds checking.
 * @tparam is_fp32_dest_acc_en Whether Dest is in FP32 mode.
 * @tparam APPROX Accepted for ABI parity; none of the six branches on it.
 * @tparam ITERATIONS Number of SFPU loop iterations.
 * @param dst_index Destination tile index operated on (already offset by DST_INDEX).
 * @note Must be preceded by @ref init_unary_sfpu_operation_quasar for the same op.
 */
template <SfpuType OPERATION, DstSync DST_SYNC, bool is_fp32_dest_acc_en, bool APPROX = false, int ITERATIONS = SFPU_ITERATIONS>
void call_unary_comp_operation_quasar(std::uint32_t dst_index)
{
    static_assert(is_unary_comp_op(OPERATION), "call_unary_comp_operation_quasar: OPERATION must be a unary-comparison SfpuType");

    if constexpr (OPERATION == SfpuType::unary_gt)
    {
        SFPU_UNARY_CALL(DST_SYNC, is_fp32_dest_acc_en, calculate_unary_gt, (APPROX, ITERATIONS), dst_index, VectorMode::RC, parity_constants::kHalf);
    }
    else if constexpr (OPERATION == SfpuType::unary_lt)
    {
        SFPU_UNARY_CALL(DST_SYNC, is_fp32_dest_acc_en, calculate_unary_lt, (APPROX, ITERATIONS), dst_index, VectorMode::RC, parity_constants::kHalf);
    }
    else if constexpr (OPERATION == SfpuType::unary_ge)
    {
        SFPU_UNARY_CALL(DST_SYNC, is_fp32_dest_acc_en, calculate_unary_ge, (APPROX, ITERATIONS), dst_index, VectorMode::RC, parity_constants::kHalf);
    }
    else if constexpr (OPERATION == SfpuType::unary_le)
    {
        SFPU_UNARY_CALL(DST_SYNC, is_fp32_dest_acc_en, calculate_unary_le, (APPROX, ITERATIONS), dst_index, VectorMode::RC, parity_constants::kHalf);
    }
    else if constexpr (OPERATION == SfpuType::unary_ne)
    {
        SFPU_UNARY_CALL(DST_SYNC, is_fp32_dest_acc_en, calculate_unary_ne, (APPROX, ITERATIONS), dst_index, VectorMode::RC, parity_constants::kHalf);
    }
    else
    {
        SFPU_UNARY_CALL(DST_SYNC, is_fp32_dest_acc_en, calculate_unary_eq, (APPROX, ITERATIONS), dst_index, VectorMode::RC, parity_constants::kHalf);
    }
}
#endif

/**
 * @brief Apply a Quasar unary SFPU op in-place on one Dest tile.
 *
 * @tparam OPERATION The SFPU operation type (compile-time `SfpuType` constant).
 * @tparam DST_SYNC Destination synchronization mode used for bounds checking.
 * @tparam is_fp32_dest_acc_en Whether Dest is in FP32 mode.
 * @tparam APPROX Whether operations with approximate and accurate paths use the approximate path.
 * @tparam ITERATIONS Number of SFPU loop iterations.
 * @tparam TYPECAST_IN_FORMAT Source format for the typecast op (default Float32).
 * @tparam TYPECAST_OUT_FORMAT Destination format for the typecast op (default Float16_b).
 * @param dst_index Destination tile index operated on (already offset by DST_INDEX).
 * @param sfpu_format SFPU math format; only the comp family reads it (see
 *        @ref call_zero_comp_operation_quasar), float-only ops ignore it.
 * @note Must be preceded by @ref init_unary_sfpu_operation_quasar for the same op.
 */
template <
    SfpuType OPERATION,
    DstSync DST_SYNC,
    bool is_fp32_dest_acc_en,
    bool APPROX                    = false,
    int ITERATIONS                 = SFPU_ITERATIONS,
    DataFormat TYPECAST_IN_FORMAT  = DataFormat::Float32,
    DataFormat TYPECAST_OUT_FORMAT = DataFormat::Float16_b>
void call_unary_sfpu_operation_quasar(std::uint32_t dst_index, DataFormat sfpu_format = DataFormat::Float32)
{
    if constexpr (OPERATION == SfpuType::abs)
    {
        SFPU_UNARY_CALL(DST_SYNC, is_fp32_dest_acc_en, _calculate_abs_, (ITERATIONS), dst_index, VectorMode::RC);
    }
    else if constexpr (OPERATION == SfpuType::exponential)
    {
        SFPU_UNARY_CALL(
            DST_SYNC,
            is_fp32_dest_acc_en,
            calculate_exponential,
            (APPROX, is_fp32_dest_acc_en, false, ITERATIONS),
            dst_index,
            VectorMode::RC,
            p_sfpu::kCONST_1_FP16B);
    }
    else if constexpr (OPERATION == SfpuType::gelu)
    {
        SFPU_UNARY_CALL(DST_SYNC, is_fp32_dest_acc_en, calculate_gelu, (APPROX, is_fp32_dest_acc_en, ITERATIONS), dst_index, VectorMode::RC);
    }
    else if constexpr (OPERATION == SfpuType::relu)
    {
        SFPU_UNARY_CALL(DST_SYNC, is_fp32_dest_acc_en, _calculate_relu_, (ITERATIONS), dst_index, VectorMode::RC);
    }
    else if constexpr (OPERATION == SfpuType::reciprocal)
    {
        SFPU_UNARY_CALL(DST_SYNC, is_fp32_dest_acc_en, calculate_reciprocal, (APPROX, is_fp32_dest_acc_en, ITERATIONS), dst_index, VectorMode::RC);
    }
    else if constexpr (OPERATION == SfpuType::sqrt)
    {
        SFPU_UNARY_CALL(DST_SYNC, is_fp32_dest_acc_en, _calculate_sqrt_, (true /* APPROX */, ITERATIONS), dst_index, VectorMode::RC);
    }
    else if constexpr (OPERATION == SfpuType::tanh)
    {
        SFPU_UNARY_CALL(DST_SYNC, is_fp32_dest_acc_en, calculate_tanh, (ITERATIONS), dst_index, VectorMode::RC);
    }
    else if constexpr (OPERATION == SfpuType::sigmoid)
    {
        SFPU_UNARY_CALL(DST_SYNC, is_fp32_dest_acc_en, _calculate_sigmoid_, (ITERATIONS), dst_index, VectorMode::RC);
    }
    else if constexpr (OPERATION == SfpuType::silu)
    {
        SFPU_UNARY_CALL(DST_SYNC, is_fp32_dest_acc_en, _calculate_silu_, (ITERATIONS), dst_index, VectorMode::RC);
    }
    else if constexpr (OPERATION == SfpuType::rsqrt)
    {
        SFPU_UNARY_CALL(DST_SYNC, is_fp32_dest_acc_en, calculate_rsqrt, (APPROX, ITERATIONS, is_fp32_dest_acc_en), dst_index, VectorMode::RC);
    }
    else if constexpr (OPERATION == SfpuType::square)
    {
        SFPU_UNARY_CALL(DST_SYNC, is_fp32_dest_acc_en, calculate_square, (ITERATIONS), dst_index, VectorMode::RC);
    }
    else if constexpr (is_trig_op(OPERATION))
    {
        // One op-templated kernel serves sine/cosine/acosh/asinh/atanh; OPERATION picks the branch
        // at compile time. APPROXIMATION_MODE=false selects the full-polynomial (accurate) path;
        // VectorMode::RC (the params default) runs the functor once per face.
        _llk_math_eltwise_unary_sfpu_params_(calculate_trigonometry<OPERATION, false /* APPROX */, is_fp32_dest_acc_en, ITERATIONS>, dst_index);
    }
    else if constexpr (OPERATION == SfpuType::negative)
    {
        SFPU_UNARY_CALL(DST_SYNC, is_fp32_dest_acc_en, _calculate_negative_, (false, ITERATIONS), dst_index, VectorMode::RC);
    }
    else if constexpr (OPERATION == SfpuType::softplus)
    {
        // Softplus params beta / (1/beta) / threshold as fp32 bit patterns, matching the
        // UnarySFPUGolden._softplus reference defaults (beta = 1.0, threshold = 20.0).
        SFPU_UNARY_CALL(
            DST_SYNC,
            is_fp32_dest_acc_en,
            calculate_softplus,
            (false, is_fp32_dest_acc_en, ITERATIONS),
            dst_index,
            VectorMode::RC,
            static_cast<std::uint32_t>(0x3F800000),  // beta = 1.0 (fp32)
            static_cast<std::uint32_t>(0x3F800000),  // 1/beta = 1.0 (fp32)
            static_cast<std::uint32_t>(0x41A00000)); // threshold = 20.0 (fp32)
    }
    else if constexpr (OPERATION == SfpuType::clamp)
    {
        // Clamp bounds fixed to [-1.0, +1.0] as fp32 bit patterns (matching the UnarySFPUGolden._clamp
        // reference). Extra args are forwarded to the per-face functor call.
        SFPU_UNARY_CALL(
            DST_SYNC,
            is_fp32_dest_acc_en,
            calculate_clamp,
            (false, ITERATIONS),
            dst_index,
            VectorMode::RC,
            static_cast<std::uint32_t>(0xBF800000),  // min = -1.0 (fp32)
            static_cast<std::uint32_t>(0x3F800000)); // max = +1.0 (fp32)
    }
    else if constexpr (is_zero_comp_op(OPERATION))
    {
        call_zero_comp_operation_quasar<OPERATION, DST_SYNC, is_fp32_dest_acc_en, ITERATIONS>(dst_index, sfpu_format);
    }
    else if constexpr (OPERATION == SfpuType::typecast)
    {
        SFPU_UNARY_CALL(DST_SYNC, is_fp32_dest_acc_en, calculate_typecast, (TYPECAST_IN_FORMAT, TYPECAST_OUT_FORMAT, ITERATIONS), dst_index, VectorMode::RC);
    }
    // ── SFPU parity set ──────────────────────────────────────────────────────────────
    // Signatures mirror the Blackhole dispatcher in sfpu_operations.h, including the baked
    // scalar arguments (see parity_constants). If a Quasar port lands with a different
    // signature, this is the only place that needs adjusting.
#ifdef QSR_HAS_ACTIVATIONS
    else if constexpr (OPERATION == SfpuType::hardsigmoid)
    {
        SFPU_UNARY_CALL(
            DST_SYNC, is_fp32_dest_acc_en, calculate_activation, (APPROX, ckernel::ActivationType::Hardsigmoid, ITERATIONS), dst_index, VectorMode::RC);
    }
#endif
#ifdef QSR_HAS_ADD1
    else if constexpr (OPERATION == SfpuType::add1)
    {
        SFPU_UNARY_CALL(DST_SYNC, is_fp32_dest_acc_en, calculate_add1, (APPROX, ITERATIONS), dst_index, VectorMode::RC);
    }
#endif
#ifdef QSR_HAS_BITWISE
    else if constexpr (is_unary_bitwise_op(OPERATION))
    {
        constexpr UnaryBitwiseOp kOp = (OPERATION == SfpuType::bitwise_and)  ? UnaryBitwiseOp::AND
                                       : (OPERATION == SfpuType::bitwise_or) ? UnaryBitwiseOp::OR
                                                                             : UnaryBitwiseOp::XOR;
        SFPU_UNARY_CALL(
            DST_SYNC,
            is_fp32_dest_acc_en,
            calculate_sfpu_unary_bitwise,
            (APPROX, kOp, DataFormat::Int32, ITERATIONS),
            dst_index,
            VectorMode::RC,
            parity_constants::kBitwiseValue);
    }
#endif
#ifdef QSR_HAS_BITWISE_NOT
    else if constexpr (OPERATION == SfpuType::bitwise_not)
    {
        SFPU_UNARY_CALL(DST_SYNC, is_fp32_dest_acc_en, calculate_bitwise_not, (APPROX, ITERATIONS), dst_index, VectorMode::RC);
    }
#endif
#ifdef QSR_HAS_CAST_FP32_TO_FP16A
    else if constexpr (OPERATION == SfpuType::cast_fp32_to_fp16a)
    {
        SFPU_UNARY_CALL(DST_SYNC, is_fp32_dest_acc_en, cast_fp32_to_fp16a, (APPROX, ITERATIONS), dst_index, VectorMode::RC);
    }
#endif
#ifdef QSR_HAS_CBRT
    else if constexpr (OPERATION == SfpuType::cbrt)
    {
        SFPU_UNARY_CALL(DST_SYNC, is_fp32_dest_acc_en, calculate_cube_root, (APPROX, is_fp32_dest_acc_en, ITERATIONS), dst_index, VectorMode::RC);
    }
#endif
#ifdef QSR_HAS_CELU
    else if constexpr (OPERATION == SfpuType::celu)
    {
        SFPU_UNARY_CALL(
            DST_SYNC,
            is_fp32_dest_acc_en,
            calculate_celu,
            (APPROX, is_fp32_dest_acc_en, ITERATIONS),
            dst_index,
            VectorMode::RC,
            parity_constants::kOne,  // alpha = 1.0f
            parity_constants::kOne); // 1/alpha = 1.0f
    }
#endif
#ifdef QSR_HAS_DIGAMMA
    else if constexpr (OPERATION == SfpuType::digamma)
    {
        SFPU_UNARY_CALL(DST_SYNC, is_fp32_dest_acc_en, calculate_digamma, (APPROX, ITERATIONS), dst_index, VectorMode::RC);
    }
#endif
#ifdef QSR_HAS_ELU
    else if constexpr (OPERATION == SfpuType::elu)
    {
        SFPU_UNARY_CALL(
            DST_SYNC, is_fp32_dest_acc_en, calculate_elu, (APPROX, is_fp32_dest_acc_en, ITERATIONS), dst_index, VectorMode::RC, parity_constants::kOne);
    }
#endif
#ifdef QSR_HAS_ERF
    else if constexpr (OPERATION == SfpuType::erf)
    {
        SFPU_UNARY_CALL(DST_SYNC, is_fp32_dest_acc_en, calculate_erf, (APPROX, ITERATIONS), dst_index, VectorMode::RC);
    }
#endif
#ifdef QSR_HAS_ERFC
    else if constexpr (OPERATION == SfpuType::erfc)
    {
        // calculate_erfc takes ITERATIONS only; APPROX reaches it through erfc_init.
        SFPU_UNARY_CALL(DST_SYNC, is_fp32_dest_acc_en, calculate_erfc, (ITERATIONS), dst_index, VectorMode::RC);
    }
#endif
#ifdef QSR_HAS_ERFINV
    else if constexpr (OPERATION == SfpuType::erfinv)
    {
        SFPU_UNARY_CALL(DST_SYNC, is_fp32_dest_acc_en, calculate_erfinv, (APPROX), dst_index, VectorMode::RC);
    }
#endif
#ifdef QSR_HAS_EXP2
    else if constexpr (OPERATION == SfpuType::exp2)
    {
        SFPU_UNARY_CALL(DST_SYNC, is_fp32_dest_acc_en, calculate_exp2, (APPROX, is_fp32_dest_acc_en, ITERATIONS), dst_index, VectorMode::RC);
    }
#endif
#ifdef QSR_HAS_EXPM1
    else if constexpr (OPERATION == SfpuType::expm1)
    {
        SFPU_UNARY_CALL(DST_SYNC, is_fp32_dest_acc_en, calculate_expm1, (APPROX, is_fp32_dest_acc_en, ITERATIONS), dst_index, VectorMode::RC);
    }
#endif
#ifdef QSR_HAS_FMOD
    else if constexpr (OPERATION == SfpuType::fmod)
    {
        // Divisor comes from vConstFloatPrgm0/1, programmed by init_fmod.
        SFPU_UNARY_CALL(DST_SYNC, is_fp32_dest_acc_en, calculate_fmod, (APPROX, ITERATIONS), dst_index, VectorMode::RC);
    }
#endif
#ifdef QSR_HAS_HARDMISH
    else if constexpr (OPERATION == SfpuType::hardmish)
    {
        SFPU_UNARY_CALL(DST_SYNC, is_fp32_dest_acc_en, hardmish, (APPROX, ITERATIONS), dst_index, VectorMode::RC);
    }
#endif
#ifdef QSR_HAS_HARDSHRINK
    else if constexpr (OPERATION == SfpuType::hardshrink)
    {
        SFPU_UNARY_CALL(DST_SYNC, is_fp32_dest_acc_en, calculate_hardshrink, (APPROX, ITERATIONS), dst_index, VectorMode::RC, parity_constants::kHalf);
    }
#endif
#ifdef QSR_HAS_HARDTANH
    else if constexpr (OPERATION == SfpuType::hardtanh)
    {
        // Bounds [-1, +1] encoded as bf16 half-words, matching the Blackhole dispatch.
        SFPU_UNARY_CALL(
            DST_SYNC,
            is_fp32_dest_acc_en,
            calculate_hardtanh,
            (APPROX, ITERATIONS),
            dst_index,
            VectorMode::RC,
            parity_constants::kHardtanhP0,
            parity_constants::kHardtanhP1,
            parity_constants::kHardtanhP2);
    }
#endif
#ifdef QSR_HAS_HEAVISIDE
    else if constexpr (OPERATION == SfpuType::heaviside)
    {
        SFPU_UNARY_CALL(DST_SYNC, is_fp32_dest_acc_en, calculate_heaviside, (APPROX, ITERATIONS), dst_index, VectorMode::RC, parity_constants::kHalf);
    }
#endif
#ifdef QSR_HAS_I0
    else if constexpr (OPERATION == SfpuType::i0)
    {
        SFPU_UNARY_CALL(DST_SYNC, is_fp32_dest_acc_en, calculate_i0, (APPROX, ITERATIONS), dst_index, VectorMode::RC);
    }
#endif
#ifdef QSR_HAS_I1
    else if constexpr (OPERATION == SfpuType::i1)
    {
        SFPU_UNARY_CALL(DST_SYNC, is_fp32_dest_acc_en, calculate_i1, (APPROX, ITERATIONS), dst_index, VectorMode::RC);
    }
#endif
#ifdef QSR_HAS_IDENTITY
    else if constexpr (OPERATION == SfpuType::identity)
    {
        SFPU_UNARY_CALL(DST_SYNC, is_fp32_dest_acc_en, calculate_identity, (APPROX, ITERATIONS), dst_index, VectorMode::RC);
    }
#endif
#ifdef QSR_HAS_LGAMMA
    else if constexpr (OPERATION == SfpuType::lgamma)
    {
        SFPU_UNARY_CALL(DST_SYNC, is_fp32_dest_acc_en, calculate_lgamma_stirling, (APPROX, is_fp32_dest_acc_en, ITERATIONS), dst_index, VectorMode::RC);
    }
#endif
#ifdef QSR_HAS_LOGICAL_NOT
    else if constexpr (OPERATION == SfpuType::logical_not_unary)
    {
        SFPU_UNARY_CALL(
            DST_SYNC, is_fp32_dest_acc_en, calculate_logical_not, (APPROX, ckernel::InstrModLoadStore::LO16, ITERATIONS), dst_index, VectorMode::RC);
    }
#endif
#ifdef QSR_HAS_POLYGAMMA
    else if constexpr (OPERATION == SfpuType::polygamma)
    {
        // order n = 1 (trigamma); scale = (-1)^(n+1) * n! = 1.0f.
        SFPU_UNARY_CALL(
            DST_SYNC,
            is_fp32_dest_acc_en,
            calculate_polygamma,
            (APPROX, is_fp32_dest_acc_en, ITERATIONS),
            dst_index,
            VectorMode::RC,
            parity_constants::kOne,
            parity_constants::kOne);
    }
#endif
#ifdef QSR_HAS_PRELU
    else if constexpr (OPERATION == SfpuType::prelu)
    {
        SFPU_UNARY_CALL(DST_SYNC, is_fp32_dest_acc_en, calculate_prelu, (APPROX, ITERATIONS), dst_index, VectorMode::RC, parity_constants::kQuarter);
    }
#endif
#ifdef QSR_HAS_RDIV
    else if constexpr (OPERATION == SfpuType::rdiv)
    {
        SFPU_UNARY_CALL(
            DST_SYNC,
            is_fp32_dest_acc_en,
            calculate_rdiv,
            (APPROX, is_fp32_dest_acc_en, ckernel::RoundingMode::None, ITERATIONS),
            dst_index,
            VectorMode::RC,
            parity_constants::kTwo);
    }
#endif
#ifdef QSR_HAS_REMAINDER
    else if constexpr (OPERATION == SfpuType::remainder)
    {
        // Divisor comes from vConstFloatPrgm0/1, programmed by init_remainder.
        SFPU_UNARY_CALL(DST_SYNC, is_fp32_dest_acc_en, calculate_remainder, (APPROX, ITERATIONS), dst_index, VectorMode::RC);
    }
#endif
#ifdef QSR_HAS_RPOW
    else if constexpr (OPERATION == SfpuType::rpow)
    {
        SFPU_UNARY_CALL(
            DST_SYNC, is_fp32_dest_acc_en, calculate_rpow, (APPROX, ITERATIONS, is_fp32_dest_acc_en), dst_index, VectorMode::RC, parity_constants::kTwo);
    }
#endif
#ifdef QSR_HAS_SELU
    else if constexpr (OPERATION == SfpuType::selu)
    {
        SFPU_UNARY_CALL(
            DST_SYNC,
            is_fp32_dest_acc_en,
            calculate_selu,
            (APPROX, is_fp32_dest_acc_en, ITERATIONS),
            dst_index,
            VectorMode::RC,
            parity_constants::kSeluScale,
            parity_constants::kSeluAlpha);
    }
#endif
#ifdef QSR_HAS_SIGN
    else if constexpr (OPERATION == SfpuType::sign)
    {
        SFPU_UNARY_CALL(DST_SYNC, is_fp32_dest_acc_en, calculate_sign, (APPROX, ITERATIONS), dst_index, VectorMode::RC, 0u /* exponent_size_8 */);
    }
#endif
#ifdef QSR_HAS_SOFTSHRINK
    else if constexpr (OPERATION == SfpuType::softshrink)
    {
        SFPU_UNARY_CALL(DST_SYNC, is_fp32_dest_acc_en, calculate_softshrink, (APPROX, ITERATIONS), dst_index, VectorMode::RC, parity_constants::kHalf);
    }
#endif
#ifdef QSR_HAS_SOFTSIGN
    else if constexpr (OPERATION == SfpuType::softsign)
    {
        SFPU_UNARY_CALL(DST_SYNC, is_fp32_dest_acc_en, calculate_softsign, (APPROX, ITERATIONS), dst_index, VectorMode::RC);
    }
#endif
#ifdef QSR_HAS_TANHSHRINK
    else if constexpr (OPERATION == SfpuType::tanhshrink)
    {
        SFPU_UNARY_CALL(DST_SYNC, is_fp32_dest_acc_en, calculate_tanhshrink, (is_fp32_dest_acc_en, ITERATIONS), dst_index, VectorMode::RC);
    }
#endif
#ifdef QSR_HAS_UNARY_COMP
    else if constexpr (is_unary_comp_op(OPERATION))
    {
        call_unary_comp_operation_quasar<OPERATION, DST_SYNC, is_fp32_dest_acc_en, APPROX, ITERATIONS>(dst_index);
    }
#endif
#ifdef QSR_HAS_UNARY_POWER
    else if constexpr (OPERATION == SfpuType::power)
    {
        SFPU_UNARY_CALL(
            DST_SYNC, is_fp32_dest_acc_en, calculate_unary_power, (APPROX, is_fp32_dest_acc_en, ITERATIONS), dst_index, VectorMode::RC, parity_constants::kTwo);
    }
#endif
#ifdef QSR_HAS_UNARY_SHIFT
    else if constexpr (OPERATION == SfpuType::left_shift)
    {
        SFPU_UNARY_CALL(
            DST_SYNC,
            is_fp32_dest_acc_en,
            calculate_left_shift,
            (APPROX, DataFormat::Int32, ITERATIONS),
            dst_index,
            VectorMode::RC,
            parity_constants::kShiftAmount);
    }
    else if constexpr (OPERATION == SfpuType::right_shift)
    {
        SFPU_UNARY_CALL(
            DST_SYNC,
            is_fp32_dest_acc_en,
            calculate_right_shift,
            (APPROX, DataFormat::Int32, ITERATIONS),
            dst_index,
            VectorMode::RC,
            parity_constants::kShiftAmount);
    }
#endif
#ifdef QSR_HAS_XIELU
    else if constexpr (OPERATION == SfpuType::xielu)
    {
        SFPU_UNARY_CALL(
            DST_SYNC,
            is_fp32_dest_acc_en,
            calculate_xielu,
            (APPROX, is_fp32_dest_acc_en, ITERATIONS),
            dst_index,
            VectorMode::RC,
            parity_constants::kOne,  // alpha_p = 1.0f
            parity_constants::kOne); // alpha_n = 1.0f
    }
#endif
    else
    {
        static_assert(unhandled_op<OPERATION>, "call_unary_sfpu_operation_quasar: unhandled Quasar unary SFPU operation");
    }
}

constexpr bool quasar_binary_op_is_max_min(ckernel::BinaryOp op)
{
    return op == ckernel::BinaryOp::MAX || op == ckernel::BinaryOp::MIN;
}

constexpr bool quasar_binary_op_is_quant(ckernel::BinaryOp op)
{
    return op == ckernel::BinaryOp::QUANT || op == ckernel::BinaryOp::REQUANT || op == ckernel::BinaryOp::DEQUANT;
}

// Map the shared BinaryOp enum onto the quant kernel's op-templated QuantVariant.
template <ckernel::BinaryOp OP>
constexpr ckernel::sfpu::QuantVariant quant_variant_of()
{
    if constexpr (OP == ckernel::BinaryOp::QUANT)
    {
        return ckernel::sfpu::QuantVariant::Quant;
    }
    else if constexpr (OP == ckernel::BinaryOp::REQUANT)
    {
        return ckernel::sfpu::QuantVariant::Requant;
    }
    else if constexpr (OP == ckernel::BinaryOp::DEQUANT)
    {
        return ckernel::sfpu::QuantVariant::Dequant;
    }
    else
    {
        static_assert(unhandled_op<OP>, "quant_variant_of: unhandled quant BinaryOp");
    }
}

/**
 * @brief Run the per-operation init step for a Quasar binary SFPU op.
 *
 * @tparam OP The binary op (compile-time `ckernel::BinaryOp` constant).
 * @tparam SIGN_MAGNITUDE_FORMAT Quant family only: if true, treat int32 Dest as SMAG32
 *         and skip the sign-magnitude<->2's-complement casts. Must match the calculate step.
 * @param zero_point fp32 bit-pattern of the zero-point loaded once by the quant
 *        family init (DEQUANT expects the bits of -zero_point); ignored by the
 *        other ops, which have no runtime init argument.
 * @note Pair with @ref call_binary_sfpu_operation_quasar for the calculate step.
 */
template <ckernel::BinaryOp OP, bool SIGN_MAGNITUDE_FORMAT = false>
void init_binary_sfpu_operation_quasar([[maybe_unused]] std::uint32_t zero_point = 0)
{
    if constexpr (OP == BinaryOp::MUL)
    {
        sfpu_binary_init<false /*APPROX*/, BinaryOp::MUL>(); // no-op for MUL; harmless on the int path
    }
    else if constexpr (OP == BinaryOp::DIV)
    {
        sfpu_binary_init<false /*APPROX*/, BinaryOp::DIV>();
    }
    else if constexpr (quasar_binary_op_is_max_min(OP))
    {
        _init_binary_max_min_();
    }
    else if constexpr (quasar_binary_op_is_quant(OP))
    {
        // One op-templated quant kernel; DEQUANT's caller passes bits of -zero_point.
        quant_family_init<quant_variant_of<OP>(), SIGN_MAGNITUDE_FORMAT>(zero_point);
    }
    // ── SFPU parity set ──────────────────────────────────────────────────────────────
#ifdef QSR_HAS_ATAN2
    else if constexpr (OP == BinaryOp::ATAN2)
    {
        calculate_sfpu_atan2_init<false /*APPROX*/>();
    }
#endif
#ifdef QSR_HAS_BINARY_FMOD
    else if constexpr (OP == BinaryOp::FMOD)
    {
        fmod_binary_init<false /*APPROX*/>();
    }
    else if constexpr (OP == BinaryOp::FMOD_INT32)
    {
        fmod_int32_init<false /*APPROX*/>();
    }
#endif
#ifdef QSR_HAS_BINARY_POW
    else if constexpr (OP == BinaryOp::POW)
    {
        // POW rides the shared calculate_sfpu_binary dispatch (as on Blackhole); the
        // gate is on the pow kernel header because that is what the parity set tracks.
        sfpu_binary_init<false /*APPROX*/, BinaryOp::POW>();
    }
#endif
#ifdef QSR_HAS_BINARY_REMAINDER
    else if constexpr (OP == BinaryOp::REMAINDER)
    {
        remainder_binary_init<false /*APPROX*/, false /*legacy_compat*/>();
    }
    else if constexpr (OP == BinaryOp::REMAINDER_INT32)
    {
        remainder_int32_init<false /*APPROX*/>();
    }
    else if constexpr (OP == BinaryOp::REMAINDER_UINT32)
    {
        remainder_uint32_init<false /*APPROX*/>();
    }
#endif
#ifdef QSR_HAS_DIV_INT32
    else if constexpr (OP == BinaryOp::DIV_INT32)
    {
        // Truncating int32 division writes an int32 quotient, so it needs the
        // reciprocal-polynomial constants from div_trunc_init rather than div_init's.
        div_trunc_init<false /*APPROX*/>();
    }
#endif
#ifdef QSR_HAS_DIV_INT32_FLOOR
    else if constexpr (OP == BinaryOp::DIV_INT32_FLOOR)
    {
        div_floor_init<false /*APPROX*/>();
    }
#endif
#ifdef QSR_HAS_ISCLOSE
    else if constexpr (OP == BinaryOp::ISCLOSE)
    {
        isclose_init<false /*APPROX*/>();
    }
#endif
#ifdef QSR_HAS_LOGSIGMOID
    else if constexpr (OP == BinaryOp::LOGSIGMOID)
    {
        logsigmoid_init<false /*APPROX*/>();
    }
#endif
#ifdef QSR_HAS_MASK
    else if constexpr (OP == BinaryOp::MASK)
    {
        mask_init<false /*APPROX*/>();
    }
#endif
    // ADD / SUB / GT / LT / LE / GE are stateless — no init. Among the parity ops, the
    // bitwise family and RSUB_INT32 are likewise stateless.
}

/**
 * @brief Apply a Quasar binary SFPU op over two Dest operands into a result tile.
 *
 * @tparam OP The binary op (compile-time `ckernel::BinaryOp` constant).
 * @tparam DST_SYNC Destination synchronization mode used for bounds checking.
 * @tparam is_fp32_dest_acc_en Whether Dest is in FP32 mode.
 * @tparam dst_rounding_mode Controls bf16 narrowing for ADD/SUB results. Default truncates;
 *         NearestEven applies software RNE before the store. Ignored for MUL (no narrowing)
 *         and DIV (always rounds RNE regardless). No-op when is_fp32_dest_acc_en is true.
 * @tparam ITERATIONS Number of SFPU loop iterations.
 * @tparam SIGN_MAGNITUDE_FORMAT Quant family only: if true, treat int32 Dest as SMAG32
 *         and skip the sign-magnitude<->2's-complement casts. Must match the init step.
 * @param src0_tile,src1_tile,dst_tile Operand / result tile indices.
 * @param math_format Dest math format (Int32 vs float path for MUL and max/min).
 * @note Must be preceded by @ref init_binary_sfpu_operation_quasar for the same op.
 */
template <
    ckernel::BinaryOp OP,
    DstSync DST_SYNC,
    bool is_fp32_dest_acc_en,
    ckernel::DstRoundingMode dst_rounding_mode = ckernel::DstRoundingMode::Default,
    int ITERATIONS                             = SFPU_ITERATIONS,
    bool SIGN_MAGNITUDE_FORMAT                 = false>
void call_binary_sfpu_operation_quasar(std::uint32_t src0_tile, std::uint32_t src1_tile, std::uint32_t dst_tile, [[maybe_unused]] DataFormat math_format)
{
    if constexpr (OP == BinaryOp::ADD)
    {
        if (math_format == DataFormat::Int32)
        {
            SFPU_BINARY_CALL(
                DST_SYNC, is_fp32_dest_acc_en, _add_int_, (false, ITERATIONS, DataFormat::Int32, 0, false), src0_tile, src1_tile, dst_tile, VectorMode::RC);
        }
        else
        {
            SFPU_BINARY_CALL(
                DST_SYNC,
                is_fp32_dest_acc_en,
                calculate_sfpu_binary,
                (false /*APPROX*/, BinaryOp::ADD, is_fp32_dest_acc_en, dst_rounding_mode, ITERATIONS),
                src0_tile,
                src1_tile,
                dst_tile,
                VectorMode::RC);
        }
    }
    else if constexpr (OP == BinaryOp::SUB)
    {
        // Int32 SUB is not ported to Quasar (sub_int_sfpu.h is WH-only); float path only.
        SFPU_BINARY_CALL(
            DST_SYNC,
            is_fp32_dest_acc_en,
            calculate_sfpu_binary,
            (false /*APPROX*/, BinaryOp::SUB, is_fp32_dest_acc_en, dst_rounding_mode, ITERATIONS),
            src0_tile,
            src1_tile,
            dst_tile,
            VectorMode::RC);
    }
    else if constexpr (OP == BinaryOp::GT)
    {
        SFPU_BINARY_CALL(
            DST_SYNC, is_fp32_dest_acc_en, calculate_binary_comp_int32, (false, ITERATIONS, SfpuType::gt), src0_tile, src1_tile, dst_tile, VectorMode::RC);
    }
    else if constexpr (OP == BinaryOp::LT)
    {
        SFPU_BINARY_CALL(
            DST_SYNC, is_fp32_dest_acc_en, calculate_binary_comp_int32, (false, ITERATIONS, SfpuType::lt), src0_tile, src1_tile, dst_tile, VectorMode::RC);
    }
    else if constexpr (OP == BinaryOp::LE)
    {
        SFPU_BINARY_CALL(
            DST_SYNC, is_fp32_dest_acc_en, calculate_binary_comp_int32, (false, ITERATIONS, SfpuType::le), src0_tile, src1_tile, dst_tile, VectorMode::RC);
    }
    else if constexpr (OP == BinaryOp::GE)
    {
        SFPU_BINARY_CALL(
            DST_SYNC, is_fp32_dest_acc_en, calculate_binary_comp_int32, (false, ITERATIONS, SfpuType::ge), src0_tile, src1_tile, dst_tile, VectorMode::RC);
    }
    else if constexpr (OP == BinaryOp::MUL)
    {
        if (math_format == DataFormat::Int32)
        {
            SFPU_BINARY_CALL(DST_SYNC, is_fp32_dest_acc_en, _mul_int32_, (false, ITERATIONS), src0_tile, src1_tile, dst_tile, VectorMode::RC);
        }
        else
        {
            SFPU_BINARY_CALL(
                DST_SYNC,
                is_fp32_dest_acc_en,
                calculate_sfpu_binary,
                (false /*APPROX*/, BinaryOp::MUL, is_fp32_dest_acc_en, dst_rounding_mode, ITERATIONS),
                src0_tile,
                src1_tile,
                dst_tile,
                VectorMode::RC);
        }
    }
    else if constexpr (OP == BinaryOp::DIV)
    {
        SFPU_BINARY_CALL(
            DST_SYNC,
            is_fp32_dest_acc_en,
            calculate_sfpu_binary,
            (false /*APPROX*/, BinaryOp::DIV, is_fp32_dest_acc_en, dst_rounding_mode, ITERATIONS),
            src0_tile,
            src1_tile,
            dst_tile,
            VectorMode::RC);
    }
    else if constexpr (quasar_binary_op_is_quant(OP))
    {
        SFPU_BINARY_CALL(
            DST_SYNC,
            is_fp32_dest_acc_en,
            quant_family,
            (quant_variant_of<OP>(), ITERATIONS, SIGN_MAGNITUDE_FORMAT),
            src0_tile,
            src1_tile,
            dst_tile,
            VectorMode::RC);
    }
    else if constexpr (quasar_binary_op_is_max_min(OP))
    {
        constexpr bool IS_MAX = (OP == BinaryOp::MAX);
        // All integer formats route through the Int32 path; float / MX use Float32.
        if (math_format == DataFormat::Int32)
        {
            SFPU_BINARY_CALL(
                DST_SYNC,
                is_fp32_dest_acc_en,
                calculate_binary_max_min,
                (DataFormat::Int32, IS_MAX, ITERATIONS),
                src0_tile,
                src1_tile,
                dst_tile,
                VectorMode::RC);
        }
        else
        {
            SFPU_BINARY_CALL(
                DST_SYNC,
                is_fp32_dest_acc_en,
                calculate_binary_max_min,
                (DataFormat::Float32, IS_MAX, ITERATIONS),
                src0_tile,
                src1_tile,
                dst_tile,
                VectorMode::RC);
        }
    }
    // ── SFPU parity set ──────────────────────────────────────────────────────────────
#ifdef QSR_HAS_ATAN2
    else if constexpr (OP == BinaryOp::ATAN2)
    {
        SFPU_BINARY_CALL(
            DST_SYNC,
            is_fp32_dest_acc_en,
            calculate_sfpu_atan2,
            (false /*APPROX*/, ITERATIONS, is_fp32_dest_acc_en),
            src0_tile,
            src1_tile,
            dst_tile,
            VectorMode::RC);
    }
#endif
#ifdef QSR_HAS_BINARY_BITWISE
    else if constexpr (OP == BinaryOp::BITWISE_AND || OP == BinaryOp::BITWISE_OR || OP == BinaryOp::BITWISE_XOR)
    {
        // int32 bitwise AND/OR/XOR over the raw two's-complement patterns in Dest.
        constexpr BinaryBitwiseOp kBw = (OP == BinaryOp::BITWISE_AND)  ? BinaryBitwiseOp::AND
                                        : (OP == BinaryOp::BITWISE_OR) ? BinaryBitwiseOp::OR
                                                                       : BinaryBitwiseOp::XOR;
        SFPU_BINARY_CALL(
            DST_SYNC,
            is_fp32_dest_acc_en,
            calculate_sfpu_binary_bitwise,
            (false /*APPROX*/, kBw, ckernel::InstrModLoadStore::INT32, ITERATIONS),
            src0_tile,
            src1_tile,
            dst_tile,
            VectorMode::RC);
    }
#endif
#ifdef QSR_HAS_BINARY_FMOD
    else if constexpr (OP == BinaryOp::FMOD)
    {
        SFPU_BINARY_CALL(
            DST_SYNC,
            is_fp32_dest_acc_en,
            calculate_sfpu_binary_fmod,
            (false /*APPROX*/, ITERATIONS, is_fp32_dest_acc_en),
            src0_tile,
            src1_tile,
            dst_tile,
            VectorMode::RC);
    }
    else if constexpr (OP == BinaryOp::FMOD_INT32)
    {
        SFPU_BINARY_CALL(DST_SYNC, is_fp32_dest_acc_en, calculate_fmod_int32, (false /*APPROX*/, ITERATIONS), src0_tile, src1_tile, dst_tile, VectorMode::RC);
    }
#endif
#ifdef QSR_HAS_BINARY_POW
    else if constexpr (OP == BinaryOp::POW)
    {
        SFPU_BINARY_CALL(
            DST_SYNC,
            is_fp32_dest_acc_en,
            calculate_sfpu_binary,
            (false /*APPROX*/, BinaryOp::POW, is_fp32_dest_acc_en, dst_rounding_mode, ITERATIONS),
            src0_tile,
            src1_tile,
            dst_tile,
            VectorMode::RC);
    }
#endif
#ifdef QSR_HAS_BINARY_REMAINDER
    else if constexpr (OP == BinaryOp::REMAINDER)
    {
        SFPU_BINARY_CALL(
            DST_SYNC,
            is_fp32_dest_acc_en,
            calculate_sfpu_binary_remainder,
            (false /*APPROX*/, ITERATIONS, is_fp32_dest_acc_en),
            src0_tile,
            src1_tile,
            dst_tile,
            VectorMode::RC);
    }
    else if constexpr (OP == BinaryOp::REMAINDER_INT32)
    {
        SFPU_BINARY_CALL(
            DST_SYNC, is_fp32_dest_acc_en, calculate_remainder_int32, (false /*APPROX*/, ITERATIONS), src0_tile, src1_tile, dst_tile, VectorMode::RC);
    }
    else if constexpr (OP == BinaryOp::REMAINDER_UINT32)
    {
        SFPU_BINARY_CALL(
            DST_SYNC, is_fp32_dest_acc_en, calculate_remainder_uint32, (false /*APPROX*/, ITERATIONS), src0_tile, src1_tile, dst_tile, VectorMode::RC);
    }
#endif
#ifdef QSR_HAS_DIV_INT32
    else if constexpr (OP == BinaryOp::DIV_INT32)
    {
        SFPU_BINARY_CALL(
            DST_SYNC, is_fp32_dest_acc_en, calculate_div_int32_trunc, (false /*APPROX*/, ITERATIONS), src0_tile, src1_tile, dst_tile, VectorMode::RC);
    }
#endif
#ifdef QSR_HAS_DIV_INT32_FLOOR
    else if constexpr (OP == BinaryOp::DIV_INT32_FLOOR)
    {
        SFPU_BINARY_CALL(
            DST_SYNC, is_fp32_dest_acc_en, calculate_div_int32_floor, (false /*APPROX*/, ITERATIONS), src0_tile, src1_tile, dst_tile, VectorMode::RC);
    }
#endif
#ifdef QSR_HAS_ISCLOSE
    else if constexpr (OP == BinaryOp::ISCLOSE)
    {
        // out = (|a - b| <= atol + rtol * |b|) ? 1 : 0. rtol/atol are torch's defaults
        // (1e-5 / 1e-8) as fp32 bit patterns, matching the Blackhole dispatch.
        SFPU_BINARY_CALL(
            DST_SYNC,
            is_fp32_dest_acc_en,
            calculate_sfpu_isclose,
            (false /*APPROX*/, ITERATIONS, false /*EQUAL_NAN*/),
            src0_tile,
            src1_tile,
            dst_tile,
            VectorMode::RC,
            0x3727C5ACu,  // rtol = 1e-5f
            0x322BCC77u); // atol = 1e-8f
    }
#endif
#ifdef QSR_HAS_LOGSIGMOID
    else if constexpr (OP == BinaryOp::LOGSIGMOID)
    {
        // logsigmoid(x) = -softplus(-x), with x at src0 and exp(-x) supplied at src1.
        SFPU_BINARY_CALL(DST_SYNC, is_fp32_dest_acc_en, calculate_logsigmoid, (false /*APPROX*/, ITERATIONS), src0_tile, src1_tile, dst_tile, VectorMode::RC);
    }
#endif
#ifdef QSR_HAS_MASK
    else if constexpr (OP == BinaryOp::MASK)
    {
        // out = (mask != 0) ? data : 0, with data at src0 and mask at src1.
        SFPU_BINARY_CALL(DST_SYNC, is_fp32_dest_acc_en, calculate_mask_binary, (false /*APPROX*/, ITERATIONS), src0_tile, src1_tile, dst_tile, VectorMode::RC);
    }
#endif
#ifdef QSR_HAS_RSUB_INT32
    else if constexpr (OP == BinaryOp::RSUB_INT32)
    {
        SFPU_BINARY_CALL(
            DST_SYNC,
            is_fp32_dest_acc_en,
            calculate_rsub_int,
            (false /*APPROX*/, ckernel::InstrModLoadStore::INT32, ITERATIONS),
            src0_tile,
            src1_tile,
            dst_tile,
            VectorMode::RC);
    }
#endif
    else
    {
        static_assert(unhandled_op<OP>, "call_binary_sfpu_operation_quasar: unhandled Quasar binary SFPU operation");
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Ternary SFPU parity ops (addcmul / addcdiv / lerp / snake_beta)
//
// Quasar already carries the ternary plumbing that `where` uses --
// llk_math_eltwise_ternary_sfpu_macros.h and _llk_math_eltwise_ternary_sfpu_init_ -- so
// these four need only a dispatch of their own, not new infrastructure.
// ─────────────────────────────────────────────────────────────────────────────

/**
 * @brief Whether OPERATION is one of the four ternary parity ops.
 *
 * @param op The SFPU operation type to classify.
 */
inline constexpr bool is_ternary_parity_op(SfpuType op)
{
    return op == SfpuType::addcmul || op == SfpuType::addcdiv || op == SfpuType::lerp || op == SfpuType::snake_beta;
}

/**
 * @brief Run the per-operation init step for a Quasar ternary SFPU op.
 *
 * @tparam OPERATION The ternary SFPU op (compile-time `SfpuType` constant).
 * @tparam APPROX Whether the reciprocal-based ops use the approximate path.
 * @note Pair with @ref call_ternary_sfpu_operation_quasar for the calculate step.
 */
template <SfpuType OPERATION, bool APPROX = false>
void init_ternary_sfpu_operation_quasar()
{
    // Global-namespace init; the `::` is required from inside namespace test_utils.
    ::_llk_math_eltwise_ternary_sfpu_init_<OPERATION>();
#ifdef QSR_HAS_ADDCDIV
    if constexpr (OPERATION == SfpuType::addcdiv)
    {
        // addcdiv divides through sfpu_reciprocal, so it needs that polynomial loaded.
        init_addcdiv<APPROX>();
    }
#endif
#ifdef QSR_HAS_SNAKE_BETA
    if constexpr (OPERATION == SfpuType::snake_beta)
    {
        snake_beta_init<APPROX>();
    }
#endif
    // addcmul and lerp are stateless beyond the shared addrmod setup above.
}

/**
 * @brief Apply a Quasar ternary SFPU op over three Dest operands into a result tile.
 *
 * @tparam OPERATION The ternary SFPU op (compile-time `SfpuType` constant).
 * @tparam DST_SYNC Destination synchronization mode used for bounds checking.
 * @tparam is_fp32_dest_acc_en Whether Dest is in FP32 mode.
 * @tparam APPROX Whether the reciprocal-based ops use the approximate path.
 * @tparam ITERATIONS Number of SFPU loop iterations.
 * @param src0_tile,src1_tile,src2_tile Operand tile indices (a, b, c).
 * @param dst_tile Result tile index.
 * @param value_bits fp32 bit pattern of the addc scalar; ignored by lerp and snake_beta.
 * @note Must be preceded by @ref init_ternary_sfpu_operation_quasar for the same op.
 */
template <SfpuType OPERATION, DstSync DST_SYNC, bool is_fp32_dest_acc_en, bool APPROX = false, int ITERATIONS = SFPU_ITERATIONS>
void call_ternary_sfpu_operation_quasar(
    [[maybe_unused]] std::uint32_t src0_tile,
    [[maybe_unused]] std::uint32_t src1_tile,
    [[maybe_unused]] std::uint32_t src2_tile,
    [[maybe_unused]] std::uint32_t dst_tile,
    [[maybe_unused]] std::uint32_t value_bits)
{
    static_assert(is_ternary_parity_op(OPERATION), "call_ternary_sfpu_operation_quasar: OPERATION must be a ternary parity SfpuType");

#ifdef QSR_HAS_ADDCMUL
    if constexpr (OPERATION == SfpuType::addcmul)
    {
        SFPU_TERNARY_CALL(
            DST_SYNC,
            is_fp32_dest_acc_en,
            calculate_addcmul,
            (APPROX, is_fp32_dest_acc_en, ITERATIONS),
            src0_tile,
            src1_tile,
            src2_tile,
            dst_tile,
            VectorMode::RC,
            value_bits);
    }
#endif
#ifdef QSR_HAS_ADDCDIV
    if constexpr (OPERATION == SfpuType::addcdiv)
    {
        SFPU_TERNARY_CALL(
            DST_SYNC,
            is_fp32_dest_acc_en,
            calculate_addcdiv,
            (APPROX, is_fp32_dest_acc_en, ITERATIONS),
            src0_tile,
            src1_tile,
            src2_tile,
            dst_tile,
            VectorMode::RC,
            value_bits);
    }
#endif
#ifdef QSR_HAS_LERP
    if constexpr (OPERATION == SfpuType::lerp)
    {
        SFPU_TERNARY_CALL(
            DST_SYNC,
            is_fp32_dest_acc_en,
            calculate_lerp,
            (APPROX, is_fp32_dest_acc_en, ITERATIONS),
            src0_tile,
            src1_tile,
            src2_tile,
            dst_tile,
            VectorMode::RC);
    }
#endif
#ifdef QSR_HAS_SNAKE_BETA
    if constexpr (OPERATION == SfpuType::snake_beta)
    {
        SFPU_TERNARY_CALL(
            DST_SYNC,
            is_fp32_dest_acc_en,
            calculate_snake_beta,
            (APPROX, is_fp32_dest_acc_en, ITERATIONS),
            src0_tile,
            src1_tile,
            src2_tile,
            dst_tile,
            VectorMode::RC);
    }
#endif
}

// ─────────────────────────────────────────────────────────────────────────────
// Tile-structural SFPU parity ops (tiled_prod / int_sum / alt_complex_rotate90)
//
// These address Dest by slot and combine slots with each other, so unlike every op above
// they are not element-wise and cannot share the unary dispatch's per-element contract.
// They take no operand beyond the tile itself. int_sum's two modes read slots in faces 1
// and 2, so they must be dispatched once per tile (VectorMode::RC_custom) rather than once
// per face; the other two stay inside their face.
// ─────────────────────────────────────────────────────────────────────────────

/**
 * @brief Whether OPERATION is one of the tile-structural parity ops.
 *
 * @param op The SFPU operation type to classify.
 */
inline constexpr bool is_structural_parity_op(SfpuType op)
{
    return op == SfpuType::tiled_prod || op == SfpuType::sum_int_row || op == SfpuType::sum_int_col || op == SfpuType::alt_complex_rotate90;
}

/**
 * @brief Apply a tile-structural SFPU op in-place on one Dest tile.
 *
 * @tparam OPERATION The structural SFPU op (compile-time `SfpuType` constant).
 * @tparam DST_SYNC Destination synchronization mode used for bounds checking.
 * @tparam is_fp32_dest_acc_en Whether Dest is in FP32 mode.
 * @tparam ITERATIONS Number of SFPU loop iterations; the rotate kernel halves it because
 *         it consumes two slots per step.
 * @param dst_index Destination tile index operated on (already offset by DST_INDEX).
 * @note Must be preceded by @ref init_unary_sfpu_operation_quasar for the same op, which
 *       carries these ops' init steps.
 */
template <SfpuType OPERATION, DstSync DST_SYNC, bool is_fp32_dest_acc_en, int ITERATIONS = SFPU_ITERATIONS>
void call_structural_sfpu_operation_quasar([[maybe_unused]] std::uint32_t dst_index)
{
    static_assert(is_structural_parity_op(OPERATION), "call_structural_sfpu_operation_quasar: OPERATION must be a structural parity SfpuType");

#ifdef QSR_HAS_TILED_PROD
    if constexpr (OPERATION == SfpuType::tiled_prod)
    {
        SFPU_UNARY_CALL(DST_SYNC, is_fp32_dest_acc_en, calculate_tiled_prod, (false /*APPROX*/, ITERATIONS), dst_index, VectorMode::RC);
    }
#endif
#ifdef QSR_HAS_INT_SUM
    if constexpr (OPERATION == SfpuType::sum_int_row)
    {
        SFPU_UNARY_CALL(DST_SYNC, is_fp32_dest_acc_en, calculate_sum_int_row, (false /*APPROX*/), dst_index, VectorMode::RC);
    }
    if constexpr (OPERATION == SfpuType::sum_int_col)
    {
        SFPU_UNARY_CALL(DST_SYNC, is_fp32_dest_acc_en, calculate_sum_int_col, (false /*APPROX*/), dst_index, VectorMode::RC);
    }
#endif
#ifdef QSR_HAS_ALT_COMPLEX_ROTATE90
    if constexpr (OPERATION == SfpuType::alt_complex_rotate90)
    {
        // Consumes two slots per step, so it runs half as many iterations as the others.
        SFPU_UNARY_CALL(DST_SYNC, is_fp32_dest_acc_en, calculate_alt_complex_rotate90, (false /*APPROX*/, ITERATIONS / 2), dst_index, VectorMode::RC);
    }
#endif
}

} // namespace test_utils
