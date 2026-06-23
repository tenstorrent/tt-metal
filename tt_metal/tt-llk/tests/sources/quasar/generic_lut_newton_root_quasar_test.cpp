// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// NEWTON_ROOT (magic-seed + Newton/Householder) LUT activation on Quasar
// (sim-qsr / ttsim).
//
// Quasar port of sources/generic_lut_newton_root_bh_test.cpp, which is itself a
// byte-faithful replica of the production EVAL_METHOD_NEWTON_ROOT path from
//   tt_metal/programming_examples/generic_lut_activation_embedded/kernels/
//   compute/piecewise_generic.cpp  (#if defined(EVAL_METHOD_NEWTON_ROOT)).
//
// Three flavours, selected by injected -D defines from the Python driver:
//   sqrt  : NEWTON_ROOT_N==2, no NEWTON_ROOT_RECIPROCAL  -> double-Newton
//   rsqrt : NEWTON_ROOT_N==2, NEWTON_ROOT_RECIPROCAL     -> inverse-sqrt Newton
//   cbrt  : NEWTON_ROOT_N==3                              -> cubic Householder
//
// LUT injected via -D defines:
//   NEWTON_ROOT_N, NEWTON_ROOT_ITERS, NEWTON_ROOT_MAGIC,
//   NEWTON_ROOT_C1, NEWTON_ROOT_C2 [, NEWTON_ROOT_RECIPROCAL]
//
// =====================================================================
// BH -> QUASAR SFPU translation (see generic_lut_rational_quasar_test.cpp for
// the full translation reference). The Newton-root arithmetic is shift / integer
// subtract / FMA / exponent-field ops, ALL of which exist on Quasar with ONE
// exception:
//
//   * addexp(x, -1)  (== x * 0.5 via SFPDIVP2 / the SFPADDEXP exponent-decrement
//     instruction) is MISSING on the Quasar ttsim build. It is substituted with
//     the numerically-identical multiply  x * 0.5f. (0.5 is an exact fp32 value,
//     so x*0.5f and the exponent decrement produce bit-identical results for all
//     finite normal x; subnormals are out of the newton_root domain.)
//
// All other intrinsics (reinterpret, >>, setexp/exexp, setsgn, vConst1, the int
// subtract MAGIC - i) are present on Quasar sfpi and used verbatim.
//
// Quasar-LLK harness specifics match the proven poly/rational quasar tests:
//   * Quasar semantic LLK headers (llk_unpack_unary_operand.h, NOT llk_unpack_A.h)
//   * 2-rows-per-iteration SFPU model with _incr_counters_<...SFP_ROWS...>()
//     over SFPU_ITERATIONS (NOT a 0..31 dst_reg loop)
//   * sfpi dst_reg[0]/vFloat/v_if; float-literal constants (no Converter::as_float)
//   * MAGIC/C1/C2 are immediates (self-contained; no programmable const regs)
//
// The Python golden (test_newton_root_quasar.py) replicates the EXACT same
// arithmetic (the ttpoly rangered._eval_newton_root model) so the PCC / bit-
// distance isolates the BH->Quasar translation.

#include <array>
#include <cstdint>
#include <limits>

#include "ckernel.h"
#include "llk_defs.h"
#include "llk_memory_checks.h"
#include "sfpu_stub.h"

#ifndef NEWTON_ROOT_N
#define NEWTON_ROOT_N 2
#endif
#ifndef NEWTON_ROOT_ITERS
#define NEWTON_ROOT_ITERS 2
#endif
#ifndef NEWTON_ROOT_MAGIC
#define NEWTON_ROOT_MAGIC 0x5f1110a0
#endif
#ifndef NEWTON_ROOT_C1
#define NEWTON_ROOT_C1 2.2825186f
#endif
#ifndef NEWTON_ROOT_C2
#define NEWTON_ROOT_C2 2.2533049f
#endif

#ifdef LLK_TRISC_UNPACK

#include "llk_math_common.h"
#include "llk_unpack_common.h"
#include "llk_unpack_unary_operand.h"
#include "params.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    const std::uint32_t buf_desc_id = 0;
    const std::uint32_t num_tiles   = params.TILE_CNT;

    // FPU path: UNPACK -> SrcA -> FPU datacopy (MOVA2D) -> Dest.
    set_up_dest_dvalid_per_thread<dest_dvalid_client::UNPACK>({dest_dvalid_client::FPU, dest_dvalid_client::SFPU, dest_dvalid_client::PACK});

    buffer_descriptor_u bd_val = {0};

    bd_val.f.l1_addr_16B = L1_ADDRESS(params.buffer_A[0]);
    bd_val.f.format      = static_cast<std::uint8_t>(formats.unpack_A_src);
    bd_val.f.x_dim       = params.TEST_FACE_C_DIM;
    bd_val.f.y_dim       = params.TEST_FACE_R_DIM;
    bd_val.f.z_dim       = params.num_faces;

    tdma_descriptor_t td_val;
    td_val.buf_desc        = bd_val;
    td_val.buf_desc_id     = buf_desc_id;
    td_val.reg_data_format = static_cast<std::uint8_t>(formats.unpack_A_dst);
    _configure_buf_desc_table_(td_val.buf_desc_id, td_val.buf_desc);

    _llk_unpack_configure_unary_<p_unpacr::UNP_A>(td_val);

    _llk_unpack_unary_operand_init_<p_unpacr::UNP_A, false /*transpose*/, is_fp32_dest_acc_en>(buf_desc_id, ckernel::DEFAULT_TENSOR_SHAPE, num_tiles);
    _llk_unpack_unary_operand_<p_unpacr::UNP_A>(0, ckernel::DEFAULT_TENSOR_SHAPE);
}

#endif

#ifdef LLK_TRISC_MATH

const bool is_int_fpu_en = false;

#include "cfg_defines.h"
#include "cmath_common.h"
#include "llk_math_common.h"
#include "llk_math_eltwise_unary_datacopy.h"
#include "llk_math_eltwise_unary_sfpu.h"
#include "params.h"
#include "sfpi.h"

using namespace ckernel;
using namespace ckernel::math;
using namespace ckernel::sfpu;

namespace sfpi
{

// addexp(v, -1) == v * 0.5. SFPADDEXP / SFPDIVP2 (the exponent-decrement op that
// addexp lowers to) is MISSING on the Quasar ttsim build, so multiply by the
// exact fp32 constant 0.5f instead. Bit-identical for all finite normal inputs.
sfpi_inline vFloat half(vFloat v)
{
    return v * 0.5f;
}

#if (NEWTON_ROOT_N == 2) && !defined(NEWTON_ROOT_RECIPROCAL)
// --- sqrt: magic seed + SQRT_23-bit double-Newton (native parity) ------------
inline vFloat newton_root_sqrt(vFloat x)
{
    const vInt magic_seed = (vInt)(int)(NEWTON_ROOT_MAGIC);
    const vFloat c1       = NEWTON_ROOT_C1;
    const vFloat c2       = NEWTON_ROOT_C2;

    vInt i   = reinterpret<vInt>(reinterpret<vUInt>(x) >> 1);
    vFloat y = reinterpret<vFloat>(magic_seed - i);

    vFloat xy            = x * y;
    vFloat negative_y    = -y;
    vFloat c             = negative_y * xy;
    y                    = y * (c1 + c * (c2 + c));
    xy                   = x * y;
    negative_y           = -y;
    vFloat one_minus_xyy = vConst1 + (negative_y * xy);
    vFloat half_xy       = half(xy); // addexp(xy,-1) substitute
    vFloat infinity      = sFloat16b(std::numeric_limits<float>::infinity());
    v_if (reinterpret<vInt>(x) < reinterpret<vInt>(infinity))
    {
        y = one_minus_xyy * half_xy + xy;
    }
    v_endif;
    v_if (x < 0.0f)
    {
        y = std::numeric_limits<float>::quiet_NaN();
    }
    v_endif;
    return y;
}
#endif

#if (NEWTON_ROOT_N == 2) && defined(NEWTON_ROOT_RECIPROCAL)
// --- rsqrt: classic inverse-sqrt magic seed + Newton -------------------------
inline vFloat newton_root_rsqrt(vFloat x)
{
    const vInt magic_seed = (vInt)(int)(NEWTON_ROOT_MAGIC);
    const vFloat c1       = NEWTON_ROOT_C1; // 1.5

    vInt i        = reinterpret<vInt>(reinterpret<vUInt>(x) >> 1);
    vFloat y      = reinterpret<vFloat>(magic_seed - i);
    vFloat half_x = half(x); // addexp(x,-1) substitute
#pragma GCC unroll 4
    for (int s = 0; s < NEWTON_ROOT_ITERS; s++)
    {
        y = y * (c1 - half_x * (y * y));
    }
    v_if (x < 0.0f)
    {
        y = std::numeric_limits<float>::quiet_NaN();
    }
    v_endif;
    v_if (x == 0.0f)
    {
        y = std::numeric_limits<float>::infinity();
    }
    v_endif;
    return y;
}
#endif

#if (NEWTON_ROOT_N == 3)
// --- cbrt: minimal exponent seed + DIVISION-FREE cubic Householder -----------
// No addexp here, so the body is byte-identical to BH apart from the int->float
// path: BH uses int32_to_float; Quasar's only int->float path is via vSMag.
inline vFloat newton_root_cbrt(vFloat x)
{
    vInt sign_bits = reinterpret<vInt>(x) & (vInt)0x80000000;
    vFloat ax      = setsgn(x, 0);

    vInt e_int = exexp(ax, ExponentMode::NoDebias) - 127;
    vFloat m   = setexp(ax, 127);

    const vFloat magic = reinterpret<vFloat>((vInt)(int)0x4B400000);
    // int -> float via vSMag (the only Quasar int->float path).
    vFloat ef = convert<vFloat>(convert<vSMag>(e_int), RoundMode::Nearest);
    v_if (e_int < 0)
    {
        ef = -convert<vFloat>(convert<vSMag>(~e_int + 1), RoundMode::Nearest);
    }
    v_endif;
    vInt q = reinterpret<vInt>(ef * (1.0f / 3.0f) + magic) - reinterpret<vInt>(magic);

    vFloat wm = -0.27f * m + 1.25f;
    vFloat w  = setexp(wm, exexp(wm, ExponentMode::NoDebias) - q);

    const vFloat a13 = 1.0f / 3.0f;
    const vFloat a29 = 2.0f / 9.0f;
    for (int s = 0; s < NEWTON_ROOT_ITERS; s++)
    {
        vFloat c = 1.0f - ax * (w * w * w);
        w        = w * (1.0f + c * a13 + (c * c) * a29);
    }
    vFloat y = ax * w * w;
    y        = reinterpret<vFloat>(reinterpret<vInt>(y) | sign_bits);
    v_if (x == 0.0f)
    {
        y = 0.0f;
    }
    v_endif;
    return y;
}
#endif

template <std::uint32_t DEG>
inline vFloat newton_root_eval(vFloat x)
{
    (void)DEG;
#if (NEWTON_ROOT_N == 3)
    return newton_root_cbrt(x);
#elif defined(NEWTON_ROOT_RECIPROCAL)
    return newton_root_rsqrt(x);
#else
    return newton_root_sqrt(x);
#endif
}

// Evaluate the newton-root activation on the current 2-row Dest window.
sfpi_inline void newton_root_lut_sfp_rows()
{
    const vFloat x = dst_reg[0];
    dst_reg[0]     = newton_root_eval<0>(x);
}

} // namespace sfpi

namespace
{
template <int ITERATIONS = SFPU_ITERATIONS>
inline void calculate_newton_root_lut()
{
#pragma GCC unroll 1
    for (int d = 0; d < ITERATIONS; d++)
    {
        sfpi::newton_root_lut_sfp_rows();
        ckernel::math::_incr_counters_<0x0, 0x0, ckernel::math::SFP_ROWS, 0x0>(); // dest_reg++ (advances by 2 rows)
    }
}
} // namespace

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    // FPU path: UNPACK -> SrcA -> FPU datacopy (MOVA2D) -> Dest -> SFPU -> PACK.
    set_up_dest_dvalid_per_thread<dest_dvalid_client::FPU>({dest_dvalid_client::FPU, dest_dvalid_client::SFPU, dest_dvalid_client::PACK});
    set_up_dest_dvalid_per_thread<dest_dvalid_client::SFPU>({dest_dvalid_client::FPU, dest_dvalid_client::SFPU, dest_dvalid_client::PACK});

    DataFormat src_format = static_cast<DataFormat>(formats.math);
    _llk_math_srcAB_hw_configure_<IMPLIED_MATH_FORMAT, is_fp32_dest_acc_en, is_int_fpu_en>(src_format, src_format);

    // FPU datacopy SrcA -> Dest (MOVA2D) before SFPU operates on it.
    const std::uint32_t num_rows = params.num_faces * params.TEST_FACE_R_DIM;
    _llk_math_eltwise_unary_datacopy_init_<DataCopyType::A2D, is_fp32_dest_acc_en>(num_rows, 1);
    for (std::uint32_t i = 0; i < params.TILE_CNT; ++i)
    {
        _llk_math_eltwise_unary_datacopy_(num_rows, params.DST_INDEX + i);
    }
    _llk_math_set_dvalid_<p_cleardvalid::FPU, dest_sync>();

    _llk_math_eltwise_sfpu_init_();

    for (std::uint32_t i = 0; i < params.TILE_CNT; ++i)
    {
        _llk_math_eltwise_unary_sfpu_params_(calculate_newton_root_lut<SFPU_ITERATIONS>, params.DST_INDEX + i);
    }

    _llk_math_set_dvalid_<p_cleardvalid::SFPU, dest_sync>();

    wait_sfpu_idle();
    wait_fpu_idle();
    wait_mop_idle();
}

#endif

#ifdef LLK_TRISC_PACK

#include "cfg_defines.h"
#include "llk_pack.h"
#include "llk_pack_common.h"
#include "params.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    std::uint32_t const buf_desc_id        = 8;
    const std::uint32_t num_tiles_per_pack = params.TILE_CNT;

    set_up_dest_dvalid_per_thread<dest_dvalid_client::PACK>({dest_dvalid_client::FPU, dest_dvalid_client::SFPU, dest_dvalid_client::PACK});

    buffer_descriptor_u bd_val = {0};
    bd_val.f.l1_addr_16B       = params.buffer_Res[0] / 16;
    bd_val.f.format            = static_cast<std::uint8_t>(formats.pack_dst);
    bd_val.f.x_dim             = params.TEST_FACE_C_DIM;
    bd_val.f.y_dim             = params.TEST_FACE_R_DIM;
    bd_val.f.z_dim             = params.num_faces;

    tdma_descriptor_t tdma_desc;
    tdma_desc.buf_desc        = bd_val;
    tdma_desc.buf_desc_id     = buf_desc_id;
    tdma_desc.reg_data_format = static_cast<std::uint8_t>(formats.pack_src);
    _configure_buf_desc_table_(tdma_desc.buf_desc_id, tdma_desc.buf_desc);

    _llk_pack_hw_configure_<p_pacr::PACK0>(tdma_desc);
    _llk_pack_init_(buf_desc_id, ckernel::DEFAULT_TENSOR_SHAPE, num_tiles_per_pack);
    _llk_pack_(params.DST_INDEX, 0, ckernel::DEFAULT_TENSOR_SHAPE);
    _llk_pack_dest_dvalid_section_done_<dest_sync, is_fp32_dest_acc_en>();
}
#endif
