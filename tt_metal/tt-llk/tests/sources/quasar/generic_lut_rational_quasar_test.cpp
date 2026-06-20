// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// Generic piecewise-RATIONAL LUT activation on Quasar.
//
// Sibling of generic_lut_activation_quasar_test.cpp (piecewise-polynomial),
// but each segment evaluates a rational approximation P(x)/Q(x) instead of a
// single polynomial. The numerator P and denominator Q are independent
// polynomials (degrees RAT_NUM_DEGREE / RAT_DEN_DEGREE). The division is done
// in the SFPU as  result = P(x) * (1 / Q(x))  using the sfpi-based iterative
// reciprocal (_sfpu_reciprocal_) from ckernel_sfpu_recip.h.
//
// Reciprocal choice: the iterative sfpi reciprocal (approx_recip + 2 rounds of
// Newton-Raphson) is used instead of the native SFPNONLINEAR RECIP. In ttsim,
// SFPNONLINEAR RECIP is libm-exact and so is optimistic vs real silicon; the
// iterative sfpi path uses the same approx-reciprocal seed + NR refinement that
// runs on hardware, so its PCC is representative of silicon. (See RETURN note.)
//
// Unlike the poly test, the LUT layout (boundaries, per-segment numerator and
// denominator coefficients, degrees, segment count) is NOT hard-coded here.
// It is injected at build time through build.h (pulled in via params.h) by the
// Python test's RATIONAL_LUT template parameter, so the same source serves any
// rational config the tt-polynomial-fitter emits. The Python golden replicates
// the EXACT same coefficients and division semantics so the PCC isolates kernel
// correctness on the fitter's rational coefficients.
//
// build.h provides:
//   constexpr std::uint32_t RAT_NUM_SEGMENTS;
//   constexpr std::uint32_t RAT_NUM_DEGREE;   // numerator P degree
//   constexpr std::uint32_t RAT_DEN_DEGREE;   // denominator Q degree
//   constexpr std::array<float, RAT_NUM_SEGMENTS + 1> RAT_BOUNDARIES;
//   constexpr std::array<float, RAT_NUM_SEGMENTS * (RAT_NUM_DEGREE + 1)> RAT_NUM_COEFFS;
//   constexpr std::array<float, RAT_NUM_SEGMENTS * (RAT_DEN_DEGREE + 1)> RAT_DEN_COEFFS;
// All coefficient arrays are stored low-order-first (c0, c1, c2, ...).
//
// Quasar-LLK specifics match the proven poly test: Quasar semantic LLK headers
// (no llk_unpack_A.h), the 2-rows-per-iteration SFPU model with
// _incr_counters_<...SFP_ROWS...>(), sfpi dst_reg[0]/vFloat/v_if, and float
// literal constants (no Converter::as_float).

#include <array>
#include <cstdint>

#include "ckernel.h"
#include "llk_defs.h"
#include "llk_memory_checks.h"
#include "sfpu_stub.h"

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
#include "sfpu/ckernel_sfpu_recip.h"

using namespace ckernel;
using namespace ckernel::math;
using namespace ckernel::sfpu;

namespace
{
// SFPADDI avoidance (ported from the poly kernel). The Horner step `acc * x + c`
// lowers in sfpi to a separate SFPMUL + SFPADD. When `c` is a compile-time
// constant the backend can fuse the SFPADD with the constant's SFPLOADI into a
// single SFPADDI for round values — and Quasar removed SFPADDI (ttsim aborts
// "tensix_execute_sfpaddi"). Emitting the multiply-add as one fused SFPMAD means
// the constant is the MAD's third (addend) operand, so there is no standalone
// SFPADD for the backend to fold into SFPADDI. (The reciprocal's own NR subtract
// is handled separately by -mno-tt-tensix-optimize-combine; see below.)
sfpi_inline sfpi::vFloat fma_const(sfpi::vFloat a, sfpi::vFloat b, float c)
{
    const sfpi::vFloat cv = sfpi::vFloat(c);
    return __builtin_rvtt_sfpmad(a.get(), b.get(), cv.get(), sfpi::SFPMAD_MOD1_OFFSET_NONE);
}

// Horner step recursion with COMPILE-TIME coefficient indices (the proven poly
// kernel fold). Reading the constexpr coefficient array ONLY through constexpr
// indices (never `&ARR[i]` / `ARR[k]` with a runtime index) lets the compiler
// constant-fold every coefficient into an SFPLOADI immediate and ELIDE the
// coefficient array — so it never lands in .ldm_data and cannot overflow the
// tiny 1792 B private TRISC1_LOCAL_DATA_MEM region (the "section .ldm_data not
// within region" link error on large deg/seg rational configs). The folded
// immediates live in code (16 KiB TRISC1_CODE), which has room. Applies
// coefficients c_K down to c_0 from the given constexpr array reference + BASE.
template <const auto& ARR, std::uint32_t BASE, int K>
sfpi_inline sfpi::vFloat horner_step(sfpi::vFloat acc, sfpi::vFloat x)
{
    if constexpr (K < 0)
    {
        return acc;
    }
    else
    {
        acc = fma_const(acc, x, ARR[BASE + static_cast<std::uint32_t>(K)]);
        return horner_step<ARR, BASE, K - 1>(acc, x);
    }
}

// Horner-eval a polynomial from constexpr array ARR, segment SEG, given degree
// DEG and per-segment stride DEG+1, all with compile-time indices.
template <const auto& ARR, std::uint32_t SEG, std::uint32_t DEG>
sfpi_inline sfpi::vFloat eval_poly_seg(sfpi::vFloat x)
{
    constexpr std::uint32_t base = SEG * (DEG + 1);
    const sfpi::vFloat      acc  = sfpi::vFloat(ARR[base + DEG]);
    return horner_step<ARR, base, static_cast<int>(DEG) - 1>(acc, x);
}

// Cumulative segment-override chain via template recursion over compile-time SEG
// (unrolled because RAT_NUM_SEGMENTS is constexpr). Each segment's numerator and
// denominator are evaluated on the clamped x, then selected per lane by the
// boundary compare. Mirrors the poly kernel's select_segment, carrying both P and
// Q.
template <std::uint32_t SEG>
sfpi_inline void select_segment(sfpi::vFloat& p, sfpi::vFloat& q, sfpi::vFloat x_clamped)
{
    if constexpr (SEG < RAT_NUM_SEGMENTS)
    {
        const sfpi::vFloat b_seg  = sfpi::vFloat(RAT_BOUNDARIES[SEG]);
        const sfpi::vFloat segp   = eval_poly_seg<RAT_NUM_COEFFS, SEG, RAT_NUM_DEGREE>(x_clamped);
        const sfpi::vFloat segq   = eval_poly_seg<RAT_DEN_COEFFS, SEG, RAT_DEN_DEGREE>(x_clamped);
        v_if (x_clamped >= b_seg)
        {
            p = segp;
            q = segq;
        }
        v_endif;
        select_segment<SEG + 1>(p, q, x_clamped);
    }
}

// Evaluate the injected piecewise-rational LUT on the current 2-row Dest window
// (sfpi dst_reg[0]). Clamp x to [b0, bN], select segment by boundaries, then
// compute P(x) / Q(x) = P(x) * reciprocal(Q(x)). All coefficient reads use
// compile-time indices (see horner_step) so the LUT arrays are folded away.
sfpi_inline void piecewise_rational_lut_sfp_rows()
{
    sfpi::vFloat x = sfpi::dst_reg[0];

    // Clamp x to [b0, bN] with explicit v_if (matches the poly test's clamp).
    const sfpi::vFloat b_lo = sfpi::vFloat(RAT_BOUNDARIES[0]);
    const sfpi::vFloat b_hi = sfpi::vFloat(RAT_BOUNDARIES[RAT_NUM_SEGMENTS]);

    sfpi::vFloat x_clamped = x;
    v_if (x_clamped < b_lo)
    {
        x_clamped = b_lo;
    }
    v_endif;
    v_if (x_clamped > b_hi)
    {
        x_clamped = b_hi;
    }
    v_endif;

    // Numerator and denominator of segment 0, then override with the correct
    // segment via the template-recursive cumulative chain (see select_segment).
    sfpi::vFloat p = eval_poly_seg<RAT_NUM_COEFFS, 0, RAT_NUM_DEGREE>(x_clamped);
    sfpi::vFloat q = eval_poly_seg<RAT_DEN_COEFFS, 0, RAT_DEN_DEGREE>(x_clamped);
    select_segment<1>(p, q, x_clamped);

    // result = P(x) / Q(x) = P(x) * (1 / Q(x)) using the iterative sfpi
    // reciprocal (silicon-representative; see file header).
    //
    // Note: _sfpu_reciprocal_ does the Newton-Raphson step "x*y - vConstFloatPrgm0"
    // (vConstFloatPrgm0 == 2.0). At -O3 the Tensix instruction-combine pass folds
    // this into an SFPADDI immediate, which the Quasar ttsim build implements only
    // as a throwing stub (MissingSpecification: tensix_execute_sfpaddi); high-degree
    // single-segment configs (e.g. n10d10_s1) emit it and abort. The Python test
    // compiles this source with -mno-tt-tensix-optimize-combine so the subtract
    // stays a register-operand SFPADD instead — same numerics, ttsim-supported.
    sfpi::vFloat inv_q = _sfpu_reciprocal_<2>(q);
    sfpi::dst_reg[0]   = p * inv_q;
}

template <int ITERATIONS = SFPU_ITERATIONS>
inline void calculate_rational_lut()
{
// #pragma GCC unroll 1 (not 8): the poly kernel found unroll-8 mis-schedules the
// SFPU iteration loop at high segment counts; keep it at 1 here for the same
// reason.
#pragma GCC unroll 1
    for (int d = 0; d < ITERATIONS; d++)
    {
        piecewise_rational_lut_sfp_rows();
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

    // Program vConstFloatPrgm0 = 2.0f, used by the Newton-Raphson step in
    // _sfpu_reciprocal_ (APPROXIMATION_MODE = false -> full NR refinement).
    ckernel::sfpu::_init_sfpu_reciprocal_<false>();

    // Apply the embedded piecewise-rational LUT in-place on Dest for each tile.
    for (std::uint32_t i = 0; i < params.TILE_CNT; ++i)
    {
        _llk_math_eltwise_unary_sfpu_params_(calculate_rational_lut<SFPU_ITERATIONS>, params.DST_INDEX + i);
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
