// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// Generic piecewise-polynomial LUT activation on Quasar.
//
// Models the eltwise_unary_sfpu_quasar recipe (UNPACK -> SrcA -> FPU datacopy
// (MOVA2D) -> Dest -> SFPU -> PACK), but replaces the canned unary SFPU op with
// an embedded piecewise-polynomial LUT evaluated in the SFPU using the sfpi DSL.
//
// The LUT is a generic piecewise polynomial of arbitrary degree N (POLY_DEGREE)
// and arbitrary segment count S (NUM_SEGMENTS). Layout matches the BH/WH custom
// test:
//   [b0..bS  (NUM_SEGMENTS+1 ascending boundaries),
//    then per segment (POLY_DEGREE+1) Horner coeffs c0..cN]
// No range reduction. The coefficients/boundaries are baked in at build time.
// They DEFAULT to the proven sigmoid LUT (deg-2, 4-seg) below, but the python
// test overrides them via the build header (the GENERIC_LUT_DATA template
// parameter emits LUT_POLY_DEGREE / LUT_NUM_SEGMENTS / LUT_DATA_INIT). The
// Python golden replicates these EXACT values so the PCC isolates kernel
// correctness.
//
// Key Quasar-LLK differences vs the BH/WH custom test:
//   * Includes: Quasar MATH-thread headers (cfg_defines.h, cmath_common.h,
//     llk_math_common.h, llk_math_eltwise_unary_datacopy.h,
//     llk_math_eltwise_unary_sfpu.h) + the eltwise_unary recipe's UNPACK/PACK
//     headers, NOT the WH/BH llk_unpack_A.h / llk_pack.h set.
//   * Iteration model: the SFPU iterates 2 Dest rows per iteration
//     (SFP_ROWS == 2) over SFPU_ITERATIONS, advancing the Dest window with
//     ckernel::math::_incr_counters_<...SFP_ROWS...>() — NOT a 0..31 dst_reg
//     loop.
//   * Polynomial body uses sfpi dst_reg[0] / vFloat / v_if and float literal
//     constants (Converter::as_float does not exist on Quasar).

#include <array>
#include <cstdint>

#include "ckernel.h"
#include "llk_defs.h"
#include "llk_memory_checks.h"
#include "sfpu_stub.h"

// The embedded piecewise-polynomial LUT (degree N, segment count S) is defined
// inside the LLK_TRISC_MATH block below — that is the only thread that evaluates
// it, and it is the only place where the auto-generated build header (params.h
// -> build.h, carrying the python test's LUT overrides) is in scope.

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

// =====================================================================
// Embedded piecewise-polynomial LUT. Generic in degree N (POLY_DEGREE) and
// segment count S (NUM_SEGMENTS). Layout: [b0..bS (S+1 ascending boundaries),
// then per segment (N+1) Horner coeffs c0..cN]. Defaults below replicate the
// proven sigmoid LUT (deg-2, 4-seg); the python test overrides them via the
// build header (params.h -> build.h), emitting LUT_POLY_DEGREE /
// LUT_NUM_SEGMENTS / LUT_DATA_INIT. Defined here (after build.h is in scope)
// because the MATH thread is the only consumer.
// =====================================================================
#ifndef LUT_POLY_DEGREE
#define LUT_POLY_DEGREE 2
#endif
#ifndef LUT_NUM_SEGMENTS
#define LUT_NUM_SEGMENTS 4
#endif
#ifndef LUT_DATA_INIT
// clang-format off
#define LUT_DATA_INIT {                       \
    /* boundaries b0..b4 */                   \
    -4.0f, -2.0f, 0.0f, 2.0f, 4.0f,           \
    /* seg0 coeffs: c0, c1, c2 */             \
    0.38296354f, 0.17515847f, 0.02109685f,    \
    /* seg1 */                                \
    0.50329190f, 0.27505103f, 0.04113654f,    \
    /* seg2 */                                \
    0.49670810f, 0.27505103f, -0.04113654f,   \
    /* seg3 */                                \
    0.61703646f, 0.17515847f, -0.02109685f }
// clang-format on
#endif

constexpr std::uint32_t POLY_DEGREE  = LUT_POLY_DEGREE;
constexpr std::uint32_t NUM_SEGMENTS = LUT_NUM_SEGMENTS;
constexpr std::uint32_t LUT_SIZE     = (NUM_SEGMENTS + 1) + NUM_SEGMENTS * (POLY_DEGREE + 1);

constexpr std::array<float, LUT_SIZE> LUT_DATA = LUT_DATA_INIT;

namespace
{
// SFPADDI avoidance (FIX 3). The Horner step `acc * x + c` lowers in sfpi to a
// separate SFPMUL + SFPADD. When `c` is a compile-time constant (LUT_DATA is
// constexpr) the backend fuses the SFPADD with the constant's SFPLOADI into a
// single SFPADDI for round values (e.g. 0.5 = 0x3F000000) — and Quasar removed
// SFPADDI (ttsim aborts "tensix_execute_sfpaddi"). Emitting the multiply-add as
// one fused SFPMAD instead means the constant is the MAD's third (addend)
// operand, so there is no standalone SFPADD for the backend to fold into SFPADDI.
sfpi_inline sfpi::vFloat fma_const(sfpi::vFloat a, sfpi::vFloat b, float c)
{
    const sfpi::vFloat cv = sfpi::vFloat(c);
    return __builtin_rvtt_sfpmad(a.get(), b.get(), cv.get(), sfpi::SFPMAD_MOD1_OFFSET_NONE);
}

// Horner step recursion with COMPILE-TIME coefficient indices (FIX 4). Reading
// LUT_DATA only through constexpr indices (never `&LUT_DATA[i]` with a runtime
// index) lets the compiler constant-fold every coefficient into an SFPLOADI
// immediate and ELIDE the LUT_DATA array — so it never lands in .ldm_data and
// cannot overflow the tiny 1792 B private TRISC1_LOCAL_DATA_MEM region (the
// "section .ldm_data not within region" error on large deg/seg configs). The
// folded immediates live in code (16 KiB TRISC1_CODE), which has room. (Placing
// LUT_DATA in an L1 `l1_data` section instead corrupts the SFPU reads on this
// Quasar sim, so folding it away is the robust fix.) Applies coefficients c_K
// down to c_0.
template <std::uint32_t BASE, int K>
sfpi_inline sfpi::vFloat horner_step(sfpi::vFloat acc, sfpi::vFloat x)
{
    if constexpr (K < 0)
    {
        return acc;
    }
    else
    {
        acc = fma_const(acc, x, LUT_DATA[BASE + static_cast<std::uint32_t>(K)]);
        return horner_step<BASE, K - 1>(acc, x);
    }
}

// eval_seg<SEG>: clamp x to [b_SEG, b_{SEG+1}] (top-level v_if) and Horner-eval
// segment SEG with compile-time coefficient indices, so the polynomial is never
// evaluated outside its fit range and LUT_DATA is constant-folded away.
template <std::uint32_t SEG>
sfpi_inline sfpi::vFloat eval_seg(sfpi::vFloat x_clamped)
{
    constexpr std::uint32_t COEFFS_PER_SEGMENT = POLY_DEGREE + 1;
    constexpr std::uint32_t COEFF_OFFSET       = NUM_SEGMENTS + 1; // skip boundaries
    constexpr std::uint32_t base               = COEFF_OFFSET + SEG * COEFFS_PER_SEGMENT;

    const sfpi::vFloat seg_lo = sfpi::vFloat(LUT_DATA[SEG]);
    const sfpi::vFloat seg_hi = sfpi::vFloat(LUT_DATA[SEG + 1]);
    sfpi::vFloat       xs     = x_clamped;
    v_if (xs < seg_lo)
    {
        xs = seg_lo;
    }
    v_endif;
    v_if (xs > seg_hi)
    {
        xs = seg_hi;
    }
    v_endif;
    const sfpi::vFloat acc = sfpi::vFloat(LUT_DATA[base + POLY_DEGREE]);
    return horner_step<base, static_cast<int>(POLY_DEGREE) - 1>(acc, xs);
}

// Cumulative segment-override chain via template recursion over compile-time SEG
// (the loop is unrolled because NUM_SEGMENTS is constexpr). Each segment's
// (already-bounded) evaluation is computed BEFORE the select v_if so its clamp
// v_ifs stay at top level — a nested v_if corrupts the Quasar predicate stack.
template <std::uint32_t SEG>
sfpi_inline void select_segment(sfpi::vFloat& result, sfpi::vFloat x_clamped)
{
    if constexpr (SEG < NUM_SEGMENTS)
    {
        const sfpi::vFloat b_seg  = sfpi::vFloat(LUT_DATA[SEG]);
        const sfpi::vFloat segval = eval_seg<SEG>(x_clamped);
        v_if (x_clamped >= b_seg)
        {
            result = segval;
        }
        v_endif;
        select_segment<SEG + 1>(result, x_clamped);
    }
}

// Evaluate the embedded piecewise-polynomial LUT on the current 2-row Dest
// window (sfpi dst_reg[0]). Clamp x to [b0, bN], select segment by boundaries,
// Horner-eval that segment.
//
// CRITICAL (Quasar): sfpi evaluates the RHS of a predicated assignment on ALL
// 32 lanes; the v_if predicate only gates the final conditional MOV. The
// original kernel did `for seg: v_if (x >= b[seg]) result = eval(seg, x_clamped);`,
// i.e. it ran EVERY segment's Horner on every lane's GLOBALLY clamped x. For a
// high-degree poly evaluated far outside its fit interval (e.g. a [5,5.3]-fit
// deg-8 poly at x=10) that RHS overflows to ~1e36/inf. With many segments the
// late cumulative overrides on Quasar fail to fully overwrite those poisoned
// lanes, so the top segments read back ~1e37 (PCC craters; the s32 bug). The
// segment select itself was always correct.
//
// Fix: clamp the Horner argument to EACH segment's own sub-interval
// [b_seg, b_{seg+1}] before evaluating that segment, so no polynomial is ever
// evaluated outside its fit range -> every RHS stays bounded and finite, and the
// cumulative select can no longer be poisoned, independent of segment count.
// The per-segment clamp uses TOP-LEVEL v_if blocks (NOT nested inside the select
// v_if): a nested v_if corrupts the Quasar predicate stack and silently drops
// the override write.
sfpi_inline void piecewise_generic_lut_sfp_rows()
{
    sfpi::vFloat x = sfpi::dst_reg[0];

    // Clamp x to [b0, bN] with explicit v_if (vec_min_max semantics differ
    // from WH/BH, which silently pinned everything to the lower bound).
    const sfpi::vFloat b_lo = sfpi::vFloat(LUT_DATA[0]);
    const sfpi::vFloat b_hi = sfpi::vFloat(LUT_DATA[NUM_SEGMENTS]);

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

    // Start with segment 0, then override with the correct segment per lane via
    // the template-recursive cumulative chain (see select_segment).
    sfpi::vFloat result = eval_seg<0>(x_clamped);
    select_segment<1>(result, x_clamped);

    sfpi::dst_reg[0] = result;
}

template <int ITERATIONS = SFPU_ITERATIONS>
inline void calculate_generic_lut()
{
#pragma GCC unroll 1
    for (int d = 0; d < ITERATIONS; d++)
    {
        piecewise_generic_lut_sfp_rows();
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

    // Apply the embedded piecewise-polynomial LUT in-place on Dest for each tile.
    for (std::uint32_t i = 0; i < params.TILE_CNT; ++i)
    {
        _llk_math_eltwise_unary_sfpu_params_(calculate_generic_lut<SFPU_ITERATIONS>, params.DST_INDEX + i);
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
