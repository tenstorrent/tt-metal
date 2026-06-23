// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// Generic piecewise-polynomial LUT activation on Quasar with PARITY x^2-HORNER
// and ADAPTIVE PER-SEGMENT DEGREE.
//
// Sibling of generic_lut_activation_quasar_test.cpp. It keeps the same
// UNPACK -> SrcA -> FPU datacopy (MOVA2D) -> Dest -> SFPU(embedded LUT) -> PACK
// recipe and the same Quasar 2-rows-per-iteration SFPU model, but ports two
// PURE-EVALUATOR features from the Blackhole/embedded kernels
// (piecewise_generic.cpp / piecewise_generic_specialized.cpp):
//
//   * PARITY x^2-HORNER  (POLY_PARITY_ODD / POLY_PARITY_EVEN)
//       Polynomials with a known parity have half their coefficients equal to
//       zero (odd: c0=c2=...=0; even: c1=c3=...=0). They are evaluated in the
//       x^2 basis with stride-2 coefficient access, halving the Horner FMA
//       count:
//         odd:  P(x) = x * Horner([c1,c3,c5,...], x^2)
//         even: P(x) =     Horner([c0,c2,c4,...], x^2)
//       This is the SAME concept as the embedded eval_polynomial_parity<DEG>.
//       It is DISTINCT from the trig/tan argument-parity sign flip (rr_parity_f)
//       in the RR poly test — that flips the result sign by k's parity; this
//       exploits a parity-constrained COEFFICIENT structure to skip FMAs.
//
//   * ADAPTIVE PER-SEGMENT DEGREE  (HAS_SEGMENT_DEGREES + SEGMENT_DEGREES[])
//       The LUT always stores POLY_DEGREE+1 coefficients per segment (uniform
//       stride), but each segment's EFFECTIVE degree can be lower. When the
//       build header defines SEGMENT_DEGREES_INIT, SEGMENT_DEGREES[SEG] gives the
//       per-segment degree. Because both SEGMENT_DEGREES (constexpr array) and
//       SEG (template parameter) are compile-time, SEGMENT_DEGREES[SEG] is a
//       valid template argument -> each segment's Horner unrolls to exactly its
//       own degree. Without the define every segment falls back to POLY_DEGREE.
//
// These are evaluator-only: they add NO new SFPU instruction. All multiply-adds
// go through the fused-SFPMAD helper (fma_const) so the backend never folds a
// constant SFPADD into SFPADDI (which Quasar removed). Coefficients are read
// only through COMPILE-TIME indices so LUT_DATA constant-folds away and never
// lands in the tiny TRISC1 private data region.
//
// No range reduction here (LUT_RR_METHOD is implicitly 0): parity_adaptive is
// orthogonal to RR, and the RR machinery already lives in the sibling poly test.

#include <array>
#include <cstdint>
#include <limits>

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

using namespace ckernel;
using namespace ckernel::math;
using namespace ckernel::sfpu;

// =====================================================================
// Embedded piecewise-polynomial LUT. Generic in degree N (POLY_DEGREE) and
// segment count S (NUM_SEGMENTS). Layout: [b0..bS (S+1 ascending boundaries),
// then per segment (N+1) Horner coeffs c0..cN]. Defaults below replicate the
// proven sigmoid LUT (deg-2, 4-seg); the python test overrides them via the
// build header (params.h -> build.h), emitting LUT_POLY_DEGREE /
// LUT_NUM_SEGMENTS / LUT_DATA_INIT (and, for parity_adaptive, POLY_PARITY_* and
// SEGMENT_DEGREES_INIT). Defined here (after build.h is in scope) because the
// MATH thread is the only consumer.
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

// ---- Adaptive per-segment degree -------------------------------------------
// When the build header defines SEGMENT_DEGREES_INIT, SEGMENT_DEGREES[SEG] is
// the effective (per-segment) Horner degree; otherwise every segment uses the
// nominal POLY_DEGREE. SEGMENT_DEGREES must be a namespace-scope constexpr (it
// is indexed by the SEG template parameter, so SEGMENT_DEGREES[SEG] must itself
// be a constant expression usable as a template argument).
#ifdef SEGMENT_DEGREES_INIT
#define HAS_SEGMENT_DEGREES 1
constexpr std::array<std::uint32_t, NUM_SEGMENTS> SEGMENT_DEGREES = SEGMENT_DEGREES_INIT;
#endif

namespace
{
// SFPADDI avoidance. The Horner step `acc * x + c` lowers in sfpi to a separate
// SFPMUL + SFPADD. When `c` is a compile-time constant the backend fuses the
// SFPADD with the constant's SFPLOADI into a single SFPADDI for round values —
// and Quasar removed SFPADDI. Emitting the multiply-add as one fused SFPMAD
// makes the constant the MAD's addend operand, so there is no standalone SFPADD
// for the backend to fold into SFPADDI.
sfpi_inline sfpi::vFloat fma_const(sfpi::vFloat a, sfpi::vFloat b, float c)
{
    const sfpi::vFloat cv = sfpi::vFloat(c);
    return __builtin_rvtt_sfpmad(a.get(), b.get(), cv.get(), sfpi::SFPMAD_MOD1_OFFSET_NONE);
}

// ---------------------------------------------------------------------------
// Per-segment effective degree (compile-time). Indexed by the SEG template
// parameter so it is usable as a template argument.
// ---------------------------------------------------------------------------
template <std::uint32_t SEG>
constexpr std::uint32_t seg_degree()
{
#ifdef HAS_SEGMENT_DEGREES
    return SEGMENT_DEGREES[SEG];
#else
    return POLY_DEGREE;
#endif
}

// ---------------------------------------------------------------------------
// Horner recursion (full, natural-basis). Reads LUT_DATA only through constexpr
// indices so every coefficient constant-folds into an SFPLOADI immediate and the
// LUT_DATA array is elided (it never lands in the tiny TRISC1 private data
// region). Applies coefficients c_K down to c_0.
// ---------------------------------------------------------------------------
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

#if defined(POLY_PARITY_ODD) || defined(POLY_PARITY_EVEN)
// ---------------------------------------------------------------------------
// PARITY x^2-Horner recursion. Steps by 2 in the coefficient index (x^2 basis),
// stopping once the index drops below the parity's lowest live coefficient.
//   odd  parity: live coeffs are c1,c3,c5,...  (LOW = 1)
//   even parity: live coeffs are c0,c2,c4,...  (LOW = 0)
// IDX walks down from TOP to LOW in strides of 2; each step is a fused SFPMAD
// (no SFPADDI) reading a COMPILE-TIME coefficient index (so LUT_DATA folds away).
// ---------------------------------------------------------------------------
template <std::uint32_t BASE, int IDX, int LOW>
sfpi_inline sfpi::vFloat horner_step_x2(sfpi::vFloat acc, sfpi::vFloat x2)
{
    if constexpr (IDX < LOW)
    {
        return acc;
    }
    else
    {
        acc = fma_const(acc, x2, LUT_DATA[BASE + static_cast<std::uint32_t>(IDX)]);
        return horner_step_x2<BASE, IDX - 2, LOW>(acc, x2);
    }
}

// eval the parity polynomial of effective degree DEG at base BASE.
// x2 must be x*x (the per-segment-clamped argument squared).
template <std::uint32_t BASE, std::uint32_t DEG>
sfpi_inline sfpi::vFloat eval_parity([[maybe_unused]] sfpi::vFloat x, sfpi::vFloat x2)
{
#if defined(POLY_PARITY_ODD)
    // TOP = highest odd index <= DEG ; LOW = 1 ; result *= x at the end.
    constexpr int TOP = (DEG % 2 == 1) ? static_cast<int>(DEG) : static_cast<int>(DEG) - 1;
    static_assert(TOP >= 1, "odd-parity segment has no live coefficient (DEG < 1)");
    const sfpi::vFloat seed = sfpi::vFloat(LUT_DATA[BASE + static_cast<std::uint32_t>(TOP)]);
    const sfpi::vFloat poly = horner_step_x2<BASE, TOP - 2, 1>(seed, x2);
    // final * x via fused SFPMAD (acc*x + 0) to avoid a standalone SFPMUL/SFPADD.
    return __builtin_rvtt_sfpmad(poly.get(), x.get(), sfpi::vFloat(0.0f).get(), sfpi::SFPMAD_MOD1_OFFSET_NONE);
#else // POLY_PARITY_EVEN
    // TOP = highest even index <= DEG ; LOW = 0 ; no final * x.
    constexpr int TOP = (DEG % 2 == 0) ? static_cast<int>(DEG) : static_cast<int>(DEG) - 1;
    static_assert(TOP >= 0, "even-parity segment has no live coefficient");
    const sfpi::vFloat seed = sfpi::vFloat(LUT_DATA[BASE + static_cast<std::uint32_t>(TOP)]);
    return horner_step_x2<BASE, TOP - 2, 0>(seed, x2);
#endif
}
#endif // parity

// eval_seg<SEG>: clamp x to each segment's own sub-interval [b_SEG, b_{SEG+1}]
// (top-level v_if, never nested — a nested v_if corrupts the Quasar predicate
// stack), then Horner-eval segment SEG at its EFFECTIVE degree. With parity the
// x^2 basis is used; otherwise the natural-basis horner_step.
template <std::uint32_t SEG>
sfpi_inline sfpi::vFloat eval_seg(sfpi::vFloat x_clamped)
{
    constexpr std::uint32_t COEFFS_PER_SEGMENT = POLY_DEGREE + 1;  // uniform LUT stride
    constexpr std::uint32_t COEFF_OFFSET       = NUM_SEGMENTS + 1; // skip boundaries
    constexpr std::uint32_t base               = COEFF_OFFSET + SEG * COEFFS_PER_SEGMENT;
    constexpr std::uint32_t DEG                = seg_degree<SEG>();

    const sfpi::vFloat seg_lo = sfpi::vFloat(LUT_DATA[SEG]);
    const sfpi::vFloat seg_hi = sfpi::vFloat(LUT_DATA[SEG + 1]);
    sfpi::vFloat xs           = x_clamped;
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

#if defined(POLY_PARITY_ODD) || defined(POLY_PARITY_EVEN)
    // x^2 of the per-segment-clamped argument (clamp affects x, so recompute).
    const sfpi::vFloat xs2 = __builtin_rvtt_sfpmad(xs.get(), xs.get(), sfpi::vFloat(0.0f).get(), sfpi::SFPMAD_MOD1_OFFSET_NONE);
    return eval_parity<base, DEG>(xs, xs2);
#else
    const sfpi::vFloat acc = sfpi::vFloat(LUT_DATA[base + DEG]);
    return horner_step<base, static_cast<int>(DEG) - 1>(acc, xs);
#endif
}

// Cumulative segment-override chain via template recursion over compile-time SEG
// (unrolled because NUM_SEGMENTS is constexpr). Each segment's (already-bounded)
// evaluation is computed BEFORE the select v_if so its clamp v_ifs stay at top
// level — a nested v_if corrupts the Quasar predicate stack.
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
// Horner-eval that segment at its effective degree (parity x^2-Horner when a
// parity is declared). See the per-segment-clamp rationale in the sibling poly
// test: clamping the Horner argument to each segment's own sub-interval keeps
// every RHS bounded so the cumulative select cannot be poisoned by a poly
// evaluated far outside its fit range.
sfpi_inline void piecewise_parity_lut_sfp_rows()
{
    const sfpi::vFloat x_in = sfpi::dst_reg[0];

    // ---- Clamp x to [b0, bN] (explicit v_if: vec_min_max semantics differ).
    const sfpi::vFloat b_lo = sfpi::vFloat(LUT_DATA[0]);
    const sfpi::vFloat b_hi = sfpi::vFloat(LUT_DATA[NUM_SEGMENTS]);

    sfpi::vFloat x_clamped = x_in;
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

    sfpi::vFloat result = eval_seg<0>(x_clamped);
    select_segment<1>(result, x_clamped);

    sfpi::dst_reg[0] = result;
}

template <int ITERATIONS = SFPU_ITERATIONS>
inline void calculate_parity_lut()
{
#pragma GCC unroll 1
    for (int d = 0; d < ITERATIONS; d++)
    {
        piecewise_parity_lut_sfp_rows();
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
        _llk_math_eltwise_unary_sfpu_params_(calculate_parity_lut<SFPU_ITERATIONS>, params.DST_INDEX + i);
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
