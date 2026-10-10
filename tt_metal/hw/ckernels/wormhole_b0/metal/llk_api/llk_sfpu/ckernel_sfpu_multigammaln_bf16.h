// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "ckernel_sfpu_bf16_core_bridge_rational.h"
#if defined(RANGE_REDUCTION_EXP) || defined(RANGE_REDUCTION_LOG) || defined(RANGE_REDUCTION_TRIG) || \
    defined(RATIONAL_DEN_PARITY_EVEN) || defined(RATIONAL_NUM_PARITY_ODD)
#error "aggregate cannot inherit alternate numerical selectors"
#endif
namespace ckernel::sfpu::MultigammalnBf16Config_source {
using namespace sfpi;
constexpr uint32_t NUM_DEGREE = 6, DEN_DEGREE = 4;
constexpr uint32_t NUM_SEGMENTS = 1, LUT_SIZE = 14;
constexpr std::array<float, LUT_SIZE> LUT_DATA = {
    {1.5009765625000000e+00f,
     5.0000000000000000e+01f,
     6.4772596359252930e+00f,
     -9.3999099731445312e+00f,
     2.1342811584472656e+00f,
     2.6674137115478516e+00f,
     -1.6923900842666626e+00f,
     3.1229573488235474e-01f,
     1.2563235359266400e-03f,
     -2.9412943124771118e-01f,
     1.0000000000000000e+00f,
     -8.9121544361114502e-01f,
     1.9727556407451630e-01f,
     2.6383467018604279e-02f}};

// Certified selected rational core plus aggregate Stirling tail.
constexpr uint32_t TT_SELECTED_STIRLING_CORE_LOWER_BITS = 0x3fc00000u;
constexpr uint32_t TT_SELECTED_STIRLING_CORE_UPPER_BITS = 0x42480000u;
constexpr uint32_t TT_SELECTED_STIRLING_OVERFLOW_BITS = 0x7b47f955u;
constexpr uint32_t TT_SELECTED_STIRLING_AGGREGATE_COUNT = 4u;
constexpr uint32_t TT_SELECTED_STIRLING_LOG_OFFSET_BITS = 0x40a00000u;
constexpr uint32_t TT_SELECTED_STIRLING_CONSTANT_BITS = 0x40e384a9u;
constexpr uint32_t TT_SELECTED_STIRLING_CORRECTION_SCALE_BITS = 0x41000000u;
constexpr std::array<float, 11> TT_SELECTED_STIRLING_LOG_RATIO = {
    {1.0000000000000000e+00f,
     -4.9999997019767761e-01f,
     3.3333346247673035e-01f,
     -2.5000315904617310e-01f,
     1.9999562203884125e-01f,
     -1.6651614010334015e-01f,
     1.4271874725818634e-01f,
     -1.2755781412124634e-01f,
     1.1764620989561081e-01f,
     -9.1233491897583008e-02f,
     3.7215668708086014e-02f}};
constexpr std::array<float, 13> TT_SELECTED_STIRLING_CORRECTION = {
    {2.7755575615628914e-17f,
     4.4791665673255920e-01f,
     2.9296875000000000e-02f,
     3.0097113922238350e-03f,
     3.7765502929687500e-04f,
     5.2835079259239137e-05f,
     7.8976299846544862e-06f,
     1.2328968068686663e-06f,
     1.9902516612546606e-07f,
     3.1603722305817428e-08f,
     6.6526548714307410e-09f,
     1.2773651580921808e-10f,
     4.3693662576949066e-10f}};
constexpr uint32_t TT_SELECTED_STIRLING_REFLECTION_ORIGIN_BITS = 0x40200000u;
constexpr uint32_t TT_SELECTED_STIRLING_REFLECTION_CONSTANT_BITS = 0x414d5666u;
constexpr uint32_t TT_SELECTED_STIRLING_REFLECTION_SHIFT = 7u;
constexpr uint32_t TT_SELECTED_STIRLING_REFLECTION_LOG_SCALE_BITS = 0x3f800000u;
constexpr std::array<float, 12> TT_SELECTED_STIRLING_REFLECTION_SQUARED_OFFSETS = {
    {-5.0000000000000000e-01f,
     0.0000000000000000e+00f,
     5.0000000000000000e-01f,
     1.0000000000000000e+00f,
     1.5000000000000000e+00f,
     2.0000000000000000e+00f,
     2.5000000000000000e+00f,
     3.0000000000000000e+00f,
     3.5000000000000000e+00f,
     4.0000000000000000e+00f,
     4.5000000000000000e+00f,
     5.0000000000000000e+00f}};
constexpr std::array<float, 4> TT_SELECTED_STIRLING_REFLECTION_SINGLE_OFFSETS = {
    {-1.5000000000000000e+00f, -1.0000000000000000e+00f, 5.5000000000000000e+00f, 6.0000000000000000e+00f}};
constexpr std::array<float, 7> TT_SELECTED_STIRLING_LOG_SINE_RATIO = {
    {2.2894597053527832e+00f,
     -3.2898662090301514e+00f,
     -1.0824505090713501e+00f,
     -6.7527395486831665e-01f,
     -5.3421139717102051e-01f,
     -2.2606460750102997e-01f,
     -7.6610094308853149e-01f}};
constexpr std::array<uint32_t, 2> TT_SELECTED_STIRLING_REFLECTION_STOCK_WINDOW_BITS = {{0xbb600000u, 0x3b200000u}};

#include "ckernel_sfpu_bf16_aggregate_stirling_core.inc"
#include "ckernel_sfpu_bf16_rational_interleaved_core.inc"
template <uint32_t N, uint32_t D>
inline void eval_rational_numer_denom(const float* n, const float* d, vFloat x, vFloat& a, vFloat& b) {
    eval_rational_interleaved_numer_denom<N, D>(n, d, x, a, b);
}
#include "ckernel_sfpu_bf16_rational_segment_core.inc"
inline vFloat prepare_raw_domain_input(vFloat x) { return selected_aggregate_stirling_prepare(x); }
inline vFloat apply_output_postcompose(vFloat y, vFloat) { return y; }
#include "ckernel_sfpu_bf16_domain_action_coordinate.inc"
#include "ckernel_sfpu_bf16_domain_finalize.inc"
// Upstream LLK runs several tiles in one DEST half, so the coefficients are
// immediates rather than rows parked past this tile.
inline void tile() {
    constexpr uint32_t NUM_COEFFS = NUM_DEGREE + 1, COEFF_OFFSET = NUM_SEGMENTS + 1;
    const auto& lut = LUT_DATA;
#include "ckernel_sfpu_bf16_rational_scalar_tile.inc"
}
}  // namespace ckernel::sfpu::MultigammalnBf16Config_source
namespace ckernel::sfpu {
struct MultigammalnBf16Config {
    static inline void tile() { MultigammalnBf16Config_source::tile(); }
};
}  // namespace ckernel::sfpu

namespace ckernel::sfpu {

template <int ITERATIONS = 8>
inline void calculate_multigammaln_bf16() {
    ckernel::sfpu::bf16::calculate_core_bridge_rational<ckernel::sfpu::MultigammalnBf16Config, ITERATIONS>();
}
inline void init_multigammaln_bf16() {
    ckernel::sfpu::bf16::init_core_bridge_rational<ckernel::sfpu::MultigammalnBf16Config>();
}

}  // namespace ckernel::sfpu
