// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// =============================================================================
// SFPU Optimization Story — Rung R2: INTERLEAVED Horner Rational P(x)/Q(x)
// =============================================================================
// Drop-in replacement for the adhoc slot (kernels/compute/adhoc/adhoc.cpp).
// Reuses the SAME host/CB/LLK scaffolding as the production embedded kernel:
//   - same #includes
//   - same `namespace sfpi`
//   - same `kernel_main()` (copied verbatim from piecewise_generic.cpp)
//   - same dispatch entry point: sfpi::piecewise_generic_lut_dispatch<...>(*p_lut)
//
// R2 interleaves the numerator and denominator Horner chains: instead of fully
// evaluating P(x) and then Q(x), it advances both chains in LOCKSTEP, alternating
// the two FMAs each step (compile-time recursion over the shared max degree). The
// numerator and denominator chains are data-independent, so back-to-back SFPMADs
// on the two chains keep the SFPU pipeline full and hide its latency. With n8d8
// the numerator and denominator degrees are equal, giving a perfectly balanced,
// branch-free interleave. The per-segment cascade and per-segment reciprocal are
// retained from the naive baseline.
//
// Consumes the shared benchmark rational LUT from ../common/bench_rational_lut.h.
// LUT layout: [R_NUM_SEG+1 boundaries][per-seg: num c0..c8 (9 floats)][per-seg: den c0..c8 (9 floats)]
// =============================================================================

#include <cstdint>
#include <array>

// Compute API + SFPU (sfpi) types — same includes the production kernel uses.
#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"

// ----------------------------------------------------------------------------
// Benchmark rational LUT (constexpr): BENCH_R_NUM_SEGMENTS, BENCH_R_NUM_DEGREE,
// BENCH_R_DEN_DEGREE, BENCH_R_LUT_SIZE, BENCH_R_LUT, ...
// Include path is relative to the adhoc slot (../../.. climbs to the example root).
// ----------------------------------------------------------------------------
#include "../../../tutorials/sfpu_optimization/kernels/common/bench_rational_lut.h"

// ----------------------------------------------------------------------------
// Alias the BENCH_R_* names onto the names the shared kernel_main() expects.
// The polynomial scaffold's kernel_main() reads POLY_DEGREE / NUM_SEGMENTS /
// LUT_SIZE / LUT_DATA. For the rational rung we drive the same scaffold, so the
// max degree alias maps onto the larger of the numerator/denominator degrees.
// ----------------------------------------------------------------------------
#define EMBEDDED_LUT
constexpr uint32_t POLY_DEGREE = (BENCH_R_NUM_DEGREE > BENCH_R_DEN_DEGREE) ? BENCH_R_NUM_DEGREE : BENCH_R_DEN_DEGREE;
constexpr uint32_t NUM_SEGMENTS = BENCH_R_NUM_SEGMENTS;
constexpr uint32_t LUT_SIZE = BENCH_R_LUT_SIZE;
constexpr auto& LUT_DATA = BENCH_R_LUT;

// ============================================================================
// SFPU compute (math TRISC only)
// ============================================================================
#ifdef TRISC_MATH

// Reciprocal SFPU function (Newton-Raphson). sfpu_reciprocal_iter<N> lives here.
#include "ckernel_sfpu_recip.h"

namespace sfpi {
template <int K>
__attribute__((always_inline)) static inline void hr2(
    const float* pc, const float* qc, vFloat x, vFloat& p, vFloat& q) {
    if constexpr (K >= 0) {
        p = p * x + pc[K];
        q = q * x + qc[K];
        hr2<K - 1>(pc, qc, x, p, q);
    }
}
template <uint32_t S, uint32_t NUM_SEGMENTS, uint32_t LUT_SIZE>
__attribute__((always_inline)) static inline void seg(const std::array<float, LUT_SIZE>& lut, vFloat x, vFloat& r) {
    if constexpr (S < NUM_SEGMENTS) {
        constexpr uint32_t ND = BENCH_R_NUM_DEGREE, DD = BENCH_R_DEN_DEGREE;
        constexpr uint32_t STRIDE = (ND + 1) + (DD + 1);
        constexpr uint32_t nb = (NUM_SEGMENTS + 1) + S * STRIDE;
        constexpr uint32_t db = nb + (ND + 1);
        constexpr int TOP = (int)((ND > DD) ? ND : DD);
        v_if(x >= lut[S]) {
            vFloat p = lut[nb + ND];
            vFloat q = lut[db + DD];
            hr2<TOP - 1>(&lut[nb], &lut[db], x, p, q);
            r = p * ckernel::sfpu::sfpu_reciprocal_iter<3>(q);
        }
        v_endif;
        seg<S + 1, NUM_SEGMENTS, LUT_SIZE>(lut, x, r);
    }
}
template <uint32_t POLY_DEGREE, uint32_t NUM_SEGMENTS, uint32_t LUT_SIZE>
inline void piecewise_generic_lut_dispatch(const std::array<float, LUT_SIZE>& lut) {
    for (int d = 0; d < 32; d++) {
        vFloat x = dst_reg[d];
        vFloat r = 0.0f;
        seg<0, NUM_SEGMENTS, LUT_SIZE>(lut, x, r);
        dst_reg[d] = r;
    }
}
}  // namespace sfpi
#endif  // TRISC_MATH

// ============================================================================
// Host-facing compute-kernel entry — copied VERBATIM from piecewise_generic.cpp
// (EMBEDDED_LUT path). CB / LLK / tile-regs scaffolding is identical across all
// rungs; only the eval body above changes.
// ============================================================================
void kernel_main() {
    uint32_t n_tiles = get_arg_val<uint32_t>(0);

    // Embedded LUT mode: LUT is compiled directly into the kernel.
    constexpr auto cb_in = tt::CBIndex::c_0;
    constexpr auto cb_out = tt::CBIndex::c_16;

    constexpr uint32_t poly_degree = POLY_DEGREE;
    constexpr uint32_t num_segments = NUM_SEGMENTS;
    constexpr uint32_t lut_size = LUT_SIZE;
    const auto& lut_ref = LUT_DATA;
    auto p_lut = &lut_ref;

    init_sfpu(cb_in, cb_out);

    for (uint32_t tile = 0; tile < n_tiles; tile++) {
        cb_wait_front(cb_in, 1);
        tile_regs_acquire();
        copy_tile(cb_in, 0, 0);

#ifdef TRISC_MATH
        sfpi::piecewise_generic_lut_dispatch<poly_degree, num_segments, lut_size>(*p_lut);
#endif

        tile_regs_commit();
        tile_regs_wait();
        cb_reserve_back(cb_out, 1);
        pack_tile(0, cb_out);
        cb_push_back(cb_out, 1);
        cb_pop_front(cb_in, 1);
        tile_regs_release();
    }
}
