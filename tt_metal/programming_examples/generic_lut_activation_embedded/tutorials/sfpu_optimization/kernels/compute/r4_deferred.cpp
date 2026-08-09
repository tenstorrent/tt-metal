// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// =============================================================================
// SFPU Optimization Story — Rung R4: DEFERRED RECIPROCAL Rational P(x)/Q(x)
// =============================================================================
// Drop-in replacement for the adhoc slot (kernels/compute/adhoc/adhoc.cpp).
// Reuses the SAME host/CB/LLK scaffolding as the production embedded kernel:
//   - same #includes
//   - same `namespace sfpi`
//   - same `kernel_main()` (copied verbatim from piecewise_generic.cpp)
//   - same dispatch entry point: sfpi::piecewise_generic_lut_dispatch<...>(*p_lut)
//
// R4 builds on the parity x²-Horner rung and adds DEFERRED RECIPROCAL: because
// v_if is predicated, every lane executes a per-segment reciprocal even when its
// segment is not selected, so a naive cascade pays one full Newton-Raphson
// reciprocal on ALL lanes for EVERY segment. Instead, the segment cascade merely
// SELECTS the numerator P(x) and denominator Q(x) for each lane's active
// segment (with a safe Q=1 default so unselected lanes never divide by zero),
// and then ONE reciprocal is computed OUTSIDE all v_ifs — collapsing N
// per-segment reciprocals down to a single sfpu_reciprocal_iter<3> per DST row.
// Numerator and denominator are still evaluated with parity x²-Horner (odd
// numerator via horner_odd, even denominator via horner_even).
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
__attribute__((always_inline)) static inline vFloat hodd(const float* c, vFloat t, vFloat acc) {
    if constexpr (K < 1) {
        return acc;
    } else {
        return hodd<K - 2>(c, t, acc * t + c[K]);
    }
}
template <int TOP>
__attribute__((always_inline)) static inline vFloat horner_odd(const float* c, vFloat t) {
    return hodd<TOP - 2>(c, t, vFloat(c[TOP]));
}
template <int K>
__attribute__((always_inline)) static inline vFloat heven(const float* c, vFloat t, vFloat acc) {
    if constexpr (K < 0) {
        return acc;
    } else {
        return heven<K - 2>(c, t, acc * t + c[K]);
    }
}
template <int TOP>
__attribute__((always_inline)) static inline vFloat horner_even(const float* c, vFloat t) {
    return heven<TOP - 2>(c, t, vFloat(c[TOP]));
}
template <uint32_t S, uint32_t NUM_SEGMENTS, uint32_t LUT_SIZE>
__attribute__((always_inline)) static inline void seg(
    const std::array<float, LUT_SIZE>& lut, vFloat x, vFloat t, vFloat& selP, vFloat& selQ) {
    if constexpr (S < NUM_SEGMENTS) {
        constexpr uint32_t ND = BENCH_R_NUM_DEGREE, DD = BENCH_R_DEN_DEGREE;
        constexpr uint32_t STRIDE = (ND + 1) + (DD + 1);
        constexpr uint32_t nb = (NUM_SEGMENTS + 1) + S * STRIDE;
        constexpr uint32_t db = nb + (ND + 1);
        constexpr int NTOP = (ND % 2 == 1) ? (int)ND : (int)ND - 1;
        constexpr int DTOP = (DD % 2 == 0) ? (int)DD : (int)DD - 1;
        v_if(x >= lut[S]) {
            selP = x * horner_odd<NTOP>(&lut[nb], t);
            selQ = horner_even<DTOP>(&lut[db], t);
        }
        v_endif;
        seg<S + 1, NUM_SEGMENTS, LUT_SIZE>(lut, x, t, selP, selQ);
    }
}
template <uint32_t POLY_DEGREE, uint32_t NUM_SEGMENTS, uint32_t LUT_SIZE>
inline void piecewise_generic_lut_dispatch(const std::array<float, LUT_SIZE>& lut) {
    for (int d = 0; d < 32; d++) {
        vFloat x = dst_reg[d];
        vFloat t = x * x;
        vFloat selP = 0.0f;
        vFloat selQ = 1.0f;  // safe default so unselected lanes never divide by zero
        seg<0, NUM_SEGMENTS, LUT_SIZE>(lut, x, t, selP, selQ);
        dst_reg[d] = selP * ckernel::sfpu::sfpu_reciprocal_iter<3>(selQ);  // ONE reciprocal
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
