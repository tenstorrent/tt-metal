// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// =============================================================================
// SFPU Optimization Story — Rung R0: NAIVE Rational P(x)/Q(x)
// =============================================================================
// Drop-in replacement for the adhoc slot (kernels/compute/adhoc/adhoc.cpp).
// Reuses the SAME host/CB/LLK scaffolding as the production embedded kernel:
//   - same #includes
//   - same `namespace sfpi`
//   - same `kernel_main()` (copied verbatim from piecewise_generic.cpp)
//   - same dispatch entry point: sfpi::piecewise_generic_lut_dispatch<...>(*p_lut)
//
// R0 is the deliberately naive rational baseline:
//   - runtime segment cascade   (for s = 0 .. NUM_SEGMENTS-1, predicated v_if)
//   - runtime Horner for the numerator P(x)   (degree BENCH_R_NUM_DEGREE)
//   - runtime Horner for the denominator Q(x) (degree BENCH_R_DEN_DEGREE)
//   - reciprocal PER SEGMENT inside the v_if (deliberately naive)
//   - no unroll, no interleave, no parity, no deferred reciprocal
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

// ----------------------------------------------------------------------------
// R0 eval body — naive runtime rational with a runtime segment cascade.
//
// Entry point kernel_main() calls. Signature identical to the production
// piecewise_generic_lut_dispatch so the scaffolding is unchanged.
//
// LUT layout (rational):
//   [0 .. NUM_SEGMENTS]                : (NUM_SEGMENTS + 1) segment boundaries
//   then per segment s (stride = (NUM_DEG+1) + (DEN_DEG+1)):
//     num base = (NUM_SEGMENTS + 1) + s * stride      : numerator c0..cNUM_DEG
//     den base = num base + (NUM_DEG + 1)             : denominator c0..cDEN_DEG
// ----------------------------------------------------------------------------
template <uint32_t POLY_DEGREE, uint32_t NUM_SEGMENTS, uint32_t LUT_SIZE>
inline void piecewise_generic_lut_dispatch(const std::array<float, LUT_SIZE>& lut) {
    constexpr uint32_t NUM_DEG = BENCH_R_NUM_DEGREE;
    constexpr uint32_t DEN_DEG = BENCH_R_DEN_DEGREE;
    constexpr uint32_t STRIDE = (NUM_DEG + 1) + (DEN_DEG + 1);

    // Process all 32 destination registers (one tile row each).
    for (int d = 0; d < 32; d++) {
        vFloat x = dst_reg[d];

        // Naive segment selection: walk every segment, predicated. The cascade
        // keeps overwriting `result` for every boundary the input clears, so
        // out-of-range inputs naturally land on the edge segment.
        vFloat result = 0.0f;
        for (uint32_t s = 0; s < NUM_SEGMENTS; s++) {
            v_if(x >= lut[s]) {
                const uint32_t num_base = (NUM_SEGMENTS + 1) + s * STRIDE;
                const uint32_t den_base = num_base + (NUM_DEG + 1);

                // Naive runtime Horner for numerator P(x).
                vFloat numer = lut[num_base + NUM_DEG];
                for (int k = (int)NUM_DEG - 1; k >= 0; k--) {
                    numer = numer * x + lut[num_base + k];
                }

                // Naive runtime Horner for denominator Q(x).
                vFloat denom = lut[den_base + DEN_DEG];
                for (int k = (int)DEN_DEG - 1; k >= 0; k--) {
                    denom = denom * x + lut[den_base + k];
                }

                // Reciprocal-per-segment (deliberately naive): 3 Newton-Raphson iters.
                result = numer * ckernel::sfpu::sfpu_reciprocal_iter<3>(denom);
            }
            v_endif;
        }

        dst_reg[d] = result;
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
