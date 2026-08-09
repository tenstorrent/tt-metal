// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// =============================================================================
// SFPU Optimization Story — Rung P0: NAIVE Horner
// =============================================================================
// This is a drop-in replacement for the adhoc slot (kernels/compute/adhoc/adhoc.cpp).
// It reuses the SAME host/CB/LLK scaffolding as the production embedded kernel:
//   - same #includes
//   - same `namespace sfpi`
//   - same `kernel_main()` (copied verbatim from piecewise_generic.cpp)
//   - same dispatch entry point: sfpi::piecewise_generic_lut_dispatch<...>(*p_lut)
//
// The ONLY thing that differs between rungs is the eval body inside
// piecewise_generic_lut_dispatch(). P0 is the deliberately naive baseline:
//   - runtime segment cascade   (for s = 0 .. NUM_SEGMENTS-1, predicated v_if)
//   - runtime Horner loop        (for d = MAX_DEGREE-1 .. 0)
//   - no unroll, no parity, no dual-eval, no adaptive degree
//
// Instead of an embedded per-CSV LUT, every rung consumes the shared,
// deterministically-generated benchmark LUT from ../common/bench_lut.h.
// =============================================================================

#include <cstdint>
#include <array>

// Compute API + SFPU (sfpi) types — same includes the production kernel uses.
// Pulled in at the top so `sfpi::vFloat` / `dst_reg` / `v_if` are visible to the
// eval body and `init_sfpu` / `tile_regs_*` are visible to kernel_main().
#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"

// ----------------------------------------------------------------------------
// Benchmark LUT (constexpr): BENCH_NUM_SEGMENTS, BENCH_MAX_DEGREE, BENCH_LUT, ...
// Layout: [NUM_SEG+1 boundaries][per-seg c0..cMAX].
//
// NOTE on the include path: the driver copies this rung into the adhoc slot at
//   generic_lut_activation_embedded/kernels/compute/adhoc/adhoc.cpp
// and JIT resolves quoted includes relative to THAT directory. The bench header
// lives under tutorials/sfpu_optimization/kernels/common/, so the path is
// relative to the adhoc slot (../../.. climbs to the example root). Every rung
// uses this exact include.
// ----------------------------------------------------------------------------
#include "../../../tutorials/sfpu_optimization/kernels/common/bench_lut.h"

// ----------------------------------------------------------------------------
// Alias the BENCH_* names onto the names the shared kernel_main() expects.
// kernel_main() (below, copied verbatim) reads POLY_DEGREE / NUM_SEGMENTS /
// LUT_SIZE / LUT_DATA, exactly like a generated adhoc.cpp header.
// ----------------------------------------------------------------------------
#define EMBEDDED_LUT
constexpr uint32_t POLY_DEGREE = BENCH_MAX_DEGREE;
constexpr uint32_t NUM_SEGMENTS = BENCH_NUM_SEGMENTS;
constexpr uint32_t LUT_SIZE = BENCH_LUT_SIZE;
constexpr auto& LUT_DATA = BENCH_LUT;

// ============================================================================
// SFPU compute (math TRISC only)
// ============================================================================
#ifdef TRISC_MATH

namespace sfpi {

// ----------------------------------------------------------------------------
// P0 eval body — naive runtime Horner with a runtime segment cascade.
//
// This is the entry point kernel_main() calls. Its signature is identical to
// the production piecewise_generic_lut_dispatch so the scaffolding is unchanged.
// ----------------------------------------------------------------------------
template <uint32_t POLY_DEGREE, uint32_t NUM_SEGMENTS, uint32_t LUT_SIZE>
inline void piecewise_generic_lut_dispatch(const std::array<float, LUT_SIZE>& lut) {
    // Coefficient block starts right after the (NUM_SEGMENTS + 1) boundaries.
    // Each segment has (POLY_DEGREE + 1) coefficients: c0, c1, ..., cMAX.

    // Process all 32 destination registers (one tile row each).
    for (int d = 0; d < 32; d++) {
        vFloat x = dst_reg[d];

        // Naive segment selection: walk every segment, predicated. Out-of-range
        // inputs naturally fall onto the edge segment because the cascade keeps
        // overwriting `result` for every boundary the input clears.
        vFloat result = 0.0f;
        for (uint32_t s = 0; s < NUM_SEGMENTS; s++) {
            v_if(x >= lut[s]) {
                // Coefficient base for this segment.
                const uint32_t base = (NUM_SEGMENTS + 1) + s * (POLY_DEGREE + 1);

                // Naive runtime Horner: acc = ((cMAX*x + cMAX-1)*x + ...)*x + c0
                vFloat acc = lut[base + POLY_DEGREE];
                for (int k = (int)POLY_DEGREE - 1; k >= 0; k--) {
                    acc = acc * x + lut[base + k];
                }
                result = acc;
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
