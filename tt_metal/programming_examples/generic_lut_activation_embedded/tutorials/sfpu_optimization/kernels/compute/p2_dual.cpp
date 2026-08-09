// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// =============================================================================
// SFPU Optimization Story — Rung P2: DUAL-EVAL
// =============================================================================
// This rung processes TWO destination registers per loop iteration instead of
// one, running two independent Horner chains side by side. Crucially, each
// coefficient load from the LUT is SHARED across both chains: a single c[K] is
// fetched and applied to both accumulators (a0, a1) back-to-back. The two
// chains are data-independent, so their FMA (sfpmad) instructions can be issued
// in lockstep — this exposes instruction-level parallelism that hides the
// SFPU's per-instruction pipeline latency, roughly doubling useful throughput
// without doubling coefficient-fetch traffic. The segment cascade and Horner
// depth are otherwise the same as the unrolled baseline; only the eval body is
// widened to two lanes (x0, x1 / r0, r1). The benchmark is ODD parity (max
// degree 8, 16 segments), and BENCH_SEGMENT_DEGREES[] holds each segment's
// effective degree, though this rung evaluates the full POLY_DEGREE per chain.
//
// Like every rung, it is a drop-in replacement for the adhoc slot and reuses
// the SAME host/CB/LLK scaffolding (#includes, namespace sfpi, kernel_main(),
// and the dispatch entry point sfpi::piecewise_generic_lut_dispatch<...>).
// It consumes the shared, deterministically-generated benchmark LUT from
// ../common/bench_lut.h.
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
template <int K>
__attribute__((always_inline)) static inline void horner2_rec(
    const float* c, vFloat x0, vFloat x1, vFloat& a0, vFloat& a1) {
    if constexpr (K >= 0) {
        vFloat ck = c[K];
        a0 = a0 * x0 + ck;
        a1 = a1 * x1 + ck;
        horner2_rec<K - 1>(c, x0, x1, a0, a1);
    }
}
template <int D>
__attribute__((always_inline)) static inline void horner2(
    const float* c, vFloat x0, vFloat x1, vFloat& a0, vFloat& a1) {
    a0 = vFloat(c[D]);
    a1 = vFloat(c[D]);
    horner2_rec<D - 1>(c, x0, x1, a0, a1);
}
template <uint32_t S, uint32_t POLY_DEGREE, uint32_t NUM_SEGMENTS, uint32_t LUT_SIZE>
__attribute__((always_inline)) static inline void seg2(
    const std::array<float, LUT_SIZE>& lut, vFloat x0, vFloat x1, vFloat& r0, vFloat& r1) {
    if constexpr (S < NUM_SEGMENTS) {
        constexpr uint32_t base = (NUM_SEGMENTS + 1) + S * (POLY_DEGREE + 1);
        vFloat a0, a1;
        horner2<(int)POLY_DEGREE>(&lut[base], x0, x1, a0, a1);
        v_if(x0 >= lut[S]) { r0 = a0; }
        v_endif;
        v_if(x1 >= lut[S]) { r1 = a1; }
        v_endif;
        seg2<S + 1, POLY_DEGREE, NUM_SEGMENTS, LUT_SIZE>(lut, x0, x1, r0, r1);
    }
}
template <uint32_t POLY_DEGREE, uint32_t NUM_SEGMENTS, uint32_t LUT_SIZE>
inline void piecewise_generic_lut_dispatch(const std::array<float, LUT_SIZE>& lut) {
    for (int d = 0; d < 32; d += 2) {
        vFloat x0 = dst_reg[d];
        vFloat x1 = dst_reg[d + 1];
        vFloat r0 = 0.0f, r1 = 0.0f;
        seg2<0, POLY_DEGREE, NUM_SEGMENTS, LUT_SIZE>(lut, x0, x1, r0, r1);
        dst_reg[d] = r0;
        dst_reg[d + 1] = r1;
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
