// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Per-row RMSNorm and LayerNorm Compute API for packed (M, 32) and stacked (16/32, 32) tiles
// (M in {1, 2, 4, 8}, tile_h in {M, 16, 32}). `stats_tile` is relative to `first_tile`; Face 0
// holds mean and Face 1 holds sum-of-squares / inv_std (hi+lo double-BF16 when !stats_is_fp32).
#pragma once

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#if defined(TRISC_MATH) && defined(ARCH_BLACKHOLE)
#include "experimental/llk_sfpu/ckernel_sfpu_row_norm.h"
#include "llk_math_eltwise_unary_sfpu_macros.h"
#endif

namespace ckernel {

#if defined(ARCH_BLACKHOLE)

#if defined(TRISC_MATH)
namespace detail {

template <bool is_fp32_dest_acc_en, std::uint32_t needed_tiles>
ALWI void check_row_norm_dest_range(const std::uint32_t first_tile) {
    constexpr std::uint32_t max_dest_tiles =
        get_dest_max_tiles<DST_SYNC_MODE, is_fp32_dest_acc_en, DstTileShape::Tile32x32>();
    static_assert(needed_tiles <= max_dest_tiles, "row_norm: tile range exceeds DST_SYNC_MODE capacity");
    LLK_ASSERT(
        (first_tile + needed_tiles <= get_dest_max_tiles_rt<DST_SYNC_MODE, DstTileShape::Tile32x32>()),
        "row_norm: [first_tile, first_tile + needed_tiles) exceeds DEST capacity");
}

}  // namespace detail
#endif  // TRISC_MATH

/** @brief Initialize SFPU address modes and rsqrt constants for `row_*` norm ops. */
ALWI void row_norm_init() { MATH(SFPU_UNARY_INIT_FN(rsqrt, sfpu::_row_norm_init_, (APPROX))); }

/**
 * @brief In-place per-row RMSNorm across `[first_tile, first_tile + num_tiles)`.
 * @tparam num_rows Logical rows per tile (`1`, `2`, `4`, or `8`).
 * @tparam tile_h Physical tile height (`num_rows`, `16`, or `32`).
 * @tparam num_tiles Contiguous DST tiles in the row span (`1 <= num_tiles <= 8`).
 * @param first_tile Base DST tile index.
 * @param inv_width_bits IEEE-754 FP32 bits of `1 / width` (ignored when `!scale_sum`).
 * @param epsilon_bits IEEE-754 FP32 bits of `epsilon`.
 */
template <
    std::uint32_t num_rows,
    std::uint32_t tile_h,
    std::uint32_t num_tiles,
    bool fast_approx = false,
    bool scale_sum = true,
    bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void row_rmsnorm(std::uint32_t first_tile, std::uint32_t inv_width_bits, std::uint32_t epsilon_bits) {
    MATH((detail::check_row_norm_dest_range<is_fp32_dest_acc_en, num_tiles>(first_tile)));
    MATH(SFPU_UNARY_CALL(
        DST_SYNC_MODE,
        is_fp32_dest_acc_en,
        _calculate_row_rmsnorm_,
        (APPROX, fast_approx, is_fp32_dest_acc_en, num_rows, tile_h, num_tiles, scale_sum),
        first_tile,
        VectorMode::RC_custom,
        inv_width_bits,
        epsilon_bits));
}

/**
 * @brief FPU-assisted RMSNorm tail: reduce pre-squared tiles and replicate `inv_rms` into `first_tile`.
 *
 * Sums `[first_tile, first_tile + num_tiles)` in FP32, computes `rsqrt(sum * inv_width + epsilon)`,
 * and replicates `inv_rms` into column 0 of `first_tile`.
 *
 * @tparam face0_only Set true when consumed immediately by `bcast_dest_reuse_tiles` (`BroadcastType::COL`);
 *                    set false when packed to L1 for `mul_tiles_bcast_cols`.
 * @tparam is_fp32_dest_acc_en Whether FP32 DEST accumulation is enabled (defaults to `DST_ACCUM_MODE`).
 */
template <
    std::uint32_t num_rows,
    std::uint32_t tile_h,
    std::uint32_t num_tiles,
    bool fast_approx = false,
    bool scale_sum = true,
    bool face0_only = false,
    bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void row_reduce_scale_add_rsqrt(
    std::uint32_t first_tile, std::uint32_t inv_width_bits, std::uint32_t epsilon_bits) {
    MATH((detail::check_row_norm_dest_range<is_fp32_dest_acc_en, num_tiles>(first_tile)));
    MATH(SFPU_UNARY_CALL(
        DST_SYNC_MODE,
        is_fp32_dest_acc_en,
        _calculate_row_reduce_scale_add_rsqrt_,
        (APPROX, fast_approx, is_fp32_dest_acc_en, num_rows, tile_h, num_tiles, scale_sum, face0_only),
        first_tile,
        VectorMode::RC_custom,
        inv_width_bits,
        epsilon_bits));
}

/**
 * @brief Single-pass in-DST per-row LayerNorm across `num_tiles` starting at `first_tile`.
 *
 * Computes mean and centered variance in FP32 registers and stores `(x - mean) * rsqrt(var + epsilon)`.
 *
 * @note When `num_rows <= 2` and `tile_h > num_rows` (stacked `M in {1, 2}`), `DST[first_tile + num_tiles]`
 *       is used as a temporary scratch slot to hold mean across `subvec_transp` and is clobbered.
 */
template <
    std::uint32_t num_rows,
    std::uint32_t tile_h,
    std::uint32_t num_tiles,
    bool fast_approx = false,
    bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void row_layernorm(std::uint32_t first_tile, std::uint32_t inv_width_bits, std::uint32_t epsilon_bits) {
    constexpr std::uint32_t needed_tiles = num_tiles + ((num_rows <= 2 && tile_h > num_rows) ? 1u : 0u);
    MATH((detail::check_row_norm_dest_range<is_fp32_dest_acc_en, needed_tiles>(first_tile)));
    MATH(SFPU_UNARY_CALL(
        DST_SYNC_MODE,
        is_fp32_dest_acc_en,
        _calculate_row_layernorm_,
        (APPROX, fast_approx, is_fp32_dest_acc_en, num_rows, tile_h, num_tiles),
        first_tile,
        VectorMode::RC_custom,
        inv_width_bits,
        epsilon_bits));
}

/**
 * @brief Wide-row LayerNorm Pass 1a tail: reduce FPU-summed tiles and store mean in `stats_tile` Face 0.
 *
 * @tparam stats_tile Offset relative to `first_tile` holding the compact 1-tile LayerNorm stats.
 * @tparam stats_is_fp32 True if the stats CB is FP32; false uses hi+lo double-BF16 encoding.
 * @tparam is_fp32_dest_acc_en Whether FP32 DEST accumulation is enabled (defaults to `DST_ACCUM_MODE`).
 */
template <
    std::uint32_t num_rows,
    std::uint32_t tile_h,
    std::uint32_t num_tiles,
    std::uint32_t stats_tile,
    bool stats_is_fp32 = DST_ACCUM_MODE,
    bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void row_reduce_mean(std::uint32_t first_tile, std::uint32_t inv_width_bits) {
    constexpr std::uint32_t needed_tiles = (stats_tile >= num_tiles) ? (stats_tile + 1u) : num_tiles;
    MATH((detail::check_row_norm_dest_range<is_fp32_dest_acc_en, needed_tiles>(first_tile)));
    MATH(SFPU_UNARY_CALL(
        DST_SYNC_MODE,
        is_fp32_dest_acc_en,
        _calculate_row_reduce_mean_,
        (is_fp32_dest_acc_en, num_rows, tile_h, num_tiles, stats_tile, stats_is_fp32),
        first_tile,
        VectorMode::RC_custom,
        inv_width_bits));
}

/**
 * @brief Wide-row LayerNorm Pass 1b chunk: accumulate `sum((x - mean)^2)` into `stats_tile` Face 1.
 *
 * Reads mean from Face 0 of `(first_tile + stats_tile)` unless `zero_mean` is true. Call
 * `row_finalize_inv_std` once after the last chunk to convert the sum into `inv_std`.
 *
 * @tparam stats_is_fp32 Must match the value used by `row_reduce_mean`.
 * @tparam is_fp32_dest_acc_en Whether FP32 DEST accumulation is enabled (defaults to `DST_ACCUM_MODE`).
 */
template <
    std::uint32_t num_rows,
    std::uint32_t tile_h,
    std::uint32_t num_tiles,
    std::uint32_t stats_tile,
    bool accumulate = false,
    bool zero_mean = false,
    bool stats_is_fp32 = DST_ACCUM_MODE,
    bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void row_sum_centered_squares(std::uint32_t first_tile) {
    constexpr std::uint32_t needed_tiles = (stats_tile >= num_tiles) ? (stats_tile + 1u) : num_tiles;
    MATH((detail::check_row_norm_dest_range<is_fp32_dest_acc_en, needed_tiles>(first_tile)));
    MATH(SFPU_UNARY_CALL(
        DST_SYNC_MODE,
        is_fp32_dest_acc_en,
        _calculate_row_sum_centered_squares_,
        (is_fp32_dest_acc_en, num_rows, tile_h, num_tiles, stats_tile, accumulate, zero_mean, stats_is_fp32),
        first_tile,
        VectorMode::RC_custom));
}

/**
 * @brief Wide-row LayerNorm Pass 1b finalizer: convert accumulated sum in `stats_tile` Face 1 to `inv_std`.
 *
 * Computes `inv_std = rsqrt(sum * inv_width + epsilon)` in place on Face 1 of `(first_tile + stats_tile)`.
 */
template <
    std::uint32_t num_rows,
    std::uint32_t stats_tile,
    bool fast_approx = false,
    bool stats_is_fp32 = DST_ACCUM_MODE,
    bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void row_finalize_inv_std(std::uint32_t first_tile, std::uint32_t inv_width_bits, std::uint32_t epsilon_bits) {
    MATH((detail::check_row_norm_dest_range<is_fp32_dest_acc_en, stats_tile + 1u>(first_tile)));
    MATH(SFPU_UNARY_CALL(
        DST_SYNC_MODE,
        is_fp32_dest_acc_en,
        _calculate_row_finalize_inv_std_,
        (APPROX, fast_approx, is_fp32_dest_acc_en, num_rows, stats_tile, stats_is_fp32),
        first_tile,
        VectorMode::RC_custom,
        inv_width_bits,
        epsilon_bits));
}

/**
 * @brief Wide-row LayerNorm Pass 2 chunk: apply `(x - mean) * inv_std` using `stats_tile`.
 *
 * Reads mean (Face 0, unless `zero_mean` is true) and `inv_std` (Face 1) from `(first_tile + stats_tile)`
 * and updates `[first_tile, first_tile + num_tiles)` in place.
 */
template <
    std::uint32_t num_rows,
    std::uint32_t tile_h,
    std::uint32_t num_tiles,
    std::uint32_t stats_tile,
    bool zero_mean = false,
    bool stats_is_fp32 = DST_ACCUM_MODE,
    bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void row_center_scale(std::uint32_t first_tile) {
    constexpr std::uint32_t needed_tiles = (stats_tile >= num_tiles) ? (stats_tile + 1u) : num_tiles;
    MATH((detail::check_row_norm_dest_range<is_fp32_dest_acc_en, needed_tiles>(first_tile)));
    MATH(SFPU_UNARY_CALL(
        DST_SYNC_MODE,
        is_fp32_dest_acc_en,
        _calculate_row_center_scale_,
        (is_fp32_dest_acc_en, num_rows, tile_h, num_tiles, stats_tile, zero_mean, stats_is_fp32),
        first_tile,
        VectorMode::RC_custom));
}

#endif  // ARCH_BLACKHOLE

}  // namespace ckernel
