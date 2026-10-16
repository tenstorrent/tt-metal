// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include "api/compute/common_globals.h"
#include "tensor_shape.h"
#ifdef TRISC_MATH
#include "ckernel_sfpu_recip.h"
#endif

namespace ckernel {

// Strongly typed so removed legacy_compat Boolean arguments cannot silently select DEST precision.
enum class ReciprocalDestAcc { BF16, FP32 };
enum class ReciprocalApproxMode { Default, Precise, Approximate };

/**
 * Please refer to documentation for any_init.
 * DEST precision defaults to DST_ACCUM_MODE. Explicit overrides must use ReciprocalDestAcc,
 * e.g. recip_tile_init<ReciprocalDestAcc::FP32>(); use the same mode for recip_tile.
 * approximation defaults to the kernel's APPROX setting; override it for callers that need a more
 * accurate reciprocal without changing the approximation of other operations in the kernel.
 * On Wormhole, precise BF16 reciprocal is correctly rounded only with the seed loaded here;
 * other SFPU inits may overwrite it, so call recip_tile_init again before recip_tile after them.
 * The former legacy_compat Boolean template argument is no longer accepted.
 */
template <
    ReciprocalDestAcc dest_acc = DST_ACCUM_MODE ? ReciprocalDestAcc::FP32 : ReciprocalDestAcc::BF16,
    ReciprocalApproxMode approximation = ReciprocalApproxMode::Default>
ALWI void recip_tile_init() {
    MATH((sfpu::Reciprocal<
          (approximation == ReciprocalApproxMode::Default ? APPROX
                                                          : approximation == ReciprocalApproxMode::Approximate),
          (dest_acc == ReciprocalDestAcc::FP32)>::init()));
}
// clang-format off
/**
 * Performs element-wise computation of the reciprocal on each element of a tile
 * in DST register at index tile_index. The DST register buffer must be in
 * acquired state via *acquire_dst* call. This call is blocking and is only
 * available on the compute engine.
 * Only works for Float32, Float16_b, Bfp8_b data formats for full accuracy.
 * DEST precision defaults to DST_ACCUM_MODE and approximation defaults to APPROX.
 * Explicit destination and approximation overrides must match recip_tile_init.
 * Legacy Boolean template arguments are no longer accepted.
 *
 * The TENSOR_SHAPE template parameter selects the tile to process, e.g.
 * tensor_shape_from_tile_dims(32, 16) for the left column of faces; the default is the full 32x32 tile.
 *
 * Return value: None
 *
 * | Argument       | Description                                                                | Type     | Valid Range                                           | Required |
 * |----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | idst           | The index of the tile in DST register buffer to perform the computation on | uint32_t | Must be less than the size of the DST register buffer | True     |
 */
// clang-format on
template <
    ReciprocalDestAcc dest_acc = DST_ACCUM_MODE ? ReciprocalDestAcc::FP32 : ReciprocalDestAcc::BF16,
    ReciprocalApproxMode approximation = ReciprocalApproxMode::Default,
    TensorShape TENSOR_SHAPE = DEFAULT_TENSOR_SHAPE>
ALWI void recip_tile(std::uint32_t idst) {
    MATH((sfpu::Reciprocal<
          (approximation == ReciprocalApproxMode::Default ? APPROX
                                                          : approximation == ReciprocalApproxMode::Approximate),
          (dest_acc == ReciprocalDestAcc::FP32)>::template run<TENSOR_SHAPE>(idst)));
}

/**
 * Legacy overload selecting the faces to process with a VectorMode. Prefer the TensorShape template
 * parameter of the overload above: VectorMode::R and VectorMode::C correspond to
 * tensor_shape_from_tile_dims(16, 32) and tensor_shape_from_tile_dims(32, 16).
 */
template <
    ReciprocalDestAcc dest_acc = DST_ACCUM_MODE ? ReciprocalDestAcc::FP32 : ReciprocalDestAcc::BF16,
    ReciprocalApproxMode approximation = ReciprocalApproxMode::Default>
ALWI void recip_tile(std::uint32_t idst, VectorMode vector_mode) {
    MATH((sfpu::Reciprocal<
          (approximation == ReciprocalApproxMode::Default ? APPROX
                                                          : approximation == ReciprocalApproxMode::Approximate),
          (dest_acc == ReciprocalDestAcc::FP32)>::run_vector_mode(vector_mode, idst)));
}
}  // namespace ckernel
