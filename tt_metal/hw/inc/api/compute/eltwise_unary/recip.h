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
/**
 * Please refer to documentation for any_init.
 */
template <bool legacy_compat = true, bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void recip_tile_init() {
    MATH((sfpu::Reciprocal<APPROX, is_fp32_dest_acc_en, 8 /* ITERATIONS */, legacy_compat>::init()));
}
// clang-format off
/**
 * Performs element-wise computation of the reciprocal on each element of a tile
 * in DST register at index tile_index. The DST register buffer must be in
 * acquired state via *acquire_dst* call. This call is blocking and is only
 * available on the compute engine.
 * Only works for Float32, Float16_b, Bfp8_b data formats for full accuracy.
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
    bool legacy_compat = true,
    bool is_fp32_dest_acc_en = DST_ACCUM_MODE,
    TensorShape TENSOR_SHAPE = DEFAULT_TENSOR_SHAPE>
ALWI void recip_tile(std::uint32_t idst) {
    MATH((sfpu::Reciprocal<APPROX, is_fp32_dest_acc_en, 8 /* ITERATIONS */, legacy_compat>::template run<TENSOR_SHAPE>(
        idst)));
}

/**
 * Legacy overload selecting the faces to process with a VectorMode. Prefer the TensorShape template
 * parameter of the overload above: VectorMode::R and VectorMode::C correspond to
 * tensor_shape_from_tile_dims(16, 32) and tensor_shape_from_tile_dims(32, 16).
 */
template <bool legacy_compat = true, bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void recip_tile(std::uint32_t idst, VectorMode vector_mode) {
    MATH((sfpu::Reciprocal<APPROX, is_fp32_dest_acc_en, 8 /* ITERATIONS */, legacy_compat>::run_vector_mode(
        vector_mode, idst)));
}
}  // namespace ckernel
