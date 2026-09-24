// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include "api/compute/compute_kernel_api.h"
#include "api/compute/common.h"
#include "tensor_shape.h"
// Blackhole-only: the add_rsqrt SFPU functor lives only in the Blackhole llk_api tree.
#if defined(TRISC_MATH) && defined(ARCH_BLACKHOLE)
#include "experimental/llk_sfpu/ckernel_sfpu_add_rsqrt.h"
#endif

namespace ckernel {

#if defined(ARCH_BLACKHOLE)

/**
 * Initialize for add + rsqrt operation: result = rsqrt(x + addend)
 * Useful for operations like RMSNorm: rsqrt(variance + epsilon)
 */
ALWI void add_rsqrt_tile_init() { MATH((sfpu::AddRsqrt<APPROX>::init())); }

/**
 * Perform add + rsqrt operation: result = rsqrt(x + addend)
 *
 * typed_bf16_store=true selects the direct BF16 store used by fused normalization.
 * The TENSOR_SHAPE template parameter selects the tile to process, e.g.
 * tensor_shape_from_tile_dims(16, 16) for a single face; the default is the full 32x32 tile.
 *
 * @param idst The index of the tile in DST register buffer
 * @param addend The bit representation of a float to add before computing rsqrt
 */
template <
    bool fast_and_approx = false,
    TensorShape TENSOR_SHAPE = DEFAULT_TENSOR_SHAPE,
    int ITERATIONS = 8,
    bool typed_bf16_store = false>
ALWI void add_rsqrt_tile(std::uint32_t idst, std::uint32_t addend) {
    MATH((sfpu::AddRsqrt<APPROX, ITERATIONS, DST_ACCUM_MODE, fast_and_approx, typed_bf16_store>::template run<
          TENSOR_SHAPE>(idst, addend)));
}

/**
 * Legacy overload selecting the faces to process with a VectorMode. Prefer the TensorShape template
 * parameter of the overload above: VectorMode::R, VectorMode::C and VectorMode::RC_custom correspond to
 * tensor_shape_from_tile_dims(16, 32), tensor_shape_from_tile_dims(32, 16) and tensor_shape_from_tile_dims(16, 16).
 */
template <bool fast_and_approx, VectorMode vec_mode, int ITERATIONS = 8, bool typed_bf16_store = false>
ALWI void add_rsqrt_tile(std::uint32_t idst, std::uint32_t addend) {
    MATH((sfpu::AddRsqrt<APPROX, ITERATIONS, DST_ACCUM_MODE, fast_and_approx, typed_bf16_store>::run_vector_mode(
        vec_mode, idst, addend)));
}

#endif  // ARCH_BLACKHOLE

}  // namespace ckernel
