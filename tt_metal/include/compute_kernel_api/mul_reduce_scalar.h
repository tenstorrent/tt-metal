// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "compute_kernel_api/common.h"
#include "compute_kernel_api/eltwise_binary.h"

#ifdef TRISC_MATH
#include "llk_math_eltwise_binary.h"
#include "llk_math_reduce_api.h"
#include "ckernel_sfpu.h"
#include "sfpu/ckernel_sfpu_fill.h"
#endif

namespace ckernel {

// clang-format off
/**
 * Initializes the fused multiply-reduce-scalar operation.
 *
 * This function initializes all necessary components (UNPACK, MATH, PACK) for the fused
 * multiply + reduce scalar operation. The operation computes: scalar = sum(A * B) where
 * A and B are input tensors (single tile each) and the result is a single scalar value.
 *
 * Must be called before mul_reduce_scalar_tile().
 *
 * | Argument       | Description                                                   | Type     | Valid Range | Required |
 * |----------------|---------------------------------------------------------------|----------|-------------|----------|
 * | icb0           | Input circular buffer 0 (tensor A)                            | uint32_t | 0 to 31     | True     |
 * | icb1           | Input circular buffer 1 (tensor B)                            | uint32_t | 0 to 31     | True     |
 * | ocb            | Output circular buffer (scalar result)                        | uint32_t | 0 to 31     | True     |
 *
 * Return value: None
 */
// clang-format on
template <PoolType reduce_type = PoolType::SUM>
ALWI void mul_reduce_scalar_init(uint32_t icb0, uint32_t icb1, uint32_t ocb) {
    // UNPACK initialization - configure and init for both input CBs
    UNPACK((llk_unpack_hw_configure<DST_ACCUM_MODE>(icb0, icb1)));
    UNPACK((llk_unpack_AB_init<BroadcastType::NONE>(icb0, icb1)));

    // MATH initialization - configure sync and hardware, then init eltwise binary (multiply)
    MATH((llk_math_pack_sync_init<DST_ACCUM_MODE>()));
    MATH((llk_math_hw_configure<DST_ACCUM_MODE>(icb0, icb1)));
    MATH((llk_math_eltwise_binary_init_with_operands<EltwiseBinaryType::ELWMUL, NONE, MATH_FIDELITY>(
        icb0, icb1, false /*acc_to_dest*/)));

    // PACK initialization
    PACK((llk_pack_hw_configure_disaggregated<DST_ACCUM_MODE, false>(ocb)));
    PACK((llk_pack_init(ocb)));
    PACK((llk_pack_dest_init<DST_ACCUM_MODE, false>()));
}

// clang-format off
/**
 * Performs a fused multiply-reduce-scalar operation on tiles.
 *
 * This function performs:
 * 1. Element-wise multiplication: C = A * B
 * 2. Scalar reduction: result = sum(all elements of C)
 *
 * The final scalar result is stored in dest[0] at element position [0].
 *
 * The scalar result will be at element [0] of the output tile.
 *
 * | Argument       | Description                                                   | Type     | Valid Range | Required |
 * |----------------|---------------------------------------------------------------|----------|-------------|----------|
 * | icb0           | Input circular buffer 0 (tensor A)                            | uint32_t | 0 to 31     | True     |
 * | icb1           | Input circular buffer 1 (tensor B)                            | uint32_t | 0 to 31     | True     |
 * | itile0         | Input tile index for tensor A (usually 0)                     | uint32_t | 0+          | True     |
 * | itile1         | Input tile index for tensor B (usually 0)                     | uint32_t | 0+          | True     |
 * | num_tiles      | Number of tiles to process                                    | uint32_t | 1+          | True     |
 *
 * Return value: None
 */
// clang-format on
template <PoolType reduce_type = PoolType::SUM>
ALWI void mul_reduce_scalar_tile(uint32_t icb0, uint32_t icb1, uint32_t itile0, uint32_t itile1, uint32_t num_tiles) {
    // Step 1: Unpack input tiles from both circular buffers and perform multiplication
    UNPACK((llk_unpack_AB(icb0, icb1, itile0, itile1)));

    // Perform element-wise multiplication into dest[i]
    MATH((llk_math_eltwise_binary<
          EltwiseBinaryType::ELWMUL,
          NONE,
          DST_ACCUM_MODE,
          MATH_FIDELITY,
          EltwiseBinaryReuseDestType::NONE>(0)));

    // Step 2: Uninit eltwise binary - reset counters before reduce operation
    MATH((_fused_eltwise_binary_uninit_()));

    // Step 3: Switch banks - UNPACK thread switches srcA/srcB banks for reduce
    UNPACK((llk_unpack_AB_fused()));

    // Step 4: Initialize reduce operation for scalar reduction
    MATH((llk_math_reduce_init<reduce_type, ReduceDim::REDUCE_SCALAR, DST_ACCUM_MODE, MATH_FIDELITY>()));

    // Step 5: Prepare data for first tile's scalar reduction
    // Move dest[0] (first multiply result) to srcA
    MATH((eltwise_binary_reuse_dest_as_src_tile<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(0)));

    // Populate srcB with ones for the scaler
    MATH((ckernel::sfpu::_populate_first_tile_with_ones_()));
    MATH((eltwise_binary_reuse_dest_as_src_tile<EltwiseBinaryReuseDestType::DEST_TO_SRCB>(0)));

    // Clear dest[0] - this will accumulate scalar reduction results from all tiles
    MATH((ckernel::sfpu::_populate_first_tile_with_zeroes_()));

    // Step 6: Configure packer for scalar reduction
    PACK((llk_pack_reduce_mask_config<false /*untilize*/, ReduceDim::REDUCE_SCALAR>()));

    // Step 7: Perform scalar reduction
    MATH((llk_math_reduce_scalar_fused<
          reduce_type,
          ReduceDim::REDUCE_SCALAR,
          DST_ACCUM_MODE,
          MATH_FIDELITY,
          false,
          false>(0)));
}

// clang-format off
/**
 * NOTE: Should possibly be removed in the future.
 * Clears data valid flags after reduce operation.
 *
 *
 * Call sequence:
 * 1. mul_reduce_scalar_tile()
 * 2. pack_tile(0, ocb)
 * 3. mul_reduce_scalar_clear_dvalid()  <- MUST be here
 * 4. mul_reduce_scalar_uninit()
 *
 * Return value: None
 */
// clang-format on
ALWI void mul_reduce_scalar_clear_dvalid() { MATH((llk_math_reduce_clear_dvalid_after_for_loop())); }

// clang-format off
/**
 * Uninitializes the fused multiply-reduce-scalar operation.
 *
 * This function cleans up the reduce operation and should be called after
 * mul_reduce_scalar_tile() operations are complete.
 *
 * Return value: None
 */
// clang-format on
ALWI void mul_reduce_scalar_uninit() { PACK((llk_pack_reduce_mask_clear())); }

}  // namespace ckernel
