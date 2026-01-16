// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "compute_kernel_api/common.h"
#ifdef TRISC_MATH
#include "../../hw/ckernels/blackhole/metal/llk_api/llk_math_custom_mm_api.h"
#endif
#ifdef TRISC_UNPACK
#include "../../hw/ckernels/blackhole/metal/llk_api/llk_unpack_AB_custom_mm_api.h"
#endif
namespace ckernel {

// clang-format off
/**
 * Initialization for custom_mm_block operation. Must be called before custom_mm_block.
 *
 * Custom matmul that uses MOP to loop both srcA and srcB along inner dim. Output height
 * and width should be single tile with tile shape [1, 32]. Further work will uplift the
 * custom mm to support for tiles along the width.
 *
 * This is optimized for K-dimension reduction where ct_dim=1, rt_dim=1, and kt_dim>1.
 * The MOP replay buffer can unpack both SrcA and SrcB, looping over the K dimension
 * with hardware pipelining for maximum throughput (up to 128 K tiles per MOP call).
 *
 * NOTE: This API only supports ct_dim=1 and rt_dim=1 (single output tile).
 *
 * Return value: None
 *
 * | Argument       | Description                                                   | Type     | Valid Range                                      | Required |
 * |----------------|---------------------------------------------------------------|----------|--------------------------------------------------|----------|
 * | in0_cb_id      | The identifier of the first input circular buffer (CB)        | uint32_t | 0 to 31                                          | False    |
 * | in1_cb_id      | The identifier of the second input circular buffer (CB)       | uint32_t | 0 to 31                                          | False    |
 * | out_cb_id      | The identifier of the output circular buffer (CB)             | uint32_t | 0 to 31                                          | False    |
 * | transpose      | The transpose flag for performing transpose operation on B    | uint32_t | Any positive value will indicate tranpose is set | False    |
 * | kt_dim         | The inner dim of the input matrices in tiles                  | uint32_t | 1 to 128 per MOP call, chunked if larger        | False    |
 */
// clang-format on
ALWI void custom_mm_block_init(
    uint32_t in0_cb_id, uint32_t in1_cb_id, uint32_t out_cb_id, const uint32_t transpose = 0, uint32_t kt_dim = 1) {
    UNPACK((llk_unpack_AB_custom_mm_hw_configure_disaggregated<DST_ACCUM_MODE>(in0_cb_id, in1_cb_id)));
    UNPACK((llk_unpack_AB_custom_mm_init(in0_cb_id, in1_cb_id, transpose, kt_dim)));

    MATH((llk_math_custom_mm_init<MATH_FIDELITY>(in0_cb_id, in1_cb_id, transpose, kt_dim)));
    MATH((llk_math_pack_sync_init<DST_ACCUM_MODE>()));
    MATH((llk_math_hw_configure<DST_ACCUM_MODE>(in0_cb_id, in1_cb_id)));

    PACK((llk_pack_hw_configure_disaggregated<DST_ACCUM_MODE, false>(out_cb_id)));
    PACK((llk_pack_init<false, false>(out_cb_id)));
    PACK((llk_pack_dest_init<DST_ACCUM_MODE, false>()));
}

// clang-format off
/**
 * Performs block-sized matrix multiplication *C=A\*B* between the blocks in two
 * different input CBs and writes the result to DST. The DST register buffer
 * must be in acquired state via *acquire_dst* call. This call is blocking and
 * is only available on the compute engine.
 *
 * NOTE: This API only supports ct_dim=1 and rt_dim=1 (single output tile).
 * For K-dimension optimization (ct_dim=1, rt_dim=1, kt_dim>1), this will use
 * a single MOP replay that handles up to 128 K tiles, eliminating software loop overhead.
 *
 * Return value: None
 *
 * | Argument       | Description                                                             | Type     | Valid Range                                    | Required |
 * |----------------|-------------------------------------------------------------------------|----------|------------------------------------------------|----------|
 * | in0_cb_id      | The identifier of the first input circular buffer (CB)                  | uint32_t | 0 to 31                                        | True     |
 * | in1_cb_id      | The identifier of the second input circular buffer (CB)                 | uint32_t | 0 to 31                                        | True     |
 * | in0_tile_index | The index of the tile in block A from the first input CB                | uint32_t | Must be less than the size of the CB           | True     |
 * | in1_tile_index | The index of the tile in block B from the second input CB               | uint32_t | Must be less than the size of the CB           | True     |
 * | idst           | The index of the tile in DST REG to which the result C will be written. | uint32_t | Must be less than the acquired size of DST REG | True     |
 * | transpose      | The transpose flag for performing transpose operation on tiles in B.    | bool     | Must be true or false                          | True     |
 * | kt_dim         | The inner dimension (K reduction dimension).                            | uint32_t | Must be equal to block A column dimension      | True     |
 */
// clang-format on
ALWI void custom_mm_block(
    uint32_t in0_cb_id,
    uint32_t in1_cb_id,
    uint32_t in0_tile_index,
    uint32_t in1_tile_index,
    uint32_t idst,
    const uint32_t transpose,
    uint32_t kt_dim) {
    UNPACK((llk_unpack_AB_custom_mm(in0_cb_id, in1_cb_id, in0_tile_index, in1_tile_index, kt_dim)));
    MATH((llk_math_custom_mm<MATH_FIDELITY>(idst, transpose, kt_dim)));
}

ALWI void custom_mm_block_unpack(
    uint32_t in0_cb_id,
    uint32_t in1_cb_id,
    uint32_t in0_tile_index,
    uint32_t in1_tile_index,
    uint32_t idst,
    const uint32_t transpose,
    uint32_t kt_dim) {
    UNPACK((llk_unpack_AB_custom_mm(in0_cb_id, in1_cb_id, in0_tile_index, in1_tile_index, kt_dim)));
}

ALWI void custom_mm_block_math(
    uint32_t in0_cb_id,
    uint32_t in1_cb_id,
    uint32_t in0_tile_index,
    uint32_t in1_tile_index,
    uint32_t idst,
    const uint32_t transpose,
    uint32_t kt_dim) {
    MATH((llk_math_custom_mm<MATH_FIDELITY>(idst, transpose, kt_dim)));
}

}  // namespace ckernel
