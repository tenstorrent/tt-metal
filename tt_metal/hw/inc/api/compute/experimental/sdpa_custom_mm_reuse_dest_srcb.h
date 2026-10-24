// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <type_traits>
#include "api/compute/common.h"
#ifdef TRISC_MATH
#include "experimental/llk_math_sdpa_custom_mm_reuse_dest_srcb_api.h"
#endif
#ifdef TRISC_UNPACK
#include "experimental/llk_unpack_AB_sdpa_custom_mm_reuse_dest_srcb_api.h"
#include "experimental/llk_unpack_A_sdpa_api.h"
#endif
namespace ckernel {

// clang-format off
/**
 * Initialization for sdpa_custom_mm_reuse_dest_srcb_block operation. Must be called before sdpa_custom_mm_reuse_dest_srcb_block.
 *
 * Custom matmul that reuses SrcB from dest and only unpacks SrcA. Output height
 * and width should be single tile with tile shape [1, 32]. Further work will uplift the
 * custom mm to support for tiles along the width.
 *
 * This is optimized for K-dimension reduction where ct_dim=1, rt_dim=1, and kt_dim>1.
 * The MOP replay buffer unpacks SrcA using CFGSHIFTMASK for address auto-increment,
 * collapsing the nt_dim loop into a single MOP call per kt_dim iteration.
 *
 * NOTE: This API only supports ct_dim=1 and rt_dim=1 (single output tile).
 * Operand B is used as stored: a transpose of operand B is not implemented, and
 * compute_sdpa_chunk rejects transpose_v at compile time.
 *
 * Return value: None
 *
 * | Argument       | Description                                                   | Type     | Valid Range                                      | Required |
 * |----------------|---------------------------------------------------------------|----------|--------------------------------------------------|----------|
 * | in0_cb_id      | The identifier of the first input circular buffer (CB)        | uint32_t | 0 to 31                                          | True     |
 * | in1_cb_id      | The identifier of the second input circular buffer (CB)       | uint32_t | 0 to 31                                          | True     |
 * | out_cb_id      | The identifier of the output circular buffer (CB)             | uint32_t | 0 to 31                                          | True     |
 * | kt_dim         | The inner dim of the input matrices in tiles                  | uint32_t | even number from 2 to 256                        | True     |
 * | nt_dim         | The number of SrcA tiles per K iteration                      | uint32_t | 1 to 16                                          | True     |
 *
 * All arguments are required: the former transpose argument sat between out_cb_id and
 * kt_dim, so a legacy argument list must not re-bind transpose as kt_dim. A bool at
 * the old transpose position is rejected at compile time; an integer there binds as
 * kt_dim and is caught only at run time, by the kt_dim/nt_dim range asserts, and only
 * when LLK asserts are enabled (TT_METAL_LLK_ASSERTS); a default build runs it.
 */
// clang-format on
ALWI void sdpa_custom_mm_reuse_dest_srcb_block_init(
    std::uint32_t in0_cb_id,
    std::uint32_t in1_cb_id,
    std::uint32_t out_cb_id,
    std::uint32_t kt_dim,
    std::uint32_t nt_dim) {
    // Intentionally swap in0 and in1 as operation specific hw_configures are deprecated
    UNPACK((llk_unpack_hw_configure<DST_ACCUM_MODE>(in1_cb_id, in0_cb_id)));
    UNPACK((llk_unpack_AB_sdpa_custom_mm_reuse_dest_srcb_init(in0_cb_id, in1_cb_id, nt_dim)));

    MATH((llk_math_sdpa_custom_mm_reuse_dest_srcb_init<MATH_FIDELITY>(in0_cb_id, in1_cb_id, kt_dim)));
    MATH((llk_math_pack_sync_init<DST_ACCUM_MODE>()));
    MATH((llk_math_hw_configure<DST_ACCUM_MODE>(in0_cb_id, in1_cb_id)));

    PACK((llk_pack_hw_configure<DST_ACCUM_MODE>(out_cb_id)));
    PACK((llk_pack_init<ckernel::PackMode::Default, false>(out_cb_id)));
    PACK((llk_pack_dest_init<DST_ACCUM_MODE, ckernel::PackMode::Default>(out_cb_id)));
}

// Legacy (in0, in1, out, transpose, kt_dim) argument lists have the same arity as the
// current (in0, in1, out, kt_dim, nt_dim) list. This overload is viable only for an exact
// bool at the old transpose position and is deleted, so a bool transpose does not re-bind
// as kt_dim; integers resolve to the live overload. A transpose of operand B is not
// implemented: see the transpose_v static_assert in compute_sdpa_chunk.
template <typename T, std::enable_if_t<std::is_same_v<T, bool>, int> = 0>
ALWI void sdpa_custom_mm_reuse_dest_srcb_block_init(
    std::uint32_t in0_cb_id, std::uint32_t in1_cb_id, std::uint32_t out_cb_id, T transpose, std::uint32_t kt_dim) =
    delete;

ALWI void sdpa_custom_mm_reuse_dest_srcb_block_init_short(
    std::uint32_t in0_cb_id,
    std::uint32_t in1_cb_id,
    std::uint32_t out_cb_id,
    std::uint32_t kt_dim,
    std::uint32_t nt_dim) {
    UNPACK((llk_unpack_AB_sdpa_custom_mm_reuse_dest_srcb_init(in0_cb_id, in1_cb_id, nt_dim)));
    MATH((llk_math_sdpa_custom_mm_reuse_dest_srcb_init<MATH_FIDELITY>(in0_cb_id, in1_cb_id, kt_dim)));
}

// Deleted for the same reason as the sdpa_custom_mm_reuse_dest_srcb_block_init overload above.
template <typename T, std::enable_if_t<std::is_same_v<T, bool>, int> = 0>
ALWI void sdpa_custom_mm_reuse_dest_srcb_block_init_short(
    std::uint32_t in0_cb_id, std::uint32_t in1_cb_id, std::uint32_t out_cb_id, T transpose, std::uint32_t kt_dim) =
    delete;

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
 * Operand B is used as stored: a transpose of operand B is not implemented.
 *
 * Runtime parameter signal_output:
 *   false: Normal operation without signaling
 *   true: Signal SFPU semaphore for pipelining with subsequent operations
 *
 * Usage pattern for partial K:
 *   for (k = 0; k < num_k_subblocks - 1; k++) {
 *     sdpa_custom_mm_reuse_dest_srcb_block(..., false);  // Accumulate without signaling
 *   }
 *   sdpa_custom_mm_reuse_dest_srcb_block(..., true);  // Final K subblock with signaling
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
 * | kt_dim         | The inner dimension (K reduction dimension).                            | uint32_t | Must be equal to block A column dimension      | True     |
 * | nt_dim         | The number of SrcA tiles per K iteration.                               | uint32_t | 1 to 16                                        | True     |
 * | in1_k_stride   | Stride between K tiles in in1 CB (1 for contiguous K tiles).            | uint32_t | >= 1                                           | True     |
 * | signal_output  | Signal SFPU semaphore for pipelining.                                   | bool     | true or false                                  | True     |
 *
 * All arguments are required: the former transpose argument sat between idst and kt_dim,
 * so a legacy argument list must not re-bind transpose as kt_dim. A bool at the old
 * transpose position is rejected at compile time; an integer there binds as kt_dim and
 * is caught only at run time, by the kt_dim/nt_dim range asserts, and only when LLK
 * asserts are enabled (TT_METAL_LLK_ASSERTS); a default build runs it.
 */
// clang-format on
template <std::uint32_t output_granularity, std::uint32_t input_granularity = 1>
ALWI void sdpa_custom_mm_reuse_dest_srcb_block(
    std::uint32_t in0_cb_id,
    std::uint32_t in1_cb_id,
    std::uint32_t in0_tile_index,
    std::uint32_t in1_tile_index,
    std::uint32_t isrc,
    std::uint32_t idst,
    std::uint32_t kt_dim,
    std::uint32_t nt_dim,
    std::uint32_t in1_k_stride,
    bool signal_output) {
    UNPACK((llk_unpack_A_sdpa_set_srcb_dummy_valid()));
    UNPACK((llk_unpack_AB_sdpa_custom_mm_reuse_dest_srcb(
        in0_cb_id, in1_cb_id, in0_tile_index, in1_tile_index, kt_dim, nt_dim, in1_k_stride)));
    MATH((llk_math_sdpa_custom_mm_reuse_dest_srcb<MATH_FIDELITY, output_granularity, input_granularity>(
        isrc, idst, kt_dim, nt_dim, signal_output)));
}

// Legacy (..., idst, transpose, kt_dim, nt_dim, in1_k_stride) argument lists have the same
// arity as the current (..., idst, kt_dim, nt_dim, in1_k_stride, signal_output) list. This
// overload is viable only for an exact bool at the old transpose position and is deleted,
// so a bool transpose does not re-bind as kt_dim; integers resolve to the live overload.
// A transpose of operand B is not implemented: see the transpose_v static_assert in
// compute_sdpa_chunk.
template <
    std::uint32_t output_granularity,
    std::uint32_t input_granularity = 1,
    typename T,
    std::enable_if_t<std::is_same_v<T, bool>, int> = 0>
ALWI void sdpa_custom_mm_reuse_dest_srcb_block(
    std::uint32_t in0_cb_id,
    std::uint32_t in1_cb_id,
    std::uint32_t in0_tile_index,
    std::uint32_t in1_tile_index,
    std::uint32_t isrc,
    std::uint32_t idst,
    T transpose,
    std::uint32_t kt_dim,
    std::uint32_t nt_dim,
    std::uint32_t in1_k_stride) = delete;

}  // namespace ckernel
