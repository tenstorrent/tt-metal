// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include "api/compute/common_globals.h"
#include "api/compute/experimental/2_0/llk_operand.h"

#ifdef TRISC_MATH
#include "llk_math_reduce_api.h"
#endif

#ifdef TRISC_UNPACK
#include "llk_unpack_AB_reduce_api.h"
#endif

#ifdef TRISC_PACK
#include "llk_pack_reduce_api.h"
#endif

namespace ckernel {
namespace experimental {

// Id-free (2.0) reduce (data + scaler -> reduced). Takes one LLKOperand per input (data, scaler) and one for
// the output. Format-free at the op level (formats set at compute_kernel_hw_startup); every LLK core here
// consumes only geometry (data tile shape / output face_r_dim) + the two runtime input addresses. Geometry
// (for MATH + unpack init) comes from the DATA operand. Packing is separate (experimental::pack_tile); the
// packer edge mask is programmed here by reduce_init.

// clang-format off
/**
 * Reduce init: programs UNPACK (AB reduce), MATH, and the PACK edge-mask. compute_kernel_hw_startup(data,
 * scaler, out) must already have programmed the formats. Uses only DATA geometry and OUT's face_r_dim;
 * the scaler operand contributes nothing at init (it's passed to reduce_tile).
 *
 * | Template | reduce_type | SUM / AVG / MAX                                | PoolType  | | True |
 * | Template | reduce_dim  | REDUCE_ROW / REDUCE_COL / REDUCE_SCALAR       | ReduceDim | | True |
 * | Template | is_fp32_dest_acc_en | fp32 dest-accumulate mode              | bool       | | False |
 * | Function | data        | Input operand A (reduced); drives geometry    | LLKOperand | | True |
 * | Function | out         | Output operand (drives the packer edge mask)  | LLKOperand | | True |
 */
// clang-format on
template <
    PoolType reduce_type,
    ReduceDim reduce_dim,
    bool is_fp32_dest_acc_en = DST_ACCUM_MODE,
    DataFormat DF,
    TensorShape DS,
    DataFormat OF,
    TensorShape OS>
ALWI void reduce_init(LLKOperand<DF, DS> /*data*/, LLKOperand<OF, OS> /*out*/) {
    static_assert(is_legal_tile_shape(DS), "reduce_init: illegal data tile shape.");
    static_assert(is_legal_tile_shape(OS), "reduce_init: illegal output tile shape.");
    UNPACK((llk_unpack_AB_reduce_init_impl<reduce_type, reduce_dim>(DS)));
    MATH((llk_math_reduce_init_impl<reduce_type, reduce_dim, is_fp32_dest_acc_en, MATH_FIDELITY>(DS)));
    PACK((llk_pack_reduce_mask_config_impl<reduce_dim, PackMode::Default>(OS.face_r_dim)));
}

// clang-format off
/**
 * Reduce one tile of data (with the scaler tile) into DST[idst]. Pair with reduce_init. DST must be acquired.
 * itile / itile_scaler index within data / scaler. Per-tile input addresses are derived from each operand's
 * geometry via tile_stride_words (one-tile page; exp section included for block floats).
 *
 * | Template | is_fp32_dest_acc_en    | fp32 dest-accumulate mode          | bool       | | False |
 * | Function | data / scaler          | Input operands                    | LLKOperand | | True |
 * | Function | itile / itile_scaler   | Tile indices within data / scaler | uint32_t   | | True |
 * | Function | idst                   | DST register index for the result | uint32_t   | | True |
 */
// clang-format on
template <
    PoolType reduce_type,
    ReduceDim reduce_dim,
    bool is_fp32_dest_acc_en = DST_ACCUM_MODE,
    DataFormat DF,
    TensorShape DS,
    DataFormat SF,
    TensorShape SS>
ALWI void reduce_tile(
    LLKOperand<DF, DS> data,
    LLKOperand<SF, SS> scaler,
    std::uint32_t itile,
    std::uint32_t itile_scaler,
    std::uint32_t idst) {
    static_assert(is_legal_tile_shape(DS), "reduce_tile: illegal data tile shape.");
    static_assert(is_legal_tile_shape(SS), "reduce_tile: illegal scaler tile shape.");
    // REDUCE_ROW for any pool other than MAX physically swaps the SrcA/SrcB byte streams in the LLK, but the
    // src-register formats are programmed once at compute_kernel_hw_startup and not re-derived here -- so the
    // swap only produces correct results when the data and scaler share one L1 format.
    static_assert(
        !(reduce_dim == ReduceDim::REDUCE_ROW && reduce_type != PoolType::MAX) || DF == SF,
        "reduce_tile: REDUCE_ROW (non-MAX) swaps SrcA/SrcB, so data and scaler must have the same data format.");
    MATH((llk_math_reduce<reduce_type, reduce_dim, is_fp32_dest_acc_en, MATH_FIDELITY>(idst, DS)));
    UNPACK((llk_unpack_AB_reduce_impl<reduce_type, reduce_dim>(
        detail::tile_address(data, itile), detail::tile_address(scaler, itile_scaler))));
}

// clang-format off
/**
 * Reduce `ntiles` consecutive data tiles (each combined with the scaler tile), writing each tile's reduction
 * to its OWN DST slot: data tile (start_itile + i) -> DST[start_idst + i]. Loop form of reduce_tile (NOT a
 * cross-block accumulation into one slot -- for that, loop reduce_tile at a fixed idst). Pair with
 * reduce_init. DST must be acquired.
 *
 * | Template | reduce_type   | SUM / AVG / MAX                                 | PoolType   | | True |
 * | Template | reduce_dim    | REDUCE_ROW / REDUCE_COL / REDUCE_SCALAR        | ReduceDim  | | True |
 * | Function | data / scaler | Input operands                                 | LLKOperand | | True |
 * | Function | start_itile   | Index of the first data tile within `data`     | uint32_t   | | True |
 * | Function | itile_scaler  | Tile index within `scaler`                      | uint32_t   | | True |
 * | Function | start_idst    | DST register index for the first tile's result | uint32_t   | start_idst + ntiles <= DST size | True |
 * | Function | ntiles        | Number of consecutive data tiles to reduce     | uint32_t   | | True |
 * | Template | is_fp32_dest_acc_en | fp32 dest-accumulate mode                 | bool       | | False |
 */
// clang-format on
template <
    PoolType reduce_type,
    ReduceDim reduce_dim,
    bool is_fp32_dest_acc_en = DST_ACCUM_MODE,
    DataFormat DF,
    TensorShape DS,
    DataFormat SF,
    TensorShape SS>
ALWI void reduce_block(
    LLKOperand<DF, DS> data,
    LLKOperand<SF, SS> scaler,
    std::uint32_t start_itile,
    std::uint32_t itile_scaler,
    std::uint32_t start_idst,
    std::uint32_t ntiles) {
    static_assert(is_legal_tile_shape(DS), "reduce_block: illegal data tile shape.");
    static_assert(is_legal_tile_shape(SS), "reduce_block: illegal scaler tile shape.");
    for (std::uint32_t i = 0; i < ntiles; ++i) {
        reduce_tile<reduce_type, reduce_dim, is_fp32_dest_acc_en>(
            data, scaler, start_itile + i, itile_scaler, start_idst + i);
    }
}

// clang-format off
/**
 * Math-only reduction into DST[idst]: the MATH half of reduce_tile with no unpack/pack. Source tiles must
 * already be in the source registers (e.g. staged by a preceding fused op like tilize); takes no LLKOperand.
 * Geometry comes from num_faces (row-major face layout). Pair with reduce_init. DST must be acquired. For a
 * non-default face layout, use the TensorShape overload below.
 *
 * | Template | reduce_type | SUM / AVG / MAX                          | PoolType  | | True |
 * | Template | reduce_dim  | REDUCE_ROW / REDUCE_COL / REDUCE_SCALAR | ReduceDim | | True |
 * | Function | idst        | DST register index for the result       | uint32_t  | | True |
 * | Function | num_faces   | Number of faces to reduce (default 4)   | uint32_t  | 1 / 2 / 4 | False |
 */
// clang-format on
template <PoolType reduce_type, ReduceDim reduce_dim>
ALWI void reduce_tile_math(std::uint32_t idst, std::uint32_t num_faces = MAX_NUM_FACES) {
    MATH((llk_math_reduce<reduce_type, reduce_dim, DST_ACCUM_MODE, MATH_FIDELITY>(
        idst, tensor_shape_from_num_faces(MAX_FACE_R_DIM, num_faces))));
}

// clang-format off
/**
 * Math-only reduction into DST[idst] with an explicit tile geometry. As above, source tiles must already be
 * in the source registers; takes no LLKOperand. Pair with reduce_init. DST must be acquired.
 *
 * | Template | reduce_type  | SUM / AVG / MAX                          | PoolType             | | True |
 * | Template | reduce_dim   | REDUCE_ROW / REDUCE_COL / REDUCE_SCALAR | ReduceDim            | | True |
 * | Function | idst         | DST register index for the result       | uint32_t             | | True |
 * | Function | tensor_shape | Tile geometry to reduce                  | ckernel::TensorShape | | True |
 */
// clang-format on
template <PoolType reduce_type, ReduceDim reduce_dim>
ALWI void reduce_tile_math(std::uint32_t idst, const ckernel::TensorShape& tensor_shape) {
    MATH((llk_math_reduce<reduce_type, reduce_dim, DST_ACCUM_MODE, MATH_FIDELITY>(idst, tensor_shape)));
}

// clang-format off
/**
 * Reduce uninit: reset the MATH reduce state and clear the packer edge mask back to default.
 */
// clang-format on
ALWI void reduce_uninit() {
    MATH((llk_math_reduce_uninit()));
    PACK((llk_pack_reduce_mask_clear()));
}

}  // namespace experimental
}  // namespace ckernel
