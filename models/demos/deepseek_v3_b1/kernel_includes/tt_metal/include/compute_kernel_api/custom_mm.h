// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/compute/common.h"
#ifdef TRISC_MATH
#include "../../hw/ckernels/blackhole/metal/llk_api/llk_math_custom_mm_api.h"
#endif
#ifdef TRISC_UNPACK
#include "../../hw/ckernels/blackhole/metal/llk_api/llk_unpack_AB_custom_mm_api.h"
#endif
namespace ckernel {

template <bool transpose = false, bool split_acc = false, bool dense_packing = false, bool fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void custom_mm_block_init(
    const std::uint32_t in0_cb_id,
    const std::uint32_t in1_cb_id,
    const std::uint32_t out_cb_id,
    const std::uint32_t ct_dim = 1) {
    UNPACK((llk_unpack_hw_configure<fp32_dest_acc_en>(in1_cb_id, in0_cb_id)));
    UNPACK((llk_unpack_AB_custom_mm_init<transpose>(in0_cb_id, in1_cb_id, ct_dim)));

    MATH((llk_math_pack_sync_init<fp32_dest_acc_en>()));
    MATH((llk_math_hw_configure<fp32_dest_acc_en>(in0_cb_id, in1_cb_id)));
    MATH((llk_math_custom_mm_init<transpose, split_acc, dense_packing>(in0_cb_id, in1_cb_id, ct_dim)));

    PACK((llk_pack_dest_init<fp32_dest_acc_en, false>()));
    PACK((llk_pack_hw_configure<fp32_dest_acc_en>(out_cb_id)));
    PACK((llk_pack_init<false, false>(out_cb_id)));
    if constexpr (dense_packing) {
        PACK((cfg_reg_rmw_tensix<PCK0_ADDR_CTRL_ZW_REG_0_Wstride_RMW>(
            (TILE_NUM_FACES / 2) * FACE_C_DIM * FACE_R_DIM * 2)));
    }
}

template <
    bool transpose = false,
    bool split_acc = false,
    bool dense_packing = false,
    bool fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void custom_mm_block_init_short(
    const std::uint32_t in0_cb_id,
    const std::uint32_t in1_cb_id,
    const std::uint32_t out_cb_id,
    const std::uint32_t ct_dim = 1) {
    UNPACK((llk_unpack_AB_custom_mm_init<transpose>(in0_cb_id, in1_cb_id, ct_dim)));

    MATH((llk_math_custom_mm_init<transpose, split_acc, dense_packing>(in0_cb_id, in1_cb_id, ct_dim)));

    if constexpr (dense_packing) {
        PACK((cfg_reg_rmw_tensix<PCK0_ADDR_CTRL_ZW_REG_0_Wstride_RMW>(
            (TILE_NUM_FACES / 2) * FACE_C_DIM * FACE_R_DIM * 2)));
    }
}

template <bool finalize = true, bool read_transposed = false>
ALWI void custom_mm_block(
    const std::uint32_t in0_cb_id,
    const std::uint32_t in1_cb_id,
    const std::uint32_t in0_tile_index,
    const std::uint32_t in1_tile_index,
    const std::uint32_t dst_index,
    const std::uint32_t kt_dim,
    const std::uint32_t ct_dim = 1) {
    UNPACK((llk_unpack_AB_custom_mm<read_transposed>(
        in0_cb_id, in1_cb_id, in0_tile_index, in1_tile_index, kt_dim, ct_dim)));
    MATH((llk_math_custom_mm<finalize>(in0_cb_id, in1_cb_id, dst_index, kt_dim, ct_dim)));
}

template <bool read_transposed = false>
ALWI void custom_mm_block_unpack(
    const std::uint32_t in0_cb_id,
    const std::uint32_t in1_cb_id,
    const std::uint32_t in0_tile_index,
    const std::uint32_t in1_tile_index,
    const std::uint32_t kt_dim,
    const std::uint32_t ct_dim = 1) {
    UNPACK((llk_unpack_AB_custom_mm<read_transposed>(
        in0_cb_id, in1_cb_id, in0_tile_index, in1_tile_index, kt_dim, ct_dim)));
}

template <bool finalize = true>
ALWI void custom_mm_block_math(
    const std::uint32_t in0_cb_id,
    const std::uint32_t in1_cb_id,
    const std::uint32_t dst_index,
    const std::uint32_t kt_dim,
    const std::uint32_t ct_dim = 1) {
    MATH((llk_math_custom_mm<finalize>(in0_cb_id, in1_cb_id, dst_index, kt_dim, ct_dim)));
}

template <bool dense_packing = false>
ALWI void custom_mm_block_uninit() {
    if constexpr (dense_packing) {
        PACK((cfg_reg_rmw_tensix<PCK0_ADDR_CTRL_ZW_REG_0_Wstride_RMW>(TILE_NUM_FACES * FACE_C_DIM * FACE_R_DIM * 2)));
    }
}

}  // namespace ckernel
