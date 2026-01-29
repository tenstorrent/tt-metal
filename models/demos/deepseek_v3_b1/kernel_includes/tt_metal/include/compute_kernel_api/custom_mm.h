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

template <bool transpose = false, bool split_acc = false>
ALWI void custom_mm_block_init(
    const std::uint32_t in0_cb_id, const std::uint32_t in1_cb_id, const std::uint32_t out_cb_id) {
    UNPACK((llk_unpack_hw_configure<DST_ACCUM_MODE>(in1_cb_id, in0_cb_id)));
    UNPACK((llk_unpack_AB_custom_mm_init<transpose>(in0_cb_id, in1_cb_id)));

    MATH((llk_math_pack_sync_init<DST_ACCUM_MODE>()));
    MATH((llk_math_hw_configure<DST_ACCUM_MODE>(in0_cb_id, in1_cb_id)));
    MATH((llk_math_custom_mm_init<transpose, split_acc>(in0_cb_id, in1_cb_id)));

    PACK((llk_pack_dest_init<DST_ACCUM_MODE, false>()));
    PACK((llk_pack_hw_configure<DST_ACCUM_MODE>(out_cb_id)));
    PACK((llk_pack_init<false, false>(out_cb_id)));
}

template <bool finalize = true>
ALWI void custom_mm_block(
    const std::uint32_t in0_cb_id,
    const std::uint32_t in1_cb_id,
    const std::uint32_t in0_tile_index,
    const std::uint32_t in1_tile_index,
    const std::uint32_t dst_index,
    const std::uint32_t kt_dim) {
    UNPACK((llk_unpack_AB_custom_mm(in0_cb_id, in1_cb_id, in0_tile_index, in1_tile_index, kt_dim)));
    MATH((llk_math_custom_mm<finalize>(dst_index, kt_dim)));
}

ALWI void custom_mm_block_unpack(
    const std::uint32_t in0_cb_id,
    const std::uint32_t in1_cb_id,
    const std::uint32_t in0_tile_index,
    const std::uint32_t in1_tile_index,
    const std::uint32_t kt_dim) {
    UNPACK((llk_unpack_AB_custom_mm(in0_cb_id, in1_cb_id, in0_tile_index, in1_tile_index, kt_dim)));
}

template <bool finalize = true>
ALWI void custom_mm_block_math(const std::uint32_t dst_index, const std::uint32_t kt_dim) {
    MATH((llk_math_custom_mm<finalize>(dst_index, kt_dim)));
}

}  // namespace ckernel
