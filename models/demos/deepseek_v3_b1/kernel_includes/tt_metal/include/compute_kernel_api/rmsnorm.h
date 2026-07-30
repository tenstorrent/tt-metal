// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/compute/compute_kernel_api.h"
#include "api/compute/common.h"
#ifdef TRISC_MATH
#include "../../hw/ckernels/blackhole/metal/llk_api/llk_math_rmsnorm_bcast_scalar_dest_reuse_api.h"
#endif
#ifdef TRISC_UNPACK
#include "../../hw/ckernels/blackhole/metal/llk_api/llk_unpack_A_rmsnorm_api.h"
#endif

#if !defined(UCK_CHLKC_MATH) && !defined(UCK_CHLKC_PACK)
[[maybe_unused]] static constexpr ckernel::MathFidelity MATH_FIDELITY = ckernel::MathFidelity::HiFi4;
#endif

namespace ckernel {

template <
    EltwiseBinaryType eltwise_binary_type = EltwiseBinaryType::ELWADD,
    uint32_t num_tiles,
    MathFidelity math_fidelity = MATH_FIDELITY,
    bool unpack_full_transpose = false>
ALWI void rmsnorm_bcast_scalar_reuse_tiles_init(uint32_t icb0) {
    UNPACK((llk_unpack_A_rmsnorm_init<num_tiles, BroadcastType::SCALAR, true, EltwiseBinaryReuseDestType::DEST_TO_SRCB>(
        unpack_full_transpose /*transpose_of_faces*/, unpack_full_transpose /*within_face_16x16_transpose*/, icb0)));
    MATH((llk_math_rmsnorm_bcast_scalar_dest_reuse_init_with_operands<eltwise_binary_type, num_tiles, math_fidelity>(
        icb0, icb0, false /*acc_to_dest*/)));
}

template <
    EltwiseBinaryType eltwise_binary_type = EltwiseBinaryType::ELWADD,
    uint32_t num_tiles,
    MathFidelity math_fidelity = MATH_FIDELITY,
    bool clear_dest = false>
ALWI void rmsnorm_bcast_scalar_reuse_tiles(
    uint32_t in_cb_id, uint32_t in_tile_index, uint32_t src_tile_index, uint32_t dst_tile_index) {
    UNPACK(
        (llk_unpack_A<BroadcastType::SCALAR, true, EltwiseBinaryReuseDestType::DEST_TO_SRCB>(in_cb_id, in_tile_index)));
    MATH((llk_math_rmsnorm_bcast_scalar_dest_reuse<
          eltwise_binary_type,
          num_tiles,
          DST_ACCUM_MODE,
          math_fidelity,
          clear_dest>(src_tile_index, dst_tile_index)));
}

template <uint32_t num_tiles>
ALWI void rmsnorm_mul_bcast_scalar_reuse_tiles_init(uint32_t icb0) {
    rmsnorm_bcast_scalar_reuse_tiles_init<EltwiseBinaryType::ELWMUL, num_tiles>(icb0);
}

template <uint32_t num_tiles, bool clear_dest = false>
ALWI void rmsnorm_mul_bcast_scalar_reuse_tiles(
    uint32_t in_cb_id, uint32_t in_tile_index, uint32_t src_tile_index, uint32_t dst_tile_index) {
    rmsnorm_bcast_scalar_reuse_tiles<EltwiseBinaryType::ELWMUL, num_tiles, MATH_FIDELITY, clear_dest>(
        in_cb_id, in_tile_index, src_tile_index, dst_tile_index);
}
}  // namespace ckernel
