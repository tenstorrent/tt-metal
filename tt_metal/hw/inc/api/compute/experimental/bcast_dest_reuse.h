// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#if defined(TRISC_MATH) && defined(ARCH_BLACKHOLE)
#include "experimental/llk_math_bcast_dest_reuse_api.h"
#endif
#if defined(TRISC_UNPACK) && defined(ARCH_BLACKHOLE)
#include "experimental/llk_unpack_A_bcast_dest_reuse_api.h"
#endif

namespace ckernel {

#if defined(ARCH_BLACKHOLE)

/**
 * @brief Initialize unpack and math for broadcast dest-reuse (`SCALAR` or `COL`) with explicit fidelity.
 * @note Do not interleave `copy_init` / `datacopy` before `bcast_dest_reuse_tiles` without re-initializing.
 */
template <
    EltwiseBinaryType eltwise_binary_type,
    BroadcastType bcast_type,
    std::uint32_t num_tiles,
    MathFidelity math_fidelity,
    bool unpack_full_transpose = false>
ALWI void bcast_dest_reuse_tiles_init(std::uint32_t icb0) {
    UNPACK((llk_unpack_A_bcast_dest_reuse_init<num_tiles, bcast_type, true, EltwiseBinaryReuseDestType::DEST_TO_SRCB>(
        unpack_full_transpose /*transpose_of_faces*/, unpack_full_transpose /*within_face_16x16_transpose*/, icb0)));
    MATH((llk_math_bcast_dest_reuse_init<eltwise_binary_type, bcast_type, num_tiles, math_fidelity>(
        icb0, false /*acc_to_dest*/)));
}

/**
 * @brief Initialize unpack and math for broadcast dest-reuse binary ops using the kernel's `MATH_FIDELITY`.
 *
 * Overload of `bcast_dest_reuse_tiles_init` when callers want the compile-time program fidelity
 * (`MATH_FIDELITY`, defined on TRISC_MATH only) instead of specifying `MathFidelity` explicitly.
 */
template <
    EltwiseBinaryType eltwise_binary_type,
    BroadcastType bcast_type,
    std::uint32_t num_tiles,
    bool unpack_full_transpose = false>
ALWI void bcast_dest_reuse_tiles_init(std::uint32_t icb0) {
    UNPACK((llk_unpack_A_bcast_dest_reuse_init<num_tiles, bcast_type, true, EltwiseBinaryReuseDestType::DEST_TO_SRCB>(
        unpack_full_transpose /*transpose_of_faces*/, unpack_full_transpose /*within_face_16x16_transpose*/, icb0)));
    MATH((llk_math_bcast_dest_reuse_init<eltwise_binary_type, bcast_type, num_tiles, MATH_FIDELITY>(
        icb0, false /*acc_to_dest*/)));
}

/**
 * @brief `DST[dst_tile_index + t] = in_cb_id[in_tile_index + t] op bcast(DST[src_tile_index])`
 *        for `t in [0, num_tiles)` with explicit `MathFidelity`.
 * @tparam clear_dest Zero the acquired DEST section after `MOVD2B` (`ELWMUL` accumulates into DEST).
 */
template <
    EltwiseBinaryType eltwise_binary_type,
    BroadcastType bcast_type,
    std::uint32_t num_tiles,
    MathFidelity math_fidelity,
    bool clear_dest = false,
    bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void bcast_dest_reuse_tiles(
    std::uint32_t in_cb_id, std::uint32_t in_tile_index, std::uint32_t src_tile_index, std::uint32_t dst_tile_index) {
    UNPACK((llk_unpack_A<bcast_type, true, EltwiseBinaryReuseDestType::DEST_TO_SRCB>(in_cb_id, in_tile_index)));
    MATH((llk_math_bcast_dest_reuse<
          eltwise_binary_type,
          bcast_type,
          num_tiles,
          is_fp32_dest_acc_en,
          math_fidelity,
          clear_dest>(in_cb_id, src_tile_index, dst_tile_index)));
}

/**
 * @brief Apply broadcast dest-reuse binary op using the kernel's `MATH_FIDELITY`.
 *
 * Pair with the program-fidelity overload of `bcast_dest_reuse_tiles_init`.
 */
template <
    EltwiseBinaryType eltwise_binary_type,
    BroadcastType bcast_type,
    std::uint32_t num_tiles,
    bool clear_dest = false,
    bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void bcast_dest_reuse_tiles(
    std::uint32_t in_cb_id, std::uint32_t in_tile_index, std::uint32_t src_tile_index, std::uint32_t dst_tile_index) {
    UNPACK((llk_unpack_A<bcast_type, true, EltwiseBinaryReuseDestType::DEST_TO_SRCB>(in_cb_id, in_tile_index)));
    MATH((llk_math_bcast_dest_reuse<
          eltwise_binary_type,
          bcast_type,
          num_tiles,
          is_fp32_dest_acc_en,
          MATH_FIDELITY,
          clear_dest>(in_cb_id, src_tile_index, dst_tile_index)));
}

#endif  // ARCH_BLACKHOLE

}  // namespace ckernel
