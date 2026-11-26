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
 * @brief Initialize unpack and math for broadcast dest-reuse binary ops (explicit fidelity).
 *
 * Computes `DST[dst_tile_index + t] = icb0[in_tile_index + t] op bcast(DST[src_tile_index])`
 * for `t in [0, num_tiles)`, where `bcast_type` is either `BroadcastType::SCALAR` (element `[0]`
 * of `DST[src_tile_index]`) or `BroadcastType::COL` (the 16-row column-0 vector of Face 0 in
 * `DST[src_tile_index]`, shared across all faces).
 *
 * @tparam eltwise_binary_type FPU binary op (`ELWADD`, `ELWSUB`, or `ELWMUL`).
 * @tparam bcast_type Broadcast mode (`BroadcastType::SCALAR` or `BroadcastType::COL`).
 * @tparam num_tiles Number of consecutive tiles walked per call.
 * @tparam math_fidelity FPU math fidelity (non-LoFi phases apply to `ELWMUL`).
 * @tparam unpack_full_transpose Fold a full 32x32 transpose into the unpack (`num_tiles == 1`,
 *         4-face tiles only).
 * @param icb0 Input circular buffer (16-row faces with 2 or 4 faces for `BroadcastType::COL`;
 *        1, 2, or 4 faces for `BroadcastType::SCALAR`).
 * @note Call before `bcast_dest_reuse_tiles`. Do not interleave `copy_init` / `datacopy` between
 *       init and execution without re-initializing.
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
 * @brief Apply `DST[dst_tile_index + t] = in_cb_id[in_tile_index + t] op bcast(DST[src_tile_index])`
 *        with explicit `MathFidelity`.
 *
 * @tparam eltwise_binary_type FPU binary op (`ELWADD`, `ELWSUB`, or `ELWMUL`).
 * @tparam bcast_type Broadcast mode (`BroadcastType::SCALAR` or `BroadcastType::COL`).
 * @tparam num_tiles Number of consecutive tiles walked per call (must match init).
 * @tparam math_fidelity FPU math fidelity (must match init).
 * @tparam clear_dest When true, zero the acquired DEST section between `MOVD2B` and the FPU MOP.
 *         `ELWMUL` accumulates into DEST, so set `true` unless the output tiles are already zero.
 * @tparam is_fp32_dest_acc_en Whether FP32 DEST accumulation is enabled (defaults to `DST_ACCUM_MODE`).
 * @param in_cb_id Input circular buffer ID.
 * @param in_tile_index Starting tile index in `in_cb_id`.
 * @param src_tile_index DEST tile index holding the scalar (`SCALAR`) or Face 0 column-0 vector (`COL`).
 * @param dst_tile_index Starting DEST tile index for the output tiles.
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
