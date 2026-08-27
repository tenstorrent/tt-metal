// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/compute/compute_kernel_api.h"
#include "api/compute/common.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/fill.h"
#include "api/compute/experimental/mul_reduce_scalar.h"
// Blackhole-only: the rmsnorm LLKs live only in the Blackhole llk_api / llk_lib trees.
#if defined(TRISC_MATH) && defined(ARCH_BLACKHOLE)
#include "experimental/llk_math_rmsnorm_bcast_scalar_dest_reuse_api.h"
#endif
#if defined(TRISC_UNPACK) && defined(ARCH_BLACKHOLE)
#include "experimental/llk_unpack_A_rmsnorm_api.h"
#endif

namespace ckernel {

#if defined(ARCH_BLACKHOLE)

template <EltwiseBinaryType eltwise_binary_type = EltwiseBinaryType::ELWADD, uint32_t num_tiles>
ALWI void rmsnorm_bcast_scalar_reuse_tiles_init(uint32_t icb0) {
    UNPACK((llk_unpack_A_rmsnorm_init<num_tiles, BroadcastType::SCALAR, true, EltwiseBinaryReuseDestType::DEST_TO_SRCB>(
        false, false, icb0)));
    MATH((llk_math_rmsnorm_bcast_scalar_dest_reuse_init_with_operands<eltwise_binary_type, num_tiles, MATH_FIDELITY>(
        icb0, icb0, false /*acc_to_dest*/)));
}

template <
    EltwiseBinaryType eltwise_binary_type = EltwiseBinaryType::ELWADD,
    uint32_t num_tiles,
    bool clear_dest = false,
    bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void rmsnorm_bcast_scalar_reuse_tiles(
    uint32_t in_cb_id, uint32_t in_tile_index, uint32_t src_tile_index, uint32_t dst_tile_index) {
    UNPACK(
        (llk_unpack_A<BroadcastType::SCALAR, true, EltwiseBinaryReuseDestType::DEST_TO_SRCB>(in_cb_id, in_tile_index)));
    MATH((llk_math_rmsnorm_bcast_scalar_dest_reuse<
          eltwise_binary_type,
          num_tiles,
          is_fp32_dest_acc_en,
          MATH_FIDELITY,
          clear_dest>(src_tile_index, dst_tile_index)));
}

// Explicit-fidelity variants for callers that need transpose-fold or per-use precision.
// MATH_FIDELITY is only defined on the math thread, so it cannot be a default template
// argument here; these separately-named variants take fidelity explicitly instead.
template <
    EltwiseBinaryType eltwise_binary_type,
    uint32_t num_tiles,
    MathFidelity math_fidelity,
    bool unpack_full_transpose = false>
ALWI void rmsnorm_bcast_scalar_reuse_tiles_init_fidelity(uint32_t icb0) {
    UNPACK((llk_unpack_A_rmsnorm_init<num_tiles, BroadcastType::SCALAR, true, EltwiseBinaryReuseDestType::DEST_TO_SRCB>(
        unpack_full_transpose /*transpose_of_faces*/, unpack_full_transpose /*within_face_16x16_transpose*/, icb0)));
    MATH((llk_math_rmsnorm_bcast_scalar_dest_reuse_init_with_operands<eltwise_binary_type, num_tiles, math_fidelity>(
        icb0, icb0, false /*acc_to_dest*/)));
}

template <
    EltwiseBinaryType eltwise_binary_type,
    uint32_t num_tiles,
    MathFidelity math_fidelity,
    bool clear_dest = false,
    bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void rmsnorm_bcast_scalar_reuse_tiles_fidelity(
    uint32_t in_cb_id, uint32_t in_tile_index, uint32_t src_tile_index, uint32_t dst_tile_index) {
    UNPACK(
        (llk_unpack_A<BroadcastType::SCALAR, true, EltwiseBinaryReuseDestType::DEST_TO_SRCB>(in_cb_id, in_tile_index)));
    MATH((llk_math_rmsnorm_bcast_scalar_dest_reuse<
          eltwise_binary_type,
          num_tiles,
          is_fp32_dest_acc_en,
          math_fidelity,
          clear_dest>(src_tile_index, dst_tile_index)));
}

template <uint32_t num_tiles>
ALWI void rmsnorm_mul_bcast_scalar_reuse_tiles_init(uint32_t icb0) {
    rmsnorm_bcast_scalar_reuse_tiles_init<EltwiseBinaryType::ELWMUL, num_tiles>(icb0);
}

template <uint32_t num_tiles, bool clear_dest = false, bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void rmsnorm_mul_bcast_scalar_reuse_tiles(
    uint32_t in_cb_id, uint32_t in_tile_index, uint32_t src_tile_index, uint32_t dst_tile_index) {
    rmsnorm_bcast_scalar_reuse_tiles<EltwiseBinaryType::ELWMUL, num_tiles, clear_dest, is_fp32_dest_acc_en>(
        in_cb_id, in_tile_index, src_tile_index, dst_tile_index);
}

/**
 * Reduce a row of products whose tile count exceeds DST capacity.
 *
 * One DST slot is reserved as a cross-chunk accumulator; the remaining
 * dst_capacity - 1 slots stage products. The caller must initialize
 * mul_reduce_scalar and add_binary, then acquire DST before calling.
 *
 * ocb programs the packer's face_r_dim for the reduce mask. On return,
 * DST[dst_capacity - 1] contains the scaled sum of products.
 * The reduce pack mask remains configured; call
 * mul_reduce_scalar_uninit() before normal packing.
 */
template <
    uint32_t num_tiles,
    uint32_t dst_capacity,
    PoolType reduce_type = PoolType::SUM,
    bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void mul_reduce_scalar_chunked_tile(uint32_t icb0, uint32_t icb1, uint32_t ocb, float scaler = 1.0f) {
    static_assert(reduce_type == PoolType::SUM, "Only SUM reduction is currently supported");
    static_assert(dst_capacity >= 2 && dst_capacity <= 8, "Chunked reduction requires 2 to 8 DST slots");
    static_assert(
        dst_capacity <= get_dest_max_tiles<DST_SYNC_MODE, is_fp32_dest_acc_en, DstTileShape::Tile32x32>(),
        "dst_capacity exceeds the DST tiles available in this sync/accum mode");
    static_assert(num_tiles > dst_capacity, "Use mul_reduce_scalar_tile when the row fits DST");

    constexpr uint32_t batch_size = dst_capacity - 1;
    constexpr uint32_t accumulator = batch_size;
    constexpr uint32_t num_batches = (num_tiles + batch_size - 1) / batch_size;
    constexpr uint32_t last_batch_size = num_tiles - (num_batches - 1) * batch_size;

    fill_tile(accumulator, 0.0f);

    for (uint32_t batch = 0; batch < num_batches; ++batch) {
        const uint32_t input_start = batch * batch_size;
        const uint32_t count = batch + 1 < num_batches ? batch_size : last_batch_size;

        // Each reduction consumes the UNPACK/MATH state. The caller provides
        // the first initialization; subsequent chunks restore it here.
        if (batch > 0) {
            mul_reduce_scalar_init(icb0, icb1);
        }
        for (uint32_t j = 0; j < count; ++j) {
            UNPACK((llk_unpack_AB(icb0, icb1, input_start + j, input_start + j)));
            MATH((llk_math_eltwise_mul_reduce_scalar<is_fp32_dest_acc_en, MATH_FIDELITY>(j, icb0)));
        }

        UNPACK((llk_unpack_mul_reduce_scalar_switch_to_reduce()));
        MATH((llk_math_mul_reduce_scalar_reduce_init<is_fp32_dest_acc_en, MATH_FIDELITY>()));
        MATH((llk_math_mul_reduce_scalar_move_dest_to_src<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(0)));
        MATH(SFPU_UNARY_CALL(
            DST_SYNC_MODE,
            is_fp32_dest_acc_en,
            _calculate_fill_,
            (APPROX, 2 /*ITERATIONS*/),
            0 /*dst_index*/,
            VectorMode::RC_custom,
            scaler));
        MATH((llk_math_mul_reduce_scalar_move_dest_to_src<EltwiseBinaryReuseDestType::DEST_TO_SRCB>(0)));
        MATH(SFPU_UNARY_CALL(
            DST_SYNC_MODE,
            is_fp32_dest_acc_en,
            _calculate_fill_,
            (APPROX, 2 /*ITERATIONS*/),
            0 /*dst_index*/,
            VectorMode::RC_custom,
            0.0f));

        if (batch == 0) {
            PACK((llk_pack_reduce_mask_config<ReduceDim::REDUCE_SCALAR, ckernel::PackMode::Default>(ocb)));
        }
        MATH((llk_math_mul_reduce_column<MATH_FIDELITY>(0, icb0)));
        for (uint32_t j = 1; j < count; ++j) {
            MATH((llk_math_mul_reduce_scalar_move_dest_to_src<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(j)));
            MATH((llk_math_mul_reduce_column<MATH_FIDELITY>(0, icb0)));
        }
        MATH((llk_math_mul_reduce_scalar<MATH_FIDELITY>()));
        MATH((llk_math_mul_reduce_scalar_clear_dvalid()));
        add_binary_tile<DstRoundingMode::Default, is_fp32_dest_acc_en>(accumulator, 0, accumulator);
    }
}

#endif  // ARCH_BLACKHOLE

}  // namespace ckernel
