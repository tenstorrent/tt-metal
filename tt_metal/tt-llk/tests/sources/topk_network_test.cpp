// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// TopK bitonic network test: runs the SFPU network entry points on one freshly loaded 2-tile slab
// (2 value tiles + 2 index tiles, transposed into column layout on unpack) per 32-row tile row and
// packs all four tiles back, so every datum the network touched is visible in the result.
//
//   TOPK_NETWORK_OP == 0: local sort only, phases 0..TOPK_LOGK-1 (sorted runs of TOPK_K datums)
//   TOPK_NETWORK_OP == 1: local sort + merge(m_iter = 0, K) + rebuild(m_iter = 0, K, skip_second)
//   TOPK_RAW_FP32: Float32 value and index words unpacked straight to 32-bit DEST without the
//   transpose (the network then sorts DEST columns of the untransposed tiles); index words are
//   small integers carried as raw bits, so values and indices both round-trip bit-exactly.
//
// Unlike topk_test.cpp (fixed K = 32, final top-K tile only) this sweeps K, the local-sort
// direction and the rebuild skip_second argument, and runs the unstable / comparator-stable
// network in both 16-bit and 32-bit DEST.

#include <cstdint>

#include "ckernel.h"
#include "llk_defs.h"
#include "params.h"

std::uint32_t unp_cfg_context          = 0;
std::uint32_t pack_sync_tile_dst_ptr   = 0;
std::uint32_t math_sync_tile_dst_index = 0;

constexpr int NUM_STAGES          = 2; // values, indices
constexpr int NUM_TILES_PER_STAGE = 2;
constexpr int NUM_TILES_PER_ROW   = NUM_STAGES * NUM_TILES_PER_STAGE;

#ifdef LLK_TRISC_UNPACK
#include "llk_unpack_A.h"
#include "llk_unpack_common.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    // TOPK_RAW_FP32: both stages are raw 32-bit words unpacked straight into DEST (no transpose),
    // so full fp32 bit patterns (signed zeros, denormals, NaN payloads) reach the network intact.
    const std::uint32_t unpack_src_data_types[NUM_STAGES] = {
        formats.unpack_A_src, TOPK_RAW_FP32 ? formats.unpack_A_src : ckernel::to_underlying(DataFormat::UInt16)};
    const std::uint32_t unpack_dst_data_types[NUM_STAGES] = {
        formats.unpack_A_dst, TOPK_RAW_FP32 ? formats.unpack_A_dst : ckernel::to_underlying(DataFormat::UInt16)};
    constexpr std::uint32_t TRANSPOSE = TOPK_RAW_FP32 ? 0 : 1;

    for (std::uint32_t tile_row = 0; tile_row < params.FULL_RT_DIM; ++tile_row)
    {
        for (int stage = 0; stage < NUM_STAGES; ++stage)
        {
            if (tile_row == 0 && stage == 0)
            {
                _llk_unpack_hw_configure_<is_fp32_dest_acc_en>(
                    unpack_src_data_types[stage],
                    unpack_src_data_types[stage],
                    unpack_dst_data_types[stage],
                    unpack_dst_data_types[stage],
                    FACE_R_DIM,
                    FACE_R_DIM,
                    4 /* num_faces */,
                    4 /* num_faces */);
            }
            else
            {
                _llk_unpack_reconfig_data_format_srca_impl_<is_fp32_dest_acc_en, p_dim_stride_target::IGNORE, false /* to_from_int8 */>(
                    unpack_src_data_types[stage], unpack_dst_data_types[stage], 16 * 16 * 4 /* tile_size */);
            }

            // Transpose into the column layout the network sorts along.
            _llk_unpack_A_init_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(
                TRANSPOSE /* transpose_of_faces */,
                TRANSPOSE /* within_face_16x16_transpose */,
                ckernel::DEFAULT_TENSOR_SHAPE,
                unpack_src_data_types[stage],
                unpack_dst_data_types[stage]);

            const std::uint32_t first_tile = tile_row * NUM_TILES_PER_ROW + stage * NUM_TILES_PER_STAGE;
            for (int t = 0; t < NUM_TILES_PER_STAGE; ++t)
            {
                _llk_unpack_A_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(
                    L1_ADDRESS(params.buffer_A[first_tile + t]), unpack_src_data_types[stage], unpack_dst_data_types[stage]);
            }
        }
    }
}
#endif // LLK_TRISC_UNPACK

#ifdef LLK_TRISC_MATH
#include "ckernel_sfpu.h"
#include "llk_lib_math_wrappers.h"

using namespace ckernel;

#define DST_SYNC_MODE  dest_sync
#define DST_ACCUM_MODE is_fp32_dest_acc_en
#include "llk_sfpu/ckernel_sfpu_topk.h"
#include "llk_sfpu/llk_math_eltwise_unary_sfpu_macros.h"
#undef DST_SYNC_MODE
#undef DST_ACCUM_MODE

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    constexpr bool APPROX         = false;
    constexpr bool TOPK_LARGEST   = (TOPK_IDIR == 0); // idir 0 = descending / largest-first
    constexpr auto TOPK_TIE_ORDER = TOPK_LARGEST ? ckernel::sfpu::TopkTieOrder::Descending : ckernel::sfpu::TopkTieOrder::Ascending;
    static_assert(!TOPK_FUSED_STABLE || is_fp32_dest_acc_en, "fused keys require 32-bit DEST");
    static_assert(!TOPK_RANK_STAMPED || is_fp32_dest_acc_en, "rank-stamped keys require 32-bit DEST");
    constexpr std::uint32_t dst_index = 0;
    constexpr VectorMode vector_mode  = VectorMode::RC_custom;

    const std::uint32_t math_data_types[NUM_STAGES] = {formats.math, TOPK_RAW_FP32 ? formats.math : ckernel::to_underlying(DataFormat::UInt16)};
    static_assert(
        !TOPK_RAW_FP32 || (is_fp32_dest_acc_en && unpack_to_dest && !TOPK_FUSED_STABLE && !TOPK_RANK_STAMPED),
        "raw fp32 mode: 32-bit DEST, unpack to DEST, plain keys");

    _llk_math_pack_sync_init_<dest_sync, is_fp32_dest_acc_en>();
    _llk_math_eltwise_unary_sfpu_init_<SfpuType::topk_local_sort>();
    if constexpr (TOPK_FUSED_STABLE)
    {
        ckernel::sfpu::_init_topk_fused_();
    }
    else if constexpr (TOPK_RANK_STAMPED)
    {
        ckernel::sfpu::_init_topk_rank_stamped_<16>();
    }
    else
    {
        ckernel::sfpu::_init_topk();
    }

    for (std::uint32_t tile_row = 0; tile_row < params.FULL_RT_DIM; ++tile_row)
    {
        _llk_math_wait_for_dest_available_<dest_sync>();

        for (int stage = 0; stage < NUM_STAGES; ++stage)
        {
            const std::uint32_t math_format = math_data_types[stage];
            if (tile_row == 0 && stage == 0)
            {
                _llk_math_hw_configure_<is_fp32_dest_acc_en>(math_format, math_format);
            }
            else
            {
                _llk_math_reconfig_data_format_srca_<is_fp32_dest_acc_en, false /* to_from_int8 */>(math_format);
            }
            _llk_math_eltwise_unary_datacopy_init_wrapper_<
                DataCopyType::A2D,
                is_fp32_dest_acc_en,
                BroadcastType::NONE,
                false /* is_int_fpu_en */,
                PackMode::Default>(
                /*num_rows_per_matrix=*/4, /*math_format=*/math_format);
            for (int t = 0; t < NUM_TILES_PER_STAGE; ++t)
            {
                _llk_math_eltwise_unary_datacopy_<DataCopyType::A2D, DstSync::SyncHalf, is_fp32_dest_acc_en, BroadcastType::NONE, unpack_to_dest>(
                    /*dst_tile_index=*/stage * NUM_TILES_PER_STAGE + t, math_format, math_format);
            }
        }

        // Per-slab key preparation, exactly as the ttnn engines do it.
        if constexpr (TOPK_FUSED_STABLE)
        {
            SFPU_UNARY_CALL(dest_sync, is_fp32_dest_acc_en, calculate_topk_fuse, (APPROX, TOPK_LARGEST), dst_index, vector_mode);
        }
        if constexpr (TOPK_RANK_STAMPED)
        {
            SFPU_UNARY_CALL(dest_sync, is_fp32_dest_acc_en, calculate_topk_stamp_local_positions, (APPROX, TOPK_LARGEST, 16), dst_index, vector_mode);
        }
        if constexpr (TOPK_STABLE_SORT)
        {
            SFPU_UNARY_CALL(dest_sync, is_fp32_dest_acc_en, calculate_topk_canonicalize_negzero, (APPROX, is_fp32_dest_acc_en), dst_index, vector_mode);
        }

        SFPU_UNARY_CALL(
            dest_sync,
            is_fp32_dest_acc_en,
            calculate_bitonic_topk_phases_steps,
            (APPROX, is_fp32_dest_acc_en, TOPK_STABLE_SORT, TOPK_FUSED_STABLE, TOPK_RANK_STAMPED, TOPK_TIE_ORDER),
            dst_index,
            vector_mode,
            TOPK_IDIR,
            TOPK_LOGK - 1 /* end_phase */,
            0 /* start_phase */,
            0 /* end_step */,
            0 /* start_step */);

        if constexpr (TOPK_NETWORK_OP == 1)
        {
            SFPU_UNARY_CALL(
                dest_sync,
                is_fp32_dest_acc_en,
                calculate_bitonic_topk_merge,
                (APPROX, is_fp32_dest_acc_en, TOPK_IDIR, TOPK_STABLE_SORT, TOPK_FUSED_STABLE, TOPK_RANK_STAMPED, TOPK_TIE_ORDER, 16),
                dst_index,
                vector_mode,
                0 /* m_iter */,
                TOPK_K);
            SFPU_UNARY_CALL(
                dest_sync,
                is_fp32_dest_acc_en,
                calculate_bitonic_topk_rebuild,
                (APPROX, is_fp32_dest_acc_en, TOPK_STABLE_SORT, TOPK_FUSED_STABLE, TOPK_RANK_STAMPED, TOPK_TIE_ORDER),
                dst_index,
                vector_mode,
                TOPK_IDIR,
                0 /* m_iter */,
                TOPK_K,
                TOPK_LOGK,
                TOPK_REBUILD_SKIP_SECOND);
        }

        // Pack-side preparation: bf16 value words and u16 index words in the halves the packer reads.
        if constexpr (TOPK_FUSED_STABLE)
        {
            SFPU_UNARY_CALL(
                dest_sync,
                is_fp32_dest_acc_en,
                calculate_topk_defuse,
                (APPROX, TOPK_LARGEST, ckernel::sfpu::TOPK_SFPSTORE_MODE_PACK_UINT16),
                dst_index,
                vector_mode,
                2 /*num_tiles*/);
        }
        else if constexpr (is_fp32_dest_acc_en && !TOPK_RAW_FP32)
        {
            if constexpr (TOPK_RANK_STAMPED)
            {
                ckernel::sfpu::_topk_strip_rank_tags_<16>(0);
                ckernel::sfpu::_topk_strip_rank_tags_<16>(1);
            }
            ckernel::sfpu::_topk_uint16_move_dest_tile_to_pack_half_(2);
            ckernel::sfpu::_topk_uint16_move_dest_tile_to_pack_half_(3);
        }

        _llk_math_dest_section_done_<dest_sync, is_fp32_dest_acc_en>();
    }
}
#endif // LLK_TRISC_MATH

#ifdef LLK_TRISC_PACK
#include "llk_lib_pack_wrappers.h"
#include "llk_pack_common.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    _llk_pack_dest_init_wrapper_<dest_sync, is_fp32_dest_acc_en, PackMode::Default>();

    const std::uint32_t pack_src_data_types[NUM_STAGES] = {formats.pack_src, TOPK_RAW_FP32 ? formats.pack_src : ckernel::to_underlying(DataFormat::UInt16)};
    const std::uint32_t pack_dst_data_types[NUM_STAGES] = {formats.pack_dst, TOPK_RAW_FP32 ? formats.pack_dst : ckernel::to_underlying(DataFormat::UInt16)};

    for (std::uint32_t tile_row = 0; tile_row < params.FULL_RT_DIM; ++tile_row)
    {
        _llk_packer_wait_for_math_done_();
        for (int stage = 0; stage < NUM_STAGES; ++stage)
        {
            if (tile_row == 0 && stage == 0)
            {
                _llk_pack_hw_configure_wrapper_<is_fp32_dest_acc_en, PackMode::Default>(
                    pack_src_data_types[stage], pack_dst_data_types[stage], 16 * 16 * 4 /* tile_size */);
            }
            else
            {
                _llk_pack_reconfig_data_format_wrapper_<is_fp32_dest_acc_en, false /* is_tile_dim_reconfig_en */>(
                    pack_src_data_types[stage],
                    pack_dst_data_types[stage],
                    16 * 16 * 4 /* tile_size */,
                    FACE_R_DIM,
                    TILE_C_DIM,
                    4 /* num_faces */,
                    false /* partial_face */,
                    false /* narrow_tile */,
                    1 /* num_tiles */);
            }
            _llk_pack_init_wrapper_<PackMode::Default, false /* zero_output */>(pack_dst_data_types[stage]);

            for (int t = 0; t < NUM_TILES_PER_STAGE; ++t)
            {
                const int dst_tile = stage * NUM_TILES_PER_STAGE + t;
                _llk_pack_<dest_sync, is_fp32_dest_acc_en, ckernel::PackMode::Default>(
                    dst_tile, L1_ADDRESS(params.buffer_Res[tile_row * NUM_TILES_PER_ROW + dst_tile]));
            }
        }
        _llk_pack_dest_section_done_<dest_sync, is_fp32_dest_acc_en>();
    }
}
#endif // LLK_TRISC_PACK
