// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// The insertion step of the single-core topk kernel on one 2-tile slab, for the tile0_sorted argument of topk_local_sort:
// the slab is sorted with i_end_phase 5, tile 1 is copied in again and the slab is sorted a second time with TOPK_TILE0_SORTED.

#include <cstdint>

#include "ckernel.h"
#include "llk_defs.h"
#include "params.h"

std::uint32_t unp_cfg_context          = 0;
std::uint32_t pack_sync_tile_dst_ptr   = 0;
std::uint32_t math_sync_tile_dst_index = 0;

static_assert(TOPK_NUM_ITERATIONS == 1, "topk_presorted_test sorts one 2-tile slab (a matrix of 2 value and 2 index tiles)");
static_assert(!TOPK_FUSED_STABLE, "the fused mode is not part of the insertion step this test covers");

constexpr int NUM_STAGES = 2; // values, indices

// ============================================================================
#ifdef LLK_TRISC_UNPACK
#include "llk_unpack_A.h"
#include "llk_unpack_common.h"

inline void unpack_tile(std::uint32_t l1_address, std::uint32_t src_format, std::uint32_t dst_format, bool first_configuration)
{
    if (first_configuration)
    {
        _llk_unpack_hw_configure_<is_fp32_dest_acc_en>(src_format, src_format, dst_format, dst_format, FACE_R_DIM, FACE_R_DIM, 4, 4);
    }
    else
    {
        _llk_unpack_reconfig_data_format_srca_impl_<is_fp32_dest_acc_en, p_dim_stride_target::IGNORE, false>(src_format, dst_format, 16 * 16 * 4);
    }
    _llk_unpack_A_init_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(
        1 /* transpose_of_faces */, 1 /* within_face_16x16_transpose */, ckernel::DEFAULT_TENSOR_SHAPE, src_format, dst_format);
    _llk_unpack_A_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(l1_address, src_format, dst_format);
}

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    const std::uint32_t src[NUM_STAGES] = {formats.unpack_A_src, ckernel::to_underlying(DataFormat::UInt16)};
    const std::uint32_t dst[NUM_STAGES] = {formats.unpack_A_dst, ckernel::to_underlying(DataFormat::UInt16)};

    // The slab: value tiles 0 and 1, index tiles 2 and 3.
    unpack_tile(L1_ADDRESS(params.buffer_A[0]), src[0], dst[0], true);
    unpack_tile(L1_ADDRESS(params.buffer_A[1]), src[0], dst[0], false);
    unpack_tile(L1_ADDRESS(params.buffer_A[2]), src[1], dst[1], false);
    unpack_tile(L1_ADDRESS(params.buffer_A[3]), src[1], dst[1], false);
    // The incoming tile of the second sort: tile 1 again (values, then indices).
    unpack_tile(L1_ADDRESS(params.buffer_A[1]), src[0], dst[0], false);
    unpack_tile(L1_ADDRESS(params.buffer_A[3]), src[1], dst[1], false);
}
#endif

// ============================================================================
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

inline void copy_tile_to_dest(std::uint32_t dst_tile, std::uint32_t math_format, bool first_configuration)
{
    if (first_configuration)
    {
        _llk_math_hw_configure_<is_fp32_dest_acc_en>(math_format, math_format);
    }
    else
    {
        _llk_math_reconfig_data_format_srca_<is_fp32_dest_acc_en, false>(math_format);
    }
    _llk_math_eltwise_unary_datacopy_init_wrapper_<DataCopyType::A2D, is_fp32_dest_acc_en, BroadcastType::NONE, false, PackMode::Default>(4, math_format);
    _llk_math_eltwise_unary_datacopy_<DataCopyType::A2D, DstSync::SyncHalf, is_fp32_dest_acc_en, BroadcastType::NONE, unpack_to_dest>(
        dst_tile, math_format, math_format);
}

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    constexpr bool APPROX              = false;
    constexpr bool NETWORK_STABLE_SORT = TOPK_STABLE_SORT;
    constexpr bool TOPK_LARGEST        = (TOPK_SORT_DIRECTION == 0);
    constexpr auto TOPK_TIE_ORDER      = TOPK_LARGEST ? ckernel::sfpu::TopkTieOrder::Descending : ckernel::sfpu::TopkTieOrder::Ascending;
    static_assert(!(TOPK_RANK_STAMPED && TOPK_STABLE_SORT), "rank-stamped and comparator stable modes are mutually exclusive");
    static_assert(!TOPK_RANK_STAMPED || is_fp32_dest_acc_en, "rank-stamped stable topk requires 32-bit DEST (dest_acc)");
    constexpr std::uint32_t dst_index = 0;
    constexpr int end_phase           = 5;
    constexpr VectorMode vector_mode  = VectorMode::RC_custom;

    const std::uint32_t math_format[NUM_STAGES] = {formats.math, ckernel::to_underlying(DataFormat::UInt16)};

    _llk_math_pack_sync_init_<dest_sync, is_fp32_dest_acc_en>();
    _llk_math_eltwise_unary_sfpu_init_<SfpuType::topk_local_sort>();
    if constexpr (TOPK_RANK_STAMPED)
    {
        ckernel::sfpu::_init_topk_rank_stamped_<TOPK_TAG_BITS>();
    }
    else
    {
        ckernel::sfpu::_init_topk();
    }

    _llk_math_wait_for_dest_available_<dest_sync>();

    // The slab into DEST tiles 0 to 3.
    copy_tile_to_dest(0, math_format[0], true);
    copy_tile_to_dest(1, math_format[0], false);
    copy_tile_to_dest(2, math_format[1], false);
    copy_tile_to_dest(3, math_format[1], false);

    for (int sort = 0; sort < 2; sort++)
    {
        if (sort == 1)
        {
            // The incoming tile of the insertion step: tile 1 again, values then indices.
            copy_tile_to_dest(1, math_format[0], false);
            copy_tile_to_dest(3, math_format[1], false);
        }
        if constexpr (TOPK_RANK_STAMPED)
        {
            // Re-stamp both tiles before every sort, as the kernel does.
            SFPU_UNARY_CALL(dest_sync, is_fp32_dest_acc_en, calculate_topk_stamp_local_positions, (APPROX, TOPK_LARGEST, TOPK_TAG_BITS), dst_index, vector_mode);
        }
        if constexpr (NETWORK_STABLE_SORT)
        {
            SFPU_UNARY_CALL(dest_sync, is_fp32_dest_acc_en, calculate_topk_canonicalize_negzero, (APPROX, is_fp32_dest_acc_en), dst_index, vector_mode);
        }
        const std::uint32_t tile0_sorted = (sort == 1 && TOPK_TILE0_SORTED) ? 1u : 0u;
        SFPU_UNARY_CALL(
            dest_sync,
            is_fp32_dest_acc_en,
            calculate_bitonic_topk_local_sort,
            (APPROX, is_fp32_dest_acc_en, NETWORK_STABLE_SORT, false /* FUSED */, TOPK_RANK_STAMPED, TOPK_TIE_ORDER),
            dst_index,
            vector_mode,
            TOPK_SORT_DIRECTION,
            end_phase,
            0 /* start_phase */,
            0 /* end_step */,
            0 /* start_step */,
            tile0_sorted);
    }

    if constexpr (TOPK_RANK_STAMPED)
    {
        ckernel::sfpu::_topk_strip_rank_tags_<TOPK_TAG_BITS>(0);
        ckernel::sfpu::_topk_strip_rank_tags_<TOPK_TAG_BITS>(1);
    }
    if constexpr (is_fp32_dest_acc_en)
    {
        // The uint16 index tiles into the half of the 32-bit DEST words the packer reads.
        ckernel::sfpu::_topk_uint16_move_dest_tile_to_pack_half_(2);
        ckernel::sfpu::_topk_uint16_move_dest_tile_to_pack_half_(3);
    }

    _llk_math_dest_section_done_<dest_sync, is_fp32_dest_acc_en>();
}
#endif

// ============================================================================
#ifdef LLK_TRISC_PACK
#include "llk_lib_pack_wrappers.h"
#include "llk_pack_common.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    const std::uint32_t pack_src[NUM_STAGES] = {formats.pack_src, ckernel::to_underlying(DataFormat::UInt16)};
    const std::uint32_t pack_dst[NUM_STAGES] = {formats.pack_dst, ckernel::to_underlying(DataFormat::UInt16)};

    _llk_pack_dest_init_wrapper_<dest_sync, is_fp32_dest_acc_en, PackMode::Default>();
    _llk_packer_wait_for_math_done_();

    // DEST tiles 0 and 1 (values) to result tiles 0 and 1, DEST tiles 2 and 3 (indices) to result tiles 2 and 3.
    for (int stage = 0; stage < NUM_STAGES; stage++)
    {
        if (stage == 0)
        {
            _llk_pack_hw_configure_wrapper_<is_fp32_dest_acc_en, PackMode::Default>(pack_src[0], pack_dst[0], 16 * 16 * 4);
        }
        else
        {
            _llk_pack_reconfig_data_format_wrapper_<is_fp32_dest_acc_en, false>(pack_src[1], pack_dst[1], 16 * 16 * 4, FACE_R_DIM, TILE_C_DIM, 4, false, false, 1);
        }
        _llk_pack_init_wrapper_<PackMode::Default, false>(pack_dst[stage]);
        for (int t = 0; t < 2; t++)
        {
            const int tile = stage * 2 + t;
            _llk_pack_<dest_sync, is_fp32_dest_acc_en, PackMode::Default>(tile, L1_ADDRESS(params.buffer_Res[tile]));
        }
    }

    _llk_pack_dest_section_done_<dest_sync, is_fp32_dest_acc_en>();
}
#endif
