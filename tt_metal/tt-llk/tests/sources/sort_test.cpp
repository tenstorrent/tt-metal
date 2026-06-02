// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// LLK regression for ttnn.sort's bitonic-merge algorithm. Mirrors
//   ttnn/cpp/ttnn/operations/data_movement/sort/device/kernels/compute/sort_single_row_single_core.cpp
//   ttnn/cpp/ttnn/operations/data_movement/sort/device/kernels/compute/sort_common.hpp
// at the LLK level — local_sort + merge ONLY (no rebuild), which is the path
// that fails in tt-metal#37571 (topk_test.cpp covers the rebuild path).
//
// buffer_A per tile-row: [0..Wt-1] = value tiles, [Wt..2*Wt-1] = uint16 index tiles.

#include <algorithm>
#include <cstdint>
#include <type_traits>

#include "ckernel.h"
#include "llk_defs.h"
#include "params.h"

// Globals required by the test framework.
std::uint32_t unp_cfg_context          = 0;
std::uint32_t pack_sync_tile_dst_ptr   = 0;
std::uint32_t math_sync_tile_dst_index = 0;

// Stage indices: 0 = Values, 1 = Indices.
constexpr int NUM_STAGES = 2;

// DEST register layout (matches metal sort kernel):
//   DEST[0] = value tile low,  DEST[1] = value tile high
//   DEST[2] = index tile low,  DEST[3] = index tile high
constexpr int input_dest_start = 0;
constexpr int input_dest_end   = 1;
constexpr int index_dest_start = 2;
constexpr int index_dest_end   = 3;

// Same hard-coded constants as the metal sort kernel.
constexpr int SORT_K         = 64;
constexpr int SORT_END_PHASE = 5; // log2(64)-1

// Sort direction (this LLK reproducer always sorts ascending).
constexpr bool ASCENDING = true;

// Helper: count log2 of a power-of-2 integer.
static constexpr std::uint32_t ilog2_ce(std::uint32_t n)
{
    std::uint32_t r = 0;
    while ((1u << r) < n)
    {
        ++r;
    }
    return r;
}

// ============================================================================
// UNPACK TRISC
// ============================================================================

#ifdef LLK_TRISC_UNPACK
#include "llk_unpack_A.h"
#include "llk_unpack_common.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    const std::uint32_t FULL_CT_DIM = params.FULL_CT_DIM;
    const std::uint32_t Wt          = FULL_CT_DIM / NUM_STAGES;
    const std::uint32_t NUM_ROWS    = params.FULL_RT_DIM;
    const std::uint32_t STAGES      = ilog2_ce(Wt);

    const std::uint32_t unpack_src_data_types[NUM_STAGES] = {formats.unpack_A_src, ckernel::to_underlying(DataFormat::UInt16)};
    const std::uint32_t unpack_dst_data_types[NUM_STAGES] = {formats.unpack_A_dst, ckernel::to_underlying(DataFormat::UInt16)};

    _llk_unpack_hw_configure_<is_fp32_dest_acc_en>(
        unpack_src_data_types[0],
        unpack_src_data_types[0],
        unpack_dst_data_types[0],
        unpack_dst_data_types[0],
        /*unpA_face_r_dim=*/FACE_R_DIM,
        /*unpB_face_r_dim=*/FACE_R_DIM,
        /*unpA_num_faces=*/4,
        /*unpB_num_faces=*/4);

    for (std::uint32_t row = 0; row < NUM_ROWS; ++row)
    {
        const std::uint32_t row_offset = row * FULL_CT_DIM;

        // ----- 1. Bitonic-sequence formation -----
        for (std::uint32_t wt = 0; wt < Wt; wt += 2)
        {
            // Values pair (wt, wt+1) - face-level transpose.
            _llk_unpack_reconfig_data_format_srca_impl_<is_fp32_dest_acc_en, p_dim_stride_target::IGNORE, false>(
                unpack_src_data_types[0], unpack_dst_data_types[0], /*tile_size=*/16 * 16 * 4);
            _llk_unpack_A_init_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(
                /*transpose_of_faces=*/1, /*within_face_16x16_transpose=*/1, FACE_R_DIM, /*num_faces=*/4, unpack_src_data_types[0], unpack_dst_data_types[0]);
            _llk_unpack_A_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(
                L1_ADDRESS(params.buffer_A[row_offset + wt]), unpack_src_data_types[0], unpack_dst_data_types[0]);
            _llk_unpack_A_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(
                L1_ADDRESS(params.buffer_A[row_offset + wt + 1]), unpack_src_data_types[0], unpack_dst_data_types[0]);

            // Indices pair (Wt+wt, Wt+wt+1) - face-level transpose.
            _llk_unpack_reconfig_data_format_srca_impl_<is_fp32_dest_acc_en, p_dim_stride_target::IGNORE, false>(
                unpack_src_data_types[1], unpack_dst_data_types[1], /*tile_size=*/16 * 16 * 4);
            _llk_unpack_A_init_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(
                /*transpose_of_faces=*/1, /*within_face_16x16_transpose=*/1, FACE_R_DIM, /*num_faces=*/4, unpack_src_data_types[1], unpack_dst_data_types[1]);
            _llk_unpack_A_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(
                L1_ADDRESS(params.buffer_A[row_offset + Wt + wt]), unpack_src_data_types[1], unpack_dst_data_types[1]);
            _llk_unpack_A_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(
                L1_ADDRESS(params.buffer_A[row_offset + Wt + wt + 1]), unpack_src_data_types[1], unpack_dst_data_types[1]);
        }

        // ----- 2. Bitonic merge stages -----
        for (std::uint32_t stage = 2; stage <= STAGES; ++stage)
        {
            for (std::uint32_t sub = stage; sub > 0; --sub)
            {
                const std::uint32_t sub_dist = 1u << (sub - 1);
                for (std::uint32_t i = 0; i < Wt; ++i)
                {
                    const std::uint32_t j = i ^ sub_dist;
                    if (j > i)
                    {
                        // Values - no face transpose (data already column-wise after local_sort).
                        _llk_unpack_reconfig_data_format_srca_impl_<is_fp32_dest_acc_en, p_dim_stride_target::IGNORE, false>(
                            unpack_src_data_types[0], unpack_dst_data_types[0], /*tile_size=*/16 * 16 * 4);
                        _llk_unpack_A_init_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(
                            /*transpose_of_faces=*/0,
                            /*within_face_16x16_transpose=*/0,
                            FACE_R_DIM,
                            /*num_faces=*/4,
                            unpack_src_data_types[0],
                            unpack_dst_data_types[0]);
                        _llk_unpack_A_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(
                            L1_ADDRESS(params.buffer_A[row_offset + i]), unpack_src_data_types[0], unpack_dst_data_types[0]);
                        _llk_unpack_A_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(
                            L1_ADDRESS(params.buffer_A[row_offset + j]), unpack_src_data_types[0], unpack_dst_data_types[0]);

                        // Indices - no face transpose.
                        _llk_unpack_reconfig_data_format_srca_impl_<is_fp32_dest_acc_en, p_dim_stride_target::IGNORE, false>(
                            unpack_src_data_types[1], unpack_dst_data_types[1], /*tile_size=*/16 * 16 * 4);
                        _llk_unpack_A_init_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(
                            /*transpose_of_faces=*/0,
                            /*within_face_16x16_transpose=*/0,
                            FACE_R_DIM,
                            /*num_faces=*/4,
                            unpack_src_data_types[1],
                            unpack_dst_data_types[1]);
                        _llk_unpack_A_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(
                            L1_ADDRESS(params.buffer_A[row_offset + Wt + i]), unpack_src_data_types[1], unpack_dst_data_types[1]);
                        _llk_unpack_A_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(
                            L1_ADDRESS(params.buffer_A[row_offset + Wt + j]), unpack_src_data_types[1], unpack_dst_data_types[1]);
                    }
                }
            }
        }

        // ----- 3. Final result-copy pass: unpack each tile (no transpose) -----
        for (std::uint32_t t = 0; t < FULL_CT_DIM; ++t)
        {
            const std::uint32_t stage_idx = (t < Wt) ? 0u : 1u;
            _llk_unpack_reconfig_data_format_srca_impl_<is_fp32_dest_acc_en, p_dim_stride_target::IGNORE, false>(
                unpack_src_data_types[stage_idx], unpack_dst_data_types[stage_idx], /*tile_size=*/16 * 16 * 4);
            _llk_unpack_A_init_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(
                /*transpose_of_faces=*/0,
                /*within_face_16x16_transpose=*/0,
                FACE_R_DIM,
                /*num_faces=*/4,
                unpack_src_data_types[stage_idx],
                unpack_dst_data_types[stage_idx]);
            _llk_unpack_A_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(
                L1_ADDRESS(params.buffer_A[row_offset + t]), unpack_src_data_types[stage_idx], unpack_dst_data_types[stage_idx]);
        }
    }
}
#endif // LLK_TRISC_UNPACK

// ============================================================================
// MATH TRISC
// ============================================================================

#ifdef LLK_TRISC_MATH
#include "ckernel_sfpu.h"
#include "llk_math_common.h"
#include "llk_math_eltwise_unary_datacopy.h"

using namespace ckernel;

#define DST_SYNC_MODE  dest_sync
#define DST_ACCUM_MODE is_fp32_dest_acc_en
#include "llk_sfpu/llk_math_eltwise_unary_sfpu_topk.h"
#undef DST_SYNC_MODE
#undef DST_ACCUM_MODE

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    const std::uint32_t FULL_CT_DIM = params.FULL_CT_DIM;
    const std::uint32_t Wt          = FULL_CT_DIM / NUM_STAGES;
    const std::uint32_t NUM_ROWS    = params.FULL_RT_DIM;
    const std::uint32_t STAGES      = ilog2_ce(Wt);

    constexpr bool APPROX             = false;
    constexpr std::uint32_t dst_index = 0;
    constexpr VectorMode vector_mode  = VectorMode::RC_custom;
    constexpr int start_phase         = 0;
    constexpr int end_step            = 0;
    constexpr int start_step          = 0;

    const std::uint32_t math_data_types[NUM_STAGES] = {formats.math, ckernel::to_underlying(DataFormat::UInt16)};

    _llk_math_pack_sync_init_<dest_sync, is_fp32_dest_acc_en>();
    _llk_math_eltwise_unary_sfpu_init_<SfpuType::topk_local_sort>();
    ckernel::sfpu::_init_topk();

    _llk_math_hw_configure_<is_fp32_dest_acc_en>(math_data_types[0], math_data_types[0]);

    for (std::uint32_t row = 0; row < NUM_ROWS; ++row)
    {
        // ----- 1. Bitonic-sequence formation -----
        bool ascending_local = ASCENDING;
        for (std::uint32_t wt = 0; wt < Wt; wt += 2)
        {
            _llk_math_wait_for_dest_available_<dest_sync>();

            // Values reconfig + datacopy.
            _llk_math_reconfig_data_format_srca_<is_fp32_dest_acc_en, false>(math_data_types[0]);
#ifdef ARCH_BLACKHOLE
            _llk_math_eltwise_unary_datacopy_init_<DataCopyType::A2D, is_fp32_dest_acc_en, BroadcastType::NONE, false, PackMode::Default>(
                /*num_faces=*/4, /*dst_format=*/math_data_types[0]);
#else
            _llk_math_eltwise_unary_datacopy_init_<DataCopyType::A2D, is_fp32_dest_acc_en, BroadcastType::NONE, false>(
                /*num_faces=*/4, /*dst_format=*/math_data_types[0]);
#endif
            _llk_math_eltwise_unary_datacopy_<DataCopyType::A2D, DstSync::SyncHalf, is_fp32_dest_acc_en, BroadcastType::NONE, unpack_to_dest>(
                input_dest_start, math_data_types[0], math_data_types[0]);
            _llk_math_eltwise_unary_datacopy_<DataCopyType::A2D, DstSync::SyncHalf, is_fp32_dest_acc_en, BroadcastType::NONE, unpack_to_dest>(
                input_dest_end, math_data_types[0], math_data_types[0]);

            // Indices reconfig + datacopy.
            _llk_math_reconfig_data_format_srca_<is_fp32_dest_acc_en, false>(math_data_types[1]);
#ifdef ARCH_BLACKHOLE
            _llk_math_eltwise_unary_datacopy_init_<DataCopyType::A2D, is_fp32_dest_acc_en, BroadcastType::NONE, false, PackMode::Default>(
                /*num_faces=*/4, /*dst_format=*/math_data_types[1]);
#else
            _llk_math_eltwise_unary_datacopy_init_<DataCopyType::A2D, is_fp32_dest_acc_en, BroadcastType::NONE, false>(
                /*num_faces=*/4, /*dst_format=*/math_data_types[1]);
#endif
            _llk_math_eltwise_unary_datacopy_<DataCopyType::A2D, DstSync::SyncHalf, is_fp32_dest_acc_en, BroadcastType::NONE, unpack_to_dest>(
                index_dest_start, math_data_types[1], math_data_types[1]);
            _llk_math_eltwise_unary_datacopy_<DataCopyType::A2D, DstSync::SyncHalf, is_fp32_dest_acc_en, BroadcastType::NONE, unpack_to_dest>(
                index_dest_end, math_data_types[1], math_data_types[1]);

            _llk_math_eltwise_unary_sfpu_params_(
                ckernel::sfpu::calculate_bitonic_topk_phases_steps<APPROX, is_fp32_dest_acc_en, /*STABLE=*/false>,
                dst_index,
                vector_mode,
                /*idir=*/(int)ascending_local,
                SORT_END_PHASE,
                start_phase,
                end_step,
                start_step);

            _llk_math_dest_section_done_<dest_sync, is_fp32_dest_acc_en>();
            ascending_local = !ascending_local;
        }

        // ----- 2. Bitonic merge stages -----
        for (std::uint32_t stage = 2; stage <= STAGES; ++stage)
        {
            const std::uint32_t m_iter = stage - 1;
            for (std::uint32_t sub = stage; sub > 0; --sub)
            {
                const std::uint32_t sub_dist = 1u << (sub - 1);
                for (std::uint32_t i = 0; i < Wt; ++i)
                {
                    const std::uint32_t j = i ^ sub_dist;
                    if (j > i)
                    {
                        const bool ascending_block = ((i >> stage) & 1) == 0;
                        const bool dir             = (ascending_block == ASCENDING);

                        _llk_math_wait_for_dest_available_<dest_sync>();

                        // Values reconfig + datacopy.
                        _llk_math_reconfig_data_format_srca_<is_fp32_dest_acc_en, false>(math_data_types[0]);
#ifdef ARCH_BLACKHOLE
                        _llk_math_eltwise_unary_datacopy_init_<DataCopyType::A2D, is_fp32_dest_acc_en, BroadcastType::NONE, false, PackMode::Default>(
                            /*num_faces=*/4, /*dst_format=*/math_data_types[0]);
#else
                        _llk_math_eltwise_unary_datacopy_init_<DataCopyType::A2D, is_fp32_dest_acc_en, BroadcastType::NONE, false>(
                            /*num_faces=*/4, /*dst_format=*/math_data_types[0]);
#endif
                        _llk_math_eltwise_unary_datacopy_<DataCopyType::A2D, DstSync::SyncHalf, is_fp32_dest_acc_en, BroadcastType::NONE, unpack_to_dest>(
                            input_dest_start, math_data_types[0], math_data_types[0]);
                        _llk_math_eltwise_unary_datacopy_<DataCopyType::A2D, DstSync::SyncHalf, is_fp32_dest_acc_en, BroadcastType::NONE, unpack_to_dest>(
                            input_dest_end, math_data_types[0], math_data_types[0]);

                        // Indices reconfig + datacopy.
                        _llk_math_reconfig_data_format_srca_<is_fp32_dest_acc_en, false>(math_data_types[1]);
#ifdef ARCH_BLACKHOLE
                        _llk_math_eltwise_unary_datacopy_init_<DataCopyType::A2D, is_fp32_dest_acc_en, BroadcastType::NONE, false, PackMode::Default>(
                            /*num_faces=*/4, /*dst_format=*/math_data_types[1]);
#else
                        _llk_math_eltwise_unary_datacopy_init_<DataCopyType::A2D, is_fp32_dest_acc_en, BroadcastType::NONE, false>(
                            /*num_faces=*/4, /*dst_format=*/math_data_types[1]);
#endif
                        _llk_math_eltwise_unary_datacopy_<DataCopyType::A2D, DstSync::SyncHalf, is_fp32_dest_acc_en, BroadcastType::NONE, unpack_to_dest>(
                            index_dest_start, math_data_types[1], math_data_types[1]);
                        _llk_math_eltwise_unary_datacopy_<DataCopyType::A2D, DstSync::SyncHalf, is_fp32_dest_acc_en, BroadcastType::NONE, unpack_to_dest>(
                            index_dest_end, math_data_types[1], math_data_types[1]);

                        if (sub == 1)
                        {
                            _llk_math_eltwise_unary_sfpu_params_(
                                ckernel::sfpu::calculate_bitonic_topk_phases_steps<APPROX, is_fp32_dest_acc_en, /*STABLE=*/false>,
                                dst_index,
                                vector_mode,
                                /*idir=*/(int)dir,
                                SORT_END_PHASE,
                                start_phase,
                                end_step,
                                start_step);
                        }
                        else
                        {
                            // top_min = false (compile-time), matching metal kernel (default idir=false).
                            _llk_math_eltwise_unary_sfpu_params_(
                                ckernel::sfpu::calculate_bitonic_topk_merge<APPROX, is_fp32_dest_acc_en, /*top_min=*/false, /*STABLE=*/false>,
                                dst_index,
                                vector_mode,
                                m_iter,
                                SORT_K);
                        }

                        _llk_math_dest_section_done_<dest_sync, is_fp32_dest_acc_en>();
                    }
                }
            }
        }

        // ----- 3. Final result-copy pass: datacopy each tile to DEST[0] -----
        for (std::uint32_t t = 0; t < FULL_CT_DIM; ++t)
        {
            const std::uint32_t stage_idx = (t < Wt) ? 0u : 1u;
            _llk_math_wait_for_dest_available_<dest_sync>();
            _llk_math_reconfig_data_format_srca_<is_fp32_dest_acc_en, false>(math_data_types[stage_idx]);
#ifdef ARCH_BLACKHOLE
            _llk_math_eltwise_unary_datacopy_init_<DataCopyType::A2D, is_fp32_dest_acc_en, BroadcastType::NONE, false, PackMode::Default>(
                /*num_faces=*/4, /*dst_format=*/math_data_types[stage_idx]);
#else
            _llk_math_eltwise_unary_datacopy_init_<DataCopyType::A2D, is_fp32_dest_acc_en, BroadcastType::NONE, false>(
                /*num_faces=*/4, /*dst_format=*/math_data_types[stage_idx]);
#endif
            _llk_math_eltwise_unary_datacopy_<DataCopyType::A2D, DstSync::SyncHalf, is_fp32_dest_acc_en, BroadcastType::NONE, unpack_to_dest>(
                /*dst_index=*/0, math_data_types[stage_idx], math_data_types[stage_idx]);
            _llk_math_dest_section_done_<dest_sync, is_fp32_dest_acc_en>();
        }
    }
}
#endif // LLK_TRISC_MATH

// ============================================================================
// PACK TRISC
// ============================================================================

#ifdef LLK_TRISC_PACK
#include "llk_lib_pack_wrappers.h"
#include "llk_pack.h"
#include "llk_pack_common.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    const std::uint32_t FULL_CT_DIM = params.FULL_CT_DIM;
    const std::uint32_t Wt          = FULL_CT_DIM / NUM_STAGES;
    const std::uint32_t NUM_ROWS    = params.FULL_RT_DIM;
    const std::uint32_t STAGES      = ilog2_ce(Wt);

#ifdef ARCH_BLACKHOLE
    _llk_pack_dest_init_<dest_sync, is_fp32_dest_acc_en>();
#else
    _llk_pack_dest_init_<dest_sync, false, PackMode::Default>();
#endif

    const std::uint32_t pack_src_data_types[NUM_STAGES] = {formats.pack_src, ckernel::to_underlying(DataFormat::UInt16)};
    const std::uint32_t pack_dst_data_types[NUM_STAGES] = {formats.pack_dst, ckernel::to_underlying(DataFormat::UInt16)};

    _llk_pack_hw_configure_<is_fp32_dest_acc_en, PackMode::Default>(pack_src_data_types[0], pack_dst_data_types[0], /*tile_size=*/16 * 16 * 4);
    _llk_pack_init_wrapper_</*pack_mode=*/PackMode::Default, /*zero_output=*/false>(pack_dst_data_types[0]);

    for (std::uint32_t row = 0; row < NUM_ROWS; ++row)
    {
        const std::uint32_t row_offset = row * FULL_CT_DIM;

        // ----- 1. Bitonic-sequence formation -----
        for (std::uint32_t wt = 0; wt < Wt; wt += 2)
        {
            _llk_packer_wait_for_math_done_();

            // Values reconfig + pack.
#ifdef ARCH_BLACKHOLE
            _llk_pack_reconfig_data_format_<is_fp32_dest_acc_en>(
                pack_src_data_types[0],
                pack_dst_data_types[0],
                /*tile_size=*/16 * 16 * 4,
                FACE_R_DIM,
                TILE_C_DIM,
                /*num_faces=*/4,
                /*partial_face=*/false);
#else
            _llk_pack_reconfig_data_format_<is_fp32_dest_acc_en>(
                pack_src_data_types[0],
                pack_dst_data_types[0],
                /*tile_size=*/16 * 16 * 4,
                FACE_R_DIM,
                /*num_faces=*/4,
                /*partial_face=*/false,
                /*narrow_tile=*/false);
#endif
            _llk_pack_init_wrapper_</*pack_mode=*/PackMode::Default, /*zero_output=*/false>(pack_dst_data_types[0]);
            _llk_pack_<dest_sync, is_fp32_dest_acc_en, PackMode::Default>(input_dest_start, L1_ADDRESS(params.buffer_A[row_offset + wt]));
            _llk_pack_<dest_sync, is_fp32_dest_acc_en, PackMode::Default>(input_dest_end, L1_ADDRESS(params.buffer_A[row_offset + wt + 1]));

            // Indices reconfig + pack.
#ifdef ARCH_BLACKHOLE
            _llk_pack_reconfig_data_format_<is_fp32_dest_acc_en>(
                pack_src_data_types[1],
                pack_dst_data_types[1],
                /*tile_size=*/16 * 16 * 4,
                FACE_R_DIM,
                TILE_C_DIM,
                /*num_faces=*/4,
                /*partial_face=*/false);
#else
            _llk_pack_reconfig_data_format_<is_fp32_dest_acc_en>(
                pack_src_data_types[1],
                pack_dst_data_types[1],
                /*tile_size=*/16 * 16 * 4,
                FACE_R_DIM,
                /*num_faces=*/4,
                /*partial_face=*/false,
                /*narrow_tile=*/false);
#endif
            _llk_pack_init_wrapper_</*pack_mode=*/PackMode::Default, /*zero_output=*/false>(pack_dst_data_types[1]);
            _llk_pack_<dest_sync, is_fp32_dest_acc_en, PackMode::Default>(index_dest_start, L1_ADDRESS(params.buffer_A[row_offset + Wt + wt]));
            _llk_pack_<dest_sync, is_fp32_dest_acc_en, PackMode::Default>(index_dest_end, L1_ADDRESS(params.buffer_A[row_offset + Wt + wt + 1]));

            _llk_pack_dest_section_done_<dest_sync, is_fp32_dest_acc_en>();
        }

        // ----- 2. Bitonic merge stages -----
        for (std::uint32_t stage = 2; stage <= STAGES; ++stage)
        {
            for (std::uint32_t sub = stage; sub > 0; --sub)
            {
                const std::uint32_t sub_dist = 1u << (sub - 1);
                for (std::uint32_t i = 0; i < Wt; ++i)
                {
                    const std::uint32_t j = i ^ sub_dist;
                    if (j > i)
                    {
                        const bool ascending_block = ((i >> stage) & 1) == 0;
                        const bool dir             = (ascending_block == ASCENDING);

                        // Direction handling: for sub != 1 (merge with top_min=false),
                        // DEST[0] holds the SMALLER value and DEST[1] holds the LARGER.
                        // - dir == true (ascending): low tile id (i) gets smaller -> DEST[0],
                        //   high tile id (j) gets larger -> DEST[1].  But the metal code shows
                        //   the OPPOSITE swap: when dir, swap so tile_input_low = input_dest_end (1).
                        //   Reading the metal kernel carefully:
                        //     "topk_merge puts smallest values in DEST[0] and largest in DEST[1]"
                        //     "We swap their indices when using descending order"
                        //   But the code is:  if (dir) { swap; }   and `dir` is computed as
                        //   `dir = ascending_block == ascending` where ascending=true means
                        //   the global request is ascending.  So `dir==true` means ASCENDING.
                        //   The comment says "swap when descending" but the code swaps when `dir==true`.
                        //   This is consistent if we read it as: top_min=false means DEST[0]=largest
                        //   and DEST[1]=smallest (top-K of largest values goes to DEST[0]).
                        //   In ascending order, smaller goes left -> we want DEST[1] in left, hence swap.
                        //   We faithfully mirror the metal code's behaviour.
                        int v_low_dst, v_high_dst, x_low_dst, x_high_dst;
                        if (sub != 1 && dir)
                        {
                            v_low_dst  = input_dest_end;
                            v_high_dst = input_dest_start;
                            x_low_dst  = index_dest_end;
                            x_high_dst = index_dest_start;
                        }
                        else
                        {
                            v_low_dst  = input_dest_start;
                            v_high_dst = input_dest_end;
                            x_low_dst  = index_dest_start;
                            x_high_dst = index_dest_end;
                        }

                        _llk_packer_wait_for_math_done_();

                        // Values reconfig + pack.
#ifdef ARCH_BLACKHOLE
                        _llk_pack_reconfig_data_format_<is_fp32_dest_acc_en>(
                            pack_src_data_types[0],
                            pack_dst_data_types[0],
                            /*tile_size=*/16 * 16 * 4,
                            FACE_R_DIM,
                            TILE_C_DIM,
                            /*num_faces=*/4,
                            /*partial_face=*/false);
#else
                        _llk_pack_reconfig_data_format_<is_fp32_dest_acc_en>(
                            pack_src_data_types[0],
                            pack_dst_data_types[0],
                            /*tile_size=*/16 * 16 * 4,
                            FACE_R_DIM,
                            /*num_faces=*/4,
                            /*partial_face=*/false,
                            /*narrow_tile=*/false);
#endif
                        _llk_pack_init_wrapper_</*pack_mode=*/PackMode::Default, /*zero_output=*/false>(pack_dst_data_types[0]);
                        _llk_pack_<dest_sync, is_fp32_dest_acc_en, PackMode::Default>(v_low_dst, L1_ADDRESS(params.buffer_A[row_offset + i]));
                        _llk_pack_<dest_sync, is_fp32_dest_acc_en, PackMode::Default>(v_high_dst, L1_ADDRESS(params.buffer_A[row_offset + j]));

                        // Indices reconfig + pack.
#ifdef ARCH_BLACKHOLE
                        _llk_pack_reconfig_data_format_<is_fp32_dest_acc_en>(
                            pack_src_data_types[1],
                            pack_dst_data_types[1],
                            /*tile_size=*/16 * 16 * 4,
                            FACE_R_DIM,
                            TILE_C_DIM,
                            /*num_faces=*/4,
                            /*partial_face=*/false);
#else
                        _llk_pack_reconfig_data_format_<is_fp32_dest_acc_en>(
                            pack_src_data_types[1],
                            pack_dst_data_types[1],
                            /*tile_size=*/16 * 16 * 4,
                            FACE_R_DIM,
                            /*num_faces=*/4,
                            /*partial_face=*/false,
                            /*narrow_tile=*/false);
#endif
                        _llk_pack_init_wrapper_</*pack_mode=*/PackMode::Default, /*zero_output=*/false>(pack_dst_data_types[1]);
                        _llk_pack_<dest_sync, is_fp32_dest_acc_en, PackMode::Default>(x_low_dst, L1_ADDRESS(params.buffer_A[row_offset + Wt + i]));
                        _llk_pack_<dest_sync, is_fp32_dest_acc_en, PackMode::Default>(x_high_dst, L1_ADDRESS(params.buffer_A[row_offset + Wt + j]));

                        _llk_pack_dest_section_done_<dest_sync, is_fp32_dest_acc_en>();
                    }
                }
            }
        }

        // ----- 3. Final result-copy pass: pack each tile from DEST[0] to buffer_Res -----
        for (std::uint32_t t = 0; t < FULL_CT_DIM; ++t)
        {
            const std::uint32_t stage_idx = (t < Wt) ? 0u : 1u;
            _llk_packer_wait_for_math_done_();
#ifdef ARCH_BLACKHOLE
            _llk_pack_reconfig_data_format_<is_fp32_dest_acc_en>(
                pack_src_data_types[stage_idx],
                pack_dst_data_types[stage_idx],
                /*tile_size=*/16 * 16 * 4,
                FACE_R_DIM,
                TILE_C_DIM,
                /*num_faces=*/4,
                /*partial_face=*/false);
#else
            _llk_pack_reconfig_data_format_<is_fp32_dest_acc_en>(
                pack_src_data_types[stage_idx],
                pack_dst_data_types[stage_idx],
                /*tile_size=*/16 * 16 * 4,
                FACE_R_DIM,
                /*num_faces=*/4,
                /*partial_face=*/false,
                /*narrow_tile=*/false);
#endif
            _llk_pack_init_wrapper_</*pack_mode=*/PackMode::Default, /*zero_output=*/false>(pack_dst_data_types[stage_idx]);
            _llk_pack_<dest_sync, is_fp32_dest_acc_en, PackMode::Default>(/*tile_index=*/0, L1_ADDRESS(params.buffer_Res[row_offset + t]));
            _llk_pack_dest_section_done_<dest_sync, is_fp32_dest_acc_en>();
        }
    }
}
#endif // LLK_TRISC_PACK
