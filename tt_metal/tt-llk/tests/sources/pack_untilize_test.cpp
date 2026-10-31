// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cstdint>
#include <cstdio>

#include "ckernel.h"
#include "counters.h"
#include "llk_assert.h"
#include "llk_defs.h"
#include "params.h"
#include "perf.h"
#include "profiler.h"

// Globals
std::uint32_t unp_cfg_context          = 0;
std::uint32_t pack_sync_tile_dst_ptr   = 0;
std::uint32_t math_sync_tile_dst_index = 0;

// Only modes supported are L1_TO_L1, PACK_ISOLATE and L1_CONGESTION.
static_assert(PERF_RUN_TYPE != PerfRunType::MATH_ISOLATE, "Math isolation not supported for pack_untilize");
static_assert(PERF_RUN_TYPE != PerfRunType::UNPACK_ISOLATE, "Unpack isolation not supported for pack_untilize");

static constexpr std::uint32_t MAX_TILES_DEST = is_fp32_dest_acc_en ? 4 : 8;
static_assert(BLOCK_CT_DIM <= MAX_TILES_DEST, "Block must fit in Dest register");
static_assert(FULL_CT_DIM % BLOCK_CT_DIM == 0, "FULL_CT_DIM must be divisible by BLOCK_CT_DIM");

constexpr static std::uint32_t format_size_in_bytes(std::uint32_t data_format)
{
    switch (static_cast<DataFormat>(data_format & 0xF))
    {
        case DataFormat::Int32:
        case DataFormat::Float32:
            return 4;

        case DataFormat::Float16:
        case DataFormat::Float16_b:
        case DataFormat::UInt16:
            return 2;

        default:
            return 1;
    }
}

#ifdef LLK_TRISC_UNPACK

#include "llk_unpack_A.h"
#include "llk_unpack_common.h"
#include "params.h"

template <typename T>
std::uint32_t unpack_address(const T& params, std::uint32_t tile)
{
    if constexpr (PACK_UNTILIZE_IS_PERF)
    {
        return PERF_ADDRESS(PERF_INPUT_A, tile);
    }
    else
    {
        return L1_ADDRESS(params.buffer_A[tile]);
    }
}

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
#ifndef SPEED_OF_LIGHT
    const std::uint32_t LOOP_FACTOR = params.LOOP_FACTOR;
    const std::uint32_t TILE_CNT    = params.TILE_CNT;
    const std::uint32_t num_faces   = params.num_faces;
#else
    constexpr std::uint32_t num_faces = ckernel::TILE_NUM_FACES;
#endif
    {
        START_PERF_MEASURE("INIT")
        _llk_unpack_hw_configure_<is_fp32_dest_acc_en>(
            formats.unpack_A_src, formats.unpack_B_src, formats.unpack_A_dst, formats.unpack_B_dst, FACE_R_DIM, FACE_R_DIM, num_faces, num_faces);
        _llk_unpack_A_init_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(
            0 /* transpose_of_faces */,
            0 /* within_face_16x16_transpose */,
            ckernel::make_tensor_shape_from_legacy(FACE_R_DIM, num_faces),
            formats.unpack_A_src,
            formats.unpack_A_dst);
        PROFILER_SYNC();
    }

    {
        START_PERF_MEASURE("TILE_LOOP")
        if constexpr (PERF_RUN_TYPE != PerfRunType::PACK_ISOLATE)
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
            {
                for (std::uint32_t tile = 0; tile < TILE_CNT; ++tile)
                {
                    const std::uint32_t input_addr = unpack_address(params, tile);
                    _llk_unpack_A_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(
                        input_addr, formats.unpack_A_src, formats.unpack_A_dst);
                }
            }
        }
        PROFILER_SYNC();
    }
}
#endif

#ifdef LLK_TRISC_MATH

#include "llk_lib_math_wrappers.h"
#include "params.h"

using namespace ckernel;

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
#ifndef SPEED_OF_LIGHT
    const std::uint32_t LOOP_FACTOR = params.LOOP_FACTOR;
    const std::uint32_t TILE_CNT    = params.TILE_CNT;
    const std::uint32_t num_faces   = params.num_faces;
#else
    constexpr std::uint32_t num_faces = ckernel::TILE_NUM_FACES;
#endif
    constexpr bool is_int_fpu_en = false;

    {
        START_PERF_MEASURE("INIT")
        _llk_math_eltwise_unary_datacopy_init_wrapper_<DataCopyType::A2D, is_fp32_dest_acc_en, BroadcastType::NONE, is_int_fpu_en, PackMode::Default>(
            num_faces, formats.math);
        _llk_math_pack_sync_init_<dest_sync, is_fp32_dest_acc_en>();
        _llk_math_hw_configure_<is_fp32_dest_acc_en>(formats.math, formats.math);
        _llk_math_reconfig_remap_wrapper_(true);
        PROFILER_SYNC();
    }

    {
        START_PERF_MEASURE("TILE_LOOP")
        if constexpr (PERF_RUN_TYPE == PerfRunType::PACK_ISOLATE)
        {
        }
        else if constexpr (PERF_RUN_TYPE == PerfRunType::L1_CONGESTION)
        {
            if constexpr (!unpack_to_dest)
            {
                _perf_math_loop_clear_valid<true, true>(LOOP_FACTOR * TILE_CNT * TILE_NUM_FACES);
            }
            else
            {
                for (std::uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
                {
                    for (std::uint32_t block = 0; block < TILE_CNT / BLOCK_CT_DIM; ++block)
                    {
                        for (std::uint32_t block_tile = 0; block_tile < BLOCK_CT_DIM; ++block_tile)
                        {
                            LLK_ASSERT(
                                (block_tile < get_dest_max_tiles<dest_sync, is_fp32_dest_acc_en, DstTileShape::Tile32x32>()),
                                "Block tile index exceeds maximum destination tiles");
                            _llk_math_eltwise_unary_datacopy_<DataCopyType::A2D, dest_sync, is_fp32_dest_acc_en, BroadcastType::NONE, unpack_to_dest>(
                                block_tile, formats.math, formats.math);
                        }
                    }
                }
            }
        }
        else
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
            {
                for (std::uint32_t block = 0; block < TILE_CNT / BLOCK_CT_DIM; ++block)
                {
                    _llk_math_wait_for_dest_available_<dest_sync>();
                    for (std::uint32_t block_tile = 0; block_tile < BLOCK_CT_DIM; ++block_tile)
                    {
                        LLK_ASSERT(
                            (block_tile + TILE_DST_CT_OFFSET < get_dest_max_tiles<dest_sync, is_fp32_dest_acc_en, DstTileShape::Tile32x32>()),
                            "Block tile index + TILE_DST_CT_OFFSET exceeds maximum destination tiles");
                        _llk_math_eltwise_unary_datacopy_<DataCopyType::A2D, dest_sync, is_fp32_dest_acc_en, BroadcastType::NONE, unpack_to_dest>(
                            block_tile + TILE_DST_CT_OFFSET, formats.math, formats.math);
                    }
                    _llk_math_dest_section_done_<dest_sync, is_fp32_dest_acc_en>();
                }
            }
        }
        PROFILER_SYNC();
    }
}

#endif

#ifdef LLK_TRISC_PACK

#include "llk_lib_pack_wrappers.h"
#include "llk_pack_common.h"
#include "llk_pack_untilize.h"
#include "params.h"

constexpr std::uint32_t L1_ACCESS_ADDRESS_GRANULARITY = 16; // in bytes

template <typename T>
std::uint32_t functional_pack_address(const T& params, std::uint32_t rt, std::uint32_t block_num, std::uint32_t num_faces, std::uint32_t output_format)
{
    const std::uint32_t num_datums_in_tile = FACE_R_DIM * FACE_C_DIM * num_faces;
    const std::uint32_t row_stride_16B     = (FULL_CT_DIM * num_datums_in_tile * format_size_in_bytes(output_format)) / L1_ACCESS_ADDRESS_GRANULARITY;
    const std::uint32_t block_stride_16B =
        (BLOCK_CT_DIM * ((num_faces > 2) ? num_faces / 2 : num_faces) * FACE_C_DIM * format_size_in_bytes(output_format)) / L1_ACCESS_ADDRESS_GRANULARITY;
    return L1_ADDRESS(params.buffer_Res[0]) + rt * row_stride_16B + block_num * block_stride_16B;
}

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
#ifndef SPEED_OF_LIGHT
    const std::uint32_t LOOP_FACTOR = params.LOOP_FACTOR;
    const std::uint32_t TILE_CNT    = params.TILE_CNT;
    const std::uint32_t num_faces   = params.num_faces;
#else
    constexpr std::uint32_t num_faces = ckernel::TILE_NUM_FACES;
#endif
    constexpr bool UNTILIZE                = true;
    const std::uint32_t NUM_DATUMS_IN_TILE = FACE_R_DIM * FACE_C_DIM * num_faces;

    {
        START_PERF_MEASURE("INIT")
        _llk_pack_hw_configure_wrapper_<is_fp32_dest_acc_en, llk_test_pack_mode_v<UNTILIZE, false>>(
            formats.pack_src, formats.pack_dst, NUM_DATUMS_IN_TILE /* tile_size */);
        _llk_pack_dest_init_wrapper_<dest_sync, is_fp32_dest_acc_en, llk_test_pack_mode_v<UNTILIZE, false>>();
        _llk_pack_untilize_init_wrapper_<BLOCK_CT_DIM, FULL_CT_DIM>(formats.pack_src, formats.pack_dst, FACE_R_DIM, num_faces);
        PROFILER_SYNC();
    }

    {
        START_PERF_MEASURE("TILE_LOOP")
        if constexpr (PERF_RUN_TYPE == PerfRunType::PACK_ISOLATE || PERF_RUN_TYPE == PerfRunType::L1_CONGESTION)
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
            {
                for (std::uint32_t tile = 0; tile < TILE_CNT; tile += BLOCK_CT_DIM)
                {
                    _llk_pack_untilize_wrapper_<BLOCK_CT_DIM, FULL_CT_DIM, false /* diagonal */, false /* narrow_row */, TILE_C_DIM, TILE_DST_CT_OFFSET>(
                        PERF_ADDRESS(PERF_OUTPUT, tile), formats.pack_dst, FACE_R_DIM, num_faces, 0 /* tile_dst_rt_offset */);
                }
            }
        }
        else if constexpr (PACK_UNTILIZE_IS_PERF)
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
            {
                for (std::uint32_t tile = 0; tile < TILE_CNT; tile += BLOCK_CT_DIM)
                {
                    _llk_packer_wait_for_math_done_();
                    _llk_pack_untilize_wrapper_<BLOCK_CT_DIM, FULL_CT_DIM, false /* diagonal */, false /* narrow_row */, TILE_C_DIM, TILE_DST_CT_OFFSET>(
                        PERF_ADDRESS(PERF_OUTPUT, tile), formats.pack_dst, FACE_R_DIM, num_faces, 0 /* tile_dst_rt_offset */);
                    _llk_pack_dest_section_done_<dest_sync, is_fp32_dest_acc_en>();
                }
            }
        }
        else
        {
            const std::uint32_t num_blocks_per_col = FULL_CT_DIM / BLOCK_CT_DIM;
            for (std::uint32_t rt = 0; rt < FULL_RT_DIM; ++rt)
            {
                for (std::uint32_t block_num = 0; block_num < num_blocks_per_col; ++block_num)
                {
                    const std::uint32_t pack_addr_16B = functional_pack_address(params, rt, block_num, num_faces, formats.pack_dst);
                    _llk_packer_wait_for_math_done_();
                    _llk_pack_untilize_wrapper_<BLOCK_CT_DIM, FULL_CT_DIM, false /* diagonal */, false /* narrow_row */, TILE_C_DIM, TILE_DST_CT_OFFSET>(
                        pack_addr_16B, formats.pack_dst, FACE_R_DIM, num_faces, 0 /* tile_dst_rt_offset */);
                    _llk_pack_dest_section_done_<dest_sync, is_fp32_dest_acc_en>();
                }
            }
        }
        PROFILER_SYNC();
    }

    _llk_pack_untilize_uninit_wrapper_(formats.pack_src, FACE_R_DIM);
}

#endif
