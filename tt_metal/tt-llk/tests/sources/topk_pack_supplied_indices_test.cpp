// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Bit-exact supplied-index packing, including both 512-element score halves.
// Raw UInt32 inputs isolate packing from score conversion and sorting.

#include <cstdint>
#include <initializer_list>

#include "ckernel.h"
#include "counters.h"
#include "llk_defs.h"
#include "params.h"
#include "perf.h"
#include "profiler.h"

// Globals required by the test framework.
std::uint32_t unp_cfg_context          = 0;
std::uint32_t pack_sync_tile_dst_ptr   = 0;
std::uint32_t math_sync_tile_dst_index = 0;

static constexpr std::uint32_t ELEMENTS_PER_TILE = ckernel::TILE_R_DIM * ckernel::TILE_C_DIM;
static constexpr std::uint32_t INPUT_TILES       = (PACK_K + ELEMENTS_PER_TILE - 1) / ELEMENTS_PER_TILE;
static constexpr std::uint32_t SCORES            = 0;
static constexpr std::uint32_t IDS               = 2;
static constexpr std::uint32_t OUT               = PACK_ALIAS == 0 ? 4 : (PACK_ALIAS == 1 ? SCORES : IDS);

static_assert(unpack_to_dest, "supplied-index packing requires UInt32 unpack directly to Dest");

#ifdef LLK_TRISC_UNPACK

#include "llk_unpack_A.h"
#include "llk_unpack_common.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
#ifndef SPEED_OF_LIGHT
    const std::uint32_t LOOP_FACTOR = params.LOOP_FACTOR;
    const Operand& buffer_A         = params.buffer_A;
    const Operand& buffer_B         = params.buffer_B;
#endif
    {
        START_PERF_MEASURE("INIT")
        _llk_unpack_hw_configure_<is_fp32_dest_acc_en>(
            formats.unpack_A_src, formats.unpack_B_src, formats.unpack_A_dst, formats.unpack_B_dst, FACE_R_DIM, FACE_R_DIM, TILE_NUM_FACES, TILE_NUM_FACES);
        _llk_unpack_A_init_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(
            0 /* transpose_of_faces */, 0 /* within_face_16x16_transpose */, ckernel::DEFAULT_TENSOR_SHAPE, formats.unpack_A_src, formats.unpack_A_dst);
        PROFILER_SYNC();
    }
    {
        START_PERF_MEASURE("TILE_LOOP")
        if constexpr (PERF_RUN_TYPE == PerfRunType::PACK_ISOLATE)
        {
        }
        else if constexpr (PERF_RUN_TYPE == PerfRunType::MATH_ISOLATE)
        {
            for (std::uint32_t tile = 0; tile < LOOP_FACTOR * 2 * INPUT_TILES; ++tile)
            {
                (void)mailbox_read(ThreadId::MathThreadId);
                wait_for_dest_available();
                t6_semaphore_post<p_stall::UNPACK0>(semaphore::UNPACK_TO_DEST);
            }
        }
        else
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
            {
                for (const auto buffer : {buffer_A, buffer_B})
                {
                    for (std::uint32_t t = 0; t < INPUT_TILES; ++t)
                    {
                        _llk_unpack_A_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(
                            L1_ADDRESS(buffer[t]), formats.unpack_A_src, formats.unpack_A_dst);
                    }
                }
            }
        }
        PROFILER_SYNC();
    }
}

#endif

#ifdef LLK_TRISC_MATH

#include "ckernel_sfpu.h"
#include "llk_lib_math_wrappers.h"
#include "llk_math_eltwise_binary_sfpu_params.h"
#include "llk_math_eltwise_unary_sfpu.h"
#include "sfpu/experimental/ckernel_sfpu_topk_pack_indices.h"

using namespace ckernel;

static_assert(is_fp32_dest_acc_en, "supplied-index packing addresses 32-bit DEST words");

inline void topk_perf_math_drain_direct(std::uint32_t dst_tile)
{
    math_unpack_to_dest_math_ready();
    math::set_dst_write_addr<DstTileShape::Tile32x32, UnpackDestination::DestReg>(dst_tile);
    math::math_unpack_to_dest_tile_ready();
}

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
#ifndef SPEED_OF_LIGHT
    const std::uint32_t LOOP_FACTOR = params.LOOP_FACTOR;
#endif
    const ckernel::TensorShape tensor_shape = ckernel::DEFAULT_TENSOR_SHAPE;
    {
        START_PERF_MEASURE("INIT")
        _llk_math_hw_configure_<is_fp32_dest_acc_en>(formats.math, formats.math);
        _llk_math_pack_sync_init_<dest_sync, is_fp32_dest_acc_en>();
        _llk_math_eltwise_unary_datacopy_init_wrapper_<
            DataCopyType::A2D,
            is_fp32_dest_acc_en,
            BroadcastType::NONE,
            false /* is_int_fpu_en */,
            PackMode::Default>(tensor_shape, formats.math);
        _llk_math_eltwise_unary_sfpu_init_<SfpuType::unused>();
        ckernel::sfpu::_topk_xl_pack_supplied_indices_init_();
        PROFILER_SYNC();
    }
    {
        START_PERF_MEASURE("TILE_LOOP")
        if constexpr (PERF_RUN_TYPE == PerfRunType::PACK_ISOLATE)
        {
        }
        else if constexpr (PERF_RUN_TYPE == PerfRunType::UNPACK_ISOLATE)
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
            {
                for (const auto base_tile : {SCORES, IDS})
                {
                    for (std::uint32_t t = 0; t < INPUT_TILES; ++t)
                    {
                        topk_perf_math_drain_direct(base_tile + t);
                    }
                }
            }
        }
        else
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
            {
                if constexpr (PERF_RUN_TYPE == PerfRunType::L1_TO_L1)
                {
                    _llk_math_wait_for_dest_available_<dest_sync>();
                }
                for (const auto base_tile : {SCORES, IDS})
                {
                    for (std::uint32_t t = 0; t < INPUT_TILES; ++t)
                    {
                        _llk_math_eltwise_unary_datacopy_<DataCopyType::A2D, dest_sync, is_fp32_dest_acc_en, BroadcastType::NONE, unpack_to_dest>(
                            base_tile + t /* dst_index */, formats.math, formats.math, tensor_shape);
                    }
                }
                _llk_math_eltwise_binary_sfpu_params_(
                    ckernel::sfpu::_topk_xl_pack_supplied_indices_<PACK_K>, SCORES, IDS, OUT, VectorMode::RC_custom, PACK_HALF);
                if constexpr (PERF_RUN_TYPE == PerfRunType::L1_TO_L1)
                {
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

void run_kernel(RUNTIME_PARAMETERS params)
{
#ifndef SPEED_OF_LIGHT
    const std::uint32_t LOOP_FACTOR = params.LOOP_FACTOR;
    const Operand& buffer_Res       = params.buffer_Res;
#endif
    // Preserve the complete packed score/ID word.
    constexpr std::uint32_t pack_format = ckernel::to_underlying(DataFormat::UInt32);
    {
        START_PERF_MEASURE("INIT")
        _llk_pack_hw_configure_wrapper_<is_fp32_dest_acc_en, PackMode::Default>(pack_format, pack_format, FACE_R_DIM * FACE_C_DIM * TILE_NUM_FACES);
        _llk_pack_init_wrapper_<PackMode::Default, false /* zero_output */>(pack_format);
        _llk_pack_dest_init_wrapper_<dest_sync, is_fp32_dest_acc_en, PackMode::Default>();
        PROFILER_SYNC();
    }
    {
        START_PERF_MEASURE("TILE_LOOP")
        if constexpr (PERF_RUN_TYPE == PerfRunType::UNPACK_ISOLATE || PERF_RUN_TYPE == PerfRunType::MATH_ISOLATE)
        {
        }
        else
        {
            constexpr bool sync_with_math = PERF_RUN_TYPE == PerfRunType::L1_TO_L1;
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
            {
                if constexpr (sync_with_math)
                {
                    _llk_packer_wait_for_math_done_();
                }
                for (std::uint32_t t = 0; t < INPUT_TILES; ++t)
                {
                    _llk_pack_<dest_sync, is_fp32_dest_acc_en, ckernel::PackMode::Default>(OUT + t /* tile_index */, L1_ADDRESS(buffer_Res[t]));
                }
                if constexpr (sync_with_math)
                {
                    _llk_pack_dest_section_done_<dest_sync, is_fp32_dest_acc_en>();
                }
            }
        }
        PROFILER_SYNC();
    }
}

#endif
