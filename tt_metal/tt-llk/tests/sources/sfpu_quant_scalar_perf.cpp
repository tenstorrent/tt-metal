// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// LLK SFPU quantization perf kernel: quant, requant and dequant (QUANT_OP 0, 1, 2) with a per-tensor scale as a DEST
// tile (QUANT_SCALE_FORM 0, two copied tiles) or loaded once by the init (QUANT_SCALE_FORM 1, one copied tile);
// QUANT_ZP_BITS and QUANT_SCALE_BITS are the zero point and the scale as fp32 bits.

#include <algorithm>
#include <cstdint>

#include "ckernel.h"
#include "ckernel_defs.h"
#include "counters.h"
#include "llk_defs.h"
#include "params.h"
#include "perf.h"
#include "profiler.h"

// Globals
std::uint32_t unp_cfg_context                   = 0;
std::uint32_t pack_sync_tile_dst_ptr            = 0;
std::uint32_t math_sync_tile_dst_index          = 0;
static constexpr ckernel::DstSync DST_SYNC_MODE = ckernel::DstSync::SyncHalf;

#ifndef QUANT_OP
#define QUANT_OP 0
#endif
#ifndef QUANT_SCALE_FORM
#define QUANT_SCALE_FORM 0
#endif
#ifndef QUANT_ZP_BITS
#define QUANT_ZP_BITS 0x40400000u // 3.0f
#endif
#ifndef QUANT_SCALE_BITS
#define QUANT_SCALE_BITS 0x3F000000u // 0.5f
#endif

static constexpr std::uint32_t QUANT_COPIES      = (QUANT_SCALE_FORM == 0) ? 2 : 1;
static constexpr std::uint32_t DATA_TILE         = 0;
static constexpr std::uint32_t SCALE_TILE        = 1;
static constexpr std::uint32_t RESULT_TILE       = 0;
static constexpr std::uint32_t QUANT_NEG_ZP_BITS = QUANT_ZP_BITS ^ 0x80000000u; // dequant takes the negated zero point

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
    const std::uint32_t num_faces   = params.num_faces;
    const std::uint32_t TILE_CNT    = params.TILE_CNT;
#endif
    {
        START_PERF_MEASURE("INIT")
        _llk_unpack_hw_configure_<is_fp32_dest_acc_en>(
            formats.unpack_A_src, formats.unpack_B_src, formats.unpack_A_dst, formats.unpack_B_dst, FACE_R_DIM, FACE_R_DIM, num_faces, num_faces);
        _llk_unpack_A_init_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(
            0, 0, ckernel::make_tensor_shape_from_legacy(FACE_R_DIM, num_faces), formats.unpack_A_src, formats.unpack_A_dst);
        PROFILER_SYNC();
    }
    {
        START_PERF_MEASURE("TILE_LOOP")
        if constexpr (PERF_RUN_TYPE == PerfRunType::MATH_ISOLATE)
        {
            if constexpr (!unpack_to_dest)
            {
                _perf_unpack_loop_set_valid<true, is_fp32_dest_acc_en>(num_faces * TILE_CNT * LOOP_FACTOR * QUANT_COPIES);
            }
        }
        else if constexpr (PERF_RUN_TYPE != PerfRunType::PACK_ISOLATE)
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
            {
                for (std::uint32_t i = 0; i < TILE_CNT; ++i)
                {
                    for (std::uint32_t c = 0; c < QUANT_COPIES; ++c)
                    {
                        // the input tile and, with the tile scale, the same tile again as the scale operand
                        _llk_unpack_A_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(
                            PERF_ADDRESS(PERF_INPUT_A, i), formats.unpack_A_src, formats.unpack_A_dst);
                    }
                }
            }
        }
        PROFILER_SYNC();
    }
}

#endif // LLK_TRISC_UNPACK

#ifdef LLK_TRISC_MATH

#include "ckernel_sfpu.h"
#include "llk_lib_math_wrappers.h"
#include "llk_math_eltwise_binary_sfpu.h"
#include "llk_math_eltwise_binary_sfpu_params.h"
#include "llk_sfpu/ckernel_sfpu_quant.h"

using namespace ckernel;

#define QUANT_DATACOPY(T) \
    _llk_math_eltwise_unary_datacopy_<DataCopyType::A2D, DST_SYNC_MODE, is_fp32_dest_acc_en, BroadcastType::NONE, unpack_to_dest>(T, formats.math, formats.math)

inline void quant_op_init()
{
    if constexpr (QUANT_OP == 0)
    {
        _llk_math_eltwise_binary_sfpu_init_<SfpuType::quant_int32>();
        if constexpr (QUANT_SCALE_FORM == 0)
        {
            sfpu::quant_init<false, false, DataFormat::Int32>(QUANT_ZP_BITS);
        }
        else
        {
            sfpu::quant_init_scalar_scale<false, false, DataFormat::Int32>(QUANT_ZP_BITS, QUANT_SCALE_BITS);
        }
    }
    else if constexpr (QUANT_OP == 1)
    {
        _llk_math_eltwise_binary_sfpu_init_<SfpuType::requant_int32>();
        if constexpr (QUANT_SCALE_FORM == 0)
        {
            sfpu::requant_init<false, false, DataFormat::Int32, false>(QUANT_ZP_BITS);
        }
        else
        {
            sfpu::requant_init_scalar_scale<false, false, DataFormat::Int32, false>(QUANT_ZP_BITS, QUANT_SCALE_BITS);
        }
    }
    else
    {
        _llk_math_eltwise_binary_sfpu_init_<SfpuType::dequant_int32>();
        if constexpr (QUANT_SCALE_FORM == 0)
        {
            sfpu::dequant_init<false, false, false>(QUANT_NEG_ZP_BITS);
        }
        else
        {
            sfpu::dequant_init_scalar_scale<false, false, false>(QUANT_NEG_ZP_BITS, QUANT_SCALE_BITS);
        }
    }
}

inline void quant_op_tile()
{
    constexpr bool SCALAR = (QUANT_SCALE_FORM == 1);
    if constexpr (QUANT_OP == 0)
    {
        _llk_math_eltwise_binary_sfpu_params_(sfpu::calculate_quant_int32<false, 8, false, SCALAR>, DATA_TILE, SCALE_TILE, RESULT_TILE, VectorMode::RC);
    }
    else if constexpr (QUANT_OP == 1)
    {
        _llk_math_eltwise_binary_sfpu_params_(
            sfpu::calculate_requant_int32<false, 8, false, false, SCALAR>, DATA_TILE, SCALE_TILE, RESULT_TILE, VectorMode::RC);
    }
    else
    {
        _llk_math_eltwise_binary_sfpu_params_(
            sfpu::calculate_dequant_int32<false, 8, false, false, SCALAR>, DATA_TILE, SCALE_TILE, RESULT_TILE, VectorMode::RC);
    }
}

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
#ifndef SPEED_OF_LIGHT
    const std::uint32_t LOOP_FACTOR = params.LOOP_FACTOR;
    const std::uint32_t num_faces   = params.num_faces;
    const std::uint32_t TILE_CNT    = params.TILE_CNT;
#endif
    {
        START_PERF_MEASURE("INIT")
        _llk_math_eltwise_unary_datacopy_init_wrapper_<DataCopyType::A2D, is_fp32_dest_acc_en, BroadcastType::NONE, false, PackMode::Default>(
            num_faces, formats.math);
        _llk_math_hw_configure_<is_fp32_dest_acc_en>(formats.math, formats.math);
        _llk_math_pack_sync_init_<DST_SYNC_MODE, is_fp32_dest_acc_en>();
        quant_op_init();
        PROFILER_SYNC();
    }
    {
        START_PERF_MEASURE("TILE_LOOP")
        if constexpr (PERF_RUN_TYPE == PerfRunType::PACK_ISOLATE)
        {
        }
        else if constexpr (PERF_RUN_TYPE == PerfRunType::UNPACK_ISOLATE || PERF_RUN_TYPE == PerfRunType::L1_CONGESTION)
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
            {
                for (std::uint32_t tile = 0; tile < TILE_CNT; ++tile)
                {
                    for (std::uint32_t c = 0; c < QUANT_COPIES; ++c)
                    {
                        if constexpr (unpack_to_dest)
                        {
                            QUANT_DATACOPY(c);
                        }
                        else
                        {
                            _perf_math_loop_clear_valid<true, true>(num_faces);
                        }
                    }
                }
            }
        }
        else if constexpr (PERF_RUN_TYPE == PerfRunType::MATH_ISOLATE)
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
            {
                for (std::uint32_t tile = 0; tile < TILE_CNT; ++tile)
                {
                    if constexpr (!unpack_to_dest)
                    {
                        for (std::uint32_t c = 0; c < QUANT_COPIES; ++c)
                        {
                            QUANT_DATACOPY(c);
                        }
                    }
                    quant_op_tile();
                }
            }
        }
        else // L1_TO_L1
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
            {
                for (std::uint32_t tile = 0; tile < TILE_CNT; ++tile)
                {
                    _llk_math_wait_for_dest_available_<DST_SYNC_MODE>();
                    for (std::uint32_t c = 0; c < QUANT_COPIES; ++c)
                    {
                        QUANT_DATACOPY(c);
                    }
                    quant_op_tile();
                    _llk_math_dest_section_done_<DST_SYNC_MODE, is_fp32_dest_acc_en>();
                }
            }
        }
        PROFILER_SYNC();
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
#ifndef SPEED_OF_LIGHT
    const std::uint32_t LOOP_FACTOR = params.LOOP_FACTOR;
    const std::uint32_t num_faces   = params.num_faces;
    const std::uint32_t TILE_CNT    = params.TILE_CNT;
#endif
    {
        START_PERF_MEASURE("INIT")
        _llk_pack_hw_configure_wrapper_<is_fp32_dest_acc_en, PackMode::Default>(formats.pack_src, formats.pack_dst, FACE_R_DIM * FACE_C_DIM * num_faces);
        _llk_pack_init_wrapper_<PackMode::Default, false>(formats.pack_dst, FACE_R_DIM, TILE_C_DIM, num_faces);
        _llk_pack_dest_init_<DST_SYNC_MODE, is_fp32_dest_acc_en>();
        PROFILER_SYNC();
    }
    {
        START_PERF_MEASURE("TILE_LOOP")
        if constexpr (PERF_RUN_TYPE == PerfRunType::PACK_ISOLATE || PERF_RUN_TYPE == PerfRunType::L1_CONGESTION)
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
            {
                for (std::uint32_t tile = 0; tile < TILE_CNT; ++tile)
                {
                    _llk_pack_<DST_SYNC_MODE, is_fp32_dest_acc_en, ckernel::PackMode::Default>(RESULT_TILE, PERF_ADDRESS(PERF_OUTPUT, tile));
                }
            }
        }
        else if constexpr (PERF_RUN_TYPE == PerfRunType::L1_TO_L1)
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
            {
                for (std::uint32_t tile = 0; tile < TILE_CNT; ++tile)
                {
                    _llk_packer_wait_for_math_done_();
                    _llk_pack_<DST_SYNC_MODE, is_fp32_dest_acc_en, ckernel::PackMode::Default>(RESULT_TILE, PERF_ADDRESS(PERF_OUTPUT, tile));
                    _llk_pack_dest_section_done_<DST_SYNC_MODE, is_fp32_dest_acc_en>();
                }
            }
        }
        PROFILER_SYNC();
    }
}

#endif // LLK_TRISC_PACK
