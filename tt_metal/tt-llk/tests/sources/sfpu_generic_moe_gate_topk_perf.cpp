// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// Perf kernel of the generic, SFPU-only DeepSeek MoE gate top-k: per iteration one token, two datacopies, the gate
// (PERF_STAGE 1) or nothing more (PERF_STAGE 0), then the pack of scores and indices. MATH_ISOLATE keeps the real unpack.

#include <cstdint>

#include "ckernel.h"
#include "ckernel_defs.h"
#include "counters.h"
#include "llk_defs.h"
#include "params.h"
#include "perf.h"
#include "profiler.h"

std::uint32_t unp_cfg_context          = 0;
std::uint32_t pack_sync_tile_dst_ptr   = 0;
std::uint32_t math_sync_tile_dst_index = 0;

static constexpr ckernel::DstSync DST_SYNC               = ckernel::DstSync::SyncHalf;
static constexpr std::uint32_t MOE_GATE_SCORES_DST_TILE  = 0;
static constexpr std::uint32_t MOE_GATE_INDICES_DST_TILE = 1;
static constexpr std::uint32_t MOE_GATE_BIAS_DST_TILE    = 2;
static constexpr std::uint32_t MOE_GATE_NUM_INPUT_TILES  = 2;

#ifdef LLK_TRISC_UNPACK

#include "llk_unpack_A.h"
#include "llk_unpack_common.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    const std::uint32_t LOOP_FACTOR = params.LOOP_FACTOR;
    {
        START_PERF_MEASURE("INIT")
        _llk_unpack_hw_configure_<is_fp32_dest_acc_en>(
            formats.unpack_A_src, formats.unpack_B_src, formats.unpack_A_dst, formats.unpack_B_dst, FACE_R_DIM, FACE_R_DIM, TILE_NUM_FACES, TILE_NUM_FACES);
        _llk_unpack_A_init_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(
            0, 0, ckernel::DEFAULT_TENSOR_SHAPE, formats.unpack_A_src, formats.unpack_A_dst);
        PROFILER_SYNC();
    }
    {
        START_PERF_MEASURE("TILE_LOOP")
        if constexpr (PERF_RUN_TYPE != PerfRunType::PACK_ISOLATE)
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
            {
                for (std::uint32_t tile = 0; tile < MOE_GATE_NUM_INPUT_TILES; ++tile)
                {
                    _llk_unpack_A_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(
                        L1_ADDRESS(params.buffer_A[tile]), formats.unpack_A_src, formats.unpack_A_dst);
                }
            }
        }
        PROFILER_SYNC();
    }
}

#endif

#ifdef LLK_TRISC_MATH

#include "ckernel_sfpu.h"
#include "generic_moe_gate_test_helpers.h"
#include "llk_lib_math_wrappers.h"
#include "llk_math_eltwise_unary_sfpu.h"
#include "llk_math_eltwise_unary_sfpu_params.h"
#include "llk_sfpu/llk_math_eltwise_unary_sfpu_macros.h"
#include "sfpu/experimental/ckernel_sfpu_generic_moe_gate_topk.h"

using namespace ckernel;

inline void token_math(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    (void)params;
    _llk_math_eltwise_unary_datacopy_<DataCopyType::A2D, DST_SYNC, is_fp32_dest_acc_en, BroadcastType::NONE, unpack_to_dest>(
        MOE_GATE_SCORES_DST_TILE, formats.math, formats.math);
    _llk_math_eltwise_unary_datacopy_<DataCopyType::A2D, DST_SYNC, is_fp32_dest_acc_en, BroadcastType::NONE, unpack_to_dest>(
        MOE_GATE_BIAS_DST_TILE, formats.math, formats.math);
    if constexpr (PERF_STAGE >= 1)
    {
        if constexpr (!MOE_GATE_GENERATE_INDICES)
        {
            SFPU_UNARY_CALL(DST_SYNC, is_fp32_dest_acc_en, _moe_gate_test_seed_indices_, (0), MOE_GATE_SCORES_DST_TILE, VectorMode::RC_custom);
        }
        SFPU_UNARY_CALL(
            DST_SYNC,
            is_fp32_dest_acc_en,
            _generic_moe_gate_topk_,
            (MOE_GATE_NORMALIZE,
             MOE_GATE_NUM_SELECTED_EXPERTS,
             MOE_GATE_NUM_TOTAL_EXPERTS,
             MOE_GATE_ZERO_TAIL,
             MOE_GATE_FULL_SORT,
             MOE_GATE_GENERATE_INDICES,
             false,
             MOE_GATE_SCORES_INCLUDE_BIAS),
            MOE_GATE_SCORES_DST_TILE,
            VectorMode::RC_custom,
            MOE_GATE_EPS_BITS,
            MOE_GATE_SCALE_BITS,
            MOE_GATE_EXTRA_SCALE_BITS);
    }
}

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    const std::uint32_t LOOP_FACTOR = params.LOOP_FACTOR;
    {
        START_PERF_MEASURE("INIT")
        _llk_math_hw_configure_<is_fp32_dest_acc_en>(formats.math, formats.math);
        _llk_math_pack_sync_init_<DST_SYNC, is_fp32_dest_acc_en>();
        _llk_math_eltwise_unary_datacopy_init_wrapper_<DataCopyType::A2D, is_fp32_dest_acc_en, BroadcastType::NONE, false, PackMode::Default>(
            TILE_NUM_FACES, formats.math);
        _llk_math_eltwise_unary_sfpu_init_<SfpuType::unused>();
        ckernel::sfpu::_init_generic_moe_gate_topk_();
        PROFILER_SYNC();
    }
    {
        START_PERF_MEASURE("TILE_LOOP")
        if constexpr (PERF_RUN_TYPE == PerfRunType::PACK_ISOLATE)
        {
        }
        else if constexpr (PERF_RUN_TYPE == PerfRunType::MATH_ISOLATE)
        {
            _llk_math_wait_for_dest_available_<DST_SYNC>();
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
            {
                token_math(params);
            }
        }
        else
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
            {
                _llk_math_wait_for_dest_available_<DST_SYNC>();
                token_math(params);
                _llk_math_dest_section_done_<DST_SYNC, is_fp32_dest_acc_en>();
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
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    const std::uint32_t LOOP_FACTOR   = params.LOOP_FACTOR;
    constexpr std::uint32_t TILE_SIZE = FACE_R_DIM * FACE_C_DIM * TILE_NUM_FACES;
    const std::uint32_t index_format  = ckernel::to_underlying(DataFormat::UInt16);
    {
        START_PERF_MEASURE("INIT")
        _llk_pack_hw_configure_wrapper_<is_fp32_dest_acc_en, PackMode::Default>(formats.pack_src, formats.pack_dst, TILE_SIZE);
        _llk_pack_init_wrapper_<PackMode::Default, false>(formats.pack_dst, FACE_R_DIM, TILE_C_DIM, TILE_NUM_FACES);
        _llk_pack_dest_init_wrapper_<DST_SYNC, is_fp32_dest_acc_en, PackMode::Default>();
        PROFILER_SYNC();
    }
    {
        START_PERF_MEASURE("TILE_LOOP")
        if constexpr (PERF_RUN_TYPE == PerfRunType::L1_TO_L1 || PERF_RUN_TYPE == PerfRunType::L1_CONGESTION)
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
            {
                _llk_packer_wait_for_math_done_();
                _llk_pack_reconfig_data_format_wrapper_<is_fp32_dest_acc_en, false>(
                    formats.pack_src, formats.pack_dst, TILE_SIZE, FACE_R_DIM, TILE_C_DIM, TILE_NUM_FACES, false, false, 1);
                _llk_pack_init_wrapper_<PackMode::Default, false>(formats.pack_dst);
                _llk_pack_<DST_SYNC, is_fp32_dest_acc_en, ckernel::PackMode::Default>(MOE_GATE_SCORES_DST_TILE, L1_ADDRESS(params.buffer_Res[0]));
                _llk_pack_reconfig_data_format_wrapper_<is_fp32_dest_acc_en, false>(
                    index_format, index_format, TILE_SIZE, FACE_R_DIM, TILE_C_DIM, TILE_NUM_FACES, false, false, 1);
                _llk_pack_init_wrapper_<PackMode::Default, false>(index_format);
                _llk_pack_<DST_SYNC, is_fp32_dest_acc_en, ckernel::PackMode::Default>(MOE_GATE_INDICES_DST_TILE, L1_ADDRESS(params.buffer_Res[1]));
                _llk_pack_dest_section_done_<DST_SYNC, is_fp32_dest_acc_en>();
            }
        }
        PROFILER_SYNC();
    }
}

#endif
