// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// Perf twin of top32_rm_test.cpp: the top32_rm chunk walk (64-element chunks of one row-major row and its index row, each
// copied into Dest, sorted and merged into the running top 32), the sequence of the DeepSeek sampling kernel, repeated
// LOOP_FACTOR times inside TILE_LOOP and reported per 64-element chunk. L1_TO_L1 only. Blackhole-only.

#include <cstdint>

#include "ckernel.h"
#include "counters.h"
#include "llk_defs.h"
#include "params.h"
#include "perf.h"
#include "profiler.h"

std::uint32_t unp_cfg_context          = 0;
std::uint32_t pack_sync_tile_dst_ptr   = 0;
std::uint32_t math_sync_tile_dst_index = 0;

static_assert(PERF_RUN_TYPE == PerfRunType::L1_TO_L1, "top32_rm_perf runs L1_TO_L1 only");

static constexpr ckernel::DstSync DST_SYNC = ckernel::DstSync::SyncHalf;

static constexpr std::uint32_t VALUE_TILE       = 0;
static constexpr std::uint32_t INDEX_TILE       = 2;
static constexpr std::uint32_t STAGE_VALUE_TILE = 1;
static constexpr std::uint32_t STAGE_INDEX_TILE = 3;

static constexpr std::uint32_t ELEMENTS_PER_CHUNK = 4 * 16;
static constexpr std::uint32_t CHUNK_ADDR_STRIDE  = (ELEMENTS_PER_CHUNK * TOP32_DATUM_BYTES) / 16; // unpacker addresses count 16-byte words

static_assert(TOP32_ROW_ELEMENTS % ELEMENTS_PER_CHUNK == 0, "the perf rows walk whole 64-element chunks");

#ifdef LLK_TRISC_UNPACK

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-variable"
#pragma GCC diagnostic ignored "-Wunused-parameter"
#include "experimental/llk_unpack_A_top32_rm.h"
#pragma GCC diagnostic pop
#include "llk_unpack_A.h"
#include "llk_unpack_common.h"

inline void unpack_chunk(std::uint32_t base_address, std::uint32_t chunk, std::uint32_t src_format, std::uint32_t dst_format)
{
    _llk_unpack_A_top32_rm_init_<unpack_to_dest>(unpack_to_dest ? 0 : 1, src_format, dst_format);
    _llk_unpack_A_top32_rm_<unpack_to_dest>(4, base_address + chunk * CHUNK_ADDR_STRIDE, src_format, dst_format);
    if constexpr (unpack_to_dest)
    {
        _llk_unpack_set_srcb_dummy_valid_();
    }
}

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    const std::uint32_t LOOP_FACTOR  = params.LOOP_FACTOR;
    const std::uint32_t src_format   = formats.unpack_A_src;
    const std::uint32_t dst_format   = formats.unpack_A_dst;
    const std::uint32_t values_base  = L1_ADDRESS(params.buffer_A[0]);
    const std::uint32_t indices_base = L1_ADDRESS(params.buffer_B[0]);
    {
        START_PERF_MEASURE("INIT")
        _llk_unpack_hw_configure_<is_fp32_dest_acc_en>(
            formats.unpack_A_src, formats.unpack_B_src, formats.unpack_A_dst, formats.unpack_B_dst, FACE_R_DIM, FACE_R_DIM, TILE_NUM_FACES, TILE_NUM_FACES);
        PROFILER_SYNC();
    }
    {
        START_PERF_MEASURE("TILE_LOOP")
        for (std::uint32_t loop = 0; loop < LOOP_FACTOR; loop++)
        {
            for (std::uint32_t chunk = 0; chunk < TOP32_ROW_ELEMENTS / ELEMENTS_PER_CHUNK; chunk++)
            {
                unpack_chunk(values_base, chunk, src_format, dst_format);
                unpack_chunk(indices_base, chunk, src_format, dst_format);
            }
        }
        PROFILER_SYNC();
    }
}

#endif // LLK_TRISC_UNPACK

#ifdef LLK_TRISC_MATH

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-variable"
#pragma GCC diagnostic ignored "-Wunused-parameter"
#include "experimental/llk_math_top32_rm.h"
#pragma GCC diagnostic pop
#include "llk_lib_math_wrappers.h"
#include "llk_math_common.h"
#include "llk_math_eltwise_unary_datacopy.h"
#include "llk_math_eltwise_unary_sfpu.h"
#include "llk_math_eltwise_unary_sfpu_params.h"
#include "sfpu/experimental/ckernel_sfpu_deepseek_top32_rm.h"

using namespace ckernel;

static constexpr bool DESCENDING = false;
static constexpr bool ASCENDING  = true;

inline void top32_phases_steps(std::uint32_t dst_tile, bool direction)
{
    _llk_math_eltwise_unary_sfpu_params_(
        ckernel::sfpu::_bitonic_top32_phases_steps_<false, is_fp32_dest_acc_en>, dst_tile, VectorMode::RC_custom, static_cast<int>(direction));
}

inline void top32_merge(std::uint32_t dst_tile, bool across_tiles)
{
    _llk_math_eltwise_unary_sfpu_params_(ckernel::sfpu::_bitonic_top32_merge_<false, is_fp32_dest_acc_en, false>, dst_tile, VectorMode::RC_custom, across_tiles);
}

inline void top32_rebuild(std::uint32_t dst_tile, bool direction, bool skip_second)
{
    _llk_math_eltwise_unary_sfpu_params_(
        ckernel::sfpu::_bitonic_top32_rebuild_<false, is_fp32_dest_acc_en>, dst_tile, VectorMode::RC_custom, direction, skip_second);
}

inline void copy_chunk_to_dest(std::uint32_t dst_tile, std::uint32_t math_format)
{
    _llk_math_top32_rm_init_<is_fp32_dest_acc_en>(4, math_format);
    if constexpr (unpack_to_dest)
    {
        _llk_math_transpose_dest_init_<false, true>();
        _llk_math_top32_rm_<DST_SYNC, is_fp32_dest_acc_en, true>(dst_tile, math_format, math_format, 4);
        _llk_math_transpose_dest_wrapper_<is_fp32_dest_acc_en, false, true>(dst_tile);
    }
    else
    {
        _llk_math_top32_rm_<DST_SYNC, is_fp32_dest_acc_en, false>(dst_tile, math_format, math_format, 4);
    }
}

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    const std::uint32_t LOOP_FACTOR = params.LOOP_FACTOR;
    const std::uint32_t math_format = formats.math;
    {
        START_PERF_MEASURE("INIT")
        _llk_math_pack_sync_init_<DST_SYNC, is_fp32_dest_acc_en>();
        _llk_math_hw_configure_<is_fp32_dest_acc_en>(math_format, math_format);
        PROFILER_SYNC();
    }
    {
        START_PERF_MEASURE("TILE_LOOP")
        for (std::uint32_t loop = 0; loop < LOOP_FACTOR; loop++)
        {
            _llk_math_wait_for_dest_available_<DST_SYNC>();
            _llk_math_eltwise_unary_sfpu_init_<SfpuType::unused>();
            ckernel::sfpu::_top32_rm_init_();
            for (std::uint32_t first = 0; first < TOP32_ROW_ELEMENTS; first += ELEMENTS_PER_CHUNK)
            {
                if (first == 0)
                {
                    copy_chunk_to_dest(VALUE_TILE, math_format);
                    copy_chunk_to_dest(INDEX_TILE, math_format);
                    top32_phases_steps(VALUE_TILE, DESCENDING);
                    top32_merge(VALUE_TILE, false);
                    top32_rebuild(VALUE_TILE, DESCENDING, true);
                    continue;
                }
                copy_chunk_to_dest(STAGE_VALUE_TILE, math_format);
                copy_chunk_to_dest(STAGE_INDEX_TILE, math_format);
                top32_phases_steps(STAGE_VALUE_TILE, DESCENDING);
                top32_merge(STAGE_VALUE_TILE, false);
                top32_rebuild(STAGE_VALUE_TILE, ASCENDING, true);
                top32_merge(VALUE_TILE, true);
                top32_rebuild(VALUE_TILE, DESCENDING, true);
            }
            _llk_math_dest_section_done_<DST_SYNC, is_fp32_dest_acc_en>();
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
    const std::uint32_t LOOP_FACTOR = params.LOOP_FACTOR;
    {
        START_PERF_MEASURE("INIT")
        _llk_pack_hw_configure_wrapper_<is_fp32_dest_acc_en, PackMode::Default>(formats.pack_src, formats.pack_dst, FACE_R_DIM * FACE_C_DIM * TILE_NUM_FACES);
        _llk_pack_init_wrapper_<PackMode::Default, false>(formats.pack_dst, FACE_R_DIM, TILE_C_DIM, TILE_NUM_FACES);
        _llk_pack_dest_init_wrapper_<DST_SYNC, is_fp32_dest_acc_en, PackMode::Default>();
        PROFILER_SYNC();
    }
    {
        START_PERF_MEASURE("TILE_LOOP")
        for (std::uint32_t loop = 0; loop < LOOP_FACTOR; loop++)
        {
            _llk_packer_wait_for_math_done_();
            // One datum per row: the survivors sit in the first column of Dest rows 0 to 31.
            TTI_SETADCXX(p_setadc::PAC, 1 - 1, 0x0);
            _llk_pack_<DST_SYNC, is_fp32_dest_acc_en, ckernel::PackMode::Default>(VALUE_TILE, L1_ADDRESS(params.buffer_Res[0]));
            _llk_pack_<DST_SYNC, is_fp32_dest_acc_en, ckernel::PackMode::Default>(INDEX_TILE, L1_ADDRESS(params.buffer_Res[1]));
            _llk_pack_dest_section_done_<DST_SYNC, is_fp32_dest_acc_en>();
        }
        PROFILER_SYNC();
    }
}

#endif // LLK_TRISC_PACK
