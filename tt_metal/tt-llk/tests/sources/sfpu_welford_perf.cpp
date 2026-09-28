// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Perf driver for the Welford SFPU kernel (ckernel_sfpu_welfords.h). Mirrors
// sources/sfpu_welford_test.cpp with the tile loop wrapped in the perf markers and
// repeated LOOP_FACTOR times to amortise profiler overhead.
//
// MATH_ISOLATE is the run type that matters: the per-tile body is a datacopy into
// dst 0 followed by _calculate_welfords_tile_<WELFORD_LUT_SIZE> (the same shape as
// ttnn welford_reduce_h / layernorm_welford). As in eltwise_unary_sfpu_perf.cpp, the
// datacopy stays in because it retires the SrcA valid bits unpack sets; it is a
// fixed cost that cancels in a before/after delta. WELFORD_DATACOPY_ONLY drops the
// Welford call and gives that fixed cost on its own (a flat control).
//
// The running mean / M2 stay in LREG4 / LREG5 across tiles; start_idx advances by
// 32 per tile and wraps each LOOP_FACTOR iteration so the LUT index stays in range.

#include <array>
#include <cstdint>

#include "ckernel.h"
#include "ckernel_defs.h"
#include "counters.h"
#include "llk_defs.h"
#include "params.h"
#include "perf.h"
#include "profiler.h"

// Globals
std::uint32_t unp_cfg_context              = 0;
std::uint32_t pack_sync_tile_dst_ptr       = 0;
std::uint32_t math_sync_tile_dst_index     = 0;
static constexpr ckernel::DstSync DST_SYNC = ckernel::DstSync::SyncHalf;

static constexpr std::uint32_t WELFORD_INPUT_DST_INDEX = 0;
static constexpr std::uint32_t WELFORD_MEAN_DST_INDEX  = 1;

#ifdef LLK_TRISC_UNPACK

#include "llk_unpack_A.h"
#include "llk_unpack_common.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
    {
        START_PERF_MEASURE("INIT")

        _llk_unpack_hw_configure_<is_fp32_dest_acc_en>(
            formats.unpack_A_src, formats.unpack_B_src, formats.unpack_A_dst, formats.unpack_B_dst, FACE_R_DIM, FACE_R_DIM, num_faces, num_faces);

        _llk_unpack_A_init_<BroadcastType::NONE, false /* acc_to_dest */, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(
            UNPACK_TRANSPOSE_FACES,
            UNPACK_TRANSPOSE_WITHIN_FACE,
            ckernel::make_tensor_shape_from_legacy(FACE_R_DIM, num_faces),
            formats.unpack_A_src,
            formats.unpack_A_dst);

        PROFILER_SYNC();
    }
    {
        START_PERF_MEASURE("TILE_LOOP")

        if constexpr (PERF_RUN_TYPE == PerfRunType::MATH_ISOLATE)
        {
            // One valid per face, matching what the math-side datacopy retires.
            _perf_unpack_loop_set_valid</* src A */ true, /* src B */ is_fp32_dest_acc_en>(num_faces * TILE_CNT * LOOP_FACTOR);
        }
        else if constexpr (PERF_RUN_TYPE != PerfRunType::PACK_ISOLATE)
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
            {
                for (std::uint32_t i = 0; i < TILE_CNT; ++i)
                {
                    _llk_unpack_A_<BroadcastType::NONE, false /* acc_to_dest */, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(
                        PERF_ADDRESS(PERF_INPUT_A, /* tile_idx */ i), formats.unpack_A_src, formats.unpack_A_dst);
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
#include "llk_math_eltwise_unary_sfpu.h"
#include "llk_math_welfords_sfpu.h"
#include "llk_math_welfords_sfpu_params.h"

using namespace ckernel;

static_assert(WELFORD_LUT_SIZE == 0 || WELFORD_LUT_SIZE >= TILE_CNT * 32, "reciprocal LUT must cover every sample index");

static std::array<std::uint32_t, WELFORD_LUT_SIZE> welford_reciprocal_lut;

static void fill_reciprocal_lut()
{
    if constexpr (WELFORD_LUT_SIZE > 0)
    {
        for (std::uint32_t i = 0; i < WELFORD_LUT_SIZE; ++i)
        {
            const float reciprocal = 1.0f / static_cast<float>(i + 1);
            std::uint32_t bits;
            __builtin_memcpy(&bits, &reciprocal, sizeof(bits));
            welford_reciprocal_lut[i] = bits;
        }
    }
}

static inline void welford_step(const std::uint32_t tile)
{
    _llk_math_eltwise_unary_datacopy_<DataCopyType::A2D, DST_SYNC, is_fp32_dest_acc_en, BroadcastType::NONE, unpack_to_dest>(
        WELFORD_INPUT_DST_INDEX, formats.math, formats.math);
    if constexpr (!WELFORD_DATACOPY_ONLY)
    {
        _llk_math_welfords_sfpu_params_(
            ckernel::sfpu::_calculate_welfords_tile_<WELFORD_LUT_SIZE>, WELFORD_INPUT_DST_INDEX, tile * 32, welford_reciprocal_lut);
    }
}

void run_kernel(RUNTIME_PARAMETERS params)
{
    {
        START_PERF_MEASURE("INIT")

        fill_reciprocal_lut();
        _llk_math_hw_configure_<is_fp32_dest_acc_en>(formats.math, formats.math);
        _llk_math_pack_sync_init_<DST_SYNC, is_fp32_dest_acc_en>();
        _llk_math_welfords_sfpu_init_();
        ckernel::sfpu::_clear_previous_mean_and_m2_();
        _llk_math_eltwise_unary_datacopy_init_wrapper_<
            DataCopyType::A2D,
            is_fp32_dest_acc_en,
            BroadcastType::NONE,
            false /* is_int_fpu_en */,
            PackMode::Default>(num_faces, formats.math);

        PROFILER_SYNC();
    }
    {
        START_PERF_MEASURE("TILE_LOOP")

        if constexpr (PERF_RUN_TYPE == PerfRunType::PACK_ISOLATE)
        {
            return;
        }
        else if constexpr (PERF_RUN_TYPE == PerfRunType::UNPACK_ISOLATE || PERF_RUN_TYPE == PerfRunType::L1_CONGESTION)
        {
            _perf_math_loop_clear_valid</* clear A */ true, /* clear B */ false>(TILE_CNT * LOOP_FACTOR);
        }
        else if constexpr (PERF_RUN_TYPE == PerfRunType::MATH_ISOLATE)
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
            {
                for (std::uint32_t tile = 0; tile < TILE_CNT; ++tile)
                {
                    welford_step(tile);
                }
            }
        }
        else // L1_TO_L1
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
            {
                for (std::uint32_t tile = 0; tile < TILE_CNT; ++tile)
                {
                    _llk_math_wait_for_dest_available_<DST_SYNC>();
                    welford_step(tile);
                    _llk_math_welfords_sfpu_params_(ckernel::sfpu::_store_mean_m2_to_dst_, WELFORD_MEAN_DST_INDEX);
                    _llk_math_dest_section_done_<DST_SYNC, is_fp32_dest_acc_en>();
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
    {
        START_PERF_MEASURE("INIT")

        _llk_pack_hw_configure_wrapper_<is_fp32_dest_acc_en, PackMode::Default>(formats.pack_src, formats.pack_dst, FACE_R_DIM * FACE_C_DIM * num_faces);
        _llk_pack_init_wrapper_<PackMode::Default, false /* zero_output */>(formats.pack_dst, FACE_R_DIM, TILE_C_DIM, num_faces);
        _llk_pack_dest_init_<DST_SYNC, is_fp32_dest_acc_en>();

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
                    _llk_pack_<DST_SYNC, is_fp32_dest_acc_en, ckernel::PackMode::Default>(WELFORD_MEAN_DST_INDEX, PERF_ADDRESS(PERF_OUTPUT, tile));
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
                    _llk_pack_<DST_SYNC, is_fp32_dest_acc_en, ckernel::PackMode::Default>(WELFORD_MEAN_DST_INDEX, PERF_ADDRESS(PERF_OUTPUT, tile));
                    _llk_pack_dest_section_done_<DST_SYNC, is_fp32_dest_acc_en>();
                }
            }
        }

        PROFILER_SYNC();
    }
}

#endif
