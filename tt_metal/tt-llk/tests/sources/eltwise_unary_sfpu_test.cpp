// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <type_traits>

#include "ckernel.h"
#include "ckernel_defs.h"
#include "counters.h"
#include "llk_defs.h"
#include "params.h"
#include "profiler.h"

// Globals
std::uint32_t unp_cfg_context              = 0;
std::uint32_t pack_sync_tile_dst_ptr       = 0;
std::uint32_t math_sync_tile_dst_index     = 0;
static constexpr ckernel::DstSync DST_SYNC = ckernel::DstSync::SyncHalf;

#ifdef LLK_TRISC_UNPACK

#include "llk_unpack_A.h"
#include "llk_unpack_common.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    _llk_unpack_hw_configure_<is_fp32_dest_acc_en>(
        formats.unpack_A_src, formats.unpack_B_src, formats.unpack_A_dst, formats.unpack_B_dst, FACE_R_DIM, FACE_R_DIM, TILE_NUM_FACES, TILE_NUM_FACES);

    _llk_unpack_A_init_<BroadcastType::NONE, false /* is_fp32_dest_acc_en - why true does not work? */, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(
        0 /* transpose_of_faces */,
        0 /* within_face_16x16_transpose */,
        ckernel::make_tensor_shape_from_legacy(FACE_R_DIM, TILE_NUM_FACES),
        formats.unpack_A_src,
        formats.unpack_A_dst);

    for (std::uint32_t i = 0; i < params.NUM_BLOCKS * params.NUM_TILES_IN_BLOCK; ++i)
    {
        _llk_unpack_A_<BroadcastType::NONE, false /* is_fp32_dest_acc_en - why true does not work? */, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(
            L1_ADDRESS(params.buffer_A[i]), formats.unpack_A_src, formats.unpack_A_dst);
    }
}

#endif

#ifdef LLK_TRISC_MATH

#include "ckernel_sfpu.h"
#include "llk_lib_math_wrappers.h"
#include "llk_math_eltwise_unary_sfpu.h"
#include "sfpu_operations.h"
#include "fresh_cpp_operations.h"

#ifndef FRESH_CPP_IMPL
#define FRESH_CPP_IMPL 0
#endif

using namespace ckernel;
using namespace ckernel::sfpu;

const int iterations = 32;

// Fresh semantic-C++ reciprocal.  The body names only typed Dst values and
// reciprocal arithmetic; physical LREGs, macro templates, replay ranges, and
// instruction scheduling remain compiler responsibilities.
template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_reciprocal_semantic()
{
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; ++d)
    {
        sfpi::vFloat input = sfpi::dst_reg[0];
#ifdef ARCH_BLACKHOLE
        sfpi::vFloat result = sfpi::approx_recip(input);
        if constexpr (!APPROXIMATION_MODE)
        {
            // Cubic Newton correction in value space.  min maps the NaN error
            // produced at zero/infinity to 1.0 under SFPU min semantics, so
            // the final y+y preserves the architectural pole behavior without
            // the v_if form that currently trips rvtt_expand SSA verification.
            sfpi::vFloat error = 1.0f - input * result;
            sfpi::vFloat correction = error * error + error;
            correction = correction * error + error;
            correction = sfpi::min(correction, 1.0f);
            result = correction * result + result;
        }
#else
        // Wormhole's typed reciprocal helper implements the same semantic
        // operation with its polynomial seed; approx_recip is BH-only.
        sfpi::vFloat result = sfpu_reciprocal<APPROXIMATION_MODE>(input);
#endif
        sfpi::dst_reg[0] = result;
        sfpi::dst_reg++;
    }
}

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
// copy srca to dest
    _llk_math_eltwise_unary_datacopy_init_wrapper_<DataCopyType::A2D, is_fp32_dest_acc_en, BroadcastType::NONE, false /* is_int_fpu_en */, PackMode::Default>(
        TILE_NUM_FACES, formats.math);
    _llk_math_hw_configure_<is_fp32_dest_acc_en>(formats.math, formats.math);
    _llk_math_pack_sync_init_<DST_SYNC, is_fp32_dest_acc_en>();

    test_utils::call_unary_sfpu_operation_init<
        SFPU_UNARY_OPERATION,
        APPROX_MODE,
        is_fp32_dest_acc_en,
        iterations,
        FAST_MODE,
        false /* STABLE_SORT */,
        CLAMP_NEGATIVE>();

    LLK_ASSERT(
        (params.NUM_TILES_IN_BLOCK <= get_dest_max_tiles<DST_SYNC, is_fp32_dest_acc_en, DstTileShape::Tile32x32>()),
        "NUM_TILES_IN_BLOCK exceeds max dest tiles");

    for (int block_start = 0; block_start < params.NUM_BLOCKS; block_start++)
    {
        _llk_math_wait_for_dest_available_<DST_SYNC>();
        for (std::uint32_t block_tile = 0; block_tile < params.NUM_TILES_IN_BLOCK; ++block_tile)
        {
            _llk_math_eltwise_unary_datacopy_<DataCopyType::A2D, DST_SYNC, is_fp32_dest_acc_en, BroadcastType::NONE, unpack_to_dest>(
                block_tile, formats.math, formats.math);

            // calculation of sfpu operation on dest
            // calling sfpu function from ckernel
            // this part is where parametrization of operation takes part
            {
                START_PERF_MEASURE("RECIPROCAL_BODY")
            if constexpr (FRESH_CPP_IMPL == 1 && SFPU_UNARY_OPERATION == SfpuType::exponential)
            {
                static_assert(!APPROX_MODE && !is_fp32_dest_acc_en);
                SFPU_UNARY_CALL(
                    DST_SYNC,
                    is_fp32_dest_acc_en,
                    calculate_exp_fresh_cpp,
                    (iterations),
                    block_tile,
                    VectorMode::None);
            }
            else if constexpr (FRESH_CPP_IMPL == 1 && SFPU_UNARY_OPERATION == SfpuType::sigmoid_appx)
            {
                SFPU_UNARY_CALL(
                    DST_SYNC,
                    is_fp32_dest_acc_en,
                    calculate_sigmoid_appx_fresh_cpp,
                    (iterations),
                    block_tile,
                    VectorMode::None);
            }
            else if constexpr (RECIPROCAL_IMPL == 1 && SFPU_UNARY_OPERATION == SfpuType::reciprocal)
            {
                _llk_math_eltwise_unary_sfpu_params_(
                    calculate_reciprocal_semantic<APPROX_MODE, iterations>,
                    block_tile,
                    VectorMode::None);
            }
            else
            {
                test_utils::call_unary_sfpu_operation<
                    DST_SYNC,
                    is_fp32_dest_acc_en,
                    SFPU_UNARY_OPERATION,
                    APPROX_MODE,
                    is_fp32_dest_acc_en,
                    iterations,
                    FAST_MODE,
                    false /* STABLE_SORT */,
                    CLAMP_NEGATIVE>(block_tile, formats.math);
            }
            }
        }

        _llk_math_dest_section_done_<DST_SYNC, is_fp32_dest_acc_en>();
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
    _llk_pack_hw_configure_wrapper_<is_fp32_dest_acc_en, PackMode::Default>(formats.pack_src, formats.pack_dst, FACE_R_DIM * FACE_C_DIM * TILE_NUM_FACES);
    _llk_pack_init_wrapper_<PackMode::Default, false /* zero_output */>(formats.pack_dst, FACE_R_DIM, TILE_C_DIM, TILE_NUM_FACES);
    _llk_pack_dest_init_<DST_SYNC, is_fp32_dest_acc_en>();
    LLK_ASSERT(
        (params.NUM_TILES_IN_BLOCK <= get_dest_max_tiles<DST_SYNC, is_fp32_dest_acc_en, DstTileShape::Tile32x32>()),
        "NUM_TILES_IN_BLOCK exceeds max dest tiles");

    for (int block_start = 0; block_start < params.NUM_BLOCKS; block_start++)
    {
        _llk_packer_wait_for_math_done_();
        for (std::uint32_t block_tile = 0; block_tile < params.NUM_TILES_IN_BLOCK; ++block_tile)
        {
            _llk_pack_<DST_SYNC, is_fp32_dest_acc_en, ckernel::PackMode::Default>(
                block_tile, L1_ADDRESS(params.buffer_Res[block_start * params.NUM_TILES_IN_BLOCK + block_tile]));
        }
        _llk_pack_dest_section_done_<DST_SYNC, is_fp32_dest_acc_en>();
    }
}

#endif
