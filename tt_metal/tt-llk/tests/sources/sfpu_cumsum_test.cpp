// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Dedicated driver for the cumsum SFPU kernel pair (lane FK cumsum-fresh
// registration, first vehicle for this op):
//   * hand arm (CUMSUM_IMPL 0): the production raw-TTI kernel
//     tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/
//     ckernel_sfpu_cumsum.h (SFPTRANSP-bracketed replay blocks, LREG4-7
//     running-prefix cross-call ABI);
//   * fresh arm (CUMSUM_IMPL 1): the first typed semantic body
//     (fresh_cpp/cumsum.h), caller-held typed prefix state.
//
// Contract: inclusive prefix sum down the tile rows (per column), in place
// (input tile at dst index 0, output packed from dst index 0); consecutive
// tiles CONTINUE the row sequence — a [TILE_CNT*32, 32] input is a cumsum
// over TILE_CNT*32 rows for 32 parallel columns (the compute-API
// cumsum_tile(first=...) NWH contract at Wt=1).

#include <cstdint>

#include "ckernel.h"
#include "counters.h"
#include "llk_defs.h"
#include "params.h"
#include "profiler.h"

// Globals required by the test framework.
std::uint32_t unp_cfg_context          = 0;
std::uint32_t pack_sync_tile_dst_ptr   = 0;
std::uint32_t math_sync_tile_dst_index = 0;

static constexpr ckernel::DstSync DST_SYNC = ckernel::DstSync::SyncHalf;

#ifndef CUMSUM_IMPL
#define CUMSUM_IMPL 0
#endif

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

    _llk_unpack_A_init_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(
        0 /* transpose_of_faces */, 0 /* within_face_16x16_transpose */, ckernel::DEFAULT_TENSOR_SHAPE, formats.unpack_A_src, formats.unpack_A_dst);

    for (std::uint32_t tile = 0; tile < params.TILE_CNT; ++tile)
    {
        _llk_unpack_A_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(
            L1_ADDRESS(params.buffer_A[tile]), formats.unpack_A_src, formats.unpack_A_dst);
    }
}

#endif

#ifdef LLK_TRISC_MATH

#include "ckernel_sfpu.h"
#include "llk_lib_math_wrappers.h"
#include "llk_math_eltwise_unary_sfpu.h"
#include "llk_math_eltwise_unary_sfpu_params.h"
#include "llk_sfpu/llk_math_eltwise_unary_sfpu_macros.h"

#if CUMSUM_IMPL == 0
#include "llk_sfpu/ckernel_sfpu_cumsum.h"
#else
#include "fresh_cpp/cumsum.h"
#endif

using namespace ckernel;

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    _llk_math_hw_configure_<is_fp32_dest_acc_en>(formats.math, formats.math);
    _llk_math_pack_sync_init_<DST_SYNC, is_fp32_dest_acc_en>();

    _llk_math_eltwise_unary_datacopy_init_wrapper_<DataCopyType::A2D, is_fp32_dest_acc_en, BroadcastType::NONE, false /* is_int_fpu_en */, PackMode::Default>(
        TILE_NUM_FACES, formats.math);

    _llk_math_eltwise_unary_sfpu_init_<SfpuType::unused>();
#if CUMSUM_IMPL == 0
    // Production init: records the two 8-add replay chains.
    sfpu::cumsum_init<false>();
#else
    // Fresh typed arm: the running prefix is a caller-held typed quad pair
    // (the typed spelling of the hand kernel's LREG4-7 cross-call ABI).
    sfpu::CumsumFreshState cumsum_state;
    sfpu::cumsum_fresh_state_init(cumsum_state);
#endif

    for (std::uint32_t tile = 0; tile < params.TILE_CNT; ++tile)
    {
        _llk_math_wait_for_dest_available_<DST_SYNC>();

        _llk_math_eltwise_unary_datacopy_<DataCopyType::A2D, DST_SYNC, is_fp32_dest_acc_en, BroadcastType::NONE, unpack_to_dest>(
            0 /* dst_index */, formats.math, formats.math);

        // RC_custom: the kernel does its own DEST addressing (in place).
        {
            // Isolated device-profile zone (test_sfpu_cumsum_device_profile):
            // profiler-only L1 bookkeeping, compiles away in functional
            // builds. Replay record / state init stay outside the zone.
            START_PERF_MEASURE("CUMSUM_BODY")
#if CUMSUM_IMPL == 0
            SFPU_UNARY_CALL(DST_SYNC, is_fp32_dest_acc_en, calculate_cumsum, (false /* APPROXIMATE */), 0 /* dst_index */, VectorMode::RC_custom, tile == 0 /* first */);
#else
            _llk_math_eltwise_sfpu_start_(0);
            sfpu::_calculate_cumsum_fresh_tile_(cumsum_state);
            _llk_math_eltwise_sfpu_done_();
#endif
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
    _llk_pack_dest_init_wrapper_<DST_SYNC, is_fp32_dest_acc_en, PackMode::Default>();

    for (std::uint32_t tile = 0; tile < params.TILE_CNT; ++tile)
    {
        _llk_packer_wait_for_math_done_();
        _llk_pack_<DST_SYNC, is_fp32_dest_acc_en, ckernel::PackMode::Default>(0 /* tile_index */, L1_ADDRESS(params.buffer_Res[tile]));
        _llk_pack_dest_section_done_<DST_SYNC, is_fp32_dest_acc_en>();
    }
}

#endif
