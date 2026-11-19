// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
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
#include "perf.h"
#include "profiler.h"

// Globals
std::uint32_t unp_cfg_context              = 0;
std::uint32_t pack_sync_tile_dst_ptr       = 0;
std::uint32_t math_sync_tile_dst_index     = 0;
static constexpr ckernel::DstSync DST_SYNC = ckernel::DstSync::SyncHalf;

// Dest layout from the generated build header (BCAST_L_*), so the data tile can sit at a
// nonzero index and the bcast / result tiles below it. BCAST_L_PRE_DATA >= 0 adds a first call on that scratch tile
// (in place, same bcast tile) before the measured one, so the second call must re-record with a different layout.
static constexpr std::uint32_t INPUT_TILE_A = BCAST_L_DATA;
static constexpr std::uint32_t INPUT_TILE_B = BCAST_L_BCAST;
static constexpr std::uint32_t RESULT_TILE  = BCAST_L_OUT;
// BCAST_L_RUNTIME = 1: the measured call's tile indices pass through an empty asm, so the compiler cannot fold them
// into the TT_ encodings (models a runtime dst index, e.g. a non-unrolled tile loop in a compute kernel).
#if BCAST_L_RUNTIME
#define BCAST_OPAQUE(x)              \
    ({                               \
        std::uint32_t v_ = (x);      \
        asm volatile("" : "+r"(v_)); \
        v_;                          \
    })
#define BCAST_CALL_ARGS BCAST_OPAQUE(INPUT_TILE_A), BCAST_OPAQUE(INPUT_TILE_B), BCAST_OPAQUE(RESULT_TILE)
#else
#define BCAST_CALL_ARGS INPUT_TILE_A, INPUT_TILE_B, RESULT_TILE
#endif
static constexpr std::uint32_t INPUT_TILES_PER_LOOP = 2;

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
    const Operand& buffer_A         = params.buffer_A;
    const Operand& buffer_B         = params.buffer_B;
#endif
    // Mocks post the same src dvalids per tile as _llk_unpack_A_.
    const std::uint32_t src_handshake_iters = LOOP_FACTOR * INPUT_TILES_PER_LOOP * num_faces;

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
        if constexpr (PERF_RUN_TYPE == PerfRunType::PACK_ISOLATE)
        {
        }
        else if constexpr (PERF_RUN_TYPE == PerfRunType::MATH_ISOLATE)
        {
            if constexpr (!unpack_to_dest)
            {
                // SrcA plus a zeroed SrcB every face, including dest_acc=No, as _llk_unpack_A_ (BroadcastType::NONE) posts them.
                _perf_unpack_stream_set_valid<true /*set_a*/, true /*set_b*/, PERF_RUN_TYPE>(src_handshake_iters /*iterations*/);
            }
        }
        else
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
            {
                _llk_unpack_A_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(
                    L1_ADDRESS(buffer_A[0]), formats.unpack_A_src, formats.unpack_A_dst);
                _llk_unpack_A_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(
                    L1_ADDRESS(buffer_B[0]), formats.unpack_A_src, formats.unpack_A_dst);
            }
        }
        PROFILER_SYNC();
    }
    _llk_unpack_A_uninit_<BroadcastType::NONE>();
}

#endif

#ifdef LLK_TRISC_MATH

#include "ckernel_sfpu.h"
#include "llk_lib_math_wrappers.h"
#include "llk_math_eltwise_binary_sfpu.h"
#include "llk_sfpu/ckernel_sfpu_binary_bcast.h"

using namespace ckernel;
using namespace ckernel::sfpu;

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
#ifndef SPEED_OF_LIGHT
    const std::uint32_t LOOP_FACTOR = params.LOOP_FACTOR;
    const std::uint32_t num_faces   = params.num_faces;
#endif
    const bool is_int_fpu_en                = false;
    const std::uint32_t src_handshake_iters = LOOP_FACTOR * INPUT_TILES_PER_LOOP * num_faces;

    {
        START_PERF_MEASURE("INIT")
        _llk_math_eltwise_unary_datacopy_init_wrapper_<DataCopyType::A2D, is_fp32_dest_acc_en, BroadcastType::NONE, is_int_fpu_en, PackMode::Default>(
            ckernel::make_tensor_shape_from_legacy(FACE_R_DIM, num_faces), formats.math);
        _llk_math_hw_configure_<is_fp32_dest_acc_en>(formats.math, formats.math);
        _llk_math_pack_sync_init_<DST_SYNC, is_fp32_dest_acc_en>();
        _llk_math_eltwise_binary_sfpu_init_<SfpuType::add1>();
        _sfpu_binary_bcast_init_<BCAST_DIM>();
        PROFILER_SYNC();
    }
    {
        START_PERF_MEASURE("TILE_LOOP")
        if constexpr (PERF_RUN_TYPE == PerfRunType::PACK_ISOLATE)
        {
        }
        else if constexpr (PERF_RUN_TYPE == PerfRunType::UNPACK_ISOLATE)
        {
            if constexpr (unpack_to_dest)
            {
                for (std::uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
                {
                    _llk_math_eltwise_unary_datacopy_<DataCopyType::A2D, DST_SYNC, is_fp32_dest_acc_en, BroadcastType::NONE, unpack_to_dest>(
                        INPUT_TILE_A, formats.math, formats.math);
                    _llk_math_eltwise_unary_datacopy_<DataCopyType::A2D, DST_SYNC, is_fp32_dest_acc_en, BroadcastType::NONE, unpack_to_dest>(
                        INPUT_TILE_B, formats.math, formats.math);
                }
            }
            else
            {
                _perf_math_stream_clear_valid<true /*clear_a*/, true /*clear_b*/, PERF_RUN_TYPE>(src_handshake_iters /*iterations*/);
            }
        }
        else if constexpr (PERF_RUN_TYPE == PerfRunType::MATH_ISOLATE)
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
            {
                if constexpr (!unpack_to_dest)
                {
                    _llk_math_eltwise_unary_datacopy_<DataCopyType::A2D, DST_SYNC, is_fp32_dest_acc_en, BroadcastType::NONE, unpack_to_dest>(
                        INPUT_TILE_A, formats.math, formats.math);
                    _llk_math_eltwise_unary_datacopy_<DataCopyType::A2D, DST_SYNC, is_fp32_dest_acc_en, BroadcastType::NONE, unpack_to_dest>(
                        INPUT_TILE_B, formats.math, formats.math);
                }

                _llk_math_eltwise_sfpu_start_(0);
                _calculate_sfpu_binary_bcast_full_tile_<SFPU_BINARY_OPERATION, BCAST_DIM>(BCAST_CALL_ARGS);
                _llk_math_eltwise_sfpu_done_();
            }
        }
        else
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
            {
                _llk_math_wait_for_dest_available_<DST_SYNC>();
                _llk_math_eltwise_unary_datacopy_<DataCopyType::A2D, DST_SYNC, is_fp32_dest_acc_en, BroadcastType::NONE, unpack_to_dest>(
                    INPUT_TILE_A, formats.math, formats.math);
                _llk_math_eltwise_unary_datacopy_<DataCopyType::A2D, DST_SYNC, is_fp32_dest_acc_en, BroadcastType::NONE, unpack_to_dest>(
                    INPUT_TILE_B, formats.math, formats.math);

#if BCAST_L_PRE_DATA >= 0
                _llk_math_eltwise_sfpu_start_(0);
                _calculate_sfpu_binary_bcast_full_tile_<SFPU_BINARY_OPERATION, BCAST_DIM>(BCAST_L_PRE_DATA, INPUT_TILE_B, BCAST_L_PRE_DATA);
                _llk_math_eltwise_sfpu_done_();
#endif
                _llk_math_eltwise_sfpu_start_(0);
                _calculate_sfpu_binary_bcast_full_tile_<SFPU_BINARY_OPERATION, BCAST_DIM>(BCAST_CALL_ARGS);
                _llk_math_eltwise_sfpu_done_();
                _llk_math_dest_section_done_<DST_SYNC, is_fp32_dest_acc_en>();
            }
        }
        PROFILER_SYNC();
    }
    _llk_math_eltwise_unary_datacopy_uninit_<BroadcastType::NONE, unpack_to_dest>();
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
#ifndef SPEED_OF_LIGHT
    const std::uint32_t LOOP_FACTOR = params.LOOP_FACTOR;
    const std::uint32_t num_faces   = params.num_faces;
    const Operand& buffer_Res       = params.buffer_Res;
#endif
    {
        START_PERF_MEASURE("INIT")
        _llk_pack_hw_configure_wrapper_<is_fp32_dest_acc_en, PackMode::Default>(formats.pack_src, formats.pack_dst, FACE_R_DIM * FACE_C_DIM * num_faces);
        _llk_pack_init_wrapper_<PackMode::Default, false /* zero_output */>(formats.pack_dst, FACE_R_DIM, TILE_C_DIM, num_faces);
        _llk_pack_dest_init_<DST_SYNC, is_fp32_dest_acc_en>();
        PROFILER_SYNC();
    }
    {
        START_PERF_MEASURE("TILE_LOOP")
        if constexpr (PERF_RUN_TYPE == PerfRunType::UNPACK_ISOLATE || PERF_RUN_TYPE == PerfRunType::MATH_ISOLATE)
        {
        }
        else if constexpr (PERF_RUN_TYPE == PerfRunType::PACK_ISOLATE)
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
            {
                _llk_pack_<DST_SYNC, is_fp32_dest_acc_en, ckernel::PackMode::Default>(RESULT_TILE, L1_ADDRESS(buffer_Res[0]));
                // Wormhole: release the dest section as production's tile_regs_release does, so every section starts with drained packers
                if constexpr (PERF_PACK_SECTION_RELEASE)
                {
                    _llk_pack_dest_section_done_<DST_SYNC, is_fp32_dest_acc_en>();
                }
            }
        }
        else
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
            {
                _llk_packer_wait_for_math_done_();
                _llk_pack_<DST_SYNC, is_fp32_dest_acc_en, ckernel::PackMode::Default>(RESULT_TILE, L1_ADDRESS(buffer_Res[0]));
                _llk_pack_dest_section_done_<DST_SYNC, is_fp32_dest_acc_en>();
            }
        }
        PROFILER_SYNC();
    }
}

#endif
