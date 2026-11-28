// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// _calculate_sfpu_normalize_bcast_scalar_full_tile_: out = (data - mean[0][0]) * inv_std[0][0].
// buffer_A[0] = data tile, buffer_B[0] = mean tile, buffer_B[1] = inv_std tile. The dest layout comes from the
// generated build header (NORM_S_*): NORM_S_DATA / NORM_S_MEAN / NORM_S_INV / NORM_S_OUT tile indices, NORM_S_PACK = packed tile.
// NORM_S_SEQ: 0 = measured call only; 1 = a scalar call with NORM_S_PRE_OUT as output first (forces a re-record with a
// different out offset); 2 = BCAST_ROW add (data NORM_S_MEAN, bcast NORM_S_INV, out NORM_S_PRE_OUT) first; 3 = after the
// measured call a BCAST_ROW add (data NORM_S_OUT, bcast NORM_S_DATA, out NORM_S_OUT).
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

static constexpr std::uint32_t TILE_DATA            = NORM_S_DATA;
static constexpr std::uint32_t TILE_MEAN            = NORM_S_MEAN;
static constexpr std::uint32_t TILE_INV             = NORM_S_INV;
static constexpr std::uint32_t TILE_OUT             = NORM_S_OUT;
static constexpr std::uint32_t TILE_PACK            = NORM_S_PACK;
static constexpr std::uint32_t INPUT_TILES_PER_LOOP = 3;
// NORM_S_RUNTIME = 1: the measured call's tile indices pass through an empty asm, so the compiler cannot fold them into
// the TT_ encodings (models a runtime dst index, e.g. a non-unrolled tile loop in a compute kernel).
#if NORM_S_RUNTIME
#define NORM_S_OPAQUE(x)             \
    ({                               \
        std::uint32_t v_ = (x);      \
        asm volatile("" : "+r"(v_)); \
        v_;                          \
    })
#define NORM_S_CALL_ARGS NORM_S_OPAQUE(TILE_DATA), NORM_S_OPAQUE(TILE_MEAN), NORM_S_OPAQUE(TILE_OUT), NORM_S_OPAQUE(TILE_INV)
#else
#define NORM_S_CALL_ARGS TILE_DATA, TILE_MEAN, TILE_OUT, TILE_INV
#endif

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
                _llk_unpack_A_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(
                    L1_ADDRESS(buffer_B[1]), formats.unpack_A_src, formats.unpack_A_dst);
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

template <bool unpack_to_dest_>
inline void copy_inputs(const FormatConfig& formats)
{
    _llk_math_eltwise_unary_datacopy_<DataCopyType::A2D, DST_SYNC, is_fp32_dest_acc_en, BroadcastType::NONE, unpack_to_dest_>(
        TILE_DATA, formats.math, formats.math);
    _llk_math_eltwise_unary_datacopy_<DataCopyType::A2D, DST_SYNC, is_fp32_dest_acc_en, BroadcastType::NONE, unpack_to_dest_>(
        TILE_MEAN, formats.math, formats.math);
    _llk_math_eltwise_unary_datacopy_<DataCopyType::A2D, DST_SYNC, is_fp32_dest_acc_en, BroadcastType::NONE, unpack_to_dest_>(
        TILE_INV, formats.math, formats.math);
}

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
        // = sfpu_normalize_bcast_scalar_init()
        _sfpu_binary_bcast_init_<BroadcastType::ROW>();
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
                    copy_inputs<unpack_to_dest>(formats);
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
                    copy_inputs<unpack_to_dest>(formats);
                }
                _llk_math_eltwise_sfpu_start_(0);
                _calculate_sfpu_normalize_bcast_scalar_full_tile_(NORM_S_CALL_ARGS);
                _llk_math_eltwise_sfpu_done_();
            }
        }
        else
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
            {
                _llk_math_wait_for_dest_available_<DST_SYNC>();
                copy_inputs<unpack_to_dest>(formats);
#if NORM_S_SEQ == 1
                _llk_math_eltwise_sfpu_start_(0);
                _calculate_sfpu_normalize_bcast_scalar_full_tile_(TILE_DATA, TILE_MEAN, NORM_S_PRE_OUT, TILE_INV);
                _llk_math_eltwise_sfpu_done_();
#elif NORM_S_SEQ == 2
                _llk_math_eltwise_sfpu_start_(0);
                _calculate_sfpu_binary_bcast_full_tile_<BinaryOp::ADD, BroadcastType::ROW>(TILE_MEAN, TILE_INV, NORM_S_PRE_OUT);
                _llk_math_eltwise_sfpu_done_();
#endif
                _llk_math_eltwise_sfpu_start_(0);
                _calculate_sfpu_normalize_bcast_scalar_full_tile_(NORM_S_CALL_ARGS);
                _llk_math_eltwise_sfpu_done_();
#if NORM_S_SEQ == 3
                _llk_math_eltwise_sfpu_start_(0);
                _calculate_sfpu_binary_bcast_full_tile_<BinaryOp::ADD, BroadcastType::ROW>(TILE_OUT, TILE_DATA, TILE_OUT);
                _llk_math_eltwise_sfpu_done_();
#endif
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
                _llk_pack_<DST_SYNC, is_fp32_dest_acc_en, ckernel::PackMode::Default>(TILE_PACK, L1_ADDRESS(buffer_Res[0]));
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
                _llk_pack_<DST_SYNC, is_fp32_dest_acc_en, ckernel::PackMode::Default>(TILE_PACK, L1_ADDRESS(buffer_Res[0]));
                _llk_pack_dest_section_done_<DST_SYNC, is_fp32_dest_acc_en>();
            }
        }
        PROFILER_SYNC();
    }
}

#endif
