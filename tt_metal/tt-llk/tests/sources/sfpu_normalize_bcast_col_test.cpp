// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// SFPU BCAST_COL normalize kernels (column 0 of the statistics tiles broadcast across the tile):
//   NORM_C_VAR 0: _calculate_sfpu_normalize_bcast_col_full_tile_(data, mean, out, inv)      out = (x - m) * s
//   NORM_C_VAR 1: _calculate_sfpu_normalize_bcast_col_two_tiles_(data, data2, out, m, s)   both data tiles in place
//   NORM_C_VAR 2: _calculate_sfpu_residual_normalize_bcast_col_full_tile_(data, res, out, m, s)  out = (x + r - m) * s
// Inputs: buffer_A[0..NORM_C_NA-1] (data, then data2 / residual), buffer_B[0] = mean, buffer_B[1] = inv_std, unpacked
// in that order to dest tiles NORM_C_IN0..NORM_C_IN3 (data, second, mean, inv). NORM_C_PACK0/NORM_C_PACK1 = packed tiles.
// NORM_C_SEQ 1: after the measured call, BCAST_COL add (data NORM_C_PACK0 += col 0 of the mean tile) without re-running
// the COL init (the add/sub/mul-after-normalize contract).
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

static constexpr std::uint32_t TILE_DATA            = NORM_C_IN0;
static constexpr std::uint32_t TILE_SECOND          = NORM_C_IN1;
static constexpr std::uint32_t TILE_MEAN            = NORM_C_IN2;
static constexpr std::uint32_t TILE_INV             = NORM_C_IN3;
static constexpr std::uint32_t TILE_OUT             = NORM_C_OUT;
static constexpr std::uint32_t INPUT_TILES_PER_LOOP = NORM_C_NA + 2;
// NORM_C_RUNTIME = 1: the measured call's tile indices pass through an empty asm, so the compiler cannot fold them into
// the TT_ encodings (models a runtime dst index, e.g. a non-unrolled tile loop in a compute kernel).
#if NORM_C_RUNTIME
#define NORM_C_OPAQUE(x)             \
    ({                               \
        std::uint32_t v_ = (x);      \
        asm volatile("" : "+r"(v_)); \
        v_;                          \
    })
#else
#define NORM_C_OPAQUE(x) (x)
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
                for (std::uint32_t a = 0; a < NORM_C_NA; ++a)
                {
                    _llk_unpack_A_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(
                        L1_ADDRESS(buffer_A[a]), formats.unpack_A_src, formats.unpack_A_dst);
                }
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
inline void copy_one(std::uint32_t tile, const FormatConfig& formats)
{
    _llk_math_eltwise_unary_datacopy_<DataCopyType::A2D, DST_SYNC, is_fp32_dest_acc_en, BroadcastType::NONE, unpack_to_dest_>(tile, formats.math, formats.math);
}

template <bool unpack_to_dest_>
inline void copy_inputs(const FormatConfig& formats)
{
    copy_one<unpack_to_dest_>(TILE_DATA, formats);
    if constexpr (NORM_C_NA == 2)
    {
        copy_one<unpack_to_dest_>(TILE_SECOND, formats);
    }
    copy_one<unpack_to_dest_>(TILE_MEAN, formats);
    copy_one<unpack_to_dest_>(TILE_INV, formats);
}

inline void measured_call()
{
#if NORM_C_VAR == 0
    _calculate_sfpu_normalize_bcast_col_full_tile_(NORM_C_OPAQUE(TILE_DATA), NORM_C_OPAQUE(TILE_MEAN), NORM_C_OPAQUE(TILE_OUT), NORM_C_OPAQUE(TILE_INV));
#elif NORM_C_VAR == 1
    _calculate_sfpu_normalize_bcast_col_two_tiles_(
        NORM_C_OPAQUE(TILE_DATA), NORM_C_OPAQUE(TILE_SECOND), NORM_C_OPAQUE(TILE_OUT), NORM_C_OPAQUE(TILE_MEAN), NORM_C_OPAQUE(TILE_INV));
#else
    _calculate_sfpu_residual_normalize_bcast_col_full_tile_(
        NORM_C_OPAQUE(TILE_DATA), NORM_C_OPAQUE(TILE_SECOND), NORM_C_OPAQUE(TILE_OUT), NORM_C_OPAQUE(TILE_MEAN), NORM_C_OPAQUE(TILE_INV));
#endif
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
        // = sfpu_bcast_col_init()
        _sfpu_binary_bcast_init_<BroadcastType::COL>();
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
                measured_call();
                _llk_math_eltwise_sfpu_done_();
            }
        }
        else
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
            {
                _llk_math_wait_for_dest_available_<DST_SYNC>();
                copy_inputs<unpack_to_dest>(formats);
                _llk_math_eltwise_sfpu_start_(0);
                measured_call();
                _llk_math_eltwise_sfpu_done_();
#if NORM_C_SEQ == 1
                _llk_math_eltwise_sfpu_start_(0);
                _calculate_sfpu_binary_bcast_full_tile_<BinaryOp::ADD, BroadcastType::COL>(NORM_C_PACK0, TILE_MEAN, NORM_C_PACK0);
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
                _llk_pack_<DST_SYNC, is_fp32_dest_acc_en, ckernel::PackMode::Default>(NORM_C_PACK0, L1_ADDRESS(buffer_Res[0]));
#if NORM_C_NOUT == 2
                _llk_pack_<DST_SYNC, is_fp32_dest_acc_en, ckernel::PackMode::Default>(NORM_C_PACK1, L1_ADDRESS(buffer_Res[1]));
#endif
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
                _llk_pack_<DST_SYNC, is_fp32_dest_acc_en, ckernel::PackMode::Default>(NORM_C_PACK0, L1_ADDRESS(buffer_Res[0]));
#if NORM_C_NOUT == 2
                _llk_pack_<DST_SYNC, is_fp32_dest_acc_en, ckernel::PackMode::Default>(NORM_C_PACK1, L1_ADDRESS(buffer_Res[1]));
#endif
                _llk_pack_dest_section_done_<DST_SYNC, is_fp32_dest_acc_en>();
            }
        }
        PROFILER_SYNC();
    }
}

#endif
