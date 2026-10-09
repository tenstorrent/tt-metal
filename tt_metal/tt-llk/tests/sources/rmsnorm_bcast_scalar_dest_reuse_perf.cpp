// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// Perf kernel of the Blackhole rmsnorm bcast-scalar dest-reuse multiply. L1_TO_L1 runs the functional kernel's whole
// call per iteration (seed datacopy, inits, op, pack); the isolates seed and init once in INIT and run the op alone.

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

static constexpr ckernel::DstSync DST_SYNC = ckernel::DstSync::SyncHalf;
constexpr bool FULL_SEQUENCE               = (PERF_RUN_TYPE == PerfRunType::L1_TO_L1) || (PERF_RUN_TYPE == PerfRunType::L1_CONGESTION);
constexpr std::uint32_t SRCA_VALIDS_PER_TILE = RMSNORM_WHOLE_TILE ? 1 : RMSNORM_NUM_FACES;

#ifdef LLK_TRISC_UNPACK

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-variable"
#pragma GCC diagnostic ignored "-Wunused-parameter"
#include "experimental/llk_unpack_A_rmsnorm.h"
#pragma GCC diagnostic pop
#include "llk_unpack_A.h"
#include "llk_unpack_common.h"

inline void seed_unpack(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    _llk_unpack_A_init_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(
        0, 0, ckernel::DEFAULT_TENSOR_SHAPE, formats.unpack_A_src, formats.unpack_A_dst);
    _llk_unpack_A_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(
        L1_ADDRESS(params.buffer_B[0]), formats.unpack_A_src, formats.unpack_A_dst);
}

inline void op_unpack_init()
{
    _llk_unpack_A_rmsnorm_init_<RMSNORM_NUM_TILES, BroadcastType::SCALAR, true, EltwiseBinaryReuseDestType::DEST_TO_SRCB>(
        RMSNORM_UNPACK_FULL_TRANSPOSE, RMSNORM_UNPACK_FULL_TRANSPOSE, FACE_R_DIM, RMSNORM_NUM_FACES, 0, 0, RMSNORM_WHOLE_TILE);
}

inline void op_unpack(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    _llk_unpack_A_<BroadcastType::SCALAR, true, EltwiseBinaryReuseDestType::DEST_TO_SRCB>(
        L1_ADDRESS(params.buffer_A[0]), formats.unpack_A_src, formats.unpack_A_dst);
}

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
        if constexpr (!FULL_SEQUENCE && PERF_RUN_TYPE != PerfRunType::PACK_ISOLATE)
        {
            seed_unpack(params);
            op_unpack_init();
        }
        PROFILER_SYNC();
    }
    {
        START_PERF_MEASURE("TILE_LOOP")
        if constexpr (PERF_RUN_TYPE == PerfRunType::PACK_ISOLATE)
        {
        }
        else if constexpr (PERF_RUN_TYPE == PerfRunType::MATH_ISOLATE)
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
            {
                _perf_unpack_set_valid(ckernel::SrcB);
                for (std::uint32_t i = 0; i < RMSNORM_NUM_TILES * SRCA_VALIDS_PER_TILE; ++i)
                {
                    _perf_unpack_set_valid(ckernel::SrcA);
                }
            }
        }
        else if constexpr (PERF_RUN_TYPE == PerfRunType::UNPACK_ISOLATE)
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
            {
                op_unpack(params);
            }
        }
        else
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
            {
                seed_unpack(params);
                op_unpack_init();
                op_unpack(params);
            }
        }
        PROFILER_SYNC();
    }
}

#endif

#ifdef LLK_TRISC_MATH

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-variable"
#pragma GCC diagnostic ignored "-Wunused-parameter"
#include "experimental/llk_math_rmsnorm_bcast_scalar_dest_reuse.h"
#pragma GCC diagnostic pop
#include "llk_lib_math_wrappers.h"
#include "llk_math_common.h"
#include "llk_math_eltwise_unary_datacopy.h"

inline void seed_math(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    _llk_math_eltwise_unary_datacopy_init_wrapper_<DataCopyType::A2D, is_fp32_dest_acc_en, BroadcastType::NONE, false, PackMode::Default>(
        TILE_NUM_FACES, formats.math);
    _llk_math_eltwise_unary_datacopy_<DataCopyType::A2D, DST_SYNC, is_fp32_dest_acc_en, BroadcastType::NONE, unpack_to_dest>(0, formats.math, formats.math);
}

inline void op_math_init()
{
    _llk_math_rmsnorm_bcast_scalar_dest_reuse_init_<ELTWISE_BINARY_OP, RMSNORM_NUM_TILES, MATH_FIDELITY>(RMSNORM_NUM_FACES, 0 /* acc_to_dest */, RMSNORM_WHOLE_TILE);
}

inline void op_math()
{
    _llk_math_rmsnorm_bcast_scalar_dest_reuse_<ELTWISE_BINARY_OP, RMSNORM_NUM_TILES, DST_SYNC, is_fp32_dest_acc_en, MATH_FIDELITY, RMSNORM_CLEAR_DEST>(0, 0);
}

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    const std::uint32_t LOOP_FACTOR = params.LOOP_FACTOR;
    {
        START_PERF_MEASURE("INIT")
        _llk_math_pack_sync_init_<DST_SYNC, is_fp32_dest_acc_en>();
        _llk_math_hw_configure_<is_fp32_dest_acc_en>(formats.math, formats.math);
        if constexpr (!FULL_SEQUENCE && PERF_RUN_TYPE != PerfRunType::PACK_ISOLATE)
        {
            _llk_math_wait_for_dest_available_<DST_SYNC>();
            seed_math(params);
            op_math_init();
        }
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
                for (std::uint32_t i = 0; i < RMSNORM_NUM_TILES * SRCA_VALIDS_PER_TILE; ++i)
                {
                    _perf_math_clear_valid(ckernel::SrcA);
                }
                _perf_math_clear_valid(ckernel::SrcB);
            }
        }
        else if constexpr (PERF_RUN_TYPE == PerfRunType::MATH_ISOLATE)
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
            {
                op_math();
            }
        }
        else
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
            {
                _llk_math_wait_for_dest_available_<DST_SYNC>();
                seed_math(params);
                op_math_init();
                op_math();
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
        if constexpr (PERF_RUN_TYPE == PerfRunType::MATH_ISOLATE || PERF_RUN_TYPE == PerfRunType::UNPACK_ISOLATE)
        {
        }
        else if constexpr (PERF_RUN_TYPE == PerfRunType::PACK_ISOLATE || PERF_RUN_TYPE == PerfRunType::L1_CONGESTION)
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
            {
                for (std::uint32_t tile = 0; tile < RMSNORM_NUM_TILES; ++tile)
                {
                    _llk_pack_<DST_SYNC, is_fp32_dest_acc_en, ckernel::PackMode::Default>(tile, L1_ADDRESS(params.buffer_Res[tile]));
                }
            }
        }
        else
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
            {
                _llk_packer_wait_for_math_done_();
                for (std::uint32_t tile = 0; tile < RMSNORM_NUM_TILES; ++tile)
                {
                    _llk_pack_<DST_SYNC, is_fp32_dest_acc_en, ckernel::PackMode::Default>(tile, L1_ADDRESS(params.buffer_Res[tile]));
                }
                _llk_pack_dest_section_done_<DST_SYNC, is_fp32_dest_acc_en>();
            }
        }
        PROFILER_SYNC();
    }
}

#endif
