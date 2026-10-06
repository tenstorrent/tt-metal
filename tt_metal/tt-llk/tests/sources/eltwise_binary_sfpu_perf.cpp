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
#include "perf.h"
#include "profiler.h"

// Globals
// Globals
std::uint32_t unp_cfg_context                          = 0;
std::uint32_t pack_sync_tile_dst_ptr                   = 0;
std::uint32_t math_sync_tile_dst_index                 = 0;
static constexpr std::uint32_t MAX_TILES_DEST          = is_fp32_dest_acc_en ? 4 : 8;
static constexpr ckernel::DstSync DST_SYNC_MODE        = ckernel::DstSync::SyncHalf;
static constexpr ckernel::BroadcastType BROADCAST_TYPE = ckernel::BroadcastType::NONE;

#ifdef LLK_TRISC_UNPACK

#include "llk_unpack_A.h"
#ifdef TT_POLY_LLK_PERF_HAS_FPU
#include "llk_unpack_AB.h"
#endif
#include "llk_unpack_common.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif

#ifndef SPEED_OF_LIGHT
    const std::uint32_t LOOP_FACTOR = params.LOOP_FACTOR;
    const std::uint32_t num_faces   = params.num_faces;

    const std::uint32_t TILE_CNT = params.TILE_CNT;

    const bool UNPACK_TRANSPOSE_FACES       = params.UNPACK_TRANSPOSE_FACES;
    const bool UNPACK_TRANSPOSE_WITHIN_FACE = params.UNPACK_TRANSPOSE_WITHIN_FACE;
#endif

    const EltwiseBinaryReuseDestType reuse_dest_type = EltwiseBinaryReuseDestType::NONE;

    {
        START_PERF_MEASURE("INIT")

        _llk_unpack_hw_configure_<is_fp32_dest_acc_en>(
            formats.unpack_A_src, formats.unpack_B_src, formats.unpack_A_dst, formats.unpack_B_dst, FACE_R_DIM, FACE_R_DIM, num_faces, num_faces);

        _llk_unpack_A_init_<BROADCAST_TYPE, false /* acc_to_dest */, reuse_dest_type, unpack_to_dest>(
            UNPACK_TRANSPOSE_FACES,
            UNPACK_TRANSPOSE_WITHIN_FACE,
            ckernel::make_tensor_shape_from_legacy(FACE_R_DIM, num_faces),
            formats.unpack_A_src,
            formats.unpack_A_dst);
        PROFILER_SYNC();
    }
    {
        START_PERF_MEASURE("TILE_LOOP")

#ifdef TT_POLY_LLK_PERF_PAIRED
        for (std::uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
        {
            for (std::uint32_t stage = 0; stage < TT_POLY_LLK_STAGE_COUNT; ++stage)
            {
#ifdef TT_POLY_LLK_PERF_HAS_FPU
                if (TT_POLY_LLK_STAGE_FPU[stage])
                {
                    _llk_unpack_AB_init_<>(DEFAULT_TENSOR_SHAPE);
                }
                else
                {
                    _llk_unpack_A_init_<BROADCAST_TYPE, false, reuse_dest_type, unpack_to_dest>(
                        UNPACK_TRANSPOSE_FACES,
                        UNPACK_TRANSPOSE_WITHIN_FACE,
                        ckernel::make_tensor_shape_from_legacy(FACE_R_DIM, num_faces),
                        formats.unpack_A_src,
                        formats.unpack_A_dst);
                }
#endif
                for (std::uint32_t pair = 0; pair < TILE_CNT * TT_POLY_LLK_INPUT_ARITY; pair += TT_POLY_LLK_INPUT_ARITY)
                {
#ifdef TT_POLY_LLK_PERF_HAS_FPU
                    if (TT_POLY_LLK_STAGE_FPU[stage])
                    {
                        const auto address = [&](int slot)
                        {
                            return slot < 0   ? L1_ADDRESS(params.buffer_B[-slot - 1])
                                   : slot < 2 ? L1_ADDRESS(params.buffer_A[pair + slot])
                                              : L1_ADDRESS(params.buffer_C[(slot - 2) * TILE_CNT + pair / TT_POLY_LLK_INPUT_ARITY]);
                        };
                        _llk_unpack_AB_<>(address(TT_POLY_LLK_STAGE_INPUT[stage][0]), address(TT_POLY_LLK_STAGE_INPUT[stage][1]));
                        continue;
                    }
#endif
                    for (std::uint32_t lane = 0; lane < TT_POLY_LLK_STAGE_ARITY[stage]; ++lane)
                    {
                        const auto slot    = TT_POLY_LLK_STAGE_INPUT[stage][lane];
                        const auto address = slot < 0   ? L1_ADDRESS(params.buffer_B[-slot - 1])
                                             : slot < 2 ? L1_ADDRESS(params.buffer_A[pair + slot])
                                                        : L1_ADDRESS(params.buffer_C[(slot - 2) * TILE_CNT + pair / TT_POLY_LLK_INPUT_ARITY]);
#ifdef TT_POLY_LLK_PERF_DEST_REUSE
                        if (TT_POLY_LLK_STAGE_DEST_REUSE[stage] && lane == 1)
                        {
                            _llk_unpack_A_init_<BroadcastType::NONE, true, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(
                                false, false, DEFAULT_TENSOR_SHAPE, formats.unpack_A_src, formats.unpack_A_dst);
                            _llk_unpack_A_<BroadcastType::NONE, true, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(
                                address, formats.unpack_A_src, formats.unpack_A_dst);
                            continue;
                        }
                        _llk_unpack_A_init_<BROADCAST_TYPE, false, reuse_dest_type, unpack_to_dest>(
                            UNPACK_TRANSPOSE_FACES, UNPACK_TRANSPOSE_WITHIN_FACE, DEFAULT_TENSOR_SHAPE, formats.unpack_A_src, formats.unpack_A_dst);
#endif
                        _llk_unpack_A_<BROADCAST_TYPE, false, reuse_dest_type, unpack_to_dest>(address, formats.unpack_A_src, formats.unpack_A_dst);
                    }
                }
#ifdef TT_POLY_LLK_PERF_STAGED
                tensix_sync();
                llk_profiler::sync_threads();
#endif
            }
        }
#else
        if constexpr (PERF_RUN_TYPE == PerfRunType::MATH_ISOLATE)
        {
            // In case of math isolate, we don't want any software synchronization from unpack to math.
            // So we just set/clear valid bits here - which is unavoidable hardware synchronization.
            // When unpack_to_dest is used, we assume the data is immediately ready in destination register.
            // Otherwise, we assume the data is immediately ready in source A/B registers.
            if (!unpack_to_dest)
            {
                // Set valid for source A always.
                // Set valid for source B only if dest_acc is enabled.
                // Works only when unpacking to dest is not used.
                _perf_unpack_loop_set_valid<
                    /* src A */ true,
                    /* src B */ is_fp32_dest_acc_en>(
                    /* iterations*/ num_faces * TILE_CNT * LOOP_FACTOR);
            }
        }
        else if constexpr (PERF_RUN_TYPE != PerfRunType::PACK_ISOLATE) // UNPACK_ISOLATE, L1_TO_L1, L1_CONGESTION
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
            {
                for (std::uint32_t i = 0; i < TILE_CNT; ++i)
                {
                    _llk_unpack_A_<BROADCAST_TYPE, false /* acc_to_dest */, reuse_dest_type, unpack_to_dest>(
                        PERF_ADDRESS(PERF_INPUT_A, /* tile_idx */ i), formats.unpack_A_src, formats.unpack_A_dst);
                }
            }
        }
#endif
        PROFILER_SYNC();
    }
}

#endif // LLK_TRISC_UNPACK

#ifdef LLK_TRISC_MATH
#include "llk_math_common.h"
#include "llk_math_eltwise_binary_sfpu.h"
#if defined(TT_POLY_LLK_PERF_HAS_FPU) || defined(TT_POLY_LLK_PERF_DEST_REUSE)
#include "llk_math_eltwise_binary.h"
#endif
#include "llk_math_eltwise_unary_datacopy.h"
#include "sfpu_operations.h"
#ifdef TT_POLY_LLK_TEST_FACTOR_HEADER
#include TT_POLY_LLK_TEST_FACTOR_HEADER
#endif
#ifdef TT_POLY_LLK_TEST_HEADER
#include TT_POLY_LLK_TEST_HEADER
#endif

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif

#ifndef SPEED_OF_LIGHT
    const std::uint32_t LOOP_FACTOR = params.LOOP_FACTOR;
    const std::uint32_t num_faces   = params.num_faces;

    const std::uint32_t TILE_CNT = params.TILE_CNT;
#endif

    const DataCopyType data_copy_type = DataCopyType::A2D;

    {
        START_PERF_MEASURE("INIT")

        _llk_math_eltwise_unary_datacopy_init_<data_copy_type, is_fp32_dest_acc_en>(num_faces, formats.math);
        _llk_math_pack_sync_init_<DST_SYNC_MODE, is_fp32_dest_acc_en>();
        _llk_math_hw_configure_<is_fp32_dest_acc_en>(formats.math, formats.math);

#ifdef TT_POLY_LLK_PERF_PAIRED
#ifdef TT_POLY_LLK_TEST_HEADER
#if defined(TT_POLY_LLK_TEST_NO_STOCK_INIT) && !defined(TT_POLY_LLK_TEST_REPLACE_INIT)
        ckernel::llk_math_eltwise_unary_sfpu_init<SfpuType::unused, is_fp32_dest_acc_en>();
#elif !defined(TT_POLY_LLK_TEST_REPLACE_INIT)
        test_utils::call_unary_sfpu_operation_init<SFPU_UNARY_OPERATION, APPROX_MODE, is_fp32_dest_acc_en, ITERATIONS, false, false /* STABLE_SORT */, false>();
#endif
#ifdef TT_POLY_LLK_TEST_INIT
#ifdef TT_POLY_LLK_TEST_REPLACE_INIT
#ifdef TT_POLY_LLK_TEST_PRECISION_SPLIT
        ckernel::llk_math_eltwise_unary_sfpu_init<SFPU_UNARY_OPERATION>(ckernel::sfpu::TT_POLY_LLK_TEST_INIT<APPROX_MODE, true>);
#else
        ckernel::llk_math_eltwise_unary_sfpu_init<SFPU_UNARY_OPERATION>(ckernel::sfpu::TT_POLY_LLK_TEST_INIT<>);
#endif
#else
        ckernel::sfpu::TT_POLY_LLK_TEST_INIT();
#endif
#endif
#endif
#ifdef TT_POLY_LLK_TEST_FACTOR_HEADER
        test_utils::call_binary_sfpu_operation_init<APPROX_MODE, is_fp32_dest_acc_en, SFPU_BINARY_OPERATION, ITERATIONS>();
        ckernel::llk_math_eltwise_unary_sfpu_init<SfpuType::unused, is_fp32_dest_acc_en>();
#ifdef TT_POLY_LLK_TEST_FACTOR_INIT
        ckernel::sfpu::TT_POLY_LLK_TEST_FACTOR_INIT();
#endif
#endif
#else
        test_utils::call_binary_sfpu_operation_init<APPROX_MODE, is_fp32_dest_acc_en, SFPU_BINARY_OPERATION, ITERATIONS>();
#endif
        PROFILER_SYNC();
    }
    {
        START_PERF_MEASURE("TILE_LOOP")

#ifdef TT_POLY_LLK_PERF_PAIRED
        // One input/gradient pair owns DST; upper tiles remain callback scratch.
        static_assert(PERF_RUN_TYPE == PerfRunType::L1_TO_L1);
        for (std::uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
        {
            for (std::uint32_t stage = 0; stage < TT_POLY_LLK_STAGE_COUNT; ++stage)
            {
#ifdef TT_POLY_LLK_PERF_STOCK_CHAIN
                TT_POLY_LLK_STOCK_STAGE_INIT(stage);
#endif
                for (std::uint32_t pair = 0; pair < TILE_CNT * TT_POLY_LLK_INPUT_ARITY; pair += TT_POLY_LLK_INPUT_ARITY)
                {
                    _llk_math_wait_for_dest_available_<DST_SYNC_MODE>();
#ifdef TT_POLY_LLK_PERF_HAS_FPU
                    if (!TT_POLY_LLK_STAGE_FPU[stage])
#endif
#ifdef TT_POLY_LLK_PERF_INLINE_LOAD
                        if (!TT_POLY_LLK_STAGE_INLINE_LOAD[stage])
#endif
                            for (std::uint32_t tile = 0; tile < TT_POLY_LLK_STAGE_ARITY[stage]; ++tile)
                            {
#ifdef TT_POLY_LLK_TEST_COPY_REBASE
                                _llk_math_eltwise_unary_datacopy_<data_copy_type, DST_SYNC_MODE, is_fp32_dest_acc_en, BROADCAST_TYPE, unpack_to_dest>(
                                    1, formats.math, formats.math);
                                TT_SETC16(DEST_TARGET_REG_CFG_MATH_Offset_ADDR32, ckernel::get_dest_buffer_base());
#else
                        _llk_math_eltwise_unary_datacopy_<data_copy_type, DST_SYNC_MODE, is_fp32_dest_acc_en, BROADCAST_TYPE, unpack_to_dest>(
                            TT_POLY_LLK_STAGE_DST[stage][tile], formats.math, formats.math);
#endif
                            }
#ifdef TT_POLY_LLK_TEST_HEADER
#ifdef TT_POLY_LLK_TEST_CALC
                    SFPU_UNARY_CALL(
                        DST_SYNC_MODE, is_fp32_dest_acc_en, TT_POLY_LLK_TEST_CALC, (TT_POLY_LLK_TEST_ITERATIONS), 0, VectorMode::TT_POLY_LLK_TEST_VECTOR_MODE);
#elif defined(TT_POLY_LLK_TEST_STOCK_CALL)
                    TT_POLY_LLK_TEST_STOCK_CALL
#else
                    test_utils::call_unary_sfpu_operation<
                        DST_SYNC_MODE,
                        is_fp32_dest_acc_en,
                        SFPU_UNARY_OPERATION,
                        APPROX_MODE,
                        is_fp32_dest_acc_en,
                        ITERATIONS,
                        false,
                        false /* STABLE_SORT */,
                        false>(0, formats.math);
#endif
#elif defined(TT_POLY_LLK_TEST_FACTOR_HEADER)
                    if (TT_POLY_LLK_STAGE_GENERATED[stage])
                    {
                        SFPU_UNARY_CALL(DST_SYNC_MODE, is_fp32_dest_acc_en, TT_POLY_LLK_TEST_FACTOR_CALC, (32), 0, VectorMode::None);
#ifdef TT_POLY_LLK_TEST_FACTOR_CALC_2
                        if constexpr (TT_POLY_LLK_TEST_FACTOR_FINISH_IF)
                        {
                            SFPU_UNARY_CALL(DST_SYNC_MODE, is_fp32_dest_acc_en, TT_POLY_LLK_TEST_FACTOR_CALC_2, (32), 0, VectorMode::None);
                        }
#endif
                    }
#ifdef TT_POLY_LLK_PERF_STOCK_CHAIN
                    else
                    {
                        TT_POLY_LLK_STOCK_STAGE_CALC(stage);
                    }
#endif
#elif defined(TT_POLY_LLK_PERF_STOCK_CHAIN)
                    TT_POLY_LLK_STOCK_STAGE_CALC(stage);
#else
                    // The stock heterogeneous chain re-initializes each operation per tile.
                    test_utils::call_unary_sfpu_operation_init<SFPU_UNARY_OPERATION, false, is_fp32_dest_acc_en, ITERATIONS>();
                    test_utils::call_unary_sfpu_operation<DST_SYNC_MODE, is_fp32_dest_acc_en, SFPU_UNARY_OPERATION, false, is_fp32_dest_acc_en, ITERATIONS>(
                        0, formats.math);
                    // Gradient is the first multiply operand, as in the public kernel.
                    test_utils::call_binary_sfpu_operation_init<false, is_fp32_dest_acc_en, SFPU_BINARY_OPERATION, ITERATIONS>();
                    test_utils::call_binary_sfpu_operation<DST_SYNC_MODE, is_fp32_dest_acc_en, false, SFPU_BINARY_OPERATION, ITERATIONS, formats.math>(1, 0, 0);
#endif
                    _llk_math_dest_section_done_<DST_SYNC_MODE, is_fp32_dest_acc_en>();
                }
#ifdef TT_POLY_LLK_PERF_STAGED
                tensix_sync();
                llk_profiler::sync_threads();
#endif
            }
        }
#else
        if constexpr (PERF_RUN_TYPE == PerfRunType::UNPACK_ISOLATE)
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
            {
                for (std::uint32_t i = 0; i < TILE_CNT; ++i)
                {
                    // For unpack isolate scenario, math should only perform necessary synchronization and nothing else.
                    if constexpr (unpack_to_dest)
                    {
                        // In this case, unpacker needs software synchronization from math - to acknowledge that destination register is
                        // "consumed" and can be overwritten with new data.
                        // Due to the fact that BROADCAST_TYPE is always NONE in the test and combination of unpack_to_dest and 32b data is always set,
                        // this method will perform synchronization only and no actual data copy.
                        _llk_math_eltwise_unary_datacopy_<data_copy_type, DST_SYNC_MODE, is_fp32_dest_acc_en, BROADCAST_TYPE, unpack_to_dest>(
                            i % MAX_TILES_DEST, formats.math, formats.math);
                    }
                    else
                    {
                        // Perform only necessary hardware synchronization to indicate that source registers are consumed.
                        _perf_math_loop_clear_valid<
                            /* src A */ true,
                            /* src B */ true>(
                            /* iterations*/ num_faces);
                    }
                }
            }
        }
        else if constexpr (PERF_RUN_TYPE == PerfRunType::L1_CONGESTION)
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
            {
                for (std::uint32_t block_start = 0; block_start < TILE_CNT; block_start += MAX_TILES_DEST)
                {
                    int block_tiles = std::min(TILE_CNT - block_start, MAX_TILES_DEST);

                    for (int block_tile = 0; block_tile < block_tiles; ++block_tile)
                    {
                        if constexpr (unpack_to_dest)
                        {
                            // In this case, unpacker needs software synchronization from math - to acknowledge that destination register is
                            // "consumed" and can be overwritten with new data.
                            // Due to the fact that BROADCAST_TYPE is always NONE in the test and combination of unpack_to_dest and 32b data is always set,
                            // this method will perform synchronization only and no actual data copy.
                            _llk_math_eltwise_unary_datacopy_<data_copy_type, DST_SYNC_MODE, is_fp32_dest_acc_en, BROADCAST_TYPE, unpack_to_dest>(
                                block_tile, formats.math, formats.math);
                        }
                        else
                        {
                            // Perform only necessary hardware synchronization to indicate that source registers are consumed.
                            _perf_math_loop_clear_valid<
                                /* src A */ true,
                                /* src B */ true>(
                                /* iterations*/ num_faces);
                        }
                    }
                }
            }
        }
        else if constexpr (PERF_RUN_TYPE == PerfRunType::MATH_ISOLATE)
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
            {
                for (std::uint32_t block_start = 0; block_start < TILE_CNT; block_start += MAX_TILES_DEST)
                {
                    std::uint32_t block_tiles = std::min(TILE_CNT - block_start, MAX_TILES_DEST);

                    for (std::uint32_t block_tile = 0; block_tile < block_tiles; ++block_tile)
                    {
                        // When data is not unpacked to dest, math needs to copy data from srcA to dest before starting SFPU operation.
                        // Otherwise, data is immediately ready in destination register.
                        if constexpr (!unpack_to_dest)
                        {
                            LLK_ASSERT(
                                (block_tile < get_dest_max_tiles<DstSync::SyncHalf, is_fp32_dest_acc_en, DstTileShape::Tile32x32>()),
                                "block_tile exceeds max dest tiles");
                            _llk_math_eltwise_unary_datacopy_<data_copy_type, DST_SYNC_MODE, is_fp32_dest_acc_en, BROADCAST_TYPE, unpack_to_dest>(
                                block_tile, formats.math, formats.math);
                        }

                        test_utils::
                            call_binary_sfpu_operation<DST_SYNC_MODE, is_fp32_dest_acc_en, APPROX_MODE, SFPU_BINARY_OPERATION, ITERATIONS, formats.math>(
                                block_tile, (block_tile + 1) % MAX_TILES_DEST, block_tile);
                    }
                }
            }
        }
        else if constexpr (PERF_RUN_TYPE == PerfRunType::L1_TO_L1)
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
            {
                for (std::uint32_t block_start = 0; block_start < TILE_CNT; block_start += MAX_TILES_DEST)
                {
                    std::uint32_t block_tiles = std::min(TILE_CNT - block_start, MAX_TILES_DEST);

                    _llk_math_wait_for_dest_available_<DST_SYNC_MODE>();

                    // Copy from srcA to dest
                    for (std::uint32_t block_tile = 0; block_tile < block_tiles; ++block_tile)
                    {
                        LLK_ASSERT(
                            (block_tile < get_dest_max_tiles<DST_SYNC_MODE, is_fp32_dest_acc_en, DstTileShape::Tile32x32>()),
                            "block_tile exceeds max dest tiles");
                        _llk_math_eltwise_unary_datacopy_<data_copy_type, DST_SYNC_MODE, is_fp32_dest_acc_en, BROADCAST_TYPE, unpack_to_dest>(
                            block_tile, formats.math, formats.math);

                        // Start SFPU binary operation
                        test_utils::
                            call_binary_sfpu_operation<DST_SYNC_MODE, is_fp32_dest_acc_en, APPROX_MODE, SFPU_BINARY_OPERATION, ITERATIONS, formats.math>(
                                block_tile, (block_tile + 1) % MAX_TILES_DEST, block_tile);
                    }

                    _llk_math_dest_section_done_<DST_SYNC_MODE, is_fp32_dest_acc_en>();
                }
            }
        }
#endif
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

    const std::uint32_t TILE_CNT = params.TILE_CNT;
#endif

    {
        START_PERF_MEASURE("INIT")

        // Configure packer hardware
        _llk_pack_hw_configure_<is_fp32_dest_acc_en, ckernel::PackMode::Default>(formats.pack_src, formats.pack_dst, FACE_R_DIM * FACE_C_DIM * num_faces);

        _llk_pack_init_wrapper_<PackMode::Default, false /* zero_output */>(formats.pack_dst, FACE_R_DIM, TILE_C_DIM, num_faces);
        // Initialize destination for packing
        _llk_pack_dest_init_<DST_SYNC_MODE, is_fp32_dest_acc_en>();

        PROFILER_SYNC();
    }
    {
        START_PERF_MEASURE("TILE_LOOP")

#ifdef TT_POLY_LLK_PERF_PAIRED
        for (std::uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
        {
            for (std::uint32_t stage = 0; stage < TT_POLY_LLK_STAGE_COUNT; ++stage)
            {
                for (std::uint32_t pair = 0; pair < TILE_CNT * TT_POLY_LLK_INPUT_ARITY; pair += TT_POLY_LLK_INPUT_ARITY)
                {
                    _llk_packer_wait_for_math_done_();
                    const auto slot    = TT_POLY_LLK_STAGE_OUTPUT[stage];
                    const auto address = slot == TT_POLY_LLK_RESULT_SLOT ? L1_ADDRESS(params.buffer_Res[pair / TT_POLY_LLK_INPUT_ARITY])
                                                                         : L1_ADDRESS(params.buffer_C[(slot - 2) * TILE_CNT + pair / TT_POLY_LLK_INPUT_ARITY]);
                    _llk_pack_<DST_SYNC_MODE, is_fp32_dest_acc_en>(0, address);
                    _llk_pack_dest_section_done_<DST_SYNC_MODE, is_fp32_dest_acc_en>();
                }
#ifdef TT_POLY_LLK_PERF_STAGED
                // Section completion drains PACK before publishing the L1 tensor.
                tensix_sync();
                llk_profiler::sync_threads();
#endif
            }
        }
#else
        if constexpr (PERF_RUN_TYPE == PerfRunType::PACK_ISOLATE || PERF_RUN_TYPE == PerfRunType::L1_CONGESTION)
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
            {
                for (std::uint32_t block_start = 0; block_start < TILE_CNT; block_start += MAX_TILES_DEST)
                {
                    std::uint32_t block_tiles = std::min(TILE_CNT - block_start, MAX_TILES_DEST);

                    for (std::uint32_t block_tile = 0; block_tile < block_tiles; block_tile++)
                    {
                        LLK_ASSERT(
                            (block_tile < get_dest_max_tiles<DST_SYNC_MODE, is_fp32_dest_acc_en, DstTileShape::Tile32x32>()),
                            "block_tile exceeds max dest tiles");
                        _llk_pack_<DST_SYNC_MODE, is_fp32_dest_acc_en>(block_tile, PERF_ADDRESS(PERF_OUTPUT, block_start + block_tile));
                    }
                }
            }
        }
        else if constexpr (PERF_RUN_TYPE == PerfRunType::L1_TO_L1)
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
            {
                for (std::uint32_t block_start = 0; block_start < TILE_CNT; block_start += MAX_TILES_DEST)
                {
                    std::uint32_t block_tiles = std::min(TILE_CNT - block_start, MAX_TILES_DEST);

                    _llk_packer_wait_for_math_done_();
                    for (std::uint32_t block_tile = 0; block_tile < block_tiles; block_tile++)
                    {
                        LLK_ASSERT(
                            (block_tile < get_dest_max_tiles<DST_SYNC_MODE, is_fp32_dest_acc_en, DstTileShape::Tile32x32>()),
                            "block_tile exceeds max dest tiles");
                        _llk_pack_<DST_SYNC_MODE, is_fp32_dest_acc_en>(block_tile, PERF_ADDRESS(PERF_OUTPUT, block_start + block_tile));
                    }
                    _llk_pack_dest_section_done_<DST_SYNC_MODE, is_fp32_dest_acc_en>();
                }
            }
        }

#endif
        PROFILER_SYNC();
    }
}

#endif // LLK_TRISC_PACK
