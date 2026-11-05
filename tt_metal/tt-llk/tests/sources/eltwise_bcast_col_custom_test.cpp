// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "ckernel.h"
#include "ckernel_defs.h"
#include "counters.h"
#include "llk_defs.h"
#include "params.h"
#include "perf.h"
#include "profiler.h"
#include "tensor_shape.h"

// Globals
std::uint32_t unp_cfg_context          = 0;
std::uint32_t pack_sync_tile_dst_ptr   = 0;
std::uint32_t math_sync_tile_dst_index = 0;

#ifdef LLK_TRISC_UNPACK

#include "experimental/llk_unpack_AB_sub_bcast_col_custom.h"
#include "llk_unpack_common.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif

#ifndef SPEED_OF_LIGHT
    const std::uint32_t loop_factor = params.LOOP_FACTOR;
#else
    constexpr std::uint32_t loop_factor = LOOP_FACTOR;
#endif
#ifdef LLK_PROFILER
    constexpr std::uint32_t num_blocks         = 1;
    constexpr std::uint32_t num_tiles_in_block = CT_DIM;
    constexpr std::uint32_t test_face_r_dim    = FACE_R_DIM;
    constexpr std::uint32_t test_num_faces     = TILE_NUM_FACES;
#else
    const std::uint32_t num_blocks         = params.NUM_BLOCKS;
    const std::uint32_t num_tiles_in_block = params.NUM_TILES_IN_BLOCK;
    const std::uint32_t test_face_r_dim    = params.TEST_FACE_R_DIM;
    const std::uint32_t test_num_faces     = params.num_faces;
#endif
    const ckernel::TensorShape tensor_shape = ckernel::tensor_shape_from_num_faces(test_face_r_dim, test_num_faces);

    {
        START_PERF_MEASURE("INIT")
        _llk_unpack_hw_configure_<is_fp32_dest_acc_en>(
            formats.unpack_A_src,
            formats.unpack_B_src,
            formats.unpack_A_dst,
            formats.unpack_B_dst,
            test_face_r_dim,
            test_face_r_dim,
            test_num_faces,
            test_num_faces);
#ifdef LLK_PROFILER
        _llk_unpack_AB_sub_bcast_col_init_custom_(tensor_shape);
#endif
        PROFILER_SYNC();
    }
    {
        START_PERF_MEASURE("TILE_LOOP")
        if constexpr (PERF_RUN_TYPE == PerfRunType::PACK_ISOLATE)
        {
            return;
        }
        else if constexpr (PERF_RUN_TYPE == PerfRunType::MATH_ISOLATE)
        {
            for (std::uint32_t loop = 0; loop < loop_factor; ++loop)
            {
                for (std::uint32_t block = 0; block < num_blocks; ++block)
                {
                    _perf_unpack_set_valid(ckernel::SrcB);
                    for (std::uint32_t tile = 0; tile < num_tiles_in_block; ++tile)
                    {
                        _perf_unpack_set_valid(ckernel::SrcA);
                    }
                }
            }
        }
        else
        {
#ifdef LLK_PROFILER
            // Preserve the deleted perf kernel's flat loop: an extra one-trip
            // block loop changes the Wormhole unpack ELF and steady-state cycles.
            for (std::uint32_t loop = 0; loop < loop_factor; ++loop)
            {
                _llk_unpack_AB_sub_bcast_col_custom_(PERF_ADDRESS(PERF_INPUT_A, 0), PERF_ADDRESS(PERF_INPUT_B, 0), num_tiles_in_block);
            }
#else
            for (std::uint32_t block = 0; block < num_blocks; ++block)
            {
                _llk_unpack_AB_sub_bcast_col_init_custom_(tensor_shape);
                _llk_unpack_AB_sub_bcast_col_custom_(
                    L1_ADDRESS(params.buffer_A[block * num_tiles_in_block]), L1_ADDRESS(params.buffer_B[0]), num_tiles_in_block);
            }
#endif
        }
        PROFILER_SYNC();
    }
}

#endif

#ifdef LLK_TRISC_MATH

#include "experimental/llk_math_eltwise_binary_custom.h"
#include "llk_math_common.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif

#ifndef SPEED_OF_LIGHT
    const std::uint32_t loop_factor = params.LOOP_FACTOR;
#else
    constexpr std::uint32_t loop_factor = LOOP_FACTOR;
#endif
#ifdef LLK_PROFILER
    constexpr std::uint32_t num_blocks         = 1;
    constexpr std::uint32_t num_tiles_in_block = CT_DIM;
    constexpr std::uint32_t test_face_r_dim    = FACE_R_DIM;
    constexpr std::uint32_t test_num_faces     = TILE_NUM_FACES;
#else
    const std::uint32_t num_blocks         = params.NUM_BLOCKS;
    const std::uint32_t num_tiles_in_block = params.NUM_TILES_IN_BLOCK;
    const std::uint32_t test_face_r_dim    = params.TEST_FACE_R_DIM;
    const std::uint32_t test_num_faces     = params.num_faces;
#endif
    const ckernel::TensorShape tensor_shape = ckernel::tensor_shape_from_num_faces(test_face_r_dim, test_num_faces);

    {
        START_PERF_MEASURE("INIT")
        _llk_math_pack_sync_init_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
        _llk_math_hw_configure_<is_fp32_dest_acc_en>(formats.math, formats.math);
#ifdef LLK_PROFILER
#ifdef ARCH_BLACKHOLE
        _llk_math_eltwise_binary_init_custom_<ELTWISE_BINARY_OP, BROADCAST_TYPE, MATH_FIDELITY>(test_num_faces);
#else
        _llk_math_eltwise_binary_init_custom_<ELTWISE_BINARY_OP, BROADCAST_TYPE>(test_num_faces);
#endif
#else
        if constexpr (PRESERVE_SRC_ZERO_FLAG)
        {
            // A preceding copy_init leaves the Src zero flag at keep. The custom
            // init must restore the operand-driven state so denormal SrcB flushes.
            math::_configure_preserve_zero_flag_state_();
        }
#endif
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
            for (std::uint32_t loop = 0; loop < loop_factor; ++loop)
            {
                for (std::uint32_t block = 0; block < num_blocks; ++block)
                {
                    _perf_math_clear_valid(ckernel::SrcB);
                    for (std::uint32_t tile = 0; tile < num_tiles_in_block; ++tile)
                    {
                        _perf_math_clear_valid(ckernel::SrcA);
                    }
                }
            }
        }
        else
        {
#ifdef LLK_PROFILER
            for (std::uint32_t loop = 0; loop < loop_factor; ++loop)
            {
                if constexpr (PERF_RUN_TYPE == PerfRunType::L1_TO_L1)
                {
                    _llk_math_wait_for_dest_available_<DstSync::SyncHalf>();
                }
#ifdef ARCH_BLACKHOLE
                _llk_math_bcast_cols_reuse_custom_<ELTWISE_BINARY_OP, MATH_FIDELITY>(num_tiles_in_block, tensor_shape);
#else
                _llk_math_sub_bcast_cols_reuse_custom_(num_tiles_in_block, tensor_shape);
#endif
                if constexpr (PERF_RUN_TYPE == PerfRunType::L1_TO_L1)
                {
                    _llk_math_dest_section_done_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
                }
            }
#else
            for (std::uint32_t block = 0; block < num_blocks; ++block)
            {
                _llk_math_wait_for_dest_available_<DstSync::SyncHalf>();
#ifdef ARCH_BLACKHOLE
                _llk_math_eltwise_binary_init_custom_<ELTWISE_BINARY_OP, BROADCAST_TYPE, MATH_FIDELITY>(test_num_faces);
                _llk_math_bcast_cols_reuse_custom_<ELTWISE_BINARY_OP, MATH_FIDELITY>(num_tiles_in_block, tensor_shape);
#else
                _llk_math_eltwise_binary_init_custom_<ELTWISE_BINARY_OP, BROADCAST_TYPE>(test_num_faces);
                _llk_math_sub_bcast_cols_reuse_custom_(num_tiles_in_block, tensor_shape);
#endif
                _llk_math_dest_section_done_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
            }
#endif
        }
        PROFILER_SYNC();
    }
}

#endif

#ifdef LLK_TRISC_PACK

#include "llk_lib_pack_wrappers.h"
#include "llk_pack.h"
#include "llk_pack_common.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif

#ifdef LLK_PROFILER
#ifndef SPEED_OF_LIGHT
    const std::uint32_t loop_factor = params.LOOP_FACTOR;
#else
    constexpr std::uint32_t loop_factor = LOOP_FACTOR;
#endif
    constexpr std::uint32_t num_tiles_in_block = CT_DIM;
    constexpr std::uint32_t test_face_r_dim    = FACE_R_DIM;
    constexpr std::uint32_t test_num_faces     = TILE_NUM_FACES;
#else
    const std::uint32_t num_blocks         = params.NUM_BLOCKS;
    const std::uint32_t num_tiles_in_block = params.NUM_TILES_IN_BLOCK;
    const std::uint32_t test_face_r_dim    = params.TEST_FACE_R_DIM;
    const std::uint32_t test_num_faces     = params.num_faces;
#endif

    {
        START_PERF_MEASURE("INIT")
        _llk_pack_hw_configure_wrapper_<is_fp32_dest_acc_en, PackMode::Default>(
            formats.pack_src, formats.pack_dst, test_face_r_dim * FACE_C_DIM * test_num_faces /* tile_size */, test_face_r_dim, TILE_C_DIM, test_num_faces);
        _llk_pack_init_wrapper_<PackMode::Default, false /* zero_output */>(formats.pack_dst, test_face_r_dim, TILE_C_DIM, test_num_faces);
        _llk_pack_dest_init_wrapper_<DstSync::SyncHalf, is_fp32_dest_acc_en, PackMode::Default>(test_face_r_dim);
        PROFILER_SYNC();
    }
    {
        START_PERF_MEASURE("TILE_LOOP")
        if constexpr (PERF_RUN_TYPE == PerfRunType::UNPACK_ISOLATE || PERF_RUN_TYPE == PerfRunType::MATH_ISOLATE)
        {
            return;
        }
        else
        {
#ifdef LLK_PROFILER
            for (std::uint32_t loop = 0; loop < loop_factor; ++loop)
            {
                if constexpr (PERF_RUN_TYPE == PerfRunType::L1_TO_L1)
                {
                    _llk_packer_wait_for_math_done_();
                }
                for (std::uint32_t tile = 0; tile < num_tiles_in_block; ++tile)
                {
                    _llk_pack_<DstSync::SyncHalf, is_fp32_dest_acc_en, ckernel::PackMode::Default>(tile, PERF_ADDRESS(PERF_OUTPUT, tile));
                }
                if constexpr (PERF_RUN_TYPE == PerfRunType::L1_TO_L1)
                {
                    _llk_pack_dest_section_done_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
                }
            }
#else
            for (std::uint32_t block = 0; block < num_blocks; ++block)
            {
                _llk_packer_wait_for_math_done_();
                for (std::uint32_t tile = 0; tile < num_tiles_in_block; ++tile)
                {
                    const std::uint32_t result_tile = block * num_tiles_in_block + tile;
                    _llk_pack_<DstSync::SyncHalf, is_fp32_dest_acc_en, ckernel::PackMode::Default>(tile, L1_ADDRESS(params.buffer_Res[result_tile]));
                }
                _llk_pack_dest_section_done_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
            }
#endif
        }
        PROFILER_SYNC();
    }
}

#endif
