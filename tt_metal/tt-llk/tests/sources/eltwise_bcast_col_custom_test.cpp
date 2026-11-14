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

// Loop shape. L1_TO_L1 is the functional run repeated LOOP_FACTOR times: every block does the
// per-block init and the custom call. Each isolate run type runs one thread's L1_TO_L1 loop
// unchanged and replaces only the threads it does not measure.
//
// Geometry and buffers are the same in every build; speed-of-light folds the runtime parameters
// into generated constants, and nothing is selected on LLK_PROFILER.

#ifdef LLK_TRISC_UNPACK

#include "experimental/llk_unpack_AB_sub_bcast_col_custom.h"
#include "llk_unpack_common.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif

#ifndef SPEED_OF_LIGHT
    const std::uint32_t loop_factor                 = params.LOOP_FACTOR;
    const std::uint32_t num_blocks                  = params.NUM_BLOCKS;
    const std::uint32_t num_tiles_in_block          = params.NUM_TILES_IN_BLOCK;
    const std::uint32_t test_face_r_dim             = params.TEST_FACE_R_DIM;
    const std::uint32_t test_num_faces              = params.num_faces;
    const Operand& buffer_A                         = params.buffer_A;
    const Operand& buffer_B                         = params.buffer_B;
#else
    constexpr std::uint32_t loop_factor        = LOOP_FACTOR;
    constexpr std::uint32_t num_blocks         = NUM_BLOCKS;
    constexpr std::uint32_t num_tiles_in_block = NUM_TILES_IN_BLOCK;
    constexpr std::uint32_t test_face_r_dim    = TEST_FACE_R_DIM;
    constexpr std::uint32_t test_num_faces     = num_faces;
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
        PROFILER_SYNC();
    }
    {
        START_PERF_MEASURE("TILE_LOOP")
        if constexpr (PERF_RUN_TYPE == PerfRunType::PACK_ISOLATE)
        {
            // Idle: pack runs on its own.
        }
        else if constexpr (PERF_RUN_TYPE == PerfRunType::MATH_ISOLATE)
        {
            // Stand-in for unpack: post the source dvalids math consumes, SrcB once and SrcA per tile.
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
            for (std::uint32_t loop = 0; loop < loop_factor; ++loop)
            {
                for (std::uint32_t block = 0; block < num_blocks; ++block)
                {
                    _llk_unpack_AB_sub_bcast_col_init_custom_(tensor_shape);
                    _llk_unpack_AB_sub_bcast_col_custom_(L1_ADDRESS(buffer_A[block * num_tiles_in_block]), L1_ADDRESS(buffer_B[0]), num_tiles_in_block);
                }
            }
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
    const std::uint32_t loop_factor        = params.LOOP_FACTOR;
    const std::uint32_t num_blocks         = params.NUM_BLOCKS;
    const std::uint32_t num_tiles_in_block = params.NUM_TILES_IN_BLOCK;
    const std::uint32_t test_face_r_dim    = params.TEST_FACE_R_DIM;
    const std::uint32_t test_num_faces     = params.num_faces;
#else
    constexpr std::uint32_t loop_factor        = LOOP_FACTOR;
    constexpr std::uint32_t num_blocks         = NUM_BLOCKS;
    constexpr std::uint32_t num_tiles_in_block = NUM_TILES_IN_BLOCK;
    constexpr std::uint32_t test_face_r_dim    = TEST_FACE_R_DIM;
    constexpr std::uint32_t test_num_faces     = num_faces;
#endif
    const ckernel::TensorShape tensor_shape = ckernel::tensor_shape_from_num_faces(test_face_r_dim, test_num_faces);

    {
        START_PERF_MEASURE("INIT")
        _llk_math_pack_sync_init_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
        _llk_math_hw_configure_<is_fp32_dest_acc_en>(formats.math, formats.math);
        if constexpr (PRESERVE_SRC_ZERO_FLAG)
        {
            // A preceding copy_init leaves the Src zero flag at keep. The custom
            // init must restore the operand-driven state so denormal SrcB flushes.
            math::_configure_preserve_zero_flag_state_();
        }
        PROFILER_SYNC();
    }
    {
        START_PERF_MEASURE("TILE_LOOP")
        if constexpr (PERF_RUN_TYPE == PerfRunType::PACK_ISOLATE)
        {
            // Idle: pack runs on its own.
        }
        else if constexpr (PERF_RUN_TYPE == PerfRunType::UNPACK_ISOLATE)
        {
            // Stand-in for math: clear the source dvalids unpack posts, SrcB once and SrcA per tile.
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
            for (std::uint32_t loop = 0; loop < loop_factor; ++loop)
            {
                for (std::uint32_t block = 0; block < num_blocks; ++block)
                {
                    // MATH_ISOLATE runs the block without the Dest handshake with the idle pack.
                    if constexpr (PERF_RUN_TYPE == PerfRunType::L1_TO_L1)
                    {
                        _llk_math_wait_for_dest_available_<DstSync::SyncHalf>();
                    }
#ifdef ARCH_BLACKHOLE
                    _llk_math_eltwise_binary_init_custom_<ELTWISE_BINARY_OP, BROADCAST_TYPE, MATH_FIDELITY>(test_num_faces);
                    _llk_math_bcast_cols_reuse_custom_<ELTWISE_BINARY_OP, MATH_FIDELITY>(num_tiles_in_block, tensor_shape);
#else
                    _llk_math_eltwise_binary_init_custom_<ELTWISE_BINARY_OP, BROADCAST_TYPE>(test_num_faces);
                    _llk_math_sub_bcast_cols_reuse_custom_(num_tiles_in_block, tensor_shape);
#endif
                    if constexpr (PERF_RUN_TYPE == PerfRunType::L1_TO_L1)
                    {
                        _llk_math_dest_section_done_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
                    }
                }
            }
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

#ifndef SPEED_OF_LIGHT
    const std::uint32_t loop_factor        = params.LOOP_FACTOR;
    const std::uint32_t num_blocks         = params.NUM_BLOCKS;
    const std::uint32_t num_tiles_in_block = params.NUM_TILES_IN_BLOCK;
    const std::uint32_t test_face_r_dim    = params.TEST_FACE_R_DIM;
    const std::uint32_t test_num_faces     = params.num_faces;
    const Operand& buffer_Res              = params.buffer_Res;
#else
    constexpr std::uint32_t loop_factor        = LOOP_FACTOR;
    constexpr std::uint32_t num_blocks         = NUM_BLOCKS;
    constexpr std::uint32_t num_tiles_in_block = NUM_TILES_IN_BLOCK;
    constexpr std::uint32_t test_face_r_dim    = TEST_FACE_R_DIM;
    constexpr std::uint32_t test_num_faces     = num_faces;
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
            // Idle: unpack and math run without pack.
        }
        else
        {
            for (std::uint32_t loop = 0; loop < loop_factor; ++loop)
            {
                for (std::uint32_t block = 0; block < num_blocks; ++block)
                {
                    // PACK_ISOLATE packs without waiting for math; on Wormhole it still releases the Dest section.
                    if constexpr (PERF_RUN_TYPE == PerfRunType::L1_TO_L1)
                    {
                        _llk_packer_wait_for_math_done_();
                    }
                    for (std::uint32_t tile = 0; tile < num_tiles_in_block; ++tile)
                    {
                        const std::uint32_t result_tile = block * num_tiles_in_block + tile;
                        _llk_pack_<DstSync::SyncHalf, is_fp32_dest_acc_en, ckernel::PackMode::Default>(tile, L1_ADDRESS(buffer_Res[result_tile]));
                    }
                    // Wormhole: the PACK_ISOLATE loop also releases the dest section as production's tile_regs_release does
                    if constexpr (PERF_RUN_TYPE == PerfRunType::L1_TO_L1 || PERF_PACK_SECTION_RELEASE)
                    {
                        _llk_pack_dest_section_done_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
                    }
                }
            }
        }
        PROFILER_SYNC();
    }
}

#endif
