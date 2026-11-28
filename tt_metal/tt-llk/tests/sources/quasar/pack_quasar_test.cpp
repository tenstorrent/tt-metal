// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cstdint>
#include <cstdio>

#include "ckernel.h"
#include "counters.h"
#include "llk_defs.h"
#include "llk_memory_checks.h"
#include "perf.h"
#include "profiler.h"
#include "quasar_test_common.h"
#include "sfpu_stub.h"

#ifdef LLK_TRISC_UNPACK

#include "ckernel_buf_desc.h"
#include "llk_dest_borrow.h"
#include "llk_math_common.h"
#include "llk_unpack_common.h"
#include "llk_unpack_unary_operand.h"
#include "llk_unpack_unary_operand_to_dest.h"
#include "params.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
#ifndef SPEED_OF_LIGHT
    const std::uint32_t LOOP_FACTOR        = params.LOOP_FACTOR;
    const std::uint32_t TILE_CNT           = params.TILE_CNT;
    const std::uint32_t NUM_BLOCKS         = params.NUM_BLOCKS;
    const std::uint32_t NUM_TILES_IN_BLOCK = params.NUM_TILES_IN_BLOCK;
    const Operand& buffer_A                = params.buffer_A;
#endif
    const std::uint32_t num_blocks            = static_cast<std::uint32_t>(NUM_BLOCKS);
    const std::uint32_t tiles_in_block        = static_cast<std::uint32_t>(NUM_TILES_IN_BLOCK);
    constexpr std::uint32_t SELECTED_UNPACKER = unpack_to_dest ? p_unpacr::UNP_DEST : p_unpacr::UNP_A;

    {
        START_PERF_MEASURE("INIT")
        if constexpr (unpack_to_dest && PERF_RUN_TYPE != PerfRunType::L1_TO_L1)
        {
            // CFG persists across run types, so isolates must not inherit a dest-dvalid wait mask.
            set_up_zero_dest_dvalid_handshake_for_unpack();
        }

        const ckernel::TensorShape tensor_shape_A = TENSOR_SHAPE_FROM_PARAMS(params);

        ckernel::trisc::program_bfd<ckernel::trisc::BFD_ID_UNP0>(tensor_shape_A, L1_ADDRESS(buffer_A[0]), formats.unpack_A_src);
        constexpr std::uint8_t bfd_unpack = ckernel::trisc::BFD_ID_UNP0;

        if constexpr (is_fp32_dest_acc_en && !unpack_to_dest)
        {
            _llk_unpack_configure_binary_<p_unpacr::UNP_A, p_unpacr::UNP_B>(
                static_cast<DataFormat>(formats.unpack_A_dst), static_cast<DataFormat>(formats.unpack_A_dst));
        }
        else
        {
            _llk_unpack_configure_unary_<SELECTED_UNPACKER>(static_cast<DataFormat>(formats.unpack_A_dst));
        }
        if constexpr (unpack_to_dest)
        {
            // The to-dest placers carry no synchronization: in L1_TO_L1 a Dest loan from math brackets them and
            // hands over the section base, which the placers count in tiles of tensor_shape_A.
            _llk_unpack_dest_init_();
            _llk_unpack_unary_operand_to_dest_init_(bfd_unpack, tensor_shape_A);
        }
        else
        {
            _llk_unpack_unary_operand_init_<SELECTED_UNPACKER, false /*transpose*/, is_fp32_dest_acc_en>(bfd_unpack, tensor_shape_A, tiles_in_block);
        }
        PROFILER_SYNC();
    }
    {
        START_PERF_MEASURE("TILE_LOOP")
        const ckernel::TensorShape tensor_shape_A = TENSOR_SHAPE_FROM_PARAMS(params);

        if constexpr (PERF_RUN_TYPE == PerfRunType::PACK_ISOLATE)
        {
        }
        else if constexpr (PERF_RUN_TYPE == PerfRunType::MATH_ISOLATE)
        {
            if constexpr (!unpack_to_dest)
            {
                const std::uint32_t src_handshake_iters = LOOP_FACTOR * TILE_CNT;
                if constexpr (is_fp32_dest_acc_en)
                {
                    _perf_unpack_loop_set_valid<true /*set_a*/, true /*set_b*/>(src_handshake_iters);
                }
                else
                {
                    _perf_unpack_loop_set_valid<true /*set_a*/, false /*set_b*/>(src_handshake_iters);
                }
            }
        }
        else
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; loop++)
            {
                for (std::uint32_t block = 0; block < num_blocks; block++)
                {
                    if constexpr (unpack_to_dest && PERF_RUN_TYPE == PerfRunType::L1_TO_L1)
                    {
                        _llk_tile_regs_borrow_<DstClient::MATH, DstClient::UNPACK>();
                    }
                    if constexpr (unpack_to_dest)
                    {
                        _llk_unpack_unary_operand_to_dest_block_(
                            block * tiles_in_block /*l1_tile_idx*/, 0 /*dst_tile_idx*/, tiles_in_block, 1 /*l1_tile_stride*/, tensor_shape_A);
                    }
                    else
                    {
                        _llk_unpack_unary_operand_<SELECTED_UNPACKER>(block * tiles_in_block /*l1_tile_idx*/, tensor_shape_A);
                    }
                    if constexpr (unpack_to_dest && PERF_RUN_TYPE == PerfRunType::L1_TO_L1)
                    {
                        _llk_tile_regs_return_<DstClient::MATH, DstClient::UNPACK>();
                    }
                    if constexpr (unpack_to_dest && PERF_RUN_TYPE == PerfRunType::UNPACK_ISOLATE && dest_sync == ckernel::DstSync::SyncHalf)
                    {
                        // Stands in for the loan's bank switch: production unpack takes each section's DEST bank from math.
                        _llk_unpack_dest_section_advance_<is_fp32_dest_acc_en>();
                    }
                }
            }
        }
        PROFILER_SYNC();
    }
}

#endif

#ifdef LLK_TRISC_MATH

#include "llk_dest_borrow.h"
#include "llk_math_common.h"
#include "llk_math_eltwise_unary_datacopy.h"
#include "params.h"

using namespace ckernel;

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
#ifndef SPEED_OF_LIGHT
    const std::uint32_t LOOP_FACTOR        = params.LOOP_FACTOR;
    const std::uint32_t TILE_CNT           = params.TILE_CNT;
    const std::uint32_t NUM_BLOCKS         = params.NUM_BLOCKS;
    const std::uint32_t NUM_TILES_IN_BLOCK = params.NUM_TILES_IN_BLOCK;
    const std::uint32_t num_faces          = params.num_faces;
    const std::uint32_t TEST_FACE_R_DIM    = params.TEST_FACE_R_DIM;
#endif
    const std::uint32_t num_blocks     = static_cast<std::uint32_t>(NUM_BLOCKS);
    const std::uint32_t tiles_in_block = static_cast<std::uint32_t>(NUM_TILES_IN_BLOCK);
    // Every thread opens both zones even when it has no work here: the rendezvous waits for all four.
    // compute_kernel_hw_startup work: production runs it before the op's init, so it stays out of INIT.
    if constexpr (PERF_RUN_TYPE == PerfRunType::L1_TO_L1 || PERF_RUN_TYPE == PerfRunType::MATH_ISOLATE)
    {
        _llk_math_pack_sync_init_<dest_sync>();
    }
    {
        START_PERF_MEASURE("INIT")
        if constexpr (!unpack_to_dest)
        {
            // MATH_ISOLATE keeps the FPU dest-dvalid producer setup; the remaining isolate modes have no FPU consumer.
            if constexpr (PERF_RUN_TYPE == PerfRunType::MATH_ISOLATE)
            {
                set_up_fpu_to_pack_dest_dvalid_chain<dest_dvalid_client::FPU>();
            }

            _llk_math_srcAB_hw_configure_<is_fp32_dest_acc_en>(static_cast<DataFormat>(formats.math), static_cast<DataFormat>(formats.math));

            _llk_math_eltwise_unary_datacopy_init_<DataCopyType::A2D, is_fp32_dest_acc_en>(
                num_faces * TEST_FACE_R_DIM /*num_rows_per_matrix*/, 1 /*num_matrices*/);
        }
        PROFILER_SYNC();
    }
    {
        START_PERF_MEASURE("TILE_LOOP")
        if constexpr (!unpack_to_dest)
        {
            if constexpr (PERF_RUN_TYPE == PerfRunType::PACK_ISOLATE)
            {
            }
            else if constexpr (PERF_RUN_TYPE == PerfRunType::UNPACK_ISOLATE)
            {
                const std::uint32_t src_handshake_iters = LOOP_FACTOR * TILE_CNT;
                if constexpr (is_fp32_dest_acc_en)
                {
                    _perf_math_loop_clear_valid<true /*clear_a*/, true /*clear_b*/>(src_handshake_iters);
                }
                else
                {
                    _perf_math_loop_clear_valid<true /*clear_a*/, false /*clear_b*/>(src_handshake_iters);
                }
            }
            else if constexpr (PERF_RUN_TYPE == PerfRunType::MATH_ISOLATE)
            {
                for (std::uint32_t loop = 0; loop < LOOP_FACTOR; loop++)
                {
                    for (std::uint32_t block = 0; block < num_blocks; block++)
                    {
                        for (std::uint32_t i = 0; i < tiles_in_block; ++i)
                        {
                            _llk_math_eltwise_unary_datacopy_(i);
                        }
                        _llk_math_dest_section_done_<dest_sync, is_fp32_dest_acc_en>();
                        // Stands in for the idle pack's release, so MATH_PACK never fills up.
                        ckernel::trisc::t6_semaphore_get(ckernel::trisc::semaphore::MATH_PACK);
                    }
                }
            }
            else
            {
                for (std::uint32_t loop = 0; loop < LOOP_FACTOR; loop++)
                {
                    for (std::uint32_t block = 0; block < num_blocks; block++)
                    {
                        _llk_math_wait_for_dest_available_();
                        for (std::uint32_t i = 0; i < tiles_in_block; ++i)
                        {
                            _llk_math_eltwise_unary_datacopy_(i);
                        }
                        _llk_math_dest_section_done_<dest_sync, is_fp32_dest_acc_en>();
                    }
                }
            }
        }
        else if constexpr (PERF_RUN_TYPE == PerfRunType::L1_TO_L1)
        {
            // Math owns each Dest section and lends it to the unpack-to-dest copy.
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; loop++)
            {
                for (std::uint32_t block = 0; block < num_blocks; block++)
                {
                    _llk_math_wait_for_dest_available_();
                    _llk_tile_regs_borrow_<DstClient::MATH, DstClient::UNPACK>();
                    _llk_tile_regs_return_<DstClient::MATH, DstClient::UNPACK>();
                    _llk_math_dest_section_done_<dest_sync, is_fp32_dest_acc_en>();
                }
            }
        }
        else if constexpr (PERF_RUN_TYPE == PerfRunType::MATH_ISOLATE)
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; loop++)
            {
                for (std::uint32_t block = 0; block < num_blocks; block++)
                {
                    _llk_math_dest_section_done_<dest_sync, is_fp32_dest_acc_en>();
                    // Stands in for the idle pack's release, so MATH_PACK never fills up.
                    ckernel::trisc::t6_semaphore_get(ckernel::trisc::semaphore::MATH_PACK);
                }
            }
        }
        PROFILER_SYNC();
    }
}

#endif

#ifdef LLK_TRISC_PACK

#include "ckernel_buf_desc.h"
#include "llk_pack.h"
#include "llk_pack_common.h"
#include "params.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
#ifndef SPEED_OF_LIGHT
    const std::uint32_t LOOP_FACTOR        = params.LOOP_FACTOR;
    const std::uint32_t NUM_BLOCKS         = params.NUM_BLOCKS;
    const std::uint32_t NUM_TILES_IN_BLOCK = params.NUM_TILES_IN_BLOCK;
    const int RELU_CONFIG                  = params.RELU_CONFIG;
    const Operand& buffer_Res              = params.buffer_Res;
#endif
    const std::uint32_t num_blocks     = static_cast<std::uint32_t>(NUM_BLOCKS);
    const std::uint32_t tiles_in_block = static_cast<std::uint32_t>(NUM_TILES_IN_BLOCK);

    // compute_kernel_hw_startup work: production runs it before the op's init, so it stays out of INIT.
    if constexpr (PERF_RUN_TYPE == PerfRunType::L1_TO_L1 || PERF_RUN_TYPE == PerfRunType::PACK_ISOLATE)
    {
        _llk_pack_dest_init_<p_pacr::PACK0, dest_sync>();
    }
    {
        START_PERF_MEASURE("INIT")
        // PACK_ISOLATE packs without a math↔pack handshake.
        // Explicitly clear wait_mask — CFG can persist across run-types in the same session.
        if constexpr (PERF_RUN_TYPE == PerfRunType::PACK_ISOLATE)
        {
            set_up_zero_dest_dvalid_handshake_for_pack();
        }

        const ckernel::TensorShape tensor_shape_A = TENSOR_SHAPE_FROM_PARAMS(params);

        ckernel::trisc::program_bfd<ckernel::trisc::BFD_ID_PACK0>(tensor_shape_A, L1_ADDRESS(buffer_Res[0]), formats.pack_dst);
        constexpr std::uint8_t bfd_pack = ckernel::trisc::BFD_ID_PACK0;
        _llk_pack_hw_configure_<p_pacr::PACK0, is_fp32_dest_acc_en>(static_cast<DataFormat>(formats.pack_src), ckernel::ReluConfig::none());
        const ckernel::ReluConfig relu_config = ckernel::ReluConfig::from_packed(RELU_CONFIG);
#ifdef TEST_PACK_BLOCK
        _llk_pack_init_(bfd_pack, tensor_shape_A); // one-tile MOP, as metal's llk_pack_init programs
#else
        _llk_pack_init_(bfd_pack, tensor_shape_A, tiles_in_block);
#endif
        _llk_pack_relu_config_<p_pacr::PACK0, is_fp32_dest_acc_en>(relu_config);
        PROFILER_SYNC();
    }
    {
        START_PERF_MEASURE("TILE_LOOP")
        const ckernel::TensorShape tensor_shape_A = TENSOR_SHAPE_FROM_PARAMS(params);

        if constexpr (PERF_RUN_TYPE == PerfRunType::MATH_ISOLATE || PERF_RUN_TYPE == PerfRunType::UNPACK_ISOLATE)
        {
        }
        else if constexpr (PERF_RUN_TYPE == PerfRunType::PACK_ISOLATE)
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; loop++)
            {
                for (std::uint32_t block = 0; block < num_blocks; block++)
                {
                    // One call packs all tiles_in_block tiles (MOP programmed in _llk_pack_init_); WH/BH call _llk_pack_ per tile.
                    _llk_pack_(0 /*start_math_dest_tile_idx*/, block * tiles_in_block /*start_l1_tile_idx*/, tensor_shape_A);
                    // Stands in for the idle math's post, so the section done's SEMGET never underflows.
                    ckernel::trisc::t6_semaphore_post(ckernel::trisc::semaphore::MATH_PACK);
                    _llk_pack_dest_semaphore_section_done_<p_pacr::PACK0, dest_sync, is_fp32_dest_acc_en>();
                }
            }
        }
        else
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; loop++)
            {
                for (std::uint32_t block = 0; block < num_blocks; block++)
                {
                    _llk_packer_wait_for_math_done_();
#ifdef TEST_PACK_BLOCK
                    const std::uint32_t out = block * tiles_in_block;
                    // Block 0 in growing chunks at nonzero DEST offsets, later blocks in one call.
                    if (block == 0)
                    {
                        for (std::uint32_t start = 0, count = 1; start < tiles_in_block; ++count)
                        {
                            const std::uint32_t ntiles = std::min(count, tiles_in_block - start);
                            _llk_pack_(start, out + start, tensor_shape_A, ntiles);
                            start += ntiles;
                        }
                    }
                    else
                    {
                        _llk_pack_(0 /*start_math_dest_tile_idx*/, out, tensor_shape_A, tiles_in_block);
                    }
#else
                    _llk_pack_(0 /*start_math_dest_tile_idx*/, block * tiles_in_block /*start_l1_tile_idx*/, tensor_shape_A);
#endif
                    _llk_pack_dest_semaphore_section_done_<p_pacr::PACK0, dest_sync, is_fp32_dest_acc_en>();
                }
            }
        }
        PROFILER_SYNC();
    }
}
#endif
