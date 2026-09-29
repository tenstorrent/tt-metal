// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <cstdio>

#include "ckernel.h"
#include "llk_defs.h"
#include "llk_memory_checks.h"
#include "perf.h"
#include "profiler.h"
#include "quasar_test_common.h"
#include "sfpu_stub.h"

#ifdef LLK_TRISC_UNPACK

#include "llk_bfd_alloc.h"
#include "llk_math_common.h"
#include "llk_unpack_common.h"
#include "llk_unpack_unary_operand.h"
#include "params.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
#ifndef SPEED_OF_LIGHT
    const std::uint32_t LOOP_FACTOR               = params.LOOP_FACTOR;
    const std::uint32_t TILE_CNT                  = params.TILE_CNT;
    const std::uint32_t OUTPUT_NUM_TILES_IN_BLOCK = params.OUTPUT_NUM_TILES_IN_BLOCK;
    const std::uint32_t OUTPUT_NUM_BLOCKS         = params.OUTPUT_NUM_BLOCKS;
    const Operand& buffer_A                       = params.buffer_A;
#endif
    constexpr std::uint32_t SELECTED_UNPACKER = unpack_to_dest ? p_unpacr::UNP_DEST : p_unpacr::UNP_A;

    {
        ZONE_SCOPED("INIT")
        if constexpr (unpack_to_dest)
        {
            if constexpr (PERF_RUN_TYPE == PerfRunType::L1_TO_L1)
            {
                set_up_unpack_to_pack_dest_dvalid_chain<dest_dvalid_client::UNPACK>();
            }
            else
            {
                // CFG persists across run types, so non-L1_TO_L1 runs must not
                // inherit the unpack-to-dest handshake.
                set_up_zero_dest_dvalid_handshake_for_unpack();
            }
        }

        ckernel::trisc::bfd_alloc_and_program<ckernel::trisc::BfdResource::Unp0>(
            ckernel::tensor_shape_from_num_faces(params.TEST_FACE_R_DIM, params.num_faces), L1_ADDRESS(buffer_A[0]), formats.unpack_A_src);
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
            // ISSUE tt-llk #988: For unpack to dest cannot init the unpacker with 1 tile per unpack, because it will
            // keep writing to dest_idx=0.
            _llk_unpack_unary_operand_init_<SELECTED_UNPACKER, false /*transpose*/, is_fp32_dest_acc_en>(
                ckernel::trisc::bfd_current<ckernel::trisc::BfdResource::Unp0>(), ckernel::DEFAULT_TENSOR_SHAPE, OUTPUT_NUM_TILES_IN_BLOCK /*num_tiles_per_unpack*/);
        }
        else
        {
            _llk_unpack_unary_operand_init_<SELECTED_UNPACKER, false /*transpose*/, is_fp32_dest_acc_en>(
                ckernel::trisc::bfd_current<ckernel::trisc::BfdResource::Unp0>(), ckernel::DEFAULT_TENSOR_SHAPE, 1 /*num_tiles_per_unpack*/);
        }
        PROFILER_SYNC();
    }
    {
        ZONE_SCOPED("TILE_LOOP")

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
        else if constexpr (unpack_to_dest)
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; loop++)
            {
                for (std::uint32_t block = 0; block < static_cast<std::uint32_t>(OUTPUT_NUM_BLOCKS); block++)
                {
                    _llk_unpack_unary_operand_<SELECTED_UNPACKER>(block * OUTPUT_NUM_TILES_IN_BLOCK, ckernel::DEFAULT_TENSOR_SHAPE);
                    if constexpr (PERF_RUN_TYPE == PerfRunType::L1_TO_L1)
                    {
                        _llk_unpack_dest_dvalid_section_done_<dest_sync>();
                    }
                }
            }
        }
        else
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; loop++)
            {
                for (std::uint32_t i = 0; i < TILE_CNT; ++i)
                {
                    _llk_unpack_unary_operand_<SELECTED_UNPACKER>(i, ckernel::DEFAULT_TENSOR_SHAPE);
                }
            }
        }
        PROFILER_SYNC();
    }
}

#endif

#ifdef LLK_TRISC_MATH

#include "llk_math_common.h"
#include "llk_math_eltwise_unary_datacopy.h"
#include "params.h"

using namespace ckernel;
using namespace ckernel::math;

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
#ifndef SPEED_OF_LIGHT
    const std::uint32_t LOOP_FACTOR               = params.LOOP_FACTOR;
    const std::uint32_t num_faces                 = params.num_faces;
    const std::uint32_t TEST_FACE_R_DIM           = params.TEST_FACE_R_DIM;
    const std::uint32_t OUTPUT_NUM_TILES_IN_BLOCK = params.OUTPUT_NUM_TILES_IN_BLOCK;
    const std::uint32_t INPUT_NUM_BLOCKS          = params.INPUT_NUM_BLOCKS;
#endif
    if constexpr (!unpack_to_dest)
    {
        {
            ZONE_SCOPED("INIT")
            // Only end-to-end and math-isolate runs use the FPU→PACK
            // dest-dvalid handshake.
            if constexpr (PERF_RUN_TYPE == PerfRunType::L1_TO_L1 || PERF_RUN_TYPE == PerfRunType::MATH_ISOLATE)
            {
                set_up_fpu_to_pack_dest_dvalid_chain<dest_dvalid_client::FPU>();
            }

            configure_math_hardware_for_float32_int32_or_default<IMPLIED_MATH_FORMAT, is_fp32_dest_acc_en>(
                static_cast<DataFormat>(formats.math), static_cast<DataFormat>(formats.pack_src));

            _llk_math_eltwise_unary_datacopy_init_<DataCopyType::A2D, is_fp32_dest_acc_en>(
                num_faces * TEST_FACE_R_DIM /*num_rows_per_matrix*/, 1 /*num_matrices*/);
            PROFILER_SYNC();
        }
        {
            ZONE_SCOPED("TILE_LOOP")
            const std::uint32_t tiles_in_block = OUTPUT_NUM_TILES_IN_BLOCK;
            const std::uint32_t num_blocks     = static_cast<std::uint32_t>(INPUT_NUM_BLOCKS);

            if constexpr (PERF_RUN_TYPE == PerfRunType::PACK_ISOLATE)
            {
            }
            else if constexpr (PERF_RUN_TYPE == PerfRunType::UNPACK_ISOLATE || PERF_RUN_TYPE == PerfRunType::L1_CONGESTION)
            {
                const std::uint32_t src_handshake_iters = LOOP_FACTOR * num_blocks * tiles_in_block;
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
                        for (std::uint32_t tile = 0; tile < tiles_in_block; tile++)
                        {
                            _llk_math_eltwise_unary_datacopy_(tile);
                        }
                    }
                }
            }
            else
            {
                for (std::uint32_t loop = 0; loop < LOOP_FACTOR; loop++)
                {
                    for (std::uint32_t block = 0; block < num_blocks; block++)
                    {
                        for (std::uint32_t tile = 0; tile < tiles_in_block; tile++)
                        {
                            _llk_math_eltwise_unary_datacopy_(tile);
                        }
                        _llk_math_set_dvalid_<p_cleardvalid::FPU, dest_sync>();
                    }
                }
            }
            PROFILER_SYNC();
        }
    }
}

#endif

#ifdef LLK_TRISC_PACK

#include "llk_bfd_alloc.h"
#include "llk_pack.h"
#include "llk_pack_common.h"
#include "params.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
#ifndef SPEED_OF_LIGHT
    const std::uint32_t LOOP_FACTOR               = params.LOOP_FACTOR;
    const int RELU_CONFIG                         = params.RELU_CONFIG;
    const std::uint32_t OUTPUT_NUM_BLOCKS         = params.OUTPUT_NUM_BLOCKS;
    const std::uint32_t OUTPUT_NUM_TILES_IN_BLOCK = params.OUTPUT_NUM_TILES_IN_BLOCK;
    const Operand& buffer_Res                     = params.buffer_Res;
#endif

    {
        ZONE_SCOPED("INIT")
        // PACK_ISOLATE and L1_CONGESTION pack without a math↔pack handshake.
        // Explicitly clear wait_mask — CFG can persist across run-types in the same session.
        if constexpr (PERF_RUN_TYPE == PerfRunType::PACK_ISOLATE || PERF_RUN_TYPE == PerfRunType::L1_CONGESTION)
        {
            set_up_zero_dest_dvalid_handshake_for_pack();
        }
        else if constexpr (PERF_RUN_TYPE == PerfRunType::L1_TO_L1)
        {
            if constexpr (unpack_to_dest)
            {
                set_up_unpack_to_pack_dest_dvalid_chain<dest_dvalid_client::PACK>();
            }
            else
            {
                set_up_fpu_to_pack_dest_dvalid_chain<dest_dvalid_client::PACK>();
            }
        }

        ckernel::trisc::bfd_alloc_and_program<ckernel::trisc::BfdResource::Pack0>(
            ckernel::tensor_shape_from_num_faces(params.TEST_FACE_R_DIM, params.num_faces), L1_ADDRESS(buffer_Res[0]), formats.pack_dst);
        _llk_pack_hw_configure_<p_pacr::PACK0, is_fp32_dest_acc_en>(static_cast<DataFormat>(formats.pack_src), ckernel::ReluConfig::none());
        const ckernel::ReluConfig relu_config = ckernel::ReluConfig::from_packed(RELU_CONFIG);
        _llk_pack_init_(ckernel::trisc::bfd_current<ckernel::trisc::BfdResource::Pack0>(), ckernel::DEFAULT_TENSOR_SHAPE, 1 /*num_tiles_per_pack*/);
        _llk_pack_relu_config_<p_pacr::PACK0, is_fp32_dest_acc_en>(relu_config);
        PROFILER_SYNC();
    }
    {
        ZONE_SCOPED("TILE_LOOP")
        const std::uint32_t output_num_blocks     = static_cast<std::uint32_t>(OUTPUT_NUM_BLOCKS);
        const std::uint32_t output_tiles_in_block = OUTPUT_NUM_TILES_IN_BLOCK;

        if constexpr (PERF_RUN_TYPE == PerfRunType::MATH_ISOLATE || PERF_RUN_TYPE == PerfRunType::UNPACK_ISOLATE)
        {
        }
        else if constexpr (PERF_RUN_TYPE == PerfRunType::PACK_ISOLATE || PERF_RUN_TYPE == PerfRunType::L1_CONGESTION)
        {
            // No dest-dvalid section_done: WH/BH isolate packs without math handshake.
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; loop++)
            {
                for (std::uint32_t block = 0; block < output_num_blocks; block++)
                {
                    _llk_pack_set_l1_acc_<p_pacr::PACK0>(block == 0 ? false : true /*l1_acc_en*/);
                    for (std::uint32_t tile = 0; tile < output_tiles_in_block; tile++)
                    {
                        _llk_pack_(tile, tile, ckernel::DEFAULT_TENSOR_SHAPE);
                    }
                }
            }
        }
        else
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; loop++)
            {
                for (std::uint32_t block = 0; block < output_num_blocks; block++)
                {
                    _llk_pack_set_l1_acc_<p_pacr::PACK0>(block == 0 ? false : true /*l1_acc_en*/);
                    for (std::uint32_t tile = 0; tile < output_tiles_in_block; tile++)
                    {
                        _llk_pack_(tile, tile, ckernel::DEFAULT_TENSOR_SHAPE);
                    }
                    _llk_pack_dest_dvalid_section_done_<dest_sync, is_fp32_dest_acc_en>();
                }
            }
        }
        _llk_pack_set_l1_acc_<p_pacr::PACK0>(false /*l1_acc_en*/);
        PROFILER_SYNC();
    }
}

#endif
