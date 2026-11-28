// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

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
#include "tensor_shape.h"

// Globals
std::uint32_t unp_cfg_context          = 0;
std::uint32_t pack_sync_tile_dst_ptr   = 0;
std::uint32_t math_sync_tile_dst_index = 0;

#ifdef LLK_TRISC_UNPACK

#include "ckernel_buf_desc.h"
#include "llk_unpack_binary_operands.h"
#include "llk_unpack_common.h"
#include "params.h"

// Out of line and after run_kernel, so adding the zone leaves run_kernel's code unchanged.
__attribute__((noipa, section(".text.uninit_zone"))) void uninit_zone();

void run_kernel(RUNTIME_PARAMETERS params)
{
    const auto tensor_shape = TENSOR_SHAPE_FROM_PARAMS(params);
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
#ifndef SPEED_OF_LIGHT
    const std::uint32_t LOOP_FACTOR               = params.LOOP_FACTOR;
    const std::uint32_t TILE_CNT                  = params.TILE_CNT;
    const std::uint32_t NUM_BLOCKS                = params.NUM_BLOCKS;
    const std::uint32_t INPUT_NUM_TILES_IN_BLOCK  = params.INPUT_NUM_TILES_IN_BLOCK;
    const std::uint32_t OUTPUT_NUM_TILES_IN_BLOCK = params.OUTPUT_NUM_TILES_IN_BLOCK;
    const Operand& buffer_A                       = params.buffer_A;
    const Operand& buffer_B                       = params.buffer_B;
#endif
    // compute_kernel_hw_startup work: production runs it before the op's init, so it stays out of INIT.
    _llk_unpack_configure_binary_<p_unpacr::UNP_A, p_unpacr::UNP_B>(
        static_cast<DataFormat>(formats.unpack_A_dst), static_cast<DataFormat>(formats.unpack_B_dst));
    {
        START_PERF_MEASURE("INIT")
        if constexpr (PERF_RUN_TYPE != PerfRunType::L1_TO_L1)
        {
            set_up_fpu_to_pack_dest_dvalid_chain<dest_dvalid_client::UNPACK>();
        }

        ckernel::trisc::program_bfd<ckernel::trisc::BFD_ID_UNP0>(tensor_shape, L1_ADDRESS(buffer_A[0]), formats.unpack_A_src);
        constexpr std::uint8_t bfd_a = ckernel::trisc::BFD_ID_UNP0;
        ckernel::trisc::program_bfd<ckernel::trisc::BFD_ID_UNP1>(tensor_shape, L1_ADDRESS(buffer_B[0]), formats.unpack_B_src);
        constexpr std::uint8_t bfd_b = ckernel::trisc::BFD_ID_UNP1;
        _llk_unpack_binary_operands_init_(bfd_a, bfd_b, tensor_shape, 1 /*num_tiles*/);
        PROFILER_SYNC();
    }
    {
        START_PERF_MEASURE("TILE_LOOP")
        if constexpr (PERF_RUN_TYPE == PerfRunType::PACK_ISOLATE)
        {
        }
        else if constexpr (PERF_RUN_TYPE == PerfRunType::MATH_ISOLATE)
        {
            _perf_unpack_loop_set_valid<true /*set_a*/, true /*set_b*/>(LOOP_FACTOR * TILE_CNT);
        }
        else
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; loop++)
            {
                // add_tiles / mul_tiles unpack one tile per call, acc_to_dest included.
                const std::uint32_t num_tiles_per_accum = INPUT_NUM_TILES_IN_BLOCK / OUTPUT_NUM_TILES_IN_BLOCK;
                for (std::uint32_t block = 0; block < NUM_BLOCKS; block++)
                {
                    for (std::uint32_t dest_idx = 0; dest_idx < OUTPUT_NUM_TILES_IN_BLOCK; dest_idx++)
                    {
                        for (std::uint32_t tile = 0; tile < num_tiles_per_accum; tile++)
                        {
                            const std::uint32_t i = block * INPUT_NUM_TILES_IN_BLOCK + dest_idx * num_tiles_per_accum + tile;
                            _llk_unpack_binary_operands_(i, i, tensor_shape);
                        }
                    }
                }
            }
        }
        PROFILER_SYNC();
    }
    uninit_zone();
}

void uninit_zone()
{
    ZONE_SCOPED("UNINIT")
    // No uninit: the production op has none on Quasar.
}

#endif

#ifdef LLK_TRISC_MATH

#ifdef FORMAT_INT32
const bool is_int_fpu_en = true;
#else
const bool is_int_fpu_en = false;
#endif

#include "llk_math_common.h"
#include "llk_math_eltwise_binary.h"
#include "params.h"
#include "tensor_shape.h"

using namespace ckernel;

// Out of line and after run_kernel, so adding the zone leaves run_kernel's code unchanged.
__attribute__((noipa, section(".text.uninit_zone"))) void uninit_zone();

void run_kernel(RUNTIME_PARAMETERS params)
{
    const auto tensor_shape = TENSOR_SHAPE_FROM_PARAMS(params);
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
#ifndef SPEED_OF_LIGHT
    const std::uint32_t LOOP_FACTOR               = params.LOOP_FACTOR;
    const std::uint32_t TILE_CNT                  = params.TILE_CNT;
    const std::uint32_t NUM_BLOCKS                = params.NUM_BLOCKS;
    const std::uint32_t INPUT_NUM_TILES_IN_BLOCK  = params.INPUT_NUM_TILES_IN_BLOCK;
    const std::uint32_t OUTPUT_NUM_TILES_IN_BLOCK = params.OUTPUT_NUM_TILES_IN_BLOCK;
#endif
    // compute_kernel_hw_startup work: production runs it before the op's init, so it stays out of INIT.
    if constexpr (PERF_RUN_TYPE == PerfRunType::L1_TO_L1 || PERF_RUN_TYPE == PerfRunType::MATH_ISOLATE)
    {
        _llk_math_pack_sync_init_<dest_sync>();
    }
    DataFormat math_format = static_cast<DataFormat>(formats.math);
    _llk_math_srcAB_hw_configure_<is_fp32_dest_acc_en>(math_format, math_format);
    {
        START_PERF_MEASURE("INIT")
        if constexpr (PERF_RUN_TYPE == PerfRunType::MATH_ISOLATE)
        {
            // Math isolate requires FPU destination ownership.
            set_up_fpu_to_pack_dest_dvalid_chain<dest_dvalid_client::FPU>();
        }

        _llk_math_eltwise_binary_init_<ELTWISE_BINARY_OP, MATH_FIDELITY, REUSE_DEST_TYPE, ENABLE_DIRECT_INDEXING>(
            static_cast<DataFormat>(formats.math), static_cast<DataFormat>(formats.math), tensor_shape, ACC_TO_DEST);

        PROFILER_SYNC();
    }
    {
        START_PERF_MEASURE("TILE_LOOP")
        if constexpr (PERF_RUN_TYPE == PerfRunType::PACK_ISOLATE)
        {
        }
        else if constexpr (PERF_RUN_TYPE == PerfRunType::UNPACK_ISOLATE)
        {
            _perf_math_loop_clear_valid<true /*clear_a*/, true /*clear_b*/>(LOOP_FACTOR * TILE_CNT);
        }
        else
        {
            // acc_to_dest folds num_tiles_per_accum input tiles into one dest tile.
            const std::uint32_t num_tiles_per_accum = INPUT_NUM_TILES_IN_BLOCK / OUTPUT_NUM_TILES_IN_BLOCK;
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; loop++)
            {
                for (std::uint32_t block = 0; block < NUM_BLOCKS; block++)
                {
                    if constexpr (PERF_RUN_TYPE == PerfRunType::L1_TO_L1)
                    {
                        _llk_math_wait_for_dest_available_();
                    }
                    for (std::uint32_t dest_idx = 0; dest_idx < OUTPUT_NUM_TILES_IN_BLOCK; ++dest_idx)
                    {
                        for (std::uint32_t tile = 0; tile < num_tiles_per_accum; ++tile)
                        {
                            _llk_math_eltwise_binary_<ELTWISE_BINARY_OP>(dest_idx, tensor_shape);
                        }
                    }
                    if constexpr (PERF_RUN_TYPE == PerfRunType::L1_TO_L1 || PERF_RUN_TYPE == PerfRunType::MATH_ISOLATE)
                    {
                        _llk_math_dest_section_done_<dest_sync, is_fp32_dest_acc_en>();
                    }
                    if constexpr (PERF_RUN_TYPE == PerfRunType::MATH_ISOLATE)
                    {
                        // Stands in for the idle pack's release, so MATH_PACK never fills up.
                        ckernel::trisc::t6_semaphore_get(ckernel::trisc::semaphore::MATH_PACK);
                    }
                }
            }
        }
        PROFILER_SYNC();
    }
    uninit_zone();
}

void uninit_zone()
{
    ZONE_SCOPED("UNINIT")
}

#endif

#ifdef LLK_TRISC_PACK

#include "ckernel_buf_desc.h"
#include "llk_pack.h"
#include "llk_pack_common.h"
#include "params.h"

// Out of line and after run_kernel, so adding the zone leaves run_kernel's code unchanged.
__attribute__((noipa, section(".text.uninit_zone"))) void uninit_zone();

void run_kernel(RUNTIME_PARAMETERS params)
{
    const auto tensor_shape = TENSOR_SHAPE_FROM_PARAMS(params);
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
#ifndef SPEED_OF_LIGHT
    const std::uint32_t LOOP_FACTOR               = params.LOOP_FACTOR;
    const std::uint32_t NUM_BLOCKS                = params.NUM_BLOCKS;
    const std::uint32_t OUTPUT_NUM_TILES_IN_BLOCK = params.OUTPUT_NUM_TILES_IN_BLOCK;
    const Operand& buffer_Res                     = params.buffer_Res;
#endif

    // compute_kernel_hw_startup work: production runs it before the op's init, so it stays out of INIT.
    if constexpr (PERF_RUN_TYPE == PerfRunType::L1_TO_L1 || PERF_RUN_TYPE == PerfRunType::PACK_ISOLATE)
    {
        _llk_pack_dest_init_<p_pacr::PACK0, dest_sync>();
    }
    constexpr std::uint8_t bfd_pack = ckernel::trisc::BFD_ID_PACK0;
    ckernel::trisc::program_bfd<ckernel::trisc::BFD_ID_PACK0>(tensor_shape, L1_ADDRESS(buffer_Res[0]), formats.pack_dst);
    _llk_pack_hw_configure_<p_pacr::PACK0, is_fp32_dest_acc_en>(static_cast<DataFormat>(formats.pack_src), ckernel::ReluConfig::none());
    _llk_pack_init_(bfd_pack, tensor_shape, 1 /*num_tiles_per_pack*/);
    {
        START_PERF_MEASURE("INIT")
        // Clear a stale FPU-to-PACK wait in modes without a math handshake.
        if constexpr (PERF_RUN_TYPE == PerfRunType::PACK_ISOLATE)
        {
            set_up_zero_dest_dvalid_handshake_for_pack();
        }

        PROFILER_SYNC();
    }
    {
        START_PERF_MEASURE("TILE_LOOP")
        if constexpr (PERF_RUN_TYPE == PerfRunType::MATH_ISOLATE || PERF_RUN_TYPE == PerfRunType::UNPACK_ISOLATE)
        {
        }
        else
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; loop++)
            {
                for (std::uint32_t block = 0; block < NUM_BLOCKS; block++)
                {
                    if constexpr (PERF_RUN_TYPE == PerfRunType::L1_TO_L1)
                    {
                        _llk_packer_wait_for_math_done_();
                    }
                    for (std::uint32_t i = 0; i < OUTPUT_NUM_TILES_IN_BLOCK; ++i)
                    {
                        _llk_pack_(i, block * OUTPUT_NUM_TILES_IN_BLOCK + i, tensor_shape);
                    }
                    if constexpr (PERF_RUN_TYPE == PerfRunType::PACK_ISOLATE)
                    {
                        // Stands in for the idle math's post, so the section done's SEMGET never underflows.
                        ckernel::trisc::t6_semaphore_post(ckernel::trisc::semaphore::MATH_PACK);
                    }
                    if constexpr (PERF_RUN_TYPE == PerfRunType::L1_TO_L1 || PERF_RUN_TYPE == PerfRunType::PACK_ISOLATE)
                    {
                        _llk_pack_dest_semaphore_section_done_<p_pacr::PACK0, dest_sync, is_fp32_dest_acc_en>();
                    }
                }
            }
        }
        PROFILER_SYNC();
    }
    uninit_zone();
}

void uninit_zone()
{
    ZONE_SCOPED("UNINIT")
}

#endif
