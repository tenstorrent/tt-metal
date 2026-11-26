// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Test for eltwise binary operations with reuse_dest on Quasar.

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

// Globals
std::uint32_t unp_cfg_context          = 0;
std::uint32_t pack_sync_tile_dst_ptr   = 0;
std::uint32_t math_sync_tile_dst_index = 0;

#ifdef LLK_TRISC_UNPACK

#include "ckernel_buf_desc.h"
#include "llk_unpack_common.h"
#include "llk_unpack_unary_operand.h"
#include "params.h"

// Out of line and after run_kernel, so adding the zone leaves run_kernel's code unchanged.
__attribute__((noipa, section(".text.uninit_zone"))) void uninit_zone();

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
#ifndef SPEED_OF_LIGHT
    const std::uint32_t LOOP_FACTOR     = params.LOOP_FACTOR;
    const std::uint32_t INPUT_TILE_CNT  = params.INPUT_TILE_CNT;
    const std::uint32_t num_faces       = params.num_faces;
    const std::uint32_t TEST_FACE_R_DIM = params.TEST_FACE_R_DIM;
    const Operand& buffer_A             = params.buffer_A;
    const Operand& buffer_B             = params.buffer_B;
#endif
    constexpr bool TRANSPOSE_EN             = false;
    constexpr bool IS_32B_DEST_EN           = false;
    // The reuse MOP picks the engine itself; llk_unpack_A_init passes UNP_A and programs the BFD that MOP reads.
    constexpr std::uint8_t bfd_phase2 =
        (REUSE_DEST_TYPE == EltwiseBinaryReuseDestType::DEST_TO_SRCA) ? ckernel::trisc::BFD_ID_UNP1 : ckernel::trisc::BFD_ID_UNP0;
    const ckernel::TensorShape tensor_shape = ckernel::tensor_shape_from_num_faces(TEST_FACE_R_DIM, num_faces);

    // compute_kernel_hw_startup work: production runs it before the op's init, so it stays out of INIT.
    _llk_unpack_configure_binary_<p_unpacr::UNP_A, p_unpacr::UNP_B>(
        static_cast<DataFormat>(formats.unpack_A_dst), static_cast<DataFormat>(formats.unpack_B_dst));
    {
        START_PERF_MEASURE("INIT")
        if constexpr (PERF_RUN_TYPE != PerfRunType::L1_TO_L1)
        {
            set_up_fpu_to_pack_dest_dvalid_chain<dest_dvalid_client::UNPACK>();
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
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; loop++)
            {
                // Phase 1 datacopy consumes SrcA once per tile.
                _perf_unpack_loop_set_valid<true /*set_a*/, false /*set_b*/>(INPUT_TILE_CNT);
                // Phase 2 reuse-dest binary consumes both sources per face.
                _perf_unpack_loop_set_valid<true /*set_a*/, true /*set_b*/>(INPUT_TILE_CNT * num_faces);
            }
        }
        else
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; loop++)
            {
                // Phase 1 and phase 2 share unpack's bank0 instruction
                // buffer, so select the phase 1 MOP before executing it.
                // Each init programs its BFD first, as llk_unpack_A_init does.
                ckernel::trisc::program_bfd<ckernel::trisc::BFD_ID_UNP0>(tensor_shape, L1_ADDRESS(buffer_A[0]), formats.unpack_A_src);
                _llk_unpack_unary_operand_init_<p_unpacr::UNP_A, TRANSPOSE_EN, IS_32B_DEST_EN>(
                    ckernel::trisc::BFD_ID_UNP0, tensor_shape, 1 /*num_tiles_per_unpack*/);
                for (std::uint32_t i = 0; i < INPUT_TILE_CNT; ++i)
                {
                    _llk_unpack_unary_operand_<p_unpacr::UNP_A>(i, tensor_shape);
                }

                // Select the reuse-dest phase 2 MOP after phase 1 completes.
                if constexpr (bfd_phase2 == ckernel::trisc::BFD_ID_UNP1)
                {
                    ckernel::trisc::program_bfd<ckernel::trisc::BFD_ID_UNP1>(tensor_shape, L1_ADDRESS(buffer_B[0]), formats.unpack_B_src);
                }
                else
                {
                    ckernel::trisc::program_bfd<ckernel::trisc::BFD_ID_UNP0>(tensor_shape, L1_ADDRESS(buffer_A[0]), formats.unpack_A_src);
                }
                _llk_unpack_unary_operand_init_<p_unpacr::UNP_A, TRANSPOSE_EN, IS_32B_DEST_EN, REUSE_DEST_TYPE>(
                    bfd_phase2, tensor_shape, 1 /*num_tiles_per_unpack*/);
                for (std::uint32_t i = 0; i < INPUT_TILE_CNT; ++i)
                {
                    _llk_unpack_unary_operand_<p_unpacr::UNP_A, REUSE_DEST_TYPE>(i, tensor_shape);
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

#include "llk_math_common.h"
#include "llk_math_eltwise_binary.h"
#include "llk_math_eltwise_unary_datacopy.h"
#include "params.h"
#include "tensor_shape.h"

using namespace ckernel;

// Out of line and after run_kernel, so adding the zone leaves run_kernel's code unchanged.
__attribute__((noipa, section(".text.uninit_zone"))) void uninit_zone();

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
#ifndef SPEED_OF_LIGHT
    const std::uint32_t LOOP_FACTOR               = params.LOOP_FACTOR;
    const std::uint32_t INPUT_TILE_CNT            = params.INPUT_TILE_CNT;
    const std::uint32_t num_faces                 = params.num_faces;
    const std::uint32_t TEST_FACE_R_DIM           = params.TEST_FACE_R_DIM;
    const std::uint32_t INPUT_NUM_TILES_IN_BLOCK  = params.INPUT_NUM_TILES_IN_BLOCK;
    const std::uint32_t INPUT_NUM_BLOCKS          = params.INPUT_NUM_BLOCKS;
    const std::uint32_t OUTPUT_NUM_TILES_IN_BLOCK = params.OUTPUT_NUM_TILES_IN_BLOCK;
#endif
    const ckernel::TensorShape tensor_shape = ckernel::tensor_shape_from_num_faces(TEST_FACE_R_DIM, num_faces);
    // compute_kernel_hw_startup work: production runs it before the op's init, so it stays out of INIT.
    if constexpr (PERF_RUN_TYPE == PerfRunType::L1_TO_L1 || PERF_RUN_TYPE == PerfRunType::MATH_ISOLATE)
    {
        _llk_math_pack_sync_init_<dest_sync>();
    }
    DataFormat src_format = static_cast<DataFormat>(formats.math);
    _llk_math_srcAB_hw_configure_<is_fp32_dest_acc_en>(src_format, src_format);
    {
        START_PERF_MEASURE("INIT")
        if constexpr (PERF_RUN_TYPE == PerfRunType::MATH_ISOLATE)
        {
            // Math isolate keeps the FPU→PACK dest-dvalid producer setup.
            set_up_fpu_to_pack_dest_dvalid_chain<dest_dvalid_client::FPU>();
        }

        PROFILER_SYNC();
    }
    {
        START_PERF_MEASURE("TILE_LOOP")
        const int num_total_tiles = INPUT_NUM_TILES_IN_BLOCK * INPUT_NUM_BLOCKS;
        const int tiles_in_block  = OUTPUT_NUM_TILES_IN_BLOCK;
        const int num_tiles_accum = INPUT_NUM_TILES_IN_BLOCK / tiles_in_block;
        const int num_blocks      = INPUT_NUM_BLOCKS;

        if constexpr (PERF_RUN_TYPE == PerfRunType::PACK_ISOLATE)
        {
        }
        else if constexpr (PERF_RUN_TYPE == PerfRunType::UNPACK_ISOLATE)
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; loop++)
            {
                _perf_math_loop_clear_valid<true /*clear_a*/, false /*clear_b*/>(INPUT_TILE_CNT);
                _perf_math_loop_clear_valid<true /*clear_a*/, true /*clear_b*/>(INPUT_TILE_CNT * num_faces);
            }
        }
        else
        {
            // One Dest section per pass: the kernel supports a single output block.
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; loop++)
            {
                if constexpr (PERF_RUN_TYPE == PerfRunType::L1_TO_L1)
                {
                    _llk_math_wait_for_dest_available_();
                }
                // Datacopy and binary share math's bank0 instruction buffer; each init runs its ALU config first.
                const DataFormat math_format = static_cast<DataFormat>(formats.math);
                _configure_alu_cfg_<AluCfgType::DEFAULT, is_fp32_dest_acc_en>(math_format, math_format);
                _llk_math_eltwise_unary_datacopy_init_<DataCopyType::A2D, is_fp32_dest_acc_en>(
                    num_faces * TEST_FACE_R_DIM /*num_rows_per_matrix*/, 1 /*num_matrices*/);
                for (int i = 0; i < num_total_tiles; ++i)
                {
                    _llk_math_eltwise_unary_datacopy_(i);
                }

                const DataFormat dest_src_format = ckernel::math::dest_src_format<is_fp32_dest_acc_en>(math_format);
                _configure_alu_cfg_<AluCfgType::DEFAULT, is_fp32_dest_acc_en>(math_format, math_format);
                _llk_math_eltwise_binary_init_<ELTWISE_BINARY_OP, MATH_FIDELITY, REUSE_DEST_TYPE>(
                    REUSE_DEST_TYPE == EltwiseBinaryReuseDestType::DEST_TO_SRCA ? dest_src_format : math_format,
                    REUSE_DEST_TYPE == EltwiseBinaryReuseDestType::DEST_TO_SRCB ? dest_src_format : math_format,
                    tensor_shape);
                for (int block = 0; block < num_blocks; block++)
                {
                    for (int n = 0; n < num_tiles_accum; n++)
                    {
                        for (int tile = 0; tile < tiles_in_block; tile++)
                        {
                            const int global_tile_idx = block * tiles_in_block + tile;
                            _llk_math_eltwise_binary_<ELTWISE_BINARY_OP, REUSE_DEST_TYPE>(global_tile_idx, tensor_shape);
                        }
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
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
#ifndef SPEED_OF_LIGHT
    const std::uint32_t LOOP_FACTOR               = params.LOOP_FACTOR;
    const std::uint32_t OUTPUT_NUM_TILES_IN_BLOCK = params.OUTPUT_NUM_TILES_IN_BLOCK;
    const std::uint32_t OUTPUT_NUM_BLOCKS         = params.OUTPUT_NUM_BLOCKS;
    const std::uint32_t TEST_FACE_R_DIM           = params.TEST_FACE_R_DIM;
    const std::uint32_t num_faces                 = params.num_faces;
    const Operand& buffer_Res                     = params.buffer_Res;
#endif
    const ckernel::TensorShape tensor_shape = ckernel::tensor_shape_from_num_faces(TEST_FACE_R_DIM, num_faces);

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
        // PACK_ISOLATE packs without a math↔pack handshake.
        // Explicitly clear wait_mask — CFG can persist across run-types in the same session.
        if constexpr (PERF_RUN_TYPE == PerfRunType::PACK_ISOLATE)
        {
            set_up_zero_dest_dvalid_handshake_for_pack();
        }

        PROFILER_SYNC();
    }
    {
        START_PERF_MEASURE("TILE_LOOP")
        const int output_tiles_in_block = OUTPUT_NUM_TILES_IN_BLOCK;
        const int output_num_blocks     = OUTPUT_NUM_BLOCKS;

        if constexpr (PERF_RUN_TYPE == PerfRunType::MATH_ISOLATE || PERF_RUN_TYPE == PerfRunType::UNPACK_ISOLATE)
        {
        }
        else
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; loop++)
            {
                if constexpr (PERF_RUN_TYPE == PerfRunType::L1_TO_L1)
                {
                    _llk_packer_wait_for_math_done_();
                }
                for (int block = 0; block < output_num_blocks; block++)
                {
                    for (int tile = 0; tile < output_tiles_in_block; tile++)
                    {
                        int res_tile_idx = (block * output_tiles_in_block) + tile;
                        _llk_pack_(res_tile_idx, res_tile_idx, tensor_shape);
                    }
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
        PROFILER_SYNC();
    }
    uninit_zone();
}

void uninit_zone()
{
    ZONE_SCOPED("UNINIT")
}

#endif
