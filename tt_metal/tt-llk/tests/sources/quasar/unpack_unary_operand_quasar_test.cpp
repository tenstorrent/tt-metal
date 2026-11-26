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

// Out of line and after run_kernel, so adding the zone leaves run_kernel's code unchanged.
__attribute__((noipa, section(".text.uninit_zone"))) void uninit_zone();

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
    const Operand& buffer_B                = params.buffer_B;
#endif
    const std::uint32_t num_blocks     = static_cast<std::uint32_t>(NUM_BLOCKS);
    const std::uint32_t tiles_in_block = static_cast<std::uint32_t>(NUM_TILES_IN_BLOCK);

    // compute_kernel_hw_startup work: production runs it before the op's init, so it stays out of INIT.
    if constexpr (is_fp32_dest_acc_en && !unpack_to_dest)
    {
        _llk_unpack_configure_binary_<p_unpacr::UNP_A, p_unpacr::UNP_B>(
            static_cast<DataFormat>(formats.unpack_A_dst), static_cast<DataFormat>(formats.unpack_A_dst));
    }
    if constexpr (unpack_to_dest)
    {
        // The placers carry no synchronization: in L1_TO_L1 a Dest loan from math brackets them and sets the
        // section base, so every section starts at DEST tile 0, as the bare UNP_DEST MOP had it.
        _llk_unpack_dest_init_();
    }
    {
        START_PERF_MEASURE("INIT")
        if constexpr (PERF_RUN_TYPE != PerfRunType::L1_TO_L1)
        {
            if constexpr (unpack_to_dest)
            {
                // Isolates have no unpack-to-DEST consumer pulse and must not
                // inherit a handshake from a preceding run type.
                set_up_zero_dest_dvalid_handshake_for_unpack();
            }
            else
            {
                set_up_fpu_to_pack_dest_dvalid_chain<dest_dvalid_client::UNPACK>();
            }
        }

        const ckernel::TensorShape tensor_shape_A = TENSOR_SHAPE_FROM_PARAMS(params);
        constexpr bool TRANSPOSE_EN               = UNPACK_TRANSPOSE_FACES && UNPACK_TRANSPOSE_WITHIN_FACE;

        unsigned l1_addr_16B;
        if constexpr (UNPACKER_ENGINE_SEL == p_unpacr::UNP_A || UNPACKER_ENGINE_SEL == p_unpacr::UNP_DEST)
        {
            l1_addr_16B = L1_ADDRESS(buffer_A[0]);
        }
        else if constexpr (UNPACKER_ENGINE_SEL == p_unpacr::UNP_B)
        {
            l1_addr_16B = L1_ADDRESS(buffer_B[0]);
        }

        // Record the descriptor under the engine UNPACKER_ENGINE_SEL actually drives (UNP_B -> UNPACR1/Unp1).
        constexpr auto unp_res = (UNPACKER_ENGINE_SEL == p_unpacr::UNP_B) ? ckernel::trisc::BFD_ID_UNP1 : ckernel::trisc::BFD_ID_UNP0;
        ckernel::trisc::program_bfd<unp_res>(tensor_shape_A, l1_addr_16B, formats.unpack_A_src);
        constexpr std::uint8_t bfd_unpack = unp_res;

        if constexpr (!(is_fp32_dest_acc_en && !unpack_to_dest))
        {
            _llk_unpack_configure_unary_<UNPACKER_ENGINE_SEL>(static_cast<DataFormat>(formats.unpack_A_dst));
        }

        if constexpr (unpack_to_dest)
        {
            _llk_unpack_unary_operand_to_dest_init_(bfd_unpack, tensor_shape_A);
        }
        else
        {
            _llk_unpack_unary_operand_init_<UNPACKER_ENGINE_SEL, TRANSPOSE_EN, is_fp32_dest_acc_en>(bfd_unpack, tensor_shape_A, tiles_in_block);
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
                if constexpr (DATA_COPY_TYPE == DataCopyType::A2D)
                {
                    _perf_unpack_loop_set_valid<true /*set_a*/, false /*set_b*/>(src_handshake_iters);
                }
                else
                {
                    _perf_unpack_loop_set_valid<false /*set_a*/, true /*set_b*/>(src_handshake_iters);
                }
            }
        }
        else
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; loop++)
            {
                for (std::uint32_t block = 0; block < num_blocks; block++)
                {
                    if constexpr (unpack_to_dest)
                    {
                        // One section of tiles_in_block tiles at DEST tile 0, on a loan from math.
                        if constexpr (PERF_RUN_TYPE == PerfRunType::L1_TO_L1 || PERF_RUN_TYPE == PerfRunType::UNPACK_ISOLATE)
                        {
                            _llk_tile_regs_borrow_<DstClient::MATH, DstClient::UNPACK>();
                        }
                        _llk_unpack_unary_operand_to_dest_block_(
                            block * tiles_in_block /*l1_tile_idx*/, 0 /*dst_tile_idx*/, tiles_in_block, 1 /*l1_tile_stride*/, tensor_shape_A);
                        if constexpr (PERF_RUN_TYPE == PerfRunType::L1_TO_L1 || PERF_RUN_TYPE == PerfRunType::UNPACK_ISOLATE)
                        {
                            _llk_tile_regs_return_<DstClient::MATH, DstClient::UNPACK>();
                        }
                    }
                    else
                    {
                        _llk_unpack_unary_operand_<UNPACKER_ENGINE_SEL>(block * tiles_in_block /*l1_tile_idx*/, tensor_shape_A);
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

#include "llk_dest_borrow.h"
#include "llk_math_common.h"
#include "llk_math_eltwise_unary_datacopy.h"
#include "params.h"

using namespace ckernel;

// Out of line and after run_kernel, so adding the zone leaves run_kernel's code unchanged.
__attribute__((noipa, section(".text.uninit_zone"))) void uninit_zone();

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
    if constexpr (
        PERF_RUN_TYPE == PerfRunType::L1_TO_L1 || PERF_RUN_TYPE == PerfRunType::MATH_ISOLATE ||
        (PERF_RUN_TYPE == PerfRunType::UNPACK_ISOLATE && unpack_to_dest))
    {
        _llk_math_pack_sync_init_<dest_sync>();
    }
    if constexpr (!unpack_to_dest)
    {
        _llk_math_srcAB_hw_configure_<is_fp32_dest_acc_en>(static_cast<DataFormat>(formats.math), static_cast<DataFormat>(formats.math));
    }
    {
        START_PERF_MEASURE("INIT")
        if constexpr (!unpack_to_dest)
        {
            // Math isolate keeps the FPU→PACK dest-dvalid producer setup.
            if constexpr (PERF_RUN_TYPE == PerfRunType::MATH_ISOLATE)
            {
                set_up_fpu_to_pack_dest_dvalid_chain<dest_dvalid_client::FPU>();
            }

            _llk_math_eltwise_unary_datacopy_init_<DATA_COPY_TYPE, is_fp32_dest_acc_en>(
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
                if constexpr (DATA_COPY_TYPE == DataCopyType::A2D)
                {
                    _perf_math_loop_clear_valid<true /*clear_a*/, false /*clear_b*/>(src_handshake_iters);
                }
                else
                {
                    _perf_math_loop_clear_valid<false /*clear_a*/, true /*clear_b*/>(src_handshake_iters);
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
        else if constexpr (PERF_RUN_TYPE == PerfRunType::UNPACK_ISOLATE)
        {
            // Stand-in lender: unpack takes each section's DEST bank through the loan, as in production.
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; loop++)
            {
                for (std::uint32_t block = 0; block < num_blocks; block++)
                {
                    _llk_tile_regs_borrow_<DstClient::MATH, DstClient::UNPACK>();
                    _llk_tile_regs_return_<DstClient::MATH, DstClient::UNPACK>();
                    _llk_math_dest_section_done_<dest_sync, is_fp32_dest_acc_en>();
                    // Pack is idle, so math takes MATH_PACK back.
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
    const std::uint32_t LOOP_FACTOR        = params.LOOP_FACTOR;
    const std::uint32_t NUM_BLOCKS         = params.NUM_BLOCKS;
    const std::uint32_t NUM_TILES_IN_BLOCK = params.NUM_TILES_IN_BLOCK;
    const Operand& buffer_Res              = params.buffer_Res;
#endif
    const std::uint32_t num_blocks     = static_cast<std::uint32_t>(NUM_BLOCKS);
    const std::uint32_t tiles_in_block = static_cast<std::uint32_t>(NUM_TILES_IN_BLOCK);

    // compute_kernel_hw_startup work: production runs it before the op's init, so it stays out of INIT.
    if constexpr (PERF_RUN_TYPE == PerfRunType::L1_TO_L1 || PERF_RUN_TYPE == PerfRunType::PACK_ISOLATE)
    {
        _llk_pack_dest_init_<p_pacr::PACK0, dest_sync>();
    }
    const ckernel::TensorShape tensor_shape_A = TENSOR_SHAPE_FROM_PARAMS(params);
    constexpr std::uint8_t bfd_pack           = ckernel::trisc::BFD_ID_PACK0;
    ckernel::trisc::program_bfd<ckernel::trisc::BFD_ID_PACK0>(tensor_shape_A, L1_ADDRESS(buffer_Res[0]), formats.pack_dst);
    _llk_pack_hw_configure_<p_pacr::PACK0, is_fp32_dest_acc_en>(static_cast<DataFormat>(formats.pack_src), ckernel::ReluConfig::none());
    _llk_pack_init_(bfd_pack, tensor_shape_A, tiles_in_block);
    {
        START_PERF_MEASURE("INIT")
        // PACK_ISOLATE has no DEST producer.
        // Explicitly clear wait_mask because CFG persists across run types.
        if constexpr (PERF_RUN_TYPE == PerfRunType::PACK_ISOLATE)
        {
            set_up_zero_dest_dvalid_handshake_for_pack();
        }

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
                    _llk_pack_(0 /*start_math_dest_tile_idx*/, block * tiles_in_block /*start_l1_tile_idx*/, tensor_shape_A);
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
