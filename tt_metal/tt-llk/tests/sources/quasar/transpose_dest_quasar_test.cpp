// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
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
    const std::uint32_t LOOP_FACTOR    = params.LOOP_FACTOR;
    const std::uint32_t tiles_in_block = params.OUTPUT_NUM_TILES_IN_BLOCK;
    const std::uint32_t num_blocks     = static_cast<std::uint32_t>(params.INPUT_NUM_BLOCKS);
    const Operand& buffer_A            = params.buffer_A;
    const Operand& buffer_B            = params.buffer_B;
#endif

    {
        ZONE_SCOPED("INIT")
        if constexpr (unpack_to_dest)
        {
            if constexpr (PERF_RUN_TYPE == PerfRunType::L1_TO_L1)
            {
                set_up_unpack_to_fpu_to_pack_dest_dvalid_chain<dest_dvalid_client::UNPACK>();
            }
            else
            {
                // L1_TO_L1 leaves UNPACK waiting for DEST ownership. Other run
                // types have no unpack-to-dest consumer pulse.
                set_up_zero_dest_dvalid_handshake_for_unpack();
            }
        }
        else
        {
            set_up_fpu_to_pack_dest_dvalid_chain<dest_dvalid_client::UNPACK>();
        }

        unsigned l1_addr_16B;
        if constexpr (UNPACKER_ENGINE_SEL == p_unpacr::UNP_B)
        {
            l1_addr_16B = L1_ADDRESS(buffer_B[0]);
        }
        else
        {
            l1_addr_16B = L1_ADDRESS(buffer_A[0]);
        }

        ckernel::trisc::bfd_alloc_and_program<ckernel::trisc::BfdResource::Unp0>(TENSOR_SHAPE_FROM_PARAMS(params), l1_addr_16B, formats.unpack_A_src);
        if constexpr (is_fp32_dest_acc_en && !unpack_to_dest)
        {
            // If Dest is in 32bit mode and operation is Mov2D, we need both SrcA/B fmts to be configured since Mov2D will be implemented via ELWADD
            _llk_unpack_configure_binary_<p_unpacr::UNP_A, p_unpacr::UNP_B>(
                static_cast<DataFormat>(formats.unpack_A_dst), static_cast<DataFormat>(formats.unpack_A_dst));
        }
        else
        {
            _llk_unpack_configure_unary_<UNPACKER_ENGINE_SEL>(static_cast<DataFormat>(formats.unpack_A_dst));
        }

        _llk_unpack_unary_operand_init_<UNPACKER_ENGINE_SEL, false /*transpose*/, is_fp32_dest_acc_en>(
            ckernel::trisc::bfd_current<ckernel::trisc::BfdResource::Unp0>(), ckernel::DEFAULT_TENSOR_SHAPE, tiles_in_block);
        PROFILER_SYNC();
    }
    {
        ZONE_SCOPED("TILE_LOOP")
        if constexpr (PERF_RUN_TYPE == PerfRunType::PACK_ISOLATE)
        {
        }
        else if constexpr (PERF_RUN_TYPE == PerfRunType::MATH_ISOLATE)
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; loop++)
            {
                // Match real unpack's per-block ordering. Batching all SrcA
                // blocks before SrcB deadlocks once the test spans multiple
                // DEST blocks because math consumes SrcA then SrcB per block.
                for (std::uint32_t block = 0; block < num_blocks; block++)
                {
                    if constexpr (!unpack_to_dest)
                    {
                        // FP32 datacopy uses ELWADD, so SrcA and SrcB must become
                        // valid together before the separate transpose SrcB batch.
                        _perf_unpack_loop_set_valid<true /*set_a*/, is_fp32_dest_acc_en /*set_b*/>(tiles_in_block);
                    }
                    _perf_unpack_loop_set_valid<false /*set_a*/, true /*set_b*/>(tiles_in_block);
                }
            }
        }
        else
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; loop++)
            {
                for (std::uint32_t block = 0; block < num_blocks; block++)
                {
                    _llk_unpack_unary_operand_<UNPACKER_ENGINE_SEL>(block * tiles_in_block /*l1_tile_idx*/, ckernel::DEFAULT_TENSOR_SHAPE);

                    if constexpr (unpack_to_dest)
                    {
                        if constexpr (PERF_RUN_TYPE == PerfRunType::L1_TO_L1)
                        {
                            _llk_unpack_dest_dvalid_section_done_<dest_sync>();
                        }
                    }

                    // After datacopy consumes SrcA and clears its dvalid, provide
                    // dummy SrcA+SrcB dvalid so transpose dest can use srcA/B.
                    for (std::uint32_t i = 0; i < tiles_in_block; ++i)
                    {
                        _llk_unpack_set_srcB_dummy_valid_();
                    }
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
#include "llk_math_transpose_dest.h"
#include "params.h"

using namespace ckernel;

template <bool set_dvalid>
inline void run_datacopy_transpose_loop(
    const DataFormat math_format,
    const std::uint32_t loop_factor,
    const std::uint32_t num_blocks,
    const std::uint32_t tiles_in_block,
    const std::uint32_t num_faces,
    const std::uint32_t face_r_dim,
    const int dst_index)
{
    for (std::uint32_t loop = 0; loop < loop_factor; loop++)
    {
        for (std::uint32_t block = 0; block < num_blocks; block++)
        {
            if constexpr (!unpack_to_dest)
            {
                // Datacopy and transpose both use bank0's instruction buffer, so
                // each operation must program its MOP immediately before execution.
                _configure_default_alu_data_format_state_<IMPLIED_MATH_FORMAT, is_fp32_dest_acc_en>(math_format, math_format);
                _llk_math_eltwise_unary_datacopy_init_<DATA_COPY_TYPE, is_fp32_dest_acc_en>(num_faces * face_r_dim /*num_rows_per_matrix*/, 1 /*num_matrices*/);

                for (std::uint32_t i = 0; i < tiles_in_block; ++i)
                {
                    _llk_math_eltwise_unary_datacopy_(dst_index + i);
                }
            }

            // Int32/Float32 transpose dest requires non-default SrcA/SrcB format
            // settings and disables implied math format.
            _configure_mov_ops_explicit_alu_data_format_state_<is_fp32_dest_acc_en>(math_format, math_format);
            _llk_math_transpose_dest_init_<MATH_TRANSPOSE_FACES, is_fp32_dest_acc_en>();
            for (std::uint32_t i = 0; i < tiles_in_block; ++i)
            {
                _llk_math_transpose_dest_(dst_index + i);
            }

            if constexpr (set_dvalid)
            {
                _llk_math_set_dvalid_<p_cleardvalid::FPU, dest_sync>();
            }
        }
    }
}

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
#ifndef SPEED_OF_LIGHT
    const std::uint32_t LOOP_FACTOR     = params.LOOP_FACTOR;
    const std::uint32_t tiles_in_block  = params.OUTPUT_NUM_TILES_IN_BLOCK;
    const std::uint32_t num_blocks      = params.INPUT_NUM_BLOCKS;
    const std::uint32_t num_faces       = params.num_faces;
    const std::uint32_t TEST_FACE_R_DIM = params.TEST_FACE_R_DIM;
    const int DST_INDEX                 = params.DST_INDEX;
#endif
    const DataFormat math_format = static_cast<DataFormat>(formats.math);
    {
        ZONE_SCOPED("INIT")
        if constexpr (PERF_RUN_TYPE == PerfRunType::L1_TO_L1)
        {
            if constexpr (!unpack_to_dest)
            {
                set_up_fpu_to_pack_dest_dvalid_chain<dest_dvalid_client::FPU>();
            }
            else
            {
                set_up_unpack_to_fpu_to_pack_dest_dvalid_chain<dest_dvalid_client::FPU>();
            }
        }
        else if constexpr (PERF_RUN_TYPE == PerfRunType::MATH_ISOLATE)
        {
            // Math isolate has no destination producer before FPU, so make FPU
            // the producer and restore immediate ownership of the destination.
            set_up_fpu_to_pack_dest_dvalid_chain<dest_dvalid_client::FPU>();
        }

        configure_math_hardware_for_float32_int32_or_default<IMPLIED_MATH_FORMAT, is_fp32_dest_acc_en>(math_format, static_cast<DataFormat>(formats.pack_src));
        PROFILER_SYNC();
    }
    {
        ZONE_SCOPED("TILE_LOOP")
        if constexpr (PERF_RUN_TYPE == PerfRunType::PACK_ISOLATE)
        {
        }
        else if constexpr (PERF_RUN_TYPE == PerfRunType::UNPACK_ISOLATE || PERF_RUN_TYPE == PerfRunType::L1_CONGESTION)
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; loop++)
            {
                // Real unpack emits SrcA then dummy SrcB for each block.
                // Clearing all SrcA blocks first waits on data that unpack
                // cannot produce until the preceding SrcB block is consumed.
                for (std::uint32_t block = 0; block < num_blocks; block++)
                {
                    if constexpr (!unpack_to_dest)
                    {
                        // Match real 32-bit unpack: datacopy consumes and clears
                        // paired SrcA/SrcB before transpose consumes dummy SrcB.
                        _perf_math_loop_clear_valid<true /*clear_a*/, is_fp32_dest_acc_en /*clear_b*/>(tiles_in_block);
                    }
                    _perf_math_loop_clear_valid<false /*clear_a*/, true /*clear_b*/>(tiles_in_block);
                }
            }
        }
        else if constexpr (PERF_RUN_TYPE == PerfRunType::MATH_ISOLATE)
        {
            run_datacopy_transpose_loop<false /*set_dvalid*/>(math_format, LOOP_FACTOR, num_blocks, tiles_in_block, num_faces, TEST_FACE_R_DIM, DST_INDEX);
        }
        else
        {
            run_datacopy_transpose_loop<true /*set_dvalid*/>(math_format, LOOP_FACTOR, num_blocks, tiles_in_block, num_faces, TEST_FACE_R_DIM, DST_INDEX);
        }
        PROFILER_SYNC();
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
    const std::uint32_t LOOP_FACTOR           = params.LOOP_FACTOR;
    const std::uint32_t output_num_blocks     = params.OUTPUT_NUM_BLOCKS;
    const std::uint32_t output_tiles_in_block = params.OUTPUT_NUM_TILES_IN_BLOCK;
    const int DST_INDEX                       = params.DST_INDEX;
    const Operand& buffer_Res                 = params.buffer_Res;
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
                set_up_unpack_to_fpu_to_pack_dest_dvalid_chain<dest_dvalid_client::PACK>();
            }
            else
            {
                set_up_fpu_to_pack_dest_dvalid_chain<dest_dvalid_client::PACK>();
            }
        }

        ckernel::trisc::bfd_alloc_and_program<ckernel::trisc::BfdResource::Pack0>(
            TENSOR_SHAPE_FROM_PARAMS(params), L1_ADDRESS(buffer_Res[0]), formats.pack_dst);
        _llk_pack_hw_configure_<p_pacr::PACK0, is_fp32_dest_acc_en>(static_cast<DataFormat>(formats.pack_src), ckernel::ReluConfig::none());
        _llk_pack_init_(ckernel::trisc::bfd_current<ckernel::trisc::BfdResource::Pack0>(), ckernel::DEFAULT_TENSOR_SHAPE, output_tiles_in_block);
        PROFILER_SYNC();
    }
    {
        ZONE_SCOPED("TILE_LOOP")
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
                    _llk_pack_(DST_INDEX, block * output_tiles_in_block /*start_l1_tile_idx*/, ckernel::DEFAULT_TENSOR_SHAPE);
                }
            }
        }
        else
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; loop++)
            {
                for (std::uint32_t block = 0; block < output_num_blocks; block++)
                {
                    _llk_pack_(DST_INDEX, block * output_tiles_in_block /*start_l1_tile_idx*/, ckernel::DEFAULT_TENSOR_SHAPE);
                    _llk_pack_dest_dvalid_section_done_<dest_sync, is_fp32_dest_acc_en>();
                }
            }
        }
        PROFILER_SYNC();
    }
}
#endif
