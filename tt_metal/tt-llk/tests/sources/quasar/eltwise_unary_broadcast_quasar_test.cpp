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
#include "llk_unpack_unary_broadcast_operands.h"
#include "params.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
#ifndef SPEED_OF_LIGHT
    const std::uint32_t LOOP_FACTOR    = params.LOOP_FACTOR;
    const std::uint32_t num_blocks     = params.INPUT_NUM_BLOCKS;
    const std::uint32_t tiles_in_block = params.INPUT_NUM_TILES_IN_BLOCK;
    const int num_faces_r_dim_A        = params.num_faces_r_dim_A;
    const int num_faces_c_dim_A        = params.num_faces_c_dim_A;
    const Operand& buffer_B            = params.buffer_B;
#endif
    // Unpack to dest must use the num tiles per unpack parameter in order to unpack multiple tiles per Dest bank
    const std::uint32_t num_tiles_per_unpack = unpack_to_dest ? tiles_in_block : 1;

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
                // CFG persists across run types, so isolates must not inherit
                // the L1_TO_L1 unpack-to-dest handshake.
                set_up_zero_dest_dvalid_handshake_for_unpack();
            }
        }
        else
        {
            set_up_fpu_to_pack_dest_dvalid_chain<dest_dvalid_client::UNPACK>();
        }

        const ckernel::TensorShape tensor_shape = TENSOR_SHAPE_FROM_PARAMS(params);

        // Record the descriptor under the engine UNPACKER_ENGINE_SEL actually drives (UNP_B -> UNPACR1/Unp1).
        constexpr auto unp_res = (UNPACKER_ENGINE_SEL == p_unpacr::UNP_B) ? ckernel::trisc::BfdResource::Unp1 : ckernel::trisc::BfdResource::Unp0;
        ckernel::trisc::bfd_alloc_and_program<unp_res>(tensor_shape, L1_ADDRESS(buffer_B[0]), formats.unpack_A_src);

        _llk_unpack_configure_unary_<UNPACKER_ENGINE_SEL>(static_cast<DataFormat>(formats.unpack_A_dst));
        _llk_unpack_unary_broadcast_operands_init_<UNPACKER_ENGINE_SEL, BROADCAST_TYPE, unpack_to_dest>(
            ckernel::trisc::bfd_current<unp_res>(), num_tiles_per_unpack);

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
                const std::uint32_t dvalids_per_tile =
                    (BROADCAST_TYPE == BroadcastType::SCALAR) ? 1u : static_cast<std::uint32_t>(num_faces_r_dim_A * num_faces_c_dim_A);
                _perf_unpack_loop_set_valid<false /*set_a*/, true /*set_b*/>(LOOP_FACTOR * num_blocks * tiles_in_block * dvalids_per_tile);
            }
            else
            {
                // SrcB dummy dvalid needed for the unpack to dest path
                _perf_unpack_loop_set_valid<false /*set_a*/, true /*set_b*/>(LOOP_FACTOR * num_blocks * tiles_in_block);
            }
        }
        else if constexpr (unpack_to_dest)
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; loop++)
            {
                for (std::uint32_t block = 0; block < num_blocks; block++)
                {
                    _llk_unpack_unary_broadcast_operands_<UNPACKER_ENGINE_SEL, unpack_to_dest>(block * tiles_in_block);
                    if constexpr (PERF_RUN_TYPE == PerfRunType::L1_TO_L1)
                    {
                        _llk_unpack_dest_dvalid_section_done_<dest_sync>();
                    }
                    for (std::uint32_t tile = 0; tile < tiles_in_block; tile++)
                    {
                        _llk_unpack_set_srcB_dummy_valid_();
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
                        const std::uint32_t input_tile_idx = (block * tiles_in_block) + tile;
                        _llk_unpack_unary_broadcast_operands_<UNPACKER_ENGINE_SEL, unpack_to_dest>(input_tile_idx);
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
#include "llk_math_unary_broadcast.h"
#include "params.h"

using namespace ckernel;
using namespace ckernel::math;

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
#ifndef SPEED_OF_LIGHT
    const std::uint32_t LOOP_FACTOR    = params.LOOP_FACTOR;
    const std::uint32_t num_faces      = params.num_faces;
    const std::uint32_t tiles_in_block = params.OUTPUT_NUM_TILES_IN_BLOCK;
    const std::uint32_t num_blocks     = params.INPUT_NUM_BLOCKS;
#endif
    const DataFormat math_format            = static_cast<DataFormat>(formats.math);
    const ckernel::TensorShape tensor_shape = TENSOR_SHAPE_FROM_PARAMS(params);

    {
        ZONE_SCOPED("INIT")
        if constexpr (PERF_RUN_TYPE == PerfRunType::L1_TO_L1)
        {
            if constexpr (unpack_to_dest)
            {
                set_up_unpack_to_fpu_to_pack_dest_dvalid_chain<dest_dvalid_client::FPU>();
            }
            else
            {
                set_up_fpu_to_pack_dest_dvalid_chain<dest_dvalid_client::FPU>();
            }
        }
        else if constexpr (PERF_RUN_TYPE == PerfRunType::MATH_ISOLATE && unpack_to_dest)
        {
            // L1_TO_L1 leaves FPU configured as the middle client in the
            // UNPACK→FPU→PACK chain. Math isolate has no unpack destination
            // pulse, so make FPU the producer and restore immediate ownership
            // of the destination register.
            set_up_fpu_to_pack_dest_dvalid_chain<dest_dvalid_client::FPU>();
        }

        if (is_fp32_dest_acc_en && static_cast<DataFormat>(formats.pack_src) == DataFormat::Int32)
        {
            _llk_math_srcAB_hw_configure_<IMPLIED_MATH_FORMAT, false /*fp32_dest*/, true /*int32_dest*/>(math_format, math_format);
        }
        else
        {
            _llk_math_srcAB_hw_configure_<IMPLIED_MATH_FORMAT, is_fp32_dest_acc_en, false /*int32_dest*/>(math_format, math_format);
        }
        if constexpr (unpack_to_dest)
        {
            _configure_mov_ops_explicit_alu_data_format_state_<is_fp32_dest_acc_en>(math_format, math_format);
        }
        else
        {
            _configure_default_alu_data_format_state_<IMPLIED_MATH_FORMAT, is_fp32_dest_acc_en>(math_format, math_format);
        }
        _llk_math_eltwise_unary_broadcast_init_<BROADCAST_TYPE, unpack_to_dest>(tensor_shape);

        PROFILER_SYNC();
    }
    {
        ZONE_SCOPED("TILE_LOOP")

        if constexpr (PERF_RUN_TYPE == PerfRunType::PACK_ISOLATE)
        {
        }
        else if constexpr (PERF_RUN_TYPE == PerfRunType::UNPACK_ISOLATE || PERF_RUN_TYPE == PerfRunType::L1_CONGESTION)
        {
            if constexpr (!unpack_to_dest)
            {
                const std::uint32_t dvalids_per_tile = (BROADCAST_TYPE == BroadcastType::SCALAR) ? 1u : num_faces;
                _perf_math_loop_clear_valid<false /*clear_a*/, true /*clear_b*/>(LOOP_FACTOR * num_blocks * tiles_in_block * dvalids_per_tile);
            }
            else
            {
                _perf_math_loop_clear_valid<false /*clear_a*/, true /*clear_b*/>(LOOP_FACTOR * num_blocks * tiles_in_block);
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
                        _llk_math_eltwise_unary_broadcast_(tile);
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
                        _llk_math_eltwise_unary_broadcast_(tile);
                    }
                    _llk_math_set_dvalid_<p_cleardvalid::FPU, dest_sync>();
                }
            }
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

        const ckernel::TensorShape tensor_shape = TENSOR_SHAPE_FROM_PARAMS(params);

        ckernel::trisc::bfd_alloc_and_program<ckernel::trisc::BfdResource::Pack0>(tensor_shape, L1_ADDRESS(buffer_Res[0]), formats.pack_dst);
        _llk_pack_hw_configure_<p_pacr::PACK0, is_fp32_dest_acc_en>(static_cast<DataFormat>(formats.pack_src), ckernel::ReluConfig::none());
        _llk_pack_init_(ckernel::trisc::bfd_current<ckernel::trisc::BfdResource::Pack0>(), tensor_shape, 1 /*num_tiles*/);
        PROFILER_SYNC();
    }
    {
        ZONE_SCOPED("TILE_LOOP")
        const ckernel::TensorShape tensor_shape = TENSOR_SHAPE_FROM_PARAMS(params);

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
                    for (std::uint32_t tile = 0; tile < output_tiles_in_block; tile++)
                    {
                        const std::uint32_t res_tile_idx = (block * output_tiles_in_block) + tile;
                        _llk_pack_(tile, res_tile_idx, tensor_shape);
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
                    for (std::uint32_t tile = 0; tile < output_tiles_in_block; tile++)
                    {
                        const std::uint32_t res_tile_idx = (block * output_tiles_in_block) + tile;
                        _llk_pack_(tile, res_tile_idx, tensor_shape);
                    }
                    _llk_pack_dest_dvalid_section_done_<dest_sync, is_fp32_dest_acc_en>();
                }
            }
        }
        PROFILER_SYNC();
    }
}

#endif
