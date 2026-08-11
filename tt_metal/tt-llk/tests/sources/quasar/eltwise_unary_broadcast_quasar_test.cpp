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
    tdma_descriptor_t td_val_A, td_val_B;
    const std::uint32_t buf_desc_id_a = 0;
    const std::uint32_t buf_desc_id_b = 1;
    // Unpack to dest must use the num tiles per unpack parameter in order to unpack multiple tiles per Dest bank
    const std::uint32_t num_tiles_per_unpack = unpack_to_dest ? tiles_in_block : 1;

    {
        ZONE_SCOPED("INIT")
        if constexpr (unpack_to_dest)
        {
            if constexpr (PERF_RUN_TYPE == PerfRunType::L1_TO_L1)
            {
                set_up_dest_dvalid_per_thread<dest_dvalid_client::UNPACK>({dest_dvalid_client::UNPACK, dest_dvalid_client::FPU, dest_dvalid_client::PACK});
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
            set_up_dest_dvalid_per_thread<dest_dvalid_client::UNPACK>({dest_dvalid_client::FPU, dest_dvalid_client::PACK});
        }

        const ckernel::TensorShape tensor_shape = TENSOR_SHAPE_FROM_PARAMS(params);

        td_val_A = ckernel::trisc::construct_tdma_desc(tensor_shape, L1_ADDRESS(buffer_B[0]), formats.unpack_A_src, buf_desc_id_a, formats.unpack_A_dst);
        td_val_B = ckernel::trisc::construct_tdma_desc(tensor_shape, L1_ADDRESS(buffer_B[0]), formats.unpack_A_src, buf_desc_id_b, formats.unpack_A_dst);

        _configure_buf_desc_table_(td_val_A.buf_desc_id, td_val_A.buf_desc);
        _configure_buf_desc_table_(td_val_B.buf_desc_id, td_val_B.buf_desc);

        _llk_unpack_configure_unary_<UNPACKER_ENGINE_SEL>(unpack_to_dest ? td_val_A : td_val_B);
        _llk_unpack_unary_broadcast_operands_init_<UNPACKER_ENGINE_SEL, BROADCAST_TYPE, unpack_to_dest>(
            unpack_to_dest ? buf_desc_id_a : buf_desc_id_b, num_tiles_per_unpack);

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
                    const std::uint32_t input_tile_idx = block * tiles_in_block;
                    _llk_unpack_unary_broadcast_operands_<UNPACKER_ENGINE_SEL, unpack_to_dest>(input_tile_idx);
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
    const DataFormat pack_src_format        = static_cast<DataFormat>(formats.pack_src);
    const ckernel::TensorShape tensor_shape = TENSOR_SHAPE_FROM_PARAMS(params);

    {
        ZONE_SCOPED("INIT")
        if constexpr (PERF_RUN_TYPE == PerfRunType::L1_TO_L1)
        {
            if constexpr (unpack_to_dest)
            {
                set_up_dest_dvalid_per_thread<dest_dvalid_client::FPU>({dest_dvalid_client::UNPACK, dest_dvalid_client::FPU, dest_dvalid_client::PACK});
            }
            else
            {
                set_up_dest_dvalid_per_thread<dest_dvalid_client::FPU>({dest_dvalid_client::FPU, dest_dvalid_client::PACK});
            }
        }
        else if constexpr (PERF_RUN_TYPE == PerfRunType::MATH_ISOLATE && unpack_to_dest)
        {
            // L1_TO_L1 leaves FPU configured as the middle client in the
            // UNPACK→FPU→PACK chain. Math isolate has no unpack destination
            // pulse, so make FPU the producer and restore immediate ownership
            // of the destination register.
            set_up_dest_dvalid_per_thread<dest_dvalid_client::FPU>({dest_dvalid_client::FPU, dest_dvalid_client::PACK});
        }

        if (is_fp32_dest_acc_en && pack_src_format == DataFormat::Int32)
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

    constexpr std::uint32_t buf_desc_id    = 8;
    const std::uint32_t num_tiles_per_pack = 1;

    {
        ZONE_SCOPED("INIT")
        // PACK_ISOLATE and L1_CONGESTION pack without a math↔pack handshake.
        // Explicitly clear wait_mask — CFG can persist across run-types in the same session.
        if constexpr (PERF_RUN_TYPE == PerfRunType::PACK_ISOLATE || PERF_RUN_TYPE == PerfRunType::L1_CONGESTION)
        {
            auto cfg                                    = (volatile std::uint32_t*)TENSIX_CFG_BASE;
            cfg[PACK_DEST_DVALID_CTRL_wait_mask_ADDR32] = 0;
        }
        else if constexpr (PERF_RUN_TYPE == PerfRunType::L1_TO_L1)
        {
            if constexpr (unpack_to_dest)
            {
                set_up_dest_dvalid_per_thread<dest_dvalid_client::PACK>({dest_dvalid_client::UNPACK, dest_dvalid_client::FPU, dest_dvalid_client::PACK});
            }
            else
            {
                set_up_dest_dvalid_per_thread<dest_dvalid_client::PACK>({dest_dvalid_client::FPU, dest_dvalid_client::PACK});
            }
        }

        const ckernel::TensorShape tensor_shape = TENSOR_SHAPE_FROM_PARAMS(params);

        tdma_descriptor_t tdma_desc =
            ckernel::trisc::construct_tdma_desc(tensor_shape, L1_ADDRESS(buffer_Res[0]), formats.pack_dst, buf_desc_id, formats.pack_src);

        _configure_buf_desc_table_(tdma_desc.buf_desc_id, tdma_desc.buf_desc);
        _llk_pack_hw_configure_<p_pacr::PACK0, is_fp32_dest_acc_en>(tdma_desc, ckernel::ReluConfig::none());
        _llk_pack_init_(buf_desc_id, tensor_shape, num_tiles_per_pack);
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
