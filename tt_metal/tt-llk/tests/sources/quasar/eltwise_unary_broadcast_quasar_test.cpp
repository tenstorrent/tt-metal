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

// Globals
std::uint32_t unp_cfg_context          = 0;
std::uint32_t pack_sync_tile_dst_ptr   = 0;
std::uint32_t math_sync_tile_dst_index = 0;

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
    const std::uint32_t LOOP_FACTOR = params.LOOP_FACTOR;
    const std::uint32_t TILE_CNT    = params.TILE_CNT;
#endif
    tdma_descriptor_t td_val_A, td_val_B;
    const std::uint32_t buf_desc_id_a        = 0;
    const std::uint32_t buf_desc_id_b        = 1;
    const std::uint32_t num_tiles_per_unpack = params.TILE_CNT;

    {
        ZONE_SCOPED("INIT")
        if (unpack_to_dest)
        {
            set_up_dest_dvalid_per_thread<dest_dvalid_client::UNPACK>({dest_dvalid_client::UNPACK, dest_dvalid_client::PACK});
            _llk_math_upk_to_dest_hw_configure_<IMPLIED_MATH_FORMAT, is_fp32_dest_acc_en, false /*is_int_fpu_en*/>();
        }
        else
        {
            set_up_dest_dvalid_per_thread<dest_dvalid_client::UNPACK>({dest_dvalid_client::FPU, dest_dvalid_client::PACK});
        }

        const auto tensor_shape = tensor_shape_from_params(params);

        td_val_A = ckernel::trisc::construct_tdma_desc(tensor_shape, L1_ADDRESS(params.buffer_B[0]), formats.unpack_A_src, buf_desc_id_a, formats.unpack_A_dst);
        td_val_B = ckernel::trisc::construct_tdma_desc(tensor_shape, L1_ADDRESS(params.buffer_B[0]), formats.unpack_A_src, buf_desc_id_b, formats.unpack_A_dst);

        _configure_buf_desc_table_(td_val_A.buf_desc_id, td_val_A.buf_desc);
        _configure_buf_desc_table_(td_val_B.buf_desc_id, td_val_B.buf_desc);

        if (is_fp32_dest_acc_en)
        {
            if (unpack_to_dest)
            {
                _llk_unpack_configure_binary_<p_unpacr::UNP_A, p_unpacr::UNP_B>(td_val_A, td_val_B);
            }
            else
            {
                _llk_unpack_configure_unary_<p_unpacr::UNP_B>(td_val_B);
            }
        }
        else
        {
            if constexpr (unpack_to_dest)
            {
                _llk_unpack_configure_unary_<p_unpacr::UNP_A>(td_val_A);
            }
            else
            {
                _llk_unpack_configure_unary_<p_unpacr::UNP_B>(td_val_B);
            }
        }

        if constexpr (unpack_to_dest)
        {
            _llk_unpack_unary_broadcast_operands_init_<p_unpacr::UNP_A, BROADCAST_TYPE, unpack_to_dest, is_fp32_dest_acc_en>(buf_desc_id_a, 1);
        }
        else
        {
            _llk_unpack_unary_broadcast_operands_init_<p_unpacr::UNP_B, BROADCAST_TYPE, unpack_to_dest, is_fp32_dest_acc_en>(buf_desc_id_b, 1);
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
            if constexpr (unpack_to_dest)
            {
                _perf_unpack_loop_set_valid<true, false>(LOOP_FACTOR * TILE_CNT);
            }
            else
            {
                _perf_unpack_loop_set_valid<false, true>(LOOP_FACTOR * TILE_CNT);
            }
        }
        else if constexpr (unpack_to_dest)
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; loop++)
            {
                for (std::uint32_t i = 0; i < num_tiles_per_unpack; ++i)
                {
                    _llk_unpack_unary_broadcast_operands_<p_unpacr::UNP_A, unpack_to_dest>(i);
                }
            }
        }
        else
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; loop++)
            {
                for (std::uint32_t i = 0; i < num_tiles_per_unpack; ++i)
                {
                    _llk_unpack_unary_broadcast_operands_<p_unpacr::UNP_B, unpack_to_dest>(i);
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
    const std::uint32_t LOOP_FACTOR = params.LOOP_FACTOR;
#endif
    DataFormat src_format = static_cast<DataFormat>(formats.math);
    _llk_math_srcAB_hw_configure_<IMPLIED_MATH_FORMAT, is_fp32_dest_acc_en, false /*int32_dest*/>(src_format, src_format);

    if constexpr (!unpack_to_dest)
    {
        {
            ZONE_SCOPED("INIT")
            set_up_dest_dvalid_per_thread<dest_dvalid_client::FPU>({dest_dvalid_client::FPU, dest_dvalid_client::PACK});

            const auto tensor_shape = tensor_shape_from_params(params);

            _llk_math_eltwise_unary_broadcast_init_<BROADCAST_TYPE, unpack_to_dest, is_fp32_dest_acc_en>(tensor_shape);
            PROFILER_SYNC();
        }
        {
            ZONE_SCOPED("TILE_LOOP")
            const auto tensor_shape = tensor_shape_from_params(params);

            const std::uint32_t tiles_in_block = params.OUTPUT_NUM_TILES_IN_BLOCK;
            const std::uint32_t num_blocks     = static_cast<std::uint32_t>(params.INPUT_NUM_BLOCKS);

            if constexpr (PERF_RUN_TYPE == PerfRunType::PACK_ISOLATE)
            {
            }
            else if constexpr (PERF_RUN_TYPE == PerfRunType::UNPACK_ISOLATE || PERF_RUN_TYPE == PerfRunType::L1_CONGESTION)
            {
                _perf_math_loop_clear_valid<false, true>(LOOP_FACTOR * num_blocks * tiles_in_block * params.num_faces);
            }
            else if constexpr (PERF_RUN_TYPE == PerfRunType::MATH_ISOLATE)
            {
                for (std::uint32_t loop = 0; loop < LOOP_FACTOR; loop++)
                {
                    for (std::uint32_t block = 0; block < num_blocks; block++)
                    {
                        for (std::uint32_t tile = 0; tile < tiles_in_block; tile++)
                        {
                            _llk_math_eltwise_unary_broadcast_<BROADCAST_TYPE, unpack_to_dest, is_fp32_dest_acc_en>(tile, tensor_shape);
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
                            _llk_math_eltwise_unary_broadcast_<BROADCAST_TYPE, unpack_to_dest, is_fp32_dest_acc_en>(tile, tensor_shape);
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

#include "llk_pack.h"
#include "llk_pack_common.h"
#include "params.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
#ifndef SPEED_OF_LIGHT
    const std::uint32_t LOOP_FACTOR = params.LOOP_FACTOR;
#endif

    std::uint32_t const buf_desc_id = 8;

    {
        ZONE_SCOPED("INIT")
        if (unpack_to_dest)
        {
            set_up_dest_dvalid_per_thread<dest_dvalid_client::PACK>({dest_dvalid_client::UNPACK, dest_dvalid_client::PACK});
        }
        else
        {
            set_up_dest_dvalid_per_thread<dest_dvalid_client::PACK>({dest_dvalid_client::FPU, dest_dvalid_client::PACK});
        }

        const auto tensor_shape = tensor_shape_from_params(params);

        tdma_descriptor_t tdma_desc =
            ckernel::trisc::construct_tdma_desc(tensor_shape, L1_ADDRESS(params.buffer_Res[0]), formats.pack_dst, buf_desc_id, formats.pack_src);

        _configure_buf_desc_table_(tdma_desc.buf_desc_id, tdma_desc.buf_desc);
        _llk_pack_hw_configure_<p_pacr::PACK0>(tdma_desc);
        _llk_pack_init_(buf_desc_id, tensor_shape, 1);
        PROFILER_SYNC();
    }
    {
        ZONE_SCOPED("TILE_LOOP")
        const auto tensor_shape = tensor_shape_from_params(params);

        const std::uint32_t output_num_blocks     = static_cast<std::uint32_t>(params.OUTPUT_NUM_BLOCKS);
        const std::uint32_t output_tiles_in_block = params.OUTPUT_NUM_TILES_IN_BLOCK;

        if constexpr (PERF_RUN_TYPE == PerfRunType::MATH_ISOLATE || PERF_RUN_TYPE == PerfRunType::UNPACK_ISOLATE)
        {
        }
        else if constexpr (PERF_RUN_TYPE == PerfRunType::PACK_ISOLATE || PERF_RUN_TYPE == PerfRunType::L1_CONGESTION)
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
