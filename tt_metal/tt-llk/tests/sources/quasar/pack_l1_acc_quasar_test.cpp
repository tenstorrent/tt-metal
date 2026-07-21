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
#include "sfpu_stub.h"

#ifdef LLK_TRISC_UNPACK

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
    const std::uint32_t TEST_FACE_C_DIM           = params.TEST_FACE_C_DIM;
    const std::uint32_t TEST_FACE_R_DIM           = params.TEST_FACE_R_DIM;
    const std::uint32_t num_faces                 = params.num_faces;
    const std::uint32_t OUTPUT_NUM_TILES_IN_BLOCK = params.OUTPUT_NUM_TILES_IN_BLOCK;
    const std::uint32_t OUTPUT_NUM_BLOCKS         = params.OUTPUT_NUM_BLOCKS;
    const Operand& buffer_A                       = params.buffer_A;
#endif
    const std::uint32_t SELECTED_UNPACKER = unpack_to_dest ? p_unpacr::UNP_DEST : p_unpacr::UNP_A;
    tdma_descriptor_t td_val;
    const std::uint32_t buf_desc_id = 0;

    {
        ZONE_SCOPED("INIT")
        if constexpr (unpack_to_dest)
        {
            if constexpr (PERF_RUN_TYPE == PerfRunType::L1_TO_L1)
            {
                set_up_dest_dvalid_per_thread<dest_dvalid_client::UNPACK>({dest_dvalid_client::UNPACK, dest_dvalid_client::PACK});
            }

            DataFormat pack_src_format = static_cast<DataFormat>(formats.pack_src);
            if (is_fp32_dest_acc_en && pack_src_format == DataFormat::Float32)
            {
                _llk_math_upk_to_dest_hw_configure_<IMPLIED_MATH_FORMAT, true /*fp32_dest*/, false /*int32_dest*/>();
            }
            else if (is_fp32_dest_acc_en && pack_src_format == DataFormat::Int32)
            {
                _llk_math_upk_to_dest_hw_configure_<IMPLIED_MATH_FORMAT, false /*fp32_dest*/, true /*int32_dest*/>();
            }
            else
            {
                _llk_math_upk_to_dest_hw_configure_<IMPLIED_MATH_FORMAT, false /*fp32_dest*/, false /*int32_dest*/>();
            }
        }

        buffer_descriptor_u bd_val = {0};
        bd_val.f.l1_addr_16B       = L1_ADDRESS(buffer_A[0]);
        bd_val.f.format            = static_cast<std::uint8_t>(formats.unpack_A_src);
        bd_val.f.x_dim             = TEST_FACE_C_DIM;
        bd_val.f.y_dim             = TEST_FACE_R_DIM;
        bd_val.f.z_dim             = num_faces;

        td_val.buf_desc        = bd_val;
        td_val.buf_desc_id     = buf_desc_id;
        td_val.reg_data_format = static_cast<std::uint8_t>(formats.unpack_A_dst);

        _configure_buf_desc_table_(td_val.buf_desc_id, td_val.buf_desc);
        if constexpr (is_fp32_dest_acc_en && !unpack_to_dest)
        {
            _llk_unpack_configure_binary_<p_unpacr::UNP_A, p_unpacr::UNP_B>(td_val, td_val);
        }
        else
        {
            _llk_unpack_configure_unary_<SELECTED_UNPACKER>(td_val);
        }

        if constexpr (unpack_to_dest)
        {
            const std::uint32_t tiles_in_block = OUTPUT_NUM_TILES_IN_BLOCK;
            _llk_unpack_unary_operand_init_<SELECTED_UNPACKER, false /*transpose*/, is_fp32_dest_acc_en>(
                buf_desc_id, ckernel::DEFAULT_TENSOR_SHAPE, tiles_in_block /*num_tiles_per_unpack*/);
        }
        else
        {
            _llk_unpack_unary_operand_init_<SELECTED_UNPACKER, false /*transpose*/, is_fp32_dest_acc_en>(
                buf_desc_id, ckernel::DEFAULT_TENSOR_SHAPE, 1 /*num_tiles_per_unpack*/);
        }
        PROFILER_SYNC();
    }
    {
        ZONE_SCOPED("TILE_LOOP")
        const std::uint32_t tiles_in_block = OUTPUT_NUM_TILES_IN_BLOCK;
        const std::uint32_t num_blocks     = static_cast<std::uint32_t>(OUTPUT_NUM_BLOCKS);

        if constexpr (PERF_RUN_TYPE == PerfRunType::PACK_ISOLATE)
        {
        }
        else if constexpr (PERF_RUN_TYPE == PerfRunType::MATH_ISOLATE)
        {
            if constexpr (!unpack_to_dest)
            {
                if constexpr (is_fp32_dest_acc_en)
                {
                    _perf_unpack_loop_set_valid<true, true>(LOOP_FACTOR * TILE_CNT);
                }
                else
                {
                    _perf_unpack_loop_set_valid<true, false>(LOOP_FACTOR * TILE_CNT);
                }
            }
        }
        else if constexpr (unpack_to_dest)
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; loop++)
            {
                for (std::uint32_t block = 0; block < num_blocks; block++)
                {
                    _llk_unpack_unary_operand_<SELECTED_UNPACKER>(block * tiles_in_block, ckernel::DEFAULT_TENSOR_SHAPE);
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
            // PACK_ISOLATE measures pack alone (WH/BH style): skip FPU→PACK dest-dvalid.
            if constexpr (PERF_RUN_TYPE == PerfRunType::L1_TO_L1 || PERF_RUN_TYPE == PerfRunType::MATH_ISOLATE)
            {
                set_up_dest_dvalid_per_thread<dest_dvalid_client::FPU>({dest_dvalid_client::FPU, dest_dvalid_client::PACK});
            }

            DataFormat math_format     = static_cast<DataFormat>(formats.math);
            DataFormat pack_src_format = static_cast<DataFormat>(formats.pack_src);
            if (is_fp32_dest_acc_en && pack_src_format == DataFormat::Float32)
            {
                _llk_math_srcAB_hw_configure_<IMPLIED_MATH_FORMAT, true /*fp32_dest*/, false /*int32_dest*/>(math_format, math_format);
            }
            else if (is_fp32_dest_acc_en && pack_src_format == DataFormat::Int32)
            {
                _llk_math_srcAB_hw_configure_<IMPLIED_MATH_FORMAT, false /*fp32_dest*/, true /*int32_dest*/>(math_format, math_format);
            }
            else
            {
                _llk_math_srcAB_hw_configure_<IMPLIED_MATH_FORMAT, false /*fp32_dest*/, false /*int32_dest*/>(math_format, math_format);
            }

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
                if constexpr (is_fp32_dest_acc_en)
                {
                    _perf_math_loop_clear_valid<true, true>(LOOP_FACTOR * num_blocks * tiles_in_block);
                }
                else
                {
                    _perf_math_loop_clear_valid<true, false>(LOOP_FACTOR * num_blocks * tiles_in_block);
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
    const std::uint32_t TEST_FACE_C_DIM           = params.TEST_FACE_C_DIM;
    const std::uint32_t TEST_FACE_R_DIM           = params.TEST_FACE_R_DIM;
    const std::uint32_t num_faces                 = params.num_faces;
    const int RELU_CONFIG                         = params.RELU_CONFIG;
    const std::uint32_t OUTPUT_NUM_BLOCKS         = params.OUTPUT_NUM_BLOCKS;
    const std::uint32_t OUTPUT_NUM_TILES_IN_BLOCK = params.OUTPUT_NUM_TILES_IN_BLOCK;
    const Operand& buffer_Res                     = params.buffer_Res;
#endif

    std::uint32_t const buf_desc_id = 8;

    {
        ZONE_SCOPED("INIT")
        // Match WH/BH PACK_ISOLATE: no math↔pack handshake; pack from whatever is in dest.
        // Explicitly clear wait_mask — CFG can persist across run-types in the same session.
        if constexpr (PERF_RUN_TYPE == PerfRunType::PACK_ISOLATE || PERF_RUN_TYPE == PerfRunType::L1_CONGESTION)
        {
            auto cfg                                    = (std::uint32_t volatile*)TENSIX_CFG_BASE;
            cfg[PACK_DEST_DVALID_CTRL_wait_mask_ADDR32] = 0;
        }
        else if constexpr (PERF_RUN_TYPE == PerfRunType::L1_TO_L1)
        {
            if constexpr (unpack_to_dest)
            {
                set_up_dest_dvalid_per_thread<dest_dvalid_client::PACK>({dest_dvalid_client::UNPACK, dest_dvalid_client::PACK});
            }
            else
            {
                set_up_dest_dvalid_per_thread<dest_dvalid_client::PACK>({dest_dvalid_client::FPU, dest_dvalid_client::PACK});
            }
        }

        buffer_descriptor_u bd_val = {0};
        tdma_descriptor_t tdma_desc;

        bd_val.f.l1_addr_16B = L1_ADDRESS(buffer_Res[0]);
        bd_val.f.format      = static_cast<std::uint8_t>(formats.pack_dst);
        bd_val.f.x_dim       = TEST_FACE_C_DIM;
        bd_val.f.y_dim       = TEST_FACE_R_DIM;
        bd_val.f.z_dim       = num_faces;

        tdma_desc.buf_desc        = bd_val;
        tdma_desc.buf_desc_id     = buf_desc_id;
        tdma_desc.reg_data_format = static_cast<std::uint8_t>(formats.pack_src);

        _configure_buf_desc_table_(tdma_desc.buf_desc_id, tdma_desc.buf_desc);
        _llk_pack_hw_configure_<p_pacr::PACK0>(tdma_desc);
        const ckernel::ReluConfig relu_config = ckernel::ReluConfig::from_packed(RELU_CONFIG);
        _llk_pack_init_<is_fp32_dest_acc_en>(buf_desc_id, ckernel::DEFAULT_TENSOR_SHAPE, 1 /*num_tiles_per_pack*/, relu_config);
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
