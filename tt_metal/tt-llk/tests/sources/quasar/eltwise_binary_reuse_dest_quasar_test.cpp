// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Test for eltwise binary operations with reuse_dest on Quasar.

#include <cstdint>
#include <cstdio>

#include "ckernel.h"
#include "llk_defs.h"
#include "llk_memory_checks.h"
#include "perf.h"
#include "profiler.h"
#include "sfpu_stub.h"

// Globals
std::uint32_t unp_cfg_context          = 0;
std::uint32_t pack_sync_tile_dst_ptr   = 0;
std::uint32_t math_sync_tile_dst_index = 0;

#ifdef LLK_TRISC_UNPACK

#include "llk_unpack_common.h"
#include "llk_unpack_unary_operand.h"
#include "params.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
#ifndef SPEED_OF_LIGHT
    const std::uint32_t LOOP_FACTOR     = params.LOOP_FACTOR;
    const std::uint32_t INPUT_TILE_CNT  = params.INPUT_TILE_CNT;
    const std::uint32_t num_faces       = params.num_faces;
    const std::uint32_t TEST_FACE_C_DIM = params.TEST_FACE_C_DIM;
    const std::uint32_t TEST_FACE_R_DIM = params.TEST_FACE_R_DIM;
    const Operand& buffer_A             = params.buffer_A;
    const Operand& buffer_B             = params.buffer_B;
#endif
    tdma_descriptor_t td_val_A, td_val_B;
    const std::uint32_t buf_desc_id_a          = 0;
    const std::uint32_t buf_desc_id_b          = 1;
    constexpr bool TRANSPOSE_EN                = false;
    constexpr bool IS_32B_DEST_EN              = false;
    constexpr std::uint32_t buf_desc_id_phase2 = (REUSE_DEST_TYPE == EltwiseBinaryReuseDestType::DEST_TO_SRCB) ? buf_desc_id_a : buf_desc_id_b;
    constexpr std::uint32_t unp_sel_phase2     = (REUSE_DEST_TYPE == EltwiseBinaryReuseDestType::DEST_TO_SRCA) ? p_unpacr::UNP_B : p_unpacr::UNP_A;

    {
        ZONE_SCOPED("INIT")
        // Setup data valid scheme
        set_up_dest_dvalid_per_thread<dest_dvalid_client::UNPACK>({dest_dvalid_client::FPU, dest_dvalid_client::PACK});

        buffer_descriptor_u bd_val_A {};
        bd_val_A.f.l1_addr_16B = L1_ADDRESS(buffer_A[0]);
        bd_val_A.f.format      = static_cast<std::uint8_t>(formats.unpack_A_src);
        bd_val_A.f.x_dim       = TEST_FACE_C_DIM;
        bd_val_A.f.y_dim       = TEST_FACE_R_DIM;
        bd_val_A.f.z_dim       = num_faces;

        td_val_A.buf_desc        = bd_val_A;
        td_val_A.buf_desc_id     = buf_desc_id_a;
        td_val_A.reg_data_format = static_cast<std::uint8_t>(formats.unpack_A_dst);

        buffer_descriptor_u bd_val_B {};
        bd_val_B.f.l1_addr_16B = L1_ADDRESS(buffer_B[0]);
        bd_val_B.f.format      = static_cast<std::uint8_t>(formats.unpack_B_src);
        bd_val_B.f.x_dim       = TEST_FACE_C_DIM;
        bd_val_B.f.y_dim       = TEST_FACE_R_DIM;
        bd_val_B.f.z_dim       = num_faces;

        td_val_B.buf_desc        = bd_val_B;
        td_val_B.buf_desc_id     = buf_desc_id_b;
        td_val_B.reg_data_format = static_cast<std::uint8_t>(formats.unpack_B_dst);

        _configure_buf_desc_table_(td_val_A.buf_desc_id, td_val_A.buf_desc);
        _configure_buf_desc_table_(td_val_B.buf_desc_id, td_val_B.buf_desc);
        _llk_unpack_configure_binary_<p_unpacr::UNP_A, p_unpacr::UNP_B>(td_val_A, td_val_B);

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
                // Phase 1 datacopy consumes SrcA once per tile.
                _perf_unpack_loop_set_valid<true, false>(INPUT_TILE_CNT);
                // Phase 2 reuse-dest binary consumes both sources per face.
                _perf_unpack_loop_set_valid<true, true>(INPUT_TILE_CNT * num_faces);
            }
        }
        else
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; loop++)
            {
                // Phase 1 and phase 2 share unpack's bank0 instruction
                // buffer, so select the phase 1 MOP before executing it.
                _llk_unpack_unary_operand_init_<p_unpacr::UNP_A, TRANSPOSE_EN, IS_32B_DEST_EN>(buf_desc_id_a, ckernel::DEFAULT_TENSOR_SHAPE, 1);
                for (std::uint32_t i = 0; i < INPUT_TILE_CNT; ++i)
                {
                    _llk_unpack_unary_operand_<p_unpacr::UNP_A>(i, ckernel::DEFAULT_TENSOR_SHAPE);
                }

                // Select the reuse-dest phase 2 MOP after phase 1 completes.
                _llk_unpack_unary_operand_init_<unp_sel_phase2, TRANSPOSE_EN, IS_32B_DEST_EN, REUSE_DEST_TYPE>(
                    buf_desc_id_phase2, ckernel::DEFAULT_TENSOR_SHAPE, 1);
                for (std::uint32_t i = 0; i < INPUT_TILE_CNT; ++i)
                {
                    _llk_unpack_unary_operand_<unp_sel_phase2, REUSE_DEST_TYPE>(i, ckernel::DEFAULT_TENSOR_SHAPE);
                }
            }
        }
        PROFILER_SYNC();
    }
}

#endif

#ifdef LLK_TRISC_MATH

#include "llk_math_common.h"
#include "llk_math_eltwise_binary.h"
#include "llk_math_eltwise_unary_datacopy.h"
#include "params.h"
#include "tensor_shape.h"

using namespace ckernel;

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
    {
        ZONE_SCOPED("INIT")
        // PACK_ISOLATE measures pack alone (WH/BH style): skip FPU→PACK dest-dvalid.
        if constexpr (PERF_RUN_TYPE == PerfRunType::L1_TO_L1 || PERF_RUN_TYPE == PerfRunType::MATH_ISOLATE)
        {
            set_up_dest_dvalid_per_thread<dest_dvalid_client::FPU>({dest_dvalid_client::FPU, dest_dvalid_client::PACK});
        }

        DataFormat src_format = static_cast<DataFormat>(formats.math);
        _llk_math_srcAB_hw_configure_<IMPLIED_MATH_FORMAT, is_fp32_dest_acc_en, false /*int32*/>(src_format, src_format);

        PROFILER_SYNC();
    }
    {
        ZONE_SCOPED("TILE_LOOP")
        const int num_total_tiles = INPUT_NUM_TILES_IN_BLOCK * INPUT_NUM_BLOCKS;
        const int tiles_in_block  = OUTPUT_NUM_TILES_IN_BLOCK;
        const int num_tiles_accum = INPUT_NUM_TILES_IN_BLOCK / tiles_in_block;
        const int num_blocks      = INPUT_NUM_BLOCKS;

        if constexpr (PERF_RUN_TYPE == PerfRunType::PACK_ISOLATE)
        {
        }
        else if constexpr (PERF_RUN_TYPE == PerfRunType::UNPACK_ISOLATE || PERF_RUN_TYPE == PerfRunType::L1_CONGESTION)
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; loop++)
            {
                _perf_math_loop_clear_valid<true, false>(INPUT_TILE_CNT);
                _perf_math_loop_clear_valid<true, true>(INPUT_TILE_CNT * num_faces);
            }
        }
        else if constexpr (PERF_RUN_TYPE == PerfRunType::MATH_ISOLATE)
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; loop++)
            {
                // Datacopy and binary share math's bank0 instruction buffer.
                _llk_math_eltwise_unary_datacopy_init_<DataCopyType::A2D, is_fp32_dest_acc_en>(num_faces * TEST_FACE_R_DIM, 1);
                for (int i = 0; i < num_total_tiles; ++i)
                {
                    _llk_math_eltwise_unary_datacopy_(num_faces * TEST_FACE_R_DIM, i);
                }

                _llk_math_eltwise_binary_init_<ELTWISE_BINARY_OP, MATH_FIDELITY, REUSE_DEST_TYPE>(ckernel::DEFAULT_TENSOR_SHAPE);
                for (int block = 0; block < num_blocks; block++)
                {
                    for (int n = 0; n < num_tiles_accum; n++)
                    {
                        for (int tile = 0; tile < tiles_in_block; tile++)
                        {
                            const int global_tile_idx = block * tiles_in_block + tile;
                            _llk_math_eltwise_binary_<ELTWISE_BINARY_OP, REUSE_DEST_TYPE>(
                                global_tile_idx, ckernel::DEFAULT_TENSOR_SHAPE, is_fp32_dest_acc_en /*clear_fp32_mode*/);
                        }
                    }
                }
            }
        }
        else
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; loop++)
            {
                _llk_math_eltwise_unary_datacopy_init_<DataCopyType::A2D, is_fp32_dest_acc_en>(num_faces * TEST_FACE_R_DIM, 1);
                for (int i = 0; i < num_total_tiles; ++i)
                {
                    _llk_math_eltwise_unary_datacopy_(num_faces * TEST_FACE_R_DIM, i);
                }

                _llk_math_eltwise_binary_init_<ELTWISE_BINARY_OP, MATH_FIDELITY, REUSE_DEST_TYPE>(ckernel::DEFAULT_TENSOR_SHAPE);
                for (int block = 0; block < num_blocks; block++)
                {
                    for (int n = 0; n < num_tiles_accum; n++)
                    {
                        for (int tile = 0; tile < tiles_in_block; tile++)
                        {
                            const int global_tile_idx = block * tiles_in_block + tile;
                            _llk_math_eltwise_binary_<ELTWISE_BINARY_OP, REUSE_DEST_TYPE>(
                                global_tile_idx, ckernel::DEFAULT_TENSOR_SHAPE, is_fp32_dest_acc_en /*clear_fp32_mode*/);
                        }
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
    const std::uint32_t LOOP_FACTOR               = params.LOOP_FACTOR;
    const std::uint32_t TEST_FACE_C_DIM           = params.TEST_FACE_C_DIM;
    const std::uint32_t TEST_FACE_R_DIM           = params.TEST_FACE_R_DIM;
    const std::uint32_t num_faces                 = params.num_faces;
    const std::uint32_t OUTPUT_NUM_TILES_IN_BLOCK = params.OUTPUT_NUM_TILES_IN_BLOCK;
    const std::uint32_t OUTPUT_NUM_BLOCKS         = params.OUTPUT_NUM_BLOCKS;
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
            set_up_dest_dvalid_per_thread<dest_dvalid_client::PACK>({dest_dvalid_client::FPU, dest_dvalid_client::PACK});
        }

        buffer_descriptor_u bd_val {};
        bd_val.f.l1_addr_16B = L1_ADDRESS(buffer_Res[0]);
        bd_val.f.format      = static_cast<std::uint8_t>(formats.pack_dst);
        bd_val.f.x_dim       = TEST_FACE_C_DIM;
        bd_val.f.y_dim       = TEST_FACE_R_DIM;
        bd_val.f.z_dim       = num_faces;

        tdma_descriptor_t tdma_desc;
        tdma_desc.buf_desc        = bd_val;
        tdma_desc.buf_desc_id     = buf_desc_id;
        tdma_desc.reg_data_format = static_cast<std::uint8_t>(formats.pack_src);

        _configure_buf_desc_table_(tdma_desc.buf_desc_id, tdma_desc.buf_desc);
        _llk_pack_hw_configure_<p_pacr::PACK0>(tdma_desc);
        _llk_pack_init_(buf_desc_id, ckernel::DEFAULT_TENSOR_SHAPE, 1);
        PROFILER_SYNC();
    }
    {
        ZONE_SCOPED("TILE_LOOP")
        const int output_tiles_in_block = OUTPUT_NUM_TILES_IN_BLOCK;
        const int output_num_blocks     = OUTPUT_NUM_BLOCKS;

        if constexpr (PERF_RUN_TYPE == PerfRunType::MATH_ISOLATE || PERF_RUN_TYPE == PerfRunType::UNPACK_ISOLATE)
        {
        }
        else if constexpr (PERF_RUN_TYPE == PerfRunType::PACK_ISOLATE || PERF_RUN_TYPE == PerfRunType::L1_CONGESTION)
        {
            // No dest-dvalid section_done: WH/BH isolate packs without math handshake.
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; loop++)
            {
                for (int block = 0; block < output_num_blocks; block++)
                {
                    for (int tile = 0; tile < output_tiles_in_block; tile++)
                    {
                        int res_tile_idx = (block * output_tiles_in_block) + tile;
                        _llk_pack_(res_tile_idx, res_tile_idx, ckernel::DEFAULT_TENSOR_SHAPE);
                    }
                }
            }
        }
        else
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; loop++)
            {
                for (int block = 0; block < output_num_blocks; block++)
                {
                    for (int tile = 0; tile < output_tiles_in_block; tile++)
                    {
                        int res_tile_idx = (block * output_tiles_in_block) + tile;
                        _llk_pack_(res_tile_idx, res_tile_idx, ckernel::DEFAULT_TENSOR_SHAPE);
                    }
                    _llk_pack_dest_dvalid_section_done_<dest_sync, is_fp32_dest_acc_en>();
                }
            }
        }
        PROFILER_SYNC();
    }
}

#endif
