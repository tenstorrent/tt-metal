// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <cstdio>

#include "ckernel.h"
#include "llk_defs.h"
#include "perf.h"
#include "profiler.h"
#include "sfpu_stub.h"

// Globals
std::uint32_t unp_cfg_context          = 0;
std::uint32_t pack_sync_tile_dst_ptr   = 0;
std::uint32_t math_sync_tile_dst_index = 0;

#ifdef LLK_TRISC_UNPACK

#include "llk_unpack_binary_operands.h"
#include "llk_unpack_common.h"
#include "params.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
#ifndef SPEED_OF_LIGHT
    const std::uint32_t LOOP_FACTOR     = params.LOOP_FACTOR;
    const std::uint32_t INPUT_TILE_CNT  = params.INPUT_TILE_CNT;
    const std::uint32_t TEST_FACE_C_DIM = params.TEST_FACE_C_DIM;
    const std::uint32_t TEST_FACE_R_DIM = params.TEST_FACE_R_DIM;
    const std::uint32_t num_faces       = params.num_faces;
    const Operand& buffer_A             = params.buffer_A;
    const Operand& buffer_B             = params.buffer_B;
#endif
    const std::uint32_t buf_desc_id_a = 0;
    const std::uint32_t buf_desc_id_b = 1;

    {
        ZONE_SCOPED("INIT")
        set_up_dest_dvalid_per_thread<dest_dvalid_client::UNPACK>({dest_dvalid_client::FPU, dest_dvalid_client::PACK});

        buffer_descriptor_u bd_val_A {};
        bd_val_A.f.l1_addr_16B = buffer_A[0] / 16;
        bd_val_A.f.format      = static_cast<std::uint8_t>(formats.unpack_A_src);
        bd_val_A.f.x_dim       = TEST_FACE_C_DIM;
        bd_val_A.f.y_dim       = TEST_FACE_R_DIM;
        bd_val_A.f.z_dim       = num_faces;

        buffer_descriptor_u bd_val_B {};
        bd_val_B.f.l1_addr_16B = buffer_B[0] / 16;
        bd_val_B.f.format      = static_cast<std::uint8_t>(formats.unpack_B_src);
        bd_val_B.f.x_dim       = TEST_FACE_C_DIM;
        bd_val_B.f.y_dim       = TEST_FACE_R_DIM;
        bd_val_B.f.z_dim       = num_faces;

        _configure_buf_desc_table_(buf_desc_id_a, bd_val_A);
        _configure_buf_desc_table_(buf_desc_id_b, bd_val_B);
        _llk_unpack_configure_binary_<p_unpacr::UNP_A, p_unpacr::UNP_B>(
            static_cast<DataFormat>(formats.unpack_A_dst), static_cast<DataFormat>(formats.unpack_B_dst));
        _llk_unpack_binary_operands_init_(buf_desc_id_a, buf_desc_id_b, 1 /*num_tiles_per_unpack*/);
        PROFILER_SYNC();
    }
    {
        ZONE_SCOPED("TILE_LOOP")
        if constexpr (PERF_RUN_TYPE == PerfRunType::PACK_ISOLATE)
        {
        }
        else if constexpr (PERF_RUN_TYPE == PerfRunType::MATH_ISOLATE)
        {
            _perf_unpack_loop_set_valid<true /*set_a*/, true /*set_b*/>(LOOP_FACTOR * INPUT_TILE_CNT);
        }
        else
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; loop++)
            {
                for (std::uint32_t i = 0; i < INPUT_TILE_CNT; ++i)
                {
                    _llk_unpack_binary_operands_(i, i);
                }
            }
        }
        PROFILER_SYNC();
    }
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

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
#ifndef SPEED_OF_LIGHT
    const std::uint32_t LOOP_FACTOR        = params.LOOP_FACTOR;
    const std::uint32_t INPUT_TILE_CNT     = params.INPUT_TILE_CNT;
    const std::uint32_t NUM_TILES_IN_BLOCK = params.NUM_TILES_IN_BLOCK;
#endif
    {
        ZONE_SCOPED("INIT")
        // End-to-end and math-isolate runs require FPU destination ownership.
        if constexpr (PERF_RUN_TYPE == PerfRunType::L1_TO_L1 || PERF_RUN_TYPE == PerfRunType::MATH_ISOLATE)
        {
            set_up_dest_dvalid_per_thread<dest_dvalid_client::FPU>({dest_dvalid_client::FPU, dest_dvalid_client::PACK});
        }

        DataFormat math_format     = static_cast<DataFormat>(formats.math);
        DataFormat pack_src_format = static_cast<DataFormat>(formats.pack_src);
        if (is_fp32_dest_acc_en && pack_src_format == DataFormat::Int32)
        {
            _llk_math_srcAB_hw_configure_<IMPLIED_MATH_FORMAT, false /*fp32_dest*/, true /*int32_dest*/>(math_format, math_format);
        }
        else
        {
            _llk_math_srcAB_hw_configure_<IMPLIED_MATH_FORMAT, is_fp32_dest_acc_en, false /*int32_dest*/>(math_format, math_format);
        }
        _llk_math_eltwise_binary_init_<ELTWISE_BINARY_OP, MATH_FIDELITY>(ckernel::DEFAULT_TENSOR_SHAPE, ACC_TO_DEST);
        PROFILER_SYNC();
    }
    {
        ZONE_SCOPED("TILE_LOOP")
        if constexpr (PERF_RUN_TYPE == PerfRunType::PACK_ISOLATE)
        {
        }
        else if constexpr (PERF_RUN_TYPE == PerfRunType::UNPACK_ISOLATE || PERF_RUN_TYPE == PerfRunType::L1_CONGESTION)
        {
            _perf_math_loop_clear_valid<true /*clear_a*/, true /*clear_b*/>(LOOP_FACTOR * INPUT_TILE_CNT);
        }
        else
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; loop++)
            {
                std::uint32_t dest_idx = 0;
                for (std::uint32_t remaining_tiles = INPUT_TILE_CNT; remaining_tiles > 0; remaining_tiles -= std::min(remaining_tiles, NUM_TILES_IN_BLOCK))
                {
                    const std::uint32_t num_tiles_in_block = std::min(remaining_tiles, NUM_TILES_IN_BLOCK);
                    for (std::uint32_t tile = 0; tile < num_tiles_in_block; ++tile)
                    {
                        _llk_math_eltwise_binary_<ELTWISE_BINARY_OP>(dest_idx, ckernel::DEFAULT_TENSOR_SHAPE);
                    }
                    ++dest_idx;
                }
                if constexpr (PERF_RUN_TYPE != PerfRunType::MATH_ISOLATE)
                {
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
    const std::uint32_t LOOP_FACTOR     = params.LOOP_FACTOR;
    const std::uint32_t OUTPUT_TILE_CNT = params.OUTPUT_TILE_CNT;
    const std::uint32_t TEST_FACE_C_DIM = params.TEST_FACE_C_DIM;
    const std::uint32_t TEST_FACE_R_DIM = params.TEST_FACE_R_DIM;
    const std::uint32_t num_faces       = params.num_faces;
    const Operand& buffer_Res           = params.buffer_Res;
#endif
    std::uint32_t const buf_desc_id = 8;

    {
        ZONE_SCOPED("INIT")
        // Clear a stale FPU-to-PACK wait in modes without a math handshake.
        if constexpr (PERF_RUN_TYPE == PerfRunType::PACK_ISOLATE || PERF_RUN_TYPE == PerfRunType::L1_CONGESTION)
        {
            volatile std::uint32_t* cfg                 = (volatile std::uint32_t*)TENSIX_CFG_BASE;
            cfg[PACK_DEST_DVALID_CTRL_wait_mask_ADDR32] = 0;
        }
        else if constexpr (PERF_RUN_TYPE == PerfRunType::L1_TO_L1)
        {
            set_up_dest_dvalid_per_thread<dest_dvalid_client::PACK>({dest_dvalid_client::FPU, dest_dvalid_client::PACK});
        }

        buffer_descriptor_u bd_val {};
        bd_val.f.l1_addr_16B = buffer_Res[0] / 16;
        bd_val.f.format      = static_cast<std::uint8_t>(formats.pack_dst);
        bd_val.f.x_dim       = TEST_FACE_C_DIM;
        bd_val.f.y_dim       = TEST_FACE_R_DIM;
        bd_val.f.z_dim       = num_faces;

        _configure_buf_desc_table_(buf_desc_id, bd_val);
        _llk_pack_hw_configure_<p_pacr::PACK0, is_fp32_dest_acc_en>(static_cast<DataFormat>(formats.pack_src), ckernel::ReluConfig::none());
        _llk_pack_init_(buf_desc_id, ckernel::DEFAULT_TENSOR_SHAPE, 1 /*num_tiles_per_pack*/);
        PROFILER_SYNC();
    }
    {
        ZONE_SCOPED("TILE_LOOP")
        if constexpr (PERF_RUN_TYPE == PerfRunType::MATH_ISOLATE || PERF_RUN_TYPE == PerfRunType::UNPACK_ISOLATE)
        {
        }
        else
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; loop++)
            {
                for (std::uint32_t i = 0; i < OUTPUT_TILE_CNT; ++i)
                {
                    _llk_pack_(i, i, ckernel::DEFAULT_TENSOR_SHAPE);
                }
                if constexpr (PERF_RUN_TYPE != PerfRunType::PACK_ISOLATE && PERF_RUN_TYPE != PerfRunType::L1_CONGESTION)
                {
                    _llk_pack_dest_dvalid_section_done_<dest_sync, is_fp32_dest_acc_en>();
                }
            }
        }
        PROFILER_SYNC();
    }
}

#endif
