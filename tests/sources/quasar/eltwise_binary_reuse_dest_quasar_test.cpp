// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Test for eltwise binary operations with reuse_dest on Quasar.

#include <cstdint>
#include <cstdio>

#include "ckernel.h"
#include "llk_defs.h"
#include "llk_memory_checks.h"
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
    tdma_descriptor_t td_val_A, td_val_B;
    const std::uint32_t buf_desc_id_a = 0;
    const std::uint32_t buf_desc_id_b = 1;
    constexpr bool TRANSPOSE_EN       = false;
    constexpr bool IS_32B_DEST_EN     = false;

    // Setup data valid scheme
    set_up_dest_dvalid_per_thread<dest_dvalid_client::UNPACK>({dest_dvalid_client::FPU, dest_dvalid_client::PACK});

    // Configure Source A buffer descriptor — BD base points to tile 0;
    // tile_idx=i in each unpack call offsets automatically: buffer_A[0] + i*stride = buffer_A[i]
    buffer_descriptor_u bd_val_A {};
    bd_val_A.f.l1_addr_16B = L1_ADDRESS(params.buffer_A[0]);
    bd_val_A.f.format      = static_cast<std::uint8_t>(formats.unpack_A_src);
    bd_val_A.f.x_dim       = params.TEST_FACE_C_DIM;
    bd_val_A.f.y_dim       = params.TEST_FACE_R_DIM;
    bd_val_A.f.z_dim       = params.num_faces;

    td_val_A.buf_desc        = bd_val_A;
    td_val_A.buf_desc_id     = buf_desc_id_a;
    td_val_A.reg_data_format = static_cast<std::uint8_t>(formats.unpack_A_dst);

    // Configure Source B buffer descriptor — BD base points to tile 0
    buffer_descriptor_u bd_val_B {};
    bd_val_B.f.l1_addr_16B = L1_ADDRESS(params.buffer_B[0]);
    bd_val_B.f.format      = static_cast<std::uint8_t>(formats.unpack_B_src);
    bd_val_B.f.x_dim       = params.TEST_FACE_C_DIM;
    bd_val_B.f.y_dim       = params.TEST_FACE_R_DIM;
    bd_val_B.f.z_dim       = params.num_faces;

    td_val_B.buf_desc        = bd_val_B;
    td_val_B.buf_desc_id     = buf_desc_id_b;
    td_val_B.reg_data_format = static_cast<std::uint8_t>(formats.unpack_B_dst);

    _configure_buf_desc_table_(td_val_A.buf_desc_id, td_val_A.buf_desc);
    _configure_buf_desc_table_(td_val_B.buf_desc_id, td_val_B.buf_desc);
    _llk_unpack_configure_binary_<p_unpacr::UNP_A, p_unpacr::UNP_B>(td_val_A, td_val_B);

    _llk_unpack_unary_operand_init_<p_unpacr::UNP_A, TRANSPOSE_EN, IS_32B_DEST_EN>(buf_desc_id_a, 1);
    for (int i = 0; i < params.INPUT_TILE_CNT; ++i)
    {
        _llk_unpack_unary_operand_<p_unpacr::UNP_A>(i);
    }

    constexpr std::uint32_t buf_desc_id_phase2 = (REUSE_DEST_TYPE == EltwiseBinaryReuseDestType::DEST_TO_SRCB) ? buf_desc_id_a : buf_desc_id_b;
    // Phase 2: unpack the other operand. DEST_TO_SRCA → SrcB from buffer B (UNP_B); DEST_TO_SRCB → SrcA from buffer A (UNP_A).
    constexpr std::uint32_t unp_sel_phase2 = (REUSE_DEST_TYPE == EltwiseBinaryReuseDestType::DEST_TO_SRCA) ? p_unpacr::UNP_B : p_unpacr::UNP_A;
    _llk_unpack_unary_operand_init_<unp_sel_phase2, TRANSPOSE_EN, IS_32B_DEST_EN, REUSE_DEST_TYPE>(buf_desc_id_phase2, 1, params.num_faces);
    for (int i = 0; i < params.INPUT_TILE_CNT; ++i)
    {
        _llk_unpack_unary_operand_<unp_sel_phase2, REUSE_DEST_TYPE>(i);
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
    set_up_dest_dvalid_per_thread<dest_dvalid_client::FPU>({dest_dvalid_client::FPU, dest_dvalid_client::PACK});

    DataFormat src_format = static_cast<DataFormat>(formats.math);
    _llk_math_srcAB_hw_configure_<IMPLIED_MATH_FORMAT, is_fp32_dest_acc_en, false /*int32*/>(src_format, src_format);

    const int num_total_tiles = params.INPUT_NUM_TILES_IN_BLOCK * params.INPUT_NUM_BLOCKS;
    const int tiles_in_block  = params.OUTPUT_NUM_TILES_IN_BLOCK;
    const int num_tiles_accum = params.INPUT_NUM_TILES_IN_BLOCK / tiles_in_block;
    const int num_blocks      = params.INPUT_NUM_BLOCKS;

    // Datacopy src_A from SrcA to DEST (all input tiles)
    _llk_math_eltwise_unary_datacopy_init_<DataCopyType::A2D, is_fp32_dest_acc_en>(params.num_faces * params.TEST_FACE_R_DIM, 1);
    for (int i = 0; i < num_total_tiles; ++i)
    {
        _llk_math_eltwise_unary_datacopy_(params.num_faces * params.TEST_FACE_R_DIM, i);
    }

    // Binary with reuse_dest: SrcA = DEST (from datacopy), SrcB = unpacked B. Compute op(SrcA, SrcB) -> DEST.
    _llk_math_eltwise_binary_init_<ELTWISE_BINARY_OP, MATH_FIDELITY, false /*EN_DI*/, REUSE_DEST_TYPE>(
        ckernel::DEFAULT_TENSOR_SHAPE); // tiny-tile testing not yet supported

    for (int block = 0; block < num_blocks; block++)
    {
        for (int n = 0; n < num_tiles_accum; n++)
        {
            for (int tile = 0; tile < tiles_in_block; tile++)
            {
                const int global_tile_idx = block * tiles_in_block + tile;
                _llk_math_eltwise_binary_<REUSE_DEST_TYPE>(global_tile_idx, params.num_faces);
            }
        }
        _llk_math_set_dvalid_<p_cleardvalid::FPU>();
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
    std::uint32_t const buf_desc_id = 8;

    set_up_dest_dvalid_per_thread<dest_dvalid_client::PACK>({dest_dvalid_client::FPU, dest_dvalid_client::PACK});

    buffer_descriptor_u bd_val {};
    bd_val.f.l1_addr_16B = L1_ADDRESS(params.buffer_Res[0]);
    bd_val.f.format      = static_cast<std::uint8_t>(formats.pack_dst);
    bd_val.f.x_dim       = params.TEST_FACE_C_DIM;
    bd_val.f.y_dim       = params.TEST_FACE_R_DIM;
    bd_val.f.z_dim       = params.num_faces;

    tdma_descriptor_t tdma_desc;
    tdma_desc.buf_desc        = bd_val;
    tdma_desc.buf_desc_id     = buf_desc_id;
    tdma_desc.reg_data_format = static_cast<std::uint8_t>(formats.pack_src);

    _configure_buf_desc_table_(tdma_desc.buf_desc_id, tdma_desc.buf_desc);
    _llk_pack_hw_configure_<p_pacr::PACK0>(tdma_desc);
    _llk_pack_init_<p_pacr::PACK0>(buf_desc_id, 1);

    const int output_tiles_in_block = params.OUTPUT_NUM_TILES_IN_BLOCK;
    const int output_num_blocks     = params.OUTPUT_NUM_BLOCKS;

    for (int block = 0; block < output_num_blocks; block++)
    {
        for (int tile = 0; tile < output_tiles_in_block; tile++)
        {
            int res_tile_idx = (block * output_tiles_in_block) + tile;
            _llk_pack_<p_pacr::PACK0>(res_tile_idx, res_tile_idx);
        }
        _llk_pack_dest_dvalid_section_done_<dest_sync, is_fp32_dest_acc_en>();
    }
}

#endif
