// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <cstdio>

#include "ckernel.h"
#include "llk_defs.h"
#include "llk_memory_checks.h"
#include "sfpu_stub.h"

#ifdef LLK_TRISC_UNPACK

#include "llk_unpack_common.h"
#include "llk_unpack_tilize_operands.h"
#include "params.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    const std::uint32_t buf_desc_id_a = 0;
    const std::uint32_t buf_desc_id_b = 1;

    set_up_dest_dvalid_per_thread<dest_dvalid_client::UNPACK>({dest_dvalid_client::FPU, dest_dvalid_client::PACK});

    buffer_descriptor_u bd_val_A {};
    bd_val_A.f.l1_addr_16B = L1_ADDRESS(params.buffer_A[0]);
    bd_val_A.f.format      = static_cast<std::uint8_t>(formats.unpack_A_src);
    bd_val_A.f.x_dim       = TEST_FACE_C_DIM;
    bd_val_A.f.y_dim       = TEST_FACE_R_DIM;
    bd_val_A.f.z_dim       = num_faces;

    tdma_descriptor_t td_val_A;
    td_val_A.buf_desc        = bd_val_A;
    td_val_A.buf_desc_id     = buf_desc_id_a;
    td_val_A.reg_data_format = static_cast<std::uint8_t>(formats.unpack_A_dst);

    buffer_descriptor_u bd_val_B {};
    bd_val_B.f.l1_addr_16B = L1_ADDRESS(params.buffer_B[0]);
    bd_val_B.f.format      = static_cast<std::uint8_t>(formats.unpack_B_src);
    bd_val_B.f.x_dim       = TEST_FACE_C_DIM;
    bd_val_B.f.y_dim       = TEST_FACE_R_DIM;
    bd_val_B.f.z_dim       = num_faces;

    tdma_descriptor_t td_val_B;
    td_val_B.buf_desc        = bd_val_B;
    td_val_B.buf_desc_id     = buf_desc_id_b;
    td_val_B.reg_data_format = static_cast<std::uint8_t>(formats.unpack_B_dst);

    _configure_buf_desc_table_(td_val_A.buf_desc_id, td_val_A.buf_desc);
    _configure_buf_desc_table_(td_val_B.buf_desc_id, td_val_B.buf_desc);
    _llk_unpack_configure_binary_<p_unpacr::UNP_A, p_unpacr::UNP_B>(td_val_A, td_val_B);

    constexpr ckernel::TensorShape tensor_shape = ckernel::DEFAULT_TENSOR_SHAPE;
    _llk_unpack_tilize_operands_init_<TILIZE_UNP_SEL>(buf_desc_id_a, buf_desc_id_b, FULL_CT_DIM, tensor_shape);

    const std::uint32_t y_stride_external = FULL_CT_DIM * tensor_shape.num_faces_r_dim * tensor_shape.face_r_dim;
    for (std::uint32_t block_rt = 0; block_rt < BLOCK_RT_DIM; block_rt++)
    {
        std::uint32_t offset = block_rt * y_stride_external;
        for (std::uint32_t block_ct = 0; block_ct < BLOCK_CT_DIM; block_ct++)
        {
            const std::uint32_t l1_unpack_tilize_idx = offset + block_ct;
            const std::uint32_t l1_unpack_idx        = block_rt * BLOCK_CT_DIM + block_ct;
            if constexpr (TILIZE_UNP_SEL == TilizeUnpackerSel::UnpA)
            {
                _llk_unpack_tilize_operands_<TILIZE_UNP_SEL>(l1_unpack_tilize_idx, l1_unpack_idx);
            }
            else if constexpr (TILIZE_UNP_SEL == TilizeUnpackerSel::UnpB)
            {
                _llk_unpack_tilize_operands_<TILIZE_UNP_SEL>(l1_unpack_idx, l1_unpack_tilize_idx);
            }
            else // UnpAB: both operands are tilized
            {
                _llk_unpack_tilize_operands_<TILIZE_UNP_SEL>(l1_unpack_tilize_idx, l1_unpack_tilize_idx);
            }
        }
    }
}

#endif

#ifdef LLK_TRISC_MATH

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
    set_up_dest_dvalid_per_thread<dest_dvalid_client::FPU>({dest_dvalid_client::FPU, dest_dvalid_client::PACK});

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

    _llk_math_eltwise_binary_init_<ELTWISE_BINARY_OP, MATH_FIDELITY>(ckernel::DEFAULT_TENSOR_SHAPE, false /*acc_to_dest*/);

    for (std::uint32_t i = 0; i < TILE_CNT; ++i)
    {
        _llk_math_eltwise_binary_<ELTWISE_BINARY_OP>(i, ckernel::DEFAULT_TENSOR_SHAPE);
    }

    _llk_math_set_dvalid_<p_cleardvalid::FPU, dest_sync>();
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
    std::uint32_t const buf_desc_id        = 8;
    const std::uint32_t num_tiles_per_pack = TILE_CNT;

    set_up_dest_dvalid_per_thread<dest_dvalid_client::PACK>({dest_dvalid_client::FPU, dest_dvalid_client::PACK});

    buffer_descriptor_u bd_val = {0};
    bd_val.f.l1_addr_16B       = L1_ADDRESS(params.buffer_Res[0]);
    bd_val.f.format            = static_cast<std::uint8_t>(formats.pack_dst);
    bd_val.f.x_dim             = TEST_FACE_C_DIM;
    bd_val.f.y_dim             = TEST_FACE_R_DIM;
    bd_val.f.z_dim             = num_faces;

    tdma_descriptor_t tdma_desc;
    tdma_desc.buf_desc        = bd_val;
    tdma_desc.buf_desc_id     = buf_desc_id;
    tdma_desc.reg_data_format = static_cast<std::uint8_t>(formats.pack_src);

    _configure_buf_desc_table_(tdma_desc.buf_desc_id, tdma_desc.buf_desc);
    _llk_pack_hw_configure_<p_pacr::PACK0>(tdma_desc);

    _llk_pack_init_(buf_desc_id, ckernel::DEFAULT_TENSOR_SHAPE, num_tiles_per_pack);
    _llk_pack_(0, 0, ckernel::DEFAULT_TENSOR_SHAPE);
    _llk_pack_dest_dvalid_section_done_<dest_sync, is_fp32_dest_acc_en>();
}

#endif
