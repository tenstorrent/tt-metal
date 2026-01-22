// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cstdint>
#include <cstdio>

#include "ckernel.h"
#include "llk_defs.h"

#ifdef LLK_TRISC_UNPACK

#include "llk_unpack_common.h"
#include "llk_unpack_reduce.h"
#include "params.h"

void run_kernel(const volatile struct RuntimeParams *params)
{
    tdma_descriptor_t td_val_A;
    tdma_descriptor_t td_val_B;
    const uint buf_desc_id_a = 0;
    const uint buf_desc_id_b = 1;

    // Setup data valid scheme
    set_up_dest_dvalid_per_thread<dest_dvalid_client::UNPACK>({dest_dvalid_client::FPU, dest_dvalid_client::PACK});

    buffer_descriptor_u bd_val_A = {0};
    buffer_descriptor_u bd_val_B = {0};

    bd_val_A.f.l1_addr_16B = buffer_A[0] / 16;
    bd_val_A.f.format      = static_cast<uint8_t>(formats.unpack_src);
    bd_val_A.f.x_dim       = params->TEST_FACE_C_DIM;
    bd_val_A.f.y_dim       = params->TEST_FACE_R_DIM;
    bd_val_A.f.z_dim       = params->num_faces;

    td_val_A.buf_desc        = bd_val_A;
    td_val_A.buf_desc_id     = buf_desc_id_a;
    td_val_A.reg_data_format = static_cast<uint8_t>(formats.unpack_dst);

    bd_val_B.f.l1_addr_16B = buffer_B[0] / 16;
    bd_val_B.f.format      = static_cast<uint8_t>(formats.unpack_src);
    bd_val_B.f.x_dim       = params->TEST_FACE_C_DIM;
    bd_val_B.f.y_dim       = params->TEST_FACE_R_DIM;
    bd_val_B.f.z_dim       = params->num_faces;

    td_val_B.buf_desc        = bd_val_B;
    td_val_B.buf_desc_id     = buf_desc_id_b;
    td_val_B.reg_data_format = static_cast<uint8_t>(formats.unpack_dst);

    TileShape tile_shape_A = {
        .num_faces = params->num_faces, .face_r_dim = params->TEST_FACE_R_DIM, .face_c_dim = params->TEST_FACE_C_DIM, .narrow_tile = false};

    _configure_buf_desc_table_(td_val_A.buf_desc_id, td_val_A.buf_desc);
    _configure_buf_desc_table_(td_val_B.buf_desc_id, td_val_B.buf_desc);
    _llk_unpack_configure_binary_<p_unpacr::UNP_A, p_unpacr::UNP_B>(td_val_A, td_val_B);
    _llk_unpack_reduce_init_<REDUCE_DIM>(buf_desc_id_a, buf_desc_id_b, 1 /*num_tiles_per_pack*/, tile_shape_A);
    for (int i = 0; i < params->TILE_CNT; ++i)
    {
        _llk_unpack_reduce_(i, 0);
    }
}

#endif

#ifdef LLK_TRISC_MATH

#include "llk_math_common.h"
#include "llk_math_reduce.h"
#include "params.h"

using namespace ckernel;

void run_kernel(const volatile struct RuntimeParams *params)
{
    // Setup data valid scheme
    set_up_dest_dvalid_per_thread<dest_dvalid_client::FPU>({dest_dvalid_client::FPU, dest_dvalid_client::PACK});

    TileShape tile_shape_A = {
        .num_faces = params->num_faces, .face_r_dim = params->TEST_FACE_R_DIM, .face_c_dim = params->TEST_FACE_C_DIM, .narrow_tile = false};

    DataFormat src_format = static_cast<DataFormat>(formats.math);
    _llk_math_srcAB_hw_configure_<IMPLIED_MATH_FORMAT, is_fp32_dest_acc_en, false /* int32 dest */>(src_format, src_format);

    _llk_math_reduce_init_<POOL_TYPE, REDUCE_DIM, static_cast<MathFidelity>(MATH_FIDELITY)>(tile_shape_A);
    for (int i = 0; i < params->TILE_CNT; ++i)
    {
        _llk_math_reduce_(i);
    }
    _llk_math_set_dvalid_<p_cleardvalid::FPU>();
}

#endif

#ifdef LLK_TRISC_PACK

#include "llk_pack.h"
#include "llk_pack_common.h"
#include "params.h"

void run_kernel(const volatile struct RuntimeParams *params)
{
    uint32_t const buf_desc_id = 8;

    set_up_dest_dvalid_per_thread<dest_dvalid_client::PACK>({dest_dvalid_client::FPU, dest_dvalid_client::PACK});

    buffer_descriptor_u bd_val = {0};
    tdma_descriptor_t tdma_desc;

    bd_val.f.l1_addr_16B = buffer_Res[0] / 16;
    bd_val.f.format      = static_cast<uint8_t>(formats.pack_dst);
    bd_val.f.x_dim       = params->TEST_FACE_C_DIM;
    bd_val.f.y_dim       = params->TEST_FACE_R_DIM;
    bd_val.f.z_dim       = params->num_faces;

    tdma_desc.buf_desc        = bd_val;
    tdma_desc.buf_desc_id     = buf_desc_id;
    tdma_desc.reg_data_format = static_cast<uint8_t>(formats.pack_src);

    _configure_buf_desc_table_(tdma_desc.buf_desc_id, tdma_desc.buf_desc);
    _llk_pack_hw_configure_<p_pacr::PACK0>(tdma_desc);
    _llk_pack_init_<p_pacr::PACK0>(buf_desc_id, 1 /*num_tiles_per_pack*/);
    _llk_pack_reduce_mask_config_<REDUCE_DIM>();
    for (int i = 0; i < params->TILE_CNT; ++i)
    {
        _llk_pack_<p_pacr::PACK0>(i, i);
    }
    _llk_pack_dest_dvalid_section_done_<dest_sync, is_fp32_dest_acc_en>();
    _llk_pack_reduce_mask_clear_();
}
#endif
