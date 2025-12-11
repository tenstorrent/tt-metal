// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <cstdio>

#include "ckernel.h"
#include "llk_defs.h"

#ifdef LLK_TRISC_UNPACK

#include "llk_unpack_binary_broadcast_operands.h"
#include "llk_unpack_common.h"
#include "params.h"

void run_kernel()
{
    tdma_descriptor_t td_val_A, td_val_B;
    const uint BUF_DESC_ID_A        = 0;
    const uint BUF_DESC_ID_B        = 1;
    const uint num_tiles_per_unpack = TILE_CNT;

    // Setup data valid scheme
    set_up_dest_dvalid_per_thread<dest_dvalid_client::UNPACK>({dest_dvalid_client::FPU, dest_dvalid_client::PACK});

    // Configure Source A buffer descriptor
    buffer_descriptor_u bd_val_A = {0};
    bd_val_A.f.l1_addr_16B       = buffer_A[0] / 16;
    bd_val_A.f.format            = static_cast<uint8_t>(formats.unpack_src);
    bd_val_A.f.x_dim             = TEST_FACE_C_DIM;
    bd_val_A.f.y_dim             = TEST_FACE_R_DIM;
    bd_val_A.f.z_dim             = num_faces;

    td_val_A.buf_desc        = bd_val_A;
    td_val_A.buf_desc_id     = BUF_DESC_ID_A;
    td_val_A.reg_data_format = static_cast<uint8_t>(formats.unpack_dst);

    // Configure Source B buffer descriptor
    buffer_descriptor_u bd_val_B = {0};
    bd_val_B.f.l1_addr_16B       = buffer_B[0] / 16;
    bd_val_B.f.format            = static_cast<uint8_t>(formats.unpack_src);
    bd_val_B.f.x_dim             = TEST_FACE_C_DIM;
    bd_val_B.f.y_dim             = TEST_FACE_R_DIM;
    bd_val_B.f.z_dim             = num_faces;

    td_val_B.buf_desc        = bd_val_B;
    td_val_B.buf_desc_id     = BUF_DESC_ID_B;
    td_val_B.reg_data_format = static_cast<uint8_t>(formats.unpack_dst);

    _llk_unpack_configure_binary_<p_unpacr::UNP_A, p_unpacr::UNP_B>(td_val_A, td_val_B);
    _llk_unpack_binary_broadcast_operands_init_<BUF_DESC_ID_A, BUF_DESC_ID_B, BROADCAST_TYPE>(num_tiles_per_unpack);
    _llk_unpack_binary_broadcast_operands_(0, 0);
}

#endif

#ifdef LLK_TRISC_MATH

#include "llk_math_common.h"
#include "llk_math_eltwise_binary_broadcast.h"
#include "params.h"

using namespace ckernel;

void run_kernel()
{
    set_up_dest_dvalid_per_thread<dest_dvalid_client::FPU>({dest_dvalid_client::FPU, dest_dvalid_client::PACK});

    constexpr DataFormat src_format = static_cast<DataFormat>(formats.math);
    _llk_math_srcAB_hw_configure_<IMPLIED_MATH_FORMAT, is_fp32_dest_acc_en, false /*int32_dest*/, src_format, src_format>();

    TileShape tile_shape = {.num_faces = num_faces, .face_r_dim = TEST_FACE_R_DIM, .face_c_dim = TEST_FACE_C_DIM, .narrow_tile = false};
    _llk_math_eltwise_binary_broadcast_init_<ELTWISE_BINARY_OP, BROADCAST_TYPE, static_cast<MathFidelity>(MATH_FIDELITY)>(tile_shape);

    for (int i = 0; i < TILE_CNT; ++i)
    {
        _llk_math_eltwise_binary_broadcast_(i);
    }
    _llk_math_set_dvalid_<p_cleardvalid::FPU>();
}

#endif

#ifdef LLK_TRISC_PACK

#include "llk_pack.h"
#include "llk_pack_common.h"
#include "params.h"

void run_kernel()
{
    uint32_t const BUF_DESC       = 8;
    const uint num_tiles_per_pack = TILE_CNT;

    set_up_dest_dvalid_per_thread<dest_dvalid_client::PACK>({dest_dvalid_client::FPU, dest_dvalid_client::PACK});

    buffer_descriptor_u bd_val = {0};
    bd_val.f.l1_addr_16B       = buffer_Res[0] / 16;
    bd_val.f.format            = static_cast<uint8_t>(formats.pack_dst);
    bd_val.f.x_dim             = TEST_FACE_C_DIM;
    bd_val.f.y_dim             = TEST_FACE_R_DIM;
    bd_val.f.z_dim             = num_faces;

    tdma_descriptor_t tdma_desc;

    tdma_desc.buf_desc        = bd_val;
    tdma_desc.buf_desc_id     = BUF_DESC;
    tdma_desc.reg_data_format = static_cast<uint8_t>(formats.pack_src);

    _llk_pack_hw_configure_<p_pacr::PACK0>(tdma_desc);
    _llk_pack_init_<p_pacr::PACK0, BUF_DESC>(num_tiles_per_pack);
    _llk_pack_<p_pacr::PACK0>(0, 0);
    _llk_pack_dest_dvalid_section_done_<dest_sync, is_fp32_dest_acc_en>();
}

#endif
