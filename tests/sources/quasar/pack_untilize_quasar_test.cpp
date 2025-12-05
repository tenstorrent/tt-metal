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
#include "llk_unpack_unary_operand.h"
#include "params.h"

void run_kernel()
{
    // Setup data valid scheme
    set_up_dest_dvalid_per_thread<dest_dvalid_client::UNPACK>({dest_dvalid_client::FPU, dest_dvalid_client::PACK});

    tdma_descriptor_t td_val;
    const uint BUF_DESC_ID          = 0;
    const uint num_tiles_per_unpack = TILE_CNT;

    buffer_descriptor_u bd_val = {0};

    bd_val.f.l1_addr_16B = buffer_A[0] / 16;
    bd_val.f.format      = static_cast<uint8_t>(formats.unpack_src);
    bd_val.f.x_dim       = TEST_FACE_C_DIM;
    bd_val.f.y_dim       = TEST_FACE_R_DIM;
    bd_val.f.z_dim       = num_faces;

    td_val.buf_desc        = bd_val;
    td_val.buf_desc_id     = BUF_DESC_ID;
    td_val.reg_data_format = static_cast<uint8_t>(formats.unpack_dst);

    _llk_unpack_configure_binary_<p_unpacr::UNP_A, p_unpacr::UNP_B>(td_val, td_val);
    _llk_unpack_unary_operand_init_<UNPACKER_ENGINE_SEL, BUF_DESC_ID, false /*transpose*/, is_fp32_dest_acc_en>(num_tiles_per_unpack);
    _llk_unpack_unary_operand_<UNPACKER_ENGINE_SEL>(0);
}

#endif

#ifdef LLK_TRISC_MATH

#include "llk_math_common.h"
#include "llk_math_eltwise_unary_datacopy.h"
#include "params.h"

using namespace ckernel;

void run_kernel()
{
#ifdef FORMAT_INT32
    const bool is_int_fpu_en = true;
#else
    const bool is_int_fpu_en = false;
#endif
    // Setup data valid scheme
    set_up_dest_dvalid_per_thread<dest_dvalid_client::FPU>({dest_dvalid_client::FPU, dest_dvalid_client::PACK});

    constexpr DataFormat src_format = static_cast<DataFormat>(formats.math);
    _llk_math_srcAB_hw_configure_<IMPLIED_MATH_FORMAT, is_fp32_dest_acc_en, is_int_fpu_en, src_format, src_format>();

    _llk_math_eltwise_unary_datacopy_init_<DataCopyType::A2D, is_fp32_dest_acc_en>(num_faces * TEST_FACE_R_DIM /*num_rows_per_matrix*/, 1 /*num_matrices*/);
    for (uint block_rt = 0; block_rt < BLOCK_RT_DIM; block_rt++)
    {
        for (uint block_ct = 0; block_ct < BLOCK_CT_DIM; block_ct++)
        {
            _llk_math_eltwise_unary_datacopy_<num_faces * TEST_FACE_R_DIM /*num_rows_per_tile*/>(block_ct);
        }
        _llk_math_set_dvalid_<p_cleardvalid::FPU>();
    }
}

#endif

#ifdef LLK_TRISC_PACK

#include "llk_pack_common.h"
#include "llk_pack_untilize.h"
#include "params.h"

void run_kernel()
{
    // Setup data valid scheme
    set_up_dest_dvalid_per_thread<dest_dvalid_client::PACK>({dest_dvalid_client::FPU, dest_dvalid_client::PACK});

    tdma_descriptor_t tdma_desc;
    uint32_t const BUF_DESC = 31;

    buffer_descriptor_u bd_val = {0};

    bd_val.f.l1_addr_16B = buffer_Res[0] / 16;
    bd_val.f.format      = static_cast<uint8_t>(formats.pack_dst);
    bd_val.f.x_dim       = TEST_FACE_C_DIM;
    bd_val.f.y_dim       = TEST_FACE_R_DIM;
    bd_val.f.z_dim       = num_faces;

    tdma_desc.buf_desc        = bd_val;
    tdma_desc.buf_desc_id     = BUF_DESC;
    tdma_desc.reg_data_format = static_cast<uint8_t>(formats.pack_src);

    constexpr TileShape tile_shape = {.num_faces = num_faces, .face_r_dim = TEST_FACE_R_DIM, .face_c_dim = TEST_FACE_C_DIM, .narrow_tile = 0};

    constexpr uint32_t C_DIM_FACES = (tile_shape.narrow_tile ? 1 : 2);                    // Tile width in faces
    constexpr uint32_t R_DIM_FACES = (num_faces == 2 && !tile_shape.narrow_tile) ? 1 : 2; // Tile height in faces

    _llk_pack_hw_configure_<p_pacr::PACK0>(tdma_desc);
    _llk_pack_untilize_init_<BUF_DESC, FULL_CT_DIM, BLOCK_CT_DIM, C_DIM_FACES>(tile_shape);

    // One _llk_pack_untilize_ call packs one block ct_dim of tiles (one tile row)
    // The internal parts of the strides are applied inside of the _llk_ itself, the external parts are passed to the _llk_pack_untilize_ call
    // x_stride = x_stride_internal = col dim of a tile in L1 in units of 16 datums (1 face);
    // y_stride = y_stride_external + x_stride_internal
    // In this case x = 0 because the entire tile row fits into Dest
    uint y_stride_external = FULL_CT_DIM * R_DIM_FACES * TEST_FACE_R_DIM;
    for (uint y = 0; y < BLOCK_RT_DIM; y++)
    {
        _llk_pack_untilize_(0, y * y_stride_external /*  + 0 * x_stride  */);
        _llk_pack_dest_dvalid_section_done_<dest_sync, is_fp32_dest_acc_en>();
    }
}

#endif
