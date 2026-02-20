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
#include "llk_unpack_tilize.h"
#include "params.h"

void run_kernel(const volatile struct RuntimeParams* params)
{
    tdma_descriptor_t td_val;
    const std::uint32_t buf_desc_id = 0;

    // Setup data valid scheme
    set_up_dest_dvalid_per_thread<dest_dvalid_client::UNPACK>({dest_dvalid_client::FPU, dest_dvalid_client::PACK});

    buffer_descriptor_u bd_val = {0};

    unsigned l1_addr_16B;
    if constexpr (UNPACKER_ENGINE_SEL == p_unpacr::UNP_A || UNPACKER_ENGINE_SEL == p_unpacr::UNP_DEST)
    {
        l1_addr_16B = params->buffer_A[0] / 16;
    }
    else if constexpr (UNPACKER_ENGINE_SEL == p_unpacr::UNP_B)
    {
        l1_addr_16B = params->buffer_B[0] / 16;
    }

    bd_val.f.l1_addr_16B = l1_addr_16B;
    bd_val.f.format      = static_cast<std::uint8_t>(formats.unpack_A_src);
    bd_val.f.x_dim       = TEST_FACE_C_DIM;
    bd_val.f.y_dim       = TEST_FACE_R_DIM;
    bd_val.f.z_dim       = num_faces;

    td_val.buf_desc        = bd_val;
    td_val.buf_desc_id     = buf_desc_id;
    td_val.reg_data_format = static_cast<std::uint8_t>(formats.unpack_A_dst);

    constexpr TileShape tile_shape      = {.num_faces = num_faces, .face_r_dim = TEST_FACE_R_DIM, .face_c_dim = TEST_FACE_C_DIM, .narrow_tile = 0};
    constexpr std::uint32_t C_DIM_FACES = (tile_shape.narrow_tile ? 1 : 2);                    // Tile width in faces
    constexpr std::uint32_t R_DIM_FACES = (num_faces == 2 && !tile_shape.narrow_tile) ? 1 : 2; // Tile height in faces

    _configure_buf_desc_table_(td_val.buf_desc_id, td_val.buf_desc);
    if (is_fp32_dest_acc_en)
    {
        // If Dst fmt is 32b and operation is Mov2D, we need both SrcA/B fmts to be configured since Mov2D will be implemented via ELWADD
        _llk_unpack_configure_binary_<p_unpacr::UNP_A, p_unpacr::UNP_B>(td_val, td_val);
    }
    else
    {
        _llk_unpack_configure_unary_<UNPACKER_ENGINE_SEL>(td_val);
    }
    _llk_unpack_tilize_init_<UNPACKER_ENGINE_SEL, is_fp32_dest_acc_en, FULL_CT_DIM, BLOCK_CT_DIM, C_DIM_FACES>(buf_desc_id);

    // One _llk_unpack_tilize_ call unpacks one block ct_dim of tiles (one tile row)
    // The internal parts of the strides are applied inside of the _llk_ itself, the external parts are passed to the _llk_unpack_tilize_ call
    // x_stride = x_stride_internal = col dim of a tile in L1 in units of 16 datums (1 face);
    // y_stride = y_stride_external + x_stride_internal
    // In this case x = 0 because the entire tile row fits into Dest
    std::uint32_t y_stride_external = FULL_CT_DIM * R_DIM_FACES * TEST_FACE_R_DIM;
    for (std::uint32_t y = 0; y < BLOCK_RT_DIM; y++)
    {
        _llk_unpack_tilize_<UNPACKER_ENGINE_SEL>(y * y_stride_external /*  + 0 * x_stride  */);
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
#include "llk_math_eltwise_unary_datacopy.h"
#include "params.h"

using namespace ckernel;

void run_kernel(const volatile struct RuntimeParams*)
{
    set_up_dest_dvalid_per_thread<dest_dvalid_client::FPU>({dest_dvalid_client::FPU, dest_dvalid_client::PACK});

    DataFormat src_format = static_cast<DataFormat>(formats.math);
    _llk_math_srcAB_hw_configure_<IMPLIED_MATH_FORMAT, is_fp32_dest_acc_en, is_int_fpu_en>(src_format, src_format);

    _llk_math_eltwise_unary_datacopy_init_<DATA_COPY_TYPE, is_fp32_dest_acc_en>(num_faces * TEST_FACE_R_DIM /*num_rows_per_matrix*/, 1 /*num_matrices*/);
    for (int i = 0; i < TILE_CNT; ++i)
    {
        _llk_math_eltwise_unary_datacopy_(num_faces * TEST_FACE_R_DIM /*num_rows_per_tile*/, i);
    }
    _llk_math_set_dvalid_<p_cleardvalid::FPU>();
}

#endif

#ifdef LLK_TRISC_PACK

#include "llk_pack.h"
#include "llk_pack_common.h"
#include "params.h"

void run_kernel(const volatile struct RuntimeParams* params)
{
    std::uint32_t const buf_desc_id        = 8;
    const std::uint32_t num_tiles_per_pack = TILE_CNT;

    set_up_dest_dvalid_per_thread<dest_dvalid_client::PACK>({dest_dvalid_client::FPU, dest_dvalid_client::PACK});

    buffer_descriptor_u bd_val = {0};
    bd_val.f.l1_addr_16B       = params->buffer_Res[0] / 16;
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
    _llk_pack_init_<p_pacr::PACK0>(buf_desc_id, num_tiles_per_pack);
    _llk_pack_<p_pacr::PACK0>(0, 0);
    _llk_pack_dest_dvalid_section_done_<dest_sync, is_fp32_dest_acc_en>();
}
#endif
