// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cstdint>
#include <cstdio>

#include "ckernel.h"
#include "llk_defs.h"

// Globals
uint32_t unp_cfg_context          = 0;
uint32_t pack_sync_tile_dst_ptr   = 0;
uint32_t math_sync_tile_dst_index = 0;

// Buffer descriptor IDs for TDMA engines - these are indices into the hardware buffer descriptor table
constexpr uint32_t BUF_DESC_ID_SRC_A = 29; // Source A matrix input buffer
constexpr uint32_t BUF_DESC_ID_SRC_B = 30; // Source B matrix input buffer
constexpr uint32_t BUF_DESC_ID_DST   = 31; // Destination matrix output buffer

#ifdef LLK_TRISC_UNPACK

#include "llk_unpack_matmul.h"
#include "params.h"

void run_kernel()
{
    // Setup sync for unpack
    set_up_dest_dvalid_per_thread<dest_dvalid_client::UNPACK>({dest_dvalid_client::FPU, dest_dvalid_client::PACK});
    set_ttsync_enables<TRACK_ALL>(ckernel::unpack::TRISC_ID);
    // src A input configuration
    tdma_descriptor_t tdma_desc_src_a;
    tdma_desc_src_a.buf_desc.f.l1_addr_16B  = L1_ADDRESS(buffer_A[0]);
    tdma_desc_src_a.buf_desc.f.format       = static_cast<uint8_t>(formats.unpack_src);
    tdma_desc_src_a.buf_desc.f.lmt_addr_16B = 0;
    tdma_desc_src_a.buf_desc.f.x_dim        = FACE_C_DIM;  // Default face dimension is 16, tiny tiles not supported for quasar
    tdma_desc_src_a.buf_desc.f.y_dim        = FACE_R_DIM;  // Default face dimension is 16, tiny tiles not supported for quasar
    tdma_desc_src_a.buf_desc.f.z_dim        = num_faces_A; // Number of faces = 4, tiny tiles not supported for quasar
    tdma_desc_src_a.buf_desc_id             = BUF_DESC_ID_SRC_A;
    tdma_desc_src_a.reg_data_format         = (uint)formats.unpack_dst;

    // src B input configuration
    tdma_descriptor_t tdma_desc_src_b;
    tdma_desc_src_b.buf_desc.f.l1_addr_16B  = L1_ADDRESS(buffer_B[0]);
    tdma_desc_src_b.buf_desc.f.format       = static_cast<uint8_t>(formats.unpack_src);
    tdma_desc_src_b.buf_desc.f.lmt_addr_16B = 0;
    tdma_desc_src_b.buf_desc.f.x_dim        = FACE_C_DIM;  // Default face dimension is 16, tiny tiles not supported for quasar
    tdma_desc_src_b.buf_desc.f.y_dim        = FACE_R_DIM;  // Default face dimension is 16, tiny tiles not supported for quasar
    tdma_desc_src_b.buf_desc.f.z_dim        = num_faces_B; // Number of faces = 4, tiny tiles not supported for quasar
    tdma_desc_src_b.buf_desc_id             = BUF_DESC_ID_SRC_B;
    tdma_desc_src_b.reg_data_format         = (uint)formats.unpack_dst;

    _llk_unpack_hw_configure_<ckernel::p_unpacr::UNP_B>(tdma_desc_src_a);
    _llk_unpack_hw_configure_<ckernel::p_unpacr::UNP_A>(tdma_desc_src_b);

    _llk_unpack_matmul_init_<BUF_DESC_ID_SRC_A, BUF_DESC_ID_SRC_B, UNPACK_TRANSPOSE_FACES, CT_DIM, RT_DIM, KT_DIM>(); // transpose in src_A not supported for
                                                                                                                      // quasar

    for (uint32_t j = 0; j < KT_DIM; j++)
    {
        _llk_unpack_matmul_<BUF_DESC_ID_SRC_A, BUF_DESC_ID_SRC_B, CT_DIM, RT_DIM, KT_DIM>(j, j * CT_DIM);
    }
}

#endif

#ifdef LLK_TRISC_MATH

#include "llk_math_common.h"
#include "llk_math_matmul.h"
#include "params.h"

void run_kernel()
{
    set_up_dest_dvalid_per_thread<dest_dvalid_client::FPU>({dest_dvalid_client::FPU, dest_dvalid_client::PACK});

    _llk_math_srcAB_hw_configure_<
        IMPLIED_MATH_FORMAT,
        is_fp32_dest_acc_en,
        false,
        static_cast<DataFormat>(formats.math),
        static_cast<DataFormat>(formats.math)>();
    _llk_math_matmul_init_<(ckernel::MathFidelity)MATH_FIDELITY, CT_DIM, RT_DIM, false, false>(); // disable flags for matmul with indexing and mxfp_2x not part
                                                                                                  // of P0 test suite

    for (uint32_t i = 0; i < KT_DIM; i++)
    {
        _llk_math_matmul_block_<CT_DIM, RT_DIM>();
    }
    _llk_math_set_dvalid_<p_cleardvalid::FPU>();
}

#endif

#ifdef LLK_TRISC_PACK

#include "llk_pack.h"
#include "llk_pack_matmul.h"
#include "params.h"

void run_kernel()
{
    set_up_dest_dvalid_per_thread<dest_dvalid_client::PACK>({dest_dvalid_client::FPU, dest_dvalid_client::PACK});

    tdma_descriptor_t tdma_desc_dst;
    tdma_desc_dst.buf_desc.f.l1_addr_16B  = L1_ADDRESS(buffer_Res[0]);
    tdma_desc_dst.buf_desc.f.lmt_addr_16B = 0;
    tdma_desc_dst.buf_desc.f.format       = static_cast<uint8_t>(formats.pack_dst);
    tdma_desc_dst.buf_desc.f.x_dim        = FACE_C_DIM;
    tdma_desc_dst.buf_desc.f.y_dim        = FACE_R_DIM;
    tdma_desc_dst.buf_desc.f.z_dim        = num_faces;
    tdma_desc_dst.buf_desc_id             = BUF_DESC_ID_DST;
    tdma_desc_dst.reg_data_format         = static_cast<uint8_t>(formats.pack_src);

    _llk_pack_hw_configure_<p_pacr::PACK0>(tdma_desc_dst);
    _llk_pack_matmul_init_<p_pacr::PACK0, BUF_DESC_ID_DST, RT_DIM, CT_DIM, 1>(); // Use destination buffer descriptor for packing output

    _llk_pack_matmul_<p_pacr::PACK0>(0, 0);
    _llk_pack_dest_dvalid_section_done_<dest_sync, is_fp32_dest_acc_en>();
}

#endif
