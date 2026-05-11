// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Pack test for the MxFp4_2x_A/B datacopy path on Quasar.
//
// MOV MXFP4_2x was removed from Quasar RTL (TEN-3634, 2025-09-19): the FP4-2x sub-datum
// expansion is now gated on op_mmul, so SrcA(MxFp4_2x_A/B) -> Dest must go through a matmul.
// This kernel exercises that via _llk_math_eltwise_unary_datacopy_x2_, which wraps an
// EN_DI/EN_X2 matmul block with rt_dim=1, ct_dim=TILE_CNT and SrcB pre-loaded with an
// identity tile in the same 2x family. MVMULDI then yields Dest[i] = SrcA[i], dequantized
// into Float16 (A-family) or Float16_b (B-family).
//
// Buffer plumbing:
//   buffer_A : TILE_CNT data tiles in MxFp4    -> unpack -> SrcA (MxFp4_2x_A/B)
//   buffer_B : 1 identity tile in MxFp4         -> unpack -> SrcB (MxFp4_2x_A/B, matching A's family)
//   buffer_Res: TILE_CNT output tiles in pack_dst format

#include <algorithm>
#include <cstdint>
#include <cstdio>

#include "ckernel.h"
#include "llk_defs.h"
#include "llk_memory_checks.h"
#include "sfpu_stub.h"

// Naming follows the python side: buffer_A holds data tiles (-> SrcA register), buffer_B holds
// the single identity tile (-> SrcB register). _llk_unpack_matmul_init_'s argument order is
// (in0, in1) = (SrcB-bound, SrcA-bound); see llk_unpack_matmul.h:86-87 ("Input 0 -> SrcB,
// Input 1 -> SrcA"). UNP_A unpacker engine drives SrcA; UNP_B drives SrcB.
constexpr std::uint32_t buf_desc_id_data     = 29; // -> SrcA register (params.buffer_A)
constexpr std::uint32_t buf_desc_id_identity = 30; // -> SrcB register (params.buffer_B)
constexpr std::uint32_t buf_desc_id_dst      = 31; // output tiles in L1

#ifdef LLK_TRISC_UNPACK

#include "llk_unpack_matmul.h"
#include "params.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif

    set_up_dest_dvalid_per_thread<dest_dvalid_client::UNPACK>({dest_dvalid_client::FPU, dest_dvalid_client::PACK});
    set_ttsync_enables<TRACK_ALL>(ckernel::unpack::TRISC_ID);

    // Data-tile descriptor (buffer_A) -> SrcA register via UNP_A.
    tdma_descriptor_t tdma_desc_data;
    tdma_desc_data.buf_desc.f.l1_addr_16B  = L1_ADDRESS(params.buffer_A[0]);
    tdma_desc_data.buf_desc.f.format       = static_cast<std::uint8_t>(formats.unpack_A_src);
    tdma_desc_data.buf_desc.f.lmt_addr_16B = 0;
    tdma_desc_data.buf_desc.f.x_dim        = params.TEST_FACE_C_DIM;
    tdma_desc_data.buf_desc.f.y_dim        = params.TEST_FACE_R_DIM;
    tdma_desc_data.buf_desc.f.z_dim        = params.num_faces;
    tdma_desc_data.buf_desc_id             = buf_desc_id_data;
    tdma_desc_data.reg_data_format         = static_cast<std::uint32_t>(formats.unpack_A_dst);

    // Identity-tile descriptor (buffer_B) -> SrcB register via UNP_B.
    tdma_descriptor_t tdma_desc_identity;
    tdma_desc_identity.buf_desc.f.l1_addr_16B  = L1_ADDRESS(params.buffer_B[0]);
    tdma_desc_identity.buf_desc.f.format       = static_cast<std::uint8_t>(formats.unpack_B_src);
    tdma_desc_identity.buf_desc.f.lmt_addr_16B = 0;
    tdma_desc_identity.buf_desc.f.x_dim        = params.TEST_FACE_C_DIM;
    tdma_desc_identity.buf_desc.f.y_dim        = params.TEST_FACE_R_DIM;
    tdma_desc_identity.buf_desc.f.z_dim        = params.num_faces;
    tdma_desc_identity.buf_desc_id             = buf_desc_id_identity;
    tdma_desc_identity.reg_data_format         = static_cast<std::uint32_t>(formats.unpack_B_dst);

    _configure_buf_desc_table_(tdma_desc_data.buf_desc_id, tdma_desc_data.buf_desc);
    _configure_buf_desc_table_(tdma_desc_identity.buf_desc_id, tdma_desc_identity.buf_desc);
    // UNP_A engine drives SrcA register; UNP_B engine drives SrcB register.
    _llk_unpack_hw_configure_<ckernel::p_unpacr::UNP_A>(tdma_desc_data);
    _llk_unpack_hw_configure_<ckernel::p_unpacr::UNP_B>(tdma_desc_identity);

    // _llk_unpack_matmul_init_ arg order is (in0, in1) = (SrcB, SrcA): identity first, data second.
    // Single k-step, N=TILE_CNT data tiles on SrcA, 1 identity tile on SrcB reused across them.
    _llk_unpack_matmul_init_<false /*transpose*/>(
        buf_desc_id_identity /*in0 -> SrcB*/, buf_desc_id_data /*in1 -> SrcA*/, params.TILE_CNT /*ct_dim*/, 1 /*rt_dim*/, 1 /*kt_dim*/);

    _llk_unpack_matmul_(params.TILE_CNT /*ct_dim*/, 1 /*rt_dim*/, 1 /*kt_dim*/, 0 /*k_idx*/, 0 /*tile_idx*/);
}

#endif

#ifdef LLK_TRISC_MATH

#include "llk_math_common.h"
#include "llk_math_eltwise_unary_datacopy.h"
#include "params.h"

using namespace ckernel;

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    set_up_dest_dvalid_per_thread<dest_dvalid_client::FPU>({dest_dvalid_client::FPU, dest_dvalid_client::PACK});

    DataFormat math_format = static_cast<DataFormat>(formats.math);
    _llk_math_srcAB_hw_configure_<IMPLIED_MATH_FORMAT, is_fp32_dest_acc_en, false /*int32_dest*/>(math_format, math_format);

    // MxFp4_2x datacopy: SrcA holds TILE_CNT data tiles, SrcB holds 1 identity tile (reused).
    _llk_math_eltwise_unary_datacopy_x2_init_<ckernel::MathFidelity::LoFi>(params.TILE_CNT);
    _llk_math_eltwise_unary_datacopy_x2_(params.TILE_CNT);

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
    set_up_dest_dvalid_per_thread<dest_dvalid_client::PACK>({dest_dvalid_client::FPU, dest_dvalid_client::PACK});

    tdma_descriptor_t tdma_desc;
    tdma_desc.buf_desc.f.l1_addr_16B  = L1_ADDRESS(params.buffer_Res[0]);
    tdma_desc.buf_desc.f.lmt_addr_16B = 0;
    tdma_desc.buf_desc.f.format       = static_cast<std::uint8_t>(formats.pack_dst);
    tdma_desc.buf_desc.f.x_dim        = params.TEST_FACE_C_DIM;
    tdma_desc.buf_desc.f.y_dim        = params.TEST_FACE_R_DIM;
    tdma_desc.buf_desc.f.z_dim        = params.num_faces;
    tdma_desc.buf_desc_id             = buf_desc_id_dst;
    tdma_desc.reg_data_format         = static_cast<std::uint8_t>(formats.pack_src);

    _configure_buf_desc_table_(tdma_desc.buf_desc_id, tdma_desc.buf_desc);
    _llk_pack_hw_configure_<p_pacr::PACK0>(tdma_desc);
    const ckernel::ReluConfig relu_config = ckernel::ReluConfig::from_packed(params.RELU_CONFIG);
    _llk_pack_init_<is_fp32_dest_acc_en>(buf_desc_id_dst, params.TILE_CNT, relu_config);
    _llk_pack_(0, 0);
    _llk_pack_dest_dvalid_section_done_<dest_sync, is_fp32_dest_acc_en>();
}
#endif
