// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Quasar compressed_custom_mm: in0 [M, kt_dim * 32] (M in {1, 2, 4, 8}) x in1 [kt_dim * 32, ct_dim * 32], where
// every in1 tile has its own MX format and zero tiles are not stored. The contract is in
// experimental/llk_compressed_custom_mm_common.h.
//
// Buffers, all written to L1 verbatim by the Python test:
//   buffer_A: in0, kt_dim tiles of two dense M x 16 faces
//   buffer_B: the compressed in1 stream
//   buffer_C: one metadata word per in1 tile
//   buffer_Res: ct_dim output tiles of two dense M x 16 faces

#include <cstdint>

#include "ckernel.h"
#include "llk_defs.h"
#include "llk_memory_checks.h"
#include "quasar_test_common.h"
#include "sfpu_stub.h"

// Globals
std::uint32_t unp_cfg_context          = 0;
std::uint32_t pack_sync_tile_dst_ptr   = 0;
std::uint32_t math_sync_tile_dst_index = 0;

#ifdef LLK_TRISC_UNPACK

#include "experimental/llk_unpack_compressed_custom_mm.h"
#include "llk_bfd_alloc.h"
#include "params.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    const ckernel::TensorShape tensor_shape_A =
        ckernel::make_tensor_shape(params.in0_face_r_dim, params.in0_face_c_dim, params.num_faces_r_dim_A, params.num_faces_c_dim_A);

    set_ttsync_enables<TRACK_ALL>(ckernel::TRISC_ID);
    // Same unpacker roles as matmul: in0 -> UNPACR1/SrcB, in1 -> UNPACR0/SrcA. In ARG_FIFO mode 1 the in1
    // descriptor only provides the 32x32 tile dimensions; each tile's address and format come from the metadata.
    const auto bfd_a =
        ckernel::trisc::bfd_alloc_and_program<ckernel::trisc::BfdResource::Unp1>(tensor_shape_A, L1_ADDRESS(params.buffer_A[0]), formats.unpack_A_src);
    const auto bfd_b = ckernel::trisc::bfd_alloc_and_program<ckernel::trisc::BfdResource::Unp0>(
        ckernel::DEFAULT_TENSOR_SHAPE, L1_ADDRESS(params.buffer_B[0]), formats.unpack_B_src);
    _llk_unpack_hw_configure_<ckernel::p_unpacr::UNP_B>(static_cast<DataFormat>(formats.unpack_A_dst));
    _llk_unpack_hw_configure_<ckernel::p_unpacr::UNP_A>(static_cast<DataFormat>(formats.unpack_B_dst));

    _llk_unpack_compressed_custom_mm_init_(bfd_a, bfd_b, tensor_shape_A);
    _llk_unpack_compressed_custom_mm_(
        L1_ADDRESS(params.buffer_B[0]),
        params.buffer_C[0], // metadata: raw byte address, the RISC-V dereferences it
        params.KT_DIM,
        params.CT_DIM);
    _llk_unpack_compressed_custom_mm_uninit_();
}

#endif

#ifdef LLK_TRISC_MATH

#include "experimental/llk_math_compressed_custom_mm.h"
#include "llk_math_common.h"
#include "params.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    const ckernel::TensorShape tensor_shape_A =
        ckernel::make_tensor_shape(params.in0_face_r_dim, params.in0_face_c_dim, params.num_faces_r_dim_A, params.num_faces_c_dim_A);

    set_up_fpu_to_pack_dest_dvalid_chain<dest_dvalid_client::FPU>();
    const DataFormat math_format = static_cast<DataFormat>(formats.math);
    _llk_math_srcAB_hw_configure_<false /*EN_IMPLIED_MATH_FORMAT*/, is_fp32_dest_acc_en>(math_format, math_format);

    _llk_math_compressed_custom_mm_init_<static_cast<ckernel::MathFidelity>(MATH_FIDELITY)>(math_format, math_format, tensor_shape_A);
    _llk_math_compressed_custom_mm_(0 /*dst_index*/, params.KT_DIM, params.CT_DIM);
    _llk_math_set_dvalid_<p_cleardvalid::FPU, dest_sync>();
}

#endif

#ifdef LLK_TRISC_PACK

#include "llk_bfd_alloc.h"
#include "llk_pack.h"
#include "llk_pack_matmul.h"
#include "params.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    // Output tile [M, 32]: in0's rows by in1's columns.
    const ckernel::TensorShape output_shape =
        ckernel::make_tensor_shape(params.in0_face_r_dim, params.in1_face_c_dim, params.num_faces_r_dim_A, params.num_faces_c_dim_B);

    set_up_fpu_to_pack_dest_dvalid_chain<dest_dvalid_client::PACK>();
    const auto bfd_pack =
        ckernel::trisc::bfd_alloc_and_program<ckernel::trisc::BfdResource::Pack0>(output_shape, L1_ADDRESS(params.buffer_Res[0]), formats.pack_dst);
    _llk_pack_hw_configure_<p_pacr::PACK0, is_fp32_dest_acc_en>(static_cast<DataFormat>(formats.pack_src), ckernel::ReluConfig::none());

    _llk_pack_matmul_init_(bfd_pack, 1 /*subblock_r_dim*/, params.CT_DIM, 1 /*num_subblocks_c_dim*/, output_shape);
    _llk_pack_matmul_(0 /*start_math_dest_tile_idx*/, 0 /*start_l1_tile_idx*/, output_shape);
    _llk_pack_dest_dvalid_section_done_<dest_sync, is_fp32_dest_acc_en>();
}

#endif
