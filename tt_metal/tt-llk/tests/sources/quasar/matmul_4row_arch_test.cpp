// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// 4row_arch INT8_2x matmul kernel. Identical to matmul_quasar_test.cpp on the unpack/pack
// threads; the math thread uses llk_math_matmul_4row_arch.h (the 16-MVMULDI DI+X2 traversal
// for INT8_2x) instead of the Quasar llk_math_matmul.h (8-MVMULDI MxFp4_2x traversal).

#include <algorithm>
#include <cstdint>
#include <cstdio>

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

#include "llk_bfd_alloc.h"
#include "llk_unpack_matmul.h"
#include "params.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif

    // Setup sync for unpack
    set_ttsync_enables<TRACK_ALL>(ckernel::TRISC_ID);
    // Full 32x32 tiles: 2x2 faces of 16x16 (tiny tiles not supported for quasar).
    // Matmul flips the unpacker roles: _llk_unpack_matmul_init_ arg0 drives UNPACR1/SrcB, arg1 drives
    // UNPACR0/SrcA -- so operand A is recorded under Unp1 and operand B under Unp0 (matches product).
    ckernel::trisc::bfd_alloc_and_program<ckernel::trisc::BfdResource::Unp1>(
        ckernel::tensor_shape_from_num_faces(FACE_R_DIM, num_faces_A), L1_ADDRESS(params.buffer_A[0]), formats.unpack_A_src);
    ckernel::trisc::bfd_alloc_and_program<ckernel::trisc::BfdResource::Unp0>(
        ckernel::tensor_shape_from_num_faces(FACE_R_DIM, num_faces_B), L1_ADDRESS(params.buffer_B[0]), formats.unpack_B_src);
    _llk_unpack_hw_configure_<ckernel::p_unpacr::UNP_B>(static_cast<DataFormat>(formats.unpack_A_dst));
    _llk_unpack_hw_configure_<ckernel::p_unpacr::UNP_A>(static_cast<DataFormat>(formats.unpack_B_dst));

    _llk_unpack_matmul_init_<UNPACK_TRANSPOSE_FACES>(
        ckernel::trisc::bfd_current<ckernel::trisc::BfdResource::Unp1>(),
        ckernel::trisc::bfd_current<ckernel::trisc::BfdResource::Unp0>(),
        CT_DIM,
        RT_DIM,
        KT_DIM); // transpose in src_A not supported for quasar

    for (std::uint32_t j = 0; j < KT_DIM; j++)
    {
        _llk_unpack_matmul_(CT_DIM, RT_DIM, KT_DIM, j, j * CT_DIM);
    }
}

#endif

#ifdef LLK_TRISC_MATH

#include "llk_math_common.h"
#include "llk_math_matmul_4row_arch.h"
#include "params.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    set_up_fpu_to_pack_dest_dvalid_chain<dest_dvalid_client::FPU>();

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
    // ENABLE_2X_FORMAT enables the INT8_2x matmul path (16 MVMULDI per tile; two int8 packed
    // per SrcA/SrcB datum drive the 32-way dot-product reduction, result in INT32). Set when
    // SrcA/SrcB are configured as Int8_2x / UInt8_2x.
    // ENABLE_DIRECT_INDEXING selects the DI variant (MVMULDI with explicit indices); the
    // 4row_arch X2 traversal is implemented on the DI path.
    // NOTE: the 4row_arch matmul API takes CT_DIM/RT_DIM as compile-time template params.
    _llk_math_matmul_init_<(ckernel::MathFidelity)MATH_FIDELITY, CT_DIM, RT_DIM, ENABLE_DIRECT_INDEXING, ENABLE_2X_FORMAT>();

    for (std::uint32_t i = 0; i < KT_DIM; i++)
    {
        _llk_math_matmul_block_<CT_DIM, RT_DIM>();
    }
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
    set_up_fpu_to_pack_dest_dvalid_chain<dest_dvalid_client::PACK>();

    // Full 32x32 tiles: 2x2 faces of 16x16 (tiny tiles not supported for quasar).
    ckernel::trisc::bfd_alloc_and_program<ckernel::trisc::BfdResource::Pack0>(
        ckernel::tensor_shape_from_num_faces(FACE_R_DIM, num_faces), L1_ADDRESS(params.buffer_Res[0]), formats.pack_dst);
    _llk_pack_hw_configure_<p_pacr::PACK0, is_fp32_dest_acc_en>(static_cast<DataFormat>(formats.pack_src), ckernel::ReluConfig::none());
    _llk_pack_matmul_init_(ckernel::trisc::bfd_current<ckernel::trisc::BfdResource::Pack0>(), RT_DIM, CT_DIM, 1 /*num_subblocks_c_dim*/);

    _llk_pack_matmul_(0, 0);
    _llk_pack_dest_dvalid_section_done_<dest_sync, is_fp32_dest_acc_en>();
}

#endif
