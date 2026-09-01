// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Matmul without a MOP (Quasar).
//
// Unpack and pack are the plain matmul paths; only math differs from matmul_quasar_test.cpp: instead of
// running MOP BANK0 it issues REPLAY + MVMUL straight from the RISC core. The result must match the
// MOP-based matmul bit for bit, so the golden remains ordinary MatmulGolden.

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
    const std::uint32_t CT_DIM      = params.CT_DIM;
    const std::uint32_t RT_DIM      = params.RT_DIM;
    const std::uint32_t KT_DIM      = params.KT_DIM;
    const std::uint32_t num_faces_A = params.num_faces_A;
    const std::uint32_t num_faces_B = params.num_faces_B;
    const Operand& buffer_A         = params.buffer_A;
    const Operand& buffer_B         = params.buffer_B;

    set_ttsync_enables<TRACK_ALL>(ckernel::TRISC_ID);

    // No-mop matmul is full-tile only: the replayed MVMUL walk covers all four 16x16 faces, so
    // num_faces_A/num_faces_B are always 4.
    // Matmul flips the unpacker roles: _llk_unpack_matmul_init_ arg0 drives UNPACR1/SrcB, arg1 drives
    // UNPACR0/SrcA -- so operand A is recorded under Unp1 and operand B under Unp0 (matches product).
    ckernel::trisc::bfd_alloc_and_program<ckernel::trisc::BfdResource::Unp1>(
        ckernel::tensor_shape_from_num_faces(FACE_R_DIM, num_faces_A), L1_ADDRESS(buffer_A[0]), formats.unpack_A_src);
    ckernel::trisc::bfd_alloc_and_program<ckernel::trisc::BfdResource::Unp0>(
        ckernel::tensor_shape_from_num_faces(FACE_R_DIM, num_faces_B), L1_ADDRESS(buffer_B[0]), formats.unpack_B_src);
    _llk_unpack_hw_configure_<ckernel::p_unpacr::UNP_B>(static_cast<DataFormat>(formats.unpack_A_dst));
    _llk_unpack_hw_configure_<ckernel::p_unpacr::UNP_A>(static_cast<DataFormat>(formats.unpack_B_dst));

    // transpose in src_A is not supported for quasar
    _llk_unpack_matmul_init_<UNPACK_TRANSPOSE_FACES>(
        ckernel::trisc::bfd_current<ckernel::trisc::BfdResource::Unp1>(),
        ckernel::trisc::bfd_current<ckernel::trisc::BfdResource::Unp0>(),
        CT_DIM,
        RT_DIM,
        KT_DIM);

    for (std::uint32_t j = 0; j < KT_DIM; j++)
    {
        _llk_unpack_matmul_(CT_DIM, RT_DIM, KT_DIM, j, j * CT_DIM);
    }
}

#endif

#ifdef LLK_TRISC_MATH

#include "experimental/llk_math_matmul_custom_no_mop.h"
#include "llk_math_common.h"
#include "params.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    const std::uint32_t CT_DIM = params.CT_DIM;
    const std::uint32_t RT_DIM = params.RT_DIM;
    const std::uint32_t KT_DIM = params.KT_DIM;

    set_up_fpu_to_pack_dest_dvalid_chain<dest_dvalid_client::FPU>();

    DataFormat math_format     = static_cast<DataFormat>(formats.math);
    DataFormat pack_src_format = static_cast<DataFormat>(formats.pack_src);
    if constexpr (is_fp32_dest_acc_en)
    {
        if (pack_src_format == DataFormat::Int32)
        {
            _llk_math_srcAB_hw_configure_<IMPLIED_MATH_FORMAT, false /*fp32_dest*/, true /*int32_dest*/>(math_format, math_format);
        }
        else
        {
            _llk_math_srcAB_hw_configure_<IMPLIED_MATH_FORMAT, true /*fp32_dest*/, false /*int32_dest*/>(math_format, math_format);
        }
    }
    else
    {
        _llk_math_srcAB_hw_configure_<IMPLIED_MATH_FORMAT, false /*fp32_dest*/, false /*int32_dest*/>(math_format, math_format);
    }

    // ENABLE_2X_FORMAT enables the 2x-packed FP4 matmul path (8 MVMULs per tile vs 16, K-dim halved per
    // MVMUL via the SrcA 2x sub-datum expansion). Set when SrcA/SrcB are configured as MxFp4_2x_A/B.
    _llk_math_matmul_init_no_mop_<(ckernel::MathFidelity)MATH_FIDELITY, ENABLE_2X_FORMAT>(CT_DIM, RT_DIM);

    for (std::uint32_t i = 0; i < KT_DIM; i++)
    {
        _llk_math_matmul_block_no_mop_<(ckernel::MathFidelity)MATH_FIDELITY, ENABLE_2X_FORMAT>(CT_DIM, RT_DIM);
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
    const std::uint32_t CT_DIM    = params.CT_DIM;
    const std::uint32_t RT_DIM    = params.RT_DIM;
    const std::uint32_t num_faces = params.num_faces;
    const Operand& buffer_Res     = params.buffer_Res;

    set_up_fpu_to_pack_dest_dvalid_chain<dest_dvalid_client::PACK>();

    // Full 32x32 tiles: 2x2 faces of 16x16 (no-mop matmul is full-tile only).
    ckernel::trisc::bfd_alloc_and_program<ckernel::trisc::BfdResource::Pack0>(
        ckernel::tensor_shape_from_num_faces(FACE_R_DIM, num_faces), L1_ADDRESS(buffer_Res[0]), formats.pack_dst);
    _llk_pack_hw_configure_<p_pacr::PACK0, is_fp32_dest_acc_en>(static_cast<DataFormat>(formats.pack_src), ckernel::ReluConfig::none());
    _llk_pack_matmul_init_(ckernel::trisc::bfd_current<ckernel::trisc::BfdResource::Pack0>(), RT_DIM, CT_DIM, 1 /*num_subblocks_c_dim*/);

    _llk_pack_matmul_(0 /*start_math_dest_tile_idx*/, 0 /*start_l1_tile_idx*/);
    _llk_pack_dest_dvalid_section_done_<dest_sync, is_fp32_dest_acc_en>();
}

#endif
