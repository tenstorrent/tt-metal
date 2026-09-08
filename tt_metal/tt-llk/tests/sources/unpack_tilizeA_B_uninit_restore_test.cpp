// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Teardown test for `_llk_unpack_tilizeA_B_uninit_`.
//
// The unpacker runs tilizeA_B init, then uninit, then a plain `_llk_unpack_A_`
// datacopy of operand A with NO data-format reconfig in between. The datacopy is
// an identity copy, so the result must equal operand A. tilizeA_B itself is not
// executed: the teardown is what is under test, and it writes the same config
// whether or not any tile was unpacked.
//
// With no reconfig between the two, uninit is the only thing that puts the
// unpacker back at the operand baseline programmed by `configure_unpack_AB`. A
// wrong `Tile_x_dim_cntx0` there makes the datacopy read the operand with the
// wrong per-row datum count and the result diverges.
//
// Run this with face_r_dim < 16, where a fixed 16x16 restore and the operand
// baseline disagree.

#include <cstdint>

#include "ckernel.h"
#include "llk_defs.h"

// Globals
std::uint32_t unp_cfg_context          = 0;
std::uint32_t pack_sync_tile_dst_ptr   = 0;
std::uint32_t math_sync_tile_dst_index = 0;

#ifdef LLK_TRISC_UNPACK

#include "llk_lib_unpack_wrappers.h"
#include "llk_unpack_A.h"
#include "llk_unpack_common.h"
#include "params.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    const std::uint32_t num_faces  = params.num_faces;
    const std::uint32_t face_r_dim = params.TEST_FACE_R_DIM;

    _llk_unpack_hw_configure_<is_fp32_dest_acc_en>(
        formats.unpack_A_src, formats.unpack_B_src, formats.unpack_A_dst, formats.unpack_B_dst, face_r_dim, face_r_dim, num_faces, num_faces);

    _llk_unpack_tilizeA_B_init_wrapper_(formats.unpack_A_src, formats.unpack_A_dst, 1 /* ct_dim */, num_faces, face_r_dim);

    // Teardown under test. No reconfig follows it.
    _llk_unpack_tilizeA_B_uninit_wrapper_(formats.unpack_A_dst, num_faces, face_r_dim);

    _llk_unpack_A_init_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(
        0 /* transpose_of_faces */,
        0 /* within_face_16x16_transpose */,
        ckernel::make_tensor_shape_from_legacy(face_r_dim, num_faces),
        formats.unpack_A_src,
        formats.unpack_A_dst);
    _llk_unpack_A_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(
        L1_ADDRESS(params.buffer_A[0]), formats.unpack_A_src, formats.unpack_A_dst);
}

#endif

#ifdef LLK_TRISC_MATH

#include "llk_lib_math_wrappers.h"
#include "params.h"

using namespace ckernel;

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    const std::uint32_t num_faces   = params.num_faces;
    const bool is_int_fpu_en        = false;
    const std::uint32_t res_dst_idx = 0;

    _llk_math_eltwise_unary_datacopy_init_wrapper_<
        DataCopyType::A2D,
        is_fp32_dest_acc_en,
        BroadcastType::NONE,
        is_int_fpu_en,
        llk_test_pack_mode_v<false, false /* tilize */>>(num_faces, formats.math);
    _llk_math_pack_sync_init_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
    _llk_math_hw_configure_<is_fp32_dest_acc_en>(formats.math, formats.math);

    _llk_math_wait_for_dest_available_<DstSync::SyncHalf>();
    _llk_math_eltwise_unary_datacopy_wrapper_<DataCopyType::A2D, DstSync::SyncHalf, is_fp32_dest_acc_en, BroadcastType::NONE, unpack_to_dest>(
        res_dst_idx, formats.math, formats.math, num_faces);
    _llk_math_dest_section_done_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
}

#endif

#ifdef LLK_TRISC_PACK

#include "llk_lib_pack_wrappers.h"
#include "llk_pack_common.h"
#include "params.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    const std::uint32_t num_faces   = params.num_faces;
    const std::uint32_t face_r_dim  = params.TEST_FACE_R_DIM;
    const std::uint32_t res_dst_idx = 0;
    const std::uint32_t tile_size   = face_r_dim * params.TEST_FACE_C_DIM * num_faces;

    _llk_pack_hw_configure_wrapper_<is_fp32_dest_acc_en, ckernel::PackMode::Default>(
        formats.pack_src, formats.pack_dst, tile_size, face_r_dim, TILE_C_DIM, num_faces);
    _llk_pack_init_wrapper_<ckernel::PackMode::Default, false /* zero_output */>(formats.pack_dst, face_r_dim, TILE_C_DIM, num_faces);
    _llk_pack_dest_init_wrapper_<DstSync::SyncHalf, is_fp32_dest_acc_en, ckernel::PackMode::Default>();

    _llk_packer_wait_for_math_done_();
    _llk_pack_<DstSync::SyncHalf, is_fp32_dest_acc_en, ckernel::PackMode::Default>(res_dst_idx, L1_ADDRESS(params.buffer_Res[0]));
    _llk_pack_dest_section_done_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
}

#endif
