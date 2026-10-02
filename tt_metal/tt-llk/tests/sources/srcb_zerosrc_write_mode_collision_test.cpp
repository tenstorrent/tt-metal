// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Wormhole: a math-thread ZEROSRC carrying write_mode can redirect an unpacker
// source clear onto the Matrix Unit's live SrcB bank.
//
// Unpack publishes one face into SrcA and SrcB, then issues NUM_UNPACK_CLEARS
// SrcB clears that wait only on the unpacker's own (free) bank. Math, while it
// holds the published SrcB bank, issues NUM_MATH_ZEROSRC ZEROSRC(SrcA,
// write_mode=MATH_WRITE_MODE) -- write_mode=1 is the form reduce-row MAX uses --
// and then ELWADDs SrcA (zero) onto SrcB, so the packed face is exactly SrcB. Any zeroed datum means an
// unpacker clear landed on the bank math was reading. MATH_WRITE_MODE selects
// the write_mode bit of the math burst, so the same timing can be run with the
// bit clear as a control. SrcA arrives as zeros from L1, so no arm relies on the
// math ZEROSRC for its result.

#include <cstdint>

#include "ckernel.h"
#include "llk_defs.h"
#include "tensor_shape.h"

std::uint32_t unp_cfg_context          = 0;
std::uint32_t pack_sync_tile_dst_ptr   = 0;
std::uint32_t math_sync_tile_dst_index = 0;

// One 16x16 face, so unpack AB publishes exactly one SrcA and one SrcB bank.
static constexpr ckernel::TensorShape single_face = {16, 16, 1, 1};

#ifdef LLK_TRISC_UNPACK

#include "llk_unpack_AB.h"
#include "llk_unpack_common.h"
#include "params.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    _llk_unpack_hw_configure_<false>(
        formats.unpack_A_src,
        formats.unpack_B_src,
        formats.unpack_A_dst,
        formats.unpack_B_dst,
        single_face.face_r_dim,
        single_face.face_r_dim,
        single_face.total_num_faces(),
        single_face.total_num_faces(),
        params.TILE_SIZE_UNPACK_A,
        params.TILE_SIZE_UNPACK_B);
    _llk_unpack_AB_init_<BroadcastType::NONE>(single_face, ckernel::Transpose::None);
    _llk_unpack_AB_<BroadcastType::NONE>(L1_ADDRESS(params.buffer_A[0]), L1_ADDRESS(params.buffer_B[0]));

    // The published SrcB bank now belongs to math; the unpacker's own bank is free.
    for (int i = 0; i < NUM_UNPACK_CLEARS; ++i)
    {
        if constexpr (UNPACK_CLEAR_MODE == UnpackClearMode::OwnBankWait)
        {
            TTI_UNPACR_NOP(SrcB, p_unpacr_nop::UNP_ZEROSRC_STALL_RESET_WR_RDY);
        }
        else if constexpr (UNPACK_CLEAR_MODE == UnpackClearMode::DefaultWait)
        {
            TTI_UNPACR_NOP(SrcB, p_unpacr_nop::UNP_ZEROSRC);
        }
    }
}

#endif

#ifdef LLK_TRISC_MATH

#include "llk_math_common.h"
#include "llk_math_eltwise_binary.h"
#include "params.h"

using namespace ckernel;

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    _llk_math_pack_sync_init_<DstSync::SyncHalf, false>();
    _llk_math_hw_configure_<false>(formats.math, formats.math);
    _llk_math_eltwise_binary_init_<EltwiseBinaryType::ELWADD, BroadcastType::NONE, MathFidelity::LoFi, EltwiseBinaryReuseDestType::NONE>(
        single_face, false /* acc_to_dest */);

    _llk_math_wait_for_dest_available_<DstSync::SyncHalf>();

    // Hold the ZEROSRC burst until both published banks belong to math, so it
    // overlaps the unpacker's clear burst.
    TTI_STALLWAIT(p_stall::STALL_MATH, p_stall::SRCA_VLD | p_stall::SRCB_VLD);
    for (int i = 0; i < NUM_MATH_ZEROSRC; ++i)
    {
        TTI_ZEROSRC(0 /* zero_val */, MATH_WRITE_MODE, 0 /* bank_mask */, 1 /* SrcA */);
    }

    _llk_math_eltwise_binary_<EltwiseBinaryType::ELWADD, BroadcastType::NONE, DstSync::SyncHalf, false, MathFidelity::LoFi, EltwiseBinaryReuseDestType::NONE>(
        single_face, 0 /* dst_index */, false /* clear_fp32_dst_acc */);
    _llk_math_dest_section_done_<DstSync::SyncHalf, false>();
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
    const std::uint32_t num_faces = single_face.total_num_faces();
    const bool narrow_tile        = (single_face.num_faces_c_dim == 1);
    _llk_pack_hw_configure_wrapper_<false, PackMode::Default>(
        formats.pack_src,
        formats.pack_dst,
        single_face.total_tensor_size(),
        single_face.face_r_dim,
        single_face.total_col_dim(),
        num_faces,
        false /* partial_face */,
        narrow_tile);
    _llk_pack_init_wrapper_<PackMode::Default, false /* zero_output */>(
        formats.pack_dst, single_face.face_r_dim, single_face.total_col_dim(), num_faces, false /* partial_face */, narrow_tile);
    _llk_pack_dest_init_wrapper_<DstSync::SyncHalf, false, PackMode::Default>(single_face.face_r_dim, narrow_tile);

    _llk_packer_wait_for_math_done_();
    _llk_pack_<DstSync::SyncHalf, false, ckernel::PackMode::Default>(0, L1_ADDRESS(params.buffer_Res[0]));
    _llk_pack_dest_section_done_<DstSync::SyncHalf, false>();
}

#endif
