// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// A partial-face (face_r_dim <= 8) ELWADD/ELWSUB may only write the rows its tile has. The MATH
// MOP keeps the 16-row face spacing in DEST with a counter-only INCRWC, so DEST rows 8..15 of
// every face slot must stay untouched. This kernel makes that visible without a DEST dump:
//
//   * unpack feeds full 16-row faces (16x32 tiles), so SrcA/SrcB rows 8..15 hold real data;
//   * math runs the op with an 8-row-face shape of the same two faces;
//   * pack reads the slot back as a full 16x32 tile.
//
// Boot ZEROACCs all of DEST and each pack ZEROACCs the packed half, so a row nothing wrote reads
// as zero: rows 8..15 of each packed face must be zero. If the MOP replaces the INCRWC with a
// second FPU op, those rows carry SrcA[8..15] (+/-) SrcB[8..15] instead.
//
// REUSE_DEST_TYPE selects the MOP under test: NONE runs the standard op once per tile (one
// output tile per input tile); DEST_TO_SRCA / DEST_TO_SRCB seed DEST with tile 0 through the
// standard op and fold the remaining tiles through the dest-reuse op (one output tile).
// BROADCAST_TYPE selects the SrcB broadcast the MOP is built for; the same partial-face MOP
// serves NONE/ROW/COL/SCALAR, so rows 8..15 must stay untouched under every one of them.

#include <cstdint>

#include "ckernel.h"
#include "llk_defs.h"
#include "tensor_shape.h"

// Globals
std::uint32_t unp_cfg_context          = 0;
std::uint32_t pack_sync_tile_dst_ptr   = 0;
std::uint32_t math_sync_tile_dst_index = 0;

// Unpack and pack see a 16x32 tile (two 16-row faces); math sees the same two faces as 8-row faces.
constexpr ckernel::TensorShape FULL_SHAPE    = {ckernel::FACE_R_DIM, ckernel::FACE_C_DIM, 1 /*num_faces_r_dim*/, 2 /*num_faces_c_dim*/};
constexpr ckernel::TensorShape PARTIAL_SHAPE = {ckernel::FACE_R_DIM / 2, ckernel::FACE_C_DIM, 1 /*num_faces_r_dim*/, 2 /*num_faces_c_dim*/};

#ifdef LLK_TRISC_UNPACK

#include "llk_unpack_AB.h"
#include "llk_unpack_common.h"
#include "params.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
#ifndef SPEED_OF_LIGHT
    const std::uint32_t TILE_CNT           = params.TILE_CNT;
    const std::uint32_t TILE_SIZE_UNPACK_A = params.TILE_SIZE_UNPACK_A;
    const std::uint32_t TILE_SIZE_UNPACK_B = params.TILE_SIZE_UNPACK_B;
    const Operand& buffer_A                = params.buffer_A;
    const Operand& buffer_B                = params.buffer_B;
#endif

    _llk_unpack_hw_configure_<is_fp32_dest_acc_en>(
        formats.unpack_A_src,
        formats.unpack_B_src,
        formats.unpack_A_dst,
        formats.unpack_B_dst,
        FULL_SHAPE.face_r_dim,
        FULL_SHAPE.face_r_dim,
        FULL_SHAPE.total_num_faces(),
        FULL_SHAPE.total_num_faces(),
        TILE_SIZE_UNPACK_A,
        TILE_SIZE_UNPACK_B);
    _llk_unpack_configure_stoch_rnd_<StochRndType::None>();
    _llk_unpack_AB_init_<BROADCAST_TYPE>(FULL_SHAPE, ckernel::Transpose::None);

    for (std::uint32_t i = 0; i < TILE_CNT; ++i)
    {
        _llk_unpack_AB_<BROADCAST_TYPE>(L1_ADDRESS(buffer_A[i]), L1_ADDRESS(buffer_B[i]));
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
#ifndef SPEED_OF_LIGHT
    const std::uint32_t TILE_CNT = params.TILE_CNT;
#endif

    _llk_math_pack_sync_init_<dest_sync, is_fp32_dest_acc_en>();
    _llk_math_hw_configure_<is_fp32_dest_acc_en>(formats.math, formats.math);
    _llk_math_eltwise_binary_init_<ELTWISE_BINARY_OP, BROADCAST_TYPE, MathFidelity::LoFi, EltwiseBinaryReuseDestType::NONE>(PARTIAL_SHAPE, 0 /*acc_to_dest*/);

    if constexpr (REUSE_DEST_TYPE == EltwiseBinaryReuseDestType::NONE)
    {
        // One tile per DEST section so every tile lands at dest index 0 of a freshly released half.
        for (std::uint32_t i = 0; i < TILE_CNT; ++i)
        {
            _llk_math_wait_for_dest_available_<dest_sync>();
            _llk_math_eltwise_binary_<ELTWISE_BINARY_OP, BROADCAST_TYPE, dest_sync, is_fp32_dest_acc_en, MathFidelity::LoFi, EltwiseBinaryReuseDestType::NONE>(
                PARTIAL_SHAPE, 0 /*dst_index*/, false /*clear_fp32_dst_acc*/);
            _llk_math_dest_section_done_<dest_sync, is_fp32_dest_acc_en>();
        }
    }
    else
    {
        // Seed dest index 0 with tile 0, then fold tiles 1..TILE_CNT-1 into it through the reuse op.
        _llk_math_wait_for_dest_available_<dest_sync>();
        _llk_math_eltwise_binary_<ELTWISE_BINARY_OP, BROADCAST_TYPE, dest_sync, is_fp32_dest_acc_en, MathFidelity::LoFi, EltwiseBinaryReuseDestType::NONE>(
            PARTIAL_SHAPE, 0 /*dst_index*/, false /*clear_fp32_dst_acc*/);
        _llk_math_eltwise_binary_init_<ELTWISE_BINARY_OP, BROADCAST_TYPE, MathFidelity::LoFi, REUSE_DEST_TYPE>(PARTIAL_SHAPE, 0 /*acc_to_dest*/);
        for (std::uint32_t i = 1; i < TILE_CNT; ++i)
        {
            _llk_math_eltwise_binary_<ELTWISE_BINARY_OP, BROADCAST_TYPE, dest_sync, is_fp32_dest_acc_en, MathFidelity::LoFi, REUSE_DEST_TYPE>(
                PARTIAL_SHAPE, 0 /*dst_index*/, false /*clear_fp32_dst_acc*/);
        }
        _llk_math_dest_section_done_<dest_sync, is_fp32_dest_acc_en>();
    }
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
#ifndef SPEED_OF_LIGHT
    const std::uint32_t TILE_CNT = params.TILE_CNT;
    const Operand& buffer_Res    = params.buffer_Res;
#endif
    const std::uint32_t tile_size = FULL_SHAPE.total_tensor_size();
    const std::uint32_t num_faces = FULL_SHAPE.total_num_faces();
    constexpr bool partial_face   = false;
    constexpr bool narrow_tile    = false;

    _llk_pack_hw_configure_wrapper_<is_fp32_dest_acc_en, PackMode::Default>(
        formats.pack_src, formats.pack_dst, tile_size, FULL_SHAPE.face_r_dim, FULL_SHAPE.total_col_dim(), num_faces, partial_face, narrow_tile);
    _llk_pack_init_wrapper_<PackMode::Default, false /*zero_output*/>(
        formats.pack_dst, FULL_SHAPE.face_r_dim, FULL_SHAPE.total_col_dim(), num_faces, partial_face, narrow_tile);
    _llk_pack_dest_init_wrapper_<dest_sync, is_fp32_dest_acc_en, PackMode::Default>(FULL_SHAPE.face_r_dim, narrow_tile);

    // The standard op packs one tile per input tile; the dest-reuse variants fold everything into one.
    const std::uint32_t output_tiles = (REUSE_DEST_TYPE == ckernel::EltwiseBinaryReuseDestType::NONE) ? TILE_CNT : 1;
    for (std::uint32_t i = 0; i < output_tiles; ++i)
    {
        _llk_packer_wait_for_math_done_();
        _llk_pack_<dest_sync, is_fp32_dest_acc_en, ckernel::PackMode::Default>(0 /*dst_index*/, L1_ADDRESS(buffer_Res[i]));
        _llk_pack_dest_section_done_<dest_sync, is_fp32_dest_acc_en>();
    }
}

#endif
