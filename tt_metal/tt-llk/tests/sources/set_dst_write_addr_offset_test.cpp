// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Behaviour test for the shared sort-SFPU helper set_dst_write_addr_offset
// (sfpu/experimental/ckernel_sfpu_set_dst_write_addr_offset.h, extracted by tt-metal
// #52713) -- Blackhole only.
//
// Why this exists separately from sort_headers_coexist_test.cpp: that test proves the two
// sort headers COMPILE in one translation unit and deliberately asserts nothing about the
// offset, because the datacopy it uses to read DEST back calls
// math::set_dst_write_addr<Tile32x32, SrcRegs>(dst_index) -- the same
// DEST_TARGET_REG_CFG_MATH_Offset_ADDR32 the helper writes -- before anything touches
// DEST, discarding whatever the helper left. So the helper the whole extraction is about
// had no behavioural coverage at all. This driver supplies it.
//
// How the offset is made observable
// ---------------------------------
// _llk_math_eltwise_unary_sfpu_params_ has the same problem: it calls
// _llk_math_eltwise_sfpu_start_(dst_index) -> set_dst_write_addr(dst_index) first. The
// helper therefore has to run INSIDE the SFPU body, after start_, which is exactly how
// the real consumers use it. VectorMode::None is required, not incidental: RC invokes the
// body once per face with _llk_math_eltwise_sfpu_inc_dst_face_addr_() in between, and the
// helper writes an ABSOLUTE address (base + addr), so under RC every face would be
// redirected to the same rows. The consumers likewise do their own addressing.
//
// The body then negates one face in place -- 8 dst_reg steps, since a face is 16x16 = 256
// datums and the SFPU moves 32 lanes per step. Negation is used because it is exact in
// both DEST widths and, on a strictly positive input, unambiguous: a negated datum can
// only have come from the SFPU body. Reads and writes both follow the offset, so the
// effect is "the face starting at Dst row OFFSET_ROWS is negated".
//
// Units. The offset register counts Dst ROWS, and one 32x32 tile is 64 of them
// (DstTileSizeLog2[DstTileShape::Tile32x32] == 6, and math::set_dst_write_addr computes
// tile_index << 6). That is what makes the two real call patterns meaningful:
//   64  whole-tile rebase   -- deepseek_top32_rm's `tile_offset` (dst_index << 6)
//    2  column-group flip   -- topk_xl's `odd_col_offset`, and deepseek's too
//
// What the variants are for
// -------------------------
// OFFSET_ENABLED=false is the control, and the reason this test can assert something
// exact rather than just "the output changed": with the helper absent, SFPU_DST_INDEX
// alone decides where the negate lands. So
//
//     helper(OFFSET_ROWS = N * 64) at SFPU_DST_INDEX = 0
//         must be bit-identical to
//     no helper at SFPU_DST_INDEX = N
//
// which pins the helper against the LLK's own canonical addressing function without this
// test needing to model DEST layout. SFPU_ENABLED=false gives the datacopy-only baseline
// the negate is diffed against.
//
// Not covered: the LLK_ASSERT on addr >= DEST_REGISTER_HALF_SIZE. Nothing in the suite
// expects an LLK assert (conftest treats LLKAssertException as a failure), and tripping
// one mid-kernel risks leaving the device wedged for the tests that follow.

#include <cstdint>

#include "ckernel.h"
#include "llk_defs.h"
#include "params.h"

// Globals required by the test framework.
std::uint32_t unp_cfg_context          = 0;
std::uint32_t pack_sync_tile_dst_ptr   = 0;
std::uint32_t math_sync_tile_dst_index = 0;

static constexpr ckernel::DstSync DST_SYNC = ckernel::DstSync::SyncHalf;

#ifdef LLK_TRISC_UNPACK

#include "llk_unpack_A.h"
#include "llk_unpack_common.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    _llk_unpack_hw_configure_<is_fp32_dest_acc_en>(
        formats.unpack_A_src, formats.unpack_B_src, formats.unpack_A_dst, formats.unpack_B_dst, FACE_R_DIM, FACE_R_DIM, TILE_NUM_FACES, TILE_NUM_FACES);

    _llk_unpack_A_init_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(
        0 /* transpose_of_faces */, 0 /* within_face_16x16_transpose */, ckernel::DEFAULT_TENSOR_SHAPE, formats.unpack_A_src, formats.unpack_A_dst);

    for (std::uint32_t tile = 0; tile < PACK_NUM_TILES; ++tile)
    {
        _llk_unpack_A_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(
            L1_ADDRESS(params.buffer_A[tile]), formats.unpack_A_src, formats.unpack_A_dst);
    }
}

#endif

#ifdef LLK_TRISC_MATH

#include "ckernel_sfpu.h"
#include "llk_lib_math_wrappers.h"
#include "llk_math_common.h"
#include "llk_math_eltwise_unary_datacopy.h"
#include "llk_math_eltwise_unary_sfpu.h"
#include "llk_math_eltwise_unary_sfpu_params.h"
#include "sfpu/experimental/ckernel_sfpu_set_dst_write_addr_offset.h"

using namespace ckernel;

// A face is FACE_R_DIM x FACE_C_DIM = 256 datums and the SFPU moves 32 lanes per
// dst_reg step, so one face is 8 steps.
constexpr int FACE_SFPU_STEPS = (FACE_R_DIM * FACE_C_DIM) / 32;

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    _llk_math_pack_sync_init_<DST_SYNC, is_fp32_dest_acc_en>();
    _llk_math_hw_configure_<is_fp32_dest_acc_en>(formats.math, formats.math);

    _llk_math_eltwise_unary_datacopy_init_wrapper_<DataCopyType::A2D, is_fp32_dest_acc_en, BroadcastType::NONE, false /* is_int_fpu_en */, PackMode::Default>(
        TILE_NUM_FACES, formats.math);

    _llk_math_eltwise_unary_sfpu_init_<SfpuType::unused>();

    _llk_math_wait_for_dest_available_<DST_SYNC>();

    // Known contents at the standard 64-row tile spacing. All PACK_NUM_TILES stay
    // resident in this one DEST section, so a whole-tile offset has somewhere to land.
    for (std::uint32_t tile = 0; tile < PACK_NUM_TILES; ++tile)
    {
        _llk_math_eltwise_unary_datacopy_<DataCopyType::A2D, DST_SYNC, is_fp32_dest_acc_en, BroadcastType::NONE, unpack_to_dest>(
            tile, formats.math, formats.math);
    }

    if constexpr (SFPU_ENABLED)
    {
        _llk_math_eltwise_unary_sfpu_params_(
            []
            {
                // THE FUNCTION UNDER TEST. Inside the body on purpose: start_ has
                // already programmed the write pointer from SFPU_DST_INDEX, and this
                // overwrites it absolutely.
                if constexpr (OFFSET_ENABLED)
                {
                    ckernel::sfpu::set_dst_write_addr_offset(OFFSET_ROWS);
                }

                for (int d = 0; d < FACE_SFPU_STEPS; ++d)
                {
                    const sfpi::vFloat v = sfpi::dst_reg[0];
                    sfpi::dst_reg[0]     = -v;
                    sfpi::dst_reg++;
                }
            },
            SFPU_DST_INDEX,
            VectorMode::None);
    }

    _llk_math_dest_section_done_<DST_SYNC, is_fp32_dest_acc_en>();
}

#endif

#ifdef LLK_TRISC_PACK

#include "llk_lib_pack_wrappers.h"
#include "llk_pack_common.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    _llk_pack_hw_configure_wrapper_<is_fp32_dest_acc_en, PackMode::Default>(formats.pack_src, formats.pack_dst, FACE_R_DIM * FACE_C_DIM * TILE_NUM_FACES);
    _llk_pack_init_wrapper_<PackMode::Default, false /* zero_output */>(formats.pack_dst, FACE_R_DIM, TILE_C_DIM, TILE_NUM_FACES);
    _llk_pack_dest_init_wrapper_<DST_SYNC, is_fp32_dest_acc_en, PackMode::Default>();

    _llk_packer_wait_for_math_done_();

    for (std::uint32_t tile = 0; tile < PACK_NUM_TILES; ++tile)
    {
        _llk_pack_<DST_SYNC, is_fp32_dest_acc_en, ckernel::PackMode::Default>(tile, L1_ADDRESS(params.buffer_Res[tile]));
    }

    _llk_pack_dest_section_done_<DST_SYNC, is_fp32_dest_acc_en>();
}

#endif
