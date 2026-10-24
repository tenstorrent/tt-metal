// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Zero-flag probe for the 32-bit unpack-to-dest datacopy.
//
// After a tile is unpacked straight into Dest, the Blackhole datacopy re-validates the rows of
// that tile with a zero-flag-clearing ZEROACC. The unpack write itself already validates the
// tile's rows (unless the same-cycle race the ZEROACC guards against hits), so where the ZEROACC
// lands is only observable through a tile it should not touch. This kernel checks that the
// re-validation for DST_INDEX does not land on STALE_TILE = DST_INDEX ^ 4, the tile a
// dst_index-folded-into-four-slots index would hit:
//
//   section 1: unpack-to-dest a tile of known data into Dest tile STALE_TILE, pack it. The
//              section release marks every Dest row as zero, so STALE_TILE now packs as zeros
//              although its stale data is still in the SRAM.
//   section 2: (PROBE_COPY) unpack-to-dest a second tile into Dest tile DST_INDEX, then pack
//              DST_INDEX and STALE_TILE. STALE_TILE was not rewritten and must still pack as
//              zeros; stale data coming out means the re-validation hit STALE_TILE. With
//              PROBE_COPY off, section 2 packs the two tiles without any copy and is the control
//              that packing a released tile yields zeros at all. With PARK_OFFSET the math dest
//              offset is left on STALE_TILE right before the copy, as a SrcRegs-path op targeting
//              that tile leaves it: the datacopy must re-pin the offset before its ZEROACC or the
//              re-validation lands relative to STALE_TILE.
//
// Compile-time: dest_sync, is_fp32_dest_acc_en, unpack_to_dest, PROBE_COPY, PARK_OFFSET, STALE_TILE.
// Runtime: DST_INDEX (the probe tile), buffer_A[0] = stale-tile data, buffer_A[1] = probe data,
// buffer_Res[0..2] = STALE_TILE after section 1, DST_INDEX after section 2, STALE_TILE after section 2.

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
#ifndef SPEED_OF_LIGHT
    const Operand& buffer_A = params.buffer_A;
#endif

    _llk_unpack_hw_configure_<is_fp32_dest_acc_en>(
        formats.unpack_A_src, formats.unpack_B_src, formats.unpack_A_dst, formats.unpack_B_dst, FACE_R_DIM, FACE_R_DIM, TILE_NUM_FACES, TILE_NUM_FACES);
    _llk_unpack_A_init_<BroadcastType::NONE, false /*acc_to_dest*/, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(
        0 /*transpose_of_faces*/,
        0 /*within_face_16x16_transpose*/,
        ckernel::make_tensor_shape_from_legacy(FACE_R_DIM, TILE_NUM_FACES),
        formats.unpack_A_src,
        formats.unpack_A_dst);

    // Section 1: the tile that is left stale in Dest.
    _llk_unpack_A_<BroadcastType::NONE, false /*acc_to_dest*/, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(
        L1_ADDRESS(buffer_A[0]), formats.unpack_A_src, formats.unpack_A_dst);

    // Section 2: the probe tile.
    if constexpr (PROBE_COPY)
    {
        _llk_unpack_A_<BroadcastType::NONE, false /*acc_to_dest*/, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(
            L1_ADDRESS(buffer_A[1]), formats.unpack_A_src, formats.unpack_A_dst);
    }
}

#endif

#ifdef LLK_TRISC_MATH

#include "llk_lib_math_wrappers.h"
#include "llk_lib_unpack_wrappers.h"
#include "params.h"

using namespace ckernel;

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
#ifndef SPEED_OF_LIGHT
    const int DST_INDEX = params.DST_INDEX;
#endif
    const std::uint32_t probe_tile = static_cast<std::uint32_t>(DST_INDEX);

    constexpr std::uint32_t MAX_TILES = get_dest_max_tiles<dest_sync, is_fp32_dest_acc_en, DstTileShape::Tile32x32>();
    static_assert(STALE_TILE < MAX_TILES, "STALE_TILE exceeds max dest tiles");
    LLK_ASSERT(probe_tile < MAX_TILES, "DST_INDEX exceeds max dest tiles");
    LLK_ASSERT(probe_tile != STALE_TILE, "the probe tile must differ from the stale tile");

    _llk_math_eltwise_unary_datacopy_init_wrapper_<DataCopyType::A2D, is_fp32_dest_acc_en, BroadcastType::NONE, false /*is_int_fpu_en*/, PackMode::Default>(
        TILE_NUM_FACES, formats.math);
    _llk_math_pack_sync_init_<dest_sync, is_fp32_dest_acc_en>();
    _llk_math_hw_configure_<is_fp32_dest_acc_en>(formats.math, formats.math);

    // Section 1: write the stale tile.
    _llk_math_wait_for_dest_available_<dest_sync>();
    _llk_math_eltwise_unary_datacopy_wrapper_<DataCopyType::A2D, dest_sync, is_fp32_dest_acc_en, BroadcastType::NONE, unpack_to_dest>(
        STALE_TILE, formats.math, formats.math, TILE_NUM_FACES);
    _llk_math_dest_section_done_<dest_sync, is_fp32_dest_acc_en>();

    // Section 2: write the probe tile only; the stale tile is left as the release marked it.
    _llk_math_wait_for_dest_available_<dest_sync>();
    if constexpr (PARK_OFFSET)
    {
        // What a preceding SrcRegs-path op on STALE_TILE leaves in the math dest offset.
        math::set_dst_write_addr<DstTileShape::Tile32x32, UnpackDestination::SrcRegs>(STALE_TILE);
    }
    if constexpr (PROBE_COPY)
    {
        _llk_math_eltwise_unary_datacopy_wrapper_<DataCopyType::A2D, dest_sync, is_fp32_dest_acc_en, BroadcastType::NONE, unpack_to_dest>(
            probe_tile, formats.math, formats.math, TILE_NUM_FACES);
    }
    _llk_math_dest_section_done_<dest_sync, is_fp32_dest_acc_en>();
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
    const int DST_INDEX       = params.DST_INDEX;
    const Operand& buffer_Res = params.buffer_Res;
#endif
    const std::uint32_t probe_tile = static_cast<std::uint32_t>(DST_INDEX);

    _llk_pack_hw_configure_wrapper_<is_fp32_dest_acc_en, PackMode::Default>(
        formats.pack_src, formats.pack_dst, FACE_R_DIM * FACE_C_DIM * TILE_NUM_FACES /*tile_size*/, FACE_R_DIM, TILE_C_DIM, TILE_NUM_FACES);
    _llk_pack_init_wrapper_<PackMode::Default, false /*zero_output*/>(formats.pack_dst, FACE_R_DIM, TILE_C_DIM, TILE_NUM_FACES);
    _llk_pack_dest_init_<dest_sync, is_fp32_dest_acc_en>();

    // Section 1: pack the stale tile, then release Dest. The release marks every Dest row as zero
    // without touching the SRAM, so the stale data stays behind its cleared flags.
    _llk_packer_wait_for_math_done_();
    _llk_pack_<dest_sync, is_fp32_dest_acc_en, ckernel::PackMode::Default>(STALE_TILE, L1_ADDRESS(buffer_Res[0]));
    _llk_pack_dest_section_done_<dest_sync, is_fp32_dest_acc_en>();

    // Section 2: the probe tile, then the stale tile again (expected to pack as zeros).
    _llk_packer_wait_for_math_done_();
    _llk_pack_<dest_sync, is_fp32_dest_acc_en, ckernel::PackMode::Default>(probe_tile, L1_ADDRESS(buffer_Res[1]));
    _llk_pack_<dest_sync, is_fp32_dest_acc_en, ckernel::PackMode::Default>(STALE_TILE, L1_ADDRESS(buffer_Res[2]));
    _llk_pack_dest_section_done_<dest_sync, is_fp32_dest_acc_en>();
}

#endif
