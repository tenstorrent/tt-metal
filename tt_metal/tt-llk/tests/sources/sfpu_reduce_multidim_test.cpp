// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Multi-dimensional (column-then-row) SFPU reduce test.
//
// This kernel reduces over BOTH dims of every 32x32 tile by chaining a REDUCE_COL pass and a
// REDUCE_ROW pass under a SINGLE shared `init_reduce`, mirroring how a multi-axis reduce
// (e.g. ttir.max dim=[1,2]) is lowered: one `sfpu_reduce_init` for the fused op followed by two
// `sfpu_reduce` calls, with no re-init in between.
//
// This is exactly the configuration that exposed an Int32 MAX/MIN regression: the column Int32 path
// (INT32_2S_COMP, two's-complement, and its own init_reduce_max_min_int32 that flips the
// SFPSWAP-direction config bit) runs first, then the row path runs trusting state established by the
// shared init. With the column stage's inverted comparator direction left in place, the row stage
// misordered values and produced wrong results once negatives/extremes were present. Single-axis
// reduces only exercise one path with its matching init and therefore pass, so the bug is only
// reachable here.
//
// Geometry: a column of BLOCK_RT_DIM row-tiles, one column-tile wide (BLOCK_CT_DIM == 1).
//   Phase 1 (REDUCE_COL, per tile): row 0 of each tile holds that tile's 32 per-column extremes.
//   Phase 2 (REDUCE_ROW, per row-tile): each tile's element [0][0] ends up holding the extreme
//            over that whole 32x32 tile (the per-tile multi-axis reduction result).
// Reading [0][0] of every tile yields one independent multi-axis result per tile.

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <type_traits>

#include "ckernel.h"
#include "llk_defs.h"
#include "profiler.h"

// Globals
std::uint32_t unp_cfg_context          = 0;
std::uint32_t pack_sync_tile_dst_ptr   = 0;
std::uint32_t math_sync_tile_dst_index = 0;

#ifdef LLK_TRISC_UNPACK

#include "llk_unpack_A.h"
#include "llk_unpack_common.h"
#include "params.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    _llk_unpack_hw_configure_<is_fp32_dest_acc_en>(
        formats.unpack_A_src, formats.unpack_B_src, formats.unpack_A_dst, formats.unpack_B_dst, FACE_R_DIM, FACE_R_DIM, 4 /* num_faces */, 4 /* num_faces */);
    _llk_unpack_A_init_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(
        0, 0, ckernel::make_tensor_shape_from_legacy(FACE_R_DIM, 4), formats.unpack_A_src, formats.unpack_A_dst);

    const std::uint32_t num_total_tiles = params.NUM_TILES_IN_BLOCK * params.NUM_BLOCKS;

    for (std::uint32_t tile = 0; tile < num_total_tiles; ++tile)
    {
        _llk_unpack_A_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(
            L1_ADDRESS(params.buffer_A[tile]), formats.unpack_A_src, formats.unpack_A_dst);
    }
}

#endif

#ifdef LLK_TRISC_MATH

#include "ckernel_sfpu.h"
#include "llk_lib_math_wrappers.h"
#include "llk_math_eltwise_unary_sfpu.h"
#include "llk_sfpu/ckernel_sfpu_reduce.h"
#include "params.h"

using namespace ckernel;
using namespace ckernel::sfpu;

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    // copy srca to dest
    _llk_math_eltwise_unary_datacopy_init_wrapper_<DataCopyType::A2D, is_fp32_dest_acc_en, BroadcastType::NONE, false /* is_int_fpu_en */, PackMode::Default>(
        4 /* num_faces */, formats.math);
    _llk_math_pack_sync_init_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
    _llk_math_hw_configure_<is_fp32_dest_acc_en>(formats.math, formats.math);

    // Row reduction (phase 2) needs every tile resident in dest at once, so the whole block is a
    // single dest section (NUM_BLOCKS == 1 is enforced by the Python harness).
    const std::uint32_t num_tiles = params.NUM_TILES_IN_BLOCK * params.NUM_BLOCKS;

    _llk_math_eltwise_unary_sfpu_init_<SfpuType::reduce>();

    // SINGLE shared init for both reduce passes (the crux of the multi-axis regression: the row
    // pass below trusts SFPU state established here / by the column pass, with no re-init between).
    ckernel::sfpu::init_reduce<POOL_TYPE, static_cast<DataFormat>(formats.math), is_fp32_dest_acc_en>();

    _llk_math_wait_for_dest_available_<DstSync::SyncHalf>();

    for (std::uint32_t tile = 0; tile < num_tiles; ++tile)
    {
        LLK_ASSERT(
            (tile < get_dest_max_tiles<DstSync::SyncHalf, is_fp32_dest_acc_en, DstTileShape::Tile32x32>()),
            "Block tile index exceeds maximum destination tiles");
        _llk_math_eltwise_unary_datacopy_<DataCopyType::A2D, DstSync::SyncHalf, is_fp32_dest_acc_en, BroadcastType::NONE, unpack_to_dest>(
            tile, formats.math, formats.math);
    }

    // Phase 1: column reduce each tile (REDUCE_COL). Leaves each col-tile's 32 per-column extremes
    // in its row 0.
    for (std::uint32_t tile = 0; tile < num_tiles; ++tile)
    {
        _llk_math_eltwise_sfpu_start_(tile);
        ckernel::sfpu::calculate_reduce<
            POOL_TYPE,
            ckernel::ReduceDim::REDUCE_COL,
            static_cast<DataFormat>(formats.math),
            is_fp32_dest_acc_en,
            static_cast<DataFormat>(formats.pack_dst)>();
    }

    // Phase 2: row reduce each row-tile (REDUCE_ROW). Reduces every tile's rows so each tile's
    // element [0][0] holds the extreme over its (already column-reduced) row 0, i.e. the extreme
    // over the whole 32x32 tile.
    _llk_math_eltwise_sfpu_start_(0);
    ckernel::sfpu::calculate_reduce<
        POOL_TYPE,
        ckernel::ReduceDim::REDUCE_ROW,
        static_cast<DataFormat>(formats.math),
        is_fp32_dest_acc_en,
        static_cast<DataFormat>(formats.pack_dst)>(BLOCK_CT_DIM, BLOCK_RT_DIM);

    _llk_math_eltwise_sfpu_done_();
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
    _llk_pack_hw_configure_wrapper_<is_fp32_dest_acc_en, PackMode::Default>(formats.pack_src, formats.pack_dst, 16 * 16 * 4 /* tile_size */);

    _llk_pack_init_wrapper_<PackMode::Default, false /* zero_output */>(formats.pack_dst);

    _llk_pack_dest_init_wrapper_<DstSync::SyncHalf, is_fp32_dest_acc_en, PackMode::Default>();

    const std::uint32_t num_tiles = params.NUM_TILES_IN_BLOCK * params.NUM_BLOCKS;

    // Each tile holds its own multi-axis reduction result at element [0][0]; pack them all so the
    // host can read one independent result per tile.
    _llk_packer_wait_for_math_done_();
    for (std::uint32_t tile = 0; tile < num_tiles; ++tile)
    {
        LLK_ASSERT(
            (tile < get_dest_max_tiles<DstSync::SyncHalf, is_fp32_dest_acc_en, DstTileShape::Tile32x32>()),
            "Block tile index exceeds maximum destination tiles");
        _llk_pack_<DstSync::SyncHalf, is_fp32_dest_acc_en, ckernel::PackMode::Default>(tile, L1_ADDRESS(params.buffer_Res[tile]));
    }
    _llk_pack_dest_section_done_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
}

#endif
