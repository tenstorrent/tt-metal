// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include "api/compute/common.h"
#if defined(TRISC_MATH) && defined(ARCH_BLACKHOLE)
#include "experimental/llk_math_custom_mm_reuse_dest_srcb_api.h"
#endif
#if defined(TRISC_UNPACK) && defined(ARCH_BLACKHOLE)
#include "experimental/llk_unpack_AB_sdpa_custom_mm_reuse_dest_srcb_api.h"
#include "experimental/llk_unpack_A_sdpa_api.h"
#endif

namespace ckernel {

#if defined(ARCH_BLACKHOLE)

// DEST geometry.  A DEST "row" is 16 datums.
// custom_mm<dense_packing=true> output tile: 2 faces, 16 rows apart.
constexpr std::uint32_t CUSTOM_MM_DEST_TILE_ROWS = 32;
// custom_mm_reuse_dest_srcb accumulator tile: 2 faces, 8 rows apart.
constexpr std::uint32_t CUSTOM_MM_REUSE_DEST_TILE_ROWS = 16;
// A standard Tile32x32 DEST slot: 4 faces x 16 rows.
constexpr std::uint32_t CUSTOM_MM_DEST_ROWS_PER_TILE32 = 64;
// Rows addressable by math, for the current sync/accum configuration.
constexpr std::uint32_t CUSTOM_MM_MAX_DEST_ROWS =
    get_dest_max_tiles<DST_SYNC_MODE, DST_ACCUM_MODE, DstTileShape::Tile32x32>() * CUSTOM_MM_DEST_ROWS_PER_TILE32;

// clang-format off
/**
 * Points the packer at the dense accumulator layout custom_mm_reuse_dest_srcb
 * writes: faces 8 DEST rows apart, tiles 16.  Only the DEST-side read strides
 * change; the bytes packed to L1 are unchanged.
 *
 * Pair with custom_mm_reuse_dest_srcb_pack_uninit().  While active, the
 * `tile_index` given to pack_block_contiguous counts 16-row slots, not 32.
 */
// clang-format on
ALWI void custom_mm_reuse_dest_srcb_pack_init() {
    PACK((cfg_reg_rmw_tensix<PCK0_ADDR_CTRL_ZW_REG_0_Zstride_RMW>(FACE_C_DIM * 8 * 2)));
    PACK((cfg_reg_rmw_tensix<PCK0_ADDR_CTRL_ZW_REG_0_Wstride_RMW>((TILE_NUM_FACES / 2) * FACE_C_DIM * 8 * 2)));
}

/**
 * Restores both packer strides to their defaults (face 16 rows, tile 64).
 * A full llk_pack_init (skip_packer_strides = false) does reprogram these
 * fields; the restore matters for follow-on ops that pack without one, e.g.
 * MOP-only pack_block_contiguous paths, which would otherwise inherit the
 * dense strides.
 *
 * A chaining core therefore needs only this call, not also
 * custom_mm_block_uninit<dense_packing>().
 */
ALWI void custom_mm_reuse_dest_srcb_pack_uninit() {
    PACK((cfg_reg_rmw_tensix<PCK0_ADDR_CTRL_ZW_REG_0_Zstride_RMW>(FACE_C_DIM * FACE_R_DIM * 2)));
    PACK((cfg_reg_rmw_tensix<PCK0_ADDR_CTRL_ZW_REG_0_Wstride_RMW>(TILE_NUM_FACES * FACE_C_DIM * FACE_R_DIM * 2)));
}

/**
 * Loads the math replay program for custom_mm_reuse_dest_srcb_block.  It occupies a
 * disjoint slice of the FPU replay window from custom_mm's, so issue it alongside
 * custom_mm_block_init_short rather than between the two matmuls.
 */
ALWI void custom_mm_reuse_dest_srcb_replay_init() { MATH((llk_math_custom_mm_reuse_dest_srcb_replay_init())); }

// clang-format off
/**
 * Short initialization for custom_mm_reuse_dest_srcb_block.  Safe to call in the
 * middle of a kernel; it reprograms the math ADDR_MODs, so it must come after the
 * producing matmul's last math instruction.
 *
 * | Argument     | Description                                             | Type     | Valid Range            | Required |
 * |--------------|---------------------------------------------------------|----------|------------------------|----------|
 * | load_replay  | Also load the replay program (template)                 | bool     | true, false            | False    |
 * | in0_cb_id    | CB whose tile descriptor describes the DEST-resident in0 | uint32_t | 0 to 31                | True     |
 * | in1_cb_id    | CB holding the weights (unpacked into SrcA)             | uint32_t | 0 to 31                | True     |
 * | nt_dim       | Output width in tiles                                   | uint32_t | 1 to 16                | True     |
 *
 * Pass load_replay = false only when custom_mm_reuse_dest_srcb_replay_init() has
 * already run earlier in the same invocation; that keeps the replay load off the
 * chain between the two matmuls.
 */
// clang-format on
template <bool load_replay = true>
ALWI void custom_mm_reuse_dest_srcb_block_init_short(
    const std::uint32_t in0_cb_id, const std::uint32_t in1_cb_id, const std::uint32_t nt_dim) {
    UNPACK((llk_unpack_AB_sdpa_custom_mm_reuse_dest_srcb_init(in0_cb_id, in1_cb_id, /*transpose=*/0, nt_dim)));
    MATH((llk_math_custom_mm_reuse_dest_srcb_init<load_replay>()));
}

// clang-format off
/**
 * Matmul whose in0 is read out of DEST instead of a CB: computes
 * out[in0_tile_r_dim, nt_dim*32] += in0_from_dest[in0_tile_r_dim, kt_dim*32]
 *     @ in1[kt_dim*32, nt_dim*32].
 *
 * in0 is whatever the preceding custom_mm left at DEST row `isrc`, laid out
 * one tile every `src_tile_stride` rows; the math LLK copies `in0_tile_r_dim`
 * source rows per tile (1, 2, 4 or 8).  The result accumulates the same
 * `in0_tile_r_dim` output rows into DEST at row `idst` in the dense
 * 16-rows-per-tile layout.
 *
 * | Argument         | Description                                                    | Type     | Valid Range                    | Required |
 * |------------------|----------------------------------------------------------------|----------|--------------------------------|----------|
 * | in0_tile_r_dim   | Height of the in0 tile (template)                              | uint32_t | 1, 2, 4, 8                     | True     |
 * | in0_cb_id        | Currently unused: the unpack init reads geometry from in1 only and the math LLK takes no CB id (in0 geometry comes solely from in0_tile_r_dim). Kept for signature stability with the tt-blaze caller | uint32_t | 0 to 31                        | True     |
 * | in1_cb_id        | Weights CB                                                      | uint32_t | 0 to 31                        | True     |
 * | in1_tile_index   | First weight tile to read                                       | uint32_t | < CB size                      | True     |
 * | isrc             | DEST row of the first in0 tile                                  | uint32_t | < half DEST                    | True     |
 * | idst             | DEST row of the first output tile. The caller must zero DEST at idst first: the MVMULs accumulate (+=) and nothing in this call clears DEST | uint32_t | < half DEST                    | True     |
 * | kt_dim           | Inner dimension in tiles. All kt_dim in0 tiles must be DEST-resident at once: kt_dim * src_tile_stride + nt_dim * 16 <= CUSTOM_MM_MAX_DEST_ROWS, and the producing custom_mm's ct_dim caps it at 16 | uint32_t | even; see bound                | True     |
 * | nt_dim           | Output width in tiles                                           | uint32_t | 1 to 16                        | True     |
 * | in1_k_stride     | Weight tiles between consecutive K rows (nt_dim if contiguous)  | uint32_t | >= nt_dim                      | True     |
 * | src_tile_stride  | DEST rows between consecutive in0 tiles                         | uint32_t | 32 for custom_mm dense_packing, 64 without | True     |
 */
// clang-format on
template <std::uint32_t in0_tile_r_dim>
ALWI void custom_mm_reuse_dest_srcb_block(
    const std::uint32_t in0_cb_id,
    const std::uint32_t in1_cb_id,
    const std::uint32_t in1_tile_index,
    const std::uint32_t isrc,
    const std::uint32_t idst,
    const std::uint32_t kt_dim,
    const std::uint32_t nt_dim,
    const std::uint32_t in1_k_stride,
    const std::uint32_t src_tile_stride = CUSTOM_MM_DEST_TILE_ROWS) {
    UNPACK((llk_unpack_A_sdpa_set_srcb_dummy_valid()));
    UNPACK((llk_unpack_AB_sdpa_custom_mm_reuse_dest_srcb(
        in0_cb_id, in1_cb_id, /*tile_index_0=*/0, in1_tile_index, kt_dim, nt_dim, in1_k_stride)));
    MATH((llk_math_custom_mm_reuse_dest_srcb<in0_tile_r_dim>(isrc, idst, kt_dim, nt_dim, src_tile_stride)));
}

#endif  // ARCH_BLACKHOLE

}  // namespace ckernel
