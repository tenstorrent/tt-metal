// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// -----------------------------------------------------------------------------
// pack_rows_to_addr  (Blackhole experimental compute API, merged #51361)
//
// Header under test: api/compute/experimental/pack_rows_to_addr.h
//
//   ALWI void pack_rows_to_addr_init(uint32_t num_rows) { PACK((llk_pack_rows_init(num_rows))); }
//   ALWI void pack_rows_to_addr(uint32_t idst, uint32_t l1_addr) { PACK((_llk_pack_rows_(idst, l1_addr - 1))); }
//   ALWI void pack_rows_to_addr_uninit()                { PACK((llk_pack_rows_uninit()));
//                                                          PACK((_llk_pack_configure_addrmod_<PackMode::Default>())); }
//
// Difference vs the base pack_rows path (sources/pack_rows_test.cpp):
//   * base test:  _llk_pack_rows_(tile, L1_ADDRESS(buffer_Res[t]))               // packs to a CB tile
//   * to-addr:    pack_rows_to_addr(tile, l1_addr) -> _llk_pack_rows_(tile, l1_addr - 1)
//     i.e. the compute-API wrapper decrements the caller-supplied L1 address by 1
//     before handing it to the raw LLK. It is the "arbitrary L1 address" path used
//     by cache-update ops; it bypasses the CB dst_index bound check.
//
// This unit test harness is TRISC-split (LLK_TRISC_PACK etc.) and cannot include
// the metal compute-API header (PACK()/ALWI belong to the compute_kernel_api
// framework, not the llk_lib harness). We therefore mirror the compute API's
// exact math at the LLK layer: the caller computes the target address the way a
// pack_rows_to_addr caller does (target = L1_ADDRESS(buffer_Res[t]) + 1), and we
// feed (target - 1) into _llk_pack_rows_ — reproducing the wrapper's `l1_addr - 1`.
// The net effect is that data lands at L1_ADDRESS(buffer_Res[t]), exactly where
// the golden reads it back, which is what validates the address computation.
//
// GOLDEN (see helpers/golden_generators.py::PackRowsGolden):
//   Each 32x32 DEST tile is 1024 datums in row-major (64 rows x 16 datums).
//   _llk_pack_rows_init_(N) sets the packer to pack N rows of 16 datums each.
//   The op writes the first N rows (N*16 datums) of tile `tile` starting at the
//   target byte address. So golden = first (num_rows_to_pack * 16) datums of each
//   input tile, concatenated over tiles. Only those defined datums are validated.
// -----------------------------------------------------------------------------

#include <cstdint>

#include "ckernel.h"
#include "llk_defs.h"

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
        formats.unpack_A_src, formats.unpack_B_src, formats.unpack_A_dst, formats.unpack_B_dst, FACE_R_DIM, FACE_R_DIM, 4 /*num_faces*/, 4 /*num_faces*/);
    _llk_unpack_A_init_<BroadcastType::NONE, false /*acc_to_dest*/, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(
        0 /*transpose_of_faces*/, 0 /*within_face_16x16_transpose*/, ckernel::DEFAULT_TENSOR_SHAPE, formats.unpack_A_src, formats.unpack_A_dst);

    const int num_total_tiles = params.NUM_TILES_IN_BLOCK * params.NUM_BLOCKS;

    for (int tile = 0; tile < num_total_tiles; ++tile)
    {
        _llk_unpack_A_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(
            L1_ADDRESS(params.buffer_A[tile]), formats.unpack_A_src, formats.unpack_A_dst);
    }
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
    const bool is_int_fpu_en = false;

    _llk_math_hw_configure_<is_fp32_dest_acc_en>(formats.math, formats.math);
    _llk_math_eltwise_unary_datacopy_init_wrapper_<DataCopyType::A2D, is_fp32_dest_acc_en, BroadcastType::NONE, is_int_fpu_en, PackMode::Default>(
        4 /*num_faces*/, formats.math);
    _llk_math_pack_sync_init_<DstSync::SyncHalf, is_fp32_dest_acc_en>();

    const std::uint32_t num_blocks         = params.NUM_BLOCKS;
    const std::uint32_t num_tiles_in_block = params.NUM_TILES_IN_BLOCK;

    for (std::uint32_t block = 0; block < num_blocks; ++block)
    {
        _llk_math_wait_for_dest_available_<DstSync::SyncHalf>();
        for (std::uint32_t tile = 0; tile < num_tiles_in_block; ++tile)
        {
            LLK_ASSERT(
                (tile < get_dest_max_tiles<DstSync::SyncHalf, is_fp32_dest_acc_en, DstTileShape::Tile32x32>()),
                "Block tile index exceeds maximum destination tiles");
            _llk_math_eltwise_unary_datacopy_<DataCopyType::A2D, DstSync::SyncHalf, is_fp32_dest_acc_en, BroadcastType::NONE, unpack_to_dest>(
                tile, formats.math, formats.math);
        }
        _llk_math_dest_section_done_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
    }
}

#endif

#ifdef LLK_TRISC_PACK

#include "llk_lib_pack_wrappers.h"
#include "llk_pack_common.h"
#include "llk_pack_rows.h"
#include "params.h"

// Mirror of api/compute/experimental/pack_rows_to_addr.h at the llk layer.
// The compute API's pack_rows_to_addr(idst, l1_addr) forwards (l1_addr - 1) to
// _llk_pack_rows_. We reproduce that decrement here so the golden readback at
// L1_ADDRESS(buffer_Res[t]) validates the address computation.
static inline void pack_rows_to_addr_llk(std::uint32_t idst, std::uint32_t l1_addr)
{
    _llk_pack_rows_(idst, l1_addr - 1);
}

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    const bool UNTILIZE = false;

    _llk_pack_hw_configure_<is_fp32_dest_acc_en, ckernel::PackMode::Default>(formats.pack_src, formats.pack_dst, FACE_R_DIM * FACE_C_DIM * TILE_NUM_FACES);
    _llk_pack_dest_init_wrapper_<DstSync::SyncHalf, is_fp32_dest_acc_en, llk_test_pack_mode_v<UNTILIZE, false>>();
    // pack_rows_to_addr_init(num_rows) -> llk_pack_rows_init(num_rows) -> _llk_pack_rows_init_(num_rows)
    _llk_pack_rows_init_(params.NUM_ROWS_TO_PACK);

    const std::uint32_t num_blocks         = params.NUM_BLOCKS;
    const std::uint32_t num_tiles_in_block = params.NUM_TILES_IN_BLOCK;

    for (std::uint32_t block = 0; block < num_blocks; ++block)
    {
        _llk_packer_wait_for_math_done_();
        for (std::uint32_t tile = 0; tile < num_tiles_in_block; ++tile)
        {
            std::uint32_t res_tile_idx = (block * num_tiles_in_block) + tile;
            LLK_ASSERT(
                (tile < get_dest_max_tiles<DstSync::SyncHalf, is_fp32_dest_acc_en, DstTileShape::Tile32x32>()),
                "Block tile index exceeds maximum destination tiles");
            // Caller supplies the target the way a pack_rows_to_addr caller does:
            // target = L1_ADDRESS(buffer_Res[t]) + 1. The wrapper's `l1_addr - 1`
            // then lands data exactly at L1_ADDRESS(buffer_Res[t]).
            const std::uint32_t target_l1_addr = L1_ADDRESS(params.buffer_Res[res_tile_idx]) + 1;
            pack_rows_to_addr_llk(tile, target_l1_addr);
        }
        _llk_pack_dest_section_done_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
    }
    // pack_rows_to_addr_uninit() -> llk_pack_rows_uninit() (+ Default addrmod restore
    // in the compute API; the raw _llk_pack_rows_uninit_ is the harness equivalent).
    _llk_pack_rows_uninit_();
}

#endif
