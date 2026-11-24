// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Quasar 2.0 op hooks: the init claim, the debug no-tail check, and the Dest section size. Dest sync itself is the
// legacy tile_regs_acquire / commit / wait / release (api/compute/reg_api.h); ops that write Dest from UNPACK or the
// SFPU thread borrow the section from MATH inside the op (tile_regs_borrow / tile_regs_return).

#ifdef ARCH_QUASAR

#include <cstdint>

#include "api/compute/common_globals.h"
#include "llk_compute_api_claim.h"
#include "llk_dest_borrow.h"  // DstClient

#ifdef ENABLE_LLK_ASSERT
namespace llk_2_0 {
// Tiles per block op of each Dest client since that client's last init (0: none yet), indexed by DstClient.
// Debug only; defined in tt_metal/hw/firmware/src/tt-2xx/trisc.cc. See detail::check_block_tiles.
extern thread_local std::uint32_t block_tiles[4];
}  // namespace llk_2_0
#endif

namespace ckernel {
namespace experimental {
namespace detail {

// Op-init hook: claims the kernel for 2.0 and starts a new block size (check_block_tiles). Without LLK asserts the
// body is empty (the claim is compile time).
template <DstClient CLIENT>
inline void op_init() {
    LLK_COMPUTE_API_CLAIM(2_0, legacy);  // every 2.0 op init
#ifdef ENABLE_LLK_ASSERT
    llk_2_0::block_tiles[static_cast<int>(CLIENT)] = 0;
#endif
}

// clang-format off
/**
 * Debug-only check that block ops never take a tail (2.0 decision): every block op of Dest client `CLIENT` between two
 * of its inits must cover the same number of tiles, so the kernel's tile total divides evenly by it. The first block
 * op after an init (op_init) sets the count; a block op with another count asserts. LLK_ASSERT only: compiles to
 * nothing without LLK asserts.
 *
 * | Param Type | Name   | Description                            | Type      | Valid Range                     | Required |
 * |------------|--------|----------------------------------------|-----------|---------------------------------|----------|
 * | Template   | CLIENT | Dest client running the block op       | DstClient | UNPACK, MATH, SFPU, PACK        | True     |
 * | Function   | ntiles | Tiles this block op covers             | uint32_t  | == first block since the init   | True     |
 */
// clang-format on
template <DstClient CLIENT>
inline void check_block_tiles([[maybe_unused]] std::uint32_t ntiles) {
#ifdef ENABLE_LLK_ASSERT
    std::uint32_t& block = llk_2_0::block_tiles[static_cast<int>(CLIENT)];
    if (block == 0) {
        block = ntiles;
    }
    LLK_ASSERT(ntiles == block, "2.0 block op: tile count differs from the first block since init (no tails)");
#endif
}

// Dest tiles per section for the kernel's sync mode and Dest width.
template <bool is_fp32_dest_acc_en>
constexpr std::uint32_t dest_section_tiles() {
    return trisc::get_dest_max_tiles<DST_SYNC_MODE, is_fp32_dest_acc_en, trisc::DstTileShape::Tile32x32>();
}

}  // namespace detail
}  // namespace experimental
}  // namespace ckernel

#endif  // ARCH_QUASAR
