// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Quasar 2.0 Dest synchronization on Tensix semaphores (tt-llk llk_dest_chain.h).
//
// The ops hand Dest from unit to unit themselves: each op init joins its unit (UNPACK for unpack-to-dest, FPU, PACK)
// and each op takes Dest for its unit, once per block. A take by a unit that does not hold Dest hands it over: the
// holder drains its engine and posts a semaphore, the taker waits for it. dest_acquire / dest_release delimit a
// section: release hands Dest back to the section's first unit, and in SyncHalf the next section uses the other bank.
// Rules for kernel authors:
//   * Every thread runs the same program: inits, ops, dest_acquire/release are thread-common code (never inside
//     UNPACK((..)) / MATH((..)) / PACK((..))), so every thread sees every take in the same order.
//   * Every op runs inside a section; ops run in unit order inside it (e.g. copy then pack). Hand Dest over per block
//     of tiles, not per tile: every handoff drains an engine.
//   * At most get_dest_max_tiles<DST_SYNC_MODE, DST_ACCUM_MODE>() Dest tiles per section (8/4 SyncHalf fp16/fp32,
//     16/8 SyncFull).
//   * Nothing zeroes Dest between sections: an op that accumulates onto Dest (matmul) zeroes its bank when its turn
//     starts.
// After kernel_main, run_kernel (chlkc_list.h) calls _llk_dest_chain_end_, which returns the semaphores to 0.

#ifdef ARCH_QUASAR

#include <cstdint>

#include "api/compute/common_globals.h"
#include "llk_compute_api_claim.h"
#include "llk_dest_chain.h"

#define LLK_2_0_DEST_CHAIN 1  // run_kernel (chlkc_list.h) ends the chain after kernel_main

#ifdef ENABLE_LLK_ASSERT
namespace llk_2_0 {
// Tiles per block op of each Dest unit since that unit's last init (0: none yet), indexed by dest_dvalid_client.
// Debug only; defined in tt_metal/hw/firmware/src/tt-2xx/trisc.cc. See detail::check_block_tiles.
extern thread_local std::uint32_t block_tiles[4];
}  // namespace llk_2_0
#endif

namespace ckernel {
namespace experimental {
namespace detail {

// Join / take wrappers: one noinline copy per unit per thread, so every op call site costs a call (TRISC2 size).
template <dest_dvalid_client UNIT>
__attribute__((noinline)) void dest_join() {
    LLK_COMPUTE_API_CLAIM(2_0, legacy);  // every 2.0 op init joins
#ifdef ENABLE_LLK_ASSERT
    llk_2_0::block_tiles[static_cast<int>(UNIT)] = 0;  // an init starts a new block size (check_block_tiles)
#endif
    _llk_dest_chain_join_<UNIT>();
}

template <dest_dvalid_client UNIT>
__attribute__((noinline)) void dest_take() {
    _llk_dest_chain_take_<UNIT, DST_SYNC_MODE, DST_ACCUM_MODE>();
}

__attribute__((noinline)) inline void dest_release() { _llk_dest_chain_release_<DST_SYNC_MODE>(); }

// clang-format off
/**
 * Debug-only check that block ops never take a tail (2.0 decision): every block op of Dest unit `UNIT` between two
 * of its inits must cover the same number of tiles, so the kernel's tile total divides evenly by it. The first block
 * op after an init (dest_join) sets the count; a block op with another count asserts. LLK_ASSERT only: compiles to
 * nothing without LLK asserts.
 *
 * | Param Type | Name   | Description                            | Type               | Valid Range                     | Required |
 * |------------|--------|----------------------------------------|--------------------|---------------------------------|----------|
 * | Template   | UNIT   | Dest unit running the block op         | dest_dvalid_client | UNPACK, FPU, PACK               | True     |
 * | Function   | ntiles | Tiles this block op covers             | uint32_t           | == first block since the init   | True     |
 */
// clang-format on
template <dest_dvalid_client UNIT>
ALWI void check_block_tiles([[maybe_unused]] std::uint32_t ntiles) {
#ifdef ENABLE_LLK_ASSERT
    std::uint32_t& block = llk_2_0::block_tiles[static_cast<int>(UNIT)];
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

// clang-format off
/**
 * Opens a Dest section (Quasar 2.0). Emits nothing: the section's first op takes Dest from the unit the previous
 * section on this bank ended with (MATH at kernel start). Call on every thread (thread-common code), paired with
 * dest_release.
 *
 * | Param Type | Name | Description | Type | Valid Range | Required |
 * |------------|------|-------------|------|-------------|----------|
 * | (none)     |      |             |      |             |          |
 */
// clang-format on
ALWI void dest_acquire() {}

// clang-format off
/**
 * Closes the Dest section opened by dest_acquire (Quasar 2.0): the unit holding Dest drains and hands it back to the
 * section's first unit, so the next section on this bank starts as soon as it is done. In SyncHalf the next section
 * uses the other bank. An empty section emits nothing. Call on every thread (thread-common code).
 *
 * | Param Type | Name | Description | Type | Valid Range | Required |
 * |------------|------|-------------|------|-------------|----------|
 * | (none)     |      |             |      |             |          |
 */
// clang-format on
ALWI void dest_release() { detail::dest_release(); }

}  // namespace experimental
}  // namespace ckernel

#endif  // ARCH_QUASAR
