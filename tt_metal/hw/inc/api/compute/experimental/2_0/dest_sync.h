// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Quasar 2.0 Dest synchronization on the take-driven dvalid chain (tt-llk llk_dest_chain.h).
//
// The kernel owns every Dest section: dest_acquire() opens it, dest_release() closes it. Inside a section the ops
// hand Dest from unit to unit themselves: each op init joins its unit to the chain (no uninit; nothing leaves until
// kernel end) and each op execute takes Dest for its unit. Rules for kernel authors:
//   * Every thread runs the same program: inits, ops, dest_acquire/release are thread-common code (never inside
//     UNPACK((..)) / MATH((..)) / PACK((..))), so every thread sees every take in the same order.
//   * Ops run in client order inside a section (e.g. copy then pack). A section must contain at least one op.
//   * At most get_dest_max_tiles<DST_SYNC_MODE, DST_ACCUM_MODE>() Dest tiles per section (8/4 SyncHalf fp16/fp32,
//     16/8 SyncFull).
// compute_kernel_hw_startup resets the chain at kernel start.

#ifdef ARCH_QUASAR

#include <cstdint>

#include "api/compute/common_globals.h"
#include "llk_dest_chain.h"

namespace ckernel {
namespace experimental {
namespace detail {

// Join / take wrappers: one noinline copy per unit per thread, so every op call site costs a call (TRISC2 size).
template <dest_dvalid_client UNIT>
__attribute__((noinline)) void dest_join() {
    _llk_dest_chain_join_<UNIT>();
}

template <dest_dvalid_client UNIT>
__attribute__((noinline)) void dest_take() {
    _llk_dest_chain_take_<UNIT, DST_SYNC_MODE, DST_ACCUM_MODE>();
}

// Dest tiles per section for the kernel's sync mode and Dest width.
template <bool is_fp32_dest_acc_en>
constexpr std::uint32_t dest_section_tiles() {
    return trisc::get_dest_max_tiles<DST_SYNC_MODE, is_fp32_dest_acc_en, trisc::DstTileShape::Tile32x32>();
}

}  // namespace detail

// clang-format off
/**
 * Opens a Dest section (Quasar 2.0). Bookkeeping only: the first op of the section takes Dest. Call on every thread
 * (thread-common code), paired with dest_release.
 *
 * | Param Type | Name | Description | Type | Valid Range | Required |
 * |------------|------|-------------|------|-------------|----------|
 * | (none)     |      |             |      |             |          |
 */
// clang-format on
ALWI void dest_acquire() { _llk_dest_chain_section_begin_(); }

// clang-format off
/**
 * Closes the Dest section opened by dest_acquire (Quasar 2.0): the last unit that took Dest releases it and the
 * section's bank is published to the next section's first unit. In SyncHalf the next section uses the other bank.
 * The section must contain at least one op. Call on every thread (thread-common code).
 *
 * | Param Type | Name | Description | Type | Valid Range | Required |
 * |------------|------|-------------|------|-------------|----------|
 * | (none)     |      |             |      |             |          |
 */
// clang-format on
__attribute__((noinline)) inline void dest_release() {
    _llk_dest_chain_section_end_<DST_SYNC_MODE, DST_ACCUM_MODE>();
}

}  // namespace experimental
}  // namespace ckernel

#endif  // ARCH_QUASAR
