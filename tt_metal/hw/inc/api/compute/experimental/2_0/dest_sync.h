// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Quasar 2.0 Dest synchronization on the take-driven dvalid chain (tt-llk llk_dest_chain.h).
//
// The ops hand Dest from unit to unit themselves: each op init joins its unit to the chain (no uninit; an op init whose
// unpacker does not write Dest takes UNPACK out of the chain) and each op execute takes Dest for its unit. A section
// ends with PACK's turn: in SyncHalf the next take ends that bank's use and opens the other bank; SyncFull is one
// handoff list on bank 0. dest_acquire/dest_release mark a section in the kernel source (2.0 API / Blackhole parity)
// and emit nothing on Quasar. Rules for kernel authors:
//   * Every thread runs the same program: inits, ops, dest_acquire/release are thread-common code (never inside
//     UNPACK((..)) / MATH((..)) / PACK((..))), so every thread sees every take in the same order.
//   * Ops run in client order inside a section (e.g. copy then pack); every section ends with a pack.
//   * At most get_dest_max_tiles<DST_SYNC_MODE, DST_ACCUM_MODE>() Dest tiles per section (8/4 SyncHalf fp16/fp32,
//     16/8 SyncFull).
//   * Nothing zeroes Dest between sections: an op that accumulates onto Dest (matmul) zeroes its bank when its turn
//     starts.
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
__attribute__((noinline)) void dest_leave() {
    _llk_dest_chain_leave_<UNIT>();
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
 * Marks the start of a Dest section (Quasar 2.0). Emits nothing: the section's first op takes Dest, and in SyncHalf
 * that take also ends the previous section's bank use. Kept for the 2.0 API surface (Blackhole parity). Call on every
 * thread (thread-common code), paired with dest_release.
 *
 * | Param Type | Name | Description | Type | Valid Range | Required |
 * |------------|------|-------------|------|-------------|----------|
 * | (none)     |      |             |      |             |          |
 */
// clang-format on
ALWI void dest_acquire() {}

// clang-format off
/**
 * Marks the end of the Dest section opened by dest_acquire (Quasar 2.0). Emits nothing: PACK, the section's last
 * unit, keeps Dest until the next section's first take, which hands it on (SyncHalf: to the other bank). Kept for the
 * 2.0 API surface (Blackhole parity). Call on every thread (thread-common code).
 *
 * | Param Type | Name | Description | Type | Valid Range | Required |
 * |------------|------|-------------|------|-------------|----------|
 * | (none)     |      |             |      |             |          |
 */
// clang-format on
ALWI void dest_release() {}

}  // namespace experimental
}  // namespace ckernel

#endif  // ARCH_QUASAR
