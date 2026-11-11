// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/compute/reg_api.h"

#ifdef ARCH_BLACKHOLE
#ifdef TRISC_MATH
#include "llk_math_common_api.h"
#endif

namespace ckernel {

/**
 * Acquire and clear a DST section on MATH, avoiding the race between PACK's
 * ZEROACC and writes into the other half of double-buffered DST on Blackhole.
 *
 * Use this and tile_regs_release_math_clear() for EVERY section in the kernel;
 * do not mix with ordinary acquire/release. Commit and wait are unchanged.
 * Keep DST writes between acquire and commit; commit drains previous writes.
 * Unpack-to-DST operations must wait for MATH's ready signal (as copy_tile does).
 * Independent unpacker/SFPU DST writers are not supported. This invalidates rows,
 * not their contents: initialise complete rows before SFPU reads or partial writes.
 */
template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void tile_regs_acquire_math_clear() {
    tile_regs_acquire();
    MATH((_llk_math_clear_dest_section_<DST_SYNC_MODE, is_fp32_dest_acc_en>()));
}

/**
 * Wait for PACK to finish, release the section and advance its half without
 * clearing DST. Must be paired with tile_regs_acquire_math_clear() throughout
 * the kernel so MATH clears each section before reusing it.
 */
ALWI void tile_regs_release_math_clear() {
    PACK((_llk_pack_dest_section_done_<DST_SYNC_MODE, DST_ACCUM_MODE, false /* clear_dest */>()));
}

}  // namespace ckernel
#endif
