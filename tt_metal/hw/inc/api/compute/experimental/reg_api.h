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

// Math clears the section it acquires, so the pack release skips its clear; only for kernels whose one DEST writer is
// math (a 32-bit unpack source fails to compile). End with tile_regs_math_only_end() before default acquires.
template <bool math_clears_dest = true>
ALWI void tile_regs_acquire_math_only() {
    MATH((llk_math_wait_for_dest_available<math_clears_dest>()));
}

template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void tile_regs_release_math_only() {
    PACK((llk_pack_dest_section_done<is_fp32_dest_acc_en, true>()));
}

// tile_regs_release() without the DMANOP pair after the offset write; fp32 DEST keeps the regular one, faster there.
template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void tile_regs_release_lean() {
    if constexpr (is_fp32_dest_acc_en) {
        tile_regs_release<is_fp32_dest_acc_en>();
    } else {
        PACK((llk_pack_dest_section_done<is_fp32_dest_acc_en, false, 3>()));
    }
}

// Math clears all of DEST once the pack released the last math-cleared section, as the default acquire expects.
ALWI void tile_regs_math_only_end() { MATH((llk_math_dest_clear_region_end())); }

}  // namespace ckernel
#else
namespace ckernel {
// Other architectures keep the one release sequence.
template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void tile_regs_release_lean() {
    tile_regs_release<is_fp32_dest_acc_en>();
}
}  // namespace ckernel
#endif
