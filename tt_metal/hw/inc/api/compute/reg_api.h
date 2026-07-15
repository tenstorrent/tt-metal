// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/compute/common_globals.h"
#ifdef TRISC_PACK
#include "llk_pack_common_api.h"
#endif

namespace ckernel {

/**
 * @deprecated This function is deprecated, please use `tile_regs_acquire()` instead.
 * See https://github.com/tenstorrent/tt-metal/issues/5868#issuecomment-2101726935
 *
 * Acquires an exclusive lock on the internal DST register for the current
 * Tensix core.
 *
 * This register is an array of 16 tiles of 32x32 elements each.
 * This is a blocking function, i.e. this function will wait until the lock is acquired.
 *
 * This is only available on the compute engine.
 *
 * Return value: None
 *
 * How the destination register will be shared and synchronized between TRISC threads will depend on the compute kernel
 * configuration.
 */
[[deprecated("Use tile_regs_acquire() instead")]]
ALWI void acquire_dst() {
    MATH((llk_math_wait_for_dest_available()));

    PACK((llk_packer_wait_for_math_done()));
}

// new APIs, TODO: migrate all kernels to these

/**
 * Acquire an exclusive lock on the DST register for the MATH thread.
 * This register is an array of 16 tiles of 32x32 elements each.
 * This is a blocking function, i.e. this function will wait until the lock is acquired.
 */
ALWI void tile_regs_acquire() {
    MATH((llk_math_wait_for_dest_available()));
#ifdef ARCH_QUASAR
    // Reset the packer's per-section unpack-to-dest flag to its default; copy_tile re-flags it per
    // operand for this section (see llk_pack_set_dest_filled_by_unpack). This is what makes a
    // copy_tile-free section (e.g. matmul/reduce-only) correctly take the regular path. A single
    // per-section flag cannot represent mixed operand provenance within one acquired section; a
    // section is expected to use one routing mode (general fix is programmable dest-dvalid).
    PACK((llk_pack_clear_dest_filled_by_unpack()));
#endif
}

/**
 * Acquire an exclusive lock on the DST register for the PACK thread.
 * It waits for the MATH thread to commit the DST register.
 * This is a blocking function, i.e. this function will wait until the lock is acquired.
 */
ALWI void tile_regs_wait() {
    PACK((llk_packer_wait_for_math_done()));
}

/**
 * @deprecated This function is deprecated, please use `tile_regs_release()` instead.
 * See https://github.com/tenstorrent/tt-metal/issues/5868#issuecomment-2101726935
 *
 * Releases the exclusive lock on the internal DST register for the current
 * Tensix core. This lock had to be previously acquired with acquire_dst. This
 * call is blocking and is only available on the compute engine.
 *
 * Return value: None
 *
 * How the destination register will be shared and synchronized between TRISC threads will depend on the compute kernel
 * configuration.
 */
[[deprecated("Use tile_regs_release() instead")]]
ALWI void release_dst() {
    MATH((llk_math_dest_section_done<DST_ACCUM_MODE>()));
    PACK((llk_pack_dest_section_done<DST_ACCUM_MODE>()));
}

// new APIs, TODO: migrate all kernels to these

/**
 * Release lock on DST register by MATH thread. The lock had to be previously acquired with tile_regs_acquire.
 */
ALWI void tile_regs_commit() { MATH((llk_math_dest_section_done<DST_ACCUM_MODE>())); }

/**
 * Release lock on DST register by PACK thread. The lock had to be previously acquired with tile_regs_wait.
 */
ALWI void tile_regs_release() {
    PACK((llk_pack_dest_section_done<DST_ACCUM_MODE>()));
}

}  // namespace ckernel
