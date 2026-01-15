// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/compute/common_globals.h"

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
#ifndef ARCH_QUASAR
    MATH((llk_math_wait_for_dest_available()));

    PACK((llk_packer_wait_for_math_done()));
#endif
}

// new APIs, TODO: migrate all kernels to these

/**
 * Acquire an exclusive lock on the DST register for the MATH thread.
 * This register is an array of 16 tiles of 32x32 elements each.
 * This is a blocking function, i.e. this function will wait until the lock is acquired.
 *
 * On Quasar, this function is a no-op.
 */
ALWI void tile_regs_acquire() {
#ifndef ARCH_QUASAR
    MATH((llk_math_wait_for_dest_available()));
#endif
}

/**
 * Acquire an exclusive lock on the DST register for the PACK thread.
 * It waits for the MATH thread to commit the DST register.
 * This is a blocking function, i.e. this function will wait until the lock is acquired.
 *
 * On Quasar, this function is a no-op.
 */
ALWI void tile_regs_wait() {
#ifndef ARCH_QUASAR
    PACK((llk_packer_wait_for_math_done()));
#endif
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
#ifndef ARCH_QUASAR
    MATH((llk_math_dest_section_done<DST_ACCUM_MODE>()));
    PACK((llk_pack_dest_section_done<DST_ACCUM_MODE>()));
#else
    // TODO: Expand programmability in order to support the dest dvalid scheme with different clients
    // either FPU or SFPU can clear dvalid
    MATH((llk_math_set_dvalid<p_cleardvalid::FPU>()));
    PACK((llk_pack_dest_dvalid_section_done<DST_SYNC_MODE, DST_ACCUM_MODE>()));
#endif
}

// new APIs, TODO: migrate all kernels to these

/**
 * Release lock on DST register by MATH thread. The lock had to be previously acquired with tile_regs_acquire.
 */
ALWI void tile_regs_commit() {
#ifndef ARCH_QUASAR
    MATH((llk_math_dest_section_done<DST_ACCUM_MODE>()));
#else
    // TODO: Expand programmability in order to support the dest dvalid scheme with different clients
    // either FPU or SFPU can clear dvalid
    MATH((llk_math_set_dvalid<p_cleardvalid::FPU>()));
#endif
}

/**
 * Release lock on DST register by PACK thread. The lock had to be previously acquired with tile_regs_wait.
 */
ALWI void tile_regs_release() {
#ifndef ARCH_QUASAR
    PACK((llk_pack_dest_section_done<DST_ACCUM_MODE>()));
#else
    PACK((llk_pack_dest_dvalid_section_done<DST_SYNC_MODE, DST_ACCUM_MODE>()));
#endif
}

}  // namespace ckernel
