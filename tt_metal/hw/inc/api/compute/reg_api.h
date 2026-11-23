// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/compute/common_globals.h"
#ifdef ARCH_QUASAR
#include "llk_dest_borrow.h"
#endif
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
 *
 * NOTE: On Quasar, unpack-to-dest copies need a loan from MATH: wrap them in tile_regs_borrow and
 * tile_regs_return.
 */
ALWI void tile_regs_acquire() { MATH((llk_math_wait_for_dest_available())); }

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
template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
[[deprecated("Use tile_regs_release() instead")]]
ALWI void release_dst() {
    MATH((llk_math_dest_section_done<is_fp32_dest_acc_en>()));
    PACK((llk_pack_dest_section_done<is_fp32_dest_acc_en>()));
}

// new APIs, TODO: migrate all kernels to these

/**
 * Release lock on DST register by MATH thread. The lock had to be previously acquired with tile_regs_acquire.
 *
 * NOTE: On Quasar, call tile_regs_return for any loan of this section first.
 */
template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void tile_regs_commit() {
    MATH((llk_math_dest_section_done<is_fp32_dest_acc_en>()));
}

/**
 * Release lock on DST register by PACK thread. The lock had to be previously acquired with tile_regs_wait.
 *
 * NOTE: On Quasar, call tile_regs_return for any loan of this section first.
 */
template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void tile_regs_release() {
    PACK((llk_pack_dest_section_done<is_fp32_dest_acc_en>()));
}

#ifdef ARCH_QUASAR
/**
 * Lends the current Dest section from its owner FROM to TO (Quasar only). FROM is MATH between
 * tile_regs_acquire and tile_regs_commit, or PACK between tile_regs_wait and tile_regs_release.
 * TO is UNPACK, for unpack-to-dest copies, or SFPU. Call it on every thread, like the other
 * tile_regs_* calls; only FROM and TO do any work.
 *
 * NOTE: Until tile_regs_return, FROM must not touch Dest, and no thread can start a second loan.
 *
 * NOTE: Nothing checks these rules at runtime. Without a borrow, the unpacker writes Dest with no
 * sync and can land in the wrong section. Without a return, the threads lose sync and can hang.
 */
template <DstClient FROM, DstClient TO>
ALWI void tile_regs_borrow() {
    _llk_tile_regs_borrow_<FROM, TO>();
}

/**
 * Ends a Dest loan started by tile_regs_borrow with the same FROM and TO (Quasar only). TO waits
 * for its Dest writes to finish, and FROM waits for TO. After this call FROM owns the section
 * again and can use it, commit it, or release it.
 */
template <DstClient FROM, DstClient TO>
ALWI void tile_regs_return() {
    _llk_tile_regs_return_<FROM, TO>();
}
#endif

}  // namespace ckernel
