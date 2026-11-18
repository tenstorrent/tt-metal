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
 * NOTE: On Quasar, use tile_regs_borrow to lend the acquired section before unpack writes to Dest.
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
 * Return any borrowed section before committing it.
 */
template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void tile_regs_commit() {
    MATH((llk_math_dest_section_done<is_fp32_dest_acc_en>()));
}

/**
 * Release lock on DST register by PACK thread. The lock had to be previously acquired with tile_regs_wait.
 */
template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void tile_regs_release() {
    PACK((llk_pack_dest_section_done<is_fp32_dest_acc_en>()));
}

#ifdef ARCH_QUASAR
/**
 * @brief Borrow a Dest bank from its owner.
 *
 * Each thread can hold only one loan at a time, and the lender must wait for its return before using the bank again.
 */
template <DstClient FROM, DstClient TO>
ALWI void tile_regs_borrow() {
    _llk_tile_regs_borrow_<FROM, TO>();
}

/**
 * @brief Return the borrowed DST section and make the lender wait until the borrower has finished.
 */
template <DstClient FROM, DstClient TO>
ALWI void tile_regs_return() {
    _llk_tile_regs_return_<FROM, TO>();
}
#endif

}  // namespace ckernel
