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
    UNPACK((llk_unpack_wait_for_dest_available()));
    MATH((llk_math_wait_for_dest_available()));

    PACK((llk_packer_wait_for_math_done()));
}

// new APIs, TODO: migrate all kernels to these

/**
 * Acquire an exclusive lock on the DST register for the MATH thread.
 * This register is an array of 16 tiles of 32x32 elements each.
 * This is a blocking function, i.e. this function will wait until the lock is acquired.
 *
 * NOTE: On Quasar with unpack-to-dest the UNPACK thread also opens its DST section here. Every unpack-to-dest
 * copy_tile / copy_block / transpose_tile of the section must come after this call and before tile_regs_commit,
 * each at its own DST index; MATH does not start on the section before that commit.
 */
ALWI void tile_regs_acquire() {
    // The UNPACK step is a no-op except where the unpacker writes DEST under a per-section handshake (Quasar
    // unpack-to-dest). There it opens the unpack thread's DEST section, so every copy_tile / copy_block /
    // transpose_tile between acquire and commit lands in one section.
    UNPACK((llk_unpack_wait_for_dest_available()));
    MATH((llk_math_wait_for_dest_available()));
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
template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
[[deprecated("Use tile_regs_release() instead")]]
ALWI void release_dst() {
    UNPACK((llk_unpack_dest_section_done<is_fp32_dest_acc_en>()));
    MATH((llk_math_dest_section_done<is_fp32_dest_acc_en>()));
    PACK((llk_pack_dest_section_done<is_fp32_dest_acc_en>()));
}

// new APIs, TODO: migrate all kernels to these

/**
 * The UNPACK thread's half of tile_regs_commit: closes its DST section, handing it to MATH. A no-op except on Quasar
 * with unpack-to-dest, where MATH starts a section only after this hand-off, so the section's SrcA / SrcB unpacks
 * (matmul_block, unary_bcast, binary ops) stall both threads once the src banks are full unless it comes first.
 * A section with src-path work calls it after its last unpack-to-dest copy (or right after tile_regs_acquire if it has
 * none) and before its first src-path unpack, and then ends with tile_regs_math_commit, not tile_regs_commit.
 */
template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void tile_regs_unpack_commit() {
    UNPACK((llk_unpack_dest_section_done<is_fp32_dest_acc_en>()));
}

/**
 * The MATH thread's half of tile_regs_commit: ends a section whose UNPACK half went earlier in tile_regs_unpack_commit.
 */
template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void tile_regs_math_commit() {
    MATH((llk_math_dest_section_done<is_fp32_dest_acc_en>()));
}

/**
 * Release lock on DST register by MATH thread. The lock had to be previously acquired with tile_regs_acquire.
 *
 * NOTE: On Quasar with unpack-to-dest the UNPACK thread also closes its DST section here, handing it to MATH.
 */
template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void tile_regs_commit() {
    tile_regs_unpack_commit<is_fp32_dest_acc_en>();
    tile_regs_math_commit<is_fp32_dest_acc_en>();
}

/**
 * Release lock on DST register by PACK thread. The lock had to be previously acquired with tile_regs_wait.
 */
template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void tile_regs_release() {
    PACK((llk_pack_dest_section_done<is_fp32_dest_acc_en>()));
}

}  // namespace ckernel
