// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "compute_kernel_api/common_globals.h"

namespace ckernel {

/**
 * Acquires an exclusive lock on the internal DST register for the current
 * Tensix core.
 *
 * This register is an array of 16 tiles of 32x32 elements each.
 * This is a blocking function, i.e. this function will wait until the lock is acquired.
 *
 * This is only available on the compute engine.
 *
 * DOX-TODO(Describe meanings of dst_mode values).
 *
 * Return value: None
 *
 * | Argument | Description                                                | Type     | Valid Range         | Required |
 * |----------|------------------------------------------------------------|----------|---------------------|----------|
 * | dst_mode | Specifies how the destination register is going to be used | DstMode  | Full, Half, Tile    | True     |
 */
ALWI void acquire_dst(tt::DstMode mode) {
    MATH(( llk_math_wait_for_dest_available<SYNC>()  ));

    PACK(( llk_packer_wait_for_math_done()  ));
}

// new APIs, TODO: migrate all kernels to these
ALWI void tile_regs_acquire() {
    MATH(( llk_math_wait_for_dest_available<SYNC>()  ));
}

ALWI void tile_regs_wait() {
    PACK(( llk_packer_wait_for_math_done()  ));
}

/**
 * Releases the exclusive lock on the internal DST register for the current
 * Tensix core. This lock had to be previously acquired with acquire_dst. This
 * call is blocking and is only available on the compute engine.
 *
 * Return value: None
 *
 * DOX-TODO(Describe meanings of dst_mode values).
 *
 * | Argument | Description                                                | Type     | Valid Range                                 | Required |
 * |----------|------------------------------------------------------------|----------|---------------------------------------------|----------|
 * | dst_mode | Specifies how the destination register is going to be used | uint32_t | DstMode::Full, DstMode::Half, DstMode::Tile | True     |
 */
ALWI void release_dst(tt::DstMode mode) {
    MATH(( llk_math_dest_section_done<SYNC>()  ));

    PACK(( llk_pack_dest_section_done<SYNC>()  ));
}

// new APIs, TODO: migrate all kernels to these
ALWI void tile_regs_commit() {
    MATH(( llk_math_dest_section_done<SYNC>()  ));
}

ALWI void tile_regs_release() {
    PACK(( llk_pack_dest_section_done<SYNC>()  ));
}

} // namespace ckernel
