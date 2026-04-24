// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/compute/common.h"

#ifdef TRISC_UNPACK
#include "debug/llk_hash_cb_api.h"
#endif
#ifdef TRISC_MATH
#include "debug/llk_hash_cb_api.h"
#endif
#ifdef TRISC_PACK
#include "debug/llk_hash_cb_api.h"
#endif

namespace ckernel {

// clang-format off
/**
 * Compute a single-u32 FNV-1a checksum over a circular buffer's L1 bytes and emit it
 * via DPRINT. Used to bisect non-deterministic kernels by diffing per-stage hashes
 * across runs: same input + same hash => stage is deterministic, look downstream;
 * same input + different hash => the op feeding that CB is the culprit.
 *
 * The printed line is stable across runs for diffing:
 *     hash[<label hex>] cb=<cb_id dec> tiles=<n dec> = <hash hex>
 *
 * Entirely gated on the compile flag DEBUG_CB_HASH; compiles to an empty inline when
 * undefined so there is zero overhead in release builds.
 *
 * Defaults to UNPACK because the CB read pointer (fifo_rd_ptr) is only populated
 * on the UNPACK thread — MATH has no cb_interface[] at all (UCK_CHLKC_MATH gate
 * in trisc.cc), and PACK tracks write pointers only. To move the probe onto a
 * different TRISC, wrap the call explicitly, e.g. PACK((llk_hash_cb(...))), and
 * arrange for that thread to have a valid L1 address by other means.
 *
 * Return value: None
 *
 * | Argument  | Description                                              | Type     | Valid Range | Required |
 * |-----------|----------------------------------------------------------|----------|-------------|----------|
 * | cb_id     | The index of the circular buffer (CB) to hash            | uint32_t | 0 to 31     | True     |
 * | num_tiles | The number of tiles from the front of the CB to include  | uint32_t | >= 1        | True     |
 * | label     | A caller-chosen tag to identify this probe in the output | uint32_t | any         | True     |
 */
// clang-format on
ALWI void hash_cb(uint32_t cb_id, uint32_t num_tiles, uint32_t label) {
    UNPACK((llk_hash_cb(cb_id, num_tiles, label)));
}

}  // namespace ckernel
