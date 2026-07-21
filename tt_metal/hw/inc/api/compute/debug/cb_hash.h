// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"  // copy_tile (UNPACK + MATH A→DEST)
#include "dev_mem_map.h"

#ifndef ARCH_QUASAR
#ifdef TRISC_UNPACK
#include "debug/llk_hash_cb_api.h"
#endif
#ifdef TRISC_MATH
#include "debug/llk_hash_cb_api.h"
#include "debug/llk_math_hash_cb_api.h"
#endif
#ifdef TRISC_PACK
#include "debug/llk_hash_cb_api.h"
#endif
#endif  // !ARCH_QUASAR

namespace ckernel {

// ===========================================================================
// CB hash probes for bisecting non-deterministic kernels. Both hash an input
// CB's contents and are gated on the compile flag DEBUG_CB_HASH (zero overhead
// when off). Pick the variant by what you can afford to disturb:
//
//   hash_cb_trisc()  — TRISC scalar, no Tensix Engine state touched. Runs as
//                      plain RISC-V on the UNPACK thread (override possible) and
//                      DPRINTs the hash in a diff-friendly line:
//                          hash[<label hex>] cb=<cb dec> tiles=<n dec> = <hash hex>
//                      Cheap to drop in anywhere; the safe default when the SFPU
//                      or DEST pipeline is itself under suspicion.
//                      Algorithm: FNV-1a-32 (single u32, full 32-bit).
//
//   hash_cb_sfpu()   — Lanewise SFPU hash on MATH. Exercises the same TRISC
//                      pipeline as production compute (SFPU + DEST + the
//                      DEST -> PACK -> L1 path), so it can surface races that
//                      hash_cb_trisc hides. Uses DEST and SFPU LReg state —
//                      unsuitable when those are themselves under suspicion.
//                      Does not print: it leaves a result tile in DEST for the
//                      caller to pack out and XOR-fold host-side.
//                      Algorithm: 23-bit FNV-like ("FNV23"), driven by the SFPU
//                      multiplier width (SFPMUL24 on BH; shift-and-add on WH).
//                      Not bit-equal to hash_cb_trisc — compare SFPU to SFPU only.
// ===========================================================================

// clang-format off
/**
 * Scalar (TRISC-side) CB hash. Runs FNV-1a-32 over the L1 bytes of `cb_id` on
 * whichever TRISC dispatches the LLK — UNPACK by default. The resulting hash
 * is printed via DPRINT in a stable diff-friendly format:
 *     hash[0x<label>] cb=<cb_id> tiles=<n> = 0x<hash>
 *
 * Pure RISC-V scalar code: no Tensix Engine instructions are issued and no
 * DEST / SFPU state is touched, which makes this the safe default when the
 * SFPU or DEST pipeline is itself under suspicion.
 *
 * Defaults to UNPACK because the CB read pointer (fifo_rd_ptr) is only
 * populated on the UNPACK thread — cb_interface[] is not even allocated on
 * MATH (UCK_CHLKC_MATH gate in trisc.cc) and PACK tracks write pointers only.
 * To move the probe onto a different TRISC, call llk_hash_cb_trisc directly
 * inside a PACK((...)) / MATH((...)) macro and arrange for that thread to see
 * a valid L1 address by other means.
 *
 * Return value: None
 *
 * | Param Type | Name | Description | Type | Valid Range | Required |
 * |------------|------|-------------|------|-------------|----------|
 * | Function | cb_id | Index of the circular buffer to hash | uint32_t | 0 to 31 | True |
 * | Function | num_tiles | Tiles from the front of the CB to include | uint32_t | >= 1 | True |
 * | Function | label | Caller tag echoed in the DPRINT line | uint32_t | any | True |
 */
// clang-format on
#ifndef ARCH_QUASAR
ALWI void hash_cb_trisc(uint32_t cb_id, uint32_t num_tiles, uint32_t label) {
    UNPACK((llk_hash_cb_trisc(cb_id, num_tiles, label)));
}
#else
ALWI void hash_cb_trisc(uint32_t cb_id, uint32_t num_tiles, uint32_t label) {
    (void)cb_id;
    (void)num_tiles;
    (void)label;
}
#endif  // !ARCH_QUASAR

// clang-format off
/**
 * SFPU-side CB hash. Computes a 23-bit ("FNV23") lanewise multiplicative hash
 * over the tile data of `in_cb` after it has been unpacked/moved to DEST (not
 * the raw L1 bytes — the values are subject to the INT32 SFPLOAD UnshuffleFP32
 * permutation), then leaves the result in DEST slot 0 for the caller to pack
 * out: DEST row 0 holds the 32 per-lane accumulators and the rest of the tile
 * is zeroed. The caller packs that tile to its output CB; a host/scalar consumer
 * XOR-folds the whole tile, which yields XOR(32 accumulators) — a deterministic,
 * input-sensitive fingerprint independent of the SFPU lane <-> tile-position
 * permutation. See the tt-llk standalone test (tests/sources/hash_cb_test.cpp)
 * and its pytest for the end-to-end fold.
 *
 * This routes through the proven SFPU -> DEST -> PACK -> L1 path; it does NOT
 * use the DEST debug-bus read-back (which does not round-trip 32-bit DEST).
 *
 * Trade-offs versus hash_cb_trisc:
 *   - Touches SFPU LReg state and clobbers DEST slot 0. Wrap the call in
 *     tile_regs_acquire and pack DEST slot 0 afterwards (tile_regs_commit/wait).
 *   - Input CB must hold INT32 tiles; the kernel must be built with the
 *     unpack-to-dest path enabled for 32-bit formats (the standard config for
 *     INT32).
 *   - Produces a tile (not a printed line): the caller packs DEST and folds the
 *     result host-side. Hash value is NOT bit-equal to hash_cb_trisc's FNV-1a-32.
 *
 * Return value: None
 *
 * | Param Type | Name | Description | Type | Valid Range | Required |
 * |------------|------|-------------|------|-------------|----------|
 * | Function | in_cb | CB holding the tiles to hash (INT32 format expected) | uint32_t | 0 to 31 | True |
 * | Function | num_tiles | Tiles from the front of in_cb to include | uint32_t | >= 1 | True |
 */
// clang-format on
#ifndef ARCH_QUASAR
ALWI void hash_cb_sfpu(uint32_t in_cb, uint32_t num_tiles) {
#ifdef DEBUG_CB_HASH
    MATH((llk_math_hash_cb_init()));
    cb_wait_front(in_cb, num_tiles);

    // Per tile: copy_tile drives UNPACK (CB → SRC or DEST per UnpackToDestEn)
    // and MATH (SRC → DEST when applicable). After it returns, DEST slot 0
    // holds the tile's INT32 values for the SFPU hash step to fold in.
    for (uint32_t t = 0; t < num_tiles; ++t) {
        copy_tile(in_cb, t, /*dst_tile_index=*/0);
        MATH((llk_math_hash_cb_tile(/*dst_tile_idx=*/0)));
    }

    cb_pop_front(in_cb, num_tiles);

    // Write the 32 per-lane accumulators back into DEST slot 0 (rest zeroed)
    // so the caller can pack the result tile to L1 and fold it host-side.
    MATH((llk_math_hash_cb_store_to_dest()));
#else
    (void)in_cb;
    (void)num_tiles;
#endif
}
#else
ALWI void hash_cb_sfpu(uint32_t in_cb, uint32_t num_tiles) {
    (void)in_cb;
    (void)num_tiles;
}
#endif  // !ARCH_QUASAR

}  // namespace ckernel
