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
// CB hash probes for bisecting non-deterministic kernels.
//
// Both probes hash the L1 bytes of an input CB, DPRINT the result in a stable
// diff-friendly format, and are gated on the compile flag DEBUG_CB_HASH (zero
// overhead when off).
//
// Pick the variant by what you can afford to disturb and what you want to
// catch:
//
//   hash_cb_trisc()       — TRISC scalar, no Tensix Engine state touched.
//                     Runs as plain RISC-V on the UNPACK thread (override
//                     possible). Cheap to drop in anywhere. Use when you
//                     want a baseline hash that does not perturb the
//                     compute pipeline.
//                     Algorithm: FNV-1a-32 (single u32, full 32-bit).
//
//   hash_cb_sfpu()  — Lanewise SFPU hash on MATH, with a DEST → L1 → UNPACK
//                     round trip. Exercises the same TRISC pipeline as
//                     production compute (SFPU + DEST + cross-TRISC L1
//                     handoff), so it can surface races that
//                     hash_cb_trisc hides. Uses DEST and SFPU LReg state —
//                     unsuitable when those are themselves under suspicion.
//                     Algorithm: 23-bit FNV-like ("FNV23"), driven by the
//                     SFPU multiplier width (SFPMUL24 on BH; shift-and-add
//                     on WH). Not bit-equal to hash_cb_trisc — only compare
//                     SFPU-hashes to SFPU-hashes.
//
// Output format (both variants):
//     hash[<label hex>] cb=<cb_id dec> tiles=<n dec> = <hash hex>
// ===========================================================================

// ---------------------------------------------------------------------------
// L1 hand-off slot for hash_cb_sfpu (MATH → UNPACK).
//
// hash_cb_sfpu has MATH stash the reduced hash at a fixed offset inside the
// reserved MEM_LLK_DEBUG region, then UNPACK polls a ready flag and DPRINTs.
// Using the existing debug region avoids requiring callers to allocate an
// output CB. The slot is placed at offset 64 to stay clear of the checkpoint
// state (api/debug/checkpoint.h, ~16 bytes at offset 0). hash_cb_sfpu and
// DEBUG_CHECKPOINT are both debug-only tools and should not be needed
// concurrently.
// ---------------------------------------------------------------------------
constexpr uint32_t DEBUG_HASH_L1_OFFSET = 64;
constexpr uint32_t DEBUG_HASH_L1_HASH_ADDR = MEM_LLK_DEBUG_BASE + DEBUG_HASH_L1_OFFSET;
constexpr uint32_t DEBUG_HASH_L1_READY_ADDR = MEM_LLK_DEBUG_BASE + DEBUG_HASH_L1_OFFSET + 4;

// Compile-time guard: if the debug region shrinks or the offset moves, this
// fires instead of silently corrupting adjacent memory.
static_assert(
    DEBUG_HASH_L1_OFFSET + 8 <= MEM_LLK_DEBUG_SIZE,
    "DEBUG_HASH_L1 slot (hash u32 + ready u32) exceeds the MEM_LLK_DEBUG region");

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
 * | Argument  | Description                                              | Type     | Valid Range | Required |
 * |-----------|----------------------------------------------------------|----------|-------------|----------|
 * | cb_id     | The index of the circular buffer (CB) to hash            | uint32_t | 0 to 31     | True     |
 * | num_tiles | The number of tiles from the front of the CB to include  | uint32_t | >= 1        | True     |
 * | label     | A caller-chosen tag to identify this probe in the output | uint32_t | any         | True     |
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
 * permutation). Routes the single-u32 result through L1 (MATH writes the
 * reduced hash to a fixed slot in the MEM_LLK_DEBUG region, UNPACK polls a
 * ready flag and reads it back) before printing via DPRINT in the same format
 * as hash_cb_trisc:
 *     hash[0x<label>] cb=<cb_id> tiles=<n> = 0x<hash>
 *
 * The L1 round trip is deliberate: it exercises the same MATH → L1 → UNPACK
 * handoff that production compute kernels use, so this variant can flag
 * cross-TRISC races that the engine-neutral hash_cb hides.
 *
 * Trade-offs versus hash_cb_trisc:
 *   - Touches SFPU LReg state and DEST slot 0. Wrap the call in
 *     tile_regs_acquire / tile_regs_commit, and expect DEST slot 0 to be
 *     clobbered.
 *   - Input CB must hold INT32 tiles; the kernel must be built with the
 *     unpack-to-dest path enabled for 32-bit formats (the standard config
 *     for INT32). See the tt-llk standalone test for an end-to-end example.
 *   - Hash value is NOT bit-equal to hash_cb_trisc's FNV-1a-32. Compare SFPU
 *     hashes only to other SFPU hashes.
 *
 * No caller-side output buffer is required — the L1 slot lives at a fixed
 * address inside the reserved MEM_LLK_DEBUG region (see DEBUG_HASH_L1_*
 * constants above). hash_cb_sfpu and DEBUG_CHECKPOINT share that region and
 * are not safe to invoke concurrently.
 *
 * Return value: None
 *
 * | Argument  | Description                                              | Type     | Valid Range | Required |
 * |-----------|----------------------------------------------------------|----------|-------------|----------|
 * | in_cb     | CB holding the tiles to hash (INT32 format expected)     | uint32_t | 0 to 31     | True     |
 * | num_tiles | The number of tiles from the front of in_cb to include   | uint32_t | >= 1        | True     |
 * | label     | A caller-chosen tag to identify this probe in the output | uint32_t | any         | True     |
 */
// clang-format on
#ifndef ARCH_QUASAR
ALWI void hash_cb_sfpu(uint32_t in_cb, uint32_t num_tiles, uint32_t label) {
#ifdef DEBUG_CB_HASH
    // UNPACK pre-clears the ready flag so the post-compute poll only fires on
    // MATH's write for *this* call.
    UNPACK((llk_hash_cb_sfpu_reset_ready(DEBUG_HASH_L1_READY_ADDR)));

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

    // MATH reduces to lane 0, reads the result out of DEST, stashes it in L1,
    // then publishes the ready flag.
    MATH((llk_math_hash_cb_finish_to_l1(DEBUG_HASH_L1_HASH_ADDR, DEBUG_HASH_L1_READY_ADDR)));

    // UNPACK polls the ready flag, reads the hash back out of L1, and prints
    // in the same line format as hash_cb_trisc for diff-tool parity.
    UNPACK((llk_hash_cb_sfpu_print_from_l1(
        DEBUG_HASH_L1_HASH_ADDR, DEBUG_HASH_L1_READY_ADDR, in_cb, num_tiles, label)));
#else
    (void)in_cb;
    (void)num_tiles;
    (void)label;
#endif
}
#else
ALWI void hash_cb_sfpu(uint32_t in_cb, uint32_t num_tiles, uint32_t label) {
    (void)in_cb;
    (void)num_tiles;
    (void)label;
}
#endif  // !ARCH_QUASAR

}  // namespace ckernel
