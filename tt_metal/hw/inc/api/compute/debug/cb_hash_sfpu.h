// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/compute/common.h"
#include "api/compute/cb_api.h"
#include "api/compute/matmul.h"  // for acquire_dst / release_dst helpers

#ifdef TRISC_UNPACK
#include "internal/circular_buffer_interface.h"
#include "api/debug/dprint.h"
#include "llk_unpack_A_api.h"
#endif
#ifdef TRISC_MATH
#include "experimental/llk_math_hash_cb_api.h"
#include "llk_math_unary_datacopy_api.h"
#endif
#ifdef TRISC_PACK
#include "llk_pack_api.h"
#endif

namespace ckernel {

// clang-format off
/**
 * SFPU-backed variant of hash_cb. Computes a 32-lane-parallel 23-bit
 * multiplicative hash (FNV23) over the contents of `in_cb`, packs the single
 * u32 result to `out_cb`, and DPRINTs it from the UNPACK thread.
 *
 * Unlike the plain-scalar hash_cb in cb_hash.h, this variant:
 *   - uses DEST (1 row) and SFPU state -- unsuitable when DEST/SFPU is itself
 *     under suspicion.
 *   - requires the caller to own an output CB (`out_cb`, 1 tile, INT32 format)
 *     dedicated to hash results. The caller's host-side program must configure
 *     this CB; typical convention is to call it `cb_hash`.
 *   - produces a different hash value than the scalar variant -- the algorithm
 *     is 23-bit FNV-like, not exact FNV-1a-32 (see the LLK-lib comment for
 *     the rationale: SFPMUL24 is 23b on Blackhole).
 *   - is Blackhole-only in this PR; see llk_math_hash_cb.h for WH B0 status.
 *
 * The printed line shares the scalar variant's diff-friendly format:
 *     hash[<label hex>] cb=<in_cb dec> tiles=<n dec> = <hash hex>
 *
 * Entirely gated on DEBUG_CB_HASH; empty inline when the flag is off.
 *
 * Return value: None
 *
 * | Argument  | Description                                                    | Type     | Valid Range | Required |
 * |-----------|----------------------------------------------------------------|----------|-------------|----------|
 * | in_cb     | CB holding the tiles to hash (INT32 format expected)           | uint32_t | 0 to 31     | True     |
 * | num_tiles | Number of tiles from the front of in_cb to include             | uint32_t | >= 1        | True     |
 * | out_cb    | 1-tile output CB (INT32 format) to receive the hash word       | uint32_t | 0 to 31     | True     |
 * | label     | Caller tag identifying this probe in the DPRINT output         | uint32_t | any         | True     |
 */
// clang-format on
ALWI void hash_cb_sfpu(uint32_t in_cb, uint32_t num_tiles, uint32_t out_cb, uint32_t label) {
#ifdef DEBUG_CB_HASH
#ifdef ARCH_BLACKHOLE
    // MATH: set up the per-lane accumulator once for the probe.
    MATH((llk_math_hash_cb_init()));

    // Per-tile: UNPACK moves the tile into DEST[0]; MATH folds it into the hash.
    UNPACK((llk_wait_tiles(in_cb, num_tiles)));

    for (uint32_t t = 0; t < num_tiles; ++t) {
        UNPACK((llk_unpack_A(in_cb, t)));
        MATH((llk_math_hash_cb_tile(/*dst_tile_idx=*/0)));
    }

    UNPACK((llk_pop_tiles(in_cb, num_tiles)));

    // MATH: reduce 32 lanes -> lane 0 and store into DEST[0][0].
    MATH((llk_math_hash_cb_reduce_and_store(/*dst_tile_idx=*/0)));

    // PACK: write the single-element hash tile to the output CB.
    PACK((llk_wait_for_free_tiles<false, false, false>(out_cb, 1)));
    PACK((llk_pack<false, false>(/*dst_tile_idx=*/0, out_cb)));
    PACK((llk_push_tiles<false, false>(out_cb, 1)));

    // UNPACK: read the first u32 of the packed output tile out of L1 and DPRINT it
    // using the same line format as the scalar variant so both hash streams diff
    // the same way.
    UNPACK((llk_wait_tiles(out_cb, 1)));
    UNPACK({
        const uint32_t hash_bytes_addr = get_local_cb_interface(out_cb).fifo_rd_ptr << cb_addr_shift;
        const uint32_t h = *reinterpret_cast<volatile uint32_t*>(hash_bytes_addr);
        DPRINT << "hash[0x" << HEX() << label << "] cb=" << DEC() << in_cb << " tiles=" << num_tiles << " = 0x" << HEX()
               << h << DEC() << ENDL();
    });
    UNPACK((llk_pop_tiles(out_cb, 1)));
#else
    // SFPU variant is Blackhole-only in this PR -- fall through to nothing on WH.
    (void)in_cb;
    (void)num_tiles;
    (void)out_cb;
    (void)label;
#endif
#else
    (void)in_cb;
    (void)num_tiles;
    (void)out_cb;
    (void)label;
#endif
}

}  // namespace ckernel
