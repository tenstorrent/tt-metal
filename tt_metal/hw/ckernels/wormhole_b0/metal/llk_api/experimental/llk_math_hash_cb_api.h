// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#ifdef DEBUG_CB_HASH
#include "llk_math_common_api.h"
#include "experimental/llk_math_hash_cb.h"
#endif

// ===========================================================================
// LLK MATH HASH CB (SFPU) — MATH-side surface for hash_cb_sfpu.
//
// Wraps the SFPU-backed FNV23 lanewise hash in llk_lib/experimental/
// llk_math_hash_cb.h. Orchestration (UNPACK + cross-thread handoff) is in
// api/compute/debug/cb_hash.h; see that header for the user-facing contract.
//
// Call sequence:
//   1. llk_math_hash_cb_init()                — seed per-lane accumulators
//   2. llk_math_hash_cb_tile(dst_tile_idx)    — fold one DEST tile in
//      (per input tile)
//   3. llk_math_hash_cb_finish_to_l1(...)     — reduce 32 → 1, read out of
//                                              DEST, write u32 + ready flag
//                                              to L1 for UNPACK to consume
// ===========================================================================

inline void llk_math_hash_cb_init() {
#ifdef DEBUG_CB_HASH
    ckernel::sfpu::_llk_math_hash_cb_init_();
#endif
}

// Accumulate one DEST tile (already unpacked to DEST at `dst_tile_idx`, INT32
// format) into the 32 per-lane FNV23 accumulators.
inline void llk_math_hash_cb_tile(uint32_t dst_tile_idx) {
#ifdef DEBUG_CB_HASH
    ckernel::sfpu::_llk_math_hash_cb_tile_(dst_tile_idx);
#else
    (void)dst_tile_idx;
#endif
}

// Reduce the 32 per-lane accumulators to a single u32, drain the SFPU
// pipeline, read the result out of DEST via the debug array path, write it
// to `l1_hash_addr`, then publish `l1_ready_addr = 1` to release the UNPACK
// reader. l1_*_addr both live inside the MEM_LLK_DEBUG region — see the
// DEBUG_HASH_L1_* constants in api/compute/debug/cb_hash.h.
inline void llk_math_hash_cb_finish_to_l1(uint32_t l1_hash_addr, uint32_t l1_ready_addr) {
#ifdef DEBUG_CB_HASH
    ckernel::sfpu::_llk_math_hash_cb_finish_to_l1_(l1_hash_addr, l1_ready_addr);
#else
    (void)l1_hash_addr;
    (void)l1_ready_addr;
#endif
}
