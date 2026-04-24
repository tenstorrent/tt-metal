// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#ifdef DEBUG_CB_HASH
#include "llk_math_common_api.h"
#include "experimental/llk_math_hash_cb.h"
#endif

/*************************************************************************
 * LLK MATH HASH CB (SFPU) - experimental
 *
 * Thin MATH-thread wrappers around the SFPU-backed FNV23 lanewise hash
 * defined in llk_lib/experimental/llk_math_hash_cb.h. See that file for the
 * algorithmic details and the "best-effort draft" hardware-validation caveat.
 *************************************************************************/

inline void llk_math_hash_cb_init() {
#ifdef DEBUG_CB_HASH
    ckernel::sfpu::_llk_math_hash_cb_init_();
#endif
}

// Accumulate one DEST tile (already unpacked to DEST at `dst_tile_idx`, INT32
// format) into the per-lane hash state.
inline void llk_math_hash_cb_tile(uint32_t dst_tile_idx) {
#ifdef DEBUG_CB_HASH
    ckernel::sfpu::_llk_math_hash_cb_tile_(dst_tile_idx);
#else
    (void)dst_tile_idx;
#endif
}

// Fold the 32 per-lane accumulators down to a single u32 in DEST[dst_tile_idx][0][0]
// ready for the packer to write to the cb_hash output CB.
inline void llk_math_hash_cb_reduce_and_store(uint32_t dst_tile_idx) {
#ifdef DEBUG_CB_HASH
    ckernel::sfpu::_llk_math_hash_cb_reduce_and_store_(dst_tile_idx);
#else
    (void)dst_tile_idx;
#endif
}
