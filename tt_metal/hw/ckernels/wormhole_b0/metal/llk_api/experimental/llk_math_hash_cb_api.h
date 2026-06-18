// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#ifdef DEBUG_CB_HASH
#include "llk_math_common_api.h"
#include "experimental/llk_math_hash_cb.h"
#endif

// Wormhole B0 mirror of the SFPU LLK-API surface. The compute-API shim gates
// the SFPU variant on ARCH_BLACKHOLE and never calls these on WH, but the
// header is provided for symmetry with the scalar-variant layout so the same
// include pattern compiles on both archs.

inline void llk_math_hash_cb_init() {
#ifdef DEBUG_CB_HASH
    ckernel::sfpu::_llk_math_hash_cb_init_();
#endif
}

inline void llk_math_hash_cb_tile(uint32_t dst_tile_idx) {
#ifdef DEBUG_CB_HASH
    ckernel::sfpu::_llk_math_hash_cb_tile_(dst_tile_idx);
#else
    (void)dst_tile_idx;
#endif
}

inline void llk_math_hash_cb_reduce_and_store(uint32_t dst_tile_idx) {
#ifdef DEBUG_CB_HASH
    ckernel::sfpu::_llk_math_hash_cb_reduce_and_store_(dst_tile_idx);
#else
    (void)dst_tile_idx;
#endif
}
