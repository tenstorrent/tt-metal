// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_math_common_api.h"
#include "experimental/llk_math_hadamard.h"

template <MathFidelity math_fidelity = MathFidelity::HiFi4, bool normalize = true>
inline void llk_math_hadamard_h128_init() {
    _llk_math_hadamard_h128_init_<math_fidelity, normalize>();
}

template <MathFidelity math_fidelity = MathFidelity::HiFi4, bool normalize = true>
inline void llk_math_hadamard_h128(uint32_t dst_index) {
    LLK_ASSERT((dst_index < get_dest_max_tiles<DST_SYNC_MODE, DST_ACCUM_MODE, DstTileShape::Tile32x32>()), "");
    _llk_math_hadamard_h128_<math_fidelity, normalize>(dst_index);
}

inline void llk_math_hadamard_h128_uninit() { _llk_math_hadamard_h128_uninit_(); }
