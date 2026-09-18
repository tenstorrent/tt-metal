// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_math_common_api.h"
#include "experimental/llk_math_hadamard.h"
#include "sanitizer/api.h"

template <MathFidelity math_fidelity = MathFidelity::HiFi4, bool normalize = true>
inline void llk_math_hadamard_h128_init() {
    SAN_HOOK(unsupported());
    _llk_math_hadamard_h128_init_<math_fidelity, normalize>();
}

template <MathFidelity math_fidelity = MathFidelity::HiFi4, bool normalize = true>
inline void llk_math_hadamard_h128(uint32_t dst_index) {
    SAN_HOOK(unsupported());
    LLK_ASSERT((dst_index < get_dest_max_tiles_rt<DST_SYNC_MODE, DstTileShape::Tile32x32>()), "");
    _llk_math_hadamard_h128_<math_fidelity, normalize>(dst_index);
}

inline void llk_math_hadamard_h128_uninit() {
    SAN_HOOK(unsupported());
    _llk_math_hadamard_h128_uninit_();
}
