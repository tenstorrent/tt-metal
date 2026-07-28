// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>

#include "sfpi.h"

namespace ckernel::sfpu {

inline void smoothstep_tile_face(float edge0, [[maybe_unused]] float edge1, float inv_delta) {
    constexpr size_t vectors_per_face = 8;
    for (size_t i = 0; i < vectors_per_face; i++) {
        sfpi::vFloat x = sfpi::dst_reg[i];
        sfpi::vFloat t = (x - edge0) * inv_delta;
        t = sfpi::clamp(t, 0.0f, 1.0f);
        sfpi::vFloat result = t * t * (3.0f - 2.0f * t);
        sfpi::dst_reg[i] = result;
    }
}

}  // namespace ckernel::sfpu
