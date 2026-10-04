// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
// Include inside namespace sfpi.

inline vFloat raw_daz_action_coordinate(vFloat x_raw) {
    constexpr float min_normal = std::numeric_limits<float>::min();
    vFloat effective = x_raw;
    v_if(effective > -min_normal) { effective = 0.0f; }
    v_endif;
    v_if(x_raw >= min_normal) { effective = x_raw; }
    v_endif;
    return effective;
}
