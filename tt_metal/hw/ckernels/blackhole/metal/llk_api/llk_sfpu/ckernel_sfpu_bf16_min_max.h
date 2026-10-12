// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Include inside namespace sfpi, like the evaluator helpers consuming this.
// SFPI deleted vec_min_max; retain its former in-place implementation exactly.
// Both results come from one SFPSWAP, preserving operand and assignment order.
inline void ordered_min_max(vFloat& a, vFloat& b) {
    auto r = min_max(a, b);
    a = r.first;
    b = r.second;
}
