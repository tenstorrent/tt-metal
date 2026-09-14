// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The logical parity-snake order, and nothing else.
//
// Split out of parity_snake.hpp so that cyclic_schedule.hpp can use it: the
// dense schedule is stated directly in terms of a core's position on the
// snake, while parity_snake.hpp depends on cyclic_schedule.hpp for its "no
// such core" sentinel. This header depends on neither.
//
// For C cores the order lists the even cores in increasing order and then the
// odd cores in decreasing order: 2, 4, 6, 8, 7, 5, 3, 1 for C = 8.

#pragma once

#include <cstdint>

namespace ttml::metal::ops::cyclic_sdpa_bw {

//: The core at snake position k. Precondition: k < C.
constexpr uint32_t snake_core_at(uint32_t C, uint32_t k) {
    const uint32_t evens = C / 2u;
    if (k < evens) {
        return 2u * (k + 1u);
    }
    const uint32_t largest_odd = (C % 2u != 0u) ? C : C - 1u;
    return largest_odd - 2u * (k - evens);
}

//: The snake position of core c, the inverse of snake_core_at.
//: Precondition: 1 <= c <= C.
constexpr uint32_t snake_index_of(uint32_t C, uint32_t c) {
    if (c % 2u == 0u) {
        return c / 2u - 1u;
    }
    const uint32_t largest_odd = (C % 2u != 0u) ? C : C - 1u;
    return C / 2u + (largest_odd - c) / 2u;
}

}  // namespace ttml::metal::ops::cyclic_sdpa_bw
