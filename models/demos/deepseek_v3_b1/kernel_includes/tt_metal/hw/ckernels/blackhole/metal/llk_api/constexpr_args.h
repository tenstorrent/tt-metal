// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include "api/compile_time_args.h"

namespace compressed {

/**
 * @brief Fill a constexpr array from positional compile-time args.
 *
 * Usage:
 *   constexpr auto arr = fill_cta_array<uint32_t, START_IDX, COUNT>();
 *
 * Each element is get_compile_time_arg_val(START_IDX + i).
 */
template <typename T, size_t START_IDX, size_t COUNT, size_t I = 0>
struct CTAArrayFiller {
    static constexpr void fill(std::array<T, COUNT>& arr) {
        arr[I] = static_cast<T>(get_compile_time_arg_val(START_IDX + I));
        if constexpr (I + 1 < COUNT) {
            CTAArrayFiller<T, START_IDX, COUNT, I + 1>::fill(arr);
        }
    }
};

template <typename T, size_t START_IDX, size_t COUNT>
constexpr std::array<T, COUNT> fill_cta_array() {
    std::array<T, COUNT> arr{};
    if constexpr (COUNT > 0) {
        CTAArrayFiller<T, START_IDX, COUNT>::fill(arr);
    }
    return arr;
}

}  // namespace compressed
