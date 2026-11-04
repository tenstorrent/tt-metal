// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// TEMP experiment (GPT_OSS_DECODE_EXPERTS_SPLIT=1): compile-time arrays of the compressed custom_mm per-tile metadata
// words (fused_decode/mxfp4_split.py metadata_words). Each element is get_compile_time_arg_val(START + i).

#pragma once

#include <array>
#include <cstdint>

namespace split_meta {

template <std::uint32_t START, std::uint32_t COUNT, std::uint32_t I = 0>
constexpr void fill(std::array<std::uint32_t, COUNT>& arr) {
    arr[I] = get_compile_time_arg_val(START + I);
    if constexpr (I + 1 < COUNT) {
        fill<START, COUNT, I + 1>(arr);
    }
}

template <std::uint32_t START, std::uint32_t COUNT>
constexpr std::array<std::uint32_t, COUNT> cta_array() {
    std::array<std::uint32_t, COUNT> arr{};
    fill<START, COUNT>(arr);
    return arr;
}

}  // namespace split_meta
