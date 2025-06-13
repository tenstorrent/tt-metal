// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

#include "const.hpp"
#include "helpers.hpp"

namespace nd_sharding {
namespace detail {
template <uint32_t... Elements>
struct ArrayStaticWrapper {
    constexpr static bool is_static = true;
    static constexpr uint32_t size = sizeof...(Elements);
    static constexpr std::array<uint32_t, size> elements = {Elements...};
    constexpr explicit ArrayStaticWrapper() = delete;
    static_assert(size > 0, "Array size must be greater than 0!");
};

struct ArrayDynamicWrapper {
    constexpr static bool is_static = false;
    static constexpr uint32_t size = static_cast<uint32_t>(-1);
    static constexpr std::array<uint32_t, 0> elements = {};
    constexpr explicit ArrayDynamicWrapper() = delete;
};

template <bool ShapeStatic, size_t StartIdx, uint32_t Size>
struct ArrayWrapperTypeSelector;

template <size_t StartIdx, uint32_t Size>
struct ArrayWrapperTypeSelector<true, StartIdx, Size> {
    // Both size and elements are known at compile time -- we can construct a static wrapper
    using type = struct_cta_sequence_wrapper_t<ArrayStaticWrapper, StartIdx, Size>;
};

template <size_t StartIdx, uint32_t Size>
struct ArrayWrapperTypeSelector<false, StartIdx, Size> {
    // Size maybe known at compile time, but elements are not
    using type = ArrayDynamicWrapper;
};

}  // namespace detail
}  // namespace nd_sharding
