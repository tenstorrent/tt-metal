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
    static constexpr uint32_t size = sizeof...(Elements);
    using ShapeBase = std::array<uint32_t, size>;
    static constexpr ShapeBase elements = {Elements...};

    static_assert(size > 0, "Array size must be greater than 0!");

    constexpr explicit ArrayStaticWrapper() = default;
    constexpr explicit ArrayStaticWrapper(const ShapeBase&) {}
    constexpr explicit ArrayStaticWrapper(ShapeBase&&) {}
};

template <uint32_t Size>
struct ArrayStaticSizeDynamicElementsWrapper {
    static constexpr uint32_t size = Size;
    using ShapeBase = std::array<uint32_t, size>;
    ShapeBase elements;
    static_assert(size > 0, "Shape size must be greater than 0!");

    constexpr explicit ArrayStaticSizeDynamicElementsWrapper() = default;
    constexpr explicit ArrayStaticSizeDynamicElementsWrapper(const ShapeBase& elements) : elements{elements} {}
    constexpr explicit ArrayStaticSizeDynamicElementsWrapper(ShapeBase&& elements) : elements{std::move(elements)} {}
};

struct ArrayDynamicWrapper {
    static constexpr uint32_t size = static_cast<uint32_t>(-1);  // Size is not known at compile time
    using ShapeBase = Span<uint32_t>;
    ShapeBase elements;

    constexpr explicit ArrayDynamicWrapper() = default;
    constexpr explicit ArrayDynamicWrapper(const ShapeBase& elements) : elements(elements) {}
    explicit ArrayDynamicWrapper(ShapeBase&& elements) : elements(std::move(elements)) {}
};

template <bool SizeCTA, bool ShapeStatic, size_t StartIdx, uint32_t Size>
struct ArrayWrapperTypeSelector;

template <size_t StartIdx, uint32_t Size>
struct ArrayWrapperTypeSelector<true, true, StartIdx, Size> {
    // Both size and elements are known at compile time -- we can construct a static wrapper
    using type = struct_cta_sequence_wrapper_t<ArrayStaticWrapper, StartIdx, Size>;
};

template <size_t StartIdx, uint32_t Size>
struct ArrayWrapperTypeSelector<true, false, StartIdx, Size> {
    // Size is known at compile time, but elements are not
    using type = ArrayStaticSizeDynamicElementsWrapper<Size>;
};

template <bool ShapeStatic, size_t StartIdx, uint32_t Size>
struct ArrayWrapperTypeSelector<false, ShapeStatic, StartIdx, Size> {
    // Size is not known at compile time, doesn't matter if elements are known or not, use poorly dynamic wrapper
    using type = ArrayDynamicWrapper;
};

}  // namespace detail
}  // namespace nd_sharding
