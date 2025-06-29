// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

#include "helpers.h"

namespace tensor_accessor {
template <typename T, T... Elements>
struct ArrayStaticWrapper {
    constexpr static bool is_static = true;
    static constexpr uint32_t size = sizeof...(Elements);
    static constexpr std::array<T, size> elements = {Elements...};
    constexpr explicit ArrayStaticWrapper() = delete;
    static_assert(size > 0, "Array size must be greater than 0!");
};

// Type aliases for convenience
template <uint32_t... Elements>
using ArrayStaticWrapperU32 = ArrayStaticWrapper<uint32_t, Elements...>;

template <uint16_t... Elements>
using ArrayStaticWrapperU16 = ArrayStaticWrapper<uint16_t, Elements...>;

struct ArrayDynamicWrapper {
    constexpr static bool is_static = false;
    static constexpr uint32_t size = static_cast<uint32_t>(-1);
    static constexpr std::array<uint32_t, 0> elements = {};
    constexpr explicit ArrayDynamicWrapper() = delete;
};

// Implementation for extracting two uint16_t values from each uint32_t CTA
template <typename IndexSeq, size_t BaseIdx>
struct make_u16_array_from_packed_u32_sequence;

template <size_t... Is, size_t BaseIdx>
struct make_u16_array_from_packed_u32_sequence<std::index_sequence<Is...>, BaseIdx> {
    template <size_t I>
    static constexpr uint16_t high_part() {
        return get_compile_time_arg_val(BaseIdx + I / 2) >> 16;
    }

    template <size_t I>
    static constexpr uint16_t low_part() {
        return get_compile_time_arg_val(BaseIdx + I / 2) & 0xffff;
    }

    using type = ArrayStaticWrapperU16<(Is % 2 == 0 ? low_part<Is>() : high_part<Is>())...>;
};

// Main wrapper to generate ArrayStaticWrapperU16 from sequence of uint32_t CTA values
template <size_t BaseIdx, size_t U16Count>
using struct_cta_sequence_wrapper_packed_u16_from_u32_t =
    typename make_u16_array_from_packed_u32_sequence<std::make_index_sequence<U16Count>, BaseIdx>::type;

template <bool ShapeStatic, size_t StartIdx, uint32_t Size>
struct ArrayWrapperTypeSelectorPackedU16;

template <size_t StartIdx, uint32_t Size>
struct ArrayWrapperTypeSelectorPackedU16<true, StartIdx, Size> {
    // Both size and elements are known at compile time -- we can construct a static wrapper
    using type = struct_cta_sequence_wrapper_packed_u16_from_u32_t<StartIdx, Size>;
};

template <size_t StartIdx, uint32_t Size>
struct ArrayWrapperTypeSelectorPackedU16<false, StartIdx, Size> {
    // Size maybe known at compile time, but elements are not
    using type = ArrayDynamicWrapper;
};

template <bool ShapeStatic, size_t StartIdx, uint32_t Size>
struct ArrayWrapperTypeSelectorU32;

template <size_t StartIdx, uint32_t Size>
struct ArrayWrapperTypeSelectorU32<true, StartIdx, Size> {
    // Both size and elements are known at compile time -- we can construct a static wrapper
    using type = detail::struct_cta_sequence_wrapper_t<ArrayStaticWrapperU32, StartIdx, Size>;
};

template <size_t StartIdx, uint32_t Size>
struct ArrayWrapperTypeSelectorU32<false, StartIdx, Size> {
    // Size maybe known at compile time, but elements are not
    using type = ArrayDynamicWrapper;
};

}  // namespace tensor_accessor
