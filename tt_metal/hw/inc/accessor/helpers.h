// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <utility>
#include <cstddef>
#include <cstdint>

#include "compile_time_args.h"

namespace tensor_accessor {
namespace detail {
template <template <uint32_t...> class Wrapper, size_t BASE_IDX, size_t... Is>
constexpr auto make_struct_from_sequence_wrapper(std::index_sequence<Is...>)
    -> Wrapper<get_compile_time_arg_val(BASE_IDX + Is)...>;

template <template <uint32_t...> class Wrapper, size_t base, uint32_t rank>
using struct_cta_sequence_wrapper_t =
    decltype(make_struct_from_sequence_wrapper<Wrapper, base>(std::make_index_sequence<rank>{}));

// Statically conditional field of class with type T
template <bool Enable, typename T = void>
struct ConditionalField {
    T value;
    // Constructor that forwards a single argument
    template <typename T_>
    ConditionalField(T_&& val) : value(std::forward<T_>(val)) {}

    // Variadic constructor that forwards multiple arguments to T's constructor
    template <typename... Args>
    ConditionalField(Args&&... args) : value(std::forward<Args>(args)...) {}

    ConditionalField() = default;
};

template <typename T>
struct ConditionalField<false, T> {
    // Constructor that ignores a single argument
    template <typename T_>
    ConditionalField(T_&& val) {}  // Ignore value if passed to constructor

    // Variadic constructor that ignores all arguments
    template <typename... Args>
    ConditionalField(Args&&... args) {}  // Ignore all arguments if passed to constructor

    ConditionalField() = default;
};

template <typename T, bool Enable>
struct ConditionalStaticInstance {};

// Statically conditionall static instance of type T
template <typename T>
struct ConditionalStaticInstance<T, true> {
    static constexpr T instance{/* args */};
};

// Trait to detect operator[]
template <typename, typename = void>
struct has_subscript_operator : std::false_type {};

template <typename T>
struct has_subscript_operator<T, std::void_t<decltype(std::declval<T>()[std::declval<std::size_t>()])>>
    : std::true_type {};

template <typename T>
constexpr bool has_subscript_operator_v = has_subscript_operator<T>::value;

// Helper for template-dependent static_assert that only fails when instantiated
template <typename...>
constexpr bool always_false_v = false;

// No c++20 == no std::span :(
template <typename T>
struct Span {
    T* _data;
    std::size_t _size;

    constexpr Span() : _data(nullptr), _size(0) {}
    constexpr Span(T* data, std::size_t size) : _data(data), _size(size) {}

    T& operator[](std::size_t idx) const { return _data[idx]; }
    T* begin() const { return _data; }
    T* end() const { return _data + _size; }
    std::size_t size() const { return _size; }
};

}  // namespace detail
}  // namespace tensor_accessor
