#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

namespace nd_sharding {
namespace detail {
namespace {
template <template <uint32_t...> class Wrapper, size_t BASE_IDX, size_t... Is>
constexpr auto make_struct_from_sequence_wrapper(std::index_sequence<Is...>)
    -> Wrapper<get_compile_time_arg_val(BASE_IDX + Is)...>;

template <std::size_t Base, std::size_t... Is>
constexpr std::array<uint32_t, sizeof...(Is)> make_crta_array_from_sequence(std::index_sequence<Is...>) {
    return {get_common_arg_val<uint32_t>(Base + Is)...};
}

template <std::size_t Base, std::size_t... Is>
constexpr std::array<uint32_t, sizeof...(Is)> make_rta_array_from_sequence(std::index_sequence<Is...>) {
    return {get_arg_val<uint32_t>(Base + Is)...};
}

}  // namespace

template <template <uint32_t...> class Wrapper, size_t base, uint32_t rank>
using struct_cta_sequence_wrapper_t =
    decltype(make_struct_from_sequence_wrapper<Wrapper, base>(std::make_index_sequence<rank>{}));

template <std::size_t Base, std::size_t Size>
constexpr auto array_crta_sequence_wrapper() {
    return make_crta_array_from_sequence<Base>(std::make_index_sequence<Size>{});
}

template <std::size_t Base, std::size_t Size>
constexpr auto array_rta_sequence_wrapper() {
    return make_rta_array_from_sequence<Base>(std::make_index_sequence<Size>{});
}

// Statically conditional static buffer
template <bool Enable, typename T = void, std::size_t N = 0>
struct ConditionalBuffer {
    T value[N];  // when Enable == true
};

template <std::size_t N, typename T>
struct ConditionalBuffer<false, T, N> {};

// Statically conditional field of class with type T
template <bool Emable, typename T = void>
struct ConditionalField {
    T value;
    template <typename T_>
    ConditionalField(T_&& val) : value(std::forward<T_>(val)) {}
    ConditionalField() = default;
};

template <typename T>
struct ConditionalField<false, T> {
    template <typename T_>
    ConditionalField(T_&& val) {}  // Ignore value if passed to constructor
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
}  // namespace nd_sharding
