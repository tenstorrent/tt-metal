#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

namespace nd_sharding {
namespace detail {
namespace {
template <template <size_t...> class Wrapper, size_t BASE_IDX, size_t... Is>
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

template <template <size_t...> class Wrapper, size_t base, size_t rank>
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

template <bool Enable, typename T = void, std::size_t N = 0>
struct ConditionalBuffer {
    T value[N];  // when Enable == true
};

// Specialization for false: empty struct
template <std::size_t N, typename T>
struct ConditionalBuffer<false, T, N> {};

// Trait to detect operator[]
template <typename, typename = void>
struct has_subscript_operator : std::false_type {};

template <typename T>
struct has_subscript_operator<T, std::void_t<decltype(std::declval<T>()[std::declval<std::size_t>()])>>
    : std::true_type {};

template <typename T>
constexpr bool has_subscript_operator_v = has_subscript_operator<T>::value;

// No c++20 == to std::span :(
template <typename T>
struct Span {
    T* _data;
    std::size_t _size;

    constexpr Span(T* data, std::size_t size) : _data(data), _size(size) {}

    T& operator[](std::size_t idx) const { return _data[idx]; }
    T* begin() const { return _data; }
    T* end() const { return _data + _size; }
    std::size_t size() const { return _size; }
};

}  // namespace detail
}  // namespace nd_sharding
