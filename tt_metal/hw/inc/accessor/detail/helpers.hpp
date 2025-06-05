#pragma once

#include <array>
#include <cstddef>

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
}  // namespace detail
}  // namespace nd_sharding
