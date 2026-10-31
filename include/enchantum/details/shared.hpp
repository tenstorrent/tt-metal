#pragma once
#include "../common.hpp"
#include <array>
#include <climits>
#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace enchantum::details {


template<typename E, typename = void>
inline constexpr std::size_t prefix_length_or_zero = 0;

template<typename E>
inline constexpr auto prefix_length_or_zero<E, decltype((void)enum_traits<E>::prefix_length)> = std::size_t{
  enum_traits<E>::prefix_length};

  template<typename Underlying, std::size_t ArraySize>
  struct ReflectStringReturnValue {
    Underlying   values[ArraySize]{};
    std::uint8_t string_lengths[ArraySize]{};
    // the sum of all character names must be less than the size of this array
    // no one will likely hit this unless you for some odd reason have extremely long names
    char         strings[1024*8]{};
    std::size_t  total_string_length = 0;
    std::size_t  valid_count         = 0;
  };


} // namespace enchantum::details