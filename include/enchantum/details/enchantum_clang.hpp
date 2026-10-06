#pragma once

// Clang <= 12 outputs "NUMBER" if casting
// Clang > 12 outputs "(E)NUMBER".

#if defined __has_warning
  #if __has_warning("-Wenum-constexpr-conversion")
    #pragma clang diagnostic push
    #pragma clang diagnostic ignored "-Wenum-constexpr-conversion"
  #endif
#endif

#include "../common.hpp"
#include "../type_name.hpp"
#include "shared.hpp"
#include "string_view.hpp"
#include <array>
#include <cassert>
#include <climits>
#include <cstdint>
#include <type_traits>
#include <utility>

namespace enchantum {

namespace details {
  constexpr auto enum_in_array_name(const std::string_view raw_type_name, const bool is_scoped_enum) noexcept
  {
    if (is_scoped_enum)
      return raw_type_name;

    if (const auto pos = raw_type_name.rfind(':'); pos != string_view::npos)
      return raw_type_name.substr(0, pos - 1);
    return string_view();
  }

#define SZC(x) (sizeof(x) - 1)

  template<auto... Vs>
  constexpr auto var_name() noexcept
  {
    // "auto enchantum::details::var_name() [Vs = <(A)0, a, b, c, e, d, (A)6>]"
    return __PRETTY_FUNCTION__ + SZC("auto enchantum::details::var_name() [Vs = <");
  }

  template<bool IsBitFlag, typename IntType>
  constexpr void parse_string(
    std::size_t         index_check,
    const char*         str,
    const std::size_t   least_length_when_casting,
    const std::size_t   least_length_when_value,
    const IntType       min,
    const std::size_t   array_size,
    const bool          null_terminated,
    IntType* const      values,
    std::uint8_t* const string_lengths,
    char* const         strings,
    std::size_t&        total_string_length,
    std::size_t&        valid_count)
  {
    (void)index_check;
    for (std::size_t index = 0; index < array_size; ++index) {
#if __clang_major__ > 12
      // check if cast (starts with '(')
      if (str[index_check] == '(')
#else
      // check if it is a number or negative sign
      if (str[0] == '-' || (str[0] >= '0' && str[0] <= '9'))
#endif
      {
        str = __builtin_char_memchr(str + least_length_when_casting, ',', UINT8_MAX) + SZC(", ");
      }
      else {
        str += least_length_when_value;
        const auto commapos = static_cast<std::size_t>(__builtin_char_memchr(str, ',', UINT8_MAX) - str);
        if constexpr (IsBitFlag)
          values[valid_count] = index == 0 ? IntType{} : static_cast<IntType>(IntType{1} << (index - 1));
        else
          values[valid_count] = static_cast<IntType>(min + static_cast<IntType>(index));
        string_lengths[valid_count++] = static_cast<std::uint8_t>(commapos);
        __builtin_memcpy(strings + total_string_length, str, commapos);
        total_string_length += commapos + null_terminated;
        str += commapos + SZC(", ");
      }
    }
  }

  template<typename E, bool NullTerminated, auto Min, std::size_t... Is>
  constexpr auto reflect(std::index_sequence<Is...>) noexcept
  {
    using MinT       = decltype(Min);
    using T          = std::underlying_type_t<E>;
    using Underlying = std::make_unsigned_t<std::conditional_t<std::is_same_v<bool, T>, unsigned char, T>>;

    constexpr auto elements_local = []() {
      constexpr auto ArraySize = sizeof...(Is) + is_bitflag<E>;
      const auto     str       = [](auto dependant) {
        constexpr bool always_true = sizeof(dependant) != 0;
        // dummy 0
        if constexpr (always_true && is_bitflag<E>) // sizeof... to make contest dependant
        {
          return details::var_name<static_cast<E>(!always_true), static_cast<E>(Underlying(1) << Is)..., 0>();
        }
        else {
          return details::var_name<static_cast<E>(static_cast<MinT>(Is) + Min)..., int(!always_true)>();
        }
      }(0);

      constexpr auto enum_in_array_name = details::enum_in_array_name(raw_type_name<E>, is_scoped_enum<E>);
      constexpr auto enum_in_array_len  = enum_in_array_name.size();
      // Ubuntu Clang 20 complains about using local constexpr variables in a local struct
      ReflectStringReturnValue<std::underlying_type_t<E>, ArraySize> ret;

      // ((anonymous namespace)::A)0
      // (anonymous namespace)::a
      // this is needed to determine whether the above are cast expression if 2 braces are
      // next to eachother then it is a cast but only for anonymoused namespaced enums
      constexpr std::size_t index_check = !enum_in_array_name.empty() && enum_in_array_name.front() == '(' ? 1 : 0;

      details::parse_string<is_bitflag<E>>(
        /*index_check=*/index_check,
        /*str = */ str,
#if __clang_major__ > 12
        /*least_length_when_casting=*/SZC("(") + enum_in_array_len + SZC(")0"),
#else
        /*least_length_when_casting=*/1,
#endif
        /*least_length_when_value=*/details::prefix_length_or_zero<E> +
          (enum_in_array_len != 0 ? enum_in_array_len + SZC("::") : 0),
        /*min = */ static_cast<T>(Min),
        /*array_size = */ ArraySize,
        /*null_terminated= */ NullTerminated,
        /*enum_values= */ ret.values,
        /*string_lengths= */ ret.string_lengths,
        /*strings= */ ret.strings,
        /*total_string_length*/ ret.total_string_length,
        /*valid_count*/ ret.valid_count);

      return ret;
    }();

    using Strings = std::array<char, elements_local.total_string_length>;

    struct {
      decltype(elements_local) elements;
      Strings                  strings;
    } data = {elements_local, [](const char* const strings) {
                Strings ret{};
                __builtin_memcpy(ret.data(), strings, ret.size());
                return ret;
              }(elements_local.strings)};

    return data;
  } // namespace details

} // namespace details


//template<Enum E>
//constexpr std::size_t enum_count = details::enum_count<E>;


} // namespace enchantum

#if defined __has_warning
  #if __has_warning("-Wenum-constexpr-conversion")
    #pragma clang diagnostic pop
  #endif
#endif
#undef SZC