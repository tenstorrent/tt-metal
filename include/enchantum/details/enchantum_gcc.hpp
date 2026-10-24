#include "../common.hpp"
#include "../type_name.hpp"
#include "shared.hpp"
#include <array>
#include <cassert>
#include <climits>
#include <cstdint>
#include <type_traits>
#include <utility>

#include "string_view.hpp"

namespace enchantum {
namespace details {
#define SZC(x) (sizeof(x) - 1)
  // this is needed since gcc transforms "{anonymous}" into "<unnamed>" for values
  template<auto Enum>
  constexpr auto enum_in_array_name_size() noexcept
  {
    // constexpr auto f() [with auto _ = (
    //constexpr auto f() [with auto _ = (Scoped)0]
    auto s  = string_view(__PRETTY_FUNCTION__ +
                           SZC("constexpr auto enchantum::details::enum_in_array_name_size() [with auto Enum = "),
                         SZC(__PRETTY_FUNCTION__) -
                           SZC("constexpr auto enchantum::details::enum_in_array_name_size() [with auto Enum = ]"));
    using E = decltype(Enum);
    // if scoped
    if constexpr (!std::is_convertible_v<E, std::underlying_type_t<E>>) {
      return s.front() == '(' ? s.size() - SZC("()0") : s.rfind(':') - 1;
    }
    else {
      if (s.front() == '(') {
        s.remove_prefix(SZC("("));
        s.remove_suffix(SZC(")0"));
      }
      if (const auto pos = s.rfind(':'); pos != s.npos)
        return pos - 1;
      return std::size_t{0};
    }
  }

  template<typename Enum>
  constexpr auto length_of_enum_in_template_array_if_casting() noexcept
  {
    if constexpr (is_scoped_enum<Enum>) {
      return details::enum_in_array_name_size<Enum{}>();
    }
    else {
      constexpr auto  s      = details::enum_in_array_name_size<Enum{}>();
      constexpr auto& tyname = raw_type_name<Enum>;
      if (constexpr auto pos = tyname.rfind("::"); pos != tyname.npos) {
        return s + tyname.substr(pos).size();
      }
      else {
        return s + tyname.size();
      }
    }
  }

  template<auto... Vs>
  constexpr auto var_name() noexcept
  {
    return __PRETTY_FUNCTION__ + SZC("constexpr auto enchantum::details::var_name() [with auto ...Vs = {");
  }


  template<bool IsBitFlag, typename IntType>
  constexpr void parse_string(
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
    for (std::size_t index = 0; index < array_size; ++index) {
      if (*str == '(') {
        str = std::char_traits<char>::find(str + least_length_when_casting, UINT8_MAX, ',') + SZC(", ");
      }
      else {
        str += least_length_when_value;
        // although gcc implementation of std::char_traits::find is using a for loop internally
        // copying the code of the function makes it way slower to compile, this was surprising.
        const auto commapos = static_cast<std::size_t>(std::char_traits<char>::find(str, UINT8_MAX, ',') - str);
        if constexpr (IsBitFlag)
          values[valid_count] = index == 0 ? IntType{} : static_cast<IntType>(IntType{1} << (index - 1));
        else
          values[valid_count] = static_cast<IntType>(min + static_cast<IntType>(index));
        string_lengths[valid_count++] = static_cast<std::uint8_t>(commapos);
        for (std::size_t i = 0; i < commapos; ++i)
          strings[total_string_length++] = str[i];
        total_string_length += null_terminated;
        str += commapos + SZC(", ");
      }
    }
  }


  template<typename E, bool NullTerminated, auto Min, std::size_t... Is>
  constexpr auto reflect(std::index_sequence<Is...>) noexcept
  {

    constexpr auto elements_local = []() {
      constexpr auto ArraySize = sizeof...(Is) + is_bitflag<E>;
      using Under              = std::underlying_type_t<E>;
      using Underlying = std::make_unsigned_t<std::conditional_t<std::is_same_v<bool, Under>, unsigned char, Under>>;


      constexpr auto str = [](const auto dependant) {
        // __builtin_bit_cast used to silence errors when casting out of unscoped enums range
        // dummy 0
        if constexpr (sizeof(dependant) && is_bitflag<E>) // sizeof... to make contest dependant
          return details::var_name<E{}, __builtin_bit_cast(E, static_cast<Under>(Underlying{1} << Is))..., 0>();
        else
          return details::var_name<__builtin_bit_cast(E, static_cast<Under>(static_cast<decltype(Min)>(Is) + Min))..., 0>();
      }(0);

      constexpr auto enum_in_array_len = details::enum_in_array_name_size<E{}>();
      constexpr auto length_of_enum_in_template_array_casting = details::length_of_enum_in_template_array_if_casting<E>();

      ReflectStringReturnValue<std::underlying_type_t<E>, ArraySize> ret;
      details::parse_string<is_bitflag<E>>(
        /*str = */ str,
        /*least_length_when_casting=*/SZC("(") + length_of_enum_in_template_array_casting + SZC(")0"),
        /*least_length_when_value=*/details::prefix_length_or_zero<E> +
          (enum_in_array_len != 0 ? enum_in_array_len + SZC("::") : 0),
        /*min = */ static_cast<std::underlying_type_t<E>>(Min),
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
                Strings     ret{};
                const auto  size     = ret.size();
                auto* const ret_data = ret.data();
                for (std::size_t i = 0; i < size; ++i)
                  ret_data[i] = strings[i];
                return ret;
              }(elements_local.strings)};

    return data;
  }

} // namespace details

} // namespace enchantum

#undef SZC