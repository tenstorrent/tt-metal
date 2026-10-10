#pragma once
#include "details/string_view.hpp"
#include <array>

namespace enchantum {

namespace details {
#define SZC(x) (sizeof(x) - 1)
  constexpr string_view extract_name_from_type_name(const string_view type_name) noexcept
  {
    if (const auto n = type_name.rfind(':'); n != type_name.npos)
      return type_name.substr(n + 1);
    else
      return type_name;
  }

  template<typename T>
  constexpr auto raw_type_name_func() noexcept
  {
#if defined(__NVCOMPILER)
    constexpr std::size_t prefix = 0;
    constexpr auto s = string_view(__PRETTY_FUNCTION__ + SZC("constexpr auto enchantum::details::raw_type_name_func() noexcept [with T = "),
            SZC(__PRETTY_FUNCTION__) - SZC("constexpr auto enchantum::details::raw_type_name_func() noexcept [with T = ]"));
#elif defined(__clang__)
    constexpr std::size_t prefix = 0;
    constexpr auto s = string_view(__PRETTY_FUNCTION__ + SZC("auto enchantum::details::raw_type_name_func() [_ = "),
                                   SZC(__PRETTY_FUNCTION__) - SZC("auto enchantum::details::raw_type_name_func() [_ = ]"));
#elif defined(_MSC_VER)
    constexpr auto s = string_view(__FUNCSIG__ + SZC("auto __cdecl enchantum::details::raw_type_name_func<"),
                                   SZC(__FUNCSIG__) - SZC("auto __cdecl enchantum::details::raw_type_name_func<") -
                                     SZC(">(void) noexcept"));

    // clang-format off
    constexpr auto prefix = std::is_enum_v<T> ? SZC("enum ") : 
        std::is_class_v<T> ?  SZC("struct ") - (s[0] == 'c') :
        0;
// clang-format on
#elif defined(__GNUG__)
    constexpr std::size_t prefix = 0;
    constexpr auto        s      = string_view(__PRETTY_FUNCTION__ +
                                     SZC("constexpr auto enchantum::details::raw_type_name_func() [with _ = "),
                                   SZC(__PRETTY_FUNCTION__) -
                                     SZC("constexpr auto enchantum::details::raw_type_name_func() [with _ = ]"));
#endif
    std::array<char, 1 + s.size() - prefix> ret{};
    auto* const                             ret_data = ret.data();
    const auto* const                       s_data   = s.data();

    for (std::size_t i = 0; i < ret.size() - 1; ++i)
      ret_data[i] = s_data[i + prefix];
    return ret;
  }

  template<typename T>
  inline constexpr auto raw_type_name_func_var = raw_type_name_func<T>();


  template<typename T>
  constexpr auto type_name_func() noexcept
  {
    static_assert(!std::is_function_v<std::remove_pointer_t<T>> && !std::is_member_function_pointer_v<T>,
                  "enchantum::type_name<T> does not work well with function pointers or functions or member function\n"
                  "pointers");

    constexpr auto& array = raw_type_name_func_var<T>;
    static_assert(array[array.size() - 2] != '>', "enchantum::type_name<T> does not work well with a templated type");

    constexpr auto  s     = details::extract_name_from_type_name(string_view(array.data(), array.size() - 1));
    std::array<char, s.size() + 1> ret{};
    for (std::size_t i = 0; i < s.size(); ++i)
      ret[i] = s[i];
    return ret;
  }

  template<typename T>
  inline constexpr auto type_name_func_var = type_name_func<T>();

#undef SZC

} // namespace details

template<typename T>
inline constexpr auto type_name = string_view(details::type_name_func_var<T>.data(),
                                              details::type_name_func_var<T>.size() - 1);

template<typename T>
inline constexpr auto raw_type_name = string_view(details::raw_type_name_func_var<T>.data(),
                                                  details::raw_type_name_func_var<T>.size() - 1);


} // namespace enchantum