#pragma once
#include "common.hpp"
#include "details/string.hpp"
#include "details/string_view.hpp"
#include "enchantum.hpp"
#include "generators.hpp"

namespace enchantum {

template<typename E>
inline constexpr E value_ors = [] {
  static_assert(is_bitflag<E>, "");
  using T = std::underlying_type_t<E>;
  T ret{};
  for (const auto val : values_generator<E>)
    ret |= static_cast<T>(val);
  return static_cast<E>(ret);
}();


template<ENCHANTUM_DETAILS_ENUM_BITFLAG_CONCEPT(E)>
[[nodiscard]] constexpr bool contains_bitflag(const std::underlying_type_t<E> value) noexcept
{
  using T = std::underlying_type_t<E>;
  if constexpr (is_contiguous_bitflag<E>) {
    return value >= static_cast<T>(min<E>) && value <= static_cast<T>(value_ors<E>);
  }
  else {
    if (value == 0)
      return has_zero_flag<E>;
    T valid_bits = 0;

    for (auto i = std::size_t{has_zero_flag<E>}; i < count<E>; ++i) {
      const auto v = static_cast<T>(values_generator<E>[i]);
      if ((value & v) == v)
        valid_bits |= v;
    }
    return valid_bits == value;
  }
}

template<ENCHANTUM_DETAILS_ENUM_BITFLAG_CONCEPT(E)>
[[nodiscard]] constexpr bool contains_bitflag(const E value) noexcept
{
  return enchantum::contains_bitflag<E>(static_cast<std::underlying_type_t<E>>(value));
}

template<ENCHANTUM_DETAILS_ENUM_BITFLAG_CONCEPT(E),
         ENCHANTUM_DETAILS_CONCEPT_OR_TYPENAME(std::predicate<string_view, string_view>) BinaryPred>
[[nodiscard]] constexpr bool contains_bitflag(const string_view s, const char sep, const BinaryPred binary_pred) noexcept
{
  std::size_t pos = 0;
  for (std::size_t i = s.find(sep); i != s.npos; i = s.find(sep, pos)) {
    if (!enchantum::contains<E>(s.substr(pos, i - pos), binary_pred))
      return false;
    pos = i + 1;
  }
  return enchantum::contains<E>(s.substr(pos), binary_pred);
}


template<ENCHANTUM_DETAILS_ENUM_BITFLAG_CONCEPT(E)>
[[nodiscard]] constexpr bool contains_bitflag(const string_view s, const char sep = '|') noexcept
{
  std::size_t pos = 0;
  for (std::size_t i = s.find(sep); i != s.npos; i = s.find(sep, pos)) {
    if (!enchantum::contains<E>(s.substr(pos, i - pos)))
      return false;
    pos = i + 1;
  }
  return enchantum::contains<E>(s.substr(pos));
}


template<typename String = string, ENCHANTUM_DETAILS_ENUM_BITFLAG_CONCEPT(E)>
[[nodiscard]] constexpr String to_string_bitflag(const E value, const char sep = '|')
{
  using T = std::underlying_type_t<E>;
  if constexpr (has_zero_flag<E>)
    if (static_cast<T>(value) == 0)
      return String(names_generator<E>[0]);

  String name;
  T      check_value = 0;
  for (auto i = std::size_t{has_zero_flag<E>}; i < count<E>; ++i) {
    const auto v = static_cast<T>(values<E>[i]);
    if (v == (static_cast<T>(value) & v)) {
      const auto s = names_generator<E>[i];
      if (!name.empty())
        name.append(1, sep);           // append separator if not the first value
      name.append(s.data(), s.size()); // not using operator += since this may not be std::string_view always
      check_value |= v;
    }
  }
  if (check_value == static_cast<T>(value))
    return name;
  return String();
}

template<ENCHANTUM_DETAILS_ENUM_BITFLAG_CONCEPT(E),
         ENCHANTUM_DETAILS_CONCEPT_OR_TYPENAME(std::predicate<string_view, string_view>) BinaryPred>
[[nodiscard]] constexpr optional<E> cast_bitflag(const string_view s, const char sep, const BinaryPred binary_pred) noexcept
{
  using T = std::underlying_type_t<E>;
  T           check_value{};
  std::size_t pos = 0;
  for (std::size_t i = s.find(sep); i != s.npos; i = s.find(sep, pos)) {
    if (const auto v = enchantum::cast<E>(s.substr(pos, i - pos), binary_pred))
      check_value |= static_cast<T>(*v);
    else
      return optional<E>();
    pos = i + 1;
  }

  if (const auto v = enchantum::cast<E>(s.substr(pos), binary_pred))
    return optional<E>(static_cast<E>(check_value | static_cast<T>(*v)));
  return optional<E>();
}

template<ENCHANTUM_DETAILS_ENUM_BITFLAG_CONCEPT(E)>
[[nodiscard]] constexpr optional<E> cast_bitflag(const string_view s, const char sep = '|') noexcept
{
  return enchantum::cast_bitflag<E>(s, sep, [](const auto& a, const auto& b) { return a == b; });
}

template<ENCHANTUM_DETAILS_ENUM_BITFLAG_CONCEPT(E)>
[[nodiscard]] constexpr optional<E> cast_bitflag(const std::underlying_type_t<E> value) noexcept
{
  return enchantum::contains_bitflag<E>(value) ? optional<E>(static_cast<E>(value)) : optional<E>();
}

} // namespace enchantum