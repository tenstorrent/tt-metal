#pragma once

#include "common.hpp"
#include "details/optional.hpp"
#include "details/string_view.hpp"
#include "entries.hpp"
#include "generators.hpp"
#include <type_traits>
#include <utility>

namespace enchantum {

namespace details {
  constexpr std::pair<std::size_t, std::size_t> minmax_string_size(const string_view* begin, const string_view* const end)
  {
    using T     = std::size_t;
    auto minmax = std::pair<T, T>(std::numeric_limits<T>::max(), 0);

    for (; begin != end; ++begin) {
      const auto size = begin->size();
      minmax.first    = minmax.first < size ? minmax.first : size;
      minmax.second   = minmax.second > size ? minmax.second : size;
    }
    return minmax;
  }

} // namespace details




template<ENCHANTUM_DETAILS_ENUM_CONCEPT(E)>
[[nodiscard]] constexpr bool contains(const std::underlying_type_t<E> value) noexcept
{
  using T = std::underlying_type_t<E>;

  if (value < T(min<E>) || value > T(max<E>))
    return false;

  if constexpr (is_contiguous_bitflag<E>) {
    if constexpr (has_zero_flag<E>)
      if (value == 0)
        return true;
    const auto u = static_cast<std::make_unsigned_t<T>>(value);

    // std::has_single_bit
    return u != 0 && (u & (u - 1)) == 0;
  }
  else if constexpr (is_contiguous<E>) {
    return true;
  }
  else {
    for (const auto v : values_generator<E>)
      if (static_cast<T>(v) == value)
        return true;
    return false;
  }
}

template<ENCHANTUM_DETAILS_ENUM_CONCEPT(E)>
[[nodiscard]] constexpr bool contains(const E value) noexcept
{
  return enchantum::contains<E>(static_cast<std::underlying_type_t<E>>(value));
}

template<ENCHANTUM_DETAILS_ENUM_CONCEPT(E)>
[[nodiscard]] constexpr bool contains(const string_view name) noexcept
{
  constexpr auto minmax = details::minmax_string_size(names<E>.data(), names<E>.data() + names<E>.size());
  if (const auto size = name.size(); size < minmax.first || size > minmax.second)
    return false;

  for (const auto s : names_generator<E>)
    if (s == name)
      return true;
  return false;
}


template<ENCHANTUM_DETAILS_ENUM_CONCEPT(E), 
    ENCHANTUM_DETAILS_CONCEPT_OR_TYPENAME(std::predicate<string_view, string_view>) BinaryPredicate>
[[nodiscard]] constexpr bool contains(const string_view name, const BinaryPredicate binary_predicate) noexcept
{
  for (const auto s : names_generator<E>)
    if (binary_predicate(name, s))
      return true;
  return false;
}


namespace details {
  template<ENCHANTUM_DETAILS_ENUM_CONCEPT(E)>
  struct index_to_enum_functor {
    [[nodiscard]] constexpr optional<E> operator()(const std::size_t index) const noexcept
    {
      if (index < count<E>)
        return optional<E>(values_generator<E>[index]);
      return optional<E>();
    }
  };

  struct enum_to_index_functor {
    template<ENCHANTUM_DETAILS_ENUM_CONCEPT(E)>
    [[nodiscard]] constexpr optional<std::size_t> operator()(const E e) const noexcept
    {
      using T = std::underlying_type_t<E>;

      if constexpr (is_contiguous<E>) {
        if (enchantum::contains(e)) {
          return optional<std::size_t>(std::size_t(T(e) - T(min<E>)));
        }
      }
      else if constexpr (is_contiguous_bitflag<E>) {
        if (enchantum::contains(e)) {
          constexpr bool has_zero = has_zero_flag<E>;
          if constexpr (has_zero)
            if (static_cast<T>(e) == 0)
              return optional<std::size_t>(0); // assumes 0 is the index of value `0`

          using U = std::make_unsigned_t<T>;
          return has_zero + details::countr_zero(static_cast<U>(e)) - details::countr_zero(static_cast<U>(values_generator<E>[has_zero]));
        }
      }
      else {
        for (std::size_t i = 0; i < count<E>; ++i) {
          if (values_generator<E>[i] == e)
            return optional<std::size_t>(i);
        }
      }
      return optional<std::size_t>();
    }
  };


  template<ENCHANTUM_DETAILS_ENUM_CONCEPT(E)>
  struct cast_functor {
    [[nodiscard]] constexpr optional<E> operator()(const std::underlying_type_t<E> value) const noexcept
    {
      if (!enchantum::contains<E>(value))
        return optional<E>();
      return optional<E>(static_cast<E>(value));
    }

    [[nodiscard]] constexpr optional<E> operator()(const string_view name) const noexcept
    {
      constexpr auto minmax = details::minmax_string_size(names<E>.data(), names<E>.data() + names<E>.size());
      if (const auto size = name.size(); size < minmax.first || size > minmax.second)
        return optional<E>(); // nullopt

      for (std::size_t i = 0; i < count<E>; ++i) {
        if (names_generator<E>[i] == name) {
          return optional<E>(values_generator<E>[i]);
        }
      }
      return optional<E>(); // nullopt
    }

    template<ENCHANTUM_DETAILS_CONCEPT_OR_TYPENAME(std::predicate<string_view, string_view>) BinaryPred>
    [[nodiscard]] constexpr optional<E> operator()(const string_view name, const BinaryPred binary_predicate) const noexcept
    {
      for (std::size_t i = 0; i < count<E>; ++i) {
        if (binary_predicate(name, names_generator<E>[i])) {
          return optional<E>(values_generator<E>[i]);
        }
      }
      return optional<E>();
    }
  };

} // namespace details

template<ENCHANTUM_DETAILS_ENUM_CONCEPT(E)>
inline constexpr details::index_to_enum_functor<E> index_to_enum{};

inline constexpr details::enum_to_index_functor enum_to_index{};

template<ENCHANTUM_DETAILS_ENUM_CONCEPT(E)>
inline constexpr details::cast_functor<E> cast{};


namespace details {
  struct to_string_functor {
    template<ENCHANTUM_DETAILS_ENUM_CONCEPT(E)>
    [[nodiscard]] constexpr string_view operator()(const E value) const noexcept
    {
      if (const auto i = enchantum::enum_to_index(value))
        return names_generator<E>[*i];
      return string_view();
    }
  };

} // namespace details
inline constexpr details::to_string_functor to_string{};


} // namespace enchantum