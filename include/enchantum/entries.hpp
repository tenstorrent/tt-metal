#pragma once

#include "details/string_view.hpp"
#if defined(__NVCOMPILER)
  #include "details/enchantum_nvcc.hpp"
#elif defined(__clang__)
  #include "details/enchantum_clang.hpp"
#elif defined(__GNUC__) || defined(__GNUG__)
  #include "details/enchantum_gcc.hpp"
#elif defined(_MSC_VER)
  #include "details/enchantum_msvc.hpp"
#else
  #error unsupported compiler please open an issue for enchantum
#endif

#include "common.hpp"
#include <type_traits>
#include <utility>

namespace enchantum {

#ifdef __cpp_lib_to_underlying
using ::std::to_underlying;
#else
template<ENCHANTUM_DETAILS_ENUM_CONCEPT(E)>
[[nodiscard]] constexpr auto to_underlying(const E e) noexcept
{
  return static_cast<std::underlying_type_t<E>>(e);
}
#endif


namespace details {

  template<typename Int>
  constexpr std::size_t get_index_sequence_max(
    const bool        is_bitflag,
    const bool        has_fixed_underlying,
    const std::size_t sizeof_enum,
    const Int         min,
    const Int         max,
    const bool        is_signed)
  {
    (void)has_fixed_underlying;
    if (!is_bitflag)
      return static_cast<std::size_t>(max - min + 1);

#if __clang_major__ >= 20
    if (!has_fixed_underlying) {
      auto        v = max;
      std::size_t r = 1;
      while (v >>= 1)
        r++;
      return r;
    }
#endif
    return (sizeof_enum * CHAR_BIT) - is_signed;
  }

  template<typename E,typename StringLengthType,std::size_t Size>
  struct FinalReflectionResult
  {
    std::array<E, Size> values{};
    // +1 for easier iteration on on last string
    std::array<StringLengthType, Size + 1> string_indices{};
  };

  template<typename E, bool NullTerminated, auto Min = enum_traits<E>::min, decltype(Min) Max = enum_traits<E>::max>
  inline constexpr auto reflection_data_impl = details::reflect<E, NullTerminated, Min>(
    std::make_index_sequence<details::get_index_sequence_max(is_bitflag<E>,
                                                             has_fixed_underlying_type<E>,
                                                             sizeof(E),
                                                             Min,
                                                             Max,
                                                             std::is_signed_v<std::underlying_type_t<E>>)>{});


  template<typename E,bool NullTerminated>
  constexpr auto get_reflection_data() noexcept {
    constexpr auto elements = reflection_data_impl<E, NullTerminated>.elements;
    using StringLengthType = std::conditional_t<(elements.total_string_length < UINT8_MAX), std::uint8_t, std::uint16_t>;

    FinalReflectionResult<E, StringLengthType, elements.valid_count> ret;

    std::size_t      i            = 0;
    StringLengthType string_index = 0;
    for (; i < elements.valid_count; ++i) {
      ret.values[i] = static_cast<E>(elements.values[i]);
      // "aabc"

      ret.string_indices[i] = string_index;
      string_index += static_cast<StringLengthType>(elements.string_lengths[i] + NullTerminated);
    }
    ret.string_indices[i] = string_index;
    return ret;
  }



  
  template<typename E, bool NullTerminated>
  inline constexpr auto reflection_data_string_storage = details::reflection_data_impl<E, NullTerminated>.strings;

  template<typename E, bool NullTerminated>
  inline constexpr auto reflection_data = details::get_reflection_data<E, NullTerminated>();

  template<typename E, bool NullTerminated>
  inline constexpr auto reflection_string_indices = reflection_data<E, NullTerminated>.string_indices;
} // namespace details

#ifdef __cpp_concepts
template<Enum E, typename Pair = std::pair<E, string_view>, bool NullTerminated = true>
#else
template<typename E, typename Pair = std::pair<E, string_view>, bool NullTerminated = true, std::enable_if_t<std::is_enum_v<E>, int> = 0>
#endif
inline constexpr auto entries = []() {

#if defined(__NVCOMPILER)
  // nvc++ had issues with that and did not allow it. it just did not work after testing in godbolt and I don't know why
  const auto reflected = details::reflection_data<E, NullTerminated>;
  const auto strings   = details::reflection_data_string_storage<E, NullTerminated>.data();
#else
  const auto reflected = details::reflection_data<std::remove_cv_t<E>, NullTerminated>;
  const auto strings   = details::reflection_data_string_storage<std::remove_cv_t<E>, NullTerminated>.data();
#endif
  using Pairs = std::array<Pair, sizeof(reflected.values) / sizeof(reflected.values[0])>;
  Pairs          ret{};
  constexpr auto size = ret.size();
  static_assert(size != 0,
                "enchantum failed to reflect this enum.\n"
                "Please read https://github.com/ZXShady/enchantum/blob/main/docs/limitations.md before opening an "
                "issue\n"
                "with your enum type with all its namespace/classes it is defined inside to help the creator debug the "
                "issues.");
  auto* const ret_data = ret.data();


  for (std::size_t i = 0; i < size; ++i) {
    auto& [e, s]     = ret_data[i];
    e                = reflected.values[i];
    using StringView = std::remove_cv_t<std::remove_reference_t<decltype(s)>>;
    s                = StringView(strings + reflected.string_indices[i],
                   reflected.string_indices[i + 1] - reflected.string_indices[i] - NullTerminated);
  }
  return ret;
}();

namespace details
{
  template<typename E>
  constexpr auto get_values() noexcept
  {
    constexpr auto             enums = entries<E>;
    std::array<E, enums.size()> ret{};
    const auto* const           enums_data = enums.data();
    for (std::size_t i = 0; i < ret.size(); ++i)
      ret[i] = enums_data[i].first;
    return ret;
  }

  template<typename E,typename String,bool NullTerminated>
  constexpr auto get_names() noexcept
  {
    constexpr auto                  enums = entries<E, std::pair<E, String>, NullTerminated>;
    std::array<String, enums.size()> ret{};
    const auto* const                enums_data = enums.data();
    for (std::size_t i = 0; i < ret.size(); ++i)
      ret[i] = enums_data[i].second;
    return ret;
  }

}

template<ENCHANTUM_DETAILS_ENUM_CONCEPT(E)>
inline constexpr auto values = details::get_values<E>();

#ifdef __cpp_concepts
template<Enum E, typename String = string_view, bool NullTerminated = true>
#else
template<typename E, typename String = string_view, bool NullTerminated = true, std::enable_if_t<std::is_enum_v<E>, int> = 0>
#endif
inline constexpr auto names = details::get_names<E, String, NullTerminated>();

template<ENCHANTUM_DETAILS_ENUM_CONCEPT(E)>
inline constexpr auto min = entries<E>.front().first;

template<ENCHANTUM_DETAILS_ENUM_CONCEPT(E)>
inline constexpr auto max = entries<E>.back().first;

template<ENCHANTUM_DETAILS_ENUM_CONCEPT(E)>
inline constexpr std::size_t count = entries<E>.size();


template<typename E>
inline constexpr bool has_zero_flag = [](const auto is_bitflag) {
  if constexpr (is_bitflag.value) {
    for (const auto v : values<E>)
      if (static_cast<std::underlying_type_t<E>>(v) == 0)
        return true;
  }
  return false;
}(std::bool_constant<is_bitflag<E>>{});

template<typename E>
inline constexpr bool is_contiguous = static_cast<std::size_t>(
                                        enchantum::to_underlying(max<E>) - enchantum::to_underlying(min<E>)) +
    1 ==
  count<E>;


template<typename E>
inline constexpr bool is_contiguous_bitflag = [](const auto is_bitflag) {
  if constexpr (is_bitflag.value) {
    constexpr auto& enums = entries<E>;
    using T               = std::underlying_type_t<E>;
    for (auto i = std::size_t{has_zero_flag<E>}; i < enums.size() - 1; ++i)
      if (T(enums[i].first) << 1 != T(enums[i + 1].first))
        return false;
    return true;
  }
  else {
    return false;
  }
}(std::bool_constant<is_bitflag<E>>{});

#ifdef __cpp_concepts
template<typename E>
concept ContiguousEnum = Enum<E> && is_contiguous<E>;
template<typename E>
concept ContiguousBitFlagEnum = BitFlagEnum<E> && is_contiguous_bitflag<E>;
#endif

} // namespace enchantum