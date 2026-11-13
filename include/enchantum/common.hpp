#pragma once
#ifdef __cpp_concepts
#include <concepts>
#endif
#include <limits>
#include <string_view>
#include <type_traits>

#ifndef ENCHANTUM_ASSERT
  #include <cassert>
// clang-format off
  #define ENCHANTUM_ASSERT(cond, msg, ...) assert(cond && msg)
// clang-format on
#endif

#ifndef ENCHANTUM_THROW
  // additional info such as local variables are here
  #define ENCHANTUM_THROW(exception, ...) throw exception
#endif

#ifndef ENCHANTUM_MAX_RANGE
  #define ENCHANTUM_MAX_RANGE 256
#endif
#ifndef ENCHANTUM_MIN_RANGE
  #define ENCHANTUM_MIN_RANGE (-ENCHANTUM_MAX_RANGE)
#endif

namespace enchantum {

template<typename T, bool = std::is_enum_v<T>>
inline constexpr bool is_scoped_enum = false;

template<typename E>
inline constexpr bool is_scoped_enum<E, true> = !std::is_convertible_v<E, std::underlying_type_t<E>>;

template<typename E>
inline constexpr bool is_unscoped_enum = std::is_enum_v<E> && !is_scoped_enum<E>;

template<typename E,typename = void>
inline constexpr bool has_fixed_underlying_type = false;

template<typename E>
inline constexpr bool has_fixed_underlying_type<E,decltype(void(E{0}))> = std::is_enum_v<E>;


#ifdef __cpp_concepts

template<typename T>
concept Enum = std::is_enum_v<T>;

template<Enum E>
inline constexpr bool is_bitflag = requires(E e) {
  requires std::same_as<decltype(e & e), bool> || std::same_as<decltype(e & e), E>;
  { ~e } -> std::same_as<E>;
  { e | e } -> std::same_as<E>;
  { e &= e } -> std::same_as<E&>;
  { e |= e } -> std::same_as<E&>;
};


template<typename T>
concept SignedEnum = Enum<T> && std::signed_integral<std::underlying_type_t<T>>;

template<typename T>
concept UnsignedEnum = Enum<T> && !SignedEnum<T>;

template<typename T>
concept ScopedEnum = Enum<T> && (!std::is_convertible_v<T, std::underlying_type_t<T>>);

template<typename T>
concept UnscopedEnum = Enum<T> && !ScopedEnum<T>;

template<typename E, typename Underlying>
concept EnumOfUnderlying = Enum<E> && std::same_as<std::underlying_type_t<E>, Underlying>;

template<typename T>
concept BitFlagEnum = Enum<T> && is_bitflag<T>;

template<typename T>
concept EnumFixedUnderlying = Enum<T> && requires { T{0}; };

#else


template<typename E, typename = void>
inline constexpr bool is_bitflag = false;

// clang-format off
template<typename E>
inline constexpr bool is_bitflag<E, 
    std::void_t<
    decltype(E{} & E{}),
    decltype(~E{}), 
    decltype(E{} | E{}), 
    decltype(std::declval<E&>() &= E{}), 
    decltype(std::declval<E&>() |= E{})
    >> =  std::is_enum_v<E>
    &&    (std::is_same_v<decltype(E{} & E{}),bool>  || std::is_same_v<decltype(E{} & E{}), E>) 
    &&    std::is_same_v<decltype(~E{}), E> 
    &&    std::is_same_v<decltype(E{} | E{}), E>
    &&    std::is_same_v<decltype(std::declval<E&>() &= E{}), E&>
    &&    std::is_same_v<decltype(std::declval<E&>() |= E{}), E&>
    ;
// clang-format on
#endif


namespace details {
  template<typename T, typename U>
  constexpr auto Max(T a, U b)
  {
    return a < b ? b : a;
  }
  template<typename T, typename U>
  constexpr auto Min(T a, U b)
  {
    return a > b ? b : a;
  }
#if !defined(__NVCOMPILER) && defined(__clang__) && __clang_major__ >= 20
  template<typename E, auto V, typename = void>
  inline constexpr bool is_valid_cast = false;

  template<typename E, auto V>
  inline constexpr bool is_valid_cast<E, V, std::void_t<std::integral_constant<E, static_cast<E>(V)>>> = true;

  template<typename E, std::underlying_type_t<E> range, decltype(range) old_range>
  constexpr auto valid_cast_range_recurse() noexcept
  {
    // this tests whether `static_cast`ing range is valid
    // because C style enums stupidly is like a bit field
    // `enum E { a,b,c,d = 3};` is like a bitfield `struct E { int val : 2;}`
    // which means giving E.val a larger than 2 bit value is UB so is it for enums
    // and gcc and msvc ignore this (for good)
    // while clang makes it a subsituation failure which we can check for
    // using std::inegral_constant makes sure this is a constant expression situation
    // for SFINAE to occur
    if constexpr (is_valid_cast<E, range>)
      return valid_cast_range_recurse<E, range * 2, range>();
    else
      return old_range > 0 ? old_range * 2 - 1 : old_range;
  }
  template<typename E, int max_range>
  constexpr auto valid_cast_range() noexcept
  {
    using T = std::underlying_type_t<E>;
    using L = std::numeric_limits<T>;

    if constexpr (max_range == 0)
      return T{0};
    else if constexpr (max_range > 0 && is_valid_cast<E, (L::max)()>)
      return L::max();
    else if constexpr (max_range < 0 && is_valid_cast<E, (L::min)()>)
      return L::min();
    else
      return details::valid_cast_range_recurse<E, max_range, 0>();
  }

#endif

  template<typename E>
  constexpr auto enum_range_of(const int max_range)
  {
    using T = std::underlying_type_t<E>;
    using L = std::numeric_limits<T>;
#if !defined(__NVCOMPILER) && defined(__clang__) && __clang_major__ >= 20
    constexpr auto Max = has_fixed_underlying_type<E> ? (L::max)() : details::valid_cast_range<E, 1>();
    constexpr auto Min = has_fixed_underlying_type<E> ? (L::min)() : details::valid_cast_range<E, std::is_signed_v<T> ? -1 : 0>();
#else
    constexpr auto Max = (L::max)();
    constexpr auto Min = (L::min)();
#endif

    if constexpr (std::is_signed_v<T>) {
      return max_range > 0 ? details::Min(ENCHANTUM_MAX_RANGE, Max) : details::Max(ENCHANTUM_MIN_RANGE, Min);
    }
    else {
      if constexpr (std::is_same_v<bool, T>)
        return max_range > 0;
      else
        return max_range > 0 ? details::Min(static_cast<unsigned int>(ENCHANTUM_MAX_RANGE), Max) : 0;
    }
  }
} // namespace details


template<typename E>
struct enum_traits {
private:
  using T = std::underlying_type_t<E>;
public:
  static constexpr auto          max = details::enum_range_of<E>(1);
  static constexpr decltype(max) min = details::enum_range_of<E>(-1);
};

} // namespace enchantum

#ifdef __cpp_concepts
  #define ENCHANTUM_DETAILS_ENUM_CONCEPT(Name) Enum Name
  #define ENCHANTUM_DETAILS_CONCEPT_OR_TYPENAME(...) __VA_ARGS__
  #define ENCHANTUM_DETAILS_ENUM_BITFLAG_CONCEPT(Name) BitFlagEnum Name
#else
  #define ENCHANTUM_DETAILS_CONCEPT_OR_TYPENAME(...) typename
  #define ENCHANTUM_DETAILS_ENUM_CONCEPT(Name) typename Name, std::enable_if_t<std::is_enum_v<Name>, int> = 0
  #define ENCHANTUM_DETAILS_ENUM_BITFLAG_CONCEPT(Name) typename Name, std::enable_if_t<is_bitflag<Name>, int> = 0
#endif