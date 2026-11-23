// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Values whose kind, compile time or run time, is part of their type. Arch-independent, no dependencies but the
// standard library. From Strahinja Stanisic's proposal (sstanisic/llk2-ctrt-tensor, wrapper/wrapper.h), cut down to
// what the layers use.
//
//   ct<V>          a value known at compile time. Empty: the value is in the type (ct<5>, ct<DataFormat::Float32>)
//   rt<T>          a value known at run time, held as data. rt x = n; deduces rt<type of n>
//   ctrt<X, T>     a constraint: X is a ct, an rt or a plain type that can be a T. For a parameter of either kind:
//                      template <class N, ctrt<N, std::uint32_t> = 0>
//                      void run(const N& tiles);
//   wrapper::value(x)                     the value of a ct, an rt or a plain x
//   wrapper::is_ct_v<X>, is_rt_v<X>       the kind, for `if constexpr`
//
// Why. Code that needs a constant (a loop bound the compiler must see inside the function that loops, an instruction
// immediate, a static_assert) reads wrapper::value(n) of a ct and gets the constant because the type has it, not
// because the optimizer folded an argument after inlining: the same at -O0. One body serves both kinds. The user is
// llk-lib's execute, whose loop counts are a ct, or an rt for a tensor size known only at run time
// (resolver/work_counts.h).
//
// A ct never becomes a number silently: it has no conversion, to its value or to an rt; wrapper::value(c) is the way
// out. An rt converts from and to its value implicitly: nothing is lost.
//
// Builds with the kernels' flags (-std=c++17 -ftt-nttp -ftt-constinit -ftt-consteval) and with plain -std=c++17 on
// the host, where a ct's value cannot be of class type (that is -ftt-nttp's, or C++20's). The rules are
// static_asserts in llk2/tests/host/wrapper_test.cpp.
//
// Not here, of the proposal: arithmetic, comparison and logical operators on ct / rt, min / max, value_cast, and
// require() (a static_assert or an LLK_ASSERT by the condition's kind). No layer or kernel calls them: every size a
// kernel writes is a template argument or a plain run-time value, and the layers only pass counts down and read them.

#pragma once

#include <type_traits>
#include <utility>

namespace llk2
{

/// A value known at compile time. Empty: V is in the type. No conversion: wrapper::value(c) reads it.
template <auto V>
struct ct
{
    using type                  = decltype(V);
    static constexpr type value = V;
};

/// A value known at run time. Converts from and to a T implicitly.
template <class T>
struct rt
{
    using type = T;
    T value;

    constexpr __attribute__((always_inline)) rt(T v) : value(v)
    {
    }

    constexpr __attribute__((always_inline)) operator T() const
    {
        return value;
    }
};

namespace wrapper
{

namespace detail
{
template <class X>
using bare = std::remove_cv_t<std::remove_reference_t<X>>;

template <class X>
inline constexpr bool is_ct = false;
template <auto V>
inline constexpr bool is_ct<ct<V>> = true;

template <class X>
inline constexpr bool is_rt = false;
template <class T>
inline constexpr bool is_rt<rt<T>> = true;

/// Whether the constant V can be a T without losing its value: T {V} is no narrowing (for a constant, list
/// initialization checks the value, so ct<-1> is no std::uint32_t and ct<256> no std::uint8_t).
template <class T, auto V, class = void>
inline constexpr bool constant_fits = false;
template <class T, auto V>
inline constexpr bool constant_fits<T, V, std::void_t<decltype(T {V})>> = std::is_convertible_v<decltype(V), T>;
} // namespace detail

/// True for a ct<V> (also const or a reference to one).
template <class X>
inline constexpr bool is_ct_v = detail::is_ct<detail::bare<X>>;

/// True for an rt<T> (also const or a reference to one).
template <class X>
inline constexpr bool is_rt_v = detail::is_rt<detail::bare<X>>;

/// The value of a ct, an rt or a plain x. The way out of a ct.
template <class X>
constexpr __attribute__((always_inline)) auto value(const X& x)
{
    if constexpr (is_ct_v<X>)
    {
        return X::value;
    }
    else if constexpr (is_rt_v<X>)
    {
        return x.value;
    }
    else
    {
        return x;
    }
}

/// Whether an X can be a T in the kind it has: a ct whose value fits T, or an rt or a plain type that converts to T
/// implicitly (as a function argument does: its value is not checked). The condition of ctrt<X, T>, as a bool. X is a
/// plain type: no const, no reference.
template <class X, class T>
inline constexpr bool is_ctrt_v = std::is_convertible_v<decltype(value(std::declval<const X&>())), T>;
template <auto V, class T>
inline constexpr bool is_ctrt_v<ct<V>, T> = detail::constant_fits<T, V>;

} // namespace wrapper

/// Constrains a template to an X that is a T of either kind: a ct whose value fits T, an rt or a plain type that
/// converts to T. Written as a template parameter after the type it constrains:
///
///   template <class N, ctrt<N, std::uint32_t> = 0>
///   void run(const N& tiles);     run(ct<8> {});  run(rt {n});  run(n);    not: run(ct<-1> {})
template <class X, class T>
using ctrt = std::enable_if_t<wrapper::is_ctrt_v<wrapper::detail::bare<X>, T>, int>;

} // namespace llk2
