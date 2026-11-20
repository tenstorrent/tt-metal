// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <type_traits>
#include <utility>

#include "stdx/type_traits.hpp"

// todo(sstanisic): this is 99% it, it's just missing some final touches (and unit tests)

namespace llk
{

template <decltype(auto) V>
struct ct;

template <class T>
struct rt;

/**
 * @brief A T that is known at compile time or at run time: the type of a function parameter that takes either
 *
 * Inside the function the parameter behaves as the ct or rt it came from: the operators, wrapper::value, conversion to its
 * value. It is a copy, complete on its own: it can be copied again, stored and passed on.
 *
 * @code
 * template <class A, class B>
 * constexpr auto func(ctrt<A, std::uint32_t> a, ctrt<B, std::uint8_t> b) { return a + b; }
 *
 * func(ct<std::uint32_t {3}> {}, rt<std::uint8_t> {4});     // matches
 * func(ct<3> {}, rt<int> {4});                              // no matching function: an int is not a std::uint32_t
 * func(value_cast<std::uint32_t>(ct<3> {}), value_cast<std::uint8_t>(rt<int> {4}));   // matches
 * @endcode
 *
 * Declared here because ct and rt inherit it; its two forms are defined further down.
 *
 * @tparam C The wrapper it is the base of, deduced for a parameter: a ct<V> or an rt<T>
 * @tparam T The type of the value
 */
template <class C, class T>
class ctrt;

namespace wrapper
{

/**
 * @brief Check if X is a ctrt
 *
 * @tparam X The type to inspect
 */
template <class X>
struct is_ctrt : std::false_type {};

/**
 * @brief Specialization of is_ctrt for a ctrt
 */
template <class C, class T>
struct is_ctrt<ctrt<C, T>> : std::true_type {};

/**
 * @brief Variable template to check if a type is a ctrt
 *
 * @tparam X The type to inspect
 */
template <class X>
inline constexpr bool is_ctrt_v = is_ctrt<stdx::remove_cvref_t<X>>::value;

/**
 * @brief Check if T is a compile-time constant wrapper (ct)
 * @details Evaluates to true if T is a ct<V>, false otherwise.
 *
 * @tparam T The type to inspect
 */
template <class T>
struct is_ct : std::false_type {};

/**
 * @brief Specialization of is_ct for valid compile-time constant wrappers (ct)
 *
 * @tparam V The compile-time value wrapped by ct
 */
template <decltype(auto) V>
struct is_ct<ct<V>> : std::true_type {};

/**
 * @brief Specialization of is_ct for compile-time constant wrappers (ct) in the union wrapper (ctrt)
 *
 * @tparam V The compile-time value wrapped by ct
 */
template <decltype(auto) V>
struct is_ct<ctrt<ct<V>, stdx::remove_cvref_t<decltype(V)>>> : std::true_type {};

/**
 * @brief Variable template to check if a type is a compile-time constant wrapper (ct)
 *
 * @tparam T The type to inspect
 */
template <class T>
inline constexpr bool is_ct_v = is_ct<stdx::remove_cvref_t<T>>::value;

/**
 * @brief Check if T is a run-time value wrapper (rt)
 * @details Evaluates to true if T is an `rt<U>`, false otherwise.
 *
 * @tparam T The type to inspect
 */
template <class T>
struct is_rt : std::false_type {};

/**
 * @brief Specialization of is_rt for run-time value wrappers (rt)
 *
 * @tparam T The type of the value wrapped by rt
 */
template <class T>
struct is_rt<rt<T>> : std::true_type {};

/**
 * @brief Specialization of is_rt for run-time value wrappers (rt) in the union wrapper (ctrt)
 *
 * @tparam T The type of the value wrapped by rt
 */
template <class T>
struct is_rt<ctrt<rt<T>, T>> : std::true_type {};

/**
 * @brief Variable template to check if a type is a run-time value wrapper (rt)
 *
 * @tparam T The type to inspect
 */
template <class T>
inline constexpr bool is_rt_v = is_rt<stdx::remove_cvref_t<T>>::value;

/**
 * @brief Variable template to check if a type is a wrapper: a compile-time constant (ct) or a run-time value (rt) or a union wrapper (ctrt)
 *
 * @tparam T The type to inspect
 */
template <class T>
inline constexpr bool is_wrapper_v = is_ct_v<T> || is_rt_v<T>;


/**
 * @brief Unwrap the value of a ct<V>, an rt<V> or a ctrt if it's wrapped, otherwise return the value itself
 *
 * @tparam T Type of the value to unwrap
 * @param object to unwrap
 * @return The unwrapped value if object is a ct<V> or rt<V>, otherwise object itself
 */
template <class T>
constexpr decltype(auto) value(const T& object)
{
    if constexpr (is_wrapper_v<T>)
    {
        return (object.value);
    }
    else
    {
        return (object);
    }
}

/**
 * @brief The constant of a ct or of a compile-time ctrt
 *
 * Read from the type, so it is a constant where the object is not: a function's reference parameter cannot be used
 * in a constant expression.
 *
 * @tparam C The type to read it from
 */
template <class C, std::enable_if_t<is_ct_v<C>, int> = 0>
inline constexpr auto constant = stdx::remove_cvref_t<C>::value;

/**
 * @brief The type of the constant of a ct or of a compile-time ctrt
 *
 * @tparam C The type to read it from
 */
template <class C, std::enable_if_t<is_ct_v<C>, int> = 0>
using type = decltype(stdx::remove_cvref_t<C>::value);

namespace detail
{

/// The type value(n) gives for an N
template <class N>
using value_t = decltype(value(std::declval<const N&>()));


/**
 * @brief Check if From converts to To implicitly
 *
 * @tparam From The source type
 * @tparam To The target type
 */
template <class From, class To>
inline constexpr bool is_implicit_conversion_v = std::is_convertible<From, To>::value;


/**
 * @brief Check if From converts to To **ONLY** explicitly
 *
 * @tparam From Source type
 * @tparam To Target type
 */
template <class From, class To>
inline constexpr bool is_explicit_conversion_v = std::is_constructible<To, From>::value && !std::is_convertible<From, To>::value;


/* Pre-requisite reading to understand the following magic:
 * - https://en.cppreference.com/cpp/language/list_initialization
 */

// T[] under one name: `T[] {...}` does not parse, `array_of<T> {...}` does
template <class T>
using array_of = T[];

/**
 * @brief Check if `C::value` can copy-list-initialize a T (`T t = {C::value};`)
 *
 * Evaluates to true if the constant converts to T implicitly and without narrowing. Narrowing is judged on the value.
 *
 * Examples, with `enum class E : std::uint8_t` and `struct X { explicit X(int); }`:
 *
 * | C       | T             | Result | Reason                                  |
 * |---------|---------------|--------|-----------------------------------------|
 * | ct<1>   | std::uint32_t | true   |                                         |
 * | ct<-1>  | std::uint32_t | false  | narrowing                               |
 * | ct<255> | std::uint8_t  | true   |                                         |
 * | ct<256> | std::uint8_t  | false  | narrowing                               |
 * | ct<1>   | float         | true   | 1 is exact in a float                   |
 * | ct<1>   | E             | false  | no implicit conversion to an enum class |
 * | ct<1>   | X             | false  | the constructor is explicit             |
 *
 * @tparam C Constant wrapper
 * @tparam T Target type
 */
template <class C, class T, class = void>
struct is_copy_list_init : std::false_type {};

/**
 * @brief Specialization for when `C::value` can copy-list-initialize a T
 */
template <class C, class T>
struct is_copy_list_init<C, T, std::void_t<decltype(array_of<T>{{C::value}})>> : std::true_type {};

/**
 * @brief Check if `C::value` can direct-list-initialize a T (`T {C::value}`)
 *
 * Evaluates to true if the constant converts to T, implicitly or explicitly, and without narrowing. Narrowing is
 * judged on the value.
 *
 * Examples, with `enum class E : std::uint8_t` and `struct X { explicit X(int); }`:
 *
 * | C       | T             | Result | Reason                                           |
 * |---------|---------------|--------|--------------------------------------------------|
 * | ct<1>   | std::uint32_t | true   |                                                  |
 * | ct<-1>  | std::uint32_t | false  | narrowing                                        |
 * | ct<1>   | E             | true   | an enum class takes its underlying type's values |
 * | ct<256> | E             | false  | narrowing                                        |
 * | ct<1>   | X             | true   | explicit constructors are considered             |
 * | ct<1>   | const char*   | false  | no conversion                                    |
 *
 * @tparam C Constant wrapper
 * @tparam T Target type
 */
template <class C, class T, class = void>
struct is_direct_list_init : std::false_type {};

/**
 * @brief Specialization for when `C::value` can direct-list-initialize a T
 */
template <class C, class T>
struct is_direct_list_init<C, T, std::void_t<decltype(T {C::value})>> : std::true_type {};

/**
 * @brief Check if the constant From, converted to To's type, is To: ct<From> can become ct<To>
 *
 * Evaluates to true if From direct-list-initializes To's type (is_direct_list_init) and the result equals To. From and
 * To are values, not types. Whether the conversion is implicit or explicit is left to the caller.
 *
 * Examples, with `enum class E : std::uint8_t`:
 *
 * | From  | To          | Result | Reason                              |
 * |-------|-------------|--------|-------------------------------------|
 * | 1     | 1u          | true   |                                     |
 * | 1u    | 1           | true   |                                     |
 * | 1     | 2u          | false  | a different value                   |
 * | -1    | 4294967295u | false  | narrowing                           |
 * | 1     | E {1}       | true   | explicit conversions are considered |
 * | 256   | E {0}       | false  | narrowing                           |
 * | E {1} | 1           | false  | no conversion from an enum class    |
 * | 1     | nullptr     | false  | no conversion                       |
 *
 * @tparam From Source constant
 * @tparam To Target constant
 */
template <decltype(auto) From, decltype(auto) To>
constexpr bool is_value_preserving()
{
    using T = stdx::remove_cvref_t<decltype(To)>;
    if constexpr (is_direct_list_init<ct<From>, T>::value)
    {
        return T {From} == To;
    }
    else
    {
        return false;
    }
}

/**
 * @brief Check if `static_cast<To>(from)` is valid for a `from` of type From
 *
 * Evaluates to true for every implicit and explicit conversion, and for the conversions only a cast can do. Judged on
 * the types, not the value: nothing is checked for narrowing.
 *
 * Examples, with `enum class E : std::uint8_t` and `struct Derived : Base`:
 *
 * | From       | To            | Result | Reason                                |
 * |------------|---------------|--------|---------------------------------------|
 * | int        | std::uint32_t | true   |                                       |
 * | int        | std::uint8_t  | true   | narrowing is allowed: the value wraps |
 * | int        | E             | true   | integer to enum class                 |
 * | E          | int           | true   | enum class to integer                 |
 * | Base*      | Derived*      | true   | downcast                              |
 * | void*      | int*          | true   |                                       |
 * | int        | const char*   | false  | needs reinterpret_cast                |
 * | int*       | float*        | false  | needs reinterpret_cast                |
 * | const int* | int*          | false  | needs const_cast                      |
 *
 * @tparam From Source type
 * @tparam To Target type
 */
template <class From, class To, class = void>
struct is_static_castable : std::false_type {};

/**
 * @brief Specialization for when `static_cast<To>(from)` is valid
 */
template <class From, class To>
struct is_static_castable<From, To, std::void_t<decltype(static_cast<To>(std::declval<From>()))>> : std::true_type {};

/**
 * @brief Variable template to check if `static_cast<To>(from)` is valid for a `from` of type From
 *
 * @tparam From Source type
 * @tparam To Target type
 */
template <class From, class To>
inline constexpr bool is_static_castable_v = is_static_castable<From, To>::value;

/**
 * @brief Check if A and B are the operands of a wrapper operator
 *
 * Evaluates to true if at least one is a wrapper. The other side can be anything: an operator its value does not have
 * fails to compile inside the operator.
 *
 * @tparam A Left operand type
 * @tparam B Right operand type
 */
template <class A, class B>
inline constexpr bool operands_v = is_wrapper_v<A> || is_wrapper_v<B>;

/**
 * @brief Check if A and B are the operands of a wrapper shift
 *
 * As operands_v, and unless both are a ct the left value is not a class, so that `stream << wrapper` keeps the
 * stream's own operator.
 *
 * @tparam A Left operand type
 * @tparam B Right operand type
 */
template <class A, class B>
inline constexpr bool shift_operands_v =
    operands_v<A, B> && ((is_ct_v<A> && is_ct_v<B>) || !std::is_class<stdx::remove_cvref_t<value_t<A>>>::value);

} // namespace detail


/**
 * @brief Check if N can be a compile-time T: only a ct whose value initializes a T
 *
 * | N       | T             | Result | Reason                    |
 * |---------|---------------|--------|---------------------------|
 * | ct<1>   | std::uint32_t | true   |                           |
 * | ct<-1>  | std::uint32_t | false  | narrowing                 |
 * | rt<int> | std::uint32_t | false  | not known at compile time |
 * | int     | std::uint32_t | false  | not known at compile time |
 *
 * @tparam N The type to inspect
 * @tparam T The type the value must fit
 */
template <class N, class T>
inline constexpr bool is_comptime_v = is_ct_v<N> && detail::is_copy_list_init<stdx::remove_cvref_t<N>, T>::value;


/**
 * @brief Check if N can be a run-time T: a ct, an rt or a plain type whose value converts implicitly to T
 *
 * | N           | T             | Result | Reason                          |
 * |-------------|---------------|--------|---------------------------------|
 * | ct<1>       | std::uint32_t | true   |                                 |
 * | ct<-1>      | std::uint32_t | true   | int converts: not value-checked |
 * | rt<int>     | std::uint32_t | true   |                                 |
 * | int         | std::uint32_t | true   |                                 |
 * | const char* | std::uint32_t | false  | no conversion                   |
 *
 * @tparam N The type to inspect
 * @tparam T The type the value must convert to
 */
template <class N, class T>
inline constexpr bool is_runtime_v = detail::is_implicit_conversion_v<detail::value_t<N>, T>;


/**
 * @brief Check if N can be a T in either form: a ct, an rt or a plain type
 *
 * | N           | T             | Result | Reason                                  |
 * |-------------|---------------|--------|-----------------------------------------|
 * | ct<1>       | std::uint32_t | true   | is_comptime_v                           |
 * | ct<-1>      | std::uint32_t | false  | narrowing                               |
 * | rt<int>     | std::uint32_t | true   | is_runtime_v                            |
 * | int         | std::uint32_t | true   | is_runtime_v: also for the value -1     |
 * | const char* | std::uint32_t | false  | no conversion                           |
 * | ct<1>, int  | an enum class | false  | no implicit conversion to an enum class |
 *
 * @tparam N The type to inspect: a ct, an rt or a plain type
 * @tparam T The type the value must fit
 */
template <class N, class T>
inline constexpr bool is_anytime_v = is_ct_v<N> ? is_comptime_v<N, T> : is_runtime_v<N, T>;


} // namespace wrapper

// ---- the wrappers ----------------------------------------------------------------------------------------------

/**
 * @brief A compile-time ctrt: the base of a ct<V> whose value is a T
 *
 * Empty: the value is in the type.
 */
template <decltype(auto) V, class T>
class ctrt<ct<V>, T>
{
    static_assert(std::is_same<T, stdx::remove_cvref_t<decltype(V)>>::value, "ctrt<ct<V>, T>: T must be the type of V");

   public:
    using type                         = T;
    static constexpr decltype(V) value = V;

    /// Implicit conversion to the value
    constexpr operator T() const
    {
        return value;
    }
};

/**
 * @brief A value known at compile time: the value is part of the type
 *
 * ct<5> and ct<k> hold a copy of the value (its type must be structural). ct<(x)> refers to the object x itself, for
 * objects whose identity matters or whose members are private.
 *
 * Converts from another ct when the value is preserved (is_value_preserving): implicitly if the value's type converts
 * implicitly, explicitly if it converts only explicitly. Converts to its value implicitly.
 *
 * | From   | To              | Result   | Reason            |
 * |--------|-----------------|----------|-------------------|
 * | ct<1>  | ct<1u>          | implicit |                   |
 * | ct<1>  | ct<2u>          | none     | a different value |
 * | ct<-1> | ct<4294967295u> | none     | narrowing         |
 * | ct<1>  | int             | implicit | its value         |
 *
 * @tparam V The value, or a reference to the object when written as ct<(x)>
 */
template <decltype(auto) V>
struct ct : ctrt<ct<V>, stdx::remove_cvref_t<decltype(V)>>
{
    using type                         = decltype(V);
    static constexpr decltype(V) value = V;

    constexpr ct() = default;

    /// Implicit conversion from a ct with the same value
    template <decltype(auto) U, std::enable_if_t<wrapper::detail::is_value_preserving<U, V>() && wrapper::detail::is_implicit_conversion_v<decltype(U), type>, int> = 0>
    constexpr ct(ct<U>)
    {
    }

    /// Explicit conversion from a ct with the same value, when its type converts only explicitly
    template <decltype(auto) U, std::enable_if_t<wrapper::detail::is_value_preserving<U, V>() && wrapper::detail::is_explicit_conversion_v<decltype(U), type>, int> = 0>
    constexpr explicit ct(ct<U>)
    {
    }

    /// Implicit conversion to the value
    constexpr operator type() const
    {
        return value;
    }
};


/**
 * @brief A run-time ctrt: the base of an rt<T>, holding its value
 */
template <class T>
class ctrt<rt<T>, T>
{
   public:
    using type = T;
    T value;

    /// Implicit conversion to the value
    constexpr operator const T&() const
    {
        return value;
    }

   protected: // made by its rt; a ctrt itself is only copied
    constexpr ctrt(T v) : value(v)
    {
    }
};

/**
* @brief A value known at run time, held as data
*
* Converts from a plain value, an rt or a ct: implicitly if the value's type converts implicitly to T, explicitly if
* it converts only explicitly. Judged on the types, as for a function parameter of type T: nothing is checked for
* narrowing. Converts to its value implicitly. `rt x = 5u` deduces rt<unsigned>.
*
* Examples, with `struct X { explicit X(int); }`:
*
* | From        | To                | Result   | Reason                      |
* |-------------|-------------------|----------|-----------------------------|
* | int         | rt<std::uint32_t> | implicit |                             |
* | rt<int>     | rt<std::uint32_t> | implicit |                             |
* | ct<1>       | rt<std::uint32_t> | implicit |                             |
* | int         | rt<std::uint8_t>  | implicit | narrowing is not checked    |
* | int         | rt<X>             | explicit | the constructor is explicit |
* | const char* | rt<std::uint32_t> | none     | no conversion               |
* | rt<int>     | int               | implicit | its value                   |
*
* @tparam T The type of the value
*/
template <class T>
struct rt : ctrt<rt<T>, T>
{
    using type = T;
    using ctrt<rt<T>, T>::value; // the value is held by the base, so that a ctrt carries it

    constexpr rt(T v) : ctrt<rt<T>, T>(v)
    {
    }

    /// Implicit conversion from a plain value, an rt or a ct
    template <class U, std::enable_if_t<wrapper::detail::is_implicit_conversion_v<wrapper::detail::value_t<U>, T>, int> = 0>
    constexpr rt(const U& u) : ctrt<rt<T>, T>(wrapper::value(u))
    {
    }

    /// Explicit conversion from a plain value, an rt or a ct, when its type converts only explicitly
    template <class U, std::enable_if_t<wrapper::detail::is_explicit_conversion_v<wrapper::detail::value_t<U>, T>, int> = 0>
    constexpr explicit rt(const U& u) : ctrt<rt<T>, T>(static_cast<T>(wrapper::value(u)))
    {
    }

    /// Implicit conversion to the value
    constexpr operator const T&() const
    {
        return value;
    }
};

/**
 * @brief Constrain a function template to the template arguments for which C holds
 *
 * Written as a template argument: the function is removed from overload resolution when C is false.
 *
 * @code
 * template <class N, Requires<wrapper::is_ct_v<N>> = 0> void set(N n);
 * @endcode
 *
 * @tparam C The condition
 */
template <bool C>
using Requires = std::enable_if_t<C, int>;

// ---- Addition --------------------------------------------------------------

template <class X, class Y, std::enable_if_t<wrapper::detail::operands_v<X, Y>, int> = 0>
constexpr auto operator+(const X& x, const Y& y)
{
    using namespace wrapper;
    if constexpr (is_ct_v<X> && is_ct_v<Y>)
    {
        return ct<(constant<X> + constant<Y>)> {};
    }
    else
    {
        return rt(value(x) + value(y));
    }
}

// ---- Subtraction -----------------------------------------------------------

template <class X, class Y, std::enable_if_t<wrapper::detail::operands_v<X, Y>, int> = 0>
constexpr auto operator-(const X& x, const Y& y)
{
    using namespace wrapper;
    if constexpr (is_ct_v<X> && is_ct_v<Y>)
    {
        return ct<(constant<X> - constant<Y>)> {};
    }
    else
    {
        return rt(value(x) - value(y));
    }
}

// ---- Multiplication --------------------------------------------------------

// todo(sstanisic) : propagate ct<0> * rt<integral> -> ct<0>

template <class X, class Y, std::enable_if_t<wrapper::detail::operands_v<X, Y>, int> = 0>
constexpr auto operator*(const X& x, const Y& y)
{
    using namespace wrapper;
    if constexpr (is_ct_v<X> && is_ct_v<Y>)
    {
        return ct<(constant<X> * constant<Y>)> {};
    }
    else
    {
        return rt(value(x) * value(y));
    }
}

// ---- Division --------------------------------------------------------------

// todo(sstanisic) : propagate ct<0> / rt<integral> -> ct<0>

template <class X, class Y, std::enable_if_t<wrapper::detail::operands_v<X, Y>, int> = 0>
constexpr auto operator/(const X& x, const Y& y)
{
    using namespace wrapper;
    if constexpr (is_ct_v<X> && is_ct_v<Y>)
    {
        return ct<(constant<X> / constant<Y>)> {};
    }
    else
    {
        return rt(value(x) / value(y));
    }
}

// ---- Modulo ---------------------------------------------------------------

// todo(sstanisic) : propagate ct<0> % rt<integral> -> ct<0>

template <class X, class Y, std::enable_if_t<wrapper::detail::operands_v<X, Y>, int> = 0>
constexpr auto operator%(const X& x, const Y& y)
{
    using namespace wrapper;
    if constexpr (is_ct_v<X> && is_ct_v<Y>)
    {
        return ct<(constant<X> % constant<Y>)> {};
    }
    else
    {
        return rt(value(x) % value(y));
    }
}

// ---- Less than -------------------------------------------------------------

template <class X, class Y, std::enable_if_t<wrapper::detail::operands_v<X, Y>, int> = 0>
constexpr auto operator<(const X& x, const Y& y)
{
    using namespace wrapper;
    if constexpr (is_ct_v<X> && is_ct_v<Y>)
    {
        return ct<(constant<X> < constant<Y>)> {};
    }
    else
    {
        return rt(value(x) < value(y));
    }
}

// ---- Less than or equal ----------------------------------------------------

template <class X, class Y, std::enable_if_t<wrapper::detail::operands_v<X, Y>, int> = 0>
constexpr auto operator<=(const X& x, const Y& y)
{
    using namespace wrapper;
    if constexpr (is_ct_v<X> && is_ct_v<Y>)
    {
        return ct<(constant<X> <= constant<Y>)> {};
    }
    else
    {
        return rt(value(x) <= value(y));
    }
}

// ---- Greater than ----------------------------------------------------------

template <class X, class Y, std::enable_if_t<wrapper::detail::operands_v<X, Y>, int> = 0>
constexpr auto operator>(const X& x, const Y& y)
{
    using namespace wrapper;
    if constexpr (is_ct_v<X> && is_ct_v<Y>)
    {
        return ct<(constant<X> > constant<Y>)> {};
    }
    else
    {
        return rt(value(x) > value(y));
    }
}

// ---- Greater than or equal -------------------------------------------------

template <class X, class Y, std::enable_if_t<wrapper::detail::operands_v<X, Y>, int> = 0>
constexpr auto operator>=(const X& x, const Y& y)
{
    using namespace wrapper;
    if constexpr (is_ct_v<X> && is_ct_v<Y>)
    {
        return ct<(constant<X> >= constant<Y>)> {};
    }
    else
    {
        return rt(value(x) >= value(y));
    }
}

// ---- Equal ---------------------------------------------------------------

template <class X, class Y, std::enable_if_t<wrapper::detail::operands_v<X, Y>, int> = 0>
constexpr auto operator==(const X& x, const Y& y)
{
    using namespace wrapper;
    if constexpr (is_ct_v<X> && is_ct_v<Y>)
    {
        return ct<(constant<X> == constant<Y>)> {};
    }
    else
    {
        return rt(value(x) == value(y));
    }
}

// ---- Not equal -----------------------------------------------------------

template <class X, class Y, std::enable_if_t<wrapper::detail::operands_v<X, Y>, int> = 0>
constexpr auto operator!=(const X& x, const Y& y)
{
    using namespace wrapper;
    if constexpr (is_ct_v<X> && is_ct_v<Y>)
    {
        return ct<(constant<X> != constant<Y>)> {};
    }
    else
    {
        return rt(value(x) != value(y));
    }
}

// ---- Bitwise AND ----------------------------------------------------------

// todo(sstanisic) : propagate ct<0> & rt<integral> -> ct<0>

template <class X, class Y, std::enable_if_t<wrapper::detail::operands_v<X, Y>, int> = 0>
constexpr auto operator&(const X& x, const Y& y)
{
    using namespace wrapper;
    if constexpr (is_ct_v<X> && is_ct_v<Y>)
    {
        return ct<(constant<X> & constant<Y>)> {};
    }
    else
    {
        return rt(value(x) & value(y));
    }
}

// ---- Bitwise OR -----------------------------------------------------------

// todo(sstanisic) : propagate ct<0b1...1> | rt<integral> -> ct<0b1...1>

template <class X, class Y, std::enable_if_t<wrapper::detail::operands_v<X, Y>, int> = 0>
constexpr auto operator|(const X& x, const Y& y)
{
    using namespace wrapper;
    if constexpr (is_ct_v<X> && is_ct_v<Y>)
    {
        return ct<(constant<X> | constant<Y>)> {};
    }
    else
    {
        return rt(value(x) | value(y));
    }
}

// ---- Bitwise XOR -----------------------------------------------------------

template <class X, class Y, std::enable_if_t<wrapper::detail::operands_v<X, Y>, int> = 0>
constexpr auto operator^(const X& x, const Y& y)
{
    using namespace wrapper;
    if constexpr (is_ct_v<X> && is_ct_v<Y>)
    {
        return ct<(constant<X> ^ constant<Y>)> {};
    }
    else
    {
        return rt(value(x) ^ value(y));
    }
}

// ---- Left shift ------------------------------------------------------------

template <class X, class Y, std::enable_if_t<wrapper::detail::shift_operands_v<X, Y>, int> = 0>
constexpr auto operator<<(const X& x, const Y& y)
{
    using namespace wrapper;
    if constexpr (is_ct_v<X> && is_ct_v<Y>)
    {
        return ct<(constant<X> << constant<Y>)> {};
    }
    else
    {
        return rt(value(x) << value(y));
    }
}

// ---- Right shift -----------------------------------------------------------

template <class X, class Y, std::enable_if_t<wrapper::detail::shift_operands_v<X, Y>, int> = 0>
constexpr auto operator>>(const X& x, const Y& y)
{
    using namespace wrapper;
    if constexpr (is_ct_v<X> && is_ct_v<Y>)
    {
        return ct<(constant<X> >> constant<Y>)> {};
    }
    else
    {
        return rt(value(x) >> value(y));
    }
}

// ---- Logical AND -----------------------------------------------------------

// WARN: Does not short-circuit: both operands are always evaluated

// todo(sstanisic) : propagate ct<false> && rt<bool> -> ct<false>

template <class X, class Y, std::enable_if_t<wrapper::detail::operands_v<X, Y>, int> = 0>
constexpr auto operator&&(const X& x, const Y& y)
{
    using namespace wrapper;
    if constexpr (is_ct_v<X> && is_ct_v<Y>)
    {
        return ct<(constant<X> && constant<Y>)> {};
    }
    else
    {
        return rt(value(x) && value(y));
    }
}

// ---- Logical OR ------------------------------------------------------------

// WARN: Does not short-circuit: both operands are always evaluated

// todo(sstanisic) : propagate ct<true> || rt<bool> -> ct<true>

template <class X, class Y, std::enable_if_t<wrapper::detail::operands_v<X, Y>, int> = 0>
constexpr auto operator||(const X& x, const Y& y)
{
    using namespace wrapper;
    if constexpr (is_ct_v<X> && is_ct_v<Y>)
    {
        return ct<(constant<X> || constant<Y>)> {};
    }
    else
    {
        return rt(value(x) || value(y));
    }
}

// ---- Unary plus ------------------------------------------------------------

template <decltype(auto) A>
constexpr ct<(+A)> operator+(ct<A>)
{
    return {};
}

template <class T>
constexpr auto operator+(const rt<T>& x)
{
    return rt(+x.value);
}

// ---- Unary minus -----------------------------------------------------------

template <decltype(auto) A>
constexpr ct<(-A)> operator-(ct<A>)
{
    return {};
}

template <class T>
constexpr auto operator-(const rt<T>& x)
{
    return rt(-x.value);
}

// ---- Logical NOT ------------------------------------------------------------

template <decltype(auto) A>
constexpr ct<(!A)> operator!(ct<A>)
{
    return {};
}

template <class T>
constexpr auto operator!(const rt<T>& x)
{
    return rt(!x.value);
}

// ---- Bitwise NOT ------------------------------------------------------------

template <decltype(auto) A>
constexpr ct<(~A)> operator~(ct<A>)
{
    return {};
}

template <class T>
constexpr auto operator~(const rt<T>& x)
{
    return rt(~x.value);
}

// ---- unary operators on a ctrt ---------------------------------------------------------------------------------

template <class C, class T>
constexpr auto operator+(const ctrt<C, T>& x)
{
    using namespace wrapper;
    if constexpr (is_ct_v<C>)
    {
        return ct<(+C::value)> {};
    }
    else
    {
        return rt(+x.value);
    }
}

template <class C, class T>
constexpr auto operator-(const ctrt<C, T>& x)
{
    using namespace wrapper;
    if constexpr (is_ct_v<C>)
    {
        return ct<(-C::value)> {};
    }
    else
    {
        return rt(-x.value);
    }
}

template <class C, class T>
constexpr auto operator!(const ctrt<C, T>& x)
{
    using namespace wrapper;
    if constexpr (is_ct_v<C>)
    {
        return ct<(!C::value)> {};
    }
    else
    {
        return rt(!x.value);
    }
}

template <class C, class T>
constexpr auto operator~(const ctrt<C, T>& x)
{
    using namespace wrapper;
    if constexpr (is_ct_v<C>)
    {
        return ct<(~C::value)> {};
    }
    else
    {
        return rt(~x.value);
    }
}

// ---- casts -----------------------------------------------------------------------------------------------------

/**
 * @brief Cast the value of a ct to T with static_cast; the result is a new ct, still compile time
 *
 * Follows static_cast's rules (is_static_castable): nothing is checked for narrowing.
 *
 * Examples, with `enum class E : std::uint8_t`:
 *
 * | Expression                           | Result          |
 * |--------------------------------------|-----------------|
 * | value_cast<E>(ct<1> {})              | ct<E {1}>       |
 * | value_cast<std::uint32_t>(ct<-1> {}) | ct<4294967295u> |
 *
 * @tparam T The type to cast the value to
 * @tparam V The value of the ct, deduced
 * @return A ct of the cast value
 */
template <class T, decltype(auto) V, std::enable_if_t<wrapper::detail::is_static_castable_v<decltype(V), T>, int> = 0>
constexpr ct<static_cast<T>(V)> value_cast(ct<V>)
{
    return {};
}

/**
 * @brief Cast the value of an rt to T with static_cast; the result is an rt<T>
 *
 * @tparam T The type to cast the value to
 * @tparam U The type of the rt's value, deduced
 * @param v The rt to cast
 * @return An rt<T> of the cast value
 */
template <class T, class U, std::enable_if_t<wrapper::detail::is_static_castable_v<const U&, T>, int> = 0>
constexpr rt<T> value_cast(const rt<U>& v)
{
    return rt<T>(static_cast<T>(v.value));
}

/**
 * @brief Cast a plain value to T with static_cast; the result is an rt<T>, so that value_cast gives a ct or an rt for
 * a value in any form
 *
 * @tparam T The type to cast the value to
 * @tparam U The type of the value, deduced
 * @param v The value to cast
 * @return An rt<T> of the cast value
 */
template <class T, class U, std::enable_if_t<!wrapper::is_wrapper_v<U> && wrapper::detail::is_static_castable_v<const U&, T>, int> = 0>
constexpr rt<T> value_cast(const U& v)
{
    return rt<T>(static_cast<T>(v));
}

/**
 * @brief Cast a compile-time ctrt to T with static_cast; the result is a ct, still compile time
 *
 * For passing a parameter on to a function whose parameter has another type.
 *
 * @tparam T The type to cast the value to
 * @tparam V The value of the ct the parameter is, deduced
 * @return A ct of the cast value
 */
template <class T, decltype(auto) V, class S, std::enable_if_t<wrapper::detail::is_static_castable_v<decltype(V), T>, int> = 0>
constexpr ct<static_cast<T>(V)> value_cast(const ctrt<ct<V>, S>&)
{
    return {};
}

/**
 * @brief Cast a run-time ctrt to T with static_cast; the result is an rt<T>
 *
 * @tparam T The type to cast the value to
 * @tparam U The type of the parameter's value, deduced
 * @param parameter The parameter to cast
 * @return An rt<T> of the cast value
 */
template <class T, class U, std::enable_if_t<wrapper::detail::is_static_castable_v<const U&, T>, int> = 0>
constexpr rt<T> value_cast(const ctrt<rt<U>, U>& parameter)
{
    return rt<T>(static_cast<T>(parameter.value));
}

} // namespace llk
