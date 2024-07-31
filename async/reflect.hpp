// <!--
//
// Copyright (c) 2024 Kris Jusiak (kris at jusiak dot net)
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)
//
#ifdef README
// -->
[![Boost Licence](http://img.shields.io/badge/license-boost-blue.svg)](http://www.boost.org/LICENSE_1_0.txt)
[![Version](https://badge.fury.io/gh/qlibs%2Freflect.svg)](https://github.com/qlibs/reflect/releases)
[![build](https://img.shields.io/badge/build-blue.svg)](https://godbolt.org/z/zvooxGPP9)
[![Try it online](https://img.shields.io/badge/try%20it-online-blue.svg)](https://godbolt.org/z/oYhh1hfeo)

---------------------------------------

## reflect: C++20 Static Reflection library

> https://en.wikipedia.org/wiki/Reflective_programming

### Features

- Single header (https://raw.githubusercontent.com/qlibs/reflect/main/reflect - for integration see [FAQ](#faq))
- Minimal [API](#api)
- Verifies itself upon include (can be disabled with `-DNTEST` - see [FAQ](#faq))
    - Basically guarantees no UB, no memory leaks*
- Compiles cleanly with ([`-fno-exceptions -fno-rtti -Wall -Wextra -Werror -pedantic -pedantic-errors | /W4 /WX`](https://godbolt.org/z/M747ocGfx))
- Agnostic to compiler changes (no ifdefs for the compiler specific implementations - see [FAQ](#faq))
- Optimized run-time execution and binary size (see [performance](#perf))
- Fast compilation times (see [compilation times](#comp))

### Requirements

- C++20 ([gcc-12+, clang-15+, msvc-19.36+](https://godbolt.org/z/xPc19Moef))
    - STL headers (array, string_view, source_location, type_traits, utility, [tuple])

---

### Hello world (https://godbolt.org/z/oYhh1hfeo)

```cpp
#include <reflect>

enum E { A, B };
struct foo {
    int a;
    E b; };

constexpr auto f = foo{.a = 42, .b = B};

// reflect::size
static_assert(2 == reflect::size(f));

// reflect::type_id
static_assert(reflect::type_id(f.a) != reflect::type_id(f.b));

// reflect::type_name
static_assert("foo"sv == reflect::type_name(f));
static_assert("int"sv == reflect::type_name(f.a));
static_assert("E"sv   == reflect::type_name(f.b));

// reflect::enum_name
static_assert("B"sv == reflect::enum_name(f.b));

// reflect::member_name
static_assert("a"sv == reflect::member_name<0>(f));
static_assert("b"sv == reflect::member_name<1>(f));

// reflect::get
static_assert(42 == reflect::get<0>(f)); // by index
static_assert(B  == reflect::get<1>(f));

static_assert(42 == reflect::get<"a">(f)); // by name
static_assert(B  == reflect::get<"b">(f));

// reflect::to
constexpr auto t = reflect::to<std::tuple>(f);
static_assert(42 == std::get<0>(t));
static_assert(B  == std::get<1>(t));

int main() {
    reflect::for_each(
        [](auto I) {
            std::print(
                "{}.{}:{}={} ({}/{}/{})\n",
                reflect::type_name(f),                   // foo, foo
                reflect::member_name<I>(f),              // a  , b
                reflect::type_name(reflect::get<I>(f)),  // int, E
                reflect::get<I>(f),                      // 42 , B
                reflect::size_of<I>(f),                  // 4  , 4
                reflect::align_of<I>(f),                 // 4  , 4
                reflect::offset_of<I>(f));               // 0  , 4
        },
        f);
}

// and more (see API)...
```

---

### Examples

- [feature] Opt-in mixins - https://godbolt.org/z/sj7fYKoc3
- [feature] Meta-programming (https://github.com/qlibs/mp) - https://godbolt.org/z/ds3KMGhqP
- [future] Structured Bindings can introduce a Pack (https://wg21.link/P1061) - https://godbolt.org/z/Ga3bc3KKW
- [performance] Minimal Perfect Hashing based `enum_name` (https://github.com/qlibs/mph) - https://godbolt.org/z/WM155vTfv

---

<a name="perf"></a>
### Performance/Binary size (https://godbolt.org/z/7TbobjWfj)

```cpp
struct foo {
    int bar; };
auto type_name(const foo& f) {
    return reflect::type_name(f); }
```

```asm
type_name(foo const&): // $CXX -O3 -DNDEBUG
        lea     rdx, [rip + type_name<foo>]
        mov     eax, 3
        ret

type_name<foo>
        .ascii  "foo"
```

```cpp
struct foo {
    int bar; };
auto member_name(const foo& f) {
    return reflect::member_name<0>(f); }
```

```asm
member_name(foo const&): // $CXX -O3 -DNDEBUG
        lea     rdx, [rip + member_name<0ul, foo>]
        mov     eax, 3
        ret

member_name<0ul, foo>
        .ascii  "bar"
```

```cpp
enum class E { A, B, };
auto enum_name(const E e) {
    return reflect::enum_name(e); }
```

```asm
enum_name(E): // $CXX -O3 -DNDEBUG (generates switch)
        xor     eax, eax
        xor     ecx, ecx
        cmp     edi, 1
        sete    cl
        lea     rdx, [rip + enum_name<0>]
        cmove   rax, rdx
        test    edi, edi
        lea     rdx, [rip + enum_name<1>]
        cmovne  rdx, rax
        mov     eax, 1
        cmovne  rax, rcx
        ret

enum_name<0ul>:
        .ascii  "A"

enum_name<1ul>:
        .ascii  "B"
```

<a name="comp"></a>
### Compilation times

> [include] https://raw.githubusercontent.com/qlibs/reflect/main/reflect

```cpp
time g++-13.2 -x c++ -std=c++20 reflect -c -DNTEST   # 0.113s
time g++-13.2 -x c++ -std=c++20 reflect -c           # 0.253s
```

```cpp
time clang++-17 -x c++ -std=c++20 reflect -c -DNTEST # 0.119s
time clang++-17 -x c++ -std=c++20 reflect -c         # 0.322s
```

---

### API

```cpp
template <class Fn, class T> requires std::is_aggregate_v<std::remove_cvref_t<T>>
[[nodiscard]] constexpr auto visit(Fn&& fn, T&& t) noexcept;
```

```cpp
struct foo {
    int a;
    int b; };
static_assert(2 == visit([](auto&&... args) {
    return sizeof...(args); }, foo{}));
```

```cpp
template<class T> requires std::is_aggregate_v<T>
[[nodiscard]] constexpr auto size() -> std::size_t;

template<class T> requires std::is_aggregate_v<T>
[[nodiscard]] constexpr auto size(const T&) -> std::size_t;
```

```cpp
struct foo {
    int a;
    int b; } f;
static_assert(2 == size<foo>());
static_assert(2 == size(f));
```

```cpp
template <class T> [[nodiscard]] constexpr auto type_name() noexcept;
template <class T> [[nodiscard]] constexpr auto type_name(const T&) noexcept;
```

```cpp
struct foo {
    int a;
    int b; };
static_assert(std::string_view{"foo"} == type_name<foo>());
static_assert(std::string_view{"foo"} == type_name(foo{}));
```

```cpp
template <class T> [[nodiscard]] constexpr auto type_id() noexcept;
template <class T> [[nodiscard]] constexpr auto type_id(T&&) noexcept;
```

```cpp
struct foo { };
struct bar { };
static_assert(type_id(foo{}) == type_id(foo{}));
static_assert(type_id(bar{}) != type_id<foo>());
```

```cpp
template<class E>
[[nodiscard]] constexpr auto to_underlying(const E e) noexcept;

template<class E> requires std::is_enum_v<E>
consteval auto enum_min(const E = {}) {
    return REFLECT_ENUM_MIN; }

template<class E> requires std::is_enum_v<E>
consteval auto enum_max(const E = {}) {
    return REFLECT_ENUM_MAX; }

template<class E,
         fixed_string unknown = "",
         auto Min = enum_min(E{}),
         auto Max = enum_max(E{})>
  requires (std::is_enum_v<E> and Max > Min)
[[nodiscard]] constexpr auto enum_name(const E e) noexcept -> std::string_view {
    ```

```cpp enum class Enum {
        foo = 1,
        bar = 2
    };
    static_assert(std::string_view{"foo"} == enum_name(Enum::foo));
    static_assert(std::string_view{"bar"} == enum_name(Enum::bar));
    ```

```cpp enum class Enum {
        foo = 1,
        bar = 1024
    };
    consteval auto enum_min(Enum) { return Enum::foo; }
    consteval auto enum_max(Enum) { return Enum::bar; }

    static_assert(std::string_view{"foo"} == enum_name(Enum::foo));
    static_assert(std::string_view{"bar"} == enum_name(Enum::bar));
    ```

```cpp template <std::size_t N, class T>
        requires(std::is_aggregate_v<T> and N < size<T>())
    [[nodiscard]] constexpr auto member_name(const T& = {}) noexcept;
    ```

```cpp struct foo {
        int a;
        int b;
    };
    static_assert(std::string_view{"a"} == member_name<0, foo>());
    static_assert(std::string_view{"a"} == member_name<0>(foo{}));
    static_assert(std::string_view{"b"} == member_name<1, foo>());
    static_assert(std::string_view{"b"} == member_name<1>(foo{}));
    ```

```cpp template <std::size_t N, class T>
        requires(std::is_aggregate_v<std::remove_cvref_t<T>> and N < size<std::remove_cvref_t<T>>())
    [[nodiscard]] constexpr decltype(auto) get(T && t) noexcept;
    ```

```cpp struct foo {
        int a;
        bool b;
    };
    constexpr auto f = foo{.i = 42, .b = true};
    static_assert(42 == get<0>(f));
    static_assert(true == get<1>(f));
    ```

```cpp template <class T, fixed_string Name>
        requires std::is_aggregate_v<T>
    concept has_member_name = /*unspecified*/
```

```cpp struct foo {
        int a;
        int b;
    };
    static_assert(has_member_name<foo, "a">);
    static_assert(has_member_name<foo, "b">);
    static_assert(not has_member_name<foo, "c">);
    ```

```cpp template <fixed_string Name, class T>
        requires has_member_name<T, Name>
    constexpr decltype(auto) get(T && t) noexcept;
    ```

```cpp struct foo {
        int a;
        int b;
    };
    constexpr auto f = foo{.i = 42, .b = true};
    static_assert(42 == get<"a">(f));
    static_assert(true == get<"b">(f));
    ```

```cpp template <fixed_string... Members, class TSrc, class TDst>
        requires(std::is_aggregate_v<TSrc> and std::is_aggregate_v<TDst>)
    constexpr auto copy(const TSrc& src, TDst& dst) noexcept -> void;
    ```

```cpp struct foo {
        int a;
        int b;
    };
    struct bar {
        int a{};
        int b{};
    };

    bar b{};
    foo f{};

    copy(f, b);
    assert(b.a == f.a);
    assert(b.b == f.b);

    copy<"a">(f, b);
    assert(b.a == f.a);
    assert(0 == b.b);
    ```

```cpp template <template <class...> class R, class T>
        requires std::is_aggregate_v<std::remove_cvref_t<T>>
    [[nodiscard]] constexpr auto to(T && t) noexcept;
    ```

```cpp struct foo {
        int a;
        int b;
    };

    constexpr auto t = to<std::tuple>(foo{.a = 4, .b = 2});
    static_assert(4 == std::get<0>(t));
    static_assert(2 == std::get<1>(t));

    auto f = foo{.a = 4, .b = 2};
    auto t = to<std::tuple>(f);
    std::get<0>(t) *= 10;
    f.b = 42;
    assert(40 == std::get<0>(t) and 40 == f.a);
    assert(42 == std::get<1>(t) and 42 == f.b);
    ```

```cpp template <class R, class T>
    [[nodiscard]] constexpr auto to(T && t);
    ```

```cpp struct foo {
        int a;
        int b;
    };
    struct baz {
        int a{};
        int c{};
    };

    const auto b = to<baz>(foo{.a = 4, .b = 2});
    assert(4 == b.a and 0 == b.c);
    ```

```cpp template <std::size_t N, class T>
        requires std::is_aggregate_v<T>
    [[nodiscard]] constexpr auto size_of() -> std::size_t;

    template <std::size_t N, class T>
        requires std::is_aggregate_v<T>
    [[nodiscard]] constexpr auto size_of(T&&) -> std::size_t;

    template <std::size_t N, class T>
        requires std::is_aggregate_v<T>
    [[nodiscard]] constexpr auto align_of() -> std::size_t;

    template <std::size_t N, class T>
        requires std::is_aggregate_v<T>
    [[nodiscard]] constexpr auto align_of(T&&) -> std::size_t;

    template <std::size_t N, class T>
        requires std::is_aggregate_v<T>
    [[nodiscard]] constexpr auto offset_of() -> std::size_t;

    template <std::size_t N, class T>
        requires std::is_aggregate_v<T>
    [[nodiscard]] constexpr auto offset_of(T&&) -> std::size_t;
    ```

```cpp struct foo {
        int a;
        bool b;
    };

    static_assert(4 == size_of<0, foo>());
    static_assert(1 == size_of<1, foo>());
    static_assert(4 == align_of<0, foo>());
    static_assert(1 == align_of<1, foo>());
    static_assert(0 == offset_of<0, foo>());
    static_assert(4 == offset_of<1, foo>());
    ```

```cpp template <class Fn, class T>
        requires std::is_aggregate_v<std::remove_cvref_t<T>>
    constexpr auto for_each(Fn && fn)->void;

    template <class Fn, class T>
        requires std::is_aggregate_v<std::remove_cvref_t<T>>
    constexpr auto for_each(Fn && fn, T && t)->void;
    ```

```cpp struct {
        int a;
        int b;
    } f;

    reflect::for_each(
        [&f](const auto I) {
            std::print("{}:{}={}", member_name<I>(f), get<I>(f));  // prints a:int=4, b:int=2
        },
        f);
    ```

        > Configuration

```cpp
#define REFLECT 1'2'2  // Current library version (SemVer)
#define REFLECT_ENUM_MIN \
    0  // Min size for enum name (can be overridden)
       // For example: `-DREFLECT_ENUM_MIN=-1`
#define REFLECT_ENUM_MAX \
    128        // Max size for enum name (can be overridden)
               // For example: `-DREFLECT_ENUM_MAX=32`
#define NTEST  // Disables running static_asserts tests
               // Not enabled by default (use with caution)
``` -- -

              ## #FAQ

              - How does `reflect` compare to https :  // wg21.link/P2996?

                                                       > `reflect` library only provides basic reflection primitives,
        mostly via hacks and workarounds to deal with lack of the reflection.https
        :  // wg21.link/P2996 is a language proposal with many more features and capabilities.

           -How does `reflect` work under the hood
        ?

        > There are many different ways to implement reflection. `reflect` uses C++ 20's structure bindings,
        concepts and source_location to do it.See `visit` implementation for more details.

            - How can `reflect` be agnostic to compiler changes
        ?

        > `reflect` precomputes required prefixes /
                    postfixes to find required names from the `source_location::function_name()` output for each compiler
                        upon inclusion.Any compiler change will end up with new prefixes /
                    postfixes and wont require additional maintenance.

                - What does it mean that `reflect` tests itself upon include
        ?

        > `reflect` runs all tests(via static_asserts) upon include.If the include compiles it means all tests
                are passing and the library works correctly on given compiler,
        environment.

            - What is compile - time overhead of `reflect` library
        ?

        > `reflect` include takes ~.2s(that includes running all tests)
                            .The most expensive calls are `visit` and `enum_to_name` whose timing will depend on the
                                number of reflected elements and /
                or min / max values provided.There are no recursive template instantiations in the library.

                       - Can I disable running tests at compile - time for faster compilation times
            ?

            > When `- DNTEST` is defined static_asserts tests wont be executed upon inclusion.Note
            : Use with caution as disabling tests means that there are no guarantees upon inclusion that the given
                          compiler /
                      env combination works as expected.

                  - How to extend the number of members to be reflected(default : 64)
            ?

            > Override `visit`,
        for example - https :             // godbolt.org/z/Ga3bc3KKW

    ```cpp template <class Fn, class T>  // requires https://wg21.link/P1061
                               [[nodiscard]] constexpr decltype(auto) visit(Fn && fn, T && t) noexcept {
        auto&& [... ts] = std::forward<T>(t);
        return std::forward<Fn>(fn)(std::forward_like<T>(ts)...);
    }
    ```

        - How to integrate with CMake / CPM
        ?

    ``` CPMAddPackage(Name reflect GITHUB_REPOSITORY qlibs / reflect GIT_TAG v1 .2.2) add_library(reflect INTERFACE)
            target_include_directories(reflect SYSTEM INTERFACE ${reflect_SOURCE_DIR})
                add_library(reflect::reflect ALIAS reflect)
    ```

    ``` target_link_libraries(${PROJECT_NAME} reflect::reflect);
```

    - Similar projects
    ? > [boost.pfr](https
                    :  // github.com/boostorg/pfr), [glaze](https://github.com/stephenberry/glaze),
                       // [reflect-cpp](https://github.com/getml/reflect-cpp),
                       // [magic_enum](https://github.com/Neargye/magic_enum)
                    < !--
#else
#ifndef REFLECT
#define REFLECT 1'2'2  // SemVer

#include <array>
#include <cstdint>
#include <limits>
#include <source_location>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <utility>

#ifndef REFLECT_ENUM_MIN
#define REFLECT_ENUM_MIN 0
#endif

#ifndef REFLECT_ENUM_MAX
#define REFLECT_ENUM_MAX 128
#endif

namespace {
template <bool Cond>
struct REFLECT_FWD_LIKE {
    template <class T>
    using type = std::remove_reference_t<T>&&;
};
template <>
struct REFLECT_FWD_LIKE<true> {
    template <class T>
    using type = std::remove_reference_t<T>&;
};
}  // namespace

#define REFLECT_FWD(...) static_cast<decltype(__VA_ARGS__)&&>(__VA_ARGS__)
#define REFLECT_FWD_LIKE(T, ...)                                                                                     \
    static_cast<typename ::REFLECT_FWD_LIKE<::std::is_lvalue_reference_v<T>>::template type<decltype(__VA_ARGS__)>>( \
        __VA_ARGS__)
struct REFLECT_STRUCT {
    void* MEMBER;
    enum class ENUM { VALUE };
};  // has to be in the global namespace

namespace reflect::inline v1_2_2 {
namespace detail {
template <class T>
extern const T ext{};
struct any {
    template <class T>
    operator T() const noexcept;
};
template <class T>
struct any_except_base_of {
    template <class U>
        requires(not std::is_base_of_v<U, T>)
    operator U() const noexcept;
};
template <auto...>
struct auto_ {
    constexpr explicit(false) auto_(auto&&...) noexcept {}
};
template <class T>
struct ref {
    T& ref_;
};
template <class T>
ref(T&) -> ref<T>;

template <std::size_t N>
constexpr auto nth_pack_element_impl = []<auto... Ns>(std::index_sequence<Ns...>) -> decltype(auto) {
    return [](auto_<Ns>&&..., auto&& nth, auto&&...) -> decltype(auto) { return REFLECT_FWD(nth); };
}(std::make_index_sequence<N>{});

template <std::size_t N, class... Ts>
    requires(N < sizeof...(Ts))
[[nodiscard]] constexpr decltype(auto) nth_pack_element(Ts&&... args) noexcept {
    return nth_pack_element_impl<N>(REFLECT_FWD(args)...);
}

template <auto... Vs>
[[nodiscard]] constexpr auto function_name() noexcept -> std::string_view {
    return std::source_location::current().function_name();
}
template <class... Ts>
[[nodiscard]] constexpr auto function_name() noexcept -> std::string_view {
    return std::source_location::current().function_name();
}

template <class T>
struct type_name_info {
    static constexpr auto name = function_name<int>();
    static constexpr auto begin = name.find("int");
    static constexpr auto end = name.substr(begin + std::size(std::string_view{"int"}));
};

template <class T>
    requires std::is_class_v<T>
struct type_name_info<T> {
    static constexpr auto name = function_name<REFLECT_STRUCT>();
    static constexpr auto begin = name.find("REFLECT_STRUCT");
    static constexpr auto end = name.substr(begin + std::size(std::string_view{"REFLECT_STRUCT"}));
};

template <class T>
    requires std::is_enum_v<T>
struct type_name_info<T> {
    static constexpr auto name = function_name<REFLECT_STRUCT::ENUM>();
    static constexpr auto begin = name.find("REFLECT_STRUCT::ENUM");
    static constexpr auto end = name.substr(begin + std::size(std::string_view{"REFLECT_STRUCT::ENUM"}));
};

struct enum_name_info {
    static constexpr auto name = function_name<REFLECT_STRUCT::ENUM::VALUE>();
    static constexpr auto begin = name.find("REFLECT_STRUCT::ENUM::VALUE");
    static constexpr auto end = std::size(name) - (name.find("REFLECT_STRUCT::ENUM::VALUE") +
                                                   std::size(std::string_view{"REFLECT_STRUCT::ENUM::VALUE"}));
};

struct member_name_info {
    static constexpr auto name = function_name<ref{ext<REFLECT_STRUCT>.MEMBER}>();
    static constexpr auto begin = name[name.find("MEMBER") - 1];
    static constexpr auto end = name.substr(name.find("MEMBER") + std::size(std::string_view{"MEMBER"}));
};
}  // namespace detail

template <class T, std::size_t Size>
struct fixed_string {
    constexpr explicit(false) fixed_string(const T* str) {
        for (decltype(Size) i{}; i < Size; ++i) {
            data[i] = str[i];
        }
        data[Size] = T();
    }
    [[nodiscard]] constexpr auto operator<=>(const fixed_string&) const = default;
    [[nodiscard]] constexpr explicit(false) operator std::string_view() const { return {std::data(data), Size}; }
    [[nodiscard]] constexpr auto size() const { return Size; }
    T data[Size + 1];
};
template <class T, std::size_t Capacity, std::size_t Size = Capacity - 1>
fixed_string(const T (&str)[Capacity]) -> fixed_string<T, Size>;

namespace detail {
template <class T, std::size_t Bases = 0, class... Ts>
    requires std::is_aggregate_v<T>
[[nodiscard]] constexpr auto size() -> std::size_t {
    if constexpr (requires { T{Ts{}...}; } and not requires { T{Ts{}..., detail::any{}}; }) {
        return sizeof...(Ts) - Bases;
    } else if constexpr (
        Bases == sizeof...(Ts) and requires { T{Ts{}...}; } and
        not requires { T{Ts{}..., detail::any_except_base_of<T>{}}; }) {
        return size<T, Bases + 1, Ts..., detail::any>();
    } else {
        return size<T, Bases, Ts..., detail::any>();
    }
}
}  // namespace detail

template <class T>
    requires std::is_aggregate_v<std::remove_cvref_t<T>>
[[nodiscard]] constexpr auto size() -> std::size_t {
    return detail::size<std::remove_cvref_t<T>>();
}

template <class T>
    requires std::is_aggregate_v<T>
[[nodiscard]] constexpr auto size(const T&) -> std::size_t {
    return detail::size<T>();
}

namespace detail {
template <class Fn, class T>
[[nodiscard]] constexpr decltype(auto) visit(Fn&& fn, T&&, std::integral_constant<std::size_t, 0>) noexcept {
    return REFLECT_FWD(fn)();
}
template <class Fn, class T>
[[nodiscard]] constexpr decltype(auto) visit(Fn&& fn, T&& t, std::integral_constant<std::size_t, 1>) noexcept {
    auto&& [_1] = REFLECT_FWD(t);
    return REFLECT_FWD(fn)(REFLECT_FWD_LIKE(T, _1));
}
template <class Fn, class T>
[[nodiscard]] constexpr decltype(auto) visit(Fn&& fn, T&& t, std::integral_constant<std::size_t, 2>) noexcept {
    auto&& [_1, _2] = REFLECT_FWD(t);
    return REFLECT_FWD(fn)(REFLECT_FWD_LIKE(T, _1), REFLECT_FWD_LIKE(T, _2));
}
template <class Fn, class T>
[[nodiscard]] constexpr decltype(auto) visit(Fn&& fn, T&& t, std::integral_constant<std::size_t, 3>) noexcept {
    auto&& [_1, _2, _3] = REFLECT_FWD(t);
    return REFLECT_FWD(fn)(REFLECT_FWD_LIKE(T, _1), REFLECT_FWD_LIKE(T, _2), REFLECT_FWD_LIKE(T, _3));
}
template <class Fn, class T>
[[nodiscard]] constexpr decltype(auto) visit(Fn&& fn, T&& t, std::integral_constant<std::size_t, 4>) noexcept {
    auto&& [_1, _2, _3, _4] = REFLECT_FWD(t);
    return REFLECT_FWD(fn)(
        REFLECT_FWD_LIKE(T, _1), REFLECT_FWD_LIKE(T, _2), REFLECT_FWD_LIKE(T, _3), REFLECT_FWD_LIKE(T, _4));
}
template <class Fn, class T>
[[nodiscard]] constexpr decltype(auto) visit(Fn&& fn, T&& t, std::integral_constant<std::size_t, 5>) noexcept {
    auto&& [_1, _2, _3, _4, _5] = REFLECT_FWD(t);
    return REFLECT_FWD(fn)(
        REFLECT_FWD_LIKE(T, _1),
        REFLECT_FWD_LIKE(T, _2),
        REFLECT_FWD_LIKE(T, _3),
        REFLECT_FWD_LIKE(T, _4),
        REFLECT_FWD_LIKE(T, _5));
}
template <class Fn, class T>
[[nodiscard]] constexpr decltype(auto) visit(Fn&& fn, T&& t, std::integral_constant<std::size_t, 6>) noexcept {
    auto&& [_1, _2, _3, _4, _5, _6] = REFLECT_FWD(t);
    return REFLECT_FWD(fn)(
        REFLECT_FWD_LIKE(T, _1),
        REFLECT_FWD_LIKE(T, _2),
        REFLECT_FWD_LIKE(T, _3),
        REFLECT_FWD_LIKE(T, _4),
        REFLECT_FWD_LIKE(T, _5),
        REFLECT_FWD_LIKE(T, _6));
}
template <class Fn, class T>
[[nodiscard]] constexpr decltype(auto) visit(Fn&& fn, T&& t, std::integral_constant<std::size_t, 7>) noexcept {
    auto&& [_1, _2, _3, _4, _5, _6, _7] = REFLECT_FWD(t);
    return REFLECT_FWD(fn)(
        REFLECT_FWD_LIKE(T, _1),
        REFLECT_FWD_LIKE(T, _2),
        REFLECT_FWD_LIKE(T, _3),
        REFLECT_FWD_LIKE(T, _4),
        REFLECT_FWD_LIKE(T, _5),
        REFLECT_FWD_LIKE(T, _6),
        REFLECT_FWD_LIKE(T, _7));
}
template <class Fn, class T>
[[nodiscard]] constexpr decltype(auto) visit(Fn&& fn, T&& t, std::integral_constant<std::size_t, 8>) noexcept {
    auto&& [_1, _2, _3, _4, _5, _6, _7, _8] = REFLECT_FWD(t);
    return REFLECT_FWD(fn)(
        REFLECT_FWD_LIKE(T, _1),
        REFLECT_FWD_LIKE(T, _2),
        REFLECT_FWD_LIKE(T, _3),
        REFLECT_FWD_LIKE(T, _4),
        REFLECT_FWD_LIKE(T, _5),
        REFLECT_FWD_LIKE(T, _6),
        REFLECT_FWD_LIKE(T, _7),
        REFLECT_FWD_LIKE(T, _8));
}
template <class Fn, class T>
[[nodiscard]] constexpr decltype(auto) visit(Fn&& fn, T&& t, std::integral_constant<std::size_t, 9>) noexcept {
    auto&& [_1, _2, _3, _4, _5, _6, _7, _8, _9] = REFLECT_FWD(t);
    return REFLECT_FWD(fn)(
        REFLECT_FWD_LIKE(T, _1),
        REFLECT_FWD_LIKE(T, _2),
        REFLECT_FWD_LIKE(T, _3),
        REFLECT_FWD_LIKE(T, _4),
        REFLECT_FWD_LIKE(T, _5),
        REFLECT_FWD_LIKE(T, _6),
        REFLECT_FWD_LIKE(T, _7),
        REFLECT_FWD_LIKE(T, _8),
        REFLECT_FWD_LIKE(T, _9));
}
template <class Fn, class T>
[[nodiscard]] constexpr decltype(auto) visit(Fn&& fn, T&& t, std::integral_constant<std::size_t, 10>) noexcept {
    auto&& [_1, _2, _3, _4, _5, _6, _7, _8, _9, _10] = REFLECT_FWD(t);
    return REFLECT_FWD(fn)(
        REFLECT_FWD_LIKE(T, _1),
        REFLECT_FWD_LIKE(T, _2),
        REFLECT_FWD_LIKE(T, _3),
        REFLECT_FWD_LIKE(T, _4),
        REFLECT_FWD_LIKE(T, _5),
        REFLECT_FWD_LIKE(T, _6),
        REFLECT_FWD_LIKE(T, _7),
        REFLECT_FWD_LIKE(T, _8),
        REFLECT_FWD_LIKE(T, _9),
        REFLECT_FWD_LIKE(T, _10));
}
template <class Fn, class T>
[[nodiscard]] constexpr decltype(auto) visit(Fn&& fn, T&& t, std::integral_constant<std::size_t, 11>) noexcept {
    auto&& [_1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11] = REFLECT_FWD(t);
    return REFLECT_FWD(fn)(
        REFLECT_FWD_LIKE(T, _1),
        REFLECT_FWD_LIKE(T, _2),
        REFLECT_FWD_LIKE(T, _3),
        REFLECT_FWD_LIKE(T, _4),
        REFLECT_FWD_LIKE(T, _5),
        REFLECT_FWD_LIKE(T, _6),
        REFLECT_FWD_LIKE(T, _7),
        REFLECT_FWD_LIKE(T, _8),
        REFLECT_FWD_LIKE(T, _9),
        REFLECT_FWD_LIKE(T, _10),
        REFLECT_FWD_LIKE(T, _11));
}
template <class Fn, class T>
[[nodiscard]] constexpr decltype(auto) visit(Fn&& fn, T&& t, std::integral_constant<std::size_t, 12>) noexcept {
    auto&& [_1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12] = REFLECT_FWD(t);
    return REFLECT_FWD(fn)(
        REFLECT_FWD_LIKE(T, _1),
        REFLECT_FWD_LIKE(T, _2),
        REFLECT_FWD_LIKE(T, _3),
        REFLECT_FWD_LIKE(T, _4),
        REFLECT_FWD_LIKE(T, _5),
        REFLECT_FWD_LIKE(T, _6),
        REFLECT_FWD_LIKE(T, _7),
        REFLECT_FWD_LIKE(T, _8),
        REFLECT_FWD_LIKE(T, _9),
        REFLECT_FWD_LIKE(T, _10),
        REFLECT_FWD_LIKE(T, _11),
        REFLECT_FWD_LIKE(T, _12));
}
template <class Fn, class T>
[[nodiscard]] constexpr decltype(auto) visit(Fn&& fn, T&& t, std::integral_constant<std::size_t, 13>) noexcept {
    auto&& [_1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13] = REFLECT_FWD(t);
    return REFLECT_FWD(fn)(
        REFLECT_FWD_LIKE(T, _1),
        REFLECT_FWD_LIKE(T, _2),
        REFLECT_FWD_LIKE(T, _3),
        REFLECT_FWD_LIKE(T, _4),
        REFLECT_FWD_LIKE(T, _5),
        REFLECT_FWD_LIKE(T, _6),
        REFLECT_FWD_LIKE(T, _7),
        REFLECT_FWD_LIKE(T, _8),
        REFLECT_FWD_LIKE(T, _9),
        REFLECT_FWD_LIKE(T, _10),
        REFLECT_FWD_LIKE(T, _11),
        REFLECT_FWD_LIKE(T, _12),
        REFLECT_FWD_LIKE(T, _13));
}
template <class Fn, class T>
[[nodiscard]] constexpr decltype(auto) visit(Fn&& fn, T&& t, std::integral_constant<std::size_t, 14>) noexcept {
    auto&& [_1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14] = REFLECT_FWD(t);
    return REFLECT_FWD(fn)(
        REFLECT_FWD_LIKE(T, _1),
        REFLECT_FWD_LIKE(T, _2),
        REFLECT_FWD_LIKE(T, _3),
        REFLECT_FWD_LIKE(T, _4),
        REFLECT_FWD_LIKE(T, _5),
        REFLECT_FWD_LIKE(T, _6),
        REFLECT_FWD_LIKE(T, _7),
        REFLECT_FWD_LIKE(T, _8),
        REFLECT_FWD_LIKE(T, _9),
        REFLECT_FWD_LIKE(T, _10),
        REFLECT_FWD_LIKE(T, _11),
        REFLECT_FWD_LIKE(T, _12),
        REFLECT_FWD_LIKE(T, _13),
        REFLECT_FWD_LIKE(T, _14));
}
template <class Fn, class T>
[[nodiscard]] constexpr decltype(auto) visit(Fn&& fn, T&& t, std::integral_constant<std::size_t, 15>) noexcept {
    auto&& [_1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15] = REFLECT_FWD(t);
    return REFLECT_FWD(fn)(
        REFLECT_FWD_LIKE(T, _1),
        REFLECT_FWD_LIKE(T, _2),
        REFLECT_FWD_LIKE(T, _3),
        REFLECT_FWD_LIKE(T, _4),
        REFLECT_FWD_LIKE(T, _5),
        REFLECT_FWD_LIKE(T, _6),
        REFLECT_FWD_LIKE(T, _7),
        REFLECT_FWD_LIKE(T, _8),
        REFLECT_FWD_LIKE(T, _9),
        REFLECT_FWD_LIKE(T, _10),
        REFLECT_FWD_LIKE(T, _11),
        REFLECT_FWD_LIKE(T, _12),
        REFLECT_FWD_LIKE(T, _13),
        REFLECT_FWD_LIKE(T, _14),
        REFLECT_FWD_LIKE(T, _15));
}
template <class Fn, class T>
[[nodiscard]] constexpr decltype(auto) visit(Fn&& fn, T&& t, std::integral_constant<std::size_t, 16>) noexcept {
    auto&& [_1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16] = REFLECT_FWD(t);
    return REFLECT_FWD(fn)(
        REFLECT_FWD_LIKE(T, _1),
        REFLECT_FWD_LIKE(T, _2),
        REFLECT_FWD_LIKE(T, _3),
        REFLECT_FWD_LIKE(T, _4),
        REFLECT_FWD_LIKE(T, _5),
        REFLECT_FWD_LIKE(T, _6),
        REFLECT_FWD_LIKE(T, _7),
        REFLECT_FWD_LIKE(T, _8),
        REFLECT_FWD_LIKE(T, _9),
        REFLECT_FWD_LIKE(T, _10),
        REFLECT_FWD_LIKE(T, _11),
        REFLECT_FWD_LIKE(T, _12),
        REFLECT_FWD_LIKE(T, _13),
        REFLECT_FWD_LIKE(T, _14),
        REFLECT_FWD_LIKE(T, _15),
        REFLECT_FWD_LIKE(T, _16));
}
template <class Fn, class T>
[[nodiscard]] constexpr decltype(auto) visit(Fn&& fn, T&& t, std::integral_constant<std::size_t, 17>) noexcept {
    auto&& [_1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16, _17] = REFLECT_FWD(t);
    return REFLECT_FWD(fn)(
        REFLECT_FWD_LIKE(T, _1),
        REFLECT_FWD_LIKE(T, _2),
        REFLECT_FWD_LIKE(T, _3),
        REFLECT_FWD_LIKE(T, _4),
        REFLECT_FWD_LIKE(T, _5),
        REFLECT_FWD_LIKE(T, _6),
        REFLECT_FWD_LIKE(T, _7),
        REFLECT_FWD_LIKE(T, _8),
        REFLECT_FWD_LIKE(T, _9),
        REFLECT_FWD_LIKE(T, _10),
        REFLECT_FWD_LIKE(T, _11),
        REFLECT_FWD_LIKE(T, _12),
        REFLECT_FWD_LIKE(T, _13),
        REFLECT_FWD_LIKE(T, _14),
        REFLECT_FWD_LIKE(T, _15),
        REFLECT_FWD_LIKE(T, _16),
        REFLECT_FWD_LIKE(T, _17));
}
template <class Fn, class T>
[[nodiscard]] constexpr decltype(auto) visit(Fn&& fn, T&& t, std::integral_constant<std::size_t, 18>) noexcept {
    auto&& [_1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16, _17, _18] = REFLECT_FWD(t);
    return REFLECT_FWD(fn)(
        REFLECT_FWD_LIKE(T, _1),
        REFLECT_FWD_LIKE(T, _2),
        REFLECT_FWD_LIKE(T, _3),
        REFLECT_FWD_LIKE(T, _4),
        REFLECT_FWD_LIKE(T, _5),
        REFLECT_FWD_LIKE(T, _6),
        REFLECT_FWD_LIKE(T, _7),
        REFLECT_FWD_LIKE(T, _8),
        REFLECT_FWD_LIKE(T, _9),
        REFLECT_FWD_LIKE(T, _10),
        REFLECT_FWD_LIKE(T, _11),
        REFLECT_FWD_LIKE(T, _12),
        REFLECT_FWD_LIKE(T, _13),
        REFLECT_FWD_LIKE(T, _14),
        REFLECT_FWD_LIKE(T, _15),
        REFLECT_FWD_LIKE(T, _16),
        REFLECT_FWD_LIKE(T, _17),
        REFLECT_FWD_LIKE(T, _18));
}
template <class Fn, class T>
[[nodiscard]] constexpr decltype(auto) visit(Fn&& fn, T&& t, std::integral_constant<std::size_t, 19>) noexcept {
    auto&& [_1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16, _17, _18, _19] = REFLECT_FWD(t);
    return REFLECT_FWD(fn)(
        REFLECT_FWD_LIKE(T, _1),
        REFLECT_FWD_LIKE(T, _2),
        REFLECT_FWD_LIKE(T, _3),
        REFLECT_FWD_LIKE(T, _4),
        REFLECT_FWD_LIKE(T, _5),
        REFLECT_FWD_LIKE(T, _6),
        REFLECT_FWD_LIKE(T, _7),
        REFLECT_FWD_LIKE(T, _8),
        REFLECT_FWD_LIKE(T, _9),
        REFLECT_FWD_LIKE(T, _10),
        REFLECT_FWD_LIKE(T, _11),
        REFLECT_FWD_LIKE(T, _12),
        REFLECT_FWD_LIKE(T, _13),
        REFLECT_FWD_LIKE(T, _14),
        REFLECT_FWD_LIKE(T, _15),
        REFLECT_FWD_LIKE(T, _16),
        REFLECT_FWD_LIKE(T, _17),
        REFLECT_FWD_LIKE(T, _18),
        REFLECT_FWD_LIKE(T, _19));
}
template <class Fn, class T>
[[nodiscard]] constexpr decltype(auto) visit(Fn&& fn, T&& t, std::integral_constant<std::size_t, 20>) noexcept {
    auto&& [_1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16, _17, _18, _19, _20] = REFLECT_FWD(t);
    return REFLECT_FWD(fn)(
        REFLECT_FWD_LIKE(T, _1),
        REFLECT_FWD_LIKE(T, _2),
        REFLECT_FWD_LIKE(T, _3),
        REFLECT_FWD_LIKE(T, _4),
        REFLECT_FWD_LIKE(T, _5),
        REFLECT_FWD_LIKE(T, _6),
        REFLECT_FWD_LIKE(T, _7),
        REFLECT_FWD_LIKE(T, _8),
        REFLECT_FWD_LIKE(T, _9),
        REFLECT_FWD_LIKE(T, _10),
        REFLECT_FWD_LIKE(T, _11),
        REFLECT_FWD_LIKE(T, _12),
        REFLECT_FWD_LIKE(T, _13),
        REFLECT_FWD_LIKE(T, _14),
        REFLECT_FWD_LIKE(T, _15),
        REFLECT_FWD_LIKE(T, _16),
        REFLECT_FWD_LIKE(T, _17),
        REFLECT_FWD_LIKE(T, _18),
        REFLECT_FWD_LIKE(T, _19),
        REFLECT_FWD_LIKE(T, _20));
}
template <class Fn, class T>
[[nodiscard]] constexpr decltype(auto) visit(Fn&& fn, T&& t, std::integral_constant<std::size_t, 21>) noexcept {
    auto&& [_1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16, _17, _18, _19, _20, _21] =
        REFLECT_FWD(t);
    return REFLECT_FWD(fn)(
        REFLECT_FWD_LIKE(T, _1),
        REFLECT_FWD_LIKE(T, _2),
        REFLECT_FWD_LIKE(T, _3),
        REFLECT_FWD_LIKE(T, _4),
        REFLECT_FWD_LIKE(T, _5),
        REFLECT_FWD_LIKE(T, _6),
        REFLECT_FWD_LIKE(T, _7),
        REFLECT_FWD_LIKE(T, _8),
        REFLECT_FWD_LIKE(T, _9),
        REFLECT_FWD_LIKE(T, _10),
        REFLECT_FWD_LIKE(T, _11),
        REFLECT_FWD_LIKE(T, _12),
        REFLECT_FWD_LIKE(T, _13),
        REFLECT_FWD_LIKE(T, _14),
        REFLECT_FWD_LIKE(T, _15),
        REFLECT_FWD_LIKE(T, _16),
        REFLECT_FWD_LIKE(T, _17),
        REFLECT_FWD_LIKE(T, _18),
        REFLECT_FWD_LIKE(T, _19),
        REFLECT_FWD_LIKE(T, _20),
        REFLECT_FWD_LIKE(T, _21));
}
template <class Fn, class T>
[[nodiscard]] constexpr decltype(auto) visit(Fn&& fn, T&& t, std::integral_constant<std::size_t, 22>) noexcept {
    auto&& [_1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16, _17, _18, _19, _20, _21, _22] =
        REFLECT_FWD(t);
    return REFLECT_FWD(fn)(
        REFLECT_FWD_LIKE(T, _1),
        REFLECT_FWD_LIKE(T, _2),
        REFLECT_FWD_LIKE(T, _3),
        REFLECT_FWD_LIKE(T, _4),
        REFLECT_FWD_LIKE(T, _5),
        REFLECT_FWD_LIKE(T, _6),
        REFLECT_FWD_LIKE(T, _7),
        REFLECT_FWD_LIKE(T, _8),
        REFLECT_FWD_LIKE(T, _9),
        REFLECT_FWD_LIKE(T, _10),
        REFLECT_FWD_LIKE(T, _11),
        REFLECT_FWD_LIKE(T, _12),
        REFLECT_FWD_LIKE(T, _13),
        REFLECT_FWD_LIKE(T, _14),
        REFLECT_FWD_LIKE(T, _15),
        REFLECT_FWD_LIKE(T, _16),
        REFLECT_FWD_LIKE(T, _17),
        REFLECT_FWD_LIKE(T, _18),
        REFLECT_FWD_LIKE(T, _19),
        REFLECT_FWD_LIKE(T, _20),
        REFLECT_FWD_LIKE(T, _21),
        REFLECT_FWD_LIKE(T, _22));
}
template <class Fn, class T>
[[nodiscard]] constexpr decltype(auto) visit(Fn&& fn, T&& t, std::integral_constant<std::size_t, 23>) noexcept {
    auto&& [_1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16, _17, _18, _19, _20, _21, _22, _23] =
        REFLECT_FWD(t);
    return REFLECT_FWD(fn)(
        REFLECT_FWD_LIKE(T, _1),
        REFLECT_FWD_LIKE(T, _2),
        REFLECT_FWD_LIKE(T, _3),
        REFLECT_FWD_LIKE(T, _4),
        REFLECT_FWD_LIKE(T, _5),
        REFLECT_FWD_LIKE(T, _6),
        REFLECT_FWD_LIKE(T, _7),
        REFLECT_FWD_LIKE(T, _8),
        REFLECT_FWD_LIKE(T, _9),
        REFLECT_FWD_LIKE(T, _10),
        REFLECT_FWD_LIKE(T, _11),
        REFLECT_FWD_LIKE(T, _12),
        REFLECT_FWD_LIKE(T, _13),
        REFLECT_FWD_LIKE(T, _14),
        REFLECT_FWD_LIKE(T, _15),
        REFLECT_FWD_LIKE(T, _16),
        REFLECT_FWD_LIKE(T, _17),
        REFLECT_FWD_LIKE(T, _18),
        REFLECT_FWD_LIKE(T, _19),
        REFLECT_FWD_LIKE(T, _20),
        REFLECT_FWD_LIKE(T, _21),
        REFLECT_FWD_LIKE(T, _22),
        REFLECT_FWD_LIKE(T, _23));
}
template <class Fn, class T>
[[nodiscard]] constexpr decltype(auto) visit(Fn&& fn, T&& t, std::integral_constant<std::size_t, 24>) noexcept {
    auto&& [_1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16, _17, _18, _19, _20, _21, _22, _23, _24] =
        REFLECT_FWD(t);
    return REFLECT_FWD(fn)(
        REFLECT_FWD_LIKE(T, _1),
        REFLECT_FWD_LIKE(T, _2),
        REFLECT_FWD_LIKE(T, _3),
        REFLECT_FWD_LIKE(T, _4),
        REFLECT_FWD_LIKE(T, _5),
        REFLECT_FWD_LIKE(T, _6),
        REFLECT_FWD_LIKE(T, _7),
        REFLECT_FWD_LIKE(T, _8),
        REFLECT_FWD_LIKE(T, _9),
        REFLECT_FWD_LIKE(T, _10),
        REFLECT_FWD_LIKE(T, _11),
        REFLECT_FWD_LIKE(T, _12),
        REFLECT_FWD_LIKE(T, _13),
        REFLECT_FWD_LIKE(T, _14),
        REFLECT_FWD_LIKE(T, _15),
        REFLECT_FWD_LIKE(T, _16),
        REFLECT_FWD_LIKE(T, _17),
        REFLECT_FWD_LIKE(T, _18),
        REFLECT_FWD_LIKE(T, _19),
        REFLECT_FWD_LIKE(T, _20),
        REFLECT_FWD_LIKE(T, _21),
        REFLECT_FWD_LIKE(T, _22),
        REFLECT_FWD_LIKE(T, _23),
        REFLECT_FWD_LIKE(T, _24));
}
template <class Fn, class T>
[[nodiscard]] constexpr decltype(auto) visit(Fn&& fn, T&& t, std::integral_constant<std::size_t, 25>) noexcept {
    auto&& [_1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16, _17, _18, _19, _20, _21, _22, _23, _24, _25] =
        REFLECT_FWD(t);
    return REFLECT_FWD(fn)(
        REFLECT_FWD_LIKE(T, _1),
        REFLECT_FWD_LIKE(T, _2),
        REFLECT_FWD_LIKE(T, _3),
        REFLECT_FWD_LIKE(T, _4),
        REFLECT_FWD_LIKE(T, _5),
        REFLECT_FWD_LIKE(T, _6),
        REFLECT_FWD_LIKE(T, _7),
        REFLECT_FWD_LIKE(T, _8),
        REFLECT_FWD_LIKE(T, _9),
        REFLECT_FWD_LIKE(T, _10),
        REFLECT_FWD_LIKE(T, _11),
        REFLECT_FWD_LIKE(T, _12),
        REFLECT_FWD_LIKE(T, _13),
        REFLECT_FWD_LIKE(T, _14),
        REFLECT_FWD_LIKE(T, _15),
        REFLECT_FWD_LIKE(T, _16),
        REFLECT_FWD_LIKE(T, _17),
        REFLECT_FWD_LIKE(T, _18),
        REFLECT_FWD_LIKE(T, _19),
        REFLECT_FWD_LIKE(T, _20),
        REFLECT_FWD_LIKE(T, _21),
        REFLECT_FWD_LIKE(T, _22),
        REFLECT_FWD_LIKE(T, _23),
        REFLECT_FWD_LIKE(T, _24),
        REFLECT_FWD_LIKE(T, _25));
}
template <class Fn, class T>
[[nodiscard]] constexpr decltype(auto) visit(Fn&& fn, T&& t, std::integral_constant<std::size_t, 26>) noexcept {
    auto&& [_1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16, _17, _18, _19, _20, _21, _22, _23, _24, _25, _26] =
        REFLECT_FWD(t);
    return REFLECT_FWD(fn)(
        REFLECT_FWD_LIKE(T, _1),
        REFLECT_FWD_LIKE(T, _2),
        REFLECT_FWD_LIKE(T, _3),
        REFLECT_FWD_LIKE(T, _4),
        REFLECT_FWD_LIKE(T, _5),
        REFLECT_FWD_LIKE(T, _6),
        REFLECT_FWD_LIKE(T, _7),
        REFLECT_FWD_LIKE(T, _8),
        REFLECT_FWD_LIKE(T, _9),
        REFLECT_FWD_LIKE(T, _10),
        REFLECT_FWD_LIKE(T, _11),
        REFLECT_FWD_LIKE(T, _12),
        REFLECT_FWD_LIKE(T, _13),
        REFLECT_FWD_LIKE(T, _14),
        REFLECT_FWD_LIKE(T, _15),
        REFLECT_FWD_LIKE(T, _16),
        REFLECT_FWD_LIKE(T, _17),
        REFLECT_FWD_LIKE(T, _18),
        REFLECT_FWD_LIKE(T, _19),
        REFLECT_FWD_LIKE(T, _20),
        REFLECT_FWD_LIKE(T, _21),
        REFLECT_FWD_LIKE(T, _22),
        REFLECT_FWD_LIKE(T, _23),
        REFLECT_FWD_LIKE(T, _24),
        REFLECT_FWD_LIKE(T, _25),
        REFLECT_FWD_LIKE(T, _26));
}
template <class Fn, class T>
[[nodiscard]] constexpr decltype(auto) visit(Fn&& fn, T&& t, std::integral_constant<std::size_t, 27>) noexcept {
    auto&& [_1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16, _17, _18, _19, _20, _21, _22, _23, _24, _25, _26, _27] =
        REFLECT_FWD(t);
    return REFLECT_FWD(fn)(
        REFLECT_FWD_LIKE(T, _1),
        REFLECT_FWD_LIKE(T, _2),
        REFLECT_FWD_LIKE(T, _3),
        REFLECT_FWD_LIKE(T, _4),
        REFLECT_FWD_LIKE(T, _5),
        REFLECT_FWD_LIKE(T, _6),
        REFLECT_FWD_LIKE(T, _7),
        REFLECT_FWD_LIKE(T, _8),
        REFLECT_FWD_LIKE(T, _9),
        REFLECT_FWD_LIKE(T, _10),
        REFLECT_FWD_LIKE(T, _11),
        REFLECT_FWD_LIKE(T, _12),
        REFLECT_FWD_LIKE(T, _13),
        REFLECT_FWD_LIKE(T, _14),
        REFLECT_FWD_LIKE(T, _15),
        REFLECT_FWD_LIKE(T, _16),
        REFLECT_FWD_LIKE(T, _17),
        REFLECT_FWD_LIKE(T, _18),
        REFLECT_FWD_LIKE(T, _19),
        REFLECT_FWD_LIKE(T, _20),
        REFLECT_FWD_LIKE(T, _21),
        REFLECT_FWD_LIKE(T, _22),
        REFLECT_FWD_LIKE(T, _23),
        REFLECT_FWD_LIKE(T, _24),
        REFLECT_FWD_LIKE(T, _25),
        REFLECT_FWD_LIKE(T, _26),
        REFLECT_FWD_LIKE(T, _27));
}
template <class Fn, class T>
[[nodiscard]] constexpr decltype(auto) visit(Fn&& fn, T&& t, std::integral_constant<std::size_t, 28>) noexcept {
    auto&& [_1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16, _17, _18, _19, _20, _21, _22, _23, _24, _25, _26, _27, _28] =
        REFLECT_FWD(t);
    return REFLECT_FWD(fn)(
        REFLECT_FWD_LIKE(T, _1),
        REFLECT_FWD_LIKE(T, _2),
        REFLECT_FWD_LIKE(T, _3),
        REFLECT_FWD_LIKE(T, _4),
        REFLECT_FWD_LIKE(T, _5),
        REFLECT_FWD_LIKE(T, _6),
        REFLECT_FWD_LIKE(T, _7),
        REFLECT_FWD_LIKE(T, _8),
        REFLECT_FWD_LIKE(T, _9),
        REFLECT_FWD_LIKE(T, _10),
        REFLECT_FWD_LIKE(T, _11),
        REFLECT_FWD_LIKE(T, _12),
        REFLECT_FWD_LIKE(T, _13),
        REFLECT_FWD_LIKE(T, _14),
        REFLECT_FWD_LIKE(T, _15),
        REFLECT_FWD_LIKE(T, _16),
        REFLECT_FWD_LIKE(T, _17),
        REFLECT_FWD_LIKE(T, _18),
        REFLECT_FWD_LIKE(T, _19),
        REFLECT_FWD_LIKE(T, _20),
        REFLECT_FWD_LIKE(T, _21),
        REFLECT_FWD_LIKE(T, _22),
        REFLECT_FWD_LIKE(T, _23),
        REFLECT_FWD_LIKE(T, _24),
        REFLECT_FWD_LIKE(T, _25),
        REFLECT_FWD_LIKE(T, _26),
        REFLECT_FWD_LIKE(T, _27),
        REFLECT_FWD_LIKE(T, _28));
}
template <class Fn, class T>
[[nodiscard]] constexpr decltype(auto) visit(Fn&& fn, T&& t, std::integral_constant<std::size_t, 29>) noexcept {
    auto&& [_1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16, _17, _18, _19, _20, _21, _22, _23, _24, _25, _26, _27, _28, _29] =
        REFLECT_FWD(t);
    return REFLECT_FWD(fn)(
        REFLECT_FWD_LIKE(T, _1),
        REFLECT_FWD_LIKE(T, _2),
        REFLECT_FWD_LIKE(T, _3),
        REFLECT_FWD_LIKE(T, _4),
        REFLECT_FWD_LIKE(T, _5),
        REFLECT_FWD_LIKE(T, _6),
        REFLECT_FWD_LIKE(T, _7),
        REFLECT_FWD_LIKE(T, _8),
        REFLECT_FWD_LIKE(T, _9),
        REFLECT_FWD_LIKE(T, _10),
        REFLECT_FWD_LIKE(T, _11),
        REFLECT_FWD_LIKE(T, _12),
        REFLECT_FWD_LIKE(T, _13),
        REFLECT_FWD_LIKE(T, _14),
        REFLECT_FWD_LIKE(T, _15),
        REFLECT_FWD_LIKE(T, _16),
        REFLECT_FWD_LIKE(T, _17),
        REFLECT_FWD_LIKE(T, _18),
        REFLECT_FWD_LIKE(T, _19),
        REFLECT_FWD_LIKE(T, _20),
        REFLECT_FWD_LIKE(T, _21),
        REFLECT_FWD_LIKE(T, _22),
        REFLECT_FWD_LIKE(T, _23),
        REFLECT_FWD_LIKE(T, _24),
        REFLECT_FWD_LIKE(T, _25),
        REFLECT_FWD_LIKE(T, _26),
        REFLECT_FWD_LIKE(T, _27),
        REFLECT_FWD_LIKE(T, _28),
        REFLECT_FWD_LIKE(T, _29));
}
template <class Fn, class T>
[[nodiscard]] constexpr decltype(auto) visit(Fn&& fn, T&& t, std::integral_constant<std::size_t, 30>) noexcept {
    auto&& [_1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16, _17, _18, _19, _20, _21, _22, _23, _24, _25, _26, _27, _28, _29, _30] =
        REFLECT_FWD(t);
    return REFLECT_FWD(fn)(
        REFLECT_FWD_LIKE(T, _1),
        REFLECT_FWD_LIKE(T, _2),
        REFLECT_FWD_LIKE(T, _3),
        REFLECT_FWD_LIKE(T, _4),
        REFLECT_FWD_LIKE(T, _5),
        REFLECT_FWD_LIKE(T, _6),
        REFLECT_FWD_LIKE(T, _7),
        REFLECT_FWD_LIKE(T, _8),
        REFLECT_FWD_LIKE(T, _9),
        REFLECT_FWD_LIKE(T, _10),
        REFLECT_FWD_LIKE(T, _11),
        REFLECT_FWD_LIKE(T, _12),
        REFLECT_FWD_LIKE(T, _13),
        REFLECT_FWD_LIKE(T, _14),
        REFLECT_FWD_LIKE(T, _15),
        REFLECT_FWD_LIKE(T, _16),
        REFLECT_FWD_LIKE(T, _17),
        REFLECT_FWD_LIKE(T, _18),
        REFLECT_FWD_LIKE(T, _19),
        REFLECT_FWD_LIKE(T, _20),
        REFLECT_FWD_LIKE(T, _21),
        REFLECT_FWD_LIKE(T, _22),
        REFLECT_FWD_LIKE(T, _23),
        REFLECT_FWD_LIKE(T, _24),
        REFLECT_FWD_LIKE(T, _25),
        REFLECT_FWD_LIKE(T, _26),
        REFLECT_FWD_LIKE(T, _27),
        REFLECT_FWD_LIKE(T, _28),
        REFLECT_FWD_LIKE(T, _29),
        REFLECT_FWD_LIKE(T, _30));
}
template <class Fn, class T>
[[nodiscard]] constexpr decltype(auto) visit(Fn&& fn, T&& t, std::integral_constant<std::size_t, 31>) noexcept {
    auto&& [_1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16, _17, _18, _19, _20, _21, _22, _23, _24, _25, _26, _27, _28, _29, _30, _31] =
        REFLECT_FWD(t);
    return REFLECT_FWD(fn)(
        REFLECT_FWD_LIKE(T, _1),
        REFLECT_FWD_LIKE(T, _2),
        REFLECT_FWD_LIKE(T, _3),
        REFLECT_FWD_LIKE(T, _4),
        REFLECT_FWD_LIKE(T, _5),
        REFLECT_FWD_LIKE(T, _6),
        REFLECT_FWD_LIKE(T, _7),
        REFLECT_FWD_LIKE(T, _8),
        REFLECT_FWD_LIKE(T, _9),
        REFLECT_FWD_LIKE(T, _10),
        REFLECT_FWD_LIKE(T, _11),
        REFLECT_FWD_LIKE(T, _12),
        REFLECT_FWD_LIKE(T, _13),
        REFLECT_FWD_LIKE(T, _14),
        REFLECT_FWD_LIKE(T, _15),
        REFLECT_FWD_LIKE(T, _16),
        REFLECT_FWD_LIKE(T, _17),
        REFLECT_FWD_LIKE(T, _18),
        REFLECT_FWD_LIKE(T, _19),
        REFLECT_FWD_LIKE(T, _20),
        REFLECT_FWD_LIKE(T, _21),
        REFLECT_FWD_LIKE(T, _22),
        REFLECT_FWD_LIKE(T, _23),
        REFLECT_FWD_LIKE(T, _24),
        REFLECT_FWD_LIKE(T, _25),
        REFLECT_FWD_LIKE(T, _26),
        REFLECT_FWD_LIKE(T, _27),
        REFLECT_FWD_LIKE(T, _28),
        REFLECT_FWD_LIKE(T, _29),
        REFLECT_FWD_LIKE(T, _30),
        REFLECT_FWD_LIKE(T, _31));
}
template <class Fn, class T>
[[nodiscard]] constexpr decltype(auto) visit(Fn&& fn, T&& t, std::integral_constant<std::size_t, 32>) noexcept {
    auto&& [_1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16, _17, _18, _19, _20, _21, _22, _23, _24, _25, _26, _27, _28, _29, _30, _31, _32] =
        REFLECT_FWD(t);
    return REFLECT_FWD(fn)(
        REFLECT_FWD_LIKE(T, _1),
        REFLECT_FWD_LIKE(T, _2),
        REFLECT_FWD_LIKE(T, _3),
        REFLECT_FWD_LIKE(T, _4),
        REFLECT_FWD_LIKE(T, _5),
        REFLECT_FWD_LIKE(T, _6),
        REFLECT_FWD_LIKE(T, _7),
        REFLECT_FWD_LIKE(T, _8),
        REFLECT_FWD_LIKE(T, _9),
        REFLECT_FWD_LIKE(T, _10),
        REFLECT_FWD_LIKE(T, _11),
        REFLECT_FWD_LIKE(T, _12),
        REFLECT_FWD_LIKE(T, _13),
        REFLECT_FWD_LIKE(T, _14),
        REFLECT_FWD_LIKE(T, _15),
        REFLECT_FWD_LIKE(T, _16),
        REFLECT_FWD_LIKE(T, _17),
        REFLECT_FWD_LIKE(T, _18),
        REFLECT_FWD_LIKE(T, _19),
        REFLECT_FWD_LIKE(T, _20),
        REFLECT_FWD_LIKE(T, _21),
        REFLECT_FWD_LIKE(T, _22),
        REFLECT_FWD_LIKE(T, _23),
        REFLECT_FWD_LIKE(T, _24),
        REFLECT_FWD_LIKE(T, _25),
        REFLECT_FWD_LIKE(T, _26),
        REFLECT_FWD_LIKE(T, _27),
        REFLECT_FWD_LIKE(T, _28),
        REFLECT_FWD_LIKE(T, _29),
        REFLECT_FWD_LIKE(T, _30),
        REFLECT_FWD_LIKE(T, _31),
        REFLECT_FWD_LIKE(T, _32));
}
template <class Fn, class T>
[[nodiscard]] constexpr decltype(auto) visit(Fn&& fn, T&& t, std::integral_constant<std::size_t, 33>) noexcept {
    auto&& [_1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16, _17, _18, _19, _20, _21, _22, _23, _24, _25, _26, _27, _28, _29, _30, _31, _32, _33] =
        REFLECT_FWD(t);
    return REFLECT_FWD(fn)(
        REFLECT_FWD_LIKE(T, _1),
        REFLECT_FWD_LIKE(T, _2),
        REFLECT_FWD_LIKE(T, _3),
        REFLECT_FWD_LIKE(T, _4),
        REFLECT_FWD_LIKE(T, _5),
        REFLECT_FWD_LIKE(T, _6),
        REFLECT_FWD_LIKE(T, _7),
        REFLECT_FWD_LIKE(T, _8),
        REFLECT_FWD_LIKE(T, _9),
        REFLECT_FWD_LIKE(T, _10),
        REFLECT_FWD_LIKE(T, _11),
        REFLECT_FWD_LIKE(T, _12),
        REFLECT_FWD_LIKE(T, _13),
        REFLECT_FWD_LIKE(T, _14),
        REFLECT_FWD_LIKE(T, _15),
        REFLECT_FWD_LIKE(T, _16),
        REFLECT_FWD_LIKE(T, _17),
        REFLECT_FWD_LIKE(T, _18),
        REFLECT_FWD_LIKE(T, _19),
        REFLECT_FWD_LIKE(T, _20),
        REFLECT_FWD_LIKE(T, _21),
        REFLECT_FWD_LIKE(T, _22),
        REFLECT_FWD_LIKE(T, _23),
        REFLECT_FWD_LIKE(T, _24),
        REFLECT_FWD_LIKE(T, _25),
        REFLECT_FWD_LIKE(T, _26),
        REFLECT_FWD_LIKE(T, _27),
        REFLECT_FWD_LIKE(T, _28),
        REFLECT_FWD_LIKE(T, _29),
        REFLECT_FWD_LIKE(T, _30),
        REFLECT_FWD_LIKE(T, _31),
        REFLECT_FWD_LIKE(T, _32),
        REFLECT_FWD_LIKE(T, _33));
}
template <class Fn, class T>
[[nodiscard]] constexpr decltype(auto) visit(Fn&& fn, T&& t, std::integral_constant<std::size_t, 34>) noexcept {
    auto&& [_1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16, _17, _18, _19, _20, _21, _22, _23, _24, _25, _26, _27, _28, _29, _30, _31, _32, _33, _34] =
        REFLECT_FWD(t);
    return REFLECT_FWD(fn)(
        REFLECT_FWD_LIKE(T, _1),
        REFLECT_FWD_LIKE(T, _2),
        REFLECT_FWD_LIKE(T, _3),
        REFLECT_FWD_LIKE(T, _4),
        REFLECT_FWD_LIKE(T, _5),
        REFLECT_FWD_LIKE(T, _6),
        REFLECT_FWD_LIKE(T, _7),
        REFLECT_FWD_LIKE(T, _8),
        REFLECT_FWD_LIKE(T, _9),
        REFLECT_FWD_LIKE(T, _10),
        REFLECT_FWD_LIKE(T, _11),
        REFLECT_FWD_LIKE(T, _12),
        REFLECT_FWD_LIKE(T, _13),
        REFLECT_FWD_LIKE(T, _14),
        REFLECT_FWD_LIKE(T, _15),
        REFLECT_FWD_LIKE(T, _16),
        REFLECT_FWD_LIKE(T, _17),
        REFLECT_FWD_LIKE(T, _18),
        REFLECT_FWD_LIKE(T, _19),
        REFLECT_FWD_LIKE(T, _20),
        REFLECT_FWD_LIKE(T, _21),
        REFLECT_FWD_LIKE(T, _22),
        REFLECT_FWD_LIKE(T, _23),
        REFLECT_FWD_LIKE(T, _24),
        REFLECT_FWD_LIKE(T, _25),
        REFLECT_FWD_LIKE(T, _26),
        REFLECT_FWD_LIKE(T, _27),
        REFLECT_FWD_LIKE(T, _28),
        REFLECT_FWD_LIKE(T, _29),
        REFLECT_FWD_LIKE(T, _30),
        REFLECT_FWD_LIKE(T, _31),
        REFLECT_FWD_LIKE(T, _32),
        REFLECT_FWD_LIKE(T, _33),
        REFLECT_FWD_LIKE(T, _34));
}
template <class Fn, class T>
[[nodiscard]] constexpr decltype(auto) visit(Fn&& fn, T&& t, std::integral_constant<std::size_t, 35>) noexcept {
    auto&& [_1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16, _17, _18, _19, _20, _21, _22, _23, _24, _25, _26, _27, _28, _29, _30, _31, _32, _33, _34, _35] =
        REFLECT_FWD(t);
    return REFLECT_FWD(fn)(
        REFLECT_FWD_LIKE(T, _1),
        REFLECT_FWD_LIKE(T, _2),
        REFLECT_FWD_LIKE(T, _3),
        REFLECT_FWD_LIKE(T, _4),
        REFLECT_FWD_LIKE(T, _5),
        REFLECT_FWD_LIKE(T, _6),
        REFLECT_FWD_LIKE(T, _7),
        REFLECT_FWD_LIKE(T, _8),
        REFLECT_FWD_LIKE(T, _9),
        REFLECT_FWD_LIKE(T, _10),
        REFLECT_FWD_LIKE(T, _11),
        REFLECT_FWD_LIKE(T, _12),
        REFLECT_FWD_LIKE(T, _13),
        REFLECT_FWD_LIKE(T, _14),
        REFLECT_FWD_LIKE(T, _15),
        REFLECT_FWD_LIKE(T, _16),
        REFLECT_FWD_LIKE(T, _17),
        REFLECT_FWD_LIKE(T, _18),
        REFLECT_FWD_LIKE(T, _19),
        REFLECT_FWD_LIKE(T, _20),
        REFLECT_FWD_LIKE(T, _21),
        REFLECT_FWD_LIKE(T, _22),
        REFLECT_FWD_LIKE(T, _23),
        REFLECT_FWD_LIKE(T, _24),
        REFLECT_FWD_LIKE(T, _25),
        REFLECT_FWD_LIKE(T, _26),
        REFLECT_FWD_LIKE(T, _27),
        REFLECT_FWD_LIKE(T, _28),
        REFLECT_FWD_LIKE(T, _29),
        REFLECT_FWD_LIKE(T, _30),
        REFLECT_FWD_LIKE(T, _31),
        REFLECT_FWD_LIKE(T, _32),
        REFLECT_FWD_LIKE(T, _33),
        REFLECT_FWD_LIKE(T, _34),
        REFLECT_FWD_LIKE(T, _35));
}
template <class Fn, class T>
[[nodiscard]] constexpr decltype(auto) visit(Fn&& fn, T&& t, std::integral_constant<std::size_t, 36>) noexcept {
    auto&& [_1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16, _17, _18, _19, _20, _21, _22, _23, _24, _25, _26, _27, _28, _29, _30, _31, _32, _33, _34, _35, _36] =
        REFLECT_FWD(t);
    return REFLECT_FWD(fn)(
        REFLECT_FWD_LIKE(T, _1),
        REFLECT_FWD_LIKE(T, _2),
        REFLECT_FWD_LIKE(T, _3),
        REFLECT_FWD_LIKE(T, _4),
        REFLECT_FWD_LIKE(T, _5),
        REFLECT_FWD_LIKE(T, _6),
        REFLECT_FWD_LIKE(T, _7),
        REFLECT_FWD_LIKE(T, _8),
        REFLECT_FWD_LIKE(T, _9),
        REFLECT_FWD_LIKE(T, _10),
        REFLECT_FWD_LIKE(T, _11),
        REFLECT_FWD_LIKE(T, _12),
        REFLECT_FWD_LIKE(T, _13),
        REFLECT_FWD_LIKE(T, _14),
        REFLECT_FWD_LIKE(T, _15),
        REFLECT_FWD_LIKE(T, _16),
        REFLECT_FWD_LIKE(T, _17),
        REFLECT_FWD_LIKE(T, _18),
        REFLECT_FWD_LIKE(T, _19),
        REFLECT_FWD_LIKE(T, _20),
        REFLECT_FWD_LIKE(T, _21),
        REFLECT_FWD_LIKE(T, _22),
        REFLECT_FWD_LIKE(T, _23),
        REFLECT_FWD_LIKE(T, _24),
        REFLECT_FWD_LIKE(T, _25),
        REFLECT_FWD_LIKE(T, _26),
        REFLECT_FWD_LIKE(T, _27),
        REFLECT_FWD_LIKE(T, _28),
        REFLECT_FWD_LIKE(T, _29),
        REFLECT_FWD_LIKE(T, _30),
        REFLECT_FWD_LIKE(T, _31),
        REFLECT_FWD_LIKE(T, _32),
        REFLECT_FWD_LIKE(T, _33),
        REFLECT_FWD_LIKE(T, _34),
        REFLECT_FWD_LIKE(T, _35),
        REFLECT_FWD_LIKE(T, _36));
}
template <class Fn, class T>
[[nodiscard]] constexpr decltype(auto) visit(Fn&& fn, T&& t, std::integral_constant<std::size_t, 37>) noexcept {
    auto&& [_1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16, _17, _18, _19, _20, _21, _22, _23, _24, _25, _26, _27, _28, _29, _30, _31, _32, _33, _34, _35, _36, _37] =
        REFLECT_FWD(t);
    return REFLECT_FWD(fn)(
        REFLECT_FWD_LIKE(T, _1),
        REFLECT_FWD_LIKE(T, _2),
        REFLECT_FWD_LIKE(T, _3),
        REFLECT_FWD_LIKE(T, _4),
        REFLECT_FWD_LIKE(T, _5),
        REFLECT_FWD_LIKE(T, _6),
        REFLECT_FWD_LIKE(T, _7),
        REFLECT_FWD_LIKE(T, _8),
        REFLECT_FWD_LIKE(T, _9),
        REFLECT_FWD_LIKE(T, _10),
        REFLECT_FWD_LIKE(T, _11),
        REFLECT_FWD_LIKE(T, _12),
        REFLECT_FWD_LIKE(T, _13),
        REFLECT_FWD_LIKE(T, _14),
        REFLECT_FWD_LIKE(T, _15),
        REFLECT_FWD_LIKE(T, _16),
        REFLECT_FWD_LIKE(T, _17),
        REFLECT_FWD_LIKE(T, _18),
        REFLECT_FWD_LIKE(T, _19),
        REFLECT_FWD_LIKE(T, _20),
        REFLECT_FWD_LIKE(T, _21),
        REFLECT_FWD_LIKE(T, _22),
        REFLECT_FWD_LIKE(T, _23),
        REFLECT_FWD_LIKE(T, _24),
        REFLECT_FWD_LIKE(T, _25),
        REFLECT_FWD_LIKE(T, _26),
        REFLECT_FWD_LIKE(T, _27),
        REFLECT_FWD_LIKE(T, _28),
        REFLECT_FWD_LIKE(T, _29),
        REFLECT_FWD_LIKE(T, _30),
        REFLECT_FWD_LIKE(T, _31),
        REFLECT_FWD_LIKE(T, _32),
        REFLECT_FWD_LIKE(T, _33),
        REFLECT_FWD_LIKE(T, _34),
        REFLECT_FWD_LIKE(T, _35),
        REFLECT_FWD_LIKE(T, _36),
        REFLECT_FWD_LIKE(T, _37));
}
template <class Fn, class T>
[[nodiscard]] constexpr decltype(auto) visit(Fn&& fn, T&& t, std::integral_constant<std::size_t, 38>) noexcept {
    auto&& [_1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16, _17, _18, _19, _20, _21, _22, _23, _24, _25, _26, _27, _28, _29, _30, _31, _32, _33, _34, _35, _36, _37, _38] =
        REFLECT_FWD(t);
    return REFLECT_FWD(fn)(
        REFLECT_FWD_LIKE(T, _1),
        REFLECT_FWD_LIKE(T, _2),
        REFLECT_FWD_LIKE(T, _3),
        REFLECT_FWD_LIKE(T, _4),
        REFLECT_FWD_LIKE(T, _5),
        REFLECT_FWD_LIKE(T, _6),
        REFLECT_FWD_LIKE(T, _7),
        REFLECT_FWD_LIKE(T, _8),
        REFLECT_FWD_LIKE(T, _9),
        REFLECT_FWD_LIKE(T, _10),
        REFLECT_FWD_LIKE(T, _11),
        REFLECT_FWD_LIKE(T, _12),
        REFLECT_FWD_LIKE(T, _13),
        REFLECT_FWD_LIKE(T, _14),
        REFLECT_FWD_LIKE(T, _15),
        REFLECT_FWD_LIKE(T, _16),
        REFLECT_FWD_LIKE(T, _17),
        REFLECT_FWD_LIKE(T, _18),
        REFLECT_FWD_LIKE(T, _19),
        REFLECT_FWD_LIKE(T, _20),
        REFLECT_FWD_LIKE(T, _21),
        REFLECT_FWD_LIKE(T, _22),
        REFLECT_FWD_LIKE(T, _23),
        REFLECT_FWD_LIKE(T, _24),
        REFLECT_FWD_LIKE(T, _25),
        REFLECT_FWD_LIKE(T, _26),
        REFLECT_FWD_LIKE(T, _27),
        REFLECT_FWD_LIKE(T, _28),
        REFLECT_FWD_LIKE(T, _29),
        REFLECT_FWD_LIKE(T, _30),
        REFLECT_FWD_LIKE(T, _31),
        REFLECT_FWD_LIKE(T, _32),
        REFLECT_FWD_LIKE(T, _33),
        REFLECT_FWD_LIKE(T, _34),
        REFLECT_FWD_LIKE(T, _35),
        REFLECT_FWD_LIKE(T, _36),
        REFLECT_FWD_LIKE(T, _37),
        REFLECT_FWD_LIKE(T, _38));
}
template <class Fn, class T>
[[nodiscard]] constexpr decltype(auto) visit(Fn&& fn, T&& t, std::integral_constant<std::size_t, 39>) noexcept {
    auto&& [_1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16, _17, _18, _19, _20, _21, _22, _23, _24, _25, _26, _27, _28, _29, _30, _31, _32, _33, _34, _35, _36, _37, _38, _39] =
        REFLECT_FWD(t);
    return REFLECT_FWD(fn)(
        REFLECT_FWD_LIKE(T, _1),
        REFLECT_FWD_LIKE(T, _2),
        REFLECT_FWD_LIKE(T, _3),
        REFLECT_FWD_LIKE(T, _4),
        REFLECT_FWD_LIKE(T, _5),
        REFLECT_FWD_LIKE(T, _6),
        REFLECT_FWD_LIKE(T, _7),
        REFLECT_FWD_LIKE(T, _8),
        REFLECT_FWD_LIKE(T, _9),
        REFLECT_FWD_LIKE(T, _10),
        REFLECT_FWD_LIKE(T, _11),
        REFLECT_FWD_LIKE(T, _12),
        REFLECT_FWD_LIKE(T, _13),
        REFLECT_FWD_LIKE(T, _14),
        REFLECT_FWD_LIKE(T, _15),
        REFLECT_FWD_LIKE(T, _16),
        REFLECT_FWD_LIKE(T, _17),
        REFLECT_FWD_LIKE(T, _18),
        REFLECT_FWD_LIKE(T, _19),
        REFLECT_FWD_LIKE(T, _20),
        REFLECT_FWD_LIKE(T, _21),
        REFLECT_FWD_LIKE(T, _22),
        REFLECT_FWD_LIKE(T, _23),
        REFLECT_FWD_LIKE(T, _24),
        REFLECT_FWD_LIKE(T, _25),
        REFLECT_FWD_LIKE(T, _26),
        REFLECT_FWD_LIKE(T, _27),
        REFLECT_FWD_LIKE(T, _28),
        REFLECT_FWD_LIKE(T, _29),
        REFLECT_FWD_LIKE(T, _30),
        REFLECT_FWD_LIKE(T, _31),
        REFLECT_FWD_LIKE(T, _32),
        REFLECT_FWD_LIKE(T, _33),
        REFLECT_FWD_LIKE(T, _34),
        REFLECT_FWD_LIKE(T, _35),
        REFLECT_FWD_LIKE(T, _36),
        REFLECT_FWD_LIKE(T, _37),
        REFLECT_FWD_LIKE(T, _38),
        REFLECT_FWD_LIKE(T, _39));
}
template <class Fn, class T>
[[nodiscard]] constexpr decltype(auto) visit(Fn&& fn, T&& t, std::integral_constant<std::size_t, 40>) noexcept {
    auto&& [_1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16, _17, _18, _19, _20, _21, _22, _23, _24, _25, _26, _27, _28, _29, _30, _31, _32, _33, _34, _35, _36, _37, _38, _39, _40] =
        REFLECT_FWD(t);
    return REFLECT_FWD(fn)(
        REFLECT_FWD_LIKE(T, _1),
        REFLECT_FWD_LIKE(T, _2),
        REFLECT_FWD_LIKE(T, _3),
        REFLECT_FWD_LIKE(T, _4),
        REFLECT_FWD_LIKE(T, _5),
        REFLECT_FWD_LIKE(T, _6),
        REFLECT_FWD_LIKE(T, _7),
        REFLECT_FWD_LIKE(T, _8),
        REFLECT_FWD_LIKE(T, _9),
        REFLECT_FWD_LIKE(T, _10),
        REFLECT_FWD_LIKE(T, _11),
        REFLECT_FWD_LIKE(T, _12),
        REFLECT_FWD_LIKE(T, _13),
        REFLECT_FWD_LIKE(T, _14),
        REFLECT_FWD_LIKE(T, _15),
        REFLECT_FWD_LIKE(T, _16),
        REFLECT_FWD_LIKE(T, _17),
        REFLECT_FWD_LIKE(T, _18),
        REFLECT_FWD_LIKE(T, _19),
        REFLECT_FWD_LIKE(T, _20),
        REFLECT_FWD_LIKE(T, _21),
        REFLECT_FWD_LIKE(T, _22),
        REFLECT_FWD_LIKE(T, _23),
        REFLECT_FWD_LIKE(T, _24),
        REFLECT_FWD_LIKE(T, _25),
        REFLECT_FWD_LIKE(T, _26),
        REFLECT_FWD_LIKE(T, _27),
        REFLECT_FWD_LIKE(T, _28),
        REFLECT_FWD_LIKE(T, _29),
        REFLECT_FWD_LIKE(T, _30),
        REFLECT_FWD_LIKE(T, _31),
        REFLECT_FWD_LIKE(T, _32),
        REFLECT_FWD_LIKE(T, _33),
        REFLECT_FWD_LIKE(T, _34),
        REFLECT_FWD_LIKE(T, _35),
        REFLECT_FWD_LIKE(T, _36),
        REFLECT_FWD_LIKE(T, _37),
        REFLECT_FWD_LIKE(T, _38),
        REFLECT_FWD_LIKE(T, _39),
        REFLECT_FWD_LIKE(T, _40));
}
template <class Fn, class T>
[[nodiscard]] constexpr decltype(auto) visit(Fn&& fn, T&& t, std::integral_constant<std::size_t, 41>) noexcept {
    auto&& [_1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16, _17, _18, _19, _20, _21, _22, _23, _24, _25, _26, _27, _28, _29, _30, _31, _32, _33, _34, _35, _36, _37, _38, _39, _40, _41] =
        REFLECT_FWD(t);
    return REFLECT_FWD(fn)(
        REFLECT_FWD_LIKE(T, _1),
        REFLECT_FWD_LIKE(T, _2),
        REFLECT_FWD_LIKE(T, _3),
        REFLECT_FWD_LIKE(T, _4),
        REFLECT_FWD_LIKE(T, _5),
        REFLECT_FWD_LIKE(T, _6),
        REFLECT_FWD_LIKE(T, _7),
        REFLECT_FWD_LIKE(T, _8),
        REFLECT_FWD_LIKE(T, _9),
        REFLECT_FWD_LIKE(T, _10),
        REFLECT_FWD_LIKE(T, _11),
        REFLECT_FWD_LIKE(T, _12),
        REFLECT_FWD_LIKE(T, _13),
        REFLECT_FWD_LIKE(T, _14),
        REFLECT_FWD_LIKE(T, _15),
        REFLECT_FWD_LIKE(T, _16),
        REFLECT_FWD_LIKE(T, _17),
        REFLECT_FWD_LIKE(T, _18),
        REFLECT_FWD_LIKE(T, _19),
        REFLECT_FWD_LIKE(T, _20),
        REFLECT_FWD_LIKE(T, _21),
        REFLECT_FWD_LIKE(T, _22),
        REFLECT_FWD_LIKE(T, _23),
        REFLECT_FWD_LIKE(T, _24),
        REFLECT_FWD_LIKE(T, _25),
        REFLECT_FWD_LIKE(T, _26),
        REFLECT_FWD_LIKE(T, _27),
        REFLECT_FWD_LIKE(T, _28),
        REFLECT_FWD_LIKE(T, _29),
        REFLECT_FWD_LIKE(T, _30),
        REFLECT_FWD_LIKE(T, _31),
        REFLECT_FWD_LIKE(T, _32),
        REFLECT_FWD_LIKE(T, _33),
        REFLECT_FWD_LIKE(T, _34),
        REFLECT_FWD_LIKE(T, _35),
        REFLECT_FWD_LIKE(T, _36),
        REFLECT_FWD_LIKE(T, _37),
        REFLECT_FWD_LIKE(T, _38),
        REFLECT_FWD_LIKE(T, _39),
        REFLECT_FWD_LIKE(T, _40),
        REFLECT_FWD_LIKE(T, _41));
}
template <class Fn, class T>
[[nodiscard]] constexpr decltype(auto) visit(Fn&& fn, T&& t, std::integral_constant<std::size_t, 42>) noexcept {
    auto&& [_1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16, _17, _18, _19, _20, _21, _22, _23, _24, _25, _26, _27, _28, _29, _30, _31, _32, _33, _34, _35, _36, _37, _38, _39, _40, _41, _42] =
        REFLECT_FWD(t);
    return REFLECT_FWD(fn)(
        REFLECT_FWD_LIKE(T, _1),
        REFLECT_FWD_LIKE(T, _2),
        REFLECT_FWD_LIKE(T, _3),
        REFLECT_FWD_LIKE(T, _4),
        REFLECT_FWD_LIKE(T, _5),
        REFLECT_FWD_LIKE(T, _6),
        REFLECT_FWD_LIKE(T, _7),
        REFLECT_FWD_LIKE(T, _8),
        REFLECT_FWD_LIKE(T, _9),
        REFLECT_FWD_LIKE(T, _10),
        REFLECT_FWD_LIKE(T, _11),
        REFLECT_FWD_LIKE(T, _12),
        REFLECT_FWD_LIKE(T, _13),
        REFLECT_FWD_LIKE(T, _14),
        REFLECT_FWD_LIKE(T, _15),
        REFLECT_FWD_LIKE(T, _16),
        REFLECT_FWD_LIKE(T, _17),
        REFLECT_FWD_LIKE(T, _18),
        REFLECT_FWD_LIKE(T, _19),
        REFLECT_FWD_LIKE(T, _20),
        REFLECT_FWD_LIKE(T, _21),
        REFLECT_FWD_LIKE(T, _22),
        REFLECT_FWD_LIKE(T, _23),
        REFLECT_FWD_LIKE(T, _24),
        REFLECT_FWD_LIKE(T, _25),
        REFLECT_FWD_LIKE(T, _26),
        REFLECT_FWD_LIKE(T, _27),
        REFLECT_FWD_LIKE(T, _28),
        REFLECT_FWD_LIKE(T, _29),
        REFLECT_FWD_LIKE(T, _30),
        REFLECT_FWD_LIKE(T, _31),
        REFLECT_FWD_LIKE(T, _32),
        REFLECT_FWD_LIKE(T, _33),
        REFLECT_FWD_LIKE(T, _34),
        REFLECT_FWD_LIKE(T, _35),
        REFLECT_FWD_LIKE(T, _36),
        REFLECT_FWD_LIKE(T, _37),
        REFLECT_FWD_LIKE(T, _38),
        REFLECT_FWD_LIKE(T, _39),
        REFLECT_FWD_LIKE(T, _40),
        REFLECT_FWD_LIKE(T, _41),
        REFLECT_FWD_LIKE(T, _42));
}
template <class Fn, class T>
[[nodiscard]] constexpr decltype(auto) visit(Fn&& fn, T&& t, std::integral_constant<std::size_t, 43>) noexcept {
    auto&& [_1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16, _17, _18, _19, _20, _21, _22, _23, _24, _25, _26, _27, _28, _29, _30, _31, _32, _33, _34, _35, _36, _37, _38, _39, _40, _41, _42, _43] =
        REFLECT_FWD(t);
    return REFLECT_FWD(fn)(
        REFLECT_FWD_LIKE(T, _1),
        REFLECT_FWD_LIKE(T, _2),
        REFLECT_FWD_LIKE(T, _3),
        REFLECT_FWD_LIKE(T, _4),
        REFLECT_FWD_LIKE(T, _5),
        REFLECT_FWD_LIKE(T, _6),
        REFLECT_FWD_LIKE(T, _7),
        REFLECT_FWD_LIKE(T, _8),
        REFLECT_FWD_LIKE(T, _9),
        REFLECT_FWD_LIKE(T, _10),
        REFLECT_FWD_LIKE(T, _11),
        REFLECT_FWD_LIKE(T, _12),
        REFLECT_FWD_LIKE(T, _13),
        REFLECT_FWD_LIKE(T, _14),
        REFLECT_FWD_LIKE(T, _15),
        REFLECT_FWD_LIKE(T, _16),
        REFLECT_FWD_LIKE(T, _17),
        REFLECT_FWD_LIKE(T, _18),
        REFLECT_FWD_LIKE(T, _19),
        REFLECT_FWD_LIKE(T, _20),
        REFLECT_FWD_LIKE(T, _21),
        REFLECT_FWD_LIKE(T, _22),
        REFLECT_FWD_LIKE(T, _23),
        REFLECT_FWD_LIKE(T, _24),
        REFLECT_FWD_LIKE(T, _25),
        REFLECT_FWD_LIKE(T, _26),
        REFLECT_FWD_LIKE(T, _27),
        REFLECT_FWD_LIKE(T, _28),
        REFLECT_FWD_LIKE(T, _29),
        REFLECT_FWD_LIKE(T, _30),
        REFLECT_FWD_LIKE(T, _31),
        REFLECT_FWD_LIKE(T, _32),
        REFLECT_FWD_LIKE(T, _33),
        REFLECT_FWD_LIKE(T, _34),
        REFLECT_FWD_LIKE(T, _35),
        REFLECT_FWD_LIKE(T, _36),
        REFLECT_FWD_LIKE(T, _37),
        REFLECT_FWD_LIKE(T, _38),
        REFLECT_FWD_LIKE(T, _39),
        REFLECT_FWD_LIKE(T, _40),
        REFLECT_FWD_LIKE(T, _41),
        REFLECT_FWD_LIKE(T, _42),
        REFLECT_FWD_LIKE(T, _43));
}
template <class Fn, class T>
[[nodiscard]] constexpr decltype(auto) visit(Fn&& fn, T&& t, std::integral_constant<std::size_t, 44>) noexcept {
    auto&& [_1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16, _17, _18, _19, _20, _21, _22, _23, _24, _25, _26, _27, _28, _29, _30, _31, _32, _33, _34, _35, _36, _37, _38, _39, _40, _41, _42, _43, _44] =
        REFLECT_FWD(t);
    return REFLECT_FWD(fn)(
        REFLECT_FWD_LIKE(T, _1),
        REFLECT_FWD_LIKE(T, _2),
        REFLECT_FWD_LIKE(T, _3),
        REFLECT_FWD_LIKE(T, _4),
        REFLECT_FWD_LIKE(T, _5),
        REFLECT_FWD_LIKE(T, _6),
        REFLECT_FWD_LIKE(T, _7),
        REFLECT_FWD_LIKE(T, _8),
        REFLECT_FWD_LIKE(T, _9),
        REFLECT_FWD_LIKE(T, _10),
        REFLECT_FWD_LIKE(T, _11),
        REFLECT_FWD_LIKE(T, _12),
        REFLECT_FWD_LIKE(T, _13),
        REFLECT_FWD_LIKE(T, _14),
        REFLECT_FWD_LIKE(T, _15),
        REFLECT_FWD_LIKE(T, _16),
        REFLECT_FWD_LIKE(T, _17),
        REFLECT_FWD_LIKE(T, _18),
        REFLECT_FWD_LIKE(T, _19),
        REFLECT_FWD_LIKE(T, _20),
        REFLECT_FWD_LIKE(T, _21),
        REFLECT_FWD_LIKE(T, _22),
        REFLECT_FWD_LIKE(T, _23),
        REFLECT_FWD_LIKE(T, _24),
        REFLECT_FWD_LIKE(T, _25),
        REFLECT_FWD_LIKE(T, _26),
        REFLECT_FWD_LIKE(T, _27),
        REFLECT_FWD_LIKE(T, _28),
        REFLECT_FWD_LIKE(T, _29),
        REFLECT_FWD_LIKE(T, _30),
        REFLECT_FWD_LIKE(T, _31),
        REFLECT_FWD_LIKE(T, _32),
        REFLECT_FWD_LIKE(T, _33),
        REFLECT_FWD_LIKE(T, _34),
        REFLECT_FWD_LIKE(T, _35),
        REFLECT_FWD_LIKE(T, _36),
        REFLECT_FWD_LIKE(T, _37),
        REFLECT_FWD_LIKE(T, _38),
        REFLECT_FWD_LIKE(T, _39),
        REFLECT_FWD_LIKE(T, _40),
        REFLECT_FWD_LIKE(T, _41),
        REFLECT_FWD_LIKE(T, _42),
        REFLECT_FWD_LIKE(T, _43),
        REFLECT_FWD_LIKE(T, _44));
}
template <class Fn, class T>
[[nodiscard]] constexpr decltype(auto) visit(Fn&& fn, T&& t, std::integral_constant<std::size_t, 45>) noexcept {
    auto&& [_1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16, _17, _18, _19, _20, _21, _22, _23, _24, _25, _26, _27, _28, _29, _30, _31, _32, _33, _34, _35, _36, _37, _38, _39, _40, _41, _42, _43, _44, _45] =
        REFLECT_FWD(t);
    return REFLECT_FWD(fn)(
        REFLECT_FWD_LIKE(T, _1),
        REFLECT_FWD_LIKE(T, _2),
        REFLECT_FWD_LIKE(T, _3),
        REFLECT_FWD_LIKE(T, _4),
        REFLECT_FWD_LIKE(T, _5),
        REFLECT_FWD_LIKE(T, _6),
        REFLECT_FWD_LIKE(T, _7),
        REFLECT_FWD_LIKE(T, _8),
        REFLECT_FWD_LIKE(T, _9),
        REFLECT_FWD_LIKE(T, _10),
        REFLECT_FWD_LIKE(T, _11),
        REFLECT_FWD_LIKE(T, _12),
        REFLECT_FWD_LIKE(T, _13),
        REFLECT_FWD_LIKE(T, _14),
        REFLECT_FWD_LIKE(T, _15),
        REFLECT_FWD_LIKE(T, _16),
        REFLECT_FWD_LIKE(T, _17),
        REFLECT_FWD_LIKE(T, _18),
        REFLECT_FWD_LIKE(T, _19),
        REFLECT_FWD_LIKE(T, _20),
        REFLECT_FWD_LIKE(T, _21),
        REFLECT_FWD_LIKE(T, _22),
        REFLECT_FWD_LIKE(T, _23),
        REFLECT_FWD_LIKE(T, _24),
        REFLECT_FWD_LIKE(T, _25),
        REFLECT_FWD_LIKE(T, _26),
        REFLECT_FWD_LIKE(T, _27),
        REFLECT_FWD_LIKE(T, _28),
        REFLECT_FWD_LIKE(T, _29),
        REFLECT_FWD_LIKE(T, _30),
        REFLECT_FWD_LIKE(T, _31),
        REFLECT_FWD_LIKE(T, _32),
        REFLECT_FWD_LIKE(T, _33),
        REFLECT_FWD_LIKE(T, _34),
        REFLECT_FWD_LIKE(T, _35),
        REFLECT_FWD_LIKE(T, _36),
        REFLECT_FWD_LIKE(T, _37),
        REFLECT_FWD_LIKE(T, _38),
        REFLECT_FWD_LIKE(T, _39),
        REFLECT_FWD_LIKE(T, _40),
        REFLECT_FWD_LIKE(T, _41),
        REFLECT_FWD_LIKE(T, _42),
        REFLECT_FWD_LIKE(T, _43),
        REFLECT_FWD_LIKE(T, _44),
        REFLECT_FWD_LIKE(T, _45));
}
template <class Fn, class T>
[[nodiscard]] constexpr decltype(auto) visit(Fn&& fn, T&& t, std::integral_constant<std::size_t, 46>) noexcept {
    auto&& [_1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16, _17, _18, _19, _20, _21, _22, _23, _24, _25, _26, _27, _28, _29, _30, _31, _32, _33, _34, _35, _36, _37, _38, _39, _40, _41, _42, _43, _44, _45, _46] =
        REFLECT_FWD(t);
    return REFLECT_FWD(fn)(
        REFLECT_FWD_LIKE(T, _1),
        REFLECT_FWD_LIKE(T, _2),
        REFLECT_FWD_LIKE(T, _3),
        REFLECT_FWD_LIKE(T, _4),
        REFLECT_FWD_LIKE(T, _5),
        REFLECT_FWD_LIKE(T, _6),
        REFLECT_FWD_LIKE(T, _7),
        REFLECT_FWD_LIKE(T, _8),
        REFLECT_FWD_LIKE(T, _9),
        REFLECT_FWD_LIKE(T, _10),
        REFLECT_FWD_LIKE(T, _11),
        REFLECT_FWD_LIKE(T, _12),
        REFLECT_FWD_LIKE(T, _13),
        REFLECT_FWD_LIKE(T, _14),
        REFLECT_FWD_LIKE(T, _15),
        REFLECT_FWD_LIKE(T, _16),
        REFLECT_FWD_LIKE(T, _17),
        REFLECT_FWD_LIKE(T, _18),
        REFLECT_FWD_LIKE(T, _19),
        REFLECT_FWD_LIKE(T, _20),
        REFLECT_FWD_LIKE(T, _21),
        REFLECT_FWD_LIKE(T, _22),
        REFLECT_FWD_LIKE(T, _23),
        REFLECT_FWD_LIKE(T, _24),
        REFLECT_FWD_LIKE(T, _25),
        REFLECT_FWD_LIKE(T, _26),
        REFLECT_FWD_LIKE(T, _27),
        REFLECT_FWD_LIKE(T, _28),
        REFLECT_FWD_LIKE(T, _29),
        REFLECT_FWD_LIKE(T, _30),
        REFLECT_FWD_LIKE(T, _31),
        REFLECT_FWD_LIKE(T, _32),
        REFLECT_FWD_LIKE(T, _33),
        REFLECT_FWD_LIKE(T, _34),
        REFLECT_FWD_LIKE(T, _35),
        REFLECT_FWD_LIKE(T, _36),
        REFLECT_FWD_LIKE(T, _37),
        REFLECT_FWD_LIKE(T, _38),
        REFLECT_FWD_LIKE(T, _39),
        REFLECT_FWD_LIKE(T, _40),
        REFLECT_FWD_LIKE(T, _41),
        REFLECT_FWD_LIKE(T, _42),
        REFLECT_FWD_LIKE(T, _43),
        REFLECT_FWD_LIKE(T, _44),
        REFLECT_FWD_LIKE(T, _45),
        REFLECT_FWD_LIKE(T, _46));
}
template <class Fn, class T>
[[nodiscard]] constexpr decltype(auto) visit(Fn&& fn, T&& t, std::integral_constant<std::size_t, 47>) noexcept {
    auto&& [_1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16, _17, _18, _19, _20, _21, _22, _23, _24, _25, _26, _27, _28, _29, _30, _31, _32, _33, _34, _35, _36, _37, _38, _39, _40, _41, _42, _43, _44, _45, _46, _47] =
        REFLECT_FWD(t);
    return REFLECT_FWD(fn)(
        REFLECT_FWD_LIKE(T, _1),
        REFLECT_FWD_LIKE(T, _2),
        REFLECT_FWD_LIKE(T, _3),
        REFLECT_FWD_LIKE(T, _4),
        REFLECT_FWD_LIKE(T, _5),
        REFLECT_FWD_LIKE(T, _6),
        REFLECT_FWD_LIKE(T, _7),
        REFLECT_FWD_LIKE(T, _8),
        REFLECT_FWD_LIKE(T, _9),
        REFLECT_FWD_LIKE(T, _10),
        REFLECT_FWD_LIKE(T, _11),
        REFLECT_FWD_LIKE(T, _12),
        REFLECT_FWD_LIKE(T, _13),
        REFLECT_FWD_LIKE(T, _14),
        REFLECT_FWD_LIKE(T, _15),
        REFLECT_FWD_LIKE(T, _16),
        REFLECT_FWD_LIKE(T, _17),
        REFLECT_FWD_LIKE(T, _18),
        REFLECT_FWD_LIKE(T, _19),
        REFLECT_FWD_LIKE(T, _20),
        REFLECT_FWD_LIKE(T, _21),
        REFLECT_FWD_LIKE(T, _22),
        REFLECT_FWD_LIKE(T, _23),
        REFLECT_FWD_LIKE(T, _24),
        REFLECT_FWD_LIKE(T, _25),
        REFLECT_FWD_LIKE(T, _26),
        REFLECT_FWD_LIKE(T, _27),
        REFLECT_FWD_LIKE(T, _28),
        REFLECT_FWD_LIKE(T, _29),
        REFLECT_FWD_LIKE(T, _30),
        REFLECT_FWD_LIKE(T, _31),
        REFLECT_FWD_LIKE(T, _32),
        REFLECT_FWD_LIKE(T, _33),
        REFLECT_FWD_LIKE(T, _34),
        REFLECT_FWD_LIKE(T, _35),
        REFLECT_FWD_LIKE(T, _36),
        REFLECT_FWD_LIKE(T, _37),
        REFLECT_FWD_LIKE(T, _38),
        REFLECT_FWD_LIKE(T, _39),
        REFLECT_FWD_LIKE(T, _40),
        REFLECT_FWD_LIKE(T, _41),
        REFLECT_FWD_LIKE(T, _42),
        REFLECT_FWD_LIKE(T, _43),
        REFLECT_FWD_LIKE(T, _44),
        REFLECT_FWD_LIKE(T, _45),
        REFLECT_FWD_LIKE(T, _46),
        REFLECT_FWD_LIKE(T, _47));
}
template <class Fn, class T>
[[nodiscard]] constexpr decltype(auto) visit(Fn&& fn, T&& t, std::integral_constant<std::size_t, 48>) noexcept {
    auto&& [_1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16, _17, _18, _19, _20, _21, _22, _23, _24, _25, _26, _27, _28, _29, _30, _31, _32, _33, _34, _35, _36, _37, _38, _39, _40, _41, _42, _43, _44, _45, _46, _47, _48] =
        REFLECT_FWD(t);
    return REFLECT_FWD(fn)(
        REFLECT_FWD_LIKE(T, _1),
        REFLECT_FWD_LIKE(T, _2),
        REFLECT_FWD_LIKE(T, _3),
        REFLECT_FWD_LIKE(T, _4),
        REFLECT_FWD_LIKE(T, _5),
        REFLECT_FWD_LIKE(T, _6),
        REFLECT_FWD_LIKE(T, _7),
        REFLECT_FWD_LIKE(T, _8),
        REFLECT_FWD_LIKE(T, _9),
        REFLECT_FWD_LIKE(T, _10),
        REFLECT_FWD_LIKE(T, _11),
        REFLECT_FWD_LIKE(T, _12),
        REFLECT_FWD_LIKE(T, _13),
        REFLECT_FWD_LIKE(T, _14),
        REFLECT_FWD_LIKE(T, _15),
        REFLECT_FWD_LIKE(T, _16),
        REFLECT_FWD_LIKE(T, _17),
        REFLECT_FWD_LIKE(T, _18),
        REFLECT_FWD_LIKE(T, _19),
        REFLECT_FWD_LIKE(T, _20),
        REFLECT_FWD_LIKE(T, _21),
        REFLECT_FWD_LIKE(T, _22),
        REFLECT_FWD_LIKE(T, _23),
        REFLECT_FWD_LIKE(T, _24),
        REFLECT_FWD_LIKE(T, _25),
        REFLECT_FWD_LIKE(T, _26),
        REFLECT_FWD_LIKE(T, _27),
        REFLECT_FWD_LIKE(T, _28),
        REFLECT_FWD_LIKE(T, _29),
        REFLECT_FWD_LIKE(T, _30),
        REFLECT_FWD_LIKE(T, _31),
        REFLECT_FWD_LIKE(T, _32),
        REFLECT_FWD_LIKE(T, _33),
        REFLECT_FWD_LIKE(T, _34),
        REFLECT_FWD_LIKE(T, _35),
        REFLECT_FWD_LIKE(T, _36),
        REFLECT_FWD_LIKE(T, _37),
        REFLECT_FWD_LIKE(T, _38),
        REFLECT_FWD_LIKE(T, _39),
        REFLECT_FWD_LIKE(T, _40),
        REFLECT_FWD_LIKE(T, _41),
        REFLECT_FWD_LIKE(T, _42),
        REFLECT_FWD_LIKE(T, _43),
        REFLECT_FWD_LIKE(T, _44),
        REFLECT_FWD_LIKE(T, _45),
        REFLECT_FWD_LIKE(T, _46),
        REFLECT_FWD_LIKE(T, _47),
        REFLECT_FWD_LIKE(T, _48));
}
template <class Fn, class T>
[[nodiscard]] constexpr decltype(auto) visit(Fn&& fn, T&& t, std::integral_constant<std::size_t, 49>) noexcept {
    auto&& [_1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16, _17, _18, _19, _20, _21, _22, _23, _24, _25, _26, _27, _28, _29, _30, _31, _32, _33, _34, _35, _36, _37, _38, _39, _40, _41, _42, _43, _44, _45, _46, _47, _48, _49] =
        REFLECT_FWD(t);
    return REFLECT_FWD(fn)(
        REFLECT_FWD_LIKE(T, _1),
        REFLECT_FWD_LIKE(T, _2),
        REFLECT_FWD_LIKE(T, _3),
        REFLECT_FWD_LIKE(T, _4),
        REFLECT_FWD_LIKE(T, _5),
        REFLECT_FWD_LIKE(T, _6),
        REFLECT_FWD_LIKE(T, _7),
        REFLECT_FWD_LIKE(T, _8),
        REFLECT_FWD_LIKE(T, _9),
        REFLECT_FWD_LIKE(T, _10),
        REFLECT_FWD_LIKE(T, _11),
        REFLECT_FWD_LIKE(T, _12),
        REFLECT_FWD_LIKE(T, _13),
        REFLECT_FWD_LIKE(T, _14),
        REFLECT_FWD_LIKE(T, _15),
        REFLECT_FWD_LIKE(T, _16),
        REFLECT_FWD_LIKE(T, _17),
        REFLECT_FWD_LIKE(T, _18),
        REFLECT_FWD_LIKE(T, _19),
        REFLECT_FWD_LIKE(T, _20),
        REFLECT_FWD_LIKE(T, _21),
        REFLECT_FWD_LIKE(T, _22),
        REFLECT_FWD_LIKE(T, _23),
        REFLECT_FWD_LIKE(T, _24),
        REFLECT_FWD_LIKE(T, _25),
        REFLECT_FWD_LIKE(T, _26),
        REFLECT_FWD_LIKE(T, _27),
        REFLECT_FWD_LIKE(T, _28),
        REFLECT_FWD_LIKE(T, _29),
        REFLECT_FWD_LIKE(T, _30),
        REFLECT_FWD_LIKE(T, _31),
        REFLECT_FWD_LIKE(T, _32),
        REFLECT_FWD_LIKE(T, _33),
        REFLECT_FWD_LIKE(T, _34),
        REFLECT_FWD_LIKE(T, _35),
        REFLECT_FWD_LIKE(T, _36),
        REFLECT_FWD_LIKE(T, _37),
        REFLECT_FWD_LIKE(T, _38),
        REFLECT_FWD_LIKE(T, _39),
        REFLECT_FWD_LIKE(T, _40),
        REFLECT_FWD_LIKE(T, _41),
        REFLECT_FWD_LIKE(T, _42),
        REFLECT_FWD_LIKE(T, _43),
        REFLECT_FWD_LIKE(T, _44),
        REFLECT_FWD_LIKE(T, _45),
        REFLECT_FWD_LIKE(T, _46),
        REFLECT_FWD_LIKE(T, _47),
        REFLECT_FWD_LIKE(T, _48),
        REFLECT_FWD_LIKE(T, _49));
}
template <class Fn, class T>
[[nodiscard]] constexpr decltype(auto) visit(Fn&& fn, T&& t, std::integral_constant<std::size_t, 50>) noexcept {
    auto&& [_1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16, _17, _18, _19, _20, _21, _22, _23, _24, _25, _26, _27, _28, _29, _30, _31, _32, _33, _34, _35, _36, _37, _38, _39, _40, _41, _42, _43, _44, _45, _46, _47, _48, _49, _50] =
        REFLECT_FWD(t);
    return REFLECT_FWD(fn)(
        REFLECT_FWD_LIKE(T, _1),
        REFLECT_FWD_LIKE(T, _2),
        REFLECT_FWD_LIKE(T, _3),
        REFLECT_FWD_LIKE(T, _4),
        REFLECT_FWD_LIKE(T, _5),
        REFLECT_FWD_LIKE(T, _6),
        REFLECT_FWD_LIKE(T, _7),
        REFLECT_FWD_LIKE(T, _8),
        REFLECT_FWD_LIKE(T, _9),
        REFLECT_FWD_LIKE(T, _10),
        REFLECT_FWD_LIKE(T, _11),
        REFLECT_FWD_LIKE(T, _12),
        REFLECT_FWD_LIKE(T, _13),
        REFLECT_FWD_LIKE(T, _14),
        REFLECT_FWD_LIKE(T, _15),
        REFLECT_FWD_LIKE(T, _16),
        REFLECT_FWD_LIKE(T, _17),
        REFLECT_FWD_LIKE(T, _18),
        REFLECT_FWD_LIKE(T, _19),
        REFLECT_FWD_LIKE(T, _20),
        REFLECT_FWD_LIKE(T, _21),
        REFLECT_FWD_LIKE(T, _22),
        REFLECT_FWD_LIKE(T, _23),
        REFLECT_FWD_LIKE(T, _24),
        REFLECT_FWD_LIKE(T, _25),
        REFLECT_FWD_LIKE(T, _26),
        REFLECT_FWD_LIKE(T, _27),
        REFLECT_FWD_LIKE(T, _28),
        REFLECT_FWD_LIKE(T, _29),
        REFLECT_FWD_LIKE(T, _30),
        REFLECT_FWD_LIKE(T, _31),
        REFLECT_FWD_LIKE(T, _32),
        REFLECT_FWD_LIKE(T, _33),
        REFLECT_FWD_LIKE(T, _34),
        REFLECT_FWD_LIKE(T, _35),
        REFLECT_FWD_LIKE(T, _36),
        REFLECT_FWD_LIKE(T, _37),
        REFLECT_FWD_LIKE(T, _38),
        REFLECT_FWD_LIKE(T, _39),
        REFLECT_FWD_LIKE(T, _40),
        REFLECT_FWD_LIKE(T, _41),
        REFLECT_FWD_LIKE(T, _42),
        REFLECT_FWD_LIKE(T, _43),
        REFLECT_FWD_LIKE(T, _44),
        REFLECT_FWD_LIKE(T, _45),
        REFLECT_FWD_LIKE(T, _46),
        REFLECT_FWD_LIKE(T, _47),
        REFLECT_FWD_LIKE(T, _48),
        REFLECT_FWD_LIKE(T, _49),
        REFLECT_FWD_LIKE(T, _50));
}
template <class Fn, class T>
[[nodiscard]] constexpr decltype(auto) visit(Fn&& fn, T&& t, std::integral_constant<std::size_t, 51>) noexcept {
    auto&& [_1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16, _17, _18, _19, _20, _21, _22, _23, _24, _25, _26, _27, _28, _29, _30, _31, _32, _33, _34, _35, _36, _37, _38, _39, _40, _41, _42, _43, _44, _45, _46, _47, _48, _49, _50, _51] =
        REFLECT_FWD(t);
    return REFLECT_FWD(fn)(
        REFLECT_FWD_LIKE(T, _1),
        REFLECT_FWD_LIKE(T, _2),
        REFLECT_FWD_LIKE(T, _3),
        REFLECT_FWD_LIKE(T, _4),
        REFLECT_FWD_LIKE(T, _5),
        REFLECT_FWD_LIKE(T, _6),
        REFLECT_FWD_LIKE(T, _7),
        REFLECT_FWD_LIKE(T, _8),
        REFLECT_FWD_LIKE(T, _9),
        REFLECT_FWD_LIKE(T, _10),
        REFLECT_FWD_LIKE(T, _11),
        REFLECT_FWD_LIKE(T, _12),
        REFLECT_FWD_LIKE(T, _13),
        REFLECT_FWD_LIKE(T, _14),
        REFLECT_FWD_LIKE(T, _15),
        REFLECT_FWD_LIKE(T, _16),
        REFLECT_FWD_LIKE(T, _17),
        REFLECT_FWD_LIKE(T, _18),
        REFLECT_FWD_LIKE(T, _19),
        REFLECT_FWD_LIKE(T, _20),
        REFLECT_FWD_LIKE(T, _21),
        REFLECT_FWD_LIKE(T, _22),
        REFLECT_FWD_LIKE(T, _23),
        REFLECT_FWD_LIKE(T, _24),
        REFLECT_FWD_LIKE(T, _25),
        REFLECT_FWD_LIKE(T, _26),
        REFLECT_FWD_LIKE(T, _27),
        REFLECT_FWD_LIKE(T, _28),
        REFLECT_FWD_LIKE(T, _29),
        REFLECT_FWD_LIKE(T, _30),
        REFLECT_FWD_LIKE(T, _31),
        REFLECT_FWD_LIKE(T, _32),
        REFLECT_FWD_LIKE(T, _33),
        REFLECT_FWD_LIKE(T, _34),
        REFLECT_FWD_LIKE(T, _35),
        REFLECT_FWD_LIKE(T, _36),
        REFLECT_FWD_LIKE(T, _37),
        REFLECT_FWD_LIKE(T, _38),
        REFLECT_FWD_LIKE(T, _39),
        REFLECT_FWD_LIKE(T, _40),
        REFLECT_FWD_LIKE(T, _41),
        REFLECT_FWD_LIKE(T, _42),
        REFLECT_FWD_LIKE(T, _43),
        REFLECT_FWD_LIKE(T, _44),
        REFLECT_FWD_LIKE(T, _45),
        REFLECT_FWD_LIKE(T, _46),
        REFLECT_FWD_LIKE(T, _47),
        REFLECT_FWD_LIKE(T, _48),
        REFLECT_FWD_LIKE(T, _49),
        REFLECT_FWD_LIKE(T, _50),
        REFLECT_FWD_LIKE(T, _51));
}
template <class Fn, class T>
[[nodiscard]] constexpr decltype(auto) visit(Fn&& fn, T&& t, std::integral_constant<std::size_t, 52>) noexcept {
    auto&& [_1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16, _17, _18, _19, _20, _21, _22, _23, _24, _25, _26, _27, _28, _29, _30, _31, _32, _33, _34, _35, _36, _37, _38, _39, _40, _41, _42, _43, _44, _45, _46, _47, _48, _49, _50, _51, _52] = REFLECT_FWD(
        t);
    return REFLECT_FWD(fn)(
        REFLECT_FWD_LIKE(T, _1),
        REFLECT_FWD_LIKE(T, _2),
        REFLECT_FWD_LIKE(T, _3),
        REFLECT_FWD_LIKE(T, _4),
        REFLECT_FWD_LIKE(T, _5),
        REFLECT_FWD_LIKE(T, _6),
        REFLECT_FWD_LIKE(T, _7),
        REFLECT_FWD_LIKE(T, _8),
        REFLECT_FWD_LIKE(T, _9),
        REFLECT_FWD_LIKE(T, _10),
        REFLECT_FWD_LIKE(T, _11),
        REFLECT_FWD_LIKE(T, _12),
        REFLECT_FWD_LIKE(T, _13),
        REFLECT_FWD_LIKE(T, _14),
        REFLECT_FWD_LIKE(T, _15),
        REFLECT_FWD_LIKE(T, _16),
        REFLECT_FWD_LIKE(T, _17),
        REFLECT_FWD_LIKE(T, _18),
        REFLECT_FWD_LIKE(T, _19),
        REFLECT_FWD_LIKE(T, _20),
        REFLECT_FWD_LIKE(T, _21),
        REFLECT_FWD_LIKE(T, _22),
        REFLECT_FWD_LIKE(T, _23),
        REFLECT_FWD_LIKE(T, _24),
        REFLECT_FWD_LIKE(T, _25),
        REFLECT_FWD_LIKE(T, _26),
        REFLECT_FWD_LIKE(T, _27),
        REFLECT_FWD_LIKE(T, _28),
        REFLECT_FWD_LIKE(T, _29),
        REFLECT_FWD_LIKE(T, _30),
        REFLECT_FWD_LIKE(T, _31),
        REFLECT_FWD_LIKE(T, _32),
        REFLECT_FWD_LIKE(T, _33),
        REFLECT_FWD_LIKE(T, _34),
        REFLECT_FWD_LIKE(T, _35),
        REFLECT_FWD_LIKE(T, _36),
        REFLECT_FWD_LIKE(T, _37),
        REFLECT_FWD_LIKE(T, _38),
        REFLECT_FWD_LIKE(T, _39),
        REFLECT_FWD_LIKE(T, _40),
        REFLECT_FWD_LIKE(T, _41),
        REFLECT_FWD_LIKE(T, _42),
        REFLECT_FWD_LIKE(T, _43),
        REFLECT_FWD_LIKE(T, _44),
        REFLECT_FWD_LIKE(T, _45),
        REFLECT_FWD_LIKE(T, _46),
        REFLECT_FWD_LIKE(T, _47),
        REFLECT_FWD_LIKE(T, _48),
        REFLECT_FWD_LIKE(T, _49),
        REFLECT_FWD_LIKE(T, _50),
        REFLECT_FWD_LIKE(T, _51),
        REFLECT_FWD_LIKE(T, _52));
}
template <class Fn, class T>
[[nodiscard]] constexpr decltype(auto) visit(Fn&& fn, T&& t, std::integral_constant<std::size_t, 53>) noexcept {
    auto&& [_1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16, _17, _18, _19, _20, _21, _22, _23, _24, _25, _26, _27, _28, _29, _30, _31, _32, _33, _34, _35, _36, _37, _38, _39, _40, _41, _42, _43, _44, _45, _46, _47, _48, _49, _50, _51, _52, _53] =
        REFLECT_FWD(t);
    return REFLECT_FWD(fn)(
        REFLECT_FWD_LIKE(T, _1),
        REFLECT_FWD_LIKE(T, _2),
        REFLECT_FWD_LIKE(T, _3),
        REFLECT_FWD_LIKE(T, _4),
        REFLECT_FWD_LIKE(T, _5),
        REFLECT_FWD_LIKE(T, _6),
        REFLECT_FWD_LIKE(T, _7),
        REFLECT_FWD_LIKE(T, _8),
        REFLECT_FWD_LIKE(T, _9),
        REFLECT_FWD_LIKE(T, _10),
        REFLECT_FWD_LIKE(T, _11),
        REFLECT_FWD_LIKE(T, _12),
        REFLECT_FWD_LIKE(T, _13),
        REFLECT_FWD_LIKE(T, _14),
        REFLECT_FWD_LIKE(T, _15),
        REFLECT_FWD_LIKE(T, _16),
        REFLECT_FWD_LIKE(T, _17),
        REFLECT_FWD_LIKE(T, _18),
        REFLECT_FWD_LIKE(T, _19),
        REFLECT_FWD_LIKE(T, _20),
        REFLECT_FWD_LIKE(T, _21),
        REFLECT_FWD_LIKE(T, _22),
        REFLECT_FWD_LIKE(T, _23),
        REFLECT_FWD_LIKE(T, _24),
        REFLECT_FWD_LIKE(T, _25),
        REFLECT_FWD_LIKE(T, _26),
        REFLECT_FWD_LIKE(T, _27),
        REFLECT_FWD_LIKE(T, _28),
        REFLECT_FWD_LIKE(T, _29),
        REFLECT_FWD_LIKE(T, _30),
        REFLECT_FWD_LIKE(T, _31),
        REFLECT_FWD_LIKE(T, _32),
        REFLECT_FWD_LIKE(T, _33),
        REFLECT_FWD_LIKE(T, _34),
        REFLECT_FWD_LIKE(T, _35),
        REFLECT_FWD_LIKE(T, _36),
        REFLECT_FWD_LIKE(T, _37),
        REFLECT_FWD_LIKE(T, _38),
        REFLECT_FWD_LIKE(T, _39),
        REFLECT_FWD_LIKE(T, _40),
        REFLECT_FWD_LIKE(T, _41),
        REFLECT_FWD_LIKE(T, _42),
        REFLECT_FWD_LIKE(T, _43),
        REFLECT_FWD_LIKE(T, _44),
        REFLECT_FWD_LIKE(T, _45),
        REFLECT_FWD_LIKE(T, _46),
        REFLECT_FWD_LIKE(T, _47),
        REFLECT_FWD_LIKE(T, _48),
        REFLECT_FWD_LIKE(T, _49),
        REFLECT_FWD_LIKE(T, _50),
        REFLECT_FWD_LIKE(T, _51),
        REFLECT_FWD_LIKE(T, _52),
        REFLECT_FWD_LIKE(T, _53));
}
template <class Fn, class T>
[[nodiscard]] constexpr decltype(auto) visit(Fn&& fn, T&& t, std::integral_constant<std::size_t, 54>) noexcept {
    auto&& [_1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16, _17, _18, _19, _20, _21, _22, _23, _24, _25, _26, _27, _28, _29, _30, _31, _32, _33, _34, _35, _36, _37, _38, _39, _40, _41, _42, _43, _44, _45, _46, _47, _48, _49, _50, _51, _52, _53, _54] =
        REFLECT_FWD(t);
    return REFLECT_FWD(fn)(
        REFLECT_FWD_LIKE(T, _1),
        REFLECT_FWD_LIKE(T, _2),
        REFLECT_FWD_LIKE(T, _3),
        REFLECT_FWD_LIKE(T, _4),
        REFLECT_FWD_LIKE(T, _5),
        REFLECT_FWD_LIKE(T, _6),
        REFLECT_FWD_LIKE(T, _7),
        REFLECT_FWD_LIKE(T, _8),
        REFLECT_FWD_LIKE(T, _9),
        REFLECT_FWD_LIKE(T, _10),
        REFLECT_FWD_LIKE(T, _11),
        REFLECT_FWD_LIKE(T, _12),
        REFLECT_FWD_LIKE(T, _13),
        REFLECT_FWD_LIKE(T, _14),
        REFLECT_FWD_LIKE(T, _15),
        REFLECT_FWD_LIKE(T, _16),
        REFLECT_FWD_LIKE(T, _17),
        REFLECT_FWD_LIKE(T, _18),
        REFLECT_FWD_LIKE(T, _19),
        REFLECT_FWD_LIKE(T, _20),
        REFLECT_FWD_LIKE(T, _21),
        REFLECT_FWD_LIKE(T, _22),
        REFLECT_FWD_LIKE(T, _23),
        REFLECT_FWD_LIKE(T, _24),
        REFLECT_FWD_LIKE(T, _25),
        REFLECT_FWD_LIKE(T, _26),
        REFLECT_FWD_LIKE(T, _27),
        REFLECT_FWD_LIKE(T, _28),
        REFLECT_FWD_LIKE(T, _29),
        REFLECT_FWD_LIKE(T, _30),
        REFLECT_FWD_LIKE(T, _31),
        REFLECT_FWD_LIKE(T, _32),
        REFLECT_FWD_LIKE(T, _33),
        REFLECT_FWD_LIKE(T, _34),
        REFLECT_FWD_LIKE(T, _35),
        REFLECT_FWD_LIKE(T, _36),
        REFLECT_FWD_LIKE(T, _37),
        REFLECT_FWD_LIKE(T, _38),
        REFLECT_FWD_LIKE(T, _39),
        REFLECT_FWD_LIKE(T, _40),
        REFLECT_FWD_LIKE(T, _41),
        REFLECT_FWD_LIKE(T, _42),
        REFLECT_FWD_LIKE(T, _43),
        REFLECT_FWD_LIKE(T, _44),
        REFLECT_FWD_LIKE(T, _45),
        REFLECT_FWD_LIKE(T, _46),
        REFLECT_FWD_LIKE(T, _47),
        REFLECT_FWD_LIKE(T, _48),
        REFLECT_FWD_LIKE(T, _49),
        REFLECT_FWD_LIKE(T, _50),
        REFLECT_FWD_LIKE(T, _51),
        REFLECT_FWD_LIKE(T, _52),
        REFLECT_FWD_LIKE(T, _53),
        REFLECT_FWD_LIKE(T, _54));
}
template <class Fn, class T>
[[nodiscard]] constexpr decltype(auto) visit(Fn&& fn, T&& t, std::integral_constant<std::size_t, 55>) noexcept {
    auto&& [_1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16, _17, _18, _19, _20, _21, _22, _23, _24, _25, _26, _27, _28, _29, _30, _31, _32, _33, _34, _35, _36, _37, _38, _39, _40, _41, _42, _43, _44, _45, _46, _47, _48, _49, _50, _51, _52, _53, _54, _55] =
        REFLECT_FWD(t);
    return REFLECT_FWD(fn)(
        REFLECT_FWD_LIKE(T, _1),
        REFLECT_FWD_LIKE(T, _2),
        REFLECT_FWD_LIKE(T, _3),
        REFLECT_FWD_LIKE(T, _4),
        REFLECT_FWD_LIKE(T, _5),
        REFLECT_FWD_LIKE(T, _6),
        REFLECT_FWD_LIKE(T, _7),
        REFLECT_FWD_LIKE(T, _8),
        REFLECT_FWD_LIKE(T, _9),
        REFLECT_FWD_LIKE(T, _10),
        REFLECT_FWD_LIKE(T, _11),
        REFLECT_FWD_LIKE(T, _12),
        REFLECT_FWD_LIKE(T, _13),
        REFLECT_FWD_LIKE(T, _14),
        REFLECT_FWD_LIKE(T, _15),
        REFLECT_FWD_LIKE(T, _16),
        REFLECT_FWD_LIKE(T, _17),
        REFLECT_FWD_LIKE(T, _18),
        REFLECT_FWD_LIKE(T, _19),
        REFLECT_FWD_LIKE(T, _20),
        REFLECT_FWD_LIKE(T, _21),
        REFLECT_FWD_LIKE(T, _22),
        REFLECT_FWD_LIKE(T, _23),
        REFLECT_FWD_LIKE(T, _24),
        REFLECT_FWD_LIKE(T, _25),
        REFLECT_FWD_LIKE(T, _26),
        REFLECT_FWD_LIKE(T, _27),
        REFLECT_FWD_LIKE(T, _28),
        REFLECT_FWD_LIKE(T, _29),
        REFLECT_FWD_LIKE(T, _30),
        REFLECT_FWD_LIKE(T, _31),
        REFLECT_FWD_LIKE(T, _32),
        REFLECT_FWD_LIKE(T, _33),
        REFLECT_FWD_LIKE(T, _34),
        REFLECT_FWD_LIKE(T, _35),
        REFLECT_FWD_LIKE(T, _36),
        REFLECT_FWD_LIKE(T, _37),
        REFLECT_FWD_LIKE(T, _38),
        REFLECT_FWD_LIKE(T, _39),
        REFLECT_FWD_LIKE(T, _40),
        REFLECT_FWD_LIKE(T, _41),
        REFLECT_FWD_LIKE(T, _42),
        REFLECT_FWD_LIKE(T, _43),
        REFLECT_FWD_LIKE(T, _44),
        REFLECT_FWD_LIKE(T, _45),
        REFLECT_FWD_LIKE(T, _46),
        REFLECT_FWD_LIKE(T, _47),
        REFLECT_FWD_LIKE(T, _48),
        REFLECT_FWD_LIKE(T, _49),
        REFLECT_FWD_LIKE(T, _50),
        REFLECT_FWD_LIKE(T, _51),
        REFLECT_FWD_LIKE(T, _52),
        REFLECT_FWD_LIKE(T, _53),
        REFLECT_FWD_LIKE(T, _54),
        REFLECT_FWD_LIKE(T, _55));
}
template <class Fn, class T>
[[nodiscard]] constexpr decltype(auto) visit(Fn&& fn, T&& t, std::integral_constant<std::size_t, 56>) noexcept {
    auto&& [_1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16, _17, _18, _19, _20, _21, _22, _23, _24, _25, _26, _27, _28, _29, _30, _31, _32, _33, _34, _35, _36, _37, _38, _39, _40, _41, _42, _43, _44, _45, _46, _47, _48, _49, _50, _51, _52, _53, _54, _55, _56] =
        REFLECT_FWD(t);
    return REFLECT_FWD(fn)(
        REFLECT_FWD_LIKE(T, _1),
        REFLECT_FWD_LIKE(T, _2),
        REFLECT_FWD_LIKE(T, _3),
        REFLECT_FWD_LIKE(T, _4),
        REFLECT_FWD_LIKE(T, _5),
        REFLECT_FWD_LIKE(T, _6),
        REFLECT_FWD_LIKE(T, _7),
        REFLECT_FWD_LIKE(T, _8),
        REFLECT_FWD_LIKE(T, _9),
        REFLECT_FWD_LIKE(T, _10),
        REFLECT_FWD_LIKE(T, _11),
        REFLECT_FWD_LIKE(T, _12),
        REFLECT_FWD_LIKE(T, _13),
        REFLECT_FWD_LIKE(T, _14),
        REFLECT_FWD_LIKE(T, _15),
        REFLECT_FWD_LIKE(T, _16),
        REFLECT_FWD_LIKE(T, _17),
        REFLECT_FWD_LIKE(T, _18),
        REFLECT_FWD_LIKE(T, _19),
        REFLECT_FWD_LIKE(T, _20),
        REFLECT_FWD_LIKE(T, _21),
        REFLECT_FWD_LIKE(T, _22),
        REFLECT_FWD_LIKE(T, _23),
        REFLECT_FWD_LIKE(T, _24),
        REFLECT_FWD_LIKE(T, _25),
        REFLECT_FWD_LIKE(T, _26),
        REFLECT_FWD_LIKE(T, _27),
        REFLECT_FWD_LIKE(T, _28),
        REFLECT_FWD_LIKE(T, _29),
        REFLECT_FWD_LIKE(T, _30),
        REFLECT_FWD_LIKE(T, _31),
        REFLECT_FWD_LIKE(T, _32),
        REFLECT_FWD_LIKE(T, _33),
        REFLECT_FWD_LIKE(T, _34),
        REFLECT_FWD_LIKE(T, _35),
        REFLECT_FWD_LIKE(T, _36),
        REFLECT_FWD_LIKE(T, _37),
        REFLECT_FWD_LIKE(T, _38),
        REFLECT_FWD_LIKE(T, _39),
        REFLECT_FWD_LIKE(T, _40),
        REFLECT_FWD_LIKE(T, _41),
        REFLECT_FWD_LIKE(T, _42),
        REFLECT_FWD_LIKE(T, _43),
        REFLECT_FWD_LIKE(T, _44),
        REFLECT_FWD_LIKE(T, _45),
        REFLECT_FWD_LIKE(T, _46),
        REFLECT_FWD_LIKE(T, _47),
        REFLECT_FWD_LIKE(T, _48),
        REFLECT_FWD_LIKE(T, _49),
        REFLECT_FWD_LIKE(T, _50),
        REFLECT_FWD_LIKE(T, _51),
        REFLECT_FWD_LIKE(T, _52),
        REFLECT_FWD_LIKE(T, _53),
        REFLECT_FWD_LIKE(T, _54),
        REFLECT_FWD_LIKE(T, _55),
        REFLECT_FWD_LIKE(T, _56));
}
template <class Fn, class T>
[[nodiscard]] constexpr decltype(auto) visit(Fn&& fn, T&& t, std::integral_constant<std::size_t, 57>) noexcept {
    auto&& [_1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16, _17, _18, _19, _20, _21, _22, _23, _24, _25, _26, _27, _28, _29, _30, _31, _32, _33, _34, _35, _36, _37, _38, _39, _40, _41, _42, _43, _44, _45, _46, _47, _48, _49, _50, _51, _52, _53, _54, _55, _56, _57] =
        REFLECT_FWD(t);
    return REFLECT_FWD(fn)(
        REFLECT_FWD_LIKE(T, _1),
        REFLECT_FWD_LIKE(T, _2),
        REFLECT_FWD_LIKE(T, _3),
        REFLECT_FWD_LIKE(T, _4),
        REFLECT_FWD_LIKE(T, _5),
        REFLECT_FWD_LIKE(T, _6),
        REFLECT_FWD_LIKE(T, _7),
        REFLECT_FWD_LIKE(T, _8),
        REFLECT_FWD_LIKE(T, _9),
        REFLECT_FWD_LIKE(T, _10),
        REFLECT_FWD_LIKE(T, _11),
        REFLECT_FWD_LIKE(T, _12),
        REFLECT_FWD_LIKE(T, _13),
        REFLECT_FWD_LIKE(T, _14),
        REFLECT_FWD_LIKE(T, _15),
        REFLECT_FWD_LIKE(T, _16),
        REFLECT_FWD_LIKE(T, _17),
        REFLECT_FWD_LIKE(T, _18),
        REFLECT_FWD_LIKE(T, _19),
        REFLECT_FWD_LIKE(T, _20),
        REFLECT_FWD_LIKE(T, _21),
        REFLECT_FWD_LIKE(T, _22),
        REFLECT_FWD_LIKE(T, _23),
        REFLECT_FWD_LIKE(T, _24),
        REFLECT_FWD_LIKE(T, _25),
        REFLECT_FWD_LIKE(T, _26),
        REFLECT_FWD_LIKE(T, _27),
        REFLECT_FWD_LIKE(T, _28),
        REFLECT_FWD_LIKE(T, _29),
        REFLECT_FWD_LIKE(T, _30),
        REFLECT_FWD_LIKE(T, _31),
        REFLECT_FWD_LIKE(T, _32),
        REFLECT_FWD_LIKE(T, _33),
        REFLECT_FWD_LIKE(T, _34),
        REFLECT_FWD_LIKE(T, _35),
        REFLECT_FWD_LIKE(T, _36),
        REFLECT_FWD_LIKE(T, _37),
        REFLECT_FWD_LIKE(T, _38),
        REFLECT_FWD_LIKE(T, _39),
        REFLECT_FWD_LIKE(T, _40),
        REFLECT_FWD_LIKE(T, _41),
        REFLECT_FWD_LIKE(T, _42),
        REFLECT_FWD_LIKE(T, _43),
        REFLECT_FWD_LIKE(T, _44),
        REFLECT_FWD_LIKE(T, _45),
        REFLECT_FWD_LIKE(T, _46),
        REFLECT_FWD_LIKE(T, _47),
        REFLECT_FWD_LIKE(T, _48),
        REFLECT_FWD_LIKE(T, _49),
        REFLECT_FWD_LIKE(T, _50),
        REFLECT_FWD_LIKE(T, _51),
        REFLECT_FWD_LIKE(T, _52),
        REFLECT_FWD_LIKE(T, _53),
        REFLECT_FWD_LIKE(T, _54),
        REFLECT_FWD_LIKE(T, _55),
        REFLECT_FWD_LIKE(T, _56),
        REFLECT_FWD_LIKE(T, _57));
}
template <class Fn, class T>
[[nodiscard]] constexpr decltype(auto) visit(Fn&& fn, T&& t, std::integral_constant<std::size_t, 58>) noexcept {
    auto&& [_1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16, _17, _18, _19, _20, _21, _22, _23, _24, _25, _26, _27, _28, _29, _30, _31, _32, _33, _34, _35, _36, _37, _38, _39, _40, _41, _42, _43, _44, _45, _46, _47, _48, _49, _50, _51, _52, _53, _54, _55, _56, _57, _58] =
        REFLECT_FWD(t);
    return REFLECT_FWD(fn)(
        REFLECT_FWD_LIKE(T, _1),
        REFLECT_FWD_LIKE(T, _2),
        REFLECT_FWD_LIKE(T, _3),
        REFLECT_FWD_LIKE(T, _4),
        REFLECT_FWD_LIKE(T, _5),
        REFLECT_FWD_LIKE(T, _6),
        REFLECT_FWD_LIKE(T, _7),
        REFLECT_FWD_LIKE(T, _8),
        REFLECT_FWD_LIKE(T, _9),
        REFLECT_FWD_LIKE(T, _10),
        REFLECT_FWD_LIKE(T, _11),
        REFLECT_FWD_LIKE(T, _12),
        REFLECT_FWD_LIKE(T, _13),
        REFLECT_FWD_LIKE(T, _14),
        REFLECT_FWD_LIKE(T, _15),
        REFLECT_FWD_LIKE(T, _16),
        REFLECT_FWD_LIKE(T, _17),
        REFLECT_FWD_LIKE(T, _18),
        REFLECT_FWD_LIKE(T, _19),
        REFLECT_FWD_LIKE(T, _20),
        REFLECT_FWD_LIKE(T, _21),
        REFLECT_FWD_LIKE(T, _22),
        REFLECT_FWD_LIKE(T, _23),
        REFLECT_FWD_LIKE(T, _24),
        REFLECT_FWD_LIKE(T, _25),
        REFLECT_FWD_LIKE(T, _26),
        REFLECT_FWD_LIKE(T, _27),
        REFLECT_FWD_LIKE(T, _28),
        REFLECT_FWD_LIKE(T, _29),
        REFLECT_FWD_LIKE(T, _30),
        REFLECT_FWD_LIKE(T, _31),
        REFLECT_FWD_LIKE(T, _32),
        REFLECT_FWD_LIKE(T, _33),
        REFLECT_FWD_LIKE(T, _34),
        REFLECT_FWD_LIKE(T, _35),
        REFLECT_FWD_LIKE(T, _36),
        REFLECT_FWD_LIKE(T, _37),
        REFLECT_FWD_LIKE(T, _38),
        REFLECT_FWD_LIKE(T, _39),
        REFLECT_FWD_LIKE(T, _40),
        REFLECT_FWD_LIKE(T, _41),
        REFLECT_FWD_LIKE(T, _42),
        REFLECT_FWD_LIKE(T, _43),
        REFLECT_FWD_LIKE(T, _44),
        REFLECT_FWD_LIKE(T, _45),
        REFLECT_FWD_LIKE(T, _46),
        REFLECT_FWD_LIKE(T, _47),
        REFLECT_FWD_LIKE(T, _48),
        REFLECT_FWD_LIKE(T, _49),
        REFLECT_FWD_LIKE(T, _50),
        REFLECT_FWD_LIKE(T, _51),
        REFLECT_FWD_LIKE(T, _52),
        REFLECT_FWD_LIKE(T, _53),
        REFLECT_FWD_LIKE(T, _54),
        REFLECT_FWD_LIKE(T, _55),
        REFLECT_FWD_LIKE(T, _56),
        REFLECT_FWD_LIKE(T, _57),
        REFLECT_FWD_LIKE(T, _58));
}
template <class Fn, class T>
[[nodiscard]] constexpr decltype(auto) visit(Fn&& fn, T&& t, std::integral_constant<std::size_t, 59>) noexcept {
    auto&& [_1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16, _17, _18, _19, _20, _21, _22, _23, _24, _25, _26, _27, _28, _29, _30, _31, _32, _33, _34, _35, _36, _37, _38, _39, _40, _41, _42, _43, _44, _45, _46, _47, _48, _49, _50, _51, _52, _53, _54, _55, _56, _57, _58, _59] =
        REFLECT_FWD(t);
    return REFLECT_FWD(fn)(
        REFLECT_FWD_LIKE(T, _1),
        REFLECT_FWD_LIKE(T, _2),
        REFLECT_FWD_LIKE(T, _3),
        REFLECT_FWD_LIKE(T, _4),
        REFLECT_FWD_LIKE(T, _5),
        REFLECT_FWD_LIKE(T, _6),
        REFLECT_FWD_LIKE(T, _7),
        REFLECT_FWD_LIKE(T, _8),
        REFLECT_FWD_LIKE(T, _9),
        REFLECT_FWD_LIKE(T, _10),
        REFLECT_FWD_LIKE(T, _11),
        REFLECT_FWD_LIKE(T, _12),
        REFLECT_FWD_LIKE(T, _13),
        REFLECT_FWD_LIKE(T, _14),
        REFLECT_FWD_LIKE(T, _15),
        REFLECT_FWD_LIKE(T, _16),
        REFLECT_FWD_LIKE(T, _17),
        REFLECT_FWD_LIKE(T, _18),
        REFLECT_FWD_LIKE(T, _19),
        REFLECT_FWD_LIKE(T, _20),
        REFLECT_FWD_LIKE(T, _21),
        REFLECT_FWD_LIKE(T, _22),
        REFLECT_FWD_LIKE(T, _23),
        REFLECT_FWD_LIKE(T, _24),
        REFLECT_FWD_LIKE(T, _25),
        REFLECT_FWD_LIKE(T, _26),
        REFLECT_FWD_LIKE(T, _27),
        REFLECT_FWD_LIKE(T, _28),
        REFLECT_FWD_LIKE(T, _29),
        REFLECT_FWD_LIKE(T, _30),
        REFLECT_FWD_LIKE(T, _31),
        REFLECT_FWD_LIKE(T, _32),
        REFLECT_FWD_LIKE(T, _33),
        REFLECT_FWD_LIKE(T, _34),
        REFLECT_FWD_LIKE(T, _35),
        REFLECT_FWD_LIKE(T, _36),
        REFLECT_FWD_LIKE(T, _37),
        REFLECT_FWD_LIKE(T, _38),
        REFLECT_FWD_LIKE(T, _39),
        REFLECT_FWD_LIKE(T, _40),
        REFLECT_FWD_LIKE(T, _41),
        REFLECT_FWD_LIKE(T, _42),
        REFLECT_FWD_LIKE(T, _43),
        REFLECT_FWD_LIKE(T, _44),
        REFLECT_FWD_LIKE(T, _45),
        REFLECT_FWD_LIKE(T, _46),
        REFLECT_FWD_LIKE(T, _47),
        REFLECT_FWD_LIKE(T, _48),
        REFLECT_FWD_LIKE(T, _49),
        REFLECT_FWD_LIKE(T, _50),
        REFLECT_FWD_LIKE(T, _51),
        REFLECT_FWD_LIKE(T, _52),
        REFLECT_FWD_LIKE(T, _53),
        REFLECT_FWD_LIKE(T, _54),
        REFLECT_FWD_LIKE(T, _55),
        REFLECT_FWD_LIKE(T, _56),
        REFLECT_FWD_LIKE(T, _57),
        REFLECT_FWD_LIKE(T, _58),
        REFLECT_FWD_LIKE(T, _59));
}
template <class Fn, class T>
[[nodiscard]] constexpr decltype(auto) visit(Fn&& fn, T&& t, std::integral_constant<std::size_t, 60>) noexcept {
    auto&& [_1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16, _17, _18, _19, _20, _21, _22, _23, _24, _25, _26, _27, _28, _29, _30, _31, _32, _33, _34, _35, _36, _37, _38, _39, _40, _41, _42, _43, _44, _45, _46, _47, _48, _49, _50, _51, _52, _53, _54, _55, _56, _57, _58, _59, _60] =
        REFLECT_FWD(t);
    return REFLECT_FWD(fn)(
        REFLECT_FWD_LIKE(T, _1),
        REFLECT_FWD_LIKE(T, _2),
        REFLECT_FWD_LIKE(T, _3),
        REFLECT_FWD_LIKE(T, _4),
        REFLECT_FWD_LIKE(T, _5),
        REFLECT_FWD_LIKE(T, _6),
        REFLECT_FWD_LIKE(T, _7),
        REFLECT_FWD_LIKE(T, _8),
        REFLECT_FWD_LIKE(T, _9),
        REFLECT_FWD_LIKE(T, _10),
        REFLECT_FWD_LIKE(T, _11),
        REFLECT_FWD_LIKE(T, _12),
        REFLECT_FWD_LIKE(T, _13),
        REFLECT_FWD_LIKE(T, _14),
        REFLECT_FWD_LIKE(T, _15),
        REFLECT_FWD_LIKE(T, _16),
        REFLECT_FWD_LIKE(T, _17),
        REFLECT_FWD_LIKE(T, _18),
        REFLECT_FWD_LIKE(T, _19),
        REFLECT_FWD_LIKE(T, _20),
        REFLECT_FWD_LIKE(T, _21),
        REFLECT_FWD_LIKE(T, _22),
        REFLECT_FWD_LIKE(T, _23),
        REFLECT_FWD_LIKE(T, _24),
        REFLECT_FWD_LIKE(T, _25),
        REFLECT_FWD_LIKE(T, _26),
        REFLECT_FWD_LIKE(T, _27),
        REFLECT_FWD_LIKE(T, _28),
        REFLECT_FWD_LIKE(T, _29),
        REFLECT_FWD_LIKE(T, _30),
        REFLECT_FWD_LIKE(T, _31),
        REFLECT_FWD_LIKE(T, _32),
        REFLECT_FWD_LIKE(T, _33),
        REFLECT_FWD_LIKE(T, _34),
        REFLECT_FWD_LIKE(T, _35),
        REFLECT_FWD_LIKE(T, _36),
        REFLECT_FWD_LIKE(T, _37),
        REFLECT_FWD_LIKE(T, _38),
        REFLECT_FWD_LIKE(T, _39),
        REFLECT_FWD_LIKE(T, _40),
        REFLECT_FWD_LIKE(T, _41),
        REFLECT_FWD_LIKE(T, _42),
        REFLECT_FWD_LIKE(T, _43),
        REFLECT_FWD_LIKE(T, _44),
        REFLECT_FWD_LIKE(T, _45),
        REFLECT_FWD_LIKE(T, _46),
        REFLECT_FWD_LIKE(T, _47),
        REFLECT_FWD_LIKE(T, _48),
        REFLECT_FWD_LIKE(T, _49),
        REFLECT_FWD_LIKE(T, _50),
        REFLECT_FWD_LIKE(T, _51),
        REFLECT_FWD_LIKE(T, _52),
        REFLECT_FWD_LIKE(T, _53),
        REFLECT_FWD_LIKE(T, _54),
        REFLECT_FWD_LIKE(T, _55),
        REFLECT_FWD_LIKE(T, _56),
        REFLECT_FWD_LIKE(T, _57),
        REFLECT_FWD_LIKE(T, _58),
        REFLECT_FWD_LIKE(T, _59),
        REFLECT_FWD_LIKE(T, _60));
}
template <class Fn, class T>
[[nodiscard]] constexpr decltype(auto) visit(Fn&& fn, T&& t, std::integral_constant<std::size_t, 61>) noexcept {
    auto&& [_1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16, _17, _18, _19, _20, _21, _22, _23, _24, _25, _26, _27, _28, _29, _30, _31, _32, _33, _34, _35, _36, _37, _38, _39, _40, _41, _42, _43, _44, _45, _46, _47, _48, _49, _50, _51, _52, _53, _54, _55, _56, _57, _58, _59, _60, _61] =
        REFLECT_FWD(t);
    return REFLECT_FWD(fn)(
        REFLECT_FWD_LIKE(T, _1),
        REFLECT_FWD_LIKE(T, _2),
        REFLECT_FWD_LIKE(T, _3),
        REFLECT_FWD_LIKE(T, _4),
        REFLECT_FWD_LIKE(T, _5),
        REFLECT_FWD_LIKE(T, _6),
        REFLECT_FWD_LIKE(T, _7),
        REFLECT_FWD_LIKE(T, _8),
        REFLECT_FWD_LIKE(T, _9),
        REFLECT_FWD_LIKE(T, _10),
        REFLECT_FWD_LIKE(T, _11),
        REFLECT_FWD_LIKE(T, _12),
        REFLECT_FWD_LIKE(T, _13),
        REFLECT_FWD_LIKE(T, _14),
        REFLECT_FWD_LIKE(T, _15),
        REFLECT_FWD_LIKE(T, _16),
        REFLECT_FWD_LIKE(T, _17),
        REFLECT_FWD_LIKE(T, _18),
        REFLECT_FWD_LIKE(T, _19),
        REFLECT_FWD_LIKE(T, _20),
        REFLECT_FWD_LIKE(T, _21),
        REFLECT_FWD_LIKE(T, _22),
        REFLECT_FWD_LIKE(T, _23),
        REFLECT_FWD_LIKE(T, _24),
        REFLECT_FWD_LIKE(T, _25),
        REFLECT_FWD_LIKE(T, _26),
        REFLECT_FWD_LIKE(T, _27),
        REFLECT_FWD_LIKE(T, _28),
        REFLECT_FWD_LIKE(T, _29),
        REFLECT_FWD_LIKE(T, _30),
        REFLECT_FWD_LIKE(T, _31),
        REFLECT_FWD_LIKE(T, _32),
        REFLECT_FWD_LIKE(T, _33),
        REFLECT_FWD_LIKE(T, _34),
        REFLECT_FWD_LIKE(T, _35),
        REFLECT_FWD_LIKE(T, _36),
        REFLECT_FWD_LIKE(T, _37),
        REFLECT_FWD_LIKE(T, _38),
        REFLECT_FWD_LIKE(T, _39),
        REFLECT_FWD_LIKE(T, _40),
        REFLECT_FWD_LIKE(T, _41),
        REFLECT_FWD_LIKE(T, _42),
        REFLECT_FWD_LIKE(T, _43),
        REFLECT_FWD_LIKE(T, _44),
        REFLECT_FWD_LIKE(T, _45),
        REFLECT_FWD_LIKE(T, _46),
        REFLECT_FWD_LIKE(T, _47),
        REFLECT_FWD_LIKE(T, _48),
        REFLECT_FWD_LIKE(T, _49),
        REFLECT_FWD_LIKE(T, _50),
        REFLECT_FWD_LIKE(T, _51),
        REFLECT_FWD_LIKE(T, _52),
        REFLECT_FWD_LIKE(T, _53),
        REFLECT_FWD_LIKE(T, _54),
        REFLECT_FWD_LIKE(T, _55),
        REFLECT_FWD_LIKE(T, _56),
        REFLECT_FWD_LIKE(T, _57),
        REFLECT_FWD_LIKE(T, _58),
        REFLECT_FWD_LIKE(T, _59),
        REFLECT_FWD_LIKE(T, _60),
        REFLECT_FWD_LIKE(T, _61));
}
template <class Fn, class T>
[[nodiscard]] constexpr decltype(auto) visit(Fn&& fn, T&& t, std::integral_constant<std::size_t, 62>) noexcept {
    auto&& [_1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16, _17, _18, _19, _20, _21, _22, _23, _24, _25, _26, _27, _28, _29, _30, _31, _32, _33, _34, _35, _36, _37, _38, _39, _40, _41, _42, _43, _44, _45, _46, _47, _48, _49, _50, _51, _52, _53, _54, _55, _56, _57, _58, _59, _60, _61, _62] =
        REFLECT_FWD(t);
    return REFLECT_FWD(fn)(
        REFLECT_FWD_LIKE(T, _1),
        REFLECT_FWD_LIKE(T, _2),
        REFLECT_FWD_LIKE(T, _3),
        REFLECT_FWD_LIKE(T, _4),
        REFLECT_FWD_LIKE(T, _5),
        REFLECT_FWD_LIKE(T, _6),
        REFLECT_FWD_LIKE(T, _7),
        REFLECT_FWD_LIKE(T, _8),
        REFLECT_FWD_LIKE(T, _9),
        REFLECT_FWD_LIKE(T, _10),
        REFLECT_FWD_LIKE(T, _11),
        REFLECT_FWD_LIKE(T, _12),
        REFLECT_FWD_LIKE(T, _13),
        REFLECT_FWD_LIKE(T, _14),
        REFLECT_FWD_LIKE(T, _15),
        REFLECT_FWD_LIKE(T, _16),
        REFLECT_FWD_LIKE(T, _17),
        REFLECT_FWD_LIKE(T, _18),
        REFLECT_FWD_LIKE(T, _19),
        REFLECT_FWD_LIKE(T, _20),
        REFLECT_FWD_LIKE(T, _21),
        REFLECT_FWD_LIKE(T, _22),
        REFLECT_FWD_LIKE(T, _23),
        REFLECT_FWD_LIKE(T, _24),
        REFLECT_FWD_LIKE(T, _25),
        REFLECT_FWD_LIKE(T, _26),
        REFLECT_FWD_LIKE(T, _27),
        REFLECT_FWD_LIKE(T, _28),
        REFLECT_FWD_LIKE(T, _29),
        REFLECT_FWD_LIKE(T, _30),
        REFLECT_FWD_LIKE(T, _31),
        REFLECT_FWD_LIKE(T, _32),
        REFLECT_FWD_LIKE(T, _33),
        REFLECT_FWD_LIKE(T, _34),
        REFLECT_FWD_LIKE(T, _35),
        REFLECT_FWD_LIKE(T, _36),
        REFLECT_FWD_LIKE(T, _37),
        REFLECT_FWD_LIKE(T, _38),
        REFLECT_FWD_LIKE(T, _39),
        REFLECT_FWD_LIKE(T, _40),
        REFLECT_FWD_LIKE(T, _41),
        REFLECT_FWD_LIKE(T, _42),
        REFLECT_FWD_LIKE(T, _43),
        REFLECT_FWD_LIKE(T, _44),
        REFLECT_FWD_LIKE(T, _45),
        REFLECT_FWD_LIKE(T, _46),
        REFLECT_FWD_LIKE(T, _47),
        REFLECT_FWD_LIKE(T, _48),
        REFLECT_FWD_LIKE(T, _49),
        REFLECT_FWD_LIKE(T, _50),
        REFLECT_FWD_LIKE(T, _51),
        REFLECT_FWD_LIKE(T, _52),
        REFLECT_FWD_LIKE(T, _53),
        REFLECT_FWD_LIKE(T, _54),
        REFLECT_FWD_LIKE(T, _55),
        REFLECT_FWD_LIKE(T, _56),
        REFLECT_FWD_LIKE(T, _57),
        REFLECT_FWD_LIKE(T, _58),
        REFLECT_FWD_LIKE(T, _59),
        REFLECT_FWD_LIKE(T, _60),
        REFLECT_FWD_LIKE(T, _61),
        REFLECT_FWD_LIKE(T, _62));
}
template <class Fn, class T>
[[nodiscard]] constexpr decltype(auto) visit(Fn&& fn, T&& t, std::integral_constant<std::size_t, 63>) noexcept {
    auto&& [_1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16, _17, _18, _19, _20, _21, _22, _23, _24, _25, _26, _27, _28, _29, _30, _31, _32, _33, _34, _35, _36, _37, _38, _39, _40, _41, _42, _43, _44, _45, _46, _47, _48, _49, _50, _51, _52, _53, _54, _55, _56, _57, _58, _59, _60, _61, _62, _63] =
        REFLECT_FWD(t);
    return REFLECT_FWD(fn)(
        REFLECT_FWD_LIKE(T, _1),
        REFLECT_FWD_LIKE(T, _2),
        REFLECT_FWD_LIKE(T, _3),
        REFLECT_FWD_LIKE(T, _4),
        REFLECT_FWD_LIKE(T, _5),
        REFLECT_FWD_LIKE(T, _6),
        REFLECT_FWD_LIKE(T, _7),
        REFLECT_FWD_LIKE(T, _8),
        REFLECT_FWD_LIKE(T, _9),
        REFLECT_FWD_LIKE(T, _10),
        REFLECT_FWD_LIKE(T, _11),
        REFLECT_FWD_LIKE(T, _12),
        REFLECT_FWD_LIKE(T, _13),
        REFLECT_FWD_LIKE(T, _14),
        REFLECT_FWD_LIKE(T, _15),
        REFLECT_FWD_LIKE(T, _16),
        REFLECT_FWD_LIKE(T, _17),
        REFLECT_FWD_LIKE(T, _18),
        REFLECT_FWD_LIKE(T, _19),
        REFLECT_FWD_LIKE(T, _20),
        REFLECT_FWD_LIKE(T, _21),
        REFLECT_FWD_LIKE(T, _22),
        REFLECT_FWD_LIKE(T, _23),
        REFLECT_FWD_LIKE(T, _24),
        REFLECT_FWD_LIKE(T, _25),
        REFLECT_FWD_LIKE(T, _26),
        REFLECT_FWD_LIKE(T, _27),
        REFLECT_FWD_LIKE(T, _28),
        REFLECT_FWD_LIKE(T, _29),
        REFLECT_FWD_LIKE(T, _30),
        REFLECT_FWD_LIKE(T, _31),
        REFLECT_FWD_LIKE(T, _32),
        REFLECT_FWD_LIKE(T, _33),
        REFLECT_FWD_LIKE(T, _34),
        REFLECT_FWD_LIKE(T, _35),
        REFLECT_FWD_LIKE(T, _36),
        REFLECT_FWD_LIKE(T, _37),
        REFLECT_FWD_LIKE(T, _38),
        REFLECT_FWD_LIKE(T, _39),
        REFLECT_FWD_LIKE(T, _40),
        REFLECT_FWD_LIKE(T, _41),
        REFLECT_FWD_LIKE(T, _42),
        REFLECT_FWD_LIKE(T, _43),
        REFLECT_FWD_LIKE(T, _44),
        REFLECT_FWD_LIKE(T, _45),
        REFLECT_FWD_LIKE(T, _46),
        REFLECT_FWD_LIKE(T, _47),
        REFLECT_FWD_LIKE(T, _48),
        REFLECT_FWD_LIKE(T, _49),
        REFLECT_FWD_LIKE(T, _50),
        REFLECT_FWD_LIKE(T, _51),
        REFLECT_FWD_LIKE(T, _52),
        REFLECT_FWD_LIKE(T, _53),
        REFLECT_FWD_LIKE(T, _54),
        REFLECT_FWD_LIKE(T, _55),
        REFLECT_FWD_LIKE(T, _56),
        REFLECT_FWD_LIKE(T, _57),
        REFLECT_FWD_LIKE(T, _58),
        REFLECT_FWD_LIKE(T, _59),
        REFLECT_FWD_LIKE(T, _60),
        REFLECT_FWD_LIKE(T, _61),
        REFLECT_FWD_LIKE(T, _62),
        REFLECT_FWD_LIKE(T, _63));
}
template <class Fn, class T>
[[nodiscard]] constexpr decltype(auto) visit(Fn&& fn, T&& t, std::integral_constant<std::size_t, 64>) noexcept {
    auto&& [_1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16, _17, _18, _19, _20, _21, _22, _23, _24, _25, _26, _27, _28, _29, _30, _31, _32, _33, _34, _35, _36, _37, _38, _39, _40, _41, _42, _43, _44, _45, _46, _47, _48, _49, _50, _51, _52, _53, _54, _55, _56, _57, _58, _59, _60, _61, _62, _63, _64] = REFLECT_FWD(
        t);
    return REFLECT_FWD(fn)(
        REFLECT_FWD_LIKE(T, _1),
        REFLECT_FWD_LIKE(T, _2),
        REFLECT_FWD_LIKE(T, _3),
        REFLECT_FWD_LIKE(T, _4),
        REFLECT_FWD_LIKE(T, _5),
        REFLECT_FWD_LIKE(T, _6),
        REFLECT_FWD_LIKE(T, _7),
        REFLECT_FWD_LIKE(T, _8),
        REFLECT_FWD_LIKE(T, _9),
        REFLECT_FWD_LIKE(T, _10),
        REFLECT_FWD_LIKE(T, _11),
        REFLECT_FWD_LIKE(T, _12),
        REFLECT_FWD_LIKE(T, _13),
        REFLECT_FWD_LIKE(T, _14),
        REFLECT_FWD_LIKE(T, _15),
        REFLECT_FWD_LIKE(T, _16),
        REFLECT_FWD_LIKE(T, _17),
        REFLECT_FWD_LIKE(T, _18),
        REFLECT_FWD_LIKE(T, _19),
        REFLECT_FWD_LIKE(T, _20),
        REFLECT_FWD_LIKE(T, _21),
        REFLECT_FWD_LIKE(T, _22),
        REFLECT_FWD_LIKE(T, _23),
        REFLECT_FWD_LIKE(T, _24),
        REFLECT_FWD_LIKE(T, _25),
        REFLECT_FWD_LIKE(T, _26),
        REFLECT_FWD_LIKE(T, _27),
        REFLECT_FWD_LIKE(T, _28),
        REFLECT_FWD_LIKE(T, _29),
        REFLECT_FWD_LIKE(T, _30),
        REFLECT_FWD_LIKE(T, _31),
        REFLECT_FWD_LIKE(T, _32),
        REFLECT_FWD_LIKE(T, _33),
        REFLECT_FWD_LIKE(T, _34),
        REFLECT_FWD_LIKE(T, _35),
        REFLECT_FWD_LIKE(T, _36),
        REFLECT_FWD_LIKE(T, _37),
        REFLECT_FWD_LIKE(T, _38),
        REFLECT_FWD_LIKE(T, _39),
        REFLECT_FWD_LIKE(T, _40),
        REFLECT_FWD_LIKE(T, _41),
        REFLECT_FWD_LIKE(T, _42),
        REFLECT_FWD_LIKE(T, _43),
        REFLECT_FWD_LIKE(T, _44),
        REFLECT_FWD_LIKE(T, _45),
        REFLECT_FWD_LIKE(T, _46),
        REFLECT_FWD_LIKE(T, _47),
        REFLECT_FWD_LIKE(T, _48),
        REFLECT_FWD_LIKE(T, _49),
        REFLECT_FWD_LIKE(T, _50),
        REFLECT_FWD_LIKE(T, _51),
        REFLECT_FWD_LIKE(T, _52),
        REFLECT_FWD_LIKE(T, _53),
        REFLECT_FWD_LIKE(T, _54),
        REFLECT_FWD_LIKE(T, _55),
        REFLECT_FWD_LIKE(T, _56),
        REFLECT_FWD_LIKE(T, _57),
        REFLECT_FWD_LIKE(T, _58),
        REFLECT_FWD_LIKE(T, _59),
        REFLECT_FWD_LIKE(T, _60),
        REFLECT_FWD_LIKE(T, _61),
        REFLECT_FWD_LIKE(T, _62),
        REFLECT_FWD_LIKE(T, _63),
        REFLECT_FWD_LIKE(T, _64));
}
}  // namespace detail

template <class Fn, class T>
    requires std::is_aggregate_v<std::remove_cvref_t<T>>
[[nodiscard]] constexpr decltype(auto) visit(Fn&& fn, T&& t, auto...) noexcept {
#if (__cpp_structured_bindings >= 202601L)
    auto&& [... ts] = REFLECT_FWD(t);
    return REFLECT_FWD(fn)(REFLECT_FWD_LIKE(T, ts)...);
#else
    return detail::visit(
        REFLECT_FWD(fn), REFLECT_FWD(t), std::integral_constant<std::size_t, size<std::remove_cvref_t<T>>()>{});
#endif
}

template <class T>
[[nodiscard]] constexpr auto type_name() noexcept -> std::string_view {
    using type_name_info = detail::type_name_info<std::remove_pointer_t<std::remove_cvref_t<T>>>;
    constexpr std::string_view function_name = detail::function_name<std::remove_pointer_t<std::remove_cvref_t<T>>>();
    constexpr std::string_view qualified_type_name =
        function_name.substr(type_name_info::begin, function_name.find(type_name_info::end) - type_name_info::begin);
    constexpr std::string_view tmp_type_name = qualified_type_name.substr(0, qualified_type_name.find_first_of("<", 1));
    constexpr std::string_view type_name = tmp_type_name.substr(tmp_type_name.find_last_of("::") + 1);
    static_assert(std::size(type_name) > 0u);
    if (std::is_constant_evaluated()) {
        return type_name;
    } else {
        return [&] {
            static constexpr const auto name =
                fixed_string<std::remove_cvref_t<decltype(type_name[0])>, std::size(type_name)>{std::data(type_name)};
            return std::string_view{name};
        }();
    }
}

template <class T>
[[nodiscard]] constexpr auto type_name(T&&) noexcept -> std::string_view {
    return type_name<std::remove_cvref_t<T>>();
}

template <class E>
[[nodiscard]] constexpr auto to_underlying(const E e) noexcept {
    return static_cast<std::underlying_type_t<E>>(e);
}

namespace detail {
template <auto V>
consteval const auto& data() {
    return V.data;
}
template <class T, std::size_t Size>
struct static_vector {
    constexpr static_vector() = default;
    constexpr auto push_back(const T& value) { values_[size_++] = value; }
    [[nodiscard]] constexpr const auto& operator[](auto i) const { return values_[i]; }
    [[nodiscard]] constexpr auto begin() const { return &values_[0]; }
    [[nodiscard]] constexpr auto end() const { return &values_[0] + size_; }
    [[nodiscard]] constexpr auto size() const { return size_; }
    [[nodiscard]] constexpr auto empty() const { return not size_; }
    [[nodiscard]] constexpr auto capacity() const { return Size; }
    std::array<T, Size> values_{};
    std::size_t size_{};
};
#if defined(__clang__) and (__clang_major__ > 15)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wenum-constexpr-conversion"
#endif
template <class E, auto N>
    requires std::is_enum_v<E>
[[nodiscard]] constexpr auto enum_name() {
    constexpr auto fn_name = detail::function_name<static_cast<E>(N)>();
    constexpr auto name = fn_name.substr(
        detail::enum_name_info::begin,
        std::size(fn_name) - detail::enum_name_info::end - detail::enum_name_info::begin);
    constexpr auto enum_name = name.substr(name.find_last_of("::") + 1);
    static_assert(std::size(enum_name) > 0u);
    return data<fixed_string<std::remove_cvref_t<decltype(enum_name[0])>, std::size(enum_name)>{
        std::data(enum_name)}>();
}
template <class E, auto Min, auto Max>
    requires(std::is_enum_v<E> and Max > Min)
inline constexpr auto enum_cases = []<auto... Ns>(std::index_sequence<Ns...>) {
    const auto names = detail::function_name<static_cast<E>(Ns + Min)...>();
    const auto begin = detail::enum_name_info::begin;
    const auto end = std::size(names) - detail::enum_name_info::end;
    detail::static_vector<std::underlying_type_t<E>, sizeof...(Ns)> cases{};
    std::underlying_type_t<E> index{};
    auto valid = true;
    for (auto i = begin; i < end; ++i) {
        if (names[i] == '(' and names[i + 1] != ')') {
            valid = false;
        } else if (names[i] == ' ') {
            valid = true;
        } else if (names[i] == ',' or i == end - 1) {
            if (valid) {
                cases.push_back(index + Min);
            }
            ++index;
            valid = true;
        }
    }
    return cases;
}(std::make_index_sequence<Max - Min + 1 /*inclusive*/>{});
#if defined(__clang__) and (__clang_major__ > 15)
#pragma clang diagnostic pop
#endif
}  // namespace detail

template <class E>
    requires std::is_enum_v<E>
consteval auto enum_min(const E) -> int {
    return REFLECT_ENUM_MIN;
}
template <class E, class T = std::underlying_type_t<E>>
    requires std::is_enum_v<E>
consteval auto enum_max(const E) -> int {
    return std::numeric_limits<T>::max() < REFLECT_ENUM_MAX ? std::numeric_limits<T>::max() : REFLECT_ENUM_MAX;
}

template <class E, int Min = enum_min(E{}), int Max = enum_max(E{}), auto enum_cases = detail::enum_cases<E, Min, Max>>
inline constexpr auto enumerators = []<auto... Ns>(std::index_sequence<Ns...>) {
    return std::array{std::pair{enum_cases[Ns], detail::enum_name<E, enum_cases[Ns]>()}...};
}(std::make_index_sequence<enum_cases.size()>{});

template <class E, fixed_string unknown = "", auto Min = enum_min(E{}), auto Max = enum_max(E{})>
    requires(std::is_enum_v<E> and Max > Min)
[[nodiscard]] constexpr auto enum_name(const E e) noexcept -> std::string_view {
    if constexpr (constexpr auto cases = enumerators<E, Min, Max>; std::empty(cases)) {
        return unknown;
    } else {
        const auto switch_case = [&]<auto I = 0>(auto switch_case, const auto value) -> std::string_view {
            if constexpr (I == std::size(cases)) {
                return unknown;
            } else {
                switch (value) {
                    default: return switch_case.template operator()<I + 1>(switch_case, value);
                    case cases[I].first: return cases[I].second;
                }
            }
        };
        return switch_case(switch_case, to_underlying(e));
    }
}

template <class T>
[[nodiscard]] constexpr auto type_id() -> std::size_t {
    std::size_t result{};
    for (const auto c : type_name<T>()) {
        (result ^= c) <<= 1;
    }
    return result;
}

template <class T>
[[nodiscard]] constexpr auto type_id(const T&) noexcept -> std::size_t {
    return type_id<T>();
}

template <std::size_t N, class T>
    requires(std::is_aggregate_v<std::remove_cvref_t<T>> and N < size<T>())
[[nodiscard]] constexpr auto member_name() noexcept -> std::string_view {
    constexpr std::string_view function_name = detail::function_name<visit(
        [](auto&&... args) { return detail::ref{detail::nth_pack_element<N>(REFLECT_FWD(args)...)}; },
        detail::ext<std::remove_cvref_t<T>>)>();
    constexpr std::string_view tmp_member_name =
        function_name.substr(0, function_name.find(detail::member_name_info::end));
    constexpr std::string_view member_name =
        tmp_member_name.substr(tmp_member_name.find_last_of(detail::member_name_info::begin) + 1);
    static_assert(std::size(member_name) > 0u);
    if (std::is_constant_evaluated()) {
        return member_name;
    } else {
        return [&] {
            static constexpr const auto name =
                fixed_string<std::remove_cvref_t<decltype(member_name[0])>, std::size(member_name)>{
                    std::data(member_name)};
            return std::string_view{name};
        }();
    }
}

template <std::size_t N, class T>
    requires(std::is_aggregate_v<T> and N < size<T>())
[[nodiscard]] constexpr auto member_name(const T&) noexcept -> std::string_view {
    return member_name<N, T>();
}

template <std::size_t N, class T>
    requires(std::is_aggregate_v<std::remove_cvref_t<T>> and N < size<std::remove_cvref_t<T>>())
[[nodiscard]] constexpr decltype(auto) get(T&& t) noexcept {
    return visit(
        [](auto&&... args) -> decltype(auto) { return detail::nth_pack_element<N>(REFLECT_FWD(args)...); },
        REFLECT_FWD(t));
}

template <std::size_t N, class T>
    requires(std::is_aggregate_v<T> and N < size<T>())
using member_type = std::remove_cvref_t<decltype(reflect::get<N>(std::declval<T&&>()))>;

namespace detail {
template <class T, auto Name>
inline constexpr bool has_member_name_impl = []<auto... Ns>(std::index_sequence<Ns...>) {
    return ((Name == member_name<Ns, T>()) or ...);
}(std::make_index_sequence<size<T>()>{});

template <class T, fixed_string Name>
[[nodiscard]] consteval auto diagnose_member_name() {
    constexpr std::string_view prefix = "`";
    constexpr std::string_view type = type_name<T>();
    if constexpr (size<T>()) {
        constexpr std::string_view infix1 = "` has no data member named `";
        constexpr std::string_view incorrect = Name;
        constexpr std::string_view infix2 = "`. Did you mean `";
        constexpr std::string_view correct = [] {
            constexpr auto distance = [](std::string_view correct) {
                constexpr std::string_view incorrect = Name;
                std::array<std::size_t, incorrect.size() + 1> prev;
                std::array<std::size_t, incorrect.size() + 1> curr{};
                for (decltype(prev.size()) i{}; i < prev.size(); ++i) prev[i] = i;
                for (decltype(correct.size()) i{}; i < correct.size(); ++i) {
                    curr[0] = i + 1;
                    for (decltype(incorrect.size()) j{}; j < incorrect.size(); ++j) {
                        const auto del = prev[j + 1] + 1;
                        const auto ins = curr[j] + 1;
                        const auto sub = prev[j] + (correct[i] != incorrect[j]);
                        const auto min_del_ins = del < ins ? del : ins;
                        curr[j + 1] = min_del_ins < sub ? min_del_ins : sub;
                    }
                    auto temp = curr;
                    curr = prev;
                    prev = temp;
                }
                return prev.back();
            };
            const auto member_names = []<auto... Ns>(std::index_sequence<Ns...>) {
                return std::array<std::string_view, sizeof...(Ns)>{member_name<Ns, T>()...};
            }(std::make_index_sequence<size<T>()>{});
            auto closest_name = member_names[0];
            auto min_distance = distance(closest_name);
            for (decltype(member_names.size()) n{1}; n < member_names.size(); ++n) {
                const auto nth_member_name = member_names[n];
                const auto nth_distance = distance(nth_member_name);
                if (nth_distance < min_distance) {
                    closest_name = nth_member_name;
                    min_distance = nth_distance;
                }
            }
            return closest_name;
        }();
        constexpr std::string_view suffix = "`?";
        char message
            [prefix.size() + type.size() + infix1.size() + incorrect.size() + infix2.size() + correct.size() +
             suffix.size() + 1];
        auto out = message;
        for (const auto c : prefix) *out++ = c;
        for (const auto c : type) *out++ = c;
        for (const auto c : infix1) *out++ = c;
        for (const auto c : incorrect) *out++ = c;
        for (const auto c : infix2) *out++ = c;
        for (const auto c : correct) *out++ = c;
        for (const auto c : suffix) *out++ = c;
        *out = '\0';
        return fixed_string{message};
    } else {
        constexpr std::string_view suffix = "` has no data members.";
        char message[prefix.size() + type.size() + suffix.size() + 1];
        auto out = message;
        for (const auto c : prefix) *out++ = c;
        for (const auto c : type) *out++ = c;
        for (const auto c : suffix) *out++ = c;
        *out = '\0';
        return fixed_string{message};
    }
}

template <auto Message>
struct print {
    static constexpr auto value = false;
};
template <class TMessage>
concept diagnosis = TMessage::value;
template <class T, auto Name>
concept member_name_diagnosis = diagnosis<print<diagnose_member_name<T, Name>()>>;
}  // namespace detail

template <class T, fixed_string Name>
concept has_member_name = detail::has_member_name_impl<std::remove_cvref_t<T>, Name> or
                          detail::member_name_diagnosis<std::remove_cvref_t<T>, Name>;

template <fixed_string Name, class T>
    requires(std::is_aggregate_v<std::remove_cvref_t<T>> and has_member_name<std::remove_cvref_t<T>, Name>)
constexpr decltype(auto) get(T&& t) noexcept {
    return visit(
        [](auto&&... args) -> decltype(auto) {
            constexpr auto index = []<auto... Ns>(std::index_sequence<Ns...>) {
                return (
                    ((std::string_view{Name} == member_name<Ns, std::remove_cvref_t<T>>()) ? Ns : decltype(Ns){}) +
                    ...);
            }(std::make_index_sequence<size<std::remove_cvref_t<T>>()>{});
            return detail::nth_pack_element<index>(REFLECT_FWD(args)...);
        },
        REFLECT_FWD(t));
}

template <template <class...> class R, class T>
    requires std::is_aggregate_v<std::remove_cvref_t<T>>
[[nodiscard]] constexpr auto to(T&& t) noexcept {
    if constexpr (std::is_lvalue_reference_v<decltype(t)>) {
        return visit([](auto&&... args) { return R<decltype(REFLECT_FWD(args))...>{REFLECT_FWD(args)...}; }, t);
    } else {
        return visit([](auto&&... args) { return R{REFLECT_FWD(args)...}; }, t);
    }
}

template <fixed_string... Members, class TSrc, class TDst>
    requires(std::is_aggregate_v<TSrc> and std::is_aggregate_v<TDst>)
constexpr auto copy(const TSrc& src, TDst& dst) noexcept -> void {
    constexpr auto contains = []([[maybe_unused]] const auto name) {
        return sizeof...(Members) == 0u or ((name == std::string_view{Members}) or ...);
    };
    auto dst_view = to<std::tuple>(dst);
    [&]<auto... Ns>(std::index_sequence<Ns...>) {
        (
            [&] {
                if constexpr (contains(member_name<Ns, TDst>()) and requires {
                                  std::get<Ns>(dst_view) = get<fixed_string<
                                      std::remove_cvref_t<decltype(member_name<Ns, TDst>()[0])>,
                                      std::size(member_name<Ns, TDst>())>(std::data(member_name<Ns, TDst>()))>(src);
                              }) {
                    std::get<Ns>(dst_view) = get<fixed_string<
                        std::remove_cvref_t<decltype(member_name<Ns, TDst>()[0])>,
                        std::size(member_name<Ns, TDst>())>(std::data(member_name<Ns, TDst>()))>(src);
                }
            }(),
            ...);
    }(std::make_index_sequence<size<TDst>()>{});
}

template <class R, class T>
    requires std::is_aggregate_v<std::remove_cvref_t<T>>
[[nodiscard]] constexpr auto to(T&& t) noexcept {
    R r{};
    copy(REFLECT_FWD(t), r);
    return r;
}

template <std::size_t N, class T>
    requires std::is_aggregate_v<std::remove_cvref_t<T>>
[[nodiscard]] constexpr auto size_of() -> std::size_t {
    return sizeof(std::remove_cvref_t<decltype(get<N>(detail::ext<T>))>);
}

template <std::size_t N, class T>
    requires std::is_aggregate_v<std::remove_cvref_t<T>>
[[nodiscard]] constexpr auto size_of(T&&) -> std::size_t {
    return size_of<N, std::remove_cvref_t<T>>();
}

template <std::size_t N, class T>
    requires std::is_aggregate_v<std::remove_cvref_t<T>>
[[nodiscard]] constexpr auto align_of() -> std::size_t {
    return alignof(std::remove_cvref_t<decltype(get<N>(detail::ext<T>))>);
}

template <std::size_t N, class T>
    requires std::is_aggregate_v<std::remove_cvref_t<T>>
[[nodiscard]] constexpr auto align_of(T&&) -> std::size_t {
    return align_of<N, std::remove_cvref_t<T>>();
}

template <std::size_t N, class T>
    requires std::is_aggregate_v<std::remove_cvref_t<T>>
[[nodiscard]] constexpr auto offset_of() -> std::size_t {
    if constexpr (not N) {
        return {};
    } else {
        constexpr auto offset = offset_of<N - 1, T>() + size_of<N - 1, T>();
        constexpr auto alignment = std::min(alignof(T), align_of<N, T>());
        constexpr auto padding = offset % alignment;
        return offset + padding;
    }
}

template <std::size_t N, class T>
    requires std::is_aggregate_v<std::remove_cvref_t<T>>
[[nodiscard]] constexpr auto offset_of(T&&) -> std::size_t {
    return offset_of<N, std::remove_cvref_t<T>>();
}

template <class T, class Fn>
    requires std::is_aggregate_v<std::remove_cvref_t<T>>
constexpr auto for_each(Fn&& fn) -> void {
    [&]<auto... Ns>(std::index_sequence<Ns...>) {
        (REFLECT_FWD(fn)(std::integral_constant<decltype(Ns), Ns>{}), ...);
    }(std::make_index_sequence<reflect::size<std::remove_cvref_t<T>>()>{});
}

template <class Fn, class T>
    requires std::is_aggregate_v<std::remove_cvref_t<T>>
constexpr auto for_each(Fn&& fn, T&&) -> void {
    [&]<auto... Ns>(std::index_sequence<Ns...>) {
        (REFLECT_FWD(fn)(std::integral_constant<decltype(Ns), Ns>{}), ...);
    }(std::make_index_sequence<reflect::size<std::remove_cvref_t<T>>()>{});
}
}  // namespace reflect::inline v1_2_2
#undef REFLECT_FWD_LIKE
#undef REFLECT_FWD

#ifndef NTEST
namespace reflect::test {
struct empty {};
struct foo {
    using type = int;
    enum E {};
};
template <class T>
struct optional {
    constexpr optional() = default;
    constexpr optional(T t) : t{t} {}
    T t{};
};
struct bar {};
template <class T>
struct foo_t {};
namespace ns::inline v1 {
template <auto...>
struct bar_v {};
}  // namespace ns::inline v1
enum e {
    _141 = 141,
    _142 = 142,
};
enum e1 { e1a };
enum class e2 : std::uint8_t { e2a, e2b };
enum e3 {
    e3a,
    e3b,
    e3c,
};
enum class e4 : std::int32_t {
    e4a,
    e4b = 77,
    e4c,
    e4d = 42,
};
consteval auto enum_min(e) { return e::_141; }
consteval auto enum_max(e) { return e::_142; }
void failed();
auto expect = [](bool cond) {
    if (not cond) {
        failed();
    }
};
}  // namespace reflect::test

static_assert((
    [] {
        using namespace reflect;
        using test::expect;

        // nth_pack_element
        {
            using reflect::detail::nth_pack_element;

            static_assert(1 == nth_pack_element<0>(1));
            static_assert(1 == nth_pack_element<0>(1, 2));
            static_assert(2 == nth_pack_element<1>(1, 2));
            static_assert('a' == nth_pack_element<0>('a', 1, true));
            static_assert(1 == nth_pack_element<1>('a', 1, true));
            static_assert(true == nth_pack_element<2>('a', 1, true));

            {
                [[maybe_unused]] int a{};
                static_assert(std::is_same_v<int&, decltype(nth_pack_element<0>(a))>);
            }

            {
                [[maybe_unused]] const int a{};
                static_assert(std::is_same_v<const int&, decltype(nth_pack_element<0>(a))>);
            }

            {
                [[maybe_unused]] int a{};
                static_assert(std::is_same_v<int&&, decltype(nth_pack_element<0>(static_cast<int&&>(a)))>);
            }

            {
                [[maybe_unused]] const int a{};
                static_assert(std::is_same_v<const int&&, decltype(nth_pack_element<0>(static_cast<const int&&>(a)))>);
            }
        }

        // fixed_string
        {
            static_assert(0u == std::size(fixed_string{""}));
            static_assert(fixed_string{""} == fixed_string{""});
            static_assert(std::string_view{""} == std::string_view{fixed_string{""}});
            static_assert(3u == std::size(fixed_string{"foo"}));
            static_assert(std::string_view{"foo"} == std::string_view{fixed_string{"foo"}});
            static_assert(fixed_string{"foo"} == fixed_string{"foo"});
        }

        // size
        {
            struct empty {};
            struct one {
                int a;
            };
            struct two {
                int a;
                reflect::test::optional<int> o;
            };
            struct empty_with_base : empty {};
            struct one_with_base : empty {
                int a;
            };
            struct two_with_base : empty {
                int a;
                reflect::test::optional<int> o;
            };
            struct base {};
            struct empty_with_bases : empty, base {};
            struct one_with_bases : empty, base {
                int a;
            };
            struct two_with_bases : empty, base {
                int a;
                reflect::test::optional<int> o;
            };
            struct two_with_base_member : empty, base {
                int a;
                base b;
            };

            static_assert(0 == size<empty>());
            static_assert(0 == size(empty{}));
            static_assert(1 == size<one>());
            static_assert(1 == size(one{}));
            static_assert(2 == size<two>());
            static_assert(2 == size(two{}));
            static_assert(0 == size<empty_with_base>());
            static_assert(0 == size(empty_with_base{}));
            static_assert(1 == size<one_with_base>());
            static_assert(1 == size(one_with_base{}));
            static_assert(2 == size<two_with_base>());
            static_assert(2 == size(two_with_base{}));
            static_assert(0 == size<empty_with_bases>());
            static_assert(0 == size(empty_with_bases{}));
            static_assert(1 == size<one_with_bases>());
            static_assert(1 == size(one_with_bases{}));
            static_assert(2 == size<two_with_bases>());
            static_assert(2 == size(two_with_bases{}));
            static_assert(2 == size<two_with_base_member>());
            static_assert(2 == size(two_with_base_member{}));
            static_assert(0 == size<const empty>());
            static_assert(1 == size<const one>());
            static_assert(2 == size<const two>());
            static_assert(0 == size<const empty_with_base>());
            static_assert(1 == size<const one_with_base>());
            static_assert(2 == size<const two_with_base>());
            static_assert(0 == size<const empty_with_bases>());
            static_assert(1 == size<const one_with_bases>());
            static_assert(2 == size<const two_with_bases>());
            static_assert(2 == size<const two_with_base_member>());

            struct non_standard_layout {
               private:
                int _1{};

               public:
                int _2{};
            };

            struct S {
                double _1;
                non_standard_layout _2;
                float _3;
            };

            static_assert(3 == size<S>());

            constexpr auto test = []<class T> {
                struct S {
                    T _1;
                    T _2;
                    int _3;
                    T _4;
                };
                struct S5_0 {
                    int _1;
                    int _2;
                    int _3;
                    int _4;
                    T _5;
                };
                struct S5_1 {
                    T _1;
                    int _2;
                    int _3;
                    int _4;
                    int _5;
                };
                struct S5_2 {
                    int _1;
                    int _2;
                    T _3;
                    int _4;
                    int _5;
                };
                struct S5_3 {
                    int _1;
                    int _2;
                    T _3;
                    int _4;
                    T _5;
                };
                struct S5_4 {
                    T _1;
                    T _2;
                    T _3;
                    T _4;
                    T _5;
                };
                struct S6 {
                    T _1;
                    T _2;
                    T _3;
                    T _4;
                    T _5;
                    T _6;
                };

                static_assert(4 == size<S>());
                static_assert(5 == size<S5_0>());
                static_assert(5 == size<S5_1>());
                static_assert(5 == size<S5_2>());
                static_assert(5 == size<S5_3>());
                static_assert(5 == size<S5_4>());
                static_assert(6 == size<S6>());
            };

            {
                struct T {
                    T() = default;
                    T(T&&) = default;
                    T(const T&) = delete;
                    T& operator=(T&&) = default;
                    T& operator=(const T&) = delete;
                };
                test.template operator()<T>();
            }

            {
                struct T {
                    T() = default;
                    T(T&&) = default;
                    T(const T&) = delete;
                    T& operator=(T&&) = default;
                    T& operator=(const T&) = delete;
                };
                test.template operator()<T>();
            }

            {
                struct T {
                    T(T&&) = default;
                    T(const T&) = delete;
                    T& operator=(T&&) = default;
                    T& operator=(const T&) = delete;
                };
                test.template operator()<T>();
            }

            {
                struct T {
                    T(int) {}
                };
                test.template operator()<T>();
            }

            struct bf {
                unsigned int _1 : 1;
                unsigned int _2 : 1;
                unsigned int _3 : 1;
                unsigned int _4 : 1;
                unsigned int _5 : 1;
                unsigned int _6 : 1;
            };

            static_assert(6 == size<bf>());

            struct {
                int _1;
                int _2;
            } anonymous;

            static_assert(2 == size(anonymous));
        }

        // visit
        {
            struct empty {};
            static_assert(0 == visit([]([[maybe_unused]] auto&&... args) { return sizeof...(args); }, empty{}));

            struct one {
                int a;
            };
            static_assert(1 == visit([]([[maybe_unused]] auto&&... args) { return sizeof...(args); }, one{}));

            struct two {
                int a;
                int b;
            };
            static_assert(2 == visit([]([[maybe_unused]] auto&&... args) { return sizeof...(args); }, two{}));
        }

        // type_name
        {
            struct local {};

            static_assert(std::string_view{"void"} == type_name<void>());
            static_assert(std::string_view{"int"} == type_name<int>());
            static_assert(std::string_view{"empty"} == type_name<reflect::test::empty>());
            static_assert(std::string_view{"empty"} == type_name(reflect::test::empty{}));
            static_assert(std::string_view{"foo"} == type_name<reflect::test::foo>());
            static_assert(std::string_view{"foo"} == type_name(reflect::test::foo{}));
            static_assert(std::string_view{"bar"} == type_name<reflect::test::bar>());
            static_assert(std::string_view{"bar"} == type_name(reflect::test::bar{}));
            static_assert(std::string_view{"foo_t"} == type_name<reflect::test::foo_t<void>>());
            static_assert(std::string_view{"foo_t"} == type_name<reflect::test::foo_t<int>>());
            static_assert(std::string_view{"foo_t"} == type_name<reflect::test::foo_t<reflect::test::ns::bar_v<42>>>());
            static_assert(std::string_view{"bar_v"} == type_name<reflect::test::ns::bar_v<42>>());
            static_assert(std::string_view{"bar_v"} == type_name<reflect::test::ns::bar_v<>>());
            static_assert(std::string_view{"int"} == type_name(reflect::test::foo::type{}));
            static_assert(std::string_view{"E"} == type_name(reflect::test::foo::E{}));
            static_assert(std::string_view{"local"} == type_name<local>());
            static_assert(std::string_view{"local"} == type_name(local{}));
        }

        // type_id
        {
            static_assert(type_id<reflect::test::bar>() != type_id(reflect::test::foo{}));
            static_assert(type_id<int>() != type_id(reflect::test::foo{}));
            static_assert(type_id<int>() != type_id<void>());
            static_assert(type_id<void>() != type_id<int>());
            static_assert(type_id<void>() == type_id<void>());
            static_assert(type_id<int>() == type_id<int>());
            static_assert(type_id<int>() == type_id<int&>());
            static_assert(type_id<const int&>() == type_id<int&>());
            static_assert(type_id<void*>() == type_id<void>());
            static_assert(type_id<void*>() == type_id<void>());
            static_assert(type_id<reflect::test::foo>() == type_id(reflect::test::foo{}));
            static_assert(type_id<reflect::test::bar>() == type_id(reflect::test::bar{}));
        }

#ifndef _MSC_VER
        // enumerators
        {
            static_assert(1u == enumerators<reflect::test::e1>.size());
            static_assert(0u == enumerators<reflect::test::e1>[0].first);

            static_assert(2u == enumerators<reflect::test::e2>.size());
            static_assert(std::uint8_t(0) == enumerators<reflect::test::e2>[0].first);
            static_assert(std::uint8_t(1) == enumerators<reflect::test::e2>[1].first);

            static_assert(3u == enumerators<reflect::test::e3>.size());
            static_assert(0u == enumerators<reflect::test::e3>[0].first);
            static_assert(1u == enumerators<reflect::test::e3>[1].first);
            static_assert(2u == enumerators<reflect::test::e3>[2].first);

            static_assert(4u == enumerators<reflect::test::e4>.size());
            static_assert(0 == enumerators<reflect::test::e4>[0].first);
            static_assert(42 == enumerators<reflect::test::e4>[1].first);
            static_assert(77 == enumerators<reflect::test::e4>[2].first);
            static_assert(78 == enumerators<reflect::test::e4>[3].first);
        }
#endif

        // enum_name
        {
            enum class foobar { foo = 1, bar = 2 };

            enum mask : unsigned char {
                a = 0b00,
                b = 0b01,
                c = 0b10,
            };

            enum sparse {
                _128 = 128,
                _130 = 130,
            };

            enum class negative {
                unknown = -1,
                A = 0,
                B = 1,
            };

            static_assert([](const auto e) { return requires { enum_name<foobar, "", 1, 2>(e); }; }(foobar::foo));
            static_assert([](const auto e) { return requires { enum_name<mask, "", 1, 2>(e); }; }(mask::a));
            static_assert(not [](const auto e) { return requires { enum_name<foobar, "", 1, 2>(e); }; }(0));
            static_assert(not [](const auto e) { return requires { enum_name<int, "", 1, 2>(e); }; }(0));
            static_assert(not [](const auto e) { return requires { enum_name<int, "", 1, 2>(e); }; }(42u));

            static_assert(std::string_view{""} == enum_name<foobar, "", 1, 2>(static_cast<foobar>(42)));
            static_assert(std::string_view{"unknown"} == enum_name<foobar, "unknown", 1, 2>(static_cast<foobar>(42)));

            const auto e = foobar::foo;
            static_assert(std::string_view{"foo"} == enum_name<foobar, "", 1, 2>(e));

            static_assert(std::string_view{"foo"} == enum_name<foobar, "", 1, 2>(foobar::foo));
            static_assert(std::string_view{"bar"} == enum_name<foobar, "", 1, 2>(foobar::bar));

            static_assert(std::string_view{"a"} == enum_name<mask, "", 0, 2>(mask::a));
            static_assert(std::string_view{"b"} == enum_name<mask, "", 0, 2>(mask::b));
            static_assert(std::string_view{"c"} == enum_name<mask, "", 0, 2>(mask::c));

            static_assert(std::string_view{"_128"} == enum_name<sparse, "", 128, 130>(sparse::_128));
            static_assert(std::string_view{"_130"} == enum_name<sparse, "", 128, 130>(sparse::_130));

            static_assert(std::string_view{"unknown"} == enum_name<negative, "<>", -1, 1>(negative::unknown));
            static_assert(std::string_view{"A"} == enum_name<negative, "<>", -1, 1>(negative::A));
            static_assert(std::string_view{"B"} == enum_name<negative, "<>", -1, 1>(negative::B));
            static_assert(std::string_view{"<>"} == enum_name<negative, "<>", -1, 1>(static_cast<negative>(-42)));

            static_assert(std::string_view{"_141"} == enum_name<reflect::test::e>(reflect::test::e::_141));
            static_assert(std::string_view{"_142"} == enum_name<reflect::test::e>(reflect::test::e::_142));
        }

        // member_name
        {
            struct foo {
                int i;
                bool b;
                void* bar{};
            };

            static_assert(std::string_view{"i"} == member_name<0, foo>());
            static_assert(std::string_view{"i"} == member_name<0>(foo{}));

            static_assert(std::string_view{"b"} == member_name<1, foo>());
            static_assert(std::string_view{"b"} == member_name<1>(foo{}));

            static_assert(std::string_view{"bar"} == member_name<2, foo>());
            static_assert(std::string_view{"bar"} == member_name<2>(foo{}));
        }

        // get [by index]
        {
            struct foo {
                int i;
                bool b;
            };

            {
                constexpr auto f = foo{.i = 42, .b = true};

                static_assert([]<auto N> { return requires { get<N>(f); }; }.template operator()<0>());
                static_assert([]<auto N> { return requires { get<N>(f); }; }.template operator()<1>());
                static_assert(not []<auto N> { return requires { get<N>(f); }; }.template operator()<2>());

                static_assert(42 == get<0>(f));
                static_assert(true == get<1>(f));
            }

            {
                {
                    auto f = foo{};
                    auto value = get<0>(f);
                    static_assert(std::is_same_v<decltype(value), int>);
                }

                {
                    auto value = get<0>(foo{});
                    static_assert(std::is_same_v<decltype(value), int>);
                }

                {
                    auto f = foo{};
                    auto& lvalue = get<0>(f);
                    static_assert(std::is_same_v<decltype(lvalue), int&>);
                }

                { static_assert(std::is_same_v<decltype(get<0>(foo{})), int&&>); }
            }
        }

        // member_type
        {
            struct foo {
                int i;
                float j;
                bool k;
            };

            static_assert(std::is_same_v<int, member_type<0, foo>>);
            static_assert(std::is_same_v<float, member_type<1, foo>>);
            static_assert(std::is_same_v<bool, member_type<2, foo>>);
        }

        // has_member_name
        {
            struct foo {
                int bar;
            };
            static_assert(has_member_name<foo, "bar">);
            static_assert(not has_member_name<foo, "baz">);
            static_assert(not has_member_name<foo, "BAR">);
            static_assert(not has_member_name<foo, "">);
        }

        // diagnose_member_name
        {
            struct foo {
                bool flag;
                int bar;
            };
            static_assert(
                detail::diagnose_member_name<foo, "">() == "`foo` has no data member named ``. Did you mean `bar`?");
            static_assert(
                detail::diagnose_member_name<foo, "ba">() ==
                "`foo` has no data member named `ba`. Did you mean `bar`?");
            static_assert(
                detail::diagnose_member_name<foo, "fl">() ==
                "`foo` has no data member named `fl`. Did you mean `flag`?");
            static_assert(
                detail::diagnose_member_name<foo, "baz">() ==
                "`foo` has no data member named `baz`. Did you mean `bar`?");
            static_assert(
                detail::diagnose_member_name<foo, "lag">() ==
                "`foo` has no data member named `lag`. Did you mean `flag`?");

            struct empty {};
            static_assert(detail::diagnose_member_name<empty, "any">() == "`empty` has no data members.");
        }

        // get [by name]
        {
            struct foo {
                int i;
                bool b;
            };

            {
                constexpr auto f = foo{.i = 42, .b = true};

                static_assert([]<fixed_string Name> { return requires { get<Name>(f); }; }.template operator()<"i">());
                static_assert([]<fixed_string Name> { return requires { get<Name>(f); }; }.template operator()<"b">());
                static_assert(
                    not []<fixed_string Name> { return requires { get<Name>(f); }; }.template operator()<"unknown">());

                static_assert(42 == get<"i">(f));
                static_assert(true == get<"b">(f));
            }

            {
                {
                    auto f = foo{};
                    auto value = get<"i">(f);
                    static_assert(std::is_same_v<decltype(value), int>);
                }

                {
                    auto value = get<"i">(foo{});
                    static_assert(std::is_same_v<decltype(value), int>);
                }

                {
                    auto f = foo{};
                    auto& lvalue = get<"i">(f);
                    static_assert(std::is_same_v<decltype(lvalue), int&>);
                }

                { static_assert(std::is_same_v<decltype(get<"i">(foo{})), int&&>); }
            }
        }

        // to
        {
            struct foo {
                int a;
                int b;
            };

            {
                constexpr auto t = to<std::tuple>(foo{.a = 4, .b = 2});
                static_assert(4 == std::get<0>(t));
                static_assert(2 == std::get<1>(t));
            }

            {
                auto f = foo{.a = 4, .b = 2};
                auto t = to<std::tuple>(f);
                std::get<0>(t) *= 10;
                f.b = 42;
                expect(40 == std::get<0>(t) and 40 == f.a);
                expect(42 == std::get<1>(t) and 42 == f.b);
            }

            {
                const auto f = foo{.a = 4, .b = 2};
                auto t = to<std::tuple>(f);
                expect(f.a == std::get<0>(t));
                expect(f.b == std::get<1>(t));
            }
        }

        // copy
        {
            struct foo {
                int a{};
                int b{};
            };

            struct bar {
                int a{};
                int b{};
            };

            const auto f = foo{.a = 1, .b = 2};

            {
                bar b{};
                b.b = 42;
                copy<"a">(f, b);
                expect(b.a == f.a);
                expect(42 == b.b);
            }

            {
                bar b{};
                b.b = 42;
                copy<"a", "b">(f, b);
                expect(b.a == f.a);
                expect(b.b == f.b);
            }

            {
                bar b{};
                b.a = 42;
                copy<"b">(f, b);
                expect(42 == b.a);
                expect(b.b == f.b);
            }

            {
                bar b{};
                copy<"c">(f, b);  // ignores
                expect(0 == b.a);
                expect(0 == b.b);
            }

            {
                bar b{};
                copy<"a", "c", "b">(f, b);  // copies a, b; ignores c
                expect(b.a == f.a);
                expect(b.b == f.b);
            }

            {
                bar b{};
                copy(f, b);
                expect(b.a == f.a);
                expect(b.b == f.b);
            }

            {
                bar b{.a = 4, .b = 2};
                copy(f, b);  // overwrites members
                expect(b.a == f.a);
                expect(b.b == f.b);
            }

            struct baz {
                int a{};
                int c{};
            };

            {
                baz b{};
                b.c = 42;
                copy(f, b);
                expect(b.a == f.a);
                expect(42 == b.c);
            }
        }

        // to [struct]
        {
            struct foo {
                int a{};
                int b{};
            };

            struct bar {
                int a{};
                int b{};
            };

            {
                constexpr auto b = to<bar>(foo{.a = 4, .b = 2});
                static_assert(4 == b.a);
                static_assert(2 == b.b);
            }

            {
                auto f = foo{.a = 4, .b = 2};
                auto b = to<bar>(f);
                f.a = 42;
                expect(42 == f.a);
                expect(4 == b.a);
                expect(2 == b.b);
            }

            {
                const auto f = foo{.a = 4, .b = 2};
                const auto b = to<bar>(f);
                expect(4 == b.a);
                expect(2 == b.b);
            }

            struct baz {
                int a{};
                int c{};
            };

            {
                auto f = foo{.a = 4, .b = 2};
                auto b = to<bar>(f);
                b.a = 1;
                expect(1 == b.a and 4 == f.a);
                expect(2 == b.b and 2 == f.b);
            }

            {
                const auto b = to<baz>(foo{.a = 4, .b = 2});
                expect(4 == b.a and 0 == b.c);
            }

            struct foobar {
                int a{};
                enum e : int {} b;  // strong type, type conversion disabled
            };

            {
                const auto fb = to<foobar>(foo{.a = 4, .b = 2});
                expect(4 == fb.a and 0 == fb.b);
            }
        }

        // size_of
        {
            struct s {
                float a;
                char b;
                char bb;
                int c;
            };

            static_assert(sizeof(float) == size_of<0, s>());
            static_assert(sizeof(char) == size_of<1, s>());
            static_assert(sizeof(char) == size_of<2, s>());
            static_assert(sizeof(int) == size_of<3, s>());
        }

        // align_of
        {
            struct s {
                float a;
                char b;
                char bb;
                int c;
            };

            static_assert(alignof(float) == align_of<0, s>());
            static_assert(alignof(char) == align_of<1, s>());
            static_assert(alignof(char) == align_of<2, s>());
            static_assert(alignof(int) == align_of<3, s>());
        }

        // offset_of
        {
            struct s {
                float a;
                char b;
                char bb;
                int c;
            };

#pragma pack(push, 1)
            struct s2 {
                float a;
                char b;
                char bb;
                int c;
                double d;
                char e;
            };
#pragma pack(pop)

            struct a {
                int i;
                int j;
            };

            struct b {
                int i;
                int k;
            };

            struct al {
                char a;
                int b;
                char c;
            };

            static_assert(offset_of<0, a>() == 0);
            static_assert(offset_of<1, a>() == sizeof(int));
            static_assert(offset_of<0, b>() == 0);
            static_assert(offset_of<1, b>() == sizeof(int));
            static_assert(offset_of<0, s>() == 0);
            static_assert(offset_of<1, s>() == sizeof(float));
            static_assert(offset_of<2, s>() == sizeof(float) + sizeof(char));
            static_assert(offset_of<3, s>() == alignof(s) * 2);
            static_assert(offset_of<0, s2>() == 0);
            static_assert(offset_of<1, s2>() == sizeof(float));
            static_assert(offset_of<2, s2>() == sizeof(float) + sizeof(char));
            static_assert(offset_of<3, s2>() == sizeof(float) + sizeof(char) + sizeof(char));
            static_assert(offset_of<4, s2>() == sizeof(float) + sizeof(char) + sizeof(char) + sizeof(int));
            static_assert(
                offset_of<5, s2>() == sizeof(float) + sizeof(char) + sizeof(char) + sizeof(int) + sizeof(double));
            static_assert(offset_of<0, al>() == 0);
            static_assert(offset_of<1, al>() == sizeof(char) * 2);
            static_assert(offset_of<2, al>() == sizeof(char) * 2 + sizeof(int));
        }

        // for_each
        {
            struct foo {
                short _1;
                int _2;
            } f{._1 = 1, ._2 = 2};

            std::array<std::size_t, size(f)> size_of{};
            std::array<std::size_t, size(f)> align_of{};
            std::array<std::size_t, size(f)> offset_of{};
            std::array<std::string_view, size(f)> name{};
            std::array<std::string_view, size(f)> type{};
            std::array<std::common_type_t<short, int>, size(f)> value{};

            auto i = 0;
            for_each(
                [&](auto I) {
                    size_of[i] = reflect::size_of<I>(f);
                    align_of[i] = reflect::align_of<I>(f);
                    offset_of[i] = reflect::offset_of<I>(f);
                    name[i] = reflect::member_name<I>(f);
                    type[i] = reflect::type_name(reflect::get<I>(f));
                    value[i] = reflect::get<I>(f);
                    ++i;
                },
                f);

            expect(2 == i);
            expect(sizeof(f._1) == size_of[0] and sizeof(f._2) == size_of[1]);
            expect(alignof(short) == align_of[0] and alignof(int) == align_of[1]);
            expect(0 == offset_of[0] and 4 == offset_of[1]);
            expect(std::string_view{"_1"} == name[0] and std::string_view{"_2"} == name[1]);
            expect(f._1 == value[0] and f._2 == value[1]);
        }
    }(),
    true));
#endif  // NTEST
#endif  // REFLECT
#endif  // README
