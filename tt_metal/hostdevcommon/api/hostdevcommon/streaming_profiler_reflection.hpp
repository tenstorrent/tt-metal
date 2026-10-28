// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Compiled as C++17 for the device and C++20 for the host.
#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <utility>

#include "hostdevcommon/streaming_profiler_types.hpp"

namespace tt::tt_metal::experimental::streaming_profiler {

namespace detail {
// A type's full name as the compiler spells it: "... [with T = ns::Name; ...]" (GCC) or "... [T = ns::Name]" (clang)
// -> "ns::Name". GCC leaves out the namespace of a type that sits in the function's own namespace, so the function sits
// in a namespace that holds no type.
namespace type_naming {
template <typename T>
constexpr auto chars() {
    constexpr std::string_view pretty = __PRETTY_FUNCTION__;
    constexpr std::string_view kArgument = "T = ";
    constexpr std::size_t begin = pretty.find(kArgument) + kArgument.size();
    constexpr std::size_t end = pretty.find_first_of(";]", begin);
    std::array<char, end - begin + 1> name{};
    for (std::size_t k = begin; k < end; ++k) {
        name[k - begin] = pretty[k];
    }
    return name;
}
}  // namespace type_naming
template <typename T>
inline constexpr auto kTypeNameChars = type_naming::chars<T>();
template <typename T>
inline constexpr std::string_view type_name{kTypeNameChars<T>.data(), kTypeNameChars<T>.size() - 1};

// Only an enum with a fixed underlying type list-initializes from an integer.
template <typename E, typename = void>
inline constexpr bool is_fixed_enum = false;
template <typename E>
inline constexpr bool is_fixed_enum<E, std::void_t<decltype(E{std::underlying_type_t<E>{}})>> = true;

// A payload or metadata field's or an argument's type as a kind and a width in bits, which both compilers agree on:
// 'u', 'i' and 'f' for integers and floats, 'c' for plain char, 'b' for bool, 's' for a string in the kernel's ELF (its
// 64-bit hash on the device) and 't' for a Timestamp. An enum is its underlying integer.
constexpr uint16_t field_code(char kind, std::size_t bits) { return uint16_t(uint16_t(kind) << 8 | bits); }
template <typename T>
constexpr uint16_t type_code() {
    if constexpr (std::is_enum_v<T>) {
        static_assert(
            is_fixed_enum<T>, "enums without a fixed underlying type are not supported; use enum E : uint8_t");
        return type_code<std::underlying_type_t<T>>();
    } else if constexpr (std::is_same_v<T, String>) {
        return field_code('s', 64);
    } else if constexpr (std::is_same_v<T, bool>) {
        return field_code('b', 8);
    } else if constexpr (std::is_same_v<T, char>) {
        return field_code('c', 8);
    } else if constexpr (std::is_same_v<T, Timestamp>) {
        return field_code('t', 64);
    } else if constexpr (std::is_union_v<T>) {
        static_assert(!std::is_union_v<T>, "unions are not supported; send the member it holds");
        return 0;
    } else if constexpr (std::is_class_v<T>) {
        static_assert(
            !std::is_class_v<T>,
            "only aggregate structs, fixed-width scalars, enums, String, Timestamp and arrays of them are supported; "
            "this type has a constructor, private field or virtual function");
        return 0;
    } else if constexpr (std::is_arithmetic_v<T> && !std::is_same_v<T, long double>) {
        return field_code(std::is_floating_point_v<T> ? 'f' : std::is_signed_v<T> ? 'i' : 'u', sizeof(T) * 8);
    } else {
        static_assert(
            std::is_arithmetic_v<T> && !std::is_same_v<T, long double>,
            "this type is not supported; use a fixed-width integer, char, float, double, bool, enum, String, "
            "Timestamp, struct or array");
        return 0;
    }
}

// A payload or metadata struct's fields, which the device compiler and the host walk in the same order: the device to
// describe each field, the host to read each one. C++17 has no reflection, so fields are counted by brace
// initialization: such a struct is an aggregate with no unions.

struct AnyField {
    template <typename T>
    constexpr operator T() const;
};
template <typename T, typename = void, typename... A>
struct BraceInitializable : std::false_type {};
template <typename T, typename... A>
struct BraceInitializable<T, std::void_t<decltype(T{std::declval<A>()...})>, A...> : std::true_type {};
template <std::size_t>
using AnyFieldAt = AnyField;
template <typename T, std::size_t... I>
constexpr bool initializable(std::index_sequence<I...>) {
    return BraceInitializable<T, void, AnyFieldAt<I>...>::value;
}

// The most initializers T takes, one per field, or one per element of a C array field through brace elision: grown as
// 2n + 1 until T rejects it, then bisected. Each takes at least a bit, so T takes at most 8 * sizeof(T).
template <typename T, std::size_t kLo = 0, std::size_t kHi = 8 * sizeof(T)>
constexpr std::size_t initializer_count() {
    if constexpr (kLo == kHi) {
        return kLo;
    } else {
        constexpr std::size_t kMid = std::min(2 * kLo + 1, (kLo + kHi + 1) / 2);
        if constexpr (initializable<T>(std::make_index_sequence<kMid>{})) {
            return initializer_count<T, kMid, kHi>();
        } else {
            return initializer_count<T, kLo, kMid - 1>();
        }
    }
}

template <typename T>
inline constexpr bool is_std_array = false;
template <typename T, std::size_t N>
inline constexpr bool is_std_array<std::array<T, N>> = true;

// Whether T is Template<...>.
template <typename T, template <typename...> class Template>
inline constexpr bool is_specialization = false;
template <template <typename...> class Template, typename... Ts>
inline constexpr bool is_specialization<Template<Ts...>, Template> = true;

template <typename T>
inline constexpr bool is_leaf_type =
    std::is_arithmetic_v<T> || std::is_enum_v<T> || std::is_same_v<T, String> || std::is_same_v<T, Timestamp>;

// Aggregate initialization reaches the fields a base class declares, which structured bindings can't.
template <typename T, typename Reader>
constexpr T build(Reader& reader);

template <typename Reader>
struct FieldBuilder {
    Reader* reader;
    template <typename T>
    constexpr operator T() const {
        return build<T>(*reader);
    }
};
// A C array field takes one initializer per element, by brace elision, which clang flags with -Wmissing-braces.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmissing-braces"
template <typename T, typename Reader, std::size_t... I>
constexpr T build_fields(Reader& reader, std::index_sequence<I...>) {
    return T{(static_cast<void>(I), FieldBuilder<Reader>{&reader})...};  // evaluated left to right
}
#pragma GCC diagnostic pop
template <typename T, typename Reader, std::size_t... I>
constexpr T build_elements(Reader& reader, std::index_sequence<I...>) {
    return T{build<std::tuple_element_t<I, T>>(reader)...};
}
template <typename T, typename Reader>
constexpr T build(Reader& reader) {
    // A union or a class that isn't an aggregate goes as a leaf, whose type_code says what to do instead.
    if constexpr (
        !std::is_class_v<T> || is_leaf_type<T> || (!std::is_aggregate_v<T> && !is_specialization<T, std::tuple>)) {
        return reader.template leaf<T>();
    } else if constexpr (is_specialization<T, std::tuple>) {
        return build_elements<T>(reader, std::make_index_sequence<std::tuple_size_v<T>>{});
    } else {
        // A std::array opens and closes like a struct of its elements, which its layout is.
        reader.open();
        T built = [&] {
            if constexpr (is_std_array<T>) {
                return build_elements<T>(reader, std::make_index_sequence<std::tuple_size_v<T>>{});
            } else {
                return build_fields<T>(reader, std::make_index_sequence<initializer_count<T>()>{});
            }
        }();
        reader.close();
        return built;
    }
}

// The type code of each of T's leaves, in build<T>'s order: depth first, in declaration order. A call site lists them
// for a T. A List derived from CodeList adds codes of its own where a struct opens and closes.
template <std::size_t N>
struct CodeList {
    std::array<uint16_t, N> codes{};
    std::size_t count = 0;
    constexpr void put(uint16_t code) {
        if (count < N) {
            codes[count] = code;
        }
        ++count;
    }
    constexpr void open() {}
    constexpr void close() {}
    template <typename T>
    constexpr T leaf() {
        put(type_code<T>());
        return T{};
    }
};
template <typename T, template <std::size_t> class List = CodeList>
inline constexpr auto leaf_type_codes = [] {
    if constexpr (std::is_aggregate_v<T> || is_leaf_type<T> || is_specialization<T, std::tuple>) {
        constexpr std::size_t kCount = [] {
            List<0> counter;
            build<T>(counter);
            return counter.count;
        }();
        List<kCount> list;
        build<T>(list);
        return list.codes;
    } else {
        return std::array<uint16_t, 0>{};
    }
}();
template <typename T>
constexpr bool holds(uint16_t code) {
    for (const uint16_t leaf : leaf_type_codes<T>) {
        if (leaf == code) {
            return true;
        }
    }
    return false;
}

// The host matches call sites by type name, and the two compilers spell template arguments and local types
// differently, so payload and metadata types must be named, non-template types outside anonymous namespaces.
template <typename T>
constexpr bool has_plain_name() {
    constexpr bool plain = type_name<T>.find_first_of("<({") == std::string_view::npos;
    static_assert(plain, "templates, function-local types and types in an anonymous namespace are not supported");
    return plain;
}

// The rules a payload or metadata struct, and each struct passed as an argument, follows wherever it is compiled:
// kernels with the profiler on or off, and the host's Record types. Each broken rule is one compile error with its own
// message.
template <typename T>
constexpr bool check_struct() {
    static_assert(!std::is_union_v<T>, "unions are not supported; send the member it holds");
    static_assert(
        std::is_aggregate_v<T>,
        "only aggregate structs, fixed-width scalars, enums, String, Timestamp and arrays of them are supported; this "
        "type has a constructor, private field or virtual function");
    if constexpr (std::is_aggregate_v<T> && !std::is_union_v<T>) {
        static_assert(
            std::is_trivially_copyable_v<T>, "types with a custom copy, move or destructor are not supported");
        static_assert(std::is_copy_assignable_v<T>, "const and reference members are not supported");
        constexpr bool well_formed =
            has_plain_name<T>() && std::is_trivially_copyable_v<T> && std::is_copy_assignable_v<T>;
        if constexpr (well_formed) {
            return !holds<T>(0);  // type_code gives a field of a broken type code 0
        }
        return well_formed;
    } else {
        return false;
    }
}
// A metadata type is a struct, which follows check_struct's rules.
template <typename M>
constexpr bool check_metadata() {
    static_assert(std::is_class_v<M>, "metadata is a struct; put the value in one");
    if constexpr (std::is_class_v<M>) {
        return check_struct<M>();
    } else {
        return false;
    }
}
template <typename V>
constexpr bool check_argument() {
    if constexpr (std::is_enum_v<V>) {
        return has_plain_name<V>() && type_code<V>() != 0;
    } else if constexpr (is_std_array<V>) {
        using E = typename V::value_type;
        constexpr bool flat = !is_std_array<E> && !std::is_array_v<E>;
        static_assert(flat, "arrays of arrays are not supported; put the inner array in a struct");
        static_assert(std::tuple_size_v<V> != 0, "empty arrays are not supported");
        if constexpr (flat && std::tuple_size_v<V> != 0) {
            return check_argument<E>();
        } else {
            return false;
        }
    } else if constexpr (std::is_class_v<V> && !is_leaf_type<V>) {
        return check_struct<V>();
    } else {
        return type_code<V>() != 0;  // type_code rejects every other type with its own message
    }
}

}  // namespace detail

}  // namespace tt::tt_metal::experimental::streaming_profiler
