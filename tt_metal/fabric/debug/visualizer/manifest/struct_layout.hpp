// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string_view>
#include <type_traits>
#include <variant>
#include <vector>

#include <enchantum/type_name.hpp>

namespace tt::tt_fabric::layout {

// What one element is, with only the data decode needs to read it.
namespace element {

// Unsigned integer.
struct Uint {
    constexpr bool operator==(const Uint&) const = default;
};
// Signed integer.
struct Int {
    constexpr bool operator==(const Int&) const = default;
};
// Enum.
struct Enum {
    // The enum type name
    std::string_view name;
    constexpr bool operator==(const Enum&) const = default;
};
// Struct.
struct Struct {
    // The struct type name
    std::string_view name;
    constexpr bool operator==(const Struct&) const = default;
};
// Raw bytes.
struct Bytes {
    constexpr bool operator==(const Bytes&) const = default;
};
// Padding.
struct Pad {
    constexpr bool operator==(const Pad&) const = default;
};
// A table of entries packed below byte granularity, least significant bit first.
struct Packed {
    // The key decode looks up to find the binding function that reads the entries, such as "direction_table"
    std::string_view table;
    uint32_t bits_per_entry;
    constexpr bool operator==(const Packed&) const = default;
};

}  // namespace element

using Element = std::
    variant<element::Uint, element::Int, element::Enum, element::Struct, element::Bytes, element::Pad, element::Packed>;

// One element, or `count` of them. Examples:
// - uint16_t counts[2]:         {element: Uint, size: 4, count: 2}
// - RouterState state:          {element: Enum{"RouterState"}, size: 4, count: 0}
// - std::array<WorkerXY, 3> xy: {element: Struct{"WorkerXY"}, size: 12, count: 3}
struct Type {
    Element element;
    // Size of the whole thing in bytes. For an array, one element is size / count
    uint32_t size = 0;
    // 0 for a scalar, N for N elements or a packed table of N entries
    uint32_t count = 0;
    constexpr bool operator==(const Type&) const = default;
};

// A member of a struct. Example: WorkerXY worker_xy at offset 32 is
// {name: "worker_xy", offset: 32, type: {element: Struct{"WorkerXY"}, size: 4, count: 0}}
struct Member {
    std::string_view name;
    uint32_t offset;
    Type type;
};

// A struct's name and size, and its members, which cover every byte of it in order.
// This is the manifest's representation of a struct type, regardless of whether it
// could be described at compile time (ie. has a StructLayout specialization), or
// was defined during runtime (ie. arch-specific structs).
struct StructType {
    std::string_view name;
    uint32_t size = 0;
    std::vector<Member> members;
};

// Each described struct specializes StructLayout<T> with its member list. There is no primary definition, so if a
// member of struct A is itself a struct B with no StructLayout, A's member list fails to compile in type_of().
// B's StructLayout must come before A's.
// NOTE: Members should be listed in the order they are defined in the struct, using an array named members.
template <typename T>
struct StructLayout;

// A struct is described if it has a StructLayout specialization.
template <typename T>
concept Described = requires { StructLayout<T>::members; };

namespace detail {

template <typename T>
struct StdArray : std::false_type {};
template <typename T, std::size_t N>
struct StdArray<std::array<T, N>> : std::true_type {
    using element = T;
    static constexpr std::size_t count = N;
};

// A described struct's name
template <typename T>
constexpr std::string_view struct_name() {
    if constexpr (requires { StructLayout<T>::name; }) {
        return StructLayout<T>::name;
    } else {
        return enchantum::type_name<T>;
    }
}

// The Element for one element of type T: an integer, an enum or a described struct.
template <typename T>
constexpr Element element_of() {
    static_assert(
        !std::is_array_v<T> && !StdArray<std::remove_cv_t<T>>::value, "multidimensional arrays are not supported");
    if constexpr (std::is_enum_v<T>) { /* enum */
        return element::Enum{enchantum::type_name<T>};
    } else if constexpr (std::is_integral_v<T>) { /* integral (whole number) */
        if constexpr (std::is_signed_v<T>) {
            return element::Int{};
        } else {
            return element::Uint{};
        }
    } else if constexpr (Described<T>) { /* defined struct type, via StructLayout<T> */
        return element::Struct{struct_name<T>()};
    } else { /* always fails if we get here */
        static_assert(!sizeof(T*), "member type has no StructLayout specialization; describe it or use LAYOUT_BYTES");
    }
}

}  // namespace detail

// The Type of a T.
template <typename T>
constexpr Type type_of() {
    using U = std::remove_cv_t<T>;
    if constexpr (std::is_array_v<U>) {
        return Type{detail::element_of<std::remove_extent_t<U>>(), sizeof(U), static_cast<uint32_t>(std::extent_v<U>)};
    } else if constexpr (detail::StdArray<U>::value) {
        static_assert(detail::StdArray<U>::count > 0, "a zero-length array would read as a scalar");
        return Type{
            detail::element_of<typename detail::StdArray<U>::element>(),
            sizeof(U),
            static_cast<uint32_t>(detail::StdArray<U>::count)};
    } else {
        return Type{detail::element_of<U>(), sizeof(U), 0};
    }
}

// The Type of `count` elements of T, for an array whose length is known only at run time.
template <typename T>
constexpr Type array_of(uint32_t count) {
    static_assert(!std::is_array_v<T> && !detail::StdArray<std::remove_cv_t<T>>::value, "T must be one element");
    const Type one = type_of<T>();
    return Type{one.element, one.size * count, count};
}

// Ensures that every byte of a struct of `size` bytes belongs to exactly one member, listed in declaration order,
// where each member starts where the previous one ended, and the last one ends at `size`.
constexpr bool validate_members(std::span<const Member> members, std::size_t size) {
    uint32_t expected_offset = 0;
    for (const auto& member : members) {
        if (member.offset != expected_offset) {
            return false;
        }
        expected_offset += member.type.size;
    }
    return expected_offset == size;
}

// validate_members against sizeof(T), for a member list known at compile time.
template <typename T, std::size_t N>
constexpr bool validate_struct_members(const std::array<Member, N>& members) {
    return validate_members(members, sizeof(T));
}

}  // namespace tt::tt_fabric::layout

// Layout of a member whose type type_of() can describe: an integer, an enum, a described struct, or a
// one-dimensional C array or std::array of one of those.
#define LAYOUT_MEMBER(T, member)                                                              \
    ::tt::tt_fabric::layout::Member {                                                         \
        #member, offsetof(T, member), ::tt::tt_fabric::layout::type_of<decltype(T::member)>() \
    }

// Layout of a padding member, which decode skips.
#define LAYOUT_PAD(T, member)                                                                         \
    ::tt::tt_fabric::layout::Member {                                                                 \
        #member, offsetof(T, member), {::tt::tt_fabric::layout::element::Pad{}, sizeof(T::member), 0} \
    }

// Layout of a packed table member, whose type states its entry width as BITS_PER_COMPRESSED_ENTRY. `table` names the
// format, which is used as a key by decode to find the binding function that reads the entries.
#define LAYOUT_PACKED(T, member, table)                                                               \
    [] {                                                                                              \
        using M = decltype(T::member);                                                                \
        constexpr uint32_t bits = M::BITS_PER_COMPRESSED_ENTRY;                                       \
        static_assert(sizeof(M) * 8 % bits == 0, "packed table must hold a whole number of entries"); \
        return ::tt::tt_fabric::layout::Member{                                                       \
            #member,                                                                                  \
            offsetof(T, member),                                                                      \
            ::tt::tt_fabric::layout::Type{                                                            \
                ::tt::tt_fabric::layout::element::Packed{table, bits},                                \
                sizeof(M),                                                                            \
                static_cast<uint32_t>(sizeof(M) * 8 / bits)}};                                        \
    }()

// Layout of a member decode shows as raw bytes without interpreting them, such as a table not yet described.
#define LAYOUT_BYTES(T, member)                                                                         \
    ::tt::tt_fabric::layout::Member {                                                                   \
        #member, offsetof(T, member), {::tt::tt_fabric::layout::element::Bytes{}, sizeof(T::member), 0} \
    }
