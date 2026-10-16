// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string_view>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include <enchantum/enchantum.hpp>
#include <enchantum/type_name.hpp>

namespace tt::tt_fabric::layout {

// The examples in this file describe these types:
//
//   enum class Color : uint8_t { RED, GREEN, BLUE };
//   struct Point { uint16_t x; uint16_t y; };
//   struct Pixel {
//       Point at;            // offset 0
//       Color color;         // offset 4
//       int8_t brightness;   // offset 5
//       uint16_t counts[2];  // offset 6
//       uint8_t history;     // offset 10: four 2-bit Colors, entry 0 in the lowest bits
//       uint8_t reserved;    // offset 11, unused
//   };                       // 12 bytes
//
//   // A tagged union: kind says which of geometry's members are the proper
//   // interpretation of its bytes.
//   enum class Kind : uint32_t { DOT, LINE };
//   union Geometry { Point dot; uint32_t line[2]; };  // 8 bytes
//   struct Shape { Kind kind; Geometry geometry; };   // geometry at offset 4

// What one element is, with only the data decode needs to read it.
namespace element {

// Unsigned integer. Example: Pixel::counts.
struct Uint {
    constexpr bool operator==(const Uint&) const = default;
};
// Signed integer. Example: Pixel::brightness.
struct Int {
    constexpr bool operator==(const Int&) const = default;
};
// Enum. Example: Pixel::color is Enum{"Color"}.
struct Enum {
    // The enum type name
    std::string_view name;
    constexpr bool operator==(const Enum&) const = default;
};
// Struct. Example: Pixel::at is Struct{"Point"}.
struct Struct {
    // The struct type name
    std::string_view name;
    constexpr bool operator==(const Struct&) const = default;
};
// Tagged union. Example: Shape::geometry is Union{"Geometry", "kind"}.
struct Union {
    // The union type name
    std::string_view name;
    // The name of the enclosing struct's member that holds the tag. The tag
    // represents the union's member that holds the bytes.
    std::string_view tag;
    constexpr bool operator==(const Union&) const = default;
};
// Raw bytes. Example: LAYOUT_BYTES(Pixel, history) shows history without reading its Colors.
struct Bytes {
    constexpr bool operator==(const Bytes&) const = default;
};
// Padding. Example: Pixel::reserved.
struct Pad {
    constexpr bool operator==(const Pad&) const = default;
};
// A table of entries packed below byte granularity, least significant bit first. 
// Example: Pixel::history is Packed{"Color", 2}.
struct Packed {
    // The enum type name each entry holds
    std::string_view entry_enum;
    uint32_t bits_per_entry;
    constexpr bool operator==(const Packed&) const = default;
};

}  // namespace element

using Element = std::variant<
    element::Uint,
    element::Int,
    element::Enum,
    element::Struct,
    element::Union,
    element::Bytes,
    element::Pad,
    element::Packed>;

// One element, or `count` of them. Examples:
// - Pixel::color:   {element: Enum{"Color"}, size: 1, count: 0}
// - Pixel::counts:  {element: Uint, size: 4, count: 2}
// - Pixel::history: {element: Packed{"Color", 2}, size: 1, count: 4}
struct Type {
    Element element;
    // Size of the whole thing in bytes. For an array, one element is size / count. A packed table's entries are
    // bits_per_entry bits each instead
    uint32_t size = 0;
    // 0 for a scalar, N for N elements or a packed table of N entries
    uint32_t count = 0;
    constexpr bool operator==(const Type&) const = default;
};

// A member of a struct. Example: Pixel::counts is
// {name: "counts", offset: 6, type: {element: Uint, size: 4, count: 2}}
struct Member {
    std::string_view name;
    uint32_t offset;
    Type type;
};

// A struct's name and size, and its members, which cover every byte of it in order.
// This is the manifest's representation of a struct type, regardless of whether it
// could be described at compile time (ie. has a StructLayout specialization), or
// was defined during runtime (ie. arch-specific structs).
// Example: Point is {name: "Point", size: 4, members: [x at offset 0, y at offset 2]}, where each member's type is
// {element: Uint, size: 2, count: 0}.
struct StructType {
    std::string_view name;
    uint32_t size = 0;
    std::vector<Member> members;
};

// One member of a tagged union, and the name of the tag enumerator that picks it. Example: Geometry's dot is
// {member: {name: "dot", offset: 0, type: {element: Struct{"Point"}, size: 4, count: 0}}, tag_value: "DOT"}
struct UnionMember {
    Member member;
    std::optional<std::string_view> tag_value;
};

// A tagged union's name and size, the enum its tag holds, and its members, which all start at offset 0.
// Example: Geometry is {name: "Geometry", size: 8, tag_enum: "Kind", members: [dot, picked by "DOT"; line, picked
// by "LINE"]}.
struct UnionType {
    std::string_view name;
    uint32_t size = 0;
    std::string_view tag_enum;
    std::vector<UnionMember> members;
};

// Each described struct specializes StructLayout<T> with its member list. There is no primary definition, so if a
// member of struct A is itself a struct B with no StructLayout, A's member list fails to compile in type_of().
// B's StructLayout must come before A's.
// NOTE: Members should be listed in the order they are defined in the struct, using an array named members.
// Example, after a StructLayout<Point> that lists x and y:
//   template <>
//   struct StructLayout<Pixel> {
//       static constexpr std::array members = {
//           LAYOUT_MEMBER(Pixel, at),
//           LAYOUT_MEMBER(Pixel, color),
//           LAYOUT_MEMBER(Pixel, brightness),
//           LAYOUT_MEMBER(Pixel, counts),
//           LAYOUT_PACKED_BITS(Pixel, history, Color, 2),
//           LAYOUT_PAD(Pixel, reserved),
//       };
//   };
//   static_assert(validate_struct_members<Pixel>(StructLayout<Pixel>::members));
template <typename T>
struct StructLayout;

// A struct is described if it has a StructLayout specialization.
template <typename T>
concept Described = requires { StructLayout<T>::members; };

// Each described union specializes UnionLayout<U> with its tag enum, `using Tag = ...`, and an array named members of
// UnionMember, built with LAYOUT_UNION_MEMBER and LAYOUT_UNION_DEFAULT. Like StructLayout, it has no primary
// definition, and each member's type needs a StructLayout before it. Example:
//   template <>
//   struct UnionLayout<Geometry> {
//       using Tag = Kind;
//       static constexpr std::array members = {
//           LAYOUT_UNION_MEMBER(Geometry, dot, Kind::DOT),
//           LAYOUT_UNION_MEMBER(Geometry, line, Kind::LINE),
//       };
//   };
// Shape's StructLayout then lists LAYOUT_MEMBER(Shape, kind) and LAYOUT_TAGGED_UNION(Shape, geometry, kind).
template <typename U>
struct UnionLayout;

// A union is described if it has a UnionLayout specialization.
template <typename U>
concept DescribedUnion = requires {
    typename UnionLayout<U>::Tag;
    UnionLayout<U>::members;
};

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

// A described union's name
template <typename U>
constexpr std::string_view union_name() {
    if constexpr (requires { UnionLayout<U>::name; }) {
        return UnionLayout<U>::name;
    } else {
        return enchantum::type_name<U>;
    }
}

// The name of tag_value, an enumerator of union U's Tag, or empty when tag_value is no enumerator.
template <typename U>
constexpr std::string_view tag_value_name(typename UnionLayout<U>::Tag tag_value) {
    return enchantum::to_string(tag_value);
}

// Whether every enumerator of E fits in an unsigned field of `bits` bits.
template <typename E>
constexpr bool enumerators_fit(uint32_t bits) {
    return std::ranges::all_of(enchantum::values<E>, [bits](E value) {
        const auto raw = static_cast<std::underlying_type_t<E>>(value);
        return std::cmp_greater_equal(raw, 0) && std::cmp_less(raw, uint64_t{1} << bits);
    });
}

// The Element for one element of type T.
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
    } else if constexpr (std::is_union_v<T>) { /* a union needs its tag, which only the enclosing struct knows */
        static_assert(!sizeof(T*), "a union member needs LAYOUT_TAGGED_UNION, or LAYOUT_BYTES when nothing tags it");
    } else if constexpr (Described<T>) { /* defined struct type, via StructLayout<T> */
        return element::Struct{struct_name<T>()};
    } else { /* always fails if we get here */
        static_assert(!sizeof(T*), "member type has no StructLayout specialization; describe it or use LAYOUT_BYTES");
    }
}

}  // namespace detail

// The Type of a T. Example: type_of<Color>() is {element: Enum{"Color"}, size: 1, count: 0}.
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

// The Type of `count` elements of T, for an array whose length is known only at run time. Example: array_of<Point>(3)
// is {element: Struct{"Point"}, size: 12, count: 3}.
template <typename T>
constexpr Type array_of(uint32_t count) {
    static_assert(!std::is_array_v<T> && !detail::StdArray<std::remove_cv_t<T>>::value, "T must be one element");
    const Type one = type_of<T>();
    return Type{one.element, one.size * count, count};
}

// Ensures that every byte of a struct of `size` bytes belongs to exactly one member, listed in declaration order,
// where each member starts where the previous one ended, and the last one ends at `size`. Example: Pixel's members
// fail if color is listed before at, or if reserved is left out, since history then ends at 11 rather than 12.
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

// Ensures decode can read union U from its members: each starts at 0, the widest fills the union, at most one is the
// default, each tag value a member lists names an enumerator of UnionLayout<U>::Tag, and every enumerator picks
// exactly one member, the one that lists it or else the default. Example: Geometry's members fail if line is left
// out: nothing then picks LINE, and dot's 4 bytes do not fill the union's 8.
template <typename U, std::size_t N>
constexpr bool validate_union_members(const std::array<UnionMember, N>& members) {
    static_assert(std::is_union_v<U>, "validate_union_members checks a union");
    constexpr auto& tag_names = enchantum::names<typename UnionLayout<U>::Tag>;
    uint32_t widest = 0;
    std::size_t defaults = 0;
    for (const auto& [member, tag_value] : members) {
        if (member.offset != 0) {
            return false;
        }
        widest = std::max(widest, member.type.size);
        if (!tag_value.has_value()) {
            ++defaults;
        } else if (std::ranges::find(tag_names, *tag_value) == tag_names.end()) {
            return false;
        }
    }
    if (widest != sizeof(U) || defaults > 1) {
        return false;
    }
    for (const std::string_view tag : tag_names) {
        const auto listed = std::ranges::count(members, std::optional(tag), &UnionMember::tag_value);
        if (listed > 1 || (listed == 0 && defaults == 0)) {
            return false;
        }
    }
    return true;
}

// Member lists joined in order, for layouts that share a group of members. Example: if head holds Pixel's first five
// members, concat_members(head, std::array{LAYOUT_PAD(Pixel, reserved)}) is all six.
template <std::size_t... Ns>
constexpr std::array<Member, (Ns + ...)> concat_members(const std::array<Member, Ns>&... parts) {
    std::array<Member, (Ns + ...)> out{};
    std::size_t at = 0;
    ((std::ranges::copy(parts, out.begin() + at), at += Ns), ...);
    return out;
}

namespace detail {

// A string literal as a template argument.
template <std::size_t N>
struct Literal {
    char chars[N]{};
    constexpr Literal(const char (&text)[N]) { std::copy_n(text, N, chars); }
    constexpr std::string_view view() const { return {chars, N - 1}; }
};

// Writes text at out + length, unless out is null, and returns the length after it.
constexpr std::size_t append_text(char* out, std::size_t length, std::string_view text) {
    if (out != nullptr) {
        std::ranges::copy(text, out + length);
    }
    return length + text.size();
}

// Appends a template argument as C++ spells it: an integer's decimal digits, or true or false.
template <typename V>
constexpr std::size_t append_template_arg(char* out, std::size_t length, V value) {
    static_assert(std::is_integral_v<V>, "template_name spells only integer and bool arguments");
    if constexpr (std::is_same_v<V, bool>) {
        return append_text(out, length, value ? "true" : "false");
    } else {
        char digits[20]{};  // the most a 64-bit integer needs
        std::size_t count = 0;
        auto rest = static_cast<std::make_unsigned_t<V>>(value);
        do {
            digits[count++] = static_cast<char>('0' + (rest % 10));
            rest /= 10;
        } while (rest != 0);
        std::reverse(digits, digits + count);
        return append_text(out, length, {digits, count});
    }
}

template <typename V>
constexpr bool is_non_negative(V value) {
    if constexpr (std::is_signed_v<V>) {
        return value >= 0;
    } else {
        return true;
    }
}

// Writes Name<Args...> at out, unless out is null, and returns its length.
template <Literal Name, auto... Args>
constexpr std::size_t write_template_name(char* out) {
    std::size_t length = append_text(out, 0, Name.view());
    bool first = true;
    const auto append_arg = [&](auto arg) {
        length = append_template_arg(out, append_text(out, length, first ? "<" : ","), arg);
        first = false;
    };
    (append_arg(Args), ...);
    return append_text(out, length, ">");
}

template <Literal Name, auto... Args>
struct TemplateName {
    static_assert(sizeof...(Args) > 0, "a template name has at least one argument");
    static_assert((is_non_negative(Args) && ...), "template_name spells only non-negative arguments");
    static constexpr auto chars = [] {
        std::array<char, write_template_name<Name, Args...>(nullptr)> out{};
        write_template_name<Name, Args...>(out.data());
        return out;
    }();
};

}  // namespace detail

// A class template instance's name as C++ spells it, with no space after a comma, built at compile time. For
// example, template_name<"HybridMeshPacketHeaderT", 36> is "HybridMeshPacketHeaderT<36>", and
// template_name<"T", 1, false> is "T<1,false>". enchantum::type_name refuses templates, so a template's
// StructLayout sets its name with this.
template <detail::Literal Name, auto... Args>
inline constexpr std::string_view template_name{
    detail::TemplateName<Name, Args...>::chars.data(), detail::TemplateName<Name, Args...>::chars.size()};

}  // namespace tt::tt_fabric::layout

// Layout of a member whose type type_of() can describe: an integer, an enum, a described struct, or a
// one-dimensional C array or std::array of one of those. Example: LAYOUT_MEMBER(Pixel, at) is
// {name: "at", offset: 0, type: {element: Struct{"Point"}, size: 4, count: 0}}
#define LAYOUT_MEMBER(T, member)                                                              \
    ::tt::tt_fabric::layout::Member {                                                         \
        #member, offsetof(T, member), ::tt::tt_fabric::layout::type_of<decltype(T::member)>() \
    }

// Layout of a padding member, which decode skips. Example: LAYOUT_PAD(Pixel, reserved) is
// {name: "reserved", offset: 11, type: {element: Pad, size: 1, count: 0}}
#define LAYOUT_PAD(T, member)                                                                         \
    ::tt::tt_fabric::layout::Member {                                                                 \
        #member, offsetof(T, member), {::tt::tt_fabric::layout::element::Pad{}, sizeof(T::member), 0} \
    }

// Layout of the padding the compiler adds after T's last member, last_member, which has no name to list. A member
// added after last_member and not listed reads as this padding, so prefer a named pad member where adding one to T
// leaves the code that uses T unchanged. Example: struct T { uint32_t word; uint8_t byte; } is 8 bytes, so
// LAYOUT_TAIL_PAD(T, byte) is {name: "tail_padding", offset: 5, type: {element: Pad, size: 3, count: 0}}
#define LAYOUT_TAIL_PAD(T, last_member)                                                          \
    [] {                                                                                         \
        constexpr uint32_t end = offsetof(T, last_member) + sizeof(T::last_member);              \
        static_assert(sizeof(T) > end, "LAYOUT_TAIL_PAD needs padding after " #last_member);     \
        return ::tt::tt_fabric::layout::Member{                                                  \
            "tail_padding", end, {::tt::tt_fabric::layout::element::Pad{}, sizeof(T) - end, 0}}; \
    }()

// Layout of a member packed as a table of `bits`-bit entries, each holding a value of enum E. Example:
// LAYOUT_PACKED_BITS(Pixel, history, Color, 2) is
// {name: "history", offset: 10, type: {element: Packed{"Color", 2}, size: 1, count: 4}}
#define LAYOUT_PACKED_BITS(T, member, E, bits)                                                                      \
    [] {                                                                                                            \
        using M = decltype(T::member);                                                                              \
        constexpr uint32_t entry_bits = bits;                                                                       \
        static_assert(std::is_enum_v<E>, "a packed table's entries hold an enum");                                  \
        static_assert(sizeof(M) * 8 % entry_bits == 0, "packed table must hold a whole number of entries");         \
        static_assert(                                                                                              \
            ::tt::tt_fabric::layout::detail::enumerators_fit<E>(entry_bits), "every value of E must fit an entry"); \
        return ::tt::tt_fabric::layout::Member{                                                                     \
            #member,                                                                                                \
            offsetof(T, member),                                                                                    \
            ::tt::tt_fabric::layout::Type{                                                                          \
                ::tt::tt_fabric::layout::element::Packed{enchantum::type_name<E>, entry_bits},                      \
                sizeof(M),                                                                                          \
                static_cast<uint32_t>(sizeof(M) * 8 / entry_bits)}};                                                \
    }()

// Layout of a packed table member whose type states its entry width as BITS_PER_COMPRESSED_ENTRY.
#define LAYOUT_PACKED(T, member, E) LAYOUT_PACKED_BITS(T, member, E, decltype(T::member)::BITS_PER_COMPRESSED_ENTRY)

// Layout of a member decode shows as raw bytes without interpreting them, such as a table not yet described.
// Example: LAYOUT_BYTES(Pixel, history) is {name: "history", offset: 10, type: {element: Bytes, size: 1, count: 0}}
#define LAYOUT_BYTES(T, member)                                                                         \
    ::tt::tt_fabric::layout::Member {                                                                   \
        #member, offsetof(T, member), {::tt::tt_fabric::layout::element::Bytes{}, sizeof(T::member), 0} \
    }

// Layout of union U's member, picked when the tag holds tag_value, an enumerator of UnionLayout<U>::Tag. Example:
// LAYOUT_UNION_MEMBER(Geometry, dot, Kind::DOT) is {member: LAYOUT_MEMBER(Geometry, dot), tag_value: "DOT"}
#define LAYOUT_UNION_MEMBER(U, member, tag_value)                                               \
    ::tt::tt_fabric::layout::UnionMember {                                                      \
        LAYOUT_MEMBER(U, member), ::tt::tt_fabric::layout::detail::tag_value_name<U>(tag_value) \
    }

// Layout of union U's default member, picked by every tag value no other member lists. Example: with
// LAYOUT_UNION_DEFAULT(Geometry, line) in place of line's LAYOUT_UNION_MEMBER, LINE still picks line, and so would
// any Kind added later that no member lists.
#define LAYOUT_UNION_DEFAULT(U, member) \
    ::tt::tt_fabric::layout::UnionMember { LAYOUT_MEMBER(U, member), std::nullopt }

// Layout of T's member, a described union whose member is picked by T's tag_member. tag_member must hold the union's
// Tag enum, so decode can read it as one. Example: LAYOUT_TAGGED_UNION(Shape, geometry, kind) is
// {name: "geometry", offset: 4, type: {element: Union{"Geometry", "kind"}, size: 8, count: 0}}
#define LAYOUT_TAGGED_UNION(T, member, tag_member)                                                                   \
    [] {                                                                                                             \
        using U = decltype(T::member);                                                                               \
        static_assert(                                                                                               \
            std::is_same_v<                                                                                          \
                std::remove_cv_t<decltype(T::tag_member)>,                                                           \
                typename ::tt::tt_fabric::layout::UnionLayout<U>::Tag>,                                              \
            "a tagged union's tag member must hold the Tag enum of the union's UnionLayout");                        \
        return ::tt::tt_fabric::layout::Member{                                                                      \
            #member,                                                                                                 \
            offsetof(T, member),                                                                                     \
            {::tt::tt_fabric::layout::element::Union{::tt::tt_fabric::layout::detail::union_name<U>(), #tag_member}, \
             sizeof(U),                                                                                              \
             0}};                                                                                                    \
    }()
