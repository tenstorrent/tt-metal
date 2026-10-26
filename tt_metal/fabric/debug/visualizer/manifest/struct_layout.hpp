// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string_view>
#include <type_traits>
#include <utility>
#include <variant>

#include <enchantum/enchantum.hpp>
#include <enchantum/type_name.hpp>

namespace tt::tt_fabric::layout {

// The tagged union examples in this file describe these types.
//
//   enum class Kind : uint32_t { DOT, LINE };
//
//   struct Point {          // 4 bytes
//       uint16_t x;
//       uint16_t y;
//   };
//
//   union Geometry {        // 8 bytes
//       Point dot;          // holds the bytes when Shape::kind is DOT
//       uint32_t line[2];   // holds the bytes when Shape::kind is LINE
//   };
//
//   struct Shape {          // 12 bytes
//       Kind kind;          // offset 0: the tag, saying which of geometry's members holds its bytes
//       Geometry geometry;  // offset 4
//   };

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
    std::string_view type_name;
    constexpr bool operator==(const Enum&) const = default;
};
// Struct.
struct Struct {
    // The struct type name
    std::string_view type_name;
    constexpr bool operator==(const Struct&) const = default;
};
// Tagged union: a union whose member holding the bytes is picked by the value of a member of the enclosing struct, its
// tag. Example: geometry is the union, and kind is the tag that says which representation geometry's bytes hold. When
// kind is DOT they hold dot, and when it is LINE they hold line. Shape::geometry's element is
// TaggedUnion{"Geometry", "kind"}.
//   enum class Kind : uint32_t { DOT, LINE };
//   union Geometry {
//       Point dot;
//       uint32_t line[2];
//   };
//   struct Shape {
//       Kind kind;
//       Geometry geometry;
//   };
struct TaggedUnion {
    // The union type name (e.g. Geometry)
    std::string_view type_name;
    // The name of the enclosing struct's member that holds the tag (e.g. kind)
    std::string_view tag_member_name;
    constexpr bool operator==(const TaggedUnion&) const = default;
};
// Raw bytes, which we rely on decode to interpret.
struct Bytes {
    constexpr bool operator==(const Bytes&) const = default;
};
// Padding.
struct Pad {
    constexpr bool operator==(const Pad&) const = default;
};

}  // namespace element

using Element = std::variant<
    element::Uint,
    element::Int,
    element::Enum,
    element::Struct,
    element::TaggedUnion,
    element::Bytes,
    element::Pad>;

// One element, or `count` of them. Examples:
// - uint16_t counts[2]:         {element: Uint, size: 4, count: 2}
// - RouterState state:          {element: Enum{"RouterState"}, size: 4, count: 0}
// - std::array<WorkerXY, 3> xy: {element: Struct{"WorkerXY"}, size: 12, count: 3}
struct Type {
    Element element;
    // Size of the whole type in bytes. For an array, one element is size / count
    uint32_t size = 0;
    // 0 for a scalar, N for N elements
    uint32_t count = 0;
    constexpr bool operator==(const Type&) const = default;
};

// A member of a struct. Example: WorkerXY worker_xy at offset 32 is
// {name: "worker_xy", offset: 32, type: {element: Struct{"WorkerXY"}, size: 4, count: 0}, initializers: 1}
struct Member {
    std::string_view name;
    uint32_t offset;
    Type type;
    // Brace initializers the field takes in T{...}: 12 for uint8_t[12], 1 for a scalar, struct, union or std::array,
    // 0 for compiler padding, which no field backs. find_layout_error compares their sum with
    // initializer_count<T> to find fields the layout doesn't list.
    uint32_t initializers;
};

// One case of a tagged union: one of its members, and the name of the tag enumerator that picks it. A member picked by
// several tag values has a case for each. Example: Geometry has two cases.
//   dot, picked by DOT:
//     {member: {name: "dot", offset: 0, type: {element: Struct{"Point"}, size: 4, count: 0}}, tag_value: "DOT"}
//   line, picked by LINE:
//     {member: {name: "line", offset: 0, type: {element: Uint, size: 8, count: 2}}, tag_value: "LINE"}
//
//   enum class Kind : uint32_t { DOT, LINE };
//   union Geometry {
//       Point dot;
//       uint32_t line[2];
//   };
struct TaggedUnionCase {
    Member member;
    std::string_view tag_value;
};

// Each described struct specializes StructLayout<T> with its member list. There is no primary definition, so if a
// member of struct A is itself a struct B with no StructLayout, A's member list fails to compile in type_of().
// B's StructLayout must come before A's.
// NOTE: Members should be listed in the order they are defined in the struct, using an array named members.
template <typename T>
struct StructLayout;

// A struct is described if it has a StructLayout specialization, with a members list.
template <typename T>
concept Described = requires { StructLayout<T>::members; };

// Each described tagged union specializes TaggedUnionLayout<U> with its tag enum, `using Tag = ...`, and an array named
// cases of TaggedUnionCase entries, built with LAYOUT_TAGGED_UNION_CASE, with a case for every enumerator of Tag. Like
// StructLayout, it has no primary definition, and each member's type needs a StructLayout before it. Example, for
//   enum class Kind : uint32_t { DOT, LINE };
//   struct Point {
//       uint16_t x;
//       uint16_t y;
//   };
//   union Geometry {
//       Point dot;
//       uint32_t line[2];
//   };
//   struct Shape {
//       Kind kind;
//       Geometry geometry;
//   };
//
// Point needs a StructLayout first, since dot is a Point:
//   template <>
//   struct StructLayout<Point> {
//       static constexpr std::array members = {
//           LAYOUT_MEMBER(Point, x),
//           LAYOUT_MEMBER(Point, y),
//       };
//   };
//   static_assert(check_struct_layout<Point>());
//
// Geometry's TaggedUnionLayout names its tag enum and has a case for each of Kind's enumerators:
//   template <>
//   struct TaggedUnionLayout<Geometry> {
//       using Tag = Kind;
//       static constexpr std::array cases = {
//           LAYOUT_TAGGED_UNION_CASE(Geometry, dot, Kind::DOT),
//           LAYOUT_TAGGED_UNION_CASE(Geometry, line, Kind::LINE),
//       };
//   };
//   static_assert(validate_tagged_union_cases<Geometry>(TaggedUnionLayout<Geometry>::cases));
//
// Shape's StructLayout then lists the tag and the union:
//   template <>
//   struct StructLayout<Shape> {
//       static constexpr std::array members = {
//           LAYOUT_MEMBER(Shape, kind),
//           LAYOUT_TAGGED_UNION(Shape, geometry, kind),
//       };
//   };
//   static_assert(check_struct_layout<Shape>());
template <typename U>
struct TaggedUnionLayout;

namespace detail {

template <typename T>
struct StdArray : std::false_type {};
template <typename T, std::size_t N>
struct StdArray<std::array<T, N>> : std::true_type {
    using element = T;
    static constexpr std::size_t count = N;
};

// Dummy type to allow us to use any type in a template parameter pack.
template <std::size_t>
struct any {
    template <typename U>
    operator U() const;
};

// True if T{any, any, ...} with N values compiles.
template <typename T, std::size_t... I>
consteval bool accepts_initializers(std::index_sequence<I...>) {
    return requires { T{any<I>{}...}; };
}

// Binary searches for the largest number of variables T can be initialized
// with (the max N in [lo, hi] that accepts_initializers() will return True for).
template <typename T, std::size_t lo, std::size_t hi>
consteval std::size_t max_initializers() {
    if constexpr (lo == hi) {
        return lo;
    } else {
        constexpr std::size_t mid = (lo + hi + 1) / 2;
        if constexpr (accepts_initializers<T>(std::make_index_sequence<mid>{})) {
            return max_initializers<T, mid, hi>();
        } else {
            return max_initializers<T, lo, mid - 1>();
        }
    }
}

// The name of tag_value, an enumerator of union U's Tag, or empty when tag_value is no enumerator.
template <typename U>
constexpr std::string_view get_tag_value_name(typename TaggedUnionLayout<U>::Tag tag_value) {
    return enchantum::to_string(tag_value);
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
    } else if constexpr (std::is_union_v<T>) { /* a union needs its tag, which only the enclosing struct knows */
        static_assert(!sizeof(T*), "a union member needs LAYOUT_TAGGED_UNION, or LAYOUT_BYTES when nothing tags it");
    } else if constexpr (Described<T>) { /* defined struct type, via StructLayout<T> */
        return element::Struct{enchantum::type_name<T>};
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

// The max number of values aggregate T can be initialized with, T{v0, v1, ...}. Brace elision fills a C array field
// one element per value, so a field takes initializers_of<its type>() values. Each value sets at least one byte, so
// the count is at most sizeof(T).
template <typename T>
inline constexpr std::size_t initializer_count = detail::max_initializers<T, 0, sizeof(T)>();

// Number of initializers a field of type M takes in its struct's T{...}.
template <typename M>
consteval uint32_t initializers_of() {
    // C arrays are counted as their length times their element's count, since brace elision fills them one element at a
    // time. Anything else (a scalar, an enum, a struct, a union or a std::array) takes one value.
    if constexpr (std::is_array_v<M>) {
        return static_cast<uint32_t>(std::extent_v<M>) * initializers_of<std::remove_extent_t<M>>();
    } else {
        return 1;
    }
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

// The initializers the layout accounts for: the sum of its members' initializers.
constexpr std::size_t listed_initializers(std::span<const Member> members) {
    std::size_t total = 0;
    for (const auto& member : members) {
        total += member.initializers;
    }
    return total;
}

// Whether the layout lists compiler padding (bytes no field backs), from LAYOUT_PAD or LAYOUT_TAIL_PAD.
constexpr bool lists_compiler_padding(std::span<const Member> members) {
    return std::ranges::any_of(
        members, [](const Member& member) { return std::holds_alternative<element::Pad>(member.type.element); });
}

// The fields the layout lists: every member but compiler padding, one each whatever its type (a C array included).
constexpr std::size_t listed_fields(std::span<const Member> members) {
    return static_cast<std::size_t>(std::ranges::count_if(
        members, [](const Member& member) { return !std::holds_alternative<element::Pad>(member.type.element); }));
}

namespace detail {

// The most # of fields binds_exactly can bind: one line below per count.
inline constexpr std::size_t max_bound_fields = 16;

// Compiles a structured binding of N names to a T, which compiles only if T has exactly N fields. Unlike T{...}, a
// binding doesn't depend on T's constructors, so it counts a non-aggregate's fields. It needs every field to be public
// and declared directly in T (none inherited), and no anonymous union members. The binding is in a lambda that's
// never called, so T needs no constexpr constructor.
template <typename T, std::size_t N>
consteval bool binds_exactly() {
    static_assert(N >= 1 && N <= max_bound_fields, "binds_exactly has no line for this many fields; add one");
    [[maybe_unused]] auto bind = [](const T& t) {
        if constexpr (N == 1) {
            [[maybe_unused]] auto& [f0] = t;
        } else if constexpr (N == 2) {
            [[maybe_unused]] auto& [f0, f1] = t;
        } else if constexpr (N == 3) {
            [[maybe_unused]] auto& [f0, f1, f2] = t;
        } else if constexpr (N == 4) {
            [[maybe_unused]] auto& [f0, f1, f2, f3] = t;
        } else if constexpr (N == 5) {
            [[maybe_unused]] auto& [f0, f1, f2, f3, f4] = t;
        } else if constexpr (N == 6) {
            [[maybe_unused]] auto& [f0, f1, f2, f3, f4, f5] = t;
        } else if constexpr (N == 7) {
            [[maybe_unused]] auto& [f0, f1, f2, f3, f4, f5, f6] = t;
        } else if constexpr (N == 8) {
            [[maybe_unused]] auto& [f0, f1, f2, f3, f4, f5, f6, f7] = t;
        } else if constexpr (N == 9) {
            [[maybe_unused]] auto& [f0, f1, f2, f3, f4, f5, f6, f7, f8] = t;
        } else if constexpr (N == 10) {
            [[maybe_unused]] auto& [f0, f1, f2, f3, f4, f5, f6, f7, f8, f9] = t;
        } else if constexpr (N == 11) {
            [[maybe_unused]] auto& [f0, f1, f2, f3, f4, f5, f6, f7, f8, f9, f10] = t;
        } else if constexpr (N == 12) {
            [[maybe_unused]] auto& [f0, f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11] = t;
        } else if constexpr (N == 13) {
            [[maybe_unused]] auto& [f0, f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12] = t;
        } else if constexpr (N == 14) {
            [[maybe_unused]] auto& [f0, f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13] = t;
        } else if constexpr (N == 15) {
            [[maybe_unused]] auto& [f0, f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14] = t;
        } else if constexpr (N == 16) {
            [[maybe_unused]] auto& [f0, f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14, f15] = t;
        }
    };
    return true;
}

}  // namespace detail

// What's wrong with a compile-time member list for T, if anything. find_layout_error says which.
enum class LayoutError {
    NONE,
    // The list leaves a gap, overlaps, or doesn't end at sizeof(T)
    COVERAGE,
    // T is an aggregate whose initializers don't add up to the list's: T has a field the list leaves out
    UNLISTED_FIELD,
};

// The first check a compile-time member list for T fails, or NONE if it describes T. The checks, in order:
template <typename T>
constexpr LayoutError find_layout_error(std::span<const Member> members) {
    // Ensure every byte of the struct (including padding) is accounted for
    if (!validate_members(members, sizeof(T))) {
        return LayoutError::COVERAGE;
    }

    // Ensure the layout lists the same number of real (non-padding) fields as T. A non-aggregate's initializers can't
    // be counted (its constructors obscure its fields). Where its layout lists compiler padding, check_struct_layout
    // counts its fields with a structured binding instead, which can only fail the build, so it isn't one of these
    // errors. Without compiler padding, every byte is backed by a listed field, so coverage alone catches a new field.
    if constexpr (std::is_aggregate_v<T>) {
        if (listed_initializers(members) != initializer_count<T>) {
            return LayoutError::UNLISTED_FIELD;
        }
    }
    return LayoutError::NONE;
}

// find_layout_error as a static_assert per error, so a failing build says which check failed. For a non-aggregate whose
// layout lists compiler padding, it also binds T with as many names as the layout lists fields, which fails to compile
// with "decomposes into N elements, but only M names were provided" if T has a field the layout doesn't list. Each
// StructLayout is followed by static_assert(check_struct_layout<T>()). A member list that isn't T's StructLayout, such
// as one of several lists for one struct, is passed as `members`.
template <typename T, const auto& members = StructLayout<T>::members>
consteval bool check_struct_layout() {
    constexpr LayoutError error = find_layout_error<T>(members);
    static_assert(error != LayoutError::COVERAGE, "the layout leaves a gap, overlaps, or doesn't end at sizeof(T)");
    static_assert(
        error != LayoutError::UNLISTED_FIELD,
        "T has a field its layout doesn't list (the layout's initializers don't add up to initializer_count<T>)");
    if constexpr (!std::is_aggregate_v<T> && lists_compiler_padding(members)) {
        static_assert(detail::binds_exactly<T, listed_fields(members)>());
    }
    return true;
}

// Ensures decode can read tagged union U from its cases: each case's member starts at 0, the widest member fills the
// union, each case's tag value names an enumerator of TaggedUnionLayout<U>::Tag, and every enumerator has exactly one
// case. A member may have cases for several enumerators, but an enumerator can't have two cases, since decode then
// couldn't tell which member holds the bytes.
template <typename U, std::size_t N>
constexpr bool validate_tagged_union_cases(const std::array<TaggedUnionCase, N>& cases) {
    static_assert(std::is_union_v<U>, "validate_tagged_union_cases checks a union");
    constexpr auto& tag_names = enchantum::names<typename TaggedUnionLayout<U>::Tag>;
    uint32_t widest = 0;
    for (const auto& [member, tag_value] : cases) {
        if (member.offset != 0 || std::ranges::find(tag_names, tag_value) == tag_names.end()) {
            return false;
        }
        widest = std::max(widest, member.type.size);
    }
    if (widest != sizeof(U)) {
        return false;
    }
    return std::ranges::all_of(tag_names, [&cases](std::string_view tag) {
        return std::ranges::count(cases, tag, &TaggedUnionCase::tag_value) == 1;
    });
}

}  // namespace tt::tt_fabric::layout

// Layout of a member whose type type_of() can describe: an integer, an enum, a described struct, or a
// one-dimensional C array or std::array of one of those.
#define LAYOUT_MEMBER(T, member)                                                               \
    ::tt::tt_fabric::layout::Member {                                                          \
        #member, offsetof(T, member), ::tt::tt_fabric::layout::type_of<decltype(T::member)>(), \
            ::tt::tt_fabric::layout::initializers_of<decltype(T::member)>()                    \
    }

// Layout of the padding the compiler adds between T's adjacent members before and after, which has no field to list.
// A padding field the struct declares is a regular field: list it with LAYOUT_MEMBER. Decode skips padding. Example:
// struct T { uint8_t a; uint32_t b; } has 3 bytes of padding after a, so LAYOUT_PAD(T, a, b) is
// {name: "padding_after_a", offset: 1, type: {element: Pad, size: 3, count: 0}, initializers: 0}
#define LAYOUT_PAD(T, before, after)                                                                        \
    [] {                                                                                                    \
        constexpr uint32_t start = offsetof(T, before) + sizeof(T::before);                                 \
        constexpr uint32_t end = offsetof(T, after);                                                        \
        static_assert(end > start, "LAYOUT_PAD needs padding between " #before " and " #after);             \
        return ::tt::tt_fabric::layout::Member{                                                             \
            "padding_after_" #before, start, {::tt::tt_fabric::layout::element::Pad{}, end - start, 0}, 0}; \
    }()

// Layout of the padding the compiler adds after T's last member, last_member, which has no field to list. Example:
// struct T { uint32_t word; uint8_t byte; } is 8 bytes, so LAYOUT_TAIL_PAD(T, byte) is
// {name: "tail_padding", offset: 5, type: {element: Pad, size: 3, count: 0}, initializers: 0}
#define LAYOUT_TAIL_PAD(T, last_member)                                                             \
    [] {                                                                                            \
        constexpr uint32_t end = offsetof(T, last_member) + sizeof(T::last_member);                 \
        static_assert(sizeof(T) > end, "LAYOUT_TAIL_PAD needs padding after " #last_member);        \
        return ::tt::tt_fabric::layout::Member{                                                     \
            "tail_padding", end, {::tt::tt_fabric::layout::element::Pad{}, sizeof(T) - end, 0}, 0}; \
    }()

// Layout of a member decode keeps as raw bytes without interpreting them, for code that reads it by its own rules,
// such as a table of entries packed below byte granularity, or one whose shape depends on values known only at run
// time.
#define LAYOUT_BYTES(T, member)                                                                          \
    ::tt::tt_fabric::layout::Member {                                                                    \
        #member, offsetof(T, member), {::tt::tt_fabric::layout::element::Bytes{}, sizeof(T::member), 0}, \
            ::tt::tt_fabric::layout::initializers_of<decltype(T::member)>()                              \
    }

// Layout of a case of tagged union U: its member `member`, picked when the tag holds tag_value, an enumerator of
// TaggedUnionLayout<U>::Tag. A member picked by several tag values has a case for each. Example:
// LAYOUT_TAGGED_UNION_CASE(Geometry, dot, Kind::DOT) is {member: LAYOUT_MEMBER(Geometry, dot), tag_value: "DOT"}, for
//   enum class Kind : uint32_t { DOT, LINE };
//   union Geometry {
//       Point dot;
//       uint32_t line[2];
//   };
#define LAYOUT_TAGGED_UNION_CASE(U, member, tag_value)                                              \
    ::tt::tt_fabric::layout::TaggedUnionCase {                                                      \
        LAYOUT_MEMBER(U, member), ::tt::tt_fabric::layout::detail::get_tag_value_name<U>(tag_value) \
    }

// Layout of T's member, a described union whose member is picked by T's tag_member. tag_member must hold the union's
// Tag enum, so decode can read it as one. Example: LAYOUT_TAGGED_UNION(Shape, geometry, kind) is
// {name: "geometry", offset: 4, type: {element: TaggedUnion{"Geometry", "kind"}, size: 8, count: 0}}, for
//   enum class Kind : uint32_t { DOT, LINE };
//   union Geometry {        // 8 bytes, with a TaggedUnionLayout whose Tag is Kind
//       Point dot;
//       uint32_t line[2];
//   };
//   struct Shape {
//       Kind kind;          // offset 0
//       Geometry geometry;  // offset 4
//   };
#define LAYOUT_TAGGED_UNION(T, member, tag_member)                                                               \
    [] {                                                                                                         \
        using U = decltype(T::member);                                                                           \
        static_assert(                                                                                           \
            std::is_same_v<                                                                                      \
                std::remove_cv_t<decltype(T::tag_member)>,                                                       \
                typename ::tt::tt_fabric::layout::TaggedUnionLayout<U>::Tag>,                                    \
            "a tagged union's tag member must hold the Tag enum of the union's TaggedUnionLayout");              \
        return ::tt::tt_fabric::layout::Member{                                                                  \
            #member,                                                                                             \
            offsetof(T, member),                                                                                 \
            {::tt::tt_fabric::layout::element::TaggedUnion{enchantum::type_name<U>, #tag_member}, sizeof(U), 0}, \
            ::tt::tt_fabric::layout::initializers_of<U>()};                                                      \
    }()
