// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Helpers for writing struct layouts (struct_layout.hpp): joining shared groups of members, and naming class template
// instances.

#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <string_view>
#include <type_traits>

#include "tt_metal/fabric/debug/visualizer/manifest/struct_layout.hpp"

namespace tt::tt_fabric::layout {

// Member lists joined in order, for layouts that share a group of members.
// Example:
// concat_members(std::array{LAYOUT_MEMBER(Shape, kind)}, std::array{LAYOUT_TAGGED_UNION(Shape, geometry, kind)})
// lists both of Shape's members.
template <std::size_t... Ns>
constexpr std::array<Member, (Ns + ...)> concat_members(const std::array<Member, Ns>&... parts) {
    std::array<Member, (Ns + ...)> out{};
    std::size_t at = 0;
    ((std::ranges::copy(parts, out.begin() + at), at += Ns), ...);
    return out;
}

// Members that one brace initializer fills in an aggregate's T{...}, so they count as one toward initializer_count<T>:
// the fields a struct gets from its base class (the base takes one initializer however many fields it has), or the
// members listed for an anonymous union (one initializer fills the union, even where the listed member is a C array).
// The first member takes the initializer and the rest take none. Example: for
//   struct Word { union { struct { uint16_t lo; uint16_t hi; }; uint32_t all; }; };
// filled_by_one_initializer(std::array{LAYOUT_MEMBER(Word, lo), LAYOUT_MEMBER(Word, hi)}) lists lo and hi as one
// initializer, like the struct's Word{x}.
template <std::size_t N>
constexpr std::array<Member, N> filled_by_one_initializer(std::array<Member, N> members) {
    static_assert(N > 0, "one initializer fills at least one member");
    for (auto& member : members) {
        member.initializers = 0;
    }
    members[0].initializers = 1;
    return members;
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
// template_name<"T", 1, false> is "T<1,false>". enchantum::type_name can't name a template instance, so a template's
// StructLayout sets its name with this.
template <detail::Literal Name, auto... Args>
inline constexpr std::string_view template_name{
    detail::TemplateName<Name, Args...>::chars.data(), detail::TemplateName<Name, Args...>::chars.size()};

}  // namespace tt::tt_fabric::layout
