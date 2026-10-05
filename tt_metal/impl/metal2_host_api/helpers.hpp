// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string_view>
#include <type_traits>
#include <unordered_set>
#include <variant>

#include <tt-metalium/experimental/metal2_host_api/node_coord.hpp>

namespace tt::tt_metal::experimental {

// TODO: This should be upstreamed.
inline NodeRangeSet to_node_range_set(const Nodes& nodes) {
    return std::visit(
        [](const auto& n) -> NodeRangeSet {
            using T = std::decay_t<decltype(n)>;
            if constexpr (std::is_same_v<T, NodeRangeSet>) {
                return n;
            } else if constexpr (std::is_same_v<T, NodeRange>) {
                return NodeRangeSet(n);
            } else {
                // NodeCoord case
                return NodeRangeSet(NodeRange(n, n));
            }
        },
        nodes);
}

// Local accessor names for kernel resource bindings must be valid C++ identifiers
// They are used verbatim in the generated kernel source code.
// TODO: Move this to ttsl in a follow up PR
inline bool IsValidCppIdentifier(std::string_view s) {
    if (s.empty()) {
        return false;
    }
    // Reject names with non-identifier characters or an empty/leading-digit form.
    const unsigned char c0 = static_cast<unsigned char>(s[0]);
    if (!((c0 >= 'a' && c0 <= 'z') || (c0 >= 'A' && c0 <= 'Z') || c0 == '_')) {
        return false;
    }
    for (size_t i = 1; i < s.size(); ++i) {
        const unsigned char c = static_cast<unsigned char>(s[i]);
        if (!((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '_')) {
            return false;
        }
    }

    // Reject reserved identifier patterns per [lex.name]/3.
    // Names containing "__", or starting with "_" followed by an uppercase letter.
    if (s.size() >= 2 && s[0] == '_' && s[1] >= 'A' && s[1] <= 'Z') {
        return false;
    }
    if (s.find("__") != std::string_view::npos) {
        return false;
    }

    // Reject C++ keywords. Anything in this set would produce uncompilable code
    // when emitted as a variable identifier in kernel_bindings_generated.h.
    static const std::unordered_set<std::string_view> kCppKeywords = {
        "alignas",     "alignof",   "and",        "and_eq",    "asm",      "auto",         "bitand",
        "bitor",       "bool",      "break",      "case",      "catch",    "char",         "char8_t",
        "char16_t",    "char32_t",  "class",      "compl",     "concept",  "const",        "consteval",
        "constexpr",   "constinit", "const_cast", "continue",  "co_await", "co_return",    "co_yield",
        "decltype",    "default",   "delete",     "do",        "double",   "dynamic_cast", "else",
        "enum",        "explicit",  "export",     "extern",    "false",    "float",        "for",
        "friend",      "goto",      "if",         "inline",    "int",      "long",         "mutable",
        "namespace",   "new",       "noexcept",   "not",       "not_eq",   "nullptr",      "operator",
        "or",          "or_eq",     "private",    "protected", "public",   "register",     "reinterpret_cast",
        "requires",    "return",    "short",      "signed",    "sizeof",   "static",       "static_assert",
        "static_cast", "struct",    "switch",     "template",  "this",     "thread_local", "throw",
        "true",        "try",       "typedef",    "typeid",    "typename", "union",        "unsigned",
        "using",       "virtual",   "void",       "volatile",  "wchar_t",  "while",        "xor",
        "xor_eq",
    };

    // If we got this far, and the name doesn't match any keywords, it's valid.
    return !kCppKeywords.contains(s);
}

}  // namespace tt::tt_metal::experimental
