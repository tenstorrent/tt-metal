// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>
#include <cstring>
#include <functional>
#include <string>
#include <string_view>

#include <fmt/format.h>
#include <tt_stl/assert.hpp>

namespace tt::tt_metal::experimental {

// Inline, fixed-capacity key for runtime-argument names.
//
// Used as the key type wherever a runtime-argument name is stored or looked up: the per-node and
// common run-arg value Tables, the schema name lists and name->slot maps, and the enqueue-invariant
// name sets. Those structures are rebuilt on every dispatch, so a std::string key allocates on the
// heap for any name past the short-string-optimization threshold. RtaName stores the characters
// inline -- it is a trivially-copyable POD, so it relocates by memcpy, compares with a length-gated
// memcmp, and never touches the heap.
//
// Implicitly constructible from a string, so call sites keep declaring and supplying names as plain
// string literals. The capacity is validated when a ProgramSpec is built; the constructor enforces
// it as a backstop, failing loudly rather than silently truncating (truncation would alias two
// distinct names to the same key).
struct RtaName {
    static constexpr std::size_t CAP = 47;  // 48-byte key including the length; longest name in use is 28 chars
    char data_[CAP] = {};
    std::uint8_t len_ = 0;

    RtaName() = default;
    RtaName(std::string_view s) {  // NOLINT(google-explicit-constructor): implicit by design
        TT_FATAL(s.size() <= CAP, "Runtime-arg name '{}' exceeds the {}-character limit ({} chars)", s, CAP, s.size());
        len_ = static_cast<std::uint8_t>(s.size());
        std::memcpy(data_, s.data(), len_);
    }
    RtaName(const char* s) : RtaName(std::string_view(s)) {}         // NOLINT(google-explicit-constructor)
    RtaName(const std::string& s) : RtaName(std::string_view(s)) {}  // NOLINT(google-explicit-constructor)

    std::string_view view() const { return std::string_view(data_, len_); }

    // Reader-side conversion for the (minority) call sites that consume a name as a std::string.
    operator std::string() const { return std::string(view()); }  // NOLINT(google-explicit-constructor)

    bool operator==(const RtaName& o) const { return len_ == o.len_ && std::memcmp(data_, o.data_, len_) == 0; }
};

}  // namespace tt::tt_metal::experimental

template <>
struct fmt::formatter<tt::tt_metal::experimental::RtaName> : fmt::formatter<std::string_view> {
    auto format(const tt::tt_metal::experimental::RtaName& n, fmt::format_context& ctx) const {
        return fmt::formatter<std::string_view>::format(n.view(), ctx);
    }
};

template <>
struct std::hash<tt::tt_metal::experimental::RtaName> {
    std::size_t operator()(const tt::tt_metal::experimental::RtaName& n) const noexcept {
        // Delegate to the platform's optimized byte hash (chunked) rather than a naive
        // per-byte FNV loop -- the apply path does ~hundreds of slot-map lookups per dispatch.
        return std::hash<std::string_view>{}(n.view());
    }
};
