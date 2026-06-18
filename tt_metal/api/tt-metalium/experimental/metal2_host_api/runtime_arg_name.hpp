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

// Heap-free, inline, fixed-capacity name key for per-node runtime-arg Tables.
//
// Replaces std::string as the key of RuntimeArgValues (Table<RtaName, uint32_t>). The
// per-node run-args loop builds ~(#kernels x #cores) of these every dispatch; with std::string
// keys, names longer than the 15-char SSO threshold (e.g. "output_tile_start_id"=20) heap-allocate
// per entry. RtaName stores the name inline (no heap, ever) and is a trivially-copyable POD so the
// backing small-vector relocates by memcpy and comparison is a length-gated memcmp.
//
// Implicitly constructible from string literals / std::string, so op factories keep writing
// {{"name", value}} unchanged -- the public API design is untouched.
struct RtaName {
    static constexpr std::size_t CAP = 32;  // longest runtime-arg name in use is 24 chars; 32 leaves margin
    char data_[CAP] = {};
    std::uint8_t len_ = 0;

    RtaName() = default;
    RtaName(std::string_view s) {  // NOLINT(google-explicit-constructor): implicit by design
        // Fail loud rather than silently truncate: two names that differ only past CAP would alias to
        // the same key and fetch each other's args. A name this long means CAP needs raising, not trimming.
        TT_FATAL(s.size() <= CAP, "RtaName '{}' exceeds inline capacity {} ({} chars)", s, CAP, s.size());
        len_ = static_cast<std::uint8_t>(s.size());
        std::memcpy(data_, s.data(), len_);
    }
    RtaName(const char* s) : RtaName(std::string_view(s)) {}         // NOLINT(google-explicit-constructor)
    RtaName(const std::string& s) : RtaName(std::string_view(s)) {}  // NOLINT(google-explicit-constructor)

    std::string_view view() const { return std::string_view(data_, len_); }

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
        // per-byte FNV loop — the apply path does ~hundreds of slot-map lookups per dispatch.
        return std::hash<std::string_view>{}(n.view());
    }
};
