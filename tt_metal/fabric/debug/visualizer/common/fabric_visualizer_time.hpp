// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// The times the fabric visualizer's files record, and name themselves by, all in UTC. Header-only, with no
// dependencies, so every stage can include it whatever it links.

#include <array>
#include <chrono>
#include <ctime>
#include <string>

namespace tt::tt_fabric::visualizer {

namespace detail {

// The time now in UTC, to the second, as strftime writes it with `format`.
inline std::string format_utc_now(const char* format) {
    const std::time_t now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    std::tm utc{};
    gmtime_r(&now, &utc);
    std::array<char, 32> text{};
    std::strftime(text.data(), text.size(), format, &utc);
    return text.data();
}

}  // namespace detail

// The time now as ISO 8601, such as "2026-11-05T22:15:00Z": when a file says it was written, or a capture started.
inline std::string get_utc_now_iso8601() { return detail::format_utc_now("%Y-%m-%dT%H:%M:%SZ"); }

// The time now as a compact stamp, such as "20261105T221500Z", for ids: it has no characters a file name can't, and
// sorts by time, so the run and capture folders it names sort by when they were made.
inline std::string get_utc_now_compact() { return detail::format_utc_now("%Y%m%dT%H%M%SZ"); }

}  // namespace tt::tt_fabric::visualizer
