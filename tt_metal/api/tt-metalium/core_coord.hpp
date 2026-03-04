// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <fmt/base.h>
#include <nlohmann/json_fwd.hpp>
#include <stdint.h>
#include <tt_stl/span.hpp>
#include <algorithm>
#include <cstddef>
#include <functional>
#include <optional>
#include <set>
#include <string>
#include <vector>

#include <umd/device/types/xy_pair.hpp>

namespace ttsl::json {
template <typename T>
struct from_json_t;
template <typename T>
struct to_json_t;
}  // namespace ttsl::json

namespace tt::tt_metal {

using CoreCoord = tt_xy_pair;

class CoreRangeSet;

constexpr bool operator<=(const CoreCoord& a, const CoreCoord& b) { return (a < b) or (a == b); }

class CoreRange {
public:
    CoreCoord start_coord;
    CoreCoord end_coord;
    CoreRange(const CoreCoord& point);

    CoreRange(const CoreCoord& start_coord, const CoreCoord& end_coord);

    CoreRange(const CoreRange& other) = default;
    CoreRange& operator=(const CoreRange& other) = default;
    CoreRange(CoreRange&& other) noexcept = default;
    CoreRange& operator=(CoreRange&& other) noexcept = default;

    bool intersects(const CoreRange& other) const;

    std::optional<CoreRange> intersection(const CoreRange& other) const;

    bool adjacent(const CoreRange& other) const;

    bool contains(const CoreCoord& other) const;

    bool contains(const CoreRange& other) const;

    bool contains(const CoreRangeSet& other) const;

    // Merge lined-up (in x or y dimension) intersecting/adjacent rectangles
    std::optional<CoreRange> merge(const CoreRange& cr) const;

    std::string str() const;

    size_t size() const;

    CoreCoord grid_size() const;

    class CoreIterator {
    public:
        CoreIterator(const CoreCoord& current, const CoreRange& core_range);

        CoreCoord& operator*();

        CoreIterator& operator++();

        bool operator==(const CoreIterator& other) const;

        bool operator!=(const CoreIterator& other) const;

    private:
        CoreCoord current_;
        const CoreRange& range_;
    };

    CoreIterator begin() const;

    CoreIterator end() const;
};

constexpr bool operator==(const CoreRange& a, const CoreRange& b) {
    return a.start_coord == b.start_coord && a.end_coord == b.end_coord;
}

constexpr bool operator!=(const CoreRange& a, const CoreRange& b) { return !(a == b); }

constexpr bool operator<(const CoreRange& left, const CoreRange& right) {
    return (
        left.start_coord < right.start_coord ||
        (left.start_coord == right.start_coord && left.end_coord < right.end_coord));
}

class CoreRangeSet {
public:
    CoreRangeSet(tt::stl::Span<const CoreRange> core_ranges);

    CoreRangeSet(const std::set<CoreRange>& core_ranges);

    CoreRangeSet(const CoreRange& core_range);

    CoreRangeSet(tt::stl::Span<const CoreCoord> core_coords);

    CoreRangeSet() = default;

    friend void swap(CoreRangeSet& first, CoreRangeSet& second) noexcept;

    CoreRangeSet(const CoreRangeSet& other);

    CoreRangeSet& operator=(const CoreRangeSet& other) noexcept = default;

    CoreRangeSet(CoreRangeSet&& other) noexcept;

    CoreRangeSet& operator=(CoreRangeSet&& other) noexcept = default;

    CoreRangeSet(std::vector<CoreRange>&& core_ranges);

    bool empty() const;

    size_t size() const;

    template <typename T>
    CoreRangeSet merge(const T& other) const;

    bool intersects(const CoreCoord& other) const;

    bool intersects(const CoreRange& other) const;

    bool intersects(const CoreRangeSet& other) const;

    CoreRangeSet intersection(const CoreRangeSet& other) const;

    bool contains(const CoreCoord& other) const;

    bool contains(const CoreRange& other) const;

    bool contains(const CoreRangeSet& other) const;

    const std::vector<CoreRange>& ranges() const;

    std::string str() const;

    uint32_t num_cores() const;

    CoreRange bounding_box() const;

    // Return a CoreRangeSet with the same set of cores covered, but with as
    // small a number of CoreRanges as possible. This is useful to reduce the
    // amount of redundant per-core-range processing and NOC transactions for
    // code that uses this CoreRangeSet.
    CoreRangeSet merge_ranges() const;

    // Subtract the common CoreRanges between this CoreRangeSet.
    // A - (A n B)
    CoreRangeSet subtract(const CoreRangeSet& other) const;

private:
    void validate_no_overlap();
    std::vector<CoreRange> ranges_;
};

bool operator==(const CoreRangeSet& a, const CoreRangeSet& b);

std::vector<CoreCoord> grid_to_cores(
    uint32_t num_cores, uint32_t grid_size_x, uint32_t grid_size_y, bool row_wise = false);

std::vector<CoreCoord> grid_to_cores(CoreCoord start, CoreCoord end, bool row_wise = false);

// Noop cores are appended at the end with no guarantees on ordering
std::vector<CoreCoord> grid_to_cores_with_noop(
    uint32_t bbox_x, uint32_t bbox_y, uint32_t grid_size_x, uint32_t grid_size_y, bool row_wise = false);

// Noop cores are appended at the end with no guarantees on ordering
std::vector<CoreCoord> grid_to_cores_with_noop(
    const CoreRangeSet& used_cores, const CoreRangeSet& all_cores, bool row_wise = false);

std::vector<CoreCoord> corerange_to_cores(
    const CoreRangeSet& crs, std::optional<uint32_t> max_cores = std::nullopt, bool row_wise = false);

// Select a CoreRangeSet of cores from a CoreRangeSet.
// The method will traverse the given CoreRangeSet in row-wise order and return a subset of cores based on start_index
// and end_index (inclusive), where each core is represented by it's own CoreRange in the returned CoreRangeSet. Example
// usage: CoreRangeSet crs = {{0, 0, 2, 2}, {4, 0, 5, 2}}; CoreRangeSet selected_cores = select_from_corerangeset(crs,
// 0, 3); selected_cores = {{0,0}, {1,0}, {2,0}, {4,0}}
CoreRangeSet select_from_corerangeset(
    const CoreRangeSet& crs, uint32_t start_index, uint32_t end_index, bool row_wise = false);

// Select a contiguous CoreRange of cores from a CoreRangeSet.
// The method will select an x by y contiguous CoreRange of cores from the given CoreRangeSet. If multiple CoreRanges of
// size x by y are found, the method will return the lower leftmost subset of cores in the first available CoreRange.
// Example usage:
// CoreRangeSet crs = {{0, 0, 2, 2}, {4, 0, 5, 2}};
// CoreRange selected_core_range = select_contiguous_range_from_corerangeset(crs, 3, 1);
// selected_core_range = {{0,0}, {2,1}}
std::optional<CoreRange> select_contiguous_range_from_corerangeset(const CoreRangeSet& crs, uint32_t x, uint32_t y);

bool operator!=(const CoreRangeSet& a, const CoreRangeSet& b);

}  // namespace tt::tt_metal

// Adding to tt::tt_metal namespace as we transition to moving this out of global namespace eventually.
using CoreCoord [[deprecated("Use tt::tt_metal::CoreCoord")]] = tt::tt_metal::CoreCoord;
using CoreRange [[deprecated("Use tt::tt_metal::CoreRange")]] = tt::tt_metal::CoreRange;
using CoreRangeSet [[deprecated("Use tt::tt_metal::CoreRangeSet")]] = tt::tt_metal::CoreRangeSet;

// Deprecated function wrappers - use tt::tt_metal namespace versions instead
// template to depriorize the wrappers in overloading to avoid ambigous selection from compiler.

template <bool _compiler_deprioritize_this = true>
[[deprecated("Use tt::tt_metal::corerange_to_cores")]] inline std::vector<CoreCoord> corerange_to_cores(
    const CoreRangeSet& crs, std::optional<uint32_t> max_cores = std::nullopt, bool row_wise = false) {
    return tt::tt_metal::corerange_to_cores(crs, max_cores, row_wise);
}

template <bool _compiler_deprioritize_this = true>
[[deprecated("Use tt::tt_metal::grid_to_cores")]] inline std::vector<CoreCoord> grid_to_cores(
    uint32_t num_cores, uint32_t grid_size_x, uint32_t grid_size_y, bool row_wise = false) {
    return tt::tt_metal::grid_to_cores(num_cores, grid_size_x, grid_size_y, row_wise);
}

template <bool _compiler_deprioritize_this = true>
[[deprecated("Use tt::tt_metal::grid_to_cores")]] inline std::vector<CoreCoord> grid_to_cores(
    CoreCoord start, CoreCoord end, bool row_wise = false) {
    return tt::tt_metal::grid_to_cores(start, end, row_wise);
}

template <bool _compiler_deprioritize_this = true>
[[deprecated("Use tt::tt_metal::grid_to_cores_with_noop")]] inline std::vector<CoreCoord> grid_to_cores_with_noop(
    uint32_t bbox_x, uint32_t bbox_y, uint32_t grid_size_x, uint32_t grid_size_y, bool row_wise = false) {
    return tt::tt_metal::grid_to_cores_with_noop(bbox_x, bbox_y, grid_size_x, grid_size_y, row_wise);
}

template <bool _compiler_deprioritize_this = true>
[[deprecated("Use tt::tt_metal::grid_to_cores_with_noop")]] inline std::vector<CoreCoord> grid_to_cores_with_noop(
    const CoreRangeSet& used_cores, const CoreRangeSet& all_cores, bool row_wise = false) {
    return tt::tt_metal::grid_to_cores_with_noop(used_cores, all_cores, row_wise);
}

template <bool _compiler_deprioritize_this = true>
[[deprecated("Use tt::tt_metal::select_contiguous_range_from_corerangeset")]] inline std::optional<CoreRange>
select_contiguous_range_from_corerangeset(const CoreRangeSet& crs, uint32_t x, uint32_t y) {
    return tt::tt_metal::select_contiguous_range_from_corerangeset(crs, x, y);
}

template <bool _compiler_deprioritize_this = true>
[[deprecated("Use tt::tt_metal::select_from_corerangeset")]] inline CoreRangeSet select_from_corerangeset(
    const CoreRangeSet& crs, uint32_t start_index, uint32_t end_index, bool row_wise = false) {
    return tt::tt_metal::select_from_corerangeset(crs, start_index, end_index, row_wise);
}

template <>
struct fmt::formatter<tt::tt_metal::CoreRangeSet> {
    constexpr auto parse(format_parse_context& ctx) -> format_parse_context::iterator { return ctx.end(); }

    auto format(const tt::tt_metal::CoreRangeSet& core_range_set, format_context& ctx) const
        -> format_context::iterator;
};

template <>
struct fmt::formatter<tt::tt_metal::CoreRange> {
    constexpr auto parse(format_parse_context& ctx) -> format_parse_context::iterator { return ctx.end(); }

    auto format(const tt::tt_metal::CoreRange& core_range, format_context& ctx) const -> format_context::iterator;
};

template <>
struct fmt::formatter<tt::tt_metal::CoreCoord> {
    constexpr auto parse(format_parse_context& ctx) -> format_parse_context::iterator { return ctx.end(); }

    auto format(const tt::tt_metal::CoreCoord& core_coord, format_context& ctx) const -> format_context::iterator;
};

namespace std {

template <>
struct hash<tt::tt_metal::CoreRange> {
    std::size_t operator()(const tt::tt_metal::CoreRange& core_range) const;
};

template <>
struct hash<tt::tt_metal::CoreRangeSet> {
    std::size_t operator()(const tt::tt_metal::CoreRangeSet& core_range_set) const;
};

}  // namespace std

namespace ttsl::json {

template <>
struct to_json_t<tt::tt_metal::CoreCoord> {
    nlohmann::json operator()(const tt::tt_metal::CoreCoord& core_coord) noexcept;
};

template <>
struct from_json_t<tt::tt_metal::CoreCoord> {
    tt::tt_metal::CoreCoord operator()(const nlohmann::json& json) noexcept;
};

template <>
struct to_json_t<tt::tt_metal::CoreRange> {
    nlohmann::json operator()(const tt::tt_metal::CoreRange& core_range) noexcept;
};

template <>
struct from_json_t<tt::tt_metal::CoreRange> {
    tt::tt_metal::CoreRange operator()(const nlohmann::json& json) noexcept;
};

template <>
struct to_json_t<tt::tt_metal::CoreRangeSet> {
    nlohmann::json operator()(const tt::tt_metal::CoreRangeSet& core_range_set) noexcept;
};

template <>
struct from_json_t<tt::tt_metal::CoreRangeSet> {
    tt::tt_metal::CoreRangeSet operator()(const nlohmann::json& json) noexcept;
};

}  // namespace ttsl::json
