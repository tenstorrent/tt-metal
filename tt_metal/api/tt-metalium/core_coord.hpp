// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <fmt/base.h>
#include <nlohmann/json.hpp>
#include <stdint.h>
#include <tt_stl/reflection.hpp>
#include <tt_stl/span.hpp>
#include <algorithm>
#include <cstddef>
#include <functional>
#include <optional>
#include <set>
#include <string>
#include <vector>

#include <umd/device/tt_xy_pair.h>
#include <umd/device/types/xy_pair.h>

namespace tt {
namespace stl {
namespace json {
template <typename T>
struct from_json_t;
template <typename T>
struct to_json_t;
}  // namespace json
}  // namespace stl
}  // namespace tt

using CoreCoord = tt_xy_pair;

class CoreRangeSet;

template <>
struct fmt::formatter<CoreCoord> {
    constexpr auto parse(format_parse_context& ctx) -> format_parse_context::iterator { return ctx.end(); }

    auto format(const CoreCoord& core_coord, format_context& ctx) const -> format_context::iterator;
};

constexpr inline bool operator<=(const CoreCoord& a, const CoreCoord& b) { return (a < b) or (a == b); }

struct RelativeCoreCoord {
    long x = 0;
    long y = 0;

    std::string str() const;
};

constexpr inline bool operator==(const RelativeCoreCoord& a, const RelativeCoreCoord& b) {
    return a.x == b.x && a.y == b.y;
}

constexpr inline bool operator!=(const RelativeCoreCoord& a, const RelativeCoreCoord& b) { return !(a == b); }

CoreCoord get_core_coord_from_relative(const RelativeCoreCoord& in, const CoreCoord& grid_size);

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

template <>
struct fmt::formatter<CoreRange> {
    constexpr auto parse(format_parse_context& ctx) -> format_parse_context::iterator { return ctx.end(); }

    auto format(const CoreRange& core_range, format_context& ctx) const -> format_context::iterator;
};

class CoreRangeSet {
public:
    CoreRangeSet(tt::stl::Span<const CoreRange> core_ranges);

    CoreRangeSet(const std::set<CoreRange>& core_ranges);

    CoreRangeSet(const CoreRange& core_range);

    CoreRangeSet(tt::stl::Span<const CoreCoord> core_coords);

    CoreRangeSet() = default;

    friend void swap(CoreRangeSet& first, CoreRangeSet& second);

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

CoreRangeSet select_from_corerange(
    const CoreRangeSet& crs, uint32_t start_index, uint32_t end_index, bool row_wise = false);

bool operator!=(const CoreRangeSet& a, const CoreRangeSet& b);

template <>
struct fmt::formatter<CoreRangeSet> {
    constexpr auto parse(format_parse_context& ctx) -> format_parse_context::iterator { return ctx.end(); }

    auto format(const CoreRangeSet& core_range_set, format_context& ctx) const -> format_context::iterator;
};

// Adding to tt::tt_metal namespace as we transition to moving this out of global namespace eventually.
namespace tt::tt_metal {
using ::CoreCoord;
using ::CoreRange;
using ::CoreRangeSet;
}  // namespace tt::tt_metal

namespace std {

template <>
struct hash<CoreRange> {
    std::size_t operator()(const CoreRange& core_range) const;
};

template <>
struct hash<RelativeCoreCoord> {
    std::size_t operator()(RelativeCoreCoord const& o) const;
};

template <>
struct hash<CoreRangeSet> {
    std::size_t operator()(const CoreRangeSet& core_range_set) const;
};

}  // namespace std

namespace ttsl::json {

template <>
struct to_json_t<CoreCoord> {
    nlohmann::json operator()(const CoreCoord& core_coord) noexcept;
};

template <>
struct from_json_t<CoreCoord> {
    CoreCoord operator()(const nlohmann::json& json) noexcept;
};

template <>
struct to_json_t<RelativeCoreCoord> {
    nlohmann::json operator()(const RelativeCoreCoord& relative_core_coord) noexcept;
};

template <>
struct from_json_t<RelativeCoreCoord> {
    RelativeCoreCoord operator()(const nlohmann::json& json) noexcept;
};

template <>
struct to_json_t<CoreRange> {
    nlohmann::json operator()(const CoreRange& core_range) noexcept;
};

template <>
struct from_json_t<CoreRange> {
    CoreRange operator()(const nlohmann::json& json) noexcept;
};

template <>
struct to_json_t<CoreRangeSet> {
    nlohmann::json operator()(const CoreRangeSet& core_range_set) noexcept;
};

template <>
struct from_json_t<CoreRangeSet> {
    CoreRangeSet operator()(const nlohmann::json& json) noexcept;
};

}  // namespace ttsl::json
