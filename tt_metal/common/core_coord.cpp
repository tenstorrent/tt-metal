// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/common/core_coord.h"

#include <algorithm>
#include <limits>
#include <mutex>
#include <optional>
#include <set>
#include <string>
#include <vector>

#include "third_party/umd/device/tt_xy_pair.h"
#include "tt_metal/common/assert.hpp"
#include "tt_metal/third_party/tracy/public/tracy/Tracy.hpp"
#include "tt_metal/tt_stl/reflection.hpp"
#include "tt_metal/tt_stl/span.hpp"

auto fmt::formatter<CoreCoord>::format(const CoreCoord &core_coord, format_context &ctx) const
    -> format_context::iterator {
    std::stringstream ss;
    ss << core_coord.str();
    return fmt::format_to(ctx.out(), "{}", ss.str());
}

std::string RelativeCoreCoord::str() const { return "(x=" + std::to_string(x) + ",y=" + std::to_string(y) + ")"; }

CoreCoord get_core_coord_from_relative(const RelativeCoreCoord &in, const CoreCoord &grid_size) {
    CoreCoord coord;
    coord.x = in.x + ((in.x < 0) ? grid_size.x : 0);
    coord.y = in.y + ((in.y < 0) ? grid_size.y : 0);
    return coord;
}

CoreRange::CoreRange(const CoreCoord &point) : start_coord(point), end_coord(point) {}

CoreRange::CoreRange(const CoreCoord &start_coord, const CoreCoord &end_coord) {
    TT_FATAL(
        end_coord.x >= start_coord.x and end_coord.y >= start_coord.y,
        "Invalid core range for start_coord: {}, end_coord: {}",
        start_coord.str(),
        end_coord.str());

    this->start_coord = start_coord;
    this->end_coord = end_coord;
}

std::optional<CoreRange> CoreRange::intersects(const CoreRange &other) const {
    std::size_t x1 = std::max(this->start_coord.x, other.start_coord.x);
    std::size_t y1 = std::max(this->start_coord.y, other.start_coord.y);
    std::size_t x2 = std::min(this->end_coord.x, other.end_coord.x);
    std::size_t y2 = std::min(this->end_coord.y, other.end_coord.y);
    if (x1 <= x2 and y1 <= y2) {
        return CoreRange({x1, y1}, {x2, y2});
    }

    return {};
}

bool CoreRange::adjacent(const CoreRange &other) const {
    std::size_t x1 = std::max(this->start_coord.x, other.start_coord.x);
    std::size_t y1 = std::max(this->start_coord.y, other.start_coord.y);
    std::size_t x2 = std::min(this->end_coord.x, other.end_coord.x);
    std::size_t y2 = std::min(this->end_coord.y, other.end_coord.y);
    return ((x2 + 1 == x1 && y1 <= y2) || (y2 + 1 == y1 && x1 <= x2));
}

bool CoreRange::contains(const CoreRange &other) const {
    return (other.start_coord.x >= this->start_coord.x) && (other.end_coord.x <= this->end_coord.x) &&
           (other.start_coord.y >= this->start_coord.y) && (other.end_coord.y <= this->end_coord.y);
}

bool CoreRange::contains(const CoreCoord &other) const {
    return (other.x >= this->start_coord.x) && (other.x <= this->end_coord.x) && (other.y >= this->start_coord.y) &&
           (other.y <= this->end_coord.y);
}

// Merge lined-up (in x or y dimension) intersecting/adjacent rectangles
std::optional<CoreRange> CoreRange::merge(const CoreRange &cr) const {
    if (this->intersects(cr) || this->adjacent(cr)) {
        if (this->start_coord.x == cr.start_coord.x && this->end_coord.x == cr.end_coord.x)
            return CoreRange(
                {this->start_coord.x, std::min(this->start_coord.y, cr.start_coord.y)},
                {this->end_coord.x, std::max(this->end_coord.y, cr.end_coord.y)});

        else if (this->start_coord.y == cr.start_coord.y && this->end_coord.y == cr.end_coord.y)
            return CoreRange(
                {std::min(this->start_coord.x, cr.start_coord.x), this->start_coord.y},
                {std::max(this->end_coord.x, cr.end_coord.x), this->end_coord.y});
    }
    return std::nullopt;
}

std::string CoreRange::str() const { return "[" + this->start_coord.str() + " - " + this->end_coord.str() + "]"; }

size_t CoreRange::size() const {
    return (this->end_coord.x - this->start_coord.x + 1) * (this->end_coord.y - this->start_coord.y + 1);
}

CoreCoord CoreRange::grid_size() const {
    return {this->end_coord.x - this->start_coord.x + 1, this->end_coord.y - this->start_coord.y + 1};
}

CoreRange::CoreIterator::CoreIterator(const CoreCoord &current, const CoreRange &core_range) :
    current_(current), range_(core_range) {}

CoreCoord &CoreRange::CoreIterator::operator*() { return current_; }

CoreRange::CoreIterator &CoreRange::CoreIterator::operator++() {
    CoreCoord next;

    const bool is_curr_core_at_end_of_row = current_.x == range_.end_coord.x;
    if (is_curr_core_at_end_of_row) {
        // Go to the beginning of the next row
        next.x = range_.start_coord.x;
        next.y = current_.y + 1;
    } else {
        next.x = current_.x + 1;
        next.y = current_.y;
    }

    current_ = next;
    return *this;
}

CoreRange::CoreIterator CoreRange::begin() const { return CoreRange::CoreIterator(this->start_coord, *this); }

CoreRange::CoreIterator CoreRange::end() const {
    const CoreCoord iterator_end(this->start_coord.x, this->end_coord.y + 1);
    return CoreRange::CoreIterator(iterator_end, *this);
}

bool CoreRange::CoreIterator::operator==(const CoreIterator &other) const { return current_ == other.current_; }

bool CoreRange::CoreIterator::operator!=(const CoreIterator &other) const { return !(current_ == other.current_); }

auto fmt::formatter<CoreRange>::format(const CoreRange &core_range, format_context &ctx) const
    -> format_context::iterator {
    std::stringstream ss;
    ss << core_range.str();
    return fmt::format_to(ctx.out(), "{}", ss.str());
}

CoreRangeSet::CoreRangeSet(const std::vector<CoreRange> &core_ranges) :
    ranges_(core_ranges.begin(), core_ranges.end()) {
    ZoneScoped;
    this->validate_no_overlap();
}

CoreRangeSet::CoreRangeSet(const std::set<CoreRange> &core_ranges) : ranges_(core_ranges.begin(), core_ranges.end()) {
    ZoneScoped;
    this->validate_no_overlap();
}

CoreRangeSet::CoreRangeSet(const CoreRange &core_range) : ranges_{core_range} {}

void swap(CoreRangeSet &first, CoreRangeSet &second) {
    std::scoped_lock lock(first.ranges_guard, second.ranges_guard);
    std::swap(first.ranges_, second.ranges_);
}

CoreRangeSet::CoreRangeSet(const CoreRangeSet &other) {
    std::scoped_lock lock(other.ranges_guard);
    this->ranges_ = other.ranges_;
}

CoreRangeSet &CoreRangeSet::operator=(const CoreRangeSet &other) {
    std::scoped_lock lock(other.ranges_guard);
    this->ranges_ = other.ranges_;
    return *this;
}

CoreRangeSet::CoreRangeSet(CoreRangeSet &&other) { swap(*this, other); }

CoreRangeSet &CoreRangeSet::operator=(CoreRangeSet &&other) {
    swap(*this, other);
    return *this;
}

CoreRangeSet::CoreRangeSet(std::vector<CoreRange> &&core_ranges) : ranges_(std::move(core_ranges)) {
    ZoneScoped;
    this->validate_no_overlap();
}

size_t CoreRangeSet::size() const { return ranges_.size(); }

template <typename T>
CoreRangeSet CoreRangeSet::merge(const T &other) const {
    size_t min_x = std::numeric_limits<size_t>::max(), max_x = 0, min_y = std::numeric_limits<size_t>::max(), max_y = 0;
    std::set<CoreRange> crs(this->ranges_.begin(), this->ranges_.end());
    crs.insert(other.begin(), other.end());

    for (const auto &cr : crs) {
        min_x = std::min(min_x, cr.start_coord.x);
        max_x = std::max(max_x, cr.end_coord.x);
        min_y = std::min(min_y, cr.start_coord.y);
        max_y = std::max(max_y, cr.end_coord.y);
    }

    // By overallocating by one x entry, we can avoid needing to check for
    // boundary conditions when iterating, since there'll always be one
    // last false entry
    bool grid[max_y + 1][max_x + 2];
    memset(grid, 0, sizeof(grid));

    for (const auto &cr : crs)
        for (unsigned y = cr.start_coord.y; y <= cr.end_coord.y; y++)
            for (unsigned x = cr.start_coord.x; x <= cr.end_coord.x; x++) grid[y][x] = true;

    crs.clear();
    for (unsigned y = min_y; y <= max_y; y++) {
        std::set<CoreRange> filter_set, tmp, new_crs;
        std::vector<CoreRange> ranges;
        for (unsigned x = min_x; x <= max_x + 1; x++) {
            if (grid[y][x]) {
                unsigned x_start = x;
                while (grid[y][x]) x++;
                ranges.push_back(CoreRange({x_start, y}, {x - 1, y}));
            }
        }

        for (const auto &cr : ranges) {
            for (const auto &prev_cr : crs) {
                if (auto merged = cr.merge(prev_cr)) {
                    new_crs.insert(merged.value());
                    filter_set.insert(prev_cr);
                    filter_set.insert(cr);
                }
            }
            crs.insert(cr);
        }
        // Set(A) = Set(A) - Set(B)
        std::set_difference(
            std::make_move_iterator(crs.begin()),
            std::make_move_iterator(crs.end()),
            filter_set.begin(),
            filter_set.end(),
            std::inserter(tmp, tmp.end()));
        crs.swap(tmp);
        crs.insert(new_crs.begin(), new_crs.end());
    }
    return CoreRangeSet(crs);
}

template CoreRangeSet CoreRangeSet::merge<std::vector<CoreRange>>(const std::vector<CoreRange> &other) const;
template CoreRangeSet CoreRangeSet::merge<std::set<CoreRange>>(const std::set<CoreRange> &other) const;

template <>
CoreRangeSet CoreRangeSet::merge<CoreRangeSet>(const CoreRangeSet &other) const {
    return this->merge(other.ranges());
}

bool CoreRangeSet::core_coord_in_core_ranges(const CoreCoord &core_coord) const {
    ZoneScoped;
    for (const auto &cr : this->ranges_) {
        if (cr.contains(core_coord))
            return true;
    }
    return false;
}

bool CoreRangeSet::intersects(const CoreRange &cr) const {
    for (const auto &local_cr : this->ranges_) {
        if (local_cr.intersects(cr))
            return true;
    }
    return false;
}

const std::vector<CoreRange> &CoreRangeSet::ranges() const { return this->ranges_; }

std::string CoreRangeSet::str() const {
    if (this->ranges().size() > 0) {
        std::string core_range_set_str = "{";
        for (const auto &core_range : this->ranges_) {
            core_range_set_str += core_range.str() + ", ";
        }
        core_range_set_str[core_range_set_str.length() - 2] = '}';
        core_range_set_str.pop_back();
        return core_range_set_str;
    } else {
        return "{}";
    }
}

uint32_t CoreRangeSet::num_cores() const {
    uint32_t num_cores = 0;
    for (const auto &core_range : this->ranges()) {
        num_cores += core_range.size();
    }
    return num_cores;
}

CoreRange CoreRangeSet::bounding_box() const {
    TT_FATAL(this->ranges().size() > 0, "Cannot get bounding_box of an empty CoreRangeSet!");
    size_t min_x = UINT32_MAX, min_y = UINT32_MAX, max_x = 0, max_y = 0;
    for (const auto &cr : this->ranges()) {
        min_x = std::min(min_x, cr.start_coord.x);
        max_x = std::max(max_x, cr.end_coord.x);
        min_y = std::min(min_y, cr.start_coord.y);
        max_y = std::max(max_y, cr.end_coord.y);
    }
    return {{min_x, min_y}, {max_x, max_y}};
}

void CoreRangeSet::validate_no_overlap() {
    if (this->ranges_.size() < 2) {
        return;
    }
    for (auto outer_it = this->ranges_.begin(); outer_it != this->ranges_.end() - 1; outer_it++) {
        for (auto inner_it = outer_it + 1; inner_it != this->ranges_.end(); inner_it++) {
            CoreRange &first_core_range = *outer_it;
            CoreRange &second_core_range = *inner_it;
            bool first_core_left_of_second = first_core_range.end_coord.x < second_core_range.start_coord.x;
            bool first_core_right_of_second = first_core_range.start_coord.x > second_core_range.end_coord.x;
            bool first_core_above_second = first_core_range.end_coord.y < second_core_range.start_coord.y;
            bool first_core_below_second = first_core_range.start_coord.y > second_core_range.end_coord.y;
            auto no_overlap = first_core_left_of_second or first_core_right_of_second or first_core_above_second or
                              first_core_below_second;
            if (not no_overlap) {
                TT_THROW(
                    "Cannot create CoreRangeSet with specified core ranges because core ranges {} and {} overlap!",
                    first_core_range.str(),
                    second_core_range.str());
            }
        }
    }
}

bool operator==(const CoreRangeSet &a, const CoreRangeSet &b) {
    if (a.ranges().size() == b.ranges().size()) {
        auto range_a = a.ranges();
        auto range_b = b.ranges();
        for (auto it_a = range_a.begin(), it_b = range_b.begin(); it_a != range_a.end(); it_a++, it_b++) {
            if (*it_a != *it_b) {
                return false;
            }
        }
        return true;
    }
    return false;
}

std::vector<CoreCoord> grid_to_cores(uint32_t num_cores, uint32_t grid_size_x, uint32_t grid_size_y, bool row_wise) {
    std::vector<CoreCoord> cores;
    cores.reserve(num_cores);
    TT_ASSERT(
        num_cores <= grid_size_x * grid_size_y,
        "Number of cores {} exceeds grid size {}x{}",
        num_cores,
        grid_size_x,
        grid_size_y);
    if (row_wise) {
        for (uint32_t i = 0; i < num_cores; ++i) {
            cores.push_back({i % grid_size_x, i / grid_size_x});
        }
    } else {
        for (uint32_t i = 0; i < num_cores; ++i) {
            cores.push_back({i / grid_size_y, i % grid_size_y});
        }
    }
    return cores;
}

std::vector<CoreCoord> grid_to_cores(CoreCoord start, CoreCoord end, bool row_wise) {
    std::vector<CoreCoord> cores;
    auto num_cores_x = (end.x + 1) - start.x;
    auto num_cores_y = (end.y + 1) - start.y;
    uint32_t num_cores = num_cores_x * num_cores_y;
    cores.reserve(num_cores);
    if (row_wise) {
        for (uint32_t j = start.y; j < (end.y + 1); j++) {
            for (uint32_t i = start.x; i < (end.x + 1); i++) {
                cores.push_back({i, j});
            }
        }

    } else {
        for (uint32_t i = start.x; i < (end.x + 1); i++) {
            for (uint32_t j = start.y; j < (end.y + 1); j++) {
                cores.push_back({i, j});
            }
        }
    }
    return cores;
}

// Noop cores are appended at the end with no guarantees on ordering
std::vector<CoreCoord> grid_to_cores_with_noop(
    const uint32_t bbox_x,
    const uint32_t bbox_y,
    const uint32_t grid_size_x,
    const uint32_t grid_size_y,
    const bool row_wise) {
    ZoneScoped;
    std::vector<CoreCoord> cores;
    cores.reserve(grid_size_x * grid_size_y);
    TT_ASSERT(bbox_x < grid_size_x);
    TT_ASSERT(bbox_y < grid_size_y);
    const uint32_t box_size_x = bbox_x + 1;
    const uint32_t box_size_y = bbox_y + 1;

    if (row_wise) {
        for (uint32_t i = 0; i < box_size_x * box_size_y; ++i) {
            cores.push_back({i % box_size_x, i / box_size_x});
        }
    } else {
        for (uint32_t i = 0; i < box_size_x * box_size_y; ++i) {
            cores.push_back({i / box_size_y, i % box_size_y});
        }
    }

    // Right rectangle noops
    for (uint32_t x = box_size_x; x < grid_size_x; ++x) {
        for (uint32_t y = 0; y < grid_size_y; ++y) {
            cores.push_back({x, y});
        }
    }

    // Bottom rectangle noops
    for (uint32_t y = box_size_y; y < grid_size_y; ++y) {
        for (uint32_t x = 0; x < box_size_x; ++x) {
            cores.push_back({x, y});
        }
    }

    return cores;
}

std::vector<CoreCoord> corerange_to_cores(const CoreRangeSet &crs, std::optional<uint32_t> max_cores, bool row_wise) {
    uint32_t num_total_cores = 0;
    std::vector<CoreCoord> all_cores;
    uint32_t offset = 0;

    for (auto core_range : crs.ranges()) {
        auto start_coord = core_range.start_coord;
        auto end_coord = core_range.end_coord;
        auto cores = grid_to_cores(start_coord, end_coord, row_wise);
        if (max_cores.has_value()) {
            if (all_cores.size() + cores.size() > max_cores.value()) {
                uint32_t num_cores_to_add = max_cores.value() - all_cores.size();
                all_cores.insert(all_cores.end(), cores.begin(), cores.begin() + num_cores_to_add);
            } else {
                all_cores.insert(all_cores.end(), cores.begin(), cores.end());
            }
        } else {
            all_cores.insert(all_cores.end(), cores.begin(), cores.end());
        }
    }

    return all_cores;
}

bool operator!=(const CoreRangeSet &a, const CoreRangeSet &b) { return !(a == b); }

auto fmt::formatter<CoreRangeSet>::format(const CoreRangeSet &core_range_set, format_context &ctx) const
    -> format_context::iterator {
    std::stringstream ss;
    ss << core_range_set.str();
    return fmt::format_to(ctx.out(), "{}", ss.str());
}

namespace std {

std::size_t hash<RelativeCoreCoord>::operator()(RelativeCoreCoord const &o) const {
    std::size_t seed = 0;
    seed = std::hash<std::size_t>()(o.x) ^ std::hash<std::size_t>()(o.y) << 1;
    return seed;
}

std::size_t hash<CoreRange>::operator()(const CoreRange &core_range) const {
    std::size_t seed = 0;
    seed = std::hash<CoreCoord>{}(core_range.start_coord) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    seed = std::hash<CoreCoord>{}(core_range.end_coord) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    return seed;
}

std::size_t hash<CoreRangeSet>::operator()(const CoreRangeSet &core_range_set) const {
    std::size_t seed = 0;
    for (const auto &core_range : core_range_set.ranges()) {
        seed = std::hash<CoreRange>{}(core_range) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    }
    return seed;
}

}  // namespace std

namespace tt::stl::json {

nlohmann::json to_json_t<CoreCoord>::operator()(const CoreCoord &core_coord) noexcept {
    return {{"x", to_json(core_coord.x)}, {"y", to_json(core_coord.y)}};
}

CoreCoord from_json_t<CoreCoord>::operator()(const nlohmann::json &json) noexcept {
    return {from_json<uint32_t>(json.at("x")), from_json<uint32_t>(json.at("y"))};
}

nlohmann::json to_json_t<RelativeCoreCoord>::operator()(const RelativeCoreCoord &relative_core_coord) noexcept {
    return {{"x", to_json(relative_core_coord.x)}, {"y", to_json(relative_core_coord.y)}};
}

RelativeCoreCoord from_json_t<RelativeCoreCoord>::operator()(const nlohmann::json &json) noexcept {
    return {from_json<int32_t>(json.at("x")), from_json<int32_t>(json.at("y"))};
}

nlohmann::json to_json_t<CoreRange>::operator()(const CoreRange &core_range) noexcept {
    return {{"start", to_json(core_range.start_coord)}, {"end", to_json(core_range.end_coord)}};
}

CoreRange from_json_t<CoreRange>::operator()(const nlohmann::json &json) noexcept {
    return {from_json<CoreCoord>(json.at("start")), from_json<CoreCoord>(json.at("end"))};
}

nlohmann::json to_json_t<CoreRangeSet>::operator()(const CoreRangeSet &core_range_set) noexcept {
    nlohmann::json core_range_set_json = nlohmann::json::array();
    return to_json(core_range_set.ranges());
}

CoreRangeSet from_json_t<CoreRangeSet>::operator()(const nlohmann::json &json) noexcept {
    return CoreRangeSet(from_json<std::vector<CoreRange>>(json));
}

}  // namespace tt::stl::json
