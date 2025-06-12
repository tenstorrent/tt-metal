// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <assert.hpp>
#include <boost/container/vector.hpp>
#include <boost/move/utility_core.hpp>
#include <mesh_coord.hpp>
#include <tt_stl/span.hpp>
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <initializer_list>
#include <numeric>
#include <optional>
#include <ostream>
#include <utility>
#include <vector>

#include "shape_base.hpp"
#include <tt_stl/small_vector.hpp>

namespace tt::tt_metal::distributed {
namespace {

// Returns the last valid coordinate for the provided `shape`.
MeshCoordinate shape_back(const MeshShape& shape) {
    tt::stl::SmallVector<uint32_t> coords;
    for (int i = 0; i < shape.dims(); i++) {
        coords.push_back(shape[i] - 1);
    }
    return MeshCoordinate(coords);
}

// Returns a list of dimensions that differ between the two ranges.
std::vector<size_t> find_diff_dimensions(const MeshCoordinateRange& a, const MeshCoordinateRange& b) {
    TT_ASSERT(a.dims() == b.dims(), "Cannot compare ranges with different dimensions: {} != {}", a.dims(), b.dims());

    std::vector<size_t> diff_dims;
    for (size_t i = 0; i < a.dims(); ++i) {
        if (a.start_coord()[i] != b.start_coord()[i] || a.end_coord()[i] != b.end_coord()[i]) {
            diff_dims.push_back(i);
        }
    }
    return diff_dims;
}

// Returns true if the two ranges are mergeable; that is, when 2 ranges can be replaced by one.
// Ranges must either be identical, be adjacent along exactly one dimension, or contain each other.
bool check_mergeable(const MeshCoordinateRange& a, const MeshCoordinateRange& b) {
    TT_ASSERT(a.dims() == b.dims(), "Cannot compare ranges with different dimensions: {} != {}", a.dims(), b.dims());

    auto diff_dims = find_diff_dimensions(a, b);
    if (diff_dims.empty()) {
        // Ranges are identical.
        return true;
    } else if (diff_dims.size() == 1) {
        // Ranges are adjacent or overlap along one dimension.
        size_t diff_dim = diff_dims[0];
        return std::max(a.start_coord()[diff_dim], b.start_coord()[diff_dim]) <=
               std::min(a.end_coord()[diff_dim], b.end_coord()[diff_dim]) + 1;
    } else {
        return a.contains(b) || b.contains(a);
    }
}

}  // namespace

MeshShape::MeshShape(uint32_t x) : MeshShape({x}) {}
MeshShape::MeshShape(uint32_t x, uint32_t y) : MeshShape({x, y}) {}
MeshShape::MeshShape(uint32_t x, uint32_t y, uint32_t z) : MeshShape({x, y, z}) {}

MeshShape::MeshShape(const tt::stl::SmallVector<uint32_t>& shape) : ShapeBase(shape) { compute_strides(); }
MeshShape::MeshShape(tt::stl::SmallVector<uint32_t>&& shape) : ShapeBase(std::move(shape)) { compute_strides(); }
MeshShape::MeshShape(std::initializer_list<uint32_t> ilist) : ShapeBase(ilist) { compute_strides(); }
MeshShape::MeshShape(tt::stl::Span<const uint32_t> span) : ShapeBase(span) { compute_strides(); }

void MeshShape::compute_strides() {
    TT_FATAL(dims() > 0, "MeshShape cannot have 0 dimension.");
    size_t stride = 1;
    strides_.resize(dims());
    for (int dim = dims() - 1; dim >= 0; --dim) {
        strides_[dim] = stride;
        stride *= (*this)[dim];
    }
}

size_t MeshShape::get_stride(size_t dim) const { return strides_[dim]; }

size_t MeshShape::dims() const { return size(); }
size_t MeshShape::mesh_size() const {
    return empty() ? 0 : std::accumulate(value_.begin(), value_.end(), 1, std::multiplies<size_t>());
}

bool operator==(const MeshShape& lhs, const MeshShape& rhs) = default;
bool operator!=(const MeshShape& lhs, const MeshShape& rhs) = default;

std::ostream& operator<<(std::ostream& os, const MeshShape& shape) {
    os << "MeshShape([";
    for (size_t i = 0; i < shape.dims(); ++i) {
        if (i > 0) {
            os << ", ";
        }
        os << shape[i];
    }
    os << "])";
    return os;
}

bool is_line_topology(const MeshShape& shape) {
    return std::count_if(shape.cbegin(), shape.cend(), [](size_t dim) { return dim != 1; }) <= 1;
}

MeshCoordinate::MeshCoordinate(uint32_t x) : value_({x}) {}
MeshCoordinate::MeshCoordinate(uint32_t x, uint32_t y) : value_({x, y}) {}
MeshCoordinate::MeshCoordinate(uint32_t x, uint32_t y, uint32_t z) : value_({x, y, z}) {}

MeshCoordinate::MeshCoordinate(tt::stl::Span<const uint32_t> coords) : value_(coords.begin(), coords.end()) {}

MeshCoordinate MeshCoordinate::zero_coordinate(size_t dimensions) {
    return MeshCoordinate(tt::stl::SmallVector<uint32_t>(dimensions, 0));
}

size_t MeshCoordinate::dims() const { return value_.size(); }
tt::stl::Span<const uint32_t> MeshCoordinate::coords() const { return value_; }
uint32_t MeshCoordinate::operator[](size_t dim) const { return value_[dim]; }

bool operator==(const MeshCoordinate& lhs, const MeshCoordinate& rhs) {
    return lhs.dims() == rhs.dims() && std::equal(lhs.coords().begin(), lhs.coords().end(), rhs.coords().begin());
}
bool operator!=(const MeshCoordinate& lhs, const MeshCoordinate& rhs) { return !(lhs == rhs); }

bool operator<(const MeshCoordinate& lhs, const MeshCoordinate& rhs) {
    TT_FATAL(
        lhs.dims() == rhs.dims(),
        "Cannot compare coordinates with different dimensions: {} != {}",
        lhs.dims(),
        rhs.dims());
    for (size_t i = 0; i < lhs.dims(); ++i) {
        if (lhs[i] != rhs[i]) {
            return lhs[i] < rhs[i];
        }
    }
    return false;
}
bool operator>(const MeshCoordinate& lhs, const MeshCoordinate& rhs) { return rhs < lhs; }
bool operator<=(const MeshCoordinate& lhs, const MeshCoordinate& rhs) { return !(lhs > rhs); }
bool operator>=(const MeshCoordinate& lhs, const MeshCoordinate& rhs) { return !(lhs < rhs); }

std::ostream& operator<<(std::ostream& os, const MeshCoordinate& coord) {
    os << "MeshCoordinate([";
    for (size_t i = 0; i < coord.dims(); ++i) {
        if (i > 0) {
            os << ", ";
        }
        os << coord[i];
    }
    os << "])";
    return os;
}

MeshCoordinateRange::MeshCoordinateRange(const MeshCoordinate& start, const MeshCoordinate& end) :
    start_(start), end_(end) {
    TT_FATAL(
        start.dims() == end.dims(),
        "Start and end dimensions of a coordinate range do not match: {} != {}",
        start.dims(),
        end.dims());
    for (size_t i = 0; i < start.dims(); ++i) {
        TT_FATAL(start[i] <= end[i], "Start coordinate is greater than end coordinate: {} > {}", start, end);
    }
}

MeshCoordinateRange::MeshCoordinateRange(const MeshShape& shape) :
    MeshCoordinateRange(MeshCoordinate::zero_coordinate(shape.dims()), shape_back(shape)) {}

MeshCoordinateRange::MeshCoordinateRange(const MeshCoordinate& coord) : start_(coord), end_(coord) {}

size_t MeshCoordinateRange::dims() const { return start_.dims(); }
const MeshCoordinate& MeshCoordinateRange::start_coord() const { return start_; }
const MeshCoordinate& MeshCoordinateRange::end_coord() const { return end_; }

MeshShape MeshCoordinateRange::shape() const {
    tt::stl::SmallVector<uint32_t> shape_dims;
    for (size_t i = 0; i < dims(); ++i) {
        shape_dims.push_back(end_[i] - start_[i] + 1);
    }
    return MeshShape(shape_dims);
}

bool MeshCoordinateRange::contains(const MeshCoordinate& coord) const {
    TT_FATAL(coord.dims() == dims(), "Coordinate dimensions do not match: {} != {}", coord.dims(), dims());
    for (int i = 0; i < coord.dims(); ++i) {
        if (coord[i] < start_[i] || coord[i] > end_[i]) {
            return false;
        }
    }
    return true;
}

bool MeshCoordinateRange::contains(const MeshCoordinateRange& range) const {
    return contains(range.start_coord()) && contains(range.end_coord());
}

bool MeshCoordinateRange::intersects(const MeshCoordinateRange& range) const {
    TT_FATAL(range.dims() == dims(), "Coordinate dimensions do not match: {} != {}", range.dims(), dims());
    for (int i = 0; i < range.dims(); ++i) {
        if (range.end_coord()[i] < start_[i] || range.start_coord()[i] > end_[i]) {
            return false;
        }
    }
    return true;
}

std::optional<MeshCoordinateRange> MeshCoordinateRange::intersection(const MeshCoordinateRange& range) const {
    if (!intersects(range)) {
        return std::nullopt;
    }

    tt::stl::SmallVector<uint32_t> intersection_start(dims(), 0);
    tt::stl::SmallVector<uint32_t> intersection_end(dims(), 0);
    for (size_t i = 0; i < dims(); ++i) {
        intersection_start[i] = std::max(start_coord()[i], range.start_coord()[i]);
        intersection_end[i] = std::min(end_coord()[i], range.end_coord()[i]);
    }
    return MeshCoordinateRange(MeshCoordinate(intersection_start), MeshCoordinate(intersection_end));
}

MeshCoordinateRange::Iterator::Iterator(
    const MeshCoordinateRange* range, const MeshCoordinate& current, size_t linear_index) :
    range_(range), current_coord_(current), linear_index_(linear_index) {}

MeshCoordinateRange::Iterator MeshCoordinateRange::Iterator::operator++(int) {
    Iterator tmp = *this;
    ++(*this);
    return tmp;
}

MeshCoordinateRange::Iterator& MeshCoordinateRange::Iterator::operator++() {
    ++linear_index_;

    tt::stl::SmallVector<uint32_t> new_coords(current_coord_.coords().begin(), current_coord_.coords().end());
    for (int i = new_coords.size() - 1; i >= 0; --i) {
        auto& dimension_value = new_coords[i];
        if (++dimension_value > range_->end_coord()[i]) {
            dimension_value = range_->start_coord()[i];
        } else {
            break;
        }
    }
    current_coord_ = MeshCoordinate(new_coords);
    return *this;
}
const MeshCoordinate& MeshCoordinateRange::Iterator::operator*() const { return current_coord_; }
bool MeshCoordinateRange::Iterator::operator==(const Iterator& other) const {
    return range_ == other.range_ && linear_index_ == other.linear_index_;
}
bool MeshCoordinateRange::Iterator::operator!=(const Iterator& other) const { return !(*this == other); }

MeshCoordinateRange::Iterator MeshCoordinateRange::begin() const { return Iterator(this, start_, /*linear_index=*/0); }
MeshCoordinateRange::Iterator MeshCoordinateRange::end() const {
    size_t range_size = 1;
    for (size_t i = 0; i < start_.dims(); ++i) {
        range_size *= end_[i] - start_[i] + 1;
    }
    // Set `start_` coordinate but `range_size` linear index as the wrap around condition.
    return Iterator(this, start_, range_size);
}

bool operator==(const MeshCoordinateRange& lhs, const MeshCoordinateRange& rhs) {
    return lhs.start_coord() == rhs.start_coord() && lhs.end_coord() == rhs.end_coord();
}
bool operator!=(const MeshCoordinateRange& lhs, const MeshCoordinateRange& rhs) { return !(lhs == rhs); }

std::ostream& operator<<(std::ostream& os, const MeshCoordinateRange& range) {
    os << "MeshCoordinateRange(start=" << range.start_coord() << ", end=" << range.end_coord() << ")";
    return os;
}

size_t to_linear_index(const MeshShape& shape, const MeshCoordinate& coord) {
    TT_FATAL(
        shape.dims() == coord.dims(),
        "Shape and coordinate dimensions do not match: {} != {}",
        shape.dims(),
        coord.dims());

    size_t linear_index = 0;
    for (size_t dim = 0; dim < coord.dims(); ++dim) {
        TT_FATAL(coord[dim] < shape[dim], "Coordinate {} is out of bounds for shape {}", coord, shape);
        linear_index += coord[dim] * shape.get_stride(dim);
    }
    return linear_index;
}

MeshCoordinateRangeSet::MeshCoordinateRangeSet(const MeshCoordinateRange& range) { ranges_.push_back(range); }

void MeshCoordinateRangeSet::merge(const MeshCoordinateRange& to_merge) {
    TT_FATAL(
        ranges_.empty() || ranges_.front().dims() == to_merge.dims(),
        "Cannot merge range with different dimensions into a range set: {} != {}",
        ranges_.front().dims(),
        to_merge.dims());

    // Iteratively merge the new range with existing ranges until no more merges are possible.
    std::vector<MeshCoordinateRange> add_back;
    MeshCoordinateRange merged = to_merge;
    while (true) {
        bool did_merge = false;
        for (auto it = ranges_.begin(); it != ranges_.end();) {
            if (check_mergeable(merged, *it)) {
                // Can replace `it` + `merged` with a single new range.
                tt::stl::SmallVector<uint32_t> new_start;
                tt::stl::SmallVector<uint32_t> new_end;
                for (size_t i = 0; i < merged.dims(); ++i) {
                    new_start.push_back(std::min(merged.start_coord()[i], it->start_coord()[i]));
                    new_end.push_back(std::max(merged.end_coord()[i], it->end_coord()[i]));
                }
                merged = MeshCoordinateRange(MeshCoordinate(new_start), MeshCoordinate(new_end));
                ranges_.erase(it);
                did_merge = true;
                break;
            } else if (merged.intersects(*it) || it->intersects(merged)) {
                // There is an intersection between `merged` and `it`.
                // For simplicity, erase the entire `it`, but add back what isn't present in `merged`.
                for (const auto& coord : *it) {
                    if (!merged.contains(coord)) {
                        add_back.push_back(MeshCoordinateRange(coord));
                    }
                }
                it = ranges_.erase(it);
            } else {
                // Cannot merge nor intersect with `it`, proceed to the next element.
                ++it;
            }
        }

        if (!did_merge) {
            break;
        }
    }
    ranges_.push_back(merged);

    // Sort the ranges to ensure deterministic order.
    std::sort(ranges_.begin(), ranges_.end(), [](const auto& a, const auto& b) {
        return (a.start_coord() != b.start_coord()) ? a.start_coord() < b.start_coord() : a.end_coord() < b.end_coord();
    });

    // Merge back the ranges that were removed.
    for (const auto& range : add_back) {
        merge(range);
    }
}

MeshCoordinateRangeSet subtract(const MeshCoordinateRange& parent, const MeshCoordinateRange& intersection) {
    TT_FATAL(
        parent.dims() == intersection.dims(),
        "Parent and intersection dimensions do not match: {} != {}",
        parent.dims(),
        intersection.dims());

    MeshCoordinateRangeSet complement_set;
    if (parent == intersection) {
        return complement_set;
    }

    if (!parent.intersects(intersection)) {
        complement_set.merge(parent);
        return complement_set;
    }

    // Fast path: parent and intersection differ in exactly one dimension.
    auto diff_dims = find_diff_dimensions(parent, intersection);
    if (diff_dims.size() == 1) {
        const size_t diff_dim = diff_dims[0];

        // Left complement: portion before the intersection in diff_dim.
        if (parent.start_coord()[diff_dim] < intersection.start_coord()[diff_dim]) {
            tt::stl::SmallVector<uint32_t> left_start;
            tt::stl::SmallVector<uint32_t> left_end;
            for (size_t i = 0; i < parent.dims(); ++i) {
                if (i == diff_dim) {
                    left_start.push_back(parent.start_coord()[i]);
                    left_end.push_back(intersection.start_coord()[i] - 1);
                } else {
                    left_start.push_back(parent.start_coord()[i]);
                    left_end.push_back(parent.end_coord()[i]);
                }
            }
            complement_set.merge(MeshCoordinateRange(MeshCoordinate(left_start), MeshCoordinate(left_end)));
        }

        // Right complement: portion after the intersection in diff_dim.
        if (intersection.end_coord()[diff_dim] < parent.end_coord()[diff_dim]) {
            tt::stl::SmallVector<uint32_t> right_start;
            tt::stl::SmallVector<uint32_t> right_end;
            for (size_t i = 0; i < parent.dims(); ++i) {
                if (i == diff_dim) {
                    right_start.push_back(intersection.end_coord()[i] + 1);
                    right_end.push_back(parent.end_coord()[i]);
                } else {
                    right_start.push_back(parent.start_coord()[i]);
                    right_end.push_back(parent.end_coord()[i]);
                }
            }
            complement_set.merge(MeshCoordinateRange(MeshCoordinate(right_start), MeshCoordinate(right_end)));
        }

        return complement_set;
    } else {
        // Slow path: iterate over all coordinates in the parent range, and create ranges for the complement.
        for (const auto& coord : parent) {
            if (!intersection.contains(coord)) {
                complement_set.merge(MeshCoordinateRange(coord, coord));
            }
        }
        return complement_set;
    }
}

std::vector<MeshCoordinate> MeshCoordinateRangeSet::coords() const {
    std::vector<MeshCoordinate> coords;
    for (const auto& range : ranges_) {
        for (const auto& coord : range) {
            coords.push_back(coord);
        }
    }
    std::sort(coords.begin(), coords.end());
    return coords;
}

bool operator==(const MeshCoordinateRangeSet& lhs, const MeshCoordinateRangeSet& rhs) {
    return lhs.ranges() == rhs.ranges();
}

bool operator!=(const MeshCoordinateRangeSet& lhs, const MeshCoordinateRangeSet& rhs) { return !(lhs == rhs); }

std::ostream& operator<<(std::ostream& os, const MeshCoordinateRangeSet& range_set) {
    os << "MeshCoordinateRangeSet([";
    for (size_t i = 0; i < range_set.size(); ++i) {
        if (i > 0) {
            os << ", ";
        }
        os << range_set.ranges()[i];
    }
    os << "])";
    return os;
}

}  // namespace tt::tt_metal::distributed
