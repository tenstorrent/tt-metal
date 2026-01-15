// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt_stl/assert.hpp>

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
    }
    if (diff_dims.size() == 1) {
        // Ranges are adjacent or overlap along one dimension.
        size_t diff_dim = diff_dims[0];
        return std::max(a.start_coord()[diff_dim], b.start_coord()[diff_dim]) <=
               std::min(a.end_coord()[diff_dim], b.end_coord()[diff_dim]) + 1;
    }
    return a.contains(b) || b.contains(a);
}

int32_t normalize_index(int32_t index, int32_t size) {
    int32_t normalized_index = index;
    if (normalized_index < 0) {
        normalized_index += size;
    }
    TT_FATAL(
        normalized_index >= 0 && normalized_index < size,
        "Index out of bounds: {} not in [{}, {})",
        index,
        -size,
        size);

    return normalized_index;
}

}  // namespace

MeshShape::MeshShape(uint32_t s) : MeshShape({s}) {}
MeshShape::MeshShape(uint32_t s0, uint32_t s1) : MeshShape({s0, s1}) {}
MeshShape::MeshShape(uint32_t s0, uint32_t s1, uint32_t s2) : MeshShape({s0, s1, s2}) {}

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
    return empty() ? 0
                   : std::accumulate(value_.begin(), value_.end(), static_cast<size_t>(1), std::multiplies<size_t>());
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

bool MeshShape::is_line_topology() const {
    return std::count_if(cbegin(), cend(), [](size_t dim) { return dim != 1; }) <= 1;
}

MeshCoordinate::MeshCoordinate(uint32_t c) : value_({c}) {}
MeshCoordinate::MeshCoordinate(uint32_t c0, uint32_t c1) : value_({c0, c1}) {}
MeshCoordinate::MeshCoordinate(uint32_t c0, uint32_t c1, uint32_t c2) : value_({c0, c1, c2}) {}

MeshCoordinate::MeshCoordinate(tt::stl::Span<const uint32_t> coords) : value_(coords.begin(), coords.end()) {}

MeshCoordinate MeshCoordinate::zero_coordinate(size_t dimensions) {
    return MeshCoordinate(tt::stl::SmallVector<uint32_t>(dimensions, 0));
}

size_t MeshCoordinate::dims() const { return value_.size(); }
tt::stl::Span<const uint32_t> MeshCoordinate::coords() const { return value_; }
uint32_t MeshCoordinate::operator[](int32_t dim) const { return value_[normalize_index(dim, dims())]; }
uint32_t& MeshCoordinate::operator[](int32_t dim) { return value_[normalize_index(dim, dims())]; }

size_t MeshCoordinate::to_linear_index(const MeshShape& shape) const {
    TT_FATAL(shape.dims() == dims(), "Shape and coordinate dimensions do not match: {} != {}", shape.dims(), dims());

    size_t linear_index = 0;
    for (size_t dim = 0; dim < dims(); ++dim) {
        TT_FATAL(value_[dim] < shape[dim], "Coordinate {} is out of bounds for shape {}", *this, shape);
        linear_index += value_[dim] * shape.get_stride(dim);
    }
    return linear_index;
}

std::optional<MeshCoordinate> MeshCoordinate::get_neighbor(
    const MeshShape& shape, int32_t offset, int32_t dim, BoundaryMode mode) const {
    TT_FATAL(shape.dims() == dims(), "Shape and coordinate dimensions do not match: {} != {}", shape.dims(), dims());
    for (size_t i = 0; i < dims(); ++i) {
        TT_FATAL(value_[i] < shape[i], "Coordinate {} is out of bounds for shape {}", *this, shape);
    }

    dim = normalize_index(dim, shape.dims());

    const auto boundary = static_cast<int32_t>(shape[dim]);
    const int32_t current_pos = static_cast<int32_t>(value_[dim]);
    const int32_t new_pos = current_pos + offset;

    MeshCoordinate result = *this;

    switch (mode) {
        case BoundaryMode::WRAP: result[dim] = ((new_pos % boundary) + boundary) % boundary; break;
        case BoundaryMode::CLAMP: result[dim] = std::clamp(new_pos, 0, boundary - 1); break;
        case BoundaryMode::NONE:
            if (new_pos < 0 || new_pos >= boundary) {
                return std::nullopt;
            }
            result[dim] = new_pos;
            break;
    }

    return result;
}

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

MeshCoordinateRange::MeshCoordinateRange(
    const MeshCoordinate& start, const MeshCoordinate& end, const MeshShape& wraparound_shape) :
    start_(start), end_(end), wraparound_shape_(wraparound_shape) {
    TT_FATAL(
        start.dims() == end.dims(),
        "Start and end dimensions of a coordinate range do not match: {} != {}",
        start.dims(),
        end.dims());
    TT_FATAL(
        wraparound_shape_.value().dims() == start.dims(),
        "Wraparound shape dims must match range dims: {} != {}",
        wraparound_shape_.value().dims(),
        start.dims());
    // No invariant on start <= end when wraparound is provided; iterator will handle wrapping.
}

MeshCoordinateRange::MeshCoordinateRange(const MeshShape& shape) :
    MeshCoordinateRange(MeshCoordinate::zero_coordinate(shape.dims()), shape_back(shape)) {}

MeshCoordinateRange::MeshCoordinateRange(const MeshCoordinate& coord) : start_(coord), end_(coord) {}

size_t MeshCoordinateRange::dims() const { return start_.dims(); }
const MeshCoordinate& MeshCoordinateRange::start_coord() const { return start_; }
const MeshCoordinate& MeshCoordinateRange::end_coord() const { return end_; }

MeshCoordinate::BoundaryMode MeshCoordinateRange::get_boundary_mode() const {
    if (wraparound_shape_.has_value()) {
        return MeshCoordinate::BoundaryMode::WRAP;
    }
    return MeshCoordinate::BoundaryMode::NONE;
}

MeshShape MeshCoordinateRange::shape() const {
    tt::stl::SmallVector<uint32_t> shape_dims;
    if (!wraparound_shape_.has_value()) {
        for (size_t i = 0; i < dims(); ++i) {
            shape_dims.push_back(end_[i] - start_[i] + 1);
        }
        return MeshShape(shape_dims);
    }
    // Wraparound: each dimension length is circular span size between start and end inclusive.
    for (size_t i = 0; i < dims(); ++i) {
        const uint32_t S = wraparound_shape_.value()[i];
        const uint32_t s = start_[i];
        const uint32_t e = end_[i];
        if (s <= e) {
            shape_dims.push_back(e - s + 1);
        } else {
            // wraps: [s..S-1] U [0..e]
            shape_dims.push_back((S - s) + (e + 1));
        }
    }
    return MeshShape(shape_dims);
}

bool MeshCoordinateRange::contains(const MeshCoordinate& coord) const {
    TT_FATAL(coord.dims() == dims(), "Coordinate dimensions do not match: {} != {}", coord.dims(), dims());
    if (!wraparound_shape_.has_value()) {
        for (int i = 0; i < coord.dims(); ++i) {
            if (coord[i] < start_[i] || coord[i] > end_[i]) {
                return false;
            }
        }
        return true;
    }
    for (int i = 0; i < coord.dims(); ++i) {
        const uint32_t s = start_[i];
        const uint32_t e = end_[i];
        const uint32_t c = coord[i];
        if (s <= e) {
            if (c < s || c > e) {
                return false;
            }
        } else {
            // wraps: c in [s..S-1] or [0..e]
            if (!(c >= s || c <= e)) {
                return false;
            }
        }
    }
    return true;
}

bool MeshCoordinateRange::contains(const MeshCoordinateRange& range) const {
    return contains(range.start_coord()) && contains(range.end_coord());
}

bool MeshCoordinateRange::intersects(const MeshCoordinateRange& range) const {
    TT_FATAL(range.dims() == dims(), "Coordinate dimensions do not match: {} != {}", range.dims(), dims());
    // Fallback: iterate over the smaller range and test containment to cover wrap semantics.
    const MeshCoordinateRange& smaller = (range.shape().mesh_size() <= this->shape().mesh_size()) ? range : *this;
    const MeshCoordinateRange& larger = (&smaller == this) ? range : *this;
    for (const auto& c : smaller) {
        if (larger.contains(c)) {
            return true;
        }
    }
    return false;
}

std::optional<MeshCoordinateRange> MeshCoordinateRange::intersection(const MeshCoordinateRange& range) const {
    if (!intersects(range)) {
        return std::nullopt;
    }
    // Conservative implementation: gather all coords from smaller range that are contained in the larger, then
    // build a minimal bounding range w.r.t the larger's wraparound settings if present.
    const MeshCoordinateRange& smaller = (range.shape().mesh_size() <= this->shape().mesh_size()) ? range : *this;
    const MeshCoordinateRange& larger = (&smaller == this) ? range : *this;
    bool first = true;
    std::optional<MeshCoordinate> minc;
    std::optional<MeshCoordinate> maxc;
    for (const auto& c : smaller) {
        if (larger.contains(c)) {
            if (first) {
                minc = c;
                maxc = c;
                first = false;
            } else {
                tt::stl::SmallVector<uint32_t> mins;
                tt::stl::SmallVector<uint32_t> maxs;
                mins.reserve(c.dims());
                maxs.reserve(c.dims());
                for (size_t i = 0; i < c.dims(); ++i) {
                    mins.push_back(std::min((*minc)[i], c[i]));
                    maxs.push_back(std::max((*maxc)[i], c[i]));
                }
                minc = MeshCoordinate(mins);
                maxc = MeshCoordinate(maxs);
            }
        }
    }
    if (first) {
        return std::nullopt;
    }
    if (larger.wraparound_shape_.has_value()) {
        return MeshCoordinateRange(*minc, *maxc, larger.wraparound_shape_.value());
    }
    return MeshCoordinateRange(*minc, *maxc);
}

MeshCoordinateRange::Iterator::Iterator(
    const MeshCoordinateRange* range, const MeshCoordinate& current, size_t linear_index) :
    range_(range), current_coord_(current), linear_index_(linear_index) {
    if (range_ && range_->wraparound_shape_.has_value()) {
        const auto dims = current_coord_.dims();
        lengths_.resize(dims);
        local_pos_.resize(dims);
        for (size_t i = 0; i < dims; ++i) {
            const uint32_t S = range_->wraparound_shape_.value()[i];
            const uint32_t s = range_->start_coord()[i];
            const uint32_t e = range_->end_coord()[i];
            lengths_[i] = (s <= e) ? (e - s + 1) : ((S - s) + (e + 1));
            local_pos_[i] = 0;
        }
        // Ensure current_coord_ corresponds to start_ for iteration start
        current_coord_ = range_->start_coord();
    }
}

MeshCoordinateRange::Iterator MeshCoordinateRange::Iterator::operator++(int) {
    Iterator tmp = *this;
    ++(*this);
    return tmp;
}

MeshCoordinateRange::Iterator& MeshCoordinateRange::Iterator::operator++() {
    ++linear_index_;
    if (!range_->wraparound_shape_.has_value()) {
        for (int i = current_coord_.dims() - 1; i >= 0; --i) {
            if (++current_coord_[i] > range_->end_coord()[i]) {
                current_coord_[i] = range_->start_coord()[i];
            } else {
                break;
            }
        }
        return *this;
    }
    // Wrapped iteration: advance local positions row-major over lengths_, then map to coords via start and shape.
    const size_t dims = current_coord_.dims();
    for (int i = dims - 1; i >= 0; --i) {
        local_pos_[i]++;
        if (local_pos_[i] >= lengths_[i]) {
            local_pos_[i] = 0;
        } else {
            break;
        }
    }
    for (size_t i = 0; i < dims; ++i) {
        const uint32_t S = range_->wraparound_shape_.value()[i];
        const uint32_t s = range_->start_coord()[i];
        current_coord_[i] = (s + local_pos_[i]) % S;
    }
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
    if (!wraparound_shape_.has_value()) {
        for (size_t i = 0; i < start_.dims(); ++i) {
            range_size *= end_[i] - start_[i] + 1;
        }
    } else {
        // For wraparound, size is product of circular span lengths per dimension.
        for (size_t i = 0; i < start_.dims(); ++i) {
            const uint32_t S = wraparound_shape_.value()[i];
            const uint32_t s = start_[i];
            const uint32_t e = end_[i];
            const uint32_t len = (s <= e) ? (e - s + 1) : ((S - s) + (e + 1));
            range_size *= len;
        }
    }
    // Set `start_` coordinate but `range_size` linear index as the wrap around condition.
    return Iterator(this, start_, range_size);
}

bool operator==(const MeshCoordinateRange& lhs, const MeshCoordinateRange& rhs) {
    return lhs.start_coord() == rhs.start_coord() && lhs.end_coord() == rhs.end_coord();
}
bool operator!=(const MeshCoordinateRange& lhs, const MeshCoordinateRange& rhs) { return !(lhs == rhs); }
bool operator<(const MeshCoordinateRange& lhs, const MeshCoordinateRange& rhs) {
    if (lhs.start_coord() != rhs.start_coord()) {
        return lhs.start_coord() < rhs.start_coord();
    }
    return lhs.end_coord() < rhs.end_coord();
}

std::ostream& operator<<(std::ostream& os, const MeshCoordinateRange& range) {
    os << "MeshCoordinateRange(start=" << range.start_coord() << ", end=" << range.end_coord() << ")";
    return os;
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
            }
            if (merged.intersects(*it) || it->intersects(merged)) {
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

    // Merge back the ranges that were removed.
    for (const auto& range : add_back) {
        merge(range);
    }

    // Sort the ranges to ensure deterministic order.
    std::sort(ranges_.begin(), ranges_.end(), [](const auto& a, const auto& b) {
        return (a.start_coord() != b.start_coord()) ? a.start_coord() < b.start_coord() : a.end_coord() < b.end_coord();
    });

    // Do one final check to see if all ranges can collapse into a single range.
    if (ranges_.size() > 1) {
        // Calculate the bounding box of all ranges
        tt::stl::SmallVector<uint32_t> bb_start;
        tt::stl::SmallVector<uint32_t> bb_end;

        for (size_t dim = 0; dim < ranges_[0].dims(); ++dim) {
            uint32_t min_start = ranges_[0].start_coord()[dim];
            uint32_t max_end = ranges_[0].end_coord()[dim];

            for (size_t i = 1; i < ranges_.size(); ++i) {
                min_start = std::min(min_start, ranges_[i].start_coord()[dim]);
                max_end = std::max(max_end, ranges_[i].end_coord()[dim]);
            }

            bb_start.push_back(min_start);
            bb_end.push_back(max_end);
        }

        // Calculate the total size of the bounding box
        uint64_t bounding_size = 1;
        for (size_t dim = 0; dim < ranges_[0].dims(); ++dim) {
            bounding_size *= (bb_end[dim] - bb_start[dim] + 1);
        }

        // Calculate the total size of all ranges combined
        uint64_t total_size = 0;
        for (const auto& range : ranges_) {
            total_size += range.shape().mesh_size();
        }

        // If the bounding box size equals the total size, all ranges fit perfectly
        // into a single rectangle with no gaps
        if (bounding_size == total_size) {
            MeshCoordinateRange full_range = MeshCoordinateRange(MeshCoordinate(bb_start), MeshCoordinate(bb_end));
            ranges_.clear();
            ranges_.push_back(full_range);
        }
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
    }  // Slow path: iterate over all coordinates in the parent range, and create ranges for the complement.
    for (const auto& coord : parent) {
        if (!intersection.contains(coord)) {
            complement_set.merge(MeshCoordinateRange(coord, coord));
        }
    }
    return complement_set;
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
