// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <optional>
#include <type_traits>
#include <vector>

#include <tt_stl/reflection.hpp>
#include <tt_stl/assert.hpp>
#include <tt-metalium/shape_base.hpp>
#include <tt-metalium/maybe_remote.hpp>

namespace tt::tt_metal::distributed {

class MeshShape : public ShapeBase {
public:
    using ShapeBase::operator[];

    // Shorthands for constructing 1D, 2D and 3D shapes.
    // The values s0...sn are assumed to be supplied in row-major order (from outer dim to inner dim).
    explicit MeshShape(uint32_t s);
    MeshShape(uint32_t s0, uint32_t s1);
    MeshShape(uint32_t s0, uint32_t s1, uint32_t s2);

    explicit MeshShape(const tt::stl::SmallVector<uint32_t>& shape);
    explicit MeshShape(tt::stl::SmallVector<uint32_t>&& shape);
    explicit MeshShape(std::initializer_list<uint32_t> ilist);
    explicit MeshShape(tt::stl::Span<const uint32_t> span);

    // Returns the dimensionality of the mesh.
    size_t dims() const;

    // Returns the stride for the given dimension.
    size_t get_stride(size_t dim) const;

    // Returns the total number of elements in the mesh.
    size_t mesh_size() const;

    // Returns true if the mesh shape is in a line topology: at most 1 dimension can be non-unit.
    bool is_line_topology() const;

    // Needed for reflect / fmt
    static constexpr auto attribute_names = std::forward_as_tuple("value");
    auto attribute_values() const { return std::forward_as_tuple(value_); }

    friend bool operator==(const MeshShape& lhs, const MeshShape& rhs);
    friend bool operator!=(const MeshShape& lhs, const MeshShape& rhs);
    friend std::ostream& operator<<(std::ostream& os, const MeshShape& shape);

private:
    using ShapeBase::empty;
    using ShapeBase::ShapeBase;
    using ShapeBase::size;

    void compute_strides();
    tt::stl::SmallVector<size_t> strides_;
};

class MeshCoordinate {
public:
    // Shorthands for constructing 1D, 2D and 3D coordinates.
    // The values c0...cn are assumed to be supplied in row-major order (from outer dim to inner dim).
    explicit MeshCoordinate(uint32_t c);
    MeshCoordinate(uint32_t c0, uint32_t c1);
    MeshCoordinate(uint32_t c0, uint32_t c1, uint32_t c2);

    // Constructs a generic N-dimensional coordinate.
    explicit MeshCoordinate(tt::stl::Span<const uint32_t> coords);

    // Returns a zero-initialized N-dimensional coordinate.
    static MeshCoordinate zero_coordinate(size_t dimensions);

    // Returns the dimensionality of the coordinate.
    size_t dims() const;

    // Returns the coordinate values as a span.
    tt::stl::Span<const uint32_t> coords() const;

    // Provides access to the coordinate value at the given index.
    // Supports negative indexing.
    uint32_t operator[](int32_t dim) const;
    uint32_t& operator[](int32_t dim);

    // Converts a MeshCoordinate to a linear index.
    // Throws if `coord` is out of bounds of `shape`.
    size_t to_linear_index(const MeshShape& shape) const;

    // Returns a neighbor along the given dimension.
    // `BoundaryMode` specifies how to handle coordinates that are out of bounds.
    // Negative offsets and dim are allowed, and the input coordinate must be within bounds.
    enum class BoundaryMode { WRAP, CLAMP, NONE };

    std::optional<MeshCoordinate> get_neighbor(
        const MeshShape& shape, int32_t offset, int32_t dim, BoundaryMode mode = BoundaryMode::WRAP) const;

    // Needed for reflect / fmt
    static constexpr auto attribute_names = std::forward_as_tuple("value");
    auto attribute_values() const { return std::forward_as_tuple(value_); }

private:
    tt::stl::SmallVector<uint32_t> value_;
};

bool operator==(const MeshCoordinate& lhs, const MeshCoordinate& rhs);
bool operator!=(const MeshCoordinate& lhs, const MeshCoordinate& rhs);

// Compares two coordinates in lexicographical order.
bool operator<(const MeshCoordinate& lhs, const MeshCoordinate& rhs);
bool operator>(const MeshCoordinate& lhs, const MeshCoordinate& rhs);
bool operator<=(const MeshCoordinate& lhs, const MeshCoordinate& rhs);
bool operator>=(const MeshCoordinate& lhs, const MeshCoordinate& rhs);

std::ostream& operator<<(std::ostream& os, const MeshCoordinate& coord);

// Represents a range of MeshCoordinates. Requires that mesh coordinates have the same dimensionality.
class MeshCoordinateRange {
public:
    // Constructs an inclusive range that iterates between `start` and `end`.
    MeshCoordinateRange(const MeshCoordinate& start, const MeshCoordinate& end);

    // Constructs an inclusive range that iterates between `start` and `end`,
    // interpreting ranges with wraparound semantics based on the provided shape.
    // When wraparound is enabled, a dimension where start > end is treated as
    // wrapping from start..(shape[dim]-1) and then 0..end.
    MeshCoordinateRange(const MeshCoordinate& start, const MeshCoordinate& end, const MeshShape& wraparound_shape);

    // Constructs a range that iterates over all coordinates in the mesh.
    explicit MeshCoordinateRange(const MeshShape& shape);

    // Constructs a range that includes a single coordinate.
    explicit MeshCoordinateRange(const MeshCoordinate& coord);

    // Returns the dimensionality of the range.
    size_t dims() const;

    // Returns start and (inclusive) end coordinates of the range.
    const MeshCoordinate& start_coord() const;
    const MeshCoordinate& end_coord() const;

    // Returns the shape of the coordinate range (dimensions).
    MeshShape shape() const;

    // Returns the boundary mode of the range.
    MeshCoordinate::BoundaryMode get_boundary_mode() const;

    // Returns the wraparound shape if enabled.
    const std::optional<MeshShape>& wraparound_shape() const { return wraparound_shape_; }

    // Returns true if the range contains the given coordinate.
    bool contains(const MeshCoordinate& coord) const;

    // Returns true if the range contains the given range.
    bool contains(const MeshCoordinateRange& range) const;

    // Returns true if the range intersects with the given range.
    bool intersects(const MeshCoordinateRange& range) const;

    // Returns the intersection of the range with the given range.
    std::optional<MeshCoordinateRange> intersection(const MeshCoordinateRange& range) const;

    // Needed for reflect / fmt
    static constexpr auto attribute_names = std::forward_as_tuple("start", "end");
    auto attribute_values() const { return std::forward_as_tuple(start_, end_); }

    // Iterator over the range, provides access to coordinates in row-major order.
    class Iterator {
    public:
        Iterator& operator++();
        Iterator operator++(int);
        const MeshCoordinate& operator*() const;
        bool operator==(const Iterator& other) const;
        bool operator!=(const Iterator& other) const;

    private:
        Iterator(const MeshCoordinateRange* range, const MeshCoordinate& current_coord, size_t linear_index);
        friend class MeshCoordinateRange;

        const MeshCoordinateRange* range_ = nullptr;

        // For simplicity, rely on `linear_index_` for the iterator boundary check, and allow
        // MeshCoordinate to wrap around the range end.
        MeshCoordinate current_coord_;
        size_t linear_index_ = 0;

        // Local iteration state for wraparound ranges: per-dimension lengths and positions.
        // When wraparound is active and start > end in a dimension, we iterate over a circular span
        // of length: (shape[dim] - start) + (end + 1), mapping local positions to actual coords via modulo.
        std::vector<uint32_t> lengths_;
        std::vector<uint32_t> local_pos_;
    };

    Iterator begin() const;
    Iterator end() const;

private:
    MeshCoordinate start_;
    MeshCoordinate end_;
    // If present, enables wraparound semantics with these per-dimension sizes.
    std::optional<MeshShape> wraparound_shape_;
};

bool operator==(const MeshCoordinateRange& lhs, const MeshCoordinateRange& rhs);
bool operator!=(const MeshCoordinateRange& lhs, const MeshCoordinateRange& rhs);
bool operator<(const MeshCoordinateRange& lhs, const MeshCoordinateRange& rhs);
std::ostream& operator<<(std::ostream& os, const MeshCoordinateRange& range);

// Represents a set of non-overlapping MeshCoordinateRanges.
// `MeshCoordinateRangeSet` performs a best-effort merge of ranges, but does not guarantee that the set is minimal.
class MeshCoordinateRangeSet {
public:
    MeshCoordinateRangeSet() = default;

    // Constructs a set with a single range.
    explicit MeshCoordinateRangeSet(const MeshCoordinateRange&);

    // Merges the given range into the set.
    void merge(const MeshCoordinateRange& to_merge);

    // Returns the number of ranges in the set.
    size_t size() const { return ranges_.size(); }

    // Returns true if the set is empty.
    bool empty() const { return ranges_.empty(); }

    // Returns all ranges in the set, sorted in lexicographical order.
    const auto& ranges() const { return ranges_; }

    // Flattens ranges and returns all coordinates that this set covers, sorted in lexicographical order.
    std::vector<MeshCoordinate> coords() const;

    // Needed for reflect / fmt
    static constexpr auto attribute_names = std::forward_as_tuple("ranges");
    auto attribute_values() const { return std::forward_as_tuple(ranges_); }

private:
    std::vector<MeshCoordinateRange> ranges_;
};

bool operator==(const MeshCoordinateRangeSet& lhs, const MeshCoordinateRangeSet& rhs);
bool operator!=(const MeshCoordinateRangeSet& lhs, const MeshCoordinateRangeSet& rhs);
std::ostream& operator<<(std::ostream& os, const MeshCoordinateRangeSet& range_set);

// Returns the set of ranges that result from subtracting the intersection from the parent range.
MeshCoordinateRangeSet subtract(const MeshCoordinateRange& parent, const MeshCoordinateRange& intersection);

namespace detail {

// Proxy class that allows convenient structured binding to a pair of a coordinate and the value it points to.
// This supports iterator semantics similar to `std::map` / `std::unordered_map`.
template <typename T>
class MeshCoordinateValueProxy {
public:
    MeshCoordinateValueProxy(const MeshCoordinate& coord, T* value_ptr) : coord_(coord), value_ptr_(value_ptr) {}

    const MeshCoordinate& coord() const { return coord_; }
    T& value() { return *value_ptr_; }
    const T& value() const { return *value_ptr_; }

    template <std::size_t I>
    decltype(auto) get() & {
        if constexpr (I == 0) {
            return coord();
        } else if constexpr (I == 1) {
            return value();
        } else {
            static_assert(I < 2);
        }
    }

    template <std::size_t I>
    decltype(auto) get() const& {
        if constexpr (I == 0) {
            return coord();
        } else if constexpr (I == 1) {
            return value();
        } else {
            static_assert(I < 2);
        }
    }

    // Force a copy via `auto`.
    template <std::size_t I>
    auto get() const&& {
        return get<I>();
    }

private:
    MeshCoordinate coord_;
    T* value_ptr_ = nullptr;
};

}  // namespace detail

// Allows storing data in a mesh-shaped flat container, with convenient accessors and iterators.
// The iteration order and the storage memory layout is row-major.
template <typename T>
class MeshContainer {
public:
    MeshContainer(const MeshShape& shape, const T& fill_value);
    MeshContainer(const MeshShape& shape, std::vector<T> values);

    // Returns a shape of the container.
    const MeshShape& shape() const;

    // Returns (inclusive) range of coordinates in the container.
    const MeshCoordinateRange& coord_range() const;

    // Returns the number of elements in the container.
    size_t size() const;

    // Accessor methods.
    T& at(const MeshCoordinate& coord);
    const T& at(const MeshCoordinate& coord) const;

    // Allows to iterate over the container elements, returning a pair of (coordinate, value reference).
    // Note: End iterators have undefined coordinates and should not be dereferenced, following C++ iterator conventions.
    class Iterator {
    public:
        using ValueProxy = detail::MeshCoordinateValueProxy<T>;
        using iterator_category = std::forward_iterator_tag;
        using value_type = ValueProxy;
        using difference_type = std::ptrdiff_t;
        using pointer = ValueProxy*;
        using reference = ValueProxy&;

        Iterator& operator++();
        ValueProxy& operator*() { return value_proxy_; }
        const ValueProxy& operator*() const { return value_proxy_; }
        ValueProxy* operator->() { return &value_proxy_; }
        const ValueProxy* operator->() const { return &value_proxy_; }
        bool operator==(const Iterator& other) const;
        bool operator!=(const Iterator& other) const;

    private:
        Iterator(MeshContainer* container, const MeshCoordinateRange::Iterator& coord_iter, size_t linear_index);
        friend class MeshContainer;

        MeshContainer* container_ = nullptr;
        MeshCoordinateRange::Iterator coord_iter_;
        size_t linear_index_ = 0;

        // Provides mutable access to the container value along with the coordinate from the range iterator.
        ValueProxy value_proxy_;
    };

    class ConstIterator {
    public:
        using ValueProxy = detail::MeshCoordinateValueProxy<const T>;
        using iterator_category = std::forward_iterator_tag;
        using value_type = ValueProxy;
        using difference_type = std::ptrdiff_t;
        using pointer = const ValueProxy*;
        using reference = const ValueProxy&;

        ConstIterator& operator++();
        const ValueProxy& operator*() const { return value_proxy_; }
        const ValueProxy* operator->() const { return &value_proxy_; }
        bool operator==(const ConstIterator& other) const;
        bool operator!=(const ConstIterator& other) const;

    private:
        ConstIterator(
            const MeshContainer* container, const MeshCoordinateRange::Iterator& coord_iter, size_t linear_index);
        friend class MeshContainer;

        const MeshContainer* container_ = nullptr;
        MeshCoordinateRange::Iterator coord_iter_;
        size_t linear_index_ = 0;

        // Provides mutable access to the container value along with the coordinate from the range iterator.
        ValueProxy value_proxy_;
    };

    // Iterators provide a reference to the value along with the coordinate.
    Iterator begin();
    Iterator end();
    ConstIterator begin() const;
    ConstIterator end() const;

    // View of the flat container of values.
    std::vector<T>& values() { return values_; }
    const std::vector<T>& values() const { return values_; }

    friend bool operator==(const MeshContainer& lhs, const MeshContainer& rhs) {
        return lhs.shape() == rhs.shape() && lhs.coord_range() == rhs.coord_range() && lhs.values() == rhs.values();
    }
    friend bool operator!=(const MeshContainer& lhs, const MeshContainer& rhs) { return !(lhs == rhs); }

private:
    MeshShape shape_;
    MeshCoordinateRange coord_range_;
    std::vector<T> values_;
};

/**
 * A specialized MeshContainer where some values may be locally present and some are remote.
 *
 * This container simplifies the creation and management of distributed mesh structures where some values may be remote
 * (on other hosts) and some are local. The values are wrapped in MaybeRemote<T> to allow for easy distinction between
 * local and remote values.
 *
 * @tparam T The type of values stored (will be wrapped in MaybeRemote<T>)
 */
template <typename T>
class DistributedMeshContainer : public MeshContainer<MaybeRemote<T>> {
public:
    /**
     * Initialize a distributed mesh container with all remote values.
     *
     * @param global_shape The global shape of the mesh
     */
    explicit DistributedMeshContainer(const MeshShape& global_shape) :
        MeshContainer<MaybeRemote<T>>(global_shape, MaybeRemote<T>::remote()) {}

    /**
     * Initialize a distributed mesh container with a vector of values.
     *
     * @param global_shape The global shape of the mesh
     * @param values The values to populate in the container
     */
    explicit DistributedMeshContainer(const MeshShape& global_shape, std::vector<MaybeRemote<T>> values) :
        MeshContainer<MaybeRemote<T>>(global_shape, std::move(values)) {}

    /**
     * Check if a global coordinate contains a local value.
     *
     * @param coord The global coordinate to check
     * @return true if the coordinate contains a local value, false if remote
     */
    bool is_local(const MeshCoordinate& coord) const { return this->at(coord).is_local(); }
};

template <typename T>
size_t MeshContainer<T>::size() const {
    return values_.size();
}

template <typename T>
MeshContainer<T>::MeshContainer(const MeshShape& shape, const T& fill_value) :
    shape_(shape), coord_range_(shape), values_(shape.mesh_size(), fill_value) {}

template <typename T>
MeshContainer<T>::MeshContainer(const MeshShape& shape, std::vector<T> values) :
    shape_(shape), coord_range_(shape), values_(std::move(values)) {
    TT_FATAL(
        shape.mesh_size() == values_.size(),
        "Shape and values size mismatch; shape: {}, values: {}",
        shape,
        values_.size());
}

template <typename T>
const MeshShape& MeshContainer<T>::shape() const {
    return shape_;
}

template <typename T>
const MeshCoordinateRange& MeshContainer<T>::coord_range() const {
    return coord_range_;
}

template <typename T>
T& MeshContainer<T>::at(const MeshCoordinate& coord) {
    return values_.at(coord.to_linear_index(shape_));
}

template <typename T>
const T& MeshContainer<T>::at(const MeshCoordinate& coord) const {
    return values_.at(coord.to_linear_index(shape_));
}

template <typename T>
MeshContainer<T>::Iterator::Iterator(
    MeshContainer* container, const MeshCoordinateRange::Iterator& coord_iter, size_t linear_index) :
    container_(container),
    coord_iter_(coord_iter),
    linear_index_(linear_index),
    value_proxy_(
        coord_iter_ != container_->coord_range_.end() ? *coord_iter_ : MeshCoordinate::zero_coordinate(container->shape().dims()),
        linear_index_ < container_->values_.size() ? &container_->values_[linear_index_] : nullptr) {}

template <typename T>
typename MeshContainer<T>::Iterator& MeshContainer<T>::Iterator::operator++() {
    ++linear_index_;
    ++coord_iter_;
    value_proxy_ = ValueProxy(
        coord_iter_ != container_->coord_range_.end() ? *coord_iter_ : MeshCoordinate::zero_coordinate(container_->shape().dims()),
        linear_index_ < container_->values_.size() ? &container_->values_[linear_index_] : nullptr);
    return *this;
}

template <typename T>
MeshContainer<T>::ConstIterator::ConstIterator(
    const MeshContainer* container, const MeshCoordinateRange::Iterator& coord_iter, size_t linear_index) :
    container_(container),
    coord_iter_(coord_iter),
    linear_index_(linear_index),
    value_proxy_(
        coord_iter_ != container_->coord_range_.end() ? *coord_iter_ : MeshCoordinate::zero_coordinate(container->shape().dims()),
        linear_index_ < container_->values_.size() ? &container_->values_[linear_index_] : nullptr) {}

template <typename T>
typename MeshContainer<T>::ConstIterator& MeshContainer<T>::ConstIterator::operator++() {
    ++linear_index_;
    ++coord_iter_;
    value_proxy_ = ValueProxy(
        coord_iter_ != container_->coord_range_.end() ? *coord_iter_ : MeshCoordinate::zero_coordinate(container_->shape().dims()),
        linear_index_ < container_->values_.size() ? &container_->values_[linear_index_] : nullptr);
    return *this;
}

template <typename T>
bool MeshContainer<T>::Iterator::operator==(const Iterator& other) const {
    return container_ == other.container_ && coord_iter_ == other.coord_iter_ && linear_index_ == other.linear_index_;
}

template <typename T>
bool MeshContainer<T>::Iterator::operator!=(const Iterator& other) const {
    return !(*this == other);
}

template <typename T>
bool MeshContainer<T>::ConstIterator::operator==(const ConstIterator& other) const {
    return container_ == other.container_ && coord_iter_ == other.coord_iter_ && linear_index_ == other.linear_index_;
}

template <typename T>
bool MeshContainer<T>::ConstIterator::operator!=(const ConstIterator& other) const {
    return !(*this == other);
}

template <typename T>
typename MeshContainer<T>::Iterator MeshContainer<T>::begin() {
    return Iterator(this, coord_range_.begin(), /* linear_index = */ 0);
}

template <typename T>
typename MeshContainer<T>::Iterator MeshContainer<T>::end() {
    return Iterator(this, coord_range_.end(), shape_.mesh_size());
}

template <typename T>
typename MeshContainer<T>::ConstIterator MeshContainer<T>::begin() const {
    return ConstIterator(this, coord_range_.begin(), /* linear_index = */ 0);
}

template <typename T>
typename MeshContainer<T>::ConstIterator MeshContainer<T>::end() const {
    return ConstIterator(this, coord_range_.end(), shape_.mesh_size());
}

}  // namespace tt::tt_metal::distributed

namespace std {

template <typename T>
struct tuple_size<tt::tt_metal::distributed::detail::MeshCoordinateValueProxy<T>> : std::integral_constant<size_t, 2> {
};

template <typename T>
struct tuple_element<0, tt::tt_metal::distributed::detail::MeshCoordinateValueProxy<T>> {
    using type = const tt::tt_metal::distributed::MeshCoordinate;
};

template <typename T>
struct tuple_element<1, tt::tt_metal::distributed::detail::MeshCoordinateValueProxy<T>> {
    using type = T;
};

template <>
struct hash<tt::tt_metal::distributed::MeshCoordinate> {
    size_t operator()(const tt::tt_metal::distributed::MeshCoordinate& coord) const noexcept {
        return tt::stl::hash::hash_objects_with_default_seed(coord.attribute_values());
    }
};

template <>
struct hash<tt::tt_metal::distributed::MeshCoordinateRange> {
    size_t operator()(const tt::tt_metal::distributed::MeshCoordinateRange& range) const noexcept {
        return tt::stl::hash::hash_objects_with_default_seed(range.attribute_values());
    }
};

template <>
struct hash<tt::tt_metal::distributed::MeshCoordinateRangeSet> {
    size_t operator()(const tt::tt_metal::distributed::MeshCoordinateRangeSet& range_set) const noexcept {
        return tt::stl::hash::hash_objects_with_default_seed(range_set.attribute_values());
    }
};

}  // namespace std
