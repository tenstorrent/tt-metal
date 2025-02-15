// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <type_traits>
#include <vector>

#include "shape_base.hpp"

namespace tt::tt_metal::distributed {

struct MeshShape;

// TODO: #17477 - Rename to `MeshShape` when the legacy type is gone.
class SimpleMeshShape : public ShapeBase {
public:
    using ShapeBase::ShapeBase;
    using ShapeBase::operator[];

    // Shorthands for constructing 1D, 2D and 3D shapes.
    SimpleMeshShape(uint32_t x);
    SimpleMeshShape(uint32_t x, uint32_t y);
    SimpleMeshShape(uint32_t x, uint32_t y, uint32_t z);

    // Temporary constructor for transitioning to `SimpleMeshShape`.
    SimpleMeshShape(const MeshShape& legacy_shape);

    // Returns the dimensionality of the mesh.
    size_t dims() const;

    // Returns the stride for the given dimension.
    size_t get_stride(size_t dim) const;

    // Returns the total number of elements in the mesh.
    size_t mesh_size() const;

    // Needed for reflect / fmt
    static constexpr auto attribute_names = std::forward_as_tuple("value");
    auto attribute_values() const { return std::forward_as_tuple(value_); }

    friend bool operator==(const SimpleMeshShape& lhs, const SimpleMeshShape& rhs);
    friend bool operator!=(const SimpleMeshShape& lhs, const SimpleMeshShape& rhs);
    friend std::ostream& operator<<(std::ostream& os, const SimpleMeshShape& shape);

private:
    using ShapeBase::empty;
    using ShapeBase::size;

    void compute_strides();
    tt::stl::SmallVector<size_t> strides_;
};

class MeshCoordinate {
public:
    // Shorthands for constructing 1D, 2D and 3D coordinates.
    MeshCoordinate(uint32_t x);
    MeshCoordinate(uint32_t x, uint32_t y);
    MeshCoordinate(uint32_t x, uint32_t y, uint32_t z);

    // Constructs a generic N-dimensional coordinate.
    explicit MeshCoordinate(tt::stl::Span<const uint32_t> coords);

    // Returns the dimensionality of the coordinate.
    size_t dims() const;

    // Returns the coordinate values as a span.
    tt::stl::Span<const uint32_t> coords() const;

    // Returns the coordinate value at the given index.
    uint32_t operator[](size_t dim) const;

    // Needed for reflect / fmt
    static constexpr auto attribute_names = std::forward_as_tuple("value");
    auto attribute_values() const { return std::forward_as_tuple(value_); }

    friend bool operator==(const MeshCoordinate& lhs, const MeshCoordinate& rhs);
    friend bool operator!=(const MeshCoordinate& lhs, const MeshCoordinate& rhs);
    friend std::ostream& operator<<(std::ostream& os, const MeshCoordinate& shape);

private:
    tt::stl::SmallVector<uint32_t> value_;
};

// Converts a MeshCoordinate to a linear index.
// Throws if `coord` is out of bounds of `shape`.
size_t to_linear_index(const SimpleMeshShape& shape, const MeshCoordinate& coord);

// Represents a range of MeshCoordinates. Requires that mesh coordinates have the same dimensionality.
class MeshCoordinateRange {
public:
    // Constructs an inclusive range that iterates between `start` and `end`.
    MeshCoordinateRange(const MeshCoordinate& start, const MeshCoordinate& end);

    // Constructs a range that iterates over all coordinates in the mesh.
    MeshCoordinateRange(const SimpleMeshShape& shape);

    // Returns start and (inclusive) end coordinates of the range.
    const MeshCoordinate& start_coord() const;
    const MeshCoordinate& end_coord() const;

    class Iterator {
    public:
        Iterator& operator++();
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
    };

    Iterator begin() const;
    Iterator end() const;

    friend bool operator==(const MeshCoordinateRange& lhs, const MeshCoordinateRange& rhs);
    friend bool operator!=(const MeshCoordinateRange& lhs, const MeshCoordinateRange& rhs);

private:
    MeshCoordinate start_;
    MeshCoordinate end_;
};

namespace detail {

// Proxy class that allows convenient structured binding to a pair of a coordinate and the value it points to.
// This supports iterator semantics similar to `std::map` / `std::unordered_map`.
template <typename T>
class MeshCoordinateValueProxy {
public:
    MeshCoordinateValueProxy(const MeshCoordinate* coord, T* value_ptr) : coord_(coord), value_ptr_(value_ptr) {}

    const MeshCoordinate& coord() const { return *coord_; }
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
    const MeshCoordinate* coord_ = nullptr;
    T* value_ptr_ = nullptr;
};

}  // namespace detail

// Allows storing data in a mesh-shaped flat container, with convenient accessors and iterators.
// The iteration order and the storage memory layout is row-major.
template <typename T>
class MeshContainer {
public:
    MeshContainer(const SimpleMeshShape& shape, const T& fill_value);

    // Returns a shape of the container.
    const SimpleMeshShape& shape() const;

    // Accessor methods.
    T& at(const MeshCoordinate& coord);
    const T& at(const MeshCoordinate& coord) const;

    // Allows to iterate over the container elements, returning a pair of (coordinate, value reference).
    class Iterator {
    public:
        using ValueProxy = detail::MeshCoordinateValueProxy<T>;

        Iterator& operator++();
        ValueProxy& operator*();
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

        ConstIterator& operator++();
        const ValueProxy& operator*() const;
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

    Iterator begin();
    Iterator end();
    ConstIterator begin() const;
    ConstIterator end() const;

private:
    SimpleMeshShape shape_;
    MeshCoordinateRange coord_range_;
    std::vector<T> values_;
};

template <typename T>
MeshContainer<T>::MeshContainer(const SimpleMeshShape& shape, const T& fill_value) :
    shape_(shape), coord_range_(shape), values_(shape.mesh_size(), fill_value) {}

template <typename T>
const SimpleMeshShape& MeshContainer<T>::shape() const {
    return shape_;
}

template <typename T>
T& MeshContainer<T>::at(const MeshCoordinate& coord) {
    return values_.at(to_linear_index(shape_, coord));
}

template <typename T>
const T& MeshContainer<T>::at(const MeshCoordinate& coord) const {
    return values_.at(to_linear_index(shape_, coord));
}

template <typename T>
MeshContainer<T>::Iterator::Iterator(
    MeshContainer* container, const MeshCoordinateRange::Iterator& coord_iter, size_t linear_index) :
    container_(container),
    coord_iter_(coord_iter),
    linear_index_(linear_index),
    value_proxy_(&(*coord_iter_), &container_->values_[linear_index_]) {}

template <typename T>
typename MeshContainer<T>::Iterator& MeshContainer<T>::Iterator::operator++() {
    ++linear_index_;
    ++coord_iter_;
    value_proxy_ = ValueProxy(&(*coord_iter_), &container_->values_[linear_index_]);
    return *this;
}

template <typename T>
typename MeshContainer<T>::Iterator::ValueProxy& MeshContainer<T>::Iterator::operator*() {
    return value_proxy_;
}

template <typename T>
MeshContainer<T>::ConstIterator::ConstIterator(
    const MeshContainer* container, const MeshCoordinateRange::Iterator& coord_iter, size_t linear_index) :
    container_(container),
    coord_iter_(coord_iter),
    linear_index_(linear_index),
    value_proxy_(&(*coord_iter_), &container_->values_[linear_index_]) {}

template <typename T>
typename MeshContainer<T>::ConstIterator& MeshContainer<T>::ConstIterator::operator++() {
    ++linear_index_;
    ++coord_iter_;
    value_proxy_ = ValueProxy(&(*coord_iter_), &container_->values_[linear_index_]);
    return *this;
}

template <typename T>
const typename MeshContainer<T>::ConstIterator::ValueProxy& MeshContainer<T>::ConstIterator::operator*() const {
    return value_proxy_;
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

}  // namespace std
