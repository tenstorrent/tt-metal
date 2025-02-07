// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <vector>

#include "shape_base.hpp"

namespace tt::tt_metal::distributed {

struct MeshShape;

// TODO: #17477 - Rename to `MeshShape` when the legacy type is gone.
class SimpleMeshShape : public ShapeBase {
public:
    using ShapeBase::ShapeBase;
    using ShapeBase::operator[];
    using ShapeBase::cbegin;
    using ShapeBase::cend;
    using ShapeBase::empty;
    using ShapeBase::size;
    using ShapeBase::view;

    // Shorthands for constructing 2D and 3D shapes.
    SimpleMeshShape(uint32_t num_rows, uint32_t num_cols);
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
    void compute_strides();
    tt::stl::SmallVector<size_t> strides_;
};

class MeshCoordinate {
public:
    // Shorthands for constructing 2D and 3D coordinates.
    MeshCoordinate(uint32_t row, uint32_t col);
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

    const MeshCoordinate& start_coord() const;
    const MeshCoordinate& end_coord() const;

    class Iterator {
    public:
        Iterator& operator++();
        MeshCoordinate operator*() const;
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

// Allows storing data in a mesh-shaped container, with convenient accessors and iterators.
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
        using value_type = std::pair<MeshCoordinate, T>;
        using reference = std::pair<MeshCoordinate, std::reference_wrapper<T>>;

        Iterator& operator++();
        reference operator*() const;
        bool operator==(const Iterator& other) const;
        bool operator!=(const Iterator& other) const;

    private:
        Iterator(MeshContainer* container, const MeshCoordinateRange::Iterator& coord_iter, size_t linear_index);
        friend class MeshContainer;

        MeshContainer* container_ = nullptr;
        MeshCoordinateRange::Iterator coord_iter_;
        size_t linear_index_ = 0;
    };

    Iterator begin();
    Iterator end();

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
    container_(container), coord_iter_(coord_iter), linear_index_(linear_index) {}

template <typename T>
typename MeshContainer<T>::Iterator& MeshContainer<T>::Iterator::operator++() {
    ++linear_index_;
    ++coord_iter_;
    return *this;
}

template <typename T>
typename MeshContainer<T>::Iterator::reference MeshContainer<T>::Iterator::operator*() const {
    return {*coord_iter_, std::ref(container_->values_[linear_index_])};
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
typename MeshContainer<T>::Iterator MeshContainer<T>::begin() {
    return Iterator(this, coord_range_.begin(), /* linear_index = */ 0);
}

template <typename T>
typename MeshContainer<T>::Iterator MeshContainer<T>::end() {
    return Iterator(this, coord_range_.end(), shape_.mesh_size());
}

}  // namespace tt::tt_metal::distributed
