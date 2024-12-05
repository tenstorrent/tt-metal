// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <core/ttnn_all_includes.hpp>
#include <core/xtensor_all_includes.hpp>
#include <unordered_map>

#include "core/xtensor_utils.hpp"

namespace ttml::core {
template <typename T>
std::vector<xt::xarray<T>> chunk(const xt::xarray<T>& tensor, int num_chunks, int dim) {
    if (num_chunks <= 0) {
        throw std::invalid_argument("num_chunks must be > 0");
    }
    if (dim < 0 || static_cast<std::size_t>(dim) >= tensor.dimension()) {
        throw std::invalid_argument("invalid dimension index");
    }

    int size_along_dim = static_cast<int>(tensor.shape()[dim]);
    if (num_chunks > size_along_dim) {
        throw std::invalid_argument("num_chunks cannot exceed the size of the tensor along the given dimension.");
    }

    int chunk_size = size_along_dim / num_chunks;
    int remainder = size_along_dim % num_chunks;

    std::vector<xt::xarray<T>> chunks;
    chunks.reserve(static_cast<std::size_t>(num_chunks));

    int start = 0;
    for (int i = 0; i < num_chunks; ++i) {
        int current_chunk_size = chunk_size + ((i < remainder) ? 1 : 0);
        int end = start + current_chunk_size;

        // Build indices for slicing
        xt::xstrided_slice_vector indices(tensor.dimension(), xt::all());
        indices[dim] = xt::range(start, end);

        auto chunk_view = xt::strided_view(tensor, indices);

        // Construct xarray from the view
        // This forces a copy of that slice into a new xarray
        chunks.push_back(xt::xarray<T>(chunk_view));
        start = end;
    }

    return chunks;
}

template <class Derived, typename T>
class TensorToMesh {
public:
    TensorToMesh(tt::tt_metal::distributed::MeshShape mesh_shape) : m_mesh_shape(std::move(mesh_shape)) {
    }

    std::vector<xt::xarray<T>> map(const xt::xarray<T>& tensor) {
        return static_cast<Derived*>(this)->map_impl(tensor);
    }

    std::unordered_map<std::string, std::string> config() {
        return static_cast<Derived*>(this)->config_impl();
    }

protected:
    tt::tt_metal::distributed::MeshShape m_mesh_shape;
};

template <class Derived, typename T>
class MeshToTensor {
public:
    MeshToTensor(tt::tt_metal::distributed::MeshShape mesh_shape) : m_mesh_shape(std::move(mesh_shape)) {
    }

    xt::xarray<T> compose(const std::vector<xt::xarray<T>>& tensors) {
        return static_cast<Derived*>(this)->compose_impl(tensors);
    }

protected:
    tt::tt_metal::distributed::MeshShape m_mesh_shape;
};

template <typename T>
class ShardTensorToMesh : public TensorToMesh<ShardTensorToMesh<T>, T> {
public:
    using Base = TensorToMesh<ShardTensorToMesh<T>, T>;
    ShardTensorToMesh(tt::tt_metal::distributed::MeshShape mesh_shape, int dim) :
        Base(std::move(mesh_shape)), m_shard_dim(dim) {
    }

    std::vector<xt::xarray<T>> map_impl(const xt::xarray<T>& tensor) {
        int num_devices = Base::m_mesh_shape.first * Base::m_mesh_shape.second;
        auto sliced_tensors = chunk(tensor, num_devices, m_shard_dim);
        return sliced_tensors;
    }

    std::unordered_map<std::string, std::string> config_impl() {
        return {{"strategy", "shard"}, {"shard_dim", std::to_string(m_shard_dim)}};
    }

private:
    int m_shard_dim = 0;
};

template <typename T>
class ShardTensor2dMesh : public TensorToMesh<ShardTensor2dMesh<T>, T> {
public:
    using Base = TensorToMesh<ShardTensor2dMesh<T>, T>;
    ShardTensor2dMesh(
        tt::tt_metal::distributed::MeshShape mesh_shape,
        const std::pair<std::optional<int>, std::optional<int>>& dims) :
        Base(std::move(mesh_shape)), m_dims(dims) {
        // We trust the provided mesh shape and do not validate against a MeshDevice.
    }

    std::vector<xt::xarray<T>> map_impl(const xt::xarray<T>& tensor) {
        if (!m_dims.first.has_value() && !m_dims.second.has_value()) {
            throw std::invalid_argument("ShardTensor2dMesh requires at least one dimension to shard");
        }

        int rows = Base::m_mesh_shape.first;
        int cols = Base::m_mesh_shape.second;
        auto row_dim = m_dims.first;
        auto col_dim = m_dims.second;

        std::vector<xt::xarray<T>> row_tensors;

        // Shard along rows
        if (!row_dim.has_value()) {
            for (int i = 0; i < rows; ++i) {
                row_tensors.push_back(tensor);
            }
        } else {
            row_tensors = chunk(tensor, rows, row_dim.value());
        }

        std::vector<xt::xarray<T>> tensor_shards;

        // Shard along columns
        if (!col_dim.has_value()) {
            for (const auto& t : row_tensors) {
                for (int i = 0; i < cols; ++i) {
                    tensor_shards.push_back(t);
                }
            }
        } else {
            for (const auto& t : row_tensors) {
                auto col_chunks = chunk(t, cols, col_dim.value());
                tensor_shards.insert(tensor_shards.end(), col_chunks.begin(), col_chunks.end());
            }
        }

        if (static_cast<int>(tensor_shards.size()) != rows * cols) {
            throw std::runtime_error(
                "ShardTensor2dMesh: Sharding failed. Number of shards should match the product of the mesh "
                "dimensions.");
        }

        return tensor_shards;
    }

    std::unordered_map<std::string, std::string> config_impl() {
        return {
            {"strategy", "shard_2d"},
            {"mesh_shape_y", std::to_string(Base::m_mesh_shape.first)},
            {"mesh_shape_x", std::to_string(Base::m_mesh_shape.second)}};
    }

private:
    std::pair<std::optional<int>, std::optional<int>> m_dims;
};

template <typename T>
class ConcatMesh2dToTensor : public MeshToTensor<ConcatMesh2dToTensor<T>, T> {
public:
    using Base = MeshToTensor<ConcatMesh2dToTensor<T>, T>;
    ConcatMesh2dToTensor(
        tt::tt_metal::distributed::MeshShape mesh_shape, const tt::tt_metal::distributed::MeshShape& dims) :
        Base(std::move(mesh_shape)), m_dims(dims) {
        if (m_dims.first == m_dims.second) {
            throw std::invalid_argument("Both dimensions in 'dims' must be different");
        }
    }

    xt::xarray<T> compose_impl(const std::vector<xt::xarray<T>>& tensors) {
        int rows = Base::m_mesh_shape.first;
        int cols = Base::m_mesh_shape.second;
        size_t row_dim = m_dims.first;
        size_t col_dim = m_dims.second;

        // Reshape the list of shards into a 2D grid representing the mesh
        std::vector<std::vector<xt::xarray<T>>> mesh_shape_tensors;
        mesh_shape_tensors.reserve(rows);
        for (int i = 0; i < rows; ++i) {
            std::vector<xt::xarray<T>> row_tensors;
            row_tensors.reserve(cols);
            for (int j = 0; j < cols; ++j) {
                int index = i * cols + j;
                row_tensors.push_back(tensors[index]);
            }
            mesh_shape_tensors.push_back(std::move(row_tensors));
        }

        // Concatenate along columns first (within each row)
        std::vector<xt::xarray<T>> row_concatenated;
        row_concatenated.reserve(static_cast<size_t>(rows));
        for (const auto& row : mesh_shape_tensors) {
            auto concatenated_row = core::concatenate(row, col_dim);
            row_concatenated.push_back(std::move(concatenated_row));
        }

        // Then concatenate the resulting tensors along rows
        auto result = core::concatenate(row_concatenated, row_dim);
        return result;
    }

private:
    tt::tt_metal::distributed::MeshShape m_dims;
};

template <typename T>
class ReplicateTensorToMesh : public TensorToMesh<ReplicateTensorToMesh<T>, T> {
public:
    using Base = TensorToMesh<ReplicateTensorToMesh<T>, T>;
    ReplicateTensorToMesh(tt::tt_metal::distributed::MeshShape mesh_shape) : Base(std::move(mesh_shape)) {
    }

    std::vector<xt::xarray<T>> map_impl(const xt::xarray<T>& tensor) {
        int num_devices = Base::m_mesh_shape.first * Base::m_mesh_shape.second;
        std::vector<xt::xarray<T>> tensors;
        tensors.reserve(static_cast<size_t>(num_devices));
        for (int i = 0; i < num_devices; ++i) {
            tensors.push_back(tensor);  // Note: this copies the tensor
        }
        return tensors;
    }

    std::unordered_map<std::string, std::string> config_impl() {
        int num_devices = Base::m_mesh_shape.first * Base::m_mesh_shape.second;
        return {{"strategy", "replicate"}, {"replication_factor", std::to_string(num_devices)}};
    }
};

template <typename T>
class ConcatMeshToTensor : public MeshToTensor<ConcatMeshToTensor<T>, T> {
public:
    using Base = MeshToTensor<ConcatMeshToTensor<T>, T>;
    ConcatMeshToTensor(tt::tt_metal::distributed::MeshShape mesh_shape, int dim) :
        Base(std::move(mesh_shape)), m_concat_dim(dim) {
    }

    xt::xarray<T> compose_impl(const std::vector<xt::xarray<T>>& tensors) {
        auto result = core::concatenate(tensors, m_concat_dim);
        return result;
    }

private:
    int m_concat_dim = 0;
};

template <typename T>
class VectorMeshToTensor {
public:
    VectorMeshToTensor([[maybe_unused]] tt::tt_metal::distributed::MeshShape mesh_shape) {
    }
    std::vector<xt::xarray<T>> compose(const std::vector<xt::xarray<T>>& tensors) {
        return tensors;
    }
};

template <typename T>
using TensorToMeshVariant = std::variant<ShardTensorToMesh<T>, ShardTensor2dMesh<T>, ReplicateTensorToMesh<T>>;

template <typename T>
using MeshToTensorVariant = std::variant<ConcatMeshToTensor<T>, ConcatMesh2dToTensor<T>, VectorMeshToTensor<T>>;

}  // namespace ttml::core
