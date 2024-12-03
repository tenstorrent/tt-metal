// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <core/xtensor_all_includes.hpp>
#include <unordered_map>

#include "mesh_device.hpp"

namespace ttml::core {

template <typename T>
std::vector<xt::xarray<T>> chunk(const xt::xarray<T>& tensor, int num_chunks, int dim) {
    int size_along_dim = tensor.shape()[dim];
    int chunk_size = size_along_dim / num_chunks;
    int remainder = size_along_dim % num_chunks;

    std::vector<xt::xarray<T>> chunks;
    int start = 0;
    for (int i = 0; i < num_chunks; ++i) {
        int current_chunk_size = chunk_size + (i < remainder ? 1 : 0);  // Distribute remainder
        int end = start + current_chunk_size;

        // Build indices for slicing
        xt::xstrided_slice_vector indices(tensor.dimension(), xt::all());
        indices[dim] = xt::range(start, end);

        auto chunk = xt::strided_view(tensor, indices);
        chunks.push_back(xt::xarray<T>(chunk));
        start = end;
    }
    return chunks;
}

template <class Derived, typename T>
class TensorToMesh {
public:
    TensorToMesh(std::shared_ptr<ttnn::distributed::MeshDevice> device) : m_mesh_device(std::move(device)) {
    }
    std::vector<xt::xarray<T>> map(const xt::xarray<T>& tensor) {
        return static_cast<Derived*>(this)->map_impl(tensor);
    }

    std::unordered_map<std::string, std::string> config() {
        return static_cast<Derived*>(this)->config_impl();
    }

protected:
    std::shared_ptr<ttnn::distributed::MeshDevice> m_mesh_device;
};

template <class Derived, typename T>
class MeshToTensor {
public:
    xt::xarray<T> compose(const std::vector<xt::xarray<T>>& tensors) {
        return static_cast<Derived*>(this)->compose_impl(tensors);
    }
};

template <typename T>
class ShardTensorToMesh : public TensorToMesh<ShardTensorToMesh<T>, T> {
public:
    using Base = TensorToMesh<ShardTensorToMesh<T>, T>;
    ShardTensorToMesh(const std::shared_ptr<ttnn::distributed::MeshDevice>& mesh_device, int dim) :
        Base(mesh_device), m_shard_dim(dim) {
    }

    std::vector<xt::xarray<T>> map_impl(const xt::xarray<T>& tensor) {
        int num_devices = Base::m_mesh_device->num_devices();
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
        const MeshDevice& mesh_device,
        const std::pair<int, int>& mesh_shape,
        const std::pair<std::optional<int>, std::optional<int>>& dims) :
        Base(mesh_device), m_mesh_shape(mesh_shape), m_dims(dims) {
        if (m_mesh_shape.first > Base::m_mesh_device->shape().first ||
            m_mesh_shape.second > Base::m_mesh_device->shape().second) {
            throw std::invalid_argument("ShardTensor2dMesh: Device mesh shape does not match the provided mesh shape.");
        }
    }

    std::vector<xt::xarray<T>> map_impl(const xt::xarray<T>& tensor) {
        if (!m_dims.first.has_value() && !m_dims.second.has_value()) {
            throw std::invalid_argument("ShardTensor2dMesh requires at least one dimension to shard");
        }

        int rows = m_mesh_shape.first;
        int cols = m_mesh_shape.second;
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

        if (tensor_shards.size() != rows * cols) {
            throw std::runtime_error(
                "ShardTensor2dMesh: Sharding failed. Number of shards should match the product of the mesh "
                "dimensions.");
        }

        return tensor_shards;
    }

    std::unordered_map<std::string, std::string> config_impl() {
        return {
            {"strategy", "shard_2d"},
            {"mesh_shape_y", std::to_string(m_mesh_shape.first)},
            {"mesh_shape_x", std::to_string(m_mesh_shape.second)}};
    }

private:
    std::pair<int, int> m_mesh_shape;
    std::pair<std::optional<int>, std::optional<int>> m_dims;
};

// ConcatMesh2dToTensor using CRTP
template <typename T>
class ConcatMesh2dToTensor : public MeshToTensor<ConcatMesh2dToTensor<T>, T> {
public:
    using Base = TensorToMesh<ConcatMesh2dToTensor<T>, T>;
    ConcatMesh2dToTensor(
        const MeshDevice& mesh_device, const std::pair<int, int>& mesh_shape, const std::pair<int, int>& dims) :
        Base(mesh_device), m_mesh_shape(mesh_shape), m_dims(dims) {
        if (m_dims.first == m_dims.second) {
            throw std::invalid_argument("Both dimensions in 'dims' must be different");
        }
    }

    xt::xarray<T> compose_impl(const std::vector<xt::xarray<T>>& tensors) {
        int rows = m_mesh_shape.first;
        int cols = m_mesh_shape.second;
        int row_dim = m_dims.first;
        int col_dim = m_dims.second;

        // Reshape the list of shards into a 2D list representing the device mesh
        std::vector<std::vector<xt::xarray<T>>> mesh_shape_tensors;
        for (int i = 0; i < rows; ++i) {
            std::vector<xt::xarray<T>> row_tensors;
            for (int j = 0; j < cols; ++j) {
                int index = i * cols + j;
                row_tensors.push_back(tensors[index]);
            }
            mesh_shape_tensors.push_back(row_tensors);
        }

        // Concatenate along columns first (within each row)
        std::vector<xt::xarray<T>> row_concatenated;
        for (const auto& row : mesh_shape_tensors) {
            auto concatenated_row = xt::concatenate(row, col_dim);
            row_concatenated.push_back(concatenated_row);
        }

        // Then concatenate the resulting tensors along rows
        auto result = xt::concatenate(row_concatenated, row_dim);
        return result;
    }

private:
    std::pair<int, int> m_mesh_shape;
    std::pair<int, int> m_dims;
};

// ReplicateTensorToMesh using CRTP
template <typename T>
class ReplicateTensorToMesh : public TensorToMesh<ReplicateTensorToMesh<T>, T> {
public:
    using Base = TensorToMesh<ReplicateTensorToMesh<T>, T>;
    ReplicateTensorToMesh(const MeshDevice& mesh_device) : Base(mesh_device) {
    }

    std::vector<xt::xarray<T>> map_impl(const xt::xarray<T>& tensor) {
        int num_devices = Base::m_mesh_device->num_devices();
        std::vector<xt::xarray<T>> tensors;
        for (int i = 0; i < num_devices; ++i) {
            tensors.push_back(tensor);  // Note: this copies the tensor
        }
        return tensors;
    }

    std::unordered_map<std::string, std::string> config_impl() {
        return {{"strategy", "replicate"}, {"replication_factor", std::to_string(Base::m_mesh_device->num_devices())}};
    }
};

template <typename T>
class ConcatMeshToTensor : public MeshToTensor<ConcatMeshToTensor<T>, T> {
public:
    using Base = MeshToTensor<ConcatMeshToTensor<T>, T>;
    ConcatMeshToTensor(const MeshDevice& mesh_device, int dim) : Base(mesh_device), m_concat_dim(dim) {
    }

    xt::xarray<T> compose_impl(const std::vector<xt::xarray<T>>& tensors) {
        auto result = xt::concatenate(tensors, m_concat_dim);
        return result;
    }

private:
    int m_concat_dim = 0;
};

template <typename T>
class VectorMeshToTensor : public MeshToTensor<VectorMeshToTensor<T>, T> {
public:
    using Base = MeshToTensor<VectorMeshToTensor<T>, T>;
    VectorMeshToTensor(const MeshDevice& mesh_device) : Base(mesh_device) {
    }

    std::vector<xt::xarray<T>> compose(const std::vector<xt::xarray<T>>& tensors) {
        return tensors;
    }
};

}  // namespace ttml::core
