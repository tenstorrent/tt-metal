// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <core/ttnn_all_includes.hpp>
#include <unordered_map>

#include "core/xtensor_utils.hpp"
#include "ttnn/tensor/xtensor/partition.hpp"

namespace ttml::core {

template <typename T>
std::vector<xt::xarray<T>> chunk(const xt::xarray<T>& tensor, int num_chunks, int dim) {
    auto chunks = ttnn::experimental::xtensor::chunk(tensor, num_chunks, dim);
    return std::vector<xt::xarray<T>>(chunks.begin(), chunks.end());
}

template <class Derived, typename T>
class MeshToXTensor {
public:
    MeshToXTensor(tt::tt_metal::distributed::MeshShape mesh_shape) : m_mesh_shape(std::move(mesh_shape)) {
    }

    std::vector<xt::xarray<T>> compose(const std::vector<xt::xarray<T>>& tensors) const {
        return static_cast<Derived const*>(this)->compose_impl(tensors);
    }

protected:
    tt::tt_metal::distributed::MeshShape m_mesh_shape;
};

template <typename T>
class ConcatMesh2dToTensor : public MeshToXTensor<ConcatMesh2dToTensor<T>, T> {
public:
    using Base = MeshToXTensor<ConcatMesh2dToTensor<T>, T>;
    ConcatMesh2dToTensor(
        tt::tt_metal::distributed::MeshShape mesh_shape, const tt::tt_metal::distributed::MeshShape& dims) :
        Base(std::move(mesh_shape)), m_dims(dims) {
        if (m_dims[0] == m_dims[1]) {
            throw std::invalid_argument("Dimensions in 'dims' must be different");
        }
    }

    std::vector<xt::xarray<T>> compose_impl(const std::vector<xt::xarray<T>>& tensors) const {
        int rows = Base::m_mesh_shape[0];
        int cols = Base::m_mesh_shape[1];
        size_t row_dim = m_dims[0];
        size_t col_dim = m_dims[1];

        std::vector<xt::xarray<T>> row_concatenated;
        row_concatenated.reserve(static_cast<size_t>(rows));

        for (int i = 0; i < rows; ++i) {
            auto row_start = tensors.begin() + i * cols;
            auto row_end = row_start + cols;
            std::vector<xt::xarray<T>> row_tensors(row_start, row_end);

            auto concatenated_row = core::concat(row_tensors, col_dim);
            row_concatenated.push_back(std::move(concatenated_row));
        }

        auto result = core::concat(row_concatenated, row_dim);
        return {result};
    }

private:
    tt::tt_metal::distributed::MeshShape m_dims;
};

template <typename T>
class ConcatMeshToXTensor : public MeshToXTensor<ConcatMeshToXTensor<T>, T> {
public:
    using Base = MeshToXTensor<ConcatMeshToXTensor<T>, T>;
    ConcatMeshToXTensor(tt::tt_metal::distributed::MeshShape mesh_shape, int dim) :
        Base(std::move(mesh_shape)), m_concat_dim(dim) {
    }

    std::vector<xt::xarray<T>> compose_impl(const std::vector<xt::xarray<T>>& tensors) const {
        return {core::concat(tensors, m_concat_dim)};
    }

private:
    int m_concat_dim = 0;
};

template <typename T>
class VectorMeshToXTensor : public MeshToXTensor<VectorMeshToXTensor<T>, T> {
public:
    using Base = MeshToXTensor<VectorMeshToXTensor<T>, T>;
    VectorMeshToXTensor([[maybe_unused]] tt::tt_metal::distributed::MeshShape mesh_shape) : Base(mesh_shape) {
    }
    std::vector<xt::xarray<T>> compose_impl(const std::vector<xt::xarray<T>>& tensors) const {
        return tensors;
    }
};

template <typename T>
using MeshToXTensorVariant = std::variant<ConcatMeshToXTensor<T>, ConcatMesh2dToTensor<T>, VectorMeshToXTensor<T>>;

}  // namespace ttml::core
