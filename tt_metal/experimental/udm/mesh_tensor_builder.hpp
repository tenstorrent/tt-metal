// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>
#include <vector>
#include <optional>
#include <utility>
#include "tt_metal/experimental/udm/types.hpp"
#include <tt-metalium/mesh_buffer.hpp>
#include <tt-metalium/mesh_coord.hpp>

namespace tt::tt_metal::experimental::udm {

/**
 * @brief Host-side arguments builder for MeshTensorAccessor (device-side struct)
 *
 * MeshTensorBuilder extracts mesh/grid/tensor configuration from MeshBuffer,
 * reconstructs mesh tensor shape from distribution info, and generates
 * compile-time arguments for the device-side MeshTensorAccessor struct.
 * This is analogous to TensorAccessorArgs for TensorAccessor.
 */
class MeshTensorBuilder {
public:
    /**
     * @brief Create a MeshTensorBuilder with distribution information
     *
     * @param mesh_buffer The distributed mesh buffer
     * @param tensor_shape_in_pages The local tensor shape in pages (per device)
     *        Must be computed externally using TensorLayout::compute_physical_shape()
     *        and TensorLayout::compute_page_shape()
     *        Example: For TILE layout with shape (128, 512) and tile (32, 32):
     *                 tensor_shape_in_pages = (4, 16)
     * @param distribution_shape The shape of the mesh used for distribution
     * @param shard_dims Vector of optional shard dimensions for each mesh dimension
     *                   (nullopt = replicate, value = shard on that tensor dim)
     */
    MeshTensorBuilder(
        const tt::tt_metal::distributed::MeshBuffer& mesh_buffer,
        const tt::tt_metal::Shape& tensor_shape_in_pages,
        const tt::tt_metal::distributed::MeshShape& distribution_shape,
        const std::vector<std::optional<int>>& shard_dims);

    ~MeshTensorBuilder();

    // Delete copy, allow move
    MeshTensorBuilder(const MeshTensorBuilder&) = delete;
    MeshTensorBuilder& operator=(const MeshTensorBuilder&) = delete;
    MeshTensorBuilder(MeshTensorBuilder&&) noexcept;
    MeshTensorBuilder& operator=(MeshTensorBuilder&&) noexcept;

    /**
     * @brief Get compile-time arguments for tensor access in kernels
     */
    std::vector<uint32_t> get_compile_time_args() const;

    /**
     * @brief Get the buffer address
     */
    uint64_t get_buffer_address() const;

    /**
     * @brief Get the aligned page size
     */
    uint32_t get_aligned_page_size() const;

    /**
     * @brief Get the mesh tensor shape in pages (global shape across all devices)
     *
     * Note: The shape is in page units, not elements, as this is what the device-side
     * MeshTensorAccessor expects for proper work distribution and memory access.
     */
    tt::tt_metal::Shape get_mesh_tensor_shape_in_pages() const;

    /**
     * @brief Get the local tensor shape in pages (per device)
     */
    tt::tt_metal::Shape get_local_tensor_shape_in_pages() const;

    /**
     * @brief Get the mesh shape (mesh_tensor_shape / tensor_shape for each dimension)
     * This represents how tensor pages are distributed across the mesh.
     */
    tt::tt_metal::Shape get_mesh_shape() const;

    /**
     * @brief Get the distribution shape (mesh device shape used for distribution)
     */
    const tt::tt_metal::distributed::MeshShape& get_distribution_shape() const;

    /**
     * @brief Get the shard dimensions
     * @return Vector of optional ints - nullopt means replicate, value means shard on that tensor dim
     */
    const std::vector<std::optional<int>>& get_shard_dims() const;

    /**
     * @brief Get fabric node IDs reordered for tensor-space iteration
     * @return Pair of vectors: (reordered_fabric_mesh_ids, reordered_fabric_chip_ids)
     */
    std::pair<std::vector<uint32_t>, std::vector<uint32_t>> get_reordered_fabric_node_ids() const;

    /**
     * @brief Get the input mesh buffer
     */
    const tt::tt_metal::distributed::MeshBuffer& mesh_buffer() const;

    /**
     * @brief Get the mesh builder
     */
    MeshBuilder& mesh_builder();

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace tt::tt_metal::experimental::udm
