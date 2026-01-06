// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>
#include <set>
#include "tt_metal/experimental/udm/types.hpp"
#include <tt-metalium/mesh_buffer.hpp>

namespace tt::tt_metal::experimental::udm {

constexpr size_t MAX_RANK = 8;

/**
 * @brief Compute row-major strides from a shape array
 *
 * @param shape Array of shape dimensions
 * @param rank Number of dimensions
 * @param strides Output array to store computed strides
 */
void compute_strides(
    const std::array<uint32_t, MAX_RANK>& shape, uint32_t rank, std::array<uint32_t, MAX_RANK>& strides);

/**
 * @brief Adjust a single array to a target rank by prepending a fill value
 *
 * @param arr Array to adjust
 * @param current_rank Current rank (modified to target_rank)
 * @param target_rank Target rank to adjust to
 * @param fill_value Value to prepend (1 for shapes, 0 for coordinates)
 */
void adjust_array_to_rank(
    std::array<uint32_t, MAX_RANK>& arr, uint32_t& current_rank, uint32_t target_rank, uint32_t fill_value);

/**
 * @brief Adjust two arrays to have the same rank by prepending a fill value to the shorter one
 *
 * @param arr1 First array to adjust
 * @param rank1 Rank of first array (modified)
 * @param arr2 Second array to adjust
 * @param rank2 Rank of second array (modified)
 * @param fill_value Value to prepend (1 for shapes, 0 for coordinates)
 */
void adjust_shape_ranks(
    std::array<uint32_t, MAX_RANK>& arr1,
    uint32_t& rank1,
    std::array<uint32_t, MAX_RANK>& arr2,
    uint32_t& rank2,
    uint32_t fill_value = 1);

/**
 * @brief Information about how gcores map to tensor work
 *
 * Supports multi-dimensional partitioning where work is distributed across
 * multiple tensor dimensions simultaneously.
 *
 * Note: dim_offsets, dim_pages, and dim_strides contain information for ALL dimensions,
 * not just partitioned ones. Non-partitioned dimensions have offset=0 and pages=full_dim_size.
 *
 * Example usage in kernel for general ND iteration:
 * @code
 *   // Iterate through all assigned pages using multi-dimensional loop
 *   for (uint32_t iter = 0; iter < total_pages; ++iter) {
 *       // Convert linear iter to multi-dim indices
 *       // Compute page_id using offsets and strides
 *       // Use page_id to access memory...
 *   }
 * @endcode
 */
struct GlobalCoresInfo {
    std::vector<GlobalCore> gcores;  // Vector of gcores assigned to work
    uint32_t num_cores{};            // Number of cores

    // Multi-dimensional partitioning info (all in page units)
    std::vector<int> partition_dims;                 // Which dimensions are partitioned (e.g., [1, 3])
    std::vector<std::vector<uint32_t>> dim_offsets;  // [gcore_idx][dim] -> starting page index for ALL dims
    std::vector<std::vector<uint32_t>> dim_pages;    // [gcore_idx][dim] -> number of pages for ALL dims
    std::vector<std::vector<uint32_t>> dim_strides;  // [gcore_idx][dim] -> row-major stride for ALL dims
};

/**
 * @brief Map a tensor to global cores based on work partitioning strategy
 *
 * Single-dimension partitioning version.
 *
 * @param tensor_builder The MeshTensorBuilder containing tensor information
 * @param mesh_builder The MeshBuilder containing mesh and grid dimension information
 * @param partition_dim The dimension to partition work on (-1 for last dim)
 * @return GlobalCoresInfo containing gcore assignment and work distribution
 */
GlobalCoresInfo map_tensor_to_gcores(
    const class MeshTensorBuilder& tensor_builder, const class MeshBuilder& mesh_builder, int partition_dim = -1);

/**
 * @brief Map a tensor to global cores with multi-dimensional partitioning
 *
 * Partitions tensor across multiple dimensions simultaneously. Work is distributed
 * by iterating grid dimensions first, then mesh dimensions, to evenly spread work
 * across different mesh devices (grids). All gcores in the mesh×grid space will be
 * assigned entries, with edge/corner gcores receiving empty GlobalCoresInfo if they have no work.
 *
 * For example, with 4 workloads and 4 grids (mesh devices):
 * Iteration order: grid[0], grid[1], then mesh[0], mesh[1], mesh[2], mesh[3]
 * - Grid 0, Mesh 0: work 0
 * - Grid 0, Mesh 1: work 1
 * - Grid 0, Mesh 2: work 2
 * - Grid 0, Mesh 3: work 3
 * - Grid 1, Mesh 0: work 4 (or empty if no more work)
 * This ensures work is spread across mesh devices before filling cores on a single device.
 *
 * @param tensor_builder The MeshTensorBuilder containing tensor information
 * @param mesh_builder The MeshBuilder containing mesh and grid dimension information
 * @param partition_dims The dimensions to partition work on (e.g., {0, 2})
 * @return GlobalCoresInfo containing gcore assignment and work distribution for all gcores
 */
GlobalCoresInfo map_tensor_to_gcores_nd(
    const class MeshTensorBuilder& tensor_builder,
    const class MeshBuilder& mesh_builder,
    const std::vector<int>& partition_dims);

}  // namespace tt::tt_metal::experimental::udm
