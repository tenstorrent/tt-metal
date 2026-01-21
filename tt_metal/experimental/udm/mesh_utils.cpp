// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/experimental/udm/mesh_utils.hpp"
#include "tt_metal/experimental/udm/mesh_tensor_builder.hpp"
#include "tt_metal/experimental/udm/mesh_builder.hpp"
#include <tt_stl/assert.hpp>
#include <cmath>

namespace tt::tt_metal::experimental::udm {

void compute_strides(
    const std::array<uint32_t, MAX_RANK>& shape, uint32_t rank, std::array<uint32_t, MAX_RANK>& strides) {
    uint32_t stride = 1;
    for (int i = rank - 1; i >= 0; --i) {
        strides[i] = stride;
        stride *= shape[i];
    }
}

void adjust_array_to_rank(
    std::array<uint32_t, MAX_RANK>& arr, uint32_t& current_rank, uint32_t target_rank, uint32_t fill_value) {
    if (current_rank >= target_rank) {
        return;
    }

    uint32_t diff = target_rank - current_rank;
    // Shift existing elements to the right
    for (int i = static_cast<int>(current_rank) - 1; i >= 0; --i) {
        arr[diff + i] = arr[i];
    }
    // Fill beginning with fill_value
    for (uint32_t i = 0; i < diff; ++i) {
        arr[i] = fill_value;
    }
    current_rank = target_rank;
}

void adjust_shape_ranks(
    std::array<uint32_t, MAX_RANK>& arr1,
    uint32_t& rank1,
    std::array<uint32_t, MAX_RANK>& arr2,
    uint32_t& rank2,
    uint32_t fill_value) {
    if (rank1 == rank2) {
        return;
    }

    if (rank1 < rank2) {
        // Prepend fill_value to arr1
        uint32_t diff = rank2 - rank1;
        for (int i = rank1 - 1; i >= 0; --i) {
            arr1[i + diff] = arr1[i];
        }
        for (uint32_t i = 0; i < diff; ++i) {
            arr1[i] = fill_value;
        }
        rank1 = rank2;
    } else {
        // Prepend fill_value to arr2
        uint32_t diff = rank1 - rank2;
        for (int i = rank2 - 1; i >= 0; --i) {
            arr2[i + diff] = arr2[i];
        }
        for (uint32_t i = 0; i < diff; ++i) {
            arr2[i] = fill_value;
        }
        rank2 = rank1;
    }
}

namespace {

/**
 * @brief Factor num_cores into N factors, trying to keep them balanced
 *
 * For example: factor_cores(12, 2) might return {3, 4} or {4, 3}
 */
std::vector<uint32_t> factor_cores_into_dims(uint32_t num_cores, size_t num_dims) {
    TT_FATAL(num_cores > 0 && num_dims > 0, "num_cores and num_dims must be positive");

    std::vector<uint32_t> factors(num_dims, 1);

    if (num_dims == 1) {
        factors[0] = num_cores;
        return factors;
    }

    // Try to make factors as balanced as possible
    // Start with target = num_cores^(1/num_dims)
    uint32_t remaining = num_cores;
    double target = std::pow(static_cast<double>(num_cores), 1.0 / static_cast<double>(num_dims));

    for (size_t d = 0; d < num_dims - 1; ++d) {
        // Find the largest factor <= target that divides remaining
        uint32_t factor = static_cast<uint32_t>(target);
        while (factor > 1 && remaining % factor != 0) {
            factor--;
        }
        if (factor == 1 && remaining > 1) {
            factor = remaining;  // Use all remaining if no good factor found
        }

        factors[d] = factor;
        remaining /= factor;

        // Update target for remaining dimensions
        if (d < num_dims - 2) {
            target = std::pow(static_cast<double>(remaining), 1.0 / static_cast<double>(num_dims - d - 1));
        }
    }

    // Last dimension gets whatever's left
    factors[num_dims - 1] = remaining;

    return factors;
}

/**
 * @brief Calculate the number of pages and starting offset for a given core in a partitioned dimension
 *
 * Distributes dim_size_in_pages across num_cores:
 * - Each core gets at least (dim_size_in_pages / num_cores) pages
 * - The first (dim_size_in_pages % num_cores) cores get one extra page
 *
 * @return pair<num_pages, offset> where:
 *   - num_pages: actual pages for this core
 *   - offset: starting page offset for this core
 */
std::pair<uint32_t, uint32_t> get_core_pages(uint32_t dim_size_in_pages, uint32_t num_cores, uint32_t core_coord) {
    // Standard distribution: floor division + remainder distribution
    uint32_t base_pages = dim_size_in_pages / num_cores;  // floor division
    uint32_t remainder = dim_size_in_pages % num_cores;   // extra pages to distribute

    uint32_t num_pages;
    if (core_coord < remainder) {
        // First 'remainder' cores get one extra page
        num_pages = base_pages + 1;
    } else {
        // Remaining cores get base_pages (may be 0 if more cores than pages)
        num_pages = base_pages;
    }

    // Calculate the actual starting offset for this core
    // Cores 0 to (remainder-1) each get (base_pages + 1) pages
    // Cores (remainder) onwards each get (base_pages) pages
    // offset = core_coord * base_pages + min(core_coord, remainder)
    uint32_t offset = (core_coord * base_pages) + std::min(core_coord, remainder);

    return {num_pages, offset};
}

/**
 * @brief Find the partition index for a given dimension
 *
 * Returns the index in the partition_dims array, or -1 if not partitioned
 */
int find_partition_index(int dim, const std::vector<int>& partition_dims) {
    for (size_t i = 0; i < partition_dims.size(); ++i) {
        if (partition_dims[i] == dim) {
            return static_cast<int>(i);
        }
    }
    return -1;
}

/**
 * @brief Convert linear core index to multi-dimensional coordinates (row-major)
 */
void compute_core_coords(
    uint32_t core_idx, const std::vector<uint32_t>& cores_per_dim, std::vector<uint32_t>& core_coords) {
    uint32_t temp = core_idx;
    for (int d = cores_per_dim.size() - 1; d >= 0; --d) {
        core_coords[d] = temp % cores_per_dim[d];
        temp /= cores_per_dim[d];
    }
}

/**
 * @brief Check if a core would be assigned any work
 *
 * Returns false if the core would get 0 pages in any partitioned dimension
 */
bool core_has_work(
    const std::vector<uint32_t>& core_coords,
    const std::vector<uint32_t>& dim_sizes,
    const std::vector<uint32_t>& cores_per_dim) {
    for (size_t d = 0; d < core_coords.size(); ++d) {
        auto [num_pages, base_pages] = get_core_pages(dim_sizes[d], cores_per_dim[d], core_coords[d]);
        if (num_pages == 0) {
            return false;
        }
    }
    return true;
}

}  // anonymous namespace

GlobalCoresInfo map_tensor_to_gcores(
    const MeshTensorBuilder& tensor_builder, const MeshBuilder& mesh_builder, int partition_dim) {
    // Single-dimension partitioning - just call the ND version
    return map_tensor_to_gcores_nd(tensor_builder, mesh_builder, {partition_dim});
}

GlobalCoresInfo map_tensor_to_gcores_nd(
    const MeshTensorBuilder& tensor_builder, const MeshBuilder& mesh_builder, const std::vector<int>& partition_dims) {
    TT_FATAL(!partition_dims.empty(), "partition_dims cannot be empty");

    // Get mesh and grid dimensions from mesh_builder
    const auto& mesh_shape = mesh_builder.get_mesh().shape();
    const auto& grid_shape = mesh_builder.get_flattened_grid();
    uint32_t mesh_volume = mesh_shape.volume();
    uint32_t grid_volume = grid_shape.volume();
    uint32_t total_gcores = mesh_volume * grid_volume;

    TT_FATAL(total_gcores > 0, "No gcores in mesh");

    // Get the mesh tensor shape in pages (for proper work distribution based on memory pages)
    const auto& mesh_tensor_shape_in_pages = tensor_builder.get_mesh_tensor_shape_in_pages();
    uint32_t tensor_rank = mesh_tensor_shape_in_pages.rank();

    // Compute row-major strides for the mesh tensor shape
    std::vector<uint32_t> mesh_tensor_strides(tensor_rank);
    uint32_t stride = 1;
    for (int i = tensor_rank - 1; i >= 0; --i) {
        mesh_tensor_strides[i] = stride;
        stride *= mesh_tensor_shape_in_pages[i];
    }

    // Validate and normalize partition dimensions
    std::vector<int> normalized_dims;
    std::vector<uint32_t> dim_sizes;
    for (int dim : partition_dims) {
        if (dim < 0) {
            dim = tensor_rank + dim;
        }
        TT_FATAL(
            dim >= 0 && static_cast<size_t>(dim) < tensor_rank,
            "partition_dim {} out of bounds for tensor rank {}",
            dim,
            tensor_rank);
        normalized_dims.push_back(dim);
        dim_sizes.push_back(mesh_tensor_shape_in_pages[dim]);
    }

    // Factor the total gcores into partition dimensions
    // We need to distribute work across mesh×grid dimensions
    std::vector<uint32_t> cores_per_dim = factor_cores_into_dims(total_gcores, partition_dims.size());

    GlobalCoresInfo info;
    info.partition_dims = normalized_dims;

    // Reserve space for all gcores (including those with no work)
    info.gcores.reserve(total_gcores);
    info.dim_offsets.reserve(total_gcores);
    info.dim_pages.reserve(total_gcores);
    info.dim_strides.reserve(total_gcores);

    // Iterate in the order: grid dims first (outer loop), then mesh dims (inner loop)
    // This spreads work across mesh devices (grids) evenly before filling all cores on a single device
    uint32_t assigned_cores = 0;
    uint32_t gcore_idx = 0;
    std::vector<uint32_t> core_coords(partition_dims.size(), 0);

    // Helper lambda to iterate through multi-dimensional coordinates
    auto iterate_shape = [](const tt::tt_metal::Shape& shape, auto callback) {
        uint32_t volume = shape.volume();
        for (uint32_t idx = 0; idx < volume; ++idx) {
            // Convert linear index to multi-dimensional coordinate (row-major)
            std::vector<uint32_t> coord(shape.rank());
            uint32_t temp = idx;
            for (int d = shape.rank() - 1; d >= 0; --d) {
                coord[d] = temp % shape[d];
                temp /= shape[d];
            }
            callback(coord);
        }
    };

    // Iterate: grid coordinates (outer), then mesh coordinates (inner)
    // This spreads work across mesh devices: work0->grid0, work1->grid1, work2->grid2, work3->grid3
    iterate_shape(grid_shape, [&](const std::vector<uint32_t>& grid_coord_vec) {
        // Convert grid coordinate vector to MeshCoordinate
        tt::tt_metal::distributed::MeshCoordinate grid_local_coord(
            tt::stl::Span<const uint32_t>(grid_coord_vec.data(), grid_coord_vec.size()));

        iterate_shape(mesh_shape, [&](const std::vector<uint32_t>& mesh_coord_vec) {
            // Convert mesh coordinate vector to MeshCoordinate for grid lookup
            tt::tt_metal::distributed::MeshCoordinate mesh_coord(
                tt::stl::Span<const uint32_t>(mesh_coord_vec.data(), mesh_coord_vec.size()));

            // Get the gcore at this (mesh, grid) coordinate
            const auto& gcore = mesh_builder.get_gcore_with_local_coord(mesh_coord, grid_local_coord);

            // Calculate core work coordinates from gcore_idx
            compute_core_coords(gcore_idx, cores_per_dim, core_coords);

            // Check if this core has work
            bool has_work = core_has_work(core_coords, dim_sizes, cores_per_dim);

            // Add gcore to the list (even if it has no work)
            info.gcores.push_back(gcore);

            // Prepare dimension info vectors
            std::vector<uint32_t> offsets, pages, strides;
            offsets.reserve(tensor_rank);
            pages.reserve(tensor_rank);
            strides.reserve(tensor_rank);

            if (has_work) {
                // GlobalCore has work: compute actual workload
                for (uint32_t d = 0; d < tensor_rank; ++d) {
                    // Check if this dimension is partitioned
                    int partition_idx = find_partition_index(static_cast<int>(d), normalized_dims);

                    uint32_t num_pages, offset;

                    if (partition_idx >= 0) {
                        // Partitioned dimension: compute assignment for this core
                        auto [core_pages, core_offset] = get_core_pages(
                            dim_sizes[partition_idx], cores_per_dim[partition_idx], core_coords[partition_idx]);
                        num_pages = core_pages;
                        offset = core_offset;
                    } else {
                        // Non-partitioned dimension: process entire dimension
                        num_pages = mesh_tensor_shape_in_pages[d];
                        offset = 0;
                    }

                    offsets.push_back(offset);
                    pages.push_back(num_pages);
                    strides.push_back(mesh_tensor_strides[d]);
                }
                assigned_cores++;
            } else {
                // GlobalCore has no work: assign empty workload (0 pages for all dimensions)
                for (uint32_t d = 0; d < tensor_rank; ++d) {
                    offsets.push_back(0);
                    pages.push_back(0);
                    strides.push_back(mesh_tensor_strides[d]);
                }
            }

            // Add this core's dimension info to the result
            info.dim_offsets.push_back(std::move(offsets));
            info.dim_pages.push_back(std::move(pages));
            info.dim_strides.push_back(std::move(strides));

            gcore_idx++;
        });  // end mesh iteration
    });      // end grid iteration

    info.num_cores = assigned_cores;
    return info;
}

}  // namespace tt::tt_metal::experimental::udm
