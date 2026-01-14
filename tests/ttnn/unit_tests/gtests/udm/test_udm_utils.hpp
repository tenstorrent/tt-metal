// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <random>
#include <fmt/format.h>
#include <tt_stl/assert.hpp>

#include <tt-metalium/shape.hpp>
#include <tt-metalium/distributed.hpp>
#include "tt_metal/api/tt-metalium/bfloat16.hpp"

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/api/ttnn/distributed/api.hpp"
#include "ttnn/api/ttnn/distributed/distributed_tensor.hpp"
#include "ttnn/operations/data_movement/common/common.hpp"
#include "ttnn/operations/core/core.hpp"  // for ttnn::to_memory_config

#include "tt_metal/experimental/udm/mesh_program.hpp"
#include "tt_metal/experimental/udm/mesh_builder.hpp"
#include "tt_metal/experimental/udm/mesh_tensor_builder.hpp"
#include "tt_metal/experimental/udm/mesh_utils.hpp"

namespace tt::tt_metal::experimental::udm_tests {

// Use the ShardStrategy enum from ttnn::operations::data_movement
using ShardStrategy = ttnn::operations::data_movement::ShardStrategy;

// Shard order controls how tensor dimensions map to mesh dimensions
// NORMAL: tensor height on mesh dim 0, tensor width on mesh dim 1
// SWAPPED: tensor width on mesh dim 0, tensor height on mesh dim 1
enum class ShardOrder { NORMAL, SWAPPED };

/**
 * @brief Compute ND tensor shape in pages using tensor layout
 * Supports 1D, 2D, and ND tensors
 */
inline tt::tt_metal::Shape compute_tensor_shape_in_pages(
    const tt::tt_metal::Shape& tensor_shape, const tt::tt_metal::TensorLayout& tensor_layout) {
    const size_t rank = tensor_shape.rank();
    TT_FATAL(rank >= 1, "Tensor must have at least 1 dimension");

    // Get physical shape and page shape from tensor layout
    tt::tt_metal::Shape2D physical_shape = tensor_layout.compute_physical_shape(tensor_shape);
    tt::tt_metal::Shape2D page_shape = tensor_layout.compute_page_shape(physical_shape);

    std::vector<uint32_t> shape_in_pages;

    if (rank == 1) {
        // 1D tensor: total pages = (height * width) in pages
        uint32_t total_pages =
            (physical_shape.height() / page_shape.height()) * (physical_shape.width() / page_shape.width());
        shape_in_pages.push_back(total_pages);
    } else {
        // 2D and ND tensors (rank >= 2): preserve batch dims, convert last 2 dims to pages
        for (size_t i = 0; i < rank - 2; ++i) {
            shape_in_pages.push_back(tensor_shape[i]);
        }

        uint32_t h_dim_in_pages = tensor_shape[rank - 2] / page_shape.height();
        uint32_t w_dim_in_pages = tensor_shape[rank - 1] / page_shape.width();
        shape_in_pages.push_back(h_dim_in_pages);
        shape_in_pages.push_back(w_dim_in_pages);
    }

    return tt::tt_metal::Shape(shape_in_pages);
}

/**
 * @brief Create mesh mapper for block-sharded distribution
 * Shards tensor on last 2 dimensions (height and width) across mesh rows and columns
 */
inline std::unique_ptr<ttnn::distributed::TensorToMesh> create_block_sharded_mesh_mapper(
    tt::tt_metal::distributed::MeshDevice* mesh_device, uint32_t tensor_rank, bool swap_shard_order = false) {
    int height_dim = static_cast<int>(tensor_rank) - 2;
    int width_dim = static_cast<int>(tensor_rank) - 1;

    tt::tt_metal::distributed::MeshMapperConfig config;
    if (swap_shard_order) {
        // Swapped: mesh dim 0 shards width, mesh dim 1 shards height
        config.placements = {
            tt::tt_metal::distributed::MeshMapperConfig::Shard{width_dim},
            tt::tt_metal::distributed::MeshMapperConfig::Shard{height_dim}};
    } else {
        // Default: mesh dim 0 shards height, mesh dim 1 shards width
        config.placements = {
            tt::tt_metal::distributed::MeshMapperConfig::Shard{height_dim},
            tt::tt_metal::distributed::MeshMapperConfig::Shard{width_dim}};
    }

    return ttnn::distributed::create_mesh_mapper(*mesh_device, config);
}

/**
 * @brief Create mesh mapper for height-sharded distribution
 * Shards tensor on height dimension, replicates across the other mesh dimension.
 * @param swap_shard_order If false (default): shard height on mesh dim 0, replicate on mesh dim 1
 *                         If true: replicate on mesh dim 0, shard height on mesh dim 1
 */
inline std::unique_ptr<ttnn::distributed::TensorToMesh> create_height_sharded_mesh_mapper(
    tt::tt_metal::distributed::MeshDevice* mesh_device, uint32_t tensor_rank, bool swap_shard_order = false) {
    int height_dim = static_cast<int>(tensor_rank) - 2;

    tt::tt_metal::distributed::MeshMapperConfig config;
    if (swap_shard_order) {
        // Swapped: replicate on mesh dim 0, shard height on mesh dim 1
        config.placements = {
            tt::tt_metal::distributed::MeshMapperConfig::Replicate{},
            tt::tt_metal::distributed::MeshMapperConfig::Shard{height_dim}};
    } else {
        // Default: shard height on mesh dim 0, replicate on mesh dim 1
        config.placements = {
            tt::tt_metal::distributed::MeshMapperConfig::Shard{height_dim},
            tt::tt_metal::distributed::MeshMapperConfig::Replicate{}};
    }

    return ttnn::distributed::create_mesh_mapper(*mesh_device, config);
}

/**
 * @brief Create mesh mapper for width-sharded distribution
 * Shards tensor on width dimension, replicates across the other mesh dimension.
 * @param swap_shard_order If false (default): replicate on mesh dim 0, shard width on mesh dim 1
 *                         If true: shard width on mesh dim 0, replicate on mesh dim 1
 */
inline std::unique_ptr<ttnn::distributed::TensorToMesh> create_width_sharded_mesh_mapper(
    tt::tt_metal::distributed::MeshDevice* mesh_device, uint32_t tensor_rank, bool swap_shard_order = false) {
    int width_dim = static_cast<int>(tensor_rank) - 1;

    tt::tt_metal::distributed::MeshMapperConfig config;
    if (swap_shard_order) {
        // Swapped: shard width on mesh dim 0, replicate on mesh dim 1
        config.placements = {
            tt::tt_metal::distributed::MeshMapperConfig::Shard{width_dim},
            tt::tt_metal::distributed::MeshMapperConfig::Replicate{}};
    } else {
        // Default: replicate on mesh dim 0, shard width on mesh dim 1
        config.placements = {
            tt::tt_metal::distributed::MeshMapperConfig::Replicate{},
            tt::tt_metal::distributed::MeshMapperConfig::Shard{width_dim}};
    }

    return ttnn::distributed::create_mesh_mapper(*mesh_device, config);
}

/**
 * @brief Create mesh composer for aggregating block-sharded tensors
 * Concatenates shards from 2D mesh back to full tensor
 * @param swap_shard_order If true, mesh dim 0 concatenates width, mesh dim 1 concatenates height
 */
inline std::unique_ptr<ttnn::distributed::MeshToTensor> create_block_sharded_mesh_composer(
    tt::tt_metal::distributed::MeshDevice* mesh_device, uint32_t tensor_rank, bool swap_shard_order = false) {
    int height_dim = static_cast<int>(tensor_rank) - 2;
    int width_dim = static_cast<int>(tensor_rank) - 1;

    tt::tt_metal::distributed::MeshComposerConfig config;
    if (swap_shard_order) {
        // Swapped: mesh dim 0 concatenates width, mesh dim 1 concatenates height
        config.dims = {width_dim, height_dim};
    } else {
        // Default: mesh dim 0 concatenates height, mesh dim 1 concatenates width
        config.dims = {height_dim, width_dim};
    }

    return ttnn::distributed::create_mesh_composer(*mesh_device, config);
}

/**
 * @brief Create mesh composer for aggregating height-sharded tensors
 *
 * For a height-sharded distribution:
 * - swap_shard_order=false: Mesh dim 0 shards height, mesh dim 1 replicates
 * - swap_shard_order=true: Mesh dim 0 replicates, mesh dim 1 shards height
 *
 * Output shape will include replicated copies concatenated along the non-sharded dimension.
 * Callers should slice to get original shape.
 */
inline std::unique_ptr<ttnn::distributed::MeshToTensor> create_height_sharded_mesh_composer(
    tt::tt_metal::distributed::MeshDevice* mesh_device, uint32_t tensor_rank, bool swap_shard_order = false) {
    int height_dim = static_cast<int>(tensor_rank) - 2;
    int width_dim = static_cast<int>(tensor_rank) - 1;

    tt::tt_metal::distributed::MeshComposerConfig config;
    if (swap_shard_order) {
        // Swapped: mesh dim 0 concatenates width (replicated), mesh dim 1 concatenates height (sharded)
        config.dims = {width_dim, height_dim};
    } else {
        // Default: mesh dim 0 concatenates height (sharded), mesh dim 1 concatenates width (replicated)
        config.dims = {height_dim, width_dim};
    }

    return ttnn::distributed::create_mesh_composer(*mesh_device, config);
}

/**
 * @brief Create mesh composer for aggregating width-sharded tensors
 *
 * For a width-sharded distribution:
 * - swap_shard_order=false: Mesh dim 0 replicates, mesh dim 1 shards width
 * - swap_shard_order=true: Mesh dim 0 shards width, mesh dim 1 replicates
 *
 * Output shape will include replicated copies concatenated along the non-sharded dimension.
 * Callers should slice to get original shape.
 */
inline std::unique_ptr<ttnn::distributed::MeshToTensor> create_width_sharded_mesh_composer(
    tt::tt_metal::distributed::MeshDevice* mesh_device, uint32_t tensor_rank, bool swap_shard_order = false) {
    int height_dim = static_cast<int>(tensor_rank) - 2;
    int width_dim = static_cast<int>(tensor_rank) - 1;

    tt::tt_metal::distributed::MeshComposerConfig config;
    if (swap_shard_order) {
        // Swapped: mesh dim 0 concatenates width (sharded), mesh dim 1 concatenates height (replicated)
        config.dims = {width_dim, height_dim};
    } else {
        // Default: mesh dim 0 concatenates height (replicated), mesh dim 1 concatenates width (sharded)
        config.dims = {height_dim, width_dim};
    }

    return ttnn::distributed::create_mesh_composer(*mesh_device, config);
}

/**
 * @brief Create a bfloat16 tensor with random values
 */
inline ttnn::Tensor create_bfloat16_tensor_with_random_values(
    const tt::tt_metal::Shape& shape, const tt::tt_metal::TensorSpec& tensor_spec, uint32_t seed = 42) {
    uint32_t volume = 1;
    for (size_t i = 0; i < shape.rank(); ++i) {
        volume *= shape[i];
    }

    std::vector<bfloat16> src_data(volume);

    // Generate random data in bfloat16 range
    std::mt19937 gen(seed);                                  // Fixed seed for reproducibility
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);  // Small values to avoid overflow when summing
    for (uint32_t i = 0; i < volume; ++i) {
        src_data[i] = bfloat16(dis(gen));
    }

    return ttnn::Tensor::from_vector(src_data, tensor_spec);
}

/**
 * @brief Create a bfloat16 tensor with zero values
 */
inline ttnn::Tensor create_bfloat16_tensor_with_zero_values(
    const tt::tt_metal::Shape& shape, const tt::tt_metal::TensorSpec& tensor_spec) {
    uint32_t volume = 1;
    for (size_t i = 0; i < shape.rank(); ++i) {
        volume *= shape[i];
    }

    std::vector<bfloat16> src_data(volume, bfloat16(0.0f));
    return ttnn::Tensor::from_vector(src_data, tensor_spec);
}

/**
 * @brief Create tensor: mesh width-distributed, grid interleaved
 * Width is sharded across mesh devices, with explicit control over which mesh dimension.
 * @param swap_shard_order If false: replicate on mesh dim 0, shard width on mesh dim 1
 *                         If true: shard width on mesh dim 0, replicate on mesh dim 1
 * @param random_init If true, initialize with random values; if false, initialize with zeros
 */
inline ttnn::Tensor create_width_distributed_interleaved_bfloat16_tensor(
    tt::tt_metal::distributed::MeshDevice* mesh_device,
    const tt::tt_metal::Shape& global_shape,
    bool swap_shard_order = false,
    bool random_init = true) {
    tt::tt_metal::MemoryConfig mem_config(tt::tt_metal::TensorMemoryLayout::INTERLEAVED, tt::tt_metal::BufferType::L1);
    tt::tt_metal::TensorSpec tensor_spec(
        global_shape,
        tt::tt_metal::TensorLayout(
            tt::tt_metal::DataType::BFLOAT16, tt::tt_metal::PageConfig(tt::tt_metal::Layout::TILE), mem_config));

    auto host_tensor = random_init ? create_bfloat16_tensor_with_random_values(global_shape, tensor_spec)
                                   : create_bfloat16_tensor_with_zero_values(global_shape, tensor_spec);
    auto mapper = create_width_sharded_mesh_mapper(mesh_device, global_shape.rank(), swap_shard_order);
    return ttnn::distributed::distribute_tensor(host_tensor, *mapper, std::ref(*mesh_device));
}

/**
 * @brief Create tensor: mesh block-distributed, grid interleaved
 * @param swap_shard_order If true, mesh dim 0 shards width, mesh dim 1 shards height
 * @param random_init If true, initialize with random values; if false, initialize with zeros
 */
inline ttnn::Tensor create_block_distributed_interleaved_bfloat16_tensor(
    tt::tt_metal::distributed::MeshDevice* mesh_device,
    const tt::tt_metal::Shape& global_shape,
    bool swap_shard_order = false,
    bool random_init = true) {
    tt::tt_metal::MemoryConfig mem_config(tt::tt_metal::TensorMemoryLayout::INTERLEAVED, tt::tt_metal::BufferType::L1);
    tt::tt_metal::TensorSpec tensor_spec(
        global_shape,
        tt::tt_metal::TensorLayout(
            tt::tt_metal::DataType::BFLOAT16, tt::tt_metal::PageConfig(tt::tt_metal::Layout::TILE), mem_config));

    auto host_tensor = random_init ? create_bfloat16_tensor_with_random_values(global_shape, tensor_spec)
                                   : create_bfloat16_tensor_with_zero_values(global_shape, tensor_spec);
    auto mapper = create_block_sharded_mesh_mapper(mesh_device, global_shape.rank(), swap_shard_order);
    return ttnn::distributed::distribute_tensor(host_tensor, *mapper, std::ref(*mesh_device));
}

/**
 * @brief Create tensor: mesh height-distributed, grid interleaved
 * Height is sharded across mesh devices.
 * @param swap_shard_order If true, shard height on mesh dim 1 instead of mesh dim 0
 * @param random_init If true, initialize with random values; if false, initialize with zeros
 */
inline ttnn::Tensor create_height_distributed_interleaved_bfloat16_tensor(
    tt::tt_metal::distributed::MeshDevice* mesh_device,
    const tt::tt_metal::Shape& global_shape,
    bool swap_shard_order = false,
    bool random_init = true) {
    tt::tt_metal::MemoryConfig mem_config(tt::tt_metal::TensorMemoryLayout::INTERLEAVED, tt::tt_metal::BufferType::L1);
    tt::tt_metal::TensorSpec tensor_spec(
        global_shape,
        tt::tt_metal::TensorLayout(
            tt::tt_metal::DataType::BFLOAT16, tt::tt_metal::PageConfig(tt::tt_metal::Layout::TILE), mem_config));

    auto host_tensor = random_init ? create_bfloat16_tensor_with_random_values(global_shape, tensor_spec)
                                   : create_bfloat16_tensor_with_zero_values(global_shape, tensor_spec);
    auto mapper = create_height_sharded_mesh_mapper(mesh_device, global_shape.rank(), swap_shard_order);
    return ttnn::distributed::distribute_tensor(host_tensor, *mapper, std::ref(*mesh_device));
}

/**
 * @brief Create tensor: mesh block-distributed, grid block-sharded
 * @param grid_size The grid shape {num_cores_x, num_cores_y} to use for sharding within each device
 */
inline ttnn::Tensor create_block_distributed_block_sharded_bfloat16_tensor(
    tt::tt_metal::distributed::MeshDevice* mesh_device,
    const tt::tt_metal::Shape& global_shape,
    const tt::tt_metal::Shape& local_shape,
    std::pair<uint32_t, uint32_t> grid_size) {
    uint32_t num_cores_x = grid_size.first;
    uint32_t num_cores_y = grid_size.second;

    // Calculate shard shape (height and width per shard in elements)
    uint32_t shard_height = local_shape[-2] / num_cores_y;
    uint32_t shard_width = local_shape[-1] / num_cores_x;

    // Step 1: Create host tensor with global_shape and INTERLEAVED memory config
    tt::tt_metal::MemoryConfig interleaved_mem_config(
        tt::tt_metal::TensorMemoryLayout::INTERLEAVED, tt::tt_metal::BufferType::L1);

    tt::tt_metal::TensorSpec host_tensor_spec(
        global_shape,
        tt::tt_metal::TensorLayout(
            tt::tt_metal::DataType::BFLOAT16,
            tt::tt_metal::PageConfig(tt::tt_metal::Layout::TILE),
            interleaved_mem_config));

    auto host_tensor = create_bfloat16_tensor_with_random_values(global_shape, host_tensor_spec);

    // Step 2: Distribute across mesh devices (creates interleaved device tensors)
    auto mapper = create_block_sharded_mesh_mapper(mesh_device, global_shape.rank());
    auto distributed_tensor = ttnn::distributed::distribute_tensor(host_tensor, *mapper, std::ref(*mesh_device));

    // Step 3: Convert to BLOCK_SHARDED using ttnn::to_memory_config
    auto shard_spec = ShardSpec(
        CoreRangeSet({CoreRange({0, 0}, {num_cores_x - 1, num_cores_y - 1})}),
        std::array<uint32_t, 2>{shard_height, shard_width},
        ShardOrientation::ROW_MAJOR);

    tt::tt_metal::MemoryConfig sharded_mem_config(
        tt::tt_metal::TensorMemoryLayout::BLOCK_SHARDED, tt::tt_metal::BufferType::L1, shard_spec);

    return ttnn::to_memory_config(distributed_tensor, sharded_mem_config);
}

/**
 * @brief Create tensor: mesh height-distributed (replicated on width), grid height-sharded
 * Used for reduction outputs where width is reduced to a single tile
 * @param grid_size The grid shape {num_cores_x, num_cores_y} to use for sharding within each device
 */
inline ttnn::Tensor create_height_distributed_height_sharded_bfloat16_tensor(
    tt::tt_metal::distributed::MeshDevice* mesh_device,
    const tt::tt_metal::Shape& global_shape,
    const tt::tt_metal::Shape& local_shape,
    std::pair<uint32_t, uint32_t> grid_size) {
    // For height-sharded output: use only 1 core in X (width reduced to single tile)
    uint32_t num_cores_x = 1;
    uint32_t num_cores_y = grid_size.second;

    // Calculate shard shape
    uint32_t shard_height = local_shape[-2] / num_cores_y;
    uint32_t shard_width = local_shape[-1];  // Full width per core (single tile after reduction)

    // Step 1: Create host tensor with global_shape and INTERLEAVED memory config
    tt::tt_metal::MemoryConfig interleaved_mem_config(
        tt::tt_metal::TensorMemoryLayout::INTERLEAVED, tt::tt_metal::BufferType::L1);

    tt::tt_metal::TensorSpec host_tensor_spec(
        global_shape,
        tt::tt_metal::TensorLayout(
            tt::tt_metal::DataType::BFLOAT16,
            tt::tt_metal::PageConfig(tt::tt_metal::Layout::TILE),
            interleaved_mem_config));

    // Create host tensor with zeros (for output)
    uint32_t volume = 1;
    for (size_t i = 0; i < global_shape.rank(); ++i) {
        volume *= global_shape[i];
    }
    std::vector<bfloat16> src_data(volume, bfloat16(0.0f));
    auto host_tensor = ttnn::Tensor::from_vector(src_data, host_tensor_spec);

    // Step 2: Distribute across mesh devices (height-sharded, replicated on width)
    auto mapper = create_height_sharded_mesh_mapper(mesh_device, global_shape.rank());
    auto distributed_tensor = ttnn::distributed::distribute_tensor(host_tensor, *mapper, std::ref(*mesh_device));

    // Step 3: Convert to HEIGHT_SHARDED using ttnn::to_memory_config
    auto shard_spec = ShardSpec(
        CoreRangeSet({CoreRange({0, 0}, {num_cores_x - 1, num_cores_y - 1})}),
        std::array<uint32_t, 2>{shard_height, shard_width},
        ShardOrientation::ROW_MAJOR);

    tt::tt_metal::MemoryConfig sharded_mem_config(
        tt::tt_metal::TensorMemoryLayout::HEIGHT_SHARDED, tt::tt_metal::BufferType::L1, shard_spec);

    return ttnn::to_memory_config(distributed_tensor, sharded_mem_config);
}

/**
 * @brief Calculate maximum number of pages any gcore will process
 */
inline uint32_t get_max_pages_per_gcore(const tt::tt_metal::experimental::udm::GlobalCoresInfo& gcores_info) {
    uint32_t max_pages = 0;
    for (size_t gcore_idx = 0; gcore_idx < gcores_info.gcores.size(); ++gcore_idx) {
        uint32_t total_pages = 1;
        for (const auto& dim_page : gcores_info.dim_pages[gcore_idx]) {
            total_pages *= dim_page;
        }
        max_pages = std::max(max_pages, total_pages);
    }
    return max_pages;
}

/**
 * @brief Debug helper to log tensor shape info
 */
inline void log_tensor_shape_info(
    const tt::tt_metal::experimental::udm::MeshTensorBuilder& tensor_builder, const ttnn::Tensor& tensor) {
    const auto& mesh_tensor_shape = tensor_builder.get_mesh_tensor_shape_in_pages();

    log_info(tt::LogTest, "=== Tensor Shape Info ===");
    log_info(tt::LogTest, "Mesh tensor shape in pages (rank={}): [{}]", mesh_tensor_shape.rank(), [&]() {
        std::string shape_str;
        for (size_t i = 0; i < mesh_tensor_shape.rank(); ++i) {
            if (i > 0) {
                shape_str += ", ";
            }
            shape_str += std::to_string(mesh_tensor_shape[i]);
        }
        return shape_str;
    }());
    log_info(tt::LogTest, "Tensor padded_shape: {}", tensor.padded_shape());
    log_info(tt::LogTest, "========================");
}

/**
 * @brief Debug helper to log gcores info details
 */
inline void log_gcores_info(
    const tt::tt_metal::experimental::udm::GlobalCoresInfo& gcores_info,
    const tt::tt_metal::experimental::udm::MeshBuilder& mesh_builder) {
    const auto& mesh_shape = mesh_builder.get_mesh().shape();
    const auto& grid_shape = mesh_builder.get_flattened_grid();

    log_info(tt::LogTest, "=== GlobalCores Info ===");
    log_info(
        tt::LogTest,
        "Mesh shape: [{}x{}], Grid shape: [{}x{}]",
        mesh_shape[0],
        mesh_shape[1],
        grid_shape[0],
        grid_shape[1]);
    log_info(tt::LogTest, "Total gcores in result: {}", gcores_info.gcores.size());
    log_info(tt::LogTest, "GlobalCores with work: {}", gcores_info.num_cores);

    std::string partition_dims_str;
    for (size_t d = 0; d < gcores_info.partition_dims.size(); ++d) {
        if (d > 0) {
            partition_dims_str += ", ";
        }
        partition_dims_str += std::to_string(gcores_info.partition_dims[d]);
    }
    log_info(tt::LogTest, "Partition dims: [{}]", partition_dims_str);

    for (size_t i = 0; i < gcores_info.gcores.size(); ++i) {
        const auto& gcore = gcores_info.gcores[i];

        uint32_t total_pages = 1;
        for (const auto& dim_page : gcores_info.dim_pages[i]) {
            total_pages *= dim_page;
        }

        if (total_pages > 0) {
            log_debug(
                tt::LogTest,
                "GlobalCore[{}]: local_id={}, global_id={}, local_coord={}, global_coord={}",
                i,
                gcore.local_id,
                gcore.global_id,
                gcore.local_coord,
                gcore.global_coord);
            log_debug(tt::LogTest, "  Total pages: {}", total_pages);

            if (gcore.global_coord.dims() >= 2) {
                [[maybe_unused]] uint32_t mesh_row = gcore.global_coord[0] / grid_shape[0];
                [[maybe_unused]] uint32_t mesh_col = gcore.global_coord[1] / grid_shape[1];
                [[maybe_unused]] uint32_t grid_row = gcore.global_coord[0] % grid_shape[0];
                [[maybe_unused]] uint32_t grid_col = gcore.global_coord[1] % grid_shape[1];
                log_debug(
                    tt::LogTest,
                    "  → Mesh device: ({}, {}), Grid core: ({}, {}))",
                    mesh_row,
                    mesh_col,
                    grid_row,
                    grid_col);
            }

            std::string pages_str;
            for (size_t d = 0; d < gcores_info.dim_pages[i].size(); ++d) {
                if (d > 0) {
                    pages_str += ", ";
                }
                pages_str += std::to_string(gcores_info.dim_pages[i][d]);
            }
            log_debug(tt::LogTest, "  Dim pages: [{}]", pages_str);

            std::string offsets_str;
            for (size_t d = 0; d < gcores_info.dim_offsets[i].size(); ++d) {
                if (d > 0) {
                    offsets_str += ", ";
                }
                offsets_str += std::to_string(gcores_info.dim_offsets[i][d]);
            }
            log_debug(tt::LogTest, "  Dim offsets: [{}]", offsets_str);

            std::string strides_str;
            for (size_t d = 0; d < gcores_info.dim_strides[i].size(); ++d) {
                if (d > 0) {
                    strides_str += ", ";
                }
                strides_str += std::to_string(gcores_info.dim_strides[i][d]);
            }
            log_debug(tt::LogTest, "  Dim strides: [{}]", strides_str);
        }
    }
    log_info(tt::LogTest, "==================");
}

/**
 * @brief Create MeshTensorBuilder from a distributed tensor
 *
 * Extracts distribution info from the tensor's topology.
 * MeshBuilder automatically extracts the grid shape from the mesh buffer's shard spec.
 */
inline tt::tt_metal::experimental::udm::MeshTensorBuilder create_tensor_builder(const ttnn::Tensor& tensor) {
    // Extract MeshBuffer from the distributed tensor
    TT_FATAL(std::holds_alternative<tt::tt_metal::DeviceStorage>(tensor.storage()), "Tensor must be on device");
    const auto& device_storage = std::get<tt::tt_metal::DeviceStorage>(tensor.storage());
    TT_FATAL(device_storage.mesh_buffer != nullptr, "Tensor must have a MeshBuffer");

    // Extract distribution info from tensor topology
    const auto& topology = tensor.tensor_topology();
    const auto& distribution_shape = topology.distribution_shape();
    const auto& placements = topology.placements();

    // Compute tensor shape in pages from tensor layout
    const auto& tensor_shape = tensor.padded_shape();
    const auto& tensor_layout = tensor.tensor_spec().tensor_layout();
    auto tensor_shape_in_pages = compute_tensor_shape_in_pages(tensor_shape, tensor_layout);

    // Convert placements to shard_dims
    std::vector<std::optional<int>> shard_dims;
    for (const auto& placement : placements) {
        if (std::holds_alternative<ttnn::distributed::MeshMapperConfig::Replicate>(placement)) {
            shard_dims.push_back(std::nullopt);
        } else {
            const auto& shard = std::get<ttnn::distributed::MeshMapperConfig::Shard>(placement);
            shard_dims.push_back(shard.dim);
        }
    }

    return tt::tt_metal::experimental::udm::MeshTensorBuilder(
        *device_storage.mesh_buffer, tensor_shape_in_pages, distribution_shape, shard_dims);
}

/**
 * @brief Run a mesh program on all mesh coordinates
 */
inline void run_program(
    const ttnn::Tensor& /*sharded_tensor*/,
    tt::tt_metal::distributed::MeshDevice* mesh_device,
    tt::tt_metal::experimental::udm::MeshProgram& mesh_program) {
    const auto& mesh_shape = mesh_device->shape();
    auto mesh_coord_range = tt::tt_metal::distributed::MeshCoordinateRange(mesh_shape);

    for (const auto& coord : mesh_coord_range) {
        if (!mesh_program.has_kernel(coord)) {
            continue;
        }

        auto mesh_workload = tt::tt_metal::distributed::MeshWorkload();
        mesh_workload.add_program(
            tt::tt_metal::distributed::MeshCoordinateRange(coord), std::move(mesh_program.program_at(coord)));

        tt::tt_metal::distributed::EnqueueMeshWorkload(mesh_device->mesh_command_queue(), mesh_workload, false);
    }

    tt::tt_metal::distributed::Finish(mesh_device->mesh_command_queue());
}

}  // namespace tt::tt_metal::experimental::udm_tests
