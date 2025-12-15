// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <fmt/format.h>
#include <tt_stl/assert.hpp>
#include <utility>
#include <random>

#include "tests/tt_metal/tt_metal/common/multi_device_fixture.hpp"
#include "tt_metal/test_utils/stimulus.hpp"

#include <tt-metalium/shape.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/api/ttnn/distributed/api.hpp"
#include "ttnn/api/ttnn/distributed/distributed_tensor.hpp"
#include "ttnn/operations/data_movement/common/common.hpp"

#include "tt_metal/udm/mesh_program.hpp"
#include "tt_metal/udm/mesh_builder.hpp"
#include "tt_metal/udm/mesh_kernel.hpp"
#include "tt_metal/udm/mesh_utils.hpp"
#include "tt_metal/udm/mesh_tensor_builder.hpp"
#include "tt_metal/udm/mesh_circular_buffer.hpp"

namespace tt::tt_metal::experimental::udm_tests {

// Use the ShardStrategy enum from ttnn::operations::data_movement
using ShardStrategy = ttnn::operations::data_movement::ShardStrategy;

/**
 * @brief Compute ND tensor shape in pages using tensor layout
 * Supports 1D, 2D, and ND tensors
 */
tt::tt_metal::Shape compute_tensor_shape_in_pages(
    const tt::tt_metal::Shape& tensor_shape, const tt::tt_metal::TensorLayout& tensor_layout) {
    const size_t rank = tensor_shape.rank();
    TT_ASSERT(rank >= 1, "Tensor must have at least 1 dimension");

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
 * @brief Create mesh mapper for width-sharded distribution
 * Shards tensor along last dimension (width) across all devices in the mesh (treated as 1D)
 *
 * @param mesh_device The mesh device
 * @param tensor_rank The rank of the tensor to be sharded
 */
std::unique_ptr<ttnn::distributed::TensorToMesh> create_width_sharded_mesh_mapper(
    tt::tt_metal::distributed::MeshDevice* mesh_device, uint32_t tensor_rank) {
    // Shard on last dimension: for rank N, that's dimension N-1
    int shard_dim = static_cast<int>(tensor_rank) - 1;
    return ttnn::distributed::shard_tensor_to_mesh_mapper(*mesh_device, shard_dim);
}

/**
 * @brief Create mesh composer for aggregating width-sharded tensors
 * Concatenates shards along last dimension (width)
 *
 * @param mesh_device The mesh device
 * @param tensor_rank The rank of the tensor to be composed
 */
std::unique_ptr<ttnn::distributed::MeshToTensor> create_width_sharded_mesh_composer(
    tt::tt_metal::distributed::MeshDevice* mesh_device, uint32_t tensor_rank) {
    // Concat on last dimension: for rank N, that's dimension N-1
    int concat_dim = static_cast<int>(tensor_rank) - 1;
    return ttnn::distributed::concat_mesh_to_tensor_composer(*mesh_device, concat_dim);
}

/**
 * @brief Create mesh mapper for block-sharded distribution
 * Shards tensor on last 2 dimensions (height and width) across mesh rows and columns
 *
 * @param mesh_device The mesh device (must be 2D, e.g., 2x4)
 * @param tensor_rank The rank of the tensor to be sharded
 */
std::unique_ptr<ttnn::distributed::TensorToMesh> create_block_sharded_mesh_mapper(
    tt::tt_metal::distributed::MeshDevice* mesh_device, uint32_t tensor_rank) {
    // Block shard on last 2 dimensions:
    // - Mesh dim 0 (rows): shard on tensor's height dimension (rank-2)
    // - Mesh dim 1 (cols): shard on tensor's width dimension (rank-1)
    int height_dim = static_cast<int>(tensor_rank) - 2;
    int width_dim = static_cast<int>(tensor_rank) - 1;

    // Create MeshMapperConfig with placements for each mesh dimension
    tt::tt_metal::distributed::MeshMapperConfig config;
    config.placements = {
        tt::tt_metal::distributed::MeshMapperConfig::Shard{height_dim},  // Mesh row shards on height
        tt::tt_metal::distributed::MeshMapperConfig::Shard{width_dim}    // Mesh col shards on width
    };

    return ttnn::distributed::create_mesh_mapper(*mesh_device, config);
}

/**
 * @brief Create mesh composer for aggregating block-sharded tensors
 * Concatenates shards from 2D mesh back to full tensor
 *
 * @param mesh_device The mesh device (must be 2D, e.g., 2x4)
 * @param tensor_rank The rank of the tensor to be composed
 */
std::unique_ptr<ttnn::distributed::MeshToTensor> create_block_sharded_mesh_composer(
    tt::tt_metal::distributed::MeshDevice* mesh_device, uint32_t tensor_rank) {
    // Concat on last 2 dimensions matching the sharding
    int height_dim = static_cast<int>(tensor_rank) - 2;
    int width_dim = static_cast<int>(tensor_rank) - 1;

    // Create MeshComposerConfig with both dimensions
    tt::tt_metal::distributed::MeshComposerConfig config;
    config.dims = {height_dim, width_dim};

    return ttnn::distributed::create_mesh_composer(*mesh_device, config);
}

/**
 * @brief Create a tensor on host with random uint16 values
 */
ttnn::Tensor create_tensor_with_random_values(
    const tt::tt_metal::Shape& shape, const tt::tt_metal::TensorSpec& tensor_spec) {
    uint32_t volume = 1;
    for (size_t i = 0; i < shape.rank(); ++i) {
        volume *= shape[i];
    }

    std::vector<uint16_t> src_data(volume);

    // Generate random data
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<uint16_t> dis(0, 65535);
    for (uint32_t i = 0; i < volume; ++i) {
        src_data[i] = dis(gen);
    }

    return ttnn::Tensor::from_vector(src_data, tensor_spec);
}

/**
 * @brief Create a width-sharded tensor on 1×4 mesh
 * Global shape: (4, 16) tiles
 * Per-device: (4, 4) tiles
 */
ttnn::Tensor create_width_sharded_tensor(
    tt::tt_metal::distributed::MeshDevice* mesh_device,
    const tt::tt_metal::Shape& global_shape,   // (4, 16) in tiles -> (128, 512) in elements for tile size 32
    const tt::tt_metal::Shape& local_shape) {  // (4, 4) in tiles per device

    // Create interleaved memory config
    tt::tt_metal::MemoryConfig mem_config(tt::tt_metal::TensorMemoryLayout::INTERLEAVED, tt::tt_metal::BufferType::L1);
    tt::tt_metal::TensorSpec tensor_spec(
        global_shape,
        tt::tt_metal::TensorLayout(
            tt::tt_metal::DataType::UINT16, tt::tt_metal::PageConfig(tt::tt_metal::Layout::TILE), mem_config));

    // Create host tensor with random values
    auto host_tensor = create_tensor_with_random_values(global_shape, tensor_spec);

    // Distribute tensor using width-sharded mapper
    auto mapper = create_width_sharded_mesh_mapper(mesh_device, global_shape.rank());
    return ttnn::distributed::distribute_tensor(host_tensor, *mapper, std::ref(*mesh_device));
}

/**
 * @brief Create a block-sharded tensor on 2×4 mesh
 * Shards on both height (across mesh rows) and width (across mesh cols)
 */
ttnn::Tensor create_block_sharded_tensor(
    tt::tt_metal::distributed::MeshDevice* mesh_device,
    const tt::tt_metal::Shape& global_shape,
    const tt::tt_metal::Shape& local_shape) {
    // Create interleaved memory config
    tt::tt_metal::MemoryConfig mem_config(tt::tt_metal::TensorMemoryLayout::INTERLEAVED, tt::tt_metal::BufferType::L1);
    tt::tt_metal::TensorSpec tensor_spec(
        global_shape,
        tt::tt_metal::TensorLayout(
            tt::tt_metal::DataType::UINT16, tt::tt_metal::PageConfig(tt::tt_metal::Layout::TILE), mem_config));

    // Create host tensor with random values
    auto host_tensor = create_tensor_with_random_values(global_shape, tensor_spec);

    // Distribute tensor using block-sharded mapper
    auto mapper = create_block_sharded_mesh_mapper(mesh_device, global_shape.rank());
    return ttnn::distributed::distribute_tensor(host_tensor, *mapper, std::ref(*mesh_device));
}

/**
 * @brief Calculate maximum number of pages any gcore will process
 */
uint32_t get_max_pages_per_gcore(const tt::tt_metal::experimental::udm::GcoresInfo& gcores_info) {
    uint32_t max_pages = 0;
    for (size_t gcore_idx = 0; gcore_idx < gcores_info.gcores.size(); ++gcore_idx) {
        uint32_t total_pages = 1;
        for (size_t d = 0; d < gcores_info.dim_pages[gcore_idx].size(); ++d) {
            total_pages *= gcores_info.dim_pages[gcore_idx][d];
        }
        max_pages = std::max(max_pages, total_pages);
    }
    return max_pages;
}

/**
 * @brief Debug helper to log tensor shape info
 */
void log_tensor_shape_info(
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
void log_gcores_info(
    const tt::tt_metal::experimental::udm::GcoresInfo& gcores_info,
    const tt::tt_metal::experimental::udm::MeshBuilder& mesh_builder) {
    const auto& mesh_shape = mesh_builder.get_mesh().shape();
    const auto& grid_shape = mesh_builder.get_flattened_grid();

    log_info(tt::LogTest, "=== Gcores Info ===");
    log_info(
        tt::LogTest,
        "Mesh shape: [{}x{}], Grid shape: [{}x{}]",
        mesh_shape[0],
        mesh_shape[1],
        grid_shape[0],
        grid_shape[1]);
    log_info(tt::LogTest, "Total gcores in result: {}", gcores_info.gcores.size());
    log_info(tt::LogTest, "Gcores with work: {}", gcores_info.num_cores);

    // Build partition dims string
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

        // Calculate total pages for this gcore
        uint32_t total_pages = 1;
        for (size_t d = 0; d < gcores_info.dim_pages[i].size(); ++d) {
            total_pages *= gcores_info.dim_pages[i][d];
        }

        // Only log gcores with work to reduce noise
        if (total_pages > 0) {
            log_info(
                tt::LogTest,
                "Gcore[{}]: local_id={}, global_id={}, local_coord={}, global_coord={}",
                i,
                gcore.local_id,
                gcore.global_id,
                gcore.local_coord,
                gcore.global_coord);
            log_info(tt::LogTest, "  Total pages: {}", total_pages);

            // Decode which mesh device this gcore belongs to
            // global_coord is in flattened mesh space where dims = mesh * grid
            if (gcore.global_coord.dims() >= 2) {
                uint32_t mesh_row = gcore.global_coord[0] / grid_shape[0];
                uint32_t mesh_col = gcore.global_coord[1] / grid_shape[1];
                uint32_t grid_row = gcore.global_coord[0] % grid_shape[0];
                uint32_t grid_col = gcore.global_coord[1] % grid_shape[1];
                log_info(
                    tt::LogTest,
                    "  → Mesh device: ({}, {}), Grid core: ({}, {})",
                    mesh_row,
                    mesh_col,
                    grid_row,
                    grid_col);
            }

            // Build dim pages string
            std::string pages_str;
            for (size_t d = 0; d < gcores_info.dim_pages[i].size(); ++d) {
                if (d > 0) {
                    pages_str += ", ";
                }
                pages_str += std::to_string(gcores_info.dim_pages[i][d]);
            }
            log_info(tt::LogTest, "  Dim pages: [{}]", pages_str);

            // Build dim offsets string
            std::string offsets_str;
            for (size_t d = 0; d < gcores_info.dim_offsets[i].size(); ++d) {
                if (d > 0) {
                    offsets_str += ", ";
                }
                offsets_str += std::to_string(gcores_info.dim_offsets[i][d]);
            }
            log_info(tt::LogTest, "  Dim offsets: [{}]", offsets_str);

            // Build dim strides string
            std::string strides_str;
            for (size_t d = 0; d < gcores_info.dim_strides[i].size(); ++d) {
                if (d > 0) {
                    strides_str += ", ";
                }
                strides_str += std::to_string(gcores_info.dim_strides[i][d]);
            }
            log_info(tt::LogTest, "  Dim strides: [{}]", strides_str);
        }
    }
    log_info(tt::LogTest, "==================");
}

/**
 * @brief Create UDM program that copies tensor from input to output
 */
tt::tt_metal::experimental::udm::MeshTensorBuilder create_tensor_builders(const ttnn::Tensor& tensor) {
    // 1. Extract MeshBuffer from the distributed tensor
    TT_ASSERT(std::holds_alternative<tt::tt_metal::DeviceStorage>(tensor.storage()), "Tensor must be on device");
    const auto& device_storage = std::get<tt::tt_metal::DeviceStorage>(tensor.storage());
    TT_ASSERT(device_storage.mesh_buffer != nullptr, "Tensor must have a MeshBuffer");

    // 2. Extract distribution info from tensor topology
    const auto& topology = tensor.tensor_topology();
    const auto& distribution_shape = topology.distribution_shape();
    const auto& placements = topology.placements();

    // 3. Compute tensor shape in pages from tensor layout
    // After distribute_tensor, tensor.padded_shape() is already the local shape per device
    const auto& tensor_shape = tensor.padded_shape();

    const auto& tensor_layout = tensor.tensor_spec().tensor_layout();
    auto tensor_shape_in_pages = compute_tensor_shape_in_pages(tensor_shape, tensor_layout);

    // 5. Convert placements to shard_dims: nullopt for replicate, value for shard dimension
    std::vector<std::optional<int>> shard_dims;
    for (const auto& placement : placements) {
        if (std::holds_alternative<ttnn::distributed::MeshMapperConfig::Replicate>(placement)) {
            shard_dims.push_back(std::nullopt);
        } else {
            const auto& shard = std::get<ttnn::distributed::MeshMapperConfig::Shard>(placement);
            shard_dims.push_back(shard.dim);
        }
    }

    // 6. Create MeshTensorBuilder with local shape in pages
    auto mesh_tensor_builder = tt::tt_metal::experimental::udm::MeshTensorBuilder(
        *device_storage.mesh_buffer,
        tensor_shape_in_pages,  // Pass shape in pages (computed from tensor layout)
        distribution_shape,
        shard_dims);

    return mesh_tensor_builder;
}

tt::tt_metal::experimental::udm::MeshProgram create_program(
    const ttnn::Tensor& input_tensor, const ttnn::Tensor& output_tensor) {
    auto input_mesh_tensor_builder = create_tensor_builders(input_tensor);
    auto output_mesh_tensor_builder = create_tensor_builders(output_tensor);

    // Use the mesh_builder from MeshTensorBuilder (they're identical since both created from same mesh_buffer)
    auto& mesh_builder = input_mesh_tensor_builder.mesh_builder();

    // Create MeshProgram
    auto program = tt::tt_metal::experimental::udm::CreateMeshProgram(mesh_builder);

    // Log tensor shape info for debugging
    log_tensor_shape_info(input_mesh_tensor_builder, input_tensor);

    // Map buffer to gcores using UDM API
    // Partition work on dimension 0 (rows) - each worker processes 1 row
    // Data is width-sharded (dim 1), so each row spans multiple devices
    int partition_dim = 0;
    auto gcores_info = tt::tt_metal::experimental::udm::map_tensor_to_gcores(
        input_mesh_tensor_builder,
        mesh_builder,  // Pass mesh_builder which contains mesh and grid dimensions
        partition_dim  // partition_dim = 0 (rows)
    );

    // Log gcores info for debugging
    log_gcores_info(gcores_info, mesh_builder);

    // Get compile-time args from both input and output MeshTensorBuilders
    auto input_compile_time_args = input_mesh_tensor_builder.get_compile_time_args();
    auto output_compile_time_args = output_mesh_tensor_builder.get_compile_time_args();

    // Combine compile-time args: input args first, then output args
    // TODO: once we can allow two risc work in parallel, seperate them into two kernels
    std::vector<uint32_t> compile_time_args = input_compile_time_args;
    compile_time_args.insert(compile_time_args.end(), output_compile_time_args.begin(), output_compile_time_args.end());

    // Create mesh circular buffer for tile storage
    // Get data format and compute tile size (following pattern from other ops)
    tt::DataFormat data_format = tt::tt_metal::datatype_to_dataformat_converter(input_tensor.dtype());
    uint32_t tile_size = tt::tile_size(data_format);
    constexpr uint32_t cb_id = 0;
    tt::tt_metal::CircularBufferConfig cb_config =
        tt::tt_metal::CircularBufferConfig(2 * tile_size, {{cb_id, data_format}}).set_page_size(cb_id, tile_size);

    // Create CB on ALL gcores in mesh to ensure identical L1 layout across devices
    auto mesh_cb_handle = tt::tt_metal::experimental::udm::CreateMeshCircularBuffer(mesh_builder, program, cb_config);

    // Create kernel on all mapped gcores
    tt::tt_metal::experimental::udm::MeshKernelHandle kernel_id = tt::tt_metal::experimental::udm::CreateMeshKernel(
        mesh_builder,
        program,
        "tests/tt_metal/udm/kernels/copy.cpp",
        gcores_info.gcores,
        tt::tt_metal::DataMovementConfig{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
            .noc = tt::tt_metal::NOC::RISCV_0_default,
            .compile_args = compile_time_args,
        });

    // Set runtime args for each gcore
    // Since input and output tensors have identical shape and sharding, they share the same runtime args
    uint32_t gcore_idx = 0;
    for (const auto& gcore : gcores_info.gcores) {
        // Build runtime args: rank, then for each dim: (pages, offset, stride)
        // gcores_info contains ALL dimensions (partitioned and non-partitioned)
        uint32_t rank = gcores_info.dim_pages[gcore_idx].size();

        std::vector<uint32_t> runtime_args;
        runtime_args.push_back(rank);

        for (uint32_t d = 0; d < rank; ++d) {
            runtime_args.push_back(gcores_info.dim_pages[gcore_idx][d]);
            runtime_args.push_back(gcores_info.dim_offsets[gcore_idx][d]);
            runtime_args.push_back(gcores_info.dim_strides[gcore_idx][d]);
        }

        tt::tt_metal::experimental::udm::SetMeshKernelRuntimeArgs(
            mesh_builder, program, kernel_id, gcore, runtime_args);
        gcore_idx++;
    }

    return program;
}

/**
 * @brief Run a mesh program on all mesh coordinates
 *
 * Dispatches each device separately to allow different program structures per device.
 * Skips devices that don't have any kernels registered.
 */
void run_program(
    const ttnn::Tensor& sharded_tensor,
    tt::tt_metal::distributed::MeshDevice* mesh_device,
    tt::tt_metal::experimental::udm::MeshProgram& mesh_program) {
    // Get the mesh shape from the mesh device, not the tensor topology
    // (they may differ if tensor is replicated on some dimensions)
    const auto& mesh_shape = mesh_device->shape();

    // Create a mesh coordinate range that covers all coordinates in the mesh
    auto mesh_coord_range = tt::tt_metal::distributed::MeshCoordinateRange(mesh_shape);

    // Dispatch each device separately
    for (const auto& coord : mesh_coord_range) {
        // Skip programs that don't have any kernels
        if (!mesh_program.has_kernel(coord)) {
            continue;
        }

        auto mesh_workload = tt::tt_metal::distributed::MeshWorkload();
        mesh_workload.add_program(
            tt::tt_metal::distributed::MeshCoordinateRange(coord), std::move(mesh_program.program_at(coord)));

        tt::tt_metal::distributed::EnqueueMeshWorkload(mesh_device->mesh_command_queue(), mesh_workload, false);
    }

    // Wait for all dispatched programs to complete
    tt::tt_metal::distributed::Finish(mesh_device->mesh_command_queue());
}

/**
 * @brief Validate that output tensor matches input tensor
 *
 * Reads both distributed tensors from device and compares their values.
 * Uses aggregate_tensor to gather distributed tensors back to replicated form.
 */
void validate(
    const ttnn::Tensor& input_tensor,
    const ttnn::Tensor& output_tensor,
    ShardStrategy shard_strategy = ShardStrategy::WIDTH) {
    // Get mesh device
    auto* mesh_device = input_tensor.device();

    // Create appropriate composer based on sharding strategy
    std::unique_ptr<ttnn::distributed::MeshToTensor> composer;
    switch (shard_strategy) {
        case ShardStrategy::WIDTH:
            composer = create_width_sharded_mesh_composer(mesh_device, input_tensor.padded_shape().rank());
            break;
        case ShardStrategy::BLOCK:
            composer = create_block_sharded_mesh_composer(mesh_device, input_tensor.padded_shape().rank());
            break;
        case ShardStrategy::HEIGHT:
            // Height sharding not yet implemented
            TT_THROW("HEIGHT sharding strategy not yet implemented");
            break;
    }

    // Aggregate tensors and convert to vectors
    auto input_data = ttnn::distributed::aggregate_tensor(input_tensor, *composer).to_vector<uint16_t>();
    auto output_data = ttnn::distributed::aggregate_tensor(output_tensor, *composer).to_vector<uint16_t>();

    // Compare values
    uint32_t volume = input_data.size();
    uint32_t mismatches = 0;
    const uint32_t max_print_mismatches = 10;

    for (uint32_t i = 0; i < volume; ++i) {
        if (input_data[i] != output_data[i]) {
            if (mismatches < max_print_mismatches) {
                log_error(tt::LogTest, "Mismatch at index {}: input={}, output={}", i, input_data[i], output_data[i]);
            }
            mismatches++;
        }
    }

    if (mismatches > 0) {
        log_error(tt::LogTest, "Total mismatches: {} / {}", mismatches, volume);
        TT_THROW("Validation failed: output tensor does not match input tensor");
    }

    log_info(tt::LogTest, "Validation passed: all {} values match", volume);
}

/**
 * @brief Helper function to run UDM copy test with given tensor shapes (width-sharded)
 *
 * @param mesh_device The mesh device to use
 * @param global_shape Global tensor shape across all devices
 * @param local_shape Local tensor shape per device
 */
void run_udm_copy_test(
    tt::tt_metal::distributed::MeshDevice* mesh_device,
    const tt::tt_metal::Shape& global_shape,
    const tt::tt_metal::Shape& local_shape) {
    auto input_tensor = create_width_sharded_tensor(mesh_device, global_shape, local_shape);

    // Create output tensor with same shape and sharding strategy as input
    auto output_tensor = create_width_sharded_tensor(mesh_device, global_shape, local_shape);

    // Create program
    auto program = create_program(input_tensor, output_tensor);

    // Run program
    auto* tensor_mesh_device = input_tensor.device();
    ASSERT_NE(tensor_mesh_device, nullptr) << "Tensor must be on device";
    run_program(input_tensor, tensor_mesh_device, program);

    // Validate output matches input
    validate(input_tensor, output_tensor, ShardStrategy::WIDTH);
}

/**
 * @brief Helper function to run UDM copy test with block-sharded tensors
 *
 * @param mesh_device The mesh device to use (must be 2D)
 * @param global_shape Global tensor shape across all devices
 * @param local_shape Local tensor shape per device
 */
void run_udm_copy_test_block_sharded(
    tt::tt_metal::distributed::MeshDevice* mesh_device,
    const tt::tt_metal::Shape& global_shape,
    const tt::tt_metal::Shape& local_shape) {
    auto input_tensor = create_block_sharded_tensor(mesh_device, global_shape, local_shape);

    // Create output tensor with same shape and sharding strategy as input
    auto output_tensor = create_block_sharded_tensor(mesh_device, global_shape, local_shape);

    // Create program
    auto program = create_program(input_tensor, output_tensor);

    // Run program
    auto* tensor_mesh_device = input_tensor.device();
    ASSERT_NE(tensor_mesh_device, nullptr) << "Tensor must be on device";
    run_program(input_tensor, tensor_mesh_device, program);

    // Validate output matches input
    validate(input_tensor, output_tensor, ShardStrategy::BLOCK);
}

/**
 * @brief Test UDM program with width-sharded tensor
 *
 * Setup:
 * - Mesh: 1×4 (4 devices in a row)
 * - Global tensor: (4, 16) tiles, width-sharded across devices
 * - Per-device tensor: (4, 4) tiles, interleaved
 * - Grid: 1×16 workers per device (flattened from 4×4 device grid)
 * - Block: 1×4 (same as mesh)
 *
 * Operation:
 * - Each worker reads one local row (4 tiles)
 * - Copies to output tensor
 * - Writes back
 *
 * Worker assignment:
 * - 4 rows per device → 4 workers per device
 * - Total: 16 gcores (4 per device × 4 devices)
 * - Gcore 0-3: device 0, rows 0-3
 * - Gcore 4-7: device 1, rows 0-3
 * - Gcore 8-11: device 2, rows 0-3
 * - Gcore 12-15: device 3, rows 0-3
 */
using MeshDevice1x4Fabric2DUDMFixture = tt::tt_metal::MeshDevice1x4Fabric2DUDMFixture;

TEST_F(MeshDevice1x4Fabric2DUDMFixture, TestMeshWidthShardedCopy2D_Small) {
    // Small 2D tensor: (4, 16) tiles = (128, 512) elements
    tt::tt_metal::Shape global_shape({128, 512});  // (4, 16) tiles in element count
    tt::tt_metal::Shape local_shape({128, 128});   // (4, 4) tiles per device

    run_udm_copy_test(mesh_device_.get(), global_shape, local_shape);
}

TEST_F(MeshDevice1x4Fabric2DUDMFixture, TestMeshWidthShardedCopy2D_Large) {
    // Larger 2D tensor: (32, 64) tiles = (1024, 2048) elements
    tt::tt_metal::Shape global_shape({1024, 2048});  // (32, 64) tiles in element count
    tt::tt_metal::Shape local_shape({1024, 512});    // (32, 16) tiles per device

    run_udm_copy_test(mesh_device_.get(), global_shape, local_shape);
}

TEST_F(MeshDevice1x4Fabric2DUDMFixture, TestMeshWidthShardedCopy3D) {
    // 3D tensor: (2, 16, 32) tiles = (2, 512, 8192) elements
    // Sharded along last dimension (width)
    tt::tt_metal::Shape global_shape({2, 512, 8192});  // (2, 16, 256) tiles
    tt::tt_metal::Shape local_shape({2, 512, 2048});   // (2, 16, 64) tiles per device

    run_udm_copy_test(mesh_device_.get(), global_shape, local_shape);
}

TEST_F(MeshDevice1x4Fabric2DUDMFixture, TestMeshWidthShardedCopy4D) {
    // 4D tensor: (2, 4, 8, 16) tiles = (2, 4, 256, 512) elements
    // Sharded along last dimension (width)
    tt::tt_metal::Shape global_shape({2, 4, 256, 8192});  // (2, 4, 8, 256) tiles
    tt::tt_metal::Shape local_shape({2, 4, 256, 2048});   // (2, 4, 8, 64) tiles per device

    run_udm_copy_test(mesh_device_.get(), global_shape, local_shape);
}

// ============================================================================
// Block-Sharded Tests (2x4 Mesh)
// ============================================================================

using MeshDevice2x4Fabric2DUDMFixture = tt::tt_metal::MeshDevice2x4Fabric2DUDMFixture;

TEST_F(MeshDevice2x4Fabric2DUDMFixture, TestMeshBlockShardedCopy2D_Small) {
    // Small 2D tensor: (8, 16) tiles = (256, 512) elements
    // Block-sharded: height across 2 mesh rows, width across 4 mesh cols
    // Per-device: (4, 4) tiles = (128, 128) elements
    tt::tt_metal::Shape global_shape({256, 512});  // (8, 16) tiles
    tt::tt_metal::Shape local_shape({128, 128});   // (4, 4) tiles per device

    run_udm_copy_test_block_sharded(mesh_device_.get(), global_shape, local_shape);
}

TEST_F(MeshDevice2x4Fabric2DUDMFixture, TestMeshBlockShardedCopy2D_Large) {
    // Larger 2D tensor: (64, 128) tiles = (2048, 4096) elements
    // Block-sharded: height across 2 mesh rows, width across 4 mesh cols
    // Per-device: (32, 32) tiles = (1024, 1024) elements
    tt::tt_metal::Shape global_shape({2048, 4096});  // (64, 128) tiles
    tt::tt_metal::Shape local_shape({1024, 1024});   // (32, 32) tiles per device

    run_udm_copy_test_block_sharded(mesh_device_.get(), global_shape, local_shape);
}

TEST_F(MeshDevice2x4Fabric2DUDMFixture, TestMeshBlockShardedCopy3D) {
    // 3D tensor: (2, 16, 32) tiles = (2, 512, 1024) elements
    // Block-sharded on last 2 dims: height across 2 mesh rows, width across 4 mesh cols
    // Per-device: (2, 8, 8) tiles = (2, 256, 256) elements
    tt::tt_metal::Shape global_shape({2, 512, 1024});  // (2, 16, 32) tiles
    tt::tt_metal::Shape local_shape({2, 256, 256});    // (2, 8, 8) tiles per device

    run_udm_copy_test_block_sharded(mesh_device_.get(), global_shape, local_shape);
}

TEST_F(MeshDevice2x4Fabric2DUDMFixture, TestMeshBlockShardedCopy4D) {
    // 4D tensor: (2, 4, 16, 32) tiles = (2, 4, 512, 1024) elements
    // Block-sharded on last 2 dims: height across 2 mesh rows, width across 4 mesh cols
    // Per-device: (2, 4, 8, 8) tiles = (2, 4, 256, 256) elements
    tt::tt_metal::Shape global_shape({2, 4, 512, 1024});  // (2, 4, 16, 32) tiles
    tt::tt_metal::Shape local_shape({2, 4, 256, 256});    // (2, 4, 8, 8) tiles per device

    run_udm_copy_test_block_sharded(mesh_device_.get(), global_shape, local_shape);
}

}  // namespace tt::tt_metal::experimental::udm_tests
