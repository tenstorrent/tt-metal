// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include "tests/tt_metal/tt_metal/common/multi_device_fixture.hpp"
#include "tests/tt_metal/udm/test_udm_utils.hpp"

#include "tt_metal/udm/mesh_kernel.hpp"
#include "tt_metal/udm/mesh_utils.hpp"
#include "tt_metal/udm/mesh_circular_buffer.hpp"

namespace tt::tt_metal::experimental::udm_tests {

/**
 * @brief Create UDM program that copies tensor from input to output
 */
tt::tt_metal::experimental::udm::MeshProgram create_program(
    const ttnn::Tensor& input_tensor, const ttnn::Tensor& output_tensor) {
    auto input_mesh_tensor_builder = create_tensor_builder(input_tensor);
    auto output_mesh_tensor_builder = create_tensor_builder(output_tensor);

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
 * @brief Helper function to run UDM copy test with given tensor shapes (width-sharded)
 */
void run_udm_copy_test(
    tt::tt_metal::distributed::MeshDevice* mesh_device,
    const tt::tt_metal::Shape& global_shape,
    const tt::tt_metal::Shape& local_shape,
    const ShardStrategy& shard_strategy) {
    // Create tensors based on sharding strategy
    ttnn::Tensor input_tensor;
    ttnn::Tensor output_tensor;

    switch (shard_strategy) {
        case ShardStrategy::WIDTH:
            input_tensor = create_width_sharded_tensor(mesh_device, global_shape, local_shape);
            output_tensor = create_width_sharded_tensor(mesh_device, global_shape, local_shape);
            break;
        case ShardStrategy::BLOCK:
            input_tensor = create_block_sharded_tensor(mesh_device, global_shape, local_shape);
            output_tensor = create_block_sharded_tensor(mesh_device, global_shape, local_shape);
            break;
        case ShardStrategy::HEIGHT: TT_THROW("HEIGHT sharding strategy not yet implemented"); break;
    }

    // Create program
    auto program = create_program(input_tensor, output_tensor);

    // Run program
    auto* tensor_mesh_device = input_tensor.device();
    ASSERT_NE(tensor_mesh_device, nullptr) << "Tensor must be on device";
    run_program(input_tensor, tensor_mesh_device, program);

    // Validate output matches input
    validate(input_tensor, output_tensor, shard_strategy);
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

    run_udm_copy_test(mesh_device_.get(), global_shape, local_shape, ShardStrategy::WIDTH);
}

TEST_F(MeshDevice1x4Fabric2DUDMFixture, TestMeshWidthShardedCopy2D_Large) {
    // Larger 2D tensor: (32, 64) tiles = (1024, 2048) elements
    tt::tt_metal::Shape global_shape({1024, 2048});  // (32, 64) tiles in element count
    tt::tt_metal::Shape local_shape({1024, 512});    // (32, 16) tiles per device

    run_udm_copy_test(mesh_device_.get(), global_shape, local_shape, ShardStrategy::WIDTH);
}

TEST_F(MeshDevice1x4Fabric2DUDMFixture, TestMeshWidthShardedCopy3D) {
    // 3D tensor: (2, 16, 32) tiles = (2, 512, 8192) elements
    // Sharded along last dimension (width)
    tt::tt_metal::Shape global_shape({2, 512, 8192});  // (2, 16, 256) tiles
    tt::tt_metal::Shape local_shape({2, 512, 2048});   // (2, 16, 64) tiles per device

    run_udm_copy_test(mesh_device_.get(), global_shape, local_shape, ShardStrategy::WIDTH);
}

TEST_F(MeshDevice1x4Fabric2DUDMFixture, TestMeshWidthShardedCopy4D) {
    // 4D tensor: (2, 4, 8, 16) tiles = (2, 4, 256, 512) elements
    // Sharded along last dimension (width)
    tt::tt_metal::Shape global_shape({2, 4, 256, 8192});  // (2, 4, 8, 256) tiles
    tt::tt_metal::Shape local_shape({2, 4, 256, 2048});   // (2, 4, 8, 64) tiles per device

    run_udm_copy_test(mesh_device_.get(), global_shape, local_shape, ShardStrategy::WIDTH);
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

    run_udm_copy_test(mesh_device_.get(), global_shape, local_shape, ShardStrategy::BLOCK);
}

TEST_F(MeshDevice2x4Fabric2DUDMFixture, TestMeshBlockShardedCopy2D_Large) {
    // Larger 2D tensor: (64, 128) tiles = (2048, 4096) elements
    // Block-sharded: height across 2 mesh rows, width across 4 mesh cols
    // Per-device: (32, 32) tiles = (1024, 1024) elements
    tt::tt_metal::Shape global_shape({2048, 4096});  // (64, 128) tiles
    tt::tt_metal::Shape local_shape({1024, 1024});   // (32, 32) tiles per device

    run_udm_copy_test(mesh_device_.get(), global_shape, local_shape, ShardStrategy::BLOCK);
}

TEST_F(MeshDevice2x4Fabric2DUDMFixture, TestMeshBlockShardedCopy3D) {
    // 3D tensor: (2, 16, 32) tiles = (2, 512, 1024) elements
    // Block-sharded on last 2 dims: height across 2 mesh rows, width across 4 mesh cols
    // Per-device: (2, 8, 8) tiles = (2, 256, 256) elements
    tt::tt_metal::Shape global_shape({2, 512, 1024});  // (2, 16, 32) tiles
    tt::tt_metal::Shape local_shape({2, 256, 256});    // (2, 8, 8) tiles per device

    run_udm_copy_test(mesh_device_.get(), global_shape, local_shape, ShardStrategy::BLOCK);
}

TEST_F(MeshDevice2x4Fabric2DUDMFixture, TestMeshBlockShardedCopy4D) {
    // 4D tensor: (2, 4, 16, 32) tiles = (2, 4, 512, 1024) elements
    // Block-sharded on last 2 dims: height across 2 mesh rows, width across 4 mesh cols
    // Per-device: (2, 4, 8, 8) tiles = (2, 4, 256, 256) elements
    tt::tt_metal::Shape global_shape({2, 4, 512, 1024});  // (2, 4, 16, 32) tiles
    tt::tt_metal::Shape local_shape({2, 4, 256, 256});    // (2, 4, 8, 8) tiles per device

    run_udm_copy_test(mesh_device_.get(), global_shape, local_shape, ShardStrategy::BLOCK);
}

}  // namespace tt::tt_metal::experimental::udm_tests
