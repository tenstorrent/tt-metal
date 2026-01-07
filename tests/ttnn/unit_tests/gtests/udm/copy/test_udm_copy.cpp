// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include "tests/tt_metal/tt_metal/common/multi_device_fixture.hpp"
#include "tests/ttnn/unit_tests/gtests/udm/test_udm_utils.hpp"

#include "tt_metal/experimental/udm/mesh_kernel.hpp"
#include "tt_metal/experimental/udm/mesh_utils.hpp"
#include "tt_metal/experimental/udm/mesh_circular_buffer.hpp"

namespace tt::tt_metal::experimental::udm_tests {
namespace {

/**
 * @brief Create UDM program that copies tensor from input to output
 *
 * @param input_mesh_tensor_builder Builder for input tensor (contains mesh tensor shape info)
 * @param output_mesh_tensor_builder Builder for output tensor
 */
tt::tt_metal::experimental::udm::MeshProgram create_program(
    tt::tt_metal::experimental::udm::MeshTensorBuilder& input_mesh_tensor_builder,
    tt::tt_metal::experimental::udm::MeshTensorBuilder& output_mesh_tensor_builder) {
    // Use bfloat16 data format for circular buffer configuration
    tt::DataFormat data_format = tt::DataFormat::Float16_b;
    // Use the mesh_builder from MeshTensorBuilder (they're identical since both created from same mesh_buffer)
    auto& mesh_builder = input_mesh_tensor_builder.mesh_builder();

    // Create MeshProgram
    auto program = tt::tt_metal::experimental::udm::CreateMeshProgram(mesh_builder);

    // Map buffer to gcores using UDM API
    // Partition work on all non-last dimensions (0, 1, ..., rank-2)
    // Last dimension (width) is not partitioned as it's the innermost loop in the kernel
    const auto& mesh_tensor_shape = input_mesh_tensor_builder.get_mesh_tensor_shape_in_pages();
    uint32_t rank = mesh_tensor_shape.rank();
    std::vector<int> partition_dims;
    partition_dims.reserve(rank - 1);
    for (uint32_t d = 0; d < rank - 1; ++d) {
        partition_dims.push_back(static_cast<int>(d));
    }
    auto gcores_info = tt::tt_metal::experimental::udm::map_tensor_to_gcores_nd(
        input_mesh_tensor_builder,
        mesh_builder,   // Pass mesh_builder which contains mesh and grid dimensions
        partition_dims  // partition on all non-last dimensions
    );

    // Log gcores info for debugging
    log_gcores_info(gcores_info, mesh_builder);

    // Get compile-time args from both input and output MeshTensorBuilders
    auto input_compile_time_args = input_mesh_tensor_builder.get_compile_time_args();
    auto output_compile_time_args = output_mesh_tensor_builder.get_compile_time_args();

    // Combine compile-time args: input args first, then output args
    // TODO(#34704): once we can allow two risc work in parallel, seperate them into two kernels
    std::vector<uint32_t> compile_time_args = input_compile_time_args;
    compile_time_args.insert(compile_time_args.end(), output_compile_time_args.begin(), output_compile_time_args.end());

    // Create mesh circular buffer for tile storage
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
        "tests/ttnn/unit_tests/gtests/udm/copy/kernels/copy.cpp",
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
 * @brief Validate that output tensor matches expected values
 */
inline void validate(
    const ttnn::Tensor& expected_tensor,
    const ttnn::Tensor& actual_tensor,
    ShardStrategy shard_strategy = ShardStrategy::WIDTH,
    ShardOrder shard_order = ShardOrder::NORMAL) {
    auto* mesh_device = expected_tensor.device();
    bool swap_shard_order = (shard_order == ShardOrder::SWAPPED);

    // Create appropriate composer based on sharding strategy
    std::unique_ptr<ttnn::distributed::MeshToTensor> composer;
    switch (shard_strategy) {
        case ShardStrategy::WIDTH:
            composer = create_width_sharded_mesh_composer(
                mesh_device, expected_tensor.padded_shape().rank(), swap_shard_order);
            break;
        case ShardStrategy::BLOCK:
            composer = create_block_sharded_mesh_composer(
                mesh_device, expected_tensor.padded_shape().rank(), swap_shard_order);
            break;
        case ShardStrategy::HEIGHT:
            composer = create_height_sharded_mesh_composer(
                mesh_device, expected_tensor.padded_shape().rank(), swap_shard_order);
            break;
    }

    // Aggregate tensors and convert to vectors
    auto expected_data = ttnn::distributed::aggregate_tensor(expected_tensor, *composer).to_vector<bfloat16>();
    auto actual_data = ttnn::distributed::aggregate_tensor(actual_tensor, *composer).to_vector<bfloat16>();

    // Compare values
    uint32_t volume = expected_data.size();
    uint32_t mismatches = 0;
    const uint32_t max_print_mismatches = 10;

    for (uint32_t i = 0; i < volume; ++i) {
        if (expected_data[i] != actual_data[i]) {
            if (mismatches < max_print_mismatches) {
                log_error(
                    tt::LogTest,
                    "Mismatch at index {}: expected={}, actual={}",
                    i,
                    static_cast<float>(expected_data[i]),
                    static_cast<float>(actual_data[i]));
            }
            mismatches++;
        }
    }

    if (mismatches > 0) {
        log_error(tt::LogTest, "Total mismatches: {} / {}", mismatches, volume);
        TT_THROW("Validation failed: output tensor does not match expected tensor");
    }

    log_info(tt::LogTest, "Validation passed: all {} values match", volume);
}

/**
 * @brief Helper function to run UDM copy test with given tensor shapes
 */
void run_udm_copy_test(
    tt::tt_metal::distributed::MeshDevice* mesh_device,
    const tt::tt_metal::Shape& global_shape,
    const tt::tt_metal::Shape& local_shape,
    const ShardStrategy& shard_strategy,
    const ShardOrder& shard_order = ShardOrder::NORMAL) {
    // Create tensors based on sharding strategy
    ttnn::Tensor input_tensor;
    ttnn::Tensor output_tensor;
    bool swap_shard_order = (shard_order == ShardOrder::SWAPPED);

    switch (shard_strategy) {
        case ShardStrategy::WIDTH:
            input_tensor =
                create_width_distributed_interleaved_bfloat16_tensor(mesh_device, global_shape, swap_shard_order);
            output_tensor =
                create_width_distributed_interleaved_bfloat16_tensor(mesh_device, global_shape, swap_shard_order);
            break;
        case ShardStrategy::BLOCK:
            input_tensor =
                create_block_distributed_interleaved_bfloat16_tensor(mesh_device, global_shape, swap_shard_order);
            output_tensor =
                create_block_distributed_interleaved_bfloat16_tensor(mesh_device, global_shape, swap_shard_order);
            break;
        case ShardStrategy::HEIGHT:
            input_tensor =
                create_height_distributed_interleaved_bfloat16_tensor(mesh_device, global_shape, swap_shard_order);
            output_tensor =
                create_height_distributed_interleaved_bfloat16_tensor(mesh_device, global_shape, swap_shard_order);
            break;
    }

    // Build tensor builders from tensors (extracts mesh tensor shape info)
    auto input_mesh_tensor_builder = create_tensor_builder(input_tensor);
    auto output_mesh_tensor_builder = create_tensor_builder(output_tensor);

    // Create program using tensor builders (not raw tensors)
    auto program = create_program(input_mesh_tensor_builder, output_mesh_tensor_builder);

    // Run program
    auto* tensor_mesh_device = input_tensor.device();
    ASSERT_NE(tensor_mesh_device, nullptr) << "Tensor must be on device";
    run_program(input_tensor, tensor_mesh_device, program);

    // Validate output matches input
    validate(input_tensor, output_tensor, shard_strategy, shard_order);
}

}  // namespace

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
 * - GlobalCore 0-3: device 0, rows 0-3
 * - GlobalCore 4-7: device 1, rows 0-3
 * - GlobalCore 8-11: device 2, rows 0-3
 * - GlobalCore 12-15: device 3, rows 0-3
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

// ============================================================================
// Block-Sharded Tests with Swapped Order (2x4 Mesh)
// Swapped distribution: mesh dim 0 shards tensor width, mesh dim 1 shards tensor height
// This is the inverse of the normal block sharding pattern
// ============================================================================

TEST_F(MeshDevice2x4Fabric2DUDMFixture, TestMeshBlockShardedSwappedCopy2D_Small) {
    // Small 2D tensor: (16, 8) tiles = (512, 256) elements
    // Block-sharded with swapped order: width across 2 mesh rows, height across 4 mesh cols
    // Per-device: (4, 4) tiles = (128, 128) elements
    tt::tt_metal::Shape global_shape({512, 256});  // (16, 8) tiles
    tt::tt_metal::Shape local_shape({128, 128});   // (4, 4) tiles per device

    run_udm_copy_test(mesh_device_.get(), global_shape, local_shape, ShardStrategy::BLOCK, ShardOrder::SWAPPED);
}

TEST_F(MeshDevice2x4Fabric2DUDMFixture, TestMeshBlockShardedSwappedCopy2D_Large) {
    // Larger 2D tensor: (128, 64) tiles = (4096, 2048) elements
    // Block-sharded with swapped order: width across 2 mesh rows, height across 4 mesh cols
    // Per-device: (32, 32) tiles = (1024, 1024) elements
    tt::tt_metal::Shape global_shape({4096, 2048});  // (128, 64) tiles
    tt::tt_metal::Shape local_shape({1024, 1024});   // (32, 32) tiles per device

    run_udm_copy_test(mesh_device_.get(), global_shape, local_shape, ShardStrategy::BLOCK, ShardOrder::SWAPPED);
}

TEST_F(MeshDevice2x4Fabric2DUDMFixture, TestMeshBlockShardedSwappedCopy3D) {
    // 3D tensor: (2, 32, 16) tiles = (2, 1024, 512) elements
    // Block-sharded with swapped order on last 2 dims: width across 2 mesh rows, height across 4 mesh cols
    // Per-device: (2, 8, 8) tiles = (2, 256, 256) elements
    tt::tt_metal::Shape global_shape({2, 1024, 512});  // (2, 32, 16) tiles
    tt::tt_metal::Shape local_shape({2, 256, 256});    // (2, 8, 8) tiles per device

    run_udm_copy_test(mesh_device_.get(), global_shape, local_shape, ShardStrategy::BLOCK, ShardOrder::SWAPPED);
}

TEST_F(MeshDevice2x4Fabric2DUDMFixture, TestMeshBlockShardedSwappedCopy4D) {
    // 4D tensor: (2, 4, 32, 16) tiles = (2, 4, 1024, 512) elements
    // Block-sharded with swapped order on last 2 dims: width across 2 mesh rows, height across 4 mesh cols
    // Per-device: (2, 4, 8, 8) tiles = (2, 4, 256, 256) elements
    tt::tt_metal::Shape global_shape({2, 4, 1024, 512});  // (2, 4, 32, 16) tiles
    tt::tt_metal::Shape local_shape({2, 4, 256, 256});    // (2, 4, 8, 8) tiles per device

    run_udm_copy_test(mesh_device_.get(), global_shape, local_shape, ShardStrategy::BLOCK, ShardOrder::SWAPPED);
}

// ============================================================================
// Height-Sharded Tests with Normal Order (2x4 Mesh)
// Normal: shard height on mesh dim 0 (2 rows), replicate on mesh dim 1 (4 cols)
// ============================================================================

TEST_F(MeshDevice2x4Fabric2DUDMFixture, TestMeshHeightShardedCopy2D) {
    // 2D tensor: (8, 16) tiles = (256, 512) elements
    // Height-sharded: height across 2 mesh rows, replicated across 4 mesh cols
    // Per-device: (4, 16) tiles = (128, 512) elements
    tt::tt_metal::Shape global_shape({256, 512});  // (8, 16) tiles
    tt::tt_metal::Shape local_shape({128, 512});   // (4, 16) tiles per device

    run_udm_copy_test(mesh_device_.get(), global_shape, local_shape, ShardStrategy::HEIGHT, ShardOrder::NORMAL);
}

TEST_F(MeshDevice2x4Fabric2DUDMFixture, TestMeshHeightShardedCopy3D) {
    // 3D tensor: (2, 8, 16) tiles = (2, 256, 512) elements
    // Height-sharded on height dim: height across 2 mesh rows, replicated across 4 mesh cols
    // Per-device: (2, 4, 16) tiles = (2, 128, 512) elements
    tt::tt_metal::Shape global_shape({2, 256, 512});  // (2, 8, 16) tiles
    tt::tt_metal::Shape local_shape({2, 128, 512});   // (2, 4, 16) tiles per device

    run_udm_copy_test(mesh_device_.get(), global_shape, local_shape, ShardStrategy::HEIGHT, ShardOrder::NORMAL);
}

// ============================================================================
// Height-Sharded Tests with Swapped Order (2x4 Mesh)
// Swapped: replicate on mesh dim 0 (2 rows), shard height on mesh dim 1 (4 cols)
// ============================================================================

TEST_F(MeshDevice2x4Fabric2DUDMFixture, TestMeshHeightShardedSwappedCopy2D) {
    // 2D tensor: (16, 8) tiles = (512, 256) elements
    // Height-sharded swapped: replicated across 2 mesh rows, height across 4 mesh cols
    // Per-device: (4, 8) tiles = (128, 256) elements
    tt::tt_metal::Shape global_shape({512, 256});  // (16, 8) tiles
    tt::tt_metal::Shape local_shape({128, 256});   // (4, 8) tiles per device

    run_udm_copy_test(mesh_device_.get(), global_shape, local_shape, ShardStrategy::HEIGHT, ShardOrder::SWAPPED);
}

TEST_F(MeshDevice2x4Fabric2DUDMFixture, TestMeshHeightShardedSwappedCopy3D) {
    // 3D tensor: (2, 16, 8) tiles = (2, 512, 256) elements
    // Height-sharded swapped: replicated across 2 mesh rows, height across 4 mesh cols
    // Per-device: (2, 4, 8) tiles = (2, 128, 256) elements
    tt::tt_metal::Shape global_shape({2, 512, 256});  // (2, 16, 8) tiles
    tt::tt_metal::Shape local_shape({2, 128, 256});   // (2, 4, 8) tiles per device

    run_udm_copy_test(mesh_device_.get(), global_shape, local_shape, ShardStrategy::HEIGHT, ShardOrder::SWAPPED);
}

// ============================================================================
// Width-Sharded Tests with Normal Order (2x4 Mesh)
// Normal: replicate on mesh dim 0 (2 rows), shard width on mesh dim 1 (4 cols)
// ============================================================================

TEST_F(MeshDevice2x4Fabric2DUDMFixture, TestMeshWidthSharded2DCopy2D) {
    // 2D tensor: (8, 16) tiles = (256, 512) elements
    // Width-sharded: replicated across 2 mesh rows, width across 4 mesh cols
    // Per-device: (8, 4) tiles = (256, 128) elements
    tt::tt_metal::Shape global_shape({256, 512});  // (8, 16) tiles
    tt::tt_metal::Shape local_shape({256, 128});   // (8, 4) tiles per device

    run_udm_copy_test(mesh_device_.get(), global_shape, local_shape, ShardStrategy::WIDTH, ShardOrder::NORMAL);
}

TEST_F(MeshDevice2x4Fabric2DUDMFixture, TestMeshWidthSharded2DCopy3D) {
    // 3D tensor: (2, 8, 16) tiles = (2, 256, 512) elements
    // Width-sharded: replicated across 2 mesh rows, width across 4 mesh cols
    // Per-device: (2, 8, 4) tiles = (2, 256, 128) elements
    tt::tt_metal::Shape global_shape({2, 256, 512});  // (2, 8, 16) tiles
    tt::tt_metal::Shape local_shape({2, 256, 128});   // (2, 8, 4) tiles per device

    run_udm_copy_test(mesh_device_.get(), global_shape, local_shape, ShardStrategy::WIDTH, ShardOrder::NORMAL);
}

// ============================================================================
// Width-Sharded Tests with Swapped Order (2x4 Mesh)
// Swapped: shard width on mesh dim 0 (2 rows), replicate on mesh dim 1 (4 cols)
// ============================================================================

TEST_F(MeshDevice2x4Fabric2DUDMFixture, TestMeshWidthShardedSwappedCopy2D) {
    // 2D tensor: (16, 8) tiles = (512, 256) elements
    // Width-sharded swapped: width across 2 mesh rows, replicated across 4 mesh cols
    // Per-device: (16, 4) tiles = (512, 128) elements
    tt::tt_metal::Shape global_shape({512, 256});  // (16, 8) tiles
    tt::tt_metal::Shape local_shape({512, 128});   // (16, 4) tiles per device

    run_udm_copy_test(mesh_device_.get(), global_shape, local_shape, ShardStrategy::WIDTH, ShardOrder::SWAPPED);
}

TEST_F(MeshDevice2x4Fabric2DUDMFixture, TestMeshWidthShardedSwappedCopy3D) {
    // 3D tensor: (2, 16, 8) tiles = (2, 512, 256) elements
    // Width-sharded swapped: width across 2 mesh rows, replicated across 4 mesh cols
    // Per-device: (2, 16, 4) tiles = (2, 512, 128) elements
    tt::tt_metal::Shape global_shape({2, 512, 256});  // (2, 16, 8) tiles
    tt::tt_metal::Shape local_shape({2, 512, 128});   // (2, 16, 4) tiles per device

    run_udm_copy_test(mesh_device_.get(), global_shape, local_shape, ShardStrategy::WIDTH, ShardOrder::SWAPPED);
}

}  // namespace tt::tt_metal::experimental::udm_tests
