// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include "tests/tt_metal/tt_metal/common/multi_device_fixture.hpp"
#include "tests/ttnn/unit_tests/gtests/udm/test_udm_utils.hpp"
#include "tt_metal/programming_examples/matmul/matmul_common/bmm_op.hpp"

#include "tt_metal/experimental/udm/mesh_kernel.hpp"
#include "tt_metal/experimental/udm/mesh_utils.hpp"
#include "tt_metal/experimental/udm/mesh_circular_buffer.hpp"

namespace tt::tt_metal::experimental::udm_tests {
namespace {

/**
 * @brief Create UDM program that adds two tensors: output = input_a + input_b
 *
 * @param input_a_mesh_tensor_builder Builder for input tensor A
 * @param input_b_mesh_tensor_builder Builder for input tensor B
 * @param output_mesh_tensor_builder Builder for output tensor
 */
tt::tt_metal::experimental::udm::MeshProgram create_program(
    tt::tt_metal::experimental::udm::MeshTensorBuilder& input_a_mesh_tensor_builder,
    tt::tt_metal::experimental::udm::MeshTensorBuilder& input_b_mesh_tensor_builder,
    tt::tt_metal::experimental::udm::MeshTensorBuilder& output_mesh_tensor_builder) {
    // Use bfloat16 data format for circular buffer configuration
    tt::DataFormat data_format = tt::DataFormat::Float16_b;
    // Use the mesh_builder from MeshTensorBuilder (they're identical since all created from same mesh)
    auto& mesh_builder = input_a_mesh_tensor_builder.mesh_builder();

    // Create MeshProgram
    auto program = tt::tt_metal::experimental::udm::CreateMeshProgram(mesh_builder);

    // Map buffer to gcores using UDM API
    // Partition work on all non-last dimensions (0, 1, ..., rank-2)
    // Last dimension (width) is not partitioned as it's the innermost loop in the kernel
    const auto& mesh_tensor_shape = input_a_mesh_tensor_builder.get_mesh_tensor_shape_in_pages();
    uint32_t rank = mesh_tensor_shape.rank();
    std::vector<int> partition_dims;
    partition_dims.reserve(rank - 1);
    for (uint32_t d = 0; d < rank - 1; ++d) {
        partition_dims.push_back(static_cast<int>(d));
    }
    auto gcores_info = tt::tt_metal::experimental::udm::map_tensor_to_gcores_nd(
        input_a_mesh_tensor_builder,
        mesh_builder,   // Pass mesh_builder which contains mesh and grid dimensions
        partition_dims  // partition on all non-last dimensions
    );

    // Log gcores info for debugging
    log_gcores_info(gcores_info, mesh_builder);

    // Get compile-time args from all MeshTensorBuilders
    auto input_a_compile_time_args = input_a_mesh_tensor_builder.get_compile_time_args();
    auto input_b_compile_time_args = input_b_mesh_tensor_builder.get_compile_time_args();
    auto output_compile_time_args = output_mesh_tensor_builder.get_compile_time_args();

    // Combine compile-time args: input A, input B, then output
    std::vector<uint32_t> dataflow_compile_time_args = input_a_compile_time_args;
    dataflow_compile_time_args.insert(
        dataflow_compile_time_args.end(), input_b_compile_time_args.begin(), input_b_compile_time_args.end());
    dataflow_compile_time_args.insert(
        dataflow_compile_time_args.end(), output_compile_time_args.begin(), output_compile_time_args.end());

    // Create mesh circular buffers for tile storage
    uint32_t tile_size = tt::tile_size(data_format);
    constexpr uint32_t cb_in0 = 0;  // Input A
    constexpr uint32_t cb_in1 = 1;  // Input B
    constexpr uint32_t cb_out = 2;  // Output

    // CB for input A
    tt::tt_metal::CircularBufferConfig cb_in0_config =
        tt::tt_metal::CircularBufferConfig(2 * tile_size, {{cb_in0, data_format}}).set_page_size(cb_in0, tile_size);
    auto mesh_cb_in0 = tt::tt_metal::experimental::udm::CreateMeshCircularBuffer(mesh_builder, program, cb_in0_config);

    // CB for input B
    tt::tt_metal::CircularBufferConfig cb_in1_config =
        tt::tt_metal::CircularBufferConfig(2 * tile_size, {{cb_in1, data_format}}).set_page_size(cb_in1, tile_size);
    auto mesh_cb_in1 = tt::tt_metal::experimental::udm::CreateMeshCircularBuffer(mesh_builder, program, cb_in1_config);

    // CB for output
    tt::tt_metal::CircularBufferConfig cb_out_config =
        tt::tt_metal::CircularBufferConfig(2 * tile_size, {{cb_out, data_format}}).set_page_size(cb_out, tile_size);
    auto mesh_cb_out = tt::tt_metal::experimental::udm::CreateMeshCircularBuffer(mesh_builder, program, cb_out_config);

    // Create dataflow kernel (reader/writer combined)
    tt::tt_metal::experimental::udm::MeshKernelHandle dataflow_kernel_id =
        tt::tt_metal::experimental::udm::CreateMeshKernel(
            mesh_builder,
            program,
            "tests/ttnn/unit_tests/gtests/udm/eltwise/kernels/dataflow_add.cpp",
            gcores_info.gcores,
            tt::tt_metal::DataMovementConfig{
                .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
                .noc = tt::tt_metal::NOC::RISCV_0_default,
                .compile_args = dataflow_compile_time_args,
            });

    // Create compute kernel for element-wise addition
    // n_tiles is passed as runtime arg (not compile-time) so non-workers can exit early
    tt::tt_metal::experimental::udm::MeshKernelHandle compute_kernel_id =
        tt::tt_metal::experimental::udm::CreateMeshKernel(
            mesh_builder,
            program,
            "tests/ttnn/unit_tests/gtests/udm/eltwise/kernels/compute_add.cpp",
            gcores_info.gcores,
            tt::tt_metal::ComputeConfig{});

    // Set runtime args for each gcore
    uint32_t gcore_idx = 0;
    for (const auto& gcore : gcores_info.gcores) {
        // Build dataflow runtime args: rank, then for each dim: (pages, offset, stride)
        uint32_t rank = gcores_info.dim_pages[gcore_idx].size();

        std::vector<uint32_t> dataflow_runtime_args;
        dataflow_runtime_args.push_back(rank);

        // Calculate n_tiles for this gcore (product of all dim_pages)
        uint32_t n_tiles = 1;
        for (uint32_t d = 0; d < rank; ++d) {
            dataflow_runtime_args.push_back(gcores_info.dim_pages[gcore_idx][d]);
            dataflow_runtime_args.push_back(gcores_info.dim_offsets[gcore_idx][d]);
            dataflow_runtime_args.push_back(gcores_info.dim_strides[gcore_idx][d]);
            n_tiles *= gcores_info.dim_pages[gcore_idx][d];
        }

        tt::tt_metal::experimental::udm::SetMeshKernelRuntimeArgs(
            mesh_builder, program, dataflow_kernel_id, gcore, dataflow_runtime_args);

        // Set compute runtime args: n_tiles
        std::vector<uint32_t> compute_runtime_args = {n_tiles};
        tt::tt_metal::experimental::udm::SetMeshKernelRuntimeArgs(
            mesh_builder, program, compute_kernel_id, gcore, compute_runtime_args);

        gcore_idx++;
    }

    return program;
}

/**
 * @brief Validate that output tensor equals element-wise sum of input tensors using PCC
 */
inline void validate(
    const ttnn::Tensor& input_a_tensor,
    const ttnn::Tensor& input_b_tensor,
    const ttnn::Tensor& output_tensor,
    ShardStrategy shard_strategy = ShardStrategy::WIDTH,
    ShardOrder shard_order = ShardOrder::NORMAL) {
    auto* mesh_device = input_a_tensor.device();
    bool swap_shard_order = (shard_order == ShardOrder::SWAPPED);

    // Create appropriate composer based on sharding strategy
    std::unique_ptr<ttnn::distributed::MeshToTensor> composer;
    switch (shard_strategy) {
        case ShardStrategy::WIDTH:
            composer =
                create_width_sharded_mesh_composer(mesh_device, input_a_tensor.padded_shape().rank(), swap_shard_order);
            break;
        case ShardStrategy::BLOCK:
            composer =
                create_block_sharded_mesh_composer(mesh_device, input_a_tensor.padded_shape().rank(), swap_shard_order);
            break;
        case ShardStrategy::HEIGHT:
            composer = create_height_sharded_mesh_composer(
                mesh_device, input_a_tensor.padded_shape().rank(), swap_shard_order);
            break;
    }

    // Aggregate tensors and convert to vectors
    auto input_a_data = ttnn::distributed::aggregate_tensor(input_a_tensor, *composer).to_vector<bfloat16>();
    auto input_b_data = ttnn::distributed::aggregate_tensor(input_b_tensor, *composer).to_vector<bfloat16>();
    auto output_data = ttnn::distributed::aggregate_tensor(output_tensor, *composer).to_vector<bfloat16>();

    // Build expected and actual vectors for PCC comparison
    uint32_t volume = input_a_data.size();
    std::vector<bfloat16> expected_values;
    std::vector<bfloat16> actual_values;
    expected_values.reserve(volume);
    actual_values.reserve(volume);

    for (uint32_t i = 0; i < volume; ++i) {
        float expected = static_cast<float>(input_a_data[i]) + static_cast<float>(input_b_data[i]);
        expected_values.push_back(bfloat16(expected));
        actual_values.push_back(output_data[i]);

        // Debug: Print first few values
        if (i < 8) {
            log_info(
                tt::LogTest,
                "  Index {}: A={:.4f}, B={:.4f}, expected={:.4f}, actual={:.4f}",
                i,
                static_cast<float>(input_a_data[i]),
                static_cast<float>(input_b_data[i]),
                expected,
                static_cast<float>(output_data[i]));
        }
    }

    // Check PCC between expected and actual values
    float pcc = check_bfloat16_vector_pcc(expected_values, actual_values);
    const float pcc_threshold = 0.9999f;

    log_info(tt::LogTest, "PCC: {:.6f} (threshold: {:.4f})", pcc, pcc_threshold);

    // Use !(pcc >= threshold) to also catch NaN (NaN comparisons always return false)
    if (!(pcc >= pcc_threshold)) {
        TT_THROW("Add validation failed: PCC {:.6f} below threshold {:.4f}", pcc, pcc_threshold);
    }

    log_info(tt::LogTest, "Add validation passed: PCC={:.6f} for {} elements (A + B = C)", pcc, volume);
}

/**
 * @brief Helper function to run UDM add test with given tensor shapes
 */
void run_udm_add_test(
    tt::tt_metal::distributed::MeshDevice* mesh_device,
    const tt::tt_metal::Shape& global_shape,
    const ShardStrategy& shard_strategy,
    const ShardOrder& shard_order = ShardOrder::NORMAL) {
    // Create tensors based on sharding strategy
    ttnn::Tensor input_a_tensor;
    ttnn::Tensor input_b_tensor;
    ttnn::Tensor output_tensor;
    bool swap_shard_order = (shard_order == ShardOrder::SWAPPED);

    switch (shard_strategy) {
        case ShardStrategy::WIDTH:
            input_a_tensor =
                create_width_distributed_interleaved_bfloat16_tensor(mesh_device, global_shape, swap_shard_order);
            input_b_tensor =
                create_width_distributed_interleaved_bfloat16_tensor(mesh_device, global_shape, swap_shard_order);
            output_tensor =
                create_width_distributed_interleaved_bfloat16_tensor(mesh_device, global_shape, swap_shard_order);
            break;
        case ShardStrategy::BLOCK:
            input_a_tensor =
                create_block_distributed_interleaved_bfloat16_tensor(mesh_device, global_shape, swap_shard_order);
            input_b_tensor =
                create_block_distributed_interleaved_bfloat16_tensor(mesh_device, global_shape, swap_shard_order);
            output_tensor =
                create_block_distributed_interleaved_bfloat16_tensor(mesh_device, global_shape, swap_shard_order);
            break;
        case ShardStrategy::HEIGHT:
            input_a_tensor =
                create_height_distributed_interleaved_bfloat16_tensor(mesh_device, global_shape, swap_shard_order);
            input_b_tensor =
                create_height_distributed_interleaved_bfloat16_tensor(mesh_device, global_shape, swap_shard_order);
            output_tensor =
                create_height_distributed_interleaved_bfloat16_tensor(mesh_device, global_shape, swap_shard_order);
            break;
    }

    // Build tensor builders from tensors (extracts mesh tensor shape info)
    auto input_a_mesh_tensor_builder = create_tensor_builder(input_a_tensor);
    auto input_b_mesh_tensor_builder = create_tensor_builder(input_b_tensor);
    auto output_mesh_tensor_builder = create_tensor_builder(output_tensor);

    // Create program using tensor builders
    auto program = create_program(input_a_mesh_tensor_builder, input_b_mesh_tensor_builder, output_mesh_tensor_builder);

    // Run program
    auto* tensor_mesh_device = input_a_tensor.device();
    ASSERT_NE(tensor_mesh_device, nullptr) << "Tensor must be on device";
    run_program(input_a_tensor, tensor_mesh_device, program);

    // Validate output = input_a + input_b
    validate(input_a_tensor, input_b_tensor, output_tensor, shard_strategy, shard_order);
}

}  // namespace

// ============================================================================
// Width-Sharded Tests (1x4 Mesh)
// ============================================================================

using MeshDevice1x4Fabric2DUDMFixture = tt::tt_metal::MeshDevice1x4Fabric2DUDMFixture;

TEST_F(MeshDevice1x4Fabric2DUDMFixture, TestMeshWidthShardedAdd2D_Small) {
    // Small 2D tensor: (4, 16) tiles = (128, 512) elements
    tt::tt_metal::Shape global_shape({128, 512});
    run_udm_add_test(mesh_device_.get(), global_shape, ShardStrategy::WIDTH);
}

TEST_F(MeshDevice1x4Fabric2DUDMFixture, TestMeshWidthShardedAdd2D_Large) {
    // Larger 2D tensor: (32, 64) tiles = (1024, 2048) elements
    tt::tt_metal::Shape global_shape({1024, 2048});
    run_udm_add_test(mesh_device_.get(), global_shape, ShardStrategy::WIDTH);
}

TEST_F(MeshDevice1x4Fabric2DUDMFixture, TestMeshWidthShardedAdd3D) {
    // 3D tensor: (2, 16, 256) tiles = (2, 512, 8192) elements
    // Sharded along last dimension (width)
    tt::tt_metal::Shape global_shape({2, 512, 8192});
    run_udm_add_test(mesh_device_.get(), global_shape, ShardStrategy::WIDTH);
}

TEST_F(MeshDevice1x4Fabric2DUDMFixture, TestMeshWidthShardedAdd4D) {
    // 4D tensor: (2, 4, 8, 256) tiles = (2, 4, 256, 8192) elements
    // Sharded along last dimension (width)
    tt::tt_metal::Shape global_shape({2, 4, 256, 8192});
    run_udm_add_test(mesh_device_.get(), global_shape, ShardStrategy::WIDTH);
}

// ============================================================================
// Block-Sharded Tests (2x4 Mesh)
// ============================================================================

using MeshDevice2x4Fabric2DUDMFixture = tt::tt_metal::MeshDevice2x4Fabric2DUDMFixture;

TEST_F(MeshDevice2x4Fabric2DUDMFixture, TestMeshBlockShardedAdd2D_Small) {
    // Small 2D tensor: (8, 16) tiles = (256, 512) elements
    // Block-sharded: height across 2 mesh rows, width across 4 mesh cols
    tt::tt_metal::Shape global_shape({256, 512});
    run_udm_add_test(mesh_device_.get(), global_shape, ShardStrategy::BLOCK);
}

TEST_F(MeshDevice2x4Fabric2DUDMFixture, TestMeshBlockShardedAdd2D_Large) {
    // Larger 2D tensor: (64, 128) tiles = (2048, 4096) elements
    // Block-sharded: height across 2 mesh rows, width across 4 mesh cols
    tt::tt_metal::Shape global_shape({2048, 4096});
    run_udm_add_test(mesh_device_.get(), global_shape, ShardStrategy::BLOCK);
}

TEST_F(MeshDevice2x4Fabric2DUDMFixture, TestMeshBlockShardedAdd3D) {
    // 3D tensor: (2, 16, 32) tiles = (2, 512, 1024) elements
    // Block-sharded on last 2 dims: height across 2 mesh rows, width across 4 mesh cols
    tt::tt_metal::Shape global_shape({2, 512, 1024});
    run_udm_add_test(mesh_device_.get(), global_shape, ShardStrategy::BLOCK);
}

TEST_F(MeshDevice2x4Fabric2DUDMFixture, TestMeshBlockShardedAdd4D) {
    // 4D tensor: (2, 4, 16, 32) tiles = (2, 4, 512, 1024) elements
    // Block-sharded on last 2 dims: height across 2 mesh rows, width across 4 mesh cols
    tt::tt_metal::Shape global_shape({2, 4, 512, 1024});
    run_udm_add_test(mesh_device_.get(), global_shape, ShardStrategy::BLOCK);
}

// ============================================================================
// Block-Sharded Tests with Swapped Order (2x4 Mesh)
// Swapped distribution: mesh dim 0 shards tensor width, mesh dim 1 shards tensor height
// ============================================================================

TEST_F(MeshDevice2x4Fabric2DUDMFixture, TestMeshBlockShardedSwappedAdd2D_Small) {
    // Small 2D tensor: (16, 8) tiles = (512, 256) elements
    // Block-sharded with swapped order: width across 2 mesh rows, height across 4 mesh cols
    tt::tt_metal::Shape global_shape({512, 256});
    run_udm_add_test(mesh_device_.get(), global_shape, ShardStrategy::BLOCK, ShardOrder::SWAPPED);
}

TEST_F(MeshDevice2x4Fabric2DUDMFixture, TestMeshBlockShardedSwappedAdd2D_Large) {
    // Larger 2D tensor: (128, 64) tiles = (4096, 2048) elements
    // Block-sharded with swapped order: width across 2 mesh rows, height across 4 mesh cols
    tt::tt_metal::Shape global_shape({4096, 2048});
    run_udm_add_test(mesh_device_.get(), global_shape, ShardStrategy::BLOCK, ShardOrder::SWAPPED);
}

TEST_F(MeshDevice2x4Fabric2DUDMFixture, TestMeshBlockShardedSwappedAdd3D) {
    // 3D tensor: (2, 32, 16) tiles = (2, 1024, 512) elements
    // Block-sharded with swapped order on last 2 dims
    tt::tt_metal::Shape global_shape({2, 1024, 512});
    run_udm_add_test(mesh_device_.get(), global_shape, ShardStrategy::BLOCK, ShardOrder::SWAPPED);
}

TEST_F(MeshDevice2x4Fabric2DUDMFixture, TestMeshBlockShardedSwappedAdd4D) {
    // 4D tensor: (2, 4, 32, 16) tiles = (2, 4, 1024, 512) elements
    // Block-sharded with swapped order on last 2 dims
    tt::tt_metal::Shape global_shape({2, 4, 1024, 512});
    run_udm_add_test(mesh_device_.get(), global_shape, ShardStrategy::BLOCK, ShardOrder::SWAPPED);
}

// ============================================================================
// Height-Sharded Tests with Normal Order (2x4 Mesh)
// Normal: shard height on mesh dim 0 (2 rows), replicate on mesh dim 1 (4 cols)
// ============================================================================

TEST_F(MeshDevice2x4Fabric2DUDMFixture, TestMeshHeightShardedAdd2D) {
    // 2D tensor: (8, 16) tiles = (256, 512) elements
    // Height-sharded: height across 2 mesh rows, replicated across 4 mesh cols
    tt::tt_metal::Shape global_shape({256, 512});
    run_udm_add_test(mesh_device_.get(), global_shape, ShardStrategy::HEIGHT, ShardOrder::NORMAL);
}

TEST_F(MeshDevice2x4Fabric2DUDMFixture, TestMeshHeightShardedAdd3D) {
    // 3D tensor: (2, 8, 16) tiles = (2, 256, 512) elements
    // Height-sharded on height dim
    tt::tt_metal::Shape global_shape({2, 256, 512});
    run_udm_add_test(mesh_device_.get(), global_shape, ShardStrategy::HEIGHT, ShardOrder::NORMAL);
}

// ============================================================================
// Height-Sharded Tests with Swapped Order (2x4 Mesh)
// Swapped: replicate on mesh dim 0 (2 rows), shard height on mesh dim 1 (4 cols)
// ============================================================================

TEST_F(MeshDevice2x4Fabric2DUDMFixture, TestMeshHeightShardedSwappedAdd2D) {
    // 2D tensor: (16, 8) tiles = (512, 256) elements
    // Height-sharded swapped: replicated across 2 mesh rows, height across 4 mesh cols
    tt::tt_metal::Shape global_shape({512, 256});
    run_udm_add_test(mesh_device_.get(), global_shape, ShardStrategy::HEIGHT, ShardOrder::SWAPPED);
}

TEST_F(MeshDevice2x4Fabric2DUDMFixture, TestMeshHeightShardedSwappedAdd3D) {
    // 3D tensor: (2, 16, 8) tiles = (2, 512, 256) elements
    // Height-sharded swapped
    tt::tt_metal::Shape global_shape({2, 512, 256});
    run_udm_add_test(mesh_device_.get(), global_shape, ShardStrategy::HEIGHT, ShardOrder::SWAPPED);
}

// ============================================================================
// Width-Sharded Tests with Normal Order (2x4 Mesh)
// Normal: replicate on mesh dim 0 (2 rows), shard width on mesh dim 1 (4 cols)
// ============================================================================

TEST_F(MeshDevice2x4Fabric2DUDMFixture, TestMeshWidthSharded2DAdd2D) {
    // 2D tensor: (8, 16) tiles = (256, 512) elements
    // Width-sharded: replicated across 2 mesh rows, width across 4 mesh cols
    tt::tt_metal::Shape global_shape({256, 512});
    run_udm_add_test(mesh_device_.get(), global_shape, ShardStrategy::WIDTH, ShardOrder::NORMAL);
}

TEST_F(MeshDevice2x4Fabric2DUDMFixture, TestMeshWidthSharded2DAdd3D) {
    // 3D tensor: (2, 8, 16) tiles = (2, 256, 512) elements
    // Width-sharded
    tt::tt_metal::Shape global_shape({2, 256, 512});
    run_udm_add_test(mesh_device_.get(), global_shape, ShardStrategy::WIDTH, ShardOrder::NORMAL);
}

// ============================================================================
// Width-Sharded Tests with Swapped Order (2x4 Mesh)
// Swapped: shard width on mesh dim 0 (2 rows), replicate on mesh dim 1 (4 cols)
// ============================================================================

TEST_F(MeshDevice2x4Fabric2DUDMFixture, TestMeshWidthShardedSwappedAdd2D) {
    // 2D tensor: (16, 8) tiles = (512, 256) elements
    // Width-sharded swapped: width across 2 mesh rows, replicated across 4 mesh cols
    tt::tt_metal::Shape global_shape({512, 256});
    run_udm_add_test(mesh_device_.get(), global_shape, ShardStrategy::WIDTH, ShardOrder::SWAPPED);
}

TEST_F(MeshDevice2x4Fabric2DUDMFixture, TestMeshWidthShardedSwappedAdd3D) {
    // 3D tensor: (2, 16, 8) tiles = (2, 512, 256) elements
    // Width-sharded swapped
    tt::tt_metal::Shape global_shape({2, 512, 256});
    run_udm_add_test(mesh_device_.get(), global_shape, ShardStrategy::WIDTH, ShardOrder::SWAPPED);
}

}  // namespace tt::tt_metal::experimental::udm_tests
