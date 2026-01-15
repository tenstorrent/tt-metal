// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <cmath>

#include "tests/tt_metal/tt_metal/common/multi_device_fixture.hpp"
#include "tests/ttnn/unit_tests/gtests/udm/test_udm_utils.hpp"
#include "tt_metal/programming_examples/matmul/matmul_common/bmm_op.hpp"
#include "tt_metal/api/tt-metalium/bfloat16.hpp"

#include "tt_metal/experimental/udm/mesh_kernel.hpp"
#include "tt_metal/experimental/udm/mesh_utils.hpp"
#include "tt_metal/experimental/udm/mesh_circular_buffer.hpp"

namespace tt::tt_metal::experimental::udm_tests {
namespace {

/**
 * @brief Create UDM program for width reduction (interleaved version)
 *
 * Each core reduces a subset of rows across width dimension.
 * Input: [H, W] tiles -> Output: [H, 1] tiles
 *
 * @param input_mesh_tensor_builder Builder for input tensor
 * @param output_mesh_tensor_builder Builder for output tensor
 */
tt::tt_metal::experimental::udm::MeshProgram create_program(
    tt::tt_metal::experimental::udm::MeshTensorBuilder& input_mesh_tensor_builder,
    tt::tt_metal::experimental::udm::MeshTensorBuilder& output_mesh_tensor_builder) {
    // Use bfloat16 data format
    tt::DataFormat data_format = tt::DataFormat::Float16_b;
    auto& mesh_builder = input_mesh_tensor_builder.mesh_builder();

    // Create MeshProgram
    auto program = tt::tt_metal::experimental::udm::CreateMeshProgram(mesh_builder);

    // Map tensor to gcores - partition on height (dim 0) so each core handles some rows
    int partition_dim = 0;
    auto gcores_info =
        tt::tt_metal::experimental::udm::map_tensor_to_gcores(input_mesh_tensor_builder, mesh_builder, partition_dim);

    // Log gcores info for debugging
    log_gcores_info(gcores_info, mesh_builder);

    // Get compile-time args from MeshTensorBuilders
    auto input_compile_time_args = input_mesh_tensor_builder.get_compile_time_args();
    auto output_compile_time_args = output_mesh_tensor_builder.get_compile_time_args();

    // Create packed scaler for SUM reduction (1.0)
    bfloat16 bfloat_scaler = bfloat16(1.0f);
    uint32_t packed_scaler = pack_two_bfloat16_into_uint32({bfloat_scaler, bfloat_scaler});

    // Combine compile-time args: input, output, then packed scaler
    std::vector<uint32_t> dataflow_compile_time_args = input_compile_time_args;
    dataflow_compile_time_args.insert(
        dataflow_compile_time_args.end(), output_compile_time_args.begin(), output_compile_time_args.end());
    dataflow_compile_time_args.push_back(packed_scaler);

    // Get input tensor shape info for CB sizing
    const auto& input_shape = input_mesh_tensor_builder.get_mesh_tensor_shape_in_pages();
    uint32_t input_width_tiles = input_shape[-1];  // Width in tiles (for CB sizing)

    // Create mesh circular buffers
    uint32_t tile_size = tt::tile_size(data_format);
    constexpr uint32_t cb_in0 = 0;     // Input tiles (row of width tiles)
    constexpr uint32_t cb_scaler = 1;  // Scaler tile
    constexpr uint32_t cb_out = 2;     // Output tiles

    // CB for input - needs to hold a full row of tiles
    uint32_t input_cb_tiles = input_width_tiles;
    tt::tt_metal::CircularBufferConfig cb_in0_config =
        tt::tt_metal::CircularBufferConfig(input_cb_tiles * tile_size, {{cb_in0, data_format}})
            .set_page_size(cb_in0, tile_size);
    auto mesh_cb_in0 = tt::tt_metal::experimental::udm::CreateMeshCircularBuffer(mesh_builder, program, cb_in0_config);

    // CB for scaler
    tt::tt_metal::CircularBufferConfig cb_scaler_config =
        tt::tt_metal::CircularBufferConfig(tile_size, {{cb_scaler, data_format}}).set_page_size(cb_scaler, tile_size);
    auto mesh_cb_scaler =
        tt::tt_metal::experimental::udm::CreateMeshCircularBuffer(mesh_builder, program, cb_scaler_config);

    // CB for output - double buffer
    tt::tt_metal::CircularBufferConfig cb_out_config =
        tt::tt_metal::CircularBufferConfig(2 * tile_size, {{cb_out, data_format}}).set_page_size(cb_out, tile_size);
    auto mesh_cb_out = tt::tt_metal::experimental::udm::CreateMeshCircularBuffer(mesh_builder, program, cb_out_config);

    // Create dataflow kernel
    tt::tt_metal::experimental::udm::MeshKernelHandle dataflow_kernel_id =
        tt::tt_metal::experimental::udm::CreateMeshKernel(
            mesh_builder,
            program,
            "tests/ttnn/unit_tests/gtests/udm/reduction/interleaved/kernels/dataflow_reduce.cpp",
            gcores_info.gcores,
            tt::tt_metal::DataMovementConfig{
                .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
                .noc = tt::tt_metal::NOC::RISCV_0_default,
                .compile_args = dataflow_compile_time_args,
            });

    // Create compute kernel
    tt::tt_metal::experimental::udm::MeshKernelHandle compute_kernel_id =
        tt::tt_metal::experimental::udm::CreateMeshKernel(
            mesh_builder,
            program,
            "tests/ttnn/unit_tests/gtests/udm/reduction/interleaved/kernels/compute_reduce.cpp",
            gcores_info.gcores,
            tt::tt_metal::ComputeConfig{
                .fp32_dest_acc_en = true,  // Use FP32 accumulation for better precision
            });

    // Map output tensor to gcores to get output strides
    auto output_gcores_info =
        tt::tt_metal::experimental::udm::map_tensor_to_gcores(output_mesh_tensor_builder, mesh_builder, partition_dim);

    // Set runtime args for each gcore
    uint32_t gcore_idx = 0;
    for (const auto& gcore : gcores_info.gcores) {
        uint32_t rank = gcores_info.dim_pages[gcore_idx].size();

        // Dataflow runtime args: rank, then for each dim: (pages, offset, stride), then output strides
        std::vector<uint32_t> dataflow_runtime_args;
        dataflow_runtime_args.push_back(rank);
        for (uint32_t d = 0; d < rank; ++d) {
            dataflow_runtime_args.push_back(gcores_info.dim_pages[gcore_idx][d]);
            dataflow_runtime_args.push_back(gcores_info.dim_offsets[gcore_idx][d]);
            dataflow_runtime_args.push_back(gcores_info.dim_strides[gcore_idx][d]);
        }
        // Add output strides for each row dimension (all dims except last which is width)
        for (uint32_t d = 0; d < rank - 1; ++d) {
            dataflow_runtime_args.push_back(output_gcores_info.dim_strides[gcore_idx][d]);
        }

        // Compute runtime args: rank, then pages per dim (no offset/stride needed)
        std::vector<uint32_t> compute_runtime_args;
        compute_runtime_args.push_back(rank);
        for (uint32_t d = 0; d < rank; ++d) {
            compute_runtime_args.push_back(gcores_info.dim_pages[gcore_idx][d]);
        }

        tt::tt_metal::experimental::udm::SetMeshKernelRuntimeArgs(
            mesh_builder, program, dataflow_kernel_id, gcore, dataflow_runtime_args);

        tt::tt_metal::experimental::udm::SetMeshKernelRuntimeArgs(
            mesh_builder, program, compute_kernel_id, gcore, compute_runtime_args);

        gcore_idx++;
    }

    return program;
}

/**
 * @brief Validate width reduction result using PCC
 *
 * For input tensor of shape (H, W), output should be (H, 1) where
 * output[h] = sum(input[h, :])
 *
 * Output tensor is height-sharded + replicated across mesh columns.
 * After aggregation with height_sharded_composer, output shape is [H, W * num_replicas].
 * We use the first replica (first W_original elements per row) for validation.
 */
void validate(
    const ttnn::Tensor& input_tensor,
    const ttnn::Tensor& output_tensor,
    ShardStrategy shard_strategy,
    ShardOrder shard_order = ShardOrder::NORMAL) {
    auto* mesh_device = input_tensor.device();
    bool swap_shard_order = (shard_order == ShardOrder::SWAPPED);

    // Create appropriate composer for input based on sharding strategy
    std::unique_ptr<ttnn::distributed::MeshToTensor> input_composer;

    switch (shard_strategy) {
        case ShardStrategy::WIDTH:
            input_composer =
                create_width_sharded_mesh_composer(mesh_device, input_tensor.padded_shape().rank(), swap_shard_order);
            break;
        case ShardStrategy::BLOCK:
            input_composer =
                create_block_sharded_mesh_composer(mesh_device, input_tensor.padded_shape().rank(), swap_shard_order);
            break;
        case ShardStrategy::HEIGHT:
            input_composer =
                create_height_sharded_mesh_composer(mesh_device, input_tensor.padded_shape().rank(), swap_shard_order);
            break;
    }

    // Output tensor is height-distributed (Shard{height}, Replicate{})
    // Composer produces [H, W * num_replicas] - we use the first W_original elements per row
    auto output_composer =
        create_height_sharded_mesh_composer(mesh_device, output_tensor.padded_shape().rank(), swap_shard_order);

    // Aggregate tensors
    auto input_aggregated = ttnn::distributed::aggregate_tensor(input_tensor, *input_composer);
    auto output_aggregated = ttnn::distributed::aggregate_tensor(output_tensor, *output_composer);

    auto input_data = input_aggregated.to_vector<bfloat16>();
    auto output_data = output_aggregated.to_vector<bfloat16>();

    // Get dimensions
    // Input is 2D: [H, W]
    uint32_t input_height = input_aggregated.padded_shape()[-2];
    uint32_t input_width = input_aggregated.padded_shape()[-1];

    // Output is 2D after aggregation: [H, W * num_replicas]
    // We use the first replica, so stride is the full aggregated width
    uint32_t output_aggregated_width = output_aggregated.padded_shape()[-1];
    // Original output width (before replication) from the local tensor
    uint32_t output_original_width = output_tensor.padded_shape()[-1];

    log_info(tt::LogTest, "Input shape: {}x{}", input_height, input_width);
    log_info(
        tt::LogTest,
        "Output aggregated shape: {} (using first {} elements per row)",
        output_aggregated.padded_shape(),
        output_original_width);

    // Build expected and actual vectors for PCC
    std::vector<bfloat16> expected_values;
    std::vector<bfloat16> actual_values;
    expected_values.reserve(input_height);
    actual_values.reserve(input_height);

    for (uint32_t row = 0; row < input_height; ++row) {
        // Compute expected sum for this row
        float expected_sum = 0.0f;
        for (uint32_t col = 0; col < input_width; ++col) {
            uint32_t input_idx = row * input_width + col;
            expected_sum += static_cast<float>(input_data[input_idx]);
        }
        expected_values.push_back(bfloat16(expected_sum));

        // Get actual value from first replica (first element of row in aggregated output)
        // Output layout: [H, W * num_replicas] -> index = row * aggregated_width + 0
        uint32_t output_idx = row * output_aggregated_width;
        actual_values.push_back(output_data[output_idx]);

        // Debug: Print first few rows
        if (row < 8) {
            log_info(
                tt::LogTest,
                "  Row {}: expected={:.4f}, actual={:.4f}",
                row,
                expected_sum,
                static_cast<float>(output_data[output_idx]));
        }
    }

    // Check PCC
    float pcc = check_bfloat16_vector_pcc(expected_values, actual_values);
    const float pcc_threshold = 0.9999f;

    log_info(tt::LogTest, "PCC: {:.6f} (threshold: {:.4f})", pcc, pcc_threshold);

    // Use !(pcc >= threshold) to also catch NaN (NaN comparisons always return false)
    if (!(pcc >= pcc_threshold)) {
        TT_THROW("Width reduction validation failed: PCC {:.6f} below threshold {:.4f}", pcc, pcc_threshold);
    }

    log_info(tt::LogTest, "Width reduction validation passed: PCC={:.6f} for {} rows", pcc, input_height);
}

/**
 * @brief Helper function to run width reduction test with interleaved tensors
 */
void run_width_reduction_interleaved_test(
    tt::tt_metal::distributed::MeshDevice* mesh_device,
    const tt::tt_metal::Shape& input_global_shape,
    const ShardStrategy& shard_strategy,
    const ShardOrder& shard_order = ShardOrder::NORMAL) {
    bool swap_shard_order = (shard_order == ShardOrder::SWAPPED);

    // Create input tensor - interleaved
    ttnn::Tensor input_tensor;
    switch (shard_strategy) {
        case ShardStrategy::WIDTH:
            input_tensor =
                create_width_distributed_interleaved_bfloat16_tensor(mesh_device, input_global_shape, swap_shard_order);
            break;
        case ShardStrategy::BLOCK:
            input_tensor =
                create_block_distributed_interleaved_bfloat16_tensor(mesh_device, input_global_shape, swap_shard_order);
            break;
        case ShardStrategy::HEIGHT:
            input_tensor = create_height_distributed_interleaved_bfloat16_tensor(
                mesh_device, input_global_shape, swap_shard_order);
            break;
    }

    // Create output tensor - width reduced to 1 tile (32 elements)
    // Output tensor is HEIGHT-DISTRIBUTED across all devices (not width distributed)
    // because output width is only 1 tile and cannot be split further.
    // Each device owns different height tiles based on work partitioning.
    tt::tt_metal::Shape output_shape = input_global_shape;
    output_shape[-1] = 32;  // Reduce width to single tile

    ttnn::Tensor output_tensor = create_height_distributed_interleaved_bfloat16_tensor(
        mesh_device, output_shape, swap_shard_order, /*random_init=*/false);

    // Build tensor builders
    auto input_mesh_tensor_builder = create_tensor_builder(input_tensor);
    auto output_mesh_tensor_builder = create_tensor_builder(output_tensor);

    // Log tensor info
    log_tensor_shape_info(input_mesh_tensor_builder, input_tensor);
    log_tensor_shape_info(output_mesh_tensor_builder, output_tensor);

    // Create and run program
    auto program = create_program(input_mesh_tensor_builder, output_mesh_tensor_builder);

    auto* tensor_mesh_device = input_tensor.device();
    ASSERT_NE(tensor_mesh_device, nullptr) << "Tensor must be on device";
    run_program(input_tensor, tensor_mesh_device, program);

    // Validate
    validate(input_tensor, output_tensor, shard_strategy, shard_order);
}

}  // namespace

// ============================================================================
// Width-Distributed Tests (1x4 Mesh) - Interleaved
// ============================================================================

using MeshDevice1x4Fabric2DUDMFixture = tt::tt_metal::MeshDevice1x4Fabric2DUDMFixture;

TEST_F(MeshDevice1x4Fabric2DUDMFixture, TestWidthReductionInterleaved2D_Small) {
    // Small 2D tensor: (4, 16) tiles = (128, 512) elements
    // Width-distributed: each device gets (4, 4) tiles
    // Output: (4, 1) tiles per device
    tt::tt_metal::Shape input_global_shape({128, 512});
    run_width_reduction_interleaved_test(mesh_device_.get(), input_global_shape, ShardStrategy::WIDTH);
}

TEST_F(MeshDevice1x4Fabric2DUDMFixture, TestWidthReductionInterleaved2D_Medium) {
    // Medium 2D tensor: (16, 32) tiles = (512, 1024) elements
    // Width-distributed: each device gets (16, 8) tiles
    tt::tt_metal::Shape input_global_shape({512, 1024});
    run_width_reduction_interleaved_test(mesh_device_.get(), input_global_shape, ShardStrategy::WIDTH);
}

TEST_F(MeshDevice1x4Fabric2DUDMFixture, TestWidthReductionInterleaved2D_Large) {
    // Large 2D tensor: (32, 64) tiles = (1024, 2048) elements
    // Width-distributed: each device gets (32, 16) tiles
    tt::tt_metal::Shape input_global_shape({1024, 2048});
    run_width_reduction_interleaved_test(mesh_device_.get(), input_global_shape, ShardStrategy::WIDTH);
}

TEST_F(MeshDevice1x4Fabric2DUDMFixture, TestWidthReductionInterleaved3D) {
    // 3D tensor: (2, 16, 64) tiles = (2, 512, 2048) elements
    // Width-distributed: each device gets (2, 16, 16) tiles
    tt::tt_metal::Shape input_global_shape({2, 512, 2048});
    run_width_reduction_interleaved_test(mesh_device_.get(), input_global_shape, ShardStrategy::WIDTH);
}

TEST_F(MeshDevice1x4Fabric2DUDMFixture, TestWidthReductionInterleaved4D) {
    // 4D tensor: (2, 4, 8, 64) tiles = (2, 4, 256, 2048) elements
    // Width-distributed: each device gets (2, 4, 8, 16) tiles
    tt::tt_metal::Shape input_global_shape({2, 4, 256, 2048});
    run_width_reduction_interleaved_test(mesh_device_.get(), input_global_shape, ShardStrategy::WIDTH);
}

// ============================================================================
// Block-Distributed Tests (2x4 Mesh) - Interleaved
// ============================================================================

using MeshDevice2x4Fabric2DUDMFixture = tt::tt_metal::MeshDevice2x4Fabric2DUDMFixture;

TEST_F(MeshDevice2x4Fabric2DUDMFixture, TestWidthReductionInterleaved2D_Small) {
    // Small 2D tensor: (8, 16) tiles = (256, 512) elements
    // Block-distributed: each device gets (4, 4) tiles
    tt::tt_metal::Shape input_global_shape({256, 512});
    run_width_reduction_interleaved_test(mesh_device_.get(), input_global_shape, ShardStrategy::BLOCK);
}

TEST_F(MeshDevice2x4Fabric2DUDMFixture, TestWidthReductionInterleaved2D_Medium) {
    // Medium 2D tensor: (32, 64) tiles = (1024, 2048) elements
    // Block-distributed: each device gets (16, 16) tiles
    tt::tt_metal::Shape input_global_shape({1024, 2048});
    run_width_reduction_interleaved_test(mesh_device_.get(), input_global_shape, ShardStrategy::BLOCK);
}

TEST_F(MeshDevice2x4Fabric2DUDMFixture, TestWidthReductionInterleaved2D_Large) {
    // Large 2D tensor: (64, 128) tiles = (2048, 4096) elements
    // Block-distributed: each device gets (32, 32) tiles
    tt::tt_metal::Shape input_global_shape({2048, 4096});
    run_width_reduction_interleaved_test(mesh_device_.get(), input_global_shape, ShardStrategy::BLOCK);
}

TEST_F(MeshDevice2x4Fabric2DUDMFixture, TestWidthReductionInterleaved3D) {
    // 3D tensor: (2, 16, 32) tiles = (2, 512, 1024) elements
    // Block-distributed: each device gets (2, 8, 8) tiles
    tt::tt_metal::Shape input_global_shape({2, 512, 1024});
    run_width_reduction_interleaved_test(mesh_device_.get(), input_global_shape, ShardStrategy::BLOCK);
}

TEST_F(MeshDevice2x4Fabric2DUDMFixture, TestWidthReductionInterleaved4D) {
    // 4D tensor: (2, 4, 16, 32) tiles = (2, 4, 512, 1024) elements
    // Block-distributed: each device gets (2, 4, 8, 8) tiles
    tt::tt_metal::Shape input_global_shape({2, 4, 512, 1024});
    run_width_reduction_interleaved_test(mesh_device_.get(), input_global_shape, ShardStrategy::BLOCK);
}

// ============================================================================
// Block-Distributed Tests with Swapped Order (2x4 Mesh) - Interleaved
// Swapped: mesh dim 0 shards width, mesh dim 1 shards height
// ============================================================================

TEST_F(MeshDevice2x4Fabric2DUDMFixture, TestWidthReductionInterleavedSwapped2D_Small) {
    // Small 2D tensor: (16, 8) tiles = (512, 256) elements
    // Block-distributed swapped: width across 2 mesh rows, height across 4 mesh cols
    tt::tt_metal::Shape input_global_shape({512, 256});
    run_width_reduction_interleaved_test(
        mesh_device_.get(), input_global_shape, ShardStrategy::BLOCK, ShardOrder::SWAPPED);
}

TEST_F(MeshDevice2x4Fabric2DUDMFixture, TestWidthReductionInterleavedSwapped2D_Large) {
    // Larger 2D tensor: (128, 64) tiles = (4096, 2048) elements
    // Block-distributed swapped: width across 2 mesh rows, height across 4 mesh cols
    tt::tt_metal::Shape input_global_shape({4096, 2048});
    run_width_reduction_interleaved_test(
        mesh_device_.get(), input_global_shape, ShardStrategy::BLOCK, ShardOrder::SWAPPED);
}

TEST_F(MeshDevice2x4Fabric2DUDMFixture, TestWidthReductionInterleavedSwapped3D) {
    // 3D tensor: (2, 32, 16) tiles = (2, 1024, 512) elements
    // Block-distributed swapped on last 2 dims
    tt::tt_metal::Shape input_global_shape({2, 1024, 512});
    run_width_reduction_interleaved_test(
        mesh_device_.get(), input_global_shape, ShardStrategy::BLOCK, ShardOrder::SWAPPED);
}

TEST_F(MeshDevice2x4Fabric2DUDMFixture, TestWidthReductionInterleavedSwapped4D) {
    // 4D tensor: (2, 4, 32, 16) tiles = (2, 4, 1024, 512) elements
    // Block-distributed swapped on last 2 dims
    tt::tt_metal::Shape input_global_shape({2, 4, 1024, 512});
    run_width_reduction_interleaved_test(
        mesh_device_.get(), input_global_shape, ShardStrategy::BLOCK, ShardOrder::SWAPPED);
}

}  // namespace tt::tt_metal::experimental::udm_tests
