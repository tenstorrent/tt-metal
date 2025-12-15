// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include "tests/tt_metal/tt_metal/common/multi_device_fixture.hpp"
#include "tests/tt_metal/udm/test_udm_utils.hpp"

#include "tt_metal/udm/mesh_kernel.hpp"
#include "tt_metal/udm/mesh_utils.hpp"
#include "tt_metal/udm/mesh_circular_buffer.hpp"
#include "tt_metal/udm/mesh_semaphore.hpp"
#include "tt_metal/api/tt-metalium/bfloat16.hpp"

namespace tt::tt_metal::experimental::udm_tests {

/**
 * @brief Create UDM program for width reduction using MeshGcoreAccessor
 */
tt::tt_metal::experimental::udm::MeshProgram create_program(
    const ttnn::Tensor& input_tensor, const ttnn::Tensor& output_tensor) {
    auto input_mesh_tensor_builder = create_tensor_builder(input_tensor);
    auto output_mesh_tensor_builder = create_tensor_builder(output_tensor);

    auto& mesh_builder = input_mesh_tensor_builder.mesh_builder();

    // Create MeshProgram
    auto program = tt::tt_metal::experimental::udm::CreateMeshProgram(mesh_builder);

    log_tensor_shape_info(input_mesh_tensor_builder, input_tensor);

    // Map buffer to gcores - partition on width dimension (last dim)
    // Each gcore gets a portion of the width
    int partition_dim = -1;  // Last dimension (width)
    auto gcores_info =
        tt::tt_metal::experimental::udm::map_tensor_to_gcores(input_mesh_tensor_builder, mesh_builder, partition_dim);

    // Also get output tensor gcore info for output strides
    auto output_gcores_info =
        tt::tt_metal::experimental::udm::map_tensor_to_gcores(output_mesh_tensor_builder, mesh_builder, partition_dim);

    log_gcores_info(gcores_info, mesh_builder);

    // Get compile-time args for tensor accessors
    auto input_compile_time_args = input_mesh_tensor_builder.get_compile_time_args();
    auto output_compile_time_args = output_mesh_tensor_builder.get_compile_time_args();

    // Combine compile-time args
    std::vector<uint32_t> compile_time_args = input_compile_time_args;
    compile_time_args.insert(compile_time_args.end(), output_compile_time_args.begin(), output_compile_time_args.end());

    // Get MeshGcoreAccessor defines
    auto gcore_defines = mesh_builder.get_compile_time_defines();

    // Create separate mesh circular buffers for each purpose
    tt::DataFormat data_format = tt::tt_metal::datatype_to_dataformat_converter(input_tensor.dtype());
    uint32_t tile_size = tt::tile_size(data_format);

    // CB 0: Input tiles (dataflow -> compute)
    constexpr uint32_t cb_id_in = 0;
    tt::tt_metal::CircularBufferConfig cb_config_in =
        tt::tt_metal::CircularBufferConfig(tile_size, {{cb_id_in, data_format}}).set_page_size(cb_id_in, tile_size);
    auto mesh_cb_in = tt::tt_metal::experimental::udm::CreateMeshCircularBuffer(mesh_builder, program, cb_config_in);

    // CB 1: Scaler for reduction (compute)
    constexpr uint32_t cb_id_scaler = 1;
    tt::tt_metal::CircularBufferConfig cb_config_scaler =
        tt::tt_metal::CircularBufferConfig(tile_size, {{cb_id_scaler, data_format}})
            .set_page_size(cb_id_scaler, tile_size);
    auto mesh_cb_scaler =
        tt::tt_metal::experimental::udm::CreateMeshCircularBuffer(mesh_builder, program, cb_config_scaler);

    // CB 2: Reduced output tiles (compute -> dataflow)
    constexpr uint32_t cb_id_reduced = 2;
    tt::tt_metal::CircularBufferConfig cb_config_reduced =
        tt::tt_metal::CircularBufferConfig(tile_size, {{cb_id_reduced, data_format}})
            .set_page_size(cb_id_reduced, tile_size);
    auto mesh_cb_reduced =
        tt::tt_metal::experimental::udm::CreateMeshCircularBuffer(mesh_builder, program, cb_config_reduced);

    // CB 3: All reduced tiles from all gcores (dataflow)
    // Space for total_gcores * num_output_rows tiles (including first gcore)
    // Calculate num_output_rows for CB sizing
    uint32_t rank = gcores_info.dim_pages[0].size();
    uint32_t num_output_rows = 1;
    for (uint32_t d = 0; d < rank - 1; ++d) {
        num_output_rows *= gcores_info.dim_pages[0][d];
    }

    constexpr uint32_t cb_id_received = 3;
    uint32_t num_all_gcore_tiles = total_gcores * num_output_rows;
    tt::tt_metal::CircularBufferConfig cb_config_received =
        tt::tt_metal::CircularBufferConfig(num_all_gcore_tiles * tile_size, {{cb_id_received, data_format}})
            .set_page_size(cb_id_received, tile_size);
    auto mesh_cb_received =
        tt::tt_metal::experimental::udm::CreateMeshCircularBuffer(mesh_builder, program, cb_config_received);

    // CB 4: Inter-gcore addition result (compute -> dataflow)
    constexpr uint32_t cb_id_add_result = 4;
    tt::tt_metal::CircularBufferConfig cb_config_add_result =
        tt::tt_metal::CircularBufferConfig(tile_size, {{cb_id_add_result, data_format}})
            .set_page_size(cb_id_add_result, tile_size);
    auto mesh_cb_add_result =
        tt::tt_metal::experimental::udm::CreateMeshCircularBuffer(mesh_builder, program, cb_config_add_result);

    // Create global semaphore for synchronization across all gcores in the mesh
    // All gcores (including first) will increment this semaphore when they send tiles
    constexpr uint32_t semaphore_id = 0;
    tt::tt_metal::experimental::udm::CreateMeshSemaphore(mesh_builder, program, 0);

    // Packed scaler value for reduction (1.0 = no scaling, just sum)
    bfloat16 bfloat_scaler_value = bfloat16(1.0f);
    uint32_t packed_scaler_value = pack_two_bfloat16_into_uint32({bfloat_scaler_value, bfloat_scaler_value});

    // Dataflow kernel compile-time args: packed scaler + CB IDs + semaphore ID + tensor accessors
    std::vector<uint32_t> dataflow_compile_time_args = {
        packed_scaler_value, cb_id_in, cb_id_scaler, cb_id_reduced, cb_id_received, cb_id_add_result, semaphore_id};
    dataflow_compile_time_args.insert(
        dataflow_compile_time_args.end(), compile_time_args.begin(), compile_time_args.end());

    // Compute kernel compile-time args: total_gcores + CB IDs
    std::vector<uint32_t> compute_compile_time_args = {
        total_gcores, cb_id_in, cb_id_scaler, cb_id_reduced, cb_id_received, cb_id_add_result};

    // Create dataflow kernel (reader + writer) on all mapped gcores
    tt::tt_metal::experimental::udm::MeshKernelHandle dataflow_kernel_id =
        tt::tt_metal::experimental::udm::CreateMeshKernel(
            mesh_builder,
            program,
            "tests/tt_metal/udm/kernels/width_reduction_dataflow.cpp",
            gcores_info.gcores,
            tt::tt_metal::DataMovementConfig{
                .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
                .noc = tt::tt_metal::NOC::RISCV_0_default,
                .compile_args = dataflow_compile_time_args,
                .defines = gcore_defines,
            });

    // Create compute kernel on all mapped gcores
    // Merge gcore_defines with compute-specific defines
    std::map<std::string, std::string> compute_defines = gcore_defines;
    compute_defines["REDUCE_OP"] = "PoolType::SUM";
    compute_defines["REDUCE_DIM"] = "ReduceDim::REDUCE_ROW";

    tt::tt_metal::experimental::udm::MeshKernelHandle compute_kernel_id =
        tt::tt_metal::experimental::udm::CreateMeshKernel(
            mesh_builder,
            program,
            "tests/tt_metal/udm/kernels/width_reduction_compute.cpp",
            gcores_info.gcores,
            tt::tt_metal::ComputeConfig{
                .compile_args = compute_compile_time_args,
                .defines = compute_defines,
            });

    // Set runtime args for each gcore
    uint32_t total_gcores = gcores_info.gcores.size();

    for (uint32_t gcore_idx = 0; gcore_idx < total_gcores; ++gcore_idx) {
        const auto& gcore = gcores_info.gcores[gcore_idx];
        uint32_t rank = gcores_info.dim_pages[gcore_idx].size();

        std::vector<uint32_t> runtime_args;

        // Input tensor work distribution
        runtime_args.push_back(rank);
        for (uint32_t d = 0; d < rank; ++d) {
            runtime_args.push_back(gcores_info.dim_pages[gcore_idx][d]);
            runtime_args.push_back(gcores_info.dim_offsets[gcore_idx][d]);
            runtime_args.push_back(gcores_info.dim_strides[gcore_idx][d]);
        }

        // Output tensor strides (pages/offsets same as input for first rank-1 dims, different strides)
        for (uint32_t d = 0; d < rank; ++d) {
            runtime_args.push_back(output_gcores_info.dim_strides[gcore_idx][d]);
        }

        // Gcore sequence information
        runtime_args.push_back(gcore_idx);     // gcore_sequence_id
        runtime_args.push_back(total_gcores);  // total_gcores

        // First gcore coordinate (for ALL gcores to send to, including first gcore writing to itself)
        const auto& first_gcore = gcores_info.gcores[0];
        const auto& first_coord = first_gcore.global_coord;

        runtime_args.push_back(first_coord.dims());  // first_gcore_coord_rank
        for (size_t d = 0; d < first_coord.dims(); ++d) {
            runtime_args.push_back(first_coord[d]);
        }

        tt::tt_metal::experimental::udm::SetMeshKernelRuntimeArgs(
            mesh_builder, program, dataflow_kernel_id, gcore, runtime_args);

        // Compute kernel runtime args (similar to dataflow: rank, dim_pages, dim_offsets, dim_strides)
        std::vector<uint32_t> compute_runtime_args;
        compute_runtime_args.push_back(gcore_idx == 0 ? 1 : 0);  // is_first_gcore

        // Input tensor work distribution
        compute_runtime_args.push_back(rank);
        for (uint32_t d = 0; d < rank; ++d) {
            compute_runtime_args.push_back(gcores_info.dim_pages[gcore_idx][d]);
            compute_runtime_args.push_back(gcores_info.dim_offsets[gcore_idx][d]);
            compute_runtime_args.push_back(gcores_info.dim_strides[gcore_idx][d]);
        }

        tt::tt_metal::experimental::udm::SetMeshKernelRuntimeArgs(
            mesh_builder, program, compute_kernel_id, gcore, compute_runtime_args);
    }

    return program;
}

/**
 * @brief Validate width reduction result for N-dimensional tensors
 *
 * For an N-D input tensor of shape (D0, D1, ..., D_{N-2}, D_{N-1}),
 * reduction on the last dimension produces output shape (D0, D1, ..., D_{N-2}, 1).
 *
 * Examples:
 *   - input (2, 32, 128) → output (2, 32, 1) where output[i,j] = sum(input[i,j,:])
 *   - input (3, 4, 5, 100) → output (3, 4, 5, 1) where output[i,j,k] = sum(input[i,j,k,:])
 */
void validate(const ttnn::Tensor& input_tensor, const ttnn::Tensor& output_tensor) {
    auto* mesh_device = input_tensor.device();

    // Aggregate tensors from distributed format
    auto input_composer = create_width_sharded_mesh_composer(mesh_device, input_tensor.padded_shape().rank());
    auto input_data = ttnn::distributed::aggregate_tensor(input_tensor, *input_composer).to_vector<uint16_t>();

    auto output_composer = create_width_sharded_mesh_composer(mesh_device, output_tensor.padded_shape().rank());
    auto output_data = ttnn::distributed::aggregate_tensor(output_tensor, *output_composer).to_vector<uint16_t>();

    // Get input shape
    const auto& input_shape = input_tensor.padded_shape();
    uint32_t rank = input_shape.rank();
    TT_ASSERT(rank >= 1, "Tensor must have at least 1 dimension");

    // Get output shape
    const auto& output_shape = output_tensor.padded_shape();

    // Log shapes
    std::string input_shape_str = "(";
    std::string output_shape_str = "(";
    for (uint32_t i = 0; i < rank; ++i) {
        if (i > 0) {
            input_shape_str += ", ";
            output_shape_str += ", ";
        }
        input_shape_str += std::to_string(input_shape[i]);
        output_shape_str += std::to_string(output_shape[i]);
    }
    input_shape_str += ")";
    output_shape_str += ")";

    log_info(tt::LogTest, "Input shape: {}", input_shape_str);
    log_info(tt::LogTest, "Output shape: {}", output_shape_str);

    // Verify all dimensions except the last match
    for (uint32_t i = 0; i < rank - 1; ++i) {
        TT_ASSERT(
            output_shape[i] == input_shape[i],
            "Output shape dimension {} must match input (expected {}, got {})",
            i,
            input_shape[i],
            output_shape[i]);
    }

    // Compute number of "rows" to validate (product of all dims except last)
    uint32_t num_rows = 1;
    for (uint32_t i = 0; i < rank - 1; ++i) {
        num_rows *= input_shape[i];
    }

    uint32_t input_last_dim = input_shape[rank - 1];
    uint32_t output_last_dim = output_shape[rank - 1];

    log_info(tt::LogTest, "Validating {} rows, reducing {} elements per row", num_rows, input_last_dim);

    // Validate each "row" (combination of indices in all dims except last)
    uint32_t mismatches = 0;
    const uint32_t max_print_mismatches = 10;

    for (uint32_t row = 0; row < num_rows; ++row) {
        // Compute expected sum for this row (sum across last dimension)
        uint64_t expected_sum = 0;
        for (uint32_t last_idx = 0; last_idx < input_last_dim; ++last_idx) {
            uint32_t input_idx = row * input_last_dim + last_idx;
            expected_sum += input_data[input_idx];
        }

        // Get actual value from output (first element along last dimension)
        uint32_t output_idx = row * output_last_dim;
        uint64_t actual_sum = output_data[output_idx];

        // Compare
        if (expected_sum != actual_sum) {
            if (mismatches < max_print_mismatches) {
                log_error(tt::LogTest, "Mismatch at row {}: expected={}, actual={}", row, expected_sum, actual_sum);
            }
            mismatches++;
        }
    }

    if (mismatches > 0) {
        log_error(tt::LogTest, "Total mismatches: {} / {}", mismatches, num_rows);
        TT_THROW("Width reduction validation failed: output does not match expected reduced values");
    }

    log_info(tt::LogTest, "Width reduction validation passed: all {} rows match!", num_rows);
}

/**
 * @brief Run width reduction test
 */
void run_width_reduction_test(
    tt::tt_metal::distributed::MeshDevice* mesh_device,
    const tt::tt_metal::Shape& global_shape,
    const tt::tt_metal::Shape& local_shape) {
    // Create input tensor with known values for easy validation
    auto input_tensor = create_width_sharded_tensor(mesh_device, global_shape, local_shape);

    // Create output tensor for reduced result
    // Output shape is same as input except last dimension is reduced to 32 (one tile)
    tt::tt_metal::Shape output_global_shape = global_shape;
    tt::tt_metal::Shape output_local_shape = local_shape;
    output_global_shape[-1] = 32;  // Reduce last dimension to single tile
    output_local_shape[-1] = 32;

    auto output_tensor = create_width_sharded_tensor(mesh_device, output_global_shape, output_local_shape);

    // Create reduction program
    auto program = create_program(input_tensor, output_tensor);

    // Run program
    auto* tensor_mesh_device = input_tensor.device();
    ASSERT_NE(tensor_mesh_device, nullptr) << "Tensor must be on device";
    run_program(input_tensor, tensor_mesh_device, program);

    // Validate result
    validate(input_tensor, output_tensor);
}

/**
 * @brief Test width reduction with MeshGcoreAccessor
 *
 * Setup:
 * - Mesh: 1×4 (4 devices in a row)
 * - Input tensor: width-sharded across devices
 * - Each gcore reads its portion, reduces locally
 * - Sends partial result to next gcore via MeshGcoreAccessor
 * - Last gcore writes final sum to output
 *
 * Operation:
 * - Gcore 0: local_sum[0] → send to Gcore 1
 * - Gcore 1: local_sum[1] + received → send to Gcore 2
 * - Gcore 2: local_sum[2] + received → send to Gcore 3
 * - Gcore 3: local_sum[3] + received → write to output
 */
using MeshDevice1x4Fabric2DUDMFixture = tt::tt_metal::MeshDevice1x4Fabric2DUDMFixture;

TEST_F(MeshDevice1x4Fabric2DUDMFixture, TestWidthReduction2D_Small) {
    // Small 2D tensor: (4, 16) tiles = (128, 512) elements
    // Width-sharded: each device gets (4, 4) tiles = (128, 128) elements
    tt::tt_metal::Shape global_shape({128, 512});
    tt::tt_metal::Shape local_shape({128, 128});

    run_width_reduction_test(mesh_device_.get(), global_shape, local_shape);
}

TEST_F(MeshDevice1x4Fabric2DUDMFixture, TestWidthReduction2D_Medium) {
    // Medium 2D tensor: (8, 32) tiles
    tt::tt_metal::Shape global_shape({256, 1024});
    tt::tt_metal::Shape local_shape({256, 256});

    run_width_reduction_test(mesh_device_.get(), global_shape, local_shape);
}

}  // namespace tt::tt_metal::experimental::udm_tests
