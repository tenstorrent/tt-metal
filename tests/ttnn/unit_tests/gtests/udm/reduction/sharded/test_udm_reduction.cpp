// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

/**
 * Width Reduction Test using LayerNorm-Style Distributed Pattern
 *
 * This test implements a multi-device width reduction using the pattern from
 * layernorm_op_multi_core_sharded.cpp, adapted for UDM APIs.
 *
 * Key changes from original simple reduction:
 * 1. Uses layernorm-style CB layout (CB 0, 2, 8, 9, 10, 15, 16)
 * 2. Sender/receiver kernel split (like layernorm mcast sender/receiver)
 * 3. Distributed reduction: partial → gather → global reduce → distribute
 * 4. Uses flattened mesh dimensions (mesh_shape[1] for width)
 * 5. Gets global shape from mesh_tensor_builder, not local tensor
 *
 * Pattern (simplified from layernorm):
 * - Phase 1: All cores do local partial reduction (reduce across width)
 * - Phase 2: Sender coordinates and gathers all partials via UDM async_read
 * - Phase 3: Sender performs global reduction across all core partials
 * - Phase 4: Sender distributes results to all cores (unicast, TODO: mcast)
 *
 * TODO items:
 * - Replace unicast with mcast when UDM API supports it
 * - Create actual sender/receiver/compute kernel implementations
 */

#include <gtest/gtest.h>
#include <cmath>

#include "tests/tt_metal/tt_metal/common/multi_device_fixture.hpp"
#include "tests/ttnn/unit_tests/gtests/udm/test_udm_utils.hpp"
#include "tt_metal/programming_examples/matmul/matmul_common/bmm_op.hpp"

#include "ttnn/operations/core/core.hpp"  // for ttnn::to_memory_config

#include "tt_metal/experimental/udm/mesh_kernel.hpp"
#include "tt_metal/experimental/udm/mesh_utils.hpp"
#include "tt_metal/experimental/udm/mesh_circular_buffer.hpp"
#include "tt_metal/experimental/udm/mesh_semaphore.hpp"
#include "tt_metal/api/tt-metalium/bfloat16.hpp"

namespace tt::tt_metal::experimental::udm_tests {
namespace {

/**
 * @brief Create UDM program for width reduction using LayerNorm-style pattern
 * Pattern: Distributed reduction with sender/receiver coordination
 *
 * @param input_mesh_tensor_builder Builder for input tensor (contains mesh tensor shape info)
 * @param output_mesh_tensor_builder Builder for output tensor
 */
tt::tt_metal::experimental::udm::MeshProgram create_program(
    tt::tt_metal::experimental::udm::MeshTensorBuilder& input_mesh_tensor_builder,
    tt::tt_metal::experimental::udm::MeshTensorBuilder& output_mesh_tensor_builder) {
    auto& mesh_builder = input_mesh_tensor_builder.mesh_builder();

    // Get underlying buffers from mesh tensor builders for globally allocated CBs
    auto& input_buffer = *input_mesh_tensor_builder.mesh_buffer().get_reference_buffer();
    auto& output_buffer = *output_mesh_tensor_builder.mesh_buffer().get_reference_buffer();
    auto program = tt::tt_metal::experimental::udm::CreateMeshProgram(mesh_builder);

    // ===== GRID SETUP - Use flattened mesh =====
    const auto& mesh_shape = mesh_builder.get_flattened_mesh();  // Mesh with dims
    const auto& gcores = mesh_builder.get_all_gcores_in_mesh();  // Array of GlobalCores

    // For row-wise reduction: cores in the same row reduce together
    uint32_t num_cores_y = mesh_shape[-2];  // Number of rows in mesh (independent reduction groups)
    uint32_t num_cores_x = mesh_shape[-1];  // Number of cores per row (reduction dimension)

    TT_FATAL(num_cores_x > 1, "Need multiple cores per row for reduction");

    // ===== GET SHAPE FROM MESH TENSOR BUILDER =====
    auto shape_in_pages = input_mesh_tensor_builder.get_mesh_tensor_shape_in_pages();

    // For tiled layout, pages are tiles
    // Shape in pages already gives us tile dimensions
    uint32_t num_height_tiles = shape_in_pages[-2];
    uint32_t num_width_tiles = shape_in_pages[-1];

    // ===== WORK DISTRIBUTION =====
    // Height is distributed across Y dimension (rows)
    // Width is distributed across X dimension (columns)
    uint32_t block_ht = num_height_tiles / num_cores_y;  // Height per row
    uint32_t block_wt = num_width_tiles / num_cores_x;   // Width per column
    uint32_t in0_block_tiles = block_wt * block_ht;

    // For distributed global reduction: each core in a row reduces a subset of rows
    // Ensure each core has at least 1 row to work with
    TT_FATAL(
        block_ht >= num_cores_x,
        "Insufficient rows for distributed reduction: block_ht={} must be >= num_cores_x={}. "
        "Either increase tensor height or reduce num_cores_x.",
        block_ht,
        num_cores_x);

    // For distributing final reduced results among all-to-all workers in a row
    // (This is for the sender to distribute work when gathering/distributing results)
    uint32_t num_rows_per_worker = tt::div_up(block_ht, num_cores_x);
    uint32_t remainder = block_ht % num_rows_per_worker;
    uint32_t num_rows_per_worker_last = (remainder == 0) ? num_rows_per_worker : remainder;

    // ===== CIRCULAR BUFFERS (consecutive indices) =====
    tt::DataFormat data_format = tt::DataFormat::Float16_b;  // bfloat16
    uint32_t single_tile_size = tt::tile_size(data_format);
    uint32_t bfloat16_tile_size = tt::tile_size(tt::DataFormat::Float16_b);

    // CB 0: Input (globally allocated to sharded buffer)
    uint32_t in0_cb_index = tt::CBIndex::c_0;
    tt::tt_metal::CircularBufferConfig in0_cb_config =
        tt::tt_metal::CircularBufferConfig(in0_block_tiles * single_tile_size, {{in0_cb_index, data_format}})
            .set_page_size(in0_cb_index, single_tile_size)
            .set_globally_allocated_address(input_buffer);
    auto cb_in0 = tt::tt_metal::experimental::udm::CreateMeshCircularBuffer(mesh_builder, program, in0_cb_config);

    // CB 1: Scaler
    uint32_t scaler_cb_index = tt::CBIndex::c_1;
    tt::tt_metal::CircularBufferConfig scaler_cb_config =
        tt::tt_metal::CircularBufferConfig(bfloat16_tile_size, {{scaler_cb_index, tt::DataFormat::Float16_b}})
            .set_page_size(scaler_cb_index, bfloat16_tile_size);
    auto cb_scaler = tt::tt_metal::experimental::udm::CreateMeshCircularBuffer(mesh_builder, program, scaler_cb_config);

    // CB 2: Partial reduction per core
    uint32_t cb_partial_index = tt::CBIndex::c_2;
    tt::tt_metal::CircularBufferConfig cb_partial_config =
        tt::tt_metal::CircularBufferConfig(block_ht * single_tile_size, {{cb_partial_index, data_format}})
            .set_page_size(cb_partial_index, single_tile_size);
    auto cb_partial =
        tt::tt_metal::experimental::udm::CreateMeshCircularBuffer(mesh_builder, program, cb_partial_config);

    // CB 3: Global reduced result
    uint32_t cb_reduced_index = tt::CBIndex::c_3;
    tt::tt_metal::CircularBufferConfig cb_reduced_config =
        tt::tt_metal::CircularBufferConfig(num_rows_per_worker * single_tile_size, {{cb_reduced_index, data_format}})
            .set_page_size(cb_reduced_index, single_tile_size);
    auto cb_reduced =
        tt::tt_metal::experimental::udm::CreateMeshCircularBuffer(mesh_builder, program, cb_reduced_config);

    // CB 4: External (for gathering remote partials)
    uint32_t cb_external_index = tt::CBIndex::c_4;
    tt::tt_metal::CircularBufferConfig cb_external_config =
        tt::tt_metal::CircularBufferConfig(num_cores_x * single_tile_size, {{cb_external_index, data_format}})
            .set_page_size(cb_external_index, single_tile_size);
    auto cb_external =
        tt::tt_metal::experimental::udm::CreateMeshCircularBuffer(mesh_builder, program, cb_external_config);

    // CB 5: Output (globally allocated, dataflow gathers directly here)
    uint32_t output_cb_index = tt::CBIndex::c_5;
    uint32_t output_tiles = block_ht;
    tt::tt_metal::CircularBufferConfig output_cb_config =
        tt::tt_metal::CircularBufferConfig(output_tiles * single_tile_size, {{output_cb_index, data_format}})
            .set_page_size(output_cb_index, single_tile_size)
            .set_globally_allocated_address(output_buffer);
    auto cb_output = tt::tt_metal::experimental::udm::CreateMeshCircularBuffer(mesh_builder, program, output_cb_config);

    // ===== SEMAPHORES =====
    // GlobalSemaphore provides the same L1 address on all devices
    // Initialize to 0 so increment/wait semantics work correctly
    auto reduce_sender_semaphore_addr = tt::tt_metal::experimental::udm::CreateMeshSemaphore(mesh_builder, program, 0);
    auto reduce_receiver_semaphore_addr =
        tt::tt_metal::experimental::udm::CreateMeshSemaphore(mesh_builder, program, 0);

    // ===== CORE SETS =====
    // Each row has independent reduction: first core in row is sender, rest are receivers
    // No reduction between rows - each row operates independently

    std::vector<tt::tt_metal::experimental::udm::GlobalCore> sender_gcores;
    std::vector<tt::tt_metal::experimental::udm::GlobalCore> receiver_gcores;
    std::vector<tt::tt_metal::experimental::udm::GlobalCore> all_gcores;

    for (uint32_t y = 0; y < num_cores_y; ++y) {
        // First core in this row is sender
        const auto& sender_gcore = gcores[y * num_cores_x + 0];
        sender_gcores.push_back(sender_gcore);
        all_gcores.push_back(sender_gcore);

        // Rest of cores in this row are receivers
        for (uint32_t x = 1; x < num_cores_x; ++x) {
            const auto& receiver_gcore = gcores[y * num_cores_x + x];
            receiver_gcores.push_back(receiver_gcore);
            all_gcores.push_back(receiver_gcore);
        }
    }

    // ===== COMPILE-TIME ARGS =====
    // For SUM reduction, scaler should be 1.0 (not 1/W which is for MEAN)
    bfloat16 bfloat_scaler = bfloat16(1.0f);
    uint32_t packed_winv = pack_two_bfloat16_into_uint32({bfloat_scaler, bfloat_scaler});

    // Get coordinate dimensions from any gcore
    uint32_t coord_dims = gcores[0].global_coord.dims();

    // Sender kernel args: sem_recv_addr, sem_send_addr, num_blocks, block_ht, rows_per_worker, rows_per_worker_last,
    // winv, coord_dims
    std::vector<uint32_t> reader_sender_compile_time_args = {
        reduce_receiver_semaphore_addr.at(0),  // GlobalSemaphore address (same on all devices)
        reduce_sender_semaphore_addr.at(0),
        (uint32_t)num_cores_x,
        (uint32_t)block_ht,
        (uint32_t)num_rows_per_worker,
        (uint32_t)num_rows_per_worker_last,
        packed_winv,
        (uint32_t)coord_dims};

    // Receiver kernel args: sem_recv_addr, sem_send_addr, num_blocks, block_ht, rows_per_worker, rows_per_worker_last,
    // winv, coord_dims
    std::vector<uint32_t> reader_receiver_compile_time_args = {
        reduce_receiver_semaphore_addr.at(0),  // GlobalSemaphore address (same on all devices)
        reduce_sender_semaphore_addr.at(0),
        (uint32_t)num_cores_x,
        (uint32_t)block_ht,
        (uint32_t)num_rows_per_worker,
        (uint32_t)num_rows_per_worker_last,
        packed_winv,
        (uint32_t)coord_dims};

    // ===== CREATE READER KERNELS =====
    auto reader_sender_kernel_id = tt::tt_metal::experimental::udm::CreateMeshKernel(
        mesh_builder,
        program,
        "tests/ttnn/unit_tests/gtests/udm/reduction/sharded/kernels/reader_sender_unary_sharded_reduce.cpp",
        sender_gcores,
        tt::tt_metal::DataMovementConfig{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
            .noc = tt::tt_metal::NOC::RISCV_0_default,
            .compile_args = reader_sender_compile_time_args});

    auto reader_receiver_kernel_id = tt::tt_metal::experimental::udm::CreateMeshKernel(
        mesh_builder,
        program,
        "tests/ttnn/unit_tests/gtests/udm/reduction/sharded/kernels/reader_receiver_unary_sharded_reduce.cpp",
        receiver_gcores,
        tt::tt_metal::DataMovementConfig{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
            .noc = tt::tt_metal::NOC::RISCV_0_default,
            .compile_args = reader_receiver_compile_time_args});

    // ===== COMPUTE COMPILE-TIME ARGS =====
    // Compute kernel args: num_blocks, block_ht, block_wt
    std::vector<uint32_t> compute_all_to_all_args = {(uint32_t)num_cores_x, (uint32_t)block_ht, (uint32_t)block_wt};

    // ===== CREATE COMPUTE KERNEL =====
    auto compute_kernel_id = tt::tt_metal::experimental::udm::CreateMeshKernel(
        mesh_builder,
        program,
        "tests/ttnn/unit_tests/gtests/udm/reduction/sharded/kernels/compute_sharded_reduce.cpp",
        all_gcores,
        tt::tt_metal::ComputeConfig{.fp32_dest_acc_en = true, .compile_args = compute_all_to_all_args});

    // ===== SET RUNTIME ARGS =====
    // Set runtime args for each row independently
    for (uint32_t y = 0; y < num_cores_y; ++y) {
        // Sender core for this row
        const auto& sender_gcore = gcores[y * num_cores_x + 0];

        // Collect all cores' gcores in this row (sender + receivers)
        std::vector<const tt::tt_metal::experimental::udm::GlobalCore*> row_gcores;
        row_gcores.reserve(num_cores_x);
        for (uint32_t x = 0; x < num_cores_x; ++x) {
            row_gcores.push_back(&gcores[y * num_cores_x + x]);
        }

        // Helper lambda to add all coordinates to runtime args
        auto add_all_coords = [&](std::vector<uint32_t>& args) {
            for (const auto* gcore : row_gcores) {
                const auto& coord = gcore->global_coord;
                for (size_t d = 0; d < coord_dims; ++d) {
                    args.push_back(coord[d]);
                }
            }
        };

        // Sender runtime args: coordinates of ALL cores in this row
        // Format: [coord0_d0, coord0_d1, ..., coord1_d0, coord1_d1, ..., ...]
        std::vector<uint32_t> sender_runtime_args;
        add_all_coords(sender_runtime_args);

        tt::tt_metal::experimental::udm::SetMeshKernelRuntimeArgs(
            mesh_builder, program, reader_sender_kernel_id, sender_gcore, sender_runtime_args);

        // Sender compute runtime args
        std::vector<uint32_t> compute_sender_rt_args = {(uint32_t)block_wt, (uint32_t)num_rows_per_worker};
        tt::tt_metal::experimental::udm::SetMeshKernelRuntimeArgs(
            mesh_builder, program, compute_kernel_id, sender_gcore, compute_sender_rt_args);

        // Receiver runtime args: core index, then coordinates of ALL cores
        // Format: [core_idx, coord0_d0, coord0_d1, ..., coord1_d0, coord1_d1, ..., ...]
        for (uint32_t receiver_idx = 0; receiver_idx < num_cores_x - 1; ++receiver_idx) {
            const auto* receiver_gcore = row_gcores[receiver_idx + 1];
            std::vector<uint32_t> receiver_runtime_args;

            // Core index (1-based for receivers, 0 is sender)
            receiver_runtime_args.push_back(receiver_idx + 1);

            // All coordinates
            add_all_coords(receiver_runtime_args);

            tt::tt_metal::experimental::udm::SetMeshKernelRuntimeArgs(
                mesh_builder, program, reader_receiver_kernel_id, *receiver_gcore, receiver_runtime_args);

            // Receiver compute runtime args
            // Last receiver gets num_rows_per_worker_last
            uint32_t receiver_num_rows =
                (receiver_idx == num_cores_x - 2) ? num_rows_per_worker_last : num_rows_per_worker;
            std::vector<uint32_t> compute_receiver_rt_args = {(uint32_t)block_wt, receiver_num_rows};
            tt::tt_metal::experimental::udm::SetMeshKernelRuntimeArgs(
                mesh_builder, program, compute_kernel_id, *receiver_gcore, compute_receiver_rt_args);
        }
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

    // Convert from sharded to interleaved before aggregation
    // The shard spec is designed for local tensor shape, not global shape
    tt::tt_metal::MemoryConfig interleaved_mem_config(
        tt::tt_metal::TensorMemoryLayout::INTERLEAVED, tt::tt_metal::BufferType::L1);
    auto input_interleaved = ttnn::to_memory_config(input_tensor, interleaved_mem_config);
    auto output_interleaved = ttnn::to_memory_config(output_tensor, interleaved_mem_config);

    // Aggregate tensors from distributed format
    // Input is block-sharded, output is height-sharded (replicated on width)
    auto input_composer = create_block_sharded_mesh_composer(mesh_device, input_interleaved.padded_shape().rank());
    auto input_aggregated = ttnn::distributed::aggregate_tensor(input_interleaved, *input_composer);
    auto input_data = input_aggregated.to_vector<bfloat16>();

    // Output is height-distributed (Shard{height}, Replicate{})
    // Composer produces [H, W * num_replicas] - we use first W_original elements per row
    auto output_composer = create_height_sharded_mesh_composer(mesh_device, output_interleaved.padded_shape().rank());
    auto output_aggregated = ttnn::distributed::aggregate_tensor(output_interleaved, *output_composer);
    auto output_data = output_aggregated.to_vector<bfloat16>();

    // Get dimensions from aggregated tensors
    // Input is 2D: [H, W]
    uint32_t global_height = input_aggregated.padded_shape()[-2];
    uint32_t global_width = input_aggregated.padded_shape()[-1];

    // Output is 2D after aggregation: [H, W * num_replicas]
    // We use the first replica, so stride is the full aggregated width
    uint32_t output_aggregated_width = output_aggregated.padded_shape()[-1];

    // Log shapes
    log_info(tt::LogTest, "Aggregated input shape: {}", input_aggregated.padded_shape());
    log_info(tt::LogTest, "Aggregated output shape: {} (using first replica)", output_aggregated.padded_shape());

    // Compute number of "rows" to validate
    uint32_t num_rows = global_height;

    log_info(tt::LogTest, "Validating {} rows, reducing {} elements per row", num_rows, global_width);

    // Build expected and actual vectors for PCC comparison
    std::vector<bfloat16> expected_values;
    std::vector<bfloat16> actual_values;
    expected_values.reserve(num_rows);
    actual_values.reserve(num_rows);

    for (uint32_t row = 0; row < num_rows; ++row) {
        // Compute expected sum for this row (sum across global width)
        float expected_sum = 0.0f;
        for (uint32_t last_idx = 0; last_idx < global_width; ++last_idx) {
            uint32_t input_idx = row * global_width + last_idx;
            expected_sum += float(input_data[input_idx]);
        }
        expected_values.push_back(bfloat16(expected_sum));

        // Get actual value from first replica (first element of row in aggregated output)
        // Output layout: [H, W * num_replicas] -> index = row * aggregated_width + 0
        uint32_t output_idx = row * output_aggregated_width;
        actual_values.push_back(output_data[output_idx]);

        // Debug: Print first 16 rows and any mismatches
        float actual_f = float(output_data[output_idx]);
        float expected_f = expected_sum;
        float diff = std::abs(actual_f - expected_f);
        if (row < 16 || diff > 0.1f * std::abs(expected_f)) {
            log_info(
                tt::LogTest, "  Row {}: expected={:.2f}, actual={:.2f}, diff={:.2f}", row, expected_f, actual_f, diff);
        }
    }

    // Check PCC between expected and actual values
    float pcc = check_bfloat16_vector_pcc(expected_values, actual_values);
    const float pcc_threshold = 0.9999f;

    log_info(tt::LogTest, "PCC: {:.6f} (threshold: {:.2f})", pcc, pcc_threshold);

    // Use !(pcc >= threshold) to also catch NaN (NaN comparisons always return false)
    if (!(pcc >= pcc_threshold)) {
        TT_THROW("Width reduction validation failed: PCC {:.6f} below threshold {:.2f}", pcc, pcc_threshold);
    }

    log_info(tt::LogTest, "Width reduction validation passed: PCC={:.6f} for {} rows", pcc, num_rows);
}

/**
 * @brief Run width reduction test
 * @param grid_size The grid shape {num_cores_x, num_cores_y} to use for sharding within each device
 */
void run_width_reduction_test(
    tt::tt_metal::distributed::MeshDevice* mesh_device,
    const tt::tt_metal::Shape& global_shape,
    const tt::tt_metal::Shape& local_shape,
    std::pair<uint32_t, uint32_t> grid_size) {
    // Create input tensor - mesh block-distributed, grid block-sharded
    auto input_tensor =
        create_block_distributed_block_sharded_bfloat16_tensor(mesh_device, global_shape, local_shape, grid_size);

    // Create output tensor for reduced result
    // Output shape is same as input except last dimension is reduced to 32 (one tile)
    // Output is mesh height-distributed (replicated on width), grid height-sharded
    tt::tt_metal::Shape output_global_shape = global_shape;
    tt::tt_metal::Shape output_local_shape = local_shape;
    output_global_shape[-1] = 32;  // Reduce last dimension to single tile
    output_local_shape[-1] = 32;

    auto output_tensor = create_height_distributed_height_sharded_bfloat16_tensor(
        mesh_device, output_global_shape, output_local_shape, grid_size);

    // Build tensor builders from tensors (extracts mesh tensor shape info)
    auto input_mesh_tensor_builder = create_tensor_builder(input_tensor);
    auto output_mesh_tensor_builder = create_tensor_builder(output_tensor);

    // Log tensor shape info for debugging
    log_tensor_shape_info(input_mesh_tensor_builder, input_tensor);

    // Create reduction program using tensor builders (not raw tensors/buffers)
    auto program = create_program(input_mesh_tensor_builder, output_mesh_tensor_builder);

    // Run program
    auto* tensor_mesh_device = input_tensor.device();
    ASSERT_NE(tensor_mesh_device, nullptr) << "Tensor must be on device";
    run_program(input_tensor, tensor_mesh_device, program);

    // Validate result
    validate(input_tensor, output_tensor);
}

}  // namespace

/**
 * @brief Test width reduction with LayerNorm-style distributed pattern
 *
 * Setup:
 * - Mesh: MxN (e.g., 1×4 for 4 devices in a row)
 * - Input tensor: block-sharded, bfloat16
 * - Constraint: block_ht (height per mesh row) >= num_cores_x (mesh width)
 *   to ensure each core has at least one row for global reduction
 *
 * Pattern (LayerNorm-style distributed reduction):
 * Phase 1: All cores - Local partial reduction
 *   - Each core reduces its block (block_ht × block_wt) across width → block_ht partial tiles
 *
 * Phase 2: All cores - Distributed global reduction
 *   - Work is distributed: each core reduces different rows
 *   - Core i reduces rows [i*k, (i+1)*k) across all cores' partials
 *   - Each core reads assigned rows' partials from ALL cores and reduces them
 *
 * Phase 3: Sender - Gather and distribute
 *   - Sender gathers reduced results from all cores
 *   - Sender distributes complete result to all cores (TODO: use mcast when available)
 */
using MeshDevice1x4Fabric2DUDMFixture = tt::tt_metal::MeshDevice1x4Fabric2DUDMFixture;

TEST_F(MeshDevice1x4Fabric2DUDMFixture, TestWidthReduction2D_Small) {
    // Small 2D tensor: (8, 16) tiles = (256, 512) elements
    // Mesh: 1×4, each device gets (8, 4) tiles = (256, 128) elements locally
    // Grid: 2x1 - shard shape = (256, 64) = (8, 2) tiles per core
    tt::tt_metal::Shape global_shape({256, 512});
    tt::tt_metal::Shape local_shape({256, 128});
    std::pair<uint32_t, uint32_t> grid_size = {2, 1};

    run_width_reduction_test(mesh_device_.get(), global_shape, local_shape, grid_size);
}

TEST_F(MeshDevice1x4Fabric2DUDMFixture, TestWidthReduction2D_Medium) {
    // Medium 2D tensor: (16, 32) tiles = (512, 1024) elements
    // Mesh: 1×4, each device gets (16, 8) tiles = (512, 256) elements locally
    // Grid: 2x2 - shard shape = (256, 128) = (8, 4) tiles per core
    tt::tt_metal::Shape global_shape({512, 1024});
    tt::tt_metal::Shape local_shape({512, 256});
    std::pair<uint32_t, uint32_t> grid_size = {2, 2};

    run_width_reduction_test(mesh_device_.get(), global_shape, local_shape, grid_size);
}

TEST_F(MeshDevice1x4Fabric2DUDMFixture, TestWidthReduction2D_Large) {
    // Large 2D tensor: (64, 128) tiles = (2048, 4096) elements
    // Mesh: 1×4, each device gets (64, 32) tiles = (2048, 1024) elements locally
    // Grid: 4x4 - shard shape = (512, 256) = (16, 8) tiles per core
    // Constraint: block_ht = 64/4 = 16, num_cores_x = 4*4 = 16, so 16 >= 16 ✓
    tt::tt_metal::Shape global_shape({2048, 4096});
    tt::tt_metal::Shape local_shape({2048, 1024});
    std::pair<uint32_t, uint32_t> grid_size = {4, 4};

    run_width_reduction_test(mesh_device_.get(), global_shape, local_shape, grid_size);
}

// ============================================================================
// 2x4 Mesh Tests - Height sharded across 2 devices, width across 4 devices
// ============================================================================
using MeshDevice2x4Fabric2DUDMFixture = tt::tt_metal::MeshDevice2x4Fabric2DUDMFixture;

TEST_F(MeshDevice2x4Fabric2DUDMFixture, TestWidthReduction2D_Small) {
    // Small 2D tensor: (16, 16) tiles = (512, 512) elements
    // Mesh: 2×4, each device gets (8, 4) tiles = (256, 128) elements locally
    // Grid: 2x1 - shard shape = (256, 64) = (8, 2) tiles per core
    // Constraint: block_ht = 16/2 = 8, num_cores_x = 4*2 = 8, so 8 >= 8 ✓
    tt::tt_metal::Shape global_shape({512, 512});
    tt::tt_metal::Shape local_shape({256, 128});
    std::pair<uint32_t, uint32_t> grid_size = {2, 1};

    run_width_reduction_test(mesh_device_.get(), global_shape, local_shape, grid_size);
}

TEST_F(MeshDevice2x4Fabric2DUDMFixture, TestWidthReduction2D_Medium) {
    // Medium 2D tensor: (32, 32) tiles = (1024, 1024) elements
    // Mesh: 2×4, each device gets (16, 8) tiles = (512, 256) elements locally
    // Grid: 2x2 - shard shape = (256, 128) = (8, 4) tiles per core
    // Constraint: block_ht = 32/4 = 8, num_cores_x = 4*2 = 8, so 8 >= 8 ✓
    tt::tt_metal::Shape global_shape({1024, 1024});
    tt::tt_metal::Shape local_shape({512, 256});
    std::pair<uint32_t, uint32_t> grid_size = {2, 2};

    run_width_reduction_test(mesh_device_.get(), global_shape, local_shape, grid_size);
}

TEST_F(MeshDevice2x4Fabric2DUDMFixture, TestWidthReduction2D_Large) {
    // Large 2D tensor: (128, 128) tiles = (4096, 4096) elements
    // Mesh: 2×4, each device gets (64, 32) tiles = (2048, 1024) elements locally
    // Grid: 4x4 - shard shape = (512, 256) = (16, 8) tiles per core
    // Constraint: block_ht = 128/8 = 16, num_cores_x = 4*4 = 16, so 16 >= 16 ✓
    tt::tt_metal::Shape global_shape({4096, 4096});
    tt::tt_metal::Shape local_shape({2048, 1024});
    std::pair<uint32_t, uint32_t> grid_size = {4, 4};

    run_width_reduction_test(mesh_device_.get(), global_shape, local_shape, grid_size);
}

}  // namespace tt::tt_metal::experimental::udm_tests
