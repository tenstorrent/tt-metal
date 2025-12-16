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
#include <random>
#include <cmath>

#include "tests/tt_metal/tt_metal/common/multi_device_fixture.hpp"
#include "tests/tt_metal/udm/test_udm_utils.hpp"
#include "tt_metal/programming_examples/matmul/matmul_common/bmm_op.hpp"

#include "tt_metal/udm/mesh_kernel.hpp"
#include "tt_metal/udm/mesh_utils.hpp"
#include "tt_metal/udm/mesh_circular_buffer.hpp"
#include "tt_metal/udm/mesh_semaphore.hpp"
#include "tt_metal/api/tt-metalium/bfloat16.hpp"

namespace tt::tt_metal::experimental::udm_tests {

/**
 * @brief Create UDM program for width reduction using LayerNorm-style pattern
 * Pattern: Distributed reduction with sender/receiver coordination
 */
tt::tt_metal::experimental::udm::MeshProgram create_program(
    tt::tt_metal::experimental::udm::MeshTensorBuilder& input_mesh_tensor_builder,
    tt::tt_metal::Buffer& input_buffer,
    tt::tt_metal::Buffer& output_buffer) {
    auto& mesh_builder = input_mesh_tensor_builder.mesh_builder();
    auto program = tt::tt_metal::experimental::udm::CreateMeshProgram(mesh_builder);

    // ===== GRID SETUP - Use flattened mesh =====
    const auto& mesh_shape = mesh_builder.get_flattened_mesh();  // Mesh with dims
    const auto& gcores = mesh_builder.get_all_gcores_in_mesh();  // Array of Gcores

    // For row-wise reduction: cores in the same row reduce together
    uint32_t num_cores_y = mesh_shape[-2];  // Number of rows in mesh (independent reduction groups)
    uint32_t num_cores_x = mesh_shape[-1];  // Number of cores per row (reduction dimension)

    TT_ASSERT(num_cores_x > 1, "Need multiple cores per row for reduction");

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
    auto reduce_sender_semaphore_id =
        tt::tt_metal::experimental::udm::CreateMeshSemaphore(mesh_builder, program, INVALID);
    auto reduce_receiver_semaphore_id =
        tt::tt_metal::experimental::udm::CreateMeshSemaphore(mesh_builder, program, INVALID);

    // ===== CORE SETS =====
    // Each row has independent reduction: first core in row is sender, rest are receivers
    // No reduction between rows - each row operates independently

    std::vector<tt::tt_metal::experimental::udm::Gcore> sender_gcores;
    std::vector<tt::tt_metal::experimental::udm::Gcore> receiver_gcores;
    std::vector<tt::tt_metal::experimental::udm::Gcore> all_gcores;

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
    bfloat16 bfloat_winv = bfloat16(1.0f / block_wt);
    uint32_t packed_winv = pack_two_bfloat16_into_uint32({bfloat_winv, bfloat_winv});

    // Get coordinate dimensions from any gcore
    uint32_t coord_dims = gcores[0].global_coord.dims();

    // Sender kernel args: sem_recv, sem_send, num_blocks, block_ht, rows_per_worker, rows_per_worker_last, winv,
    // coord_dims
    std::vector<uint32_t> reader_sender_compile_time_args = {
        reduce_receiver_semaphore_id.at(0),  // Get semaphore for grid 0
        reduce_sender_semaphore_id.at(0),
        (uint32_t)num_cores_x,
        (uint32_t)block_ht,
        (uint32_t)num_rows_per_worker,
        (uint32_t)num_rows_per_worker_last,
        packed_winv,
        (uint32_t)coord_dims};

    // Receiver kernel args: sem_recv, sem_send, num_blocks, block_ht, rows_per_worker, rows_per_worker_last, winv,
    // coord_dims
    std::vector<uint32_t> reader_receiver_compile_time_args = {
        reduce_receiver_semaphore_id.at(0),  // Get semaphore for grid 0
        reduce_sender_semaphore_id.at(0),
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
        "tests/tt_metal/udm/kernels/reader_sender_unary_sharded_reduce.cpp",
        sender_gcores,
        tt::tt_metal::DataMovementConfig{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
            .noc = tt::tt_metal::NOC::RISCV_0_default,
            .compile_args = reader_sender_compile_time_args});

    auto reader_receiver_kernel_id = tt::tt_metal::experimental::udm::CreateMeshKernel(
        mesh_builder,
        program,
        "tests/tt_metal/udm/kernels/reader_receiver_unary_sharded_reduce.cpp",
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
        "tests/tt_metal/udm/kernels/compute_sharded_reduce.cpp",
        all_gcores,
        tt::tt_metal::ComputeConfig{.compile_args = compute_all_to_all_args});

    // ===== SET RUNTIME ARGS =====
    // Set runtime args for each row independently
    for (uint32_t y = 0; y < num_cores_y; ++y) {
        // Sender core for this row
        const auto& sender_gcore = gcores[y * num_cores_x + 0];

        // Collect all cores' gcores in this row (sender + receivers)
        std::vector<const tt::tt_metal::experimental::udm::Gcore*> row_gcores;
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

    // Aggregate tensors from distributed format
    // Input is block-sharded, output is height-sharded (replicated on width)
    auto input_composer = create_block_sharded_mesh_composer(mesh_device, input_tensor.padded_shape().rank());
    auto input_data = ttnn::distributed::aggregate_tensor(input_tensor, *input_composer).to_vector<bfloat16>();

    auto output_composer = create_height_sharded_mesh_composer(mesh_device, output_tensor.padded_shape().rank());
    auto output_data = ttnn::distributed::aggregate_tensor(output_tensor, *output_composer).to_vector<bfloat16>();

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

    // Build expected and actual vectors for PCC comparison
    std::vector<bfloat16> expected_values;
    std::vector<bfloat16> actual_values;
    expected_values.reserve(num_rows);
    actual_values.reserve(num_rows);

    for (uint32_t row = 0; row < num_rows; ++row) {
        // Compute expected sum for this row (sum across last dimension)
        float expected_sum = 0.0f;
        for (uint32_t last_idx = 0; last_idx < input_last_dim; ++last_idx) {
            uint32_t input_idx = row * input_last_dim + last_idx;
            expected_sum += float(input_data[input_idx]);
        }
        expected_values.push_back(bfloat16(expected_sum));

        // Get actual value from output (first element along last dimension)
        uint32_t output_idx = row * output_last_dim;
        actual_values.push_back(output_data[output_idx]);
    }

    // Check PCC between expected and actual values
    float pcc = check_bfloat16_vector_pcc(expected_values, actual_values);
    const float pcc_threshold = 0.99f;

    log_info(tt::LogTest, "PCC: {:.6f} (threshold: {:.2f})", pcc, pcc_threshold);

    if (pcc < pcc_threshold) {
        TT_THROW("Width reduction validation failed: PCC {:.6f} below threshold {:.2f}", pcc, pcc_threshold);
    }

    log_info(tt::LogTest, "Width reduction validation passed: PCC={:.6f} for {} rows", pcc, num_rows);
}

/**
 * @brief Create a block-sharded bfloat16 tensor for reduction
 * TODO: Move this to test_udm_utils.hpp once stabilized
 */
inline ttnn::Tensor create_block_sharded_bfloat16_tensor(
    tt::tt_metal::distributed::MeshDevice* mesh_device,
    const tt::tt_metal::Shape& global_shape,
    const tt::tt_metal::Shape& local_shape) {
    // Get device grid for sharding
    auto compute_grid = mesh_device->compute_with_storage_grid_size();

    // Calculate how many cores to use in each dimension
    // Ensure at least 1 tile (32 elements) per core in width
    constexpr uint32_t TILE_WIDTH = 32;
    uint32_t num_cores_x = std::min((uint32_t)compute_grid.x, local_shape[-1] / TILE_WIDTH);
    uint32_t num_cores_y = compute_grid.y;

    // Calculate shard shape (height and width per shard in elements)
    uint32_t shard_height = local_shape[-2] / num_cores_y;
    uint32_t shard_width = local_shape[-1] / num_cores_x;

    // Create shard spec for block sharding
    // When num_cores_x = 1, this effectively becomes height sharding
    auto shard_spec = ShardSpec(
        CoreRangeSet({CoreRange({0, 0}, {num_cores_x - 1, num_cores_y - 1})}),
        std::array<uint32_t, 2>{shard_height, shard_width},
        ShardOrientation::ROW_MAJOR);

    // Use BLOCK_SHARDED memory layout with bfloat16 and shard spec
    tt::tt_metal::MemoryConfig mem_config(
        tt::tt_metal::TensorMemoryLayout::BLOCK_SHARDED, tt::tt_metal::BufferType::L1, shard_spec);

    tt::tt_metal::TensorSpec tensor_spec(
        global_shape,
        tt::tt_metal::TensorLayout(
            tt::tt_metal::DataType::BFLOAT16, tt::tt_metal::PageConfig(tt::tt_metal::Layout::TILE), mem_config));

    // Create host tensor with random bfloat16 values
    uint32_t volume = 1;
    for (size_t i = 0; i < global_shape.rank(); ++i) {
        volume *= global_shape[i];
    }

    std::vector<bfloat16> src_data(volume);

    // Generate random data in bfloat16 range
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0.0f, 10.0f);
    for (uint32_t i = 0; i < volume; ++i) {
        src_data[i] = bfloat16(dis(gen));
    }

    auto host_tensor = ttnn::Tensor::from_vector(src_data, tensor_spec);

    // Use block-sharded mapper
    auto mapper = create_block_sharded_mesh_mapper(mesh_device, global_shape.rank());
    return ttnn::distributed::distribute_tensor(host_tensor, *mapper, std::ref(*mesh_device));
}

/**
 * @brief Create a height-sharded bfloat16 tensor (replicated across width dimension)
 * Used for reduction outputs where width is reduced to a single tile
 */
inline ttnn::Tensor create_height_sharded_bfloat16_tensor(
    tt::tt_metal::distributed::MeshDevice* mesh_device,
    const tt::tt_metal::Shape& global_shape,
    const tt::tt_metal::Shape& local_shape) {
    // Get device grid for sharding
    auto compute_grid = mesh_device->compute_with_storage_grid_size();

    // For height-sharded: use only 1 core in X, full grid in Y
    uint32_t num_cores_x = 1;
    uint32_t num_cores_y = compute_grid.y;

    // Calculate shard shape
    uint32_t shard_height = local_shape[-2] / num_cores_y;
    uint32_t shard_width = local_shape[-1];  // Full width per core

    // Create shard spec - single column of cores
    auto shard_spec = ShardSpec(
        CoreRangeSet({CoreRange({0, 0}, {num_cores_x - 1, num_cores_y - 1})}),
        std::array<uint32_t, 2>{shard_height, shard_width},
        ShardOrientation::ROW_MAJOR);

    // Use BLOCK_SHARDED memory layout (effectively height sharding with 1 core in X)
    tt::tt_metal::MemoryConfig mem_config(
        tt::tt_metal::TensorMemoryLayout::BLOCK_SHARDED, tt::tt_metal::BufferType::L1, shard_spec);

    tt::tt_metal::TensorSpec tensor_spec(
        global_shape,
        tt::tt_metal::TensorLayout(
            tt::tt_metal::DataType::BFLOAT16, tt::tt_metal::PageConfig(tt::tt_metal::Layout::TILE), mem_config));

    // Create host tensor with zeros (output will be written by kernel)
    uint32_t volume = 1;
    for (size_t i = 0; i < global_shape.rank(); ++i) {
        volume *= global_shape[i];
    }

    std::vector<bfloat16> src_data(volume, bfloat16(0.0f));

    auto host_tensor = ttnn::Tensor::from_vector(src_data, tensor_spec);

    // Use height-sharded mapper (shard on height, replicate on width)
    auto mapper = create_height_sharded_mesh_mapper(mesh_device, global_shape.rank());
    return ttnn::distributed::distribute_tensor(host_tensor, *mapper, std::ref(*mesh_device));
}

/**
 * @brief Run width reduction test
 */
void run_width_reduction_test(
    tt::tt_metal::distributed::MeshDevice* mesh_device,
    const tt::tt_metal::Shape& global_shape,
    const tt::tt_metal::Shape& local_shape) {
    // Create input tensor - block-sharded bfloat16
    auto input_tensor = create_block_sharded_bfloat16_tensor(mesh_device, global_shape, local_shape);

    // Create output tensor for reduced result
    // Output shape is same as input except last dimension is reduced to 32 (one tile)
    // Output is height-sharded only (replicated across mesh width dimension)
    tt::tt_metal::Shape output_global_shape = global_shape;
    tt::tt_metal::Shape output_local_shape = local_shape;
    output_global_shape[-1] = 32;  // Reduce last dimension to single tile
    output_local_shape[-1] = 32;

    auto output_tensor = create_height_sharded_bfloat16_tensor(mesh_device, output_global_shape, output_local_shape);

    // Build tensor builder from input tensor
    auto input_mesh_tensor_builder = create_tensor_builder(input_tensor);

    // Create reduction program
    auto program = create_program(input_mesh_tensor_builder, *input_tensor.buffer(), *output_tensor.buffer());

    // Run program
    auto* tensor_mesh_device = input_tensor.device();
    ASSERT_NE(tensor_mesh_device, nullptr) << "Tensor must be on device";
    run_program(input_tensor, tensor_mesh_device, program);

    // Validate result
    validate(input_tensor, output_tensor);
}

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
    // Small 2D tensor: (4, 16) tiles = (128, 512) elements
    // Mesh: 1×4, so block_ht = 4/1 = 4 tiles, num_cores_x = 4
    // Constraint satisfied: block_ht (4) >= num_cores_x (4) ✓
    // Each device gets (4, 4) tiles = (128, 128) elements locally
    tt::tt_metal::Shape global_shape({128, 512});
    tt::tt_metal::Shape local_shape({128, 128});

    run_width_reduction_test(mesh_device_.get(), global_shape, local_shape);
}

TEST_F(MeshDevice1x4Fabric2DUDMFixture, TestWidthReduction2D_Medium) {
    // Medium 2D tensor: (8, 32) tiles = (256, 1024) elements
    // Mesh: 1×4, so block_ht = 8/1 = 8 tiles, num_cores_x = 4
    // Constraint satisfied: block_ht (8) >= num_cores_x (4) ✓
    // Each device gets (8, 8) tiles = (256, 256) elements locally
    tt::tt_metal::Shape global_shape({256, 1024});
    tt::tt_metal::Shape local_shape({256, 256});

    run_width_reduction_test(mesh_device_.get(), global_shape, local_shape);
}

}  // namespace tt::tt_metal::experimental::udm_tests
