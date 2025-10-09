// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <iostream>

#include <tt-metalium/distributed.hpp>
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

using namespace tt;
using namespace tt::tt_metal;
using namespace tt::tt_metal::distributed;

Program CreateEltwiseAddProgram(
    const std::shared_ptr<MeshBuffer>& a,
    const std::shared_ptr<MeshBuffer>& b,
    const std::shared_ptr<MeshBuffer>& c,
    size_t tile_size_bytes,
    uint32_t num_tiles,
    uint32_t r_tiles,
    uint32_t tiles_per_shard,
    const std::shared_ptr<MeshDevice>& mesh) {

    auto program = CreateProgram();
    auto core_grid = mesh->compute_with_storage_grid_size();
    auto target_tensix_cores = tt::tt_metal::CoreRangeSet({tt::tt_metal::CoreRange({0, 0}, {core_grid.x - 1, core_grid.y - 1})});

    constexpr uint32_t src0_cb_index = tt::CBIndex::c_0;
    // Scalable memory allocation: Works for any number of tiles by using minimal buffering
    // Conservative target to ensure stability across all workload sizes
    constexpr uint32_t target_total_cb_memory = 48 * 1024;  // 48KB conservative target
    constexpr uint32_t tile_size = 4 * 1024;  // 4KB per tile
    constexpr uint32_t cb2_cb16_tiles = 4;  // CB2 (2) + CB16 (2) = 4 tiles fixed
    uint32_t available_tiles = (target_total_cb_memory / tile_size) - cb2_cb16_tiles;  // ~8 tiles available

    // Scalable CB allocation strategy:
    // - Use minimal buffering that works for any tile count
    // - Prioritize correctness over performance for large workloads
    uint32_t cb0_tiles, cb1_tiles;

    // For large workloads, use streaming approach with minimal buffers
    if (tiles_per_shard > 16 || r_tiles > 16) {
        // Large workload: Use minimal streaming buffers
        cb0_tiles = 2;  // Minimal A tile buffering (streaming)
        cb1_tiles = 2;  // Minimal B tile buffering (streaming)
        std::cout << "INFO: Large workload detected (" << tiles_per_shard << " A tiles, "
                  << r_tiles << " B tiles) - using streaming mode" << std::endl;
    } else {
        // Small/medium workload: Use optimal buffering
        uint32_t cb0_optimal = std::min(6u, tiles_per_shard + 2);  // Optimal A buffering
        uint32_t cb1_optimal = std::min(r_tiles, 6u);             // Optimal B buffering

        if (cb0_optimal + cb1_optimal <= available_tiles) {
            cb0_tiles = cb0_optimal;
            cb1_tiles = cb1_optimal;
        } else {
            // Medium pressure: Balance between A and B
            cb0_tiles = std::max(2u, available_tiles / 2);
            cb1_tiles = available_tiles - cb0_tiles;
        }
    }

    // Debug output for buffer configuration
    uint32_t total_cb_tiles = cb0_tiles + cb1_tiles + cb2_cb16_tiles;
    uint32_t total_cb_memory = total_cb_tiles * tile_size;
    std::cout << "DEBUG: Scalable CB allocation - CB0:" << cb0_tiles << ", CB1:" << cb1_tiles
              << ", CB2+CB16:4 tiles. Total:" << total_cb_tiles << " tiles ("
              << (total_cb_memory/1024) << "KB/" << (target_total_cb_memory/1024) << "KB)" << std::endl;

    // Additional debug for large workloads
    if (tiles_per_shard > 16 || r_tiles > 16) {
        std::cout << "DEBUG: Streaming mode active - kernels will process tiles in small batches" << std::endl;
        std::cout << "DEBUG: Expected behavior: Slower but stable execution for large workloads" << std::endl;
    }
    constexpr uint32_t cb2_tiles = 2;  // Intermediate can stay small
    CircularBufferConfig cb_src0_config =
        CircularBufferConfig(cb0_tiles * tile_size_bytes, {{src0_cb_index, tt::DataFormat::Float32}})
            .set_page_size(src0_cb_index, tile_size_bytes);
    tt_metal::CreateCircularBuffer(program, target_tensix_cores, cb_src0_config);

    constexpr uint32_t src1_cb_index = tt::CBIndex::c_1;
    CircularBufferConfig cb_src1_config =
        CircularBufferConfig(cb1_tiles * tile_size_bytes, {{src1_cb_index, tt::DataFormat::Float32}})
            .set_page_size(src1_cb_index, tile_size_bytes);
    tt_metal::CreateCircularBuffer(program, target_tensix_cores, cb_src1_config);

    constexpr uint32_t cb_interm = tt::CBIndex::c_2;
    CircularBufferConfig cb_interm_config =
        CircularBufferConfig(cb2_tiles * tile_size_bytes, {{cb_interm, tt::DataFormat::Float32}})
            .set_page_size(cb_interm, tile_size_bytes);
    tt_metal::CreateCircularBuffer(program, target_tensix_cores, cb_interm_config);

    constexpr uint32_t output_cb_index = tt::CBIndex::c_16;
    constexpr uint32_t cb16_tiles = 2;  // Output can stay small (double buffered)
    CircularBufferConfig cb_output_config =
        CircularBufferConfig(cb16_tiles * tile_size_bytes, {{output_cb_index, tt::DataFormat::Float32}})
            .set_page_size(output_cb_index, tile_size_bytes);
    tt_metal::CreateCircularBuffer(program, target_tensix_cores, cb_output_config);

    std::vector<uint32_t> reader_compile_time_args;
    TensorAccessorArgs(*a->get_reference_buffer()).append_to(reader_compile_time_args);
    TensorAccessorArgs(*b->get_reference_buffer()).append_to(reader_compile_time_args);
    KernelHandle reader = CreateKernel(
        program,
        "tt_metal/programming_examples/alexp_examples/e4/elementwise_add/kernels/replicated_read.cpp",
        target_tensix_cores,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1,
            .noc = NOC::RISCV_1_default,
            .compile_args = reader_compile_time_args});

    std::vector<uint32_t> writer_compile_time_args;
    TensorAccessorArgs(*c->get_reference_buffer()).append_to(writer_compile_time_args);
    KernelHandle writer = CreateKernel(
        program,
        "tt_metal/programming_examples/contributed/vecadd/kernels/tile_write.cpp",
        target_tensix_cores,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .compile_args = writer_compile_time_args});

    std::vector<UnpackToDestMode> unpack_modes(NUM_CIRCULAR_BUFFERS, UnpackToDestMode::Default);
    unpack_modes[static_cast<uint32_t>(tt::CBIndex::c_0)] = UnpackToDestMode::UnpackToDestFp32;
    unpack_modes[static_cast<uint32_t>(tt::CBIndex::c_1)] = UnpackToDestMode::UnpackToDestFp32;
    unpack_modes[static_cast<uint32_t>(tt::CBIndex::c_2)] = UnpackToDestMode::UnpackToDestFp32;
    unpack_modes[static_cast<uint32_t>(tt::CBIndex::c_16)] = UnpackToDestMode::UnpackToDestFp32;

    auto compute = CreateKernel(
        program,
        "tt_metal/programming_examples/alexp_examples/e4/elementwise_add/kernels/replicated_add.cpp",
        target_tensix_cores,
        ComputeConfig{
            .math_fidelity = MathFidelity::HiFi4,
            .fp32_dest_acc_en = true,
            .dst_full_sync_en = true,
            .unpack_to_dest_mode = unpack_modes,
            .math_approx_mode = false,
            .compile_args = {},
        });

    SetRuntimeArgs(program, reader, target_tensix_cores, {a->address(), b->address(), num_tiles, r_tiles});
    SetRuntimeArgs(program, writer, target_tensix_cores, {c->address(), num_tiles});
    SetRuntimeArgs(program, compute, target_tensix_cores, {num_tiles, r_tiles});

    return program;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <num_tiles>" << std::endl;
        return 1;
    }

    uint32_t num_tiles;
    try {
        num_tiles = std::stoul(argv[1]);
    } catch (const std::exception& e) {
        std::cerr << "Invalid argument: " << argv[1] << ". Please provide a positive integer." << std::endl;
        return 1;
    }

    if (num_tiles == 0) {
        std::cerr << "Total number of tiles must be a positive integer." << std::endl;
        return 1;
    }

    // Add helpful info for large workloads
    if (num_tiles > 64) {
        std::cout << "INFO: Large workload (" << num_tiles << " tiles) detected." << std::endl;
        std::cout << "INFO: Using conservative memory allocation for stability." << std::endl;
    }

    // Try to create mesh device with 1x2 configuration, fallback to single device if not available
    std::shared_ptr<MeshDevice> mesh_device;
    try {
        mesh_device = MeshDevice::create(MeshDeviceConfig(MeshShape(1, 2)));
        std::cout << "Using 1x2 mesh configuration with " << mesh_device->num_devices() << " devices" << std::endl;
    } catch (const std::exception& e) {
        std::cout << "Failed to create 1x2 mesh, falling back to single device: " << e.what() << std::endl;
        mesh_device = MeshDevice::create_unit_mesh(0);
        std::cout << "Using single device configuration" << std::endl;
    }

    auto tile_shape = Shape2D{32, 32};
    auto mesh_rows = mesh_device->num_rows(); // 1
    auto mesh_cols = mesh_device->num_cols(); // 2

    // Ensure all tiles are covered by using ceiling division
    uint32_t tiles_per_shard = (num_tiles + mesh_rows * mesh_cols - 1) / (mesh_rows * mesh_cols);

    std::cout << "DEBUG: num_tiles=" << num_tiles << ", mesh_devices=" << (mesh_rows * mesh_cols)
              << ", tiles_per_shard=" << tiles_per_shard << std::endl;

    auto shard_shape = Shape2D{
        tiles_per_shard * tile_shape.height(),
        tile_shape.width()
    };

    auto distributed_buffer_shape = Shape2D{
        shard_shape.height() * mesh_rows,
        shard_shape.width() * mesh_cols
    };

    auto tile_size_bytes = 32 * 32 * sizeof(float); // 4096 bytes
    // Buffer size should accommodate the actual shard distribution
    auto total_sharded_tiles = tiles_per_shard * mesh_rows * mesh_cols;
    auto distributed_buffer_size_bytes = tile_size_bytes * total_sharded_tiles;
    size_t total_float_values = num_tiles * 32 * 32;

    std::cout << "DEBUG: total_sharded_tiles=" << total_sharded_tiles
              << ", buffer_size=" << distributed_buffer_size_bytes << " bytes" << std::endl;

    // Debug: Sharding configuration
    std::cout << "\nDEBUG: Sharding Configuration:" << std::endl;
    std::cout << "Shard shape: [" << shard_shape.height() << ", " << shard_shape.width() << "]" << std::endl;
    std::cout << "Distributed buffer shape: [" << distributed_buffer_shape.height() << ", " << distributed_buffer_shape.width() << "]" << std::endl;

    auto local_buffer_config =
        DeviceLocalBufferConfig{.page_size = tile_size_bytes, .buffer_type = BufferType::DRAM, .bottom_up = false};
    auto distributed_buffer_config = tt::tt_metal::distributed::ShardedBufferConfig{
        .global_size = distributed_buffer_size_bytes,
        .global_buffer_shape = distributed_buffer_shape,
        .shard_shape = shard_shape,
        .shard_orientation = ShardOrientation::ROW_MAJOR};

    int rep_num_tiles = num_tiles;

    auto replicated_buffer_config = tt::tt_metal::distributed::ReplicatedBufferConfig{
        .size = tile_size_bytes * rep_num_tiles,
    };

    auto a = MeshBuffer::create(distributed_buffer_config, local_buffer_config, mesh_device.get());
    auto b = MeshBuffer::create(replicated_buffer_config, local_buffer_config, mesh_device.get());
    auto c = MeshBuffer::create(distributed_buffer_config, local_buffer_config, mesh_device.get());

    std::vector<float> a_data, b_data;
    a_data.reserve(total_float_values);
    b_data.reserve(rep_num_tiles * 1024);

    /*
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dist(0, 10); // Generates integers 0-10

    for (int i = 0; i < total_float_values; ++i) {
        a_data.push_back(static_cast<float>(dist(gen))); // Store as float
    }
    */

    int l = 0;
    for (int i = 0; i < total_float_values; ++i) {
        if (i % 1024 == 0) l++;
        a_data[i] = l;
    }

    l = 0;
    for (int i = 0; i < rep_num_tiles * 1024; ++i) {
        if (i % 1024 == 0) l++;
        b_data[i] = l;
    }

    auto& cq = mesh_device->mesh_command_queue();
    EnqueueWriteMeshBuffer(cq, a, a_data, false /* blocking */);
    EnqueueWriteMeshBuffer(cq, b, b_data, false /* blocking */);

    auto program = CreateEltwiseAddProgram(a, b, c, tile_size_bytes, num_tiles, rep_num_tiles, tiles_per_shard, mesh_device);

    auto mesh_workload = MeshWorkload();
    auto device_range = MeshCoordinateRange(mesh_device->shape());

    mesh_workload.add_program(device_range, std::move(program));
    EnqueueMeshWorkload(cq, mesh_workload, false /* blocking */);

    std::vector<float> result_data(total_float_values, 0);
    EnqueueReadMeshBuffer(cq, result_data, c, true /* blocking */);

    int sum_scalars = rep_num_tiles * (rep_num_tiles + 1) / 2;
    std::vector<float> golden_data(total_float_values);
    for (int i = 0; i < total_float_values; ++i) {
        golden_data[i] = a_data[i] + sum_scalars;
    }

    // Debug: Memory layout and indexing analysis
    std::cout << "\nDEBUG: Memory Layout Analysis:" << std::endl;
    std::cout << "Total tiles: " << num_tiles << ", tile size: 1024 floats (4096 bytes)" << std::endl;
    std::cout << "Tiles per shard: " << tiles_per_shard << ", mesh devices: " << (mesh_rows * mesh_cols) << std::endl;
    std::cout << "Device 0 should handle tiles 0-" << (tiles_per_shard-1) << std::endl;
    std::cout << "Device 1 should handle tiles " << tiles_per_shard << "-" << (num_tiles-1) << std::endl;


    size_t num_failures = 0;
    for (int i = 0; i < total_float_values; i++) {
        if (std::abs(result_data[i] - golden_data[i]) > 1e-6) {
            num_failures++;
            if (num_failures < 10) {
                std::cout << "Mismatch at index " << i << ": expected " << golden_data[i]
                          << ", got " << result_data[i] << std::endl;
            }
            if (i < 5) {
                std::cout << "Input a[" << i << "] = " << a_data[i] << ", Input b[" << i << "] = " << b_data[i % (rep_num_tiles * 1024)] << std::endl;
            }
        }
    }

    std::cout << "Total values: " << total_float_values << "\n";
    std::cout << "Distributed elementwise add verification: " << (total_float_values - num_failures)
              << " / " << total_float_values << " passed\n";

    return 0;
}
