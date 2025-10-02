// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

#include <iostream>
#include <vector>
#include <cmath>
#include <memory>
#include <algorithm>
#include <string>
#include <map>

#include <tt-metalium/distributed.hpp>
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

using namespace tt;
using namespace tt::tt_metal;
using namespace tt::tt_metal::distributed;

Program CreateMultiTensixEltwiseAddProgram(
    const std::shared_ptr<MeshBuffer>& a,
    const std::shared_ptr<MeshBuffer>& b,
    const std::shared_ptr<MeshBuffer>& c,
    size_t tile_size_bytes,
    uint32_t num_tiles,
    uint32_t r_tiles,
    uint32_t tensix_cores_per_device,
    const std::shared_ptr<MeshDevice>& mesh) {

    auto program = CreateProgram();
    auto core_grid = mesh->compute_with_storage_grid_size();

    // Calculate optimal Tensix distribution
    uint32_t max_tensix_cores = std::min(tensix_cores_per_device, static_cast<uint32_t>(core_grid.x * core_grid.y));
    uint32_t mesh_devices = mesh->num_devices();
    uint32_t tiles_per_device = (num_tiles + mesh_devices - 1) / mesh_devices;
    uint32_t tiles_per_tensix = std::max(1u, (tiles_per_device + max_tensix_cores - 1) / max_tensix_cores);

    std::cout << "DEBUG: Multi-Tensix Configuration:" << std::endl;
    std::cout << "  Mesh devices: " << mesh_devices << std::endl;
    std::cout << "  Available Tensix cores per device: " << (core_grid.x * core_grid.y) << std::endl;
    std::cout << "  Using Tensix cores per device: " << max_tensix_cores << std::endl;
    std::cout << "  Total Tensix cores: " << (mesh_devices * max_tensix_cores) << std::endl;
    std::cout << "  Tiles per device: " << tiles_per_device << std::endl;
    std::cout << "  Tiles per Tensix core: " << tiles_per_tensix << std::endl;

    // Create a grid of Tensix cores to use
    uint32_t cores_x = std::min(max_tensix_cores, static_cast<uint32_t>(core_grid.x));
    uint32_t cores_y = (max_tensix_cores + cores_x - 1) / cores_x;
    cores_y = std::min(cores_y, static_cast<uint32_t>(core_grid.y));
    uint32_t actual_cores = cores_x * cores_y;

    auto target_tensix_cores = tt::tt_metal::CoreRangeSet({
        tt::tt_metal::CoreRange({0, 0}, {cores_x - 1, cores_y - 1})
    });

    std::cout << "  Tensix grid: " << cores_x << "x" << cores_y << " = " << actual_cores << " cores" << std::endl;

    // Reduced memory pressure per Tensix core
    constexpr uint32_t tile_size = 4 * 1024;  // 4KB per tile
    constexpr uint32_t target_cb_memory_per_core = 32 * 1024;  // 32KB per core (much less pressure!)
    constexpr uint32_t cb2_cb16_tiles = 4;  // CB2 (2) + CB16 (2) = 4 tiles fixed
    uint32_t available_tiles_per_core = (target_cb_memory_per_core / tile_size) - cb2_cb16_tiles;  // ~4 tiles available per core

    // Smart CB allocation per Tensix core
    uint32_t cb0_tiles = std::min(tiles_per_tensix + 1, available_tiles_per_core / 2);  // Half for CB0
    uint32_t cb1_tiles = available_tiles_per_core - cb0_tiles;  // Rest for CB1
    cb1_tiles = std::min(cb1_tiles, r_tiles);  // Don't exceed r_tiles

    std::cout << "  CB allocation per core: CB0=" << cb0_tiles << ", CB1=" << cb1_tiles
              << " tiles (" << ((cb0_tiles + cb1_tiles + cb2_cb16_tiles) * tile_size / 1024)
              << "KB/" << (target_cb_memory_per_core/1024) << "KB)" << std::endl;

    constexpr uint32_t src0_cb_index = tt::CBIndex::c_0;
    constexpr uint32_t src1_cb_index = tt::CBIndex::c_1;
    constexpr uint32_t cb2_tiles = 2;
    constexpr uint32_t output_cb_index = tt::CBIndex::c_16;
    constexpr uint32_t cb16_tiles = 2;

    // Create circular buffers for all Tensix cores
    CircularBufferConfig cb_src0_config =
        CircularBufferConfig(cb0_tiles * tile_size_bytes, {{src0_cb_index, tt::DataFormat::Float32}})
            .set_page_size(src0_cb_index, tile_size_bytes);
    tt_metal::CreateCircularBuffer(program, target_tensix_cores, cb_src0_config);

    CircularBufferConfig cb_src1_config =
        CircularBufferConfig(cb1_tiles * tile_size_bytes, {{src1_cb_index, tt::DataFormat::Float32}})
            .set_page_size(src1_cb_index, tile_size_bytes);
    tt_metal::CreateCircularBuffer(program, target_tensix_cores, cb_src1_config);

    CircularBufferConfig cb_interm_config =
        CircularBufferConfig(cb2_tiles * tile_size_bytes, {{tt::CBIndex::c_2, tt::DataFormat::Float32}})
            .set_page_size(tt::CBIndex::c_2, tile_size_bytes);
    tt_metal::CreateCircularBuffer(program, target_tensix_cores, cb_interm_config);

    CircularBufferConfig cb_output_config =
        CircularBufferConfig(cb16_tiles * tile_size_bytes, {{output_cb_index, tt::DataFormat::Float32}})
            .set_page_size(output_cb_index, tile_size_bytes);
    tt_metal::CreateCircularBuffer(program, target_tensix_cores, cb_output_config);

    // Compile-time arguments for kernels
    std::vector<uint32_t> reader_compile_time_args = {};
    std::vector<uint32_t> compute_compile_time_args = {};
    std::vector<uint32_t> writer_compile_time_args = {};

    // Create kernels for all Tensix cores
    KernelHandle reader = CreateKernel(
        program,
        "tt_metal/programming_examples/alexp_examples/e4/multi_tensix_elementwise_add/kernels/multi_tensix_read.cpp",
        target_tensix_cores,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1,
            .noc = NOC::RISCV_1_default,
            .compile_args = reader_compile_time_args});

    std::vector<UnpackToDestMode> unpack_modes(NUM_CIRCULAR_BUFFERS, UnpackToDestMode::Default);
    unpack_modes[static_cast<uint32_t>(tt::CBIndex::c_0)] = UnpackToDestMode::UnpackToDestFp32;
    unpack_modes[static_cast<uint32_t>(tt::CBIndex::c_1)] = UnpackToDestMode::UnpackToDestFp32;
    unpack_modes[static_cast<uint32_t>(tt::CBIndex::c_2)] = UnpackToDestMode::UnpackToDestFp32;
    unpack_modes[static_cast<uint32_t>(tt::CBIndex::c_16)] = UnpackToDestMode::UnpackToDestFp32;
    auto compute = CreateKernel(
        program,
        "tt_metal/programming_examples/alexp_examples/e4/multi_tensix_elementwise_add/kernels/multi_tensix_add.cpp",
        target_tensix_cores,
        ComputeConfig{
            .math_fidelity = MathFidelity::HiFi4,
            .fp32_dest_acc_en = true,
            .dst_full_sync_en = true,
            .unpack_to_dest_mode = unpack_modes,
            .math_approx_mode = false,
            .compile_args = compute_compile_time_args});

    KernelHandle writer = CreateKernel(
        program,
        "tt_metal/programming_examples/alexp_examples/e4/multi_tensix_elementwise_add/kernels/multi_tensix_write.cpp",
        target_tensix_cores,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .compile_args = writer_compile_time_args});

    // Set runtime arguments for each Tensix core
    uint32_t core_idx = 0;
    for (uint32_t y = 0; y < cores_y; y++) {
        for (uint32_t x = 0; x < cores_x; x++) {
            if (core_idx >= actual_cores) break;

            CoreCoord core = {x, y};

            // Calculate tile range for this Tensix core
            uint32_t tiles_start = core_idx * tiles_per_tensix;
            uint32_t tiles_end = std::min(tiles_start + tiles_per_tensix, tiles_per_device);
            uint32_t core_num_tiles = (tiles_end > tiles_start) ? (tiles_end - tiles_start) : 0;

            if (core_num_tiles > 0) {
                // Reader kernel arguments
                SetRuntimeArgs(program, reader, core, {
                    a->address(),      // a_addr
                    b->address(),      // b_addr
                    core_num_tiles,          // n_tiles for this core
                    r_tiles,                 // r_tiles
                    tiles_start              // tile_offset
                });

                // Compute kernel arguments
                SetRuntimeArgs(program, compute, core, {
                    core_num_tiles,          // n_tiles for this core
                    r_tiles                  // r_tiles
                });

                // Writer kernel arguments
                SetRuntimeArgs(program, writer, core, {
                    c->address(),     // c_addr
                    core_num_tiles,          // n_tiles for this core
                    tiles_start              // tile_offset
                });

                std::cout << "  Core (" << x << "," << y << "): tiles " << tiles_start
                          << "-" << (tiles_end-1) << " (" << core_num_tiles << " tiles)" << std::endl;
            }

            core_idx++;
        }
    }

    return program;
}

int main(int argc, char** argv) {
    uint32_t num_tiles = 16;  // Default
    uint32_t tensix_cores_per_device = 4;  // Default: use 4 cores per device

    if (argc >= 2) {
        num_tiles = std::stoi(argv[1]);
    }
    if (argc >= 3) {
        tensix_cores_per_device = std::stoi(argv[2]);
    }

    if (num_tiles == 0 || tensix_cores_per_device == 0) {
        std::cout << "Usage: " << argv[0] << " [num_tiles] [tensix_cores_per_device]" << std::endl;
        std::cout << "Example: " << argv[0] << " 32 8  # 32 tiles using 8 Tensix cores per device" << std::endl;
        return 1;
    }

    constexpr uint32_t rep_num_tiles = 20;  // B tensor tiles (replicated)
    constexpr uint32_t tile_size_bytes = 4 * 1024;  // 4KB per tile
    uint32_t total_float_values = num_tiles * 1024;

    std::cout << "=== Multi-Tensix Distributed Elementwise Add ===" << std::endl;
    std::cout << "A tiles (distributed): " << num_tiles << std::endl;
    std::cout << "B tiles (replicated): " << rep_num_tiles << std::endl;
    std::cout << "Tensix cores per device: " << tensix_cores_per_device << std::endl;
    std::cout << "Total elements: " << total_float_values << std::endl;
    std::cout << std::endl;

    // Create mesh device
    std::shared_ptr<MeshDevice> mesh_device;
    try {
        mesh_device = MeshDevice::create(MeshDeviceConfig(MeshShape(1, 2)));
        std::cout << "Using 1x2 mesh configuration with " << mesh_device->num_devices() << " devices" << std::endl;
    } catch (const std::exception& e) {
        std::cout << "Failed to create 1x2 mesh, falling back to single device: " << e.what() << std::endl;
        mesh_device = MeshDevice::create_unit_mesh(0);
        std::cout << "Using single device configuration" << std::endl;
    }

    auto mesh_rows = mesh_device->num_rows();
    auto mesh_cols = mesh_device->num_cols();
    uint32_t tiles_per_shard = (num_tiles + mesh_rows * mesh_cols - 1) / (mesh_rows * mesh_cols);

    // Create distributed and replicated buffers
    auto shard_shape = Shape2D{
        tiles_per_shard * 32,  // 32 floats per row in a tile
        32                     // 32 floats per column in a tile
    };

    auto distributed_buffer_shape = Shape2D{
        num_tiles * 32,
        32
    };

    auto replicated_buffer_shape = Shape2D{
        rep_num_tiles * 32,
        32
    };

    auto local_buffer_config = DeviceLocalBufferConfig{
        .page_size = tile_size_bytes,
        .buffer_type = BufferType::DRAM,
        .bottom_up = false
    };
    auto distributed_buffer_config = tt::tt_metal::distributed::ShardedBufferConfig{
        .global_size = tiles_per_shard * tile_size_bytes * mesh_device->num_devices(),
        .global_buffer_shape = distributed_buffer_shape,
        .shard_shape = shard_shape,
        .shard_orientation = ShardOrientation::ROW_MAJOR
    };
    auto a = MeshBuffer::create(distributed_buffer_config, local_buffer_config, mesh_device.get());

    auto replicated_buffer_config = tt::tt_metal::distributed::ReplicatedBufferConfig{
        .size = rep_num_tiles * tile_size_bytes,
    };
    auto b = MeshBuffer::create(replicated_buffer_config, local_buffer_config, mesh_device.get());

    auto c = MeshBuffer::create(distributed_buffer_config, local_buffer_config, mesh_device.get());

    // Initialize data
    std::vector<float> a_data(total_float_values);
    std::vector<float> b_data(rep_num_tiles * 1024);

    // A data: simple incrementing values
    for (int i = 0; i < total_float_values; ++i) {
        a_data[i] = static_cast<float>(i % 256 + 1);  // 1-256 repeating
    }

    // B data: tile index + 1 (so tile 0 has value 1, tile 1 has value 2, etc.)
    for (int tile = 0; tile < rep_num_tiles; ++tile) {
        for (int j = 0; j < 1024; ++j) {
            b_data[tile * 1024 + j] = static_cast<float>(tile + 1);
        }
    }

    // Calculate expected sum of B values
    float sum_scalars = 0.0f;
    for (int i = 0; i < rep_num_tiles; ++i) {
        sum_scalars += static_cast<float>(i + 1);
    }

    std::cout << "Expected sum of B scalars: " << sum_scalars << " (sum of 1 to " << rep_num_tiles << ")" << std::endl;

    // Write data to device
    auto& cq = mesh_device->mesh_command_queue();
    EnqueueWriteMeshBuffer(cq, a, a_data, false);
    EnqueueWriteMeshBuffer(cq, b, b_data, false);

    auto program = CreateMultiTensixEltwiseAddProgram(a, b, c, tile_size_bytes, num_tiles, rep_num_tiles, tensix_cores_per_device, mesh_device);

    auto mesh_workload = CreateMeshWorkload();
    auto device_range = MeshCoordinateRange(mesh_device->shape());
    AddProgramToMeshWorkload(mesh_workload, std::move(program), device_range);

    std::cout << "\n=== Executing Multi-Tensix Workload ===" << std::endl;
    EnqueueMeshWorkload(cq, mesh_workload, false);
    Finish(cq);

    // Read results
    std::vector<float> result_data(total_float_values, 0);
    EnqueueReadMeshBuffer(cq, result_data, c, true);

    // Verify results
    std::vector<float> golden_data(total_float_values);
    for (int i = 0; i < total_float_values; ++i) {
        golden_data[i] = a_data[i] + sum_scalars;
    }

    uint32_t num_passed = 0;
    uint32_t num_failed = 0;

    for (int i = 0; i < total_float_values; ++i) {
        if (std::abs(result_data[i] - golden_data[i]) < 1e-6) {
            num_passed++;
        } else {
            num_failed++;
            if (num_failed <= 10) {  // Show first 10 failures
                std::cout << "Mismatch at index " << i << ": expected " << golden_data[i]
                          << ", got " << result_data[i] << std::endl;
            }
        }
    }

    std::cout << "\n=== Multi-Tensix Results ===" << std::endl;
    std::cout << "Multi-Tensix elementwise add verification: " << num_passed << " / " << total_float_values << " passed" << std::endl;

    if (num_passed == total_float_values) {
        std::cout << "✅ SUCCESS: All elements passed verification!" << std::endl;
    } else {
        std::cout << "❌ FAILED: " << num_failed << " elements failed verification" << std::endl;
        std::cout << "Success rate: " << (100.0 * num_passed / total_float_values) << "%" << std::endl;
    }

    return (num_passed == total_float_values) ? 0 : 1;
}
