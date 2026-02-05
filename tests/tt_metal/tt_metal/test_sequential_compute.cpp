// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

/**
 * Test: Sequential Compute
 *
 * Tests running multiple elementwise operations sequentially on the same core.
 * Each phase runs to completion before the next phase starts.
 *
 * Phase 0: A + B → scratch
 * Phase 1: scratch * C → scratch
 * Phase 2: scratch + D → output
 *
 * Expected result: (A + B) * C + D
 */

#include "common/command_queue_fixture.hpp"

#include <cstdint>
#include <vector>

#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/circular_buffer_config.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-logger/tt-logger.hpp>
#include "test_gold_impls.hpp"

using namespace tt;
using namespace tt::tt_metal;

namespace {

void run_sequential_compute_test(
    const std::shared_ptr<distributed::MeshDevice>& mesh_device,
    distributed::MeshCommandQueue& cq,
    uint32_t num_tiles,
    uint32_t num_phases) {
    log_info(LogTest, "====================================================================");
    log_info(LogTest, "Running sequential compute test: num_tiles={}, num_phases={}", num_tiles, num_phases);

    Program program = CreateProgram();
    distributed::MeshWorkload mesh_workload;
    CoreCoord core = {0, 0};

    uint32_t single_tile_size = 2 * 1024;  // bfloat16
    uint32_t dram_buffer_size = single_tile_size * num_tiles;
    uint32_t page_size = single_tile_size;

    distributed::DeviceLocalBufferConfig device_local_config{
        .page_size = page_size,
        .buffer_type = BufferType::DRAM,
    };

    distributed::ReplicatedBufferConfig buffer_config{
        .size = dram_buffer_size,
    };

    // Create DRAM buffers for inputs A, B, C, D and output
    auto input_a_buffer = distributed::MeshBuffer::create(buffer_config, device_local_config, mesh_device.get());
    auto input_b_buffer = distributed::MeshBuffer::create(buffer_config, device_local_config, mesh_device.get());
    auto input_c_buffer = distributed::MeshBuffer::create(buffer_config, device_local_config, mesh_device.get());
    auto input_d_buffer = distributed::MeshBuffer::create(buffer_config, device_local_config, mesh_device.get());
    auto output_buffer = distributed::MeshBuffer::create(buffer_config, device_local_config, mesh_device.get());

    // Create circular buffers
    // CB 0: Input A (phase 0)
    CircularBufferConfig cb_in0_config =
        CircularBufferConfig(single_tile_size * 2, {{0, tt::DataFormat::Float16_b}}).set_page_size(0, single_tile_size);
    CreateCircularBuffer(program, core, cb_in0_config);

    // CB 1: Input B (phase 0)
    CircularBufferConfig cb_in1_config =
        CircularBufferConfig(single_tile_size * 2, {{1, tt::DataFormat::Float16_b}}).set_page_size(1, single_tile_size);
    CreateCircularBuffer(program, core, cb_in1_config);

    // CB 2: Scratch buffer 1 for intermediate results (phase 0 output)
    // Must be large enough to hold all tiles since each phase completes fully before the next starts
    uint32_t scratch_cb_num_tiles = std::max(num_tiles, 2u);
    CircularBufferConfig cb_scratch1_config =
        CircularBufferConfig(single_tile_size * scratch_cb_num_tiles, {{2, tt::DataFormat::Float16_b}})
            .set_page_size(2, single_tile_size);
    CreateCircularBuffer(program, core, cb_scratch1_config);

    // CB 5: Scratch buffer 2 for intermediate results (phase 1 output)
    CircularBufferConfig cb_scratch2_config =
        CircularBufferConfig(single_tile_size * scratch_cb_num_tiles, {{5, tt::DataFormat::Float16_b}})
            .set_page_size(5, single_tile_size);
    CreateCircularBuffer(program, core, cb_scratch2_config);

    // CB 3: Input C (phase 1)
    CircularBufferConfig cb_in2_config =
        CircularBufferConfig(single_tile_size * 2, {{3, tt::DataFormat::Float16_b}}).set_page_size(3, single_tile_size);
    CreateCircularBuffer(program, core, cb_in2_config);

    // CB 4: Input D (phase 2)
    CircularBufferConfig cb_in3_config =
        CircularBufferConfig(single_tile_size * 2, {{4, tt::DataFormat::Float16_b}}).set_page_size(4, single_tile_size);
    CreateCircularBuffer(program, core, cb_in3_config);

    // CB 16: Output
    CircularBufferConfig cb_out_config = CircularBufferConfig(single_tile_size * 2, {{16, tt::DataFormat::Float16_b}})
                                             .set_page_size(16, single_tile_size);
    CreateCircularBuffer(program, core, cb_out_config);

    // Create reader kernel
    auto reader_kernel = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/fused/sequential/device/kernels/reader_sequential_eltwise.cpp",
        core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});

    // Create writer kernel
    auto writer_kernel = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/fused/sequential/device/kernels/writer_sequential_eltwise.cpp",
        core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});

    // Create compute kernel
    auto compute_kernel = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/fused/sequential/device/kernels/compute_sequential_eltwise.cpp",
        core,
        ComputeConfig{.math_fidelity = MathFidelity::HiFi4});

    // Set runtime args for compute
    SetRuntimeArgs(program, compute_kernel, core, {num_tiles, num_phases});

    // Generate stimulus data
    std::vector<uint32_t> packed_input_a = create_random_vector_of_bfloat16(dram_buffer_size, 1.0f, 42);
    std::vector<uint32_t> packed_input_b = create_random_vector_of_bfloat16(dram_buffer_size, 1.0f, 43);
    std::vector<uint32_t> packed_input_c = create_random_vector_of_bfloat16(dram_buffer_size, 1.0f, 44);
    std::vector<uint32_t> packed_input_d = create_random_vector_of_bfloat16(dram_buffer_size, 1.0f, 45);

    // Write input data to device
    distributed::EnqueueWriteMeshBuffer(cq, input_a_buffer, packed_input_a, false);
    distributed::EnqueueWriteMeshBuffer(cq, input_b_buffer, packed_input_b, false);
    distributed::EnqueueWriteMeshBuffer(cq, input_c_buffer, packed_input_c, false);
    distributed::EnqueueWriteMeshBuffer(cq, input_d_buffer, packed_input_d, false);

    // Set runtime args for reader
    SetRuntimeArgs(
        program,
        reader_kernel,
        core,
        {
            input_a_buffer->address(),
            input_b_buffer->address(),
            input_c_buffer->address(),
            input_d_buffer->address(),
            num_tiles,
            num_phases,
        });

    // Set runtime args for writer
    SetRuntimeArgs(
        program,
        writer_kernel,
        core,
        {
            output_buffer->address(),
            num_tiles,
        });

    mesh_workload.add_program(distributed::MeshCoordinateRange(mesh_device->shape()), std::move(program));

    // Execute
    distributed::EnqueueMeshWorkload(cq, mesh_workload, false);

    // Read output
    std::vector<uint32_t> result_vec;
    distributed::ReadShard(cq, result_vec, output_buffer, distributed::MeshCoordinate(0, 0));

    // Compute golden output: (A + B) * C + D
    std::vector<bfloat16> input_a_bf16(dram_buffer_size / sizeof(bfloat16));
    std::vector<bfloat16> input_b_bf16(dram_buffer_size / sizeof(bfloat16));
    std::vector<bfloat16> input_c_bf16(dram_buffer_size / sizeof(bfloat16));
    std::vector<bfloat16> input_d_bf16(dram_buffer_size / sizeof(bfloat16));

    std::memcpy(input_a_bf16.data(), packed_input_a.data(), dram_buffer_size);
    std::memcpy(input_b_bf16.data(), packed_input_b.data(), dram_buffer_size);
    std::memcpy(input_c_bf16.data(), packed_input_c.data(), dram_buffer_size);
    std::memcpy(input_d_bf16.data(), packed_input_d.data(), dram_buffer_size);

    std::vector<bfloat16> golden(input_a_bf16.size());
    for (size_t i = 0; i < input_a_bf16.size(); ++i) {
        float a = static_cast<float>(input_a_bf16[i]);
        float b = static_cast<float>(input_b_bf16[i]);
        float c = static_cast<float>(input_c_bf16[i]);
        float d = static_cast<float>(input_d_bf16[i]);

        float result;
        if (num_phases == 1) {
            result = a + b;
        } else if (num_phases == 2) {
            result = (a + b) * c;
        } else {
            result = (a + b) * c + d;
        }
        golden[i] = bfloat16(result);
    }

    // Compare results
    std::vector<bfloat16> result_bf16(result_vec.size() * 2);
    std::memcpy(result_bf16.data(), result_vec.data(), result_vec.size() * sizeof(uint32_t));

    bool pass = true;
    for (size_t i = 0; i < golden.size(); ++i) {
        float expected = static_cast<float>(golden[i]);
        float actual = static_cast<float>(result_bf16[i]);
        float diff = std::abs(expected - actual);
        float tolerance = 0.02f * std::max(1.0f, std::abs(expected));
        if (diff > tolerance) {
            log_error(LogTest, "Mismatch at index {}: expected={}, actual={}, diff={}", i, expected, actual, diff);
            pass = false;
            if (i > 10) {
                log_error(LogTest, "Too many errors, stopping comparison");
                break;
            }
        }
    }

    EXPECT_TRUE(pass);
    log_info(LogTest, "Test {} for num_tiles={}, num_phases={}", pass ? "PASSED" : "FAILED", num_tiles, num_phases);
}

}  // namespace

TEST_F(UnitMeshCQFixture, TensixSequentialComputeSingleTileOnePhase) {
    for (auto& device : devices_) {
        run_sequential_compute_test(device, device->mesh_command_queue(), 1, 1);
    }
}

TEST_F(UnitMeshCQFixture, TensixSequentialComputeSingleTileTwoPhases) {
    for (auto& device : devices_) {
        run_sequential_compute_test(device, device->mesh_command_queue(), 1, 2);
    }
}

TEST_F(UnitMeshCQFixture, TensixSequentialComputeSingleTileThreePhases) {
    for (auto& device : devices_) {
        run_sequential_compute_test(device, device->mesh_command_queue(), 1, 3);
    }
}

TEST_F(UnitMeshCQFixture, TensixSequentialComputeMultipleTilesThreePhases) {
    for (auto& device : devices_) {
        run_sequential_compute_test(device, device->mesh_command_queue(), 16, 3);
    }
}
