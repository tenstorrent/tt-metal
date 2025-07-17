// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <random>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/bfloat16.hpp>
#include "tt-metalium/constants.hpp"

using namespace tt;
using namespace tt::tt_metal;
#ifndef OVERRIDE_KERNEL_PREFIX
#define OVERRIDE_KERNEL_PREFIX ""
#endif
int main() {
    // Initialize a device
    IDevice* device = CreateDevice(0);

    // Create a command queue and program
    // Command queue
    //    * Submit work (execute programs and read/write buffers) to the device
    // Program
    //    * Contains kernels that perform computations or data movement
    CommandQueue& cq = device->command_queue();
    Program program = CreateProgram();
    // We will only be using one Tensix core for this particular example. As Tenstorrent processors are a 2D grid of
    // cores we can specify the core coordinates as (0, 0).
    constexpr CoreCoord core = {0, 0};

    // The compute engines operates on tiles of data, which is usually a 32x32 grid of values.
    constexpr uint32_t n_elements_per_tile = tt::constants::TILE_WIDTH * tt::constants::TILE_WIDTH;
    constexpr uint32_t single_tile_size = sizeof(bfloat16) * n_elements_per_tile;
    tt_metal::InterleavedBufferConfig dram_config{
        .device = device,
        .size = single_tile_size,
        .page_size = single_tile_size,
        .buffer_type = tt_metal::BufferType::DRAM};

    // Create 3 buffers in DRAM to hold the 2 input tiles and 1 output tile.
    auto src0_dram_buffer = CreateBuffer(dram_config);
    auto src1_dram_buffer = CreateBuffer(dram_config);
    auto dst_dram_buffer = CreateBuffer(dram_config);

    // Use L1 circular buffers to set input and output buffers that the compute engine will use
    // Configure circular buffers for input and output
    constexpr uint32_t num_tiles = 1;
    auto make_cb_config = [&](uint32_t cb_index) {
        return CircularBufferConfig(num_tiles * single_tile_size, {{cb_index, tt::DataFormat::Float16_b}})
            .set_page_size(cb_index, single_tile_size);
    };

    tt_metal::CreateCircularBuffer(program, core, make_cb_config(CBIndex::c_0));
    tt_metal::CreateCircularBuffer(program, core, make_cb_config(CBIndex::c_1));
    tt_metal::CreateCircularBuffer(program, core, make_cb_config(CBIndex::c_16));

    // Create the kernels that will perform the data movement and compute operations.
    KernelHandle binary_reader_kernel_id = CreateKernel(
        program,
        OVERRIDE_KERNEL_PREFIX "add_2_integers_in_compute/kernels/dataflow/reader_binary_1_tile.cpp",
        core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});

    KernelHandle unary_writer_kernel_id = CreateKernel(
        program,
        OVERRIDE_KERNEL_PREFIX "add_2_integers_in_compute/kernels/dataflow/writer_1_tile.cpp",
        core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});

    /* Use the add_tiles operation in the compute kernel */
    KernelHandle eltwise_binary_kernel_id = CreateKernel(
        program,
        OVERRIDE_KERNEL_PREFIX "add_2_integers_in_compute/kernels/compute/add_2_tiles.cpp",
        core,
        ComputeConfig{.math_fidelity = MathFidelity::HiFi4, .fp32_dest_acc_en = false, .math_approx_mode = false});

    /* Create source data and write to DRAM */
    std::vector<bfloat16> src0_vec(n_elements_per_tile);
    std::vector<bfloat16> src1_vec(n_elements_per_tile);
    std::mt19937 rng(std::random_device{}());
    std::uniform_real_distribution<float> dist1(0.0f, 14.0f);
    std::uniform_real_distribution<float> dist2(0.0f, 8.0f);
    for (size_t i = 0; i < n_elements_per_tile; ++i) {
        src0_vec[i] = bfloat16(dist1(rng));
        src1_vec[i] = bfloat16(dist2(rng));
    }

    EnqueueWriteBuffer(cq, src0_dram_buffer, src0_vec, false);
    EnqueueWriteBuffer(cq, src1_dram_buffer, src1_vec, false);

    // Setup arguments for the kernels in the program
    SetRuntimeArgs(program, binary_reader_kernel_id, core, {src0_dram_buffer->address(), src1_dram_buffer->address()});
    SetRuntimeArgs(program, eltwise_binary_kernel_id, core, {});
    SetRuntimeArgs(program, unary_writer_kernel_id, core, {dst_dram_buffer->address()});

    EnqueueProgram(cq, program, false);
    Finish(cq);

    // Read the results from the destination DRAM buffer into host memory.
    std::vector<bfloat16> result_vec;
    EnqueueReadBuffer(cq, dst_dram_buffer, result_vec, true);

    bool success = true;
    for (size_t i = 0; i < n_elements_per_tile; ++i) {
        float expected = src0_vec[i].to_float() + src1_vec[i].to_float();
        if (result_vec[i].to_float() != expected) {
            fmt::print(stderr, "Mismatch at index {}: expected {}, got {}\n", i, expected, result_vec[i].to_float());
            success = false;
        }
    }
    if (!success) {
        fmt::print("Error: Result does not match expected value!\n");
    } else {
        fmt::print("Success: Result matches expected value!\n");
    }
    CloseDevice(device);
}
