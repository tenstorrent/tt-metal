// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cmath>
#include <random>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

using namespace tt;
using namespace tt::tt_metal;

/*
 * 1. Host creates one vector of data.
 * 2. Device eltwise performs a unary SFPU operation on the data.
 * 3. Read result back and compare to golden.
 * */
#ifndef OVERRIDE_KERNEL_PREFIX
#define OVERRIDE_KERNEL_PREFIX ""
#endif
int main() {
    bool pass = true;

    try {
        // Initialize the device (here we use the 1st device, but you can use any device)
        constexpr int device_id = 0;
        std::shared_ptr<distributed::MeshDevice> mesh_device = distributed::MeshDevice::create_unit_mesh(device_id);

        // In Metalium, submitting operations to the device is done through a command queue. This includes
        // uploading/downloading data to/from the device, and executing programs.
        distributed::MeshCommandQueue& cq = mesh_device->mesh_command_queue();
        // A program is a collection of kernels. Note that unlike OpenCL/CUDA where every core must run the
        // same kernel at a given time. Metalium allows you to run different kernels on different cores
        // simultaneously.
        distributed::MeshWorkload workload;
        distributed::MeshCoordinateRange device_range = distributed::MeshCoordinateRange(mesh_device->shape());
        Program program = CreateProgram();

        // This example program will only use 1 Tensix core. So we set the core to {0, 0}.
        constexpr CoreCoord core = {0, 0};

        constexpr uint32_t n_tiles = 64;
        constexpr uint32_t elements_per_tile = tt::constants::TILE_WIDTH * tt::constants::TILE_HEIGHT;
        constexpr uint32_t tile_size_bytes = sizeof(bfloat16) * elements_per_tile;

        // Allocate DRAM buffers for the input and output data.
        distributed::DeviceLocalBufferConfig dram_config{
            .page_size = tile_size_bytes, .buffer_type = tt_metal::BufferType::DRAM};
        distributed::ReplicatedBufferConfig buffer_config{
            .size = tile_size_bytes * n_tiles};  // Replicated across the mesh (unit mesh ⇒ single device)

        // Allocate an input and output buffer on DRAM. We will perform a unary operation on the input buffer and write
        // the result.
        std::shared_ptr<distributed::MeshBuffer> src0_dram_buffer =
            distributed::MeshBuffer::create(buffer_config, dram_config, mesh_device.get());
        std::shared_ptr<distributed::MeshBuffer> dst_dram_buffer =
            distributed::MeshBuffer::create(buffer_config, dram_config, mesh_device.get());
        // Each handle refers to the per-device allocation; on a 1x1 mesh it's a single device buffer.

        // Allocate 2 circular buffers for input and output.
        constexpr uint32_t src0_cb_index = tt::CBIndex::c_0;
        constexpr uint32_t num_input_tiles = 2;
        CircularBufferConfig cb_src0_config =
            CircularBufferConfig(num_input_tiles * tile_size_bytes, {{src0_cb_index, tt::DataFormat::Float16_b}})
                .set_page_size(src0_cb_index, tile_size_bytes);
        tt_metal::CreateCircularBuffer(program, core, cb_src0_config);

        constexpr uint32_t output_cb_index = tt::CBIndex::c_16;
        CircularBufferConfig cb_output_config =
            CircularBufferConfig(num_input_tiles * tile_size_bytes, {{output_cb_index, tt::DataFormat::Float16_b}})
                .set_page_size(output_cb_index, tile_size_bytes);
        tt_metal::CreateCircularBuffer(program, core, cb_output_config);

        // Create the 2 data movement kernels and the compute kernel.
        std::vector<uint32_t> reader_compile_time_args;
        TensorAccessorArgs(*src0_dram_buffer).append_to(reader_compile_time_args);
        KernelHandle unary_reader_kernel_id = CreateKernel(
            program,
            OVERRIDE_KERNEL_PREFIX "eltwise_sfpu/kernels/dataflow/read_tile.cpp",
            core,
            DataMovementConfig{
                .processor = DataMovementProcessor::RISCV_1,
                .noc = NOC::RISCV_1_default,
                .compile_args = reader_compile_time_args});

        std::vector<uint32_t> writer_compile_time_args;
        TensorAccessorArgs(*dst_dram_buffer).append_to(writer_compile_time_args);
        KernelHandle unary_writer_kernel_id = CreateKernel(
            program,
            OVERRIDE_KERNEL_PREFIX "eltwise_sfpu/kernels/dataflow/write_tile.cpp",
            core,
            DataMovementConfig{
                .processor = DataMovementProcessor::RISCV_0,
                .noc = NOC::RISCV_0_default,
                .compile_args = writer_compile_time_args});
        KernelHandle eltwise_sfpu_kernel_id = CreateKernel(
            program,
            OVERRIDE_KERNEL_PREFIX "eltwise_sfpu/kernels/compute/eltwise_sfpu.cpp",
            core,
            ComputeConfig{
                .math_fidelity = MathFidelity::HiFi4,
                .math_approx_mode = false,
            });

        // Initialize the input data with random values and use as the input to the kernel.
        std::mt19937 rng(std::random_device{}());
        std::uniform_real_distribution<float> dist(0.f, 1.0f);
        std::vector<bfloat16> src0_vec(n_tiles * elements_per_tile);
        for (bfloat16& v : src0_vec) {
            v = bfloat16(dist(rng));
        }

        // Write the data on host to the input buffer on the device.
        // setting blocking to false allows us to overlap the data movement and following host operations (
        // setting kerenel args) in this case
        distributed::EnqueueWriteMeshBuffer(cq, src0_dram_buffer, src0_vec, /*blocking=*/false);

        // Set up the runtime arguments for the kernels.
        SetRuntimeArgs(program, eltwise_sfpu_kernel_id, core, {n_tiles});
        SetRuntimeArgs(
            program,
            unary_reader_kernel_id,
            core,
            {
                src0_dram_buffer->address(),
                n_tiles,
            });

        SetRuntimeArgs(program, unary_writer_kernel_id, core, {dst_dram_buffer->address(), n_tiles});

        // Enqueue the program as a mesh workload (non-blocking) and wait for completion before reading results.
        workload.add_program(device_range, std::move(program));
        distributed::EnqueueMeshWorkload(cq, workload, false);
        distributed::Finish(cq);

        // Read the result (from shard at mesh coordinate {0,0} on a unit mesh) and compare to our expected result.
        std::vector<bfloat16> result_vec;
        distributed::EnqueueReadMeshBuffer(cq, result_vec, dst_dram_buffer, true);

        // Compute the same thing on CPU for comparison

        // Compare the result to the golden vector. Loose tolerance of 2e-2f because bfloat16 is not as accurate as 32
        // bit float.
        constexpr float eps = 5e-2f;
        for (uint32_t i = 0; i < result_vec.size(); ++i) {
            float expected = static_cast<float>(bfloat16(std::exp(static_cast<float>(src0_vec[i]))));
            float result = static_cast<float>(result_vec[i]);
            if (std::abs(expected - result) > eps) {
                pass = false;
                fmt::print(stderr, "Result mismatch at index {}: {} != {}\n", i, expected, result);
            }
        }

        // Finally, close the device.
        pass &= mesh_device->close();

    } catch (const std::exception& e) {
        fmt::print(stderr, "Test failed with exception!\n");
        fmt::print(stderr, "{}\n", e.what());

        throw;
    }

    if (pass) {
        fmt::print("Test Passed\n");
    } else {
        TT_THROW("Test Failed");
    }

    return 0;
}
