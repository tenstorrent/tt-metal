// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/bfloat16.hpp>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>
#include <string_view>
#include <vector>
#include "tt-metalium/base_types.hpp"

using namespace tt::tt_metal;
#ifndef OVERRIDE_KERNEL_PREFIX
#define OVERRIDE_KERNEL_PREFIX ""
#endif
int main(int argc, char** argv) {
    bool pass = true;

    // clang-format off
    try {
        // Initialize the device (here we use the 1st device, but you can use any device)
        constexpr int device_id = 0;
        IDevice* device = CreateDevice(device_id);

        // In Metalium, submitting operations to the device is done through a command queue. This includes
        // uploading/downloading data to/from the device, and executing programs.
        CommandQueue& cq = device->command_queue();
        // A program is a collection of kernels. Note that unlike OpenCL/CUDA where every core must run the
        // same kernel at a given time. Metalium allows you to run different kernels on different cores
        // simultaneously.
        Program program = CreateProgram();

        // This example program will only use 1 Tensix core. So we set the core to {0, 0}.
        constexpr CoreCoord core = {0, 0};

        // Define some constants that will be used throughout the program.
        // * Processing 64 tiles
        // * Each tile is 32x32 elements
        // * Each element is a bfloat16 (2 bytes)
        constexpr uint32_t n_tiles = 64;
        constexpr uint32_t elements_per_tile = tt::constants::TILE_WIDTH * tt::constants::TILE_HEIGHT;
        constexpr uint32_t tile_size_bytes = sizeof(bfloat16) * elements_per_tile;

        // Create 3 buffers on DRAM. These will hold the input and output data. src0 and src1 are the input buffers, dst is the
        // output buffer.
        InterleavedBufferConfig config{
            .device = device,                       // The device to create the buffer on
            .size = n_tiles * tile_size_bytes,      // The size of the buffer in bytes
            .page_size = tile_size_bytes,           // The page size of the buffer in bytes. Unlike the `loopback` example, we
                                                    // need the page size to be the same as the tile size for a large portion of
                                                    // the NoC transfer APIs to work.
            .buffer_type = BufferType::DRAM};       // This is a DRAM buffer.
        auto src0_dram_buffer = CreateBuffer(config);
        auto src1_dram_buffer = CreateBuffer(config);
        auto dst_dram_buffer = CreateBuffer(config);

        // Initialize the input buffers with random data. For this example, src0 is a random vector of bfloat16 values
        std::mt19937 rng(std::random_device{}());
        std::uniform_real_distribution<float> distribution(0.0f, 1.0f);
        std::vector<bfloat16> a_data(elements_per_tile * n_tiles);
        for(auto& val : a_data) {
            val = bfloat16(distribution(rng));
        }

        // ... and src1 is a vector of bfloat16 values initialized to -1.0f.
        constexpr float val_to_add = -1.0f;
        std::vector<bfloat16> b_data(elements_per_tile * n_tiles, bfloat16(val_to_add));

        // Upload the data from host to the device.
        EnqueueWriteBuffer(cq, src0_dram_buffer, a_data, false);
        EnqueueWriteBuffer(cq, src1_dram_buffer, b_data, false);

        // Create 3 circular buffers. Think them like pipes moving data from one core to another. cb_src0 and cb_src1 are used to
        // move data from the reader kernel to the compute kernel. cb_dst is used to move data from the compute kernel to the writer
        // kernel. Each circular buffer is made up of 2 tiles. Thus when one tile is pushed and being used by the receiving end, the
        // sending end can get the next piece of data ready to be pushed. Overlapping the operations. Leading to better performance.
        // However there is a trade off, The more tiles in a circular buffer, the more memory is used. And Circular buffers are
        // backed by L1(SRAM) memory and L1 is a precious resource.
        // The hardware supports up to 32 circular buffers and they all act the same.
        constexpr uint32_t tiles_per_cb = 2;
        tt::CBIndex src0_cb_index = tt::CBIndex::c_0;
        CreateCircularBuffer(program, core, CircularBufferConfig(
            /*total_size=*/tiles_per_cb * tile_size_bytes,                    // The total size of the circular buffer in bytes
            /*data_format_spec=*/{{src0_cb_index, tt::DataFormat::Float16_b}})// The circular buffer index and data format it'll hold
            .set_page_size(src0_cb_index, tile_size_bytes));                  // Since we will be sending one tile at a time, we set
                                                                              // the page size to the tile size (and thus
                                                                              // total_size / page_size = tiles_per is the number of
                                                                              // entries in the circular buffer)
        tt::CBIndex src1_cb_index = tt::CBIndex::c_1;
        CreateCircularBuffer(program, core, CircularBufferConfig(
            /*total_size=*/tiles_per_cb * tile_size_bytes,
            /*data_format_spec=*/{{src1_cb_index, tt::DataFormat::Float16_b}})
            .set_page_size(src1_cb_index, tile_size_bytes));
        tt::CBIndex dst_cb_index = tt::CBIndex::c_16;
        CreateCircularBuffer(program, core, CircularBufferConfig(
            /*total_size=*/tiles_per_cb * tile_size_bytes,
            /*data_format_spec=*/{{dst_cb_index, tt::DataFormat::Float16_b}})
            .set_page_size(dst_cb_index, tile_size_bytes));

        // Create the reader, writer and compute kernels. The kernels do the following:
        // * Reader: Reads data from the DRAM buffer and pushes it into the circular buffer.
        // * Compute: Waits for data to be available in the circular buffer, pops it, adds the two inputs together and pushes the result
        //   into the output circular buffer.
        // * Writer: Waits for data to be available in the output circular buffer, pops it and writes it back into DRAM.
        // These kernels work together to form a pipeline. The reader reads data from the DRAM buffer and makes them available in the
        // compute kernel. The compute kernel does math and pushes the result into the writer kernel. The writer kernel writes the result
        // back to DRAM.
        auto reader = CreateKernel(
            program,
            OVERRIDE_KERNEL_PREFIX "eltwise_binary/kernels/dataflow/read_tiles.cpp",
            core,
            DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});
        auto writer = CreateKernel(
            program,
            OVERRIDE_KERNEL_PREFIX "eltwise_binary/kernels/dataflow/write_tile.cpp",
            core,
            DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});
        auto compute = CreateKernel(
            program,
            OVERRIDE_KERNEL_PREFIX "eltwise_binary/kernels/compute/tiles_add.cpp",
            core,
            ComputeConfig{.math_fidelity = MathFidelity::HiFi4});   // There's different math fidelity modes (for the tensor engine)
                                                                // that trade off performance for accuracy. HiFi4 is the most accurate
                                                                // mode. The other modes are HiFi3, HiFi2, HiFi1 and LoFi. The
                                                                // difference between them is the number of bits used during computation.

        // Set the runtime arguments for the kernels. This also registers
        // the kernels with the program.
        SetRuntimeArgs(program, reader, core, {src0_dram_buffer->address(), src1_dram_buffer->address(), n_tiles});
        SetRuntimeArgs(program, writer, core, {dst_dram_buffer->address(), n_tiles});
        SetRuntimeArgs(program, compute, core, {n_tiles});

        // We have setup the program. Now we queue the kernel for execution. The final argument is set to false. This indicates
        // to Metalium that the operation is non-blocking. The function is allowed to return upon the kernel being queued. We must
        // ensure that the kernel is finished before we read the output buffer. This is done by calling Finish(cq) which waits until
        // all operations in the command queue are finished. This is equivalent to calling EnqueueProgram(cq, program, true); telling
        // Metalium to wait until the program is finished before returning.
        EnqueueProgram(cq, program, false);
        Finish(cq);
        // Equivalently:
        // EnqueueProgram(cq, program, true);

        // Read the output buffer and compare it with the expected output.
        std::vector<bfloat16> result_vec;
        EnqueueReadBuffer(cq, dst_dram_buffer, result_vec, true);

        constexpr float eps = 1e-2f; // loose tolerance because of the nature of bfloat16
        TT_FATAL(result_vec.size() == a_data.size(), "Result vector size mismatch");
        for (size_t i = 0; i < result_vec.size(); ++i) {
            const float expected = a_data[i].to_float() + val_to_add;
            const float actual = result_vec[i].to_float();

            if (std::abs(expected - actual) > eps) {
                pass = false;
                fmt::print(stderr, "Result mismatch at index {}: expected {}, got {}\n", i, expected, actual);
            }
        }

        // Finally, we close the device.
        pass &= CloseDevice(device);
    } catch (const std::exception& e) {
        fmt::print(stderr, "Test failed with exception!\n");
        fmt::print(stderr, "{}\n", e.what());

        throw;
    }
    // clang-format on

    if (pass) {
        fmt::print("Test Passed\n");
    } else {
        TT_THROW("Test Failed");
    }

    return 0;
}
