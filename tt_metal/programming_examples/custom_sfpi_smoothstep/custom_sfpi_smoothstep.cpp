// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <fmt/ostream.h>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>
#include <vector>

using namespace tt::tt_metal;
#ifndef OVERRIDE_KERNEL_PREFIX
#define OVERRIDE_KERNEL_PREFIX ""
#endif
int main(int /*argc*/, char** /*argv*/) {
    bool pass = true;

    // clang-format off
    try {
        // Create a 1x1 mesh on device 0 (same API scales to multi-device meshes)
        constexpr int device_id = 0;
        auto mesh_device = distributed::MeshDevice::create_unit_mesh(device_id);

        // Submit work via the mesh command queue: uploads/downloads and program execution.
        distributed::MeshCommandQueue& cq = mesh_device->mesh_command_queue();
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
        constexpr uint32_t dram_buffer_size = tile_size_bytes * n_tiles;

        // Configure mesh buffers. Use single-tile page size so transfers operate tile-by-tile.
        distributed::DeviceLocalBufferConfig dram_config{
            .page_size = tile_size_bytes,    // Number of bytes when round-robin between banks
            .buffer_type = BufferType::DRAM  // Type of buffer (DRAM or L1)
        };
        distributed::ReplicatedBufferConfig dram_buffer_config{
            // Size per device (replicated across mesh). Since we are operating on a unit mesh this is the total size.
            .size = dram_buffer_size};

        // Allocate the buffers (replicated across mesh; on unit mesh ⇒ single device allocation)
        // src0 is input buffer; dst is output buffer
        auto src0_dram_buffer = distributed::MeshBuffer::create(dram_buffer_config, dram_config, mesh_device.get());
        auto dst_dram_buffer = distributed::MeshBuffer::create(dram_buffer_config, dram_config, mesh_device.get());

        // Initialize the input buffers with random data. For this example, src0 is a random vector of bfloat16 values
        std::mt19937 rng(std::random_device{}());
        std::uniform_real_distribution<float> distribution(0.0f, 1.0f);
        std::vector<bfloat16> a_data(elements_per_tile * n_tiles);
        for(auto& val : a_data) {
            val = bfloat16(distribution(rng));
        }


        // Upload the data from host to the device.
        distributed::EnqueueWriteMeshBuffer(cq, src0_dram_buffer, a_data, false);

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
        std::vector<uint32_t> reader_compile_time_args;
        TensorAccessorArgs(*src0_dram_buffer->get_backing_buffer()).append_to(reader_compile_time_args);
        auto reader = CreateKernel(
            program,
            OVERRIDE_KERNEL_PREFIX "custom_sfpi_smoothstep/kernels/dataflow/read_tiles.cpp",
            core,
            DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default, .compile_args = reader_compile_time_args});
        std::vector<uint32_t> writer_compile_time_args;
        TensorAccessorArgs(*dst_dram_buffer->get_backing_buffer()).append_to(writer_compile_time_args);
        auto writer = CreateKernel(
            program,
            OVERRIDE_KERNEL_PREFIX "custom_sfpi_smoothstep/kernels/dataflow/write_tile.cpp",
            core,
            DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default, .compile_args = writer_compile_time_args});
        auto compute = CreateKernel(
            program,
            OVERRIDE_KERNEL_PREFIX "custom_sfpi_smoothstep/kernels/compute/tiles_smoothstep.cpp",
            core,
            ComputeConfig{
                .fp32_dest_acc_en = false, // We don't need the destination accumulator to be FP32 as input and output are BFP16
            });

        // Set the runtime arguments for the kernels. This also registers
        // the kernels with the program.
        SetRuntimeArgs(program, reader, core, {src0_dram_buffer->address(), n_tiles});
        SetRuntimeArgs(program, writer, core, {dst_dram_buffer->address(), n_tiles});
        SetRuntimeArgs(program, compute, core, {n_tiles});

        // A MeshWorkload is a collection of programs that will be executed on the mesh. Each workload is
        // local to a single device. Here we create a workload for our single-device mesh.
        distributed::MeshWorkload workload;
        distributed::MeshCoordinateRange device_range = distributed::MeshCoordinateRange(mesh_device->shape());
        // Add the program to the workload for the mesh.
        workload.add_program(device_range, std::move(program));
        // Enqueue the workload for execution on the mesh (non-blocking) and wait for completion before reading back.
        distributed::EnqueueMeshWorkload(cq, workload, /*blocking=*/false);
        distributed::Finish(cq);
        // NOTE: The above is equivalent to a blocking enqueue of the workload.

        // Read the output buffer and compare it with the expected output.
        std::vector<bfloat16> result_vec;
        distributed::EnqueueReadMeshBuffer(cq, result_vec, dst_dram_buffer, /*blocking*/ true);

        constexpr float eps = 1e-2f; // loose tolerance because of the nature of bfloat16
        TT_FATAL(result_vec.size() == a_data.size(), "Result vector size mismatch");
        for (size_t i = 0; i < result_vec.size(); ++i) {
            auto smoothstep = [](float edge0, float edge1, float x) {
                // Scale, bias and saturate x to 0..1 range
                x = (x - edge0) / (edge1 - edge0);
                x = std::clamp(x, 0.0f, 1.0f);
                // Evaluate polynomial
                return x * x * (3 - 2 * x);
            };
            const float expected = smoothstep(0.0f, 1.0f, static_cast<float>(a_data[i]));
            const float actual = static_cast<float>(result_vec[i]);

            if (std::abs(expected - actual) > eps) {
                pass = false;
                fmt::print(stderr, "Result mismatch at index {}: expected {}, got {}\n", i, expected, actual);
            }
        }

        // Finally, we close the device.
        if (!mesh_device->close()) {
            pass = false;
        }
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
