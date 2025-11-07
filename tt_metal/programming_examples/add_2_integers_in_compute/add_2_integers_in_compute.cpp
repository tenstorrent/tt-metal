// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <random>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/bfloat16.hpp>
#include "tt-metalium/constants.hpp"
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/tensor/tensor.hpp>

using namespace tt;
using namespace tt::tt_metal;
#ifndef OVERRIDE_KERNEL_PREFIX
#define OVERRIDE_KERNEL_PREFIX ""
#endif
int main() {
    // Ensure printing from kernel is enabled (so we can see the output of the Data Movement kernels).
    char* env_var = std::getenv("TT_METAL_DPRINT_CORES");
    if (env_var == nullptr) {
        fmt::print(
            "WARNING: Please set the environment variable TT_METAL_DPRINT_CORES to 0,0 to see the output of the Data "
            "Movement kernels.\n");
        fmt::print("WARNING: For example, export TT_METAL_DPRINT_CORES=0,0\n");
    }

    // A MeshDevice is a software concept that allows developers to virtualize a cluster of connected devices as a
    // single object, maintaining uniform memory and runtime state across all physical devices. A UnitMesh is a 1x1
    // MeshDevice that allows users to interface with a single physical device.
    std::shared_ptr<distributed::MeshDevice> mesh_device = distributed::MeshDevice::create_unit_mesh(0);

    // In Metalium, submitting operations to the device is done through a command queue. This includes
    // uploading/downloading data to/from the device, and executing programs.
    // A MeshCommandQueue is a software concept that allows developers to submit operations to a MeshDevice.
    distributed::MeshCommandQueue& cq = mesh_device->mesh_command_queue();

    // A MeshWorkload is a collection of programs that are executed on a MeshDevice.
    // The specific physical devices that the workload is executed on are determined by the MeshCoordinateRange.
    distributed::MeshWorkload workload;
    distributed::MeshCoordinateRange device_range = distributed::MeshCoordinateRange(mesh_device->shape());
    // A program is a collection of kernels. Note that unlike OpenCL/CUDA where every core must run the
    // same kernel at a given time. Metalium allows you to run different kernels on different cores
    // simultaneously.
    Program program = CreateProgram();
    // We will only be using one Tensix core for this particular example. As Tenstorrent processors are a 2D grid of
    // cores we can specify the core coordinates as (0, 0).
    constexpr CoreCoord core = {0, 0};

    // Most data on Tensix is stored in tiles. A tile is a 2D array of (usually) 32x32 values. And the Tensix uses
    // BFloat16 as the most well supported data type. Thus the tile size is 32x32x2 = 2048 bytes.
    constexpr uint32_t n_elements_per_tile = tt::constants::TILE_WIDTH * tt::constants::TILE_WIDTH;
    constexpr uint32_t single_tile_size = sizeof(bfloat16) * n_elements_per_tile;

#ifdef FALSE
    // MeshBuffer Creation:
    // To create a MeshBuffer, we need to specify the page size, the buffer type, and the size of the buffer.
    // For this example, we will be using a DRAM buffer.
    // A DeviceLocalBufferConfig is a configuration object that specifies the properties of a buffer that is allocated
    // on a single device. A ReplicatedBufferConfig is a configuration object that specifies the properties of a buffer
    // that is replicated across all devices in the Mesh.
    distributed::DeviceLocalBufferConfig dram_config{
        .page_size = single_tile_size,  // Number of bytes when round-robin between banks. Usually this is the same
                                        // as the tile size for efficiency.
        .buffer_type = tt_metal::BufferType::DRAM};  // Type of buffer (DRAM or L1(SRAM))
    distributed::ReplicatedBufferConfig distributed_buffer_config{
        .size = single_tile_size  // Size of the buffer in bytes
    };
    // Create 3 buffers in DRAM to hold the 2 input tiles and 1 output tile.
    auto src0_dram_buffer = distributed::MeshBuffer::create(distributed_buffer_config, dram_config, mesh_device.get());
    auto src1_dram_buffer = distributed::MeshBuffer::create(distributed_buffer_config, dram_config, mesh_device.get());
    auto dst_dram_buffer = distributed::MeshBuffer::create(distributed_buffer_config, dram_config, mesh_device.get());
#endif

    // Create 3 circular buffers. Think them like pipes moving data from one core to another. cb_src0 and cb_src1 are
    // used to move data from the reader kernel to the compute kernel. cb_dst is used to move data from the compute
    // kernel to the writer kernel. Each circular buffer is made up of 2 tiles. Thus when one tile is pushed and being
    // used by the receiving end, the sending end can get the next piece of data ready to be pushed. Overlapping the
    // operations. Leading to better performance. However there is a trade off, The more tiles in a circular buffer, the
    // more memory is used. And Circular buffers are backed by L1(SRAM) memory and L1 is a precious resource. The
    // hardware supports up to 32 circular buffers and they all act the same.
    constexpr uint32_t num_tiles = 1;
    auto make_cb_config = [&](CBIndex cb_index) {
        return CircularBufferConfig(num_tiles * single_tile_size, {{cb_index, DataFormat::Float16_b}})
            .set_page_size(cb_index, single_tile_size);
    };

    tt_metal::CreateCircularBuffer(program, core, make_cb_config(CBIndex::c_0));
    tt_metal::CreateCircularBuffer(program, core, make_cb_config(CBIndex::c_1));
    tt_metal::CreateCircularBuffer(program, core, make_cb_config(CBIndex::c_16));

    // Create the reader, writer and compute kernels. The kernels do the following:
    // * Reader: Reads data from the DRAM buffer and pushes it into the circular buffer.
    // * Compute: Waits for data to be available in the circular buffer, pops it, adds the two inputs together and
    // pushes the result
    //   into the output circular buffer.
    // * Writer: Waits for data to be available in the output circular buffer, pops it and writes it back into DRAM.
    // These kernels work together to form a pipeline. The reader reads data from the DRAM buffer and makes them
    // available in the compute kernel. The compute kernel does math and pushes the result into the writer kernel. The
    // writer kernel writes the result back to DRAM.
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

    // This kernel performs the actual addition of the two input tiles
    KernelHandle eltwise_binary_kernel_id = CreateKernel(
        program,
        OVERRIDE_KERNEL_PREFIX "add_2_integers_in_compute/kernels/compute/add_2_tiles.cpp",
        core,
        ComputeConfig{.math_fidelity = MathFidelity::HiFi4, .fp32_dest_acc_en = false, .math_approx_mode = false});

    // Create the data that will be used as input to the kernels.
    // src0 is a vector of bfloat16 values initialized to random values between 0.0f and 14.0f.
    // src1 is a vector of bfloat16 values initialized to random values between 0.0f and 8.0f.
    std::vector<bfloat16> src0_vec(n_elements_per_tile);
    std::vector<bfloat16> src1_vec(n_elements_per_tile);
    std::mt19937 rng(std::random_device{}());
    std::uniform_real_distribution<float> dist1(0.0f, 14.0f);
    std::uniform_real_distribution<float> dist2(0.0f, 8.0f);
    for (size_t i = 0; i < n_elements_per_tile; ++i) {
        src0_vec[i] = bfloat16(dist1(rng));
        src1_vec[i] = bfloat16(dist2(rng));
    }

#ifdef FALSE
    // Upload the data from host to the device. The last argument indicates if the operation is blocking or not.
    // Setting it to false allows the function to immediately return after queuing the operation, enabling
    // overlapping of data transfers with other operations. At the cost of user responsibility to ensure the data
    // is not released before the operation is complete.
    // In this case, we will wait for the program to finish eventually in the same scope, so we can set it
    // to false safely.
    EnqueueWriteMeshBuffer(cq, src0_dram_buffer, src0_vec, false);
    EnqueueWriteMeshBuffer(cq, src1_dram_buffer, src1_vec, false);
#endif

    TensorSpec tensor_spec(
        Shape({tt::constants::TILE_WIDTH, tt::constants::TILE_HEIGHT}),
        TensorLayout(DataType::BFLOAT16, Layout::TILE, MemoryConfig{}));
    Tensor src0_dram_tensor = Tensor::from_vector(src0_vec, tensor_spec, mesh_device.get());
    Tensor src1_dram_tensor = Tensor::from_vector(src1_vec, tensor_spec, mesh_device.get());
    Tensor dst_dram_tensor = allocate_tensor_on_device(tensor_spec, mesh_device.get());

    // Setup arguments for the kernels in the program.
    // Unlike OpenCL/CUDA, every kernel can have its own set of arguments.
    SetRuntimeArgs(
        program,
        binary_reader_kernel_id,
        core,
        {(uint32_t)src0_dram_tensor.mesh_buffer()->address(), (uint32_t)src1_dram_tensor.mesh_buffer()->address()});
    SetRuntimeArgs(program, eltwise_binary_kernel_id, core, {});
    SetRuntimeArgs(program, unary_writer_kernel_id, core, {(uint32_t)dst_dram_tensor.mesh_buffer()->address()});

    // Add the program to the workload and execute it.
    workload.add_program(device_range, std::move(program));
    distributed::EnqueueMeshWorkload(cq, workload, false);
    distributed::Finish(cq);

#ifdef FALSE
    // Data can be read from a MeshBuffer using the ReadShard function. This function is used to read data from a
    // specific shard of a MeshBuffer. The shard is specified by the MeshCoordinate. The last argument indicates if the
    // operation is blocking or not.
    std::vector<bfloat16> result_vec;
    distributed::EnqueueReadMeshBuffer(cq, result_vec, dst_dram_buffer, true);
#endif

    auto result_vec = dst_dram_tensor.to_vector<bfloat16>();
    // compare the results with the expected values.
    bool success = true;
    for (size_t i = 0; i < n_elements_per_tile; ++i) {
        float expected = static_cast<float>(src0_vec[i]) + static_cast<float>(src1_vec[i]);
        if (std::abs(expected - static_cast<float>(result_vec[i])) > 3e-1f) {
            fmt::print(
                stderr, "Mismatch at index {}: expected {}, got {}\n", i, expected, static_cast<float>(result_vec[i]));
            success = false;
        }
    }
    if (!success) {
        fmt::print("Error: Result does not match expected value!\n");
    } else {
        fmt::print("Success: Result matches expected value!\n");
    }
    mesh_device->close();
}
