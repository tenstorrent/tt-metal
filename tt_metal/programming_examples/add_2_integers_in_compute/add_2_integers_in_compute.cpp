// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <random>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/bfloat16.hpp>
#include "tt-metalium/constants.hpp"
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/experimental/metal2_host_api/program.hpp>
#include <tt-metalium/tensor/host_tensor.hpp>
#include <tt-metalium/tensor/mesh_tensor.hpp>

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
    // We will only be using one Tensix core for this particular example. As Tenstorrent processors are a 2D grid of
    // cores we can specify the core coordinates as (0, 0).
    constexpr experimental::NodeCoord core = {0, 0};

    // Most data on Tensix is stored in tiles. A tile is a 2D array of (usually) 32x32 values. And the Tensix uses
    // BFloat16 as the most well supported data type. Thus the tile size is 32x32x2 = 2048 bytes.
    constexpr uint32_t n_elements_per_tile = tt::constants::TILE_WIDTH * tt::constants::TILE_WIDTH;
    constexpr uint32_t single_tile_size = sizeof(bfloat16) * n_elements_per_tile;

    // MeshTensor Creation:
    // To create a MeshTensor, we need to specify the page size, the buffer type, and the size of the buffer.
    // For this example, we will be using a DRAM buffer.
    // A TensorSpec is a configuration object that specifies the properties of a tensor. The tensor is replicated
    // across all devices in the Mesh.
    // Each row of a row-major tensor is one page. The page size is the number of bytes when round-robin between
    // banks. Usually this is the same as the tile size for efficiency, so a row holds one tile's worth of elements.
    // The size of the buffer in bytes is the number of rows (1) times the page size.
    TensorSpec dram_tensor_spec(
        Shape({1, n_elements_per_tile}),
        TensorLayout(
            DataType::BFLOAT16,
            PageConfig(Layout::ROW_MAJOR),
            MemoryConfig{TensorMemoryLayout::INTERLEAVED, BufferType::DRAM}));  // Type of buffer (DRAM or L1(SRAM))
    // Create 3 tensors in DRAM to hold the 2 input tiles and 1 output tile.
    MeshTensor src0_dram_tensor = MeshTensor::allocate_on_device(*mesh_device, dram_tensor_spec);
    MeshTensor src1_dram_tensor = MeshTensor::allocate_on_device(*mesh_device, dram_tensor_spec);
    MeshTensor dst_dram_tensor = MeshTensor::allocate_on_device(*mesh_device, dram_tensor_spec);

    // Names identify the kernels and the dataflow buffers within the program spec.
    const experimental::KernelSpecName binary_reader_kernel{"binary_reader"};
    const experimental::KernelSpecName unary_writer_kernel{"unary_writer"};
    const experimental::KernelSpecName eltwise_binary_kernel{"eltwise_binary"};
    const experimental::DFBSpecName src0_dfb{"src0"};
    const experimental::DFBSpecName src1_dfb{"src1"};
    const experimental::DFBSpecName dst_dfb{"dst"};

    // Describe 3 dataflow buffers. Think them like pipes moving data from one core to another. src0 and src1 are
    // used to move data from the reader kernel to the compute kernel. dst is used to move data from the compute
    // kernel to the writer kernel. Each dataflow buffer is made up of 2 tiles. Thus when one tile is pushed and being
    // used by the receiving end, the sending end can get the next piece of data ready to be pushed. Overlapping the
    // operations. Leading to better performance. However there is a trade off, The more tiles in a dataflow buffer,
    // the more memory is used. And dataflow buffers are backed by L1(SRAM) memory and L1 is a precious resource. The
    // hardware supports up to 32 dataflow buffers and they all act the same.
    constexpr uint32_t num_tiles = 1;
    auto make_dfb_spec = [&](const experimental::DFBSpecName& dfb_name) {
        return experimental::DataflowBufferSpec{
            .unique_id = dfb_name,
            .entry_size = single_tile_size,
            .num_entries = num_tiles,
            .data_format_metadata = DataFormat::Float16_b,
        };
    };

    // Describe the reader, writer and compute kernels. The kernels do the following:
    // * Reader: Reads data from the DRAM buffer and pushes it into the dataflow buffer.
    // * Compute: Waits for data to be available in the dataflow buffer, pops it, adds the two inputs together and
    // pushes the result
    //   into the output dataflow buffer.
    // * Writer: Waits for data to be available in the output dataflow buffer, pops it and writes it back into DRAM.
    // These kernels work together to form a pipeline. The reader reads data from the DRAM buffer and makes them
    // available in the compute kernel. The compute kernel does math and pushes the result into the writer kernel. The
    // writer kernel writes the result back to DRAM.
    experimental::KernelSpec binary_reader_kernel_spec{
        .unique_id = binary_reader_kernel,
        .source = OVERRIDE_KERNEL_PREFIX "add_2_integers_in_compute/kernels/dataflow/reader_binary_1_tile.cpp",
        .dfb_bindings = {experimental::ProducerOf(src0_dfb, "in0"), experimental::ProducerOf(src1_dfb, "in1")},
        .runtime_arg_schema = {.runtime_arg_names = {"in0_addr", "in1_addr"}},
        .hw_config =
            experimental::DataMovementHardwareConfig{
                .config_1xx =
                    experimental::DataMovementHardwareConfig::DataMovement1XXConfig{
                        .processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default},
            },
    };

    experimental::KernelSpec unary_writer_kernel_spec{
        .unique_id = unary_writer_kernel,
        .source = OVERRIDE_KERNEL_PREFIX "add_2_integers_in_compute/kernels/dataflow/writer_1_tile.cpp",
        .dfb_bindings = {experimental::ConsumerOf(dst_dfb, "out0")},
        .runtime_arg_schema = {.runtime_arg_names = {"dst_addr"}},
        .hw_config =
            experimental::DataMovementHardwareConfig{
                .config_1xx =
                    experimental::DataMovementHardwareConfig::DataMovement1XXConfig{
                        .processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default},
            },
    };

    // This kernel performs the actual addition of the two input tiles
    experimental::KernelSpec eltwise_binary_kernel_spec{
        .unique_id = eltwise_binary_kernel,
        .source = OVERRIDE_KERNEL_PREFIX "add_2_integers_in_compute/kernels/compute/add_2_tiles.cpp",
        .compiler_options = {.opt_level = KernelBuildOptLevel::O3},
        .dfb_bindings =
            {experimental::ConsumerOf(src0_dfb, "in0"),
             experimental::ConsumerOf(src1_dfb, "in1"),
             experimental::ProducerOf(dst_dfb, "out0")},
        .hw_config =
            experimental::ComputeHardwareConfig{
                .fpu_math_fidelity = tt::tt_metal::MathFidelity::HiFi4,
                .sfpu_precision_mode = Precision::Precise,
                .enable_32_bit_dest = false,
            },
    };

    // A program is a collection of kernels. Note that unlike OpenCL/CUDA where every core must run the
    // same kernel at a given time. Metalium allows you to run different kernels on different cores
    // simultaneously.
    // A program spec describes the program: its kernels, the dataflow buffers between them, and the cores the
    // kernels run on (the work units).
    experimental::ProgramSpec program_spec{
        .name = "add_2_integers_in_compute",
        .kernels = {binary_reader_kernel_spec, unary_writer_kernel_spec, eltwise_binary_kernel_spec},
        .dataflow_buffers = {make_dfb_spec(src0_dfb), make_dfb_spec(src1_dfb), make_dfb_spec(dst_dfb)},
        .work_units =
            {{.name = "add_2_integers_in_compute",
              .kernels = {binary_reader_kernel, unary_writer_kernel, eltwise_binary_kernel},
              .target_nodes = core}},
    };
    Program program = experimental::MakeProgramFromSpec(*mesh_device, program_spec);

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

    // Upload the data from host to the device. The operation is not blocking.
    // It allows the function to immediately return after queuing the operation, enabling
    // overlapping of data transfers with other operations. At the cost of user responsibility to ensure the data
    // (the host tensor) is not released before the operation is complete.
    // In this case, we will wait for the program to finish eventually in the same scope, so this is safe.
    HostTensor src0_host_tensor = HostTensor::from_vector(src0_vec, dram_tensor_spec);
    HostTensor src1_host_tensor = HostTensor::from_vector(src1_vec, dram_tensor_spec);
    cq.enqueue_write_tensor(src0_host_tensor, src0_dram_tensor);
    cq.enqueue_write_tensor(src1_host_tensor, src1_dram_tensor);

    // Setup arguments for the kernels in the program.
    // Unlike OpenCL/CUDA, every kernel can have its own set of arguments.
    // The compute kernel has no arguments.
    experimental::ProgramRunArgs run_args{
        .kernel_run_args =
            {{.kernel = binary_reader_kernel,
              .runtime_arg_values = experimental::MakeRuntimeArgsForSingleNode(
                  core,
                  {{"in0_addr", static_cast<uint32_t>(src0_dram_tensor.address())},
                   {"in1_addr", static_cast<uint32_t>(src1_dram_tensor.address())}})},
             {.kernel = unary_writer_kernel,
              .runtime_arg_values = experimental::MakeRuntimeArgsForSingleNode(
                  core, {{"dst_addr", static_cast<uint32_t>(dst_dram_tensor.address())}})}},
    };
    experimental::SetProgramRunArgs(program, run_args);

    // Add the program to the workload and execute it.
    workload.add_program(device_range, std::move(program));
    distributed::EnqueueMeshWorkload(cq, workload, false);
    distributed::Finish(cq);

    // Data can be read from a MeshTensor using the enqueue_read_tensor function. The last argument indicates if the
    // operation is blocking or not.
    std::vector<bfloat16> result_vec = cq.enqueue_read_tensor(dst_dram_tensor, true).to_vector<bfloat16>();

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
