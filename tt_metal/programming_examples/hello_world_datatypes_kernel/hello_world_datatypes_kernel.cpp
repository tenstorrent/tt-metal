// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/device.hpp>
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
    // This example demonstrates that though the RISC-V cores (Both the data movement and compute cores) are
    // RV32IM and the Tensix relies on the SFPU and FPU attached to the compute cores to perform the bulk of the
    // floating point operations, it is still possible to operate on floating point data types directly on the
    // RISC-V cores as they are fully programmable and the compiler can generate the necessary software floating point
    // operations.

    char* env_var = std::getenv("TT_METAL_DPRINT_CORES");
    if (env_var == nullptr) {
        fmt::print(
            "WARNING: Please set the environment variable TT_METAL_DPRINT_CORES to 0,0 to see the output of the Data "
            "Movement kernels.\n");
        fmt::print("WARNING: For example, export TT_METAL_DPRINT_CORES=0,0\n");
    }

    // Initialize mesh device, command queue, workload, and device range
    constexpr experimental::NodeCoord core = {0, 0};
    std::shared_ptr<distributed::MeshDevice> mesh_device = distributed::MeshDevice::create_unit_mesh(0);
    distributed::MeshCommandQueue& cq = mesh_device->mesh_command_queue();
    distributed::MeshWorkload workload;
    distributed::MeshCoordinateRange device_range = distributed::MeshCoordinateRange(mesh_device->shape());

    // Define and create a DRAM-backed replicated mesh tensor with float data type
    // The tensor is allocated identically on each device in the mesh (unit mesh ⇒ single device)
    // Each row of a row-major tensor is one page, so a single row of buffer_size bytes gives one page of buffer_size.
    constexpr uint32_t buffer_size = 2 * 1024;
    TensorSpec dram_tensor_spec(
        Shape({1, buffer_size / sizeof(float)}),
        TensorLayout(
            DataType::FLOAT32,
            PageConfig(Layout::ROW_MAJOR),
            MemoryConfig{TensorMemoryLayout::INTERLEAVED, BufferType::DRAM}));
    MeshTensor dram_tensor = MeshTensor::allocate_on_device(*mesh_device, dram_tensor_spec);

    // Configure and create an L1 dataflow buffer (to move data from DRAM to L1)
    // Set entry size equal to buffer_size so one entry equals one transfer unit
    // The kernel both fills and reads this dataflow buffer, so it is bound to the kernel as producer and consumer.
    const experimental::DFBSpecName src0_dfb{"src0"};
    experimental::DataflowBufferSpec src0_dfb_spec{
        .unique_id = src0_dfb,
        .entry_size = buffer_size,
        .num_entries = 1,
        .data_format_metadata = tt::DataFormat::Float16_b,
    };

    // Configure and create the kernel
    const experimental::KernelSpecName data_reader_kernel{"data_reader"};
    experimental::KernelSpec data_reader_kernel_spec{
        .unique_id = data_reader_kernel,
        .source = OVERRIDE_KERNEL_PREFIX "hello_world_datatypes_kernel/kernels/dataflow/float_dataflow_kernel.cpp",
        .dfb_bindings = {experimental::ProducerOf(src0_dfb, "src0"), experimental::ConsumerOf(src0_dfb, "src0")},
        .runtime_arg_schema = {.runtime_arg_names = {"dram_addr"}},
        .hw_config =
            experimental::DataMovementHardwareConfig{
                .config_1xx =
                    experimental::DataMovementHardwareConfig::DataMovement1XXConfig{
                        .processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default},
            },
    };

    // A program spec describes the program: its kernels, its dataflow buffers, and the cores the kernels run on
    // (the work units).
    experimental::ProgramSpec program_spec{
        .name = "hello_world_datatypes_kernel",
        .kernels = {data_reader_kernel_spec},
        .dataflow_buffers = {src0_dfb_spec},
        .work_units = {{.name = "hello_world_datatypes_kernel", .kernels = {data_reader_kernel}, .target_nodes = core}},
    };
    Program program = experimental::MakeProgramFromSpec(*mesh_device, program_spec);

    // Initialize Float data on host and upload to the DRAM tensor (non-blocking upload)
    // Keep the host tensor alive until the upload is complete.
    std::vector<float> init_data(buffer_size / sizeof(float), 1.23);
    HostTensor init_host_tensor = HostTensor::from_vector(init_data, dram_tensor_spec);
    cq.enqueue_write_tensor(init_host_tensor, dram_tensor);

    // Set runtime args, add program to mesh workload, and enqueue (non-blocking)
    experimental::ProgramRunArgs run_args{
        .kernel_run_args = {{
            .kernel = data_reader_kernel,
            .runtime_arg_values = experimental::MakeRuntimeArgsForSingleNode(
                core, {{"dram_addr", static_cast<uint32_t>(dram_tensor.address())}}),
        }},
    };
    experimental::SetProgramRunArgs(program, run_args);
    workload.add_program(device_range, std::move(program));
    distributed::EnqueueMeshWorkload(cq, workload, false);

    fmt::print("Hello, Core {{0, 0}} on Device 0, please handle the data.\n");

    // Wait for completion and close the mesh device
    distributed::Finish(cq);
    fmt::print("Thank you, Core {{0, 0}} on Device 0, for handling the data.\n");
    mesh_device->close();

    return 0;
}
