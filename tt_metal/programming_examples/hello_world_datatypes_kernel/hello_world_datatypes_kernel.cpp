// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/distributed.hpp>
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

    // Initialize mesh device, command queue, workload, device range, and program
    constexpr CoreCoord core = {0, 0};
    std::shared_ptr<distributed::MeshDevice> mesh_device = distributed::MeshDevice::create_unit_mesh(0);
    distributed::MeshCommandQueue& cq = mesh_device->mesh_command_queue();
    distributed::MeshWorkload workload;
    distributed::MeshCoordinateRange device_range = distributed::MeshCoordinateRange(mesh_device->shape());
    Program program = CreateProgram();

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

    // Configure and create an L1 circular buffer (to move data from DRAM to L1)
    // Set page size equal to buffer_size so one page equals one transfer unit
    constexpr uint32_t src0_cb_index = CBIndex::c_0;
    CircularBufferConfig cb_src0_config =
        CircularBufferConfig(buffer_size, {{src0_cb_index, tt::DataFormat::Float16_b}})
            .set_page_size(src0_cb_index, buffer_size);
    tt_metal::CreateCircularBuffer(program, core, cb_src0_config);

    // Configure and create the kernel
    KernelHandle data_reader_kernel_id = CreateKernel(
        program,
        OVERRIDE_KERNEL_PREFIX "hello_world_datatypes_kernel/kernels/dataflow/float_dataflow_kernel.cpp",
        core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});

    // Initialize Float data on host and upload to the DRAM tensor (non-blocking upload)
    // Keep the host tensor alive until the upload is complete.
    std::vector<float> init_data(buffer_size / sizeof(float), 1.23);
    HostTensor init_host_tensor = HostTensor::from_vector(init_data, dram_tensor_spec);
    cq.enqueue_write_tensor(init_host_tensor, dram_tensor);

    // Set runtime args, add program to mesh workload, and enqueue (non-blocking)
    SetRuntimeArgs(program, data_reader_kernel_id, core, {dram_tensor.address()});
    workload.add_program(device_range, std::move(program));
    distributed::EnqueueMeshWorkload(cq, workload, false);

    fmt::print("Hello, Core {{0, 0}} on Device 0, please handle the data.\n");

    // Wait for completion and close the mesh device
    distributed::Finish(cq);
    fmt::print("Thank you, Core {{0, 0}} on Device 0, for handling the data.\n");
    mesh_device->close();

    return 0;
}
