// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <memory>
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
    // A Program contains kernels that perform computations or data movement.
    Program program = CreateProgram();
    // We will only be using one Tensix core for this particular example. As Tenstorrent processors are a 2D grid of
    // cores we can specify the core coordinates as (0, 0).
    constexpr CoreCoord core = {0, 0};

    // Adding 2 integers in RISC-V thus a buffer size of 4 bytes.
    constexpr uint32_t buffer_size = sizeof(uint32_t);

    // There are many modes of buffer allocation, here we use interleaved buffers. Interleaved buffers are the most
    // flexible and generally recommended buffer type for most applications. As the Tensix core does not have direct
    // access to DRAM, an extra buffer on L1 (SRAM) is required to read/write data from/to DRAM.
    // page_size is the size of each page in the buffer. In most applications this will be set to the size of a tile.
    // But for this example we set it to the size of a single integer as that is what we are adding.
    // Each row of a row-major tensor is one page, so a row of one integer gives a page size of buffer_size.
    constexpr uint32_t elements_per_page = buffer_size / sizeof(uint32_t);
    TensorSpec dram_tensor_spec(
        Shape({1, elements_per_page}),
        TensorLayout(
            DataType::UINT32,
            PageConfig(Layout::ROW_MAJOR),
            MemoryConfig{TensorMemoryLayout::INTERLEAVED, BufferType::DRAM}));
    TensorSpec l1_tensor_spec(
        Shape({1, elements_per_page}),
        TensorLayout(
            DataType::UINT32,
            PageConfig(Layout::ROW_MAJOR),
            MemoryConfig{TensorMemoryLayout::INTERLEAVED, BufferType::L1}));

    // Create the DRAM and SRAM tensors:
    // MeshTensor objects are allocated per device according to the TensorSpec (location, page size, size).
    // Here we create three DRAM tensors for inputs/outputs and three L1 tensors used as on-core scratch space for
    // the Data Movement kernels.
    MeshTensor src0_dram_tensor = MeshTensor::allocate_on_device(*mesh_device, dram_tensor_spec);
    MeshTensor src1_dram_tensor = MeshTensor::allocate_on_device(*mesh_device, dram_tensor_spec);
    MeshTensor dst_dram_tensor = MeshTensor::allocate_on_device(*mesh_device, dram_tensor_spec);
    MeshTensor src0_l1_tensor = MeshTensor::allocate_on_device(*mesh_device, l1_tensor_spec);
    MeshTensor src1_l1_tensor = MeshTensor::allocate_on_device(*mesh_device, l1_tensor_spec);
    MeshTensor dst_l1_tensor = MeshTensor::allocate_on_device(*mesh_device, l1_tensor_spec);

    // Create source data and write to DRAM
    std::vector<uint32_t> src0_vec = {14};
    std::vector<uint32_t> src1_vec = {7};

    // Enqueue write operations to copy data from host tensors to DRAM tensors. The write operation does not block
    // until the data is written to the device. This allows the program to continue executing while the data is being
    // written, so the host tensors must stay alive until the write is done. This is recommended for most writes to
    // device in applications to improve performance.
    HostTensor src0_host_tensor = HostTensor::from_vector(src0_vec, dram_tensor_spec);
    HostTensor src1_host_tensor = HostTensor::from_vector(src1_vec, dram_tensor_spec);
    cq.enqueue_write_tensor(src0_host_tensor, src0_dram_tensor);
    cq.enqueue_write_tensor(src1_host_tensor, src1_dram_tensor);

    // Create the kernel (code that runs on the Tensix core) that will perform the addition of the 2 integers.
    // The Data Movement cores are the only cores that can read/write data from/to DRAM. Thus we use them for
    // demonstration purposes here. In practice, you would perform addition using the compute kernel which have access
    // to much more powerful vector and matrix engines. The Data Movement cores are used for data movement tasks such as
    // reading/writing data from/to DRAM. But they are still fully capable of performing simple arithmetic operations
    // like addition. Just slower.
    KernelHandle kernel_id = CreateKernel(
        program,
        OVERRIDE_KERNEL_PREFIX "add_2_integers_in_riscv/kernels/reader_writer_add_in_riscv.cpp",
        core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});

    // Set the arguments for the kernel. The arguments are set in the order they are defined in the kernel source code.
    SetRuntimeArgs(
        program,
        kernel_id,
        core,
        {
            static_cast<uint32_t>(src0_dram_tensor.address()),
            static_cast<uint32_t>(src1_dram_tensor.address()),
            static_cast<uint32_t>(dst_dram_tensor.address()),
            static_cast<uint32_t>(src0_l1_tensor.address()),
            static_cast<uint32_t>(src1_l1_tensor.address()),
            static_cast<uint32_t>(dst_l1_tensor.address()),
        });

    // Add the program to the workload and enqueue it for execution on the MeshDevice.
    // Setting blocking=false returns immediately; commands on the queue execute in FIFO order.
    workload.add_program(device_range, std::move(program));
    distributed::EnqueueMeshWorkload(cq, workload, /*blocking=*/false);

    // Read the destination MeshTensor back to host. The last argument controls blocking.
    // This time we set blocking=true since we must have the data before comparing.
    // NOTE: Everything on the command queue executes in order; a read will not run before the prior kernel finishes.
    std::vector<uint32_t> result_vec = cq.enqueue_read_tensor(dst_dram_tensor, /*blocking=*/true).to_vector<uint32_t>();
    if (result_vec.size() != 1) {
        std::cout << "Error: Expected result vector size of 1, got " << result_vec.size() << std::endl;
        mesh_device->close();
        return -1;
    }
    if (result_vec[0] != 21) {
        std::cout << "Error: Expected result of 21, got " << result_vec[0] << std::endl;
        mesh_device->close();
        return -1;
    }

    std::cout << "Success: Result is " << result_vec[0] << std::endl;
    mesh_device->close();
}
