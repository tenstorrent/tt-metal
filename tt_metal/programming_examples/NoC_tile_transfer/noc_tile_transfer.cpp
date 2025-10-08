// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/distributed.hpp>

#include <cstdint>
#include <vector>

#ifndef OVERRIDE_KERNEL_PREFIX
#define OVERRIDE_KERNEL_PREFIX ""
#endif

using namespace tt;
using namespace tt::tt_metal;

int main() {
    // Create a 1x1 mesh device (Mesh API). For multi-device setups, create a larger mesh shape.
    std::shared_ptr<distributed::MeshDevice> mesh_device = distributed::MeshDevice::create_unit_mesh(0);

    // Mesh command queue and program setup
    distributed::MeshCommandQueue& cq = mesh_device->mesh_command_queue();
    distributed::MeshWorkload workload;
    distributed::MeshCoordinateRange device_range = distributed::MeshCoordinateRange(mesh_device->shape());
    Program program = CreateProgram();

    // Core range setup
    constexpr CoreCoord core0 = {0, 0};
    constexpr CoreCoord core1 = {0, 1};
    const auto core0_physical_coord = mesh_device->worker_core_from_logical_core(core0);
    const auto core1_physical_coord = mesh_device->worker_core_from_logical_core(core1);

    CoreRange sem_core_range = CoreRange(core0, core1);

    // Check if the environment variable for kernels print is set
    char* env_var = std::getenv("TT_METAL_DPRINT_CORES");
    if (env_var == nullptr) {
        fmt::print(
            stderr,
            "WARNING: Please set the environment variable TT_METAL_DPRINT_CORES to (0,0),(0,1) to see the output of "
            "the Data Movement kernels. Command: export TT_METAL_DPRINT_CORES=(0,0),(0,1)\n");
    }

    // Input data preparation
    constexpr uint32_t single_tile_size = sizeof(uint16_t) * tt::constants::TILE_HW;
    distributed::DeviceLocalBufferConfig dram_config{
        .page_size = single_tile_size, .buffer_type = tt_metal::BufferType::DRAM};
    distributed::ReplicatedBufferConfig buffer_config{.size = single_tile_size};

    auto src_dram_buffer =
        distributed::MeshBuffer::create(buffer_config, dram_config, mesh_device.get());  // replicated per device
    auto dst_dram_buffer = distributed::MeshBuffer::create(buffer_config, dram_config, mesh_device.get());

    // Core synchronization semaphore setup
    const uint32_t sem_id = CreateSemaphore(program, sem_core_range, 0);

    // Source data preparation and DRAM transfer
    const uint16_t input_data = 14;  // Example input data
    std::vector<uint16_t> src_vec(1, input_data);
    distributed::EnqueueWriteMeshBuffer(cq, src_dram_buffer, src_vec, false);

    // L1 circular buffer setup
    constexpr uint32_t src0_cb_index = CBIndex::c_0;
    CircularBufferConfig cb_src0_config =
        CircularBufferConfig(single_tile_size, {{src0_cb_index, tt::DataFormat::UInt16}})
            .set_page_size(src0_cb_index, single_tile_size);
    tt_metal::CreateCircularBuffer(program, sem_core_range, cb_src0_config);

    constexpr uint32_t src1_cb_index = CBIndex::c_1;
    CircularBufferConfig cb_src1_config =
        CircularBufferConfig(single_tile_size, {{src1_cb_index, tt::DataFormat::UInt16}})
            .set_page_size(src1_cb_index, single_tile_size);
    tt_metal::CreateCircularBuffer(program, sem_core_range, cb_src1_config);

    // Kernels setup
    // Core 0 kernels
    std::vector<uint32_t> reader_compile_time_args = {src0_cb_index};
    TensorAccessorArgs(*src_dram_buffer).append_to(reader_compile_time_args);
    KernelHandle core0_reader_kernel_id = CreateKernel(
        program,
        OVERRIDE_KERNEL_PREFIX "NoC_tile_transfer/kernels/dataflow/reader0.cpp",
        core0,
        tt::tt_metal::ReaderDataMovementConfig{reader_compile_time_args});
    KernelHandle core0_writer_kernel_id = CreateKernel(
        program,
        OVERRIDE_KERNEL_PREFIX "NoC_tile_transfer/kernels/dataflow/writer0.cpp",
        core0,
        tt::tt_metal::WriterDataMovementConfig{{src0_cb_index, src1_cb_index}});

    // Core 1 kernels
    KernelHandle core1_reader_kernel_id = CreateKernel(
        program,
        OVERRIDE_KERNEL_PREFIX "NoC_tile_transfer/kernels/dataflow/reader1.cpp",
        core1,
        tt::tt_metal::ReaderDataMovementConfig{{src0_cb_index, src1_cb_index}});
    std::vector<uint32_t> writer_compile_time_args = {src1_cb_index};
    TensorAccessorArgs(*dst_dram_buffer).append_to(writer_compile_time_args);
    KernelHandle core1_writer_kernel_id = CreateKernel(
        program,
        OVERRIDE_KERNEL_PREFIX "NoC_tile_transfer/kernels/dataflow/writer1.cpp",
        core1,
        tt::tt_metal::WriterDataMovementConfig{writer_compile_time_args});

    // Runtime args setup
    SetRuntimeArgs(program, core0_reader_kernel_id, core0, {src_dram_buffer->address()});
    SetRuntimeArgs(program, core0_writer_kernel_id, core0, {core1_physical_coord.x, core1_physical_coord.y, sem_id});
    SetRuntimeArgs(program, core1_reader_kernel_id, core1, {core0_physical_coord.x, core0_physical_coord.y, sem_id});
    SetRuntimeArgs(program, core1_writer_kernel_id, core1, {dst_dram_buffer->address()});

    // Program enqueue (non-blocking). Wait for completion before reading back.
    workload.add_program(device_range, std::move(program));
    distributed::EnqueueMeshWorkload(cq, workload, false);
    distributed::Finish(cq);

    // Data transfer back to host machine
    std::vector<uint16_t> result_vec;
    distributed::EnqueueReadMeshBuffer(cq, result_vec, dst_dram_buffer, true);

    fmt::print("Result = {} : Expected = {}\n", result_vec[0], input_data);

    mesh_device->close();
}
