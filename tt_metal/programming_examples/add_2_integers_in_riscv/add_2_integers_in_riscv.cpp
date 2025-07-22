// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/distributed.hpp>

using namespace tt;
using namespace tt::tt_metal;
#ifndef OVERRIDE_KERNEL_PREFIX
#define OVERRIDE_KERNEL_PREFIX ""
#endif
int main() {
    /* Silicon accelerator setup */

    /* A MeshDevice is a software concept that allows developers to virtualize a cluster
    of connected devices as a single object, maintaining uniform memory and runtime state
    across all physical devices. A UnitMesh is a 1x1 MeshDevice that allows users to interface
    with a single physical device. */
    std::shared_ptr<distributed::MeshDevice> mesh_device = distributed::MeshDevice::create_unit_mesh(
        0, DEFAULT_L1_SMALL_SIZE, DEFAULT_TRACE_REGION_SIZE, 1, DispatchCoreType::WORKER);

    /* Setup program to execute along with its buffers and kernels to use */
    distributed::MeshCommandQueue& cq = mesh_device->mesh_command_queue();
    distributed::MeshWorkload workload;
    distributed::MeshCoordinateRange device_range = distributed::MeshCoordinateRange(mesh_device->shape());
    Program program = CreateProgram();

    const auto device_coord = distributed::MeshCoordinate(0, 0);
    constexpr CoreCoord core = {0, 0};

    constexpr uint32_t single_tile_size = 2 * 1024;
    distributed::DeviceLocalBufferConfig dram_config{
        .page_size = single_tile_size,
        .buffer_type = tt_metal::BufferType::DRAM,
        .bottom_up = false
    };
    const distributed::ReplicatedBufferConfig buffer_config {
        .size = single_tile_size
    };

    std::shared_ptr<tt::tt_metal::distributed::MeshBuffer> src0_dram_buffer = distributed::MeshBuffer::create(buffer_config, dram_config, mesh_device.get());
    std::shared_ptr<tt::tt_metal::distributed::MeshBuffer> src1_dram_buffer = distributed::MeshBuffer::create(buffer_config, dram_config, mesh_device.get());
    std::shared_ptr<tt::tt_metal::distributed::MeshBuffer> dst_dram_buffer = distributed::MeshBuffer::create(buffer_config, dram_config, mesh_device.get());

    // Since all interleaved buffers have size == page_size, they are entirely contained in the first DRAM bank
    uint32_t src0_bank_id = 0;
    uint32_t src1_bank_id = 0;
    uint32_t dst_bank_id = 0;

    /* Create source data and write to DRAM */
    std::vector<uint32_t> src0_vec(1, 14);
    std::vector<uint32_t> src1_vec(1, 7);

    // We're writing to a shard allocated on Device Coordinate 0, 0, since this is a 1x1
    //  When the MeshDevice is 2 dimensional, this API can be used to target specific physical devices
    distributed::WriteShard(cq, src0_dram_buffer, src0_vec, device_coord);
    distributed::WriteShard(cq, src1_dram_buffer, src1_vec, device_coord);

    /* Use L1 circular buffers to set input buffers */
    constexpr uint32_t src0_cb_index = CBIndex::c_0;
    CircularBufferConfig cb_src0_config =
        CircularBufferConfig(single_tile_size, {{src0_cb_index, tt::DataFormat::Float16_b}})
            .set_page_size(src0_cb_index, single_tile_size);
    tt_metal::CreateCircularBuffer(program, core, cb_src0_config);

    constexpr uint32_t src1_cb_index = CBIndex::c_1;
    CircularBufferConfig cb_src1_config =
        CircularBufferConfig(single_tile_size, {{src1_cb_index, tt::DataFormat::Float16_b}})
            .set_page_size(src1_cb_index, single_tile_size);
    tt_metal::CreateCircularBuffer(program, core, cb_src1_config);

    /* Specify data movement kernel for reading/writing data to/from DRAM */
    KernelHandle binary_reader_kernel_id = CreateKernel(
        program,
        OVERRIDE_KERNEL_PREFIX "add_2_integers_in_riscv/kernels/reader_writer_add_in_riscv.cpp",
        core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});

    /* Configure program and runtime kernel arguments, then execute */
    SetRuntimeArgs(
        program,
        binary_reader_kernel_id,
        core,
        {src0_dram_buffer->address(),
         src1_dram_buffer->address(),
         dst_dram_buffer->address(),
         src0_bank_id,
         src1_bank_id,
         dst_bank_id});

    distributed::AddProgramToMeshWorkload(workload, std::move(program), device_range);
    distributed::EnqueueMeshWorkload(cq, workload, false);
    Finish(cq);

    /* Read in result into a host vector */
    std::vector<uint32_t> result_vec;

    // We're reading from a shard allocated on Device Coordinate 0, 0, since this is a 1x1
    //  When the MeshDevice is 2 dimensional, this API can be used to target specific physical devices
    distributed::ReadShard(cq, result_vec, dst_dram_buffer, device_coord);
    printf("Result = %d : Expected = 21\n", result_vec[0]);
    mesh_device.reset();
}
