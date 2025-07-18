// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/distributed.hpp>

using namespace tt;
using namespace tt::tt_metal;
#ifndef OVERRIDE_KERNEL_PREFIX
#define OVERRIDE_KERNEL_PREFIX ""
#endif
int main() {
    /* Silicon accelerator setup */

    //A MeshDevice is a software concept that allows developers to virtualize a cluster of connected devices as a single object,
    // maintaining uniform memory and runtime state across all physical devices.
    //A UnitMesh is a 1x1 MeshDevice that allows users to interface with a single physical device.
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

    /* Use L1 circular buffers to set input and output buffers that the compute engine will use */
    constexpr uint32_t src0_cb_index = CBIndex::c_0;
    constexpr uint32_t num_input_tiles = 1;
    CircularBufferConfig cb_src0_config =
        CircularBufferConfig(num_input_tiles * single_tile_size, {{src0_cb_index, tt::DataFormat::Float16_b}})
            .set_page_size(src0_cb_index, single_tile_size);
    tt_metal::CreateCircularBuffer(program, core, cb_src0_config);

    constexpr uint32_t src1_cb_index = CBIndex::c_1;
    CircularBufferConfig cb_src1_config =
        CircularBufferConfig(num_input_tiles * single_tile_size, {{src1_cb_index, tt::DataFormat::Float16_b}})
            .set_page_size(src1_cb_index, single_tile_size);
    tt_metal::CreateCircularBuffer(program, core, cb_src1_config);

    constexpr uint32_t output_cb_index = CBIndex::c_16;
    constexpr uint32_t num_output_tiles = 1;
    CircularBufferConfig cb_output_config =
        CircularBufferConfig(num_output_tiles * single_tile_size, {{output_cb_index, tt::DataFormat::Float16_b}})
            .set_page_size(output_cb_index, single_tile_size);
    tt_metal::CreateCircularBuffer(program, core, cb_output_config);

    /* Specify data movement kernels for reading/writing data to/from DRAM */
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

    /* Set the parameters that the compute kernel will use */
    std::vector<uint32_t> compute_kernel_args = {};

    /* Use the add_tiles operation in the compute kernel */
    KernelHandle eltwise_binary_kernel_id = CreateKernel(
        program,
        OVERRIDE_KERNEL_PREFIX "add_2_integers_in_compute/kernels/compute/add_2_tiles.cpp",
        core,
        ComputeConfig{
            .math_fidelity = MathFidelity::HiFi4,
            .fp32_dest_acc_en = false,
            .math_approx_mode = false,
            .compile_args = compute_kernel_args,
        });

    /* Configure program and runtime kernel arguments, then add to workload */
    SetRuntimeArgs(
        program,
        binary_reader_kernel_id,
        core,
        {src0_dram_buffer->address(), src1_dram_buffer->address(), src0_bank_id, src1_bank_id});
    SetRuntimeArgs(program, eltwise_binary_kernel_id, core, {});
    SetRuntimeArgs(program, unary_writer_kernel_id, core, {dst_dram_buffer->address(), dst_bank_id});

    distributed::AddProgramToMeshWorkload(workload, std::move(program), device_range);
    /* Create source data and write to DRAM */
    std::vector<uint32_t> src0_vec;
    std::vector<uint32_t> src1_vec;
    src0_vec = create_constant_vector_of_bfloat16(single_tile_size, 14.0f);
    src1_vec = create_constant_vector_of_bfloat16(single_tile_size, 8.0f);

    // We're writing to a shard allocated on Device Coordinate 0, 0, since this is a 1x1
    //  When the MeshDevice is 2 dimensional, this API can be used to target specific physical devices
    distributed::WriteShard(cq, src0_dram_buffer, src0_vec, device_coord);
    distributed::WriteShard(cq, src1_dram_buffer, src1_vec, device_coord);

    /* Execute the workload */
    distributed::EnqueueMeshWorkload(cq, workload, false);
    Finish(cq);

    /* Read in result into a host vector */
    std::vector<uint32_t> result_vec;

    // We're reading from a shard allocated on Device Coordinate 0, 0, since this is a 1x1
    //  When the MeshDevice is 2 dimensional, this API can be used to target specific physical devices
    distributed::ReadShard(cq, result_vec, dst_dram_buffer, device_coord);

    printf("Result = %d\n", result_vec[0]);  // 22 = 1102070192
    printf(
        "Expected = %d\n",
        pack_two_bfloat16_into_uint32(std::pair<bfloat16, bfloat16>(bfloat16(22.0f), bfloat16(22.0f))));
    mesh_device.reset();
}
