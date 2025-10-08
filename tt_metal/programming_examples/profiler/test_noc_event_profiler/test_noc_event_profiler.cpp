// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/tt_metal_profiler.hpp>
#include <tt-metalium/distributed.hpp>

using namespace tt::tt_metal;

/*
 * This test serves as a simple, stable tt_metal executable that issues both
 * reads and writes from Tensix to the NoC. It is used to do sanity checking of
 * the Device Profiler's NoC event capture feature during CI in
 * test_device_profiler.py.
 */

int main() {
    if (getenv("TT_METAL_SLOW_DISPATCH_MODE") != nullptr) {
        TT_THROW("Test not supported w/ slow dispatch, exiting");
    }

    bool pass = true;

    try {
        constexpr int device_id = 0;
        std::shared_ptr<distributed::MeshDevice> mesh_device = distributed::MeshDevice::create_unit_mesh(device_id);
        distributed::MeshCommandQueue& cq = mesh_device->mesh_command_queue();
        distributed::MeshWorkload workload;
        distributed::MeshCoordinateRange device_range = distributed::MeshCoordinateRange(mesh_device->shape());
        Program program = CreateProgram();

        constexpr CoreCoord core = {0, 0};

        // See kernel cpp code for details on which noc calls are captured
        KernelHandle dram_copy_kernel_id = CreateKernel(
            program,
            "tt_metal/programming_examples/profiler/test_noc_event_profiler/kernels/loopback_dram_copy.cpp",
            core,
            DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});

        // boilerplate setup for reading and writing multiple tiles from DRAM
        constexpr uint32_t single_tile_size = 2 * (32 * 32);
        constexpr uint32_t num_tiles = 5;
        constexpr uint32_t dram_buffer_size = single_tile_size * num_tiles;

        distributed::DeviceLocalBufferConfig dram_config{
            .page_size = dram_buffer_size, .buffer_type = tt::tt_metal::BufferType::DRAM};
        distributed::DeviceLocalBufferConfig l1_config{
            .page_size = dram_buffer_size, .buffer_type = tt::tt_metal::BufferType::L1};
        distributed::ReplicatedBufferConfig buffer_config{.size = dram_buffer_size};

        auto l1_buffer = distributed::MeshBuffer::create(buffer_config, l1_config, mesh_device.get());

        auto input_dram_buffer = distributed::MeshBuffer::create(buffer_config, dram_config, mesh_device.get());

        auto output_dram_buffer = distributed::MeshBuffer::create(buffer_config, dram_config, mesh_device.get());

        // Since all interleaved buffers have size == page_size, they are entirely contained in the first DRAM bank
        const uint32_t input_bank_id = 0;
        const uint32_t output_bank_id = 0;

        const std::vector<uint32_t> runtime_args = {
            l1_buffer->address(),
            input_dram_buffer->address(),
            input_bank_id,
            output_dram_buffer->address(),
            output_bank_id,
            l1_buffer->size()};
        SetRuntimeArgs(program, dram_copy_kernel_id, core, runtime_args);

        workload.add_program(device_range, std::move(program));
        distributed::EnqueueMeshWorkload(cq, workload, false);
        distributed::Finish(cq);

        // It is necessary to explictly read profile results at the end of the
        // program to get noc traces for standalone tt_metal programs.  For
        // ttnn, this is called _automatically_
        ReadMeshDeviceProfilerResults(*mesh_device);

        pass &= mesh_device->close();

    } catch (const std::exception& e) {
        fmt::print(stderr, "Test failed with exception!\n");
        fmt::print(stderr, "{}\n", e.what());

        throw;
    }

    if (pass) {
        fmt::print("Test Passed\n");
    } else {
        TT_THROW("Test Failed");
    }

    return 0;
}
