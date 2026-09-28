// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/experimental/metal2_host_api/program.hpp>

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

        const experimental::NodeCoord node{0, 0};
        const experimental::KernelSpecName DRAM_COPY_KERNEL{"loopback_dram_copy"};

        // See kernel cpp code for details on which noc calls are captured
        experimental::KernelSpec dram_copy_kernel{
            .unique_id = DRAM_COPY_KERNEL,
            .source = "tt_metal/programming_examples/profiler/test_noc_event_profiler/kernels/loopback_dram_copy.cpp",
            .num_threads = 1,
            .hw_config = experimental::DataMovementHardwareConfig{experimental::DataMovementGen1Config{
                .processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default}},
            // With no named args, vararg i is get_arg_val<uint32_t>(i), so the kernel reads its args unchanged
            .advanced_options = experimental::KernelAdvancedOptions{.num_runtime_varargs = 6},
        };
        if (mesh_device->arch() == tt::ARCH::QUASAR) {
            dram_copy_kernel.hw_config =
                experimental::DataMovementHardwareConfig{experimental::DataMovementGen2Config{}};
        }

        experimental::WorkUnitSpec wu{
            .name = "noc_event_profiler",
            .kernels = {DRAM_COPY_KERNEL},
            .target_nodes = node,
        };
        experimental::ProgramSpec spec{
            .name = "noc_event_profiler",
            .kernels = {dram_copy_kernel},
            .work_units = {wu},
        };
        Program program = experimental::MakeProgramFromSpec(*mesh_device, spec);

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

        experimental::ProgramRunArgs params;
        params.kernel_run_args.push_back(experimental::ProgramRunArgs::KernelRunArgs{
            .kernel = DRAM_COPY_KERNEL,
            .advanced_options =
                experimental::AdvancedKernelRunArgs{
                    .runtime_varargs =
                        {{node,
                          {static_cast<uint32_t>(l1_buffer->address()),
                           static_cast<uint32_t>(input_dram_buffer->address()),
                           input_bank_id,
                           static_cast<uint32_t>(output_dram_buffer->address()),
                           output_bank_id,
                           static_cast<uint32_t>(l1_buffer->size())}}}},
        });
        experimental::SetProgramRunArgs(program, params);

        workload.add_program(device_range, std::move(program));
        distributed::EnqueueMeshWorkload(cq, workload, false);
        distributed::Finish(cq);

        // It is necessary to explicitly read profile results at the end of the
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
