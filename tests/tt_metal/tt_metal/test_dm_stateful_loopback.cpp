// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "common/device_fixture.hpp"

#include <tt-metalium/device.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/experimental/metal2_host_api/program.hpp>
#include <tt-metalium/tt_metal.hpp>
#include "impl/context/metal_context.hpp"
#include "llrt/tt_cluster.hpp"

#ifndef OVERRIDE_KERNEL_PREFIX
#define OVERRIDE_KERNEL_PREFIX ""
#endif

using namespace tt;
using namespace tt::tt_metal;

// Device coverage for the stateful NoC APIs (set_state / with_state pairs):
// one read state reused across chunked DRAM->L1 reads (one-packet flavor) and
// one write state reused across chunked L1->DRAM writes (any-len flavor).
// This test requires a simulator environment.
TEST_F(QuasarMeshDeviceSingleCardFixture, DmStatefulLoopback) {
    char* env_var = std::getenv("TT_METAL_SIMULATOR");
    if (env_var == nullptr) {
        GTEST_SKIP() << "This test can only be run using a simulator. Set TT_METAL_SIMULATOR environment variable.";
    }

    IDevice* dev = devices_[0]->get_devices()[0];
    auto mesh_device = devices_[0];
    const experimental::NodeCoord node{0, 0};

    constexpr uint32_t chunk_bytes = 64;
    constexpr uint32_t num_chunks = 4;
    constexpr uint32_t total_bytes = chunk_bytes * num_chunks;

    uint32_t l1_address = MetalContext::instance().hal().get_dev_addr(
        HalProgrammableCoreType::TENSIX, HalL1MemAddrType::DEFAULT_UNRESERVED);
    uint32_t dram_in_address = MetalContext::instance().hal().get_dev_addr(HalDramMemAddrType::UNRESERVED);
    uint32_t dram_out_address = dram_in_address + 4096;

    std::vector<uint32_t> inputs(total_bytes / sizeof(uint32_t));
    for (uint32_t i = 0; i < inputs.size(); i++) {
        inputs[i] = 0xA0000000u | i;
    }
    // Poison the output region so a silently-dropped write cannot pass.
    std::vector<uint32_t> poison(total_bytes / sizeof(uint32_t), 0xDEADBEEF);
    tt_metal::detail::WriteToDeviceDRAMChannel(dev, 0, dram_in_address, inputs);
    tt_metal::detail::WriteToDeviceDRAMChannel(dev, 0, dram_out_address, poison);
    MetalContext::instance().get_cluster().dram_barrier(dev->id());

    distributed::MeshCommandQueue& cq = mesh_device->mesh_command_queue();
    distributed::MeshWorkload workload;
    distributed::MeshCoordinateRange device_range = distributed::MeshCoordinateRange(mesh_device->shape());

    const experimental::KernelSpecName STATEFUL{"stateful_dram_loopback"};

    experimental::ProgramSpec spec{
        .name = "dm_stateful_loopback",
        .kernels = {experimental::KernelSpec{
            .unique_id = STATEFUL,
            .source = OVERRIDE_KERNEL_PREFIX "tests/tt_metal/tt_metal/test_kernels/dataflow/stateful_dram_loopback.cpp",
            .num_threads = 1,
            .runtime_arg_schema =
                {
                    .runtime_arg_names = {"dram_in_addr", "dram_out_addr", "l1_addr", "dram_bank_id"},
                },
            .hw_config = experimental::DataMovementGen2Config{},
        }},
        .work_units = {experimental::WorkUnitSpec{
            .name = "main",
            .kernels = {STATEFUL},
            .target_nodes = node,
        }},
    };
    Program program = experimental::MakeProgramFromSpec(*mesh_device, spec);

    experimental::ProgramRunArgs params;
    params.kernel_run_args.push_back(experimental::ProgramRunArgs::KernelRunArgs{
        .kernel = STATEFUL,
        .runtime_arg_values = experimental::MakeRuntimeArgsForSingleNode(
            node,
            {{"dram_in_addr", dram_in_address},
             {"dram_out_addr", dram_out_address},
             {"l1_addr", l1_address},
             {"dram_bank_id", 0u}})});
    experimental::SetProgramRunArgs(program, params);

    workload.add_program(device_range, std::move(program));
    distributed::EnqueueMeshWorkload(cq, workload, true);

    std::vector<uint32_t> outputs(total_bytes / sizeof(uint32_t), 0);
    tt_metal::detail::ReadFromDeviceDRAMChannel(dev, 0, dram_out_address, total_bytes, outputs);

    for (uint32_t i = 0; i < inputs.size(); i++) {
        ASSERT_EQ(outputs[i], inputs[i]) << "word " << i << ": got 0x" << std::hex << outputs[i] << " expected 0x"
                                         << inputs[i];
    }
}
