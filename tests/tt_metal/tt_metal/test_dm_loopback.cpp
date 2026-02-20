// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "common/device_fixture.hpp"

#include <tt-metalium/device.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/experimental/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>
#include "hw/inc/internal/tt-2xx/quasar/dev_mem_map.h"
#include "impl/context/metal_context.hpp"
#include "llrt/tt_cluster.hpp"

#ifndef OVERRIDE_KERNEL_PREFIX
#define OVERRIDE_KERNEL_PREFIX ""
#endif

using namespace tt;
using namespace tt::tt_metal;

// This test requires simulator environment
TEST_F(MeshDeviceSingleCardFixture, DmLoopback) {
    // Skip if simulator is not available
    char* env_var = std::getenv("TT_METAL_SIMULATOR");
    if (env_var == nullptr) {
        GTEST_SKIP() << "This test can only be run using a simulator. Set TT_METAL_SIMULATOR environment variable.";
    }
    env_var = std::getenv("TT_METAL_DPRINT_CORES");
    if (env_var == nullptr) {
        std::cerr << "WARNING: Please set the environment variable TT_METAL_DPRINT_CORES to 0,0 to see the output of "
                     "the Data Movement kernels."
                  << std::endl;
        std::cerr << "WARNING: For example, export TT_METAL_DPRINT_CORES=0,0" << std::endl;
    }

    IDevice* dev = devices_[0]->get_devices()[0];
    auto mesh_device = devices_[0];
    constexpr CoreCoord core = {0, 0};

    // These addresses have been randomly chosen
    uint32_t signal_address = 999 * 1024;
    uint32_t l1_address = 1000 * 1024;
    uint32_t dram_address = 30000 * 1024;
    std::vector<uint32_t> value = {0x12345678};

    std::vector<uint32_t> signal = {0};
    tt_metal::detail::WriteToDeviceL1(dev, core, signal_address, signal);
    tt_metal::detail::WriteToDeviceDRAMChannel(dev, 0, dram_address, value);
    MetalContext::instance().get_cluster().dram_barrier(dev->id());

    distributed::MeshCommandQueue& cq = mesh_device->mesh_command_queue();
    distributed::MeshWorkload workload;
    distributed::MeshCoordinateRange device_range = distributed::MeshCoordinateRange(mesh_device->shape());
    Program program = CreateProgram();

    std::vector<KernelHandle> dm_dram_to_l1_kernels;
    dm_dram_to_l1_kernels.reserve(4);
    for (uint32_t i = 0; i < 4; i++) {
        dm_dram_to_l1_kernels.push_back(experimental::quasar::CreateKernel(
            program,
            OVERRIDE_KERNEL_PREFIX "tests/tt_metal/tt_metal/test_kernels/dataflow/dram_to_l1.cpp",
            core,
            experimental::quasar::QuasarDataMovementConfig{.num_processors_per_cluster = 1}));
    }

    std::vector<KernelHandle> dm_l1_to_dram_kernels;
    dm_l1_to_dram_kernels.reserve(4);
    for (uint32_t i = 0; i < 4; i++) {
        dm_l1_to_dram_kernels.push_back(experimental::quasar::CreateKernel(
            program,
            OVERRIDE_KERNEL_PREFIX "tests/tt_metal/tt_metal/test_kernels/dataflow/l1_to_dram.cpp",
            core,
            experimental::quasar::QuasarDataMovementConfig{.num_processors_per_cluster = 1}));
    }

    for (uint32_t i = 0; i < 4; i++) {
        SetRuntimeArgs(
            program,
            dm_dram_to_l1_kernels[i],
            core,
            {dram_address, l1_address, MEM_L1_UNCACHED_BASE + signal_address, 4, 0, signal[0]});
        signal[0]++;
        dram_address += 1024;

        SetRuntimeArgs(
            program,
            dm_l1_to_dram_kernels[i],
            core,
            {dram_address, l1_address, MEM_L1_UNCACHED_BASE + signal_address, 4, 0, signal[0]});
        signal[0]++;
        l1_address += sizeof(uint32_t);
    }

    workload.add_program(device_range, std::move(program));
    distributed::EnqueueMeshWorkload(cq, workload, true);

    std::vector<uint32_t> outputs{0};
    tt_metal::detail::ReadFromDeviceDRAMChannel(dev, 0, dram_address, sizeof(uint32_t), outputs);

    ASSERT_EQ(outputs[0], value[0]) << "Got the value " << std::hex << outputs[0] << " instead of " << value[0];
}
