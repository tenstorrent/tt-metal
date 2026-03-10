// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <cstdint>
#include <vector>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/program.hpp>
#include <umd/device/types/arch.hpp>

#include "device_fixture.hpp"
#include "impl/context/metal_context.hpp"
#include "impl/kernels/kernel.hpp"
#include "llrt/hal.hpp"
#include "llrt/tt_cluster.hpp"

using namespace tt;
using namespace tt::tt_metal;

// Smoke test: run a single DRAM kernel that writes a compile-time constant to L1,
// then read it back via the host and verify.
TEST_F(BlackholeSingleCardFixture, DramKernelWriteToL1) {
    constexpr uint32_t kMagicValue = 0xDEADBEEF;

    auto mesh_device = devices_[0];
    auto& cq = mesh_device->mesh_command_queue();
    auto zero_coord = distributed::MeshCoordinate(0, 0);
    auto device_range = distributed::MeshCoordinateRange(zero_coord);

    // Pick the first logical DRAM worker core (bank=0, subchannel=0).
    CoreCoord logical_dram_core{0, 0};
    auto virtual_dram_core = mesh_device->virtual_core_from_logical_core(logical_dram_core, CoreType::DRAM_WORKER);

    const auto& hal = MetalContext::instance().hal();
    uint64_t l1_noc_offset = hal.get_l1_noc_offset(HalProgrammableCoreType::DRAM);
    uint32_t result_l1_addr = hal.get_dev_addr(HalProgrammableCoreType::DRAM, HalL1MemAddrType::UNRESERVED);

    distributed::MeshWorkload workload;
    Program program = CreateProgram();
    workload.add_program(device_range, std::move(program));
    auto& prog = workload.get_programs().at(device_range);

    CreateKernel(
        prog,
        "tests/tt_metal/tt_metal/test_kernels/misc/dram_write_one_uint32.cpp",
        logical_dram_core,
        DramConfig{
            .noc = NOC::NOC_0,
            .compile_args = {result_l1_addr, kMagicValue},
        });

    distributed::EnqueueMeshWorkload(cq, workload, false);
    distributed::Finish(cq);

    // Read back from DRAM core L1 (requires 64-bit NOC offset, so use cluster API directly).
    uint64_t read_addr = static_cast<uint64_t>(result_l1_addr) + l1_noc_offset;
    std::vector<uint32_t> result(1, 0);
    MetalContext::instance().get_cluster().read_core(
        result.data(), sizeof(uint32_t), tt_cxy_pair(mesh_device->id(), virtual_dram_core), read_addr);

    EXPECT_EQ(result[0], kMagicValue);
}

// Run the same kernel across multiple DRAM cores.
TEST_F(BlackholeSingleCardFixture, DramKernelOnMultipleCores) {
    constexpr uint32_t kMagicBase = 0xCAFE0000;

    auto mesh_device = devices_[0];
    auto& cq = mesh_device->mesh_command_queue();
    auto zero_coord = distributed::MeshCoordinate(0, 0);
    auto device_range = distributed::MeshCoordinateRange(zero_coord);

    const auto& hal = MetalContext::instance().hal();
    uint32_t result_l1_addr = hal.get_dev_addr(HalProgrammableCoreType::DRAM, HalL1MemAddrType::UNRESERVED);
    uint64_t l1_noc_offset = hal.get_l1_noc_offset(HalProgrammableCoreType::DRAM);

    // Use internal SoC descriptor to get DRAM worker grid: (num_banks, num_subchannels).
    const auto& soc_desc = MetalContext::instance().get_cluster().get_soc_desc(mesh_device->id());
    auto dram_worker_grid = soc_desc.get_grid_size(CoreType::DRAM);
    // Test the first row of DRAM worker cores (subchannel=0, up to 4 banks to keep it fast).
    uint32_t num_cores = std::min(static_cast<size_t>(dram_worker_grid.x), static_cast<size_t>(4));

    for (uint32_t col = 0; col < num_cores; col++) {
        CoreCoord logical_dram_core{col, 0};
        auto virtual_dram_core =
            mesh_device->virtual_core_from_logical_core(logical_dram_core, CoreType::DRAM_WORKER);
        uint32_t expected_value = kMagicBase + col;

        distributed::MeshWorkload workload;
        Program program = CreateProgram();
        workload.add_program(device_range, std::move(program));
        auto& prog = workload.get_programs().at(device_range);

        CreateKernel(
            prog,
            "tests/tt_metal/tt_metal/test_kernels/misc/dram_write_one_uint32.cpp",
            logical_dram_core,
            DramConfig{
                .noc = NOC::NOC_0,
                .compile_args = {result_l1_addr, expected_value},
            });

        distributed::EnqueueMeshWorkload(cq, workload, false);
        distributed::Finish(cq);

        // Read back from DRAM core L1 (requires 64-bit NOC offset, so use cluster API directly).
        uint64_t read_addr = static_cast<uint64_t>(result_l1_addr) + l1_noc_offset;
        std::vector<uint32_t> result(1, 0);
        MetalContext::instance().get_cluster().read_core(
            result.data(), sizeof(uint32_t), tt_cxy_pair(mesh_device->id(), virtual_dram_core), read_addr);

        EXPECT_EQ(result[0], expected_value) << "Failed for DRAM core col=" << col;
    }
}
