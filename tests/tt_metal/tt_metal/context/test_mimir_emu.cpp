// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <fmt/format.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <cstdlib>
#include <memory>
#include <set>
#include <string_view>
#include <vector>

#include "impl/context/metal_context.hpp"
#include "impl/device/device_manager.hpp"
#include "impl/dispatch/slow_dispatch.hpp"
#include "impl/kernels/kernel.hpp"
#include "llrt/hal.hpp"
#include "dev_mem_map.h"
#include <tt-metalium/allocator.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>
#include "mimir_contract.hpp"
#include <tt-umd/types/arch.hpp>
#include <tt-umd/types/core_coordinates.hpp>

namespace tt::tt_metal {
namespace {

using namespace mimir_contract;

// Same open as the Wormhole/Blackhole DRISC tests: one MeshDevice over chip 0. Host reads and writes
// use the physical device, because a MeshDevice id is a mesh id. Run with TT_METAL_SLOW_DISPATCH_MODE=1
// so the mesh command queue takes the slow-dispatch path.
std::shared_ptr<distributed::MeshDevice> open_unit_mesh() { return distributed::MeshDevice::create_unit_mesh(0); }

IDevice* physical_device(distributed::MeshDevice& mesh) {
    return &slow_dispatch::physical_device_from_unit_mesh(mesh);
}

void LaunchProgram(distributed::MeshDevice& mesh, Program& program) {
    distributed::MeshWorkload workload;
    workload.add_program(distributed::MeshCoordinateRange(mesh.shape()), std::move(program));
    distributed::EnqueueMeshWorkload(mesh.mesh_command_queue(), workload, /*blocking=*/true);
}

// A Mimir CCE's uncached alias is the 4 MiB worker window (its SRAM is 4 MiB).
static_assert(MEM_L1_UNCACHED_BASE == 4 * 1024 * 1024);

class MimirCceSramRoundTripThroughUnitMesh : public ::testing::TestWithParam<uint32_t> {};

TEST_P(MimirCceSramRoundTripThroughUnitMesh, RoundTrip) {
    skip_unless_programmable_dram_package();

    auto mesh_device = open_unit_mesh();
    IDevice* device = physical_device(*mesh_device);
    ASSERT_NE(device, nullptr);
    ASSERT_EQ(device->arch(), tt::ARCH::QUASAR);

    const uint32_t mimir_cce_index = GetParam();
    const uint32_t num_mimir_cces = num_mimir_cces_on(device);
    if (mimir_cce_index >= num_mimir_cces) {
        GTEST_SKIP() << "Configured descriptor does not expose CCE " << mimir_cce_index << ".";
    }

    const CoreCoord logical_dram_core = logical_mimir_cce_core(soc_desc_of(device), mimir_cce_index);
    const uint64_t address = mimir_cce_sram_noc_addr(kMimirCceSramTestOffset);
    const uint32_t written = 0xC0FFEE03 + mimir_cce_index;
    uint32_t read_back = 0;
    write_mimir_cce_sram(device, logical_dram_core, address, &written, sizeof(written));
    read_mimir_cce_sram(device, logical_dram_core, address, &read_back, sizeof(read_back));
    EXPECT_EQ(read_back, written);
}

TEST(MimirEmu, RuntimeFirmwareInitializesThroughPublicDeviceApi) {
    skip_unless_programmable_dram_package();

    std::shared_ptr<distributed::MeshDevice> mesh_device;
    ASSERT_NO_THROW(mesh_device = open_unit_mesh());
    IDevice* device = physical_device(*mesh_device);
    ASSERT_NE(device, nullptr);
    ASSERT_EQ(device->arch(), tt::ARCH::QUASAR);
}

class HartZeroRunsDramKernel : public ::testing::TestWithParam<uint32_t> {};

TEST_P(HartZeroRunsDramKernel, WritesMagic) {
    skip_unless_programmable_dram_package();

    auto mesh_device = open_unit_mesh();
    IDevice* device = physical_device(*mesh_device);
    ASSERT_NE(device, nullptr);
    ASSERT_EQ(device->arch(), tt::ARCH::QUASAR);

    const uint32_t mimir_cce_index = GetParam();
    const uint32_t num_mimir_cces = num_mimir_cces_on(device);
    if (mimir_cce_index >= num_mimir_cces) {
        GTEST_SKIP() << "Configured descriptor does not expose CCE " << mimir_cce_index << ".";
    }

    const CoreCoord logical_dram_core = logical_mimir_cce_core(soc_desc_of(device), mimir_cce_index);
    const uint32_t magic = 0xC0FFEE04 + mimir_cce_index;
    const auto& hal = MetalContext::instance().hal();
    const uint32_t result_dev_addr =
        MEM_L1_UNCACHED_BASE + hal.get_dev_addr(HalProgrammableCoreType::DRAM, HalL1MemAddrType::UNRESERVED);
    const uint64_t result_noc_addr = hal.get_dev_noc_addr(HalProgrammableCoreType::DRAM, HalL1MemAddrType::UNRESERVED);

    Program program = CreateProgram();
    CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/misc/dram_write_one_uint32.cpp",
        logical_dram_core,
        DramConfig{.noc = NOC::NOC_0, .compile_args = {result_dev_addr, magic}});

    LaunchProgram(*mesh_device, program);

    uint32_t result = 0;
    read_mimir_cce_sram(device, logical_dram_core, result_noc_addr, &result, sizeof(result));
    EXPECT_EQ(result, magic);
}

class AllHartsRunDramKernel : public ::testing::TestWithParam<uint32_t> {};

TEST_P(AllHartsRunDramKernel, WritesMagic) {
    skip_unless_programmable_dram_package();

    auto mesh_device = open_unit_mesh();
    IDevice* device = physical_device(*mesh_device);
    ASSERT_NE(device, nullptr);
    ASSERT_EQ(device->arch(), tt::ARCH::QUASAR);

    const uint32_t mimir_cce_index = GetParam();
    const uint32_t num_mimir_cces = num_mimir_cces_on(device);
    if (mimir_cce_index >= num_mimir_cces) {
        GTEST_SKIP() << "Configured descriptor does not expose CCE " << mimir_cce_index << ".";
    }

    const CoreCoord logical_dram_core = logical_mimir_cce_core(soc_desc_of(device), mimir_cce_index);
    const auto& hal = MetalContext::instance().hal();
    const uint32_t num_harts = hal.get_num_risc_processors(HalProgrammableCoreType::DRAM);
    ASSERT_EQ(num_harts, 8u);

    const uint32_t result_dev_base =
        MEM_L1_UNCACHED_BASE + hal.get_dev_addr(HalProgrammableCoreType::DRAM, HalL1MemAddrType::UNRESERVED);
    const uint64_t result_noc_base = hal.get_dev_noc_addr(HalProgrammableCoreType::DRAM, HalL1MemAddrType::UNRESERVED);

    Program program = CreateProgram();
    std::vector<uint32_t> expected(num_harts);
    for (uint32_t hart = 0; hart < num_harts; hart++) {
        expected[hart] = 0xC0FFEE10 + (mimir_cce_index << 8) + hart;
        CreateKernel(
            program,
            "tests/tt_metal/tt_metal/test_kernels/misc/dram_write_one_uint32.cpp",
            logical_dram_core,
            DramConfig{
                .processor = static_cast<DataMovementProcessor>(hart),
                .noc = NOC::NOC_0,
                .compile_args = {result_dev_base + hart * static_cast<uint32_t>(sizeof(uint32_t)), expected[hart]}});
    }

    LaunchProgram(*mesh_device, program);

    std::vector<uint32_t> result(num_harts, 0);
    read_mimir_cce_sram(device, logical_dram_core, result_noc_base, result.data(), result.size() * sizeof(uint32_t));
    EXPECT_EQ(result, expected);
}

}  // namespace
}  // namespace tt::tt_metal
