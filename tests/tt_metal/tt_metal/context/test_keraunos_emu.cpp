// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Keraunos HSIO CCE kernels on the MMK emulation (steps 5 and 6 of grendel_keraunos.md): a kernel on a CCE writes
// its own SRAM, every hart runs, every ethernet core runs, and one core stores into the others' SRAM.

#include <gtest/gtest.h>

#include <fmt/format.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "impl/context/metal_context.hpp"
#include "impl/device/device_manager.hpp"
#include "impl/dispatch/slow_dispatch.hpp"
#include "impl/kernels/kernel.hpp"
#include "llrt/hal.hpp"
#include "dev_mem_map.h"
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>
#include "mimir_contract.hpp"
#include <tt-umd/types/arch.hpp>
#include <tt-umd/types/core_coordinates.hpp>

namespace tt::tt_metal {
namespace {

using namespace mimir_contract;

// One MeshDevice over chip 0, with host access through the physical device. Run with TT_METAL_SLOW_DISPATCH_MODE=1.
std::shared_ptr<distributed::MeshDevice> keraunos_open_unit_mesh() {
    return distributed::MeshDevice::create_unit_mesh(0);
}

IDevice* keraunos_physical_device(distributed::MeshDevice& mesh) {
    return &slow_dispatch::physical_device_from_unit_mesh(mesh);
}

void keraunos_launch_program(distributed::MeshDevice& mesh, Program& program) {
    distributed::MeshWorkload workload;
    workload.add_program(distributed::MeshCoordinateRange(mesh.shape()), std::move(program));
    distributed::EnqueueMeshWorkload(mesh.mesh_command_queue(), workload, /*blocking=*/true);
}

void expect_ethernet_kernel_writes_local_sram(
    distributed::MeshDevice& mesh_device, IDevice* device, CoreCoord logical_eth, uint32_t magic) {
    const auto& hal = MetalContext::instance().hal();
    // The kernel reaches this through its own uncached alias, so the host passes the core-local address.
    const uint32_t result_dev_addr =
        hal.get_dev_addr(HalProgrammableCoreType::IDLE_ETH, HalL1MemAddrType::UNRESERVED);
    const uint64_t result_noc_addr =
        hal.get_dev_noc_addr(HalProgrammableCoreType::IDLE_ETH, HalL1MemAddrType::UNRESERVED);

    Program program = CreateProgram();
    CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/misc/keraunos/keraunos_cce_write_one_uint32.cpp",
        logical_eth,
        EthernetConfig{.eth_mode = Eth::IDLE, .noc = NOC::NOC_0, .compile_args = {result_dev_addr, magic}});

    keraunos_launch_program(mesh_device, program);

    uint32_t result = 0;
    read_eth_sram(device, logical_eth, result_noc_addr, &result, sizeof(result));
    EXPECT_EQ(result, magic) << "ethernet " << logical_eth.x << "-" << logical_eth.y;
}

// Descriptor NOC0 (4, y) is HSIO y CCE 0. Metal's logical ethernet coordinate is (0, channel).
CoreCoord logical_ethernet_core(const IDevice* device, size_t noc0_x, size_t noc0_y) {
    return soc_desc_of(device).get_logical_ethernet_core_from_physical({noc0_x, noc0_y});
}

// Step 5 of grendel_keraunos.md. HSIO 0 CCE 0 is descriptor core (4, 0). The kernel writes its own SRAM.
TEST(KeraunosEmu, HartZeroRunsEthernetKernelWritesLocalSram) {
    skip_unless_ethernet_package();

    auto mesh_device = keraunos_open_unit_mesh();
    IDevice* device = keraunos_physical_device(*mesh_device);
    ASSERT_NE(device, nullptr);
    ASSERT_EQ(device->arch(), tt::ARCH::QUASAR);

    const CoreCoord logical_eth = logical_ethernet_core(device, 4, 0);
    if (!device->ethernet_cores().contains(logical_eth)) {
        GTEST_SKIP() << "Configured descriptor does not list ethernet core (4, 0).";
    }

    expect_ethernet_kernel_writes_local_sram(*mesh_device, device, logical_eth, 0xCE0A0001);
}

// The DMRISC remapper is per CCE, so harts 1-7 store through the same local window as hart 0.
// Each hart writes its own word in one program. The host reads the eight words back in one transfer.
TEST(KeraunosEmu, AllHartsRunEthernetKernelWritesLocalSram) {
    skip_unless_ethernet_package();

    auto mesh_device = keraunos_open_unit_mesh();
    IDevice* device = keraunos_physical_device(*mesh_device);
    ASSERT_NE(device, nullptr);
    ASSERT_EQ(device->arch(), tt::ARCH::QUASAR);

    const CoreCoord logical_eth = logical_ethernet_core(device, 4, 0);
    if (!device->ethernet_cores().contains(logical_eth)) {
        GTEST_SKIP() << "Configured descriptor does not list ethernet core (4, 0).";
    }

    const auto& hal = MetalContext::instance().hal();
    // One CCE per core, eight harts. EthernetConfig names all eight.
    ASSERT_EQ(hal.get_num_risc_processors(HalProgrammableCoreType::IDLE_ETH), 8u);
    constexpr uint32_t num_harts = 8;

    const uint32_t result_dev_base =
        hal.get_dev_addr(HalProgrammableCoreType::IDLE_ETH, HalL1MemAddrType::UNRESERVED);
    const uint64_t result_noc_base =
        hal.get_dev_noc_addr(HalProgrammableCoreType::IDLE_ETH, HalL1MemAddrType::UNRESERVED);

    Program program = CreateProgram();
    std::vector<uint32_t> expected(num_harts);
    for (uint32_t hart = 0; hart < num_harts; hart++) {
        expected[hart] = 0xCE0B0000u + hart;
        CreateKernel(
            program,
            "tests/tt_metal/tt_metal/test_kernels/misc/keraunos/keraunos_cce_write_one_uint32.cpp",
            logical_eth,
            EthernetConfig{
                .eth_mode = Eth::IDLE,
                .noc = NOC::NOC_0,
                .processor = static_cast<DataMovementProcessor>(hart),
                .compile_args = {result_dev_base + hart * static_cast<uint32_t>(sizeof(uint32_t)), expected[hart]}});
    }

    keraunos_launch_program(*mesh_device, program);

    std::vector<uint32_t> result(num_harts, 0);
    read_eth_sram(device, logical_eth, result_noc_base, result.data(), result.size() * sizeof(uint32_t));
    EXPECT_EQ(result, expected);
}

// Step 6. The same local write on every ethernet core the descriptor lists.
TEST(KeraunosEmu, EveryEthernetCoreWritesLocalSram) {
    skip_unless_ethernet_package();

    auto mesh_device = keraunos_open_unit_mesh();
    IDevice* device = keraunos_physical_device(*mesh_device);
    ASSERT_NE(device, nullptr);
    ASSERT_EQ(device->arch(), tt::ARCH::QUASAR);

    const auto& ethernet_cores = device->ethernet_cores();
    ASSERT_FALSE(ethernet_cores.empty());
    for (const CoreCoord& logical_eth : ethernet_cores) {
        const uint32_t magic = 0xCE0A0000u | ((logical_eth.x & 0xFu) << 4) | (logical_eth.y & 0xFu);
        expect_ethernet_kernel_writes_local_sram(*mesh_device, device, logical_eth, magic);
    }
}

// Step 6. One core stores into every other core's SRAM through a host-written table.
// Each table entry is the global HSIO SRAM address, 0x1206800000 + hsio * tile_stride + eth_cce * eth_l1_size.
// The host still reads the chiplet-local window. Remap entries 6-9 deliver the global store there.
TEST(KeraunosEmu, EthernetKernelWritesOtherTileSram) {
    skip_unless_ethernet_package();

    auto mesh_device = keraunos_open_unit_mesh();
    IDevice* device = keraunos_physical_device(*mesh_device);
    ASSERT_NE(device, nullptr);
    ASSERT_EQ(device->arch(), tt::ARCH::QUASAR);

    const auto& arch = soc_desc_of(device).get_arch_descriptor();
    const auto& chiplet = arch.get_keraunos();
    if (chiplet.cores.size() < 2 || !chiplet.local_window.has_value() || !chiplet.tile_stride.has_value()) {
        GTEST_SKIP() << "Configured descriptor does not list two ethernet cores with a chiplet-local SRAM window.";
    }

    const auto& hal = MetalContext::instance().hal();
    const uint32_t sram_offset = hal.get_dev_addr(HalProgrammableCoreType::IDLE_ETH, HalL1MemAddrType::UNRESERVED);
    const uint64_t sram_noc_addr =
        hal.get_dev_noc_addr(HalProgrammableCoreType::IDLE_ETH, HalL1MemAddrType::UNRESERVED);
    const uint64_t region_size = arch.get_eth_l1_size();
    // 0x1204000000 is HSIO 0's global window; CCE SRAM begins 0x02800000 above that.
    constexpr uint64_t kHsioGlobalSramBase = 0x1206800000ULL;
    constexpr uint32_t source_index = 0;
    constexpr uint32_t magic = 0xCE0C0000;

    const auto& source = chiplet.cores[source_index];
    const CoreCoord logical_src = logical_ethernet_core(device, source.coord.x, source.coord.y);
    ASSERT_TRUE(device->ethernet_cores().contains(logical_src));

    std::vector<uint32_t> args{source_index, static_cast<uint32_t>(chiplet.cores.size()), sram_offset, magic};
    for (const auto& core : chiplet.cores) {
        const uint64_t base = kHsioGlobalSramBase + static_cast<uint64_t>(core.hsio) * *chiplet.tile_stride +
                              static_cast<uint64_t>(core.eth_cce) * region_size;
        args.push_back(static_cast<uint32_t>(base));
        args.push_back(static_cast<uint32_t>(base >> 32));
    }

    const uint32_t zero = 0;
    for (uint32_t core_index = 0; core_index < chiplet.cores.size(); ++core_index) {
        if (core_index == source_index) {
            continue;
        }
        const auto& core = chiplet.cores[core_index];
        const CoreCoord logical_dest = logical_ethernet_core(device, core.coord.x, core.coord.y);
        write_eth_sram(device, logical_dest, sram_noc_addr, &zero, sizeof(zero));
    }

    Program program = CreateProgram();
    const KernelHandle kernel = CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/misc/eth_write_tile_sram.cpp",
        logical_src,
        EthernetConfig{.eth_mode = Eth::IDLE, .noc = NOC::NOC_0});
    SetRuntimeArgs(program, kernel, logical_src, args);
    keraunos_launch_program(*mesh_device, program);

    for (uint32_t core_index = 0; core_index < chiplet.cores.size(); ++core_index) {
        if (core_index == source_index) {
            continue;
        }
        const auto& core = chiplet.cores[core_index];
        const CoreCoord logical_dest = logical_ethernet_core(device, core.coord.x, core.coord.y);
        uint32_t result = 0;
        read_eth_sram(device, logical_dest, sram_noc_addr, &result, sizeof(result));
        EXPECT_EQ(result, magic | core_index) << "ethernet " << core.coord.x << "-" << core.coord.y;
    }
}

}  // namespace
}  // namespace tt::tt_metal
