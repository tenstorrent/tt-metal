// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Semaphores on Keraunos Ethernet cores (CCE), on the two-chip test descriptors in this directory.
// Run with TT_METAL_SLOW_DISPATCH_MODE=1. On Mock these tests check what the host computes and skip
// before launch: Quasar Mock builds no kernel binaries, and UMD MockChip reads and writes are no-ops.

#include <gtest/gtest.h>

#include <fmt/format.h>
#include <fmt/ranges.h>

#include <algorithm>
#include <chrono>
#include <iterator>
#include <cstdint>
#include <memory>
#include <set>
#include <string>
#include <thread>
#include <vector>

#include <enchantum/enchantum.hpp>
#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/allocator.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>

#include "impl/context/metal_context.hpp"
#include "impl/dispatch/slow_dispatch.hpp"
#include "impl/kernels/kernel.hpp"
#include "impl/program/program_impl.hpp"
#include "impl/program/slow_dispatch.hpp"
#include "llrt/hal.hpp"

namespace tt::tt_metal {
namespace {

// HSIO 0's global window is 0x1204000000 and CCE SRAM begins 0x02800000 above it
// (KERAUNOS-HW-FINDINGS Q1, E100 Table 14). Same constant as test_mimir_emu.cpp.
constexpr uint64_t kHsioGlobalSramBase = 0x1206800000ULL;

constexpr uint32_t kCounterSem = 0;
constexpr uint32_t kStageSem = 1;
constexpr uint32_t kProbeSem = 2;
constexpr uint32_t kObservedSem = 3;
constexpr uint32_t kStageDone = 0x5E00000F;

bool is_mock() {
    return MetalContext::instance().get_cluster().get_target_device_type() == tt::TargetDevice::Mock;
}

// One Keraunos Ethernet core of a chip: its logical coordinate, HAL slot, and HSIO tile and CCE.
struct EthCore {
    CoreCoord logical;
    HalProgrammableCoreType type;
    uint32_t hsio;
    uint32_t cce;
};

CoreType core_type_of(HalProgrammableCoreType type) {
    return type == HalProgrammableCoreType::ACTIVE_ETH ? CoreType::ACTIVE_ETH : CoreType::IDLE_ETH;
}

Eth eth_mode_of(HalProgrammableCoreType type) {
    return type == HalProgrammableCoreType::ACTIVE_ETH ? Eth::SENDER : Eth::IDLE;
}

// Every Keraunos core the chip's descriptor lists, in eth-list order. Linked cores are ACTIVE_ETH.
std::vector<EthCore> keraunos_eth_cores(IDevice* device) {
    const auto& soc_desc = MetalContext::instance().get_cluster().get_soc_desc(device->id());
    const auto& chiplet = soc_desc.get_arch_descriptor().get_keraunos();
    const auto active = device->get_active_ethernet_cores();
    std::vector<EthCore> cores;
    for (const auto& core : chiplet.cores) {
        const CoreCoord logical = soc_desc.get_logical_ethernet_core_from_physical({core.coord.x, core.coord.y});
        const auto type =
            active.contains(logical) ? HalProgrammableCoreType::ACTIVE_ETH : HalProgrammableCoreType::IDLE_ETH;
        cores.push_back({logical, type, core.hsio, core.cce});
    }
    return cores;
}

uint32_t semaphore_address(IDevice* device, Program& program, const EthCore& core, uint32_t id) {
    const uint32_t align = MetalContext::instance().hal().get_alignment(HalMemType::L1);
    return program.impl().get_sem_base_addr(device, core.logical, core_type_of(core.type)) + id * align;
}

void expect_in_kernel_config(const EthCore& core, uint32_t address) {
    const auto& hal = MetalContext::instance().hal();
    const uint64_t base = hal.get_dev_addr(core.type, HalL1MemAddrType::KERNEL_CONFIG);
    const uint64_t size = hal.get_dev_size(core.type, HalL1MemAddrType::KERNEL_CONFIG);
    EXPECT_GE(address, base) << fmt::format("core {} {}", core.logical.str(), enchantum::to_string(core.type));
    EXPECT_LE(address + sizeof(uint32_t), base + size)
        << fmt::format("core {} {}", core.logical.str(), enchantum::to_string(core.type));
    log_info(
        tt::LogTest,
        "Keraunos semaphore: core {} {} hsio {} address {:#x} in KERNEL_CONFIG [{:#x}, {:#x})",
        core.logical.str(),
        enchantum::to_string(core.type),
        core.hsio,
        address,
        base,
        base + size);
}

uint32_t read_word(distributed::MeshDevice& mesh, const EthCore& core, uint32_t address) {
    std::vector<uint32_t> value;
    slow_dispatch::ReadFromL1(mesh, core.logical, address, sizeof(uint32_t), value, CoreType::ETH);
    TT_FATAL(value.size() == 1, "Read of {:#x} on {} returned {} words", address, core.logical.str(), value.size());
    return value[0];
}

// Each chip claims its own Metal DRAM cores, none shared, and opens with DRAM banks (the V3 per-chip SoC
// descriptors in variants/).
TEST(KeraunosSocDesc, MetalDramCoresPerChip) {
    auto& cluster = MetalContext::instance().get_cluster();
    std::vector<std::set<CoreCoord>> per_chip;
    for (ChipId chip : cluster.all_chip_ids()) {
        const auto cores = cluster.get_soc_desc(chip).get_metal_dram_cores(tt::CoordSystem::NOC0);
        per_chip.emplace_back(cores.begin(), cores.end());
        std::vector<std::string> names;
        for (const auto& c : per_chip.back()) {
            names.push_back(c.str());
        }
        log_info(tt::LogTest, "Chip {} Metal DRAM cores (NOC0): [{}]", chip, fmt::join(names, ", "));
        EXPECT_FALSE(per_chip.back().empty()) << "chip " << chip;
    }
    for (size_t a = 0; a < per_chip.size(); a++) {
        for (size_t b = a + 1; b < per_chip.size(); b++) {
            std::vector<CoreCoord> shared;
            std::set_intersection(
                per_chip[a].begin(), per_chip[a].end(), per_chip[b].begin(), per_chip[b].end(),
                std::back_inserter(shared));
            EXPECT_TRUE(shared.empty()) << "chips " << a << " and " << b << " share " << shared.size()
                                        << " Metal DRAM cores";
        }
    }
    for (ChipId chip : cluster.all_chip_ids()) {
        auto mesh = distributed::MeshDevice::create_unit_mesh(chip);
        EXPECT_GT(mesh->allocator()->get_num_banks(BufferType::DRAM), 0u) << "chip " << chip;
    }
}

// CreateSemaphore on every Keraunos core of chip 0, one kernel per core so each HAL slot
// (ACTIVE_ETH, IDLE_ETH) gets a program config. The address must lie in that slot's KERNEL_CONFIG.
TEST(KeraunosEthSemaphore, CreateSemaphoreOnIdleAndActiveEth) {
    auto mesh = distributed::MeshDevice::create_unit_mesh(0);
    IDevice* device = &slow_dispatch::physical_device_from_unit_mesh(*mesh);
    ASSERT_EQ(device->arch(), tt::ARCH::QUASAR);
    const auto cores = keraunos_eth_cores(device);
    ASSERT_FALSE(cores.empty()) << "Chip 0's descriptor lists no Keraunos cores.";

    Program program = CreateProgram();
    std::vector<CoreCoord> logical;
    for (const auto& core : cores) {
        logical.push_back(core.logical);
        CreateKernel(
            program,
            "tests/tt_metal/tt_metal/test_kernels/dataflow/blank.cpp",
            core.logical,
            EthernetConfig{.eth_mode = eth_mode_of(core.type), .noc = NOC::NOC_0});
    }
    const CoreRangeSet all(logical);
    constexpr uint32_t kInitial = 0xC0DE0005;
    ASSERT_EQ(CreateSemaphore(program, all, kInitial, CoreType::ETH), kCounterSem);

    program.impl().compile(device);
    program.impl().finalize_offsets(device);
    for (const auto& core : cores) {
        expect_in_kernel_config(core, semaphore_address(device, program, core, kCounterSem));
    }
    if (is_mock()) {
        GTEST_SKIP() << "Mock: addresses checked. Quasar Mock builds no kernel binaries, so nothing launches.";
    }

    slow_dispatch::LaunchProgramAsync(*device, program, /*force_slow_dispatch=*/false);
    slow_dispatch::WaitProgramDone(*device, program);

    for (const auto& core : cores) {
        const uint32_t address = semaphore_address(device, program, core, kCounterSem);
        EXPECT_EQ(read_word(*mesh, core, address), kInitial) << core.logical.str();
        std::vector<uint32_t> value{0xA5A50000u | core.hsio};
        slow_dispatch::WriteToL1(*mesh, core.logical, address, value, CoreType::ETH);
        EXPECT_EQ(read_word(*mesh, core, address), value[0]) << core.logical.str();
    }
}

// Each Keraunos core of chip 0 increments the counter semaphore of the next core in eth-list order
// (tile 0 -> tile 2 and tile 2 -> tile 0 on chip A). One writer per counter.
class KeraunosCrossTileSemaphore : public ::testing::TestWithParam<uint32_t> {};

TEST_P(KeraunosCrossTileSemaphore, IncrementPeerCounter) {
    const uint32_t mode = GetParam();
    constexpr uint32_t kIncrements = 4;
    constexpr uint32_t kCounterInitial = 100;
    constexpr uint32_t kProbeBase = 0x5EED0000;
    constexpr auto kTimeout = std::chrono::minutes(10);

    auto mesh = distributed::MeshDevice::create_unit_mesh(0);
    IDevice* device = &slow_dispatch::physical_device_from_unit_mesh(*mesh);
    ASSERT_EQ(device->arch(), tt::ARCH::QUASAR);
    const auto cores = keraunos_eth_cores(device);
    ASSERT_GE(cores.size(), 2u) << "Chip 0's descriptor lists fewer than two Keraunos cores.";
    const auto& soc_desc = MetalContext::instance().get_cluster().get_soc_desc(device->id());
    const auto& chiplet = soc_desc.get_arch_descriptor().get_keraunos();
    ASSERT_TRUE(chiplet.tile_stride.has_value()) << "Descriptor has no keraunos.tile_stride.";
    const uint64_t eth_l1_size = soc_desc.get_arch_descriptor().get_eth_l1_size();

    Program program = CreateProgram();
    std::vector<KernelHandle> kernels;
    std::vector<CoreCoord> logical;
    for (const auto& core : cores) {
        logical.push_back(core.logical);
        kernels.push_back(CreateKernel(
            program,
            "tests/tt_metal/tt_fabric/keraunos/kernels/cross_tile_semaphore_increment.cpp",
            core.logical,
            EthernetConfig{.eth_mode = eth_mode_of(core.type), .noc = NOC::NOC_0}));
        // Final argument count before finalize_offsets sizes the runtime-arg region.
        SetRuntimeArgs(program, kernels.back(), core.logical, std::vector<uint32_t>(8, 0));
    }
    const CoreRangeSet all(logical);
    ASSERT_EQ(CreateSemaphore(program, all, kCounterInitial, CoreType::ETH), kCounterSem);
    ASSERT_EQ(CreateSemaphore(program, all, 0, CoreType::ETH), kStageSem);
    // Probe values differ per core, so one semaphore per core.
    for (const auto& core : cores) {
        ASSERT_EQ(CreateSemaphore(program, core.logical, kProbeBase | core.hsio, CoreType::ETH), kProbeSem);
    }
    ASSERT_EQ(CreateSemaphore(program, all, 0, CoreType::ETH), kObservedSem);

    program.impl().compile(device);
    program.impl().finalize_offsets(device);

    for (size_t i = 0; i < cores.size(); i++) {
        const EthCore& self = cores[i];
        const EthCore& peer = cores[(i + 1) % cores.size()];
        const uint32_t peer_counter = semaphore_address(device, program, peer, kCounterSem);
        const uint64_t peer_global = kHsioGlobalSramBase + static_cast<uint64_t>(peer.hsio) * *chiplet.tile_stride +
                                     static_cast<uint64_t>(peer.cce) * eth_l1_size + peer_counter;
        const CoreCoord peer_noc = device->virtual_core_from_logical_core(peer.logical, CoreType::ETH);
        const uint32_t stage = semaphore_address(device, program, self, kStageSem);
        expect_in_kernel_config(self, stage);
        expect_in_kernel_config(peer, peer_counter);
        log_info(
            tt::LogTest,
            "Mode {}: hsio {} ({}) -> hsio {} counter {:#x}, global {:#x}, NOC ({}, {})",
            mode,
            self.hsio,
            self.logical.str(),
            peer.hsio,
            peer_counter,
            peer_global,
            peer_noc.x,
            peer_noc.y);
        SetRuntimeArgs(
            program,
            kernels[i],
            self.logical,
            {mode,
             kIncrements,
             static_cast<uint32_t>(peer_global),
             static_cast<uint32_t>(peer_global >> 32),
             static_cast<uint32_t>(peer_noc.x),
             static_cast<uint32_t>(peer_noc.y),
             peer_counter,
             stage});
    }

    if (is_mock()) {
        GTEST_SKIP() << "Mock: addresses and arguments checked. Quasar Mock builds no kernel binaries, so nothing "
                        "launches.";
    }
    slow_dispatch::LaunchProgramAsync(*device, program, /*force_slow_dispatch=*/false);

    // Poll the stage words rather than block in WaitProgramDone, so a hang reports where it stopped.
    std::vector<uint32_t> stage_addresses;
    for (const auto& core : cores) {
        stage_addresses.push_back(semaphore_address(device, program, core, kStageSem));
    }
    const auto deadline = std::chrono::steady_clock::now() + kTimeout;
    std::vector<uint32_t> stages(cores.size(), 0);
    while (std::ranges::any_of(stages, [](uint32_t s) { return s != kStageDone; })) {
        if (std::chrono::steady_clock::now() > deadline) {
            for (size_t i = 0; i < cores.size(); i++) {
                ADD_FAILURE() << fmt::format("hsio {} stuck at stage {:#x}", cores[i].hsio, stages[i]);
            }
            return;
        }
        std::this_thread::sleep_for(std::chrono::seconds(1));
        for (size_t i = 0; i < cores.size(); i++) {
            if (stages[i] != kStageDone) {
                stages[i] = read_word(*mesh, cores[i], stage_addresses[i]);
            }
        }
    }
    slow_dispatch::WaitProgramDone(*device, program);

    for (const auto& core : cores) {
        EXPECT_EQ(
            read_word(*mesh, core, semaphore_address(device, program, core, kCounterSem)),
            kCounterInitial + kIncrements)
            << "counter on hsio " << core.hsio;
        EXPECT_EQ(read_word(*mesh, core, semaphore_address(device, program, core, kObservedSem)), kProbeBase | core.hsio)
            << "probe copy on hsio " << core.hsio;
    }
}

INSTANTIATE_TEST_SUITE_P(
    Modes,
    KeraunosCrossTileSemaphore,
    // Mode 1 (noc_semaphore_inc) needs coordinate-addressed NOC writes from a CCE, which do not route yet.
    ::testing::Values(0u),
    [](const ::testing::TestParamInfo<uint32_t>& info) {
        return std::string(info.param == 0 ? "LoadAddStore" : "NocSemaphoreInc");
    });

}  // namespace
}  // namespace tt::tt_metal
