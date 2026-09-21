// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <fmt/format.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <set>
#include <string_view>
#include <vector>

#include "impl/context/metal_context.hpp"
#include "impl/kernels/kernel.hpp"
#include "llrt/hal.hpp"
#include "llrt/rtoptions.hpp"
#include "llrt/tt_cluster.hpp"
#include "dev_mem_map.h"
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>
#include "umd/device/types/arch.hpp"
#include "umd/device/types/core_coordinates.hpp"

namespace tt::tt_metal {
namespace {

static_assert(MEM_L1_UNCACHED_BASE == MEM_L1_SIZE);

// Kernel stores use MEM_L1_UNCACHED_BASE. Host cluster SRAM uses MEM_CCE_L1_NOC_OFFSET plus a
// 0-based offset; UMD maps that onto the host AXI window. Those are different windows.
constexpr uint64_t kCceSramTestOffset = 0x800;
constexpr uint32_t kDramOffset = 0x800;
constexpr uint64_t kGddrSlotStride = 1ull << 30;
// Mimir/MMK emu GDDR is an 8GiB SPA with a 1GiB sparse backing store
// (8 channels * 4096 pages * 32KiB). Unique 32KiB pages persist until reset.
constexpr uint64_t kEmuGddrBackingBudget = 1ull << 30;
constexpr uint64_t kEmuGddrBackingPage = 32ull << 10;

struct GddrBackingTracker {
    std::set<uint64_t> pages;

    void note(uint32_t mimir_index, uint64_t address, uint64_t size) {
        if (size == 0) {
            return;
        }
        const uint64_t first = address / kEmuGddrBackingPage;
        const uint64_t last = (address + size - 1) / kEmuGddrBackingPage;
        for (uint64_t page = first; page <= last; ++page) {
            pages.insert((static_cast<uint64_t>(mimir_index) << 32) | page);
        }
        const uint64_t used = pages.size() * kEmuGddrBackingPage;
        ASSERT_LE(used, kEmuGddrBackingBudget)
            << "EMU GDDR unique backing " << used << " B exceeds the 1GiB sparse budget";
    }
};

thread_local GddrBackingTracker* t_gddr_backing = nullptr;

struct GddrBackingScope {
    GddrBackingTracker tracker;
    GddrBackingScope() { t_gddr_backing = &tracker; }
    ~GddrBackingScope() { t_gddr_backing = nullptr; }
};

uint32_t mimir_index_for_cce(uint32_t cce_index) { return cce_index / 2; }

uint32_t location_for_cce(uint32_t cce_index) { return cce_index % 2; }

CoreCoord logical_cce_core(uint32_t cce_index) { return {mimir_index_for_cce(cce_index), location_for_cce(cce_index)}; }

uint32_t num_cces_from_soc(const metal_SocDescriptor& soc_desc) {
    return static_cast<uint32_t>(soc_desc.get_num_dram_channels() * soc_desc.get_grid_size(tt::CoreType::DRAM).y);
}

uint32_t num_cces_on(const IDevice* device) {
    return num_cces_from_soc(MetalContext::instance().get_cluster().get_soc_desc(device->id()));
}

struct GddrSweep {
    uint32_t num_slots = 0;
    uint32_t slot_stride = 0;
};

GddrSweep gddr_sweep_from_soc(const IDevice* device) {
    const uint64_t view_size = MetalContext::instance().get_cluster().get_soc_desc(device->id()).dram_view_size;
    if (view_size < kGddrSlotStride || view_size % kGddrSlotStride != 0) {
        return {};
    }
    return {static_cast<uint32_t>(view_size / kGddrSlotStride), static_cast<uint32_t>(kGddrSlotStride)};
}

struct GddrTarget {
    uint32_t mimir_index = 0;
    uint32_t location = 0;
    uint32_t slot_base = 0;
    uint32_t num_slots = 0;
    uint32_t slot_stride = 0;
};

uint32_t host_channel_for_mimir(uint32_t mimir_index) { return mimir_index; }

uint32_t num_mimirs_on(const IDevice* device) { return device->num_dram_channels(); }

bool is_cross_mimir_case() {
    const char* suite = ::testing::UnitTest::GetInstance()->current_test_info()->test_suite_name();
    return std::string_view(suite).find("CrossMimir") != std::string_view::npos;
}

uint32_t locations_per_mimir(const IDevice* device) {
    return static_cast<uint32_t>(
        MetalContext::instance().get_cluster().get_soc_desc(device->id()).get_grid_size(tt::CoreType::DRAM).y);
}

std::vector<GddrTarget> gddr_targets_for(const IDevice* device, uint32_t cce_index, const GddrSweep& sweep) {
    const uint32_t local = mimir_index_for_cce(cce_index);
    const uint32_t num_mimirs = num_mimirs_on(device);
    std::vector<GddrTarget> targets;
    if (!is_cross_mimir_case()) {
        targets.push_back({local, 0, 0, sweep.num_slots, sweep.slot_stride});
        return targets;
    }
    const uint32_t num_locations = locations_per_mimir(device);
    if (num_locations == 0 || sweep.num_slots % num_locations != 0) {
        return {};
    }
    const uint32_t slots_per_location = sweep.num_slots / num_locations;
    for (uint32_t mimir_index = 0; mimir_index < num_mimirs; ++mimir_index) {
        if (mimir_index == local) {
            continue;
        }
        for (uint32_t location = 0; location < num_locations; ++location) {
            targets.push_back(
                {mimir_index, location, location * slots_per_location, slots_per_location, sweep.slot_stride});
        }
    }
    return targets;
}

void write_mimir_gddr(IDevice* device, uint32_t mimir_index, uint64_t address, const std::vector<uint32_t>& data) {
    const uint32_t size = static_cast<uint32_t>(data.size() * sizeof(uint32_t));
    if (t_gddr_backing != nullptr) {
        t_gddr_backing->note(mimir_index, address, size);
    }
    MetalContext::instance().get_cluster().write_dram_vec(
        data.data(), size, device->id(), static_cast<int>(host_channel_for_mimir(mimir_index)), address);
}

void read_mimir_gddr(
    IDevice* device, uint32_t mimir_index, uint64_t address, uint32_t size, std::vector<uint32_t>& data) {
    if (t_gddr_backing != nullptr) {
        t_gddr_backing->note(mimir_index, address, size);
    }
    data.assign(size / sizeof(uint32_t), 0);
    MetalContext::instance().get_cluster().dram_barrier(device->id());
    MetalContext::instance().get_cluster().read_dram_vec(
        data.data(), size, device->id(), static_cast<int>(host_channel_for_mimir(mimir_index)), address);
}

bool emu_server_configured() {
    return std::getenv("TT_METAL_EMU_SERVER") != nullptr && std::getenv("TT_METAL_EMU_SOC_DESC") != nullptr;
}

bool sival_bringup() {
    const char* bringup = std::getenv("TT_METAL_EMU_BRINGUP");
    return bringup != nullptr && std::string_view(bringup) == "sival";
}

constexpr std::string_view kSivalSramOnly =
    "TT_METAL_EMU_BRINGUP=sival supports one-Mimir CCE SRAM access only; use umd_server for this test.";

CoreCoord translated_dram_core(const metal_SocDescriptor& soc_desc, uint32_t cce_index) {
    const auto core = soc_desc.translate_coord_to(
        tt::umd::CoreCoord(
            mimir_index_for_cce(cce_index), location_for_cce(cce_index), CoreType::DRAM, CoordSystem::LOGICAL),
        CoordSystem::TRANSLATED);
    return {core.x, core.y};
}

class CceSramRoundTripThroughMetalCluster : public ::testing::TestWithParam<uint32_t> {};

TEST_P(CceSramRoundTripThroughMetalCluster, RoundTrip) {
    if (!emu_server_configured()) {
        GTEST_SKIP() << "Set TT_METAL_EMU_SERVER=host:port and TT_METAL_EMU_SOC_DESC=<mimir YAML>.";
    }

    llrt::RunTimeOptions rtoptions;
    Cluster cluster(rtoptions);

    ASSERT_EQ(cluster.arch(), tt::ARCH::QUASAR);
    ASSERT_EQ(cluster.all_chip_ids().size(), 1);

    constexpr ChipId chip_id = 0;
    const uint32_t cce_index = GetParam();
    const auto& soc_desc = cluster.get_soc_desc(chip_id);
    const uint32_t num_cces = num_cces_from_soc(soc_desc);
    if (sival_bringup() && num_cces > 2) {
        GTEST_SKIP() << "TT_METAL_EMU_BRINGUP=sival requires the single-Mimir mimir_1x1.yaml descriptor.";
    }
    if (cce_index >= num_cces) {
        GTEST_SKIP() << "Configured descriptor exposes only " << num_cces << " CCEs.";
    }
    const CoreCoord cce = translated_dram_core(soc_desc, cce_index);
    constexpr uint64_t address = MEM_CCE_L1_NOC_OFFSET + kCceSramTestOffset;
    const uint32_t written = 0xC0FFEE01 + cce_index;
    uint32_t read_back = 0;

    cluster.write_core(&written, sizeof(written), {chip_id, cce}, address);
    cluster.read_core(&read_back, sizeof(read_back), {chip_id, cce}, address);

    EXPECT_EQ(read_back, written);
}

TEST(MimirEmu, CceSramChannelsDoNotAliasThroughMetalCluster) {
    if (!emu_server_configured()) {
        GTEST_SKIP() << "Set TT_METAL_EMU_SERVER=host:port and TT_METAL_EMU_SOC_DESC=<mimir YAML>.";
    }

    llrt::RunTimeOptions rtoptions;
    Cluster cluster(rtoptions);

    ASSERT_EQ(cluster.arch(), tt::ARCH::QUASAR);
    ASSERT_EQ(cluster.all_chip_ids().size(), 1);

    constexpr ChipId chip_id = 0;
    const auto& soc_desc = cluster.get_soc_desc(chip_id);
    const uint32_t num_cces = num_cces_from_soc(soc_desc);
    if (sival_bringup() && num_cces > 2) {
        GTEST_SKIP() << "TT_METAL_EMU_BRINGUP=sival requires the single-Mimir mimir_1x1.yaml descriptor.";
    }
    ASSERT_GE(num_cces, 2u);

    // Every channel is written before any is read: an aliasing window only shows up once a later
    // channel has landed on top of an earlier one.
    constexpr uint64_t address = MEM_CCE_L1_NOC_OFFSET + kCceSramTestOffset;
    for (uint32_t cce_index = 0; cce_index < num_cces; ++cce_index) {
        const uint32_t written = 0xC0FFEE01 + cce_index;
        cluster.write_core(&written, sizeof(written), {chip_id, translated_dram_core(soc_desc, cce_index)}, address);
    }
    for (uint32_t cce_index = 0; cce_index < num_cces; ++cce_index) {
        uint32_t read_back = 0;
        cluster.read_core(&read_back, sizeof(read_back), {chip_id, translated_dram_core(soc_desc, cce_index)}, address);
        EXPECT_EQ(read_back, 0xC0FFEE01 + cce_index) << "CCE" << cce_index;
    }
}

class CceSramRoundTripThroughMinimalDevice : public ::testing::TestWithParam<uint32_t> {};

TEST_P(CceSramRoundTripThroughMinimalDevice, RoundTrip) {
    if (!emu_server_configured()) {
        GTEST_SKIP() << "Set TT_METAL_EMU_SERVER=host:port and TT_METAL_EMU_SOC_DESC=<mimir YAML>.";
    }

    std::unique_ptr<IDevice> device(CreateDeviceMinimal(0));
    ASSERT_NE(device, nullptr);
    ASSERT_EQ(device->arch(), tt::ARCH::QUASAR);

    const uint32_t cce_index = GetParam();
    const uint32_t num_cces = num_cces_on(device.get());
    if (sival_bringup() && num_cces > 2) {
        device.reset();
        GTEST_SKIP() << "TT_METAL_EMU_BRINGUP=sival requires the single-Mimir mimir_1x1.yaml descriptor.";
    }
    if (cce_index >= num_cces) {
        device.reset();
        GTEST_SKIP() << "Configured descriptor does not expose CCE " << cce_index << ".";
    }

    Cluster& cluster = MetalContext::instance().get_cluster();
    const CoreCoord cce = device->virtual_core_from_logical_core(logical_cce_core(cce_index), CoreType::DRAM);
    constexpr uint64_t address = MEM_CCE_L1_NOC_OFFSET + kCceSramTestOffset;
    const uint32_t written = 0xC0FFEE03 + cce_index;
    uint32_t read_back = 0;
    cluster.write_core(&written, sizeof(written), {device->id(), cce}, address);
    cluster.read_core(&read_back, sizeof(read_back), {device->id(), cce}, address);
    EXPECT_EQ(read_back, written);
}

TEST(MimirEmu, RuntimeFirmwareInitializesThroughPublicDeviceApi) {
    if (!emu_server_configured()) {
        GTEST_SKIP() << "Set TT_METAL_EMU_SERVER=host:port and TT_METAL_EMU_SOC_DESC=<mimir YAML>.";
    }

    IDevice* device = nullptr;
    ASSERT_NO_THROW(device = CreateDevice(0));
    ASSERT_NE(device, nullptr);
    ASSERT_EQ(device->arch(), tt::ARCH::QUASAR);
    const uint32_t num_cces = num_cces_on(device);
    if (sival_bringup() && num_cces > 2) {
        EXPECT_TRUE(CloseDevice(device));
        GTEST_SKIP() << "TT_METAL_EMU_BRINGUP=sival requires the single-Mimir mimir_1x1.yaml descriptor.";
    }
    EXPECT_TRUE(CloseDevice(device));
}

class HartZeroRunsDramKernel : public ::testing::TestWithParam<uint32_t> {};

TEST_P(HartZeroRunsDramKernel, WritesMagic) {
    if (!emu_server_configured()) {
        GTEST_SKIP() << "Set TT_METAL_EMU_SERVER=host:port and TT_METAL_EMU_SOC_DESC=<mimir YAML>.";
    }

    IDevice* device = CreateDevice(0);
    ASSERT_NE(device, nullptr);
    ASSERT_EQ(device->arch(), tt::ARCH::QUASAR);

    const uint32_t cce_index = GetParam();
    const uint32_t num_cces = num_cces_on(device);
    if (sival_bringup() && num_cces > 2) {
        EXPECT_TRUE(CloseDevice(device));
        GTEST_SKIP() << "TT_METAL_EMU_BRINGUP=sival requires the single-Mimir mimir_1x1.yaml descriptor.";
    }
    if (cce_index >= num_cces) {
        EXPECT_TRUE(CloseDevice(device));
        GTEST_SKIP() << "Configured descriptor does not expose CCE " << cce_index << ".";
    }

    const CoreCoord logical_dram_core = logical_cce_core(cce_index);
    const uint32_t magic = 0xC0FFEE04 + cce_index;
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

    detail::LaunchProgram(device, program, /*wait_until_cores_done=*/true, /*force_slow_dispatch=*/true);

    uint32_t result = 0;
    const CoreCoord virtual_dram_core = device->virtual_core_from_logical_core(logical_dram_core, CoreType::DRAM);
    MetalContext::instance().get_cluster().read_core(
        &result, sizeof(result), {device->id(), virtual_dram_core}, result_noc_addr);
    EXPECT_EQ(result, magic);
    EXPECT_TRUE(CloseDevice(device));
}

class AllHartsRunDramKernel : public ::testing::TestWithParam<uint32_t> {};

TEST_P(AllHartsRunDramKernel, WritesMagic) {
    if (!emu_server_configured()) {
        GTEST_SKIP() << "Set TT_METAL_EMU_SERVER=host:port and TT_METAL_EMU_SOC_DESC=<mimir YAML>.";
    }

    IDevice* device = CreateDevice(0);
    ASSERT_NE(device, nullptr);
    ASSERT_EQ(device->arch(), tt::ARCH::QUASAR);

    const uint32_t cce_index = GetParam();
    const uint32_t num_cces = num_cces_on(device);
    if (sival_bringup() && num_cces > 2) {
        EXPECT_TRUE(CloseDevice(device));
        GTEST_SKIP() << "TT_METAL_EMU_BRINGUP=sival requires the single-Mimir mimir_1x1.yaml descriptor.";
    }
    if (cce_index >= num_cces) {
        EXPECT_TRUE(CloseDevice(device));
        GTEST_SKIP() << "Configured descriptor does not expose CCE " << cce_index << ".";
    }

    const CoreCoord logical_dram_core = logical_cce_core(cce_index);
    const auto& hal = MetalContext::instance().hal();
    const uint32_t num_harts = hal.get_num_risc_processors(HalProgrammableCoreType::DRAM);
    ASSERT_EQ(num_harts, 8u);

    const uint32_t result_dev_base =
        MEM_L1_UNCACHED_BASE + hal.get_dev_addr(HalProgrammableCoreType::DRAM, HalL1MemAddrType::UNRESERVED);
    const uint64_t result_noc_base = hal.get_dev_noc_addr(HalProgrammableCoreType::DRAM, HalL1MemAddrType::UNRESERVED);

    Program program = CreateProgram();
    std::vector<uint32_t> expected(num_harts);
    for (uint32_t hart = 0; hart < num_harts; hart++) {
        expected[hart] = 0xC0FFEE10 + (cce_index << 8) + hart;
        CreateKernel(
            program,
            "tests/tt_metal/tt_metal/test_kernels/misc/dram_write_one_uint32.cpp",
            logical_dram_core,
            DramConfig{
                .processor = static_cast<DataMovementProcessor>(hart),
                .noc = NOC::NOC_0,
                .compile_args = {result_dev_base + hart * static_cast<uint32_t>(sizeof(uint32_t)), expected[hart]}});
    }

    detail::LaunchProgram(device, program, /*wait_until_cores_done=*/true, /*force_slow_dispatch=*/true);

    std::vector<uint32_t> result(num_harts, 0);
    const CoreCoord virtual_dram_core = device->virtual_core_from_logical_core(logical_dram_core, CoreType::DRAM);
    MetalContext::instance().get_cluster().read_core(
        result.data(), result.size() * sizeof(uint32_t), {device->id(), virtual_dram_core}, result_noc_base);
    EXPECT_EQ(result, expected);
    EXPECT_TRUE(CloseDevice(device));
}

class CopiesGddrWithCompileTimeArgs : public ::testing::TestWithParam<uint32_t> {};

TEST_P(CopiesGddrWithCompileTimeArgs, RoundTrip) {
    if (!emu_server_configured()) {
        GTEST_SKIP() << "Set TT_METAL_EMU_SERVER=host:port and TT_METAL_EMU_SOC_DESC=<mimir YAML>.";
    }
    if (sival_bringup()) {
        GTEST_SKIP() << kSivalSramOnly;
    }

    IDevice* device = CreateDevice(0);
    ASSERT_NE(device, nullptr);
    ASSERT_EQ(device->arch(), tt::ARCH::QUASAR);

    const uint32_t cce_index = GetParam();
    const uint32_t num_mimirs = num_mimirs_on(device);
    if (cce_index >= num_cces_on(device) || num_mimirs == 0 || (is_cross_mimir_case() && num_mimirs < 2)) {
        EXPECT_TRUE(CloseDevice(device));
        GTEST_SKIP() << "Configured descriptor does not expose the requested CCE/Mimir.";
    }
    const GddrSweep sweep = gddr_sweep_from_soc(device);
    ASSERT_NE(sweep.num_slots, 0u) << "dram_view_size must be a non-zero multiple of 1GiB";
    const std::vector<GddrTarget> targets = gddr_targets_for(device, cce_index, sweep);
    ASSERT_FALSE(targets.empty()) << "No GDDR targets for this CCE/suite.";
    GddrBackingScope gddr_backing;

    const auto& hal = MetalContext::instance().hal();
    const uint32_t transfer_size = hal.get_alignment(HalMemType::DRAM);
    const uint32_t src_dram_offset = hal.get_dev_addr(HalDramMemAddrType::UNRESERVED);
    const uint32_t dst_dram_offset = src_dram_offset + transfer_size;
    const uint32_t staging_dev_addr = hal.get_dev_addr(HalProgrammableCoreType::DRAM, HalL1MemAddrType::UNRESERVED);
    const uint64_t staging_noc_addr = hal.get_dev_noc_addr(HalProgrammableCoreType::DRAM, HalL1MemAddrType::UNRESERVED);
    const CoreCoord logical_dram_core = logical_cce_core(cce_index);
    const CoreCoord virtual_dram_core = device->virtual_core_from_logical_core(logical_dram_core, CoreType::DRAM);
    std::vector<uint32_t> cleared(transfer_size / sizeof(uint32_t), 0);

    for (const GddrTarget& target : targets) {
        std::vector<std::vector<uint32_t>> inputs(
            target.num_slots, std::vector<uint32_t>(transfer_size / sizeof(uint32_t)));
        for (uint32_t slot = 0; slot < target.num_slots; ++slot) {
            for (std::size_t i = 0; i < inputs[slot].size(); ++i) {
                inputs[slot][i] = 0xC0FF0000u | (cce_index << 12) | (target.mimir_index << 8) | (target.location << 4) |
                                  static_cast<uint32_t>(i);
            }
            const uint64_t slot_offset = static_cast<uint64_t>(target.slot_base + slot) * target.slot_stride;
            write_mimir_gddr(device, target.mimir_index, src_dram_offset + slot_offset, inputs[slot]);
            write_mimir_gddr(device, target.mimir_index, dst_dram_offset + slot_offset, cleared);
        }

        Program program = CreateProgram();
        CreateKernel(
            program,
            "tests/tt_metal/tt_metal/test_kernels/misc/dram_gddr_round_trip.cpp",
            logical_dram_core,
            DramConfig{
                .noc = NOC::NOC_0,
                .compile_args = {
                    src_dram_offset,
                    dst_dram_offset,
                    target.mimir_index,
                    staging_dev_addr,
                    static_cast<uint32_t>(cleared.size()),
                    target.num_slots,
                    target.slot_stride,
                    target.slot_base}});

        detail::LaunchProgram(device, program, /*wait_until_cores_done=*/true, /*force_slow_dispatch=*/true);

        std::vector<uint32_t> staged(cleared.size(), 0);
        MetalContext::instance().get_cluster().read_core(
            staged.data(), transfer_size, {device->id(), virtual_dram_core}, staging_noc_addr);
        EXPECT_EQ(staged, inputs.back()) << "mimir " << target.mimir_index << " location " << target.location;

        for (uint32_t slot = 0; slot < target.num_slots; ++slot) {
            std::vector<uint32_t> output;
            read_mimir_gddr(
                device,
                target.mimir_index,
                dst_dram_offset + static_cast<uint64_t>(target.slot_base + slot) * target.slot_stride,
                transfer_size,
                output);
            EXPECT_EQ(output, inputs[slot])
                << "mimir " << target.mimir_index << " location " << target.location << " slot " << slot;
        }
    }
    EXPECT_TRUE(CloseDevice(device));
}

class CopiesGddrThroughAllocatedBuffer : public ::testing::TestWithParam<uint32_t> {};

TEST_P(CopiesGddrThroughAllocatedBuffer, RoundTrip) {
    if (!emu_server_configured()) {
        GTEST_SKIP() << "Set TT_METAL_EMU_SERVER=host:port and TT_METAL_EMU_SOC_DESC=<mimir YAML>.";
    }
    if (sival_bringup()) {
        GTEST_SKIP() << kSivalSramOnly;
    }

    IDevice* device = CreateDevice(0);
    ASSERT_NE(device, nullptr);
    ASSERT_EQ(device->arch(), tt::ARCH::QUASAR);

    const uint32_t cce_index = GetParam();
    const uint32_t num_mimirs = num_mimirs_on(device);
    if (cce_index >= num_cces_on(device) || num_mimirs == 0 || (is_cross_mimir_case() && num_mimirs < 2)) {
        EXPECT_TRUE(CloseDevice(device));
        GTEST_SKIP() << "Configured descriptor does not expose the requested CCE/Mimir.";
    }
    const GddrSweep sweep = gddr_sweep_from_soc(device);
    ASSERT_NE(sweep.num_slots, 0u) << "dram_view_size must be a non-zero multiple of 1GiB";
    const std::vector<GddrTarget> targets = gddr_targets_for(device, cce_index, sweep);
    ASSERT_FALSE(targets.empty()) << "No GDDR targets for this CCE/suite.";
    GddrBackingScope gddr_backing;

    const auto& hal = MetalContext::instance().hal();
    const uint32_t transfer_size = hal.get_alignment(HalMemType::DRAM);
    const uint32_t src_dram_offset = hal.get_dev_addr(HalDramMemAddrType::UNRESERVED);
    const uint32_t dst_dram_offset = src_dram_offset + transfer_size;
    const uint32_t staging_dev_addr = hal.get_dev_addr(HalProgrammableCoreType::DRAM, HalL1MemAddrType::UNRESERVED);
    const CoreCoord logical_dram_core = logical_cce_core(cce_index);
    std::vector<uint32_t> cleared(transfer_size / sizeof(uint32_t), 0);

    for (const GddrTarget& target : targets) {
        std::vector<std::vector<uint32_t>> inputs(
            target.num_slots, std::vector<uint32_t>(transfer_size / sizeof(uint32_t)));
        for (uint32_t slot = 0; slot < target.num_slots; ++slot) {
            for (std::size_t i = 0; i < inputs[slot].size(); ++i) {
                inputs[slot][i] = 0xC0FE0000u | (cce_index << 12) | (target.mimir_index << 8) | (target.location << 4) |
                                  static_cast<uint32_t>(i);
            }
            const uint64_t slot_offset = static_cast<uint64_t>(target.slot_base + slot) * target.slot_stride;
            write_mimir_gddr(device, target.mimir_index, src_dram_offset + slot_offset, inputs[slot]);
            write_mimir_gddr(device, target.mimir_index, dst_dram_offset + slot_offset, cleared);
        }

        Program program = CreateProgram();
        CreateKernel(
            program,
            "tests/tt_metal/tt_metal/test_kernels/misc/dram_gddr_round_trip.cpp",
            logical_dram_core,
            DramConfig{
                .noc = NOC::NOC_0,
                .compile_args = {
                    src_dram_offset,
                    dst_dram_offset,
                    target.mimir_index,
                    staging_dev_addr,
                    static_cast<uint32_t>(cleared.size()),
                    target.num_slots,
                    target.slot_stride,
                    target.slot_base}});

        detail::LaunchProgram(device, program, /*wait_until_cores_done=*/true, /*force_slow_dispatch=*/true);

        for (uint32_t slot = 0; slot < target.num_slots; ++slot) {
            std::vector<uint32_t> output;
            read_mimir_gddr(
                device,
                target.mimir_index,
                dst_dram_offset + static_cast<uint64_t>(target.slot_base + slot) * target.slot_stride,
                transfer_size,
                output);
            EXPECT_EQ(output, inputs[slot])
                << "mimir " << target.mimir_index << " location " << target.location << " slot " << slot;
        }
    }
    EXPECT_TRUE(CloseDevice(device));
}

class CopiesAllocatedBufferWithRuntimeArgs : public ::testing::TestWithParam<uint32_t> {};

TEST_P(CopiesAllocatedBufferWithRuntimeArgs, RoundTrip) {
    if (!emu_server_configured()) {
        GTEST_SKIP() << "Set TT_METAL_EMU_SERVER=host:port and TT_METAL_EMU_SOC_DESC=<mimir YAML>.";
    }
    if (sival_bringup()) {
        GTEST_SKIP() << kSivalSramOnly;
    }

    IDevice* device = CreateDevice(0);
    ASSERT_NE(device, nullptr);
    ASSERT_EQ(device->arch(), tt::ARCH::QUASAR);

    const uint32_t cce_index = GetParam();
    const uint32_t num_mimirs = num_mimirs_on(device);
    if (cce_index >= num_cces_on(device) || num_mimirs == 0 || (is_cross_mimir_case() && num_mimirs < 2)) {
        EXPECT_TRUE(CloseDevice(device));
        GTEST_SKIP() << "Configured descriptor does not expose the requested CCE/Mimir.";
    }
    const GddrSweep sweep = gddr_sweep_from_soc(device);
    ASSERT_NE(sweep.num_slots, 0u) << "dram_view_size must be a non-zero multiple of 1GiB";
    const std::vector<GddrTarget> targets = gddr_targets_for(device, cce_index, sweep);
    ASSERT_FALSE(targets.empty()) << "No GDDR targets for this CCE/suite.";
    GddrBackingScope gddr_backing;

    const auto& hal = MetalContext::instance().hal();
    const uint32_t transfer_size = hal.get_alignment(HalMemType::DRAM);
    const uint32_t src_dram_offset = hal.get_dev_addr(HalDramMemAddrType::UNRESERVED);
    const uint32_t dst_dram_offset = src_dram_offset + transfer_size;
    const uint32_t staging_dev_addr = hal.get_dev_addr(HalProgrammableCoreType::DRAM, HalL1MemAddrType::UNRESERVED);
    const CoreCoord logical_dram_core = logical_cce_core(cce_index);
    std::vector<uint32_t> cleared(transfer_size / sizeof(uint32_t), 0);

    for (const GddrTarget& target : targets) {
        std::vector<std::vector<uint32_t>> inputs(
            target.num_slots, std::vector<uint32_t>(transfer_size / sizeof(uint32_t)));
        for (uint32_t slot = 0; slot < target.num_slots; ++slot) {
            for (std::size_t i = 0; i < inputs[slot].size(); ++i) {
                inputs[slot][i] = 0xD15C0000u | (cce_index << 12) | (target.mimir_index << 8) | (target.location << 4) |
                                  static_cast<uint32_t>(i);
            }
            const uint64_t slot_offset = static_cast<uint64_t>(target.slot_base + slot) * target.slot_stride;
            write_mimir_gddr(device, target.mimir_index, src_dram_offset + slot_offset, inputs[slot]);
            write_mimir_gddr(device, target.mimir_index, dst_dram_offset + slot_offset, cleared);
        }

        Program program = CreateProgram();
        const KernelHandle kernel = CreateKernel(
            program,
            "tests/tt_metal/tt_metal/test_kernels/misc/cce_dram_buffer_round_trip.cpp",
            logical_dram_core,
            DramConfig{.noc = NOC::NOC_0});
        SetRuntimeArgs(
            program,
            kernel,
            logical_dram_core,
            {src_dram_offset,
             dst_dram_offset,
             target.mimir_index,
             staging_dev_addr,
             static_cast<uint32_t>(cleared.size()),
             target.num_slots,
             target.slot_stride,
             target.slot_base});

        detail::LaunchProgram(device, program, /*wait_until_cores_done=*/true, /*force_slow_dispatch=*/true);

        for (uint32_t slot = 0; slot < target.num_slots; ++slot) {
            std::vector<uint32_t> output;
            read_mimir_gddr(
                device,
                target.mimir_index,
                dst_dram_offset + static_cast<uint64_t>(target.slot_base + slot) * target.slot_stride,
                transfer_size,
                output);
            EXPECT_EQ(output, inputs[slot])
                << "mimir " << target.mimir_index << " location " << target.location << " slot " << slot;
        }
    }
    EXPECT_TRUE(CloseDevice(device));
}

class AllocatedDramBufferHostLoopback : public ::testing::TestWithParam<uint32_t> {};

TEST_P(AllocatedDramBufferHostLoopback, RoundTrip) {
    if (!emu_server_configured()) {
        GTEST_SKIP() << "Set TT_METAL_EMU_SERVER=host:port and TT_METAL_EMU_SOC_DESC=<mimir YAML>.";
    }
    if (sival_bringup()) {
        GTEST_SKIP() << kSivalSramOnly;
    }

    IDevice* device = CreateDevice(0);
    ASSERT_NE(device, nullptr);
    ASSERT_EQ(device->arch(), tt::ARCH::QUASAR);

    const uint32_t dram_partition = GetParam();
    if (dram_partition >= device->num_dram_channels()) {
        EXPECT_TRUE(CloseDevice(device));
        GTEST_SKIP() << "Configured descriptor does not expose partition " << dram_partition << ".";
    }
    GddrBackingScope gddr_backing;

    const auto& hal = MetalContext::instance().hal();
    const uint32_t page_size = hal.get_alignment(HalMemType::DRAM);
    const uint32_t buffer_size = page_size * device->num_dram_channels();
    auto buffer = CreateBuffer(BufferConfig{device, buffer_size, page_size, BufferType::DRAM});
    ASSERT_NE(buffer, nullptr);
    for (uint32_t partition = 0; partition < device->num_dram_channels(); ++partition) {
        gddr_backing.tracker.note(partition, buffer->address(), page_size);
    }

    const std::size_t words_per_page = page_size / sizeof(uint32_t);
    std::vector<uint32_t> input(buffer_size / sizeof(uint32_t));
    for (uint32_t partition = 0; partition < device->num_dram_channels(); ++partition) {
        for (std::size_t i = 0; i < words_per_page; ++i) {
            input[partition * words_per_page + i] = 0xA1100000u | (partition << 8) | static_cast<uint32_t>(i);
        }
    }
    detail::WriteToBuffer(*buffer, input);

    std::vector<uint32_t> output;
    detail::ReadFromBuffer(*buffer, output);
    EXPECT_EQ(output, input);

    const auto expected_begin = input.begin() + dram_partition * words_per_page;
    const std::vector<uint32_t> expected(expected_begin, expected_begin + words_per_page);
    std::vector<uint32_t> channel_page;
    ASSERT_TRUE(detail::ReadFromDeviceDRAMChannel(device, dram_partition, buffer->address(), page_size, channel_page));
    EXPECT_EQ(channel_page, expected);

    buffer.reset();
    EXPECT_TRUE(CloseDevice(device));
}

class CopiesEveryMimirBlock : public ::testing::TestWithParam<uint32_t> {};

TEST_P(CopiesEveryMimirBlock, RoundTrip) {
    if (!emu_server_configured()) {
        GTEST_SKIP() << "Set TT_METAL_EMU_SERVER=host:port and TT_METAL_EMU_SOC_DESC=<mimir YAML>.";
    }
    if (sival_bringup()) {
        GTEST_SKIP() << kSivalSramOnly;
    }

    IDevice* device = CreateDevice(0);
    ASSERT_NE(device, nullptr);
    ASSERT_EQ(device->arch(), tt::ARCH::QUASAR);

    const uint32_t cce_index = GetParam();
    const uint32_t num_mimirs = num_mimirs_on(device);
    if (cce_index >= num_cces_on(device) || num_mimirs == 0) {
        EXPECT_TRUE(CloseDevice(device));
        GTEST_SKIP() << "Configured descriptor does not expose CCE " << cce_index << ".";
    }
    GddrBackingScope gddr_backing;

    const auto& hal = MetalContext::instance().hal();
    const uint32_t transfer_size = hal.get_alignment(HalMemType::DRAM);
    const uint32_t src_dram_offset = hal.get_dev_addr(HalDramMemAddrType::UNRESERVED);
    const uint32_t dst_dram_offset = src_dram_offset + transfer_size;
    const uint32_t staging_dev_addr = hal.get_dev_addr(HalProgrammableCoreType::DRAM, HalL1MemAddrType::UNRESERVED);

    std::vector<std::vector<uint32_t>> inputs(num_mimirs, std::vector<uint32_t>(transfer_size / sizeof(uint32_t)));
    std::vector<uint32_t> cleared(transfer_size / sizeof(uint32_t), 0);
    for (uint32_t mimir_index = 0; mimir_index < num_mimirs; ++mimir_index) {
        for (std::size_t i = 0; i < inputs[mimir_index].size(); ++i) {
            inputs[mimir_index][i] = 0xE1E10000u | (cce_index << 12) | (mimir_index << 8) | static_cast<uint32_t>(i);
        }
        write_mimir_gddr(device, mimir_index, src_dram_offset, inputs[mimir_index]);
        write_mimir_gddr(device, mimir_index, dst_dram_offset, cleared);
    }

    const CoreCoord logical_dram_core = logical_cce_core(cce_index);
    Program program = CreateProgram();
    const KernelHandle kernel = CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/misc/cce_dram_copy_all_partitions.cpp",
        logical_dram_core,
        DramConfig{.noc = NOC::NOC_0});
    SetRuntimeArgs(
        program,
        kernel,
        logical_dram_core,
        {src_dram_offset, dst_dram_offset, num_mimirs, staging_dev_addr, static_cast<uint32_t>(cleared.size())});

    detail::LaunchProgram(device, program, /*wait_until_cores_done=*/true, /*force_slow_dispatch=*/true);

    for (uint32_t mimir_index = 0; mimir_index < num_mimirs; ++mimir_index) {
        std::vector<uint32_t> output;
        read_mimir_gddr(device, mimir_index, dst_dram_offset, transfer_size, output);
        EXPECT_EQ(output, inputs[mimir_index]) << "mimir " << mimir_index;
    }
    EXPECT_TRUE(CloseDevice(device));
}

class CopiesAllocatedBufferLargerThanAlignment : public ::testing::TestWithParam<uint32_t> {};

TEST_P(CopiesAllocatedBufferLargerThanAlignment, RoundTrip) {
    if (!emu_server_configured()) {
        GTEST_SKIP() << "Set TT_METAL_EMU_SERVER=host:port and TT_METAL_EMU_SOC_DESC=<mimir YAML>.";
    }
    if (sival_bringup()) {
        GTEST_SKIP() << kSivalSramOnly;
    }

    IDevice* device = CreateDevice(0);
    ASSERT_NE(device, nullptr);
    ASSERT_EQ(device->arch(), tt::ARCH::QUASAR);

    const uint32_t cce_index = GetParam();
    const uint32_t num_mimirs = num_mimirs_on(device);
    if (cce_index >= num_cces_on(device) || num_mimirs == 0 || (is_cross_mimir_case() && num_mimirs < 2)) {
        EXPECT_TRUE(CloseDevice(device));
        GTEST_SKIP() << "Configured descriptor does not expose the requested CCE/Mimir.";
    }
    const GddrSweep sweep = gddr_sweep_from_soc(device);
    ASSERT_NE(sweep.num_slots, 0u) << "dram_view_size must be a non-zero multiple of 1GiB";
    const std::vector<GddrTarget> targets = gddr_targets_for(device, cce_index, sweep);
    ASSERT_FALSE(targets.empty()) << "No GDDR targets for this CCE/suite.";
    GddrBackingScope gddr_backing;

    const auto& hal = MetalContext::instance().hal();
    const uint32_t alignment = hal.get_alignment(HalMemType::DRAM);
    constexpr uint32_t kPagesPerBank = 16;
    const uint32_t transfer_size = std::min(alignment * kPagesPerBank, static_cast<uint32_t>(kEmuGddrBackingPage));
    const uint32_t src_dram_offset = hal.get_dev_addr(HalDramMemAddrType::UNRESERVED);
    const uint32_t dst_dram_offset = src_dram_offset + transfer_size;
    const uint32_t staging_dev_addr = hal.get_dev_addr(HalProgrammableCoreType::DRAM, HalL1MemAddrType::UNRESERVED);
    const CoreCoord logical_dram_core = logical_cce_core(cce_index);
    std::vector<uint32_t> cleared(transfer_size / sizeof(uint32_t), 0);

    for (const GddrTarget& target : targets) {
        std::vector<std::vector<uint32_t>> inputs(
            target.num_slots, std::vector<uint32_t>(transfer_size / sizeof(uint32_t)));
        for (uint32_t slot = 0; slot < target.num_slots; ++slot) {
            for (std::size_t i = 0; i < inputs[slot].size(); ++i) {
                inputs[slot][i] = 0x1A6E0000u | (cce_index << 12) | (target.mimir_index << 8) | (target.location << 4) |
                                  static_cast<uint32_t>(i);
            }
            const uint64_t slot_offset = static_cast<uint64_t>(target.slot_base + slot) * target.slot_stride;
            write_mimir_gddr(device, target.mimir_index, src_dram_offset + slot_offset, inputs[slot]);
            write_mimir_gddr(device, target.mimir_index, dst_dram_offset + slot_offset, cleared);
        }

        Program program = CreateProgram();
        const KernelHandle kernel = CreateKernel(
            program,
            "tests/tt_metal/tt_metal/test_kernels/misc/cce_dram_buffer_round_trip.cpp",
            logical_dram_core,
            DramConfig{.noc = NOC::NOC_0});
        SetRuntimeArgs(
            program,
            kernel,
            logical_dram_core,
            {src_dram_offset,
             dst_dram_offset,
             target.mimir_index,
             staging_dev_addr,
             static_cast<uint32_t>(cleared.size()),
             target.num_slots,
             target.slot_stride,
             target.slot_base});

        detail::LaunchProgram(device, program, /*wait_until_cores_done=*/true, /*force_slow_dispatch=*/true);

        for (uint32_t slot = 0; slot < target.num_slots; ++slot) {
            std::vector<uint32_t> output;
            read_mimir_gddr(
                device,
                target.mimir_index,
                dst_dram_offset + static_cast<uint64_t>(target.slot_base + slot) * target.slot_stride,
                transfer_size,
                output);
            EXPECT_EQ(output, inputs[slot])
                << "mimir " << target.mimir_index << " location " << target.location << " slot " << slot;
        }
    }
    EXPECT_TRUE(CloseDevice(device));
}

TEST(MimirEmu, AllCcesCopyLocalMimirBlock) {
    if (!emu_server_configured()) {
        GTEST_SKIP() << "Set TT_METAL_EMU_SERVER=host:port and TT_METAL_EMU_SOC_DESC=<mimir YAML>.";
    }
    if (sival_bringup()) {
        GTEST_SKIP() << kSivalSramOnly;
    }

    IDevice* device = CreateDevice(0);
    ASSERT_NE(device, nullptr);
    ASSERT_EQ(device->arch(), tt::ARCH::QUASAR);
    const uint32_t num_cces = num_cces_on(device);
    ASSERT_GE(num_cces, 1u);
    const uint32_t num_mimirs = num_mimirs_on(device);
    ASSERT_GE(num_mimirs, 1u);
    const GddrSweep sweep = gddr_sweep_from_soc(device);
    ASSERT_NE(sweep.num_slots, 0u) << "dram_view_size must be a non-zero multiple of 1GiB";
    GddrBackingScope gddr_backing;

    const auto& hal = MetalContext::instance().hal();
    const uint32_t transfer_size = hal.get_alignment(HalMemType::DRAM);
    const uint32_t src_dram_offset = hal.get_dev_addr(HalDramMemAddrType::UNRESERVED);
    const uint32_t staging_dev_addr = hal.get_dev_addr(HalProgrammableCoreType::DRAM, HalL1MemAddrType::UNRESERVED);

    std::vector<std::vector<uint32_t>> inputs(num_cces, std::vector<uint32_t>(transfer_size / sizeof(uint32_t)));
    std::vector<uint32_t> cleared(transfer_size / sizeof(uint32_t), 0);
    Program program = CreateProgram();
    for (uint32_t cce_index = 0; cce_index < num_cces; ++cce_index) {
        const uint32_t mimir_index = mimir_index_for_cce(cce_index);
        for (std::size_t i = 0; i < inputs[cce_index].size(); ++i) {
            inputs[cce_index][i] = 0xB07C0000u | (cce_index << 8) | static_cast<uint32_t>(i);
        }
        const uint32_t cce_src = src_dram_offset + cce_index * 2 * transfer_size;
        const uint32_t cce_dst = cce_src + transfer_size;
        write_mimir_gddr(device, mimir_index, cce_src, inputs[cce_index]);
        write_mimir_gddr(device, mimir_index, cce_dst, cleared);

        const CoreCoord logical_dram_core = logical_cce_core(cce_index);
        const KernelHandle kernel = CreateKernel(
            program,
            "tests/tt_metal/tt_metal/test_kernels/misc/cce_dram_buffer_round_trip.cpp",
            logical_dram_core,
            DramConfig{.noc = NOC::NOC_0});
        SetRuntimeArgs(
            program,
            kernel,
            logical_dram_core,
            {cce_src,
             cce_dst,
             mimir_index,
             staging_dev_addr,
             static_cast<uint32_t>(cleared.size()),
             1u,
             sweep.slot_stride,
             0u});
    }

    detail::LaunchProgram(device, program, /*wait_until_cores_done=*/true, /*force_slow_dispatch=*/true);

    for (uint32_t cce_index = 0; cce_index < num_cces; ++cce_index) {
        std::vector<uint32_t> output;
        const uint32_t cce_dst = src_dram_offset + cce_index * 2 * transfer_size + transfer_size;
        read_mimir_gddr(device, mimir_index_for_cce(cce_index), cce_dst, transfer_size, output);
        EXPECT_EQ(output, inputs[cce_index]) << "cce " << cce_index;
    }
    EXPECT_TRUE(CloseDevice(device));
}

class WritesCrossMimirCceSram : public ::testing::TestWithParam<uint32_t> {};

TEST_P(WritesCrossMimirCceSram, WritesMagic) {
    if (!emu_server_configured()) {
        GTEST_SKIP() << "Set TT_METAL_EMU_SERVER=host:port and TT_METAL_EMU_SOC_DESC=<mimir YAML>.";
    }
    if (sival_bringup()) {
        GTEST_SKIP() << kSivalSramOnly;
    }

    IDevice* device = CreateDevice(0);
    ASSERT_NE(device, nullptr);
    ASSERT_EQ(device->arch(), tt::ARCH::QUASAR);

    const uint32_t src_cce = GetParam();
    const uint32_t num_cces = num_cces_on(device);
    if (src_cce >= num_cces || num_mimirs_on(device) < 2) {
        EXPECT_TRUE(CloseDevice(device));
        GTEST_SKIP() << "Configured descriptor does not expose a remote Mimir CCE.";
    }

    const uint32_t local_mimir = mimir_index_for_cce(src_cce);
    std::vector<uint32_t> dest_cces;
    for (uint32_t cce_index = 0; cce_index < num_cces; ++cce_index) {
        if (mimir_index_for_cce(cce_index) != local_mimir) {
            dest_cces.push_back(cce_index);
        }
    }
    if (dest_cces.size() < 2) {
        EXPECT_TRUE(CloseDevice(device));
        GTEST_SKIP() << "Remote Mimir does not expose two CCEs.";
    }

    const auto& hal = MetalContext::instance().hal();
    const uint32_t sram_offset = hal.get_dev_addr(HalProgrammableCoreType::DRAM, HalL1MemAddrType::UNRESERVED);
    const uint64_t noc_addr = hal.get_dev_noc_addr(HalProgrammableCoreType::DRAM, HalL1MemAddrType::UNRESERVED);
    const uint32_t magic_base = 0xC0CE0000u | (src_cce << 8);
    const uint32_t zero = 0;

    Cluster& cluster = MetalContext::instance().get_cluster();
    for (uint32_t dest_cce : dest_cces) {
        const CoreCoord dest_core = device->virtual_core_from_logical_core(logical_cce_core(dest_cce), CoreType::DRAM);
        cluster.write_core(&zero, sizeof(zero), {device->id(), dest_core}, noc_addr);
    }

    const CoreCoord logical_src = logical_cce_core(src_cce);
    Program program = CreateProgram();
    const KernelHandle kernel = CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/misc/cce_write_cross_mimir_sram.cpp",
        logical_src,
        DramConfig{.noc = NOC::NOC_0});
    SetRuntimeArgs(program, kernel, logical_src, {dest_cces[0], dest_cces[1], sram_offset, magic_base});

    detail::LaunchProgram(device, program, /*wait_until_cores_done=*/true, /*force_slow_dispatch=*/true);

    for (uint32_t dest_cce : dest_cces) {
        uint32_t result = 0;
        const CoreCoord dest_core = device->virtual_core_from_logical_core(logical_cce_core(dest_cce), CoreType::DRAM);
        cluster.read_core(&result, sizeof(result), {device->id(), dest_core}, noc_addr);
        EXPECT_EQ(result, magic_base | dest_cce) << "src cce " << src_cce << " dest cce " << dest_cce;
    }
    EXPECT_TRUE(CloseDevice(device));
}

class PrefetchesLocalGddrToRemoteCceSram : public ::testing::TestWithParam<uint32_t> {};

TEST_P(PrefetchesLocalGddrToRemoteCceSram, RoundTrip) {
    if (!emu_server_configured()) {
        GTEST_SKIP() << "Set TT_METAL_EMU_SERVER=host:port and TT_METAL_EMU_SOC_DESC=<mimir YAML>.";
    }
    if (sival_bringup()) {
        GTEST_SKIP() << kSivalSramOnly;
    }

    IDevice* device = CreateDevice(0);
    ASSERT_NE(device, nullptr);
    ASSERT_EQ(device->arch(), tt::ARCH::QUASAR);

    const uint32_t src_cce = GetParam();
    const uint32_t num_cces = num_cces_on(device);
    const uint32_t local_mimir = mimir_index_for_cce(src_cce);
    if (src_cce >= num_cces || num_mimirs_on(device) < 2) {
        EXPECT_TRUE(CloseDevice(device));
        GTEST_SKIP() << "Configured descriptor does not expose a remote Mimir CCE.";
    }

    std::vector<uint32_t> dest_cces;
    for (uint32_t cce_index = 0; cce_index < num_cces; ++cce_index) {
        if (mimir_index_for_cce(cce_index) != local_mimir) {
            dest_cces.push_back(cce_index);
        }
    }
    if (dest_cces.size() < 2) {
        EXPECT_TRUE(CloseDevice(device));
        GTEST_SKIP() << "Remote Mimir does not expose two CCEs.";
    }

    GddrBackingScope gddr_backing;
    const auto& hal = MetalContext::instance().hal();
    const uint32_t transfer_size = hal.get_alignment(HalMemType::DRAM);
    const uint32_t src_gddr_offset = hal.get_dev_addr(HalDramMemAddrType::UNRESERVED);
    const uint32_t staging_dev_addr = hal.get_dev_addr(HalProgrammableCoreType::DRAM, HalL1MemAddrType::UNRESERVED);
    const uint32_t sram_offset = staging_dev_addr;
    const uint64_t noc_addr = hal.get_dev_noc_addr(HalProgrammableCoreType::DRAM, HalL1MemAddrType::UNRESERVED);
    const uint32_t num_words = transfer_size / sizeof(uint32_t);

    std::vector<uint32_t> input(num_words);
    std::vector<uint32_t> cleared(num_words, 0);
    for (uint32_t i = 0; i < num_words; ++i) {
        input[i] = 0xD2D10000u | (src_cce << 8) | i;
    }
    write_mimir_gddr(device, local_mimir, src_gddr_offset, input);

    Cluster& cluster = MetalContext::instance().get_cluster();
    for (uint32_t dest_cce : dest_cces) {
        const CoreCoord dest_core = device->virtual_core_from_logical_core(logical_cce_core(dest_cce), CoreType::DRAM);
        cluster.write_core(cleared.data(), transfer_size, {device->id(), dest_core}, noc_addr);
    }

    const CoreCoord logical_src = logical_cce_core(src_cce);
    Program program = CreateProgram();
    const KernelHandle kernel = CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/misc/cce_prefetch_local_gddr_to_remote_sram.cpp",
        logical_src,
        DramConfig{.noc = NOC::NOC_0});
    SetRuntimeArgs(
        program,
        kernel,
        logical_src,
        {src_gddr_offset, local_mimir, staging_dev_addr, num_words, dest_cces[0], dest_cces[1], sram_offset});

    detail::LaunchProgram(device, program, /*wait_until_cores_done=*/true, /*force_slow_dispatch=*/true);

    for (uint32_t dest_cce : dest_cces) {
        std::vector<uint32_t> output(num_words, 0);
        const CoreCoord dest_core = device->virtual_core_from_logical_core(logical_cce_core(dest_cce), CoreType::DRAM);
        cluster.read_core(output.data(), transfer_size, {device->id(), dest_core}, noc_addr);
        EXPECT_EQ(output, input) << "src cce " << src_cce << " dest cce " << dest_cce;
    }
    EXPECT_TRUE(CloseDevice(device));
}

class AllHartsWriteAllocatedBuffer : public ::testing::TestWithParam<uint32_t> {};

TEST_P(AllHartsWriteAllocatedBuffer, WritesMagic) {
    if (!emu_server_configured()) {
        GTEST_SKIP() << "Set TT_METAL_EMU_SERVER=host:port and TT_METAL_EMU_SOC_DESC=<mimir YAML>.";
    }
    if (sival_bringup()) {
        GTEST_SKIP() << kSivalSramOnly;
    }

    IDevice* device = CreateDevice(0);
    ASSERT_NE(device, nullptr);
    ASSERT_EQ(device->arch(), tt::ARCH::QUASAR);

    const uint32_t cce_index = GetParam();
    const uint32_t num_mimirs = num_mimirs_on(device);
    if (cce_index >= num_cces_on(device) || num_mimirs == 0 || (is_cross_mimir_case() && num_mimirs < 2)) {
        EXPECT_TRUE(CloseDevice(device));
        GTEST_SKIP() << "Configured descriptor does not expose the requested CCE/Mimir.";
    }
    const GddrSweep sweep = gddr_sweep_from_soc(device);
    ASSERT_NE(sweep.num_slots, 0u) << "dram_view_size must be a non-zero multiple of 1GiB";
    const std::vector<GddrTarget> targets = gddr_targets_for(device, cce_index, sweep);
    ASSERT_FALSE(targets.empty()) << "No GDDR targets for this CCE/suite.";
    GddrBackingScope gddr_backing;

    const auto& hal = MetalContext::instance().hal();
    const uint32_t num_harts = hal.get_num_risc_processors(HalProgrammableCoreType::DRAM);
    ASSERT_EQ(num_harts, 8u);

    const uint32_t dst_dram_offset = hal.get_dev_addr(HalDramMemAddrType::UNRESERVED);
    const uint32_t staging_dev_base = hal.get_dev_addr(HalProgrammableCoreType::DRAM, HalL1MemAddrType::UNRESERVED);
    const CoreCoord logical_dram_core = logical_cce_core(cce_index);
    std::vector<uint32_t> cleared(num_harts, 0);

    for (const GddrTarget& target : targets) {
        for (uint32_t slot = 0; slot < target.num_slots; ++slot) {
            write_mimir_gddr(
                device,
                target.mimir_index,
                dst_dram_offset + static_cast<uint64_t>(target.slot_base + slot) * target.slot_stride,
                cleared);
        }

        Program program = CreateProgram();
        std::vector<uint32_t> expected(num_harts, 0);
        for (uint32_t hart = 0; hart < num_harts; hart++) {
            expected[hart] =
                0xA11B0000u | (cce_index << 12) | (target.mimir_index << 8) | (target.location << 4) | hart;
            CreateKernel(
                program,
                "tests/tt_metal/tt_metal/test_kernels/misc/cce_gddr_write_one_uint32.cpp",
                logical_dram_core,
                DramConfig{
                    .processor = static_cast<DataMovementProcessor>(hart),
                    .noc = NOC::NOC_0,
                    .compile_args = {
                        dst_dram_offset + hart * static_cast<uint32_t>(sizeof(uint32_t)),
                        target.mimir_index,
                        expected[hart],
                        staging_dev_base + hart * static_cast<uint32_t>(sizeof(uint32_t)),
                        target.num_slots,
                        target.slot_stride,
                        target.slot_base}});
        }

        detail::LaunchProgram(device, program, /*wait_until_cores_done=*/true, /*force_slow_dispatch=*/true);

        for (uint32_t slot = 0; slot < target.num_slots; ++slot) {
            std::vector<uint32_t> output;
            read_mimir_gddr(
                device,
                target.mimir_index,
                dst_dram_offset + static_cast<uint64_t>(target.slot_base + slot) * target.slot_stride,
                static_cast<uint32_t>(num_harts * sizeof(uint32_t)),
                output);
            EXPECT_EQ(output, expected) << "mimir " << target.mimir_index << " location " << target.location << " slot "
                                        << slot;
        }
    }
    EXPECT_TRUE(CloseDevice(device));
}

INSTANTIATE_TEST_SUITE_P(
    MimirEmu,
    CceSramRoundTripThroughMetalCluster,
    ::testing::Range(0u, 4u),
    [](const ::testing::TestParamInfo<uint32_t>& info) { return fmt::format("Cce{}", info.param); });

INSTANTIATE_TEST_SUITE_P(
    MimirEmu,
    CceSramRoundTripThroughMinimalDevice,
    ::testing::Range(0u, 4u),
    [](const ::testing::TestParamInfo<uint32_t>& info) { return fmt::format("Cce{}", info.param); });

INSTANTIATE_TEST_SUITE_P(
    MimirEmu, HartZeroRunsDramKernel, ::testing::Range(0u, 4u), [](const ::testing::TestParamInfo<uint32_t>& info) {
        return fmt::format("Cce{}", info.param);
    });

INSTANTIATE_TEST_SUITE_P(
    MimirEmu, AllHartsRunDramKernel, ::testing::Range(0u, 4u), [](const ::testing::TestParamInfo<uint32_t>& info) {
        return fmt::format("Cce{}", info.param);
    });

std::string cce_param_name(const ::testing::TestParamInfo<uint32_t>& info) { return fmt::format("Cce{}", info.param); }

INSTANTIATE_TEST_SUITE_P(MimirEmu, WritesCrossMimirCceSram, ::testing::Range(0u, 4u), cce_param_name);

INSTANTIATE_TEST_SUITE_P(MimirEmu, PrefetchesLocalGddrToRemoteCceSram, ::testing::Range(0u, 4u), cce_param_name);

INSTANTIATE_TEST_SUITE_P(
    MimirEmuSameMimirDirect, CopiesGddrWithCompileTimeArgs, ::testing::Range(0u, 4u), cce_param_name);

INSTANTIATE_TEST_SUITE_P(
    MimirEmuCrossMimirNoc, CopiesGddrWithCompileTimeArgs, ::testing::Range(0u, 4u), cce_param_name);

INSTANTIATE_TEST_SUITE_P(
    MimirEmuSameMimirDirect, CopiesGddrThroughAllocatedBuffer, ::testing::Range(0u, 4u), cce_param_name);

INSTANTIATE_TEST_SUITE_P(
    MimirEmuCrossMimirNoc, CopiesGddrThroughAllocatedBuffer, ::testing::Range(0u, 4u), cce_param_name);

INSTANTIATE_TEST_SUITE_P(
    MimirEmu,
    AllocatedDramBufferHostLoopback,
    ::testing::Range(0u, 4u),
    [](const ::testing::TestParamInfo<uint32_t>& info) { return fmt::format("Partition{}", info.param); });

INSTANTIATE_TEST_SUITE_P(
    MimirEmuSameMimirDirect, CopiesAllocatedBufferWithRuntimeArgs, ::testing::Range(0u, 4u), cce_param_name);

INSTANTIATE_TEST_SUITE_P(
    MimirEmuCrossMimirNoc, CopiesAllocatedBufferWithRuntimeArgs, ::testing::Range(0u, 4u), cce_param_name);

INSTANTIATE_TEST_SUITE_P(MimirEmu, CopiesEveryMimirBlock, ::testing::Range(0u, 4u), cce_param_name);

INSTANTIATE_TEST_SUITE_P(
    MimirEmuSameMimirDirect, CopiesAllocatedBufferLargerThanAlignment, ::testing::Range(0u, 4u), cce_param_name);

INSTANTIATE_TEST_SUITE_P(
    MimirEmuCrossMimirNoc, CopiesAllocatedBufferLargerThanAlignment, ::testing::Range(0u, 4u), cce_param_name);

INSTANTIATE_TEST_SUITE_P(
    MimirEmuSameMimirDirect, AllHartsWriteAllocatedBuffer, ::testing::Range(0u, 4u), cce_param_name);

INSTANTIATE_TEST_SUITE_P(MimirEmuCrossMimirNoc, AllHartsWriteAllocatedBuffer, ::testing::Range(0u, 4u), cce_param_name);

class DramChannelsDoNotAliasThroughPublicDeviceApi : public ::testing::TestWithParam<uint32_t> {};

TEST_P(DramChannelsDoNotAliasThroughPublicDeviceApi, RoundTrip) {
    if (!emu_server_configured()) {
        GTEST_SKIP() << "Set TT_METAL_EMU_SERVER=host:port and TT_METAL_EMU_SOC_DESC=<mimir YAML>.";
    }
    if (sival_bringup()) {
        GTEST_SKIP() << kSivalSramOnly;
    }

    std::unique_ptr<IDevice> device(CreateDeviceMinimal(0));
    ASSERT_NE(device, nullptr);

    const uint32_t dram_partition = GetParam();
    const uint32_t num_partitions = device->num_dram_channels();
    if (dram_partition >= num_partitions) {
        device.reset();
        GTEST_SKIP() << "Configured descriptor does not expose partition " << dram_partition << ".";
    }
    GddrBackingScope gddr_backing;
    const uint32_t other_partition = (dram_partition + num_partitions / 2) % num_partitions;
    gddr_backing.tracker.note(dram_partition, kDramOffset, sizeof(uint32_t));
    gddr_backing.tracker.note(other_partition, kDramOffset, sizeof(uint32_t));
    std::vector<uint32_t> written{0xAAAA1111 + dram_partition};
    std::vector<uint32_t> other_written{0xBBBB2222 + other_partition};
    ASSERT_TRUE(detail::WriteToDeviceDRAMChannel(device.get(), other_partition, kDramOffset, other_written));
    ASSERT_TRUE(detail::WriteToDeviceDRAMChannel(device.get(), dram_partition, kDramOffset, written));

    std::vector<uint32_t> read_back;
    std::vector<uint32_t> other_read_back;
    ASSERT_TRUE(
        detail::ReadFromDeviceDRAMChannel(device.get(), dram_partition, kDramOffset, sizeof(uint32_t), read_back));
    ASSERT_TRUE(detail::ReadFromDeviceDRAMChannel(
        device.get(), other_partition, kDramOffset, sizeof(uint32_t), other_read_back));

    EXPECT_EQ(read_back, written);
    EXPECT_EQ(other_read_back, other_written);
}

INSTANTIATE_TEST_SUITE_P(
    MimirEmu,
    DramChannelsDoNotAliasThroughPublicDeviceApi,
    ::testing::Range(0u, 4u),
    [](const ::testing::TestParamInfo<uint32_t>& info) { return fmt::format("Partition{}", info.param); });

}  // namespace
}  // namespace tt::tt_metal
