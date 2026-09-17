// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <cstdint>
#include <cstdlib>
#include <memory>
#include <vector>

#include "impl/context/metal_context.hpp"
#include "llrt/rtoptions.hpp"
#include "llrt/tt_cluster.hpp"
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>
#include "umd/device/types/arch.hpp"
#include "umd/device/types/core_coordinates.hpp"

namespace tt::tt_metal {
namespace {

constexpr uint64_t kCceL1NocOffset = 0x2000000000ULL;
constexpr uint64_t kCceSramTestOffset = 0x800;
constexpr uint32_t kDramOffset = 0x800;

bool emu_server_configured() {
    return std::getenv("TT_METAL_EMU_SERVER") != nullptr && std::getenv("TT_METAL_EMU_SOC_DESC") != nullptr;
}

CoreCoord translated_dram_core(const metal_SocDescriptor& soc_desc, uint32_t channel) {
    const auto core = soc_desc.translate_coord_to(
        tt::umd::CoreCoord(channel, 0, CoreType::DRAM, CoordSystem::LOGICAL), CoordSystem::TRANSLATED);
    return {core.x, core.y};
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
    const CoreCoord cce0 = translated_dram_core(soc_desc, 0);
    const CoreCoord cce1 = translated_dram_core(soc_desc, 1);
    constexpr uint64_t address = kCceL1NocOffset + kCceSramTestOffset;
    constexpr uint32_t first = 0xC0FFEE01;
    constexpr uint32_t second = 0xC0FFEE02;
    uint32_t read_first = 0;
    uint32_t read_second = 0;

    cluster.write_core(&first, sizeof(first), {chip_id, cce0}, address);
    cluster.write_core(&second, sizeof(second), {chip_id, cce1}, address);
    cluster.read_core(&read_first, sizeof(read_first), {chip_id, cce0}, address);
    cluster.read_core(&read_second, sizeof(read_second), {chip_id, cce1}, address);

    EXPECT_EQ(read_first, first);
    EXPECT_EQ(read_second, second);
}

TEST(MimirEmu, CceSramRoundTripThroughMinimalDevice) {
    if (!emu_server_configured()) {
        GTEST_SKIP() << "Set TT_METAL_EMU_SERVER=host:port and TT_METAL_EMU_SOC_DESC=<mimir YAML>.";
    }

    std::unique_ptr<IDevice> device(CreateDeviceMinimal(0));
    ASSERT_NE(device, nullptr);
    ASSERT_EQ(device->arch(), tt::ARCH::QUASAR);

    Cluster& cluster = MetalContext::instance().get_cluster();
    const CoreCoord cce = device->virtual_core_from_logical_core({0, 0}, CoreType::DRAM);
    constexpr uint64_t address = kCceL1NocOffset + kCceSramTestOffset;
    constexpr uint32_t written = 0xC0FFEE03;
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
    EXPECT_EQ(device->arch(), tt::ARCH::QUASAR);
    EXPECT_TRUE(CloseDevice(device));
}

// DRAM is opt-in. test_emu_server.py runs SMC boot so GDDR has slaves; test_sival_server.py
// does not, and an unconfigured access stalls the AXI master. Same gate as UMD's
// EmuTTDevice DRAM test: TT_METAL_EMU_DRAM or TT_UMD_EMU_DRAM.
TEST(MimirEmu, DramChannelsDoNotAliasThroughPublicDeviceApi) {
    if (!emu_server_configured()) {
        GTEST_SKIP() << "Set TT_METAL_EMU_SERVER=host:port and TT_METAL_EMU_SOC_DESC=<mimir YAML>.";
    }
    if (std::getenv("TT_METAL_EMU_DRAM") == nullptr && std::getenv("TT_UMD_EMU_DRAM") == nullptr) {
        GTEST_SKIP() << "DRAM needs GDDR bringup; set TT_METAL_EMU_DRAM=1 on a configured model.";
    }

    std::unique_ptr<IDevice> device(CreateDeviceMinimal(0));
    ASSERT_NE(device, nullptr);
    ASSERT_EQ(device->num_dram_channels(), 2);

    std::vector<uint32_t> first{0xAAAA1111};
    std::vector<uint32_t> second{0xBBBB2222};
    ASSERT_TRUE(detail::WriteToDeviceDRAMChannel(device.get(), 0, kDramOffset, first));
    ASSERT_TRUE(detail::WriteToDeviceDRAMChannel(device.get(), 1, kDramOffset, second));

    std::vector<uint32_t> read_first;
    std::vector<uint32_t> read_second;
    ASSERT_TRUE(detail::ReadFromDeviceDRAMChannel(device.get(), 0, kDramOffset, sizeof(uint32_t), read_first));
    ASSERT_TRUE(detail::ReadFromDeviceDRAMChannel(device.get(), 1, kDramOffset, sizeof(uint32_t), read_second));

    EXPECT_EQ(read_first, first);
    EXPECT_EQ(read_second, second);
}

}  // namespace
}  // namespace tt::tt_metal
