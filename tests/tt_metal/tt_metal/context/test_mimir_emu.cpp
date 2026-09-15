// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <array>
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

void write_smc64(Cluster& cluster, const tt_cxy_pair& smc, uint64_t address, uint64_t value) {
    cluster.write_core(&value, sizeof(value), smc, address);
}

uint64_t read_smc64(Cluster& cluster, const tt_cxy_pair& smc, uint64_t address) {
    uint64_t value = 0;
    cluster.read_core(&value, sizeof(value), smc, address);
    return value;
}

void configure_cce_sram(Cluster& cluster, ChipId chip_id) {
    const auto smc_cores = cluster.get_soc_desc(chip_id).get_cores(CoreType::SMC, CoordSystem::TRANSLATED);
    ASSERT_EQ(smc_cores.size(), 1);
    const tt_cxy_pair smc(chip_id, smc_cores.front().x, smc_cores.front().y);

    constexpr uint64_t kColdReset = 0x2040;
    constexpr uint64_t kWarmReset = 0x2044;
    constexpr uint64_t kCce0ResetBits = (1ULL << 8) | (1ULL << 9) | (1ULL << 10);
    constexpr uint64_t kCce1ResetBits = (1ULL << 12) | (1ULL << 13) | (1ULL << 14);
    constexpr uint64_t kInfrastructureResetBits = (1ULL << 1) | (1ULL << 2);
    constexpr uint64_t kCceResetBits = kCce0ResetBits | kCce1ResetBits;

    write_smc64(
        cluster, smc, kColdReset, read_smc64(cluster, smc, kColdReset) | kInfrastructureResetBits | kCceResetBits);
    write_smc64(cluster, smc, kWarmReset, read_smc64(cluster, smc, kWarmReset) | kCceResetBits);

    // Open the non-secure and secure catch-all entries for the memory, CCE, and CCE-config
    // interconnect tiles, matching Mimir's run_cce_via_sival.py bring-up.
    constexpr uint64_t kFirewallNonSecure =
        (1ULL << 0) | (1ULL << 1) | (1ULL << 4) | (1ULL << 8) | (7ULL << 12) | (1ULL << 24);
    constexpr uint64_t kFirewallSecure = (1ULL << 0) | (1ULL << 1) | (1ULL << 4) | (7ULL << 12) | (1ULL << 24);
    constexpr uint64_t kFirewallEnd = 0xFFFFFFFFFFFULL;
    constexpr std::array<std::pair<uint64_t, uint64_t>, 6> kFirewallEntries{{
        {0x04000000, 0x040001E0},
        {0x04000800, 0x040009E0},
        {0x04001000, 0x040011E0},
        {0x04001800, 0x040019E0},
        {0x04002000, 0x040021E0},
        {0x04002800, 0x040029E0},
    }};
    for (const auto& [non_secure, secure] : kFirewallEntries) {
        for (const auto& [base, config] : std::array<std::pair<uint64_t, uint64_t>, 2>{
                 {{non_secure, kFirewallNonSecure}, {secure, kFirewallSecure}}}) {
            write_smc64(cluster, smc, base, config);
            write_smc64(cluster, smc, base + 0x08, 0);
            write_smc64(cluster, smc, base + 0x10, kFirewallEnd);
        }
    }

    // Release each CCE uncore. The eight RISC-V cores remain in reset; SRAM is now host-accessible.
    write_smc64(cluster, smc, 0x02200000, 0x1);
    write_smc64(cluster, smc, 0x03200000, 0x1);
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
    configure_cce_sram(cluster, chip_id);

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
    configure_cce_sram(cluster, device->id());
    const CoreCoord cce = device->virtual_core_from_logical_core({0, 0}, CoreType::DRAM);
    constexpr uint64_t address = kCceL1NocOffset + kCceSramTestOffset;
    constexpr uint32_t written = 0xC0FFEE03;
    uint32_t read_back = 0;
    cluster.write_core(&written, sizeof(written), {device->id(), cce}, address);
    cluster.read_core(&read_back, sizeof(read_back), {device->id(), cce}, address);
    EXPECT_EQ(read_back, written);
}

// DRAM is opt-in. The SiVal server skips GDDR bringup unless the model has been configured;
// an unconfigured access stalls the AXI master and wedges the session. Same gate as UMD's
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
