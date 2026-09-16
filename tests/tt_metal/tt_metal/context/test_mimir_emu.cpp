// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <array>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <thread>
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

void write_smc32(Cluster& cluster, const tt_cxy_pair& smc, uint64_t address, uint32_t value) {
    cluster.write_core(&value, sizeof(value), smc, address);
}

CoreCoord translated_dram_core(const metal_SocDescriptor& soc_desc, uint32_t channel) {
    const auto core = soc_desc.translate_coord_to(
        tt::umd::CoreCoord(channel, 0, CoreType::DRAM, CoordSystem::LOGICAL), CoordSystem::TRANSLATED);
    return {core.x, core.y};
}

std::vector<uint8_t> read_firmware(const char* path) {
    std::ifstream stream(path, std::ios::binary | std::ios::ate);
    if (!stream) {
        return {};
    }
    const std::streamsize size = stream.tellg();
    if (size <= 0) {
        return {};
    }
    stream.seekg(0);
    std::vector<uint8_t> firmware(static_cast<size_t>(size));
    if (!stream.read(reinterpret_cast<char*>(firmware.data()), size)) {
        return {};
    }
    firmware.resize((firmware.size() + sizeof(uint32_t) - 1) & ~(sizeof(uint32_t) - 1), 0);
    return firmware;
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

TEST(MimirEmu, Cce0ExecutesPrebuiltFirmware) {
    if (!emu_server_configured()) {
        GTEST_SKIP() << "Set TT_METAL_EMU_SERVER=host:port and TT_METAL_EMU_SOC_DESC=<mimir YAML>.";
    }
    const char* firmware_path = std::getenv("TT_METAL_EMU_CCE_FIRMWARE");
    if (firmware_path == nullptr) {
        GTEST_SKIP() << "Set TT_METAL_EMU_CCE_FIRMWARE to a Mimir CCE hello_cce or scratch_test .bin.";
    }

    const std::vector<uint8_t> firmware = read_firmware(firmware_path);
    ASSERT_FALSE(firmware.empty()) << "Could not read CCE firmware: " << firmware_path;
    ASSERT_LE(firmware.size(), 0x10000) << "CCE smoke-test firmware exceeds the 64 KiB preload region.";

    llrt::RunTimeOptions rtoptions;
    Cluster cluster(rtoptions);
    constexpr ChipId chip_id = 0;
    const auto smc_core = cluster.get_soc_desc(chip_id).get_cores(CoreType::SMC, CoordSystem::TRANSLATED).front();
    const tt_cxy_pair smc(chip_id, smc_core.x, smc_core.y);

    const CoreCoord cce0 = translated_dram_core(cluster.get_soc_desc(chip_id), 0);
    const tt_cxy_pair cce(chip_id, cce0);
    constexpr uint64_t kFirmwareAddress = kCceL1NocOffset;
    std::vector<uint8_t> zeroes(0x10000, 0);
    cluster.write_core(zeroes.data(), static_cast<uint32_t>(zeroes.size()), cce, kFirmwareAddress);
    cluster.write_core(firmware.data(), static_cast<uint32_t>(firmware.size()), cce, kFirmwareAddress);
    constexpr std::array<uint64_t, 3> kSyncOffsets{0x3F000, 0x3F008, 0x3F010};
    for (uint64_t offset : kSyncOffsets) {
        const uint64_t zero = 0;
        cluster.write_core(&zero, sizeof(zero), cce, kCceL1NocOffset + offset);
    }

    constexpr uint64_t kCce0Postcode = 0x02000040;
    constexpr uint32_t kPass = 0xACAFACA1;
    constexpr uint32_t kFail = 0xDEADBEEF;
    uint32_t postcode = 0;
    write_smc32(cluster, smc, kCce0Postcode, postcode);

    // PF_CTRL_RESET bit 0 is the uncore, bits 1-8 are harts 0-7. Release only the boot hart
    // plus the already-running uncore: the Mimir CCE linker script sets __boot_hart = 1, and
    // only the boot hart reaches main()/WRITE_TEST_PASS(). Non-boot harts run secondary_main().
    write_smc64(cluster, smc, 0x02200000, 0x5);
    for (uint32_t poll = 0; poll < 500 && postcode != kPass && postcode != kFail; ++poll) {
        cluster.read_core(&postcode, sizeof(postcode), smc, kCce0Postcode);
        if (postcode != kPass && postcode != kFail) {
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
        }
    }

    // Leave the CCE RISC-V cores in reset even when the firmware reports failure.
    write_smc64(cluster, smc, 0x02200000, 0x1);
    ASSERT_NE(postcode, kFail) << "CCE0 firmware reported failure.";
    EXPECT_EQ(postcode, kPass) << "CCE0 firmware did not report PASS before the 10-second timeout.";
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
