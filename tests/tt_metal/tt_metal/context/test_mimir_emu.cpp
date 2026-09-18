// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <fmt/format.h>

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <tuple>
#include <vector>

#include "impl/context/metal_context.hpp"
#include "impl/kernels/kernel.hpp"
#include "llrt/hal.hpp"
#include "llrt/rtoptions.hpp"
#include "llrt/tt_cluster.hpp"
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>
#include "umd/device/types/arch.hpp"
#include "umd/device/types/core_coordinates.hpp"

namespace tt::tt_metal {
namespace {

constexpr uint64_t kCceL1NocOffset = 0x2000000000ULL;
// The hart caches its SRAM but the NOC does not snoop that cache, so a kernel store only becomes
// host-visible through the uncached alias one SRAM size above the cached window
// (MEM_CCE_L1_BASE + MEM_CCE_L1_SIZE). Host/NOC access stays 0-based.
constexpr uint32_t kCceSramUncachedBase = 0x400000;
constexpr uint64_t kCceSramTestOffset = 0x800;
constexpr uint32_t kDramOffset = 0x800;
// The SPA window a CCE hart issues into to reach GDDR. Its remapper translates this to the GDDR
// physical base (0x800000000), which is where the host's AXI view sees the same memory -- issuing
// that physical address from a hart instead matches no remap entry and quietly reads zeros.
// One 16 GiB window covers both 8 GiB partitions (chippy tile0/tile1).
constexpr uint64_t kGddrSpaWindowBase = 0x1000000000000ULL;
constexpr uint64_t kGddrSpaPartitionStride = 0x200000000ULL;

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

class HartZeroRunsDramKernel : public ::testing::TestWithParam<uint32_t> {};

TEST_P(HartZeroRunsDramKernel, WritesMagic) {
    if (!emu_server_configured()) {
        GTEST_SKIP() << "Set TT_METAL_EMU_SERVER=host:port and TT_METAL_EMU_SOC_DESC=<mimir YAML>.";
    }

    IDevice* device = CreateDevice(0);
    ASSERT_NE(device, nullptr);
    ASSERT_EQ(device->arch(), tt::ARCH::QUASAR);

    const uint32_t cce_index = GetParam();
    ASSERT_LT(cce_index, device->num_dram_channels());

    const CoreCoord logical_dram_core{cce_index, 0};
    const uint32_t magic = 0xC0FFEE04 + cce_index;
    const auto& hal = MetalContext::instance().hal();
    // Same word, two addresses: the kernel stores through the uncached alias so the NOC can see it,
    // while the host reaches it 0-based through the DRAM core plus the L1 NOC tag. Both CCEs share
    // this hart-local map; the host coordinate selects which SRAM the NOC tag lands in.
    const uint32_t result_dev_addr =
        kCceSramUncachedBase + hal.get_dev_addr(HalProgrammableCoreType::DRAM, HalL1MemAddrType::UNRESERVED);
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

class CopiesGddrThroughRemapper : public ::testing::TestWithParam<std::tuple<uint32_t, uint32_t>> {};

TEST_P(CopiesGddrThroughRemapper, RoundTrip) {
    if (!emu_server_configured()) {
        GTEST_SKIP() << "Set TT_METAL_EMU_SERVER=host:port and TT_METAL_EMU_SOC_DESC=<mimir YAML>.";
    }

    IDevice* device = CreateDevice(0);
    ASSERT_NE(device, nullptr);
    ASSERT_EQ(device->arch(), tt::ARCH::QUASAR);

    const auto [cce_index, dram_partition] = GetParam();
    ASSERT_LT(cce_index, device->num_dram_channels());
    ASSERT_LT(dram_partition, device->num_dram_channels());

    const auto& hal = MetalContext::instance().hal();
    const uint32_t transfer_size = hal.get_alignment(HalMemType::DRAM);
    const uint32_t src_dram_offset = hal.get_dev_addr(HalDramMemAddrType::UNRESERVED);
    const uint32_t dst_dram_offset = src_dram_offset + transfer_size;
    const uint64_t partition_base = kGddrSpaWindowBase + dram_partition * kGddrSpaPartitionStride;
    const uint64_t src_gddr_addr = partition_base + src_dram_offset;
    const uint64_t dst_gddr_addr = partition_base + dst_dram_offset;
    // The kernel stages through the uncached alias so the host can read the halfway point back.
    const uint32_t staging_dev_addr =
        kCceSramUncachedBase + hal.get_dev_addr(HalProgrammableCoreType::DRAM, HalL1MemAddrType::UNRESERVED);
    const uint64_t staging_noc_addr = hal.get_dev_noc_addr(HalProgrammableCoreType::DRAM, HalL1MemAddrType::UNRESERVED);

    std::vector<uint32_t> input(transfer_size / sizeof(uint32_t));
    for (std::size_t i = 0; i < input.size(); ++i) {
        input[i] = 0xC0FF0000u | (cce_index << 12) | (dram_partition << 8) | static_cast<uint32_t>(i);
    }
    std::vector<uint32_t> cleared(input.size(), 0);

    ASSERT_TRUE(detail::WriteToDeviceDRAMChannel(device, dram_partition, src_dram_offset, input));
    ASSERT_TRUE(detail::WriteToDeviceDRAMChannel(device, dram_partition, dst_dram_offset, cleared));

    const CoreCoord logical_dram_core{cce_index, 0};
    Program program = CreateProgram();
    CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/misc/dram_gddr_round_trip.cpp",
        logical_dram_core,
        DramConfig{
            .noc = NOC::NOC_0,
            .compile_args = {
                static_cast<uint32_t>(src_gddr_addr),
                static_cast<uint32_t>(src_gddr_addr >> 32),
                static_cast<uint32_t>(dst_gddr_addr),
                static_cast<uint32_t>(dst_gddr_addr >> 32),
                staging_dev_addr,
                static_cast<uint32_t>(input.size())}});

    detail::LaunchProgram(device, program, /*wait_until_cores_done=*/true, /*force_slow_dispatch=*/true);

    const CoreCoord virtual_dram_core = device->virtual_core_from_logical_core(logical_dram_core, CoreType::DRAM);
    std::vector<uint32_t> staged(input.size(), 0);
    MetalContext::instance().get_cluster().read_core(
        staged.data(), transfer_size, {device->id(), virtual_dram_core}, staging_noc_addr);
    EXPECT_EQ(staged, input);

    std::vector<uint32_t> output;
    ASSERT_TRUE(detail::ReadFromDeviceDRAMChannel(device, dram_partition, dst_dram_offset, transfer_size, output));
    EXPECT_EQ(output, input);
    EXPECT_TRUE(CloseDevice(device));
}

class CopiesGddrThroughAllocatedBuffer : public ::testing::TestWithParam<std::tuple<uint32_t, uint32_t>> {};

TEST_P(CopiesGddrThroughAllocatedBuffer, RoundTrip) {
    if (!emu_server_configured()) {
        GTEST_SKIP() << "Set TT_METAL_EMU_SERVER=host:port and TT_METAL_EMU_SOC_DESC=<mimir YAML>.";
    }

    IDevice* device = CreateDevice(0);
    ASSERT_NE(device, nullptr);
    ASSERT_EQ(device->arch(), tt::ARCH::QUASAR);

    const auto [cce_index, dram_partition] = GetParam();
    ASSERT_LT(cce_index, device->num_dram_channels());
    ASSERT_LT(dram_partition, device->num_dram_channels());

    const auto& hal = MetalContext::instance().hal();
    const uint32_t transfer_size = hal.get_alignment(HalMemType::DRAM);
    const uint32_t buffer_size = transfer_size * device->num_dram_channels();
    auto src_buffer = CreateBuffer(BufferConfig{device, buffer_size, transfer_size, BufferType::DRAM});
    auto dst_buffer = CreateBuffer(BufferConfig{device, buffer_size, transfer_size, BufferType::DRAM});
    ASSERT_NE(src_buffer, nullptr);
    ASSERT_NE(dst_buffer, nullptr);

    const uint64_t partition_base = kGddrSpaWindowBase + dram_partition * kGddrSpaPartitionStride;
    const uint64_t src_gddr_addr = partition_base + src_buffer->address();
    const uint64_t dst_gddr_addr = partition_base + dst_buffer->address();
    const uint32_t staging_dev_addr =
        kCceSramUncachedBase + hal.get_dev_addr(HalProgrammableCoreType::DRAM, HalL1MemAddrType::UNRESERVED);
    const uint64_t staging_noc_addr = hal.get_dev_noc_addr(HalProgrammableCoreType::DRAM, HalL1MemAddrType::UNRESERVED);

    const std::size_t words_per_page = transfer_size / sizeof(uint32_t);
    std::vector<uint32_t> input(buffer_size / sizeof(uint32_t));
    for (uint32_t partition = 0; partition < device->num_dram_channels(); ++partition) {
        for (std::size_t i = 0; i < words_per_page; ++i) {
            input[partition * words_per_page + i] =
                0xC0FF0000u | (cce_index << 12) | (partition << 8) | static_cast<uint32_t>(i);
        }
    }
    std::vector<uint32_t> cleared(input.size(), 0);

    detail::WriteToBuffer(*src_buffer, input);
    detail::WriteToBuffer(*dst_buffer, cleared);

    const CoreCoord logical_dram_core{cce_index, 0};
    Program program = CreateProgram();
    CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/misc/dram_gddr_round_trip.cpp",
        logical_dram_core,
        DramConfig{
            .noc = NOC::NOC_0,
            .compile_args = {
                static_cast<uint32_t>(src_gddr_addr),
                static_cast<uint32_t>(src_gddr_addr >> 32),
                static_cast<uint32_t>(dst_gddr_addr),
                static_cast<uint32_t>(dst_gddr_addr >> 32),
                staging_dev_addr,
                static_cast<uint32_t>(words_per_page)}});

    detail::LaunchProgram(device, program, /*wait_until_cores_done=*/true, /*force_slow_dispatch=*/true);

    const CoreCoord virtual_dram_core = device->virtual_core_from_logical_core(logical_dram_core, CoreType::DRAM);
    const auto expected_begin = input.begin() + dram_partition * words_per_page;
    const std::vector<uint32_t> expected(expected_begin, expected_begin + words_per_page);
    std::vector<uint32_t> staged(words_per_page, 0);
    MetalContext::instance().get_cluster().read_core(
        staged.data(), transfer_size, {device->id(), virtual_dram_core}, staging_noc_addr);
    EXPECT_EQ(staged, expected);

    std::vector<uint32_t> output;
    detail::ReadFromBuffer(*dst_buffer, output);
    ASSERT_EQ(output.size(), cleared.size());
    const std::vector<uint32_t> zero_page(words_per_page, 0);
    for (uint32_t partition = 0; partition < device->num_dram_channels(); ++partition) {
        const auto output_begin = output.begin() + partition * words_per_page;
        const std::vector<uint32_t> output_page(output_begin, output_begin + words_per_page);
        EXPECT_EQ(output_page, partition == dram_partition ? expected : zero_page);
    }
    src_buffer.reset();
    dst_buffer.reset();
    EXPECT_TRUE(CloseDevice(device));
}

INSTANTIATE_TEST_SUITE_P(
    MimirEmu, HartZeroRunsDramKernel, ::testing::Values(0u, 1u), [](const ::testing::TestParamInfo<uint32_t>& info) {
        return fmt::format("Cce{}", info.param);
    });

INSTANTIATE_TEST_SUITE_P(
    MimirEmu,
    CopiesGddrThroughRemapper,
    ::testing::Combine(::testing::Values(0u, 1u), ::testing::Values(0u, 1u)),
    [](const ::testing::TestParamInfo<std::tuple<uint32_t, uint32_t>>& info) {
        return fmt::format("Cce{}_Partition{}", std::get<0>(info.param), std::get<1>(info.param));
    });

INSTANTIATE_TEST_SUITE_P(
    MimirEmu,
    CopiesGddrThroughAllocatedBuffer,
    ::testing::Combine(::testing::Values(0u, 1u), ::testing::Values(0u, 1u)),
    [](const ::testing::TestParamInfo<std::tuple<uint32_t, uint32_t>>& info) {
        return fmt::format("Cce{}_Partition{}", std::get<0>(info.param), std::get<1>(info.param));
    });

TEST(MimirEmu, DramChannelsDoNotAliasThroughPublicDeviceApi) {
    if (!emu_server_configured()) {
        GTEST_SKIP() << "Set TT_METAL_EMU_SERVER=host:port and TT_METAL_EMU_SOC_DESC=<mimir YAML>.";
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
