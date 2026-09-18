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

class CceSramChannelsDoNotAliasThroughMetalCluster : public ::testing::TestWithParam<uint32_t> {};

TEST_P(CceSramChannelsDoNotAliasThroughMetalCluster, RoundTrip) {
    if (!emu_server_configured()) {
        GTEST_SKIP() << "Set TT_METAL_EMU_SERVER=host:port and TT_METAL_EMU_SOC_DESC=<mimir YAML>.";
    }

    llrt::RunTimeOptions rtoptions;
    Cluster cluster(rtoptions);

    ASSERT_EQ(cluster.arch(), tt::ARCH::QUASAR);
    ASSERT_EQ(cluster.all_chip_ids().size(), 1);

    constexpr ChipId chip_id = 0;
    const uint32_t cce_index = GetParam();
    const uint32_t other_cce_index = 1u - cce_index;
    const auto& soc_desc = cluster.get_soc_desc(chip_id);
    const CoreCoord cce = translated_dram_core(soc_desc, cce_index);
    const CoreCoord other_cce = translated_dram_core(soc_desc, other_cce_index);
    constexpr uint64_t address = MEM_CCE_L1_NOC_OFFSET + kCceSramTestOffset;
    const uint32_t written = 0xC0FFEE01 + cce_index;
    const uint32_t other_written = 0xC0FFEE01 + other_cce_index;
    uint32_t read_back = 0;
    uint32_t other_read_back = 0;

    cluster.write_core(&other_written, sizeof(other_written), {chip_id, other_cce}, address);
    cluster.write_core(&written, sizeof(written), {chip_id, cce}, address);
    cluster.read_core(&read_back, sizeof(read_back), {chip_id, cce}, address);
    cluster.read_core(&other_read_back, sizeof(other_read_back), {chip_id, other_cce}, address);

    EXPECT_EQ(read_back, written);
    EXPECT_EQ(other_read_back, other_written);
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
    ASSERT_LT(cce_index, device->num_dram_channels());

    Cluster& cluster = MetalContext::instance().get_cluster();
    const CoreCoord cce = device->virtual_core_from_logical_core({cce_index, 0}, CoreType::DRAM);
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
    ASSERT_LT(cce_index, device->num_dram_channels());

    const CoreCoord logical_dram_core{cce_index, 0};
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
    const uint32_t staging_dev_addr = hal.get_dev_addr(HalProgrammableCoreType::DRAM, HalL1MemAddrType::UNRESERVED);
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
    const uint32_t staging_dev_addr = hal.get_dev_addr(HalProgrammableCoreType::DRAM, HalL1MemAddrType::UNRESERVED);
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

class CopiesAllocatedBufferWithRuntimeArgs : public ::testing::TestWithParam<std::tuple<uint32_t, uint32_t>> {};

TEST_P(CopiesAllocatedBufferWithRuntimeArgs, RoundTrip) {
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

    const std::size_t words_per_page = transfer_size / sizeof(uint32_t);
    std::vector<uint32_t> input(buffer_size / sizeof(uint32_t));
    for (uint32_t partition = 0; partition < device->num_dram_channels(); ++partition) {
        for (std::size_t i = 0; i < words_per_page; ++i) {
            input[partition * words_per_page + i] =
                0xD15C0000u | (cce_index << 12) | (partition << 8) | static_cast<uint32_t>(i);
        }
    }
    std::vector<uint32_t> cleared(input.size(), 0);
    detail::WriteToBuffer(*src_buffer, input);
    detail::WriteToBuffer(*dst_buffer, cleared);

    const uint32_t staging_dev_addr = hal.get_dev_addr(HalProgrammableCoreType::DRAM, HalL1MemAddrType::UNRESERVED);
    const CoreCoord logical_dram_core{cce_index, 0};
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
        {src_buffer->address(),
         dst_buffer->address(),
         dram_partition,
         staging_dev_addr,
         static_cast<uint32_t>(words_per_page)});

    detail::LaunchProgram(device, program, /*wait_until_cores_done=*/true, /*force_slow_dispatch=*/true);

    std::vector<uint32_t> output;
    detail::ReadFromBuffer(*dst_buffer, output);
    ASSERT_EQ(output.size(), cleared.size());
    const auto expected_begin = input.begin() + dram_partition * words_per_page;
    const std::vector<uint32_t> expected(expected_begin, expected_begin + words_per_page);
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

class AllocatedDramBufferHostLoopback : public ::testing::TestWithParam<uint32_t> {};

TEST_P(AllocatedDramBufferHostLoopback, RoundTrip) {
    if (!emu_server_configured()) {
        GTEST_SKIP() << "Set TT_METAL_EMU_SERVER=host:port and TT_METAL_EMU_SOC_DESC=<mimir YAML>.";
    }

    IDevice* device = CreateDevice(0);
    ASSERT_NE(device, nullptr);
    ASSERT_EQ(device->arch(), tt::ARCH::QUASAR);

    const uint32_t dram_partition = GetParam();
    ASSERT_LT(dram_partition, device->num_dram_channels());

    const auto& hal = MetalContext::instance().hal();
    const uint32_t page_size = hal.get_alignment(HalMemType::DRAM);
    const uint32_t buffer_size = page_size * device->num_dram_channels();
    auto buffer = CreateBuffer(BufferConfig{device, buffer_size, page_size, BufferType::DRAM});
    ASSERT_NE(buffer, nullptr);

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

class CopiesEveryPartitionOfAllocatedBuffer : public ::testing::TestWithParam<uint32_t> {};

TEST_P(CopiesEveryPartitionOfAllocatedBuffer, RoundTrip) {
    if (!emu_server_configured()) {
        GTEST_SKIP() << "Set TT_METAL_EMU_SERVER=host:port and TT_METAL_EMU_SOC_DESC=<mimir YAML>.";
    }

    IDevice* device = CreateDevice(0);
    ASSERT_NE(device, nullptr);
    ASSERT_EQ(device->arch(), tt::ARCH::QUASAR);

    const uint32_t cce_index = GetParam();
    ASSERT_LT(cce_index, device->num_dram_channels());
    const uint32_t num_partitions = device->num_dram_channels();

    const auto& hal = MetalContext::instance().hal();
    const uint32_t page_size = hal.get_alignment(HalMemType::DRAM);
    const uint32_t buffer_size = page_size * num_partitions;
    auto src_buffer = CreateBuffer(BufferConfig{device, buffer_size, page_size, BufferType::DRAM});
    auto dst_buffer = CreateBuffer(BufferConfig{device, buffer_size, page_size, BufferType::DRAM});
    ASSERT_NE(src_buffer, nullptr);
    ASSERT_NE(dst_buffer, nullptr);

    const std::size_t words_per_page = page_size / sizeof(uint32_t);
    std::vector<uint32_t> input(buffer_size / sizeof(uint32_t));
    for (uint32_t partition = 0; partition < num_partitions; ++partition) {
        for (std::size_t i = 0; i < words_per_page; ++i) {
            input[partition * words_per_page + i] =
                0xE1E10000u | (cce_index << 12) | (partition << 8) | static_cast<uint32_t>(i);
        }
    }
    std::vector<uint32_t> cleared(input.size(), 0);
    detail::WriteToBuffer(*src_buffer, input);
    detail::WriteToBuffer(*dst_buffer, cleared);

    const uint32_t staging_dev_addr = hal.get_dev_addr(HalProgrammableCoreType::DRAM, HalL1MemAddrType::UNRESERVED);
    const CoreCoord logical_dram_core{cce_index, 0};
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
        {src_buffer->address(),
         dst_buffer->address(),
         num_partitions,
         staging_dev_addr,
         static_cast<uint32_t>(words_per_page)});

    detail::LaunchProgram(device, program, /*wait_until_cores_done=*/true, /*force_slow_dispatch=*/true);

    std::vector<uint32_t> output;
    detail::ReadFromBuffer(*dst_buffer, output);
    EXPECT_EQ(output, input);

    src_buffer.reset();
    dst_buffer.reset();
    EXPECT_TRUE(CloseDevice(device));
}

class CopiesAllocatedBufferLargerThanAlignment : public ::testing::TestWithParam<std::tuple<uint32_t, uint32_t>> {};

TEST_P(CopiesAllocatedBufferLargerThanAlignment, RoundTrip) {
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
    const uint32_t alignment = hal.get_alignment(HalMemType::DRAM);
    constexpr uint32_t kPagesPerBank = 16;
    const uint32_t transfer_size = alignment * kPagesPerBank;
    const uint32_t buffer_size = transfer_size * device->num_dram_channels();
    auto src_buffer = CreateBuffer(BufferConfig{device, buffer_size, transfer_size, BufferType::DRAM});
    auto dst_buffer = CreateBuffer(BufferConfig{device, buffer_size, transfer_size, BufferType::DRAM});
    ASSERT_NE(src_buffer, nullptr);
    ASSERT_NE(dst_buffer, nullptr);

    const std::size_t words_per_page = transfer_size / sizeof(uint32_t);
    std::vector<uint32_t> input(buffer_size / sizeof(uint32_t));
    for (uint32_t partition = 0; partition < device->num_dram_channels(); ++partition) {
        for (std::size_t i = 0; i < words_per_page; ++i) {
            input[partition * words_per_page + i] =
                0x1A6E0000u | (cce_index << 12) | (partition << 8) | static_cast<uint32_t>(i);
        }
    }
    std::vector<uint32_t> cleared(input.size(), 0);
    detail::WriteToBuffer(*src_buffer, input);
    detail::WriteToBuffer(*dst_buffer, cleared);

    const uint32_t staging_dev_addr = hal.get_dev_addr(HalProgrammableCoreType::DRAM, HalL1MemAddrType::UNRESERVED);
    const CoreCoord logical_dram_core{cce_index, 0};
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
        {src_buffer->address(),
         dst_buffer->address(),
         dram_partition,
         staging_dev_addr,
         static_cast<uint32_t>(words_per_page)});

    detail::LaunchProgram(device, program, /*wait_until_cores_done=*/true, /*force_slow_dispatch=*/true);

    std::vector<uint32_t> output;
    detail::ReadFromBuffer(*dst_buffer, output);
    ASSERT_EQ(output.size(), cleared.size());
    const auto expected_begin = input.begin() + dram_partition * words_per_page;
    const std::vector<uint32_t> expected(expected_begin, expected_begin + words_per_page);
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

TEST(MimirEmu, BothCcesCopyAllocatedBuffer) {
    if (!emu_server_configured()) {
        GTEST_SKIP() << "Set TT_METAL_EMU_SERVER=host:port and TT_METAL_EMU_SOC_DESC=<mimir YAML>.";
    }

    IDevice* device = CreateDevice(0);
    ASSERT_NE(device, nullptr);
    ASSERT_EQ(device->arch(), tt::ARCH::QUASAR);
    ASSERT_GE(device->num_dram_channels(), 2);

    const auto& hal = MetalContext::instance().hal();
    const uint32_t page_size = hal.get_alignment(HalMemType::DRAM);
    const uint32_t num_partitions = device->num_dram_channels();
    const uint32_t buffer_size = page_size * num_partitions;
    auto src_buffer = CreateBuffer(BufferConfig{device, buffer_size, page_size, BufferType::DRAM});
    auto dst_buffer = CreateBuffer(BufferConfig{device, buffer_size, page_size, BufferType::DRAM});
    ASSERT_NE(src_buffer, nullptr);
    ASSERT_NE(dst_buffer, nullptr);

    const std::size_t words_per_page = page_size / sizeof(uint32_t);
    std::vector<uint32_t> input(buffer_size / sizeof(uint32_t));
    for (uint32_t partition = 0; partition < num_partitions; ++partition) {
        for (std::size_t i = 0; i < words_per_page; ++i) {
            input[partition * words_per_page + i] = 0xB07C0000u | (partition << 8) | static_cast<uint32_t>(i);
        }
    }
    std::vector<uint32_t> cleared(input.size(), 0);
    detail::WriteToBuffer(*src_buffer, input);
    detail::WriteToBuffer(*dst_buffer, cleared);

    const uint32_t staging_dev_addr = hal.get_dev_addr(HalProgrammableCoreType::DRAM, HalL1MemAddrType::UNRESERVED);
    Program program = CreateProgram();
    for (uint32_t cce_index = 0; cce_index < 2; ++cce_index) {
        const CoreCoord logical_dram_core{cce_index, 0};
        const KernelHandle kernel = CreateKernel(
            program,
            "tests/tt_metal/tt_metal/test_kernels/misc/cce_dram_buffer_round_trip.cpp",
            logical_dram_core,
            DramConfig{.noc = NOC::NOC_0});
        SetRuntimeArgs(
            program,
            kernel,
            logical_dram_core,
            {src_buffer->address(),
             dst_buffer->address(),
             cce_index,
             staging_dev_addr,
             static_cast<uint32_t>(words_per_page)});
    }

    detail::LaunchProgram(device, program, /*wait_until_cores_done=*/true, /*force_slow_dispatch=*/true);

    std::vector<uint32_t> output;
    detail::ReadFromBuffer(*dst_buffer, output);
    EXPECT_EQ(output, input);

    src_buffer.reset();
    dst_buffer.reset();
    EXPECT_TRUE(CloseDevice(device));
}

class AllHartsWriteAllocatedBuffer : public ::testing::TestWithParam<std::tuple<uint32_t, uint32_t>> {};

TEST_P(AllHartsWriteAllocatedBuffer, WritesMagic) {
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
    const uint32_t num_harts = hal.get_num_risc_processors(HalProgrammableCoreType::DRAM);
    ASSERT_EQ(num_harts, 8u);

    const uint32_t page_size = hal.get_alignment(HalMemType::DRAM);
    ASSERT_GE(page_size, num_harts * sizeof(uint32_t));
    const uint32_t buffer_size = page_size * device->num_dram_channels();
    auto dst_buffer = CreateBuffer(BufferConfig{device, buffer_size, page_size, BufferType::DRAM});
    ASSERT_NE(dst_buffer, nullptr);

    std::vector<uint32_t> cleared(buffer_size / sizeof(uint32_t), 0);
    detail::WriteToBuffer(*dst_buffer, cleared);

    const uint32_t staging_dev_base = hal.get_dev_addr(HalProgrammableCoreType::DRAM, HalL1MemAddrType::UNRESERVED);
    const CoreCoord logical_dram_core{cce_index, 0};
    Program program = CreateProgram();
    std::vector<uint32_t> expected(page_size / sizeof(uint32_t), 0);
    for (uint32_t hart = 0; hart < num_harts; hart++) {
        expected[hart] = 0xA11B0000u | (cce_index << 12) | (dram_partition << 8) | hart;
        CreateKernel(
            program,
            "tests/tt_metal/tt_metal/test_kernels/misc/cce_gddr_write_one_uint32.cpp",
            logical_dram_core,
            DramConfig{
                .processor = static_cast<DataMovementProcessor>(hart),
                .noc = NOC::NOC_0,
                .compile_args = {
                    dst_buffer->address() + hart * static_cast<uint32_t>(sizeof(uint32_t)),
                    dram_partition,
                    expected[hart],
                    staging_dev_base + hart * static_cast<uint32_t>(sizeof(uint32_t))}});
    }

    detail::LaunchProgram(device, program, /*wait_until_cores_done=*/true, /*force_slow_dispatch=*/true);

    std::vector<uint32_t> output;
    detail::ReadFromBuffer(*dst_buffer, output);
    ASSERT_EQ(output.size(), cleared.size());
    const std::size_t words_per_page = page_size / sizeof(uint32_t);
    const std::vector<uint32_t> zero_page(words_per_page, 0);
    for (uint32_t partition = 0; partition < device->num_dram_channels(); ++partition) {
        const auto output_begin = output.begin() + partition * words_per_page;
        const std::vector<uint32_t> output_page(output_begin, output_begin + words_per_page);
        EXPECT_EQ(output_page, partition == dram_partition ? expected : zero_page);
    }

    dst_buffer.reset();
    EXPECT_TRUE(CloseDevice(device));
}

INSTANTIATE_TEST_SUITE_P(
    MimirEmu,
    CceSramChannelsDoNotAliasThroughMetalCluster,
    ::testing::Values(0u, 1u),
    [](const ::testing::TestParamInfo<uint32_t>& info) { return fmt::format("Cce{}", info.param); });

INSTANTIATE_TEST_SUITE_P(
    MimirEmu,
    CceSramRoundTripThroughMinimalDevice,
    ::testing::Values(0u, 1u),
    [](const ::testing::TestParamInfo<uint32_t>& info) { return fmt::format("Cce{}", info.param); });

INSTANTIATE_TEST_SUITE_P(
    MimirEmu, HartZeroRunsDramKernel, ::testing::Values(0u, 1u), [](const ::testing::TestParamInfo<uint32_t>& info) {
        return fmt::format("Cce{}", info.param);
    });

INSTANTIATE_TEST_SUITE_P(
    MimirEmu, AllHartsRunDramKernel, ::testing::Values(0u, 1u), [](const ::testing::TestParamInfo<uint32_t>& info) {
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

INSTANTIATE_TEST_SUITE_P(
    MimirEmu,
    AllocatedDramBufferHostLoopback,
    ::testing::Values(0u, 1u),
    [](const ::testing::TestParamInfo<uint32_t>& info) { return fmt::format("Partition{}", info.param); });

INSTANTIATE_TEST_SUITE_P(
    MimirEmu,
    CopiesAllocatedBufferWithRuntimeArgs,
    ::testing::Combine(::testing::Values(0u, 1u), ::testing::Values(0u, 1u)),
    [](const ::testing::TestParamInfo<std::tuple<uint32_t, uint32_t>>& info) {
        return fmt::format("Cce{}_Partition{}", std::get<0>(info.param), std::get<1>(info.param));
    });

INSTANTIATE_TEST_SUITE_P(
    MimirEmu,
    CopiesEveryPartitionOfAllocatedBuffer,
    ::testing::Values(0u, 1u),
    [](const ::testing::TestParamInfo<uint32_t>& info) { return fmt::format("Cce{}", info.param); });

INSTANTIATE_TEST_SUITE_P(
    MimirEmu,
    CopiesAllocatedBufferLargerThanAlignment,
    ::testing::Combine(::testing::Values(0u, 1u), ::testing::Values(0u, 1u)),
    [](const ::testing::TestParamInfo<std::tuple<uint32_t, uint32_t>>& info) {
        return fmt::format("Cce{}_Partition{}", std::get<0>(info.param), std::get<1>(info.param));
    });

INSTANTIATE_TEST_SUITE_P(
    MimirEmu,
    AllHartsWriteAllocatedBuffer,
    ::testing::Combine(::testing::Values(0u, 1u), ::testing::Values(0u, 1u)),
    [](const ::testing::TestParamInfo<std::tuple<uint32_t, uint32_t>>& info) {
        return fmt::format("Cce{}_Partition{}", std::get<0>(info.param), std::get<1>(info.param));
    });

class DramChannelsDoNotAliasThroughPublicDeviceApi : public ::testing::TestWithParam<uint32_t> {};

TEST_P(DramChannelsDoNotAliasThroughPublicDeviceApi, RoundTrip) {
    if (!emu_server_configured()) {
        GTEST_SKIP() << "Set TT_METAL_EMU_SERVER=host:port and TT_METAL_EMU_SOC_DESC=<mimir YAML>.";
    }

    std::unique_ptr<IDevice> device(CreateDeviceMinimal(0));
    ASSERT_NE(device, nullptr);
    ASSERT_EQ(device->num_dram_channels(), 2);

    const uint32_t dram_partition = GetParam();
    const uint32_t other_partition = 1u - dram_partition;
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
    ::testing::Values(0u, 1u),
    [](const ::testing::TestParamInfo<uint32_t>& info) { return fmt::format("Partition{}", info.param); });

}  // namespace
}  // namespace tt::tt_metal
