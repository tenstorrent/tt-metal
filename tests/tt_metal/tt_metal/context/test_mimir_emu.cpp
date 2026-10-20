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
#include "impl/kernels/kernel.hpp"
#include "llrt/hal.hpp"
#include "dev_mem_map.h"
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>
#include "mimir_contract.hpp"
#include "umd/device/types/arch.hpp"
#include "umd/device/types/core_coordinates.hpp"

namespace tt::tt_metal {
namespace {

using namespace mimir_contract;

static_assert(MEM_L1_UNCACHED_BASE == MEM_L1_SIZE);

class CceSramRoundTripThroughMinimalDevice : public ::testing::TestWithParam<uint32_t> {};

TEST_P(CceSramRoundTripThroughMinimalDevice, RoundTrip) {
    skip_unless_programmable_dram_package();

    std::unique_ptr<IDevice> device(CreateDeviceMinimal(0));
    ASSERT_NE(device, nullptr);
    ASSERT_EQ(device->arch(), tt::ARCH::QUASAR);

    const uint32_t cce_index = GetParam();
    const uint32_t num_cces = num_cces_on(device.get());
    if (cce_index >= num_cces) {
        device.reset();
        GTEST_SKIP() << "Configured descriptor does not expose CCE " << cce_index << ".";
    }

    const CoreCoord logical_dram_core = logical_cce_core(soc_desc_of(device.get()), cce_index);
    const uint64_t address = cce_sram_noc_addr(kCceSramTestOffset);
    const uint32_t written = 0xC0FFEE03 + cce_index;
    uint32_t read_back = 0;
    write_cce_sram(device.get(), logical_dram_core, address, &written, sizeof(written));
    read_cce_sram(device.get(), logical_dram_core, address, &read_back, sizeof(read_back));
    EXPECT_EQ(read_back, written);
}

TEST(MimirEmu, RuntimeFirmwareInitializesThroughPublicDeviceApi) {
    skip_unless_programmable_dram_package();

    IDevice* device = nullptr;
    ASSERT_NO_THROW(device = CreateDevice(0));
    ASSERT_NE(device, nullptr);
    ASSERT_EQ(device->arch(), tt::ARCH::QUASAR);
    EXPECT_TRUE(CloseDevice(device));
}

class HartZeroRunsDramKernel : public ::testing::TestWithParam<uint32_t> {};

TEST_P(HartZeroRunsDramKernel, WritesMagic) {
    skip_unless_programmable_dram_package();

    IDevice* device = CreateDevice(0);
    ASSERT_NE(device, nullptr);
    ASSERT_EQ(device->arch(), tt::ARCH::QUASAR);

    const uint32_t cce_index = GetParam();
    const uint32_t num_cces = num_cces_on(device);
    if (cce_index >= num_cces) {
        EXPECT_TRUE(CloseDevice(device));
        GTEST_SKIP() << "Configured descriptor does not expose CCE " << cce_index << ".";
    }

    const CoreCoord logical_dram_core = logical_cce_core(soc_desc_of(device), cce_index);
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
    read_cce_sram(device, logical_dram_core, result_noc_addr, &result, sizeof(result));
    EXPECT_EQ(result, magic);
    EXPECT_TRUE(CloseDevice(device));
}

class AllHartsRunDramKernel : public ::testing::TestWithParam<uint32_t> {};

TEST_P(AllHartsRunDramKernel, WritesMagic) {
    skip_unless_programmable_dram_package();

    IDevice* device = CreateDevice(0);
    ASSERT_NE(device, nullptr);
    ASSERT_EQ(device->arch(), tt::ARCH::QUASAR);

    const uint32_t cce_index = GetParam();
    const uint32_t num_cces = num_cces_on(device);
    if (cce_index >= num_cces) {
        EXPECT_TRUE(CloseDevice(device));
        GTEST_SKIP() << "Configured descriptor does not expose CCE " << cce_index << ".";
    }

    const CoreCoord logical_dram_core = logical_cce_core(soc_desc_of(device), cce_index);
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
    read_cce_sram(device, logical_dram_core, result_noc_base, result.data(), result.size() * sizeof(uint32_t));
    EXPECT_EQ(result, expected);
    EXPECT_TRUE(CloseDevice(device));
}

class CopiesGddrWithCompileTimeArgs : public ::testing::TestWithParam<CcePartition> {};

TEST_P(CopiesGddrWithCompileTimeArgs, RoundTrip) {
    skip_unless_programmable_dram_package();
    const CcePartition param = GetParam();
    const uint32_t cce_index = param.cce_index;
    const bool cross = is_cross_mimir_case();
    if (!cce_partition_case_applies(param, cross)) {
        GTEST_SKIP() << "Configured descriptor does not expose CCE " << cce_index << " partition " << param.partition
                     << (cross ? " as a remote partition." : " as a local partition.");
    }
    IDevice* device = CreateDevice(0);
    ASSERT_NE(device, nullptr);
    ASSERT_EQ(device->arch(), tt::ARCH::QUASAR);
    const metal_SocDescriptor& soc_desc = soc_desc_of(device);
    const GddrSweep sweep = gddr_sweep_from_soc(device);
    ASSERT_NE(sweep.num_slots, 0u) << "dram_view_size must hold at least one GDDR slot";
    const std::vector<GddrTarget> targets = {gddr_target_for_view(soc_desc, param.partition, sweep)};
    GddrBackingScope gddr_backing;

    const auto& hal = MetalContext::instance().hal();
    const uint32_t transfer_size = hal.get_alignment(HalMemType::DRAM);
    const uint32_t src_dram_offset = hal.get_dev_addr(HalDramMemAddrType::UNRESERVED);
    const uint32_t dst_dram_offset = src_dram_offset + transfer_size;
    const uint32_t staging_dev_addr = hal.get_dev_addr(HalProgrammableCoreType::DRAM, HalL1MemAddrType::UNRESERVED);
    const uint64_t staging_noc_addr = hal.get_dev_noc_addr(HalProgrammableCoreType::DRAM, HalL1MemAddrType::UNRESERVED);
    const CoreCoord logical_dram_core = logical_cce_core(soc_desc_of(device), cce_index);
    std::vector<uint32_t> cleared(transfer_size / sizeof(uint32_t), 0);

    for (const GddrTarget& target : targets) {
        std::vector<std::vector<uint32_t>> inputs(
            target.num_slots, std::vector<uint32_t>(transfer_size / sizeof(uint32_t)));
        for (uint32_t slot = 0; slot < target.num_slots; ++slot) {
            for (std::size_t i = 0; i < inputs[slot].size(); ++i) {
                inputs[slot][i] = 0xC0FF0000u | (cce_index << 12) | (target.mimir_index << 8) | (target.location << 4) |
                                  static_cast<uint32_t>(i);
            }
            const uint64_t slot_offset = host_slot_offset(target, slot);
            write_gddr_view(device, target.host_view, src_dram_offset + slot_offset, inputs[slot]);
            write_gddr_view(device, target.host_view, dst_dram_offset + slot_offset, cleared);
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
                    target.slot_base,
                    mimir_index_for_cce(cce_index)}});

        detail::LaunchProgram(device, program, /*wait_until_cores_done=*/true, /*force_slow_dispatch=*/true);

        std::vector<uint32_t> staged(cleared.size(), 0);
        read_cce_sram(device, logical_dram_core, staging_noc_addr, staged.data(), transfer_size);
        EXPECT_EQ(staged, inputs.back()) << "mimir " << target.mimir_index << " location " << target.location;

        for (uint32_t slot = 0; slot < target.num_slots; ++slot) {
            std::vector<uint32_t> output;
            read_gddr_view(
                device, target.host_view, dst_dram_offset + host_slot_offset(target, slot), transfer_size, output);
            EXPECT_EQ(output, inputs[slot])
                << "mimir " << target.mimir_index << " location " << target.location << " slot " << slot;
        }
    }
    EXPECT_TRUE(CloseDevice(device));
}

class CopiesGddrThroughAllocatedBuffer : public ::testing::TestWithParam<CcePartition> {};

TEST_P(CopiesGddrThroughAllocatedBuffer, RoundTrip) {
    skip_unless_programmable_dram_package();
    const CcePartition param = GetParam();
    const uint32_t cce_index = param.cce_index;
    const bool cross = is_cross_mimir_case();
    if (!cce_partition_case_applies(param, cross)) {
        GTEST_SKIP() << "Configured descriptor does not expose CCE " << cce_index << " partition " << param.partition
                     << (cross ? " as a remote partition." : " as a local partition.");
    }
    IDevice* device = CreateDevice(0);
    ASSERT_NE(device, nullptr);
    ASSERT_EQ(device->arch(), tt::ARCH::QUASAR);
    const metal_SocDescriptor& soc_desc = soc_desc_of(device);
    const GddrSweep sweep = gddr_sweep_from_soc(device);
    ASSERT_NE(sweep.num_slots, 0u) << "dram_view_size must hold at least one GDDR slot";
    const std::vector<GddrTarget> targets = {gddr_target_for_view(soc_desc, param.partition, sweep)};
    GddrBackingScope gddr_backing;

    const auto& hal = MetalContext::instance().hal();
    const uint32_t transfer_size = hal.get_alignment(HalMemType::DRAM);
    const uint32_t src_dram_offset = hal.get_dev_addr(HalDramMemAddrType::UNRESERVED);
    const uint32_t dst_dram_offset = src_dram_offset + transfer_size;
    const uint32_t staging_dev_addr = hal.get_dev_addr(HalProgrammableCoreType::DRAM, HalL1MemAddrType::UNRESERVED);
    const CoreCoord logical_dram_core = logical_cce_core(soc_desc_of(device), cce_index);
    std::vector<uint32_t> cleared(transfer_size / sizeof(uint32_t), 0);

    for (const GddrTarget& target : targets) {
        std::vector<std::vector<uint32_t>> inputs(
            target.num_slots, std::vector<uint32_t>(transfer_size / sizeof(uint32_t)));
        for (uint32_t slot = 0; slot < target.num_slots; ++slot) {
            for (std::size_t i = 0; i < inputs[slot].size(); ++i) {
                inputs[slot][i] = 0xC0FE0000u | (cce_index << 12) | (target.mimir_index << 8) | (target.location << 4) |
                                  static_cast<uint32_t>(i);
            }
            const uint64_t slot_offset = host_slot_offset(target, slot);
            write_gddr_view(device, target.host_view, src_dram_offset + slot_offset, inputs[slot]);
            write_gddr_view(device, target.host_view, dst_dram_offset + slot_offset, cleared);
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
                    target.slot_base,
                    mimir_index_for_cce(cce_index)}});

        detail::LaunchProgram(device, program, /*wait_until_cores_done=*/true, /*force_slow_dispatch=*/true);

        for (uint32_t slot = 0; slot < target.num_slots; ++slot) {
            std::vector<uint32_t> output;
            read_gddr_view(
                device, target.host_view, dst_dram_offset + host_slot_offset(target, slot), transfer_size, output);
            EXPECT_EQ(output, inputs[slot])
                << "mimir " << target.mimir_index << " location " << target.location << " slot " << slot;
        }
    }
    EXPECT_TRUE(CloseDevice(device));
}

class CopiesAllocatedBufferWithRuntimeArgs : public ::testing::TestWithParam<CcePartition> {};

TEST_P(CopiesAllocatedBufferWithRuntimeArgs, RoundTrip) {
    skip_unless_programmable_dram_package();
    const CcePartition param = GetParam();
    const uint32_t cce_index = param.cce_index;
    const bool cross = is_cross_mimir_case();
    if (!cce_partition_case_applies(param, cross)) {
        GTEST_SKIP() << "Configured descriptor does not expose CCE " << cce_index << " partition " << param.partition
                     << (cross ? " as a remote partition." : " as a local partition.");
    }
    IDevice* device = CreateDevice(0);
    ASSERT_NE(device, nullptr);
    ASSERT_EQ(device->arch(), tt::ARCH::QUASAR);
    const metal_SocDescriptor& soc_desc = soc_desc_of(device);
    const GddrSweep sweep = gddr_sweep_from_soc(device);
    ASSERT_NE(sweep.num_slots, 0u) << "dram_view_size must hold at least one GDDR slot";
    const std::vector<GddrTarget> targets = {gddr_target_for_view(soc_desc, param.partition, sweep)};
    GddrBackingScope gddr_backing;

    const auto& hal = MetalContext::instance().hal();
    const uint32_t transfer_size = hal.get_alignment(HalMemType::DRAM);
    const uint32_t src_dram_offset = hal.get_dev_addr(HalDramMemAddrType::UNRESERVED);
    const uint32_t dst_dram_offset = src_dram_offset + transfer_size;
    const uint32_t staging_dev_addr = hal.get_dev_addr(HalProgrammableCoreType::DRAM, HalL1MemAddrType::UNRESERVED);
    const CoreCoord logical_dram_core = logical_cce_core(soc_desc_of(device), cce_index);
    std::vector<uint32_t> cleared(transfer_size / sizeof(uint32_t), 0);

    for (const GddrTarget& target : targets) {
        std::vector<std::vector<uint32_t>> inputs(
            target.num_slots, std::vector<uint32_t>(transfer_size / sizeof(uint32_t)));
        for (uint32_t slot = 0; slot < target.num_slots; ++slot) {
            for (std::size_t i = 0; i < inputs[slot].size(); ++i) {
                inputs[slot][i] = 0xD15C0000u | (cce_index << 12) | (target.mimir_index << 8) | (target.location << 4) |
                                  static_cast<uint32_t>(i);
            }
            const uint64_t slot_offset = host_slot_offset(target, slot);
            write_gddr_view(device, target.host_view, src_dram_offset + slot_offset, inputs[slot]);
            write_gddr_view(device, target.host_view, dst_dram_offset + slot_offset, cleared);
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
             target.slot_base,
             mimir_index_for_cce(cce_index)});

        detail::LaunchProgram(device, program, /*wait_until_cores_done=*/true, /*force_slow_dispatch=*/true);

        for (uint32_t slot = 0; slot < target.num_slots; ++slot) {
            std::vector<uint32_t> output;
            read_gddr_view(
                device, target.host_view, dst_dram_offset + host_slot_offset(target, slot), transfer_size, output);
            EXPECT_EQ(output, inputs[slot])
                << "mimir " << target.mimir_index << " location " << target.location << " slot " << slot;
        }
    }
    EXPECT_TRUE(CloseDevice(device));
}

class AllocatedDramBufferHostLoopback : public ::testing::TestWithParam<uint32_t> {};

TEST_P(AllocatedDramBufferHostLoopback, RoundTrip) {
    skip_unless_programmable_dram_package();
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
    skip_unless_programmable_dram_package();
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

    const CoreCoord logical_dram_core = logical_cce_core(soc_desc_of(device), cce_index);
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
        {src_dram_offset,
         dst_dram_offset,
         num_mimirs,
         staging_dev_addr,
         static_cast<uint32_t>(cleared.size()),
         mimir_index_for_cce(cce_index)});

    detail::LaunchProgram(device, program, /*wait_until_cores_done=*/true, /*force_slow_dispatch=*/true);

    for (uint32_t mimir_index = 0; mimir_index < num_mimirs; ++mimir_index) {
        std::vector<uint32_t> output;
        read_mimir_gddr(device, mimir_index, dst_dram_offset, transfer_size, output);
        EXPECT_EQ(output, inputs[mimir_index]) << "mimir " << mimir_index;
    }
    EXPECT_TRUE(CloseDevice(device));
}

class CopiesAllocatedBufferLargerThanAlignment : public ::testing::TestWithParam<CcePartition> {};

TEST_P(CopiesAllocatedBufferLargerThanAlignment, RoundTrip) {
    skip_unless_programmable_dram_package();
    const CcePartition param = GetParam();
    const uint32_t cce_index = param.cce_index;
    const bool cross = is_cross_mimir_case();
    if (!cce_partition_case_applies(param, cross)) {
        GTEST_SKIP() << "Configured descriptor does not expose CCE " << cce_index << " partition " << param.partition
                     << (cross ? " as a remote partition." : " as a local partition.");
    }
    IDevice* device = CreateDevice(0);
    ASSERT_NE(device, nullptr);
    ASSERT_EQ(device->arch(), tt::ARCH::QUASAR);
    const metal_SocDescriptor& soc_desc = soc_desc_of(device);
    const GddrSweep sweep = gddr_sweep_from_soc(device);
    ASSERT_NE(sweep.num_slots, 0u) << "dram_view_size must hold at least one GDDR slot";
    const std::vector<GddrTarget> targets = {gddr_target_for_view(soc_desc, param.partition, sweep)};
    GddrBackingScope gddr_backing;

    const auto& hal = MetalContext::instance().hal();
    const uint32_t alignment = hal.get_alignment(HalMemType::DRAM);
    constexpr uint32_t kPagesPerBank = 16;
    const uint32_t transfer_size = std::min(alignment * kPagesPerBank, static_cast<uint32_t>(kEmuGddrBackingPage));
    const uint32_t src_dram_offset = hal.get_dev_addr(HalDramMemAddrType::UNRESERVED);
    const uint32_t dst_dram_offset = src_dram_offset + transfer_size;
    const uint32_t staging_dev_addr = hal.get_dev_addr(HalProgrammableCoreType::DRAM, HalL1MemAddrType::UNRESERVED);
    const CoreCoord logical_dram_core = logical_cce_core(soc_desc_of(device), cce_index);
    std::vector<uint32_t> cleared(transfer_size / sizeof(uint32_t), 0);

    for (const GddrTarget& target : targets) {
        std::vector<std::vector<uint32_t>> inputs(
            target.num_slots, std::vector<uint32_t>(transfer_size / sizeof(uint32_t)));
        for (uint32_t slot = 0; slot < target.num_slots; ++slot) {
            for (std::size_t i = 0; i < inputs[slot].size(); ++i) {
                inputs[slot][i] = 0x1A6E0000u | (cce_index << 12) | (target.mimir_index << 8) | (target.location << 4) |
                                  static_cast<uint32_t>(i);
            }
            const uint64_t slot_offset = host_slot_offset(target, slot);
            write_gddr_view(device, target.host_view, src_dram_offset + slot_offset, inputs[slot]);
            write_gddr_view(device, target.host_view, dst_dram_offset + slot_offset, cleared);
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
             target.slot_base,
             mimir_index_for_cce(cce_index)});

        detail::LaunchProgram(device, program, /*wait_until_cores_done=*/true, /*force_slow_dispatch=*/true);

        for (uint32_t slot = 0; slot < target.num_slots; ++slot) {
            std::vector<uint32_t> output;
            read_gddr_view(
                device, target.host_view, dst_dram_offset + host_slot_offset(target, slot), transfer_size, output);
            EXPECT_EQ(output, inputs[slot])
                << "mimir " << target.mimir_index << " location " << target.location << " slot " << slot;
        }
    }
    EXPECT_TRUE(CloseDevice(device));
}

TEST(MimirEmu, AllCcesCopyLocalMimirBlock) {
    skip_unless_programmable_dram_package();
    IDevice* device = CreateDevice(0);
    ASSERT_NE(device, nullptr);
    ASSERT_EQ(device->arch(), tt::ARCH::QUASAR);
    const uint32_t num_cces = num_cces_on(device);
    ASSERT_GE(num_cces, 1u);
    const uint32_t num_mimirs = num_mimirs_on(device);
    ASSERT_GE(num_mimirs, 1u);
    const GddrSweep sweep = gddr_sweep_from_soc(device);
    ASSERT_NE(sweep.num_slots, 0u) << "dram_view_size must hold at least one GDDR slot";
    GddrBackingScope gddr_backing;

    const auto& hal = MetalContext::instance().hal();
    const uint32_t transfer_size = hal.get_alignment(HalMemType::DRAM);
    const uint32_t src_dram_offset = hal.get_dev_addr(HalDramMemAddrType::UNRESERVED);
    const uint32_t staging_dev_addr = hal.get_dev_addr(HalProgrammableCoreType::DRAM, HalL1MemAddrType::UNRESERVED);

    const metal_SocDescriptor& soc_desc = soc_desc_of(device);
    std::vector<std::vector<uint32_t>> inputs(num_cces, std::vector<uint32_t>(transfer_size / sizeof(uint32_t)));
    std::vector<uint32_t> cleared(transfer_size / sizeof(uint32_t), 0);
    Program program = CreateProgram();
    for (uint32_t cce_index = 0; cce_index < num_cces; ++cce_index) {
        const GddrTarget local = gddr_target_for_view(soc_desc, local_dram_view_for_cce(soc_desc, cce_index), sweep);
        for (std::size_t i = 0; i < inputs[cce_index].size(); ++i) {
            inputs[cce_index][i] = 0xB07C0000u | (cce_index << 8) | static_cast<uint32_t>(i);
        }
        const uint32_t cce_src = src_dram_offset + cce_index * 2 * transfer_size;
        const uint32_t cce_dst = cce_src + transfer_size;
        write_gddr_view(device, local.host_view, cce_src, inputs[cce_index]);
        write_gddr_view(device, local.host_view, cce_dst, cleared);

        const CoreCoord logical_dram_core = logical_cce_core(soc_desc_of(device), cce_index);
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
             local.mimir_index,
             staging_dev_addr,
             static_cast<uint32_t>(cleared.size()),
             1u,
             sweep.slot_stride,
             local.slot_base,
             local.mimir_index});
    }

    detail::LaunchProgram(device, program, /*wait_until_cores_done=*/true, /*force_slow_dispatch=*/true);

    for (uint32_t cce_index = 0; cce_index < num_cces; ++cce_index) {
        std::vector<uint32_t> output;
        const uint32_t cce_dst = src_dram_offset + cce_index * 2 * transfer_size + transfer_size;
        read_gddr_view(device, local_dram_view_for_cce(soc_desc, cce_index), cce_dst, transfer_size, output);
        EXPECT_EQ(output, inputs[cce_index]) << "cce " << cce_index;
    }
    EXPECT_TRUE(CloseDevice(device));
}

class WritesCrossMimirCceSram : public ::testing::TestWithParam<uint32_t> {};

TEST_P(WritesCrossMimirCceSram, WritesMagic) {
    skip_unless_programmable_dram_package();

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

    for (uint32_t dest_cce : dest_cces) {
        const CoreCoord dest_core = logical_cce_core(soc_desc_of(device), dest_cce);
        write_cce_sram(device, dest_core, noc_addr, &zero, sizeof(zero));
    }

    const CoreCoord logical_src = logical_cce_core(soc_desc_of(device), src_cce);
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
        const CoreCoord dest_core = logical_cce_core(soc_desc_of(device), dest_cce);
        read_cce_sram(device, dest_core, noc_addr, &result, sizeof(result));
        EXPECT_EQ(result, magic_base | dest_cce) << "src cce " << src_cce << " dest cce " << dest_cce;
    }
    EXPECT_TRUE(CloseDevice(device));
}

class PrefetchesLocalGddrToRemoteCceSram : public ::testing::TestWithParam<uint32_t> {};

TEST_P(PrefetchesLocalGddrToRemoteCceSram, RoundTrip) {
    skip_unless_programmable_dram_package();
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
    const metal_SocDescriptor& soc_desc = soc_desc_of(device);
    const uint32_t local_view = local_dram_view_for_cce(soc_desc, src_cce);
    const uint64_t local_address_offset = soc_desc.get_address_offset(static_cast<int>(local_view));
    ASSERT_EQ(local_address_offset % kGddrSlotStride, 0ull);
    const uint32_t local_base_slots = static_cast<uint32_t>(local_address_offset / kGddrSlotStride);
    write_gddr_view(device, local_view, src_gddr_offset, input);

    for (uint32_t dest_cce : dest_cces) {
        const CoreCoord dest_core = logical_cce_core(soc_desc_of(device), dest_cce);
        write_cce_sram(device, dest_core, noc_addr, cleared.data(), transfer_size);
    }

    const CoreCoord logical_src = logical_cce_core(soc_desc_of(device), src_cce);
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
        {src_gddr_offset,
         local_mimir,
         staging_dev_addr,
         num_words,
         dest_cces[0],
         dest_cces[1],
         sram_offset,
         local_base_slots,
         static_cast<uint32_t>(kGddrSlotStride)});

    detail::LaunchProgram(device, program, /*wait_until_cores_done=*/true, /*force_slow_dispatch=*/true);

    for (uint32_t dest_cce : dest_cces) {
        std::vector<uint32_t> output(num_words, 0);
        const CoreCoord dest_core = logical_cce_core(soc_desc_of(device), dest_cce);
        read_cce_sram(device, dest_core, noc_addr, output.data(), transfer_size);
        EXPECT_EQ(output, input) << "src cce " << src_cce << " dest cce " << dest_cce;
    }
    EXPECT_TRUE(CloseDevice(device));
}

class AllHartsWriteAllocatedBuffer : public ::testing::TestWithParam<CcePartition> {};

TEST_P(AllHartsWriteAllocatedBuffer, WritesMagic) {
    skip_unless_programmable_dram_package();
    const CcePartition param = GetParam();
    const uint32_t cce_index = param.cce_index;
    const bool cross = is_cross_mimir_case();
    if (!cce_partition_case_applies(param, cross)) {
        GTEST_SKIP() << "Configured descriptor does not expose CCE " << cce_index << " partition " << param.partition
                     << (cross ? " as a remote partition." : " as a local partition.");
    }
    IDevice* device = CreateDevice(0);
    ASSERT_NE(device, nullptr);
    ASSERT_EQ(device->arch(), tt::ARCH::QUASAR);
    const metal_SocDescriptor& soc_desc = soc_desc_of(device);
    const GddrSweep sweep = gddr_sweep_from_soc(device);
    ASSERT_NE(sweep.num_slots, 0u) << "dram_view_size must hold at least one GDDR slot";
    const std::vector<GddrTarget> targets = {gddr_target_for_view(soc_desc, param.partition, sweep)};
    GddrBackingScope gddr_backing;

    const auto& hal = MetalContext::instance().hal();
    const uint32_t num_harts = hal.get_num_risc_processors(HalProgrammableCoreType::DRAM);
    ASSERT_EQ(num_harts, 8u);

    const uint32_t dst_dram_offset = hal.get_dev_addr(HalDramMemAddrType::UNRESERVED);
    const uint32_t staging_dev_base = hal.get_dev_addr(HalProgrammableCoreType::DRAM, HalL1MemAddrType::UNRESERVED);
    const CoreCoord logical_dram_core = logical_cce_core(soc_desc_of(device), cce_index);
    std::vector<uint32_t> cleared(num_harts, 0);

    for (const GddrTarget& target : targets) {
        for (uint32_t slot = 0; slot < target.num_slots; ++slot) {
            write_gddr_view(device, target.host_view, dst_dram_offset + host_slot_offset(target, slot), cleared);
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
                        target.slot_base,
                        mimir_index_for_cce(cce_index)}});
        }

        detail::LaunchProgram(device, program, /*wait_until_cores_done=*/true, /*force_slow_dispatch=*/true);

        for (uint32_t slot = 0; slot < target.num_slots; ++slot) {
            std::vector<uint32_t> output;
            read_gddr_view(
                device,
                target.host_view,
                dst_dram_offset + host_slot_offset(target, slot),
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
    MimirEmuSameMimirDirect,
    CopiesGddrWithCompileTimeArgs,
    ::testing::ValuesIn(cce_partition_params()),
    cce_partition_name);

INSTANTIATE_TEST_SUITE_P(
    MimirEmuCrossMimirNoc,
    CopiesGddrWithCompileTimeArgs,
    ::testing::ValuesIn(cce_partition_params()),
    cce_partition_name);

INSTANTIATE_TEST_SUITE_P(
    MimirEmuSameMimirDirect,
    CopiesGddrThroughAllocatedBuffer,
    ::testing::ValuesIn(cce_partition_params()),
    cce_partition_name);

INSTANTIATE_TEST_SUITE_P(
    MimirEmuCrossMimirNoc,
    CopiesGddrThroughAllocatedBuffer,
    ::testing::ValuesIn(cce_partition_params()),
    cce_partition_name);

INSTANTIATE_TEST_SUITE_P(
    MimirEmu,
    AllocatedDramBufferHostLoopback,
    ::testing::Range(0u, 4u),
    [](const ::testing::TestParamInfo<uint32_t>& info) { return fmt::format("Partition{}", info.param); });

INSTANTIATE_TEST_SUITE_P(
    MimirEmuSameMimirDirect,
    CopiesAllocatedBufferWithRuntimeArgs,
    ::testing::ValuesIn(cce_partition_params()),
    cce_partition_name);

INSTANTIATE_TEST_SUITE_P(
    MimirEmuCrossMimirNoc,
    CopiesAllocatedBufferWithRuntimeArgs,
    ::testing::ValuesIn(cce_partition_params()),
    cce_partition_name);

INSTANTIATE_TEST_SUITE_P(MimirEmu, CopiesEveryMimirBlock, ::testing::Range(0u, 4u), cce_param_name);

INSTANTIATE_TEST_SUITE_P(
    MimirEmuSameMimirDirect,
    CopiesAllocatedBufferLargerThanAlignment,
    ::testing::ValuesIn(cce_partition_params()),
    cce_partition_name);

INSTANTIATE_TEST_SUITE_P(
    MimirEmuCrossMimirNoc,
    CopiesAllocatedBufferLargerThanAlignment,
    ::testing::ValuesIn(cce_partition_params()),
    cce_partition_name);

INSTANTIATE_TEST_SUITE_P(
    MimirEmuSameMimirDirect,
    AllHartsWriteAllocatedBuffer,
    ::testing::ValuesIn(cce_partition_params()),
    cce_partition_name);

INSTANTIATE_TEST_SUITE_P(
    MimirEmuCrossMimirNoc,
    AllHartsWriteAllocatedBuffer,
    ::testing::ValuesIn(cce_partition_params()),
    cce_partition_name);

class DramChannelsDoNotAliasThroughPublicDeviceApi : public ::testing::TestWithParam<uint32_t> {};

TEST_P(DramChannelsDoNotAliasThroughPublicDeviceApi, RoundTrip) {
    skip_unless_programmable_dram_package();
    std::unique_ptr<IDevice> device(CreateDeviceMinimal(0));
    ASSERT_NE(device, nullptr);

    const uint32_t dram_partition = GetParam();
    const uint32_t num_partitions = device->num_dram_channels();
    if (num_partitions < 2) {
        device.reset();
        GTEST_SKIP() << "Alias check needs at least two DRAM channels; descriptor exposes " << num_partitions << ".";
    }
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
