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
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>
#include "mimir_contract.hpp"
#include "umd/device/types/arch.hpp"
#include "umd/device/types/core_coordinates.hpp"

namespace tt::tt_metal {
namespace {

using namespace mimir_contract;

// Same open as the Wormhole/Blackhole DRISC tests: one MeshDevice over chip 0. Host reads and writes
// use the physical device, because a MeshDevice id is a mesh id. Run with TT_METAL_SLOW_DISPATCH_MODE=1
// so the mesh command queue takes the slow-dispatch path.
std::shared_ptr<distributed::MeshDevice> open_unit_mesh() { return distributed::MeshDevice::create_unit_mesh(0); }

IDevice* physical_device(distributed::MeshDevice& mesh) {
    return mesh.get_device(distributed::MeshCoordinate::zero_coordinate(mesh.shape().dims()));
}

void LaunchProgram(distributed::MeshDevice& mesh, Program& program) {
    distributed::MeshWorkload workload;
    workload.add_program(distributed::MeshCoordinateRange(mesh.shape()), std::move(program));
    distributed::EnqueueMeshWorkload(mesh.mesh_command_queue(), workload, /*blocking=*/true);
}

static_assert(MEM_L1_UNCACHED_BASE == MEM_L1_SIZE);

class CceSramRoundTripThroughUnitMesh : public ::testing::TestWithParam<uint32_t> {};

TEST_P(CceSramRoundTripThroughUnitMesh, RoundTrip) {
    skip_unless_programmable_dram_package();

    auto mesh_device = open_unit_mesh();
    IDevice* device = physical_device(*mesh_device);
    ASSERT_NE(device, nullptr);
    ASSERT_EQ(device->arch(), tt::ARCH::QUASAR);

    const uint32_t cce_index = GetParam();
    const uint32_t num_cces = num_cces_on(device);
    if (cce_index >= num_cces) {
        GTEST_SKIP() << "Configured descriptor does not expose CCE " << cce_index << ".";
    }

    const CoreCoord logical_dram_core = logical_cce_core(soc_desc_of(device), cce_index);
    const uint64_t address = cce_sram_noc_addr(kCceSramTestOffset);
    const uint32_t written = 0xC0FFEE03 + cce_index;
    uint32_t read_back = 0;
    write_cce_sram(device, logical_dram_core, address, &written, sizeof(written));
    read_cce_sram(device, logical_dram_core, address, &read_back, sizeof(read_back));
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

    const uint32_t cce_index = GetParam();
    const uint32_t num_cces = num_cces_on(device);
    if (cce_index >= num_cces) {
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

    // Nothing else writes UNRESERVED, so without the poison a kernel that never runs reads the last run's magic.
    const uint32_t poison = 0xDEADBEEFu;
    uint32_t result = 0;
    write_cce_sram(device, logical_dram_core, result_noc_addr, &poison, sizeof(poison));
    read_cce_sram(device, logical_dram_core, result_noc_addr, &result, sizeof(result));
    ASSERT_EQ(result, poison);

    LaunchProgram(*mesh_device, program);

    read_cce_sram(device, logical_dram_core, result_noc_addr, &result, sizeof(result));
    EXPECT_EQ(result, magic);
}

class AllHartsRunDramKernel : public ::testing::TestWithParam<uint32_t> {};

TEST_P(AllHartsRunDramKernel, WritesMagic) {
    skip_unless_programmable_dram_package();

    auto mesh_device = open_unit_mesh();
    IDevice* device = physical_device(*mesh_device);
    ASSERT_NE(device, nullptr);
    ASSERT_EQ(device->arch(), tt::ARCH::QUASAR);

    const uint32_t cce_index = GetParam();
    const uint32_t num_cces = num_cces_on(device);
    if (cce_index >= num_cces) {
        GTEST_SKIP() << "Configured descriptor does not expose CCE " << cce_index << ".";
    }

    const CoreCoord logical_dram_core = logical_cce_core(soc_desc_of(device), cce_index);
    const auto& hal = MetalContext::instance().hal();
    const uint32_t num_harts = hal.get_num_risc_processors(HalProgrammableCoreType::DRAM);
    ASSERT_EQ(num_harts, 8u);

    const uint32_t echo_dev_base = hal.get_dev_addr(HalProgrammableCoreType::DRAM, HalL1MemAddrType::UNRESERVED);
    const uint64_t echo_noc_base = hal.get_dev_noc_addr(HalProgrammableCoreType::DRAM, HalL1MemAddrType::UNRESERVED);
    // The echo kernel writes {magic, mhartid, marker} to the slot its mhartid picks, so a hart that runs another
    // hart's kernel leaves the wrong marker. The fourth word is never written and must keep the poison.
    constexpr uint32_t slot_words = 4;
    const uint32_t poison = 0xDEADBEEFu;
    const uint32_t echo_magic = 0xC0FFEE10 + (cce_index << 8);
    const uint32_t echo_bytes = num_harts * slot_words * sizeof(uint32_t);

    Program program = CreateProgram();
    std::vector<uint32_t> expected(num_harts * slot_words, poison);
    for (uint32_t hart = 0; hart < num_harts; hart++) {
        const uint32_t marker = 0xC0DE0000 + (cce_index << 8) + hart;
        expected[hart * slot_words] = echo_magic;
        expected[hart * slot_words + 1] = hart;
        expected[hart * slot_words + 2] = marker;
        CreateKernel(
            program,
            "tests/tt_metal/tt_metal/test_kernels/cce/cce_args_echo.cpp",
            logical_dram_core,
            DramConfig{
                .processor = static_cast<DataMovementProcessor>(hart),
                .noc = NOC::NOC_0,
                .compile_args = {
                    echo_dev_base,
                    marker,
                    echo_magic,
                    slot_words * static_cast<uint32_t>(sizeof(uint32_t)),
                    0,
                    0,
                    0,
                    0}});
    }

    // Nothing else writes UNRESERVED, so without the poison a launch where no hart runs reads the last run's slots.
    const std::vector<uint32_t> poison_words(num_harts * slot_words, poison);
    std::vector<uint32_t> result(poison_words.size(), 0);
    write_cce_sram(device, logical_dram_core, echo_noc_base, poison_words.data(), echo_bytes);
    read_cce_sram(device, logical_dram_core, echo_noc_base, result.data(), echo_bytes);
    ASSERT_EQ(result, poison_words);

    LaunchProgram(*mesh_device, program);

    read_cce_sram(device, logical_dram_core, echo_noc_base, result.data(), echo_bytes);
    EXPECT_EQ(result, expected);
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
    auto mesh_device = open_unit_mesh();
    IDevice* device = physical_device(*mesh_device);
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
            write_gddr_view(*mesh_device, target.host_view, src_dram_offset + slot_offset, inputs[slot]);
            write_gddr_view(*mesh_device, target.host_view, dst_dram_offset + slot_offset, cleared);
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

        LaunchProgram(*mesh_device, program);

        std::vector<uint32_t> staged(cleared.size(), 0);
        read_cce_sram(device, logical_dram_core, staging_noc_addr, staged.data(), transfer_size);
        EXPECT_EQ(staged, inputs.back()) << "mimir " << target.mimir_index << " location " << target.location;

        for (uint32_t slot = 0; slot < target.num_slots; ++slot) {
            std::vector<uint32_t> output;
            read_gddr_view(
                *mesh_device,
                target.host_view,
                dst_dram_offset + host_slot_offset(target, slot),
                transfer_size,
                output);
            EXPECT_EQ(output, inputs[slot])
                << "mimir " << target.mimir_index << " location " << target.location << " slot " << slot;
        }
    }
}

class CopiesGddrWithRuntimeArgs : public ::testing::TestWithParam<CcePartition> {};

TEST_P(CopiesGddrWithRuntimeArgs, RoundTrip) {
    skip_unless_programmable_dram_package();
    const CcePartition param = GetParam();
    const uint32_t cce_index = param.cce_index;
    const bool cross = is_cross_mimir_case();
    if (!cce_partition_case_applies(param, cross)) {
        GTEST_SKIP() << "Configured descriptor does not expose CCE " << cce_index << " partition " << param.partition
                     << (cross ? " as a remote partition." : " as a local partition.");
    }
    auto mesh_device = open_unit_mesh();
    IDevice* device = physical_device(*mesh_device);
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
            write_gddr_view(*mesh_device, target.host_view, src_dram_offset + slot_offset, inputs[slot]);
            write_gddr_view(*mesh_device, target.host_view, dst_dram_offset + slot_offset, cleared);
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

        LaunchProgram(*mesh_device, program);

        for (uint32_t slot = 0; slot < target.num_slots; ++slot) {
            std::vector<uint32_t> output;
            read_gddr_view(
                *mesh_device,
                target.host_view,
                dst_dram_offset + host_slot_offset(target, slot),
                transfer_size,
                output);
            EXPECT_EQ(output, inputs[slot])
                << "mimir " << target.mimir_index << " location " << target.location << " slot " << slot;
        }
    }
}

class AllocatedDramBufferHostLoopback : public ::testing::TestWithParam<uint32_t> {};

TEST_P(AllocatedDramBufferHostLoopback, RoundTrip) {
    skip_unless_programmable_dram_package();
    auto mesh_device = open_unit_mesh();
    IDevice* device = physical_device(*mesh_device);
    ASSERT_NE(device, nullptr);
    ASSERT_EQ(device->arch(), tt::ARCH::QUASAR);

    const uint32_t dram_partition = GetParam();
    if (dram_partition >= device->num_dram_channels()) {
        GTEST_SKIP() << "Configured descriptor does not expose partition " << dram_partition << ".";
    }
    GddrBackingScope gddr_backing;

    const auto& hal = MetalContext::instance().hal();
    const uint32_t page_size = hal.get_alignment(HalMemType::DRAM);
    const uint32_t buffer_size = page_size * device->num_dram_channels();
    auto buffer = distributed::MeshBuffer::create(
        distributed::ReplicatedBufferConfig{.size = buffer_size},
        {.page_size = page_size, .buffer_type = BufferType::DRAM},
        mesh_device.get());
    ASSERT_NE(buffer, nullptr);
    for (uint32_t partition = 0; partition < device->num_dram_channels(); ++partition) {
        gddr_backing.tracker.note(partition, buffer->address(), page_size);
    }

    // GDDR keeps its contents across processes and the input is the same every run, so without the poison a dropped
    // write reads the last run's data.
    std::vector<uint32_t> poison(buffer_size / sizeof(uint32_t));
    for (std::size_t i = 0; i < poison.size(); ++i) {
        poison[i] = 0xDEADBEEFu ^ static_cast<uint32_t>(i);
    }
    slow_dispatch::WriteToBuffer(*buffer, poison);
    std::vector<uint32_t> poison_read_back;
    slow_dispatch::ReadFromBuffer(*buffer, poison_read_back);
    ASSERT_EQ(poison_read_back, poison);

    const std::size_t words_per_page = page_size / sizeof(uint32_t);
    std::vector<uint32_t> input(buffer_size / sizeof(uint32_t));
    for (uint32_t partition = 0; partition < device->num_dram_channels(); ++partition) {
        for (std::size_t i = 0; i < words_per_page; ++i) {
            input[partition * words_per_page + i] = 0xA1100000u | (partition << 8) | static_cast<uint32_t>(i);
        }
    }
    slow_dispatch::WriteToBuffer(*buffer, input);

    std::vector<uint32_t> output;
    slow_dispatch::ReadFromBuffer(*buffer, output);
    EXPECT_EQ(output, input);

    const auto expected_begin = input.begin() + dram_partition * words_per_page;
    const std::vector<uint32_t> expected(expected_begin, expected_begin + words_per_page);
    std::vector<uint32_t> channel_page;
    ASSERT_TRUE(
        slow_dispatch::ReadFromDRAMChannel(*mesh_device, dram_partition, buffer->address(), page_size, channel_page));
    EXPECT_EQ(channel_page, expected);

    buffer.reset();
}

class CopiesEveryMimirBlock : public ::testing::TestWithParam<uint32_t> {};

TEST_P(CopiesEveryMimirBlock, RoundTrip) {
    skip_unless_programmable_dram_package();
    auto mesh_device = open_unit_mesh();
    IDevice* device = physical_device(*mesh_device);
    ASSERT_NE(device, nullptr);
    ASSERT_EQ(device->arch(), tt::ARCH::QUASAR);

    const uint32_t cce_index = GetParam();
    const uint32_t num_mimirs = num_mimirs_on(device);
    if (cce_index >= num_cces_on(device) || num_mimirs == 0) {
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
        write_mimir_gddr(*mesh_device, mimir_index, src_dram_offset, inputs[mimir_index]);
        write_mimir_gddr(*mesh_device, mimir_index, dst_dram_offset, cleared);
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
        {src_dram_offset, dst_dram_offset, num_mimirs, staging_dev_addr, static_cast<uint32_t>(cleared.size())});

    LaunchProgram(*mesh_device, program);

    for (uint32_t mimir_index = 0; mimir_index < num_mimirs; ++mimir_index) {
        std::vector<uint32_t> output;
        read_mimir_gddr(*mesh_device, mimir_index, dst_dram_offset, transfer_size, output);
        EXPECT_EQ(output, inputs[mimir_index]) << "mimir " << mimir_index;
    }
}

class CopiesGddrLargerThanAlignment : public ::testing::TestWithParam<CcePartition> {};

TEST_P(CopiesGddrLargerThanAlignment, RoundTrip) {
    skip_unless_programmable_dram_package();
    const CcePartition param = GetParam();
    const uint32_t cce_index = param.cce_index;
    const bool cross = is_cross_mimir_case();
    if (!cce_partition_case_applies(param, cross)) {
        GTEST_SKIP() << "Configured descriptor does not expose CCE " << cce_index << " partition " << param.partition
                     << (cross ? " as a remote partition." : " as a local partition.");
    }
    auto mesh_device = open_unit_mesh();
    IDevice* device = physical_device(*mesh_device);
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
            write_gddr_view(*mesh_device, target.host_view, src_dram_offset + slot_offset, inputs[slot]);
            write_gddr_view(*mesh_device, target.host_view, dst_dram_offset + slot_offset, cleared);
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

        LaunchProgram(*mesh_device, program);

        for (uint32_t slot = 0; slot < target.num_slots; ++slot) {
            std::vector<uint32_t> output;
            read_gddr_view(
                *mesh_device,
                target.host_view,
                dst_dram_offset + host_slot_offset(target, slot),
                transfer_size,
                output);
            EXPECT_EQ(output, inputs[slot])
                << "mimir " << target.mimir_index << " location " << target.location << " slot " << slot;
        }
    }
}

TEST(MimirEmu, AllCcesCopyLocalMimirBlock) {
    skip_unless_programmable_dram_package();
    auto mesh_device = open_unit_mesh();
    IDevice* device = physical_device(*mesh_device);
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
        write_gddr_view(*mesh_device, local.host_view, cce_src, inputs[cce_index]);
        write_gddr_view(*mesh_device, local.host_view, cce_dst, cleared);

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
             local.slot_base});
    }

    LaunchProgram(*mesh_device, program);

    for (uint32_t cce_index = 0; cce_index < num_cces; ++cce_index) {
        std::vector<uint32_t> output;
        const uint32_t cce_dst = src_dram_offset + cce_index * 2 * transfer_size + transfer_size;
        read_gddr_view(*mesh_device, local_dram_view_for_cce(soc_desc, cce_index), cce_dst, transfer_size, output);
        EXPECT_EQ(output, inputs[cce_index]) << "cce " << cce_index;
    }
}

class WritesCrossMimirCceSram : public ::testing::TestWithParam<uint32_t> {};

TEST_P(WritesCrossMimirCceSram, WritesMagic) {
    skip_unless_programmable_dram_package();

    auto mesh_device = open_unit_mesh();
    IDevice* device = physical_device(*mesh_device);
    ASSERT_NE(device, nullptr);
    ASSERT_EQ(device->arch(), tt::ARCH::QUASAR);

    const uint32_t src_cce = GetParam();
    const uint32_t num_cces = num_cces_on(device);
    if (src_cce >= num_cces || num_mimirs_on(device) < 2) {
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

    LaunchProgram(*mesh_device, program);

    for (uint32_t dest_cce : dest_cces) {
        uint32_t result = 0;
        const CoreCoord dest_core = logical_cce_core(soc_desc_of(device), dest_cce);
        read_cce_sram(device, dest_core, noc_addr, &result, sizeof(result));
        EXPECT_EQ(result, magic_base | dest_cce) << "src cce " << src_cce << " dest cce " << dest_cce;
    }
}

class PrefetchesLocalGddrToRemoteCceSram : public ::testing::TestWithParam<uint32_t> {};

TEST_P(PrefetchesLocalGddrToRemoteCceSram, RoundTrip) {
    skip_unless_programmable_dram_package();
    auto mesh_device = open_unit_mesh();
    IDevice* device = physical_device(*mesh_device);
    ASSERT_NE(device, nullptr);
    ASSERT_EQ(device->arch(), tt::ARCH::QUASAR);

    const uint32_t src_cce = GetParam();
    const uint32_t num_cces = num_cces_on(device);
    const uint32_t local_mimir = mimir_index_for_cce(src_cce);
    if (src_cce >= num_cces || num_mimirs_on(device) < 2) {
        GTEST_SKIP() << "Configured descriptor does not expose a remote Mimir CCE.";
    }

    std::vector<uint32_t> dest_cces;
    for (uint32_t cce_index = 0; cce_index < num_cces; ++cce_index) {
        if (mimir_index_for_cce(cce_index) != local_mimir) {
            dest_cces.push_back(cce_index);
        }
    }
    if (dest_cces.size() < 2) {
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
    write_gddr_view(*mesh_device, local_view, src_gddr_offset, input);

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

    LaunchProgram(*mesh_device, program);

    for (uint32_t dest_cce : dest_cces) {
        std::vector<uint32_t> output(num_words, 0);
        const CoreCoord dest_core = logical_cce_core(soc_desc_of(device), dest_cce);
        read_cce_sram(device, dest_core, noc_addr, output.data(), transfer_size);
        EXPECT_EQ(output, input) << "src cce " << src_cce << " dest cce " << dest_cce;
    }
}

class AllHartsWriteGddr : public ::testing::TestWithParam<CcePartition> {};

TEST_P(AllHartsWriteGddr, WritesMagic) {
    skip_unless_programmable_dram_package();
    const CcePartition param = GetParam();
    const uint32_t cce_index = param.cce_index;
    const bool cross = is_cross_mimir_case();
    if (!cce_partition_case_applies(param, cross)) {
        GTEST_SKIP() << "Configured descriptor does not expose CCE " << cce_index << " partition " << param.partition
                     << (cross ? " as a remote partition." : " as a local partition.");
    }
    auto mesh_device = open_unit_mesh();
    IDevice* device = physical_device(*mesh_device);
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
            write_gddr_view(*mesh_device, target.host_view, dst_dram_offset + host_slot_offset(target, slot), cleared);
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

        LaunchProgram(*mesh_device, program);

        for (uint32_t slot = 0; slot < target.num_slots; ++slot) {
            std::vector<uint32_t> output;
            read_gddr_view(
                *mesh_device,
                target.host_view,
                dst_dram_offset + host_slot_offset(target, slot),
                static_cast<uint32_t>(num_harts * sizeof(uint32_t)),
                output);
            EXPECT_EQ(output, expected) << "mimir " << target.mimir_index << " location " << target.location << " slot "
                                        << slot;
        }
    }
}

INSTANTIATE_TEST_SUITE_P(MimirEmu, CceSramRoundTripThroughUnitMesh, ::testing::ValuesIn(cce_indices()), cce_param_name);

INSTANTIATE_TEST_SUITE_P(MimirEmu, HartZeroRunsDramKernel, ::testing::ValuesIn(cce_indices()), cce_param_name);

INSTANTIATE_TEST_SUITE_P(MimirEmu, AllHartsRunDramKernel, ::testing::ValuesIn(cce_indices()), cce_param_name);

INSTANTIATE_TEST_SUITE_P(MimirEmu, WritesCrossMimirCceSram, ::testing::ValuesIn(cce_indices()), cce_param_name);

INSTANTIATE_TEST_SUITE_P(
    MimirEmu, PrefetchesLocalGddrToRemoteCceSram, ::testing::ValuesIn(cce_indices()), cce_param_name);

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
    MimirEmu,
    AllocatedDramBufferHostLoopback,
    ::testing::ValuesIn(dram_view_indices()),
    [](const ::testing::TestParamInfo<uint32_t>& info) { return fmt::format("Partition{}", info.param); });

INSTANTIATE_TEST_SUITE_P(
    MimirEmuSameMimirDirect,
    CopiesGddrWithRuntimeArgs,
    ::testing::ValuesIn(cce_partition_params()),
    cce_partition_name);

INSTANTIATE_TEST_SUITE_P(
    MimirEmuCrossMimirNoc, CopiesGddrWithRuntimeArgs, ::testing::ValuesIn(cce_partition_params()), cce_partition_name);

INSTANTIATE_TEST_SUITE_P(MimirEmu, CopiesEveryMimirBlock, ::testing::ValuesIn(cce_indices()), cce_param_name);

INSTANTIATE_TEST_SUITE_P(
    MimirEmuSameMimirDirect,
    CopiesGddrLargerThanAlignment,
    ::testing::ValuesIn(cce_partition_params()),
    cce_partition_name);

INSTANTIATE_TEST_SUITE_P(
    MimirEmuCrossMimirNoc,
    CopiesGddrLargerThanAlignment,
    ::testing::ValuesIn(cce_partition_params()),
    cce_partition_name);

INSTANTIATE_TEST_SUITE_P(
    MimirEmuSameMimirDirect, AllHartsWriteGddr, ::testing::ValuesIn(cce_partition_params()), cce_partition_name);

INSTANTIATE_TEST_SUITE_P(
    MimirEmuCrossMimirNoc, AllHartsWriteGddr, ::testing::ValuesIn(cce_partition_params()), cce_partition_name);

class DramChannelsDoNotAliasThroughPublicDeviceApi : public ::testing::TestWithParam<uint32_t> {};

TEST_P(DramChannelsDoNotAliasThroughPublicDeviceApi, RoundTrip) {
    skip_unless_programmable_dram_package();
    auto mesh_device = open_unit_mesh();
    IDevice* device = physical_device(*mesh_device);
    ASSERT_NE(device, nullptr);

    const uint32_t dram_partition = GetParam();
    const uint32_t num_partitions = device->num_dram_channels();
    if (num_partitions < 2) {
        GTEST_SKIP() << "Alias check needs at least two DRAM channels; descriptor exposes " << num_partitions << ".";
    }
    if (dram_partition >= num_partitions) {
        GTEST_SKIP() << "Configured descriptor does not expose partition " << dram_partition << ".";
    }
    GddrBackingScope gddr_backing;
    const uint32_t other_partition = (dram_partition + num_partitions / 2) % num_partitions;
    // WriteToDRAMChannel rejects the firmware-reserved region below the allocator base.
    const uint32_t dram_offset =
        static_cast<uint32_t>(device->allocator()->get_base_allocator_addr(HalMemType::DRAM)) + kDramOffset;
    gddr_backing.tracker.note(dram_partition, dram_offset, sizeof(uint32_t));
    gddr_backing.tracker.note(other_partition, dram_offset, sizeof(uint32_t));
    std::vector<uint32_t> written{0xAAAA1111 + dram_partition};
    std::vector<uint32_t> other_written{0xBBBB2222 + other_partition};
    ASSERT_TRUE(slow_dispatch::WriteToDRAMChannel(*mesh_device, other_partition, dram_offset, other_written));
    ASSERT_TRUE(slow_dispatch::WriteToDRAMChannel(*mesh_device, dram_partition, dram_offset, written));

    std::vector<uint32_t> read_back;
    std::vector<uint32_t> other_read_back;
    ASSERT_TRUE(
        slow_dispatch::ReadFromDRAMChannel(*mesh_device, dram_partition, dram_offset, sizeof(uint32_t), read_back));
    ASSERT_TRUE(slow_dispatch::ReadFromDRAMChannel(
        *mesh_device, other_partition, dram_offset, sizeof(uint32_t), other_read_back));

    EXPECT_EQ(read_back, written);
    EXPECT_EQ(other_read_back, other_written);
}

INSTANTIATE_TEST_SUITE_P(
    MimirEmu,
    DramChannelsDoNotAliasThroughPublicDeviceApi,
    ::testing::ValuesIn(dram_view_indices()),
    [](const ::testing::TestParamInfo<uint32_t>& info) { return fmt::format("Partition{}", info.param); });

}  // namespace
}  // namespace tt::tt_metal
