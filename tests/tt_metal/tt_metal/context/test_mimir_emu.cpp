// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <fmt/format.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <cstdlib>
#include <memory>
#include <numeric>
#include <set>
#include <string>
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
#include "tt_metal/tt_metal/dram/mimir_idma_copy.hpp"
#include "tt_metal/tt_metal/test_kernels/cce/cce_idma_copy_abi.h"
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
    ASSERT_EQ(num_harts, static_cast<uint32_t>(MEM_CCE_LOCAL_HARTS));

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

// MIM-407 request VC of an own-TL1 -> GDDR iDMA copy: the destination tile's column.
uint32_t idma_gddr_write_backend(const GddrTarget& target, uint32_t dram_offset) {
    const uint64_t chiplet_offset = static_cast<uint64_t>(target.slot_base) * target.slot_stride + dram_offset;
    return mimir_test::gddr_backend_for(chiplet_offset);
}

// The CopiesGddrWithCompileTimeArgs round trip on the local Mimir, moved by iDMA instead of hart loads and stores.
class IdmaCopiesGddrWithCompileTimeArgs : public ::testing::TestWithParam<CcePartition> {};

TEST_P(IdmaCopiesGddrWithCompileTimeArgs, RoundTrip) {
    skip_unless_programmable_dram_package();
    const CcePartition param = GetParam();
    const uint32_t cce_index = param.cce_index;
    if (!cce_partition_case_applies(param, /*cross=*/false)) {
        GTEST_SKIP() << "Configured descriptor does not expose CCE " << cce_index << " partition " << param.partition
                     << " as a local partition.";
    }
    auto mesh_device = open_unit_mesh();
    IDevice* device = physical_device(*mesh_device);
    ASSERT_NE(device, nullptr);
    ASSERT_EQ(device->arch(), tt::ARCH::QUASAR);
    const metal_SocDescriptor& soc_desc = soc_desc_of(device);
    const GddrSweep sweep = gddr_sweep_from_soc(device);
    ASSERT_NE(sweep.num_slots, 0u) << "dram_view_size must hold at least one GDDR slot";
    const GddrTarget target = gddr_target_for_view(soc_desc, param.partition, sweep);
    GddrBackingScope gddr_backing;

    const auto& hal = MetalContext::instance().hal();
    const uint32_t transfer_size = hal.get_alignment(HalMemType::DRAM);
    const uint32_t src_dram_offset = hal.get_dev_addr(HalDramMemAddrType::UNRESERVED);
    const uint32_t dst_dram_offset = src_dram_offset + transfer_size;
    const uint32_t staging_dev_addr = hal.get_dev_addr(HalProgrammableCoreType::DRAM, HalL1MemAddrType::UNRESERVED);
    const uint64_t staging_noc_addr = hal.get_dev_noc_addr(HalProgrammableCoreType::DRAM, HalL1MemAddrType::UNRESERVED);
    const CoreCoord logical_dram_core = logical_cce_core(soc_desc, cce_index);
    const std::vector<uint32_t> cleared(transfer_size / sizeof(uint32_t), 0);
    // TL1 and GDDR keep their contents across processes and the inputs are the same every run, so without the poison a
    // read that never lands would leave the last run's staging for the write to copy.
    const std::vector<uint32_t> staging_poison(transfer_size / sizeof(uint32_t), 0xDEADBEEFu);

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
    write_cce_sram(device, logical_dram_core, staging_noc_addr, staging_poison.data(), transfer_size);

    Program program = CreateProgram();
    CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/cce/cce_idma_gddr_round_trip.cpp",
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
                cce_index,
                idma_gddr_write_backend(target, dst_dram_offset)}});

    LaunchProgram(*mesh_device, program);

    std::vector<uint32_t> staged(cleared.size(), 0);
    read_cce_sram(device, logical_dram_core, staging_noc_addr, staged.data(), transfer_size);
    EXPECT_EQ(staged, inputs.back()) << "mimir " << target.mimir_index << " location " << target.location;

    for (uint32_t slot = 0; slot < target.num_slots; ++slot) {
        std::vector<uint32_t> output;
        read_gddr_view(
            *mesh_device, target.host_view, dst_dram_offset + host_slot_offset(target, slot), transfer_size, output);
        EXPECT_EQ(output, inputs[slot]) << "mimir " << target.mimir_index << " location " << target.location << " slot "
                                        << slot;
    }
}

// Rdcycle cycles per kernel poll, as kPollBudgetCycles in dram/mimir_dm_common.hpp.
constexpr uint64_t kIdmaPollBudgetCycles = 1ull << 22;
// The status line sits past the largest staging block (kEmuGddrBackingPage) at L1 UNRESERVED.
constexpr uint32_t kIdmaStatusOffsetFromStaging = static_cast<uint32_t>(kEmuGddrBackingPage);

// One cce_idma_copy kModeTridBatch copy on hart 0 of a CCE, between its TL1 staging block at L1 UNRESERVED and a GDDR
// chiplet offset of its local Mimir, in commands of at most kMaxBytesInPacket. A read into the staging is
// host-verified: the hart polls the XOR of the staging against expected_xor. A write to GDDR is kernel-verified: the
// hart polls the destination against the staging.
struct IdmaStagingCopy {
    uint32_t cce_index = 0;
    uint32_t mimir_index = 0;
    bool reads_gddr = false;
    uint64_t chiplet_offset = 0;
    uint32_t num_bytes = 0;
    uint32_t backend_vc = 0;
    uint32_t expected_xor = 0;
};

uint32_t idma_command_bytes(const IdmaStagingCopy& copy) {
    return std::min(copy.num_bytes, cce_idma_copy_abi::kMaxBytesInPacket);
}

// The trid is the CCE index: whether CCEs share pending counters is not known.
std::vector<uint32_t> idma_staging_copy_args(const IdmaStagingCopy& copy, uint32_t staging_dev_addr) {
    namespace abi = cce_idma_copy_abi;
    const uint32_t len_bytes = idma_command_bytes(copy);
    const mimir_test::IdmaOperand gddr{abi::kKindGddr, copy.mimir_index, copy.chiplet_offset};
    const mimir_test::IdmaOperand staging{abi::kKindOwnTl1, copy.cce_index, staging_dev_addr};
    return mimir_test::idma_copy_args(
        {.mode = abi::kModeTridBatch,
         .backend_vc = copy.backend_vc,
         .status_offset = staging_dev_addr + kIdmaStatusOffsetFromStaging,
         .len_bytes = len_bytes,
         .poll_budget_cycles = kIdmaPollBudgetCycles,
         .source = copy.reads_gddr ? gddr : staging,
         .destination = copy.reads_gddr ? staging : gddr,
         .expected_xor = copy.expected_xor,
         .num_elements = copy.num_bytes / len_bytes,
         .source_stride_bytes = len_bytes,
         .trid = copy.cce_index,
         .destination_stride_bytes = len_bytes,
         .host_verified = copy.reads_gddr});
}

// Runs every copy in one blocking Program and returns their status lines in copy order. Every kernel poll is bounded,
// so a copy that never completes ends with a timeout code instead of holding the launch.
std::vector<std::vector<uint32_t>> run_idma_staging_copies(
    distributed::MeshDevice& mesh, IDevice* device, const std::vector<IdmaStagingCopy>& copies) {
    namespace abi = cce_idma_copy_abi;
    const auto& hal = MetalContext::instance().hal();
    const uint32_t staging_dev_addr = hal.get_dev_addr(HalProgrammableCoreType::DRAM, HalL1MemAddrType::UNRESERVED);
    const uint64_t status_noc_addr = hal.get_dev_noc_addr(HalProgrammableCoreType::DRAM, HalL1MemAddrType::UNRESERVED) +
                                     kIdmaStatusOffsetFromStaging;
    const uint32_t status_bytes = abi::kStatusWords * static_cast<uint32_t>(sizeof(uint32_t));
    const std::vector<uint32_t> status_poison(abi::kStatusWords, abi::kStatusPoison);
    Program program = CreateProgram();
    for (const IdmaStagingCopy& copy : copies) {
        const CoreCoord logical_dram_core = logical_cce_core(soc_desc_of(device), copy.cce_index);
        write_cce_sram(device, logical_dram_core, status_noc_addr, status_poison.data(), status_bytes);
        const KernelHandle kernel =
            CreateKernel(program, mimir_test::kIdmaCopyKernel, logical_dram_core, DramConfig{.noc = NOC::NOC_0});
        SetRuntimeArgs(program, kernel, logical_dram_core, idma_staging_copy_args(copy, staging_dev_addr));
    }

    LaunchProgram(mesh, program);

    std::vector<std::vector<uint32_t>> statuses;
    for (const IdmaStagingCopy& copy : copies) {
        std::vector<uint32_t> status(abi::kStatusWords, 0);
        read_cce_sram(
            device,
            logical_cce_core(soc_desc_of(device), copy.cce_index),
            status_noc_addr,
            status.data(),
            status_bytes);
        statuses.push_back(status);
    }
    return statuses;
}

// Every copy wrote its status line with its success code after issuing every command.
::testing::AssertionResult idma_copies_completed(
    const std::vector<IdmaStagingCopy>& copies, const std::vector<std::vector<uint32_t>>& statuses) {
    namespace abi = cce_idma_copy_abi;
    std::string failures;
    for (std::size_t index = 0; index < copies.size(); ++index) {
        const IdmaStagingCopy& copy = copies[index];
        const std::vector<uint32_t>& status = statuses[index];
        const uint32_t expected_code = copy.reads_gddr ? abi::kCodeAcked : abi::kCodeLanded;
        const uint32_t expected_commands = copy.num_bytes / idma_command_bytes(copy);
        if (status[abi::kStatusMagic] != abi::kDoneMagic || status[abi::kStatusCode] != expected_code ||
            status[abi::kStatusElementsIssued] != expected_commands) {
            failures += fmt::format(
                "cce {} {}: magic 0x{:08x}, code {} (expected {}), {} of {} commands issued, last poll {} ready {}, "
                "pending high water {}, kernel mismatches {} (first word {} observed 0x{:08x} expected 0x{:08x})\n",
                copy.cce_index,
                copy.reads_gddr ? "GDDR to staging" : "staging to GDDR",
                status[abi::kStatusMagic],
                status[abi::kStatusCode],
                expected_code,
                status[abi::kStatusElementsIssued],
                expected_commands,
                status[abi::kStatusLastPoll],
                status[abi::kStatusLastPollReady],
                status[abi::kStatusPendingHighWater],
                status[abi::kStatusMismatchedWords],
                status[abi::kStatusFirstMismatchWord],
                status[abi::kStatusObserved],
                status[abi::kStatusExpected]);
        }
    }
    if (failures.empty()) {
        return ::testing::AssertionSuccess();
    }
    return ::testing::AssertionFailure()
           << failures
           << "A timeout code may leave a command in flight; start a fresh emulator session before the next run "
              "(MIM-601).";
}

// GDDR -> own TL1 staging -> GDDR on the local Mimir by iDMA, one slot at a time, `pages_per_copy` DRAM alignments per
// copy (at most kEmuGddrBackingPage). Each word holds `value_tag`, the CCE, Mimir, location and slot in its high half
// and its index in its low half, so no two words of a run are equal.
void idma_round_trip_through_staging(const CcePartition& param, uint32_t pages_per_copy, uint32_t value_tag) {
    skip_unless_programmable_dram_package();
    const uint32_t cce_index = param.cce_index;
    if (!cce_partition_case_applies(param, /*cross=*/false)) {
        GTEST_SKIP() << "Configured descriptor does not expose CCE " << cce_index << " partition " << param.partition
                     << " as a local partition.";
    }
    auto mesh_device = open_unit_mesh();
    IDevice* device = physical_device(*mesh_device);
    ASSERT_NE(device, nullptr);
    ASSERT_EQ(device->arch(), tt::ARCH::QUASAR);
    const metal_SocDescriptor& soc_desc = soc_desc_of(device);
    const GddrSweep sweep = gddr_sweep_from_soc(device);
    ASSERT_NE(sweep.num_slots, 0u) << "dram_view_size must hold at least one GDDR slot";
    const GddrTarget target = gddr_target_for_view(soc_desc, param.partition, sweep);
    GddrBackingScope gddr_backing;

    const auto& hal = MetalContext::instance().hal();
    const uint32_t transfer_size =
        std::min(hal.get_alignment(HalMemType::DRAM) * pages_per_copy, static_cast<uint32_t>(kEmuGddrBackingPage));
    const uint32_t src_dram_offset = hal.get_dev_addr(HalDramMemAddrType::UNRESERVED);
    const uint32_t dst_dram_offset = src_dram_offset + transfer_size;
    const uint64_t staging_noc_addr = hal.get_dev_noc_addr(HalProgrammableCoreType::DRAM, HalL1MemAddrType::UNRESERVED);
    const CoreCoord logical_dram_core = logical_cce_core(soc_desc, cce_index);
    const std::vector<uint32_t> cleared(transfer_size / sizeof(uint32_t), 0);
    // TL1 and GDDR keep their contents across processes and the inputs are the same every run, so without the poison a
    // read that never lands would leave the last run's staging for the write to copy.
    const std::vector<uint32_t> staging_poison(transfer_size / sizeof(uint32_t), 0xDEADBEEFu);

    for (uint32_t slot = 0; slot < target.num_slots; ++slot) {
        const uint32_t location_tag =
            (cce_index << 12) | (target.mimir_index << 8) | (target.location << 4) | (slot & 0xFu);
        std::vector<uint32_t> input(transfer_size / sizeof(uint32_t));
        for (std::size_t i = 0; i < input.size(); ++i) {
            input[i] = value_tag ^ (location_tag << 16) ^ static_cast<uint32_t>(i);
        }
        const uint64_t slot_offset = host_slot_offset(target, slot);
        const uint64_t slot_chiplet_offset = static_cast<uint64_t>(target.slot_base + slot) * target.slot_stride;
        write_gddr_view(*mesh_device, target.host_view, src_dram_offset + slot_offset, input);
        write_gddr_view(*mesh_device, target.host_view, dst_dram_offset + slot_offset, cleared);
        write_cce_sram(device, logical_dram_core, staging_noc_addr, staging_poison.data(), transfer_size);
        const std::string label =
            fmt::format("mimir {} location {} slot {}", target.mimir_index, target.location, slot);

        const std::vector<IdmaStagingCopy> reads = {IdmaStagingCopy{
            .cce_index = cce_index,
            .mimir_index = target.mimir_index,
            .reads_gddr = true,
            .chiplet_offset = slot_chiplet_offset + src_dram_offset,
            .num_bytes = transfer_size,
            .backend_vc = cce_idma_copy_abi::kBackendFromRemote,
            .expected_xor = std::accumulate(input.begin(), input.end(), 0u, std::bit_xor<uint32_t>())}};
        const std::vector<std::vector<uint32_t>> read_statuses = run_idma_staging_copies(*mesh_device, device, reads);
        std::vector<uint32_t> staged(input.size(), 0);
        read_cce_sram(device, logical_dram_core, staging_noc_addr, staged.data(), transfer_size);
        EXPECT_EQ(staged, input) << label << " staging";
        ASSERT_TRUE(idma_copies_completed(reads, read_statuses)) << label;

        const std::vector<IdmaStagingCopy> writes = {IdmaStagingCopy{
            .cce_index = cce_index,
            .mimir_index = target.mimir_index,
            .chiplet_offset = slot_chiplet_offset + dst_dram_offset,
            .num_bytes = transfer_size,
            .backend_vc = idma_gddr_write_backend(target, static_cast<uint32_t>(slot_offset) + dst_dram_offset)}};
        const std::vector<std::vector<uint32_t>> write_statuses = run_idma_staging_copies(*mesh_device, device, writes);
        std::vector<uint32_t> output;
        read_gddr_view(*mesh_device, target.host_view, dst_dram_offset + slot_offset, transfer_size, output);
        EXPECT_EQ(output, input) << label;
        ASSERT_TRUE(idma_copies_completed(writes, write_statuses)) << label;
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

// CopiesGddrWithRuntimeArgs on the local Mimir through cce_idma_copy, whose operands all arrive as runtime args.
class IdmaCopiesGddrWithRuntimeArgs : public ::testing::TestWithParam<CcePartition> {};

TEST_P(IdmaCopiesGddrWithRuntimeArgs, RoundTrip) {
    idma_round_trip_through_staging(GetParam(), /*pages_per_copy=*/1, 0xD15C0000u);
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

// CopiesGddrLargerThanAlignment on the local Mimir through cce_idma_copy.
class IdmaCopiesGddrLargerThanAlignment : public ::testing::TestWithParam<CcePartition> {};

TEST_P(IdmaCopiesGddrLargerThanAlignment, RoundTrip) {
    constexpr uint32_t kPagesPerBank = 16;
    idma_round_trip_through_staging(GetParam(), kPagesPerBank, 0x1A6E0000u);
}

// Every CCE copies one block of its local view through its TL1 staging by iDMA, all CCEs in one Program per direction.
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
    const uint64_t staging_noc_addr = hal.get_dev_noc_addr(HalProgrammableCoreType::DRAM, HalL1MemAddrType::UNRESERVED);

    const metal_SocDescriptor& soc_desc = soc_desc_of(device);
    std::vector<std::vector<uint32_t>> inputs(num_cces, std::vector<uint32_t>(transfer_size / sizeof(uint32_t)));
    std::vector<uint32_t> cleared(transfer_size / sizeof(uint32_t), 0);
    // TL1 and GDDR keep their contents across processes and the inputs are the same every run, so without the poison a
    // read that never lands would leave the last run's staging for the write to copy.
    const std::vector<uint32_t> staging_poison(transfer_size / sizeof(uint32_t), 0xDEADBEEFu);
    std::vector<IdmaStagingCopy> reads;
    std::vector<IdmaStagingCopy> writes;
    for (uint32_t cce_index = 0; cce_index < num_cces; ++cce_index) {
        const GddrTarget local = gddr_target_for_view(soc_desc, local_dram_view_for_cce(soc_desc, cce_index), sweep);
        for (std::size_t i = 0; i < inputs[cce_index].size(); ++i) {
            inputs[cce_index][i] = 0xB07C0000u | (cce_index << 8) | static_cast<uint32_t>(i);
        }
        const uint32_t cce_src = src_dram_offset + cce_index * 2 * transfer_size;
        const uint32_t cce_dst = cce_src + transfer_size;
        write_gddr_view(*mesh_device, local.host_view, cce_src, inputs[cce_index]);
        write_gddr_view(*mesh_device, local.host_view, cce_dst, cleared);
        write_cce_sram(
            device, logical_cce_core(soc_desc, cce_index), staging_noc_addr, staging_poison.data(), transfer_size);

        const uint64_t view_chiplet_offset = static_cast<uint64_t>(local.slot_base) * local.slot_stride;
        reads.push_back(IdmaStagingCopy{
            .cce_index = cce_index,
            .mimir_index = local.mimir_index,
            .reads_gddr = true,
            .chiplet_offset = view_chiplet_offset + cce_src,
            .num_bytes = transfer_size,
            .backend_vc = cce_idma_copy_abi::kBackendFromRemote,
            .expected_xor =
                std::accumulate(inputs[cce_index].begin(), inputs[cce_index].end(), 0u, std::bit_xor<uint32_t>())});
        writes.push_back(IdmaStagingCopy{
            .cce_index = cce_index,
            .mimir_index = local.mimir_index,
            .chiplet_offset = view_chiplet_offset + cce_dst,
            .num_bytes = transfer_size,
            .backend_vc = idma_gddr_write_backend(local, cce_dst)});
    }

    const std::vector<std::vector<uint32_t>> read_statuses = run_idma_staging_copies(*mesh_device, device, reads);
    for (uint32_t cce_index = 0; cce_index < num_cces; ++cce_index) {
        std::vector<uint32_t> staged(cleared.size(), 0);
        read_cce_sram(device, logical_cce_core(soc_desc, cce_index), staging_noc_addr, staged.data(), transfer_size);
        EXPECT_EQ(staged, inputs[cce_index]) << "cce " << cce_index << " staging";
    }
    ASSERT_TRUE(idma_copies_completed(reads, read_statuses));

    const std::vector<std::vector<uint32_t>> write_statuses = run_idma_staging_copies(*mesh_device, device, writes);
    for (uint32_t cce_index = 0; cce_index < num_cces; ++cce_index) {
        std::vector<uint32_t> output;
        const uint32_t cce_dst = src_dram_offset + cce_index * 2 * transfer_size + transfer_size;
        read_gddr_view(*mesh_device, local_dram_view_for_cce(soc_desc, cce_index), cce_dst, transfer_size, output);
        EXPECT_EQ(output, inputs[cce_index]) << "cce " << cce_index;
    }
    EXPECT_TRUE(idma_copies_completed(writes, write_statuses));
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
    ASSERT_EQ(num_harts, static_cast<uint32_t>(MEM_CCE_LOCAL_HARTS));

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

// AllHartsWriteGddr on the local Mimir, with each hart's word copied by iDMA instead of a hart store. The value still
// reaches the hart only as a compile-time arg.
class AllHartsIdmaWriteGddr : public ::testing::TestWithParam<CcePartition> {};

TEST_P(AllHartsIdmaWriteGddr, WritesMagic) {
    skip_unless_programmable_dram_package();
    const CcePartition param = GetParam();
    const uint32_t cce_index = param.cce_index;
    if (!cce_partition_case_applies(param, /*cross=*/false)) {
        GTEST_SKIP() << "Configured descriptor does not expose CCE " << cce_index << " partition " << param.partition
                     << " as a local partition.";
    }
    auto mesh_device = open_unit_mesh();
    IDevice* device = physical_device(*mesh_device);
    ASSERT_NE(device, nullptr);
    ASSERT_EQ(device->arch(), tt::ARCH::QUASAR);
    const metal_SocDescriptor& soc_desc = soc_desc_of(device);
    const GddrSweep sweep = gddr_sweep_from_soc(device);
    ASSERT_NE(sweep.num_slots, 0u) << "dram_view_size must hold at least one GDDR slot";
    const GddrTarget target = gddr_target_for_view(soc_desc, param.partition, sweep);
    GddrBackingScope gddr_backing;

    const auto& hal = MetalContext::instance().hal();
    const uint32_t num_harts = hal.get_num_risc_processors(HalProgrammableCoreType::DRAM);
    ASSERT_EQ(num_harts, static_cast<uint32_t>(MEM_CCE_LOCAL_HARTS));

    const uint32_t dst_dram_offset = hal.get_dev_addr(HalDramMemAddrType::UNRESERVED);
    const uint32_t staging_dev_base = hal.get_dev_addr(HalProgrammableCoreType::DRAM, HalL1MemAddrType::UNRESERVED);
    const uint64_t staging_noc_base = hal.get_dev_noc_addr(HalProgrammableCoreType::DRAM, HalL1MemAddrType::UNRESERVED);
    const CoreCoord logical_dram_core = logical_cce_core(soc_desc, cce_index);
    const uint32_t backend_vc = idma_gddr_write_backend(target, dst_dram_offset);
    const std::vector<uint32_t> cleared(num_harts, 0);
    // The values are the same every run and TL1 keeps its contents across processes, so without the poison an iDMA that
    // read the staging word before the hart's store landed would copy the last run's value.
    const std::vector<uint32_t> staging_poison(num_harts, 0xDEADBEEFu);

    for (uint32_t slot = 0; slot < target.num_slots; ++slot) {
        write_gddr_view(*mesh_device, target.host_view, dst_dram_offset + host_slot_offset(target, slot), cleared);
    }
    write_cce_sram(
        device,
        logical_dram_core,
        staging_noc_base,
        staging_poison.data(),
        static_cast<uint32_t>(num_harts * sizeof(uint32_t)));

    Program program = CreateProgram();
    std::vector<uint32_t> expected(num_harts, 0);
    for (uint32_t hart = 0; hart < num_harts; hart++) {
        expected[hart] = 0xA11B0000u | (cce_index << 12) | (target.mimir_index << 8) | (target.location << 4) | hart;
        CreateKernel(
            program,
            "tests/tt_metal/tt_metal/test_kernels/cce/cce_idma_gddr_write_one_uint32.cpp",
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
                    cce_index,
                    backend_vc,
                    hart}});
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

INSTANTIATE_TEST_SUITE_P(MimirEmu, CceSramRoundTripThroughUnitMesh, ::testing::ValuesIn(cce_indices()), cce_param_name);

INSTANTIATE_TEST_SUITE_P(MimirEmu, HartZeroRunsDramKernel, ::testing::ValuesIn(cce_indices()), cce_param_name);

INSTANTIATE_TEST_SUITE_P(MimirEmu, AllHartsRunDramKernel, ::testing::ValuesIn(cce_indices()), cce_param_name);

INSTANTIATE_TEST_SUITE_P(MimirEmu, WritesCrossMimirCceSram, ::testing::ValuesIn(cce_indices()), cce_param_name);

INSTANTIATE_TEST_SUITE_P(
    MimirEmu, PrefetchesLocalGddrToRemoteCceSram, ::testing::ValuesIn(cce_indices()), cce_param_name);

INSTANTIATE_TEST_SUITE_P(
    MimirEmuSameMimirDirect,
    IdmaCopiesGddrWithCompileTimeArgs,
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
    IdmaCopiesGddrWithRuntimeArgs,
    ::testing::ValuesIn(cce_partition_params()),
    cce_partition_name);

INSTANTIATE_TEST_SUITE_P(
    MimirEmuCrossMimirNoc, CopiesGddrWithRuntimeArgs, ::testing::ValuesIn(cce_partition_params()), cce_partition_name);

INSTANTIATE_TEST_SUITE_P(MimirEmu, CopiesEveryMimirBlock, ::testing::ValuesIn(cce_indices()), cce_param_name);

INSTANTIATE_TEST_SUITE_P(
    MimirEmuSameMimirDirect,
    IdmaCopiesGddrLargerThanAlignment,
    ::testing::ValuesIn(cce_partition_params()),
    cce_partition_name);

INSTANTIATE_TEST_SUITE_P(
    MimirEmuCrossMimirNoc,
    CopiesGddrLargerThanAlignment,
    ::testing::ValuesIn(cce_partition_params()),
    cce_partition_name);

INSTANTIATE_TEST_SUITE_P(
    MimirEmuSameMimirDirect, AllHartsIdmaWriteGddr, ::testing::ValuesIn(cce_partition_params()), cce_partition_name);

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
