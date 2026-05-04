// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <cstdint>
#include <vector>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/program.hpp>
#include <umd/device/types/arch.hpp>

#include "device_fixture.hpp"
#include "impl/context/metal_context.hpp"
#include "impl/kernels/kernel.hpp"
#include "llrt/hal.hpp"
#include "llrt/tt_cluster.hpp"

using namespace tt;
using namespace tt::tt_metal;

// Smoke test: run a single DRAM kernel that writes a compile-time constant to L1,
// then read it back via the host and verify.
TEST_F(BlackholeSingleCardFixture, DramKernelWriteToL1) {
    const auto& hal = MetalContext::instance().hal();
    if (!hal.has_programmable_core_type(HalProgrammableCoreType::DRAM)) {
        GTEST_SKIP() << "DRAM programmable cores not enabled";
    }

    constexpr uint32_t kMagicValue = 0xDEADBEEF;

    auto mesh_device = devices_[0];
    auto& cq = mesh_device->mesh_command_queue();
    auto zero_coord = distributed::MeshCoordinate(0, 0);
    auto device_range = distributed::MeshCoordinateRange(zero_coord);

    // Pick the first logical DRAM worker core (bank=0, subchannel=0).
    CoreCoord logical_dram_core{0, 0};
    auto virtual_dram_core = mesh_device->virtual_core_from_logical_core(logical_dram_core, CoreType::DRAM);

    uint64_t l1_noc_offset = hal.get_l1_noc_offset(HalProgrammableCoreType::DRAM);
    uint32_t result_l1_addr = hal.get_dev_addr(HalProgrammableCoreType::DRAM, HalL1MemAddrType::UNRESERVED);

    distributed::MeshWorkload workload;
    Program program = CreateProgram();
    workload.add_program(device_range, std::move(program));
    auto& prog = workload.get_programs().at(device_range);

    CreateKernel(
        prog,
        "tests/tt_metal/tt_metal/test_kernels/misc/dram_write_one_uint32.cpp",
        logical_dram_core,
        DramConfig{
            .noc = NOC::NOC_0,
            .compile_args = {result_l1_addr, kMagicValue},
        });

    distributed::EnqueueMeshWorkload(cq, workload, false);
    distributed::Finish(cq);

    // Read back from DRAM core L1 (requires 64-bit NOC offset, so use cluster API directly).
    uint64_t read_addr = static_cast<uint64_t>(result_l1_addr) + l1_noc_offset;
    std::vector<uint32_t> result(1, 0);
    MetalContext::instance().get_cluster().read_core(
        result.data(), sizeof(uint32_t), tt_cxy_pair(mesh_device->build_id(), virtual_dram_core), read_addr);

    EXPECT_EQ(result[0], kMagicValue);
}

// Run the same kernel across multiple DRAM cores.
TEST_F(BlackholeSingleCardFixture, DramKernelOnMultipleCores) {
    const auto& hal = MetalContext::instance().hal();
    if (!hal.has_programmable_core_type(HalProgrammableCoreType::DRAM)) {
        GTEST_SKIP() << "DRAM programmable cores not enabled";
    }

    constexpr uint32_t kMagicBase = 0xCAFE0000;

    auto mesh_device = devices_[0];
    auto& cq = mesh_device->mesh_command_queue();
    auto zero_coord = distributed::MeshCoordinate(0, 0);
    auto device_range = distributed::MeshCoordinateRange(zero_coord);
    uint32_t result_l1_addr = hal.get_dev_addr(HalProgrammableCoreType::DRAM, HalL1MemAddrType::UNRESERVED);
    uint64_t l1_noc_offset = hal.get_l1_noc_offset(HalProgrammableCoreType::DRAM);

    // DRAM compute grid: x = num_banks, y = num_endpoints_per_bank.
    const auto& soc_desc = MetalContext::instance().get_cluster().get_soc_desc(mesh_device->build_id());
    auto dram_compute_grid = soc_desc.get_dram_compute_grid_size();
    uint32_t num_banks = std::min(static_cast<size_t>(dram_compute_grid.x), static_cast<size_t>(4));
    uint32_t num_endpoints = dram_compute_grid.y;

    for (uint32_t row = 0; row < num_endpoints; row++) {
        for (uint32_t col = 0; col < num_banks; col++) {
            CoreCoord logical_dram_core{col, row};
            auto virtual_dram_core = mesh_device->virtual_core_from_logical_core(logical_dram_core, CoreType::DRAM);
            uint32_t expected_value = kMagicBase + (row * num_banks) + col;

            distributed::MeshWorkload workload;
            Program program = CreateProgram();
            workload.add_program(device_range, std::move(program));
            auto& prog = workload.get_programs().at(device_range);

            CreateKernel(
                prog,
                "tests/tt_metal/tt_metal/test_kernels/misc/dram_write_one_uint32.cpp",
                logical_dram_core,
                DramConfig{
                    .noc = NOC::NOC_0,
                    .compile_args = {result_l1_addr, expected_value},
                });

            distributed::EnqueueMeshWorkload(cq, workload, false);
            distributed::Finish(cq);

            // Read back from DRAM core L1 (requires 64-bit NOC offset, so use cluster API directly).
            uint64_t read_addr = static_cast<uint64_t>(result_l1_addr) + l1_noc_offset;
            std::vector<uint32_t> result(1, 0);
            MetalContext::instance().get_cluster().read_core(
                result.data(), sizeof(uint32_t), tt_cxy_pair(mesh_device->build_id(), virtual_dram_core), read_addr);

            EXPECT_EQ(result[0], expected_value) << "Failed for DRAM core (bank=" << col << ", endpoint=" << row << ")";
        }
    }
}

// Test Tensix reading from DRISC L1 using a DRISC<->Tensix semaphore
// handshake:
//  - DRISC enters stream mode, seeds DRISC L1, and signals readiness.
//  - Tensix waits for readiness, reads DRISC L1, then signals done.
//  - DRISC waits for done and restores NOC2AXI.
TEST_F(BlackholeSingleCardFixture, DramKernelTensixReadFromDRISCL1) {
    const auto& hal = MetalContext::instance().hal();
    if (!hal.has_programmable_core_type(HalProgrammableCoreType::DRAM)) {
        GTEST_SKIP() << "DRAM programmable cores not enabled";
    }

    constexpr uint32_t kMagicValue = 0xCAFEBABE;
    auto mesh_device = devices_[0];
    auto* device = mesh_device->get_devices()[0];
    auto zero_coord = distributed::MeshCoordinate(0, 0);
    auto device_range = distributed::MeshCoordinateRange(zero_coord);

    CoreCoord logical_core_drisc{0, 0};
    CoreCoord logical_core_tensix{0, 0};
    CoreCoord tensix_virtual = device->virtual_core_from_logical_core(logical_core_tensix, CoreType::WORKER);
    CoreCoord drisc_virtual = device->virtual_core_from_logical_core(logical_core_drisc, CoreType::DRAM);

    uint32_t drisc_l1_addr = hal.get_dev_addr(HalProgrammableCoreType::DRAM, HalL1MemAddrType::UNRESERVED);
    uint32_t tensix_l1_addr = device->allocator()->get_base_allocator_addr(HalMemType::L1);
    distributed::MeshWorkload workload;
    Program program = CreateProgram();

    // Two semaphores, one local to each core:
    //   stream_ready (Tensix-local): DRISC remote-incs to 1 to indicate stream mode is on
    //   tensix_done (DRISC-local):  Tensix remote-incs to 1 when the read is complete
    uint32_t stream_ready_sem_id = CreateSemaphore(program, logical_core_tensix, 0, CoreType::WORKER);
    uint32_t tensix_done_sem_id = CreateSemaphore(program, logical_core_drisc, 0, CoreType::DRAM);

    auto drisc_kid = CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/misc/drisc_l1_transfer.cpp",
        logical_core_drisc,
        DramConfig{.noc = NOC::NOC_0, .compile_args = {drisc_l1_addr, tensix_virtual.x, tensix_virtual.y}});
    SetRuntimeArgs(program, drisc_kid, logical_core_drisc, {stream_ready_sem_id, tensix_done_sem_id, kMagicValue});

    CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/misc/tensix_read_from_drisc.cpp",
        logical_core_tensix,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::NOC_0,
            .compile_args = {
                tensix_l1_addr,
                drisc_l1_addr,
                drisc_virtual.x,
                drisc_virtual.y,
                stream_ready_sem_id,
                tensix_done_sem_id}});

    workload.add_program(device_range, std::move(program));
    distributed::EnqueueMeshWorkload(mesh_device->mesh_command_queue(), workload, false);
    distributed::Finish(mesh_device->mesh_command_queue());

    // Verify Tensix read the seeded value.
    std::vector<uint32_t> result;
    tt::tt_metal::detail::ReadFromDeviceL1(
        device, logical_core_tensix, tensix_l1_addr, sizeof(kMagicValue), result, CoreType::WORKER);
    log_info(LogTest, "Tensix L1 result: 0x{:X} (expected: 0x{:X})", result[0], kMagicValue);
    EXPECT_EQ(result[0], kMagicValue) << "Tensix should have read the value from DRISC L1";
}

// Test DRISC reading from Tensix L1
// Host writes magic value to Tensix L1, then DRISC reads it into DRISC L1
TEST_F(BlackholeSingleCardFixture, DramKernelDRISCReadFromTensixL1) {
    const auto& hal = MetalContext::instance().hal();
    if (!hal.has_programmable_core_type(HalProgrammableCoreType::DRAM)) {
        GTEST_SKIP() << "DRAM programmable cores not enabled";
    }

    constexpr uint32_t kMagicValue = 0xDEADBEEF;
    auto mesh_device = devices_[0];
    auto* device = mesh_device->get_devices()[0];
    auto zero_coord = distributed::MeshCoordinate(0, 0);
    auto device_range = distributed::MeshCoordinateRange(zero_coord);

    CoreCoord logical_core_drisc{0, 0};
    CoreCoord logical_core_tensix{0, 0};
    CoreCoord tensix_virtual = device->virtual_core_from_logical_core(logical_core_tensix, CoreType::WORKER);
    CoreCoord drisc_virtual = device->virtual_core_from_logical_core(logical_core_drisc, CoreType::DRAM);

    uint32_t l1_unreserved_base_tensix = device->allocator()->get_base_allocator_addr(HalMemType::L1);
    uint32_t l1_unreserved_base_drisc = hal.get_dev_addr(HalProgrammableCoreType::DRAM, HalL1MemAddrType::UNRESERVED);

    // Host writes magic value to Tensix L1
    std::vector<uint32_t> write_data = {kMagicValue};
    tt::tt_metal::detail::WriteToDeviceL1(
        device, logical_core_tensix, l1_unreserved_base_tensix, write_data, CoreType::WORKER);

    distributed::MeshWorkload workload;
    Program program = CreateProgram();

    // DRISC kernel reads from Tensix L1 into DRISC L1 and restores NOC2AXI at the end.
    auto kid = CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/misc/drisc_l1_transfer.cpp",
        logical_core_drisc,
        DramConfig{
            .noc = NOC::NOC_0,
            .compile_args = {l1_unreserved_base_drisc, tensix_virtual.x, tensix_virtual.y},
            .defines = {{"MODE_TENSIX_TO_DRISC", "1"}}});
    SetRuntimeArgs(program, kid, logical_core_drisc, {l1_unreserved_base_tensix});

    workload.add_program(device_range, std::move(program));
    distributed::EnqueueMeshWorkload(mesh_device->mesh_command_queue(), workload, false);
    distributed::Finish(mesh_device->mesh_command_queue());

    // Verify by reading from DRISC L1
    uint64_t read_addr = hal.get_dev_noc_addr(HalProgrammableCoreType::DRAM, HalL1MemAddrType::UNRESERVED);
    std::vector<uint32_t> result(1, 0);
    MetalContext::instance().get_cluster().read_core(
        result.data(), sizeof(uint32_t), tt_cxy_pair(mesh_device->build_id(), drisc_virtual), read_addr);
    log_info(LogTest, "DRISC L1 result: 0x{:X} (expected: 0x{:X})", result[0], kMagicValue);
    EXPECT_EQ(result[0], kMagicValue) << "DRISC should have read the value from Tensix L1";
}
