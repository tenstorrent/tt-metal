// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <cstdint>
#include <vector>
#include <chrono>
#include <tt-metalium/bfloat16.hpp>
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

static double compute_bw_gbs(uint64_t total_bytes, uint64_t cycles, uint32_t clk_hz) {
    return static_cast<double>(total_bytes) * clk_hz / cycles / 1e9;
}

// Fixture for DRISC/DRAM-kernel tests
class DramKernelFixture : public BlackholeSingleCardFixture {
protected:
    void SetUp() override {
        BlackholeSingleCardFixture::SetUp();
        if (devices_.empty()) {
            return;  // parent was skipped
        }
        const auto& hal = MetalContext::instance().hal();
        if (!hal.has_programmable_core_type(HalProgrammableCoreType::DRAM)) {
            GTEST_SKIP() << "DRAM programmable cores not enabled";
        }
        mesh_device_ = devices_[0].get();
        device_ = mesh_device_->get_devices()[0];
        device_range_ = distributed::MeshCoordinateRange(distributed::MeshCoordinate(0, 0));
        drisc_l1_base_ = hal.get_dev_addr(HalProgrammableCoreType::DRAM, HalL1MemAddrType::UNRESERVED);
        drisc_l1_noc_addr_ = hal.get_dev_noc_addr(HalProgrammableCoreType::DRAM, HalL1MemAddrType::UNRESERVED);
        tensix_l1_base_ = device_->allocator()->get_base_allocator_addr(HalMemType::L1);
        dram_unreserved_size_ = hal.get_dev_size(HalDramMemAddrType::UNRESERVED);
    }

    void run_workload(Program program) {
        distributed::MeshWorkload workload;
        workload.add_program(device_range_, std::move(program));
        distributed::EnqueueMeshWorkload(mesh_device_->mesh_command_queue(), workload, false);
        distributed::Finish(mesh_device_->mesh_command_queue());
    }

    // Reads the two-word (lo, hi) timestamp written by kernels at noc_addr.
    uint64_t read_timing_cycles(CoreCoord vcore, uint64_t noc_addr) {
        std::vector<uint32_t> t(2);
        MetalContext::instance().get_cluster().read_core(
            t.data(), sizeof(uint64_t), tt_cxy_pair(mesh_device_->build_id(), vcore), noc_addr);
        return (static_cast<uint64_t>(t[1]) << 32) | t[0];
    }

    distributed::MeshDevice* mesh_device_{};
    IDevice* device_{};
    distributed::MeshCoordinateRange device_range_{distributed::MeshCoordinate(0, 0)};
    uint32_t drisc_l1_base_{};
    uint64_t drisc_l1_noc_addr_{};
    uint32_t tensix_l1_base_{};
    uint32_t dram_unreserved_size_{};
};

class DramKernelDRISCBWFixture : public DramKernelFixture, public testing::WithParamInterface<uint32_t> {};

// Smoke test: run a single DRAM kernel that writes a compile-time constant to L1,
// then read it back via the host and verify.
TEST_F(DramKernelFixture, DramKernelWriteToL1) {
    constexpr uint32_t kMagicValue = 0xDEADBEEF;
    // Pick the first logical DRAM worker core (bank=0, subchannel=0).
    CoreCoord logical_dram_core{0, 0};
    auto virtual_dram_core = mesh_device_->virtual_core_from_logical_core(logical_dram_core, CoreType::DRAM);

    Program program = CreateProgram();
    CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/misc/dram_write_one_uint32.cpp",
        logical_dram_core,
        DramConfig{
            .noc = NOC::NOC_0,
            .compile_args = {drisc_l1_base_, kMagicValue},
        });
    run_workload(std::move(program));

    // Read back from DRAM core L1 (requires 64-bit NOC offset, so use cluster API directly).
    std::vector<uint32_t> result(1, 0);
    MetalContext::instance().get_cluster().read_core(
        result.data(), sizeof(uint32_t), tt_cxy_pair(mesh_device_->build_id(), virtual_dram_core), drisc_l1_noc_addr_);

    EXPECT_EQ(result[0], kMagicValue);
}

// Run the same kernel across multiple DRAM cores.
TEST_F(DramKernelFixture, DramKernelOnMultipleCores) {
    constexpr uint32_t kMagicBase = 0xCAFE0000;
    // DRAM compute grid: x = num_banks, y = num_endpoints_per_bank.
    const auto& soc_desc = MetalContext::instance().get_cluster().get_soc_desc(mesh_device_->build_id());
    auto dram_compute_grid = soc_desc.get_dram_compute_grid_size();
    uint32_t num_banks = std::min(static_cast<size_t>(dram_compute_grid.x), static_cast<size_t>(4));
    uint32_t num_endpoints = dram_compute_grid.y;

    for (uint32_t row = 0; row < num_endpoints; row++) {
        for (uint32_t col = 0; col < num_banks; col++) {
            CoreCoord logical_dram_core{col, row};
            auto virtual_dram_core = mesh_device_->virtual_core_from_logical_core(logical_dram_core, CoreType::DRAM);
            uint32_t expected_value = kMagicBase + (row * num_banks) + col;

            Program program = CreateProgram();
            CreateKernel(
                program,
                "tests/tt_metal/tt_metal/test_kernels/misc/dram_write_one_uint32.cpp",
                logical_dram_core,
                DramConfig{
                    .noc = NOC::NOC_0,
                    .compile_args = {drisc_l1_base_, expected_value},
                });
            run_workload(std::move(program));

            std::vector<uint32_t> result(1, 0);
            MetalContext::instance().get_cluster().read_core(
                result.data(),
                sizeof(uint32_t),
                tt_cxy_pair(mesh_device_->build_id(), virtual_dram_core),
                drisc_l1_noc_addr_);

            EXPECT_EQ(result[0], expected_value) << "Failed for DRAM core (bank=" << col << ", endpoint=" << row << ")";
        }
    }
}

// Test Tensix reading from DRISC L1 using a DRISC<->Tensix semaphore
// handshake:
//  - DRISC enters stream mode, seeds DRISC L1, and signals readiness.
//  - Tensix waits for readiness, reads DRISC L1, then signals done.
//  - DRISC waits for done and restores NOC2AXI.
TEST_F(DramKernelFixture, DramKernelTensixReadFromDRISCL1) {
    constexpr uint32_t kMagicValue = 0xCAFEBABE;
    CoreCoord logical_core_drisc{0, 0};
    CoreCoord logical_core_tensix{0, 0};
    CoreCoord tensix_virtual = device_->virtual_core_from_logical_core(logical_core_tensix, CoreType::WORKER);
    CoreCoord drisc_virtual = device_->virtual_core_from_logical_core(logical_core_drisc, CoreType::DRAM);

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
        DramConfig{.noc = NOC::NOC_0, .compile_args = {drisc_l1_base_, tensix_virtual.x, tensix_virtual.y}});
    SetRuntimeArgs(program, drisc_kid, logical_core_drisc, {stream_ready_sem_id, tensix_done_sem_id, kMagicValue});

    CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/misc/tensix_read_from_drisc.cpp",
        logical_core_tensix,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::NOC_0,
            .compile_args = {
                tensix_l1_base_,
                drisc_l1_base_,
                drisc_virtual.x,
                drisc_virtual.y,
                stream_ready_sem_id,
                tensix_done_sem_id}});

    run_workload(std::move(program));

    // Verify Tensix read the seeded value.
    std::vector<uint32_t> result;
    tt::tt_metal::detail::ReadFromDeviceL1(
        device_, logical_core_tensix, tensix_l1_base_, sizeof(kMagicValue), result, CoreType::WORKER);
    log_info(LogTest, "Tensix L1 result: 0x{:X} (expected: 0x{:X})", result[0], kMagicValue);
    EXPECT_EQ(result[0], kMagicValue) << "Tensix should have read the value from DRISC L1";
}

// Test DRISC reading from Tensix L1
// Host writes magic value to Tensix L1, then DRISC reads it into DRISC L1
TEST_F(DramKernelFixture, DramKernelDRISCReadFromTensixL1) {
    constexpr uint32_t kMagicValue = 0xDEADBEEF;
    CoreCoord logical_core_drisc{0, 0};
    CoreCoord logical_core_tensix{0, 0};
    CoreCoord tensix_virtual = device_->virtual_core_from_logical_core(logical_core_tensix, CoreType::WORKER);
    CoreCoord drisc_virtual = device_->virtual_core_from_logical_core(logical_core_drisc, CoreType::DRAM);

    // Host writes magic value to Tensix L1
    std::vector<uint32_t> write_data = {kMagicValue};
    MetalContext::instance().get_cluster().write_core(
        write_data.data(),
        write_data.size() * sizeof(uint32_t),
        tt_cxy_pair(mesh_device_->build_id(), tensix_virtual),
        tensix_l1_base_);

    Program program = CreateProgram();
    // DRISC kernel reads from Tensix L1 into DRISC L1 and restores NOC2AXI at the end.
    auto kid = CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/misc/drisc_l1_transfer.cpp",
        logical_core_drisc,
        DramConfig{
            .noc = NOC::NOC_0,
            .compile_args = {drisc_l1_base_, tensix_virtual.x, tensix_virtual.y},
            .defines = {{"MODE_TENSIX_TO_DRISC", "1"}}});
    SetRuntimeArgs(program, kid, logical_core_drisc, {tensix_l1_base_});
    run_workload(std::move(program));

    // Verify by reading from DRISC L1
    std::vector<uint32_t> result(1, 0);
    MetalContext::instance().get_cluster().read_core(
        result.data(), sizeof(uint32_t), tt_cxy_pair(mesh_device_->build_id(), drisc_virtual), drisc_l1_noc_addr_);
    log_info(LogTest, "DRISC L1 result: 0x{:X} (expected: 0x{:X})", result[0], kMagicValue);
    EXPECT_EQ(result[0], kMagicValue) << "DRISC should have read the value from Tensix L1";
}

// Stress + Bandwidth test: DRISC L1 write to DRAM GDDR - all banks x N endpoints concurrently and measure aggregate BW
TEST_P(DramKernelDRISCBWFixture, DramKernelDRISCWriteToDRAM) {
    const auto& soc_desc = MetalContext::instance().get_cluster().get_soc_desc(mesh_device_->build_id());
    auto dram_compute_grid = soc_desc.get_dram_compute_grid_size();
    uint32_t num_banks = dram_compute_grid.x;
    uint32_t num_endpoints = std::min(GetParam(), static_cast<uint32_t>(dram_compute_grid.y));

    const uint32_t bytes_per_iter = 64 * 1024;
    constexpr uint32_t iters = 1000;
    const uint32_t total_bytes_per_core = iters * bytes_per_iter;

    // Endpoints within the same bank share GDDR address space - partition by endpoint row.
    TT_FATAL(
        dram_unreserved_size_ >= num_endpoints * total_bytes_per_core,
        "Not enough DRAM: need {} bytes per bank, have {}",
        num_endpoints * total_bytes_per_core,
        dram_unreserved_size_);

    // One page per bank: interleaved allocation gives every bank the same bank-relative
    // base address, so each DRISC DMA can write into its own bank at that address.
    auto dram_buffer = CreateBuffer(InterleavedBufferConfig{
        .device = device_,
        .size = num_banks * num_endpoints * total_bytes_per_core,
        .page_size = num_endpoints * total_bytes_per_core,
        .buffer_type = BufferType::DRAM,
    });
    uint32_t dram_addr = dram_buffer->address();

    auto seed = std::chrono::system_clock::now().time_since_epoch().count();
    log_info(LogTest, "Random seed: {}", seed);
    std::vector<uint32_t> data = create_random_vector_of_bfloat16(bytes_per_iter, 1000.0f, seed);
    Program program = CreateProgram();

    for (uint32_t row = 0; row < num_endpoints; row++) {
        for (uint32_t col = 0; col < num_banks; col++) {
            CoreCoord logical_core{col, row};
            CoreCoord virtual_core = device_->virtual_core_from_logical_core(logical_core, CoreType::DRAM);
            MetalContext::instance().get_cluster().write_core(
                data.data(), bytes_per_iter, tt_cxy_pair(mesh_device_->build_id(), virtual_core), drisc_l1_noc_addr_);
            auto k_id = CreateKernel(
                program,
                "tests/tt_metal/tt_metal/test_kernels/misc/drisc_l1_dram_dma.cpp",
                logical_core,
                DramConfig{.noc = NOC::NOC_0, .defines = {{"L1_TO_GDDR_WRITE_TEST", "1"}}});
            // Partition DRAM gddr dst addr by endpoint row
            const uint32_t dram_dst_gddr_addr = dram_addr + row * total_bytes_per_core;
            SetRuntimeArgs(program, k_id, logical_core, {dram_dst_gddr_addr, drisc_l1_base_, bytes_per_iter, iters});
        }
    }

    run_workload(std::move(program));

    // Kernel writes timing immediately after the data buffer in DRISC L1
    uint64_t timing_noc_addr = drisc_l1_noc_addr_ + static_cast<uint64_t>(bytes_per_iter);
    uint32_t clk_hz = MetalContext::instance().get_cluster().get_device_aiclk(device_->id()) * 1000000u;
    uint64_t max_cycles = 0;

    // Verify all DRISCs writes to DRAM and calculate
    // the aggregate write bandwidth from DRISC L1 to DRAM across all endpoints
    for (uint32_t row = 0; row < num_endpoints; row++) {
        for (uint32_t col = 0; col < num_banks; col++) {
            uint32_t dram_channel =
                device_->dram_channel_from_logical_core(CoreCoord{col, 0});  // channel maps by bank (col)
            // Verify the last chunk written; all chunks hold identical data so one read suffices.
            // ReadFromDeviceDRAMChannel is slow (host-device round-trip); avoid reading all iters.
            std::vector<uint32_t> result(bytes_per_iter / sizeof(uint32_t));
            tt::tt_metal::detail::ReadFromDeviceDRAMChannel(
                device_,
                dram_channel,
                dram_addr + bytes_per_iter * (iters - 1) + total_bytes_per_core * row,
                bytes_per_iter,
                result);
            EXPECT_EQ(result, data) << "Data mismatch on DRAM from core (bank=" << col << ", endpoint=" << row << ")";
            CoreCoord virtual_core = device_->virtual_core_from_logical_core({col, row}, CoreType::DRAM);
            max_cycles = std::max(max_cycles, read_timing_cycles(virtual_core, timing_noc_addr));
        }
    }

    uint64_t total_bytes_all = static_cast<uint64_t>(num_banks) * num_endpoints * total_bytes_per_core;
    log_info(
        LogTest,
        "DRISC DMA Multi-Endpoint Write BW: {:.2f} GB/s ({} banks x {} endpoints, {:.0f} MB total, {} max cycles)",
        compute_bw_gbs(total_bytes_all, max_cycles, clk_hz),
        num_banks,
        num_endpoints,
        total_bytes_all / 1e6,
        max_cycles);
}

// Stress + Bandwidth test: Read from DRAM GDDR to DRISC L1 over DMA - all banks x endpoints concurrently and measure
// aggregate BW
TEST_P(DramKernelDRISCBWFixture, DramKernelDRISCReadFromDRAM) {
    const auto& soc_desc = MetalContext::instance().get_cluster().get_soc_desc(mesh_device_->build_id());
    auto dram_compute_grid = soc_desc.get_dram_compute_grid_size();
    uint32_t num_banks = dram_compute_grid.x;
    uint32_t num_endpoints = std::min(GetParam(), static_cast<uint32_t>(dram_compute_grid.y));

    const uint32_t bytes_per_iter = 64 * 1024;
    constexpr uint32_t iters = 1000;
    const uint32_t total_bytes_per_core = iters * bytes_per_iter;

    // Each endpoint within the same bank reads from its own DRAM slot (partitioned by row).
    TT_FATAL(
        dram_unreserved_size_ >= num_endpoints * bytes_per_iter,
        "Not enough DRAM for {} endpoint source regions",
        num_endpoints);

    // One page per bank: interleaved allocation gives every bank the same bank-relative
    // base address, so each DRISC DMA reads from its own bank at that address.
    auto dram_buffer = CreateBuffer(InterleavedBufferConfig{
        .device = device_,
        .size = num_banks * num_endpoints * bytes_per_iter,
        .page_size = num_endpoints * bytes_per_iter,
        .buffer_type = BufferType::DRAM,
    });
    uint32_t dram_addr = dram_buffer->address();

    auto seed = std::chrono::system_clock::now().time_since_epoch().count();
    log_info(LogTest, "Random seed: {}", seed);
    std::vector<uint32_t> data = create_random_vector_of_bfloat16(bytes_per_iter, 1000.0f, seed);

    // Write data from DRISCs to read to all DRAM Banks
    for (uint32_t col = 0; col < num_banks; col++) {
        uint32_t dram_channel = device_->dram_channel_from_logical_core(CoreCoord{col, 0});
        for (uint32_t row = 0; row < num_endpoints; row++) {
            tt::tt_metal::detail::WriteToDeviceDRAMChannel(
                device_, dram_channel, dram_addr + row * bytes_per_iter, data);
        }
    }

    Program program = CreateProgram();
    for (uint32_t row = 0; row < num_endpoints; row++) {
        for (uint32_t col = 0; col < num_banks; col++) {
            CoreCoord logical_core{col, row};
            auto k_id = CreateKernel(
                program,
                "tests/tt_metal/tt_metal/test_kernels/misc/drisc_l1_dram_dma.cpp",
                logical_core,
                DramConfig{.noc = NOC::NOC_0});
            // Partition DRAM gddr src addr by endpoint row
            const uint32_t dram_src_gddr_addr = dram_addr + row * bytes_per_iter;
            SetRuntimeArgs(program, k_id, logical_core, {dram_src_gddr_addr, drisc_l1_base_, bytes_per_iter, iters});
        }
    }

    run_workload(std::move(program));

    // Kernel writes timing immediately after the data buffer in DRISC L1.
    uint64_t timing_noc_addr = drisc_l1_noc_addr_ + static_cast<uint64_t>(bytes_per_iter);
    uint32_t clk_hz = MetalContext::instance().get_cluster().get_device_aiclk(device_->id()) * 1000000u;
    uint64_t max_cycles = 0;

    // Verify all reads into DRISC L1 over DMA from DRAM are correct
    for (uint32_t row = 0; row < num_endpoints; row++) {
        for (uint32_t col = 0; col < num_banks; col++) {
            CoreCoord virtual_core = device_->virtual_core_from_logical_core({col, row}, CoreType::DRAM);
            std::vector<uint32_t> result(data.size());
            MetalContext::instance().get_cluster().read_core(
                result.data(), bytes_per_iter, tt_cxy_pair(mesh_device_->build_id(), virtual_core), drisc_l1_noc_addr_);
            EXPECT_EQ(result, data) << "Data mismatch on core (bank=" << col << ", endpoint=" << row << ")";
            max_cycles = std::max(max_cycles, read_timing_cycles(virtual_core, timing_noc_addr));
        }
    }

    uint64_t total_bytes_all = static_cast<uint64_t>(num_banks) * num_endpoints * total_bytes_per_core;
    log_info(
        LogTest,
        "DRISC DMA Multi-Endpoint Read BW: {:.2f} GB/s ({} banks x {} endpoints, {:.0f} MB total, {} max cycles)",
        compute_bw_gbs(total_bytes_all, max_cycles, clk_hz),
        num_banks,
        num_endpoints,
        total_bytes_all / 1e6,
        max_cycles);
}

INSTANTIATE_TEST_SUITE_P(
    EndpointSweep,
    DramKernelDRISCBWFixture,
    testing::Values(1u, 2u, 3u),
    [](const testing::TestParamInfo<uint32_t>& info) { return std::to_string(info.param) + "_endpoints"; });

// Read from GDDR over DMA into DRISC L1 and then multicast from DRISC L1 to a grid of 6x6 Tensix L1
TEST_F(DramKernelFixture, DramKernelDRISCReadFromDRAMMcastToTensix) {
    const uint32_t total_bytes = 64 * 1024;

    TT_FATAL(
        dram_unreserved_size_ >= total_bytes,
        "Not enough DRAM: need {} bytes, have {}",
        total_bytes,
        dram_unreserved_size_);

    auto seed = std::chrono::system_clock::now().time_since_epoch().count();
    log_info(LogTest, "Random seed: {}", seed);
    std::vector<uint32_t> data = create_random_vector_of_bfloat16(total_bytes, 1000.0f, seed);

    CoreCoord logical_core{0, 0};
    uint32_t dram_channel = device_->dram_channel_from_logical_core(logical_core);
    uint32_t num_cols = 6;
    uint32_t num_rows = 6;
    uint32_t num_subordinates = num_cols * num_rows;  // 6x6 Tensix grid
    CoreCoord tensix_sub_logical_start_coord{0, 0};
    CoreCoord tensix_sub_logical_end_coord{num_cols - 1, num_rows - 1};
    CoreCoord sub_worker_start_coord =
        device_->virtual_core_from_logical_core(tensix_sub_logical_start_coord, CoreType::WORKER);
    CoreCoord sub_worker_end_coord =
        device_->virtual_core_from_logical_core(tensix_sub_logical_end_coord, CoreType::WORKER);

    // Allocate a single-page DRAM buffer. Page_size == size pins it to bank 0 (logical_core {0,0})
    auto dram_buffer = CreateBuffer(InterleavedBufferConfig{
        .device = device_,
        .size = total_bytes,
        .page_size = total_bytes,
        .buffer_type = BufferType::DRAM,
    });
    uint32_t dram_addr = dram_buffer->address();

    // Write data into DRAM for DRISCs to read
    tt::tt_metal::detail::WriteToDeviceDRAMChannel(device_, dram_channel, dram_addr, data);

    Program program = CreateProgram();
    auto mcast_k_id = CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/misc/drisc_mcast_writes_tensix.cpp",
        logical_core,
        DramConfig{.noc = NOC::NOC_0, .defines = {{"MULTICAST", "1"}}});
    SetRuntimeArgs(
        program,
        mcast_k_id,
        logical_core,
        {dram_addr,
         drisc_l1_base_,
         tensix_l1_base_,
         sub_worker_start_coord.x,
         sub_worker_start_coord.y,
         sub_worker_end_coord.x,
         sub_worker_end_coord.y,
         total_bytes,
         num_subordinates});

    run_workload(std::move(program));

    // Verify all multicasts into Tensix L1 from DRISC are correct
    for (uint32_t row = 0; row < num_rows; row++) {
        for (uint32_t col = 0; col < num_cols; col++) {
            CoreCoord virtual_core = device_->virtual_core_from_logical_core({col, row}, CoreType::WORKER);
            std::vector<uint32_t> result(data.size());
            MetalContext::instance().get_cluster().read_core(
                result.data(),
                data.size() * sizeof(uint32_t),
                tt_cxy_pair(mesh_device_->build_id(), virtual_core),
                tensix_l1_base_);
            EXPECT_EQ(result, data) << "Data mismatch on core (" << col << ", " << row << ")";
        }
    }
}

// Stress test: Read from GDDR over DMA into DRISC L1 all endpoints of a single bank and from a 6x6 Tensix grid in
// parallel
TEST_F(DramKernelFixture, DramKernelDRISCRTensixParallelDRAMReads) {
    const uint32_t total_bytes = 64 * 1024;

    const auto& soc_desc = MetalContext::instance().get_cluster().get_soc_desc(mesh_device_->build_id());
    auto dram_compute_grid = soc_desc.get_dram_compute_grid_size();
    uint32_t num_endpoints = dram_compute_grid.y;  // all endpoints of a bank

    TT_FATAL(
        dram_unreserved_size_ >= total_bytes,
        "Not enough DRAM: need {} bytes, have {}",
        total_bytes,
        dram_unreserved_size_);

    auto seed = std::chrono::system_clock::now().time_since_epoch().count();
    log_info(LogTest, "Random seed: {}", seed);
    std::vector<uint32_t> data = create_random_vector_of_bfloat16(total_bytes, 1000.0f, seed);

    CoreCoord logical_core{0, 0};
    uint32_t dram_channel = device_->dram_channel_from_logical_core(logical_core);
    uint32_t num_cols = 6;
    uint32_t num_rows = 6;
    CoreCoord worker_start{0, 0};
    CoreCoord worker_end{num_cols - 1, num_rows - 1};  // 6x6 Tensix grid
    uint32_t bank_id = 0;                              // for DRISCs: Single Bank, all endpoints
    CoreCoord drisc_endpoint_start{bank_id, 0};
    CoreCoord drisc_endpoint_end{bank_id, num_endpoints - 1};
    CoreRangeSet tensix_range({CoreRange(worker_start, worker_end)});
    CoreRangeSet drisc_endpoint_range({CoreRange(drisc_endpoint_start, drisc_endpoint_end)});

    // Allocate a single-page DRAM buffer. Page_size == size pins it to bank 0 (logical_core {0,0})
    auto dram_buffer = CreateBuffer(InterleavedBufferConfig{
        .device = device_,
        .size = total_bytes,
        .page_size = total_bytes,
        .buffer_type = BufferType::DRAM,
    });
    uint32_t dram_addr = dram_buffer->address();

    // Write data into the DRAM for DRISCs and Tensix to read
    tt::tt_metal::detail::WriteToDeviceDRAMChannel(device_, dram_channel, dram_addr, data);

    Program program = CreateProgram();

    // DRISC Kernel
    auto drisc_k_id = CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/misc/drisc_l1_dram_dma.cpp",
        drisc_endpoint_range,
        DramConfig{.noc = NOC::NOC_0});
    SetRuntimeArgs(program, drisc_k_id, drisc_endpoint_range, {dram_addr, drisc_l1_base_, total_bytes, 1});

    // Tensix Kernel
    auto tensix_k_id = CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/misc/tensix_dram_reads.cpp",
        tensix_range,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::NOC_0});
    SetRuntimeArgs(program, tensix_k_id, tensix_range, {bank_id, dram_addr, tensix_l1_base_, total_bytes});

    run_workload(std::move(program));

    // Verify DRISC L1 reads are correct
    for (uint32_t endpoint = 0; endpoint < num_endpoints; endpoint++) {
        CoreCoord virtual_core = device_->virtual_core_from_logical_core({bank_id, endpoint}, CoreType::DRAM);
        std::vector<uint32_t> result(data.size());
        MetalContext::instance().get_cluster().read_core(
            result.data(),
            data.size() * sizeof(uint32_t),
            tt_cxy_pair(mesh_device_->build_id(), virtual_core),
            drisc_l1_noc_addr_);
        EXPECT_EQ(result, data) << "Data mismatch on core (" << endpoint << ")";
    }

    // Verify Tensix L1 reads are correct
    for (uint32_t row = 0; row < num_rows; row++) {
        for (uint32_t col = 0; col < num_cols; col++) {
            CoreCoord virtual_core = device_->virtual_core_from_logical_core({col, row}, CoreType::WORKER);
            std::vector<uint32_t> result(data.size());
            MetalContext::instance().get_cluster().read_core(
                result.data(),
                data.size() * sizeof(uint32_t),
                tt_cxy_pair(mesh_device_->build_id(), virtual_core),
                tensix_l1_base_);
            EXPECT_EQ(result, data) << "Data mismatch on core (" << col << ", " << row << ")";
        }
    }
}

// Fixture for DRISC + Tensix GDDR BW sweep tests. Inherits DramKernelFixture so
// it skips when DRAM programmable cores are not enabled.
class DramKernelDRISCGDDRBWSweepFixture : public DramKernelFixture, public testing::WithParamInterface<uint32_t> {};

// DRISC DMA GDDR -> L1 + NOC unicast to Tensix L1, double-buffered
TEST_P(DramKernelDRISCGDDRBWSweepFixture, DRISCDMAUcastToTensix) {
    const uint32_t bytes_per_iter = GetParam();
    constexpr uint32_t iters = 1000;
    const uint32_t total_bytes = iters * bytes_per_iter;

    TT_FATAL(
        dram_unreserved_size_ >= total_bytes,
        "Not enough DRAM: need {} bytes, have {}",
        total_bytes,
        dram_unreserved_size_);

    auto seed = std::chrono::system_clock::now().time_since_epoch().count();
    log_info(LogTest, "Random seed: {}", seed);
    std::vector<uint32_t> data = create_random_vector_of_bfloat16(total_bytes, 1000.0f, seed);

    CoreCoord logical_core{0, 0};
    uint32_t dram_channel = device_->dram_channel_from_logical_core(logical_core);
    CoreCoord tensix_logical{0, 0};
    CoreCoord sub_worker = device_->virtual_core_from_logical_core(tensix_logical, CoreType::WORKER);

    // Allocate a single-page DRAM buffer. Page_size == size pins it to bank 0 (logical_core {0,0})
    auto dram_buffer = CreateBuffer(InterleavedBufferConfig{
        .device = device_,
        .size = total_bytes,
        .page_size = total_bytes,
        .buffer_type = BufferType::DRAM,
    });
    uint32_t dram_addr = dram_buffer->address();
    tt::tt_metal::detail::WriteToDeviceDRAMChannel(device_, dram_channel, dram_addr, data);

    Program program = CreateProgram();
    auto drisc_ucast_k_id = CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/misc/drisc_mcast_writes_tensix.cpp",
        logical_core,
        DramConfig{.noc = NOC::NOC_0});
    SetRuntimeArgs(
        program,
        drisc_ucast_k_id,
        logical_core,
        {
            dram_addr,
            drisc_l1_base_,
            tensix_l1_base_,
            sub_worker.x,
            sub_worker.y,
            sub_worker.x,
            sub_worker.y,
            bytes_per_iter,
            0,  // num_subordinates: unused in unicast path
            iters,
        });
    run_workload(std::move(program));

    CoreCoord tensix_virtual = device_->virtual_core_from_logical_core(tensix_logical, CoreType::WORKER);
    const uint32_t elems_per_iter = bytes_per_iter / sizeof(uint32_t);
    std::vector<uint32_t> result(elems_per_iter);
    MetalContext::instance().get_cluster().read_core(
        result.data(), bytes_per_iter, tt_cxy_pair(mesh_device_->build_id(), tensix_virtual), tensix_l1_base_);
    // Kernel overwrites the same Tensix L1 address each iteration; verify the last chunk landed.
    std::vector<uint32_t> last_chunk(data.end() - elems_per_iter, data.end());
    EXPECT_EQ(result, last_chunk);

    CoreCoord dram_virtual = device_->virtual_core_from_logical_core(logical_core, CoreType::DRAM);
    // Kernel writes timing immediately after the data buffer in DRISC L1.
    uint64_t timing_noc_addr = drisc_l1_noc_addr_ + static_cast<uint64_t>(bytes_per_iter);
    uint32_t clk_hz = MetalContext::instance().get_cluster().get_device_aiclk(device_->id()) * 1000000u;
    uint64_t cycles = read_timing_cycles(dram_virtual, timing_noc_addr);

    log_info(
        LogTest,
        "DRISC DMA + NOC ucast BW ({}KB): {:.2f} GB/s ({:.0f} MB total, {} cycles)",
        bytes_per_iter / 1024,
        compute_bw_gbs(total_bytes, cycles, clk_hz),
        total_bytes / 1e6,
        cycles);
}

INSTANTIATE_TEST_SUITE_P(
    SizeSweep,
    DramKernelDRISCGDDRBWSweepFixture,
    testing::Values(2048u, 4096u, 8192u, 16384u, 32768u, 65536u),
    [](const testing::TestParamInfo<uint32_t>& info) { return std::to_string(info.param / 1024) + "KB"; });
