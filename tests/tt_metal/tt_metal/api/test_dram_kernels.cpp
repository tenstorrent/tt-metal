// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <fmt/format.h>
#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>
#include <chrono>
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/experimental/sockets/d2h_socket.hpp>
#include <tt-metalium/program.hpp>
#include <umd/device/types/arch.hpp>

#include "device_fixture.hpp"
#include "tt_metal/impl/dispatch/slow_dispatch.hpp"
#include "distributed/mesh_device_impl.hpp"
#include "impl/context/metal_context.hpp"
#include "impl/kernels/kernel.hpp"
#include "hostdevcommon/profiler_common.h"
#include "llrt/hal.hpp"
#include "llrt/tt_cluster.hpp"

using namespace tt;
using namespace tt::tt_metal;

static double compute_bw_gbs(uint64_t total_bytes, uint64_t cycles, uint32_t clk_hz) {
    return static_cast<double>(total_bytes) * clk_hz / cycles / 1e9;
}

// Logical DRAM endpoint (for CreateKernel) that bank `dram_view`'s reads traverse on `noc`.
// Not UMD translate_coord_to(.., LOGICAL): that returns the raw subchannel, but CreateKernel(DramConfig)
// indexes dram_bank_endpoint_coords (preferred-first), so UMD would land on the wrong DRISC. Instead we
// invert dram_bank_endpoint_coords against the preferred coord -- the value firmware uses as the read target.
static CoreCoord logical_dram_endpoint_for_noc(const metal_SocDescriptor& soc_desc, uint32_t dram_view, NOC noc) {
    CoreCoord pref = soc_desc.get_preferred_worker_core_for_dram_view(dram_view, static_cast<uint8_t>(noc));
    const auto& endpoints = soc_desc.dram_bank_endpoint_coords.at(dram_view);
    for (uint32_t i = 0; i < endpoints.size(); i++) {
        if (endpoints[i] == pref) {
            return CoreCoord{dram_view, i};
        }
    }
    TT_FATAL(false, "Preferred DRAM endpoint ({}, {}) for bank {} not found", pref.x, pref.y, dram_view);
    return {};
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
        device_range_ = distributed::MeshCoordinateRange(distributed::MeshCoordinate(0, 0));
        drisc_l1_base_ = hal.get_dev_addr(HalProgrammableCoreType::DRAM, HalL1MemAddrType::UNRESERVED);
        drisc_l1_noc_addr_ = hal.get_dev_noc_addr(HalProgrammableCoreType::DRAM, HalL1MemAddrType::UNRESERVED);
        tensix_l1_base_ = mesh_device_->allocator()->get_base_allocator_addr(HalMemType::L1);
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

    // Logical subchannel indices of `bank` that run a DRISC kernel: every DRAM endpoint except the
    // NOC0 worker endpoint (logical subchannel 0), which is owned by the syseng firmware and left in
    // reset, so no DRISC kernel can be launched there. DRISC tests must only target these.
    std::vector<uint32_t> usable_dram_endpoints(uint32_t bank) const {
        const auto& soc_desc = MetalContext::instance().get_cluster().get_soc_desc(mesh_device_->build_id());
        const uint32_t noc0_sub = logical_dram_endpoint_for_noc(soc_desc, bank, NOC::NOC_0).y;
        const uint32_t num_endpoints = soc_desc.get_dram_compute_grid_size().y;
        std::vector<uint32_t> usable;
        for (uint32_t sub = 0; sub < num_endpoints; ++sub) {
            if (sub != noc0_sub) {
                usable.push_back(sub);
            }
        }
        return usable;
    }

    // First DRISC-usable endpoint subchannel for `bank` (skips the NOC0 endpoint).
    uint32_t first_usable_dram_endpoint(uint32_t bank) const {
        const std::vector<uint32_t> usable = usable_dram_endpoints(bank);
        TT_FATAL(!usable.empty(), "DRAM bank {} has no DRISC-usable endpoint (only the NOC0 endpoint?)", bank);
        return usable.front();
    }

    distributed::MeshDevice* mesh_device_{};
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
    // Pick the first DRISC-usable endpoint of bank 0 (subchannel 0 is the syseng-owned NOC0 endpoint).
    CoreCoord logical_dram_core{0, first_usable_dram_endpoint(0)};
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
    // Skip the syseng-owned NOC0 endpoint; only DRISC-usable subchannels run a kernel.
    const std::vector<uint32_t> usable_endpoints = usable_dram_endpoints(0);

    for (uint32_t i = 0; i < usable_endpoints.size(); i++) {
        const uint32_t row = usable_endpoints[i];
        for (uint32_t col = 0; col < num_banks; col++) {
            CoreCoord logical_dram_core{col, row};
            auto virtual_dram_core = mesh_device_->virtual_core_from_logical_core(logical_dram_core, CoreType::DRAM);
            uint32_t expected_value = kMagicBase + (i * num_banks) + col;

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

// Test Tensix reading from DRISC L1 in NOC2AXI mode: host seeds DRISC L1 directly,
// Tensix reads it using the 5-arg noc_read_with_state to preserve the 64-bit DRAM_L1_NOC_OFFSET address.
TEST_F(DramKernelFixture, DramKernelTensixReadFromDRISCL1) {
    constexpr uint32_t kMagicValue = 0xCAFEBABE;
    CoreCoord logical_core_drisc{0, first_usable_dram_endpoint(0)};
    CoreCoord logical_core_tensix{0, 0};
    CoreCoord drisc_virtual = mesh_device_->virtual_core_from_logical_core(logical_core_drisc, CoreType::DRAM);

    Program program = CreateProgram();

    uint32_t magic = kMagicValue;
    MetalContext::instance().get_cluster().write_core(
        &magic, sizeof(uint32_t), tt_cxy_pair(mesh_device_->build_id(), drisc_virtual), drisc_l1_noc_addr_);

    // Split 64-bit DRISC L1 NOC addr (with DRAM_L1_NOC_OFFSET bit 37) into two uint32_t compile args.
    const uint32_t drisc_l1_noc_addr_low = static_cast<uint32_t>(drisc_l1_noc_addr_);
    const uint32_t drisc_l1_noc_addr_high = static_cast<uint32_t>(drisc_l1_noc_addr_ >> 32);

    CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/misc/tensix_read_from_drisc.cpp",
        logical_core_tensix,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::NOC_0,
            .compile_args = {
                tensix_l1_base_, drisc_l1_noc_addr_low, drisc_l1_noc_addr_high, drisc_virtual.x, drisc_virtual.y}});

    run_workload(std::move(program));

    // Verify Tensix read the seeded value.
    std::vector<uint32_t> result;
    slow_dispatch::ReadFromL1(
        *mesh_device_, logical_core_tensix, tensix_l1_base_, sizeof(kMagicValue), result, CoreType::WORKER);
    log_info(LogTest, "Tensix L1 result: 0x{:X} (expected: 0x{:X})", result[0], kMagicValue);
    EXPECT_EQ(result[0], kMagicValue) << "Tensix should have read the value from DRISC L1";
}

// Test DRISC reading from Tensix L1
// Host writes magic value to Tensix L1, then DRISC reads it into DRISC L1
TEST_F(DramKernelFixture, DramKernelDRISCReadFromTensixL1) {
    constexpr uint32_t kMagicValue = 0xDEADBEEF;
    CoreCoord logical_core_drisc{0, first_usable_dram_endpoint(0)};
    CoreCoord logical_core_tensix{0, 0};
    CoreCoord tensix_virtual = mesh_device_->virtual_core_from_logical_core(logical_core_tensix, CoreType::WORKER);
    CoreCoord drisc_virtual = mesh_device_->virtual_core_from_logical_core(logical_core_drisc, CoreType::DRAM);

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
        DramConfig{.noc = NOC::NOC_0, .compile_args = {drisc_l1_base_, tensix_virtual.x, tensix_virtual.y}});
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
    // Skip the syseng-owned NOC0 endpoint; only DRISC-usable subchannels run a kernel.
    const std::vector<uint32_t> usable_endpoints = usable_dram_endpoints(0);
    uint32_t num_endpoints = std::min(GetParam(), static_cast<uint32_t>(usable_endpoints.size()));

    const uint32_t bytes_per_iter = 64 * 1024;
    constexpr uint32_t iters = 1000;
    const uint32_t total_bytes_per_core = iters * bytes_per_iter;
    const uint32_t elements_per_endpoint = bytes_per_iter / sizeof(uint32_t);

    // Endpoints within the same bank share GDDR address space - partition by active-endpoint index.
    TT_FATAL(
        dram_unreserved_size_ >= num_endpoints * total_bytes_per_core,
        "Not enough DRAM: need {} bytes per bank, have {}",
        num_endpoints * total_bytes_per_core,
        dram_unreserved_size_);

    // One page per bank: interleaved allocation gives every bank the same bank-relative
    // base address, so each DRISC DMA can write into its own bank at that address.
    auto dram_buffer = distributed::MeshBuffer::create(
        distributed::ReplicatedBufferConfig{.size = num_banks * num_endpoints * total_bytes_per_core},
        {.page_size = num_endpoints * total_bytes_per_core, .buffer_type = BufferType::DRAM},
        mesh_device_);
    uint32_t dram_addr = dram_buffer->address();

    auto seed = std::chrono::system_clock::now().time_since_epoch().count();
    log_info(LogTest, "Random seed: {}", seed);
    // Unique data per endpoint proves each endpoint has independent L1.
    std::vector<uint32_t> data =
        create_random_vector_of_bfloat16(num_banks * num_endpoints * bytes_per_iter, 1000.0f, seed);
    auto endpoint_offset = [&](uint32_t i, uint32_t col) { return (i * num_banks + col) * elements_per_endpoint; };
    Program program = CreateProgram();

    for (uint32_t i = 0; i < num_endpoints; i++) {
        const uint32_t row = usable_endpoints[i];
        for (uint32_t col = 0; col < num_banks; col++) {
            CoreCoord logical_core{col, row};
            CoreCoord virtual_core = mesh_device_->virtual_core_from_logical_core(logical_core, CoreType::DRAM);
            MetalContext::instance().get_cluster().write_core(
                data.data() + endpoint_offset(i, col),
                bytes_per_iter,
                tt_cxy_pair(mesh_device_->build_id(), virtual_core),
                drisc_l1_noc_addr_);
            auto k_id = CreateKernel(
                program,
                "tests/tt_metal/tt_metal/test_kernels/misc/drisc_l1_dram_dma.cpp",
                logical_core,
                DramConfig{.noc = NOC::NOC_0, .defines = {{"L1_TO_GDDR_WRITE_TEST", "1"}}});
            // Partition DRAM gddr dst addr by active-endpoint index
            const uint32_t dram_dst_gddr_addr = dram_addr + i * total_bytes_per_core;
            SetRuntimeArgs(program, k_id, logical_core, {dram_dst_gddr_addr, drisc_l1_base_, bytes_per_iter, iters});
        }
    }

    run_workload(std::move(program));

    // Kernel writes timing immediately after the data buffer in DRISC L1
    uint64_t timing_noc_addr = drisc_l1_noc_addr_ + static_cast<uint64_t>(bytes_per_iter);
    const auto device_id = mesh_device_->get_device_ids()[0];
    uint32_t clk_hz = MetalContext::instance().get_cluster().get_device_aiclk(device_id) * 1000000u;
    uint64_t max_cycles = 0;

    // Verify all DRISCs writes to DRAM and calculate
    // the aggregate write bandwidth from DRISC L1 to DRAM across all endpoints
    for (uint32_t i = 0; i < num_endpoints; i++) {
        const uint32_t row = usable_endpoints[i];
        for (uint32_t col = 0; col < num_banks; col++) {
            auto begin = data.begin() + endpoint_offset(i, col);
            std::vector<uint32_t> endpoint_data(begin, begin + elements_per_endpoint);
            uint32_t dram_channel =
                mesh_device_->dram_channel_from_logical_core(CoreCoord{col, 0});  // channel maps by bank (col)
            // ReadFromDeviceDRAMChannel is slow (host-device round-trip); avoid reading all iters.
            std::vector<uint32_t> result(elements_per_endpoint);
            slow_dispatch::ReadFromDRAMChannel(
                *mesh_device_,
                dram_channel,
                dram_addr + bytes_per_iter * (iters - 1) + total_bytes_per_core * i,
                bytes_per_iter,
                result);
            EXPECT_EQ(result, endpoint_data)
                << "Data mismatch on DRAM from core (bank=" << col << ", endpoint=" << row << ")";
            CoreCoord virtual_core = mesh_device_->virtual_core_from_logical_core({col, row}, CoreType::DRAM);
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
    // Skip the syseng-owned NOC0 endpoint; only DRISC-usable subchannels run a kernel.
    const std::vector<uint32_t> usable_endpoints = usable_dram_endpoints(0);
    uint32_t num_endpoints = std::min(GetParam(), static_cast<uint32_t>(usable_endpoints.size()));

    const uint32_t bytes_per_iter = 64 * 1024;
    constexpr uint32_t iters = 1000;
    const uint32_t total_bytes_per_core = iters * bytes_per_iter;
    const uint32_t elements_per_endpoint = bytes_per_iter / sizeof(uint32_t);

    // Each active endpoint within the same bank reads from its own DRAM slot (partitioned by index).
    TT_FATAL(
        dram_unreserved_size_ >= num_endpoints * bytes_per_iter,
        "Not enough DRAM for {} endpoint source regions",
        num_endpoints);

    // One page per bank: interleaved allocation gives every bank the same bank-relative
    // base address, so each DRISC DMA reads from its own bank at that address.
    auto dram_buffer = distributed::MeshBuffer::create(
        distributed::ReplicatedBufferConfig{.size = num_banks * num_endpoints * bytes_per_iter},
        {.page_size = num_endpoints * bytes_per_iter, .buffer_type = BufferType::DRAM},
        mesh_device_);
    uint32_t dram_addr = dram_buffer->address();

    auto seed = std::chrono::system_clock::now().time_since_epoch().count();
    log_info(LogTest, "Random seed: {}", seed);
    // Unique data per endpoint proves each endpoint has independent L1.
    std::vector<uint32_t> data =
        create_random_vector_of_bfloat16(num_banks * num_endpoints * bytes_per_iter, 1000.0f, seed);
    auto endpoint_offset = [&](uint32_t i, uint32_t col) { return (i * num_banks + col) * elements_per_endpoint; };

    // Write data from DRISCs to read to all DRAM Banks
    for (uint32_t col = 0; col < num_banks; col++) {
        uint32_t dram_channel = mesh_device_->dram_channel_from_logical_core(CoreCoord{col, 0});
        for (uint32_t i = 0; i < num_endpoints; i++) {
            auto begin = data.begin() + endpoint_offset(i, col);
            std::vector<uint32_t> endpoint_data(begin, begin + elements_per_endpoint);
            slow_dispatch::WriteToDRAMChannel(
                *mesh_device_, dram_channel, dram_addr + i * bytes_per_iter, endpoint_data);
        }
    }

    Program program = CreateProgram();
    for (uint32_t i = 0; i < num_endpoints; i++) {
        const uint32_t row = usable_endpoints[i];
        for (uint32_t col = 0; col < num_banks; col++) {
            CoreCoord logical_core{col, row};
            auto k_id = CreateKernel(
                program,
                "tests/tt_metal/tt_metal/test_kernels/misc/drisc_l1_dram_dma.cpp",
                logical_core,
                DramConfig{.noc = NOC::NOC_0});
            // Partition DRAM gddr src addr by active-endpoint index
            const uint32_t dram_src_gddr_addr = dram_addr + i * bytes_per_iter;
            SetRuntimeArgs(program, k_id, logical_core, {dram_src_gddr_addr, drisc_l1_base_, bytes_per_iter, iters});
        }
    }

    run_workload(std::move(program));

    // Kernel writes timing immediately after the data buffer in DRISC L1.
    uint64_t timing_noc_addr = drisc_l1_noc_addr_ + static_cast<uint64_t>(bytes_per_iter);
    const auto device_id = mesh_device_->get_device_ids()[0];
    uint32_t clk_hz = MetalContext::instance().get_cluster().get_device_aiclk(device_id) * 1000000u;
    uint64_t max_cycles = 0;

    // Verify all reads into DRISC L1 over DMA from DRAM are correct
    for (uint32_t i = 0; i < num_endpoints; i++) {
        const uint32_t row = usable_endpoints[i];
        for (uint32_t col = 0; col < num_banks; col++) {
            CoreCoord virtual_core = mesh_device_->virtual_core_from_logical_core({col, row}, CoreType::DRAM);
            auto begin = data.begin() + endpoint_offset(i, col);
            std::vector<uint32_t> endpoint_data(begin, begin + elements_per_endpoint);
            std::vector<uint32_t> result(elements_per_endpoint);
            MetalContext::instance().get_cluster().read_core(
                result.data(), bytes_per_iter, tt_cxy_pair(mesh_device_->build_id(), virtual_core), drisc_l1_noc_addr_);
            EXPECT_EQ(result, endpoint_data) << "Data mismatch on core (bank=" << col << ", endpoint=" << row << ")";
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

// At most 2 DRISC-usable endpoints per bank: subchannel 0 is the syseng-owned NOC0 endpoint, leaving
// the NOC1 endpoint and the free subchannel. The per-test min() against usable_dram_endpoints() also
// clamps this, so a larger value would just repeat the 2-endpoint case.
INSTANTIATE_TEST_SUITE_P(
    EndpointSweep, DramKernelDRISCBWFixture, testing::Values(1u, 2u), [](const testing::TestParamInfo<uint32_t>& info) {
        return std::to_string(info.param) + "_endpoints";
    });

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

    // Bank 0, first DRISC-usable endpoint (subchannel 0 is the syseng-owned NOC0 endpoint).
    CoreCoord logical_core{0, first_usable_dram_endpoint(0)};
    uint32_t dram_channel = mesh_device_->dram_channel_from_logical_core(logical_core);
    uint32_t num_cols = 6;
    uint32_t num_rows = 6;
    uint32_t num_subordinates = num_cols * num_rows;  // 6x6 Tensix grid
    CoreCoord tensix_sub_logical_start_coord{0, 0};
    CoreCoord tensix_sub_logical_end_coord{num_cols - 1, num_rows - 1};
    CoreCoord sub_worker_start_coord =
        mesh_device_->virtual_core_from_logical_core(tensix_sub_logical_start_coord, CoreType::WORKER);
    CoreCoord sub_worker_end_coord =
        mesh_device_->virtual_core_from_logical_core(tensix_sub_logical_end_coord, CoreType::WORKER);

    // Allocate a single-page DRAM buffer. Page_size == size pins it to bank 0 (logical_core x==0)
    auto dram_buffer = distributed::MeshBuffer::create(
        distributed::ReplicatedBufferConfig{.size = total_bytes},
        {.page_size = total_bytes, .buffer_type = BufferType::DRAM},
        mesh_device_);
    uint32_t dram_addr = dram_buffer->address();

    // Write data into DRAM for DRISCs to read
    slow_dispatch::WriteToDRAMChannel(*mesh_device_, dram_channel, dram_addr, data);

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
         num_subordinates,
         1u});

    run_workload(std::move(program));

    // Verify all multicasts into Tensix L1 from DRISC are correct
    for (uint32_t row = 0; row < num_rows; row++) {
        for (uint32_t col = 0; col < num_cols; col++) {
            CoreCoord virtual_core = mesh_device_->virtual_core_from_logical_core({col, row}, CoreType::WORKER);
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

    // DRISC-usable endpoints of bank 0 (subchannel 0 is the syseng-owned NOC0 endpoint); contiguous.
    const std::vector<uint32_t> usable_endpoints = usable_dram_endpoints(0);

    TT_FATAL(
        dram_unreserved_size_ >= total_bytes,
        "Not enough DRAM: need {} bytes, have {}",
        total_bytes,
        dram_unreserved_size_);

    auto seed = std::chrono::system_clock::now().time_since_epoch().count();
    log_info(LogTest, "Random seed: {}", seed);
    std::vector<uint32_t> data = create_random_vector_of_bfloat16(total_bytes, 1000.0f, seed);

    CoreCoord logical_core{0, 0};
    uint32_t dram_channel = mesh_device_->dram_channel_from_logical_core(logical_core);
    uint32_t num_cols = 6;
    uint32_t num_rows = 6;
    CoreCoord worker_start{0, 0};
    CoreCoord worker_end{num_cols - 1, num_rows - 1};  // 6x6 Tensix grid
    uint32_t bank_id = 0;                              // for DRISCs: single bank, all usable endpoints
    CoreCoord drisc_endpoint_start{bank_id, usable_endpoints.front()};
    CoreCoord drisc_endpoint_end{bank_id, usable_endpoints.back()};
    CoreRangeSet tensix_range({CoreRange(worker_start, worker_end)});
    CoreRangeSet drisc_endpoint_range({CoreRange(drisc_endpoint_start, drisc_endpoint_end)});

    // Allocate a single-page DRAM buffer. Page_size == size pins it to bank 0 (logical_core x==0)
    auto dram_buffer = distributed::MeshBuffer::create(
        distributed::ReplicatedBufferConfig{.size = total_bytes},
        {.page_size = total_bytes, .buffer_type = BufferType::DRAM},
        mesh_device_);
    uint32_t dram_addr = dram_buffer->address();

    // Write data into the DRAM for DRISCs and Tensix to read
    slow_dispatch::WriteToDRAMChannel(*mesh_device_, dram_channel, dram_addr, data);

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
    SetRuntimeArgs(program, tensix_k_id, tensix_range, {bank_id, dram_addr, tensix_l1_base_, total_bytes, 1u});

    run_workload(std::move(program));

    // Verify DRISC L1 reads are correct
    for (uint32_t endpoint : usable_endpoints) {
        CoreCoord virtual_core = mesh_device_->virtual_core_from_logical_core({bank_id, endpoint}, CoreType::DRAM);
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
            CoreCoord virtual_core = mesh_device_->virtual_core_from_logical_core({col, row}, CoreType::WORKER);
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

    // Bank 0, first DRISC-usable endpoint (subchannel 0 is the syseng-owned NOC0 endpoint).
    CoreCoord logical_core{0, first_usable_dram_endpoint(0)};
    uint32_t dram_channel = mesh_device_->dram_channel_from_logical_core(logical_core);
    CoreCoord tensix_logical{0, 0};
    CoreCoord sub_worker = mesh_device_->virtual_core_from_logical_core(tensix_logical, CoreType::WORKER);

    // Allocate a single-page DRAM buffer. Page_size == size pins it to bank 0 (logical_core x==0)
    auto dram_buffer = distributed::MeshBuffer::create(
        distributed::ReplicatedBufferConfig{.size = total_bytes},
        {.page_size = total_bytes, .buffer_type = BufferType::DRAM},
        mesh_device_);
    uint32_t dram_addr = dram_buffer->address();
    slow_dispatch::WriteToDRAMChannel(*mesh_device_, dram_channel, dram_addr, data);

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

    CoreCoord tensix_virtual = mesh_device_->virtual_core_from_logical_core(tensix_logical, CoreType::WORKER);
    const uint32_t elems_per_iter = bytes_per_iter / sizeof(uint32_t);
    std::vector<uint32_t> result(elems_per_iter);
    MetalContext::instance().get_cluster().read_core(
        result.data(), bytes_per_iter, tt_cxy_pair(mesh_device_->build_id(), tensix_virtual), tensix_l1_base_);
    // Kernel overwrites the same Tensix L1 address each iteration; verify the last chunk landed.
    std::vector<uint32_t> last_chunk(data.end() - elems_per_iter, data.end());
    EXPECT_EQ(result, last_chunk);

    CoreCoord dram_virtual = mesh_device_->virtual_core_from_logical_core(logical_core, CoreType::DRAM);
    // Kernel writes timing immediately after the data buffer in DRISC L1.
    uint64_t timing_noc_addr = drisc_l1_noc_addr_ + static_cast<uint64_t>(bytes_per_iter);
    const auto device_id = mesh_device_->get_device_ids()[0];
    uint32_t clk_hz = MetalContext::instance().get_cluster().get_device_aiclk(device_id) * 1000000u;
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

struct DRISCNocModeParams {
    NOC drisc_noc;   // DRISC drives this NIU in stream mode (DMA reads + multicast)
    NOC tensix_noc;  // opposite NIU, left in NOC2AXI mode for the concurrent Tensix DRAM read
};

class DramKernelDRISCNocModeFixture : public DramKernelFixture,
                                      public testing::WithParamInterface<DRISCNocModeParams> {};

// Exercises both NIUs of a single DRISC simultaneously: its drisc_noc NIU runs in stream mode
// (DRISC-initiated DMA reads from GDDR + multicast to a 4x3 Tensix grid) while its tensix_noc NIU
// stays in NOC2AXI mode servicing a concurrent Tensix DRAM read.
//
// A bank's read on tensix_noc deterministically routes to that bank's preferred DRAM endpoint for that
// NOC (NOC0 and NOC1 use different endpoints), so the DRISC kernel is placed on that same endpoint,
// guaranteeing both NIUs belong to one DRISC. The Tensix reader sits just below the mcast grid.
//
// Only tensix_noc == NOC1 is exercised: tensix_noc == NOC0 would place the DRISC kernel on the NOC0
// endpoint, which is owned by the syseng firmware and runs no DRISC kernel (guarded below).
TEST_P(DramKernelDRISCNocModeFixture, DramKernelDRISCNocModeStress) {
    auto [drisc_noc, tensix_noc] = GetParam();

    const auto& soc_desc = MetalContext::instance().get_cluster().get_soc_desc(mesh_device_->build_id());

    constexpr uint32_t bank = 0;
    constexpr uint32_t mcast_cols = 4;
    constexpr uint32_t mcast_rows = 3;
    constexpr uint32_t num_subordinates = mcast_cols * mcast_rows;  // 4 x 3 Tensix grid
    constexpr uint32_t iters = 1000;
    const uint32_t bytes_per_iter = 64 * 1024;  // 64K chunks
    const uint32_t elements_per_iter = bytes_per_iter / sizeof(uint32_t);
    const uint32_t total_bytes = iters * bytes_per_iter;

    TT_FATAL(
        dram_unreserved_size_ >= total_bytes,
        "Not enough DRAM: need {} bytes, have {}",
        total_bytes,
        dram_unreserved_size_);

    // Place the DRISC kernel on the endpoint that tensix_noc reads route to, so one DRISC owns both NIUs
    // (stream on drisc_noc, NOC2AXI on tensix_noc).
    CoreCoord drisc_logical = logical_dram_endpoint_for_noc(soc_desc, bank, tensix_noc);
    // The NOC0 worker endpoint is owned by the syseng firmware and runs no DRISC kernel, so the
    // tensix-on-NOC0 configuration (which would place the DRISC kernel there) can't be exercised.
    if (drisc_logical.y == logical_dram_endpoint_for_noc(soc_desc, bank, NOC::NOC_0).y) {
        GTEST_SKIP() << "DRISC kernel cannot run on the syseng-owned NOC0 DRAM endpoint";
    }
    const uint32_t dram_channel = mesh_device_->dram_channel_from_logical_core(drisc_logical);

    // Fill GDDR with iters random distinct chunks. DRISC and the Tensix reader both walk the same region
    // Only the final chunk remains in L1 after the run, so verification compares against it
    auto dram_buffer = distributed::MeshBuffer::create(
        distributed::ReplicatedBufferConfig{.size = total_bytes},
        {.page_size = total_bytes, .buffer_type = BufferType::DRAM},
        mesh_device_);
    uint32_t dram_addr = dram_buffer->address();

    auto seed = std::chrono::system_clock::now().time_since_epoch().count();
    log_info(LogTest, "Random seed: {}", seed);
    std::vector<uint32_t> data = create_random_vector_of_bfloat16(total_bytes, 1000.0f, seed);
    slow_dispatch::WriteToDRAMChannel(*mesh_device_, dram_channel, dram_addr, data);
    std::vector<uint32_t> last_chunk(data.end() - elements_per_iter, data.end());

    Program program = CreateProgram();

    // DRISC stream kernel: read GDDR chunks for multiple iterations, multicasting each to the 4x3 grid
    CoreCoord mcast_start = mesh_device_->virtual_core_from_logical_core({0, 0}, CoreType::WORKER);
    CoreCoord mcast_end =
        mesh_device_->virtual_core_from_logical_core({mcast_cols - 1, mcast_rows - 1}, CoreType::WORKER);
    auto drisc_k = CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/misc/drisc_mcast_writes_tensix.cpp",
        drisc_logical,
        DramConfig{.noc = drisc_noc, .defines = {{"MULTICAST", "1"}}});
    SetRuntimeArgs(
        program,
        drisc_k,
        drisc_logical,
        {dram_addr,
         drisc_l1_base_,
         tensix_l1_base_,
         mcast_start.x,
         mcast_start.y,
         mcast_end.x,
         mcast_end.y,
         bytes_per_iter,
         num_subordinates,
         iters});

    // Tensix DRAM reader on tensix_noc: walks the same iters chunks through the endpoint's NOC2AXI NIU.
    // Placed at row mcast_rows, just outside the mcast destination rows (0, 1, ... mcast_rows-1)
    CoreCoord tensix_reader_logical{0, mcast_rows};
    auto tensix_k = CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/misc/tensix_dram_reads.cpp",
        tensix_reader_logical,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = tensix_noc});
    SetRuntimeArgs(program, tensix_k, tensix_reader_logical, {bank, dram_addr, tensix_l1_base_, bytes_per_iter, iters});

    run_workload(std::move(program));

    // Verify the 4x3 mcast grid received the last chunk.
    for (uint32_t row = 0; row < mcast_rows; row++) {
        for (uint32_t col = 0; col < mcast_cols; col++) {
            CoreCoord v = mesh_device_->virtual_core_from_logical_core({col, row}, CoreType::WORKER);
            std::vector<uint32_t> result(elements_per_iter);
            MetalContext::instance().get_cluster().read_core(
                result.data(), bytes_per_iter, tt_cxy_pair(mesh_device_->build_id(), v), tensix_l1_base_);
            EXPECT_EQ(result, last_chunk) << "Mcast last-chunk mismatch at Tensix (" << col << ", " << row << ")";
        }
    }

    // Verify the Tensix DRAM reader received the last chunk via the NOC2AXI NIU.
    CoreCoord reader_v = mesh_device_->virtual_core_from_logical_core(tensix_reader_logical, CoreType::WORKER);
    std::vector<uint32_t> result(elements_per_iter);
    MetalContext::instance().get_cluster().read_core(
        result.data(), bytes_per_iter, tt_cxy_pair(mesh_device_->build_id(), reader_v), tensix_l1_base_);
    EXPECT_EQ(result, last_chunk) << "Tensix DRAM read via NOC2AXI NIU last-chunk mismatch";
}

// Only the NOC0-stream / NOC1-NOC2AXI configuration is exercised: it places the DRISC kernel on the
// NOC1 endpoint. The mirror config (NOC1 stream / NOC0 NOC2AXI) would route the Tensix read to the
// NOC0 endpoint and thus require the DRISC kernel there, but that endpoint is owned by the syseng
// firmware and runs no DRISC kernel, so that case is no longer supported.
INSTANTIATE_TEST_SUITE_P(
    NocModeSweep,
    DramKernelDRISCNocModeFixture,
    testing::Values(DRISCNocModeParams{NOC::NOC_0, NOC::NOC_1}),  // NOC0 = stream, NOC1 = NOC2AXI
    [](const testing::TestParamInfo<DRISCNocModeParams>& info) {
        return info.param.drisc_noc == NOC::NOC_0 ? "Noc0StreamNoc1Noc2Axi" : "Noc1StreamNoc0Noc2Axi";
    });

// ---------------------------------------------------------------------------------------------
// DRISC scatter-read microbenchmark.
//
// How fast can one DRISC pull profiler-sized markers out of worker-core L1 over the NoC, and does
// adding a second/fourth DRISC scale or crater? DRISCs on different banks are physically separate
// cores, so contention for a shared read port is not a mechanism available here.
//
// Reads only; no D2H egress. The numbers are an ingest ceiling to size the drainer against.
// ---------------------------------------------------------------------------------------------
class DramKernelDRISCScatterFixture : public DramKernelFixture {
protected:
    static constexpr uint32_t kMarkerBytes = 8;  // real kernel_profiler 2-word marker
    static constexpr const char* kBenchKernel = "tests/tt_metal/tt_metal/test_kernels/misc/drisc_rdrbench.cpp";

    // Per-DRISC bytes moved per config -- sets how long each run takes. Raise it for clock calibration,
    // where the point is to make device time dominate the fixed launch overhead in the host wall clock.
    uint64_t target_bytes_per_drisc_ = 64ull << 20;

    void SetUp() override {
        DramKernelFixture::SetUp();
        if (devices_.empty() || IsSkipped()) {
            return;
        }
        drisc_l1_unreserved_size_ =
            MetalContext::instance().hal().get_dev_size(HalProgrammableCoreType::DRAM, HalL1MemAddrType::UNRESERVED);
    }

    struct BenchResult {
        uint64_t max_cycles = 0;   // slowest participating DRISC
        uint64_t total_bytes = 0;  // summed across DRISCs
        uint64_t total_visits = 0;
        uint32_t timer_overhead_cycles = 0;  // cost of one get_timestamp() on the DRISC
        double wall_s = 0.0;
    };

    // Virtual coords of the whole worker grid, packed x | y<<16 -- the lane table the kernel walks.
    std::vector<uint32_t> worker_coords() const {
        std::vector<uint32_t> coords;
        const CoreCoord grid = mesh_device_->compute_with_storage_grid_size();
        for (uint32_t y = 0; y < grid.y; y++) {
            for (uint32_t x = 0; x < grid.x; x++) {
                const CoreCoord v = mesh_device_->virtual_core_from_logical_core({x, y}, CoreType::WORKER);
                coords.push_back((v.x & 0xFFFFu) | ((v.y & 0xFFFFu) << 16));
            }
        }
        return coords;
    }

    // Prime the polled window on every worker so the kernel's checksum can distinguish "read
    // 35 MB/s of real markers" from "read zeros because the NIU never left NOC2AXI".
    void prime_worker_l1(uint32_t bytes) const {
        std::vector<uint32_t> pattern(bytes / sizeof(uint32_t));
        for (size_t i = 0; i < pattern.size(); i++) {
            pattern[i] = 0xA5A50000u + static_cast<uint32_t>(i);
        }
        const CoreCoord grid = mesh_device_->compute_with_storage_grid_size();
        for (uint32_t y = 0; y < grid.y; y++) {
            for (uint32_t x = 0; x < grid.x; x++) {
                const CoreCoord v = mesh_device_->virtual_core_from_logical_core({x, y}, CoreType::WORKER);
                MetalContext::instance().get_cluster().write_core(
                    pattern.data(), bytes, tt_cxy_pair(mesh_device_->build_id(), v), tensix_l1_base_);
            }
        }
    }

    // The free (non-endpoint) subchannel of each bank -- the only DRISC safe to flip into stream
    // mode, since the NOC1 worker endpoint is how Tensix reaches that bank's DRAM.
    std::vector<CoreCoord> free_drisc_cores(uint32_t count) const {
        std::vector<CoreCoord> cores;
        for (uint32_t bank = 0; bank < count; bank++) {
            cores.push_back(mesh_device_->impl().pick_unused_dram_logical_core(mesh_device_->get_devices()[0], bank));
        }
        return cores;
    }

    // Run one config. The polled grid is partitioned across `drisc_cores` (each DRISC owns a
    // contiguous slice), which is how a real multi-DRISC drainer would divide the grid.
    BenchResult run_bench(
        const std::vector<CoreCoord>& drisc_cores,
        uint32_t markers_per_read,
        uint32_t reads_in_flight,
        uint32_t poll_examine = 0,
        uint32_t ring_slots = 0) {  // 0 = one distinct slot per outstanding read
        const std::vector<uint32_t> coords = worker_coords();
        if (ring_slots == 0) {
            ring_slots = reads_in_flight;
        }
        const uint32_t bytes_per_read = markers_per_read * kMarkerBytes;
        const uint32_t ring_bytes = ring_slots * bytes_per_read;
        const uint32_t results_addr = drisc_l1_base_ + ring_bytes;

        TT_FATAL(
            ring_bytes + 6 * sizeof(uint32_t) <= drisc_l1_unreserved_size_,
            "ring ({} B) + results overflow DRISC L1 unreserved ({} B)",
            ring_bytes,
            drisc_l1_unreserved_size_);

        Program program = CreateProgram();
        std::vector<uint32_t> iters_per_drisc;
        const uint32_t num_drisc = drisc_cores.size();
        for (uint32_t d = 0; d < num_drisc; d++) {
            const uint32_t begin = (coords.size() * d) / num_drisc;
            const uint32_t end = (coords.size() * (d + 1)) / num_drisc;
            const uint32_t slice = end - begin;
            TT_FATAL(slice > 0, "DRISC {} got an empty slice of {} cores", d, coords.size());

            // Scale iterations so every config runs for roughly the same wall time regardless of
            // how many bytes a visit moves -- keeps launch overhead a fixed small fraction.
            const uint64_t bytes_per_iter = static_cast<uint64_t>(slice) * bytes_per_read;
            const uint32_t iters =
                std::clamp<uint32_t>(static_cast<uint32_t>(target_bytes_per_drisc_ / bytes_per_iter), 32u, 2000000u);
            iters_per_drisc.push_back(iters);

            std::vector<uint32_t> rtas = {slice, iters, tensix_l1_base_};
            rtas.insert(rtas.end(), coords.begin() + begin, coords.begin() + end);

            auto kid = CreateKernel(
                program,
                kBenchKernel,
                drisc_cores[d],
                DramConfig{
                    .noc = NOC::NOC_0,
                    .compile_args = {
                        markers_per_read, reads_in_flight, drisc_l1_base_, results_addr, poll_examine, ring_slots}});
            SetRuntimeArgs(program, kid, drisc_cores[d], rtas);
        }

        const auto t0 = std::chrono::steady_clock::now();
        run_workload(std::move(program));
        const auto t1 = std::chrono::steady_clock::now();

        BenchResult result;
        result.wall_s = std::chrono::duration<double>(t1 - t0).count();
        for (uint32_t d = 0; d < num_drisc; d++) {
            const CoreCoord v = mesh_device_->virtual_core_from_logical_core(drisc_cores[d], CoreType::DRAM);
            std::vector<uint32_t> out(6);
            MetalContext::instance().get_cluster().read_core(
                out.data(),
                out.size() * sizeof(uint32_t),
                tt_cxy_pair(mesh_device_->build_id(), v),
                drisc_l1_noc_addr_ + ring_bytes);
            const uint64_t cycles = (static_cast<uint64_t>(out[1]) << 32) | out[0];
            EXPECT_NE(out[2], 0u) << "checksum zero on DRISC (" << drisc_cores[d].x << "," << drisc_cores[d].y
                                  << ") -- reads never landed, bandwidth below is meaningless";
            result.max_cycles = std::max(result.max_cycles, cycles);
            result.total_visits += out[3];
            result.total_bytes += static_cast<uint64_t>(out[3]) * bytes_per_read;
            result.timer_overhead_cycles = std::max(result.timer_overhead_cycles, out[4]);
        }
        return result;
    }

    // Round-robin control-vector poll: how long one DRISC takes to sweep every core's 64-word
    // (256 B) control vector -- the "is there anything to drain" scan that precedes the
    // adaptive bulk decision. Reports the per-core cost and the full-grid sweep period, both from
    // the DRISC's own wall clock, plus the measured DRISC clock so those numbers stand on a
    // device-side measurement rather than on the host's aiclk reading.
    void log_poll_row(const std::string& label, const BenchResult& r, uint32_t num_cores) const {
        const uint32_t aiclk_hz =
            MetalContext::instance().get_cluster().get_device_aiclk(mesh_device_->id()) * 1000000u;
        // Device cycles per host second: a lower bound on the DRISC clock (host wall includes launch
        // overhead), converging upward as the run lengthens.
        const double measured_hz = static_cast<double>(r.max_cycles) / r.wall_s;
        const double cycles_per_visit = static_cast<double>(r.max_cycles) / static_cast<double>(r.total_visits);
        const uint64_t sweeps = r.total_visits / num_cores;
        const double cycles_per_sweep = static_cast<double>(r.max_cycles) / static_cast<double>(sweeps);
        log_info(
            LogTest,
            "{:<34} {:>7.1f} cyc/core {:>7.2f} ns/core | sweep of {} cores: {:>9.0f} cyc {:>7.2f} us "
            "| timer {} cyc | clk: aiclk {:.3f} GHz vs measured >={:.3f} GHz",
            label,
            cycles_per_visit,
            cycles_per_visit * 1e9 / aiclk_hz,
            num_cores,
            cycles_per_sweep,
            cycles_per_sweep * 1e6 / aiclk_hz,
            r.timer_overhead_cycles,
            aiclk_hz / 1e9,
            measured_hz / 1e9);
    }

    void log_row(const std::string& label, const BenchResult& r, uint32_t markers_per_read) const {
        const uint32_t clk_hz = MetalContext::instance().get_cluster().get_device_aiclk(mesh_device_->id()) * 1000000u;
        const double secs = static_cast<double>(r.max_cycles) / clk_hz;
        const uint64_t markers = r.total_visits * markers_per_read;
        log_info(
            LogTest,
            "{:<28} {:>7.3f} GB/s (dev) {:>7.3f} GB/s (wall) {:>8.2f} ns/marker {:>8.2f} ns/visit  "
            "[{} visits, {:.0f} MB, {} cycles]",
            label,
            compute_bw_gbs(r.total_bytes, r.max_cycles, clk_hz),
            static_cast<double>(r.total_bytes) / r.wall_s / 1e9,
            secs * 1e9 / static_cast<double>(markers),
            secs * 1e9 / static_cast<double>(r.total_visits),
            r.total_visits,
            r.total_bytes / 1e6,
            r.max_cycles);
    }

    uint32_t drisc_l1_unreserved_size_{};
};

// Round-robin poll of every core's 64-word (256 B) profiler control vector -- the scan that decides
// whether to bulk-drain. Two variants per depth: read-only, and read + the adaptive switch's tail-delta
// arithmetic (full += tails[r] - heads[c*NRISC+r] over 5 RISCs), so the difference isolates
// what the CPU-side poll work costs on top of the NoC read.
//
// All timings come from the DRISC's own wall clock (RISCV_DEBUG_REG_WALL_CLOCK). The kernel also times
// 1024 back-to-back get_timestamp() calls so the instrument's own cost is on the record: a ~40-cycle
// per-read cost cannot be timed by bracketing with a timer that costs more, which is why the phases are
// separated by ablation rather than by in-loop timestamps.
TEST_F(DramKernelDRISCScatterFixture, DRISCControlVectorPollRoundRobin) {
    constexpr uint32_t kControlVectorWords = 64;                                    // PROFILER_L1_CONTROL_VECTOR_SIZE
    constexpr uint32_t kK = kControlVectorWords * sizeof(uint32_t) / kMarkerBytes;  // 32 -> 256 B
    static_assert(kK * 8 == 256, "control vector read must be 256 B");

    const uint32_t all = static_cast<uint32_t>(worker_coords().size());
    prime_worker_l1(kControlVectorWords * sizeof(uint32_t));
    const std::vector<CoreCoord> drisc = free_drisc_cores(1);

    // Long run: device time dominates the fixed launch overhead, so cycles/wall_s converges on the
    // real DRISC clock instead of under-reporting it.
    target_bytes_per_drisc_ = 512ull << 20;

    log_info(LogTest, "control-vector poll: 256 B x {} cores, 1 DRISC, B={} means issue-all", all, all);
    for (uint32_t b : {1u, 8u, 32u, all}) {
        for (uint32_t examine : {0u, 1u}) {
            const BenchResult r = run_bench(drisc, kK, b, examine);
            log_poll_row(fmt::format("B={:<4} {}", b, examine ? "read+tail-deltas" : "read only       "), r, all);
        }
    }
}

// DRISC -> host egress over a real D2H socket. Ingest on this core measured far above anything the
// host path is likely to absorb, so this is the number that decides whether that headroom is usable.
//
// Two ordering constraints drive the shape of this test:
//   - A DRISC cannot initiate NoC traffic in NOC2AXI mode, and the socket's config write must land in
//     DRISC L1 before the sender kernel runs. So the NIU is flipped to stream mode by its own program
//     first, and restored at the end.
//   - ExternalConfigBuffer::address is uint32_t and cannot carry DRISC L1's 0x2000000000 NoC tag. In
//     stream mode inbound traffic terminates at L1 and plain local addresses work, which is what makes
//     the uint32_t field usable. Every host access to DRISC L1 in this test therefore uses the PLAIN
//     address, not drisc_l1_noc_addr_.
TEST_F(DramKernelDRISCScatterFixture, DRISCD2HSocketEgress) {
    const auto& soc_desc = MetalContext::instance().get_cluster().get_soc_desc(mesh_device_->build_id());
    const CoreCoord drisc_logical = mesh_device_->impl().pick_unused_dram_logical_core(mesh_device_->get_devices()[0], 0);
    const CoreCoord drisc_translated = soc_desc.dram_bank_endpoint_coords.at(drisc_logical.x).at(drisc_logical.y);
    const tt::umd::CoreCoord drisc_phys = soc_desc.translate_coord_to(
        tt::umd::CoreCoord(drisc_translated.x, drisc_translated.y, CoreType::DRAM, CoordSystem::TRANSLATED),
        CoordSystem::NOC0);

    const uint32_t src_l1 = drisc_l1_base_;
    const uint32_t cfg_l1 = drisc_l1_base_ + 48 * 1024;
    const uint32_t res_l1 = drisc_l1_base_ + 56 * 1024;
    TT_FATAL(
        distributed::D2HSocket::required_config_buffer_size() <= 8 * 1024,
        "config buffer no longer fits the 8 KB carved out for it");

    auto set_niu_mode = [&](uint32_t stream) {
        Program p = CreateProgram();
        CreateKernel(
            p,
            "tests/tt_metal/tt_metal/test_kernels/misc/drisc_niu_mode.cpp",
            drisc_logical,
            DramConfig{.noc = NOC::NOC_0, .compile_args = {stream}});
        run_workload(std::move(p));
    };

    set_niu_mode(1);  // stream mode, left on for the socket
    log_info(
        LogTest,
        "DRISC ({},{}) logical, ({},{}) physical -- NIU in stream mode for socket config",
        drisc_logical.x,
        drisc_logical.y,
        drisc_phys.x,
        drisc_phys.y);

    for (uint32_t page_size : {2048u, 8192u, 32768u}) {
        constexpr uint32_t kNumPages = 4000;
        const uint32_t fifo_size = 32 * page_size;
        {
            // In paced mode the page count depends on the controller, so drain until the device is done
            // rather than against a precomputed count.
            distributed::D2HSocket socket(
                devices_[0],
                distributed::MeshCoreCoord(distributed::MeshCoordinate(0, 0), CoreCoord(drisc_phys.x, drisc_phys.y)),
                fifo_size,
                distributed::D2HSocket::ExternalConfigBuffer{.address = cfg_l1, .sender_uses_physical_noc_addr = true});
            socket.set_page_size(page_size);

            std::vector<uint32_t> payload(page_size / sizeof(uint32_t), 0xD2D2D2D2u);
            const CoreCoord drisc_virtual = mesh_device_->virtual_core_from_logical_core(drisc_logical, CoreType::DRAM);
            MetalContext::instance().get_cluster().write_core(
                payload.data(), page_size, tt_cxy_pair(mesh_device_->build_id(), drisc_virtual), src_l1);

            Program program = CreateProgram();
            CreateKernel(
                program,
                "tests/tt_metal/tt_metal/test_kernels/misc/socket/drisc_d2h_egress.cpp",
                drisc_logical,
                DramConfig{
                    .noc = NOC::NOC_0,
                    .compile_args = {socket.get_config_buffer_address(), src_l1, page_size, res_l1, kNumPages}});

            distributed::MeshWorkload workload;
            workload.add_program(device_range_, std::move(program));
            distributed::EnqueueMeshWorkload(mesh_device_->mesh_command_queue(), workload, false);

            std::vector<uint32_t> sink(page_size / sizeof(uint32_t));
            const auto t0 = std::chrono::steady_clock::now();
            for (uint32_t i = 0; i < kNumPages; i++) {
                socket.read(sink.data(), 1);
            }
            socket.barrier();
            const auto t1 = std::chrono::steady_clock::now();
            distributed::Finish(mesh_device_->mesh_command_queue());

            std::vector<uint32_t> out(5);
            MetalContext::instance().get_cluster().read_core(
                out.data(),
                out.size() * sizeof(uint32_t),
                tt_cxy_pair(mesh_device_->build_id(), drisc_virtual),
                res_l1);
            const uint64_t cycles = (static_cast<uint64_t>(out[1]) << 32) | out[0];
            const uint64_t wait_cycles = (static_cast<uint64_t>(out[3]) << 32) | out[2];
            const uint32_t clk_hz =
                MetalContext::instance().get_cluster().get_device_aiclk(mesh_device_->id()) * 1000000u;
            const uint64_t total_bytes = static_cast<uint64_t>(kNumPages) * page_size;
            const double host_s = std::chrono::duration<double>(t1 - t0).count();

            log_info(
                LogTest,
                "page {:>6} B x {} | device {:>6.3f} GB/s ({:>7.1f} ns/page, {:>5.1f}% spent waiting on host) | "
                "host-observed {:>6.3f} GB/s",
                page_size,
                kNumPages,
                compute_bw_gbs(total_bytes, cycles, clk_hz),
                static_cast<double>(cycles) * 1e9 / clk_hz / kNumPages,
                100.0 * static_cast<double>(wait_cycles) / static_cast<double>(cycles),
                static_cast<double>(total_bytes) / host_s / 1e9);
        }
    }

    set_niu_mode(0);  // restore NOC2AXI -- NIU_CFG_0 persists across programs
}

// What the reserved DRAM profiler region actually gives a DRISC drainer. The region sits at a fixed
// bank-relative offset (DRAM_PROFILER_BASE), so it exists at the same address in every bank -- which is
// the geometry a DRISC needs, since its DMA engine only reaches its own bank. Size is computed from
// runtime options, so it is read from the HAL rather than derived.
TEST_F(DramKernelDRISCScatterFixture, DRISCProfilerDramRegion) {
    const auto& hal = MetalContext::instance().hal();
    const auto& soc_desc = MetalContext::instance().get_cluster().get_soc_desc(mesh_device_->build_id());
    const uint64_t base = hal.get_dev_addr(HalDramMemAddrType::PROFILER);
    const uint64_t per_bank = hal.get_dev_size(HalDramMemAddrType::PROFILER);
    const uint32_t banks = soc_desc.get_dram_compute_grid_size().x;
    const uint64_t view_size = soc_desc.dram_view_size;

    constexpr uint64_t kMarkerBytes = 8;
    constexpr uint64_t kZoneBytes = 16;  // 2 markers
    constexpr double kLegAGbs = 43.18;   // measured: one DRISC, ingest + DMA

    log_info(LogTest, "DRAM profiler region: base 0x{:x}, {} B per bank ({:.2f} MB)", base, per_bank, per_bank / 1e6);
    log_info(
        LogTest,
        "  one DRISC / one bank: {} markers, {} zones | whole bank view is {:.2f} GB",
        per_bank / kMarkerBytes,
        per_bank / kZoneBytes,
        view_size / 1e9);
    log_info(
        LogTest,
        "  fill time at leg A ({:.2f} GB/s): {:.2f} ms | across all {} banks: {} B ({:.2f} MB)",
        kLegAGbs,
        per_bank / (kLegAGbs * 1e9) * 1e3,
        banks,
        per_bank * banks,
        per_bank * banks / 1e6);
}

// Host pull rate from the reserved DRAM profiler region, over the exact path the existing profiler
// uses (cluster.read_dram_vec per channel). This is the number that decides whether a second DRISC is
// needed to drain the DRAM buffer to host at all: if the host can pull fast enough, leg B does not
// need to exist.
TEST_F(DramKernelDRISCScatterFixture, HostPullFromDramProfilerRegion) {
    const auto& hal = MetalContext::instance().hal();
    const auto& cluster = MetalContext::instance().get_cluster();
    const auto& soc_desc = cluster.get_soc_desc(mesh_device_->build_id());
    const uint64_t base = hal.get_dev_addr(HalDramMemAddrType::PROFILER);
    const uint32_t per_bank = hal.get_dev_size(HalDramMemAddrType::PROFILER);
    const uint32_t banks = soc_desc.get_dram_compute_grid_size().x;
    const auto dev_id = mesh_device_->id();
    constexpr uint32_t kRepeats = 5;

    log_info(LogTest, "host pull from DRAM profiler region: base 0x{:x}, {} B/bank, {} banks", base, per_bank, banks);

    std::vector<uint32_t> dst(per_bank / sizeof(uint32_t));
    for (uint32_t size : {64u << 10, 256u << 10, 1u << 20, 4u << 20, per_bank}) {
        if (size > per_bank) {
            continue;
        }
        // Single channel.
        std::vector<double> gbs;
        for (uint32_t rep = 0; rep < kRepeats; rep++) {
            const auto t0 = std::chrono::steady_clock::now();
            cluster.read_dram_vec(dst.data(), size, dev_id, 0, base);
            const auto t1 = std::chrono::steady_clock::now();
            gbs.push_back(size / std::chrono::duration<double>(t1 - t0).count() / 1e9);
        }
        std::sort(gbs.begin(), gbs.end());

        // All channels, as the profiler readback loop does.
        std::vector<double> all;
        for (uint32_t rep = 0; rep < kRepeats; rep++) {
            const auto t0 = std::chrono::steady_clock::now();
            for (uint32_t ch = 0; ch < banks; ch++) {
                cluster.read_dram_vec(dst.data(), size, dev_id, ch, base);
            }
            const auto t1 = std::chrono::steady_clock::now();
            all.push_back(static_cast<double>(size) * banks / std::chrono::duration<double>(t1 - t0).count() / 1e9);
        }
        std::sort(all.begin(), all.end());

        log_info(
            LogTest,
            "  {:>8} B | 1 channel: median {:>6.3f} GB/s (best {:>6.3f}) | all {} channels: median {:>6.3f} GB/s "
            "(best {:>6.3f}) | full {} B region takes {:.2f} ms",
            size,
            gbs[kRepeats / 2],
            gbs.back(),
            banks,
            all[kRepeats / 2],
            all.back(),
            per_bank * banks,
            static_cast<double>(per_bank) * banks / (all[kRepeats / 2] * 1e9) * 1e3);
    }
}

// Fused monitor+drain: no poll phase. One read per core covers the control vector and all five rings
// (profiler_msg_t is contiguous, 10,496 B, still one NoC packet), so the adaptive decision becomes
// "should I push what I already have?" rather than "should I read this core?". Dropping the poll ring
// frees L1 for 8-core batches. Compared directly against the poll-based version at the same hot counts.
TEST_F(DramKernelDRISCScatterFixture, DRISCFusedDrainToHost) {
    constexpr uint32_t kRingCapWords = 512;
    constexpr uint32_t kNumRisc = 5;
    constexpr uint32_t kCvBytes = 64 * sizeof(uint32_t);           // 256
    constexpr uint32_t kPageBytes = kNumRisc * kRingCapWords * 4;  // 10240, one core's rings
    constexpr uint32_t kCoreSpan = kCvBytes + kPageBytes;          // 10496, control vector + rings
    constexpr uint32_t kCoresPerBatch = 8;                         // 83,968 B of the 86 KB L1
    constexpr uint32_t kThresholdWords = 4 * kRingCapWords;
    constexpr uint32_t kNumSweeps = 200;
    constexpr uint32_t kPagesPerRead = 8;

    const std::vector<uint32_t> coords = worker_coords();
    const uint32_t num_cores = static_cast<uint32_t>(coords.size());
    const uint32_t clk_hz = MetalContext::instance().get_cluster().get_device_aiclk(mesh_device_->id()) * 1000000u;
    const auto& soc_desc = MetalContext::instance().get_cluster().get_soc_desc(mesh_device_->build_id());
    const CoreCoord grid = mesh_device_->compute_with_storage_grid_size();

    const uint32_t buf = drisc_l1_base_;
    const uint32_t cfg = buf + kCoresPerBatch * kCoreSpan;
    const uint32_t res = cfg + 1024;
    TT_FATAL(
        (res + 64 - drisc_l1_base_) <= drisc_l1_unreserved_size_,
        "{} x {} B batch overflows {} B of DRISC L1",
        kCoresPerBatch,
        kCoreSpan,
        drisc_l1_unreserved_size_);

    const CoreCoord d_logical = mesh_device_->impl().pick_unused_dram_logical_core(mesh_device_->get_devices()[0], 0);
    const CoreCoord d_virtual = mesh_device_->virtual_core_from_logical_core(d_logical, CoreType::DRAM);
    const CoreCoord d_tr = soc_desc.dram_bank_endpoint_coords.at(d_logical.x).at(d_logical.y);
    const tt::umd::CoreCoord d_phys = soc_desc.translate_coord_to(
        tt::umd::CoreCoord(d_tr.x, d_tr.y, CoreType::DRAM, CoordSystem::TRANSLATED), CoordSystem::NOC0);

    auto set_niu_mode = [&](uint32_t stream) {
        Program p = CreateProgram();
        CreateKernel(
            p,
            "tests/tt_metal/tt_metal/test_kernels/misc/drisc_niu_mode.cpp",
            d_logical,
            DramConfig{.noc = NOC::NOC_0, .compile_args = {stream}});
        run_workload(std::move(p));
    };

    prime_worker_l1(kCoreSpan);
    set_niu_mode(1);
    log_info(
        LogTest,
        "fused drain: {} cores, one {} B read per core (cv {} + data {}), {} cores/batch",
        num_cores,
        kCoreSpan,
        kCvBytes,
        kPageBytes,
        kCoresPerBatch);

    auto run = [&](uint32_t hot) {
        std::vector<uint32_t> cv(kCvBytes / sizeof(uint32_t), 0);
        for (uint32_t c = 0; c < num_cores; c++) {
            const uint32_t tail = (c < hot) ? (kThresholdWords / kNumRisc + 1) : 1u;
            for (uint32_t r = 0; r < kNumRisc; r++) {
                cv[5 + r] = tail;
            }
            cv[0] = 0xA5A50000u + c;
            const CoreCoord v =
                mesh_device_->virtual_core_from_logical_core({c % grid.x, c / grid.x}, CoreType::WORKER);
            MetalContext::instance().get_cluster().write_core(
                cv.data(), cv.size() * sizeof(uint32_t), tt_cxy_pair(mesh_device_->build_id(), v), tensix_l1_base_);
        }

        const uint32_t total_pages = hot * kNumSweeps;

        distributed::D2HSocket socket(
            devices_[0],
            distributed::MeshCoreCoord(distributed::MeshCoordinate(0, 0), CoreCoord(d_phys.x, d_phys.y)),
            32 * kPageBytes,
            distributed::D2HSocket::ExternalConfigBuffer{.address = cfg, .sender_uses_physical_noc_addr = true});
        socket.set_page_size(kPageBytes);

        Program program = CreateProgram();
        auto kid = CreateKernel(
            program,
            "tests/tt_metal/tt_metal/test_kernels/misc/socket/drisc_fused_to_host.cpp",
            d_logical,
            DramConfig{
                .noc = NOC::NOC_0,
                .compile_args = {
                    kCoresPerBatch,
                    kCoreSpan,
                    kCvBytes,
                    kPageBytes,
                    kThresholdWords,
                    buf,
                    socket.get_config_buffer_address(),
                    res}});
        std::vector<uint32_t> rtas = {num_cores, kNumSweeps, tensix_l1_base_};
        rtas.insert(rtas.end(), coords.begin(), coords.end());
        SetRuntimeArgs(program, kid, d_logical, rtas);

        distributed::MeshWorkload workload;
        workload.add_program(device_range_, std::move(program));
        distributed::EnqueueMeshWorkload(mesh_device_->mesh_command_queue(), workload, false);

        std::vector<uint32_t> sink(static_cast<size_t>(kPageBytes) * kPagesPerRead / sizeof(uint32_t));
        uint32_t left = total_pages;
        const auto t0 = std::chrono::steady_clock::now();
        const auto deadline = t0 + std::chrono::seconds(60);
        while (left > 0) {
            const uint32_t avail = socket.pages_available();
            if (avail > 0) {
                const uint32_t take = std::min(avail, std::min(left, kPagesPerRead));
                socket.read(sink.data(), take);
                left -= take;
            }
            if (std::chrono::steady_clock::now() > deadline) {
                log_warning(LogTest, "drain timed out with {} pages left", left);
                break;
            }
        }
        socket.barrier(20000);
        distributed::Finish(mesh_device_->mesh_command_queue());

        std::vector<uint32_t> o(11);
        MetalContext::instance().get_cluster().read_core(
            o.data(), o.size() * sizeof(uint32_t), tt_cxy_pair(mesh_device_->build_id(), d_virtual), res);
        const uint64_t cyc = (static_cast<uint64_t>(o[1]) << 32) | o[0];
        const uint64_t rd = (static_cast<uint64_t>(o[5]) << 32) | o[4];
        const uint64_t dec = (static_cast<uint64_t>(o[7]) << 32) | o[6];
        const uint64_t psh = (static_cast<uint64_t>(o[9]) << 32) | o[8];
        EXPECT_EQ(o[2], total_pages) << "pages pushed != expected";

        const double ns = 1e9 / clk_hz;
        const uint64_t drained = static_cast<uint64_t>(hot) * kPageBytes * kNumSweeps;
        log_info(
            LogTest,
            "  {:>3}/{} hot | {:>7.2f} us/sweep | drained {:>6.2f} GB/s | wbar {:>4.1f}% read {:>5.1f}% "
            "decide+push {:>5.1f}% (push {:>5.1f}%)",
            hot,
            num_cores,
            static_cast<double>(cyc) / kNumSweeps * ns / 1e3,
            drained > 0 ? compute_bw_gbs(drained, cyc, clk_hz) : 0.0,
            100.0 * static_cast<double>(o[3]) / static_cast<double>(cyc),
            100.0 * static_cast<double>(rd) / static_cast<double>(cyc),
            100.0 * static_cast<double>(dec) / static_cast<double>(cyc),
            100.0 * static_cast<double>(psh) / static_cast<double>(cyc));
    };

    for (uint32_t rep = 0; rep < 2; rep++) {
        for (uint32_t hot : {0u, 4u, 12u, 60u, 120u}) {
            run(hot);
        }
    }

    set_niu_mode(0);
}

// Two-tier adaptive drainer: poll the 64-word tails, then per core either take the whole core in one
// bulk read (if ANY lane is >= 70% full) or read just the valid run of each non-empty lane.
//
// Swept by uniform per-lane occupancy so the threshold crossing is visible: below 358 words every core
// takes the partial path, at or above it every core takes the bulk path. The interesting columns are
// the useful (valid) drain rate versus the pushed rate -- bulk moves whole rings whether or not they
// are full, and egress is the scarce resource.
TEST_F(DramKernelDRISCScatterFixture, DRISCTwoTierDrainToHost) {
    constexpr uint32_t kRingCapWords = 512;
    constexpr uint32_t kNumRisc = 5;
    constexpr uint32_t kCvBytes = 256;
    constexpr uint32_t kRingCapBytes = kRingCapWords * 4;                // 2048
    constexpr uint32_t kCoreSpan = kCvBytes + kNumRisc * kRingCapBytes;  // 10496
    constexpr uint32_t kBulkThresholdWords = kRingCapWords * 70 / 100;   // 358 = 70% of a ring
    constexpr uint32_t kNumSweeps = 100;
    constexpr uint32_t kPagesPerRead = 8;

    const std::vector<uint32_t> coords = worker_coords();
    const uint32_t num_cores = static_cast<uint32_t>(coords.size());
    const uint32_t clk_hz = MetalContext::instance().get_cluster().get_device_aiclk(mesh_device_->id()) * 1000000u;
    const auto& soc_desc = MetalContext::instance().get_cluster().get_soc_desc(mesh_device_->build_id());
    const CoreCoord grid = mesh_device_->compute_with_storage_grid_size();

    const uint32_t poll_ring = drisc_l1_base_;
    const uint32_t page_buf = poll_ring + num_cores * kCvBytes;
    const uint32_t head_scratch = page_buf + 65536;  // past the largest page used here

    const CoreCoord d_logical = mesh_device_->impl().pick_unused_dram_logical_core(mesh_device_->get_devices()[0], 0);
    const CoreCoord d_virtual = mesh_device_->virtual_core_from_logical_core(d_logical, CoreType::DRAM);
    const CoreCoord d_tr = soc_desc.dram_bank_endpoint_coords.at(d_logical.x).at(d_logical.y);
    const tt::umd::CoreCoord d_phys = soc_desc.translate_coord_to(
        tt::umd::CoreCoord(d_tr.x, d_tr.y, CoreType::DRAM, CoordSystem::TRANSLATED), CoordSystem::NOC0);

    auto set_niu_mode = [&](uint32_t stream) {
        Program p = CreateProgram();
        CreateKernel(
            p,
            "tests/tt_metal/tt_metal/test_kernels/misc/drisc_niu_mode.cpp",
            d_logical,
            DramConfig{.noc = NOC::NOC_0, .compile_args = {stream}});
        run_workload(std::move(p));
    };

    prime_worker_l1(kCoreSpan);
    set_niu_mode(1);
    log_info(
        LogTest,
        "two-tier drain: {} cores, bulk when any lane >= {} words ({}% of {})",
        num_cores,
        kBulkThresholdWords,
        70,
        kRingCapWords);

    // 40960 is not a multiple of the 10496 B whole-core item, so the bulk tier pads every page.
    // 41984 = 4 x 10496 packs exactly.
    // A threshold no lane can reach forces the partial tier at every occupancy: always poll the tails,
    // always read each RISC individually, sized to its run. Never bulk.
    constexpr uint32_t kNeverBulk = 0xFFFFFFFFu;
    auto run = [&](uint32_t tail_words,
                   uint32_t kPageBytes,
                   uint32_t hysteresis,
                   uint32_t no_adaptive = 0,
                   uint32_t threshold = kBulkThresholdWords,
                   uint32_t paced = 0,
                   uint32_t prod_mw_per_us = 0,
                   uint32_t writeback = 0) {
        const uint32_t cfg = page_buf + kPageBytes;
        const uint32_t res = cfg + 1024;
        TT_FATAL(
            (res + 64 - drisc_l1_base_) <= drisc_l1_unreserved_size_,
            "poll ring {} B + page {} B overflow {} B",
            num_cores * kCvBytes,
            kPageBytes,
            drisc_l1_unreserved_size_);
        // Prime the control vector at the enum-derived offsets. The identity word stands in for what
        // BRISC FW would write once at init.
        std::vector<uint32_t> cv(kCvBytes / sizeof(uint32_t), 0);
        for (uint32_t r = 0; r < kNumRisc; r++) {
            cv[kernel_profiler::SPSC_RING_TAIL_0 + r] = tail_words;
        }
        for (uint32_t c = 0; c < num_cores; c++) {
            cv[kernel_profiler::SPSC_CORE_XY] = ((c / grid.x) << 16) | (c % grid.x);  // (y<<16)|x
            cv[kernel_profiler::SPSC_RING_HEAD_0] = 0xA5A50000u + c;
            const CoreCoord v =
                mesh_device_->virtual_core_from_logical_core({c % grid.x, c / grid.x}, CoreType::WORKER);
            MetalContext::instance().get_cluster().write_core(
                cv.data(), cv.size() * sizeof(uint32_t), tt_cxy_pair(mesh_device_->build_id(), v), tensix_l1_base_);
        }

        // Mirror the kernel's packing exactly so the host knows how many pages to expect.
        const bool bulk = paced || ((no_adaptive == 0) && tail_words >= threshold);
        const uint32_t lane_bytes = no_adaptive ? kRingCapBytes : ((tail_words * 4u + 15u) & ~15u);
        uint64_t fill = 0;
        uint64_t exp_pages = 0;
        uint64_t exp_valid = 0;
        for (uint32_t s = 0; s < kNumSweeps; s++) {
            for (uint32_t c = 0; c < num_cores; c++) {
                const uint32_t n_items = bulk ? 1u : kNumRisc;
                for (uint32_t it = 0; it < n_items; it++) {
                    const uint32_t bytes = bulk ? kCoreSpan : lane_bytes;
                    if (bytes == 0) {
                        continue;
                    }
                    if (fill + bytes > kPageBytes) {
                        exp_pages++;
                        fill = 0;
                    }
                    fill += bytes;
                    exp_valid += bytes;
                }
            }
        }
        if (fill > 0) {
            exp_pages++;
        }

        distributed::D2HSocket socket(
            devices_[0],
            distributed::MeshCoreCoord(distributed::MeshCoordinate(0, 0), CoreCoord(d_phys.x, d_phys.y)),
            32 * kPageBytes,
            distributed::D2HSocket::ExternalConfigBuffer{.address = cfg, .sender_uses_physical_noc_addr = true});
        socket.set_page_size(kPageBytes);

        Program program = CreateProgram();
        auto kid = CreateKernel(
            program,
            "tests/tt_metal/tt_metal/test_kernels/misc/socket/drisc_two_tier_to_host.cpp",
            d_logical,
            DramConfig{
                .noc = NOC::NOC_0,
                .compile_args = {
                    kCvBytes,
                    kRingCapBytes,
                    kCoreSpan,
                    threshold,
                    kPageBytes,
                    poll_ring,
                    page_buf,
                    socket.get_config_buffer_address(),
                    res,
                    hysteresis,
                    no_adaptive,
                    paced,
                    prod_mw_per_us,
                    kRingCapWords * 90 / 100,  // high watermark: 90% full
                    kRingCapWords * 80 / 100,  // low watermark: 80% full
                    clk_hz / 1000000u,
                    6750u,  // delay step ~5 us
                    writeback,
                    head_scratch}});
        std::vector<uint32_t> rtas = {num_cores, kNumSweeps, tensix_l1_base_, tensix_l1_base_ + kCvBytes};
        rtas.insert(rtas.end(), coords.begin(), coords.end());
        SetRuntimeArgs(program, kid, d_logical, rtas);

        distributed::MeshWorkload workload;
        workload.add_program(device_range_, std::move(program));
        distributed::EnqueueMeshWorkload(mesh_device_->mesh_command_queue(), workload, false);

        std::vector<uint32_t> sink(static_cast<size_t>(kPageBytes) * kPagesPerRead / sizeof(uint32_t));
        uint64_t left = paced ? static_cast<uint64_t>(kNumSweeps) * num_cores * kCoreSpan / kPageBytes : exp_pages;
        const auto t0 = std::chrono::steady_clock::now();
        const auto deadline = t0 + std::chrono::seconds(90);
        while (left > 0) {
            const uint32_t avail = socket.pages_available();
            if (avail > 0) {
                const uint32_t take =
                    static_cast<uint32_t>(std::min<uint64_t>(avail, std::min<uint64_t>(left, kPagesPerRead)));
                socket.read(sink.data(), take);
                left -= take;
            }
            if (std::chrono::steady_clock::now() > deadline) {
                log_warning(LogTest, "drain timed out with {} pages left of {}", left, exp_pages);
                break;
            }
        }
        socket.barrier(20000);
        distributed::Finish(mesh_device_->mesh_command_queue());

        std::vector<uint32_t> o(26);
        MetalContext::instance().get_cluster().read_core(
            o.data(), o.size() * sizeof(uint32_t), tt_cxy_pair(mesh_device_->build_id(), d_virtual), res);
        const uint64_t cyc = (static_cast<uint64_t>(o[1]) << 32) | o[0];
        const uint64_t valid = (static_cast<uint64_t>(o[6]) << 32) | o[5];
        const uint64_t poll = (static_cast<uint64_t>(o[8]) << 32) | o[7];
        const uint64_t fetch = (static_cast<uint64_t>(o[11]) << 32) | o[10];
        const uint64_t push = (static_cast<uint64_t>(o[13]) << 32) | o[12];
        if (!paced) {
            EXPECT_EQ(o[2], exp_pages) << "kernel pages != host packing model";
            EXPECT_EQ(valid, exp_valid) << "kernel valid bytes != host model";
        }

        const uint64_t pushed = static_cast<uint64_t>(o[2]) * kPageBytes;
        const uint64_t live = (static_cast<uint64_t>(o[16]) << 32) | o[15];
        const double ns = 1e9 / clk_hz;
        if (paced) {
            log_info(
                LogTest,
                "  prod {:>5} mw/us/lane wb={} | {:>8.1f} us/sweep | live {:>6.2f} GB/s | WASTE {:>5.1f}% | avg "
                "occ {:>3}/{} ({:>4.1f}%) | overflow {:>3} | wait {:>4.1f}% | head-wb {:>4.1f}%",
                prod_mw_per_us,
                writeback,
                static_cast<double>(cyc) / kNumSweeps * ns / 1e3,
                compute_bw_gbs(live, cyc, clk_hz),
                100.0 * (1.0 - static_cast<double>(live) / static_cast<double>(pushed)),
                o[18],
                kRingCapWords,
                100.0 * o[18] / kRingCapWords,
                o[17],
                100.0 * static_cast<double>((static_cast<uint64_t>(o[20]) << 32) | o[19]) / static_cast<double>(cyc),
                100.0 * static_cast<double>((static_cast<uint64_t>(o[23]) << 32) | o[22]) / static_cast<double>(cyc));
            return;
        }
        log_info(
            LogTest,
            "  {:>3} w/lane ({:>4.1f}% full) {:<8} | {:>7.2f} us/sweep | live {:>6.2f} | xfer {:>6.2f} | pushed "
            "{:>6.2f} GB/s | WASTE {:>5.1f}% | poll {:>4.1f}% decide {:>4.1f}% fetch {:>4.1f}% push {:>4.1f}%",
            tail_words,
            100.0 * tail_words / kRingCapWords,
            bulk ? "BULK" : "PARTIAL",
            static_cast<double>(cyc) / kNumSweeps * ns / 1e3,
            compute_bw_gbs(live, cyc, clk_hz),
            compute_bw_gbs(valid, cyc, clk_hz),
            compute_bw_gbs(pushed, cyc, clk_hz),
            100.0 * (1.0 - static_cast<double>(live) / static_cast<double>(pushed)),
            100.0 * static_cast<double>(poll) / static_cast<double>(cyc),
            100.0 * static_cast<double>(o[9]) / static_cast<double>(cyc),
            100.0 * static_cast<double>(fetch) / static_cast<double>(cyc),
            100.0 * static_cast<double>(push) / static_cast<double>(cyc));
    };

    run(256u, 41984u, 0u, 0u, kNeverBulk);  // warm the host staging allocation
    log_info(LogTest, "-- paced always-bulk, wb=0 no head write-back / wb=1 with it --");
    for (uint32_t wb : {0u, 1u}) {
        for (uint32_t mw : {2000u, 4000u, 7500u, 12000u}) {
            run(480u, 41984u, 0u, 0u, 0u, /*paced=*/1u, mw, wb);
        }
    }
    for (uint32_t rep = 0; rep < 0; rep++) {
        // Threshold sweep: at each occupancy run both tiers unconditionally (threshold 0 = always bulk,
        // kNeverBulk = always partial) and compare live-marker throughput. Where BULK overtakes PARTIAL
        // is where the threshold belongs.
        for (uint32_t t : {32u, 64u, 128u, 192u, 256u, 320u, 384u, 448u, 512u}) {
            run(t, 41984u, 0u, 0u, kNeverBulk);
            run(t, 41984u, 0u, 0u, 0u);
        }
    }

    set_niu_mode(0);
}

// The complete drainer: monitor the 64-word control vector on every core, run the adaptive
// threshold, whole-core-read only the cores that trip it, and push those to the host.
//
// The host sets each core's tails so a chosen number trip ADAPT_THRESH. Hot counts are multiples of
// the page width so every sweep produces whole pages and the host knows exactly how many to expect.
TEST_F(DramKernelDRISCScatterFixture, DRISCAdaptiveDrainToHost) {
    constexpr uint32_t kRingCapWords = 512;  // PROFILER_L1_VECTOR_SIZE
    constexpr uint32_t kNumRisc = 5;
    constexpr uint32_t kCvWords = 64;                                 // PROFILER_L1_CONTROL_VECTOR_SIZE
    constexpr uint32_t kPollBytes = kCvWords * sizeof(uint32_t);      // 256
    constexpr uint32_t kBytesPerCore = kNumRisc * kRingCapWords * 4;  // 10240
    constexpr uint32_t kCoresPerPage = 4;  // 40960 B page; 8 would not fit beside the poll ring
    constexpr uint32_t kPageBytes = kCoresPerPage * kBytesPerCore;
    constexpr uint32_t kThresholdWords = 4 * kRingCapWords;  // ADAPT_THRESH
    constexpr uint32_t kNumSweeps = 200;
    constexpr uint32_t kPagesPerRead = 8;

    const std::vector<uint32_t> coords = worker_coords();
    const uint32_t num_cores = static_cast<uint32_t>(coords.size());
    const uint32_t clk_hz = MetalContext::instance().get_cluster().get_device_aiclk(mesh_device_->id()) * 1000000u;
    const auto& soc_desc = MetalContext::instance().get_cluster().get_soc_desc(mesh_device_->build_id());
    const CoreCoord grid = mesh_device_->compute_with_storage_grid_size();

    const uint32_t poll_ring = drisc_l1_base_;
    const uint32_t page_buf = poll_ring + num_cores * kPollBytes;
    const uint32_t cfg = page_buf + kPageBytes;
    const uint32_t res = cfg + 1024;
    TT_FATAL(
        (res + 64 - drisc_l1_base_) <= drisc_l1_unreserved_size_,
        "poll ring {} B + page {} B overflow {} B of DRISC L1",
        num_cores * kPollBytes,
        kPageBytes,
        drisc_l1_unreserved_size_);

    const CoreCoord d_logical = mesh_device_->impl().pick_unused_dram_logical_core(mesh_device_->get_devices()[0], 0);
    const CoreCoord d_virtual = mesh_device_->virtual_core_from_logical_core(d_logical, CoreType::DRAM);
    const CoreCoord d_tr = soc_desc.dram_bank_endpoint_coords.at(d_logical.x).at(d_logical.y);
    const tt::umd::CoreCoord d_phys = soc_desc.translate_coord_to(
        tt::umd::CoreCoord(d_tr.x, d_tr.y, CoreType::DRAM, CoordSystem::TRANSLATED), CoordSystem::NOC0);

    auto set_niu_mode = [&](uint32_t stream) {
        Program p = CreateProgram();
        CreateKernel(
            p,
            "tests/tt_metal/tt_metal/test_kernels/misc/drisc_niu_mode.cpp",
            d_logical,
            DramConfig{.noc = NOC::NOC_0, .compile_args = {stream}});
        run_workload(std::move(p));
    };

    // Ring payload is the same on every core and never changes; only the control vector is rewritten.
    prime_worker_l1(kPollBytes + kBytesPerCore);
    set_niu_mode(1);
    log_info(
        LogTest,
        "adaptive drain to host: {} cores, poll {} B, whole-core {} B, page {} cores, ADAPT_THRESH {} words",
        num_cores,
        kPollBytes,
        kBytesPerCore,
        kCoresPerPage,
        kThresholdWords);

    auto run = [&](uint32_t hot) {
        // Tails high enough that 5 lanes cross the threshold, or low enough that they never do.
        std::vector<uint32_t> cv(kCvWords, 0);
        for (uint32_t c = 0; c < num_cores; c++) {
            const uint32_t tail = (c < hot) ? (kThresholdWords / kNumRisc + 1) : 1u;
            for (uint32_t r = 0; r < kNumRisc; r++) {
                cv[5 + r] = tail;
            }
            cv[0] = 0xA5A50000u + c;
            const CoreCoord v =
                mesh_device_->virtual_core_from_logical_core({c % grid.x, c / grid.x}, CoreType::WORKER);
            MetalContext::instance().get_cluster().write_core(
                cv.data(), cv.size() * sizeof(uint32_t), tt_cxy_pair(mesh_device_->build_id(), v), tensix_l1_base_);
        }

        const uint32_t pages_per_sweep = hot / kCoresPerPage;
        const uint32_t total_pages = pages_per_sweep * kNumSweeps;

        distributed::D2HSocket socket(
            devices_[0],
            distributed::MeshCoreCoord(distributed::MeshCoordinate(0, 0), CoreCoord(d_phys.x, d_phys.y)),
            32 * kPageBytes,
            distributed::D2HSocket::ExternalConfigBuffer{.address = cfg, .sender_uses_physical_noc_addr = true});
        socket.set_page_size(kPageBytes);

        Program program = CreateProgram();
        auto kid = CreateKernel(
            program,
            "tests/tt_metal/tt_metal/test_kernels/misc/socket/drisc_adaptive_to_host.cpp",
            d_logical,
            DramConfig{
                .noc = NOC::NOC_0,
                .compile_args = {
                    kPollBytes,
                    kBytesPerCore,
                    kCoresPerPage,
                    kThresholdWords,
                    poll_ring,
                    page_buf,
                    socket.get_config_buffer_address(),
                    res}});
        std::vector<uint32_t> rtas = {num_cores, kNumSweeps, tensix_l1_base_, tensix_l1_base_ + kPollBytes};
        rtas.insert(rtas.end(), coords.begin(), coords.end());
        SetRuntimeArgs(program, kid, d_logical, rtas);

        distributed::MeshWorkload workload;
        workload.add_program(device_range_, std::move(program));
        distributed::EnqueueMeshWorkload(mesh_device_->mesh_command_queue(), workload, false);

        std::vector<uint32_t> sink(static_cast<size_t>(kPageBytes) * kPagesPerRead / sizeof(uint32_t));
        uint32_t left = total_pages;
        const auto t0 = std::chrono::steady_clock::now();
        const auto deadline = t0 + std::chrono::seconds(60);
        while (left > 0) {
            const uint32_t avail = socket.pages_available();
            if (avail > 0) {
                const uint32_t take = std::min(avail, std::min(left, kPagesPerRead));
                socket.read(sink.data(), take);
                left -= take;
            }
            if (std::chrono::steady_clock::now() > deadline) {
                log_warning(LogTest, "drain timed out with {} pages left", left);
                break;
            }
        }
        socket.barrier(20000);
        distributed::Finish(mesh_device_->mesh_command_queue());

        std::vector<uint32_t> o(13);
        MetalContext::instance().get_cluster().read_core(
            o.data(), o.size() * sizeof(uint32_t), tt_cxy_pair(mesh_device_->build_id(), d_virtual), res);
        const uint64_t cyc = (static_cast<uint64_t>(o[1]) << 32) | o[0];
        const uint64_t poll = (static_cast<uint64_t>(o[5]) << 32) | o[4];
        const uint64_t dec = (static_cast<uint64_t>(o[7]) << 32) | o[6];
        const uint64_t bulk = (static_cast<uint64_t>(o[9]) << 32) | o[8];
        const uint64_t push = (static_cast<uint64_t>(o[11]) << 32) | o[10];
        EXPECT_EQ(o[2], total_pages) << "pages pushed != expected";
        EXPECT_EQ(o[3], hot * kNumSweeps) << "bulk count != expected";

        const double ns = 1e9 / clk_hz;
        const double us_sweep = static_cast<double>(cyc) / kNumSweeps * ns / 1e3;
        const uint64_t drained = static_cast<uint64_t>(hot) * kBytesPerCore * kNumSweeps;
        log_info(
            LogTest,
            "  {:>3}/{} hot | {:>7.2f} us/sweep | drained {:>6.2f} GB/s | poll {:>5.1f}% decide {:>5.1f}% "
            "bulk {:>5.1f}% push {:>5.1f}%",
            hot,
            num_cores,
            us_sweep,
            drained > 0 ? compute_bw_gbs(drained, cyc, clk_hz) : 0.0,
            100.0 * static_cast<double>(poll) / static_cast<double>(cyc),
            100.0 * static_cast<double>(dec) / static_cast<double>(cyc),
            100.0 * static_cast<double>(bulk) / static_cast<double>(cyc),
            100.0 * static_cast<double>(push) / static_cast<double>(cyc));
    };

    for (uint32_t rep = 0; rep < 2; rep++) {
        for (uint32_t hot : {0u, 4u, 12u, 60u, 120u}) {
            run(hot);
        }
    }

    set_niu_mode(0);
}

// Scale the direct drainer: N DRISCs, each owning a slice of the grid and pushing to its own D2H
// socket. Each reader is also its own egress path, so there is no relay and no shared ring.
//
// The grid is split by whole pages (15 pages of 8 cores), so slices are as even as 15/N allows:
// N=2 -> 8+7, N=3 -> 5+5+5, N=4 -> 4+4+4+3. The largest slice sets the sweep time.
//
// The host drains all N sockets round-robin on one thread. That is a real serialisation, so each N is
// run twice: with the memcpy consumer (end-to-end truth for a single-threaded host) and with
// discard_pending_pages (host cost removed, so device-side scaling is visible).
TEST_F(DramKernelDRISCScatterFixture, DRISCMultiDrainerScaling) {
    constexpr uint32_t kBytesPerCore = 1280 * 8;  // 10240, whole core
    constexpr uint32_t kCoresPerPage = 8;         // 81920 B page, the measured-best size
    constexpr uint32_t kPageBytes = kCoresPerPage * kBytesPerCore;
    constexpr uint32_t kNumSweeps = 200;
    constexpr uint32_t kPagesPerRead = 8;
    constexpr uint32_t kFifoPages = 32;

    const std::vector<uint32_t> coords = worker_coords();
    const uint32_t num_cores = static_cast<uint32_t>(coords.size());
    TT_FATAL(num_cores % kCoresPerPage == 0, "{} cores must divide by {}", num_cores, kCoresPerPage);
    const uint32_t total_pages_per_sweep = num_cores / kCoresPerPage;
    const uint32_t clk_hz = MetalContext::instance().get_cluster().get_device_aiclk(mesh_device_->id()) * 1000000u;
    const auto& soc_desc = MetalContext::instance().get_cluster().get_soc_desc(mesh_device_->build_id());
    const uint32_t banks = soc_desc.get_dram_compute_grid_size().x;

    const uint32_t buf = drisc_l1_base_;
    const uint32_t cfg = buf + kPageBytes;
    const uint32_t res = cfg + 1024;
    TT_FATAL((res + 64 - drisc_l1_base_) <= drisc_l1_unreserved_size_, "layout overflows DRISC L1");

    auto set_niu_modes = [&](const std::vector<CoreCoord>& cores, uint32_t stream) {
        Program p = CreateProgram();
        for (const auto& c : cores) {
            CreateKernel(
                p,
                "tests/tt_metal/tt_metal/test_kernels/misc/drisc_niu_mode.cpp",
                c,
                DramConfig{.noc = NOC::NOC_0, .compile_args = {stream}});
        }
        run_workload(std::move(p));
    };

    prime_worker_l1(kBytesPerCore);
    log_info(
        LogTest,
        "multi-drainer: {} cores, {} pages/sweep of {} B, {} banks available",
        num_cores,
        total_pages_per_sweep,
        kPageBytes,
        banks);

    auto run = [&](uint32_t n_drisc, bool discard) {
        std::vector<CoreCoord> logical;
        std::vector<CoreCoord> virt;
        std::vector<tt::umd::CoreCoord> phys;
        for (uint32_t i = 0; i < n_drisc; i++) {
            const CoreCoord lg = mesh_device_->impl().pick_unused_dram_logical_core(mesh_device_->get_devices()[0], i);
            logical.push_back(lg);
            virt.push_back(mesh_device_->virtual_core_from_logical_core(lg, CoreType::DRAM));
            const CoreCoord tr = soc_desc.dram_bank_endpoint_coords.at(lg.x).at(lg.y);
            phys.push_back(soc_desc.translate_coord_to(
                tt::umd::CoreCoord(tr.x, tr.y, CoreType::DRAM, CoordSystem::TRANSLATED), CoordSystem::NOC0));
        }
        set_niu_modes(logical, 1);

        // Split the sweep into whole pages, as evenly as it divides.
        std::vector<uint32_t> pages(n_drisc, total_pages_per_sweep / n_drisc);
        for (uint32_t i = 0; i < total_pages_per_sweep % n_drisc; i++) {
            pages[i]++;
        }

        std::vector<std::unique_ptr<distributed::D2HSocket>> sockets;
        Program program = CreateProgram();
        uint32_t core_off = 0;
        for (uint32_t i = 0; i < n_drisc; i++) {
            sockets.push_back(std::make_unique<distributed::D2HSocket>(
                devices_[0],
                distributed::MeshCoreCoord(distributed::MeshCoordinate(0, 0), CoreCoord(phys[i].x, phys[i].y)),
                kFifoPages * kPageBytes,
                distributed::D2HSocket::ExternalConfigBuffer{.address = cfg, .sender_uses_physical_noc_addr = true}));
            sockets.back()->set_page_size(kPageBytes);

            auto kid = CreateKernel(
                program,
                "tests/tt_metal/tt_metal/test_kernels/misc/socket/drisc_ingest_to_host.cpp",
                logical[i],
                DramConfig{
                    .noc = NOC::NOC_0,
                    .compile_args = {
                        kCoresPerPage,
                        kBytesPerCore,
                        1u,  // one page buffer -- measured best for the socket path
                        buf,
                        sockets.back()->get_config_buffer_address(),
                        res}});
            const uint32_t slice_cores = pages[i] * kCoresPerPage;
            std::vector<uint32_t> rtas = {slice_cores, kNumSweeps, tensix_l1_base_};
            rtas.insert(rtas.end(), coords.begin() + core_off, coords.begin() + core_off + slice_cores);
            SetRuntimeArgs(program, kid, logical[i], rtas);
            core_off += slice_cores;
        }

        distributed::MeshWorkload workload;
        workload.add_program(device_range_, std::move(program));
        distributed::EnqueueMeshWorkload(mesh_device_->mesh_command_queue(), workload, false);

        std::vector<uint32_t> sink(static_cast<size_t>(kPageBytes) * kPagesPerRead / sizeof(uint32_t));
        std::vector<uint32_t> remaining(n_drisc);
        for (uint32_t i = 0; i < n_drisc; i++) {
            remaining[i] = pages[i] * kNumSweeps;
        }
        const auto t0 = std::chrono::steady_clock::now();
        const auto deadline = t0 + std::chrono::seconds(60);
        bool timed_out = false;
        for (;;) {
            uint64_t left = 0;
            for (uint32_t i = 0; i < n_drisc; i++) {
                if (remaining[i] == 0) {
                    continue;
                }
                // Check availability first in both modes. Spinning on discard_pending_pages() with
                // nothing pending issues a notify_sender() PCIe write per call, and at round-robin rate
                // across several sockets that is enough to wedge the card.
                const uint32_t avail = sockets[i]->pages_available();
                if (avail == 0) {
                    continue;
                }
                if (discard) {
                    const uint32_t d = sockets[i]->discard_pending_pages();
                    remaining[i] -= std::min(d, remaining[i]);
                } else {
                    const uint32_t take = std::min(avail, std::min(remaining[i], kPagesPerRead));
                    sockets[i]->read(sink.data(), take);
                    remaining[i] -= take;
                }
            }
            for (uint32_t i = 0; i < n_drisc; i++) {
                left += remaining[i];
            }
            if (left == 0) {
                break;
            }
            if (std::chrono::steady_clock::now() > deadline) {
                log_warning(LogTest, "drain timed out with {} pages left", left);
                timed_out = true;
                break;
            }
        }
        for (auto& s : sockets) {
            s->barrier(20000);
        }
        const auto t1 = std::chrono::steady_clock::now();
        distributed::Finish(mesh_device_->mesh_command_queue());

        uint64_t max_cyc = 0;
        double worst_reserve = 0.0;
        for (uint32_t i = 0; i < n_drisc; i++) {
            std::vector<uint32_t> o(10);
            MetalContext::instance().get_cluster().read_core(
                o.data(), o.size() * sizeof(uint32_t), tt_cxy_pair(mesh_device_->build_id(), virt[i]), res);
            const uint64_t cyc = (static_cast<uint64_t>(o[1]) << 32) | o[0];
            const uint64_t rsv = (static_cast<uint64_t>(o[7]) << 32) | o[6];
            max_cyc = std::max(max_cyc, cyc);
            worst_reserve = std::max(worst_reserve, 100.0 * static_cast<double>(rsv) / static_cast<double>(cyc));
            EXPECT_EQ(o[2], pages[i] * kNumSweeps) << "DRISC " << i << " did not push all pages";
        }
        const uint64_t total_bytes = static_cast<uint64_t>(kNumSweeps) * num_cores * kBytesPerCore;
        const double host_s = std::chrono::duration<double>(t1 - t0).count();

        std::string split;
        for (uint32_t i = 0; i < n_drisc; i++) {
            split += fmt::format("{}{}", i ? "+" : "", pages[i]);
        }
        log_info(
            LogTest,
            "  {} DRISC ({:<9}) {:<8} | device {:>6.2f} GB/s | host {:>6.2f} GB/s | worst reserve {:>5.1f}%{}",
            n_drisc,
            split,
            discard ? "DISCARD" : "memcpy",
            compute_bw_gbs(total_bytes, max_cyc, clk_hz),
            static_cast<double>(total_bytes) / host_s / 1e9,
            worst_reserve,
            timed_out ? "  [TIMED OUT]" : "");

        sockets.clear();
        set_niu_modes(logical, 0);
    };

    // Warm the host staging allocation once so the first timed run is not paying first-touch faults.
    run(1, /*discard=*/false);
    for (uint32_t n : {1u, 2u, 3u, 4u}) {
        run(n, /*discard=*/false);
    }
    // NOTE: the discard (zero-cost consumer) variant is deliberately not swept past one DRISC. With the
    // host consumer removed, two DRISCs push hard enough to saturate the single PCIe link, and the
    // host's own small MMIO reads get starved past UMD's 2 ms budget -- reproducibly fatal
    // ("MMIO per-op timeout: 4B load took 220 ms"), taking the card down with it. The memcpy consumer
    // throttles the senders and keeps the link out of saturation.
    run(1, /*discard=*/true);
}

// The direct alternative to the two-DRISC DRAM pipeline: one DRISC reads whole cores out of worker L1
// straight into a socket page and pushes it to the host. No DRAM hop, no second core.
TEST_F(DramKernelDRISCScatterFixture, DRISCIngestStraightToHost) {
    constexpr uint32_t kBytesPerCore = 1280 * 8;  // 10240, whole core
    constexpr uint32_t kNumSweeps = 200;          // 200 x 1.2288 MB = 246 MB, same volume as the pipeline
    constexpr uint32_t kPagesPerRead = 8;         // host-side batching, the measured-best setting

    const std::vector<uint32_t> coords = worker_coords();
    const uint32_t num_cores = static_cast<uint32_t>(coords.size());
    const uint32_t clk_hz = MetalContext::instance().get_cluster().get_device_aiclk(mesh_device_->id()) * 1000000u;

    const auto& soc_desc = MetalContext::instance().get_cluster().get_soc_desc(mesh_device_->build_id());
    const CoreCoord d_logical = mesh_device_->impl().pick_unused_dram_logical_core(mesh_device_->get_devices()[0], 0);
    const CoreCoord d_virtual = mesh_device_->virtual_core_from_logical_core(d_logical, CoreType::DRAM);
    const CoreCoord d_translated = soc_desc.dram_bank_endpoint_coords.at(d_logical.x).at(d_logical.y);
    const tt::umd::CoreCoord d_phys = soc_desc.translate_coord_to(
        tt::umd::CoreCoord(d_translated.x, d_translated.y, CoreType::DRAM, CoordSystem::TRANSLATED), CoordSystem::NOC0);

    auto set_niu_mode = [&](uint32_t stream) {
        Program p = CreateProgram();
        CreateKernel(
            p,
            "tests/tt_metal/tt_metal/test_kernels/misc/drisc_niu_mode.cpp",
            d_logical,
            DramConfig{.noc = NOC::NOC_0, .compile_args = {stream}});
        run_workload(std::move(p));
    };

    set_niu_mode(1);
    prime_worker_l1(kBytesPerCore);
    log_info(LogTest, "ingest straight to host: 1 DRISC at {},{}, {} cores", d_logical.x, d_logical.y, num_cores);

    auto run = [&](uint32_t cores_per_page, uint32_t num_buffers, bool discard = false) {
        const uint32_t page_bytes = cores_per_page * kBytesPerCore;
        const uint32_t buf = drisc_l1_base_;
        const uint32_t cfg = buf + num_buffers * page_bytes;
        const uint32_t res = cfg + 1024;
        TT_FATAL(
            (res + 64 - drisc_l1_base_) <= drisc_l1_unreserved_size_,
            "{} buffers x {} B overflow {} B of DRISC L1",
            num_buffers,
            page_bytes,
            drisc_l1_unreserved_size_);
        TT_FATAL(num_cores % cores_per_page == 0, "{} cores must divide by {}", num_cores, cores_per_page);
        const uint32_t total_pages = kNumSweeps * (num_cores / cores_per_page);
        TT_FATAL(total_pages % kPagesPerRead == 0, "{} pages must divide by {}", total_pages, kPagesPerRead);

        distributed::D2HSocket socket(
            devices_[0],
            distributed::MeshCoreCoord(distributed::MeshCoordinate(0, 0), CoreCoord(d_phys.x, d_phys.y)),
            32 * page_bytes,
            distributed::D2HSocket::ExternalConfigBuffer{.address = cfg, .sender_uses_physical_noc_addr = true});
        socket.set_page_size(page_bytes);

        Program program = CreateProgram();
        auto kid = CreateKernel(
            program,
            "tests/tt_metal/tt_metal/test_kernels/misc/socket/drisc_ingest_to_host.cpp",
            d_logical,
            DramConfig{
                .noc = NOC::NOC_0,
                .compile_args = {
                    cores_per_page, kBytesPerCore, num_buffers, buf, socket.get_config_buffer_address(), res}});
        std::vector<uint32_t> rtas = {num_cores, kNumSweeps, tensix_l1_base_};
        rtas.insert(rtas.end(), coords.begin(), coords.end());
        SetRuntimeArgs(program, kid, d_logical, rtas);

        distributed::MeshWorkload workload;
        workload.add_program(device_range_, std::move(program));
        distributed::EnqueueMeshWorkload(mesh_device_->mesh_command_queue(), workload, false);

        std::vector<uint32_t> sink(static_cast<size_t>(page_bytes) * kPagesPerRead / sizeof(uint32_t));
        const auto t0 = std::chrono::steady_clock::now();
        if (discard) {
            // Ack without reading the data region: isolates the memcpy from everything else.
            uint32_t done = 0;
            const auto deadline = t0 + std::chrono::seconds(30);
            while (done < total_pages) {
                done += socket.discard_pending_pages();
                if (std::chrono::steady_clock::now() > deadline) {
                    log_warning(LogTest, "discard timed out at {}/{}", done, total_pages);
                    break;
                }
            }
        } else {
            for (uint32_t p = 0; p < total_pages / kPagesPerRead; p++) {
                socket.read(sink.data(), kPagesPerRead);
            }
        }
        socket.barrier(20000);
        const auto t1 = std::chrono::steady_clock::now();
        distributed::Finish(mesh_device_->mesh_command_queue());

        std::vector<uint32_t> o(10);
        MetalContext::instance().get_cluster().read_core(
            o.data(), o.size() * sizeof(uint32_t), tt_cxy_pair(mesh_device_->build_id(), d_virtual), res);
        const uint64_t cyc = (static_cast<uint64_t>(o[1]) << 32) | o[0];
        const uint64_t rd = (static_cast<uint64_t>(o[5]) << 32) | o[4];
        const uint64_t rsv = (static_cast<uint64_t>(o[7]) << 32) | o[6];
        const uint64_t psh = (static_cast<uint64_t>(o[9]) << 32) | o[8];
        const uint64_t total_bytes = static_cast<uint64_t>(kNumSweeps) * num_cores * kBytesPerCore;
        const double host_s = std::chrono::duration<double>(t1 - t0).count();
        EXPECT_EQ(o[2], total_pages) << "not all pages pushed";

        log_info(
            LogTest,
            "  {}buf x {:>2} cores ({:>6} B page) {:<8} | dev {:>6.2f} GB/s | host {:>6.2f} GB/s | wbar {:>4.1f}% "
            "read {:>4.1f}% reserve {:>4.1f}% push {:>4.1f}%",
            num_buffers,
            cores_per_page,
            page_bytes,
            discard ? "DISCARD" : "memcpy",
            compute_bw_gbs(total_bytes, cyc, clk_hz),
            static_cast<double>(total_bytes) / host_s / 1e9,
            100.0 * static_cast<double>(o[3]) / static_cast<double>(cyc),
            100.0 * static_cast<double>(rd) / static_cast<double>(cyc),
            100.0 * static_cast<double>(rsv) / static_cast<double>(cyc),
            100.0 * static_cast<double>(psh) / static_cast<double>(cyc));
    };

    for (uint32_t rep = 0; rep < 2; rep++) {
        run(4, 2);
        run(8, 1);
        run(8, 1, /*discard=*/true);
        run(4, 2, /*discard=*/true);
    }

    set_niu_mode(0);
}

// Full circle: DRISC A sweeps the worker grid into a DRAM ping/pong buffer, DRISC B drains the other
// buffer to host over a D2H socket, and the two exchange credits so A blocks when B falls behind.
//
// A is on bank 0's free subchannel and DMAs into bank 0's slice of the reserved DRAM profiler region.
// B is on bank 1's free subchannel and NoC-reads bank 0's DRAM through bank 0's NOC1 worker endpoint,
// which stays in NOC2AXI and therefore forwards DRAM-range addresses to GDDR. B cannot use DMA for
// this: gddr_dma only reaches the DRISC's own bank.
//
// The headline is A's block time: it is how much of the pipeline A spends waiting on B.
TEST_F(DramKernelDRISCScatterFixture, DRISCPipelineAtoBtoHost) {
    constexpr uint32_t kBytesPerCore = 1280 * 8;  // 10240, whole core
    constexpr uint32_t kCoresPerBatch = 4;        // measured-optimal A geometry
    constexpr uint32_t kPageBytes = 81920;        // B's socket page, also its L1 staging buffer
    constexpr uint32_t kNocChunk = 16384;         // NOC_MAX_BURST_SIZE; 5 of these fill a page
    constexpr uint32_t kNumSweeps = 200;

    const std::vector<uint32_t> coords = worker_coords();
    const uint32_t num_cores = static_cast<uint32_t>(coords.size());
    const uint32_t sweep_bytes = num_cores * kBytesPerCore;
    TT_FATAL(sweep_bytes % kPageBytes == 0, "sweep {} B must be a whole number of {} B pages", sweep_bytes, kPageBytes);

    const auto& hal = MetalContext::instance().hal();
    const auto& soc_desc = MetalContext::instance().get_cluster().get_soc_desc(mesh_device_->build_id());
    const uint32_t dram_base = static_cast<uint32_t>(hal.get_dev_addr(HalDramMemAddrType::PROFILER));
    const uint32_t dram_region = hal.get_dev_size(HalDramMemAddrType::PROFILER);
    TT_FATAL(
        2u * sweep_bytes <= dram_region,
        "ping+pong ({} B) exceed the {} B profiler region per bank",
        2u * sweep_bytes,
        dram_region);

    // A on bank 0, B on bank 1 -- different banks, so B reaches A's DRAM over the NoC, not by DMA.
    const CoreCoord a_logical = mesh_device_->impl().pick_unused_dram_logical_core(mesh_device_->get_devices()[0], 0);
    const CoreCoord b_logical = mesh_device_->impl().pick_unused_dram_logical_core(mesh_device_->get_devices()[0], 1);
    const CoreCoord a_virtual = mesh_device_->virtual_core_from_logical_core(a_logical, CoreType::DRAM);
    const CoreCoord b_virtual = mesh_device_->virtual_core_from_logical_core(b_logical, CoreType::DRAM);
    const CoreCoord bank0_noc1_ep = soc_desc.get_preferred_worker_core_for_dram_view(0, /*noc=*/1);

    const CoreCoord b_translated = soc_desc.dram_bank_endpoint_coords.at(b_logical.x).at(b_logical.y);
    const tt::umd::CoreCoord b_phys = soc_desc.translate_coord_to(
        tt::umd::CoreCoord(b_translated.x, b_translated.y, CoreType::DRAM, CoordSystem::TRANSLATED), CoordSystem::NOC0);

    // A's L1: two 40 KB batch buffers, then results and the drain flag B writes.
    const uint32_t a_buf = drisc_l1_base_;
    const uint32_t a_res = a_buf + 2 * kCoresPerBatch * kBytesPerCore;
    const uint32_t a_drain_flag = a_res + 1024;
    // B's L1: one 80 KB page buffer, socket config, results, then the fill flag A writes.
    const uint32_t b_buf = drisc_l1_base_;
    const uint32_t b_cfg = b_buf + kPageBytes;
    const uint32_t b_res = b_cfg + 1024;
    const uint32_t b_fill_flag = b_res + 1024;
    TT_FATAL(
        (b_fill_flag + 64 - drisc_l1_base_) <= drisc_l1_unreserved_size_,
        "B's layout overflows {} B of DRISC L1",
        drisc_l1_unreserved_size_);

    auto set_niu_mode = [&](const CoreCoord& core, uint32_t stream) {
        Program p = CreateProgram();
        CreateKernel(
            p,
            "tests/tt_metal/tt_metal/test_kernels/misc/drisc_niu_mode.cpp",
            core,
            DramConfig{.noc = NOC::NOC_0, .compile_args = {stream}});
        run_workload(std::move(p));
    };

    // B's NIU must be in stream mode before the socket is constructed: ExternalConfigBuffer::address is
    // uint32_t and cannot carry the 0x2000000000 tag, and only stream mode makes a plain address land in L1.
    set_niu_mode(b_logical, 1);
    prime_worker_l1(kBytesPerCore);

    log_info(
        LogTest,
        "pipeline: A={},{} (bank 0) -> DRAM 0x{:x} ping/pong {} B each -> B={},{} (bank 1) via NOC1 ep {},{} -> host",
        a_logical.x,
        a_logical.y,
        dram_base,
        sweep_bytes,
        b_logical.x,
        b_logical.y,
        bank0_noc1_ep.x,
        bank0_noc1_ep.y);

    const uint32_t pages_per_sweep = sweep_bytes / kPageBytes;
    const uint32_t clk_hz = MetalContext::instance().get_cluster().get_device_aiclk(mesh_device_->id()) * 1000000u;

    auto run_pipeline = [&](uint32_t pages_per_read) {
        distributed::D2HSocket socket(
            devices_[0],
            distributed::MeshCoreCoord(distributed::MeshCoordinate(0, 0), CoreCoord(b_phys.x, b_phys.y)),
            32 * kPageBytes,
            distributed::D2HSocket::ExternalConfigBuffer{.address = b_cfg, .sender_uses_physical_noc_addr = true});
        socket.set_page_size(kPageBytes);

        Program program = CreateProgram();
        auto ka = CreateKernel(
            program,
            "tests/tt_metal/tt_metal/test_kernels/misc/drisc_pipeline_a.cpp",
            a_logical,
            DramConfig{
                .noc = NOC::NOC_0,
                .compile_args = {kCoresPerBatch, kBytesPerCore, a_buf, a_res, a_drain_flag, sweep_bytes}});
        std::vector<uint32_t> a_rtas = {
            num_cores,
            kNumSweeps,
            tensix_l1_base_,
            dram_base,
            0u,
            static_cast<uint32_t>((b_virtual.x & 0xFFFF) | (b_virtual.y << 16)),
            b_fill_flag};
        a_rtas.insert(a_rtas.end(), coords.begin(), coords.end());
        SetRuntimeArgs(program, ka, a_logical, a_rtas);

        auto kb = CreateKernel(
            program,
            "tests/tt_metal/tt_metal/test_kernels/misc/socket/drisc_pipeline_b.cpp",
            b_logical,
            DramConfig{
                .noc = NOC::NOC_0,
                .compile_args = {
                    socket.get_config_buffer_address(),
                    b_buf,
                    kPageBytes,
                    kNocChunk,
                    b_res,
                    b_fill_flag,
                    sweep_bytes}});
        SetRuntimeArgs(
            program,
            kb,
            b_logical,
            {kNumSweeps,
             static_cast<uint32_t>((bank0_noc1_ep.x & 0xFFFF) | (bank0_noc1_ep.y << 16)),
             dram_base,
             static_cast<uint32_t>((a_virtual.x & 0xFFFF) | (a_virtual.y << 16)),
             a_drain_flag});

        distributed::MeshWorkload workload;
        workload.add_program(device_range_, std::move(program));
        distributed::EnqueueMeshWorkload(mesh_device_->mesh_command_queue(), workload, false);

        const uint32_t total_pages = kNumSweeps * pages_per_sweep;
        TT_FATAL(total_pages % pages_per_read == 0, "{} pages must divide by {}", total_pages, pages_per_read);
        std::vector<uint32_t> sink(static_cast<size_t>(kPageBytes) * pages_per_read / sizeof(uint32_t));
        const auto t0 = std::chrono::steady_clock::now();
        for (uint32_t p = 0; p < total_pages / pages_per_read; p++) {
            socket.read(sink.data(), pages_per_read);
        }
        socket.barrier(20000);
        const auto t1 = std::chrono::steady_clock::now();
        distributed::Finish(mesh_device_->mesh_command_queue());

        const double host_s = std::chrono::duration<double>(t1 - t0).count();
        const uint64_t total_bytes = static_cast<uint64_t>(kNumSweeps) * sweep_bytes;

        std::vector<uint32_t> ao(6);
        MetalContext::instance().get_cluster().read_core(
            ao.data(),
            ao.size() * sizeof(uint32_t),
            tt_cxy_pair(mesh_device_->build_id(), a_virtual),
            drisc_l1_noc_addr_ + (a_res - drisc_l1_base_));
        std::vector<uint32_t> bo(9);
        MetalContext::instance().get_cluster().read_core(
            bo.data(), bo.size() * sizeof(uint32_t), tt_cxy_pair(mesh_device_->build_id(), b_virtual), b_res);

        const uint64_t a_cyc = (static_cast<uint64_t>(ao[1]) << 32) | ao[0];
        const uint64_t a_blk = (static_cast<uint64_t>(ao[3]) << 32) | ao[2];
        const uint64_t b_cyc = (static_cast<uint64_t>(bo[1]) << 32) | bo[0];
        const uint64_t b_blk = (static_cast<uint64_t>(bo[3]) << 32) | bo[2];
        const uint64_t b_dram = (static_cast<uint64_t>(bo[6]) << 32) | bo[5];
        const uint64_t b_push = (static_cast<uint64_t>(bo[8]) << 32) | bo[7];

        EXPECT_EQ(ao[4], kNumSweeps) << "A did not complete all sweeps";
        EXPECT_EQ(bo[4], kNumSweeps) << "B did not drain all sweeps";

        log_info(
            LogTest,
            "  {} pg/read | e2e {:>6.2f} GB/s | A {:>6.2f} ({:>4.1f}% blocked) | B {:>6.2f} ({:>4.1f}% blocked, "
            "dram-read {:>4.1f}% push {:>4.1f}%)",
            pages_per_read,
            static_cast<double>(total_bytes) / host_s / 1e9,
            compute_bw_gbs(total_bytes, a_cyc, clk_hz),
            100.0 * static_cast<double>(a_blk) / static_cast<double>(a_cyc),
            compute_bw_gbs(total_bytes, b_cyc, clk_hz),
            100.0 * static_cast<double>(b_blk) / static_cast<double>(b_cyc),
            100.0 * static_cast<double>(b_dram) / static_cast<double>(b_cyc),
            100.0 * static_cast<double>(b_push) / static_cast<double>(b_cyc));
    };

    for (uint32_t rep = 0; rep < 2; rep++) {
        for (uint32_t ppr : {1u, 2u, 4u, 8u}) {
            run_pipeline(ppr);
        }
    }

    set_niu_mode(b_logical, 0);
}

// Ingest + DMA on one DRISC. Reads land in L1 via the NIU and leave it via the DMA engine, so every
// byte crosses L1 twice -- unavoidable, since a DRISC in stream mode cannot land NoC traffic in GDDR.
// Whether that double-crossing costs anything is the question the DRAM-buffer design hinges on.
//
// Three modes over identical batching and buffer layout, so any difference is the interaction and not
// a change in access pattern:
//   read only   the NoC leg alone, at this batch's depth
//   dma only    the DMA leg alone, moving whatever is already in L1
//   both        the real drainer leg
TEST_F(DramKernelDRISCScatterFixture, DRISCCombinedReadAndDma) {
    constexpr uint32_t kBytesPerCore = 1280 * 8;  // 10240, a whole core
    constexpr uint32_t kGddrRingBytes = 16u << 20;
    constexpr uint32_t kRepeats = 3;

    const std::vector<uint32_t> coords = worker_coords();
    const uint32_t num_cores = static_cast<uint32_t>(coords.size());
    const uint32_t buf_base = drisc_l1_base_;

    const auto& soc_desc = MetalContext::instance().get_cluster().get_soc_desc(mesh_device_->build_id());
    const uint32_t num_banks = soc_desc.get_dram_compute_grid_size().x;
    const uint32_t clk_hz = MetalContext::instance().get_cluster().get_device_aiclk(mesh_device_->id()) * 1000000u;

    // Interleaved with one page per bank gives every bank the same bank-relative base, so the DRISC's
    // DMA writes into its own channel at that address.
    auto dram_buffer = CreateBuffer(InterleavedBufferConfig{
        .device = mesh_device_,
        .size = static_cast<uint64_t>(num_banks) * kGddrRingBytes,
        .page_size = kGddrRingBytes,
        .buffer_type = BufferType::DRAM,
    });
    const uint32_t gddr_addr = dram_buffer->address();

    prime_worker_l1(kBytesPerCore);
    const std::vector<CoreCoord> drisc = free_drisc_cores(1);
    const CoreCoord drisc_virtual = mesh_device_->virtual_core_from_logical_core(drisc[0], CoreType::DRAM);
    log_info(
        LogTest,
        "combined read+DMA: {} cores x {} B, GDDR ring {} MB at 0x{:x} ({} banks), DRISC L1 {} B",
        num_cores,
        kBytesPerCore,
        kGddrRingBytes >> 20,
        gddr_addr,
        num_banks,
        drisc_l1_unreserved_size_);

    auto run = [&](uint32_t do_read, uint32_t do_dma, uint32_t cores_per_batch, uint32_t num_buffers) {
        const uint32_t batch_bytes = cores_per_batch * kBytesPerCore;
        const uint32_t res_l1 = buf_base + num_buffers * batch_bytes;
        TT_FATAL(
            (res_l1 - drisc_l1_base_) + 64 <= drisc_l1_unreserved_size_,
            "{} buffers x {} B do not fit {} B of DRISC L1",
            num_buffers,
            batch_bytes,
            drisc_l1_unreserved_size_);
        constexpr uint64_t kTargetBytes = 512ull << 20;
        const uint32_t iters = std::clamp<uint32_t>(
            static_cast<uint32_t>(kTargetBytes / (static_cast<uint64_t>(num_cores) * kBytesPerCore)), 8u, 100000u);

        Program program = CreateProgram();
        auto kid = CreateKernel(
            program,
            "tests/tt_metal/tt_metal/test_kernels/misc/drisc_read_dma_combined.cpp",
            drisc[0],
            DramConfig{
                .noc = NOC::NOC_0,
                .compile_args = {
                    cores_per_batch, kBytesPerCore, buf_base, res_l1, do_read, do_dma, kGddrRingBytes, num_buffers}});
        std::vector<uint32_t> rtas = {num_cores, iters, tensix_l1_base_, gddr_addr, 0u};
        rtas.insert(rtas.end(), coords.begin(), coords.end());
        SetRuntimeArgs(program, kid, drisc[0], rtas);
        run_workload(std::move(program));

        // The kernel restores NOC2AXI on exit, so reaching DRISC L1 from the host needs the tagged NoC
        // address -- a plain address in this range is forwarded to GDDR instead.
        // The kernel restores NOC2AXI on exit, so reaching DRISC L1 from the host needs the tagged NoC
        // address -- a plain address in this range is forwarded to GDDR instead.
        std::vector<uint32_t> out(9);
        MetalContext::instance().get_cluster().read_core(
            out.data(),
            out.size() * sizeof(uint32_t),
            tt_cxy_pair(mesh_device_->build_id(), drisc_virtual),
            drisc_l1_noc_addr_ + (res_l1 - drisc_l1_base_));
        const uint64_t cycles = (static_cast<uint64_t>(out[1]) << 32) | out[0];
        const uint64_t visits = out[3];
        const uint64_t total_bytes = visits * kBytesPerCore;
        if (do_read) {
            EXPECT_NE(out[2], 0u) << "checksum zero -- NoC reads never landed";
        }
        const uint64_t batches = visits / out[8];
        const double ns = 1e9 / clk_hz;
        log_info(
            LogTest,
            "      per batch of {} cores ({} B): total {:>7.1f} ns = dma-wait {:>6.1f} + read-issue {:>6.1f} "
            "+ read-wait {:>6.1f} + dma-issue {:>6.1f} + loop {:>5.1f}",
            out[8],
            out[8] * kBytesPerCore,
            static_cast<double>(cycles) / batches * ns,
            static_cast<double>(out[4]) / batches * ns,
            static_cast<double>(out[5]) / batches * ns,
            static_cast<double>(out[6]) / batches * ns,
            static_cast<double>(out[7]) / batches * ns,
            static_cast<double>(cycles - out[4] - out[5] - out[6] - out[7]) / batches * ns);
        return std::tuple<double, double, double>{
            compute_bw_gbs(total_bytes, cycles, clk_hz),
            100.0 * static_cast<double>(out[6]) / static_cast<double>(cycles),
            100.0 * static_cast<double>(out[4] + out[7]) / static_cast<double>(cycles)};
    };

    struct Cfg {
        const char* label;
        uint32_t do_read;
        uint32_t do_dma;
        uint32_t cores;
        uint32_t bufs;
    };
    for (auto c : std::vector<Cfg>{
             {"read only   2buf x 4", 1u, 0u, 4u, 2u},
             {"read only   1buf x 8", 1u, 0u, 8u, 1u},
             {"dma only    2buf x 4", 0u, 1u, 4u, 2u},
             {"read+DMA    2buf x 4", 1u, 1u, 4u, 2u},
             {"read+DMA    1buf x 8", 1u, 1u, 8u, 1u},
             {"read+DMA    1buf x 4", 1u, 1u, 4u, 1u},
             {"read+DMA    2buf x 2", 1u, 1u, 2u, 2u},
             {"read+DMA    2buf x 3", 1u, 1u, 3u, 2u}}) {
        std::vector<double> bw;
        std::string detail;
        double rd_pct = 0.0;
        double dm_pct = 0.0;
        for (uint32_t rep = 0; rep < kRepeats; rep++) {
            auto [b, r, d] = run(c.do_read, c.do_dma, c.cores, c.bufs);
            bw.push_back(b);
            rd_pct = r;
            dm_pct = d;
            detail += fmt::format(" {:>6.2f}", b);
        }
        std::sort(bw.begin(), bw.end());
        log_info(
            LogTest,
            "{} | median {:>6.2f} GB/s | spread {:>4.2f}x | read phase {:>5.1f}% dma phase {:>5.1f}% |{}",
            c.label,
            bw[bw.size() / 2],
            bw.back() / bw.front(),
            rd_pct,
            dm_pct,
            detail);
    }
}

// Tuning the host side of D2H egress. The baseline test showed the DRISC idle ~80% of the time
// waiting in socket_reserve_pages, so 16.55 GB/s was the host consumption rate, not the device's
// (which runs at the ~86 GB/s NoC port limit). Every knob here is host-side:
//
//   page size        fewer, larger transfers -- up to ~84 KB, all that fits beside the config
//                    buffer in 86 KB of DRISC L1
//   pages per read   one memcpy and one bytes_acked PCIe write per call instead of per page
//   ack batching     notify_sender=false on most calls, so the device sees fewer ack round-trips
//   discard mode     discard_pending_pages() acks without touching the data region, which
//                    isolates how much of the cost is the memcpy versus the socket protocol
//
// Every config is repeated: a first pass showed the same parameters landing 74% apart between
// sections, so single samples here are not trustworthy.
//
// Teardown is defensive because an earlier version of this test wedged the card: the discard loop is
// bounded by a deadline, every barrier has a timeout, and the NIU restore runs even if the sweep
// throws. Leaving the NIU in stream mode is survivable (DRISC firmware forces NOC2AXI at boot) but a
// hung socket is not.
TEST_F(DramKernelDRISCScatterFixture, DRISCD2HSocketEgressTuned) {
    const auto& soc_desc = MetalContext::instance().get_cluster().get_soc_desc(mesh_device_->build_id());
    const CoreCoord drisc_logical = mesh_device_->impl().pick_unused_dram_logical_core(mesh_device_->get_devices()[0], 0);
    const CoreCoord drisc_translated = soc_desc.dram_bank_endpoint_coords.at(drisc_logical.x).at(drisc_logical.y);
    const tt::umd::CoreCoord drisc_phys = soc_desc.translate_coord_to(
        tt::umd::CoreCoord(drisc_translated.x, drisc_translated.y, CoreType::DRAM, CoordSystem::TRANSLATED),
        CoordSystem::NOC0);
    const CoreCoord drisc_virtual = mesh_device_->virtual_core_from_logical_core(drisc_logical, CoreType::DRAM);
    const uint32_t clk_hz = MetalContext::instance().get_cluster().get_device_aiclk(mesh_device_->id()) * 1000000u;
    const uint32_t cfg_bytes = distributed::D2HSocket::required_config_buffer_size();
    constexpr uint32_t kRepeats = 3;

    auto set_niu_mode = [&](uint32_t stream) {
        Program p = CreateProgram();
        CreateKernel(
            p,
            "tests/tt_metal/tt_metal/test_kernels/misc/drisc_niu_mode.cpp",
            drisc_logical,
            DramConfig{.noc = NOC::NOC_0, .compile_args = {stream}});
        run_workload(std::move(p));
    };

    set_niu_mode(1);
    log_info(
        LogTest,
        "egress tuning: DRISC L1 unreserved {} B, socket config buffer {} B, {} repeats per config",
        drisc_l1_unreserved_size_,
        cfg_bytes,
        kRepeats);

    auto run_one = [&](uint32_t page_size,
                       uint32_t pages_per_read,
                       uint32_t ack_every,
                       bool discard,
                       uint32_t notify_every = 1) {
        constexpr uint64_t kTargetBytes = 256ull << 20;
        constexpr uint32_t kFifoPages = 32;

        const uint32_t src_l1 = drisc_l1_base_;
        const uint32_t cfg_l1 = tt::align(drisc_l1_base_ + page_size, 1024u);
        const uint32_t res_l1 = tt::align(cfg_l1 + cfg_bytes, 1024u);
        TT_FATAL(
            (res_l1 - drisc_l1_base_) + 32 <= drisc_l1_unreserved_size_,
            "page {} B leaves no room for config+results in {} B of DRISC L1",
            page_size,
            drisc_l1_unreserved_size_);

        uint32_t num_pages = static_cast<uint32_t>(kTargetBytes / page_size);
        num_pages -= num_pages % pages_per_read;

        distributed::D2HSocket socket(
            devices_[0],
            distributed::MeshCoreCoord(distributed::MeshCoordinate(0, 0), CoreCoord(drisc_phys.x, drisc_phys.y)),
            kFifoPages * page_size,
            distributed::D2HSocket::ExternalConfigBuffer{.address = cfg_l1, .sender_uses_physical_noc_addr = true});
        socket.set_page_size(page_size);

        std::vector<uint32_t> payload(page_size / sizeof(uint32_t), 0xD2D2D2D2u);
        MetalContext::instance().get_cluster().write_core(
            payload.data(), page_size, tt_cxy_pair(mesh_device_->build_id(), drisc_virtual), src_l1);

        Program program = CreateProgram();
        CreateKernel(
            program,
            "tests/tt_metal/tt_metal/test_kernels/misc/socket/drisc_d2h_egress.cpp",
            drisc_logical,
            DramConfig{
                .noc = NOC::NOC_0,
                .compile_args = {
                    socket.get_config_buffer_address(), src_l1, page_size, res_l1, num_pages, notify_every}});

        distributed::MeshWorkload workload;
        workload.add_program(device_range_, std::move(program));
        distributed::EnqueueMeshWorkload(mesh_device_->mesh_command_queue(), workload, false);

        std::vector<uint32_t> sink(static_cast<size_t>(page_size) * pages_per_read / sizeof(uint32_t));
        const auto t0 = std::chrono::steady_clock::now();
        if (discard) {
            uint32_t done = 0;
            const auto deadline = t0 + std::chrono::seconds(30);
            while (done < num_pages) {
                done += socket.discard_pending_pages();
                if (std::chrono::steady_clock::now() > deadline) {
                    log_warning(LogTest, "discard timed out at {}/{} pages", done, num_pages);
                    break;
                }
            }
        } else {
            const uint32_t calls = num_pages / pages_per_read;
            for (uint32_t c = 0; c < calls; c++) {
                socket.read(sink.data(), pages_per_read, ((c + 1) % ack_every) == 0 || c + 1 == calls);
            }
        }
        // Always bounded: an untimed barrier here is what let the earlier version hang the board.
        socket.barrier(10000);
        const auto t1 = std::chrono::steady_clock::now();
        distributed::Finish(mesh_device_->mesh_command_queue());

        std::vector<uint32_t> out(10);
        MetalContext::instance().get_cluster().read_core(
            out.data(), out.size() * sizeof(uint32_t), tt_cxy_pair(mesh_device_->build_id(), drisc_virtual), res_l1);
        const uint64_t cycles = (static_cast<uint64_t>(out[1]) << 32) | out[0];
        const uint64_t wait = (static_cast<uint64_t>(out[3]) << 32) | out[2];
        const uint64_t wr = (static_cast<uint64_t>(out[6]) << 32) | out[5];
        const uint64_t nt = (static_cast<uint64_t>(out[8]) << 32) | out[7];
        const uint64_t total_bytes = static_cast<uint64_t>(num_pages) * page_size;
        const double host_s = std::chrono::duration<double>(t1 - t0).count();
        const double ns = 1e9 / clk_hz;

        // Per-page phase breakdown, in ns. `other` is whatever the three phases do not account for
        // (loop overhead, the final socket_barrier, timer probes).
        log_info(
            LogTest,
            "    breakdown/page: total {:>7.0f} ns = wait {:>7.0f} + write {:>7.0f} + notify {:>6.0f} "
            "+ other {:>6.0f} ns   (timer {} cyc x4/page)",
            static_cast<double>(cycles) / num_pages * ns,
            static_cast<double>(wait) / num_pages * ns,
            static_cast<double>(wr) / num_pages * ns,
            static_cast<double>(nt) / num_pages * ns,
            static_cast<double>(cycles - wait - wr - nt) / num_pages * ns,
            out[9]);

        return std::tuple<double, double, double>{
            compute_bw_gbs(total_bytes, cycles, clk_hz),
            static_cast<double>(total_bytes) / host_s / 1e9,
            100.0 * static_cast<double>(wait) / static_cast<double>(cycles)};
    };

    auto sweep =
        [&](uint32_t page_size, uint32_t pages_per_read, uint32_t ack_every, bool discard, uint32_t notify_every = 1) {
            std::vector<double> dev;
            std::string detail;
            for (uint32_t rep = 0; rep < kRepeats; rep++) {
                auto [d, h, w] = run_one(page_size, pages_per_read, ack_every, discard, notify_every);
                dev.push_back(d);
                detail += fmt::format(" {:>6.2f}({:>4.1f}%)", d, w);
            }
            std::sort(dev.begin(), dev.end());
            log_info(
                LogTest,
                "page {:>6} B | {:>2} pg/read | ack {:<2} | notify/{:<2} | {:<7} | median {:>6.2f} GB/s | spread "
                "{:>5.2f}x |{}",
                page_size,
                pages_per_read,
                ack_every,
                notify_every,
                discard ? "DISCARD" : "memcpy",
                dev[dev.size() / 2],
                dev.back() / dev.front(),
                detail);
        };

    try {
        log_info(LogTest, "-- tuned baseline: 80 KB x 8 pages/read, notify every page --");
        sweep(81920u, 8, 1, false, 1);
        log_info(LogTest, "-- device-side notify batching, memcpy host --");
        for (uint32_t ne : {2u, 4u, 8u, 16u}) {
            sweep(81920u, 8, 1, false, ne);
        }
        log_info(LogTest, "-- device-side notify batching, protocol ceiling (DISCARD) --");
        for (uint32_t ne : {1u, 2u, 4u, 8u, 16u}) {
            sweep(81920u, 8, 1, true, ne);
        }
    } catch (const std::exception& e) {
        ADD_FAILURE() << "egress sweep threw: " << e.what();
    }

    set_niu_mode(0);
}

// Whole-core (10 KB) reads with reads-in-flight pushed past what DRISC L1 can hold, by letting
// outstanding reads share landing slots. NOC_MAX_TRANSACTION_ID_COUNT is 255 per trid, so the NIU can
// track far more than the 8 distinct 10 KB buffers that fit in 86 KB -- this separates the hardware
// depth limit from the buffer limit.
//
// Slot-sharing corrupts the landed data and is therefore NOT a usable drainer configuration. It is a
// transport measurement only: it says whether a bigger buffer (or smaller reads) would buy anything.
TEST_F(DramKernelDRISCScatterFixture, DRISCWholeCoreDepthBeyondBuffer) {
    constexpr uint32_t kK = 1280;  // 10 KB, a whole core
    const uint32_t all = static_cast<uint32_t>(worker_coords().size());
    const uint32_t max_slots = drisc_l1_unreserved_size_ / (kK * kMarkerBytes);

    prime_worker_l1(kK * kMarkerBytes);
    const std::vector<CoreCoord> drisc = free_drisc_cores(1);
    log_info(
        LogTest,
        "whole-core 10 KB reads, {} cores, {} distinct slots fit in {} KB of DRISC L1",
        all,
        max_slots,
        drisc_l1_unreserved_size_ / 1024);

    for (uint32_t b : {1u, 4u, 8u, 16u, 32u, 64u, all}) {
        const uint32_t slots = std::min(b, max_slots);
        const BenchResult r = run_bench(drisc, kK, b, /*poll_examine=*/0, slots);
        const uint32_t clk_hz = MetalContext::instance().get_cluster().get_device_aiclk(mesh_device_->id()) * 1000000u;
        const double cyc_per_core = static_cast<double>(r.max_cycles) / static_cast<double>(r.total_visits);
        log_info(
            LogTest,
            "B={:<4} ({:>2} slots{}) {:>7.1f} cyc/core {:>7.2f} ns/core | sweep of {} cores {:>7.2f} us | {:>6.2f} "
            "GB/s",
            b,
            slots,
            slots < b ? ", SHARED" : "        ",
            cyc_per_core,
            cyc_per_core * 1e9 / clk_hz,
            all,
            cyc_per_core * all * 1e6 / clk_hz,
            compute_bw_gbs(r.total_bytes, r.max_cycles, clk_hz));
    }
}

// The 8 us kernel-train question, measured on the DRISC: a full adaptive sweep (poll every core,
// run the adaptive threshold decision, whole-core bulk read for each core that trips it) against
// an 8 us budget.
//
// The host sets each core's control-vector tails so a chosen number of cores trip ADAPT_THRESH.
// That is the knob that matters: at steady state only a small fraction of cores have accumulated
// 4 rings' worth between sweeps, and the sweep cost is dominated by polling the rest.
TEST_F(DramKernelDRISCScatterFixture, DRISCAdaptiveDrainSteadyState) {
    constexpr uint32_t kRingCapWords = 512;  // PROFILER_L1_VECTOR_SIZE
    constexpr uint32_t kNumRisc = 5;
    constexpr uint32_t kCvWords = 64;                              // PROFILER_L1_CONTROL_VECTOR_SIZE
    constexpr uint32_t kPollBytes = kCvWords * sizeof(uint32_t);   // 256
    constexpr uint32_t kBulkBytes = kNumRisc * kRingCapWords * 4;  // 10240, whole core
    constexpr uint32_t kThresholdWords = 4 * kRingCapWords;        // ADAPT_THRESH
    constexpr uint32_t kBulkDepth = 4;
    constexpr uint32_t kIters = 2000;
    constexpr double kTrainBudgetUs = 8.0;
    constexpr const char* kDrainKernel = "tests/tt_metal/tt_metal/test_kernels/misc/drisc_adaptive_drain.cpp";

    const std::vector<uint32_t> coords = worker_coords();
    const uint32_t num_cores = static_cast<uint32_t>(coords.size());
    const uint32_t poll_ring = drisc_l1_base_;
    const uint32_t bulk_ring = poll_ring + num_cores * kPollBytes;
    const uint32_t results_addr = bulk_ring + kBulkDepth * kBulkBytes;
    TT_FATAL(
        (results_addr - drisc_l1_base_) + 7 * sizeof(uint32_t) <= drisc_l1_unreserved_size_,
        "poll ring {} B + bulk ring {} B exceeds DRISC L1 unreserved {} B",
        num_cores * kPollBytes,
        kBulkDepth * kBulkBytes,
        drisc_l1_unreserved_size_);

    // Ring payload is identical on every core and never changes; only the control vector is rewritten
    // per configuration, so prime the 10 KB once.
    prime_worker_l1(kPollBytes + kBulkBytes);

    const std::vector<CoreCoord> drisc = free_drisc_cores(1);
    const uint32_t aiclk_hz = MetalContext::instance().get_cluster().get_device_aiclk(mesh_device_->id()) * 1000000u;
    log_info(
        LogTest,
        "adaptive drain: {} cores, poll {} B, bulk {} B, ADAPT_THRESH {} words, bulk depth {}, budget {} us",
        num_cores,
        kPollBytes,
        kBulkBytes,
        kThresholdWords,
        kBulkDepth,
        kTrainBudgetUs);

    for (uint32_t hot : {0u, 1u, num_cores / 10, num_cores / 2, num_cores}) {
        // Tails high enough that 5 lanes sum past the threshold, or low enough that they never do.
        std::vector<uint32_t> cv(kCvWords, 0);
        for (uint32_t c = 0; c < num_cores; c++) {
            const uint32_t tail = (c < hot) ? (kThresholdWords / kNumRisc + 1) : 1u;
            for (uint32_t r = 0; r < kNumRisc; r++) {
                cv[5 + r] = tail;
            }
            cv[0] = 0xA5A50000u + c;  // nonzero, so the kernel's checksum guard still means something
            const CoreCoord v = mesh_device_->virtual_core_from_logical_core(
                {c % mesh_device_->compute_with_storage_grid_size().x,
                 c / mesh_device_->compute_with_storage_grid_size().x},
                CoreType::WORKER);
            MetalContext::instance().get_cluster().write_core(
                cv.data(), cv.size() * sizeof(uint32_t), tt_cxy_pair(mesh_device_->build_id(), v), tensix_l1_base_);
        }

        Program program = CreateProgram();
        auto kid = CreateKernel(
            program,
            kDrainKernel,
            drisc[0],
            DramConfig{
                .noc = NOC::NOC_0,
                .compile_args = {
                    kPollBytes, kBulkBytes, kThresholdWords, poll_ring, bulk_ring, kBulkDepth, results_addr}});
        std::vector<uint32_t> rtas = {num_cores, kIters, tensix_l1_base_, tensix_l1_base_ + kPollBytes};
        rtas.insert(rtas.end(), coords.begin(), coords.end());
        SetRuntimeArgs(program, kid, drisc[0], rtas);
        run_workload(std::move(program));

        const CoreCoord v = mesh_device_->virtual_core_from_logical_core(drisc[0], CoreType::DRAM);
        std::vector<uint32_t> out(7);
        MetalContext::instance().get_cluster().read_core(
            out.data(),
            out.size() * sizeof(uint32_t),
            tt_cxy_pair(mesh_device_->build_id(), v),
            drisc_l1_noc_addr_ + (results_addr - drisc_l1_base_));
        EXPECT_NE(out[2], 0u) << "checksum zero -- poll reads never landed";

        const uint64_t cycles = (static_cast<uint64_t>(out[1]) << 32) | out[0];
        const double cyc_per_sweep = static_cast<double>(cycles) / kIters;
        const double us_per_sweep = cyc_per_sweep * 1e6 / aiclk_hz;
        const double bulks_per_sweep = static_cast<double>(out[5]) / kIters;
        const double drained_bytes = bulks_per_sweep * kBulkBytes;
        log_info(
            LogTest,
            "{:>4}/{} cores hot | {:>9.0f} cyc {:>7.2f} us/sweep | {:>6.1f} bulk reads/sweep, {:>7.1f} KB drained "
            "| {:>5.2f}x the {} us budget | drained {:.1f} GB/s",
            hot,
            num_cores,
            cyc_per_sweep,
            us_per_sweep,
            bulks_per_sweep,
            drained_bytes / 1024.0,
            us_per_sweep / kTrainBudgetUs,
            kTrainBudgetUs,
            drained_bytes / (us_per_sweep * 1e-6) / 1e9);
    }
}

// Sweep burst size x reads-in-flight on a single DRISC over the full worker grid.
// The first pass showed per-visit cost depends on B only, not K (issue-bound, not bandwidth-bound),
// so this sweep pushes both axes far enough to find where that breaks: K until wire time overtakes
// the ~29 ns/read issue cost, B until the ~231 ns round-trip is fully hidden.
// Bursts are not the limit here -- NOC_MAX_BURST_SIZE is 16 KB (256 words x 64 B) -- the B*K landing
// ring against DRISC L1 UNRESERVED is, so over-budget combinations are skipped and reported.
TEST_F(DramKernelDRISCScatterFixture, DRISCScatterReadBurstAndDepthSweep) {
    // B == num_cores means "issue a read to every core, then one single wait" -- the maximum-outstanding
    // point, where the round-trip is hidden as far as it can be and only issue cost remains.
    const uint32_t all = static_cast<uint32_t>(worker_coords().size());

    // Reference points from the real profiler geometry (profiler_common.h / dev_msgs.h):
    //   K=256  = one full (core, RISC) ring          (PROFILER_L1_VECTOR_SIZE 512 words, 2 words/marker)
    //   K=1280 = a whole core, all 5 RISC rings      (~10 KB, still under NOC_MAX_BURST_SIZE of 16 KB)
    // Anything above K=256 is not reachable per-lane; it is included to locate the wire-time wall.
    // B*K*kMarkerBytes is the landing ring, so large K forces small B -- over-budget combos are skipped.
    const std::vector<std::pair<uint32_t, uint32_t>> configs = {
        {1, 1},   {1, 8},   {1, 32},  {1, all},                         // pure per-visit cost vs depth
        {32, 1},  {32, 8},  {32, 16}, {32, 32},  {32, 64},  {32, all},  // the K=32 row, full depth sweep
        {64, 8},  {64, 16}, {64, 64}, {64, all}, {128, 8},  {128, 16}, {128, 32},
        {256, 1}, {256, 4}, {256, 8}, {256, 16},                                   // one full RISC ring
        {512, 1}, {512, 4}, {512, 8}, {1280, 1}, {1280, 2}, {1280, 4}, {1280, 8},  // one whole core per read
    };

    uint32_t max_k = 0;
    for (const auto& [k, b] : configs) {
        max_k = std::max(max_k, k);
    }
    prime_worker_l1(max_k * kMarkerBytes);

    const std::vector<CoreCoord> drisc = free_drisc_cores(1);
    log_info(
        LogTest,
        "1 DRISC (bank 0 free subchannel), {} worker cores polled, DRISC L1 unreserved {} KB, B={} means issue-all",
        all,
        drisc_l1_unreserved_size_ / 1024,
        all);

    for (const auto& [k, b] : configs) {
        const uint32_t ring_bytes = b * k * kMarkerBytes;
        if (ring_bytes + 4 * sizeof(uint32_t) > drisc_l1_unreserved_size_) {
            log_info(
                LogTest,
                "K={:<5} B={:<4} SKIPPED: {} KB ring exceeds {} KB DRISC L1",
                k,
                b,
                ring_bytes / 1024,
                drisc_l1_unreserved_size_ / 1024);
            continue;
        }
        const BenchResult r = run_bench(drisc, k, b);
        log_row(fmt::format("K={:<5}({:>5}B)  B={:<4}", k, k * kMarkerBytes, b), r, k);
    }
}

// Poll-floor sweep: hold the burst config and shrink the polled grid. At K=1 the per-visit cost
// dominates, which is the per-core poll floor.
TEST_F(DramKernelDRISCScatterFixture, DRISCScatterReadPollFloor) {
    constexpr uint32_t kReadsInFlight = 8;
    prime_worker_l1(32 * kMarkerBytes);  // cover the largest K exercised below
    const std::vector<CoreCoord> drisc = free_drisc_cores(1);

    for (uint32_t k : {1u, 32u}) {
        const BenchResult r = run_bench(drisc, k, kReadsInFlight);
        log_row(fmt::format("full grid  K={:<3} B={}", k, kReadsInFlight), r, k);
    }
}

// Multi-DRISC scaling: each DRISC is a separate core on a separate bank with its own NIU pair,
// so this is the test of whether aggregate ingest scales with DRISC count or hits a cliff.
TEST_F(DramKernelDRISCScatterFixture, DRISCScatterReadMultiCoreScaling) {
    constexpr uint32_t kMarkersPerRead = 32;
    constexpr uint32_t kReadsInFlight = 8;

    const auto& soc_desc = MetalContext::instance().get_cluster().get_soc_desc(mesh_device_->build_id());
    const uint32_t num_banks = soc_desc.get_num_dram_views();
    prime_worker_l1(kMarkersPerRead * kMarkerBytes);

    for (uint32_t n : {1u, 2u, 4u}) {
        if (n > num_banks) {
            log_info(LogTest, "skipping {} DRISCs: only {} banks", n, num_banks);
            continue;
        }
        const BenchResult r = run_bench(free_drisc_cores(n), kMarkersPerRead, kReadsInFlight);
        log_row(fmt::format("{} DRISC(s)  grid partitioned", n), r, kMarkersPerRead);
    }
}

// The first end-to-end test against REAL producers. Every DRISC measurement before this one drained
// synthetic tails the host had written into the control vector; here Tensix RISCs emit zones through
// the ordinary DeviceZoneScopedN path and a DRISC has to keep them from blocking.
//
// All five SPSC lanes of every producing core are live: BRISC (lane 0), NCRISC (lane 1) and a compute
// kernel covering TRISC0-2 (lanes 2-4). That is what exercises the drainer's per-lane indexing and its
// five-head write-back, rather than just lane 0.
//
// The test is self-falsifying: each lane is asked to push far more than its 512-word ring can hold, and
// kernel_profiler.hpp's producer BLOCKS on a full ring by design ("a profiled run REQUIRES the consumer
// to be draining"). If the drainer's flow control is wrong -- wrong tail offset, no head write-back,
// desynced mirror -- the workers never finish and the program hangs rather than failing a comparison.
//
// It runs twice on purpose. Tails are monotonic for the whole FW session, so by the second round the
// heads are far from zero, and a drainer that assumed a zero-based stream would compute a garbage run.
TEST_F(DramKernelDRISCScatterFixture, DRISCServicesRealProfiledWorkers) {
    constexpr uint32_t kWorkPerZone = 4;
    constexpr uint32_t kQuietStop = 4096;
    constexpr uint32_t kMaxSweeps = 2000000;
    constexpr uint32_t kNumRisc = 5;

    const auto& hal = MetalContext::instance().hal();
    const uint32_t cv_src = hal.get_dev_addr(HalProgrammableCoreType::TENSIX, HalL1MemAddrType::PROFILER);
    const uint32_t poll_bytes = kernel_profiler::PROFILER_L1_CONTROL_VECTOR_SIZE * sizeof(uint32_t);
    const uint32_t ring_words = kernel_profiler::PROFILER_L1_VECTOR_SIZE;
    const uint32_t data_bytes = kNumRisc * ring_words * sizeof(uint32_t);
    const uint32_t ring0_src = cv_src + poll_bytes;
    const uint32_t clk_hz = MetalContext::instance().get_cluster().get_device_aiclk(mesh_device_->id()) * 1000000u;

    const CoreCoord drisc_logical = free_drisc_cores(1)[0];
    const CoreCoord drisc_virtual = mesh_device_->virtual_core_from_logical_core(drisc_logical, CoreType::DRAM);

    auto set_niu_mode = [&](uint32_t stream) {
        Program p = CreateProgram();
        CreateKernel(
            p,
            "tests/tt_metal/tt_metal/test_kernels/misc/drisc_niu_mode.cpp",
            drisc_logical,
            DramConfig{.noc = NOC::NOC_0, .compile_args = {stream}});
        run_workload(std::move(p));
    };
    set_niu_mode(1);

    // One round: `grid` worth of producing cores, `zones` zones per lane, drained by one DRISC.
    auto run_round = [&](const char* label, CoreCoord grid, uint32_t zones) {
        const CoreRange producer_range({0, 0}, {grid.x - 1, grid.y - 1});
        std::vector<CoreCoord> producer_virtual;
        std::vector<uint32_t> coords;
        for (uint32_t y = 0; y < grid.y; y++) {
            for (uint32_t x = 0; x < grid.x; x++) {
                const CoreCoord v = mesh_device_->virtual_core_from_logical_core({x, y}, CoreType::WORKER);
                producer_virtual.push_back(v);
                coords.push_back((v.x & 0xFFFFu) | ((v.y & 0xFFFFu) << 16));
            }
        }
        const uint32_t num_cores = static_cast<uint32_t>(coords.size());

        // Heads before the round, so the round's own contribution can be isolated from a stream that
        // is already far along (tails are monotonic for the whole FW session).
        uint64_t tail_sum_before = 0;
        for (uint32_t c = 0; c < num_cores; c++) {
            std::vector<uint32_t> cv(kernel_profiler::PROFILER_L1_CONTROL_VECTOR_SIZE, 0);
            MetalContext::instance().get_cluster().read_core(
                cv.data(), poll_bytes, tt_cxy_pair(mesh_device_->build_id(), producer_virtual[c]), cv_src);
            for (uint32_t r = 0; r < kNumRisc; r++) {
                tail_sum_before += cv[kernel_profiler::SPSC_RING_TAIL_0 + r];
            }
        }

        const uint32_t poll_ring = drisc_l1_base_;
        const uint32_t data_buf = poll_ring + num_cores * poll_bytes;
        const uint32_t head_scratch = data_buf + data_bytes;
        const uint32_t results_addr = head_scratch + 16 * 32;
        ASSERT_LT(results_addr + 64 - drisc_l1_base_, drisc_l1_unreserved_size_);

        Program program = CreateProgram();
        auto brisc_id = CreateKernel(
            program,
            "tests/tt_metal/tt_metal/test_kernels/misc/profiler_zone_producer.cpp",
            producer_range,
            DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});
        SetRuntimeArgs(program, brisc_id, producer_range, {zones, kWorkPerZone});

        auto ncrisc_id = CreateKernel(
            program,
            "tests/tt_metal/tt_metal/test_kernels/misc/profiler_zone_producer.cpp",
            producer_range,
            DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});
        SetRuntimeArgs(program, ncrisc_id, producer_range, {zones, kWorkPerZone});

        CreateKernel(
            program,
            "tests/tt_metal/tt_metal/test_kernels/misc/profiler_zone_producer_compute.cpp",
            producer_range,
            ComputeConfig{.compile_args = {zones, kWorkPerZone}});

        auto drain_id = CreateKernel(
            program,
            "tests/tt_metal/tt_metal/test_kernels/misc/drisc_service_workers.cpp",
            drisc_logical,
            DramConfig{
                .noc = NOC::NOC_0,
                .compile_args = {
                    poll_bytes,
                    data_bytes,
                    ring_words,
                    poll_ring,
                    data_buf,
                    head_scratch,
                    results_addr,
                    kQuietStop,
                    kMaxSweeps}});
        std::vector<uint32_t> drain_args = {num_cores, cv_src, ring0_src};
        drain_args.insert(drain_args.end(), coords.begin(), coords.end());
        SetRuntimeArgs(program, drain_id, drisc_logical, drain_args);

        const uint64_t expected = static_cast<uint64_t>(num_cores) * kNumRisc * zones * 4;
        log_info(
            LogTest,
            "[{}] {} cores x 5 lanes x {} zones = {} words minimum; each lane overflows its {}-word ring "
            "~{}x",
            label,
            num_cores,
            zones,
            expected,
            ring_words,
            (zones * 4) / ring_words);

        const auto t0 = std::chrono::steady_clock::now();
        run_workload(std::move(program));
        const double wall_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

        std::vector<uint32_t> out(10, 0);
        MetalContext::instance().get_cluster().read_core(
            out.data(),
            out.size() * sizeof(uint32_t),
            tt_cxy_pair(mesh_device_->build_id(), drisc_virtual),
            drisc_l1_noc_addr_ + (results_addr - drisc_l1_base_));

        const uint64_t cycles = (static_cast<uint64_t>(out[1]) << 32) | out[0];
        const uint64_t total_words = (static_cast<uint64_t>(out[3]) << 32) | out[2];

        log_info(
            LogTest,
            "[{}] drained {} words ({:.2f} KB) in {} sweeps, {} core-visits with work, max run {}, "
            "overflows {}, checksum 0x{:08x}",
            label,
            total_words,
            total_words * 4.0 / 1024.0,
            out[4],
            out[5],
            out[6],
            out[7],
            out[8]);
        log_info(LogTest, "[{}] device {:.3f} ms, host wall {:.3f} s", label, cycles * 1000.0 / clk_hz, wall_s);

        if (total_words == 0) {
            GTEST_SKIP() << "no markers produced -- run with TT_METAL_DEVICE_PROFILER=1 TT_METAL_DRISC_PROFILER=1";
        }

        // The real proof: every lane fully drained, head == tail on every producing core.
        uint32_t lanes_behind = 0;
        uint32_t lanes_silent = 0;
        uint64_t tail_sum_after = 0;
        for (uint32_t c = 0; c < num_cores; c++) {
            std::vector<uint32_t> cv(kernel_profiler::PROFILER_L1_CONTROL_VECTOR_SIZE, 0);
            MetalContext::instance().get_cluster().read_core(
                cv.data(), poll_bytes, tt_cxy_pair(mesh_device_->build_id(), producer_virtual[c]), cv_src);
            for (uint32_t r = 0; r < kNumRisc; r++) {
                const uint32_t head = cv[kernel_profiler::SPSC_RING_HEAD_0 + r];
                const uint32_t tail = cv[kernel_profiler::SPSC_RING_TAIL_0 + r];
                tail_sum_after += tail;
                if (head != tail) {
                    lanes_behind++;
                }
                if (tail == 0) {
                    lanes_silent++;
                }
            }
            EXPECT_EQ(
                cv[kernel_profiler::SPSC_CORE_XY], (producer_virtual[c].y << 16) | (producer_virtual[c].x & 0xFFFF))
                << label << " core " << c << " identity stamp";
        }
        const uint64_t produced_this_round = tail_sum_after - tail_sum_before;
        log_info(
            LogTest,
            "[{}] tails advanced {} words this round; drained {}; {} lanes behind, {} lanes silent",
            label,
            produced_this_round,
            total_words,
            lanes_behind,
            lanes_silent);

        EXPECT_EQ(out[7], 0u) << label << ": ring overflow -- a producer got ahead of the drainer";
        EXPECT_LE(out[6], ring_words) << label << ": run exceeded ring capacity";
        EXPECT_NE(out[8], 0u) << label << ": checksum zero, drained words were all zero";
        EXPECT_EQ(lanes_behind, 0u) << label << ": drainer exited with lanes still holding markers";
        EXPECT_EQ(lanes_silent, 0u) << label << ": some lane never produced -- all 5 should be live";
        EXPECT_EQ(produced_this_round, total_words) << label << ": drained != produced this round";
        EXPECT_GE(produced_this_round, expected) << label << ": fewer words than producers must have emitted";
    };

    const CoreCoord full_grid = mesh_device_->compute_with_storage_grid_size();
    run_round("4x4", {4, 4}, 2000);
    run_round("full-grid", full_grid, 500);

    set_niu_mode(0);
}
