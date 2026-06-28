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
    CoreCoord drisc_virtual = device_->virtual_core_from_logical_core(logical_core_drisc, CoreType::DRAM);

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
    tt::tt_metal::detail::ReadFromDeviceL1(
        device_, logical_core_tensix, tensix_l1_base_, sizeof(kMagicValue), result, CoreType::WORKER);
    log_info(LogTest, "Tensix L1 result: 0x{:X} (expected: 0x{:X})", result[0], kMagicValue);
    EXPECT_EQ(result[0], kMagicValue) << "Tensix should have read the value from DRISC L1";
}

// Test DRISC reading from Tensix L1
// Host writes magic value to Tensix L1, then DRISC reads it into DRISC L1
TEST_F(DramKernelFixture, DramKernelDRISCReadFromTensixL1) {
    constexpr uint32_t kMagicValue = 0xDEADBEEF;
    CoreCoord logical_core_drisc{0, first_usable_dram_endpoint(0)};
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
    auto dram_buffer = CreateBuffer(InterleavedBufferConfig{
        .device = device_,
        .size = num_banks * num_endpoints * total_bytes_per_core,
        .page_size = num_endpoints * total_bytes_per_core,
        .buffer_type = BufferType::DRAM,
    });
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
            CoreCoord virtual_core = device_->virtual_core_from_logical_core(logical_core, CoreType::DRAM);
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
    uint32_t clk_hz = MetalContext::instance().get_cluster().get_device_aiclk(device_->id()) * 1000000u;
    uint64_t max_cycles = 0;

    // Verify all DRISCs writes to DRAM and calculate
    // the aggregate write bandwidth from DRISC L1 to DRAM across all endpoints
    for (uint32_t i = 0; i < num_endpoints; i++) {
        const uint32_t row = usable_endpoints[i];
        for (uint32_t col = 0; col < num_banks; col++) {
            auto begin = data.begin() + endpoint_offset(i, col);
            std::vector<uint32_t> endpoint_data(begin, begin + elements_per_endpoint);
            uint32_t dram_channel =
                device_->dram_channel_from_logical_core(CoreCoord{col, 0});  // channel maps by bank (col)
            // ReadFromDeviceDRAMChannel is slow (host-device round-trip); avoid reading all iters.
            std::vector<uint32_t> result(elements_per_endpoint);
            tt::tt_metal::detail::ReadFromDeviceDRAMChannel(
                device_,
                dram_channel,
                dram_addr + bytes_per_iter * (iters - 1) + total_bytes_per_core * i,
                bytes_per_iter,
                result);
            EXPECT_EQ(result, endpoint_data)
                << "Data mismatch on DRAM from core (bank=" << col << ", endpoint=" << row << ")";
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
    auto dram_buffer = CreateBuffer(InterleavedBufferConfig{
        .device = device_,
        .size = num_banks * num_endpoints * bytes_per_iter,
        .page_size = num_endpoints * bytes_per_iter,
        .buffer_type = BufferType::DRAM,
    });
    uint32_t dram_addr = dram_buffer->address();

    auto seed = std::chrono::system_clock::now().time_since_epoch().count();
    log_info(LogTest, "Random seed: {}", seed);
    // Unique data per endpoint proves each endpoint has independent L1.
    std::vector<uint32_t> data =
        create_random_vector_of_bfloat16(num_banks * num_endpoints * bytes_per_iter, 1000.0f, seed);
    auto endpoint_offset = [&](uint32_t i, uint32_t col) { return (i * num_banks + col) * elements_per_endpoint; };

    // Write data from DRISCs to read to all DRAM Banks
    for (uint32_t col = 0; col < num_banks; col++) {
        uint32_t dram_channel = device_->dram_channel_from_logical_core(CoreCoord{col, 0});
        for (uint32_t i = 0; i < num_endpoints; i++) {
            auto begin = data.begin() + endpoint_offset(i, col);
            std::vector<uint32_t> endpoint_data(begin, begin + elements_per_endpoint);
            tt::tt_metal::detail::WriteToDeviceDRAMChannel(
                device_, dram_channel, dram_addr + i * bytes_per_iter, endpoint_data);
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
    uint32_t clk_hz = MetalContext::instance().get_cluster().get_device_aiclk(device_->id()) * 1000000u;
    uint64_t max_cycles = 0;

    // Verify all reads into DRISC L1 over DMA from DRAM are correct
    for (uint32_t i = 0; i < num_endpoints; i++) {
        const uint32_t row = usable_endpoints[i];
        for (uint32_t col = 0; col < num_banks; col++) {
            CoreCoord virtual_core = device_->virtual_core_from_logical_core({col, row}, CoreType::DRAM);
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

    // Allocate a single-page DRAM buffer. Page_size == size pins it to bank 0 (logical_core x==0)
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
         num_subordinates,
         1u});

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
    uint32_t dram_channel = device_->dram_channel_from_logical_core(logical_core);
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
    SetRuntimeArgs(program, tensix_k_id, tensix_range, {bank_id, dram_addr, tensix_l1_base_, total_bytes, 1u});

    run_workload(std::move(program));

    // Verify DRISC L1 reads are correct
    for (uint32_t endpoint : usable_endpoints) {
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

    // Bank 0, first DRISC-usable endpoint (subchannel 0 is the syseng-owned NOC0 endpoint).
    CoreCoord logical_core{0, first_usable_dram_endpoint(0)};
    uint32_t dram_channel = device_->dram_channel_from_logical_core(logical_core);
    CoreCoord tensix_logical{0, 0};
    CoreCoord sub_worker = device_->virtual_core_from_logical_core(tensix_logical, CoreType::WORKER);

    // Allocate a single-page DRAM buffer. Page_size == size pins it to bank 0 (logical_core x==0)
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
    const uint32_t dram_channel = device_->dram_channel_from_logical_core(drisc_logical);

    // Fill GDDR with iters random distinct chunks. DRISC and the Tensix reader both walk the same region
    // Only the final chunk remains in L1 after the run, so verification compares against it
    auto dram_buffer = CreateBuffer(InterleavedBufferConfig{
        .device = device_,
        .size = total_bytes,
        .page_size = total_bytes,  // single bank (bank 0)
        .buffer_type = BufferType::DRAM,
    });
    uint32_t dram_addr = dram_buffer->address();

    auto seed = std::chrono::system_clock::now().time_since_epoch().count();
    log_info(LogTest, "Random seed: {}", seed);
    std::vector<uint32_t> data = create_random_vector_of_bfloat16(total_bytes, 1000.0f, seed);
    tt::tt_metal::detail::WriteToDeviceDRAMChannel(device_, dram_channel, dram_addr, data);
    std::vector<uint32_t> last_chunk(data.end() - elements_per_iter, data.end());

    Program program = CreateProgram();

    // DRISC stream kernel: read GDDR chunks for multiple iterations, multicasting each to the 4x3 grid
    CoreCoord mcast_start = device_->virtual_core_from_logical_core({0, 0}, CoreType::WORKER);
    CoreCoord mcast_end = device_->virtual_core_from_logical_core({mcast_cols - 1, mcast_rows - 1}, CoreType::WORKER);
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
            CoreCoord v = device_->virtual_core_from_logical_core({col, row}, CoreType::WORKER);
            std::vector<uint32_t> result(elements_per_iter);
            MetalContext::instance().get_cluster().read_core(
                result.data(), bytes_per_iter, tt_cxy_pair(mesh_device_->build_id(), v), tensix_l1_base_);
            EXPECT_EQ(result, last_chunk) << "Mcast last-chunk mismatch at Tensix (" << col << ", " << row << ")";
        }
    }

    // Verify the Tensix DRAM reader received the last chunk via the NOC2AXI NIU.
    CoreCoord reader_v = device_->virtual_core_from_logical_core(tensix_reader_logical, CoreType::WORKER);
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
