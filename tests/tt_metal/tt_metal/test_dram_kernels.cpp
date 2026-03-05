// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <cstdint>
#include <vector>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/program.hpp>
#include <umd/device/types/arch.hpp>

#include "common/device_fixture.hpp"
#include "impl/context/metal_context.hpp"
#include "impl/kernels/kernel.hpp"
#include "llrt/hal.hpp"
#include "llrt/tt_cluster.hpp"

using namespace tt;
using namespace tt::tt_metal;

// Skip if not Blackhole (DRAM kernels are BH-only).
class DramKernelFixture : public MeshDeviceSingleCardFixture {
protected:
    void SetUp() override {
        MeshDeviceSingleCardFixture::SetUp();
        if (this->arch_ != ARCH::BLACKHOLE) {
            GTEST_SKIP() << "DramKernel is only supported on Blackhole";
        }
        dev_ = devices_[0]->get_devices()[0];
    }

    IDevice* dev_ = nullptr;
};

// Smoke test: run a single DRAM kernel that writes a compile-time constant to L1,
// then read it back via the host and verify.
TEST_F(DramKernelFixture, DramKernelWriteToL1) {
    constexpr uint32_t kMagicValue = 0xDEADBEEF;

    // Pick the first logical DRAM worker core (bank=0, subchannel=0).
    CoreCoord logical_dram_core{0, 0};
    auto virtual_dram_core = dev_->virtual_core_from_logical_core(logical_dram_core, CoreType::DRAM_WORKER);

    // We'll write to MEM_DRISC_KERNEL_CONFIG_BASE + one word past the RTA/semaphore area.
    // Use a safe scratch address in DRAM core L1 (after the firmware/kernel config area).
    const auto& hal = MetalContext::instance().hal();
    uint64_t l1_noc_offset = hal.get_l1_noc_offset(HalProgrammableCoreType::DRAM);
    uint32_t kernel_config_base = hal.get_dev_addr(HalProgrammableCoreType::DRAM, HalL1MemAddrType::KERNEL_CONFIG);
    uint32_t kernel_config_size = hal.get_dev_size(HalProgrammableCoreType::DRAM, HalL1MemAddrType::KERNEL_CONFIG);
    uint32_t result_l1_addr = kernel_config_base + kernel_config_size;

    Program program = CreateProgram();

    CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/misc/dram_write_one_uint32.cpp",
        logical_dram_core,
        DramConfig{
            .noc = NOC::NOC_0,
            .compile_args = {result_l1_addr, kMagicValue},
        });

    detail::CompileProgram(dev_, program);
    detail::WriteRuntimeArgsToDevice(dev_, program);
    tt::tt_metal::detail::LaunchProgram(dev_, program, /*wait_until_cores_done=*/true, /*force_slow_dispatch=*/true);

    // Read back from DRAM core L1 (requires NOC offset).
    uint64_t read_addr = static_cast<uint64_t>(result_l1_addr) + l1_noc_offset;
    std::vector<uint32_t> result(1, 0);
    MetalContext::instance().get_cluster().read_core(
        result.data(), sizeof(uint32_t), tt_cxy_pair(dev_->id(), virtual_dram_core), read_addr);

    EXPECT_EQ(result[0], kMagicValue);
}

// Run the same kernel across multiple DRAM cores.
TEST_F(DramKernelFixture, DramKernelOnMultipleCores) {
    constexpr uint32_t kMagicBase = 0xCAFE0000;

    const auto& hal = MetalContext::instance().hal();
    uint32_t kernel_config_base = hal.get_dev_addr(HalProgrammableCoreType::DRAM, HalL1MemAddrType::KERNEL_CONFIG);
    uint32_t kernel_config_size = hal.get_dev_size(HalProgrammableCoreType::DRAM, HalL1MemAddrType::KERNEL_CONFIG);
    uint32_t result_l1_addr = kernel_config_base + kernel_config_size;

    // Use internal SoC descriptor to get DRAM worker grid: (num_banks, num_subchannels).
    const auto& soc_desc = MetalContext::instance().get_cluster().get_soc_desc(dev_->id());
    auto dram_worker_grid = soc_desc.get_grid_size(CoreType::DRAM);
    // Test the first row of DRAM worker cores (subchannel=0, up to 4 banks to keep it fast).
    uint32_t num_cores = std::min(static_cast<size_t>(dram_worker_grid.x), static_cast<size_t>(4));

    for (uint32_t col = 0; col < num_cores; col++) {
        CoreCoord logical_dram_core{col, 0};
        auto virtual_dram_core = dev_->virtual_core_from_logical_core(logical_dram_core, CoreType::DRAM_WORKER);
        uint32_t expected_value = kMagicBase + col;

        Program program = CreateProgram();

        CreateKernel(
            program,
            "tests/tt_metal/tt_metal/test_kernels/misc/dram_write_one_uint32.cpp",
            logical_dram_core,
            DramConfig{
                .noc = NOC::NOC_0,
                .compile_args = {result_l1_addr, expected_value},
            });

        detail::CompileProgram(dev_, program);
        detail::WriteRuntimeArgsToDevice(dev_, program);
        tt::tt_metal::detail::LaunchProgram(
            dev_, program, /*wait_until_cores_done=*/true, /*force_slow_dispatch=*/true);

        uint64_t l1_noc_offset = hal.get_l1_noc_offset(HalProgrammableCoreType::DRAM);
        uint64_t read_addr = static_cast<uint64_t>(result_l1_addr) + l1_noc_offset;
        std::vector<uint32_t> result(1, 0);
        MetalContext::instance().get_cluster().read_core(
            result.data(), sizeof(uint32_t), tt_cxy_pair(dev_->id(), virtual_dram_core), read_addr);

        EXPECT_EQ(result[0], expected_value) << "Failed for DRAM core col=" << col;
    }
}
