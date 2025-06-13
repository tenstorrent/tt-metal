// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <chrono>
#include <fmt/base.h>
#include <stddef.h>
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-logger/tt-logger.hpp>
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <type_traits>
#include <unordered_set>
#include <variant>
#include <vector>

#include <tt-metalium/allocator.hpp>
#include <tt-metalium/assert.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/data_types.hpp>
#include <tt-metalium/device.hpp>
#include "context/metal_context.hpp"
#include "dispatch_fixture.hpp"
#include "gtest/gtest.h"
#include <tt-metalium/hal.hpp>
#include <tt-metalium/hal_types.hpp>
#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/program.hpp>
#include <tt_stl/span.hpp>
#include <tt-metalium/tt_align.hpp>
#include "umd/device/types/xy_pair.h"

using namespace tt;

namespace unit_tests_common::dram::test_dram {
struct DRAMConfig {
    using CfgVariant = std::variant<tt_metal::DataMovementConfig, tt_metal::EthernetConfig>;

    static constexpr size_t k_KernelCfgDM = 0;
    static constexpr size_t k_KernelCfgETH = 1;

    // CoreRange, Kernel, dram_buffer_size
    CoreRange core_range;
    std::string kernel_file;
    std::uint32_t dram_buffer_size;
    std::uint32_t l1_buffer_addr;
    CfgVariant kernel_cfg;
};

tt::tt_metal::KernelHandle CreateKernelFromVariant(tt::tt_metal::Program& program, DRAMConfig cfg) {
    tt::tt_metal::KernelHandle kernel;
    std::visit([&](auto&& cfg_variant) {
        if constexpr (std::is_same_v<std::decay_t<decltype(cfg_variant)>, tt::tt_metal::EthernetConfig>) {
            kernel = tt_metal::CreateKernel(program, cfg.kernel_file, cfg.core_range, cfg_variant);
        } else if constexpr (std::is_same_v<std::decay_t<decltype(cfg_variant)>, tt::tt_metal::DataMovementConfig>) {
            kernel = tt_metal::CreateKernel(program, cfg.kernel_file, cfg.core_range, cfg_variant);
        }
    }, cfg.kernel_cfg);
    return kernel;
}

bool dram_single_core_db(tt::tt_metal::DispatchFixture* fixture, tt_metal::IDevice* device) {
    tt_metal::Program program = tt_metal::CreateProgram();

    CoreCoord core = {0, 0};

    uint32_t single_tile_size = 2 * 1024;
    uint32_t num_tiles = 256;
    uint32_t dram_buffer_size_bytes = single_tile_size * num_tiles;

    // L1 buffer is double buffered
    // We read and write total_l1_buffer_size_tiles / 2 tiles from and to DRAM
    uint32_t l1_buffer_addr = 400 * 1024;
    uint32_t total_l1_buffer_size_tiles = num_tiles / 2;
    TT_FATAL(total_l1_buffer_size_tiles % 2 == 0, "Error");
    uint32_t total_l1_buffer_size_bytes = total_l1_buffer_size_tiles * single_tile_size;

    tt_metal::InterleavedBufferConfig dram_config{
        .device = device,
        .size = dram_buffer_size_bytes,
        .page_size = dram_buffer_size_bytes,
        .buffer_type = tt_metal::BufferType::DRAM};

    auto input_dram_buffer = CreateBuffer(dram_config);
    uint32_t input_dram_buffer_addr = input_dram_buffer->address();

    auto output_dram_buffer = CreateBuffer(dram_config);
    uint32_t output_dram_buffer_addr = output_dram_buffer->address();

    auto dram_copy_kernel = tt_metal::CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/dram_copy_db.cpp",
        core,
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_0, .noc = tt_metal::NOC::RISCV_0_default});

    std::vector<uint32_t> input_vec = create_random_vector_of_bfloat16(
        dram_buffer_size_bytes, 100, std::chrono::system_clock::now().time_since_epoch().count());
    fixture->WriteBuffer(device, input_dram_buffer, input_vec);

    tt_metal::SetRuntimeArgs(
        program,
        dram_copy_kernel,
        core,
        {input_dram_buffer_addr,
        (std::uint32_t)0,
        output_dram_buffer_addr,
        (std::uint32_t)0,
        dram_buffer_size_bytes,
        num_tiles,
        l1_buffer_addr,
        total_l1_buffer_size_tiles,
        total_l1_buffer_size_bytes});

    fixture->RunProgram(device, program);

    std::vector<uint32_t> result_vec;
    fixture->ReadBuffer(device, output_dram_buffer, result_vec);

    return input_vec == result_vec;
}

bool dram_single_core(
    tt::tt_metal::DispatchFixture* fixture,
    tt_metal::IDevice* device,
    const DRAMConfig& cfg) {
    std::vector<uint32_t> src_vec =
        create_random_vector_of_bfloat16(cfg.dram_buffer_size, 100, std::chrono::system_clock::now().time_since_epoch().count());

    // Create a program
    tt_metal::Program program = tt::tt_metal::CreateProgram();

    tt_metal::InterleavedBufferConfig dram_config{
        .device = device,
        .size = cfg.dram_buffer_size,
        .page_size = cfg.dram_buffer_size,
        .buffer_type = tt_metal::BufferType::DRAM};
    auto input_dram_buffer = tt_metal::CreateBuffer(dram_config);
    uint32_t input_dram_buffer_addr = input_dram_buffer->address();

    auto output_dram_buffer = tt_metal::CreateBuffer(dram_config);
    uint32_t output_dram_buffer_addr = output_dram_buffer->address();

    log_info(tt::LogTest, "Creating kernel at {}", cfg.core_range.str());
    log_info(tt::LogTest, "Input DRAM Address  = {:#x}", input_dram_buffer_addr);
    log_info(tt::LogTest, "Output DRAM Address = {:#x}", output_dram_buffer_addr);
    log_info(tt::LogTest, "L1 Buffer Address   = {:#x}", cfg.l1_buffer_addr);
    // Create the kernel
    tt::tt_metal::KernelHandle kernel = CreateKernelFromVariant(program, cfg);

    fixture->WriteBuffer(device, input_dram_buffer, src_vec);

    tt_metal::SetRuntimeArgs(
        program,
        kernel,
        cfg.core_range,
        {cfg.l1_buffer_addr,
        input_dram_buffer_addr,
        (std::uint32_t)0,
        output_dram_buffer_addr,
        (std::uint32_t)0,
        cfg.dram_buffer_size});

    fixture->RunProgram(device, program);

    std::vector<uint32_t> result_vec;
    fixture->ReadBuffer(device, output_dram_buffer, result_vec);
    return result_vec == src_vec;
}

bool dram_single_core_pre_allocated(
    tt::tt_metal::DispatchFixture* fixture,
    tt_metal::IDevice* device,
    const DRAMConfig& cfg) {
    std::vector<uint32_t> src_vec =
        create_random_vector_of_bfloat16(cfg.dram_buffer_size, 100, std::chrono::system_clock::now().time_since_epoch().count());
    // Create a program
    tt_metal::Program program = tt::tt_metal::CreateProgram();

    tt_metal::InterleavedBufferConfig dram_config{
        .device = device,
        .size = cfg.dram_buffer_size,
        .page_size = cfg.dram_buffer_size,
        .buffer_type = tt_metal::BufferType::DRAM};

    auto input_dram_buffer = tt_metal::CreateBuffer(dram_config);
    uint32_t input_dram_buffer_addr = input_dram_buffer->address();
    auto input_dram_pre_allocated_buffer = tt_metal::CreateBuffer(dram_config, input_dram_buffer_addr);
    uint32_t input_dram_pre_allocated_buffer_addr = input_dram_pre_allocated_buffer->address();

    EXPECT_EQ(input_dram_buffer_addr, input_dram_pre_allocated_buffer_addr);

    auto output_dram_buffer = tt_metal::CreateBuffer(dram_config);
    uint32_t output_dram_buffer_addr = output_dram_buffer->address();
    auto output_dram_pre_allocated_buffer = tt_metal::CreateBuffer(dram_config, output_dram_buffer_addr);
    uint32_t output_dram_pre_allocated_buffer_addr = output_dram_pre_allocated_buffer->address();

    EXPECT_EQ(output_dram_buffer_addr, output_dram_pre_allocated_buffer_addr);

    // Create the kernel
    tt::tt_metal::KernelHandle dram_kernel = CreateKernelFromVariant(program, cfg);
    fixture->WriteBuffer(device, input_dram_pre_allocated_buffer, src_vec);

    tt_metal::SetRuntimeArgs(
        program,
        dram_kernel,
        cfg.core_range,
        {cfg.l1_buffer_addr,
         input_dram_pre_allocated_buffer_addr,
         (std::uint32_t)0,
         output_dram_pre_allocated_buffer_addr,
         (std::uint32_t)0,
         cfg.dram_buffer_size});

    fixture->RunProgram(device, program);

    std::vector<uint32_t> result_vec;
    fixture->ReadBuffer(device, output_dram_pre_allocated_buffer, result_vec);

    return result_vec == src_vec;
}
}  // namespace unit_tests_common::dram::test_dram

namespace tt::tt_metal {

TEST_F(DispatchFixture, TensixDRAMLoopbackSingleCore) {
    constexpr uint32_t buffer_size = 2 * 1024 * 25;
    unit_tests_common::dram::test_dram::DRAMConfig dram_test_config = {
        .core_range = {{0, 0}, {0, 0}},
        .kernel_file = "tests/tt_metal/tt_metal/test_kernels/dataflow/dram_copy.cpp",
        .dram_buffer_size = buffer_size,
        .l1_buffer_addr = 400 * 1024,
        .kernel_cfg =
            tt::tt_metal::DataMovementConfig{.processor = tt_metal::DataMovementProcessor::RISCV_0, .noc = tt_metal::NOC::RISCV_0_default},
    };
    for (unsigned int id = 0; id < devices_.size(); id++) {
        ASSERT_TRUE(
            unit_tests_common::dram::test_dram::dram_single_core(this, devices_.at(id), dram_test_config));
    }
}

TEST_F(DispatchFixture, TensixDRAMLoopbackSingleCorePreAllocated) {
    constexpr uint32_t buffer_size = 2 * 1024 * 25;
    unit_tests_common::dram::test_dram::DRAMConfig dram_test_config = {
        .core_range = {{0, 0}, {0, 0}},
        .kernel_file = "tests/tt_metal/tt_metal/test_kernels/dataflow/dram_copy.cpp",
        .dram_buffer_size = buffer_size,
        .l1_buffer_addr = 400 * 1024,
        .kernel_cfg =
            tt::tt_metal::DataMovementConfig{.processor = tt_metal::DataMovementProcessor::RISCV_0, .noc = tt_metal::NOC::RISCV_0_default},
    };
    for (unsigned int id = 0; id < devices_.size(); id++) {
        ASSERT_TRUE(unit_tests_common::dram::test_dram::dram_single_core_pre_allocated(
            this, devices_.at(id), dram_test_config));
    }
}

TEST_F(DispatchFixture, TensixDRAMLoopbackSingleCoreDB) {
    if (!this->IsSlowDispatch()) {
        log_info(tt::LogTest, "This test is only supported in slow dispatch mode");
        GTEST_SKIP();
    }
    for (unsigned int id = 0; id < devices_.size(); id++) {
        ASSERT_TRUE(unit_tests_common::dram::test_dram::dram_single_core_db(this, devices_.at(id)));
    }
}

TEST_F(DispatchFixture, ActiveEthDRAMLoopbackSingleCore) {
    constexpr uint32_t buffer_size = 2 * 1024 * 25;

    unit_tests_common::dram::test_dram::DRAMConfig dram_test_config = {
        .core_range = {{0, 0}, {0, 0}},
        .kernel_file = "tests/tt_metal/tt_metal/test_kernels/dataflow/dram_copy.cpp",
        .dram_buffer_size = buffer_size,
        .l1_buffer_addr = tt::align(
            tt::tt_metal::hal::get_erisc_l1_unreserved_base(),
            MetalContext::instance().hal().get_alignment(HalMemType::DRAM)),
        .kernel_cfg = tt_metal::EthernetConfig{},
    };

    for (unsigned int id = 0; id < devices_.size(); id++) {
        for (auto active_eth_core : devices_.at(id)->get_active_ethernet_cores(true)) {
            const auto erisc_count = tt::tt_metal::MetalContext::instance().hal().get_processor_classes_count(
                HalProgrammableCoreType::ACTIVE_ETH);
            for (int erisc_idx = 0; erisc_idx < erisc_count; ++erisc_idx) {
                log_info(tt::LogTest, "Active Eth DM{} Loopback. Logical core {}", erisc_idx, active_eth_core.str());
                dram_test_config.core_range = {active_eth_core, active_eth_core};
                dram_test_config.kernel_cfg = tt_metal::EthernetConfig{
                    .eth_mode = Eth::RECEIVER,
                    .noc = tt_metal::NOC::NOC_0,
                    .processor = static_cast<DataMovementProcessor>(erisc_idx)};
                ASSERT_TRUE(
                    unit_tests_common::dram::test_dram::dram_single_core(this, devices_.at(id), dram_test_config));
            }
        }
    }
}

TEST_F(DispatchFixture, IdleEthDRAMLoopbackSingleCore) {
    constexpr uint32_t buffer_size = 2 * 1024 * 25;

    if (!this->IsSlowDispatch()) {
        log_info(tt::LogTest, "This test is only supported in slow dispatch mode");
        GTEST_SKIP();
    }

    unit_tests_common::dram::test_dram::DRAMConfig dram_test_config = {
        .core_range = {{0, 0}, {0, 0}},  // Set below
        .kernel_file = "tests/tt_metal/tt_metal/test_kernels/dataflow/dram_copy.cpp",
        .dram_buffer_size = buffer_size,
        .l1_buffer_addr = tt::align(
            tt::tt_metal::hal::get_erisc_l1_unreserved_base(),
            MetalContext::instance().hal().get_alignment(HalMemType::DRAM)),
        .kernel_cfg = tt_metal::EthernetConfig{},
    };

    for (unsigned int id = 0; id < devices_.size(); id++) {
        for (auto idle_eth_core : devices_.at(id)->get_inactive_ethernet_cores()) {
            const auto erisc_count = tt::tt_metal::MetalContext::instance().hal().get_processor_classes_count(
                HalProgrammableCoreType::IDLE_ETH);
            for (int erisc_idx = 0; erisc_idx < erisc_count; ++erisc_idx) {
                log_info(tt::LogTest, "Single Idle Eth DM{} Loopback. Logical core {}", erisc_idx, idle_eth_core.str());
                dram_test_config.core_range = {idle_eth_core, idle_eth_core};
                dram_test_config.kernel_cfg = tt_metal::EthernetConfig{
                    .eth_mode = Eth::IDLE,
                    .noc = tt_metal::NOC::NOC_0,
                    .processor = static_cast<DataMovementProcessor>(erisc_idx)};
                unit_tests_common::dram::test_dram::dram_single_core(this, devices_.at(id), dram_test_config);
            }
        }
    }
}

// This test will hang on BH when both nocs use the same DRAM endpoint due to SYS-1419, hang can be reproduced by
// increasing `num_iterations` DRAM arbiter seems to drop requests from one noc when both nocs are issuing requests to
// the same DRAM endpoint at a fast rate.
TEST_F(DispatchFixture, TensixLoopDRAMReadSingleCoreBothProcessors) {
    auto device = devices_.at(0);
    tt_metal::Program program = tt_metal::CreateProgram();
    CoreCoord core = {0, 0};

    const uint32_t l1_address = device->allocator()->get_base_allocator_addr(HalMemType::L1);
    const uint32_t dram_size = device->dram_size_per_channel();
    const uint32_t num_drams = device->num_dram_channels();

    constexpr uint32_t page_size = 2048;
    constexpr uint32_t num_iterations = 1000;

    const uint32_t brisc_base_addr = 8519744;
    uint32_t ncrisc_base_addr = MetalContext::instance().hal().get_dev_addr(HalDramMemAddrType::UNRESERVED);

    uint32_t brisc_num_pages_to_read = 43264;
    uint32_t ncrisc_num_pages_to_read = ((brisc_base_addr - ncrisc_base_addr) / page_size) * num_drams;

    tt_metal::KernelHandle brisc_kernel = tt_metal::CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/dram_arbiter_hang.cpp",
        core,
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_0, .noc = tt_metal::NOC::RISCV_0_default});

    tt_metal::SetRuntimeArgs(
        program, brisc_kernel, core, {brisc_base_addr, page_size, l1_address, brisc_num_pages_to_read, num_iterations});

    tt_metal::KernelHandle ncrisc_kernel = tt_metal::CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/dram_arbiter_hang.cpp",
        core,
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_1, .noc = tt_metal::NOC::RISCV_1_default});

    tt_metal::SetRuntimeArgs(
        program,
        ncrisc_kernel,
        core,
        {
            ncrisc_base_addr,
            page_size,
            l1_address,
            ncrisc_num_pages_to_read,
            num_iterations,
        });

    this->RunProgram(device, program);
}

}  // namespace tt::tt_metal
