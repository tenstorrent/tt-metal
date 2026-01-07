// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <chrono>
#include <fmt/base.h>
#include <cstddef>
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
#include <tt_stl/assert.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/data_types.hpp>
#include <tt-metalium/device.hpp>
#include "context/metal_context.hpp"
#include "mesh_dispatch_fixture.hpp"
#include "gtest/gtest.h"
#include <tt-metalium/hal.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/hal_types.hpp>
#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/program.hpp>
#include <tt_stl/span.hpp>
#include <tt-metalium/tt_align.hpp>
#include <umd/device/types/xy_pair.hpp>

using namespace tt;

namespace tt::tt_metal {

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
    std::visit(
        [&](auto&& cfg_variant) {
            using T = std::decay_t<decltype(cfg_variant)>;
            if constexpr (
                std::is_same_v<T, tt::tt_metal::EthernetConfig> ||
                std::is_same_v<T, tt::tt_metal::DataMovementConfig>) {
                kernel = tt_metal::CreateKernel(program, cfg.kernel_file, cfg.core_range, cfg_variant);
            }
        },
        cfg.kernel_cfg);
    return kernel;
}

bool dram_single_core_db(
    tt::tt_metal::MeshDispatchFixture* fixture, const std::shared_ptr<distributed::MeshDevice>& mesh_device) {
    distributed::MeshWorkload workload;
    auto zero_coord = distributed::MeshCoordinate(0, 0);
    distributed::MeshCoordinateRange device_range = distributed::MeshCoordinateRange(zero_coord, zero_coord);
    tt_metal::Program program = tt_metal::CreateProgram();
    workload.add_program(device_range, std::move(program));
    auto& program_ = workload.get_programs().at(device_range);

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

    distributed::DeviceLocalBufferConfig dram_config{
        .page_size = dram_buffer_size_bytes, .buffer_type = tt_metal::BufferType::DRAM, .bottom_up = false};
    distributed::ReplicatedBufferConfig buffer_config{.size = dram_buffer_size_bytes};

    auto input_dram_buffer = distributed::MeshBuffer::create(buffer_config, dram_config, mesh_device.get());
    uint32_t input_dram_buffer_addr = input_dram_buffer->address();

    auto output_dram_buffer = distributed::MeshBuffer::create(buffer_config, dram_config, mesh_device.get());
    uint32_t output_dram_buffer_addr = output_dram_buffer->address();

    auto dram_copy_kernel = tt_metal::CreateKernel(
        program_,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/dram_copy_db.cpp",
        core,
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_0, .noc = tt_metal::NOC::RISCV_0_default});

    std::vector<uint32_t> input_vec = create_random_vector_of_bfloat16(
        dram_buffer_size_bytes, 100, std::chrono::system_clock::now().time_since_epoch().count());
    fixture->WriteBuffer(mesh_device, input_dram_buffer, input_vec);

    tt_metal::SetRuntimeArgs(
        program_,
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

    fixture->RunProgram(mesh_device, workload);

    std::vector<uint32_t> result_vec;
    fixture->ReadBuffer(mesh_device, output_dram_buffer, result_vec);

    return input_vec == result_vec;
}

bool dram_single_core(
    tt::tt_metal::MeshDispatchFixture* fixture,
    const std::shared_ptr<distributed::MeshDevice>& mesh_device,
    const DRAMConfig& cfg) {
    std::vector<uint32_t> src_vec =
        create_random_vector_of_bfloat16(cfg.dram_buffer_size, 100, std::chrono::system_clock::now().time_since_epoch().count());

    // Create a program
    distributed::MeshWorkload workload;
    auto zero_coord = distributed::MeshCoordinate(0, 0);
    distributed::MeshCoordinateRange device_range = distributed::MeshCoordinateRange(zero_coord, zero_coord);
    tt_metal::Program program = tt_metal::CreateProgram();
    workload.add_program(device_range, std::move(program));
    auto& program_ = workload.get_programs().at(device_range);

    distributed::DeviceLocalBufferConfig dram_config{
        .page_size = cfg.dram_buffer_size, .buffer_type = tt_metal::BufferType::DRAM, .bottom_up = false};
    distributed::ReplicatedBufferConfig buffer_config{.size = cfg.dram_buffer_size};

    auto input_dram_buffer = distributed::MeshBuffer::create(buffer_config, dram_config, mesh_device.get());
    uint32_t input_dram_buffer_addr = input_dram_buffer->address();

    auto output_dram_buffer = distributed::MeshBuffer::create(buffer_config, dram_config, mesh_device.get());
    uint32_t output_dram_buffer_addr = output_dram_buffer->address();

    log_info(tt::LogTest, "Creating kernel at {}", cfg.core_range.str());
    log_info(tt::LogTest, "Input DRAM Address  = {:#x}", input_dram_buffer_addr);
    log_info(tt::LogTest, "Output DRAM Address = {:#x}", output_dram_buffer_addr);
    log_info(tt::LogTest, "L1 Buffer Address   = {:#x}", cfg.l1_buffer_addr);
    // Create the kernel
    tt::tt_metal::KernelHandle kernel = CreateKernelFromVariant(program_, cfg);

    fixture->WriteBuffer(mesh_device, input_dram_buffer, src_vec);

    tt_metal::SetRuntimeArgs(
        program_,
        kernel,
        cfg.core_range,
        {cfg.l1_buffer_addr,
         input_dram_buffer_addr,
         (std::uint32_t)0,
         output_dram_buffer_addr,
         (std::uint32_t)0,
         cfg.dram_buffer_size});

    fixture->RunProgram(mesh_device, workload, true);

    std::vector<uint32_t> result_vec(cfg.dram_buffer_size / sizeof(uint32_t));
    fixture->ReadBuffer(mesh_device, output_dram_buffer, result_vec);

    return result_vec == src_vec;
}

bool dram_single_core_pre_allocated(
    tt::tt_metal::MeshDispatchFixture* fixture,
    const std::shared_ptr<distributed::MeshDevice>& mesh_device,
    const DRAMConfig& cfg) {
    std::vector<uint32_t> src_vec =
        create_random_vector_of_bfloat16(cfg.dram_buffer_size, 100, std::chrono::system_clock::now().time_since_epoch().count());
    // Create a program
    distributed::MeshWorkload workload;
    auto zero_coord = distributed::MeshCoordinate(0, 0);
    distributed::MeshCoordinateRange device_range = distributed::MeshCoordinateRange(zero_coord, zero_coord);
    tt_metal::Program program = tt_metal::CreateProgram();
    workload.add_program(device_range, std::move(program));
    auto& program_ = workload.get_programs().at(device_range);

    distributed::DeviceLocalBufferConfig dram_config{
        .page_size = cfg.dram_buffer_size, .buffer_type = tt_metal::BufferType::DRAM, .bottom_up = false};
    distributed::ReplicatedBufferConfig buffer_config{.size = cfg.dram_buffer_size};

    auto input_dram_buffer = distributed::MeshBuffer::create(buffer_config, dram_config, mesh_device.get());
    uint32_t input_dram_buffer_addr = input_dram_buffer->address();
    auto input_dram_pre_allocated_buffer =
        distributed::MeshBuffer::create(buffer_config, dram_config, mesh_device.get(), input_dram_buffer_addr);
    uint32_t input_dram_pre_allocated_buffer_addr = input_dram_pre_allocated_buffer->address();

    EXPECT_EQ(input_dram_buffer_addr, input_dram_pre_allocated_buffer_addr);

    auto output_dram_buffer = distributed::MeshBuffer::create(buffer_config, dram_config, mesh_device.get());
    uint32_t output_dram_buffer_addr = output_dram_buffer->address();
    auto output_dram_pre_allocated_buffer =
        distributed::MeshBuffer::create(buffer_config, dram_config, mesh_device.get(), output_dram_buffer_addr);
    uint32_t output_dram_pre_allocated_buffer_addr = output_dram_pre_allocated_buffer->address();

    EXPECT_EQ(output_dram_buffer_addr, output_dram_pre_allocated_buffer_addr);

    // Create the kernel
    tt::tt_metal::KernelHandle dram_kernel = CreateKernelFromVariant(program_, cfg);
    fixture->WriteBuffer(mesh_device, input_dram_pre_allocated_buffer, src_vec);

    tt_metal::SetRuntimeArgs(
        program_,
        dram_kernel,
        cfg.core_range,
        {cfg.l1_buffer_addr,
         input_dram_pre_allocated_buffer_addr,
         (std::uint32_t)0,
         output_dram_pre_allocated_buffer_addr,
         (std::uint32_t)0,
         cfg.dram_buffer_size});

    fixture->RunProgram(mesh_device, workload);

    std::vector<uint32_t> result_vec;
    fixture->ReadBuffer(mesh_device, output_dram_pre_allocated_buffer, result_vec);

    return result_vec == src_vec;
}
}  // namespace unit_tests_common::dram::test_dram

TEST_F(MeshDispatchFixture, TensixDRAMLoopbackSingleCore) {
    constexpr uint32_t buffer_size = 2 * 1024 * 25;

    unit_tests_common::dram::test_dram::DRAMConfig dram_test_config = {
        .core_range = {{0, 0}, {0, 0}},
        .kernel_file = "tests/tt_metal/tt_metal/test_kernels/dataflow/dram_copy.cpp",
        .dram_buffer_size = buffer_size,
        .l1_buffer_addr = 400 * 1024,
        .kernel_cfg =
            tt::tt_metal::DataMovementConfig{.processor = tt_metal::DataMovementProcessor::RISCV_0, .noc = tt_metal::NOC::RISCV_0_default},
    };
    for (const auto& mesh_device : devices_) {
        ASSERT_TRUE(unit_tests_common::dram::test_dram::dram_single_core(this, mesh_device, dram_test_config));
    }
}

TEST_F(MeshDispatchFixture, TensixDRAMLoopbackSingleCorePreAllocated) {
    constexpr uint32_t buffer_size = 2 * 1024 * 25;
    unit_tests_common::dram::test_dram::DRAMConfig dram_test_config = {
        .core_range = {{0, 0}, {0, 0}},
        .kernel_file = "tests/tt_metal/tt_metal/test_kernels/dataflow/dram_copy.cpp",
        .dram_buffer_size = buffer_size,
        .l1_buffer_addr = 400 * 1024,
        .kernel_cfg =
            tt::tt_metal::DataMovementConfig{.processor = tt_metal::DataMovementProcessor::RISCV_0, .noc = tt_metal::NOC::RISCV_0_default},
    };
    for (const auto& mesh_device : devices_) {
        ASSERT_TRUE(
            unit_tests_common::dram::test_dram::dram_single_core_pre_allocated(this, mesh_device, dram_test_config));
    }
}

TEST_F(MeshDispatchFixture, TensixDRAMLoopbackSingleCoreDB) {
    if (!this->IsSlowDispatch()) {
        log_info(tt::LogTest, "This test is only supported in slow dispatch mode");
        GTEST_SKIP();
    }
    for (const auto& mesh_device : devices_) {
        ASSERT_TRUE(unit_tests_common::dram::test_dram::dram_single_core_db(this, mesh_device));
    }
}

TEST_F(MeshDispatchFixture, ActiveEthDRAMLoopbackSingleCore) {
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

    for (const auto& mesh_device : devices_) {
        auto* device = mesh_device->get_devices()[0];
        for (auto active_eth_core : device->get_active_ethernet_cores(true)) {
            log_info(tt::LogTest, "Active Eth Loopback. Logical core {}", active_eth_core.str());
            dram_test_config.core_range = {active_eth_core, active_eth_core};
            const auto erisc_count = tt::tt_metal::MetalContext::instance().hal().get_num_risc_processors(
                HalProgrammableCoreType::ACTIVE_ETH);
            for (uint32_t erisc_idx = 0; erisc_idx < erisc_count; ++erisc_idx) {
                log_info(tt::LogTest, "Active Eth DM{} Loopback. Logical core {}", erisc_idx, active_eth_core.str());
                dram_test_config.kernel_cfg = tt_metal::EthernetConfig{
                    .eth_mode = Eth::RECEIVER,
                    .noc = static_cast<tt_metal::NOC>(erisc_idx),
                    .processor = static_cast<DataMovementProcessor>(erisc_idx)};
                ASSERT_TRUE(unit_tests_common::dram::test_dram::dram_single_core(this, mesh_device, dram_test_config));
            }
        }
    }
}

TEST_F(MeshDispatchFixture, IdleEthDRAMLoopbackSingleCore) {
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

    for (const auto& mesh_device : devices_) {
        auto* device = mesh_device->get_devices()[0];
        for (auto idle_eth_core : device->get_inactive_ethernet_cores()) {
            log_info(tt::LogTest, "Single Idle Eth Loopback. Logical core {}", idle_eth_core.str());
            dram_test_config.core_range = {idle_eth_core, idle_eth_core};
            const auto erisc_count =
                tt::tt_metal::MetalContext::instance().hal().get_num_risc_processors(HalProgrammableCoreType::IDLE_ETH);
            for (uint32_t erisc_idx = 0; erisc_idx < erisc_count; ++erisc_idx) {
                log_info(tt::LogTest, "Single Idle Eth DM{} Loopback. Logical core {}", erisc_idx, idle_eth_core.str());
                dram_test_config.kernel_cfg = tt_metal::EthernetConfig{
                    .eth_mode = Eth::IDLE,
                    .noc = static_cast<tt_metal::NOC>(erisc_idx),
                    .processor = static_cast<DataMovementProcessor>(erisc_idx)};
                ASSERT_TRUE(unit_tests_common::dram::test_dram::dram_single_core(this, mesh_device, dram_test_config));
            }
        }
    }
}

// This test will hang on BH when both nocs use the same DRAM endpoint due to SYS-1419, hang can be reproduced by
// increasing `num_iterations` DRAM arbiter seems to drop requests from one noc when both nocs are issuing requests to
// the same DRAM endpoint at a fast rate.
// This test should be kept to facilitate getting a scandump for root-causing SYS-1419 but keep it DISABLED so it doesn't run on CI
TEST_F(MeshDispatchFixture, DISABLED_TensixLoopDRAMReadSingleCoreBothProcessors) {
    if (this->arch_ != tt::ARCH::BLACKHOLE) {
        GTEST_SKIP() << "This test has hardcoded parameters for Blackhole to repro hang described in SYS-1419";
    }
    auto mesh_device = devices_.at(0);
    distributed::MeshWorkload workload;
    auto zero_coord = distributed::MeshCoordinate(0, 0);
    distributed::MeshCoordinateRange device_range = distributed::MeshCoordinateRange(zero_coord, zero_coord);
    tt_metal::Program program = tt_metal::CreateProgram();
    workload.add_program(device_range, std::move(program));
    auto& program_ = workload.get_programs().at(device_range);
    CoreCoord core = {0, 0};

    const uint32_t l1_address = mesh_device->allocator()->get_base_allocator_addr(HalMemType::L1);
    const uint32_t num_drams = mesh_device->num_dram_channels();

    constexpr uint32_t page_size = 2048;
    constexpr uint32_t num_iterations = 100;

    const uint32_t brisc_base_addr = 8519744;
    uint32_t ncrisc_base_addr = MetalContext::instance().hal().get_dev_addr(HalDramMemAddrType::UNRESERVED);

    uint32_t brisc_num_pages_to_read = 43264;
    uint32_t ncrisc_num_pages_to_read = ((brisc_base_addr - ncrisc_base_addr) / page_size) * num_drams;

    std::vector<uint32_t> brisc_compile_time_args = {};
    tt_metal::TensorAccessorArgs::create_dram_interleaved().append_to(brisc_compile_time_args);

    tt_metal::KernelHandle brisc_kernel = tt_metal::CreateKernel(
        program_,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/dram_arbiter_hang.cpp",
        core,
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_0,
            .noc = tt_metal::NOC::RISCV_0_default,
            .compile_args = brisc_compile_time_args});

    tt_metal::SetRuntimeArgs(
        program_,
        brisc_kernel,
        core,
        {brisc_base_addr, page_size, l1_address, brisc_num_pages_to_read, num_iterations});

    std::vector<uint32_t> ncrisc_compile_time_args = {};
    tt_metal::TensorAccessorArgs::create_dram_interleaved().append_to(ncrisc_compile_time_args);

    tt_metal::KernelHandle ncrisc_kernel = tt_metal::CreateKernel(
        program_,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/dram_arbiter_hang.cpp",
        core,
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_1,
            .noc = tt_metal::NOC::RISCV_1_default,
            .compile_args = ncrisc_compile_time_args});

    tt_metal::SetRuntimeArgs(
        program_,
        ncrisc_kernel,
        core,
        {
            ncrisc_base_addr,
            page_size,
            l1_address,
            ncrisc_num_pages_to_read,
            num_iterations,
        });

    this->RunProgram(mesh_device, workload);
}
}  // namespace tt::tt_metal
