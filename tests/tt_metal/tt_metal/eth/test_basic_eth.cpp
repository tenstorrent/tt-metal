// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include "command_queue_fixture.hpp"
#include "device_fixture.hpp"
#include "dispatch_fixture.hpp"
#include "multi_device_fixture.hpp"
#include "tt_metal/detail/tt_metal.hpp"
#include "host_api.hpp"
#include "tt_metal/impl/kernels/kernel.hpp"
#include "tt_metal/test_utils/stimulus.hpp"
#include "tt_metal/llrt/llrt.hpp"

// TODO: ARCH_NAME specific, must remove
#include "eth_l1_address_map.h"

using namespace tt;
using namespace tt::test_utils;
// using namespace tt::test_utils::df;

namespace {
namespace CMAKE_UNIQUE_NAMESPACE {
constexpr std::int32_t WORD_SIZE = 16;  // 16 bytes per eth send packet
constexpr std::int32_t MAX_NUM_WORDS =
    (eth_l1_mem::address_map::MAX_L1_LOADING_SIZE - eth_l1_mem::address_map::ERISC_L1_UNRESERVED_BASE) / WORD_SIZE;
}  // namespace CMAKE_UNIQUE_NAMESPACE
}  // namespace

namespace unit_tests::erisc::kernels {

/*
 *                                         ███╗░░██╗░█████╗░░█████╗░
 *                                         ████╗░██║██╔══██╗██╔══██╗
 *                                         ██╔██╗██║██║░░██║██║░░╚═╝
 *                                         ██║╚████║██║░░██║██║░░██╗
 *                                         ██║░╚███║╚█████╔╝╚█████╔╝
 *                                         ╚═╝░░╚══╝░╚════╝░░╚════╝░
 */

bool reader_kernel_no_send(
    DispatchFixture* fixture,
    tt_metal::Device* device,
    const size_t& byte_size,
    const size_t& eth_l1_byte_address,
    const CoreCoord& eth_reader_core,
    const tt_metal::EthernetConfig& ethernet_config = tt_metal::EthernetConfig{.noc = tt_metal::NOC::NOC_0}) {
    bool pass = true;
    ////////////////////////////////////////////////////////////////////////////
    //                      Application Setup
    ////////////////////////////////////////////////////////////////////////////
    tt_metal::Program program = tt_metal::Program();

    tt::tt_metal::InterleavedBufferConfig dram_config{
        .device = device, .size = byte_size, .page_size = byte_size, .buffer_type = tt::tt_metal::BufferType::DRAM};

    auto input_dram_buffer = CreateBuffer(dram_config);
    uint32_t dram_byte_address = input_dram_buffer->address();
    auto eth_noc_xy = device->ethernet_core_from_logical_core(eth_reader_core);
    log_debug(
        tt::LogTest,
        "Device {}: reading {} bytes from dram bank 0 addr {} to ethernet core {} addr {}",
        device->id(),
        byte_size,
        dram_byte_address,
        eth_reader_core.str(),
        eth_l1_byte_address);

    auto eth_reader_kernel = tt_metal::CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/unit_tests/erisc/direct_reader_dram_to_l1.cpp",
        eth_reader_core,
        ethernet_config);

    ////////////////////////////////////////////////////////////////////////////
    //                      Compile and Execute Application
    ////////////////////////////////////////////////////////////////////////////

    auto inputs = generate_uniform_random_vector<uint32_t>(0, 100, byte_size / sizeof(uint32_t));
    fixture->WriteBuffer(device, input_dram_buffer, inputs);

    // Clear expected value at ethernet L1 address
    std::vector<uint32_t> all_zeros(inputs.size(), 0);
    llrt::write_hex_vec_to_core(device->id(), eth_noc_xy, all_zeros, eth_l1_byte_address);

    tt_metal::SetRuntimeArgs(
        program,
        eth_reader_kernel,
        eth_reader_core,
        {
            (uint32_t)dram_byte_address,
            0,
            (uint32_t)byte_size,
            (uint32_t)eth_l1_byte_address,
        });

    fixture->RunProgram(device, program);

    auto readback_vec = llrt::read_hex_vec_from_core(device->id(), eth_noc_xy, eth_l1_byte_address, byte_size);
    pass &= (readback_vec == inputs);
    if (not pass) {
        std::cout << "Mismatch at Core: " << eth_noc_xy.str() << std::endl;
    }
    return pass;
}

bool writer_kernel_no_receive(
    DispatchFixture* fixture,
    tt_metal::Device* device,
    const size_t& byte_size,
    const size_t& eth_l1_byte_address,
    const CoreCoord& eth_writer_core,
    const tt_metal::EthernetConfig& ethernet_config = tt_metal::EthernetConfig{.noc = tt_metal::NOC::NOC_0}) {
    bool pass = true;
    ////////////////////////////////////////////////////////////////////////////
    //                      Application Setup
    ////////////////////////////////////////////////////////////////////////////
    tt_metal::Program program = tt_metal::Program();

    tt::tt_metal::InterleavedBufferConfig dram_config{
        .device = device, .size = byte_size, .page_size = byte_size, .buffer_type = tt::tt_metal::BufferType::DRAM};

    auto output_dram_buffer = CreateBuffer(dram_config);
    uint32_t dram_byte_address = output_dram_buffer->address();
    auto eth_noc_xy = device->ethernet_core_from_logical_core(eth_writer_core);
    log_debug(
        tt::LogTest,
        "Device {}: writing {} bytes from ethernet core {} addr {} to dram bank 0 addr {}",
        device->id(),
        byte_size,
        eth_writer_core.str(),
        eth_l1_byte_address,
        dram_byte_address);

    auto eth_writer_kernel = tt_metal::CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/unit_tests/erisc/direct_writer_l1_to_dram.cpp",
        eth_writer_core,
        ethernet_config);

    ////////////////////////////////////////////////////////////////////////////
    //                      Compile and Execute Application
    ////////////////////////////////////////////////////////////////////////////

    auto inputs = generate_uniform_random_vector<uint32_t>(0, 100, byte_size / sizeof(uint32_t));
    llrt::write_hex_vec_to_core(device->id(), eth_noc_xy, inputs, eth_l1_byte_address);

    // Clear expected value at ethernet L1 address
    std::vector<uint32_t> all_zeros(inputs.size(), 0);
    fixture->WriteBuffer(device, output_dram_buffer, all_zeros);

    tt_metal::SetRuntimeArgs(
        program,
        eth_writer_kernel,
        eth_writer_core,
        {
            (uint32_t)dram_byte_address,
            0,
            (uint32_t)byte_size,
            (uint32_t)eth_l1_byte_address,
        });

    fixture->RunProgram(device, program);

    std::vector<uint32_t> readback_vec;
    fixture->ReadBuffer(device, output_dram_buffer, readback_vec);
    pass &= (readback_vec == inputs);
    if (not pass) {
        std::cout << "Mismatch" << std::endl;
    }
    return pass;
}

bool noc_reader_and_writer_kernels(
    tt_metal::Device* device,
    const uint32_t byte_size,
    const uint32_t eth_dst_l1_address,
    const uint32_t eth_src_l1_address,
    const CoreCoord& logical_eth_core,
    const tt_metal::EthernetConfig& reader_eth_config,
    const tt_metal::EthernetConfig& writer_eth_config) {
    bool pass = true;

    tt_metal::Program program = tt_metal::Program();

    tt_metal::InterleavedBufferConfig dram_config{
        .device = device, .size = byte_size, .page_size = byte_size, .buffer_type = tt_metal::BufferType::DRAM};

    auto reader_dram_buffer = CreateBuffer(dram_config);
    auto writer_dram_buffer = CreateBuffer(dram_config);

    log_debug(
        tt::LogTest,
        "Device {}: reading {} bytes from dram bank 0 addr {} to ethernet core {} addr {}",
        device->id(),
        byte_size,
        reader_dram_buffer->address(),
        logical_eth_core.str(),
        eth_dst_l1_address);
    log_debug(
        tt::LogTest,
        "Device {}: writing {} bytes from ethernet core {} addr {} to dram bank 0 addr {}",
        device->id(),
        byte_size,
        logical_eth_core.str(),
        eth_src_l1_address,
        writer_dram_buffer->address());

    auto eth_noc_xy = device->ethernet_core_from_logical_core(logical_eth_core);

    auto eth_reader_kernel = tt_metal::CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/unit_tests/erisc/direct_reader_dram_to_l1.cpp",
        logical_eth_core,
        reader_eth_config);

    tt_metal::SetRuntimeArgs(
        program,
        eth_reader_kernel,
        logical_eth_core,
        {
            (uint32_t)reader_dram_buffer->address(),
            0,
            (uint32_t)byte_size,
            (uint32_t)eth_dst_l1_address,
        });

    auto eth_writer_kernel = tt_metal::CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/unit_tests/erisc/direct_writer_l1_to_dram.cpp",
        logical_eth_core,
        writer_eth_config);

    tt_metal::SetRuntimeArgs(
        program,
        eth_writer_kernel,
        logical_eth_core,
        {
            (uint32_t)writer_dram_buffer->address(),
            0,
            (uint32_t)byte_size,
            (uint32_t)eth_src_l1_address,
        });

    auto reader_inputs = generate_uniform_random_vector<uint32_t>(0, 100, byte_size / sizeof(uint32_t));
    tt_metal::detail::WriteToBuffer(reader_dram_buffer, reader_inputs);

    auto writer_inputs = generate_uniform_random_vector<uint32_t>(0, 100, byte_size / sizeof(uint32_t));
    llrt::write_hex_vec_to_core(device->id(), eth_noc_xy, writer_inputs, eth_src_l1_address);

    // Clear expected values at output locations
    std::vector<uint32_t> all_zeros(byte_size / sizeof(uint32_t), 0);
    llrt::write_hex_vec_to_core(device->id(), eth_noc_xy, all_zeros, eth_dst_l1_address);
    tt_metal::detail::WriteToBuffer(writer_dram_buffer, all_zeros);

    tt_metal::detail::LaunchProgram(device, program);

    auto eth_readback_vec = llrt::read_hex_vec_from_core(device->id(), eth_noc_xy, eth_dst_l1_address, byte_size);
    pass &= (eth_readback_vec == reader_inputs);
    if (not pass) {
        log_info(
            tt::LogTest,
            "Mismatch at eth core: {}, eth kernel read incorrect values from DRAM",
            logical_eth_core.str());
    }
    std::vector<uint32_t> dram_readback_vec;
    tt_metal::detail::ReadFromBuffer(writer_dram_buffer, dram_readback_vec);
    pass &= (dram_readback_vec == writer_inputs);
    if (not pass) {
        log_info(
            tt::LogTest, "Mismatch at eth core: {}, eth kernel wrote incorrect values to DRAM", logical_eth_core.str());
    }

    return pass;
}

}  // namespace unit_tests::erisc::kernels

TEST_F(CommandQueueSingleCardProgramFixture, ActiveEthKernelsNocReadNoSend) {
    using namespace CMAKE_UNIQUE_NAMESPACE;
    const size_t src_eth_l1_byte_address = eth_l1_mem::address_map::ERISC_L1_UNRESERVED_BASE;

    for (const auto& device : devices_) {
        for (const auto& eth_core : device->get_active_ethernet_cores(true)) {
            ASSERT_TRUE(unit_tests::erisc::kernels::reader_kernel_no_send(
                static_cast<DispatchFixture*>(this), device, WORD_SIZE, src_eth_l1_byte_address, eth_core));
            ASSERT_TRUE(unit_tests::erisc::kernels::reader_kernel_no_send(
                static_cast<DispatchFixture*>(this), device, WORD_SIZE * 1024, src_eth_l1_byte_address, eth_core));
            ASSERT_TRUE(unit_tests::erisc::kernels::reader_kernel_no_send(
                static_cast<DispatchFixture*>(this), device, WORD_SIZE * 2048, src_eth_l1_byte_address, eth_core));
        }
    }
}

TEST_F(CommandQueueSingleCardProgramFixture, ActiveEthKernelsNocWriteNoReceive) {
    using namespace CMAKE_UNIQUE_NAMESPACE;
    const size_t src_eth_l1_byte_address = eth_l1_mem::address_map::ERISC_L1_UNRESERVED_BASE;

    for (const auto& device : devices_) {
        for (const auto& eth_core : device->get_active_ethernet_cores(true)) {
            ASSERT_TRUE(unit_tests::erisc::kernels::writer_kernel_no_receive(
                static_cast<DispatchFixture*>(this), device, WORD_SIZE, src_eth_l1_byte_address, eth_core));
            ASSERT_TRUE(unit_tests::erisc::kernels::writer_kernel_no_receive(
                static_cast<DispatchFixture*>(this), device, WORD_SIZE * 1024, src_eth_l1_byte_address, eth_core));
            ASSERT_TRUE(unit_tests::erisc::kernels::writer_kernel_no_receive(
                static_cast<DispatchFixture*>(this), device, WORD_SIZE * 2048, src_eth_l1_byte_address, eth_core));
        }
    }
}

TEST_F(N300DeviceFixture, ActiveEthKernelsNocReadNoSend) {
    using namespace CMAKE_UNIQUE_NAMESPACE;
    GTEST_SKIP();
    const auto& device_0 = devices_.at(0);
    const auto& device_1 = devices_.at(1);

    const size_t src_eth_l1_byte_address = eth_l1_mem::address_map::ERISC_L1_UNRESERVED_BASE;

    for (const auto& eth_core : device_0->get_active_ethernet_cores(true)) {
        ASSERT_TRUE(unit_tests::erisc::kernels::reader_kernel_no_send(
            static_cast<DispatchFixture*>(this), device_0, WORD_SIZE, src_eth_l1_byte_address, eth_core));
        ASSERT_TRUE(unit_tests::erisc::kernels::reader_kernel_no_send(
            static_cast<DispatchFixture*>(this), device_0, WORD_SIZE * 1024, src_eth_l1_byte_address, eth_core));
        ASSERT_TRUE(unit_tests::erisc::kernels::reader_kernel_no_send(
            static_cast<DispatchFixture*>(this), device_0, WORD_SIZE * 2048, src_eth_l1_byte_address, eth_core));
    }

    for (const auto& eth_core : device_1->get_active_ethernet_cores(true)) {
        ASSERT_TRUE(unit_tests::erisc::kernels::reader_kernel_no_send(
            static_cast<DispatchFixture*>(this), device_1, WORD_SIZE, src_eth_l1_byte_address, eth_core));
        ASSERT_TRUE(unit_tests::erisc::kernels::reader_kernel_no_send(
            static_cast<DispatchFixture*>(this), device_1, WORD_SIZE * 1024, src_eth_l1_byte_address, eth_core));
        ASSERT_TRUE(unit_tests::erisc::kernels::reader_kernel_no_send(
            static_cast<DispatchFixture*>(this), device_1, WORD_SIZE * 2048, src_eth_l1_byte_address, eth_core));
    }
}

TEST_F(N300DeviceFixture, ActiveEthKernelsNocWriteNoReceive) {
    using namespace CMAKE_UNIQUE_NAMESPACE;
    GTEST_SKIP();
    const auto& device_0 = devices_.at(0);
    const auto& device_1 = devices_.at(1);

    const size_t src_eth_l1_byte_address = eth_l1_mem::address_map::ERISC_L1_UNRESERVED_BASE;

    for (const auto& eth_core : device_0->get_active_ethernet_cores(true)) {
        ASSERT_TRUE(unit_tests::erisc::kernels::writer_kernel_no_receive(
            static_cast<DispatchFixture*>(this), device_0, WORD_SIZE, src_eth_l1_byte_address, eth_core));
        ASSERT_TRUE(unit_tests::erisc::kernels::writer_kernel_no_receive(
            static_cast<DispatchFixture*>(this), device_0, WORD_SIZE * 1024, src_eth_l1_byte_address, eth_core));
        ASSERT_TRUE(unit_tests::erisc::kernels::writer_kernel_no_receive(
            static_cast<DispatchFixture*>(this), device_0, WORD_SIZE * 2048, src_eth_l1_byte_address, eth_core));
    }

    for (const auto& eth_core : device_1->get_active_ethernet_cores(true)) {
        ASSERT_TRUE(unit_tests::erisc::kernels::writer_kernel_no_receive(
            static_cast<DispatchFixture*>(this), device_1, WORD_SIZE, src_eth_l1_byte_address, eth_core));
        ASSERT_TRUE(unit_tests::erisc::kernels::writer_kernel_no_receive(
            static_cast<DispatchFixture*>(this), device_1, WORD_SIZE * 1024, src_eth_l1_byte_address, eth_core));
        ASSERT_TRUE(unit_tests::erisc::kernels::writer_kernel_no_receive(
            static_cast<DispatchFixture*>(this), device_1, WORD_SIZE * 2048, src_eth_l1_byte_address, eth_core));
    }
}

/*
 *
 *                                         ███████╗████████╗██╗░░██╗
 *                                         ██╔════╝╚══██╔══╝██║░░██║
 *                                         █████╗░░░░░██║░░░███████║
 *                                         ██╔══╝░░░░░██║░░░██╔══██║
 *                                         ███████╗░░░██║░░░██║░░██║
 *                                         ╚══════╝░░░╚═╝░░░╚═╝░░╚═╝
 */

// TODO #14640: Run this on WH when i$ flush issue is addressed
TEST_F(BlackholeSingleCardFixture, IdleEthKernelOnIdleErisc0) {
    using namespace CMAKE_UNIQUE_NAMESPACE;
    uint32_t eth_l1_address = hal.get_dev_addr(HalProgrammableCoreType::IDLE_ETH, HalL1MemAddrType::UNRESERVED);
    tt_metal::EthernetConfig noc0_ethernet_config{
        .eth_mode = Eth::IDLE, .noc = tt_metal::NOC::NOC_0, .processor = tt_metal::DataMovementProcessor::RISCV_0};
    tt_metal::EthernetConfig noc1_ethernet_config{
        .eth_mode = Eth::IDLE, .noc = tt_metal::NOC::NOC_1, .processor = tt_metal::DataMovementProcessor::RISCV_0};

    for (const auto& eth_core : device_->get_inactive_ethernet_cores()) {
        ASSERT_TRUE(unit_tests::erisc::kernels::reader_kernel_no_send(
            static_cast<DispatchFixture*>(this),
            device_,
            WORD_SIZE * 2048,
            eth_l1_address,
            eth_core,
            noc0_ethernet_config));
        ASSERT_TRUE(unit_tests::erisc::kernels::reader_kernel_no_send(
            static_cast<DispatchFixture*>(this),
            device_,
            WORD_SIZE * 2048,
            eth_l1_address,
            eth_core,
            noc1_ethernet_config));
        ASSERT_TRUE(unit_tests::erisc::kernels::writer_kernel_no_receive(
            static_cast<DispatchFixture*>(this),
            device_,
            WORD_SIZE * 2048,
            eth_l1_address,
            eth_core,
            noc0_ethernet_config));
        ASSERT_TRUE(unit_tests::erisc::kernels::writer_kernel_no_receive(
            static_cast<DispatchFixture*>(this),
            device_,
            WORD_SIZE * 2048,
            eth_l1_address,
            eth_core,
            noc1_ethernet_config));
    }
}

TEST_F(BlackholeSingleCardFixture, IdleEthKernelOnIdleErisc1) {
    using namespace CMAKE_UNIQUE_NAMESPACE;
    uint32_t eth_l1_address = hal.get_dev_addr(HalProgrammableCoreType::IDLE_ETH, HalL1MemAddrType::UNRESERVED);
    tt_metal::EthernetConfig noc0_ethernet_config{
        .eth_mode = Eth::IDLE, .noc = tt_metal::NOC::NOC_0, .processor = tt_metal::DataMovementProcessor::RISCV_1};
    tt_metal::EthernetConfig noc1_ethernet_config{
        .eth_mode = Eth::IDLE, .noc = tt_metal::NOC::NOC_1, .processor = tt_metal::DataMovementProcessor::RISCV_1};

    for (const auto& eth_core : device_->get_inactive_ethernet_cores()) {
        ASSERT_TRUE(unit_tests::erisc::kernels::reader_kernel_no_send(
            static_cast<DispatchFixture*>(this),
            device_,
            WORD_SIZE * 2048,
            eth_l1_address,
            eth_core,
            noc0_ethernet_config));
        ASSERT_TRUE(unit_tests::erisc::kernels::reader_kernel_no_send(
            static_cast<DispatchFixture*>(this),
            device_,
            WORD_SIZE * 2048,
            eth_l1_address,
            eth_core,
            noc1_ethernet_config));
        ASSERT_TRUE(unit_tests::erisc::kernels::writer_kernel_no_receive(
            static_cast<DispatchFixture*>(this),
            device_,
            WORD_SIZE * 2048,
            eth_l1_address,
            eth_core,
            noc0_ethernet_config));
        ASSERT_TRUE(unit_tests::erisc::kernels::writer_kernel_no_receive(
            static_cast<DispatchFixture*>(this),
            device_,
            WORD_SIZE * 2048,
            eth_l1_address,
            eth_core,
            noc1_ethernet_config));
    }
}

TEST_F(BlackholeSingleCardFixture, IdleEthKernelOnBothIdleEriscs) {
    using namespace CMAKE_UNIQUE_NAMESPACE;
    uint32_t read_write_size_bytes = WORD_SIZE * 2048;
    uint32_t reader_dst_address = hal.get_dev_addr(HalProgrammableCoreType::IDLE_ETH, HalL1MemAddrType::UNRESERVED);
    uint32_t writer_src_address = reader_dst_address + read_write_size_bytes;
    tt_metal::EthernetConfig erisc0_ethernet_config{
        .eth_mode = Eth::IDLE, .noc = tt_metal::NOC::NOC_0, .processor = tt_metal::DataMovementProcessor::RISCV_0};
    tt_metal::EthernetConfig erisc1_ethernet_config{
        .eth_mode = Eth::IDLE, .noc = tt_metal::NOC::NOC_0, .processor = tt_metal::DataMovementProcessor::RISCV_1};

    for (const auto& eth_core : device_->get_inactive_ethernet_cores()) {
        ASSERT_TRUE(unit_tests::erisc::kernels::noc_reader_and_writer_kernels(
            device_,
            read_write_size_bytes,
            reader_dst_address,
            writer_src_address,
            eth_core,
            erisc0_ethernet_config,
            erisc1_ethernet_config));
        erisc0_ethernet_config.noc = tt_metal::NOC::NOC_1;
        erisc1_ethernet_config.noc = tt_metal::NOC::NOC_1;
        ASSERT_TRUE(unit_tests::erisc::kernels::noc_reader_and_writer_kernels(
            device_,
            read_write_size_bytes,
            reader_dst_address,
            writer_src_address,
            eth_core,
            erisc0_ethernet_config,
            erisc1_ethernet_config));
    }
}
