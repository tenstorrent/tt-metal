// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0


#include <gtest/gtest.h>

#include <algorithm>
#include <functional>
#include <random>

#include "device_fixture.hpp"
#include "n300_device_fixture.hpp"
#include "tt_metal/detail/tt_metal.hpp"
#include "tt_metal/host_api.hpp"
#include "tt_metal/hostdevcommon/common_runtime_address_map.h"  // FIXME: Should remove dependency on this
#include "tt_metal/impl/kernels/kernel.hpp"
#include "tt_metal/test_utils/comparison.hpp"
#include "tt_metal/test_utils/df/df.hpp"
#include "tt_metal/test_utils/print_helpers.hpp"
#include "tt_metal/test_utils/stimulus.hpp"

using namespace tt;
using namespace tt::test_utils;
using namespace tt::test_utils::df;

namespace unit_tests::erisc::kernels {
// Read from chip's worker core or dram L1 to ethernet L1 and check correctness
// Ethernet does not send
bool reader_kernel_no_send(
    tt_metal::Device* device,
    const size_t& byte_size,
    const size_t& eth_l1_byte_address,
    const CoreCoord& eth_reader_core) {
    bool pass = true;
    ////////////////////////////////////////////////////////////////////////////
    //                      Application Setup
    ////////////////////////////////////////////////////////////////////////////
    tt_metal::Program program = tt_metal::Program();

    auto input_dram_buffer = CreateBuffer(device, byte_size, byte_size, tt_metal::BufferType::DRAM);
    uint32_t dram_byte_address = input_dram_buffer.address();
    auto dram_noc_xy = input_dram_buffer.noc_coordinates();
    log_info(
        tt::LogTest,
        "Reading from noc {} addr {} to ethernet core {} addr {}",
        dram_noc_xy.str(),
        dram_byte_address,
        eth_reader_core.str(),
        eth_l1_byte_address);
    auto eth_reader_kernel = tt_metal::CreateKernel(
        program,
        "tt_metal/kernels/dataflow/unit_tests/erisc/direct_reader_dram_to_l1.cpp",
        eth_reader_core,
        tt_metal::EthernetConfig{.eth_mode = tt_metal::Eth::SENDER, .noc = tt_metal::NOC::NOC_0});

    ////////////////////////////////////////////////////////////////////////////
    //                      Compile and Execute Application
    ////////////////////////////////////////////////////////////////////////////

    auto inputs = generate_uniform_random_vector<uint32_t>(0, 100, byte_size / sizeof(uint32_t));
    tt_metal::detail::WriteToBuffer(input_dram_buffer, inputs);

    // Clear expected value at ethernet L1 address
    std::vector<uint32_t> all_zeros(inputs.size(), 0);
    llrt::write_hex_vec_to_core(device->id(), eth_reader_core, all_zeros, eth_l1_byte_address);

    tt_metal::SetRuntimeArgs(
        program,
        eth_reader_kernel,
        eth_reader_core,
        {
            (uint32_t)dram_byte_address,
            (uint32_t)dram_noc_xy.x,
            (uint32_t)dram_noc_xy.y,
            (uint32_t)byte_size,
            (uint32_t)eth_l1_byte_address,
        });

    tt_metal::detail::LaunchProgram(device, program);

    std::vector<uint32_t> readback_vec;
    // tt_metal::ReadFromBuffer(l1_buffer, dest_core_data);
    tt_metal::detail::ReadFromDeviceL1(device, eth_reader_core, eth_l1_byte_address, byte_size, readback_vec);
    pass &= (readback_vec == inputs);
    for (const auto& v : readback_vec) {
        std::cout << v << std::endl;
    }
    if (not pass) {
        std::cout << "Mismatch at Core: " << eth_reader_core.str() << std::endl;
    }
    return pass;
}

bool eth_direct_sender_receiver_kernels(
    tt_metal::Device* sender_device,
    tt_metal::Device* receiver_device,
    const size_t& byte_size,
    const size_t& eth_l1_byte_address,
    const CoreCoord& eth_sender_core,
    const CoreCoord& eth_receiver_core) {
    bool pass = true;
    log_debug(
        tt::LogTest,
        "Sending {} bytes from device {} eth core {} addr {} to device {} eth core {} addr {}",
        byte_size,
        sender_device->id(),
        eth_sender_core.str(),
        eth_l1_byte_address,
        receiver_device->id(),
        eth_receiver_core.str(),
        eth_l1_byte_address);

    // Generate inputs
    auto inputs = generate_uniform_random_vector<uint32_t>(0, 100, byte_size / sizeof(uint32_t));
    llrt::write_hex_vec_to_core(
        sender_device->id(),
        sender_device->ethernet_core_from_logical_core(eth_sender_core),
        inputs,
        eth_l1_byte_address);

    // Clear expected value at ethernet L1 address
    std::vector<uint32_t> all_zeros(inputs.size(), 0);
    llrt::write_hex_vec_to_core(
        receiver_device->id(),
        receiver_device->ethernet_core_from_logical_core(eth_receiver_core),
        all_zeros,
        eth_l1_byte_address);

    ////////////////////////////////////////////////////////////////////////////
    //                      Sender Device
    ////////////////////////////////////////////////////////////////////////////
    tt_metal::Program sender_program = tt_metal::Program();

    auto eth_sender_kernel = tt_metal::CreateKernel(
        sender_program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/unit_tests/erisc/eth_l1_direct_send.cpp",
        eth_sender_core,
        tt_metal::EthernetConfig{.eth_mode = tt_metal::Eth::SENDER, .noc = tt_metal::NOC::NOC_0});

    tt_metal::SetRuntimeArgs(
        sender_program,
        eth_sender_kernel,
        eth_sender_core,
        {
            (uint32_t)eth_l1_byte_address,
            (uint32_t)eth_l1_byte_address,
            (uint32_t)byte_size,
        });

    ////////////////////////////////////////////////////////////////////////////
    //                      Receiver Device
    ////////////////////////////////////////////////////////////////////////////
    tt_metal::Program receiver_program = tt_metal::Program();

    auto eth_receiver_kernel = tt_metal::CreateKernel(
        receiver_program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/unit_tests/erisc/eth_l1_direct_receive.cpp",
        eth_receiver_core,
        tt_metal::EthernetConfig{
            .eth_mode = tt_metal::Eth::RECEIVER, .noc = tt_metal::NOC::NOC_0});  // probably want to use NOC_1 here

    tt_metal::SetRuntimeArgs(
        receiver_program,
        eth_receiver_kernel,
        eth_receiver_core,
        {
            (uint32_t)byte_size,
        });

    ////////////////////////////////////////////////////////////////////////////
    //                      Execute Programs
    ////////////////////////////////////////////////////////////////////////////

    std::thread th1 = std::thread([&] {
        // TODO: needed to ensure binaries get sent to remote chip first
        // will be removed when context switching is added to Task 2
        if (sender_device->id() == 0) {
            sleep(1);
        }
        tt_metal::detail::LaunchProgram(sender_device, sender_program);
    });
    std::thread th2 = std::thread([&] {
        if (receiver_device->id() == 0) {
            sleep(1);
        }
        tt_metal::detail::LaunchProgram(receiver_device, receiver_program);
    });

    th1.join();
    th2.join();
    // tt_metal::ReadFromBuffer(l1_buffer, dest_core_data);
    auto readback_vec = llrt::read_hex_vec_from_core(
        receiver_device->id(),
        receiver_device->ethernet_core_from_logical_core(eth_receiver_core),
        eth_l1_byte_address,
        byte_size);
    pass &= (readback_vec == inputs);
    if (not pass) {
        std::cout << "Mismatch at Core: " << eth_receiver_core.str() << std::endl;
        std::cout << readback_vec[0] << std::endl;
    }
    return pass;
}

bool eth_scatter_sender_receiver_kernels(
    tt_metal::Device* sender_device,
    tt_metal::Device* receiver_device,
    const size_t& byte_size,
    const size_t& src_eth_l1_byte_addr,
    const std::vector<size_t>& dst_eth_l1_byte_addrs,
    const CoreCoord& eth_sender_core,
    const CoreCoord& eth_receiver_core) {
    TT_ASSERT(dst_eth_l1_byte_addrs.size() == 4);
    bool pass = true;
    log_debug(
        tt::LogTest,
        "Sending {} bytes from device {} eth core {} addr {} to device {} eth core {} multiple addresses",
        byte_size,
        sender_device->id(),
        eth_sender_core.str(),
        src_eth_l1_byte_addr,
        receiver_device->id(),
        eth_receiver_core.str());

    // Generate inputs
    auto inputs = generate_uniform_random_vector<uint32_t>(0, 100, byte_size / sizeof(uint32_t));
    llrt::write_hex_vec_to_core(
        sender_device->id(),
        sender_device->ethernet_core_from_logical_core(eth_sender_core),
        inputs,
        src_eth_l1_byte_addr);

    // Clear expected value at ethernet L1 address
    std::vector<uint32_t> all_zeros(inputs.size(), 0);
    for (const auto& dst_addr : dst_eth_l1_byte_addrs) {
        llrt::write_hex_vec_to_core(
            receiver_device->id(),
            receiver_device->ethernet_core_from_logical_core(eth_receiver_core),
            all_zeros,
            dst_addr);
    }

    ////////////////////////////////////////////////////////////////////////////
    //                      Sender Device
    ////////////////////////////////////////////////////////////////////////////
    tt_metal::Program sender_program = tt_metal::Program();

    auto eth_sender_kernel = tt_metal::CreateKernel(
        sender_program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/unit_tests/erisc/eth_l1_scatter_send.cpp",
        eth_sender_core,
        tt_metal::EthernetConfig{.eth_mode = tt_metal::Eth::SENDER, .noc = tt_metal::NOC::NOC_0});

    tt_metal::SetRuntimeArgs(
        sender_program,
        eth_sender_kernel,
        eth_sender_core,
        {
            (uint32_t)src_eth_l1_byte_addr,
            (uint32_t)dst_eth_l1_byte_addrs.at(0),
            (uint32_t)dst_eth_l1_byte_addrs.at(1),
            (uint32_t)dst_eth_l1_byte_addrs.at(2),
            (uint32_t)dst_eth_l1_byte_addrs.at(3),
            (uint32_t)byte_size,  // per transfer size
        });

    ////////////////////////////////////////////////////////////////////////////
    //                      Receiver Device
    ////////////////////////////////////////////////////////////////////////////
    tt_metal::Program receiver_program = tt_metal::Program();

    auto eth_receiver_kernel = tt_metal::CreateKernel(
        receiver_program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/unit_tests/erisc/eth_l1_direct_receive.cpp",
        eth_receiver_core,
        tt_metal::EthernetConfig{
            .eth_mode = tt_metal::Eth::RECEIVER, .noc = tt_metal::NOC::NOC_0});  // probably want to use NOC_1 here

    tt_metal::SetRuntimeArgs(
        receiver_program,
        eth_receiver_kernel,
        eth_receiver_core,
        {
            (uint32_t)(byte_size * 4),  // total transfer size
        });

    ////////////////////////////////////////////////////////////////////////////
    //                      Execute Programs
    ////////////////////////////////////////////////////////////////////////////

    std::thread th1 = std::thread([&] {
        if (sender_device->id() == 0) {
            sleep(1);
        }
        tt_metal::detail::LaunchProgram(sender_device, sender_program);
    });
    std::thread th2 = std::thread([&] {
        if (receiver_device->id() == 0) {
            sleep(1);
        }
        tt_metal::detail::LaunchProgram(receiver_device, receiver_program);
    });

    th1.join();
    th2.join();
    // tt_metal::ReadFromBuffer(l1_buffer, dest_core_data);
    for (const auto& dst_addr : dst_eth_l1_byte_addrs) {
        auto readback_vec = llrt::read_hex_vec_from_core(
            receiver_device->id(),
            receiver_device->ethernet_core_from_logical_core(eth_receiver_core),
            dst_addr,
            byte_size);
        pass &= (readback_vec == inputs);
    }
    if (not pass) {
        std::cout << "Mismatch at Core: " << eth_receiver_core.str() << std::endl;
    }
    return pass;
}

bool eth_hung_kernels(
    tt_metal::Device* hung_device, tt_metal::Device* remote_device, const std::vector<CoreCoord>& hung_cores) {
    bool pass = true;

    ////////////////////////////////////////////////////////////////////////////
    //                     Load Program to Hang Device
    ////////////////////////////////////////////////////////////////////////////
    tt_metal::Program program = tt_metal::Program();

    auto eth_kernel_0 = tt_metal::CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/unit_tests/erisc/hung_kernel.cpp",
        hung_cores[0],
        tt_metal::EthernetConfig{.eth_mode = tt_metal::Eth::RECEIVER, .noc = tt_metal::NOC::NOC_0});
    // Runtime arg at 0 will be used to kill kernel

    auto eth_kernel_1 = tt_metal::CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/unit_tests/erisc/hung_kernel.cpp",
        hung_cores[1],
        tt_metal::EthernetConfig{.eth_mode = tt_metal::Eth::RECEIVER, .noc = tt_metal::NOC::NOC_0});

    std::thread th1 = std::thread([&] {
        tt_metal::detail::LaunchProgram(hung_device, program);
        sleep(4);
        for (const auto& core : hung_cores) {
            llrt::write_hex_vec_to_core(
                hung_device->id(),
                hung_device->ethernet_core_from_logical_core(core),
                {0x1},
                eth_l1_mem::address_map::ERISC_L1_ARG_BASE);
        }
    });

    ////////////////////////////////////////////////////////////////////////////
    //                      Execute Remote Transfers
    ////////////////////////////////////////////////////////////////////////////
    std::thread th2 = std::thread([&] {
        sleep(2);
        // Generate inputs
        uint32_t target_addr = L1_UNRESERVED_BASE;
        CoreCoord target_core = {0, 0};
        uint32_t byte_size = 2048;
        auto inputs = generate_uniform_random_vector<uint32_t>(0, 100, byte_size / sizeof(uint32_t));
        // Clear expected value at ethernet L1 address
        std::vector<uint32_t> all_zeros(inputs.size(), 0);
        llrt::write_hex_vec_to_core(remote_device->id(), target_core, all_zeros, target_addr);

        // Send data
        llrt::write_hex_vec_to_core(remote_device->id(), {0, 0}, inputs, target_addr);

        auto readback_vec = llrt::read_hex_vec_from_core(remote_device->id(), target_core, target_addr, byte_size);
        pass &= (readback_vec == inputs);
    });

    th2.join();
    th1.join();
    std::cout << " done kernel test" << std::endl;
    if (not pass) {
        std::cout << "Mismatched data when ethernet cores are context switching" << std::endl;
    }
    return pass;
}
}  // namespace unit_tests::erisc::kernels

TEST_F(N300DeviceFixture, EthKernelsDirectSendChip0ToChip1) {
    const auto& device_0 = devices_.at(0);
    const auto& device_1 = devices_.at(1);
    CoreCoord sender_core_0 = {.x = 2, .y = 0};
    CoreCoord sender_core_1 = {.x = 2, .y = 1};

    CoreCoord receiver_core_0 = {.x = 0, .y = 0};
    CoreCoord receiver_core_1 = {.x = 0, .y = 1};

    ASSERT_TRUE(unit_tests::erisc::kernels::eth_direct_sender_receiver_kernels(
        device_0, device_1, 16, eth_l1_mem::address_map::ERISC_L1_UNRESERVED_BASE, sender_core_0, receiver_core_0));
    ASSERT_TRUE(unit_tests::erisc::kernels::eth_direct_sender_receiver_kernels(
        device_0, device_1, 4 * 16, eth_l1_mem::address_map::ERISC_L1_UNRESERVED_BASE, sender_core_0, receiver_core_0));
    ASSERT_TRUE(unit_tests::erisc::kernels::eth_direct_sender_receiver_kernels(
        device_0,
        device_1,
        256 * 16,
        eth_l1_mem::address_map::ERISC_L1_UNRESERVED_BASE,
        sender_core_0,
        receiver_core_0));
    ASSERT_TRUE(unit_tests::erisc::kernels::eth_direct_sender_receiver_kernels(
        device_0,
        device_1,
        1000 * 16,
        eth_l1_mem::address_map::ERISC_L1_UNRESERVED_BASE,
        sender_core_0,
        receiver_core_0));
    ASSERT_TRUE(unit_tests::erisc::kernels::eth_direct_sender_receiver_kernels(
        device_0, device_1, 16, eth_l1_mem::address_map::ERISC_L1_UNRESERVED_BASE, sender_core_1, receiver_core_1));
    ASSERT_TRUE(unit_tests::erisc::kernels::eth_direct_sender_receiver_kernels(
        device_0, device_1, 4 * 16, eth_l1_mem::address_map::ERISC_L1_UNRESERVED_BASE, sender_core_1, receiver_core_1));
    ASSERT_TRUE(unit_tests::erisc::kernels::eth_direct_sender_receiver_kernels(
        device_0,
        device_1,
        256 * 16,
        eth_l1_mem::address_map::ERISC_L1_UNRESERVED_BASE,
        sender_core_1,
        receiver_core_1));
    ASSERT_TRUE(unit_tests::erisc::kernels::eth_direct_sender_receiver_kernels(
        device_0,
        device_1,
        1000 * 16,
        eth_l1_mem::address_map::ERISC_L1_UNRESERVED_BASE,
        sender_core_1,
        receiver_core_1));
}

TEST_F(N300DeviceFixture, EthKernelsDirectSendChip1ToChip0) {
    const auto& device_0 = devices_.at(0);
    const auto& device_1 = devices_.at(1);
    CoreCoord sender_core_0 = {.x = 0, .y = 0};
    CoreCoord sender_core_1 = {.x = 0, .y = 1};

    CoreCoord receiver_core_0 = {.x = 2, .y = 0};
    CoreCoord receiver_core_1 = {.x = 2, .y = 1};

    ASSERT_TRUE(unit_tests::erisc::kernels::eth_direct_sender_receiver_kernels(
        device_1, device_0, 16, eth_l1_mem::address_map::ERISC_L1_UNRESERVED_BASE, sender_core_0, receiver_core_0));
    ASSERT_TRUE(unit_tests::erisc::kernels::eth_direct_sender_receiver_kernels(
        device_1, device_0, 4 * 16, eth_l1_mem::address_map::ERISC_L1_UNRESERVED_BASE, sender_core_0, receiver_core_0));
    ASSERT_TRUE(unit_tests::erisc::kernels::eth_direct_sender_receiver_kernels(
        device_1,
        device_0,
        256 * 16,
        eth_l1_mem::address_map::ERISC_L1_UNRESERVED_BASE,
        sender_core_0,
        receiver_core_0));
    ASSERT_TRUE(unit_tests::erisc::kernels::eth_direct_sender_receiver_kernels(
        device_1,
        device_0,
        1000 * 16,
        eth_l1_mem::address_map::ERISC_L1_UNRESERVED_BASE,
        sender_core_0,
        receiver_core_0));
    ASSERT_TRUE(unit_tests::erisc::kernels::eth_direct_sender_receiver_kernels(
        device_1, device_0, 16, eth_l1_mem::address_map::ERISC_L1_UNRESERVED_BASE, sender_core_1, receiver_core_1));
    ASSERT_TRUE(unit_tests::erisc::kernels::eth_direct_sender_receiver_kernels(
        device_1, device_0, 4 * 16, eth_l1_mem::address_map::ERISC_L1_UNRESERVED_BASE, sender_core_1, receiver_core_1));
    ASSERT_TRUE(unit_tests::erisc::kernels::eth_direct_sender_receiver_kernels(
        device_1,
        device_0,
        256 * 16,
        eth_l1_mem::address_map::ERISC_L1_UNRESERVED_BASE,
        sender_core_1,
        receiver_core_1));
    ASSERT_TRUE(unit_tests::erisc::kernels::eth_direct_sender_receiver_kernels(
        device_1,
        device_0,
        1000 * 16,
        eth_l1_mem::address_map::ERISC_L1_UNRESERVED_BASE,
        sender_core_1,
        receiver_core_1));
}

TEST_F(N300DeviceFixture, EthKernelsScatterSend) {
    GTEST_SKIP();
    const auto& device_0 = devices_.at(0);
    const auto& device_1 = devices_.at(1);
    CoreCoord sender_core_0 = {.x = 2, .y = 0};
    CoreCoord sender_core_1 = {.x = 2, .y = 1};

    CoreCoord receiver_core_0 = {.x = 0, .y = 0};
    CoreCoord receiver_core_1 = {.x = 0, .y = 1};

    std::size_t src_addr = eth_l1_mem::address_map::ERISC_L1_UNRESERVED_BASE;
    std::vector<size_t> dst_addrs = {
        eth_l1_mem::address_map::ERISC_L1_UNRESERVED_BASE,
        eth_l1_mem::address_map::ERISC_L1_UNRESERVED_BASE + 2500 * 16,
        eth_l1_mem::address_map::ERISC_L1_UNRESERVED_BASE + 2500 * 16 * 2,
        eth_l1_mem::address_map::ERISC_L1_UNRESERVED_BASE + 2500 * 16 * 3};

    // ASSERT_TRUE(
    //    unit_tests::erisc::kernels::eth_scatter_sender_receiver_kernels(device_1, device_0, 1 * 16, src_addr,
    //    dst_addrs, receiver_core_0, sender_core_0));
    ASSERT_TRUE(unit_tests::erisc::kernels::eth_scatter_sender_receiver_kernels(
        device_0, device_1, 1 * 16, src_addr, dst_addrs, sender_core_0, receiver_core_0));
    // ASSERT_TRUE(
    //   unit_tests::erisc::kernels::eth_scatter_sender_receiver_kernels(device_0, device_1, 2000 * 16, src_addr,
    //   dst_addrs, sender_core_0, receiver_core_0));
}

TEST_F(N300DeviceFixture, HungEthKernelsContextSwitch) {
    GTEST_SKIP();
    const auto& mmio_device = devices_.at(0);
    const auto& remote_device = devices_.at(1);
    std::vector<CoreCoord> hung_cores = {{.x = 2, .y = 0}, {.x = 2, .y = 1}};

    ASSERT_TRUE(unit_tests::erisc::kernels::eth_hung_kernels(mmio_device, remote_device, hung_cores));
}
