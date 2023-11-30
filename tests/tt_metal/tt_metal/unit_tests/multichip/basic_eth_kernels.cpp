// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0


#include <gtest/gtest.h>

#include <algorithm>
#include <functional>
#include <random>

#include "n300_device_fixture.hpp"
#include "tt_metal/detail/tt_metal.hpp"
#include "tt_metal/host_api.hpp"
#include "tt_metal/impl/kernels/kernel.hpp"
#include "tt_metal/test_utils/comparison.hpp"
#include "tt_metal/test_utils/df/df.hpp"
#include "tt_metal/test_utils/print_helpers.hpp"
#include "tt_metal/test_utils/stimulus.hpp"

using namespace tt;
using namespace tt::test_utils;
using namespace tt::test_utils::df;

constexpr std::int32_t WORD_SIZE = 16;  // 16 bytes per eth send packet
constexpr std::int32_t MAX_NUM_WORDS =
    (eth_l1_mem::address_map::MAX_L1_LOADING_SIZE - eth_l1_mem::address_map::ERISC_L1_UNRESERVED_BASE) / WORD_SIZE;

namespace unit_tests::erisc::kernels {

const size_t get_rand_32_byte_aligned_address(const size_t& base, const size_t& max) {
    TT_ASSERT(!(base & 0x1F) and !(max & 0x1F));
    size_t word_size = (max >> 5) - (base >> 5);
    return (((rand() % word_size) << 5) + base);
}

/*
 *                                         ███╗░░██╗░█████╗░░█████╗░
 *                                         ████╗░██║██╔══██╗██╔══██╗
 *                                         ██╔██╗██║██║░░██║██║░░╚═╝
 *                                         ██║╚████║██║░░██║██║░░██╗
 *                                         ██║░╚███║╚█████╔╝╚█████╔╝
 *                                         ╚═╝░░╚══╝░╚════╝░░╚════╝░
 */

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
    auto eth_noc_xy = device->ethernet_core_from_logical_core(eth_reader_core);
    log_debug(
        tt::LogTest,
        "Device {}: reading {} bytes from dram {} addr {} to ethernet core {} addr {}",
        device->id(),
        byte_size,
        dram_noc_xy.str(),
        dram_byte_address,
        eth_reader_core.str(),
        eth_l1_byte_address);

    auto eth_reader_kernel = tt_metal::CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/unit_tests/erisc/direct_reader_dram_to_l1.cpp",
        eth_reader_core,
        tt_metal::experimental::EthernetConfig{.eth_mode = tt_metal::Eth::SENDER, .noc = tt_metal::NOC::NOC_0});

    ////////////////////////////////////////////////////////////////////////////
    //                      Compile and Execute Application
    ////////////////////////////////////////////////////////////////////////////

    auto inputs = generate_uniform_random_vector<uint32_t>(0, 100, byte_size / sizeof(uint32_t));
    tt_metal::detail::WriteToBuffer(input_dram_buffer, inputs);

    // Clear expected value at ethernet L1 address
    std::vector<uint32_t> all_zeros(inputs.size(), 0);
    llrt::write_hex_vec_to_core(device->id(), eth_noc_xy, all_zeros, eth_l1_byte_address);

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

    auto readback_vec = llrt::read_hex_vec_from_core(device->id(), eth_noc_xy, eth_l1_byte_address, byte_size);
    pass &= (readback_vec == inputs);
    if (not pass) {
        std::cout << "Mismatch at Core: " << eth_noc_xy.str() << std::endl;
    }
    return pass;
}

bool writer_kernel_no_receive(
    tt_metal::Device* device,
    const size_t& byte_size,
    const size_t& eth_l1_byte_address,
    const CoreCoord& eth_writer_core) {
    bool pass = true;
    ////////////////////////////////////////////////////////////////////////////
    //                      Application Setup
    ////////////////////////////////////////////////////////////////////////////
    tt_metal::Program program = tt_metal::Program();

    auto output_dram_buffer = CreateBuffer(device, byte_size, byte_size, tt_metal::BufferType::DRAM);
    uint32_t dram_byte_address = output_dram_buffer.address();
    auto dram_noc_xy = output_dram_buffer.noc_coordinates();
    auto eth_noc_xy = device->ethernet_core_from_logical_core(eth_writer_core);
    log_debug(
        tt::LogTest,
        "Device {}: writing {} bytes from ethernet core {} addr {} to dram {} addr {}",
        device->id(),
        byte_size,
        eth_writer_core.str(),
        eth_l1_byte_address,
        dram_noc_xy.str(),
        dram_byte_address);

    auto eth_writer_kernel = tt_metal::CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/unit_tests/erisc/direct_writer_l1_to_dram.cpp",
        eth_writer_core,
        tt_metal::experimental::EthernetConfig{.eth_mode = tt_metal::Eth::SENDER, .noc = tt_metal::NOC::NOC_0});

    ////////////////////////////////////////////////////////////////////////////
    //                      Compile and Execute Application
    ////////////////////////////////////////////////////////////////////////////

    auto inputs = generate_uniform_random_vector<uint32_t>(0, 100, byte_size / sizeof(uint32_t));
    llrt::write_hex_vec_to_core(device->id(), eth_noc_xy, inputs, eth_l1_byte_address);

    // Clear expected value at ethernet L1 address
    std::vector<uint32_t> all_zeros(inputs.size(), 0);
    tt_metal::detail::WriteToBuffer(output_dram_buffer, all_zeros);

    tt_metal::SetRuntimeArgs(
        program,
        eth_writer_kernel,
        eth_writer_core,
        {
            (uint32_t)dram_byte_address,
            (uint32_t)dram_noc_xy.x,
            (uint32_t)dram_noc_xy.y,
            (uint32_t)byte_size,
            (uint32_t)eth_l1_byte_address,
        });

    tt_metal::detail::LaunchProgram(device, program);

    auto readback_vec = llrt::read_hex_vec_from_core(device->id(), dram_noc_xy, dram_byte_address, byte_size);
    pass &= (readback_vec == inputs);
    if (not pass) {
        std::cout << "Mismatch at Core: " << dram_noc_xy.str() << std::endl;
    }
    return pass;
}

TEST_F(N300DeviceFixture, EthKernelsNocReadNoSend) {
    const auto& device_0 = devices_.at(0);
    const auto& device_1 = devices_.at(1);

    const size_t src_eth_l1_byte_address = eth_l1_mem::address_map::ERISC_L1_UNRESERVED_BASE;

    for (const auto& eth_core : device_0->get_active_ethernet_cores()) {
        ASSERT_TRUE(
            unit_tests::erisc::kernels::reader_kernel_no_send(device_0, WORD_SIZE, src_eth_l1_byte_address, eth_core));
        ASSERT_TRUE(unit_tests::erisc::kernels::reader_kernel_no_send(
            device_0, WORD_SIZE * 1024, src_eth_l1_byte_address, eth_core));
        ASSERT_TRUE(unit_tests::erisc::kernels::reader_kernel_no_send(
            device_0, WORD_SIZE * 2048, src_eth_l1_byte_address, eth_core));
    }

    for (const auto& eth_core : device_1->get_active_ethernet_cores()) {
        ASSERT_TRUE(
            unit_tests::erisc::kernels::reader_kernel_no_send(device_1, WORD_SIZE, src_eth_l1_byte_address, eth_core));
        ASSERT_TRUE(unit_tests::erisc::kernels::reader_kernel_no_send(
            device_1, WORD_SIZE * 1024, src_eth_l1_byte_address, eth_core));
        ASSERT_TRUE(unit_tests::erisc::kernels::reader_kernel_no_send(
            device_1, WORD_SIZE * 2048, src_eth_l1_byte_address, eth_core));
    }
}

TEST_F(N300DeviceFixture, EthKernelsNocWriteNoReceive) {
    const auto& device_0 = devices_.at(0);
    const auto& device_1 = devices_.at(1);

    const size_t src_eth_l1_byte_address = eth_l1_mem::address_map::ERISC_L1_UNRESERVED_BASE;

    for (const auto& eth_core : device_0->get_active_ethernet_cores()) {
        ASSERT_TRUE(unit_tests::erisc::kernels::writer_kernel_no_receive(
            device_0, WORD_SIZE, src_eth_l1_byte_address, eth_core));
        ASSERT_TRUE(unit_tests::erisc::kernels::writer_kernel_no_receive(
            device_0, WORD_SIZE * 1024, src_eth_l1_byte_address, eth_core));
        ASSERT_TRUE(unit_tests::erisc::kernels::writer_kernel_no_receive(
            device_0, WORD_SIZE * 2048, src_eth_l1_byte_address, eth_core));
    }

    for (const auto& eth_core : device_1->get_active_ethernet_cores()) {
        ASSERT_TRUE(unit_tests::erisc::kernels::writer_kernel_no_receive(
            device_1, WORD_SIZE, src_eth_l1_byte_address, eth_core));
        ASSERT_TRUE(unit_tests::erisc::kernels::writer_kernel_no_receive(
            device_1, WORD_SIZE * 1024, src_eth_l1_byte_address, eth_core));
        ASSERT_TRUE(unit_tests::erisc::kernels::writer_kernel_no_receive(
            device_1, WORD_SIZE * 2048, src_eth_l1_byte_address, eth_core));
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
bool eth_direct_sender_receiver_kernels(
    tt_metal::Device* sender_device,
    tt_metal::Device* receiver_device,
    const size_t& byte_size,
    const size_t& src_eth_l1_byte_address,
    const size_t& dst_eth_l1_byte_address,
    const CoreCoord& eth_sender_core,
    const CoreCoord& eth_receiver_core,
    uint32_t num_bytes_per_send = 16) {
    bool pass = true;
    log_debug(
        tt::LogTest,
        "Sending {} bytes from device {} eth core {} addr {} to device {} eth core {} addr {}",
        byte_size,
        sender_device->id(),
        eth_sender_core.str(),
        src_eth_l1_byte_address,
        receiver_device->id(),
        eth_receiver_core.str(),
        dst_eth_l1_byte_address);
    // Generate inputs
    auto inputs = generate_uniform_random_vector<uint32_t>(0, 100, byte_size / sizeof(uint32_t));
    llrt::write_hex_vec_to_core(
        sender_device->id(),
        sender_device->ethernet_core_from_logical_core(eth_sender_core),
        inputs,
        src_eth_l1_byte_address);

    // Clear expected value at ethernet L1 address
    std::vector<uint32_t> all_zeros(inputs.size(), 0);
    llrt::write_hex_vec_to_core(
        receiver_device->id(),
        receiver_device->ethernet_core_from_logical_core(eth_receiver_core),
        all_zeros,
        dst_eth_l1_byte_address);

    ////////////////////////////////////////////////////////////////////////////
    //                      Sender Device
    ////////////////////////////////////////////////////////////////////////////
    tt_metal::Program sender_program = tt_metal::Program();

    auto eth_sender_kernel = tt_metal::CreateKernel(
        sender_program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/unit_tests/erisc/eth_l1_direct_send.cpp",
        eth_sender_core,
        tt_metal::experimental::EthernetConfig{
            .eth_mode = tt_metal::Eth::SENDER,
            .noc = tt_metal::NOC::NOC_0,
            .compile_args = {uint32_t(num_bytes_per_send), uint32_t(num_bytes_per_send >> 4)}});

    tt_metal::SetRuntimeArgs(
        sender_program,
        eth_sender_kernel,
        eth_sender_core,
        {
            (uint32_t)src_eth_l1_byte_address,
            (uint32_t)dst_eth_l1_byte_address,
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
        tt_metal::experimental::EthernetConfig{
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
        tt_metal::detail::LaunchProgram(sender_device, sender_program);
    });
    std::thread th2 = std::thread([&] {
        tt_metal::detail::LaunchProgram(receiver_device, receiver_program);
    });

    th1.join();
    th2.join();
    // tt_metal::ReadFromBuffer(l1_buffer, dest_core_data);
    auto readback_vec = llrt::read_hex_vec_from_core(
        receiver_device->id(),
        receiver_device->ethernet_core_from_logical_core(eth_receiver_core),
        dst_eth_l1_byte_address,
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
        tt_metal::experimental::EthernetConfig{.eth_mode = tt_metal::Eth::SENDER, .noc = tt_metal::NOC::NOC_0});

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
        tt_metal::experimental::EthernetConfig{
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
        tt_metal::experimental::EthernetConfig{.eth_mode = tt_metal::Eth::RECEIVER, .noc = tt_metal::NOC::NOC_0});
    // Runtime arg at 0 will be used to kill kernel

    auto eth_kernel_1 = tt_metal::CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/unit_tests/erisc/hung_kernel.cpp",
        hung_cores[1],
        tt_metal::experimental::EthernetConfig{.eth_mode = tt_metal::Eth::RECEIVER, .noc = tt_metal::NOC::NOC_0});

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

    const size_t src_eth_l1_byte_address = eth_l1_mem::address_map::ERISC_L1_UNRESERVED_BASE;
    const size_t dst_eth_l1_byte_address = eth_l1_mem::address_map::ERISC_L1_UNRESERVED_BASE;

    for (const auto& sender_core : device_0->get_active_ethernet_cores()) {
        CoreCoord receiver_core = std::get<1>(device_0->get_connected_ethernet_core(sender_core));
        ASSERT_TRUE(unit_tests::erisc::kernels::eth_direct_sender_receiver_kernels(
            device_0,
            device_1,
            WORD_SIZE,
            src_eth_l1_byte_address,
            dst_eth_l1_byte_address,
            sender_core,
            receiver_core));
        ASSERT_TRUE(unit_tests::erisc::kernels::eth_direct_sender_receiver_kernels(
            device_0,
            device_1,
            4 * WORD_SIZE,
            src_eth_l1_byte_address,
            dst_eth_l1_byte_address,
            sender_core,
            receiver_core));
        ASSERT_TRUE(unit_tests::erisc::kernels::eth_direct_sender_receiver_kernels(
            device_0,
            device_1,
            256 * WORD_SIZE,
            src_eth_l1_byte_address,
            dst_eth_l1_byte_address,
            sender_core,
            receiver_core));
        ASSERT_TRUE(unit_tests::erisc::kernels::eth_direct_sender_receiver_kernels(
            device_0,
            device_1,
            1000 * WORD_SIZE,
            src_eth_l1_byte_address,
            dst_eth_l1_byte_address,
            sender_core,
            receiver_core));
    }
}

TEST_F(N300DeviceFixture, EthKernelsDirectSendChip1ToChip0) {
    const auto& device_0 = devices_.at(0);
    const auto& device_1 = devices_.at(1);

    const size_t src_eth_l1_byte_address = eth_l1_mem::address_map::ERISC_L1_UNRESERVED_BASE;
    const size_t dst_eth_l1_byte_address = eth_l1_mem::address_map::ERISC_L1_UNRESERVED_BASE;

    for (const auto& sender_core : device_1->get_active_ethernet_cores()) {
        CoreCoord receiver_core = std::get<1>(device_1->get_connected_ethernet_core(sender_core));
        ASSERT_TRUE(unit_tests::erisc::kernels::eth_direct_sender_receiver_kernels(
            device_1,
            device_0,
            WORD_SIZE,
            src_eth_l1_byte_address,
            dst_eth_l1_byte_address,
            sender_core,
            receiver_core));
        ASSERT_TRUE(unit_tests::erisc::kernels::eth_direct_sender_receiver_kernels(
            device_1,
            device_0,
            4 * WORD_SIZE,
            src_eth_l1_byte_address,
            dst_eth_l1_byte_address,
            sender_core,
            receiver_core));
        ASSERT_TRUE(unit_tests::erisc::kernels::eth_direct_sender_receiver_kernels(
            device_1,
            device_0,
            256 * WORD_SIZE,
            src_eth_l1_byte_address,
            dst_eth_l1_byte_address,
            sender_core,
            receiver_core));
        ASSERT_TRUE(unit_tests::erisc::kernels::eth_direct_sender_receiver_kernels(
            device_1,
            device_0,
            1000 * WORD_SIZE,
            src_eth_l1_byte_address,
            dst_eth_l1_byte_address,
            sender_core,
            receiver_core));
    }
}

TEST_F(N300DeviceFixture, EthKernelsBidirectionalDirectSend) {
    const auto& device_0 = devices_.at(0);
    const auto& device_1 = devices_.at(1);

    const size_t src_eth_l1_byte_address = eth_l1_mem::address_map::ERISC_L1_UNRESERVED_BASE;
    const size_t dst_eth_l1_byte_address = eth_l1_mem::address_map::ERISC_L1_UNRESERVED_BASE;

    for (const auto& sender_core : device_0->get_active_ethernet_cores()) {
        CoreCoord receiver_core = std::get<1>(device_0->get_connected_ethernet_core(sender_core));
        ASSERT_TRUE(unit_tests::erisc::kernels::eth_direct_sender_receiver_kernels(
            device_0,
            device_1,
            WORD_SIZE,
            src_eth_l1_byte_address,
            dst_eth_l1_byte_address,
            sender_core,
            receiver_core));
        ASSERT_TRUE(unit_tests::erisc::kernels::eth_direct_sender_receiver_kernels(
            device_1,
            device_0,
            WORD_SIZE,
            src_eth_l1_byte_address,
            dst_eth_l1_byte_address,
            receiver_core,
            sender_core));
    }
    for (const auto& sender_core : device_0->get_active_ethernet_cores()) {
        CoreCoord receiver_core = std::get<1>(device_0->get_connected_ethernet_core(sender_core));
        ASSERT_TRUE(unit_tests::erisc::kernels::eth_direct_sender_receiver_kernels(
            device_0,
            device_1,
            WORD_SIZE * 256,
            src_eth_l1_byte_address,
            dst_eth_l1_byte_address,
            sender_core,
            receiver_core));
        ASSERT_TRUE(unit_tests::erisc::kernels::eth_direct_sender_receiver_kernels(
            device_1,
            device_0,
            WORD_SIZE * 256,
            src_eth_l1_byte_address,
            dst_eth_l1_byte_address,
            receiver_core,
            sender_core));
    }
    for (const auto& sender_core : device_0->get_active_ethernet_cores()) {
        CoreCoord receiver_core = std::get<1>(device_0->get_connected_ethernet_core(sender_core));
        ASSERT_TRUE(unit_tests::erisc::kernels::eth_direct_sender_receiver_kernels(
            device_0,
            device_1,
            WORD_SIZE * 1024,
            src_eth_l1_byte_address,
            dst_eth_l1_byte_address,
            sender_core,
            receiver_core));
        ASSERT_TRUE(unit_tests::erisc::kernels::eth_direct_sender_receiver_kernels(
            device_1,
            device_0,
            WORD_SIZE * 1024,
            src_eth_l1_byte_address,
            dst_eth_l1_byte_address,
            receiver_core,
            sender_core));
    }
    for (const auto& sender_core : device_0->get_active_ethernet_cores()) {
        CoreCoord receiver_core = std::get<1>(device_0->get_connected_ethernet_core(sender_core));
        ASSERT_TRUE(unit_tests::erisc::kernels::eth_direct_sender_receiver_kernels(
            device_0,
            device_1,
            WORD_SIZE * MAX_NUM_WORDS,
            src_eth_l1_byte_address,
            dst_eth_l1_byte_address,
            sender_core,
            receiver_core));
        ASSERT_TRUE(unit_tests::erisc::kernels::eth_direct_sender_receiver_kernels(
            device_1,
            device_0,
            WORD_SIZE * MAX_NUM_WORDS,
            src_eth_l1_byte_address,
            dst_eth_l1_byte_address,
            receiver_core,
            sender_core));
    }
}

TEST_F(N300DeviceFixture, EthKernelsRepeatedDirectSends) {
    const auto& device_0 = devices_.at(0);
    const auto& device_1 = devices_.at(1);

    const size_t src_eth_l1_byte_address = eth_l1_mem::address_map::ERISC_L1_UNRESERVED_BASE;
    const size_t dst_eth_l1_byte_address = eth_l1_mem::address_map::ERISC_L1_UNRESERVED_BASE;

    for (const auto& sender_core : device_0->get_active_ethernet_cores()) {
        CoreCoord receiver_core = std::get<1>(device_0->get_connected_ethernet_core(sender_core));
        for (int i = 0; i < 10; i++) {
            ASSERT_TRUE(unit_tests::erisc::kernels::eth_direct_sender_receiver_kernels(
                device_0,
                device_1,
                WORD_SIZE,
                src_eth_l1_byte_address + WORD_SIZE * i,
                dst_eth_l1_byte_address + WORD_SIZE * i,
                sender_core,
                receiver_core));
        }
        for (int i = 0; i < 10; i++) {
            ASSERT_TRUE(unit_tests::erisc::kernels::eth_direct_sender_receiver_kernels(
                device_1,
                device_0,
                WORD_SIZE,
                src_eth_l1_byte_address + WORD_SIZE * i,
                dst_eth_l1_byte_address + WORD_SIZE * i,
                receiver_core,
                sender_core));
        }
    }
}

TEST_F(N300DeviceFixture, EthKernelsRandomDirectSendTests) {
    srand(0);
    const auto& device_0 = devices_.at(0);
    const auto& device_1 = devices_.at(1);

    std::map<std::tuple<int, CoreCoord>, std::tuple<int, CoreCoord>> connectivity = {};
    for (const auto& sender_core : device_0->get_active_ethernet_cores()) {
        const auto& receiver_core = device_0->get_connected_ethernet_core(sender_core);
        connectivity.insert({{0, sender_core}, receiver_core});
    }
    for (const auto& sender_core : device_1->get_active_ethernet_cores()) {
        const auto& receiver_core = device_1->get_connected_ethernet_core(sender_core);
        connectivity.insert({{1, sender_core}, receiver_core});
    }
    for (int i = 0; i < 1000; i++) {
        auto it = connectivity.begin();
        std::advance(it, rand() % (connectivity.size()));

        const auto& send_chip = devices_.at(std::get<0>(it->first));
        CoreCoord sender_core = std::get<1>(it->first);
        const auto& receiver_chip = devices_.at(std::get<0>(it->second));
        CoreCoord receiver_core = std::get<1>(it->second);

        const size_t src_eth_l1_byte_address = unit_tests::erisc::kernels::get_rand_32_byte_aligned_address(
            eth_l1_mem::address_map::ERISC_L1_UNRESERVED_BASE, eth_l1_mem::address_map::MAX_L1_LOADING_SIZE);
        const size_t dst_eth_l1_byte_address = unit_tests::erisc::kernels::get_rand_32_byte_aligned_address(
            eth_l1_mem::address_map::ERISC_L1_UNRESERVED_BASE, eth_l1_mem::address_map::MAX_L1_LOADING_SIZE);

        int max_words = (eth_l1_mem::address_map::MAX_L1_LOADING_SIZE -
                         std::max(src_eth_l1_byte_address, dst_eth_l1_byte_address)) /
                        WORD_SIZE;
        int num_words = rand() % max_words + 1;

        ASSERT_TRUE(unit_tests::erisc::kernels::eth_direct_sender_receiver_kernels(
            send_chip,
            receiver_chip,
            WORD_SIZE * num_words,
            src_eth_l1_byte_address,
            dst_eth_l1_byte_address,
            sender_core,
            receiver_core));
    }
}
TEST_F(N300DeviceFixture, EthKernelsRandomEthPacketSizeDirectSendTests) {
    srand(0);
    const auto& device_0 = devices_.at(0);
    const auto& device_1 = devices_.at(1);

    std::map<std::tuple<int, CoreCoord>, std::tuple<int, CoreCoord>> connectivity = {};
    for (const auto& sender_core : device_0->get_active_ethernet_cores()) {
        const auto& receiver_core = device_0->get_connected_ethernet_core(sender_core);
        connectivity.insert({{0, sender_core}, receiver_core});
    }
    for (const auto& sender_core : device_1->get_active_ethernet_cores()) {
        const auto& receiver_core = device_1->get_connected_ethernet_core(sender_core);
        connectivity.insert({{1, sender_core}, receiver_core});
    }
    std::vector<uint32_t> num_bytes_per_send_test_vals = {
        16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536};
    for (const auto& num_bytes_per_send : num_bytes_per_send_test_vals) {
        log_info(tt::LogTest, "Random eth send tests with {} bytes per packet", num_bytes_per_send);
        for (int i = 0; i < 10; i++) {
            auto it = connectivity.begin();
            std::advance(it, rand() % (connectivity.size()));

            const auto& send_chip = devices_.at(std::get<0>(it->first));
            CoreCoord sender_core = std::get<1>(it->first);
            const auto& receiver_chip = devices_.at(std::get<0>(it->second));
            CoreCoord receiver_core = std::get<1>(it->second);

            const size_t src_eth_l1_byte_address = unit_tests::erisc::kernels::get_rand_32_byte_aligned_address(
                eth_l1_mem::address_map::ERISC_L1_UNRESERVED_BASE,
                eth_l1_mem::address_map::MAX_L1_LOADING_SIZE - 65536);
            const size_t dst_eth_l1_byte_address = unit_tests::erisc::kernels::get_rand_32_byte_aligned_address(
                eth_l1_mem::address_map::ERISC_L1_UNRESERVED_BASE,
                eth_l1_mem::address_map::MAX_L1_LOADING_SIZE - 65536);

            int max_words = (eth_l1_mem::address_map::MAX_L1_LOADING_SIZE -
                             std::max(src_eth_l1_byte_address, dst_eth_l1_byte_address)) /
                            num_bytes_per_send;
            int num_words = rand() % max_words + 1;

            ASSERT_TRUE(unit_tests::erisc::kernels::eth_direct_sender_receiver_kernels(
                send_chip,
                receiver_chip,
                num_bytes_per_send * num_words,
                src_eth_l1_byte_address,
                dst_eth_l1_byte_address,
                sender_core,
                receiver_core,
                num_bytes_per_send));
        }
    }
}

TEST_F(N300DeviceFixture, EthKernelsScatterSend) {
    GTEST_SKIP();
    const auto& device_0 = devices_.at(0);
    const auto& device_1 = devices_.at(1);
    CoreCoord sender_core_0 = {.x = 0, .y = 8};
    CoreCoord sender_core_1 = {.x = 0, .y = 9};

    CoreCoord receiver_core_0 = {.x = 0, .y = 0};
    CoreCoord receiver_core_1 = {.x = 0, .y = 1};

    std::size_t src_addr = eth_l1_mem::address_map::ERISC_L1_UNRESERVED_BASE;
    std::vector<size_t> dst_addrs = {
        eth_l1_mem::address_map::ERISC_L1_UNRESERVED_BASE,
        eth_l1_mem::address_map::ERISC_L1_UNRESERVED_BASE + 2500 * WORD_SIZE,
        eth_l1_mem::address_map::ERISC_L1_UNRESERVED_BASE + 2500 * WORD_SIZE * 2,
        eth_l1_mem::address_map::ERISC_L1_UNRESERVED_BASE + 2500 * WORD_SIZE * 3};

    // ASSERT_TRUE(
    //    unit_tests::erisc::kernels::eth_scatter_sender_receiver_kernels(device_1, device_0, 1 * WORD_SIZE, src_addr,
    //    dst_addrs, receiver_core_0, sender_core_0));
    ASSERT_TRUE(unit_tests::erisc::kernels::eth_scatter_sender_receiver_kernels(
        device_0, device_1, 1 * WORD_SIZE, src_addr, dst_addrs, sender_core_0, receiver_core_0));
    // ASSERT_TRUE(
    //   unit_tests::erisc::kernels::eth_scatter_sender_receiver_kernels(device_0, device_1, 2000 * WORD_SIZE, src_addr,
    //   dst_addrs, sender_core_0, receiver_core_0));
}

TEST_F(N300DeviceFixture, HungEthKernelsContextSwitch) {
    GTEST_SKIP();
    const auto& mmio_device = devices_.at(0);
    const auto& remote_device = devices_.at(1);
    std::vector<CoreCoord> hung_cores = {{.x = 0, .y = 8}, {.x = 0, .y = 9}};

    ASSERT_TRUE(unit_tests::erisc::kernels::eth_hung_kernels(mmio_device, remote_device, hung_cores));
}
