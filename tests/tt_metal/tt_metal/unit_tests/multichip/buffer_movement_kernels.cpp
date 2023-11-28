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

namespace unit_tests::erisc::kernels {

bool chip_to_chip_dram_transfer(
    tt_metal::Device* sender_device,
    tt_metal::Device* receiver_device,
    const CoreCoord& eth_sender_core,
    const CoreCoord& eth_receiver_core,
    const size_t& byte_size) {
    bool pass = true;

    // Create source buffer on sender device
    auto input_dram_buffer = CreateBuffer(sender_device, byte_size, byte_size, tt_metal::BufferType::DRAM);
    uint32_t input_dram_byte_address = input_dram_buffer.address();
    auto input_dram_noc_xy = input_dram_buffer.noc_coordinates();

    // Create dest buffer on receiver device
    auto output_dram_buffer = CreateBuffer(receiver_device, byte_size, byte_size, tt_metal::BufferType::DRAM);
    uint32_t output_dram_byte_address = output_dram_buffer.address();
    auto output_dram_noc_xy = output_dram_buffer.noc_coordinates();

    log_info(
        tt::LogTest,
        "Sending {} bytes from device {} dram {} addr {} to device {} dram {} addr {}, using eth core {} and {}",
        byte_size,
        sender_device->id(),
        input_dram_noc_xy.str(),
        input_dram_byte_address,
        receiver_device->id(),
        output_dram_noc_xy.str(),
        output_dram_byte_address,
        eth_sender_core.str(),
        eth_receiver_core.str());

    // Generate inputs
    auto inputs = generate_uniform_random_vector<uint32_t>(0, 100, byte_size / sizeof(uint32_t));

    tt_metal::detail::WriteToBuffer(input_dram_buffer, inputs);

    const uint32_t MAX_BUFFER =
        (eth_l1_mem::address_map::MAX_L1_LOADING_SIZE - eth_l1_mem::address_map::ERISC_L1_UNRESERVED_BASE);
    uint32_t num_loops = (uint32_t)(byte_size / MAX_BUFFER);
    uint32_t remaining_bytes = (uint32_t)(byte_size % MAX_BUFFER);
    // Clear expected value at ethernet L1 address
    std::vector<uint32_t> all_zeros(inputs.size(), 0);

    tt_metal::detail::WriteToBuffer(output_dram_buffer, all_zeros);

    ////////////////////////////////////////////////////////////////////////////
    //                      Sender Device
    ////////////////////////////////////////////////////////////////////////////
    tt_metal::Program sender_program = tt_metal::Program();

    auto eth_sender_kernel = tt_metal::CreateKernel(
        sender_program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/unit_tests/erisc/direct_dram_to_dram_sender.cpp",
        eth_sender_core,
        tt_metal::experimental::EthernetConfig{.eth_mode = tt_metal::Eth::SENDER, .noc = tt_metal::NOC::NOC_0});

    tt_metal::SetRuntimeArgs(
        sender_program,
        eth_sender_kernel,
        eth_sender_core,
        {
            (uint32_t)input_dram_byte_address,
            (uint32_t)input_dram_noc_xy.x,
            (uint32_t)input_dram_noc_xy.y,
            (uint32_t)remaining_bytes,
            (uint32_t)num_loops,
            (uint32_t)MAX_BUFFER,
        });

    ////////////////////////////////////////////////////////////////////////////
    //                      Receiver Device
    ////////////////////////////////////////////////////////////////////////////
    tt_metal::Program receiver_program = tt_metal::Program();

    auto eth_receiver_kernel = tt_metal::CreateKernel(
        receiver_program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/unit_tests/erisc/direct_dram_to_dram_receiver.cpp",
        eth_receiver_core,
        tt_metal::experimental::EthernetConfig{
            .eth_mode = tt_metal::Eth::RECEIVER, .noc = tt_metal::NOC::NOC_0});  // probably want to use NOC_1 here

    tt_metal::SetRuntimeArgs(
        receiver_program,
        eth_receiver_kernel,
        eth_receiver_core,
        {
            (uint32_t)output_dram_byte_address,
            (uint32_t)output_dram_noc_xy.x,
            (uint32_t)output_dram_noc_xy.y,
            (uint32_t)remaining_bytes,
            (uint32_t)num_loops,
            (uint32_t)MAX_BUFFER,
        });

    ////////////////////////////////////////////////////////////////////////////
    //                      Execute Programs
    ////////////////////////////////////////////////////////////////////////////

    std::thread th1 = std::thread([&] { tt_metal::detail::LaunchProgram(sender_device, sender_program); });
    std::thread th2 = std::thread([&] { tt_metal::detail::LaunchProgram(receiver_device, receiver_program); });

    th1.join();
    th2.join();
    std::vector<uint32_t> dest_dram_data;
    tt_metal::detail::ReadFromBuffer(output_dram_buffer, dest_dram_data);
    pass &= (dest_dram_data == inputs);
    if (not pass) {
        std::cout << "Mismatch at Core: " << output_dram_noc_xy.str() << std::endl;
        std::cout << dest_dram_data[0] << std::endl;
    }
    return pass;
}

}  // namespace unit_tests::erisc::kernels

TEST_F(N300DeviceFixture, EthKernelsSendDramBufferChip0ToChip1) {
    const auto& sender_device = devices_.at(0);
    const auto& receiver_device = devices_.at(1);

    for (const auto& sender_eth_core : sender_device->get_active_ethernet_cores()) {
        CoreCoord receiver_eth_core = std::get<1>(sender_device->get_connected_ethernet_core(sender_eth_core));

        ASSERT_TRUE(unit_tests::erisc::kernels::chip_to_chip_dram_transfer(
            sender_device, receiver_device, sender_eth_core, receiver_eth_core, 16));
        ASSERT_TRUE(unit_tests::erisc::kernels::chip_to_chip_dram_transfer(
            sender_device, receiver_device, sender_eth_core, receiver_eth_core, 1024));
        ASSERT_TRUE(unit_tests::erisc::kernels::chip_to_chip_dram_transfer(
            sender_device, receiver_device, sender_eth_core, receiver_eth_core, 16 * 1024));
        ASSERT_TRUE(unit_tests::erisc::kernels::chip_to_chip_dram_transfer(
            sender_device, receiver_device, sender_eth_core, receiver_eth_core, 1000 * 1024));
    }
}

TEST_F(N300DeviceFixture, EthKernelsSendDramBufferChip1ToChip0) {
    const auto& sender_device = devices_.at(1);
    const auto& receiver_device = devices_.at(0);

    for (const auto& sender_eth_core : sender_device->get_active_ethernet_cores()) {
        CoreCoord receiver_eth_core = std::get<1>(sender_device->get_connected_ethernet_core(sender_eth_core));

        ASSERT_TRUE(unit_tests::erisc::kernels::chip_to_chip_dram_transfer(
            sender_device, receiver_device, sender_eth_core, receiver_eth_core, 16));
        ASSERT_TRUE(unit_tests::erisc::kernels::chip_to_chip_dram_transfer(
            sender_device, receiver_device, sender_eth_core, receiver_eth_core, 1024));
        ASSERT_TRUE(unit_tests::erisc::kernels::chip_to_chip_dram_transfer(
            sender_device, receiver_device, sender_eth_core, receiver_eth_core, 16 * 1024));
        ASSERT_TRUE(unit_tests::erisc::kernels::chip_to_chip_dram_transfer(
            sender_device, receiver_device, sender_eth_core, receiver_eth_core, 1000 * 1024));
    }
}
