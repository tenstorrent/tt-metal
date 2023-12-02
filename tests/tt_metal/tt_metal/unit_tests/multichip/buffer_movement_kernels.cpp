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

struct BankedConfig {
    size_t num_tiles = 1;
    size_t size_bytes = 1 * 2 * 32 * 32;
    size_t page_size_bytes = 2 * 32 * 32;
    BufferType input_buffer_type = BufferType::L1;
    BufferType output_buffer_type = BufferType::L1;
    CoreCoord logical_core = CoreCoord({.x = 0, .y = 0});
    tt::DataFormat l1_data_format = tt::DataFormat::Float16_b;
};

namespace unit_tests::erisc::kernels {

bool chip_to_chip_dram_buffer_transfer(
    tt_metal::Device* sender_device,
    tt_metal::Device* receiver_device,
    const CoreCoord& eth_sender_core,
    const CoreCoord& eth_receiver_core,
    const size_t& byte_size) {
    bool pass = true;


    tt::tt_metal::InterleavedBufferConfig sender_dram_config{
                    .device= sender_device,
                    .size = byte_size,
                    .page_size = byte_size,
                    .buffer_type = tt::tt_metal::BufferType::DRAM
        };
    tt::tt_metal::InterleavedBufferConfig receiver_dram_config{
                    .device= receiver_device,
                    .size = byte_size,
                    .page_size = byte_size,
                    .buffer_type = tt::tt_metal::BufferType::DRAM
        };


    // Create source buffer on sender device
    auto input_dram_buffer = CreateBuffer(sender_dram_config);
    uint32_t input_dram_byte_address = input_dram_buffer.address();
    auto input_dram_noc_xy = input_dram_buffer.noc_coordinates();

    // Create dest buffer on receiver device
    auto output_dram_buffer = CreateBuffer(receiver_dram_config);
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

bool chip_to_chip_interleaved_buffer_transfer(
    tt_metal::Device* sender_device,
    tt_metal::Device* receiver_device,
    const CoreCoord& eth_sender_core,
    const CoreCoord& eth_receiver_core,
    const BankedConfig& cfg) {
    bool pass = true;

    log_info(
        tt::LogTest,
        "Sending interleaved buffer from device {} to device {}, using eth core {} and {}",
        sender_device->id(),
        receiver_device->id(),
        eth_sender_core.str(),
        eth_receiver_core.str());

    const uint32_t input0_cb_index = 0;
    const uint32_t output_cb_index = 16;

    TT_FATAL(cfg.num_tiles * cfg.page_size_bytes == cfg.size_bytes);
    constexpr uint32_t num_pages_cb = 1;

    ////////////////////////////////////////////////////////////////////////////
    //                      Sender Device
    ////////////////////////////////////////////////////////////////////////////
    tt_metal::Program sender_program = tt_metal::Program();

    std::vector<uint32_t> input_packed =
        tt::test_utils::generate_packed_uniform_random_vector<uint32_t, tt::test_utils::df::bfloat16>(
            -1.0f,
            1.0f,
            cfg.size_bytes / tt::test_utils::df::bfloat16::SIZEOF,
            std::chrono::system_clock::now().time_since_epoch().count());

    auto input_buffer = CreateBuffer(sender_device, cfg.size_bytes, cfg.page_size_bytes, cfg.input_buffer_type);
    bool input_is_dram = cfg.input_buffer_type == BufferType::DRAM;

    CircularBufferConfig l1_input_cb_config =
        CircularBufferConfig(cfg.page_size_bytes, {{input0_cb_index, cfg.l1_data_format}})
            .set_page_size(input0_cb_index, cfg.page_size_bytes);
    auto l1_input_cb = CreateCircularBuffer(sender_program, cfg.logical_core, l1_input_cb_config);

    auto eth_sender_kernel = tt_metal::CreateKernel(
        sender_program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/unit_tests/erisc/interleaved_buffer_to_buffer_sender.cpp",
        eth_sender_core,
        tt_metal::experimental::EthernetConfig{
            .eth_mode = tt_metal::Eth::SENDER, .noc = tt_metal::NOC::NOC_0, .compile_args = {(uint32_t)input_is_dram}});

    tt_metal::SetRuntimeArgs(
        sender_program,
        eth_sender_kernel,
        eth_sender_core,
        {
            (uint32_t)input_buffer.address(),
            (uint32_t)cfg.num_tiles,
            (uint32_t)input0_cb_index,
            (uint32_t)cfg.size_bytes,
        });

    ////////////////////////////////////////////////////////////////////////////
    //                      Receiver Device
    ////////////////////////////////////////////////////////////////////////////
    tt_metal::Program receiver_program = tt_metal::Program();

    auto output_buffer = CreateBuffer(receiver_device, cfg.size_bytes, cfg.page_size_bytes, cfg.output_buffer_type);
    bool output_is_dram = cfg.output_buffer_type == BufferType::DRAM;

    CircularBufferConfig l1_output_cb_config =
        CircularBufferConfig(cfg.page_size_bytes, {{output_cb_index, cfg.l1_data_format}})
            .set_page_size(output_cb_index, cfg.page_size_bytes);
    auto l1_output_cb = CreateCircularBuffer(receiver_program, cfg.logical_core, l1_output_cb_config);

    auto eth_receiver_kernel = tt_metal::CreateKernel(
        receiver_program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/unit_tests/erisc/interleaved_buffer_to_buffer_receiver.cpp",
        eth_receiver_core,
        tt_metal::experimental::EthernetConfig{
            .eth_mode = tt_metal::Eth::RECEIVER,
            .noc = tt_metal::NOC::NOC_0,
            .compile_args = {(uint32_t)output_is_dram}});

    tt_metal::SetRuntimeArgs(
        receiver_program,
        eth_receiver_kernel,
        eth_receiver_core,
        {
            (uint32_t)output_buffer.address(),
            (uint32_t)cfg.num_tiles,
            (uint32_t)output_cb_index,
            (uint32_t)cfg.size_bytes,
        });

    ////////////////////////////////////////////////////////////////////////////
    //                      Execute Programs
    ////////////////////////////////////////////////////////////////////////////

    std::thread th1 = std::thread([&] { tt_metal::detail::LaunchProgram(sender_device, sender_program); });
    std::thread th2 = std::thread([&] { tt_metal::detail::LaunchProgram(receiver_device, receiver_program); });

    th1.join();
    th2.join();
    std::vector<uint32_t> dest_buffer_data;
    tt_metal::detail::ReadFromBuffer(output_buffer, dest_buffer_data);
    pass &= input_packed == dest_buffer_data;
    return pass;
}

}  // namespace unit_tests::erisc::kernels

TEST_F(N300DeviceFixture, EthKernelsSendDramBufferChip0ToChip1) {
    const auto& sender_device = devices_.at(0);
    const auto& receiver_device = devices_.at(1);

    for (const auto& sender_eth_core : sender_device->get_active_ethernet_cores()) {
        CoreCoord receiver_eth_core = std::get<1>(sender_device->get_connected_ethernet_core(sender_eth_core));

        ASSERT_TRUE(unit_tests::erisc::kernels::chip_to_chip_dram_buffer_transfer(
            sender_device, receiver_device, sender_eth_core, receiver_eth_core, 16));
        ASSERT_TRUE(unit_tests::erisc::kernels::chip_to_chip_dram_buffer_transfer(
            sender_device, receiver_device, sender_eth_core, receiver_eth_core, 1024));
        ASSERT_TRUE(unit_tests::erisc::kernels::chip_to_chip_dram_buffer_transfer(
            sender_device, receiver_device, sender_eth_core, receiver_eth_core, 16 * 1024));
        ASSERT_TRUE(unit_tests::erisc::kernels::chip_to_chip_dram_buffer_transfer(
            sender_device, receiver_device, sender_eth_core, receiver_eth_core, 1000 * 1024));
    }
}

TEST_F(N300DeviceFixture, EthKernelsSendDramBufferChip1ToChip0) {
    const auto& sender_device = devices_.at(1);
    const auto& receiver_device = devices_.at(0);

    for (const auto& sender_eth_core : sender_device->get_active_ethernet_cores()) {
        CoreCoord receiver_eth_core = std::get<1>(sender_device->get_connected_ethernet_core(sender_eth_core));

        ASSERT_TRUE(unit_tests::erisc::kernels::chip_to_chip_dram_buffer_transfer(
            sender_device, receiver_device, sender_eth_core, receiver_eth_core, 16));
        ASSERT_TRUE(unit_tests::erisc::kernels::chip_to_chip_dram_buffer_transfer(
            sender_device, receiver_device, sender_eth_core, receiver_eth_core, 1024));
        ASSERT_TRUE(unit_tests::erisc::kernels::chip_to_chip_dram_buffer_transfer(
            sender_device, receiver_device, sender_eth_core, receiver_eth_core, 16 * 1024));
        ASSERT_TRUE(unit_tests::erisc::kernels::chip_to_chip_dram_buffer_transfer(
            sender_device, receiver_device, sender_eth_core, receiver_eth_core, 1000 * 1024));
    }
}

TEST_F(N300DeviceFixture, EthKernelsSendInterleavedBufferChip0ToChip1) {
    const auto& sender_device = devices_.at(0);
    const auto& receiver_device = devices_.at(1);

    BankedConfig test_config;
    for (const auto& sender_eth_core : sender_device->get_active_ethernet_cores()) {
        CoreCoord receiver_eth_core = std::get<1>(sender_device->get_connected_ethernet_core(sender_eth_core));

        ASSERT_TRUE(unit_tests::erisc::kernels::chip_to_chip_interleaved_buffer_transfer(
            sender_device, receiver_device, sender_eth_core, receiver_eth_core, test_config));
    }
}
