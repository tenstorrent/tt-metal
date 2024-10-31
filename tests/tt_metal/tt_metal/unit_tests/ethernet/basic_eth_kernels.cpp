// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <algorithm>
#include <functional>
#include <random>

#include "device_fixture.hpp"
#include "multi_device_fixture.hpp"
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

namespace {
namespace CMAKE_UNIQUE_NAMESPACE {
constexpr std::int32_t WORD_SIZE = 16;  // 16 bytes per eth send packet
constexpr std::int32_t MAX_NUM_WORDS =
    (eth_l1_mem::address_map::MAX_L1_LOADING_SIZE - eth_l1_mem::address_map::ERISC_L1_UNRESERVED_BASE) / WORD_SIZE;
}
}

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
    const CoreCoord& eth_reader_core,
    const tt_metal::EthernetConfig &ethernet_config = tt_metal::EthernetConfig{.noc = tt_metal::NOC::NOC_0}) {
    bool pass = true;
    ////////////////////////////////////////////////////////////////////////////
    //                      Application Setup
    ////////////////////////////////////////////////////////////////////////////
    tt_metal::Program program = tt_metal::Program();

    tt::tt_metal::InterleavedBufferConfig dram_config{
                    .device=device,
                    .size = byte_size,
                    .page_size = byte_size,
                    .buffer_type = tt::tt_metal::BufferType::DRAM
        };

    auto input_dram_buffer = CreateBuffer(dram_config);
    uint32_t dram_byte_address = input_dram_buffer->address();
    auto dram_noc_xy = input_dram_buffer->noc_coordinates();
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
        ethernet_config);

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
    const CoreCoord& eth_writer_core,
    const tt_metal::EthernetConfig &ethernet_config = tt_metal::EthernetConfig{.noc = tt_metal::NOC::NOC_0}) {
    bool pass = true;
    ////////////////////////////////////////////////////////////////////////////
    //                      Application Setup
    ////////////////////////////////////////////////////////////////////////////
    tt_metal::Program program = tt_metal::Program();

    tt::tt_metal::InterleavedBufferConfig dram_config{
                    .device=device,
                    .size = byte_size,
                    .page_size = byte_size,
                    .buffer_type = tt::tt_metal::BufferType::DRAM
        };

    auto output_dram_buffer = CreateBuffer(dram_config);
    uint32_t dram_byte_address = output_dram_buffer->address();
    auto dram_noc_xy = output_dram_buffer->noc_coordinates();
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
        ethernet_config);

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

bool noc_reader_and_writer_kernels(
    tt_metal::Device *device,
    const uint32_t byte_size,
    const uint32_t eth_dst_l1_address,
    const uint32_t eth_src_l1_address,
    const CoreCoord &logical_eth_core,
    const tt_metal::EthernetConfig &reader_eth_config,
    const tt_metal::EthernetConfig &writer_eth_config) {
    bool pass = true;

    tt_metal::Program program = tt_metal::Program();

    tt_metal::InterleavedBufferConfig dram_config{
        .device=device,
        .size = byte_size,
        .page_size = byte_size,
        .buffer_type = tt_metal::BufferType::DRAM
    };

    auto reader_dram_buffer = CreateBuffer(dram_config);
    auto writer_dram_buffer = CreateBuffer(dram_config);

    auto reader_dram_noc_xy = reader_dram_buffer->noc_coordinates();
    auto writer_dram_noc_xy = writer_dram_buffer->noc_coordinates();

    log_debug(
        tt::LogTest,
        "Device {}: reading {} bytes from dram {} addr {} to ethernet core {} addr {}",
        device->id(),
        byte_size,
        reader_dram_noc_xy.str(),
        reader_dram_buffer->address(),
        logical_eth_core.str(),
        eth_dst_l1_address);
    log_debug(
        tt::LogTest,
        "Device {}: writing {} bytes from ethernet core {} addr {} to dram {} addr {}",
        device->id(),
        byte_size,
        logical_eth_core.str(),
        eth_src_l1_address,
        writer_dram_noc_xy.str(),
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
            (uint32_t)reader_dram_noc_xy.x,
            (uint32_t)reader_dram_noc_xy.y,
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
            (uint32_t)writer_dram_noc_xy.x,
            (uint32_t)writer_dram_noc_xy.y,
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
        log_info(tt::LogTest, "Mismatch at eth core: {}, eth kernel read incorrect values from DRAM", logical_eth_core.str());
    }
    std::vector<uint32_t> dram_readback_vec;
    tt_metal::detail::ReadFromBuffer(writer_dram_buffer, dram_readback_vec);
    pass &= (dram_readback_vec == writer_inputs);
    if (not pass) {
        log_info(tt::LogTest, "Mismatch at eth core: {}, eth kernel wrote incorrect values to DRAM", logical_eth_core.str());
    }

    return pass;
}

TEST_F(N300DeviceFixture, ActiveEthEthKernelsNocReadNoSend) {
    using namespace CMAKE_UNIQUE_NAMESPACE;
    GTEST_SKIP();
    const auto& device_0 = devices_.at(0);
    const auto& device_1 = devices_.at(1);

    const size_t src_eth_l1_byte_address = eth_l1_mem::address_map::ERISC_L1_UNRESERVED_BASE;

    for (const auto& eth_core : device_0->get_active_ethernet_cores(true)) {
        ASSERT_TRUE(
            unit_tests::erisc::kernels::reader_kernel_no_send(device_0, WORD_SIZE, src_eth_l1_byte_address, eth_core));
        ASSERT_TRUE(unit_tests::erisc::kernels::reader_kernel_no_send(
            device_0, WORD_SIZE * 1024, src_eth_l1_byte_address, eth_core));
        ASSERT_TRUE(unit_tests::erisc::kernels::reader_kernel_no_send(
            device_0, WORD_SIZE * 2048, src_eth_l1_byte_address, eth_core));
    }

    for (const auto& eth_core : device_1->get_active_ethernet_cores(true)) {
        ASSERT_TRUE(
            unit_tests::erisc::kernels::reader_kernel_no_send(device_1, WORD_SIZE, src_eth_l1_byte_address, eth_core));
        ASSERT_TRUE(unit_tests::erisc::kernels::reader_kernel_no_send(
            device_1, WORD_SIZE * 1024, src_eth_l1_byte_address, eth_core));
        ASSERT_TRUE(unit_tests::erisc::kernels::reader_kernel_no_send(
            device_1, WORD_SIZE * 2048, src_eth_l1_byte_address, eth_core));
    }
}

TEST_F(N300DeviceFixture, ActiveEthEthKernelsNocWriteNoReceive) {
    using namespace CMAKE_UNIQUE_NAMESPACE;
    GTEST_SKIP();
    const auto& device_0 = devices_.at(0);
    const auto& device_1 = devices_.at(1);

    const size_t src_eth_l1_byte_address = eth_l1_mem::address_map::ERISC_L1_UNRESERVED_BASE;

    for (const auto& eth_core : device_0->get_active_ethernet_cores(true)) {
        ASSERT_TRUE(unit_tests::erisc::kernels::writer_kernel_no_receive(
            device_0, WORD_SIZE, src_eth_l1_byte_address, eth_core));
        ASSERT_TRUE(unit_tests::erisc::kernels::writer_kernel_no_receive(
            device_0, WORD_SIZE * 1024, src_eth_l1_byte_address, eth_core));
        ASSERT_TRUE(unit_tests::erisc::kernels::writer_kernel_no_receive(
            device_0, WORD_SIZE * 2048, src_eth_l1_byte_address, eth_core));
    }

    for (const auto& eth_core : device_1->get_active_ethernet_cores(true)) {
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
        tt_metal::EthernetConfig{
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
        tt_metal::EthernetConfig{.noc = tt_metal::NOC::NOC_0});  // probably want to use NOC_1 here

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



}  // namespace unit_tests::erisc::kernels

TEST_F(N300DeviceFixture, ActiveEthEthKernelsDirectSendChip0ToChip1) {
    using namespace CMAKE_UNIQUE_NAMESPACE;
    GTEST_SKIP();
    const auto& device_0 = devices_.at(0);
    const auto& device_1 = devices_.at(1);

    const size_t src_eth_l1_byte_address = eth_l1_mem::address_map::ERISC_L1_UNRESERVED_BASE;
    const size_t dst_eth_l1_byte_address = eth_l1_mem::address_map::ERISC_L1_UNRESERVED_BASE;

    for (const auto& sender_core : device_0->get_active_ethernet_cores(true)) {
        auto [device_id, receiver_core] = device_0->get_connected_ethernet_core(sender_core);
        if (device_1->id() != device_id) {
            continue;
        }
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

TEST_F(N300DeviceFixture, ActiveEthEthKernelsDirectSendChip1ToChip0) {
    using namespace CMAKE_UNIQUE_NAMESPACE;
    GTEST_SKIP();
    const auto& device_0 = devices_.at(0);
    const auto& device_1 = devices_.at(1);

    const size_t src_eth_l1_byte_address = eth_l1_mem::address_map::ERISC_L1_UNRESERVED_BASE;
    const size_t dst_eth_l1_byte_address = eth_l1_mem::address_map::ERISC_L1_UNRESERVED_BASE;

    for (const auto& sender_core : device_1->get_active_ethernet_cores(true)) {
        auto [device_id, receiver_core] = device_1->get_connected_ethernet_core(sender_core);
        if (device_0->id() != device_id) {
            continue;
        }
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

TEST_F(DeviceFixture, ActiveEthEthKernelsDirectSendAllConnectedChips) {
    using namespace CMAKE_UNIQUE_NAMESPACE;
    const size_t src_eth_l1_byte_address = eth_l1_mem::address_map::ERISC_L1_UNRESERVED_BASE;
    const size_t dst_eth_l1_byte_address = eth_l1_mem::address_map::ERISC_L1_UNRESERVED_BASE;
    for (const auto& sender_device : devices_) {
        for (const auto& receiver_device : devices_) {
            if (sender_device->id() == receiver_device->id()) {
                continue;
            }
            for (const auto& sender_core : sender_device->get_active_ethernet_cores(true)) {
                auto [device_id, receiver_core] = sender_device->get_connected_ethernet_core(sender_core);
                if (receiver_device->id() != device_id) {
                    continue;
                }
                ASSERT_TRUE(unit_tests::erisc::kernels::eth_direct_sender_receiver_kernels(
                    sender_device,
                    receiver_device,
                    WORD_SIZE,
                    src_eth_l1_byte_address,
                    dst_eth_l1_byte_address,
                    sender_core,
                    receiver_core));
                ASSERT_TRUE(unit_tests::erisc::kernels::eth_direct_sender_receiver_kernels(
                    sender_device,
                    receiver_device,
                    4 * WORD_SIZE,
                    src_eth_l1_byte_address,
                    dst_eth_l1_byte_address,
                    sender_core,
                    receiver_core));
                ASSERT_TRUE(unit_tests::erisc::kernels::eth_direct_sender_receiver_kernels(
                    sender_device,
                    receiver_device,
                    256 * WORD_SIZE,
                    src_eth_l1_byte_address,
                    dst_eth_l1_byte_address,
                    sender_core,
                    receiver_core));
                ASSERT_TRUE(unit_tests::erisc::kernels::eth_direct_sender_receiver_kernels(
                    sender_device,
                    receiver_device,
                    1000 * WORD_SIZE,
                    src_eth_l1_byte_address,
                    dst_eth_l1_byte_address,
                    sender_core,
                    receiver_core));
            }
        }
    }
}

TEST_F(N300DeviceFixture, ActiveEthEthKernelsBidirectionalDirectSend) {
    using namespace CMAKE_UNIQUE_NAMESPACE;
    const auto& device_0 = devices_.at(0);
    const auto& device_1 = devices_.at(1);

    const size_t src_eth_l1_byte_address = eth_l1_mem::address_map::ERISC_L1_UNRESERVED_BASE;
    const size_t dst_eth_l1_byte_address = eth_l1_mem::address_map::ERISC_L1_UNRESERVED_BASE;

    for (const auto& sender_core : device_0->get_active_ethernet_cores(true)) {
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
    for (const auto& sender_core : device_0->get_active_ethernet_cores(true)) {
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
    for (const auto& sender_core : device_0->get_active_ethernet_cores(true)) {
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
    for (const auto& sender_core : device_0->get_active_ethernet_cores(true)) {
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

TEST_F(N300DeviceFixture, ActiveEthEthKernelsRepeatedDirectSends) {
    using namespace CMAKE_UNIQUE_NAMESPACE;
    const auto& device_0 = devices_.at(0);
    const auto& device_1 = devices_.at(1);

    const size_t src_eth_l1_byte_address = eth_l1_mem::address_map::ERISC_L1_UNRESERVED_BASE;
    const size_t dst_eth_l1_byte_address = eth_l1_mem::address_map::ERISC_L1_UNRESERVED_BASE;

    for (const auto& sender_core : device_0->get_active_ethernet_cores(true)) {
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

TEST_F(N300DeviceFixture, ActiveEthEthKernelsRandomDirectSendTests) {
    using namespace CMAKE_UNIQUE_NAMESPACE;
    srand(0);
    const auto& device_0 = devices_.at(0);
    const auto& device_1 = devices_.at(1);

    std::map<std::tuple<int, CoreCoord>, std::tuple<int, CoreCoord>> connectivity = {};
    for (const auto& sender_core : device_0->get_active_ethernet_cores(true)) {
        const auto& receiver_core = device_0->get_connected_ethernet_core(sender_core);
        connectivity.insert({{0, sender_core}, receiver_core});
    }
    for (const auto& sender_core : device_1->get_active_ethernet_cores(true)) {
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
TEST_F(N300DeviceFixture, ActiveEthEthKernelsRandomEthPacketSizeDirectSendTests) {
    srand(0);
    const auto& device_0 = devices_.at(0);
    const auto& device_1 = devices_.at(1);

    std::map<std::tuple<int, CoreCoord>, std::tuple<int, CoreCoord>> connectivity = {};
    for (const auto& sender_core : device_0->get_active_ethernet_cores(true)) {
        const auto& receiver_core = device_0->get_connected_ethernet_core(sender_core);
        connectivity.insert({{0, sender_core}, receiver_core});
    }
    for (const auto& sender_core : device_1->get_active_ethernet_cores(true)) {
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

// TODO #14640: Run this on WH when i$ flush issue is addressed
TEST_F(BlackholeSingleCardFixture, EthKernelOnIdleErisc0) {
    using namespace CMAKE_UNIQUE_NAMESPACE;
    uint32_t eth_l1_address = hal.get_dev_addr(HalProgrammableCoreType::IDLE_ETH, HalL1MemAddrType::UNRESERVED);
    tt_metal::EthernetConfig noc0_ethernet_config{.eth_mode = Eth::IDLE, .noc = tt_metal::NOC::NOC_0, .processor = tt_metal::DataMovementProcessor::RISCV_0};
    tt_metal::EthernetConfig noc1_ethernet_config{.eth_mode = Eth::IDLE, .noc = tt_metal::NOC::NOC_1, .processor = tt_metal::DataMovementProcessor::RISCV_0};

    for (const auto& eth_core : device_->get_inactive_ethernet_cores()) {
        ASSERT_TRUE(unit_tests::erisc::kernels::reader_kernel_no_send(
            device_, WORD_SIZE * 2048, eth_l1_address, eth_core, noc0_ethernet_config));
        ASSERT_TRUE(unit_tests::erisc::kernels::reader_kernel_no_send(
            device_, WORD_SIZE * 2048, eth_l1_address, eth_core, noc1_ethernet_config));
        ASSERT_TRUE(unit_tests::erisc::kernels::writer_kernel_no_receive(
            device_, WORD_SIZE * 2048, eth_l1_address, eth_core, noc0_ethernet_config));
        ASSERT_TRUE(unit_tests::erisc::kernels::writer_kernel_no_receive(
            device_, WORD_SIZE * 2048, eth_l1_address, eth_core, noc1_ethernet_config));
    }
}

TEST_F(BlackholeSingleCardFixture, EthKernelOnIdleErisc1) {
    using namespace CMAKE_UNIQUE_NAMESPACE;
    uint32_t eth_l1_address = hal.get_dev_addr(HalProgrammableCoreType::IDLE_ETH, HalL1MemAddrType::UNRESERVED);
    tt_metal::EthernetConfig noc0_ethernet_config{.eth_mode = Eth::IDLE, .noc = tt_metal::NOC::NOC_0, .processor = tt_metal::DataMovementProcessor::RISCV_1};
    tt_metal::EthernetConfig noc1_ethernet_config{.eth_mode = Eth::IDLE, .noc = tt_metal::NOC::NOC_1, .processor = tt_metal::DataMovementProcessor::RISCV_1};

    for (const auto& eth_core : device_->get_inactive_ethernet_cores()) {
        ASSERT_TRUE(unit_tests::erisc::kernels::reader_kernel_no_send(
            device_, WORD_SIZE * 2048, eth_l1_address, eth_core, noc0_ethernet_config));
        ASSERT_TRUE(unit_tests::erisc::kernels::reader_kernel_no_send(
            device_, WORD_SIZE * 2048, eth_l1_address, eth_core, noc1_ethernet_config));
        ASSERT_TRUE(unit_tests::erisc::kernels::writer_kernel_no_receive(
            device_, WORD_SIZE * 2048, eth_l1_address, eth_core, noc0_ethernet_config));
        ASSERT_TRUE(unit_tests::erisc::kernels::writer_kernel_no_receive(
            device_, WORD_SIZE * 2048, eth_l1_address, eth_core, noc1_ethernet_config));
    }
}

TEST_F(BlackholeSingleCardFixture, EthKernelOnBothIdleEriscs) {
    using namespace CMAKE_UNIQUE_NAMESPACE;
    uint32_t read_write_size_bytes = WORD_SIZE * 2048;
    uint32_t reader_dst_address = hal.get_dev_addr(HalProgrammableCoreType::IDLE_ETH, HalL1MemAddrType::UNRESERVED);
    uint32_t writer_src_address = reader_dst_address + read_write_size_bytes;
    tt_metal::EthernetConfig erisc0_ethernet_config{.eth_mode = Eth::IDLE, .noc = tt_metal::NOC::NOC_0, .processor = tt_metal::DataMovementProcessor::RISCV_0};
    tt_metal::EthernetConfig erisc1_ethernet_config{.eth_mode = Eth::IDLE, .noc = tt_metal::NOC::NOC_0, .processor = tt_metal::DataMovementProcessor::RISCV_1};

    for (const auto& eth_core : device_->get_inactive_ethernet_cores()) {
        ASSERT_TRUE(unit_tests::erisc::kernels::noc_reader_and_writer_kernels(
            device_, read_write_size_bytes, reader_dst_address, writer_src_address, eth_core, erisc0_ethernet_config, erisc1_ethernet_config
        ));
        erisc0_ethernet_config.noc = tt_metal::NOC::NOC_1;
        erisc1_ethernet_config.noc = tt_metal::NOC::NOC_1;
        ASSERT_TRUE(unit_tests::erisc::kernels::noc_reader_and_writer_kernels(
            device_, read_write_size_bytes, reader_dst_address, writer_src_address, eth_core, erisc0_ethernet_config, erisc1_ethernet_config
        ));
    }
}
