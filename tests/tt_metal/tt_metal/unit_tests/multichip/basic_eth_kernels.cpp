// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <algorithm>

#include "device_fixture.hpp"
#include "n300_device_fixture.hpp"
#include "tt_metal/detail/tt_metal.hpp"
#include "tt_metal/host_api.hpp"
#include "tt_metal/impl/kernels/kernel.hpp"
#include "tt_metal/test_utils/stimulus.hpp"
#include "tt_metal/impl/program/program_pool.hpp"

using namespace tt;
using namespace tt::test_utils;

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
    auto program = CreateScopedProgram();

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
        tt_metal::EthernetConfig{.noc = tt_metal::NOC::NOC_0});

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

    auto* program_ptr = tt::tt_metal::ProgramPool::instance().get_program(program);
    tt_metal::detail::LaunchProgram(device, *program_ptr);

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
    auto program = CreateScopedProgram();

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
        tt_metal::EthernetConfig{.noc = tt_metal::NOC::NOC_0});

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

    auto* program_ptr = tt::tt_metal::ProgramPool::instance().get_program(program);
    tt_metal::detail::LaunchProgram(device, *program_ptr);

    auto readback_vec = llrt::read_hex_vec_from_core(device->id(), dram_noc_xy, dram_byte_address, byte_size);
    pass &= (readback_vec == inputs);
    if (not pass) {
        std::cout << "Mismatch at Core: " << dram_noc_xy.str() << std::endl;
    }
    return pass;
}

TEST_F(N300DeviceFixture, EthKernelsNocReadNoSend) {
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

TEST_F(N300DeviceFixture, EthKernelsNocWriteNoReceive) {
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
    auto sender_program = tt_metal::CreateScopedProgram();

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
    auto receiver_program = tt_metal::CreateScopedProgram();

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

    auto* sender_program_ptr = tt::tt_metal::ProgramPool::instance().get_program(sender_program);
    auto* receiver_program_ptr = tt::tt_metal::ProgramPool::instance().get_program(receiver_program);

    std::thread th1 = std::thread([&] {
        tt_metal::detail::LaunchProgram(sender_device, *sender_program_ptr);
    });
    std::thread th2 = std::thread([&] {
        tt_metal::detail::LaunchProgram(receiver_device, *receiver_program_ptr);
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

TEST_F(N300DeviceFixture, EthKernelsDirectSendChip0ToChip1) {
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

TEST_F(N300DeviceFixture, EthKernelsDirectSendChip1ToChip0) {
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

TEST_F(DeviceFixture, EthKernelsDirectSendAllConnectedChips) {
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

TEST_F(N300DeviceFixture, EthKernelsBidirectionalDirectSend) {
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

TEST_F(N300DeviceFixture, EthKernelsRepeatedDirectSends) {
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

TEST_F(N300DeviceFixture, EthKernelsRandomDirectSendTests) {
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
TEST_F(N300DeviceFixture, EthKernelsRandomEthPacketSizeDirectSendTests) {
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
