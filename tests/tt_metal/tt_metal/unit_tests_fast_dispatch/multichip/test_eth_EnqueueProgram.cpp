// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <algorithm>
#include <functional>
#include <random>

#include "command_queue_fixture.hpp"
#include "tt_metal/detail/tt_metal.hpp"
#include "tt_metal/host_api.hpp"
#include "tt_metal/impl/kernels/kernel.hpp"
#include "tt_metal/impl/buffers/buffer.hpp"
#include "tt_metal/impl/device/device.hpp"

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
constexpr std::int32_t MAX_BUFFER_SIZE =
    (eth_l1_mem::address_map::MAX_L1_LOADING_SIZE - eth_l1_mem::address_map::ERISC_L1_UNRESERVED_BASE);

struct BankedConfig {
    size_t num_pages = 1;
    size_t size_bytes = 1 * 2 * 32 * 32;
    size_t page_size_bytes = 2 * 32 * 32;
    tt_metal::BufferType input_buffer_type = tt_metal::BufferType::L1;
    tt_metal::BufferType output_buffer_type = tt_metal::BufferType::L1;
    tt::DataFormat l1_data_format = tt::DataFormat::Float16_b;
};

namespace fd_unit_tests::erisc::kernels {

const size_t get_rand_32_byte_aligned_address(const size_t& base, const size_t& max) {
    TT_ASSERT(!(base & 0x1F) and !(max & 0x1F));
    size_t word_size = (max >> 5) - (base >> 5);
    return (((rand() % word_size) << 5) + base);
}

bool test_dummy_EnqueueProgram_with_runtime_args(Device* device, const CoreCoord& eth_core_coord) {
    Program program;
    bool pass = true;
    auto eth_noc_xy = device->ethernet_core_from_logical_core(eth_core_coord);

    constexpr uint32_t num_runtime_args0 = 9;
    constexpr uint32_t rta_base0 = eth_l1_mem::address_map::ERISC_L1_UNRESERVED_BASE;
    std::map<string, string> dummy_defines0 = {{"DATA_MOVEMENT", "1"},
                                               {"NUM_RUNTIME_ARGS", std::to_string(num_runtime_args0)},
                                               {"RESULTS_ADDR", std::to_string(rta_base0)}};
    auto dummy_kernel0 = CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/misc/runtime_args_kernel.cpp",
        eth_core_coord,
        tt_metal::EthernetConfig{.noc = tt_metal::NOC::NOC_0, .defines = dummy_defines0});

    vector<uint32_t> dummy_kernel0_args = {0, 1, 2, 3, 4, 5, 6, 7, 8};
    tt::tt_metal::SetRuntimeArgs(program, dummy_kernel0, eth_core_coord, dummy_kernel0_args);

    tt::tt_metal::detail::CompileProgram(device, program);
    auto& cq = device->command_queue();
    EnqueueProgram(cq, &program, false);
    Finish(cq);

    vector<uint32_t> dummy_kernel0_args_readback = llrt::read_hex_vec_from_core(
        device->id(),
        eth_noc_xy,
        eth_l1_mem::address_map::ERISC_L1_UNRESERVED_BASE,
        dummy_kernel0_args.size() * sizeof(uint32_t));

    pass &= (dummy_kernel0_args == dummy_kernel0_args_readback);

    return pass;
}

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

    tt::tt_metal::InterleavedBufferConfig dram_config{
        .device = device, .size = byte_size, .page_size = byte_size, .buffer_type = tt::tt_metal::BufferType::DRAM};

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

    auto& cq = device->command_queue();
    tt::tt_metal::detail::CompileProgram(device, program);
    EnqueueProgram(cq, &program, false);
    Finish(cq);

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

    tt::tt_metal::InterleavedBufferConfig dram_config{
        .device = device, .size = byte_size, .page_size = byte_size, .buffer_type = tt::tt_metal::BufferType::DRAM};

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

    auto& cq = device->command_queue();
    tt::tt_metal::detail::CompileProgram(device, program);
    EnqueueProgram(cq, &program, false);
    Finish(cq);

    auto readback_vec = llrt::read_hex_vec_from_core(device->id(), dram_noc_xy, dram_byte_address, byte_size);
    pass &= (readback_vec == inputs);
    if (not pass) {
        std::cout << "Mismatch at Core: " << dram_noc_xy.str() << std::endl;
    }
    return pass;
}

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
    //                      Compile and Execute Application
    ////////////////////////////////////////////////////////////////////////////

    tt::tt_metal::detail::CompileProgram(sender_device, sender_program);
    tt::tt_metal::detail::CompileProgram(receiver_device, receiver_program);

    EnqueueProgram(sender_device->command_queue(), &sender_program, false);
    EnqueueProgram(receiver_device->command_queue(), &receiver_program, false);
    Finish(sender_device->command_queue());
    Finish(receiver_device->command_queue());

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

bool chip_to_chip_dram_buffer_transfer(
    tt_metal::Device* sender_device,
    tt_metal::Device* receiver_device,
    const CoreCoord& eth_sender_core,
    const CoreCoord& eth_receiver_core,
    const size_t& byte_size) {
    bool pass = true;

    tt::tt_metal::InterleavedBufferConfig sender_dram_config{
        .device = sender_device,
        .size = byte_size,
        .page_size = byte_size,
        .buffer_type = tt::tt_metal::BufferType::DRAM};
    tt::tt_metal::InterleavedBufferConfig receiver_dram_config{
        .device = receiver_device,
        .size = byte_size,
        .page_size = byte_size,
        .buffer_type = tt::tt_metal::BufferType::DRAM};

    // Create source buffer on sender device
    auto input_dram_buffer = CreateBuffer(sender_dram_config);
    uint32_t input_dram_byte_address = input_dram_buffer->address();
    auto input_dram_noc_xy = input_dram_buffer->noc_coordinates();

    // Create dest buffer on receiver device
    auto output_dram_buffer = CreateBuffer(receiver_dram_config);
    uint32_t output_dram_byte_address = output_dram_buffer->address();
    auto output_dram_noc_xy = output_dram_buffer->noc_coordinates();

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
        tt_metal::EthernetConfig{.noc = tt_metal::NOC::NOC_0});

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
        tt_metal::EthernetConfig{.noc = tt_metal::NOC::NOC_0});  // probably want to use NOC_1 here

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
    //                      Compile and Execute Application
    ////////////////////////////////////////////////////////////////////////////

    tt::tt_metal::detail::CompileProgram(sender_device, sender_program);
    tt::tt_metal::detail::CompileProgram(receiver_device, receiver_program);

    EnqueueProgram(sender_device->command_queue(), &sender_program, false);
    EnqueueProgram(receiver_device->command_queue(), &receiver_program, false);
    Finish(sender_device->command_queue());
    Finish(receiver_device->command_queue());

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
    const BankedConfig& cfg,
    const uint32_t& max_transfer_size) {
    bool pass = true;

    const uint32_t input0_cb_index = 0;
    const uint32_t output_cb_index = 16;

    TT_FATAL(cfg.num_pages * cfg.page_size_bytes == cfg.size_bytes);
    constexpr uint32_t num_pages_cb = 1;

    ////////////////////////////////////////////////////////////////////////////
    //                      Sender Device
    ////////////////////////////////////////////////////////////////////////////
    tt_metal::Program sender_program = tt_metal::Program();

    auto input_packed = generate_uniform_random_vector<uint32_t>(0, 100, cfg.size_bytes / sizeof(uint32_t));

    tt::tt_metal::InterleavedBufferConfig sender_config{
        .device = sender_device,
        .size = cfg.size_bytes,
        .page_size = cfg.page_size_bytes,
        .buffer_type = cfg.input_buffer_type};
    tt::tt_metal::InterleavedBufferConfig receiver_config{
        .device = receiver_device,
        .size = cfg.size_bytes,
        .page_size = cfg.page_size_bytes,
        .buffer_type = cfg.output_buffer_type};
    auto input_buffer = CreateBuffer(sender_config);
    bool input_is_dram = cfg.input_buffer_type == BufferType::DRAM;

    tt_metal::detail::WriteToBuffer(input_buffer, input_packed);

    const uint32_t max_buffer = round_down(max_transfer_size, cfg.page_size_bytes);
    uint32_t pages_per_loop = max_buffer / cfg.page_size_bytes;
    uint32_t num_loops = (uint32_t)(cfg.size_bytes / max_buffer);
    uint32_t remaining_bytes = (uint32_t)(cfg.size_bytes % max_buffer);
    uint32_t remaining_pages = remaining_bytes / cfg.page_size_bytes;

    auto eth_sender_kernel = tt_metal::CreateKernel(
        sender_program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/unit_tests/erisc/interleaved_buffer_to_buffer_sender.cpp",
        eth_sender_core,
        tt_metal::EthernetConfig{.noc = tt_metal::NOC::NOC_0, .compile_args = {(uint32_t)input_is_dram}});

    tt_metal::SetRuntimeArgs(
        sender_program,
        eth_sender_kernel,
        eth_sender_core,
        {(uint32_t)input_buffer->address(),
         (uint32_t)cfg.page_size_bytes,
         (uint32_t)max_buffer,
         (uint32_t)num_loops,
         (uint32_t)pages_per_loop,
         (uint32_t)remaining_bytes,
         (uint32_t)remaining_pages});

    ////////////////////////////////////////////////////////////////////////////
    //                      Receiver Device
    ////////////////////////////////////////////////////////////////////////////
    tt_metal::Program receiver_program = tt_metal::Program();

    auto output_buffer = CreateBuffer(receiver_config);
    bool output_is_dram = cfg.output_buffer_type == BufferType::DRAM;
    std::vector<uint32_t> all_zeros(cfg.size_bytes / sizeof(uint32_t), 0);

    tt_metal::detail::WriteToBuffer(output_buffer, all_zeros);

    auto eth_receiver_kernel = tt_metal::CreateKernel(
        receiver_program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/unit_tests/erisc/interleaved_buffer_to_buffer_receiver.cpp",
        eth_receiver_core,
        tt_metal::EthernetConfig{.noc = tt_metal::NOC::NOC_1, .compile_args = {(uint32_t)output_is_dram}});

    tt_metal::SetRuntimeArgs(
        receiver_program,
        eth_receiver_kernel,
        eth_receiver_core,
        {
            (uint32_t)output_buffer->address(),
            (uint32_t)cfg.page_size_bytes,
            (uint32_t)max_buffer,
            (uint32_t)num_loops,
            (uint32_t)pages_per_loop,
            (uint32_t)remaining_bytes,
            (uint32_t)remaining_pages,
        });

    ////////////////////////////////////////////////////////////////////////////
    //                      Compile and Execute Application
    ////////////////////////////////////////////////////////////////////////////

    tt::tt_metal::detail::CompileProgram(sender_device, sender_program);
    tt::tt_metal::detail::CompileProgram(receiver_device, receiver_program);

    EnqueueProgram(sender_device->command_queue(), &sender_program, false);
    EnqueueProgram(receiver_device->command_queue(), &receiver_program, false);
    Finish(sender_device->command_queue());
    Finish(receiver_device->command_queue());

    std::vector<uint32_t> dest_buffer_data;
    tt_metal::detail::ReadFromBuffer(output_buffer, dest_buffer_data);
    pass &= input_packed == dest_buffer_data;
    return pass;
}
}  // namespace fd_unit_tests::erisc::kernels

TEST_F(CommandQueueSingleCardFixture, EnqueueDummyProgramOnEthCore) {
    for (const auto& device : devices_) {
        for (const auto& eth_core : device->get_active_ethernet_cores(true)) {
            ASSERT_TRUE(fd_unit_tests::erisc::kernels::test_dummy_EnqueueProgram_with_runtime_args(device, eth_core));
        }
    }
}

TEST_F(CommandQueueSingleCardFixture, EthKernelsNocReadNoSend) {
    const size_t src_eth_l1_byte_address = eth_l1_mem::address_map::ERISC_L1_UNRESERVED_BASE;

    for (const auto& device : devices_) {
        for (const auto& eth_core : device->get_active_ethernet_cores(true)) {
            ASSERT_TRUE(fd_unit_tests::erisc::kernels::reader_kernel_no_send(
                device, WORD_SIZE, src_eth_l1_byte_address, eth_core));
            ASSERT_TRUE(fd_unit_tests::erisc::kernels::reader_kernel_no_send(
                device, WORD_SIZE * 1024, src_eth_l1_byte_address, eth_core));
            ASSERT_TRUE(fd_unit_tests::erisc::kernels::reader_kernel_no_send(
                device, WORD_SIZE * 2048, src_eth_l1_byte_address, eth_core));
        }
    }
}

TEST_F(CommandQueueSingleCardFixture, EthKernelsNocWriteNoReceive) {
    const size_t src_eth_l1_byte_address = eth_l1_mem::address_map::ERISC_L1_UNRESERVED_BASE;

    for (const auto& device : devices_) {
        for (const auto& eth_core : device->get_active_ethernet_cores(true)) {
            ASSERT_TRUE(fd_unit_tests::erisc::kernels::writer_kernel_no_receive(
                device, WORD_SIZE, src_eth_l1_byte_address, eth_core));
            ASSERT_TRUE(fd_unit_tests::erisc::kernels::writer_kernel_no_receive(
                device, WORD_SIZE * 1024, src_eth_l1_byte_address, eth_core));
            ASSERT_TRUE(fd_unit_tests::erisc::kernels::writer_kernel_no_receive(
                device, WORD_SIZE * 2048, src_eth_l1_byte_address, eth_core));
        }
    }
}

TEST_F(CommandQueueMultiDeviceFixture, EthKernelsDirectSendAllConnectedChips) {
    tt::ARCH arch = tt::get_arch_from_string(tt::test_utils::get_env_arch_name());
    const size_t src_eth_l1_byte_address = eth_l1_mem::address_map::ERISC_L1_UNRESERVED_BASE;
    const size_t dst_eth_l1_byte_address = eth_l1_mem::address_map::ERISC_L1_UNRESERVED_BASE;
    for (const auto& sender_device : devices_) {
        for (const auto& receiver_device : devices_) {
            if (sender_device->id() >= receiver_device->id()) {
                continue;
            }
            for (const auto& sender_core : sender_device->get_active_ethernet_cores(true)) {
                auto [device_id, receiver_core] = sender_device->get_connected_ethernet_core(sender_core);
                if (receiver_device->id() != device_id) {
                    continue;
                }
                ASSERT_TRUE(fd_unit_tests::erisc::kernels::eth_direct_sender_receiver_kernels(
                    sender_device,
                    receiver_device,
                    WORD_SIZE,
                    src_eth_l1_byte_address,
                    dst_eth_l1_byte_address,
                    sender_core,
                    receiver_core));
                ASSERT_TRUE(fd_unit_tests::erisc::kernels::eth_direct_sender_receiver_kernels(
                    sender_device,
                    receiver_device,
                    4 * WORD_SIZE,
                    src_eth_l1_byte_address,
                    dst_eth_l1_byte_address,
                    sender_core,
                    receiver_core));
                ASSERT_TRUE(fd_unit_tests::erisc::kernels::eth_direct_sender_receiver_kernels(
                    sender_device,
                    receiver_device,
                    256 * WORD_SIZE,
                    src_eth_l1_byte_address,
                    dst_eth_l1_byte_address,
                    sender_core,
                    receiver_core));
                ASSERT_TRUE(fd_unit_tests::erisc::kernels::eth_direct_sender_receiver_kernels(
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

TEST_F(CommandQueueMultiDeviceFixture, EthKernelsSendDramBufferAllConnectedChips) {
    tt::ARCH arch = tt::get_arch_from_string(tt::test_utils::get_env_arch_name());
    for (const auto& sender_device : devices_) {
        for (const auto& receiver_device : devices_) {
            if (sender_device->id() >= receiver_device->id()) {
                continue;
            }
            for (const auto& sender_eth_core : sender_device->get_active_ethernet_cores(true)) {
                auto [device_id, receiver_eth_core] = sender_device->get_connected_ethernet_core(sender_eth_core);
                if (receiver_device->id() != device_id) {
                    continue;
                }
                log_info(
                    tt::LogTest,
                    "Sending dram buffer from device {} to device {}, using eth core {} and {}",
                    sender_device->id(),
                    receiver_device->id(),
                    sender_eth_core.str(),
                    receiver_eth_core.str());

                ASSERT_TRUE(fd_unit_tests::erisc::kernels::chip_to_chip_dram_buffer_transfer(
                    sender_device, receiver_device, sender_eth_core, receiver_eth_core, 16));
                ASSERT_TRUE(fd_unit_tests::erisc::kernels::chip_to_chip_dram_buffer_transfer(
                    sender_device, receiver_device, sender_eth_core, receiver_eth_core, 1024));
                ASSERT_TRUE(fd_unit_tests::erisc::kernels::chip_to_chip_dram_buffer_transfer(
                    sender_device, receiver_device, sender_eth_core, receiver_eth_core, 16 * 1024));
                ASSERT_TRUE(fd_unit_tests::erisc::kernels::chip_to_chip_dram_buffer_transfer(
                    sender_device, receiver_device, sender_eth_core, receiver_eth_core, 1000 * 1024));
            }
        }
    }
}

TEST_F(CommandQueueMultiDeviceFixture, EthKernelsSendInterleavedBufferAllConnectedChips) {
    tt::ARCH arch = tt::get_arch_from_string(tt::test_utils::get_env_arch_name());
    for (const auto& sender_device : devices_) {
        for (const auto& receiver_device : devices_) {
            if (sender_device->id() >= receiver_device->id()) {
                continue;
            }
            for (const auto& sender_eth_core : sender_device->get_active_ethernet_cores(true)) {
                auto [device_id, receiver_eth_core] = sender_device->get_connected_ethernet_core(sender_eth_core);
                if (receiver_device->id() != device_id) {
                    continue;
                }

                log_info(
                    tt::LogTest,
                    "Sending interleaved buffer from device {} to device {}, using eth core {} and {}",
                    sender_device->id(),
                    receiver_device->id(),
                    sender_eth_core.str(),
                    receiver_eth_core.str());
                BankedConfig test_config = BankedConfig{
                    .num_pages = 200,
                    .size_bytes = 200 * 2 * 32 * 32,
                    .page_size_bytes = 2 * 32 * 32,
                    .input_buffer_type = BufferType::L1,
                    .output_buffer_type = BufferType::DRAM};

                ASSERT_TRUE(fd_unit_tests::erisc::kernels::chip_to_chip_interleaved_buffer_transfer(
                    sender_device,
                    receiver_device,
                    sender_eth_core,
                    receiver_eth_core,
                    test_config,
                    test_config.page_size_bytes));
                ASSERT_TRUE(fd_unit_tests::erisc::kernels::chip_to_chip_interleaved_buffer_transfer(
                    sender_device, receiver_device, sender_eth_core, receiver_eth_core, test_config, MAX_BUFFER_SIZE));
                test_config = BankedConfig{
                    .num_pages = 200,
                    .size_bytes = 200 * 2 * 32 * 32,
                    .page_size_bytes = 2 * 32 * 32,
                    .input_buffer_type = BufferType::DRAM,
                    .output_buffer_type = BufferType::L1};
                ASSERT_TRUE(fd_unit_tests::erisc::kernels::chip_to_chip_interleaved_buffer_transfer(
                    sender_device,
                    receiver_device,
                    sender_eth_core,
                    receiver_eth_core,
                    test_config,
                    test_config.page_size_bytes));
                ASSERT_TRUE(fd_unit_tests::erisc::kernels::chip_to_chip_interleaved_buffer_transfer(
                    sender_device, receiver_device, sender_eth_core, receiver_eth_core, test_config, MAX_BUFFER_SIZE));
            }
        }
    }
}
