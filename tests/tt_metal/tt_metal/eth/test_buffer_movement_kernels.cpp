// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <fmt/base.h>
#include <gtest/gtest.h>
#include <stddef.h>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/math.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <cstdint>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <thread>
#include <tuple>
#include <unordered_set>
#include <variant>
#include <vector>

#include <tt-metalium/assert.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/buffer_types.hpp>
#include "command_queue_fixture.hpp"
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/data_types.hpp>
#include <tt-metalium/device.hpp>
#include "device_fixture.hpp"
#include "dispatch_fixture.hpp"
#include <tt-metalium/kernel_types.hpp>
#include <tt-logger/tt-logger.hpp>
#include "multi_device_fixture.hpp"
#include <tt-metalium/program.hpp>
#include <tt_stl/span.hpp>
#include <tt-metalium/tt_backend_api_types.hpp>
#include "impl/context/metal_context.hpp"
#include "tt_metal/test_utils/stimulus.hpp"
#include "umd/device/types/arch.h"
#include "umd/device/types/xy_pair.h"

using namespace tt;
using namespace tt::test_utils;

namespace {
namespace CMAKE_UNIQUE_NAMESPACE {

struct BankedConfig {
    size_t num_pages = 1;
    size_t size_bytes = 1 * 2 * 32 * 32;
    size_t page_size_bytes = 2 * 32 * 32;
    tt_metal::BufferType input_buffer_type = tt_metal::BufferType::L1;
    tt_metal::BufferType output_buffer_type = tt_metal::BufferType::L1;
    tt::DataFormat l1_data_format = tt::DataFormat::Float16_b;
};
}  // namespace CMAKE_UNIQUE_NAMESPACE
}  // namespace

namespace unit_tests::erisc::kernels {

bool chip_to_chip_dram_buffer_transfer(
    tt_metal::DispatchFixture* fixture,
    tt_metal::IDevice* sender_device,
    tt_metal::IDevice* receiver_device,
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

    // Create dest buffer on receiver device
    auto output_dram_buffer = CreateBuffer(receiver_dram_config);
    uint32_t output_dram_byte_address = output_dram_buffer->address();

    log_info(
        tt::LogTest,
        "Sending {} bytes from device {} dram bank 0 addr {} to device {} dram bank 0 addr {}, using eth core {} and "
        "{}",
        byte_size,
        sender_device->id(),
        input_dram_byte_address,
        receiver_device->id(),
        output_dram_byte_address,
        eth_sender_core.str(),
        eth_receiver_core.str());

    // Generate inputs
    auto inputs = generate_uniform_random_vector<uint32_t>(0, 100, byte_size / sizeof(uint32_t));

    fixture->WriteBuffer(sender_device, input_dram_buffer, inputs);
    uint32_t MAX_BUFFER = tt::tt_metal::MetalContext::instance().hal().get_dev_size(
        tt::tt_metal::HalProgrammableCoreType::ACTIVE_ETH, tt::tt_metal::HalL1MemAddrType::UNRESERVED);

    uint32_t num_loops = (uint32_t)(byte_size / MAX_BUFFER);
    uint32_t remaining_bytes = (uint32_t)(byte_size % MAX_BUFFER);
    // Clear expected value at ethernet L1 address
    std::vector<uint32_t> all_zeros(inputs.size(), 0);

    fixture->WriteBuffer(receiver_device, output_dram_buffer, all_zeros);

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
            0,
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
            0,
            (uint32_t)remaining_bytes,
            (uint32_t)num_loops,
            (uint32_t)MAX_BUFFER,
        });

    ////////////////////////////////////////////////////////////////////////////
    //                      Execute Programs
    ////////////////////////////////////////////////////////////////////////////
    std::thread t1;
    std::thread t2;
    if (fixture->IsSlowDispatch()) {
        t1 = std::thread([&]() { fixture->RunProgram(sender_device, sender_program); });
        t2 = std::thread([&]() { fixture->RunProgram(receiver_device, receiver_program); });
    } else {
        fixture->RunProgram(sender_device, sender_program, true);
        fixture->RunProgram(receiver_device, receiver_program, true);
    }

    fixture->FinishCommands(sender_device);
    fixture->FinishCommands(receiver_device);

    if (fixture->IsSlowDispatch()) {
        t1.join();
        t2.join();
    }

    std::vector<uint32_t> dest_dram_data;
    fixture->ReadBuffer(receiver_device, output_dram_buffer, dest_dram_data);
    pass &= (dest_dram_data == inputs);
    if (not pass) {
        std::cout << "Mismatch" << std::endl;
        std::cout << dest_dram_data[0] << std::endl;
    }
    return pass;
}

bool chip_to_chip_interleaved_buffer_transfer(
    tt_metal::DispatchFixture* fixture,
    tt_metal::IDevice* sender_device,
    tt_metal::IDevice* receiver_device,
    const CoreCoord& eth_sender_core,
    const CoreCoord& eth_receiver_core,
    const CMAKE_UNIQUE_NAMESPACE::BankedConfig& cfg,
    const uint32_t& max_transfer_size) {
    bool pass = true;

    const uint32_t input0_cb_index = 0;
    const uint32_t output_cb_index = 16;

    TT_FATAL(cfg.num_pages * cfg.page_size_bytes == cfg.size_bytes, "Error");
    constexpr uint32_t num_pages_cb = 1;

    ////////////////////////////////////////////////////////////////////////////
    //                      Sender Device
    ////////////////////////////////////////////////////////////////////////////
    tt_metal::Program sender_program = tt_metal::Program();

    auto input_packed = generate_uniform_random_vector<uint32_t>(0, 100, cfg.size_bytes / sizeof(uint32_t));
    /*std::vector<uint32_t> input_packed =
        tt::test_utils::generate_packed_uniform_random_vector<uint32_t, bfloat16>(
            -1.0f,
            1.0f,
            cfg.size_bytes / bfloat16::SIZEOF,
            std::chrono::system_clock::now().time_since_epoch().count());*/

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
    bool input_is_dram = cfg.input_buffer_type == tt_metal::BufferType::DRAM;

    fixture->WriteBuffer(sender_device, input_buffer, input_packed);

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
    bool output_is_dram = cfg.output_buffer_type == tt_metal::BufferType::DRAM;
    std::vector<uint32_t> all_zeros(cfg.size_bytes / sizeof(uint32_t), 0);

    tt_metal::detail::WriteToBuffer(output_buffer, all_zeros);
    fixture->WriteBuffer(receiver_device, output_buffer, all_zeros);

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
    //                      Execute Programs
    ////////////////////////////////////////////////////////////////////////////
    std::thread t1;
    std::thread t2;
    if (fixture->IsSlowDispatch()) {
        t1 = std::thread([&]() { fixture->RunProgram(sender_device, sender_program); });
        t2 = std::thread([&]() { fixture->RunProgram(receiver_device, receiver_program); });
    } else {
        fixture->RunProgram(sender_device, sender_program, true);
        fixture->RunProgram(receiver_device, receiver_program, true);
    }

    fixture->FinishCommands(sender_device);
    fixture->FinishCommands(receiver_device);

    if (fixture->IsSlowDispatch()) {
        t1.join();
        t2.join();
    }

    std::vector<uint32_t> dest_buffer_data;
    tt_metal::detail::ReadFromBuffer(output_buffer, dest_buffer_data);
    fixture->ReadBuffer(receiver_device, output_buffer, dest_buffer_data);
    pass &= input_packed == dest_buffer_data;
    return pass;
}

}  // namespace unit_tests::erisc::kernels

namespace tt::tt_metal {

TEST_F(TwoDeviceFixture, ActiveEthKernelsSendDramBufferChip0ToChip1) {
    if (arch_ == ARCH::BLACKHOLE) {
        GTEST_SKIP() << "See GH Issue #18384";
    }
    const auto& sender_device = devices_.at(0);
    const auto& receiver_device = devices_.at(1);

    for (const auto& sender_eth_core : sender_device->get_active_ethernet_cores(true)) {
        if (not tt::tt_metal::MetalContext::instance().get_cluster().is_ethernet_link_up(
                sender_device->id(), sender_eth_core)) {
            continue;
        }
        CoreCoord receiver_eth_core = std::get<1>(sender_device->get_connected_ethernet_core(sender_eth_core));

        ASSERT_TRUE(unit_tests::erisc::kernels::chip_to_chip_dram_buffer_transfer(
            static_cast<DispatchFixture*>(this),
            sender_device,
            receiver_device,
            sender_eth_core,
            receiver_eth_core,
            16));
        ASSERT_TRUE(unit_tests::erisc::kernels::chip_to_chip_dram_buffer_transfer(
            static_cast<DispatchFixture*>(this),
            sender_device,
            receiver_device,
            sender_eth_core,
            receiver_eth_core,
            1024));
        ASSERT_TRUE(unit_tests::erisc::kernels::chip_to_chip_dram_buffer_transfer(
            static_cast<DispatchFixture*>(this),
            sender_device,
            receiver_device,
            sender_eth_core,
            receiver_eth_core,
            16 * 1024));
        ASSERT_TRUE(unit_tests::erisc::kernels::chip_to_chip_dram_buffer_transfer(
            static_cast<DispatchFixture*>(this),
            sender_device,
            receiver_device,
            sender_eth_core,
            receiver_eth_core,
            1000 * 1024));
    }
}

TEST_F(TwoDeviceFixture, ActiveEthKernelsSendDramBufferChip1ToChip0) {
    if (arch_ == ARCH::BLACKHOLE) {
        GTEST_SKIP() << "See GH Issue #18384";
    }
    const auto& sender_device = devices_.at(1);
    const auto& receiver_device = devices_.at(0);

    for (const auto& sender_eth_core : sender_device->get_active_ethernet_cores(true)) {
        if (not tt::tt_metal::MetalContext::instance().get_cluster().is_ethernet_link_up(
                sender_device->id(), sender_eth_core)) {
            continue;
        }
        CoreCoord receiver_eth_core = std::get<1>(sender_device->get_connected_ethernet_core(sender_eth_core));

        ASSERT_TRUE(unit_tests::erisc::kernels::chip_to_chip_dram_buffer_transfer(
            static_cast<DispatchFixture*>(this),
            sender_device,
            receiver_device,
            sender_eth_core,
            receiver_eth_core,
            16));
        ASSERT_TRUE(unit_tests::erisc::kernels::chip_to_chip_dram_buffer_transfer(
            static_cast<DispatchFixture*>(this),
            sender_device,
            receiver_device,
            sender_eth_core,
            receiver_eth_core,
            1024));
        ASSERT_TRUE(unit_tests::erisc::kernels::chip_to_chip_dram_buffer_transfer(
            static_cast<DispatchFixture*>(this),
            sender_device,
            receiver_device,
            sender_eth_core,
            receiver_eth_core,
            16 * 1024));
        ASSERT_TRUE(unit_tests::erisc::kernels::chip_to_chip_dram_buffer_transfer(
            static_cast<DispatchFixture*>(this),
            sender_device,
            receiver_device,
            sender_eth_core,
            receiver_eth_core,
            1000 * 1024));
    }
}

TEST_F(N300DeviceFixture, ActiveEthKernelsSendInterleavedBufferChip0ToChip1) {
    using namespace CMAKE_UNIQUE_NAMESPACE;
    GTEST_SKIP();
    const auto& sender_device = devices_.at(0);
    const auto& receiver_device = devices_.at(1);
    uint32_t MAX_BUFFER_SIZE =
        MetalContext::instance().hal().get_dev_size(HalProgrammableCoreType::ACTIVE_ETH, HalL1MemAddrType::UNRESERVED);

    for (const auto& sender_eth_core : sender_device->get_active_ethernet_cores(true)) {
        CoreCoord receiver_eth_core = std::get<1>(sender_device->get_connected_ethernet_core(sender_eth_core));

        log_info(
            tt::LogTest,
            "Sending interleaved buffer from device {} to device {}, using eth core {} and {}",
            sender_device->id(),
            receiver_device->id(),
            sender_eth_core.str(),
            receiver_eth_core.str());
        BankedConfig test_config;
        ASSERT_TRUE(unit_tests::erisc::kernels::chip_to_chip_interleaved_buffer_transfer(
            static_cast<DispatchFixture*>(this),
            sender_device,
            receiver_device,
            sender_eth_core,
            receiver_eth_core,
            test_config,
            test_config.page_size_bytes));
        test_config = BankedConfig{.num_pages = 200, .size_bytes = 200 * 2 * 32 * 32, .page_size_bytes = 2 * 32 * 32};

        ASSERT_TRUE(unit_tests::erisc::kernels::chip_to_chip_interleaved_buffer_transfer(
            static_cast<DispatchFixture*>(this),
            sender_device,
            receiver_device,
            sender_eth_core,
            receiver_eth_core,
            test_config,
            test_config.page_size_bytes));
        ASSERT_TRUE(unit_tests::erisc::kernels::chip_to_chip_interleaved_buffer_transfer(
            static_cast<DispatchFixture*>(this),
            sender_device,
            receiver_device,
            sender_eth_core,
            receiver_eth_core,
            test_config,
            MAX_BUFFER_SIZE));
        test_config = BankedConfig{
            .num_pages = 200,
            .size_bytes = 200 * 2 * 32 * 32,
            .page_size_bytes = 2 * 32 * 32,
            .input_buffer_type = BufferType::DRAM,
            .output_buffer_type = BufferType::DRAM};
        ASSERT_TRUE(unit_tests::erisc::kernels::chip_to_chip_interleaved_buffer_transfer(
            static_cast<DispatchFixture*>(this),
            sender_device,
            receiver_device,
            sender_eth_core,
            receiver_eth_core,
            test_config,
            test_config.page_size_bytes));
        ASSERT_TRUE(unit_tests::erisc::kernels::chip_to_chip_interleaved_buffer_transfer(
            static_cast<DispatchFixture*>(this),
            sender_device,
            receiver_device,
            sender_eth_core,
            receiver_eth_core,
            test_config,
            MAX_BUFFER_SIZE));
    }
}

TEST_F(DeviceFixture, ActiveEthKernelsSendInterleavedBufferAllConnectedChips) {
    using namespace CMAKE_UNIQUE_NAMESPACE;
    uint32_t MAX_BUFFER_SIZE =
        MetalContext::instance().hal().get_dev_size(HalProgrammableCoreType::ACTIVE_ETH, HalL1MemAddrType::UNRESERVED);
    for (const auto& sender_device : devices_) {
        for (const auto& receiver_device : devices_) {
            if (sender_device->id() == receiver_device->id()) {
                continue;
            }
            for (const auto& sender_eth_core : sender_device->get_active_ethernet_cores(true)) {
                if (not tt::tt_metal::MetalContext::instance().get_cluster().is_ethernet_link_up(
                        sender_device->id(), sender_eth_core)) {
                    continue;
                }
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

                ASSERT_TRUE(unit_tests::erisc::kernels::chip_to_chip_interleaved_buffer_transfer(
                    static_cast<DispatchFixture*>(this),
                    sender_device,
                    receiver_device,
                    sender_eth_core,
                    receiver_eth_core,
                    test_config,
                    test_config.page_size_bytes));
                ASSERT_TRUE(unit_tests::erisc::kernels::chip_to_chip_interleaved_buffer_transfer(
                    static_cast<DispatchFixture*>(this),
                    sender_device,
                    receiver_device,
                    sender_eth_core,
                    receiver_eth_core,
                    test_config,
                    MAX_BUFFER_SIZE));
                test_config = BankedConfig{
                    .num_pages = 200,
                    .size_bytes = 200 * 2 * 32 * 32,
                    .page_size_bytes = 2 * 32 * 32,
                    .input_buffer_type = BufferType::DRAM,
                    .output_buffer_type = BufferType::L1};
                ASSERT_TRUE(unit_tests::erisc::kernels::chip_to_chip_interleaved_buffer_transfer(
                    static_cast<DispatchFixture*>(this),
                    sender_device,
                    receiver_device,
                    sender_eth_core,
                    receiver_eth_core,
                    test_config,
                    test_config.page_size_bytes));
                ASSERT_TRUE(unit_tests::erisc::kernels::chip_to_chip_interleaved_buffer_transfer(
                    static_cast<DispatchFixture*>(this),
                    sender_device,
                    receiver_device,
                    sender_eth_core,
                    receiver_eth_core,
                    test_config,
                    MAX_BUFFER_SIZE));
            }
        }
    }
}

TEST_F(CommandQueueMultiDeviceProgramFixture, ActiveEthKernelsSendDramBufferAllConnectedChips) {
    if (arch_ == ARCH::BLACKHOLE) {
        GTEST_SKIP() << "See GH Issue #18384";
    }
    for (const auto& sender_device : devices_) {
        for (const auto& receiver_device : devices_) {
            if (sender_device->id() >= receiver_device->id()) {
                continue;
            }
            for (const auto& sender_eth_core : sender_device->get_active_ethernet_cores(true)) {
                if (not tt::tt_metal::MetalContext::instance().get_cluster().is_ethernet_link_up(
                        sender_device->id(), sender_eth_core)) {
                    continue;
                }
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

                ASSERT_TRUE(unit_tests::erisc::kernels::chip_to_chip_dram_buffer_transfer(
                    static_cast<DispatchFixture*>(this),
                    sender_device,
                    receiver_device,
                    sender_eth_core,
                    receiver_eth_core,
                    16));
                ASSERT_TRUE(unit_tests::erisc::kernels::chip_to_chip_dram_buffer_transfer(
                    static_cast<DispatchFixture*>(this),
                    sender_device,
                    receiver_device,
                    sender_eth_core,
                    receiver_eth_core,
                    1024));
                ASSERT_TRUE(unit_tests::erisc::kernels::chip_to_chip_dram_buffer_transfer(
                    static_cast<DispatchFixture*>(this),
                    sender_device,
                    receiver_device,
                    sender_eth_core,
                    receiver_eth_core,
                    16 * 1024));
                ASSERT_TRUE(unit_tests::erisc::kernels::chip_to_chip_dram_buffer_transfer(
                    static_cast<DispatchFixture*>(this),
                    sender_device,
                    receiver_device,
                    sender_eth_core,
                    receiver_eth_core,
                    1000 * 1024));
            }
        }
    }
}

TEST_F(CommandQueueMultiDeviceProgramFixture, ActiveEthKernelsSendInterleavedBufferAllConnectedChips) {
    using namespace CMAKE_UNIQUE_NAMESPACE;
    uint32_t MAX_BUFFER_SIZE =
        MetalContext::instance().hal().get_dev_size(HalProgrammableCoreType::ACTIVE_ETH, HalL1MemAddrType::UNRESERVED);
    for (const auto& sender_device : devices_) {
        for (const auto& receiver_device : devices_) {
            if (sender_device->id() >= receiver_device->id()) {
                continue;
            }
            for (const auto& sender_eth_core : sender_device->get_active_ethernet_cores(true)) {
                if (not tt::tt_metal::MetalContext::instance().get_cluster().is_ethernet_link_up(
                        sender_device->id(), sender_eth_core)) {
                    continue;
                }
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

                ASSERT_TRUE(unit_tests::erisc::kernels::chip_to_chip_interleaved_buffer_transfer(
                    static_cast<DispatchFixture*>(this),
                    sender_device,
                    receiver_device,
                    sender_eth_core,
                    receiver_eth_core,
                    test_config,
                    test_config.page_size_bytes));
                ASSERT_TRUE(unit_tests::erisc::kernels::chip_to_chip_interleaved_buffer_transfer(
                    static_cast<DispatchFixture*>(this),
                    sender_device,
                    receiver_device,
                    sender_eth_core,
                    receiver_eth_core,
                    test_config,
                    MAX_BUFFER_SIZE));
                test_config = BankedConfig{
                    .num_pages = 200,
                    .size_bytes = 200 * 2 * 32 * 32,
                    .page_size_bytes = 2 * 32 * 32,
                    .input_buffer_type = BufferType::DRAM,
                    .output_buffer_type = BufferType::L1};
                ASSERT_TRUE(unit_tests::erisc::kernels::chip_to_chip_interleaved_buffer_transfer(
                    static_cast<DispatchFixture*>(this),
                    sender_device,
                    receiver_device,
                    sender_eth_core,
                    receiver_eth_core,
                    test_config,
                    test_config.page_size_bytes));
                ASSERT_TRUE(unit_tests::erisc::kernels::chip_to_chip_interleaved_buffer_transfer(
                    static_cast<DispatchFixture*>(this),
                    sender_device,
                    receiver_device,
                    sender_eth_core,
                    receiver_eth_core,
                    test_config,
                    MAX_BUFFER_SIZE));
            }
        }
    }
}

}  // namespace tt::tt_metal
