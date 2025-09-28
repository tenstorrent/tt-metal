// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <fmt/base.h>
#include <gtest/gtest.h>
#include <stddef.h>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <cstdint>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <unordered_set>
#include <variant>
#include <vector>

#include <tt-metalium/buffer.hpp>
#include <tt-metalium/buffer_types.hpp>
#include "command_queue_fixture.hpp"
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/data_types.hpp>
#include <tt-metalium/device.hpp>
#include "device_fixture.hpp"
#include "mesh_dispatch_fixture.hpp"
#include <tt-metalium/hal.hpp>
#include <tt-metalium/hal_types.hpp>
#include <tt-metalium/kernel_types.hpp>
#include "llrt.hpp"
#include <tt-logger/tt-logger.hpp>
#include "multi_device_fixture.hpp"
#include <tt-metalium/program.hpp>
#include <tt_stl/span.hpp>
#include "tt_metal/test_utils/stimulus.hpp"
#include <umd/device/types/arch.hpp>

using namespace tt;
using namespace tt::tt_metal;
using namespace tt::test_utils;
// using namespace tt::test_utils::df;

namespace {
namespace CMAKE_UNIQUE_NAMESPACE {
constexpr std::int32_t WORD_SIZE = 16;  // 16 bytes per eth send packet
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
    tt_metal::MeshDispatchFixture* fixture,
    const std::shared_ptr<tt_metal::distributed::MeshDevice>& mesh_device,
    const size_t& byte_size,
    const size_t& eth_l1_byte_address,
    const CoreCoord& eth_reader_core,
    const tt_metal::EthernetConfig& ethernet_config = tt_metal::EthernetConfig{.noc = tt_metal::NOC::NOC_0}) {
    bool pass = true;
    ////////////////////////////////////////////////////////////////////////////
    //                      Application Setup
    ////////////////////////////////////////////////////////////////////////////
    distributed::MeshWorkload workload;
    auto zero_coord = distributed::MeshCoordinate(0, 0);
    auto device_range = distributed::MeshCoordinateRange(zero_coord, zero_coord);
    tt_metal::Program program = tt_metal::Program();
    auto device = mesh_device->get_devices()[0];

    distributed::DeviceLocalBufferConfig dram_config{
        .page_size = byte_size, .buffer_type = tt::tt_metal::BufferType::DRAM};
    distributed::ReplicatedBufferConfig buffer_config{.size = byte_size};

    auto input_dram_buffer = distributed::MeshBuffer::create(buffer_config, dram_config, mesh_device.get());
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
    fixture->WriteBuffer(mesh_device, input_dram_buffer, inputs);

    // Clear expected value at ethernet L1 address
    std::vector<uint32_t> all_zeros(inputs.size(), 0);
    tt::tt_metal::MetalContext::instance().get_cluster().write_core(
        device->id(), eth_noc_xy, all_zeros, eth_l1_byte_address);

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

    distributed::AddProgramToMeshWorkload(workload, std::move(program), device_range);
    fixture->RunProgram(mesh_device, workload);

    auto readback_vec = tt::tt_metal::MetalContext::instance().get_cluster().read_core(
        device->id(), eth_noc_xy, eth_l1_byte_address, byte_size);
    pass &= (readback_vec == inputs);
    if (not pass) {
        std::cout << "Mismatch at Core: " << eth_noc_xy.str() << std::endl;
    }
    return pass;
}

bool writer_kernel_no_receive(
    tt_metal::MeshDispatchFixture* fixture,
    const std::shared_ptr<tt_metal::distributed::MeshDevice>& mesh_device,
    const size_t& byte_size,
    const size_t& eth_l1_byte_address,
    const CoreCoord& eth_writer_core,
    const tt_metal::EthernetConfig& ethernet_config = tt_metal::EthernetConfig{.noc = tt_metal::NOC::NOC_0}) {
    bool pass = true;
    ////////////////////////////////////////////////////////////////////////////
    //                      Application Setup
    ////////////////////////////////////////////////////////////////////////////
    distributed::MeshWorkload workload;
    auto zero_coord = distributed::MeshCoordinate(0, 0);
    auto device_range = distributed::MeshCoordinateRange(zero_coord, zero_coord);
    tt_metal::Program program = tt_metal::Program();
    auto device = mesh_device->get_devices()[0];

    distributed::DeviceLocalBufferConfig dram_config{
        .page_size = byte_size, .buffer_type = tt::tt_metal::BufferType::DRAM};
    distributed::ReplicatedBufferConfig buffer_config{.size = byte_size};

    auto output_dram_buffer = distributed::MeshBuffer::create(buffer_config, dram_config, mesh_device.get());
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
    tt::tt_metal::MetalContext::instance().get_cluster().write_core(
        device->id(), eth_noc_xy, inputs, eth_l1_byte_address);

    // Clear expected value at ethernet L1 address
    std::vector<uint32_t> all_zeros(inputs.size(), 0);
    fixture->WriteBuffer(mesh_device, output_dram_buffer, all_zeros);

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

    distributed::AddProgramToMeshWorkload(workload, std::move(program), device_range);
    fixture->RunProgram(mesh_device, workload);

    std::vector<uint32_t> readback_vec;
    fixture->ReadBuffer(mesh_device, output_dram_buffer, readback_vec);
    pass &= (readback_vec == inputs);
    if (not pass) {
        std::cout << "Mismatch" << std::endl;
    }
    return pass;
}

bool noc_reader_and_writer_kernels(
    const std::shared_ptr<tt_metal::distributed::MeshDevice>& mesh_device,
    const uint32_t byte_size,
    const uint32_t eth_dst_l1_address,
    const uint32_t eth_src_l1_address,
    const CoreCoord& logical_eth_core,
    const tt_metal::EthernetConfig& reader_eth_config,
    const tt_metal::EthernetConfig& writer_eth_config) {
    bool pass = true;

    distributed::MeshWorkload workload;
    auto zero_coord = distributed::MeshCoordinate(0, 0);
    auto device_range = distributed::MeshCoordinateRange(zero_coord, zero_coord);
    tt_metal::Program program = tt_metal::Program();
    auto device = mesh_device->get_devices()[0];
    auto& cq = mesh_device->mesh_command_queue();

    distributed::DeviceLocalBufferConfig dram_config{.page_size = byte_size, .buffer_type = tt_metal::BufferType::DRAM};
    distributed::ReplicatedBufferConfig buffer_config{.size = byte_size};

    auto reader_dram_buffer = distributed::MeshBuffer::create(buffer_config, dram_config, mesh_device.get());
    auto writer_dram_buffer = distributed::MeshBuffer::create(buffer_config, dram_config, mesh_device.get());

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
    distributed::WriteShard(cq, reader_dram_buffer, reader_inputs, zero_coord);

    auto writer_inputs = generate_uniform_random_vector<uint32_t>(0, 100, byte_size / sizeof(uint32_t));
    tt::tt_metal::MetalContext::instance().get_cluster().write_core(
        device->id(), eth_noc_xy, writer_inputs, eth_src_l1_address);

    // Clear expected values at output locations
    std::vector<uint32_t> all_zeros(byte_size / sizeof(uint32_t), 0);
    tt::tt_metal::MetalContext::instance().get_cluster().write_core(
        device->id(), eth_noc_xy, all_zeros, eth_dst_l1_address);
    distributed::WriteShard(cq, writer_dram_buffer, all_zeros, zero_coord);

    distributed::AddProgramToMeshWorkload(workload, std::move(program), device_range);
    distributed::EnqueueMeshWorkload(cq, workload, false);

    auto eth_readback_vec = tt::tt_metal::MetalContext::instance().get_cluster().read_core(
        device->id(), eth_noc_xy, eth_dst_l1_address, byte_size);
    pass &= (eth_readback_vec == reader_inputs);
    if (not pass) {
        log_info(
            tt::LogTest,
            "Mismatch at eth core: {}, eth kernel read incorrect values from DRAM",
            logical_eth_core.str());
    }
    std::vector<uint32_t> dram_readback_vec;
    distributed::ReadShard(cq, dram_readback_vec, writer_dram_buffer, zero_coord);
    pass &= (dram_readback_vec == writer_inputs);
    if (not pass) {
        log_info(
            tt::LogTest, "Mismatch at eth core: {}, eth kernel wrote incorrect values to DRAM", logical_eth_core.str());
    }

    return pass;
}

}  // namespace unit_tests::erisc::kernels

namespace tt::tt_metal {

TEST_F(UnitMeshCQSingleCardProgramFixture, ActiveEthKernelsNocReadNoSend) {
    using namespace CMAKE_UNIQUE_NAMESPACE;
    const size_t src_eth_l1_byte_address = tt::tt_metal::MetalContext::instance().hal().get_dev_addr(
        HalProgrammableCoreType::ACTIVE_ETH, HalL1MemAddrType::UNRESERVED);

    for (const auto& mesh_device : devices_) {
        auto device = mesh_device->get_devices()[0];
        for (const auto& eth_core : device->get_active_ethernet_cores(true)) {
            ASSERT_TRUE(unit_tests::erisc::kernels::reader_kernel_no_send(
                static_cast<UnitMeshCQSingleCardProgramFixture*>(this),
                mesh_device,
                WORD_SIZE,
                src_eth_l1_byte_address,
                eth_core));
            ASSERT_TRUE(unit_tests::erisc::kernels::reader_kernel_no_send(
                static_cast<UnitMeshCQSingleCardProgramFixture*>(this),
                mesh_device,
                WORD_SIZE * 1024,
                src_eth_l1_byte_address,
                eth_core));
            ASSERT_TRUE(unit_tests::erisc::kernels::reader_kernel_no_send(
                static_cast<UnitMeshCQSingleCardProgramFixture*>(this),
                mesh_device,
                WORD_SIZE * 2048,
                src_eth_l1_byte_address,
                eth_core));
        }
    }
}

TEST_F(UnitMeshCQSingleCardProgramFixture, ActiveEthKernelsNocWriteNoReceive) {
    if (arch_ == ARCH::BLACKHOLE) {
        GTEST_SKIP() << "See GH Issue #18384";
    }
    using namespace CMAKE_UNIQUE_NAMESPACE;
    const size_t src_eth_l1_byte_address = tt::tt_metal::MetalContext::instance().hal().get_dev_addr(
        HalProgrammableCoreType::ACTIVE_ETH, HalL1MemAddrType::UNRESERVED);

    for (const auto& mesh_device : devices_) {
        auto device = mesh_device->get_devices()[0];
        for (const auto& eth_core : device->get_active_ethernet_cores(true)) {
            ASSERT_TRUE(unit_tests::erisc::kernels::writer_kernel_no_receive(
                static_cast<UnitMeshCQSingleCardProgramFixture*>(this),
                mesh_device,
                WORD_SIZE,
                src_eth_l1_byte_address,
                eth_core));
            ASSERT_TRUE(unit_tests::erisc::kernels::writer_kernel_no_receive(
                static_cast<UnitMeshCQSingleCardProgramFixture*>(this),
                mesh_device,
                WORD_SIZE * 1024,
                src_eth_l1_byte_address,
                eth_core));
            ASSERT_TRUE(unit_tests::erisc::kernels::writer_kernel_no_receive(
                static_cast<UnitMeshCQSingleCardProgramFixture*>(this),
                mesh_device,
                WORD_SIZE * 2048,
                src_eth_l1_byte_address,
                eth_core));
        }
    }
}

TEST_F(N300MeshDeviceFixture, ActiveEthKernelsNocReadNoSend) {
    using namespace CMAKE_UNIQUE_NAMESPACE;
    const auto& mesh_device_0 = devices_.at(0);
    const auto& mesh_device_1 = devices_.at(1);
    const auto device_0 = mesh_device_0->get_devices()[0];
    const auto device_1 = mesh_device_1->get_devices()[0];

    const size_t src_eth_l1_byte_address = tt::tt_metal::MetalContext::instance().hal().get_dev_addr(
        HalProgrammableCoreType::ACTIVE_ETH, HalL1MemAddrType::UNRESERVED);

    for (const auto& eth_core : device_0->get_ethernet_sockets(device_1->id())) {
        ASSERT_TRUE(unit_tests::erisc::kernels::reader_kernel_no_send(
            static_cast<N300MeshDeviceFixture*>(this), mesh_device_0, WORD_SIZE, src_eth_l1_byte_address, eth_core));
        ASSERT_TRUE(unit_tests::erisc::kernels::reader_kernel_no_send(
            static_cast<N300MeshDeviceFixture*>(this),
            mesh_device_0,
            WORD_SIZE * 1024,
            src_eth_l1_byte_address,
            eth_core));
        ASSERT_TRUE(unit_tests::erisc::kernels::reader_kernel_no_send(
            static_cast<N300MeshDeviceFixture*>(this),
            mesh_device_0,
            WORD_SIZE * 2048,
            src_eth_l1_byte_address,
            eth_core));
    }

    for (const auto& eth_core : device_1->get_ethernet_sockets(device_0->id())) {
        ASSERT_TRUE(unit_tests::erisc::kernels::reader_kernel_no_send(
            static_cast<N300MeshDeviceFixture*>(this), mesh_device_1, WORD_SIZE, src_eth_l1_byte_address, eth_core));
        ASSERT_TRUE(unit_tests::erisc::kernels::reader_kernel_no_send(
            static_cast<N300MeshDeviceFixture*>(this),
            mesh_device_1,
            WORD_SIZE * 1024,
            src_eth_l1_byte_address,
            eth_core));
        ASSERT_TRUE(unit_tests::erisc::kernels::reader_kernel_no_send(
            static_cast<N300MeshDeviceFixture*>(this),
            mesh_device_1,
            WORD_SIZE * 2048,
            src_eth_l1_byte_address,
            eth_core));
    }
}

TEST_F(N300MeshDeviceFixture, ActiveEthKernelsNocWriteNoReceive) {
    using namespace CMAKE_UNIQUE_NAMESPACE;
    const auto& mesh_device_0 = devices_.at(0);
    const auto& mesh_device_1 = devices_.at(1);
    const auto device_0 = mesh_device_0->get_devices()[0];
    const auto device_1 = mesh_device_1->get_devices()[0];

    const size_t src_eth_l1_byte_address = tt::tt_metal::MetalContext::instance().hal().get_dev_addr(
        HalProgrammableCoreType::ACTIVE_ETH, HalL1MemAddrType::UNRESERVED);

    for (const auto& eth_core : device_0->get_ethernet_sockets(device_1->id())) {
        ASSERT_TRUE(unit_tests::erisc::kernels::writer_kernel_no_receive(
            static_cast<N300MeshDeviceFixture*>(this), mesh_device_0, WORD_SIZE, src_eth_l1_byte_address, eth_core));
        ASSERT_TRUE(unit_tests::erisc::kernels::writer_kernel_no_receive(
            static_cast<N300MeshDeviceFixture*>(this),
            mesh_device_0,
            WORD_SIZE * 1024,
            src_eth_l1_byte_address,
            eth_core));
        ASSERT_TRUE(unit_tests::erisc::kernels::writer_kernel_no_receive(
            static_cast<N300MeshDeviceFixture*>(this),
            mesh_device_0,
            WORD_SIZE * 2048,
            src_eth_l1_byte_address,
            eth_core));
    }

    for (const auto& eth_core : device_1->get_ethernet_sockets(device_0->id())) {
        ASSERT_TRUE(unit_tests::erisc::kernels::writer_kernel_no_receive(
            static_cast<N300MeshDeviceFixture*>(this), mesh_device_1, WORD_SIZE, src_eth_l1_byte_address, eth_core));
        ASSERT_TRUE(unit_tests::erisc::kernels::writer_kernel_no_receive(
            static_cast<N300MeshDeviceFixture*>(this),
            mesh_device_1,
            WORD_SIZE * 1024,
            src_eth_l1_byte_address,
            eth_core));
        ASSERT_TRUE(unit_tests::erisc::kernels::writer_kernel_no_receive(
            static_cast<N300MeshDeviceFixture*>(this),
            mesh_device_1,
            WORD_SIZE * 2048,
            src_eth_l1_byte_address,
            eth_core));
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
    const auto& mesh_device = devices_.at(0);
    const auto device = mesh_device->get_devices()[0];

    using namespace CMAKE_UNIQUE_NAMESPACE;
    uint32_t eth_l1_address =
        MetalContext::instance().hal().get_dev_addr(HalProgrammableCoreType::IDLE_ETH, HalL1MemAddrType::UNRESERVED);
    tt_metal::EthernetConfig noc0_ethernet_config{
        .eth_mode = Eth::IDLE, .noc = tt_metal::NOC::NOC_0, .processor = tt_metal::DataMovementProcessor::RISCV_0};
    tt_metal::EthernetConfig noc1_ethernet_config{
        .eth_mode = Eth::IDLE, .noc = tt_metal::NOC::NOC_1, .processor = tt_metal::DataMovementProcessor::RISCV_0};

    for (const auto& eth_core : device->get_inactive_ethernet_cores()) {
        ASSERT_TRUE(unit_tests::erisc::kernels::reader_kernel_no_send(
            static_cast<BlackholeSingleCardFixture*>(this),
            mesh_device,
            WORD_SIZE * 2048,
            eth_l1_address,
            eth_core,
            noc0_ethernet_config));
        ASSERT_TRUE(unit_tests::erisc::kernels::reader_kernel_no_send(
            static_cast<BlackholeSingleCardFixture*>(this),
            mesh_device,
            WORD_SIZE * 2048,
            eth_l1_address,
            eth_core,
            noc1_ethernet_config));
        ASSERT_TRUE(unit_tests::erisc::kernels::writer_kernel_no_receive(
            static_cast<BlackholeSingleCardFixture*>(this),
            mesh_device,
            WORD_SIZE * 2048,
            eth_l1_address,
            eth_core,
            noc0_ethernet_config));
        ASSERT_TRUE(unit_tests::erisc::kernels::writer_kernel_no_receive(
            static_cast<BlackholeSingleCardFixture*>(this),
            mesh_device,
            WORD_SIZE * 2048,
            eth_l1_address,
            eth_core,
            noc1_ethernet_config));
    }
}

TEST_F(BlackholeSingleCardFixture, IdleEthKernelOnIdleErisc1) {
    const auto& mesh_device = devices_.at(0);
    const auto device = mesh_device->get_devices()[0];

    using namespace CMAKE_UNIQUE_NAMESPACE;
    uint32_t eth_l1_address =
        MetalContext::instance().hal().get_dev_addr(HalProgrammableCoreType::IDLE_ETH, HalL1MemAddrType::UNRESERVED);
    tt_metal::EthernetConfig noc0_ethernet_config{
        .eth_mode = Eth::IDLE, .noc = tt_metal::NOC::NOC_0, .processor = tt_metal::DataMovementProcessor::RISCV_1};
    tt_metal::EthernetConfig noc1_ethernet_config{
        .eth_mode = Eth::IDLE, .noc = tt_metal::NOC::NOC_1, .processor = tt_metal::DataMovementProcessor::RISCV_1};

    for (const auto& eth_core : device->get_inactive_ethernet_cores()) {
        ASSERT_TRUE(unit_tests::erisc::kernels::reader_kernel_no_send(
            static_cast<BlackholeSingleCardFixture*>(this),
            mesh_device,
            WORD_SIZE * 2048,
            eth_l1_address,
            eth_core,
            noc0_ethernet_config));
        ASSERT_TRUE(unit_tests::erisc::kernels::reader_kernel_no_send(
            static_cast<BlackholeSingleCardFixture*>(this),
            mesh_device,
            WORD_SIZE * 2048,
            eth_l1_address,
            eth_core,
            noc1_ethernet_config));
        ASSERT_TRUE(unit_tests::erisc::kernels::writer_kernel_no_receive(
            static_cast<BlackholeSingleCardFixture*>(this),
            mesh_device,
            WORD_SIZE * 2048,
            eth_l1_address,
            eth_core,
            noc0_ethernet_config));
        ASSERT_TRUE(unit_tests::erisc::kernels::writer_kernel_no_receive(
            static_cast<BlackholeSingleCardFixture*>(this),
            mesh_device,
            WORD_SIZE * 2048,
            eth_l1_address,
            eth_core,
            noc1_ethernet_config));
    }
}

TEST_F(BlackholeSingleCardFixture, IdleEthKernelOnBothIdleEriscs) {
    const auto& mesh_device = devices_.at(0);
    const auto device = mesh_device->get_devices()[0];

    using namespace CMAKE_UNIQUE_NAMESPACE;
    uint32_t read_write_size_bytes = WORD_SIZE * 2048;
    uint32_t reader_dst_address =
        MetalContext::instance().hal().get_dev_addr(HalProgrammableCoreType::IDLE_ETH, HalL1MemAddrType::UNRESERVED);
    uint32_t writer_src_address = reader_dst_address + read_write_size_bytes;
    tt_metal::EthernetConfig erisc0_ethernet_config{
        .eth_mode = Eth::IDLE, .noc = tt_metal::NOC::NOC_0, .processor = tt_metal::DataMovementProcessor::RISCV_0};
    tt_metal::EthernetConfig erisc1_ethernet_config{
        .eth_mode = Eth::IDLE, .noc = tt_metal::NOC::NOC_0, .processor = tt_metal::DataMovementProcessor::RISCV_1};

    for (const auto& eth_core : device->get_inactive_ethernet_cores()) {
        ASSERT_TRUE(unit_tests::erisc::kernels::noc_reader_and_writer_kernels(
            mesh_device,
            read_write_size_bytes,
            reader_dst_address,
            writer_src_address,
            eth_core,
            erisc0_ethernet_config,
            erisc1_ethernet_config));
        erisc0_ethernet_config.noc = tt_metal::NOC::NOC_1;
        erisc1_ethernet_config.noc = tt_metal::NOC::NOC_1;
        ASSERT_TRUE(unit_tests::erisc::kernels::noc_reader_and_writer_kernels(
            mesh_device,
            read_write_size_bytes,
            reader_dst_address,
            writer_src_address,
            eth_core,
            erisc0_ethernet_config,
            erisc1_ethernet_config));
    }
}

}  // namespace tt::tt_metal
