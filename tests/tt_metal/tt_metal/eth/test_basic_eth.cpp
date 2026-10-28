// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <fmt/base.h>
#include <fmt/format.h>
#include <gtest/gtest.h>
#include <array>
#include <chrono>
#include <cstddef>
#include <tt-umd/types/core_coordinates.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <cstdint>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <thread>
#include <unordered_set>
#include <utility>
#include <variant>
#include <vector>

#include <tt-metalium/buffer.hpp>
#include <tt-metalium/buffer_types.hpp>
#include "command_queue_fixture.hpp"
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/device.hpp>
#include "device_fixture.hpp"
#include "mesh_dispatch_fixture.hpp"
#include <tt-metalium/hal.hpp>
#include <tt-metalium/hal_types.hpp>
#include "llrt.hpp"
#include <tt-logger/tt-logger.hpp>
#include "multi_device_fixture.hpp"
#include <tt-metalium/program.hpp>
#include <tt_stl/span.hpp>
#include "tt_metal/test_utils/stimulus.hpp"
#include <tt-umd/types/arch.hpp>
#include "eth_test_common.hpp"
#include "impl/context/metal_env_accessor.hpp"
#include "distributed/mesh_device_impl.hpp"
#include "impl/program/slow_dispatch.hpp"

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
    tt_metal::EthernetConfig ethernet_config = tt_metal::EthernetConfig{.noc = tt_metal::NOC::NOC_0}) {
    bool pass = true;
    ////////////////////////////////////////////////////////////////////////////
    //                      Application Setup
    ////////////////////////////////////////////////////////////////////////////
    distributed::MeshWorkload workload;
    auto zero_coord = distributed::MeshCoordinate(0, 0);
    auto device_range = distributed::MeshCoordinateRange(zero_coord, zero_coord);
    tt_metal::Program program = tt_metal::CreateProgram(*mesh_device);
    auto* device = mesh_device->impl().get_devices()[0];

    distributed::DeviceLocalBufferConfig dram_config{
        .page_size = byte_size, .buffer_type = tt::tt_metal::BufferType::DRAM};
    distributed::ReplicatedBufferConfig buffer_config{.size = byte_size};

    auto input_dram_buffer = distributed::MeshBuffer::create(buffer_config, dram_config, mesh_device.get());
    uint32_t dram_byte_address = input_dram_buffer->address();
    auto eth_noc_xy = device->ethernet_core_from_logical_core(eth_reader_core);
    eth_test_common::set_arch_specific_eth_config(ethernet_config);
    log_info(
        tt::LogTest,
        "Device {}: reading {} bytes from dram bank 0 addr {} to ethernet core {} addr {} risc {} noc {}",
        device->id(),
        byte_size,
        dram_byte_address,
        eth_reader_core.str(),
        eth_l1_byte_address,
        (uint32_t)ethernet_config.processor,
        (uint32_t)ethernet_config.noc);

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

    workload.add_program(device_range, std::move(program));
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
    tt_metal::EthernetConfig ethernet_config = tt_metal::EthernetConfig{.noc = tt_metal::NOC::NOC_0}) {
    bool pass = true;
    ////////////////////////////////////////////////////////////////////////////
    //                      Application Setup
    ////////////////////////////////////////////////////////////////////////////
    distributed::MeshWorkload workload;
    auto zero_coord = distributed::MeshCoordinate(0, 0);
    auto device_range = distributed::MeshCoordinateRange(zero_coord, zero_coord);
    tt_metal::Program program = tt_metal::CreateProgram(*mesh_device);
    auto* device = mesh_device->impl().get_devices()[0];

    distributed::DeviceLocalBufferConfig dram_config{
        .page_size = byte_size, .buffer_type = tt::tt_metal::BufferType::DRAM};
    distributed::ReplicatedBufferConfig buffer_config{.size = byte_size};

    auto output_dram_buffer = distributed::MeshBuffer::create(buffer_config, dram_config, mesh_device.get());
    uint32_t dram_byte_address = output_dram_buffer->address();
    auto eth_noc_xy = device->ethernet_core_from_logical_core(eth_writer_core);
    eth_test_common::set_arch_specific_eth_config(ethernet_config);
    log_info(
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

    workload.add_program(device_range, std::move(program));
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
    tt_metal::EthernetConfig reader_eth_config,
    tt_metal::EthernetConfig writer_eth_config) {
    bool pass = true;

    distributed::MeshWorkload workload;
    auto zero_coord = distributed::MeshCoordinate(0, 0);
    auto device_range = distributed::MeshCoordinateRange(zero_coord, zero_coord);
    tt_metal::Program program = tt_metal::CreateProgram(*mesh_device);
    auto* device = mesh_device->impl().get_devices()[0];
    auto& cq = mesh_device->mesh_command_queue();

    distributed::DeviceLocalBufferConfig dram_config{.page_size = byte_size, .buffer_type = tt_metal::BufferType::DRAM};
    distributed::ReplicatedBufferConfig buffer_config{.size = byte_size};

    auto reader_dram_buffer = distributed::MeshBuffer::create(buffer_config, dram_config, mesh_device.get());
    auto writer_dram_buffer = distributed::MeshBuffer::create(buffer_config, dram_config, mesh_device.get());

    eth_test_common::set_arch_specific_eth_config(reader_eth_config);
    eth_test_common::set_arch_specific_eth_config(writer_eth_config);

    log_info(
        tt::LogTest,
        "Device {}: reading {} bytes from dram bank 0 addr {} to ethernet core {} addr {}",
        device->id(),
        byte_size,
        reader_dram_buffer->address(),
        logical_eth_core.str(),
        eth_dst_l1_address);
    log_info(
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

    workload.add_program(device_range, std::move(program));
    distributed::EnqueueMeshWorkload(cq, workload, false);
    distributed::Finish(cq);

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

namespace {
namespace CMAKE_UNIQUE_NAMESPACE {

constexpr const char* SPIN_UNTIL_L1_FLAG_KERNEL =
    "tests/tt_metal/tt_metal/test_kernels/dataflow/unit_tests/erisc/spin_until_l1_flag.cpp";

// Labels returned by llrt::eth_fw_stage_to_string().
constexpr const char* STAGE_APP_FW = "APP_FW";
constexpr const char* STAGE_KERNEL = "KERNEL";

// Returns nullptr when the eth firmware stage breadcrumb can be observed, otherwise why it can't.
const char* eth_fw_stage_skip_reason(MetalEnvImpl& env) {
    if (env.get_cluster().get_target_device_type() != tt::TargetDevice::Silicon) {
        return "eth FW stage breadcrumb needs silicon";
    }
    if (env.get_hal().get_dev_size(HalProgrammableCoreType::ACTIVE_ETH, HalL1MemAddrType::FW_STAGE) == 0) {
        return "eth FW stage breadcrumb is not mapped on this arch";
    }
    return nullptr;
}

// Decoded eth firmware stage of each processor on an ethernet core, indexed by processor.
std::vector<std::string> read_eth_fw_stages(
    MetalEnvImpl& env, IDevice* device, const CoreCoord& logical_eth_core, HalProgrammableCoreType core_type) {
    const CoreCoord virtual_core = device->ethernet_core_from_logical_core(logical_eth_core);
    const std::vector<uint32_t> raw_stages =
        llrt::read_eth_fw_stages(env.get_hal(), env.get_cluster(), device->id(), virtual_core, core_type);
    std::vector<std::string> stages;
    stages.reserve(raw_stages.size());
    for (const uint32_t stage : raw_stages) {
        stages.push_back(llrt::eth_fw_stage_to_string(stage));
    }
    return stages;
}

std::array<std::pair<HalProgrammableCoreType, std::vector<CoreCoord>>, 2> eth_cores_by_type(IDevice* device) {
    const auto active = device->get_active_ethernet_cores(true);
    const auto inactive = device->get_inactive_ethernet_cores();
    return {{
        {HalProgrammableCoreType::ACTIVE_ETH, {active.begin(), active.end()}},
        {HalProgrammableCoreType::IDLE_ETH, {inactive.begin(), inactive.end()}},
    }};
}

}  // namespace CMAKE_UNIQUE_NAMESPACE
}  // namespace

namespace tt::tt_metal {

TEST_F(UnitMeshCQSingleCardProgramFixture, ActiveEthKernelsNocReadNoSend) {
    if (not this->slow_dispatch_) {
        GTEST_SKIP() << "fast dispatch to active ethernet has been removed";
    }
    using namespace CMAKE_UNIQUE_NAMESPACE;
    const size_t src_eth_l1_byte_address = tt::tt_metal::MetalContext::instance().hal().get_dev_addr(
        HalProgrammableCoreType::ACTIVE_ETH, HalL1MemAddrType::UNRESERVED);
    const auto erisc_count =
        tt::tt_metal::MetalContext::instance().hal().get_num_risc_processors(HalProgrammableCoreType::ACTIVE_ETH);

    for (const auto& mesh_device : devices_) {
        auto* device = mesh_device->impl().get_devices()[0];
        for (const auto& eth_core : device->get_active_ethernet_cores(true)) {
            for (uint32_t erisc_idx = 0; erisc_idx < erisc_count; ++erisc_idx) {
                const auto ethernet_config = tt_metal::EthernetConfig{
                    .noc = static_cast<NOC>(erisc_idx), .processor = static_cast<DataMovementProcessor>(erisc_idx)};
                ASSERT_TRUE(unit_tests::erisc::kernels::reader_kernel_no_send(
                    static_cast<UnitMeshCQSingleCardProgramFixture*>(this),
                    mesh_device,
                    WORD_SIZE,
                    src_eth_l1_byte_address,
                    eth_core,
                    ethernet_config));
                ASSERT_TRUE(unit_tests::erisc::kernels::reader_kernel_no_send(
                    static_cast<UnitMeshCQSingleCardProgramFixture*>(this),
                    mesh_device,
                    WORD_SIZE * 1024,
                    src_eth_l1_byte_address,
                    eth_core,
                    ethernet_config));
                ASSERT_TRUE(unit_tests::erisc::kernels::reader_kernel_no_send(
                    static_cast<UnitMeshCQSingleCardProgramFixture*>(this),
                    mesh_device,
                    WORD_SIZE * 2048,
                    src_eth_l1_byte_address,
                    eth_core,
                    ethernet_config));
            }
        }
    }
}

TEST_F(UnitMeshCQSingleCardProgramFixture, ActiveEthKernelsNocWriteNoReceive) {
    if (not this->slow_dispatch_) {
        GTEST_SKIP() << "fast dispatch to active ethernet has been removed";
    }
    if (arch_ == ARCH::BLACKHOLE) {
        GTEST_SKIP() << "See GH Issue #18384";
    }
    using namespace CMAKE_UNIQUE_NAMESPACE;
    const size_t src_eth_l1_byte_address = tt::tt_metal::MetalContext::instance().hal().get_dev_addr(
        HalProgrammableCoreType::ACTIVE_ETH, HalL1MemAddrType::UNRESERVED);
    const auto erisc_count =
        tt::tt_metal::MetalContext::instance().hal().get_num_risc_processors(HalProgrammableCoreType::ACTIVE_ETH);

    for (const auto& mesh_device : devices_) {
        auto* device = mesh_device->impl().get_devices()[0];
        for (const auto& eth_core : device->get_active_ethernet_cores(true)) {
            for (uint32_t erisc_idx = 0; erisc_idx < erisc_count; ++erisc_idx) {
                const auto ethernet_config = tt_metal::EthernetConfig{
                    .noc = static_cast<tt_metal::NOC>(erisc_idx),
                    .processor = static_cast<DataMovementProcessor>(erisc_idx)};
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
                    eth_core,
                    ethernet_config));
            }
        }
    }
}

TEST_F(N300MeshDeviceFixture, ActiveEthKernelsNocReadNoSend) {
    if (not this->slow_dispatch_) {
        GTEST_SKIP() << "fast dispatch to active ethernet has been removed";
    }
    using namespace CMAKE_UNIQUE_NAMESPACE;
    const auto& mesh_device_0 = devices_.at(0);
    const auto& mesh_device_1 = devices_.at(1);
    auto* const device_0 = mesh_device_0->impl().get_devices()[0];
    auto* const device_1 = mesh_device_1->impl().get_devices()[0];

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
    if (not this->slow_dispatch_) {
        GTEST_SKIP() << "fast dispatch to active ethernet has been removed";
    }
    using namespace CMAKE_UNIQUE_NAMESPACE;
    const auto& mesh_device_0 = devices_.at(0);
    const auto& mesh_device_1 = devices_.at(1);
    auto* const device_0 = mesh_device_0->impl().get_devices()[0];
    auto* const device_1 = mesh_device_1->impl().get_devices()[0];

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
    auto* const device = mesh_device->impl().get_devices()[0];

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
    auto* const device = mesh_device->impl().get_devices()[0];

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
    auto* const device = mesh_device->impl().get_devices()[0];

    using namespace CMAKE_UNIQUE_NAMESPACE;
    uint32_t read_write_size_bytes = WORD_SIZE * 2048;
    uint32_t reader_dst_address =
        MetalContext::instance().hal().get_dev_addr(HalProgrammableCoreType::IDLE_ETH, HalL1MemAddrType::UNRESERVED);
    uint32_t writer_src_address = reader_dst_address + read_write_size_bytes;
    tt_metal::EthernetConfig erisc0_ethernet_config{
        .eth_mode = Eth::IDLE, .noc = tt_metal::NOC::NOC_0, .processor = tt_metal::DataMovementProcessor::RISCV_0};
    tt_metal::EthernetConfig erisc1_ethernet_config{
        .eth_mode = Eth::IDLE, .noc = tt_metal::NOC::NOC_1, .processor = tt_metal::DataMovementProcessor::RISCV_1};

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

TEST_F(BlackholeSingleCardFixture, ActiveEthPtpTraceStamped) {
    auto& env = MetalEnvAccessor(MetalContext::instance().get_env()).impl();

    // PTP stamping is compiled out under watcher, outside 2-erisc mode, and on older eth FW.
    const char* disabled = nullptr;
    if (env.get_rtoptions().get_watcher_enabled()) {
        disabled = "watcher build";
    } else if (!env.get_rtoptions().get_enable_2_erisc_mode()) {
        disabled = "single-erisc mode";
    } else if (!env.get_rtoptions().get_eth_ptp_trace()) {
        disabled = "eth FW predates debug_buf_t::scratchpad";
    }
    if (disabled != nullptr) {
        GTEST_SKIP() << "PTP trace is compiled out: " << disabled;
    }

    const auto trace_addr =
        env.get_hal().get_dev_addr(HalProgrammableCoreType::ACTIVE_ETH, HalL1MemAddrType::ETH_PTP_TRACE);

    // Must match AERISC_PTP_TRACE_MAGIC in eth_fw_api.h.
    constexpr uint32_t expected_magic = 0x1234ABCD;
    enum : size_t { kMagic, kRunCount, kEntryLo, kEntryHi, kExitLo, kExitHi, kExitValid, kNumWords };

    for (const auto& mesh_device : devices_) {
        auto* device = mesh_device->impl().get_devices()[0];
        for (const auto& eth_core : device->get_active_ethernet_cores(true)) {
            const auto eth_noc_xy = device->ethernet_core_from_logical_core(eth_core);
            const auto trace =
                env.get_cluster().read_core(device->id(), eth_noc_xy, trace_addr, kNumWords * sizeof(uint32_t));

            EXPECT_EQ(trace[kMagic], expected_magic) << "no PTP trace on eth core " << eth_noc_xy.str();
            if (trace[kMagic] != expected_magic) {
                continue;
            }

            const uint64_t entry = (static_cast<uint64_t>(trace[kEntryHi]) << 32) | trace[kEntryLo];
            EXPECT_NE(entry, 0u) << "empty entry stamp on " << eth_noc_xy.str();
            if (trace[kExitValid] != 0) {
                const uint64_t exit = (static_cast<uint64_t>(trace[kExitHi]) << 32) | trace[kExitLo];
                EXPECT_GT(exit, entry) << "exit precedes entry on " << eth_noc_xy.str();
            }
        }
    }
}

TEST(EthFwStage, DecodesStageValues) {
    EXPECT_EQ(llrt::eth_fw_stage_to_string(0), "UNKNOWN/BASE_FW");
    EXPECT_EQ(llrt::eth_fw_stage_to_string(0x45534142), "BASE_FW");
    EXPECT_EQ(llrt::eth_fw_stage_to_string(0x494e4946), "APP_FW_INIT");
    EXPECT_EQ(llrt::eth_fw_stage_to_string(0x50554c46), "APP_FW");
    EXPECT_EQ(llrt::eth_fw_stage_to_string(0x4c4e524b), "KERNEL");
    EXPECT_EQ(llrt::eth_fw_stage_to_string(0x494e4942), "BASE_INIT");
    EXPECT_EQ(llrt::eth_fw_stage_to_string(0x47534d42), "BASE_MSG");
    EXPECT_EQ(llrt::eth_fw_stage_to_string(0x49504142), "API_REQUEST");
    EXPECT_EQ(llrt::eth_fw_stage_to_string(0x44434241), "0x44434241 ('ABCD')");
    EXPECT_EQ(llrt::eth_fw_stage_to_string(0x00000001), "0x00000001 ('....')");
}

TEST_F(BlackholeSingleCardFixture, EthFwStageIdleInAppFw) {
    using namespace CMAKE_UNIQUE_NAMESPACE;
    auto& env = MetalEnvAccessor(MetalContext::instance().get_env()).impl();
    if (const char* reason = eth_fw_stage_skip_reason(env)) {
        GTEST_SKIP() << reason;
    }

    for (const auto& mesh_device : devices_) {
        auto* device = mesh_device->impl().get_devices()[0];
        for (const auto& [core_type, logical_cores] : eth_cores_by_type(device)) {
            for (const auto& logical_core : logical_cores) {
                const auto stages = read_eth_fw_stages(env, device, logical_core, core_type);
                for (size_t processor = 0; processor < stages.size(); processor++) {
                    EXPECT_EQ(stages[processor], STAGE_APP_FW) << "device " << device->id() << " eth core "
                                                               << logical_core.str() << " processor " << processor;
                }
            }
        }
    }
}

// Triage reads the raw stage words as one per physical erisc. In single-erisc mode metal FW runs on erisc1 while
// base FW keeps erisc0, so metal must stamp erisc1's word and leave erisc0's to base FW.
TEST_F(BlackholeSingleCardFixture, EthFwStageSlotsFollowPhysicalErisc) {
    using namespace CMAKE_UNIQUE_NAMESPACE;
    auto& env = MetalEnvAccessor(MetalContext::instance().get_env()).impl();
    if (const char* reason = eth_fw_stage_skip_reason(env)) {
        GTEST_SKIP() << reason;
    }
    const auto& hal = env.get_hal();
    constexpr auto core_type = HalProgrammableCoreType::ACTIVE_ETH;
    const uint32_t num_slots = hal.get_dev_size(core_type, HalL1MemAddrType::FW_STAGE) / sizeof(uint32_t);
    std::vector<bool> metal_slot(num_slots, false);
    for (uint32_t processor = 0; processor < hal.get_num_risc_processors(core_type); processor++) {
        metal_slot.at(hal.get_active_eth_physical_erisc(processor)) = true;
    }

    for (const auto& mesh_device : devices_) {
        auto* device = mesh_device->impl().get_devices()[0];
        for (const auto& logical_core : device->get_active_ethernet_cores(true)) {
            const CoreCoord virtual_core = device->ethernet_core_from_logical_core(logical_core);
            std::vector<uint32_t> slots(num_slots, 0);
            env.get_cluster().read_core(
                slots.data(),
                num_slots * sizeof(uint32_t),
                tt_cxy_pair(device->id(), virtual_core),
                hal.get_dev_noc_addr(core_type, HalL1MemAddrType::FW_STAGE));
            for (uint32_t slot = 0; slot < num_slots; slot++) {
                const std::string stage = llrt::eth_fw_stage_to_string(slots[slot]);
                const auto where = fmt::format("device {} eth core {} erisc{}", device->id(), logical_core.str(), slot);
                if (metal_slot[slot]) {
                    EXPECT_EQ(stage, STAGE_APP_FW) << where;
                } else {
                    EXPECT_NE(stage, "APP_FW_INIT") << where;
                    EXPECT_NE(stage, STAGE_APP_FW) << where;
                    EXPECT_NE(stage, STAGE_KERNEL) << where;
                }
            }
        }
    }
}

TEST_F(BlackholeSingleCardFixture, EthFwStageKernelWhileKernelRuns) {
    using namespace CMAKE_UNIQUE_NAMESPACE;
    auto& env = MetalEnvAccessor(MetalContext::instance().get_env()).impl();
    if (const char* reason = eth_fw_stage_skip_reason(env)) {
        GTEST_SKIP() << reason;
    }
    const auto& hal = env.get_hal();
    auto& cluster = env.get_cluster();
    constexpr auto ack_timeout = std::chrono::seconds(10);

    for (const auto& mesh_device : devices_) {
        auto* device = mesh_device->impl().get_devices()[0];
        for (const auto& [core_type, logical_cores] : eth_cores_by_type(device)) {
            if (logical_cores.empty()) {
                continue;
            }
            const uint32_t num_processors = hal.get_num_risc_processors(core_type);
            const uint32_t l1_base = hal.get_dev_addr(core_type, HalL1MemAddrType::UNRESERVED);
            const uint32_t l1_alignment = hal.get_alignment(HalMemType::L1);
            const auto ack_addr = [&](uint32_t processor) { return l1_base + (2 * processor) * l1_alignment; };
            const auto release_addr = [&](uint32_t processor) { return l1_base + (2 * processor + 1) * l1_alignment; };

            const std::vector<uint32_t> zeros(2 * num_processors * l1_alignment / sizeof(uint32_t), 0);
            for (const auto& logical_core : logical_cores) {
                cluster.write_core(device->id(), device->ethernet_core_from_logical_core(logical_core), zeros, l1_base);
            }

            const CoreRangeSet core_ranges{ttsl::Span<const CoreCoord>(logical_cores)};
            Program program = CreateProgram();
            for (uint32_t processor = 0; processor < num_processors; processor++) {
                EthernetConfig config;
                config.eth_mode = core_type == HalProgrammableCoreType::IDLE_ETH ? Eth::IDLE : Eth::SENDER;
                config.noc = static_cast<NOC>(processor);
                config.processor = static_cast<DataMovementProcessor>(processor);
                eth_test_common::set_arch_specific_eth_config(config);
                const auto kernel = CreateKernel(program, SPIN_UNTIL_L1_FLAG_KERNEL, core_ranges, config);
                SetRuntimeArgs(program, kernel, core_ranges, {ack_addr(processor), release_addr(processor)});
            }
            slow_dispatch::LaunchProgramAsync(*device, program, /*force_slow_dispatch=*/false);

            const auto wait_for_ack = [&](const CoreCoord& virtual_core, uint32_t processor) {
                const auto deadline = std::chrono::steady_clock::now() + ack_timeout;
                while (std::chrono::steady_clock::now() < deadline) {
                    if (cluster.read_core(device->id(), virtual_core, ack_addr(processor), sizeof(uint32_t))[0] != 0) {
                        return true;
                    }
                    std::this_thread::sleep_for(std::chrono::milliseconds(1));
                }
                return false;
            };
            for (const auto& logical_core : logical_cores) {
                const CoreCoord virtual_core = device->ethernet_core_from_logical_core(logical_core);
                for (uint32_t processor = 0; processor < num_processors; processor++) {
                    EXPECT_TRUE(wait_for_ack(virtual_core, processor))
                        << "kernel never started on device " << device->id() << " eth core " << logical_core.str()
                        << " processor " << processor;
                }
                const auto stages = read_eth_fw_stages(env, device, logical_core, core_type);
                for (size_t processor = 0; processor < stages.size(); processor++) {
                    EXPECT_EQ(stages[processor], STAGE_KERNEL) << "device " << device->id() << " eth core "
                                                               << logical_core.str() << " processor " << processor;
                }
            }

            const std::vector<uint32_t> release{1};
            for (const auto& logical_core : logical_cores) {
                const CoreCoord virtual_core = device->ethernet_core_from_logical_core(logical_core);
                for (uint32_t processor = 0; processor < num_processors; processor++) {
                    cluster.write_core(device->id(), virtual_core, release, release_addr(processor));
                }
            }
            slow_dispatch::WaitProgramDone(*device, program);

            for (const auto& logical_core : logical_cores) {
                const auto stages = read_eth_fw_stages(env, device, logical_core, core_type);
                for (size_t processor = 0; processor < stages.size(); processor++) {
                    EXPECT_EQ(stages[processor], STAGE_APP_FW)
                        << "device " << device->id() << " eth core " << logical_core.str() << " processor " << processor
                        << " after the kernel finished";
                }
            }
        }
    }
}

}  // namespace tt::tt_metal
