// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <fmt/base.h>
#include <gtest/gtest.h>
#include <cstdlib>
#include <tt-metalium/host_api.hpp>
#include <tt-logger/tt-logger.hpp>
#include <algorithm>
#include <cstdint>
#include <iostream>
#include <iterator>
#include <enchantum/enchantum.hpp>
#include <map>
#include <string>
#include <thread>
#include <tuple>
#include <unordered_set>
#include <utility>
#include <variant>
#include <vector>

#include <tt_stl/assert.hpp>
#include "command_queue_fixture.hpp"
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/data_types.hpp>
#include <tt-metalium/device.hpp>
#include "device_fixture.hpp"
#include "mesh_dispatch_fixture.hpp"
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/hal_types.hpp>
#include "jit_build/build.hpp"
#include <tt-metalium/kernel_types.hpp>
#include "llrt.hpp"
#include "mesh_device.hpp"
#include "multi_device_fixture.hpp"
#include <tt-metalium/program.hpp>
#include <tt_stl/span.hpp>
#include "impl/context/metal_context.hpp"
#include "tt_memory.h"
#include "tt_metal/jit_build/build_env_manager.hpp"
#include "tt_metal/test_utils/stimulus.hpp"
#include <umd/device/types/xy_pair.hpp>
#include "eth_test_common.hpp"

namespace {
namespace CMAKE_UNIQUE_NAMESPACE {
constexpr std::int32_t WORD_SIZE = 16;  // 16 bytes per eth send packet

struct erisc_info_t {
    volatile uint32_t num_bytes;
    volatile uint32_t mode;
    volatile uint32_t reserved_0_;
    volatile uint32_t reserved_1_;
    volatile uint32_t bytes_done;
    volatile uint32_t reserverd_2_;
    volatile uint32_t reserverd_3_;
    volatile uint32_t reserverd_4_;
};
}  // namespace CMAKE_UNIQUE_NAMESPACE
}  // namespace

using namespace tt;
using namespace tt::test_utils;
using namespace tt::tt_metal;

namespace unit_tests::erisc::direct_send {
size_t get_rand_32_byte_aligned_address(const size_t& base, const size_t& max) {
    TT_FATAL(!(base & 0x1F) and !(max & 0x1F), "base and max must be 32-byte aligned");
    size_t word_size = (max >> 5) - (base >> 5);
    return (((rand() % word_size) << 5) + base);
}

template <typename FIXTURE>
bool eth_direct_sender_receiver_kernels(
    FIXTURE* fixture,
    std::shared_ptr<distributed::MeshDevice> sender_mesh_device,
    std::shared_ptr<distributed::MeshDevice> receiver_mesh_device,
    const size_t& byte_size,
    const size_t& src_eth_l1_byte_address,
    const size_t& dst_eth_l1_byte_address,
    const CoreCoord& eth_sender_core,
    const CoreCoord& eth_receiver_core,
    DataMovementProcessor processor = DataMovementProcessor::RISCV_0,
    uint32_t num_bytes_per_send = 16) {
    auto* const sender_device = sender_mesh_device->get_devices()[0];
    auto* const receiver_device = receiver_mesh_device->get_devices()[0];
    bool pass = true;
    log_info(
        tt::LogTest,
        "Sending {} bytes from device {} eth core {} addr {} to device {} eth core {} addr {} processor {}",
        byte_size,
        sender_device->id(),
        eth_sender_core.str(),
        src_eth_l1_byte_address,
        receiver_device->id(),
        eth_receiver_core.str(),
        dst_eth_l1_byte_address,
        processor);
    // Generate inputs
    auto inputs = generate_uniform_random_vector<uint32_t>(0, 100, byte_size / sizeof(uint32_t));
    tt::tt_metal::MetalContext::instance().get_cluster().write_core(
        sender_device->id(),
        sender_device->ethernet_core_from_logical_core(eth_sender_core),
        inputs,
        src_eth_l1_byte_address);

    // Clear expected value at ethernet L1 address
    std::vector<uint32_t> all_zeros(inputs.size(), 0);
    tt::tt_metal::MetalContext::instance().get_cluster().write_core(
        receiver_device->id(),
        receiver_device->ethernet_core_from_logical_core(eth_receiver_core),
        all_zeros,
        dst_eth_l1_byte_address);

    ////////////////////////////////////////////////////////////////////////////
    //                      Sender Device
    ////////////////////////////////////////////////////////////////////////////
    auto zero_coord = distributed::MeshCoordinate(0, 0);
    auto device_range = distributed::MeshCoordinateRange(zero_coord, zero_coord);
    distributed::MeshWorkload sender_workload;
    tt_metal::Program sender_program = tt_metal::Program();

    auto ethernet_config = tt_metal::EthernetConfig{
        .noc = tt_metal::NOC::NOC_0,
        .processor = processor,
        .compile_args = {uint32_t(num_bytes_per_send), uint32_t(num_bytes_per_send >> 4)}};
    eth_test_common::set_arch_specific_eth_config(ethernet_config);

    auto eth_sender_kernel = tt_metal::CreateKernel(
        sender_program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/unit_tests/erisc/eth_l1_direct_send.cpp",
        eth_sender_core,
        ethernet_config);

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
    distributed::MeshWorkload receiver_workload;
    tt_metal::Program receiver_program = tt_metal::Program();

    auto eth_receiver_kernel = tt_metal::CreateKernel(
        receiver_program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/unit_tests/erisc/eth_l1_direct_receive.cpp",
        eth_receiver_core,
        ethernet_config);

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
    std::thread t1;
    std::thread t2;
    if (fixture->IsSlowDispatch()) {
        sender_workload.add_program(device_range, std::move(sender_program));
        receiver_workload.add_program(device_range, std::move(receiver_program));
        t1 = std::thread([&]() { fixture->RunProgram(sender_mesh_device, sender_workload); });
        t2 = std::thread([&]() { fixture->RunProgram(receiver_mesh_device, receiver_workload); });
    } else {
        sender_workload.add_program(device_range, std::move(sender_program));
        receiver_workload.add_program(device_range, std::move(receiver_program));
        fixture->RunProgram(sender_mesh_device, sender_workload, true);
        fixture->RunProgram(receiver_mesh_device, receiver_workload, true);
    }

    fixture->FinishCommands(sender_mesh_device);
    fixture->FinishCommands(receiver_mesh_device);

    if (fixture->IsSlowDispatch()) {
        t1.join();
        t2.join();
    }

    auto readback_vec = tt::tt_metal::MetalContext::instance().get_cluster().read_core(
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

// Tests ethernet direct send/receive from ERISC_L1_UNRESERVED_BASE
bool send_over_eth(
    const std::shared_ptr<distributed::MeshDevice>& sender_mesh_device,
    const std::shared_ptr<distributed::MeshDevice>& receiver_mesh_device,
    const CoreCoord& sender_core,
    const CoreCoord& receiver_core,
    const size_t& byte_size) {
    auto* const sender_device = sender_mesh_device->get_devices()[0];
    auto* const receiver_device = receiver_mesh_device->get_devices()[0];
    log_debug(
        tt::LogTest,
        "Running direct send test with sender chip {} core {}, receiver chip {} core {}, sending {} bytes",
        sender_device->id(),
        sender_core.str(),
        receiver_device->id(),
        receiver_core.str(),
        byte_size);
    std::vector<CoreCoord> eth_cores = {
        CoreCoord(9, 0),
        CoreCoord(1, 0),
        CoreCoord(8, 0),
        CoreCoord(2, 0),
        CoreCoord(9, 6),
        CoreCoord(1, 6),
        CoreCoord(8, 6),
        CoreCoord(2, 6),
        CoreCoord(7, 0),
        CoreCoord(3, 0),
        CoreCoord(6, 0),
        CoreCoord(4, 0),
        CoreCoord(7, 6),
        CoreCoord(3, 6),
        CoreCoord(6, 6),
        CoreCoord(4, 6)};

    // Disable all eth core runtime app flags, zero out data write counter
    std::vector<uint32_t> run_test_app_flag = {0x0};
    uint32_t active_erisc_core_type_idx = tt::tt_metal::MetalContext::instance().hal().get_programmable_core_type_index(
        tt::tt_metal::HalProgrammableCoreType::ACTIVE_ETH);
    uint32_t dm_class_idx = enchantum::to_underlying(tt::tt_metal::HalProcessorClassType::DM);
    uint32_t dm0_type_idx = enchantum::to_underlying(tt::tt_metal::DataMovementProcessor::RISCV_0);
    const auto& erisc_jit_build_config = tt::tt_metal::MetalContext::instance().hal().get_jit_build_config(
        active_erisc_core_type_idx, dm_class_idx, dm0_type_idx);
    uint32_t fw_launch_addr = erisc_jit_build_config.fw_launch_addr;
    uint32_t fw_base_addr = erisc_jit_build_config.fw_base_addr;
    uint32_t erisc_unreserved_base_addr = tt::tt_metal::MetalContext::instance().hal().get_dev_addr(
        tt::tt_metal::HalProgrammableCoreType::ACTIVE_ETH, tt::tt_metal::HalL1MemAddrType::UNRESERVED);
    uint32_t app_sync_info_base_addr = tt::tt_metal::MetalContext::instance().hal().get_dev_addr(
        tt::tt_metal::HalProgrammableCoreType::ACTIVE_ETH, tt::tt_metal::HalL1MemAddrType::APP_SYNC_INFO);
    for (const auto& eth_core : eth_cores) {
        tt::tt_metal::MetalContext::instance().get_cluster().write_core(
            sender_device->id(), eth_core, run_test_app_flag, fw_launch_addr);
        tt::tt_metal::MetalContext::instance().get_cluster().write_core(
            receiver_device->id(), eth_core, run_test_app_flag, fw_launch_addr);
        std::vector<uint32_t> zero = {0, 0, 0, 0, 0, 0, 0, 0};
        tt::tt_metal::MetalContext::instance().get_cluster().write_core(
            sender_device->id(), eth_core, zero, app_sync_info_base_addr);
        tt::tt_metal::MetalContext::instance().get_cluster().write_core(
            receiver_device->id(), eth_core, zero, app_sync_info_base_addr);
    }

    // TODO: is it possible that receiver core app is stil running when we push inputs here???
    auto inputs = generate_uniform_random_vector<uint32_t>(0, 100, byte_size / sizeof(uint32_t));
    tt::tt_metal::MetalContext::instance().get_cluster().write_core(
        sender_device->id(), sender_core, inputs, erisc_unreserved_base_addr);

    // Zero out receiving address to ensure no stale data is causing tests to pass
    std::vector<uint32_t> all_zeros(inputs.size(), 0);
    tt::tt_metal::MetalContext::instance().get_cluster().write_core(
        receiver_device->id(), receiver_core, all_zeros, erisc_unreserved_base_addr);

    std::vector<uint32_t> args_0 = {uint32_t(byte_size), 0};
    tt::tt_metal::MetalContext::instance().get_cluster().write_core(
        sender_device->id(), sender_core, args_0, app_sync_info_base_addr);
    std::vector<uint32_t> args_1 = {uint32_t(byte_size), 1};
    tt::tt_metal::MetalContext::instance().get_cluster().write_core(
        receiver_device->id(), receiver_core, args_1, app_sync_info_base_addr);

    // TODO: this should be updated to use kernel api
    uint32_t active_eth_index = tt_metal::MetalContext::instance().hal().get_programmable_core_type_index(
        tt_metal::HalProgrammableCoreType::ACTIVE_ETH);
    auto sender_firmware_path = tt_metal::BuildEnvManager::get_instance()
                                    .get_firmware_build_state(sender_device->build_id(), active_eth_index, 0, 0)
                                    .get_target_out_path("");
    auto receiver_firmware_path = tt_metal::BuildEnvManager::get_instance()
                                      .get_firmware_build_state(receiver_device->build_id(), active_eth_index, 0, 0)
                                      .get_target_out_path("");
    const ll_api::memory& binary_mem_send = llrt::get_risc_binary(sender_firmware_path);
    const ll_api::memory& binary_mem_receive = llrt::get_risc_binary(receiver_firmware_path);

    for (const auto& eth_core : eth_cores) {
        tt::tt_metal::MetalContext::instance().get_cluster().write_core(
            sender_device->id(), eth_core, binary_mem_send.data(), fw_base_addr);
        tt::tt_metal::MetalContext::instance().get_cluster().write_core(
            receiver_device->id(), eth_core, binary_mem_receive.data(), fw_base_addr);
    }

    // Activate sender core runtime app
    run_test_app_flag = {0x1};
    // send remote first, otherwise eth core may be blocked, very ugly for now...
    if (receiver_device->id() == 1) {
        tt::tt_metal::MetalContext::instance().get_cluster().write_core(
            1, receiver_core, run_test_app_flag, fw_launch_addr);
    } else {
        tt::tt_metal::MetalContext::instance().get_cluster().write_core(
            1, sender_core, run_test_app_flag, fw_launch_addr);
    }
    if (sender_device->id() == 0) {
        tt::tt_metal::MetalContext::instance().get_cluster().write_core(
            0, sender_core, run_test_app_flag, fw_launch_addr);
    } else {
        tt::tt_metal::MetalContext::instance().get_cluster().write_core(
            0, receiver_core, run_test_app_flag, fw_launch_addr);
    }

    bool pass = true;
    auto readback_vec = tt::tt_metal::MetalContext::instance().get_cluster().read_core(
        receiver_device->id(), receiver_core, erisc_unreserved_base_addr, byte_size);
    pass &= (readback_vec == inputs);

    return pass;
}

}  // namespace unit_tests::erisc::direct_send

namespace tt::tt_metal {

TEST_F(N300MeshDeviceFixture, ActiveEthSingleCoreDirectSendChip0ToChip1) {
    using namespace CMAKE_UNIQUE_NAMESPACE;
    const auto& mesh_device_0 = devices_.at(0);
    const auto& mesh_device_1 = devices_.at(1);
    auto* const device_0 = mesh_device_0->get_devices()[0];
    auto* const device_1 = mesh_device_1->get_devices()[0];

    auto send_cores = device_0->get_ethernet_sockets(device_1->id());
    auto receiver_cores = device_1->get_ethernet_sockets(device_0->id());

    if (send_cores.size() != 2 || receiver_cores.size() != 2) {
        GTEST_SKIP() << "Not enough eth cores to run test. Need 2 on each device.";
    }

    CoreCoord sender_core_0 = send_cores[0];
    CoreCoord sender_core_1 = send_cores[1];

    CoreCoord receiver_core_0 = receiver_cores[0];
    CoreCoord receiver_core_1 = receiver_cores[1];

    uint32_t MAX_NUM_WORDS =
        MetalContext::instance().hal().get_dev_size(HalProgrammableCoreType::ACTIVE_ETH, HalL1MemAddrType::UNRESERVED) /
        WORD_SIZE;

    ASSERT_TRUE(unit_tests::erisc::direct_send::send_over_eth(
        mesh_device_0, mesh_device_1, sender_core_0, receiver_core_0, WORD_SIZE));
    ASSERT_TRUE(unit_tests::erisc::direct_send::send_over_eth(
        mesh_device_0, mesh_device_1, sender_core_1, receiver_core_1, WORD_SIZE));
    ASSERT_TRUE(unit_tests::erisc::direct_send::send_over_eth(
        mesh_device_0, mesh_device_1, sender_core_0, receiver_core_0, WORD_SIZE * 256));
    ASSERT_TRUE(unit_tests::erisc::direct_send::send_over_eth(
        mesh_device_0, mesh_device_1, sender_core_1, receiver_core_1, WORD_SIZE * 256));
    ASSERT_TRUE(unit_tests::erisc::direct_send::send_over_eth(
        mesh_device_0, mesh_device_1, sender_core_0, receiver_core_0, WORD_SIZE * 1024));
    ASSERT_TRUE(unit_tests::erisc::direct_send::send_over_eth(
        mesh_device_0, mesh_device_1, sender_core_1, receiver_core_1, WORD_SIZE * 1024));
    ASSERT_TRUE(unit_tests::erisc::direct_send::send_over_eth(
        mesh_device_0, mesh_device_1, sender_core_0, receiver_core_0, WORD_SIZE * MAX_NUM_WORDS));
    ASSERT_TRUE(unit_tests::erisc::direct_send::send_over_eth(
        mesh_device_0, mesh_device_1, sender_core_1, receiver_core_1, WORD_SIZE * MAX_NUM_WORDS));
}

TEST_F(N300MeshDeviceFixture, ActiveEthSingleCoreDirectSendChip1ToChip0) {
    using namespace CMAKE_UNIQUE_NAMESPACE;
    const auto& mesh_device_0 = devices_.at(0);
    const auto& mesh_device_1 = devices_.at(1);
    auto* const device_0 = mesh_device_0->get_devices()[0];
    auto* const device_1 = mesh_device_1->get_devices()[0];

    auto send_cores = device_1->get_ethernet_sockets(device_0->id());
    auto receiver_cores = device_0->get_ethernet_sockets(device_1->id());

    if (send_cores.size() != 2 || receiver_cores.size() != 2) {
        GTEST_SKIP() << "Not enough eth cores to run test. Need 2 on each device.";
    }

    CoreCoord sender_core_0 = send_cores[0];
    CoreCoord sender_core_1 = send_cores[1];

    CoreCoord receiver_core_0 = receiver_cores[0];
    CoreCoord receiver_core_1 = receiver_cores[1];

    uint32_t MAX_NUM_WORDS =
        MetalContext::instance().hal().get_dev_size(HalProgrammableCoreType::ACTIVE_ETH, HalL1MemAddrType::UNRESERVED) /
        WORD_SIZE;

    ASSERT_TRUE(unit_tests::erisc::direct_send::send_over_eth(
        mesh_device_1, mesh_device_0, sender_core_0, receiver_core_0, WORD_SIZE));
    ASSERT_TRUE(unit_tests::erisc::direct_send::send_over_eth(
        mesh_device_1, mesh_device_0, sender_core_1, receiver_core_1, WORD_SIZE));
    ASSERT_TRUE(unit_tests::erisc::direct_send::send_over_eth(
        mesh_device_1, mesh_device_0, sender_core_0, receiver_core_0, WORD_SIZE * 256));
    ASSERT_TRUE(unit_tests::erisc::direct_send::send_over_eth(
        mesh_device_1, mesh_device_0, sender_core_1, receiver_core_1, WORD_SIZE * 256));
    ASSERT_TRUE(unit_tests::erisc::direct_send::send_over_eth(
        mesh_device_1, mesh_device_0, sender_core_0, receiver_core_0, WORD_SIZE * 1024));
    ASSERT_TRUE(unit_tests::erisc::direct_send::send_over_eth(
        mesh_device_1, mesh_device_0, sender_core_1, receiver_core_1, WORD_SIZE * 1024));
    ASSERT_TRUE(unit_tests::erisc::direct_send::send_over_eth(
        mesh_device_1, mesh_device_0, sender_core_0, receiver_core_0, WORD_SIZE * MAX_NUM_WORDS));
    ASSERT_TRUE(unit_tests::erisc::direct_send::send_over_eth(
        mesh_device_1, mesh_device_0, sender_core_1, receiver_core_1, WORD_SIZE * MAX_NUM_WORDS));
}

TEST_F(N300MeshDeviceFixture, ActiveEthBidirectionalCoreDirectSend) {
    using namespace CMAKE_UNIQUE_NAMESPACE;
    const auto& mesh_device_0 = devices_.at(0);
    const auto& mesh_device_1 = devices_.at(1);
    auto* const device_0 = mesh_device_0->get_devices()[0];
    auto* const device_1 = mesh_device_1->get_devices()[0];

    auto send_cores = device_1->get_ethernet_sockets(device_0->id());
    auto receiver_cores = device_0->get_ethernet_sockets(device_1->id());

    if (send_cores.size() != 2 || receiver_cores.size() != 2) {
        GTEST_SKIP() << "Not enough eth cores to run test. Need 2 on each device.";
    }

    CoreCoord sender_core_0 = send_cores[0];
    CoreCoord sender_core_1 = send_cores[1];

    CoreCoord receiver_core_0 = receiver_cores[0];
    CoreCoord receiver_core_1 = receiver_cores[1];

    uint32_t MAX_NUM_WORDS =
        MetalContext::instance().hal().get_dev_size(HalProgrammableCoreType::ACTIVE_ETH, HalL1MemAddrType::UNRESERVED) /
        WORD_SIZE;

    ASSERT_TRUE(unit_tests::erisc::direct_send::send_over_eth(
        mesh_device_0, mesh_device_1, sender_core_0, receiver_core_0, WORD_SIZE));
    ASSERT_TRUE(unit_tests::erisc::direct_send::send_over_eth(
        mesh_device_1, mesh_device_0, receiver_core_0, sender_core_0, WORD_SIZE));
    ASSERT_TRUE(unit_tests::erisc::direct_send::send_over_eth(
        mesh_device_0, mesh_device_1, sender_core_1, receiver_core_1, WORD_SIZE));
    ASSERT_TRUE(unit_tests::erisc::direct_send::send_over_eth(
        mesh_device_1, mesh_device_0, receiver_core_1, sender_core_1, WORD_SIZE));
    ASSERT_TRUE(unit_tests::erisc::direct_send::send_over_eth(
        mesh_device_0, mesh_device_1, sender_core_0, receiver_core_0, WORD_SIZE * 256));
    ASSERT_TRUE(unit_tests::erisc::direct_send::send_over_eth(
        mesh_device_1, mesh_device_0, receiver_core_0, sender_core_0, WORD_SIZE * 256));
    ASSERT_TRUE(unit_tests::erisc::direct_send::send_over_eth(
        mesh_device_0, mesh_device_1, sender_core_1, receiver_core_1, WORD_SIZE * 256));
    ASSERT_TRUE(unit_tests::erisc::direct_send::send_over_eth(
        mesh_device_1, mesh_device_0, receiver_core_1, sender_core_1, WORD_SIZE * 256));
    ASSERT_TRUE(unit_tests::erisc::direct_send::send_over_eth(
        mesh_device_0, mesh_device_1, sender_core_0, receiver_core_0, WORD_SIZE * 1024));
    ASSERT_TRUE(unit_tests::erisc::direct_send::send_over_eth(
        mesh_device_1, mesh_device_0, receiver_core_0, sender_core_0, WORD_SIZE * 1024));
    ASSERT_TRUE(unit_tests::erisc::direct_send::send_over_eth(
        mesh_device_0, mesh_device_1, sender_core_1, receiver_core_1, WORD_SIZE * 1024));
    ASSERT_TRUE(unit_tests::erisc::direct_send::send_over_eth(
        mesh_device_1, mesh_device_0, receiver_core_1, sender_core_1, WORD_SIZE * 1024));
    ASSERT_TRUE(unit_tests::erisc::direct_send::send_over_eth(
        mesh_device_0, mesh_device_1, sender_core_0, receiver_core_0, WORD_SIZE * MAX_NUM_WORDS));
    ASSERT_TRUE(unit_tests::erisc::direct_send::send_over_eth(
        mesh_device_1, mesh_device_0, receiver_core_0, sender_core_0, WORD_SIZE * MAX_NUM_WORDS));
    ASSERT_TRUE(unit_tests::erisc::direct_send::send_over_eth(
        mesh_device_0, mesh_device_1, sender_core_1, receiver_core_1, WORD_SIZE * MAX_NUM_WORDS));
    ASSERT_TRUE(unit_tests::erisc::direct_send::send_over_eth(
        mesh_device_1, mesh_device_0, receiver_core_1, sender_core_1, WORD_SIZE * MAX_NUM_WORDS));
}

TEST_F(N300MeshDeviceFixture, ActiveEthRandomDirectSendTests) {
    using namespace CMAKE_UNIQUE_NAMESPACE;
    GTEST_SKIP();
    srand(0);

    uint32_t MAX_NUM_WORDS =
        MetalContext::instance().hal().get_dev_size(HalProgrammableCoreType::ACTIVE_ETH, HalL1MemAddrType::UNRESERVED) /
        WORD_SIZE;

    std::map<std::pair<int, CoreCoord>, std::pair<int, CoreCoord>> connectivity = {
        {{0, CoreCoord(9, 6)}, {1, CoreCoord(9, 0)}},
        {{1, CoreCoord(9, 0)}, {0, CoreCoord(9, 6)}},
        {{0, CoreCoord(1, 6)}, {1, CoreCoord(1, 0)}},
        {{1, CoreCoord(1, 0)}, {0, CoreCoord(1, 6)}}};
    for (int i = 0; i < 1000; i++) {
        auto it = connectivity.begin();
        std::advance(it, rand() % (connectivity.size()));

        const auto& send_chip = devices_.at(std::get<0>(it->first));
        CoreCoord sender_core = std::get<1>(it->first);
        const auto& receiver_chip = devices_.at(std::get<0>(it->second));
        CoreCoord receiver_core = std::get<1>(it->second);
        int num_words = 0;
        if (MAX_NUM_WORDS != 0) {
            num_words = rand() % MAX_NUM_WORDS + 1;
        }

        ASSERT_TRUE(unit_tests::erisc::direct_send::send_over_eth(
            send_chip, receiver_chip, sender_core, receiver_core, WORD_SIZE * num_words));
    }
}

TEST_F(N300MeshDeviceFixture, ActiveEthKernelsDirectSendChip0ToChip1) {
    using namespace CMAKE_UNIQUE_NAMESPACE;
    const auto& mesh_device_0 = devices_.at(0);
    const auto& mesh_device_1 = devices_.at(1);
    auto* const device_0 = mesh_device_0->get_devices()[0];
    auto* const device_1 = mesh_device_1->get_devices()[0];

    const size_t src_eth_l1_byte_address =
        MetalContext::instance().hal().get_dev_addr(HalProgrammableCoreType::ACTIVE_ETH, HalL1MemAddrType::UNRESERVED);
    const size_t dst_eth_l1_byte_address =
        MetalContext::instance().hal().get_dev_addr(HalProgrammableCoreType::ACTIVE_ETH, HalL1MemAddrType::UNRESERVED);

    for (const auto& sender_core : device_0->get_active_ethernet_cores(true)) {
        if (not tt::tt_metal::MetalContext::instance().get_cluster().is_ethernet_link_up(device_0->id(), sender_core)) {
            continue;
        }
        auto [device_id, receiver_core] = device_0->get_connected_ethernet_core(sender_core);
        if (device_1->id() != device_id) {
            continue;
        }
        ASSERT_TRUE(unit_tests::erisc::direct_send::eth_direct_sender_receiver_kernels(
            static_cast<MeshDispatchFixture*>(this),
            mesh_device_0,
            mesh_device_1,
            WORD_SIZE,
            src_eth_l1_byte_address,
            dst_eth_l1_byte_address,
            sender_core,
            receiver_core));
        ASSERT_TRUE(unit_tests::erisc::direct_send::eth_direct_sender_receiver_kernels(
            static_cast<MeshDispatchFixture*>(this),
            mesh_device_0,
            mesh_device_1,
            4 * WORD_SIZE,
            src_eth_l1_byte_address,
            dst_eth_l1_byte_address,
            sender_core,
            receiver_core));
        ASSERT_TRUE(unit_tests::erisc::direct_send::eth_direct_sender_receiver_kernels(
            static_cast<MeshDispatchFixture*>(this),
            mesh_device_0,
            mesh_device_1,
            256 * WORD_SIZE,
            src_eth_l1_byte_address,
            dst_eth_l1_byte_address,
            sender_core,
            receiver_core));
        ASSERT_TRUE(unit_tests::erisc::direct_send::eth_direct_sender_receiver_kernels(
            static_cast<MeshDispatchFixture*>(this),
            mesh_device_0,
            mesh_device_1,
            1000 * WORD_SIZE,
            src_eth_l1_byte_address,
            dst_eth_l1_byte_address,
            sender_core,
            receiver_core));
    }
}

TEST_F(N300MeshDeviceFixture, ActiveEthKernelsDirectSendChip1ToChip0) {
    using namespace CMAKE_UNIQUE_NAMESPACE;
    const auto& mesh_device_0 = devices_.at(0);
    const auto& mesh_device_1 = devices_.at(1);
    auto* const device_0 = mesh_device_0->get_devices()[0];
    auto* const device_1 = mesh_device_1->get_devices()[0];

    const size_t src_eth_l1_byte_address =
        MetalContext::instance().hal().get_dev_addr(HalProgrammableCoreType::ACTIVE_ETH, HalL1MemAddrType::UNRESERVED);
    const size_t dst_eth_l1_byte_address =
        MetalContext::instance().hal().get_dev_addr(HalProgrammableCoreType::ACTIVE_ETH, HalL1MemAddrType::UNRESERVED);

    for (const auto& sender_core : device_1->get_active_ethernet_cores(true)) {
        if (not tt::tt_metal::MetalContext::instance().get_cluster().is_ethernet_link_up(device_1->id(), sender_core)) {
            continue;
        }
        auto [device_id, receiver_core] = device_1->get_connected_ethernet_core(sender_core);
        if (device_0->id() != device_id) {
            continue;
        }
        ASSERT_TRUE(unit_tests::erisc::direct_send::eth_direct_sender_receiver_kernels(
            static_cast<MeshDispatchFixture*>(this),
            mesh_device_1,
            mesh_device_0,
            WORD_SIZE,
            src_eth_l1_byte_address,
            dst_eth_l1_byte_address,
            sender_core,
            receiver_core));
        ASSERT_TRUE(unit_tests::erisc::direct_send::eth_direct_sender_receiver_kernels(
            static_cast<MeshDispatchFixture*>(this),
            mesh_device_1,
            mesh_device_0,
            4 * WORD_SIZE,
            src_eth_l1_byte_address,
            dst_eth_l1_byte_address,
            sender_core,
            receiver_core));
        ASSERT_TRUE(unit_tests::erisc::direct_send::eth_direct_sender_receiver_kernels(
            static_cast<MeshDispatchFixture*>(this),
            mesh_device_1,
            mesh_device_0,
            256 * WORD_SIZE,
            src_eth_l1_byte_address,
            dst_eth_l1_byte_address,
            sender_core,
            receiver_core));
        ASSERT_TRUE(unit_tests::erisc::direct_send::eth_direct_sender_receiver_kernels(
            static_cast<MeshDispatchFixture*>(this),
            mesh_device_1,
            mesh_device_0,
            1000 * WORD_SIZE,
            src_eth_l1_byte_address,
            dst_eth_l1_byte_address,
            sender_core,
            receiver_core));
    }
}

TEST_F(MeshDeviceFixture, ActiveEthKernelsDirectSendAllConnectedChips) {
    using namespace CMAKE_UNIQUE_NAMESPACE;
    const size_t src_eth_l1_byte_address =
        MetalContext::instance().hal().get_dev_addr(HalProgrammableCoreType::ACTIVE_ETH, HalL1MemAddrType::UNRESERVED);
    const size_t dst_eth_l1_byte_address =
        MetalContext::instance().hal().get_dev_addr(HalProgrammableCoreType::ACTIVE_ETH, HalL1MemAddrType::UNRESERVED);
    for (const auto& sender_mesh_device : devices_) {
        auto* const sender_device = sender_mesh_device->get_devices()[0];
        for (const auto& receiver_mesh_device : devices_) {
            auto* const receiver_device = receiver_mesh_device->get_devices()[0];
            if (sender_device->id() == receiver_device->id()) {
                continue;
            }
            for (const auto& sender_core : sender_device->get_active_ethernet_cores(true)) {
                if (not tt::tt_metal::MetalContext::instance().get_cluster().is_ethernet_link_up(
                        sender_device->id(), sender_core)) {
                    continue;
                }
                auto [device_id, receiver_core] = sender_device->get_connected_ethernet_core(sender_core);
                if (receiver_device->id() != device_id) {
                    continue;
                }
                ASSERT_TRUE(unit_tests::erisc::direct_send::eth_direct_sender_receiver_kernels(
                    static_cast<MeshDispatchFixture*>(this),
                    sender_mesh_device,
                    receiver_mesh_device,
                    WORD_SIZE,
                    src_eth_l1_byte_address,
                    dst_eth_l1_byte_address,
                    sender_core,
                    receiver_core));
                ASSERT_TRUE(unit_tests::erisc::direct_send::eth_direct_sender_receiver_kernels(
                    static_cast<MeshDispatchFixture*>(this),
                    sender_mesh_device,
                    receiver_mesh_device,
                    4 * WORD_SIZE,
                    src_eth_l1_byte_address,
                    dst_eth_l1_byte_address,
                    sender_core,
                    receiver_core));
                ASSERT_TRUE(unit_tests::erisc::direct_send::eth_direct_sender_receiver_kernels(
                    static_cast<MeshDispatchFixture*>(this),
                    sender_mesh_device,
                    receiver_mesh_device,
                    256 * WORD_SIZE,
                    src_eth_l1_byte_address,
                    dst_eth_l1_byte_address,
                    sender_core,
                    receiver_core));
                ASSERT_TRUE(unit_tests::erisc::direct_send::eth_direct_sender_receiver_kernels(
                    static_cast<MeshDispatchFixture*>(this),
                    sender_mesh_device,
                    receiver_mesh_device,
                    1000 * WORD_SIZE,
                    src_eth_l1_byte_address,
                    dst_eth_l1_byte_address,
                    sender_core,
                    receiver_core));
            }
        }
    }
}

TEST_F(TwoMeshDeviceFixture, ActiveEthKernelsBidirectionalDirectSend) {
    using namespace CMAKE_UNIQUE_NAMESPACE;
    const auto& mesh_device_0 = devices_.at(0);
    const auto& mesh_device_1 = devices_.at(1);
    auto* const device_0 = mesh_device_0->get_devices()[0];

    const size_t src_eth_l1_byte_address =
        MetalContext::instance().hal().get_dev_addr(HalProgrammableCoreType::ACTIVE_ETH, HalL1MemAddrType::UNRESERVED);
    const size_t dst_eth_l1_byte_address =
        MetalContext::instance().hal().get_dev_addr(HalProgrammableCoreType::ACTIVE_ETH, HalL1MemAddrType::UNRESERVED);

    uint32_t MAX_NUM_WORDS =
        MetalContext::instance().hal().get_dev_size(HalProgrammableCoreType::ACTIVE_ETH, HalL1MemAddrType::UNRESERVED) /
        WORD_SIZE;

    for (const auto& sender_core : device_0->get_active_ethernet_cores(true)) {
        if (not tt::tt_metal::MetalContext::instance().get_cluster().is_ethernet_link_up(device_0->id(), sender_core)) {
            continue;
        }
        CoreCoord receiver_core = std::get<1>(device_0->get_connected_ethernet_core(sender_core));
        ASSERT_TRUE(unit_tests::erisc::direct_send::eth_direct_sender_receiver_kernels(
            static_cast<MeshDispatchFixture*>(this),
            mesh_device_0,
            mesh_device_1,
            WORD_SIZE,
            src_eth_l1_byte_address,
            dst_eth_l1_byte_address,
            sender_core,
            receiver_core));
        ASSERT_TRUE(unit_tests::erisc::direct_send::eth_direct_sender_receiver_kernels(
            static_cast<MeshDispatchFixture*>(this),
            mesh_device_1,
            mesh_device_0,
            WORD_SIZE,
            src_eth_l1_byte_address,
            dst_eth_l1_byte_address,
            receiver_core,
            sender_core));
    }
    for (const auto& sender_core : device_0->get_active_ethernet_cores(true)) {
        if (not tt::tt_metal::MetalContext::instance().get_cluster().is_ethernet_link_up(device_0->id(), sender_core)) {
            continue;
        }
        CoreCoord receiver_core = std::get<1>(device_0->get_connected_ethernet_core(sender_core));
        ASSERT_TRUE(unit_tests::erisc::direct_send::eth_direct_sender_receiver_kernels(
            static_cast<MeshDispatchFixture*>(this),
            mesh_device_0,
            mesh_device_1,
            WORD_SIZE * 256,
            src_eth_l1_byte_address,
            dst_eth_l1_byte_address,
            sender_core,
            receiver_core));
        ASSERT_TRUE(unit_tests::erisc::direct_send::eth_direct_sender_receiver_kernels(
            static_cast<MeshDispatchFixture*>(this),
            mesh_device_1,
            mesh_device_0,
            WORD_SIZE * 256,
            src_eth_l1_byte_address,
            dst_eth_l1_byte_address,
            receiver_core,
            sender_core));
    }
    for (const auto& sender_core : device_0->get_active_ethernet_cores(true)) {
        if (not tt::tt_metal::MetalContext::instance().get_cluster().is_ethernet_link_up(device_0->id(), sender_core)) {
            continue;
        }
        CoreCoord receiver_core = std::get<1>(device_0->get_connected_ethernet_core(sender_core));
        ASSERT_TRUE(unit_tests::erisc::direct_send::eth_direct_sender_receiver_kernels(
            static_cast<MeshDispatchFixture*>(this),
            mesh_device_0,
            mesh_device_1,
            WORD_SIZE * 1024,
            src_eth_l1_byte_address,
            dst_eth_l1_byte_address,
            sender_core,
            receiver_core));
        ASSERT_TRUE(unit_tests::erisc::direct_send::eth_direct_sender_receiver_kernels(
            static_cast<MeshDispatchFixture*>(this),
            mesh_device_1,
            mesh_device_0,
            WORD_SIZE * 1024,
            src_eth_l1_byte_address,
            dst_eth_l1_byte_address,
            receiver_core,
            sender_core));
    }
    for (const auto& sender_core : device_0->get_active_ethernet_cores(true)) {
        if (not tt::tt_metal::MetalContext::instance().get_cluster().is_ethernet_link_up(device_0->id(), sender_core)) {
            continue;
        }
        CoreCoord receiver_core = std::get<1>(device_0->get_connected_ethernet_core(sender_core));
        ASSERT_TRUE(unit_tests::erisc::direct_send::eth_direct_sender_receiver_kernels(
            static_cast<MeshDispatchFixture*>(this),
            mesh_device_0,
            mesh_device_1,
            WORD_SIZE * MAX_NUM_WORDS,
            src_eth_l1_byte_address,
            dst_eth_l1_byte_address,
            sender_core,
            receiver_core));
        ASSERT_TRUE(unit_tests::erisc::direct_send::eth_direct_sender_receiver_kernels(
            static_cast<MeshDispatchFixture*>(this),
            mesh_device_1,
            mesh_device_0,
            WORD_SIZE * MAX_NUM_WORDS,
            src_eth_l1_byte_address,
            dst_eth_l1_byte_address,
            receiver_core,
            sender_core));
    }
}

TEST_F(TwoMeshDeviceFixture, ActiveEthKernelsRepeatedDirectSends) {
    using namespace CMAKE_UNIQUE_NAMESPACE;
    const auto& mesh_device_0 = devices_.at(0);
    const auto& mesh_device_1 = devices_.at(1);
    auto* const device_0 = mesh_device_0->get_devices()[0];

    const size_t src_eth_l1_byte_address =
        MetalContext::instance().hal().get_dev_addr(HalProgrammableCoreType::ACTIVE_ETH, HalL1MemAddrType::UNRESERVED);
    const size_t dst_eth_l1_byte_address =
        MetalContext::instance().hal().get_dev_addr(HalProgrammableCoreType::ACTIVE_ETH, HalL1MemAddrType::UNRESERVED);

    for (const auto& sender_core : device_0->get_active_ethernet_cores(true)) {
        if (not tt::tt_metal::MetalContext::instance().get_cluster().is_ethernet_link_up(device_0->id(), sender_core)) {
            continue;
        }
        CoreCoord receiver_core = std::get<1>(device_0->get_connected_ethernet_core(sender_core));
        for (int i = 0; i < 10; i++) {
            ASSERT_TRUE(unit_tests::erisc::direct_send::eth_direct_sender_receiver_kernels(
                static_cast<MeshDispatchFixture*>(this),
                mesh_device_0,
                mesh_device_1,
                WORD_SIZE,
                src_eth_l1_byte_address + WORD_SIZE * i,
                dst_eth_l1_byte_address + WORD_SIZE * i,
                sender_core,
                receiver_core));
        }
        for (int i = 0; i < 10; i++) {
            ASSERT_TRUE(unit_tests::erisc::direct_send::eth_direct_sender_receiver_kernels(
                static_cast<MeshDispatchFixture*>(this),
                mesh_device_1,
                mesh_device_0,
                WORD_SIZE,
                src_eth_l1_byte_address + WORD_SIZE * i,
                dst_eth_l1_byte_address + WORD_SIZE * i,
                receiver_core,
                sender_core));
        }
    }
}

TEST_F(TwoMeshDeviceFixture, ActiveEthKernelsRandomDirectSendTests) {
    using namespace CMAKE_UNIQUE_NAMESPACE;
    srand(0);
    const auto& mesh_device_0 = devices_.at(0);
    const auto& mesh_device_1 = devices_.at(1);
    auto* const device_0 = mesh_device_0->get_devices()[0];
    auto* const device_1 = mesh_device_1->get_devices()[0];

    std::map<std::tuple<int, CoreCoord>, std::tuple<int, CoreCoord>> connectivity = {};
    for (const auto& sender_core : device_0->get_active_ethernet_cores(true)) {
        if (not tt::tt_metal::MetalContext::instance().get_cluster().is_ethernet_link_up(device_0->id(), sender_core)) {
            continue;
        }
        const auto& receiver_core = device_0->get_connected_ethernet_core(sender_core);
        connectivity.insert({{0, sender_core}, receiver_core});
    }
    for (const auto& sender_core : device_1->get_active_ethernet_cores(true)) {
        if (not tt::tt_metal::MetalContext::instance().get_cluster().is_ethernet_link_up(device_1->id(), sender_core)) {
            continue;
        }
        const auto& receiver_core = device_1->get_connected_ethernet_core(sender_core);
        connectivity.insert({{1, sender_core}, receiver_core});
    }
    for (int i = 0; i < 1000; i++) {
        auto it = connectivity.begin();
        std::advance(it, rand() % (connectivity.size()));

        const auto& send_chip = devices_.at(std::get<0>(it->first));
        CoreCoord sender_core = std::get<1>(it->first);

        // gotcha: devices_ are mesh devices. Mesh device IDs are not the same as actual device IDs needed in the
        // cluster
        auto* send_device = send_chip->get_devices()[0];
        if (not tt::tt_metal::MetalContext::instance().get_cluster().is_ethernet_link_up(
                send_device->id(), sender_core)) {
            continue;
        }

        const auto& receiver_chip = devices_.at(std::get<0>(it->second));
        CoreCoord receiver_core = std::get<1>(it->second);

        uint32_t erisc_unreserved_base_addr = MetalContext::instance().hal().get_dev_addr(
            HalProgrammableCoreType::ACTIVE_ETH, HalL1MemAddrType::UNRESERVED);
        uint32_t max_l1_loading_addr =
            erisc_unreserved_base_addr + MetalContext::instance().hal().get_dev_size(
                                             HalProgrammableCoreType::ACTIVE_ETH, HalL1MemAddrType::UNRESERVED);

        const size_t src_eth_l1_byte_address = unit_tests::erisc::direct_send::get_rand_32_byte_aligned_address(
            erisc_unreserved_base_addr, max_l1_loading_addr);
        const size_t dst_eth_l1_byte_address = unit_tests::erisc::direct_send::get_rand_32_byte_aligned_address(
            erisc_unreserved_base_addr, max_l1_loading_addr);

        int max_words = (max_l1_loading_addr - std::max(src_eth_l1_byte_address, dst_eth_l1_byte_address)) / WORD_SIZE;
        int num_words = (rand() % max_words) + 1;

        ASSERT_TRUE(unit_tests::erisc::direct_send::eth_direct_sender_receiver_kernels(
            static_cast<MeshDispatchFixture*>(this),
            send_chip,
            receiver_chip,
            WORD_SIZE * num_words,
            src_eth_l1_byte_address,
            dst_eth_l1_byte_address,
            sender_core,
            receiver_core));
    }
}
TEST_F(TwoMeshDeviceFixture, ActiveEthKernelsRandomEthPacketSizeDirectSendTests) {
    srand(0);
    const auto& mesh_device_0 = devices_.at(0);
    const auto& mesh_device_1 = devices_.at(1);
    auto* const device_0 = mesh_device_0->get_devices()[0];
    auto* const device_1 = mesh_device_1->get_devices()[0];

    std::map<std::tuple<int, CoreCoord>, std::tuple<int, CoreCoord>> connectivity = {};
    for (const auto& sender_core : device_0->get_active_ethernet_cores(true)) {
        if (not tt::tt_metal::MetalContext::instance().get_cluster().is_ethernet_link_up(device_0->id(), sender_core)) {
            continue;
        }
        const auto& receiver_core = device_0->get_connected_ethernet_core(sender_core);
        connectivity.insert({{0, sender_core}, receiver_core});
    }
    for (const auto& sender_core : device_1->get_active_ethernet_cores(true)) {
        if (not tt::tt_metal::MetalContext::instance().get_cluster().is_ethernet_link_up(device_1->id(), sender_core)) {
            continue;
        }
        const auto& receiver_core = device_1->get_connected_ethernet_core(sender_core);
        connectivity.insert({{1, sender_core}, receiver_core});
    }
    std::vector<uint32_t> num_bytes_per_send_test_vals = {
        16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536};
    uint32_t erisc_unreserved_base_addr =
        MetalContext::instance().hal().get_dev_addr(HalProgrammableCoreType::ACTIVE_ETH, HalL1MemAddrType::UNRESERVED);
    uint32_t max_l1_loading_addr =
        erisc_unreserved_base_addr +
        MetalContext::instance().hal().get_dev_size(HalProgrammableCoreType::ACTIVE_ETH, HalL1MemAddrType::UNRESERVED);
    for (const auto& num_bytes_per_send : num_bytes_per_send_test_vals) {
        for (int i = 0; i < 10; i++) {
            auto it = connectivity.begin();
            std::advance(it, rand() % (connectivity.size()));

            const auto& send_chip = devices_.at(std::get<0>(it->first));
            CoreCoord sender_core = std::get<1>(it->first);
            const auto& receiver_chip = devices_.at(std::get<0>(it->second));
            CoreCoord receiver_core = std::get<1>(it->second);

            const size_t src_eth_l1_byte_address = unit_tests::erisc::direct_send::get_rand_32_byte_aligned_address(
                erisc_unreserved_base_addr, max_l1_loading_addr - 65536);
            const size_t dst_eth_l1_byte_address = unit_tests::erisc::direct_send::get_rand_32_byte_aligned_address(
                erisc_unreserved_base_addr, max_l1_loading_addr - 65536);

            int max_words =
                (max_l1_loading_addr - std::max(src_eth_l1_byte_address, dst_eth_l1_byte_address)) / num_bytes_per_send;
            int num_words = (rand() % max_words) + 1;

            const auto num_eriscs =
                MetalContext::instance().hal().get_num_risc_processors(HalProgrammableCoreType::ACTIVE_ETH);
            for (uint32_t erisc_idx = 0; erisc_idx < num_eriscs; erisc_idx++) {
                ASSERT_TRUE(unit_tests::erisc::direct_send::eth_direct_sender_receiver_kernels(
                    static_cast<MeshDispatchFixture*>(this),
                    send_chip,
                    receiver_chip,
                    num_bytes_per_send * num_words,
                    src_eth_l1_byte_address,
                    dst_eth_l1_byte_address,
                    sender_core,
                    receiver_core,
                    static_cast<DataMovementProcessor>(erisc_idx),
                    num_bytes_per_send));
            }
        }
    }
}

TEST_F(UnitMeshCQMultiDeviceProgramFixture, ActiveEthKernelsDirectSendAllConnectedChips) {
    using namespace CMAKE_UNIQUE_NAMESPACE;
    const size_t src_eth_l1_byte_address =
        MetalContext::instance().hal().get_dev_addr(HalProgrammableCoreType::ACTIVE_ETH, HalL1MemAddrType::UNRESERVED);
    const size_t dst_eth_l1_byte_address =
        MetalContext::instance().hal().get_dev_addr(HalProgrammableCoreType::ACTIVE_ETH, HalL1MemAddrType::UNRESERVED);
    const auto num_eriscs = MetalContext::instance().hal().get_num_risc_processors(HalProgrammableCoreType::ACTIVE_ETH);
    for (const auto& sender_mesh_device : devices_) {
        auto* const sender_device = sender_mesh_device->get_devices()[0];
        for (const auto& receiver_mesh_device : devices_) {
            auto* const receiver_device = receiver_mesh_device->get_devices()[0];
            if (sender_device->id() >= receiver_device->id()) {
                continue;
            }
            for (const auto& sender_core : sender_device->get_active_ethernet_cores(true)) {
                if (not tt::tt_metal::MetalContext::instance().get_cluster().is_ethernet_link_up(
                        sender_device->id(), sender_core)) {
                    continue;
                }
                auto [device_id, receiver_core] = sender_device->get_connected_ethernet_core(sender_core);
                if (receiver_device->id() != device_id) {
                    continue;
                }
                for (uint32_t erisc_idx = 0; erisc_idx < num_eriscs; erisc_idx++) {
                    const auto processor = static_cast<DataMovementProcessor>(erisc_idx);
                    ASSERT_TRUE(unit_tests::erisc::direct_send::eth_direct_sender_receiver_kernels(
                        static_cast<MeshDispatchFixture*>(this),
                        sender_mesh_device,
                        receiver_mesh_device,
                        WORD_SIZE,
                        src_eth_l1_byte_address,
                        dst_eth_l1_byte_address,
                        sender_core,
                        receiver_core,
                        processor));
                    ASSERT_TRUE(unit_tests::erisc::direct_send::eth_direct_sender_receiver_kernels(
                        static_cast<MeshDispatchFixture*>(this),
                        sender_mesh_device,
                        receiver_mesh_device,
                        4 * WORD_SIZE,
                        src_eth_l1_byte_address,
                        dst_eth_l1_byte_address,
                        sender_core,
                        receiver_core,
                        processor));
                    ASSERT_TRUE(unit_tests::erisc::direct_send::eth_direct_sender_receiver_kernels(
                        static_cast<MeshDispatchFixture*>(this),
                        sender_mesh_device,
                        receiver_mesh_device,
                        256 * WORD_SIZE,
                        src_eth_l1_byte_address,
                        dst_eth_l1_byte_address,
                        sender_core,
                        receiver_core,
                        processor));
                    ASSERT_TRUE(unit_tests::erisc::direct_send::eth_direct_sender_receiver_kernels(
                        static_cast<MeshDispatchFixture*>(this),
                        sender_mesh_device,
                        receiver_mesh_device,
                        1000 * WORD_SIZE,
                        src_eth_l1_byte_address,
                        dst_eth_l1_byte_address,
                        sender_core,
                        receiver_core,
                        processor));
                }
            }
        }
    }
}

}  // namespace tt::tt_metal
