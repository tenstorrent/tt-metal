// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <fmt/base.h>
#include <gtest/gtest.h>
#include <stdlib.h>
#include <umd/device/types/arch.h>
#include <tt-metalium/host_api.hpp>
#include <tt-logger/tt-logger.hpp>
#include <cstdint>

#include <tt-metalium/assert.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/data_types.hpp>
#include <tt-metalium/device.hpp>
#include "dispatch_fixture.hpp"
#include "hal.hpp"
#include "multi_device_fixture.hpp"
#include <tt-metalium/program.hpp>
#include "impl/context/metal_context.hpp"

using namespace tt;
using namespace tt::test_utils;
namespace unit_tests::erisc::direct_send {

static void eth_direct_send_multi_txq_rxq(
    tt_metal::DispatchFixture* fixture,
    tt_metal::IDevice* sender_device,
    tt_metal::IDevice* receiver_device,
    const CoreCoord& eth_sender_core,
    const CoreCoord& eth_receiver_core,
    uint32_t data_txq_id,
    uint32_t ack_txq_id,
    uint32_t num_messages) {
    ////////////////////////////////////////////////////////////////////////////
    //                      Sender Device
    ////////////////////////////////////////////////////////////////////////////
    tt_metal::Program sender_program = tt_metal::Program();

    constexpr size_t PAYLOAD_SIZE = 32;
    const size_t unreserved_l1_start = tt::tt_metal::MetalContext::instance().hal().get_dev_size(
        tt::tt_metal::HalProgrammableCoreType::ACTIVE_ETH, tt::tt_metal::HalL1MemAddrType::UNRESERVED);
    auto eth_sender_kernel = tt_metal::CreateKernel(
        sender_program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/unit_tests/erisc/eth_multi_txq_rxq_bidirectional.cpp",
        eth_sender_core,
        tt_metal::EthernetConfig{.compile_args = {data_txq_id, ack_txq_id, PAYLOAD_SIZE, false, 0, 1}});

    size_t local_eth_l1_src_addr = unreserved_l1_start + 16;
    size_t receiver_credit_ack_src = local_eth_l1_src_addr + PAYLOAD_SIZE;
    size_t receiver_credit_ack_dest = receiver_credit_ack_src + 32;
    size_t remote_eth_l1_dst_addr = receiver_credit_ack_dest + 32;

    tt_metal::SetRuntimeArgs(
        sender_program,
        eth_sender_kernel,
        eth_sender_core,
        {unreserved_l1_start,
         true,  // HS sender
         local_eth_l1_src_addr,
         receiver_credit_ack_src,
         receiver_credit_ack_dest,
         remote_eth_l1_dst_addr,
         num_messages});

    ////////////////////////////////////////////////////////////////////////////
    //                      Receiver Device
    ////////////////////////////////////////////////////////////////////////////
    tt_metal::Program receiver_program = tt_metal::Program();

    auto eth_receiver_kernel = tt_metal::CreateKernel(
        receiver_program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/unit_tests/erisc/eth_multi_txq_rxq_bidirectional.cpp",
        eth_receiver_core,
        tt_metal::EthernetConfig{
            .compile_args = {data_txq_id, ack_txq_id, PAYLOAD_SIZE, false, 0, 1}});  // probably want to use NOC_1 here

    tt_metal::SetRuntimeArgs(
        receiver_program,
        eth_receiver_kernel,
        eth_receiver_core,
        {unreserved_l1_start,
         false,  // HS sender
         local_eth_l1_src_addr,
         receiver_credit_ack_src,
         receiver_credit_ack_dest,
         remote_eth_l1_dst_addr,
         num_messages});

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
}

static void eth_direct_send_multi_txq_rxq_dual_erisc(
    tt_metal::DispatchFixture* fixture,
    tt_metal::IDevice* sender_device,
    tt_metal::IDevice* receiver_device,
    const CoreCoord& eth_sender_core,
    const CoreCoord& eth_receiver_core,
    uint32_t num_messages) {
    ////////////////////////////////////////////////////////////////////////////
    //                      Sender Device
    ////////////////////////////////////////////////////////////////////////////
    tt_metal::Program sender_program = tt_metal::Program();

    constexpr size_t PAYLOAD_SIZE = 32;
    const size_t unreserved_l1_start = tt::tt_metal::MetalContext::instance().hal().get_dev_size(
        tt::tt_metal::HalProgrammableCoreType::ACTIVE_ETH, tt::tt_metal::HalL1MemAddrType::UNRESERVED);
    auto eth_sender_0_kernel = tt_metal::CreateKernel(
        sender_program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/unit_tests/erisc/eth_multi_txq_rxq_bidirectional.cpp",
        eth_sender_core,
        tt_metal::EthernetConfig{
            .noc = tt::tt_metal::NOC::NOC_0,
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
            .compile_args = {0, 0, PAYLOAD_SIZE, true, 0, 1}});

    auto eth_sender_1_kernel = tt_metal::CreateKernel(
        sender_program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/unit_tests/erisc/eth_multi_txq_rxq_bidirectional.cpp",
        eth_sender_core,
        tt_metal::EthernetConfig{
            .noc = tt::tt_metal::NOC::NOC_1,
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_1,
            .compile_args = {1, 1, PAYLOAD_SIZE, true, 2, 3}});

    // Memory layout
    size_t send_receive_0_handshake_addr = unreserved_l1_start;
    size_t send_receive_1_handshake_addr = send_receive_0_handshake_addr + 32;
    size_t local_eth_l1_src_addr_0 = send_receive_1_handshake_addr + 32;
    size_t local_eth_l1_src_addr_1 = local_eth_l1_src_addr_0 + PAYLOAD_SIZE;
    size_t receiver_0_credit_ack_src = local_eth_l1_src_addr_1 + PAYLOAD_SIZE;
    size_t receiver_0_credit_ack_dest = receiver_0_credit_ack_src + 32;
    size_t receiver_1_credit_ack_src = receiver_0_credit_ack_dest + 32;
    size_t receiver_1_credit_ack_dest = receiver_1_credit_ack_src + 32;
    size_t sender_0_payload_dest = receiver_1_credit_ack_dest + 32;
    size_t sender_1_payload_dest = sender_0_payload_dest + PAYLOAD_SIZE;

    tt_metal::SetRuntimeArgs(
        sender_program,
        eth_sender_0_kernel,
        eth_sender_core,
        {send_receive_0_handshake_addr,
         true,  // HS sender
         local_eth_l1_src_addr_0,
         receiver_0_credit_ack_src,
         receiver_0_credit_ack_dest,
         sender_0_payload_dest,
         num_messages});

    tt_metal::SetRuntimeArgs(
        sender_program,
        eth_sender_1_kernel,
        eth_sender_core,
        {send_receive_1_handshake_addr,
         true,  // HS sender
         local_eth_l1_src_addr_1,
         receiver_1_credit_ack_src,
         receiver_1_credit_ack_dest,
         sender_1_payload_dest,
         num_messages});

    ////////////////////////////////////////////////////////////////////////////
    //                      Receiver Device
    ////////////////////////////////////////////////////////////////////////////
    tt_metal::Program receiver_program = tt_metal::Program();

    auto eth_receiver_0_kernel = tt_metal::CreateKernel(
        receiver_program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/unit_tests/erisc/eth_multi_txq_rxq_bidirectional.cpp",
        eth_receiver_core,
        tt_metal::EthernetConfig{
            .noc = tt::tt_metal::NOC::NOC_0,
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
            .compile_args = {0, 0, PAYLOAD_SIZE, true, 0, 1}});

    auto eth_receiver_1_kernel = tt_metal::CreateKernel(
        receiver_program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/unit_tests/erisc/eth_multi_txq_rxq_bidirectional.cpp",
        eth_receiver_core,
        tt_metal::EthernetConfig{
            .noc = tt::tt_metal::NOC::NOC_1,
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_1,
            .compile_args = {1, 1, PAYLOAD_SIZE, true, 2, 3}});

    tt_metal::SetRuntimeArgs(
        receiver_program,
        eth_receiver_0_kernel,
        eth_receiver_core,
        {send_receive_0_handshake_addr,
         false,  // HS sender
         local_eth_l1_src_addr_0,
         receiver_0_credit_ack_src,
         receiver_0_credit_ack_dest,
         sender_0_payload_dest,
         num_messages});

    tt_metal::SetRuntimeArgs(
        receiver_program,
        eth_receiver_1_kernel,
        eth_receiver_core,
        {send_receive_1_handshake_addr,
         false,  // HS sender
         local_eth_l1_src_addr_1,
         receiver_1_credit_ack_src,
         receiver_1_credit_ack_dest,
         sender_1_payload_dest,
         num_messages});

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
}

}  // namespace unit_tests::erisc::direct_send

namespace tt::tt_metal {

static void run_multi_txq_rxq_test(
    DispatchFixture* fixture,
    IDevice* device_0,
    IDevice* device_1,
    uint32_t data_txq_id,
    uint32_t ack_txq_id,
    uint32_t num_messages,
    bool dual_erisc) {
    // Find ethernet cores that connect device_0 and device_1 using standard metal APIs
    std::optional<CoreCoord> sender_core_0;
    std::optional<CoreCoord> receiver_core_0;

    // Get active ethernet cores from device_0
    const auto& active_eth_cores = device_0->get_active_ethernet_cores(false);

    // Find an ethernet core on device_0 that connects to device_1
    for (const auto& eth_core : active_eth_cores) {
        chip_id_t connected_device_id;
        CoreCoord connected_eth_core;
        std::tie(connected_device_id, connected_eth_core) = device_0->get_connected_ethernet_core(eth_core);

        if (connected_device_id == device_1->id()) {
            sender_core_0 = eth_core;
            receiver_core_0 = connected_eth_core;
            break;
        }
    }

    // Verify we found a connection
    ASSERT_TRUE(sender_core_0.has_value() && receiver_core_0.has_value());

    if (dual_erisc) {
        unit_tests::erisc::direct_send::eth_direct_send_multi_txq_rxq_dual_erisc(
            fixture, device_0, device_1, sender_core_0.value(), receiver_core_0.value(), num_messages);
    } else {
        unit_tests::erisc::direct_send::eth_direct_send_multi_txq_rxq(
            fixture,
            device_0,
            device_1,
            sender_core_0.value(),
            receiver_core_0.value(),
            data_txq_id,
            ack_txq_id,
            num_messages);
    }
}

TEST_F(TwoDeviceBlackholeFixture, ActiveEthChipToChipMultiTxqRxq_Both0) {
    run_multi_txq_rxq_test(this, this->devices_.at(0), this->devices_.at(1), 0, 0, 100000, false);
}
TEST_F(TwoDeviceBlackholeFixture, ActiveEthChipToChipMultiTxqRxq_Qs_0_and_1) {
    run_multi_txq_rxq_test(this, this->devices_.at(0), this->devices_.at(1), 0, 1, 100000, false);
}

TEST_F(TwoDeviceBlackholeFixture, ActiveEthChipToChipMultiTxqRxq_Qs_0_and_1_DualErisc) {
    if (tt::tt_metal::MetalContext::instance().hal().get_num_risc_processors(
            tt::tt_metal::HalProgrammableCoreType::ACTIVE_ETH) < 2) {
        GTEST_SKIP() << "Skipping test because device does not have dual erisc cores";
    }
    run_multi_txq_rxq_test(this, this->devices_.at(0), this->devices_.at(1), 0, 1, 100000, true);
}

}  // namespace tt::tt_metal
