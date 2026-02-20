// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <fmt/base.h>
#include <gtest/gtest.h>
#include <cstdlib>
#include <umd/device/types/arch.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-logger/tt-logger.hpp>
#include <cstdint>

#include <tt_stl/assert.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/data_types.hpp>
#include <tt-metalium/device.hpp>
#include "eth_test_common.hpp"
#include "mesh_dispatch_fixture.hpp"
#include "multi_device_fixture.hpp"
#include <tt-metalium/program.hpp>
#include "impl/context/metal_context.hpp"

using namespace tt;
using namespace tt::tt_metal;
using namespace tt::test_utils;
namespace unit_tests::erisc::direct_send {

static void eth_direct_send_multi_txq_rxq(
    tt_metal::MeshDispatchFixture* fixture,
    std::shared_ptr<tt_metal::distributed::MeshDevice> sender_mesh_device,
    std::shared_ptr<tt_metal::distributed::MeshDevice> receiver_mesh_device,
    const CoreCoord& eth_sender_core,
    const CoreCoord& eth_receiver_core,
    uint32_t data_txq_id,
    uint32_t ack_txq_id,
    uint32_t num_messages) {
    ////////////////////////////////////////////////////////////////////////////
    //                      Sender Device
    ////////////////////////////////////////////////////////////////////////////
    distributed::MeshWorkload sender_workload;
    auto zero_coord = distributed::MeshCoordinate(0, 0);
    auto device_range = distributed::MeshCoordinateRange(zero_coord, zero_coord);
    tt_metal::Program sender_program = tt_metal::Program();

    constexpr size_t PAYLOAD_SIZE = 32;
    const size_t unreserved_l1_start = tt::tt_metal::MetalContext::instance().hal().get_dev_size(
        tt::tt_metal::HalProgrammableCoreType::ACTIVE_ETH, tt::tt_metal::HalL1MemAddrType::UNRESERVED);
    auto ethernet_config = tt_metal::EthernetConfig{.compile_args = {data_txq_id, ack_txq_id, PAYLOAD_SIZE}};
    eth_test_common::set_arch_specific_eth_config(ethernet_config);
    auto eth_sender_kernel = tt_metal::CreateKernel(
        sender_program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/unit_tests/erisc/eth_multi_txq_rxq_bidirectional.cpp",
        eth_sender_core,
        ethernet_config);

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
    distributed::MeshWorkload receiver_workload;
    tt_metal::Program receiver_program = tt_metal::Program();

    auto eth_receiver_kernel = tt_metal::CreateKernel(
        receiver_program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/unit_tests/erisc/eth_multi_txq_rxq_bidirectional.cpp",
        eth_receiver_core,
        ethernet_config);

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
}

}  // namespace unit_tests::erisc::direct_send

namespace tt::tt_metal {

static void run_multi_txq_rxq_test(
    MeshDispatchFixture* fixture,
    const std::shared_ptr<tt_metal::distributed::MeshDevice>& mesh_device_0,
    const std::shared_ptr<tt_metal::distributed::MeshDevice>& mesh_device_1,
    uint32_t data_txq_id,
    uint32_t ack_txq_id,
    uint32_t num_messages) {
    auto* device_0 = mesh_device_0->get_devices()[0];
    auto* device_1 = mesh_device_1->get_devices()[0];
    // Find ethernet cores that connect device_0 and device_1 using standard metal APIs
    std::optional<CoreCoord> sender_core_0;
    std::optional<CoreCoord> receiver_core_0;

    // Get active ethernet cores from device_0
    const auto& active_eth_cores = device_0->get_active_ethernet_cores(false);

    // Find an ethernet core on device_0 that connects to device_1
    for (const auto& eth_core : active_eth_cores) {
        ChipId connected_device_id;
        CoreCoord connected_eth_core;
        std::tie(connected_device_id, connected_eth_core) = device_0->get_connected_ethernet_core(eth_core);

        if (connected_device_id == device_1->id()) {
            sender_core_0 = eth_core;
            receiver_core_0 = connected_eth_core;
            break;
        }
    }

    // Verify we found a connection
    TT_FATAL(
        sender_core_0.has_value() && receiver_core_0.has_value(),
        "No ethernet connection found between device_0 and device_1");

    unit_tests::erisc::direct_send::eth_direct_send_multi_txq_rxq(
        fixture,
        mesh_device_0,
        mesh_device_1,
        sender_core_0.value(),
        receiver_core_0.value(),
        data_txq_id,
        ack_txq_id,
        num_messages);

}  // namespace tt::tt_metal

TEST_F(TwoDeviceBlackholeFixture, ActiveEthChipToChipMultiTxqRxq_Both0) {
    run_multi_txq_rxq_test(this, this->devices_.at(0), this->devices_.at(1), 0, 0, 100000);
}
TEST_F(TwoDeviceBlackholeFixture, ActiveEthChipToChipMultiTxqRxq_Qs_0_and_1) {
    run_multi_txq_rxq_test(this, this->devices_.at(0), this->devices_.at(1), 0, 1, 100000);
}

}  // namespace tt::tt_metal
