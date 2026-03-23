// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/tt_metal/deployment/eth/common.hpp"
#include "tt_metal/tt_metal/deployment/deployment_common.hpp"
#include "tt_metal/tt_metal/deployment/kernels/sync_types.hpp"

#include <gtest/gtest.h>
#include <tt-logger/tt-logger.hpp>
#include <tt_stl/assert.hpp>

#include "tt_metal/test_utils/stimulus.hpp"
#include "command_queue_fixture.hpp"
#include "tt_metal/tt_metal/eth/eth_test_common.hpp"

#define BANDWIDTH_THRESHOLD_BIDIR 300.0

namespace tt::tt_metal {

using namespace std;
using namespace tt;
using namespace tt::test_utils;

static void prepare_receiver_bidir(
    tt::tt_metal::IDevice* const recv_device,
    const CoreCoord& recv_core,
    uint32_t transfer_size,
    uint32_t transfer_count,
    std::vector<uint32_t>& all_zeros,
    DataMovementProcessor processor,
    uint32_t recv_l1_address,
    uint32_t barrier_address,
    tt_metal::Program* recv_program) {
    tt::tt_metal::MetalContext::instance().get_cluster().write_core(
        recv_device->id(), recv_device->ethernet_core_from_logical_core(recv_core), all_zeros, recv_l1_address);

    auto recv_eth_config = tt_metal::EthernetConfig{
        .noc = tt_metal::NOC::NOC_0,
        .processor = processor,
        .compile_args =
            {
                transfer_size,
                transfer_count,
                barrier_address,
            },
    };
    eth_test_common::set_arch_specific_eth_config(recv_eth_config);

    auto recv_kernel = tt_metal::CreateKernel(
        *recv_program,
        "tests/tt_metal/tt_metal/deployment/kernels/eth_bidir_recv_kernel.cpp",
        recv_core,
        recv_eth_config);

    tt_metal::SetRuntimeArgs(*recv_program, recv_kernel, recv_core, {});
}

static void prepare_sender_bidir(
    tt::tt_metal::IDevice* const send_device,
    const CoreCoord& send_core,
    uint32_t transfer_size,
    uint32_t transfer_count,
    uint32_t send_delta_addr,
    std::vector<uint32_t>& inputs,
    DataMovementProcessor processor,
    uint32_t num_bytes_per_send,
    uint32_t send_l1_address,
    uint32_t recv_l1_address,
    uint32_t barrier_address,
    tt_metal::Program* send_program) {
    tt::tt_metal::MetalContext::instance().get_cluster().write_core(
        send_device->id(), send_device->ethernet_core_from_logical_core(send_core), inputs, send_l1_address);

    auto send_eth_config = tt_metal::EthernetConfig{
        .noc = tt_metal::NOC::NOC_0,
        .processor = processor,
        .compile_args =
            {
                num_bytes_per_send,
                transfer_size,
                transfer_count,
                send_delta_addr,
                send_l1_address,
                recv_l1_address,
                barrier_address,
            },
    };
    eth_test_common::set_arch_specific_eth_config(send_eth_config);

    auto send_kernel = tt_metal::CreateKernel(
        *send_program,
        "tests/tt_metal/tt_metal/deployment/kernels/eth_bidir_send_kernel.cpp",
        send_core,
        send_eth_config);

    tt_metal::SetRuntimeArgs(*send_program, send_kernel, send_core, {});
}

template <typename FIXTURE>
static bool run_test_bandwidth_bidir(
    FIXTURE* fixture,
    const std::shared_ptr<distributed::MeshDevice>& send_mesh_device,
    const std::shared_ptr<distributed::MeshDevice>& recv_mesh_device,
    const CoreCoord& send_core,
    const CoreCoord& recv_core) {
    bool same_device = send_mesh_device == recv_mesh_device;
    auto* const send_device = send_mesh_device->get_devices()[0];
    auto* const recv_device = recv_mesh_device->get_devices()[0];
    uint32_t transfer_size = 160 * 1024;
    uint32_t num_bytes_per_send = transfer_size / 2;
    uint32_t transfer_count = 20 << 10;
    uint64_t total_transferred = (uint64_t)transfer_size * transfer_count;
    DataMovementProcessor processor0 = DataMovementProcessor::RISCV_0;
    DataMovementProcessor processor1 = DataMovementProcessor::RISCV_1;

    auto inputs = generate_uniform_random_vector<uint32_t>(0, 100, transfer_size / sizeof(uint32_t));
    std::vector<uint32_t> all_zeros(inputs.size(), 0);

    struct l1_allocator alloc = new_erisc_allocator();

    uint32_t barrier_address = l1_alloc(&alloc, sizeof(struct barrier));
    uint32_t recv_l1_address = l1_alloc(&alloc, transfer_size);
    uint32_t send_delta_addr = l1_alloc(&alloc, sizeof(uint64_t));
    uint32_t send_l1_address = l1_alloc(&alloc, transfer_size);

    tt_metal::Program send_program = tt_metal::Program(), recv_program_ = tt_metal::Program();
    tt_metal::Program& recv_program = same_device ? send_program : recv_program_;

    /* Receivers */
    prepare_receiver_bidir(
        recv_device,
        recv_core,
        transfer_size,
        transfer_count,
        all_zeros,
        processor0,
        recv_l1_address,
        barrier_address,
        &recv_program);

    prepare_receiver_bidir(
        send_device,
        send_core,
        transfer_size,
        transfer_count,
        all_zeros,
        processor1,
        recv_l1_address,
        barrier_address,
        &send_program);

    /* Senders */
    prepare_sender_bidir(
        send_device,
        send_core,
        transfer_size,
        transfer_count,
        send_delta_addr,
        inputs,
        processor0,
        num_bytes_per_send,
        send_l1_address,
        recv_l1_address,
        barrier_address,
        &send_program);

    prepare_sender_bidir(
        recv_device,
        recv_core,
        transfer_size,
        transfer_count,
        send_delta_addr,
        inputs,
        processor1,
        num_bytes_per_send,
        send_l1_address,
        recv_l1_address,
        barrier_address,
        &recv_program);

    auto zero_coord = distributed::MeshCoordinate(0, 0);
    auto device_range = distributed::MeshCoordinateRange(zero_coord, zero_coord);
    wait_to_finish(fixture, send_program, recv_program, send_mesh_device, recv_mesh_device, device_range);

    bool pass = true;
    pass &= bandwidth_check(send_device, send_core, send_delta_addr, total_transferred, BANDWIDTH_THRESHOLD_BIDIR);
    pass &= bandwidth_check(recv_device, recv_core, send_delta_addr, total_transferred, BANDWIDTH_THRESHOLD_BIDIR);

    pass &= data_check(recv_device, recv_core, recv_l1_address, inputs);
    pass &= data_check(send_device, send_core, recv_l1_address, inputs);

    return pass;
}

TEST_F(UnitMeshCQProgramFixture, TensixDeploymentEthernetBandwidthBidir) {
    bool pass = true;

    SignalGuard g(SIGINT, handle_sigint);

    for (const auto& sender_mesh_device : devices_) {
        auto* const sender_device = sender_mesh_device->get_devices()[0];
        for (const auto& receiver_mesh_device : devices_) {
            auto* const receiver_device = receiver_mesh_device->get_devices()[0];

            log_info(
                tt::LogTest,
                "sender device id: {}, receiver device id: {}",
                sender_device->id(),
                receiver_device->id());

            for (const auto& sender_core : sender_device->get_active_ethernet_cores(true)) {
                auto [device_id, receiver_core] = sender_device->get_connected_ethernet_core(sender_core);
                if (receiver_device->id() != device_id) {
                    continue;
                }

                if (g_stop_requested.load()) {
                    GTEST_SKIP() << "Test interrupted by user after current test finished.";
                    return;
                }

                log_info(tt::LogTest, "  sender core: {}, receiver core: {}", sender_core, receiver_core);
                pass &= run_test_bandwidth_bidir(
                    this, sender_mesh_device, receiver_mesh_device, sender_core, receiver_core);
            }
        }
    }

    ASSERT_TRUE(pass);
}

}  // namespace tt::tt_metal
