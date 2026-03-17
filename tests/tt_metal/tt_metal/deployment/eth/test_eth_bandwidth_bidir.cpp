// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/tt_metal/deployment/eth/common.hpp"
#include "tt_metal/tt_metal/deployment/deployment_common.hpp"

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
    uint32_t transfer_size = 150 * 1024;
    uint32_t num_bytes_per_send = 2 << 10;  // transfer_size / 8;
    uint32_t transfer_count = 8;            // 10240;
    uint64_t total_transferred = (uint64_t)transfer_size * transfer_count;
    DataMovementProcessor processor0 = DataMovementProcessor::RISCV_0;
    DataMovementProcessor processor1 = DataMovementProcessor::RISCV_1;

    auto inputs = generate_uniform_random_vector<uint32_t>(0, 100, transfer_size / sizeof(uint32_t));

    struct l1_allocator send_allocator = new_erisc_allocator();
    struct l1_allocator recv_allocator = new_erisc_allocator();
    // struct l1_allocator& recv_allocator = send_allocator;  // new_erisc_allocator();

    uint32_t recv_l1_address0 = 0;
    uint32_t recv_l1_address1 = 0;

    tt_metal::Program send_program = tt_metal::Program(), recv_program_ = tt_metal::Program();
    tt_metal::Program& recv_program = same_device ? send_program : recv_program_;

    prepare_receiver(
        recv_device,
        recv_core,
        &recv_allocator,
        transfer_size,
        transfer_count,
        inputs,
        processor0,
        &recv_l1_address0,
        &recv_program);

    prepare_receiver(
        send_device,
        send_core,
        &send_allocator,
        transfer_size,
        transfer_count,
        inputs,
        processor1,
        &recv_l1_address1,
        &send_program);

    uint32_t send_delta_addr0 = 0;
    uint32_t send_delta_addr1 = 0;

    prepare_sender(
        send_device,
        send_core,
        &send_allocator,
        transfer_size,
        transfer_count,
        &send_delta_addr0,
        inputs,
        processor0,
        num_bytes_per_send,
        recv_l1_address0,
        &send_program);

    prepare_sender(
        recv_device,
        recv_core,
        &recv_allocator,
        transfer_size,
        transfer_count,
        &send_delta_addr1,
        inputs,
        processor1,
        num_bytes_per_send,
        recv_l1_address1,
        &recv_program);

    auto zero_coord = distributed::MeshCoordinate(0, 0);
    auto device_range = distributed::MeshCoordinateRange(zero_coord, zero_coord);
    wait_to_finish(fixture, send_program, recv_program, send_mesh_device, recv_mesh_device, device_range);

    bool pass = true;
    pass &= bandwidth_check(send_device, send_core, send_delta_addr0, total_transferred, BANDWIDTH_THRESHOLD_BIDIR);
    pass &= bandwidth_check(recv_device, recv_core, send_delta_addr1, total_transferred, BANDWIDTH_THRESHOLD_BIDIR);

    pass &= data_check(recv_device, recv_core, recv_l1_address0, inputs);
    pass &= data_check(send_device, send_core, recv_l1_address1, inputs);

    return pass;
}

TEST_F(UnitMeshCQProgramFixture, TensixDeploymentEthernetBandwidthBidir) {
    bool pass = true;

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

                log_info(tt::LogTest, "  sender core: {}, receiver core: {}", sender_core, receiver_core);
                pass &= run_test_bandwidth_bidir(
                    this, sender_mesh_device, receiver_mesh_device, sender_core, receiver_core);
            }
        }
    }

    ASSERT_TRUE(pass);
}

}  // namespace tt::tt_metal
