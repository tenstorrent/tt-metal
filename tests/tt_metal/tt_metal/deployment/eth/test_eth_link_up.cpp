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

namespace tt::tt_metal {

using namespace std;
using namespace tt;
using namespace tt::test_utils;

template <typename FIXTURE>
static bool run_test(
    FIXTURE* fixture,
    const std::shared_ptr<distributed::MeshDevice>& send_mesh_device,
    const std::shared_ptr<distributed::MeshDevice>& recv_mesh_device,
    const CoreCoord& send_core,
    const CoreCoord& recv_core,
    DataMovementProcessor processor = DataMovementProcessor::RISCV_0) {
    bool same_device = send_mesh_device == recv_mesh_device;
    auto* const send_device = send_mesh_device->get_devices()[0];
    auto* const recv_device = recv_mesh_device->get_devices()[0];
    uint32_t num_bytes_per_send = 16;
    uint32_t transfer_size = 1024;
    uint32_t transfer_count = 1;

    auto inputs = generate_uniform_random_vector<uint32_t>(0, 100, transfer_size / sizeof(uint32_t));

    struct l1_allocator send_allocator = new_erisc_allocator();
    struct l1_allocator recv_allocator = new_erisc_allocator();

    uint32_t recv_l1_address = 0;

    tt_metal::Program send_program = tt_metal::Program(), recv_program_ = tt_metal::Program();
    tt_metal::Program& recv_program = same_device ? send_program : recv_program_;

    prepare_receiver(
        recv_device,
        recv_core,
        &recv_allocator,
        transfer_size,
        transfer_count,
        inputs,
        processor,
        &recv_l1_address,
        &recv_program);

    uint32_t send_delta_addr = 0;
    prepare_sender(
        send_device,
        send_core,
        &send_allocator,
        transfer_size,
        transfer_count,
        &send_delta_addr,
        inputs,
        processor,
        num_bytes_per_send,
        recv_l1_address,
        &send_program);

    auto zero_coord = distributed::MeshCoordinate(0, 0);
    auto device_range = distributed::MeshCoordinateRange(zero_coord, zero_coord);
    wait_to_finish(fixture, send_program, recv_program, send_mesh_device, recv_mesh_device, device_range);

    bool pass = true;
    pass &= data_check(recv_device, recv_core, recv_l1_address, inputs);
    return pass;
}

TEST_F(UnitMeshCQProgramFixture, TensixDeploymentEthernetLinkUp) {
    const auto num_eriscs = MetalContext::instance().hal().get_num_risc_processors(HalProgrammableCoreType::ACTIVE_ETH);

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
                for (uint32_t erisc_idx = 0; erisc_idx < num_eriscs; erisc_idx++) {
                    const auto processor = static_cast<DataMovementProcessor>(erisc_idx);

                    log_info(tt::LogTest, "    running on {}", processor);
                    pass &=
                        run_test(this, sender_mesh_device, receiver_mesh_device, sender_core, receiver_core, processor);
                }
            }
        }
    }

    ASSERT_TRUE(pass);
}

}  // namespace tt::tt_metal
