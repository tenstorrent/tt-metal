// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
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

namespace tt::tt_metal {

using namespace std;
using namespace tt;
using namespace tt::test_utils;

template <typename FIXTURE>
static bool run_test_stress(
    FIXTURE* fixture,
    const std::shared_ptr<distributed::MeshDevice>& send_mesh_device,
    const std::shared_ptr<distributed::MeshDevice>& recv_mesh_device,
    const CoreCoord& send_core,
    const CoreCoord& recv_core,
    DataMovementProcessor processor0,
    span<uint32_t> inputs) {
    /* =================== */
    // bool same_device = send_mesh_device == recv_mesh_device;
    auto* const send_device = send_mesh_device->get_devices()[0];
    auto* const recv_device = recv_mesh_device->get_devices()[0];

    TEST_PARAM(uint32_t, transfer_size, 160 * 1024, "ETH_TEST_TRANSFER_SIZE");
    TEST_PARAM(uint32_t, transfer_count, 20 << 10, "ETH_TEST_TRANSFER_COUNT");

    uint32_t num_bytes_per_send = transfer_size / 2;
    uint64_t total_transferred = (uint64_t)transfer_size * transfer_count;

    span<uint32_t> inp = inputs.subspan(0, transfer_size / sizeof inp[0]);

    struct l1_allocator alloc = new_erisc_allocator();

    uint32_t iter_l1_address = l1_alloc(&alloc, sizeof(uint64_t));
    uint32_t recv_l1_address = l1_alloc(&alloc, transfer_size);
    uint32_t send_delta_addr = l1_alloc(&alloc, sizeof(uint64_t));
    uint32_t send_l1_address = l1_alloc(&alloc, transfer_size);

    // tt_metal::Program send_program = tt_metal::Program(), recv_program_ = tt_metal::Program();
    // tt_metal::Program& recv_program = same_device ? send_program : recv_program_;

    std::map<std::shared_ptr<distributed::MeshDevice>, std::shared_ptr<tt_metal::Program>> programs;

    programs[send_mesh_device] = make_shared<Program>();
    programs[recv_mesh_device] = make_shared<Program>();

    prepare_bidir(
        send_device,
        send_core,
        transfer_size,
        transfer_count,
        send_delta_addr,
        inp,
        processor0,
        num_bytes_per_send,
        iter_l1_address,
        send_l1_address,
        recv_l1_address,
        0,
        1,
        programs[send_mesh_device].get());

    prepare_bidir(
        recv_device,
        recv_core,
        transfer_size,
        transfer_count,
        send_delta_addr,
        inp,
        processor0,
        num_bytes_per_send,
        iter_l1_address,
        send_l1_address,
        recv_l1_address,
        1,
        0,
        programs[recv_mesh_device].get());

    auto zero_coord = distributed::MeshCoordinate(0, 0);
    auto device_range = distributed::MeshCoordinateRange(zero_coord, zero_coord);

    auto cores = std::vector<struct core_setup>{
        {
            .program = programs[send_mesh_device],
            .mesh_device = send_mesh_device,
            .device_range = device_range,
            .core = send_core,
        },
        {
            .program = programs[recv_mesh_device],
            .mesh_device = recv_mesh_device,
            .device_range = device_range,
            .core = recv_core,
        },
    };
    wait_to_finish_eth_timeout_cores(fixture, cores, device_range, iter_l1_address, transfer_count);

    double threshold = get_eth_bw() * 0.7;

    bool pass = true;
    pass &= bandwidth_check(send_device, send_core, send_delta_addr, total_transferred, threshold);
    pass &= bandwidth_check(recv_device, recv_core, send_delta_addr, total_transferred, threshold);

    pass &= data_check(recv_device, recv_core, recv_l1_address, inp);
    pass &= data_check(send_device, send_core, recv_l1_address, inp);

    return pass;
}

TEST_F(MeshDispatchFixture, TensixDeploymentEthernet05StressTest) {
    const auto num_eriscs = MetalContext::instance().hal().get_num_risc_processors(HalProgrammableCoreType::ACTIVE_ETH);

    vector<LinkError> errors;
    int n = 0;

    vector<uint32_t> inputs = generate_uniform_random_vector<uint32_t>(0, 100, 1 << 20);

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
                for (uint32_t erisc_idx = 0; erisc_idx < num_eriscs; erisc_idx++) {
                    const auto processor = static_cast<DataMovementProcessor>(erisc_idx);

                    log_info(tt::LogTest, "    running on {}", processor);
                    bool passed = run_test_stress(
                        this, sender_mesh_device, receiver_mesh_device, sender_core, receiver_core, processor, inputs);
                    if (!passed) {
                        errors.emplace_back(
                            sender_device->id(), receiver_device->id(), sender_core, receiver_core, processor);
                    }
                    n++;
                }
            }
        }
    }

    log_info(tt::LogTest, "Ran {} tests", n);

    print_summary(errors);
    ASSERT_TRUE(!errors.size());
}

}  // namespace tt::tt_metal
