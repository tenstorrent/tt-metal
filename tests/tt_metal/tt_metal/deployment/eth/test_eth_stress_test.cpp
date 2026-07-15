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
static void run_test_stress(
    FIXTURE* fixture,
    const std::shared_ptr<distributed::MeshDevice>& send_mesh_device,
    const std::shared_ptr<distributed::MeshDevice>& recv_mesh_device,
    const CoreCoord& send_core,
    const CoreCoord& recv_core,
    DataMovementProcessor processor0,
    span<uint32_t> inputs,
    string locinfo,
    vector<struct core_setup>& cores,
    map<shared_ptr<distributed::MeshDevice>, shared_ptr<tt_metal::Program>> programs) {
    /* =================== */
    (void)fixture;
    auto* const send_device = send_mesh_device->get_devices()[0];
    auto* const recv_device = recv_mesh_device->get_devices()[0];

    TEST_PARAM(uint32_t, transfer_size, 160 * 1024, "ETH_TEST_TRANSFER_SIZE");
    TEST_PARAM(uint32_t, transfer_count, 2 << 20, "ETH_TEST_TRANSFER_COUNT");

    uint32_t num_bytes_per_send = transfer_size / 2;
    uint64_t total_transferred = (uint64_t)transfer_size * transfer_count;

    span<uint32_t> inp = inputs.subspan(0, transfer_size / sizeof inp[0]);

    struct l1_allocator alloc = new_erisc_allocator();

    uint32_t iter_l1_address = l1_alloc(&alloc, sizeof(uint64_t));
    uint32_t recv_l1_address = l1_alloc(&alloc, transfer_size);
    uint32_t send_delta_addr = l1_alloc(&alloc, sizeof(uint64_t));
    uint32_t send_l1_address = l1_alloc(&alloc, transfer_size);

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

    double threshold = get_eth_bw() * 0.7;

    cores.emplace_back(
        programs[send_mesh_device],
        send_mesh_device,
        send_core,
        locinfo,
        iter_l1_address,
        transfer_count,
        send_delta_addr,
        total_transferred,
        threshold,
        recv_l1_address,
        inp);

    cores.emplace_back(
        programs[recv_mesh_device],
        recv_mesh_device,
        recv_core,
        std::move(locinfo),
        iter_l1_address,
        transfer_count,
        send_delta_addr,
        total_transferred,
        threshold,
        recv_l1_address,
        inp);

    log_info(tt::LogTest, "      set up");
}

TEST_F(MeshDispatchFixture, TensixDeploymentEthernet05StressTest) {
    vector<struct core_setup> cores;
    vector<LinkError> errors;
    map<shared_ptr<distributed::MeshDevice>, shared_ptr<tt_metal::Program>> programs;
    int n = 0;

    vector<uint32_t> inputs = generate_uniform_random_vector<uint32_t>(0, 100, 1 << 20);

    print_detected_devices();
    ASSERT_TRUE(ensure_links(devices_));

    for (int i = 0; i < devices_.size(); i++) {
        const auto& sender_mesh_device = devices_[i];
        auto* const sender_device = sender_mesh_device->get_devices()[0];

        for (int j = i; j < devices_.size(); j++) {
            const auto& receiver_mesh_device = devices_[j];
            auto* const receiver_device = receiver_mesh_device->get_devices()[0];

            log_info(
                tt::LogTest,
                "sender device id: {} ({}, {}), receiver device id: {} ({}, {})",
                sender_device->id(),
                pci_bdf_for_device_id(sender_device->id()),
                get_ubb(sender_device),
                receiver_device->id(),
                pci_bdf_for_device_id(receiver_device->id()),
                get_ubb(receiver_device));

            std::set<CoreCoord> tested;

            for (const auto& sender_core : sender_device->get_active_ethernet_cores(true)) {
                auto [device_id, receiver_core] = sender_device->get_connected_ethernet_core(sender_core);
                if (receiver_device->id() != device_id) {
                    continue;
                }

                if (tested.contains(sender_core)) {
                    continue;
                }
                if (tested.contains(receiver_core)) {
                    continue;
                }

                tested.insert(sender_core);
                tested.insert(receiver_core);

                if (!programs.contains(sender_mesh_device)) {
                    programs[sender_mesh_device] = make_shared<Program>();
                }
                if (!programs.contains(receiver_mesh_device)) {
                    programs[receiver_mesh_device] = make_shared<Program>();
                }

                log_info(
                    tt::LogTest,
                    "  sender core: {}, receiver core: {} ({})",
                    sender_core,
                    receiver_core,
                    get_connector(sender_device, sender_core));

                const auto processor = static_cast<DataMovementProcessor>(0);

                log_info(tt::LogTest, "    running on {}", processor);
                string locinfo = get_locinfo(sender_device, sender_core, receiver_device, receiver_core, processor);

                run_test_stress(
                    this,
                    sender_mesh_device,
                    receiver_mesh_device,
                    sender_core,
                    receiver_core,
                    processor,
                    inputs,
                    std::move(locinfo),
                    cores,
                    programs);
                // if (!passed) {
                //     errors.emplace_back(
                //         sender_device->id(), receiver_device->id(), sender_core, receiver_core, processor);
                // }
                n++;
            }
        }
    }

    wait_to_finish_eth_timeout_cores(this, cores, programs);

    bool pass = true;

    pass &= test_check_cores(cores);

    log_info(tt::LogTest, "Ran {} tests", n);
    ASSERT_TRUE(pass);

    print_summary(errors);
    ASSERT_TRUE(errors.empty());
}

}  // namespace tt::tt_metal
