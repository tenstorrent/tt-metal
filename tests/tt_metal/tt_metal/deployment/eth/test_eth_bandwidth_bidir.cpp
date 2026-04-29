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

static void prepare_bidir(
    tt::tt_metal::IDevice* const send_device,
    const CoreCoord& send_core,
    uint32_t transfer_size,
    uint32_t transfer_count,
    uint32_t send_delta_addr,
    std::vector<uint32_t>& inputs,
    DataMovementProcessor processor,
    uint32_t num_bytes_per_send,
    uint32_t iter_l1_address,
    uint32_t send_l1_address,
    uint32_t recv_l1_address,
    uint32_t channel0,
    uint32_t channel1,
    tt_metal::Program* send_program) {
    /* =================== */
    tt::tt_metal::MetalContext::instance().get_cluster().write_core(
        send_device->id(), send_device->ethernet_core_from_logical_core(send_core), inputs, send_l1_address);

    auto send_eth_config = tt_metal::EthernetConfig{
        .noc = tt_metal::NOC::NOC_0,
        .processor = processor,
        .compile_args =
            {
                channel0,
                channel1,
                iter_l1_address,
                num_bytes_per_send,
                transfer_size,
                transfer_count,
                send_delta_addr,
                send_l1_address,
                recv_l1_address,
            },
    };
    eth_test_common::set_arch_specific_eth_config(send_eth_config);

    auto send_kernel = tt_metal::CreateKernel(
        *send_program, "tests/tt_metal/tt_metal/deployment/kernels/eth_bidir_kernel.cpp", send_core, send_eth_config);

    tt_metal::SetRuntimeArgs(*send_program, send_kernel, send_core, {});
}

template <typename FIXTURE>
static bool run_test_bandwidth_bidir(
    FIXTURE* fixture,
    const std::shared_ptr<distributed::MeshDevice>& send_mesh_device,
    const std::shared_ptr<distributed::MeshDevice>& recv_mesh_device,
    const CoreCoord& send_core,
    const CoreCoord& recv_core,
    DataMovementProcessor processor0) {
    /* =================== */
    bool same_device = send_mesh_device == recv_mesh_device;
    auto* const send_device = send_mesh_device->get_devices()[0];
    auto* const recv_device = recv_mesh_device->get_devices()[0];

    TEST_PARAM(uint32_t, transfer_size, 160 * 1024, "ETH_TEST_TRANSFER_SIZE");
    TEST_PARAM(uint32_t, transfer_count, 20 << 10, "ETH_TEST_TRANSFER_COUNT");

    uint32_t num_bytes_per_send = transfer_size / 2;
    uint64_t total_transferred = (uint64_t)transfer_size * transfer_count;

    auto inputs = generate_uniform_random_vector<uint32_t>(0, 100, transfer_size / sizeof(uint32_t));
    std::vector<uint32_t> all_zeros(inputs.size(), 0);

    struct l1_allocator alloc = new_erisc_allocator();

    uint32_t iter_l1_address = l1_alloc(&alloc, sizeof(uint64_t));
    uint32_t recv_l1_address = l1_alloc(&alloc, transfer_size);
    uint32_t send_delta_addr = l1_alloc(&alloc, sizeof(uint64_t));
    uint32_t send_l1_address = l1_alloc(&alloc, transfer_size);

    tt_metal::Program send_program = tt_metal::Program(), recv_program_ = tt_metal::Program();
    tt_metal::Program& recv_program = same_device ? send_program : recv_program_;

    prepare_bidir(
        send_device,
        send_core,
        transfer_size,
        transfer_count,
        send_delta_addr,
        inputs,
        processor0,
        num_bytes_per_send,
        iter_l1_address,
        send_l1_address,
        recv_l1_address,
        0,
        1,
        &send_program);

    prepare_bidir(
        recv_device,
        recv_core,
        transfer_size,
        transfer_count,
        send_delta_addr,
        inputs,
        processor0,
        num_bytes_per_send,
        iter_l1_address,
        send_l1_address,
        recv_l1_address,
        1,
        0,
        &recv_program);

    auto zero_coord = distributed::MeshCoordinate(0, 0);
    auto device_range = distributed::MeshCoordinateRange(zero_coord, zero_coord);
    wait_to_finish_eth_timeout(
        fixture,
        send_program,
        recv_program,
        send_mesh_device,
        recv_mesh_device,
        device_range,
        send_core,
        recv_core,
        recv_l1_address,
        iter_l1_address,
        transfer_count);

    bool pass = true;
    pass &= bandwidth_check(send_device, send_core, send_delta_addr, total_transferred, BANDWIDTH_THRESHOLD_BIDIR);
    pass &= bandwidth_check(recv_device, recv_core, send_delta_addr, total_transferred, BANDWIDTH_THRESHOLD_BIDIR);

    // pass &= data_check(recv_device, recv_core, recv_l1_address, inputs) || true;
    // pass &= data_check(send_device, send_core, recv_l1_address, inputs) || true;

    return pass;
}

TEST_F(MeshDispatchFixture, TensixDeploymentEthernetBandwidthBidir) {
    const auto num_eriscs = MetalContext::instance().hal().get_num_risc_processors(HalProgrammableCoreType::ACTIVE_ETH);

    bool pass = true;
    int n = 0;

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
                    pass &= run_test_bandwidth_bidir(
                        this, sender_mesh_device, receiver_mesh_device, sender_core, receiver_core, processor);
                    n++;
                }
            }
        }
    }

    log_info(tt::LogTest, "Ran {} tests", n);
    ASSERT_TRUE(pass);
}

}  // namespace tt::tt_metal
