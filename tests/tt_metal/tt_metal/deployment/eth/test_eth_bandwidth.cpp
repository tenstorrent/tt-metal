// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/tt_metal/deployment/deployment_common.hpp"

#include <gtest/gtest.h>
#include <tt-logger/tt-logger.hpp>
#include <tt_stl/assert.hpp>

#include "tt_metal/test_utils/stimulus.hpp"
#include "command_queue_fixture.hpp"
#include "tt_metal/tt_metal/eth/eth_test_common.hpp"

#define BANDWIDTH_THRESHOLD 300.0

namespace tt::tt_metal {

using namespace std;
using namespace tt;
using namespace tt::test_utils;

template <typename FIXTURE>
static bool run_test_bandwidth(
    FIXTURE* fixture,
    const std::shared_ptr<distributed::MeshDevice>& send_mesh_device,
    const std::shared_ptr<distributed::MeshDevice>& recv_mesh_device,
    const CoreCoord& send_core,
    const CoreCoord& recv_core,
    DataMovementProcessor processor = DataMovementProcessor::RISCV_0) {
    bool same_device = send_mesh_device == recv_mesh_device;
    auto* const send_device = send_mesh_device->get_devices()[0];
    auto* const recv_device = recv_mesh_device->get_devices()[0];
    uint32_t num_bytes_per_send = 20000;
    uint32_t transfer_size = 200 * 1024;
    uint32_t transfer_count = 1024;
    uint64_t total_transferred = (uint64_t)transfer_size * transfer_count;

    auto inputs = generate_uniform_random_vector<uint32_t>(0, 100, transfer_size / sizeof(uint32_t));
    std::vector<uint32_t> all_zeros(inputs.size(), 0);

    struct l1_allocator send_allocator = new_erisc_allocator();
    uint32_t send_delta_addr = l1_alloc(send_allocator, sizeof(uint64_t));
    uint32_t send_l1_address = l1_alloc(send_allocator, transfer_size);
    tt::tt_metal::MetalContext::instance().get_cluster().write_core(
        send_device->id(), send_device->ethernet_core_from_logical_core(send_core), inputs, send_l1_address);

    struct l1_allocator recv_allocator = new_erisc_allocator();
    uint32_t recv_l1_address = l1_alloc(recv_allocator, transfer_size);
    tt::tt_metal::MetalContext::instance().get_cluster().write_core(
        recv_device->id(), recv_device->ethernet_core_from_logical_core(recv_core), all_zeros, recv_l1_address);

    auto zero_coord = distributed::MeshCoordinate(0, 0);
    auto device_range = distributed::MeshCoordinateRange(zero_coord, zero_coord);
    distributed::MeshWorkload send_workload;
    tt_metal::Program send_program = tt_metal::Program();

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
            },
    };
    eth_test_common::set_arch_specific_eth_config(send_eth_config);

    auto send_kernel = tt_metal::CreateKernel(
        send_program,
        "tests/tt_metal/tt_metal/deployment/kernels/eth_simple_send_kernel.cpp",
        send_core,
        send_eth_config);

    tt_metal::SetRuntimeArgs(send_program, send_kernel, send_core, {});

    distributed::MeshWorkload recv_workload_;
    tt_metal::Program recv_program_ = tt_metal::Program();

    distributed::MeshWorkload& recv_workload = same_device ? send_workload : recv_workload_;
    tt_metal::Program& recv_program = same_device ? send_program : recv_program_;

    auto recv_eth_config = tt_metal::EthernetConfig{
        .noc = tt_metal::NOC::NOC_0,
        .processor = processor,
        .compile_args =
            {
                transfer_size,
                transfer_count,
            },
    };
    eth_test_common::set_arch_specific_eth_config(recv_eth_config);

    auto recv_kernel = tt_metal::CreateKernel(
        recv_program,
        "tests/tt_metal/tt_metal/deployment/kernels/eth_simple_recv_kernel.cpp",
        recv_core,
        recv_eth_config);

    tt_metal::SetRuntimeArgs(recv_program, recv_kernel, recv_core, {});

    send_workload.add_program(device_range, std::move(send_program));
    if (!same_device) {
        recv_workload.add_program(device_range, std::move(recv_program));
    }

    fixture->RunProgram(send_mesh_device, send_workload, true);
    if (!same_device) {
        fixture->RunProgram(recv_mesh_device, recv_workload, true);
    }

    fixture->FinishCommands(send_mesh_device);
    if (!same_device) {
        fixture->FinishCommands(recv_mesh_device);
    }

    auto readback_vec = tt::tt_metal::MetalContext::instance().get_cluster().read_core(
        recv_device->id(), recv_device->ethernet_core_from_logical_core(recv_core), recv_l1_address, transfer_size);

    uint64_t delta = read_l1_u64(send_device, send_core, send_delta_addr);
    double deltas = delta / 1.35e9; /* Assuming fixed max frequency */
    double bandwidth = 8 * total_transferred / 1e9 / deltas;
    log_info(tt::LogTest, "      Bandwidth {:.3f} Gbps, {:.3f} ms", bandwidth, deltas * 1000);

    bool pass = true;
    if (bandwidth < BANDWIDTH_THRESHOLD) {
        pass = false;
        log_info(tt::LogTest, "      Expected at least: {} Gbps, got {:.2f} Gbps", BANDWIDTH_THRESHOLD, bandwidth);
    }

    if (readback_vec != inputs) {
        pass = false;
        for (int i = 0; i < inputs.size(); i++) {
            if (inputs[i] != readback_vec[i]) {
                log_info(tt::LogTest, "      Mismatch at index: {}", i);
            }
        }
        log_info(tt::LogTest, "      Mismatch at Core: {}", recv_core);
    }

    return pass;
}

TEST_F(UnitMeshCQProgramFixture, TensixDeploymentEthernetBandwidth) {
    const auto num_eriscs = MetalContext::instance().hal().get_num_risc_processors(HalProgrammableCoreType::ACTIVE_ETH);

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
                for (uint32_t erisc_idx = 0; erisc_idx < num_eriscs; erisc_idx++) {
                    const auto processor = static_cast<DataMovementProcessor>(erisc_idx);

                    log_info(tt::LogTest, "    running on {}", processor);
                    pass &= run_test_bandwidth(
                        this, sender_mesh_device, receiver_mesh_device, sender_core, receiver_core, processor);
                }
            }
        }
    }

    ASSERT_TRUE(pass);
}

}  // namespace tt::tt_metal
