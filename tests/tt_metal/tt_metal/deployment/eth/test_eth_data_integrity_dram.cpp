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

#define BANDWIDTH_THRESHOLD_DATA_INTEGRITY 210.0

namespace tt::tt_metal {

using namespace std;
using namespace tt;
using namespace tt::test_utils;

static void prepare_receiver_integrity_dram(
    tt::tt_metal::IDevice* const recv_device,
    const CoreCoord& recv_core,
    uint32_t transfer_size,
    DataMovementProcessor processor,
    uint32_t recv_buffer0,
    uint32_t recv_buffer1,
    uint32_t dram_start_addr,
    uint32_t dram_end_addr,
    uint32_t dram_bank_id,
    tt_metal::Program* recv_program) {
    /* ============= */
    uint64_t total_transferred = dram_end_addr - dram_start_addr;
    size_t wordcount = total_transferred / sizeof(uint32_t);
    vector<uint32_t> zeros(wordcount);
    detail::WriteToDeviceDRAMChannel(recv_device, dram_bank_id, dram_start_addr, zeros);

    auto recv_eth_config = tt_metal::EthernetConfig{
        .noc = tt_metal::NOC::NOC_0,
        .processor = processor,
        .compile_args =
            {
                transfer_size,
                recv_buffer0,
                recv_buffer1,
                dram_start_addr,
                dram_end_addr,
                dram_bank_id,
            },
    };
    eth_test_common::set_arch_specific_eth_config(recv_eth_config);

    auto recv_kernel = tt_metal::CreateKernel(
        *recv_program,
        "tests/tt_metal/tt_metal/deployment/kernels/eth_integrity_dram_recv_kernel.cpp",
        recv_core,
        recv_eth_config);

    tt_metal::SetRuntimeArgs(*recv_program, recv_kernel, recv_core, {});
}

static void prepare_sender_integrity_dram(
    tt::tt_metal::IDevice* const send_device,
    const CoreCoord& send_core,
    vector<uint32_t>& inputs,
    uint32_t transfer_size,
    uint32_t send_delta_addr,
    DataMovementProcessor processor,
    uint32_t num_bytes_per_send,
    uint32_t send_buffer0,
    uint32_t send_buffer1,
    uint32_t recv_buffer0,
    uint32_t recv_buffer1,
    uint32_t dram_start_addr,
    uint32_t dram_end_addr,
    uint32_t dram_bank_id,
    tt_metal::Program* send_program) {
    detail::WriteToDeviceDRAMChannel(send_device, dram_bank_id, dram_start_addr, inputs);

    auto send_eth_config = tt_metal::EthernetConfig{
        .noc = tt_metal::NOC::NOC_0,
        .processor = processor,
        .compile_args =
            {
                num_bytes_per_send,
                transfer_size,
                send_delta_addr,
                send_buffer0,
                send_buffer1,
                recv_buffer0,
                recv_buffer1,
                dram_start_addr,
                dram_end_addr,
                dram_bank_id,
            },
    };
    eth_test_common::set_arch_specific_eth_config(send_eth_config);

    auto send_kernel = tt_metal::CreateKernel(
        *send_program,
        "tests/tt_metal/tt_metal/deployment/kernels/eth_integrity_dram_send_kernel.cpp",
        send_core,
        send_eth_config);

    tt_metal::SetRuntimeArgs(*send_program, send_kernel, send_core, {});
}

template <typename FIXTURE>
static bool run_test_integrity_dram(
    FIXTURE* fixture,
    const std::shared_ptr<distributed::MeshDevice>& send_mesh_device,
    const std::shared_ptr<distributed::MeshDevice>& recv_mesh_device,
    const CoreCoord& send_core,
    const CoreCoord& recv_core,
    const DataMovementProcessor processor) {
    /* ============= */
    bool same_device = send_mesh_device == recv_mesh_device;
    auto* const send_device = send_mesh_device->get_devices()[0];
    auto* const recv_device = recv_mesh_device->get_devices()[0];

    uint32_t next_bank_id = 0;
    uint32_t dram_start_addr = 0x500000u;
    uint32_t dram_end_addr = 0xff000000u;
    TT_FATAL(dram_end_addr > dram_start_addr, "End address must be greater than start address");

    uint32_t transfer_size = 160 * 1024;
    uint32_t num_bytes_per_send = transfer_size;
    uint64_t total_transferred = dram_end_addr - dram_start_addr;

    uint32_t c = 0;
    size_t wordcount = total_transferred / sizeof(uint32_t);
    vector<uint32_t> inputs;
    inputs.reserve(wordcount);
    for (long i = 0; i < wordcount; i++) {
        inputs.push_back(c++);
    }

    struct l1_allocator send_alloc = new_erisc_allocator();
    struct l1_allocator recv_alloc = new_erisc_allocator();

    uint32_t recv_buffer0 = l1_alloc(&recv_alloc, transfer_size);
    uint32_t recv_buffer1 = l1_alloc(&recv_alloc, transfer_size);
    uint32_t send_delta_addr = l1_alloc(&send_alloc, sizeof(uint64_t));
    uint32_t send_buffer0 = l1_alloc(&send_alloc, transfer_size);
    uint32_t send_buffer1 = l1_alloc(&send_alloc, transfer_size);

    tt_metal::Program send_program = tt_metal::Program(), recv_program_ = tt_metal::Program();
    tt_metal::Program& recv_program = same_device ? send_program : recv_program_;

    /* Receivers */
    uint32_t recv_bank_id = 0;
    prepare_receiver_integrity_dram(
        recv_device,
        recv_core,
        transfer_size,
        processor,
        recv_buffer0,
        recv_buffer1,
        dram_start_addr,
        dram_end_addr,
        recv_bank_id = next_bank_id++,
        &recv_program);

    /* Senders */
    prepare_sender_integrity_dram(
        send_device,
        send_core,
        inputs,
        transfer_size,
        send_delta_addr,
        processor,
        num_bytes_per_send,
        send_buffer0,
        send_buffer1,
        recv_buffer0,
        recv_buffer1,
        dram_start_addr,
        dram_end_addr,
        next_bank_id++,
        &send_program);

    auto zero_coord = distributed::MeshCoordinate(0, 0);
    auto device_range = distributed::MeshCoordinateRange(zero_coord, zero_coord);
    wait_to_finish(fixture, send_program, recv_program, send_mesh_device, recv_mesh_device, device_range);

    bool pass = true;
    pass &= eth_bandwidth_check(
        send_device, send_core, send_delta_addr, total_transferred, BANDWIDTH_THRESHOLD_DATA_INTEGRITY);

    // Reading from chip too slow, use tensix to compare?
    pass &= dram_data_check(recv_device, dram_start_addr, dram_end_addr, recv_bank_id, inputs);
    return pass;
}

TEST_F(UnitMeshCQProgramFixture, TensixDeploymentEthernetDataIntegrityDram) {
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

            for (const auto& sender_core : get_intra_cluster_active_eth_cores(sender_device)) {
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
                    pass &= run_test_integrity_dram(
                        this, sender_mesh_device, receiver_mesh_device, sender_core, receiver_core, processor);
                }
            }
        }
    }

    ASSERT_TRUE(pass);
}

}  // namespace tt::tt_metal
