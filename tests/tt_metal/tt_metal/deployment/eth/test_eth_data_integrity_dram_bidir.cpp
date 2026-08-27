// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/tt_metal/deployment/deployment_common.hpp"
#include "tt_metal/tt_metal/deployment/eth/common.hpp"
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
static void prepare_bidir_integrity(
    FIXTURE* fixture,
    const std::shared_ptr<distributed::MeshDevice>& mesh_device,
    const CoreCoord& send_core,
    uint32_t transfer_size,
    uint32_t send_delta_addr,
    DataMovementProcessor processor,
    uint32_t dram_start_addr,
    uint32_t dram_end_addr,
    uint32_t num_bytes_per_send,
    uint32_t iter_l1_address,
    uint32_t channel0,
    uint32_t channel1,
    uint32_t read_bank,
    uint32_t write_bank,
    uint32_t sendbuf0,
    uint32_t sendbuf1,
    uint32_t recvbuf0,
    uint32_t recvbuf1,
    tt_metal::Program* send_program) {
    /* =================== */
    tensix_counter_dram(fixture, mesh_device, dram_start_addr, dram_end_addr, read_bank);
    tensix_zero_dram(fixture, mesh_device, dram_start_addr, dram_end_addr, write_bank);

    auto send_eth_config = tt_metal::EthernetConfig{
        .noc = tt_metal::NOC::NOC_0,
        .processor = processor,
        .compile_args =
            {
                num_bytes_per_send,
                transfer_size,
                iter_l1_address,
                dram_start_addr,
                dram_end_addr,
                send_delta_addr,
                sendbuf0,
                sendbuf1,
                recvbuf0,
                recvbuf1,
            },
    };
    eth_test_common::set_arch_specific_eth_config(send_eth_config);

    auto send_kernel = tt_metal::CreateKernel(
        *send_program,
        "tests/tt_metal/tt_metal/deployment/kernels/eth_integrity_dram_bidir_kernel.cpp",
        send_core,
        send_eth_config);

    tt_metal::SetRuntimeArgs(
        *send_program,
        send_kernel,
        send_core,
        {
            channel0,
            channel1,
            read_bank,
            write_bank,
        });
}

template <typename FIXTURE>
static bool run_test_integrity_dram_bidir(
    FIXTURE* fixture,
    const std::shared_ptr<distributed::MeshDevice>& send_mesh_device,
    const std::shared_ptr<distributed::MeshDevice>& recv_mesh_device,
    const CoreCoord& send_core,
    const CoreCoord& recv_core,
    DataMovementProcessor processor0 = DataMovementProcessor::RISCV_0) {
    /* ============= */
    auto* const send_device = send_mesh_device->get_devices()[0];
    auto* const recv_device = recv_mesh_device->get_devices()[0];

    TEST_PARAM(uint32_t, dram_start_addr, 0x500000, "ETH_TEST_START_ADDR");
    TEST_PARAM(uint32_t, dram_end_addr, 0xfe000000u, "ETH_TEST_END_ADDR");
    // TEST_PARAM(uint32_t, dram_end_addr, 0x758000, "ETH_TEST_END_ADDR");
    TEST_PARAM(uint32_t, transfer_size, 80 * 1024, "ETH_TEST_TRANSFER_SIZE");

    TT_FATAL(dram_end_addr > dram_start_addr, "End address must be greater than start address");

    // TODO
    dram_end_addr = dram_start_addr + ((dram_end_addr - dram_start_addr) / transfer_size) * transfer_size;

    uint32_t num_bytes_per_send = transfer_size;
    uint64_t total_transferred = dram_end_addr - dram_start_addr;

    TT_FATAL(total_transferred % 16 == 0, "Transfers need to be done in 16 byte divisible sizes");

    uint32_t next_bank_id = 0;
    const uint32_t send_bank_id0 = next_bank_id++;
    const uint32_t recv_bank_id0 = next_bank_id++;
    const uint32_t send_bank_id1 = next_bank_id++;
    const uint32_t recv_bank_id1 = next_bank_id++;

    struct l1_allocator alloc = new_erisc_allocator();

    uint32_t iter_l1_address = l1_alloc(&alloc, sizeof(uint64_t));
    uint32_t send_delta_addr = l1_alloc(&alloc, sizeof(uint64_t));
    uint32_t sendbuf0 = l1_alloc(&alloc, transfer_size);
    uint32_t sendbuf1 = l1_alloc(&alloc, transfer_size);
    uint32_t recvbuf0 = l1_alloc(&alloc, transfer_size);
    uint32_t recvbuf1 = l1_alloc(&alloc, transfer_size);

    map<shared_ptr<distributed::MeshDevice>, shared_ptr<tt_metal::Program>> programs = {
        {send_mesh_device, make_shared<Program>()},
        {recv_mesh_device, make_shared<Program>()},
    };

    prepare_bidir_integrity(
        fixture,
        send_mesh_device,
        send_core,
        transfer_size,
        send_delta_addr,
        processor0,
        dram_start_addr,
        dram_end_addr,
        num_bytes_per_send,
        iter_l1_address,
        0,
        1,
        send_bank_id0,
        recv_bank_id0,
        sendbuf0,
        sendbuf1,
        recvbuf0,
        recvbuf1,
        programs[send_mesh_device].get());

    prepare_bidir_integrity(
        fixture,
        recv_mesh_device,
        recv_core,
        transfer_size,
        send_delta_addr,
        processor0,
        dram_start_addr,
        dram_end_addr,
        num_bytes_per_send,
        iter_l1_address,
        1,
        0,
        send_bank_id1,
        recv_bank_id1,
        sendbuf0,
        sendbuf1,
        recvbuf0,
        recvbuf1,
        programs[recv_mesh_device].get());

    auto zero_coord = distributed::MeshCoordinate(0, 0);
    auto device_range = distributed::MeshCoordinateRange(zero_coord, zero_coord);

    vector<struct core_setup> cores = {
        {
            .program = programs[send_mesh_device],
            .mesh_device = send_mesh_device,
            .core = send_core,
            .iter_l1_addr = iter_l1_address,
            .expected_count = dram_end_addr,
        },
        {
            .program = programs[recv_mesh_device],
            .mesh_device = recv_mesh_device,
            .core = recv_core,
            .iter_l1_addr = iter_l1_address,
            .expected_count = dram_end_addr,
        },
    };
    wait_to_finish_eth_timeout_cores(fixture, cores, programs);

    double threshold = 140; /* NOTE: Same on both bh glx and p150 */

    bool pass = true;
    pass &= bandwidth_check(send_device, send_core, send_delta_addr, total_transferred, threshold);
    pass &= bandwidth_check(recv_device, recv_core, send_delta_addr, total_transferred, threshold);

    pass &= tensix_compare_dram_banks(
        fixture, send_mesh_device, dram_start_addr, dram_end_addr, send_bank_id0, recv_bank_id0);
    pass &= tensix_compare_dram_banks(
        fixture, recv_mesh_device, dram_start_addr, dram_end_addr, send_bank_id1, recv_bank_id1);

    return pass;
}

TEST_F(MeshDispatchFixture, TensixDeploymentEthernet04DataIntegrityDramBidir) {
    vector<LinkError> errors;
    int n = 0;

    print_detected_devices();
    ASSERT_TRUE(ensure_links(devices_));

    for (const auto& sender_mesh_device : devices_) {
        auto* const sender_device = sender_mesh_device->get_devices()[0];
        for (const auto& receiver_mesh_device : devices_) {
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

            for (const auto& sender_core : sender_device->get_active_ethernet_cores(true)) {
                if (!eth_core_connects_within_cluster(sender_device, sender_core)) {
                    continue;
                }
                auto [device_id, receiver_core] = sender_device->get_connected_ethernet_core(sender_core);
                if (receiver_device->id() != device_id) {
                    continue;
                }

                log_info(
                    tt::LogTest,
                    "  sender core: {}, receiver core: {} ({})",
                    sender_core,
                    receiver_core,
                    get_connector(sender_device, sender_core));

                bool passed = run_test_integrity_dram_bidir(
                    this, sender_mesh_device, receiver_mesh_device, sender_core, receiver_core);
                if (!passed) {
                    errors.emplace_back(
                        sender_device->id(),
                        receiver_device->id(),
                        sender_core,
                        receiver_core,
                        DataMovementProcessor::RISCV_0);
                }
                log_info(tt::LogTest, "    done");

                n++;
            }
        }
    }

    log_info(tt::LogTest, "Ran {} tests", n);

    print_summary(errors);
    ASSERT_TRUE(errors.empty());
}

}  // namespace tt::tt_metal
