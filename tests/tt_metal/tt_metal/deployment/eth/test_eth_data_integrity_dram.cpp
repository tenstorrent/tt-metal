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
static void prepare_receiver_integrity_dram(
    FIXTURE* fixture,
    const std::shared_ptr<distributed::MeshDevice>& recv_mesh_device,
    const CoreCoord& recv_core,
    uint32_t transfer_size,
    DataMovementProcessor processor,
    uint32_t progress_counter,
    uint32_t recv_buffer0,
    uint32_t recv_buffer1,
    uint32_t dram_start_addr,
    uint32_t dram_end_addr,
    uint32_t dram_bank_id,
    uint32_t send_bank_id,
    tt_metal::Program* recv_program,
    bool init_recv_dram) {
    /* ============= */
    tensix_zero_dram(fixture, recv_mesh_device, dram_start_addr, dram_end_addr, dram_bank_id);
    if (init_recv_dram) {
        log_info(tt::LogTest, "      initing ram bank {}", dram_bank_id);
        tensix_counter_dram(fixture, recv_mesh_device, dram_start_addr, dram_end_addr, send_bank_id);
    }

    auto recv_eth_config = tt_metal::EthernetConfig{
        .noc = tt_metal::NOC::NOC_0,
        .processor = processor,
        .compile_args =
            {
                transfer_size,
                recv_buffer0,
                recv_buffer1,
                progress_counter,
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

template <typename FIXTURE>
static void prepare_sender_integrity_dram(
    FIXTURE* fixture,
    const std::shared_ptr<distributed::MeshDevice>& send_mesh_device,
    const CoreCoord& send_core,
    uint32_t transfer_size,
    uint32_t send_delta_addr,
    DataMovementProcessor processor,
    uint32_t progress_counter,
    uint32_t num_bytes_per_send,
    uint32_t send_buffer0,
    uint32_t send_buffer1,
    uint32_t recv_buffer0,
    uint32_t recv_buffer1,
    uint32_t dram_start_addr,
    uint32_t dram_end_addr,
    uint32_t dram_bank_id,
    tt_metal::Program* send_program,
    bool init_send_dram) {
    /* ============= */

    if (init_send_dram) {
        log_info(tt::LogTest, "      initing ram bank {}", dram_bank_id);
        tensix_counter_dram(fixture, send_mesh_device, dram_start_addr, dram_end_addr, dram_bank_id);
    }

    auto send_eth_config = tt_metal::EthernetConfig{
        .noc = tt_metal::NOC::NOC_0,
        .processor = processor,
        .compile_args =
            {
                progress_counter,
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

struct test_config {
    uint32_t dram_start_addr;
    uint32_t dram_end_addr;
    uint32_t transfer_size;
    uint32_t num_bytes_per_send;
};

template <typename FIXTURE>
static bool run_test_integrity_dram(
    FIXTURE* fixture,
    const std::shared_ptr<distributed::MeshDevice>& send_mesh_device,
    const std::shared_ptr<distributed::MeshDevice>& recv_mesh_device,
    const CoreCoord& send_core,
    const CoreCoord& recv_core,
    const DataMovementProcessor processor,
    bool init_send_dram,
    bool init_recv_dram) {
    /* ============= */
    TEST_PARAM(uint32_t, dram_start_addr, 0x500000u, "ETH_TEST_START_ADDR");
    TEST_PARAM(uint32_t, dram_end_addr, 0xfe000000u, "ETH_TEST_END_ADDR");
    TEST_PARAM(uint32_t, transfer_size, 160 * 1024, "ETH_TEST_TRANSFER_SIZE");

    TT_FATAL(dram_end_addr > dram_start_addr, "End address must be greater than start address");

    uint32_t num_bytes_per_send = transfer_size;
    uint64_t total_transferred = dram_end_addr - dram_start_addr;

    struct l1_allocator send_alloc = new_erisc_allocator();
    struct l1_allocator recv_alloc = new_erisc_allocator();

    uint32_t progress_counter = l1_alloc(&recv_alloc, sizeof(uint64_t));
    uint32_t send_progress_counter = l1_alloc(&send_alloc, sizeof(uint64_t));
    TT_FATAL(progress_counter == send_progress_counter, "Progress counters should be at the same address");

    uint32_t recv_buffer0 = l1_alloc(&recv_alloc, transfer_size);
    uint32_t recv_buffer1 = l1_alloc(&recv_alloc, transfer_size);

    uint32_t send_delta_addr = l1_alloc(&send_alloc, sizeof(uint64_t));
    uint32_t send_buffer0 = l1_alloc(&send_alloc, transfer_size);
    uint32_t send_buffer1 = l1_alloc(&send_alloc, transfer_size);

    uint32_t next_bank_id = 2;
    const uint32_t recv_bank_id = next_bank_id++;
    const uint32_t send_bank_id = next_bank_id++;

    map<shared_ptr<distributed::MeshDevice>, shared_ptr<tt_metal::Program>> programs = {
        {send_mesh_device, make_shared<Program>()},
        {recv_mesh_device, make_shared<Program>()},
    };

    /* Receivers */
    prepare_receiver_integrity_dram(
        fixture,
        recv_mesh_device,
        recv_core,
        transfer_size,
        processor,
        progress_counter,
        recv_buffer0,
        recv_buffer1,
        dram_start_addr,
        dram_end_addr,
        recv_bank_id,
        send_bank_id,
        programs[recv_mesh_device].get(),
        init_recv_dram);

    /* Senders */
    prepare_sender_integrity_dram(
        fixture,
        send_mesh_device,
        send_core,
        transfer_size,
        send_delta_addr,
        processor,
        send_progress_counter,
        num_bytes_per_send,
        send_buffer0,
        send_buffer1,
        recv_buffer0,
        recv_buffer1,
        dram_start_addr,
        dram_end_addr,
        send_bank_id,
        programs[send_mesh_device].get(),
        init_send_dram);

    vector<struct core_setup> cores = {
        {
            .program = programs[send_mesh_device],
            .mesh_device = send_mesh_device,
            .core = send_core,
            .iter_l1_addr = progress_counter,
            .expected_count = dram_end_addr,
        },
        {
            .program = programs[recv_mesh_device],
            .mesh_device = recv_mesh_device,
            .core = recv_core,
            .iter_l1_addr = progress_counter,
            .expected_count = dram_end_addr,
        },
    };
    wait_to_finish_eth_timeout_cores(fixture, cores, programs);

    auto* const send_device = send_mesh_device->get_devices()[0];
    double threshold = get_eth_bw() * 0.5;

    bool pass = true;
    pass &= bandwidth_check(send_device, send_core, send_delta_addr, total_transferred, threshold);
    pass &= tensix_compare_dram_banks(
        fixture, recv_mesh_device, dram_start_addr, dram_end_addr, send_bank_id, recv_bank_id);

    return pass;
}

TEST_F(MeshDispatchFixture, TensixDeploymentEthernet03DataIntegrityDram) {
    const auto num_eriscs = MetalContext::instance().hal().get_num_risc_processors(HalProgrammableCoreType::ACTIVE_ETH);

    vector<LinkError> errors;
    int n = 0;

    distributed::MeshDevice* prev_sender = nullptr;
    distributed::MeshDevice* prev_recv = nullptr;

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

                for (uint32_t erisc_idx = 0; erisc_idx < num_eriscs; erisc_idx++) {
                    const auto processor = static_cast<DataMovementProcessor>(erisc_idx);

                    log_info(tt::LogTest, "    running on {}", processor);

                    bool passed = run_test_integrity_dram(
                        this,
                        sender_mesh_device,
                        receiver_mesh_device,
                        sender_core,
                        receiver_core,
                        processor,
                        prev_sender != sender_mesh_device.get(),
                        prev_recv != receiver_mesh_device.get());

                    if (!passed) {
                        errors.emplace_back(
                            sender_device->id(), receiver_device->id(), sender_core, receiver_core, processor);
                    }
                    log_info(tt::LogTest, "    done");

                    prev_recv = receiver_mesh_device.get();
                    prev_sender = sender_mesh_device.get();
                    n++;
                }
            }
        }
    }

    log_info(tt::LogTest, "Ran {} tests", n);

    print_summary(errors);
    ASSERT_TRUE(errors.empty());
}

}  // namespace tt::tt_metal
