// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
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
#include "distributed/mesh_device_impl.hpp"

namespace tt::tt_metal {

using namespace std;
using namespace tt;
using namespace tt::tt_metal;
using namespace tt::test_utils;

template <typename FIXTURE>
static bool run_test_integrity_dram_dist(
    FIXTURE* fixture,
    const std::shared_ptr<distributed::MeshDevice>& mesh_device,
    const CoreCoord& core,
    int is_sender,
    const DataMovementProcessor processor,
    bool init_dram) {
    /* ============= */
    TEST_PARAM(uint32_t, dram_start_addr, 0x500000u, "ETH_TEST_START_ADDR");
    TEST_PARAM(uint32_t, dram_end_addr, 0xfe000000u, "ETH_TEST_END_ADDR");
    TEST_PARAM(uint32_t, transfer_size, 160 * 1024, "ETH_TEST_TRANSFER_SIZE");

    TT_FATAL(dram_end_addr > dram_start_addr, "End address must be greater than start address");

    uint32_t num_bytes_per_send = transfer_size;
    // uint64_t total_transferred = dram_end_addr - dram_start_addr;

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
        {mesh_device, make_shared<Program>()},
    };

    /* Receivers */
    prepare_receiver_integrity_dram(
        fixture,
        mesh_device,
        core,
        transfer_size,
        processor,
        progress_counter,
        recv_buffer0,
        recv_buffer1,
        dram_start_addr,
        dram_end_addr,
        recv_bank_id,
        send_bank_id,
        is_sender ? NULL : programs[mesh_device].get(),
        init_dram);

    /* Senders */
    prepare_sender_integrity_dram(
        fixture,
        mesh_device,
        core,
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
        !is_sender ? NULL : programs[mesh_device].get(),
        init_dram);

    vector<struct core_setup> cores = {
        {
            .program = programs[mesh_device],
            .mesh_device = mesh_device,
            .core = core,
            .iter_l1_addr = 0,  // TODO progress_counter,
            .expected_count = dram_end_addr,
        },
    };
    if (getenv("ETH_TEST_PRECOMPILE")) {
        precompile(fixture, cores, programs);
        return true;
    }

    wait_to_finish_eth_timeout_cores(fixture, cores, programs);

    // double threshold = get_eth_bw() * 0.5;

    bool pass = true;
    if (!is_sender) {
        pass &=
            tensix_compare_dram_banks(fixture, mesh_device, dram_start_addr, dram_end_addr, send_bank_id, recv_bank_id);
    }
    // auto* const send_device = send_mesh_device->impl().get_devices()[0];
    // pass &= bandwidth_check(send_device, send_core, send_delta_addr, total_transferred, threshold);

    return pass;
}

TEST_F(MeshDispatchFixture, TensixDeploymentEthernet03DataIntegrityDramDist) {
    if (not this->slow_dispatch_) {
        GTEST_SKIP() << "fast dispatch to active ethernet has been removed";
    }
    // TODO MetalContext::instance().hal().get_num_risc_processors(HalProgrammableCoreType::ACTIVE_ETH);
    const auto num_eriscs = 1;

    vector<LinkError> errors;
    int n = 0;

    print_detected_devices();
    ASSERT_TRUE(ensure_links(devices_));

    delay_until_env();

    for (auto& core : get_dist_cores_to_test(devices_)) {
        for (uint32_t erisc_idx = 0; erisc_idx < num_eriscs; erisc_idx++) {
            const auto processor = static_cast<DataMovementProcessor>(erisc_idx);

            log_info(tt::LogTest, "    testing dist {} {}", get_eth_unique_id(core.chip, core.core), processor);

            bool passed = run_test_integrity_dram_dist(
                this, core.mesh_device, core.core, core.direction, processor, /* TODO */ true);

            if (!passed) {
                errors.emplace_back(core.chip->id(), core.chip->id(), core.core, core.core, processor);
            }

            log_info(tt::LogTest, "    done");

            n++;
        }
    }

    log_info(tt::LogTest, "Ran {} tests", n);

    print_summary(errors);
    ASSERT_TRUE(errors.empty());
}

}  // namespace tt::tt_metal
