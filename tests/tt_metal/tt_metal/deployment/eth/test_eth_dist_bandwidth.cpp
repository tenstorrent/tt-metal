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
static bool run_test_bandwidth(
    FIXTURE* fixture,
    const std::shared_ptr<distributed::MeshDevice>& mesh_device,
    const CoreCoord& core,
    int is_sender,
    DataMovementProcessor processor = DataMovementProcessor::RISCV_0) {
    /* ======================= */
    auto* const device = mesh_device->impl().get_devices()[0];

    TEST_PARAM(uint32_t, transfer_size, 160 * 1024, "ETH_TEST_TRANSFER_SIZE");
    TEST_PARAM(uint32_t, transfer_count, 10 << 10, "ETH_TEST_TRANSFER_COUNT");

    uint32_t num_bytes_per_send = transfer_size / 2;
    uint64_t total_transferred = (uint64_t)transfer_size * transfer_count;

    // TODO generate same inputs on both hosts
    auto inputs = generate_uniform_random_vector<uint32_t>(0, 100, transfer_size / sizeof(uint32_t));

    struct l1_allocator send_allocator = new_erisc_allocator();
    struct l1_allocator recv_allocator = new_erisc_allocator();

    uint32_t progress_counter = l1_alloc(&recv_allocator, sizeof(uint32_t));
    uint32_t send_progress_counter = l1_alloc(&send_allocator, sizeof(uint32_t));
    TT_FATAL(progress_counter == send_progress_counter, "Progress counters should be at the same address");

    uint32_t recv_l1_address = 0;

    map<shared_ptr<distributed::MeshDevice>, shared_ptr<tt_metal::Program>> programs = {
        {mesh_device, make_shared<Program>()},
    };

    prepare_receiver(
        device,
        core,
        &recv_allocator,
        transfer_size,
        transfer_count,
        inputs,
        processor,
        progress_counter,
        &recv_l1_address,
        is_sender ? NULL : programs[mesh_device].get());

    uint32_t send_delta_addr = 0;
    prepare_sender(
        device,
        core,
        &send_allocator,
        transfer_size,
        transfer_count,
        &send_delta_addr,
        inputs,
        processor,
        num_bytes_per_send,
        send_progress_counter,
        recv_l1_address,
        !is_sender ? NULL : programs[mesh_device].get());

    vector<struct core_setup> cores = {
        {
            .program = programs[mesh_device],
            .mesh_device = mesh_device,
            .core = core,
            .iter_l1_addr = 0,  // progress_counter, // TODO
            .expected_count = transfer_count,
        },
    };
    if (getenv("ETH_TEST_PRECOMPILE")) {
        precompile(fixture, cores, programs);
        return true;
    }

    wait_to_finish_eth_timeout_cores(fixture, cores, programs);

    double threshold = get_eth_bw() * 0.75;

    bool pass = true;
    if (!is_sender) {
        // TODO generate same inputs on both hosts
        // pass &= data_check(device, core, recv_l1_address, inputs);
    } else {
        pass &= bandwidth_check(device, core, send_delta_addr, total_transferred, threshold);
    }
    return pass;
}

TEST_F(MeshDispatchFixture, TensixDeploymentEthernet01BandwidthDist) {
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

            bool passed = run_test_bandwidth(this, core.mesh_device, core.core, core.direction, processor);

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
