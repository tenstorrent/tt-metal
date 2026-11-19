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
static bool run_test_integrity_dram_bidir_dist(
    FIXTURE* fixture,
    const std::shared_ptr<distributed::MeshDevice>& mesh_device,
    const CoreCoord& core,
    int direction,
    DataMovementProcessor processor0 = DataMovementProcessor::RISCV_0) {
    /* ============= */
    auto* const device = mesh_device->impl().get_devices()[0];

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

    struct l1_allocator alloc = new_erisc_allocator();

    uint32_t iter_l1_address = l1_alloc(&alloc, sizeof(uint64_t));
    uint32_t send_delta_addr = l1_alloc(&alloc, sizeof(uint64_t));
    uint32_t sendbuf0 = l1_alloc(&alloc, transfer_size);
    uint32_t sendbuf1 = l1_alloc(&alloc, transfer_size);
    uint32_t recvbuf0 = l1_alloc(&alloc, transfer_size);
    uint32_t recvbuf1 = l1_alloc(&alloc, transfer_size);

    map<shared_ptr<distributed::MeshDevice>, shared_ptr<tt_metal::Program>> programs = {
        {mesh_device, make_shared<Program>()},
    };

    prepare_bidir_integrity(
        fixture,
        mesh_device,
        core,
        transfer_size,
        send_delta_addr,
        processor0,
        dram_start_addr,
        dram_end_addr,
        num_bytes_per_send,
        iter_l1_address,
        direction,
        !direction,
        send_bank_id0,
        recv_bank_id0,
        sendbuf0,
        sendbuf1,
        recvbuf0,
        recvbuf1,
        programs[mesh_device].get());

    auto zero_coord = distributed::MeshCoordinate(0, 0);
    auto device_range = distributed::MeshCoordinateRange(zero_coord, zero_coord);

    vector<struct core_setup> cores = {
        {
            .program = programs[mesh_device],
            .mesh_device = mesh_device,
            .core = core,
            .iter_l1_addr = 0,  // TODO iter_l1_address,
            .expected_count = dram_end_addr,
        },
    };
    if (getenv("ETH_TEST_PRECOMPILE")) {
        precompile(fixture, cores, programs);
        return true;
    }

    wait_to_finish_eth_timeout_cores(fixture, cores, programs);

    double threshold = 140; /* NOTE: Same on both bh glx and p150 */

    bool pass = true;
    pass &= bandwidth_check(device, core, send_delta_addr, total_transferred, threshold);

    pass &=
        tensix_compare_dram_banks(fixture, mesh_device, dram_start_addr, dram_end_addr, send_bank_id0, recv_bank_id0);

    return pass;
}

TEST_F(MeshDispatchFixture, TensixDeploymentEthernet04DataIntegrityDramBidirDist) {
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

            bool passed =
                run_test_integrity_dram_bidir_dist(this, core.mesh_device, core.core, core.direction, processor);

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
