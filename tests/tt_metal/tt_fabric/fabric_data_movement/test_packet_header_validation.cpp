// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Unit test for TX-side packet-header validation (tt::tt_fabric::is_valid / is_valid_payload_size).
//
// The predicates are device code, so the test launches a small worker kernel that runs each
// validation case (one valid header plus one corrupted field per check) and writes the boolean
// verdicts to L1. The host reads them back and compares against the expected verdicts. This mirrors
// the standard fabric data-movement tests (kernel launch + L1 readback) but does not need a fabric
// topology, since it exercises the header predicates directly.

#include <gtest/gtest.h>
#include <cstdint>
#include <vector>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/experimental/fabric/fabric_edm_types.hpp>

#include "fabric_fixture.hpp"

namespace tt::tt_fabric::fabric_router_tests {

TEST_F(Fabric1DFixture, PacketHeaderValidation) {
    const auto& device = get_devices()[0];
    const tt::tt_metal::CoreCoord logical_core = {0, 0};
    auto worker_mem_map = generate_worker_mem_map(device, Topology::Linear);

    auto program = tt_metal::CreateProgram();
    auto kernel = tt_metal::CreateKernel(
        program,
        "tests/tt_metal/tt_fabric/fabric_data_movement/kernels/fabric_packet_validation_test_kernel.cpp",
        {logical_core},
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_0, .noc = tt_metal::NOC::RISCV_0_default});

    std::vector<uint32_t> runtime_args = {worker_mem_map.test_results_address};
    tt_metal::SetRuntimeArgs(program, kernel, logical_core, runtime_args);

    RunProgramNonblocking(device, std::move(program));
    WaitForSingleProgramDone(device);

    // Expected verdict per case (must match the result-word order in the kernel):
    //  0 LL valid, 1 LL bad noc_send_type, 2 LL nonzero payload on header-only op,
    //  3 1D-dyn valid, 4 1D-dyn bad chip_send_type,
    //  5 2D valid, 6 2D hop_index OOB, 7 2D bad noc_send_type,
    //  8 payload_size consistent, 9 payload_size inconsistent
    const std::vector<uint32_t> expected = {1, 0, 0, 1, 0, 1, 0, 0, 1, 0};

    std::vector<uint32_t> results;
    tt_metal::slow_dispatch::ReadFromL1(
        *device,
        logical_core,
        worker_mem_map.test_results_address,
        expected.size() * sizeof(uint32_t),
        results,
        CoreType::WORKER);

    ASSERT_GE(results.size(), expected.size());
    for (size_t i = 0; i < expected.size(); i++) {
        EXPECT_EQ(results[i], expected[i]) << "packet-header validation case " << i << " gave unexpected verdict";
    }
}

}  // namespace tt::tt_fabric::fabric_router_tests
