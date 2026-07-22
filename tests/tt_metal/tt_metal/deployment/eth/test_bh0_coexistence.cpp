// SPDX-License-Identifier: Apache-2.0
//
// BH.0 coexistence smoke (gtest form) — TT-RDMA Blackhole bring-up gate #4.
// Guaranteed-compile in-tree variant of tt_metal/tt_rdma/bh0 (see that dir's README
// and docs/tt-rdma-v1/tt-rdma-bh-bf3-impl-plan.md §12 bootstrap).
//
// Proves the coexistence model the whole port rests on: a resident kernel on RISC1
// (subordinate) of an active eth core, while RISC0 (active_erisc) keeps yielding to the
// bh-erisc base FW so the link stays UP. Uses the BOUNDED kernel (num_beats>0) so the
// deployment wait helper can complete cleanly; asserts links are UP before AND after.
//
// This is a CI-friendly short smoke (~seconds). The full 10-min observe-and-hold gate is the
// standalone tool tt_metal/tt_rdma/bh0/bh0_heartbeat_host.cpp (num_beats=0, persistent).

#include "tt_metal/tt_metal/deployment/eth/common.hpp"  // ensure_links, new_erisc_allocator, l1_alloc, core_setup, wait_to_finish_eth_timeout_cores
#include "tt_metal/tt_metal/deployment/deployment_common.hpp"

#include <gtest/gtest.h>
#include <tt_stl/assert.hpp>

#include "command_queue_fixture.hpp"
#include "tt_metal/tt_metal/eth/eth_test_common.hpp"

namespace tt::tt_metal {

using namespace std;

// TODO(BH.0): once topology-role selection lands, pin to an EXTERNAL (gateway-facing) rail
// instead of the first active core, so this exercises exactly the RDMA rails (§11).
TEST_F(MeshDispatchFixture, TtRdmaBH0CoexistenceHeartbeat) {
    constexpr uint32_t kSpinPerBeat = 20000u;  // pace
    constexpr uint32_t kNumBeats = 5000u;      // bounded -> a few seconds of resident RISC1 load

    auto mesh_device = devices_.at(0);
    auto* device = mesh_device->get_devices()[0];

    const auto active = device->get_active_ethernet_cores(/*skip_reserved=*/true);
    if (active.empty()) {
        GTEST_SKIP() << "no active ethernet cores on device 0";
    }
    const CoreCoord eth_core = *active.begin();

    std::vector<std::shared_ptr<distributed::MeshDevice>> devs{mesh_device};

    // Link must be UP before we start.
    ASSERT_TRUE(ensure_links(devs)) << "link not up before the resident kernel";

    // A progress/heartbeat counter in erisc L1 (deployment allocator convention).
    struct l1_allocator alloc = new_erisc_allocator();
    const uint32_t counter_addr = l1_alloc(&alloc, sizeof(uint32_t));

    std::map<std::shared_ptr<distributed::MeshDevice>, std::shared_ptr<Program>> programs = {
        {mesh_device, std::make_shared<Program>()},
    };

    // RISC1 (subordinate) — base FW owns NOC0 on RISC0, so RISC1 uses NOC1.
    const KernelHandle k = CreateKernel(
        *programs[mesh_device],
        "tt_metal/tt_rdma/bh0/kernels/bh_rdma_heartbeat.cpp",
        eth_core,
        EthernetConfig{.eth_mode = Eth::IDLE, .noc = NOC::NOC_1, .processor = DataMovementProcessor::RISCV_1});
    SetRuntimeArgs(*programs[mesh_device], k, eth_core, {counter_addr, kSpinPerBeat, kNumBeats});

    // Run to completion (kernel writes beat -> kNumBeats). Reuses the deployment wait helper.
    std::vector<struct core_setup> cores = {
        {
            .program = programs[mesh_device],
            .mesh_device = mesh_device,
            .core = eth_core,
            .iter_l1_addr = counter_addr,
            .expected_count = kNumBeats,
        },
    };
    wait_to_finish_eth_timeout_cores(this, cores, programs);

    // The gate: the link SURVIVED the resident RISC1 kernel.
    ASSERT_TRUE(ensure_links(devs)) << "link dropped while the RISC1 kernel was resident "
                                       "(coexistence model failed: RISC0 not yielding to base FW)";
}

}  // namespace tt::tt_metal
