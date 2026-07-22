// SPDX-License-Identifier: Apache-2.0
//
// BH.0 coexistence smoke (gtest) — TT-RDMA Blackhole bring-up gate #4.
// Guaranteed-compile in-tree variant of tt_metal/tt_rdma/bh0 (see that dir's README +
// docs/tt-rdma-v1/tt-rdma-bh-bf3-impl-plan.md).
//
// Proves the coexistence model: a resident (bounded) kernel on RISC1 (subordinate) of an active
// eth core while RISC0 (active_erisc) keeps yielding to the bh-erisc base FW -> the link stays UP.
// Asserts ensure_links() before AND after, and that the kernel ran to completion.
//
// v2 fix (after the first run timed out at +34ms): uses the DEFAULT eth config via
// eth_test_common::set_arch_specific_eth_config (NOT Eth::IDLE — an idle core isn't dispatched, so
// the kernel never ran and the counter stayed 0) and distributed::Finish for deterministic
// completion (NOT wait_to_finish_eth_timeout_cores, whose no-progress timeout is tuned to the
// eth-transfer tests' counter-advance rate). Re-validate on a freshly-reset device.

#include "tt_metal/tt_metal/deployment/eth/common.hpp"  // ensure_links, new_erisc_allocator, l1_alloc, read_eth_l1_u32
#include "tt_metal/tt_metal/deployment/deployment_common.hpp"

#include <gtest/gtest.h>
#include <tt_stl/assert.hpp>

#include "command_queue_fixture.hpp"
#include "tt_metal/tt_metal/eth/eth_test_common.hpp"

namespace tt::tt_metal {

using namespace std;

// TODO(BH.0): pin to an EXTERNAL (gateway-facing) rail by topology role once that lands (§11).
TEST_F(MeshDispatchFixture, TtRdmaBH0CoexistenceHeartbeat) {
    constexpr uint32_t kSpinPerBeat = 2000u;  // pace (small so the bounded run is ~ms)
    constexpr uint32_t kNumBeats = 2000u;     // bounded -> kernel returns; final counter == kNumBeats

    auto mesh_device = devices_.at(0);
    auto* device = mesh_device->get_devices()[0];

    const auto active = device->get_active_ethernet_cores(/*skip_reserved=*/true);
    if (active.empty()) {
        GTEST_SKIP() << "no active ethernet cores on device 0";
    }
    const CoreCoord eth_core = *active.begin();

    std::vector<std::shared_ptr<distributed::MeshDevice>> devs{mesh_device};
    ASSERT_TRUE(ensure_links(devs)) << "link not up before the resident kernel";

    // Heartbeat counter in erisc L1 (deployment allocator convention).
    l1_allocator alloc = new_erisc_allocator();
    const uint32_t counter_addr = l1_alloc(&alloc, sizeof(uint32_t));

    Program program = CreateProgram();
    // RISC1 (subordinate). set_arch_specific_eth_config sets the NOC correctly (NOC1 when base FW
    // owns NOC0); eth_mode stays the DEFAULT (a dispatchable active-eth kernel), NOT Eth::IDLE.
    EthernetConfig cfg{.processor = DataMovementProcessor::RISCV_1};
    eth_test_common::set_arch_specific_eth_config(cfg);
    const KernelHandle k = CreateKernel(program, "tt_metal/tt_rdma/bh0/kernels/bh_rdma_heartbeat.cpp", eth_core, cfg);
    // arg3 stop_addr=0: bounded run only (num_beats>0), no host stop flag. Kernel returns after
    // kNumBeats and is reaped by Finish() below — the same clean lifecycle the soak tool now uses.
    SetRuntimeArgs(program, k, eth_core, {counter_addr, kSpinPerBeat, kNumBeats, /*stop_addr=*/0u});

    // Run the bounded kernel to completion (deterministic; no counter-rate timeout).
    distributed::MeshWorkload workload;
    distributed::MeshCoordinateRange device_range(mesh_device->shape());
    workload.add_program(device_range, std::move(program));
    distributed::MeshCommandQueue& cq = mesh_device->mesh_command_queue();
    distributed::EnqueueMeshWorkload(cq, workload, /*blocking=*/false);
    distributed::Finish(cq);

    // Kernel actually ran on RISC1 ...
    const uint32_t final_beat = read_eth_l1_u32(device, eth_core, counter_addr);
    EXPECT_EQ(final_beat, kNumBeats) << "heartbeat kernel did not run to completion on RISC1";
    // ... and the link SURVIVED it.
    ASSERT_TRUE(ensure_links(devs)) << "link dropped while the RISC1 kernel was resident "
                                       "(coexistence failed: RISC0 not yielding to base FW)";
}

}  // namespace tt::tt_metal
