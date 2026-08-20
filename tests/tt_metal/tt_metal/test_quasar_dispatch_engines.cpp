// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "common/device_fixture.hpp"
#include "dispatch/dispatch_engine_cores.hpp"
#include "host_api/temp_quasar_api.hpp"
#include "test_kernels/misc/quasar_fds_signal_status.h"

#include <optional>
#include <string>
#include <vector>

#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>
#include "impl/context/metal_context.hpp"
#include "llrt/rtoptions.hpp"

#ifndef OVERRIDE_KERNEL_PREFIX
#define OVERRIDE_KERNEL_PREFIX ""
#endif

using namespace tt;
using namespace tt::tt_metal;

namespace {

constexpr const char* kDispatchKernel =
    OVERRIDE_KERNEL_PREFIX "tests/tt_metal/tt_metal/test_kernels/misc/quasar_dispatch_engine_signal.cpp";
constexpr const char* kWorkerKernel =
    OVERRIDE_KERNEL_PREFIX "tests/tt_metal/tt_metal/test_kernels/misc/quasar_fds_worker_signal.cpp";

// Group ids stand for sub-device indices, as the dispatch design intends: group_id =
// sub_device_index + 1, with group 0 reserved because it is the idle value on the wire.
constexpr uint32_t kGroupId = 1;
constexpr uint32_t kSecondGroupId = 2;
constexpr uint32_t kNoQuietGroup = 0;
// The mapping from FDS wire index to core is not established, so a targeted mask cannot distinguish
// "wrong wire" from "no transport at all". The go is aimed at all 32 NEO wires and each worker
// accepts from all 3 dispatch instances, which covers the whole mapping space in one run.
constexpr uint32_t kWorkerMask = 0xFFFFFFFF;
constexpr uint32_t kDispatchMask = 0x7;
// One TENSIX_TO_DISPATCH inbox register per NEO wire, which bounds how many workers can ever be
// counted in one group whatever the wire-to-core mapping turns out to be.
constexpr uint32_t kNumNeoWires = 32;
// Each side gives up rather than spinning forever, so a missing signal fails the test with a
// readable status word instead of hanging it. Kept modest because this runs under a cycle
// simulator, where a million iterations costs minutes of wall clock. Both signals are held rather
// than pulsed, so a shorter wait cannot miss one.
constexpr uint32_t kPollIterations = 100000;

// Why this test cannot run here, or nullopt if it can.
std::optional<std::string> fds_skip_reason(IDevice* dev) {
    const auto& rtoptions = MetalContext::instance().rtoptions();
    // Emulation compiles kernels for the host, where the FDS ROCC custom instructions do not
    // exist, so this test needs the real simulator rather than is_simulator_or_emulated().
    if (!rtoptions.get_simulator_enabled()) {
        return "This test can only be run under the simulator. Set TT_METAL_SIMULATOR.";
    }
    if (!getenv("TT_METAL_SLOW_DISPATCH_MODE")) {
        return "Kernels on dispatch-engine cores require slow dispatch. Set TT_METAL_SLOW_DISPATCH_MODE.";
    }
    if (rtoptions.get_use_quasar_tensix_dispatch_cores()) {
        return "Requires native dispatch-engine cores. Unset TT_METAL_TENSIX_DISPATCH_CORES.";
    }
    if (detail::sd_cq_kernel_tests_should_skip(dev)) {
        return "No dispatch-engine cores in the soc descriptor.";
    }
    return std::nullopt;
}

// One group's worth of workers. Each set is launched with its own group id, so a set stands for a
// sub-device under the group_id = sub_device_index + 1 mapping.
struct WorkerSet {
    CoreRangeSet cores;
    uint32_t group_id = kGroupId;
};

struct WorkerReport {
    CoreCoord core;
    uint32_t group_id = 0;
    std::vector<uint32_t> status;

    bool ran() const { return status[quasar_fds_test::kSlotStarted] == quasar_fds_test::kStarted; }
    bool saw_own_go() const { return status[quasar_fds_test::kSlotResult] == quasar_fds_test::kComplete; }
    uint32_t observed_go() const { return status[quasar_fds_test::kSlotObserved]; }
    uint32_t group_status() const { return status[quasar_fds_test::kSlotGroupStatus]; }
};

struct HandshakeResult {
    CoreCoord dispatch_core;
    std::vector<uint32_t> dispatch_status;
    std::vector<WorkerReport> workers;

    bool dispatch_ran() const { return dispatch_status[quasar_fds_test::kSlotStarted] == quasar_fds_test::kStarted; }
    bool collected_all_dones() const {
        return dispatch_status[quasar_fds_test::kSlotResult] == quasar_fds_test::kComplete;
    }
    uint32_t done_count() const { return dispatch_status[quasar_fds_test::kSlotObserved]; }
    uint32_t quiet_group_count() const { return dispatch_status[quasar_fds_test::kSlotQuietGroupCount]; }
};

std::vector<uint32_t> read_status(IDevice* dev, const CoreCoord& core, uint32_t addr, CoreType core_type) {
    std::vector<uint32_t> status(quasar_fds_test::kNumSlots, 0);
    detail::ReadFromDeviceL1(dev, core, addr, status.size() * sizeof(uint32_t), status, core_type);
    return status;
}

// Runs one epoch of the handshake. The dispatch engine sends a go for signalled_group only and
// waits for a done from every worker in that group; each worker waits for its own group's go and
// answers. quiet_group is configured on the dispatch side but never signalled, so its done count is
// evidence about leakage between groups; pass kNoQuietGroup to leave it out.
//
// One data-movement core per tile. A tile has a single FDS register block shared by all of its
// data-movement cores, so two cores on one tile would overwrite each other's configuration and
// consume each other's status. Separate tiles have separate blocks, which is what makes fanning out
// across the grid meaningful where fanning out within a tile is not.
HandshakeResult run_handshake(
    IDevice* dev, const std::vector<WorkerSet>& worker_sets, uint32_t signalled_group, uint32_t quiet_group) {
    HandshakeResult result;
    result.dispatch_core = detail::dispatch_engine_core(dev, 0);

    const auto& hal = MetalContext::instance().hal();
    const uint32_t dispatch_l1 =
        hal.get_dev_addr(HalProgrammableCoreType::DISPATCH, HalL1MemAddrType::DEFAULT_UNRESERVED);
    const uint32_t worker_l1 = hal.get_dev_addr(HalProgrammableCoreType::TENSIX, HalL1MemAddrType::DEFAULT_UNRESERVED);

    std::vector<uint32_t> cleared(quasar_fds_test::kNumSlots, 0);
    detail::WriteToDeviceL1(dev, result.dispatch_core, dispatch_l1, cleared, CoreType::DISPATCH);
    for (const WorkerSet& set : worker_sets) {
        for (const CoreCoord& core : corerange_to_cores(set.cores)) {
            detail::WriteToDeviceL1(dev, core, worker_l1, cleared, CoreType::WORKER);
            result.workers.push_back(WorkerReport{.core = core, .group_id = set.group_id});
        }
    }

    // Only the signalled group's workers can answer, so only they count towards the wait.
    uint32_t done_threshold = 0;
    for (const WorkerReport& worker : result.workers) {
        done_threshold += (worker.group_id == signalled_group) ? 1 : 0;
    }

    Program program = CreateProgram();

    detail::CreateDispatchEngineKernel(
        program,
        kDispatchKernel,
        result.dispatch_core,
        experimental::quasar::QuasarDataMovementConfig{
            .num_threads_per_cluster = 1,
            .named_compile_args = {
                {"l1_address", dispatch_l1},
                {"group_id", signalled_group},
                {"worker_mask", kWorkerMask},
                {"done_threshold", done_threshold},
                {"quiet_group_id", quiet_group},
                {"poll_iterations", kPollIterations}}});

    for (const WorkerSet& set : worker_sets) {
        experimental::quasar::CreateKernel(
            program,
            kWorkerKernel,
            set.cores,
            experimental::quasar::QuasarDataMovementConfig{
                .num_threads_per_cluster = 1,
                .named_compile_args = {
                    {"l1_address", worker_l1},
                    {"group_id", set.group_id},
                    {"dispatch_mask", kDispatchMask},
                    {"poll_iterations", kPollIterations}}});
    }

    detail::LaunchProgram(dev, program, /*wait_until_cores_done=*/true);

    result.dispatch_status = read_status(dev, result.dispatch_core, dispatch_l1, CoreType::DISPATCH);
    for (WorkerReport& worker : result.workers) {
        worker.status = read_status(dev, worker.core, worker_l1, CoreType::WORKER);
    }
    return result;
}

// Names what each side ended up seeing, so a partial result is readable without a rerun.
void log_handshake(const HandshakeResult& result) {
    for (const WorkerReport& worker : result.workers) {
        log_info(
            tt::LogTest,
            "worker core {} group {}: result={:#x} observed_go={} group_status={}",
            worker.core.str(),
            worker.group_id,
            worker.status[quasar_fds_test::kSlotResult],
            worker.observed_go(),
            worker.group_status());
    }
    log_info(
        tt::LogTest,
        "dispatch core {}: result={:#x} done count={} quiet group count={}",
        result.dispatch_core.str(),
        result.dispatch_status[quasar_fds_test::kSlotResult],
        result.done_count(),
        result.quiet_group_count());
}

}  // namespace

// Drives the Quasar FDS sideband end to end between one dispatch engine and one worker: the
// dispatch-engine kernel writes L1, sends a go signal and waits for the worker's done signal; the
// worker kernel waits for the go and answers with done.
TEST_F(QuasarMeshDeviceSingleCardFixture, DispatchEngineSingleWorker) {
    IDevice* dev = devices_[0]->get_devices()[0];
    if (const auto reason = fds_skip_reason(dev)) {
        GTEST_SKIP() << *reason;
    }

    const HandshakeResult result =
        run_handshake(dev, {WorkerSet{.cores = CoreRangeSet(CoreRange({0, 0}, {0, 0}))}}, kGroupId, kNoQuietGroup);
    log_handshake(result);

    ASSERT_TRUE(result.dispatch_ran()) << "The dispatch-engine kernel did not run.";
    ASSERT_TRUE(result.workers[0].ran()) << "The worker kernel did not run.";

    EXPECT_TRUE(result.workers[0].saw_own_go()) << "The worker never observed the FDS go signal.";
    EXPECT_TRUE(result.collected_all_dones()) << "The dispatch engine never observed the worker's FDS done signal.";
}

// The same handshake fanned out to every worker tile the device offers, which is what a real
// worker-completion path has to do. One worker cannot distinguish a correct implementation from a
// wrong one: a go aimed at the wrong lane, a mask covering more lanes than exist, or a done count
// that stops at the first arrival all still pass with a single worker. Here the dispatch engine
// must accumulate one done per tile before its wait is satisfied.
TEST_F(QuasarMeshDeviceSingleCardFixture, DispatchEngineAllWorkers) {
    IDevice* dev = devices_[0]->get_devices()[0];
    if (const auto reason = fds_skip_reason(dev)) {
        GTEST_SKIP() << *reason;
    }

    const CoreCoord worker_grid = dev->compute_with_storage_grid_size();
    const uint32_t num_workers = worker_grid.x * worker_grid.y;
    // The done count aggregates one lane per worker, so a grid larger than the lane count cannot be
    // covered by a single group however the lanes turn out to be mapped to cores.
    ASSERT_LE(num_workers, kNumNeoWires) << "More worker tiles (" << num_workers << ") than FDS done lanes ("
                                         << kNumNeoWires << "); this test needs more than one group.";
    if (num_workers < 2) {
        GTEST_SKIP() << "Only one worker tile in the soc descriptor, so this adds nothing over "
                        "DispatchEngineSingleWorker.";
    }

    const CoreRange worker_cores({0, 0}, {worker_grid.x - 1, worker_grid.y - 1});
    const HandshakeResult result =
        run_handshake(dev, {WorkerSet{.cores = CoreRangeSet(worker_cores)}}, kGroupId, kNoQuietGroup);
    log_handshake(result);

    ASSERT_TRUE(result.dispatch_ran()) << "The dispatch-engine kernel did not run.";

    // Reported per worker rather than as a count, so a partly working mapping names the tiles that
    // worked and the tiles that did not.
    for (const WorkerReport& worker : result.workers) {
        ASSERT_TRUE(worker.ran()) << "The worker kernel did not run on core " << worker.core.str() << ".";
        EXPECT_TRUE(worker.saw_own_go()) << "The worker on core " << worker.core.str()
                                         << " never observed the FDS go signal.";
    }

    EXPECT_TRUE(result.collected_all_dones()) << "The dispatch engine collected " << result.done_count() << " of "
                                              << num_workers << " expected worker done signals.";
}

// Confirms that a go addressed to one group reaches only that group's workers, which is the
// hardware property the sub-device design rests on: it plans to give each sub-device its own group
// id, so a launch on one sub-device must not release the workers of another.
//
// This tests the FDS group mechanism, not the sub-device API. Nothing connects a `SubDevice` to a
// group id yet — that is the dispatch work this plan gates — so the two disjoint worker sets here
// stand in for two sub-devices under the intended group_id = sub_device_index + 1 mapping.
//
// Isolation is asserted in both directions available in one epoch. On the worker side, the quiet
// group's tiles must not latch a go for their own group; if they see the signalled group's value in
// a raw inbox that is reported, because the go wire may be shared and only the group status says
// whether it was accepted. On the dispatch side, the quiet group's done count must stay at zero
// while the signalled group's reaches its full total.
TEST_F(QuasarMeshDeviceSingleCardFixture, DispatchEngineSubDeviceGroupIsolation) {
    IDevice* dev = devices_[0]->get_devices()[0];
    if (const auto reason = fds_skip_reason(dev)) {
        GTEST_SKIP() << *reason;
    }

    const CoreCoord worker_grid = dev->compute_with_storage_grid_size();
    const uint32_t num_workers = worker_grid.x * worker_grid.y;
    if (num_workers < 2) {
        GTEST_SKIP() << "Two disjoint worker tiles are needed to stand for two sub-devices; the soc descriptor "
                        "offers one.";
    }

    // Halve the flat list of worker tiles into two sub-device stand-ins, the first signalled and the
    // second not. Splitting the list rather than a grid axis keeps this correct for a grid of any
    // shape, including the single-column ones a descriptor may offer.
    const std::vector<CoreCoord> all_workers =
        corerange_to_cores(CoreRangeSet(CoreRange({0, 0}, {worker_grid.x - 1, worker_grid.y - 1})));
    std::vector<CoreRange> signalled_cores;
    std::vector<CoreRange> quiet_cores;
    for (size_t i = 0; i < all_workers.size(); i++) {
        ((i < all_workers.size() / 2) ? signalled_cores : quiet_cores).push_back(CoreRange(all_workers[i]));
    }
    const std::vector<WorkerSet> worker_sets = {
        WorkerSet{.cores = CoreRangeSet(signalled_cores), .group_id = kGroupId},
        WorkerSet{.cores = CoreRangeSet(quiet_cores), .group_id = kSecondGroupId}};

    const HandshakeResult result = run_handshake(dev, worker_sets, kGroupId, kSecondGroupId);
    log_handshake(result);

    ASSERT_TRUE(result.dispatch_ran()) << "The dispatch-engine kernel did not run.";

    uint32_t signalled_workers = 0;
    for (const WorkerReport& worker : result.workers) {
        const std::string core = worker.core.str();
        ASSERT_TRUE(worker.ran()) << "The worker kernel did not run on core " << core << ".";
        if (worker.group_id == kGroupId) {
            signalled_workers++;
            EXPECT_TRUE(worker.saw_own_go()) << "The worker on core " << core << " in the signalled group " << kGroupId
                                             << " never observed its go signal.";
            continue;
        }
        EXPECT_FALSE(worker.saw_own_go())
            << "The worker on core " << core << " belongs to group " << worker.group_id
            << ", which was never signalled, but it accepted a go for that group. Groups are not isolated.";
        EXPECT_EQ(worker.group_status(), 0u)
            << "The worker on core " << core << " latched group status " << worker.group_status() << " for group "
            << worker.group_id << ", which was never signalled.";
        if (worker.observed_go() != 0) {
            log_info(
                tt::LogTest,
                "worker core {} in quiet group {} saw group {} on a raw inbox without latching it, so the go wire "
                "is shared across groups and the group filter is what separates them",
                core,
                worker.group_id,
                worker.observed_go());
        }
    }
    ASSERT_GT(signalled_workers, 0u) << "The split left the signalled group with no workers.";

    EXPECT_EQ(result.quiet_group_count(), 0u)
        << "Group " << kSecondGroupId << " was configured but never signalled, yet its done count reached "
        << result.quiet_group_count() << ". Done signals are crediting the wrong group.";
    EXPECT_TRUE(result.collected_all_dones())
        << "The dispatch engine collected " << result.done_count() << " of " << signalled_workers
        << " expected done signals from group " << kGroupId << ".";
}
