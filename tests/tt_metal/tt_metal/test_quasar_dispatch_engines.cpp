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
constexpr const char* kReArmDispatchKernel =
    OVERRIDE_KERNEL_PREFIX "tests/tt_metal/tt_metal/test_kernels/misc/quasar_dispatch_engine_rearm.cpp";
constexpr const char* kReArmWorkerKernel =
    OVERRIDE_KERNEL_PREFIX "tests/tt_metal/tt_metal/test_kernels/misc/quasar_fds_worker_rearm.cpp";
constexpr const char* kLaneMapDispatchKernel =
    OVERRIDE_KERNEL_PREFIX "tests/tt_metal/tt_metal/test_kernels/misc/quasar_dispatch_engine_lane_map.cpp";
constexpr const char* kOrderedReadDispatchKernel =
    OVERRIDE_KERNEL_PREFIX "tests/tt_metal/tt_metal/test_kernels/misc/quasar_dispatch_engine_ordered_read.cpp";
constexpr const char* kOrderedWriteWorkerKernel =
    OVERRIDE_KERNEL_PREFIX "tests/tt_metal/tt_metal/test_kernels/misc/quasar_fds_worker_ordered_write.cpp";
constexpr const char* kDriveDoneWorkerKernel =
    OVERRIDE_KERNEL_PREFIX "tests/tt_metal/tt_metal/test_kernels/misc/quasar_fds_worker_drive_done.cpp";

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
// Long enough to outlast the deglitch interval, so that a receive inbox reading zero straight after
// a clear can be told apart from one that reads zero and stays there.
constexpr uint32_t kSettleIterations = 2000;
// How long a tile keeps running after driving its done in the lane-mapping experiment. The done is a
// held level and survives the kernel, but staying alive while the dispatch engine scans keeps the
// measurement independent of whether firmware teardown disturbs the register.
constexpr uint32_t kHoldIterations = 20000;
// Group ids are a four-bit field with 0 reserved as the idle value, so 15 tiles can carry distinct
// ids in one lane-mapping run.
constexpr uint32_t kMaxUsableGroups = 15;

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

std::vector<uint32_t> read_status(
    IDevice* dev, const CoreCoord& core, uint32_t addr, CoreType core_type, uint32_t num_slots) {
    std::vector<uint32_t> status(num_slots, 0);
    detail::ReadFromDeviceL1(dev, core, addr, status.size() * sizeof(uint32_t), status, core_type);
    return status;
}

std::vector<uint32_t> read_status(IDevice* dev, const CoreCoord& core, uint32_t addr, CoreType core_type) {
    return read_status(dev, core, addr, core_type, quasar_fds_test::kNumSlots);
}

// L1 addresses the kernels write their status blocks to.
uint32_t dispatch_status_address() {
    return MetalContext::instance().hal().get_dev_addr(
        HalProgrammableCoreType::DISPATCH, HalL1MemAddrType::DEFAULT_UNRESERVED);
}

uint32_t worker_status_address() {
    return MetalContext::instance().hal().get_dev_addr(
        HalProgrammableCoreType::TENSIX, HalL1MemAddrType::DEFAULT_UNRESERVED);
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

// Two epochs of the same group, which is the question that decides whether the straightforward
// completion protocol is viable at all.
//
// Both directions are held levels rather than pulses, so a second done for a group is not
// distinguishable from the first one still being driven unless something de-asserts in between. The
// kernels run the full cycle and record each step: a done, a receive-inbox clear taken while the
// worker is still driving that same done, a go de-assert and re-assert, and a second done. The two
// counts either side of the clear are the measurement that matters — a clear that does not hold
// against a live source means the design needs an explicit epoch in every go instead.
TEST_F(QuasarMeshDeviceSingleCardFixture, DispatchEngineSameGroupReArm) {
    IDevice* dev = devices_[0]->get_devices()[0];
    if (const auto reason = fds_skip_reason(dev)) {
        GTEST_SKIP() << *reason;
    }

    const CoreCoord dispatch_core = detail::dispatch_engine_core(dev, 0);
    const CoreCoord worker_core{0, 0};
    const uint32_t dispatch_l1 = dispatch_status_address();
    const uint32_t worker_l1 = worker_status_address();

    std::vector<uint32_t> cleared(quasar_fds_test::rearm::kNumSlots, 0);
    detail::WriteToDeviceL1(dev, dispatch_core, dispatch_l1, cleared, CoreType::DISPATCH);
    detail::WriteToDeviceL1(dev, worker_core, worker_l1, cleared, CoreType::WORKER);

    Program program = CreateProgram();
    detail::CreateDispatchEngineKernel(
        program,
        kReArmDispatchKernel,
        dispatch_core,
        experimental::quasar::QuasarDataMovementConfig{
            .num_threads_per_cluster = 1,
            .named_compile_args = {
                {"l1_address", dispatch_l1},
                {"group_id", kGroupId},
                {"worker_mask", kWorkerMask},
                {"poll_iterations", kPollIterations},
                {"settle_iterations", kSettleIterations}}});
    experimental::quasar::CreateKernel(
        program,
        kReArmWorkerKernel,
        worker_core,
        experimental::quasar::QuasarDataMovementConfig{
            .num_threads_per_cluster = 1,
            .named_compile_args = {
                {"l1_address", worker_l1},
                {"group_id", kGroupId},
                {"dispatch_mask", kDispatchMask},
                {"poll_iterations", kPollIterations}}});

    detail::LaunchProgram(dev, program, /*wait_until_cores_done=*/true);

    const auto dispatch =
        read_status(dev, dispatch_core, dispatch_l1, CoreType::DISPATCH, quasar_fds_test::rearm::kNumSlots);
    const auto worker = read_status(dev, worker_core, worker_l1, CoreType::WORKER, quasar_fds_test::rearm::kNumSlots);

    const uint32_t round1_count = dispatch[quasar_fds_test::rearm::kSlotRound1Count];
    const uint32_t count_after_clear = dispatch[quasar_fds_test::rearm::kSlotCountAfterClear];
    const uint32_t count_after_settle = dispatch[quasar_fds_test::rearm::kSlotCountAfterSettle];
    const uint32_t round2_count = dispatch[quasar_fds_test::rearm::kSlotRound2Count];

    log_info(
        tt::LogTest,
        "re-arm dispatch: round1 count={} count after inbox clear={} after settle={} round2 count={}",
        round1_count,
        count_after_clear,
        count_after_settle,
        round2_count);
    log_info(
        tt::LogTest,
        "re-arm worker: round1 go={} go de-assert seen={} group status after de-assert={} round2 go={}",
        worker[quasar_fds_test::rearm::kSlotRound1Go],
        worker[quasar_fds_test::rearm::kSlotDeassertSeen],
        worker[quasar_fds_test::rearm::kSlotStatusAfterDeassert],
        worker[quasar_fds_test::rearm::kSlotRound2Go]);

    ASSERT_EQ(dispatch[quasar_fds_test::rearm::kSlotStarted], quasar_fds_test::kStarted)
        << "The dispatch-engine kernel did not run.";
    ASSERT_EQ(worker[quasar_fds_test::rearm::kSlotStarted], quasar_fds_test::kStarted)
        << "The worker kernel did not run.";

    // Epoch one has to work before anything after it means anything.
    ASSERT_EQ(worker[quasar_fds_test::rearm::kSlotRound1Go], 1u) << "The worker never saw the first go.";
    ASSERT_GE(round1_count, 1u) << "The dispatch engine never counted the first done.";

    // The heart of the experiment: a sink-side clear taken while the source still drives the same
    // value. Both reads must be zero — an immediate zero that does not survive the settle is a
    // clear that did not hold.
    EXPECT_EQ(count_after_clear, 0u) << "Clearing the receive inboxes left the group count at " << count_after_clear
                                     << " while the worker was still driving the same done.";
    EXPECT_EQ(count_after_settle, 0u)
        << "The group count returned to " << count_after_settle << " after the settle period while the worker was "
        << "still driving the same done, so a sink-side clear does not hold against a live source.";

    EXPECT_EQ(worker[quasar_fds_test::rearm::kSlotDeassertSeen], 1u)
        << "The worker never saw the go de-assert, so it has no way to tell a second epoch of this "
           "group from the first.";
    EXPECT_EQ(worker[quasar_fds_test::rearm::kSlotRound2Go], 1u) << "The worker never saw the second go.";
    EXPECT_GE(round2_count, 1u) << "The dispatch engine never counted a second done for the same group.";

    if (count_after_settle != 0) {
        log_info(
            tt::LogTest,
            "the second-epoch result above is uninterpretable: the count never returned to zero, so a credit "
            "in round 2 cannot be distinguished from round 1's credit surviving the clear");
    }
}

// Which physical lane each worker tile drives, which the dispatch design needs and currently
// guesses. Nothing sends a go: every tile drives a done carrying its own group id, and the raw
// receive inbox registers on the dispatch side name the lane each value arrived on.
//
// The per-group done counts come along for free and are the done-direction isolation check. Every
// group is enabled on every lane, so a group whose tile never drove must still read zero. The
// go/done handshake tests cannot check this, because an unsignalled group's workers never drive
// anything.
TEST_F(QuasarMeshDeviceSingleCardFixture, DispatchEngineLaneMap) {
    IDevice* dev = devices_[0]->get_devices()[0];
    if (const auto reason = fds_skip_reason(dev)) {
        GTEST_SKIP() << *reason;
    }

    const CoreCoord worker_grid = dev->compute_with_storage_grid_size();
    const std::vector<CoreCoord> worker_cores =
        corerange_to_cores(CoreRangeSet(CoreRange({0, 0}, {worker_grid.x - 1, worker_grid.y - 1})));
    ASSERT_LE(worker_cores.size(), kMaxUsableGroups)
        << worker_cores.size() << " worker tiles cannot take distinct group ids in one run; only " << kMaxUsableGroups
        << " are usable.";

    const CoreCoord dispatch_core = detail::dispatch_engine_core(dev, 0);
    const uint32_t dispatch_l1 = dispatch_status_address();
    const uint32_t worker_l1 = worker_status_address();

    std::vector<uint32_t> cleared_dispatch(quasar_fds_test::lane_map::kNumSlots, 0);
    std::vector<uint32_t> cleared_worker(quasar_fds_test::kNumSlots, 0);
    detail::WriteToDeviceL1(dev, dispatch_core, dispatch_l1, cleared_dispatch, CoreType::DISPATCH);
    for (const CoreCoord& core : worker_cores) {
        detail::WriteToDeviceL1(dev, core, worker_l1, cleared_worker, CoreType::WORKER);
    }

    Program program = CreateProgram();
    detail::CreateDispatchEngineKernel(
        program,
        kLaneMapDispatchKernel,
        dispatch_core,
        experimental::quasar::QuasarDataMovementConfig{
            .num_threads_per_cluster = 1,
            .named_compile_args = {
                {"l1_address", dispatch_l1},
                {"worker_mask", kWorkerMask},
                {"expected_lanes", static_cast<uint32_t>(worker_cores.size())},
                {"poll_iterations", kPollIterations}}});

    // Tile i drives group i + 1, so the value arriving on a lane names the tile that sent it.
    for (size_t i = 0; i < worker_cores.size(); i++) {
        experimental::quasar::CreateKernel(
            program,
            kDriveDoneWorkerKernel,
            worker_cores[i],
            experimental::quasar::QuasarDataMovementConfig{
                .num_threads_per_cluster = 1,
                .named_compile_args = {
                    {"l1_address", worker_l1},
                    {"group_id", static_cast<uint32_t>(i + 1)},
                    {"hold_iterations", kHoldIterations}}});
    }

    detail::LaunchProgram(dev, program, /*wait_until_cores_done=*/true);

    const auto dispatch =
        read_status(dev, dispatch_core, dispatch_l1, CoreType::DISPATCH, quasar_fds_test::lane_map::kNumSlots);

    ASSERT_EQ(dispatch[quasar_fds_test::lane_map::kSlotStarted], quasar_fds_test::kStarted)
        << "The dispatch-engine kernel did not run.";
    for (size_t i = 0; i < worker_cores.size(); i++) {
        const auto worker = read_status(dev, worker_cores[i], worker_l1, CoreType::WORKER);
        ASSERT_EQ(worker[quasar_fds_test::kSlotStarted], quasar_fds_test::kStarted)
            << "The worker kernel did not run on core " << worker_cores[i].str() << ".";
        ASSERT_EQ(worker[quasar_fds_test::kSlotObserved], static_cast<uint32_t>(i + 1))
            << "The worker on core " << worker_cores[i].str() << " drove the wrong group.";
    }

    const uint32_t idle_lane_map = dispatch[quasar_fds_test::lane_map::kSlotIdleLaneMap];
    log_info(
        tt::LogTest,
        "lane map: {} of {} lanes driving, group 0 status (idle lanes)={:#x}",
        dispatch[quasar_fds_test::lane_map::kSlotLanesDriving],
        worker_cores.size(),
        idle_lane_map);

    // The mapping itself, which is the output of the experiment.
    //
    // How many lanes one tile drives is not asserted, because it is one of the things being
    // measured: the register block is named for the Tensix engine and there are 32 lanes, which
    // would fit either one lane per tile or one per engine across eight tiles. Either answer is
    // reported rather than failed. What is asserted is that every tile appears somewhere, that no
    // lane carries a group nobody drove, and that the aggregated counts agree with the raw lanes.
    std::vector<uint32_t> lanes_per_group(worker_cores.size() + 1, 0);
    for (uint32_t lane = 0; lane < quasar_fds_test::lane_map::kNumLanes; lane++) {
        const uint32_t value = dispatch[quasar_fds_test::lane_map::kSlotLaneBase + lane];
        if (value == 0) {
            continue;
        }
        const bool names_a_tile = (value <= worker_cores.size());
        log_info(
            tt::LogTest,
            "  lane {} carries group {} -> {}",
            lane,
            value,
            names_a_tile ? worker_cores[value - 1].str() : std::string("no tile drove this group"));
        EXPECT_TRUE(names_a_tile) << "Lane " << lane << " carries group " << value
                                  << ", which no tile in this run drove.";
        if (names_a_tile) {
            lanes_per_group[value]++;
        }
        // Group 0's status is the map of lanes carrying nothing, so a driving lane must have dropped
        // out of it. Disagreement between the two readings would undermine the status-mask model the
        // owner publish loop is built on.
        EXPECT_EQ((idle_lane_map >> lane) & 1u, 0u)
            << "Lane " << lane << " is carrying group " << value
            << " but group 0's status still reports it idle; the two readings disagree.";
    }

    for (size_t group = 1; group <= worker_cores.size(); group++) {
        EXPECT_GT(lanes_per_group[group], 0u) << "Group " << group << ", driven by core "
                                              << worker_cores[group - 1].str() << ", did not appear on any lane.";
        if (lanes_per_group[group] > 1) {
            log_info(
                tt::LogTest,
                "core {} drives {} lanes, so a lane is not one per tile — this is the answer to what the 32 "
                "done lanes correspond to",
                worker_cores[group - 1].str(),
                lanes_per_group[group]);
        }
    }

    // The aggregated count must agree with the raw lanes, and a group nobody drove must have counted
    // nothing — the done-direction isolation check.
    for (uint32_t group = 1; group < quasar_fds_test::lane_map::kNumGroups; group++) {
        const uint32_t count = dispatch[quasar_fds_test::lane_map::kSlotGroupCountBase + group];
        if (group <= worker_cores.size()) {
            EXPECT_EQ(count, lanes_per_group[group])
                << "Group " << group << " counted " << count << " dones but appears on " << lanes_per_group[group]
                << " lanes; aggregation disagrees with the raw inboxes.";
        } else {
            EXPECT_EQ(count, 0u) << "Group " << group << " was driven by no tile but counted " << count
                                 << " dones, so a done is crediting a group that did not send it.";
        }
    }

    EXPECT_EQ(dispatch[quasar_fds_test::lane_map::kSlotResult], quasar_fds_test::kComplete)
        << "Only " << dispatch[quasar_fds_test::lane_map::kSlotLanesDriving] << " of " << worker_cores.size()
        << " tiles' dones ever reached the dispatch engine.";
}

namespace {

struct OrderingResult {
    bool worker_ran = false;
    bool worker_signalled = false;
    bool dispatch_saw_done = false;
    uint32_t mismatches = 0;
    uint32_t first_mismatch_index = 0;
    uint32_t first_mismatch_value = 0;
    uint32_t tail_word = 0;
    bool stale = false;
};

// One arm of the write-ordering experiment. The worker writes a payload into the dispatch core's L1
// over the NOC and signals completion; the dispatch engine reads that payload the moment the signal
// appears. With barrier_before_done set, the worker drains the write first, which is what the kernel
// contract requires. Without it, the write is left in flight on purpose. signal_via_fds selects the
// FDS sideband or, for the control arm, the NOC atomic increment the current completion path uses.
OrderingResult run_ordering_arm(IDevice* dev, bool barrier_before_done, bool signal_via_fds) {
    const CoreCoord dispatch_core = detail::dispatch_engine_core(dev, 0);
    const CoreCoord dispatch_virtual = detail::dispatch_engine_virtual_core(dev, 0);
    const CoreCoord worker_core{0, 0};

    const uint32_t dispatch_l1 = dispatch_status_address();
    const uint32_t worker_l1 = worker_status_address();
    const uint32_t dispatch_payload = dispatch_l1 + quasar_fds_test::ordering::kPayloadOffset;
    const uint32_t dispatch_counter = dispatch_l1 + quasar_fds_test::ordering::kCounterOffset;
    const uint32_t worker_payload = worker_l1 + quasar_fds_test::ordering::kPayloadOffset;

    std::vector<uint32_t> cleared(quasar_fds_test::ordering::kNumSlots, 0);
    detail::WriteToDeviceL1(dev, dispatch_core, dispatch_l1, cleared, CoreType::DISPATCH);
    detail::WriteToDeviceL1(dev, worker_core, worker_l1, cleared, CoreType::WORKER);

    // Pre-fill the destination with a value the payload never contains, so a word that still holds it
    // is a word the write had not delivered rather than one that happens to match.
    std::vector<uint32_t> prefill(quasar_fds_test::ordering::kPayloadWords, quasar_fds_test::ordering::kPrefillWord);
    detail::WriteToDeviceL1(dev, dispatch_core, dispatch_payload, prefill, CoreType::DISPATCH);
    std::vector<uint32_t> zero_counter(1, 0);
    detail::WriteToDeviceL1(dev, dispatch_core, dispatch_counter, zero_counter, CoreType::DISPATCH);

    Program program = CreateProgram();
    detail::CreateDispatchEngineKernel(
        program,
        kOrderedReadDispatchKernel,
        dispatch_core,
        experimental::quasar::QuasarDataMovementConfig{
            .num_threads_per_cluster = 1,
            .named_compile_args = {
                {"l1_address", dispatch_l1},
                {"payload_address", dispatch_payload},
                {"counter_address", dispatch_counter},
                {"signal_via_fds", signal_via_fds ? 1u : 0u},
                {"group_id", kGroupId},
                {"worker_mask", kWorkerMask},
                {"poll_iterations", kPollIterations}}});
    experimental::quasar::CreateKernel(
        program,
        kOrderedWriteWorkerKernel,
        worker_core,
        experimental::quasar::QuasarDataMovementConfig{
            .num_threads_per_cluster = 1,
            .named_compile_args = {
                {"l1_address", worker_l1},
                {"payload_src_address", worker_payload},
                {"dest_noc_x", static_cast<uint32_t>(dispatch_virtual.x)},
                {"dest_noc_y", static_cast<uint32_t>(dispatch_virtual.y)},
                {"dest_address", dispatch_payload},
                {"counter_address", dispatch_counter},
                {"signal_via_fds", signal_via_fds ? 1u : 0u},
                {"group_id", kGroupId},
                {"dispatch_mask", kDispatchMask},
                {"poll_iterations", kPollIterations},
                {"barrier_before_done", barrier_before_done ? 1u : 0u},
                {"signal_via_fds", signal_via_fds ? 1u : 0u}}});

    detail::LaunchProgram(dev, program, /*wait_until_cores_done=*/true);

    const auto dispatch =
        read_status(dev, dispatch_core, dispatch_l1, CoreType::DISPATCH, quasar_fds_test::ordering::kNumSlots);
    const auto worker =
        read_status(dev, worker_core, worker_l1, CoreType::WORKER, quasar_fds_test::ordering::kNumSlots);

    OrderingResult result;
    result.worker_ran = worker[quasar_fds_test::ordering::kSlotStarted] == quasar_fds_test::kStarted;
    result.worker_signalled = worker[quasar_fds_test::ordering::kSlotResult] == quasar_fds_test::kComplete;
    result.dispatch_saw_done = dispatch[quasar_fds_test::ordering::kSlotResult] == quasar_fds_test::kComplete;
    result.mismatches = dispatch[quasar_fds_test::ordering::kSlotMismatches];
    result.first_mismatch_index = dispatch[quasar_fds_test::ordering::kSlotFirstMismatchIndex];
    result.first_mismatch_value = dispatch[quasar_fds_test::ordering::kSlotFirstMismatchValue];
    result.tail_word = dispatch[quasar_fds_test::ordering::kSlotTailWord];

    const uint32_t expected_tail =
        quasar_fds_test::ordering::kPayloadSeed + quasar_fds_test::ordering::kPayloadWords - 1;
    result.stale = result.dispatch_saw_done && (result.mismatches > 0 || result.tail_word != expected_tail);
    log_info(
        tt::LogTest,
        "ordering arm barrier={} signal={}: worker signalled={} dispatch saw signal={} tail word={:#x} "
        "(expected {:#x}) mismatched words={} of {} (first at index {}, value {:#x})",
        barrier_before_done,
        signal_via_fds ? "fds" : "noc-atomic",
        result.worker_signalled,
        result.dispatch_saw_done,
        result.tail_word,
        expected_tail,
        result.mismatches,
        quasar_fds_test::ordering::kPayloadWords,
        result.first_mismatch_index,
        result.first_mismatch_value);
    return result;
}

}  // namespace

// Whether data a worker writes over the NOC is visible when its FDS done is observed. This is the
// only test here that exercises the path a completion fence protects; the others have the worker
// write to its own L1 and nobody reads it until the program has finished.
//
// Two arms. With a barrier the worker drains its write before signalling, which is what the kernel
// contract requires, and the payload must be intact — that is a real invariant and a hard assertion.
// Without a barrier the write is left in flight deliberately, and the outcome is reported rather than
// asserted, because either answer is informative and neither is a defect in this test:
//
//   - Mismatches: the hazard is real on this platform and a fence is mandatory. That is the finding
//     Gate 0b is looking for, and it arrives as a log line rather than a red test.
//   - No mismatches: this platform did not expose it. That is not evidence of safety. A functional
//     simulator may not reorder at all, which is why the plan requires silicon or a model certified
//     for NOC and FDS ordering before this gate can be called closed.
TEST_F(QuasarMeshDeviceSingleCardFixture, DispatchEngineWriteOrdering) {
    IDevice* dev = devices_[0]->get_devices()[0];
    if (const auto reason = fds_skip_reason(dev)) {
        GTEST_SKIP() << *reason;
    }

    const OrderingResult barriered = run_ordering_arm(dev, /*barrier_before_done=*/true, /*signal_via_fds=*/true);

    ASSERT_TRUE(barriered.worker_ran) << "The worker kernel did not run.";
    ASSERT_TRUE(barriered.worker_signalled) << "The worker never saw the go, so it never wrote the payload.";
    ASSERT_TRUE(barriered.dispatch_saw_done) << "The dispatch engine never observed the worker's done.";
    EXPECT_EQ(barriered.mismatches, 0u)
        << barriered.mismatches << " payload words were stale even though the worker drained its write before "
        << "signalling; the first was index " << barriered.first_mismatch_index << " reading "
        << barriered.first_mismatch_value
        << ". Either the barrier does not guarantee remote visibility or the destination was read wrongly.";

    const OrderingResult unbarriered = run_ordering_arm(dev, /*barrier_before_done=*/false, /*signal_via_fds=*/true);

    ASSERT_TRUE(unbarriered.worker_signalled) << "The worker never saw the go in the unbarriered arm.";
    ASSERT_TRUE(unbarriered.dispatch_saw_done) << "The dispatch engine never observed the done in the unbarriered arm.";

    if (unbarriered.mismatches > 0) {
        log_info(
            tt::LogTest,
            "WRITE-ORDERING HAZARD OBSERVED: without a barrier, {} of {} payload words were still stale when the "
            "FDS done was observed, the first at index {}. A completion fence is mandatory before FDS can carry "
            "worker completion.",
            unbarriered.mismatches,
            quasar_fds_test::ordering::kPayloadWords,
            unbarriered.first_mismatch_index);
    } else {
        log_info(
            tt::LogTest,
            "no write-ordering hazard observed on this platform: the payload was intact even without a barrier. "
            "This is NOT evidence that a fence is unnecessary — a functional simulator may not reorder at all. "
            "Gate 0b still needs silicon or a model certified for NOC and FDS ordering.");
    }

    // The control arm: identical in every way except that completion is announced by a NOC atomic
    // increment, the mechanism the current path uses, on the same virtual channel as the payload
    // write. Comparing it against the unbarriered FDS arm says whether today's completion path is
    // safe because kernels drain their writes, or safe by accident because NOC ordering holds the
    // atomic behind the data — and therefore what moving to a sideband actually gives up.
    const OrderingResult atomic_unbarriered =
        run_ordering_arm(dev, /*barrier_before_done=*/false, /*signal_via_fds=*/false);

    ASSERT_TRUE(atomic_unbarriered.dispatch_saw_done)
        << "The dispatch engine never observed the worker's NOC atomic increment, so the control arm measured "
           "nothing.";

    if (unbarriered.stale && !atomic_unbarriered.stale) {
        log_info(
            tt::LogTest,
            "NOC ORDERING IS ACTING AS A BACKSTOP: the same un-drained write was stale when announced over FDS "
            "and intact when announced by a NOC atomic. The current completion path is safe by accident, not by "
            "contract, and that protection is exactly what moving to the sideband gives up. A fence is not "
            "merely advisable for FDS, it replaces something real.");
    } else if (unbarriered.stale && atomic_unbarriered.stale) {
        log_info(
            tt::LogTest,
            "the NOC atomic is no safer: the payload was stale under both mechanisms, so the current completion "
            "path carries the same latent hazard and only the kernel drain contract prevents it. Re-enabling the "
            "post-kernel NOC flush assertions in dmk.cc matters regardless of FDS.");
    } else if (!unbarriered.stale) {
        log_info(
            tt::LogTest,
            "the FDS arm showed no staleness, so this comparison says nothing about the NOC atomic. Tune the "
            "payload size or the signal-to-read gap until the FDS arm exposes the race, then read this again.");
    }
}
