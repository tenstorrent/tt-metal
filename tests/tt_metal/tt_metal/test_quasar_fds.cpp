// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "common/device_fixture.hpp"
#include "context/metal_context.hpp"
#include "dispatch/dispatch_engine_cores.hpp"
#include "host_api/temp_quasar_api.hpp"

#include <fmt/ranges.h>
#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>
#include "llrt/rtoptions.hpp"

#include <algorithm>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#ifndef OVERRIDE_KERNEL_PREFIX
#define OVERRIDE_KERNEL_PREFIX ""
#endif

using namespace tt;
using namespace tt::tt_metal;

namespace {

// Every kernel in this suite writes a status block into its own L1, slot 0 being a result word.
// Each kernel stamps the result on entry and again on exit, and the host clears the block before
// launch, so the word separates a kernel that never ran (zero) from one that ran and stalled (the
// entry stamp) from one that ran to the end (kComplete) from one whose wait expired (a per-kernel
// failure code naming the step; the codes are documented in each kernel file).
constexpr uint32_t kSlotResult = 0;
constexpr uint32_t kComplete = 0x5A5A0002;

// Slots past the result word in the standard dispatch-engine kernel
// (quasar_dispatch_engine_signal.cpp).
constexpr uint32_t kSlotDoneCount = 1;
constexpr uint32_t kSlotCreditedQuietGroup = 2;
constexpr uint32_t kNumHandshakeDispatchSlots = 3;
constexpr uint32_t kNumHandshakeWorkerSlots = 1;

// Depth of the two FDS input-bus register arrays: one TENSIX_TO_DISPATCH register per NEO wire on
// the dispatch side, one DISPATCH_TO_TENSIX register per dispatch instance on the NEO side. The
// generated register headers name the individual registers but emit no count.
constexpr uint32_t kNumNeoWires = 32;
constexpr uint32_t kNumDispatchInstances = 3;

// Depth of the group register arrays, which both sides carry one of. Group 0 is the idle value on
// the wire, and groups 14 and 15 are the ready tokens of the kernels' initial handshake, leaving
// groups 1..13 to hand out.
constexpr uint32_t kNumAssignableGroups = 13;

// Every lane of a register array of the given depth, for masks that enable all of them.
constexpr uint32_t all_lanes_mask(uint32_t num_lanes) {
    return (num_lanes >= 32) ? ~uint32_t{0} : ((uint32_t{1} << num_lanes) - 1);
}

// The wire-to-core mapping is not established, so a targeted mask cannot distinguish "wrong wire"
// from "no transport at all". Enabling every lane covers the whole mapping space in one run. Each
// kernel scans the lanes its mask names rather than a count of its own, so these also bound the
// kernels' scan loops.
constexpr uint32_t kWorkerMask = all_lanes_mask(kNumNeoWires);
constexpr uint32_t kDispatchMask = all_lanes_mask(kNumDispatchInstances);

constexpr uint32_t kGroupId = 1;
// Each side gives up rather than spinning forever, so a missing signal fails the test with a
// readable status word instead of hanging it. Kept modest because this runs under a cycle
// simulator, where a million iterations costs minutes of wall clock. Both signals are held rather
// than pulsed, so a shorter wait cannot miss one.
constexpr uint32_t kPollIterations = 100000;
// Length of the windows in which a kernel asserts that nothing appears. Long enough that the event
// under test lands early in the window with orders of magnitude to spare, short enough not to
// dominate simulator wall clock.
constexpr uint32_t kSilenceIterations = 20000;

std::vector<CoreCoord> all_dispatch_engine_cores(IDevice* dev) {
    return detail::get_quasar_soc_dispatch_engine_logical_cores(
        MetalContext::instance().get_cluster().get_soc_desc(dev->id()));
}

std::string fds_tests_skip_reason(IDevice* dev) {
    if (!MetalContext::instance().rtoptions().is_simulator_or_emulated()) {
        return "This test can only be run under the simulator or emulator. "
               "Set TT_METAL_SIMULATOR or TT_METAL_EMULE_MODE=1.";
    }
    if (MetalContext::instance().rtoptions().get_fast_dispatch()) {
        return "This test can only run in slow dispatch mode. Set TT_METAL_SLOW_DISPATCH_MODE=1.";
    }
    if (MetalContext::instance().rtoptions().get_use_quasar_tensix_dispatch_cores()) {
        return "This test requires dispatch engines. Unset TT_METAL_TENSIX_DISPATCH_CORES.";
    }
    if (detail::sd_cq_kernel_tests_should_skip(dev)) {
        return "No dispatch engines detected.";
    }
    return {};
}

std::string fds_kernel_path(const std::string& kernel_name) {
    return OVERRIDE_KERNEL_PREFIX "tests/tt_metal/tt_metal/test_kernels/misc/fds/" + kernel_name;
}

const CoreRangeSet kSingleWorkerCore(CoreRange({0, 0}, {0, 0}));

// One group's worth of workers. Each set is launched with its own group id.
struct WorkerSet {
    CoreRangeSet cores;
    uint32_t group_id = kGroupId;
};

struct WorkerReport {
    CoreCoord core;
    uint32_t group_id = 0;
    std::vector<uint32_t> status;

    bool ran() const { return status[kSlotResult] != 0; }
    bool saw_own_go() const { return status[kSlotResult] == kComplete; }
    uint32_t result_word() const { return status[kSlotResult]; }
};

struct HandshakeResult {
    CoreCoord dispatch_core;
    std::vector<uint32_t> dispatch_status;
    std::vector<WorkerReport> workers;

    bool dispatch_ran() const { return dispatch_status[kSlotResult] != 0; }
    bool collected_all_dones() const { return dispatch_status[kSlotResult] == kComplete; }
    uint32_t result_word() const { return dispatch_status[kSlotResult]; }
    uint32_t done_count() const { return dispatch_status[kSlotDoneCount]; }
    uint32_t credited_quiet_group() const { return dispatch_status[kSlotCreditedQuietGroup]; }
};

std::vector<uint32_t> read_status(
    IDevice* dev, const CoreCoord& core, uint32_t addr, uint32_t num_words, CoreType core_type) {
    const CoreCoord virtual_core = dev->virtual_core_from_logical_core(core, core_type);
    return MetalContext::instance().get_cluster().read_core(
        dev->id(), virtual_core, addr, num_words * sizeof(uint32_t));
}

struct CoreStatus {
    CoreCoord core;
    std::vector<uint32_t> status;
};

struct FdsProgramResult {
    std::vector<CoreStatus> dispatch;
    std::vector<CoreStatus> workers;
};

// One instantiation of the worker kernel: the cores to run it on and the arguments they run with.
// Several are needed only when one program must give different groups different arguments.
struct WorkerGroup {
    CoreRangeSet cores;
    std::unordered_map<std::string, uint32_t> args;
};

// What one epoch of this suite launches. The l1_address arguments are filled in by the launcher, so
// no caller states them.
struct FdsProgram {
    std::vector<CoreCoord> dispatch_cores;
    std::string dispatch_kernel;
    std::unordered_map<std::string, uint32_t> dispatch_args;
    uint32_t num_dispatch_slots = 1;
    std::string worker_kernel;
    std::vector<WorkerGroup> worker_groups;
    uint32_t num_worker_slots = 1;
};

// Runs one epoch: the named kernel on each dispatch engine, the named kernel on each worker group,
// status blocks cleared before launch and read back after. One data movement core per node: a node
// has a single FDS register block shared by all of its data movement cores, so two cores on one
// node would overwrite each other's configuration and consume each other's status.
FdsProgramResult run_fds_program(IDevice* dev, FdsProgram spec) {
    const Hal& hal = MetalContext::instance().hal();
    const uint32_t dispatch_l1 =
        hal.get_dev_addr(HalProgrammableCoreType::DISPATCH, HalL1MemAddrType::DEFAULT_UNRESERVED);
    const uint32_t worker_l1 = hal.get_dev_addr(HalProgrammableCoreType::TENSIX, HalL1MemAddrType::DEFAULT_UNRESERVED);
    spec.dispatch_args["l1_address"] = dispatch_l1;

    std::vector<uint32_t> cleared_dispatch(spec.num_dispatch_slots, 0);
    std::vector<uint32_t> cleared_worker(spec.num_worker_slots, 0);

    Program program = CreateProgram();
    for (const CoreCoord& core : spec.dispatch_cores) {
        detail::WriteToDeviceL1(dev, core, dispatch_l1, cleared_dispatch, CoreType::DISPATCH);
        detail::CreateDispatchEngineKernel(
            program,
            spec.dispatch_kernel,
            core,
            experimental::quasar::QuasarDataMovementConfig{
                .num_threads_per_cluster = 1, .named_compile_args = spec.dispatch_args});
    }

    std::vector<CoreCoord> worker_cores;
    for (WorkerGroup& group : spec.worker_groups) {
        group.args["l1_address"] = worker_l1;
        for (const CoreCoord& core : corerange_to_cores(group.cores)) {
            detail::WriteToDeviceL1(dev, core, worker_l1, cleared_worker, CoreType::WORKER);
            worker_cores.push_back(core);
        }
        experimental::quasar::CreateKernel(
            program,
            spec.worker_kernel,
            group.cores,
            experimental::quasar::QuasarDataMovementConfig{
                .num_threads_per_cluster = 1, .named_compile_args = group.args});
    }

    detail::LaunchProgram(dev, program, /*wait_until_cores_done=*/true);
    MetalContext::instance().get_cluster().l1_barrier(dev->id());

    FdsProgramResult result;
    for (const CoreCoord& core : spec.dispatch_cores) {
        result.dispatch.push_back(
            CoreStatus{core, read_status(dev, core, dispatch_l1, spec.num_dispatch_slots, CoreType::DISPATCH)});
    }
    for (const CoreCoord& core : worker_cores) {
        result.workers.push_back(
            CoreStatus{core, read_status(dev, core, worker_l1, spec.num_worker_slots, CoreType::WORKER)});
    }
    return result;
}

// Runs one epoch of the handshake on the named dispatch engine. The engine sends a go for
// signalled_group only and waits for a done from every worker in that group; each worker waits for
// its own group's go and answers. Every other group named by worker_sets is configured on the
// dispatch side but never signalled, so its done count is evidence about leakage between groups.
HandshakeResult run_handshake(
    IDevice* dev, const CoreCoord& dispatch_core, const std::vector<WorkerSet>& worker_sets, uint32_t signalled_group) {
    // Only the signalled group's workers can answer, so only they count towards the wait; every
    // worker, whatever its group, signals ready and so gates the go.
    uint32_t done_threshold = 0;
    uint32_t num_workers = 0;
    uint32_t quiet_group_mask = 0;
    std::vector<WorkerGroup> worker_groups;
    worker_groups.reserve(worker_sets.size());
    for (const WorkerSet& set : worker_sets) {
        const uint32_t set_size = set.cores.num_cores();
        num_workers += set_size;
        if (set.group_id == signalled_group) {
            done_threshold += set_size;
        } else {
            quiet_group_mask |= 1u << set.group_id;
        }
        worker_groups.push_back(WorkerGroup{
            .cores = set.cores,
            .args = {
                {"group_id", set.group_id}, {"dispatch_mask", kDispatchMask}, {"poll_iterations", kPollIterations}}});
    }

    const FdsProgramResult program_result = run_fds_program(
        dev,
        FdsProgram{
            .dispatch_cores = {dispatch_core},
            .dispatch_kernel = fds_kernel_path("quasar_dispatch_engine_signal.cpp"),
            .dispatch_args =
                {{"group_id", signalled_group},
                 {"worker_mask", kWorkerMask},
                 {"done_threshold", done_threshold},
                 {"num_workers", num_workers},
                 {"quiet_group_mask", quiet_group_mask},
                 {"poll_iterations", kPollIterations}},
            .num_dispatch_slots = kNumHandshakeDispatchSlots,
            .worker_kernel = fds_kernel_path("quasar_fds_worker_signal.cpp"),
            .worker_groups = std::move(worker_groups),
            .num_worker_slots = kNumHandshakeWorkerSlots});

    // The launcher reports workers in the order it created them, which is the order of worker_sets,
    // so the group each core belongs to is recovered by walking the sets the same way.
    HandshakeResult result;
    result.dispatch_core = dispatch_core;
    result.dispatch_status = program_result.dispatch[0].status;
    size_t worker_index = 0;
    for (const WorkerSet& set : worker_sets) {
        for (uint32_t i = 0; i < set.cores.num_cores(); i++, worker_index++) {
            const CoreStatus& reported = program_result.workers[worker_index];
            result.workers.push_back(
                WorkerReport{.core = reported.core, .group_id = set.group_id, .status = reported.status});
        }
    }
    return result;
}

// Names what each side ended up seeing, so a partial result is readable without a rerun.
void log_handshake(const HandshakeResult& result) {
    for (const WorkerReport& worker : result.workers) {
        log_info(
            tt::LogTest,
            "worker core {} group {}: result={:#x}",
            worker.core.str(),
            worker.group_id,
            worker.result_word());
    }
    log_info(
        tt::LogTest,
        "dispatch core {}: result={:#x} done count={} credited quiet group={}",
        result.dispatch_core.str(),
        result.result_word(),
        result.done_count(),
        result.credited_quiet_group());
}

void log_fds_program(const FdsProgramResult& result) {
    for (const CoreStatus& reported : result.dispatch) {
        log_info(tt::LogTest, "dispatch core {}: status={:#x}", reported.core.str(), fmt::join(reported.status, " "));
    }
    for (const CoreStatus& reported : result.workers) {
        log_info(tt::LogTest, "worker core {}: status={:#x}", reported.core.str(), fmt::join(reported.status, " "));
    }
}

}  // namespace

class QuasarFdsFixture : public QuasarMeshDeviceSingleCardFixture {
protected:
    void SetUp() override {
        QuasarMeshDeviceSingleCardFixture::SetUp();
        if (IsSkipped()) {
            return;
        }
        device_ = devices_[0]->get_devices()[0];
        if (const std::string reason = fds_tests_skip_reason(device_); !reason.empty()) {
            GTEST_SKIP() << reason;
        }
    }

    CoreRangeSet full_worker_grid() const {
        const CoreCoord worker_grid = device_->compute_with_storage_grid_size();
        return CoreRangeSet(CoreRange({0, 0}, {worker_grid.x - 1, worker_grid.y - 1}));
    }

    IDevice* device_ = nullptr;
};

// Drives the Quasar FDS sideband end to end between each dispatch engine and one worker: the
// dispatch engine kernel writes L1, sends a go signal and waits for the worker's done signal; the
// worker kernel waits for the go and answers with done. Every engine on the chip takes its turn,
// because a sideband that works from one engine says nothing about the others: each engine reaches
// a worker over its own set of wires.
//
// Engines take turns rather than signalling together. A worker answers a group with a single done,
// so two engines signalling the same group at once would produce one done with nothing to say which
// engine's go released it.
TEST_F(QuasarFdsFixture, DispatchEngineSingleWorker) {
    for (const CoreCoord& dispatch_core : all_dispatch_engine_cores(device_)) {
        SCOPED_TRACE("dispatch engine " + dispatch_core.str());
        const HandshakeResult result =
            run_handshake(device_, dispatch_core, {WorkerSet{.cores = kSingleWorkerCore}}, kGroupId);
        log_handshake(result);

        if (!result.dispatch_ran()) {
            ADD_FAILURE() << "dispatch engine kernel did not run";
            continue;
        }
        if (!result.workers[0].ran()) {
            ADD_FAILURE() << "worker kernel did not run";
            continue;
        }

        EXPECT_TRUE(result.workers[0].saw_own_go()) << "worker never observed the go signal";
        EXPECT_TRUE(result.collected_all_dones()) << "dispatch engine never observed the worker's done signal";
    }
}

// The same handshake fanned out to every worker node the device offers, repeated for every dispatch
// engine on the chip, which is what a real worker-completion path has to do. One worker cannot
// distinguish a correct implementation from a wrong one: a go aimed at the wrong lane, a mask
// covering more lanes than exist, or a done count that stops at the first arrival all still pass
// with a single worker. Here the dispatch engine must accumulate one done per node before its wait
// is satisfied, which also proves each node lands on its own lane: a live count of N is a popcount
// over N distinct captured registers.
TEST_F(QuasarFdsFixture, DispatchEngineAllWorkers) {
    const CoreRangeSet worker_cores = full_worker_grid();
    const uint32_t num_workers = worker_cores.num_cores();
    if (num_workers < 2) {
        GTEST_SKIP() << "Test requires at least two worker nodes";
    }

    for (const CoreCoord& dispatch_core : all_dispatch_engine_cores(device_)) {
        SCOPED_TRACE("dispatch engine " + dispatch_core.str());
        const HandshakeResult result =
            run_handshake(device_, dispatch_core, {WorkerSet{.cores = worker_cores}}, kGroupId);
        log_handshake(result);

        if (!result.dispatch_ran()) {
            ADD_FAILURE() << "dispatch engine kernel did not run";
            continue;
        }

        for (const WorkerReport& worker : result.workers) {
            if (!worker.ran()) {
                ADD_FAILURE() << "worker kernel did not run on core " << worker.core.str();
                continue;
            }
            EXPECT_TRUE(worker.saw_own_go())
                << "worker on core " << worker.core.str() << " never observed the go signal";
        }

        EXPECT_TRUE(result.collected_all_dones())
            << "dispatch engine collected " << result.done_count() << " of " << num_workers << " worker done signals";
    }
}

// Confirms that a go addressed to one group reaches only that group's workers: a launch for one
// group must not release the workers of another.
//
// As many disjoint worker sets are formed as there are assignable groups, because a group filter
// that admits one wrong group need not admit every wrong group: the fewer groups compete, the less
// a passing run says.
//
// The dispatch side carries the discriminating check: no quiet group's done count may leave zero
// while the signalled group's reaches its full total, which exercises the per-group count decode
// (and only because the quiet groups are given a full enable mask — an unconfigured group counts
// nothing, whatever leaks). The worker side also asserts that quiet nodes never observe a go for
// their own group, but each engine drives a single value that workers decode themselves, so that
// direction can only fail if go delivery is rebuilt outright; it is insurance, not the test.
TEST_F(QuasarFdsFixture, DispatchEngineGroupIsolation) {
    // The initial handshake gates the go on every worker's ready token, all carried in one 32-lane
    // status word, so even split into groups this test cannot cover more nodes than lanes.
    const CoreRangeSet all_workers = full_worker_grid();
    const uint32_t num_workers = all_workers.num_cores();
    if (num_workers < 2) {
        GTEST_SKIP() << "Test requires at least two worker nodes";
    }

    const uint32_t num_groups = std::min(num_workers, kNumAssignableGroups);

    // Cut the flat list of worker nodes into that many runs, as evenly as the count divides, the
    // first signalled and the rest not. Splitting the list rather than a grid axis keeps this
    // correct for a grid of any shape.
    std::vector<WorkerSet> worker_sets;
    worker_sets.reserve(num_groups);
    for (uint32_t group_index = 0; group_index < num_groups; group_index++) {
        const uint32_t first_worker = num_workers * group_index / num_groups;
        const uint32_t last_worker = num_workers * (group_index + 1) / num_groups - 1;
        worker_sets.push_back(WorkerSet{
            .cores = select_from_corerangeset(all_workers, first_worker, last_worker),
            .group_id = kGroupId + group_index});
    }
    const uint32_t signalled_workers = worker_sets[0].cores.num_cores();

    // The group filter belongs to each engine's own configuration, so isolation has to hold from
    // whichever engine sends the go, not only from the first one.
    for (const CoreCoord& dispatch_core : all_dispatch_engine_cores(device_)) {
        SCOPED_TRACE("dispatch engine " + dispatch_core.str());
        const HandshakeResult result = run_handshake(device_, dispatch_core, worker_sets, kGroupId);
        log_handshake(result);

        if (!result.dispatch_ran()) {
            ADD_FAILURE() << "dispatch engine kernel did not run";
            continue;
        }

        for (const WorkerReport& worker : result.workers) {
            if (!worker.ran()) {
                ADD_FAILURE() << "worker kernel did not run on core " << worker.core.str();
                continue;
            }
            if (worker.group_id == kGroupId) {
                EXPECT_TRUE(worker.saw_own_go()) << "worker on core " << worker.core.str() << " in signalled group "
                                                 << kGroupId << " never observed its go signal";
                continue;
            }
            EXPECT_FALSE(worker.saw_own_go()) << "worker on core " << worker.core.str() << " accepted a go for group "
                                              << worker.group_id << ", which was never signalled";
        }

        EXPECT_EQ(result.credited_quiet_group(), 0u)
            << "group " << result.credited_quiet_group()
            << " was configured but never signalled, and was credited a done signal anyway";
        EXPECT_TRUE(result.collected_all_dones()) << "dispatch engine collected " << result.done_count() << " of "
                                                  << signalled_workers << " done signals from group " << kGroupId;
    }
}

// Several consecutive go/done rounds on the same group in one program, which is the re-signalling
// path a real launch sequence exercises and the single-round tests never do. Capture is
// change-triggered, so the protocol's clearing discipline — the engine dropping its go to zero
// between rounds, each worker dropping its done — is exactly what makes round N+1 observable. A
// capture that never re-arms after the intervening zero passes round one and fails round two, and
// a count that accumulated instead of falling fails round one's teardown. A model that simply
// followed the wire level passes here; DispatchEngineCaptureIsChangeTriggered is what catches it.
TEST_F(QuasarFdsFixture, DispatchEngineConsecutivePhases) {
    constexpr uint32_t kNumPhases = 3;
    // Slots and result codes of quasar_fds_phases_dispatch.cpp / quasar_fds_phases_worker.cpp.
    constexpr uint32_t kSlotPhasesDone = 1;
    constexpr uint32_t kNumSlots = 2;

    const CoreRangeSet workers = full_worker_grid();

    for (const CoreCoord& dispatch_core : all_dispatch_engine_cores(device_)) {
        SCOPED_TRACE("dispatch engine " + dispatch_core.str());
        const FdsProgramResult result = run_fds_program(
            device_,
            FdsProgram{
                .dispatch_cores = {dispatch_core},
                .dispatch_kernel = fds_kernel_path("quasar_fds_phases_dispatch.cpp"),
                .dispatch_args =
                    {{"group_id", kGroupId},
                     {"worker_mask", kWorkerMask},
                     {"done_threshold", workers.num_cores()},
                     {"num_phases", kNumPhases},
                     {"poll_iterations", kPollIterations}},
                .num_dispatch_slots = kNumSlots,
                .worker_kernel = fds_kernel_path("quasar_fds_phases_worker.cpp"),
                .worker_groups = {WorkerGroup{
                    .cores = workers,
                    .args =
                        {{"group_id", kGroupId},
                         {"dispatch_mask", kDispatchMask},
                         {"num_phases", kNumPhases},
                         {"poll_iterations", kPollIterations}}}},
                .num_worker_slots = kNumSlots});
        log_fds_program(result);

        const std::vector<uint32_t>& dispatch_status = result.dispatch[0].status;
        EXPECT_EQ(dispatch_status[kSlotResult], kComplete)
            << "dispatch engine completed " << dispatch_status[kSlotPhasesDone] << " of " << kNumPhases << " phases";
        for (const CoreStatus& worker : result.workers) {
            EXPECT_EQ(worker.status[kSlotResult], kComplete)
                << "worker on core " << worker.core.str() << " completed " << worker.status[kSlotPhasesDone] << " of "
                << kNumPhases << " phases";
        }
    }
}

// The decisive capture-semantics case: a software clear of an input register sticks while the
// sender holds its value, a rewrite of the identical value is not recaptured, and a change through
// zero is. A model that latched the level, or re-triggered on the write rather than on the wire
// changing, fails a different one of the three steps, and the worker's result word names which.
TEST_F(QuasarFdsFixture, DispatchEngineCaptureIsChangeTriggered) {
    // Slots of quasar_fds_capture_worker.cpp; the dispatch side carries only the result word.
    constexpr uint32_t kSlotObservedValue = 1;
    constexpr uint32_t kNumWorkerSlots = 2;

    for (const CoreCoord& dispatch_core : all_dispatch_engine_cores(device_)) {
        SCOPED_TRACE("dispatch engine " + dispatch_core.str());
        const FdsProgramResult result = run_fds_program(
            device_,
            FdsProgram{
                .dispatch_cores = {dispatch_core},
                .dispatch_kernel = fds_kernel_path("quasar_fds_capture_dispatch.cpp"),
                .dispatch_args =
                    {{"group_id", kGroupId}, {"worker_mask", kWorkerMask}, {"poll_iterations", kPollIterations}},
                .worker_kernel = fds_kernel_path("quasar_fds_capture_worker.cpp"),
                .worker_groups = {WorkerGroup{
                    .cores = kSingleWorkerCore,
                    .args =
                        {{"group_id", kGroupId},
                         {"dispatch_mask", kDispatchMask},
                         {"silence_iterations", kSilenceIterations},
                         {"poll_iterations", kPollIterations}}}},
                .num_worker_slots = kNumWorkerSlots});
        log_fds_program(result);

        EXPECT_EQ(result.dispatch[0].status[kSlotResult], kComplete);
        const std::vector<uint32_t>& worker_status = result.workers[0].status;
        EXPECT_EQ(worker_status[kSlotResult], kComplete)
            << "value observed during a silence window: " << worker_status[kSlotObservedValue];
    }
}

// Status, count and the interrupt are recomputed from the input registers every cycle; nothing
// accumulates. The dispatch-side kernel asserts the three observable consequences: the enable mask
// filters counting but never status, the count falls back to zero when the input registers are
// cleared under held senders, and group 0's status is nothing but the live map of idle lanes.
TEST_F(QuasarFdsFixture, DispatchEngineCountIsDerived) {
    // Slots of quasar_fds_derived_count_dispatch.cpp.
    constexpr uint32_t kSlotCountUnderEmptyEnable = 1;
    constexpr uint32_t kSlotStatusAfterClear = 2;
    constexpr uint32_t kSlotIdleStatus = 3;
    constexpr uint32_t kNumSlots = 4;

    const CoreRangeSet workers = full_worker_grid();

    for (const CoreCoord& dispatch_core : all_dispatch_engine_cores(device_)) {
        SCOPED_TRACE("dispatch engine " + dispatch_core.str());
        const FdsProgramResult result = run_fds_program(
            device_,
            FdsProgram{
                .dispatch_cores = {dispatch_core},
                .dispatch_kernel = fds_kernel_path("quasar_fds_derived_count_dispatch.cpp"),
                .dispatch_args =
                    {{"group_id", kGroupId},
                     {"worker_mask", kWorkerMask},
                     {"num_workers", workers.num_cores()},
                     {"silence_iterations", kSilenceIterations},
                     {"poll_iterations", kPollIterations}},
                .num_dispatch_slots = kNumSlots,
                .worker_kernel = fds_kernel_path("quasar_fds_worker_signal.cpp"),
                .worker_groups = {WorkerGroup{
                    .cores = workers,
                    .args =
                        {{"group_id", kGroupId},
                         {"dispatch_mask", kDispatchMask},
                         {"poll_iterations", kPollIterations}}}},
                .num_worker_slots = kNumHandshakeWorkerSlots});
        log_fds_program(result);

        const std::vector<uint32_t>& dispatch_status = result.dispatch[0].status;
        EXPECT_EQ(dispatch_status[kSlotResult], kComplete)
            << "count under empty enable=" << dispatch_status[kSlotCountUnderEmptyEnable] << " status after clear=0x"
            << std::hex << dispatch_status[kSlotStatusAfterClear] << " idle status=0x"
            << dispatch_status[kSlotIdleStatus];
        for (const CoreStatus& worker : result.workers) {
            EXPECT_EQ(worker.status[kSlotResult], kComplete) << "worker on core " << worker.core.str();
        }
    }
}

// Every dispatch engine signals the same group at once, and one worker refuses to answer until its
// own group status shows every engine's go held simultaneously. This is the only test of the
// worker-side group decode across its input lanes — the standard worker polls raw registers with a
// threshold of one. The simultaneity requirement proves the lanes do not collapse onto one
// register; it cannot prove full independence, since one engine's go mirrored onto every lane
// would pass just the same.
TEST_F(QuasarFdsFixture, DispatchEngineConcurrentEngines) {
    const std::vector<CoreCoord> engines = all_dispatch_engine_cores(device_);
    if (engines.size() < 2) {
        GTEST_SKIP() << "Test requires at least two dispatch engines";
    }

    // Slots of quasar_fds_aggregate_worker.cpp; the dispatch side is the standard handshake kernel.
    constexpr uint32_t kSlotStatusMask = 1;
    constexpr uint32_t kNumWorkerSlots = 2;
    constexpr uint32_t kDonesPerEngine = 1;

    const FdsProgramResult result = run_fds_program(
        device_,
        FdsProgram{
            .dispatch_cores = engines,
            .dispatch_kernel = fds_kernel_path("quasar_dispatch_engine_signal.cpp"),
            .dispatch_args =
                {{"group_id", kGroupId},
                 {"worker_mask", kWorkerMask},
                 {"done_threshold", kDonesPerEngine},
                 {"num_workers", kDonesPerEngine},
                 {"quiet_group_mask", 0},
                 {"poll_iterations", kPollIterations}},
            .num_dispatch_slots = kNumHandshakeDispatchSlots,
            .worker_kernel = fds_kernel_path("quasar_fds_aggregate_worker.cpp"),
            .worker_groups = {WorkerGroup{
                .cores = kSingleWorkerCore,
                .args =
                    {{"group_id", kGroupId},
                     {"dispatch_mask", kDispatchMask},
                     {"num_engines", static_cast<uint32_t>(engines.size())},
                     {"poll_iterations", kPollIterations}}}},
            .num_worker_slots = kNumWorkerSlots});
    log_fds_program(result);

    for (const CoreStatus& engine : result.dispatch) {
        EXPECT_EQ(engine.status[kSlotResult], kComplete)
            << "dispatch engine " << engine.core.str() << " never observed the worker's done signal";
    }
    const std::vector<uint32_t>& worker_status = result.workers[0].status;
    EXPECT_EQ(worker_status[kSlotResult], kComplete)
        << "worker observed go lanes 0x" << std::hex << worker_status[kSlotStatusMask] << " and needed " << std::dec
        << engines.size() << " held at once";
}

// The de-glitch filter does what it claims: a value replaced before the programmed threshold
// elapses is lost, a value held stable is captured at the same threshold, and the specification's
// floor threshold of 7 passes held values. Every other test runs with the filter off, so this is
// the only case where the threshold register's value matters at all.
TEST_F(QuasarFdsFixture, DispatchEngineDeglitchFilter) {
    // Far above the couple of cycles the pulse lives on the wire, far below the wait budgets.
    constexpr uint32_t kLongFilter = 5000;
    // The specification's minimum, set by wire-skew analysis.
    constexpr uint32_t kFloorFilter = 7;
    // Slots of quasar_fds_filter_worker.cpp; the dispatch side carries only the result word.
    constexpr uint32_t kSlotObservedValue = 1;
    constexpr uint32_t kNumWorkerSlots = 2;

    for (const CoreCoord& dispatch_core : all_dispatch_engine_cores(device_)) {
        SCOPED_TRACE("dispatch engine " + dispatch_core.str());
        const FdsProgramResult result = run_fds_program(
            device_,
            FdsProgram{
                .dispatch_cores = {dispatch_core},
                .dispatch_kernel = fds_kernel_path("quasar_fds_filter_dispatch.cpp"),
                .dispatch_args = {{"worker_mask", kWorkerMask}, {"poll_iterations", kPollIterations}},
                .worker_kernel = fds_kernel_path("quasar_fds_filter_worker.cpp"),
                .worker_groups = {WorkerGroup{
                    .cores = kSingleWorkerCore,
                    .args =
                        {{"dispatch_mask", kDispatchMask},
                         {"long_filter", kLongFilter},
                         {"floor_filter", kFloorFilter},
                         {"silence_iterations", kSilenceIterations},
                         {"poll_iterations", kPollIterations}}}},
                .num_worker_slots = kNumWorkerSlots});
        log_fds_program(result);

        EXPECT_EQ(result.dispatch[0].status[kSlotResult], kComplete);
        const std::vector<uint32_t>& worker_status = result.workers[0].status;
        EXPECT_EQ(worker_status[kSlotResult], kComplete)
            << "value observed during the silence window: " << worker_status[kSlotObservedValue];
    }
}

// Worker-side auto dispatch, the mode the specification's programming rules assume workers use for
// their dones: a done write is diverted into the queue rather than the output register — the stale
// register readback is the architected signature of the queued path — and the queued value is
// released onto the wire and reaches the engine. That the released value also holds after the
// queue drains is part of the architecture but not asserted here: the engine's captured input
// cannot tell a held wire from a pulse.
TEST_F(QuasarFdsFixture, DispatchEngineAutoDispatchDone) {
    // Release interval of the worker's queue, in its core cycles. Nothing here paces more than one
    // value, so it only needs to be sane.
    constexpr uint32_t kAutoDispatchCycles = 64;
    // Slots of quasar_fds_auto_done_worker.cpp; the dispatch side carries only the result word.
    constexpr uint32_t kSlotDoneReadback = 1;
    constexpr uint32_t kNumWorkerSlots = 2;

    // A distinct group id per engine, as insurance: the same worker core serves every iteration,
    // and if the queue's last released value survived the teardown it would satisfy the next
    // engine the moment the worker re-enables its queue — before it writes a done — and the
    // register readback cannot tell that apart, since zero is the correct answer either way.
    // Distinct ids keep any leftover unable to satisfy the wait, so each engine's pass requires a
    // done that really travelled through the queue. The kernels' static_assert against the ready
    // tokens bounds the range.
    uint32_t group_id = kGroupId;
    for (const CoreCoord& dispatch_core : all_dispatch_engine_cores(device_)) {
        SCOPED_TRACE("dispatch engine " + dispatch_core.str() + " group " + std::to_string(group_id));
        const FdsProgramResult result = run_fds_program(
            device_,
            FdsProgram{
                .dispatch_cores = {dispatch_core},
                .dispatch_kernel = fds_kernel_path("quasar_fds_auto_done_dispatch.cpp"),
                .dispatch_args =
                    {{"group_id", group_id}, {"worker_mask", kWorkerMask}, {"poll_iterations", kPollIterations}},
                .worker_kernel = fds_kernel_path("quasar_fds_auto_done_worker.cpp"),
                .worker_groups = {WorkerGroup{
                    .cores = kSingleWorkerCore,
                    .args =
                        {{"group_id", group_id},
                         {"dispatch_mask", kDispatchMask},
                         {"auto_dispatch_cycles", kAutoDispatchCycles},
                         {"poll_iterations", kPollIterations}}}},
                .num_worker_slots = kNumWorkerSlots});
        log_fds_program(result);

        EXPECT_EQ(result.dispatch[0].status[kSlotResult], kComplete)
            << "dispatch engine never observed the queued done";
        const std::vector<uint32_t>& worker_status = result.workers[0].status;
        EXPECT_EQ(worker_status[kSlotResult], kComplete)
            << "output register readback after the queued done: " << worker_status[kSlotDoneReadback];
        group_id++;
    }
}

// Auto dispatch pacing and backpressure on the engine's four-deep queue: six go values written
// back to back, counting upwards so every release is a change and every value is unique. The
// writes go through the polling helper, which waits out the queue-full flag rather than writing
// into a full queue, so what this asserts is pacing and ordering — the worker must capture every
// value exactly once, in order — plus that the full flag really asserted while the burst outran
// the cadence, which is also what catches a queue deeper than four or one that drops without
// raising its flag. The hardware's own block-on-full write path is not exercised here.
TEST_F(QuasarFdsFixture, DispatchEngineAutoDispatchPacing) {
    constexpr uint32_t kBurstLength = 6;
    // Release interval in engine core cycles: far above the worker's polling period so no released
    // value can be overwritten unseen, and far above the burst's write time so the queue genuinely
    // fills.
    constexpr uint32_t kAutoDispatchCycles = 8192;
    // Slots of the two kernels and the burst values, mirrored from
    // quasar_fds_auto_pacing_dispatch.cpp / quasar_fds_record_worker.cpp.
    constexpr uint32_t kSlotSawQueueFull = 1;
    constexpr uint32_t kSlotRecordedCount = 1;
    constexpr uint32_t kSlotFirstValue = 2;
    constexpr uint32_t kBurstValueBase = 2;
    constexpr uint32_t kNumDispatchSlots = 2;

    for (const CoreCoord& dispatch_core : all_dispatch_engine_cores(device_)) {
        SCOPED_TRACE("dispatch engine " + dispatch_core.str());
        const FdsProgramResult result = run_fds_program(
            device_,
            FdsProgram{
                .dispatch_cores = {dispatch_core},
                .dispatch_kernel = fds_kernel_path("quasar_fds_auto_pacing_dispatch.cpp"),
                .dispatch_args =
                    {{"worker_mask", kWorkerMask},
                     {"burst_length", kBurstLength},
                     {"auto_dispatch_cycles", kAutoDispatchCycles},
                     {"poll_iterations", kPollIterations}},
                .num_dispatch_slots = kNumDispatchSlots,
                .worker_kernel = fds_kernel_path("quasar_fds_record_worker.cpp"),
                .worker_groups = {WorkerGroup{
                    .cores = kSingleWorkerCore,
                    .args =
                        {{"dispatch_mask", kDispatchMask},
                         {"burst_length", kBurstLength},
                         {"poll_iterations", kPollIterations}}}},
                .num_worker_slots = kSlotFirstValue + kBurstLength});
        log_fds_program(result);

        const std::vector<uint32_t>& dispatch_status = result.dispatch[0].status;
        EXPECT_EQ(dispatch_status[kSlotResult], kComplete);
        EXPECT_NE(dispatch_status[kSlotSawQueueFull], 0u)
            << "the queue never reported full although the burst outran the release cadence";

        const std::vector<uint32_t>& worker_status = result.workers[0].status;
        EXPECT_EQ(worker_status[kSlotResult], kComplete)
            << "worker recorded " << worker_status[kSlotRecordedCount] << " of " << kBurstLength << " values";
        for (uint32_t i = 0; i < kBurstLength; i++) {
            EXPECT_EQ(worker_status[kSlotFirstValue + i], kBurstValueBase + i)
                << "value " << i << " arrived out of order";
        }
    }
}

// The auto dispatch trigger compares the full untruncated write address, while ordinary register
// decode uses only the low nine bits — the one place the OFFSET-form and ADDR-form register macros
// do not alias. An outbox programmed with the OFFSET form silently delivers nothing
// against the ADDR-form write the shipped fds_go issues; the matching ADDR form delivers. A model
// that truncated the comparison would deliver in both cases and hide the shipped hazard.
TEST_F(QuasarFdsFixture, DispatchEngineAutoDispatchOutboxMismatch) {
    constexpr uint32_t kAutoDispatchCycles = 64;
    // Slots of quasar_fds_outbox_mismatch_worker.cpp; the dispatch side carries only the result
    // word.
    constexpr uint32_t kSlotObservedValue = 1;
    constexpr uint32_t kNumWorkerSlots = 2;

    for (const CoreCoord& dispatch_core : all_dispatch_engine_cores(device_)) {
        SCOPED_TRACE("dispatch engine " + dispatch_core.str());
        const FdsProgramResult result = run_fds_program(
            device_,
            FdsProgram{
                .dispatch_cores = {dispatch_core},
                .dispatch_kernel = fds_kernel_path("quasar_fds_outbox_mismatch_dispatch.cpp"),
                .dispatch_args =
                    {{"worker_mask", kWorkerMask},
                     {"auto_dispatch_cycles", kAutoDispatchCycles},
                     {"poll_iterations", kPollIterations}},
                .worker_kernel = fds_kernel_path("quasar_fds_outbox_mismatch_worker.cpp"),
                .worker_groups = {WorkerGroup{
                    .cores = kSingleWorkerCore,
                    .args =
                        {{"dispatch_mask", kDispatchMask},
                         {"silence_iterations", kSilenceIterations},
                         {"poll_iterations", kPollIterations}}}},
                .num_worker_slots = kNumWorkerSlots});
        log_fds_program(result);

        EXPECT_EQ(result.dispatch[0].status[kSlotResult], kComplete);
        const std::vector<uint32_t>& worker_status = result.workers[0].status;
        EXPECT_EQ(worker_status[kSlotResult], kComplete)
            << "value observed on the wire: " << worker_status[kSlotObservedValue];
    }
}
