// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "common/device_fixture.hpp"
#include "dispatch/dispatch_engine_cores.hpp"
#include "host_api/temp_quasar_api.hpp"
#include "test_kernels/misc/quasar_fds_signal_status.h"

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

constexpr uint32_t kGroupId = 1;
// The mapping from FDS wire index to core is not established yet, and a targeted mask cannot
// distinguish "wrong wire" from "no transport at all". So aim the go at all 32 NEO wires and let
// the worker accept from all 3 dispatch instances: one run then covers the whole mapping space,
// and the kernels report which wire and instance actually carried the signal.
constexpr uint32_t kWorkerMask = 0xFFFFFFFF;
constexpr uint32_t kDispatchMask = 0x7;
// Each side gives up rather than spinning forever, so a missing signal fails the test with a
// readable status word instead of hanging it. Kept modest because this runs under a cycle
// simulator, where a million iterations costs minutes of wall clock. Six worker processors now
// poll at once, so the budget is smaller again; both signals are held rather than pulsed, so a
// shorter wait cannot miss one.
constexpr uint32_t kPollIterations = 100000;

std::vector<uint32_t> read_status(
    IDevice* device, const CoreCoord& core, uint32_t addr, CoreType core_type, uint32_t num_blocks) {
    std::vector<uint32_t> status(num_blocks * quasar_fds_test::kSlotsPerProcessor, 0);
    detail::ReadFromDeviceL1(device, core, addr, status.size() * sizeof(uint32_t), status, core_type);
    return status;
}

}  // namespace

// Drives the Quasar FDS sideband end to end: the dispatch-engine kernel writes L1, sends a go
// signal to a worker NEO and waits for that worker's done signal; the worker kernel waits for the
// go and answers with done.
//
// The worker drives its done part way through its wait whether or not a go arrived, so the two
// expectations below report on the two directions independently: a go that never lands no longer
// leaves the done direction untested.
//
// Both kernels run on every data-movement core of their tile rather than one. That began as a
// search for which processor the sideband reaches, and it found that the question does not arise:
// a tile has one register block shared by all of its data-movement cores, so every processor was
// reading and writing the same registers. The sweeps are kept because each processor also stamps
// its own index into a register and reads it back, which is what established that.
TEST_F(QuasarMeshDeviceSingleCardFixture, DispatchEngineSingleWorker) {
    auto& rtoptions = MetalContext::instance().rtoptions();

    // Emulation compiles kernels for the host, where the FDS ROCC custom instructions do not
    // exist, so this test needs the real simulator rather than is_simulator_or_emulated().
    if (!rtoptions.get_simulator_enabled()) {
        GTEST_SKIP() << "This test can only be run under the simulator. Set TT_METAL_SIMULATOR.";
    }
    if (!getenv("TT_METAL_SLOW_DISPATCH_MODE")) {
        GTEST_SKIP() << "Kernels on dispatch-engine cores require slow dispatch. "
                        "Set TT_METAL_SLOW_DISPATCH_MODE.";
    }
    if (rtoptions.get_use_quasar_tensix_dispatch_cores()) {
        GTEST_SKIP() << "Requires native dispatch-engine cores. Unset TT_METAL_TENSIX_DISPATCH_CORES.";
    }

    auto mesh_device = devices_[0];
    IDevice* dev = mesh_device->get_devices()[0];
    if (detail::sd_cq_kernel_tests_should_skip(dev)) {
        GTEST_SKIP() << "No dispatch-engine cores in the soc descriptor.";
    }

    const CoreCoord dispatch_core = detail::dispatch_engine_core(dev, 0);
    // Every worker tile the descriptor offers, not just the first: the dispatch engine's group 0
    // status watches all 32 done lanes at once, so the more processors drive done in one run, the
    // more of the lane space a single result covers.
    const CoreCoord worker_grid = dev->compute_with_storage_grid_size();
    const CoreRange worker_cores({0, 0}, {worker_grid.x - 1, worker_grid.y - 1});

    const auto& hal = MetalContext::instance().hal();
    const uint32_t dispatch_l1 =
        hal.get_dev_addr(HalProgrammableCoreType::DISPATCH, HalL1MemAddrType::DEFAULT_UNRESERVED);
    const uint32_t worker_l1 = hal.get_dev_addr(HalProgrammableCoreType::TENSIX, HalL1MemAddrType::DEFAULT_UNRESERVED);

    std::vector<uint32_t> cleared_dispatch(
        quasar_fds_test::kNumDispatchProcessors * quasar_fds_test::kSlotsPerProcessor, 0);
    std::vector<uint32_t> cleared_worker(
        quasar_fds_test::kNumWorkerProcessors * quasar_fds_test::kSlotsPerProcessor, 0);
    detail::WriteToDeviceL1(dev, dispatch_core, dispatch_l1, cleared_dispatch, CoreType::DISPATCH);
    for (const CoreCoord& core : corerange_to_cores(worker_cores)) {
        detail::WriteToDeviceL1(dev, core, worker_l1, cleared_worker, CoreType::WORKER);
    }

    Program program = CreateProgram();

    // Every data-movement core on the dispatch-engine tile, which reserves none of them. They all
    // share one register block, so this is no longer a search for the wired processor; it is what
    // lets each of them stamp the shared register and read back a neighbour's stamp.
    //
    // These cores take one kernel per processor with an explicit pin rather than one kernel with
    // several threads: the dispatch-engine entry point requires a single thread per cluster, and
    // the explicit-pin overload exists for cases like this that need to target a chosen processor.
    for (uint32_t processor = 0; processor < quasar_fds_test::kNumDispatchProcessors; processor++) {
        detail::CreateDispatchEngineKernel(
            program,
            OVERRIDE_KERNEL_PREFIX "tests/tt_metal/tt_metal/test_kernels/misc/quasar_dispatch_engine_signal.cpp",
            dispatch_core,
            static_cast<DataMovementProcessor>(processor),
            experimental::quasar::QuasarDataMovementConfig{
                .num_threads_per_cluster = 1,
                .named_compile_args = {
                    {"l1_address", dispatch_l1},
                    {"group_id", kGroupId},
                    {"worker_mask", kWorkerMask},
                    {"poll_iterations", kPollIterations}}});
    }

    // Every user data-movement core on every worker tile, for the same reasons: one shared block
    // per tile, and each processor stamping it so the sharing stays visible in every run.
    experimental::quasar::CreateKernel(
        program,
        OVERRIDE_KERNEL_PREFIX "tests/tt_metal/tt_metal/test_kernels/misc/quasar_fds_worker_signal.cpp",
        worker_cores,
        experimental::quasar::QuasarDataMovementConfig{
            .num_threads_per_cluster = experimental::quasar::QUASAR_NUM_USER_DM_CORES_PER_CLUSTER,
            .named_compile_args = {
                {"l1_address", worker_l1},
                {"group_id", kGroupId},
                {"dispatch_mask", kDispatchMask},
                {"poll_iterations", kPollIterations}}});

    // And every TRISC of every Tensix engine. The register block is named for the engine and the
    // 32 done lanes are one per engine across eight tiles, so the engine's own processors are the
    // last candidate endpoint after every data-movement core came back identical and idle.
    experimental::quasar::CreateKernel(
        program,
        OVERRIDE_KERNEL_PREFIX "tests/tt_metal/tt_metal/test_kernels/misc/quasar_fds_tensix_engine_signal.cpp",
        worker_cores,
        experimental::quasar::QuasarComputeConfig{
            .num_threads_per_cluster = experimental::quasar::QUASAR_NUM_TENSIX_ENGINES_PER_CLUSTER,
            .named_compile_args = {
                {"l1_address", worker_l1},
                {"group_id", kGroupId},
                {"dispatch_mask", kDispatchMask},
                {"poll_iterations", kPollIterations}}});

    detail::LaunchProgram(dev, program, /*wait_until_cores_done=*/true);

    const auto dispatch_status =
        read_status(dev, dispatch_core, dispatch_l1, CoreType::DISPATCH, quasar_fds_test::kNumDispatchProcessors);

    // Same treatment as the worker side: count the processors that took part and name only the
    // ones that saw something, so a single responding processor is not lost among the rest.
    uint32_t dispatch_processors_that_ran = 0;
    bool any_dispatch_saw_done = false;
    for (uint32_t processor = 0; processor < quasar_fds_test::kNumDispatchProcessors; processor++) {
        const uint32_t* slots = &dispatch_status[processor * quasar_fds_test::kSlotsPerProcessor];
        if (slots[quasar_fds_test::kSlotStarted] != quasar_fds_test::kStarted) {
            continue;
        }
        dispatch_processors_that_ran++;
        const bool saw_done = (slots[quasar_fds_test::kSlotResult] == quasar_fds_test::kComplete);
        any_dispatch_saw_done |= saw_done;
        if (saw_done) {
            log_info(
                tt::LogTest,
                "dispatch core {} processor {} OBSERVED A DONE: group done count={}",
                dispatch_core.str(),
                processor,
                slots[quasar_fds_test::kSlotObserved]);
        }
    }
    log_info(tt::LogTest, "dispatch processors that ran: {}", dispatch_processors_that_ran);

    // Count every processor on every tile that took part, and name any that reached the go, so a
    // wired one stands out against the rest whether or not the handshake completes.
    uint32_t processors_that_ran = 0;
    bool any_processor_saw_go = false;
    for (const CoreCoord& core : corerange_to_cores(worker_cores)) {
        const auto worker_status =
            read_status(dev, core, worker_l1, CoreType::WORKER, quasar_fds_test::kNumWorkerProcessors);
        for (uint32_t processor = 0; processor < quasar_fds_test::kNumWorkerProcessors; processor++) {
            const uint32_t* slots = &worker_status[processor * quasar_fds_test::kSlotsPerProcessor];
            if (slots[quasar_fds_test::kSlotStarted] != quasar_fds_test::kStarted) {
                continue;
            }
            processors_that_ran++;
            const bool saw_go = (slots[quasar_fds_test::kSlotResult] == quasar_fds_test::kComplete);
            any_processor_saw_go |= saw_go;
            // Only the ones that reached a go are named individually. Listing forty idle
            // processors buries the one line that would matter.
            if (saw_go) {
                log_info(
                    tt::LogTest,
                    "worker core {} processor {} OBSERVED THE GO: go_value={}",
                    core.str(),
                    processor,
                    slots[quasar_fds_test::kSlotObserved]);
            }
        }
    }
    log_info(tt::LogTest, "worker processors that ran across all tiles: {}", processors_that_ran);

    ASSERT_GT(dispatch_processors_that_ran, 0u) << "No dispatch-engine kernel ran on any processor.";
    ASSERT_GT(processors_that_ran, 0u) << "No worker kernel ran on any processor.";

    EXPECT_TRUE(any_processor_saw_go) << "None of the " << processors_that_ran
                                      << " worker processors that ran observed the FDS go signal.";
    EXPECT_TRUE(any_dispatch_saw_done) << "None of the " << dispatch_processors_that_ran
                                       << " dispatch-engine processors that ran observed a worker's FDS done signal.";
}
