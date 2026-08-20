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
// The mapping from FDS wire index to core is not established, so a targeted mask cannot distinguish
// "wrong wire" from "no transport at all". The go is aimed at all 32 NEO wires and the worker
// accepts from all 3 dispatch instances, which covers the whole mapping space in one run.
constexpr uint32_t kWorkerMask = 0xFFFFFFFF;
constexpr uint32_t kDispatchMask = 0x7;
// Each side gives up rather than spinning forever, so a missing signal fails the test with a
// readable status word instead of hanging it. Kept modest because this runs under a cycle
// simulator, where a million iterations costs minutes of wall clock. Both signals are held rather
// than pulsed, so a shorter wait cannot miss one.
constexpr uint32_t kPollIterations = 100000;

std::vector<uint32_t> read_status(IDevice* device, const CoreCoord& core, uint32_t addr, CoreType core_type) {
    std::vector<uint32_t> status(quasar_fds_test::kNumSlots, 0);
    detail::ReadFromDeviceL1(device, core, addr, status.size() * sizeof(uint32_t), status, core_type);
    return status;
}

}  // namespace

// Drives the Quasar FDS sideband end to end: the dispatch-engine kernel writes L1, sends a go
// signal to a worker NEO and waits for that worker's done signal; the worker kernel waits for the
// go and answers with done.
//
// One data-movement core per side. A tile has a single FDS register block shared by all of its
// data-movement cores, so two cores on one tile would overwrite each other's configuration and
// consume each other's status.
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
    const CoreCoord worker_core{0, 0};

    const auto& hal = MetalContext::instance().hal();
    const uint32_t dispatch_l1 =
        hal.get_dev_addr(HalProgrammableCoreType::DISPATCH, HalL1MemAddrType::DEFAULT_UNRESERVED);
    const uint32_t worker_l1 = hal.get_dev_addr(HalProgrammableCoreType::TENSIX, HalL1MemAddrType::DEFAULT_UNRESERVED);

    std::vector<uint32_t> cleared(quasar_fds_test::kNumSlots, 0);
    detail::WriteToDeviceL1(dev, dispatch_core, dispatch_l1, cleared, CoreType::DISPATCH);
    detail::WriteToDeviceL1(dev, worker_core, worker_l1, cleared, CoreType::WORKER);

    Program program = CreateProgram();

    detail::CreateDispatchEngineKernel(
        program,
        OVERRIDE_KERNEL_PREFIX "tests/tt_metal/tt_metal/test_kernels/misc/quasar_dispatch_engine_signal.cpp",
        dispatch_core,
        experimental::quasar::QuasarDataMovementConfig{
            .num_threads_per_cluster = 1,
            .named_compile_args = {
                {"l1_address", dispatch_l1},
                {"group_id", kGroupId},
                {"worker_mask", kWorkerMask},
                {"poll_iterations", kPollIterations}}});

    experimental::quasar::CreateKernel(
        program,
        OVERRIDE_KERNEL_PREFIX "tests/tt_metal/tt_metal/test_kernels/misc/quasar_fds_worker_signal.cpp",
        worker_core,
        experimental::quasar::QuasarDataMovementConfig{
            .num_threads_per_cluster = 1,
            .named_compile_args = {
                {"l1_address", worker_l1},
                {"group_id", kGroupId},
                {"dispatch_mask", kDispatchMask},
                {"poll_iterations", kPollIterations}}});

    detail::LaunchProgram(dev, program, /*wait_until_cores_done=*/true);

    const auto dispatch_status = read_status(dev, dispatch_core, dispatch_l1, CoreType::DISPATCH);
    const auto worker_status = read_status(dev, worker_core, worker_l1, CoreType::WORKER);

    ASSERT_EQ(dispatch_status[quasar_fds_test::kSlotStarted], quasar_fds_test::kStarted)
        << "The dispatch-engine kernel did not run.";
    ASSERT_EQ(worker_status[quasar_fds_test::kSlotStarted], quasar_fds_test::kStarted)
        << "The worker kernel did not run.";

    log_info(
        tt::LogTest,
        "worker core {}: result={:#x} go_value={}",
        worker_core.str(),
        worker_status[quasar_fds_test::kSlotResult],
        worker_status[quasar_fds_test::kSlotObserved]);
    log_info(
        tt::LogTest,
        "dispatch core {}: result={:#x} group done count={}",
        dispatch_core.str(),
        dispatch_status[quasar_fds_test::kSlotResult],
        dispatch_status[quasar_fds_test::kSlotObserved]);

    EXPECT_EQ(worker_status[quasar_fds_test::kSlotResult], quasar_fds_test::kComplete)
        << "The worker never observed the FDS go signal.";
    EXPECT_EQ(dispatch_status[quasar_fds_test::kSlotResult], quasar_fds_test::kComplete)
        << "The dispatch engine never observed the worker's FDS done signal.";
}
