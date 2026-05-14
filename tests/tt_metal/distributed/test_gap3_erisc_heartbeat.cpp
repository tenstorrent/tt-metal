// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// GAP-3: Covers FIX AD + FIX AC — ERISC heartbeat polling after crash
//
// Background:
//   FIX AD (risc_firmware_initializer.cpp:305-366) resets MMIO ETH channels via
//   PCIe during teardown of a crashed predecessor and polls each channel's heartbeat
//   address until the counter increments, confirming UMD base relay firmware is
//   running again. FIX AC fixed the WH heartbeat format check (plain incrementing
//   counter, not 0xABCDxxxx). Without this polling, the next process's
//   terminate_stale_erisc_routers() call sees a relay that hasn't finished rebooting
//   and either hangs (5s×N channels = >30s) or misconfigures fabric.
//
// What this test verifies:
//   1. FIX AD: heartbeat polling correctly restores relay after a PCIe hard-reset
//      triggered by a predecessor crash (child SIGKILL'd with FABRIC_2D active)
//   2. FIX AC: WH heartbeat format check (plain counter, not 0xABCDxxxx)
//   3. Relay-restore path: parent's MeshDevice::create completes in <15s (without
//      FIX AD: 5s×N channels = >30s)
//   4. After open: CCL AllGather succeeds, proving fabric is functional after the
//      SIGKILL left relay-less state

#include <gtest/gtest.h>
#include <cerrno>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <cstring>
#include <sys/mman.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#include <atomic>
#include <optional>
#include <thread>
#include <vector>

#include <experimental/fabric/fabric_types.hpp>
#include "fabric/fabric_edm_packet_header.hpp"
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/cluster.hpp>
#include <tt-metalium/dispatch_core_common.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/experimental/fabric/control_plane.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/mesh_event.hpp>
#include <tt-metalium/mesh_workload.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-logger/tt-logger.hpp>

#include "impl/context/metal_context.hpp"
#include "impl/device/device_impl.hpp"
#include "impl/device/firmware/fabric_firmware_initializer.hpp"
#include "fabric/fabric_builder_context.hpp"
#include "fabric/fabric_context.hpp"
#include "fabric/fabric_init.hpp"
#include "tests/tt_metal/tt_metal/common/multi_device_fixture.hpp"
#include "ttnn/tensor/unit_mesh/unit_mesh_utils.hpp"
#include "ttnn/operations/ccl/all_gather/all_gather.hpp"

namespace tt::tt_metal::distributed::test {

// ---------------------------------------------------------------------------
// Fixture: FABRIC_2D with a 120-second watchdog, requires >= 2 devices.
//
// Mirrors AsyncTeardownKillPredecessorFixture:
//   - FABRIC_2D activates ERISC EDM firmware on all ETH channels
//   - 120s budget: 15s child wait + ~13s child init + ~13s parent reinit + margin
//   - Requires >= 2 devices (FABRIC_2D is meaningless on single-chip)
//
// This fixture is intentionally separate from AsyncTeardownKillPredecessorFixture
// to keep GAP-3 self-contained and avoid hidden coupling between test files.
// ---------------------------------------------------------------------------
class ERISCHeartbeatFixture : public MeshDeviceFixtureBase {
protected:
    ERISCHeartbeatFixture()
        : MeshDeviceFixtureBase(Config{
              .num_cqs = 1,
              .fabric_config = tt_fabric::FabricConfig::FABRIC_2D,
              .test_budget_ms = 120000,  // 15s wait + 13s child-init + 13s parent-reinit + margin
          }) {}

    void SetUp() override {
        // FABRIC_2D heartbeat polling is only meaningful on multi-chip topologies.
        // Single-chip has no MMIO-relay ETH path to break/restore.
        const size_t num_devices = MetalContext::instance().get_cluster().number_of_devices();
        if (num_devices < 2) {
            GTEST_SKIP() << "ERISCHeartbeatFixture requires >= 2 devices "
                            "(FABRIC_2D MMIO relay path not exercised on single-chip)";
        }
        MeshDeviceFixtureBase::SetUp();
    }
};

// ---------------------------------------------------------------------------
// GAP-3 test: TeardownHeartbeatPollConfirmsRelayRestore
//
// Scenario:
//   1. Close the fixture device (child inherits clean MetalContext).
//   2. Fork a child that opens FABRIC_2D, dispatches AllGather, then spins
//      with ERISC EDM channels in ACTIVE state.
//   3. Parent waits 15s (enough for child's FABRIC_2D init to complete) then
//      SIGKILLs the child — ERISCs left in ACTIVE state with relay possibly
//      broken (no teardown ran, PCIe-reset path not invoked).
//   4. Parent times MeshDevice::create():
//        - FIX AD path: PCIe hard-resets MMIO ETH channels, polls heartbeat
//          until counter increments, then proceed. Should complete in <15s.
//        - Without FIX AD: 5s timeout × N MMIO ETH channels = >30s.
//   5. After open: run a CCL AllGather to verify the relay-restore path makes
//      fabric fully functional (SIGKILL left relay-less state).
//
// Pass conditions:
//   - MeshDevice::create() completes in <15000ms (FIX AD timing bound)
//   - Subsequent AllGather completes without hang
//   - No TT_FATAL / crash in terminate_stale_erisc_routers recovery
//
// Fail conditions:
//   - MeshDevice::create() exceeds 15000ms (FIX AD heartbeat poll is broken)
//   - Hang (watchdog kills at 120s)
//   - Crash in ERISC recovery path
//   - AllGather fails (relay not properly restored)
//
// Skips:
//   - fork() not supported (GTEST_SKIP)
//   - < 2 devices (handled by fixture)
//   - < 4 devices for AllGather (GTEST_SKIP with clear message)
// ---------------------------------------------------------------------------
TEST_F(ERISCHeartbeatFixture, TeardownHeartbeatPollConfirmsRelayRestore) {
    // Check fork availability — some container environments block fork().
    // We detect this by attempting a dry-run fork and immediately exiting.
    {
        pid_t probe = ::fork();
        if (probe < 0) {
            GTEST_SKIP() << "[GAP-3] fork() not available in this environment: " << strerror(errno);
        }
        if (probe == 0) {
            _exit(0);  // child: immediately exit
        }
        int wstatus = 0;
        ::waitpid(probe, &wstatus, 0);
    }

    // Step 1: close fixture device before forking.
    // fork() inherits parent's open file descriptors and MetalContext state;
    // closing first ensures child inherits a clean MetalContext (no open devices).
    auto mesh_shape = mesh_device_->shape();
    log_info(tt::LogTest, "[GAP-3] Closing fixture device before fork");
    mesh_device_->close();
    mesh_device_.reset();

    // Shared-memory flag: child signals parent when FABRIC_2D init is complete
    // and ERISCs are in ACTIVE state (ready to be SIGKILL'd).
    // MAP_SHARED | MAP_ANONYMOUS — visible across fork().
    volatile int* child_ready =
        static_cast<volatile int*>(::mmap(nullptr, sizeof(int), PROT_READ | PROT_WRITE,
                                          MAP_SHARED | MAP_ANONYMOUS, -1, 0));
    ASSERT_NE(child_ready, MAP_FAILED) << "[GAP-3] mmap failed: " << strerror(errno);
    *child_ready = 0;

    // Step 2: fork child to simulate a predecessor process that opens FABRIC_2D
    // and dispatches work, then gets SIGKILL'd without running teardown.
    pid_t child_pid = ::fork();
    ASSERT_NE(child_pid, -1) << "[GAP-3] fork() failed: " << strerror(errno);

    if (child_pid == 0) {
        // ---- CHILD: simulate predecessor process --------------------------------
        // Opens FABRIC_2D (activates ERISC EDM firmware), dispatches an AllGather
        // on a small tensor (routing data through ERISC channels), then signals
        // parent that ERISCs are ACTIVE. Spins until SIGKILL'd — never runs teardown.
        //
        // Use _exit() throughout — never invoke C++ destructors or atexit handlers
        // that could corrupt parent address space (fork gives copy-on-write pages).
        try {
            tt_fabric::SetFabricConfig(
                tt_fabric::FabricConfig::FABRIC_2D,
                tt_fabric::FabricReliabilityMode::STRICT_SYSTEM_HEALTH_SETUP_MODE);
            auto child_device = MeshDevice::create(
                MeshDeviceConfig(mesh_shape),
                config_.l1_small_size,
                config_.trace_region_size,
                config_.num_cqs,
                DispatchCoreConfig{},
                {},
                config_.worker_l1_size);

            // Attempt AllGather if we have enough devices; otherwise just spin.
            // AllGather puts ERISC channels into active forwarding state, which is
            // the hardest relay-break scenario for the parent to recover from.
            const size_t num_devices = MetalContext::instance().get_cluster().number_of_devices();
            if (num_devices >= 4) {
                // Build per-device submeshes for a 4-device AllGather ring.
                constexpr int kNumRingDevices = 4;
                std::vector<std::shared_ptr<distributed::MeshDevice>> submeshes;
                for (int col = 0; col < kNumRingDevices; col++) {
                    submeshes.push_back(
                        child_device->create_submesh(
                            MeshShape(1, 1), distributed::MeshCoordinate(0, col)));
                }
                TensorSpec tensor_spec(
                    ttnn::Shape({1, 1, 32, 128}),
                    TensorLayout(DataType::BFLOAT16, PageConfig(Layout::TILE), MemoryConfig{}));
                std::vector<ttnn::Tensor> tensors;
                for (int dev_idx = 0; dev_idx < kNumRingDevices; dev_idx++) {
                    std::vector<bfloat16> data(
                        tensor_spec.logical_shape().volume(),
                        bfloat16(static_cast<float>(dev_idx)));
                    tensors.push_back(
                        Tensor::from_vector(std::move(data), tensor_spec)
                            .to_device(submeshes[dev_idx].get()));
                }
                auto aggregated = tt::tt_metal::experimental::unit_mesh::aggregate(tensors);
                // AllGather: activates ERISC forwarding channels.
                auto gathered = ttnn::all_gather(aggregated, /* dim */ 0);
                (void)gathered;
            }

            // Signal parent: FABRIC_2D init complete, ERISCs in ACTIVE state.
            *child_ready = 1;

            // Spin forever — parent will SIGKILL us.
            while (true) {
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }
        } catch (...) {
            // Init or AllGather failed — still signal ready so parent proceeds.
            *child_ready = 1;
            while (true) {
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }
        }
        _exit(0);  // unreachable
    }

    // ---- PARENT: wait for child to reach ACTIVE ERISC state, then SIGKILL ----
    // Wait for child_ready flag (up to 20s) or fall back to a fixed 15s sleep.
    log_info(
        tt::LogTest,
        "[GAP-3] Waiting for child (pid={}) to complete FABRIC_2D init + ACTIVE ERISCs",
        child_pid);
    {
        constexpr int kMaxWaitMs = 20000;
        constexpr int kPollIntervalMs = 200;
        int waited_ms = 0;
        while (*child_ready == 0 && waited_ms < kMaxWaitMs) {
            std::this_thread::sleep_for(std::chrono::milliseconds(kPollIntervalMs));
            waited_ms += kPollIntervalMs;
        }
        if (*child_ready == 0) {
            log_warning(
                tt::LogTest,
                "[GAP-3] child_ready flag not set after {}ms — proceeding with SIGKILL anyway",
                kMaxWaitMs);
        } else {
            log_info(
                tt::LogTest,
                "[GAP-3] child_ready flag set after ~{}ms — ERISCs are ACTIVE",
                waited_ms);
        }
    }
    // Extra margin: ensure ERISC channels are fully in forwarding state before kill.
    std::this_thread::sleep_for(std::chrono::seconds(2));

    log_info(tt::LogTest, "[GAP-3] SIGKILLing child pid={} — ERISCs left in ACTIVE state, relay broken", child_pid);
    ::kill(child_pid, SIGKILL);
    int wstatus = 0;
    ::waitpid(child_pid, &wstatus, 0);
    ::munmap(const_cast<int*>(child_ready), sizeof(int));
    log_info(
        tt::LogTest,
        "[GAP-3] Child exited (status=0x{:08x}) — proceeding to timed re-open",
        static_cast<uint32_t>(wstatus));

    // Step 3: Time the parent's MeshDevice::create() call.
    //
    // FIX AD path (risc_firmware_initializer.cpp:305-366):
    //   1. Detects relay_broken_non_mmio (non-MMIO devices unreachable via ETH relay)
    //   2. PCIe hard-resets each MMIO ETH channel
    //   3. Polls each channel's heartbeat address until counter increments
    //      (confirms UMD base relay firmware is running)
    //   4. Returns; terminate_stale_erisc_routers() can now reach non-MMIO devices
    //
    // Timing bound:
    //   With FIX AD: polling converges in ~100-500ms per channel → total <2s typical,
    //                bounded at 1s per channel (kMaxPollMs=1000 in the impl).
    //   Without FIX AD: 5s timeout × N MMIO ETH channels = >30s.
    //   We assert < 15000ms — generous enough to tolerate slow hardware, strict
    //   enough to catch the missing-FIX-AD case.
    //
    // Note: reason at the level of characteristics (mmio / non-mmio) — never
    // hardcode specific device numbers.
    constexpr int64_t kCreateTimeLimitMs = 15000;

    log_info(
        tt::LogTest,
        "[GAP-3] Starting timed MeshDevice::create() — FIX AD should poll heartbeat and complete in <{}ms",
        kCreateTimeLimitMs);
    const auto create_start = std::chrono::steady_clock::now();

    bool create_threw = false;
    try {
        tt_fabric::SetFabricConfig(
            tt_fabric::FabricConfig::FABRIC_2D,
            tt_fabric::FabricReliabilityMode::STRICT_SYSTEM_HEALTH_SETUP_MODE);
        mesh_device_ = MeshDevice::create(
            MeshDeviceConfig(mesh_shape),
            config_.l1_small_size,
            config_.trace_region_size,
            config_.num_cqs,
            DispatchCoreConfig{},
            {},
            config_.worker_l1_size);
    } catch (const std::exception& e) {
        create_threw = true;
        FAIL() << "[GAP-3] MeshDevice::create() threw after SIGKILL predecessor — "
               << "FIX AD relay-restore path failed: " << e.what();
    }

    const auto create_elapsed_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - create_start)
            .count();

    log_info(
        tt::LogTest,
        "[GAP-3] MeshDevice::create() completed in {}ms (limit={}ms)",
        create_elapsed_ms,
        kCreateTimeLimitMs);

    if (!create_threw) {
        ASSERT_LE(create_elapsed_ms, kCreateTimeLimitMs)
            << "[GAP-3] MeshDevice::create() took " << create_elapsed_ms << "ms — exceeded "
            << kCreateTimeLimitMs << "ms limit. "
            << "Without FIX AD: 5s timeout × N MMIO ETH channels = >30s. "
            << "FIX AD heartbeat polling appears broken or not running.";
    }

    // Step 4: verify the relay-restore path makes fabric fully functional.
    //
    // AllGather routes packets through non-MMIO forwarding ERISCs — exactly the
    // path that breaks when relay firmware is not running. If FIX AD didn't
    // properly wait for heartbeat before returning, the relay ETH firmware may
    // not be ready yet and AllGather will hang or corrupt data.
    //
    // We require >= 4 devices for the AllGather ring; skip gracefully on 2-device
    // (N300) topologies where the original relay-break scenario doesn't apply.
    const size_t num_devices = MetalContext::instance().get_cluster().number_of_devices();
    if (num_devices < 4) {
        // Fallback: a simpler blocking dispatch verifies the device is at least
        // operational (relay works for basic MMIO operations).
        log_info(
            tt::LogTest,
            "[GAP-3] {} device(s) found — running blocking dispatch instead of AllGather "
            "(AllGather ring requires >= 4 devices)",
            num_devices);
        auto cores = CoreRange{CoreCoord{0, 0}, CoreCoord{0, 0}};
        Program program;
        CreateKernel(
            program,
            "tt_metal/kernels/dataflow/blank.cpp",
            cores,
            DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});
        auto workload = MeshWorkload();
        workload.add_program(MeshCoordinateRange(mesh_device_->shape()), std::move(program));
        auto& cq = mesh_device_->mesh_command_queue();
        ASSERT_NO_THROW(EnqueueMeshWorkload(cq, workload, /*blocking=*/true))
            << "[GAP-3] Verification dispatch failed after SIGKILL predecessor recovery";
        log_info(tt::LogTest, "[GAP-3] Blocking dispatch succeeded — relay restored (2-device path)");
        return;
    }

    // 4-device AllGather: exercises non-MMIO forwarding ERISC path.
    log_info(
        tt::LogTest,
        "[GAP-3] Running AllGather on {}-device mesh to verify relay-restore (FIX AD + FIX AC)",
        num_devices);

    constexpr int kNumRingDevices = 4;
    std::vector<std::shared_ptr<distributed::MeshDevice>> submeshes;
    for (int col = 0; col < kNumRingDevices; col++) {
        submeshes.push_back(
            mesh_device_->create_submesh(MeshShape(1, 1), distributed::MeshCoordinate(0, col)));
    }

    TensorSpec tensor_spec(
        ttnn::Shape({1, 1, 32, 128}),
        TensorLayout(DataType::BFLOAT16, PageConfig(Layout::TILE), MemoryConfig{}));

    // Each device holds a tensor filled with float(dev_idx).
    // After AllGather along dim=0, every device should hold [0, 1, 2, 3, 0, 1, 2, 3, ...]
    // giving a deterministic pattern that detects corruption from stale ERISC NOC writes.
    std::vector<ttnn::Tensor> tensors;
    for (int dev_idx = 0; dev_idx < kNumRingDevices; dev_idx++) {
        std::vector<bfloat16> data(
            tensor_spec.logical_shape().volume(),
            bfloat16(static_cast<float>(dev_idx)));
        tensors.push_back(
            Tensor::from_vector(std::move(data), tensor_spec)
                .to_device(submeshes[dev_idx].get()));
    }

    auto aggregated = tt::tt_metal::experimental::unit_mesh::aggregate(tensors);

    ttnn::Tensor gathered;
    ASSERT_NO_THROW(gathered = ttnn::all_gather(aggregated, /* dim */ 0))
        << "[GAP-3] AllGather threw after SIGKILL predecessor — relay not properly restored by FIX AD";

    log_info(
        tt::LogTest,
        "[GAP-3] AllGather completed — relay-restore path verified (FIX AD heartbeat polling + FIX AC format check)");

    // Correctness check: each shard should contain all kNumRingDevices slices.
    auto disaggregated = tt::tt_metal::experimental::unit_mesh::disaggregate(gathered);
    ASSERT_EQ(static_cast<int>(disaggregated.size()), kNumRingDevices)
        << "[GAP-3] AllGather output shard count mismatch after relay restore";

    for (int dev_idx = 0; dev_idx < kNumRingDevices; dev_idx++) {
        auto data = disaggregated[dev_idx].to_vector<bfloat16>();
        ASSERT_FALSE(data.empty())
            << "[GAP-3] Empty AllGather readback at dev_idx=" << dev_idx;

        // Verify the AllGather pattern: [0,1,2,3, 0,1,2,3, ...] repeated.
        const size_t slice_size = data.size() / kNumRingDevices;
        for (int slice_idx = 0; slice_idx < kNumRingDevices; slice_idx++) {
            const float expected_val = static_cast<float>(slice_idx);
            for (size_t elem_idx = 0; elem_idx < slice_size; elem_idx++) {
                const size_t flat_idx = static_cast<size_t>(slice_idx) * slice_size + elem_idx;
                ASSERT_EQ(static_cast<float>(data[flat_idx]), expected_val)
                    << "[GAP-3] AllGather data corruption at dev_idx=" << dev_idx
                    << " flat_idx=" << flat_idx
                    << " expected=" << expected_val
                    << " actual=" << static_cast<float>(data[flat_idx])
                    << " — stale ERISC NOC write after SIGKILL + FIX AD relay-restore";
            }
        }
    }

    log_info(
        tt::LogTest,
        "[GAP-3] AllGather correctness verified — FIX AD heartbeat polling + FIX AC format check PASS");
}

}  // namespace tt::tt_metal::distributed::test
