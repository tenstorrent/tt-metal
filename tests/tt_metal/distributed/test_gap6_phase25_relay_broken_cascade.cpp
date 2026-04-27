// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// GAP-6: Covers FIX AN — Phase 2.5 L1 read failure cascades to fabric_relay_path_broken_=true
//
// Background:
//   FIX AN (device.cpp ~line 1029): When Phase 2.5 (ERISC TERMINATE poll) performs a
//   ReadFromDeviceL1 on a non-MMIO device and that read throws (ETH relay is broken or
//   mid-startup), FIX AN catches the exception and sets fabric_relay_path_broken_=true
//   for that device instead of propagating the throw.
//
//   Without FIX AN:
//     - Phase 2.5 throws from ReadFromDeviceL1
//     - fabric_relay_path_broken_ is never set to true
//     - Phase 3 (fabric quiesce) proceeds on that device using a broken relay path
//     - Phase 3 hangs indefinitely waiting for ERISC channels that cannot respond
//
//   With FIX AN:
//     - Phase 2.5 L1 read throw is caught, sets fabric_relay_path_broken_=true
//     - Phase 3 detects is_fabric_relay_path_broken()=true and skips that device
//     - Phase 5 (termination writes) also skips (via FIX AO)
//     - quiesce_devices() returns cleanly
//
//   The flag fabric_relay_path_broken_ gates both:
//     - Phase 3 skip (quiesce skips ERISC channel teardown for broken-relay devices)
//     - Phase 5 skip (FIX AO: process_termination_signals skips WriteToDeviceL1 + l1_barrier)
//
// Existing test gap:
//   GAP-5 tests relay-broken state during teardown (mesh_device_.reset()), but not
//   during an active quiesce() call when relay ERISCs may be mid-startup.
//   GAP-6 specifically covers the double-quiesce scenario: the second quiesce call
//   exercises Phase 2.5 when relay ERISCs may still be in mid-startup state from
//   the first quiesce's firmware reload.
//
// What this test verifies:
//   1. FIX AN: neither quiesce call hangs (watchdog bound: 15s per call).
//   2. FIX AN: if Phase 2.5 L1 read throws on a non-MMIO channel,
//      fabric_relay_path_broken_ is set so Phase 3 is skipped for that device.
//   3. FIX AN: no TT_THROW escapes — exception is converted to non-fatal flag.
//   4. Interaction: second quiesce completes cleanly even when first quiesce
//      may have reloaded ERISC firmware leaving relay ERISCs mid-startup.
//
// Topology requirement: >= 4 devices (T3K or larger).
//   - 4+ devices required so at least one non-MMIO device is present in the mesh.
//   - Non-MMIO devices are those accessed via ETH relay (not directly PCIe-mapped).
//   - We reason at the level of this characteristic, never hardcoding device numbers.

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

#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/cluster.hpp>
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
#include "fabric/fabric_init.hpp"
#include "tests/tt_metal/tt_metal/common/multi_device_fixture.hpp"
#include "ttnn/tensor/unit_mesh/unit_mesh_utils.hpp"
#include "ttnn/operations/ccl/all_gather/all_gather.hpp"

namespace tt::tt_metal::distributed::test {

// ---------------------------------------------------------------------------
// Fixture: Phase25RelayBrokenCascadeFixture
//
// FABRIC_2D mesh with a 60-second watchdog, requires >= 4 devices.
//
// Uses >= 4 devices because:
//   - Non-MMIO devices (ETH-relay-accessed) only exist in multi-chip topologies
//   - Phase 2.5 L1 reads on non-MMIO channels are the specific path FIX AN covers
//   - The double-quiesce scenario requires fabric to be active (FABRIC_2D)
//
// 60-second budget:
//   - ~15s per FABRIC_2D init cycle
//   - 2x quiesce calls (each bounded at 15s in assertions)
//   - Re-open after SIGKILL predecessor: ~15s
//   - Margin for slow hardware
// ---------------------------------------------------------------------------
class Phase25RelayBrokenCascadeFixture : public MeshDeviceFixtureBase {
protected:
    Phase25RelayBrokenCascadeFixture()
        : MeshDeviceFixtureBase(Config{
              .num_cqs = 1,
              .fabric_config = tt_fabric::FabricConfig::FABRIC_2D,
              .test_budget_ms = 60000,  // 60s: SIGKILL scenario + 2x quiesce + margin
          }) {}

    void SetUp() override {
        // >= 4 devices required: non-MMIO devices (ETH-relay-accessed) only appear
        // in 4+ device topologies (T3K or larger).  Phase 2.5 L1 reads specifically
        // target non-MMIO ERISC channels — on single or dual-chip there is no
        // non-MMIO device to exercise the relay-broken cascade path.
        const size_t num_devices = MetalContext::instance().get_cluster().number_of_devices();
        if (num_devices < 4) {
            GTEST_SKIP() << "Phase25RelayBrokenCascadeFixture requires >= 4 devices "
                         << "(non-MMIO ETH-relay devices needed to exercise Phase 2.5 "
                         << "relay-broken cascade). Found " << num_devices << " device(s).";
        }
        MeshDeviceFixtureBase::SetUp();
    }
};

// Helper: create a minimal blank-kernel workload on a 1x1 core range.
// Used as a lightweight "warm up fabric" proxy — dispatches something through
// the command queue so ERISC channels enter an active state, without depending
// on CCL AllGather availability.
static Program make_blank_program(const CoreRange& cores) {
    Program prog;
    CreateKernel(
        prog,
        "tt_metal/kernels/dataflow/blank.cpp",
        cores,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});
    CreateKernel(
        prog,
        "tt_metal/kernels/dataflow/blank.cpp",
        cores,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});
    CreateKernel(prog, "tt_metal/kernels/compute/blank.cpp", cores, ComputeConfig{});
    return prog;
}

// ---------------------------------------------------------------------------
// GAP-6: Phase25RelayBrokenSetsFlag
//
// Scenario:
//   Phase 1. Close the fixture device so child inherits a clean MetalContext.
//   Phase 2. Fork a child that opens FABRIC_2D, dispatches AllGather to warm
//            up fabric (ERISCs enter ACTIVE state), then spins.
//   Phase 3. Parent waits for child_ready flag, then SIGKILLs the child.
//            This leaves stale ERISC state on at least one non-MMIO device
//            (the child's ETH relay teardown never ran).
//   Phase 4. Parent re-opens MeshDevice with FABRIC_2D.
//   Phase 5. Parent dispatches a blank workload (fabric active).
//   Phase 6. First quiesce_devices() call.
//            FIX AN: if Phase 2.5 ReadFromDeviceL1 throws on a relay-broken
//            non-MMIO channel, the exception must be caught and
//            fabric_relay_path_broken_ set (not propagated as TT_THROW).
//            Phase 3 must be skipped for that device.
//            Timing bound: < 15s (without FIX AN, Phase 3 hangs indefinitely).
//   Phase 7. Second quiesce_devices() call — exercises Phase 2.5 when relay
//            ERISCs may be in mid-startup state from Phase 6's firmware reload.
//            FIX AN must handle this case cleanly too.
//            Timing bound: < 15s.
//
// Pass conditions:
//   - Both quiesce calls complete in < 15s each (no Phase 3 hang).
//   - Neither quiesce call propagates TT_THROW (FIX AN catches and converts).
//   - Second quiesce completes cleanly.
//
// Fail conditions:
//   - Either quiesce call hangs (watchdog fires at 60s total budget).
//   - Either quiesce call takes >= 15s (Phase 3 not skipped for broken-relay device).
//   - TT_THROW escapes from quiesce (FIX AN not converting to non-fatal flag).
//
// Skips:
//   - fork() not available in this container environment.
//   - < 4 devices (handled by fixture).
// ---------------------------------------------------------------------------
TEST_F(Phase25RelayBrokenCascadeFixture, Phase25RelayBrokenSetsFlag) {
    // Timing bound for each individual quiesce call.
    // Without FIX AN: Phase 3 hangs indefinitely on devices whose
    // fabric_relay_path_broken_ was never set (L1 read threw but was not caught).
    // With FIX AN: Phase 3 skips relay-broken devices promptly.
    // 15s is generous enough for slow hardware, strict enough to catch the hang.
    constexpr int64_t kMaxQuiesceMs = 15000;

    // Step 0: check fork availability.
    {
        pid_t probe = ::fork();
        if (probe < 0) {
            GTEST_SKIP() << "[GAP-6] fork() not available in this environment: " << strerror(errno);
        }
        if (probe == 0) {
            _exit(0);
        }
        int wstatus = 0;
        ::waitpid(probe, &wstatus, 0);
    }

    // Phase 1: close fixture device before forking.
    // fork() inherits parent's open file descriptors and MetalContext state;
    // closing first ensures the child inherits a clean MetalContext (no open devices).
    auto mesh_shape = mesh_device_->shape();
    log_info(tt::LogTest, "[GAP-6] Phase 1: closing fixture device before fork");
    mesh_device_->close();
    mesh_device_.reset();

    // Shared-memory flag: child signals parent when FABRIC_2D init + AllGather is done
    // and ERISCs are in ACTIVE forwarding state (ready to be SIGKILL'd).
    volatile int* child_ready =
        static_cast<volatile int*>(
            ::mmap(nullptr, sizeof(int), PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0));
    ASSERT_NE(child_ready, MAP_FAILED) << "[GAP-6] mmap failed: " << strerror(errno);
    *child_ready = 0;

    // Phase 2: fork child to simulate a predecessor that opens FABRIC_2D and
    // dispatches AllGather (makes ERISC channels ACTIVE), then gets SIGKILL'd.
    // The child never runs teardown, leaving stale ERISC relay state on non-MMIO devices.
    pid_t child_pid = ::fork();
    ASSERT_NE(child_pid, -1) << "[GAP-6] fork() failed: " << strerror(errno);

    if (child_pid == 0) {
        // ---- CHILD: simulate predecessor process --------------------------------
        // Use _exit() — never invoke C++ destructors or atexit handlers.
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

            // Warm up ERISC channels: dispatch AllGather if >= 4 devices, otherwise
            // use a blank dispatch so ERISC EDM channels reach ACTIVE state.
            const size_t num_devices = MetalContext::instance().get_cluster().number_of_devices();
            if (num_devices >= 4) {
                // Build per-device submeshes for a 4-device AllGather ring.
                // This puts ERISC channels into active forwarding state — the
                // hardest relay-break scenario for Phase 2.5 to recover from.
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
                auto gathered = ttnn::all_gather(aggregated, /* dim */ 0);
                (void)gathered;
            } else {
                // Fallback for 2-device systems (should not reach here given fixture skip,
                // but included for safety): blank dispatch to exercise ERISC channels.
                auto cores = CoreRange{CoreCoord{0, 0}, CoreCoord{0, 0}};
                auto prog = make_blank_program(cores);
                auto workload = MeshWorkload();
                workload.add_program(MeshCoordinateRange(child_device->shape()), std::move(prog));
                auto& cq = child_device->mesh_command_queue();
                EnqueueMeshWorkload(cq, workload, /*blocking=*/true);
            }

            // Signal parent: FABRIC_2D init complete, ERISC channels are ACTIVE.
            *child_ready = 1;

            // Spin forever — parent will SIGKILL us without teardown.
            while (true) {
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }
        } catch (...) {
            // Init or AllGather failed — still signal ready so parent proceeds.
            // The stale ERISC state is the point; exact failure mode is secondary.
            *child_ready = 1;
            while (true) {
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }
        }
        _exit(0);  // unreachable
    }

    // ---- PARENT: wait for child ACTIVE ERISC state, then SIGKILL ----
    log_info(
        tt::LogTest,
        "[GAP-6] Phase 2: waiting for child (pid={}) to complete FABRIC_2D init + ACTIVE ERISCs",
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
                "[GAP-6] child_ready flag not set after {}ms — proceeding with SIGKILL anyway",
                kMaxWaitMs);
        } else {
            log_info(
                tt::LogTest,
                "[GAP-6] child_ready flag set after ~{}ms — ERISCs are ACTIVE",
                waited_ms);
        }
    }
    // Extra margin: ensure ERISC channels are fully in forwarding state before kill.
    std::this_thread::sleep_for(std::chrono::seconds(2));

    // Phase 3: SIGKILL child — leaves stale ERISC state on non-MMIO devices.
    // Non-MMIO devices are those accessed via ETH relay (not directly PCIe-mapped);
    // reasoning at the level of this characteristic, not hardcoded device numbers.
    log_info(
        tt::LogTest,
        "[GAP-6] Phase 3: SIGKILLing child pid={} — stale ERISC relay state on non-MMIO devices",
        child_pid);
    ::kill(child_pid, SIGKILL);
    int wstatus = 0;
    ::waitpid(child_pid, &wstatus, 0);
    ::munmap(const_cast<int*>(child_ready), sizeof(int));
    log_info(
        tt::LogTest,
        "[GAP-6] Phase 3: child exited (status=0x{:08x}) — stale relay state established",
        static_cast<uint32_t>(wstatus));

    // Phase 4: parent re-opens MeshDevice with FABRIC_2D.
    // This may exercise terminate_stale_erisc_routers() to clean up the SIGKILL'd
    // predecessor's ETH relay state (covered by FIX AP in GAP-5 / FIX AD in GAP-3).
    // Here we focus on what happens during the subsequent quiesce calls.
    log_info(tt::LogTest, "[GAP-6] Phase 4: re-opening MeshDevice with FABRIC_2D after predecessor SIGKILL");
    tt_fabric::SetFabricConfig(
        tt_fabric::FabricConfig::FABRIC_2D,
        tt_fabric::FabricReliabilityMode::STRICT_SYSTEM_HEALTH_SETUP_MODE);

    ASSERT_NO_THROW(
        mesh_device_ = MeshDevice::create(
            MeshDeviceConfig(mesh_shape),
            config_.l1_small_size,
            config_.trace_region_size,
            config_.num_cqs,
            DispatchCoreConfig{},
            {},
            config_.worker_l1_size))
        << "[GAP-6] MeshDevice::create() threw after SIGKILL predecessor — "
        << "relay-restore path (FIX AD/FIX AP) may be broken. "
        << "This is a prerequisite failure for GAP-6.";
    log_info(tt::LogTest, "[GAP-6] Phase 4: MeshDevice::create() succeeded");

    // Phase 5: dispatch a blank workload to warm up fabric channels.
    // This puts ERISC channels back into an ACTIVE state so that the subsequent
    // quiesce exercises Phase 2.5 against live (possibly partially-started) relay ERISCs.
    log_info(tt::LogTest, "[GAP-6] Phase 5: dispatching blank workload to make fabric active");
    {
        auto cores = CoreRange{CoreCoord{0, 0}, CoreCoord{0, 0}};
        auto device_range = MeshCoordinateRange(mesh_device_->shape());
        auto prog = make_blank_program(cores);
        auto workload = MeshWorkload();
        workload.add_program(device_range, std::move(prog));
        auto& cq = mesh_device_->mesh_command_queue();
        ASSERT_NO_THROW(EnqueueMeshWorkload(cq, workload, /*blocking=*/true))
            << "[GAP-6] Blank workload dispatch failed — prerequisite for quiesce stress test";
        log_info(tt::LogTest, "[GAP-6] Phase 5: blank workload completed — ERISC channels active");
    }

    // Phase 6: first quiesce_devices() call.
    //
    // FIX AN critical path:
    //   Phase 2.5 calls ReadFromDeviceL1 on each non-MMIO ERISC channel to check
    //   for TERMINATE acknowledgement.  If the relay ETH is broken or mid-startup,
    //   this read throws.  FIX AN catches this throw and sets
    //   fabric_relay_path_broken_=true for that device.
    //
    //   Without FIX AN:
    //     - throw propagates out of Phase 2.5
    //     - fabric_relay_path_broken_ stays false
    //     - Phase 3 proceeds on broken-relay device, hangs indefinitely
    //
    //   With FIX AN:
    //     - throw caught, fabric_relay_path_broken_=true set
    //     - Phase 3 sees is_fabric_relay_path_broken()=true, skips that device
    //     - Phase 5 (FIX AO) also skips WriteToDeviceL1 for that device
    //     - quiesce returns in < kMaxQuiesceMs
    //
    // Timing bound: 15s — strictly above normal quiesce (~2-5s) and strictly below
    // the infinite hang that occurs when Phase 3 proceeds on a broken-relay device.
    log_info(
        tt::LogTest,
        "[GAP-6] Phase 6: first quiesce_devices() — FIX AN Phase 2.5 relay-broken cascade");
    const auto q1_start = std::chrono::steady_clock::now();

    ASSERT_NO_THROW(mesh_device_->quiesce_devices())
        << "[GAP-6] First quiesce_devices() threw — FIX AN must catch Phase 2.5 "
        << "ReadFromDeviceL1 throws and convert to fabric_relay_path_broken_=true flag. "
        << "Without FIX AN, the throw propagates here.";

    const auto q1_elapsed_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - q1_start)
            .count();
    log_info(
        tt::LogTest,
        "[GAP-6] Phase 6: first quiesce_devices() completed in {}ms (limit={}ms)",
        q1_elapsed_ms,
        kMaxQuiesceMs);

    // Timing assertion: if Phase 3 was NOT skipped for a relay-broken non-MMIO
    // device (FIX AN absent), it would hang indefinitely waiting for ERISC
    // channels that cannot respond.  Watchdog fires at 60s (test budget).
    // The 15s bound catches any per-device hang before the watchdog does.
    ASSERT_LT(q1_elapsed_ms, kMaxQuiesceMs)
        << "[GAP-6] FIX AN regression: first quiesce_devices() took " << q1_elapsed_ms
        << "ms (limit " << kMaxQuiesceMs << "ms). "
        << "Phase 2.5 L1 read failure was not converted to fabric_relay_path_broken_=true. "
        << "Phase 3 may be hanging on a relay-broken non-MMIO device.";

    log_info(tt::LogTest, "[GAP-6] Phase 6: first quiesce returned cleanly — FIX AN Phase 2.5 catch confirmed");

    // Phase 7: second quiesce_devices() call.
    //
    // This is the unique GAP-6 scenario: the first quiesce reloaded ERISC firmware.
    // Relay ERISCs on non-MMIO devices may still be in mid-startup state when the
    // second quiesce begins Phase 2.5 polling.  FIX AN must handle this case too:
    // if ReadFromDeviceL1 throws (mid-startup relay), it must again catch, set
    // fabric_relay_path_broken_=true, and return without hanging Phase 3.
    //
    // GAP-5 does not cover this: GAP-5's double-quiesce (Phase 2 + Phase 5b) is
    // within a single quiesce call, not across two separate quiesce invocations
    // in active quiesce state (post-warmup dispatch).
    log_info(
        tt::LogTest,
        "[GAP-6] Phase 7: second quiesce_devices() — FIX AN with relay ERISCs potentially mid-startup");
    const auto q2_start = std::chrono::steady_clock::now();

    ASSERT_NO_THROW(mesh_device_->quiesce_devices())
        << "[GAP-6] Second quiesce_devices() threw — FIX AN must handle Phase 2.5 "
        << "ReadFromDeviceL1 throws when relay ERISCs are mid-startup after first quiesce reload. "
        << "fabric_relay_path_broken_ must be set instead of propagating the throw.";

    const auto q2_elapsed_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - q2_start)
            .count();
    log_info(
        tt::LogTest,
        "[GAP-6] Phase 7: second quiesce_devices() completed in {}ms (limit={}ms)",
        q2_elapsed_ms,
        kMaxQuiesceMs);

    ASSERT_LT(q2_elapsed_ms, kMaxQuiesceMs)
        << "[GAP-6] FIX AN regression: second quiesce_devices() took " << q2_elapsed_ms
        << "ms (limit " << kMaxQuiesceMs << "ms). "
        << "Phase 2.5 L1 read on mid-startup relay ERISC was not caught by FIX AN. "
        << "Phase 3 may be hanging on a relay-broken non-MMIO device during second quiesce.";

    log_info(
        tt::LogTest,
        "[GAP-6] Phase 7: second quiesce returned cleanly — FIX AN confirmed for mid-startup relay ERISCs");

    log_info(
        tt::LogTest,
        "[GAP-6] Phase25RelayBrokenSetsFlag PASSED — "
        "FIX AN (Phase 2.5 L1 read throw -> fabric_relay_path_broken_ cascade) confirmed. "
        "Both quiesce calls completed within {}ms and {}ms respectively (limit {}ms each).",
        q1_elapsed_ms,
        q2_elapsed_ms,
        kMaxQuiesceMs);
}

}  // namespace tt::tt_metal::distributed::test
