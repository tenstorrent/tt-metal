// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <vector>
#include <map>
#include <memory>
#include <string>
#include <gtest/gtest.h>
#include <tt-logger/tt-logger.hpp>

#include <tt-metalium/allocator.hpp>
#include <tt_stl/assert.hpp>
#include <tt-metalium/base_types.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/hal_types.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/program.hpp>
#include <tt-metalium/tt_metal.hpp>
#include "impl/context/metal_context.hpp"
#include "common/mesh_dispatch_fixture.hpp"
#include "jit_build/jit_build_settings.hpp"  // ::SemScope, host mirror of the device enum
#include <tt-metalium/experimental/metal2_host_api/program.hpp>

namespace tt::tt_metal {

// ============================================================================
// Scoped semaphore (SemScope) tests.
// ============================================================================
// Exercises the scoped device Semaphore class (noc_semaphore.h) end-to-end
// across all three mechanisms -- LOCAL_NONATOMIC (plain L1 RMW),
// DM_LOCAL_CACHED (32-bit AMO on the cached alias), EXTERNAL (self-targeted
// NoC atomic). The host resolves each semaphore's mechanism purely from a
// binder-topology census (program_spec.cpp) -- nothing is configurable on the
// SemaphoreSpec -- so every test here constructs the SHAPE that makes the
// census pick the mechanism under test. Raw hardware atomicity is covered by
// the keystone tests (NocSelfAtomicFixture).
// ============================================================================
class SemScopeFixture : public MeshDispatchFixture {
protected:
    static constexpr experimental::NodeCoord core = {0, 0};
    static constexpr uint32_t iterations{64};
    static constexpr uint32_t concurrent_iterations{20};  // per-thread; NoC round-trips are slow on emu
    const std::string kernel_path = "tests/tt_metal/tt_metal/test_kernels/dataflow/sem_scope_smoke.cpp";
    const std::string kernel_path_concurrent = "tests/tt_metal/tt_metal/test_kernels/dataflow/sem_scope_concurrent.cpp";
    const std::string kernel_path_coexist = "tests/tt_metal/tt_metal/test_kernels/dataflow/sem_scope_coexist.cpp";
    const std::string kernel_path_census = "tests/tt_metal/tt_metal/test_kernels/dataflow/sem_census_probe.cpp";
    const std::string kernel_path_remote = "tests/tt_metal/tt_metal/test_kernels/dataflow/sem_scope_remote.cpp";
    const std::string kernel_path_slot_probe = "tests/tt_metal/tt_metal/test_kernels/dataflow/sem_scope_slot_probe.cpp";
    uint32_t report_addr{0};
    uint32_t num_dms_{0};
    std::shared_ptr<distributed::MeshDevice> mesh_device_;
    IDevice* device_{nullptr};
    std::vector<uint32_t> result;

    void SetUp() override {
        MeshDispatchFixture::SetUp();
        if (arch_ != tt::ARCH::QUASAR) {
            GTEST_SKIP() << "SemScope suite is Gen2 (Quasar) only: its specs use DataMovementGen2Config";
        }
        mesh_device_ = devices_[0];
        device_ = mesh_device_->get_devices()[0];
        report_addr = device_->allocator()->get_base_allocator_addr(HalMemType::L1);
        num_dms_ = MetalContext::instance().hal().get_processor_types_count(HalProgrammableCoreType::TENSIX, 0);
        num_dms_ = std::min(num_dms_, 6u);  // Metal 2.0 reserves DM0/DM1
    }

    // Runs the smoke kernel and returns the value it read back from the semaphore after
    // `iterations` increments (and, with_down, decrements). `want` is the scope the census must
    // resolve for the shape this helper builds:
    //   LOCAL_NONATOMIC -> sole binder kernel, one thread: nothing can race, the cheap pick
    //                      (resolution pinned by TestCensusSingleWriterPicksLocal)
    //   EXTERNAL        -> a read-only OBSERVER kernel binds the semaphore from second_node():
    //                      an OFF-NODE binder is what forces the NoC atomic (pinned by
    //                      TestCensusOffNodeBinderBlocksCached) -- an on-node observer would
    //                      stay DM_LOCAL_CACHED now. Callers must skip-guard on has_second_node().
    // DM_LOCAL_CACHED is NOT constructible here: the cached pick needs >= 2 writer threads and
    // the smoke kernel reports without cross-thread sync, so there is no deterministic readback.
    // Cached coverage lives in run_concurrent and the census-probe tests.
    uint32_t run_scope(SemScope want, bool with_down = false, bool sentinel_down = false) {
        if (want == SemScope::DM_LOCAL_CACHED) {
            ADD_FAILURE() << "run_scope cannot build a cached shape (needs >= 2 unsynchronised writer threads)";
            return 0u;
        }
        // Prefill with a sentinel, not 0: the with_down tests expect 0, so a zero prefill would
        // pass even if the kernel never reported.
        std::vector<uint32_t> sentinel(1, kNoReport);
        tt::tt_metal::detail::WriteToDeviceL1(device_, core, report_addr, sentinel);

        distributed::MeshWorkload workload;
        Program program;
        distributed::MeshCoordinate zero_coord{0, 0};
        distributed::MeshCoordinateRange device_range{zero_coord, zero_coord};

        experimental::SemaphoreSpec counter_sem{
            .unique_id = experimental::SemaphoreSpecName{"counter_sem"},
            .target_nodes = core,
        };

        std::map<std::string, std::string> defs;
        if (sentinel_down) {
            defs.emplace("SEM_SCOPE_SENTINEL_DOWN", "1");  // set(0xFFFFFFFF) then down(1) twice
        }
        if (with_down) {
            defs.emplace("SEM_SCOPE_UPDOWN", "1");  // kernel also does down(N) after up(N)
        }
        experimental::KernelSpec::CompilerOptions::Defines defines_obj(defs);

        const experimental::KernelSpecName DM_KERNEL{"sem_scope_kernel"};
        const experimental::KernelSpecName OBSERVER{"sem_scope_observer"};
        std::vector<experimental::KernelSpec> kernel_specs;
        kernel_specs.push_back(experimental::KernelSpec{
            .unique_id = DM_KERNEL,
            .source = kernel_path,
            .num_threads = 1,
            .compiler_options = {.defines = defines_obj},
            .semaphore_bindings =
                {{.semaphore_spec_name = experimental::SemaphoreSpecName{"counter_sem"}, .accessor_name = "counter"}},
            .runtime_arg_schema = {.runtime_arg_names = {"report_addr", "increment_times"}},
            .hw_config = experimental::DataMovementGen2Config{},
        });
        std::vector<experimental::WorkUnitSpec> work_units{experimental::WorkUnitSpec{
            .name = "main",
            .kernels = {DM_KERNEL},
            .target_nodes = core,
        }};
        if (want == SemScope::EXTERNAL) {
            // Off-node binder kernel: the census probe with 0 increments and no report is a pure
            // observer; placed on second_node() it gives the semaphore off-node reach, so the
            // census must pick EXTERNAL. (On-node it would no longer demote: any number of
            // on-node DM binder kernels is cached geometry.)
            kernel_specs.push_back(experimental::KernelSpec{
                .unique_id = OBSERVER,
                .source = kernel_path_census,
                .num_threads = 1,
                .semaphore_bindings =
                    {{.semaphore_spec_name = experimental::SemaphoreSpecName{"counter_sem"},
                      .accessor_name = "counter"}},
                .runtime_arg_schema =
                    {.runtime_arg_names =
                         {"report_addr", "increment_times", "is_reporter", "barrier_idx", "wait_min_total"}},
                .hw_config = experimental::DataMovementGen2Config{},
            });
            // WorkUnitSpecs must have DISJOINT target nodes -> the observer gets its own WU.
            work_units.push_back(experimental::WorkUnitSpec{
                .name = "observer",
                .kernels = {OBSERVER},
                .target_nodes = second_node(),
            });
        }

        experimental::ProgramSpec spec{
            .name = "sem_scope_smoke",
            .kernels = kernel_specs,
            .semaphores = {counter_sem},
            .work_units = work_units,
        };
        program = experimental::MakeProgramFromSpec(*mesh_device_, spec);

        experimental::ProgramRunArgs params;
        params.kernel_run_args = {
            experimental::ProgramRunArgs::KernelRunArgs{
                .kernel = DM_KERNEL,
                .runtime_arg_values = experimental::MakeRuntimeArgsForSingleNode(
                    core, {{"report_addr", report_addr}, {"increment_times", iterations}}),
            },
        };
        if (want == SemScope::EXTERNAL) {
            params.kernel_run_args.push_back(experimental::ProgramRunArgs::KernelRunArgs{
                .kernel = OBSERVER,
                .runtime_arg_values = experimental::MakeRuntimeArgsForSingleNode(
                    second_node(),
                    {{"report_addr", report_addr},
                     {"increment_times", 0u},
                     {"is_reporter", 0u},
                     {"barrier_idx", 0u},
                     {"wait_min_total", 0u}}),
            });
        }
        experimental::SetProgramRunArgs(program, params);

        workload.add_program(device_range, std::move(program));
        RunProgram(mesh_device_, workload);

        tt::tt_metal::detail::ReadFromDeviceL1(device_, core, report_addr, sizeof(uint32_t), result);
        EXPECT_EQ(result.size(), 1u);
        // The kernel must have overwritten the sentinel, or the 0-expecting tests pass vacuously.
        EXPECT_NE(result.empty() ? kNoReport : result[0], kNoReport) << "kernel never reported";
        return result.empty() ? 0u : result[0];
    }

    // Multi-DM concurrency proof (roles gated by mhartid in the kernel). num_dms threads
    // share one bound Semaphore; mode picks CONCURRENT_UP, PRODUCER_CONSUMER or
    // MULTI_CONSUMER. `want` is the scope the census must resolve for the shape built here:
    //   DM_LOCAL_CACHED -> both semaphores single-node with every binder (here one multi-threaded
    //                      DM kernel) on that node = cached geometry (pinned by
    //                      TestCachedSemLivesInPoolNotRing). No longer the ONLY cached shape --
    //                      any number of on-node DM binder kernels stays cached; the multi-kernel
    //                      variant is pinned by TestCachedMultiKernelExactCount.
    //   EXTERNAL        -> both semaphores span a SECOND node, which disqualifies the per-core
    //                      pool (pinned by TestCensusMultiNodeSemPicksExternal); the kernel's
    //                      threads still all run on `core` and hammer the local cell. An extra
    //                      binder kernel is NOT used here: an on-node one no longer demotes to
    //                      EXTERNAL, and it would eat a DM hart and could displace hart 2, the
    //                      kernel's mhartid-gated reporter.
    // Callers wanting EXTERNAL must skip-guard on has_second_node().
    // Returns the reporter's value() readback, and asserts the kernel-reported baked scope of
    // BOTH bound semaphores equals `want` -- so a census change can't silently demote these
    // shapes to a different (still count-exact) mechanism while the tests stay green.
    uint32_t run_concurrent(SemScope want, const std::string& mode_define) {
        const bool has_watchdog = mode_define == "MODE_MULTI_CONSUMER";
        // Sentinel prefill: see run_scope. The watchdog word lives on its OWN 64B line.
        std::vector<uint32_t> sentinel(3, kNoReport);
        tt::tt_metal::detail::WriteToDeviceL1(device_, core, report_addr, sentinel);
        if (has_watchdog) {
            std::vector<uint32_t> wd_sentinel(1, kNoReport);
            tt::tt_metal::detail::WriteToDeviceL1(device_, core, report_addr + 64u, wd_sentinel);
        }

        distributed::MeshWorkload workload;
        Program program;
        distributed::MeshCoordinate zero_coord{0, 0};
        distributed::MeshCoordinateRange device_range{zero_coord, zero_coord};

        experimental::Nodes sem_target = core;
        if (want == SemScope::EXTERNAL) {
            sem_target = experimental::NodeRange{core, second_node()};
        }
        experimental::SemaphoreSpec counter_sem{
            .unique_id = experimental::SemaphoreSpecName{"counter_sem"}, .target_nodes = sem_target};
        experimental::SemaphoreSpec done_sem{
            .unique_id = experimental::SemaphoreSpecName{"done_sem"}, .target_nodes = sem_target};

        std::map<std::string, std::string> defs{{mode_define, "1"}};
        experimental::KernelSpec::CompilerOptions::Defines defines_obj(defs);

        const experimental::KernelSpecName DM_KERNEL{"sem_scope_concurrent_kernel"};
        experimental::KernelSpec kernel_spec{
            .unique_id = DM_KERNEL,
            .source = kernel_path_concurrent,
            .num_threads = num_dms_,
            .compiler_options = {.defines = defines_obj},
            .semaphore_bindings =
                {{.semaphore_spec_name = experimental::SemaphoreSpecName{"counter_sem"}, .accessor_name = "counter"},
                 {.semaphore_spec_name = experimental::SemaphoreSpecName{"done_sem"}, .accessor_name = "done"}},
            .runtime_arg_schema = {.runtime_arg_names = {"report_addr", "increment_times", "num_threads"}},
            .hw_config = experimental::DataMovementGen2Config{},
        };

        experimental::WorkUnitSpec main_wu{.name = "main", .kernels = {DM_KERNEL}, .target_nodes = core};
        experimental::ProgramSpec spec{
            .name = "sem_scope_concurrent",
            .kernels = {kernel_spec},
            .semaphores = {counter_sem, done_sem},
            .work_units = {main_wu},
        };
        program = experimental::MakeProgramFromSpec(*mesh_device_, spec);

        experimental::ProgramRunArgs params;
        params.kernel_run_args = {
            experimental::ProgramRunArgs::KernelRunArgs{
                .kernel = DM_KERNEL,
                .runtime_arg_values = experimental::MakeRuntimeArgsForSingleNode(
                    core,
                    {{"report_addr", report_addr},
                     {"increment_times", concurrent_iterations},
                     {"num_threads", num_dms_}}),
            },
        };
        experimental::SetProgramRunArgs(program, params);
        workload.add_program(device_range, std::move(program));
        RunProgram(mesh_device_, workload);

        tt::tt_metal::detail::ReadFromDeviceL1(device_, core, report_addr, 3 * sizeof(uint32_t), result);
        EXPECT_EQ(result.size(), 3u);
        if (result.size() < 3) {
            return 0u;
        }
        // The kernel must have overwritten the sentinel, or the 0-expecting tests pass vacuously.
        EXPECT_NE(result[0], kNoReport) << "kernel never reported";
        EXPECT_EQ(result[1], static_cast<uint32_t>(want)) << "counter_sem baked scope != shape intent";
        EXPECT_EQ(result[2], static_cast<uint32_t>(want)) << "done_sem baked scope != shape intent";
        if (has_watchdog) {
            // The final count is conservation-blind to a broken down() lock (fixed decrement
            // counts: a double-spend's wrap self-cancels). The watchdog's max-observed value is
            // the detector: it can never exceed the credits issued under a correct lock.
            std::vector<uint32_t> wd;
            tt::tt_metal::detail::ReadFromDeviceL1(device_, core, report_addr + 64u, sizeof(uint32_t), wd);
            EXPECT_EQ(wd.size(), 1u);
            if (!wd.empty()) {
                const uint32_t total_credits = (num_dms_ - 2) * concurrent_iterations;
                EXPECT_NE(wd[0], kNoReport) << "watchdog never reported";
                EXPECT_LE(wd[0], total_credits)
                    << "watchdog saw the semaphore exceed the credits issued: a double-spent "
                       "credit wrapped the word -> the multi-consumer down() lock is broken";
            }
        }
        return result[0];
    }

    // Coexistence run: a DM_LOCAL_CACHED semaphore (pool) + an EXTERNAL semaphore (ring) hammered
    // concurrently by all DMs; returns {cached_final, external_final}, both expected num_dms*iters.
    // Shapes select the mechanisms: cached_sem is single-node with all binders (one multi-threaded
    // DM kernel) on its node (= cached geometry -> pool); external_sem/done_sem span a second node, which
    // disqualifies the per-core pool (-> ring). Callers must skip-guard on has_second_node().
    // Also asserts the kernel-reported baked scopes: cached_sem must actually be the pool
    // (DM_LOCAL_CACHED) and external_sem the ring (EXTERNAL), or the coexistence proof is void.
    std::pair<uint32_t, uint32_t> run_coexist() {
        std::vector<uint32_t> zero(4, 0);
        tt::tt_metal::detail::WriteToDeviceL1(device_, core, report_addr, zero);

        distributed::MeshWorkload workload;
        Program program;
        distributed::MeshCoordinate zero_coord{0, 0};
        distributed::MeshCoordinateRange device_range{zero_coord, zero_coord};

        const experimental::NodeRange two_nodes{core, second_node()};
        experimental::SemaphoreSpec cached_sem{
            .unique_id = experimental::SemaphoreSpecName{"cached_sem"}, .target_nodes = core};  // -> pool
        experimental::SemaphoreSpec external_sem{
            .unique_id = experimental::SemaphoreSpecName{"external_sem"}, .target_nodes = two_nodes};  // -> ring
        experimental::SemaphoreSpec done_sem{
            .unique_id = experimental::SemaphoreSpecName{"done_sem"}, .target_nodes = two_nodes};  // barrier

        const experimental::KernelSpecName DM_KERNEL{"sem_coexist_kernel"};
        experimental::KernelSpec kernel_spec{
            .unique_id = DM_KERNEL,
            .source = kernel_path_coexist,
            .num_threads = num_dms_,
            .semaphore_bindings =
                {{.semaphore_spec_name = experimental::SemaphoreSpecName{"cached_sem"}, .accessor_name = "cached"},
                 {.semaphore_spec_name = experimental::SemaphoreSpecName{"external_sem"}, .accessor_name = "external"},
                 {.semaphore_spec_name = experimental::SemaphoreSpecName{"done_sem"}, .accessor_name = "done"}},
            .runtime_arg_schema = {.runtime_arg_names = {"report_addr", "increment_times", "num_threads"}},
            .hw_config = experimental::DataMovementGen2Config{},
        };

        experimental::WorkUnitSpec main_wu{.name = "main", .kernels = {DM_KERNEL}, .target_nodes = core};
        experimental::ProgramSpec spec{
            .name = "sem_scope_coexist",
            .kernels = {kernel_spec},
            .semaphores = {cached_sem, external_sem, done_sem},
            .work_units = {main_wu},
        };
        program = experimental::MakeProgramFromSpec(*mesh_device_, spec);

        experimental::ProgramRunArgs params;
        params.kernel_run_args = {
            experimental::ProgramRunArgs::KernelRunArgs{
                .kernel = DM_KERNEL,
                .runtime_arg_values = experimental::MakeRuntimeArgsForSingleNode(
                    core,
                    {{"report_addr", report_addr},
                     {"increment_times", concurrent_iterations},
                     {"num_threads", num_dms_}}),
            },
        };
        experimental::SetProgramRunArgs(program, params);
        workload.add_program(device_range, std::move(program));
        RunProgram(mesh_device_, workload);

        tt::tt_metal::detail::ReadFromDeviceL1(device_, core, report_addr, 4 * sizeof(uint32_t), result);
        EXPECT_EQ(result.size(), 4u);
        if (result.size() < 4) {
            return {0u, 0u};
        }
        EXPECT_EQ(result[2], static_cast<uint32_t>(SemScope::DM_LOCAL_CACHED)) << "cached_sem is not in the pool";
        EXPECT_EQ(result[3], static_cast<uint32_t>(SemScope::EXTERNAL)) << "external_sem is not on the ring";
        return {result[0], result[1]};
    }

    // Remote-up run: a SENDER kernel on second_node() bumps sem::counter on `core` purely via
    // Semaphore::up(noc, x, y, 1) while a RECEIVER kernel on `core` waits for the exact total,
    // then reports {baked scope, value()}. Two binder kernels with one off the semaphore's node:
    // the census must resolve EXTERNAL. Callers must skip-guard on has_second_node().
    std::pair<uint32_t, uint32_t> run_remote(uint32_t sender_threads, uint32_t iters) {
        const uint32_t expected = sender_threads * iters;
        // Sentinel prefill: see run_scope.
        std::vector<uint32_t> sentinel(2, kNoReport);
        tt::tt_metal::detail::WriteToDeviceL1(device_, core, report_addr, sentinel);

        // The sender addresses the semaphore's node by its virtual NoC coords.
        const CoreCoord core_virtual = mesh_device_->worker_core_from_logical_core(core);

        distributed::MeshWorkload workload;
        distributed::MeshCoordinate zero_coord{0, 0};
        distributed::MeshCoordinateRange device_range{zero_coord, zero_coord};

        experimental::SemaphoreSpec counter_sem{
            .unique_id = experimental::SemaphoreSpecName{"counter_sem"}, .target_nodes = core};

        const experimental::KernelSpecName SENDER{"sem_remote_sender"};
        const experimental::KernelSpecName RECEIVER{"sem_remote_receiver"};
        experimental::KernelSpec sender_spec{
            .unique_id = SENDER,
            .source = kernel_path_remote,
            .num_threads = sender_threads,
            .compiler_options = {.defines = {{"REMOTE_SENDER", "1"}}},
            .semaphore_bindings =
                {{.semaphore_spec_name = experimental::SemaphoreSpecName{"counter_sem"}, .accessor_name = "counter"}},
            .runtime_arg_schema = {.runtime_arg_names = {"increment_times", "remote_noc_x", "remote_noc_y"}},
            .hw_config = experimental::DataMovementGen2Config{},
        };
        experimental::KernelSpec receiver_spec{
            .unique_id = RECEIVER,
            .source = kernel_path_remote,
            .num_threads = 1,
            .semaphore_bindings =
                {{.semaphore_spec_name = experimental::SemaphoreSpecName{"counter_sem"}, .accessor_name = "counter"}},
            .runtime_arg_schema = {.runtime_arg_names = {"report_addr", "expected"}},
            .hw_config = experimental::DataMovementGen2Config{},
        };

        experimental::WorkUnitSpec wu_recv{.name = "wu_recv", .kernels = {RECEIVER}, .target_nodes = core};
        experimental::WorkUnitSpec wu_send{.name = "wu_send", .kernels = {SENDER}, .target_nodes = second_node()};
        experimental::ProgramSpec spec{
            .name = "sem_scope_remote",
            .kernels = {sender_spec, receiver_spec},
            .semaphores = {counter_sem},
            .work_units = {wu_recv, wu_send},
        };
        Program program = experimental::MakeProgramFromSpec(*mesh_device_, spec);

        experimental::ProgramRunArgs params;
        params.kernel_run_args = {
            experimental::ProgramRunArgs::KernelRunArgs{
                .kernel = SENDER,
                .runtime_arg_values = experimental::MakeRuntimeArgsForSingleNode(
                    second_node(),
                    {{"increment_times", iters},
                     {"remote_noc_x", static_cast<uint32_t>(core_virtual.x)},
                     {"remote_noc_y", static_cast<uint32_t>(core_virtual.y)}}),
            },
            experimental::ProgramRunArgs::KernelRunArgs{
                .kernel = RECEIVER,
                .runtime_arg_values = experimental::MakeRuntimeArgsForSingleNode(
                    core, {{"report_addr", report_addr}, {"expected", expected}}),
            },
        };
        experimental::SetProgramRunArgs(program, params);
        workload.add_program(device_range, std::move(program));
        RunProgram(mesh_device_, workload);

        tt::tt_metal::detail::ReadFromDeviceL1(device_, core, report_addr, 2 * sizeof(uint32_t), result);
        EXPECT_EQ(result.size(), 2u);
        if (result.size() < 2) {
            return {kNoReport, 0u};
        }
        // The receiver must have overwritten the sentinel, or the assertions pass vacuously.
        EXPECT_NE(result[0], kNoReport) << "receiver never reported";
        return {result[0], result[1]};
    }

    // ---- Census probe harness ----
    // One binding kernel in a census shape. All kernels are single-node so per-node runtime args stay
    // simple; off-node shapes set CensusKernel::node (sole kernel elsewhere, or a second kernel there).
    struct CensusKernel {
        experimental::NodeCoord node{core};
        uint32_t num_threads = 1;
        uint32_t increments = 0;  // 0 for a binder that only reads
        bool reporter = false;    // exactly one kernel should report
        // Nonzero: the reporter spins counter.wait_min(wait_min_total) before reading value().
        // Its kernel barrier only orders its OWN threads, so this read-only spin on the bound
        // semaphore (the scope's view is coherent) is how it waits out ANOTHER binder kernel's
        // increments, making multi-kernel counts exact.
        uint32_t wait_min_total = 0;
    };

    // Build + run a program with the given census shape; returns {baked_scope, counter_value}.
    // The scope is read back from the device -- the census's ACTUAL decision, since counts
    // alone look right under any correct mechanism.
    std::pair<uint32_t, uint32_t> run_census(
        const experimental::Nodes& sem_target, const std::vector<CensusKernel>& kernels) {
        // Sentinel prefill (see run_scope), at the REPORTER's node: the probe writes its local L1.
        experimental::NodeCoord reporter_node = core;
        for (const auto& k : kernels) {
            if (k.reporter) {
                reporter_node = k.node;
            }
        }
        std::vector<uint32_t> sentinel(3, kNoReport);
        tt::tt_metal::detail::WriteToDeviceL1(device_, reporter_node, report_addr, sentinel);

        distributed::MeshWorkload workload;
        distributed::MeshCoordinate zero_coord{0, 0};
        distributed::MeshCoordinateRange device_range{zero_coord, zero_coord};

        experimental::SemaphoreSpec sem{
            .unique_id = experimental::SemaphoreSpecName{"counter_sem"}, .target_nodes = sem_target};

        std::vector<experimental::KernelSpec> kernel_specs;
        experimental::ProgramRunArgs params;
        // Each kernel gets barrier slot = its index; only NUM_KERNEL_BARRIERS(2) slots exist and
        // wait_threads() does not bounds-check. Both are free here: census kernels use no DFBs
        // (whose runtime claims slots 0/1) and cached-pool seeding uses NO barrier at all
        // (claim-and-publish on the pool row itself).
        EXPECT_LE(kernels.size(), 2u) << "run_census supports at most 2 kernels (barrier slots)";
        if (kernels.size() > 2u) {
            return {kNoReport, 0u};
        }
        // WorkUnitSpecs must have DISJOINT target nodes -> group kernels by node into one WU each.
        std::vector<std::pair<experimental::NodeCoord, std::vector<experimental::KernelSpecName>>> by_node;
        for (size_t i = 0; i < kernels.size(); i++) {
            const auto& k = kernels[i];
            const experimental::KernelSpecName name{"census_k" + std::to_string(i)};
            kernel_specs.push_back(experimental::KernelSpec{
                .unique_id = name,
                .source = kernel_path_census,
                .num_threads = k.num_threads,
                .semaphore_bindings =
                    {{.semaphore_spec_name = experimental::SemaphoreSpecName{"counter_sem"},
                      .accessor_name = "counter"}},
                .runtime_arg_schema =
                    {.runtime_arg_names =
                         {"report_addr", "increment_times", "is_reporter", "barrier_idx", "wait_min_total"}},
                .hw_config = experimental::DataMovementGen2Config{},
            });
            bool placed = false;
            for (auto& [node, names] : by_node) {
                if (node == k.node) {
                    names.push_back(name);
                    placed = true;
                    break;
                }
            }
            if (!placed) {
                by_node.push_back({k.node, {name}});
            }
            params.kernel_run_args.push_back(experimental::ProgramRunArgs::KernelRunArgs{
                .kernel = name,
                .runtime_arg_values = experimental::MakeRuntimeArgsForSingleNode(
                    k.node,
                    {{"report_addr", report_addr},
                     {"increment_times", k.increments},
                     {"is_reporter", k.reporter ? 1u : 0u},
                     // Distinct barrier slot per kernel (see the safety note on the cap above).
                     {"barrier_idx", static_cast<uint32_t>(i)},
                     {"wait_min_total", k.wait_min_total}}),
            });
        }
        std::vector<experimental::WorkUnitSpec> work_units;
        work_units.reserve(by_node.size());
        for (size_t w = 0; w < by_node.size(); w++) {
            work_units.push_back(experimental::WorkUnitSpec{
                .name = "wu" + std::to_string(w), .kernels = by_node[w].second, .target_nodes = by_node[w].first});
        }

        experimental::ProgramSpec spec{
            .name = "sem_census", .kernels = kernel_specs, .semaphores = {sem}, .work_units = work_units};
        Program program = experimental::MakeProgramFromSpec(*mesh_device_, spec);
        experimental::SetProgramRunArgs(program, params);
        workload.add_program(device_range, std::move(program));
        RunProgram(mesh_device_, workload);

        tt::tt_metal::detail::ReadFromDeviceL1(device_, reporter_node, report_addr, 3 * sizeof(uint32_t), result);
        EXPECT_EQ(result.size(), 3u);
        if (result.size() < 3) {
            return {kNoReport, 0u};
        }
        census_ring_word_ = result[2];
        // Catch "the probe never reported" instead of silently reading it as LOCAL_NONATOMIC(0).
        EXPECT_NE(result[0], kNoReport) << "census probe never reported: the reporter kernel/thread did not run";
        return {result[0], result[1]};
    }

    // A node other than `core`, on whichever axis the device actually exposes.
    bool has_second_node() const {
        const auto grid = mesh_device_->compute_with_storage_grid_size();
        return grid.x >= 2 || grid.y >= 2;
    }
    experimental::NodeCoord second_node() const {
        const auto grid = mesh_device_->compute_with_storage_grid_size();
        return grid.x >= 2 ? experimental::NodeCoord{1, 0} : experimental::NodeCoord{0, 1};
    }

    static constexpr uint32_t kNoReport = 0xDEADBEEFu;  // sentinel: outside every SemScope value
    static uint32_t scope_val(SemScope s) { return static_cast<uint32_t>(s); }
    // The semaphore's RING slot as observed by the last run_census() (report word 2). For a cached
    // semaphore this must stay at the initial value, proving the count lives in the pool.
    uint32_t census_ring_word_{kNoReport};
};

// EXTERNAL: local up() is a self-targeted NoC atomic increment; value() reads via the uncached
// alias. Shape: a read-only observer kernel binds the semaphore from a SECOND node -- the
// off-node binder is what makes the census pick the NoC atomic. Single writer -> value() ==
// iterations.
TEST_F(SemScopeFixture, TestExternalScopeIncrement) {
    if (!has_second_node()) {
        GTEST_SKIP() << "needs >= 2 worker nodes: the EXTERNAL shape binds an observer kernel off-node";
    }
    const uint32_t observed = run_scope(SemScope::EXTERNAL);
    log_info(LogTest, "EXTERNAL scope value(): {} (expected {})", observed, iterations);
    EXPECT_EQ(observed, iterations)
        << "Semaphore<EXTERNAL>::up()/value() did not produce the expected single-writer count.";
}

// DM_LOCAL_CACHED: local up() is a 32-bit RISC-V AMO on the cached alias. The census picks the
// pool for any single-node semaphore whose binders are all DM kernels on its node with >= 2
// total instances (a single instance is LOCAL_NONATOMIC), so the smallest cached shape is
// 2 threads; the census probe barrier-syncs before reporting, making the count exact. Asserts
// the resolution AND the count.
TEST_F(SemScopeFixture, TestDmLocalCachedScopeIncrement) {
    if (num_dms_ < 2) {
        GTEST_SKIP() << "needs >= 2 user DMs: a 1-instance shape resolves LOCAL_NONATOMIC";
    }
    const auto [scope, count] = run_census(core, {{.num_threads = 2, .increments = iterations, .reporter = true}});
    log_info(LogTest, "DM_LOCAL_CACHED scope={} count={} (expected {})", scope, count, 2 * iterations);
    EXPECT_EQ(scope, scope_val(SemScope::DM_LOCAL_CACHED))
        << "the smallest cached-geometry shape must resolve DM_LOCAL_CACHED";
    EXPECT_EQ(count, 2 * iterations) << "Semaphore<DM_LOCAL_CACHED>::up()/value() did not produce the expected count.";
}

// LOCAL_NONATOMIC: the legacy plain L1 read-modify-write. A single-writer single-node shape is
// the census's cheap pick (pinned by TestCensusSingleWriterPicksLocal) -- the default path
// existing DFB/CCL/SDPA callers get. value() == iterations.
TEST_F(SemScopeFixture, TestLocalNonatomicScopeIncrement) {
    const uint32_t observed = run_scope(SemScope::LOCAL_NONATOMIC);
    log_info(LogTest, "LOCAL_NONATOMIC scope value(): {} (expected {})", observed, iterations);
    EXPECT_EQ(observed, iterations)
        << "Semaphore<LOCAL_NONATOMIC>::up()/value() (legacy default) did not produce the expected count.";
}

// up(N) then down(N) must leave the semaphore at 0, per scope. For EXTERNAL this exercises the
// atomic NoC decrement (INCR_GET of a negative value); for LOCAL_NONATOMIC the legacy decrement.
// (DM_LOCAL_CACHED has no single-writer shape under the census; its LR/SC decrement is covered
// by TestDmLocalCachedProducerConsumer and TestDmLocalCachedMultiConsumerDown.)
TEST_F(SemScopeFixture, TestExternalScopeUpDown) {
    if (!has_second_node()) {
        GTEST_SKIP() << "needs >= 2 worker nodes: the EXTERNAL shape binds an observer kernel off-node";
    }
    const uint32_t observed = run_scope(SemScope::EXTERNAL, /*with_down=*/true);
    log_info(LogTest, "EXTERNAL up/down value(): {} (expected 0)", observed);
    EXPECT_EQ(observed, 0u) << "Semaphore<EXTERNAL>::down() (atomic NoC decrement) did not return to 0.";
}

TEST_F(SemScopeFixture, TestLocalNonatomicScopeUpDown) {
    const uint32_t observed = run_scope(SemScope::LOCAL_NONATOMIC, /*with_down=*/true);
    log_info(LogTest, "LOCAL_NONATOMIC up/down value(): {} (expected 0)", observed);
    EXPECT_EQ(observed, 0u) << "Semaphore<LOCAL_NONATOMIC>::down() (legacy) did not return to 0.";
}

// NOTE: the scope-table dispatch is exercised by every test here: sem::counter is a plain
// constexpr id, the kernels construct via `Semaphore s(sem::counter)`, and the mechanism
// comes from the census-resolved per-kernel scope table the host bakes.

// ---- Concurrency proofs (Quasar-only): the single-writer tests above cannot tell an
// atomic path from a non-atomic one; these do. ----

// G1: all num_dms DMs concurrently up(1) a shared Semaphore. Exact num_dms*iters proves
// up() routes to the atomic mechanism (a non-atomic up() would lose updates -> less).
// EXTERNAL down() must survive a semaphore that VIOLATES the never-0xFFFFFFFF invariant (that
// value is the CAS-return sentinel): the subtract's pre-op return IS the sentinel, so the bounded
// poll inside the lock must give up instead of wedging, leaving the ret slot usable. The kernel
// set()s the word to 0xFFFFFFFF and down(1)s TWICE: completing at all proves no wedge, and the
// second down() proves the next lock acquire still works. Exact final value pins the arithmetic.
TEST_F(SemScopeFixture, TestExternalSentinelValueDoesNotWedgeDown) {
    if (!has_second_node()) {
        GTEST_SKIP() << "needs >= 2 worker nodes: the EXTERNAL shape binds an observer kernel off-node";
    }
    const uint32_t observed = run_scope(SemScope::EXTERNAL, /*with_down=*/false, /*sentinel_down=*/true);
    log_info(LogTest, "EXTERNAL sentinel-value down value(): {:#x} (expected 0xfffffffd)", observed);
    EXPECT_EQ(observed, 0xFFFFFFFDu) << "down() from the sentinel value must complete and subtract exactly";
}

TEST_F(SemScopeFixture, TestExternalConcurrentUp) {
    if (!has_second_node()) {
        GTEST_SKIP() << "needs >= 2 worker nodes: the EXTERNAL shape spans the semaphore across two nodes";
    }
    const uint32_t observed = run_concurrent(SemScope::EXTERNAL, "MODE_CONCURRENT_UP");
    const uint32_t expected = num_dms_ * concurrent_iterations;
    log_info(LogTest, "EXTERNAL concurrent up value(): {} (expected {})", observed, expected);
    EXPECT_EQ(observed, expected) << "Semaphore<EXTERNAL>::up() lost updates under concurrency (non-atomic route?).";
}

TEST_F(SemScopeFixture, TestDmLocalCachedConcurrentUp) {
    if (num_dms_ < 2) {
        GTEST_SKIP() << "needs >= 2 user DMs: a 1-instance shape resolves LOCAL_NONATOMIC, not cached";
    }
    const uint32_t observed = run_concurrent(SemScope::DM_LOCAL_CACHED, "MODE_CONCURRENT_UP");
    const uint32_t expected = num_dms_ * concurrent_iterations;
    log_info(LogTest, "DM_LOCAL_CACHED concurrent up value(): {} (expected {})", observed, expected);
    EXPECT_EQ(observed, expected) << "Semaphore<DM_LOCAL_CACHED>::up() lost updates under concurrency.";
}

// G2: (num_dms-1) producers up(1) while a SINGLE consumer down(1)s them all. Exact 0 proves
// down()'s decrement is atomic vs concurrent producer increments (a non-atomic down() loses
// producer increments -> the consumer starves and the test times out).
TEST_F(SemScopeFixture, TestExternalProducerConsumer) {
    if (num_dms_ < 2) {
        GTEST_SKIP() << "needs >= 2 user DMs";
    }
    if (!has_second_node()) {
        GTEST_SKIP() << "needs >= 2 worker nodes: the EXTERNAL shape spans the semaphore across two nodes";
    }
    const uint32_t observed = run_concurrent(SemScope::EXTERNAL, "MODE_PRODUCER_CONSUMER");
    log_info(LogTest, "EXTERNAL producer/consumer value(): {} (expected 0)", observed);
    EXPECT_EQ(observed, 0u) << "Semaphore<EXTERNAL>::down() lost a concurrent producer increment (non-atomic?).";
}

TEST_F(SemScopeFixture, TestDmLocalCachedProducerConsumer) {
    if (num_dms_ < 2) {
        GTEST_SKIP() << "needs >= 2 user DMs";
    }
    const uint32_t observed = run_concurrent(SemScope::DM_LOCAL_CACHED, "MODE_PRODUCER_CONSUMER");
    log_info(LogTest, "DM_LOCAL_CACHED producer/consumer value(): {} (expected 0)", observed);
    EXPECT_EQ(observed, 0u) << "Semaphore<DM_LOCAL_CACHED>::down() lost a concurrent producer increment.";
}

// G3: ONE producer feeds single credits while (num_dms-1) consumers CONCURRENTLY down(iters).
// Exact 0 proves the cached down()'s CAS retry loop never double-spends a credit
// (an overdraw wraps unsigned; a lost credit hangs the run).
TEST_F(SemScopeFixture, TestDmLocalCachedMultiConsumerDown) {
    if (num_dms_ < 4) {
        GTEST_SKIP() << "needs >= 4 user DMs: producer + watchdog + at least two RACING consumers";
    }
    const uint32_t observed = run_concurrent(SemScope::DM_LOCAL_CACHED, "MODE_MULTI_CONSUMER");
    log_info(LogTest, "DM_LOCAL_CACHED multi-consumer down value(): {} (expected 0)", observed);
    EXPECT_EQ(observed, 0u)
        << "Semaphore<DM_LOCAL_CACHED>::down() double-spent a credit under multi-consumer contention "
           "(the LR/SC CAS retry loop is broken -- two consumers passed the >= check on one credit).";
}

// EXTERNAL multi-consumer down(): the NoC-CAS lock serializes consumers while the producer's
// plain increments commute. Exact 0 = no credit double-spent, none lost.
TEST_F(SemScopeFixture, TestExternalMultiConsumerDown) {
    if (num_dms_ < 4) {
        GTEST_SKIP() << "needs >= 4 user DMs: producer + watchdog + at least two RACING consumers";
    }
    if (!has_second_node()) {
        GTEST_SKIP() << "needs >= 2 worker nodes: the EXTERNAL shape spans the semaphore across two nodes";
    }
    const uint32_t observed = run_concurrent(SemScope::EXTERNAL, "MODE_MULTI_CONSUMER");
    log_info(LogTest, "EXTERNAL multi-consumer down value(): {} (expected 0)", observed);
    EXPECT_EQ(observed, 0u) << "EXTERNAL multi-consumer down() double-spent or lost a credit";
}

// ---- Cached-pool / EXTERNAL coexistence ----

// A DM_LOCAL_CACHED sem (pool) + an EXTERNAL sem (ring) hammered concurrently by all DMs.
// The pool is physically disjoint from the NoC-written ring (MEM_DM_CACHED_SEM_BASE <
// MEM_MAP_END <= kernel_config ring base), so the cached AMO's 64B dirty-line write-back
// cannot clobber the ring word. Both counts exact => coexistence works.
TEST_F(SemScopeFixture, TestCachedExternalCoexistence) {
    if (num_dms_ < 2) {
        GTEST_SKIP() << "needs >= 2 user DMs: a 1-instance cached_sem shape resolves LOCAL_NONATOMIC";
    }
    if (!has_second_node()) {
        GTEST_SKIP() << "needs >= 2 worker nodes: the EXTERNAL shape spans those semaphores across two nodes";
    }
    const auto [cached_val, external_val] = run_coexist();
    const uint32_t expected = num_dms_ * concurrent_iterations;
    log_info(LogTest, "coexistence: cached={} external={} (each expected {})", cached_val, external_val, expected);
    EXPECT_EQ(cached_val, expected)
        << "DM_LOCAL_CACHED (pool) count wrong -> cached AMO lost updates or the pool was not initialised.";
    EXPECT_EQ(external_val, expected)
        << "EXTERNAL (ring) count wrong -> the cached sem's dirty-line write-back clobbered the NoC-written "
           "ring word (pool separation FAILED).";
}

// ---- Remote up() proofs: Semaphore::up(noc, x, y, v) driven from a SECOND node. ----
// The receiver wait_min()s for the expected total before reporting, so a LOST increment
// manifests as a hang; the exact-count EXPECTs catch overshoot and wrong-word landings.

// One off-node sender thread. Exact count proves remote up() reaches the semaphore's word and
// loses nothing; the scope report proves the census demoted the off-node-written semaphore to
// EXTERNAL (any local mechanism would split or lose the remote increments).
TEST_F(SemScopeFixture, TestExternalRemoteUpExactCount) {
    if (!has_second_node()) {
        GTEST_SKIP() << "needs >= 2 worker nodes for an off-node sender";
    }
    const auto [scope, value] = run_remote(/*sender_threads=*/1, iterations);
    log_info(LogTest, "EXTERNAL remote up: scope={} value={} (expected {})", scope, value, iterations);
    EXPECT_EQ(scope, scope_val(SemScope::EXTERNAL))
        << "the census must resolve an off-node-written semaphore to EXTERNAL in both kernels";
    EXPECT_EQ(value, iterations)
        << "Semaphore::up(noc, x, y, 1) from an off-node single sender overshot or hit the wrong word "
           "(a LOST increment would hang in the receiver's wait_min, not fail here).";
}

// All user-DM sender threads hammer the SAME remote word. Exact count proves the remote
// increments from independent harts stay mutually atomic through the class API.
TEST_F(SemScopeFixture, TestExternalRemoteUpConcurrentExactCount) {
    if (!has_second_node()) {
        GTEST_SKIP() << "needs >= 2 worker nodes for an off-node sender";
    }
    const auto [scope, value] = run_remote(num_dms_, concurrent_iterations);
    const uint32_t expected = num_dms_ * concurrent_iterations;
    log_info(LogTest, "EXTERNAL concurrent remote up: scope={} value={} (expected {})", scope, value, expected);
    EXPECT_EQ(scope, scope_val(SemScope::EXTERNAL))
        << "the census must resolve an off-node-written semaphore to EXTERNAL in both kernels";
    EXPECT_EQ(value, expected)
        << "concurrent Semaphore::up(noc, x, y, 1) from " << num_dms_
        << " sender threads overshot or landed on the wrong word (undercount from lost updates would "
           "hang in the receiver's wait_min before reporting).";
}

// ============================================================================
// Census stress suite.
// ============================================================================
// Each test builds a census shape and asserts the scope the host ACTUALLY baked
// (read back from the device) -- counts alone look right under any correct mechanism.
// NOTE: no "compute kernel binds a cached semaphore" shape -- Metal 2.0 forbids
// semaphore bindings on compute kernels outright (ValidateProgramSpec), so a
// binder is always a data-movement kernel.
// ============================================================================

// 1 writer instance on a single-cell semaphore -> nothing can race -> cheapest path.
TEST_F(SemScopeFixture, TestCensusSingleWriterPicksLocal) {
    const auto [scope, count] = run_census(core, {{.num_threads = 1, .increments = iterations, .reporter = true}});
    log_info(LogTest, "census single-writer: scope={} count={}", scope, count);
    EXPECT_EQ(scope, scope_val(SemScope::LOCAL_NONATOMIC)) << "single writer should take the cheap uncached path";
    EXPECT_EQ(count, iterations);
}

// Many writer instances (threads) all on the semaphore's ONE node -> the census picks the
// cached-pool AMO (asserted via the baked-scope readback), and the count is exact under
// concurrency. RESIDENCY: the count must live in the POOL, so the kernel_config RING slot must
// still hold the untouched initial value -- every other cached assertion is count-only and would
// pass even if the semaphore had silently fallen back to the ring; this is the test that can tell.
TEST_F(SemScopeFixture, TestCachedSemLivesInPoolNotRing) {
    if (num_dms_ < 2) {
        GTEST_SKIP() << "needs >= 2 user DMs";
    }
    const auto [scope, count] =
        run_census(core, {{.num_threads = num_dms_, .increments = concurrent_iterations, .reporter = true}});
    ASSERT_EQ(scope, scope_val(SemScope::DM_LOCAL_CACHED)) << "expected the cached pick for this shape";
    EXPECT_EQ(count, num_dms_ * concurrent_iterations);
    log_info(LogTest, "cached residency: count={} ring_slot={}", count, census_ring_word_);
    EXPECT_EQ(census_ring_word_, 0u)
        << "the ring slot changed (" << census_ring_word_
        << "), so the cached semaphore is NOT living in the pool -- the pool routing silently fell back "
           "to the kernel_config ring.";
}

// POSITIVE CONTROL for the residency probe: under EXTERNAL the live count IS in the ring slot,
// so report[2] must carry it (non-zero). Without this, a broken ring readback that always
// returns 0 would make TestCachedSemLivesInPoolNotRing pass vacuously. A read-only observer
// binder kernel on second_node() (0 increments) gives the semaphore off-node reach, so the
// census picks EXTERNAL (an on-node observer would stay cached now); being off-node it costs
// no writer hart, and the count stays exact (the observer never writes).
TEST_F(SemScopeFixture, TestRingResidencyProbePositiveControl) {
    if (!has_second_node()) {
        GTEST_SKIP() << "needs >= 2 worker nodes: only an off-node observer binder forces EXTERNAL";
    }
    const uint32_t writer_threads = num_dms_;
    const auto [scope, count] = run_census(
        core,
        {{.num_threads = writer_threads, .increments = concurrent_iterations, .reporter = true},
         {.node = second_node(), .num_threads = 1, .increments = 0}});
    ASSERT_EQ(scope, scope_val(SemScope::EXTERNAL)) << "an off-node binder kernel must resolve EXTERNAL";
    EXPECT_EQ(count, writer_threads * concurrent_iterations);
    log_info(LogTest, "ring residency positive control: count={} ring_slot={}", count, census_ring_word_);
    EXPECT_EQ(census_ring_word_, count)
        << "an EXTERNAL semaphore's count lives in the ring slot, but the probe's ring readback did "
           "not see it -- the residency check could pass vacuously";
}

// TWO cached semaphores, each with its own multi-threaded binder kernel, co-resident on ONE
// node: BOTH must resolve DM_LOCAL_CACHED. This shape is now a CAPABILITY, not a conflict --
// each semaphore lives in its own pool row, each row is seeded by an independent
// claim-and-publish on that row, and there is NO shared node-wide seeder barrier left for the
// two groups to mix (the old design demoted this shape to EXTERNAL because unequal thread
// counts would hang its shared rendezvous). Unequal thread counts on purpose: the exact shape
// the old rule could not cache must now complete and count exactly.
TEST_F(SemScopeFixture, TestCensusTwoCachedSemsOneNodeBothCached) {
    // threads_b = num_dms_ - 2 must be >= 2 (so BOTH kernels are multi-threaded cached shapes)
    // and != 2 (unequal counts, the old design's hanging variant). num_dms_ >= 5 guarantees both.
    if (num_dms_ < 5) {
        GTEST_SKIP() << "needs >= 5 user DMs so both kernels are multi-threaded cached shapes "
                        "with different thread counts";
    }
    std::vector<uint32_t> sentinel(3, kNoReport);
    tt::tt_metal::detail::WriteToDeviceL1(device_, core, report_addr, sentinel);

    // Two semaphores, two kernels of DIFFERENT thread counts, both on `core`, one kernel per semaphore.
    const experimental::SemaphoreSpecName SEM_A{"sem_a"};
    const experimental::SemaphoreSpecName SEM_B{"sem_b"};
    experimental::SemaphoreSpec sem_a{.unique_id = SEM_A, .target_nodes = core};
    experimental::SemaphoreSpec sem_b{.unique_id = SEM_B, .target_nodes = core};
    const experimental::KernelSpecName KA{"cached_ka"};
    const experimental::KernelSpecName KB{"cached_kb"};
    const uint32_t threads_a = 2;
    const uint32_t threads_b = num_dms_ - threads_a;  // different count -> the old design's hanging variant

    auto make_k = [&](const experimental::KernelSpecName& name,
                      const experimental::SemaphoreSpecName& sem_name,
                      uint32_t threads) {
        return experimental::KernelSpec{
            .unique_id = name,
            .source = kernel_path_census,
            .num_threads = threads,
            .semaphore_bindings = {{.semaphore_spec_name = sem_name, .accessor_name = "counter"}},
            .runtime_arg_schema =
                {.runtime_arg_names =
                     {"report_addr", "increment_times", "is_reporter", "barrier_idx", "wait_min_total"}},
            .hw_config = experimental::DataMovementGen2Config{},
        };
    };
    experimental::WorkUnitSpec wu{.name = "wu", .kernels = {KA, KB}, .target_nodes = core};
    experimental::ProgramSpec spec{
        .name = "two_cached_one_node",
        .kernels = {make_k(KA, SEM_A, threads_a), make_k(KB, SEM_B, threads_b)},
        .semaphores = {sem_a, sem_b},
        .work_units = {wu}};
    Program program = experimental::MakeProgramFromSpec(*mesh_device_, spec);

    experimental::ProgramRunArgs params;
    params.kernel_run_args = {
        experimental::ProgramRunArgs::KernelRunArgs{
            .kernel = KA,
            .runtime_arg_values = experimental::MakeRuntimeArgsForSingleNode(
                core,
                {{"report_addr", report_addr},
                 {"increment_times", concurrent_iterations},
                 {"is_reporter", 1u},
                 {"barrier_idx", 0u},
                 {"wait_min_total", 0u}}),
        },
        experimental::ProgramRunArgs::KernelRunArgs{
            .kernel = KB,
            .runtime_arg_values = experimental::MakeRuntimeArgsForSingleNode(
                core,
                {{"report_addr", report_addr},
                 {"increment_times", concurrent_iterations},
                 {"is_reporter", 0u},
                 {"barrier_idx", 1u},
                 {"wait_min_total", 0u}}),
        },
    };
    experimental::SetProgramRunArgs(program, params);
    distributed::MeshWorkload workload;
    distributed::MeshCoordinate zero_coord{0, 0};
    workload.add_program(distributed::MeshCoordinateRange{zero_coord, zero_coord}, std::move(program));
    RunProgram(mesh_device_, workload);  // must COMPLETE: the two rows' seed protocols are independent

    tt::tt_metal::detail::ReadFromDeviceL1(device_, core, report_addr, 3 * sizeof(uint32_t), result);
    ASSERT_EQ(result.size(), 3u);
    ASSERT_NE(result[0], kNoReport) << "probe never reported: the reporter kernel/thread did not run";
    log_info(LogTest, "two cached sems on one node: scope={} count={}", result[0], result[1]);
    EXPECT_EQ(result[0], scope_val(SemScope::DM_LOCAL_CACHED))
        << "two co-resident cached-binder kernels get independent pool rows -- the per-node "
           "conflict demotion is gone, so the reporter's sem must stay cached";
    EXPECT_EQ(result[1], threads_a * concurrent_iterations)
        << "sem_a's count is off -- its row's claim-and-publish seed raced or the rows collided";
}

// The headline multi-kernel cached keystone: TWO multi-threaded DM kernels on `core` both bind
// ONE shared single-node semaphore and both hammer it -- the census must keep it
// DM_LOCAL_CACHED (any number of on-node DM binder kernels is cached geometry) and the total
// must be EXACT across kernel (not just thread) boundaries, proving the row is seeded exactly
// once for ALL binder harts and the cached AMO loses nothing between kernels. Cross-kernel
// completion: kernel A's reporter cannot barrier with kernel B (each kernel syncs internally on
// its own slot, 0 and 1, as run_census hands out), so it spins counter.wait_min(total) -- a
// read-only op on the bound Semaphore, coherent on-node -- before reading value(). A LOST
// increment therefore hangs the run instead of underreporting; overshoot still fails the EXPECT.
TEST_F(SemScopeFixture, TestCachedMultiKernelExactCount) {
    // threads_b = num_dms_ - 2 must be >= 2 (both kernels multi-threaded) and != 2 (unequal
    // counts, so nothing about the kernels' internal barriers accidentally lines up).
    // num_dms_ >= 5 guarantees both.
    if (num_dms_ < 5) {
        GTEST_SKIP() << "needs >= 5 user DMs for two multi-threaded kernels with different thread counts";
    }
    const uint32_t threads_a = 2;
    const uint32_t threads_b = num_dms_ - threads_a;
    const uint32_t total = (threads_a + threads_b) * concurrent_iterations;
    const auto [scope, count] = run_census(
        core,
        {{.num_threads = threads_a, .increments = concurrent_iterations, .reporter = true, .wait_min_total = total},
         {.num_threads = threads_b, .increments = concurrent_iterations}});
    log_info(LogTest, "cached multi-kernel: scope={} count={} (expected {})", scope, count, total);
    EXPECT_EQ(scope, scope_val(SemScope::DM_LOCAL_CACHED))
        << "two on-node multi-threaded DM binder kernels sharing one single-node semaphore are "
           "cached geometry and must stay DM_LOCAL_CACHED";
    EXPECT_EQ(count, total) << "cross-kernel cached count is off: an up() overshot, or the pool row was seeded more "
                               "than once (a re-seed erases earlier increments)";
}

// Self-restore keystone: the LAST binder hart out of a program clears the pool row's protocol
// words (entered/exited/seeded), so the NEXT program's first hart re-seeds the counter from the
// ring init value. THREE back-to-back FRESH programs with the same cached shape must all resolve
// DM_LOCAL_CACHED and all count exactly. Three runs, not two: a fully missing reset already
// breaks run 2 (stale seeded -> no re-seed -> count carries run 1's total), but a PARTIAL reset
// (e.g. only `exited` left stale) leaves run 2 healthy and only breaks deterministically on run
// 3 (the restore branch never fires again, so run 3 sees stale entered/seeded -> nobody seeds
// -> count reads high or the entry spin hangs).
TEST_F(SemScopeFixture, TestCachedSelfRestoresAcrossLaunches) {
    if (num_dms_ < 2) {
        GTEST_SKIP() << "needs >= 2 user DMs: a 1-instance shape resolves LOCAL_NONATOMIC";
    }
    const uint32_t expected = num_dms_ * concurrent_iterations;
    for (uint32_t run = 0; run < 3; run++) {
        const auto [scope, count] =
            run_census(core, {{.num_threads = num_dms_, .increments = concurrent_iterations, .reporter = true}});
        log_info(LogTest, "self-restore run {}: scope={} count={} (expected {})", run, scope, count, expected);
        EXPECT_EQ(scope, scope_val(SemScope::DM_LOCAL_CACHED)) << "run " << run << " did not resolve cached";
        EXPECT_EQ(count, expected)
            << (run == 0 ? "first launch miscounted"
                         : "a later launch started from a stale pool row -- the exit-stub "
                           "self-restore did not fully reset the protocol words");
    }
}

// Cached seeding is IMMUNE to a co-resident kernel's barrier-slot choice BY CONSTRUCTION: the
// seed protocol uses no barrier at all -- each pool row is seeded once per program by
// claim-and-publish on the row itself and self-restored by the last binder hart out -- so
// there is no rendezvous KB's slot pick could ever mix with. This test pins that end-to-end:
// KA (cached sem) runs beside KB, which syncs on user array slot 1 with a DIFFERENT thread
// count and binds a two-node sem (EXTERNAL by shape -- the coexistence partner). The run must
// complete (a seed protocol regressed onto a barrier could hang KA at entry against KB's
// group), KA must stay cached, and its count must be exact.
TEST_F(SemScopeFixture, TestCachedSeederImmuneToUserBarrierSlots) {
    // Same geometry as TestCensusTwoCachedSemsOneNodeBothCached: threads_b >= 2 and != threads_a.
    if (num_dms_ < 5) {
        GTEST_SKIP() << "needs >= 5 user DMs for two multi-threaded kernels with different thread counts";
    }
    if (!has_second_node()) {
        GTEST_SKIP() << "needs >= 2 worker nodes: KB's semaphore spans two nodes to resolve EXTERNAL";
    }
    std::vector<uint32_t> sentinel(3, kNoReport);
    tt::tt_metal::detail::WriteToDeviceL1(device_, core, report_addr, sentinel);

    const experimental::SemaphoreSpecName SEM_A{"sem_cached"};
    const experimental::SemaphoreSpecName SEM_B{"sem_external"};
    experimental::SemaphoreSpec sem_a{.unique_id = SEM_A, .target_nodes = core};
    // Multi-node -> the census resolves sem_b EXTERNAL; KB neither binds nor seeds sem_a.
    experimental::SemaphoreSpec sem_b{.unique_id = SEM_B, .target_nodes = experimental::NodeRange{core, second_node()}};
    const experimental::KernelSpecName KA{"cached_binder"};
    const experimental::KernelSpecName KB{"slot1_user"};
    const uint32_t threads_a = 2;
    const uint32_t threads_b = num_dms_ - threads_a;

    auto make_k = [&](const experimental::KernelSpecName& name,
                      const experimental::SemaphoreSpecName& sem_name,
                      uint32_t threads) {
        return experimental::KernelSpec{
            .unique_id = name,
            .source = kernel_path_census,
            .num_threads = threads,
            .semaphore_bindings = {{.semaphore_spec_name = sem_name, .accessor_name = "counter"}},
            .runtime_arg_schema =
                {.runtime_arg_names =
                     {"report_addr", "increment_times", "is_reporter", "barrier_idx", "wait_min_total"}},
            .hw_config = experimental::DataMovementGen2Config{},
        };
    };
    experimental::WorkUnitSpec wu{.name = "wu", .kernels = {KA, KB}, .target_nodes = core};
    experimental::ProgramSpec spec{
        .name = "cached_seeder_vs_user_slot",
        .kernels = {make_k(KA, SEM_A, threads_a), make_k(KB, SEM_B, threads_b)},
        .semaphores = {sem_a, sem_b},
        .work_units = {wu}};
    Program program = experimental::MakeProgramFromSpec(*mesh_device_, spec);

    experimental::ProgramRunArgs params;
    params.kernel_run_args = {
        experimental::ProgramRunArgs::KernelRunArgs{
            .kernel = KA,
            .runtime_arg_values = experimental::MakeRuntimeArgsForSingleNode(
                core,
                {{"report_addr", report_addr},
                 {"increment_times", concurrent_iterations},
                 {"is_reporter", 1u},
                 {"barrier_idx", 0u},
                 {"wait_min_total", 0u}}),
        },
        experimental::ProgramRunArgs::KernelRunArgs{
            .kernel = KB,
            .runtime_arg_values = experimental::MakeRuntimeArgsForSingleNode(
                core,
                {{"report_addr", report_addr},
                 {"increment_times", concurrent_iterations},
                 {"is_reporter", 0u},
                 {"barrier_idx", 1u},  // a user array slot -- has no seeder rendezvous to collide with
                 {"wait_min_total", 0u}}),
        },
    };
    experimental::SetProgramRunArgs(program, params);
    distributed::MeshWorkload workload;
    distributed::MeshCoordinate zero_coord{0, 0};
    workload.add_program(distributed::MeshCoordinateRange{zero_coord, zero_coord}, std::move(program));
    // Must COMPLETE: a seed protocol regressed onto a kernel barrier could hang KA against KB's group.
    RunProgram(mesh_device_, workload);

    tt::tt_metal::detail::ReadFromDeviceL1(device_, core, report_addr, 3 * sizeof(uint32_t), result);
    ASSERT_EQ(result.size(), 3u);
    ASSERT_NE(result[0], kNoReport) << "probe never reported: the reporter kernel/thread did not run";
    log_info(LogTest, "cached seeding vs user barrier slot: scope={} count={}", result[0], result[1]);
    EXPECT_EQ(result[0], scope_val(SemScope::DM_LOCAL_CACHED))
        << "sem_a is single-node with every binder on-node -> cached; KB's co-residency and "
           "barrier-slot choice must not affect the pick";
    EXPECT_EQ(result[1], threads_a * concurrent_iterations)
        << "count off -> a binder hart incremented before the row's claim-and-publish seed landed";
}

// A semaphore spanning >1 node is >1 physical cell; the pool is per-core, so it must take the NoC
// atomic instead.
TEST_F(SemScopeFixture, TestCensusMultiNodeSemPicksExternal) {
    if (!has_second_node()) {
        GTEST_SKIP() << "needs >= 2 worker nodes for a multi-node semaphore";
    }
    const experimental::NodeRange two_nodes{experimental::NodeCoord{0, 0}, second_node()};
    const auto [scope, count] =
        run_census(two_nodes, {{.num_threads = num_dms_, .increments = concurrent_iterations, .reporter = true}});
    (void)count;
    log_info(LogTest, "census multi-node sem: scope={}", scope);
    EXPECT_EQ(scope, scope_val(SemScope::EXTERNAL)) << "a multi-node semaphore must not take the per-core cached pool";
}

// Node confinement in ISOLATION: a single multi-threaded binder kernel on a different node than the
// semaphore. Every other cached conjunct holds (all binders are DM kernels, multi-writer,
// single-cell sem), so only the node-confinement check can demote this to EXTERNAL. Count is not
// asserted: the binder's node has no host-initialized word for this semaphore.
TEST_F(SemScopeFixture, TestCensusOffNodeSoleBinderPicksExternal) {
    if (num_dms_ < 2) {
        GTEST_SKIP() << "needs >= 2 user DMs";
    }
    if (!has_second_node()) {
        GTEST_SKIP() << "needs >= 2 worker nodes to place the binder off the semaphore's node";
    }
    const auto [scope, count] = run_census(
        core,
        {{.node = second_node(), .num_threads = num_dms_, .increments = concurrent_iterations, .reporter = true}});
    (void)count;
    log_info(LogTest, "census off-node sole binder: scope={}", scope);
    EXPECT_EQ(scope, scope_val(SemScope::EXTERNAL))
        << "with every other cached conjunct satisfied, node confinement alone must demote to EXTERNAL";
}

// The census is conservative: bindings carry no read-only declaration, so EVERY binder is a
// possible writer -- a second binder kernel means two instances, never LOCAL_NONATOMIC. But a
// second ON-node kernel no longer forces EXTERNAL: ANY number of binder kernels/threads stays
// DM_LOCAL_CACHED as long as the semaphore is single-node and every binder is a DM kernel
// confined to its node (the node's DM cache domain is coherent, and each pool row seeds by
// claim-and-publish, so extra kernels bring no rendezvous to conflict). The variants this shape
// folds in (two 1-thread writers; a writer plus a would-be consumer) resolve cached the same
// way -- the cached AMO is multi-writer- and multi-consumer-safe. Off-node binder coverage
// lives in the OffNode tests (TestCensusOffNodeBinderBlocksCached,
// TestCensusOffNodeSoleBinderPicksExternal).
TEST_F(SemScopeFixture, TestCensusSecondBinderKernelStaysCached) {
    const auto [scope, count] = run_census(
        core, {{.num_threads = 1, .increments = iterations, .reporter = true}, {.num_threads = 1, .increments = 0}});
    log_info(LogTest, "census second binder kernel: scope={} count={}", scope, count);
    EXPECT_EQ(scope, scope_val(SemScope::DM_LOCAL_CACHED))
        << "a second ON-node DM binder kernel must stay cached: single-node sem + all binders "
           "DM kernels confined to its node is cached geometry regardless of kernel count";
    EXPECT_EQ(count, iterations);  // exact: the second kernel binds but never increments
}

// An off-node binder blocks the cached path: it would address that node's pool copy. With the
// per-node conflict rule gone, off-node reach is the ONLY demoter in this shape -- a second
// ON-node binder kernel stays cached (TestCensusSecondBinderKernelStaysCached).
TEST_F(SemScopeFixture, TestCensusOffNodeBinderBlocksCached) {
    if (num_dms_ < 2) {
        GTEST_SKIP() << "needs >= 2 user DMs";
    }
    if (!has_second_node()) {
        GTEST_SKIP() << "needs >= 2 worker nodes to place a binder off the semaphore's node";
    }
    const experimental::NodeCoord other = second_node();
    const auto [scope, count] = run_census(
        core,
        {{.num_threads = num_dms_, .increments = concurrent_iterations, .reporter = true},
         {.node = other, .num_threads = 1, .increments = 0}});
    (void)count;
    log_info(LogTest, "census off-node binder: scope={}", scope);
    EXPECT_EQ(scope, scope_val(SemScope::EXTERNAL))
        << "a binder off the semaphore's node must block the cached path (it would touch a different copy)";
}

// ---- Multi-consumer census shapes ----
// down() is multi-consumer-safe on both atomic tiers (cached CAS retry loop; EXTERNAL NoC-CAS
// lock), so shapes a multi-consumer pattern produces resolve by topology like any other and
// these tests pin the RESOLUTIONS.

// The 2-thread single-kernel shape passes cached geometry: it must resolve DM_LOCAL_CACHED
// (whose CAS down() is multi-consumer-safe).
TEST_F(SemScopeFixture, TestCensusMultiConsumerCachedShape) {
    if (num_dms_ < 2) {
        GTEST_SKIP() << "needs >= 2 user DMs";
    }
    const auto [scope, count] = run_census(core, {{.num_threads = 2, .increments = 1, .reporter = true}});
    log_info(LogTest, "census multi-consumer cached shape: scope={} count={}", scope, count);
    EXPECT_EQ(scope, scope_val(SemScope::DM_LOCAL_CACHED))
        << "a multi-instance shape confined to one DM kernel on the semaphore's node must resolve "
           "DM_LOCAL_CACHED -- a different scope here means the census misroutes the cached geometry";
    EXPECT_EQ(count, 2u) << "the cached AMO lost an update, or the pool was not initialised";
}

// Semaphore ids are unique per CORE, not per program: sem_near (on `core`) and sem_far (on
// second_node()) each get id 0, so this one kernel's scope table has ONE slot for both. Alone,
// sem_near's shape is cached-eligible (a 2-thread on-node DM binder kernel) while sem_far's
// off-node binder makes it EXTERNAL -- a divergent slot. The host must promote BOTH to EXTERNAL
// (one slot holds one mechanism; EXTERNAL is correct for every topology). A broken promotion
// pass cannot slip through quietly: the emitted per-binding tripwires would then contradict each
// other and fail this kernel's JIT build.
TEST_F(SemScopeFixture, TestIdCollisionPromotesToExternal) {
    if (!has_second_node()) {
        GTEST_SKIP() << "needs a second node for the disjoint-node id collision";
    }
    std::vector<uint32_t> sentinel(2, kNoReport);
    tt::tt_metal::detail::WriteToDeviceL1(device_, core, report_addr, sentinel);

    experimental::SemaphoreSpec sem_near{
        .unique_id = experimental::SemaphoreSpecName{"sem_near"}, .target_nodes = core};
    experimental::SemaphoreSpec sem_far{
        .unique_id = experimental::SemaphoreSpecName{"sem_far"}, .target_nodes = second_node()};

    const experimental::KernelSpecName K{"slot_probe"};
    experimental::KernelSpec ks{
        .unique_id = K,
        .source = kernel_path_slot_probe,
        .num_threads = 2,  // makes sem_near cached-eligible on its own
        .semaphore_bindings =
            {{.semaphore_spec_name = experimental::SemaphoreSpecName{"sem_near"}, .accessor_name = "near_sem"},
             {.semaphore_spec_name = experimental::SemaphoreSpecName{"sem_far"}, .accessor_name = "far_sem"}},
        .runtime_arg_schema = {.runtime_arg_names = {"report_addr"}},
        .hw_config = experimental::DataMovementGen2Config{},
    };
    experimental::WorkUnitSpec wu{.name = "wu", .kernels = {K}, .target_nodes = core};
    experimental::ProgramSpec spec{
        .name = "sem_id_collision", .kernels = {ks}, .semaphores = {sem_near, sem_far}, .work_units = {wu}};
    Program program = experimental::MakeProgramFromSpec(*mesh_device_, spec);

    experimental::ProgramRunArgs params;
    params.kernel_run_args = {
        experimental::ProgramRunArgs::KernelRunArgs{
            .kernel = K,
            .runtime_arg_values = experimental::MakeRuntimeArgsForSingleNode(core, {{"report_addr", report_addr}}),
        },
    };
    experimental::SetProgramRunArgs(program, params);
    distributed::MeshWorkload workload;
    distributed::MeshCoordinate zero_coord{0, 0};
    workload.add_program(distributed::MeshCoordinateRange{zero_coord, zero_coord}, std::move(program));
    RunProgram(mesh_device_, workload);

    tt::tt_metal::detail::ReadFromDeviceL1(device_, core, report_addr, 2 * sizeof(uint32_t), result);
    ASSERT_EQ(result.size(), 2u);
    ASSERT_NE(result[0], kNoReport) << "slot probe never reported";
    log_info(LogTest, "id-collision slot: near={} far={}", result[0], result[1]);
    EXPECT_EQ(result[0], scope_val(SemScope::EXTERNAL)) << "sem_near was not promoted off the divergent slot";
    EXPECT_EQ(result[1], scope_val(SemScope::EXTERNAL)) << "sem_far must stay EXTERNAL";
}

// A kernel may bind a given semaphore only once -- a second binding would double-count that kernel's
// instances in the census and so distort the mechanism choice.
TEST_F(SemScopeFixture, TestDoubleBindingRejected) {
    experimental::SemaphoreSpec sem{.unique_id = experimental::SemaphoreSpecName{"counter_sem"}, .target_nodes = core};
    const experimental::KernelSpecName K{"double_binder"};
    experimental::KernelSpec ks{
        .unique_id = K,
        .source = kernel_path_census,
        .num_threads = 1,
        .semaphore_bindings =
            {{.semaphore_spec_name = experimental::SemaphoreSpecName{"counter_sem"}, .accessor_name = "counter"},
             {.semaphore_spec_name = experimental::SemaphoreSpecName{"counter_sem"}, .accessor_name = "counter_again"}},
        .runtime_arg_schema =
            {.runtime_arg_names = {"report_addr", "increment_times", "is_reporter", "barrier_idx", "wait_min_total"}},
        .hw_config = experimental::DataMovementGen2Config{},
    };
    experimental::WorkUnitSpec wu{.name = "main", .kernels = {K}, .target_nodes = core};
    experimental::ProgramSpec spec{.name = "sem_double_bind", .kernels = {ks}, .semaphores = {sem}, .work_units = {wu}};
    EXPECT_ANY_THROW({
        Program program = experimental::MakeProgramFromSpec(*mesh_device_, spec);
        (void)program;
    }) << "binding the same semaphore twice in one kernel must be rejected (it double-counts the census)";
}

}  // namespace tt::tt_metal
