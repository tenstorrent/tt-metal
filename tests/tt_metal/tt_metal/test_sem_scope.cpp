// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
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
#include <tt-metalium/tensor/mesh_tensor.hpp>
#include <tt-metalium/tensor/spec/tensor_spec.hpp>
#include <tt-metalium/tensor/spec/layout/tensor_layout.hpp>
#include <tt-metalium/tensor/spec/layout/page_config.hpp>
#include <tt-metalium/tt_metal.hpp>
#include "impl/context/metal_context.hpp"
#include "common/mesh_dispatch_fixture.hpp"
#include "jit_build/jit_build_settings.hpp"  // ::SemScope, host mirror of the device enum
#include <tt-metalium/experimental/metal2_host_api/program.hpp>
#include "tt_metal/impl/dispatch/slow_dispatch.hpp"

namespace tt::tt_metal {

// Exercises the device Semaphore class end-to-end across all four mechanisms. The
// host resolves each semaphore's mechanism from a binder-topology census,
// with nothing configurable on the SemaphoreSpec, so every test builds the shape
// that makes the census pick the mechanism under test.
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
    const std::string kernel_path_compute = "tests/tt_metal/tt_metal/test_kernels/compute/test_compute_semaphore.cpp";
    const std::string kernel_path_cluster = "tests/tt_metal/tt_metal/test_kernels/compute/test_cluster_semaphore.cpp";
    const std::string kernel_path_contention =
        "tests/tt_metal/tt_metal/test_kernels/compute/test_cluster_sem_contention.cpp";
    const std::string kernel_path_cross_cluster =
        "tests/tt_metal/tt_metal/test_kernels/dataflow/test_cross_cluster_sem.cpp";
    uint32_t report_addr{0};
    uint32_t num_dms_{0};
    std::shared_ptr<distributed::MeshDevice> mesh_device_;
    std::vector<uint32_t> result;

    void SetUp() override {
        MeshDispatchFixture::SetUp();
        if (arch_ != tt::ARCH::QUASAR) {
            GTEST_SKIP() << "SemScope suite is Gen2 (Quasar) only: its DM specs set no config_1xx";
        }
        mesh_device_ = devices_[0];
        report_addr = mesh_device_->allocator()->get_base_allocator_addr(HalMemType::L1);
        num_dms_ = MetalContext::instance().hal().get_processor_types_count(HalProgrammableCoreType::TENSIX, 0);
        num_dms_ = std::min(num_dms_, 6u);  // Metal 2.0 reserves DM0/DM1
    }

    // Runs the kernel and returns the value it reported. `want` picks the shape:
    // for LOCAL_NONATOMIC the kernel is the only binder; for EXTERNAL a read-only
    // observer on second_node() is added, since any off-node binder forces EXTERNAL
    // (callers must skip when there is no second node). Cached shapes cannot be built
    // here; that coverage lives in run_concurrent and the census tests.
    uint32_t run_scope(SemScope want, bool with_down = false, bool all_ones_down = false) {
        if (want == SemScope::DM_LOCAL_CACHED) {
            ADD_FAILURE() << "run_scope cannot build a cached shape (needs >= 2 unsynchronised writer threads)";
            return 0u;
        }
        // Prefill with a sentinel, not 0: the with_down tests expect 0, so a zero prefill would
        // pass even if the kernel never reported.
        std::vector<uint32_t> sentinel(1, kNoReport);
        slow_dispatch::WriteToL1(*mesh_device_, core, report_addr, sentinel);

        distributed::MeshWorkload workload;
        Program program;
        distributed::MeshCoordinate zero_coord{0, 0};
        distributed::MeshCoordinateRange device_range{zero_coord, zero_coord};

        experimental::SemaphoreSpec counter_sem{
            .unique_id = experimental::SemaphoreSpecName{"counter_sem"},
            .target_nodes = core,
        };

        std::map<std::string, std::string> defs;
        if (all_ones_down) {
            defs.emplace("SEM_SCOPE_ALL_ONES_DOWN", "1");  // set(0xFFFFFFFF) then down(1) twice
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
            .hw_config = experimental::DataMovementHardwareConfig{},
        });
        std::vector<experimental::WorkUnitSpec> work_units{experimental::WorkUnitSpec{
            .name = "main",
            .kernels = {DM_KERNEL},
            .target_nodes = core,
        }};
        if (want == SemScope::EXTERNAL) {
            // The read-only observer whose off-node binding forces EXTERNAL.
            kernel_specs.push_back(experimental::KernelSpec{
                .unique_id = OBSERVER,
                .source = kernel_path_census,
                .num_threads = 1,
                .semaphore_bindings =
                    {{.semaphore_spec_name = experimental::SemaphoreSpecName{"counter_sem"},
                      .accessor_name = "counter"}},
                .runtime_arg_schema =
                    {.runtime_arg_names = {"report_addr", "increment_times", "is_reporter", "wait_min_total"}},
                .hw_config = experimental::DataMovementHardwareConfig{},
            });
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
                     {"wait_min_total", 0u}}),
            });
        }
        experimental::SetProgramRunArgs(program, params);

        workload.add_program(device_range, std::move(program));
        RunProgram(mesh_device_, workload);

        slow_dispatch::ReadFromL1(*mesh_device_, core, report_addr, sizeof(uint32_t), result);
        EXPECT_EQ(result.size(), 1u);
        EXPECT_NE(result.empty() ? kNoReport : result[0], kNoReport) << "kernel never reported";
        return result.empty() ? 0u : result[0];
    }

    // Runs sem_scope_concurrent.cpp on num_dms threads (mode_define picks its shape) and
    // returns the reporter's value() readback. On a single node the semaphores resolve to
    // DM_LOCAL_CACHED; for EXTERNAL they span a second node instead of adding a binder
    // kernel there, which would cost a DM hart (callers must skip when there is no second
    // node).
    uint32_t run_concurrent(SemScope want, const std::string& mode_define) {
        const bool has_watchdog = mode_define == "MODE_MULTI_CONSUMER";
        // Sentinel prefill
        std::vector<uint32_t> sentinel(3, kNoReport);
        slow_dispatch::WriteToL1(*mesh_device_, core, report_addr, sentinel);
        if (has_watchdog) {
            std::vector<uint32_t> wd_sentinel(1, kNoReport);
            slow_dispatch::WriteToL1(*mesh_device_, core, report_addr + 64u, wd_sentinel);
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
            .runtime_arg_schema =
                {.runtime_arg_names = {"report_addr", "increment_times", "num_threads", "self_noc_x", "self_noc_y"}},
            .hw_config = experimental::DataMovementHardwareConfig{},
        };

        experimental::WorkUnitSpec main_wu{.name = "main", .kernels = {DM_KERNEL}, .target_nodes = core};
        experimental::ProgramSpec spec{
            .name = "sem_scope_concurrent",
            .kernels = {kernel_spec},
            .semaphores = {counter_sem, done_sem},
            .work_units = {main_wu},
        };
        program = experimental::MakeProgramFromSpec(*mesh_device_, spec);

        // Its own virtual coords, for MODE_SELF_NOC_UP's self-targeted remote up.
        const CoreCoord core_virtual = mesh_device_->worker_core_from_logical_core(core);
        experimental::ProgramRunArgs params;
        params.kernel_run_args = {
            experimental::ProgramRunArgs::KernelRunArgs{
                .kernel = DM_KERNEL,
                .runtime_arg_values = experimental::MakeRuntimeArgsForSingleNode(
                    core,
                    {{"report_addr", report_addr},
                     {"increment_times", concurrent_iterations},
                     {"num_threads", num_dms_},
                     {"self_noc_x", static_cast<uint32_t>(core_virtual.x)},
                     {"self_noc_y", static_cast<uint32_t>(core_virtual.y)}}),
            },
        };
        experimental::SetProgramRunArgs(program, params);
        workload.add_program(device_range, std::move(program));
        RunProgram(mesh_device_, workload);

        slow_dispatch::ReadFromL1(*mesh_device_, core, report_addr, 3 * sizeof(uint32_t), result);
        EXPECT_EQ(result.size(), 3u);
        if (result.size() < 3) {
            return 0u;
        }
        EXPECT_NE(result[0], kNoReport) << "kernel never reported";
        EXPECT_EQ(result[1], static_cast<uint32_t>(want)) << "counter_sem baked scope != shape intent";
        EXPECT_EQ(result[2], static_cast<uint32_t>(want)) << "done_sem baked scope != shape intent";
        if (has_watchdog) {
            std::vector<uint32_t> wd;
            slow_dispatch::ReadFromL1(*mesh_device_, core, report_addr + 64u, sizeof(uint32_t), wd);
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

    // A DM_LOCAL_CACHED semaphore and an EXTERNAL semaphore hammered concurrently
    // by all DMs; returns {cached_final, external_final}, both expected num_dms*iters. The
    // shapes select the mechanisms (single-node vs spanning a second node).
    // The baked scopes are asserted.
    std::pair<uint32_t, uint32_t> run_coexist() {
        std::vector<uint32_t> zero(4, 0);
        slow_dispatch::WriteToL1(*mesh_device_, core, report_addr, zero);

        distributed::MeshWorkload workload;
        Program program;
        distributed::MeshCoordinate zero_coord{0, 0};
        distributed::MeshCoordinateRange device_range{zero_coord, zero_coord};

        const experimental::NodeRange two_nodes{core, second_node()};
        experimental::SemaphoreSpec cached_sem{
            .unique_id = experimental::SemaphoreSpecName{"cached_sem"}, .target_nodes = core};
        experimental::SemaphoreSpec external_sem{
            .unique_id = experimental::SemaphoreSpecName{"external_sem"}, .target_nodes = two_nodes};
        experimental::SemaphoreSpec done_sem{
            .unique_id = experimental::SemaphoreSpecName{"done_sem"}, .target_nodes = two_nodes};

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
            .hw_config = experimental::DataMovementHardwareConfig{},
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

        slow_dispatch::ReadFromL1(*mesh_device_, core, report_addr, 4 * sizeof(uint32_t), result);
        EXPECT_EQ(result.size(), 4u);
        if (result.size() < 4) {
            return {0u, 0u};
        }
        EXPECT_EQ(result[2], static_cast<uint32_t>(SemScope::DM_LOCAL_CACHED)) << "cached_sem is not in the pool";
        EXPECT_EQ(result[3], static_cast<uint32_t>(SemScope::EXTERNAL)) << "external_sem is not on the ring";
        return {result[0], result[1]};
    }

    // A sender kernel on second_node() bumps sem::counter via Semaphore::up(noc, x,
    // y, 1) while a receiver kernel waits for the exact total, then reports
    // {baked scope, value()}. The off-node binder makes the census resolve EXTERNAL.
    std::pair<uint32_t, uint32_t> run_remote(uint32_t sender_threads, uint32_t iters) {
        const uint32_t expected = sender_threads * iters;
        // Sentinel prefill: see run_scope.
        std::vector<uint32_t> sentinel(2, kNoReport);
        slow_dispatch::WriteToL1(*mesh_device_, core, report_addr, sentinel);

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
            .hw_config = experimental::DataMovementHardwareConfig{},
        };
        experimental::KernelSpec receiver_spec{
            .unique_id = RECEIVER,
            .source = kernel_path_remote,
            .num_threads = 1,
            .semaphore_bindings =
                {{.semaphore_spec_name = experimental::SemaphoreSpecName{"counter_sem"}, .accessor_name = "counter"}},
            .runtime_arg_schema = {.runtime_arg_names = {"report_addr", "expected"}},
            .hw_config = experimental::DataMovementHardwareConfig{},
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

        slow_dispatch::ReadFromL1(*mesh_device_, core, report_addr, 2 * sizeof(uint32_t), result);
        EXPECT_EQ(result.size(), 2u);
        if (result.size() < 2) {
            return {kNoReport, 0u};
        }
        EXPECT_NE(result[0], kNoReport) << "receiver never reported";
        return {result[0], result[1]};
    }

    struct CensusKernel {
        experimental::NodeCoord node{core};
        uint32_t num_threads = 1;
        uint32_t increments = 0;  // 0 for a binder that only reads
        bool reporter = false;    // exactly one kernel should report
        // Nonzero: the reporter spins counter.wait_min(wait_min_total) before reading value().
        // Its kernel barrier only orders its own threads, so this read-only spin is how it
        // waits out another binder kernel's increments, making multi-kernel counts exact.
        uint32_t wait_min_total = 0;
    };

    // Builds and runs a program with the given census shape; returns {baked_scope,
    // counter_value}.
    std::pair<uint32_t, uint32_t> run_census(
        const experimental::Nodes& sem_target, const std::vector<CensusKernel>& kernels) {
        // Sentinel prefill
        experimental::NodeCoord reporter_node = core;
        for (const auto& k : kernels) {
            if (k.reporter) {
                reporter_node = k.node;
            }
        }
        std::vector<uint32_t> sentinel(3, kNoReport);
        slow_dispatch::WriteToL1(*mesh_device_, reporter_node, report_addr, sentinel);

        distributed::MeshWorkload workload;
        distributed::MeshCoordinate zero_coord{0, 0};
        distributed::MeshCoordinateRange device_range{zero_coord, zero_coord};

        experimental::SemaphoreSpec sem{
            .unique_id = experimental::SemaphoreSpecName{"counter_sem"}, .target_nodes = sem_target};

        std::vector<experimental::KernelSpec> kernel_specs;
        experimental::ProgramRunArgs params;
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
                    {.runtime_arg_names = {"report_addr", "increment_times", "is_reporter", "wait_min_total"}},
                .hw_config = experimental::DataMovementHardwareConfig{},
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

        slow_dispatch::ReadFromL1(*mesh_device_, reporter_node, report_addr, 3 * sizeof(uint32_t), result);
        EXPECT_EQ(result.size(), 3u);
        if (result.size() < 3) {
            return {kNoReport, 0u};
        }
        census_ring_word_ = result[2];
        EXPECT_NE(result[0], kNoReport) << "census probe never reported: the reporter kernel/thread did not run";
        return {result[0], result[1]};
    }

    // TODO @RT: port over to the Metal 2.0 tests (metal2_host_api/integration_tests) after Quasar has been ported.
    // https://github.com/tenstorrent/tt-metal/issues/60164
    // Pattern D (`in` non-empty) also adds a DM reader and writer, moves the input tiles through DRAM and dataflow
    // buffers, and returns the output tiles in `out`.
    static constexpr uint32_t kComputeDepth = 4;  // kernel kDepth, passed as the semaphore's max_value
    static constexpr uint32_t kComputeMaxTiles = 8;
    static constexpr uint32_t kTileWords = 32 * 32 * 2 / 4;

    struct ComputeRun {
        uint32_t pattern = 0;
        uint32_t thread_sel = 0;
        uint32_t num_iters = 0;
        uint32_t nosync = 0;
        uint32_t batch = 1;
        uint32_t max_value = 0;
    };

    std::vector<uint32_t> run_compute(
        const ComputeRun& run,
        uint32_t n_report,
        const std::vector<uint32_t>& in = {},
        std::vector<uint32_t>* out = nullptr) {
        const bool dfb_datacopy = !in.empty();
        std::vector<uint32_t> zero_report(8, 0u);
        slow_dispatch::WriteToL1(*mesh_device_, core, report_addr, zero_report);

        experimental::SemaphoreSpec sem{.unique_id = experimental::SemaphoreSpecName{"sem"}, .target_nodes = core};
        sem.advanced_options.max_value = run.max_value;
        const experimental::KernelSpecName COMPUTE{"compute"};
        experimental::KernelSpec ks{
            .unique_id = COMPUTE,
            .source = kernel_path_compute,
            .num_threads = 1,
            .semaphore_bindings =
                {{.semaphore_spec_name = experimental::SemaphoreSpecName{"sem"}, .accessor_name = "sem"}},
            .compile_time_args =
                {{"pattern", run.pattern},
                 {"thread_sel", run.thread_sel},
                 {"nosync", run.nosync},
                 {"batch", run.batch}},
            .runtime_arg_schema = {.runtime_arg_names = {"num_iters", "report_addr"}},
            .hw_config = experimental::ComputeHardwareConfig{},  // double-buffered DST (pattern D: SyncHalf)
        };
        experimental::ProgramSpec spec{
            .name = "sem_compute_atomic",
            .kernels = {ks},
            .semaphores = {sem},
            .work_units = {{.name = "main", .kernels = {COMPUTE}, .target_nodes = core}},
        };
        experimental::ProgramRunArgs params;
        params.kernel_run_args = {experimental::ProgramRunArgs::KernelRunArgs{
            .kernel = COMPUTE,
            .runtime_arg_values = experimental::MakeRuntimeArgsForSingleNode(
                core, {{"num_iters", run.num_iters}, {"report_addr", report_addr}}),
        }};

        std::shared_ptr<distributed::MeshBuffer> in_buffer;
        std::shared_ptr<distributed::MeshBuffer> out_buffer;
        if (dfb_datacopy) {
            const uint32_t tile_bytes = kTileWords * 4;
            const uint32_t bytes = run.num_iters * tile_bytes;
            const distributed::ReplicatedBufferConfig dram_size{.size = bytes};
            const distributed::DeviceLocalBufferConfig dram{.page_size = bytes, .buffer_type = BufferType::DRAM};
            in_buffer = distributed::MeshBuffer::create(dram_size, dram, mesh_device_.get());
            out_buffer = distributed::MeshBuffer::create(dram_size, dram, mesh_device_.get());
            slow_dispatch::WriteToBuffer(*in_buffer, in);

            const experimental::KernelSpecName READER{"reader"};
            const experimental::KernelSpecName WRITER{"writer"};
            const experimental::DataMovementHardwareConfig dm_hw_config{
                .config_2xx = experimental::DataMovementHardwareConfig::DataMovement2XXConfig{
                    .disable_dfb_implicit_sync_for_all = true}};
            auto& compute = spec.kernels[0];
            compute.compiler_options.defines = {{"PATTERN_D_DFB", "1"}};
            // Self-loop bindings need distinct accessor names; the kernel uses dfb::mid for both sides.
            compute.dfb_bindings = {
                experimental::ConsumerOf(experimental::DFBSpecName{"in"}, "in"),
                experimental::ProducerOf(experimental::DFBSpecName{"mid"}, "mid"),
                experimental::ConsumerOf(experimental::DFBSpecName{"mid"}, "mid_in"),
                experimental::ProducerOf(experimental::DFBSpecName{"out"}, "out"),
            };
            spec.kernels.push_back(experimental::KernelSpec{
                .unique_id = READER,
                .source = "tests/tt_metal/tt_metal/test_kernels/dataflow/reader_unary_2_0.cpp",
                .num_threads = 1,
                .dfb_bindings = {experimental::ProducerOf(experimental::DFBSpecName{"in"}, "out")},
                .runtime_arg_schema = {.runtime_arg_names = {"src_addr", "bank_id", "num_tiles"}},
                .hw_config = dm_hw_config,
            });
            spec.kernels.push_back(experimental::KernelSpec{
                .unique_id = WRITER,
                .source = "tests/tt_metal/tt_metal/test_kernels/dataflow/writer_unary_2_0.cpp",
                .num_threads = 1,
                .dfb_bindings = {experimental::ConsumerOf(experimental::DFBSpecName{"out"}, "in")},
                .runtime_arg_schema = {.runtime_arg_names = {"dst_addr", "bank_id", "num_tiles"}},
                .hw_config = dm_hw_config,
            });
            for (auto [name, entries] : {std::pair{"in", 2u}, std::pair{"mid", kComputeDepth}, std::pair{"out", 2u}}) {
                spec.dataflow_buffers.push_back(experimental::DataflowBufferSpec{
                    .unique_id = experimental::DFBSpecName{name},
                    .entry_size = tile_bytes,
                    .num_entries = entries,
                    .data_format_metadata = tt::DataFormat::Float16_b,
                });
            }
            spec.work_units[0].kernels = {COMPUTE, READER, WRITER};
            params.kernel_run_args.push_back(
                {.kernel = READER,
                 .runtime_arg_values = experimental::MakeRuntimeArgsForSingleNode(
                     core, {{"src_addr", in_buffer->address()}, {"bank_id", 0u}, {"num_tiles", run.num_iters}})});
            params.kernel_run_args.push_back(
                {.kernel = WRITER,
                 .runtime_arg_values = experimental::MakeRuntimeArgsForSingleNode(
                     core, {{"dst_addr", out_buffer->address()}, {"bank_id", 0u}, {"num_tiles", run.num_iters}})});
        }
        Program program = experimental::MakeProgramFromSpec(*mesh_device_, spec);
        experimental::SetProgramRunArgs(program, params);

        distributed::MeshWorkload workload;
        workload.add_program(distributed::MeshCoordinateRange{{0, 0}, {0, 0}}, std::move(program));
        RunProgram(mesh_device_, workload);

        slow_dispatch::ReadFromL1(*mesh_device_, core, report_addr, n_report * sizeof(uint32_t), result);
        if (dfb_datacopy) {
            slow_dispatch::ReadFromBuffer(*out_buffer, *out);
        }
        return result;
    }

    // Pattern D: num_tiles distinct Float16_b tiles through the ring. Returns the report; `mismatched` is the
    // number of output tiles that differ from the input.
    std::vector<uint32_t> run_compute_datacopy(uint32_t num_tiles, uint32_t nosync, uint32_t& mismatched) {
        const auto in = datacopy_input(num_tiles);
        std::vector<uint32_t> out;
        auto report = run_compute(
            {.pattern = 3, .num_iters = num_tiles, .nosync = nosync, .max_value = kComputeDepth}, 4, in, &out);
        mismatched = mismatched_tiles(in, out);
        return report;
    }

    static std::vector<uint32_t> datacopy_input(uint32_t num_tiles) {
        // Tile t, datum k = bf16 0x4000 + (t << 7) + (k & 0x7F): exact through a datacopy, distinct per tile.
        std::vector<uint32_t> in(num_tiles * kTileWords);
        for (uint32_t t = 0; t < num_tiles; ++t) {
            for (uint32_t w = 0; w < kTileWords; ++w) {
                const uint32_t lo = 0x4000u + (t << 7) + ((2 * w) & 0x7Fu);
                const uint32_t hi = 0x4000u + (t << 7) + ((2 * w + 1) & 0x7Fu);
                in[t * kTileWords + w] = (hi << 16) | lo;
            }
        }
        return in;
    }

    // Number of output tiles that differ from the input.
    static uint32_t mismatched_tiles(const std::vector<uint32_t>& in, const std::vector<uint32_t>& out) {
        uint32_t mismatched = 0;
        for (uint32_t first = 0; first < in.size(); first += kTileWords) {
            if (!std::equal(out.begin() + first, out.begin() + first + kTileWords, in.begin() + first)) {
                ++mismatched;
            }
        }
        return mismatched;
    }

    struct ClusterRun {
        uint32_t num_neos = 2;
        uint32_t num_iters = kComputeMaxTiles;
        uint32_t active_neos = 2;
        bool closed_ring = false;
        bool nosync = false;
        uint32_t max_tiles = kComputeMaxTiles;
        uint32_t skew_spins = 2000;
        bool omit_ready_wait = false;
        bool omit_space_wait = false;
        uint32_t consumer_skew = 0;
        uint32_t depth = kComputeDepth;
        uint32_t batch_tiles = 1;
    };

    struct ClusterResult {
        std::vector<uint32_t> report;  // packed counts, then final outgoing semaphore counts
        std::vector<uint32_t> output;
    };

    ClusterResult run_cluster(const ClusterRun& run) {
        using namespace experimental;
        TT_FATAL(
            run.batch_tiles > 0 && run.depth % run.batch_tiles == 0 && run.num_iters % run.batch_tiles == 0,
            "Cluster test batches must divide both depth and iteration count");
        constexpr uint32_t tile_bytes = kTileWords * sizeof(uint32_t);
        // Mirrors the kernel: PACK on NEO n reaches only tiles n, n + num_neos, ..., so slot s of NEO n's ring
        // is ring_tile + s * num_neos + n, and output tile i (packed by the last NEO) is
        // output_tile + i * num_neos + num_neos - 1.
        const uint32_t n = run.num_neos;
        const uint32_t input_tile = n;  // padding compensates for each NEO's DFB base offset
        const uint32_t ring_tile = (input_tile + run.max_tiles + n - 1) / n * n;
        const uint32_t output_tile = ring_tile + run.depth * n;
        const uint32_t local_capacity = output_tile / n + run.max_tiles;
        const uint32_t total_entries = local_capacity * n;

        // A single-page L1 tensor reserves the full backing extent on the test core. The DFB supplies
        // tile format/address metadata to the Metal 1.0 compute APIs; its FIFO credits never circulate.
        const TensorSpec storage_spec(
            Shape{1, total_entries * kTileWords},
            TensorLayout(
                DataType::UINT32,
                PageConfig(Layout::ROW_MAJOR),
                MemoryConfig{TensorMemoryLayout::INTERLEAVED, BufferType::L1}));
        auto storage = MeshTensor::allocate_on_device(*mesh_device_, storage_spec);
        const uint32_t base = storage.address();
        const auto in = datacopy_input(run.max_tiles);
        std::vector<uint32_t> initial(total_entries * kTileWords, kNoReport);
        std::copy(in.begin(), in.end(), initial.begin() + input_tile * kTileWords);
        if (run.closed_ring) {
            for (uint32_t slot = 0; slot < run.active_neos * run.depth; ++slot) {
                const auto first = in.begin() + (slot % run.max_tiles) * kTileWords;
                const uint32_t tile = ring_tile + (slot % run.depth) * n + slot / run.depth;
                std::copy(first, first + kTileWords, initial.begin() + tile * kTileWords);
            }
        }
        slow_dispatch::WriteToL1(*mesh_device_, core, base, initial);

        const KernelSpecName compute{"compute"};
        const DFBSpecName tiles{"tiles"};
        const TensorParamName backing{"backing"};
        KernelSpec ks{
            .unique_id = compute,
            .source = kernel_path_cluster,
            .num_threads = run.num_neos,
            .dfb_bindings = {StridedConsumerOf(tiles, "tiles_in"), ProducerOf(tiles, "tiles_out")},
            .tensor_bindings = {{.tensor_parameter_name = backing, .accessor_name = "backing"}},
            .compile_time_args =
                {{"num_neos", run.num_neos},
                 {"depth", run.depth},
                 {"max_tiles", run.max_tiles},
                 {"closed_ring", run.closed_ring},
                 {"nosync", run.nosync},
                 {"skew_spins", run.skew_spins},
                 {"local_capacity", local_capacity},
                 {"omit_ready_wait", run.omit_ready_wait},
                 {"omit_space_wait", run.omit_space_wait},
                 {"consumer_skew", run.consumer_skew},
                 {"batch_tiles", run.batch_tiles}},
            .runtime_arg_schema = {.runtime_arg_names = {"num_iters", "report_addr", "active_neos"}},
            .hw_config = ComputeHardwareConfig{},
        };
        ProgramSpec spec{
            .name = "cluster_semaphore",
            .kernels = {ks},
            .dataflow_buffers =
                {{.unique_id = tiles,
                  .entry_size = tile_bytes,
                  .num_entries = total_entries,
                  .data_format_metadata = tt::DataFormat::Float16_b,
                  .borrowed_from = backing}},
            .tensor_parameters = {{.unique_id = backing, .spec = storage_spec}},
            .work_units = {{.name = "main", .kernels = {compute}, .target_nodes = core}},
        };
        Program program = MakeProgramFromSpec(*mesh_device_, spec);
        ProgramRunArgs params;
        params.kernel_run_args = {ProgramRunArgs::KernelRunArgs{
            .kernel = compute,
            .runtime_arg_values = MakeRuntimeArgsForSingleNode(
                core, {{"num_iters", run.num_iters}, {"report_addr", base}, {"active_neos", run.active_neos}}),
        }};
        params.tensor_args.emplace(backing, TensorArgument{storage});
        SetProgramRunArgs(program, params);

        distributed::MeshWorkload workload;
        workload.add_program(distributed::MeshCoordinateRange{{0, 0}, {0, 0}}, std::move(program));
        RunProgram(mesh_device_, workload);

        ClusterResult result;
        slow_dispatch::ReadFromL1(*mesh_device_, core, base, 2 * run.num_neos * sizeof(uint32_t), result.report);
        std::vector<uint32_t> l1;
        slow_dispatch::ReadFromL1(*mesh_device_, core, base, total_entries * tile_bytes, l1);
        // Gather into contiguous order: ring edge by edge, then slot; or output tile by tile.
        const uint32_t result_tiles = run.closed_ring ? run.active_neos * run.depth : run.max_tiles;
        for (uint32_t t = 0; t < result_tiles; ++t) {
            const uint32_t tile =
                run.closed_ring ? ring_tile + (t % run.depth) * n + t / run.depth : output_tile + t * n + n - 1;
            result.output.insert(
                result.output.end(), l1.begin() + tile * kTileWords, l1.begin() + (tile + 1) * kTileWords);
        }
        return result;
    }

    // Mirrors the report layout of test_cluster_sem_contention.cpp.
    static constexpr uint32_t kContentionParticipants = 16;  // 4 NEOs x 4 TRISCs
    static constexpr uint32_t kContentionRecordWords = 16;
    static constexpr uint32_t kContentionHistBuckets = 64;
    static constexpr uint32_t kContentionHistBucketTicks = 4;
    static constexpr uint32_t kContentionL1SemBaseWord = 352;
    static constexpr uint32_t kContentionL1SemStrideWords = 4;
    static constexpr uint32_t kContentionWindow = 8192;  // wall-clock ticks that each THROUGHPUT/VICTIM run lasts
    enum ContentionRecord : uint32_t {
        ACCESSES,
        START,
        END,
        MAX_SEEN,
        ZERO_GETS,
        SAMPLES,
        LAT_MIN,
        LAT_MAX,
        LAT_SUM,
        PERF_COUNT,
        PERF_TICKS
    };
    enum class ContentionMode : uint32_t { THROUGHPUT = 0, VICTIM = 1, BARRIER = 2, ACCUM = 3, SUBBANK_LOAD = 4 };
    enum class ContentionOp : uint32_t { READ = 0, POST_GET = 1 };

    struct ContentionRun {
        ContentionMode mode = ContentionMode::THROUGHPUT;
        ContentionOp op = ContentionOp::POST_GET;
        uint32_t depth = 1;     // accesses each participant issues before it uses the first result
        bool distinct = false;  // false: everyone on semaphore 0; true: participant p on semaphore p
        uint32_t active_mask = 1;
        uint32_t rounds = 0;     // BARRIER / ACCUM only
        bool use_l1 = false;     // false: cluster semaphore register; true: 16B-aligned L1 word with RISC-V amoadd.w
        uint32_t perf_ctrl = 0;  // GROUP_PERF_CTRL for each NEO's L1 client counter (l1_client_ctrl()); 0 = off
    };

    // GROUP_PERF_CTRL value: enable, subport [9:4], event [14:12] (Quasar performance counters overview, 6.5).
    static constexpr uint32_t l1_client_ctrl(uint32_t subport, uint32_t event) {
        return 1u | (subport << 4) | (event << 12);
    }

    struct ContentionResult {
        std::vector<uint32_t> records;  // kContentionRecordWords per participant
        std::vector<uint32_t> hist;
        std::vector<uint32_t> final_values;
        uint32_t at(uint32_t p, ContentionRecord field) const { return records[p * kContentionRecordWords + field]; }
    };

    // Fills one TRISC slot on every NEO before the next slot: participant p = NEO * 4 + TRISC.
    static uint32_t spread_mask(uint32_t n) {
        uint32_t mask = 0;
        for (uint32_t i = 0; i < n; ++i) {
            mask |= 1u << ((i % 4) * 4 + i / 4);
        }
        return mask;
    }
    // Fills NEO 0's four TRISCs, then NEO 1's, and so on.
    static uint32_t packed_mask(uint32_t n) { return n >= 32 ? ~0u : (1u << n) - 1; }

    ContentionResult run_contention(const ContentionRun& run) {
        using namespace experimental;
        const uint32_t record_words = kContentionParticipants * kContentionRecordWords;
        const uint32_t report_words = record_words + 1 + kContentionHistBuckets + kContentionParticipants;
        const uint32_t total_words = kContentionL1SemBaseWord + kContentionParticipants * kContentionL1SemStrideWords;
        std::vector<uint32_t> zero(total_words, 0u);
        slow_dispatch::WriteToL1(*mesh_device_, core, report_addr, zero);

        const KernelSpecName compute{"contention"};
        KernelSpec ks{
            .unique_id = compute,
            .source = kernel_path_contention,
            .num_threads = 4,
            .runtime_arg_schema =
                {.runtime_arg_names =
                     {"mode",
                      "op",
                      "depth",
                      "distinct",
                      "active_mask",
                      "window",
                      "rounds",
                      "report_addr",
                      "use_l1",
                      "perf_ctrl"}},
            .hw_config = ComputeHardwareConfig{},
        };
        ProgramSpec spec{
            .name = "cluster_sem_contention",
            .kernels = {ks},
            .work_units = {{.name = "main", .kernels = {compute}, .target_nodes = core}},
        };
        Program program = MakeProgramFromSpec(*mesh_device_, spec);
        ProgramRunArgs params;
        params.kernel_run_args = {ProgramRunArgs::KernelRunArgs{
            .kernel = compute,
            .runtime_arg_values = MakeRuntimeArgsForSingleNode(
                core,
                {{"mode", static_cast<uint32_t>(run.mode)},
                 {"op", static_cast<uint32_t>(run.op)},
                 {"depth", run.depth},
                 {"distinct", run.distinct ? 1u : 0u},
                 {"active_mask", run.active_mask},
                 {"window", kContentionWindow},
                 {"rounds", run.rounds},
                 {"report_addr", report_addr},
                 {"use_l1", run.use_l1 ? 1u : 0u},
                 {"perf_ctrl", run.perf_ctrl}}),
        }};
        SetProgramRunArgs(program, params);
        distributed::MeshWorkload workload;
        workload.add_program(distributed::MeshCoordinateRange{{0, 0}, {0, 0}}, std::move(program));
        RunProgram(mesh_device_, workload);

        std::vector<uint32_t> raw;
        slow_dispatch::ReadFromL1(*mesh_device_, core, report_addr, report_words * sizeof(uint32_t), raw);
        ContentionResult result;
        if (raw.size() != report_words) {
            ADD_FAILURE() << "contention report readback is short";
            return result;
        }
        const auto hist_begin = raw.begin() + record_words + 1;
        result.records.assign(raw.begin(), raw.begin() + record_words);
        result.hist.assign(hist_begin, hist_begin + kContentionHistBuckets);
        result.final_values.assign(hist_begin + kContentionHistBuckets, raw.end());
        return result;
    }

    struct ContentionStats {
        uint32_t active = 0;
        uint64_t accesses = 0;
        uint32_t span = 0;                 // first start to last end, wall-clock ticks
        uint32_t start_skew = 0;           // first start to last start
        double accesses_per_tick = 0;      // all active participants together
        double mean_ticks_per_access = 0;  // per participant
        double worst_ticks_per_access = 0;
        double fairness = 0;  // Jain index of per-participant access counts: 1 = equal, 1/n = one gets all
    };

    // Statistics over the participants in `mask`.
    static ContentionStats contention_stats(const ContentionResult& r, uint32_t mask) {
        ContentionStats s;
        if (mask == 0 || r.records.empty()) {
            return s;
        }
        const uint32_t ref = r.at(__builtin_ctz(mask), START);
        int64_t first_start = 0, last_start = 0, last_end = 0;
        double sum = 0, sum_sq = 0;
        for (uint32_t p = 0; p < kContentionParticipants; ++p) {
            if (((mask >> p) & 1) == 0) {
                continue;
            }
            const uint32_t accesses = r.at(p, ACCESSES);
            const int64_t start = static_cast<int32_t>(r.at(p, START) - ref);
            const int64_t end = static_cast<int32_t>(r.at(p, END) - ref);
            first_start = s.active == 0 ? start : std::min(first_start, start);
            last_start = s.active == 0 ? start : std::max(last_start, start);
            last_end = s.active == 0 ? end : std::max(last_end, end);
            const double per_access = accesses ? static_cast<double>(end - start) / accesses : 0.0;
            s.mean_ticks_per_access += per_access;
            s.worst_ticks_per_access = std::max(s.worst_ticks_per_access, per_access);
            s.accesses += accesses;
            sum += accesses;
            sum_sq += static_cast<double>(accesses) * accesses;
            ++s.active;
        }
        s.span = static_cast<uint32_t>(last_end - first_start);
        s.start_skew = static_cast<uint32_t>(last_start - first_start);
        s.accesses_per_tick = s.span ? static_cast<double>(s.accesses) / s.span : 0.0;
        s.mean_ticks_per_access /= s.active;
        s.fairness = sum_sq > 0 ? sum * sum / (s.active * sum_sq) : 0.0;
        return s;
    }

    // Smallest latency (upper edge of its bucket) that `fraction` of the victim samples do not exceed.
    static uint32_t hist_percentile(
        const std::vector<uint32_t>& hist, double fraction, uint32_t bucket_ticks = kContentionHistBucketTicks) {
        uint64_t total = 0;
        for (uint32_t c : hist) {
            total += c;
        }
        uint64_t seen = 0;
        for (uint32_t b = 0; b < hist.size(); ++b) {
            seen += hist[b];
            if (seen >= fraction * total) {
                return (b + 1) * bucket_ticks;
            }
        }
        return static_cast<uint32_t>(hist.size()) * bucket_ticks;
    }

    // Timing is only printed. These are the checks that fail a run: a lost or doubled post or get shows up as
    // a count that does not return to 0, a get that saw 0, or a count above what the participants can hold.
    static void check_contention(const ContentionRun& run, const ContentionResult& r) {
        ASSERT_EQ(r.final_values.size(), kContentionParticipants);
        const uint32_t active = __builtin_popcount(run.active_mask);
        for (uint32_t i = 0; i < kContentionParticipants; ++i) {
            uint32_t expected = 0u;
            if (run.mode == ContentionMode::BARRIER && i == 2) {
                expected = run.rounds & 0xFFFFu;
            } else if (run.mode == ContentionMode::ACCUM) {
                if (run.distinct) {
                    expected = ((run.active_mask >> i) & 1) ? (run.rounds & 0xFFFFu) : 0u;
                } else if (i == 0) {
                    expected = (active * run.rounds) & 0xFFFFu;
                }
            }
            EXPECT_EQ(r.final_values[i], expected) << "semaphore " << i << " end count";
        }
        for (uint32_t p = 0; p < kContentionParticipants; ++p) {
            const bool is_active = ((run.active_mask >> p) & 1) != 0;
            if (!is_active) {
                EXPECT_EQ(r.at(p, ACCESSES), 0u) << "inactive participant " << p << " ran";
                continue;
            }
            EXPECT_GT(r.at(p, ACCESSES), 0u) << "participant " << p << " never ran";
            EXPECT_EQ(r.at(p, ZERO_GETS), 0u) << "participant " << p << " got from a count of 0";
            if (run.mode == ContentionMode::THROUGHPUT && run.op == ContentionOp::POST_GET) {
                const uint32_t holders = run.distinct ? 1 : active;
                EXPECT_LT(r.at(p, MAX_SEEN), holders * run.depth)
                    << "participant " << p << " saw more posts than the participants can hold";
            }
        }
    }

    // Mirrors the report layout of test_cross_cluster_sem.cpp.
    static constexpr uint32_t kCrossThreads = 6;  // user DMs per cluster; home DM thread 0 is the coordinator
    static constexpr uint32_t kCrossRecordWords = 16;
    static constexpr uint32_t kCrossHomeWord = kCrossThreads * kCrossRecordWords;
    static constexpr uint32_t kCrossHistWord = 1600 / sizeof(uint32_t);
    static constexpr uint32_t kCrossHistBuckets = 56;
    static constexpr uint32_t kCrossHistBucketCycles = 16;
    static constexpr uint32_t kCrossReportWords = 2048 / sizeof(uint32_t);  // through kHistOffset + 2 * 56 words
    static constexpr uint32_t kCrossL1CounterWord = 1024 / sizeof(uint32_t);
    static constexpr uint32_t kCrossLoadDoneOffset = 1056;  // compute load report, in bytes from report_addr
    static constexpr uint32_t kCrossLoadDstOffset = 4096;   // home tile that the NoC loaders write
    static constexpr uint32_t kCrossLoadEnd = 6144;         // end of the kernel's fixed L1 area (report, src, dst)
    static constexpr uint32_t kCrossLoadMask = 0b110000;    // DM threads 4 and 5 on each node are the NoC loaders
    static constexpr uint32_t kCrossLoadTiles = 1024;       // tiles each home NEO copies in the compute load
    static constexpr uint32_t kCrossWindow = 20000;         // DM cycles that each round-trip client runs
    enum CrossRecord : uint32_t {
        X_OPS,
        X_CYCLES,
        X_LAT_MAX,
        X_MAX_SEEN,
        X_ZERO_GETS,
        X_TIMEOUT,
        X_LOAD_OPS,
        X_LAT_MIN,
        X_LAT_SUM,
        X_READ_ATTEMPTS,
    };
    enum CrossHome : uint32_t {
        X_FINAL_GS,
        X_FINAL_L1,
        X_DRAIN_CYCLES,
        X_HOME_TIMEOUT,
        X_LOAD_DONE_AT_END,
        X_REMOTE_FINAL_GS,
        X_REMOTE_FINAL_L1,
        X_FINAL_GS2,
        X_FINAL_L12,
        X_PERF_COUNT_NEO0,  // L1 client event counts of home NEOs 0..3
        X_PERF_COUNT_NEO1,
        X_PERF_COUNT_NEO2,
        X_PERF_COUNT_NEO3,
        X_PERF_NEO_TICKS,  // NEO wall-clock ticks of the counter window
        X_PERF_DM_CYCLES,  // coordinator rdcycle over the same window
    };
    enum class CrossMode : uint32_t {
        RT_GS = 0,
        RT_L1 = 1,
        POSTED_GS = 2,
        POSTED_L1 = 3,
        BARRIER_GS = 4,
        BARRIER_L1 = 5,
        BARRIER2_GS = 6,
        POSTED_GS_STATEFUL = 7,
        BARRIER2_GS_PAUSED = 8,
        BARRIER2_GS_L1REL = 9,
        BARRIER2_GS_MCAST = 10,
        BARRIER2_L1 = 11,
        MCAST_GS = 12,
        MCAST_GS_STATEFUL = 13,
        MCAST_L1 = 14,
        HANDOFF_T1_GS = 15,
        HANDOFF_T1_L1 = 16,
        HANDOFF_T2_GS = 17,
        HANDOFF_T2_L1 = 18,
        HANDOFF_T3_GS = 19,
        HANDOFF_T3_L1 = 20,
    };
    enum class CrossLoadKind : uint32_t {
        TILE_NOC = 0,
        S4_POLL = 1,
        L1_SUBBANK = 2,
        S4_WRITE = 3,
        S4_RW = 4,
    };

    struct CrossRun {
        CrossMode mode = CrossMode::RT_GS;
        uint32_t clients = 1;
        bool local_only = false;              // false: remote cluster DMs first, then home DMs; true: home DMs only
        uint32_t count = 0;                   // posts per client (POSTED) or rounds (BARRIER / HANDOFF)
        uint32_t max_remote = kCrossThreads;  // cap on remote clients; the rest run on the home cluster
        uint32_t poll_gap = 512;              // coordinator drain poll gap in DM cycles (POSTED); 0 = no drain poll
        bool trisc_load = false;              // all 4 home NEOs run a pack/unpack datacopy in L1 during the run
        bool trisc_l1_subbank = false;  // all 16 home TRISCs hammer kSubBankLoadOffset via local NEO L1 port atomics
        bool noc_load = false;          // kCrossLoadMask DMs on both nodes run the background loader
        CrossLoadKind load_kind = CrossLoadKind::TILE_NOC;
        bool home_on_second_node = false;     // true: cluster 2 (second_node) is home, cluster 1 (core) is external
        bool separate_counters = false;       // true: Producer 0 on counter 0, Producer 1 on counter 1
        uint32_t custom_home_mask = 0;        // explicit active_mask on home cluster when nonzero
        uint32_t custom_remote_mask = 0;      // explicit active_mask on external cluster when nonzero
        uint32_t custom_load_mask = 0;        // explicit loader mask on home cluster when nonzero
        uint32_t trisc_l1_mask = 0xFFFF;      // trisc_l1_subbank: which home TRISCs (NEO * 4 + TRISC) run the atomics
        std::array<uint32_t, 4> perf_ctrl{};  // per home NEO GROUP_PERF_CTRL (l1_client_ctrl()); 0 = off
    };

    struct CrossResult {
        std::vector<uint32_t> home;
        std::vector<uint32_t> remote;
        uint32_t home_mask = 0;
        uint32_t remote_mask = 0;
        uint32_t trisc_l1_ops = 0;
        uint32_t at(bool on_home, uint32_t tid, CrossRecord field) const {
            return (on_home ? home : remote)[tid * kCrossRecordWords + field];
        }
        uint32_t coordinator(CrossHome field) const { return home[kCrossHomeWord + field]; }
        std::vector<uint32_t> hist(bool on_home, uint32_t p_idx) const {
            const auto& buf = on_home ? home : remote;
            const uint32_t first = kCrossHistWord + p_idx * kCrossHistBuckets;
            return std::vector<uint32_t>(buf.begin() + first, buf.begin() + first + kCrossHistBuckets);
        }
    };

    CrossResult run_cross_cluster(const CrossRun& run) {
        using namespace experimental;
        CrossResult result;
        const bool use_custom_masks = (run.custom_home_mask != 0 || run.custom_remote_mask != 0);
        uint32_t remote_clients = 0;
        uint32_t home_clients = 0;
        if (use_custom_masks) {
            result.home_mask = run.custom_home_mask;
            result.remote_mask = run.custom_remote_mask;
            home_clients = __builtin_popcount(result.home_mask);
            remote_clients = __builtin_popcount(result.remote_mask);
        } else {
            remote_clients = run.local_only ? 0 : std::min(run.clients, run.max_remote);
            home_clients = run.clients - remote_clients;
            result.remote_mask = (1u << remote_clients) - 1;
            result.home_mask = ((1u << home_clients) - 1) << 1;  // thread 0 is the coordinator
        }
        if (home_clients >= kCrossThreads) {
            ADD_FAILURE() << "at most " << kCrossThreads - 1 << " home clients";
            return result;
        }
        const uint32_t load_mask =
            run.custom_load_mask != 0 ? run.custom_load_mask : (run.noc_load ? kCrossLoadMask : 0u);
        const uint32_t remote_load_mask = run.custom_load_mask != 0 ? 0u : load_mask;
        if ((result.home_mask & load_mask) != 0 || (result.remote_mask & remote_load_mask) != 0) {
            ADD_FAILURE() << "clients overlap the NoC loaders";
            return result;
        }
        const uint32_t total_participants = use_custom_masks ? (home_clients + remote_clients) : run.clients;
        const NodeCoord home_node = run.home_on_second_node ? second_node() : core;
        const NodeCoord peer_node = run.home_on_second_node ? core : second_node();

        std::vector<uint32_t> zero(kCrossReportWords, 0u);
        slow_dispatch::WriteToL1(*mesh_device_, home_node, report_addr, zero);
        slow_dispatch::WriteToL1(*mesh_device_, peer_node, report_addr, zero);
        const CoreCoord home_virtual = mesh_device_->worker_core_from_logical_core(home_node);
        const CoreCoord peer_virtual = mesh_device_->worker_core_from_logical_core(peer_node);

        const KernelSpecName HOME{"cross_home"};
        const KernelSpecName REMOTE{"cross_remote"};
        const auto spec_for = [&](const KernelSpecName& name) {
            return KernelSpec{
                .unique_id = name,
                .source = kernel_path_cross_cluster,
                .num_threads = kCrossThreads,
                .runtime_arg_schema = {.runtime_arg_names = {"mode",         "window",      "count",
                                                             "is_home",      "active_mask", "participants",
                                                             "report_addr",  "home_noc_x",  "home_noc_y",
                                                             "poll_gap",     "home_count",  "remote_count",
                                                             "peer_noc_x",   "peer_noc_y",  "load_mask",
                                                             "load_dst",     "load_kind",   "separate_counters",
                                                             "perf_ctrl_lo", "perf_ctrl_hi"}},
                .hw_config = DataMovementHardwareConfig{},
            };
        };
        ProgramSpec spec{
            .name = "cross_cluster_sem",
            .kernels = {spec_for(HOME), spec_for(REMOTE)},
            .work_units =
                {{.name = "wu_home", .kernels = {HOME}, .target_nodes = home_node},
                 {.name = "wu_remote", .kernels = {REMOTE}, .target_nodes = peer_node}},
        };

        // The compute load is the four-NEO chain of run_cluster() with no synchronization, so it touches no
        // cluster register while the coordinator reaches the NEO register windows (#62126). Its tiles start after
        // load_neos padding tiles. The allocator base is report_addr, so the padding covers the DM kernel's
        // fixed area below kCrossLoadEnd.
        constexpr uint32_t load_neos = 4;
        constexpr uint32_t tile_bytes = kTileWords * sizeof(uint32_t);
        static_assert(load_neos * tile_bytes >= kCrossLoadEnd);
        const uint32_t load_capacity = load_neos + 2 * kComputeMaxTiles + (load_neos - 1) * kComputeDepth;
        const TensorSpec load_spec(
            Shape{1, load_capacity * load_neos * kTileWords},
            TensorLayout(
                DataType::UINT32,
                PageConfig(Layout::ROW_MAJOR),
                MemoryConfig{TensorMemoryLayout::INTERLEAVED, BufferType::L1}));
        std::optional<MeshTensor> load_storage;
        const KernelSpecName LOAD{"compute_load"};
        const DFBSpecName load_tiles{"load_tiles"};
        const TensorParamName load_backing{"load_backing"};
        if (run.trisc_load) {
            load_storage = MeshTensor::allocate_on_device(*mesh_device_, load_spec);
            spec.kernels.push_back(KernelSpec{
                .unique_id = LOAD,
                .source = kernel_path_cluster,
                .num_threads = load_neos,
                .dfb_bindings = {StridedConsumerOf(load_tiles, "tiles_in"), ProducerOf(load_tiles, "tiles_out")},
                .tensor_bindings = {{.tensor_parameter_name = load_backing, .accessor_name = "backing"}},
                .compile_time_args =
                    {{"num_neos", load_neos},
                     {"depth", kComputeDepth},
                     {"max_tiles", kComputeMaxTiles},
                     {"closed_ring", 0u},
                     {"nosync", 1u},
                     {"skew_spins", 0u},
                     {"local_capacity", load_capacity},
                     {"omit_ready_wait", 0u},
                     {"omit_space_wait", 0u},
                     {"consumer_skew", 0u},
                     {"batch_tiles", 1u}},
                .runtime_arg_schema = {.runtime_arg_names = {"num_iters", "report_addr", "active_neos"}},
                .hw_config = ComputeHardwareConfig{},
            });
            spec.dataflow_buffers.push_back(
                {.unique_id = load_tiles,
                 .entry_size = tile_bytes,
                 .num_entries = load_capacity * load_neos,
                 .data_format_metadata = tt::DataFormat::Float16_b,
                 .borrowed_from = load_backing});
            spec.tensor_parameters.push_back({.unique_id = load_backing, .spec = load_spec});
            spec.work_units[0].kernels.push_back(LOAD);
        } else if (run.trisc_l1_subbank) {
            spec.kernels.push_back(KernelSpec{
                .unique_id = LOAD,
                .source = kernel_path_contention,
                .num_threads = load_neos,
                .runtime_arg_schema =
                    {.runtime_arg_names =
                         {"mode",
                          "op",
                          "depth",
                          "distinct",
                          "active_mask",
                          "window",
                          "rounds",
                          "report_addr",
                          "use_l1",
                          "perf_ctrl"}},
                .hw_config = ComputeHardwareConfig{},
            });
            spec.work_units[0].kernels.push_back(LOAD);
        }
        Program program = MakeProgramFromSpec(*mesh_device_, spec);
        const auto args_for = [&](NodeCoord node, bool is_home, uint32_t mask, uint32_t node_load_mask) {
            return MakeRuntimeArgsForSingleNode(
                node,
                {{"mode", static_cast<uint32_t>(run.mode)},
                 {"window", kCrossWindow},
                 {"count", run.count},
                 {"is_home", is_home ? 1u : 0u},
                 {"active_mask", mask},
                 {"participants", total_participants},
                 {"report_addr", report_addr},
                 {"home_noc_x", static_cast<uint32_t>(home_virtual.x)},
                 {"home_noc_y", static_cast<uint32_t>(home_virtual.y)},
                 {"poll_gap", run.poll_gap},
                 {"home_count", home_clients},
                 {"remote_count", remote_clients},
                 {"peer_noc_x", static_cast<uint32_t>(peer_virtual.x)},
                 {"peer_noc_y", static_cast<uint32_t>(peer_virtual.y)},
                 {"load_mask", node_load_mask},
                 {"load_dst", report_addr + kCrossLoadDstOffset},
                 {"load_kind", static_cast<uint32_t>(run.load_kind)},
                 {"separate_counters", run.separate_counters ? 1u : 0u},
                 {"perf_ctrl_lo", is_home ? (run.perf_ctrl[0] | run.perf_ctrl[1] << 16) : 0u},
                 {"perf_ctrl_hi", is_home ? (run.perf_ctrl[2] | run.perf_ctrl[3] << 16) : 0u}});
        };
        ProgramRunArgs params;
        params.kernel_run_args = {
            ProgramRunArgs::KernelRunArgs{
                .kernel = HOME, .runtime_arg_values = args_for(home_node, true, result.home_mask, load_mask)},
            ProgramRunArgs::KernelRunArgs{
                .kernel = REMOTE,
                .runtime_arg_values = args_for(peer_node, false, result.remote_mask, remote_load_mask)},
        };
        if (run.trisc_load) {
            params.kernel_run_args.push_back(ProgramRunArgs::KernelRunArgs{
                .kernel = LOAD,
                .runtime_arg_values = MakeRuntimeArgsForSingleNode(
                    home_node,
                    {{"num_iters", kCrossLoadTiles},
                     {"report_addr", report_addr + kCrossLoadDoneOffset},
                     {"active_neos", load_neos}}),
            });
            params.tensor_args.emplace(load_backing, TensorArgument{*load_storage});
        } else if (run.trisc_l1_subbank) {
            params.kernel_run_args.push_back(ProgramRunArgs::KernelRunArgs{
                .kernel = LOAD,
                .runtime_arg_values = MakeRuntimeArgsForSingleNode(
                    home_node,
                    {{"mode", static_cast<uint32_t>(ContentionMode::SUBBANK_LOAD)},
                     {"op", static_cast<uint32_t>(ContentionOp::POST_GET)},
                     {"depth", 4u},
                     {"distinct", 0u},
                     {"active_mask", run.trisc_l1_mask},
                     {"window", 0u},
                     {"rounds", 0u},
                     {"report_addr", report_addr},
                     {"use_l1", 1u},
                     {"perf_ctrl", 0u}}),
            });
        }
        SetProgramRunArgs(program, params);
        distributed::MeshWorkload workload;
        workload.add_program(distributed::MeshCoordinateRange{{0, 0}, {0, 0}}, std::move(program));
        RunProgram(mesh_device_, workload);

        const uint32_t bytes = kCrossReportWords * sizeof(uint32_t);
        slow_dispatch::ReadFromL1(*mesh_device_, home_node, report_addr, bytes, result.home);
        slow_dispatch::ReadFromL1(*mesh_device_, peer_node, report_addr, bytes, result.remote);
        if (run.trisc_l1_subbank) {
            std::vector<uint32_t> load_done_words;
            slow_dispatch::ReadFromL1(
                *mesh_device_, home_node, report_addr + kCrossLoadDoneOffset, 2 * sizeof(uint32_t), load_done_words);
            if (load_done_words.size() == 2) {
                result.trisc_l1_ops = load_done_words[1];
            }
        }
        if (result.home.size() * sizeof(uint32_t) != bytes || result.remote.size() * sizeof(uint32_t) != bytes) {
            ADD_FAILURE() << "cross-cluster report readback is short";
            result.home.assign(bytes / sizeof(uint32_t), 0u);
            result.remote.assign(bytes / sizeof(uint32_t), 0u);
        }
        return result;
    }

    // Timing is only printed. These checks fail a run: a timeout, a lost or doubled update, a client that did not
    // run, or a GET that saw 0.
    static void check_cross_cluster(const CrossRun& run, const CrossResult& r) {
        EXPECT_EQ(r.coordinator(X_HOME_TIMEOUT), 0u) << "coordinator timed out";
        uint32_t expected_gs = 0;
        uint32_t expected_l1 = 0;
        uint32_t expected_gs2 = 0;
        uint32_t expected_l12 = 0;
        uint32_t expected_remote_gs = 0;
        uint32_t expected_remote_l1 = 0;
        switch (run.mode) {
            case CrossMode::POSTED_GS:
            case CrossMode::POSTED_GS_STATEFUL:
            case CrossMode::BARRIER_GS: expected_gs = run.clients * run.count; break;
            case CrossMode::POSTED_L1:
            case CrossMode::BARRIER_L1: expected_l1 = run.clients * run.count; break;
            case CrossMode::MCAST_GS:
            case CrossMode::MCAST_GS_STATEFUL:
                expected_gs = run.clients * run.count;
                expected_remote_gs = expected_gs;
                break;
            case CrossMode::MCAST_L1:
                expected_l1 = run.clients * run.count;
                expected_remote_l1 = expected_l1;
                break;
            case CrossMode::BARRIER2_GS:  // one arrival per cluster with clients, per round
            case CrossMode::BARRIER2_GS_PAUSED:
            case CrossMode::BARRIER2_GS_L1REL:
            case CrossMode::BARRIER2_GS_MCAST:
                expected_gs = run.count * ((r.home_mask != 0) + (r.remote_mask != 0));
                break;
            case CrossMode::BARRIER2_L1: expected_l1 = run.count * ((r.home_mask != 0) + (r.remote_mask != 0)); break;
            case CrossMode::HANDOFF_T1_GS:
            case CrossMode::HANDOFF_T2_GS:
            case CrossMode::HANDOFF_T3_GS:
                if (run.clients == 2) {
                    if (run.separate_counters) {
                        expected_gs = run.count;
                        expected_gs2 = run.count;
                    } else {
                        expected_gs = run.count * 0x0101u;
                    }
                } else {
                    expected_gs = run.count;
                }
                break;
            case CrossMode::HANDOFF_T1_L1:
            case CrossMode::HANDOFF_T2_L1:
            case CrossMode::HANDOFF_T3_L1:
                if (run.clients == 2) {
                    if (run.separate_counters) {
                        expected_l1 = run.count;
                        expected_l12 = run.count;
                    } else {
                        expected_l1 = run.count * 0x0101u;
                    }
                } else {
                    expected_l1 = run.count;
                }
                break;
            default: break;
        }
        EXPECT_EQ(r.coordinator(X_FINAL_GS), expected_gs) << "semaphore end count";
        EXPECT_EQ(r.coordinator(X_FINAL_L1), expected_l1) << "L1 counter end count";
        EXPECT_EQ(r.coordinator(X_FINAL_GS2), expected_gs2) << "semaphore 2 end count";
        EXPECT_EQ(r.coordinator(X_FINAL_L12), expected_l12) << "L1 counter 2 end count";
        EXPECT_EQ(r.coordinator(X_REMOTE_FINAL_GS), expected_remote_gs) << "remote semaphore end count";
        EXPECT_EQ(r.coordinator(X_REMOTE_FINAL_L1), expected_remote_l1) << "remote L1 counter end count";
        for (const bool on_home : {true, false}) {
            const uint32_t mask = on_home ? r.home_mask : r.remote_mask;
            for (uint32_t tid = on_home ? 1 : 0; tid < kCrossThreads; ++tid) {
                const bool active = ((mask >> tid) & 1) != 0;
                if (!active) {
                    EXPECT_EQ(r.at(on_home, tid, X_OPS), 0u) << "inactive client ran, home=" << on_home << " t" << tid;
                    continue;
                }
                EXPECT_EQ(r.at(on_home, tid, X_TIMEOUT), 0u) << "client timed out, home=" << on_home << " t" << tid;
                EXPECT_GT(r.at(on_home, tid, X_OPS), 0u) << "client never ran, home=" << on_home << " t" << tid;
                EXPECT_EQ(r.at(on_home, tid, X_ZERO_GETS), 0u) << "GET saw 0, home=" << on_home << " t" << tid;
                if (run.mode == CrossMode::RT_GS) {
                    EXPECT_LT(r.at(on_home, tid, X_MAX_SEEN), run.clients) << "count above the clients' posts";
                }
            }
        }
    }

    struct CrossStats {
        double mean_cycles_per_op = 0;
        double worst_cycles_per_op = 0;
        double ops_per_kcycle = 0;  // all clients together, assuming their windows overlap
        double fairness = 0;        // Jain index of per-client op counts
        uint32_t lat_max = 0;       // slowest single op pair (RT) seen by any client
    };

    static CrossStats cross_stats(const CrossResult& r) {
        CrossStats s;
        uint32_t n = 0;
        double sum = 0, sum_sq = 0;
        for (const bool on_home : {true, false}) {
            const uint32_t mask = on_home ? r.home_mask : r.remote_mask;
            for (uint32_t tid = 0; tid < kCrossThreads; ++tid) {
                if (((mask >> tid) & 1) == 0) {
                    continue;
                }
                const double ops = r.at(on_home, tid, X_OPS);
                const double cycles = r.at(on_home, tid, X_CYCLES);
                const double per_op = ops > 0 ? cycles / ops : 0.0;
                s.mean_cycles_per_op += per_op;
                s.worst_cycles_per_op = std::max(s.worst_cycles_per_op, per_op);
                s.ops_per_kcycle += cycles > 0 ? 1000.0 * ops / cycles : 0.0;
                s.lat_max = std::max(s.lat_max, r.at(on_home, tid, X_LAT_MAX));
                sum += ops;
                sum_sq += ops * ops;
                ++n;
            }
        }
        if (n > 0) {
            s.mean_cycles_per_op /= n;
            s.fairness = sum_sq > 0 ? sum * sum / (n * sum_sq) : 0.0;
        }
        return s;
    }

    // TestCrossClusterSemLoadCounters with the tile-copy load: all four home NEOs run the unsynchronized
    // pack/unpack datacopy of test_cluster_semaphore.cpp while one remote client does its round trips. With
    // counters_on, the coordinator arms each home NEO's L1 client counter (TRISC 0's port, FLEX_STALL_CARRY).
    void run_tile_copy_load(bool counters_on) {
        const uint32_t ctrl = counters_on ? l1_client_ctrl(0 /*subport*/, 4 /*event*/) : 0u;
        for (const auto mode : {CrossMode::RT_GS, CrossMode::RT_L1}) {
            const char* counter = mode == CrossMode::RT_GS ? "rt_sem" : "rt_l1";
            CrossRun run{
                .mode = mode,
                .clients = 1,
                .count = 16,
                .max_remote = 4,
                .trisc_load = true,
                .load_kind = CrossLoadKind::TILE_NOC,
                .perf_ctrl = {ctrl, ctrl, ctrl, ctrl}};
            SCOPED_TRACE(std::string(counter) + (counters_on ? " counters on" : " counters off"));
            const auto r = run_cross_cluster(run);
            check_cross_cluster(run, r);
            EXPECT_EQ(r.coordinator(X_LOAD_DONE_AT_END), 0u)
                << "the compute load finished before the clients; raise kCrossLoadTiles";
            if (ctrl != 0) {
                EXPECT_GT(r.coordinator(X_PERF_NEO_TICKS), 0u) << "the coordinator did not report a counter window";
            }
            log_info(
                LogTest,
                "tile_copy_load,{},{},{:.1f}",
                counter,
                counters_on ? "counters_on" : "counters_off",
                cross_stats(r).mean_cycles_per_op);
        }
    }

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
    uint32_t census_ring_word_{kNoReport};
};

// A read-only observer binds the semaphore from a second node, forcing the NoC path; a
// single writer's value() must equal iterations.
TEST_F(SemScopeFixture, TestExternalScopeIncrement) {
    if (!has_second_node()) {
        GTEST_SKIP() << "needs >= 2 worker nodes: the EXTERNAL shape binds an observer kernel off-node";
    }
    const uint32_t observed = run_scope(SemScope::EXTERNAL);
    log_info(LogTest, "EXTERNAL scope value(): {} (expected {})", observed, iterations);
    EXPECT_EQ(observed, iterations) << "EXTERNAL up()/value() did not produce the expected single-writer count.";
}

// The smallest cached shape: one 2-thread on-node binder kernel (a single instance would
// resolve LOCAL_NONATOMIC). The probe barrier-syncs before reporting, so both the resolution
// and the exact count are asserted.
TEST_F(SemScopeFixture, TestDmLocalCachedScopeIncrement) {
    if (num_dms_ < 2) {
        GTEST_SKIP() << "needs >= 2 user DMs: a 1-instance shape resolves LOCAL_NONATOMIC";
    }
    const auto [scope, count] = run_census(core, {{.num_threads = 2, .increments = iterations, .reporter = true}});
    log_info(LogTest, "DM_LOCAL_CACHED scope={} count={} (expected {})", scope, count, 2 * iterations);
    EXPECT_EQ(scope, scope_val(SemScope::DM_LOCAL_CACHED))
        << "the smallest cached-geometry shape must resolve DM_LOCAL_CACHED";
    EXPECT_EQ(count, 2 * iterations) << "DM_LOCAL_CACHED up()/value() did not produce the expected count.";
}

// A single-writer single-node shape is the census's cheap pick. value() must equal iterations.
TEST_F(SemScopeFixture, TestLocalNonatomicScopeIncrement) {
    const uint32_t observed = run_scope(SemScope::LOCAL_NONATOMIC);
    log_info(LogTest, "LOCAL_NONATOMIC scope value(): {} (expected {})", observed, iterations);
    EXPECT_EQ(observed, iterations)
        << "LOCAL_NONATOMIC up()/value() (legacy default) did not produce the expected count.";
}

// up(N) then down(N) must leave the semaphore at 0, per scope. DM_LOCAL_CACHED has no
// single-writer shape under the census; its decrement is covered by the producer/consumer
// and multi-consumer tests.
TEST_F(SemScopeFixture, TestExternalScopeUpDown) {
    if (!has_second_node()) {
        GTEST_SKIP() << "needs >= 2 worker nodes: the EXTERNAL shape binds an observer kernel off-node";
    }
    const uint32_t observed = run_scope(SemScope::EXTERNAL, /*with_down=*/true);
    log_info(LogTest, "EXTERNAL up/down value(): {} (expected 0)", observed);
    EXPECT_EQ(observed, 0u) << "EXTERNAL down() (atomic NoC decrement) did not return to 0.";
}

TEST_F(SemScopeFixture, TestLocalNonatomicScopeUpDown) {
    const uint32_t observed = run_scope(SemScope::LOCAL_NONATOMIC, /*with_down=*/true);
    log_info(LogTest, "LOCAL_NONATOMIC up/down value(): {} (expected 0)", observed);
    EXPECT_EQ(observed, 0u) << "LOCAL_NONATOMIC down() (legacy) did not return to 0.";
}

TEST_F(SemScopeFixture, TestExternalDownFromAllOnes) {
    if (!has_second_node()) {
        GTEST_SKIP() << "needs >= 2 worker nodes: the EXTERNAL shape binds an observer kernel off-node";
    }
    const uint32_t observed = run_scope(SemScope::EXTERNAL, /*with_down=*/false, /*all_ones_down=*/true);
    log_info(LogTest, "EXTERNAL down() from 0xFFFFFFFF, value(): {:#x} (expected 0xfffffffd)", observed);
    EXPECT_EQ(observed, 0xFFFFFFFDu) << "down() from 0xFFFFFFFF must subtract exactly";
}

// All num_dms DMs concurrently up(1) a shared Semaphore. An exact num_dms*iters count proves
// up() routes to the atomic mechanism.
TEST_F(SemScopeFixture, TestExternalConcurrentUp) {
    if (!has_second_node()) {
        GTEST_SKIP() << "needs >= 2 worker nodes: the EXTERNAL shape spans the semaphore across two nodes";
    }
    const uint32_t observed = run_concurrent(SemScope::EXTERNAL, "MODE_CONCURRENT_UP");
    const uint32_t expected = num_dms_ * concurrent_iterations;
    log_info(LogTest, "EXTERNAL concurrent up value(): {} (expected {})", observed, expected);
    EXPECT_EQ(observed, expected) << "EXTERNAL up() lost updates under concurrency (non-atomic route?).";
}

TEST_F(SemScopeFixture, TestDmLocalCachedConcurrentUp) {
    if (num_dms_ < 2) {
        GTEST_SKIP() << "needs >= 2 user DMs: a 1-instance shape resolves LOCAL_NONATOMIC, not cached";
    }
    const uint32_t observed = run_concurrent(SemScope::DM_LOCAL_CACHED, "MODE_CONCURRENT_UP");
    const uint32_t expected = num_dms_ * concurrent_iterations;
    log_info(LogTest, "DM_LOCAL_CACHED concurrent up value(): {} (expected {})", observed, expected);
    EXPECT_EQ(observed, expected) << "DM_LOCAL_CACHED up() lost updates under concurrency.";
}

// The current WH/BH pattern, up(noc, my_x, my_y, 1) on a semaphore the census resolves to
// DM_LOCAL_CACHED: the class must serve it with the local AMO (a NoC atomic would corrupt
// the cached pool) and keep exact counts.
TEST_F(SemScopeFixture, TestDmLocalCachedSelfNocUp) {
    if (num_dms_ < 2) {
        GTEST_SKIP() << "needs >= 2 user DMs: a 1-instance shape resolves LOCAL_NONATOMIC, not cached";
    }
    const uint32_t observed = run_concurrent(SemScope::DM_LOCAL_CACHED, "MODE_SELF_NOC_UP");
    const uint32_t expected = num_dms_ * concurrent_iterations;
    log_info(LogTest, "DM_LOCAL_CACHED self-noc up value(): {} (expected {})", observed, expected);
    EXPECT_EQ(observed, expected)
        << "up(noc, my_x, my_y, 1) on a cached semaphore lost updates, the AMO redirect is broken.";
}

// Same pattern on an EXTERNAL semaphore must still take the NoC atomic path and stay exact.
TEST_F(SemScopeFixture, TestExternalSelfNocUp) {
    if (!has_second_node()) {
        GTEST_SKIP() << "needs >= 2 worker nodes: the EXTERNAL shape spans the semaphore across two nodes";
    }
    const uint32_t observed = run_concurrent(SemScope::EXTERNAL, "MODE_SELF_NOC_UP");
    const uint32_t expected = num_dms_ * concurrent_iterations;
    log_info(LogTest, "EXTERNAL self-noc up value(): {} (expected {})", observed, expected);
    EXPECT_EQ(observed, expected) << "up(noc, my_x, my_y, 1) on an EXTERNAL semaphore lost updates.";
}

// (num_dms-1) producers up(1) while a single consumer drains them all. The consumer polls
// under a spin cap, so lost increments end in a nonzero final count instead of a hang.
TEST_F(SemScopeFixture, TestExternalProducerConsumer) {
    if (num_dms_ < 2) {
        GTEST_SKIP() << "needs >= 2 user DMs";
    }
    if (!has_second_node()) {
        GTEST_SKIP() << "needs >= 2 worker nodes: the EXTERNAL shape spans the semaphore across two nodes";
    }
    const uint32_t observed = run_concurrent(SemScope::EXTERNAL, "MODE_PRODUCER_CONSUMER");
    log_info(LogTest, "EXTERNAL producer/consumer value(): {} (expected 0)", observed);
    EXPECT_EQ(observed, 0u) << "EXTERNAL down() lost a concurrent producer increment (non-atomic?).";
}

TEST_F(SemScopeFixture, TestDmLocalCachedProducerConsumer) {
    if (num_dms_ < 2) {
        GTEST_SKIP() << "needs >= 2 user DMs";
    }
    const uint32_t observed = run_concurrent(SemScope::DM_LOCAL_CACHED, "MODE_PRODUCER_CONSUMER");
    log_info(LogTest, "DM_LOCAL_CACHED producer/consumer value(): {} (expected 0)", observed);
    EXPECT_EQ(observed, 0u) << "DM_LOCAL_CACHED down() lost a concurrent producer increment.";
}

// One producer feeds single credits while (num_dms-2) consumers concurrently down() them
// under a watchdog. Exact 0 proves the cached down() never double-spends a credit; a lost
// credit hangs the run.
TEST_F(SemScopeFixture, TestDmLocalCachedMultiConsumerDown) {
    if (num_dms_ < 4) {
        GTEST_SKIP() << "needs >= 4 user DMs: producer + watchdog + at least two RACING consumers";
    }
    const uint32_t observed = run_concurrent(SemScope::DM_LOCAL_CACHED, "MODE_MULTI_CONSUMER");
    log_info(LogTest, "DM_LOCAL_CACHED multi-consumer down value(): {} (expected 0)", observed);
    EXPECT_EQ(observed, 0u) << "DM_LOCAL_CACHED down() double-spent a credit under multi-consumer contention "
                               "(the LR/SC CAS retry loop is broken, two consumers passed the >= check on one credit).";
}

// The same multi-consumer shape on EXTERNAL. Exact 0 means no credit was double-spent or lost.
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

// A cached and an EXTERNAL semaphore hammered concurrently by all DMs. Both
// counts exact proves a cached dirty-line write-back cannot clobber the NoC-written ring word.
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

// One off-node sender thread. Exact count proves remote up() reaches the semaphore's word and
// loses nothing; the scope report proves the census demoted the off-node-written semaphore to
// EXTERNAL.
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

// All user-DM sender threads hammer the same remote word. Exact count proves remote
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

// One writer instance on a single-cell semaphore: nothing can race, so the census picks the
// cheapest path.
TEST_F(SemScopeFixture, TestCensusSingleWriterPicksLocal) {
    const auto [scope, count] = run_census(core, {{.num_threads = 1, .increments = iterations, .reporter = true}});
    log_info(LogTest, "census single-writer: scope={} count={}", scope, count);
    EXPECT_EQ(scope, scope_val(SemScope::LOCAL_NONATOMIC)) << "single writer should take the cheap uncached path";
    EXPECT_EQ(count, iterations);
}

// A cached semaphore's count must live in the pool, so the kernel_config ring slot must still
// hold the untouched initial value.
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
        << "), so the cached semaphore is NOT living in the pool, the pool routing silently fell back "
           "to the kernel_config ring.";
}

// Under EXTERNAL the live count is in the ring slot, so report[2] must carry it. The off-node read-only
// observer forces EXTERNAL without costing a writer hart, so the count stays exact.
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
           "not see it, the residency check could pass vacuously";
}

// Two cached semaphores, each with its own multi-threaded binder kernel, co-resident on one
// node: both must resolve DM_LOCAL_CACHED, complete, and count exactly.
TEST_F(SemScopeFixture, TestCensusTwoCachedSemsOneNodeBothCached) {
    if (num_dms_ < 5) {
        GTEST_SKIP() << "needs >= 5 user DMs so both kernels are multi-threaded cached shapes "
                        "with different thread counts";
    }
    std::vector<uint32_t> sentinel(3, kNoReport);
    slow_dispatch::WriteToL1(*mesh_device_, core, report_addr, sentinel);

    const experimental::SemaphoreSpecName SEM_A{"sem_a"};
    const experimental::SemaphoreSpecName SEM_B{"sem_b"};
    experimental::SemaphoreSpec sem_a{.unique_id = SEM_A, .target_nodes = core};
    experimental::SemaphoreSpec sem_b{.unique_id = SEM_B, .target_nodes = core};
    const experimental::KernelSpecName KA{"cached_ka"};
    const experimental::KernelSpecName KB{"cached_kb"};
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
                {.runtime_arg_names = {"report_addr", "increment_times", "is_reporter", "wait_min_total"}},
            .hw_config = experimental::DataMovementHardwareConfig{},
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
                 {"wait_min_total", 0u}}),
        },
        experimental::ProgramRunArgs::KernelRunArgs{
            .kernel = KB,
            .runtime_arg_values = experimental::MakeRuntimeArgsForSingleNode(
                core,
                {{"report_addr", report_addr},
                 {"increment_times", concurrent_iterations},
                 {"is_reporter", 0u},
                 {"wait_min_total", 0u}}),
        },
    };
    experimental::SetProgramRunArgs(program, params);
    distributed::MeshWorkload workload;
    distributed::MeshCoordinate zero_coord{0, 0};
    workload.add_program(distributed::MeshCoordinateRange{zero_coord, zero_coord}, std::move(program));
    RunProgram(mesh_device_, workload);

    slow_dispatch::ReadFromL1(*mesh_device_, core, report_addr, 3 * sizeof(uint32_t), result);
    ASSERT_EQ(result.size(), 3u);
    ASSERT_NE(result[0], kNoReport) << "probe never reported: the reporter kernel/thread did not run";
    log_info(LogTest, "two cached sems on one node: scope={} count={}", result[0], result[1]);
    EXPECT_EQ(result[0], scope_val(SemScope::DM_LOCAL_CACHED))
        << "two co-resident cached-binder kernels get independent pool rows, the per-node "
           "conflict demotion is gone, so the reporter's sem must stay cached";
    EXPECT_EQ(result[1], threads_a * concurrent_iterations)
        << "sem_a's count is off, its row's claim-and-publish seed raced or the rows collided";
}

// Two multi-threaded DM kernels both bind and hammer one shared single-node semaphore.
// It must stay DM_LOCAL_CACHED and the total must be exact across kernel (not just thread)
// boundaries, proving the pool row is seeded exactly once for all binder harts.
TEST_F(SemScopeFixture, TestCachedMultiKernelExactCount) {
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

// The last binder hart out of a program resets the pool row so the next program re-seeds it.
// Three back-to-back fresh programs with the same cached shape must all resolve cached and
// count exactly.
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
                         : "a later launch started from a stale pool row, the pool exit "
                           "self-restore did not fully reset the protocol words");
    }
}

// Making sure cached seeding needs no barrier at all by running a user kernel that
// rendezvouses on its own barrier, which the cached seeder never took part in.
// Firmware hands the two kernels separate slots, so the user barrier cannot touch
// whatever the seeder relies on.
TEST_F(SemScopeFixture, TestCachedSeederImmuneToUserBarrierSlots) {
    if (num_dms_ < 5) {
        GTEST_SKIP() << "needs >= 5 user DMs for two multi-threaded kernels with different thread counts";
    }
    if (!has_second_node()) {
        GTEST_SKIP() << "needs >= 2 worker nodes: KB's semaphore spans two nodes to resolve EXTERNAL";
    }
    std::vector<uint32_t> sentinel(3, kNoReport);
    slow_dispatch::WriteToL1(*mesh_device_, core, report_addr, sentinel);

    const experimental::SemaphoreSpecName SEM_A{"sem_cached"};
    const experimental::SemaphoreSpecName SEM_B{"sem_external"};
    experimental::SemaphoreSpec sem_a{.unique_id = SEM_A, .target_nodes = core};
    experimental::SemaphoreSpec sem_b{.unique_id = SEM_B, .target_nodes = experimental::NodeRange{core, second_node()}};
    const experimental::KernelSpecName KA{"cached_binder"};
    const experimental::KernelSpecName KB{"barrier_user"};
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
                {.runtime_arg_names = {"report_addr", "increment_times", "is_reporter", "wait_min_total"}},
            .hw_config = experimental::DataMovementHardwareConfig{},
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
                 {"wait_min_total", 0u}}),
        },
        experimental::ProgramRunArgs::KernelRunArgs{
            .kernel = KB,
            .runtime_arg_values = experimental::MakeRuntimeArgsForSingleNode(
                core,
                {{"report_addr", report_addr},
                 {"increment_times", concurrent_iterations},
                 {"is_reporter", 0u},
                 {"wait_min_total", 0u}}),
        },
    };
    experimental::SetProgramRunArgs(program, params);
    distributed::MeshWorkload workload;
    distributed::MeshCoordinate zero_coord{0, 0};
    workload.add_program(distributed::MeshCoordinateRange{zero_coord, zero_coord}, std::move(program));
    RunProgram(mesh_device_, workload);

    slow_dispatch::ReadFromL1(*mesh_device_, core, report_addr, 3 * sizeof(uint32_t), result);
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

// A single multi-threaded binder kernel on a different node than the semaphore. Every other
// cached conjunct holds, so only the node-confinement check can demote this to EXTERNAL.
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

// A second binder kernel on the same node as the semaphore.
TEST_F(SemScopeFixture, TestCensusSecondBinderKernelStaysCached) {
    const auto [scope, count] = run_census(
        core, {{.num_threads = 1, .increments = iterations, .reporter = true}, {.num_threads = 1, .increments = 0}});
    log_info(LogTest, "census second binder kernel: scope={} count={}", scope, count);
    EXPECT_EQ(scope, scope_val(SemScope::DM_LOCAL_CACHED))
        << "a second ON-node DM binder kernel must stay cached: single-node sem + all binders "
           "DM kernels confined to its node is cached geometry regardless of kernel count";
    EXPECT_EQ(count, iterations);
}

// An off-node binder blocks the cached path.
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

// The 2-thread single-kernel shape passes cached geometry: it must resolve DM_LOCAL_CACHED.
TEST_F(SemScopeFixture, TestCensusMultiConsumerCachedShape) {
    if (num_dms_ < 2) {
        GTEST_SKIP() << "needs >= 2 user DMs";
    }
    const auto [scope, count] = run_census(core, {{.num_threads = 2, .increments = 1, .reporter = true}});
    log_info(LogTest, "census multi-consumer cached shape: scope={} count={}", scope, count);
    EXPECT_EQ(scope, scope_val(SemScope::DM_LOCAL_CACHED))
        << "a multi-instance shape confined to one DM kernel on the semaphore's node must resolve "
           "DM_LOCAL_CACHED, a different scope here means the census misroutes the cached geometry";
    EXPECT_EQ(count, 2u) << "the cached AMO lost an update, or the pool was not initialised";
}

// Semaphore ids are unique per CORE, not per program: sem_near and sem_far each get id 0, so
// this one kernel's scope table has one slot for both. Alone, sem_near would be cached and
// sem_far EXTERNAL, a divergent slot, so the host must promote both to EXTERNAL, which is
// correct for every topology.
TEST_F(SemScopeFixture, TestSameIdSemaphoresKeepDistinctScopes) {
    if (!has_second_node()) {
        GTEST_SKIP() << "needs a second node for the disjoint-node id collision";
    }
    std::vector<uint32_t> sentinel(2, kNoReport);
    slow_dispatch::WriteToL1(*mesh_device_, core, report_addr, sentinel);

    experimental::SemaphoreSpec sem_near{
        .unique_id = experimental::SemaphoreSpecName{"sem_near"}, .target_nodes = core};
    experimental::SemaphoreSpec sem_far{
        .unique_id = experimental::SemaphoreSpecName{"sem_far"}, .target_nodes = second_node()};

    const experimental::KernelSpecName K{"slot_probe"};
    experimental::KernelSpec ks{
        .unique_id = K,
        .source = kernel_path_slot_probe,
        .num_threads = 2,
        .semaphore_bindings =
            {{.semaphore_spec_name = experimental::SemaphoreSpecName{"sem_near"}, .accessor_name = "near_sem"},
             {.semaphore_spec_name = experimental::SemaphoreSpecName{"sem_far"}, .accessor_name = "far_sem"}},
        .runtime_arg_schema = {.runtime_arg_names = {"report_addr"}},
        .hw_config = experimental::DataMovementHardwareConfig{},
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

    slow_dispatch::ReadFromL1(*mesh_device_, core, report_addr, 2 * sizeof(uint32_t), result);
    ASSERT_EQ(result.size(), 2u);
    ASSERT_NE(result[0], kNoReport) << "slot probe never reported";
    log_info(LogTest, "same-id scopes: near={} far={}", result[0], result[1]);
    EXPECT_EQ(result[0], scope_val(SemScope::DM_LOCAL_CACHED)) << "on-node sem_near should be cached";
    EXPECT_EQ(result[1], scope_val(SemScope::EXTERNAL)) << "off-node sem_far must be EXTERNAL";
}

// A kernel may bind a given semaphore only once: a second binding would double-count that
// kernel's instances in the census and distort the mechanism choice.
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
            {.runtime_arg_names = {"report_addr", "increment_times", "is_reporter", "wait_min_total"}},
        .hw_config = experimental::DataMovementHardwareConfig{},
    };
    experimental::WorkUnitSpec wu{.name = "main", .kernels = {K}, .target_nodes = core};
    experimental::ProgramSpec spec{.name = "sem_double_bind", .kernels = {ks}, .semaphores = {sem}, .work_units = {wu}};
    EXPECT_ANY_THROW({
        Program program = experimental::MakeProgramFromSpec(*mesh_device_, spec);
        (void)program;
    }) << "binding the same semaphore twice in one kernel must be rejected (it double-counts the census)";
}

// ---- COMPUTE_ATOMIC: the Tensix hardware semaphore, UNPACK <-> PACK ----
// Index PACK_UNPACK on Quasar. One kernel (test_compute_semaphore.cpp), patterns A, D, E selected by the
// `pattern` compile-time arg. The Blackhole tests are in metal2_host_api/integration_tests/compute_semaphore.cpp.

// Pattern E iterations: the emulator's slow consumer keeps this short (64 = 16 ring capacities).
static constexpr uint32_t kBoundedProducerIters = 64;

// Pattern A: one thread drives set/up/down/wait/wait_min/value through a known value sequence. Run
// once on UNPACK and once on PACK -- the same primitive must work from either thread.
TEST_F(SemScopeFixture, TestComputeAtomicSelfCheckUnpack) {
    const auto r = run_compute({.pattern = 0, .thread_sel = 0}, 6);
    log_info(
        LogTest,
        "UNPACK self-check: pass={} v(after up5)={} v(after down3)={} v(after up1)={} fail_step={}",
        r[0],
        r[1],
        r[2],
        r[3],
        r[4]);
    EXPECT_EQ(r[1], 5u);
    EXPECT_EQ(r[2], 2u);
    EXPECT_EQ(r[3], 3u);
    EXPECT_EQ(r[0], 1u) << "UNPACK primitive self-check failed at step " << r[4];
}

TEST_F(SemScopeFixture, TestComputeAtomicSelfCheckPack) {
    const auto r = run_compute({.pattern = 0, .thread_sel = 2}, 6);
    log_info(
        LogTest,
        "PACK self-check: pass={} v(after up5)={} v(after down3)={} v(after up1)={} fail_step={}",
        r[0],
        r[1],
        r[2],
        r[3],
        r[4]);
    EXPECT_EQ(r[1], 5u);
    EXPECT_EQ(r[2], 2u);
    EXPECT_EQ(r[3], 3u);
    EXPECT_EQ(r[0], 1u) << "PACK primitive self-check failed at step " << r[4];
}

// Pattern D: two-hop datacopy through a 4-slot ring (a compute self-loop DFB). A DM reader and writer move
// distinct 32x32 Float16_b tiles in and out through DRAM, and the host checks out == in bit for bit. The
// semaphore is the only thing ordering PACK's write of each slot before UNPACK's read of it.
TEST_F(SemScopeFixture, TestComputeAtomicDatacopy) {
    uint32_t mismatched = 0;
    const auto r = run_compute_datacopy(kComputeMaxTiles, /*nosync=*/0, mismatched);
    log_info(LogTest, "datacopy: packed={} final_sem={} mismatched_tiles={}", r[0], r[2], mismatched);
    EXPECT_EQ(r[0], kComputeMaxTiles) << "PACK did not finish every tile";
    EXPECT_EQ(r[2], 0u) << "semaphore did not settle to 0 -- up/down unbalanced";
    EXPECT_EQ(mismatched, 0u) << "output != input: UNPACK read the ring before PACK's writes landed";
}

// Negative control: without semaphore operations, UNPACK reads each slot before PACK writes it, so the
// output must NOT equal the input. If this passes the positive test above is not a detector.
TEST_F(SemScopeFixture, TestComputeAtomicDatacopyNoSyncControl) {
    uint32_t mismatched = 0;
    const auto r = run_compute_datacopy(kComputeMaxTiles, /*nosync=*/1, mismatched);
    log_info(
        LogTest, "datacopy no-sync control: packed={} mismatched_tiles={} / {}", r[0], mismatched, kComputeMaxTiles);
    EXPECT_EQ(r[0], kComputeMaxTiles);
    EXPECT_GT(mismatched, 0u) << "unsynchronized datacopy came out correct -- the positive test is not a detector";
}

// Pattern E: PACK gates each up() with wait_not_full() against a capacity of kComputeDepth; the slow UNPACK
// consumer's high-water mark must stay at the capacity and nothing may be lost. Report: produced, consumed,
// high-water mark, final value, UNPACK timeout.
TEST_F(SemScopeFixture, TestComputeAtomicBoundedProducer) {
    const uint32_t num_iters = kBoundedProducerIters;
    const auto r = run_compute({.pattern = 4, .num_iters = num_iters, .max_value = kComputeDepth}, 6);
    log_info(
        LogTest,
        "bounded producer: produced={} consumed={} high_water={} final={}{}",
        r[0],
        r[1],
        r[2],
        r[3],
        r[4] ? " [UNPACK TIMEOUT]" : "");
    EXPECT_EQ(r[4], 0u) << "UNPACK timed out -- a post was lost";
    EXPECT_EQ(r[1], num_iters) << "UNPACK did not consume every credit";
    EXPECT_LE(r[2], kComputeDepth) << "producer ran past the capacity -- wait_not_full() did not hold it";
    EXPECT_GE(r[2], 2u) << "consumer was never behind -- the test did not exercise back-pressure";
    EXPECT_EQ(r[3], 0u) << "semaphore did not settle to 0";
}

// Batched: wait_not_full(2); up(2) against the same capacity. Takes the RISC-poll form of wait_not_full
// (no SEMWAIT condition for "room for n"; on Quasar it settles on the CSR Sync busy bit); the high-water
// mark must still stay at the capacity. A wait that only reserved one slot would let up(2) reach
// kComputeDepth + 1.
TEST_F(SemScopeFixture, TestComputeAtomicBoundedProducerBatched) {
    const uint32_t num_iters = kBoundedProducerIters;
    const auto r = run_compute({.pattern = 4, .num_iters = num_iters, .batch = 2, .max_value = kComputeDepth}, 6);
    log_info(
        LogTest,
        "bounded producer, batch 2: produced={} consumed={} high_water={} final={}{}",
        r[0],
        r[1],
        r[2],
        r[3],
        r[4] ? " [UNPACK TIMEOUT]" : "");
    EXPECT_EQ(r[4], 0u) << "UNPACK timed out -- a post was lost";
    EXPECT_EQ(r[1], num_iters) << "UNPACK did not consume every credit";
    EXPECT_LE(r[2], kComputeDepth) << "batched producer ran past the capacity -- wait_not_full(2) reserved too little";
    EXPECT_GE(r[2], 2u) << "consumer was never behind -- the test did not exercise back-pressure";
    EXPECT_EQ(r[3], 0u) << "semaphore did not settle to 0";
}

// Same run without wait_not_full().
// Quasar SEMPOST back-pressure: even without wait_not_full(), all credits survive
// and the value stays at the programmed maximum. Unlike Blackhole, this is not a negative control
// for wait_not_full(); it checks the hardware behavior on which completion of this run depends.
TEST_F(SemScopeFixture, TestComputeAtomicBoundedProducerNoWaitControl) {
    const uint32_t num_iters = kBoundedProducerIters;
    const auto r = run_compute({.pattern = 4, .num_iters = num_iters, .nosync = 1, .max_value = kComputeDepth}, 6);
    log_info(
        LogTest,
        "bounded producer no-wait control: produced={} consumed={} high_water={}{}",
        r[0],
        r[1],
        r[2],
        r[4] ? " [UNPACK TIMEOUT]" : "");
    EXPECT_EQ(r[0], num_iters) << "PACK did not finish every post";
    EXPECT_EQ(r[4], 0u) << "UNPACK timed out waiting for a credit";
    EXPECT_EQ(r[1], num_iters) << "Quasar lost a post at capacity";
    EXPECT_EQ(r[2], kComputeDepth) << "expected hardware back-pressure at the programmed maximum";
    EXPECT_EQ(r[3], 0u) << "semaphore did not settle to 0";
}

// ---- ClusterSemaphore: public Metal compute APIs, Quasar PACK -> UNPACK across NEOs ----
// Host setup and DFBs use Metal 2.0; the kernel uses ID-based startup/copy/pack until Quasar supports LLKOperand.
// TODO @RT: Move these tests to metal2_host_api/integration_tests when that suite supports Quasar execution
// and the kernel can use Quasar LLKOperand APIs. https://github.com/tenstorrent/tt-metal/issues/60164

// NEO 0 packs a four-slot ring and NEO 1 unpacks it. The DFB is used only for storage/metadata;
// ClusterSemaphore must order both publication and slot reuse; delay PACK to exercise the ready wait.
TEST_F(SemScopeFixture, TestClusterSemaphoreDatacopy) {
    const auto result = run_cluster({});
    EXPECT_EQ(result.report[0], kComputeMaxTiles);
    EXPECT_EQ(result.report[1], kComputeMaxTiles);
    EXPECT_EQ(result.report[2], 0u) << "semaphore did not settle to zero";
    EXPECT_EQ(result.output, datacopy_input(kComputeMaxTiles));
}

// Without ClusterSemaphore, the deliberately late producer cannot protect the shared ring. This also
// checks that no DFB credit synchronization has accidentally replaced the semaphore under test.
TEST_F(SemScopeFixture, TestClusterSemaphoreDatacopyNoSyncControl) {
    const auto result = run_cluster({.nosync = true});
    EXPECT_EQ(result.report[0], kComputeMaxTiles);
    EXPECT_EQ(result.report[1], kComputeMaxTiles);
    EXPECT_GT(mismatched_tiles(datacopy_input(kComputeMaxTiles), result.output), 0u)
        << "unsynchronized datacopy came out correct -- the positive test is not a detector";
}

// Four-NEO chain through three rings, without skew. Middle NEOs copy and pack back to back to catch
// semaphore updates that run ahead of MOP/REPLAY completion. Keep all 64 distinct outputs so early
// corruption cannot be hidden by later writes to the same output slots.
TEST_F(SemScopeFixture, TestClusterSemaphoreChain) {
    constexpr uint32_t num_neos = 4;
    constexpr uint32_t num_iters = 64;
    const auto result = run_cluster(
        {.num_neos = num_neos,
         .num_iters = num_iters,
         .active_neos = num_neos,
         .max_tiles = num_iters,
         .skew_spins = 0});
    ASSERT_EQ(result.output.size(), num_iters * kTileWords);
    EXPECT_EQ(mismatched_tiles(datacopy_input(num_iters), result.output), 0u);
    for (uint32_t neo = 0; neo < num_neos; ++neo) {
        EXPECT_EQ(result.report[neo], num_iters);
        if (neo + 1 < num_neos) {
            EXPECT_EQ(result.report[num_neos + neo], 0u) << "unbalanced chain edge " << neo;
        }
    }
}
// Closed rings start full and return to depth four. Exercise one through four active NEOs while all
// four NEOs participate in initialization and completion barriers.
TEST_F(SemScopeFixture, TestClusterSemaphoreClosedRing) {
    constexpr uint32_t num_neos = 4;
    constexpr uint32_t num_iters = 64;
    for (uint32_t active_neos = 1; active_neos <= num_neos; ++active_neos) {
        SCOPED_TRACE(active_neos);
        const auto result = run_cluster(
            {.num_neos = num_neos,
             .num_iters = num_iters,
             .active_neos = active_neos,
             .closed_ring = true,
             .max_tiles = num_neos * kComputeDepth,
             .skew_spins = 0});
        const auto input = datacopy_input(num_neos * kComputeDepth);
        for (uint32_t neo = 0; neo < active_neos; ++neo) {
            const uint32_t origin = (neo + active_neos - (num_iters / kComputeDepth) % active_neos) % active_neos;
            for (uint32_t slot = 0; slot < kComputeDepth; ++slot) {
                const uint32_t source_tile = (origin * kComputeDepth + slot) % (num_neos * kComputeDepth);
                EXPECT_TRUE(std::equal(
                    input.begin() + source_tile * kTileWords,
                    input.begin() + (source_tile + 1) * kTileWords,
                    result.output.begin() + (neo * kComputeDepth + slot) * kTileWords))
                    << "corrupt ring payload: edge " << neo << ", slot " << slot;
            }
            EXPECT_EQ(result.report[neo], num_iters);
            EXPECT_EQ(result.report[num_neos + neo], kComputeDepth) << "unbalanced ring edge " << neo;
        }
    }
}

// Publication and reuse are separate obligations. Keep updates enabled in these controls:
// the wrong payload must be detected even though every credit is eventually returned.
TEST_F(SemScopeFixture, TestClusterSemaphorePublicationControl) {
    const auto result = run_cluster({.omit_ready_wait = true});
    EXPECT_GT(mismatched_tiles(datacopy_input(kComputeMaxTiles), result.output), 0u);
    EXPECT_EQ(result.report[2], 0u);
}

TEST_F(SemScopeFixture, TestClusterSemaphoreSlotReuse) {
    for (bool omit_space_wait : {false, true}) {
        SCOPED_TRACE(omit_space_wait);
        const auto result = run_cluster({.skew_spins = 0, .omit_space_wait = omit_space_wait, .consumer_skew = 2000});
        const auto mismatches = mismatched_tiles(datacopy_input(kComputeMaxTiles), result.output);
        if (omit_space_wait) {
            EXPECT_GT(mismatches, 0u) << "missing capacity wait did not corrupt the ring";
        } else {
            EXPECT_EQ(mismatches, 0u);
        }
        EXPECT_EQ(result.report[2], 0u);
    }
}

// Eight credits cross the register alias limit of seven. Two batches fit in the ring, so
// the peer can make progress between split updates.
TEST_F(SemScopeFixture, TestClusterSemaphoreBatchedCredits) {
    constexpr uint32_t tiles = 64;
    const auto result =
        run_cluster({.num_iters = tiles, .max_tiles = tiles, .skew_spins = 0, .depth = 16, .batch_tiles = 8});
    EXPECT_EQ(result.output, datacopy_input(tiles));
    EXPECT_EQ(result.report[0], tiles);
    EXPECT_EQ(result.report[1], tiles);
    EXPECT_EQ(result.report[2], 0u);
}

// The ACCUM sweep of TestClusterSemContentionVsL1Atomic with the L1 client event counters on: where does an L1
// atomic add spend its time as more TRISCs share the word? TRISC 0 of each NEO that has it active counts one event
// of its NEO's TRISC L1 port around its own `rounds` adds (one subport and event per NEO per run, so each event is a
// separate run). Events, from the Quasar performance counters overview, section 6.5:
//   1 SBANK_POP           cycles in which super bank (subport % 4) of the TRISC port granted a request
//   2 ISSUE_STALL_CARRY   lane cycles a request waited at that super bank's arbiter, one count per 4
//   4 FLEX_STALL_CARRY    cycles this subport's flex FIFOs held a request they could not pass on, one count per 4
//   5 FLEX_WORK_CARRY     requests this subport passed toward L1, one count per 4
//   6 PENDING_REQS_CARRY  this subport's outstanding requests integrated over time, one count per 64
//   7 ORDER_FIFO_ACTIVE   cycles with at least one of this subport's requests outstanding
// Little's law: mean request latency = mean outstanding / request rate = (64 * e6) / (4 * e5) over equal windows.
// Events 1 and 2 cover the whole port's super bank, so they are read only for the shared word, on the super bank
// that a calibration run finds for it. The global-semaphore backend is the control: its adds must not use the port.
// Prints `l1_counters_sbank,<super bank>,<pops with L1 adds>,<pops with global-semaphore adds>` for the
// calibration, then `l1_counters,<backend>,<layout>,<n>,<ticks/add>,<port requests/add>,<request latency>,
// <mean outstanding>,<busy fraction>,<flex stall/tick>,<sbank grant fraction>,<sbank stall/tick>`.
TEST_F(SemScopeFixture, TestClusterSemContentionL1Counters) {
    constexpr uint32_t rounds = 512;
    constexpr uint32_t neos = 4;
    struct Window {
        uint64_t count = 0;
        uint64_t ticks = 0;
        uint32_t neos = 0;
        double ticks_per_add = 0;
    };
    // Sums one event over the NEOs whose TRISC 0 was active.
    const auto measure = [&](bool use_l1, bool distinct, uint32_t n, uint32_t subport, uint32_t event) {
        const ContentionRun run{
            .mode = ContentionMode::ACCUM,
            .distinct = distinct,
            .active_mask = spread_mask(n),
            .rounds = rounds,
            .use_l1 = use_l1,
            .perf_ctrl = l1_client_ctrl(subport, event)};
        const auto r = run_contention(run);
        check_contention(run, r);
        Window w;
        for (uint32_t neo = 0; neo < neos; ++neo) {
            const uint32_t p = neo * 4;
            if (((run.active_mask >> p) & 1) == 0) {
                continue;
            }
            EXPECT_GT(r.at(p, PERF_TICKS), 0u) << "TRISC 0 of NEO " << neo << " did not report a counter window";
            w.count += r.at(p, PERF_COUNT);
            w.ticks += r.at(p, PERF_TICKS);
            ++w.neos;
        }
        w.ticks_per_add = contention_stats(r, run.active_mask).mean_ticks_per_access;
        return w;
    };

    // Calibration: the super bank of the TRISC port whose grants rise when TRISC 0 adds to the L1 word.
    uint32_t word_sbank = 0;
    int64_t best_rise = 0;
    for (uint32_t sb = 0; sb < 4; ++sb) {
        const Window l1 = measure(true, false, 1, sb, 1);
        const Window gs = measure(false, false, 1, sb, 1);
        log_info(LogTest, "l1_counters_sbank,{},{},{}", sb, l1.count, gs.count);
        const int64_t rise = static_cast<int64_t>(l1.count) - static_cast<int64_t>(gs.count);
        if (rise > best_rise) {
            best_rise = rise;
            word_sbank = sb;
        }
    }
    EXPECT_GT(best_rise, 0) << "no super bank of the TRISC port granted more while TRISC 0 added to the L1 word";

    for (const bool use_l1 : {true, false}) {
        for (const bool distinct : {false, true}) {
            for (const uint32_t n : {1u, 2u, 4u, 8u, 12u, 16u}) {
                SCOPED_TRACE(
                    std::string(use_l1 ? "l1_atomic" : "global_sem") + (distinct ? " distinct" : " same") +
                    " n=" + std::to_string(n));
                const Window flex_stall = measure(use_l1, distinct, n, 0, 4);
                const Window requests = measure(use_l1, distinct, n, 0, 5);
                const Window outstanding = measure(use_l1, distinct, n, 0, 6);
                const Window busy = measure(use_l1, distinct, n, 0, 7);
                const auto rate = [](const Window& w, double scale) {
                    return w.ticks ? scale * static_cast<double>(w.count) / static_cast<double>(w.ticks) : 0.0;
                };
                const double request_rate = rate(requests, 4);
                const double mean_outstanding = rate(outstanding, 64);
                const double requests_per_add =
                    requests.neos ? 4.0 * static_cast<double>(requests.count) / (requests.neos * rounds) : 0.0;
                double sbank_grants = 0;
                double sbank_stall = 0;
                if (!distinct) {
                    sbank_grants = rate(measure(use_l1, distinct, n, word_sbank, 1), 1);
                    sbank_stall = rate(measure(use_l1, distinct, n, word_sbank, 2), 4);
                }
                if (use_l1) {
                    // Every add is one request on TRISC 0's port; the carries can lose up to one count each.
                    EXPECT_NEAR(requests_per_add, 1.0, 0.05) << "L1 adds did not show up as TRISC port requests";
                } else {
                    EXPECT_LT(requests_per_add, 0.05) << "global-semaphore adds went through the L1 port";
                }
                log_info(
                    LogTest,
                    "l1_counters,{},{},{},{:.2f},{:.3f},{:.2f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f}",
                    use_l1 ? "l1_atomic" : "global_sem",
                    distinct ? "distinct" : "same",
                    n,
                    requests.ticks_per_add,
                    requests_per_add,
                    request_rate > 0 ? mean_outstanding / request_rate : 0.0,
                    mean_outstanding,
                    rate(busy, 1),
                    rate(flex_stall, 4),
                    sbank_grants,
                    sbank_stall);
            }
        }
    }
}

// TestCrossClusterSemUnderLoad's TRISC L1-atomic loads with the home NEOs' L1 client event counters on (the
// pack/unpack datacopy load is in TestCrossClusterSemTileCopyLoadCountersOn). The home DM coordinator
// arms each NEO's counter before the clients start and reads it when they finish, so each run reports what the home
// TRISC L1 ports did while one remote client ran its round trips. All four NEOs count the same subport and event
// (section 6.5 of the Quasar performance counters overview; see TestClusterSemContentionL1Counters for the events):
//   trisc_l1_atomic  16 TRISCs doing L1 atomics next to the L1 counter: TRISC 0's port (subport 0)
//   probe_*          one TRISC (NEO 0, TRISC 0) doing the same atomics, alone, under NoC tile writes, and under NoC
//                    atomics and reads on the counter's sub-bank (l1_subbank). If the probe's request latency rises
//                    under l1_subbank but not under noc, the sub-bank is congested; if both rise alike, the NoC is.
// The coordinator also reads the NEO wall clock next to its own rdcycle, which converts DM cycles to NEO ticks.
// Prints `xcluster_counters,<counter>,<load>,<selection>,<subport>,<event>,<rt mean cycles/op>,<count NEO 0>,
// <count NEO 1>,<count NEO 2>,<count NEO 3>,<neo ticks>,<dm cycles>`.
TEST_F(SemScopeFixture, TestCrossClusterSemLoadCounters) {
    if (!has_second_node()) {
        GTEST_SKIP() << "needs >= 2 clusters";
    }
    constexpr uint32_t rounds = 16;
    struct Sel {
        const char* name;
        uint32_t subport;
        uint32_t event;
    };
    const std::vector<Sel> trisc_port = {
        {"flex_stall", 0, 4}, {"requests", 0, 5}, {"outstanding", 0, 6}, {"busy", 0, 7}};
    struct Load {
        const char* name;
        bool trisc;
        bool trisc_l1;
        uint32_t trisc_l1_mask;
        bool noc;
        CrossLoadKind kind;
        const std::vector<Sel>& sels;
    };
    const Load loads[] = {
        {"none", false, false, 0, false, CrossLoadKind::TILE_NOC, trisc_port},
        {"trisc_l1_atomic", false, true, 0xFFFF, false, CrossLoadKind::TILE_NOC, trisc_port},
        {"probe", false, true, 0x1, false, CrossLoadKind::TILE_NOC, trisc_port},
        {"probe_noc", false, true, 0x1, true, CrossLoadKind::TILE_NOC, trisc_port},
        {"probe_l1_subbank", false, true, 0x1, true, CrossLoadKind::L1_SUBBANK, trisc_port},
    };
    for (const Load& load : loads) {
        for (const auto mode : {CrossMode::RT_GS, CrossMode::RT_L1}) {
            const char* counter = mode == CrossMode::RT_GS ? "rt_sem" : "rt_l1";
            for (const Sel& sel : load.sels) {
                const uint32_t ctrl = sel.event != 0 ? l1_client_ctrl(sel.subport, sel.event) : 0u;
                CrossRun run{
                    .mode = mode,
                    .clients = 1,
                    .count = rounds,
                    .max_remote = 4,
                    .trisc_load = load.trisc,
                    .trisc_l1_subbank = load.trisc_l1,
                    .noc_load = load.noc,
                    .load_kind = load.kind,
                    .perf_ctrl = {ctrl, ctrl, ctrl, ctrl}};
                if (load.trisc_l1) {
                    run.trisc_l1_mask = load.trisc_l1_mask;
                }
                SCOPED_TRACE(std::string(counter) + " load=" + load.name + " sel=" + sel.name);
                const auto r = run_cross_cluster(run);
                check_cross_cluster(run, r);
                if (load.trisc || load.trisc_l1) {
                    EXPECT_EQ(r.coordinator(X_LOAD_DONE_AT_END), 0u)
                        << "the compute load finished before the clients; raise kCrossLoadTiles";
                }
                if (load.trisc_l1) {
                    EXPECT_GT(r.trisc_l1_ops, 0u) << "TRISC L1-atomic loader never ran";
                }
                if (ctrl != 0) {
                    EXPECT_GT(r.coordinator(X_PERF_NEO_TICKS), 0u) << "the coordinator did not report a counter window";
                }
                const auto s = cross_stats(r);
                log_info(
                    LogTest,
                    "xcluster_counters,{},{},{},{},{},{:.1f},{},{},{},{},{},{}",
                    counter,
                    load.name,
                    sel.name,
                    sel.subport,
                    sel.event,
                    s.mean_cycles_per_op,
                    r.coordinator(X_PERF_COUNT_NEO0),
                    r.coordinator(X_PERF_COUNT_NEO1),
                    r.coordinator(X_PERF_COUNT_NEO2),
                    r.coordinator(X_PERF_COUNT_NEO3),
                    r.coordinator(X_PERF_NEO_TICKS),
                    r.coordinator(X_PERF_DM_CYCLES));
            }
        }
    }
}

// The tile-copy load (all four home NEOs running a pack/unpack datacopy) with the L1 client counters off.
TEST_F(SemScopeFixture, TestCrossClusterSemTileCopyLoadCountersOff) {
    if (!has_second_node()) {
        GTEST_SKIP() << "needs >= 2 clusters";
    }
    run_tile_copy_load(/*counters_on=*/false);
}

// The same run with the L1 client counters on.
TEST_F(SemScopeFixture, TestCrossClusterSemTileCopyLoadCountersOn) {
    if (!has_second_node()) {
        GTEST_SKIP() << "needs >= 2 clusters";
    }
    run_tile_copy_load(/*counters_on=*/true);
}

}  // namespace tt::tt_metal
