// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <vector>
#include <map>
#include <memory>
#include <string>
#include <optional>
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
#include <tt-metalium/experimental/metal2_host_api/program.hpp>

namespace tt::tt_metal {

using experimental::SemaphoreScope;

// ============================================================================
// Scoped semaphore (SemScope) tests.
// ============================================================================
// Exercises the scoped device Semaphore class (noc_semaphore.h) end-to-end
// across all three mechanisms -- LOCAL_NONATOMIC (plain L1 RMW),
// DM_LOCAL_CACHED (32-bit AMO on the cached alias), EXTERNAL (self-targeted
// NoC atomic) -- plus the AUTO classifier that picks between them. Raw
// multi-writer atomicity of the underlying hardware mechanisms is covered by
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
    uint32_t report_addr{0};
    uint32_t num_dms_{0};
    std::shared_ptr<distributed::MeshDevice> mesh_device_;
    IDevice* device_{nullptr};
    std::vector<uint32_t> result;

    void SetUp() override {
        MeshDispatchFixture::SetUp();
        // Every spec here uses DataMovementGen2Config, so the suite is Gen2 (Quasar) only.
        if (arch_ != tt::ARCH::QUASAR) {
            GTEST_SKIP() << "SemScope suite is Gen2 (Quasar) only: its specs use DataMovementGen2Config";
        }
        mesh_device_ = devices_[0];
        device_ = mesh_device_->get_devices()[0];
        report_addr = device_->allocator()->get_base_allocator_addr(HalMemType::L1);
        num_dms_ = MetalContext::instance().hal().get_processor_types_count(HalProgrammableCoreType::TENSIX, 0);
        num_dms_ = std::min(num_dms_, 6u);  // Metal 2.0 reserves DM0/DM1
    }

    // Runs the smoke kernel with the given host-baked scope and returns the value the
    // kernel read back from the semaphore after `iterations` increments. The scope is set on
    // the SemaphoreSpec; the kernel is scope-agnostic and picks it up via CTAD on the emitted
    // sem::counter token.
    uint32_t run_scope(
        SemaphoreScope scope,
        bool with_down = false,
        std::optional<experimental::SemaphoreAccessType> access_override = std::nullopt) {
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
        counter_sem.scope = scope;  // host-baked mechanism; kernel deduces it via CTAD

        std::map<std::string, std::string> defs;
        if (with_down) {
            defs.emplace("SEM_SCOPE_UPDOWN", "1");  // kernel also does down(N) after up(N)
        }
        experimental::KernelSpec::CompilerOptions::Defines defines_obj(defs);

        const experimental::KernelSpecName DM_KERNEL{"sem_scope_kernel"};
        experimental::KernelSpec kernel_spec{
            .unique_id = DM_KERNEL,
            .source = kernel_path,
            .num_threads = 1,
            .compiler_options = {.defines = defines_obj},
            .semaphore_bindings =
                {{.semaphore_spec_name = experimental::SemaphoreSpecName{"counter_sem"},
                  .accessor_name = "counter",
                  // Label matches what the kernel does: SEM_SCOPE_UPDOWN really down()s.
                  // (access_override exists ONLY for the enforcement-negative test.)
                  .access_type = access_override.value_or(
                      with_down ? experimental::SemaphoreAccessType::CONSUME
                                : experimental::SemaphoreAccessType::INCREMENT)}},
            .runtime_arg_schema = {.runtime_arg_names = {"report_addr", "increment_times"}},
            .hw_config = experimental::DataMovementGen2Config{},
        };

        experimental::WorkUnitSpec main_wu{
            .name = "main",
            .kernels = {DM_KERNEL},
            .target_nodes = core,
        };
        experimental::ProgramSpec spec{
            .name = "sem_scope_smoke",
            .kernels = {kernel_spec},
            .semaphores = {counter_sem},
            .work_units = {main_wu},
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
        experimental::SetProgramRunArgs(program, params);

        workload.add_program(device_range, std::move(program));
        RunProgram(mesh_device_, workload);

        tt::tt_metal::detail::ReadFromDeviceL1(device_, core, report_addr, sizeof(uint32_t), result);
        EXPECT_EQ(result.size(), 1u);
        // The kernel must have overwritten the sentinel, or the 0-expecting tests pass vacuously.
        EXPECT_NE(result.empty() ? kNoReport : result[0], kNoReport) << "kernel never reported";
        return result.empty() ? 0u : result[0];
    }

    // Multi-DM concurrency proof (Quasar-only; roles gated by mhartid in the kernel).
    // num_dms threads share one bound Semaphore<scope>; mode picks CONCURRENT_UP,
    // PRODUCER_CONSUMER or MULTI_CONSUMER. counter_access labels the counter binding for the
    // census (modes that really down() pass CONSUME). Returns the value()
    // the reporter/consumer read back.
    uint32_t run_concurrent(
        SemaphoreScope scope,
        const std::string& mode_define,
        experimental::SemaphoreAccessType counter_access = experimental::SemaphoreAccessType::INCREMENT) {
        // Sentinel prefill: the producer-consumer tests expect 0, so a zero prefill would pass
        // even if the reporter thread never ran.
        std::vector<uint32_t> sentinel(1, kNoReport);
        tt::tt_metal::detail::WriteToDeviceL1(device_, core, report_addr, sentinel);

        distributed::MeshWorkload workload;
        Program program;
        distributed::MeshCoordinate zero_coord{0, 0};
        distributed::MeshCoordinateRange device_range{zero_coord, zero_coord};

        experimental::SemaphoreSpec counter_sem{
            .unique_id = experimental::SemaphoreSpecName{"counter_sem"}, .target_nodes = core};
        experimental::SemaphoreSpec done_sem{
            .unique_id = experimental::SemaphoreSpecName{"done_sem"}, .target_nodes = core};
        counter_sem.scope = scope;  // both semaphores share the host-baked mechanism
        done_sem.scope = scope;

        std::map<std::string, std::string> defs{{mode_define, "1"}};
        experimental::KernelSpec::CompilerOptions::Defines defines_obj(defs);

        const experimental::KernelSpecName DM_KERNEL{"sem_scope_concurrent_kernel"};
        experimental::KernelSpec kernel_spec{
            .unique_id = DM_KERNEL,
            .source = kernel_path_concurrent,
            .num_threads = num_dms_,
            .compiler_options = {.defines = defines_obj},
            .semaphore_bindings =
                {{.semaphore_spec_name = experimental::SemaphoreSpecName{"counter_sem"},
                  .accessor_name = "counter",
                  .access_type = counter_access},
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

        tt::tt_metal::detail::ReadFromDeviceL1(device_, core, report_addr, sizeof(uint32_t), result);
        EXPECT_EQ(result.size(), 1u);
        // The kernel must have overwritten the sentinel, or the 0-expecting tests pass vacuously.
        EXPECT_NE(result.empty() ? kNoReport : result[0], kNoReport) << "kernel never reported";
        return result.empty() ? 0u : result[0];
    }

    // Coexistence run: a DM_LOCAL_CACHED semaphore (pool) + an EXTERNAL semaphore (ring) hammered
    // concurrently by all DMs; returns {cached_final, external_final}, both expected num_dms*iters.
    std::pair<uint32_t, uint32_t> run_coexist() {
        std::vector<uint32_t> zero(2, 0);
        tt::tt_metal::detail::WriteToDeviceL1(device_, core, report_addr, zero);

        distributed::MeshWorkload workload;
        Program program;
        distributed::MeshCoordinate zero_coord{0, 0};
        distributed::MeshCoordinateRange device_range{zero_coord, zero_coord};

        experimental::SemaphoreSpec cached_sem{
            .unique_id = experimental::SemaphoreSpecName{"cached_sem"}, .target_nodes = core};
        experimental::SemaphoreSpec external_sem{
            .unique_id = experimental::SemaphoreSpecName{"external_sem"}, .target_nodes = core};
        experimental::SemaphoreSpec done_sem{
            .unique_id = experimental::SemaphoreSpecName{"done_sem"}, .target_nodes = core};
        cached_sem.scope = SemaphoreScope::DM_LOCAL_CACHED;  // -> pool
        external_sem.scope = SemaphoreScope::EXTERNAL;       // -> ring
        done_sem.scope = SemaphoreScope::EXTERNAL;           // completion barrier

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

        tt::tt_metal::detail::ReadFromDeviceL1(device_, core, report_addr, 2 * sizeof(uint32_t), result);
        EXPECT_EQ(result.size(), 2u);
        if (result.size() < 2) {
            return {0u, 0u};
        }
        return {result[0], result[1]};
    }

    // Remote-up run: a SENDER kernel on second_node() bumps sem::counter on `core` purely via
    // Semaphore::up(noc, x, y, 1) while a RECEIVER kernel on `core` (OBSERVE binding: it only
    // waits/reads) waits for the exact total, then reports {baked scope, value()}. Callers must
    // skip-guard on has_second_node().
    std::pair<uint32_t, uint32_t> run_remote(SemaphoreScope scope, uint32_t sender_threads, uint32_t iters) {
        const uint32_t expected = sender_threads * iters;
        // Sentinel prefill (same discipline as run_census): a zero prefill could hide a receiver
        // that never reported.
        std::vector<uint32_t> sentinel(2, kNoReport);
        tt::tt_metal::detail::WriteToDeviceL1(device_, core, report_addr, sentinel);

        // The sender addresses the semaphore's node by its virtual NoC coords.
        const CoreCoord core_virtual = mesh_device_->worker_core_from_logical_core(core);

        distributed::MeshWorkload workload;
        distributed::MeshCoordinate zero_coord{0, 0};
        distributed::MeshCoordinateRange device_range{zero_coord, zero_coord};

        experimental::SemaphoreSpec counter_sem{
            .unique_id = experimental::SemaphoreSpecName{"counter_sem"}, .target_nodes = core};
        counter_sem.scope = scope;

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
                {{.semaphore_spec_name = experimental::SemaphoreSpecName{"counter_sem"},
                  .accessor_name = "counter",
                  .access_type = experimental::SemaphoreAccessType::OBSERVE}},
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

    // ---- Census / AUTO-classifier probe harness ----
    // One binding kernel in a census shape. All kernels are single-node so per-node runtime args stay
    // simple; off-node shapes set CensusKernel::node (sole kernel elsewhere, or a second kernel there).
    struct CensusKernel {
        experimental::NodeCoord node{core};
        uint32_t num_threads = 1;
        experimental::SemaphoreAccessType access = experimental::SemaphoreAccessType::INCREMENT;
        uint32_t increments = 0;  // 0 for an OBSERVE reader (it must not write)
        bool reporter = false;    // exactly one kernel should report
    };

    // Build + run a program with the given census shape; returns {baked_scope, counter_value}.
    // The scope is read back from the device -- the classifier's ACTUAL decision, since counts
    // alone look right under any correct mechanism.
    std::pair<uint32_t, uint32_t> run_census(
        const experimental::Nodes& sem_target,
        const std::vector<CensusKernel>& kernels,
        SemaphoreScope scope = SemaphoreScope::AUTO) {
        // Sentinel prefill (LOCAL_NONATOMIC == 0, so a zero prefill would hide "never reported").
        // Prefill/read at the REPORTER's node: the probe writes its local L1.
        experimental::NodeCoord reporter_node = core;
        for (const auto& k : kernels) {
            if (k.reporter) {
                reporter_node = k.node;
            }
        }
        std::vector<uint32_t> sentinel(4, kNoReport);
        tt::tt_metal::detail::WriteToDeviceL1(device_, reporter_node, report_addr, sentinel);

        distributed::MeshWorkload workload;
        distributed::MeshCoordinate zero_coord{0, 0};
        distributed::MeshCoordinateRange device_range{zero_coord, zero_coord};

        experimental::SemaphoreSpec sem{
            .unique_id = experimental::SemaphoreSpecName{"counter_sem"}, .target_nodes = sem_target};
        sem.scope = scope;

        std::vector<experimental::KernelSpec> kernel_specs;
        experimental::ProgramRunArgs params;
        // Each kernel gets barrier slot = its index; only NUM_KERNEL_BARRIERS(3) slots exist and
        // wait_threads() does not bounds-check. All 3 slots are free HERE: census kernels use no
        // DFBs (whose runtime claims slots 0/1), and the injected cached-pool seeder rendezvouses
        // on its own dedicated barrier, not an array slot.
        EXPECT_LE(kernels.size(), 3u) << "run_census supports at most 3 kernels (barrier slots)";
        if (kernels.size() > 3u) {
            return {kNoReport, 0u};
        }
        // WorkUnitSpecs must have DISJOINT target nodes, so kernels sharing a node go into ONE work
        // unit (a work unit holds a Group of kernels). Grouped here by node, preserving order.
        std::vector<std::pair<experimental::NodeCoord, std::vector<experimental::KernelSpecName>>> by_node;
        for (size_t i = 0; i < kernels.size(); i++) {
            const auto& k = kernels[i];
            const experimental::KernelSpecName name{"census_k" + std::to_string(i)};
            kernel_specs.push_back(experimental::KernelSpec{
                .unique_id = name,
                .source = kernel_path_census,
                .num_threads = k.num_threads,
                .semaphore_bindings = {{
                    .semaphore_spec_name = experimental::SemaphoreSpecName{"counter_sem"},
                    .accessor_name = "counter",
                    .access_type = k.access,
                }},
                .runtime_arg_schema =
                    {.runtime_arg_names = {"report_addr", "increment_times", "is_reporter", "barrier_idx"}},
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
                     {"barrier_idx", static_cast<uint32_t>(i)}}),
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

        tt::tt_metal::detail::ReadFromDeviceL1(device_, reporter_node, report_addr, 4 * sizeof(uint32_t), result);
        EXPECT_EQ(result.size(), 4u);
        if (result.size() < 4) {
            return {kNoReport, 0u};
        }
        census_ring_word_ = result[2];
        census_access_ = result[3];
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
    // The baked SemAccess value as observed by the last run_census() (report word 3); only a device
    // readback can prove the host actually emitted the bit.
    uint32_t census_access_{kNoReport};

    // Where the binding kernel is placed (defaults to `core`). Set wider to build the
    // "a binder runs outside the semaphore's node" case for the cached-pool guard.
    experimental::Nodes kernel_target_{core};

    // Build (but do not run) a minimal ProgramSpec whose semaphore carries the given scope
    // intent on the given target, so ValidateProgramSpec's ResolveSemaphoreScope runs.
    // Throws (TT_FATAL) on a contradiction before compiling. Nothing runs on the device, but the
    // accepted scopes do JIT-compile (MakeProgramFromSpec builds the program).
    void make_program_with_forced_scope(SemaphoreScope scope, const experimental::Nodes& sem_target) {
        experimental::SemaphoreSpec sem{
            .unique_id = experimental::SemaphoreSpecName{"counter_sem"}, .target_nodes = sem_target};
        sem.scope = scope;
        const experimental::KernelSpecName K{"k"};
        experimental::KernelSpec ks{
            .unique_id = K,
            .source = kernel_path,
            .num_threads = 1,
            .semaphore_bindings =
                {{.semaphore_spec_name = experimental::SemaphoreSpecName{"counter_sem"}, .accessor_name = "counter"}},
            .runtime_arg_schema = {.runtime_arg_names = {"report_addr", "increment_times"}},
            .hw_config = experimental::DataMovementGen2Config{},
        };
        experimental::WorkUnitSpec wu{.name = "main", .kernels = {K}, .target_nodes = kernel_target_};
        experimental::ProgramSpec spec{
            .name = "sem_scope_validate", .kernels = {ks}, .semaphores = {sem}, .work_units = {wu}};
        Program program = experimental::MakeProgramFromSpec(*mesh_device_, spec);
        (void)program;
    }
};

// EXTERNAL scope: local up() is a self-targeted NoC atomic increment; value() reads
// via the uncached alias. Single writer -> value() == iterations.
TEST_F(SemScopeFixture, TestExternalScopeIncrement) {
    const uint32_t observed = run_scope(SemaphoreScope::EXTERNAL);
    log_info(LogTest, "EXTERNAL scope value(): {} (expected {})", observed, iterations);
    EXPECT_EQ(observed, iterations)
        << "Semaphore<EXTERNAL>::up()/value() did not produce the expected single-writer count.";
}

// DM_LOCAL_CACHED scope: local up() is a 32-bit RISC-V AMO on the cached alias;
// value() reads via the cached alias. Single writer -> value() == iterations.
TEST_F(SemScopeFixture, TestDmLocalCachedScopeIncrement) {
    const uint32_t observed = run_scope(SemaphoreScope::DM_LOCAL_CACHED);
    log_info(LogTest, "DM_LOCAL_CACHED scope value(): {} (expected {})", observed, iterations);
    EXPECT_EQ(observed, iterations)
        << "Semaphore<DM_LOCAL_CACHED>::up()/value() did not produce the expected single-writer count.";
}

// LOCAL_NONATOMIC scope: the legacy default (plain L1 read-modify-write). Single writer ->
// value() == iterations. This also confirms the default path (used by existing DFB/CCL/SDPA
// callers, and by Gen1 / Gen2-single-writer AUTO resolutions) still compiles + works after the
// token flip.
TEST_F(SemScopeFixture, TestLocalNonatomicScopeIncrement) {
    const uint32_t observed = run_scope(SemaphoreScope::LOCAL_NONATOMIC);
    log_info(LogTest, "LOCAL_NONATOMIC scope value(): {} (expected {})", observed, iterations);
    EXPECT_EQ(observed, iterations)
        << "Semaphore<LOCAL_NONATOMIC>::up()/value() (legacy default) did not produce the expected count.";
}

// up(N) then down(N) must leave the semaphore at 0, per scope. For EXTERNAL this
// exercises the atomic NoC decrement (INCR_GET of a negative value); for
// DM_LOCAL_CACHED the LR/SC conditional decrement; for LOCAL_NONATOMIC the legacy decrement.
TEST_F(SemScopeFixture, TestExternalScopeUpDown) {
    const uint32_t observed = run_scope(SemaphoreScope::EXTERNAL, /*with_down=*/true);
    log_info(LogTest, "EXTERNAL up/down value(): {} (expected 0)", observed);
    EXPECT_EQ(observed, 0u) << "Semaphore<EXTERNAL>::down() (atomic NoC decrement) did not return to 0.";
}

TEST_F(SemScopeFixture, TestDmLocalCachedScopeUpDown) {
    const uint32_t observed = run_scope(SemaphoreScope::DM_LOCAL_CACHED, /*with_down=*/true);
    log_info(LogTest, "DM_LOCAL_CACHED up/down value(): {} (expected 0)", observed);
    EXPECT_EQ(observed, 0u) << "Semaphore<DM_LOCAL_CACHED>::down() (LR/SC conditional decrement) did not return to 0.";
}

TEST_F(SemScopeFixture, TestLocalNonatomicScopeUpDown) {
    const uint32_t observed = run_scope(SemaphoreScope::LOCAL_NONATOMIC, /*with_down=*/true);
    log_info(LogTest, "LOCAL_NONATOMIC up/down value(): {} (expected 0)", observed);
    EXPECT_EQ(observed, 0u) << "Semaphore<LOCAL_NONATOMIC>::down() (legacy) did not return to 0.";
}

// NOTE: token-CTAD deduction is exercised by every test here: sem::counter is a
// SemaphoreBindingToken<id, baked-scope, baked-access> and the kernels construct via plain
// `Semaphore s(sem::counter)`.

// ---- Concurrency proofs (Quasar-only): the single-writer tests above cannot tell an
// atomic path from a non-atomic one; these do. ----

// G1: all num_dms DMs concurrently up(1) a shared Semaphore. Exact num_dms*iters proves
// up() routes to the atomic mechanism (a non-atomic up() would lose updates -> less).
TEST_F(SemScopeFixture, TestExternalConcurrentUp) {
    const uint32_t observed = run_concurrent(SemaphoreScope::EXTERNAL, "MODE_CONCURRENT_UP");
    const uint32_t expected = num_dms_ * concurrent_iterations;
    log_info(LogTest, "EXTERNAL concurrent up value(): {} (expected {})", observed, expected);
    EXPECT_EQ(observed, expected) << "Semaphore<EXTERNAL>::up() lost updates under concurrency (non-atomic route?).";
}

TEST_F(SemScopeFixture, TestDmLocalCachedConcurrentUp) {
    const uint32_t observed = run_concurrent(SemaphoreScope::DM_LOCAL_CACHED, "MODE_CONCURRENT_UP");
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
    const uint32_t observed =
        run_concurrent(SemaphoreScope::EXTERNAL, "MODE_PRODUCER_CONSUMER", experimental::SemaphoreAccessType::CONSUME);
    log_info(LogTest, "EXTERNAL producer/consumer value(): {} (expected 0)", observed);
    EXPECT_EQ(observed, 0u) << "Semaphore<EXTERNAL>::down() lost a concurrent producer increment (non-atomic?).";
}

TEST_F(SemScopeFixture, TestDmLocalCachedProducerConsumer) {
    if (num_dms_ < 2) {
        GTEST_SKIP() << "needs >= 2 user DMs";
    }
    const uint32_t observed = run_concurrent(
        SemaphoreScope::DM_LOCAL_CACHED, "MODE_PRODUCER_CONSUMER", experimental::SemaphoreAccessType::CONSUME);
    log_info(LogTest, "DM_LOCAL_CACHED producer/consumer value(): {} (expected 0)", observed);
    EXPECT_EQ(observed, 0u) << "Semaphore<DM_LOCAL_CACHED>::down() lost a concurrent producer increment.";
}

// G3: ONE producer feeds single credits while (num_dms-1) consumers CONCURRENTLY down(iters).
// Exact 0 proves the cached down()'s CAS retry loop never double-spends a credit (an overdraw
// wraps unsigned -> huge nonzero report; a lost credit blocks a consumer -> 'done' never fills
// -> RunProgram timeout). The counter binding is honestly labelled CONSUME.
TEST_F(SemScopeFixture, TestDmLocalCachedMultiConsumerDown) {
    if (num_dms_ < 2) {
        GTEST_SKIP() << "needs >= 2 user DMs for one producer plus at least one consumer";
    }
    const uint32_t observed = run_concurrent(
        SemaphoreScope::DM_LOCAL_CACHED, "MODE_MULTI_CONSUMER", experimental::SemaphoreAccessType::CONSUME);
    log_info(LogTest, "DM_LOCAL_CACHED multi-consumer down value(): {} (expected 0)", observed);
    EXPECT_EQ(observed, 0u)
        << "Semaphore<DM_LOCAL_CACHED>::down() double-spent a credit under multi-consumer contention "
           "(the LR/SC CAS retry loop is broken -- two consumers passed the >= check on one credit).";
}

// EXTERNAL multi-consumer down(): the NoC-CAS lock serializes consumers while the producer's
// plain increments commute. Exact 0 = no credit double-spent, none lost.
TEST_F(SemScopeFixture, TestExternalMultiConsumerDown) {
    if (num_dms_ < 2) {
        GTEST_SKIP() << "needs >= 2 user DMs for one producer plus at least one consumer";
    }
    const uint32_t observed =
        run_concurrent(SemaphoreScope::EXTERNAL, "MODE_MULTI_CONSUMER", experimental::SemaphoreAccessType::CONSUME);
    log_info(LogTest, "EXTERNAL multi-consumer down value(): {} (expected 0)", observed);
    EXPECT_EQ(observed, 0u) << "EXTERNAL multi-consumer down() double-spent or lost a credit";
}

// End-to-end proof the lifted census works: a >=2-CONSUME shape confined to ONE kernel on the
// semaphore's node passes cached geometry, so under AUTO it resolves DM_LOCAL_CACHED and must
// drain exactly.
TEST_F(SemScopeFixture, TestAutoMultiConsumerDown) {
    if (num_dms_ < 2) {
        GTEST_SKIP() << "needs >= 2 user DMs for one producer plus at least one consumer";
    }
    const uint32_t observed =
        run_concurrent(SemaphoreScope::AUTO, "MODE_MULTI_CONSUMER", experimental::SemaphoreAccessType::CONSUME);
    log_info(LogTest, "AUTO multi-consumer down value(): {} (expected 0)", observed);
    EXPECT_EQ(observed, 0u)
        << "AUTO on a single-kernel multi-CONSUME shape did not drain exactly: either the census still "
           "rejects/misroutes the cached-geometry shape, or the resolved mechanism's down() is not "
           "multi-consumer-safe.";
}

// ---- Cached-pool / EXTERNAL coexistence ----

// A DM_LOCAL_CACHED sem (pool) + an EXTERNAL sem (ring) hammered concurrently by all DMs in one
// program. The pool is physically disjoint from the NoC-written ring (MEM_DM_CACHED_SEM_BASE <
// MEM_MAP_END <= kernel_config ring base), so the cached AMO's 64B-line write-back cannot clobber
// the external sem's ring word (nor vice versa). Both counts exact => coexistence works.
TEST_F(SemScopeFixture, TestCachedExternalCoexistence) {
    const auto [cached_val, external_val] = run_coexist();
    const uint32_t expected = num_dms_ * concurrent_iterations;
    log_info(LogTest, "coexistence: cached={} external={} (each expected {})", cached_val, external_val, expected);
    EXPECT_EQ(cached_val, expected)
        << "DM_LOCAL_CACHED (pool) count wrong -> cached AMO lost updates or the pool was not initialised.";
    EXPECT_EQ(external_val, expected)
        << "EXTERNAL (ring) count wrong -> the cached sem's dirty-line write-back clobbered the NoC-written "
           "ring word (pool separation FAILED).";
}

// ---- AUTO classifier behavior ----

// AUTO on a MULTI-writer semaphore must resolve to an ATOMIC mechanism: exact num_dms*iters proves
// no lost updates (a wrong LOCAL_NONATOMIC pick would come up short). The specific mechanism picked
// (here the cached pool) is asserted by the TestCensus* tests.
TEST_F(SemScopeFixture, TestAutoMultiWriterIsAtomic) {
    const uint32_t observed = run_concurrent(SemaphoreScope::AUTO, "MODE_CONCURRENT_UP");
    const uint32_t expected = num_dms_ * concurrent_iterations;
    log_info(LogTest, "AUTO multi-writer up value(): {} (expected {})", observed, expected);
    EXPECT_EQ(observed, expected)
        << "AUTO on a multi-writer semaphore lost updates => the classifier wrongly picked "
           "LOCAL_NONATOMIC (the reporter waits on the separate done semaphore, so an undercount "
           "fails HERE, unlike the remote tests).";
}

// AUTO on a SINGLE-writer semaphore takes the cheap LOCAL_NONATOMIC path. Count-only end-to-end
// check; the actual resolution is asserted by TestCensusSingleWriterPicksLocal.
TEST_F(SemScopeFixture, TestAutoSingleWriterEndToEnd) {
    const uint32_t observed = run_scope(SemaphoreScope::AUTO);
    log_info(LogTest, "AUTO single-writer value(): {} (expected {})", observed, iterations);
    EXPECT_EQ(observed, iterations) << "AUTO on a single-writer semaphore did not produce the expected count.";
}

// ---- Remote up() proofs: the first exact-count tests of Semaphore::up(noc, x, y, v). ----
// Every earlier test drives the semaphore from its own node; these drive it from a SECOND node,
// through the class's remote up() (a NoC atomic under every non-cached scope).
// FAILURE DIRECTIONS: the receiver wait_min()s for the expected total before reporting, so a LOST
// increment manifests as a RunProgram hang, not a failed EXPECT; the exact-count EXPECTs catch
// overshoot and wrong-word landings.

// One off-node sender thread. Exact count proves remote up() reaches the semaphore's word and
// loses nothing; the scope report proves it went through the forced mechanism.
TEST_F(SemScopeFixture, TestExternalRemoteUpExactCount) {
    if (!has_second_node()) {
        GTEST_SKIP() << "needs >= 2 worker nodes for an off-node sender";
    }
    const auto [scope, value] = run_remote(SemaphoreScope::EXTERNAL, /*sender_threads=*/1, iterations);
    log_info(LogTest, "EXTERNAL remote up: scope={} value={} (expected {})", scope, value, iterations);
    EXPECT_EQ(scope, scope_val(SemScope::EXTERNAL)) << "forced EXTERNAL must be the scope baked into both kernels";
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
    const auto [scope, value] = run_remote(SemaphoreScope::EXTERNAL, num_dms_, concurrent_iterations);
    const uint32_t expected = num_dms_ * concurrent_iterations;
    log_info(LogTest, "EXTERNAL concurrent remote up: scope={} value={} (expected {})", scope, value, expected);
    EXPECT_EQ(scope, scope_val(SemScope::EXTERNAL)) << "forced EXTERNAL must be the scope baked into both kernels";
    EXPECT_EQ(value, expected)
        << "concurrent Semaphore::up(noc, x, y, 1) from " << num_dms_
        << " sender threads overshot or landed on the wrong word (undercount from lost updates would "
           "hang in the receiver's wait_min before reporting).";
}

// AUTO with an off-node multi-threaded writer: the census sees a binder outside the semaphore's
// node, so the cached pool is off the table and AUTO must resolve EXTERNAL -- and still count
// exactly end to end.
TEST_F(SemScopeFixture, TestAutoRemoteWriterExternalExactCount) {
    if (num_dms_ < 2) {
        GTEST_SKIP() << "needs >= 2 user DMs";
    }
    if (!has_second_node()) {
        GTEST_SKIP() << "needs >= 2 worker nodes for an off-node sender";
    }
    const auto [scope, value] = run_remote(SemaphoreScope::AUTO, num_dms_, concurrent_iterations);
    const uint32_t expected = num_dms_ * concurrent_iterations;
    log_info(LogTest, "AUTO remote writer: scope={} value={} (expected {})", scope, value, expected);
    EXPECT_EQ(scope, scope_val(SemScope::EXTERNAL))
        << "AUTO must demote an off-node-written semaphore to EXTERNAL (any local mechanism would split "
           "or lose the remote increments)";
    EXPECT_EQ(value, expected)
        << "the AUTO-resolved mechanism lost off-node increments -- remote up() and the resolved scope "
           "do not converge on one atomicity point.";
}

// ============================================================================
// AUTO-classifier / census stress suite.
// ============================================================================
// Each test builds a distinct census shape and asserts the scope the host ACTUALLY baked into the
// kernel (read back from the device), so a regression in the classifier or the census arithmetic
// fails loudly instead of hiding behind counts that look right under any correct mechanism.
// Off-node shapes use second_node() and skip only on a single-node 1x1 grid.
// ============================================================================

// 1 writer instance on a single-cell semaphore -> nothing can race -> cheapest path.
TEST_F(SemScopeFixture, TestCensusSingleWriterPicksLocal) {
    const auto [scope, count] = run_census(core, {{.num_threads = 1, .increments = iterations, .reporter = true}});
    log_info(LogTest, "census single-writer: scope={} count={}", scope, count);
    EXPECT_EQ(scope, scope_val(SemScope::LOCAL_NONATOMIC)) << "single writer should take the cheap uncached path";
    EXPECT_EQ(count, iterations);
}

// Many writer instances (threads) all on the semaphore's ONE node -> cached-pool AMO: atomic among
// the node's coherent DM cores, with no NoC round-trip. This is the auto-selected fast path.
TEST_F(SemScopeFixture, TestCensusMultiThreadPicksCached) {
    if (num_dms_ < 2) {
        GTEST_SKIP() << "needs >= 2 user DMs";
    }
    const auto [scope, count] =
        run_census(core, {{.num_threads = num_dms_, .increments = concurrent_iterations, .reporter = true}});
    const uint32_t expected = num_dms_ * concurrent_iterations;
    log_info(LogTest, "census multi-thread: scope={} count={} (expected {})", scope, count, expected);
    EXPECT_EQ(scope, scope_val(SemScope::DM_LOCAL_CACHED))
        << "multi-writer confined to one node should auto-select the cached fast path";
    EXPECT_EQ(count, expected) << "cached AMO lost updates under concurrency, or the pool was not initialised";
}

// RESIDENCY: a cached semaphore's count must live in the POOL, so its kernel_config RING slot must
// still hold the untouched initial value. Every other cached assertion is count-only and would pass
// even if the semaphore had silently fallen back to the ring -- this is the only test that can tell.
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

// POSITIVE CONTROL for the residency probe: under forced EXTERNAL the live count IS in the ring
// slot, so report[2] must carry it (non-zero). Without this, a broken ring readback that always
// returns 0 would make TestCachedSemLivesInPoolNotRing pass vacuously.
TEST_F(SemScopeFixture, TestRingResidencyProbePositiveControl) {
    if (num_dms_ < 2) {
        GTEST_SKIP() << "needs >= 2 user DMs";
    }
    const auto [scope, count] = run_census(
        core,
        {{.num_threads = num_dms_, .increments = concurrent_iterations, .reporter = true}},
        SemaphoreScope::EXTERNAL);
    ASSERT_EQ(scope, scope_val(SemScope::EXTERNAL));
    EXPECT_EQ(count, num_dms_ * concurrent_iterations);
    log_info(LogTest, "ring residency positive control: count={} ring_slot={}", count, census_ring_word_);
    EXPECT_EQ(census_ring_word_, count)
        << "an EXTERNAL semaphore's count lives in the ring slot, but the probe's ring readback did "
           "not see it -- the residency check could pass vacuously";
}

// TWO cached-eligible semaphores, each with its own single binder kernel, on one node: the injected
// pool seeders rendezvous on ONE node-wide dedicated barrier, so caching both would mix rendezvous
// groups (unequal thread counts hang at entry; equal counts let a seed land after an increment).
// Both must be demoted to EXTERNAL. Uses different thread counts -- the hanging variant.
TEST_F(SemScopeFixture, TestCensusTwoCachedSemsOneNodePicksExternal) {
    // threads_b = num_dms_ - 2 must be >= 2 (so BOTH kernels are cached candidates) and != 2
    // (unequal counts, the hanging variant). num_dms_ >= 5 guarantees both.
    if (num_dms_ < 5) {
        GTEST_SKIP() << "needs >= 5 user DMs so both kernels are multi-threaded cached candidates "
                        "with different thread counts";
    }
    std::vector<uint32_t> sentinel(4, kNoReport);
    tt::tt_metal::detail::WriteToDeviceL1(device_, core, report_addr, sentinel);

    // Two semaphores, two kernels of DIFFERENT thread counts, both on `core`, one kernel per semaphore.
    const experimental::SemaphoreSpecName SEM_A{"sem_a"};
    const experimental::SemaphoreSpecName SEM_B{"sem_b"};
    experimental::SemaphoreSpec sem_a{.unique_id = SEM_A, .target_nodes = core};
    experimental::SemaphoreSpec sem_b{.unique_id = SEM_B, .target_nodes = core};
    const experimental::KernelSpecName KA{"cached_ka"};
    const experimental::KernelSpecName KB{"cached_kb"};
    const uint32_t threads_a = 2;
    const uint32_t threads_b = num_dms_ - threads_a;  // different count -> the hanging variant

    auto make_k = [&](const experimental::KernelSpecName& name,
                      const experimental::SemaphoreSpecName& sem_name,
                      uint32_t threads) {
        return experimental::KernelSpec{
            .unique_id = name,
            .source = kernel_path_census,
            .num_threads = threads,
            .semaphore_bindings = {{.semaphore_spec_name = sem_name, .accessor_name = "counter"}},
            .runtime_arg_schema =
                {.runtime_arg_names = {"report_addr", "increment_times", "is_reporter", "barrier_idx"}},
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
                 {"barrier_idx", 0u}}),
        },
        experimental::ProgramRunArgs::KernelRunArgs{
            .kernel = KB,
            .runtime_arg_values = experimental::MakeRuntimeArgsForSingleNode(
                core,
                {{"report_addr", report_addr},
                 {"increment_times", concurrent_iterations},
                 {"is_reporter", 0u},
                 {"barrier_idx", 1u}}),
        },
    };
    experimental::SetProgramRunArgs(program, params);
    distributed::MeshWorkload workload;
    distributed::MeshCoordinate zero_coord{0, 0};
    workload.add_program(distributed::MeshCoordinateRange{zero_coord, zero_coord}, std::move(program));
    RunProgram(mesh_device_, workload);  // must COMPLETE (a cached pick here would hang at entry)

    tt::tt_metal::detail::ReadFromDeviceL1(device_, core, report_addr, 4 * sizeof(uint32_t), result);
    ASSERT_EQ(result.size(), 4u);
    ASSERT_NE(result[0], kNoReport) << "probe never reported: the reporter kernel/thread did not run";
    log_info(LogTest, "two cached sems on one node: scope={} count={}", result[0], result[1]);
    EXPECT_EQ(result[0], scope_val(SemScope::EXTERNAL))
        << "two co-resident cached-binder kernels share one node-wide seeder barrier -> both must "
           "be demoted to EXTERNAL, not cached";
    EXPECT_EQ(result[1], threads_a * concurrent_iterations);
}

// Regression: the injected cached-pool seeder must be IMMUNE to a co-resident kernel's barrier-slot
// choice. KA is the sole cached binder (seeder injected at its entry); KB binds a forced-EXTERNAL
// semaphore (so KA stays cached) and syncs on slot 2 with a DIFFERENT thread count. If the seeder
// ever moved back onto g_kernel_barrier[2], the mixed participant groups would hang KA at entry or
// release a hart before the seed store lands -- so this must complete, stay cached, and count exactly.
TEST_F(SemScopeFixture, TestCachedSeederImmuneToUserBarrierSlots) {
    // Same geometry as TestCensusTwoCachedSemsOneNodePicksExternal: threads_b >= 2 and != threads_a.
    if (num_dms_ < 5) {
        GTEST_SKIP() << "needs >= 5 user DMs for two multi-threaded kernels with different thread counts";
    }
    std::vector<uint32_t> sentinel(4, kNoReport);
    tt::tt_metal::detail::WriteToDeviceL1(device_, core, report_addr, sentinel);

    const experimental::SemaphoreSpecName SEM_A{"sem_cached"};
    const experimental::SemaphoreSpecName SEM_B{"sem_external"};
    experimental::SemaphoreSpec sem_a{.unique_id = SEM_A, .target_nodes = core};
    experimental::SemaphoreSpec sem_b{.unique_id = SEM_B, .target_nodes = core};
    sem_b.scope = experimental::SemaphoreScope::EXTERNAL;  // keeps KB off the cached census
    const experimental::KernelSpecName KA{"cached_binder"};
    const experimental::KernelSpecName KB{"slot2_user"};
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
                {.runtime_arg_names = {"report_addr", "increment_times", "is_reporter", "barrier_idx"}},
            .hw_config = experimental::DataMovementGen2Config{},
        };
    };
    experimental::WorkUnitSpec wu{.name = "wu", .kernels = {KA, KB}, .target_nodes = core};
    experimental::ProgramSpec spec{
        .name = "cached_seeder_vs_slot2",
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
                 {"barrier_idx", 0u}}),
        },
        experimental::ProgramRunArgs::KernelRunArgs{
            .kernel = KB,
            .runtime_arg_values = experimental::MakeRuntimeArgsForSingleNode(
                core,
                {{"report_addr", report_addr},
                 {"increment_times", concurrent_iterations},
                 {"is_reporter", 0u},
                 {"barrier_idx", 2u}}),  // the seeder's OLD slot -- must no longer interact
        },
    };
    experimental::SetProgramRunArgs(program, params);
    distributed::MeshWorkload workload;
    distributed::MeshCoordinate zero_coord{0, 0};
    workload.add_program(distributed::MeshCoordinateRange{zero_coord, zero_coord}, std::move(program));
    RunProgram(mesh_device_, workload);  // must COMPLETE (slot-2 seeder would hang KA at entry)

    tt::tt_metal::detail::ReadFromDeviceL1(device_, core, report_addr, 4 * sizeof(uint32_t), result);
    ASSERT_EQ(result.size(), 4u);
    ASSERT_NE(result[0], kNoReport) << "probe never reported: the reporter kernel/thread did not run";
    log_info(LogTest, "cached seeder vs user slot 2: scope={} count={}", result[0], result[1]);
    EXPECT_EQ(result[0], scope_val(SemScope::DM_LOCAL_CACHED))
        << "KA is the node's sole cached binder; KB (forced EXTERNAL) must not demote it";
    EXPECT_EQ(result[1], threads_a * concurrent_iterations)
        << "count off -> a hart left the seeder rendezvous before the seed store landed";
}

// TWO binder kernels must NOT be cached, even on one node: the pool slot is seeded by an init
// injected into each binding kernel's entry as an unsynchronised destructive store, so a co-resident
// binder could reset a counter its sibling is incrementing. Must fall through to EXTERNAL.
TEST_F(SemScopeFixture, TestCensusTwoKernelsSameNodePicksExternal) {
    const auto [scope, count] = run_census(
        core,
        {{.num_threads = 1, .increments = iterations, .reporter = true}, {.num_threads = 1, .increments = iterations}});
    (void)count;  // cross-kernel count is not barrier-synchronised; the decision is what matters here
    log_info(LogTest, "census two-kernels-same-node: scope={}", scope);
    EXPECT_EQ(scope, scope_val(SemScope::EXTERNAL))
        << "a second binder kernel would also seed (reset) the shared pool slot -> must not be cached";
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

// A WRITER on another node means the semaphore is reachable from off-node -> cached would split it.
// Two binder kernels, one off-node: EXTERNAL (the single-binder rule alone already forces it here;
// node confinement in isolation is covered by TestCensusOffNodeSoleBinderPicksExternal below).
TEST_F(SemScopeFixture, TestCensusOffNodeWriterPicksExternal) {
    if (!has_second_node()) {
        GTEST_SKIP() << "needs >= 2 worker nodes to place a binder off the semaphore's node";
    }
    const experimental::NodeCoord other = second_node();
    const auto [scope, count] = run_census(
        core,
        {{.num_threads = 1, .increments = iterations, .reporter = true},
         {.node = other, .num_threads = 1, .increments = iterations}});
    (void)count;
    log_info(LogTest, "census off-node writer: scope={}", scope);
    EXPECT_EQ(scope, scope_val(SemScope::EXTERNAL)) << "a binder off the semaphore's node must force the NoC atomic";
}

// Node confinement in ISOLATION: a single multi-threaded binder kernel on a different node than the
// semaphore. Every other cached conjunct holds (one binder kernel, multi-writer, single-cell sem, no
// node conflict), so only the node-confinement check can demote this to EXTERNAL. Count is not
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

// An OBSERVE reader must NOT inflate the writer count: 1 writer + 1 reader is still single-writer.
TEST_F(SemScopeFixture, TestCensusObserverNotCountedAsWriter) {
    const auto [scope, count] = run_census(
        core,
        {{.num_threads = 1, .increments = iterations, .reporter = true},
         {.num_threads = 1, .access = experimental::SemaphoreAccessType::OBSERVE, .increments = 0}});
    log_info(LogTest, "census writer+observer: scope={} count={}", scope, count);
    EXPECT_EQ(scope, scope_val(SemScope::LOCAL_NONATOMIC))
        << "an OBSERVE reader must not count as a writer (it would needlessly force an atomic path)";
    EXPECT_EQ(count, iterations);
}

// POSITIVE CONTROL for the AccessType->SemAccess plumbing: every other assertion here still
// passes if the host silently stops baking the access, so it is read back off the device (report
// word 3). The OBSERVE kernel must be the REPORTER: access is a property of a BINDING, so a
// writer kernel reporting on the same semaphore would correctly bake INCREMENT. Count unasserted.
TEST_F(SemScopeFixture, TestObserveBindingBakesReadOnlyAccess) {
    const auto [scope, count] = run_census(
        core,
        {{.num_threads = 1, .increments = iterations},
         {.num_threads = 1, .access = experimental::SemaphoreAccessType::OBSERVE, .increments = 0, .reporter = true}});
    (void)count;
    log_info(LogTest, "observe binding: scope={} access={}", scope, census_access_);
    EXPECT_EQ(census_access_, static_cast<uint32_t>(SemAccess::OBSERVE))
        << "an AccessType::OBSERVE binding must bake SemAccess::OBSERVE into its sem:: token; anything "
           "else means the host is not emitting the label, so every mutator static_assert is silently "
           "inert";
}

// ...and the writer half: a default binding must bake INCREMENT (mutable), or every mutating
// kernel in the tree would fail to compile; a CONSUME binding must bake CONSUME, or down() would.
TEST_F(SemScopeFixture, TestWriterBindingBakesDeclaredAccess) {
    const auto [scope, count] = run_census(core, {{.num_threads = 1, .increments = iterations, .reporter = true}});
    EXPECT_EQ(scope, scope_val(SemScope::LOCAL_NONATOMIC));
    EXPECT_EQ(count, iterations);
    EXPECT_EQ(census_access_, static_cast<uint32_t>(SemAccess::INCREMENT))
        << "a default (INCREMENT) binding must bake SemAccess::INCREMENT";

    const auto [c_scope, c_count] = run_census(
        core,
        {{.num_threads = 1, .access = experimental::SemaphoreAccessType::CONSUME, .increments = 1, .reporter = true}});
    (void)c_scope;
    (void)c_count;
    EXPECT_EQ(census_access_, static_cast<uint32_t>(SemAccess::CONSUME))
        << "a CONSUME binding must bake SemAccess::CONSUME (down() compiles only under it)";
}

// ...and an off-node OBSERVE reader also blocks the cached path: it would read that node's pool
// copy. The single-binder rule alone already forces EXTERNAL here too (a reader can't share the
// writer's kernel: double bindings are rejected), so this pins the reader rule end-to-end rather
// than in isolation.
TEST_F(SemScopeFixture, TestCensusOffNodeObserverBlocksCached) {
    if (num_dms_ < 2) {
        GTEST_SKIP() << "needs >= 2 user DMs";
    }
    if (!has_second_node()) {
        GTEST_SKIP() << "needs >= 2 worker nodes to place a reader off the semaphore's node";
    }
    const experimental::NodeCoord other = second_node();
    const auto [scope, count] = run_census(
        core,
        {{.num_threads = num_dms_, .increments = concurrent_iterations, .reporter = true},
         {.node = other, .num_threads = 1, .access = experimental::SemaphoreAccessType::OBSERVE, .increments = 0}});
    (void)count;
    log_info(LogTest, "census off-node observer: scope={}", scope);
    EXPECT_EQ(scope, scope_val(SemScope::EXTERNAL))
        << "a READER off the semaphore's node must also block the cached path (it would read a different copy)";
}

// A CONSUME binder off the semaphore's node would build and then hang (down() spins on its LOCAL
// word), so EVERY scope rejects it at build time (the FATAL sits before the scope switch); the
// three legs below cover AUTO plus the two forced escapes. increments stays 0: with
// EXPECT_ANY_THROW the program never runs. (make_program_with_forced_scope cannot express a
// CONSUME binding, so the forced legs reuse the census shape.)
TEST_F(SemScopeFixture, TestCensusOffNodeConsumerFatal) {
    if (!has_second_node()) {
        GTEST_SKIP() << "needs >= 2 worker nodes to place a consumer off the semaphore's node";
    }
    const std::vector<CensusKernel> off_node_consumer{
        {.node = second_node(),
         .num_threads = 1,
         .access = experimental::SemaphoreAccessType::CONSUME,
         .increments = 0,
         .reporter = true}};
    EXPECT_ANY_THROW(run_census(core, off_node_consumer))
        << "AUTO must reject a CONSUME binder off the semaphore's node (a guaranteed hang)";
    EXPECT_ANY_THROW(run_census(core, off_node_consumer, SemaphoreScope::EXTERNAL))
        << "forced EXTERNAL must also reject an off-node CONSUME binder";
    EXPECT_ANY_THROW(run_census(core, off_node_consumer, SemaphoreScope::LOCAL_NONATOMIC))
        << "forced LOCAL_NONATOMIC must also reject it: the hang is mechanical, not an atomicity "
           "trade-off the caller can own";
}

// Explicit scopes must still win over the classifier.
TEST_F(SemScopeFixture, TestCensusForcedScopeOverridesAuto) {
    const auto [ext_scope, ext_count] =
        run_census(core, {{.num_threads = 1, .increments = iterations, .reporter = true}}, SemaphoreScope::EXTERNAL);
    EXPECT_EQ(ext_scope, scope_val(SemScope::EXTERNAL)) << "forced EXTERNAL must override the AUTO cheap pick";
    EXPECT_EQ(ext_count, iterations);

    const auto [loc_scope, loc_count] = run_census(
        core,
        {{.num_threads = num_dms_, .increments = concurrent_iterations, .reporter = true}},
        SemaphoreScope::LOCAL_NONATOMIC);
    EXPECT_EQ(loc_scope, scope_val(SemScope::LOCAL_NONATOMIC))
        << "forced LOCAL_NONATOMIC must override the AUTO atomic pick (explicit escape hatch)";
    (void)loc_count;  // deliberately non-atomic here; the count may legitimately be short

    // Forced CACHED must bake as cached: every other forced-cached test is count-only, and
    // EXTERNAL would reproduce those counts exactly -- only this readback pins the resolution.
    if (num_dms_ >= 2) {
        const auto [cached_scope, cached_count] = run_census(
            core,
            {{.num_threads = 2, .increments = concurrent_iterations, .reporter = true}},
            SemaphoreScope::DM_LOCAL_CACHED);
        EXPECT_EQ(cached_scope, scope_val(SemScope::DM_LOCAL_CACHED))
            << "forced DM_LOCAL_CACHED must bake as cached, not silently map to another mechanism";
        EXPECT_EQ(cached_count, 2 * concurrent_iterations);
    }
}

// ENFORCEMENT NEGATIVE: a kernel that down()s under an INCREMENT-labeled binding must fail at
// JIT compile (Semaphore::down() static_asserts on SemAccess::CONSUME). Without this, the whole
// access-enforcement could silently rot into accept-everything.
TEST_F(SemScopeFixture, TestDownUnderIncrementLabelFailsToCompile) {
    EXPECT_ANY_THROW(run_scope(SemaphoreScope::AUTO, /*with_down=*/true, experimental::SemaphoreAccessType::INCREMENT))
        << "a down() through an INCREMENT-labeled binding compiled -- the SemAccess mutator "
           "static_asserts are inert and the census labels are unenforced again";
}

// The two forced-cached guards that had no negative test: a second binder KERNEL on the same
// semaphore, and a second forced-cached SEMAPHORE on the same node, must each FATAL at build.
TEST_F(SemScopeFixture, TestForcedCachedTwoBinderKernelsFatal) {
    EXPECT_ANY_THROW(run_census(
        core,
        {{.num_threads = 1, .increments = 1, .reporter = true}, {.num_threads = 1, .increments = 1}},
        SemaphoreScope::DM_LOCAL_CACHED))
        << "forced DM_LOCAL_CACHED bound by TWO kernels must FATAL: each kernel's entry re-seed "
           "would reset a counter its sibling is incrementing";
}

TEST_F(SemScopeFixture, TestForcedCachedNodeConflictFatal) {
    const experimental::SemaphoreSpecName SEM_A{"fc_a"};
    const experimental::SemaphoreSpecName SEM_B{"fc_b"};
    experimental::SemaphoreSpec sem_a{.unique_id = SEM_A, .target_nodes = core};
    experimental::SemaphoreSpec sem_b{.unique_id = SEM_B, .target_nodes = core};
    sem_a.scope = experimental::SemaphoreScope::DM_LOCAL_CACHED;
    sem_b.scope = experimental::SemaphoreScope::DM_LOCAL_CACHED;
    auto make_k = [&](const char* name, const experimental::SemaphoreSpecName& sem_name) {
        return experimental::KernelSpec{
            .unique_id = experimental::KernelSpecName{name},
            .source = kernel_path_census,
            .num_threads = 1,
            .semaphore_bindings = {{.semaphore_spec_name = sem_name, .accessor_name = "counter"}},
            .runtime_arg_schema =
                {.runtime_arg_names = {"report_addr", "increment_times", "is_reporter", "barrier_idx"}},
            .hw_config = experimental::DataMovementGen2Config{},
        };
    };
    experimental::WorkUnitSpec wu{
        .name = "wu",
        .kernels = {experimental::KernelSpecName{"fc_ka"}, experimental::KernelSpecName{"fc_kb"}},
        .target_nodes = core};
    experimental::ProgramSpec spec{
        .name = "forced_cached_conflict",
        .kernels = {make_k("fc_ka", SEM_A), make_k("fc_kb", SEM_B)},
        .semaphores = {sem_a, sem_b},
        .work_units = {wu}};
    EXPECT_ANY_THROW({ auto program = experimental::MakeProgramFromSpec(*mesh_device_, spec); })
        << "two kernels forcing cached semaphores on ONE node must FATAL: their injected seeders "
           "share the node-wide rendezvous";
}

// ---- CONSUME / SET census behavior ----
// The multi-consumer FATAL is gone: down() is multi-consumer-safe on both atomic tiers (cached
// CAS retry loop; EXTERNAL NoC-CAS lock), so those shapes resolve freely and these tests pin the
// RESOLUTIONS instead. An off-node consumer still FATALs (see TestCensusOffNodeConsumerFatal),
// and the SET guard stays mechanism-independent (no scope can make a destructive store atomic).
// Nothing in the tree labels CONSUME/SET today, so these tests are the rot protection.

// The 2-thread single-kernel CONSUME shape passes cached_geometry_ok, so only the old
// pre-decision FATAL could have blocked it: it must now BUILD and resolve DM_LOCAL_CACHED
// (whose CAS down() is multi-consumer-safe).
TEST_F(SemScopeFixture, TestCensusMultiConsumerCachedShape) {
    if (num_dms_ < 2) {
        GTEST_SKIP() << "needs >= 2 user DMs";
    }
    const auto [scope, count] = run_census(
        core,
        {{.num_threads = 2, .access = experimental::SemaphoreAccessType::CONSUME, .increments = 1, .reporter = true}});
    log_info(LogTest, "census multi-consumer cached shape: scope={} count={}", scope, count);
    EXPECT_EQ(scope, scope_val(SemScope::DM_LOCAL_CACHED))
        << "a multi-CONSUME shape confined to one DM kernel on the semaphore's node is CAS-safe and must "
           "resolve DM_LOCAL_CACHED -- a different scope here means the census still rejects or misroutes "
           "the shape the moved FATAL was supposed to unblock";
    EXPECT_EQ(count, 2u) << "the cached AMO lost an update, or the pool was not initialised";
}

// A >=2-CONSUME shape that FAILS cached geometry (two 1-thread CONSUME kernels on one node ->
// two binder kernels) resolves to EXTERNAL -- whose down() is now the NoC-CAS lock, so the
// shape BUILDS and runs instead of FATALing.
TEST_F(SemScopeFixture, TestCensusMultiConsumerExternalShapeResolves) {
    const auto [scope, count] = run_census(
        core,
        {{.num_threads = 1, .access = experimental::SemaphoreAccessType::CONSUME, .increments = 1, .reporter = true},
         {.num_threads = 1, .access = experimental::SemaphoreAccessType::CONSUME, .increments = 1}});
    (void)count;
    log_info(LogTest, "census multi-consumer external shape: scope={}", scope);
    EXPECT_EQ(scope, scope_val(SemScope::EXTERNAL))
        << "two CONSUME binder kernels fail cached geometry and must resolve to the (now "
           "multi-consumer-safe) EXTERNAL mechanism";
}

TEST_F(SemScopeFixture, TestCensusSetHonestyFatal) {
    // A SET racing another writer -> set() is a non-atomic destructive store under every scope.
    EXPECT_ANY_THROW(run_census(
        core,
        {{.num_threads = 1, .access = experimental::SemaphoreAccessType::SET, .increments = 1, .reporter = true},
         {.num_threads = 1, .increments = 1}}))
        << "AUTO must reject a SET racing another writer (set() is non-atomic under every mechanism; "
           "forced scopes remain the escape for phase-separated init-then-write)";

    // A SINGLE consumer instance is fine: no concurrency, so it takes the cheap path.
    EXPECT_NO_THROW(run_census(
        core,
        {{.num_threads = 1, .access = experimental::SemaphoreAccessType::CONSUME, .increments = 0, .reporter = true}}))
        << "a single consumer is safe and must not be rejected";

    // The guards are AUTO-only: an explicitly forced scope is the user's call to make. Probed with
    // the SET+writer shape -- the multi-CONSUME cached shape no longer throws under AUTO, so it can
    // no longer demonstrate the bypass.
    EXPECT_NO_THROW(run_census(
        core,
        {{.num_threads = 1, .access = experimental::SemaphoreAccessType::SET, .increments = 1, .reporter = true},
         {.num_threads = 1, .increments = 1}},
        SemaphoreScope::EXTERNAL))
        << "forced scopes bypass the AUTO-only honesty guards";
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
        .runtime_arg_schema = {.runtime_arg_names = {"report_addr", "increment_times", "is_reporter", "barrier_idx"}},
        .hw_config = experimental::DataMovementGen2Config{},
    };
    experimental::WorkUnitSpec wu{.name = "main", .kernels = {K}, .target_nodes = core};
    experimental::ProgramSpec spec{.name = "sem_double_bind", .kernels = {ks}, .semaphores = {sem}, .work_units = {wu}};
    EXPECT_ANY_THROW({
        Program program = experimental::MakeProgramFromSpec(*mesh_device_, spec);
        (void)program;
    }) << "binding the same semaphore twice in one kernel must be rejected (it double-counts the census)";
}

// ---- Host-side scope resolution + contradiction FATALs ----

// Explicit single-node scopes are accepted (ResolveSemaphoreScope validates, no throw).
TEST_F(SemScopeFixture, TestForcedScopeSingleNodeAccepted) {
    EXPECT_NO_THROW(make_program_with_forced_scope(SemaphoreScope::EXTERNAL, core));
    EXPECT_NO_THROW(make_program_with_forced_scope(SemaphoreScope::DM_LOCAL_CACHED, core));
}

// NOTE: no "compute kernel binds a cached semaphore" test -- Metal 2.0 forbids semaphore bindings
// on compute kernels outright (ValidateProgramSpec), so a binder is always a data-movement kernel.

// The pool is per-core, so a binder running on a node OTHER than the semaphore's node would
// increment its own node's copy -> silent split -> host FATAL at config time.
TEST_F(SemScopeFixture, TestForcedDmLocalCachedRemoteBinderFatal) {
    if (!has_second_node()) {
        GTEST_SKIP() << "needs >= 2 worker nodes to place a binder off the semaphore's node";
    }
    // Semaphore stays on a single node (core), but its binding kernel spans two nodes.
    kernel_target_ = experimental::NodeRange{experimental::NodeCoord{0, 0}, second_node()};
    EXPECT_ANY_THROW(make_program_with_forced_scope(SemaphoreScope::DM_LOCAL_CACHED, core));
    // Sanity: the same spread-out binding is fine on the NoC-atomic path.
    EXPECT_NO_THROW(make_program_with_forced_scope(SemaphoreScope::EXTERNAL, core));
}

// A forced DM_LOCAL_CACHED semaphore that spans >1 node is a contradiction -> host FATAL.
TEST_F(SemScopeFixture, TestForcedDmLocalCachedMultiNodeFatal) {
    if (!has_second_node()) {
        GTEST_SKIP() << "needs >= 2 worker nodes to form a multi-node semaphore range";
    }
    const experimental::NodeRange two_nodes{experimental::NodeCoord{0, 0}, second_node()};
    EXPECT_ANY_THROW(make_program_with_forced_scope(SemaphoreScope::DM_LOCAL_CACHED, two_nodes));
}

}  // namespace tt::tt_metal
