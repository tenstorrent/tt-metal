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
#include <tt-metalium/experimental/metal2_host_api/program.hpp>

namespace tt::tt_metal {

// ============================================================================
// Phase-1 SemScope skeleton validation.
// ============================================================================
//
// Exercises the scoped device Semaphore class (noc_semaphore.h) end-to-end: a
// single DM thread constructs Semaphore<TENSIX, Scope> over a bound semaphore,
// increments it N times via up(), reads it back with value(), and reports the
// observed value. This confirms the DM_LOCAL_CACHED (32-bit AMO) and EXTERNAL
// (self-targeted NoC atomic) code paths compile and produce the correct
// single-writer count. Multi-writer atomicity of the underlying mechanisms is
// covered by the keystone tests (NocSelfAtomicFixture).
// ============================================================================
class SemScopeFixture : public MeshDispatchFixture {
protected:
    static constexpr experimental::NodeCoord core = {0, 0};
    static constexpr uint32_t iterations{64};
    static constexpr uint32_t concurrent_iterations{20};  // per-thread; NoC round-trips are slow on emu
    const std::string kernel_path = "tests/tt_metal/tt_metal/test_kernels/dataflow/sem_scope_smoke.cpp";
    const std::string kernel_path_concurrent =
        "tests/tt_metal/tt_metal/test_kernels/dataflow/sem_scope_concurrent.cpp";
    const std::string kernel_path_coexist =
        "tests/tt_metal/tt_metal/test_kernels/dataflow/sem_scope_coexist.cpp";
    const std::string kernel_path_census =
        "tests/tt_metal/tt_metal/test_kernels/dataflow/sem_census_probe.cpp";
    uint32_t report_addr{0};
    uint32_t num_dms_{0};
    bool is_quasar{false};
    std::shared_ptr<distributed::MeshDevice> mesh_device_;
    IDevice* device_{nullptr};
    std::vector<uint32_t> result;

    void SetUp() override {
        MeshDispatchFixture::SetUp();
        // Every spec in this file uses DataMovementGen2Config, so the whole suite is Gen2 (Quasar)
        // only -- on Gen1 the specs themselves would be rejected, which would look like test failures
        // rather than an unsupported configuration.
        if (arch_ != tt::ARCH::QUASAR) {
            GTEST_SKIP() << "SemScope suite is Gen2 (Quasar) only: its specs use DataMovementGen2Config";
        }
        mesh_device_ = devices_[0];
        device_ = mesh_device_->get_devices()[0];
        report_addr = device_->allocator()->get_base_allocator_addr(HalMemType::L1);
        num_dms_ = MetalContext::instance().hal().get_processor_types_count(HalProgrammableCoreType::TENSIX, 0);
        is_quasar = arch_ == tt::ARCH::QUASAR;
        if (is_quasar) {
            num_dms_ = std::min(num_dms_, 6u);  // Metal 2.0 reserves DM0/DM1
        }
    }

    // Runs the smoke kernel with the given host-baked scope and returns the value the
    // kernel read back from the semaphore after `iterations` increments. The scope is set on
    // the SemaphoreSpec; the kernel is scope-agnostic and picks it up via CTAD on the emitted
    // sem::counter token.
    uint32_t run_scope(SemaphoreScope scope, bool with_down = false) {
        // Zero the report scratch word.
        std::vector<uint32_t> zero(1, 0);
        tt::tt_metal::detail::WriteToDeviceL1(device_, core, report_addr, zero);

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
                {{.semaphore_spec_name = experimental::SemaphoreSpecName{"counter_sem"}, .accessor_name = "counter"}},
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
        return result.empty() ? 0u : result[0];
    }

    // Multi-DM concurrency proof (Quasar-only; roles gated by mhartid in the kernel).
    // num_dms threads share one bound Semaphore<scope>; mode picks CONCURRENT_UP or
    // PRODUCER_CONSUMER. Returns the value() the reporter/consumer read back.
    uint32_t run_concurrent(SemaphoreScope scope, const std::string& mode_define) {
        std::vector<uint32_t> zero(1, 0);
        tt::tt_metal::detail::WriteToDeviceL1(device_, core, report_addr, zero);

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

        tt::tt_metal::detail::ReadFromDeviceL1(device_, core, report_addr, sizeof(uint32_t), result);
        EXPECT_EQ(result.size(), 1u);
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

    // ---- Census / AUTO-classifier probe harness ----
    // One binding kernel in a census shape. All kernels are single-node so per-node runtime args stay
    // simple; an "off-node binder" shape is built by adding a second kernel on another node.
    struct CensusKernel {
        experimental::NodeCoord node{core};
        uint32_t num_threads = 1;
        experimental::SemaphoreAccessType access = experimental::SemaphoreAccessType::INCREMENT;
        uint32_t increments = 0;  // 0 for an OBSERVE reader (it must not write)
        bool reporter = false;    // exactly one kernel should report
    };

    // Build + run a program with the given census shape; returns {baked_scope, counter_value}.
    // The baked scope is read back from the device, so it is the classifier's ACTUAL decision as
    // compiled into the kernel -- not an inference from the resulting counts (every correct mechanism
    // yields the same counts, so a count-only assertion cannot detect a wrong pick).
    std::pair<uint32_t, uint32_t> run_census(
        const experimental::Nodes& sem_target,
        const std::vector<CensusKernel>& kernels,
        SemaphoreScope scope = SemaphoreScope::AUTO) {
        // Prefill with a sentinel OUTSIDE the SemScope range: LOCAL_NONATOMIC == 0, so a zero prefill
        // would be indistinguishable from "the probe never reported" (e.g. the reporter never ran).
        std::vector<uint32_t> sentinel(3, kNoReport);
        tt::tt_metal::detail::WriteToDeviceL1(device_, core, report_addr, sentinel);

        distributed::MeshWorkload workload;
        distributed::MeshCoordinate zero_coord{0, 0};
        distributed::MeshCoordinateRange device_range{zero_coord, zero_coord};

        experimental::SemaphoreSpec sem{
            .unique_id = experimental::SemaphoreSpecName{"counter_sem"}, .target_nodes = sem_target};
        sem.scope = scope;

        std::vector<experimental::KernelSpec> kernel_specs;
        experimental::ProgramRunArgs params;
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
                .runtime_arg_schema = {.runtime_arg_names = {"report_addr", "increment_times", "is_reporter", "barrier_idx"}},
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
                     // Distinct barrier slot per kernel; slot 2 is reserved for the cached-sem seeder.
                     {"barrier_idx", static_cast<uint32_t>(i)}}),
            });
        }
        std::vector<experimental::WorkUnitSpec> work_units;
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

        tt::tt_metal::detail::ReadFromDeviceL1(device_, core, report_addr, 3 * sizeof(uint32_t), result);
        EXPECT_EQ(result.size(), 3u);
        if (result.size() < 3) {
            return {kNoReport, 0u};
        }
        census_ring_word_ = result[2];
        // Catch "the probe never reported" instead of silently reading it as LOCAL_NONATOMIC(0).
        EXPECT_NE(result[0], kNoReport) << "census probe never reported: the reporter kernel/thread did not run";
        return {result[0], result[1]};
    }

    // A node other than `core`, on whichever axis the device actually exposes. The off-node shapes
    // used to hardcode {1,0} and skip on grid.x < 2, which silently dropped their coverage on a
    // 1-wide (but taller) grid.
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

    // Where the binding kernel is placed (defaults to the semaphore's node). Set wider to build the
    // "a binder runs outside the semaphore's node" case for the cached-pool guard.
    experimental::Nodes kernel_target_{core};

    // Build (but do not run) a minimal ProgramSpec whose semaphore carries the given scope
    // intent on the given target, so ValidateProgramSpec's ResolveSemaphoreScope runs.
    // Throws (TT_FATAL) on a contradiction. No JIT/emu-kernel execution.
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
// callers, and by every AUTO-resolved semaphore) still compiles + works after the token flip.
TEST_F(SemScopeFixture, TestLocalNonatomicScopeIncrement) {
    const uint32_t observed = run_scope(SemaphoreScope::LOCAL_NONATOMIC);
    log_info(LogTest, "LOCAL_NONATOMIC scope value(): {} (expected {})", observed, iterations);
    EXPECT_EQ(observed, iterations)
        << "Semaphore<LOCAL_NONATOMIC>::up()/value() (legacy default) did not produce the expected count.";
}

// up(N) then down(N) must leave the semaphore at 0, per scope. For EXTERNAL this
// exercises the atomic NoC decrement (INCR_GET of a negative value); for
// DM_LOCAL_CACHED the AMO subtract; for LOCAL_NONATOMIC the legacy decrement.
TEST_F(SemScopeFixture, TestExternalScopeUpDown) {
    const uint32_t observed = run_scope(SemaphoreScope::EXTERNAL, /*with_down=*/true);
    log_info(LogTest, "EXTERNAL up/down value(): {} (expected 0)", observed);
    EXPECT_EQ(observed, 0u) << "Semaphore<EXTERNAL>::down() (atomic NoC decrement) did not return to 0.";
}

TEST_F(SemScopeFixture, TestDmLocalCachedScopeUpDown) {
    const uint32_t observed = run_scope(SemaphoreScope::DM_LOCAL_CACHED, /*with_down=*/true);
    log_info(LogTest, "DM_LOCAL_CACHED up/down value(): {} (expected 0)", observed);
    EXPECT_EQ(observed, 0u) << "Semaphore<DM_LOCAL_CACHED>::down() (AMO subtract) did not return to 0.";
}

TEST_F(SemScopeFixture, TestLocalNonatomicScopeUpDown) {
    const uint32_t observed = run_scope(SemaphoreScope::LOCAL_NONATOMIC, /*with_down=*/true);
    log_info(LogTest, "LOCAL_NONATOMIC up/down value(): {} (expected 0)", observed);
    EXPECT_EQ(observed, 0u) << "Semaphore<LOCAL_NONATOMIC>::down() (legacy) did not return to 0.";
}

// NOTE: token-CTAD deduction (the S1 mechanism) is now exercised by EVERY test above and
// below — after the S2b emitter flip, sem::counter IS a SemaphoreBindingToken<id, baked-scope> token and
// the kernels construct via plain `Semaphore s(sem::counter)`, so the standalone
// TestTokenCtadDeduction became redundant and was removed.

// ---- Concurrency proofs (Quasar-only): the single-writer tests above cannot tell an
// atomic path from a non-atomic one; these do. ----

// G1: all num_dms DMs concurrently up(1) a shared Semaphore. Exact num_dms*iters proves
// up() routes to the atomic mechanism (a non-atomic up() would lose updates -> less).
TEST_F(SemScopeFixture, TestExternalConcurrentUp) {
    if (!is_quasar) {
        GTEST_SKIP() << "concurrency proof gates DM roles by mhartid (Quasar-only)";
    }
    const uint32_t observed = run_concurrent(SemaphoreScope::EXTERNAL, "MODE_CONCURRENT_UP");
    const uint32_t expected = num_dms_ * concurrent_iterations;
    log_info(LogTest, "EXTERNAL concurrent up value(): {} (expected {})", observed, expected);
    EXPECT_EQ(observed, expected) << "Semaphore<EXTERNAL>::up() lost updates under concurrency (non-atomic route?).";
}

TEST_F(SemScopeFixture, TestDmLocalCachedConcurrentUp) {
    if (!is_quasar) {
        GTEST_SKIP() << "concurrency proof gates DM roles by mhartid (Quasar-only)";
    }
    const uint32_t observed = run_concurrent(SemaphoreScope::DM_LOCAL_CACHED, "MODE_CONCURRENT_UP");
    const uint32_t expected = num_dms_ * concurrent_iterations;
    log_info(LogTest, "DM_LOCAL_CACHED concurrent up value(): {} (expected {})", observed, expected);
    EXPECT_EQ(observed, expected) << "Semaphore<DM_LOCAL_CACHED>::up() lost updates under concurrency.";
}

// G2: (num_dms-1) producers up(1) while a SINGLE consumer down(1)s them all. Exact 0 proves
// down()'s decrement is atomic vs concurrent producer increments (a non-atomic down() loses
// producer increments -> the consumer starves and the test times out).
TEST_F(SemScopeFixture, TestExternalProducerConsumer) {
    if (!is_quasar) {
        GTEST_SKIP() << "concurrency proof gates DM roles by mhartid (Quasar-only)";
    }
    const uint32_t observed = run_concurrent(SemaphoreScope::EXTERNAL, "MODE_PRODUCER_CONSUMER");
    log_info(LogTest, "EXTERNAL producer/consumer value(): {} (expected 0)", observed);
    EXPECT_EQ(observed, 0u) << "Semaphore<EXTERNAL>::down() lost a concurrent producer increment (non-atomic?).";
}

TEST_F(SemScopeFixture, TestDmLocalCachedProducerConsumer) {
    if (!is_quasar) {
        GTEST_SKIP() << "concurrency proof gates DM roles by mhartid (Quasar-only)";
    }
    const uint32_t observed = run_concurrent(SemaphoreScope::DM_LOCAL_CACHED, "MODE_PRODUCER_CONSUMER");
    log_info(LogTest, "DM_LOCAL_CACHED producer/consumer value(): {} (expected 0)", observed);
    EXPECT_EQ(observed, 0u) << "Semaphore<DM_LOCAL_CACHED>::down() lost a concurrent producer increment.";
}

// ---- S3: dedicated cached-only pool coexistence proof ----

// A DM_LOCAL_CACHED sem (pool) + an EXTERNAL sem (ring) hammered concurrently by all DMs in one
// program. The pool is physically disjoint from the NoC-written ring (MEM_DM_CACHED_SEM_BASE <
// MEM_MAP_END <= kernel_config ring base), so the cached AMO's 64B-line write-back cannot clobber
// the external sem's ring word (nor vice versa). Both counts exact => coexistence works.
TEST_F(SemScopeFixture, TestCachedExternalCoexistence) {
    if (!is_quasar) {
        GTEST_SKIP() << "cached pool + AMO + mhartid roles are Quasar-only";
    }
    const auto [cached_val, external_val] = run_coexist();
    const uint32_t expected = num_dms_ * concurrent_iterations;
    log_info(
        LogTest, "coexistence: cached={} external={} (each expected {})", cached_val, external_val, expected);
    EXPECT_EQ(cached_val, expected)
        << "DM_LOCAL_CACHED (pool) count wrong -> cached AMO lost updates or the pool was not initialised.";
    EXPECT_EQ(external_val, expected)
        << "EXTERNAL (ring) count wrong -> the cached sem's dirty-line write-back clobbered the NoC-written "
           "ring word (pool separation FAILED).";
}

// ---- S5: AUTO classifier behavior (auto-select LOCAL_NONATOMIC vs EXTERNAL) ----

// AUTO on a MULTI-writer semaphore must resolve to an ATOMIC mechanism. All DMs concurrently up() an
// AUTO-scoped semaphore; exact num_dms*iters proves atomicity (a wrong LOCAL_NONATOMIC pick loses
// updates under contention -> short count). Which atomic mechanism it picks depends on confinement --
// here one multi-threaded kernel on one node, so it is the cached pool; TestCensus* assert the
// specific decision. This test's job is the end-to-end atomicity guarantee.
TEST_F(SemScopeFixture, TestAutoMultiWriterIsAtomic) {
    if (!is_quasar) {
        GTEST_SKIP() << "concurrency proof gates DM roles by mhartid (Quasar-only)";
    }
    const uint32_t observed = run_concurrent(SemaphoreScope::AUTO, "MODE_CONCURRENT_UP");
    const uint32_t expected = num_dms_ * concurrent_iterations;
    log_info(LogTest, "AUTO multi-writer up value(): {} (expected {})", observed, expected);
    EXPECT_EQ(observed, expected)
        << "AUTO on a multi-writer semaphore did not resolve to an atomic mechanism (lost updates => the "
           "classifier wrongly picked LOCAL_NONATOMIC).";
}

// AUTO on a SINGLE-writer semaphore resolves to the cheap LOCAL_NONATOMIC path (single-thread,
// single-node). Single writer -> value() == iterations; confirms the cheap path runs correctly.
TEST_F(SemScopeFixture, TestAutoSingleWriterResolvesLocal) {
    const uint32_t observed = run_scope(SemaphoreScope::AUTO);
    log_info(LogTest, "AUTO single-writer value(): {} (expected {})", observed, iterations);
    EXPECT_EQ(observed, iterations) << "AUTO on a single-writer semaphore did not produce the expected count.";
}

// ============================================================================
// AUTO-classifier / census stress suite.
// ============================================================================
// Each test builds a distinct census shape and asserts the scope the host ACTUALLY baked into the
// kernel (read back from the device), so a regression in the classifier or the census arithmetic
// fails loudly instead of hiding behind counts that look right under any correct mechanism.
// Second node used for the off-node shapes; those tests skip on a 1-wide grid.
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
    if (!is_quasar) {
        GTEST_SKIP() << "multi-threaded DM kernels + the cached pool are Quasar-only";
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
    if (!is_quasar) {
        GTEST_SKIP() << "the cached pool is Quasar-only";
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

// TWO DISTINCT cached-eligible semaphores, each with its own single binder kernel, co-resident on one
// node. Each passes the per-semaphore single-binder-kernel rule, but the injected pool seeder
// rendezvouses on ONE node-wide barrier slot -- so caching both would mix rendezvous groups (unequal
// thread counts hang at kernel entry; equal counts let a seed land after an increment). Both must be
// demoted to EXTERNAL. Uses different thread counts, which is the hanging variant.
TEST_F(SemScopeFixture, TestCensusTwoCachedSemsOneNodePicksExternal) {
    if (!is_quasar) {
        GTEST_SKIP() << "the cached pool is Quasar-only";
    }
    if (num_dms_ < 3) {
        GTEST_SKIP() << "needs >= 3 user DMs to give the two kernels different thread counts";
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
    const uint32_t threads_b = num_dms_ - threads_a;  // different count -> the hanging variant

    auto make_k = [&](const experimental::KernelSpecName& name,
                      const experimental::SemaphoreSpecName& sem_name,
                      uint32_t threads) {
        return experimental::KernelSpec{
            .unique_id = name,
            .source = kernel_path_census,
            .num_threads = threads,
            .semaphore_bindings = {{.semaphore_spec_name = sem_name, .accessor_name = "counter"}},
            .runtime_arg_schema = {.runtime_arg_names = {"report_addr", "increment_times", "is_reporter", "barrier_idx"}},
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

    tt::tt_metal::detail::ReadFromDeviceL1(device_, core, report_addr, 3 * sizeof(uint32_t), result);
    ASSERT_EQ(result.size(), 3u);
    ASSERT_NE(result[0], kNoReport) << "reporter never ran (the program hung or the seeder deadlocked)";
    log_info(LogTest, "two cached sems on one node: scope={} count={}", result[0], result[1]);
    EXPECT_EQ(result[0], scope_val(SemScope::EXTERNAL))
        << "two co-resident cached-binder kernels share one node-wide seeder barrier slot -> both must "
           "be demoted to EXTERNAL, not cached";
    EXPECT_EQ(result[1], threads_a * concurrent_iterations);
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

// ...but an OBSERVE reader DOES count for node confinement: a reader on another node would read that
// node's pool copy, so the semaphore must not be cached.
TEST_F(SemScopeFixture, TestCensusOffNodeObserverBlocksCached) {
    if (!has_second_node()) {
        GTEST_SKIP() << "needs >= 2 worker nodes to place a reader off the semaphore's node";
    }
    const experimental::NodeCoord other = second_node();
    const auto [scope, count] = run_census(
        core,
        {{.num_threads = num_dms_, .increments = concurrent_iterations, .reporter = true},
         {.node = other,
          .num_threads = 1,
          .access = experimental::SemaphoreAccessType::OBSERVE,
          .increments = 0}});
    (void)count;
    log_info(LogTest, "census off-node observer: scope={}", scope);
    EXPECT_EQ(scope, scope_val(SemScope::EXTERNAL))
        << "a READER off the semaphore's node must also block the cached path (it would read a different copy)";
}

// Explicit scopes must still win over the classifier.
TEST_F(SemScopeFixture, TestCensusForcedScopeOverridesAuto) {
    const auto [ext_scope, ext_count] = run_census(
        core, {{.num_threads = 1, .increments = iterations, .reporter = true}}, SemaphoreScope::EXTERNAL);
    EXPECT_EQ(ext_scope, scope_val(SemScope::EXTERNAL)) << "forced EXTERNAL must override the AUTO cheap pick";
    EXPECT_EQ(ext_count, iterations);

    const auto [loc_scope, loc_count] = run_census(
        core,
        {{.num_threads = num_dms_, .increments = concurrent_iterations, .reporter = true}},
        SemaphoreScope::LOCAL_NONATOMIC);
    EXPECT_EQ(loc_scope, scope_val(SemScope::LOCAL_NONATOMIC))
        << "forced LOCAL_NONATOMIC must override the AUTO atomic pick (explicit escape hatch)";
    (void)loc_count;  // deliberately non-atomic here; the count may legitimately be short
}

// The two hazard FATALs are mechanism-independent: neither the NoC atomic nor the cached AMO can make
// a multi-consumer down() or a racing set() atomic. Nothing in the tree labels CONSUME/SET today, so
// without these tests both guards are dead code that could rot unnoticed.
TEST_F(SemScopeFixture, TestCensusConsumeAndSetHonestyFatals) {
    // Two concurrent CONSUME instances (one 2-thread kernel) -> check-then-decrement is unsafe.
    EXPECT_ANY_THROW(run_census(
        core,
        {{.num_threads = 2, .access = experimental::SemaphoreAccessType::CONSUME, .increments = 1, .reporter = true}}))
        << "a multi-consumer down() must be rejected under every mechanism";

    // A SET racing another writer -> set() is a non-atomic destructive store under every scope.
    EXPECT_ANY_THROW(run_census(
        core,
        {{.num_threads = 1, .access = experimental::SemaphoreAccessType::SET, .increments = 1, .reporter = true},
         {.num_threads = 1, .increments = 1}}))
        << "a SET racing another writer must be rejected under every mechanism";

    // A SINGLE consumer instance is fine: no concurrency, so it takes the cheap path.
    EXPECT_NO_THROW(run_census(
        core,
        {{.num_threads = 1, .access = experimental::SemaphoreAccessType::CONSUME, .increments = 0, .reporter = true}}))
        << "a single consumer is safe and must not be rejected";

    // The guards are AUTO-only: an explicitly forced scope is the user's call to make.
    EXPECT_NO_THROW(run_census(
        core,
        {{.num_threads = 2, .access = experimental::SemaphoreAccessType::CONSUME, .increments = 1, .reporter = true}},
        SemaphoreScope::EXTERNAL))
        << "forced scopes bypass the AUTO-only honesty guards";
}

// A kernel may bind a given semaphore only once -- a second binding would double-count that kernel's
// instances in the census and so distort the mechanism choice.
TEST_F(SemScopeFixture, TestDoubleBindingRejected) {
    experimental::SemaphoreSpec sem{
        .unique_id = experimental::SemaphoreSpecName{"counter_sem"}, .target_nodes = core};
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
    experimental::ProgramSpec spec{
        .name = "sem_double_bind", .kernels = {ks}, .semaphores = {sem}, .work_units = {wu}};
    EXPECT_ANY_THROW({
        Program program = experimental::MakeProgramFromSpec(*mesh_device_, spec);
        (void)program;
    }) << "binding the same semaphore twice in one kernel must be rejected (it double-counts the census)";
}

// ---- Phase-2 S2a: host-side scope resolution + contradiction FATALs ----

// Explicit single-node scopes are accepted (ResolveSemaphoreScope validates, no throw).
TEST_F(SemScopeFixture, TestForcedScopeSingleNodeAccepted) {
    EXPECT_NO_THROW(make_program_with_forced_scope(SemaphoreScope::EXTERNAL, core));
    EXPECT_NO_THROW(make_program_with_forced_scope(SemaphoreScope::DM_LOCAL_CACHED, core));
}

// NOTE: there is deliberately no "compute kernel binds a cached semaphore" test -- Metal 2.0 forbids
// semaphore bindings on compute kernels outright (ValidateProgramSpec), and hw_config has only
// DataMovement/Compute alternatives, so a semaphore binder is always a data-movement kernel and the
// cached path needs no compute-specific guard.

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
    kernel_target_ = experimental::Nodes{core};  // restore for other tests
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
