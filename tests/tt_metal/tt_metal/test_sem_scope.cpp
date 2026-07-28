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
    uint32_t report_addr{0};
    uint32_t num_dms_{0};
    bool is_quasar{false};
    std::shared_ptr<distributed::MeshDevice> mesh_device_;
    IDevice* device_{nullptr};
    std::vector<uint32_t> result;

    void SetUp() override {
        MeshDispatchFixture::SetUp();
        if (arch_ == tt::ARCH::WORMHOLE_B0) {
            GTEST_SKIP() << "SemScope atomic paths target Quasar/Blackhole";
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

    // Where the binding kernel is placed (defaults to the semaphore's node). Set wider to build the
    // "a binder runs outside the semaphore's node" case for the cached-pool guard.
    experimental::Nodes kernel_target_{core};

    // Build (but do not run) a minimal ProgramSpec with a COMPUTE kernel binding the semaphore, to
    // exercise the cached-pool "no compute binder" guard. Validation only; no JIT/kernel execution.
    void make_program_with_compute_binder(SemaphoreScope scope) {
        experimental::SemaphoreSpec sem{
            .unique_id = experimental::SemaphoreSpecName{"counter_sem"}, .target_nodes = core};
        sem.scope = scope;
        const experimental::KernelSpecName K{"compute_k"};
        experimental::KernelSpec ks{
            .unique_id = K,
            .source = "tests/tt_metal/tt_metal/test_kernels/compute/simple_tls_check.cpp",
            .num_threads = 1,
            .semaphore_bindings =
                {{.semaphore_spec_name = experimental::SemaphoreSpecName{"counter_sem"}, .accessor_name = "counter"}},
            .hw_config = experimental::ComputeGen2Config{},
        };
        experimental::WorkUnitSpec wu{.name = "main", .kernels = {K}, .target_nodes = core};
        experimental::ProgramSpec spec{
            .name = "sem_scope_compute_binder", .kernels = {ks}, .semaphores = {sem}, .work_units = {wu}};
        Program program = experimental::MakeProgramFromSpec(*mesh_device_, spec);
        (void)program;
    }

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
// below — after the S2b emitter flip, sem::counter IS a SemAccessor<id, baked-scope> token and
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

// AUTO on a MULTI-writer semaphore must auto-resolve to the atomic EXTERNAL path. All DMs
// concurrently up() an AUTO-scoped semaphore; exact num_dms*iters proves the classifier picked
// EXTERNAL (a wrong LOCAL_NONATOMIC pick would lose updates under contention -> short count).
TEST_F(SemScopeFixture, TestAutoMultiWriterResolvesExternal) {
    if (!is_quasar) {
        GTEST_SKIP() << "concurrency proof gates DM roles by mhartid (Quasar-only)";
    }
    const uint32_t observed = run_concurrent(SemaphoreScope::AUTO, "MODE_CONCURRENT_UP");
    const uint32_t expected = num_dms_ * concurrent_iterations;
    log_info(LogTest, "AUTO multi-writer up value(): {} (expected {})", observed, expected);
    EXPECT_EQ(observed, expected)
        << "AUTO on a multi-writer semaphore did not resolve to the atomic EXTERNAL path (lost updates "
           "=> classifier wrongly picked LOCAL_NONATOMIC).";
}

// AUTO on a SINGLE-writer semaphore resolves to the cheap LOCAL_NONATOMIC path (single-thread,
// single-node). Single writer -> value() == iterations; confirms the cheap path runs correctly.
TEST_F(SemScopeFixture, TestAutoSingleWriterResolvesLocal) {
    const uint32_t observed = run_scope(SemaphoreScope::AUTO);
    log_info(LogTest, "AUTO single-writer value(): {} (expected {})", observed, iterations);
    EXPECT_EQ(observed, iterations) << "AUTO on a single-writer semaphore did not produce the expected count.";
}

// ---- Phase-2 S2a: host-side scope resolution + contradiction FATALs ----

// Explicit single-node scopes are accepted (ResolveSemaphoreScope validates, no throw).
TEST_F(SemScopeFixture, TestForcedScopeSingleNodeAccepted) {
    EXPECT_NO_THROW(make_program_with_forced_scope(SemaphoreScope::EXTERNAL, core));
    EXPECT_NO_THROW(make_program_with_forced_scope(SemaphoreScope::DM_LOCAL_CACHED, core));
}

// The cached-only pool is in the DM cache domain, so a COMPUTE binder would read a different word
// (the kernel_config ring) and the semaphore would silently split -> host FATAL at config time.
TEST_F(SemScopeFixture, TestForcedDmLocalCachedComputeBinderFatal) {
    EXPECT_ANY_THROW(make_program_with_compute_binder(SemaphoreScope::DM_LOCAL_CACHED));
    // Sanity: the same compute binding is perfectly legal on the NoC-atomic path.
    EXPECT_NO_THROW(make_program_with_compute_binder(SemaphoreScope::EXTERNAL));
}

// The pool is per-core, so a binder running on a node OTHER than the semaphore's node would
// increment its own node's copy -> silent split -> host FATAL at config time.
TEST_F(SemScopeFixture, TestForcedDmLocalCachedRemoteBinderFatal) {
    const auto grid = mesh_device_->compute_with_storage_grid_size();
    if (grid.x < 2) {
        GTEST_SKIP() << "needs >= 2 worker nodes to place a binder off the semaphore's node";
    }
    // Semaphore stays on a single node (core), but its binding kernel spans two nodes.
    kernel_target_ = experimental::NodeRange{experimental::NodeCoord{0, 0}, experimental::NodeCoord{1, 0}};
    EXPECT_ANY_THROW(make_program_with_forced_scope(SemaphoreScope::DM_LOCAL_CACHED, core));
    // Sanity: the same spread-out binding is fine on the NoC-atomic path.
    EXPECT_NO_THROW(make_program_with_forced_scope(SemaphoreScope::EXTERNAL, core));
    kernel_target_ = experimental::Nodes{core};  // restore for other tests
}

// A forced DM_LOCAL_CACHED semaphore that spans >1 node is a contradiction -> host FATAL.
TEST_F(SemScopeFixture, TestForcedDmLocalCachedMultiNodeFatal) {
    const auto grid = mesh_device_->compute_with_storage_grid_size();
    if (grid.x < 2) {
        GTEST_SKIP() << "needs >= 2 worker nodes to form a multi-node semaphore range";
    }
    const experimental::NodeRange two_nodes{experimental::NodeCoord{0, 0}, experimental::NodeCoord{1, 0}};
    EXPECT_ANY_THROW(make_program_with_forced_scope(SemaphoreScope::DM_LOCAL_CACHED, two_nodes));
}

}  // namespace tt::tt_metal
