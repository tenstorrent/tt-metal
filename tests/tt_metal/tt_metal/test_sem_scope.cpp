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
    const std::string kernel_path = "tests/tt_metal/tt_metal/test_kernels/dataflow/sem_scope_smoke.cpp";
    uint32_t report_addr{0};
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
    }

    // Runs the smoke kernel with the given scope define and returns the value the
    // kernel read back from the semaphore after `iterations` increments.
    uint32_t run_scope(const std::string& scope_define) {
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

        const experimental::KernelSpecName DM_KERNEL{"sem_scope_kernel"};
        experimental::KernelSpec kernel_spec{
            .unique_id = DM_KERNEL,
            .source = kernel_path,
            .num_threads = 1,
            .compiler_options = {.defines = {{scope_define, "1"}}},
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
};

// EXTERNAL scope: local up() is a self-targeted NoC atomic increment; value() reads
// via the uncached alias. Single writer -> value() == iterations.
TEST_F(SemScopeFixture, TestExternalScopeIncrement) {
    const uint32_t observed = run_scope("SEM_SCOPE_EXTERNAL");
    log_info(LogTest, "EXTERNAL scope value(): {} (expected {})", observed, iterations);
    EXPECT_EQ(observed, iterations)
        << "Semaphore<EXTERNAL>::up()/value() did not produce the expected single-writer count.";
}

// DM_LOCAL_CACHED scope: local up() is a 32-bit RISC-V AMO on the cached alias;
// value() reads via the cached alias. Single writer -> value() == iterations.
TEST_F(SemScopeFixture, TestDmLocalCachedScopeIncrement) {
    const uint32_t observed = run_scope("SEM_SCOPE_DM_LOCAL_CACHED");
    log_info(LogTest, "DM_LOCAL_CACHED scope value(): {} (expected {})", observed, iterations);
    EXPECT_EQ(observed, iterations)
        << "Semaphore<DM_LOCAL_CACHED>::up()/value() did not produce the expected single-writer count.";
}

// LOCAL_NONATOMIC scope: the legacy default (plain L1 read-modify-write). The define
// below is ignored by the kernel's scope ladder, so it falls through to
// LOCAL_NONATOMIC. Single writer -> value() == iterations. This also confirms the
// default path (used by existing DFB/CCL/SDPA callers) still compiles + works.
TEST_F(SemScopeFixture, TestLocalNonatomicScopeIncrement) {
    const uint32_t observed = run_scope("SEM_SCOPE_LEGACY");
    log_info(LogTest, "LOCAL_NONATOMIC scope value(): {} (expected {})", observed, iterations);
    EXPECT_EQ(observed, iterations)
        << "Semaphore<LOCAL_NONATOMIC>::up()/value() (legacy default) did not produce the expected count.";
}

}  // namespace tt::tt_metal
