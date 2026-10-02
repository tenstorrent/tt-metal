// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Semaphore mechanism (SemScope) resolution for bound semaphores.

#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <unordered_map>

#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program.hpp>

#include "impl/context/metal_context.hpp"
#include "impl/metal2_host_api/semaphore_scope.hpp"
#include "metal2_host_api/test_helpers/test_helpers.hpp"
#include "metal2_host_api/test_helpers/mock_device_fixtures.hpp"

namespace tt::tt_metal::experimental {
namespace {

using test_helpers::BindSemaphoreToKernels;
using test_helpers::MakeMinimalGen1ValidProgramSpec;
using test_helpers::MakeMinimalValidProgramSpec;
using test_helpers::ProgramSpecTestBlackhole;
using test_helpers::ProgramSpecTestGen1;
using test_helpers::ProgramSpecTestQuasar;

// ResolveSemaphoreScope picks how each bound semaphore is accessed, and codegen bakes the answer
// into the kernel's binding token. A wrong answer is therefore a silently wrong *mechanism* on
// device, not a build failure -- so it needs assertions here. On Blackhole a compute binding
// compiles into three TRISC binaries with two writers (UNPACK and PACK), so a compute-bound
// semaphore must resolve to COMPUTE_ATOMIC, never to a non-atomic read-modify-write.

// Resolve one semaphore's scope from a spec the way BuildProgramFromSpec does: census the binders
// against each kernel's node set, then resolve. Placement is derived from the work units, which is
// what CollectSpecData does for kernel_node_set.
SemScope ResolveScopeFor(const ProgramSpec& spec, const char* semaphore_name) {
    std::unordered_map<KernelSpecName, NodeRangeSet> kernel_node_set;
    for (const auto& work_unit : spec.work_units) {
        const NodeRangeSet nodes = to_node_range_set(work_unit.target_nodes);
        for (const auto& kernel_name : work_unit.kernels) {
            kernel_node_set[kernel_name] = kernel_node_set[kernel_name].merge(nodes);
        }
    }
    const sem_solver::SemaphoreBinderCensus census = sem_solver::CollectSemaphoreBinders(spec, kernel_node_set);
    // Resolve against the (mock-configured) context Hal, mirroring BuildProgramFromSpec; configure_mock_mode
    // in each test fixture sets that context's arch.
    return sem_solver::ResolveSemaphoreScopes(spec, census, tt::tt_metal::MetalContext::instance().hal())
        .at(SemaphoreSpecName{semaphore_name});
}

// The headline rule: a Blackhole semaphore with a compute binder gets the atomic mechanism.
TEST_F(ProgramSpecTestBlackhole, CPU_ComputeBoundSemaphoreResolvesToComputeAtomic) {
    ProgramSpec spec = MakeMinimalGen1ValidProgramSpec();
    ASSERT_TRUE(spec.kernels[1].is_compute_kernel());
    BindSemaphoreToKernels(spec, "compute_sem", {"compute_kernel"});

    EXPECT_EQ(ResolveScopeFor(spec, "compute_sem"), SemScope::COMPUTE_ATOMIC);
}

// No compute binder, no atomics: DM-only bindings keep the pre-existing non-atomic path, so
// existing DM kernels pay nothing for this feature.
TEST_F(ProgramSpecTestBlackhole, CPU_DMOnlyBoundSemaphoreResolvesToLocalNonatomic) {
    ProgramSpec spec = MakeMinimalGen1ValidProgramSpec();
    BindSemaphoreToKernels(spec, "dm_sem", {"dm_kernel"});

    EXPECT_EQ(ResolveScopeFor(spec, "dm_sem"), SemScope::LOCAL_NONATOMIC);
}

// Wormhole is Gen1 too, but has no compute semaphore implementation, so COMPUTE_ATOMIC stays
// Blackhole-only. (A WH compute binding is separately rejected by ValidateProgramSpec; this pins the
// resolver itself, so the guard survives even if that validation is ever relaxed.)
TEST_F(ProgramSpecTestGen1, CPU_WormholeComputeBoundSemaphoreDoesNotResolveToComputeAtomic) {
    ProgramSpec spec = MakeMinimalGen1ValidProgramSpec();
    BindSemaphoreToKernels(spec, "compute_sem", {"compute_kernel"});

    EXPECT_EQ(ResolveScopeFor(spec, "compute_sem"), SemScope::LOCAL_NONATOMIC);
}

// Quasar resolves a compute-only semaphore to COMPUTE_ATOMIC ahead of the Gen2 tiers: a single compute
// binder is still two writers (UNPACK and PACK), so the one-binder LOCAL_NONATOMIC rule must not apply.
TEST_F(ProgramSpecTestQuasar, CPU_ComputeBoundSemaphoreResolvesToComputeAtomic) {
    ProgramSpec spec = MakeMinimalValidProgramSpec();
    ASSERT_TRUE(spec.kernels[1].is_compute_kernel());
    BindSemaphoreToKernels(spec, "compute_sem", {"compute_kernel"});

    EXPECT_EQ(ResolveScopeFor(spec, "compute_sem"), SemScope::COMPUTE_ATOMIC);
}

TEST_F(ProgramSpecTestQuasar, CPU_SemaphoreSharedByComputeAndDMIsRejected) {
    ProgramSpec spec = MakeMinimalValidProgramSpec();
    BindSemaphoreToKernels(spec, "shared_sem", {"dm_kernel", "compute_kernel"});

    EXPECT_THAT(
        [&] { MakeProgramFromSpec(*mesh_device_, spec); },
        ::testing::ThrowsMessage<std::runtime_error>(
            ::testing::HasSubstr("bound by both a compute kernel and a data-movement kernel")));
}

}  // namespace
}  // namespace tt::tt_metal::experimental
