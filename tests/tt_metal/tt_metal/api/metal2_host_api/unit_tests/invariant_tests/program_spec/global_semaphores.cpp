// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// ProgramSpec invariants on GlobalSemaphoreParameters and the SemaphoreBindings that reference them.

#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <stdexcept>

#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program.hpp>

#include "impl/kernels/kernel.hpp"
#include "impl/program/program_impl.hpp"
#include "metal2_host_api/test_helpers/global_semaphore_test_helpers.hpp"
#include "metal2_host_api/test_helpers/mock_device_fixtures.hpp"
#include "metal2_host_api/test_helpers/test_helpers.hpp"

namespace tt::tt_metal::experimental {
namespace {

using test_helpers::BindGlobalSemaphoreToKernel;
using test_helpers::BindSemaphoreToKernels;
using test_helpers::MakeGlobalSemaphoreSpec;
using test_helpers::ProgramSpecTestBlackhole;
using test_helpers::ProgramSpecTestGen1;

#define EXPECT_GLOBAL_SEM_SPEC_REJECTED(spec, substr)      \
    EXPECT_THAT(                                           \
        [&] { MakeProgramFromSpec(*mesh_device_, spec); }, \
        ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr(substr)))

TEST_F(ProgramSpecTestGen1, CPU_GlobalSemaphoreBindingIsValid) {
    EXPECT_NO_THROW(MakeProgramFromSpec(*mesh_device_, MakeGlobalSemaphoreSpec()));
}

// One slot per parameter, in declaration order; the table header is the slot count.
TEST_F(ProgramSpecTestGen1, CPU_OneSlotPerParameterInDeclarationOrder) {
    Program program = MakeProgramFromSpec(*mesh_device_, MakeGlobalSemaphoreSpec({"first", "second"}));
    const auto& impl = program.impl();

    ASSERT_EQ(impl.num_global_semaphore_slots(), 2u);
    ASSERT_NE(impl.get_global_semaphore_parameter("first"), nullptr);
    ASSERT_NE(impl.get_global_semaphore_parameter("second"), nullptr);
    EXPECT_EQ(impl.get_global_semaphore_parameter("first")->slot, 0u);
    EXPECT_EQ(impl.get_global_semaphore_parameter("second")->slot, 1u);
    EXPECT_FALSE(impl.get_global_semaphore_parameter("first")->bound);

    const auto& table = impl.global_semaphore_table();
    ASSERT_EQ(table.size(), 3u);
    EXPECT_EQ(table[0], 2u);
}

TEST_F(ProgramSpecTestGen1, CPU_BindingKernelGetsOneTokenPerGlobalSemaphore) {
    Program program = MakeProgramFromSpec(*mesh_device_, MakeGlobalSemaphoreSpec({"first", "second"}));
    const auto& impl = program.impl();

    const auto& handles = impl.get_kernel_by_spec_name("dm_kernel")->global_semaphore_binding_handles();
    ASSERT_EQ(handles.size(), 2u);
    EXPECT_EQ(handles[0].accessor_name, "first");
    EXPECT_EQ(handles[0].slot, 0u);
    EXPECT_EQ(handles[1].accessor_name, "second");
    EXPECT_EQ(handles[1].slot, 1u);
    EXPECT_TRUE(impl.get_kernel_by_spec_name("compute_kernel")->global_semaphore_binding_handles().empty());
}

// The parameter's cores are the union of its binders' nodes.
TEST_F(ProgramSpecTestGen1, CPU_ParameterCoresAreBinderNodes) {
    const NodeRange nodes{NodeCoord{0, 0}, NodeCoord{1, 1}};
    Program program = MakeProgramFromSpec(*mesh_device_, MakeGlobalSemaphoreSpec({"gsem"}, nodes));
    const auto* binding = program.impl().get_global_semaphore_parameter("gsem");
    ASSERT_NE(binding, nullptr);
    EXPECT_EQ(binding->cores, NodeRangeSet(nodes));
}

TEST_F(ProgramSpecTestGen1, CPU_GlobalAndProgramSemaphoresCoexist) {
    ProgramSpec spec = MakeGlobalSemaphoreSpec();
    BindSemaphoreToKernels(spec, "local_sem", {"dm_kernel"});

    Program program = MakeProgramFromSpec(*mesh_device_, spec);
    EXPECT_EQ(program.impl().num_global_semaphore_slots(), 1u);
}

TEST_F(ProgramSpecTestGen1, CPU_UnknownGlobalSemaphoreParameterIsRejected) {
    ProgramSpec spec = MakeGlobalSemaphoreSpec();
    BindGlobalSemaphoreToKernel(spec.kernels[0], "missing", "missing");
    EXPECT_GLOBAL_SEM_SPEC_REJECTED(spec, "references unknown GlobalSemaphoreParameter 'missing'");
}

TEST_F(ProgramSpecTestGen1, CPU_DuplicateGlobalSemaphoreParameterNameIsRejected) {
    ProgramSpec spec = MakeGlobalSemaphoreSpec();
    spec.global_semaphore_parameters.push_back(spec.global_semaphore_parameters.front());
    EXPECT_GLOBAL_SEM_SPEC_REJECTED(spec, "Duplicate GlobalSemaphoreParameter name 'gsem'");
}

TEST_F(ProgramSpecTestGen1, CPU_UnboundGlobalSemaphoreParameterIsRejected) {
    ProgramSpec spec = MakeGlobalSemaphoreSpec();
    spec.kernels[0].semaphore_bindings.clear();
    EXPECT_GLOBAL_SEM_SPEC_REJECTED(spec, "is declared but not bound by any kernel");
}

TEST_F(ProgramSpecTestGen1, CPU_SameGlobalSemaphoreBoundTwiceByOneKernelIsRejected) {
    ProgramSpec spec = MakeGlobalSemaphoreSpec();
    BindGlobalSemaphoreToKernel(spec.kernels[0], "gsem", "gsem_again");
    EXPECT_GLOBAL_SEM_SPEC_REJECTED(spec, "more than once");
}

// GlobalSemaphores live in L1 and are reached over the NoC, which TRISCs do not drive.
TEST_F(ProgramSpecTestBlackhole, CPU_ComputeKernelBindingGlobalSemaphoreIsRejected) {
    ProgramSpec spec = MakeGlobalSemaphoreSpec();
    BindGlobalSemaphoreToKernel(spec.kernels[1], "gsem", "gsem");
    EXPECT_GLOBAL_SEM_SPEC_REJECTED(spec, "GlobalSemaphores are only available to data-movement kernels");
}

// Codegen emits `using <accessor>_t = GlobalSemaphoreBindingToken<N>;` next to the accessor itself.
TEST_F(ProgramSpecTestGen1, CPU_AccessorCollidingWithGlobalTokenTypeIsRejected) {
    ProgramSpec spec = MakeGlobalSemaphoreSpec();
    BindSemaphoreToKernels(spec, "gsem_t", {"dm_kernel"});
    EXPECT_GLOBAL_SEM_SPEC_REJECTED(spec, "collides with the token type generated for GlobalSemaphore");
}

#undef EXPECT_GLOBAL_SEM_SPEC_REJECTED

}  // namespace
}  // namespace tt::tt_metal::experimental
