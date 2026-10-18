// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// KernelAdvancedOptions fields that must (or must not) change the kernel's JIT cache key (compute_hash).

#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program.hpp>

#include "impl/kernels/kernel.hpp"
#include "impl/program/program_impl.hpp"
#include "metal2_host_api/test_helpers/test_helpers.hpp"
#include "metal2_host_api/test_helpers/mock_device_fixtures.hpp"

namespace tt::tt_metal::experimental {
namespace {

using test_helpers::BindTensorParameterToKernel;
using test_helpers::MakeMinimalGen1DMKernel;
using test_helpers::MakeMinimalGen1ValidProgramSpec;
using test_helpers::MakeMinimalTensorParameter;
using test_helpers::MakeMinimalWorkUnit;
using test_helpers::ProgramSpecTestGen1;

TEST_F(ProgramSpecTestGen1, CPU_TensorBindingSequenceMemberPartitionAffectsKernelHash) {
    // Same bindings {a, ab, bc, c}; sequences differ only by member partition {"a","bc"} vs {"ab","c"}.
    // Without per-member length delimiting those would hash identically.
    auto make_spec = [](std::vector<std::string> members) {
        ProgramSpec spec = MakeMinimalGen1ValidProgramSpec();
        spec.tensor_parameters = {
            MakeMinimalTensorParameter("t_a"),
            MakeMinimalTensorParameter("t_ab"),
            MakeMinimalTensorParameter("t_bc"),
            MakeMinimalTensorParameter("t_c"),
        };
        BindTensorParameterToKernel(spec.kernels[0], "t_a", "a");
        BindTensorParameterToKernel(spec.kernels[0], "t_ab", "ab");
        BindTensorParameterToKernel(spec.kernels[0], "t_bc", "bc");
        BindTensorParameterToKernel(spec.kernels[0], "t_c", "c");
        spec.kernels[0].advanced_options.tensor_binding_sequences = {
            KernelAdvancedOptions::TensorBindingSequence{.sequence_name = "inputs", .members = std::move(members)},
        };
        return spec;
    };

    Program prog_left = MakeProgramFromSpec(*mesh_device_, make_spec({"a", "bc"}));
    Program prog_right = MakeProgramFromSpec(*mesh_device_, make_spec({"ab", "c"}));

    auto hash_left = prog_left.impl().get_kernel_by_spec_name("dm_kernel")->compute_hash();
    auto hash_right = prog_right.impl().get_kernel_by_spec_name("dm_kernel")->compute_hash();
    EXPECT_NE(hash_left, hash_right)
        << "Tensor binding sequences with different member partitions must not share a JIT cache slot.";
}

TEST_F(ProgramSpecTestGen1, CPU_DifferentCompileTimeVarargsProducesDifferentKernelHash) {
    auto make_program = [this](std::vector<uint32_t> varargs) {
        NodeCoord node{0, 0};
        auto dm_kernel = MakeMinimalGen1DMKernel("dm_kernel");
        dm_kernel.advanced_options.compile_time_varargs = std::move(varargs);
        ProgramSpec spec;
        spec.name = "cta_varargs_hash";
        spec.kernels = {dm_kernel};
        spec.work_units = std::vector<WorkUnitSpec>{MakeMinimalWorkUnit("work_unit", node, {"dm_kernel"})};
        return MakeProgramFromSpec(*mesh_device_, spec);
    };

    Program prog_a = make_program({0x11112222u, 0x33334444u});
    Program prog_b = make_program({0xAAAABBBBu, 0xCCCCDDDDu});
    EXPECT_NE(
        prog_a.impl().get_kernel_by_spec_name("dm_kernel")->compute_hash(),
        prog_b.impl().get_kernel_by_spec_name("dm_kernel")->compute_hash());
}

TEST_F(ProgramSpecTestGen1, CPU_IdenticalCompileTimeVarargsProducesIdenticalKernelHash) {
    auto make_program = [this] {
        NodeCoord node{0, 0};
        auto dm_kernel = MakeMinimalGen1DMKernel("dm_kernel");
        dm_kernel.advanced_options.compile_time_varargs = {0xCAFEBABEu, 0xDEADBEEFu};
        ProgramSpec spec;
        spec.name = "cta_varargs_hash_ident";
        spec.kernels = {dm_kernel};
        spec.work_units = std::vector<WorkUnitSpec>{MakeMinimalWorkUnit("work_unit", node, {"dm_kernel"})};
        return MakeProgramFromSpec(*mesh_device_, spec);
    };

    Program prog_a = make_program();
    Program prog_b = make_program();
    EXPECT_EQ(
        prog_a.impl().get_kernel_by_spec_name("dm_kernel")->compute_hash(),
        prog_b.impl().get_kernel_by_spec_name("dm_kernel")->compute_hash());
}

// Prefix length is part of the cache key even when positional CTA words are unchanged.
TEST_F(ProgramSpecTestGen1, CPU_DifferentCompileTimeVarargCountProducesDifferentKernelHash) {
    NodeCoord node{0, 0};
    auto dm_kernel = MakeMinimalGen1DMKernel("dm_kernel");
    dm_kernel.advanced_options.compile_time_varargs = {0xCAFEBABEu, 0xDEADBEEFu};
    ProgramSpec spec;
    spec.name = "cta_varargs_hash_count";
    spec.kernels = {dm_kernel};
    spec.work_units = std::vector<WorkUnitSpec>{MakeMinimalWorkUnit("work_unit", node, {"dm_kernel"})};

    Program prog_a = MakeProgramFromSpec(*mesh_device_, spec);
    Program prog_b = MakeProgramFromSpec(*mesh_device_, spec);
    auto kernel_a = prog_a.impl().get_kernel_by_spec_name("dm_kernel");
    auto kernel_b = prog_b.impl().get_kernel_by_spec_name("dm_kernel");
    ASSERT_EQ(kernel_a->get_compile_time_vararg_count(), 2u);
    ASSERT_EQ(kernel_b->get_compile_time_vararg_count(), 2u);
    ASSERT_EQ(kernel_a->compute_hash(), kernel_b->compute_hash());

    kernel_b->set_compile_time_vararg_count(1u);
    EXPECT_NE(kernel_a->compute_hash(), kernel_b->compute_hash());
}

}  // namespace
}  // namespace tt::tt_metal::experimental
