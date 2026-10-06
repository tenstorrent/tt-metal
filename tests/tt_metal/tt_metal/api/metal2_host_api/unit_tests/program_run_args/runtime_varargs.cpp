// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// SetProgramRunArgs with vararg-only RTAs, including per-node vararg count overrides.

#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <vector>

#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>

#include "impl/kernels/kernel.hpp"
#include "impl/program/program_impl.hpp"
#include "metal2_host_api/test_helpers/test_helpers.hpp"
#include "metal2_host_api/test_helpers/mock_device_fixtures.hpp"
#include "metal2_host_api/test_helpers/run_args_test_helpers.hpp"

namespace tt::tt_metal::experimental {
namespace {

using test_helpers::MakeKernelRunArgs;
using test_helpers::MakeMinimalGen2DMKernel;
using test_helpers::MakeMinimalValidProgramSpec;
using test_helpers::MakeMinimalWorkUnit;
using test_helpers::ProgramRunArgsTestQuasar;

TEST_F(ProgramRunArgsTestQuasar, CPU_UniformVarargCapacityAcrossWorkUnitsSucceeds) {
    NodeCoord node_a{0, 0};
    NodeCoord node_b{1, 0};
    NodeCoord node_c{2, 0};
    NodeRangeSet ab{std::vector<NodeRange>{NodeRange{node_a, node_b}}};
    NodeRangeSet c{std::vector<NodeRange>{NodeRange{node_c, node_c}}};

    ProgramSpec spec;
    spec.name = "uniform_vararg_capacity";
    auto kernel = MakeMinimalGen2DMKernel("dm_kernel");
    kernel.runtime_arg_schema.runtime_arg_names = {"count"};
    kernel.advanced_options.num_runtime_varargs = 5;
    spec.kernels = {kernel};
    spec.work_units = {
        MakeMinimalWorkUnit("main", ab, {"dm_kernel"}),
        MakeMinimalWorkUnit("cliff", c, {"dm_kernel"}),
    };
    Program program = MakeProgramFromSpec(*mesh_device_, spec);
    auto impl = program.impl().get_kernel_by_spec_name("dm_kernel");
    // Factory construction must reserve named + vararg slots even before values are supplied.
    for (const auto& node : {node_a, node_b, node_c}) {
        EXPECT_EQ(impl->runtime_args_data(node).size(), 6u);
    }

    auto runtime_values = [&](const NodeCoord& node) {
        const auto& rta = impl->runtime_args_data(node);
        return std::span<const uint32_t>(rta.data(), rta.size());
    };

    KernelRunArgs args{.kernel = KernelSpecName{"dm_kernel"}};
    AddRuntimeArgsForNode(args.runtime_arg_values, node_a, {{"count", 2}});
    AddRuntimeArgsForNode(args.runtime_arg_values, node_b, {{"count", 5}});
    AddRuntimeArgsForNode(args.runtime_arg_values, node_c, {{"count", 0}});
    args.advanced_options.runtime_varargs = {
        {node_a, {10, 20, 0, 0, 0}},
        {node_b, {100, 200, 300, 400, 500}},
        {node_c, {0, 0, 0, 0, 0}},
    };
    ProgramRunArgs params;
    params.kernel_run_args = {args};
    ASSERT_NO_THROW(SetProgramRunArgs(program, params));
    EXPECT_THAT(runtime_values(node_a), ::testing::ElementsAre(2, 10, 20, 0, 0, 0));
    EXPECT_THAT(runtime_values(node_b), ::testing::ElementsAre(5, 100, 200, 300, 400, 500));
    EXPECT_THAT(runtime_values(node_c), ::testing::ElementsAre(0, 0, 0, 0, 0, 0));

    // Updating one core uses the same capacity and retains the other cores' sections.
    ProgramRunArgs update;
    KernelRunArgs updated{.kernel = KernelSpecName{"dm_kernel"}};
    AddRuntimeArgsForNode(updated.runtime_arg_values, node_a, {{"count", 1}});
    updated.advanced_options.runtime_varargs = {{node_a, {99, 0, 0, 0, 0}}};
    update.kernel_run_args = {updated};
    ASSERT_NO_THROW(UpdateProgramRunArgs(program, update));
    EXPECT_THAT(runtime_values(node_a), ::testing::ElementsAre(1, 99, 0, 0, 0, 0));
    EXPECT_THAT(runtime_values(node_b), ::testing::ElementsAre(5, 100, 200, 300, 400, 500));
    EXPECT_THAT(runtime_values(node_c), ::testing::ElementsAre(0, 0, 0, 0, 0, 0));
}

TEST_F(ProgramRunArgsTestQuasar, CPU_VarargCountsMustMatchOnEveryNode) {
    NodeCoord node_a{0, 0};
    NodeCoord node_b{1, 0};
    NodeRangeSet nodes{std::vector<NodeRange>{NodeRange{node_a, node_b}}};

    ProgramSpec spec;
    spec.name = "uniform_vararg_count_validation";
    auto kernel = MakeMinimalGen2DMKernel("dm_kernel");
    kernel.advanced_options.num_runtime_varargs = 3;
    spec.kernels = {kernel};
    spec.work_units = {MakeMinimalWorkUnit("main", nodes, {"dm_kernel"})};
    Program program = MakeProgramFromSpec(*mesh_device_, spec);

    ProgramRunArgs params;
    params.kernel_run_args = {KernelRunArgs{
        .kernel = KernelSpecName{"dm_kernel"},
        .advanced_options = {.runtime_varargs = {{node_a, {1, 2, 3}}, {node_b, {4, 5, 6}}}},
    }};
    ASSERT_NO_THROW(SetProgramRunArgs(program, params));

    for (const auto& node : {node_a, node_b}) {
        for (uint32_t count : {0u, 2u, 4u}) {
            auto invalid = params;
            invalid.kernel_run_args[0].advanced_options.runtime_varargs[node].resize(count);
            EXPECT_THAT(
                [&] { SetProgramRunArgs(program, invalid); },
                ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr("expects 3 vararg runtime args")));
            // Empty updates intentionally retain the previous value.
            if (count != 0) {
                EXPECT_THAT(
                    [&] { UpdateProgramRunArgs(program, invalid); },
                    ::testing::ThrowsMessage<std::runtime_error>(
                        ::testing::HasSubstr("expects 3 vararg runtime args")));
            }
        }
    }
}

TEST_F(ProgramRunArgsTestQuasar, CPU_VarargOnlyAcrossMultipleKernelsSucceeds) {
    // Two kernels, each with only vararg RTAs / CRTAs — the shape of a whole-program
    // migration where nothing has been upgraded to named args yet.
    NodeCoord node{0, 0};
    ProgramSpec spec = MakeMinimalValidProgramSpec();
    spec.kernels[0].advanced_options = KernelAdvancedOptions{.num_runtime_varargs = 3, .num_common_runtime_varargs = 1};
    spec.kernels[1].advanced_options = KernelAdvancedOptions{.num_runtime_varargs = 2, .num_common_runtime_varargs = 2};
    Program program = MakeProgramFromSpec(*mesh_device_, spec);

    ProgramRunArgs params;
    params.kernel_run_args.push_back(MakeKernelRunArgs(KernelSpecName{"dm_kernel"}, node, {1, 2, 3}, {99}));
    params.kernel_run_args.push_back(MakeKernelRunArgs(KernelSpecName{"compute_kernel"}, node, {7, 8}, {42, 43}));
    EXPECT_NO_THROW(SetProgramRunArgs(program, params));
}

TEST_F(ProgramRunArgsTestQuasar, CPU_VarargOnlyRTAsMissingNodeCoverageFails) {
    // If the schema declares varargs for a node, SetProgramRunArgs must insist on
    // values for that node. Regression canary for per-node coverage in the vararg path.
    NodeCoord node_a{0, 0};
    NodeCoord node_b{1, 0};
    NodeRangeSet nodes{std::vector<NodeRange>{NodeRange{node_a, node_a}, NodeRange{node_b, node_b}}};

    ProgramSpec spec;
    spec.name = "vararg_missing_node";
    auto kernel = MakeMinimalGen2DMKernel("dm_kernel");
    kernel.advanced_options.num_runtime_varargs = 2;  // uniform across both nodes
    spec.kernels = {kernel};
    spec.work_units = std::vector<WorkUnitSpec>{MakeMinimalWorkUnit("work_unit_0", nodes, {"dm_kernel"})};
    Program program = MakeProgramFromSpec(*mesh_device_, spec);

    ProgramRunArgs params;
    params.kernel_run_args.push_back(ProgramRunArgs::KernelRunArgs{
        .kernel = KernelSpecName{"dm_kernel"},
        .advanced_options =
            AdvancedKernelRunArgs{
                .runtime_varargs = {{node_a, {10, 20}}},  // node_b missing!
            },
    });
    EXPECT_THAT(
        [&] { SetProgramRunArgs(program, params); },
        ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr("missing vararg runtime args for node")));
}

TEST_F(ProgramRunArgsTestQuasar, CPU_VarargOnlyUnknownNodeFails) {
    // Host passes runtime_varargs for a node the kernel doesn't run on. Regression canary for
    // the domain check added alongside named-RTA validation.
    NodeCoord node{0, 0};
    NodeCoord wrong_node{3, 3};
    ProgramSpec spec = MakeMinimalValidProgramSpec();
    spec.kernels[0].advanced_options.num_runtime_varargs = 1;
    Program program = MakeProgramFromSpec(*mesh_device_, spec);

    ProgramRunArgs params;
    params.kernel_run_args.push_back(ProgramRunArgs::KernelRunArgs{
        .kernel = KernelSpecName{"dm_kernel"},
        .advanced_options =
            AdvancedKernelRunArgs{
                .runtime_varargs = {{wrong_node, {42}}},
            },
    });
    params.kernel_run_args.push_back(MakeKernelRunArgs(KernelSpecName{"compute_kernel"}, node, {}, {}));
    EXPECT_THAT(
        [&] { SetProgramRunArgs(program, params); },
        ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr("the kernel does not run on that node")));
}

}  // namespace
}  // namespace tt::tt_metal::experimental
