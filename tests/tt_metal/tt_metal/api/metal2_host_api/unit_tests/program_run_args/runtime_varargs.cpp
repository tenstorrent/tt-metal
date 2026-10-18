// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// SetProgramRunArgs with vararg-only RTAs, including per-node vararg count overrides.

#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>

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

// Shorthand for the per-node-override vararg type: a Table keyed by Nodes mapping to a
// vararg count (matches KernelAdvancedOptions::num_runtime_varargs_per_node).
using NumVarargsPerNode = Table<Nodes, uint32_t>;

// These document the "legacy kernel migrated lazily to Metal 2.0" pattern: all args as
// positional varargs, no named RTAs/CRTAs/CTAs.
TEST_F(ProgramRunArgsTestQuasar, CPU_VarargOnlyMultiNodeDifferingCountsSucceeds) {
    // A kernel on two nodes with DIFFERENT vararg counts per node. Exercises the advanced
    // num_runtime_varargs_per_node override path. The RTA dispatch buffer must be sized
    // per-node, which is a common failure mode for layout bugs.
    NodeCoord node_a{0, 0};
    NodeCoord node_b{1, 0};
    NodeRangeSet nodes{std::vector<NodeRange>{NodeRange{node_a, node_a}, NodeRange{node_b, node_b}}};

    ProgramSpec spec;
    spec.name = "vararg_differing_counts";
    auto kernel = MakeMinimalGen2DMKernel("dm_kernel");
    kernel.advanced_options.num_runtime_varargs_per_node = NumVarargsPerNode{{node_a, 2}, {node_b, 5}};
    spec.kernels = {kernel};
    spec.work_units = std::vector<WorkUnitSpec>{MakeMinimalWorkUnit("work_unit_0", nodes, {"dm_kernel"})};
    Program program = MakeProgramFromSpec(*mesh_device_, spec);

    ProgramRunArgs params;
    params.kernel_run_args.push_back(ProgramRunArgs::KernelRunArgs{
        .kernel = KernelSpecName{"dm_kernel"},
        .advanced_options =
            AdvancedKernelRunArgs{
                .runtime_varargs = {{node_a, {10, 20}}, {node_b, {100, 200, 300, 400, 500}}},
            },
    });
    EXPECT_NO_THROW(SetProgramRunArgs(program, params));
}

TEST_F(ProgramRunArgsTestQuasar, CPU_VarargPerNodeOverrideMixedEntryTypesSucceeds) {
    // Per-node override with a MIX of entry shapes: one entry groups two nodes via a
    // NodeRangeSet, another names a single NodeCoord. Exercises the schema-side expansion
    // from heterogeneous Nodes variants into per-coord validation entries — if the expansion
    // is wrong for either shape, some node won't be checked and validation will either fail
    // to require its values or fail to validate their count.
    NodeCoord node_a{0, 0};
    NodeCoord node_b{1, 0};
    NodeCoord node_c{2, 0};
    NodeRangeSet ab{std::vector<NodeRange>{NodeRange{node_a, node_a}, NodeRange{node_b, node_b}}};
    NodeRangeSet all_nodes{
        std::vector<NodeRange>{NodeRange{node_a, node_a}, NodeRange{node_b, node_b}, NodeRange{node_c, node_c}}};

    ProgramSpec spec;
    spec.name = "vararg_mixed_entry_types";
    auto kernel = MakeMinimalGen2DMKernel("dm_kernel");
    // Nodes a and b share count 3 (declared via a NodeRangeSet entry).
    // Node c has count 5 (declared via a NodeCoord entry).
    kernel.advanced_options.num_runtime_varargs_per_node = NumVarargsPerNode{{ab, 3}, {node_c, 5}};
    spec.kernels = {kernel};
    spec.work_units = std::vector<WorkUnitSpec>{MakeMinimalWorkUnit("work_unit_0", all_nodes, {"dm_kernel"})};
    Program program = MakeProgramFromSpec(*mesh_device_, spec);

    ProgramRunArgs params;
    params.kernel_run_args.push_back(ProgramRunArgs::KernelRunArgs{
        .kernel = KernelSpecName{"dm_kernel"},
        .advanced_options =
            AdvancedKernelRunArgs{
                .runtime_varargs = {{node_a, {1, 2, 3}}, {node_b, {10, 20, 30}}, {node_c, {100, 200, 300, 400, 500}}},
            },
    });
    EXPECT_NO_THROW(SetProgramRunArgs(program, params));
}

TEST_F(ProgramRunArgsTestQuasar, CPU_VarargScalarDefaultWithSparseOverrideSucceeds) {
    // Scalar provides the default count for every node the kernel runs on; the per-node
    // override covers only specific nodes. Unlisted nodes fall back to the scalar value.
    // This is the "3 on most nodes, 5 on the edges" shape that motivates the sparse
    // override design.
    NodeCoord node_a{0, 0};
    NodeCoord node_b{1, 0};
    NodeCoord node_c{2, 0};
    NodeRangeSet all_nodes{
        std::vector<NodeRange>{NodeRange{node_a, node_a}, NodeRange{node_b, node_b}, NodeRange{node_c, node_c}}};

    ProgramSpec spec;
    spec.name = "vararg_scalar_with_sparse_override";
    auto kernel = MakeMinimalGen2DMKernel("dm_kernel");
    kernel.advanced_options = KernelAdvancedOptions{
        .num_runtime_varargs = 2,                                        // default for unlisted nodes
        .num_runtime_varargs_per_node = NumVarargsPerNode{{node_c, 5}},  // node_c is the exception
    };
    spec.kernels = {kernel};
    spec.work_units = std::vector<WorkUnitSpec>{MakeMinimalWorkUnit("work_unit_0", all_nodes, {"dm_kernel"})};
    Program program = MakeProgramFromSpec(*mesh_device_, spec);

    ProgramRunArgs params;
    params.kernel_run_args.push_back(ProgramRunArgs::KernelRunArgs{
        .kernel = KernelSpecName{"dm_kernel"},
        .advanced_options =
            AdvancedKernelRunArgs{
                .runtime_varargs =
                    {
                        {node_a, {1, 2}},                     // scalar default (2 args)
                        {node_b, {10, 20}},                   // scalar default (2 args)
                        {node_c, {100, 200, 300, 400, 500}},  // override (5 args)
                    },
            },
    });
    EXPECT_NO_THROW(SetProgramRunArgs(program, params));
}

TEST_F(ProgramRunArgsTestQuasar, CPU_VarargSparseOverrideZeroErasesScalarDefault) {
    // An explicit override of 0 on a node erases the scalar default for that node.
    // Regression canary for the expansion logic: if the erase is missing, the node would
    // carry the scalar-default count and run-params validation would either require an
    // empty value list or error on count mismatch.
    NodeCoord node_a{0, 0};
    NodeCoord node_b{1, 0};
    NodeRangeSet both{std::vector<NodeRange>{NodeRange{node_a, node_a}, NodeRange{node_b, node_b}}};

    ProgramSpec spec;
    spec.name = "vararg_zero_override";
    auto kernel = MakeMinimalGen2DMKernel("dm_kernel");
    kernel.advanced_options = KernelAdvancedOptions{
        .num_runtime_varargs = 3,
        .num_runtime_varargs_per_node = NumVarargsPerNode{{node_b, 0}},  // node_b: no varargs despite scalar default
    };
    spec.kernels = {kernel};
    spec.work_units = std::vector<WorkUnitSpec>{MakeMinimalWorkUnit("work_unit_0", both, {"dm_kernel"})};
    Program program = MakeProgramFromSpec(*mesh_device_, spec);

    // node_b is treated as having no varargs — run-params needs no entry for it.
    ProgramRunArgs params;
    params.kernel_run_args.push_back(ProgramRunArgs::KernelRunArgs{
        .kernel = KernelSpecName{"dm_kernel"},
        .advanced_options =
            AdvancedKernelRunArgs{
                .runtime_varargs = {{node_a, {1, 2, 3}}},
            },
    });
    EXPECT_NO_THROW(SetProgramRunArgs(program, params));
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
