// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// UpdateProgramRunArgs: arbitrary partial updates; omitted arguments keep their previous values.

#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <set>
#include <stdexcept>
#include <vector>

#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>

#include "impl/dataflow_buffer/dataflow_buffer_impl.hpp"
#include "impl/kernels/kernel.hpp"
#include "impl/program/program_impl.hpp"
#include "metal2_host_api/test_helpers/test_helpers.hpp"
#include "metal2_host_api/test_helpers/mock_device_fixtures.hpp"
#include "metal2_host_api/test_helpers/run_args_test_helpers.hpp"

namespace tt::tt_metal::experimental {
namespace {

using test_helpers::MakeKernelRunArgs;
using test_helpers::MakeMinimalDFB;
using test_helpers::MakeMinimalGen2DMKernel;
using test_helpers::MakeMinimalWorkUnit;
using test_helpers::MakeRunArgsForMinimalSpec;
using test_helpers::MakeSpecWithNamedArgs;
using test_helpers::MakeSpecWithRTAs;
using test_helpers::ProgramRunArgsTestQuasar;

TEST_F(ProgramRunArgsTestQuasar, CPU_UpdateProgramRunArgs_RetainsOmittedCRTA) {
    NodeCoord node{0, 0};
    // Declaration order: keep @ slot 0 (omitted from the update), change @ slot 1 (supplied).
    ProgramSpec spec = MakeSpecWithNamedArgs(node, {}, {"keep", "change"});
    Program program = MakeProgramFromSpec(*mesh_device_, spec);

    ProgramRunArgs params;
    params.kernel_run_args.push_back(ProgramRunArgs::KernelRunArgs{
        .kernel = KernelSpecName{"dm_kernel"},
        .common_runtime_arg_values = {{"keep", 10}, {"change", 20}},
    });
    params.kernel_run_args.push_back(MakeKernelRunArgs(KernelSpecName{"compute_kernel"}, node, {}, {}));
    SetProgramRunArgs(program, params);

    // Supply only "change"; omit "keep" and the all-empty compute_kernel.
    ProgramRunArgs upd;
    upd.kernel_run_args.push_back(ProgramRunArgs::KernelRunArgs{
        .kernel = KernelSpecName{"dm_kernel"},
        .common_runtime_arg_values = {{"change", 99}},
    });
    EXPECT_NO_THROW(UpdateProgramRunArgs(program, upd));

    const auto* crta = program.impl().get_kernel_by_spec_name("dm_kernel")->common_runtime_args_data().data();
    EXPECT_EQ(crta[0], 10u) << "omitted 'keep' must retain its value across a partial update";
    EXPECT_EQ(crta[1], 99u) << "supplied 'change' must be updated";
}

TEST_F(ProgramRunArgsTestQuasar, CPU_UpdateProgramRunArgs_RetainsOmittedPerNodeRTA) {
    NodeCoord node{0, 0};
    ProgramSpec spec = MakeSpecWithNamedArgs(node, {"keep", "change"}, {});
    Program program = MakeProgramFromSpec(*mesh_device_, spec);

    ProgramRunArgs params;
    params.kernel_run_args.push_back(ProgramRunArgs::KernelRunArgs{
        .kernel = KernelSpecName{"dm_kernel"},
        .runtime_arg_values = MakeRuntimeArgsForSingleNode(node, {{"keep", 1}, {"change", 2}}),
    });
    params.kernel_run_args.push_back(MakeKernelRunArgs(KernelSpecName{"compute_kernel"}, node, {}, {}));
    SetProgramRunArgs(program, params);

    ProgramRunArgs upd;
    upd.kernel_run_args.push_back(ProgramRunArgs::KernelRunArgs{
        .kernel = KernelSpecName{"dm_kernel"},
        .runtime_arg_values = MakeRuntimeArgsForSingleNode(node, {{"change", 99}}),
    });
    EXPECT_NO_THROW(UpdateProgramRunArgs(program, upd));

    const auto& rta = program.impl().get_kernel_by_spec_name("dm_kernel")->runtime_args(node);
    ASSERT_GE(rta.size(), 2u);
    EXPECT_EQ(rta[0], 1u) << "omitted 'keep' must retain its value";
    EXPECT_EQ(rta[1], 99u) << "supplied 'change' must be updated";
}

// --- Still enforced: a supplied name must be declared in the schema (no extras) ---

TEST_F(ProgramRunArgsTestQuasar, CPU_UpdateProgramRunArgs_UndeclaredCRTANameFails) {
    NodeCoord node{0, 0};
    ProgramSpec spec = MakeSpecWithNamedArgs(node, {}, {"keep", "change"});
    Program program = MakeProgramFromSpec(*mesh_device_, spec);

    ProgramRunArgs full;
    full.kernel_run_args.push_back(ProgramRunArgs::KernelRunArgs{
        .kernel = KernelSpecName{"dm_kernel"},
        .common_runtime_arg_values = {{"keep", 10}, {"change", 20}},
    });
    full.kernel_run_args.push_back(MakeKernelRunArgs(KernelSpecName{"compute_kernel"}, node, {}, {}));
    SetProgramRunArgs(program, full);

    // Supplying a name that the schema never declared is still an error, even on the partial path.
    ProgramRunArgs upd;
    upd.kernel_run_args.push_back(ProgramRunArgs::KernelRunArgs{
        .kernel = KernelSpecName{"dm_kernel"},
        .common_runtime_arg_values = {{"bogus", 11}},
    });
    EXPECT_THAT(
        [&] { UpdateProgramRunArgs(program, upd); },
        ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr("not declared in the schema")));
}

// --- Precondition: a partial update before any full set fails ---

TEST_F(ProgramRunArgsTestQuasar, CPU_UpdateProgramRunArgs_BeforeSetFails) {
    NodeCoord node{0, 0};
    ProgramSpec spec = MakeSpecWithNamedArgs(node, {}, {"keep", "change"});
    Program program = MakeProgramFromSpec(*mesh_device_, spec);

    ProgramRunArgs upd;
    upd.kernel_run_args.push_back(ProgramRunArgs::KernelRunArgs{
        .kernel = KernelSpecName{"dm_kernel"},
        .common_runtime_arg_values = {{"change", 99}},
    });
    EXPECT_THAT(
        [&] { UpdateProgramRunArgs(program, upd); },
        ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr("CRTA buffer not allocated")));
}

// --- Kernel omission: an omitted kernel retains all of its args ---

TEST_F(ProgramRunArgsTestQuasar, CPU_UpdateProgramRunArgs_OmittingKernelRetainsArgs) {
    NodeCoord node{0, 0};
    ProgramSpec spec = MakeSpecWithNamedArgs(node, {}, {"change"});
    Program program = MakeProgramFromSpec(*mesh_device_, spec);

    ProgramRunArgs full;
    full.kernel_run_args.push_back(ProgramRunArgs::KernelRunArgs{
        .kernel = KernelSpecName{"dm_kernel"},
        .common_runtime_arg_values = {{"change", 20}},
    });
    full.kernel_run_args.push_back(MakeKernelRunArgs(KernelSpecName{"compute_kernel"}, node, {}, {}));
    SetProgramRunArgs(program, full);

    // Omit every kernel: an arbitrary partial update may leave all args untouched.
    ProgramRunArgs upd;
    EXPECT_NO_THROW(UpdateProgramRunArgs(program, upd));
    EXPECT_EQ(program.impl().get_kernel_by_spec_name("dm_kernel")->common_runtime_args_data().data()[0], 20u)
        << "an omitted kernel's args are retained";
}

// --- Omitted vararg sections (per-node + common) and untouched nodes are retained ---

// Regression for the arbitrary-partial-update vararg axis: a partial update that touches only one
// named arg on one node must leave every OMITTED vararg section — per-node on both nodes, and the
// common section — plus the untouched node's values intact. Guards against (a) a re-introduced
// vararg completeness check (would FATAL on the omission) and (b) a patch that clobbers a retained
// section or writes the wrong node.
TEST_F(ProgramRunArgsTestQuasar, CPU_UpdateProgramRunArgs_RetainsOmittedVarargsAcrossNodes) {
    NodeCoord node0{0, 0};
    NodeCoord node1{1, 0};
    NodeRangeSet all_nodes(std::set<NodeRange>{NodeRange{node0, node0}, NodeRange{node1, node1}});

    ProgramSpec spec;
    spec.name = "vararg_retain_program";

    // producer: one named RTA ("addr") + 2 per-node varargs + 2 common varargs, spanning both nodes.
    auto producer = MakeMinimalGen2DMKernel("producer");
    producer.runtime_arg_schema.runtime_arg_names = {"addr"};
    producer.advanced_options = KernelAdvancedOptions{.num_runtime_varargs = 2, .num_common_runtime_varargs = 2};
    auto consumer = MakeMinimalGen2DMKernel("consumer");

    auto dfb = MakeMinimalDFB("dfb");
    producer.dfb_bindings.push_back(ProducerOf(DFBSpecName{"dfb"}, "out"));
    consumer.dfb_bindings.push_back(ConsumerOf(DFBSpecName{"dfb"}, "in"));

    spec.kernels = {producer, consumer};
    spec.dataflow_buffers = {dfb};
    spec.work_units = std::vector<WorkUnitSpec>{MakeMinimalWorkUnit("work_unit", all_nodes, {"producer", "consumer"})};
    Program program = MakeProgramFromSpec(*mesh_device_, spec);

    // Full set: named RTA + per-node varargs on both nodes + common varargs.
    KernelRunArgs::RuntimeArgValues named;
    AddRuntimeArgsForNode(named, node0, {{"addr", 100}});
    AddRuntimeArgsForNode(named, node1, {{"addr", 200}});
    ProgramRunArgs full;
    full.kernel_run_args.push_back(ProgramRunArgs::KernelRunArgs{
        .kernel = KernelSpecName{"producer"},
        .runtime_arg_values = named,
        .advanced_options =
            AdvancedKernelRunArgs{
                .runtime_varargs = {{node0, {10, 11}}, {node1, {20, 21}}},
                .common_runtime_varargs = {30, 31},
            },
    });
    full.kernel_run_args.push_back(ProgramRunArgs::KernelRunArgs{
        .kernel = KernelSpecName{"consumer"},
        .advanced_options = AdvancedKernelRunArgs{.runtime_varargs = {{node0, {}}, {node1, {}}}},
    });
    SetProgramRunArgs(program, full);

    // Partial update: change ONLY the named RTA on node0. Every vararg section (both nodes' per-node,
    // and the common section) and node1's named RTA are omitted -> all must be retained.
    KernelRunArgs::RuntimeArgValues upd_named;
    AddRuntimeArgsForNode(upd_named, node0, {{"addr", 999}});
    ProgramRunArgs upd;
    upd.kernel_run_args.push_back(ProgramRunArgs::KernelRunArgs{
        .kernel = KernelSpecName{"producer"},
        .runtime_arg_values = upd_named,
    });
    EXPECT_NO_THROW(UpdateProgramRunArgs(program, upd));

    auto prod = program.impl().get_kernel_by_spec_name("producer");
    // Per-node RTA buffer layout: [named "addr" @ slot 0][vararg0 @ 1][vararg1 @ 2].
    const auto& rta0 = prod->runtime_args_data(node0);
    const auto& rta1 = prod->runtime_args_data(node1);
    ASSERT_GE(rta0.size(), 3u);
    ASSERT_GE(rta1.size(), 3u);
    EXPECT_EQ(rta0.data()[0], 999u) << "supplied named RTA on node0 updated";
    EXPECT_EQ(rta0.data()[1], 10u) << "node0 vararg 0 retained (omitted)";
    EXPECT_EQ(rta0.data()[2], 11u) << "node0 vararg 1 retained (omitted)";
    EXPECT_EQ(rta1.data()[0], 200u) << "untouched node1 named RTA retained";
    EXPECT_EQ(rta1.data()[1], 20u) << "untouched node1 vararg 0 retained";
    EXPECT_EQ(rta1.data()[2], 21u) << "untouched node1 vararg 1 retained";
    // CRTA buffer here is just the common varargs (no named CRTAs, tensor bindings, or scratchpad).
    const auto* crta = prod->common_runtime_args_data().data();
    EXPECT_EQ(crta[0], 30u) << "common vararg 0 retained (omitted)";
    EXPECT_EQ(crta[1], 31u) << "common vararg 1 retained (omitted)";
}

// --- DFB size overrides on the fast path (mock fixture: inspects config, no enqueue) ---

// UpdateProgramRunArgs applies a DFB size override, mirroring the SetProgramRunArgs path.
TEST_F(ProgramRunArgsTestQuasar, CPU_UpdateProgramRunArgs_AppliesDFBSizeOverride) {
    NodeCoord node{0, 0};
    ProgramSpec spec = MakeSpecWithRTAs(node, 0, 0);
    Program program = MakeProgramFromSpec(*mesh_device_, spec);

    // A full set must precede any partial update.
    SetProgramRunArgs(program, MakeRunArgsForMinimalSpec(node, {}, {}));

    ProgramRunArgs upd;
    upd.dfb_run_overrides.push_back({.dfb = DFBSpecName{"dfb_0"}, .num_entries = 4});
    EXPECT_NO_THROW(UpdateProgramRunArgs(program, upd));

    auto dfb = program.impl().get_dataflow_buffer(program.impl().get_dfb_handle("dfb_0"));
    EXPECT_EQ(dfb->config.entry_size, 1024u);  // unchanged
    EXPECT_EQ(dfb->config.num_entries, 4u);    // overridden via UpdateProgramRunArgs
}

// DFB size overrides are stateful: a DFB not re-specified in a later update keeps its current
// size rather than reverting to the ProgramSpec default.
TEST_F(ProgramRunArgsTestQuasar, CPU_UpdateProgramRunArgs_RetainsDFBSizeOverrideWhenUnspecified) {
    NodeCoord node{0, 0};
    ProgramSpec spec = MakeSpecWithRTAs(node, 0, 0);
    Program program = MakeProgramFromSpec(*mesh_device_, spec);

    // Establish an override (the spec default num_entries is 2).
    auto params = MakeRunArgsForMinimalSpec(node, {}, {});
    params.dfb_run_overrides.push_back({.dfb = DFBSpecName{"dfb_0"}, .num_entries = 4});
    SetProgramRunArgs(program, params);

    // A subsequent update that omits the DFB must not reset it to the spec default.
    ProgramRunArgs upd;
    EXPECT_NO_THROW(UpdateProgramRunArgs(program, upd));

    auto dfb = program.impl().get_dataflow_buffer(program.impl().get_dfb_handle("dfb_0"));
    EXPECT_EQ(dfb->config.num_entries, 4u) << "override must persist across an update that omits it";
}

}  // namespace
}  // namespace tt::tt_metal::experimental
