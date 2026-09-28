// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// SetProgramRunArgs with named RTAs / CRTAs, and where their values land in the kernel's arg buffers.

#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>

#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>

#include "impl/kernels/kernel.hpp"
#include "impl/program/program_impl.hpp"
#include "metal2_host_api/test_helpers.hpp"
#include "metal2_host_api/mock_device_fixtures.hpp"
#include "metal2_host_api/run_args_test_helpers.hpp"

namespace tt::tt_metal::experimental {
namespace {

using test_helpers::MakeKernelRunArgs;
using test_helpers::MakeMinimalGen1ValidProgramSpec;
using test_helpers::MakeMinimalValidProgramSpec;
using test_helpers::MakeSpecWithNamedArgs;
using test_helpers::ProgramRunArgsTestGen1;
using test_helpers::ProgramRunArgsTestQuasar;

TEST_F(ProgramRunArgsTestQuasar, CPU_NamedRTAsAndCRTAsSucceed) {
    NodeCoord node{0, 0};
    ProgramSpec spec = MakeSpecWithNamedArgs(node, {"input_ptr", "output_ptr"}, {"tile_count"});
    Program program = MakeProgramFromSpec(*mesh_device_, spec);

    ProgramRunArgs params;
    params.kernel_run_args.push_back(ProgramRunArgs::KernelRunArgs{
        .kernel = KernelSpecName{"dm_kernel"},
        .runtime_arg_values = MakeRuntimeArgsForSingleNode(node, {{"input_ptr", 0x1000}, {"output_ptr", 0x2000}}),
        .common_runtime_arg_values = {{"tile_count", 64}},
    });
    params.kernel_run_args.push_back(MakeKernelRunArgs(KernelSpecName{"compute_kernel"}, node, {}, {}));

    EXPECT_NO_THROW(SetProgramRunArgs(program, params));
}

TEST_F(ProgramRunArgsTestQuasar, CPU_MissingNamedRTAForNodeFails) {
    NodeCoord node{0, 0};
    ProgramSpec spec = MakeSpecWithNamedArgs(node, {"input_ptr"}, {});
    Program program = MakeProgramFromSpec(*mesh_device_, spec);

    ProgramRunArgs params;
    params.kernel_run_args.push_back(ProgramRunArgs::KernelRunArgs{
        .kernel = KernelSpecName{"dm_kernel"},
        // No runtime_arg_values for node (0,0) at all — but schema declares one.
    });
    params.kernel_run_args.push_back(MakeKernelRunArgs(KernelSpecName{"compute_kernel"}, node, {}, {}));

    EXPECT_THAT(
        [&] { SetProgramRunArgs(program, params); },
        ::testing::ThrowsMessage<std::runtime_error>(
            ::testing::HasSubstr("has named RTAs declared but no runtime_arg_values provided for node")));
}

TEST_F(ProgramRunArgsTestQuasar, CPU_MissingDeclaredNamedRTANameFails) {
    NodeCoord node{0, 0};
    ProgramSpec spec = MakeSpecWithNamedArgs(node, {"input_ptr", "output_ptr"}, {});
    Program program = MakeProgramFromSpec(*mesh_device_, spec);

    ProgramRunArgs params;
    params.kernel_run_args.push_back(ProgramRunArgs::KernelRunArgs{
        .kernel = KernelSpecName{"dm_kernel"},
        // Only one name provided — output_ptr missing.
        .runtime_arg_values = MakeRuntimeArgsForSingleNode(node, {{"input_ptr", 0x1000}}),
    });
    params.kernel_run_args.push_back(MakeKernelRunArgs(KernelSpecName{"compute_kernel"}, node, {}, {}));

    EXPECT_THAT(
        [&] { SetProgramRunArgs(program, params); },
        ::testing::ThrowsMessage<std::runtime_error>(
            ::testing::HasSubstr("expects 2 named RTAs, but 1 were provided")));
}

TEST_F(ProgramRunArgsTestQuasar, CPU_UndeclaredNamedRTAFails) {
    NodeCoord node{0, 0};
    ProgramSpec spec = MakeSpecWithNamedArgs(node, {"input_ptr"}, {});
    Program program = MakeProgramFromSpec(*mesh_device_, spec);

    ProgramRunArgs params;
    params.kernel_run_args.push_back(ProgramRunArgs::KernelRunArgs{
        .kernel = KernelSpecName{"dm_kernel"},
        .runtime_arg_values = MakeRuntimeArgsForSingleNode(node, {{"input_ptr", 0x1000}, {"not_in_schema", 0}}),
    });
    params.kernel_run_args.push_back(MakeKernelRunArgs(KernelSpecName{"compute_kernel"}, node, {}, {}));

    EXPECT_THAT(
        [&] { SetProgramRunArgs(program, params); },
        ::testing::ThrowsMessage<std::runtime_error>(
            ::testing::HasSubstr("expects 1 named RTAs, but 2 were provided")));
}

TEST_F(ProgramRunArgsTestQuasar, CPU_NamedCRTACountMismatchFails) {
    NodeCoord node{0, 0};
    ProgramSpec spec = MakeSpecWithNamedArgs(node, {}, {"tile_count", "scale"});
    Program program = MakeProgramFromSpec(*mesh_device_, spec);

    ProgramRunArgs params;
    params.kernel_run_args.push_back(ProgramRunArgs::KernelRunArgs{
        .kernel = KernelSpecName{"dm_kernel"},
        // Only one CRTA provided; schema declares two.
        .common_runtime_arg_values = {{"tile_count", 4}},
    });
    params.kernel_run_args.push_back(MakeKernelRunArgs(KernelSpecName{"compute_kernel"}, node, {}, {}));

    // Validation now reports the specific missing name first (more useful than a count mismatch).
    EXPECT_THAT(
        [&] { SetProgramRunArgs(program, params); },
        ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr("is missing named CRTA 'scale'")));
}

TEST_F(ProgramRunArgsTestQuasar, CPU_NamedAndVarargRTAsCoexistSucceeds) {
    // A kernel with both named RTAs (schema) and varargs (num_runtime_varargs).
    NodeCoord node{0, 0};
    ProgramSpec spec = MakeMinimalValidProgramSpec();
    spec.kernels[0].runtime_arg_schema.runtime_arg_names = {"input_ptr"};
    spec.kernels[0].advanced_options.num_runtime_varargs = 3;
    Program program = MakeProgramFromSpec(*mesh_device_, spec);

    ProgramRunArgs params;
    params.kernel_run_args.push_back(ProgramRunArgs::KernelRunArgs{
        .kernel = KernelSpecName{"dm_kernel"},
        .runtime_arg_values = MakeRuntimeArgsForSingleNode(node, {{"input_ptr", 0x1000}}),
        .advanced_options =
            AdvancedKernelRunArgs{
                .runtime_varargs = {{node, {7, 8, 9}}},
            },
    });
    params.kernel_run_args.push_back(MakeKernelRunArgs(KernelSpecName{"compute_kernel"}, node, {}, {}));

    EXPECT_NO_THROW(SetProgramRunArgs(program, params));
}

// SetProgramRunArgs serializes named per-node RTAs by *scattering* each supplied value to its
// declaration slot (schema name->slot index), not by walking the schema and looking each name up.
// This test pins that behavior at the value level — which the NO_THROW tests above do not:
//   - supplied OUT OF declaration order, values must still land at their declared slot;
//   - a second SetProgramRunArgs (the in-place fast path that writes into the already-allocated
//     buffer) must overwrite every slot correctly.
// Together these cover both the scatter (slot placement) and the first-vs-subsequent buffer paths.
TEST_F(ProgramRunArgsTestGen1, CPU_SetRunArgs_NamedPerNodeRTAs_ScatterToDeclarationSlots) {
    NodeCoord node{0, 0};
    ProgramSpec spec = MakeMinimalGen1ValidProgramSpec();
    spec.kernels[0].runtime_arg_schema.runtime_arg_names = {"a", "b", "c"};  // declaration slots 0,1,2
    Program program = MakeProgramFromSpec(*mesh_device_, spec);

    // First call (allocates the buffer). Supply c,a,b out of order on purpose.
    ProgramRunArgs params;
    params.kernel_run_args.push_back(ProgramRunArgs::KernelRunArgs{
        .kernel = KernelSpecName{"dm_kernel"},
        .runtime_arg_values = MakeRuntimeArgsForSingleNode(node, {{"c", 30}, {"a", 10}, {"b", 20}}),
    });
    params.kernel_run_args.push_back(MakeKernelRunArgs(KernelSpecName{"compute_kernel"}, node, {}, {}));
    SetProgramRunArgs(program, params);

    const auto dm_kernel = program.impl().get_kernel_by_spec_name("dm_kernel");
    {
        const auto* rta = dm_kernel->runtime_args_data(node).data();
        ASSERT_GE(dm_kernel->runtime_args_data(node).size(), 3u);
        EXPECT_EQ(rta[0], 10u) << "'a' must land at declaration slot 0 regardless of supplied order";
        EXPECT_EQ(rta[1], 20u) << "'b' must land at declaration slot 1";
        EXPECT_EQ(rta[2], 30u) << "'c' must land at declaration slot 2";
    }

    // Second call hits the in-place subsequent fast path. New values, again out of order.
    ProgramRunArgs params2;
    params2.kernel_run_args.push_back(ProgramRunArgs::KernelRunArgs{
        .kernel = KernelSpecName{"dm_kernel"},
        .runtime_arg_values = MakeRuntimeArgsForSingleNode(node, {{"b", 201}, {"c", 301}, {"a", 101}}),
    });
    params2.kernel_run_args.push_back(MakeKernelRunArgs(KernelSpecName{"compute_kernel"}, node, {}, {}));
    SetProgramRunArgs(program, params2);
    {
        const auto* rta = dm_kernel->runtime_args_data(node).data();
        EXPECT_EQ(rta[0], 101u) << "re-Set must overwrite slot 0 in place";
        EXPECT_EQ(rta[1], 201u) << "re-Set must overwrite slot 1 in place";
        EXPECT_EQ(rta[2], 301u) << "re-Set must overwrite slot 2 in place";
    }
}

// Fast-path coverage for the COMBINED named+vararg per-node layout [named_0,named_1, vararg_0..2].
// The fast path (subsequent Set) patches the named section (scattered by slot, supplied out of
// order) and the positional vararg section (written at offset num_named_rtas) independently and in
// place. Pins that the vararg section lands AFTER the named section and that a re-Set overwrites
// both correctly — distinct from the named-only test above, and the layout most likely to regress
// if the fast/first-call split mishandles the named-vs-vararg offset.
TEST_F(ProgramRunArgsTestGen1, CPU_SetRunArgs_NamedPlusVarargs_FastPathLayout) {
    NodeCoord node{0, 0};
    ProgramSpec spec = MakeMinimalGen1ValidProgramSpec();
    spec.kernels[0].runtime_arg_schema.runtime_arg_names = {"a", "b"};  // declaration slots 0,1
    spec.kernels[0].advanced_options = KernelAdvancedOptions{.num_runtime_varargs = 3};
    Program program = MakeProgramFromSpec(*mesh_device_, spec);

    auto make = [&](uint32_t a, uint32_t b, std::vector<uint32_t> varargs) {
        ProgramRunArgs p;
        p.kernel_run_args.push_back(ProgramRunArgs::KernelRunArgs{
            .kernel = KernelSpecName{"dm_kernel"},
            .runtime_arg_values =
                MakeRuntimeArgsForSingleNode(node, {{"b", b}, {"a", a}}),  // supplied out of declaration order
            .advanced_options = AdvancedKernelRunArgs{.runtime_varargs = {{node, std::move(varargs)}}},
        });
        p.kernel_run_args.push_back(MakeKernelRunArgs(KernelSpecName{"compute_kernel"}, node, {}, {}));
        return p;
    };

    SetProgramRunArgs(program, make(10, 20, {100, 200, 300}));  // first call: allocates the buffer
    const auto dm = program.impl().get_kernel_by_spec_name("dm_kernel");
    {
        const auto& rta = dm->runtime_args_data(node);
        ASSERT_GE(rta.size(), 5u);
        EXPECT_EQ(rta.data()[0], 10u) << "named 'a' at slot 0";
        EXPECT_EQ(rta.data()[1], 20u) << "named 'b' at slot 1";
        EXPECT_EQ(rta.data()[2], 100u) << "vararg 0 follows the named section";
        EXPECT_EQ(rta.data()[3], 200u);
        EXPECT_EQ(rta.data()[4], 300u);
    }

    SetProgramRunArgs(program, make(11, 21, {101, 201, 301}));  // fast path: in-place patch
    {
        const auto* rta = dm->runtime_args_data(node).data();
        EXPECT_EQ(rta[0], 11u) << "fast path overwrites named slot 0";
        EXPECT_EQ(rta[1], 21u);
        EXPECT_EQ(rta[2], 101u) << "fast path overwrites vararg section after named";
        EXPECT_EQ(rta[3], 201u);
        EXPECT_EQ(rta[4], 301u);
    }
}

}  // namespace
}  // namespace tt::tt_metal::experimental
