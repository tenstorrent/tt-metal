// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// SetProgramRunArgs with KernelRunArgs: kernel/node coverage, vararg counts, repeated calls,
// and kernels that may be omitted.

#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <cstdint>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include <tt-metalium/tensor/mesh_tensor.hpp>

#include "impl/kernels/kernel.hpp"
#include "impl/program/program_impl.hpp"
#include "metal2_host_api/test_helpers.hpp"
#include "metal2_host_api/mock_device_fixtures.hpp"
#include "metal2_host_api/run_args_test_helpers.hpp"

namespace tt::tt_metal::experimental {
namespace {

using test_helpers::BindTensorParameterToKernel;
using test_helpers::MakeKernelRunArgs;
using test_helpers::MakeMinimalDFB;
using test_helpers::MakeMinimalGen1ValidProgramSpec;
using test_helpers::MakeMinimalGen2DMKernel;
using test_helpers::MakeMinimalTensorParameter;
using test_helpers::MakeMinimalValidProgramSpec;
using test_helpers::MakeMinimalWorkUnit;
using test_helpers::MakeRunArgsForMinimalSpec;
using test_helpers::MakeSpecWithRTAs;
using test_helpers::ProgramRunArgsTestGen1;
using test_helpers::ProgramRunArgsTestQuasar;

// Create a ProgramSpec with RTA schemas for DM and compute kernels
inline ProgramSpec MakeSpecWithBothKernelRTAs(
    const NodeCoord& /*node*/,
    size_t dm_per_node_rtas,
    size_t dm_common_rtas,
    size_t compute_per_node_rtas,
    size_t compute_common_rtas) {
    ProgramSpec spec = MakeMinimalValidProgramSpec();

    // dm_kernel RTAs
    spec.kernels[0].advanced_options =
        KernelAdvancedOptions{.num_runtime_varargs = dm_per_node_rtas, .num_common_runtime_varargs = dm_common_rtas};

    // compute_kernel RTAs
    spec.kernels[1].advanced_options = KernelAdvancedOptions{
        .num_runtime_varargs = compute_per_node_rtas, .num_common_runtime_varargs = compute_common_rtas};

    return spec;
}

// Create a gen1 ProgramSpec with a specified RTA schema on the DM kernel
inline ProgramSpec MakeGen1SpecWithRTAs(const NodeCoord& /*node*/, size_t num_per_node_rtas, size_t num_common_rtas) {
    ProgramSpec spec = MakeMinimalGen1ValidProgramSpec();

    spec.kernels[0].advanced_options =
        KernelAdvancedOptions{.num_runtime_varargs = num_per_node_rtas, .num_common_runtime_varargs = num_common_rtas};

    // kernels[1] has no varargs (defaults)

    return spec;
}

TEST_F(ProgramRunArgsTestQuasar, CPU_UnknownKernelNameFails) {
    NodeCoord node{0, 0};
    ProgramSpec spec = MakeSpecWithRTAs(node, 0, 0);
    Program program = MakeProgramFromSpec(*mesh_device_, spec);

    ProgramRunArgs params;
    params.kernel_run_args.push_back(ProgramRunArgs::KernelRunArgs{
        .kernel = KernelSpecName{"nonexistent_kernel"},
        .advanced_options =
            AdvancedKernelRunArgs{
                .runtime_varargs = {},
                .common_runtime_varargs = {},
            },
    });

    EXPECT_THAT(
        [&] { SetProgramRunArgs(program, params); },
        ::testing::ThrowsMessage<std::runtime_error>(
            ::testing::HasSubstr("Kernel 'nonexistent_kernel' has no RTA schema registered")));
}

TEST_F(ProgramRunArgsTestQuasar, CPU_InvalidNodeForKernelFails) {
    NodeCoord node{0, 0};
    NodeCoord wrong_node{1, 1};  // Kernel doesn't run on this node
    ProgramSpec spec = MakeSpecWithRTAs(node, 2, 0);
    Program program = MakeProgramFromSpec(*mesh_device_, spec);

    ProgramRunArgs params;
    params.kernel_run_args.push_back(ProgramRunArgs::KernelRunArgs{
        .kernel = KernelSpecName{"dm_kernel"},
        .advanced_options =
            AdvancedKernelRunArgs{
                .runtime_varargs = {{wrong_node, {1, 2}}},  // Wrong node!
                .common_runtime_varargs = {},
            },
    });
    params.kernel_run_args.push_back(MakeKernelRunArgs(KernelSpecName{"compute_kernel"}, node, {}, {}));

    EXPECT_THAT(
        [&] { SetProgramRunArgs(program, params); },
        ::testing::ThrowsMessage<std::runtime_error>(
            ::testing::HasSubstr("Kernel 'dm_kernel' is setting runtime_varargs for node")));
}

TEST_F(ProgramRunArgsTestQuasar, CPU_WrongRuntimeArgsCountFails) {
    NodeCoord node{0, 0};
    ProgramSpec spec = MakeSpecWithRTAs(node, /*num_per_node_rtas=*/3, /*num_common_rtas=*/0);
    Program program = MakeProgramFromSpec(*mesh_device_, spec);

    // Provide wrong count (2 instead of 3)
    auto params = MakeRunArgsForMinimalSpec(node, {1, 2}, {});

    EXPECT_THAT(
        [&] { SetProgramRunArgs(program, params); },
        ::testing::ThrowsMessage<std::runtime_error>(
            ::testing::HasSubstr("expects 3 vararg runtime args, but 2 were provided")));
}

TEST_F(ProgramRunArgsTestQuasar, CPU_WrongCommonRuntimeArgsCountFails) {
    NodeCoord node{0, 0};
    ProgramSpec spec = MakeSpecWithRTAs(node, /*num_per_node_rtas=*/0, /*num_common_rtas=*/2);
    Program program = MakeProgramFromSpec(*mesh_device_, spec);

    // Provide wrong common args count (3 instead of 2)
    auto params = MakeRunArgsForMinimalSpec(node, {}, {1, 2, 3});

    EXPECT_THAT(
        [&] { SetProgramRunArgs(program, params); },
        ::testing::ThrowsMessage<std::runtime_error>(
            ::testing::HasSubstr("expects 2 vararg common runtime args, but 3 were provided")));
}

// TODO: Currently, we require that all kernels in a ProgramSpec have params specified.
// Should relax this to omit kernels with no RTAs or CRTAs.
TEST_F(ProgramRunArgsTestQuasar, CPU_EmptySchemaKernelOmittedFromRunArgsSucceeds) {
    NodeCoord node{0, 0};
    // Both kernels have empty RTA/CRTA schemas — neither has anything to supply per enqueue.
    ProgramSpec spec = MakeSpecWithRTAs(node, 0, 0);
    Program program = MakeProgramFromSpec(*mesh_device_, spec);

    // Only provide params for dm_kernel; omit compute_kernel.
    // Since compute_kernel's schema is empty, this should succeed.
    ProgramRunArgs params;
    params.kernel_run_args.push_back(MakeKernelRunArgs(KernelSpecName{"dm_kernel"}, node, {}, {}));

    EXPECT_NO_THROW({ SetProgramRunArgs(program, params); });
}

TEST_F(ProgramRunArgsTestQuasar, CPU_NonEmptySchemaKernelMissingFromRunArgsFails) {
    NodeCoord node{0, 0};
    // compute_kernel has a non-empty schema (2 vararg RTAs).
    ProgramSpec spec = MakeSpecWithBothKernelRTAs(
        node,
        /*dm_per_node_rtas=*/0,
        /*dm_common_rtas=*/0,
        /*compute_per_node_rtas=*/2,
        /*compute_common_rtas=*/0);
    Program program = MakeProgramFromSpec(*mesh_device_, spec);

    // Only provide params for dm_kernel; omit compute_kernel.
    // Since compute_kernel has declared RTAs, this should fail.
    ProgramRunArgs params;
    params.kernel_run_args.push_back(MakeKernelRunArgs(KernelSpecName{"dm_kernel"}, node, {}, {}));

    EXPECT_THAT(
        [&] { SetProgramRunArgs(program, params); },
        ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr(
            "Kernel 'compute_kernel' is registered in the Program with a non-empty RTA/CRTA schema "
            "but has no runtime parameters specified in ProgramRunArgs")));
}

TEST_F(ProgramRunArgsTestQuasar, CPU_MissingNodeRTAsFails) {
    NodeCoord node{0, 0};
    ProgramSpec spec = MakeSpecWithRTAs(node, /*num_per_node_rtas=*/2, /*num_common_rtas=*/0);
    Program program = MakeProgramFromSpec(*mesh_device_, spec);

    // Don't provide the per-node RTAs (empty runtime_varargs)
    ProgramRunArgs params;
    params.kernel_run_args.push_back(ProgramRunArgs::KernelRunArgs{
        .kernel = KernelSpecName{"dm_kernel"},
        .advanced_options =
            AdvancedKernelRunArgs{
                .runtime_varargs = {},  // Missing node RTAs!
                .common_runtime_varargs = {},
            },
    });
    params.kernel_run_args.push_back(MakeKernelRunArgs(KernelSpecName{"compute_kernel"}, node, {}, {}));

    EXPECT_THAT(
        [&] { SetProgramRunArgs(program, params); },
        ::testing::ThrowsMessage<std::runtime_error>(
            ::testing::HasSubstr("Kernel 'dm_kernel' is missing vararg runtime args for node")));
}

TEST_F(ProgramRunArgsTestQuasar, CPU_DuplicateKernelParamsFails) {
    NodeCoord node{0, 0};
    ProgramSpec spec = MakeSpecWithRTAs(node, 0, 0);
    Program program = MakeProgramFromSpec(*mesh_device_, spec);

    ProgramRunArgs params;
    // Push dm_kernel twice — should trigger duplicate detection.
    params.kernel_run_args.push_back(MakeKernelRunArgs(KernelSpecName{"dm_kernel"}, node, {}, {}));
    params.kernel_run_args.push_back(MakeKernelRunArgs(KernelSpecName{"dm_kernel"}, node, {}, {}));
    params.kernel_run_args.push_back(MakeKernelRunArgs(KernelSpecName{"compute_kernel"}, node, {}, {}));

    EXPECT_THAT(
        [&] { SetProgramRunArgs(program, params); },
        ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr("Duplicate kernel 'dm_kernel'")));
}

TEST_F(ProgramRunArgsTestQuasar, CPU_SetRunArgsSucceeds_ZeroRTAs) {
    NodeCoord node{0, 0};
    ProgramSpec spec = MakeSpecWithRTAs(node, /*num_per_node_rtas=*/0, /*num_common_rtas=*/0);
    Program program = MakeProgramFromSpec(*mesh_device_, spec);

    auto params = MakeRunArgsForMinimalSpec(node, {}, {});

    EXPECT_NO_THROW(SetProgramRunArgs(program, params));
}

TEST_F(ProgramRunArgsTestQuasar, CPU_SetRunArgsSucceeds_PerNodeRTAsOnly) {
    NodeCoord node{0, 0};
    ProgramSpec spec = MakeSpecWithRTAs(node, /*num_per_node_rtas=*/3, /*num_common_rtas=*/0);
    Program program = MakeProgramFromSpec(*mesh_device_, spec);

    auto params = MakeRunArgsForMinimalSpec(node, {100, 200, 300}, {});

    EXPECT_NO_THROW(SetProgramRunArgs(program, params));
}

TEST_F(ProgramRunArgsTestQuasar, CPU_SetRunArgsSucceeds_CommonRTAsOnly) {
    NodeCoord node{0, 0};
    ProgramSpec spec = MakeSpecWithRTAs(node, /*num_per_node_rtas=*/0, /*num_common_rtas=*/2);
    Program program = MakeProgramFromSpec(*mesh_device_, spec);

    auto params = MakeRunArgsForMinimalSpec(node, {}, {10, 20});

    EXPECT_NO_THROW(SetProgramRunArgs(program, params));
}

TEST_F(ProgramRunArgsTestQuasar, CPU_SetRunArgsSucceeds_BothRTATypes) {
    NodeCoord node{0, 0};
    ProgramSpec spec = MakeSpecWithRTAs(node, /*num_per_node_rtas=*/3, /*num_common_rtas=*/2);
    Program program = MakeProgramFromSpec(*mesh_device_, spec);

    auto params = MakeRunArgsForMinimalSpec(node, {100, 200, 300}, {10, 20});

    EXPECT_NO_THROW(SetProgramRunArgs(program, params));
}

TEST_F(ProgramRunArgsTestQuasar, CPU_SetRunArgsSucceeds_BothKernelsWithRTAs) {
    NodeCoord node{0, 0};
    ProgramSpec spec = MakeSpecWithBothKernelRTAs(
        node,
        /*dm_per_node_rtas=*/2,
        /*dm_common_rtas=*/1,
        /*compute_per_node_rtas=*/3,
        /*compute_common_rtas=*/2);
    Program program = MakeProgramFromSpec(*mesh_device_, spec);

    auto params = MakeRunArgsForMinimalSpec(
        node,
        /*dm_per_node_args=*/{1, 2},
        /*dm_common_args=*/{10},
        /*compute_per_node_args=*/{100, 200, 300},
        /*compute_common_args=*/{50, 60});

    EXPECT_NO_THROW(SetProgramRunArgs(program, params));
}

TEST_F(ProgramRunArgsTestQuasar, CPU_SetRunArgsTwice_SameValuesSucceeds) {
    NodeCoord node{0, 0};
    ProgramSpec spec = MakeSpecWithRTAs(node, /*num_per_node_rtas=*/2, /*num_common_rtas=*/1);
    Program program = MakeProgramFromSpec(*mesh_device_, spec);

    auto params = MakeRunArgsForMinimalSpec(node, {100, 200}, {10});

    // First call
    EXPECT_NO_THROW(SetProgramRunArgs(program, params));
    // Second call with same values
    EXPECT_NO_THROW(SetProgramRunArgs(program, params));
}

TEST_F(ProgramRunArgsTestQuasar, CPU_SetRunArgsTwice_DifferentValuesSucceeds) {
    NodeCoord node{0, 0};
    ProgramSpec spec = MakeSpecWithRTAs(node, /*num_per_node_rtas=*/2, /*num_common_rtas=*/1);
    Program program = MakeProgramFromSpec(*mesh_device_, spec);

    auto params1 = MakeRunArgsForMinimalSpec(node, {100, 200}, {10});
    auto params2 = MakeRunArgsForMinimalSpec(node, {300, 400}, {20});  // Different values

    // First call
    EXPECT_NO_THROW(SetProgramRunArgs(program, params1));
    // Second call with different values (same counts)
    EXPECT_NO_THROW(SetProgramRunArgs(program, params2));
}

TEST_F(ProgramRunArgsTestQuasar, CPU_SetRunArgsTwice_ChangingCommonRTACountFails) {
    NodeCoord node{0, 0};
    ProgramSpec spec = MakeSpecWithRTAs(node, /*num_per_node_rtas=*/0, /*num_common_rtas=*/2);
    Program program = MakeProgramFromSpec(*mesh_device_, spec);

    // First call with 2 common RTAs (matches schema)
    auto params1 = MakeRunArgsForMinimalSpec(node, {}, {10, 20});
    EXPECT_NO_THROW(SetProgramRunArgs(program, params1));

    // Change the schema expectation - this is tricky because the schema is fixed
    // The implementation checks against the schema, not the previous call.
    // So changing count will fail on validation, not on the memcpy path.
    // Actually, looking at the code, the schema validation happens first,
    // so if we try to pass wrong count, it fails at validation.

    // Let's verify that passing wrong count still fails (schema validation)
    auto params2 = MakeRunArgsForMinimalSpec(node, {}, {10, 20, 30});  // 3 instead of 2
    EXPECT_THAT(
        [&] { SetProgramRunArgs(program, params2); },
        ::testing::ThrowsMessage<std::runtime_error>(
            ::testing::HasSubstr("expects 2 vararg common runtime args, but 3 were provided")));
}

TEST_F(ProgramRunArgsTestQuasar, CPU_SetRunArgsMultipleTimes_Succeeds) {
    NodeCoord node{0, 0};
    ProgramSpec spec = MakeSpecWithRTAs(node, /*num_per_node_rtas=*/2, /*num_common_rtas=*/2);
    Program program = MakeProgramFromSpec(*mesh_device_, spec);

    // Call SetProgramRunArgs multiple times with varying values
    for (uint32_t i = 0; i < 5; i++) {
        auto params = MakeRunArgsForMinimalSpec(node, {i * 10, i * 20}, {i * 100, i * 200});
        EXPECT_NO_THROW(SetProgramRunArgs(program, params));
    }
}

TEST_F(ProgramRunArgsTestQuasar, CPU_SetRunArgsSucceeds_MultiNodeKernel) {
    // Create a program with kernels spanning multiple nodes
    NodeCoord node0{0, 0};
    NodeCoord node1{1, 0};
    NodeRangeSet all_nodes(std::set<NodeRange>{NodeRange{node0, node0}, NodeRange{node1, node1}});

    ProgramSpec spec;
    spec.name = "multi_node_program";

    // Kernels span both nodes
    auto producer = MakeMinimalGen2DMKernel("producer");
    auto consumer = MakeMinimalGen2DMKernel("consumer");

    // Throw in some varargs (the normal kind, not the weird per-node override kind)
    producer.advanced_options = KernelAdvancedOptions{.num_runtime_varargs = 2, .num_common_runtime_varargs = 1};

    // consumer has no varargs (defaults)

    // Single DFB spanning all nodes
    auto dfb = MakeMinimalDFB("dfb");

    producer.dfb_bindings.push_back(ProducerOf(DFBSpecName{"dfb"}, "out"));
    consumer.dfb_bindings.push_back(ConsumerOf(DFBSpecName{"dfb"}, "in"));

    spec.kernels = {producer, consumer};
    spec.dataflow_buffers = {dfb};
    spec.work_units = std::vector<WorkUnitSpec>{MakeMinimalWorkUnit("work_unit", all_nodes, {"producer", "consumer"})};

    Program program = MakeProgramFromSpec(*mesh_device_, spec);

    // Create run params with per-node RTAs for both nodes
    ProgramRunArgs params;
    params.kernel_run_args.push_back(ProgramRunArgs::KernelRunArgs{
        .kernel = KernelSpecName{"producer"},
        .advanced_options =
            AdvancedKernelRunArgs{
                .runtime_varargs = {{node0, {10, 20}}, {node1, {30, 40}}},
                .common_runtime_varargs = {100},
            },
    });
    params.kernel_run_args.push_back(ProgramRunArgs::KernelRunArgs{
        .kernel = KernelSpecName{"consumer"},
        .advanced_options =
            AdvancedKernelRunArgs{
                .runtime_varargs = {{node0, {}}, {node1, {}}},
                .common_runtime_varargs = {},
            },
    });

    EXPECT_NO_THROW(SetProgramRunArgs(program, params));
}

TEST_F(ProgramRunArgsTestQuasar, CPU_MultiNode_MissingOneNodeFails) {
    // Create a program with kernels spanning multiple nodes
    NodeCoord node0{0, 0};
    NodeCoord node1{1, 0};
    NodeRangeSet all_nodes(std::set<NodeRange>{NodeRange{node0, node0}, NodeRange{node1, node1}});

    ProgramSpec spec;
    spec.name = "multi_node_program";

    auto producer = MakeMinimalGen2DMKernel("producer");
    auto consumer = MakeMinimalGen2DMKernel("consumer");

    // Throw in some varargs (the normal kind, not the weird per-node override kind)
    producer.advanced_options.num_runtime_varargs = 2;
    // consumer has no varargs (defaults)

    // Single DFB spanning all nodes
    auto dfb = MakeMinimalDFB("dfb");

    producer.dfb_bindings.push_back(ProducerOf(DFBSpecName{"dfb"}, "out"));
    consumer.dfb_bindings.push_back(ConsumerOf(DFBSpecName{"dfb"}, "in"));

    spec.kernels = {producer, consumer};
    spec.dataflow_buffers = {dfb};
    spec.work_units = std::vector<WorkUnitSpec>{MakeMinimalWorkUnit("work_unit", all_nodes, {"producer", "consumer"})};

    Program program = MakeProgramFromSpec(*mesh_device_, spec);

    // Only provide RTAs for node0, missing node1
    ProgramRunArgs params;
    params.kernel_run_args.push_back(ProgramRunArgs::KernelRunArgs{
        .kernel = KernelSpecName{"producer"},
        .advanced_options =
            AdvancedKernelRunArgs{
                .runtime_varargs = {{node0, {10, 20}}},  // Missing node1!
                .common_runtime_varargs = {},
            },
    });
    params.kernel_run_args.push_back(ProgramRunArgs::KernelRunArgs{
        .kernel = KernelSpecName{"consumer"},
        .advanced_options =
            AdvancedKernelRunArgs{
                .runtime_varargs = {{node0, {}}, {node1, {}}},
                .common_runtime_varargs = {},
            },
    });

    EXPECT_THAT(
        [&] { SetProgramRunArgs(program, params); },
        ::testing::ThrowsMessage<std::runtime_error>(
            ::testing::HasSubstr("Kernel 'producer' is missing vararg runtime args for node")));
}

// A kernel can legitimately declare no RTAs / CRTAs / CTAs of any kind (named or vararg).
// Verify the whole MakeProgramFromSpec + SetProgramRunArgs pipeline handles this case
// cleanly — no missing-schema TT_FATALs, no empty-buffer write attempts, no validation errors.
TEST_F(ProgramRunArgsTestQuasar, CPU_AllEmptySchemaSucceeds) {
    ProgramSpec spec = MakeMinimalValidProgramSpec();
    // spec.kernels already default to empty runtime_arg_values / common_runtime_arg_values /
    // compile_time_args, num_runtime_varargs = 0, num_common_runtime_varargs = 0,
    // num_runtime_varargs_per_node = nullopt.
    Program program = MakeProgramFromSpec(*mesh_device_, spec);

    ProgramRunArgs params;
    params.kernel_run_args.push_back(ProgramRunArgs::KernelRunArgs{.kernel = KernelSpecName{"dm_kernel"}});
    params.kernel_run_args.push_back(ProgramRunArgs::KernelRunArgs{.kernel = KernelSpecName{"compute_kernel"}});

    EXPECT_NO_THROW(SetProgramRunArgs(program, params));
}

// Regression: a kernel that binds a tensor but declares no scalar args (no named/vararg RTAs or
// CRTAs) may be omitted from kernel_run_args. SetProgramRunArgs must still validate, and must write
// the binding's base address into that kernel's CRTA buffer via its second pass — the binding
// address is per-enqueue state that has to reach the device whether or not the user supplies an
// (otherwise-empty) kernel_run_args entry.
TEST_F(ProgramRunArgsTestQuasar, CPU_BindingOnlyKernelOmittedFromRunArgsSucceeds) {
    // dm_kernel binds a TensorParameter but has an empty RTA/CRTA schema; compute_kernel is empty
    // too. Neither has scalar args, so neither needs a kernel_run_args entry.
    ProgramSpec spec = MakeMinimalValidProgramSpec();
    const std::string tensor_param = "bound_input";
    spec.tensor_parameters = {MakeMinimalTensorParameter(tensor_param)};
    BindTensorParameterToKernel(spec.kernels[0], tensor_param, "in_ta");  // kernels[0] == dm_kernel

    Program program = MakeProgramFromSpec(*mesh_device_, spec);

    // Supply the bound tensor via tensor_args, but provide NO kernel_run_args entry for the binding
    // kernel (the whole point of the relaxation — pre-fix this aborted in ValidateProgramRunArgs).
    MeshTensor tensor = MeshTensor::allocate_on_device(*mesh_device_, spec.tensor_parameters[0].spec);
    ProgramRunArgs params;
    params.tensor_args = {{TensorParamName{tensor_param}, TensorArgument{tensor}}};

    EXPECT_NO_THROW(SetProgramRunArgs(program, params));

    // The second pass must have allocated the kernel's CRTA buffer and written the binding address.
    // MakeMinimalTensorParameter is non-sharded, so the binding is a single address word at offset 0.
    auto kernel = program.impl().get_kernel_by_spec_name("dm_kernel");
    ASSERT_FALSE(kernel->common_runtime_args().empty())
        << "binding-only kernel's CRTA buffer should have been allocated by SetProgramRunArgs";
    EXPECT_EQ(kernel->common_runtime_args()[0], static_cast<uint32_t>(tensor.address()))
        << "binding base address should be written even though the kernel was omitted from kernel_run_args";
}

TEST_F(ProgramRunArgsTestQuasar, CPU_BindingOnlyKernelOmittedFromRunArgsReSetSucceeds) {
    // Regression: SetProgramRunArgs must be re-callable on a program whose binding-only kernel is
    // omitted from kernel_run_args. The first call allocates the kernel's CRTA buffer in the second
    // pass; a second call must patch it in place — set_common_runtime_args fatals if called twice, so
    // the second pass has to use the same first-time/patch logic as the main loop.
    ProgramSpec spec = MakeMinimalValidProgramSpec();
    const std::string tensor_param = "bound_input";
    spec.tensor_parameters = {MakeMinimalTensorParameter(tensor_param)};
    BindTensorParameterToKernel(spec.kernels[0], tensor_param, "in_ta");  // kernels[0] == dm_kernel

    Program program = MakeProgramFromSpec(*mesh_device_, spec);

    MeshTensor tensor1 = MeshTensor::allocate_on_device(*mesh_device_, spec.tensor_parameters[0].spec);
    ProgramRunArgs params1;
    params1.tensor_args = {{TensorParamName{tensor_param}, TensorArgument{tensor1}}};
    EXPECT_NO_THROW(SetProgramRunArgs(program, params1));

    // Second enqueue with a different tensor: must not re-allocate (no fatal), and the binding
    // address must update in place.
    MeshTensor tensor2 = MeshTensor::allocate_on_device(*mesh_device_, spec.tensor_parameters[0].spec);
    ASSERT_NE(tensor1.address(), tensor2.address())
        << "test pre-condition: two live allocations should have distinct addresses";
    ProgramRunArgs params2;
    params2.tensor_args = {{TensorParamName{tensor_param}, TensorArgument{tensor2}}};
    EXPECT_NO_THROW(SetProgramRunArgs(program, params2));

    auto kernel = program.impl().get_kernel_by_spec_name("dm_kernel");
    EXPECT_EQ(kernel->common_runtime_args()[0], static_cast<uint32_t>(tensor2.address()))
        << "second SetProgramRunArgs should patch the binding address in place to the new tensor";
}

TEST_F(ProgramRunArgsTestGen1, CPU_SetRunArgsSucceeds_ZeroRTAs) {
    NodeCoord node{0, 0};
    ProgramSpec spec = MakeGen1SpecWithRTAs(node, 0, 0);
    Program program = MakeProgramFromSpec(*mesh_device_, spec);

    auto params = MakeRunArgsForMinimalSpec(node, {}, {});

    EXPECT_NO_THROW(SetProgramRunArgs(program, params));
}

TEST_F(ProgramRunArgsTestGen1, CPU_SetRunArgsSucceeds_PerNodeAndCommonRTAs) {
    NodeCoord node{0, 0};
    ProgramSpec spec = MakeGen1SpecWithRTAs(node, /*num_per_node_rtas=*/3, /*num_common_rtas=*/2);
    Program program = MakeProgramFromSpec(*mesh_device_, spec);

    auto params = MakeRunArgsForMinimalSpec(node, {100, 200, 300}, {10, 20});

    EXPECT_NO_THROW(SetProgramRunArgs(program, params));
}

TEST_F(ProgramRunArgsTestGen1, CPU_WrongRuntimeArgsCountFails) {
    NodeCoord node{0, 0};
    ProgramSpec spec = MakeGen1SpecWithRTAs(node, /*num_per_node_rtas=*/3, /*num_common_rtas=*/0);
    Program program = MakeProgramFromSpec(*mesh_device_, spec);

    // Provide wrong count (2 instead of 3)
    auto params = MakeRunArgsForMinimalSpec(node, {1, 2}, {});

    EXPECT_THAT(
        [&] { SetProgramRunArgs(program, params); },
        ::testing::ThrowsMessage<std::runtime_error>(
            ::testing::HasSubstr("expects 3 vararg runtime args, but 2 were provided")));
}

}  // namespace
}  // namespace tt::tt_metal::experimental
