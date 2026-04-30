// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

//---------------------------------------------------------------------------------
// Unit tests for the Metal 2.0 Host API: ProgramRunParams and SetProgramRunParameters
// These tests all use mock device (Quasar and Wormhole) for API-level validation.
//
// Test categories:
//   1. Validation tests (parameter validation errors)
//   2. Success tests (basic functionality)
//   3. Repeated call tests (calling SetProgramRunParameters multiple times)
//
//---------------------------------------------------------------------------------
// NOTE: The current tests only attempt to set ProgramRunParams for a Program that has not yet been enqueued for
// execution. The way Program currently works, dispatch data structures are not created until the first enqueue, which
// makes the argument update pathways different the first time vs. subsequent times. This is patently insane. I plan to
// fix this for ProgramSpec / Metal 2.0.
//
//---------------------------------------------------------------------------------
// NOTE: These unit tests use shortcut functions to create minimal valid ProgramSpec
// objects to cut repeated boilerplate (test_helpers.hpp)
//
// This is NOT intended as a recommended pattern for production code!
// See the Metal 2.0 Host API documentation and programming examples for
// recommended patterns for constructing ProgramSpec objects in production code.
//---------------------------------------------------------------------------------

#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_params.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/experimental/context/metal_env.hpp>
#include <tt-metalium/experimental/mock_device.hpp>

#include "test_helpers.hpp"

namespace tt::tt_metal::experimental::metal2_host_api {
namespace {

// Import shared test helpers
using test_helpers::BindDFBToKernel;
using test_helpers::MakeMinimalDFB;
using test_helpers::MakeMinimalDMKernel;
using test_helpers::MakeMinimalGen1ValidProgramSpec;
using test_helpers::MakeMinimalValidProgramSpec;
using test_helpers::MakeMinimalWorkUnit;

// Shorthand for the per-node-override vararg type (needed at call sites because
// std::optional<T> can't be brace-init from an initializer-list of T's elements).
using NumVarargsPerNode = KernelSpec::RuntimeArgSchema::NumVarargsPerNode;

// ============================================================================
// Test Fixtures
// ============================================================================

// Test fixture for ProgramRunParams on Quasar - uses Quasar mock device
class ProgramRunParamsTestQuasar : public ::testing::Test {
protected:
    void SetUp() override {
        //  Configure global mock mode for Quasar
        experimental::configure_mock_mode(tt::ARCH::QUASAR, 1);
    }
    void TearDown() override { experimental::disable_mock_mode(); }
};

// ============================================================================
// Test Utilities
// ============================================================================

// Create a ProgramSpec with specified RTA schema for the DM kernel
// (The compute kernel has no RTAs)
inline ProgramSpec MakeSpecWithRTAs(const NodeCoord& /*node*/, size_t num_per_node_rtas, size_t num_common_rtas) {
    ProgramSpec spec = MakeMinimalValidProgramSpec();

    // Set the RTA schema on the dm_kernel (first kernel)
    spec.kernels[0].runtime_arguments_schema.num_runtime_varargs = num_per_node_rtas;
    spec.kernels[0].runtime_arguments_schema.num_common_runtime_varargs = num_common_rtas;

    // compute_kernel has no RTAs (defaults: 0 / 0)

    return spec;
}

// Create a ProgramSpec with RTA schemas for DM and compute kernels
inline ProgramSpec MakeSpecWithBothKernelRTAs(
    const NodeCoord& /*node*/,
    size_t dm_per_node_rtas,
    size_t dm_common_rtas,
    size_t compute_per_node_rtas,
    size_t compute_common_rtas) {
    ProgramSpec spec = MakeMinimalValidProgramSpec();

    // dm_kernel RTAs
    spec.kernels[0].runtime_arguments_schema.num_runtime_varargs = dm_per_node_rtas;
    spec.kernels[0].runtime_arguments_schema.num_common_runtime_varargs = dm_common_rtas;

    // compute_kernel RTAs
    spec.kernels[1].runtime_arguments_schema.num_runtime_varargs = compute_per_node_rtas;
    spec.kernels[1].runtime_arguments_schema.num_common_runtime_varargs = compute_common_rtas;

    return spec;
}

// Helper to create ProgramRunParams for a single kernel
inline ProgramRunParams::KernelRunParams MakeKernelRunParams(
    const KernelSpecName& kernel_name,
    const NodeCoord& node,
    const std::vector<uint32_t>& per_node_args,
    const std::vector<uint32_t>& common_args) {
    return ProgramRunParams::KernelRunParams{
        .kernel_spec_name = kernel_name,
        .runtime_varargs = {{node, per_node_args}},
        .common_runtime_varargs = common_args,
    };
}

// Helper to create complete ProgramRunParams for the minimal spec (both kernels)
inline ProgramRunParams MakeRunParamsForMinimalSpec(
    const NodeCoord& node,
    const std::vector<uint32_t>& dm_per_node_args,
    const std::vector<uint32_t>& dm_common_args,
    const std::vector<uint32_t>& compute_per_node_args = {},
    const std::vector<uint32_t>& compute_common_args = {}) {
    ProgramRunParams params;
    params.kernel_run_params.push_back(MakeKernelRunParams("dm_kernel", node, dm_per_node_args, dm_common_args));
    params.kernel_run_params.push_back(
        MakeKernelRunParams("compute_kernel", node, compute_per_node_args, compute_common_args));
    return params;
}

// ============================================================================
// SECTION 1: Validation Tests (expect failure)
// ============================================================================

TEST_F(ProgramRunParamsTestQuasar, UnknownKernelNameFails) {
    NodeCoord node{0, 0};
    ProgramSpec spec = MakeSpecWithRTAs(node, 0, 0);
    Program program = MakeProgramFromSpec(spec);

    ProgramRunParams params;
    params.kernel_run_params.push_back({
        .kernel_spec_name = "nonexistent_kernel",
        .runtime_varargs = {},
        .common_runtime_varargs = {},
    });

    EXPECT_THAT(
        [&] { SetProgramRunParameters(program, params); },
        ::testing::ThrowsMessage<std::runtime_error>(
            ::testing::HasSubstr("Kernel 'nonexistent_kernel' has no RTA schema registered")));
}

TEST_F(ProgramRunParamsTestQuasar, InvalidNodeForKernelFails) {
    NodeCoord node{0, 0};
    NodeCoord wrong_node{1, 1};  // Kernel doesn't run on this node
    ProgramSpec spec = MakeSpecWithRTAs(node, 2, 0);
    Program program = MakeProgramFromSpec(spec);

    ProgramRunParams params;
    params.kernel_run_params.push_back({
        .kernel_spec_name = "dm_kernel",
        .runtime_varargs = {{wrong_node, {1, 2}}},  // Wrong node!
        .common_runtime_varargs = {},
    });
    params.kernel_run_params.push_back(MakeKernelRunParams("compute_kernel", node, {}, {}));

    EXPECT_THAT(
        [&] { SetProgramRunParameters(program, params); },
        ::testing::ThrowsMessage<std::runtime_error>(
            ::testing::HasSubstr("Kernel 'dm_kernel' is setting runtime_varargs for node")));
}

TEST_F(ProgramRunParamsTestQuasar, WrongRuntimeArgsCountFails) {
    NodeCoord node{0, 0};
    ProgramSpec spec = MakeSpecWithRTAs(node, /*num_per_node_rtas=*/3, /*num_common_rtas=*/0);
    Program program = MakeProgramFromSpec(spec);

    // Provide wrong count (2 instead of 3)
    auto params = MakeRunParamsForMinimalSpec(node, {1, 2}, {});

    EXPECT_THAT(
        [&] { SetProgramRunParameters(program, params); },
        ::testing::ThrowsMessage<std::runtime_error>(
            ::testing::HasSubstr("expects 3 vararg runtime args, but 2 were provided")));
}

TEST_F(ProgramRunParamsTestQuasar, WrongCommonRuntimeArgsCountFails) {
    NodeCoord node{0, 0};
    ProgramSpec spec = MakeSpecWithRTAs(node, /*num_per_node_rtas=*/0, /*num_common_rtas=*/2);
    Program program = MakeProgramFromSpec(spec);

    // Provide wrong common args count (3 instead of 2)
    auto params = MakeRunParamsForMinimalSpec(node, {}, {1, 2, 3});

    EXPECT_THAT(
        [&] { SetProgramRunParameters(program, params); },
        ::testing::ThrowsMessage<std::runtime_error>(
            ::testing::HasSubstr("expects 2 vararg common runtime args, but 3 were provided")));
}

// TODO: Currently, we require that all kernels in a ProgramSpec have params specified.
// Should relax this to omit kernels with no RTAs or CRTAs.
TEST_F(ProgramRunParamsTestQuasar, MissingKernelParamsFails) {
    NodeCoord node{0, 0};
    ProgramSpec spec = MakeSpecWithRTAs(node, 0, 0);
    Program program = MakeProgramFromSpec(spec);

    // Only provide params for dm_kernel, missing compute_kernel
    ProgramRunParams params;
    params.kernel_run_params.push_back(MakeKernelRunParams("dm_kernel", node, {}, {}));
    // Missing: compute_kernel params

    EXPECT_THAT(
        [&] { SetProgramRunParameters(program, params); },
        ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr(
            "Kernel 'compute_kernel' is registered in the Program but has no runtime parameters")));
}

TEST_F(ProgramRunParamsTestQuasar, DuplicateKernelParamsFails) {
    NodeCoord node{0, 0};
    ProgramSpec spec = MakeSpecWithRTAs(node, 0, 0);
    Program program = MakeProgramFromSpec(spec);

    // Provide params for dm_kernel twice
    ProgramRunParams params;
    params.kernel_run_params.push_back(MakeKernelRunParams("dm_kernel", node, {}, {}));
    params.kernel_run_params.push_back(MakeKernelRunParams("dm_kernel", node, {}, {}));  // Duplicate!
    params.kernel_run_params.push_back(MakeKernelRunParams("compute_kernel", node, {}, {}));

    EXPECT_THAT(
        [&] { SetProgramRunParameters(program, params); },
        ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr("Duplicate kernel_spec_name 'dm_kernel'")));
}

TEST_F(ProgramRunParamsTestQuasar, MissingNodeRTAsFails) {
    NodeCoord node{0, 0};
    ProgramSpec spec = MakeSpecWithRTAs(node, /*num_per_node_rtas=*/2, /*num_common_rtas=*/0);
    Program program = MakeProgramFromSpec(spec);

    // Don't provide the per-node RTAs (empty runtime_varargs)
    ProgramRunParams params;
    params.kernel_run_params.push_back({
        .kernel_spec_name = "dm_kernel",
        .runtime_varargs = {},  // Missing node RTAs!
        .common_runtime_varargs = {},
    });
    params.kernel_run_params.push_back(MakeKernelRunParams("compute_kernel", node, {}, {}));

    EXPECT_THAT(
        [&] { SetProgramRunParameters(program, params); },
        ::testing::ThrowsMessage<std::runtime_error>(
            ::testing::HasSubstr("Kernel 'dm_kernel' is missing vararg runtime args for node")));
}

// TODO: Replace when feature is implemented
TEST_F(ProgramRunParamsTestQuasar, DFBSizeOverrideFails) {
    NodeCoord node{0, 0};
    ProgramSpec spec = MakeSpecWithRTAs(node, 0, 0);
    Program program = MakeProgramFromSpec(spec);

    auto params = MakeRunParamsForMinimalSpec(node, {}, {});

    // Add DFB run params with size override (not implemented)
    params.dfb_run_params.push_back({
        .dfb_spec_name = "dfb_0",
        .entry_size = 2048,  // Override - not implemented!
    });

    EXPECT_THAT(
        [&] { SetProgramRunParameters(program, params); },
        ::testing::ThrowsMessage<std::runtime_error>(
            ::testing::HasSubstr("DFB size overrides are not yet implemented")));
}

// TODO: Replace when feature is implemented
TEST_F(ProgramRunParamsTestQuasar, DFBNumEntriesOverrideFails) {
    NodeCoord node{0, 0};
    ProgramSpec spec = MakeSpecWithRTAs(node, 0, 0);
    Program program = MakeProgramFromSpec(spec);

    auto params = MakeRunParamsForMinimalSpec(node, {}, {});

    // Add DFB run params with num_entries override (not implemented)
    params.dfb_run_params.push_back({
        .dfb_spec_name = "dfb_0",
        .num_entries = 4,  // Override - not implemented!
    });

    EXPECT_THAT(
        [&] { SetProgramRunParameters(program, params); },
        ::testing::ThrowsMessage<std::runtime_error>(
            ::testing::HasSubstr("DFB size overrides are not yet implemented")));
}

TEST_F(ProgramRunParamsTestQuasar, DuplicateDFBParamsFails) {
    NodeCoord node{0, 0};
    ProgramSpec spec = MakeSpecWithRTAs(node, 0, 0);
    Program program = MakeProgramFromSpec(spec);

    auto params = MakeRunParamsForMinimalSpec(node, {}, {});

    // Add the same DFB twice
    params.dfb_run_params.push_back({.dfb_spec_name = "dfb_0"});
    params.dfb_run_params.push_back({.dfb_spec_name = "dfb_0"});  // Duplicate!

    EXPECT_THAT(
        [&] { SetProgramRunParameters(program, params); },
        ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr("Duplicate dfb_spec_name 'dfb_0'")));
}

TEST_F(ProgramRunParamsTestQuasar, DuplicateNodeCoordInRuntimeArgsFails) {
    NodeCoord node{0, 0};
    ProgramSpec spec = MakeSpecWithRTAs(node, /*num_per_node_rtas=*/2, /*num_common_rtas=*/0);
    Program program = MakeProgramFromSpec(spec);

    // Provide runtime_varargs with duplicate node_coord entries
    ProgramRunParams params;
    params.kernel_run_params.push_back({
        .kernel_spec_name = "dm_kernel",
        .runtime_varargs = {{node, {1, 2}}, {node, {3, 4}}},  // Duplicate node!
        .common_runtime_varargs = {},
    });
    params.kernel_run_params.push_back(MakeKernelRunParams("compute_kernel", node, {}, {}));

    EXPECT_THAT(
        [&] { SetProgramRunParameters(program, params); },
        ::testing::ThrowsMessage<std::runtime_error>(
            ::testing::HasSubstr("in runtime_varargs for kernel 'dm_kernel'")));
}

// ============================================================================
// SECTION 2: Success Tests (basic functionality)
// ============================================================================

TEST_F(ProgramRunParamsTestQuasar, SetRunParamsSucceeds_ZeroRTAs) {
    NodeCoord node{0, 0};
    ProgramSpec spec = MakeSpecWithRTAs(node, /*num_per_node_rtas=*/0, /*num_common_rtas=*/0);
    Program program = MakeProgramFromSpec(spec);

    auto params = MakeRunParamsForMinimalSpec(node, {}, {});

    EXPECT_NO_THROW(SetProgramRunParameters(program, params));
}

TEST_F(ProgramRunParamsTestQuasar, SetRunParamsSucceeds_PerNodeRTAsOnly) {
    NodeCoord node{0, 0};
    ProgramSpec spec = MakeSpecWithRTAs(node, /*num_per_node_rtas=*/3, /*num_common_rtas=*/0);
    Program program = MakeProgramFromSpec(spec);

    auto params = MakeRunParamsForMinimalSpec(node, {100, 200, 300}, {});

    EXPECT_NO_THROW(SetProgramRunParameters(program, params));
}

TEST_F(ProgramRunParamsTestQuasar, SetRunParamsSucceeds_CommonRTAsOnly) {
    NodeCoord node{0, 0};
    ProgramSpec spec = MakeSpecWithRTAs(node, /*num_per_node_rtas=*/0, /*num_common_rtas=*/2);
    Program program = MakeProgramFromSpec(spec);

    auto params = MakeRunParamsForMinimalSpec(node, {}, {10, 20});

    EXPECT_NO_THROW(SetProgramRunParameters(program, params));
}

TEST_F(ProgramRunParamsTestQuasar, SetRunParamsSucceeds_BothRTATypes) {
    NodeCoord node{0, 0};
    ProgramSpec spec = MakeSpecWithRTAs(node, /*num_per_node_rtas=*/3, /*num_common_rtas=*/2);
    Program program = MakeProgramFromSpec(spec);

    auto params = MakeRunParamsForMinimalSpec(node, {100, 200, 300}, {10, 20});

    EXPECT_NO_THROW(SetProgramRunParameters(program, params));
}

TEST_F(ProgramRunParamsTestQuasar, SetRunParamsSucceeds_BothKernelsWithRTAs) {
    NodeCoord node{0, 0};
    ProgramSpec spec = MakeSpecWithBothKernelRTAs(
        node,
        /*dm_per_node_rtas=*/2,
        /*dm_common_rtas=*/1,
        /*compute_per_node_rtas=*/3,
        /*compute_common_rtas=*/2);
    Program program = MakeProgramFromSpec(spec);

    auto params = MakeRunParamsForMinimalSpec(
        node,
        /*dm_per_node_args=*/{1, 2},
        /*dm_common_args=*/{10},
        /*compute_per_node_args=*/{100, 200, 300},
        /*compute_common_args=*/{50, 60});

    EXPECT_NO_THROW(SetProgramRunParameters(program, params));
}

TEST_F(ProgramRunParamsTestQuasar, SetRunParamsSucceeds_DFBRunParamsWithNoOverrides) {
    NodeCoord node{0, 0};
    ProgramSpec spec = MakeSpecWithRTAs(node, 0, 0);
    Program program = MakeProgramFromSpec(spec);

    auto params = MakeRunParamsForMinimalSpec(node, {}, {});

    // DFB run params with no overrides is allowed
    params.dfb_run_params.push_back({
        .dfb_spec_name = "dfb_0",
        // No overrides - both entry_size and num_entries are nullopt
    });

    EXPECT_NO_THROW(SetProgramRunParameters(program, params));
}

// ============================================================================
// SECTION 3: Repeated Call Tests
// ============================================================================

TEST_F(ProgramRunParamsTestQuasar, SetRunParamsTwice_SameValuesSucceeds) {
    NodeCoord node{0, 0};
    ProgramSpec spec = MakeSpecWithRTAs(node, /*num_per_node_rtas=*/2, /*num_common_rtas=*/1);
    Program program = MakeProgramFromSpec(spec);

    auto params = MakeRunParamsForMinimalSpec(node, {100, 200}, {10});

    // First call
    EXPECT_NO_THROW(SetProgramRunParameters(program, params));
    // Second call with same values
    EXPECT_NO_THROW(SetProgramRunParameters(program, params));
}

TEST_F(ProgramRunParamsTestQuasar, SetRunParamsTwice_DifferentValuesSucceeds) {
    NodeCoord node{0, 0};
    ProgramSpec spec = MakeSpecWithRTAs(node, /*num_per_node_rtas=*/2, /*num_common_rtas=*/1);
    Program program = MakeProgramFromSpec(spec);

    auto params1 = MakeRunParamsForMinimalSpec(node, {100, 200}, {10});
    auto params2 = MakeRunParamsForMinimalSpec(node, {300, 400}, {20});  // Different values

    // First call
    EXPECT_NO_THROW(SetProgramRunParameters(program, params1));
    // Second call with different values (same counts)
    EXPECT_NO_THROW(SetProgramRunParameters(program, params2));
}

TEST_F(ProgramRunParamsTestQuasar, SetRunParamsTwice_ChangingCommonRTACountFails) {
    NodeCoord node{0, 0};
    ProgramSpec spec = MakeSpecWithRTAs(node, /*num_per_node_rtas=*/0, /*num_common_rtas=*/2);
    Program program = MakeProgramFromSpec(spec);

    // First call with 2 common RTAs (matches schema)
    auto params1 = MakeRunParamsForMinimalSpec(node, {}, {10, 20});
    EXPECT_NO_THROW(SetProgramRunParameters(program, params1));

    // Change the schema expectation - this is tricky because the schema is fixed
    // The implementation checks against the schema, not the previous call.
    // So changing count will fail on validation, not on the memcpy path.
    // Actually, looking at the code, the schema validation happens first,
    // so if we try to pass wrong count, it fails at validation.

    // Let's verify that passing wrong count still fails (schema validation)
    auto params2 = MakeRunParamsForMinimalSpec(node, {}, {10, 20, 30});  // 3 instead of 2
    EXPECT_THAT(
        [&] { SetProgramRunParameters(program, params2); },
        ::testing::ThrowsMessage<std::runtime_error>(
            ::testing::HasSubstr("expects 2 vararg common runtime args, but 3 were provided")));
}

TEST_F(ProgramRunParamsTestQuasar, SetRunParamsMultipleTimes_Succeeds) {
    NodeCoord node{0, 0};
    ProgramSpec spec = MakeSpecWithRTAs(node, /*num_per_node_rtas=*/2, /*num_common_rtas=*/2);
    Program program = MakeProgramFromSpec(spec);

    // Call SetProgramRunParameters multiple times with varying values
    for (uint32_t i = 0; i < 5; i++) {
        auto params = MakeRunParamsForMinimalSpec(node, {i * 10, i * 20}, {i * 100, i * 200});
        EXPECT_NO_THROW(SetProgramRunParameters(program, params));
    }
}

// ============================================================================
// SECTION 4: Multi-Node Tests
// ============================================================================

TEST_F(ProgramRunParamsTestQuasar, SetRunParamsSucceeds_MultiNodeKernel) {
    // Create a program with kernels spanning multiple nodes
    NodeCoord node0{0, 0};
    NodeCoord node1{1, 0};
    NodeRangeSet all_nodes(std::set<NodeRange>{NodeRange{node0, node0}, NodeRange{node1, node1}});

    ProgramSpec spec;
    spec.program_id = "multi_node_program";

    // Kernels span both nodes
    auto producer = MakeMinimalDMKernel("producer");
    auto consumer = MakeMinimalDMKernel("consumer");

    // Throw in some varargs (the normal kind, not the weird per-node override kind)
    producer.runtime_arguments_schema.num_runtime_varargs = 2;
    producer.runtime_arguments_schema.num_common_runtime_varargs = 1;

    // consumer has no varargs (defaults)

    // Single DFB spanning all nodes
    auto dfb = MakeMinimalDFB("dfb");

    BindDFBToKernel(producer, "dfb", "out", KernelSpec::DFBEndpointType::PRODUCER);
    BindDFBToKernel(consumer, "dfb", "in", KernelSpec::DFBEndpointType::CONSUMER);

    spec.kernels = {producer, consumer};
    spec.dataflow_buffers = {dfb};
    spec.work_units = std::vector<WorkUnitSpec>{MakeMinimalWorkUnit("work_unit", all_nodes, {"producer", "consumer"})};

    Program program = MakeProgramFromSpec(spec);

    // Create run params with per-node RTAs for both nodes
    ProgramRunParams params;
    params.kernel_run_params.push_back({
        .kernel_spec_name = "producer",
        .runtime_varargs = {{node0, {10, 20}}, {node1, {30, 40}}},
        .common_runtime_varargs = {100},
    });
    params.kernel_run_params.push_back({
        .kernel_spec_name = "consumer",
        .runtime_varargs = {{node0, {}}, {node1, {}}},
        .common_runtime_varargs = {},
    });

    EXPECT_NO_THROW(SetProgramRunParameters(program, params));
}

TEST_F(ProgramRunParamsTestQuasar, MultiNode_MissingOneNodeFails) {
    // Create a program with kernels spanning multiple nodes
    NodeCoord node0{0, 0};
    NodeCoord node1{1, 0};
    NodeRangeSet all_nodes(std::set<NodeRange>{NodeRange{node0, node0}, NodeRange{node1, node1}});

    ProgramSpec spec;
    spec.program_id = "multi_node_program";

    auto producer = MakeMinimalDMKernel("producer");
    auto consumer = MakeMinimalDMKernel("consumer");

    // Throw in some varargs (the normal kind, not the weird per-node override kind)
    producer.runtime_arguments_schema.num_runtime_varargs = 2;
    // consumer has no varargs (defaults)

    // Single DFB spanning all nodes
    auto dfb = MakeMinimalDFB("dfb");

    BindDFBToKernel(producer, "dfb", "out", KernelSpec::DFBEndpointType::PRODUCER);
    BindDFBToKernel(consumer, "dfb", "in", KernelSpec::DFBEndpointType::CONSUMER);

    spec.kernels = {producer, consumer};
    spec.dataflow_buffers = {dfb};
    spec.work_units = std::vector<WorkUnitSpec>{MakeMinimalWorkUnit("work_unit", all_nodes, {"producer", "consumer"})};

    Program program = MakeProgramFromSpec(spec);

    // Only provide RTAs for node0, missing node1
    ProgramRunParams params;
    params.kernel_run_params.push_back({
        .kernel_spec_name = "producer",
        .runtime_varargs = {{node0, {10, 20}}},  // Missing node1!
        .common_runtime_varargs = {},
    });
    params.kernel_run_params.push_back({
        .kernel_spec_name = "consumer",
        .runtime_varargs = {{node0, {}}, {node1, {}}},
        .common_runtime_varargs = {},
    });

    EXPECT_THAT(
        [&] { SetProgramRunParameters(program, params); },
        ::testing::ThrowsMessage<std::runtime_error>(
            ::testing::HasSubstr("Kernel 'producer' is missing vararg runtime args for node")));
}

// ============================================================================
// SECTION 5: Gen1 (WH/BH) Tests
// ============================================================================
// The RTA validation logic in SetProgramRunParameters is arch-agnostic; these
// tests verify the full roundtrip works on gen1 programs and that validation
// still fires when it should.

// Test fixture for ProgramRunParams on Wormhole - uses WORMHOLE_B0 mock device
// ============================================================================
// SECTION 4: Named RTA / CRTA Tests (Quasar)
// ============================================================================

// Make a ProgramSpec where the DM kernel has a named-RTA / named-CRTA schema.
inline ProgramSpec MakeSpecWithNamedArgs(
    const NodeCoord& node, const std::vector<std::string>& named_rtas, const std::vector<std::string>& named_crtas) {
    ProgramSpec spec = MakeMinimalValidProgramSpec();
    spec.kernels[0].runtime_arguments_schema.named_runtime_args = named_rtas;
    spec.kernels[0].runtime_arguments_schema.named_common_runtime_args = named_crtas;
    (void)node;  // node inherited from MakeMinimalValidProgramSpec (0,0)
    return spec;
}

TEST_F(ProgramRunParamsTestQuasar, NamedRTAsAndCRTAsSucceed) {
    NodeCoord node{0, 0};
    ProgramSpec spec = MakeSpecWithNamedArgs(node, {"input_ptr", "output_ptr"}, {"tile_count"});
    Program program = MakeProgramFromSpec(spec);

    ProgramRunParams params;
    params.kernel_run_params.push_back({
        .kernel_spec_name = "dm_kernel",
        .named_runtime_args = {{.node = node, .args = {{"input_ptr", 0x1000}, {"output_ptr", 0x2000}}}},
        .named_common_runtime_args = {{"tile_count", 64}},
    });
    params.kernel_run_params.push_back(MakeKernelRunParams("compute_kernel", node, {}, {}));

    EXPECT_NO_THROW(SetProgramRunParameters(program, params));
}

TEST_F(ProgramRunParamsTestQuasar, MissingNamedRTAForNodeFails) {
    NodeCoord node{0, 0};
    ProgramSpec spec = MakeSpecWithNamedArgs(node, {"input_ptr"}, {});
    Program program = MakeProgramFromSpec(spec);

    ProgramRunParams params;
    params.kernel_run_params.push_back({
        .kernel_spec_name = "dm_kernel",
        // No named_runtime_args for node (0,0) at all — but schema declares one.
    });
    params.kernel_run_params.push_back(MakeKernelRunParams("compute_kernel", node, {}, {}));

    EXPECT_THAT(
        [&] { SetProgramRunParameters(program, params); },
        ::testing::ThrowsMessage<std::runtime_error>(
            ::testing::HasSubstr("has named RTAs declared but no named_runtime_args provided for node")));
}

TEST_F(ProgramRunParamsTestQuasar, MissingDeclaredNamedRTANameFails) {
    NodeCoord node{0, 0};
    ProgramSpec spec = MakeSpecWithNamedArgs(node, {"input_ptr", "output_ptr"}, {});
    Program program = MakeProgramFromSpec(spec);

    ProgramRunParams params;
    params.kernel_run_params.push_back({
        .kernel_spec_name = "dm_kernel",
        // Only one name provided — output_ptr missing.
        .named_runtime_args = {{.node = node, .args = {{"input_ptr", 0x1000}}}},
    });
    params.kernel_run_params.push_back(MakeKernelRunParams("compute_kernel", node, {}, {}));

    EXPECT_THAT(
        [&] { SetProgramRunParameters(program, params); },
        ::testing::ThrowsMessage<std::runtime_error>(
            ::testing::HasSubstr("expects 2 named RTAs, but 1 were provided")));
}

TEST_F(ProgramRunParamsTestQuasar, UndeclaredNamedRTAFails) {
    NodeCoord node{0, 0};
    ProgramSpec spec = MakeSpecWithNamedArgs(node, {"input_ptr"}, {});
    Program program = MakeProgramFromSpec(spec);

    ProgramRunParams params;
    params.kernel_run_params.push_back({
        .kernel_spec_name = "dm_kernel",
        .named_runtime_args = {{.node = node, .args = {{"input_ptr", 0x1000}, {"not_in_schema", 0}}}},
    });
    params.kernel_run_params.push_back(MakeKernelRunParams("compute_kernel", node, {}, {}));

    EXPECT_THAT(
        [&] { SetProgramRunParameters(program, params); },
        ::testing::ThrowsMessage<std::runtime_error>(
            ::testing::HasSubstr("expects 1 named RTAs, but 2 were provided")));
}

TEST_F(ProgramRunParamsTestQuasar, NamedCRTACountMismatchFails) {
    NodeCoord node{0, 0};
    ProgramSpec spec = MakeSpecWithNamedArgs(node, {}, {"tile_count", "scale"});
    Program program = MakeProgramFromSpec(spec);

    ProgramRunParams params;
    params.kernel_run_params.push_back({
        .kernel_spec_name = "dm_kernel",
        // Only one CRTA provided; schema declares two.
        .named_common_runtime_args = {{"tile_count", 4}},
    });
    params.kernel_run_params.push_back(MakeKernelRunParams("compute_kernel", node, {}, {}));

    EXPECT_THAT(
        [&] { SetProgramRunParameters(program, params); },
        ::testing::ThrowsMessage<std::runtime_error>(
            ::testing::HasSubstr("expects 2 named CRTAs, but 1 were provided")));
}

// ============================================================================
// Vararg-only regression tests
// ============================================================================
//
// These document the "legacy kernel migrated to Metal 2.0" pattern: all args as positional
// varargs, no named RTAs/CRTAs/CTAs. This is expected to be the dominant migration shape —
// users who port existing kernels will default to leaving everything as positional args
// because that's how the old kernel source reads them. These tests are deliberately the
// strongest regression canaries in this file; breaking them will break migration.

TEST_F(ProgramRunParamsTestQuasar, VarargOnlyMultiNodeDifferingCountsSucceeds) {
    // A kernel on two nodes with DIFFERENT vararg counts per node. Exercises the advanced
    // num_runtime_varargs_per_node override path. The RTA dispatch buffer must be sized
    // per-node, which is a common failure mode for layout bugs.
    NodeCoord node_a{0, 0};
    NodeCoord node_b{1, 0};
    NodeRangeSet nodes{std::vector<NodeRange>{NodeRange{node_a, node_a}, NodeRange{node_b, node_b}}};

    ProgramSpec spec;
    spec.program_id = "vararg_differing_counts";
    auto kernel = MakeMinimalDMKernel("dm_kernel");
    kernel.runtime_arguments_schema.num_runtime_varargs_per_node = NumVarargsPerNode{{node_a, 2}, {node_b, 5}};
    spec.kernels = {kernel};
    spec.work_units = std::vector<WorkUnitSpec>{MakeMinimalWorkUnit("work_unit_0", nodes, {"dm_kernel"})};
    Program program = MakeProgramFromSpec(spec);

    ProgramRunParams params;
    params.kernel_run_params.push_back({
        .kernel_spec_name = "dm_kernel",
        .runtime_varargs = {{node_a, {10, 20}}, {node_b, {100, 200, 300, 400, 500}}},
    });
    EXPECT_NO_THROW(SetProgramRunParameters(program, params));
}

TEST_F(ProgramRunParamsTestQuasar, VarargPerNodeOverrideMixedEntryTypesSucceeds) {
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
    spec.program_id = "vararg_mixed_entry_types";
    auto kernel = MakeMinimalDMKernel("dm_kernel");
    // Nodes a and b share count 3 (declared via a NodeRangeSet entry).
    // Node c has count 5 (declared via a NodeCoord entry).
    kernel.runtime_arguments_schema.num_runtime_varargs_per_node = NumVarargsPerNode{{ab, 3}, {node_c, 5}};
    spec.kernels = {kernel};
    spec.work_units = std::vector<WorkUnitSpec>{MakeMinimalWorkUnit("work_unit_0", all_nodes, {"dm_kernel"})};
    Program program = MakeProgramFromSpec(spec);

    ProgramRunParams params;
    params.kernel_run_params.push_back({
        .kernel_spec_name = "dm_kernel",
        .runtime_varargs = {{node_a, {1, 2, 3}}, {node_b, {10, 20, 30}}, {node_c, {100, 200, 300, 400, 500}}},
    });
    EXPECT_NO_THROW(SetProgramRunParameters(program, params));
}

TEST_F(ProgramRunParamsTestQuasar, VarargScalarDefaultWithSparseOverrideSucceeds) {
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
    spec.program_id = "vararg_scalar_with_sparse_override";
    auto kernel = MakeMinimalDMKernel("dm_kernel");
    kernel.runtime_arguments_schema.num_runtime_varargs = 2;  // default for unlisted nodes
    kernel.runtime_arguments_schema.num_runtime_varargs_per_node =
        NumVarargsPerNode{{node_c, 5}};  // node_c is the exception
    spec.kernels = {kernel};
    spec.work_units = std::vector<WorkUnitSpec>{MakeMinimalWorkUnit("work_unit_0", all_nodes, {"dm_kernel"})};
    Program program = MakeProgramFromSpec(spec);

    ProgramRunParams params;
    params.kernel_run_params.push_back({
        .kernel_spec_name = "dm_kernel",
        .runtime_varargs =
            {
                {node_a, {1, 2}},                     // scalar default (2 args)
                {node_b, {10, 20}},                   // scalar default (2 args)
                {node_c, {100, 200, 300, 400, 500}},  // override (5 args)
            },
    });
    EXPECT_NO_THROW(SetProgramRunParameters(program, params));
}

TEST_F(ProgramRunParamsTestQuasar, VarargSparseOverrideZeroErasesScalarDefault) {
    // An explicit override of 0 on a node erases the scalar default for that node.
    // Regression canary for the expansion logic: if the erase is missing, the node would
    // carry the scalar-default count and run-params validation would either require an
    // empty value list or error on count mismatch.
    NodeCoord node_a{0, 0};
    NodeCoord node_b{1, 0};
    NodeRangeSet both{std::vector<NodeRange>{NodeRange{node_a, node_a}, NodeRange{node_b, node_b}}};

    ProgramSpec spec;
    spec.program_id = "vararg_zero_override";
    auto kernel = MakeMinimalDMKernel("dm_kernel");
    kernel.runtime_arguments_schema.num_runtime_varargs = 3;
    kernel.runtime_arguments_schema.num_runtime_varargs_per_node =
        NumVarargsPerNode{{node_b, 0}};  // node_b: no varargs despite scalar default
    spec.kernels = {kernel};
    spec.work_units = std::vector<WorkUnitSpec>{MakeMinimalWorkUnit("work_unit_0", both, {"dm_kernel"})};
    Program program = MakeProgramFromSpec(spec);

    // node_b is treated as having no varargs — run-params needs no entry for it.
    ProgramRunParams params;
    params.kernel_run_params.push_back({
        .kernel_spec_name = "dm_kernel",
        .runtime_varargs = {{node_a, {1, 2, 3}}},
    });
    EXPECT_NO_THROW(SetProgramRunParameters(program, params));
}

TEST_F(ProgramRunParamsTestQuasar, VarargOnlyAcrossMultipleKernelsSucceeds) {
    // Two kernels, each with only vararg RTAs / CRTAs — the shape of a whole-program
    // migration where nothing has been upgraded to named args yet.
    NodeCoord node{0, 0};
    ProgramSpec spec = MakeMinimalValidProgramSpec();
    spec.kernels[0].runtime_arguments_schema.num_runtime_varargs = 3;
    spec.kernels[0].runtime_arguments_schema.num_common_runtime_varargs = 1;
    spec.kernels[1].runtime_arguments_schema.num_runtime_varargs = 2;
    spec.kernels[1].runtime_arguments_schema.num_common_runtime_varargs = 2;
    Program program = MakeProgramFromSpec(spec);

    ProgramRunParams params;
    params.kernel_run_params.push_back(MakeKernelRunParams("dm_kernel", node, {1, 2, 3}, {99}));
    params.kernel_run_params.push_back(MakeKernelRunParams("compute_kernel", node, {7, 8}, {42, 43}));
    EXPECT_NO_THROW(SetProgramRunParameters(program, params));
}

TEST_F(ProgramRunParamsTestQuasar, VarargOnlyRTAsMissingNodeCoverageFails) {
    // If the schema declares varargs for a node, SetProgramRunParameters must insist on
    // values for that node. Regression canary for per-node coverage in the vararg path.
    NodeCoord node_a{0, 0};
    NodeCoord node_b{1, 0};
    NodeRangeSet nodes{std::vector<NodeRange>{NodeRange{node_a, node_a}, NodeRange{node_b, node_b}}};

    ProgramSpec spec;
    spec.program_id = "vararg_missing_node";
    auto kernel = MakeMinimalDMKernel("dm_kernel");
    kernel.runtime_arguments_schema.num_runtime_varargs = 2;  // uniform across both nodes
    spec.kernels = {kernel};
    spec.work_units = std::vector<WorkUnitSpec>{MakeMinimalWorkUnit("work_unit_0", nodes, {"dm_kernel"})};
    Program program = MakeProgramFromSpec(spec);

    ProgramRunParams params;
    params.kernel_run_params.push_back({
        .kernel_spec_name = "dm_kernel", .runtime_varargs = {{node_a, {10, 20}}},  // node_b missing!
    });
    EXPECT_THAT(
        [&] { SetProgramRunParameters(program, params); },
        ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr("missing vararg runtime args for node")));
}

TEST_F(ProgramRunParamsTestQuasar, VarargOnlyUnknownNodeFails) {
    // Host passes runtime_varargs for a node the kernel doesn't run on. Regression canary for
    // the domain check added alongside named-RTA validation.
    NodeCoord node{0, 0};
    NodeCoord wrong_node{3, 3};
    ProgramSpec spec = MakeMinimalValidProgramSpec();
    spec.kernels[0].runtime_arguments_schema.num_runtime_varargs = 1;
    Program program = MakeProgramFromSpec(spec);

    ProgramRunParams params;
    params.kernel_run_params.push_back({
        .kernel_spec_name = "dm_kernel",
        .runtime_varargs = {{wrong_node, {42}}},
    });
    params.kernel_run_params.push_back(MakeKernelRunParams("compute_kernel", node, {}, {}));
    EXPECT_THAT(
        [&] { SetProgramRunParameters(program, params); },
        ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr("the kernel does not run on that node")));
}

// ============================================================================
// Edge case: empty schema
// ============================================================================

// A kernel can legitimately declare no RTAs / CRTAs / CTAs of any kind (named or vararg).
// Verify the whole MakeProgramFromSpec + SetProgramRunParameters pipeline handles this case
// cleanly — no missing-schema TT_FATALs, no empty-buffer write attempts, no validation errors.
TEST_F(ProgramRunParamsTestQuasar, AllEmptySchemaSucceeds) {
    ProgramSpec spec = MakeMinimalValidProgramSpec();
    // spec.kernels already default to empty named_runtime_args / named_common_runtime_args /
    // compile_time_arg_bindings, num_runtime_varargs = 0, num_common_runtime_varargs = 0,
    // num_runtime_varargs_per_node = nullopt.
    Program program = MakeProgramFromSpec(spec);

    ProgramRunParams params;
    params.kernel_run_params.push_back({.kernel_spec_name = "dm_kernel"});
    params.kernel_run_params.push_back({.kernel_spec_name = "compute_kernel"});

    EXPECT_NO_THROW(SetProgramRunParameters(program, params));
}

TEST_F(ProgramRunParamsTestQuasar, NamedAndVarargRTAsCoexistSucceeds) {
    // A kernel with both named RTAs (schema) and varargs (num_runtime_varargs).
    NodeCoord node{0, 0};
    ProgramSpec spec = MakeMinimalValidProgramSpec();
    spec.kernels[0].runtime_arguments_schema.named_runtime_args = {"input_ptr"};
    spec.kernels[0].runtime_arguments_schema.num_runtime_varargs = 3;
    Program program = MakeProgramFromSpec(spec);

    ProgramRunParams params;
    params.kernel_run_params.push_back({
        .kernel_spec_name = "dm_kernel",
        .named_runtime_args = {{.node = node, .args = {{"input_ptr", 0x1000}}}},
        .runtime_varargs = {{node, {7, 8, 9}}},
    });
    params.kernel_run_params.push_back(MakeKernelRunParams("compute_kernel", node, {}, {}));

    EXPECT_NO_THROW(SetProgramRunParameters(program, params));
}

// ============================================================================
// SECTION 5: Gen1 (WH/BH) Tests
// ============================================================================

class ProgramRunParamsTestGen1 : public ::testing::Test {
protected:
    void SetUp() override { experimental::configure_mock_mode(tt::ARCH::WORMHOLE_B0, 1); }
    void TearDown() override { experimental::disable_mock_mode(); }
};

// Create a gen1 ProgramSpec with a specified RTA schema on the DM kernel
inline ProgramSpec MakeGen1SpecWithRTAs(const NodeCoord& /*node*/, size_t num_per_node_rtas, size_t num_common_rtas) {
    ProgramSpec spec = MakeMinimalGen1ValidProgramSpec();

    spec.kernels[0].runtime_arguments_schema.num_runtime_varargs = num_per_node_rtas;
    spec.kernels[0].runtime_arguments_schema.num_common_runtime_varargs = num_common_rtas;

    // kernels[1] has no varargs (defaults)

    return spec;
}

TEST_F(ProgramRunParamsTestGen1, SetRunParamsSucceeds_ZeroRTAs) {
    NodeCoord node{0, 0};
    ProgramSpec spec = MakeGen1SpecWithRTAs(node, 0, 0);
    Program program = MakeProgramFromSpec(spec);

    auto params = MakeRunParamsForMinimalSpec(node, {}, {});

    EXPECT_NO_THROW(SetProgramRunParameters(program, params));
}

TEST_F(ProgramRunParamsTestGen1, SetRunParamsSucceeds_PerNodeAndCommonRTAs) {
    NodeCoord node{0, 0};
    ProgramSpec spec = MakeGen1SpecWithRTAs(node, /*num_per_node_rtas=*/3, /*num_common_rtas=*/2);
    Program program = MakeProgramFromSpec(spec);

    auto params = MakeRunParamsForMinimalSpec(node, {100, 200, 300}, {10, 20});

    EXPECT_NO_THROW(SetProgramRunParameters(program, params));
}

TEST_F(ProgramRunParamsTestGen1, WrongRuntimeArgsCountFails) {
    NodeCoord node{0, 0};
    ProgramSpec spec = MakeGen1SpecWithRTAs(node, /*num_per_node_rtas=*/3, /*num_common_rtas=*/0);
    Program program = MakeProgramFromSpec(spec);

    // Provide wrong count (2 instead of 3)
    auto params = MakeRunParamsForMinimalSpec(node, {1, 2}, {});

    EXPECT_THAT(
        [&] { SetProgramRunParameters(program, params); },
        ::testing::ThrowsMessage<std::runtime_error>(
            ::testing::HasSubstr("expects 3 vararg runtime args, but 2 were provided")));
}

}  // namespace
}  // namespace tt::tt_metal::experimental::metal2_host_api
