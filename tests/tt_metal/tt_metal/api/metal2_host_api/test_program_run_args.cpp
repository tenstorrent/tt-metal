// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

//---------------------------------------------------------------------------------
// Unit tests for the Metal 2.0 Host API: ProgramRunArgs and SetProgramRunArgs
// These tests all use mock device (Quasar and Wormhole) for API-level validation.
//
// Test categories:
//   1. Validation tests (parameter validation errors)
//   2. Success tests (basic functionality)
//   3. Repeated call tests (calling SetProgramRunArgs multiple times)
//
//---------------------------------------------------------------------------------
// NOTE: The current tests only attempt to set ProgramRunArgs for a Program that has not yet been enqueued for
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
#include <optional>

#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include <tt-metalium/experimental/tensor/mesh_tensor.hpp>
#include <tt-metalium/experimental/tensor/topology/tensor_topology.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/experimental/context/metal_env.hpp>
#include <tt-metalium/experimental/mock_device.hpp>

#include "impl/kernels/kernel.hpp"
#include "impl/program/program_impl.hpp"
#include "test_helpers.hpp"

namespace tt::tt_metal::experimental::metal2_host_api {
namespace {

// Import shared test helpers
using test_helpers::BindDFBToKernel;
using test_helpers::BindTensorParameterToKernel;
using test_helpers::MakeMinimalDFB;
using test_helpers::MakeMinimalDMKernel;
using test_helpers::MakeMinimalGen1ValidProgramSpec;
using test_helpers::MakeMinimalTensorParameter;
using test_helpers::MakeMinimalValidProgramSpec;
using test_helpers::MakeMinimalWorkUnit;
using test_helpers::MakeShardedTensorParameter;
using test_helpers::ScopedSlowDispatchOverride;

// Shorthand for the per-node-override vararg type (needed at call sites because
// std::optional<T> can't be brace-init from an initializer-list of T's elements).
using NumVarargsPerNode = KernelAdvancedOptions::NumVarargsPerNode;

// ============================================================================
// Test Fixtures
// ============================================================================

// Test fixture for ProgramRunArgs on Quasar - uses Quasar mock device.
//
// Forces TT_METAL_SLOW_DISPATCH_MODE so that MeshDevice::create() succeeds against the
// mock cluster (mock Quasar has no dispatch-core reservation in its descriptor). These
// tests exercise pure API behavior, so slow dispatch is functionally fine.
class ProgramRunArgsTestQuasar : public ::testing::Test {
protected:
    void SetUp() override {
        slow_dispatch_override_.emplace();
        //  Configure global mock mode for Quasar
        experimental::configure_mock_mode(tt::ARCH::QUASAR, 1);
        mesh_device_ = distributed::MeshDevice::create(distributed::MeshDeviceConfig(distributed::MeshShape{1, 1}));
    }
    void TearDown() override {
        if (mesh_device_) {
            mesh_device_->close();
            mesh_device_.reset();
        }
        experimental::disable_mock_mode();
        slow_dispatch_override_.reset();
    }

    std::shared_ptr<distributed::MeshDevice> mesh_device_;
    std::optional<ScopedSlowDispatchOverride> slow_dispatch_override_;
};

// ============================================================================
// Test Utilities
// ============================================================================

// Create a ProgramSpec with specified RTA schema for the DM kernel
// (The compute kernel has no RTAs)
inline ProgramSpec MakeSpecWithRTAs(const NodeCoord& /*node*/, size_t num_per_node_rtas, size_t num_common_rtas) {
    ProgramSpec spec = MakeMinimalValidProgramSpec();

    // Set the RTA schema on the dm_kernel (first kernel)
    spec.kernels[0].advanced_options =
        KernelAdvancedOptions{.num_runtime_varargs = num_per_node_rtas, .num_common_runtime_varargs = num_common_rtas};

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
    spec.kernels[0].advanced_options =
        KernelAdvancedOptions{.num_runtime_varargs = dm_per_node_rtas, .num_common_runtime_varargs = dm_common_rtas};

    // compute_kernel RTAs
    spec.kernels[1].advanced_options = KernelAdvancedOptions{
        .num_runtime_varargs = compute_per_node_rtas, .num_common_runtime_varargs = compute_common_rtas};

    return spec;
}

// Helper to create ProgramRunArgs for a single kernel
inline ProgramRunArgs::KernelRunArgs MakeKernelRunArgs(
    const KernelSpecName& kernel_name,
    const NodeCoord& node,
    const std::vector<uint32_t>& per_node_args,
    const std::vector<uint32_t>& common_args) {
    return ProgramRunArgs::KernelRunArgs{
        .kernel_spec_name = kernel_name,
        .advanced_options =
            AdvancedKernelRunArgs{
                .runtime_varargs = {{node, per_node_args}},
                .common_runtime_varargs = common_args,
            },
    };
}

// Helper to create complete ProgramRunArgs for the minimal spec (both kernels)
inline ProgramRunArgs MakeRunParamsForMinimalSpec(
    const NodeCoord& node,
    const std::vector<uint32_t>& dm_per_node_args,
    const std::vector<uint32_t>& dm_common_args,
    const std::vector<uint32_t>& compute_per_node_args = {},
    const std::vector<uint32_t>& compute_common_args = {}) {
    ProgramRunArgs params;
    params.kernel_run_args.push_back(MakeKernelRunArgs("dm_kernel", node, dm_per_node_args, dm_common_args));
    params.kernel_run_args.push_back(
        MakeKernelRunArgs("compute_kernel", node, compute_per_node_args, compute_common_args));
    return params;
}

// ============================================================================
// SECTION 1: Validation Tests (expect failure)
// ============================================================================

TEST_F(ProgramRunArgsTestQuasar, UnknownKernelNameFails) {
    NodeCoord node{0, 0};
    ProgramSpec spec = MakeSpecWithRTAs(node, 0, 0);
    Program program = MakeProgramFromSpec(*mesh_device_, spec);

    ProgramRunArgs params;
    params.kernel_run_args.push_back({
        .kernel_spec_name = "nonexistent_kernel",
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

TEST_F(ProgramRunArgsTestQuasar, InvalidNodeForKernelFails) {
    NodeCoord node{0, 0};
    NodeCoord wrong_node{1, 1};  // Kernel doesn't run on this node
    ProgramSpec spec = MakeSpecWithRTAs(node, 2, 0);
    Program program = MakeProgramFromSpec(*mesh_device_, spec);

    ProgramRunArgs params;
    params.kernel_run_args.push_back({
        .kernel_spec_name = "dm_kernel",
        .advanced_options =
            AdvancedKernelRunArgs{
                .runtime_varargs = {{wrong_node, {1, 2}}},  // Wrong node!
                .common_runtime_varargs = {},
            },
    });
    params.kernel_run_args.push_back(MakeKernelRunArgs("compute_kernel", node, {}, {}));

    EXPECT_THAT(
        [&] { SetProgramRunArgs(program, params); },
        ::testing::ThrowsMessage<std::runtime_error>(
            ::testing::HasSubstr("Kernel 'dm_kernel' is setting runtime_varargs for node")));
}

TEST_F(ProgramRunArgsTestQuasar, WrongRuntimeArgsCountFails) {
    NodeCoord node{0, 0};
    ProgramSpec spec = MakeSpecWithRTAs(node, /*num_per_node_rtas=*/3, /*num_common_rtas=*/0);
    Program program = MakeProgramFromSpec(*mesh_device_, spec);

    // Provide wrong count (2 instead of 3)
    auto params = MakeRunParamsForMinimalSpec(node, {1, 2}, {});

    EXPECT_THAT(
        [&] { SetProgramRunArgs(program, params); },
        ::testing::ThrowsMessage<std::runtime_error>(
            ::testing::HasSubstr("expects 3 vararg runtime args, but 2 were provided")));
}

TEST_F(ProgramRunArgsTestQuasar, WrongCommonRuntimeArgsCountFails) {
    NodeCoord node{0, 0};
    ProgramSpec spec = MakeSpecWithRTAs(node, /*num_per_node_rtas=*/0, /*num_common_rtas=*/2);
    Program program = MakeProgramFromSpec(*mesh_device_, spec);

    // Provide wrong common args count (3 instead of 2)
    auto params = MakeRunParamsForMinimalSpec(node, {}, {1, 2, 3});

    EXPECT_THAT(
        [&] { SetProgramRunArgs(program, params); },
        ::testing::ThrowsMessage<std::runtime_error>(
            ::testing::HasSubstr("expects 2 vararg common runtime args, but 3 were provided")));
}

// TODO: Currently, we require that all kernels in a ProgramSpec have params specified.
// Should relax this to omit kernels with no RTAs or CRTAs.
TEST_F(ProgramRunArgsTestQuasar, EmptySchemaKernelOmittedFromRunParamsSucceeds) {
    NodeCoord node{0, 0};
    // Both kernels have empty RTA/CRTA schemas — neither has anything to supply per enqueue.
    ProgramSpec spec = MakeSpecWithRTAs(node, 0, 0);
    Program program = MakeProgramFromSpec(*mesh_device_, spec);

    // Only provide params for dm_kernel; omit compute_kernel.
    // Since compute_kernel's schema is empty, this should succeed.
    ProgramRunArgs params;
    params.kernel_run_args.push_back(MakeKernelRunArgs("dm_kernel", node, {}, {}));

    EXPECT_NO_THROW({ SetProgramRunArgs(program, params); });
}

TEST_F(ProgramRunArgsTestQuasar, NonEmptySchemaKernelMissingFromRunParamsFails) {
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
    params.kernel_run_args.push_back(MakeKernelRunArgs("dm_kernel", node, {}, {}));

    EXPECT_THAT(
        [&] { SetProgramRunArgs(program, params); },
        ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr(
            "Kernel 'compute_kernel' is registered in the Program with a non-empty RTA/CRTA schema "
            "but has no runtime parameters specified in ProgramRunArgs")));
}

TEST_F(ProgramRunArgsTestQuasar, DuplicateKernelParamsFails) {
    NodeCoord node{0, 0};
    ProgramSpec spec = MakeSpecWithRTAs(node, 0, 0);
    Program program = MakeProgramFromSpec(*mesh_device_, spec);

    // Provide params for dm_kernel twice
    ProgramRunArgs params;
    params.kernel_run_args.push_back(MakeKernelRunArgs("dm_kernel", node, {}, {}));
    params.kernel_run_args.push_back(MakeKernelRunArgs("dm_kernel", node, {}, {}));  // Duplicate!
    params.kernel_run_args.push_back(MakeKernelRunArgs("compute_kernel", node, {}, {}));

    EXPECT_THAT(
        [&] { SetProgramRunArgs(program, params); },
        ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr("Duplicate kernel_spec_name 'dm_kernel'")));
}

TEST_F(ProgramRunArgsTestQuasar, MissingNodeRTAsFails) {
    NodeCoord node{0, 0};
    ProgramSpec spec = MakeSpecWithRTAs(node, /*num_per_node_rtas=*/2, /*num_common_rtas=*/0);
    Program program = MakeProgramFromSpec(*mesh_device_, spec);

    // Don't provide the per-node RTAs (empty runtime_varargs)
    ProgramRunArgs params;
    params.kernel_run_args.push_back({
        .kernel_spec_name = "dm_kernel",
        .advanced_options =
            AdvancedKernelRunArgs{
                .runtime_varargs = {},  // Missing node RTAs!
                .common_runtime_varargs = {},
            },
    });
    params.kernel_run_args.push_back(MakeKernelRunArgs("compute_kernel", node, {}, {}));

    EXPECT_THAT(
        [&] { SetProgramRunArgs(program, params); },
        ::testing::ThrowsMessage<std::runtime_error>(
            ::testing::HasSubstr("Kernel 'dm_kernel' is missing vararg runtime args for node")));
}

// TODO: Replace when feature is implemented
TEST_F(ProgramRunArgsTestQuasar, DFBSizeOverrideFails) {
    NodeCoord node{0, 0};
    ProgramSpec spec = MakeSpecWithRTAs(node, 0, 0);
    Program program = MakeProgramFromSpec(*mesh_device_, spec);

    auto params = MakeRunParamsForMinimalSpec(node, {}, {});

    // Add DFB run params with size override (not implemented)
    params.dfb_run_overrides.push_back({
        .dfb_spec_name = "dfb_0",
        .entry_size = 2048,  // Override - not implemented!
    });

    EXPECT_THAT(
        [&] { SetProgramRunArgs(program, params); },
        ::testing::ThrowsMessage<std::runtime_error>(
            ::testing::HasSubstr("DFB size overrides are not yet implemented")));
}

// TODO: Replace when feature is implemented
TEST_F(ProgramRunArgsTestQuasar, DFBNumEntriesOverrideFails) {
    NodeCoord node{0, 0};
    ProgramSpec spec = MakeSpecWithRTAs(node, 0, 0);
    Program program = MakeProgramFromSpec(*mesh_device_, spec);

    auto params = MakeRunParamsForMinimalSpec(node, {}, {});

    // Add DFB run params with num_entries override (not implemented)
    params.dfb_run_overrides.push_back({
        .dfb_spec_name = "dfb_0",
        .num_entries = 4,  // Override - not implemented!
    });

    EXPECT_THAT(
        [&] { SetProgramRunArgs(program, params); },
        ::testing::ThrowsMessage<std::runtime_error>(
            ::testing::HasSubstr("DFB size overrides are not yet implemented")));
}

TEST_F(ProgramRunArgsTestQuasar, DuplicateDFBParamsFails) {
    NodeCoord node{0, 0};
    ProgramSpec spec = MakeSpecWithRTAs(node, 0, 0);
    Program program = MakeProgramFromSpec(*mesh_device_, spec);

    auto params = MakeRunParamsForMinimalSpec(node, {}, {});

    // Add the same DFB twice
    params.dfb_run_overrides.push_back({.dfb_spec_name = "dfb_0"});
    params.dfb_run_overrides.push_back({.dfb_spec_name = "dfb_0"});  // Duplicate!

    EXPECT_THAT(
        [&] { SetProgramRunArgs(program, params); },
        ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr("Duplicate dfb_spec_name 'dfb_0'")));
}

TEST_F(ProgramRunArgsTestQuasar, DuplicateNodeCoordInRuntimeArgsFails) {
    NodeCoord node{0, 0};
    ProgramSpec spec = MakeSpecWithRTAs(node, /*num_per_node_rtas=*/2, /*num_common_rtas=*/0);
    Program program = MakeProgramFromSpec(*mesh_device_, spec);

    // Provide runtime_varargs with duplicate node_coord entries
    ProgramRunArgs params;
    params.kernel_run_args.push_back({
        .kernel_spec_name = "dm_kernel",
        .advanced_options =
            AdvancedKernelRunArgs{
                .runtime_varargs = {{node, {1, 2}}, {node, {3, 4}}},  // Duplicate node!
                .common_runtime_varargs = {},
            },
    });
    params.kernel_run_args.push_back(MakeKernelRunArgs("compute_kernel", node, {}, {}));

    EXPECT_THAT(
        [&] { SetProgramRunArgs(program, params); },
        ::testing::ThrowsMessage<std::runtime_error>(
            ::testing::HasSubstr("in runtime_varargs for kernel 'dm_kernel'")));
}

// ============================================================================
// SECTION 2: Success Tests (basic functionality)
// ============================================================================

TEST_F(ProgramRunArgsTestQuasar, SetRunParamsSucceeds_ZeroRTAs) {
    NodeCoord node{0, 0};
    ProgramSpec spec = MakeSpecWithRTAs(node, /*num_per_node_rtas=*/0, /*num_common_rtas=*/0);
    Program program = MakeProgramFromSpec(*mesh_device_, spec);

    auto params = MakeRunParamsForMinimalSpec(node, {}, {});

    EXPECT_NO_THROW(SetProgramRunArgs(program, params));
}

TEST_F(ProgramRunArgsTestQuasar, SetRunParamsSucceeds_PerNodeRTAsOnly) {
    NodeCoord node{0, 0};
    ProgramSpec spec = MakeSpecWithRTAs(node, /*num_per_node_rtas=*/3, /*num_common_rtas=*/0);
    Program program = MakeProgramFromSpec(*mesh_device_, spec);

    auto params = MakeRunParamsForMinimalSpec(node, {100, 200, 300}, {});

    EXPECT_NO_THROW(SetProgramRunArgs(program, params));
}

TEST_F(ProgramRunArgsTestQuasar, SetRunParamsSucceeds_CommonRTAsOnly) {
    NodeCoord node{0, 0};
    ProgramSpec spec = MakeSpecWithRTAs(node, /*num_per_node_rtas=*/0, /*num_common_rtas=*/2);
    Program program = MakeProgramFromSpec(*mesh_device_, spec);

    auto params = MakeRunParamsForMinimalSpec(node, {}, {10, 20});

    EXPECT_NO_THROW(SetProgramRunArgs(program, params));
}

TEST_F(ProgramRunArgsTestQuasar, SetRunParamsSucceeds_BothRTATypes) {
    NodeCoord node{0, 0};
    ProgramSpec spec = MakeSpecWithRTAs(node, /*num_per_node_rtas=*/3, /*num_common_rtas=*/2);
    Program program = MakeProgramFromSpec(*mesh_device_, spec);

    auto params = MakeRunParamsForMinimalSpec(node, {100, 200, 300}, {10, 20});

    EXPECT_NO_THROW(SetProgramRunArgs(program, params));
}

TEST_F(ProgramRunArgsTestQuasar, SetRunParamsSucceeds_BothKernelsWithRTAs) {
    NodeCoord node{0, 0};
    ProgramSpec spec = MakeSpecWithBothKernelRTAs(
        node,
        /*dm_per_node_rtas=*/2,
        /*dm_common_rtas=*/1,
        /*compute_per_node_rtas=*/3,
        /*compute_common_rtas=*/2);
    Program program = MakeProgramFromSpec(*mesh_device_, spec);

    auto params = MakeRunParamsForMinimalSpec(
        node,
        /*dm_per_node_args=*/{1, 2},
        /*dm_common_args=*/{10},
        /*compute_per_node_args=*/{100, 200, 300},
        /*compute_common_args=*/{50, 60});

    EXPECT_NO_THROW(SetProgramRunArgs(program, params));
}

TEST_F(ProgramRunArgsTestQuasar, SetRunParamsSucceeds_DFBRunParamsWithNoOverrides) {
    NodeCoord node{0, 0};
    ProgramSpec spec = MakeSpecWithRTAs(node, 0, 0);
    Program program = MakeProgramFromSpec(*mesh_device_, spec);

    auto params = MakeRunParamsForMinimalSpec(node, {}, {});

    // DFB run params with no overrides is allowed
    params.dfb_run_overrides.push_back({
        .dfb_spec_name = "dfb_0",
        // No overrides - both entry_size and num_entries are nullopt
    });

    EXPECT_NO_THROW(SetProgramRunArgs(program, params));
}

// ============================================================================
// SECTION 3: Repeated Call Tests
// ============================================================================

TEST_F(ProgramRunArgsTestQuasar, SetRunParamsTwice_SameValuesSucceeds) {
    NodeCoord node{0, 0};
    ProgramSpec spec = MakeSpecWithRTAs(node, /*num_per_node_rtas=*/2, /*num_common_rtas=*/1);
    Program program = MakeProgramFromSpec(*mesh_device_, spec);

    auto params = MakeRunParamsForMinimalSpec(node, {100, 200}, {10});

    // First call
    EXPECT_NO_THROW(SetProgramRunArgs(program, params));
    // Second call with same values
    EXPECT_NO_THROW(SetProgramRunArgs(program, params));
}

TEST_F(ProgramRunArgsTestQuasar, SetRunParamsTwice_DifferentValuesSucceeds) {
    NodeCoord node{0, 0};
    ProgramSpec spec = MakeSpecWithRTAs(node, /*num_per_node_rtas=*/2, /*num_common_rtas=*/1);
    Program program = MakeProgramFromSpec(*mesh_device_, spec);

    auto params1 = MakeRunParamsForMinimalSpec(node, {100, 200}, {10});
    auto params2 = MakeRunParamsForMinimalSpec(node, {300, 400}, {20});  // Different values

    // First call
    EXPECT_NO_THROW(SetProgramRunArgs(program, params1));
    // Second call with different values (same counts)
    EXPECT_NO_THROW(SetProgramRunArgs(program, params2));
}

TEST_F(ProgramRunArgsTestQuasar, SetRunParamsTwice_ChangingCommonRTACountFails) {
    NodeCoord node{0, 0};
    ProgramSpec spec = MakeSpecWithRTAs(node, /*num_per_node_rtas=*/0, /*num_common_rtas=*/2);
    Program program = MakeProgramFromSpec(*mesh_device_, spec);

    // First call with 2 common RTAs (matches schema)
    auto params1 = MakeRunParamsForMinimalSpec(node, {}, {10, 20});
    EXPECT_NO_THROW(SetProgramRunArgs(program, params1));

    // Change the schema expectation - this is tricky because the schema is fixed
    // The implementation checks against the schema, not the previous call.
    // So changing count will fail on validation, not on the memcpy path.
    // Actually, looking at the code, the schema validation happens first,
    // so if we try to pass wrong count, it fails at validation.

    // Let's verify that passing wrong count still fails (schema validation)
    auto params2 = MakeRunParamsForMinimalSpec(node, {}, {10, 20, 30});  // 3 instead of 2
    EXPECT_THAT(
        [&] { SetProgramRunArgs(program, params2); },
        ::testing::ThrowsMessage<std::runtime_error>(
            ::testing::HasSubstr("expects 2 vararg common runtime args, but 3 were provided")));
}

TEST_F(ProgramRunArgsTestQuasar, SetRunParamsMultipleTimes_Succeeds) {
    NodeCoord node{0, 0};
    ProgramSpec spec = MakeSpecWithRTAs(node, /*num_per_node_rtas=*/2, /*num_common_rtas=*/2);
    Program program = MakeProgramFromSpec(*mesh_device_, spec);

    // Call SetProgramRunArgs multiple times with varying values
    for (uint32_t i = 0; i < 5; i++) {
        auto params = MakeRunParamsForMinimalSpec(node, {i * 10, i * 20}, {i * 100, i * 200});
        EXPECT_NO_THROW(SetProgramRunArgs(program, params));
    }
}

// ============================================================================
// SECTION 4: Multi-Node Tests
// ============================================================================

TEST_F(ProgramRunArgsTestQuasar, SetRunParamsSucceeds_MultiNodeKernel) {
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
    producer.advanced_options = KernelAdvancedOptions{.num_runtime_varargs = 2, .num_common_runtime_varargs = 1};

    // consumer has no varargs (defaults)

    // Single DFB spanning all nodes
    auto dfb = MakeMinimalDFB("dfb");

    BindDFBToKernel(producer, "dfb", "out", KernelSpec::DFBEndpointType::PRODUCER);
    BindDFBToKernel(consumer, "dfb", "in", KernelSpec::DFBEndpointType::CONSUMER);

    spec.kernels = {producer, consumer};
    spec.dataflow_buffers = {dfb};
    spec.work_units = std::vector<WorkUnitSpec>{MakeMinimalWorkUnit("work_unit", all_nodes, {"producer", "consumer"})};

    Program program = MakeProgramFromSpec(*mesh_device_, spec);

    // Create run params with per-node RTAs for both nodes
    ProgramRunArgs params;
    params.kernel_run_args.push_back({
        .kernel_spec_name = "producer",
        .advanced_options =
            AdvancedKernelRunArgs{
                .runtime_varargs = {{node0, {10, 20}}, {node1, {30, 40}}},
                .common_runtime_varargs = {100},
            },
    });
    params.kernel_run_args.push_back({
        .kernel_spec_name = "consumer",
        .advanced_options =
            AdvancedKernelRunArgs{
                .runtime_varargs = {{node0, {}}, {node1, {}}},
                .common_runtime_varargs = {},
            },
    });

    EXPECT_NO_THROW(SetProgramRunArgs(program, params));
}

TEST_F(ProgramRunArgsTestQuasar, MultiNode_MissingOneNodeFails) {
    // Create a program with kernels spanning multiple nodes
    NodeCoord node0{0, 0};
    NodeCoord node1{1, 0};
    NodeRangeSet all_nodes(std::set<NodeRange>{NodeRange{node0, node0}, NodeRange{node1, node1}});

    ProgramSpec spec;
    spec.program_id = "multi_node_program";

    auto producer = MakeMinimalDMKernel("producer");
    auto consumer = MakeMinimalDMKernel("consumer");

    // Throw in some varargs (the normal kind, not the weird per-node override kind)
    producer.advanced_options.num_runtime_varargs = 2;
    // consumer has no varargs (defaults)

    // Single DFB spanning all nodes
    auto dfb = MakeMinimalDFB("dfb");

    BindDFBToKernel(producer, "dfb", "out", KernelSpec::DFBEndpointType::PRODUCER);
    BindDFBToKernel(consumer, "dfb", "in", KernelSpec::DFBEndpointType::CONSUMER);

    spec.kernels = {producer, consumer};
    spec.dataflow_buffers = {dfb};
    spec.work_units = std::vector<WorkUnitSpec>{MakeMinimalWorkUnit("work_unit", all_nodes, {"producer", "consumer"})};

    Program program = MakeProgramFromSpec(*mesh_device_, spec);

    // Only provide RTAs for node0, missing node1
    ProgramRunArgs params;
    params.kernel_run_args.push_back({
        .kernel_spec_name = "producer",
        .advanced_options =
            AdvancedKernelRunArgs{
                .runtime_varargs = {{node0, {10, 20}}},  // Missing node1!
                .common_runtime_varargs = {},
            },
    });
    params.kernel_run_args.push_back({
        .kernel_spec_name = "consumer",
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

// ============================================================================
// SECTION 5: Gen1 (WH/BH) Tests
// ============================================================================
// The RTA validation logic in SetProgramRunArgs is arch-agnostic; these
// tests verify the full roundtrip works on gen1 programs and that validation
// still fires when it should.

// Test fixture for ProgramRunArgs on Wormhole - uses WORMHOLE_B0 mock device
// ============================================================================
// SECTION 4: Named RTA / CRTA Tests (Quasar)
// ============================================================================

// Make a ProgramSpec where the DM kernel has a named-RTA / named-CRTA schema.
inline ProgramSpec MakeSpecWithNamedArgs(
    const NodeCoord& node, const std::vector<std::string>& named_rtas, const std::vector<std::string>& named_crtas) {
    ProgramSpec spec = MakeMinimalValidProgramSpec();
    spec.kernels[0].runtime_arg_schema.runtime_arg_names = named_rtas;
    spec.kernels[0].runtime_arg_schema.common_runtime_arg_names = named_crtas;
    (void)node;  // node inherited from MakeMinimalValidProgramSpec (0,0)
    return spec;
}

TEST_F(ProgramRunArgsTestQuasar, NamedRTAsAndCRTAsSucceed) {
    NodeCoord node{0, 0};
    ProgramSpec spec = MakeSpecWithNamedArgs(node, {"input_ptr", "output_ptr"}, {"tile_count"});
    Program program = MakeProgramFromSpec(*mesh_device_, spec);

    ProgramRunArgs params;
    params.kernel_run_args.push_back({
        .kernel_spec_name = "dm_kernel",
        .runtime_arg_values = {{.node = node, .args = {{"input_ptr", 0x1000}, {"output_ptr", 0x2000}}}},
        .common_runtime_arg_values = {{"tile_count", 64}},
    });
    params.kernel_run_args.push_back(MakeKernelRunArgs("compute_kernel", node, {}, {}));

    EXPECT_NO_THROW(SetProgramRunArgs(program, params));
}

TEST_F(ProgramRunArgsTestQuasar, MissingNamedRTAForNodeFails) {
    NodeCoord node{0, 0};
    ProgramSpec spec = MakeSpecWithNamedArgs(node, {"input_ptr"}, {});
    Program program = MakeProgramFromSpec(*mesh_device_, spec);

    ProgramRunArgs params;
    params.kernel_run_args.push_back({
        .kernel_spec_name = "dm_kernel",
        // No runtime_arg_values for node (0,0) at all — but schema declares one.
    });
    params.kernel_run_args.push_back(MakeKernelRunArgs("compute_kernel", node, {}, {}));

    EXPECT_THAT(
        [&] { SetProgramRunArgs(program, params); },
        ::testing::ThrowsMessage<std::runtime_error>(
            ::testing::HasSubstr("has named RTAs declared but no runtime_arg_values provided for node")));
}

TEST_F(ProgramRunArgsTestQuasar, MissingDeclaredNamedRTANameFails) {
    NodeCoord node{0, 0};
    ProgramSpec spec = MakeSpecWithNamedArgs(node, {"input_ptr", "output_ptr"}, {});
    Program program = MakeProgramFromSpec(*mesh_device_, spec);

    ProgramRunArgs params;
    params.kernel_run_args.push_back({
        .kernel_spec_name = "dm_kernel",
        // Only one name provided — output_ptr missing.
        .runtime_arg_values = {{.node = node, .args = {{"input_ptr", 0x1000}}}},
    });
    params.kernel_run_args.push_back(MakeKernelRunArgs("compute_kernel", node, {}, {}));

    EXPECT_THAT(
        [&] { SetProgramRunArgs(program, params); },
        ::testing::ThrowsMessage<std::runtime_error>(
            ::testing::HasSubstr("expects 2 named RTAs, but 1 were provided")));
}

TEST_F(ProgramRunArgsTestQuasar, UndeclaredNamedRTAFails) {
    NodeCoord node{0, 0};
    ProgramSpec spec = MakeSpecWithNamedArgs(node, {"input_ptr"}, {});
    Program program = MakeProgramFromSpec(*mesh_device_, spec);

    ProgramRunArgs params;
    params.kernel_run_args.push_back({
        .kernel_spec_name = "dm_kernel",
        .runtime_arg_values = {{.node = node, .args = {{"input_ptr", 0x1000}, {"not_in_schema", 0}}}},
    });
    params.kernel_run_args.push_back(MakeKernelRunArgs("compute_kernel", node, {}, {}));

    EXPECT_THAT(
        [&] { SetProgramRunArgs(program, params); },
        ::testing::ThrowsMessage<std::runtime_error>(
            ::testing::HasSubstr("expects 1 named RTAs, but 2 were provided")));
}

TEST_F(ProgramRunArgsTestQuasar, NamedCRTACountMismatchFails) {
    NodeCoord node{0, 0};
    ProgramSpec spec = MakeSpecWithNamedArgs(node, {}, {"tile_count", "scale"});
    Program program = MakeProgramFromSpec(*mesh_device_, spec);

    ProgramRunArgs params;
    params.kernel_run_args.push_back({
        .kernel_spec_name = "dm_kernel",
        // Only one CRTA provided; schema declares two.
        .common_runtime_arg_values = {{"tile_count", 4}},
    });
    params.kernel_run_args.push_back(MakeKernelRunArgs("compute_kernel", node, {}, {}));

    // Validation now reports the specific missing name first (more useful than a count mismatch).
    EXPECT_THAT(
        [&] { SetProgramRunArgs(program, params); },
        ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr("is missing named CRTA 'scale'")));
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

TEST_F(ProgramRunArgsTestQuasar, VarargOnlyMultiNodeDifferingCountsSucceeds) {
    // A kernel on two nodes with DIFFERENT vararg counts per node. Exercises the advanced
    // num_runtime_varargs_per_node override path. The RTA dispatch buffer must be sized
    // per-node, which is a common failure mode for layout bugs.
    NodeCoord node_a{0, 0};
    NodeCoord node_b{1, 0};
    NodeRangeSet nodes{std::vector<NodeRange>{NodeRange{node_a, node_a}, NodeRange{node_b, node_b}}};

    ProgramSpec spec;
    spec.program_id = "vararg_differing_counts";
    auto kernel = MakeMinimalDMKernel("dm_kernel");
    kernel.advanced_options.num_runtime_varargs_per_node =
        KernelAdvancedOptions::NumVarargsPerNode{{node_a, 2}, {node_b, 5}};
    spec.kernels = {kernel};
    spec.work_units = std::vector<WorkUnitSpec>{MakeMinimalWorkUnit("work_unit_0", nodes, {"dm_kernel"})};
    Program program = MakeProgramFromSpec(*mesh_device_, spec);

    ProgramRunArgs params;
    params.kernel_run_args.push_back({
        .kernel_spec_name = "dm_kernel",
        .advanced_options =
            AdvancedKernelRunArgs{
                .runtime_varargs = {{node_a, {10, 20}}, {node_b, {100, 200, 300, 400, 500}}},
            },
    });
    EXPECT_NO_THROW(SetProgramRunArgs(program, params));
}

TEST_F(ProgramRunArgsTestQuasar, VarargPerNodeOverrideMixedEntryTypesSucceeds) {
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
    kernel.advanced_options.num_runtime_varargs_per_node =
        KernelAdvancedOptions::NumVarargsPerNode{{ab, 3}, {node_c, 5}};
    spec.kernels = {kernel};
    spec.work_units = std::vector<WorkUnitSpec>{MakeMinimalWorkUnit("work_unit_0", all_nodes, {"dm_kernel"})};
    Program program = MakeProgramFromSpec(*mesh_device_, spec);

    ProgramRunArgs params;
    params.kernel_run_args.push_back({
        .kernel_spec_name = "dm_kernel",
        .advanced_options =
            AdvancedKernelRunArgs{
                .runtime_varargs = {{node_a, {1, 2, 3}}, {node_b, {10, 20, 30}}, {node_c, {100, 200, 300, 400, 500}}},
            },
    });
    EXPECT_NO_THROW(SetProgramRunArgs(program, params));
}

TEST_F(ProgramRunArgsTestQuasar, VarargScalarDefaultWithSparseOverrideSucceeds) {
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
    kernel.advanced_options = KernelAdvancedOptions{
        .num_runtime_varargs = 2,  // default for unlisted nodes
        .num_runtime_varargs_per_node =
            KernelAdvancedOptions::NumVarargsPerNode{{node_c, 5}},  // node_c is the exception
    };
    spec.kernels = {kernel};
    spec.work_units = std::vector<WorkUnitSpec>{MakeMinimalWorkUnit("work_unit_0", all_nodes, {"dm_kernel"})};
    Program program = MakeProgramFromSpec(*mesh_device_, spec);

    ProgramRunArgs params;
    params.kernel_run_args.push_back({
        .kernel_spec_name = "dm_kernel",
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

TEST_F(ProgramRunArgsTestQuasar, VarargSparseOverrideZeroErasesScalarDefault) {
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
    kernel.advanced_options = KernelAdvancedOptions{
        .num_runtime_varargs = 3,
        .num_runtime_varargs_per_node =
            KernelAdvancedOptions::NumVarargsPerNode{{node_b, 0}},  // node_b: no varargs despite scalar default
    };
    spec.kernels = {kernel};
    spec.work_units = std::vector<WorkUnitSpec>{MakeMinimalWorkUnit("work_unit_0", both, {"dm_kernel"})};
    Program program = MakeProgramFromSpec(*mesh_device_, spec);

    // node_b is treated as having no varargs — run-params needs no entry for it.
    ProgramRunArgs params;
    params.kernel_run_args.push_back({
        .kernel_spec_name = "dm_kernel",
        .advanced_options =
            AdvancedKernelRunArgs{
                .runtime_varargs = {{node_a, {1, 2, 3}}},
            },
    });
    EXPECT_NO_THROW(SetProgramRunArgs(program, params));
}

TEST_F(ProgramRunArgsTestQuasar, VarargOnlyAcrossMultipleKernelsSucceeds) {
    // Two kernels, each with only vararg RTAs / CRTAs — the shape of a whole-program
    // migration where nothing has been upgraded to named args yet.
    NodeCoord node{0, 0};
    ProgramSpec spec = MakeMinimalValidProgramSpec();
    spec.kernels[0].advanced_options = KernelAdvancedOptions{.num_runtime_varargs = 3, .num_common_runtime_varargs = 1};
    spec.kernels[1].advanced_options = KernelAdvancedOptions{.num_runtime_varargs = 2, .num_common_runtime_varargs = 2};
    Program program = MakeProgramFromSpec(*mesh_device_, spec);

    ProgramRunArgs params;
    params.kernel_run_args.push_back(MakeKernelRunArgs("dm_kernel", node, {1, 2, 3}, {99}));
    params.kernel_run_args.push_back(MakeKernelRunArgs("compute_kernel", node, {7, 8}, {42, 43}));
    EXPECT_NO_THROW(SetProgramRunArgs(program, params));
}

TEST_F(ProgramRunArgsTestQuasar, VarargOnlyRTAsMissingNodeCoverageFails) {
    // If the schema declares varargs for a node, SetProgramRunArgs must insist on
    // values for that node. Regression canary for per-node coverage in the vararg path.
    NodeCoord node_a{0, 0};
    NodeCoord node_b{1, 0};
    NodeRangeSet nodes{std::vector<NodeRange>{NodeRange{node_a, node_a}, NodeRange{node_b, node_b}}};

    ProgramSpec spec;
    spec.program_id = "vararg_missing_node";
    auto kernel = MakeMinimalDMKernel("dm_kernel");
    kernel.advanced_options.num_runtime_varargs = 2;  // uniform across both nodes
    spec.kernels = {kernel};
    spec.work_units = std::vector<WorkUnitSpec>{MakeMinimalWorkUnit("work_unit_0", nodes, {"dm_kernel"})};
    Program program = MakeProgramFromSpec(*mesh_device_, spec);

    ProgramRunArgs params;
    params.kernel_run_args.push_back({
        .kernel_spec_name = "dm_kernel",
        .advanced_options =
            AdvancedKernelRunArgs{
                .runtime_varargs = {{node_a, {10, 20}}},  // node_b missing!
            },
    });
    EXPECT_THAT(
        [&] { SetProgramRunArgs(program, params); },
        ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr("missing vararg runtime args for node")));
}

TEST_F(ProgramRunArgsTestQuasar, VarargOnlyUnknownNodeFails) {
    // Host passes runtime_varargs for a node the kernel doesn't run on. Regression canary for
    // the domain check added alongside named-RTA validation.
    NodeCoord node{0, 0};
    NodeCoord wrong_node{3, 3};
    ProgramSpec spec = MakeMinimalValidProgramSpec();
    spec.kernels[0].advanced_options.num_runtime_varargs = 1;
    Program program = MakeProgramFromSpec(*mesh_device_, spec);

    ProgramRunArgs params;
    params.kernel_run_args.push_back({
        .kernel_spec_name = "dm_kernel",
        .advanced_options =
            AdvancedKernelRunArgs{
                .runtime_varargs = {{wrong_node, {42}}},
            },
    });
    params.kernel_run_args.push_back(MakeKernelRunArgs("compute_kernel", node, {}, {}));
    EXPECT_THAT(
        [&] { SetProgramRunArgs(program, params); },
        ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr("the kernel does not run on that node")));
}

// ============================================================================
// Edge case: empty schema
// ============================================================================

// A kernel can legitimately declare no RTAs / CRTAs / CTAs of any kind (named or vararg).
// Verify the whole MakeProgramFromSpec + SetProgramRunArgs pipeline handles this case
// cleanly — no missing-schema TT_FATALs, no empty-buffer write attempts, no validation errors.
TEST_F(ProgramRunArgsTestQuasar, AllEmptySchemaSucceeds) {
    ProgramSpec spec = MakeMinimalValidProgramSpec();
    // spec.kernels already default to empty runtime_arg_values / common_runtime_arg_values /
    // compile_time_args, num_runtime_varargs = 0, num_common_runtime_varargs = 0,
    // num_runtime_varargs_per_node = nullopt.
    Program program = MakeProgramFromSpec(*mesh_device_, spec);

    ProgramRunArgs params;
    params.kernel_run_args.push_back({.kernel_spec_name = "dm_kernel"});
    params.kernel_run_args.push_back({.kernel_spec_name = "compute_kernel"});

    EXPECT_NO_THROW(SetProgramRunArgs(program, params));
}

TEST_F(ProgramRunArgsTestQuasar, NamedAndVarargRTAsCoexistSucceeds) {
    // A kernel with both named RTAs (schema) and varargs (num_runtime_varargs).
    NodeCoord node{0, 0};
    ProgramSpec spec = MakeMinimalValidProgramSpec();
    spec.kernels[0].runtime_arg_schema.runtime_arg_names = {"input_ptr"};
    spec.kernels[0].advanced_options.num_runtime_varargs = 3;
    Program program = MakeProgramFromSpec(*mesh_device_, spec);

    ProgramRunArgs params;
    params.kernel_run_args.push_back({
        .kernel_spec_name = "dm_kernel",
        .runtime_arg_values = {{.node = node, .args = {{"input_ptr", 0x1000}}}},
        .advanced_options =
            AdvancedKernelRunArgs{
                .runtime_varargs = {{node, {7, 8, 9}}},
            },
    });
    params.kernel_run_args.push_back(MakeKernelRunArgs("compute_kernel", node, {}, {}));

    EXPECT_NO_THROW(SetProgramRunArgs(program, params));
}

// ============================================================================
// SECTION 4b: Borrowed-Memory DFB Attach Tests (Quasar)
// ============================================================================
//
// These tests cover the runtime attach path for DataflowBuffers that borrow their L1 storage
// from a TensorParameter (DataflowBufferSpec::borrowed_from). The flow:
//
//   1. MakeProgramFromSpec    → DFB added with config.borrows_memory = true; groups not yet
//                                created; no L1 allocated.
//   2. finalize_dataflow_buffer_configs (would normally run inside the dispatch pipeline at
//                                first enqueue) populates per-DFB `groups[].l1_by_core`
//                                entries with placeholder addr = 0.
//   3. SetProgramRunArgs / UpdateTensorArgs → AttachBorrowedDFBBuffers resolves the
//                                bound MeshTensor, extracts its reference-buffer address, and
//                                calls dfb->set_borrowed_memory_base_addr(addr), which
//                                overwrites every `groups[].l1_by_core` entry (and any
//                                populated `core_lookup_` entries) with that address.
//
// On mock device we don't run the full dispatch pipeline, so we trigger
// finalize_dataflow_buffer_configs() explicitly and verify state directly off the
// DataflowBufferImpl. allocate_dataflow_buffers requires a configured device and isn't needed
// for these checks — set_borrowed_memory_base_addr iterates `groups[].l1_by_core` regardless.

// Helper: build a ProgramSpec with a borrowed-memory DFB backed by a TensorParameter.
// DFB default size: 32 bytes (entry_size 16 * num_entries 2); fits inside
// MakeMinimalTensorParameter's 1x32 BFLOAT16 default (64 bytes).
inline ProgramSpec MakeBorrowedDFBProgramSpecForRunParams(
    const std::string& tensor_param_name = "borrowed_tensor",
    uint32_t dfb_entry_size = 16,
    uint32_t dfb_num_entries = 2) {
    NodeCoord node{0, 0};

    ProgramSpec spec;
    spec.program_id = "borrowed_dfb_test_program";

    auto producer = MakeMinimalDMKernel("producer");
    auto consumer = MakeMinimalDMKernel("consumer");
    auto dfb = MakeMinimalDFB("dfb", dfb_entry_size, dfb_num_entries);
    dfb.borrowed_from = tensor_param_name;

    BindDFBToKernel(producer, "dfb", "out", KernelSpec::DFBEndpointType::PRODUCER);
    BindDFBToKernel(consumer, "dfb", "in", KernelSpec::DFBEndpointType::CONSUMER);

    auto tensor_param = MakeMinimalTensorParameter(tensor_param_name, tt::tt_metal::BufferType::L1);
    BindTensorParameterToKernel(producer, tensor_param_name, "borrowed_t");

    spec.kernels = {producer, consumer};
    spec.dataflow_buffers = {dfb};
    spec.tensor_parameters = {tensor_param};
    spec.work_units = std::vector<WorkUnitSpec>{MakeMinimalWorkUnit("work_unit", node, {"producer", "consumer"})};
    return spec;
}

// Helper: build minimal ProgramRunArgs for the borrowed-DFB spec above. Both kernels are
// MakeMinimalDMKernel (no per-node or common args required), so kernel_run_args entries
// are empty schemas. Caller supplies the tensor_args entry separately.
inline ProgramRunArgs MakeBorrowedDFBRunParams() {
    NodeCoord node{0, 0};
    ProgramRunArgs params;
    params.kernel_run_args.push_back(MakeKernelRunArgs("producer", node, {}, {}));
    params.kernel_run_args.push_back(MakeKernelRunArgs("consumer", node, {}, {}));
    return params;
}

// Helper: peek at the DFB's per-core L1 base address as written into groups[].l1_by_core.
// All cores of a single DFB share the same address (lockstep allocation invariant), so
// returning the first entry is representative.
inline uint32_t PeekBorrowedDFBAddress(Program& program, const DFBSpecName& dfb_name) {
    const uint32_t dfb_id = program.impl().get_dfb_handle(dfb_name);
    const auto dfb = program.impl().get_dataflow_buffer(dfb_id);
    TT_FATAL(!dfb->groups.empty(), "DFB '{}' has no groups; finalize_dataflow_buffer_configs() not called?", dfb_name);
    TT_FATAL(!dfb->groups[0].l1_by_core.empty(), "DFB '{}' group 0 is empty", dfb_name);
    return dfb->groups[0].l1_by_core[0].second;
}

TEST_F(ProgramRunArgsTestQuasar, BorrowedDFB_BorrowsFlagPropagatesToConfig) {
    // MakeProgramFromSpec should set config.borrows_memory = true on the device-side DFB
    // config when (and only when) DataflowBufferSpec::borrowed_from is set.
    ProgramSpec spec = MakeBorrowedDFBProgramSpecForRunParams();
    Program program = MakeProgramFromSpec(*mesh_device_, spec);

    const uint32_t dfb_id = program.impl().get_dfb_handle("dfb");
    const auto dfb = program.impl().get_dataflow_buffer(dfb_id);
    EXPECT_TRUE(dfb->borrows_memory()) << "DFB declared borrowed_from should have config.borrows_memory = true";
}

TEST_F(ProgramRunArgsTestQuasar, BorrowedDFB_AttachWritesTensorAddressToDFB) {
    // After SetProgramRunArgs, the bound MeshTensor's address should appear in the
    // DFB's per-core L1 base address tables (overwriting Almeet's 0 placeholder).
    ProgramSpec spec = MakeBorrowedDFBProgramSpecForRunParams();
    Program program = MakeProgramFromSpec(*mesh_device_, spec);

    // Populate groups[].l1_by_core with placeholder addr = 0. (Normally invoked inside the
    // dispatch pipeline at first enqueue; we don't enqueue in unit tests.)
    program.impl().finalize_dataflow_buffer_configs();
    ASSERT_EQ(PeekBorrowedDFBAddress(program, "dfb"), 0u) << "DFB base addr before attach should be the placeholder 0";

    // Allocate the borrowed tensor and attach via SetProgramRunArgs.
    MeshTensor tensor = MeshTensor::allocate_on_device(*mesh_device_, spec.tensor_parameters[0].spec, TensorTopology{});
    ProgramRunArgs params = MakeBorrowedDFBRunParams();
    params.tensor_args = {
        ProgramRunArgs::TensorArgument{.tensor_parameter_name = "borrowed_tensor", .tensor = std::cref(tensor)},
    };
    SetProgramRunArgs(program, params);

    EXPECT_EQ(PeekBorrowedDFBAddress(program, "dfb"), static_cast<uint32_t>(tensor.address()))
        << "DFB base addr should match the borrowed MeshTensor's address after attach";
}

TEST_F(ProgramRunArgsTestQuasar, BorrowedDFB_UpdateTensorArgsRefreshesAddress) {
    // The cache-hit path: UpdateTensorArgs should re-attach the borrowed Buffer, refreshing
    // the DFB's base address to the new MeshTensor's address.
    ProgramSpec spec = MakeBorrowedDFBProgramSpecForRunParams();
    Program program = MakeProgramFromSpec(*mesh_device_, spec);
    program.impl().finalize_dataflow_buffer_configs();

    MeshTensor tensor1 =
        MeshTensor::allocate_on_device(*mesh_device_, spec.tensor_parameters[0].spec, TensorTopology{});
    ProgramRunArgs params = MakeBorrowedDFBRunParams();
    params.tensor_args = {
        ProgramRunArgs::TensorArgument{.tensor_parameter_name = "borrowed_tensor", .tensor = std::cref(tensor1)},
    };
    SetProgramRunArgs(program, params);
    ASSERT_EQ(PeekBorrowedDFBAddress(program, "dfb"), static_cast<uint32_t>(tensor1.address()));

    MeshTensor tensor2 =
        MeshTensor::allocate_on_device(*mesh_device_, spec.tensor_parameters[0].spec, TensorTopology{});
    ASSERT_NE(tensor1.address(), tensor2.address())
        << "Test pre-condition: two separate allocations should yield distinct addresses";

    std::vector<ProgramRunArgs::TensorArgument> tensor_args{
        ProgramRunArgs::TensorArgument{.tensor_parameter_name = "borrowed_tensor", .tensor = std::cref(tensor2)},
    };
    EXPECT_NO_THROW(UpdateTensorArgs(program, tensor_args));
    EXPECT_EQ(PeekBorrowedDFBAddress(program, "dfb"), static_cast<uint32_t>(tensor2.address()))
        << "DFB base addr should refresh to the new MeshTensor's address after UpdateTensorArgs";
}

// ============================================================================
// SECTION 5: Gen1 (WH/BH) Tests
// ============================================================================

class ProgramRunArgsTestGen1 : public ::testing::Test {
protected:
    void SetUp() override {
        slow_dispatch_override_.emplace();
        experimental::configure_mock_mode(tt::ARCH::WORMHOLE_B0, 1);
        mesh_device_ = distributed::MeshDevice::create(distributed::MeshDeviceConfig(distributed::MeshShape{1, 1}));
    }
    void TearDown() override {
        if (mesh_device_) {
            mesh_device_->close();
            mesh_device_.reset();
        }
        experimental::disable_mock_mode();
        slow_dispatch_override_.reset();
    }

    std::shared_ptr<distributed::MeshDevice> mesh_device_;
    std::optional<ScopedSlowDispatchOverride> slow_dispatch_override_;
};

// Create a gen1 ProgramSpec with a specified RTA schema on the DM kernel
inline ProgramSpec MakeGen1SpecWithRTAs(const NodeCoord& /*node*/, size_t num_per_node_rtas, size_t num_common_rtas) {
    ProgramSpec spec = MakeMinimalGen1ValidProgramSpec();

    spec.kernels[0].advanced_options =
        KernelAdvancedOptions{.num_runtime_varargs = num_per_node_rtas, .num_common_runtime_varargs = num_common_rtas};

    // kernels[1] has no varargs (defaults)

    return spec;
}

TEST_F(ProgramRunArgsTestGen1, SetRunParamsSucceeds_ZeroRTAs) {
    NodeCoord node{0, 0};
    ProgramSpec spec = MakeGen1SpecWithRTAs(node, 0, 0);
    Program program = MakeProgramFromSpec(*mesh_device_, spec);

    auto params = MakeRunParamsForMinimalSpec(node, {}, {});

    EXPECT_NO_THROW(SetProgramRunArgs(program, params));
}

TEST_F(ProgramRunArgsTestGen1, SetRunParamsSucceeds_PerNodeAndCommonRTAs) {
    NodeCoord node{0, 0};
    ProgramSpec spec = MakeGen1SpecWithRTAs(node, /*num_per_node_rtas=*/3, /*num_common_rtas=*/2);
    Program program = MakeProgramFromSpec(*mesh_device_, spec);

    auto params = MakeRunParamsForMinimalSpec(node, {100, 200, 300}, {10, 20});

    EXPECT_NO_THROW(SetProgramRunArgs(program, params));
}

TEST_F(ProgramRunArgsTestGen1, WrongRuntimeArgsCountFails) {
    NodeCoord node{0, 0};
    ProgramSpec spec = MakeGen1SpecWithRTAs(node, /*num_per_node_rtas=*/3, /*num_common_rtas=*/0);
    Program program = MakeProgramFromSpec(*mesh_device_, spec);

    // Provide wrong count (2 instead of 3)
    auto params = MakeRunParamsForMinimalSpec(node, {1, 2}, {});

    EXPECT_THAT(
        [&] { SetProgramRunArgs(program, params); },
        ::testing::ThrowsMessage<std::runtime_error>(
            ::testing::HasSubstr("expects 3 vararg runtime args, but 2 were provided")));
}

// ============================================================================
// SECTION: TensorParameter Run-Params Validation Tests (Gen1 / WH)
// ============================================================================
// Validation paths exercised by SetProgramRunArgs / ValidateProgramRunArgs for the
// Metal 2.0 TensorAccessor binding feature.

TEST_F(ProgramRunArgsTestGen1, MissingTensorArgFails) {
    // Spec declares a TensorParameter; user supplies no TensorArgument entry for it.
    NodeCoord node{0, 0};
    ProgramSpec spec = MakeMinimalGen1ValidProgramSpec();
    spec.tensor_parameters = {MakeMinimalTensorParameter("input_tensor")};
    BindTensorParameterToKernel(spec.kernels[0], "input_tensor", "input_ta");

    Program program = MakeProgramFromSpec(*mesh_device_, spec);

    // Run params with no tensor_args entries.
    auto params = MakeRunParamsForMinimalSpec(node, {}, {});

    EXPECT_THAT(
        [&] { SetProgramRunArgs(program, params); },
        ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr(
            "TensorParameter 'input_tensor' is declared in the Program but has no TensorArgument entry")));
}

TEST_F(ProgramRunArgsTestGen1, DuplicateTensorArgFails) {
    NodeCoord node{0, 0};
    ProgramSpec spec = MakeMinimalGen1ValidProgramSpec();
    auto binding = MakeMinimalTensorParameter("input_tensor");
    spec.tensor_parameters = {binding};
    BindTensorParameterToKernel(spec.kernels[0], "input_tensor", "input_ta");

    Program program = MakeProgramFromSpec(*mesh_device_, spec);

    // Allocate one MeshTensor; reference it twice in tensor_args under the same name.
    MeshTensor tensor = MeshTensor::allocate_on_device(*mesh_device_, binding.spec, TensorTopology{});

    auto params = MakeRunParamsForMinimalSpec(node, {}, {});
    params.tensor_args = {
        ProgramRunArgs::TensorArgument{.tensor_parameter_name = "input_tensor", .tensor = std::cref(tensor)},
        ProgramRunArgs::TensorArgument{.tensor_parameter_name = "input_tensor", .tensor = std::cref(tensor)},
    };

    EXPECT_THAT(
        [&] { SetProgramRunArgs(program, params); },
        ::testing::ThrowsMessage<std::runtime_error>(
            ::testing::HasSubstr("Duplicate tensor_parameter_name 'input_tensor'")));
}

TEST_F(ProgramRunArgsTestGen1, UnknownTensorParameterInRunParamsFails) {
    NodeCoord node{0, 0};
    ProgramSpec spec = MakeMinimalGen1ValidProgramSpec();
    auto binding = MakeMinimalTensorParameter("input_tensor");
    spec.tensor_parameters = {binding};
    BindTensorParameterToKernel(spec.kernels[0], "input_tensor", "input_ta");

    Program program = MakeProgramFromSpec(*mesh_device_, spec);

    MeshTensor tensor = MeshTensor::allocate_on_device(*mesh_device_, binding.spec, TensorTopology{});

    // tensor_parameter_name doesn't match any TensorParameter in the spec.
    auto params = MakeRunParamsForMinimalSpec(node, {}, {});
    params.tensor_args = {
        ProgramRunArgs::TensorArgument{.tensor_parameter_name = "ghost_tensor", .tensor = std::cref(tensor)},
    };

    EXPECT_THAT(
        [&] { SetProgramRunArgs(program, params); },
        ::testing::ThrowsMessage<std::runtime_error>(
            ::testing::HasSubstr("TensorArgument references unknown TensorParameter 'ghost_tensor'")));
}

TEST_F(ProgramRunArgsTestGen1, TensorSpecMismatchFails) {
    NodeCoord node{0, 0};
    ProgramSpec spec = MakeMinimalGen1ValidProgramSpec();

    // Binding declares spec with shape {1, 32}; runtime tensor has a different shape ({1, 64}).
    auto binding = MakeMinimalTensorParameter("input_tensor");  // default {1, 32}
    spec.tensor_parameters = {binding};
    BindTensorParameterToKernel(spec.kernels[0], "input_tensor", "input_ta");

    Program program = MakeProgramFromSpec(*mesh_device_, spec);

    auto page_config = tt::tt_metal::PageConfig(tt::tt_metal::Layout::ROW_MAJOR);
    auto memory_config =
        tt::tt_metal::MemoryConfig{tt::tt_metal::TensorMemoryLayout::INTERLEAVED, tt::tt_metal::BufferType::DRAM};
    auto tensor_layout = tt::tt_metal::TensorLayout(tt::tt_metal::DataType::BFLOAT16, page_config, memory_config);
    auto wrong_spec = tt::tt_metal::TensorSpec(tt::tt_metal::Shape{1, 64}, tensor_layout);  // different shape!
    MeshTensor tensor = MeshTensor::allocate_on_device(*mesh_device_, wrong_spec, TensorTopology{});

    auto params = MakeRunParamsForMinimalSpec(node, {}, {});
    params.tensor_args = {
        ProgramRunArgs::TensorArgument{.tensor_parameter_name = "input_tensor", .tensor = std::cref(tensor)},
    };

    EXPECT_THAT(
        [&] { SetProgramRunArgs(program, params); },
        ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr(
            "TensorArgument for binding 'input_tensor' supplied a MeshTensor whose TensorSpec does not match")));
}

// ============================================================================
// SECTION: UpdateTensorArgs Tests (Gen1 / WH)
// ============================================================================
// UpdateTensorArgs is the partial-update fast path: it patches only the tensor binding
// address slots in the per-kernel CRTA buffer, leaving everything else (named CRTAs,
// vararg CRTAs, all RTAs) statefully unchanged from the most recent SetProgramRunArgs
// call.

// Helper: allocate a MeshTensor matching the given TensorParameter's spec.
inline MeshTensor AllocateTensorForBinding(distributed::MeshDevice& mesh_device, const TensorParameter& binding) {
    return MeshTensor::allocate_on_device(mesh_device, binding.spec, TensorTopology{});
}

// Helper: read the patched tensor binding address out of a kernel's CRTA buffer.
// Mirrors the offset arithmetic UpdateTensorArgs uses internally, so a regression in either
// site shows up as a test failure.
inline uint32_t ReadBindingAddressFromCRTA(
    const Program& program, const KernelSpecName& kernel_name, const std::string& tensor_parameter_name) {
    auto kernel = program.impl().get_kernel_by_spec_name(kernel_name);
    for (const auto& handle : kernel->tensor_binding_handles()) {
        if (handle.tensor_parameter_name == tensor_parameter_name) {
            const uint32_t word_index = handle.addr_crta_offset / sizeof(uint32_t);
            return kernel->common_runtime_args_data().data()[word_index];
        }
    }
    ADD_FAILURE() << "No binding handle for '" << tensor_parameter_name << "' on kernel '" << kernel_name << "'";
    return 0;
}

TEST_F(ProgramRunArgsTestGen1, UpdateTensorArgs_BeforeSetProgramRunParametersFails) {
    ProgramSpec spec = MakeMinimalGen1ValidProgramSpec();
    auto binding = MakeMinimalTensorParameter("input_tensor");
    spec.tensor_parameters = {binding};
    BindTensorParameterToKernel(spec.kernels[0], "input_tensor", "input_ta");

    Program program = MakeProgramFromSpec(*mesh_device_, spec);

    MeshTensor tensor = AllocateTensorForBinding(*mesh_device_, binding);
    std::vector<ProgramRunArgs::TensorArgument> tensor_args{
        ProgramRunArgs::TensorArgument{.tensor_parameter_name = "input_tensor", .tensor = std::cref(tensor)},
    };

    // No SetProgramRunArgs call before the partial update.
    EXPECT_THAT(
        [&] { UpdateTensorArgs(program, tensor_args); },
        ::testing::ThrowsMessage<std::runtime_error>(
            ::testing::HasSubstr("UpdateTensorArgs called on Program before SetProgramRunArgs")));
}

TEST_F(ProgramRunArgsTestGen1, UpdateTensorArgs_MissingTensorArgFails) {
    NodeCoord node{0, 0};
    ProgramSpec spec = MakeMinimalGen1ValidProgramSpec();
    auto binding = MakeMinimalTensorParameter("input_tensor");
    spec.tensor_parameters = {binding};
    BindTensorParameterToKernel(spec.kernels[0], "input_tensor", "input_ta");

    Program program = MakeProgramFromSpec(*mesh_device_, spec);

    // Establish baseline state.
    MeshTensor tensor = AllocateTensorForBinding(*mesh_device_, binding);
    auto params = MakeRunParamsForMinimalSpec(node, {}, {});
    params.tensor_args = {
        ProgramRunArgs::TensorArgument{.tensor_parameter_name = "input_tensor", .tensor = std::cref(tensor)},
    };
    SetProgramRunArgs(program, params);

    // Empty tensor_args fails the completeness check.
    std::vector<ProgramRunArgs::TensorArgument> empty;
    EXPECT_THAT(
        [&] { UpdateTensorArgs(program, empty); },
        ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr(
            "TensorParameter 'input_tensor' is declared in the Program but has no TensorArgument entry")));
}

TEST_F(ProgramRunArgsTestGen1, UpdateTensorArgs_DuplicateTensorArgFails) {
    NodeCoord node{0, 0};
    ProgramSpec spec = MakeMinimalGen1ValidProgramSpec();
    auto binding = MakeMinimalTensorParameter("input_tensor");
    spec.tensor_parameters = {binding};
    BindTensorParameterToKernel(spec.kernels[0], "input_tensor", "input_ta");

    Program program = MakeProgramFromSpec(*mesh_device_, spec);

    MeshTensor tensor = AllocateTensorForBinding(*mesh_device_, binding);
    auto params = MakeRunParamsForMinimalSpec(node, {}, {});
    params.tensor_args = {
        ProgramRunArgs::TensorArgument{.tensor_parameter_name = "input_tensor", .tensor = std::cref(tensor)},
    };
    SetProgramRunArgs(program, params);

    std::vector<ProgramRunArgs::TensorArgument> tensor_args{
        ProgramRunArgs::TensorArgument{.tensor_parameter_name = "input_tensor", .tensor = std::cref(tensor)},
        ProgramRunArgs::TensorArgument{.tensor_parameter_name = "input_tensor", .tensor = std::cref(tensor)},
    };
    EXPECT_THAT(
        [&] { UpdateTensorArgs(program, tensor_args); },
        ::testing::ThrowsMessage<std::runtime_error>(
            ::testing::HasSubstr("Duplicate tensor_parameter_name 'input_tensor'")));
}

TEST_F(ProgramRunArgsTestGen1, UpdateTensorArgs_UnknownTensorParameterFails) {
    NodeCoord node{0, 0};
    ProgramSpec spec = MakeMinimalGen1ValidProgramSpec();
    auto binding = MakeMinimalTensorParameter("input_tensor");
    spec.tensor_parameters = {binding};
    BindTensorParameterToKernel(spec.kernels[0], "input_tensor", "input_ta");

    Program program = MakeProgramFromSpec(*mesh_device_, spec);

    MeshTensor tensor = AllocateTensorForBinding(*mesh_device_, binding);
    auto params = MakeRunParamsForMinimalSpec(node, {}, {});
    params.tensor_args = {
        ProgramRunArgs::TensorArgument{.tensor_parameter_name = "input_tensor", .tensor = std::cref(tensor)},
    };
    SetProgramRunArgs(program, params);

    std::vector<ProgramRunArgs::TensorArgument> tensor_args{
        ProgramRunArgs::TensorArgument{.tensor_parameter_name = "ghost_tensor", .tensor = std::cref(tensor)},
    };
    EXPECT_THAT(
        [&] { UpdateTensorArgs(program, tensor_args); },
        ::testing::ThrowsMessage<std::runtime_error>(
            ::testing::HasSubstr("TensorArgument references unknown TensorParameter 'ghost_tensor'")));
}

TEST_F(ProgramRunArgsTestGen1, UpdateTensorArgs_TensorSpecMismatchFails) {
    NodeCoord node{0, 0};
    ProgramSpec spec = MakeMinimalGen1ValidProgramSpec();
    auto binding = MakeMinimalTensorParameter("input_tensor");  // shape {1, 32}
    spec.tensor_parameters = {binding};
    BindTensorParameterToKernel(spec.kernels[0], "input_tensor", "input_ta");

    Program program = MakeProgramFromSpec(*mesh_device_, spec);

    MeshTensor tensor = AllocateTensorForBinding(*mesh_device_, binding);
    auto params = MakeRunParamsForMinimalSpec(node, {}, {});
    params.tensor_args = {
        ProgramRunArgs::TensorArgument{.tensor_parameter_name = "input_tensor", .tensor = std::cref(tensor)},
    };
    SetProgramRunArgs(program, params);

    // Different shape: {1, 64} instead of declared {1, 32}.
    auto page_config = tt::tt_metal::PageConfig(tt::tt_metal::Layout::ROW_MAJOR);
    auto memory_config =
        tt::tt_metal::MemoryConfig{tt::tt_metal::TensorMemoryLayout::INTERLEAVED, tt::tt_metal::BufferType::DRAM};
    auto tensor_layout = tt::tt_metal::TensorLayout(tt::tt_metal::DataType::BFLOAT16, page_config, memory_config);
    auto wrong_spec = tt::tt_metal::TensorSpec(tt::tt_metal::Shape{1, 64}, tensor_layout);
    MeshTensor wrong_tensor = MeshTensor::allocate_on_device(*mesh_device_, wrong_spec, TensorTopology{});

    std::vector<ProgramRunArgs::TensorArgument> tensor_args{
        ProgramRunArgs::TensorArgument{.tensor_parameter_name = "input_tensor", .tensor = std::cref(wrong_tensor)},
    };
    EXPECT_THAT(
        [&] { UpdateTensorArgs(program, tensor_args); },
        ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr(
            "TensorArgument for binding 'input_tensor' supplied a MeshTensor whose TensorSpec does not match")));
}

TEST_F(ProgramRunArgsTestGen1, UpdateTensorArgs_PatchesBindingAddress) {
    NodeCoord node{0, 0};
    ProgramSpec spec = MakeMinimalGen1ValidProgramSpec();
    auto binding = MakeMinimalTensorParameter("input_tensor");
    spec.tensor_parameters = {binding};
    BindTensorParameterToKernel(spec.kernels[0], "input_tensor", "input_ta");

    Program program = MakeProgramFromSpec(*mesh_device_, spec);

    // First enqueue: full SetProgramRunArgs with tensor T1.
    MeshTensor tensor1 = AllocateTensorForBinding(*mesh_device_, binding);
    auto params = MakeRunParamsForMinimalSpec(node, {}, {});
    params.tensor_args = {
        ProgramRunArgs::TensorArgument{.tensor_parameter_name = "input_tensor", .tensor = std::cref(tensor1)},
    };
    SetProgramRunArgs(program, params);
    ASSERT_EQ(
        ReadBindingAddressFromCRTA(program, "dm_kernel", "input_tensor"), static_cast<uint32_t>(tensor1.address()));

    // Second enqueue: partial update to tensor T2.
    MeshTensor tensor2 = AllocateTensorForBinding(*mesh_device_, binding);
    ASSERT_NE(tensor1.address(), tensor2.address())
        << "Test pre-condition: two separate allocations should yield distinct addresses";

    std::vector<ProgramRunArgs::TensorArgument> tensor_args{
        ProgramRunArgs::TensorArgument{.tensor_parameter_name = "input_tensor", .tensor = std::cref(tensor2)},
    };
    EXPECT_NO_THROW(UpdateTensorArgs(program, tensor_args));

    EXPECT_EQ(
        ReadBindingAddressFromCRTA(program, "dm_kernel", "input_tensor"), static_cast<uint32_t>(tensor2.address()));
}

TEST_F(ProgramRunArgsTestGen1, UpdateTensorArgs_LeavesNamedCRTAsUnchanged) {
    NodeCoord node{0, 0};
    ProgramSpec spec = MakeMinimalGen1ValidProgramSpec();
    auto binding = MakeMinimalTensorParameter("input_tensor");
    spec.tensor_parameters = {binding};
    // Schema with a named CRTA preceding the binding-address section.
    spec.kernels[0].runtime_arg_schema.common_runtime_arg_names = {"tile_count"};
    BindTensorParameterToKernel(spec.kernels[0], "input_tensor", "input_ta");

    Program program = MakeProgramFromSpec(*mesh_device_, spec);

    MeshTensor tensor1 = AllocateTensorForBinding(*mesh_device_, binding);
    constexpr uint32_t kNamedCRTAValue = 0xABCD1234;
    ProgramRunArgs params;
    params.kernel_run_args.push_back({
        .kernel_spec_name = "dm_kernel",
        .common_runtime_arg_values = {{"tile_count", kNamedCRTAValue}},
    });
    params.kernel_run_args.push_back(MakeKernelRunArgs("compute_kernel", node, {}, {}));
    params.tensor_args = {
        ProgramRunArgs::TensorArgument{.tensor_parameter_name = "input_tensor", .tensor = std::cref(tensor1)},
    };
    SetProgramRunArgs(program, params);

    // Capture the named CRTA's slot value before the partial update.
    auto dm_kernel = program.impl().get_kernel_by_spec_name("dm_kernel");
    const auto* crta_data_before = dm_kernel->common_runtime_args_data().data();
    ASSERT_EQ(crta_data_before[0], kNamedCRTAValue) << "named CRTA should occupy slot 0 of the CRTA buffer";

    // Partial update.
    MeshTensor tensor2 = AllocateTensorForBinding(*mesh_device_, binding);
    std::vector<ProgramRunArgs::TensorArgument> tensor_args{
        ProgramRunArgs::TensorArgument{.tensor_parameter_name = "input_tensor", .tensor = std::cref(tensor2)},
    };
    UpdateTensorArgs(program, tensor_args);

    // Named CRTA preserved; binding address patched.
    const auto* crta_data_after = dm_kernel->common_runtime_args_data().data();
    EXPECT_EQ(crta_data_after[0], kNamedCRTAValue) << "named CRTA must be untouched by UpdateTensorArgs";
    EXPECT_EQ(
        ReadBindingAddressFromCRTA(program, "dm_kernel", "input_tensor"), static_cast<uint32_t>(tensor2.address()));
}

TEST_F(ProgramRunArgsTestGen1, UpdateTensorArgs_PatchesAllKernelsBoundToSameTensor) {
    NodeCoord node{0, 0};
    ProgramSpec spec = MakeMinimalGen1ValidProgramSpec();
    auto binding = MakeMinimalTensorParameter("shared_tensor");
    spec.tensor_parameters = {binding};
    // Bind the same TensorParameter on both kernels.
    BindTensorParameterToKernel(spec.kernels[0], "shared_tensor", "shared_ta_dm");
    BindTensorParameterToKernel(spec.kernels[1], "shared_tensor", "shared_ta_compute");

    Program program = MakeProgramFromSpec(*mesh_device_, spec);

    MeshTensor tensor1 = AllocateTensorForBinding(*mesh_device_, binding);
    auto params = MakeRunParamsForMinimalSpec(node, {}, {});
    params.tensor_args = {
        ProgramRunArgs::TensorArgument{.tensor_parameter_name = "shared_tensor", .tensor = std::cref(tensor1)},
    };
    SetProgramRunArgs(program, params);

    MeshTensor tensor2 = AllocateTensorForBinding(*mesh_device_, binding);
    std::vector<ProgramRunArgs::TensorArgument> tensor_args{
        ProgramRunArgs::TensorArgument{.tensor_parameter_name = "shared_tensor", .tensor = std::cref(tensor2)},
    };
    UpdateTensorArgs(program, tensor_args);

    EXPECT_EQ(
        ReadBindingAddressFromCRTA(program, "dm_kernel", "shared_tensor"), static_cast<uint32_t>(tensor2.address()));
    EXPECT_EQ(
        ReadBindingAddressFromCRTA(program, "compute_kernel", "shared_tensor"),
        static_cast<uint32_t>(tensor2.address()));
}

// ============================================================================
// SECTION: Dynamic Tensor Shape Run-Params Tests (Gen1 / WH)
// ============================================================================
// Exercises the dynamic_tensor_shape opt-in on TensorParameter:
//   - validation: tensor_layout must still match exactly; logical_shape may vary per-dim
//     but rank must be preserved.
//   - CRTA contents: for sharded TPs, the runtime tensor's shape-in-pages is written into
//     CRTAs immediately after the binding's address slot, on both SetProgramRunArgs
//     and UpdateTensorArgs.

TEST_F(ProgramRunArgsTestGen1, DynamicTensorShape_InterleavedAcceptsDifferentShape) {
    // Declared shape {1, 32}; runtime tensor with shape {1, 64} is accepted because
    // dynamic_tensor_shape is set and everything else matches.
    NodeCoord node{0, 0};
    ProgramSpec spec = MakeMinimalGen1ValidProgramSpec();
    auto binding = MakeMinimalTensorParameter("input_tensor");  // shape {1, 32}
    binding.advanced_options = TensorParameterAdvancedOptions{.dynamic_tensor_shape = true};
    spec.tensor_parameters = {binding};
    BindTensorParameterToKernel(spec.kernels[0], "input_tensor", "input_ta");

    Program program = MakeProgramFromSpec(*mesh_device_, spec);

    // Different shape, same layout.
    auto wrong_spec_layout = binding.spec.tensor_layout();
    auto larger_spec = tt::tt_metal::TensorSpec(tt::tt_metal::Shape{1, 64}, wrong_spec_layout);
    MeshTensor tensor = MeshTensor::allocate_on_device(*mesh_device_, larger_spec, TensorTopology{});

    auto params = MakeRunParamsForMinimalSpec(node, {}, {});
    params.tensor_args = {
        ProgramRunArgs::TensorArgument{.tensor_parameter_name = "input_tensor", .tensor = std::cref(tensor)},
    };
    EXPECT_NO_THROW(SetProgramRunArgs(program, params));
}

TEST_F(ProgramRunArgsTestGen1, DynamicTensorShape_DTypeMismatchStillFails) {
    // dynamic_tensor_shape loosens shape only — dtype (part of tensor_layout) must still match.
    NodeCoord node{0, 0};
    ProgramSpec spec = MakeMinimalGen1ValidProgramSpec();
    auto binding = MakeMinimalTensorParameter("input_tensor");  // BFLOAT16
    binding.advanced_options = TensorParameterAdvancedOptions{.dynamic_tensor_shape = true};
    spec.tensor_parameters = {binding};
    BindTensorParameterToKernel(spec.kernels[0], "input_tensor", "input_ta");

    Program program = MakeProgramFromSpec(*mesh_device_, spec);

    // Different dtype (UINT32 instead of BFLOAT16).
    auto page_config = tt::tt_metal::PageConfig(tt::tt_metal::Layout::ROW_MAJOR);
    auto memory_config =
        tt::tt_metal::MemoryConfig{tt::tt_metal::TensorMemoryLayout::INTERLEAVED, tt::tt_metal::BufferType::DRAM};
    auto wrong_layout = tt::tt_metal::TensorLayout(tt::tt_metal::DataType::UINT32, page_config, memory_config);
    auto wrong_spec = tt::tt_metal::TensorSpec(tt::tt_metal::Shape{1, 32}, wrong_layout);
    MeshTensor tensor = MeshTensor::allocate_on_device(*mesh_device_, wrong_spec, TensorTopology{});

    auto params = MakeRunParamsForMinimalSpec(node, {}, {});
    params.tensor_args = {
        ProgramRunArgs::TensorArgument{.tensor_parameter_name = "input_tensor", .tensor = std::cref(tensor)},
    };
    EXPECT_THAT(
        [&] { SetProgramRunArgs(program, params); },
        ::testing::ThrowsMessage<std::runtime_error>(
            ::testing::HasSubstr("tensor_layout does not match the binding's declared layout")));
}

TEST_F(ProgramRunArgsTestGen1, DynamicTensorShape_RankMismatchFails) {
    // dynamic_tensor_shape lets per-dim shape values vary, but the rank must remain constant.
    NodeCoord node{0, 0};
    ProgramSpec spec = MakeMinimalGen1ValidProgramSpec();
    auto binding = MakeMinimalTensorParameter("input_tensor");  // rank-2 shape {1, 32}
    binding.advanced_options = TensorParameterAdvancedOptions{.dynamic_tensor_shape = true};
    spec.tensor_parameters = {binding};
    BindTensorParameterToKernel(spec.kernels[0], "input_tensor", "input_ta");

    Program program = MakeProgramFromSpec(*mesh_device_, spec);

    // Rank-3 tensor with same layout.
    auto wrong_spec = tt::tt_metal::TensorSpec(tt::tt_metal::Shape{1, 1, 32}, binding.spec.tensor_layout());
    MeshTensor tensor = MeshTensor::allocate_on_device(*mesh_device_, wrong_spec, TensorTopology{});

    auto params = MakeRunParamsForMinimalSpec(node, {}, {});
    params.tensor_args = {
        ProgramRunArgs::TensorArgument{.tensor_parameter_name = "input_tensor", .tensor = std::cref(tensor)},
    };
    EXPECT_THAT(
        [&] { SetProgramRunArgs(program, params); },
        ::testing::ThrowsMessage<std::runtime_error>(
            ::testing::HasSubstr("logical_shape rank (3) differs from the declared rank (2)")));
}

// Read the runtime field CRTA words (immediately after the address slot) for a tensor binding.
// Returns the [rank] shape words in declaration order.
inline std::vector<uint32_t> ReadBindingShapeFromCRTA(
    const Program& program, const KernelSpecName& kernel_name, const std::string& tensor_parameter_name) {
    auto kernel = program.impl().get_kernel_by_spec_name(kernel_name);
    for (const auto& handle : kernel->tensor_binding_handles()) {
        if (handle.tensor_parameter_name == tensor_parameter_name) {
            const uint32_t base_word = handle.addr_crta_offset / sizeof(uint32_t);
            const auto* data = kernel->common_runtime_args_data().data();
            std::vector<uint32_t> shape;
            shape.reserve(handle.num_runtime_field_crta_words);
            for (uint32_t i = 0; i < handle.num_runtime_field_crta_words; ++i) {
                shape.push_back(data[base_word + 1u + i]);
            }
            return shape;
        }
    }
    ADD_FAILURE() << "No binding handle for '" << tensor_parameter_name << "' on kernel '" << kernel_name << "'";
    return {};
}

// Compute the expected tensor_shape_in_pages from a TensorSpec via its BufferDistributionSpec.
// This is the source of truth for what gets written into the runtime field CRTA section.
inline std::vector<uint32_t> ExpectedShapeInPagesFromSpec(const tt::tt_metal::TensorSpec& spec) {
    const auto bds = spec.compute_buffer_sharding_args().buffer_distribution_spec();
    if (!bds.has_value()) {
        ADD_FAILURE() << "TensorSpec has no BufferDistributionSpec; expected sharded";
        return {};
    }
    const auto& shape = bds->tensor_shape_in_pages();
    std::vector<uint32_t> out;
    out.reserve(shape.rank());
    for (size_t i = 0; i < shape.rank(); ++i) {
        out.push_back(static_cast<uint32_t>(shape[i]));
    }
    return out;
}

TEST_F(ProgramRunArgsTestGen1, DynamicTensorShape_ShardedSetWritesShapeIntoCRTAs) {
    // Sharded + dynamic_tensor_shape: SetProgramRunArgs must write the actual runtime
    // tensor's tensor_shape_in_pages into the CRTA section that follows the binding's address.
    // Layout: HEIGHT_SHARDED with shard_shape {32, 32} on 2 cores → 2 shards along height.
    NodeCoord node{0, 0};
    ProgramSpec spec = MakeMinimalGen1ValidProgramSpec();
    auto binding =
        MakeShardedTensorParameter("input_tensor", tt::tt_metal::Shape{1, 1, 64, 32}, {32, 32}, /*num_cores=*/2);
    binding.advanced_options = TensorParameterAdvancedOptions{.dynamic_tensor_shape = true};
    spec.tensor_parameters = {binding};
    BindTensorParameterToKernel(spec.kernels[0], "input_tensor", "input_ta");

    Program program = MakeProgramFromSpec(*mesh_device_, spec);

    MeshTensor tensor = MeshTensor::allocate_on_device(*mesh_device_, binding.spec, TensorTopology{});
    auto params = MakeRunParamsForMinimalSpec(node, {}, {});
    params.tensor_args = {
        ProgramRunArgs::TensorArgument{.tensor_parameter_name = "input_tensor", .tensor = std::cref(tensor)},
    };
    SetProgramRunArgs(program, params);

    // BDS flattens shape per its sharding scheme; derive expected value from BDS directly
    // (source of truth — same path the runtime uses to populate the CRTA slots).
    EXPECT_EQ(
        ReadBindingShapeFromCRTA(program, "dm_kernel", "input_tensor"), ExpectedShapeInPagesFromSpec(binding.spec));
}

TEST_F(ProgramRunArgsTestGen1, DynamicTensorShape_ShardedUpdateRefreshesShape) {
    // Sharded + dynamic_tensor_shape: UpdateTensorArgs must refresh BOTH the address slot and
    // the runtime shape slots when bound to a tensor of different shape (but same layout).
    NodeCoord node{0, 0};
    ProgramSpec spec = MakeMinimalGen1ValidProgramSpec();
    auto binding =
        MakeShardedTensorParameter("input_tensor", tt::tt_metal::Shape{1, 1, 64, 32}, {32, 32}, /*num_cores=*/2);
    binding.advanced_options = TensorParameterAdvancedOptions{.dynamic_tensor_shape = true};
    spec.tensor_parameters = {binding};
    BindTensorParameterToKernel(spec.kernels[0], "input_tensor", "input_ta");

    Program program = MakeProgramFromSpec(*mesh_device_, spec);

    // First Set: tensor of declared shape (2 shards along height).
    MeshTensor tensor1 = MeshTensor::allocate_on_device(*mesh_device_, binding.spec, TensorTopology{});
    auto params = MakeRunParamsForMinimalSpec(node, {}, {});
    params.tensor_args = {
        ProgramRunArgs::TensorArgument{.tensor_parameter_name = "input_tensor", .tensor = std::cref(tensor1)},
    };
    SetProgramRunArgs(program, params);
    ASSERT_EQ(
        ReadBindingShapeFromCRTA(program, "dm_kernel", "input_tensor"), ExpectedShapeInPagesFromSpec(binding.spec));

    // Second Update: smaller-shape tensor (1 shard along height). Same shard_spec, fewer shards.
    auto smaller_spec = tt::tt_metal::TensorSpec(tt::tt_metal::Shape{1, 1, 32, 32}, binding.spec.tensor_layout());
    MeshTensor tensor2 = MeshTensor::allocate_on_device(*mesh_device_, smaller_spec, TensorTopology{});
    std::vector<ProgramRunArgs::TensorArgument> tensor_args{
        ProgramRunArgs::TensorArgument{.tensor_parameter_name = "input_tensor", .tensor = std::cref(tensor2)},
    };
    EXPECT_NO_THROW(UpdateTensorArgs(program, tensor_args));

    EXPECT_EQ(
        ReadBindingAddressFromCRTA(program, "dm_kernel", "input_tensor"), static_cast<uint32_t>(tensor2.address()));
    EXPECT_EQ(
        ReadBindingShapeFromCRTA(program, "dm_kernel", "input_tensor"), ExpectedShapeInPagesFromSpec(smaller_spec));
}

// ============================================================================
// SECTION: match_padded_shape_only Run-Params Tests (Gen1 / WH)
// ============================================================================
// Exercises the match_padded_shape_only opt-in on TensorParameter:
//   - tensor_layout must still match exactly (dtype / page_config / memory_config / alignment).
//   - padded_shape() must match exactly across binds.
//   - logical_shape() may differ provided the resulting padded_shape is unchanged.
//   - Strictly weaker than dynamic_tensor_shape; no device-side CTA/CRTA effect.

TEST_F(ProgramRunArgsTestGen1, MatchPaddedShapeOnly_AcceptsDifferentLogicalShape) {
    // Declared logical shape {1, 1, 32, 32} on TILE layout produces padded_shape {1, 1, 32, 32}.
    // A runtime tensor with logical_shape {1, 1, 20, 20} pads up to the same {1, 1, 32, 32}, so
    // match_padded_shape_only accepts the rebind.
    NodeCoord node{0, 0};
    ProgramSpec spec = MakeMinimalGen1ValidProgramSpec();
    auto page_config = tt::tt_metal::PageConfig(tt::tt_metal::Layout::TILE);
    auto memory_config =
        tt::tt_metal::MemoryConfig{tt::tt_metal::TensorMemoryLayout::INTERLEAVED, tt::tt_metal::BufferType::DRAM};
    auto layout = tt::tt_metal::TensorLayout(tt::tt_metal::DataType::BFLOAT16, page_config, memory_config);
    auto declared_spec = tt::tt_metal::TensorSpec(tt::tt_metal::Shape{1, 1, 32, 32}, layout);
    TensorParameter binding{
        .unique_id = "input_tensor",
        .spec = declared_spec,
        .advanced_options = TensorParameterAdvancedOptions{.match_padded_shape_only = true},
    };
    spec.tensor_parameters = {binding};
    BindTensorParameterToKernel(spec.kernels[0], "input_tensor", "input_ta");

    Program program = MakeProgramFromSpec(*mesh_device_, spec);

    // Runtime tensor: smaller logical shape that pads up to the same padded_shape (one tile).
    auto runtime_spec = tt::tt_metal::TensorSpec(tt::tt_metal::Shape{1, 1, 20, 20}, layout);
    ASSERT_EQ(runtime_spec.padded_shape(), declared_spec.padded_shape())
        << "Test precondition: runtime logical {1,1,20,20} should pad to the same {1,1,32,32} as declared.";
    MeshTensor tensor = MeshTensor::allocate_on_device(*mesh_device_, runtime_spec, TensorTopology{});

    auto params = MakeRunParamsForMinimalSpec(node, {}, {});
    params.tensor_args = {
        ProgramRunArgs::TensorArgument{.tensor_parameter_name = "input_tensor", .tensor = std::cref(tensor)},
    };
    EXPECT_NO_THROW(SetProgramRunArgs(program, params));
}

TEST_F(ProgramRunArgsTestGen1, MatchPaddedShapeOnly_PaddedShapeMismatchFails) {
    // Same TensorLayout, but a logical shape that pads to a DIFFERENT padded_shape is rejected.
    NodeCoord node{0, 0};
    ProgramSpec spec = MakeMinimalGen1ValidProgramSpec();
    auto page_config = tt::tt_metal::PageConfig(tt::tt_metal::Layout::TILE);
    auto memory_config =
        tt::tt_metal::MemoryConfig{tt::tt_metal::TensorMemoryLayout::INTERLEAVED, tt::tt_metal::BufferType::DRAM};
    auto layout = tt::tt_metal::TensorLayout(tt::tt_metal::DataType::BFLOAT16, page_config, memory_config);
    auto declared_spec = tt::tt_metal::TensorSpec(tt::tt_metal::Shape{1, 1, 32, 32}, layout);
    TensorParameter binding{
        .unique_id = "input_tensor",
        .spec = declared_spec,
        .advanced_options = TensorParameterAdvancedOptions{.match_padded_shape_only = true},
    };
    spec.tensor_parameters = {binding};
    BindTensorParameterToKernel(spec.kernels[0], "input_tensor", "input_ta");

    Program program = MakeProgramFromSpec(*mesh_device_, spec);

    // Runtime tensor: logical shape that pads up to a DIFFERENT padded_shape (two tiles wide).
    auto runtime_spec = tt::tt_metal::TensorSpec(tt::tt_metal::Shape{1, 1, 32, 33}, layout);
    ASSERT_NE(runtime_spec.padded_shape(), declared_spec.padded_shape())
        << "Test precondition: runtime logical {1,1,32,33} should pad to a different padded_shape than declared.";
    MeshTensor tensor = MeshTensor::allocate_on_device(*mesh_device_, runtime_spec, TensorTopology{});

    auto params = MakeRunParamsForMinimalSpec(node, {}, {});
    params.tensor_args = {
        ProgramRunArgs::TensorArgument{.tensor_parameter_name = "input_tensor", .tensor = std::cref(tensor)},
    };
    EXPECT_THAT(
        [&] { SetProgramRunArgs(program, params); },
        ::testing::ThrowsMessage<std::runtime_error>(
            ::testing::HasSubstr("padded_shape does not match the binding's declared padded_shape")));
}

TEST_F(ProgramRunArgsTestGen1, MatchPaddedShapeOnly_DTypeMismatchStillFails) {
    // match_padded_shape_only loosens only along logical_shape. tensor_layout fields (dtype here)
    // must still match exactly.
    NodeCoord node{0, 0};
    ProgramSpec spec = MakeMinimalGen1ValidProgramSpec();
    auto binding = MakeMinimalTensorParameter("input_tensor");  // BFLOAT16
    binding.advanced_options = TensorParameterAdvancedOptions{.match_padded_shape_only = true};
    spec.tensor_parameters = {binding};
    BindTensorParameterToKernel(spec.kernels[0], "input_tensor", "input_ta");

    Program program = MakeProgramFromSpec(*mesh_device_, spec);

    auto page_config = tt::tt_metal::PageConfig(tt::tt_metal::Layout::ROW_MAJOR);
    auto memory_config =
        tt::tt_metal::MemoryConfig{tt::tt_metal::TensorMemoryLayout::INTERLEAVED, tt::tt_metal::BufferType::DRAM};
    auto wrong_layout = tt::tt_metal::TensorLayout(tt::tt_metal::DataType::UINT32, page_config, memory_config);
    auto wrong_spec = tt::tt_metal::TensorSpec(binding.spec.logical_shape(), wrong_layout);
    MeshTensor tensor = MeshTensor::allocate_on_device(*mesh_device_, wrong_spec, TensorTopology{});

    auto params = MakeRunParamsForMinimalSpec(node, {}, {});
    params.tensor_args = {
        ProgramRunArgs::TensorArgument{.tensor_parameter_name = "input_tensor", .tensor = std::cref(tensor)},
    };
    EXPECT_THAT(
        [&] { SetProgramRunArgs(program, params); },
        ::testing::ThrowsMessage<std::runtime_error>(
            ::testing::HasSubstr("tensor_layout does not match the binding's declared layout")));
}

TEST_F(ProgramRunArgsTestGen1, TensorBindingOnlyKernelMissingFromRunParamsFails) {
    // A kernel with tensor bindings but an empty RTA/CRTA schema must still appear in
    // kernel_run_args: SetProgramRunArgs fills the binding section CRTAs (base
    // addresses, dynamic accessor fields) from the per-kernel entries, so omitting the
    // kernel would leave its binding CRTAs uninitialized.
    NodeCoord node{0, 0};
    ProgramSpec spec = MakeMinimalGen1ValidProgramSpec();
    auto binding = MakeMinimalTensorParameter("input_tensor");
    spec.tensor_parameters = {binding};
    // Bind to dm_kernel (compute_kernel has no bindings). dm_kernel has no named RTAs/CRTAs
    // and no varargs, so its RTA/CRTA schema is empty — the binding is the only thing forcing
    // a kernel_run_args entry.
    BindTensorParameterToKernel(spec.kernels[0], "input_tensor", "input_ta");

    Program program = MakeProgramFromSpec(*mesh_device_, spec);

    // Omit dm_kernel from kernel_run_args (only supply compute_kernel). Provide a tensor_arg
    // so that ValidateTensorArgs passes; the failure should come from the kernel-completeness
    // check on the binding-owning kernel.
    MeshTensor tensor = MeshTensor::allocate_on_device(*mesh_device_, binding.spec, TensorTopology{});
    ProgramRunArgs params;
    params.kernel_run_args.push_back(MakeKernelRunArgs("compute_kernel", node, {}, {}));
    params.tensor_args = {
        ProgramRunArgs::TensorArgument{.tensor_parameter_name = "input_tensor", .tensor = std::cref(tensor)},
    };

    EXPECT_THAT(
        [&] { SetProgramRunArgs(program, params); },
        ::testing::ThrowsMessage<std::runtime_error>(
            ::testing::HasSubstr("Kernel 'dm_kernel' is registered in the Program with a non-empty RTA/CRTA schema "
                                 "but has no runtime parameters specified in ProgramRunArgs")));
}

TEST_F(ProgramRunArgsTestGen1, MatchPaddedShapeOnly_DynamicWinsWhenBothSet) {
    // When both match_padded_shape_only and dynamic_tensor_shape are set, dynamic is more
    // permissive and wins. A runtime tensor whose padded_shape differs from declared should
    // be accepted (which match_padded_shape_only alone would reject).
    NodeCoord node{0, 0};
    ProgramSpec spec = MakeMinimalGen1ValidProgramSpec();
    auto binding = MakeMinimalTensorParameter("input_tensor");  // shape {1, 32}
    binding.advanced_options =
        TensorParameterAdvancedOptions{.match_padded_shape_only = true, .dynamic_tensor_shape = true};
    spec.tensor_parameters = {binding};
    BindTensorParameterToKernel(spec.kernels[0], "input_tensor", "input_ta");

    Program program = MakeProgramFromSpec(*mesh_device_, spec);

    // Different logical shape; for ROW_MAJOR this also gives a different padded_shape, which
    // dynamic accepts but padded_only alone would not.
    auto wrong_spec = tt::tt_metal::TensorSpec(tt::tt_metal::Shape{1, 64}, binding.spec.tensor_layout());
    MeshTensor tensor = MeshTensor::allocate_on_device(*mesh_device_, wrong_spec, TensorTopology{});

    auto params = MakeRunParamsForMinimalSpec(node, {}, {});
    params.tensor_args = {
        ProgramRunArgs::TensorArgument{.tensor_parameter_name = "input_tensor", .tensor = std::cref(tensor)},
    };
    EXPECT_NO_THROW(SetProgramRunArgs(program, params));
}

}  // namespace
}  // namespace tt::tt_metal::experimental::metal2_host_api
