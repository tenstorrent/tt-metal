// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Unit tests for the Metal 2.0 Host API: ProgramRunParams and SetProgramRunParameters
//
// Test categories:
//   1. Validation tests (parameter validation errors)
//   2. Success tests (basic functionality)
//   3. Repeated call tests (calling SetProgramRunParameters multiple times)
//
// NOTE: The current tests only attempt to set ProgramRunParams for a Program that has not yet been enqueued for
// execution. The way Program currently works, dispatch data structures are not created until the first enqueue, which
// makes the argument update pathways different the first time vs. subsequent times. This is patently insane. I plan to
// fix this for ProgramSpec / Metal 2.0.

#include <gtest/gtest.h>

#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_params.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/experimental/context/metal_env.hpp>
#include <tt-metalium/experimental/mock_device.hpp>

#include "impl/metal2_host_api/test_utils.hpp"
#include "test_helpers.hpp"

namespace tt::tt_metal::experimental::metal2_host_api {
namespace {

// Import shared test helpers
using test_helpers::BindDFBToKernel;
using test_helpers::MakeMinimalComputeKernel;
using test_helpers::MakeMinimalDFB;
using test_helpers::MakeMinimalDMKernel;
using test_helpers::MakeMinimalValidProgramSpec;
using test_helpers::MakeMinimalWorker;

// ============================================================================
// Test Fixtures
// ============================================================================

// Test fixture for ProgramRunParams on Quasar - uses Quasar mock device
class ProgramRunParamsTestQuasar : public ::testing::Test {
protected:
    void SetUp() override {
        GTEST_SKIP() << "Re-enable tests after Quasar mock device support is checked in";
        // Configure global mock mode for Quasar
        experimental::configure_mock_mode(tt::ARCH::QUASAR, 1);
    }
    void TearDown() override { experimental::disable_mock_mode(); }
};

// ============================================================================
// Test Utilities
// ============================================================================

// Create a ProgramSpec with specified RTA schema for the DM kernel
// (The compute kernel has no RTAs)
inline ProgramSpec MakeSpecWithRTAs(const NodeCoord& node, size_t num_per_node_rtas, size_t num_common_rtas) {
    ProgramSpec spec = MakeMinimalValidProgramSpec();

    // Set the RTA schema on the dm_kernel (first kernel)
    spec.kernels[0].runtime_arguments_schema.num_runtime_args_per_node = {{node, num_per_node_rtas}};
    spec.kernels[0].runtime_arguments_schema.num_common_runtime_args = num_common_rtas;

    // compute_kernel has no RTAs
    spec.kernels[1].runtime_arguments_schema.num_runtime_args_per_node = {{node, 0}};
    spec.kernels[1].runtime_arguments_schema.num_common_runtime_args = 0;

    return spec;
}

// Create a ProgramSpec with RTA schemas for DM and compute kernels
inline ProgramSpec MakeSpecWithBothKernelRTAs(
    const NodeCoord& node,
    size_t dm_per_node_rtas,
    size_t dm_common_rtas,
    size_t compute_per_node_rtas,
    size_t compute_common_rtas) {
    ProgramSpec spec = MakeMinimalValidProgramSpec();

    // dm_kernel RTAs
    spec.kernels[0].runtime_arguments_schema.num_runtime_args_per_node = {{node, dm_per_node_rtas}};
    spec.kernels[0].runtime_arguments_schema.num_common_runtime_args = dm_common_rtas;

    // compute_kernel RTAs
    spec.kernels[1].runtime_arguments_schema.num_runtime_args_per_node = {{node, compute_per_node_rtas}};
    spec.kernels[1].runtime_arguments_schema.num_common_runtime_args = compute_common_rtas;

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
        .runtime_args = {{node, per_node_args}},
        .common_runtime_args = common_args,
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
        .runtime_args = {},
        .common_runtime_args = {},
    });

    EXPECT_ANY_THROW(SetProgramRunParameters(program, params));
}

TEST_F(ProgramRunParamsTestQuasar, InvalidNodeForKernelFails) {
    NodeCoord node{0, 0};
    NodeCoord wrong_node{1, 1};  // Kernel doesn't run on this node
    ProgramSpec spec = MakeSpecWithRTAs(node, 2, 0);
    Program program = MakeProgramFromSpec(spec);

    ProgramRunParams params;
    params.kernel_run_params.push_back({
        .kernel_spec_name = "dm_kernel",
        .runtime_args = {{wrong_node, {1, 2}}},  // Wrong node!
        .common_runtime_args = {},
    });
    params.kernel_run_params.push_back(MakeKernelRunParams("compute_kernel", node, {}, {}));

    EXPECT_ANY_THROW(SetProgramRunParameters(program, params));
}

TEST_F(ProgramRunParamsTestQuasar, WrongRuntimeArgsCountFails) {
    NodeCoord node{0, 0};
    ProgramSpec spec = MakeSpecWithRTAs(node, /*per_node=*/3, /*common=*/0);
    Program program = MakeProgramFromSpec(spec);

    // Provide wrong count (2 instead of 3)
    auto params = MakeRunParamsForMinimalSpec(node, {1, 2}, {});

    EXPECT_ANY_THROW(SetProgramRunParameters(program, params));
}

TEST_F(ProgramRunParamsTestQuasar, WrongCommonRuntimeArgsCountFails) {
    NodeCoord node{0, 0};
    ProgramSpec spec = MakeSpecWithRTAs(node, /*per_node=*/0, /*common=*/2);
    Program program = MakeProgramFromSpec(spec);

    // Provide wrong common args count (3 instead of 2)
    auto params = MakeRunParamsForMinimalSpec(node, {}, {1, 2, 3});

    EXPECT_ANY_THROW(SetProgramRunParameters(program, params));
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

    EXPECT_ANY_THROW(SetProgramRunParameters(program, params));
}

TEST_F(ProgramRunParamsTestQuasar, MissingNodeRTAsFails) {
    NodeCoord node{0, 0};
    ProgramSpec spec = MakeSpecWithRTAs(node, /*per_node=*/2, /*common=*/0);
    Program program = MakeProgramFromSpec(spec);

    // Don't provide the per-node RTAs (empty runtime_args)
    ProgramRunParams params;
    params.kernel_run_params.push_back({
        .kernel_spec_name = "dm_kernel",
        .runtime_args = {},  // Missing node RTAs!
        .common_runtime_args = {},
    });
    params.kernel_run_params.push_back(MakeKernelRunParams("compute_kernel", node, {}, {}));

    EXPECT_ANY_THROW(SetProgramRunParameters(program, params));
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

    EXPECT_ANY_THROW(SetProgramRunParameters(program, params));
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

    EXPECT_ANY_THROW(SetProgramRunParameters(program, params));
}

// ============================================================================
// SECTION 2: Success Tests (basic functionality)
// ============================================================================

TEST_F(ProgramRunParamsTestQuasar, SetRunParamsSucceeds_ZeroRTAs) {
    NodeCoord node{0, 0};
    ProgramSpec spec = MakeSpecWithRTAs(node, /*per_node=*/0, /*common=*/0);
    Program program = MakeProgramFromSpec(spec);

    auto params = MakeRunParamsForMinimalSpec(node, {}, {});

    EXPECT_NO_THROW(SetProgramRunParameters(program, params));
}

TEST_F(ProgramRunParamsTestQuasar, SetRunParamsSucceeds_PerNodeRTAsOnly) {
    NodeCoord node{0, 0};
    ProgramSpec spec = MakeSpecWithRTAs(node, /*per_node=*/3, /*common=*/0);
    Program program = MakeProgramFromSpec(spec);

    auto params = MakeRunParamsForMinimalSpec(node, {100, 200, 300}, {});

    EXPECT_NO_THROW(SetProgramRunParameters(program, params));
}

TEST_F(ProgramRunParamsTestQuasar, SetRunParamsSucceeds_CommonRTAsOnly) {
    NodeCoord node{0, 0};
    ProgramSpec spec = MakeSpecWithRTAs(node, /*per_node=*/0, /*common=*/2);
    Program program = MakeProgramFromSpec(spec);

    auto params = MakeRunParamsForMinimalSpec(node, {}, {10, 20});

    EXPECT_NO_THROW(SetProgramRunParameters(program, params));
}

TEST_F(ProgramRunParamsTestQuasar, SetRunParamsSucceeds_BothRTATypes) {
    NodeCoord node{0, 0};
    ProgramSpec spec = MakeSpecWithRTAs(node, /*per_node=*/3, /*common=*/2);
    Program program = MakeProgramFromSpec(spec);

    auto params = MakeRunParamsForMinimalSpec(node, {100, 200, 300}, {10, 20});

    EXPECT_NO_THROW(SetProgramRunParameters(program, params));
}

TEST_F(ProgramRunParamsTestQuasar, SetRunParamsSucceeds_BothKernelsWithRTAs) {
    NodeCoord node{0, 0};
    ProgramSpec spec = MakeSpecWithBothKernelRTAs(
        node,
        /*dm_per_node=*/2,
        /*dm_common=*/1,
        /*compute_per_node=*/3,
        /*compute_common=*/2);
    Program program = MakeProgramFromSpec(spec);

    auto params = MakeRunParamsForMinimalSpec(
        node,
        /*dm_per_node=*/{1, 2},
        /*dm_common=*/{10},
        /*compute_per_node=*/{100, 200, 300},
        /*compute_common=*/{50, 60});

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
    ProgramSpec spec = MakeSpecWithRTAs(node, /*per_node=*/2, /*common=*/1);
    Program program = MakeProgramFromSpec(spec);

    auto params = MakeRunParamsForMinimalSpec(node, {100, 200}, {10});

    // First call
    EXPECT_NO_THROW(SetProgramRunParameters(program, params));
    // Second call with same values
    EXPECT_NO_THROW(SetProgramRunParameters(program, params));
}

TEST_F(ProgramRunParamsTestQuasar, SetRunParamsTwice_DifferentValuesSucceeds) {
    NodeCoord node{0, 0};
    ProgramSpec spec = MakeSpecWithRTAs(node, /*per_node=*/2, /*common=*/1);
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
    ProgramSpec spec = MakeSpecWithRTAs(node, /*per_node=*/0, /*common=*/2);
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
    EXPECT_ANY_THROW(SetProgramRunParameters(program, params2));
}

TEST_F(ProgramRunParamsTestQuasar, SetRunParamsMultipleTimes_Succeeds) {
    NodeCoord node{0, 0};
    ProgramSpec spec = MakeSpecWithRTAs(node, /*per_node=*/2, /*common=*/2);
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
    auto producer = MakeMinimalDMKernel("producer", all_nodes);
    auto consumer = MakeMinimalDMKernel("consumer", all_nodes);

    // Set RTA schema for both nodes
    producer.runtime_arguments_schema.num_runtime_args_per_node = {{node0, 2}, {node1, 2}};
    producer.runtime_arguments_schema.num_common_runtime_args = 1;

    consumer.runtime_arguments_schema.num_runtime_args_per_node = {{node0, 0}, {node1, 0}};
    consumer.runtime_arguments_schema.num_common_runtime_args = 0;

    // DFB is per-node (current limitation)
    auto dfb0 = MakeMinimalDFB("dfb0", node0);
    auto dfb1 = MakeMinimalDFB("dfb1", node1);

    BindDFBToKernel(producer, "dfb0", "out0", KernelSpec::DFBEndpointType::PRODUCER);
    BindDFBToKernel(producer, "dfb1", "out1", KernelSpec::DFBEndpointType::PRODUCER);
    BindDFBToKernel(consumer, "dfb0", "in0", KernelSpec::DFBEndpointType::CONSUMER);
    BindDFBToKernel(consumer, "dfb1", "in1", KernelSpec::DFBEndpointType::CONSUMER);

    spec.kernels = {producer, consumer};
    spec.dataflow_buffers = {dfb0, dfb1};
    spec.workers = std::vector<WorkerSpec>{
        MakeMinimalWorker("worker0", node0, {"producer", "consumer"}, {"dfb0"}),
        MakeMinimalWorker("worker1", node1, {"producer", "consumer"}, {"dfb1"})};

    Program program = MakeProgramFromSpec(spec);

    // Create run params with per-node RTAs for both nodes
    ProgramRunParams params;
    params.kernel_run_params.push_back({
        .kernel_spec_name = "producer",
        .runtime_args = {{node0, {10, 20}}, {node1, {30, 40}}},
        .common_runtime_args = {100},
    });
    params.kernel_run_params.push_back({
        .kernel_spec_name = "consumer",
        .runtime_args = {{node0, {}}, {node1, {}}},
        .common_runtime_args = {},
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

    auto producer = MakeMinimalDMKernel("producer", all_nodes);
    auto consumer = MakeMinimalDMKernel("consumer", all_nodes);

    // Set RTA schema for both nodes
    producer.runtime_arguments_schema.num_runtime_args_per_node = {{node0, 2}, {node1, 2}};
    producer.runtime_arguments_schema.num_common_runtime_args = 0;

    consumer.runtime_arguments_schema.num_runtime_args_per_node = {{node0, 0}, {node1, 0}};
    consumer.runtime_arguments_schema.num_common_runtime_args = 0;

    auto dfb0 = MakeMinimalDFB("dfb0", node0);
    auto dfb1 = MakeMinimalDFB("dfb1", node1);

    BindDFBToKernel(producer, "dfb0", "out0", KernelSpec::DFBEndpointType::PRODUCER);
    BindDFBToKernel(producer, "dfb1", "out1", KernelSpec::DFBEndpointType::PRODUCER);
    BindDFBToKernel(consumer, "dfb0", "in0", KernelSpec::DFBEndpointType::CONSUMER);
    BindDFBToKernel(consumer, "dfb1", "in1", KernelSpec::DFBEndpointType::CONSUMER);

    spec.kernels = {producer, consumer};
    spec.dataflow_buffers = {dfb0, dfb1};
    spec.workers = std::vector<WorkerSpec>{
        MakeMinimalWorker("worker0", node0, {"producer", "consumer"}, {"dfb0"}),
        MakeMinimalWorker("worker1", node1, {"producer", "consumer"}, {"dfb1"})};

    Program program = MakeProgramFromSpec(spec);

    // Only provide RTAs for node0, missing node1
    ProgramRunParams params;
    params.kernel_run_params.push_back({
        .kernel_spec_name = "producer",
        .runtime_args = {{node0, {10, 20}}},  // Missing node1!
        .common_runtime_args = {},
    });
    params.kernel_run_params.push_back({
        .kernel_spec_name = "consumer",
        .runtime_args = {{node0, {}}, {node1, {}}},
        .common_runtime_args = {},
    });

    EXPECT_ANY_THROW(SetProgramRunParameters(program, params));
}

}  // namespace
}  // namespace tt::tt_metal::experimental::metal2_host_api
