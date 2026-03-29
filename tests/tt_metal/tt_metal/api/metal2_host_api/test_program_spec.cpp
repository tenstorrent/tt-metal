// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Unit tests for the Metal 2.0 Host API: ProgramSpec and MakeProgramFromSpec
//
// Test categories:
//   1. ProgramSpec "structural" validation (CollectSpecData)
//   2. ProgramSpec semantic validation (ValidateProgramSpec)
//   3. WorkerSpec validation
//   4. DFB validation
//   5. Program creation (using Quasar mock device)

#include <gtest/gtest.h>

#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/experimental/context/metal_env.hpp>
#include <tt-metalium/experimental/mock_device.hpp>

#include "impl/metal2_host_api/test_utils.hpp"

namespace tt::tt_metal::experimental::metal2_host_api {
namespace {

// ============================================================================
// Test Utilities
// ============================================================================

// Minimal valid kernel source code for testing
constexpr const char* MINIMAL_DM_KERNEL_SOURCE = "void kernel_main() {}";
constexpr const char* MINIMAL_COMPUTE_KERNEL_SOURCE = "void kernel_main() {}";

// These helper utilities are used to create minimal valid Spec objects for testing.
// This enables very concise tests that alter just one thing about the minimal spec.
// These utilities (or their equivalents) are not meant to be used in production code.

// Helper to create a minimal valid KernelSpec for data movement
KernelSpec MakeMinimalDMKernel(
    const std::string& name, const std::variant<NodeCoord, NodeRange, NodeRangeSet>& nodes, uint8_t num_threads = 1) {
    return KernelSpec{
        .unique_id = name,
        .source = MINIMAL_DM_KERNEL_SOURCE,
        .source_type = KernelSpec::SourceType::SOURCE_CODE,
        .target_nodes = nodes,
        .num_threads = num_threads,
        // Fields with defaults skipped: thread_node_map, compiler_options, dfb_bindings,
        // semaphore_bindings, compile_time_arg_bindings, runtime_arguments_schema
        .config_spec =
            DataMovementConfiguration{
                .gen2_data_movement_config = DataMovementConfiguration::Gen2DataMovementConfig{},
            },
    };
}

// Helper to create a minimal valid KernelSpec for compute
KernelSpec MakeMinimalComputeKernel(
    const std::string& name, const std::variant<NodeCoord, NodeRange, NodeRangeSet>& nodes, uint8_t num_threads = 1) {
    return KernelSpec{
        .unique_id = name,
        .source = MINIMAL_COMPUTE_KERNEL_SOURCE,
        .source_type = KernelSpec::SourceType::SOURCE_CODE,
        .target_nodes = nodes,
        .num_threads = num_threads,
        // Fields with defaults skipped: thread_node_map, compiler_options, dfb_bindings,
        // semaphore_bindings, compile_time_arg_bindings, runtime_arguments_schema
        .config_spec = ComputeConfiguration{},
    };
}

// Helper to create a minimal valid DataflowBufferSpec
DataflowBufferSpec MakeMinimalDFB(
    const std::string& name,
    const std::variant<NodeCoord, NodeRange, NodeRangeSet>& nodes,
    uint32_t entry_size = 1024,
    uint32_t num_entries = 2) {
    return DataflowBufferSpec{
        .unique_id = name,
        .target_nodes = nodes,
        .entry_size = entry_size,
        .num_entries = num_entries,
    };
}

// Helper to create a minimal valid WorkerSpec
WorkerSpec MakeMinimalWorker(
    const std::string& name,
    const std::variant<NodeCoord, NodeRange, NodeRangeSet>& nodes,
    const std::vector<KernelSpecName>& kernels,
    const std::vector<DFBSpecName>& dfbs = {}) {
    return WorkerSpec{
        .unique_id = name,
        .kernels = kernels,
        .dataflow_buffers = dfbs,
        .target_nodes = nodes,
    };
}

// Helper to bind a DFB to a kernel as producer or consumer
void BindDFBToKernel(
    KernelSpec& kernel,
    const std::string& dfb_name,
    const std::string& accessor_name,
    KernelSpec::DFBEndpointType endpoint_type,
    DFBAccessPattern access_pattern = DFBAccessPattern::STRIDED) {
    kernel.dfb_bindings.push_back(KernelSpec::DFBBinding{
        .dfb_spec_name = dfb_name,
        .local_accessor_name = accessor_name,
        .endpoint_type = endpoint_type,
        .access_pattern = access_pattern,
    });
}

// Helper to create a minimal valid ProgramSpec with one kernel and one DFB
ProgramSpec MakeMinimalValidProgramSpec() {
    NodeCoord node{0, 0};

    ProgramSpec spec;
    spec.program_id = "test_program";

    // Create a DM kernel (producer) and compute kernel (consumer)
    auto dm_kernel = MakeMinimalDMKernel("dm_kernel", node);
    auto compute_kernel = MakeMinimalComputeKernel("compute_kernel", node);

    // Create a DFB with data format (required for compute endpoint)
    auto dfb = MakeMinimalDFB("dfb_0", node);
    dfb.data_format_metadata = tt::DataFormat::Float16_b;

    // Bind the DFB
    BindDFBToKernel(dm_kernel, "dfb_0", "input_dfb", KernelSpec::DFBEndpointType::PRODUCER);
    BindDFBToKernel(compute_kernel, "dfb_0", "input_dfb", KernelSpec::DFBEndpointType::CONSUMER);

    spec.kernels = {dm_kernel, compute_kernel};
    spec.dataflow_buffers = {dfb};

    // Create a WorkerSpec
    spec.workers =
        std::vector<WorkerSpec>{MakeMinimalWorker("worker_0", node, {"dm_kernel", "compute_kernel"}, {"dfb_0"})};

    return spec;
}

// ============================================================================
// Test Fixtures
// ============================================================================

// Test fixture for ProgramSpec on Quasar - uses Quasar mock device
class ProgramSpecTestQuasar : public ::testing::Test {
protected:
    void SetUp() override {
        GTEST_SKIP() << "Re-enable tests after Quasar mock device support is checked in";
        // Configure global mock mode for Quasar
        // This way, the HAL is initialized for arch check and Program creation.
        experimental::configure_mock_mode(tt::ARCH::QUASAR, 1);
    }
    void TearDown() override { experimental::disable_mock_mode(); }
};

// ============================================================================
// SECTION 1: Structural Validation Tests (CollectSpecData)
// ============================================================================
// These test the structural integrity checks that happen during spec collection.

TEST_F(ProgramSpecTestQuasar, DuplicateKernelNameFails) {
    ProgramSpec spec = MakeMinimalValidProgramSpec();

    // Add a kernel with duplicate name
    auto duplicate_kernel = MakeMinimalDMKernel("dm_kernel", NodeCoord{1, 0});
    DataMovementConfiguration dm_config;
    dm_config.gen2_data_movement_config = DataMovementConfiguration::Gen2DataMovementConfig{};
    duplicate_kernel.config_spec = dm_config;
    spec.kernels.push_back(duplicate_kernel);

    EXPECT_ANY_THROW(MakeProgramFromSpec(spec));
}

TEST_F(ProgramSpecTestQuasar, DuplicateDFBNameFails) {
    ProgramSpec spec = MakeMinimalValidProgramSpec();

    // Add a DFB with duplicate name
    auto duplicate_dfb = MakeMinimalDFB("dfb_0", NodeCoord{1, 0});
    spec.dataflow_buffers.push_back(duplicate_dfb);

    EXPECT_ANY_THROW(MakeProgramFromSpec(spec));
}

TEST_F(ProgramSpecTestQuasar, DuplicateSemaphoreNameFails) {
    ProgramSpec spec = MakeMinimalValidProgramSpec();

    // Add two semaphores with the same name
    SemaphoreSpec sem1;
    sem1.unique_id = "sem_0";
    sem1.target_nodes = NodeCoord{0, 0};

    SemaphoreSpec sem2;
    sem2.unique_id = "sem_0";  // duplicate!
    sem2.target_nodes = NodeCoord{1, 0};

    spec.semaphores = {sem1, sem2};

    EXPECT_ANY_THROW(MakeProgramFromSpec(spec));
}

TEST_F(ProgramSpecTestQuasar, KernelReferencesUnknownDFBFails) {
    NodeCoord node{0, 0};

    ProgramSpec spec;
    spec.program_id = "test_program";

    auto kernel = MakeMinimalDMKernel("kernel", node);
    // Bind to a DFB that doesn't exist
    BindDFBToKernel(kernel, "nonexistent_dfb", "accessor", KernelSpec::DFBEndpointType::PRODUCER);

    spec.kernels = {kernel};
    spec.workers = std::vector<WorkerSpec>{MakeMinimalWorker("worker", node, {"kernel"})};

    EXPECT_ANY_THROW(MakeProgramFromSpec(spec));
}

TEST_F(ProgramSpecTestQuasar, DFBWithNoBindingsFails) {
    NodeCoord node{0, 0};

    ProgramSpec spec;
    spec.program_id = "test_program";

    // Create a kernel with no DFB bindings
    auto kernel = MakeMinimalDMKernel("kernel", node);
    spec.kernels = {kernel};

    // Create a DFB that is never bound
    auto orphan_dfb = MakeMinimalDFB("orphan_dfb", node);
    spec.dataflow_buffers = {orphan_dfb};

    spec.workers = std::vector<WorkerSpec>{MakeMinimalWorker("worker", node, {"kernel"}, {"orphan_dfb"})};

    EXPECT_ANY_THROW(MakeProgramFromSpec(spec));
}

TEST_F(ProgramSpecTestQuasar, DFBWithOnlyProducerFails) {
    NodeCoord node{0, 0};

    ProgramSpec spec;
    spec.program_id = "test_program";

    auto kernel = MakeMinimalDMKernel("kernel", node);
    auto dfb = MakeMinimalDFB("dfb", node);

    // Only bind as producer, no consumer
    BindDFBToKernel(kernel, "dfb", "accessor", KernelSpec::DFBEndpointType::PRODUCER);

    spec.kernels = {kernel};
    spec.dataflow_buffers = {dfb};
    spec.workers = std::vector<WorkerSpec>{MakeMinimalWorker("worker", node, {"kernel"}, {"dfb"})};

    EXPECT_ANY_THROW(MakeProgramFromSpec(spec));
}

TEST_F(ProgramSpecTestQuasar, DFBWithOnlyConsumerFails) {
    NodeCoord node{0, 0};

    ProgramSpec spec;
    spec.program_id = "test_program";

    auto kernel = MakeMinimalDMKernel("kernel", node);
    auto dfb = MakeMinimalDFB("dfb", node);

    // Only bind as consumer, no producer
    BindDFBToKernel(kernel, "dfb", "accessor", KernelSpec::DFBEndpointType::CONSUMER);

    spec.kernels = {kernel};
    spec.dataflow_buffers = {dfb};
    spec.workers = std::vector<WorkerSpec>{MakeMinimalWorker("worker", node, {"kernel"}, {"dfb"})};

    EXPECT_ANY_THROW(MakeProgramFromSpec(spec));
}

TEST_F(ProgramSpecTestQuasar, DFBWithMultipleProducersFails) {
    NodeCoord node{0, 0};

    ProgramSpec spec;
    spec.program_id = "test_program";

    auto producer1 = MakeMinimalDMKernel("producer1", node);
    auto producer2 = MakeMinimalDMKernel("producer2", node);
    auto consumer = MakeMinimalComputeKernel("consumer", node);

    auto dfb = MakeMinimalDFB("dfb", node);
    dfb.data_format_metadata = tt::DataFormat::Float16_b;

    // Two producers for same DFB
    BindDFBToKernel(producer1, "dfb", "out", KernelSpec::DFBEndpointType::PRODUCER);
    BindDFBToKernel(producer2, "dfb", "out", KernelSpec::DFBEndpointType::PRODUCER);
    BindDFBToKernel(consumer, "dfb", "in", KernelSpec::DFBEndpointType::CONSUMER);

    spec.kernels = {producer1, producer2, consumer};
    spec.dataflow_buffers = {dfb};
    spec.workers =
        std::vector<WorkerSpec>{MakeMinimalWorker("worker", node, {"producer1", "producer2", "consumer"}, {"dfb"})};

    EXPECT_ANY_THROW(MakeProgramFromSpec(spec));
}

TEST_F(ProgramSpecTestQuasar, DFBWithMultipleConsumersFails) {
    NodeCoord node{0, 0};

    ProgramSpec spec;
    spec.program_id = "test_program";

    auto producer = MakeMinimalDMKernel("producer", node);
    auto consumer1 = MakeMinimalComputeKernel("consumer1", node);
    auto consumer2 = MakeMinimalDMKernel("consumer2", node);

    auto dfb = MakeMinimalDFB("dfb", node);
    dfb.data_format_metadata = tt::DataFormat::Float16_b;

    // Two consumers for same DFB
    BindDFBToKernel(producer, "dfb", "out", KernelSpec::DFBEndpointType::PRODUCER);
    BindDFBToKernel(consumer1, "dfb", "in", KernelSpec::DFBEndpointType::CONSUMER);
    BindDFBToKernel(consumer2, "dfb", "in", KernelSpec::DFBEndpointType::CONSUMER);

    spec.kernels = {producer, consumer1, consumer2};
    spec.dataflow_buffers = {dfb};
    spec.workers =
        std::vector<WorkerSpec>{MakeMinimalWorker("worker", node, {"producer", "consumer1", "consumer2"}, {"dfb"})};

    EXPECT_ANY_THROW(MakeProgramFromSpec(spec));
}

// ============================================================================
// SECTION 2: Semantic Validation Tests (ValidateProgramSpec)
// ============================================================================

TEST_F(ProgramSpecTestQuasar, EmptyKernelsFails) {
    ProgramSpec spec;
    spec.program_id = "empty_program";
    spec.workers = std::vector<WorkerSpec>{};  // Empty workers too

    EXPECT_ANY_THROW(MakeProgramFromSpec(spec));
}

TEST_F(ProgramSpecTestQuasar, KernelWithZeroThreadsFails) {
    NodeCoord node{0, 0};

    ProgramSpec spec;
    spec.program_id = "test_program";

    auto kernel = MakeMinimalDMKernel("kernel", node, 0);  // 0 threads!
    spec.kernels = {kernel};
    spec.workers = std::vector<WorkerSpec>{MakeMinimalWorker("worker", node, {"kernel"})};

    EXPECT_ANY_THROW(MakeProgramFromSpec(spec));
}

TEST_F(ProgramSpecTestQuasar, DMKernelExceedingMaxThreadsFails) {
    NodeCoord node{0, 0};

    ProgramSpec spec;
    spec.program_id = "test_program";

    // Quasar has 8 DM cores per node (we reserve 2 for internal use)
    auto kernel = MakeMinimalDMKernel("kernel", node, 9);  // Too many threads!
    spec.kernels = {kernel};
    spec.workers = std::vector<WorkerSpec>{MakeMinimalWorker("worker", node, {"kernel"})};

    EXPECT_ANY_THROW(MakeProgramFromSpec(spec));
}

TEST_F(ProgramSpecTestQuasar, ComputeKernelExceedingMaxThreadsFails) {
    NodeCoord node{0, 0};

    ProgramSpec spec;
    spec.program_id = "test_program";

    // Quasar has 4 Tensix cores per node
    auto kernel = MakeMinimalComputeKernel("kernel", node, 5);  // Too many threads!
    spec.kernels = {kernel};
    spec.workers = std::vector<WorkerSpec>{MakeMinimalWorker("worker", node, {"kernel"})};

    EXPECT_ANY_THROW(MakeProgramFromSpec(spec));
}

// Remove once WH/BH is implemented
TEST_F(ProgramSpecTestQuasar, DMKernelWithoutGen2ConfigFails) {
    NodeCoord node{0, 0};

    ProgramSpec spec;
    spec.program_id = "test_program";

    auto kernel = MakeMinimalDMKernel("kernel", node);
    // Remove the Gen2 config
    auto& dm_config = std::get<DataMovementConfiguration>(kernel.config_spec);
    dm_config.gen2_data_movement_config = std::nullopt;

    spec.kernels = {kernel};
    spec.workers = std::vector<WorkerSpec>{MakeMinimalWorker("worker", node, {"kernel"})};

    EXPECT_ANY_THROW(MakeProgramFromSpec(spec));
}

TEST_F(ProgramSpecTestQuasar, DMKernelWithNoConfigAtAllFails) {
    NodeCoord node{0, 0};

    ProgramSpec spec;
    spec.program_id = "test_program";

    auto kernel = MakeMinimalDMKernel("kernel", node);
    // Remove both Gen1 and Gen2 configs
    auto& dm_config = std::get<DataMovementConfiguration>(kernel.config_spec);
    dm_config.gen1_data_movement_config = std::nullopt;
    dm_config.gen2_data_movement_config = std::nullopt;

    spec.kernels = {kernel};
    spec.workers = std::vector<WorkerSpec>{MakeMinimalWorker("worker", node, {"kernel"})};

    EXPECT_ANY_THROW(MakeProgramFromSpec(spec));
}

// Remove once implemented
TEST_F(ProgramSpecTestQuasar, RemoteDFBFails) {
    // Remote DFBs are not yet implemented
    NodeCoord node{0, 0};

    ProgramSpec spec;
    spec.program_id = "test_program";

    auto producer = MakeMinimalDMKernel("producer", node);
    auto consumer = MakeMinimalDMKernel("consumer", node);
    auto dfb = MakeMinimalDFB("dfb", node);
    dfb.is_remote_dfb = true;  // Not supported!

    BindDFBToKernel(producer, "dfb", "out", KernelSpec::DFBEndpointType::PRODUCER);
    BindDFBToKernel(consumer, "dfb", "in", KernelSpec::DFBEndpointType::CONSUMER);

    spec.kernels = {producer, consumer};
    spec.dataflow_buffers = {dfb};
    spec.workers = std::vector<WorkerSpec>{MakeMinimalWorker("worker", node, {"producer", "consumer"}, {"dfb"})};

    EXPECT_ANY_THROW(MakeProgramFromSpec(spec));
}

// Remove once implemented
TEST_F(ProgramSpecTestQuasar, BorrowedMemoryDFBFails) {
    // Borrowed memory DFBs are not yet implemented
    NodeCoord node{0, 0};

    ProgramSpec spec;
    spec.program_id = "test_program";

    auto producer = MakeMinimalDMKernel("producer", node);
    auto consumer = MakeMinimalDMKernel("consumer", node);
    auto dfb = MakeMinimalDFB("dfb", node);
    dfb.uses_borrowed_memory = true;  // Not supported!

    BindDFBToKernel(producer, "dfb", "out", KernelSpec::DFBEndpointType::PRODUCER);
    BindDFBToKernel(consumer, "dfb", "in", KernelSpec::DFBEndpointType::CONSUMER);

    spec.kernels = {producer, consumer};
    spec.dataflow_buffers = {dfb};
    spec.workers = std::vector<WorkerSpec>{MakeMinimalWorker("worker", node, {"producer", "consumer"}, {"dfb"})};

    EXPECT_ANY_THROW(MakeProgramFromSpec(spec));
}

// Remove once implemented
TEST_F(ProgramSpecTestQuasar, DFBAliasingFails) {
    // DFB aliasing is not yet implemented
    NodeCoord node{0, 0};

    ProgramSpec spec;
    spec.program_id = "test_program";

    auto producer = MakeMinimalDMKernel("producer", node);
    auto consumer = MakeMinimalDMKernel("consumer", node);
    auto dfb = MakeMinimalDFB("dfb", node);
    dfb.alias_with = std::vector<DFBSpecName>{"other_dfb"};  // Not supported!

    BindDFBToKernel(producer, "dfb", "out", KernelSpec::DFBEndpointType::PRODUCER);
    BindDFBToKernel(consumer, "dfb", "in", KernelSpec::DFBEndpointType::CONSUMER);

    spec.kernels = {producer, consumer};
    spec.dataflow_buffers = {dfb};
    spec.workers = std::vector<WorkerSpec>{MakeMinimalWorker("worker", node, {"producer", "consumer"}, {"dfb"})};

    EXPECT_ANY_THROW(MakeProgramFromSpec(spec));
}

// Remove once implemented
TEST_F(ProgramSpecTestQuasar, SemaphoresFail) {
    // Semaphores are not yet implemented for Quasar
    ProgramSpec spec = MakeMinimalValidProgramSpec();

    SemaphoreSpec sem;
    sem.unique_id = "sem_0";
    sem.target_nodes = NodeCoord{0, 0};
    spec.semaphores = {sem};

    EXPECT_ANY_THROW(MakeProgramFromSpec(spec));
}

// Remove once implemented
TEST_F(ProgramSpecTestQuasar, KernelSemaphoreBindingsFail) {
    // Semaphore bindings are not yet implemented
    ProgramSpec spec = MakeMinimalValidProgramSpec();

    KernelSpec::SemaphoreBinding binding;
    binding.semaphore_spec_name = "sem_0";
    binding.accessor_name = "my_sem";
    spec.kernels[0].semaphore_bindings = {binding};

    EXPECT_ANY_THROW(MakeProgramFromSpec(spec));
}

TEST_F(ProgramSpecTestQuasar, DFBWithComputeEndpointRequiresDataFormat) {
    NodeCoord node{0, 0};

    ProgramSpec spec;
    spec.program_id = "test_program";

    auto producer = MakeMinimalDMKernel("producer", node);
    auto consumer = MakeMinimalComputeKernel("consumer", node);  // Compute!
    auto dfb = MakeMinimalDFB("dfb", node);
    // dfb.data_format_metadata is NOT set (nullopt)

    BindDFBToKernel(producer, "dfb", "out", KernelSpec::DFBEndpointType::PRODUCER);
    BindDFBToKernel(consumer, "dfb", "in", KernelSpec::DFBEndpointType::CONSUMER);

    spec.kernels = {producer, consumer};
    spec.dataflow_buffers = {dfb};
    spec.workers = std::vector<WorkerSpec>{MakeMinimalWorker("worker", node, {"producer", "consumer"}, {"dfb"})};

    EXPECT_ANY_THROW(MakeProgramFromSpec(spec));
}

TEST_F(ProgramSpecTestQuasar, ComputeConfigUnpackToDestModeReferencesUnknownDFBFails) {
    NodeCoord node{0, 0};

    ProgramSpec spec;
    spec.program_id = "test_program";

    auto producer = MakeMinimalDMKernel("producer", node);
    auto consumer = MakeMinimalComputeKernel("consumer", node);

    // Set unpack_to_dest_mode referencing a DFB that doesn't exist
    auto& compute_config = std::get<ComputeConfiguration>(consumer.config_spec);
    compute_config.unpack_to_dest_mode = {{"nonexistent_dfb", UnpackToDestMode::UnpackToDestFp32}};

    auto dfb = MakeMinimalDFB("dfb", node);
    dfb.data_format_metadata = tt::DataFormat::Float16_b;

    BindDFBToKernel(producer, "dfb", "out", KernelSpec::DFBEndpointType::PRODUCER);
    BindDFBToKernel(consumer, "dfb", "in", KernelSpec::DFBEndpointType::CONSUMER);

    spec.kernels = {producer, consumer};
    spec.dataflow_buffers = {dfb};
    spec.workers = std::vector<WorkerSpec>{MakeMinimalWorker("worker", node, {"producer", "consumer"}, {"dfb"})};

    EXPECT_ANY_THROW(MakeProgramFromSpec(spec));
}

// ============================================================================
// SECTION 3: WorkerSpec Validation Tests
// ============================================================================

TEST_F(ProgramSpecTestQuasar, MissingWorkerSpecsFails) {
    // Gen2 requires WorkerSpecs
    NodeCoord node{0, 0};

    ProgramSpec spec;
    spec.program_id = "test_program";

    auto kernel = MakeMinimalDMKernel("kernel", node);
    spec.kernels = {kernel};
    // spec.workers is NOT set (nullopt)

    EXPECT_ANY_THROW(MakeProgramFromSpec(spec));
}

TEST_F(ProgramSpecTestQuasar, EmptyWorkerSpecsFails) {
    NodeCoord node{0, 0};

    ProgramSpec spec;
    spec.program_id = "test_program";

    auto kernel = MakeMinimalDMKernel("kernel", node);
    spec.kernels = {kernel};
    spec.workers = std::vector<WorkerSpec>{};  // Empty!

    EXPECT_ANY_THROW(MakeProgramFromSpec(spec));
}

TEST_F(ProgramSpecTestQuasar, WorkerSpecWithNoKernelsFails) {
    NodeCoord node{0, 0};

    ProgramSpec spec;
    spec.program_id = "test_program";

    auto kernel = MakeMinimalDMKernel("kernel", node);
    spec.kernels = {kernel};

    WorkerSpec worker;
    worker.unique_id = "worker";
    worker.target_nodes = node;
    worker.kernels = {};  // No kernels!
    spec.workers = std::vector<WorkerSpec>{worker};

    EXPECT_ANY_THROW(MakeProgramFromSpec(spec));
}

TEST_F(ProgramSpecTestQuasar, OverlappingWorkerSpecsFails) {
    // Two workers cannot target overlapping nodes
    NodeCoord node{0, 0};

    ProgramSpec spec;
    spec.program_id = "test_program";

    auto kernel1 = MakeMinimalDMKernel("kernel1", node);
    auto kernel2 = MakeMinimalDMKernel("kernel2", node);
    spec.kernels = {kernel1, kernel2};

    // Both workers target the same node
    spec.workers = std::vector<WorkerSpec>{
        MakeMinimalWorker("worker1", node, {"kernel1"}), MakeMinimalWorker("worker2", node, {"kernel2"})};

    EXPECT_ANY_THROW(MakeProgramFromSpec(spec));
}

TEST_F(ProgramSpecTestQuasar, KernelNotInAnyWorkerSpecFails) {
    NodeCoord node{0, 0};

    ProgramSpec spec;
    spec.program_id = "test_program";

    auto kernel1 = MakeMinimalDMKernel("kernel1", node);
    auto kernel2 = MakeMinimalDMKernel("kernel2", node);  // Not in any worker!
    spec.kernels = {kernel1, kernel2};

    spec.workers = std::vector<WorkerSpec>{MakeMinimalWorker("worker", node, {"kernel1"})};  // Only kernel1

    EXPECT_ANY_THROW(MakeProgramFromSpec(spec));
}

TEST_F(ProgramSpecTestQuasar, KernelTargetNodesMismatchWorkerNodesFails) {
    // Kernel target nodes must contain worker target nodes
    NodeCoord node0{0, 0};
    NodeCoord node1{1, 0};

    ProgramSpec spec;
    spec.program_id = "test_program";

    // Kernel only targets node0
    auto kernel = MakeMinimalDMKernel("kernel", node0);
    spec.kernels = {kernel};

    // But worker targets node1
    spec.workers = std::vector<WorkerSpec>{MakeMinimalWorker("worker", node1, {"kernel"})};

    EXPECT_ANY_THROW(MakeProgramFromSpec(spec));
}

TEST_F(ProgramSpecTestQuasar, WorkerExceedsDMCoreBudgetFails) {
    NodeCoord node{0, 0};

    ProgramSpec spec;
    spec.program_id = "test_program";

    // Create enough DM kernels to exceed the 8 DM core budget
    auto kernel1 = MakeMinimalDMKernel("dm1", node, 3);
    auto kernel2 = MakeMinimalDMKernel("dm2", node, 3);
    auto kernel3 = MakeMinimalDMKernel("dm3", node, 3);  // Total: 9 > 8

    spec.kernels = {kernel1, kernel2, kernel3};
    spec.workers = std::vector<WorkerSpec>{MakeMinimalWorker("worker", node, {"dm1", "dm2", "dm3"})};

    EXPECT_ANY_THROW(MakeProgramFromSpec(spec));
}

TEST_F(ProgramSpecTestQuasar, WorkerExceedsComputeCoreBudgetFails) {
    NodeCoord node{0, 0};

    ProgramSpec spec;
    spec.program_id = "test_program";

    // Create enough compute kernels to exceed the 4 Tensix core budget
    auto kernel1 = MakeMinimalComputeKernel("compute1", node, 2);
    auto kernel2 = MakeMinimalComputeKernel("compute2", node, 3);  // Total: 5 > 4

    spec.kernels = {kernel1, kernel2};
    spec.workers = std::vector<WorkerSpec>{MakeMinimalWorker("worker", node, {"compute1", "compute2"})};

    EXPECT_ANY_THROW(MakeProgramFromSpec(spec));
}

TEST_F(ProgramSpecTestQuasar, WorkerWithMultipleComputeKernelsFails) {
    // A worker can have at most one compute kernel
    NodeCoord node{0, 0};

    ProgramSpec spec;
    spec.program_id = "test_program";

    auto compute1 = MakeMinimalComputeKernel("compute1", node, 1);
    auto compute2 = MakeMinimalComputeKernel("compute2", node, 1);

    spec.kernels = {compute1, compute2};
    spec.workers = std::vector<WorkerSpec>{MakeMinimalWorker("worker", node, {"compute1", "compute2"})};

    EXPECT_ANY_THROW(MakeProgramFromSpec(spec));
}

TEST_F(ProgramSpecTestQuasar, DFBNotInAnyWorkerSpecFails) {
    NodeCoord node{0, 0};

    ProgramSpec spec;
    spec.program_id = "test_program";

    auto producer = MakeMinimalDMKernel("producer", node);
    auto consumer = MakeMinimalDMKernel("consumer", node);
    auto dfb = MakeMinimalDFB("dfb", node);

    BindDFBToKernel(producer, "dfb", "out", KernelSpec::DFBEndpointType::PRODUCER);
    BindDFBToKernel(consumer, "dfb", "in", KernelSpec::DFBEndpointType::CONSUMER);

    spec.kernels = {producer, consumer};
    spec.dataflow_buffers = {dfb};

    // Worker doesn't include the DFB in its dataflow_buffers list
    spec.workers = std::vector<WorkerSpec>{MakeMinimalWorker("worker", node, {"producer", "consumer"}, {})};

    EXPECT_ANY_THROW(MakeProgramFromSpec(spec));
}
// ============================================================================
// SECTION 4: Programs Creation Tests
// ============================================================================
// These verify that valid configurations succeed.
// NOTE: Program creation needs full HAL support.
// TODO: Enable these tests with a Quasar mock device.

TEST_F(ProgramSpecTestQuasar, MinimalValidProgramSpecSucceeds) {
    ProgramSpec spec = MakeMinimalValidProgramSpec();

    EXPECT_NO_THROW(MakeProgramFromSpec(spec));
}

TEST_F(ProgramSpecTestQuasar, DMOnlyProgramSucceeds) {
    // A program with only DM kernels (no compute)
    NodeCoord node{0, 0};

    ProgramSpec spec;
    spec.program_id = "dm_only_program";

    auto producer = MakeMinimalDMKernel("producer", node);
    auto consumer = MakeMinimalDMKernel("consumer", node);
    auto dfb = MakeMinimalDFB("dfb", node);
    // No data_format_metadata needed for DM-only DFBs

    BindDFBToKernel(producer, "dfb", "out", KernelSpec::DFBEndpointType::PRODUCER);
    BindDFBToKernel(consumer, "dfb", "in", KernelSpec::DFBEndpointType::CONSUMER);

    spec.kernels = {producer, consumer};
    spec.dataflow_buffers = {dfb};
    spec.workers = std::vector<WorkerSpec>{MakeMinimalWorker("worker", node, {"producer", "consumer"}, {"dfb"})};

    EXPECT_NO_THROW(MakeProgramFromSpec(spec));
}

TEST_F(ProgramSpecTestQuasar, MultiNodeProgramSucceeds) {
    // A program spanning multiple nodes
    NodeRange nodes{{0, 0}, {1, 1}};  // 2x2 grid

    ProgramSpec spec;
    spec.program_id = "multi_node_program";

    auto producer = MakeMinimalDMKernel("producer", nodes);
    auto consumer = MakeMinimalDMKernel("consumer", nodes);
    auto dfb = MakeMinimalDFB("dfb", nodes);

    BindDFBToKernel(producer, "dfb", "out", KernelSpec::DFBEndpointType::PRODUCER);
    BindDFBToKernel(consumer, "dfb", "in", KernelSpec::DFBEndpointType::CONSUMER);

    spec.kernels = {producer, consumer};
    spec.dataflow_buffers = {dfb};
    spec.workers = std::vector<WorkerSpec>{MakeMinimalWorker("worker", nodes, {"producer", "consumer"}, {"dfb"})};

    EXPECT_NO_THROW(MakeProgramFromSpec(spec));
}

TEST_F(ProgramSpecTestQuasar, MultipleWorkersOnDifferentNodesSucceeds) {
    // Multiple workers on non-overlapping nodes
    NodeCoord node0{0, 0};
    NodeCoord node1{1, 0};
    NodeRangeSet all_nodes(std::set<NodeRange>{NodeRange{node0, node0}, NodeRange{node1, node1}});

    ProgramSpec spec;
    spec.program_id = "multi_worker_program";

    // Kernels span both nodes
    auto kernel = MakeMinimalDMKernel("kernel", all_nodes);
    spec.kernels = {kernel};

    // Two workers, each on a different node
    spec.workers = std::vector<WorkerSpec>{
        MakeMinimalWorker("worker0", node0, {"kernel"}), MakeMinimalWorker("worker1", node1, {"kernel"})};

    EXPECT_NO_THROW(MakeProgramFromSpec(spec));
}

TEST_F(ProgramSpecTestQuasar, MaxDMThreadsSucceeds) {
    // Use exactly 6 DM threads (the maximum available to the user)
    NodeCoord node{0, 0};

    ProgramSpec spec;
    spec.program_id = "max_dm_threads";

    auto producer = MakeMinimalDMKernel("producer", node, 3);
    auto consumer = MakeMinimalDMKernel("consumer", node, 3);  // Total: 6
    auto dfb = MakeMinimalDFB("dfb", node);

    BindDFBToKernel(producer, "dfb", "out", KernelSpec::DFBEndpointType::PRODUCER);
    BindDFBToKernel(consumer, "dfb", "in", KernelSpec::DFBEndpointType::CONSUMER);

    spec.kernels = {producer, consumer};
    spec.dataflow_buffers = {dfb};
    spec.workers = std::vector<WorkerSpec>{MakeMinimalWorker("worker", node, {"producer", "consumer"}, {"dfb"})};

    EXPECT_NO_THROW(MakeProgramFromSpec(spec));
}

TEST_F(ProgramSpecTestQuasar, MaxComputeThreadsSucceeds) {
    // Use exactly 4 compute threads (the maximum available)
    NodeCoord node{0, 0};

    ProgramSpec spec;
    spec.program_id = "max_compute_threads";

    auto dm = MakeMinimalDMKernel("dm", node);
    auto compute = MakeMinimalComputeKernel("compute", node, 4);  // Max threads

    auto dfb = MakeMinimalDFB("dfb", node);
    dfb.data_format_metadata = tt::DataFormat::Float16_b;

    BindDFBToKernel(dm, "dfb", "out", KernelSpec::DFBEndpointType::PRODUCER);
    BindDFBToKernel(compute, "dfb", "in", KernelSpec::DFBEndpointType::CONSUMER);

    spec.kernels = {dm, compute};
    spec.dataflow_buffers = {dfb};
    spec.workers = std::vector<WorkerSpec>{MakeMinimalWorker("worker", node, {"dm", "compute"}, {"dfb"})};

    EXPECT_NO_THROW(MakeProgramFromSpec(spec));
}

TEST_F(ProgramSpecTestQuasar, MultipleDFBsSucceeds) {
    NodeCoord node{0, 0};

    ProgramSpec spec;
    spec.program_id = "multi_dfb_program";

    auto producer = MakeMinimalDMKernel("producer", node);
    auto consumer = MakeMinimalComputeKernel("consumer", node);

    auto dfb1 = MakeMinimalDFB("dfb1", node);
    dfb1.data_format_metadata = tt::DataFormat::Float16_b;
    auto dfb2 = MakeMinimalDFB("dfb2", node);
    dfb2.data_format_metadata = tt::DataFormat::Bfp8_b;

    BindDFBToKernel(producer, "dfb1", "out1", KernelSpec::DFBEndpointType::PRODUCER);
    BindDFBToKernel(producer, "dfb2", "out2", KernelSpec::DFBEndpointType::PRODUCER);
    BindDFBToKernel(consumer, "dfb1", "in1", KernelSpec::DFBEndpointType::CONSUMER);
    BindDFBToKernel(consumer, "dfb2", "in2", KernelSpec::DFBEndpointType::CONSUMER);

    spec.kernels = {producer, consumer};
    spec.dataflow_buffers = {dfb1, dfb2};
    spec.workers =
        std::vector<WorkerSpec>{MakeMinimalWorker("worker", node, {"producer", "consumer"}, {"dfb1", "dfb2"})};

    EXPECT_NO_THROW(MakeProgramFromSpec(spec));
}

TEST_F(ProgramSpecTestQuasar, CompilerOptionsDefinesSucceeds) {
    ProgramSpec spec = MakeMinimalValidProgramSpec();

    // Add some defines
    spec.kernels[0].compiler_options.defines = {{"MY_DEFINE", "42"}, {"ANOTHER_DEFINE", "foo"}};

    EXPECT_NO_THROW(MakeProgramFromSpec(spec));
}

TEST_F(ProgramSpecTestQuasar, CompileTimeArgBindingsSucceeds) {
    ProgramSpec spec = MakeMinimalValidProgramSpec();

    // Add compile-time arg bindings
    spec.kernels[0].compile_time_arg_bindings = {{"arg1", 100}, {"arg2", 200}};

    EXPECT_NO_THROW(MakeProgramFromSpec(spec));
}

TEST_F(ProgramSpecTestQuasar, RuntimeArgsSchemaSucceeds) {
    ProgramSpec spec = MakeMinimalValidProgramSpec();

    // Add runtime args schema
    NodeCoord node{0, 0};
    spec.kernels[0].runtime_arguments_schema.num_runtime_args_per_node = {{node, 3}};
    spec.kernels[0].runtime_arguments_schema.num_common_runtime_args = 2;

    EXPECT_NO_THROW(MakeProgramFromSpec(spec));
}

// ============================================================================
// SECTION 5: Edge Cases and Boundary Tests
// ============================================================================
// NOTE: These also require Quasar hardware for Program creation.
// TODO: Enable these tests with a Quasar mock device.

TEST_F(ProgramSpecTestQuasar, NodeRangeSetTargetNodesSucceeds) {
    // Test with NodeRangeSet (multiple disjoint ranges)
    NodeRangeSet nodes(std::set<NodeRange>{NodeRange{{0, 0}, {0, 1}}, NodeRange{{2, 0}, {2, 1}}});

    ProgramSpec spec;
    spec.program_id = "range_set_program";

    auto producer = MakeMinimalDMKernel("producer", nodes);
    auto consumer = MakeMinimalDMKernel("consumer", nodes);
    auto dfb = MakeMinimalDFB("dfb", nodes);

    BindDFBToKernel(producer, "dfb", "out", KernelSpec::DFBEndpointType::PRODUCER);
    BindDFBToKernel(consumer, "dfb", "in", KernelSpec::DFBEndpointType::CONSUMER);

    spec.kernels = {producer, consumer};
    spec.dataflow_buffers = {dfb};
    spec.workers = std::vector<WorkerSpec>{MakeMinimalWorker("worker", nodes, {"producer", "consumer"}, {"dfb"})};

    EXPECT_NO_THROW(MakeProgramFromSpec(spec));
}

TEST_F(ProgramSpecTestQuasar, SourceCodeKernelSucceeds) {
    ProgramSpec spec = MakeMinimalValidProgramSpec();

    // Change to inline source code
    spec.kernels[0].source = "void kernel_main() {}";
    spec.kernels[0].source_type = KernelSpec::SourceType::SOURCE_CODE;

    EXPECT_NO_THROW(MakeProgramFromSpec(spec));
}

TEST_F(ProgramSpecTestQuasar, ComputeConfigMathFidelitySucceeds) {
    ProgramSpec spec = MakeMinimalValidProgramSpec();

    // Find the compute kernel and set math fidelity options
    for (auto& kernel : spec.kernels) {
        if (kernel.is_compute_kernel()) {
            auto& config = std::get<ComputeConfiguration>(kernel.config_spec);
            config.math_fidelity = MathFidelity::LoFi;
            config.fp32_dest_acc_en = true;
            config.math_approx_mode = true;
        }
    }

    EXPECT_NO_THROW(MakeProgramFromSpec(spec));
}

TEST_F(ProgramSpecTestQuasar, ValidUnpackToDestModeSucceeds) {
    ProgramSpec spec = MakeMinimalValidProgramSpec();

    // Set valid unpack_to_dest_mode (referencing an existing DFB)
    for (auto& kernel : spec.kernels) {
        if (kernel.is_compute_kernel()) {
            auto& config = std::get<ComputeConfiguration>(kernel.config_spec);
            config.unpack_to_dest_mode = {{"dfb_0", UnpackToDestMode::UnpackToDestFp32}};
        }
    }

    EXPECT_NO_THROW(MakeProgramFromSpec(spec));
}

// ============================================================================
// SECTION 6: Processor Assignment Edge Cases
// ============================================================================
// These tests document cases that expose limitations in our current processor
// assignment implementation. There are two categories:
//
// A) GREEDY ALGORITHM FAILURES
//    These are solvable with the simplifying assumption intact, but our naive
//    greedy algorithm fails. A smarter algorithm (constraint propagation, or
//    "assign multi-node kernels first") would handle these.
//    TODO: Fix the algorithm to handle these cases.
//
// B) TRUE SIMPLIFYING ASSUMPTION VIOLATIONS
//    These are fundamentally unsolvable while maintaining the assumption that
//    "a kernel uses the same processor indices on all nodes it runs on."
//    When we encounter these in production, the assumption must be removed.
//
// Our plan is to keep the simplifying assumption for now.
// But, with Op development on Quasar soon underway, we need a clear message
// if the assumption is ever violated.

// Category A: Greedy Algorithm Failure
// This test IS solvable with the simplifying assumption - a smarter algorithm would work.
// TODO: Once this test passes, remove this TODO. The algorithm has been fixed!
TEST_F(ProgramSpecTestQuasar, GreedyAlgorithmFailure_OrderDependentAssignment) {
    // This test constructs a valid ProgramSpec that SHOULD work, but MAY fail
    // with our naive greedy processor assignment algorithm.
    //
    // Scenario:
    //   - K_shared: A DM kernel spanning nodes (0,0) and (1,0), using 2 threads
    //   - K_node0_only: A DM kernel on node (0,0) only, using 4 threads
    //   - K_node1_only: A DM kernel on node (1,0) only, using 4 threads
    //   - Two WorkerSpecs, one per node
    //
    // SOLUTION EXISTS:
    //   K_shared gets processors 6-7 on both nodes
    //   K_node0_only gets processors 2-5 on node (0,0)
    //   K_node1_only gets processors 2-5 on node (1,0)
    //
    // A smarter algorithm (e.g., assign multi-node kernels first, or use
    // constraint propagation) would find this solution.

    NodeCoord node0{0, 0};
    NodeCoord node1{1, 0};
    NodeRangeSet both_nodes(std::set<NodeRange>{NodeRange{node0, node0}, NodeRange{node1, node1}});

    ProgramSpec spec;
    spec.program_id = "simplifying_assumption_test";

    // K_shared spans both nodes, needs 2 DM threads
    auto k_shared = MakeMinimalDMKernel("k_shared", both_nodes, 2);

    // K_node0_only runs only on node (0,0), needs 6 DM threads
    // This will "claim" processors 0-5 on node (0,0)
    auto k_node0_only = MakeMinimalDMKernel("k_node0_only", node0, 4);

    // K_node1_only runs only on node (1,0), needs 6 DM threads
    // This will "claim" processors 0-5 on node (1,0)
    auto k_node1_only = MakeMinimalDMKernel("k_node1_only", node1, 4);

    // DFB for the shared kernel (needs producer + consumer)
    auto dfb_shared = MakeMinimalDFB("dfb_shared", both_nodes);
    BindDFBToKernel(k_shared, "dfb_shared", "shared_out", KernelSpec::DFBEndpointType::PRODUCER);

    // Need a consumer for dfb_shared - let's add a simple consumer kernel
    auto k_consumer = MakeMinimalDMKernel("k_consumer", both_nodes, 1);
    BindDFBToKernel(k_consumer, "dfb_shared", "shared_in", KernelSpec::DFBEndpointType::CONSUMER);

    // DFBs for the node-specific kernels
    auto dfb_node0 = MakeMinimalDFB("dfb_node0", node0);
    BindDFBToKernel(k_node0_only, "dfb_node0", "out", KernelSpec::DFBEndpointType::PRODUCER);
    // Need consumer - reuse k_shared as consumer on node0
    auto k_node0_consumer = MakeMinimalDMKernel("k_node0_consumer", node0, 1);
    BindDFBToKernel(k_node0_consumer, "dfb_node0", "in", KernelSpec::DFBEndpointType::CONSUMER);

    auto dfb_node1 = MakeMinimalDFB("dfb_node1", node1);
    BindDFBToKernel(k_node1_only, "dfb_node1", "out", KernelSpec::DFBEndpointType::PRODUCER);
    auto k_node1_consumer = MakeMinimalDMKernel("k_node1_consumer", node1, 1);
    BindDFBToKernel(k_node1_consumer, "dfb_node1", "in", KernelSpec::DFBEndpointType::CONSUMER);

    spec.kernels = {k_shared, k_consumer, k_node0_only, k_node0_consumer, k_node1_only, k_node1_consumer};
    spec.dataflow_buffers = {dfb_shared, dfb_node0, dfb_node1};

    // Two WorkerSpecs, one per node
    // Worker 0 handles node (0,0)
    spec.workers = std::vector<WorkerSpec>{
        MakeMinimalWorker(
            "worker_node0",
            node0,
            {"k_shared", "k_consumer", "k_node0_only", "k_node0_consumer"},
            {"dfb_shared", "dfb_node0"}),
        MakeMinimalWorker(
            "worker_node1",
            node1,
            {"k_shared", "k_consumer", "k_node1_only", "k_node1_consumer"},
            {"dfb_shared", "dfb_node1"}),
    };

    // EXPECTED BEHAVIOR with current implementation: FAILS (greedy algorithm)
    // Update this test to use EXPECT_NO_THROW after the algorithm is fixed.
    EXPECT_ANY_THROW(MakeProgramFromSpec(spec));
}

// Category B: True Simplifying Assumption Violation
// This test is UNSOLVABLE with the simplifying assumption - no algorithm can help.
// When this case arises in production, the assumption must be removed from the codebase.
TEST_F(ProgramSpecTestQuasar, SimplifyingAssumptionViolation_OverlappingMultiNodeKernels) {
    // This test constructs a valid ProgramSpec that CANNOT work with the
    // simplifying assumption, regardless of how clever the solver is.
    //
    // Scenario (the "triangle of doom"):
    //   - Kernel A (3 threads) runs on nodes (0,0) and (0,1)
    //   - Kernel B (3 threads) runs on nodes (0,0) and (0,2)
    //   - Kernel C (3 threads) runs on nodes (0,1) and (0,2)
    //
    // Per-node requirements (6 DM cores available per node):
    //   Node (0,0): A + B = 6 threads
    //   Node (0,1): A + C = 6 threads
    //   Node (0,2): B + C = 6 threads
    //
    // With simplifying assumption (same processors on all nodes per kernel):
    //   Node (0,0): A and B must partition [2-7]. Say A=[2-4], B=[5-7].
    //   Node (0,1): A must be [2-4] (from above). So C=[5-7].
    //   Node (0,2): B must be [5-7], C must be [5-7]. CONFLICT!
    //
    // No assignment algorithm can solve this. The simplifying assumption
    // must be removed to handle this case.

    NodeCoord node_00{0, 0};
    NodeCoord node_01{0, 1};
    NodeCoord node_02{0, 2};

    // Create NodeRangeSets for each kernel's target nodes
    NodeRangeSet nodes_A(std::set<NodeRange>{NodeRange{node_00, node_00}, NodeRange{node_01, node_01}});
    NodeRangeSet nodes_B(std::set<NodeRange>{NodeRange{node_00, node_00}, NodeRange{node_02, node_02}});
    NodeRangeSet nodes_C(std::set<NodeRange>{NodeRange{node_01, node_01}, NodeRange{node_02, node_02}});
    NodeRangeSet all_nodes(
        std::set<NodeRange>{NodeRange{node_00, node_00}, NodeRange{node_01, node_01}, NodeRange{node_02, node_02}});

    ProgramSpec spec;
    spec.program_id = "triangle_of_doom";

    // Three kernels, each spanning two nodes with 4 DM threads
    auto kernel_a = MakeMinimalDMKernel("kernel_a", nodes_A, 4);
    auto kernel_b = MakeMinimalDMKernel("kernel_b", nodes_B, 4);
    auto kernel_c = MakeMinimalDMKernel("kernel_c", nodes_C, 4);

    // We need DFBs with producer/consumer pairs. Create minimal dataflow.
    // Each kernel produces to its own DFB, consumed by a shared sink kernel.
    auto dfb_a = MakeMinimalDFB("dfb_a", nodes_A);
    auto dfb_b = MakeMinimalDFB("dfb_b", nodes_B);
    auto dfb_c = MakeMinimalDFB("dfb_c", nodes_C);

    BindDFBToKernel(kernel_a, "dfb_a", "out", KernelSpec::DFBEndpointType::PRODUCER);
    BindDFBToKernel(kernel_b, "dfb_b", "out", KernelSpec::DFBEndpointType::PRODUCER);
    BindDFBToKernel(kernel_c, "dfb_c", "out", KernelSpec::DFBEndpointType::PRODUCER);

    // Consumer kernels for each DFB (1 thread each, same node coverage as DFB)
    auto consumer_a = MakeMinimalDMKernel("consumer_a", nodes_A, 1);
    auto consumer_b = MakeMinimalDMKernel("consumer_b", nodes_B, 1);
    auto consumer_c = MakeMinimalDMKernel("consumer_c", nodes_C, 1);

    BindDFBToKernel(consumer_a, "dfb_a", "in", KernelSpec::DFBEndpointType::CONSUMER);
    BindDFBToKernel(consumer_b, "dfb_b", "in", KernelSpec::DFBEndpointType::CONSUMER);
    BindDFBToKernel(consumer_c, "dfb_c", "in", KernelSpec::DFBEndpointType::CONSUMER);

    spec.kernels = {kernel_a, kernel_b, kernel_c, consumer_a, consumer_b, consumer_c};
    spec.dataflow_buffers = {dfb_a, dfb_b, dfb_c};

    // Three WorkerSpecs, one per node
    spec.workers = std::vector<WorkerSpec>{
        MakeMinimalWorker(
            "worker_00", node_00, {"kernel_a", "kernel_b", "consumer_a", "consumer_b"}, {"dfb_a", "dfb_b"}),
        MakeMinimalWorker(
            "worker_01", node_01, {"kernel_a", "kernel_c", "consumer_a", "consumer_c"}, {"dfb_a", "dfb_c"}),
        MakeMinimalWorker(
            "worker_02", node_02, {"kernel_b", "kernel_c", "consumer_b", "consumer_c"}, {"dfb_b", "dfb_c"}),
    };

    // EXPECTED BEHAVIOR with current implementation: FAILS (simplifying assumption)
    //
    // Note: If GreedyAlgorithmFailure test passes but this test still fails,
    // that's the expected intermediate state (smarter algorithm, assumption intact).
    EXPECT_ANY_THROW(MakeProgramFromSpec(spec));
}

}  // namespace
}  // namespace tt::tt_metal::experimental::metal2_host_api
