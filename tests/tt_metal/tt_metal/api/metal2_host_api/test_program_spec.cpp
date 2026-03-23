// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Unit tests for the Metal 2.0 Host API: ProgramSpec and MakeProgramFromSpec
//
// These tests validate the ProgramSpec collection, validation, and Program creation logic
// WITHOUT requiring actual hardware. Tests use ArchOverrideGuard to simulate Quasar.
//
// Test categories:
//   1. Structural validation (CollectSpecData) - duplicate names, dangling references
//   2. Semantic validation (ValidateProgramSpec) - architecture rules, resource limits
//   3. WorkerSpec validation - overlap, coverage, resource budgets
//   4. DFB validation - endpoints, bindings, format requirements
//   5. Happy path tests - valid ProgramSpecs that should succeed

#include <gtest/gtest.h>

#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program.hpp>
#include <tt-metalium/core_coord.hpp>

#include "impl/metal2_host_api/test_utils.hpp"

namespace tt::tt_metal::experimental::metal2_host_api {
namespace {

// ============================================================================
// Test Utilities
// ============================================================================

// Helper to create a minimal valid KernelSpec for data movement
KernelSpec MakeMinimalDMKernel(
    const std::string& name, const std::variant<NodeCoord, NodeRange, NodeRangeSet>& nodes, uint8_t num_threads = 1) {
    KernelSpec kernel;
    kernel.unique_id = name;
    kernel.source = "test_kernel.cpp";
    kernel.source_type = KernelSpec::SourceType::FILE_PATH;
    kernel.target_nodes = nodes;
    kernel.num_threads = num_threads;

    DataMovementConfiguration dm_config;
    dm_config.gen2_data_movement_config = DataMovementConfiguration::Gen2DataMovementConfig{};
    kernel.config_spec = dm_config;

    return kernel;
}

// Helper to create a minimal valid KernelSpec for compute
KernelSpec MakeMinimalComputeKernel(
    const std::string& name, const std::variant<NodeCoord, NodeRange, NodeRangeSet>& nodes, uint8_t num_threads = 1) {
    KernelSpec kernel;
    kernel.unique_id = name;
    kernel.source = "test_compute_kernel.cpp";
    kernel.source_type = KernelSpec::SourceType::FILE_PATH;
    kernel.target_nodes = nodes;
    kernel.num_threads = num_threads;

    ComputeConfiguration compute_config;
    kernel.config_spec = compute_config;

    return kernel;
}

// Helper to create a minimal valid DataflowBufferSpec
DataflowBufferSpec MakeMinimalDFB(
    const std::string& name,
    const std::variant<NodeCoord, NodeRange, NodeRangeSet>& nodes,
    uint32_t entry_size = 1024,
    uint32_t num_entries = 2) {
    DataflowBufferSpec dfb;
    dfb.unique_id = name;
    dfb.target_nodes = nodes;
    dfb.entry_size = entry_size;
    dfb.num_entries = num_entries;
    return dfb;
}

// Helper to create a minimal valid WorkerSpec
WorkerSpec MakeMinimalWorker(
    const std::string& name,
    const std::variant<NodeCoord, NodeRange, NodeRangeSet>& nodes,
    const std::vector<KernelSpecName>& kernels,
    const std::vector<DFBSpecName>& dfbs = {}) {
    WorkerSpec worker;
    worker.unique_id = name;
    worker.target_nodes = nodes;
    worker.kernels = kernels;
    worker.dataflow_buffers = dfbs;
    return worker;
}

// Helper to bind a DFB to a kernel as producer or consumer
void BindDFBToKernel(
    KernelSpec& kernel,
    const std::string& dfb_name,
    const std::string& accessor_name,
    KernelSpec::DFBEndpointType endpoint_type,
    DFBAccessPattern access_pattern = DFBAccessPattern::STRIDED) {
    KernelSpec::DFBBinding binding;
    binding.dfb_spec_name = dfb_name;
    binding.local_accessor_name = accessor_name;
    binding.endpoint_type = endpoint_type;
    binding.access_pattern = access_pattern;
    kernel.dfb_bindings.push_back(binding);
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

// Base fixture for validation tests - uses arch override to simulate Quasar
class ProgramSpecTest : public ::testing::Test {
protected:
    void SetUp() override {
        // All tests run with Quasar architecture override
        arch_guard_ = std::make_unique<ArchOverrideGuard>(tt::ARCH::QUASAR);
    }

    void TearDown() override { arch_guard_.reset(); }

private:
    std::unique_ptr<ArchOverrideGuard> arch_guard_;
};

// Fixture for tests that need actual Program creation (requires Quasar hardware)
// These tests are skipped when not running on Quasar
class ProgramSpecHappyPathTest : public ProgramSpecTest {
protected:
    void SetUp() override {
        ProgramSpecTest::SetUp();
        // Skip if not on actual Quasar hardware
        // The arch override makes validation pass, but Program creation needs real HAL
        // TODO: Remove this skip when we have a proper HAL mock
        GTEST_SKIP() << "Happy path tests require Quasar hardware (Program creation needs full HAL support)";
    }
};

// ============================================================================
// SECTION 1: Structural Validation Tests (CollectSpecData)
// ============================================================================
// These test the structural integrity checks that happen during spec collection.

TEST_F(ProgramSpecTest, DuplicateKernelNameFails) {
    ProgramSpec spec = MakeMinimalValidProgramSpec();

    // Add a kernel with duplicate name
    auto duplicate_kernel = MakeMinimalDMKernel("dm_kernel", NodeCoord{1, 0});
    DataMovementConfiguration dm_config;
    dm_config.gen2_data_movement_config = DataMovementConfiguration::Gen2DataMovementConfig{};
    duplicate_kernel.config_spec = dm_config;
    spec.kernels.push_back(duplicate_kernel);

    EXPECT_ANY_THROW(MakeProgramFromSpec(spec));
}

TEST_F(ProgramSpecTest, DuplicateDFBNameFails) {
    ProgramSpec spec = MakeMinimalValidProgramSpec();

    // Add a DFB with duplicate name
    auto duplicate_dfb = MakeMinimalDFB("dfb_0", NodeCoord{1, 0});
    spec.dataflow_buffers.push_back(duplicate_dfb);

    EXPECT_ANY_THROW(MakeProgramFromSpec(spec));
}

TEST_F(ProgramSpecTest, DuplicateSemaphoreNameFails) {
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

TEST_F(ProgramSpecTest, KernelReferencesUnknownDFBFails) {
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

TEST_F(ProgramSpecTest, DFBWithNoBindingsFails) {
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

TEST_F(ProgramSpecTest, DFBWithOnlyProducerFails) {
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

TEST_F(ProgramSpecTest, DFBWithOnlyConsumerFails) {
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

TEST_F(ProgramSpecTest, DFBWithMultipleProducersFails) {
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

TEST_F(ProgramSpecTest, DFBWithMultipleConsumersFails) {
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

TEST_F(ProgramSpecTest, EmptyKernelsFails) {
    ProgramSpec spec;
    spec.program_id = "empty_program";
    spec.workers = std::vector<WorkerSpec>{};  // Empty workers too

    EXPECT_ANY_THROW(MakeProgramFromSpec(spec));
}

TEST_F(ProgramSpecTest, KernelWithZeroThreadsFails) {
    NodeCoord node{0, 0};

    ProgramSpec spec;
    spec.program_id = "test_program";

    auto kernel = MakeMinimalDMKernel("kernel", node, 0);  // 0 threads!
    spec.kernels = {kernel};
    spec.workers = std::vector<WorkerSpec>{MakeMinimalWorker("worker", node, {"kernel"})};

    EXPECT_ANY_THROW(MakeProgramFromSpec(spec));
}

TEST_F(ProgramSpecTest, DMKernelExceedingMaxThreadsFails) {
    NodeCoord node{0, 0};

    ProgramSpec spec;
    spec.program_id = "test_program";

    // Quasar has 8 DM cores per node
    auto kernel = MakeMinimalDMKernel("kernel", node, 9);  // Too many threads!
    spec.kernels = {kernel};
    spec.workers = std::vector<WorkerSpec>{MakeMinimalWorker("worker", node, {"kernel"})};

    EXPECT_ANY_THROW(MakeProgramFromSpec(spec));
}

TEST_F(ProgramSpecTest, ComputeKernelExceedingMaxThreadsFails) {
    NodeCoord node{0, 0};

    ProgramSpec spec;
    spec.program_id = "test_program";

    // Quasar has 4 Tensix cores per node
    auto kernel = MakeMinimalComputeKernel("kernel", node, 5);  // Too many threads!
    spec.kernels = {kernel};
    spec.workers = std::vector<WorkerSpec>{MakeMinimalWorker("worker", node, {"kernel"})};

    EXPECT_ANY_THROW(MakeProgramFromSpec(spec));
}

TEST_F(ProgramSpecTest, DMKernelWithoutGen2ConfigFails) {
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

TEST_F(ProgramSpecTest, DMKernelWithNoConfigAtAllFails) {
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

TEST_F(ProgramSpecTest, RemoteDFBFails) {
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

TEST_F(ProgramSpecTest, BorrowedMemoryDFBFails) {
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

TEST_F(ProgramSpecTest, DFBAliasingFails) {
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

TEST_F(ProgramSpecTest, SemaphoresFail) {
    // Semaphores are not yet implemented for Quasar
    ProgramSpec spec = MakeMinimalValidProgramSpec();

    SemaphoreSpec sem;
    sem.unique_id = "sem_0";
    sem.target_nodes = NodeCoord{0, 0};
    spec.semaphores = {sem};

    EXPECT_ANY_THROW(MakeProgramFromSpec(spec));
}

TEST_F(ProgramSpecTest, KernelSemaphoreBindingsFail) {
    // Semaphore bindings are not yet implemented
    ProgramSpec spec = MakeMinimalValidProgramSpec();

    KernelSpec::SemaphoreBinding binding;
    binding.semaphore_spec_name = "sem_0";
    binding.accessor_name = "my_sem";
    spec.kernels[0].semaphore_bindings = {binding};

    EXPECT_ANY_THROW(MakeProgramFromSpec(spec));
}

TEST_F(ProgramSpecTest, DFBWithComputeEndpointRequiresDataFormat) {
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

TEST_F(ProgramSpecTest, ComputeConfigUnpackToDestModeReferencesUnknownDFBFails) {
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

TEST_F(ProgramSpecTest, MissingWorkerSpecsFails) {
    // Gen2 requires WorkerSpecs
    NodeCoord node{0, 0};

    ProgramSpec spec;
    spec.program_id = "test_program";

    auto kernel = MakeMinimalDMKernel("kernel", node);
    spec.kernels = {kernel};
    // spec.workers is NOT set (nullopt)

    EXPECT_ANY_THROW(MakeProgramFromSpec(spec));
}

TEST_F(ProgramSpecTest, EmptyWorkerSpecsFails) {
    NodeCoord node{0, 0};

    ProgramSpec spec;
    spec.program_id = "test_program";

    auto kernel = MakeMinimalDMKernel("kernel", node);
    spec.kernels = {kernel};
    spec.workers = std::vector<WorkerSpec>{};  // Empty!

    EXPECT_ANY_THROW(MakeProgramFromSpec(spec));
}

TEST_F(ProgramSpecTest, WorkerSpecWithNoKernelsFails) {
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

TEST_F(ProgramSpecTest, OverlappingWorkerSpecsFails) {
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

TEST_F(ProgramSpecTest, KernelNotInAnyWorkerSpecFails) {
    NodeCoord node{0, 0};

    ProgramSpec spec;
    spec.program_id = "test_program";

    auto kernel1 = MakeMinimalDMKernel("kernel1", node);
    auto kernel2 = MakeMinimalDMKernel("kernel2", node);  // Not in any worker!
    spec.kernels = {kernel1, kernel2};

    spec.workers = std::vector<WorkerSpec>{MakeMinimalWorker("worker", node, {"kernel1"})};  // Only kernel1

    EXPECT_ANY_THROW(MakeProgramFromSpec(spec));
}

TEST_F(ProgramSpecTest, KernelTargetNodesMismatchWorkerNodesFails) {
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

TEST_F(ProgramSpecTest, WorkerExceedsDMCoreBudgetFails) {
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

TEST_F(ProgramSpecTest, WorkerExceedsComputeCoreBudgetFails) {
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

TEST_F(ProgramSpecTest, WorkerWithMultipleComputeKernelsFails) {
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

TEST_F(ProgramSpecTest, DFBNotInAnyWorkerSpecFails) {
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
// SECTION 4: Happy Path Tests
// ============================================================================
// These verify that valid configurations succeed.
// NOTE: These tests require Quasar hardware because Program creation
// needs full HAL support. They are skipped on other architectures.

TEST_F(ProgramSpecHappyPathTest, MinimalValidProgramSpecSucceeds) {
    ProgramSpec spec = MakeMinimalValidProgramSpec();

    // Should not throw
    EXPECT_NO_THROW(MakeProgramFromSpec(spec));
}

TEST_F(ProgramSpecHappyPathTest, DMOnlyProgramSucceeds) {
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

TEST_F(ProgramSpecHappyPathTest, MultiNodeProgramSucceeds) {
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

TEST_F(ProgramSpecHappyPathTest, MultipleWorkersOnDifferentNodesSucceeds) {
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

TEST_F(ProgramSpecHappyPathTest, MaxDMThreadsSucceeds) {
    // Use exactly 8 DM threads (the maximum)
    NodeCoord node{0, 0};

    ProgramSpec spec;
    spec.program_id = "max_dm_threads";

    auto producer = MakeMinimalDMKernel("producer", node, 4);
    auto consumer = MakeMinimalDMKernel("consumer", node, 4);  // Total: 8
    auto dfb = MakeMinimalDFB("dfb", node);

    BindDFBToKernel(producer, "dfb", "out", KernelSpec::DFBEndpointType::PRODUCER);
    BindDFBToKernel(consumer, "dfb", "in", KernelSpec::DFBEndpointType::CONSUMER);

    spec.kernels = {producer, consumer};
    spec.dataflow_buffers = {dfb};
    spec.workers = std::vector<WorkerSpec>{MakeMinimalWorker("worker", node, {"producer", "consumer"}, {"dfb"})};

    EXPECT_NO_THROW(MakeProgramFromSpec(spec));
}

TEST_F(ProgramSpecHappyPathTest, MaxComputeThreadsSucceeds) {
    // Use exactly 4 compute threads (the maximum)
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

TEST_F(ProgramSpecHappyPathTest, MultipleDFBsSucceeds) {
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

TEST_F(ProgramSpecHappyPathTest, SkipValidationSucceeds) {
    // Even an invalid spec should succeed if skip_validation is true
    // (This tests the skip_validation flag, not that we should use it with invalid specs)
    ProgramSpec spec = MakeMinimalValidProgramSpec();

    EXPECT_NO_THROW(MakeProgramFromSpec(spec, /*skip_validation=*/true));
}

TEST_F(ProgramSpecHappyPathTest, CompilerOptionsDefinesSucceeds) {
    ProgramSpec spec = MakeMinimalValidProgramSpec();

    // Add some defines
    spec.kernels[0].compiler_options.defines = {{"MY_DEFINE", "42"}, {"ANOTHER_DEFINE", "foo"}};

    EXPECT_NO_THROW(MakeProgramFromSpec(spec));
}

TEST_F(ProgramSpecHappyPathTest, CompileTimeArgBindingsSucceeds) {
    ProgramSpec spec = MakeMinimalValidProgramSpec();

    // Add compile-time arg bindings
    spec.kernels[0].compile_time_arg_bindings = {{"arg1", 100}, {"arg2", 200}};

    EXPECT_NO_THROW(MakeProgramFromSpec(spec));
}

TEST_F(ProgramSpecHappyPathTest, RuntimeArgsSchemaSucceeds) {
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

TEST_F(ProgramSpecHappyPathTest, NodeRangeSetTargetNodesSucceeds) {
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

TEST_F(ProgramSpecHappyPathTest, SourceCodeKernelSucceeds) {
    ProgramSpec spec = MakeMinimalValidProgramSpec();

    // Change to inline source code
    spec.kernels[0].source = "void kernel_main() {}";
    spec.kernels[0].source_type = KernelSpec::SourceType::SOURCE_CODE;

    EXPECT_NO_THROW(MakeProgramFromSpec(spec));
}

TEST_F(ProgramSpecHappyPathTest, ComputeConfigMathFidelitySucceeds) {
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

TEST_F(ProgramSpecHappyPathTest, ValidUnpackToDestModeSucceeds) {
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

}  // namespace
}  // namespace tt::tt_metal::experimental::metal2_host_api
