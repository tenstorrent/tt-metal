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
using test_helpers::MINIMAL_COMPUTE_KERNEL_SOURCE;
using test_helpers::MINIMAL_DM_KERNEL_SOURCE;

// ============================================================================
// Test Fixtures
// ============================================================================

// Test fixture for ProgramSpec on Quasar - uses Quasar mock device
class ProgramSpecTestQuasar : public ::testing::Test {
protected:
    void SetUp() override {
        GTEST_SKIP() << "Re-enable tests after Quasar mock device support is checked in";
        //  Configure global mock mode for Quasar
        //  This way, the HAL is initialized for arch check and Program creation.
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

    // Add Gen1 config
    dm_config.gen1_data_movement_config = DataMovementConfiguration::Gen1DataMovementConfig{
        .processor = DataMovementProcessor::RISCV_0,
        .noc = NOC::RISCV_0_default,
        .noc_mode = NOC_MODE::DM_DEDICATED_NOC,
    };

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

    /*
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
    */

    // DFB currently only supports single-core.
    // For now, we test multi-node programs using single-node DFBs with multi-node kernels.
    // TODO: Fix this when DFB multi-core support is added.

    NodeCoord node0{0, 0};
    NodeCoord node1{1, 0};
    NodeRangeSet all_nodes(std::set<NodeRange>{NodeRange{node0, node0}, NodeRange{node1, node1}});

    ProgramSpec spec;
    spec.program_id = "multi_node_program";

    // Kernels span multiple nodes
    auto producer = MakeMinimalDMKernel("producer", all_nodes);
    auto consumer = MakeMinimalDMKernel("consumer", all_nodes);

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
    dfb.num_entries = 9;  // must be a multiple of the number of threads

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
    dfb.num_entries = 4;  // must be a multiple of the number of threads

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

    // TODO: Update when DFB multi-node support is added
    EXPECT_ANY_THROW(MakeProgramFromSpec(spec));
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
// Here, we test two edge cases:
//
// A) ALGORITHM FAILURE
//    The original naive greedy algorithm could either pass or fail on logically
//    equivalent ProgramSpecs, depending on the order of kernels and workers.
//    The ProgramSpec was legal and solvable (under the simplifying assumption),
//    but the algorithm failed to find a solution.
//    This class of failure should not occur with the backtracking solver.
//
// B) SIMPLIFYING ASSUMPTION VIOLATION
//    Some strictly legal ProgramSpecs are unsolvable if we assume that a kernel
//    must uses the same processor indices on all nodes it runs on.
//
// Our plan is to keep the simplifying assumption for now.
// We issue a clear message if the assumption is ever violated in the real world.

// Category A: Order-Independence Test
// This test verifies that the backtracking solver finds valid assignments,
// regardless of kernel and worker orderings.
TEST_F(ProgramSpecTestQuasar, BacktrackingSolverFindsAssignment_RegardlessOfOrder) {
    // This test verifies that semantically identical ProgramSpecs succeed
    // regardless of the order of:
    //  - workers in spec.workers
    //  - kernels within each worker
    //  - kernel order within the ProgramSpec
    //
    // Scenario:
    //   - K_a:  3 threads on node A only (single-node)
    //   - K_ab: 3 threads on nodes A and B (multi-node)
    //   - K_bc: 3 threads on nodes B and C (multi-node)
    //   - K_c:  3 threads on node C only (single-node)
    //
    // Budget check (6 DM cores per node):
    //   Node A: K_a(3) + K_ab(3) = 6
    //   Node B: K_ab(3) + K_bc(3) = 6
    //   Node C: K_bc(3) + K_c(3) = 6
    //
    // SOLUTION EXISTS:
    //   K_ab uses [2-4] on A and B
    //   K_a  uses [5-7] on A
    //   K_bc uses [5-7] on B and C
    //   K_c  uses [2-4] on C
    //
    // The original naive greedy algorithm would fail on certain orderings:
    //   - Order [K_ab, K_bc, K_a, K_c] would succeed
    //   - Order [K_a, K_c, K_ab, K_bc] would fail
    //
    // The backtracking solver should be order-independent.
    // (Though it still makes the simplifying assumption.)

    NodeCoord node_a{0, 0};
    NodeCoord node_b{0, 1};
    NodeCoord node_c{0, 2};

    NodeRangeSet nodes_ab(std::set<NodeRange>{NodeRange{node_a, node_a}, NodeRange{node_b, node_b}});
    NodeRangeSet nodes_bc(std::set<NodeRange>{NodeRange{node_b, node_b}, NodeRange{node_c, node_c}});

    auto k_a = MakeMinimalDMKernel("k_a", node_a, 3);
    auto k_ab = MakeMinimalDMKernel("k_ab", nodes_ab, 3);
    auto k_bc = MakeMinimalDMKernel("k_bc", nodes_bc, 3);
    auto k_c = MakeMinimalDMKernel("k_c", node_c, 3);

    auto worker_a1 = MakeMinimalWorker("worker_a1", node_a, {"k_a", "k_ab"}, {});
    auto worker_b1 = MakeMinimalWorker("worker_b1", node_b, {"k_ab", "k_bc"}, {});
    auto worker_c1 = MakeMinimalWorker("worker_c1", node_c, {"k_bc", "k_c"}, {});
    auto worker_c2 = MakeMinimalWorker("worker_c2", node_c, {"k_c", "k_bc"}, {});

    // Helper to create a ProgramSpec with a given id and worker ordering
    auto make_spec = [&](const std::string& id, std::vector<KernelSpec> kernels, std::vector<WorkerSpec> workers) {
        ProgramSpec spec;
        spec.program_id = id;
        spec.kernels = kernels;
        spec.workers = workers;
        return spec;
    };

    // All 24 possible permutations of k_a, k_ab, k_bc, k_c
    std::vector<std::vector<KernelSpec>> kernel_permutations = {
        {k_a, k_ab, k_bc, k_c}, {k_a, k_ab, k_c, k_bc}, {k_a, k_bc, k_ab, k_c},
        {k_a, k_bc, k_c, k_ab}, {k_a, k_c, k_ab, k_bc}, {k_a, k_c, k_bc, k_ab},

        {k_ab, k_a, k_bc, k_c}, {k_ab, k_a, k_c, k_bc}, {k_ab, k_bc, k_a, k_c},
        {k_ab, k_bc, k_c, k_a}, {k_ab, k_c, k_a, k_bc}, {k_ab, k_c, k_bc, k_a},

        {k_bc, k_a, k_ab, k_c}, {k_bc, k_a, k_c, k_ab}, {k_bc, k_ab, k_a, k_c},
        {k_bc, k_ab, k_c, k_a}, {k_bc, k_c, k_a, k_ab}, {k_bc, k_c, k_ab, k_a},

        {k_c, k_a, k_ab, k_bc}, {k_c, k_a, k_bc, k_ab}, {k_c, k_ab, k_a, k_bc},
        {k_c, k_ab, k_bc, k_a}, {k_c, k_bc, k_a, k_ab}, {k_c, k_bc, k_ab, k_a}};

    // All 6 possible permutations of worker orderings (using c1)
    std::vector<std::vector<WorkerSpec>> worker_permutations1 = {
        {worker_a1, worker_b1, worker_c1},
        {worker_a1, worker_c1, worker_b1},
        {worker_b1, worker_c1, worker_a1},
        {worker_b1, worker_a1, worker_c1},
        {worker_c1, worker_a1, worker_b1},
        {worker_c1, worker_b1, worker_a1}};

    // All 6 possible permutations of worker orderings (using c2)
    std::vector<std::vector<WorkerSpec>> worker_permutations2 = {
        {worker_a1, worker_b1, worker_c2},
        {worker_a1, worker_c2, worker_b1},
        {worker_b1, worker_c2, worker_a1},
        {worker_b1, worker_a1, worker_c2},
        {worker_c2, worker_a1, worker_b1},
        {worker_c2, worker_b1, worker_a1}};

    // All kernel permutations should succeed with all worker permutations.
    for (size_t i = 0; i < kernel_permutations.size(); i++) {
        for (size_t j = 0; j < worker_permutations1.size(); j++) {
            EXPECT_NO_THROW(MakeProgramFromSpec(make_spec("", kernel_permutations[i], worker_permutations1[j])));
        }
    }
    for (size_t i = 0; i < kernel_permutations.size(); i++) {
        for (size_t j = 0; j < worker_permutations2.size(); j++) {
            EXPECT_NO_THROW(MakeProgramFromSpec(make_spec("", kernel_permutations[i], worker_permutations2[j])));
        }
    }
}

// Category B: True Simplifying Assumption Violation
// This test is UNSOLVABLE with the simplifying assumption - no algorithm can help.
// When this case arises in production, the assumption must be removed from the codebase.
TEST_F(ProgramSpecTestQuasar, SimplifyingAssumptionViolation_OverlappingMultiNodeKernels) {
    // This test constructs a valid ProgramSpec that CANNOT work with the
    // "same DM cores on every node" simplifying assumption, regardless of
    // how clever the solver is.
    //
    // Scenario (the "triangle of doom"):
    //   - Kernel A (3 threads) runs on nodes (0,0) and (0,1)
    //   - Kernel B (3 threads) runs on nodes (0,0) and (0,2)
    //   - Kernel C (3 threads) runs on nodes (0,1) and (0,2)
    //
    // Per-node DM budget (6 DM cores available per node):
    //   Node (0,0): A(3) + B(3) = 6 ✓
    //   Node (0,1): A(3) + C(3) = 6 ✓
    //   Node (0,2): B(3) + C(3) = 6 ✓
    //
    // With simplifying assumption (each kernel uses same cores on all its nodes):
    //   Node (0,0): A and B must partition [2-7]. Say A=[2-4], B=[5-7].
    //   Node (0,1): A must be [2-4] (from above). So C=[5-7].
    //   Node (0,2): B must be [5-7], C must be [5-7]. CONFLICT!
    //
    // No matter how we assign, there's always a conflict on one node.
    // The simplifying assumption must be removed to handle this case.
    //
    // When this test starts PASSING: the simplifying assumption has been removed.
    // Change to EXPECT_NO_THROW.

    NodeCoord node_00{0, 0};
    NodeCoord node_01{0, 1};
    NodeCoord node_02{0, 2};

    NodeRangeSet nodes_A(std::set<NodeRange>{NodeRange{node_00, node_00}, NodeRange{node_01, node_01}});
    NodeRangeSet nodes_B(std::set<NodeRange>{NodeRange{node_00, node_00}, NodeRange{node_02, node_02}});
    NodeRangeSet nodes_C(std::set<NodeRange>{NodeRange{node_01, node_01}, NodeRange{node_02, node_02}});

    ProgramSpec spec;
    spec.program_id = "triangle_of_doom";

    auto kernel_a = MakeMinimalDMKernel("kernel_a", nodes_A, 3);
    auto kernel_b = MakeMinimalDMKernel("kernel_b", nodes_B, 3);
    auto kernel_c = MakeMinimalDMKernel("kernel_c", nodes_C, 3);

    spec.kernels = {kernel_a, kernel_b, kernel_c};

    spec.workers = std::vector<WorkerSpec>{
        MakeMinimalWorker("worker_00", node_00, {"kernel_a", "kernel_b"}, {}),
        MakeMinimalWorker("worker_01", node_01, {"kernel_a", "kernel_c"}, {}),
        MakeMinimalWorker("worker_02", node_02, {"kernel_b", "kernel_c"}, {}),
    };

    // EXPECTED BEHAVIOR: FAILS due to simplifying assumption violation.
    EXPECT_ANY_THROW(MakeProgramFromSpec(spec));
}

}  // namespace
}  // namespace tt::tt_metal::experimental::metal2_host_api
