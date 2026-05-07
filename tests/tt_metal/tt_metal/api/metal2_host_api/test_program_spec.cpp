// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

//---------------------------------------------------------------------------------
// Unit tests for the Metal 2.0 Host API: ProgramSpec and MakeProgramFromSpec
// These tests all use mock device (Quasar and Wormhole) for API-level validation.
//
// Test categories:
//  Quasar (Gen2):
//   1. ProgramSpec "structural" validation (CollectSpecData)
//   2. ProgramSpec semantic validation (ValidateProgramSpec)
//   3. WorkUnitSpec validation
//   4. Basic Program creation (should succeed)
//   5. Misc edge cases in Program creation
//   6. Quasar processor assignment logic correctness (includes pathological case)
//   7. Aggregate type enforcement (designated initializers must work!)
// Wormhole (Gen1):
//   8. Gen1 specific tests
//
//---------------------------------------------------------------------------------
// These unit tests use shortcut functions to create minimal valid ProgramSpec
// objects to cut repeated boilerplate (test_helpers.hpp)
//
// This is NOT intended as a recommended pattern for production code!
// See the Metal 2.0 Host API documentation and programming examples for
// recommended patterns for constructing ProgramSpec objects in production code.
//---------------------------------------------------------------------------------

#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <type_traits>

#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/experimental/context/metal_env.hpp>
#include <tt-metalium/experimental/mock_device.hpp>

#include "test_helpers.hpp"

namespace tt::tt_metal::experimental::metal2_host_api {
namespace {

// Import shared test helpers
using test_helpers::BindDFBToKernel;
using test_helpers::MakeMinimalComputeKernel;
using test_helpers::MakeMinimalDFB;
using test_helpers::MakeMinimalDMKernel;
using test_helpers::MakeMinimalGen1DMKernel;
using test_helpers::MakeMinimalGen1ValidProgramSpec;
using test_helpers::MakeMinimalValidProgramSpec;
using test_helpers::MakeMinimalWorkUnit;

// ============================================================================
// Test Fixtures
// ============================================================================

// Test fixture for ProgramSpec on Quasar - uses Quasar mock device
class ProgramSpecTestQuasar : public ::testing::Test {
protected:
    void SetUp() override {
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
    auto duplicate_kernel = MakeMinimalDMKernel("dm_kernel");
    DataMovementConfiguration dm_config;
    dm_config.gen2_data_movement_config = DataMovementConfiguration::Gen2DataMovementConfig{};
    duplicate_kernel.config_spec = dm_config;
    spec.kernels.push_back(duplicate_kernel);

    EXPECT_THAT(
        [&] { MakeProgramFromSpec(spec); },
        ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr("Duplicate KernelSpec name 'dm_kernel'")));
}

TEST_F(ProgramSpecTestQuasar, DuplicateDFBNameFails) {
    ProgramSpec spec = MakeMinimalValidProgramSpec();

    // Add a DFB with duplicate name
    auto duplicate_dfb = MakeMinimalDFB("dfb_0");
    spec.dataflow_buffers.push_back(duplicate_dfb);

    EXPECT_THAT(
        [&] { MakeProgramFromSpec(spec); },
        ::testing::ThrowsMessage<std::runtime_error>(
            ::testing::HasSubstr("Duplicate DataflowBufferSpec name 'dfb_0'")));
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

    EXPECT_THAT(
        [&] { MakeProgramFromSpec(spec); },
        ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr("Duplicate SemaphoreSpec name 'sem_0'")));
}

TEST_F(ProgramSpecTestQuasar, DuplicateWorkUnitNameFails) {
    NodeCoord node0{0, 0};
    NodeCoord node1{1, 0};

    ProgramSpec spec;
    spec.program_id = "test_program";

    // Two kernels on different nodes
    auto kernel0 = MakeMinimalDMKernel("kernel0");
    auto kernel1 = MakeMinimalDMKernel("kernel1");
    spec.kernels = {kernel0, kernel1};

    // Two work_units with the same unique_id!
    auto work_unit0 = MakeMinimalWorkUnit("same_name", node0, {"kernel0"});
    auto work_unit1 = MakeMinimalWorkUnit("same_name", node1, {"kernel1"});  // Duplicate!
    spec.work_units = std::vector<WorkUnitSpec>{work_unit0, work_unit1};

    EXPECT_THAT(
        [&] { MakeProgramFromSpec(spec); },
        ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr("Duplicate WorkUnitSpec name 'same_name'")));
}

TEST_F(ProgramSpecTestQuasar, DuplicateLocalAccessorNameFails) {
    NodeCoord node{0, 0};

    ProgramSpec spec;
    spec.program_id = "test_program";

    auto kernel = MakeMinimalDMKernel("kernel");
    auto dfb0 = MakeMinimalDFB("dfb_0");
    auto dfb1 = MakeMinimalDFB("dfb_1");

    // Bind two DFBs with the same local_accessor_name
    BindDFBToKernel(kernel, "dfb_0", "same_accessor", KernelSpec::DFBEndpointType::PRODUCER);
    BindDFBToKernel(kernel, "dfb_1", "same_accessor", KernelSpec::DFBEndpointType::CONSUMER);  // Duplicate accessor!

    spec.kernels = {kernel};
    spec.dataflow_buffers = {dfb0, dfb1};
    spec.work_units = std::vector<WorkUnitSpec>{MakeMinimalWorkUnit("work_unit", node, {"kernel"})};

    EXPECT_THAT(
        [&] { MakeProgramFromSpec(spec); },
        ::testing::ThrowsMessage<std::runtime_error>(
            ::testing::HasSubstr("Kernel 'kernel' has duplicate local_accessor_name 'same_accessor'")));
}

TEST_F(ProgramSpecTestQuasar, InvalidLocalAccessorNameFails) {
    NodeCoord node{0, 0};

    const std::vector<std::string> invalid_names = {
        "",               // empty
        "has-dash",       // hyphen
        "has space",      // whitespace
        "1starts_digit",  // leading digit
        "has.dot",        // punctuation
        "class",          // C++ keyword
        "namespace",      // C++ keyword
        "int",            // C++ keyword
        "_Foo",           // reserved: underscore + uppercase
        "__foo",          // reserved: leading double underscore
        "foo__bar",       // reserved: embedded double underscore
    };

    for (const auto& bad_name : invalid_names) {
        ProgramSpec spec;
        spec.program_id = "test_program";

        auto kernel = MakeMinimalDMKernel("kernel");
        auto dfb = MakeMinimalDFB("dfb");

        BindDFBToKernel(kernel, "dfb", bad_name, KernelSpec::DFBEndpointType::PRODUCER);

        spec.kernels = {kernel};
        spec.dataflow_buffers = {dfb};
        spec.work_units = std::vector<WorkUnitSpec>{MakeMinimalWorkUnit("work_unit", node, {"kernel"})};

        EXPECT_THAT(
            [&] { MakeProgramFromSpec(spec); },
            ::testing::ThrowsMessage<std::runtime_error>(
                ::testing::HasSubstr("DFB local_accessor_name '" + bad_name + "' must be a valid C++ identifier")))
            << "Expected rejection for name: '" << bad_name << "'";
    }
}

TEST_F(ProgramSpecTestQuasar, KernelReferencesUnknownDFBFails) {
    NodeCoord node{0, 0};

    ProgramSpec spec;
    spec.program_id = "test_program";

    auto kernel = MakeMinimalDMKernel("kernel");
    // Bind to a DFB that doesn't exist
    BindDFBToKernel(kernel, "nonexistent_dfb", "accessor", KernelSpec::DFBEndpointType::PRODUCER);

    spec.kernels = {kernel};
    spec.work_units = std::vector<WorkUnitSpec>{MakeMinimalWorkUnit("work_unit", node, {"kernel"})};

    EXPECT_THAT(
        [&] { MakeProgramFromSpec(spec); },
        ::testing::ThrowsMessage<std::runtime_error>(
            ::testing::HasSubstr("Kernel 'kernel' references unknown DFB 'nonexistent_dfb'")));
}

TEST_F(ProgramSpecTestQuasar, DFBWithNoBindingsFails) {
    NodeCoord node{0, 0};

    ProgramSpec spec;
    spec.program_id = "test_program";

    // Create a kernel with no DFB bindings
    auto kernel = MakeMinimalDMKernel("kernel");
    spec.kernels = {kernel};

    // Create a DFB that is never bound
    auto orphan_dfb = MakeMinimalDFB("orphan_dfb");
    spec.dataflow_buffers = {orphan_dfb};

    spec.work_units = std::vector<WorkUnitSpec>{MakeMinimalWorkUnit("work_unit", node, {"kernel"})};

    EXPECT_THAT(
        [&] { MakeProgramFromSpec(spec); },
        ::testing::ThrowsMessage<std::runtime_error>(
            ::testing::HasSubstr("DFB 'orphan_dfb' is defined but not bound by any kernel")));
}

TEST_F(ProgramSpecTestQuasar, DFBWithOnlyProducerFails) {
    NodeCoord node{0, 0};

    ProgramSpec spec;
    spec.program_id = "test_program";

    auto kernel = MakeMinimalDMKernel("kernel");
    auto dfb = MakeMinimalDFB("dfb");

    // Only bind as producer, no consumer
    BindDFBToKernel(kernel, "dfb", "accessor", KernelSpec::DFBEndpointType::PRODUCER);

    spec.kernels = {kernel};
    spec.dataflow_buffers = {dfb};
    spec.work_units = std::vector<WorkUnitSpec>{MakeMinimalWorkUnit("work_unit", node, {"kernel"})};

    EXPECT_THAT(
        [&] { MakeProgramFromSpec(spec); },
        ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr("DFB 'dfb' has no consumer")));
}

TEST_F(ProgramSpecTestQuasar, DFBWithOnlyConsumerFails) {
    NodeCoord node{0, 0};

    ProgramSpec spec;
    spec.program_id = "test_program";

    auto kernel = MakeMinimalDMKernel("kernel");
    auto dfb = MakeMinimalDFB("dfb");

    // Only bind as consumer, no producer
    BindDFBToKernel(kernel, "dfb", "accessor", KernelSpec::DFBEndpointType::CONSUMER);

    spec.kernels = {kernel};
    spec.dataflow_buffers = {dfb};
    spec.work_units = std::vector<WorkUnitSpec>{MakeMinimalWorkUnit("work_unit", node, {"kernel"})};

    EXPECT_THAT(
        [&] { MakeProgramFromSpec(spec); },
        ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr("DFB 'dfb' has no producer")));
}

TEST_F(ProgramSpecTestQuasar, DFBWithMultipleProducersFails) {
    NodeCoord node{0, 0};

    ProgramSpec spec;
    spec.program_id = "test_program";

    auto producer1 = MakeMinimalDMKernel("producer1");
    auto producer2 = MakeMinimalDMKernel("producer2");
    auto consumer = MakeMinimalComputeKernel("consumer");

    auto dfb = MakeMinimalDFB("dfb");
    dfb.data_format_metadata = tt::DataFormat::Float16_b;

    // Two producers for same DFB
    BindDFBToKernel(producer1, "dfb", "out", KernelSpec::DFBEndpointType::PRODUCER);
    BindDFBToKernel(producer2, "dfb", "out", KernelSpec::DFBEndpointType::PRODUCER);
    BindDFBToKernel(consumer, "dfb", "in", KernelSpec::DFBEndpointType::CONSUMER);

    spec.kernels = {producer1, producer2, consumer};
    spec.dataflow_buffers = {dfb};
    spec.work_units =
        std::vector<WorkUnitSpec>{MakeMinimalWorkUnit("work_unit", node, {"producer1", "producer2", "consumer"})};

    EXPECT_THAT(
        [&] { MakeProgramFromSpec(spec); },
        ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr("DFB 'dfb' has multiple producers")));
}

TEST_F(ProgramSpecTestQuasar, DFBWithMultipleConsumersFails) {
    NodeCoord node{0, 0};

    ProgramSpec spec;
    spec.program_id = "test_program";

    auto producer = MakeMinimalDMKernel("producer");
    auto consumer1 = MakeMinimalComputeKernel("consumer1");
    auto consumer2 = MakeMinimalDMKernel("consumer2");

    auto dfb = MakeMinimalDFB("dfb");
    dfb.data_format_metadata = tt::DataFormat::Float16_b;

    // Two consumers for same DFB
    BindDFBToKernel(producer, "dfb", "out", KernelSpec::DFBEndpointType::PRODUCER);
    BindDFBToKernel(consumer1, "dfb", "in", KernelSpec::DFBEndpointType::CONSUMER);
    BindDFBToKernel(consumer2, "dfb", "in", KernelSpec::DFBEndpointType::CONSUMER);

    spec.kernels = {producer, consumer1, consumer2};
    spec.dataflow_buffers = {dfb};
    spec.work_units =
        std::vector<WorkUnitSpec>{MakeMinimalWorkUnit("work_unit", node, {"producer", "consumer1", "consumer2"})};

    EXPECT_THAT(
        [&] { MakeProgramFromSpec(spec); },
        ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr("DFB 'dfb' has multiple consumers")));
}

// ============================================================================
// SECTION 2: Semantic Validation Tests (ValidateProgramSpec)
// ============================================================================

TEST_F(ProgramSpecTestQuasar, EmptyKernelsFails) {
    ProgramSpec spec;
    spec.program_id = "empty_program";
    spec.work_units = std::vector<WorkUnitSpec>{};  // Empty work_units too

    EXPECT_THAT(
        [&] { MakeProgramFromSpec(spec); },
        ::testing::ThrowsMessage<std::runtime_error>(
            ::testing::HasSubstr("A ProgramSpec must have at least one KernelSpec")));
}

TEST_F(ProgramSpecTestQuasar, KernelWithZeroThreadsFails) {
    NodeCoord node{0, 0};

    ProgramSpec spec;
    spec.program_id = "test_program";

    auto kernel = MakeMinimalDMKernel("kernel", 0);  // 0 threads!
    spec.kernels = {kernel};
    spec.work_units = std::vector<WorkUnitSpec>{MakeMinimalWorkUnit("work_unit", node, {"kernel"})};

    EXPECT_THAT(
        [&] { MakeProgramFromSpec(spec); },
        ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr("KernelSpec 'kernel' has no threads")));
}

TEST_F(ProgramSpecTestQuasar, DMKernelExceedingMaxThreadsFails) {
    NodeCoord node{0, 0};

    ProgramSpec spec;
    spec.program_id = "test_program";

    // Quasar has 8 DM cores per node (we reserve 2 for internal use)
    auto kernel = MakeMinimalDMKernel("kernel", 9);  // Too many threads!
    spec.kernels = {kernel};
    spec.work_units = std::vector<WorkUnitSpec>{MakeMinimalWorkUnit("work_unit", node, {"kernel"})};

    EXPECT_THAT(
        [&] { MakeProgramFromSpec(spec); },
        ::testing::ThrowsMessage<std::runtime_error>(
            ::testing::HasSubstr("KernelSpec 'kernel' has too many data movement threads")));
}

TEST_F(ProgramSpecTestQuasar, ComputeKernelExceedingMaxThreadsFails) {
    NodeCoord node{0, 0};

    ProgramSpec spec;
    spec.program_id = "test_program";

    // Quasar has 4 Tensix cores per node
    auto kernel = MakeMinimalComputeKernel("kernel", 5);  // Too many threads!
    spec.kernels = {kernel};
    spec.work_units = std::vector<WorkUnitSpec>{MakeMinimalWorkUnit("work_unit", node, {"kernel"})};

    EXPECT_THAT(
        [&] { MakeProgramFromSpec(spec); },
        ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr(
            "KernelSpec 'kernel' has too many threads. The architecture supports up to 4 for compute kernels")));
}

TEST_F(ProgramSpecTestQuasar, DMKernelWithoutGen2ConfigFails) {
    NodeCoord node{0, 0};

    ProgramSpec spec;
    spec.program_id = "test_program";

    auto kernel = MakeMinimalDMKernel("kernel");
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
    spec.work_units = std::vector<WorkUnitSpec>{MakeMinimalWorkUnit("work_unit", node, {"kernel"})};

    EXPECT_THAT(
        [&] { MakeProgramFromSpec(spec); },
        ::testing::ThrowsMessage<std::runtime_error>(
            ::testing::HasSubstr("KernelSpec 'kernel' must specify a Gen2 DM config when targeting Quasar")));
}

TEST_F(ProgramSpecTestQuasar, DMKernelWithNoConfigAtAllFails) {
    NodeCoord node{0, 0};

    ProgramSpec spec;
    spec.program_id = "test_program";

    auto kernel = MakeMinimalDMKernel("kernel");
    // Remove both Gen1 and Gen2 configs
    auto& dm_config = std::get<DataMovementConfiguration>(kernel.config_spec);
    dm_config.gen1_data_movement_config = std::nullopt;
    dm_config.gen2_data_movement_config = std::nullopt;

    spec.kernels = {kernel};
    spec.work_units = std::vector<WorkUnitSpec>{MakeMinimalWorkUnit("work_unit", node, {"kernel"})};

    EXPECT_THAT(
        [&] { MakeProgramFromSpec(spec); },
        ::testing::ThrowsMessage<std::runtime_error>(
            ::testing::HasSubstr("KernelSpec 'kernel' must specify a DM config for Gen1, Gen2, or both")));
}

// Remote DFBs are part of the API surface but not yet supported by the runtime.
TEST_F(ProgramSpecTestQuasar, RemoteDFBNotYetSupportedAtRuntime) {
    NodeCoord producer_node{0, 0};
    NodeCoord consumer_node{1, 0};

    ProgramSpec spec;
    spec.program_id = "test_program";

    auto producer = MakeMinimalDMKernel("producer");
    auto consumer = MakeMinimalDMKernel("consumer");

    BindDFBToKernel(producer, "dfb", "out", KernelSpec::DFBEndpointType::PRODUCER);
    BindDFBToKernel(consumer, "dfb", "in", KernelSpec::DFBEndpointType::CONSUMER);

    spec.kernels = {producer, consumer};
    spec.remote_dataflow_buffers = {RemoteDataflowBufferSpec{
        .dfb_spec = MakeMinimalDFB("dfb"),
        .producer_consumer_map = {{producer_node, consumer_node}},
    }};
    spec.work_units = std::vector<WorkUnitSpec>{
        MakeMinimalWorkUnit("producer_work_unit", producer_node, {"producer"}),
        MakeMinimalWorkUnit("consumer_work_unit", consumer_node, {"consumer"}),
    };

    EXPECT_THAT(
        [&] { MakeProgramFromSpec(spec); },
        ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr("not yet supported")));
}

// Remove once implemented
TEST_F(ProgramSpecTestQuasar, BorrowedMemoryDFBFails) {
    // Borrowed memory DFBs are not yet implemented
    NodeCoord node{0, 0};

    ProgramSpec spec;
    spec.program_id = "test_program";

    auto producer = MakeMinimalDMKernel("producer");
    auto consumer = MakeMinimalDMKernel("consumer");
    auto dfb = MakeMinimalDFB("dfb");
    dfb.uses_borrowed_memory = true;  // Not supported!

    BindDFBToKernel(producer, "dfb", "out", KernelSpec::DFBEndpointType::PRODUCER);
    BindDFBToKernel(consumer, "dfb", "in", KernelSpec::DFBEndpointType::CONSUMER);

    spec.kernels = {producer, consumer};
    spec.dataflow_buffers = {dfb};
    spec.work_units = std::vector<WorkUnitSpec>{MakeMinimalWorkUnit("work_unit", node, {"producer", "consumer"})};

    EXPECT_THAT(
        [&] { MakeProgramFromSpec(spec); },
        ::testing::ThrowsMessage<std::runtime_error>(
            ::testing::HasSubstr("DFB 'dfb' uses borrowed memory, but this feature is not yet implemented")));
}

// Remove once implemented
TEST_F(ProgramSpecTestQuasar, DFBAliasingFails) {
    // DFB aliasing is not yet implemented
    NodeCoord node{0, 0};

    ProgramSpec spec;
    spec.program_id = "test_program";

    auto producer = MakeMinimalDMKernel("producer");
    auto consumer = MakeMinimalDMKernel("consumer");
    auto dfb = MakeMinimalDFB("dfb");
    dfb.alias_with = std::vector<DFBSpecName>{"other_dfb"};  // Not supported yet!

    BindDFBToKernel(producer, "dfb", "out", KernelSpec::DFBEndpointType::PRODUCER);
    BindDFBToKernel(consumer, "dfb", "in", KernelSpec::DFBEndpointType::CONSUMER);

    spec.kernels = {producer, consumer};
    spec.dataflow_buffers = {dfb};
    spec.work_units = std::vector<WorkUnitSpec>{MakeMinimalWorkUnit("work_unit", node, {"producer", "consumer"})};

    EXPECT_THAT(
        [&] { MakeProgramFromSpec(spec); },
        ::testing::ThrowsMessage<std::runtime_error>(
            ::testing::HasSubstr("DFB 'dfb' has a non-empty alias_with, but DFB aliasing is not yet implemented")));
}

TEST_F(ProgramSpecTestQuasar, SemaphoresSucceed) {
    ProgramSpec spec = MakeMinimalValidProgramSpec();

    SemaphoreSpec sem;
    sem.unique_id = "sem_0";
    sem.target_nodes = NodeCoord{0, 0};
    spec.semaphores = {sem};

    EXPECT_NO_THROW(MakeProgramFromSpec(spec));
}

TEST_F(ProgramSpecTestQuasar, KernelSemaphoreBindingsSucceed) {
    ProgramSpec spec = MakeMinimalValidProgramSpec();

    SemaphoreSpec sem;
    sem.unique_id = "sem_0";
    sem.target_nodes = NodeCoord{0, 0};
    spec.semaphores = {sem};

    KernelSpec::SemaphoreBinding binding;
    binding.semaphore_spec_name = "sem_0";
    binding.accessor_name = "my_sem";
    spec.kernels[0].semaphore_bindings = {binding};

    EXPECT_NO_THROW(MakeProgramFromSpec(spec));
}

TEST_F(ProgramSpecTestQuasar, SemaphoreBoundToComputeKernelSucceedsOnQuasar) {
    // Quasar permits compute kernels to participate in semaphore signalling.
    // (The corresponding Gen1 test asserts this is rejected on WH/BH.)
    ProgramSpec spec = MakeMinimalValidProgramSpec();

    SemaphoreSpec sem;
    sem.unique_id = "sem_0";
    sem.target_nodes = NodeCoord{0, 0};
    spec.semaphores = {sem};

    // kernels[1] is the compute kernel in MakeMinimalValidProgramSpec
    ASSERT_TRUE(spec.kernels[1].is_compute_kernel());
    spec.kernels[1].semaphore_bindings = {
        KernelSpec::SemaphoreBinding{.semaphore_spec_name = "sem_0", .accessor_name = "done_flag"}};

    EXPECT_NO_THROW(MakeProgramFromSpec(spec));
}

TEST_F(ProgramSpecTestQuasar, KernelSemaphoreBindingUnknownSemaphoreFails) {
    ProgramSpec spec = MakeMinimalValidProgramSpec();

    KernelSpec::SemaphoreBinding binding;
    binding.semaphore_spec_name = "missing_sem";
    binding.accessor_name = "my_sem";
    spec.kernels[0].semaphore_bindings = {binding};

    EXPECT_THAT(
        [&] { MakeProgramFromSpec(spec); },
        ::testing::ThrowsMessage<std::runtime_error>(
            ::testing::HasSubstr("references unknown semaphore 'missing_sem'")));
}

TEST_F(ProgramSpecTestQuasar, KernelSemaphoreBindingInvalidAccessorFails) {
    ProgramSpec spec = MakeMinimalValidProgramSpec();

    SemaphoreSpec sem;
    sem.unique_id = "sem_0";
    sem.target_nodes = NodeCoord{0, 0};
    spec.semaphores = {sem};

    KernelSpec::SemaphoreBinding binding;
    binding.semaphore_spec_name = "sem_0";
    binding.accessor_name = "has-dash";
    spec.kernels[0].semaphore_bindings = {binding};

    EXPECT_THAT(
        [&] { MakeProgramFromSpec(spec); },
        ::testing::ThrowsMessage<std::runtime_error>(
            ::testing::HasSubstr("semaphore accessor_name 'has-dash' must be a valid C++ identifier")));
}

TEST_F(ProgramSpecTestQuasar, KernelSemaphoreBindingDuplicateAccessorFails) {
    ProgramSpec spec = MakeMinimalValidProgramSpec();

    SemaphoreSpec sem0;
    sem0.unique_id = "sem_0";
    sem0.target_nodes = NodeCoord{0, 0};

    SemaphoreSpec sem1;
    sem1.unique_id = "sem_1";
    sem1.target_nodes = NodeCoord{0, 0};

    spec.semaphores = {sem0, sem1};

    spec.kernels[0].semaphore_bindings = {
        KernelSpec::SemaphoreBinding{.semaphore_spec_name = "sem_0", .accessor_name = "same"},
        KernelSpec::SemaphoreBinding{.semaphore_spec_name = "sem_1", .accessor_name = "same"}};

    EXPECT_THAT(
        [&] { MakeProgramFromSpec(spec); },
        ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr("duplicate semaphore accessor_name 'same'")));
}

TEST_F(ProgramSpecTestQuasar, SemaphoreNonZeroInitialValueFailsOnQuasar) {
    ProgramSpec spec = MakeMinimalValidProgramSpec();

    SemaphoreSpec sem;
    sem.unique_id = "sem_0";
    sem.target_nodes = NodeCoord{0, 0};
    sem.initial_value = 1;
    spec.semaphores = {sem};

    EXPECT_THAT(
        [&] { MakeProgramFromSpec(spec); },
        ::testing::ThrowsMessage<std::runtime_error>(
            ::testing::HasSubstr("has initial_value=1 but only zero is supported on Quasar")));
}

// ---- Named RTA / CRTA / CTA schema validation ----

TEST_F(ProgramSpecTestQuasar, NamedRuntimeArgsSucceeds) {
    ProgramSpec spec = MakeMinimalValidProgramSpec();
    spec.kernels[0].runtime_arguments_schema.named_runtime_args = {"input_ptr", "output_ptr"};
    spec.kernels[0].runtime_arguments_schema.named_common_runtime_args = {"tile_count"};
    spec.kernels[0].compile_time_arg_bindings = {{"block_size", 64}};

    EXPECT_NO_THROW(MakeProgramFromSpec(spec));
}

TEST_F(ProgramSpecTestQuasar, InvalidNamedRtaIdentifierFails) {
    ProgramSpec spec = MakeMinimalValidProgramSpec();
    spec.kernels[0].runtime_arguments_schema.named_runtime_args = {"int"};  // C++ keyword

    EXPECT_THAT(
        [&] { MakeProgramFromSpec(spec); },
        ::testing::ThrowsMessage<std::runtime_error>(
            ::testing::HasSubstr("named RTA name 'int' is not a valid C++ identifier")));
}

TEST_F(ProgramSpecTestQuasar, InvalidNamedCrtaIdentifierFails) {
    ProgramSpec spec = MakeMinimalValidProgramSpec();
    spec.kernels[0].runtime_arguments_schema.named_common_runtime_args = {"has-dash"};

    EXPECT_THAT(
        [&] { MakeProgramFromSpec(spec); },
        ::testing::ThrowsMessage<std::runtime_error>(
            ::testing::HasSubstr("named CRTA name 'has-dash' is not a valid C++ identifier")));
}

TEST_F(ProgramSpecTestQuasar, NamedRtaCrtaCollisionFails) {
    // A single name cannot be both a named RTA and a named CRTA (they share the user namespace).
    ProgramSpec spec = MakeMinimalValidProgramSpec();
    spec.kernels[0].runtime_arguments_schema.named_runtime_args = {"count"};
    spec.kernels[0].runtime_arguments_schema.named_common_runtime_args = {"count"};

    EXPECT_THAT(
        [&] { MakeProgramFromSpec(spec); },
        ::testing::ThrowsMessage<std::runtime_error>(
            ::testing::HasSubstr("naming collision: 'count' is declared as both a named RTA and a named CRTA")));
}

TEST_F(ProgramSpecTestQuasar, NamedRtaCtaCollisionFails) {
    ProgramSpec spec = MakeMinimalValidProgramSpec();
    spec.kernels[0].runtime_arguments_schema.named_runtime_args = {"block_size"};
    spec.kernels[0].compile_time_arg_bindings = {{"block_size", 64}};  // same name as CTA

    EXPECT_THAT(
        [&] { MakeProgramFromSpec(spec); },
        ::testing::ThrowsMessage<std::runtime_error>(
            ::testing::HasSubstr("naming collision: 'block_size' is declared as both a named RTA and a named CTA")));
}

TEST_F(ProgramSpecTestQuasar, DifferentKernelsMayReuseArgNames) {
    // Collision rule is per-kernel. Two different kernels may have identically-named args.
    ProgramSpec spec = MakeMinimalValidProgramSpec();
    spec.kernels[0].runtime_arguments_schema.named_runtime_args = {"shared_name"};
    spec.kernels[1].runtime_arguments_schema.named_runtime_args = {"shared_name"};

    EXPECT_NO_THROW(MakeProgramFromSpec(spec));
}

TEST_F(ProgramSpecTestQuasar, DFBWithComputeEndpointRequiresDataFormat) {
    NodeCoord node{0, 0};

    ProgramSpec spec;
    spec.program_id = "test_program";

    auto producer = MakeMinimalDMKernel("producer");
    auto consumer = MakeMinimalComputeKernel("consumer");  // Compute!
    auto dfb = MakeMinimalDFB("dfb");
    // dfb.data_format_metadata is NOT set (nullopt)

    BindDFBToKernel(producer, "dfb", "out", KernelSpec::DFBEndpointType::PRODUCER);
    BindDFBToKernel(consumer, "dfb", "in", KernelSpec::DFBEndpointType::CONSUMER);

    spec.kernels = {producer, consumer};
    spec.dataflow_buffers = {dfb};
    spec.work_units = std::vector<WorkUnitSpec>{MakeMinimalWorkUnit("work_unit", node, {"producer", "consumer"})};

    EXPECT_THAT(
        [&] { MakeProgramFromSpec(spec); },
        ::testing::ThrowsMessage<std::runtime_error>(
            ::testing::HasSubstr("DFB 'dfb' is used by a compute kernel, but no data_format_metadata is specified")));
}

TEST_F(ProgramSpecTestQuasar, ComputeConfigUnpackToDestModeReferencesUnknownDFBFails) {
    NodeCoord node{0, 0};

    ProgramSpec spec;
    spec.program_id = "test_program";

    auto producer = MakeMinimalDMKernel("producer");
    auto consumer = MakeMinimalComputeKernel("consumer");

    // Set unpack_to_dest_mode referencing a DFB that doesn't exist
    auto& compute_config = std::get<ComputeConfiguration>(consumer.config_spec);
    compute_config.unpack_to_dest_mode = {{"nonexistent_dfb", UnpackToDestMode::UnpackToDestFp32}};

    auto dfb = MakeMinimalDFB("dfb");
    dfb.data_format_metadata = tt::DataFormat::Float16_b;

    BindDFBToKernel(producer, "dfb", "out", KernelSpec::DFBEndpointType::PRODUCER);
    BindDFBToKernel(consumer, "dfb", "in", KernelSpec::DFBEndpointType::CONSUMER);

    spec.kernels = {producer, consumer};
    spec.dataflow_buffers = {dfb};
    spec.work_units = std::vector<WorkUnitSpec>{MakeMinimalWorkUnit("work_unit", node, {"producer", "consumer"})};

    EXPECT_THAT(
        [&] { MakeProgramFromSpec(spec); },
        ::testing::ThrowsMessage<std::runtime_error>(
            ::testing::HasSubstr("Kernel 'consumer' unpack_to_dest_mode references unknown DFB 'nonexistent_dfb'")));
}

TEST_F(ProgramSpecTestQuasar, DataFormatNotSupportedOnTargetArchitectureFails) {
    NodeCoord node{0, 0};

    ProgramSpec spec;
    spec.program_id = "test_program";

    auto producer = MakeMinimalDMKernel("producer");
    auto consumer = MakeMinimalComputeKernel("consumer");
    auto dfb = MakeMinimalDFB("dfb");

    // Legacy block-float format; not supported on Quasar.
    dfb.data_format_metadata = tt::DataFormat::Bfp8;

    BindDFBToKernel(producer, "dfb", "out", KernelSpec::DFBEndpointType::PRODUCER);
    BindDFBToKernel(consumer, "dfb", "in", KernelSpec::DFBEndpointType::CONSUMER);

    spec.kernels = {producer, consumer};
    spec.dataflow_buffers = {dfb};
    spec.work_units = std::vector<WorkUnitSpec>{MakeMinimalWorkUnit("work_unit", node, {"producer", "consumer"})};

    EXPECT_THAT(
        [&] { MakeProgramFromSpec(spec); },
        ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr("DFB 'dfb' has data format")));
}

// ============================================================================
// SECTION 3: WorkUnitSpec Validation Tests
// ============================================================================

TEST_F(ProgramSpecTestQuasar, EmptyWorkUnitSpecsFails) {
    ProgramSpec spec;
    spec.program_id = "test_program";

    auto kernel = MakeMinimalDMKernel("kernel");
    spec.kernels = {kernel};
    spec.work_units = std::vector<WorkUnitSpec>{};  // Empty!

    EXPECT_THAT(
        [&] { MakeProgramFromSpec(spec); },
        ::testing::ThrowsMessage<std::runtime_error>(
            ::testing::HasSubstr("Kernel 'kernel' is not referenced by any WorkUnitSpec")));
}

TEST_F(ProgramSpecTestQuasar, WorkUnitSpecWithNoKernelsFails) {
    NodeCoord node{0, 0};

    ProgramSpec spec;
    spec.program_id = "test_program";

    auto kernel = MakeMinimalDMKernel("kernel");
    spec.kernels = {kernel};

    WorkUnitSpec work_unit;
    work_unit.unique_id = "work_unit";
    work_unit.target_nodes = node;
    work_unit.kernels = {};  // No kernels!
    spec.work_units = std::vector<WorkUnitSpec>{work_unit};

    EXPECT_THAT(
        [&] { MakeProgramFromSpec(spec); },
        ::testing::ThrowsMessage<std::runtime_error>(
            ::testing::HasSubstr("Kernel 'kernel' is not referenced by any WorkUnitSpec")));
}

TEST_F(ProgramSpecTestQuasar, WorkUnitSpecReferencesUnknownKernelFails) {
    NodeCoord node{0, 0};

    ProgramSpec spec;
    spec.program_id = "test_program";

    auto kernel = MakeMinimalDMKernel("real_kernel");
    spec.kernels = {kernel};

    // WorkUnit references a kernel that doesn't exist
    spec.work_units = std::vector<WorkUnitSpec>{MakeMinimalWorkUnit("work_unit", node, {"nonexistent_kernel"})};

    EXPECT_THAT(
        [&] { MakeProgramFromSpec(spec); },
        ::testing::ThrowsMessage<std::runtime_error>(
            ::testing::HasSubstr("WorkUnitSpec 'work_unit' references unknown kernel 'nonexistent_kernel'")));
}

TEST_F(ProgramSpecTestQuasar, OverlappingWorkUnitSpecsFails) {
    // Two work_units cannot target overlapping nodes
    NodeCoord node{0, 0};

    ProgramSpec spec;
    spec.program_id = "test_program";

    auto kernel1 = MakeMinimalDMKernel("kernel1");
    auto kernel2 = MakeMinimalDMKernel("kernel2");
    spec.kernels = {kernel1, kernel2};

    // Both work_units target the same node
    spec.work_units = std::vector<WorkUnitSpec>{
        MakeMinimalWorkUnit("work_unit1", node, {"kernel1"}), MakeMinimalWorkUnit("work_unit2", node, {"kernel2"})};

    EXPECT_THAT(
        [&] { MakeProgramFromSpec(spec); },
        ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr("overlap in target nodes")));
}

TEST_F(ProgramSpecTestQuasar, KernelNotInAnyWorkUnitSpecFails) {
    NodeCoord node{0, 0};

    ProgramSpec spec;
    spec.program_id = "test_program";

    auto kernel1 = MakeMinimalDMKernel("kernel1");
    auto kernel2 = MakeMinimalDMKernel("kernel2");  // Not in any work_unit!
    spec.kernels = {kernel1, kernel2};

    spec.work_units = std::vector<WorkUnitSpec>{MakeMinimalWorkUnit("work_unit", node, {"kernel1"})};  // Only kernel1

    EXPECT_THAT(
        [&] { MakeProgramFromSpec(spec); },
        ::testing::ThrowsMessage<std::runtime_error>(
            ::testing::HasSubstr("Kernel 'kernel2' is not referenced by any WorkUnitSpec")));
}

TEST_F(ProgramSpecTestQuasar, WorkUnitExceedsDMCoreBudgetFails) {
    NodeCoord node{0, 0};

    ProgramSpec spec;
    spec.program_id = "test_program";

    // Create enough DM kernels to exceed the 8 DM core budget
    auto kernel1 = MakeMinimalDMKernel("dm1", 3);
    auto kernel2 = MakeMinimalDMKernel("dm2", 3);
    auto kernel3 = MakeMinimalDMKernel("dm3", 3);  // Total: 9 > 8

    spec.kernels = {kernel1, kernel2, kernel3};
    spec.work_units = std::vector<WorkUnitSpec>{MakeMinimalWorkUnit("work_unit", node, {"dm1", "dm2", "dm3"})};

    EXPECT_THAT(
        [&] { MakeProgramFromSpec(spec); },
        ::testing::ThrowsMessage<std::runtime_error>(
            ::testing::HasSubstr("WorkUnitSpec 'work_unit' requests 9 data movement cores")));
}

TEST_F(ProgramSpecTestQuasar, WorkUnitExceedsComputeCoreBudgetFails) {
    NodeCoord node{0, 0};

    ProgramSpec spec;
    spec.program_id = "test_program";

    // Create enough compute kernels to exceed the 4 Tensix core budget (2+4=6).
    // (Legal thread counts on Quasar are 1, 2, 4; 3 is explicitly disallowed.)
    auto kernel1 = MakeMinimalComputeKernel("compute1", 2);
    auto kernel2 = MakeMinimalComputeKernel("compute2", 4);

    spec.kernels = {kernel1, kernel2};
    spec.work_units = std::vector<WorkUnitSpec>{MakeMinimalWorkUnit("work_unit", node, {"compute1", "compute2"})};

    EXPECT_THAT(
        [&] { MakeProgramFromSpec(spec); },
        ::testing::ThrowsMessage<std::runtime_error>(
            ::testing::HasSubstr("WorkUnitSpec 'work_unit' needs 6 Tensix engines")));
}

TEST_F(ProgramSpecTestQuasar, WorkUnitWithMultipleComputeKernelsFails) {
    // A work_unit can have at most one compute kernel
    NodeCoord node{0, 0};

    ProgramSpec spec;
    spec.program_id = "test_program";

    auto compute1 = MakeMinimalComputeKernel("compute1");
    auto compute2 = MakeMinimalComputeKernel("compute2");

    spec.kernels = {compute1, compute2};
    spec.work_units = std::vector<WorkUnitSpec>{MakeMinimalWorkUnit("work_unit", node, {"compute1", "compute2"})};

    EXPECT_THAT(
        [&] { MakeProgramFromSpec(spec); },
        ::testing::ThrowsMessage<std::runtime_error>(
            ::testing::HasSubstr("WorkUnitSpec 'work_unit' has more than one compute kernel")));
}

TEST_F(ProgramSpecTestQuasar, LocalDFBProducerConsumerWorkUnitMembershipMismatchFails) {
    // A local DFB requires its producer and consumer kernels to share IDENTICAL
    // WorkUnitSpec membership.
    NodeCoord node0{0, 0};
    NodeCoord node1{1, 0};

    ProgramSpec spec;
    spec.program_id = "test_program";

    auto producer = MakeMinimalDMKernel("producer");
    auto consumer = MakeMinimalDMKernel("consumer");
    auto dfb = MakeMinimalDFB("dfb");

    BindDFBToKernel(producer, "dfb", "out", KernelSpec::DFBEndpointType::PRODUCER);
    BindDFBToKernel(consumer, "dfb", "in", KernelSpec::DFBEndpointType::CONSUMER);

    spec.kernels = {producer, consumer};
    spec.dataflow_buffers = {dfb};
    // Producer is on work_unit_0 only; consumer is on both work_units — membership doesn't match.
    spec.work_units = std::vector<WorkUnitSpec>{
        MakeMinimalWorkUnit("work_unit_0", node0, {"producer", "consumer"}),
        MakeMinimalWorkUnit("work_unit_1", node1, {"consumer"}),
    };

    EXPECT_THAT(
        [&] { MakeProgramFromSpec(spec); },
        ::testing::ThrowsMessage<std::runtime_error>(
            ::testing::HasSubstr("do not share identical WorkUnitSpec membership")));
}
// ============================================================================
// SECTION 4: Programs Creation Tests
// ============================================================================
// These verify that valid ProgramSpec configurations produce a Program without throwing.
// They exercise the full MakeProgramFromSpec pipeline, but only on mock device.
//
// Coverage gaps (JIT compilation, device-side execution) are covered by HW tests.
// (see test_program_spec_hw.cpp)

TEST_F(ProgramSpecTestQuasar, MinimalValidProgramSpecSucceeds) {
    ProgramSpec spec = MakeMinimalValidProgramSpec();

    EXPECT_NO_THROW(MakeProgramFromSpec(spec));
}

TEST_F(ProgramSpecTestQuasar, DMOnlyProgramSucceeds) {
    // A program with only DM kernels (no compute)
    NodeCoord node{0, 0};

    ProgramSpec spec;
    spec.program_id = "dm_only_program";

    auto producer = MakeMinimalDMKernel("producer");
    auto consumer = MakeMinimalDMKernel("consumer");
    auto dfb = MakeMinimalDFB("dfb");
    // No data_format_metadata needed for DM-only DFBs

    BindDFBToKernel(producer, "dfb", "out", KernelSpec::DFBEndpointType::PRODUCER);
    BindDFBToKernel(consumer, "dfb", "in", KernelSpec::DFBEndpointType::CONSUMER);

    spec.kernels = {producer, consumer};
    spec.dataflow_buffers = {dfb};
    spec.work_units = std::vector<WorkUnitSpec>{MakeMinimalWorkUnit("work_unit", node, {"producer", "consumer"})};

    EXPECT_NO_THROW(MakeProgramFromSpec(spec));
}

TEST_F(ProgramSpecTestQuasar, MultiNodeProgramSucceeds) {
    // A program spanning multiple nodes
    NodeRange nodes{{0, 0}, {1, 1}};  // 2x2 grid

    ProgramSpec spec;
    spec.program_id = "multi_node_program";

    auto producer = MakeMinimalDMKernel("producer");
    auto consumer = MakeMinimalDMKernel("consumer");
    auto dfb = MakeMinimalDFB("dfb");

    BindDFBToKernel(producer, "dfb", "out", KernelSpec::DFBEndpointType::PRODUCER);
    BindDFBToKernel(consumer, "dfb", "in", KernelSpec::DFBEndpointType::CONSUMER);

    spec.kernels = {producer, consumer};
    spec.dataflow_buffers = {dfb};
    spec.work_units = std::vector<WorkUnitSpec>{MakeMinimalWorkUnit("work_unit", nodes, {"producer", "consumer"})};

    EXPECT_NO_THROW(MakeProgramFromSpec(spec));
}

TEST_F(ProgramSpecTestQuasar, MultipleWorkUnitsOnDifferentNodesSucceeds) {
    // Multiple work_units on non-overlapping nodes
    NodeCoord node0{0, 0};
    NodeCoord node1{1, 0};
    NodeRangeSet all_nodes(std::set<NodeRange>{NodeRange{node0, node0}, NodeRange{node1, node1}});

    ProgramSpec spec;
    spec.program_id = "multi_work_unit_program";

    // Kernels span both nodes
    auto kernel = MakeMinimalDMKernel("kernel");
    spec.kernels = {kernel};

    // Two work_units, each on a different node
    spec.work_units = std::vector<WorkUnitSpec>{
        MakeMinimalWorkUnit("work_unit0", node0, {"kernel"}), MakeMinimalWorkUnit("work_unit1", node1, {"kernel"})};

    EXPECT_NO_THROW(MakeProgramFromSpec(spec));
}

TEST_F(ProgramSpecTestQuasar, MaxDMThreadsSucceeds) {
    // Use exactly 6 DM threads (the maximum available to the user)
    NodeCoord node{0, 0};

    ProgramSpec spec;
    spec.program_id = "max_dm_threads";

    auto producer = MakeMinimalDMKernel("producer", 3);
    auto consumer = MakeMinimalDMKernel("consumer", 3);  // Total: 6
    auto dfb = MakeMinimalDFB("dfb");
    dfb.num_entries = 9;  // must be a multiple of the number of threads

    BindDFBToKernel(producer, "dfb", "out", KernelSpec::DFBEndpointType::PRODUCER);
    BindDFBToKernel(consumer, "dfb", "in", KernelSpec::DFBEndpointType::CONSUMER);

    spec.kernels = {producer, consumer};
    spec.dataflow_buffers = {dfb};
    spec.work_units = std::vector<WorkUnitSpec>{MakeMinimalWorkUnit("work_unit", node, {"producer", "consumer"})};

    EXPECT_NO_THROW(MakeProgramFromSpec(spec));
}

TEST_F(ProgramSpecTestQuasar, MaxComputeThreadsSucceeds) {
    // Use exactly 4 compute threads (the maximum available)
    NodeCoord node{0, 0};

    ProgramSpec spec;
    spec.program_id = "max_compute_threads";

    auto dm = MakeMinimalDMKernel("dm");
    auto compute = MakeMinimalComputeKernel("compute", 4);  // Max threads

    auto dfb = MakeMinimalDFB("dfb");
    dfb.data_format_metadata = tt::DataFormat::Float16_b;
    dfb.num_entries = 4;  // must be a multiple of the number of threads

    BindDFBToKernel(dm, "dfb", "out", KernelSpec::DFBEndpointType::PRODUCER);
    BindDFBToKernel(compute, "dfb", "in", KernelSpec::DFBEndpointType::CONSUMER);

    spec.kernels = {dm, compute};
    spec.dataflow_buffers = {dfb};
    spec.work_units = std::vector<WorkUnitSpec>{MakeMinimalWorkUnit("work_unit", node, {"dm", "compute"})};

    EXPECT_NO_THROW(MakeProgramFromSpec(spec));
}

TEST_F(ProgramSpecTestQuasar, MultipleDFBsSucceeds) {
    NodeCoord node{0, 0};

    ProgramSpec spec;
    spec.program_id = "multi_dfb_program";

    auto producer = MakeMinimalDMKernel("producer");
    auto consumer = MakeMinimalComputeKernel("consumer");

    auto dfb1 = MakeMinimalDFB("dfb1");
    dfb1.data_format_metadata = tt::DataFormat::Float16_b;
    auto dfb2 = MakeMinimalDFB("dfb2");
    dfb2.data_format_metadata = tt::DataFormat::Int8;

    BindDFBToKernel(producer, "dfb1", "out1", KernelSpec::DFBEndpointType::PRODUCER);
    BindDFBToKernel(producer, "dfb2", "out2", KernelSpec::DFBEndpointType::PRODUCER);
    BindDFBToKernel(consumer, "dfb1", "in1", KernelSpec::DFBEndpointType::CONSUMER);
    BindDFBToKernel(consumer, "dfb2", "in2", KernelSpec::DFBEndpointType::CONSUMER);

    spec.kernels = {producer, consumer};
    spec.dataflow_buffers = {dfb1, dfb2};
    spec.work_units = std::vector<WorkUnitSpec>{MakeMinimalWorkUnit("work_unit", node, {"producer", "consumer"})};

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
    spec.kernels[0].runtime_arguments_schema.num_runtime_varargs = 3;
    spec.kernels[0].runtime_arguments_schema.num_common_runtime_varargs = 2;

    EXPECT_NO_THROW(MakeProgramFromSpec(spec));
}

TEST_F(ProgramSpecTestQuasar, VarargPerNodeOverlapFails) {
    // Rule: overlapping entries in num_runtime_varargs_per_node are an error, even when
    // their counts agree. Overlap suggests a user mistake.
    using NumVarargsPerNode = KernelSpec::RuntimeArgSchema::NumVarargsPerNode;
    NodeCoord node_a{0, 0};
    NodeCoord node_b{1, 0};
    NodeRangeSet both{std::vector<NodeRange>{NodeRange{node_a, node_a}, NodeRange{node_b, node_b}}};

    ProgramSpec spec;
    spec.program_id = "vararg_overlap_test";
    auto kernel = MakeMinimalDMKernel("dm_kernel");
    kernel.runtime_arguments_schema.num_runtime_varargs_per_node =
        NumVarargsPerNode{{both, 3}, {node_a, 3}};  // node_a listed twice
    spec.kernels = {kernel};
    spec.work_units = std::vector<WorkUnitSpec>{MakeMinimalWorkUnit("work_unit_0", both, {"dm_kernel"})};

    EXPECT_THAT(
        [&] { MakeProgramFromSpec(spec); },
        ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr("overlapping entries")));
}

// ============================================================================
// SECTION 5: Edge Cases and Boundary Tests
// ============================================================================

TEST_F(ProgramSpecTestQuasar, NodeRangeSetTargetNodesSucceeds) {
    // Test with NodeRangeSet (multiple disjoint ranges)
    NodeRangeSet nodes(std::set<NodeRange>{NodeRange{{0, 0}, {0, 1}}, NodeRange{{2, 0}, {2, 1}}});

    ProgramSpec spec;
    spec.program_id = "range_set_program";

    auto producer = MakeMinimalDMKernel("producer");
    auto consumer = MakeMinimalDMKernel("consumer");
    auto dfb = MakeMinimalDFB("dfb");

    BindDFBToKernel(producer, "dfb", "out", KernelSpec::DFBEndpointType::PRODUCER);
    BindDFBToKernel(consumer, "dfb", "in", KernelSpec::DFBEndpointType::CONSUMER);

    spec.kernels = {producer, consumer};
    spec.dataflow_buffers = {dfb};
    spec.work_units = std::vector<WorkUnitSpec>{MakeMinimalWorkUnit("work_unit", nodes, {"producer", "consumer"})};

    EXPECT_NO_THROW(MakeProgramFromSpec(spec));
}

TEST_F(ProgramSpecTestQuasar, SourceCodeKernelSucceeds) {
    ProgramSpec spec = MakeMinimalValidProgramSpec();

    // Change to inline source code
    spec.kernels[0].source = KernelSpec::SourceCode{"void kernel_main() {}"};

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
//    equivalent ProgramSpecs, depending on the order of kernels and work_units.
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
// regardless of kernel and work_unit orderings.
TEST_F(ProgramSpecTestQuasar, BacktrackingSolverFindsAssignment_RegardlessOfOrder) {
    // This test verifies that semantically identical ProgramSpecs succeed
    // regardless of the order of:
    //  - work_units in spec.work_units
    //  - kernels within each work_unit
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

    auto k_a = MakeMinimalDMKernel("k_a", 3);
    auto k_ab = MakeMinimalDMKernel("k_ab", 3);
    auto k_bc = MakeMinimalDMKernel("k_bc", 3);
    auto k_c = MakeMinimalDMKernel("k_c", 3);

    auto work_unit_a1 = MakeMinimalWorkUnit("work_unit_a1", node_a, {"k_a", "k_ab"});
    auto work_unit_b1 = MakeMinimalWorkUnit("work_unit_b1", node_b, {"k_ab", "k_bc"});
    auto work_unit_c1 = MakeMinimalWorkUnit("work_unit_c1", node_c, {"k_bc", "k_c"});
    auto work_unit_c2 = MakeMinimalWorkUnit("work_unit_c2", node_c, {"k_c", "k_bc"});

    // Helper to create a ProgramSpec with a given id and work_unit ordering
    auto make_spec =
        [&](const std::string& id, std::vector<KernelSpec> kernels, const std::vector<WorkUnitSpec>& work_units) {
            ProgramSpec spec;
            spec.program_id = id;
            spec.kernels = std::move(kernels);
            spec.work_units = work_units;
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

    // All 6 possible permutations of work_unit orderings (using c1)
    std::vector<std::vector<WorkUnitSpec>> work_unit_permutations1 = {
        {work_unit_a1, work_unit_b1, work_unit_c1},
        {work_unit_a1, work_unit_c1, work_unit_b1},
        {work_unit_b1, work_unit_c1, work_unit_a1},
        {work_unit_b1, work_unit_a1, work_unit_c1},
        {work_unit_c1, work_unit_a1, work_unit_b1},
        {work_unit_c1, work_unit_b1, work_unit_a1}};

    // All 6 possible permutations of work_unit orderings (using c2)
    std::vector<std::vector<WorkUnitSpec>> work_unit_permutations2 = {
        {work_unit_a1, work_unit_b1, work_unit_c2},
        {work_unit_a1, work_unit_c2, work_unit_b1},
        {work_unit_b1, work_unit_c2, work_unit_a1},
        {work_unit_b1, work_unit_a1, work_unit_c2},
        {work_unit_c2, work_unit_a1, work_unit_b1},
        {work_unit_c2, work_unit_b1, work_unit_a1}};

    // All kernel permutations should succeed with all work_unit permutations.
    for (const auto& kernel_perm : kernel_permutations) {
        for (const auto& work_unit_perm : work_unit_permutations1) {
            EXPECT_NO_THROW(MakeProgramFromSpec(make_spec("", kernel_perm, work_unit_perm)));
        }
    }
    for (const auto& kernel_perm : kernel_permutations) {
        for (const auto& work_unit_perm : work_unit_permutations2) {
            EXPECT_NO_THROW(MakeProgramFromSpec(make_spec("", kernel_perm, work_unit_perm)));
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

    auto kernel_a = MakeMinimalDMKernel("kernel_a", 3);
    auto kernel_b = MakeMinimalDMKernel("kernel_b", 3);
    auto kernel_c = MakeMinimalDMKernel("kernel_c", 3);

    spec.kernels = {kernel_a, kernel_b, kernel_c};

    spec.work_units = std::vector<WorkUnitSpec>{
        MakeMinimalWorkUnit("work_unit_00", node_00, {"kernel_a", "kernel_b"}),
        MakeMinimalWorkUnit("work_unit_01", node_01, {"kernel_a", "kernel_c"}),
        MakeMinimalWorkUnit("work_unit_02", node_02, {"kernel_b", "kernel_c"}),
    };

    // EXPECTED BEHAVIOR: FAILS due to simplifying assumption violation.
    EXPECT_THAT(
        [&] { MakeProgramFromSpec(spec); },
        ::testing::ThrowsMessage<std::runtime_error>(
            ::testing::HasSubstr("Failed to find valid processor assignments for DM kernels")));
}

// ============================================================================
// SECTION 7: Aggregate Type Enforcement Tests
// ============================================================================
//
// DESIGN DECISION: All *Spec types must remain aggregates (POD-like structs).
//
// Rationale:
//   - Aggregates support designated initializers, making code self-documenting
//   - Prevents "constructor creep" where types accumulate convenience constructors
//
// What breaks aggregate status:
//   - User-declared constructors (including default/copy/move)
//   - Private/protected non-static data members
//   - Virtual functions
//   - Virtual/private/protected base classes
//
// By convention, I would strongly prefer to avoid adding member functions to Spec types.
// Extensions should be added via free functions rather than member methods, to prevent
// cruft accumulation. This will have to be enforced via code review, however.

// Compile-time enforcement: all Spec types must be aggregates
static_assert(
    std::is_aggregate_v<ProgramSpec>, "ProgramSpec must remain an aggregate to support designated initializers");
static_assert(
    std::is_aggregate_v<WorkUnitSpec>, "WorkUnitSpec must remain an aggregate to support designated initializers");
static_assert(
    std::is_aggregate_v<KernelSpec>, "KernelSpec must remain an aggregate to support designated initializers");
static_assert(
    std::is_aggregate_v<DataflowBufferSpec>,
    "DataflowBufferSpec must remain an aggregate to support designated initializers");
static_assert(
    std::is_aggregate_v<SemaphoreSpec>, "SemaphoreSpec must remain an aggregate to support designated initializers");
static_assert(
    std::is_aggregate_v<ComputeConfiguration>,
    "ComputeConfiguration must remain an aggregate to support designated initializers");
static_assert(
    std::is_aggregate_v<DataMovementConfiguration>,
    "DataMovementConfiguration must remain an aggregate to support designated initializers");
static_assert(
    std::is_aggregate_v<DataMovementConfiguration::Gen1DataMovementConfig>,
    "Gen1DataMovementConfig must remain an aggregate");
static_assert(
    std::is_aggregate_v<DataMovementConfiguration::Gen2DataMovementConfig>,
    "Gen2DataMovementConfig must remain an aggregate");
static_assert(
    std::is_aggregate_v<KernelSpec::CompilerOptions>,
    "CompilerOptions must remain an aggregate to support designated initializers");
static_assert(
    std::is_aggregate_v<KernelSpec::DFBBinding>,
    "DFBBinding must remain an aggregate to support designated initializers");
static_assert(
    std::is_aggregate_v<KernelSpec::SemaphoreBinding>,
    "SemaphoreBinding must remain an aggregate to support designated initializers");
static_assert(
    std::is_aggregate_v<KernelSpec::RuntimeArgSchema>,
    "RuntimeArgSchema must remain an aggregate to support designated initializers");
static_assert(
    std::is_aggregate_v<RemoteDataflowBufferSpec>,
    "RemoteDataflowBufferSpec must remain an aggregate to support designated initializers");

// These tests document the intended construction pattern using designated initializers.
// They serve as living documentation and will fail to compile if aggregate status is broken.

TEST(AggregateSpecTypes, KernelSpecDesignatedInitializers) {
    // Demonstrates constructing KernelSpec with designated initializers
    KernelSpec dm_kernel{
        .unique_id = "my_dm_kernel",
        .source = KernelSpec::SourceCode{"void kernel_main() {}"},
        .num_threads = 2,
        .config_spec =
            DataMovementConfiguration{
                .gen2_data_movement_config = DataMovementConfiguration::Gen2DataMovementConfig{},
            },
    };

    EXPECT_EQ(dm_kernel.unique_id, "my_dm_kernel");
    EXPECT_EQ(dm_kernel.num_threads, 2);
    EXPECT_TRUE(dm_kernel.is_dm_kernel());

    KernelSpec compute_kernel{
        .unique_id = "my_compute_kernel",
        .source = KernelSpec::SourceCode{"void kernel_main() {}"},
        .num_threads = 4,
        .compiler_options =
            KernelSpec::CompilerOptions{
                .defines = {{"MY_DEFINE", "42"}},
                .opt_level = tt::tt_metal::KernelBuildOptLevel::O3,
            },
        .config_spec =
            ComputeConfiguration{
                .math_fidelity = MathFidelity::LoFi,
                .fp32_dest_acc_en = true,
            },
    };

    EXPECT_EQ(compute_kernel.unique_id, "my_compute_kernel");
    EXPECT_TRUE(compute_kernel.is_compute_kernel());
}

TEST(AggregateSpecTypes, DataflowBufferSpecDesignatedInitializers) {
    // Demonstrates constructing DataflowBufferSpec with designated initializers
    DataflowBufferSpec dfb{
        .unique_id = "my_dfb",
        .entry_size = 2048,
        .num_entries = 4,
        .data_format_metadata = tt::DataFormat::Float16_b,
    };

    EXPECT_EQ(dfb.unique_id, "my_dfb");
    EXPECT_EQ(dfb.entry_size, 2048u);
    EXPECT_EQ(dfb.num_entries, 4u);

    // DFB with advanced options
    DataflowBufferSpec borrowed_dfb{
        .unique_id = "borrowed_dfb",
        .entry_size = 1024,
        .num_entries = 8,
        .uses_borrowed_memory = true,
        .disable_implicit_sync = true,
    };

    EXPECT_TRUE(borrowed_dfb.uses_borrowed_memory);
    EXPECT_TRUE(borrowed_dfb.disable_implicit_sync);
}

TEST(AggregateSpecTypes, WorkUnitSpecDesignatedInitializers) {
    // Demonstrates constructing WorkUnitSpec with designated initializers
    WorkUnitSpec work_unit{
        .unique_id = "my_work_unit",
        .kernels = {"kernel1", "kernel2"},
        .target_nodes = NodeCoord{0, 0},
    };

    EXPECT_EQ(work_unit.unique_id, "my_work_unit");
    EXPECT_EQ(work_unit.kernels.size(), 2u);
}

TEST(AggregateSpecTypes, RuntimeArgSchemaDesignatedInitializers) {
    // Named RTAs + CRTAs + scalar vararg counts, all via designated initializers.
    KernelSpec::RuntimeArgSchema schema{
        .named_runtime_args = {"input_ptr", "output_ptr"},
        .named_common_runtime_args = {"tile_count"},
        .num_runtime_varargs = 4,
        .num_common_runtime_varargs = 2,
    };

    EXPECT_EQ(schema.named_runtime_args.size(), 2u);
    EXPECT_EQ(schema.named_common_runtime_args.size(), 1u);
    EXPECT_EQ(schema.num_runtime_varargs, 4u);
    EXPECT_EQ(schema.num_common_runtime_varargs, 2u);
    EXPECT_FALSE(schema.num_runtime_varargs_per_node.has_value());
}

TEST(AggregateSpecTypes, RuntimeArgSchemaPerNodeOverrideDesignatedInitializers) {
    // Per-node override path (advanced): ensure designated-init through std::optional works.
    using NumVarargsPerNode = KernelSpec::RuntimeArgSchema::NumVarargsPerNode;
    KernelSpec::RuntimeArgSchema schema{
        .num_runtime_varargs_per_node = NumVarargsPerNode{{NodeCoord{0, 0}, 4}, {NodeCoord{1, 0}, 7}},
    };

    ASSERT_TRUE(schema.num_runtime_varargs_per_node.has_value());
    EXPECT_EQ(schema.num_runtime_varargs_per_node->size(), 2u);
    EXPECT_EQ(schema.num_runtime_varargs, 0u);  // scalar left at default in this example
}

TEST(AggregateSpecTypes, KernelSpecNamedRuntimeArgsDesignatedInitializers) {
    KernelSpec k{
        .unique_id = "k",
        .source = KernelSpec::SourceCode{"void kernel_main() {}"},
        .runtime_arguments_schema =
            KernelSpec::RuntimeArgSchema{
                .named_runtime_args = {"input_ptr"},
            },
        .config_spec =
            DataMovementConfiguration{
                .gen2_data_movement_config = DataMovementConfiguration::Gen2DataMovementConfig{},
            },
    };
    EXPECT_EQ(k.runtime_arguments_schema.named_runtime_args.size(), 1u);
}

TEST(AggregateSpecTypes, SemaphoreSpecDesignatedInitializers) {
    // Demonstrates constructing SemaphoreSpec with designated initializers
    SemaphoreSpec sem{
        .unique_id = "my_semaphore",
        .target_nodes = NodeCoord{0, 0},
        .initial_value = 7,
    };

    EXPECT_EQ(sem.unique_id, "my_semaphore");
    EXPECT_EQ(sem.initial_value, 7u);
}

TEST(AggregateSpecTypes, ProgramSpecDesignatedInitializers) {
    // Demonstrates constructing a complete ProgramSpec with designated initializers
    ProgramSpec spec{
        .program_id = "my_program",
        .kernels =
            {
                KernelSpec{
                    .unique_id = "producer",
                    .source = KernelSpec::SourceCode{"void kernel_main() {}"},
                    .dfb_bindings =
                        {
                            KernelSpec::DFBBinding{
                                .dfb_spec_name = "dfb",
                                .local_accessor_name = "out",
                                .endpoint_type = KernelSpec::DFBEndpointType::PRODUCER,
                                .access_pattern = DFBAccessPattern::STRIDED,
                            },
                        },
                    .config_spec =
                        DataMovementConfiguration{
                            .gen2_data_movement_config = DataMovementConfiguration::Gen2DataMovementConfig{},
                        },
                },
                KernelSpec{
                    .unique_id = "consumer",
                    .source = KernelSpec::SourceCode{"void kernel_main() {}"},
                    .dfb_bindings =
                        {
                            KernelSpec::DFBBinding{
                                .dfb_spec_name = "dfb",
                                .local_accessor_name = "in",
                                .endpoint_type = KernelSpec::DFBEndpointType::CONSUMER,
                                .access_pattern = DFBAccessPattern::STRIDED,
                            },
                        },
                    .config_spec = ComputeConfiguration{},
                },
            },
        .dataflow_buffers =
            {
                DataflowBufferSpec{
                    .unique_id = "dfb",
                    .entry_size = 1024,
                    .num_entries = 2,
                    .data_format_metadata = tt::DataFormat::Float16_b,
                },
            },
        .work_units =
            {
                WorkUnitSpec{
                    .unique_id = "work_unit",
                    .kernels = {"producer", "consumer"},
                    .target_nodes = NodeCoord{0, 0},
                },
            },
    };

    EXPECT_EQ(spec.program_id, "my_program");
    EXPECT_EQ(spec.kernels.size(), 2u);
    EXPECT_EQ(spec.dataflow_buffers.size(), 1u);
    EXPECT_EQ(spec.work_units.size(), 1u);
}

TEST(AggregateSpecTypes, NestedStructsDesignatedInitializers) {
    // Demonstrates constructing nested configuration structs with designated initializers
    KernelSpec::DFBBinding binding{
        .dfb_spec_name = "my_dfb",
        .local_accessor_name = "accessor",
        .endpoint_type = KernelSpec::DFBEndpointType::PRODUCER,
        .access_pattern = DFBAccessPattern::ALL,
    };
    EXPECT_EQ(binding.dfb_spec_name, "my_dfb");

    KernelSpec::SemaphoreBinding sem_binding{
        .semaphore_spec_name = "my_sem",
        .accessor_name = "sem_accessor",
    };
    EXPECT_EQ(sem_binding.semaphore_spec_name, "my_sem");

    KernelSpec::CompilerOptions opts{
        .include_paths = {"/path/to/include"},
        .defines = {{"DEBUG", "1"}, {"VERSION", "2"}},
        .opt_level = tt::tt_metal::KernelBuildOptLevel::O0,
    };
    EXPECT_EQ(opts.defines.size(), 2u);

    DataMovementConfiguration::Gen1DataMovementConfig gen1{
        .processor = tt::tt_metal::DataMovementProcessor::RISCV_1,
        .noc = tt::tt_metal::NOC::RISCV_1_default,
        .noc_mode = tt::tt_metal::NOC_MODE::DM_DEDICATED_NOC,
    };
    EXPECT_EQ(gen1.processor, tt::tt_metal::DataMovementProcessor::RISCV_1);

    RemoteDataflowBufferSpec remote_dfb{
        .dfb_spec =
            DataflowBufferSpec{
                .unique_id = "remote_dfb",
                .entry_size = 1024,
                .num_entries = 2,
            },
        .producer_consumer_map = {{NodeCoord{0, 0}, NodeCoord{1, 0}}},
    };
    EXPECT_EQ(remote_dfb.producer_consumer_map.size(), 1u);
    EXPECT_EQ(remote_dfb.dfb_spec.unique_id, "remote_dfb");
}

// ============================================================================
// SECTION 8: Gen1 (WH/BH) Tests
// ============================================================================

// Test fixture for ProgramSpec on Wormhole - uses WORMHOLE_B0 mock device
class ProgramSpecTestGen1 : public ::testing::Test {
protected:
    void SetUp() override { experimental::configure_mock_mode(tt::ARCH::WORMHOLE_B0, 1); }
    void TearDown() override { experimental::disable_mock_mode(); }
};

TEST_F(ProgramSpecTestGen1, MinimalValidProgramSpecSucceeds) {
    ProgramSpec spec = MakeMinimalGen1ValidProgramSpec();
    EXPECT_NO_THROW(MakeProgramFromSpec(spec));
}

TEST_F(ProgramSpecTestGen1, DMOnlyProgramSucceeds) {
    // Two DM kernels on different processors (RISCV_0 producer, RISCV_1 consumer)
    NodeCoord node{0, 0};

    ProgramSpec spec;
    spec.program_id = "dm_only_program";

    auto producer = MakeMinimalGen1DMKernel("producer", DataMovementProcessor::RISCV_0);
    auto consumer = MakeMinimalGen1DMKernel("consumer", DataMovementProcessor::RISCV_1);
    auto dfb = MakeMinimalDFB("dfb");

    BindDFBToKernel(producer, "dfb", "out", KernelSpec::DFBEndpointType::PRODUCER);
    BindDFBToKernel(consumer, "dfb", "in", KernelSpec::DFBEndpointType::CONSUMER);

    spec.kernels = {producer, consumer};
    spec.dataflow_buffers = {dfb};
    spec.work_units = std::vector<WorkUnitSpec>{MakeMinimalWorkUnit("work_unit", node, {"producer", "consumer"})};

    EXPECT_NO_THROW(MakeProgramFromSpec(spec));
}

TEST_F(ProgramSpecTestGen1, TwoDMKernelsDifferentProcessorsSucceeds) {
    // RISCV_0 and RISCV_1 on the same node — should succeed
    NodeCoord node{0, 0};

    ProgramSpec spec;
    spec.program_id = "two_dm_program";

    auto k0 = MakeMinimalGen1DMKernel("k0", DataMovementProcessor::RISCV_0);
    auto k1 = MakeMinimalGen1DMKernel("k1", DataMovementProcessor::RISCV_1);

    spec.kernels = {k0, k1};
    spec.work_units = std::vector<WorkUnitSpec>{MakeMinimalWorkUnit("work_unit", node, {"k0", "k1"})};

    EXPECT_NO_THROW(MakeProgramFromSpec(spec));
}

TEST_F(ProgramSpecTestGen1, MultiThreadedDMKernelFails) {
    NodeCoord node{0, 0};

    ProgramSpec spec;
    spec.program_id = "test_program";

    auto kernel = MakeMinimalGen1DMKernel("dm_kernel");
    kernel.num_threads = 2;

    spec.kernels = {kernel};
    spec.work_units = std::vector<WorkUnitSpec>{MakeMinimalWorkUnit("work_unit", node, {"dm_kernel"})};

    EXPECT_THAT(
        [&] { MakeProgramFromSpec(spec); },
        ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr("does not support multi-threaded kernels")));
}

TEST_F(ProgramSpecTestGen1, MultiThreadedComputeKernelFails) {
    NodeCoord node{0, 0};

    ProgramSpec spec;
    spec.program_id = "test_program";

    auto kernel = MakeMinimalComputeKernel("compute_kernel");
    kernel.num_threads = 2;

    spec.kernels = {kernel};
    spec.work_units = std::vector<WorkUnitSpec>{MakeMinimalWorkUnit("work_unit", node, {"compute_kernel"})};

    EXPECT_THAT(
        [&] { MakeProgramFromSpec(spec); },
        ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr("does not support multi-threaded kernels")));
}

TEST_F(ProgramSpecTestGen1, DMKernelWithGen2ConfigFails) {
    // On gen1, a DM kernel that only has Gen2DataMovementConfig must be rejected
    NodeCoord node{0, 0};

    ProgramSpec spec;
    spec.program_id = "test_program";

    // MakeMinimalDMKernel produces a gen2 (Quasar) DM config
    auto kernel = MakeMinimalDMKernel("dm_kernel");

    spec.kernels = {kernel};
    spec.work_units = std::vector<WorkUnitSpec>{MakeMinimalWorkUnit("work_unit", node, {"dm_kernel"})};

    EXPECT_THAT(
        [&] { MakeProgramFromSpec(spec); },
        ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr("must specify a Gen1 DM config")));
}

TEST_F(ProgramSpecTestGen1, ProcessorConflictFails) {
    // Two DM kernels both targeting RISCV_0 on the same node
    NodeCoord node{0, 0};

    ProgramSpec spec;
    spec.program_id = "test_program";

    auto k0 = MakeMinimalGen1DMKernel("k0", DataMovementProcessor::RISCV_0);
    auto k1 = MakeMinimalGen1DMKernel("k1", DataMovementProcessor::RISCV_0);  // conflict

    spec.kernels = {k0, k1};
    spec.work_units = std::vector<WorkUnitSpec>{MakeMinimalWorkUnit("work_unit", node, {"k0", "k1"})};

    EXPECT_THAT(
        [&] { MakeProgramFromSpec(spec); },
        ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr("both claim the same DM processor")));
}

// WH N150 mock grid reference (wormhole_N150.yaml, harvest_mask=0x40 = 1 row harvested):
//   - Fast dispatch: compute_grid = 8x8 (y in [0,7]; one row reserved for dispatch)
//   - Slow dispatch: compute_grid = 8x9 (y in [0,8]; full logical tensix grid, no rows reserved)
//
// The apparent grid size is different in slow dispatch vs. fast dispatch mode. CI runs with
// both, so choose OOB coordinates that will fail in both cases.
//
// These tests use the WH mock device, not real hardware.

TEST_F(ProgramSpecTestGen1, KernelTargetsNodeBeyondGridYFails) {
    // y=9 is just outside the 9-row slow-dispatch grid (also outside the 8-row fast-dispatch grid).
    const NodeCoord oob_node{0, 9};

    ProgramSpec spec;
    spec.program_id = "test_program";

    auto kernel = MakeMinimalGen1DMKernel("dm_kernel");
    spec.kernels = {kernel};
    spec.work_units = std::vector<WorkUnitSpec>{MakeMinimalWorkUnit("work_unit", oob_node, {"dm_kernel"})};

    EXPECT_THAT(
        [&] { MakeProgramFromSpec(spec); },
        ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr("out of bounds")));
}

TEST_F(ProgramSpecTestGen1, KernelTargetsOutOfBoundsNodeFails) {
    // x=8 is just outside the 8-column grid (same in fast and slow dispatch).
    const NodeCoord oob_node{8, 0};

    ProgramSpec spec;
    spec.program_id = "test_program";

    auto kernel = MakeMinimalGen1DMKernel("dm_kernel");
    spec.kernels = {kernel};
    spec.work_units = std::vector<WorkUnitSpec>{MakeMinimalWorkUnit("work_unit", oob_node, {"dm_kernel"})};

    EXPECT_THAT(
        [&] { MakeProgramFromSpec(spec); },
        ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr("out of bounds")));
}

TEST_F(ProgramSpecTestGen1, SemaphoreBoundToComputeKernelFailsOnGen1) {
    // On WH/BH, compute kernels cannot participate in semaphore signalling.
    // (The corresponding Quasar test asserts this is allowed there.)
    ProgramSpec spec = MakeMinimalGen1ValidProgramSpec();

    SemaphoreSpec sem;
    sem.unique_id = "sem_0";
    sem.target_nodes = NodeCoord{0, 0};
    spec.semaphores = {sem};

    // kernels[1] is the compute kernel in MakeMinimalGen1ValidProgramSpec
    ASSERT_TRUE(spec.kernels[1].is_compute_kernel());
    spec.kernels[1].semaphore_bindings = {
        KernelSpec::SemaphoreBinding{.semaphore_spec_name = "sem_0", .accessor_name = "done_flag"}};

    EXPECT_THAT(
        [&] { MakeProgramFromSpec(spec); },
        ::testing::ThrowsMessage<std::runtime_error>(
            ::testing::HasSubstr("has semaphore bindings, but it is a compute kernel.")));
}

TEST_F(ProgramSpecTestGen1, SemaphoreBoundToDMKernelSucceedsOnGen1) {
    // Sanity check: binding a semaphore to a DM kernel on WH/BH is allowed.
    ProgramSpec spec = MakeMinimalGen1ValidProgramSpec();

    SemaphoreSpec sem;
    sem.unique_id = "sem_0";
    sem.target_nodes = NodeCoord{0, 0};
    spec.semaphores = {sem};

    // kernels[0] is the DM kernel in MakeMinimalGen1ValidProgramSpec
    ASSERT_TRUE(spec.kernels[0].is_dm_kernel());
    spec.kernels[0].semaphore_bindings = {
        KernelSpec::SemaphoreBinding{.semaphore_spec_name = "sem_0", .accessor_name = "done_flag"}};

    EXPECT_NO_THROW(MakeProgramFromSpec(spec));
}

TEST_F(ProgramSpecTestGen1, SemaphoresWithNonZeroInitialValueSucceedOnGen1) {
    // Gen1 accepts non-zero initial values (only Quasar rejects them).
    ProgramSpec spec = MakeMinimalGen1ValidProgramSpec();

    SemaphoreSpec sem;
    sem.unique_id = "sem_0";
    sem.target_nodes = NodeCoord{0, 0};
    sem.initial_value = 3;
    spec.semaphores = {sem};

    spec.kernels[0].semaphore_bindings = {
        KernelSpec::SemaphoreBinding{.semaphore_spec_name = "sem_0", .accessor_name = "done_flag"}};

    EXPECT_NO_THROW(MakeProgramFromSpec(spec));
}

TEST_F(ProgramSpecTestGen1, DuplicateKernelNameFails) {
    // Structural validation (CollectSpecData) must catch this on gen1 too
    NodeCoord node{0, 0};

    ProgramSpec spec;
    spec.program_id = "test_program";

    auto k0 = MakeMinimalGen1DMKernel("dm_kernel", DataMovementProcessor::RISCV_0);
    auto k1 = MakeMinimalGen1DMKernel("dm_kernel", DataMovementProcessor::RISCV_1);  // duplicate name

    spec.kernels = {k0, k1};
    spec.work_units = std::vector<WorkUnitSpec>{MakeMinimalWorkUnit("work_unit", node, {"dm_kernel"})};

    EXPECT_THAT(
        [&] { MakeProgramFromSpec(spec); },
        ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr("Duplicate KernelSpec name")));
}

}  // namespace
}  // namespace tt::tt_metal::experimental::metal2_host_api
