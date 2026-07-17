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
#include <filesystem>
#include <optional>
#include <type_traits>

#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program.hpp>
#include <tt_stl/reflection.hpp>
#include "impl/host_api/temp_quasar_api.hpp"  // for QuasarComputeConfig
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/tt_metal.hpp>  // for CompileProgram (JIT trigger)
#include <hostdevcommon/tensor_accessor/arg_config.hpp>  // tensor_accessor::ArgsConfig / ArgConfig::RuntimePageSize
#include "impl/kernels/kernel.hpp"
#include "impl/program/program_impl.hpp"
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/experimental/context/metal_env.hpp>
#include <tt-metalium/experimental/mock_device/mock_device.hpp>

#include "test_helpers.hpp"

namespace tt::tt_metal::experimental {
namespace {

// Import shared test helpers
using test_helpers::BindTensorParameterToKernel;
using test_helpers::MakeMinimalDFB;
using test_helpers::MakeMinimalGen1ComputeKernel;
using test_helpers::MakeMinimalGen1DMKernel;
using test_helpers::MakeMinimalGen1ValidProgramSpec;
using test_helpers::MakeMinimalGen2ComputeKernel;
using test_helpers::MakeMinimalGen2DMKernel;
using test_helpers::MakeMinimalReaderDMKernel;
using test_helpers::MakeMinimalTensorParameter;
using test_helpers::MakeMinimalValidProgramSpec;
using test_helpers::MakeMinimalWorkUnit;
using test_helpers::MakeMinimalWriterDMKernel;
using test_helpers::MakeShardedTensorParameter;
using test_helpers::ScopedSlowDispatchOverride;

// ============================================================================
// Reflection: ProgramSpec and its subcomponents are hashable via ttsl reflection
// ============================================================================
//
// ttsl::hash (tt_stl/reflection.hpp) hashes a plain aggregate by reflecting over its fields
// and recursing into each one. These checks pin that the whole ProgramSpec tree stays
// hashable: if a future change adds a field ttsl::hash can't handle (or makes one of these
// structs non-aggregate), the build breaks here rather than at some distant call site.
//
// This can't be a requires-expression. ttsl::hash::hash_object is an unconstrained template
// whose "unsupported type" case is a static_assert in the function *body*, so the call is
// always well-formed and unhashability only surfaces once the body is instantiated. hash_one
// is never called; taking its address ODR-uses it, which forces that instantiation — and the
// recursion through T's fields — at compile time.
template <typename T>
ttsl::hash::hash_t hash_one(const T& value) {
    return ttsl::hash::hash_objects_with_default_seed(value);
}
template <typename T>
inline constexpr bool hashable_v = (static_cast<void>(&hash_one<T>), true);

// Top-level specs
static_assert(hashable_v<ProgramSpec>, "ProgramSpec must be hashable via ttsl reflection");
static_assert(hashable_v<WorkUnitSpec>, "WorkUnitSpec must be hashable via ttsl reflection");
static_assert(hashable_v<KernelSpec>, "KernelSpec must be hashable via ttsl reflection");
static_assert(hashable_v<DataflowBufferSpec>, "DataflowBufferSpec must be hashable via ttsl reflection");
static_assert(
    hashable_v<CrossNodeDataflowBufferSpec>, "CrossNodeDataflowBufferSpec must be hashable via ttsl reflection");
static_assert(hashable_v<SemaphoreSpec>, "SemaphoreSpec must be hashable via ttsl reflection");
static_assert(hashable_v<TensorParameter>, "TensorParameter must be hashable via ttsl reflection");

// KernelSpec subcomponents
static_assert(hashable_v<KernelSpec::SourceCode>, "KernelSpec::SourceCode must be hashable via ttsl reflection");
static_assert(
    hashable_v<KernelSpec::CompilerOptions>, "KernelSpec::CompilerOptions must be hashable via ttsl reflection");
static_assert(
    hashable_v<KernelSpec::RuntimeArgSchema>, "KernelSpec::RuntimeArgSchema must be hashable via ttsl reflection");
static_assert(hashable_v<DFBBinding>, "DFBBinding must be hashable via ttsl reflection");
static_assert(hashable_v<SemaphoreBinding>, "SemaphoreBinding must be hashable via ttsl reflection");
static_assert(hashable_v<TensorBinding>, "TensorBinding must be hashable via ttsl reflection");

// Kernel hardware configs
static_assert(
    hashable_v<DataMovementHardwareConfig>, "DataMovementHardwareConfig must be hashable via ttsl reflection");
static_assert(hashable_v<DataMovementGen1Config>, "DataMovementGen1Config must be hashable via ttsl reflection");
static_assert(hashable_v<DataMovementGen2Config>, "DataMovementGen2Config must be hashable via ttsl reflection");
static_assert(hashable_v<ComputeHardwareConfig>, "ComputeHardwareConfig must be hashable via ttsl reflection");
static_assert(hashable_v<ComputeGen1Config>, "ComputeGen1Config must be hashable via ttsl reflection");
static_assert(hashable_v<ComputeGen2Config>, "ComputeGen2Config must be hashable via ttsl reflection");

// Per-spec advanced options
static_assert(hashable_v<KernelAdvancedOptions>, "KernelAdvancedOptions must be hashable via ttsl reflection");
static_assert(hashable_v<DFBAdvancedOptions>, "DFBAdvancedOptions must be hashable via ttsl reflection");
static_assert(hashable_v<SemaphoreAdvancedOptions>, "SemaphoreAdvancedOptions must be hashable via ttsl reflection");
static_assert(
    hashable_v<TensorParameterAdvancedOptions>, "TensorParameterAdvancedOptions must be hashable via ttsl reflection");

TEST(ProgramSpecReflectionTest, IsHashable) {
    const ProgramSpec spec = MakeMinimalValidProgramSpec();

    // Deterministic: hashing the same spec twice yields the same value.
    EXPECT_EQ(ttsl::hash::hash_objects_with_default_seed(spec), ttsl::hash::hash_objects_with_default_seed(spec));

    // Sensitive: changing a field changes the hash.
    ProgramSpec modified = spec;
    modified.name += "_v2";
    EXPECT_NE(ttsl::hash::hash_objects_with_default_seed(spec), ttsl::hash::hash_objects_with_default_seed(modified));
}

// ============================================================================
// Test Fixtures
// ============================================================================

// Test fixture for ProgramSpec on Quasar - uses Quasar mock device.
//
// Forces TT_METAL_SLOW_DISPATCH_MODE so that MeshDevice::create() succeeds against the
// mock cluster (mock Quasar has no dispatch-core reservation in its descriptor). These
// tests exercise pure API behavior, so slow dispatch is functionally fine.
class ProgramSpecTestQuasar : public ::testing::Test {
protected:
    void SetUp() override {
        slow_dispatch_override_.emplace();
        //  Configure global mock mode for Quasar
        //  This way, the HAL is initialized for arch check and Program creation.
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
// Structural Validation Tests (CollectSpecData)
// ============================================================================
// These test the structural integrity checks that happen during spec collection.

TEST_F(ProgramSpecTestQuasar, DuplicateKernelNameFails) {
    ProgramSpec spec = MakeMinimalValidProgramSpec();

    // Add a kernel with duplicate name
    auto duplicate_kernel = MakeMinimalGen2DMKernel("dm_kernel");
    duplicate_kernel.hw_config = DataMovementGen2Config{};
    spec.kernels.push_back(duplicate_kernel);

    EXPECT_THAT(
        [&] { MakeProgramFromSpec(*mesh_device_, spec); },
        ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr("Duplicate KernelSpec name 'dm_kernel'")));
}

TEST_F(ProgramSpecTestQuasar, DuplicateDFBNameFails) {
    ProgramSpec spec = MakeMinimalValidProgramSpec();

    // Add a DFB with duplicate name
    auto duplicate_dfb = MakeMinimalDFB("dfb_0");
    spec.dataflow_buffers.push_back(duplicate_dfb);

    EXPECT_THAT(
        [&] { MakeProgramFromSpec(*mesh_device_, spec); },
        ::testing::ThrowsMessage<std::runtime_error>(
            ::testing::HasSubstr("Duplicate DataflowBufferSpec name 'dfb_0'")));
}

TEST_F(ProgramSpecTestQuasar, DuplicateSemaphoreNameFails) {
    ProgramSpec spec = MakeMinimalValidProgramSpec();

    // Add two semaphores with the same name
    SemaphoreSpec sem1;
    sem1.unique_id = SemaphoreSpecName{"sem_0"};
    sem1.target_nodes = NodeCoord{0, 0};

    SemaphoreSpec sem2;
    sem2.unique_id = SemaphoreSpecName{"sem_0"};  // duplicate!
    sem2.target_nodes = NodeCoord{1, 0};

    spec.semaphores = {sem1, sem2};

    EXPECT_THAT(
        [&] { MakeProgramFromSpec(*mesh_device_, spec); },
        ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr("Duplicate SemaphoreSpec name 'sem_0'")));
}

TEST_F(ProgramSpecTestQuasar, SharedLocalAccessorNameForDifferentDFBsFails) {
    NodeCoord node{0, 0};

    ProgramSpec spec;
    spec.name = "test_program";

    auto kernel = MakeMinimalGen2DMKernel("kernel");
    auto dfb0 = MakeMinimalDFB("dfb_0");
    auto dfb1 = MakeMinimalDFB("dfb_1");

    // Bind two *different* DFBs with the same accessor_name — illegal
    // (self-loop sharing requires the same DFB on both bindings).
    kernel.dfb_bindings.push_back(ProducerOf(DFBSpecName{"dfb_0"}, "same_accessor"));
    kernel.dfb_bindings.push_back(ConsumerOf(DFBSpecName{"dfb_1"}, "same_accessor"));

    spec.kernels = {kernel};
    spec.dataflow_buffers = {dfb0, dfb1};
    spec.work_units = std::vector<WorkUnitSpec>{MakeMinimalWorkUnit("work_unit", node, {"kernel"})};

    EXPECT_THAT(
        [&] { MakeProgramFromSpec(*mesh_device_, spec); },
        ::testing::ThrowsMessage<std::runtime_error>(
            ::testing::HasSubstr("Kernel 'kernel' uses accessor_name 'same_accessor' for two different DFBs")));
}

TEST_F(ProgramSpecTestQuasar, DuplicateProducerBindingForSameLocalAccessorNameFails) {
    NodeCoord node{0, 0};

    ProgramSpec spec;
    spec.name = "test_program";

    auto producer_kernel = MakeMinimalGen2DMKernel("producer");
    auto consumer_kernel = MakeMinimalGen2DMKernel("consumer");
    auto dfb = MakeMinimalDFB("dfb");

    // Two PRODUCER bindings on the same kernel sharing a accessor_name —
    // illegal: the self-loop relaxation requires opposite endpoint types.
    producer_kernel.dfb_bindings.push_back(ProducerOf(DFBSpecName{"dfb"}, "shared"));
    producer_kernel.dfb_bindings.push_back(ProducerOf(DFBSpecName{"dfb"}, "shared"));
    consumer_kernel.dfb_bindings.push_back(ConsumerOf(DFBSpecName{"dfb"}, "in"));

    spec.kernels = {producer_kernel, consumer_kernel};
    spec.dataflow_buffers = {dfb};
    spec.work_units = std::vector<WorkUnitSpec>{MakeMinimalWorkUnit("work_unit", node, {"producer", "consumer"})};

    EXPECT_THAT(
        [&] { MakeProgramFromSpec(*mesh_device_, spec); },
        ::testing::ThrowsMessage<std::runtime_error>(
            ::testing::HasSubstr("duplicate PRODUCER binding for accessor_name 'shared'")));
}

TEST_F(ProgramSpecTestQuasar, SelfLoopWithSharedLocalAccessorNameSucceeds) {
    NodeCoord node{0, 0};

    ProgramSpec spec;
    spec.name = "test_program";

    // A kernel that both produces and consumes the same DFB may share a single
    // accessor_name across the PRODUCER and CONSUMER bindings. Uses a COMPUTE kernel: a compute
    // self-loop is the only legal self-loop (it lowers to the intra-Tensix packer->unpacker flow),
    // so it exercises the accessor-name relaxation through a path that survives validation. (A DM
    // self-loop is rejected — see DMKernelSelfLoopFails.)
    auto kernel = MakeMinimalGen2ComputeKernel("kernel");
    auto dfb = MakeMinimalDFB("dfb");
    dfb.data_format_metadata = tt::DataFormat::Float16_b;
    kernel.dfb_bindings.push_back(ProducerOf(DFBSpecName{"dfb"}, "acc"));
    kernel.dfb_bindings.push_back(ConsumerOf(DFBSpecName{"dfb"}, "acc"));

    spec.kernels = {kernel};
    spec.dataflow_buffers = {dfb};
    spec.work_units = std::vector<WorkUnitSpec>{MakeMinimalWorkUnit("work_unit", node, {"kernel"})};

    EXPECT_NO_THROW({ MakeProgramFromSpec(*mesh_device_, spec); });
}

TEST_F(ProgramSpecTestQuasar, DMKernelSelfLoopOnGen2Fails) {
    NodeCoord node{0, 0};

    ProgramSpec spec;
    spec.name = "test_program";

    // On Gen2, a data-movement kernel may NOT self-loop a DFB (bind it as both PRODUCER and CONSUMER).
    // The DFB's tile-counter credit machinery synchronizes a producer and a consumer on DISTINCT RISCs
    // via per-side masks, and a single DM kernel's producer and consumer masks are identical — the DFB
    // backend would reject it with an opaque "producer_risc_mask and consumer_risc_mask must not
    // overlap". Caught up front at validation with an actionable message instead. The legal Gen2
    // alternatives are a private L1 scratch buffer, a LocalTensorAccessor tensor view, or a two-kernel
    // cross-bind. (On Gen1 a DM self-loop IS legal — a DFB lowers to a plain circular buffer there; see
    // DMKernelSelfLoopOnGen1Succeeds. Compute self-loops stay legal on both gens — see
    // DFBSelfLoopOnComputeKernelSucceeds.)
    auto kernel = MakeMinimalGen2DMKernel("kernel");
    auto dfb = MakeMinimalDFB("dfb");
    kernel.dfb_bindings.push_back(ProducerOf(DFBSpecName{"dfb"}, "p"));
    kernel.dfb_bindings.push_back(ConsumerOf(DFBSpecName{"dfb"}, "c"));

    spec.kernels = {kernel};
    spec.dataflow_buffers = {dfb};
    spec.work_units = std::vector<WorkUnitSpec>{MakeMinimalWorkUnit("work_unit", node, {"kernel"})};

    EXPECT_THAT(
        [&] { MakeProgramFromSpec(*mesh_device_, spec); },
        ::testing::ThrowsMessage<std::runtime_error>(::testing::AllOf(
            ::testing::HasSubstr("self-looped by data-movement kernel 'kernel'"),
            ::testing::HasSubstr("not supported for data-movement kernels on Gen2 architectures"))));
}

TEST_F(ProgramSpecTestQuasar, DFBBoundTwiceInSameRoleUnderDifferentNamesFails) {
    NodeCoord node{0, 0};

    ProgramSpec spec;
    spec.name = "test_program";

    // The wrong port of the legacy "one buffer, two names" CB-alias idiom: one kernel binds the
    // same DFB twice in the SAME role (here two CONSUMER bindings) under different accessor names,
    // yielding two accessors / DataflowBuffer objects for one FIFO. Forbidden — the right port is a
    // kernel-side handle alias over a single binding. (A producer+consumer self-loop is a different,
    // legitimate multi-binding and stays legal — see SelfLoopWithSharedLocalAccessorNameSucceeds.)
    auto producer = MakeMinimalGen2DMKernel("producer");
    auto consumer = MakeMinimalGen2DMKernel("consumer");
    auto dfb = MakeMinimalDFB("dfb");

    producer.dfb_bindings.push_back(ProducerOf(DFBSpecName{"dfb"}, "out"));
    consumer.dfb_bindings.push_back(ConsumerOf(DFBSpecName{"dfb"}, "in_a"));
    consumer.dfb_bindings.push_back(ConsumerOf(DFBSpecName{"dfb"}, "in_b"));

    spec.kernels = {producer, consumer};
    spec.dataflow_buffers = {dfb};
    spec.work_units = std::vector<WorkUnitSpec>{MakeMinimalWorkUnit("work_unit", node, {"producer", "consumer"})};

    EXPECT_THAT(
        [&] { MakeProgramFromSpec(*mesh_device_, spec); },
        ::testing::ThrowsMessage<std::runtime_error>(
            ::testing::HasSubstr("has two CONSUMER bindings to DFB 'dfb' under different accessor names")));
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
        spec.name = "test_program";

        auto kernel = MakeMinimalGen2DMKernel("kernel");
        auto dfb = MakeMinimalDFB("dfb");

        kernel.dfb_bindings.push_back(ProducerOf(DFBSpecName{"dfb"}, bad_name));

        spec.kernels = {kernel};
        spec.dataflow_buffers = {dfb};
        spec.work_units = std::vector<WorkUnitSpec>{MakeMinimalWorkUnit("work_unit", node, {"kernel"})};

        EXPECT_THAT(
            [&] { MakeProgramFromSpec(*mesh_device_, spec); },
            ::testing::ThrowsMessage<std::runtime_error>(
                ::testing::HasSubstr("DFB accessor_name '" + bad_name + "' must be a valid C++ identifier")))
            << "Expected rejection for name: '" << bad_name << "'";
    }
}

TEST_F(ProgramSpecTestQuasar, KernelReferencesUnknownDFBFails) {
    NodeCoord node{0, 0};

    ProgramSpec spec;
    spec.name = "test_program";

    auto kernel = MakeMinimalGen2DMKernel("kernel");
    // Bind to a DFB that doesn't exist
    kernel.dfb_bindings.push_back(ProducerOf(DFBSpecName{"nonexistent_dfb"}, "accessor"));

    spec.kernels = {kernel};
    spec.work_units = std::vector<WorkUnitSpec>{MakeMinimalWorkUnit("work_unit", node, {"kernel"})};

    EXPECT_THAT(
        [&] { MakeProgramFromSpec(*mesh_device_, spec); },
        ::testing::ThrowsMessage<std::runtime_error>(
            ::testing::HasSubstr("Kernel 'kernel' references unknown DFB 'nonexistent_dfb'")));
}

TEST_F(ProgramSpecTestQuasar, DFBWithNoBindingsFails) {
    NodeCoord node{0, 0};

    ProgramSpec spec;
    spec.name = "test_program";

    // Create a kernel with no DFB bindings
    auto kernel = MakeMinimalGen2DMKernel("kernel");
    spec.kernels = {kernel};

    // Create a DFB that is never bound
    auto orphan_dfb = MakeMinimalDFB("orphan_dfb");
    spec.dataflow_buffers = {orphan_dfb};

    spec.work_units = std::vector<WorkUnitSpec>{MakeMinimalWorkUnit("work_unit", node, {"kernel"})};

    EXPECT_THAT(
        [&] { MakeProgramFromSpec(*mesh_device_, spec); },
        ::testing::ThrowsMessage<std::runtime_error>(
            ::testing::HasSubstr("DFB 'orphan_dfb' is defined but not bound by any kernel")));
}

TEST_F(ProgramSpecTestQuasar, DFBWithOnlyProducerFails) {
    NodeCoord node{0, 0};

    ProgramSpec spec;
    spec.name = "test_program";

    auto kernel = MakeMinimalGen2DMKernel("kernel");
    auto dfb = MakeMinimalDFB("dfb");

    // Only bind as producer, no consumer
    kernel.dfb_bindings.push_back(ProducerOf(DFBSpecName{"dfb"}, "accessor"));

    spec.kernels = {kernel};
    spec.dataflow_buffers = {dfb};
    spec.work_units = std::vector<WorkUnitSpec>{MakeMinimalWorkUnit("work_unit", node, {"kernel"})};

    EXPECT_THAT(
        [&] { MakeProgramFromSpec(*mesh_device_, spec); },
        ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr("DFB 'dfb' has no consumer")));
}

TEST_F(ProgramSpecTestQuasar, DFBWithOnlyConsumerFails) {
    NodeCoord node{0, 0};

    ProgramSpec spec;
    spec.name = "test_program";

    auto kernel = MakeMinimalGen2DMKernel("kernel");
    auto dfb = MakeMinimalDFB("dfb");

    // Only bind as consumer, no producer
    kernel.dfb_bindings.push_back(ConsumerOf(DFBSpecName{"dfb"}, "accessor"));

    spec.kernels = {kernel};
    spec.dataflow_buffers = {dfb};
    spec.work_units = std::vector<WorkUnitSpec>{MakeMinimalWorkUnit("work_unit", node, {"kernel"})};

    EXPECT_THAT(
        [&] { MakeProgramFromSpec(*mesh_device_, spec); },
        ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr("DFB 'dfb' has no producer")));
}

TEST_F(ProgramSpecTestQuasar, DFBWithMultipleProducersInSameWorkUnitFails) {
    NodeCoord node{0, 0};

    ProgramSpec spec;
    spec.name = "test_program";

    auto producer1 = MakeMinimalGen2DMKernel("producer1");
    auto producer2 = MakeMinimalGen2DMKernel("producer2");
    auto consumer = MakeMinimalGen2ComputeKernel("consumer");

    auto dfb = MakeMinimalDFB("dfb");
    dfb.data_format_metadata = tt::DataFormat::Float16_b;

    // Two PRODUCER bindings on the same DFB, both KernelSpecs in the same WorkUnitSpec (so both
    // land on the same node). A local DFB allows only one producer instance per node, so this fails.
    producer1.dfb_bindings.push_back(ProducerOf(DFBSpecName{"dfb"}, "out"));
    producer2.dfb_bindings.push_back(ProducerOf(DFBSpecName{"dfb"}, "out"));
    consumer.dfb_bindings.push_back(ConsumerOf(DFBSpecName{"dfb"}, "in"));

    spec.kernels = {producer1, producer2, consumer};
    spec.dataflow_buffers = {dfb};
    spec.work_units =
        std::vector<WorkUnitSpec>{MakeMinimalWorkUnit("work_unit", node, {"producer1", "producer2", "consumer"})};

    EXPECT_THAT(
        [&] { MakeProgramFromSpec(*mesh_device_, spec); },
        ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr("2 producer instance(s)")));
}

TEST_F(ProgramSpecTestQuasar, DFBWithMultipleConsumersInSameWorkUnitFails) {
    NodeCoord node{0, 0};

    ProgramSpec spec;
    spec.name = "test_program";

    auto producer = MakeMinimalGen2DMKernel("producer");
    // Both consumers DM (same kind) so the per-role kind-uniformity check passes and the
    // WU-disjointness check is what fires.
    auto consumer1 = MakeMinimalGen2DMKernel("consumer1");
    auto consumer2 = MakeMinimalGen2DMKernel("consumer2");

    auto dfb = MakeMinimalDFB("dfb");
    dfb.data_format_metadata = tt::DataFormat::Float16_b;

    producer.dfb_bindings.push_back(ProducerOf(DFBSpecName{"dfb"}, "out"));
    consumer1.dfb_bindings.push_back(ConsumerOf(DFBSpecName{"dfb"}, "in"));
    consumer2.dfb_bindings.push_back(ConsumerOf(DFBSpecName{"dfb"}, "in"));

    spec.kernels = {producer, consumer1, consumer2};
    spec.dataflow_buffers = {dfb};
    spec.work_units =
        std::vector<WorkUnitSpec>{MakeMinimalWorkUnit("work_unit", node, {"producer", "consumer1", "consumer2"})};

    EXPECT_THAT(
        [&] { MakeProgramFromSpec(*mesh_device_, spec); },
        ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr("2 consumer instance(s)")));
}

TEST_F(ProgramSpecTestQuasar, DFBWithMultipleConsumersInDifferentWorkUnitsSucceeds) {
    NodeCoord node0{0, 0};
    NodeCoord node1{1, 0};

    ProgramSpec spec;
    spec.name = "test_program";

    auto producer = MakeMinimalGen2DMKernel("producer");
    auto consumer1 = MakeMinimalGen2ComputeKernel("consumer1");
    auto consumer2 = MakeMinimalGen2ComputeKernel("consumer2");

    auto dfb = MakeMinimalDFB("dfb");
    dfb.data_format_metadata = tt::DataFormat::Float16_b;

    producer.dfb_bindings.push_back(ProducerOf(DFBSpecName{"dfb"}, "out"));
    consumer1.dfb_bindings.push_back(ConsumerOf(DFBSpecName{"dfb"}, "in"));
    consumer2.dfb_bindings.push_back(ConsumerOf(DFBSpecName{"dfb"}, "in"));

    // producer covers both WUs (placed in both); each consumer covers one WU.
    spec.kernels = {producer, consumer1, consumer2};
    spec.dataflow_buffers = {dfb};
    spec.work_units = std::vector<WorkUnitSpec>{
        MakeMinimalWorkUnit("wu_g1", node0, {"producer", "consumer1"}),
        MakeMinimalWorkUnit("wu_g2", node1, {"producer", "consumer2"}),
    };

    EXPECT_NO_THROW({ MakeProgramFromSpec(*mesh_device_, spec); });
}

TEST_F(ProgramSpecTestQuasar, DFBWithMultipleProducersInDifferentWorkUnitsSucceeds) {
    NodeCoord node0{0, 0};
    NodeCoord node1{1, 0};

    ProgramSpec spec;
    spec.name = "test_program";

    auto producer1 = MakeMinimalGen2DMKernel("producer1");
    auto producer2 = MakeMinimalGen2DMKernel("producer2");
    auto consumer = MakeMinimalGen2ComputeKernel("consumer");

    auto dfb = MakeMinimalDFB("dfb");
    dfb.data_format_metadata = tt::DataFormat::Float16_b;

    producer1.dfb_bindings.push_back(ProducerOf(DFBSpecName{"dfb"}, "out"));
    producer2.dfb_bindings.push_back(ProducerOf(DFBSpecName{"dfb"}, "out"));
    consumer.dfb_bindings.push_back(ConsumerOf(DFBSpecName{"dfb"}, "in"));

    spec.kernels = {producer1, producer2, consumer};
    spec.dataflow_buffers = {dfb};
    spec.work_units = std::vector<WorkUnitSpec>{
        MakeMinimalWorkUnit("wu_g1", node0, {"producer1", "consumer"}),
        MakeMinimalWorkUnit("wu_g2", node1, {"producer2", "consumer"}),
    };

    EXPECT_NO_THROW({ MakeProgramFromSpec(*mesh_device_, spec); });
}

TEST_F(ProgramSpecTestQuasar, DFBProducerConsumerCoverageMismatchFails) {
    NodeCoord node0{0, 0};
    NodeCoord node1{1, 0};

    ProgramSpec spec;
    spec.name = "test_program";

    auto producer = MakeMinimalGen2DMKernel("producer");
    auto consumer = MakeMinimalGen2ComputeKernel("consumer");

    auto dfb = MakeMinimalDFB("dfb");
    dfb.data_format_metadata = tt::DataFormat::Float16_b;

    producer.dfb_bindings.push_back(ProducerOf(DFBSpecName{"dfb"}, "out"));
    consumer.dfb_bindings.push_back(ConsumerOf(DFBSpecName{"dfb"}, "in"));

    // Producer covers node0; consumer covers node1. Every node ends up with only one role —
    // the per-node census rejects it (each node hosting the DFB needs both a producer and consumer).
    spec.kernels = {producer, consumer};
    spec.dataflow_buffers = {dfb};
    spec.work_units = std::vector<WorkUnitSpec>{
        MakeMinimalWorkUnit("wu_p", node0, {"producer"}),
        MakeMinimalWorkUnit("wu_c", node1, {"consumer"}),
    };

    EXPECT_THAT(
        [&] { MakeProgramFromSpec(*mesh_device_, spec); },
        ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr("is malformed at node")));
}

// The canonical 2D-matmul placement: one compute consumer spanning the whole grid, with a
// separate specialized DM producer per node group. The producer role has several KernelSpecs (one
// per group's WorkUnitSpec); the single consumer joins every group's WorkUnitSpec. Each node ends
// up with exactly one producer and one consumer instance, so the per-node census accepts it.
TEST_F(ProgramSpecTestQuasar, LocalDFBAllGridConsumerWithPerGroupProducersSucceeds) {
    ProgramSpec spec;
    spec.name = "test_program";

    auto consumer = MakeMinimalGen2ComputeKernel("compute");
    auto dfb = MakeMinimalDFB("dfb");
    dfb.data_format_metadata = tt::DataFormat::Float16_b;
    consumer.dfb_bindings.push_back(ConsumerOf(DFBSpecName{"dfb"}, "in"));

    std::vector<KernelSpec> kernels;
    std::vector<WorkUnitSpec> work_units;
    for (uint32_t i = 0; i < 4; ++i) {
        const std::string dm_name = "dm" + std::to_string(i);
        auto dm = MakeMinimalGen2DMKernel(dm_name);
        dm.dfb_bindings.push_back(ProducerOf(DFBSpecName{"dfb"}, "out"));
        kernels.push_back(dm);
        // One node per group: the per-group DM lives there, and the all-grid compute joins it.
        work_units.push_back(MakeMinimalWorkUnit("wu" + std::to_string(i), NodeCoord{i, 0}, {dm_name, "compute"}));
    }
    kernels.push_back(consumer);

    spec.kernels = std::move(kernels);
    spec.dataflow_buffers = {dfb};
    spec.work_units = std::move(work_units);

    EXPECT_NO_THROW({ MakeProgramFromSpec(*mesh_device_, spec); });
}

TEST_F(ProgramSpecTestQuasar, DFBMultiBindingAccessPatternMismatchFails) {
    NodeCoord node0{0, 0};
    NodeCoord node1{1, 0};

    ProgramSpec spec;
    spec.name = "test_program";

    auto producer = MakeMinimalGen2DMKernel("producer");
    auto consumer1 = MakeMinimalGen2ComputeKernel("consumer1");
    auto consumer2 = MakeMinimalGen2ComputeKernel("consumer2");

    auto dfb = MakeMinimalDFB("dfb");
    dfb.data_format_metadata = tt::DataFormat::Float16_b;

    producer.dfb_bindings.push_back(ProducerOf(DFBSpecName{"dfb"}, "out"));
    consumer1.dfb_bindings.push_back(ConsumerOf(DFBSpecName{"dfb"}, "in"));
    consumer2.dfb_bindings.push_back(AllConsumerOf(DFBSpecName{"dfb"}, "in"));

    spec.kernels = {producer, consumer1, consumer2};
    spec.dataflow_buffers = {dfb};
    spec.work_units = std::vector<WorkUnitSpec>{
        MakeMinimalWorkUnit("wu_g1", node0, {"producer", "consumer1"}),
        MakeMinimalWorkUnit("wu_g2", node1, {"producer", "consumer2"}),
    };

    EXPECT_THAT(
        [&] { MakeProgramFromSpec(*mesh_device_, spec); },
        ::testing::ThrowsMessage<std::runtime_error>(
            ::testing::HasSubstr("DFB 'dfb' has multiple CONSUMER bindings with mismatched access_pattern")));
}

TEST_F(ProgramSpecTestQuasar, DFBMultiBindingNumThreadsMismatchFails) {
    NodeCoord node0{0, 0};
    NodeCoord node1{1, 0};

    ProgramSpec spec;
    spec.name = "test_program";

    auto producer = MakeMinimalGen2DMKernel("producer");
    auto consumer1 = MakeMinimalGen2ComputeKernel("consumer1", /*num_threads=*/1);
    auto consumer2 = MakeMinimalGen2ComputeKernel("consumer2", /*num_threads=*/2);

    auto dfb = MakeMinimalDFB("dfb");
    dfb.data_format_metadata = tt::DataFormat::Float16_b;

    producer.dfb_bindings.push_back(ProducerOf(DFBSpecName{"dfb"}, "out"));
    consumer1.dfb_bindings.push_back(ConsumerOf(DFBSpecName{"dfb"}, "in"));
    consumer2.dfb_bindings.push_back(ConsumerOf(DFBSpecName{"dfb"}, "in"));

    spec.kernels = {producer, consumer1, consumer2};
    spec.dataflow_buffers = {dfb};
    spec.work_units = std::vector<WorkUnitSpec>{
        MakeMinimalWorkUnit("wu_g1", node0, {"producer", "consumer1"}),
        MakeMinimalWorkUnit("wu_g2", node1, {"producer", "consumer2"}),
    };

    EXPECT_THAT(
        [&] { MakeProgramFromSpec(*mesh_device_, spec); },
        ::testing::ThrowsMessage<std::runtime_error>(
            ::testing::HasSubstr("DFB 'dfb' has multiple CONSUMER KernelSpecs with mismatched num_threads")));
}

TEST_F(ProgramSpecTestQuasar, DFBMultiBindingMixingComputeAndDMOnSameRoleFails) {
    NodeCoord node0{0, 0};
    NodeCoord node1{1, 0};

    ProgramSpec spec;
    spec.name = "test_program";

    // Producer side mixes a DM and a compute kernel on disjoint zones. Each individually
    // would form a valid binding, but the DFB's hardware config carries a single producer
    // processor mask per role; the two kinds occupy disjoint mask bit ranges and cannot
    // share a mask. The validator must reject upfront.
    auto dm_producer = MakeMinimalGen2DMKernel("dm_producer");
    auto compute_producer = MakeMinimalGen2ComputeKernel("compute_producer");
    auto consumer = MakeMinimalGen2DMKernel("consumer");

    auto dfb = MakeMinimalDFB("dfb");
    dfb.data_format_metadata = tt::DataFormat::Float16_b;

    dm_producer.dfb_bindings.push_back(ProducerOf(DFBSpecName{"dfb"}, "out"));
    compute_producer.dfb_bindings.push_back(ProducerOf(DFBSpecName{"dfb"}, "out"));
    consumer.dfb_bindings.push_back(ConsumerOf(DFBSpecName{"dfb"}, "in"));

    spec.kernels = {dm_producer, compute_producer, consumer};
    spec.dataflow_buffers = {dfb};
    spec.work_units = std::vector<WorkUnitSpec>{
        MakeMinimalWorkUnit("wu_g1", node0, {"dm_producer", "consumer"}),
        MakeMinimalWorkUnit("wu_g2", node1, {"compute_producer", "consumer"}),
    };

    EXPECT_THAT(
        [&] { MakeProgramFromSpec(*mesh_device_, spec); },
        ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr("mixing compute and data-movement kinds")));
}

TEST_F(ProgramSpecTestQuasar, DFBMultiBindingSelfLoopWithMatchingSidesSucceeds) {
    NodeCoord node0{0, 0};
    NodeCoord node1{1, 0};

    ProgramSpec spec;
    spec.name = "test_program";

    // Two self-looping kernels on disjoint WUs. Each binds "dfb" as both producer and
    // consumer; producer set equals consumer set = {self_loop_1, self_loop_2}. At each node,
    // exactly one kernel runs and self-loops the DFB — the local invariant holds.
    auto self_loop_1 = MakeMinimalGen2ComputeKernel("self_loop_1");
    auto self_loop_2 = MakeMinimalGen2ComputeKernel("self_loop_2");

    auto dfb = MakeMinimalDFB("dfb");
    dfb.data_format_metadata = tt::DataFormat::Float16_b;
    // INTRA-tensix self-loop DFBs have no DM endpoint; the spec-to-impl translation produces
    // enable_{producer,consumer}_implicit_sync=false automatically (no DM kernel to vote for it).

    self_loop_1.dfb_bindings.push_back(ProducerOf(DFBSpecName{"dfb"}, "p"));
    self_loop_1.dfb_bindings.push_back(ConsumerOf(DFBSpecName{"dfb"}, "c"));
    self_loop_2.dfb_bindings.push_back(ProducerOf(DFBSpecName{"dfb"}, "p"));
    self_loop_2.dfb_bindings.push_back(ConsumerOf(DFBSpecName{"dfb"}, "c"));

    spec.kernels = {self_loop_1, self_loop_2};
    spec.dataflow_buffers = {dfb};
    spec.work_units = std::vector<WorkUnitSpec>{
        MakeMinimalWorkUnit("wu_g1", node0, {"self_loop_1"}),
        MakeMinimalWorkUnit("wu_g2", node1, {"self_loop_2"}),
    };

    EXPECT_NO_THROW({ MakeProgramFromSpec(*mesh_device_, spec); });
}

TEST_F(ProgramSpecTestQuasar, DFBSelfLoopWithExtraProducerSideKernelFails) {
    NodeCoord node0{0, 0};
    NodeCoord node1{1, 0};

    ProgramSpec spec;
    spec.name = "test_program";

    // self_loop_1 binds "dfb" as BOTH producer and consumer (self-loop). On the other WU,
    // an unrelated producer-only kernel is bound, while extra_consumer covers the consume side.
    // Producer set = {self_loop_1, extra_producer}; consumer set = {self_loop_1, extra_consumer}.
    // The sets are not equal — the self-loop multi-binding rule rejects this mix.
    //
    // All three kernels are COMPUTE so the self-loop participant (self_loop_1) is a legal compute
    // self-loop — a DM self-loop would be rejected earlier on Gen2 (see DMKernelSelfLoopOnGen2Fails),
    // masking the rule under test. With compute kernels the per-role kind-uniformity check passes and
    // the self-loop set-equality refinement check is reached.
    auto self_loop_1 = MakeMinimalGen2ComputeKernel("self_loop_1");
    auto extra_producer = MakeMinimalGen2ComputeKernel("extra_producer");
    auto extra_consumer = MakeMinimalGen2ComputeKernel("extra_consumer");

    auto dfb = MakeMinimalDFB("dfb");
    dfb.data_format_metadata = tt::DataFormat::Float16_b;

    self_loop_1.dfb_bindings.push_back(ProducerOf(DFBSpecName{"dfb"}, "p"));
    self_loop_1.dfb_bindings.push_back(ConsumerOf(DFBSpecName{"dfb"}, "c"));
    extra_producer.dfb_bindings.push_back(ProducerOf(DFBSpecName{"dfb"}, "out"));
    extra_consumer.dfb_bindings.push_back(ConsumerOf(DFBSpecName{"dfb"}, "in"));

    spec.kernels = {self_loop_1, extra_producer, extra_consumer};
    spec.dataflow_buffers = {dfb};
    spec.work_units = std::vector<WorkUnitSpec>{
        MakeMinimalWorkUnit("wu_g1", node0, {"self_loop_1"}),
        MakeMinimalWorkUnit("wu_g2", node1, {"extra_producer", "extra_consumer"}),
    };

    EXPECT_THAT(
        [&] { MakeProgramFromSpec(*mesh_device_, spec); },
        ::testing::ThrowsMessage<std::runtime_error>(
            ::testing::HasSubstr("DFB 'dfb' is self-looped (some kernel appears as both producer and consumer), but "
                                 "the set of producer KernelSpecs differs from the set of consumer KernelSpecs")));
}

// ----------------------------------------------------------------------------
// DFB implicit-sync opt-out (Gen2)
// ----------------------------------------------------------------------------
// Implicit sync is ON by default for any DFB side that has a DM endpoint. A DM kernel can
// opt out per-DFB (disable_dfb_implicit_sync_for) or for all the DFBs it binds at once
// (disable_dfb_implicit_sync_for_all). These tests pin the per-kernel "all" hammer.

TEST_F(ProgramSpecTestQuasar, DisableImplicitSyncForAllDisablesProducerSide) {
    // Build the canonical DM-producer -> compute-consumer DFB, optionally hammering the
    // producer's implicit sync off, and read back the lowered DataflowBufferConfig.
    auto make_spec = [](bool disable_all) {
        ProgramSpec spec;
        spec.name = "test_program";

        auto dm_kernel = MakeMinimalGen2DMKernel("dm_kernel");
        auto compute_kernel = MakeMinimalGen2ComputeKernel("compute_kernel");
        auto& dm_hw_config =
            std::get<DataMovementGen2Config>(std::get<DataMovementHardwareConfig>(dm_kernel.hw_config));
        dm_hw_config.disable_dfb_implicit_sync_for_all = disable_all;

        auto dfb = MakeMinimalDFB("dfb_0");
        dfb.data_format_metadata = tt::DataFormat::Float16_b;

        dm_kernel.dfb_bindings.push_back(ProducerOf(DFBSpecName{"dfb_0"}, "out"));
        compute_kernel.dfb_bindings.push_back(ConsumerOf(DFBSpecName{"dfb_0"}, "in"));

        spec.kernels = {dm_kernel, compute_kernel};
        spec.dataflow_buffers = {dfb};
        spec.work_units = {MakeMinimalWorkUnit("work_unit_0", NodeCoord{0, 0}, {"dm_kernel", "compute_kernel"})};
        return spec;
    };

    // Default: the producer side has a DM kernel, so implicit sync is on.
    {
        auto program = MakeProgramFromSpec(*mesh_device_, make_spec(/*disable_all=*/false));
        const uint32_t dfb_id = program.impl().get_dfb_handle("dfb_0");
        EXPECT_TRUE(program.impl().get_dataflow_buffer(dfb_id)->config.enable_producer_implicit_sync);
    }
    // disable_dfb_implicit_sync_for_all turns it off for every DFB the kernel binds.
    {
        auto program = MakeProgramFromSpec(*mesh_device_, make_spec(/*disable_all=*/true));
        const uint32_t dfb_id = program.impl().get_dfb_handle("dfb_0");
        EXPECT_FALSE(program.impl().get_dataflow_buffer(dfb_id)->config.enable_producer_implicit_sync);
    }
}

TEST_F(ProgramSpecTestQuasar, DisableImplicitSyncForAllDisagreementAcrossProducersFails) {
    NodeCoord node0{0, 0};
    NodeCoord node1{1, 0};

    ProgramSpec spec;
    spec.name = "test_program";

    auto producer1 = MakeMinimalGen2DMKernel("producer1");
    auto producer2 = MakeMinimalGen2DMKernel("producer2");
    auto consumer = MakeMinimalGen2ComputeKernel("consumer");

    // producer1 hammers implicit sync off; producer2 leaves it on. Both bind the same DFB on
    // the producer side, so the per-side opt-out disagrees and validation must reject.
    DataMovementGen2Config& producer1_hw_config =
        std::get<DataMovementGen2Config>(std::get<DataMovementHardwareConfig>(producer1.hw_config));
    producer1_hw_config.disable_dfb_implicit_sync_for_all = true;

    auto dfb = MakeMinimalDFB("dfb");
    dfb.data_format_metadata = tt::DataFormat::Float16_b;

    producer1.dfb_bindings.push_back(ProducerOf(DFBSpecName{"dfb"}, "out"));
    producer2.dfb_bindings.push_back(ProducerOf(DFBSpecName{"dfb"}, "out"));
    consumer.dfb_bindings.push_back(ConsumerOf(DFBSpecName{"dfb"}, "in"));

    spec.kernels = {producer1, producer2, consumer};
    spec.dataflow_buffers = {dfb};
    spec.work_units = std::vector<WorkUnitSpec>{
        MakeMinimalWorkUnit("wu_g1", node0, {"producer1", "consumer"}),
        MakeMinimalWorkUnit("wu_g2", node1, {"producer2", "consumer"}),
    };

    EXPECT_THAT(
        [&] { MakeProgramFromSpec(*mesh_device_, spec); },
        ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr("disagreeing implicit-sync opt-out state")));
}

TEST_F(ProgramSpecTestQuasar, DisableImplicitSyncForAllAgreesWithExplicitList) {
    NodeCoord node0{0, 0};
    NodeCoord node1{1, 0};

    ProgramSpec spec;
    spec.name = "test_program";

    auto producer1 = MakeMinimalGen2DMKernel("producer1");
    auto producer2 = MakeMinimalGen2DMKernel("producer2");
    auto consumer = MakeMinimalGen2ComputeKernel("consumer");

    // producer1 opts out via the per-kernel hammer; producer2 opts the same DFB out by name.
    // Both express the same per-side decision (disable), so they agree and the side lowers off.
    DataMovementGen2Config& producer1_hw_config =
        std::get<DataMovementGen2Config>(std::get<DataMovementHardwareConfig>(producer1.hw_config));
    DataMovementGen2Config& producer2_hw_config =
        std::get<DataMovementGen2Config>(std::get<DataMovementHardwareConfig>(producer2.hw_config));
    producer1_hw_config.disable_dfb_implicit_sync_for_all = true;
    producer2_hw_config.disable_dfb_implicit_sync_for.push_back(DFBSpecName{"dfb"});

    auto dfb = MakeMinimalDFB("dfb");
    dfb.data_format_metadata = tt::DataFormat::Float16_b;

    producer1.dfb_bindings.push_back(ProducerOf(DFBSpecName{"dfb"}, "out"));
    producer2.dfb_bindings.push_back(ProducerOf(DFBSpecName{"dfb"}, "out"));
    consumer.dfb_bindings.push_back(ConsumerOf(DFBSpecName{"dfb"}, "in"));

    spec.kernels = {producer1, producer2, consumer};
    spec.dataflow_buffers = {dfb};
    spec.work_units = std::vector<WorkUnitSpec>{
        MakeMinimalWorkUnit("wu_g1", node0, {"producer1", "consumer"}),
        MakeMinimalWorkUnit("wu_g2", node1, {"producer2", "consumer"}),
    };

    auto program = MakeProgramFromSpec(*mesh_device_, spec);
    const uint32_t dfb_id = program.impl().get_dfb_handle("dfb");
    EXPECT_FALSE(program.impl().get_dataflow_buffer(dfb_id)->config.enable_producer_implicit_sync);
}

// ============================================================================
// Semantic Validation Tests (ValidateProgramSpec)
// ============================================================================

TEST_F(ProgramSpecTestQuasar, EmptyKernelsFails) {
    ProgramSpec spec;
    spec.name = "empty_program";
    spec.work_units = std::vector<WorkUnitSpec>{};  // Empty work_units too

    EXPECT_THAT(
        [&] { MakeProgramFromSpec(*mesh_device_, spec); },
        ::testing::ThrowsMessage<std::runtime_error>(
            ::testing::HasSubstr("A ProgramSpec must have at least one KernelSpec")));
}

TEST_F(ProgramSpecTestQuasar, KernelWithZeroThreadsFails) {
    NodeCoord node{0, 0};

    ProgramSpec spec;
    spec.name = "test_program";

    auto kernel = MakeMinimalGen2DMKernel("kernel", 0);  // 0 threads!
    spec.kernels = {kernel};
    spec.work_units = std::vector<WorkUnitSpec>{MakeMinimalWorkUnit("work_unit", node, {"kernel"})};

    EXPECT_THAT(
        [&] { MakeProgramFromSpec(*mesh_device_, spec); },
        ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr("KernelSpec 'kernel' has no threads")));
}

TEST_F(ProgramSpecTestQuasar, DMKernelExceedingMaxThreadsFails) {
    NodeCoord node{0, 0};

    ProgramSpec spec;
    spec.name = "test_program";

    // Quasar has 8 DM cores per node (we reserve 2 for internal use)
    auto kernel = MakeMinimalGen2DMKernel("kernel", 9);  // Too many threads!
    spec.kernels = {kernel};
    spec.work_units = std::vector<WorkUnitSpec>{MakeMinimalWorkUnit("work_unit", node, {"kernel"})};

    EXPECT_THAT(
        [&] { MakeProgramFromSpec(*mesh_device_, spec); },
        ::testing::ThrowsMessage<std::runtime_error>(
            ::testing::HasSubstr("KernelSpec 'kernel' has too many data movement threads")));
}

TEST_F(ProgramSpecTestQuasar, ComputeKernelExceedingMaxThreadsFails) {
    NodeCoord node{0, 0};

    ProgramSpec spec;
    spec.name = "test_program";

    // Quasar has 4 Tensix cores per node
    auto kernel = MakeMinimalGen2ComputeKernel("kernel", 5);  // Too many threads!
    spec.kernels = {kernel};
    spec.work_units = std::vector<WorkUnitSpec>{MakeMinimalWorkUnit("work_unit", node, {"kernel"})};

    EXPECT_THAT(
        [&] { MakeProgramFromSpec(*mesh_device_, spec); },
        ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr(
            "KernelSpec 'kernel' has too many threads. The architecture supports up to 4 for compute kernels")));
}

TEST_F(ProgramSpecTestQuasar, DMKernelWithGen1ConfigFails) {
    // The config's generation must match the target platform: on Gen2 (Quasar) a DM kernel must
    // carry a DataMovementGen2Config. Supplying an explicit Gen1 config is a hard error — it is not
    // silently substituted with a default Gen2 config.
    NodeCoord node{0, 0};

    ProgramSpec spec;
    spec.name = "test_program";

    auto kernel = MakeMinimalGen2DMKernel("kernel");
    // Replace the default Gen2 config with an explicit Gen1 config (wrong generation for Quasar).
    auto& dm_config = std::get<DataMovementHardwareConfig>(kernel.hw_config);
    dm_config = DataMovementGen1Config{
        .processor = DataMovementProcessor::RISCV_0,
        .noc = NOC::RISCV_0_default,
        .noc_mode = NOC_MODE::DM_DEDICATED_NOC,
    };

    spec.kernels = {kernel};
    spec.work_units = std::vector<WorkUnitSpec>{MakeMinimalWorkUnit("work_unit", node, {"kernel"})};

    EXPECT_THAT(
        [&] { MakeProgramFromSpec(*mesh_device_, spec); },
        ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr("holds a DataMovementGen1Config")));
}

TEST_F(ProgramSpecTestQuasar, DMKernelWithDefaultGen2ConfigSucceeds) {
    // On Gen2 a DM kernel needs no explicit tuning: Gen2 has a unified NOC and fully automated DM
    // placement, and a default Gen2Config (empty disable_dfb_implicit_sync_for) is all that's required.
    NodeCoord node{0, 0};

    ProgramSpec spec;
    spec.name = "test_program";

    // MakeMinimalGen2DMKernel already holds a default Gen2Config.
    auto kernel = MakeMinimalGen2DMKernel("kernel");

    spec.kernels = {kernel};
    spec.work_units = std::vector<WorkUnitSpec>{MakeMinimalWorkUnit("work_unit", node, {"kernel"})};

    EXPECT_NO_THROW({ MakeProgramFromSpec(*mesh_device_, spec); });
}

TEST_F(ProgramSpecTestQuasar, RoleBasedGen1ConfigOnGen2Fails) {
    // MakeMinimalReaderDMKernel builds a Gen1 placement (DataMovementGen1Config), which
    // is the wrong generation for Gen2 (Quasar): the platform requires a DataMovementGen2Config, so the
    // mismatch is a hard error rather than a silently-ignored role hint.
    NodeCoord node{0, 0};

    ProgramSpec spec;
    spec.name = "test_program";

    auto kernel = MakeMinimalReaderDMKernel("kernel");

    spec.kernels = {kernel};
    spec.work_units = std::vector<WorkUnitSpec>{MakeMinimalWorkUnit("work_unit", node, {"kernel"})};

    EXPECT_THAT(
        [&] { MakeProgramFromSpec(*mesh_device_, spec); },
        ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr("holds a DataMovementGen1Config")));
}

// Cross-node DFBs are part of the API surface but not yet supported by the runtime.
TEST_F(ProgramSpecTestQuasar, CrossNodeDFBNotYetSupportedAtRuntime) {
    NodeCoord producer_node{0, 0};
    NodeCoord consumer_node{1, 0};

    ProgramSpec spec;
    spec.name = "test_program";

    auto producer = MakeMinimalGen2DMKernel("producer");
    auto consumer = MakeMinimalGen2DMKernel("consumer");

    producer.dfb_bindings.push_back(ProducerOf(DFBSpecName{"dfb"}, "out"));
    consumer.dfb_bindings.push_back(ConsumerOf(DFBSpecName{"dfb"}, "in"));

    spec.kernels = {producer, consumer};
    spec.cross_node_dataflow_buffers = {CrossNodeDataflowBufferSpec{
        .dfb_spec = MakeMinimalDFB("dfb"),
        .producer_consumer_map = {{producer_node, consumer_node}},
    }};
    spec.work_units = std::vector<WorkUnitSpec>{
        MakeMinimalWorkUnit("producer_work_unit", producer_node, {"producer"}),
        MakeMinimalWorkUnit("consumer_work_unit", consumer_node, {"consumer"}),
    };

    EXPECT_THAT(
        [&] { MakeProgramFromSpec(*mesh_device_, spec); },
        ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr("not yet supported")));
}

// Helper: build a ProgramSpec with a single borrowed-memory DFB backed by a TensorParameter.
//   - DFB default size: 32 bytes (entry_size 16 * num_entries 2). Fits inside
//     MakeMinimalTensorParameter's 1x32 BFLOAT16 default (64 bytes); oversized cases
//     pass larger dfb_entry_size / dfb_num_entries via the parameters.
//   - tensor_buffer_type defaults to L1 (the only legal choice for borrowing).
inline ProgramSpec MakeBorrowedDFBProgramSpec(
    const std::string& tensor_param_name = "borrowed_tensor",
    tt::tt_metal::BufferType tensor_buffer_type = tt::tt_metal::BufferType::L1,
    uint32_t dfb_entry_size = 16,
    uint32_t dfb_num_entries = 2,
    bool bind_backing_to_kernel = true) {
    NodeCoord node{0, 0};

    ProgramSpec spec;
    spec.name = "test_program";

    auto producer = MakeMinimalGen2DMKernel("producer");
    auto consumer = MakeMinimalGen2DMKernel("consumer");
    auto dfb = MakeMinimalDFB("dfb", dfb_entry_size, dfb_num_entries);
    dfb.borrowed_from = TensorParamName{tensor_param_name};

    producer.dfb_bindings.push_back(ProducerOf(DFBSpecName{"dfb"}, "out"));
    consumer.dfb_bindings.push_back(ConsumerOf(DFBSpecName{"dfb"}, "in"));

    auto tensor_param = MakeMinimalTensorParameter(tensor_param_name, tensor_buffer_type);
    // The borrowed_from reference is itself counted as a use of the TensorParameter, so binding it
    // to a kernel is not required for referential integrity. Callers exercising the borrowed-only
    // path pass bind_backing_to_kernel=false.
    if (bind_backing_to_kernel) {
        BindTensorParameterToKernel(producer, tensor_param_name, "borrowed_t");
    }

    spec.kernels = {producer, consumer};
    spec.dataflow_buffers = {dfb};
    spec.tensor_parameters = {tensor_param};
    spec.work_units = std::vector<WorkUnitSpec>{MakeMinimalWorkUnit("work_unit", node, {"producer", "consumer"})};
    return spec;
}

TEST_F(ProgramSpecTestQuasar, BorrowedMemoryDFBSucceeds) {
    // Positive baseline: borrowed-memory DFB whose TensorParameter is L1-resident and large enough.
    ProgramSpec spec = MakeBorrowedDFBProgramSpec();
    EXPECT_NO_THROW(MakeProgramFromSpec(*mesh_device_, spec));
}

TEST_F(ProgramSpecTestQuasar, BorrowedMemoryDFBBackingParameterNeedNotBeKernelBoundSucceeds) {
    // Regression: a TensorParameter used ONLY as a borrowed-memory DFB's backing (referenced via
    // borrowed_from, never bound by a kernel) is a legitimate use. The validator must count
    // borrowed_from toward referential integrity rather than rejecting the parameter as "defined
    // but not bound by any kernel". This is the common borrowed-memory-DFB case (e.g. a borrowed
    // LUT / scratch tensor consumed only through the DFB).
    ProgramSpec spec = MakeBorrowedDFBProgramSpec(
        "borrowed_tensor",
        tt::tt_metal::BufferType::L1,
        /*dfb_entry_size=*/16,
        /*dfb_num_entries=*/2,
        /*bind_backing_to_kernel=*/false);
    EXPECT_NO_THROW(MakeProgramFromSpec(*mesh_device_, spec));
}

TEST_F(ProgramSpecTestQuasar, BorrowedMemoryDFBUnknownTensorParameterFails) {
    ProgramSpec spec = MakeBorrowedDFBProgramSpec("borrowed_tensor");
    // Re-target the DFB at a TensorParameter that wasn't declared.
    spec.dataflow_buffers[0].borrowed_from = TensorParamName{"nonexistent_tensor"};

    EXPECT_THAT(
        [&] { MakeProgramFromSpec(*mesh_device_, spec); },
        ::testing::ThrowsMessage<std::runtime_error>(
            ::testing::HasSubstr("borrows memory from TensorParameter 'nonexistent_tensor'")));
}

TEST_F(ProgramSpecTestQuasar, BorrowedMemoryDFBNonL1TensorParameterFails) {
    // DRAM-resident TensorParameter is not a legal borrow source.
    ProgramSpec spec = MakeBorrowedDFBProgramSpec("borrowed_tensor", tt::tt_metal::BufferType::DRAM);

    EXPECT_THAT(
        [&] { MakeProgramFromSpec(*mesh_device_, spec); },
        ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr("is not L1-resident")));
}

TEST_F(ProgramSpecTestQuasar, BorrowedMemoryDFBOversizedFails) {
    // DFB total bytes exceed the TensorParameter's packed size: 1*32*sizeof(bfloat16) = 64 bytes,
    // so 128 bytes of DFB (entry_size 64, num_entries 2) overruns.
    ProgramSpec spec = MakeBorrowedDFBProgramSpec(
        "borrowed_tensor", tt::tt_metal::BufferType::L1, /*dfb_entry_size=*/64, /*dfb_num_entries=*/2);

    EXPECT_THAT(
        [&] { MakeProgramFromSpec(*mesh_device_, spec); },
        ::testing::ThrowsMessage<std::runtime_error>(
            ::testing::HasSubstr("is larger than its borrowed TensorParameter")));
}

TEST_F(ProgramSpecTestQuasar, SemaphoresSucceed) {
    ProgramSpec spec = MakeMinimalValidProgramSpec();

    SemaphoreSpec sem;
    sem.unique_id = SemaphoreSpecName{"sem_0"};
    sem.target_nodes = NodeCoord{0, 0};
    spec.semaphores = {sem};

    EXPECT_NO_THROW(MakeProgramFromSpec(*mesh_device_, spec));
}

TEST_F(ProgramSpecTestQuasar, KernelSemaphoreBindingsSucceed) {
    ProgramSpec spec = MakeMinimalValidProgramSpec();

    SemaphoreSpec sem;
    sem.unique_id = SemaphoreSpecName{"sem_0"};
    sem.target_nodes = NodeCoord{0, 0};
    spec.semaphores = {sem};

    SemaphoreBinding binding;
    binding.semaphore_spec_name = SemaphoreSpecName{"sem_0"};
    binding.accessor_name = "my_sem";
    spec.kernels[0].semaphore_bindings = {binding};

    EXPECT_NO_THROW(MakeProgramFromSpec(*mesh_device_, spec));
}

TEST_F(ProgramSpecTestQuasar, SemaphoreBoundToComputeKernelFailsOnQuasar) {
    // Compute kernels cannot have semaphore bindings on any arch.
    // (This may later change for Quasar.)
    ProgramSpec spec = MakeMinimalValidProgramSpec();

    SemaphoreSpec sem;
    sem.unique_id = SemaphoreSpecName{"sem_0"};
    sem.target_nodes = NodeCoord{0, 0};
    spec.semaphores = {sem};

    // kernels[1] is the compute kernel in MakeMinimalValidProgramSpec
    ASSERT_TRUE(spec.kernels[1].is_compute_kernel());
    spec.kernels[1].semaphore_bindings = {
        SemaphoreBinding{.semaphore_spec_name = SemaphoreSpecName{"sem_0"}, .accessor_name = "done_flag"}};

    EXPECT_THAT(
        [&] { MakeProgramFromSpec(*mesh_device_, spec); },
        ::testing::ThrowsMessage<std::runtime_error>(
            ::testing::HasSubstr("Semaphore bindings are not supported for compute kernels.")));
}

TEST_F(ProgramSpecTestQuasar, KernelSemaphoreBindingUnknownSemaphoreFails) {
    ProgramSpec spec = MakeMinimalValidProgramSpec();

    SemaphoreBinding binding;
    binding.semaphore_spec_name = SemaphoreSpecName{"missing_sem"};
    binding.accessor_name = "my_sem";
    spec.kernels[0].semaphore_bindings = {binding};

    EXPECT_THAT(
        [&] { MakeProgramFromSpec(*mesh_device_, spec); },
        ::testing::ThrowsMessage<std::runtime_error>(
            ::testing::HasSubstr("references unknown semaphore 'missing_sem'")));
}

TEST_F(ProgramSpecTestQuasar, KernelSemaphoreBindingInvalidAccessorFails) {
    ProgramSpec spec = MakeMinimalValidProgramSpec();

    SemaphoreSpec sem;
    sem.unique_id = SemaphoreSpecName{"sem_0"};
    sem.target_nodes = NodeCoord{0, 0};
    spec.semaphores = {sem};

    SemaphoreBinding binding;
    binding.semaphore_spec_name = SemaphoreSpecName{"sem_0"};
    binding.accessor_name = "has-dash";
    spec.kernels[0].semaphore_bindings = {binding};

    EXPECT_THAT(
        [&] { MakeProgramFromSpec(*mesh_device_, spec); },
        ::testing::ThrowsMessage<std::runtime_error>(
            ::testing::HasSubstr("semaphore accessor_name 'has-dash' must be a valid C++ identifier")));
}

TEST_F(ProgramSpecTestQuasar, KernelSemaphoreBindingDuplicateAccessorFails) {
    ProgramSpec spec = MakeMinimalValidProgramSpec();

    SemaphoreSpec sem0;
    sem0.unique_id = SemaphoreSpecName{"sem_0"};
    sem0.target_nodes = NodeCoord{0, 0};

    SemaphoreSpec sem1;
    sem1.unique_id = SemaphoreSpecName{"sem_1"};
    sem1.target_nodes = NodeCoord{0, 0};

    spec.semaphores = {sem0, sem1};

    spec.kernels[0].semaphore_bindings = {
        SemaphoreBinding{.semaphore_spec_name = SemaphoreSpecName{"sem_0"}, .accessor_name = "same"},
        SemaphoreBinding{.semaphore_spec_name = SemaphoreSpecName{"sem_1"}, .accessor_name = "same"}};

    EXPECT_THAT(
        [&] { MakeProgramFromSpec(*mesh_device_, spec); },
        ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr("duplicate semaphore accessor_name 'same'")));
}

// ============================================================================
// Kernel Scratchpad Validation Tests (CollectSpecData)
// ============================================================================
//
// A kernel scratchpad is a private, blank, node-local L1 region bound to exactly one kernel for the
// program's execution lifetime (see scratchpad_spec.hpp). These tests pin the structural-validation
// rules enforced in CollectSpecData: name uniqueness, binding referential integrity, the
// exactly-one-binding-per-scratchpad invariant, accessor-name uniqueness per kernel, and the C++
// identifier rule for accessor names. They mirror the DFB / semaphore binding-validation tests
// above: build on MakeMinimalValidProgramSpec() and bind the scratchpad to the DM kernel
// (kernels[0]).

TEST_F(ProgramSpecTestQuasar, ValidScratchpadSucceeds) {
    // Positive baseline: one ScratchpadSpec bound by exactly one kernel.
    ProgramSpec spec = MakeMinimalValidProgramSpec();

    spec.scratchpads = {ScratchpadSpec{.unique_id = ScratchpadSpecName{"scratch_0"}, .size_per_node = 1024}};
    spec.kernels[0].scratchpad_bindings = {
        KernelSpec::ScratchpadBinding{.scratchpad_spec_name = ScratchpadSpecName{"scratch_0"}, .accessor_name = "s"}};

    EXPECT_NO_THROW(MakeProgramFromSpec(*mesh_device_, spec));
}

TEST_F(ProgramSpecTestQuasar, DuplicateScratchpadNameFails) {
    // Two ScratchpadSpecs declared with the same unique_id.
    ProgramSpec spec = MakeMinimalValidProgramSpec();

    spec.scratchpads = {
        ScratchpadSpec{.unique_id = ScratchpadSpecName{"scratch_0"}, .size_per_node = 1024},
        ScratchpadSpec{.unique_id = ScratchpadSpecName{"scratch_0"}, .size_per_node = 512},  // duplicate!
    };
    spec.kernels[0].scratchpad_bindings = {
        KernelSpec::ScratchpadBinding{.scratchpad_spec_name = ScratchpadSpecName{"scratch_0"}, .accessor_name = "s"}};

    EXPECT_THAT(
        [&] { MakeProgramFromSpec(*mesh_device_, spec); },
        ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr("Duplicate ScratchpadSpec name")));
}

TEST_F(ProgramSpecTestQuasar, ZeroSizeScratchpadFails) {
    // A ScratchpadSpec with size_per_node == 0 (the default) reserves no L1, so the device-side
    // accessor's operator[] would be out of bounds on first use. Bound to a kernel here so the
    // size check — not the unbound check — is what fires.
    ProgramSpec spec = MakeMinimalValidProgramSpec();

    spec.scratchpads = {ScratchpadSpec{.unique_id = ScratchpadSpecName{"scratch_0"}}};  // size_per_node defaults to 0
    spec.kernels[0].scratchpad_bindings = {
        KernelSpec::ScratchpadBinding{.scratchpad_spec_name = ScratchpadSpecName{"scratch_0"}, .accessor_name = "s"}};

    EXPECT_THAT(
        [&] { MakeProgramFromSpec(*mesh_device_, spec); },
        ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr("size_per_node == 0")));
}

TEST_F(ProgramSpecTestQuasar, UnknownScratchpadReferenceFails) {
    // A scratchpad_binding referencing a scratchpad_spec_name that isn't declared in spec.scratchpads.
    ProgramSpec spec = MakeMinimalValidProgramSpec();

    // No spec.scratchpads declared, but the kernel binds one.
    spec.kernels[0].scratchpad_bindings = {KernelSpec::ScratchpadBinding{
        .scratchpad_spec_name = ScratchpadSpecName{"missing_scratch"}, .accessor_name = "s"}};

    EXPECT_THAT(
        [&] { MakeProgramFromSpec(*mesh_device_, spec); },
        ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr("references unknown scratchpad")));
}

TEST_F(ProgramSpecTestQuasar, UnboundScratchpadFails) {
    // A ScratchpadSpec declared in spec.scratchpads that no kernel binds. An unbound scratchpad
    // reserves L1 no kernel can reach.
    ProgramSpec spec = MakeMinimalValidProgramSpec();

    spec.scratchpads = {ScratchpadSpec{.unique_id = ScratchpadSpecName{"orphan_scratch"}, .size_per_node = 1024}};
    // No kernel binds it.

    EXPECT_THAT(
        [&] { MakeProgramFromSpec(*mesh_device_, spec); },
        ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr("declared but not bound")));
}

TEST_F(ProgramSpecTestQuasar, ScratchpadBoundByTwoKernelsSameNodeFails) {
    // One ScratchpadSpec bound by two kernels that share a node. A scratchpad is private node-local
    // L1; binding it from two kernels on the SAME node would be true sharing, which is not yet
    // supported (the disjoint-node case IS allowed — see the next test). MakeMinimalValidProgramSpec
    // places both kernels on node {0,0}, so this is the same-node collision case.
    ProgramSpec spec = MakeMinimalValidProgramSpec();

    spec.scratchpads = {ScratchpadSpec{.unique_id = ScratchpadSpecName{"scratch_0"}, .size_per_node = 1024}};
    // kernels[0] (DM) and kernels[1] (compute) both bind it, and both run on node {0,0}.
    spec.kernels[0].scratchpad_bindings = {KernelSpec::ScratchpadBinding{
        .scratchpad_spec_name = ScratchpadSpecName{"scratch_0"}, .accessor_name = "s_dm"}};
    spec.kernels[1].scratchpad_bindings = {KernelSpec::ScratchpadBinding{
        .scratchpad_spec_name = ScratchpadSpecName{"scratch_0"}, .accessor_name = "s_compute"}};

    EXPECT_THAT(
        [&] { MakeProgramFromSpec(*mesh_device_, spec); },
        ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr("kernel instances on node")));
}

TEST_F(ProgramSpecTestQuasar, ScratchpadBoundByTwoKernelsDisjointNodesSucceeds) {
    // Complement of the same-node case above: one ScratchpadSpec bound by two kernels on DISJOINT
    // nodes is legal. Each node hosts exactly one binding kernel instance, so the per-node scratchpad
    // stays private to that kernel (allocation + CRTA delivery are per-binding-kernel, so the two
    // bindings never interact). This is the matmul-grid-style fan: one kernel source specialized into
    // multiple KernelSpecs on disjoint node ranges, all binding the same scratchpad resource.
    NodeCoord node0{0, 0};
    NodeCoord node1{1, 0};

    ProgramSpec spec;
    spec.name = "scratchpad_shared_disjoint";

    auto kernel_a = MakeMinimalGen2DMKernel("kernel_a");
    kernel_a.scratchpad_bindings.push_back(KernelSpec::ScratchpadBinding{
        .scratchpad_spec_name = ScratchpadSpecName{"scratch_shared"}, .accessor_name = "scratch"});
    auto kernel_b = MakeMinimalGen2DMKernel("kernel_b");
    kernel_b.scratchpad_bindings.push_back(KernelSpec::ScratchpadBinding{
        .scratchpad_spec_name = ScratchpadSpecName{"scratch_shared"}, .accessor_name = "scratch"});

    spec.kernels = {kernel_a, kernel_b};
    spec.scratchpads = {ScratchpadSpec{.unique_id = ScratchpadSpecName{"scratch_shared"}, .size_per_node = 1024}};
    spec.work_units = std::vector<WorkUnitSpec>{
        MakeMinimalWorkUnit("wu_a", node0, {"kernel_a"}),
        MakeMinimalWorkUnit("wu_b", node1, {"kernel_b"}),
    };

    EXPECT_NO_THROW({ MakeProgramFromSpec(*mesh_device_, spec); });
}

TEST_F(ProgramSpecTestQuasar, ScratchpadBoundTwiceInOneKernelFails) {
    // One kernel binds the SAME scratchpad twice under two different accessor_names. Illegal: a kernel
    // may bind a given scratchpad at most once (two bindings would request two separate per-node
    // allocations under one name). This is a structural input error with no node-set dependency, so
    // it is caught up front during collection rather than by the placement census.
    ProgramSpec spec = MakeMinimalValidProgramSpec();

    spec.scratchpads = {ScratchpadSpec{.unique_id = ScratchpadSpecName{"scratch_0"}, .size_per_node = 1024}};
    spec.kernels[0].scratchpad_bindings = {
        KernelSpec::ScratchpadBinding{.scratchpad_spec_name = ScratchpadSpecName{"scratch_0"}, .accessor_name = "s_a"},
        KernelSpec::ScratchpadBinding{.scratchpad_spec_name = ScratchpadSpecName{"scratch_0"}, .accessor_name = "s_b"},
    };

    EXPECT_THAT(
        [&] { MakeProgramFromSpec(*mesh_device_, spec); },
        ::testing::ThrowsMessage<std::runtime_error>(
            ::testing::HasSubstr("binds scratchpad 'scratch_0' more than once")));
}

TEST_F(ProgramSpecTestQuasar, DuplicateScratchpadAccessorNameFails) {
    // One kernel with two scratchpad_bindings to two DIFFERENT scratchpads but sharing the same
    // accessor_name. The accessor_name is the kernel-local C++ symbol, so it must be unique per
    // kernel (the per-kernel duplicate check fires before the bound-more-than-once check, since
    // the two scratchpads are distinct).
    ProgramSpec spec = MakeMinimalValidProgramSpec();

    spec.scratchpads = {
        ScratchpadSpec{.unique_id = ScratchpadSpecName{"scratch_0"}, .size_per_node = 1024},
        ScratchpadSpec{.unique_id = ScratchpadSpecName{"scratch_1"}, .size_per_node = 1024},
    };
    spec.kernels[0].scratchpad_bindings = {
        KernelSpec::ScratchpadBinding{.scratchpad_spec_name = ScratchpadSpecName{"scratch_0"}, .accessor_name = "dup"},
        KernelSpec::ScratchpadBinding{.scratchpad_spec_name = ScratchpadSpecName{"scratch_1"}, .accessor_name = "dup"},
    };

    EXPECT_THAT(
        [&] { MakeProgramFromSpec(*mesh_device_, spec); },
        ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr("duplicate scratchpad accessor_name")));
}

TEST_F(ProgramSpecTestQuasar, InvalidScratchpadAccessorNameFails) {
    // The accessor_name becomes a C++ identifier in the generated kernel_bindings header, so it must
    // be a valid C++ identifier. (Mirrors InvalidLocalAccessorNameFails / the semaphore-accessor
    // equivalent; here we just spot-check a couple of clearly-invalid names.)
    const std::vector<std::string> invalid_names = {
        "1bad",       // leading digit
        "has space",  // whitespace
    };

    for (const auto& bad_name : invalid_names) {
        ProgramSpec spec = MakeMinimalValidProgramSpec();
        spec.scratchpads = {ScratchpadSpec{.unique_id = ScratchpadSpecName{"scratch_0"}, .size_per_node = 1024}};
        spec.kernels[0].scratchpad_bindings = {KernelSpec::ScratchpadBinding{
            .scratchpad_spec_name = ScratchpadSpecName{"scratch_0"}, .accessor_name = bad_name}};

        EXPECT_THAT(
            [&] { MakeProgramFromSpec(*mesh_device_, spec); },
            ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr("must be a valid C++ identifier")))
            << "Expected rejection for scratchpad accessor_name: '" << bad_name << "'";
    }
}

TEST_F(ProgramSpecTestQuasar, MultipleScratchpadsEachBoundToOwnKernelSucceeds) {
    NodeCoord node0{0, 0};
    NodeCoord node1{1, 0};

    ProgramSpec spec;
    spec.name = "scratchpad_multi";

    // Two independent scratchpads, each bound by its own kernel on its own node — the simplest
    // multi-scratchpad case (distinct from binding one shared scratchpad across disjoint nodes).
    auto kernel_a = MakeMinimalGen2DMKernel("kernel_a");
    kernel_a.scratchpad_bindings.push_back(KernelSpec::ScratchpadBinding{
        .scratchpad_spec_name = ScratchpadSpecName{"scratch_a"}, .accessor_name = "scratch"});
    auto kernel_b = MakeMinimalGen2DMKernel("kernel_b");
    kernel_b.scratchpad_bindings.push_back(KernelSpec::ScratchpadBinding{
        .scratchpad_spec_name = ScratchpadSpecName{"scratch_b"}, .accessor_name = "scratch"});

    spec.kernels = {kernel_a, kernel_b};
    spec.scratchpads = {
        ScratchpadSpec{.unique_id = ScratchpadSpecName{"scratch_a"}, .size_per_node = 1024},
        ScratchpadSpec{.unique_id = ScratchpadSpecName{"scratch_b"}, .size_per_node = 2048},
    };
    spec.work_units = std::vector<WorkUnitSpec>{
        MakeMinimalWorkUnit("wu_a", node0, {"kernel_a"}),
        MakeMinimalWorkUnit("wu_b", node1, {"kernel_b"}),
    };

    EXPECT_NO_THROW({ MakeProgramFromSpec(*mesh_device_, spec); });
}

// ----------------------------------------------------------------------------
// Kernel hash sensitivity to scratchpad bindings
// ----------------------------------------------------------------------------
//
// The kernel's JIT cache key is its compute_hash(); a scratchpad binding flows into the device-side
// codegen (the scratch:: namespace + the CRTA-injected base address) and into the kernel's
// ScratchpadBindingHandles, so a kernel that binds a scratchpad must NOT hash equal to one that
// doesn't — otherwise it would silently reuse a stale cached binary.

TEST_F(ProgramSpecTestQuasar, ScratchpadBindingAffectsKernelHash) {
    // Same kernel source, differing only in whether the kernel binds a scratchpad. The bound variant
    // carries an extra ScratchpadBindingHandle, so the hashes must differ.
    auto make_bound_spec = [] {
        ProgramSpec spec;
        spec.name = "scratchpad_hash_bound";
        auto dm_kernel = MakeMinimalGen2DMKernel("dm_kernel");
        dm_kernel.scratchpad_bindings.push_back(KernelSpec::ScratchpadBinding{
            .scratchpad_spec_name = ScratchpadSpecName{"scratch"}, .accessor_name = "scratch"});
        spec.kernels = {dm_kernel};
        spec.scratchpads = {ScratchpadSpec{.unique_id = ScratchpadSpecName{"scratch"}, .size_per_node = 1024}};
        spec.work_units = std::vector<WorkUnitSpec>{MakeMinimalWorkUnit("work_unit", NodeCoord{0, 0}, {"dm_kernel"})};
        return spec;
    };
    auto make_unbound_spec = [] {
        ProgramSpec spec;
        spec.name = "scratchpad_hash_unbound";
        auto dm_kernel = MakeMinimalGen2DMKernel("dm_kernel");
        spec.kernels = {dm_kernel};
        spec.work_units = std::vector<WorkUnitSpec>{MakeMinimalWorkUnit("work_unit", NodeCoord{0, 0}, {"dm_kernel"})};
        return spec;
    };

    Program prog_bound = MakeProgramFromSpec(*mesh_device_, make_bound_spec());
    Program prog_unbound = MakeProgramFromSpec(*mesh_device_, make_unbound_spec());

    auto hash_bound = prog_bound.impl().get_kernel_by_spec_name("dm_kernel")->compute_hash();
    auto hash_unbound = prog_unbound.impl().get_kernel_by_spec_name("dm_kernel")->compute_hash();
    EXPECT_NE(hash_bound, hash_unbound)
        << "A kernel that binds a scratchpad must not share a JIT cache slot with one that doesn't.";
}

TEST_F(ProgramSpecTestQuasar, DifferentScratchpadSizeProducesDifferentKernelHash) {
    // Same kernel source and accessor name; the two scratchpads differ only in size_per_node, which
    // flows into the ScratchpadBindingHandle's size (and the generated scratch:: token), so the
    // hashes must differ.
    auto make_spec = [](uint32_t size_per_node) {
        ProgramSpec spec;
        spec.name = "scratchpad_hash_size";
        auto dm_kernel = MakeMinimalGen2DMKernel("dm_kernel");
        dm_kernel.scratchpad_bindings.push_back(KernelSpec::ScratchpadBinding{
            .scratchpad_spec_name = ScratchpadSpecName{"scratch"}, .accessor_name = "scratch"});
        spec.kernels = {dm_kernel};
        spec.scratchpads = {ScratchpadSpec{.unique_id = ScratchpadSpecName{"scratch"}, .size_per_node = size_per_node}};
        spec.work_units = std::vector<WorkUnitSpec>{MakeMinimalWorkUnit("work_unit", NodeCoord{0, 0}, {"dm_kernel"})};
        return spec;
    };

    Program prog_small = MakeProgramFromSpec(*mesh_device_, make_spec(/*size_per_node=*/1024));
    Program prog_large = MakeProgramFromSpec(*mesh_device_, make_spec(/*size_per_node=*/2048));

    auto hash_small = prog_small.impl().get_kernel_by_spec_name("dm_kernel")->compute_hash();
    auto hash_large = prog_large.impl().get_kernel_by_spec_name("dm_kernel")->compute_hash();
    EXPECT_NE(hash_small, hash_large) << "Scratchpads of different sizes must produce different kernel hashes.";
}

TEST_F(ProgramSpecTestQuasar, TensorBindingOnComputeKernelIsAccepted) {
    // A tensor binding on a compute kernel is legal: the kernel constructs a LocalTensorAccessor
    // (NOC-free) from the binding token rather than a TensorAccessor. ValidateProgramSpec accepts it;
    // there is no host-side residency check. (The compile/dispatch path is proven in the HW test
    // LocalTensorAccessorBindingCompileComputeKernel.)
    ProgramSpec spec = MakeMinimalValidProgramSpec();
    spec.tensor_parameters = {MakeMinimalTensorParameter("t")};
    BindTensorParameterToKernel(spec.kernels[1], "t", "t_acc");  // kernels[1] == compute_kernel

    EXPECT_NO_THROW({ MakeProgramFromSpec(*mesh_device_, spec); });
}

TEST_F(ProgramSpecTestQuasar, SemaphoreNonZeroInitialValueFailsOnQuasar) {
    ProgramSpec spec = MakeMinimalValidProgramSpec();

    SemaphoreSpec sem;
    sem.unique_id = SemaphoreSpecName{"sem_0"};
    sem.target_nodes = NodeCoord{0, 0};
    sem.advanced_options = SemaphoreAdvancedOptions{.initial_value = 1};
    spec.semaphores = {sem};

    EXPECT_THAT(
        [&] { MakeProgramFromSpec(*mesh_device_, spec); },
        ::testing::ThrowsMessage<std::runtime_error>(
            ::testing::HasSubstr("has initial_value=1 but only zero is supported on Quasar")));
}

// ---- Named RTA / CRTA / CTA schema validation ----

TEST_F(ProgramSpecTestQuasar, NamedRuntimeArgsSucceeds) {
    ProgramSpec spec = MakeMinimalValidProgramSpec();
    spec.kernels[0].runtime_arg_schema.runtime_arg_names = {"input_ptr", "output_ptr"};
    spec.kernels[0].runtime_arg_schema.common_runtime_arg_names = {"tile_count"};
    spec.kernels[0].compile_time_args = {{"block_size", 64}};

    EXPECT_NO_THROW(MakeProgramFromSpec(*mesh_device_, spec));
}

TEST_F(ProgramSpecTestQuasar, InvalidNamedRtaIdentifierFails) {
    ProgramSpec spec = MakeMinimalValidProgramSpec();
    spec.kernels[0].runtime_arg_schema.runtime_arg_names = {"int"};  // C++ keyword

    EXPECT_THAT(
        [&] { MakeProgramFromSpec(*mesh_device_, spec); },
        ::testing::ThrowsMessage<std::runtime_error>(
            ::testing::HasSubstr("named RTA name 'int' is not a valid C++ identifier")));
}

TEST_F(ProgramSpecTestQuasar, InvalidNamedCrtaIdentifierFails) {
    ProgramSpec spec = MakeMinimalValidProgramSpec();
    spec.kernels[0].runtime_arg_schema.common_runtime_arg_names = {"has-dash"};

    EXPECT_THAT(
        [&] { MakeProgramFromSpec(*mesh_device_, spec); },
        ::testing::ThrowsMessage<std::runtime_error>(
            ::testing::HasSubstr("named CRTA name 'has-dash' is not a valid C++ identifier")));
}

TEST_F(ProgramSpecTestQuasar, NamedRtaCrtaCollisionFails) {
    // A single name cannot be both a named RTA and a named CRTA (they share the user namespace).
    ProgramSpec spec = MakeMinimalValidProgramSpec();
    spec.kernels[0].runtime_arg_schema.runtime_arg_names = {"count"};
    spec.kernels[0].runtime_arg_schema.common_runtime_arg_names = {"count"};

    EXPECT_THAT(
        [&] { MakeProgramFromSpec(*mesh_device_, spec); },
        ::testing::ThrowsMessage<std::runtime_error>(
            ::testing::HasSubstr("naming collision: 'count' is declared as both a named RTA and a named CRTA")));
}

TEST_F(ProgramSpecTestQuasar, NamedRtaCtaCollisionFails) {
    ProgramSpec spec = MakeMinimalValidProgramSpec();
    spec.kernels[0].runtime_arg_schema.runtime_arg_names = {"block_size"};
    spec.kernels[0].compile_time_args = {{"block_size", 64}};  // same name as CTA

    EXPECT_THAT(
        [&] { MakeProgramFromSpec(*mesh_device_, spec); },
        ::testing::ThrowsMessage<std::runtime_error>(
            ::testing::HasSubstr("naming collision: 'block_size' is declared as both a named RTA and a named CTA")));
}

TEST_F(ProgramSpecTestQuasar, DifferentKernelsMayReuseArgNames) {
    // Collision rule is per-kernel. Two different kernels may have identically-named args.
    ProgramSpec spec = MakeMinimalValidProgramSpec();
    spec.kernels[0].runtime_arg_schema.runtime_arg_names = {"shared_name"};
    spec.kernels[1].runtime_arg_schema.runtime_arg_names = {"shared_name"};

    EXPECT_NO_THROW(MakeProgramFromSpec(*mesh_device_, spec));
}

TEST_F(ProgramSpecTestQuasar, DFBWithComputeEndpointRequiresDataFormat) {
    NodeCoord node{0, 0};

    ProgramSpec spec;
    spec.name = "test_program";

    auto producer = MakeMinimalGen2DMKernel("producer");
    auto consumer = MakeMinimalGen2ComputeKernel("consumer");  // Compute!
    auto dfb = MakeMinimalDFB("dfb");
    // dfb.data_format_metadata is NOT set (nullopt)

    producer.dfb_bindings.push_back(ProducerOf(DFBSpecName{"dfb"}, "out"));
    consumer.dfb_bindings.push_back(ConsumerOf(DFBSpecName{"dfb"}, "in"));

    spec.kernels = {producer, consumer};
    spec.dataflow_buffers = {dfb};
    spec.work_units = std::vector<WorkUnitSpec>{MakeMinimalWorkUnit("work_unit", node, {"producer", "consumer"})};

    EXPECT_THAT(
        [&] { MakeProgramFromSpec(*mesh_device_, spec); },
        ::testing::ThrowsMessage<std::runtime_error>(
            ::testing::HasSubstr("DFB 'dfb' is used by a compute kernel, but no data_format_metadata is specified")));
}

TEST_F(ProgramSpecTestQuasar, ComputeConfigUnpackToDestModeReferencesUnboundDFBFails) {
    NodeCoord node{0, 0};

    ProgramSpec spec;
    spec.name = "test_program";

    auto producer = MakeMinimalGen2DMKernel("producer");
    auto consumer = MakeMinimalGen2ComputeKernel("consumer");

    // Set an unpack_modes entry referencing a DFB this kernel doesn't bind
    // (in this case, a DFB that doesn't exist in the spec at all).
    auto& compute_config = std::get<ComputeGen2Config>(std::get<ComputeHardwareConfig>(consumer.hw_config));
    compute_config.unpack_modes = {{DFBSpecName{"nonexistent_dfb"}, UnpackMode::UnpackToDest}};

    auto dfb = MakeMinimalDFB("dfb");
    dfb.data_format_metadata = tt::DataFormat::Float16_b;

    producer.dfb_bindings.push_back(ProducerOf(DFBSpecName{"dfb"}, "out"));
    consumer.dfb_bindings.push_back(ConsumerOf(DFBSpecName{"dfb"}, "in"));

    spec.kernels = {producer, consumer};
    spec.dataflow_buffers = {dfb};
    spec.work_units = std::vector<WorkUnitSpec>{MakeMinimalWorkUnit("work_unit", node, {"producer", "consumer"})};

    EXPECT_THAT(
        [&] { MakeProgramFromSpec(*mesh_device_, spec); },
        ::testing::ThrowsMessage<std::runtime_error>(
            ::testing::HasSubstr("Kernel 'consumer' unpack_modes entry references DFB 'nonexistent_dfb', "
                                 "which the kernel does not bind")));
}

TEST_F(ProgramSpecTestQuasar, NonFP32DFBWithoutUnpackToDestModeEntrySucceeds) {
    // Non-FP32 DFBs default to Default; omitting an entry is the expected idiom.
    ProgramSpec spec = MakeMinimalValidProgramSpec();  // dfb_0 is Float16_b (non-FP32)
    EXPECT_NO_THROW(MakeProgramFromSpec(*mesh_device_, spec));
}

TEST_F(ProgramSpecTestQuasar, NonFP32DFBWithExplicitDefaultUnpackToDestModeSucceeds) {
    // Existing call sites that explicitly spell out Default for non-FP32 DFBs keep working.
    ProgramSpec spec = MakeMinimalValidProgramSpec();  // dfb_0 is Float16_b
    for (auto& kernel : spec.kernels) {
        if (kernel.is_compute_kernel()) {
            auto& config = std::get<ComputeGen2Config>(std::get<ComputeHardwareConfig>(kernel.hw_config));
            config.unpack_modes = {{DFBSpecName{"dfb_0"}, UnpackMode::UnpackToSrc}};
        }
    }
    EXPECT_NO_THROW(MakeProgramFromSpec(*mesh_device_, spec));
}

TEST_F(ProgramSpecTestQuasar, NonFP32DFBWithUnpackToDestFp32ModeSucceeds) {
    // UnpackToDest on a non-Float32 DFB is INERT: the LLK ignores the mode where the data
    // isn't FP32. The validator tolerates it (rejecting it would force porters to dtype-gate
    // legacy unpack_to_dest_mode vectors that set UnpackToDestFp32 unconditionally). With
    // enable_32_bit_dest=true the entry is coherent, so the spec validates.
    ProgramSpec spec = MakeMinimalValidProgramSpec();  // dfb_0 is Float16_b (non-FP32)
    for (auto& kernel : spec.kernels) {
        if (kernel.is_compute_kernel()) {
            auto& config = std::get<ComputeGen2Config>(std::get<ComputeHardwareConfig>(kernel.hw_config));
            config.enable_32_bit_dest = true;
            config.unpack_modes = {{DFBSpecName{"dfb_0"}, UnpackMode::UnpackToDest}};
        }
    }
    EXPECT_NO_THROW(MakeProgramFromSpec(*mesh_device_, spec));
}

TEST_F(ProgramSpecTestQuasar, FP32ConsumerWithFp32DestAccEnAndNoEntryFails) {
    // The narrow case where a choice is required: CONSUMER + FP32 + enable_32_bit_dest=true.
    ProgramSpec spec = MakeMinimalValidProgramSpec();
    for (auto& dfb : spec.dataflow_buffers) {
        if (dfb.unique_id == DFBSpecName{"dfb_0"}) {
            dfb.data_format_metadata = tt::DataFormat::Float32;
        }
    }
    for (auto& kernel : spec.kernels) {
        if (kernel.is_compute_kernel()) {
            auto& config = std::get<ComputeGen2Config>(std::get<ComputeHardwareConfig>(kernel.hw_config));
            config.enable_32_bit_dest = true;
        }
    }
    // Compute kernel intentionally has no unpack_modes entry.

    EXPECT_THAT(
        [&] { MakeProgramFromSpec(*mesh_device_, spec); },
        ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr(
            "Compute kernel 'compute_kernel' consumes FP32 DFB 'dfb_0' with enable_32_bit_dest=true, but "
            "provides no unpack_modes entry for this DFB")));
}

TEST_F(ProgramSpecTestQuasar, FP32ConsumerWithoutFp32DestAccEnDoesNotRequireEntry) {
    // Without enable_32_bit_dest, UnpackToDest is incoherent (Dest is 16-bit), so there's
    // no real choice — UnpackToSrc is the only valid value. No explicit entry required.
    ProgramSpec spec = MakeMinimalValidProgramSpec();
    for (auto& dfb : spec.dataflow_buffers) {
        if (dfb.unique_id == DFBSpecName{"dfb_0"}) {
            dfb.data_format_metadata = tt::DataFormat::Float32;
        }
    }
    // enable_32_bit_dest stays at its default (false). No unpack_modes entry.
    EXPECT_NO_THROW(MakeProgramFromSpec(*mesh_device_, spec));
}

TEST_F(ProgramSpecTestQuasar, FP32ProducerOnlyBindingDoesNotRequireEntry) {
    // A compute kernel that only PRODUCES an FP32 DFB never unpacks it, so the unpack mode
    // is dead config — no explicit entry required regardless of enable_32_bit_dest.
    NodeCoord node{0, 0};

    ProgramSpec spec;
    spec.name = "test_program";

    auto producer_compute = MakeMinimalGen2ComputeKernel("producer_compute");
    auto& producer_config = std::get<ComputeGen2Config>(std::get<ComputeHardwareConfig>(producer_compute.hw_config));
    producer_config.enable_32_bit_dest = true;

    auto consumer_dm = MakeMinimalGen2DMKernel("consumer_dm");

    auto dfb = MakeMinimalDFB("dfb_0");
    dfb.data_format_metadata = tt::DataFormat::Float32;

    producer_compute.dfb_bindings.push_back(ProducerOf(DFBSpecName{"dfb_0"}, "out"));
    consumer_dm.dfb_bindings.push_back(ConsumerOf(DFBSpecName{"dfb_0"}, "in"));

    spec.kernels = {producer_compute, consumer_dm};
    spec.dataflow_buffers = {dfb};
    spec.work_units =
        std::vector<WorkUnitSpec>{MakeMinimalWorkUnit("work_unit", node, {"producer_compute", "consumer_dm"})};

    EXPECT_NO_THROW(MakeProgramFromSpec(*mesh_device_, spec));
}

TEST_F(ProgramSpecTestQuasar, UnpackToDestFp32OnProducerBindingSucceeds) {
    // UnpackToDest on a producer-only binding is INERT (producers don't unpack), so the
    // validator tolerates it rather than rejecting. With enable_32_bit_dest=true it is coherent.
    NodeCoord node{0, 0};

    ProgramSpec spec;
    spec.name = "test_program";

    auto producer_compute = MakeMinimalGen2ComputeKernel("producer_compute");
    auto& producer_config = std::get<ComputeGen2Config>(std::get<ComputeHardwareConfig>(producer_compute.hw_config));
    producer_config.enable_32_bit_dest = true;
    producer_config.unpack_modes = {{DFBSpecName{"dfb_0"}, UnpackMode::UnpackToDest}};

    auto consumer_dm = MakeMinimalGen2DMKernel("consumer_dm");

    auto dfb = MakeMinimalDFB("dfb_0");
    dfb.data_format_metadata = tt::DataFormat::Float32;

    producer_compute.dfb_bindings.push_back(ProducerOf(DFBSpecName{"dfb_0"}, "out"));
    consumer_dm.dfb_bindings.push_back(ConsumerOf(DFBSpecName{"dfb_0"}, "in"));

    spec.kernels = {producer_compute, consumer_dm};
    spec.dataflow_buffers = {dfb};
    spec.work_units =
        std::vector<WorkUnitSpec>{MakeMinimalWorkUnit("work_unit", node, {"producer_compute", "consumer_dm"})};

    EXPECT_NO_THROW(MakeProgramFromSpec(*mesh_device_, spec));
}

TEST_F(ProgramSpecTestQuasar, UnpackToDestFp32WithoutFp32DestAccEnFails) {
    // A 32-bit-format DFB (here Float32) cannot be unpacked into a 16-bit Dest, so UnpackToDest
    // on a consumed 32-bit DFB with enable_32_bit_dest=false is rejected on every generation.
    ProgramSpec spec = MakeMinimalValidProgramSpec();
    for (auto& dfb : spec.dataflow_buffers) {
        if (dfb.unique_id == DFBSpecName{"dfb_0"}) {
            dfb.data_format_metadata = tt::DataFormat::Float32;
        }
    }
    for (auto& kernel : spec.kernels) {
        if (kernel.is_compute_kernel()) {
            auto& config = std::get<ComputeGen2Config>(std::get<ComputeHardwareConfig>(kernel.hw_config));
            // enable_32_bit_dest stays at its default (false).
            config.unpack_modes = {{DFBSpecName{"dfb_0"}, UnpackMode::UnpackToDest}};
        }
    }
    EXPECT_THAT(
        [&] { MakeProgramFromSpec(*mesh_device_, spec); },
        ::testing::ThrowsMessage<std::runtime_error>(
            ::testing::HasSubstr("A 32-bit datum cannot be unpacked into a 16-bit Dest register")));
}

TEST_F(ProgramSpecTestQuasar, ConsumerUnpackToDestBelow32BitWithoutEnableSucceeds) {
    // Gen2 has no unpack-to-Dest performance penalty, so UnpackToDest on a consumed <=16-bit DFB
    // is accepted even without enable_32_bit_dest (a 16-bit Dest holds a <=16-bit datum). On Gen1
    // the same spec is rejected as bad-for-perf — see the ProgramSpecTestGen1 counterpart.
    ProgramSpec spec = MakeMinimalValidProgramSpec();  // dfb_0 is Float16_b, consumed by compute_kernel
    for (auto& kernel : spec.kernels) {
        if (kernel.is_compute_kernel()) {
            auto& config = std::get<ComputeGen2Config>(std::get<ComputeHardwareConfig>(kernel.hw_config));
            // enable_32_bit_dest stays at its default (false).
            config.unpack_modes = {{DFBSpecName{"dfb_0"}, UnpackMode::UnpackToDest}};
        }
    }
    EXPECT_NO_THROW(MakeProgramFromSpec(*mesh_device_, spec));
}

TEST_F(ProgramSpecTestQuasar, FP32DFBWithDefaultUnpackToDestModeSucceeds) {
    // UnpackToSrc is always a valid value, even outside the (CONSUMER + FP32 + enable_32_bit_dest) triple.
    ProgramSpec spec = MakeMinimalValidProgramSpec();
    for (auto& dfb : spec.dataflow_buffers) {
        if (dfb.unique_id == DFBSpecName{"dfb_0"}) {
            dfb.data_format_metadata = tt::DataFormat::Float32;
        }
    }
    for (auto& kernel : spec.kernels) {
        if (kernel.is_compute_kernel()) {
            auto& config = std::get<ComputeGen2Config>(std::get<ComputeHardwareConfig>(kernel.hw_config));
            config.enable_32_bit_dest = true;
            config.unpack_modes = {{DFBSpecName{"dfb_0"}, UnpackMode::UnpackToSrc}};
        }
    }
    EXPECT_NO_THROW(MakeProgramFromSpec(*mesh_device_, spec));
}

TEST_F(ProgramSpecTestQuasar, DataFormatNotSupportedOnTargetArchitectureFails) {
    NodeCoord node{0, 0};

    ProgramSpec spec;
    spec.name = "test_program";

    auto producer = MakeMinimalGen2DMKernel("producer");
    auto consumer = MakeMinimalGen2ComputeKernel("consumer");
    auto dfb = MakeMinimalDFB("dfb");

    // Legacy block-float format; not supported on Quasar.
    dfb.data_format_metadata = tt::DataFormat::Bfp8;

    producer.dfb_bindings.push_back(ProducerOf(DFBSpecName{"dfb"}, "out"));
    consumer.dfb_bindings.push_back(ConsumerOf(DFBSpecName{"dfb"}, "in"));

    spec.kernels = {producer, consumer};
    spec.dataflow_buffers = {dfb};
    spec.work_units = std::vector<WorkUnitSpec>{MakeMinimalWorkUnit("work_unit", node, {"producer", "consumer"})};

    EXPECT_THAT(
        [&] { MakeProgramFromSpec(*mesh_device_, spec); },
        ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr("DFB 'dfb' has data format")));
}

TEST_F(ProgramSpecTestQuasar, TooManyDFBsFailsValidation) {
    // The hard upper limit on DFBs is hal::get_arch_num_circular_buffers().
    // Exceeding it should fail validation with a clear error, rather than blowing
    // up downstream during JIT.
    NodeCoord node{0, 0};

    ProgramSpec spec;
    spec.name = "test_program";

    auto producer = MakeMinimalGen2DMKernel("producer");
    auto consumer = MakeMinimalGen2ComputeKernel("consumer");

    const uint32_t too_many = tt::tt_metal::hal::get_arch_num_circular_buffers() + 1;
    for (uint32_t i = 0; i < too_many; ++i) {
        std::string name = "dfb_" + std::to_string(i);
        auto dfb = MakeMinimalDFB(name);
        dfb.data_format_metadata = tt::DataFormat::Float16_b;
        spec.dataflow_buffers.push_back(dfb);
        producer.dfb_bindings.push_back(ProducerOf(DFBSpecName{name}, "p_" + std::to_string(i)));
        consumer.dfb_bindings.push_back(ConsumerOf(DFBSpecName{name}, "c_" + std::to_string(i)));
    }

    spec.kernels = {producer, consumer};
    spec.work_units = std::vector<WorkUnitSpec>{MakeMinimalWorkUnit("work_unit", node, {"producer", "consumer"})};

    const std::string expected_substr = "too many DataflowBufferSpecs (" + std::to_string(too_many) + ")";
    EXPECT_THAT(
        [&] { MakeProgramFromSpec(*mesh_device_, spec); },
        ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr(expected_substr)));
}

// ============================================================================
// WorkUnitSpec Validation Tests
// ============================================================================

TEST_F(ProgramSpecTestQuasar, EmptyWorkUnitSpecsFails) {
    ProgramSpec spec;
    spec.name = "test_program";

    auto kernel = MakeMinimalGen2DMKernel("kernel");
    spec.kernels = {kernel};
    spec.work_units = std::vector<WorkUnitSpec>{};  // Empty!

    EXPECT_THAT(
        [&] { MakeProgramFromSpec(*mesh_device_, spec); },
        ::testing::ThrowsMessage<std::runtime_error>(
            ::testing::HasSubstr("Kernel 'kernel' is not referenced by any WorkUnitSpec")));
}

TEST_F(ProgramSpecTestQuasar, WorkUnitSpecWithNoKernelsFails) {
    NodeCoord node{0, 0};

    ProgramSpec spec;
    spec.name = "test_program";

    auto kernel = MakeMinimalGen2DMKernel("kernel");
    spec.kernels = {kernel};

    WorkUnitSpec work_unit;
    work_unit.name = "work_unit";
    work_unit.target_nodes = node;
    work_unit.kernels = {};  // No kernels!
    spec.work_units = std::vector<WorkUnitSpec>{work_unit};

    EXPECT_THAT(
        [&] { MakeProgramFromSpec(*mesh_device_, spec); },
        ::testing::ThrowsMessage<std::runtime_error>(
            ::testing::HasSubstr("Kernel 'kernel' is not referenced by any WorkUnitSpec")));
}

TEST_F(ProgramSpecTestQuasar, WorkUnitSpecReferencesUnknownKernelFails) {
    NodeCoord node{0, 0};

    ProgramSpec spec;
    spec.name = "test_program";

    auto kernel = MakeMinimalGen2DMKernel("real_kernel");
    spec.kernels = {kernel};

    // WorkUnit references a kernel that doesn't exist
    spec.work_units = std::vector<WorkUnitSpec>{MakeMinimalWorkUnit("work_unit", node, {"nonexistent_kernel"})};

    EXPECT_THAT(
        [&] { MakeProgramFromSpec(*mesh_device_, spec); },
        ::testing::ThrowsMessage<std::runtime_error>(
            ::testing::HasSubstr("WorkUnitSpec 'work_unit' references unknown kernel 'nonexistent_kernel'")));
}

TEST_F(ProgramSpecTestQuasar, OverlappingWorkUnitSpecsFails) {
    // Two work_units cannot target overlapping nodes
    NodeCoord node{0, 0};

    ProgramSpec spec;
    spec.name = "test_program";

    auto kernel1 = MakeMinimalGen2DMKernel("kernel1");
    auto kernel2 = MakeMinimalGen2DMKernel("kernel2");
    spec.kernels = {kernel1, kernel2};

    // Both work_units target the same node
    spec.work_units = std::vector<WorkUnitSpec>{
        MakeMinimalWorkUnit("work_unit1", node, {"kernel1"}), MakeMinimalWorkUnit("work_unit2", node, {"kernel2"})};

    EXPECT_THAT(
        [&] { MakeProgramFromSpec(*mesh_device_, spec); },
        ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr("overlap in target nodes")));
}

TEST_F(ProgramSpecTestQuasar, KernelNotInAnyWorkUnitSpecFails) {
    NodeCoord node{0, 0};

    ProgramSpec spec;
    spec.name = "test_program";

    auto kernel1 = MakeMinimalGen2DMKernel("kernel1");
    auto kernel2 = MakeMinimalGen2DMKernel("kernel2");  // Not in any work_unit!
    spec.kernels = {kernel1, kernel2};

    spec.work_units = std::vector<WorkUnitSpec>{MakeMinimalWorkUnit("work_unit", node, {"kernel1"})};  // Only kernel1

    EXPECT_THAT(
        [&] { MakeProgramFromSpec(*mesh_device_, spec); },
        ::testing::ThrowsMessage<std::runtime_error>(
            ::testing::HasSubstr("Kernel 'kernel2' is not referenced by any WorkUnitSpec")));
}

TEST_F(ProgramSpecTestQuasar, WorkUnitExceedsDMCoreBudgetFails) {
    NodeCoord node{0, 0};

    ProgramSpec spec;
    spec.name = "test_program";

    // Create enough DM kernels to exceed the 8 DM core budget
    auto kernel1 = MakeMinimalGen2DMKernel("dm1", 3);
    auto kernel2 = MakeMinimalGen2DMKernel("dm2", 3);
    auto kernel3 = MakeMinimalGen2DMKernel("dm3", 3);  // Total: 9 > 8

    spec.kernels = {kernel1, kernel2, kernel3};
    spec.work_units = std::vector<WorkUnitSpec>{MakeMinimalWorkUnit("work_unit", node, {"dm1", "dm2", "dm3"})};

    EXPECT_THAT(
        [&] { MakeProgramFromSpec(*mesh_device_, spec); },
        ::testing::ThrowsMessage<std::runtime_error>(
            ::testing::HasSubstr("WorkUnitSpec 'work_unit' requests 9 data movement cores")));
}

TEST_F(ProgramSpecTestQuasar, WorkUnitExceedsComputeCoreBudgetFails) {
    NodeCoord node{0, 0};

    ProgramSpec spec;
    spec.name = "test_program";

    // Create enough compute kernels to exceed the 4 Tensix core budget (2+4=6).
    // (Legal thread counts on Quasar are 1, 2, 4; 3 is explicitly disallowed.)
    auto kernel1 = MakeMinimalGen2ComputeKernel("compute1", 2);
    auto kernel2 = MakeMinimalGen2ComputeKernel("compute2", 4);

    spec.kernels = {kernel1, kernel2};
    spec.work_units = std::vector<WorkUnitSpec>{MakeMinimalWorkUnit("work_unit", node, {"compute1", "compute2"})};

    EXPECT_THAT(
        [&] { MakeProgramFromSpec(*mesh_device_, spec); },
        ::testing::ThrowsMessage<std::runtime_error>(
            ::testing::HasSubstr("WorkUnitSpec 'work_unit' needs 6 Tensix engines")));
}

TEST_F(ProgramSpecTestQuasar, WorkUnitWithMultipleComputeKernelsFails) {
    // A work_unit can have at most one compute kernel
    NodeCoord node{0, 0};

    ProgramSpec spec;
    spec.name = "test_program";

    auto compute1 = MakeMinimalGen2ComputeKernel("compute1");
    auto compute2 = MakeMinimalGen2ComputeKernel("compute2");

    spec.kernels = {compute1, compute2};
    spec.work_units = std::vector<WorkUnitSpec>{MakeMinimalWorkUnit("work_unit", node, {"compute1", "compute2"})};

    EXPECT_THAT(
        [&] { MakeProgramFromSpec(*mesh_device_, spec); },
        ::testing::ThrowsMessage<std::runtime_error>(
            ::testing::HasSubstr("WorkUnitSpec 'work_unit' has more than one compute kernel")));
}

TEST_F(ProgramSpecTestQuasar, LocalDFBConsumerOnNodeWithoutProducerFails) {
    // A local DFB requires every node it lives on to host both a producer and a consumer. Here the
    // consumer covers an extra node where no producer runs, so the per-node census rejects it.
    NodeCoord node0{0, 0};
    NodeCoord node1{1, 0};

    ProgramSpec spec;
    spec.name = "test_program";

    auto producer = MakeMinimalGen2DMKernel("producer");
    auto consumer = MakeMinimalGen2DMKernel("consumer");
    auto dfb = MakeMinimalDFB("dfb");

    producer.dfb_bindings.push_back(ProducerOf(DFBSpecName{"dfb"}, "out"));
    consumer.dfb_bindings.push_back(ConsumerOf(DFBSpecName{"dfb"}, "in"));

    spec.kernels = {producer, consumer};
    spec.dataflow_buffers = {dfb};
    // Producer covers node0 only; consumer covers node0 and node1 — node1 has a consumer but no producer.
    spec.work_units = std::vector<WorkUnitSpec>{
        MakeMinimalWorkUnit("work_unit_0", node0, {"producer", "consumer"}),
        MakeMinimalWorkUnit("work_unit_1", node1, {"consumer"}),
    };

    EXPECT_THAT(
        [&] { MakeProgramFromSpec(*mesh_device_, spec); },
        ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr("This node has a consumer but no producer")));
}

// ============================================================================
// Programs Creation Tests
// ============================================================================
// These verify that valid ProgramSpec configurations produce a Program without throwing.
// They exercise the full MakeProgramFromSpec pipeline, but only on mock device.
//
// Coverage gaps (JIT compilation, device-side execution) are covered by HW tests.
// (see test_program_spec_hw.cpp)

TEST_F(ProgramSpecTestQuasar, MinimalValidProgramSpecSucceeds) {
    ProgramSpec spec = MakeMinimalValidProgramSpec();

    EXPECT_NO_THROW(MakeProgramFromSpec(*mesh_device_, spec));
}

TEST_F(ProgramSpecTestQuasar, DFBSelfLoopOnComputeKernelSucceeds) {
    // A compute kernel that self-loops a DFB (binds it as both producer and consumer) is legal: it
    // lowers to the intra-Tensix packer->unpacker flow (TensixScope::INTRA), which the Metal 2.0
    // layer applies automatically. There is no user-facing self-loop scope option.
    NodeCoord node{0, 0};

    ProgramSpec spec;
    spec.name = "compute_self_loop";

    auto compute = MakeMinimalGen2ComputeKernel("compute");

    auto dfb = MakeMinimalDFB("dfb");
    dfb.data_format_metadata = tt::DataFormat::Float16_b;
    // INTRA-tensix self-loop: no DM endpoint, so the spec-to-impl translation produces
    // enable_{producer,consumer}_implicit_sync=false at the lower DFB layer automatically.

    compute.dfb_bindings.push_back(ProducerOf(DFBSpecName{"dfb"}, "out"));
    compute.dfb_bindings.push_back(ConsumerOf(DFBSpecName{"dfb"}, "in"));

    spec.kernels = {compute};
    spec.dataflow_buffers = {dfb};
    spec.work_units = std::vector<WorkUnitSpec>{MakeMinimalWorkUnit("work_unit", node, {"compute"})};

    EXPECT_NO_THROW(MakeProgramFromSpec(*mesh_device_, spec));
}

TEST_F(ProgramSpecTestQuasar, DMOnlyProgramSucceeds) {
    // A program with only DM kernels (no compute)
    NodeCoord node{0, 0};

    ProgramSpec spec;
    spec.name = "dm_only_program";

    auto producer = MakeMinimalGen2DMKernel("producer");
    auto consumer = MakeMinimalGen2DMKernel("consumer");
    auto dfb = MakeMinimalDFB("dfb");
    // No data_format_metadata needed for DM-only DFBs

    producer.dfb_bindings.push_back(ProducerOf(DFBSpecName{"dfb"}, "out"));
    consumer.dfb_bindings.push_back(ConsumerOf(DFBSpecName{"dfb"}, "in"));

    spec.kernels = {producer, consumer};
    spec.dataflow_buffers = {dfb};
    spec.work_units = std::vector<WorkUnitSpec>{MakeMinimalWorkUnit("work_unit", node, {"producer", "consumer"})};

    EXPECT_NO_THROW(MakeProgramFromSpec(*mesh_device_, spec));
}

TEST_F(ProgramSpecTestQuasar, MultiNodeProgramSucceeds) {
    // A program spanning multiple nodes
    NodeRange nodes{{0, 0}, {1, 1}};  // 2x2 grid

    ProgramSpec spec;
    spec.name = "multi_node_program";

    auto producer = MakeMinimalGen2DMKernel("producer");
    auto consumer = MakeMinimalGen2DMKernel("consumer");
    auto dfb = MakeMinimalDFB("dfb");

    producer.dfb_bindings.push_back(ProducerOf(DFBSpecName{"dfb"}, "out"));
    consumer.dfb_bindings.push_back(ConsumerOf(DFBSpecName{"dfb"}, "in"));

    spec.kernels = {producer, consumer};
    spec.dataflow_buffers = {dfb};
    spec.work_units = std::vector<WorkUnitSpec>{MakeMinimalWorkUnit("work_unit", nodes, {"producer", "consumer"})};

    EXPECT_NO_THROW(MakeProgramFromSpec(*mesh_device_, spec));
}

TEST_F(ProgramSpecTestQuasar, MultipleWorkUnitsOnDifferentNodesSucceeds) {
    // Multiple work_units on non-overlapping nodes
    NodeCoord node0{0, 0};
    NodeCoord node1{1, 0};
    NodeRangeSet all_nodes(std::set<NodeRange>{NodeRange{node0, node0}, NodeRange{node1, node1}});

    ProgramSpec spec;
    spec.name = "multi_work_unit_program";

    // Kernels span both nodes
    auto kernel = MakeMinimalGen2DMKernel("kernel");
    spec.kernels = {kernel};

    // Two work_units, each on a different node
    spec.work_units = std::vector<WorkUnitSpec>{
        MakeMinimalWorkUnit("work_unit0", node0, {"kernel"}), MakeMinimalWorkUnit("work_unit1", node1, {"kernel"})};

    EXPECT_NO_THROW(MakeProgramFromSpec(*mesh_device_, spec));
}

TEST_F(ProgramSpecTestQuasar, MaxDMThreadsSucceeds) {
    // Use exactly 6 DM threads (the maximum available to the user)
    NodeCoord node{0, 0};

    ProgramSpec spec;
    spec.name = "max_dm_threads";

    auto producer = MakeMinimalGen2DMKernel("producer", 3);
    auto consumer = MakeMinimalGen2DMKernel("consumer", 3);  // Total: 6
    auto dfb = MakeMinimalDFB("dfb");
    dfb.num_entries = 9;  // must be a multiple of the number of threads

    producer.dfb_bindings.push_back(ProducerOf(DFBSpecName{"dfb"}, "out"));
    consumer.dfb_bindings.push_back(ConsumerOf(DFBSpecName{"dfb"}, "in"));

    spec.kernels = {producer, consumer};
    spec.dataflow_buffers = {dfb};
    spec.work_units = std::vector<WorkUnitSpec>{MakeMinimalWorkUnit("work_unit", node, {"producer", "consumer"})};

    EXPECT_NO_THROW(MakeProgramFromSpec(*mesh_device_, spec));
}

TEST_F(ProgramSpecTestQuasar, MaxComputeThreadsSucceeds) {
    // Use exactly 4 compute threads (the maximum available)
    NodeCoord node{0, 0};

    ProgramSpec spec;
    spec.name = "max_compute_threads";

    auto dm = MakeMinimalGen2DMKernel("dm");
    auto compute = MakeMinimalGen2ComputeKernel("compute", 4);  // Max threads

    auto dfb = MakeMinimalDFB("dfb");
    dfb.data_format_metadata = tt::DataFormat::Float16_b;
    dfb.num_entries = 4;  // must be a multiple of the number of threads

    dm.dfb_bindings.push_back(ProducerOf(DFBSpecName{"dfb"}, "out"));
    compute.dfb_bindings.push_back(ConsumerOf(DFBSpecName{"dfb"}, "in"));

    spec.kernels = {dm, compute};
    spec.dataflow_buffers = {dfb};
    spec.work_units = std::vector<WorkUnitSpec>{MakeMinimalWorkUnit("work_unit", node, {"dm", "compute"})};

    EXPECT_NO_THROW(MakeProgramFromSpec(*mesh_device_, spec));
}

TEST_F(ProgramSpecTestQuasar, MultipleDFBsSucceeds) {
    NodeCoord node{0, 0};

    ProgramSpec spec;
    spec.name = "multi_dfb_program";

    auto producer = MakeMinimalGen2DMKernel("producer");
    auto consumer = MakeMinimalGen2ComputeKernel("consumer");

    auto dfb1 = MakeMinimalDFB("dfb1");
    dfb1.data_format_metadata = tt::DataFormat::Float16_b;
    auto dfb2 = MakeMinimalDFB("dfb2");
    dfb2.data_format_metadata = tt::DataFormat::Int8;

    producer.dfb_bindings.push_back(ProducerOf(DFBSpecName{"dfb1"}, "out1"));
    producer.dfb_bindings.push_back(ProducerOf(DFBSpecName{"dfb2"}, "out2"));
    consumer.dfb_bindings.push_back(ConsumerOf(DFBSpecName{"dfb1"}, "in1"));
    consumer.dfb_bindings.push_back(ConsumerOf(DFBSpecName{"dfb2"}, "in2"));

    spec.kernels = {producer, consumer};
    spec.dataflow_buffers = {dfb1, dfb2};
    spec.work_units = std::vector<WorkUnitSpec>{MakeMinimalWorkUnit("work_unit", node, {"producer", "consumer"})};

    EXPECT_NO_THROW(MakeProgramFromSpec(*mesh_device_, spec));
}

TEST_F(ProgramSpecTestQuasar, CompilerOptionsDefinesSucceeds) {
    ProgramSpec spec = MakeMinimalValidProgramSpec();

    // Add some defines
    spec.kernels[0].compiler_options.defines = {{"MY_DEFINE", "42"}, {"ANOTHER_DEFINE", "foo"}};

    EXPECT_NO_THROW(MakeProgramFromSpec(*mesh_device_, spec));
}

TEST_F(ProgramSpecTestQuasar, CompileTimeArgBindingsSucceeds) {
    ProgramSpec spec = MakeMinimalValidProgramSpec();

    // Add compile-time arg bindings
    spec.kernels[0].compile_time_args = {{"arg1", 100}, {"arg2", 200}};

    EXPECT_NO_THROW(MakeProgramFromSpec(*mesh_device_, spec));
}

TEST_F(ProgramSpecTestQuasar, RuntimeArgsSchemaSucceeds) {
    ProgramSpec spec = MakeMinimalValidProgramSpec();

    // Add runtime args schema
    spec.kernels[0].advanced_options = KernelAdvancedOptions{.num_runtime_varargs = 3, .num_common_runtime_varargs = 2};

    EXPECT_NO_THROW(MakeProgramFromSpec(*mesh_device_, spec));
}

TEST_F(ProgramSpecTestQuasar, VarargPerNodeOverlapFails) {
    // Rule: overlapping entries in num_runtime_varargs_per_node are an error, even when
    // their counts agree. Overlap suggests a user mistake.
    NodeCoord node_a{0, 0};
    NodeCoord node_b{1, 0};
    NodeRangeSet both{std::vector<NodeRange>{NodeRange{node_a, node_a}, NodeRange{node_b, node_b}}};

    ProgramSpec spec;
    spec.name = "vararg_overlap_test";
    auto kernel = MakeMinimalGen2DMKernel("dm_kernel");
    kernel.advanced_options = KernelAdvancedOptions{
        .num_runtime_varargs_per_node = Table<Nodes, uint32_t>{{both, 3}, {node_a, 3}},  // node_a listed twice
    };
    spec.kernels = {kernel};
    spec.work_units = std::vector<WorkUnitSpec>{MakeMinimalWorkUnit("work_unit_0", both, {"dm_kernel"})};

    EXPECT_THAT(
        [&] { MakeProgramFromSpec(*mesh_device_, spec); },
        ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr("overlapping entries")));
}

// ============================================================================
// Edge Cases and Boundary Tests
// ============================================================================

TEST_F(ProgramSpecTestQuasar, NodeRangeSetTargetNodesSucceeds) {
    // Test with NodeRangeSet (multiple disjoint ranges)
    NodeRangeSet nodes(std::set<NodeRange>{NodeRange{{0, 0}, {0, 1}}, NodeRange{{2, 0}, {2, 1}}});

    ProgramSpec spec;
    spec.name = "range_set_program";

    auto producer = MakeMinimalGen2DMKernel("producer");
    auto consumer = MakeMinimalGen2DMKernel("consumer");
    auto dfb = MakeMinimalDFB("dfb");

    producer.dfb_bindings.push_back(ProducerOf(DFBSpecName{"dfb"}, "out"));
    consumer.dfb_bindings.push_back(ConsumerOf(DFBSpecName{"dfb"}, "in"));

    spec.kernels = {producer, consumer};
    spec.dataflow_buffers = {dfb};
    spec.work_units = std::vector<WorkUnitSpec>{MakeMinimalWorkUnit("work_unit", nodes, {"producer", "consumer"})};

    EXPECT_NO_THROW(MakeProgramFromSpec(*mesh_device_, spec));
}

TEST_F(ProgramSpecTestQuasar, SourceCodeKernelSucceeds) {
    ProgramSpec spec = MakeMinimalValidProgramSpec();

    // Change to inline source code
    spec.kernels[0].source = KernelSpec::SourceCode{"void kernel_main() {}"};

    EXPECT_NO_THROW(MakeProgramFromSpec(*mesh_device_, spec));
}

TEST_F(ProgramSpecTestQuasar, ComputeConfigMathFidelitySucceeds) {
    ProgramSpec spec = MakeMinimalValidProgramSpec();

    // Find the compute kernel and set math fidelity options
    for (auto& kernel : spec.kernels) {
        if (kernel.is_compute_kernel()) {
            auto& config = std::get<ComputeGen2Config>(std::get<ComputeHardwareConfig>(kernel.hw_config));
            config.fpu_math_fidelity = MathFidelity::LoFi;
            config.enable_32_bit_dest = true;
            config.sfpu_precision_mode = tt::tt_metal::Precision::Approximate;
        }
    }

    EXPECT_NO_THROW(MakeProgramFromSpec(*mesh_device_, spec));
}

TEST_F(ProgramSpecTestQuasar, ValidUnpackToDestModeSucceeds) {
    ProgramSpec spec = MakeMinimalValidProgramSpec();

    // The full meaningfulness triple: FP32 DFB, consumed by a compute kernel with
    // enable_32_bit_dest=true. UnpackToDest is meaningful here.
    for (auto& dfb : spec.dataflow_buffers) {
        if (dfb.unique_id == DFBSpecName{"dfb_0"}) {
            dfb.data_format_metadata = tt::DataFormat::Float32;
        }
    }
    for (auto& kernel : spec.kernels) {
        if (kernel.is_compute_kernel()) {
            auto& config = std::get<ComputeGen2Config>(std::get<ComputeHardwareConfig>(kernel.hw_config));
            config.enable_32_bit_dest = true;
            config.unpack_modes = {{DFBSpecName{"dfb_0"}, UnpackMode::UnpackToDest}};
        }
    }

    EXPECT_NO_THROW(MakeProgramFromSpec(*mesh_device_, spec));
}

TEST_F(ProgramSpecTestQuasar, UnpackToDestModePlacedAtDfbIdSlot) {
    // Regression test for the unpack_to_dest_mode sizing bug: the JIT consumer
    // iterates hal::get_arch_num_circular_buffers() slots, so BuildUnpackToDestModeVector
    // must size the vector to that count and place each user-supplied mode at slot dfb_id.
    // Pre-fix code sized the vector to the number of DFBs, which produced silent
    // OOB reads downstream when num_dfbs < max_cbs.
    NodeCoord node{0, 0};

    ProgramSpec spec;
    spec.name = "test_program";

    auto producer = MakeMinimalGen2DMKernel("producer");
    auto consumer = MakeMinimalGen2ComputeKernel("consumer");

    auto dfb0 = MakeMinimalDFB("dfb_0");
    dfb0.data_format_metadata = tt::DataFormat::Float16_b;
    auto dfb1 = MakeMinimalDFB("dfb_1");
    // dfb_1 is FP32 so the user can opt into UnpackToDest on it.
    dfb1.data_format_metadata = tt::DataFormat::Float32;

    producer.dfb_bindings.push_back(ProducerOf(DFBSpecName{"dfb_0"}, "out0"));
    producer.dfb_bindings.push_back(ProducerOf(DFBSpecName{"dfb_1"}, "out1"));
    consumer.dfb_bindings.push_back(ConsumerOf(DFBSpecName{"dfb_0"}, "in0"));
    consumer.dfb_bindings.push_back(ConsumerOf(DFBSpecName{"dfb_1"}, "in1"));

    auto& compute_config = std::get<ComputeGen2Config>(std::get<ComputeHardwareConfig>(consumer.hw_config));
    compute_config.enable_32_bit_dest = true;
    compute_config.unpack_modes = {{DFBSpecName{"dfb_1"}, UnpackMode::UnpackToDest}};

    spec.kernels = {producer, consumer};
    spec.dataflow_buffers = {dfb0, dfb1};
    spec.work_units = std::vector<WorkUnitSpec>{MakeMinimalWorkUnit("work_unit", node, {"producer", "consumer"})};

    Program program = MakeProgramFromSpec(*mesh_device_, spec);

    // Inspect the constructed compute kernel's QuasarComputeConfig:
    //  - vector must be sized to max_cbs (so JIT's iteration up to max_cbs is in-bounds)
    //  - the user-supplied mode must land at slot dfb_id (not at iteration order)
    //  - other slots stay Default
    const auto& impl = program.impl();
    auto consumer_kernel = impl.get_kernel_by_spec_name("consumer");
    const auto built_config_variant = consumer_kernel->config();
    const auto& built_config = std::get<experimental::quasar::QuasarComputeConfig>(built_config_variant);

    EXPECT_EQ(built_config.unpack_to_dest_mode.size(), tt::tt_metal::hal::get_arch_num_circular_buffers());
    EXPECT_EQ(built_config.unpack_to_dest_mode[impl.get_dfb_handle("dfb_1")], UnpackToDestMode::UnpackToDestFp32);
    EXPECT_EQ(built_config.unpack_to_dest_mode[impl.get_dfb_handle("dfb_0")], UnpackToDestMode::Default);
}

// ============================================================================
// Compute-config translation stability (defaults + inversion/enum)
// ============================================================================
// These tests pin the public ComputeGen{1,2}Config -> internal
// ComputeConfig / QuasarComputeConfig translation at the boundary where the
// field rename is absorbed (MakeGen1ComputeConfig / MakeGen2ComputeConfig).
//
// Several of these knobs are performance / numerical-precision settings that do
// NOT change a functional pass/fail result, so a flipped inversion
// (dst_full_sync_en <-> double_buffer_dest) or a wrong precision-enum direction
// would be invisible to the behavioral tests. Asserting the internal values
// documents that the defaults must not change and guards the conversions
// against silent drift. (The public -> internal translation is Metal-side and
// testable here; the TTNN ComputeKernelConfig -> public bridge lives above this
// layer and is out of scope for a Metal unit test.)

TEST_F(ProgramSpecTestQuasar, ComputeGen2ConfigDefaultsMapToInternalDefaults) {
    // A default ComputeGen2Config{} must yield the historical internal QuasarComputeConfig defaults.
    ProgramSpec spec = MakeMinimalValidProgramSpec();  // compute_kernel carries a default ComputeGen2Config
    Program program = MakeProgramFromSpec(*mesh_device_, spec);

    const auto built_variant = program.impl().get_kernel_by_spec_name("compute_kernel")->config();
    const auto& built = std::get<experimental::quasar::QuasarComputeConfig>(built_variant);
    EXPECT_EQ(built.math_fidelity, MathFidelity::HiFi4);
    EXPECT_FALSE(built.fp32_dest_acc_en);
    EXPECT_FALSE(built.dst_full_sync_en);      // double_buffer_dest defaults true -> !true
    EXPECT_FALSE(built.math_approx_mode);      // sfpu_precision_mode defaults Precise
    EXPECT_FALSE(built.enable_2x_src_format);  // enable_2x_src_register defaults false
    EXPECT_FALSE(built.unpack_to_dest_en);
}

TEST_F(ProgramSpecTestQuasar, ComputeGen2ConfigInversionAndEnumMapToInternal) {
    // Non-default polarity: the double_buffer_dest inversion and the SFPU precision-enum mapping
    // must reach the internal config correctly. Guards the case a defaults-only check would miss
    // (a flip compensated by a changed default).
    ProgramSpec spec = MakeMinimalValidProgramSpec();
    for (auto& kernel : spec.kernels) {
        if (kernel.is_compute_kernel()) {
            auto& config = std::get<ComputeGen2Config>(std::get<ComputeHardwareConfig>(kernel.hw_config));
            config.double_buffer_dest = false;                                  // -> internal dst_full_sync_en == true
            config.sfpu_precision_mode = tt::tt_metal::Precision::Approximate;  // -> internal math_approx_mode == true
        }
    }
    Program program = MakeProgramFromSpec(*mesh_device_, spec);

    const auto built_variant = program.impl().get_kernel_by_spec_name("compute_kernel")->config();
    const auto& built = std::get<experimental::quasar::QuasarComputeConfig>(built_variant);
    EXPECT_TRUE(built.dst_full_sync_en);
    EXPECT_TRUE(built.math_approx_mode);
}

// ============================================================================
// Processor Assignment Edge Cases
// ============================================================================
// Here, we test several edge cases:
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
//    NOTE: Our plan is to keep the simplifying assumption for now.
//    We issue a clear message if the assumption is ever violated in the real world.
//
//   C) KERNELS COUPLED THROUGH MULTI-DFB BINDINGS
//    When a DFBSpec is bound by multiple KernelSpecs (which is legal, provided
//    that the invariant that any given DFB instance has one one producer kernel
//    instance and only one consumer kernel instance), the resulting cross-kernel
//    coupling through the shared DFB binding induces additional DM solver constraints.
//
//    NOTE: The original plan called for lifting this artificial constraint once LLK adopted
//    DFBAccessor using implicit RTAs. However, to realize performance gains, we're
//    chosen instead to GUARANTEE using implicit CTAs for DFBAccessor. This
//    constraint is therefore permanent.
//
//    If we were ever to start encountering serious unsolvable-Program issues as a result
//    we might consider revisiting this decision. However, an unsolvable Program could
//    can always be worked around by artificially dividing a KernelSpec (at the expense of
//    dispatch overhead).

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

    auto k_a = MakeMinimalGen2DMKernel("k_a", 3);
    auto k_ab = MakeMinimalGen2DMKernel("k_ab", 3);
    auto k_bc = MakeMinimalGen2DMKernel("k_bc", 3);
    auto k_c = MakeMinimalGen2DMKernel("k_c", 3);

    auto work_unit_a1 = MakeMinimalWorkUnit("work_unit_a1", node_a, {"k_a", "k_ab"});
    auto work_unit_b1 = MakeMinimalWorkUnit("work_unit_b1", node_b, {"k_ab", "k_bc"});
    auto work_unit_c1 = MakeMinimalWorkUnit("work_unit_c1", node_c, {"k_bc", "k_c"});
    auto work_unit_c2 = MakeMinimalWorkUnit("work_unit_c2", node_c, {"k_c", "k_bc"});

    // Helper to create a ProgramSpec with a given id and work_unit ordering
    auto make_spec =
        [&](const std::string& id, std::vector<KernelSpec> kernels, const std::vector<WorkUnitSpec>& work_units) {
            ProgramSpec spec;
            spec.name = id;
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
            EXPECT_NO_THROW(MakeProgramFromSpec(*mesh_device_, make_spec("", kernel_perm, work_unit_perm)));
        }
    }
    for (const auto& kernel_perm : kernel_permutations) {
        for (const auto& work_unit_perm : work_unit_permutations2) {
            EXPECT_NO_THROW(MakeProgramFromSpec(*mesh_device_, make_spec("", kernel_perm, work_unit_perm)));
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
    spec.name = "triangle_of_doom";

    auto kernel_a = MakeMinimalGen2DMKernel("kernel_a", 3);
    auto kernel_b = MakeMinimalGen2DMKernel("kernel_b", 3);
    auto kernel_c = MakeMinimalGen2DMKernel("kernel_c", 3);

    spec.kernels = {kernel_a, kernel_b, kernel_c};

    spec.work_units = std::vector<WorkUnitSpec>{
        MakeMinimalWorkUnit("work_unit_00", node_00, {"kernel_a", "kernel_b"}),
        MakeMinimalWorkUnit("work_unit_01", node_01, {"kernel_a", "kernel_c"}),
        MakeMinimalWorkUnit("work_unit_02", node_02, {"kernel_b", "kernel_c"}),
    };

    // EXPECTED BEHAVIOR: FAILS due to simplifying assumption violation.
    EXPECT_THAT(
        [&] { MakeProgramFromSpec(*mesh_device_, spec); },
        ::testing::ThrowsMessage<std::runtime_error>(
            ::testing::HasSubstr("Failed to find valid processor assignments for DM kernels")));
}

// Category C: Coupling-group constraint exercised
// Multi-bound same-role DM kernels must end up with identical DM RISC masks (the DFB's
// hardware config carries one producer_risc_mask / consumer_risc_mask per side). The
// solver implements this by treating each coupling-group equivalence class as a single
// "super-kernel" with merged node coverage. Without that constraint, an unrelated DM
// kernel competing for lanes on one zone could push the producers to different lanes on
// their respective nodes — passing the greedy assignment but failing per-role mask
// uniformity.
TEST_F(ProgramSpecTestQuasar, DFBMultiBindingForcesUniformRiscMaskAcrossProducers) {
    // Scenario: zone-specialized DM producers (producer_a on node0, producer_b on node1)
    // both bound as PRODUCER of the same DFB. An unrelated 2-thread DM kernel on node0
    // consumes lanes DM2-DM3 (the lanes the un-constrained solver would greedily hand to
    // producer_a). If the producers weren't coupled, producer_a would get bumped to DM4
    // on node0 while producer_b kept DM2 on node1 — different masks. The coupling-group
    // solver instead picks a lane available on BOTH producer nodes first, then lets the
    // unrelated kernel work around it.
    NodeCoord node0{0, 0};
    NodeCoord node1{1, 0};

    ProgramSpec spec;
    spec.name = "test_program";

    auto producer_a = MakeMinimalGen2DMKernel("producer_a", /*num_threads=*/1);
    auto producer_b = MakeMinimalGen2DMKernel("producer_b", /*num_threads=*/1);
    auto unrelated_dm = MakeMinimalGen2DMKernel("unrelated_dm", /*num_threads=*/2);
    auto consumer = MakeMinimalGen2ComputeKernel("consumer");

    auto dfb = MakeMinimalDFB("dfb");
    dfb.data_format_metadata = tt::DataFormat::Float16_b;

    producer_a.dfb_bindings.push_back(ProducerOf(DFBSpecName{"dfb"}, "out"));
    producer_b.dfb_bindings.push_back(ProducerOf(DFBSpecName{"dfb"}, "out"));
    consumer.dfb_bindings.push_back(ConsumerOf(DFBSpecName{"dfb"}, "in"));
    // unrelated_dm intentionally has no DFB bindings — it just consumes DM lanes on node0.

    spec.kernels = {producer_a, producer_b, unrelated_dm, consumer};
    spec.dataflow_buffers = {dfb};
    spec.work_units = std::vector<WorkUnitSpec>{
        MakeMinimalWorkUnit("wu_g1", node0, {"producer_a", "unrelated_dm", "consumer"}),
        MakeMinimalWorkUnit("wu_g2", node1, {"producer_b", "consumer"}),
    };

    EXPECT_NO_THROW({ MakeProgramFromSpec(*mesh_device_, spec); });
}

// ============================================================================
// Aggregate Type Enforcement Tests
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
static_assert(std::is_aggregate_v<DataMovementGen1Config>, "DataMovementGen1Config must remain an aggregate");
static_assert(std::is_aggregate_v<DataMovementGen2Config>, "DataMovementGen2Config must remain an aggregate");
static_assert(std::is_aggregate_v<ComputeGen1Config>, "ComputeGen1Config must remain an aggregate");
static_assert(std::is_aggregate_v<ComputeGen2Config>, "ComputeGen2Config must remain an aggregate");
static_assert(
    std::is_aggregate_v<KernelSpec::CompilerOptions>,
    "CompilerOptions must remain an aggregate to support designated initializers");
static_assert(
    std::is_aggregate_v<DFBBinding>, "DFBBinding must remain an aggregate to support designated initializers");
static_assert(
    std::is_aggregate_v<SemaphoreBinding>,
    "SemaphoreBinding must remain an aggregate to support designated initializers");
static_assert(
    std::is_aggregate_v<KernelSpec::RuntimeArgSchema>,
    "RuntimeArgSchema must remain an aggregate to support designated initializers");
static_assert(
    std::is_aggregate_v<CrossNodeDataflowBufferSpec>,
    "CrossNodeDataflowBufferSpec must remain an aggregate to support designated initializers");

// These tests document the intended construction pattern using designated initializers.
// They serve as living documentation and will fail to compile if aggregate status is broken.

TEST(AggregateSpecTypes, KernelSpecDesignatedInitializers) {
    // Demonstrates constructing KernelSpec with designated initializers
    KernelSpec dm_kernel{
        .unique_id = KernelSpecName{"my_dm_kernel"},
        .source = KernelSpec::SourceCode{"void kernel_main() {}"},
        .num_threads = 2,
        .hw_config = DataMovementGen2Config{},
    };

    EXPECT_EQ(dm_kernel.unique_id.get(), "my_dm_kernel");
    EXPECT_EQ(dm_kernel.num_threads, 2);
    EXPECT_TRUE(dm_kernel.is_data_movement_kernel());

    KernelSpec compute_kernel{
        .unique_id = KernelSpecName{"my_compute_kernel"},
        .source = KernelSpec::SourceCode{"void kernel_main() {}"},
        .num_threads = 4,
        .compiler_options =
            KernelSpec::CompilerOptions{
                .defines = {{"MY_DEFINE", "42"}},
                .opt_level = tt::tt_metal::KernelBuildOptLevel::O3,
            },
        .hw_config =
            ComputeHardwareConfig{
                ComputeGen2Config{
                    .fpu_math_fidelity = MathFidelity::LoFi,
                    .enable_32_bit_dest = true,
                },
            },
    };

    EXPECT_EQ(compute_kernel.unique_id.get(), "my_compute_kernel");
    EXPECT_TRUE(compute_kernel.is_compute_kernel());
}

TEST(AggregateSpecTypes, DataflowBufferSpecDesignatedInitializers) {
    // Demonstrates constructing DataflowBufferSpec with designated initializers
    DataflowBufferSpec dfb{
        .unique_id = DFBSpecName{"my_dfb"},
        .entry_size = 2048,
        .num_entries = 4,
        .data_format_metadata = tt::DataFormat::Float16_b,
    };

    EXPECT_EQ(dfb.unique_id.get(), "my_dfb");
    EXPECT_EQ(dfb.entry_size, 2048u);
    EXPECT_EQ(dfb.num_entries, 4u);

    // DFB with advanced options
    DataflowBufferSpec borrowed_dfb{
        .unique_id = DFBSpecName{"borrowed_dfb"},
        .entry_size = 1024,
        .num_entries = 8,
        .borrowed_from = TensorParamName{"input_tensor"},
    };

    EXPECT_EQ(borrowed_dfb.borrowed_from, std::optional<TensorParamName>{TensorParamName{"input_tensor"}});
}

TEST(AggregateSpecTypes, WorkUnitSpecDesignatedInitializers) {
    // Demonstrates constructing WorkUnitSpec with designated initializers
    WorkUnitSpec work_unit{
        .name = "my_work_unit",
        .kernels = {KernelSpecName{"kernel1"}, KernelSpecName{"kernel2"}},
        .target_nodes = NodeCoord{0, 0},
    };

    EXPECT_EQ(work_unit.name, "my_work_unit");
    EXPECT_EQ(work_unit.kernels.size(), 2u);
}

TEST(AggregateSpecTypes, RuntimeArgSchemaDesignatedInitializers) {
    // Named RTAs + CRTAs via designated initializers; vararg counts now live on
    // KernelAdvancedOptions (see VarargCountsOnAdvancedOptions below).
    KernelSpec::RuntimeArgSchema schema{
        .runtime_arg_names = {"input_ptr", "output_ptr"},
        .common_runtime_arg_names = {"tile_count"},
    };

    EXPECT_EQ(schema.runtime_arg_names.size(), 2u);
    EXPECT_EQ(schema.common_runtime_arg_names.size(), 1u);
}

TEST(AggregateSpecTypes, VarargCountsOnAdvancedOptions) {
    // Scalar vararg counts via designated initializers on KernelAdvancedOptions.
    KernelAdvancedOptions adv{
        .num_runtime_varargs = 4,
        .num_common_runtime_varargs = 2,
    };

    EXPECT_EQ(adv.num_runtime_varargs, 4u);
    EXPECT_EQ(adv.num_common_runtime_varargs, 2u);
    EXPECT_TRUE(adv.num_runtime_varargs_per_node.empty());
}

TEST(AggregateSpecTypes, VarargPerNodeOverrideOnAdvancedOptions) {
    // Per-node override path (advanced): ensure designated-init works.
    using NumVarargsPerNode = Table<Nodes, uint32_t>;
    KernelAdvancedOptions adv{
        .num_runtime_varargs_per_node = NumVarargsPerNode{{NodeCoord{0, 0}, 4}, {NodeCoord{1, 0}, 7}},
    };

    EXPECT_EQ(adv.num_runtime_varargs_per_node.size(), 2u);
    EXPECT_EQ(adv.num_runtime_varargs, 0u);  // scalar left at default in this example
}

TEST(AggregateSpecTypes, KernelSpecNamedRuntimeArgsDesignatedInitializers) {
    KernelSpec k{
        .unique_id = KernelSpecName{"k"},
        .source = KernelSpec::SourceCode{"void kernel_main() {}"},
        .runtime_arg_schema =
            KernelSpec::RuntimeArgSchema{
                .runtime_arg_names = {"input_ptr"},
            },
        .hw_config = DataMovementGen2Config{},
    };
    EXPECT_EQ(k.runtime_arg_schema.runtime_arg_names.size(), 1u);
}

TEST(AggregateSpecTypes, SemaphoreSpecDesignatedInitializers) {
    // Demonstrates constructing SemaphoreSpec with designated initializers
    SemaphoreSpec sem{
        .unique_id = SemaphoreSpecName{"my_semaphore"},
        .target_nodes = NodeCoord{0, 0},
        .advanced_options = SemaphoreAdvancedOptions{.initial_value = 7},
    };

    EXPECT_EQ(sem.unique_id.get(), "my_semaphore");
    EXPECT_EQ(sem.advanced_options.initial_value, 7u);
}

TEST(AggregateSpecTypes, ProgramSpecDesignatedInitializers) {
    // Demonstrates constructing a complete ProgramSpec with designated initializers
    ProgramSpec spec{
        .name = "my_program",
        .kernels =
            {
                KernelSpec{
                    .unique_id = KernelSpecName{"producer"},
                    .source = KernelSpec::SourceCode{"void kernel_main() {}"},
                    .dfb_bindings =
                        {
                            DFBBinding{
                                .dfb_spec_name = DFBSpecName{"dfb"},
                                .accessor_name = "out",
                                .endpoint_type = DFBEndpointType::PRODUCER,
                                .access_pattern = DFBAccessPattern::STRIDED,
                            },
                        },
                    .hw_config = DataMovementGen2Config{},
                },
                KernelSpec{
                    .unique_id = KernelSpecName{"consumer"},
                    .source = KernelSpec::SourceCode{"void kernel_main() {}"},
                    .dfb_bindings =
                        {
                            DFBBinding{
                                .dfb_spec_name = DFBSpecName{"dfb"},
                                .accessor_name = "in",
                                .endpoint_type = DFBEndpointType::CONSUMER,
                                .access_pattern = DFBAccessPattern::STRIDED,
                            },
                        },
                    .hw_config = ComputeHardwareConfig{},
                },
            },
        .dataflow_buffers =
            {
                DataflowBufferSpec{
                    .unique_id = DFBSpecName{"dfb"},
                    .entry_size = 1024,
                    .num_entries = 2,
                    .data_format_metadata = tt::DataFormat::Float16_b,
                },
            },
        .work_units =
            {
                WorkUnitSpec{
                    .name = "work_unit",
                    .kernels = {KernelSpecName{"producer"}, KernelSpecName{"consumer"}},
                    .target_nodes = NodeCoord{0, 0},
                },
            },
    };

    EXPECT_EQ(spec.name, "my_program");
    EXPECT_EQ(spec.kernels.size(), 2u);
    EXPECT_EQ(spec.dataflow_buffers.size(), 1u);
    EXPECT_EQ(spec.work_units.size(), 1u);
}

TEST(AggregateSpecTypes, NestedStructsDesignatedInitializers) {
    // Demonstrates constructing nested configuration structs with designated initializers
    DFBBinding binding{
        .dfb_spec_name = DFBSpecName{"my_dfb"},
        .accessor_name = "accessor",
        .endpoint_type = DFBEndpointType::PRODUCER,
        .access_pattern = DFBAccessPattern::ALL,
    };
    EXPECT_EQ(binding.dfb_spec_name.get(), "my_dfb");

    SemaphoreBinding sem_binding{
        .semaphore_spec_name = SemaphoreSpecName{"my_sem"},
        .accessor_name = "sem_accessor",
    };
    EXPECT_EQ(sem_binding.semaphore_spec_name.get(), "my_sem");

    KernelSpec::CompilerOptions opts{
        .include_paths = {"/path/to/include"},
        .defines = {{"DEBUG", "1"}, {"VERSION", "2"}},
        .opt_level = tt::tt_metal::KernelBuildOptLevel::O0,
    };
    EXPECT_EQ(opts.defines.size(), 2u);

    DataMovementGen1Config gen1{
        .processor = tt::tt_metal::DataMovementProcessor::RISCV_1,
        .noc = tt::tt_metal::NOC::RISCV_1_default,
        .noc_mode = tt::tt_metal::NOC_MODE::DM_DEDICATED_NOC,
    };
    EXPECT_EQ(gen1.processor, tt::tt_metal::DataMovementProcessor::RISCV_1);

    CrossNodeDataflowBufferSpec remote_dfb{
        .dfb_spec =
            DataflowBufferSpec{
                .unique_id = DFBSpecName{"remote_dfb"},
                .entry_size = 1024,
                .num_entries = 2,
            },
        .producer_consumer_map = {{NodeCoord{0, 0}, NodeCoord{1, 0}}},
    };
    EXPECT_EQ(remote_dfb.producer_consumer_map.size(), 1u);
    EXPECT_EQ(remote_dfb.dfb_spec.unique_id.get(), "remote_dfb");
}

// ============================================================================
// Gen1 (WH/BH) Tests
// ============================================================================

// Test fixture for ProgramSpec on Wormhole - uses WORMHOLE_B0 mock device
class ProgramSpecTestGen1 : public ::testing::Test {
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

// Gen1 counterpart of the compute-config translation-stability tests (the Gen2 pair lives in the
// Quasar suite): a default ComputeGen1Config{} must yield the historical internal ComputeConfig
// defaults. Guards the perf/precision knobs that don't move a functional pass/fail result.
TEST_F(ProgramSpecTestGen1, ComputeGen1ConfigDefaultsMapToInternalDefaults) {
    ProgramSpec spec = MakeMinimalGen1ValidProgramSpec();  // compute_kernel carries a default ComputeGen1Config
    Program program = MakeProgramFromSpec(*mesh_device_, spec);

    const auto built_variant = program.impl().get_kernel_by_spec_name("compute_kernel")->config();
    const auto& built = std::get<ComputeConfig>(built_variant);
    EXPECT_EQ(built.math_fidelity, MathFidelity::HiFi4);
    EXPECT_FALSE(built.fp32_dest_acc_en);
    EXPECT_FALSE(built.dst_full_sync_en);   // double_buffer_dest defaults true -> !true
    EXPECT_FALSE(built.bfp8_pack_precise);  // bfp_pack_precision_mode defaults Approximate
    EXPECT_FALSE(built.math_approx_mode);   // sfpu_precision_mode defaults Precise
}

TEST_F(ProgramSpecTestGen1, MinimalValidProgramSpecSucceeds) {
    ProgramSpec spec = MakeMinimalGen1ValidProgramSpec();
    EXPECT_NO_THROW(MakeProgramFromSpec(*mesh_device_, spec));
}

TEST_F(ProgramSpecTestGen1, ConsumerUnpackToDestBelow32BitWithoutEnableFailsForPerf) {
    // On Gen1, UnpackToDest on a consumed <=16-bit DFB without enable_32_bit_dest bypasses the
    // SrcA/B path for no precision benefit — rejected as bad-for-perf. (On Gen2 the identical spec
    // is accepted — see ProgramSpecTestQuasar.ConsumerUnpackToDestBelow32BitWithoutEnableSucceeds.)
    ProgramSpec spec = MakeMinimalGen1ValidProgramSpec();  // dfb_0 is Float16_b, consumed by compute_kernel
    for (auto& kernel : spec.kernels) {
        if (kernel.is_compute_kernel()) {
            auto& config = std::get<ComputeGen1Config>(std::get<ComputeHardwareConfig>(kernel.hw_config));
            // enable_32_bit_dest stays at its default (false).
            config.unpack_modes = {{DFBSpecName{"dfb_0"}, UnpackMode::UnpackToDest}};
        }
    }
    EXPECT_THAT(
        [&] { MakeProgramFromSpec(*mesh_device_, spec); },
        ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr("leads to worse performance")));
}

TEST_F(ProgramSpecTestGen1, DMOnlyProgramSucceeds) {
    // Two DM kernels on different processors (RISCV_0 producer, RISCV_1 consumer)
    NodeCoord node{0, 0};

    ProgramSpec spec;
    spec.name = "dm_only_program";

    auto producer = MakeMinimalWriterDMKernel("producer");
    auto consumer = MakeMinimalReaderDMKernel("consumer");
    auto dfb = MakeMinimalDFB("dfb");

    producer.dfb_bindings.push_back(ProducerOf(DFBSpecName{"dfb"}, "out"));
    consumer.dfb_bindings.push_back(ConsumerOf(DFBSpecName{"dfb"}, "in"));

    spec.kernels = {producer, consumer};
    spec.dataflow_buffers = {dfb};
    spec.work_units = std::vector<WorkUnitSpec>{MakeMinimalWorkUnit("work_unit", node, {"producer", "consumer"})};

    EXPECT_NO_THROW(MakeProgramFromSpec(*mesh_device_, spec));
}

TEST_F(ProgramSpecTestGen1, DMKernelSelfLoopOnGen1Succeeds) {
    // On Gen1 (WH/BH) a DFB lowers to a plain circular buffer, so a single DM kernel may bind it as
    // both PRODUCER and CONSUMER (self-loop) — the classic scratch pattern of one DM engine filling
    // and draining an L1 FIFO. There is no tile-counter credit machinery requiring disjoint
    // producer/consumer masks, so the spec validator accepts it. (On Gen2 the same spec is rejected —
    // see DMKernelSelfLoopOnGen2Fails.)
    NodeCoord node{0, 0};

    ProgramSpec spec;
    spec.name = "dm_self_loop";

    auto kernel = MakeMinimalGen1DMKernel("kernel");
    auto dfb = MakeMinimalDFB("dfb");
    kernel.dfb_bindings.push_back(ProducerOf(DFBSpecName{"dfb"}, "p"));
    kernel.dfb_bindings.push_back(ConsumerOf(DFBSpecName{"dfb"}, "c"));

    spec.kernels = {kernel};
    spec.dataflow_buffers = {dfb};
    spec.work_units = std::vector<WorkUnitSpec>{MakeMinimalWorkUnit("work_unit", node, {"kernel"})};

    EXPECT_NO_THROW(MakeProgramFromSpec(*mesh_device_, spec));
}

TEST_F(ProgramSpecTestGen1, TwoDMKernelsDifferentProcessorsSucceeds) {
    // RISCV_0 and RISCV_1 on the same node — should succeed. (MakeMinimalGen1DMKernel gives them
    // distinct NOCs, so they also satisfy the dedicated-NOC distinctness rule.)
    NodeCoord node{0, 0};

    ProgramSpec spec;
    spec.name = "two_dm_program";

    auto k0 = MakeMinimalGen1DMKernel("k0", DataMovementProcessor::RISCV_0);
    auto k1 = MakeMinimalGen1DMKernel("k1", DataMovementProcessor::RISCV_1);

    spec.kernels = {k0, k1};
    spec.work_units = std::vector<WorkUnitSpec>{MakeMinimalWorkUnit("work_unit", node, {"k0", "k1"})};

    EXPECT_NO_THROW(MakeProgramFromSpec(*mesh_device_, spec));
}

TEST_F(ProgramSpecTestGen1, MultiThreadedDMKernelFails) {
    NodeCoord node{0, 0};

    ProgramSpec spec;
    spec.name = "test_program";

    auto kernel = MakeMinimalGen1DMKernel("dm_kernel");
    kernel.num_threads = 2;

    spec.kernels = {kernel};
    spec.work_units = std::vector<WorkUnitSpec>{MakeMinimalWorkUnit("work_unit", node, {"dm_kernel"})};

    EXPECT_THAT(
        [&] { MakeProgramFromSpec(*mesh_device_, spec); },
        ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr("does not support multi-threaded kernels")));
}

TEST_F(ProgramSpecTestGen1, MultiThreadedComputeKernelFails) {
    NodeCoord node{0, 0};

    ProgramSpec spec;
    spec.name = "test_program";

    auto kernel = MakeMinimalGen1ComputeKernel("compute_kernel");
    kernel.num_threads = 2;

    spec.kernels = {kernel};
    spec.work_units = std::vector<WorkUnitSpec>{MakeMinimalWorkUnit("work_unit", node, {"compute_kernel"})};

    EXPECT_THAT(
        [&] { MakeProgramFromSpec(*mesh_device_, spec); },
        ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr("does not support multi-threaded kernels")));
}

TEST_F(ProgramSpecTestGen1, DMKernelWithGen2ConfigFails) {
    // The config's generation must match the target platform: on Gen1 (WH/BH) a DM kernel carrying a
    // Gen2 config is a hard error — it has no way to resolve its processor/NOC placement.
    NodeCoord node{0, 0};

    ProgramSpec spec;
    spec.name = "test_program";

    // MakeMinimalGen2DMKernel produces a gen2 (Quasar) DM config (no Gen1Config).
    auto kernel = MakeMinimalGen2DMKernel("dm_kernel");

    spec.kernels = {kernel};
    spec.work_units = std::vector<WorkUnitSpec>{MakeMinimalWorkUnit("work_unit", node, {"dm_kernel"})};

    EXPECT_THAT(
        [&] { MakeProgramFromSpec(*mesh_device_, spec); },
        ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr("holds a DataMovementGen2Config")));
}

TEST_F(ProgramSpecTestGen1, ProcessorConflictFails) {
    // Two DM kernels both targeting RISCV_0 on the same node
    NodeCoord node{0, 0};

    ProgramSpec spec;
    spec.name = "test_program";

    auto k0 = MakeMinimalGen1DMKernel("k0", DataMovementProcessor::RISCV_0);
    auto k1 = MakeMinimalGen1DMKernel("k1", DataMovementProcessor::RISCV_0);  // conflict

    spec.kernels = {k0, k1};
    spec.work_units = std::vector<WorkUnitSpec>{MakeMinimalWorkUnit("work_unit", node, {"k0", "k1"})};

    EXPECT_THAT(
        [&] { MakeProgramFromSpec(*mesh_device_, spec); },
        ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr("both claim the same DM processor")));
}

TEST_F(ProgramSpecTestGen1, TwoDMKernelsSameNocDedicatedFails) {
    // Two DM kernels on distinct processors (RISCV_0, RISCV_1) but pinned to the SAME NOC in
    // dedicated mode. Each kernel's NoC traffic is statically compiled to its config.noc, so both
    // would drive NOC_0 and hang the device. Validation must reject this.
    NodeCoord node{0, 0};

    ProgramSpec spec;
    spec.name = "test_program";

    auto k0 = MakeMinimalGen1DMKernel("k0", DataMovementProcessor::RISCV_0);
    auto k1 = MakeMinimalGen1DMKernel("k1", DataMovementProcessor::RISCV_1);
    // Force both onto NOC_0 (the helper would otherwise assign complementary NOCs). noc_mode
    // defaults to DM_DEDICATED_NOC.
    std::get<DataMovementGen1Config>(std::get<DataMovementHardwareConfig>(k0.hw_config)).noc = NOC::NOC_0;
    std::get<DataMovementGen1Config>(std::get<DataMovementHardwareConfig>(k1.hw_config)).noc = NOC::NOC_0;

    spec.kernels = {k0, k1};
    spec.work_units = std::vector<WorkUnitSpec>{MakeMinimalWorkUnit("work_unit", node, {"k0", "k1"})};

    EXPECT_THAT(
        [&] { MakeProgramFromSpec(*mesh_device_, spec); },
        ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr("pinned to NOC_0")));
}

TEST_F(ProgramSpecTestGen1, TwoDMKernelsDistinctNocDedicatedSucceeds) {
    // Two dedicated-NOC DM kernels on distinct processors AND distinct NOCs — the correct pairing.
    NodeCoord node{0, 0};

    ProgramSpec spec;
    spec.name = "test_program";

    auto k0 = MakeMinimalGen1DMKernel("k0", DataMovementProcessor::RISCV_0);
    auto k1 = MakeMinimalGen1DMKernel("k1", DataMovementProcessor::RISCV_1);
    std::get<DataMovementGen1Config>(std::get<DataMovementHardwareConfig>(k0.hw_config)).noc = NOC::NOC_0;
    std::get<DataMovementGen1Config>(std::get<DataMovementHardwareConfig>(k1.hw_config)).noc = NOC::NOC_1;

    spec.kernels = {k0, k1};
    spec.work_units = std::vector<WorkUnitSpec>{MakeMinimalWorkUnit("work_unit", node, {"k0", "k1"})};

    EXPECT_NO_THROW(MakeProgramFromSpec(*mesh_device_, spec));
}

TEST_F(ProgramSpecTestGen1, TwoDMKernelsSameNocDynamicSucceeds) {
    // In DM_DYNAMIC_NOC mode, two DM kernels may intentionally share a NOC (it frees the other NOC
    // for fabric). The NOC-distinctness rule is dedicated-mode only, so this is accepted even though
    // both kernels name NOC_0.
    NodeCoord node{0, 0};

    ProgramSpec spec;
    spec.name = "test_program";

    auto k0 = MakeMinimalGen1DMKernel("k0", DataMovementProcessor::RISCV_0);
    auto k1 = MakeMinimalGen1DMKernel("k1", DataMovementProcessor::RISCV_1);
    auto& cfg0 = std::get<DataMovementGen1Config>(std::get<DataMovementHardwareConfig>(k0.hw_config));
    cfg0.noc = NOC::NOC_0;
    cfg0.noc_mode = NOC_MODE::DM_DYNAMIC_NOC;
    auto& cfg1 = std::get<DataMovementGen1Config>(std::get<DataMovementHardwareConfig>(k1.hw_config));
    cfg1.noc = NOC::NOC_0;
    cfg1.noc_mode = NOC_MODE::DM_DYNAMIC_NOC;

    spec.kernels = {k0, k1};
    spec.work_units = std::vector<WorkUnitSpec>{MakeMinimalWorkUnit("work_unit", node, {"k0", "k1"})};

    EXPECT_NO_THROW(MakeProgramFromSpec(*mesh_device_, spec));
}

TEST_F(ProgramSpecTestGen1, DMProcessorBeyondRiscv1Fails) {
    // Gen1 has only RISCV_0 (BRISC) and RISCV_1 (NCRISC); RISCV_2..7 are Gen2/Quasar-only. A Gen1 DM
    // kernel requesting one must be rejected (parity with the legacy CreateDataMovementKernel guard).
    NodeCoord node{0, 0};

    ProgramSpec spec;
    spec.name = "test_program";

    auto kernel = MakeMinimalGen1DMKernel("dm_kernel", DataMovementProcessor::RISCV_0);
    std::get<DataMovementGen1Config>(std::get<DataMovementHardwareConfig>(kernel.hw_config)).processor =
        DataMovementProcessor::RISCV_2;

    spec.kernels = {kernel};
    spec.work_units = std::vector<WorkUnitSpec>{MakeMinimalWorkUnit("work_unit", node, {"dm_kernel"})};

    EXPECT_THAT(
        [&] { MakeProgramFromSpec(*mesh_device_, spec); },
        ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr("Gen1 has only")));
}

TEST_F(ProgramSpecTestGen1, TwoDMKernelsMixedNocModeFails) {
    // NOC mode configures shared per-core NOC hardware (and is compiled into each kernel binary), so
    // two DM kernels on the same node must agree on it. One dedicated + one dynamic is incoherent.
    NodeCoord node{0, 0};

    ProgramSpec spec;
    spec.name = "test_program";

    // Distinct processors and distinct NOCs (helper defaults: RISCV_0->NOC_0, RISCV_1->NOC_1), so
    // neither the processor nor the NOC-distinctness check fires — only the mode disagreement trips.
    auto k0 = MakeMinimalGen1DMKernel("k0", DataMovementProcessor::RISCV_0);
    auto k1 = MakeMinimalGen1DMKernel("k1", DataMovementProcessor::RISCV_1);
    std::get<DataMovementGen1Config>(std::get<DataMovementHardwareConfig>(k0.hw_config)).noc_mode =
        NOC_MODE::DM_DEDICATED_NOC;
    std::get<DataMovementGen1Config>(std::get<DataMovementHardwareConfig>(k1.hw_config)).noc_mode =
        NOC_MODE::DM_DYNAMIC_NOC;

    spec.kernels = {k0, k1};
    spec.work_units = std::vector<WorkUnitSpec>{MakeMinimalWorkUnit("work_unit", node, {"k0", "k1"})};

    EXPECT_THAT(
        [&] { MakeProgramFromSpec(*mesh_device_, spec); },
        ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr("different NOC modes")));
}

TEST_F(ProgramSpecTestGen1, DMKernelsSameProcessorAndNocOnDistinctNodesSucceeds) {
    // Node-scoping guard: two DM kernels with identical processor (RISCV_0), NOC (NOC_0), and
    // DM_DEDICATED_NOC mode are legal when placed on DISTINCT nodes — the processor- and
    // NOC-distinctness censuses are per-node. This would wrongly fail if either map were keyed
    // without the node component.
    NodeCoord node_a{0, 0};
    NodeCoord node_b{1, 0};

    ProgramSpec spec;
    spec.name = "test_program";

    auto k_a = MakeMinimalGen1DMKernel("k_a", DataMovementProcessor::RISCV_0);  // NOC_0, dedicated
    auto k_b = MakeMinimalGen1DMKernel("k_b", DataMovementProcessor::RISCV_0);  // NOC_0, dedicated

    spec.kernels = {k_a, k_b};
    spec.work_units = std::vector<WorkUnitSpec>{
        MakeMinimalWorkUnit("wu_a", node_a, {"k_a"}),
        MakeMinimalWorkUnit("wu_b", node_b, {"k_b"}),
    };

    EXPECT_NO_THROW(MakeProgramFromSpec(*mesh_device_, spec));
}

TEST_F(ProgramSpecTestGen1, DMKernelsDifferentNocModesOnDistinctNodesSucceeds) {
    // Node-scoping guard for NOC-mode agreement: a DM_DEDICATED_NOC kernel on one node and a
    // DM_DYNAMIC_NOC kernel on another are legal — agreement is enforced per-node, not globally.
    // This would wrongly fail if node_noc_mode were keyed without the node component.
    NodeCoord node_a{0, 0};
    NodeCoord node_b{1, 0};

    ProgramSpec spec;
    spec.name = "test_program";

    auto k_a = MakeMinimalGen1DMKernel("k_a", DataMovementProcessor::RISCV_0);
    auto k_b = MakeMinimalGen1DMKernel("k_b", DataMovementProcessor::RISCV_0);
    std::get<DataMovementGen1Config>(std::get<DataMovementHardwareConfig>(k_a.hw_config)).noc_mode =
        NOC_MODE::DM_DEDICATED_NOC;
    std::get<DataMovementGen1Config>(std::get<DataMovementHardwareConfig>(k_b.hw_config)).noc_mode =
        NOC_MODE::DM_DEDICATED_NOC;

    spec.kernels = {k_a, k_b};
    spec.work_units = std::vector<WorkUnitSpec>{
        MakeMinimalWorkUnit("wu_a", node_a, {"k_a"}),
        MakeMinimalWorkUnit("wu_b", node_b, {"k_b"}),
    };

    EXPECT_NO_THROW(MakeProgramFromSpec(*mesh_device_, spec));
}

TEST_F(ProgramSpecTestGen1, ReaderAndWriterRolesOnSameNodeSucceed) {
    // A READER and a WRITER role resolve to distinct processors (RISCV_1 and RISCV_0
    // respectively), so two role-driven DM kernels coexist on one node without conflict.
    NodeCoord node{0, 0};

    ProgramSpec spec;
    spec.name = "test_program";

    auto reader = MakeMinimalReaderDMKernel("reader");
    auto writer = MakeMinimalWriterDMKernel("writer");

    spec.kernels = {reader, writer};
    spec.work_units = std::vector<WorkUnitSpec>{MakeMinimalWorkUnit("work_unit", node, {"reader", "writer"})};

    EXPECT_NO_THROW({ MakeProgramFromSpec(*mesh_device_, spec); });
}

TEST_F(ProgramSpecTestGen1, TwoReaderRolesOnSameNodeConflict) {
    // Both READER kernels resolve to the same processor (RISCV_1), so placing them on the
    // same node is a conflict — confirming the role hint resolves to a fixed, deterministic
    // processor (the same uniqueness rule as explicit configs).
    NodeCoord node{0, 0};

    ProgramSpec spec;
    spec.name = "test_program";

    auto r0 = MakeMinimalReaderDMKernel("r0");
    auto r1 = MakeMinimalReaderDMKernel("r1");

    spec.kernels = {r0, r1};
    spec.work_units = std::vector<WorkUnitSpec>{MakeMinimalWorkUnit("work_unit", node, {"r0", "r1"})};

    EXPECT_THAT(
        [&] { MakeProgramFromSpec(*mesh_device_, spec); },
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
    spec.name = "test_program";

    auto kernel = MakeMinimalGen1DMKernel("dm_kernel");
    spec.kernels = {kernel};
    spec.work_units = std::vector<WorkUnitSpec>{MakeMinimalWorkUnit("work_unit", oob_node, {"dm_kernel"})};

    EXPECT_THAT(
        [&] { MakeProgramFromSpec(*mesh_device_, spec); },
        ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr("out of bounds")));
}

TEST_F(ProgramSpecTestGen1, KernelTargetsOutOfBoundsNodeFails) {
    // x=8 is just outside the 8-column grid (same in fast and slow dispatch).
    const NodeCoord oob_node{8, 0};

    ProgramSpec spec;
    spec.name = "test_program";

    auto kernel = MakeMinimalGen1DMKernel("dm_kernel");
    spec.kernels = {kernel};
    spec.work_units = std::vector<WorkUnitSpec>{MakeMinimalWorkUnit("work_unit", oob_node, {"dm_kernel"})};

    EXPECT_THAT(
        [&] { MakeProgramFromSpec(*mesh_device_, spec); },
        ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr("out of bounds")));
}

TEST_F(ProgramSpecTestGen1, SemaphoreBoundToComputeKernelFailsOnGen1) {
    // Compute kernels cannot have semaphore bindings on Gen 1
    ProgramSpec spec = MakeMinimalGen1ValidProgramSpec();

    SemaphoreSpec sem;
    sem.unique_id = SemaphoreSpecName{"sem_0"};
    sem.target_nodes = NodeCoord{0, 0};
    spec.semaphores = {sem};

    // kernels[1] is the compute kernel in MakeMinimalGen1ValidProgramSpec
    ASSERT_TRUE(spec.kernels[1].is_compute_kernel());
    spec.kernels[1].semaphore_bindings = {
        SemaphoreBinding{.semaphore_spec_name = SemaphoreSpecName{"sem_0"}, .accessor_name = "done_flag"}};

    EXPECT_THAT(
        [&] { MakeProgramFromSpec(*mesh_device_, spec); },
        ::testing::ThrowsMessage<std::runtime_error>(
            ::testing::HasSubstr("Semaphore bindings are not supported for compute kernels.")));
}

TEST_F(ProgramSpecTestGen1, SemaphoreBoundToDMKernelSucceedsOnGen1) {
    // Sanity check: binding a semaphore to a DM kernel on WH/BH is allowed.
    ProgramSpec spec = MakeMinimalGen1ValidProgramSpec();

    SemaphoreSpec sem;
    sem.unique_id = SemaphoreSpecName{"sem_0"};
    sem.target_nodes = NodeCoord{0, 0};
    spec.semaphores = {sem};

    // kernels[0] is the DM kernel in MakeMinimalGen1ValidProgramSpec
    ASSERT_TRUE(spec.kernels[0].is_data_movement_kernel());
    spec.kernels[0].semaphore_bindings = {
        SemaphoreBinding{.semaphore_spec_name = SemaphoreSpecName{"sem_0"}, .accessor_name = "done_flag"}};

    EXPECT_NO_THROW(MakeProgramFromSpec(*mesh_device_, spec));
}

TEST_F(ProgramSpecTestGen1, SemaphoresWithNonZeroInitialValueSucceedOnGen1) {
    // Gen1 accepts non-zero initial values (only Quasar rejects them).
    ProgramSpec spec = MakeMinimalGen1ValidProgramSpec();

    SemaphoreSpec sem;
    sem.unique_id = SemaphoreSpecName{"sem_0"};
    sem.target_nodes = NodeCoord{0, 0};
    sem.advanced_options = SemaphoreAdvancedOptions{.initial_value = 3};
    spec.semaphores = {sem};

    spec.kernels[0].semaphore_bindings = {
        SemaphoreBinding{.semaphore_spec_name = SemaphoreSpecName{"sem_0"}, .accessor_name = "done_flag"}};

    EXPECT_NO_THROW(MakeProgramFromSpec(*mesh_device_, spec));
}

TEST_F(ProgramSpecTestGen1, DuplicateKernelNameFails) {
    // Structural validation (CollectSpecData) must catch this on gen1 too
    NodeCoord node{0, 0};

    ProgramSpec spec;
    spec.name = "test_program";

    auto k0 = MakeMinimalGen1DMKernel("dm_kernel", DataMovementProcessor::RISCV_0);
    auto k1 = MakeMinimalGen1DMKernel("dm_kernel", DataMovementProcessor::RISCV_1);  // duplicate name

    spec.kernels = {k0, k1};
    spec.work_units = std::vector<WorkUnitSpec>{MakeMinimalWorkUnit("work_unit", node, {"dm_kernel"})};

    EXPECT_THAT(
        [&] { MakeProgramFromSpec(*mesh_device_, spec); },
        ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr("Duplicate KernelSpec name")));
}

// ============================================================================
// TensorParameter Validation Tests (Gen1 / WH)
// ============================================================================
// Spec-level validation for the Metal 2.0 TensorAccessor binding feature. These tests exercise
// CollectSpecData paths only — they do not need a MeshTensor at enqueue (those run-params paths
// are covered in test_program_run_args.cpp). Hardware-agnostic; using the Gen1 fixture for
// lower-likelihood-of-unrelated-mock-issues per Audrey's guidance.

// (BindTensorParameterToKernel and MakeMinimalTensorParameter using-declarations hoisted to top-of-file.)

TEST_F(ProgramSpecTestGen1, DuplicateTensorParameterNameFails) {
    ProgramSpec spec = MakeMinimalGen1ValidProgramSpec();

    auto binding_a = MakeMinimalTensorParameter("input_tensor");
    auto binding_b = MakeMinimalTensorParameter("input_tensor");  // duplicate!
    spec.tensor_parameters = {binding_a, binding_b};

    // Bind one of them to a kernel so the "every binding must be bound" check is satisfied;
    // the duplicate-name check fires first regardless.
    BindTensorParameterToKernel(spec.kernels[0], "input_tensor", "input_ta");

    EXPECT_THAT(
        [&] { MakeProgramFromSpec(*mesh_device_, spec); },
        ::testing::ThrowsMessage<std::runtime_error>(
            ::testing::HasSubstr("Duplicate TensorParameter name 'input_tensor'")));
}

TEST_F(ProgramSpecTestGen1, KernelReferencesUnknownTensorParameterFails) {
    ProgramSpec spec = MakeMinimalGen1ValidProgramSpec();

    // Reference a TensorParameter that doesn't exist in the program.
    BindTensorParameterToKernel(spec.kernels[0], "nonexistent_tensor", "input_ta");

    EXPECT_THAT(
        [&] { MakeProgramFromSpec(*mesh_device_, spec); },
        ::testing::ThrowsMessage<std::runtime_error>(
            ::testing::HasSubstr("references unknown TensorParameter 'nonexistent_tensor'")));
}

TEST_F(ProgramSpecTestGen1, UnboundTensorParameterFails) {
    ProgramSpec spec = MakeMinimalGen1ValidProgramSpec();

    // Declare a TensorParameter but don't bind it to any kernel.
    spec.tensor_parameters = {MakeMinimalTensorParameter("orphan_tensor")};

    EXPECT_THAT(
        [&] { MakeProgramFromSpec(*mesh_device_, spec); },
        ::testing::ThrowsMessage<std::runtime_error>(
            ::testing::HasSubstr("TensorParameter 'orphan_tensor' is defined but not bound by any kernel")));
}

TEST_F(ProgramSpecTestGen1, DuplicateTensorAccessorNameWithinKernelFails) {
    ProgramSpec spec = MakeMinimalGen1ValidProgramSpec();

    spec.tensor_parameters = {
        MakeMinimalTensorParameter("input_tensor"),
        MakeMinimalTensorParameter("output_tensor"),
    };
    // Two bindings on the same kernel under the same accessor_name — illegal.
    BindTensorParameterToKernel(spec.kernels[0], "input_tensor", "same_accessor");
    BindTensorParameterToKernel(spec.kernels[0], "output_tensor", "same_accessor");

    EXPECT_THAT(
        [&] { MakeProgramFromSpec(*mesh_device_, spec); },
        ::testing::ThrowsMessage<std::runtime_error>(
            ::testing::HasSubstr("has duplicate tensor accessor_name 'same_accessor'")));
}

TEST_F(ProgramSpecTestGen1, InvalidTensorAccessorNameFails) {
    // Smoke-tests the IsValidCppIdentifier check on tensor accessor names. The check is the
    // same one DFB / Semaphore use; one bad name here is sufficient (full coverage lives in
    // the DFB version of this test).
    ProgramSpec spec = MakeMinimalGen1ValidProgramSpec();

    spec.tensor_parameters = {MakeMinimalTensorParameter("input_tensor")};
    BindTensorParameterToKernel(spec.kernels[0], "input_tensor", "has-dash");  // not a valid identifier

    EXPECT_THAT(
        [&] { MakeProgramFromSpec(*mesh_device_, spec); },
        ::testing::ThrowsMessage<std::runtime_error>(
            ::testing::HasSubstr("tensor accessor_name 'has-dash' must be a valid C++ identifier")));
}

TEST_F(ProgramSpecTestGen1, AccessorNamesAcrossCategoriesAreSeparateNamespaces) {
    // DFB / Semaphore / TensorAccessor accessor names live in separate namespaces (each gets
    // its own emitted namespace in kernel_bindings_generated.h: dfb::, sem::, tensor::). Reusing
    // the same identifier across categories within one kernel must be allowed.
    ProgramSpec spec = MakeMinimalGen1ValidProgramSpec();

    spec.tensor_parameters = {MakeMinimalTensorParameter("input_tensor")};
    // The minimal program already has a DFB binding under accessor_name "input_dfb". Add a
    // semaphore and a tensor accessor, both also named "input_dfb" — the same string at a
    // C++ level — which should pass because they're in different namespaces.
    SemaphoreSpec sem;
    sem.unique_id = SemaphoreSpecName{"sem_0"};
    sem.target_nodes = NodeCoord{0, 0};
    spec.semaphores = {sem};
    spec.kernels[0].semaphore_bindings = {
        SemaphoreBinding{.semaphore_spec_name = SemaphoreSpecName{"sem_0"}, .accessor_name = "input_dfb"}};
    BindTensorParameterToKernel(spec.kernels[0], "input_tensor", "input_dfb");

    EXPECT_NO_THROW(MakeProgramFromSpec(*mesh_device_, spec));
}

TEST_F(ProgramSpecTestGen1, MinimalValidProgramSpecWithTensorParameterSucceeds) {
    // Positive baseline: a program with one TensorParameter bound to one kernel constructs
    // successfully. Exercises CollectSpecData validation, the host-side resolution helper
    // (TensorSpec → CTA payload using mesh_device.allocator() + virtual_core_from_logical_core),
    // TensorBinding address slot assignment, kernel ctor plumbing, and ProgramImpl registration.
    ProgramSpec spec = MakeMinimalGen1ValidProgramSpec();

    spec.tensor_parameters = {MakeMinimalTensorParameter("input_tensor")};
    BindTensorParameterToKernel(spec.kernels[0], "input_tensor", "input_ta");

    EXPECT_NO_THROW(MakeProgramFromSpec(*mesh_device_, spec));
}

// ============================================================================
// TensorParameter JIT Smoke Tests (Gen1 / WH)
// ============================================================================
// Codegen-path smoke test for the Metal 2.0 TensorAccessor binding feature. Ends in
// CompileProgram, so the auto-generated kernel_bindings_generated.h (with its `tensor::` namespace)
// must be syntactically valid and compose correctly with the rest of the kernel build. Doesn't
// validate runtime behavior — catches regressions in codegen string-formatting, token type alias
// generation, and include-path resolution.
//
// DM-only by design: tensor_accessor.h pulls in NoC-using headers (dataflow_api_addrgen.h,
// pages_address_iterator.h with ASSERT) that don't compile on TRISC. There are no compute-kernel
// uses of TensorAccessor in the wild; the device-side library was built for DM kernels.
// Compute-kernel TensorAccessor bindings are unsupported in this PR; making them work would
// require restructuring tensor_accessor.h to isolate the constexpr-only parts.

TEST_F(ProgramSpecTestGen1, TensorAccessorBindingJITSmokeDMKernel) {
    // DM kernel constructs a TensorAccessor from a binding token + invokes a NoC-using method.
    // Exercises: tensor:: namespace token, type alias <name>_t, the token ctor and its deduction
    // guide, get_common_arg_val for the implicit base address.
    NodeCoord node{0, 0};

    ProgramSpec spec;
    spec.name = "ta_smoke_dm";

    auto dm_kernel = MakeMinimalGen1DMKernel("dm_kernel");
    dm_kernel.source = KernelSpec::SourceCode{R"(
void kernel_main() {
    TensorAccessor accessor(tensor::input_tensor);
    auto noc_addr = accessor.get_noc_addr(0);
    (void)noc_addr;
}
)"};

    spec.kernels = {dm_kernel};
    spec.tensor_parameters = {MakeMinimalTensorParameter("input_tensor")};
    BindTensorParameterToKernel(spec.kernels[0], "input_tensor", "input_tensor");
    spec.work_units = std::vector<WorkUnitSpec>{MakeMinimalWorkUnit("work_unit", node, {"dm_kernel"})};

    Program program = MakeProgramFromSpec(*mesh_device_, spec);
    IDevice* device = mesh_device_->get_devices()[0];
    EXPECT_NO_THROW(detail::CompileProgram(device, program));
}

// ----------------------------------------------------------------------------
// Scratchpad JIT-compile smoke tests (device-side accessor composes & compiles)
// ----------------------------------------------------------------------------
//
// Like the TensorAccessor smoke above, these JIT-compile a kernel that constructs a Scratchpad from
// its binding accessor (scratch::<name>) and reads the CRTA-injected base address — exercising the
// generated scratch:: namespace + ScratchpadAccessor object and the device-side Scratchpad ctor. Compile-only on
// the mock Wormhole device (Gen1: the Quasar TRISC firmware isn't built in this checkout, so a
// Quasar JIT-compile would fail at link).

TEST_F(ProgramSpecTestGen1, ScratchpadAccessorBindingJITSmokeDMKernel) {
    // DM kernel constructs a Scratchpad from its binding token and reads the CRTA-injected base address.
    NodeCoord node{0, 0};

    ProgramSpec spec;
    spec.name = "scratch_smoke_dm";

    auto dm_kernel = MakeMinimalGen1DMKernel("dm_kernel");
    dm_kernel.source = KernelSpec::SourceCode{R"(
void kernel_main() {
    Scratchpad<int32_t> pad(scratch::scratch);
    volatile uint32_t base = pad.get_base_address();
    (void)base;
}
)"};
    dm_kernel.scratchpad_bindings.push_back(KernelSpec::ScratchpadBinding{
        .scratchpad_spec_name = ScratchpadSpecName{"scratch"}, .accessor_name = "scratch"});

    spec.kernels = {dm_kernel};
    spec.scratchpads = {ScratchpadSpec{.unique_id = ScratchpadSpecName{"scratch"}, .size_per_node = 1024}};
    spec.work_units = std::vector<WorkUnitSpec>{MakeMinimalWorkUnit("work_unit", node, {"dm_kernel"})};

    Program program = MakeProgramFromSpec(*mesh_device_, spec);
    IDevice* device = mesh_device_->get_devices()[0];
    EXPECT_NO_THROW(detail::CompileProgram(device, program));
}

// Compute-kernel counterpart. A scratchpad binding works on a compute kernel: scratchpad.h only
// forward-declares get_common_arg_val (resolved by the kernel's own API header — api/compute/common.h
// on TRISC, api/dataflow/dataflow_api.h on DM) and is otherwise NOC-free, so the device-side
// Scratchpad<T> ctor composes on the compute (TRISC) build path as well as DM.
TEST_F(ProgramSpecTestGen1, ScratchpadAccessorBindingJITSmokeComputeKernel) {
    // MakeMinimalGen1ValidProgramSpec wires a DM producer (kernels[0]) into a compute consumer
    // (kernels[1]) through a DFB; bind the scratchpad to the compute kernel and have it construct a
    // Scratchpad from its binding token.
    ProgramSpec spec = MakeMinimalGen1ValidProgramSpec();

    ASSERT_TRUE(spec.kernels[1].is_compute_kernel());
    spec.kernels[1].source = KernelSpec::SourceCode{R"(
void kernel_main() {
    Scratchpad<int32_t> pad(scratch::scratch);
    volatile uint32_t base = pad.get_base_address();
    (void)base;
}
)"};

    spec.scratchpads = {ScratchpadSpec{.unique_id = ScratchpadSpecName{"scratch"}, .size_per_node = 1024}};
    spec.kernels[1].scratchpad_bindings.push_back(KernelSpec::ScratchpadBinding{
        .scratchpad_spec_name = ScratchpadSpecName{"scratch"}, .accessor_name = "scratch"});

    Program program = MakeProgramFromSpec(*mesh_device_, spec);
    IDevice* device = mesh_device_->get_devices()[0];
    EXPECT_NO_THROW(detail::CompileProgram(device, program));
}

// Compile-only: a range-based for loop over a Scratchpad must compile. Exercises begin()/end() and the
// CoreLocalMem<T> iterator ops the loop desugars to (operator++, operator!=, operator*). Compile-only on
// the mock Gen1 device — no run: reading the uninitialized region would be UB at runtime, but this test
// only JIT-compiles the kernel, so the loop just needs to be well-formed.
TEST_F(ProgramSpecTestGen1, ScratchpadRangeBasedForCompiles) {
    NodeCoord node{0, 0};

    ProgramSpec spec;
    spec.name = "scratch_range_for";

    auto dm_kernel = MakeMinimalGen1DMKernel("dm_kernel");
    dm_kernel.source = KernelSpec::SourceCode{R"(
void kernel_main() {
    Scratchpad<int32_t> pad(scratch::scratch);
    int32_t acc = 0;
    for (auto& elem : pad) {
        acc += elem;
    }
    volatile int32_t sink = acc;  // keep the loop live so the range-for is actually instantiated
    (void)sink;
}
)"};
    dm_kernel.scratchpad_bindings.push_back(KernelSpec::ScratchpadBinding{
        .scratchpad_spec_name = ScratchpadSpecName{"scratch"}, .accessor_name = "scratch"});

    spec.kernels = {dm_kernel};
    spec.scratchpads = {ScratchpadSpec{.unique_id = ScratchpadSpecName{"scratch"}, .size_per_node = 1024}};
    spec.work_units = std::vector<WorkUnitSpec>{MakeMinimalWorkUnit("work_unit", node, {"dm_kernel"})};

    Program program = MakeProgramFromSpec(*mesh_device_, spec);
    IDevice* device = mesh_device_->get_devices()[0];
    EXPECT_NO_THROW(detail::CompileProgram(device, program));
}

// ============================================================================
// TT_KERNEL ("1st world arguments") compute-path shim — JIT compile smoke test
// ============================================================================
//
// Compiles a TT_KERNEL compute kernel through genfiles + the RISC-V compiler on a MOCK Wormhole
// device (no silicon, no dispatch) via detail::CompileProgram. This is the only no-hardware
// coverage that the generated kernel_main() shim is emitted on the COMPUTE (TRISC) compile path
// and actually compiles: the on-hardware compute test (TtKernelNamedArgsLoopbackCompute) skips in
// CI, and the shim unit tests only check the generated string, not its genfiles wiring.
//
// (A Quasar variant would also exercise the 4th TRISC, isolate_sfpu, but mock-Quasar JIT-compile
// isn't wired up in this checkout — the Quasar TRISC firmware objects aren't built, so the link
// step fails. The fix is arch-correct by construction regardless: the shim is appended to the same
// source for every TRISC, and run_kernel() calls kernel_main() on Quasar too.)

// Minimal TT_KERNEL compute entry: CTAs as template params, RTA/CRTA as function params, producing
// into a DFB. The point is solely that the kernel_main() shim is generated on the TRISC path and
// the whole thing compiles; the body avoids any arch-specific raw-L1 pokes.
constexpr const char* kTtKernelComputeShimSource = R"(
#include "api/compute/common.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"
template <uint32_t magic, uint32_t entry_size>                             // CTAs
TT_KERNEL void compute_entry(uint32_t input_offset, uint32_t num_tiles) {  // RTA, CRTA
    DataflowBuffer out(dfb::out_dfb);
    out.reserve_back(num_tiles);
    out.push_back(num_tiles);
    volatile uint32_t sink = magic ^ entry_size ^ input_offset;
    (void)sink;
}
)";

TEST_F(ProgramSpecTestGen1, TtKernelComputeShimCompiles) {
    const NodeCoord node{0, 0};
    constexpr uint32_t entry_size = 1024;

    // Compute kernel authored in TT_KERNEL form, producing into a DFB drained by a trivial consumer.
    auto compute = MakeMinimalGen1ComputeKernel("compute");
    compute.source = KernelSpec::SourceCode{kTtKernelComputeShimSource};
    compute.runtime_arg_schema.runtime_arg_names = {"input_offset"};
    compute.runtime_arg_schema.common_runtime_arg_names = {"num_tiles"};
    compute.compile_time_args = {{"magic", 0xCAFE0001u}, {"entry_size", entry_size}};

    auto consumer = MakeMinimalReaderDMKernel("consumer");  // trivial drain kernel

    auto out_dfb = MakeMinimalDFB("out_dfb", entry_size, 4);
    out_dfb.data_format_metadata = tt::DataFormat::Float16_b;  // required for a compute DFB endpoint
    compute.dfb_bindings.push_back(ProducerOf(DFBSpecName{"out_dfb"}, "out_dfb"));
    consumer.dfb_bindings.push_back(ConsumerOf(DFBSpecName{"out_dfb"}, "out_dfb"));

    ProgramSpec spec;
    spec.name = "tt_kernel_compute_shim_compile";
    spec.kernels = {compute, consumer};
    spec.dataflow_buffers = {out_dfb};
    spec.work_units = std::vector<WorkUnitSpec>{MakeMinimalWorkUnit("wu", node, {"compute", "consumer"})};

    Program program = MakeProgramFromSpec(*mesh_device_, spec);
    IDevice* device = mesh_device_->get_devices()[0];
    EXPECT_NO_THROW(detail::CompileProgram(device, program));
}

// ============================================================================
// Kernel hash sensitivity to TensorParameter spec
// ============================================================================
//
// The kernel's JIT cache key is its compute_hash(); two kernels that hash equal share a cached
// binary. The TensorParameter's TensorSpec flows into the kernel's positional CTAs (via
// ResolveTensorParameterStaticCTAs), and compile_time_args_ is part of the hash, so:
//   - same kernel source + different TensorSpec  =>  different hash  =>  forced recompile
//   - same kernel source + identical TensorSpec  =>  same hash       =>  cache reuse
// These are regression canaries for the cache-key chain. If a future refactor drops a field from
// the CTA payload, or changes compute_hash to skip compile_time_args_, the silent failure mode is
// a stale cached binary running with mismatched layout metadata — nasty to debug in prod.

TEST_F(ProgramSpecTestGen1, DifferentTensorSpecProducesDifferentKernelHash) {
    // Same kernel source, same accessor name. The two TensorParameters differ only in BufferType
    // (DRAM vs L1), which flips the is_dram bit in the binding's args_config CTA word.
    auto make_spec = [](tt::tt_metal::BufferType buffer_type) {
        ProgramSpec spec = MakeMinimalGen1ValidProgramSpec();
        spec.tensor_parameters = {MakeMinimalTensorParameter("input_tensor", buffer_type)};
        BindTensorParameterToKernel(spec.kernels[0], "input_tensor", "input_ta");
        return spec;
    };
    Program prog_dram = MakeProgramFromSpec(*mesh_device_, make_spec(tt::tt_metal::BufferType::DRAM));
    Program prog_l1 = MakeProgramFromSpec(*mesh_device_, make_spec(tt::tt_metal::BufferType::L1));

    auto hash_dram = prog_dram.impl().get_kernel_by_spec_name("dm_kernel")->compute_hash();
    auto hash_l1 = prog_l1.impl().get_kernel_by_spec_name("dm_kernel")->compute_hash();
    EXPECT_NE(hash_dram, hash_l1);
}

TEST_F(ProgramSpecTestGen1, IdenticalTensorSpecProducesIdenticalKernelHash) {
    // Determinism canary: two specs constructed identically must hash identically. If this ever
    // fails, something nondeterministic crept into the hash (iteration order over a hash map,
    // pointer values, timestamps, etc.).
    auto make_spec = [] {
        ProgramSpec spec = MakeMinimalGen1ValidProgramSpec();
        spec.tensor_parameters = {MakeMinimalTensorParameter("input_tensor")};
        BindTensorParameterToKernel(spec.kernels[0], "input_tensor", "input_ta");
        return spec;
    };
    Program prog_a = MakeProgramFromSpec(*mesh_device_, make_spec());
    Program prog_b = MakeProgramFromSpec(*mesh_device_, make_spec());

    auto hash_a = prog_a.impl().get_kernel_by_spec_name("dm_kernel")->compute_hash();
    auto hash_b = prog_b.impl().get_kernel_by_spec_name("dm_kernel")->compute_hash();
    EXPECT_EQ(hash_a, hash_b);
}

// ============================================================================
// Dynamic tensor shape — spec resolution
// ============================================================================
//
// dynamic_tensor_shape on a TensorParameter loosens the runtime spec match (covered in
// test_program_run_args.cpp) and, for sharded TensorParameters, also moves the
// tensor_shape_in_pages words out of the kernel's CTAs and into per-binding CRTA slots.
// The tests below pin both halves:
//   - CTA stability: same kernel hash across different shapes when the flag is set.
//   - Handle layout: num_runtime_field_crta_words tracks the rank when needed.

TEST_F(ProgramSpecTestGen1, DynamicTensorShape_InterleavedKernelHashStableAcrossShapes_TileLayout) {
    // Interleaved + TILE layout: page_size is constant per dtype regardless of logical_shape, so
    // the CTAs ([args_config.raw(), aligned_page_size]) are stable across shape variations.
    // The dynamic flag thus enables JIT cache reuse for tile-layout eltwise.
    //
    // (Row-major interleaved has a shape-dependent page_size — the last-dim element count. Under
    // dynamic_tensor_shape the resolver demotes that page size to a per-binding CRTA word, so its
    // CTAs become shape-stable too; that path is covered by
    // DynamicTensorShape_InterleavedRowMajorKernelHashStableAcrossWidths below. This test focuses on
    // tile, whose page size is dtype-fixed and never rode a shape-dependent CTA in the first place.)
    auto make_spec = [](tt::tt_metal::Shape shape) {
        ProgramSpec spec = MakeMinimalGen1ValidProgramSpec();
        auto page_config = tt::tt_metal::PageConfig(tt::tt_metal::Layout::TILE);
        auto memory_config =
            tt::tt_metal::MemoryConfig{tt::tt_metal::TensorMemoryLayout::INTERLEAVED, tt::tt_metal::BufferType::DRAM};
        auto tensor_layout = tt::tt_metal::TensorLayout(tt::tt_metal::DataType::BFLOAT16, page_config, memory_config);
        TensorParameter tp{
            .unique_id = TensorParamName{"input_tensor"},
            .spec = tt::tt_metal::TensorSpec(std::move(shape), std::move(tensor_layout)),
            .advanced_options = TensorParameterAdvancedOptions{.dynamic_tensor_shape = true},
        };
        spec.tensor_parameters = {tp};
        BindTensorParameterToKernel(spec.kernels[0], "input_tensor", "input_ta");
        return spec;
    };
    Program prog_a = MakeProgramFromSpec(*mesh_device_, make_spec(tt::tt_metal::Shape{1, 1, 32, 32}));
    Program prog_b = MakeProgramFromSpec(*mesh_device_, make_spec(tt::tt_metal::Shape{1, 1, 64, 64}));
    EXPECT_EQ(
        prog_a.impl().get_kernel_by_spec_name("dm_kernel")->compute_hash(),
        prog_b.impl().get_kernel_by_spec_name("dm_kernel")->compute_hash())
        << "Interleaved tile + dynamic_tensor_shape: shape variations must hash equal so the "
           "same compiled kernel binary is reused.";
}

TEST_F(ProgramSpecTestGen1, DynamicTensorShape_InterleavedRowMajorKernelHashStableAcrossWidths) {
    // The payoff of the page-size fold. For a ROW-MAJOR interleaved TensorParameter the page size
    // (last_dim_width * elem_size) is shape-dependent. WITHOUT the flag it rides a compile-time arg,
    // so two different-width tensors hash differently -- distinct cache entries, and (worse) a stale
    // page size baked into a binary that gets reused on a cache hit: the exact bug this feature
    // fixes. WITH dynamic_tensor_shape the page size moves to a CRTA, the CTAs become width-
    // independent, and the two widths hash identically -- one cached binary, refreshed per-dispatch.
    auto make_spec = [](uint32_t width, bool dynamic) {
        ProgramSpec spec = MakeMinimalGen1ValidProgramSpec();
        auto page_config = tt::tt_metal::PageConfig(tt::tt_metal::Layout::ROW_MAJOR);
        auto memory_config =
            tt::tt_metal::MemoryConfig{tt::tt_metal::TensorMemoryLayout::INTERLEAVED, tt::tt_metal::BufferType::DRAM};
        auto tensor_layout = tt::tt_metal::TensorLayout(tt::tt_metal::DataType::BFLOAT16, page_config, memory_config);
        TensorParameter tp{
            .unique_id = TensorParamName{"input_tensor"},
            .spec = tt::tt_metal::TensorSpec(tt::tt_metal::Shape{1, 1, 32, width}, std::move(tensor_layout)),
            .advanced_options = TensorParameterAdvancedOptions{.dynamic_tensor_shape = dynamic},
        };
        spec.tensor_parameters = {tp};
        BindTensorParameterToKernel(spec.kernels[0], "input_tensor", "input_ta");
        return spec;
    };

    // Baseline (flag off): different widths -> different page-size CTA -> different hash.
    Program s_a = MakeProgramFromSpec(*mesh_device_, make_spec(/*width=*/64, /*dynamic=*/false));
    Program s_b = MakeProgramFromSpec(*mesh_device_, make_spec(/*width=*/128, /*dynamic=*/false));
    EXPECT_NE(
        s_a.impl().get_kernel_by_spec_name("dm_kernel")->compute_hash(),
        s_b.impl().get_kernel_by_spec_name("dm_kernel")->compute_hash())
        << "Baseline: row-major page size rides a CTA, so different widths must hash differently.";

    // With the flag: page size -> CRTA, CTAs width-independent -> identical hash (cache reuse).
    Program d_a = MakeProgramFromSpec(*mesh_device_, make_spec(/*width=*/64, /*dynamic=*/true));
    Program d_b = MakeProgramFromSpec(*mesh_device_, make_spec(/*width=*/128, /*dynamic=*/true));
    EXPECT_EQ(
        d_a.impl().get_kernel_by_spec_name("dm_kernel")->compute_hash(),
        d_b.impl().get_kernel_by_spec_name("dm_kernel")->compute_hash())
        << "Row-major interleaved + dynamic_tensor_shape: page size moves to a CRTA, so different "
           "widths hash equally (program-cache reuse; prevents the stale-page-size bug).";
}

TEST_F(ProgramSpecTestGen1, DynamicTensorShape_ShardedKernelHashStableAcrossShapes) {
    // Sharded TensorParameters DO encode tensor_shape_in_pages in CTAs by default — so without
    // the dynamic flag, two different-shape sharded TPs hash differently. With the flag, the
    // tensor_shape words move to CRTAs and the CTAs become stable across shape variations.
    //
    // Layout: HEIGHT_SHARDED with shard_shape {32, 32} on 2 cores → 2 shards along height,
    // full width per shard. The declared (64, 32) tensor has 2 shards; the alternate (32, 32)
    // tensor needs only 1 shard (subset of the 2-core grid).
    auto make_spec = [](const tt::tt_metal::Shape& shape, bool dynamic) {
        ProgramSpec spec = MakeMinimalGen1ValidProgramSpec();
        auto tp = MakeShardedTensorParameter("input_tensor", shape, {32, 32}, /*num_cores=*/2);
        tp.advanced_options = TensorParameterAdvancedOptions{.dynamic_tensor_shape = dynamic};
        spec.tensor_parameters = {tp};
        BindTensorParameterToKernel(spec.kernels[0], "input_tensor", "input_ta");
        return spec;
    };

    // Without the flag: differently-shaped sharded TPs hash differently (regression canary).
    Program prog_static_a = MakeProgramFromSpec(*mesh_device_, make_spec(tt::tt_metal::Shape{1, 1, 64, 32}, false));
    Program prog_static_b = MakeProgramFromSpec(*mesh_device_, make_spec(tt::tt_metal::Shape{1, 1, 32, 32}, false));
    EXPECT_NE(
        prog_static_a.impl().get_kernel_by_spec_name("dm_kernel")->compute_hash(),
        prog_static_b.impl().get_kernel_by_spec_name("dm_kernel")->compute_hash())
        << "Baseline: without dynamic_tensor_shape, different shapes should hash differently.";

    // With the flag: same two shapes hash identically — CTAs are now stable.
    Program prog_dyn_a = MakeProgramFromSpec(*mesh_device_, make_spec(tt::tt_metal::Shape{1, 1, 64, 32}, true));
    Program prog_dyn_b = MakeProgramFromSpec(*mesh_device_, make_spec(tt::tt_metal::Shape{1, 1, 32, 32}, true));
    EXPECT_EQ(
        prog_dyn_a.impl().get_kernel_by_spec_name("dm_kernel")->compute_hash(),
        prog_dyn_b.impl().get_kernel_by_spec_name("dm_kernel")->compute_hash())
        << "Sharded + dynamic_tensor_shape: tensor_shape moves to CRTAs, so CTA-driven hash "
           "must be stable across shape variations.";
}

TEST_F(ProgramSpecTestGen1, DynamicTensorShape_ShardedBindingTracksShapeCRTASlots) {
    // Sharded + dynamic_tensor_shape: the TensorBindingHandle's num_runtime_field_crta_words
    // should equal the BufferDistributionSpec's tensor_shape_in_pages rank (one CRTA word per
    // shape dim, written at enqueue).
    //
    // Note: BDS flattens the logical_shape via its sharding scheme, so the BDS rank is not
    // generally the same as logical_shape.rank(). We derive the expected value from the BDS
    // directly to be robust against BDS-internal flattening conventions.
    ProgramSpec spec = MakeMinimalGen1ValidProgramSpec();
    auto tp = MakeShardedTensorParameter("input_tensor", tt::tt_metal::Shape{1, 1, 64, 32}, {32, 32}, 2);
    tp.advanced_options = TensorParameterAdvancedOptions{.dynamic_tensor_shape = true};
    spec.tensor_parameters = {tp};
    BindTensorParameterToKernel(spec.kernels[0], "input_tensor", "input_ta");

    Program program = MakeProgramFromSpec(*mesh_device_, spec);
    auto kernel = program.impl().get_kernel_by_spec_name("dm_kernel");
    const auto& handles = kernel->tensor_binding_handles();
    ASSERT_EQ(handles.size(), 1u);

    const auto bds = tp.spec.compute_buffer_sharding_args().buffer_distribution_spec();
    ASSERT_TRUE(bds.has_value());
    const auto expected_rank = bds->tensor_shape_in_pages().rank();
    EXPECT_GT(expected_rank, 0u);
    EXPECT_EQ(handles[0].num_runtime_field_crta_words, static_cast<uint32_t>(expected_rank))
        << "Sharded + dynamic_tensor_shape: runtime-field CRTA words should equal BDS shape rank.";
}

TEST_F(ProgramSpecTestGen1, DynamicTensorShape_InterleavedRowMajorBindingTracksPageSizeCRTASlot) {
    // Row-major interleaved + dynamic_tensor_shape: the page size (= last_dim_width * elem_size) is
    // part of the varying shape, so the resolver folds it from a compile-time arg into a single
    // per-binding CRTA word ("A-collapse": the page-size CTA slot is dropped and the RuntimePageSize
    // bit is set in args_config). The binding handle must advertise exactly one runtime field word,
    // tagged as the page-size kind. MakeMinimalTensorParameter is BFLOAT16 / ROW_MAJOR / interleaved.
    auto make_spec = [](bool dynamic) {
        ProgramSpec spec = MakeMinimalGen1ValidProgramSpec();
        auto tp = MakeMinimalTensorParameter("input_tensor");
        tp.advanced_options = TensorParameterAdvancedOptions{.dynamic_tensor_shape = dynamic};
        spec.tensor_parameters = {tp};
        BindTensorParameterToKernel(spec.kernels[0], "input_tensor", "input_ta");
        return spec;
    };

    Program prog_dyn = MakeProgramFromSpec(*mesh_device_, make_spec(/*dynamic=*/true));
    auto kernel = prog_dyn.impl().get_kernel_by_spec_name("dm_kernel");
    const auto& handles = kernel->tensor_binding_handles();
    ASSERT_EQ(handles.size(), 1u);
    EXPECT_EQ(handles[0].num_runtime_field_crta_words, 1u)
        << "Row-major interleaved + dynamic_tensor_shape: page size demotes to exactly one CRTA word.";
    EXPECT_TRUE(handles[0].runtime_field_is_page_size)
        << "The runtime field must be tagged as the page-size kind (not the sharded-shape kind).";

    // The binding's args_config CTA word carries the RuntimePageSize bit.
    const std::vector<uint32_t> dyn_ctas = kernel->compile_time_args();
    ASSERT_LT(handles[0].cta_offset, dyn_ctas.size());
    const auto dyn_cfg = tensor_accessor::ArgsConfig(
        static_cast<tensor_accessor::ArgsConfig::Underlying>(dyn_ctas[handles[0].cta_offset]));
    EXPECT_TRUE(dyn_cfg.test(tensor_accessor::ArgConfig::RuntimePageSize))
        << "RuntimePageSize bit must be set in the binding's args_config word.";

    // A-collapse: the dynamic binding omits the page-size CTA word that the static (bit-off) binding
    // carries, so the whole-kernel CTA count is exactly one shorter. The two specs are identical
    // apart from the flag, so the size delta is precisely the dropped page-size slot.
    Program prog_static = MakeProgramFromSpec(*mesh_device_, make_spec(/*dynamic=*/false));
    const std::vector<uint32_t> static_ctas =
        prog_static.impl().get_kernel_by_spec_name("dm_kernel")->compile_time_args();
    EXPECT_EQ(static_ctas.size(), dyn_ctas.size() + 1u)
        << "Static binding keeps the page-size CTA; the dynamic binding drops it (A-collapse).";
    const auto static_cfg = tensor_accessor::ArgsConfig(
        static_cast<tensor_accessor::ArgsConfig::Underlying>(static_ctas[handles[0].cta_offset]));
    EXPECT_FALSE(static_cfg.test(tensor_accessor::ArgConfig::RuntimePageSize))
        << "Without the flag, the RuntimePageSize bit must NOT be set.";
}

TEST_F(ProgramSpecTestGen1, DynamicTensorShape_InterleavedTileBindingHasNoRuntimeFieldCRTAs) {
    // Interleaved + TILE layout: the page size is dtype/tile-fixed, independent of logical shape, so
    // dynamic_tensor_shape does NOT demote it to a CRTA (the fold gates on ROW_MAJOR). The flag is a
    // pure host-side validation loosening here; the binding carries no runtime field words. Guards
    // the layout gate -- a regression that demoted tile page sizes would trip this.
    ProgramSpec spec = MakeMinimalGen1ValidProgramSpec();
    auto page_config = tt::tt_metal::PageConfig(tt::tt_metal::Layout::TILE);
    auto memory_config =
        tt::tt_metal::MemoryConfig{tt::tt_metal::TensorMemoryLayout::INTERLEAVED, tt::tt_metal::BufferType::DRAM};
    auto tensor_layout = tt::tt_metal::TensorLayout(tt::tt_metal::DataType::BFLOAT16, page_config, memory_config);
    TensorParameter tp{
        .unique_id = TensorParamName{"input_tensor"},
        .spec = tt::tt_metal::TensorSpec(tt::tt_metal::Shape{1, 1, 32, 32}, std::move(tensor_layout)),
        .advanced_options = TensorParameterAdvancedOptions{.dynamic_tensor_shape = true},
    };
    spec.tensor_parameters = {tp};
    BindTensorParameterToKernel(spec.kernels[0], "input_tensor", "input_ta");

    Program program = MakeProgramFromSpec(*mesh_device_, spec);
    auto kernel = program.impl().get_kernel_by_spec_name("dm_kernel");
    const auto& handles = kernel->tensor_binding_handles();
    ASSERT_EQ(handles.size(), 1u);
    EXPECT_EQ(handles[0].num_runtime_field_crta_words, 0u)
        << "Interleaved TILE + dynamic_tensor_shape is host-side-only; no runtime CRTA words.";
    EXPECT_FALSE(handles[0].runtime_field_is_page_size);
}

TEST_F(ProgramSpecTestGen1, KernelCrtaLayout_AllThreeSectionsConsistent) {
    // The Kernel's stored KernelCrtaLayout must equal what a fresh walk of (named CRTAs +
    // binding handles) would compute. This test exercises a Program in which ALL THREE
    // sections of the CRTA buffer are non-empty:
    //   - section 1: named CRTAs           (declared in runtime_arg_schema)
    //   - section 2: TensorBinding section (variable-size: a plain interleaved binding +
    //                                       a sharded-with-dynamic_tensor_shape binding)
    //   - section 3: varargs               (declared via num_common_runtime_varargs)
    //
    // The headergen bakes vararg_section_offset into the kernel's `get_common_vararg(idx)`
    // macro, so a wrong offset here would silently route vararg reads into the binding
    // section. The walk-based reference value is exactly what genfiles used to compute on
    // its own; the refactor moves that computation into ResolveTensorBindingsForKernel and
    // threads it through. This test guards against the threading silently producing a
    // different value than the walk would.
    ProgramSpec spec = MakeMinimalGen1ValidProgramSpec();

    // Section 1: named CRTAs on the DM kernel.
    spec.kernels[0].runtime_arg_schema.common_runtime_arg_names = {"foo", "bar"};
    // Section 3: vararg CRTAs on the DM kernel.
    spec.kernels[0].advanced_options.num_common_runtime_varargs = 3;

    // Section 2: two bindings — one plain (1 word), one sharded+dynamic_tensor_shape
    // (1 word + tensor_shape_in_pages rank words).
    auto plain_tp = MakeMinimalTensorParameter("plain_tensor");
    auto dyn_tp =
        MakeShardedTensorParameter("dyn_tensor", tt::tt_metal::Shape{1, 1, 64, 32}, {32, 32}, /*num_cores=*/2);
    dyn_tp.advanced_options = TensorParameterAdvancedOptions{.dynamic_tensor_shape = true};
    spec.tensor_parameters = {plain_tp, dyn_tp};
    BindTensorParameterToKernel(spec.kernels[0], "plain_tensor", "plain_ta");
    BindTensorParameterToKernel(spec.kernels[0], "dyn_tensor", "dyn_ta");

    Program program = MakeProgramFromSpec(*mesh_device_, spec);
    auto kernel = program.impl().get_kernel_by_spec_name("dm_kernel");
    const KernelCrtaLayout layout = kernel->get_crta_layout();

    // Reference values, re-derived independently of the layout struct.
    const uint32_t expected_named_words = 2u;  // "foo", "bar"
    uint32_t expected_binding_words = 0;
    for (const auto& handle : kernel->tensor_binding_handles()) {
        expected_binding_words += 1u + handle.num_runtime_field_crta_words;
    }
    const uint32_t expected_vararg_offset = expected_named_words + expected_binding_words;

    EXPECT_EQ(layout.num_named_words, expected_named_words);
    EXPECT_EQ(layout.binding_section_words, expected_binding_words);
    EXPECT_EQ(layout.vararg_section_offset, expected_vararg_offset)
        << "vararg_section_offset must equal num_named_words + binding_section_words; this is the "
           "value baked into get_common_vararg(idx) by the kernel headergen.";

    // Sanity: the dynamic-shape binding's runtime-field word count should be > 0, so this test
    // genuinely exercises a variable-size binding (not just two 1-word bindings that would also
    // pass with the old binding-count-based math).
    ASSERT_EQ(kernel->tensor_binding_handles().size(), 2u);
    EXPECT_GT(kernel->tensor_binding_handles()[1].num_runtime_field_crta_words, 0u)
        << "Test precondition: the second binding should be variable-size; otherwise the layout "
           "calculation degenerates to the pre-refactor case.";
}

TEST_F(ProgramSpecTestGen1, CompilerIncludePathsForwardedToKernelConfig) {
    // KernelSpec.compiler_options.include_paths should be picked up as `-I<path>` flags
    NodeCoord node{0, 0};

    ProgramSpec spec;
    spec.name = "test_program";

    const std::vector<std::filesystem::path> dm_paths = {"/tmp/dm_first", "/tmp/dm_second"};
    const std::vector<std::filesystem::path> compute_paths = {"/tmp/compute_only"};

    auto dm_kernel = MakeMinimalGen1DMKernel("dm_kernel", DataMovementProcessor::RISCV_0);
    dm_kernel.compiler_options.include_paths = dm_paths;

    auto compute_kernel = MakeMinimalGen1ComputeKernel("compute_kernel");
    compute_kernel.compiler_options.include_paths = compute_paths;

    auto dfb = MakeMinimalDFB("dfb_0");
    dfb.data_format_metadata = tt::DataFormat::Float16_b;
    dm_kernel.dfb_bindings.push_back(ProducerOf(DFBSpecName{"dfb_0"}, "input_dfb"));
    compute_kernel.dfb_bindings.push_back(ConsumerOf(DFBSpecName{"dfb_0"}, "input_dfb"));

    spec.kernels = {dm_kernel, compute_kernel};
    spec.dataflow_buffers = {dfb};
    spec.work_units =
        std::vector<WorkUnitSpec>{MakeMinimalWorkUnit("work_unit", node, {"dm_kernel", "compute_kernel"})};

    Program program = MakeProgramFromSpec(*mesh_device_, spec);

    const auto& impl = program.impl();
    auto built_dm = impl.get_kernel_by_spec_name("dm_kernel");
    auto built_compute = impl.get_kernel_by_spec_name("compute_kernel");

    const auto built_dm_variant = built_dm->config();
    const auto& built_dm_config = std::get<DataMovementConfig>(built_dm_variant);
    EXPECT_EQ(built_dm_config.compiler_include_paths, dm_paths);

    const auto built_compute_variant = built_compute->config();
    const auto& built_compute_config = std::get<ComputeConfig>(built_compute_variant);
    EXPECT_EQ(built_compute_config.compiler_include_paths, compute_paths);
}

// ============================================================================
// DFB alias validation
// ============================================================================

// Helper: build a minimal 1-producer / 1-consumer ProgramSpec where both DFBs are
// bound to the same producer/consumer kernels in a single WorkUnit on a single node.
namespace {
ProgramSpec MakeAliasProgramSpec(
    const NodeCoord& node,
    const DataflowBufferSpec& dfb_a,
    const DataflowBufferSpec& dfb_b) {
    ProgramSpec spec;

    KernelSpec producer = MakeMinimalGen2DMKernel("producer_kernel");
    KernelSpec consumer = MakeMinimalGen2DMKernel("consumer_kernel");

    producer.dfb_bindings.push_back(ProducerOf(dfb_a.unique_id, "out_a"));
    consumer.dfb_bindings.push_back(ConsumerOf(dfb_a.unique_id, "in_a"));

    producer.dfb_bindings.push_back(ProducerOf(dfb_b.unique_id, "out_b"));
    consumer.dfb_bindings.push_back(ConsumerOf(dfb_b.unique_id, "in_b"));

    spec.kernels = {producer, consumer};
    spec.dataflow_buffers = {dfb_a, dfb_b};
    spec.work_units = {MakeMinimalWorkUnit("wu", node, {"producer_kernel", "consumer_kernel"})};
    return spec;
}
}  // anonymous namespace

TEST_F(ProgramSpecTestQuasar, AliasDFBFailsOnMismatchedTotalSize) {
    // DFB_A: 512 * 8 = 4096 bytes, DFB_B: 256 * 8 = 2048 bytes — different totals → TT_FATAL
    auto dfb_a = MakeMinimalDFB("dfb_a", /*entry_size=*/512, /*num_entries=*/8);
    auto dfb_b = MakeMinimalDFB("dfb_b", /*entry_size=*/256, /*num_entries=*/8);
    dfb_a.advanced_options = DFBAdvancedOptions{.alias_with = {DFBSpecName{"dfb_b"}}};
    dfb_b.advanced_options = DFBAdvancedOptions{.alias_with = {DFBSpecName{"dfb_a"}}};

    const NodeCoord node{0, 0};
    auto spec = MakeAliasProgramSpec(node, dfb_a, dfb_b);

    EXPECT_THAT(
        [&] { MakeProgramFromSpec(*mesh_device_, spec); },
        ::testing::ThrowsMessage<std::runtime_error>(
            ::testing::HasSubstr("different total sizes")));
}

TEST_F(ProgramSpecTestQuasar, AliasDFBFailsOnAsymmetricDeclaration) {
    // DFB_A lists DFB_B but DFB_B does not list DFB_A — clique violation → TT_FATAL
    auto dfb_a = MakeMinimalDFB("dfb_a", /*entry_size=*/512, /*num_entries=*/8);
    auto dfb_b = MakeMinimalDFB("dfb_b", /*entry_size=*/256, /*num_entries=*/16);
    dfb_a.advanced_options = DFBAdvancedOptions{.alias_with = {DFBSpecName{"dfb_b"}}};
    // dfb_b.alias_with intentionally left empty

    const NodeCoord node{0, 0};
    auto spec = MakeAliasProgramSpec(node, dfb_a, dfb_b);

    EXPECT_THAT(
        [&] { MakeProgramFromSpec(*mesh_device_, spec); },
        ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr("do not declare the same alias group")));
}

TEST_F(ProgramSpecTestQuasar, AliasDFBMatmulStyleSucceeds) {
    // This is the nasty case from the matmul op....
    // Two DFBs share L1 but are bound to different kernels.
    //  - DFB_A is bound to {producer_kernel, consumer_kernel};
    //  - DFB_B is bound to {producer_kernel, other_kernel}.
    //
    // This looks unspeakably evil and I'd like to forbid it. But, it does work.
    // All kernels run on the same node set, so they all have the same L1.
    // And (presumably) the DFB is used in a temporally disjoint way.
    // So, nothing stops them from re-using the DFB memory.

    const NodeCoord node{0, 0};

    auto dfb_a = MakeMinimalDFB("dfb_a", /*entry_size=*/512, /*num_entries=*/8);
    auto dfb_b = MakeMinimalDFB("dfb_b", /*entry_size=*/512, /*num_entries=*/8);
    dfb_a.advanced_options = DFBAdvancedOptions{.alias_with = {DFBSpecName{"dfb_b"}}};
    dfb_b.advanced_options = DFBAdvancedOptions{.alias_with = {DFBSpecName{"dfb_a"}}};

    KernelSpec producer = MakeMinimalGen2DMKernel("producer_kernel");
    KernelSpec consumer = MakeMinimalGen2DMKernel("consumer_kernel");
    KernelSpec other = MakeMinimalGen2DMKernel("other_kernel");

    producer.dfb_bindings.push_back(ProducerOf(DFBSpecName{"dfb_a"}, "out_a"));
    consumer.dfb_bindings.push_back(ConsumerOf(DFBSpecName{"dfb_a"}, "in_a"));
    producer.dfb_bindings.push_back(ProducerOf(DFBSpecName{"dfb_b"}, "out_b"));
    other.dfb_bindings.push_back(ConsumerOf(DFBSpecName{"dfb_b"}, "in_b"));

    ProgramSpec spec;
    spec.kernels = {producer, consumer, other};
    spec.dataflow_buffers = {dfb_a, dfb_b};
    spec.work_units = {
        MakeMinimalWorkUnit("wu", node, {"producer_kernel", "consumer_kernel", "other_kernel"})};

    EXPECT_NO_THROW(MakeProgramFromSpec(*mesh_device_, spec));
}

TEST_F(ProgramSpecTestQuasar, AliasDFBFailsOnDifferentNodeCoverage) {
    // Two DFBs aliased, but bound to kernels running on disjoint nodes. The shared L1
    // region only makes sense if both members cover the same cores; otherwise the
    // secondary's address propagation would alias into L1 the primary never reserved.
    NodeCoord node_a{0, 0};
    NodeCoord node_b{1, 0};

    auto dfb_a = MakeMinimalDFB("dfb_a", /*entry_size=*/512, /*num_entries=*/8);
    auto dfb_b = MakeMinimalDFB("dfb_b", /*entry_size=*/512, /*num_entries=*/8);
    dfb_a.advanced_options = DFBAdvancedOptions{.alias_with = {DFBSpecName{"dfb_b"}}};
    dfb_b.advanced_options = DFBAdvancedOptions{.alias_with = {DFBSpecName{"dfb_a"}}};

    KernelSpec producer_a = MakeMinimalGen2DMKernel("producer_a");
    KernelSpec consumer_a = MakeMinimalGen2DMKernel("consumer_a");
    KernelSpec producer_b = MakeMinimalGen2DMKernel("producer_b");
    KernelSpec consumer_b = MakeMinimalGen2DMKernel("consumer_b");
    producer_a.dfb_bindings.push_back(ProducerOf(DFBSpecName{"dfb_a"}, "out_a"));
    consumer_a.dfb_bindings.push_back(ConsumerOf(DFBSpecName{"dfb_a"}, "in_a"));
    producer_b.dfb_bindings.push_back(ProducerOf(DFBSpecName{"dfb_b"}, "out_b"));
    consumer_b.dfb_bindings.push_back(ConsumerOf(DFBSpecName{"dfb_b"}, "in_b"));

    ProgramSpec spec;
    spec.kernels = {producer_a, consumer_a, producer_b, consumer_b};
    spec.dataflow_buffers = {dfb_a, dfb_b};
    spec.work_units = {
        MakeMinimalWorkUnit("wu_a", node_a, {"producer_a", "consumer_a"}),
        MakeMinimalWorkUnit("wu_b", node_b, {"producer_b", "consumer_b"}),
    };

    EXPECT_THAT(
        [&] { MakeProgramFromSpec(*mesh_device_, spec); },
        ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr("cover different sets of nodes")));
}

TEST_F(ProgramSpecTestQuasar, AliasDFBFailsOnInconsistentBorrowedFrom) {
    // DFB_A borrows from a TensorParameter, DFB_B does not. Within an alias group, either
    // no member borrows or all members borrow from the same TensorParameter.
    const NodeCoord node{0, 0};

    auto dfb_a = MakeMinimalDFB("dfb_a", /*entry_size=*/16, /*num_entries=*/2);
    auto dfb_b = MakeMinimalDFB("dfb_b", /*entry_size=*/16, /*num_entries=*/2);
    dfb_a.advanced_options = DFBAdvancedOptions{.alias_with = {DFBSpecName{"dfb_b"}}};
    dfb_b.advanced_options = DFBAdvancedOptions{.alias_with = {DFBSpecName{"dfb_a"}}};
    dfb_a.borrowed_from = TensorParamName{"borrowed_tensor"};
    // dfb_b.borrowed_from intentionally left unset

    KernelSpec producer = MakeMinimalGen2DMKernel("producer_kernel");
    KernelSpec consumer = MakeMinimalGen2DMKernel("consumer_kernel");
    producer.dfb_bindings.push_back(ProducerOf(DFBSpecName{"dfb_a"}, "out_a"));
    consumer.dfb_bindings.push_back(ConsumerOf(DFBSpecName{"dfb_a"}, "in_a"));
    producer.dfb_bindings.push_back(ProducerOf(DFBSpecName{"dfb_b"}, "out_b"));
    consumer.dfb_bindings.push_back(ConsumerOf(DFBSpecName{"dfb_b"}, "in_b"));

    auto tensor_param = MakeMinimalTensorParameter("borrowed_tensor", tt::tt_metal::BufferType::L1);
    BindTensorParameterToKernel(producer, "borrowed_tensor", "borrowed_t");

    ProgramSpec spec;
    spec.kernels = {producer, consumer};
    spec.dataflow_buffers = {dfb_a, dfb_b};
    spec.tensor_parameters = {tensor_param};
    spec.work_units = {MakeMinimalWorkUnit("wu", node, {"producer_kernel", "consumer_kernel"})};

    EXPECT_THAT(
        [&] { MakeProgramFromSpec(*mesh_device_, spec); },
        ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr("inconsistent borrowed_from")));
}

}  // namespace
}  // namespace tt::tt_metal::experimental
