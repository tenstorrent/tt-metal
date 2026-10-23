// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Type-level properties of every Metal 2.0 spec struct: each is an aggregate (so designated
// initializers work) and is hashable via ttsl reflection.

#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <cstdint>
#include <optional>
#include <type_traits>

#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt_stl/reflection.hpp>

#include "metal2_host_api/test_helpers/test_helpers.hpp"

namespace tt::tt_metal::experimental {
namespace {

using test_helpers::MakeMinimalValidProgramSpec;

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
static_assert(hashable_v<ScratchpadSpec>, "ScratchpadSpec must be hashable via ttsl reflection");
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
static_assert(
    hashable_v<DataMovementHardwareConfig::DataMovement1XXConfig>,
    "DataMovement1XXConfig must be hashable via ttsl reflection");
static_assert(
    hashable_v<DataMovementHardwareConfig::DataMovement2XXConfig>,
    "DataMovement2XXConfig must be hashable via ttsl reflection");
static_assert(hashable_v<ComputeHardwareConfig>, "ComputeHardwareConfig must be hashable via ttsl reflection");
static_assert(
    hashable_v<ComputeHardwareConfig::Compute1XXConfig>, "Compute1XXConfig must be hashable via ttsl reflection");
static_assert(
    hashable_v<ComputeHardwareConfig::Compute2XXConfig>, "Compute2XXConfig must be hashable via ttsl reflection");

// Per-spec advanced options
static_assert(hashable_v<KernelAdvancedOptions>, "KernelAdvancedOptions must be hashable via ttsl reflection");
static_assert(hashable_v<DFBAdvancedOptions>, "DFBAdvancedOptions must be hashable via ttsl reflection");
static_assert(hashable_v<SemaphoreAdvancedOptions>, "SemaphoreAdvancedOptions must be hashable via ttsl reflection");
static_assert(hashable_v<TensorSpecRelaxations>, "TensorSpecRelaxations must be hashable via ttsl reflection");

static_assert(hashable_v<PrefetcherPipeParameter>, "PrefetcherPipeParameter must be hashable via ttsl reflection");
static_assert(hashable_v<PrefetcherPipeBinding>, "PrefetcherPipeBinding must be hashable via ttsl reflection");

TEST(ProgramSpecReflectionTest, CPU_IsHashable) {
    const ProgramSpec spec = MakeMinimalValidProgramSpec();

    // Deterministic: hashing the same spec twice yields the same value.
    EXPECT_EQ(ttsl::hash::hash_objects_with_default_seed(spec), ttsl::hash::hash_objects_with_default_seed(spec));

    // Sensitive: changing a field changes the hash.
    ProgramSpec modified = spec;
    modified.name += "_v2";
    EXPECT_NE(ttsl::hash::hash_objects_with_default_seed(spec), ttsl::hash::hash_objects_with_default_seed(modified));
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
static_assert(
    std::is_aggregate_v<ScratchpadSpec>, "ScratchpadSpec must remain an aggregate to support designated initializers");
static_assert(std::is_aggregate_v<DataMovementHardwareConfig>, "DataMovementHardwareConfig must remain an aggregate");
static_assert(
    std::is_aggregate_v<DataMovementHardwareConfig::DataMovement1XXConfig>,
    "DataMovement1XXConfig must remain an aggregate");
static_assert(
    std::is_aggregate_v<DataMovementHardwareConfig::DataMovement2XXConfig>,
    "DataMovement2XXConfig must remain an aggregate");
static_assert(std::is_aggregate_v<ComputeHardwareConfig>, "ComputeHardwareConfig must remain an aggregate");
static_assert(
    std::is_aggregate_v<ComputeHardwareConfig::Compute1XXConfig>, "Compute1XXConfig must remain an aggregate");
static_assert(
    std::is_aggregate_v<ComputeHardwareConfig::Compute2XXConfig>, "Compute2XXConfig must remain an aggregate");
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

TEST(AggregateSpecTypes, CPU_KernelSpecDesignatedInitializers) {
    // Demonstrates constructing KernelSpec with designated initializers
    KernelSpec dm_kernel{
        .unique_id = KernelSpecName{"my_dm_kernel"},
        .source = KernelSpec::SourceCode{"void kernel_main() {}"},
        .num_threads = 2,
        .hw_config = DataMovementHardwareConfig{},
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
                .fpu_math_fidelity = MathFidelity::LoFi,
                .enable_32_bit_dest = true,
            },
    };

    EXPECT_EQ(compute_kernel.unique_id.get(), "my_compute_kernel");
    EXPECT_TRUE(compute_kernel.is_compute_kernel());
}

TEST(AggregateSpecTypes, CPU_DataflowBufferSpecDesignatedInitializers) {
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

TEST(AggregateSpecTypes, ScratchpadSpecDesignatedInitializers) {
    ScratchpadSpec pad{
        .unique_id = ScratchpadSpecName{"pad"},
        .size_per_node = 1024,
    };
    EXPECT_EQ(pad.unique_id.get(), "pad");
    EXPECT_EQ(pad.size_per_node, 1024u);
    EXPECT_FALSE(pad.data_format_metadata.has_value());
    EXPECT_FALSE(pad.tile_format_metadata.has_value());

    ScratchpadSpec with_format{
        .unique_id = ScratchpadSpecName{"pad_fmt"},
        .size_per_node = 1024,
        .data_format_metadata = tt::DataFormat::Float16_b,
    };
    EXPECT_EQ(with_format.data_format_metadata, tt::DataFormat::Float16_b);
    EXPECT_FALSE(with_format.tile_format_metadata.has_value());
}

TEST(AggregateSpecTypes, CPU_WorkUnitSpecDesignatedInitializers) {
    // Demonstrates constructing WorkUnitSpec with designated initializers
    WorkUnitSpec work_unit{
        .name = "my_work_unit",
        .kernels = {KernelSpecName{"kernel1"}, KernelSpecName{"kernel2"}},
        .target_nodes = NodeCoord{0, 0},
    };

    EXPECT_EQ(work_unit.name, "my_work_unit");
    EXPECT_EQ(work_unit.kernels.size(), 2u);
}

TEST(AggregateSpecTypes, CPU_RuntimeArgSchemaDesignatedInitializers) {
    // Named RTAs + CRTAs via designated initializers; vararg counts now live on
    // KernelAdvancedOptions (see VarargCountsOnAdvancedOptions below).
    KernelSpec::RuntimeArgSchema schema{
        .runtime_arg_names = {"input_ptr", "output_ptr"},
        .common_runtime_arg_names = {"tile_count"},
    };

    EXPECT_EQ(schema.runtime_arg_names.size(), 2u);
    EXPECT_EQ(schema.common_runtime_arg_names.size(), 1u);
}

TEST(AggregateSpecTypes, CPU_VarargCountsOnAdvancedOptions) {
    // Scalar vararg counts via designated initializers on KernelAdvancedOptions.
    KernelAdvancedOptions adv{
        .num_runtime_varargs = 4,
        .num_common_runtime_varargs = 2,
    };

    EXPECT_EQ(adv.num_runtime_varargs, 4u);
    EXPECT_EQ(adv.num_common_runtime_varargs, 2u);
    EXPECT_TRUE(adv.num_runtime_varargs_per_node.empty());
}

TEST(AggregateSpecTypes, CPU_VarargPerNodeOverrideOnAdvancedOptions) {
    // Per-node override path (advanced): ensure designated-init works.
    using NumVarargsPerNode = Table<Nodes, uint32_t>;
    KernelAdvancedOptions adv{
        .num_runtime_varargs_per_node = NumVarargsPerNode{{NodeCoord{0, 0}, 4}, {NodeCoord{1, 0}, 7}},
    };

    EXPECT_EQ(adv.num_runtime_varargs_per_node.size(), 2u);
    EXPECT_EQ(adv.num_runtime_varargs, 0u);  // scalar left at default in this example
}

TEST(AggregateSpecTypes, CPU_KernelSpecNamedRuntimeArgsDesignatedInitializers) {
    KernelSpec k{
        .unique_id = KernelSpecName{"k"},
        .source = KernelSpec::SourceCode{"void kernel_main() {}"},
        .runtime_arg_schema =
            KernelSpec::RuntimeArgSchema{
                .runtime_arg_names = {"input_ptr"},
            },
        .hw_config = DataMovementHardwareConfig{},
    };
    EXPECT_EQ(k.runtime_arg_schema.runtime_arg_names.size(), 1u);
}

TEST(AggregateSpecTypes, CPU_SemaphoreSpecDesignatedInitializers) {
    // Demonstrates constructing SemaphoreSpec with designated initializers
    SemaphoreSpec sem{
        .unique_id = SemaphoreSpecName{"my_semaphore"},
        .target_nodes = NodeCoord{0, 0},
        .advanced_options = SemaphoreAdvancedOptions{.initial_value = 7},
    };

    EXPECT_EQ(sem.unique_id.get(), "my_semaphore");
    EXPECT_EQ(sem.advanced_options.initial_value, 7u);
}

TEST(AggregateSpecTypes, CPU_ProgramSpecDesignatedInitializers) {
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
                    .hw_config = DataMovementHardwareConfig{},
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

TEST(AggregateSpecTypes, CPU_NestedStructsDesignatedInitializers) {
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

    DataMovementHardwareConfig::DataMovement1XXConfig gen1{
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

}  // namespace
}  // namespace tt::tt_metal::experimental
