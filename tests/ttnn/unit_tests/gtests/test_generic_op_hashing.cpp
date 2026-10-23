// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

////////////////////////////////////////////////////////////
// Blaze-only experimental named args
// Removal is tracked by issue #50953
////////////////////////////////////////////////////////////
//
// Regression tests for the WS5 program-cache HASHING fix in the ttnn generic-op
// path (PR #48704): ttnn::operations::generic::compute_program_descriptor_hash().
//
// This is the deepseek generic-op program-cache key. Before WS5 it hashed only
// the .size() of 3 of the 4 named-RT-arg variants (the per-core-array variant was
// missing entirely) and never the names, so kernels sharing source but differing
// in named args could collide and be served a stale binary. WS5 routes it through
// the shared experimental::blaze::hash_named_args_schema() helper.
//
// These tests lock in the generic-op hasher's contract:
//   * schema-sensitive  -- any schema difference (incl. the previously-missing
//                          per-core-array variant's name/length) => DIFFERENT hash
//   * runtime-value-insensitive  -- runtime-values-only difference => SAME hash
//   * compile-time-value-sensitive  -- compile-time value difference => DIFFERENT hash
//
// Host-only: compute_program_descriptor_hash() is a pure function, so these use
// plain TEST(...) and open no device.

#include <gtest/gtest.h>

#include <cstdint>
#include <filesystem>
#include <string>
#include <utility>
#include <vector>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/experimental/blaze/named_kernel_args.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt_stl/reflection.hpp>

#include "ttnn/tensor/shape/shape.hpp"
#include "ttnn/tensor/tensor_spec.hpp"

namespace ttnn::operations::generic {
// Defined in ttnn/cpp/ttnn/operations/generic/device/generic_op_device_operation.cpp.
// Also bound to Python as `compute_program_descriptor_hash` (generic_op_nanobind.cpp).
ttsl::hash::hash_t compute_program_descriptor_hash(const tt::tt_metal::ProgramDescriptor& program_descriptor);
}  // namespace ttnn::operations::generic

namespace genop_named_args_hash_test {

// Scoped to this namespace only (does NOT leak into the shared unity-build TU).
using namespace tt::tt_metal;

const CoreCoord kCore0{0, 0};
const CoreCoord kCore1{1, 0};

// Wrap named args in an otherwise-fixed single-kernel ProgramDescriptor and hash it
// via the ttnn generic-op hasher under test. Everything except blaze_named_args is
// held constant, so only the compile-time names/values and runtime-arg schema vary.
ttsl::hash::hash_t program_hash(const experimental::blaze::NamedKernelArgs& args) {
    KernelDescriptor kernel = {
        .kernel_source = "tests/tt_metal/tt_metal/test_kernels/misc/blaze_named_runtime_args_kernel.cpp",
        .core_ranges = CoreRangeSet(CoreRange(kCore0)),
        .blaze_named_args = args,
        .config = DataMovementConfigDescriptor{},
    };
    ProgramDescriptor descriptor{.kernels = {kernel}};
    return ttnn::operations::generic::compute_program_descriptor_hash(descriptor);
}

experimental::blaze::NamedKernelArgs common_scalar(std::string name, uint32_t value) {
    return experimental::blaze::NamedKernelArgs{.named_common_runtime_args = {{std::move(name), value}}};
}
experimental::blaze::NamedKernelArgs per_core_array(
    std::string name, std::vector<std::pair<CoreCoord, std::vector<uint32_t>>> core_values) {
    return experimental::blaze::NamedKernelArgs{
        .named_per_core_runtime_arg_arrays = {{std::move(name), std::move(core_values)}}};
}

}  // namespace genop_named_args_hash_test

TEST(GenericOpNamedArgsHash, CompileTimeValueChangesHash) {
    using namespace genop_named_args_hash_test;
    experimental::blaze::NamedKernelArgs a{.named_compile_time_args = {{"kernel.value", 1}}};
    experimental::blaze::NamedKernelArgs b{.named_compile_time_args = {{"kernel.value", 2}}};
    EXPECT_NE(program_hash(a), program_hash(b));
}

TEST(GenericOpNamedArgsHash, SchemaDifferenceChangesHash) {
    using namespace genop_named_args_hash_test;
    EXPECT_NE(program_hash(common_scalar("a.x", 0)), program_hash(common_scalar("a.y", 0)))
        << "A named-arg schema difference must change the generic-op program hash";
}

TEST(GenericOpNamedArgsHash, PerCoreArrayVariantChangesHash) {
    // The specific bug WS5 fixed: the per-core-array variant was absent from the old
    // hasher (name never hashed, length never hashed). These two descriptors differ
    // ONLY in that variant's field name and array width, and the old code collided them.
    using namespace genop_named_args_hash_test;
    EXPECT_NE(
        program_hash(per_core_array("a.x", {{kCore0, {0, 0}}})),
        program_hash(per_core_array("a.y", {{kCore0, {0, 0, 0}}})))
        << "A per-core-array schema difference (name/length) must change the generic-op program hash";
}

TEST(GenericOpNamedArgsHash, ValueOnlyDifferenceKeepsHash) {
    // Identical schema across all four variants, differing only in runtime values
    // (and per-core core count). The generic-op program hash must be unchanged.
    using namespace genop_named_args_hash_test;
    experimental::blaze::NamedKernelArgs a{
        .named_common_runtime_args = {{"a.x", 1}},
        .named_per_core_runtime_args = {{"a.y", {{kCore0, 2}}}},
        .named_common_runtime_arg_arrays = {{"a.z", {3, 4}}},
        .named_per_core_runtime_arg_arrays = {{"a.w", {{kCore0, {5, 6, 7}}}}},
    };
    experimental::blaze::NamedKernelArgs b{
        .named_common_runtime_args = {{"a.x", 111}},
        .named_per_core_runtime_args = {{"a.y", {{kCore0, 222}, {kCore1, 333}}}},
        .named_common_runtime_arg_arrays = {{"a.z", {444, 555}}},
        .named_per_core_runtime_arg_arrays = {{"a.w", {{kCore0, {6, 6, 6}}, {kCore1, {7, 7, 7}}}}},
    };
    EXPECT_EQ(program_hash(a), program_hash(b))
        << "Named-arg values (and per-core core count) must not change the generic-op program hash";
}

// ------------------------------------------------------------------------------ ProgramSpec hash
//
// compute_program_spec_hash() is the spec half of the program cache key the Metal 2.0 overload of
// generic_op uses (the other half is each bound tensor's spec under its parameter's relaxations).
// Host-only, like the descriptor tests above. The split pairs are the collisions a plain
// element-by-element fold of adjacent lists produced: every list is now folded with its own tag
// and length.

namespace ttnn::operations::generic {
ttsl::hash::hash_t compute_program_spec_hash(const tt::tt_metal::experimental::ProgramSpec& spec);
}  // namespace ttnn::operations::generic

namespace genop_spec_hash_test {

namespace m2 = tt::tt_metal::experimental;
using tt::tt_metal::CoreCoord;

const CoreCoord kCore0{0, 0};

tt::tt_metal::TensorSpec tensor_spec(const ttnn::Shape& shape) {
    return tt::tt_metal::TensorSpec(
        shape,
        tt::tt_metal::TensorLayout(
            tt::tt_metal::DataType::BFLOAT16,
            tt::tt_metal::PageConfig(tt::tt_metal::Layout::TILE),
            tt::tt_metal::MemoryConfig{}));
}

// One data-movement kernel on one core, bound to one tensor parameter; everything else default.
m2::ProgramSpec spec() {
    return m2::ProgramSpec{
        .name = "spec",
        .kernels = {m2::KernelSpec{
            .unique_id = m2::KernelSpecName("k"),
            .source =
                std::filesystem::path("tests/tt_metal/tt_metal/test_kernels/misc/blaze_named_runtime_args_kernel.cpp"),
            .tensor_bindings = {{m2::TensorParamName("t"), "t"}},
            .runtime_arg_schema = {.runtime_arg_names = {"x"}},
            .hw_config = m2::CreateReaderDataMovementConfig(),
        }},
        .tensor_parameters = {m2::TensorParameter{
            .unique_id = m2::TensorParamName("t"), .spec = tensor_spec(ttnn::Shape{1, 1, 32, 32})}},
        .work_units = {m2::WorkUnitSpec{.name = "main", .kernels = {m2::KernelSpecName("k")}, .target_nodes = kCore0}},
    };
}

ttsl::hash::hash_t spec_hash(const m2::ProgramSpec& spec) {
    return ttnn::operations::generic::compute_program_spec_hash(spec);
}

}  // namespace genop_spec_hash_test

TEST(GenericOpSpecHash, NamedRuntimeArgVsCommonRuntimeArgSplitDiffers) {
    using namespace genop_spec_hash_test;
    auto a = spec();
    a.kernels[0].runtime_arg_schema = {.runtime_arg_names = {"x", "y"}};
    auto b = spec();
    b.kernels[0].runtime_arg_schema = {.runtime_arg_names = {"x"}, .common_runtime_arg_names = {"y"}};
    EXPECT_NE(spec_hash(a), spec_hash(b))
        << "A name moved from the per-node to the common slot layout is another program";
}

TEST(GenericOpSpecHash, SemaphoreBindingVsScratchpadBindingDiffers) {
    using namespace genop_spec_hash_test;
    auto a = spec();
    a.semaphores = {m2::SemaphoreSpec{.unique_id = m2::SemaphoreSpecName("s"), .target_nodes = kCore0}};
    a.scratchpads = {m2::ScratchpadSpec{.unique_id = m2::ScratchpadSpecName("s"), .size_per_node = 64}};
    auto b = a;
    a.kernels[0].semaphore_bindings = {{m2::SemaphoreSpecName("s"), "x"}};
    b.kernels[0].scratchpad_bindings = {{m2::ScratchpadSpecName("s"), "x"}};
    EXPECT_NE(spec_hash(a), spec_hash(b)) << "Same-named semaphore and scratchpad bindings are different programs";
}

TEST(GenericOpSpecHash, TensorBindingSequenceSplitDiffers) {
    using namespace genop_spec_hash_test;
    auto a = spec();
    a.kernels[0].advanced_options.tensor_binding_sequences = {{.sequence_name = "s1", .members = {"m1", "m2"}}};
    auto b = spec();
    b.kernels[0].advanced_options.tensor_binding_sequences = {
        {.sequence_name = "s1", .members = {"m1"}}, {.sequence_name = "m2", .members = {}}};
    EXPECT_NE(spec_hash(a), spec_hash(b));
}

TEST(GenericOpSpecHash, PrefetcherPipeBindingSplitDiffers) {
    using namespace genop_spec_hash_test;
    auto a = spec();
    a.kernels[0].advanced_options.prefetcher_pipe_bindings = {
        {.pipe_parameter_names = {m2::PrefetcherPipeParamName("x"), m2::PrefetcherPipeParamName("y")},
         .accessor_name = "p"}};
    auto b = spec();
    b.kernels[0].advanced_options.prefetcher_pipe_bindings = {
        {.pipe_parameter_names = {m2::PrefetcherPipeParamName("x")}, .accessor_name = "p"},
        {.pipe_parameter_names = {}, .accessor_name = "y"}};
    EXPECT_NE(spec_hash(a), spec_hash(b));
}

TEST(GenericOpSpecHash, DebugNamesAreNotKeyed) {
    using namespace genop_spec_hash_test;
    auto a = spec();
    auto b = spec();
    b.name = "another";
    b.work_units[0].name = "another";
    EXPECT_EQ(spec_hash(a), spec_hash(b)) << "ProgramSpec::name and WorkUnitSpec::name are debug/messaging only";
}

TEST(GenericOpSpecHash, ParameterShapeIsKeyedOnlyWhereTheRelaxationLeavesItPertinent) {
    using namespace genop_spec_hash_test;
    auto a = spec();
    auto b = spec();
    b.tensor_parameters[0].spec = tensor_spec(ttnn::Shape{1, 1, 64, 64});
    EXPECT_NE(spec_hash(a), spec_hash(b)) << "Without a relaxation the declared shape is part of the program";
    a.tensor_parameters[0].relaxations.dynamic_tensor_shape = true;
    b.tensor_parameters[0].relaxations.dynamic_tensor_shape = true;
    EXPECT_EQ(spec_hash(a), spec_hash(b)) << "Under dynamic_tensor_shape one program serves every shape";
}
