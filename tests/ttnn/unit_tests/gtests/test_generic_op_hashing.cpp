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
// These tests lock in BOTH directions on the generic-op hasher:
//   * schema-sensitive  -- any schema difference (incl. the previously-missing
//                          per-core-array variant's name/length) => DIFFERENT hash
//   * value-insensitive  -- values-only difference => SAME hash
//
// Host-only: compute_program_descriptor_hash() is a pure function, so these use
// plain TEST(...) and open no device.

#include <gtest/gtest.h>

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/experimental/blaze/named_kernel_args.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt_stl/reflection.hpp>

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
// held constant, so the only varying hash input is the named-arg schema.
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
