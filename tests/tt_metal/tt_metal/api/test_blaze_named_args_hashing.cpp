// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

////////////////////////////////////////////////////////////
// Blaze-only experimental named args
// Removal is tracked by issue #50953
////////////////////////////////////////////////////////////
//
// Regression tests for the WS5 program-cache HASHING fixes (PR #48704).
//
// The JIT-generated named_args_generated.h depends on the named-RT-arg SCHEMA
// (names + array lengths + dispatch kind common-vs-per-core + order + count).
// That schema therefore MUST be part of every program-cache key, or two kernels
// that share source but differ in named args collide and a stale binary is served.
// Runtime VALUES, by contrast, are written per enqueue and never affect the
// generated header, so hashing them would only cause needless cache misses --
// they are deliberately EXCLUDED from the hash.
//
// These tests lock in BOTH directions:
//   * schema-sensitive  -- any schema difference => DIFFERENT hash
//   * value-insensitive  -- values-only difference => SAME hash
//
// Tier 1 (this file) is host-only: it exercises
//   (a) experimental::blaze::hash_named_args_schema() directly, and
//   (b) hash_kernel_descriptor() via the public std::hash<ProgramDescriptor>.
// Both are pure functions -- no device is opened, so these use plain TEST(...).

#include <gtest/gtest.h>

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/experimental/blaze/named_kernel_args.hpp>
#include <tt-metalium/program_descriptors.hpp>

using namespace tt::tt_metal;

namespace {

using experimental::blaze::NamedKernelArgs;

// ---- helpers (prefixed to stay unique inside the unity-build translation unit) ----

// Hash a named-arg schema directly via the shared helper under test.
std::uint64_t blaze_hash_schema(const NamedKernelArgs& args) {
    return experimental::blaze::hash_named_args_schema(args);
}

// Wrap named args in an otherwise-fixed single-kernel ProgramDescriptor and return
// its program-cache hash via the public std::hash<ProgramDescriptor> specialization
// (which folds in hash_kernel_descriptor per kernel). Everything except
// blaze_named_args is held constant, so the only hash input that varies between
// calls is hash_named_args_schema(kernel.blaze_named_args). Note: named-arg values
// are NOT merged into kernel.runtime_args here (that merge happens only during
// Program construction, not descriptor hashing), so value-only differences leave
// every hashed field untouched.
std::uint64_t blaze_program_hash(const NamedKernelArgs& args) {
    KernelDescriptor kernel = {
        .kernel_source = "tests/tt_metal/tt_metal/test_kernels/misc/blaze_named_runtime_args_kernel.cpp",
        .core_ranges = CoreRangeSet(CoreRange(CoreCoord{0, 0})),
        .blaze_named_args = args,
        .config = DataMovementConfigDescriptor{},
    };
    ProgramDescriptor descriptor{.kernels = {kernel}};
    return std::hash<tt::tt_metal::ProgramDescriptor>{}(descriptor);
}

// Schema builders for the four named-RT-arg variants. Each puts a single named arg
// into exactly one section, so the section it lands in encodes its dispatch
// (common vs per-core) and kind (scalar vs array).
NamedKernelArgs blaze_common_scalar(std::string name, uint32_t value) {
    return NamedKernelArgs{.named_common_runtime_args = {{std::move(name), value}}};
}
NamedKernelArgs blaze_per_core_scalar(std::string name, std::vector<std::pair<CoreCoord, uint32_t>> core_values) {
    return NamedKernelArgs{.named_per_core_runtime_args = {{std::move(name), std::move(core_values)}}};
}
NamedKernelArgs blaze_common_array(std::string name, std::vector<uint32_t> values) {
    return NamedKernelArgs{.named_common_runtime_arg_arrays = {{std::move(name), std::move(values)}}};
}
NamedKernelArgs blaze_per_core_array(
    std::string name, std::vector<std::pair<CoreCoord, std::vector<uint32_t>>> core_values) {
    return NamedKernelArgs{.named_per_core_runtime_arg_arrays = {{std::move(name), std::move(core_values)}}};
}

const CoreCoord kCore0{0, 0};
const CoreCoord kCore1{1, 0};

}  // namespace

// ============================================================================
// Tier 1a -- hash_named_args_schema() directly
// ============================================================================

TEST(NamedArgsHashSchema, EmptyEqualsEmpty) {
    EXPECT_EQ(blaze_hash_schema(NamedKernelArgs{}), blaze_hash_schema(NamedKernelArgs{}))
        << "Two empty schemas must hash identically";
}

// --- value-insensitivity: same schema, different runtime values => SAME hash ---

TEST(NamedArgsHashSchema, CommonScalarValueInsensitive) {
    EXPECT_EQ(blaze_hash_schema(blaze_common_scalar("a.x", 1)), blaze_hash_schema(blaze_common_scalar("a.x", 999)))
        << "Common-scalar value is runtime data and must not affect the schema hash";
}

TEST(NamedArgsHashSchema, PerCoreScalarValueInsensitive) {
    // Same name, different per-core values AND a different core set / core count.
    // None of the value data (nor the number of cores it targets) is part of the schema.
    EXPECT_EQ(
        blaze_hash_schema(blaze_per_core_scalar("a.x", {{kCore0, 1}})),
        blaze_hash_schema(blaze_per_core_scalar("a.x", {{kCore0, 7}, {kCore1, 42}})))
        << "Per-core-scalar values and core count are runtime data and must not affect the schema hash";
}

TEST(NamedArgsHashSchema, CommonArrayValueInsensitive) {
    EXPECT_EQ(
        blaze_hash_schema(blaze_common_array("a.x", {1, 2, 3})),
        blaze_hash_schema(blaze_common_array("a.x", {9, 8, 7})))
        << "Common-array values are runtime data; only the array LENGTH is schema";
}

TEST(NamedArgsHashSchema, PerCoreArrayValueInsensitive) {
    // Same name, same per-core array WIDTH (3), different values and different core count.
    EXPECT_EQ(
        blaze_hash_schema(blaze_per_core_array("a.x", {{kCore0, {1, 2, 3}}})),
        blaze_hash_schema(blaze_per_core_array("a.x", {{kCore0, {9, 9, 9}}, {kCore1, {5, 5, 5}}})))
        << "Per-core-array values and core count are runtime data; only the array WIDTH is schema";
}

// --- schema-sensitivity: any schema difference => DIFFERENT hash ---

TEST(NamedArgsHashSchema, DifferentFieldNameDiffers) {
    EXPECT_NE(blaze_hash_schema(blaze_common_scalar("a.x", 0)), blaze_hash_schema(blaze_common_scalar("a.y", 0)))
        << "Different field name => different generated header => different hash";
}

TEST(NamedArgsHashSchema, DifferentNamespaceDiffers) {
    EXPECT_NE(blaze_hash_schema(blaze_common_scalar("a.x", 0)), blaze_hash_schema(blaze_common_scalar("b.x", 0)))
        << "Different namespace => different generated header => different hash";
}

TEST(NamedArgsHashSchema, CommonVsPerCoreScalarDiffers) {
    // Same name "a.x", but common-dispatch vs per-core-dispatch => different Arg descriptor.
    EXPECT_NE(
        blaze_hash_schema(blaze_common_scalar("a.x", 0)),
        blaze_hash_schema(blaze_per_core_scalar("a.x", {{kCore0, 0}})))
        << "Dispatch kind (common vs per-core) is schema and must change the hash";
}

TEST(NamedArgsHashSchema, ScalarVsArrayKindDiffers) {
    // Same name "a.x", common dispatch, but scalar (Arg) vs length-1 array (ArrayArg).
    EXPECT_NE(blaze_hash_schema(blaze_common_scalar("a.x", 0)), blaze_hash_schema(blaze_common_array("a.x", {0})))
        << "Scalar vs array kind is schema and must change the hash even at length 1";
}

TEST(NamedArgsHashSchema, CommonArrayLengthDiffers) {
    EXPECT_NE(
        blaze_hash_schema(blaze_common_array("a.x", {0, 0})), blaze_hash_schema(blaze_common_array("a.x", {0, 0, 0})))
        << "Common-array length is baked into the header and must change the hash";
}

TEST(NamedArgsHashSchema, PerCoreArrayLengthDiffers) {
    // Regression: the per-core-array variant was entirely absent from the old hashers.
    EXPECT_NE(
        blaze_hash_schema(blaze_per_core_array("a.x", {{kCore0, {0, 0}}})),
        blaze_hash_schema(blaze_per_core_array("a.x", {{kCore0, {0, 0, 0}}})))
        << "Per-core-array width is schema and must change the hash";
}

TEST(NamedArgsHashSchema, PerCoreArrayNameDiffers) {
    // Regression: names in the per-core-array variant were never hashed before WS5.
    EXPECT_NE(
        blaze_hash_schema(blaze_per_core_array("a.x", {{kCore0, {0, 0}}})),
        blaze_hash_schema(blaze_per_core_array("a.y", {{kCore0, {0, 0}}})))
        << "Per-core-array field name is schema and must change the hash";
}

TEST(NamedArgsHashSchema, OrderSwapDiffers) {
    // Two common scalars in swapped order. Order determines the assigned RT-arg indices,
    // so [a.x, b.y] and [b.y, a.x] generate different headers.
    NamedKernelArgs ab{.named_common_runtime_args = {{"a.x", 0}, {"b.y", 0}}};
    NamedKernelArgs ba{.named_common_runtime_args = {{"b.y", 0}, {"a.x", 0}}};
    EXPECT_NE(blaze_hash_schema(ab), blaze_hash_schema(ba))
        << "Order within a section shifts indices => different hash";
}

TEST(NamedArgsHashSchema, CountCollisionGuard) {
    // Two names ["a","b"] must not collide with a single concatenated name ["ab"].
    // Per-section counts are hashed first precisely to prevent this.
    NamedKernelArgs two{.named_common_runtime_args = {{"a", 0}, {"b", 0}}};
    NamedKernelArgs one{.named_common_runtime_args = {{"ab", 0}}};
    EXPECT_NE(blaze_hash_schema(two), blaze_hash_schema(one)) << "['a','b'] must not collide with ['ab']";
}

// ============================================================================
// Tier 1b -- hash_kernel_descriptor via public std::hash<ProgramDescriptor>
// ============================================================================

TEST(NamedArgsHashProgramDescriptor, SchemaDifferenceChangesProgramHash) {
    // Two descriptors identical except for a named-arg field name.
    EXPECT_NE(blaze_program_hash(blaze_common_scalar("a.x", 0)), blaze_program_hash(blaze_common_scalar("a.y", 0)))
        << "A named-arg schema difference must propagate into the program hash";
}

TEST(NamedArgsHashProgramDescriptor, PerCoreArrayLengthChangesProgramHash) {
    // Regression: the per-core-array variant was missing from the descriptor hasher
    // before WS5, so a change here would previously have collided.
    EXPECT_NE(
        blaze_program_hash(blaze_per_core_array("a.x", {{kCore0, {0, 0}}})),
        blaze_program_hash(blaze_per_core_array("a.x", {{kCore0, {0, 0, 0}}})))
        << "A per-core-array schema difference must propagate into the program hash";
}

TEST(NamedArgsHashProgramDescriptor, ValueOnlyDifferenceKeepsProgramHash) {
    // Identical schema across all four variants, differing only in runtime values
    // (and per-core core count). The program hash must be unchanged so the cache hits.
    NamedKernelArgs a{
        .named_common_runtime_args = {{"a.x", 1}},
        .named_per_core_runtime_args = {{"a.y", {{kCore0, 2}}}},
        .named_common_runtime_arg_arrays = {{"a.z", {3, 4}}},
        .named_per_core_runtime_arg_arrays = {{"a.w", {{kCore0, {5, 6, 7}}}}},
    };
    NamedKernelArgs b{
        .named_common_runtime_args = {{"a.x", 111}},
        .named_per_core_runtime_args = {{"a.y", {{kCore0, 222}, {kCore1, 333}}}},
        .named_common_runtime_arg_arrays = {{"a.z", {444, 555}}},
        .named_per_core_runtime_arg_arrays = {{"a.w", {{kCore0, {6, 6, 6}}, {kCore1, {7, 7, 7}}}}},
    };
    EXPECT_EQ(blaze_program_hash(a), blaze_program_hash(b))
        << "Named-arg values (and per-core core count) must not change the program hash";
}
