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

#include <cstdlib>
#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/experimental/blaze/named_kernel_args.hpp>
#include <tt-metalium/program.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt_stl/reflection.hpp>
#include "impl/program/program_impl.hpp"

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
const CoreCoord kCore2{0, 1};
const CoreCoord kCore3{1, 1};

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

ttsl::hash::hash_t cb_program_hash(uint32_t uniform_address_group) {
    CBDescriptor cb{
        .total_size = 2048,
        .core_ranges = CoreRangeSet(CoreRange(kCore0)),
        .format_descriptors = {{.buffer_index = 0, .data_format = tt::DataFormat::Float16_b, .page_size = 2048}},
        .uniform_address_group = uniform_address_group,
    };
    ProgramDescriptor descriptor{.cbs = {cb}};
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

TEST(GenericOpCircularBufferHash, UniformAddressGroupChangesHash) {
    using namespace genop_named_args_hash_test;
    EXPECT_NE(cb_program_hash(0), cb_program_hash(1))
        << "CB address-group semantics must be part of the generic-op program-cache key";
}

TEST(ProgramDescriptorMerge, RemapsUniformAddressGroupsPerDescriptor) {
    using namespace genop_named_args_hash_test;
    auto make_cb = [](CoreCoord core, uint32_t buffer_index) {
        return CBDescriptor{
            .total_size = 2048,
            .core_ranges = CoreRangeSet(CoreRange(core)),
            .format_descriptors =
                {{.buffer_index = buffer_index, .data_format = tt::DataFormat::Float16_b, .page_size = 2048}},
            .uniform_address_group = 1,
        };
    };
    ProgramDescriptor first{.cbs = {make_cb(kCore0, 0), make_cb(kCore2, 0)}};
    ProgramDescriptor second{.cbs = {make_cb(kCore1, 1), make_cb(kCore3, 1)}};

    auto merged = merge_program_descriptors({first, second});

    ASSERT_EQ(merged.cbs.size(), 4);
    EXPECT_EQ(merged.cbs[0].uniform_address_group, 1);
    EXPECT_EQ(merged.cbs[1].uniform_address_group, 1);
    EXPECT_EQ(merged.cbs[2].uniform_address_group, 2);
    EXPECT_EQ(merged.cbs[3].uniform_address_group, 2);
}

TEST(ProgramDescriptorMerge, RejectsDifferentProgramL1Layouts) {
    using namespace tt::tt_metal;
    ProgramDescriptor uniform_descriptor;
    ProgramDescriptor per_core_descriptor;
    per_core_descriptor.program_l1_layout = ProgramL1Layout::PER_CORE;

    EXPECT_THROW(merge_program_descriptors({uniform_descriptor, per_core_descriptor}), std::exception);
}

TEST(ProgramDescriptorValidation, UniformAddressGroupsRequireCompatibleDisjointStaticDescriptors) {
    using namespace genop_named_args_hash_test;
    auto make_cb = [](CoreCoord core, uint32_t total_size, uint32_t page_size, uint32_t group = 1) {
        return CBDescriptor{
            .total_size = total_size,
            .core_ranges = CoreRangeSet(CoreRange(core)),
            .format_descriptors =
                {{.buffer_index = 0, .data_format = tt::DataFormat::Float16_b, .page_size = page_size}},
            .uniform_address_group = group,
        };
    };

    EXPECT_NO_THROW(([&] {
        ProgramDescriptor descriptor{.cbs = {make_cb(kCore0, 2048, 2048), make_cb(kCore1, 4096, 2048)}};
        Program program(descriptor);
    }()));
    EXPECT_THROW(
        ([&] {
            ProgramDescriptor descriptor{.cbs = {make_cb(kCore0, 2048, 2048)}};
            Program program(descriptor);
        }()),
        std::exception);
    EXPECT_THROW(
        ([&] {
            ProgramDescriptor descriptor{.cbs = {make_cb(kCore0, 2048, 2048), make_cb(kCore0, 4096, 2048)}};
            Program program(descriptor);
        }()),
        std::exception);
    EXPECT_THROW(
        ([&] {
            ProgramDescriptor descriptor{.cbs = {make_cb(kCore0, 2048, 2048), make_cb(kCore1, 4096, 4096)}};
            Program program(descriptor);
        }()),
        std::exception);
}

TEST(ProgramConfiguration, PerCoreProgramLayoutRequiresDescriptorContractAndEnvironment) {
    using namespace tt::tt_metal;
    const char* previous_value = std::getenv("TT_METAL_PER_CORE_PROGRAM_SIZE");
    const std::optional<std::string> saved_value =
        previous_value == nullptr ? std::nullopt : std::make_optional(previous_value);

    unsetenv("TT_METAL_PER_CORE_PROGRAM_SIZE");
    const ProgramDescriptor uniform_descriptor;
    Program default_program;
    EXPECT_FALSE(default_program.impl().uses_per_core_l1_layout());
    EXPECT_FALSE(default_program.impl().uses_per_core_cb_placement());
    const auto uniform_generic_hash = ttnn::operations::generic::compute_program_descriptor_hash(uniform_descriptor);
    const auto uniform_descriptor_hash = std::hash<ProgramDescriptor>{}(uniform_descriptor);

    ProgramDescriptor per_core_descriptor;
    per_core_descriptor.program_l1_layout = ProgramL1Layout::PER_CORE;
    EXPECT_THROW((void)Program{per_core_descriptor}, std::exception);

    setenv("TT_METAL_PER_CORE_PROGRAM_SIZE", "1", /*overwrite=*/1);
    Program uniform_program(uniform_descriptor);
    EXPECT_FALSE(uniform_program.impl().uses_per_core_l1_layout());
    EXPECT_FALSE(uniform_program.impl().uses_per_core_cb_placement());
    EXPECT_EQ(uniform_generic_hash, ttnn::operations::generic::compute_program_descriptor_hash(uniform_descriptor));
    EXPECT_EQ(uniform_descriptor_hash, std::hash<ProgramDescriptor>{}(uniform_descriptor));

    Program per_core_program(per_core_descriptor);
    EXPECT_TRUE(per_core_program.impl().uses_per_core_l1_layout());
    EXPECT_TRUE(per_core_program.impl().uses_per_core_cb_placement());
    EXPECT_NE(uniform_generic_hash, ttnn::operations::generic::compute_program_descriptor_hash(per_core_descriptor));
    EXPECT_NE(uniform_descriptor_hash, std::hash<ProgramDescriptor>{}(per_core_descriptor));

    if (saved_value.has_value()) {
        setenv("TT_METAL_PER_CORE_PROGRAM_SIZE", saved_value->c_str(), /*overwrite=*/1);
    } else {
        unsetenv("TT_METAL_PER_CORE_PROGRAM_SIZE");
    }
}
