// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// ttnn::merge_per_device_program_descriptors: host-only, plain TEST(...), no device.

#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

#include <tt-metalium/program_descriptors.hpp>

#include "ttnn/per_device_program_merge.hpp"

namespace {

using tt::tt_metal::CoreCoord;
using tt::tt_metal::CoreRange;
using tt::tt_metal::CoreRangeSet;
using tt::tt_metal::KernelDescriptor;
using tt::tt_metal::ProgramDescriptor;
using tt::tt_metal::SemaphoreDescriptor;

// One data-movement kernel on `cores` with `args` on every core and `common` common args.
ProgramDescriptor program(
    const std::vector<CoreCoord>& cores,
    const std::vector<uint32_t>& args,
    const std::vector<uint32_t>& common,
    uint32_t compile_time_arg = 7) {
    ProgramDescriptor desc;
    KernelDescriptor kernel;
    kernel.kernel_source = "kernel.cpp";
    std::vector<CoreRange> ranges;
    for (const auto& core : cores) {
        ranges.emplace_back(core, core);
        kernel.runtime_args.emplace_back(core, args);
    }
    kernel.core_ranges = CoreRangeSet(ranges);
    kernel.compile_time_args = {compile_time_arg};
    kernel.common_runtime_args = common;
    kernel.config = tt::tt_metal::DataMovementConfigDescriptor{};
    desc.kernels.push_back(std::move(kernel));
    desc.semaphores.push_back(SemaphoreDescriptor{.id = 0, .core_ranges = CoreRangeSet(ranges), .initial_value = 0});
    return desc;
}

TEST(PerDeviceProgramMerge, DifferingWordsBecomeRuns) {
    const std::vector<CoreCoord> cores{{0, 0}};
    // Words 1 and 3 differ; word 2 is the same on both devices and stays the program's own.
    const auto merged = ttnn::merge_per_device_program_descriptors(
        {program(cores, {5, 10, 20, 30}, {1, 2}), program(cores, {5, 11, 20, 31}, {1, 3})});
    ASSERT_TRUE(merged.has_value());
    ASSERT_EQ(merged->per_device_runtime_args.size(), 3u);
    const auto& a = merged->per_device_runtime_args[0];
    EXPECT_EQ(a.core, CoreCoord(0, 0));
    EXPECT_EQ(a.first_word, 1u);
    EXPECT_EQ(a.num_words, 1u);
    EXPECT_EQ(a.values_in_range_order, (std::vector<uint32_t>{10, 11}));
    const auto& b = merged->per_device_runtime_args[1];
    EXPECT_EQ(b.first_word, 3u);
    EXPECT_EQ(b.values_in_range_order, (std::vector<uint32_t>{30, 31}));
    const auto& c = merged->per_device_runtime_args[2];
    EXPECT_FALSE(c.core.has_value());
    EXPECT_EQ(c.first_word, 1u);
    EXPECT_EQ(c.values_in_range_order, (std::vector<uint32_t>{2, 3}));
    // The program's own args are the first device's.
    EXPECT_EQ(merged->descriptor.kernels[0].runtime_args.at(0).second, (std::vector<uint32_t>{5, 10, 20, 30}));
}

TEST(PerDeviceProgramMerge, IdenticalProgramsHaveNoFields) {
    const std::vector<CoreCoord> cores{{0, 0}, {1, 0}};
    const auto merged = ttnn::merge_per_device_program_descriptors(
        {program(cores, {1, 2}, {3}), program(cores, {1, 2}, {3}), program(cores, {1, 2}, {3})});
    ASSERT_TRUE(merged.has_value());
    EXPECT_TRUE(merged->per_device_runtime_args.empty());
}

TEST(PerDeviceProgramMerge, RefusesDifferentCompileTimeArgs) {
    const std::vector<CoreCoord> cores{{0, 0}};
    EXPECT_FALSE(ttnn::merge_per_device_program_descriptors(
                     {program(cores, {1}, {}, /*compile_time_arg=*/7), program(cores, {1}, {}, 8)})
                     .has_value());
}

TEST(PerDeviceProgramMerge, RefusesDifferentCoresWithoutIdleArgs) {
    EXPECT_FALSE(ttnn::merge_per_device_program_descriptors({program({{0, 0}}, {1}, {}), program({{1, 0}}, {1}, {})})
                     .has_value());
}

TEST(PerDeviceProgramMerge, UnionOfCoresWithIdleArgs) {
    ttnn::ProgramMergeOptions options;
    options.idle_runtime_args = [](size_t, const CoreCoord&) { return std::vector<uint32_t>{0}; };
    const auto merged =
        ttnn::merge_per_device_program_descriptors({program({{0, 0}}, {4}, {}), program({{1, 0}}, {6}, {})}, options);
    ASSERT_TRUE(merged.has_value());
    const auto& kernel = merged->descriptor.kernels[0];
    EXPECT_TRUE(kernel.core_ranges.contains(CoreCoord(0, 0)));
    EXPECT_TRUE(kernel.core_ranges.contains(CoreCoord(1, 0)));
    // Core (0,0): {4} on device 0, idle {0} on device 1; core (1,0): idle on device 0, {6} on device 1.
    ASSERT_EQ(merged->per_device_runtime_args.size(), 2u);
    EXPECT_EQ(merged->per_device_runtime_args[0].values_in_range_order, (std::vector<uint32_t>{4, 0}));
    EXPECT_EQ(merged->per_device_runtime_args[1].values_in_range_order, (std::vector<uint32_t>{0, 6}));
    // The semaphore covers both cores.
    ASSERT_EQ(merged->descriptor.semaphores.size(), 1u);
    EXPECT_TRUE(merged->descriptor.semaphores[0].core_ranges.contains(CoreCoord(1, 0)));
}

TEST(PerDeviceProgramMerge, PaddingIsOptIn) {
    const std::vector<CoreCoord> cores{{0, 0}};
    const std::vector<ProgramDescriptor> descriptors{program(cores, {2, 9, 9}, {}), program(cores, {1, 9}, {})};
    EXPECT_FALSE(ttnn::merge_per_device_program_descriptors(descriptors).has_value());
    ttnn::ProgramMergeOptions options;
    options.pad_runtime_args = true;
    const auto merged = ttnn::merge_per_device_program_descriptors(descriptors, options);
    ASSERT_TRUE(merged.has_value());
    // Word 0 differs; word 2 is 9 vs the zero padding.
    ASSERT_EQ(merged->per_device_runtime_args.size(), 2u);
    EXPECT_EQ(merged->per_device_runtime_args[1].first_word, 2u);
    EXPECT_EQ(merged->per_device_runtime_args[1].values_in_range_order, (std::vector<uint32_t>{9, 0}));
}

TEST(PerDeviceProgramMerge, DeclaredWordsArePerDevice) {
    const std::vector<CoreCoord> cores{{0, 0}};
    ttnn::ProgramMergeOptions options;
    options.per_device_words = {{.kernel_index = 0, .core = std::nullopt, .first_word = 1, .num_words = 2}};
    const auto merged = ttnn::merge_per_device_program_descriptors(
        {program(cores, {1}, {5, 6, 7}), program(cores, {1}, {5, 6, 7})}, options);
    ASSERT_TRUE(merged.has_value());
    ASSERT_EQ(merged->per_device_runtime_args.size(), 1u);
    const auto& field = merged->per_device_runtime_args[0];
    EXPECT_FALSE(field.core.has_value());
    EXPECT_EQ(field.first_word, 1u);
    EXPECT_EQ(field.num_words, 2u);
    EXPECT_EQ(field.values_in_range_order, (std::vector<uint32_t>{6, 7, 6, 7}));
}

TEST(PerDeviceProgramMerge, RefusesPerDeviceWordUnderBufferBinding) {
    const std::vector<CoreCoord> cores{{0, 0}};
    auto a = program(cores, {1, 2}, {});
    auto b = program(cores, {1, 3}, {});
    for (auto* d : {&a, &b}) {
        d->kernels[0].buffer_bindings.push_back(
            tt::tt_metal::BufferBinding{.core = {0, 0}, .arg_idx = 1, .buffer = nullptr});
    }
    EXPECT_FALSE(ttnn::merge_per_device_program_descriptors({a, b}).has_value());
}

TEST(PerDeviceProgramMerge, RefusesPartialBufferBinding) {
    const std::vector<CoreCoord> cores{{0, 0}};
    auto a = program(cores, {1, 2}, {});
    const auto b = program(cores, {1, 2}, {});
    a.kernels[0].buffer_bindings.push_back(
        tt::tt_metal::BufferBinding{.core = {0, 0}, .arg_idx = 1, .buffer = nullptr});
    EXPECT_FALSE(ttnn::merge_per_device_program_descriptors({a, b}).has_value());
}

TEST(PerDeviceProgramMerge, CustomHashDoesNotHideDifferences) {
    const std::vector<CoreCoord> cores{{0, 0}};
    auto a = program(cores, {1}, {}, /*compile_time_arg=*/7);
    auto b = program(cores, {1}, {}, 8);
    a.custom_program_hash = b.custom_program_hash = 42;
    EXPECT_FALSE(ttnn::merge_per_device_program_descriptors({a, b}).has_value());
}

TEST(PerDeviceProgramMerge, RefusesNamedArgs) {
    const std::vector<CoreCoord> cores{{0, 0}};
    auto a = program(cores, {1}, {});
    auto b = program(cores, {1}, {});
    a.kernels[0].blaze_named_args.named_common_runtime_args.push_back({.name = "demo.num_tiles", .value = 1});
    b.kernels[0].blaze_named_args.named_common_runtime_args.push_back({.name = "demo.num_tiles", .value = 2});
    EXPECT_FALSE(ttnn::merge_per_device_program_descriptors({a, b}).has_value());
}

TEST(PerDeviceProgramMerge, RefusesKernelsSwappingCores) {
    ttnn::ProgramMergeOptions options;
    options.idle_runtime_args = [](size_t, const CoreCoord&) { return std::vector<uint32_t>{0}; };
    // Two kernels on disjoint cores, swapped on the second device: their union would share both cores.
    auto two_kernels = [](CoreCoord first, CoreCoord second) {
        auto desc = program({first}, {1}, {});
        auto other = program({second}, {1}, {});
        desc.kernels.push_back(other.kernels[0]);
        desc.semaphores.clear();
        return desc;
    };
    EXPECT_FALSE(
        ttnn::merge_per_device_program_descriptors({two_kernels({0, 0}, {1, 0}), two_kernels({1, 0}, {0, 0})}, options)
            .has_value());
}

TEST(PerDeviceProgramMerge, RefusesCircularBuffersSwappingCores) {
    const std::vector<CoreCoord> cores{{0, 0}, {1, 0}};
    auto with_cbs = [&](CoreCoord first, CoreCoord second) {
        auto desc = program(cores, {1}, {});
        for (const auto& [core, size] : {std::pair{first, 1024u}, std::pair{second, 2048u}}) {
            tt::tt_metal::CBDescriptor cb;
            cb.total_size = size;
            cb.core_ranges = CoreRangeSet(CoreRange(core, core));
            cb.format_descriptors.push_back(tt::tt_metal::CBFormatDescriptor{
                .buffer_index = 3, .data_format = tt::DataFormat::Float16_b, .page_size = 1024});
            desc.cbs.push_back(cb);
        }
        return desc;
    };
    EXPECT_FALSE(
        ttnn::merge_per_device_program_descriptors({with_cbs({0, 0}, {1, 0}), with_cbs({1, 0}, {0, 0})}).has_value());
}

TEST(PerDeviceProgramMerge, RefusesBindingOnIdleCore) {
    ttnn::ProgramMergeOptions options;
    options.idle_runtime_args = [](size_t, const CoreCoord&) { return std::vector<uint32_t>{0, 0}; };
    auto a = program({{0, 0}}, {1, 2}, {});
    const auto b = program({{1, 0}}, {1, 2}, {});
    a.kernels[0].buffer_bindings.push_back(
        tt::tt_metal::BufferBinding{.core = {0, 0}, .arg_idx = 1, .buffer = nullptr});
    EXPECT_FALSE(ttnn::merge_per_device_program_descriptors({a, b}, options).has_value());
}

TEST(PerDeviceProgramMerge, RefusesLocalAndRemoteCircularBuffersSwappingCores) {
    const std::vector<CoreCoord> cores{{0, 0}, {1, 0}};
    auto with_cbs = [&](CoreCoord local_core, CoreCoord remote_core) {
        auto desc = program(cores, {1}, {});
        tt::tt_metal::CBDescriptor local;
        local.total_size = 1024;
        local.core_ranges = CoreRangeSet(CoreRange(local_core, local_core));
        local.format_descriptors.push_back(tt::tt_metal::CBFormatDescriptor{
            .buffer_index = 31, .data_format = tt::DataFormat::Float16_b, .page_size = 1024});
        tt::tt_metal::CBDescriptor remote;
        remote.total_size = 1024;
        remote.core_ranges = CoreRangeSet(CoreRange(remote_core, remote_core));
        remote.remote_format_descriptors.push_back(tt::tt_metal::CBFormatDescriptor{
            .buffer_index = 31, .data_format = tt::DataFormat::Float16_b, .page_size = 1024});
        desc.cbs.push_back(local);
        desc.cbs.push_back(remote);
        return desc;
    };
    EXPECT_FALSE(
        ttnn::merge_per_device_program_descriptors({with_cbs({0, 0}, {1, 0}), with_cbs({1, 0}, {0, 0})}).has_value());
}

TEST(PerDeviceProgramMerge, DeclaredWordsKeepTheirBounds) {
    const std::vector<CoreCoord> cores{{0, 0}};
    ttnn::ProgramMergeOptions options;
    options.per_device_words = {{.kernel_index = 0, .core = std::nullopt, .first_word = 1, .num_words = 1}};
    // Word 2 differs and is next to the declared word 1: two fields, the declared one with its own bounds.
    const auto merged = ttnn::merge_per_device_program_descriptors(
        {program(cores, {1}, {5, 6, 7}), program(cores, {1}, {5, 6, 8})}, options);
    ASSERT_TRUE(merged.has_value());
    ASSERT_EQ(merged->per_device_runtime_args.size(), 2u);
    EXPECT_EQ(merged->per_device_runtime_args[0].first_word, 1u);
    EXPECT_EQ(merged->per_device_runtime_args[0].num_words, 1u);
    EXPECT_EQ(merged->per_device_runtime_args[1].first_word, 2u);
    EXPECT_EQ(merged->per_device_runtime_args[1].values_in_range_order, (std::vector<uint32_t>{7, 8}));
}

}  // namespace
