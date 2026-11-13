// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// ttnn::merge_per_device_program_descriptors on the host (plain TEST), and an op that opts into the merge, run through
// the device-operation framework on a mesh (PerDeviceProgramMergeOp).

#include <gtest/gtest.h>

#include <cstdint>
#include <optional>
#include <string>
#include <variant>
#include <vector>

#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/workload_descriptor.hpp>

#include "tt_metal/tt_metal/common/multi_device_fixture.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/distributed/api.hpp"
#include "ttnn/mesh_device_operation_adapter.hpp"
#include "ttnn/per_device_program_merge.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace {

using tt::tt_metal::CoreCoord;
using tt::tt_metal::CoreRange;
using tt::tt_metal::CoreRangeSet;
using tt::tt_metal::KernelDescriptor;
using tt::tt_metal::ProgramDescriptor;
using tt::tt_metal::SemaphoreDescriptor;
using ttnn::Tensor;

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

TEST(PerDeviceProgramMerge, SemaphoresOnDifferentCoresAreUnioned) {
    // Same kernel cores, but each device has the semaphore on a different core: the merged program places it on both.
    auto with_semaphore_on = [](const CoreCoord& core) {
        ProgramDescriptor desc = program({{0, 0}}, {1}, {});
        desc.semaphores = {
            SemaphoreDescriptor{.id = 0, .core_ranges = CoreRangeSet(CoreRange(core, core)), .initial_value = 0}};
        return desc;
    };
    const auto merged =
        ttnn::merge_per_device_program_descriptors({with_semaphore_on({0, 0}), with_semaphore_on({1, 0})});
    ASSERT_TRUE(merged.has_value());
    ASSERT_EQ(merged->descriptor.semaphores.size(), 1u);
    EXPECT_TRUE(merged->descriptor.semaphores[0].core_ranges.contains(CoreCoord(0, 0)));
    EXPECT_TRUE(merged->descriptor.semaphores[0].core_ranges.contains(CoreCoord(1, 0)));
}

TEST(PerDeviceProgramMerge, RefusesSemaphoreWithTwoInitialValuesOnOneCore) {
    auto with_initial_value = [](uint32_t initial_value) {
        ProgramDescriptor desc = program({{0, 0}}, {1}, {});
        desc.semaphores[0].initial_value = initial_value;
        return desc;
    };
    EXPECT_FALSE(
        ttnn::merge_per_device_program_descriptors({with_initial_value(0), with_initial_value(1)}).has_value());
}

// ---- an op that opts into the merge, on a mesh ----

constexpr uint32_t kWords = 64;

// Reads the input page, adds this device's value (runtime arg 2, a per-device field once merged) to every word and
// writes the output page. Runtime args 0 and 1 are buffer bindings, patched on a cache hit.
constexpr const char* kAddDeviceValueKernel = R"(
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t input_address = get_arg_val<uint32_t>(0);
    const uint32_t output_address = get_arg_val<uint32_t>(1);
    const uint32_t value = get_arg_val<uint32_t>(2);
    constexpr uint32_t cb = get_compile_time_arg_val(0);
    constexpr uint32_t num_words = get_compile_time_arg_val(1);
    constexpr auto input_args = TensorAccessorArgs<2>();
    constexpr auto output_args = TensorAccessorArgs<input_args.next_compile_time_args_offset()>();
    const auto input = TensorAccessor(input_args, input_address, num_words * sizeof(uint32_t));
    const auto output = TensorAccessor(output_args, output_address, num_words * sizeof(uint32_t));
    const uint32_t l1 = get_write_ptr(cb);
    noc_async_read_page(0, input, l1);
    noc_async_read_barrier();
    auto* words = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l1);
    for (uint32_t i = 0; i < num_words; ++i) {
        words[i] += value;
    }
    noc_async_write_page(0, output, l1);
    noc_async_write_barrier();
}
)";

uint32_t device_value(const ttnn::MeshCoordinate& coord, const ttnn::MeshShape& shape) {
    return 1000 * static_cast<uint32_t>(coord.to_linear_index(shape) + 1);
}

struct PerDeviceProgramMergeOp {
    struct operation_attributes_t {
        uint32_t unused = 0;
    };
    struct tensor_args_t {
        Tensor input;
    };
    using spec_return_value_t = tt::tt_metal::TensorSpec;
    using tensor_return_value_t = Tensor;

    struct Factory {
        static constexpr bool merge_per_device_programs = true;

        static tt::tt_metal::WorkloadDescriptor create_workload_descriptor(
            const operation_attributes_t& /*attrs*/,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& output,
            const ttnn::MeshCoordinateRangeSet& tensor_coords) {
            const CoreCoord core(0, 0);
            const auto mesh_shape = tensor_args.input.device()->shape();
            tt::tt_metal::WorkloadDescriptor workload;
            for (const auto& coord : tensor_coords.coords()) {
                ProgramDescriptor desc;
                desc.cbs.push_back(tt::tt_metal::CBDescriptor{
                    .total_size = kWords * sizeof(uint32_t),
                    .core_ranges = CoreRangeSet(CoreRange(core, core)),
                    .format_descriptors = {tt::tt_metal::CBFormatDescriptor{
                        .buffer_index = 0,
                        .data_format = tt::DataFormat::UInt32,
                        .page_size = kWords * sizeof(uint32_t)}},
                });
                KernelDescriptor kernel;
                kernel.kernel_source = kAddDeviceValueKernel;
                kernel.source_type = KernelDescriptor::SourceType::SOURCE_CODE;
                kernel.core_ranges = CoreRangeSet(CoreRange(core, core));
                kernel.compile_time_args = {0, kWords};
                tt::tt_metal::TensorAccessorArgs(tensor_args.input.buffer()).append_to(kernel.compile_time_args);
                tt::tt_metal::TensorAccessorArgs(output.buffer()).append_to(kernel.compile_time_args);
                kernel.emplace_runtime_args(
                    core, {tensor_args.input.buffer(), output.buffer(), device_value(coord, mesh_shape)});
                kernel.config = tt::tt_metal::ReaderConfigDescriptor{};
                desc.kernels.push_back(std::move(kernel));
                workload.programs.push_back({ttnn::MeshCoordinateRange(coord), std::move(desc)});
            }
            return workload;
        }
    };
    using program_factory_t = std::variant<Factory>;

    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&) {}
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t& tensor_args) {
        return tensor_args.input.tensor_spec();
    }
    static tensor_return_value_t create_output_tensors(
        const operation_attributes_t& attrs, const tensor_args_t& tensor_args) {
        return ttnn::create_device_tensor(compute_output_specs(attrs, tensor_args), tensor_args.input.device());
    }
};

using MergeAdapter = ttnn::device_operation::MeshDeviceOperationAdapter<
    PerDeviceProgramMergeOp>::DescriptorMeshWorkloadAdapter<PerDeviceProgramMergeOp::Factory>;

Tensor replicated_input(ttnn::MeshDevice* mesh_device, uint32_t first) {
    const tt::tt_metal::TensorSpec spec(
        ttnn::Shape({1, 1, 1, kWords}),
        tt::tt_metal::TensorLayout(
            tt::tt_metal::DataType::UINT32,
            tt::tt_metal::PageConfig(tt::tt_metal::Layout::ROW_MAJOR),
            tt::tt_metal::MemoryConfig{}));
    std::vector<uint32_t> values(kWords);
    for (uint32_t i = 0; i < kWords; ++i) {
        values[i] = first + i;
    }
    return Tensor::from_vector(values, spec, mesh_device);
}

void expect_input_plus_device_value(const Tensor& output, uint32_t first, const ttnn::MeshShape& shape) {
    const auto shards = ttnn::distributed::get_device_tensors(output.cpu());
    ASSERT_EQ(shards.size(), shape.mesh_size());
    size_t index = 0;
    for (const auto& coord : ttnn::MeshCoordinateRange(shape)) {
        const auto words = shards[index++].to_vector<uint32_t>();
        ASSERT_EQ(words.size(), kWords);
        for (uint32_t i = 0; i < kWords; ++i) {
            ASSERT_EQ(words[i], first + i + device_value(coord, shape)) << "device " << coord << ", word " << i;
        }
    }
}

using PerDeviceProgramMergeMeshTest = tt::tt_metal::GenericMeshDeviceFixture;

TEST_F(PerDeviceProgramMergeMeshTest, OpRunsAsOneProgramWithPerDeviceValues) {
    auto* mesh_device = mesh_device_.get();
    const auto shape = mesh_device->shape();
    if (shape.mesh_size() < 2) {
        GTEST_SKIP() << "needs at least two devices";
    }
    const ttnn::MeshCoordinateRangeSet coords{ttnn::MeshCoordinateRange(shape)};
    if (!tt::tt_metal::distributed::PerDeviceRuntimeArgsSupported(*mesh_device, ttnn::MeshCoordinateRange(shape))) {
        GTEST_SKIP() << "per-device runtime args are not supported on this mesh";
    }

    // The framework passes MergePerDevicePrograms: one program for the whole mesh. A wrapping factory that calls
    // create_mesh_workload without it keeps one program per device.
    {
        const PerDeviceProgramMergeOp::tensor_args_t args{.input = replicated_input(mesh_device, 0)};
        auto output = PerDeviceProgramMergeOp::create_output_tensors({}, args);
        auto merged = MergeAdapter::create_mesh_workload({}, coords, args, output, ttnn::MergePerDevicePrograms{});
        EXPECT_EQ(merged.workload.get_programs().size(), 1u);
        auto unmerged = MergeAdapter::create_mesh_workload({}, coords, args, output);
        EXPECT_EQ(unmerged.workload.get_programs().size(), shape.mesh_size());
    }

    mesh_device->clear_program_cache();
    using MeshOp = ttnn::device_operation::MeshDeviceOperationAdapter<PerDeviceProgramMergeOp>;
    // The workload `launch` cached for these tensors.
    const auto cached_programs = [&](const PerDeviceProgramMergeOp::tensor_args_t& args) {
        const auto lookup = ttnn::device_operation::detail::look_up_program_cache<MeshOp>({}, args, mesh_device);
        EXPECT_TRUE(lookup.hit);
        return mesh_device->get_program_cache()
            .get(lookup.key)
            .cached_program.template get<MergeAdapter::cached_mesh_workload_t>()
            .workload.get_programs()
            .size();
    };

    const PerDeviceProgramMergeOp::tensor_args_t first{.input = replicated_input(mesh_device, 7)};
    const auto first_output = ttnn::device_operation::launch<PerDeviceProgramMergeOp>({}, first);
    EXPECT_EQ(mesh_device->get_program_cache().num_entries(), 1u);
    EXPECT_EQ(cached_programs(first), 1u);
    expect_input_plus_device_value(first_output, 7, shape);

    // Cache hit with new buffers (the first input and output are still alive, so the new tensors get other
    // addresses): the bindings are patched on the merged program and the per-device values stay.
    const PerDeviceProgramMergeOp::tensor_args_t second{.input = replicated_input(mesh_device, 100)};
    const auto second_output = ttnn::device_operation::launch<PerDeviceProgramMergeOp>({}, second);
    EXPECT_EQ(mesh_device->get_program_cache().num_entries(), 1u);
    EXPECT_EQ(cached_programs(second), 1u);
    EXPECT_NE(second.input.buffer()->address(), first.input.buffer()->address());
    EXPECT_NE(second_output.buffer()->address(), first_output.buffer()->address());
    expect_input_plus_device_value(second_output, 100, shape);
    expect_input_plus_device_value(first_output, 7, shape);
}

}  // namespace
