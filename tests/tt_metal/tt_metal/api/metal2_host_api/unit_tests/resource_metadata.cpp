// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include "distributed/mesh_device_impl.hpp"
#include "impl/metal2_host_api/program_spec/construction/resource/resource.hpp"
#include "metal2_host_api/test_helpers/mock_device_fixtures.hpp"
#include "metal2_host_api/test_helpers/test_helpers.hpp"

namespace tt::tt_metal::experimental {
namespace {

DataflowBufferBindingHandleMap MakeNonPublicDfbBindingHandles(const DataflowBufferSpec& spec) {
    KernelSpec kernel = test_helpers::MakeMinimalGen1ValidProgramSpec().kernels.front();
    kernel.dfb_bindings = {ConsumerOf(spec.unique_id, "buffer")};
    return MakeDataflowBufferBindingHandles(
        kernel,
        {{spec.unique_id, 0}},
        {{spec.unique_id, false}},
        {{spec.unique_id, 0xFF}},
        {{spec.unique_id, &spec}},
        {{spec.unique_id, 0}},
        {{spec.unique_id, {}}});
}

ScratchpadBindingsForKernel MakeNonPublicScratchpadBindingHandles(const ScratchpadSpec& spec) {
    KernelSpec kernel = test_helpers::MakeMinimalGen1ValidProgramSpec().kernels.front();
    kernel.scratchpad_bindings = {{.scratchpad_spec_name = spec.unique_id, .accessor_name = "pad"}};
    return ResolveScratchpadBindingsForKernel(kernel, {{spec.unique_id, &spec}}, 0);
}

// DFB binding construction returns its tile storage bytes without a later kernel-lowering pass.
TEST(ResourceBindingMetadata, DfbStorageBytesReady) {
    DataflowBufferSpec spec = test_helpers::MakeMinimalGen1ValidProgramSpec().dataflow_buffers.front();
    spec.data_format_metadata = DataFormat::Float16_b;
    spec.tile_format_metadata = Tile({16, 32});
    const auto handles = MakeNonPublicDfbBindingHandles(spec);
    EXPECT_THAT(
        handles.at("buffer").llk_metadata, ::testing::Optional(::testing::Field(&LLKMetadata::tile_size_bytes, 1024u)));
}

// Scratchpad binding construction returns its tile storage bytes without a later kernel-lowering pass.
TEST(ResourceBindingMetadata, ScratchpadStorageBytesReady) {
    const ScratchpadSpec spec{
        .unique_id = ScratchpadSpecName{"pad"},
        .size_per_node = 4096,
        .data_format_metadata = DataFormat::Float16_b,
        .tile_format_metadata = Tile({16, 32})};
    const auto bindings = MakeNonPublicScratchpadBindingHandles(spec);
    EXPECT_THAT(
        bindings.handles.front().llk_metadata,
        ::testing::Optional(::testing::Field(&LLKMetadata::tile_size_bytes, 1024u)));
}

// DFB binding construction preserves the absence of optional format metadata.
TEST(ResourceBindingMetadata, DfbWithoutFormatHasNoMetadata) {
    DataflowBufferSpec spec = test_helpers::MakeMinimalGen1ValidProgramSpec().dataflow_buffers.front();
    spec.data_format_metadata.reset();
    spec.tile_format_metadata.reset();
    const auto handles = MakeNonPublicDfbBindingHandles(spec);
    EXPECT_EQ(handles.at("buffer").llk_metadata, std::nullopt);
}

// Scratchpad binding construction preserves the absence of optional format metadata.
TEST(ResourceBindingMetadata, ScratchpadWithoutFormatHasNoMetadata) {
    const ScratchpadSpec spec{.unique_id = ScratchpadSpecName{"pad"}, .size_per_node = 4096};
    const auto bindings = MakeNonPublicScratchpadBindingHandles(spec);
    EXPECT_EQ(bindings.handles.front().llk_metadata, std::nullopt);
}

// TF32 metadata retains its format while reporting no supported tile storage encoding.
TEST(ResourceBindingMetadata, Tf32HasNoStorageSize) {
    const ScratchpadSpec spec{
        .unique_id = ScratchpadSpecName{"pad"}, .size_per_node = 4096, .data_format_metadata = DataFormat::Tf32};
    const auto bindings = MakeNonPublicScratchpadBindingHandles(spec);
    EXPECT_THAT(
        bindings.handles.front().llk_metadata,
        ::testing::Optional(::testing::AllOf(
            ::testing::Field(&LLKMetadata::format, DataFormat::Tf32),
            ::testing::Field(&LLKMetadata::tile_size_bytes, 0u))));
}

class TensorBindingMetadata : public test_helpers::ProgramSpecTestBlackhole {
protected:
    distributed::MeshDevice& fixture_device() { return *mesh_device_; }
};

// Tensor parameter registration returns storage bytes before any kernel binding is lowered.
TEST_F(TensorBindingMetadata, StorageBytesReady) {
    distributed::MeshDevice& device = fixture_device();
    detail::ProgramImpl program_impl(device.impl().get_context_id());
    ProgramSpec spec = test_helpers::MakeMinimalGen1ValidProgramSpec();
    spec.tensor_parameters = {
        test_helpers::MakeShardedTensorParameter("tensor", Shape{16, 32}, {16, 32}, 1, Layout::TILE, Tile({16, 32}))};
    test_helpers::BindTensorParameterToKernel(spec.kernels[0], "tensor", "tensor");
    const auto parameters = RegisterTensorParameters(spec, device, program_impl);
    EXPECT_EQ(parameters.at(TensorParamName{"tensor"}).llk_metadata.tile_size_bytes, 1024u);
}

// Per-node tensor registration returns storage bytes before any kernel binding is lowered.
TEST_F(TensorBindingMetadata, PerNodeStorageBytesReady) {
    distributed::MeshDevice& device = fixture_device();
    detail::ProgramImpl program_impl(device.impl().get_context_id());
    ProgramSpec spec = test_helpers::MakeMinimalGen1ValidProgramSpec();
    spec.tensor_parameters = {test_helpers::MakePerNodeTensorParameter("tensor", 1)};
    test_helpers::BindTensorParameterToKernel(spec.kernels[0], "tensor", "tensor");
    const auto parameters = RegisterTensorParameters(spec, device, program_impl);
    EXPECT_EQ(parameters.at(TensorParamName{"tensor"}).llk_metadata.tile_size_bytes, 2048u);
}

}  // namespace
}  // namespace tt::tt_metal::experimental
