// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include <array>
#include <stdexcept>

#include <tt-metalium/experimental/metal2_host_api/program.hpp>
#include <tt-metalium/experimental/per_core_allocation/memory_config.hpp>

#include "impl/metal2_host_api/program_spec/validation/resource/resource.hpp"
#include "metal2_host_api/test_helpers/mock_device_fixtures.hpp"
#include "metal2_host_api/test_helpers/test_helpers.hpp"

namespace tt::tt_metal::experimental {
namespace {

struct MxFormatCase {
    const char* name;
    DataFormat format;
    DataType dtype;
};

constexpr std::array<MxFormatCase, 8> kMxFormats{{
    {"MxFp4", DataFormat::MxFp4, DataType::MXFP4},
    {"MxFp6P", DataFormat::MxFp6P, DataType::MXFP6_E2M3},
    {"MxFp6R", DataFormat::MxFp6R, DataType::MXFP6_E3M2},
    {"MxFp8P", DataFormat::MxFp8P, DataType::MXFP8_E4M3},
    {"MxFp8R", DataFormat::MxFp8R, DataType::MXFP8_E5M2},
    {"MxInt8", DataFormat::MxInt8, DataType::MXINT8},
    {"MxInt4", DataFormat::MxInt4, DataType::MXINT4},
    {"MxInt2", DataFormat::MxInt2, DataType::MXINT2},
}};

void ValidateNonPublicResourceSpecs(const ProgramSpec& spec, tt::ARCH arch = tt::ARCH::QUASAR) {
    const CollectedSpecData collected = CollectSpecData(spec);
    ValidateResourceSpecs(ValidationContext{spec, collected}, arch, 16, [](BufferType) { return 1u; });
}

TensorSpec MakeNonPublicTensorSpec(DataType dtype, const Tile& tile, bool per_node = false) {
    const TensorParameter parameter = test_helpers::MakeShardedTensorParameter(
        "tensor", Shape{tile.get_height(), tile.get_width()}, tile.get_tile_shape(), 1, Layout::TILE, tile);
    MemoryConfig memory_config = parameter.spec.memory_config();
    per_core_allocation::set_per_core_allocation(memory_config, per_node);
    return TensorSpec(
        parameter.spec.logical_shape(), TensorLayout(dtype, PageConfig(Layout::TILE, tile), memory_config));
}

class MxResourceTileMetadata : public ::testing::TestWithParam<MxFormatCase> {};

// DFB validation rejects MX tiles that contain less than one complete scale block.
TEST_P(MxResourceTileMetadata, DfbRejectsIncompleteBlock) {
    ProgramSpec spec = test_helpers::MakeMinimalValidProgramSpec();
    spec.dataflow_buffers[0].data_format_metadata = GetParam().format;
    spec.dataflow_buffers[0].tile_format_metadata = Tile({1, 16});
    EXPECT_THAT(
        [&] { ValidateNonPublicResourceSpecs(spec); },
        ::testing::ThrowsMessage<std::runtime_error>(
            ::testing::AllOf(::testing::HasSubstr("DFB 'dfb_0'"), ::testing::HasSubstr("multiple of 32 elements"))));
}

// Scratchpad validation rejects MX tiles that contain less than one complete scale block.
TEST_P(MxResourceTileMetadata, ScratchpadRejectsIncompleteBlock) {
    ProgramSpec spec = test_helpers::MakeMinimalValidProgramSpec();
    spec.scratchpads = {ScratchpadSpec{
        .unique_id = ScratchpadSpecName{"pad"},
        .size_per_node = 1024,
        .data_format_metadata = GetParam().format,
        .tile_format_metadata = Tile({1, 16})}};
    spec.kernels[0].scratchpad_bindings = {{.scratchpad_spec_name = ScratchpadSpecName{"pad"}, .accessor_name = "pad"}};
    EXPECT_THAT(
        [&] { ValidateNonPublicResourceSpecs(spec); },
        ::testing::ThrowsMessage<std::runtime_error>(::testing::AllOf(
            ::testing::HasSubstr("ScratchpadSpec 'pad'"), ::testing::HasSubstr("multiple of 32 elements"))));
}

// Tensor parameter validation rejects MX tiles before their storage size is calculated.
TEST_P(MxResourceTileMetadata, TensorRejectsIncompleteBlock) {
    ProgramSpec spec = test_helpers::MakeMinimalValidProgramSpec();
    spec.tensor_parameters = {
        {.unique_id = TensorParamName{"tensor"}, .spec = MakeNonPublicTensorSpec(GetParam().dtype, Tile({1, 16}))}};
    test_helpers::BindTensorParameterToKernel(spec.kernels[0], "tensor", "tensor");
    EXPECT_THAT(
        [&] { ValidateNonPublicResourceSpecs(spec); },
        ::testing::ThrowsMessage<std::runtime_error>(::testing::AllOf(
            ::testing::HasSubstr("TensorParameter 'tensor'"), ::testing::HasSubstr("multiple of 32 elements"))));
}

// Per-node tensor parameters have the same complete-block requirement as common tensor parameters.
TEST_P(MxResourceTileMetadata, PerNodeTensorRejectsIncompleteBlock) {
    ProgramSpec spec = test_helpers::MakeMinimalValidProgramSpec();
    spec.tensor_parameters = {
        {.unique_id = TensorParamName{"tensor"},
         .spec = MakeNonPublicTensorSpec(GetParam().dtype, Tile({1, 16}), true)}};
    test_helpers::BindTensorParameterToKernel(spec.kernels[0], "tensor", "tensor");
    EXPECT_THAT(
        [&] { ValidateNonPublicResourceSpecs(spec); },
        ::testing::ThrowsMessage<std::runtime_error>(::testing::AllOf(
            ::testing::HasSubstr("TensorParameter 'tensor'"), ::testing::HasSubstr("multiple of 32 elements"))));
}

// DFB metadata accepts the smallest supported tile containing a complete MX block.
TEST_P(MxResourceTileMetadata, DfbAcceptsCompleteBlock) {
    ProgramSpec spec = test_helpers::MakeMinimalValidProgramSpec();
    spec.dataflow_buffers[0].data_format_metadata = GetParam().format;
    spec.dataflow_buffers[0].tile_format_metadata = Tile({1, 32});
    EXPECT_NO_THROW(ValidateNonPublicResourceSpecs(spec));
}

// Scratchpad metadata accepts the smallest supported tile containing a complete MX block.
TEST_P(MxResourceTileMetadata, ScratchpadAcceptsCompleteBlock) {
    ProgramSpec spec = test_helpers::MakeMinimalValidProgramSpec();
    spec.scratchpads = {ScratchpadSpec{
        .unique_id = ScratchpadSpecName{"pad"},
        .size_per_node = 1024,
        .data_format_metadata = GetParam().format,
        .tile_format_metadata = Tile({1, 32})}};
    spec.kernels[0].scratchpad_bindings = {{.scratchpad_spec_name = ScratchpadSpecName{"pad"}, .accessor_name = "pad"}};
    EXPECT_NO_THROW(ValidateNonPublicResourceSpecs(spec));
}

INSTANTIATE_TEST_SUITE_P(
    MxFormats,
    MxResourceTileMetadata,
    ::testing::ValuesIn(kMxFormats),
    [](const ::testing::TestParamInfo<MxFormatCase>& info) { return info.param.name; });

// A tensor with one complete MX block passes resource validation.
TEST(ResourceTileMetadata, TensorAcceptsCompleteBlock) {
    ProgramSpec spec = test_helpers::MakeMinimalValidProgramSpec();
    spec.tensor_parameters = {
        {.unique_id = TensorParamName{"tensor"}, .spec = MakeNonPublicTensorSpec(DataType::MXFP4, Tile({1, 32}))}};
    test_helpers::BindTensorParameterToKernel(spec.kernels[0], "tensor", "tensor");
    EXPECT_NO_THROW(ValidateNonPublicResourceSpecs(spec));
}

// A per-node tensor with one complete MX block passes resource validation.
TEST(ResourceTileMetadata, PerNodeTensorAcceptsCompleteBlock) {
    ProgramSpec spec = test_helpers::MakeMinimalValidProgramSpec();
    spec.tensor_parameters = {
        {.unique_id = TensorParamName{"tensor"},
         .spec = MakeNonPublicTensorSpec(DataType::MXFP4, Tile({1, 32}), true)}};
    test_helpers::BindTensorParameterToKernel(spec.kernels[0], "tensor", "tensor");
    EXPECT_NO_THROW(ValidateNonPublicResourceSpecs(spec));
}

// Omitted tile metadata uses the default full tile for MX resources.
TEST(ResourceTileMetadata, MxDefaultTileRemainsValid) {
    ProgramSpec spec = test_helpers::MakeMinimalValidProgramSpec();
    spec.dataflow_buffers[0].data_format_metadata = DataFormat::MxFp4;
    spec.scratchpads = {ScratchpadSpec{
        .unique_id = ScratchpadSpecName{"pad"}, .size_per_node = 1024, .data_format_metadata = DataFormat::MxFp4}};
    spec.kernels[0].scratchpad_bindings = {{.scratchpad_spec_name = ScratchpadSpecName{"pad"}, .accessor_name = "pad"}};
    EXPECT_NO_THROW(ValidateNonPublicResourceSpecs(spec));
}

// Sixteen-element BFP tiles remain valid for resources that do not use MX scale blocks.
TEST(ResourceTileMetadata, SmallBfpTileRemainsValid) {
    ProgramSpec spec = test_helpers::MakeMinimalGen1ValidProgramSpec();
    spec.dataflow_buffers[0].data_format_metadata = DataFormat::Bfp8_b;
    spec.dataflow_buffers[0].tile_format_metadata = Tile({1, 16});
    spec.scratchpads = {ScratchpadSpec{
        .unique_id = ScratchpadSpecName{"pad"},
        .size_per_node = 1024,
        .data_format_metadata = DataFormat::Bfp8_b,
        .tile_format_metadata = Tile({1, 16})}};
    spec.kernels[0].scratchpad_bindings = {{.scratchpad_spec_name = ScratchpadSpecName{"pad"}, .accessor_name = "pad"}};
    spec.tensor_parameters = {
        {.unique_id = TensorParamName{"tensor"}, .spec = MakeNonPublicTensorSpec(DataType::BFLOAT8_B, Tile({1, 16}))}};
    test_helpers::BindTensorParameterToKernel(spec.kernels[0], "tensor", "tensor");
    EXPECT_NO_THROW(ValidateNonPublicResourceSpecs(spec, tt::ARCH::BLACKHOLE));
}

// Sixteen-element floating-point tiles remain valid on architectures that support MX formats.
TEST(ResourceTileMetadata, SmallFloat16TileRemainsValid) {
    ProgramSpec spec = test_helpers::MakeMinimalValidProgramSpec();
    spec.dataflow_buffers[0].tile_format_metadata = Tile({1, 16});
    spec.scratchpads = {ScratchpadSpec{
        .unique_id = ScratchpadSpecName{"pad"},
        .size_per_node = 1024,
        .data_format_metadata = DataFormat::Float16_b,
        .tile_format_metadata = Tile({1, 16})}};
    spec.kernels[0].scratchpad_bindings = {{.scratchpad_spec_name = ScratchpadSpecName{"pad"}, .accessor_name = "pad"}};
    spec.tensor_parameters = {
        {.unique_id = TensorParamName{"tensor"}, .spec = MakeNonPublicTensorSpec(DataType::BFLOAT16, Tile({1, 16}))}};
    test_helpers::BindTensorParameterToKernel(spec.kernels[0], "tensor", "tensor");
    EXPECT_NO_THROW(ValidateNonPublicResourceSpecs(spec));
}

// Invalid borrowed tensor metadata is rejected before DFB validation calculates its allocation size.
TEST(ResourceTileMetadata, BorrowedTensorRejectsIncompleteBlockBeforeSizing) {
    ProgramSpec spec = test_helpers::MakeMinimalValidProgramSpec();
    spec.tensor_parameters = {
        {.unique_id = TensorParamName{"tensor"}, .spec = MakeNonPublicTensorSpec(DataType::MXFP4, Tile({1, 16}))}};
    spec.dataflow_buffers[0].borrowed_from = TensorParamName{"tensor"};
    EXPECT_THAT(
        [&] { ValidateNonPublicResourceSpecs(spec); },
        ::testing::ThrowsMessage<std::runtime_error>(::testing::AllOf(
            ::testing::HasSubstr("TensorParameter 'tensor'"), ::testing::HasSubstr("multiple of 32 elements"))));
}

class MxResourceProgramValidation : public test_helpers::ProgramSpecTestQuasar {
protected:
    distributed::MeshDevice& fixture_device() { return *mesh_device_; }
};

// Public program construction reports the DFB specification when MX block metadata is invalid.
TEST_F(MxResourceProgramValidation, DfbRejectsIncompleteBlock) {
    ProgramSpec spec = test_helpers::MakeMinimalValidProgramSpec();
    spec.dataflow_buffers[0].data_format_metadata = DataFormat::MxFp4;
    spec.dataflow_buffers[0].tile_format_metadata = Tile({1, 16});
    EXPECT_THAT(
        [&] { MakeProgramFromSpec(fixture_device(), spec); },
        ::testing::ThrowsMessage<std::runtime_error>(
            ::testing::AllOf(::testing::HasSubstr("DFB 'dfb_0'"), ::testing::HasSubstr("multiple of 32 elements"))));
}

// Public program construction reports the scratchpad specification when MX block metadata is invalid.
TEST_F(MxResourceProgramValidation, ScratchpadRejectsIncompleteBlock) {
    ProgramSpec spec = test_helpers::MakeMinimalValidProgramSpec();
    spec.scratchpads = {ScratchpadSpec{
        .unique_id = ScratchpadSpecName{"pad"},
        .size_per_node = 1024,
        .data_format_metadata = DataFormat::MxFp4,
        .tile_format_metadata = Tile({1, 16})}};
    spec.kernels[0].scratchpad_bindings = {{.scratchpad_spec_name = ScratchpadSpecName{"pad"}, .accessor_name = "pad"}};
    EXPECT_THAT(
        [&] { MakeProgramFromSpec(fixture_device(), spec); },
        ::testing::ThrowsMessage<std::runtime_error>(::testing::AllOf(
            ::testing::HasSubstr("ScratchpadSpec 'pad'"), ::testing::HasSubstr("multiple of 32 elements"))));
}

// Public program construction reports the tensor parameter when MX block metadata is invalid.
TEST_F(MxResourceProgramValidation, TensorRejectsIncompleteBlock) {
    ProgramSpec spec = test_helpers::MakeMinimalValidProgramSpec();
    spec.tensor_parameters = {
        {.unique_id = TensorParamName{"tensor"}, .spec = MakeNonPublicTensorSpec(DataType::MXFP4, Tile({1, 16}))}};
    test_helpers::BindTensorParameterToKernel(spec.kernels[0], "tensor", "tensor");
    EXPECT_THAT(
        [&] { MakeProgramFromSpec(fixture_device(), spec); },
        ::testing::ThrowsMessage<std::runtime_error>(::testing::AllOf(
            ::testing::HasSubstr("TensorParameter 'tensor'"), ::testing::HasSubstr("multiple of 32 elements"))));
}

// Public program construction applies the same MX block rule to per-node tensor parameters.
TEST_F(MxResourceProgramValidation, PerNodeTensorRejectsIncompleteBlock) {
    ProgramSpec spec = test_helpers::MakeMinimalValidProgramSpec();
    spec.tensor_parameters = {
        {.unique_id = TensorParamName{"tensor"},
         .spec = MakeNonPublicTensorSpec(DataType::MXFP4, Tile({1, 16}), true)}};
    test_helpers::BindTensorParameterToKernel(spec.kernels[0], "tensor", "tensor");
    EXPECT_THAT(
        [&] { MakeProgramFromSpec(fixture_device(), spec); },
        ::testing::ThrowsMessage<std::runtime_error>(::testing::AllOf(
            ::testing::HasSubstr("TensorParameter 'tensor'"), ::testing::HasSubstr("multiple of 32 elements"))));
}

}  // namespace
}  // namespace tt::tt_metal::experimental
