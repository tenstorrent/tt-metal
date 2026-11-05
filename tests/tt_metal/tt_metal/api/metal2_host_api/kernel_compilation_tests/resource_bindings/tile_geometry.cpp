// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <cstdint>
#include <optional>
#include <string>

#include <fmt/format.h>
#include <tt-metalium/experimental/metal2_host_api/program.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>

#include "metal2_host_api/test_helpers/mock_device_fixtures.hpp"
#include "metal2_host_api/test_helpers/test_helpers.hpp"

namespace tt::tt_metal::experimental {
namespace {

struct GeometryCase {
    const char* name;
    std::optional<Tile> tile;
    uint32_t rows;
    uint32_t columns;
    uint32_t faces;
    bool partial;
    bool narrow;
    bool per_node = false;
};

ProgramSpec MakeNonPublicGeometryProgramSpec(const GeometryCase& geometry, const std::string& kernel_source) {
    ProgramSpec spec = test_helpers::MakeMinimalGen1ValidProgramSpec();
    const std::optional<Tile>& tile = geometry.tile;
    spec.dataflow_buffers[0].tile_format_metadata = tile;
    const Tile effective_tile = tile.value_or(Tile{});
    const auto shape = effective_tile.get_tile_shape();
    spec.tensor_parameters = {
        geometry.per_node ? test_helpers::MakePerNodeTensorParameter("local", 1)
                          : test_helpers::MakeShardedTensorParameter(
                                "local", Shape{shape[0], shape[1]}, shape, 1, Layout::TILE, tile)};
    for (KernelSpec& kernel : spec.kernels) {
        const ScratchpadSpecName pad_name{"pad_" + kernel.unique_id.get()};
        spec.scratchpads.push_back(ScratchpadSpec{
            .unique_id = pad_name,
            .size_per_node = 4096,
            .data_format_metadata = tt::DataFormat::Float16_b,
            .tile_format_metadata = tile});
        kernel.scratchpad_bindings.push_back({.scratchpad_spec_name = pad_name, .accessor_name = "pad"});
        test_helpers::BindTensorParameterToKernel(kernel, "local", "local");
        kernel.source = KernelSpec::SourceCode{kernel_source};
    }
    return spec;
}

std::string MakeNonPublicTensorShapeQuerySource(const GeometryCase& geometry, const std::string& query) {
    const Tile tile = geometry.tile.value_or(Tile{});
    const auto face = tile.get_face_shape();
    return fmt::format(
        R"(
#include "api/llk_operand_from_tokens.h"
#include <type_traits>
void kernel_main() {{
    static_assert(std::is_same_v<decltype({0}), ckernel::TensorShape>);
    constexpr ckernel::TensorShape shape = {0};
    static_assert(shape.face_r_dim == {1} && shape.face_c_dim == {2} &&
                  shape.num_faces_r_dim == {3} && shape.num_faces_c_dim == {4});
}}
)",
        query,
        face[0],
        face[1],
        geometry.rows / face[0],
        geometry.columns / face[1]);
}

std::string MakeNonPublicTileFlagQuerySource(const char* getter, uint32_t expected) {
    std::string source = "#include <type_traits>\nvoid kernel_main() {\n";
    for (const char* token : {"dfb::input_dfb", "scratch::pad", "tensor::local"}) {
        source += fmt::format(
            "static_assert(std::is_same_v<decltype({0}({1})), uint32_t>);\n"
            "static_assert({0}({1}) == {2}u);\n",
            getter,
            token,
            expected);
    }
    return source + "}\n";
}

class BindingTokenGeometry : public test_helpers::ProgramSpecTestBlackhole,
                             public ::testing::WithParamInterface<GeometryCase> {
protected:
    distributed::MeshDevice& fixture_device() { return *mesh_device_; }
};

// Blackhole DFB shape queries compile in data movement and all compute stages.
TEST_P(BindingTokenGeometry, CPU_DfbTensorShape) {
    const GeometryCase geometry = GetParam();
    const std::string source = MakeNonPublicTensorShapeQuerySource(geometry, "::get_tensor_shape(dfb::input_dfb)");
    ProgramSpec spec = MakeNonPublicGeometryProgramSpec(geometry, source);
    EXPECT_NO_THROW(MakeProgramFromSpec(fixture_device(), spec));
}

// Blackhole scratchpad shape queries compile in data movement and all compute stages.
TEST_P(BindingTokenGeometry, CPU_ScratchpadTensorShape) {
    const GeometryCase geometry = GetParam();
    const std::string source = MakeNonPublicTensorShapeQuerySource(geometry, "::get_tensor_shape(scratch::pad)");
    ProgramSpec spec = MakeNonPublicGeometryProgramSpec(geometry, source);
    EXPECT_NO_THROW(MakeProgramFromSpec(fixture_device(), spec));
}

// Blackhole common and per-node tensor shape queries compile in data movement and all compute stages.
TEST_P(BindingTokenGeometry, CPU_TensorBindingTensorShape) {
    const GeometryCase geometry = GetParam();
    const std::string source =
        MakeNonPublicTensorShapeQuerySource(geometry, "tensor_accessor::get_tensor_shape(tensor::local)");
    ProgramSpec spec = MakeNonPublicGeometryProgramSpec(geometry, source);
    EXPECT_NO_THROW(MakeProgramFromSpec(fixture_device(), spec));
}

// A bound resource exposes its tile height as a constant in DM and all compute stages.
TEST_P(BindingTokenGeometry, CPU_TileHeight) {
    const GeometryCase geometry = GetParam();
    const std::string source = fmt::format(
        R"(
void kernel_main() {{
    static_assert(get_tile_r_dim(dfb::input_dfb) == {0});
    static_assert(get_tile_r_dim(scratch::pad) == {0});
    static_assert(get_tile_r_dim(tensor::local) == {0});
}}
)",
        geometry.rows);
    ProgramSpec spec = MakeNonPublicGeometryProgramSpec(geometry, source);
    EXPECT_NO_THROW(MakeProgramFromSpec(fixture_device(), spec));
}

// A bound resource exposes its tile width as a constant in DM and all compute stages.
TEST_P(BindingTokenGeometry, CPU_TileWidth) {
    const GeometryCase geometry = GetParam();
    const std::string source = fmt::format(
        R"(
void kernel_main() {{
    static_assert(get_tile_c_dim(dfb::input_dfb) == {0});
    static_assert(get_tile_c_dim(scratch::pad) == {0});
    static_assert(get_tile_c_dim(tensor::local) == {0});
}}
)",
        geometry.columns);
    ProgramSpec spec = MakeNonPublicGeometryProgramSpec(geometry, source);
    EXPECT_NO_THROW(MakeProgramFromSpec(fixture_device(), spec));
}

// A bound resource exposes its tile element count independently of its storage format.
TEST_P(BindingTokenGeometry, CPU_TileElementCount) {
    const GeometryCase geometry = GetParam();
    const std::string source = fmt::format(
        R"(
void kernel_main() {{
    static_assert(get_tile_hw(dfb::input_dfb) == {0});
    static_assert(get_tile_hw(scratch::pad) == {0});
    static_assert(get_tile_hw(tensor::local) == {0});
}}
)",
        geometry.rows * geometry.columns);
    ProgramSpec spec = MakeNonPublicGeometryProgramSpec(geometry, source);
    EXPECT_NO_THROW(MakeProgramFromSpec(fixture_device(), spec));
}

// A bound resource exposes the total number of faces, including custom face layouts.
TEST_P(BindingTokenGeometry, CPU_TileFaceCount) {
    const GeometryCase geometry = GetParam();
    const std::string source = fmt::format(
        R"(
void kernel_main() {{
    static_assert(get_tile_num_faces(dfb::input_dfb) == {0});
    static_assert(get_tile_num_faces(scratch::pad) == {0});
    static_assert(get_tile_num_faces(tensor::local) == {0});
}}
)",
        geometry.faces);
    ProgramSpec spec = MakeNonPublicGeometryProgramSpec(geometry, source);
    EXPECT_NO_THROW(MakeProgramFromSpec(fixture_device(), spec));
}

// A bound resource exposes its partial-height flag as uint32_t with the expected 0/1 value.
TEST_P(BindingTokenGeometry, CPU_PartialFace) {
    const GeometryCase geometry = GetParam();
    const std::string source = MakeNonPublicTileFlagQuerySource("get_partial_face", geometry.partial);
    ProgramSpec spec = MakeNonPublicGeometryProgramSpec(geometry, source);
    EXPECT_NO_THROW(MakeProgramFromSpec(fixture_device(), spec));
}

// A bound resource exposes its narrow-width flag as uint32_t with the expected 0/1 value.
TEST_P(BindingTokenGeometry, CPU_NarrowTile) {
    const GeometryCase geometry = GetParam();
    const std::string source = MakeNonPublicTileFlagQuerySource("get_narrow_tile", geometry.narrow);
    ProgramSpec spec = MakeNonPublicGeometryProgramSpec(geometry, source);
    EXPECT_NO_THROW(MakeProgramFromSpec(fixture_device(), spec));
}

INSTANTIATE_TEST_SUITE_P(
    ResourceTiles,
    BindingTokenGeometry,
    ::testing::Values(
        GeometryCase{"Default", std::nullopt, 32, 32, 4, false, false},
        GeometryCase{"Full", Tile({32, 32}), 32, 32, 4, false, false},
        GeometryCase{"Partial", Tile({16, 32}), 16, 32, 2, true, false},
        GeometryCase{"Narrow", Tile({32, 16}), 32, 16, 2, false, true},
        GeometryCase{"SingleFace", Tile({16, 16}), 16, 16, 1, true, true},
        GeometryCase{"ShortFaces", Tile({16, 32}, {8, 16}), 16, 32, 4, true, false},
        GeometryCase{"TinyFaces", Tile({2, 32}, {1, 16}), 2, 32, 4, true, false},
        GeometryCase{"PerNode", std::nullopt, 32, 32, 4, false, false, true}),
    [](const ::testing::TestParamInfo<GeometryCase>& info) { return info.param.name; });

class BindingTokenGeometryWormhole : public test_helpers::ProgramSpecTestGen1,
                                     public ::testing::WithParamInterface<GeometryCase> {
protected:
    distributed::MeshDevice& fixture_device() { return *mesh_device_; }
};

// Wormhole DFB shape queries compile in data movement and all compute stages.
TEST_P(BindingTokenGeometryWormhole, CPU_DfbTensorShape) {
    const GeometryCase geometry = GetParam();
    const std::string source = MakeNonPublicTensorShapeQuerySource(geometry, "::get_tensor_shape(dfb::input_dfb)");
    ProgramSpec spec = MakeNonPublicGeometryProgramSpec(geometry, source);
    EXPECT_NO_THROW(MakeProgramFromSpec(fixture_device(), spec));
}

// Wormhole scratchpad shape queries compile in data movement and all compute stages.
TEST_P(BindingTokenGeometryWormhole, CPU_ScratchpadTensorShape) {
    const GeometryCase geometry = GetParam();
    const std::string source = MakeNonPublicTensorShapeQuerySource(geometry, "::get_tensor_shape(scratch::pad)");
    ProgramSpec spec = MakeNonPublicGeometryProgramSpec(geometry, source);
    EXPECT_NO_THROW(MakeProgramFromSpec(fixture_device(), spec));
}

// Wormhole common and per-node tensor shape queries compile in data movement and all compute stages.
TEST_P(BindingTokenGeometryWormhole, CPU_TensorBindingTensorShape) {
    const GeometryCase geometry = GetParam();
    const std::string source =
        MakeNonPublicTensorShapeQuerySource(geometry, "tensor_accessor::get_tensor_shape(tensor::local)");
    ProgramSpec spec = MakeNonPublicGeometryProgramSpec(geometry, source);
    EXPECT_NO_THROW(MakeProgramFromSpec(fixture_device(), spec));
}

INSTANTIATE_TEST_SUITE_P(
    ResourceShapes,
    BindingTokenGeometryWormhole,
    ::testing::Values(
        GeometryCase{"Default", std::nullopt, 32, 32, 4, false, false},
        GeometryCase{"Partial", Tile({16, 32}), 16, 32, 2, true, false},
        GeometryCase{"Narrow", Tile({32, 16}), 32, 16, 2, false, true},
        GeometryCase{"PerNode", std::nullopt, 32, 32, 4, false, false, true}),
    [](const ::testing::TestParamInfo<GeometryCase>& info) { return info.param.name; });

std::string MakeNonPublicCustomFaceQuerySource(const GeometryCase& geometry, const char* token) {
    const Tile tile = geometry.tile.value_or(Tile{});
    const auto face = tile.get_face_shape();
    return fmt::format(
        R"(
void kernel_main() {{
    static_assert(get_face_r_dim({0}) == {1} && get_face_c_dim({0}) == {2} &&
                  get_num_faces_r_dim({0}) == {3} && get_num_faces_c_dim({0}) == {4});
    static_assert(get_tile_r_dim({0}) == {5} && get_tile_c_dim({0}) == {6} &&
                  get_tile_hw({0}) == {7} && get_tile_num_faces({0}) == {8});
    static_assert(get_partial_face({0}) == {9}u && get_narrow_tile({0}) == {10}u);
}}
)",
        token,
        face[0],
        face[1],
        geometry.rows / face[0],
        geometry.columns / face[1],
        geometry.rows,
        geometry.columns,
        geometry.rows * geometry.columns,
        geometry.faces,
        static_cast<uint32_t>(geometry.partial),
        static_cast<uint32_t>(geometry.narrow));
}

class BindingTokenCustomFaceWidth : public test_helpers::ProgramSpecTestBlackhole,
                                    public ::testing::WithParamInterface<GeometryCase> {
protected:
    distributed::MeshDevice& fixture_device() { return *mesh_device_; }
};

// A scratchpad's scalar geometry reflects the face width configured on its resource.
TEST_P(BindingTokenCustomFaceWidth, CPU_ScratchpadGeometry) {
    const GeometryCase geometry = GetParam();
    const std::string source = MakeNonPublicCustomFaceQuerySource(geometry, "scratch::pad");
    ProgramSpec spec = MakeNonPublicGeometryProgramSpec(geometry, source);
    spec.dataflow_buffers[0].tile_format_metadata.reset();
    EXPECT_NO_THROW(MakeProgramFromSpec(fixture_device(), spec));
}

// A tensor binding's scalar geometry reflects the face width configured on its resource.
TEST_P(BindingTokenCustomFaceWidth, CPU_TensorGeometry) {
    const GeometryCase geometry = GetParam();
    const std::string source = MakeNonPublicCustomFaceQuerySource(geometry, "tensor::local");
    ProgramSpec spec = MakeNonPublicGeometryProgramSpec(geometry, source);
    spec.dataflow_buffers[0].tile_format_metadata.reset();
    EXPECT_NO_THROW(MakeProgramFromSpec(fixture_device(), spec));
}

// A scratchpad's aggregate shape preserves custom face dimensions and grid counts.
TEST_P(BindingTokenCustomFaceWidth, CPU_ScratchpadTensorShape) {
    const GeometryCase geometry = GetParam();
    ProgramSpec spec = MakeNonPublicGeometryProgramSpec(geometry, "void kernel_main() {}");
    spec.dataflow_buffers[0].tile_format_metadata.reset();
    spec.kernels[1].source =
        KernelSpec::SourceCode{MakeNonPublicTensorShapeQuerySource(geometry, "::get_tensor_shape(scratch::pad)")};
    EXPECT_NO_THROW(MakeProgramFromSpec(fixture_device(), spec));
}

// A tensor binding's aggregate shape preserves custom face dimensions and grid counts.
TEST_P(BindingTokenCustomFaceWidth, CPU_TensorBindingTensorShape) {
    const GeometryCase geometry = GetParam();
    ProgramSpec spec = MakeNonPublicGeometryProgramSpec(geometry, "void kernel_main() {}");
    spec.dataflow_buffers[0].tile_format_metadata.reset();
    spec.kernels[1].source = KernelSpec::SourceCode{
        MakeNonPublicTensorShapeQuerySource(geometry, "tensor_accessor::get_tensor_shape(tensor::local)")};
    EXPECT_NO_THROW(MakeProgramFromSpec(fixture_device(), spec));
}

INSTANTIATE_TEST_SUITE_P(
    ResourceFaces,
    BindingTokenCustomFaceWidth,
    ::testing::Values(
        GeometryCase{"EightWideFaces", Tile({16, 16}, {16, 8}), 16, 16, 2, true, true},
        GeometryCase{"SquareEightWideFaces", Tile({16, 16}, {8, 8}), 16, 16, 4, true, true},
        GeometryCase{"TallEightWideFaces", Tile({32, 16}, {16, 8}), 32, 16, 4, false, true}),
    [](const ::testing::TestParamInfo<GeometryCase>& info) { return info.param.name; });

struct ShapeWithoutFormatCase {
    const char* name;
    const char* query;
    bool per_node = false;
};

class BindingTokenTensorShapeWithoutFormat : public test_helpers::ProgramSpecTestBlackhole,
                                             public ::testing::WithParamInterface<ShapeWithoutFormatCase> {
protected:
    distributed::MeshDevice& fixture_device() { return *mesh_device_; }
};

// An aggregate shape query returns default geometry without requiring a storage format.
TEST_P(BindingTokenTensorShapeWithoutFormat, CPU_DefaultGeometry) {
    const ShapeWithoutFormatCase missing = GetParam();
    const GeometryCase geometry{"Default", std::nullopt, 32, 32, 4, false, false, missing.per_node};
    ProgramSpec spec = MakeNonPublicGeometryProgramSpec(geometry, "void kernel_main() {}");
    spec.kernels[1].source = KernelSpec::SourceCode{MakeNonPublicTensorShapeQuerySource(geometry, missing.query)};
    EXPECT_NO_THROW(MakeProgramFromSpec(fixture_device(), spec));
}

INSTANTIATE_TEST_SUITE_P(
    ResourceShapes,
    BindingTokenTensorShapeWithoutFormat,
    ::testing::Values(
        ShapeWithoutFormatCase{"Dfb", "get_tensor_shape(DFBBindingToken<>{0})"},
        ShapeWithoutFormatCase{"Scratchpad", "get_tensor_shape(ScratchpadBindingToken{0, 1024})"},
        ShapeWithoutFormatCase{"Tensor", "get_tensor_shape(decltype(tensor::local){binding_details::LLKMetadata{}})"},
        ShapeWithoutFormatCase{
            "PerNodeTensor", "get_tensor_shape(decltype(tensor::local){binding_details::LLKMetadata{}})", true}),
    [](const ::testing::TestParamInfo<ShapeWithoutFormatCase>& info) { return info.param.name; });

}  // namespace
}  // namespace tt::tt_metal::experimental
