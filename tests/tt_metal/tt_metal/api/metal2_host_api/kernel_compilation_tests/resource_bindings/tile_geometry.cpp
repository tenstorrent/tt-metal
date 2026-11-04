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

class BindingTokenGeometry : public test_helpers::ProgramSpecTestBlackhole,
                             public ::testing::WithParamInterface<GeometryCase> {
protected:
    distributed::MeshDevice& fixture_device() { return *mesh_device_; }
};

// A bound resource exposes its tile height as a constant in DM and all compute stages.
TEST_P(BindingTokenGeometry, TileHeight) {
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
TEST_P(BindingTokenGeometry, TileWidth) {
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
TEST_P(BindingTokenGeometry, TileElementCount) {
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
TEST_P(BindingTokenGeometry, TileFaceCount) {
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

// A bound resource reports partial height using the same tile-height threshold as the host Tile.
TEST_P(BindingTokenGeometry, PartialFace) {
    const GeometryCase geometry = GetParam();
    const std::string source = fmt::format(
        R"(
void kernel_main() {{
    static_assert(get_partial_face(dfb::input_dfb) == {0});
    static_assert(get_partial_face(scratch::pad) == {0});
    static_assert(get_partial_face(tensor::local) == {0});
}}
)",
        geometry.partial);
    ProgramSpec spec = MakeNonPublicGeometryProgramSpec(geometry, source);
    EXPECT_NO_THROW(MakeProgramFromSpec(fixture_device(), spec));
}

// A bound resource reports narrow width independently of whether its height is partial.
TEST_P(BindingTokenGeometry, NarrowTile) {
    const GeometryCase geometry = GetParam();
    const std::string source = fmt::format(
        R"(
void kernel_main() {{
    static_assert(get_narrow_tile(dfb::input_dfb) == {0});
    static_assert(get_narrow_tile(scratch::pad) == {0});
    static_assert(get_narrow_tile(tensor::local) == {0});
}}
)",
        geometry.narrow);
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

}  // namespace
}  // namespace tt::tt_metal::experimental
