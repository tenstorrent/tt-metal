// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include <cstdint>
#include <exception>
#include <optional>
#include <string>
#include <vector>

#include <fmt/format.h>
#include <tt-metalium/experimental/metal2_host_api/program.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>

#include "metal2_host_api/test_helpers/mock_device_fixtures.hpp"
#include "metal2_host_api/test_helpers/test_helpers.hpp"
#include "impl/kernels/kernel.hpp"
#include "impl/program/program_impl.hpp"
#include "impl/metal2_host_api/llk_metadata.hpp"

namespace tt::tt_metal::experimental {
namespace {

struct TileSizeCase {
    const char* name;
    tt::DataFormat format;
    DataType data_type;
    std::optional<Tile> tile;
    uint32_t bytes;
    bool per_node = false;
};

ProgramSpec MakeNonPublicTileSizeProgramSpec(const TileSizeCase& size, const std::string& source, bool gen2 = false) {
    ProgramSpec spec =
        gen2 ? test_helpers::MakeMinimalValidProgramSpec() : test_helpers::MakeMinimalGen1ValidProgramSpec();
    spec.dataflow_buffers[0].entry_size = 8192;
    spec.dataflow_buffers[0].data_format_metadata = size.format;
    spec.dataflow_buffers[0].tile_format_metadata = size.tile;
    const Tile tile = size.tile.value_or(Tile{});
    const auto shape = tile.get_tile_shape();
    TensorParameter parameter = size.per_node
                                    ? test_helpers::MakePerNodeTensorParameter("local", 1)
                                    : test_helpers::MakeShardedTensorParameter(
                                          "local", Shape{shape[0], shape[1]}, shape, 1, Layout::TILE, size.tile);
    parameter.spec = TensorSpec(
        parameter.spec.logical_shape(),
        TensorLayout(size.data_type, PageConfig(Layout::TILE, size.tile), parameter.spec.memory_config()));
    spec.tensor_parameters = {parameter};
    for (KernelSpec& kernel : spec.kernels) {
        const ScratchpadSpecName pad_name{"pad_" + kernel.unique_id.get()};
        spec.scratchpads.push_back(ScratchpadSpec{
            .unique_id = pad_name,
            .size_per_node = 8192,
            .data_format_metadata = size.format,
            .tile_format_metadata = size.tile});
        kernel.scratchpad_bindings.push_back({.scratchpad_spec_name = pad_name, .accessor_name = "pad"});
        test_helpers::BindTensorParameterToKernel(kernel, "local", "local");
        kernel.source = KernelSpec::SourceCode{source};
    }
    return spec;
}

std::vector<std::string> GetNonPublicSerializedResourceMetadata(Program& program) {
    std::vector<std::string> metadata;
    for (const char* name : {"dm_kernel", "compute_kernel"}) {
        const Kernel& kernel = *program.impl().get_kernel_by_spec_name(name);
        kernel.process_dataflow_buffer_binding_handles([&](const auto&, auto, bool, auto, const auto& llk, auto, auto) {
            if (llk) {
                metadata.push_back(serialize_llk_metadata(*llk));
            }
        });
        kernel.process_tensor_binding_handles([&](const auto&, auto, bool, auto, auto, const LLKMetadata& llk) {
            metadata.push_back(serialize_llk_metadata(llk));
        });
        kernel.process_scratchpad_binding_handles([&](const auto&, auto, auto, const auto& llk) {
            if (llk) {
                metadata.push_back(serialize_llk_metadata(*llk));
            }
        });
    }
    return metadata;
}

class BindingTokenTileSizeBlackhole : public test_helpers::ProgramSpecTestBlackhole,
                                      public ::testing::WithParamInterface<TileSizeCase> {
protected:
    distributed::MeshDevice& fixture_device() { return *mesh_device_; }
};

class BindingTokenTileSizeQuasar : public test_helpers::ProgramSpecTestQuasar,
                                   public ::testing::WithParamInterface<TileSizeCase> {
protected:
    distributed::MeshDevice& fixture_device() { return *mesh_device_; }
};

// Resource tokens expose tile storage bytes independently of DFB entries and scratchpad allocation sizes.
TEST_P(BindingTokenTileSizeBlackhole, TileStorageBytes) {
    const TileSizeCase size = GetParam();
    const std::string source = fmt::format(
        R"(
void kernel_main() {{
    static_assert(get_tile_size(dfb::input_dfb) == {0});
    static_assert(get_tile_size(scratch::pad) == {0});
    static_assert(get_tile_size(tensor::local) == {0});
}}
)",
        size.bytes);
    ProgramSpec spec = MakeNonPublicTileSizeProgramSpec(size, source);
    EXPECT_NO_THROW(MakeProgramFromSpec(fixture_device(), spec));
}

// MX storage sizes use the host format and include block scales before hardware format remapping.
TEST_P(BindingTokenTileSizeQuasar, TileStorageBytes) {
    const TileSizeCase size = GetParam();
    ProgramSpec spec = MakeNonPublicTileSizeProgramSpec(size, "void kernel_main() {}", true);
    Program program = MakeProgramFromSpec(fixture_device(), spec);
    EXPECT_THAT(
        GetNonPublicSerializedResourceMetadata(program),
        ::testing::AllOf(
            ::testing::SizeIs(6),
            ::testing::Each(::testing::HasSubstr(fmt::format(".tile_size_bytes = {}u", size.bytes)))));
}

INSTANTIATE_TEST_SUITE_P(
    ResourceSizes,
    BindingTokenTileSizeBlackhole,
    ::testing::Values(
        TileSizeCase{"Default", tt::DataFormat::Float16_b, DataType::BFLOAT16, std::nullopt, 2048},
        TileSizeCase{"Partial", tt::DataFormat::Float16_b, DataType::BFLOAT16, Tile({16, 32}), 1024},
        TileSizeCase{"Narrow", tt::DataFormat::Float16_b, DataType::BFLOAT16, Tile({32, 16}), 1024},
        TileSizeCase{"CustomFaces", tt::DataFormat::Float16_b, DataType::BFLOAT16, Tile({2, 32}, {1, 16}), 128},
        TileSizeCase{"Float32", tt::DataFormat::Float32, DataType::FLOAT32, std::nullopt, 4096},
        TileSizeCase{"UInt8", tt::DataFormat::UInt8, DataType::UINT8, Tile({1, 16}), 16},
        TileSizeCase{"RawUInt32", tt::DataFormat::RawUInt32, DataType::UINT32, std::nullopt, 4096},
        TileSizeCase{"Bfp8", tt::DataFormat::Bfp8_b, DataType::BFLOAT8_B, std::nullopt, 1088},
        TileSizeCase{"Bfp4Tiny", tt::DataFormat::Bfp4_b, DataType::BFLOAT4_B, Tile({2, 32}, {1, 16}), 48},
        TileSizeCase{"Bfp4Narrow", tt::DataFormat::Bfp4_b, DataType::BFLOAT4_B, Tile({32, 16}), 288},
        TileSizeCase{"PerNode", tt::DataFormat::Float16_b, DataType::BFLOAT16, std::nullopt, 2048, true}),
    [](const ::testing::TestParamInfo<TileSizeCase>& info) { return info.param.name; });

INSTANTIATE_TEST_SUITE_P(
    ResourceSizes,
    BindingTokenTileSizeQuasar,
    ::testing::Values(
        TileSizeCase{"Default", tt::DataFormat::Float16_b, DataType::BFLOAT16, std::nullopt, 2048},
        TileSizeCase{"MxFp4", tt::DataFormat::MxFp4, DataType::MXFP4, std::nullopt, 544},
        TileSizeCase{"MxInt4", tt::DataFormat::MxInt4, DataType::MXINT4, std::nullopt, 544},
        TileSizeCase{"MxInt2", tt::DataFormat::MxInt2, DataType::MXINT2, std::nullopt, 288},
        TileSizeCase{"MxInt8", tt::DataFormat::MxInt8, DataType::MXINT8, std::nullopt, 1056},
        TileSizeCase{"MxFp6", tt::DataFormat::MxFp6P, DataType::MXFP6_E2M3, std::nullopt, 1056},
        TileSizeCase{"RawUInt32", tt::DataFormat::RawUInt32, DataType::UINT32, std::nullopt, 4096},
        TileSizeCase{"UInt8", tt::DataFormat::UInt8, DataType::UINT8, Tile({1, 16}), 16},
        TileSizeCase{"PerNode", tt::DataFormat::Float16_b, DataType::BFLOAT16, std::nullopt, 2048, true}),
    [](const ::testing::TestParamInfo<TileSizeCase>& info) { return info.param.name; });

struct MissingTileSizeCase {
    const char* name;
    const char* declaration;
};

class BindingTokenTileSizeMissing : public test_helpers::ProgramSpecTestBlackhole,
                                    public ::testing::WithParamInterface<MissingTileSizeCase> {
protected:
    distributed::MeshDevice& fixture_device() { return *mesh_device_; }
};

// A tile-size query without resource format metadata fails constant evaluation with a metadata diagnostic.
TEST_P(BindingTokenTileSizeMissing, RequiresMetadata) {
    const std::string source = fmt::format(
        R"(void kernel_main() {{
    {}
    static_assert(get_tile_size(token) > 0);
}})",
        GetParam().declaration);
    const TileSizeCase size{"Default", tt::DataFormat::Float16_b, DataType::BFLOAT16, std::nullopt, 2048};
    ProgramSpec spec = MakeNonPublicTileSizeProgramSpec(size, source);
    EXPECT_THAT(
        [&] { MakeProgramFromSpec(fixture_device(), spec); },
        ::testing::ThrowsMessage<std::exception>(::testing::HasSubstr("data_format_metadata_not_set")));
}

INSTANTIATE_TEST_SUITE_P(
    ResourceSizes,
    BindingTokenTileSizeMissing,
    ::testing::Values(
        MissingTileSizeCase{"Dfb", "constexpr DFBBindingToken<> token{0};"},
        MissingTileSizeCase{"Scratchpad", "constexpr ScratchpadBindingToken token{0, 8192};"},
        MissingTileSizeCase{"Tensor", "constexpr decltype(tensor::local) token{binding_details::LLKMetadata{}};"},
        MissingTileSizeCase{
            "PerNodeTensor",
            "constexpr tensor_accessor::PerNodeTensorBindingToken<0, 0> token{binding_details::LLKMetadata{}};"}),
    [](const ::testing::TestParamInfo<MissingTileSizeCase>& info) { return info.param.name; });

}  // namespace
}  // namespace tt::tt_metal::experimental
