// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <array>
#include <cstdint>
#include <cstdlib>
#include <stdexcept>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/experimental/mock_device/mock_device.hpp>
#include <tt-metalium/tile.hpp>
#include <tt-metalium/tt_backend_api_types.hpp>
#include "impl/context/metal_context.hpp"

namespace tt::tt_metal {
namespace {

using ::testing::HasSubstr;
using ::testing::ThrowsMessage;

struct KnownTileShape {
    std::array<uint32_t, 2> tile_shape;
    std::array<uint32_t, 2> default_face_shape;
};

// Must stay in sync with TILE_FACE_HW_CHOICES in tt_metal/impl/data_format/tile.cpp.
constexpr KnownTileShape kKnownTileShapes[] = {
    {{{32, 32}}, {{16, 16}}},
    {{{16, 32}}, {{16, 16}}},
    {{{32, 16}}, {{16, 16}}},
    {{{16, 16}}, {{16, 16}}},
    {{{8, 32}}, {{8, 16}}},
    {{{4, 32}}, {{4, 16}}},
    {{{2, 32}}, {{2, 16}}},
    {{{1, 32}}, {{1, 16}}},
    {{{8, 16}}, {{8, 16}}},
    {{{4, 16}}, {{4, 16}}},
    {{{2, 16}}, {{2, 16}}},
    {{{1, 16}}, {{1, 16}}},
};

void ExpectDerivedFields(const Tile& tile, std::array<uint32_t, 2> tile_shape, std::array<uint32_t, 2> face_shape) {
    const uint32_t tile_hw = tile_shape[0] * tile_shape[1];
    const uint32_t face_hw = face_shape[0] * face_shape[1];

    EXPECT_EQ(tile.get_tile_shape(), tile_shape);
    EXPECT_EQ(tile.get_face_shape(), face_shape);
    EXPECT_EQ(tile.get_height(), tile_shape[0]);
    EXPECT_EQ(tile.get_width(), tile_shape[1]);
    EXPECT_EQ(tile.get_tile_hw(), tile_hw);
    EXPECT_EQ(tile.get_face_hw(), face_hw);
    EXPECT_EQ(tile.get_num_faces(), tile_hw / face_hw);
    EXPECT_EQ(tile.get_partial_face(), static_cast<uint32_t>(tile_shape[0] < constants::TILE_HEIGHT));
    EXPECT_EQ(tile.get_narrow_tile(), static_cast<uint32_t>(tile_shape[1] < constants::TILE_WIDTH));
}

TEST(TileConstructor, DefaultIsFull32x32) {
    const Tile tile;
    ExpectDerivedFields(
        tile, {constants::TILE_HEIGHT, constants::TILE_WIDTH}, {constants::FACE_HEIGHT, constants::FACE_WIDTH});
}

TEST(TileConstructor, TileShapeOnlyDerivesDefaultFaceForKnownShapes) {
    for (const auto& known : kKnownTileShapes) {
        const Tile tile(known.tile_shape);
        ExpectDerivedFields(tile, known.tile_shape, known.default_face_shape);
    }
}

TEST(TileConstructor, TileShapeOnlyRejectsUnknownTileShape) {
    constexpr std::array<std::array<uint32_t, 2>, 6> kInvalidShapes = {{
        {3, 32},
        {5, 7},
        {3, 16},
        {0, 32},
        {32, 3},
        {6, 32},
    }};

    for (const auto& shape : kInvalidShapes) {
        EXPECT_THAT(
            [&] { (void)Tile(shape); },
            ThrowsMessage<std::runtime_error>(HasSubstr("Tile size is not valid for our hardware")));
    }
}

TEST(TileConstructor, CustomFaceShapeMatchesTileShapeOnlyWhenFaceIsDefault) {
    for (const auto& known : kKnownTileShapes) {
        const Tile derived(known.tile_shape);
        const Tile explicit_face(known.tile_shape, known.default_face_shape);
        EXPECT_EQ(derived, explicit_face);
        ExpectDerivedFields(explicit_face, known.tile_shape, known.default_face_shape);
    }
}

TEST(TileConstructor, CustomFaceShapeAcceptsFaceThatTilesEvenly) {
    // 16x32 tile with 8x16 faces: 2x2 face grid, within the LLK max of 2 faces per dim.
    constexpr std::array<uint32_t, 2> tile_shape{16, 32};
    constexpr std::array<uint32_t, 2> face_shape{8, 16};
    const Tile tile(tile_shape, face_shape);
    ExpectDerivedFields(tile, tile_shape, face_shape);
}

TEST(TileConstructor, CustomFaceShapeRejectsUnknownTileShape) {
    EXPECT_THAT(
        [] { Tile({3, 32}, {16, 16}); },
        ThrowsMessage<std::runtime_error>(HasSubstr("Tile size is not valid for our hardware")));
}

TEST(TileConstructor, CustomFaceShapeRejectsZeroDimensions) {
    constexpr std::array<Tile::FaceShape, 3> zero_face_shapes = {{{0, 0}, {0, 16}, {16, 0}}};
    for (const auto& face_shape : zero_face_shapes) {
        SCOPED_TRACE(::testing::PrintToString(face_shape));
        EXPECT_THAT([&] { Tile({32, 32}, face_shape); }, ThrowsMessage<std::runtime_error>(HasSubstr("face shape")));
    }
}

TEST(TileConstructor, CustomFaceShapeRejectsFaceThatDoesNotTileEvenly) {
    EXPECT_THAT(
        [] { Tile({32, 32}, {16, 12}); },
        ThrowsMessage<std::runtime_error>(HasSubstr("must tile evenly into face shape")));
}

TEST(TileConstructor, CustomFaceShapeRejectsFaceExceedingMaxDims) {
    EXPECT_THAT(
        [] { Tile({32, 32}, {32, 16}); },
        ThrowsMessage<std::runtime_error>(HasSubstr("exceeds the maximum supported face shape")));
}

TEST(TileConstructor, CustomFaceShapeRejectsTooManyFacesAlongADim) {
    // 32x32 / 8x16 = 4x2 faces; max faces along a dim is 2.
    EXPECT_THAT(
        [] { Tile({32, 32}, {8, 16}); },
        ThrowsMessage<std::runtime_error>(HasSubstr("exceeds the maximum supported num_faces")));
}

// transpose_tile is an extra flag on both constructors, not part of shape validation.
// Both constructors require height 16 or 32 when transpose is enabled.
TEST(TileConstructorTranspose, DefaultsToOff) {
    const Tile tile({32, 32});
    EXPECT_FALSE(tile.get_transpose_within_face());
    EXPECT_FALSE(tile.get_transpose_of_faces());
}

TEST(TileConstructorTranspose, TileShapeOnlyEnablesFlagsWhenHeightIs16Or32) {
    const Tile transposed_full({32, 32}, /*transpose_tile=*/true);
    EXPECT_TRUE(transposed_full.get_transpose_within_face());
    EXPECT_TRUE(transposed_full.get_transpose_of_faces());

    const Tile transposed_half({16, 32}, /*transpose_tile=*/true);
    EXPECT_TRUE(transposed_half.get_transpose_within_face());
    EXPECT_TRUE(transposed_half.get_transpose_of_faces());
}

TEST(TileConstructorTranspose, TileShapeOnlyRejectsHeightOtherThan16Or32) {
    EXPECT_THAT(
        [] { Tile({8, 32}, /*transpose_tile=*/true); },
        ThrowsMessage<std::runtime_error>(HasSubstr("Tile height must equal 16 or 32 in transpose mode")));
}

TEST(TileConstructorTranspose, CustomFaceShapeEnablesFlagsWhenHeightIs16Or32) {
    const Tile tile({16, 32}, {8, 16}, /*transpose_tile=*/true);
    EXPECT_TRUE(tile.get_transpose_within_face());
    EXPECT_TRUE(tile.get_transpose_of_faces());
}

TEST(TileConstructorTranspose, CustomFaceShapeRejectsHeightOtherThan16Or32) {
    EXPECT_THAT(
        [] { Tile({8, 32}, {8, 16}, /*transpose_tile=*/true); },
        ThrowsMessage<std::runtime_error>(HasSubstr("Tile height must equal 16 or 32 in transpose mode")));
}

TEST(TileFromFaceGrid, DefaultFaceProduces32x32) { EXPECT_EQ(Tile::from_face_grid(2, 2), Tile()); }

TEST(TileFromFaceGrid, CustomFaceMatchesEquivalentConstructor) {
    constexpr std::array<uint32_t, 2> face_shape{8, 16};
    const Tile from_grid = Tile::from_face_grid(2, 2, face_shape);
    const Tile from_ctor({16, 32}, face_shape);
    EXPECT_EQ(from_grid, from_ctor);
}

TEST(TileFromFaceGrid, RejectsZeroFaceDimensions) {
    constexpr std::array<Tile::FaceShape, 3> zero_face_shapes = {{{0, 0}, {0, 16}, {16, 0}}};
    for (const auto& face_shape : zero_face_shapes) {
        SCOPED_TRACE(::testing::PrintToString(face_shape));
        EXPECT_THROW((void)Tile::from_face_grid(1, 1, face_shape), std::runtime_error);
    }
}

TEST(TileFromFaceGrid, RejectsGridThatExceedsMaxFaces) {
    // 4x2 faces of 8x16 → 32x32 tile, but 4 faces along a dim exceeds the max of 2.
    EXPECT_THAT(
        [] { (void)Tile::from_face_grid(4, 2, {8, 16}); },
        ThrowsMessage<std::runtime_error>(HasSubstr("exceeds the maximum supported num_faces")));
}

constexpr std::array<DataFormat, 28> kTileSizeFormats = {
    DataFormat::Float32,   DataFormat::Float16,   DataFormat::Bfp8,      DataFormat::Bfp4,   DataFormat::Bfp2,
    DataFormat::Float16_b, DataFormat::Bfp8_b,    DataFormat::Bfp4_b,    DataFormat::Bfp2_b, DataFormat::Lf8,
    DataFormat::Fp8_e4m3,  DataFormat::MxFp4,     DataFormat::MxFp6P,    DataFormat::MxFp6R, DataFormat::MxFp8R,
    DataFormat::MxFp8P,    DataFormat::MxInt8,    DataFormat::MxInt4,    DataFormat::MxInt2, DataFormat::Int8,
    DataFormat::UInt8,     DataFormat::UInt16,    DataFormat::Int16,     DataFormat::Int32,  DataFormat::UInt32,
    DataFormat::RawUInt8,  DataFormat::RawUInt16, DataFormat::RawUInt32,
};

constexpr std::array<DataFormat, 14> kUnpackedTileSizeFormats = {
    DataFormat::Float32,
    DataFormat::Float16,
    DataFormat::Float16_b,
    DataFormat::Lf8,
    DataFormat::Fp8_e4m3,
    DataFormat::Int8,
    DataFormat::UInt8,
    DataFormat::UInt16,
    DataFormat::Int16,
    DataFormat::Int32,
    DataFormat::UInt32,
    DataFormat::RawUInt8,
    DataFormat::RawUInt16,
    DataFormat::RawUInt32,
};

TEST(TileSizeHostOnly, CPU_DoesNotCreateDefaultContext) {
    const auto old_style = GTEST_FLAG_GET(death_test_style);
    GTEST_FLAG_SET(death_test_style, "threadsafe");
    EXPECT_EXIT(
        {
            if (MetalContext::instance_exists()) {
                std::_Exit(10);
            }
            const Tile tile;
            const Tile tiny({16, 32});
            for (const auto format : kTileSizeFormats) {
                if (tile.get_tile_size(format) != tt::tile_size(format)) {
                    std::_Exit(11);
                }
                if (MetalContext::instance_exists()) {
                    std::_Exit(12);
                }
            }
            for (const auto format : kUnpackedTileSizeFormats) {
                if (tiny.get_tile_size(format) != tt::tile_size(format) / 2) {
                    std::_Exit(11);
                }
                if (MetalContext::instance_exists()) {
                    std::_Exit(12);
                }
            }
            constexpr uint32_t kL1Alignment = 16;
            const Tile row({1, 32});
            if (row.get_tile_size(DataFormat::Bfp8_b) != 32 + kL1Alignment ||
                row.get_tile_size(DataFormat::MxInt2) != 8 + kL1Alignment) {
                std::_Exit(11);
            }
            if (MetalContext::instance_exists()) {
                std::_Exit(12);
            }
            std::_Exit(0);
        },
        ::testing::ExitedWithCode(0),
        "");
    GTEST_FLAG_SET(death_test_style, old_style);
}

TEST(TileSizeHostOnly, CPU_SizesWithDefaultContext) {
    const auto old_style = GTEST_FLAG_GET(death_test_style);
    GTEST_FLAG_SET(death_test_style, "threadsafe");
    EXPECT_EXIT(
        {
            if (MetalContext::instance_exists()) {
                std::_Exit(10);
            }
            experimental::configure_mock_mode(tt::ARCH::WORMHOLE_B0, 1);
            const auto& context = MetalContext::instance();
            if (!MetalContext::instance_exists()) {
                std::_Exit(12);
            }
            const uint32_t alignment = context.hal().get_alignment(HalMemType::L1);
            const auto aligned = [alignment](uint32_t n) { return ((n + alignment - 1) / alignment) * alignment; };
            const Tile tiny({1, 32});
            if (tiny.get_tile_size(DataFormat::Bfp8_b) != 32 + aligned(2) ||
                tiny.get_tile_size(DataFormat::MxInt2) != 8 + aligned(1)) {
                std::_Exit(11);
            }
            if (!MetalContext::instance_exists()) {
                std::_Exit(12);
            }
            std::_Exit(0);
        },
        ::testing::ExitedWithCode(0),
        "");
    GTEST_FLAG_SET(death_test_style, old_style);
}

}  // namespace
}  // namespace tt::tt_metal
