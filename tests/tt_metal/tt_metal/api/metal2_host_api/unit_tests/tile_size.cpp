// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <stdexcept>

#include "impl/data_format/tile_size.hpp"

namespace tt::tt_metal {
namespace {

// Linear tile sizes count stored elements without rounding the entire tile to the L1 alignment.
TEST(TileStorageBytes, LinearStorageSize) {
    const Tile tile({1, 16});
    EXPECT_EQ(detail::tile_size_bytes(tile, tt::DataFormat::Float16_b, 64), 32u);
}

// Block-float exponent storage is padded to the supplied target alignment.
TEST(TileStorageBytes, BlockFloatSixteenByteAlignment) {
    const Tile tile({1, 16});
    EXPECT_EQ(detail::tile_size_bytes(tile, tt::DataFormat::Bfp8_b, 16), 32u);
}

// The same block-float tile can have a different size on a target with a larger alignment.
TEST(TileStorageBytes, BlockFloatSixtyFourByteAlignment) {
    const Tile tile({1, 16});
    EXPECT_EQ(detail::tile_size_bytes(tile, tt::DataFormat::Bfp8_b, 64), 80u);
}

// Custom short faces contribute one block-float exponent byte per face row.
TEST(TileStorageBytes, TwoBitCustomFaces) {
    const Tile tile({2, 32}, {1, 16});
    EXPECT_EQ(detail::tile_size_bytes(tile, tt::DataFormat::Bfp2_b, 16), 32u);
}

// MX storage uses its original host format and pads block scales to the target alignment.
TEST(TileStorageBytes, MxBlockScaleAlignment) {
    const Tile tile;
    EXPECT_EQ(detail::tile_size_bytes(tile, tt::DataFormat::MxInt4, 64), 576u);
}

// Raw storage formats retain their physical element size before hardware format conversion.
TEST(TileStorageBytes, RawUInt32StorageSize) {
    const Tile tile;
    EXPECT_EQ(detail::tile_size_bytes(tile, tt::DataFormat::RawUInt32, 16), 4096u);
}

// TF32 has no supported host Tile storage encoding.
TEST(TileStorageBytes, RejectsTf32Storage) {
    const Tile tile;
    EXPECT_THROW(detail::tile_size_bytes(tile, tt::DataFormat::Tf32, 16), std::invalid_argument);
}

}  // namespace
}  // namespace tt::tt_metal
