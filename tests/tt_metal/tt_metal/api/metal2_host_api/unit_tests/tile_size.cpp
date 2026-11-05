// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <stdexcept>

#include "impl/data_format/tile_size.hpp"
#include "impl/metal2_host_api/llk_metadata.hpp"

namespace tt::tt_metal {
namespace {

// Linear tile sizes count stored elements without rounding the entire tile to the L1 alignment.
TEST(TileStorageBytes, CPU_LinearStorageSize) {
    const Tile tile({1, 16});
    EXPECT_EQ(detail::tile_size_bytes(tile, tt::DataFormat::Float16_b, 64), 32u);
}

// Block-float exponent storage is padded to the supplied target alignment.
TEST(TileStorageBytes, CPU_BlockFloatSixteenByteAlignment) {
    const Tile tile({1, 16});
    EXPECT_EQ(detail::tile_size_bytes(tile, tt::DataFormat::Bfp8_b, 16), 32u);
}

// The same block-float tile can have a different size on a target with a larger alignment.
TEST(TileStorageBytes, CPU_BlockFloatSixtyFourByteAlignment) {
    const Tile tile({1, 16});
    EXPECT_EQ(detail::tile_size_bytes(tile, tt::DataFormat::Bfp8_b, 64), 80u);
}

// Custom short faces contribute one block-float exponent byte per face row.
TEST(TileStorageBytes, CPU_TwoBitCustomFaces) {
    const Tile tile({2, 32}, {1, 16});
    EXPECT_EQ(detail::tile_size_bytes(tile, tt::DataFormat::Bfp2_b, 16), 32u);
}

// MX storage uses its original host format and pads block scales to the target alignment.
TEST(TileStorageBytes, CPU_MxBlockScaleAlignment) {
    const Tile tile;
    EXPECT_EQ(detail::tile_size_bytes(tile, tt::DataFormat::MxInt4, 64), 576u);
}

// Raw storage formats retain their physical element size before hardware format conversion.
TEST(TileStorageBytes, CPU_RawUInt32StorageSize) {
    const Tile tile;
    EXPECT_EQ(detail::tile_size_bytes(tile, tt::DataFormat::RawUInt32, 16), 4096u);
}

// TF32 has no supported host Tile storage encoding.
TEST(TileStorageBytes, CPU_RejectsTf32Storage) {
    const Tile tile;
    EXPECT_THROW(detail::tile_size_bytes(tile, tt::DataFormat::Tf32, 16), std::invalid_argument);
}

// Binding metadata resolves tile bytes with the supplied target alignment.
TEST(TileStorageBytes, CPU_BindingMetadataUsesTargetAlignment) {
    const LLKMetadata metadata = make_llk_metadata(tt::DataFormat::Bfp8_b, Tile({1, 16}), 64);
    EXPECT_EQ(metadata.tile_size_bytes, 80u);
}

// Binding metadata preserves TF32 format queries without inventing an unsupported storage size.
TEST(TileStorageBytes, CPU_Tf32BindingHasNoStorageSize) {
    const LLKMetadata metadata = make_llk_metadata(tt::DataFormat::Tf32, Tile{}, 16);
    EXPECT_EQ(metadata.tile_size_bytes, 0u);
}

}  // namespace
}  // namespace tt::tt_metal
