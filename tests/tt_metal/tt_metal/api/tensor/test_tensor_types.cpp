// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Sanity tests for DataType-related free functions declared in tensor_types.hpp.

#include <gtest/gtest.h>

#include <utility>

#include <tt-metalium/tensor/tensor_types.hpp>
#include <tt-metalium/tt_backend_api_types.hpp>

namespace tt::tt_metal {
namespace {

// tt::tt_metal::tile_size(DataType) must return the same value as tt::tile_size(DataFormat) once the DataType is
// converted via datatype_to_dataformat_converter — it is implemented exactly that way, but this test pins down the
// contract so future changes don't silently drift.
TEST(TensorTypesTileSizeTest, MatchesDataFormatTileSize) {
    constexpr DataType kDataTypes[] = {
        DataType::BFLOAT16,
        DataType::FLOAT32,
        DataType::UINT32,
        DataType::BFLOAT8_B,
        DataType::BFLOAT4_B,
        DataType::UINT8,
        DataType::UINT16,
        DataType::INT32,
        DataType::INT8,
        DataType::MXFP8_E4M3,
        DataType::MXFP8_E5M2,
        DataType::MXFP6_E2M3,
        DataType::MXFP6_E3M2,
        DataType::MXFP4,
        DataType::MXINT8,
        DataType::MXINT4,
        DataType::MXINT2,
    };

    for (DataType dtype : kDataTypes) {
        const tt::DataFormat format = datatype_to_dataformat_converter(dtype);
        EXPECT_EQ(tt::tt_metal::tile_size(dtype), tt::tile_size(format))
            << "tile_size mismatch for DataType=" << static_cast<int>(dtype)
            << ", DataFormat=" << static_cast<int>(format);
    }
}

TEST(TensorTypesTileSizeTest, InvalidDataTypeThrows) {
    EXPECT_ANY_THROW((void)tt::tt_metal::tile_size(DataType::INVALID));
}

// Each MX DataType maps to its own MX DataFormat and back, and only MX types satisfy is_mx().
TEST(TensorTypesMxTest, DataFormatRoundTripAndPredicates) {
    const std::pair<DataType, tt::DataFormat> kMxPairs[] = {
        {DataType::MXFP8_E4M3, tt::DataFormat::MxFp8P},
        {DataType::MXFP8_E5M2, tt::DataFormat::MxFp8R},
        {DataType::MXFP6_E2M3, tt::DataFormat::MxFp6P},
        {DataType::MXFP6_E3M2, tt::DataFormat::MxFp6R},
        {DataType::MXFP4, tt::DataFormat::MxFp4},
        {DataType::MXINT8, tt::DataFormat::MxInt8},
        {DataType::MXINT4, tt::DataFormat::MxInt4},
        {DataType::MXINT2, tt::DataFormat::MxInt2},
    };

    for (const auto& [dtype, format] : kMxPairs) {
        EXPECT_EQ(datatype_to_dataformat_converter(dtype), format) << dtype;
        EXPECT_EQ(dataformat_to_datatype_converter(format), dtype) << dtype;
        EXPECT_TRUE(is_mx(dtype)) << dtype;
        EXPECT_TRUE(is_floating_point(dtype)) << dtype;
        EXPECT_FALSE(is_block_float(dtype)) << dtype;
    }

    EXPECT_FALSE(is_mx(DataType::BFLOAT8_B));
    EXPECT_FALSE(is_mx(DataType::BFLOAT16));
    EXPECT_FALSE(is_mx(DataType::INVALID));
}

}  // namespace
}  // namespace tt::tt_metal
