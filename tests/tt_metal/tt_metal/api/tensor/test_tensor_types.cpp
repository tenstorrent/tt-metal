// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Sanity tests for DataType-related free functions declared in tensor_types.hpp.

#include <gtest/gtest.h>

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

TEST(TensorTypesIsDataTypeSupportedTest, MatchesDataFormatSupport) {
    constexpr DataType kDataTypes[] = {
        DataType::BFLOAT16,
        DataType::FLOAT32,
        DataType::UINT32,
        DataType::BFLOAT8_B,
        DataType::BFLOAT4_B,
        DataType::UINT8,
        DataType::UINT16,
        DataType::INT32,
        DataType::FP8_E4M3,
        DataType::INT8,
    };
    constexpr tt::ARCH kArchs[] = {tt::ARCH::WORMHOLE_B0, tt::ARCH::BLACKHOLE, tt::ARCH::QUASAR};

    for (tt::ARCH arch : kArchs) {
        for (DataType dtype : kDataTypes) {
            EXPECT_EQ(
                is_data_type_supported(dtype, arch),
                tt::is_data_format_supported(datatype_to_dataformat_converter(dtype), arch))
                << "DataType=" << static_cast<int>(dtype) << ", arch=" << static_cast<int>(arch);
        }
        EXPECT_FALSE(is_data_type_supported(DataType::INVALID, arch));
    }
    EXPECT_TRUE(is_data_type_supported(DataType::BFLOAT16, tt::ARCH::WORMHOLE_B0));
    EXPECT_FALSE(is_data_type_supported(DataType::FP8_E4M3, tt::ARCH::WORMHOLE_B0));
}

}  // namespace
}  // namespace tt::tt_metal
