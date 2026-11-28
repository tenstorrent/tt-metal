// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <tuple>
#include <vector>

#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/bfloat2.hpp>
#include <tt-metalium/bfloat4.hpp>
#include <tt-metalium/bfloat8.hpp>
#include <tt-metalium/tt_backend_api_types.hpp>
#include <tt_stl/span.hpp>

namespace {

using tt::DataFormat;

template <typename T>
std::vector<uint32_t> pack_bfp_b_tiles(DataFormat format, ttsl::Span<const T> data, bool row_major) {
    switch (format) {
        case DataFormat::Bfp8_b: return pack_as_bfp8_tiles(data, row_major, /*is_exp_a=*/false);
        case DataFormat::Bfp4_b: return pack_as_bfp4_tiles(data, row_major, /*is_exp_a=*/false);
        default: return ::pack_as_bfp2_tiles(data, row_major, /*is_exp_a=*/false);
    }
}

std::vector<float> unpack_bfp_b_tiles(DataFormat format, ttsl::Span<const uint32_t> packed, bool row_major) {
    switch (format) {
        case DataFormat::Bfp8_b: return unpack_bfp8_tiles_into_float_vec(packed, row_major, /*is_exp_a=*/false);
        case DataFormat::Bfp4_b: return unpack_bfp4_tiles_into_float_vec(packed, row_major, /*is_exp_a=*/false);
        default: return ::unpack_bfp2_tiles_into_float_vec(packed, row_major, /*is_exp_a=*/false);
    }
}

// One tile of distinct values per sentinel position, so a wrapped offset that lands on another sentinel still fails.
template <typename T>
std::vector<T> large_offset_sentinel_tile(size_t tile_elements, int seed) {
    std::vector<T> tile(tile_elements);
    for (size_t i = 0; i < tile_elements; ++i) {
        tile[i] = static_cast<T>(static_cast<float>(((i + seed) % 7) * (seed + 1)));
    }
    return tile;
}

// Tensors above 2^32 elements: pack and unpack must use 64-bit tile and element offsets.
template <typename T>
void check_offsets_above_uint32_max(DataFormat format, bool row_major) {
    constexpr size_t tile_elements = 1024;
    constexpr size_t boundary_tile = (size_t{1} << 32) / tile_elements;
    constexpr size_t num_tiles = boundary_tile + 2;
    const std::array<size_t, 3> sentinel_tiles{0, boundary_tile, boundary_tile + 1};

    std::vector<T> input(num_tiles * tile_elements, static_cast<T>(0.0f));
    std::vector<std::vector<uint32_t>> expected_packed;
    std::vector<std::vector<float>> expected_decoded;
    for (size_t s = 0; s < sentinel_tiles.size(); ++s) {
        const auto tile = large_offset_sentinel_tile<T>(tile_elements, static_cast<int>(s) + 1);
        std::copy(tile.begin(), tile.end(), input.begin() + sentinel_tiles[s] * tile_elements);
        expected_packed.push_back(pack_bfp_b_tiles<T>(format, ttsl::make_const_span(tile), row_major));
        expected_decoded.push_back(unpack_bfp_b_tiles(format, expected_packed.back(), row_major));
    }
    const size_t words_per_tile = expected_packed[0].size();

    const auto packed = pack_bfp_b_tiles<T>(format, ttsl::make_const_span(input), row_major);
    ASSERT_EQ(packed.size(), num_tiles * words_per_tile);
    for (size_t s = 0; s < sentinel_tiles.size(); ++s) {
        EXPECT_TRUE(std::equal(
            expected_packed[s].begin(), expected_packed[s].end(), packed.begin() + sentinel_tiles[s] * words_per_tile))
            << "packed tile " << sentinel_tiles[s];
    }
    EXPECT_TRUE(std::all_of(
        packed.begin() + words_per_tile, packed.begin() + boundary_tile * words_per_tile, [](uint32_t word) {
            return word == 0;
        }));

    // Release the input before the 16 GiB decode.
    std::vector<T>().swap(input);
    const auto decoded = unpack_bfp_b_tiles(format, packed, row_major);
    ASSERT_EQ(decoded.size(), num_tiles * tile_elements);
    for (size_t s = 0; s < sentinel_tiles.size(); ++s) {
        EXPECT_TRUE(std::equal(
            expected_decoded[s].begin(),
            expected_decoded[s].end(),
            decoded.begin() + sentinel_tiles[s] * tile_elements))
            << "decoded tile " << sentinel_tiles[s];
    }
    EXPECT_TRUE(
        std::all_of(decoded.begin() + tile_elements, decoded.begin() + boundary_tile * tile_elements, [](float value) {
            return value == 0.0f;
        }));
}

class LargeBfpConversion : public ::testing::TestWithParam<std::tuple<DataFormat, bool, bool>> {};

}  // namespace

// bfloat16 input takes the vectorized pack, uint8 input the scalar pack. Uses about 21 GiB of host RAM, so it is
// opt-in: --gtest_also_run_disabled_tests --gtest_filter='*LargeBfp*'.
TEST_P(LargeBfpConversion, DISABLED_OffsetsAboveUint32Max) {
    const auto [format, row_major, uint8_input] = GetParam();
    if (uint8_input) {
        check_offsets_above_uint32_max<uint8_t>(format, row_major);
    } else {
        check_offsets_above_uint32_max<bfloat16>(format, row_major);
    }
}

INSTANTIATE_TEST_SUITE_P(
    HostOnlyTest,
    LargeBfpConversion,
    ::testing::Combine(
        ::testing::Values(DataFormat::Bfp8_b, DataFormat::Bfp4_b, DataFormat::Bfp2_b),
        ::testing::Bool(),
        ::testing::Bool()),
    [](const ::testing::TestParamInfo<LargeBfpConversion::ParamType>& info) {
        const DataFormat format = std::get<0>(info.param);
        const bool row_major = std::get<1>(info.param);
        const bool uint8_input = std::get<2>(info.param);
        const std::string name = format == DataFormat::Bfp8_b   ? "Bfp8b"
                                 : format == DataFormat::Bfp4_b ? "Bfp4b"
                                                                : "Bfp2b";
        return name + (row_major ? "RowMajor" : "Tiled") + (uint8_input ? "Uint8" : "Bfloat16");
    });
