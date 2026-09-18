// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <random>
#include <cstdint>
#include <tt-metalium/bfloat4.hpp>
#include <vector>

#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/bfloat8.hpp>
#include <tt-metalium/tilize_utils.hpp>
#include <tt_stl/span.hpp>
#include <tt-metalium/tt_backend_api_types.hpp>

using namespace tt;

TEST(HostOnlyTest, Bfp8Conversion) {
    uint32_t num_tiles = 1;

    int num_float_in_tile = 1024;
    int float_data_size = num_tiles * num_float_in_tile;

    std::vector<float> fp32_vec(float_data_size, 0);
    for (size_t i = 0; i < fp32_vec.size(); i++) {
        fp32_vec.at(i) = static_cast<float>(i);
    }

    std::vector<uint32_t> shape_vec = {1, 1, 32, 32};
    std::vector<float> tiled_fp32_vec = convert_layout(
        ttsl::make_const_span(fp32_vec), shape_vec, TensorLayoutType::LIN_ROW_MAJOR, TensorLayoutType::TILED_NFACES);

    std::vector<uint32_t> packed_bfp8b_tile_vec_rm_in =
        pack_as_bfp8_tiles(ttsl::make_const_span(fp32_vec), /*row_major_input=*/true, /*is_exp_a=*/false);
    std::vector<float> unpacked_bfp8b_tile_vec_rm_out =
        unpack_bfp8_tiles_into_float_vec(packed_bfp8b_tile_vec_rm_in, /*row_major_output*/ true, /*is_exp_a=*/false);

    std::vector<uint32_t> packed_bfp8b_tile_vec_tile_in =
        pack_as_bfp8_tiles(ttsl::make_const_span(tiled_fp32_vec), /*row_major_input=*/false, /*is_exp_a=*/false);
    std::vector<float> unpacked_bfp8b_tile_vec_tile_out =
        unpack_bfp8_tiles_into_float_vec(packed_bfp8b_tile_vec_tile_in, /*row_major_output=*/false, /*is_exp_a=*/false);

    // Validation
    std::vector<float> tiled_to_rm_fp32_vec = convert_layout(
        ttsl::make_const_span(unpacked_bfp8b_tile_vec_tile_out),
        shape_vec,
        TensorLayoutType::TILED_NFACES,
        TensorLayoutType::LIN_ROW_MAJOR);
    std::vector<float> rm_to_tiled_fp32_vec = convert_layout(
        ttsl::make_const_span(unpacked_bfp8b_tile_vec_rm_out),
        shape_vec,
        TensorLayoutType::LIN_ROW_MAJOR,
        TensorLayoutType::TILED_NFACES);

    // Ensure that passing in row_major_input=true and row_major_output=true are inverses of row_major_input=false
    // and row_major_output=false yield the same result
    EXPECT_EQ(packed_bfp8b_tile_vec_rm_in, packed_bfp8b_tile_vec_tile_in);

    ASSERT_EQ(unpacked_bfp8b_tile_vec_rm_out.size(), fp32_vec.size());
    for (size_t rm_idx = 0; rm_idx < fp32_vec.size(); rm_idx++) {
        float golden = fp32_vec.at(rm_idx);
        float converted = unpacked_bfp8b_tile_vec_rm_out.at(rm_idx);
        float atol = 8.0f;
        float rtol = 0.01f;
        EXPECT_TRUE(is_close(golden, converted, rtol, atol))
            << "Mismatch at index " << rm_idx << ": golden=" << golden << ", converted=" << converted;
    }

    ASSERT_EQ(unpacked_bfp8b_tile_vec_tile_out.size(), tiled_fp32_vec.size());
    for (size_t rm_idx = 0; rm_idx < fp32_vec.size(); rm_idx++) {
        float golden = tiled_fp32_vec.at(rm_idx);
        float converted = unpacked_bfp8b_tile_vec_tile_out.at(rm_idx);
        float atol = 8.0f;
        float rtol = 0.01f;
        EXPECT_TRUE(is_close(golden, converted, rtol, atol))
            << "Mismatch at index " << rm_idx << ": golden=" << golden << ", converted=" << converted;
    }

    EXPECT_EQ(unpacked_bfp8b_tile_vec_rm_out, tiled_to_rm_fp32_vec);
    EXPECT_EQ(unpacked_bfp8b_tile_vec_tile_out, rm_to_tiled_fp32_vec);
}

namespace {

template <bool Bfp4>
auto pack_search(const std::vector<float>& values, bool optimize, bool row_major = false) {
    if constexpr (Bfp4) {
        return pack_as_bfp4_tiles(ttsl::make_const_span(values), row_major, false, std::nullopt, optimize);
    } else {
        return pack_as_bfp8_tiles(ttsl::make_const_span(values), row_major, false, std::nullopt, optimize);
    }
}

template <bool Bfp4>
auto unpack_search(const std::vector<uint32_t>& values) {
    if constexpr (Bfp4) {
        return unpack_bfp4_tiles_into_float_vec(values, false, false);
    } else {
        return unpack_bfp8_tiles_into_float_vec(values, false, false);
    }
}

template <bool Bfp4>
void check_exponent_search() {
    // Tile-face order: every 16 consecutive elements share an exponent.
    std::vector<float> values(1024, Bfp4 ? 0.13f : 0.008f);
    for (size_t i = 0; i < values.size(); i += 16) {
        values[i] = 1.0f;
    }
    const auto original = values;
    auto packed = pack_search<Bfp4>(values, true);
    auto optimized = unpack_search<Bfp4>(packed);
    EXPECT_EQ(values, original);
    // The exponent section comes first: Emax=127, and every row should select 126.
    EXPECT_EQ(packed[0], 0x7e7e7e7eu);
    EXPECT_EQ(optimized[0], Bfp4 ? 0.875f : 0.9921875f);
    EXPECT_EQ(optimized[1], Bfp4 ? 0.125f : 0.0078125f);
    EXPECT_EQ(pack_search<Bfp4>(values, false)[0], 0x7f7f7f7fu);
    // All identical groups make row-major and tile-face input equivalent.
    EXPECT_EQ(packed, pack_search<Bfp4>(values, true, true));

    // Signed random values over a wide dynamic range; compare actual unpacked
    // values per physical group, not merely an aggregate tensor MSE.
    std::mt19937 rng(42);
    std::normal_distribution<float> normal;
    for (size_t i = 0; i < values.size(); ++i) {
        values[i] = std::ldexp(normal(rng), int((i / 16) % 31) - 15);
    }
    auto ordinary = unpack_search<Bfp4>(pack_search<Bfp4>(values, false));
    optimized = unpack_search<Bfp4>(pack_search<Bfp4>(values, true));
    for (size_t i = 0; i < values.size(); i += 16) {
        double old_error = 0, new_error = 0;
        for (size_t j = i; j < i + 16; ++j) {
            old_error += std::pow(double(values[j]) - ordinary[j], 2);
            new_error += std::pow(double(values[j]) - optimized[j], 2);
        }
        EXPECT_LE(new_error, old_error);
    }

    // Exact Emax values, zero/denormal blocks and special values retain legacy bits.
    for (float value :
         {0.0f,
          -0.0f,
          1.0f,
          -1.0f,
          std::numeric_limits<float>::denorm_min(),
          std::numeric_limits<float>::min(),
          std::numeric_limits<float>::infinity(),
          -std::numeric_limits<float>::infinity(),
          std::numeric_limits<float>::quiet_NaN()}) {
        std::fill(values.begin(), values.end(), value);
        EXPECT_EQ(pack_search<Bfp4>(values, true), pack_search<Bfp4>(values, false));
    }
}

}  // namespace

TEST(HostOnlyTest, Bfp4ExponentSearch) { check_exponent_search<true>(); }
TEST(HostOnlyTest, Bfp8ExponentSearch) { check_exponent_search<false>(); }
