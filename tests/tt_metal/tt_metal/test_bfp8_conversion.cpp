// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <cstdint>
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
        tt::stl::make_const_span(fp32_vec), shape_vec, TensorLayoutType::LIN_ROW_MAJOR, TensorLayoutType::TILED_NFACES);

    std::vector<uint32_t> packed_bfp8b_tile_vec_rm_in =
        pack_as_bfp8_tiles(tt::stl::make_const_span(fp32_vec), /*row_major_input=*/true, /*is_exp_a=*/false);
    std::vector<float> unpacked_bfp8b_tile_vec_rm_out =
        unpack_bfp8_tiles_into_float_vec(packed_bfp8b_tile_vec_rm_in, /*row_major_output*/ true, /*is_exp_a=*/false);

    std::vector<uint32_t> packed_bfp8b_tile_vec_tile_in =
        pack_as_bfp8_tiles(tt::stl::make_const_span(tiled_fp32_vec), /*row_major_input=*/false, /*is_exp_a=*/false);
    std::vector<float> unpacked_bfp8b_tile_vec_tile_out =
        unpack_bfp8_tiles_into_float_vec(packed_bfp8b_tile_vec_tile_in, /*row_major_output=*/false, /*is_exp_a=*/false);

    // Validation
    std::vector<float> tiled_to_rm_fp32_vec = convert_layout(
        tt::stl::make_const_span(unpacked_bfp8b_tile_vec_tile_out),
        shape_vec,
        TensorLayoutType::TILED_NFACES,
        TensorLayoutType::LIN_ROW_MAJOR);
    std::vector<float> rm_to_tiled_fp32_vec = convert_layout(
        tt::stl::make_const_span(unpacked_bfp8b_tile_vec_rm_out),
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
