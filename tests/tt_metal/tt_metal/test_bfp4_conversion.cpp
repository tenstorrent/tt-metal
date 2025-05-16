// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <functional>
#include <random>

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/bfloat4.hpp>
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/tilize_utils.hpp>
#include "stdio.h"

using namespace tt;

int main(int argc, char** argv) {
    bool pass = true;

    try {
        uint32_t single_bfp4_tile_size = tile_size(tt::DataFormat::Bfp4_b);
        uint32_t num_tiles = 1;
        uint32_t size_in_bytes = num_tiles * single_bfp4_tile_size;

        int num_float_in_tile = 1024;
        int float_data_size = num_tiles * num_float_in_tile;

        std::vector<float> fp32_vec(float_data_size, 0);
        for (int i = 0; i < fp32_vec.size(); i++) {
            fp32_vec.at(i) = i;
        }

        std::vector<uint32_t> shape_vec = {1, num_tiles, 32, 32};
        std::vector<float> tiled_fp32_vec = convert_layout(
            tt::stl::make_const_span(fp32_vec),
            shape_vec,
            TensorLayoutType::LIN_ROW_MAJOR,
            TensorLayoutType::TILED_NFACES);

        std::vector<uint32_t> packed_bfp4b_tile_vec_rm_in =
            pack_fp32_vec_as_bfp4_tiles(fp32_vec, /*row_major_input=*/true, /*is_exp_a=*/false);
        std::vector<float> unpacked_bfp4b_tile_vec_rm_out = unpack_bfp4_tiles_into_float_vec(
            packed_bfp4b_tile_vec_rm_in, /*row_major_output*/ true, /*is_exp_a=*/false);

        std::vector<uint32_t> packed_bfp4b_tile_vec_tile_in =
            pack_fp32_vec_as_bfp4_tiles(tiled_fp32_vec, /*row_major_input=*/false, /*is_exp_a=*/false);
        std::vector<float> unpacked_bfp4b_tile_vec_tile_out = unpack_bfp4_tiles_into_float_vec(
            packed_bfp4b_tile_vec_tile_in, /*row_major_output=*/false, /*is_exp_a=*/false);

        // ////////////////////////////////////////////////////////////////////////////
        // //                      Validation
        // ////////////////////////////////////////////////////////////////////////////
        std::vector<float> tiled_to_rm_fp32_vec = convert_layout(
            tt::stl::make_const_span(unpacked_bfp4b_tile_vec_tile_out),
            shape_vec,
            TensorLayoutType::TILED_NFACES,
            TensorLayoutType::LIN_ROW_MAJOR);
        std::vector<float> rm_to_tiled_fp32_vec = convert_layout(
            tt::stl::make_const_span(unpacked_bfp4b_tile_vec_rm_out),
            shape_vec,
            TensorLayoutType::LIN_ROW_MAJOR,
            TensorLayoutType::TILED_NFACES);

        // Ensure that passing in row_major_input=true and row_major_output=true are inverses of row_major_input=false
        // and row_major_output=false yield the same result
        pass &= (packed_bfp4b_tile_vec_rm_in == packed_bfp4b_tile_vec_tile_in);

        TT_FATAL(unpacked_bfp4b_tile_vec_rm_out.size() == fp32_vec.size(), "Error");
        for (int rm_idx = 0; rm_idx < fp32_vec.size(); rm_idx++) {
            float golden = fp32_vec.at(rm_idx);
            float converted = unpacked_bfp4b_tile_vec_rm_out.at(rm_idx);
            float atol = 8.0f;
            float rtol = 0.15f;
            bool comp = is_close(golden, converted, rtol, atol);
            pass &= comp;
        }

        TT_FATAL(unpacked_bfp4b_tile_vec_tile_out.size() == tiled_fp32_vec.size(), "Error");
        for (int rm_idx = 0; rm_idx < fp32_vec.size(); rm_idx++) {
            float golden = tiled_fp32_vec.at(rm_idx);
            float converted = unpacked_bfp4b_tile_vec_tile_out.at(rm_idx);
            float atol = 8.0f;
            float rtol = 0.15f;
            bool comp = is_close(golden, converted, rtol, atol);
            pass &= comp;
        }

        pass &= (unpacked_bfp4b_tile_vec_rm_out == tiled_to_rm_fp32_vec);
        pass &= (unpacked_bfp4b_tile_vec_tile_out == rm_to_tiled_fp32_vec);

    } catch (const std::exception& e) {
        pass = false;
        // Capture the exception error message
        log_error(LogTest, "{}", e.what());
        // Capture system call errors that may have returned from driver/kernel
        log_error(LogTest, "System error message: {}", std::strerror(errno));
    }

    if (pass) {
        log_info(LogTest, "Test Passed");
    } else {
        TT_THROW("Test Failed");
    }

    TT_FATAL(pass, "Error");

    return 0;
}
