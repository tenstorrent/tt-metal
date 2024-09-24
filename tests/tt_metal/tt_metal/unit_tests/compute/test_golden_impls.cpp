// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>


#include "test_golden_impls.hpp"
#include "common/test_tiles.hpp"
#include "common/bfloat16.hpp"
#include "tt_metal/host_api.hpp"
#include "tt_metal/detail/tt_metal.hpp"
#include "tests/tt_metal/test_utils/packing.hpp"


namespace unit_tests::compute {

std::vector<uint32_t> gold_standard_untilize(const std::vector<uint32_t> &src_vec, const GoldenConfig &config) {
    vector<uint32_t> dst_vec;

    int num_rows = config.num_tiles_r_dim * config.face_r_dim * (config.num_faces > 2 ? 2: 1);
    //Due to each element being 32 bits, for bfloat16 thats 2 elements
    int num_cols = (config.num_tiles_c_dim * config.face_c_dim *  (config.num_faces >= 2 ? 2: 1)) / 2;

    int num_tile_rows = config.num_tiles_r_dim;
    int num_tile_cols = config.num_tiles_c_dim;

    //Due to each element being 32 bits, for bfloat16 thats 2 elements
    int num_c_dim = config.face_c_dim / 2;
    //Untilize outputs correct number of r_dim & num_faces
    //But assumes increments are still default 16x16 faces
    int face_size = 16 * 8;
    int tile_size = face_size * 4;

    std::set<int> ind;

    // Iterate over tile rows
    for (int t = 0; t < num_tile_rows; t++) {

        int tile_start_index = t * num_tile_cols;

        int physical_start_for_tile_row = tile_start_index * 32 * 16;

        // Iterate over tile columns 32 times (naive, but simple for validation)
        uint32_t num_iterations = (config.num_faces > 2) ? 2 : 1;
        for (int x = 0; x < num_iterations; x++) {
            for (int i = 0; i < config.face_r_dim; i++) { // num rows in a face
                for (int j = 0; j < num_tile_cols; j++) { // num columns top two faces
                    // Left face row copy
                    for (int k = 0; k < num_c_dim; k++) {
                        int idx = physical_start_for_tile_row + i * num_c_dim + k + j * tile_size;
                        TT_FATAL(ind.find(idx) == ind.end(), "{}", t);
                        ind.insert(idx);
                        dst_vec.push_back(src_vec.at(idx));
                    }

                    if(config.num_faces > 1) {
                        // Right face row copy
                        for (int k = 0; k < num_c_dim; k++) {
                            int idx = physical_start_for_tile_row + i * num_c_dim + k + face_size + j * tile_size;
                            TT_FATAL(ind.find(idx) == ind.end(), "{}", t);
                            ind.insert(idx);
                            dst_vec.push_back(src_vec.at(idx));
                        }
                    }
                }
            }

            physical_start_for_tile_row += 2 * face_size; // Move to bottom faces
        }
    }

    return dst_vec;
}

std::vector<uint32_t> gold_standard_tilize(const std::vector<uint32_t> &src_vec, const GoldenConfig &config) {
    vector<uint32_t> dst_vec;

    //TODO: RT update this one to use variable tile sizes
    int num_rows = config.num_tiles_r_dim * config.face_r_dim * (config.num_faces > 2 ? 2: 1);
    //Due to each element being 32 bits, for bfloat16 thats 2 elements
    int num_cols = (config.num_tiles_c_dim * config.face_c_dim *  (config.num_faces >= 2 ? 2: 1)) / 2;
    for (int x = 0; x < num_rows; x += 32) {
        for (int y = 0; y < num_cols; y += 16) {
            int start = x * num_cols + y;

            // Top faces
            for (int j = 0; j < 2; j++) {
                int start_ = start + 8 * j;
                for (int k = 0; k < 16; k++) {
                    for (int i = 0; i < 8; i++) {
                        int idx = start_ + num_cols * k + i;
                        dst_vec.push_back(src_vec.at(idx));
                    }
                }
            }

            // Bottom faces
            start += 16 * num_cols;
            for (int j = 0; j < 2; j++) {
                int start_ = start + 8 * j;
                for (int k = 0; k < 16; k++) {
                    for (int i = 0; i < 8; i++) {
                        int idx = start_ + num_cols * k + i;
                        dst_vec.push_back(src_vec.at(idx));
                    }
                }
            }
        }
    }

    return dst_vec;
}

// input shape.x is assumed to have the full number of elements in bfloat16
// src_vec is expected to be untilized
// result is also untilized
std::vector<uint16_t> gold_transpose_wh(const std::vector<uint16_t> &src_vec, const std::vector<uint32_t> &shape) {
    vector<uint32_t> shapeT{shape[0], shape[1], shape[3], shape[2]};
    TensAddr addr(shape);
    TensAddr addrt(shapeT);

    vector<uint16_t> transposed(src_vec.size());
    for (int n = 0; n < shape[0]; n++)
    for (int c = 0; c < shape[1]; c++)
    for (int h = 0; h < shape[2]; h++)
    for (int w = 0; w < shape[3]; w++) {
        auto toffs = addrt.offs(n, c, w, h);
        auto offs = addr.offs(n, c, h, w);
        TT_FATAL(toffs < transposed.size() && offs < src_vec.size(), "Error");
        transposed[toffs] = src_vec[offs];
    }

    return transposed;
};

// input shape.x is assumed to have the full number of elements in bfloat16
// src_vec is expected to be untilized
// result is also untilized
std::vector<uint16_t> gold_reduce_h(const std::vector<uint16_t> &src_vec, const std::vector<uint32_t> &shape, float scaler, uint8_t red_type, bool zeropad) {
    vector<uint32_t> shape_dst{shape[0], shape[1], 1, shape[3]};
    TT_FATAL(shape[2] > 0, "Error");
    if (zeropad)
        shape_dst[2] = 32;
    TensAddr addr(shape);
    TensAddr addr_dst(shape_dst);

    vector<uint16_t> reduced(addr_dst.numel());
    std::fill(reduced.begin(), reduced.end(), 0);
    for (int n = 0; n < shape[0]; n++)
    for (int c = 0; c < shape[1]; c++)
    for (int w = 0; w < shape[3]; w++) {
        // red_type : {SUM, AVG, MAX}; i.e. {0, 1, 2};
        float sum = (red_type == 2) ? -std::numeric_limits<float>::max() : 0.0f;
        for (int h = 0; h < shape[2]; h++) {
            auto offs = addr.offs(n, c, h, w);
            if (red_type == 2)
                sum = fmaxf(bfloat16(src_vec[offs]).to_float(), sum);
            else
                sum += bfloat16(src_vec[offs]).to_float();
        }
        auto dest_offs = addr_dst.offs(n, c, 0, w);
        reduced[dest_offs] = bfloat16(sum*scaler).to_uint16();
    }

    return reduced;
};

std::vector<uint16_t> gold_reduce_w(const vector<uint16_t> &src_vec, const std::vector<uint32_t> &shape, float scaler, uint8_t red_type, bool zeropad) {
    vector<uint32_t> shape_dst{shape[0], shape[1], shape[2], 1};
    if (zeropad)
        shape_dst[3] = 32;
    TensAddr addr(shape);
    TensAddr addr_dst(shape_dst);

    vector<uint16_t> reduced(addr_dst.numel());
    std::fill(reduced.begin(), reduced.end(), 0);
    for (int n = 0; n < shape[0]; n++)
    for (int c = 0; c < shape[1]; c++)
    for (int h = 0; h < shape[2]; h++) {
        // red_type : {SUM, AVG, MAX}; i.e. {0, 1, 2};
        float sum = (red_type == 2) ? -std::numeric_limits<float>::max() : 0.0f;
        for (int w = 0; w < shape[3]; w++) {
            auto offs = addr.offs(n, c, h, w);
            if (red_type == 2)
                sum = fmaxf(bfloat16(src_vec[offs]).to_float(), sum);
            else
                sum += bfloat16(src_vec[offs]).to_float();
        }
        auto dest_offs = addr_dst.offs(n, c, h, 0);
        reduced[dest_offs] = bfloat16(sum*scaler).to_uint16();
    }
    return reduced;
}

std::vector<uint16_t> gold_reduce_hw(const std::vector<uint16_t> &src_vec, const std::vector<uint32_t> &shape, float scaler, uint8_t red_type, bool zeropad) {
    vector<uint32_t> shape_dst{shape[0], shape[1], 1, 1};
    if (zeropad) {
        shape_dst[2] = 32;
        shape_dst[3] = 32;
    }
    TensAddr addr(shape);
    TensAddr addr_dst(shape_dst);

    vector<uint16_t> reduced(addr_dst.numel());
    std::fill(reduced.begin(), reduced.end(), 0);
    for (int n = 0; n < shape[0]; n++)
    for (int c = 0; c < shape[1]; c++) {
        // red_type : {SUM, AVG, MAX}; i.e. {0, 1, 2};
        float sum = (red_type == 2) ? -std::numeric_limits<float>::max() : 0.0f;
        for (int h = 0; h < shape[2]; h++) {
            for (int w = 0; w < shape[3]; w++) {
                auto offs = addr.offs(n, c, h, w);
                if (red_type == 2)
                    sum = fmaxf(bfloat16(src_vec[offs]).to_float(), sum);
                else
                    sum += bfloat16(src_vec[offs]).to_float();
            }
        }
        auto dest_offs = addr_dst.offs(n, c, 0, 0);
        reduced[dest_offs] = bfloat16(sum*scaler).to_uint16();
    }

    return reduced;
}

std::vector<uint32_t> gold_standard_tilize_w_elwadd(const std::vector<uint32_t> &src0_vec, const std::vector<uint32_t> &src1_vec, const GoldenConfig &config) {

    std::vector<bfloat16> unpacked_tilize_src0_vec = tt::test_utils::unpack_vector<bfloat16, uint32_t>(gold_standard_tilize(src0_vec, config));
    std::vector<bfloat16> unpacked_src1_vec = tt::test_utils::unpack_vector<bfloat16, uint32_t>(src1_vec);

    std::vector<bfloat16> result_vec(unpacked_tilize_src0_vec.size());

    std::transform(
        unpacked_tilize_src0_vec.begin(),
        unpacked_tilize_src0_vec.end(),
        unpacked_src1_vec.begin(),
        result_vec.begin(),
        [&](const bfloat16& lhs, const bfloat16& rhs) {
            return (lhs.to_float() + rhs.to_float());
        });

    return tt::test_utils::pack_vector<uint32_t, bfloat16>(result_vec);
}


}   // unit_tests::compute
