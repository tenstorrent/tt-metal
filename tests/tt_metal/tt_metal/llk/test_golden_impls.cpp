// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <fmt/base.h>
#include <cmath>
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/tilize_utils.hpp>
#include <algorithm>
#include <limits>
#include <set>

#include <tt_stl/assert.hpp>
#include "test_golden_impls.hpp"
#include "tests/tt_metal/test_utils/packing.hpp"

using std::vector;

namespace unit_tests::compute {

std::vector<uint32_t> gold_standard_untilize(const std::vector<uint32_t>& src_vec, const GoldenConfig& config) {
    vector<uint32_t> dst_vec;

    int num_tile_rows = config.num_tiles_r_dim;
    int num_tile_cols = config.num_tiles_c_dim;

    // Number of uint32 words per face row: face_c_dim elements × datum_bytes / 4 bytes per uint32
    // BF16 (datum_bytes=2): 16*2/4 = 8  FP8 (datum_bytes=1): 16*1/4 = 4
    int num_c_dim = config.face_c_dim * static_cast<int>(config.datum_bytes) / 4;
    // Standard 16x16 face size in uint32 words
    int face_size = 16 * 16 * static_cast<int>(config.datum_bytes) / 4;
    int tile_size = face_size * (config.tiny_tile ? config.num_faces : 4);

    std::set<int> ind;

    // Iterate over tile rows
    for (int t = 0; t < num_tile_rows; t++) {
        int tile_start_index = t * num_tile_cols;

        int physical_start_for_tile_row = tile_start_index * tile_size;

        // Iterate over tile columns 32 times (naive, but simple for validation)
        uint32_t num_iterations = (config.num_faces > 2) ? 2 : 1;
        for (int x = 0; x < num_iterations; x++) {
            for (int i = 0; i < config.face_r_dim; i++) {  // num rows in a face
                for (int j = 0; j < num_tile_cols; j++) {  // num columns top two faces
                    // Left face row copy
                    for (int k = 0; k < num_c_dim; k++) {
                        int idx = physical_start_for_tile_row + (i * num_c_dim) + k + (j * tile_size);
                        TT_FATAL(!ind.contains(idx), "{}", t);
                        ind.insert(idx);
                        dst_vec.push_back(src_vec.at(idx));
                    }

                    if (config.num_faces > 1) {
                        // Right face row copy
                        for (int k = 0; k < num_c_dim; k++) {
                            int idx = physical_start_for_tile_row + (i * num_c_dim) + k + face_size + (j * tile_size);
                            TT_FATAL(!ind.contains(idx), "{}", t);
                            ind.insert(idx);
                            dst_vec.push_back(src_vec.at(idx));
                        }
                    }
                }
            }

            physical_start_for_tile_row += 2 * face_size;  // Move to bottom faces
        }
    }

    return dst_vec;
}

std::vector<uint32_t> gold_standard_tilize(const std::vector<uint32_t>& src_vec, const GoldenConfig& config) {
    vector<uint32_t> dst_vec;

    int num_rows = config.num_tiles_r_dim * config.face_r_dim * (config.num_faces > 2 ? 2 : 1);
    // Number of uint32 words per row: face_c_dim elements × (faces across) × datum_bytes / 4 bytes per uint32
    // BF16 (datum_bytes=2): (nc*16*2)*2/4 = nc*16   FP8 (datum_bytes=1): (nc*16*2)*1/4 = nc*8
    int num_cols = (config.num_tiles_c_dim * config.face_c_dim * (config.num_faces >= 2 ? 2 : 1)) *
                   static_cast<int>(config.datum_bytes) / 4;
    // Half-face width in uint32 words: face_c_dim/2 elements × datum_bytes / 4 bytes per uint32
    // BF16: 16*2/4 = 8   FP8: 16*1/4 = 4
    const int half_face_w = config.face_c_dim * static_cast<int>(config.datum_bytes) / 4;
    for (int x = 0; x < num_rows; x += 32) {
        for (int y = 0; y < num_cols; y += 2 * half_face_w) {
            int start = (x * num_cols) + y;

            // Top faces
            for (int j = 0; j < 2; j++) {
                int start_ = start + (half_face_w * j);
                for (int k = 0; k < 16; k++) {
                    for (int i = 0; i < half_face_w; i++) {
                        int idx = start_ + (num_cols * k) + i;
                        dst_vec.push_back(src_vec.at(idx));
                    }
                }
            }

            if (config.num_faces > 2) {
                // Bottom faces
                start += 16 * num_cols;
                for (int j = 0; j < 2; j++) {
                    int start_ = start + (half_face_w * j);
                    for (int k = 0; k < 16; k++) {
                        for (int i = 0; i < half_face_w; i++) {
                            int idx = start_ + (num_cols * k) + i;
                            dst_vec.push_back(src_vec.at(idx));
                        }
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
std::vector<uint16_t> gold_transpose_wh(const std::vector<uint16_t>& src_vec, const std::vector<uint32_t>& shape) {
    vector<uint32_t> shapeT{shape[0], shape[1], shape[3], shape[2]};
    TensAddr addr(shape);
    TensAddr addrt(shapeT);

    vector<uint16_t> transposed(src_vec.size());
    for (int n = 0; n < shape[0]; n++) {
        for (int c = 0; c < shape[1]; c++) {
            for (int h = 0; h < shape[2]; h++) {
                for (int w = 0; w < shape[3]; w++) {
                    auto toffs = addrt.offs(n, c, w, h);
                    auto offs = addr.offs(n, c, h, w);
                    TT_FATAL(toffs < transposed.size() && offs < src_vec.size(), "Error");
                    transposed[toffs] = src_vec[offs];
                }
            }
        }
    }

    return transposed;
};

// input shape.x is assumed to have the full number of elements in bfloat16
// src_vec is expected to be untilized
// result is also untilized
std::vector<uint16_t> gold_reduce_h(
    const std::vector<uint16_t>& src_vec,
    const std::vector<uint32_t>& shape,
    float scaler,
    uint8_t red_type,
    bool zeropad) {
    vector<uint32_t> shape_dst{shape[0], shape[1], 1, shape[3]};
    TT_FATAL(shape[2] > 0, "Error");
    if (zeropad) {
        shape_dst[2] = 32;
    }
    TensAddr addr(shape);
    TensAddr addr_dst(shape_dst);

    vector<uint16_t> reduced(addr_dst.numel());
    std::fill(reduced.begin(), reduced.end(), 0);
    for (int n = 0; n < shape[0]; n++) {
        for (int c = 0; c < shape[1]; c++) {
            for (int w = 0; w < shape[3]; w++) {
                // red_type : {SUM, AVG, MAX}; i.e. {0, 1, 2};
                float sum = (red_type == 2) ? -std::numeric_limits<float>::max() : 0.0f;
                for (int h = 0; h < shape[2]; h++) {
                    auto offs = addr.offs(n, c, h, w);
                    if (red_type == 2) {
                        sum = fmaxf(static_cast<float>(std::bit_cast<bfloat16>(src_vec[offs])), sum);
                    } else {
                        sum += static_cast<float>(std::bit_cast<bfloat16>(src_vec[offs]));
                    }
                }
                auto dest_offs = addr_dst.offs(n, c, 0, w);
                reduced[dest_offs] = std::bit_cast<uint16_t>(bfloat16(sum * scaler));
            }
        }
    }

    return reduced;
};

std::vector<uint16_t> gold_reduce_w(
    const vector<uint16_t>& src_vec, const std::vector<uint32_t>& shape, float scaler, uint8_t red_type, bool zeropad) {
    vector<uint32_t> shape_dst{shape[0], shape[1], shape[2], 1};
    if (zeropad) {
        shape_dst[3] = 32;
    }
    TensAddr addr(shape);
    TensAddr addr_dst(shape_dst);

    vector<uint16_t> reduced(addr_dst.numel());
    std::fill(reduced.begin(), reduced.end(), 0);
    for (int n = 0; n < shape[0]; n++) {
        for (int c = 0; c < shape[1]; c++) {
            for (int h = 0; h < shape[2]; h++) {
                // red_type : {SUM, AVG, MAX}; i.e. {0, 1, 2};
                float sum = (red_type == 2) ? -std::numeric_limits<float>::max() : 0.0f;
                for (int w = 0; w < shape[3]; w++) {
                    auto offs = addr.offs(n, c, h, w);
                    if (red_type == 2) {
                        sum = fmaxf(static_cast<float>(std::bit_cast<bfloat16>(src_vec[offs])), sum);
                    } else {
                        sum += static_cast<float>(std::bit_cast<bfloat16>(src_vec[offs]));
                    }
                }
                auto dest_offs = addr_dst.offs(n, c, h, 0);
                reduced[dest_offs] = std::bit_cast<uint16_t>(bfloat16(sum * scaler));
            }
        }
    }
    return reduced;
}

std::vector<uint16_t> gold_reduce_hw(
    const std::vector<uint16_t>& src_vec,
    const std::vector<uint32_t>& shape,
    float scaler,
    uint8_t red_type,
    bool zeropad) {
    vector<uint32_t> shape_dst{shape[0], shape[1], 1, 1};
    if (zeropad) {
        shape_dst[2] = 32;
        shape_dst[3] = 32;
    }
    TensAddr addr(shape);
    TensAddr addr_dst(shape_dst);

    vector<uint16_t> reduced(addr_dst.numel());
    std::fill(reduced.begin(), reduced.end(), 0);
    for (int n = 0; n < shape[0]; n++) {
        for (int c = 0; c < shape[1]; c++) {
            // red_type : {SUM, AVG, MAX}; i.e. {0, 1, 2};
            float sum = (red_type == 2) ? -std::numeric_limits<float>::max() : 0.0f;
            for (int h = 0; h < shape[2]; h++) {
                for (int w = 0; w < shape[3]; w++) {
                    auto offs = addr.offs(n, c, h, w);
                    if (red_type == 2) {
                        sum = fmaxf(static_cast<float>(std::bit_cast<bfloat16>(src_vec[offs])), sum);
                    } else {
                        sum += static_cast<float>(std::bit_cast<bfloat16>(src_vec[offs]));
                    }
                }
            }
            auto dest_offs = addr_dst.offs(n, c, 0, 0);
            reduced[dest_offs] = std::bit_cast<uint16_t>(bfloat16(sum * scaler));
        }
    }

    return reduced;
}

std::vector<uint32_t> gold_standard_tilize_w_elwadd(
    const std::vector<uint32_t>& src0_vec, const std::vector<uint32_t>& src1_vec, const GoldenConfig& config) {
    std::vector<bfloat16> unpacked_tilize_src0_vec =
        tt::test_utils::unpack_vector<bfloat16, uint32_t>(gold_standard_tilize(src0_vec, config));
    std::vector<bfloat16> unpacked_src1_vec = tt::test_utils::unpack_vector<bfloat16, uint32_t>(src1_vec);

    std::vector<bfloat16> result_vec(unpacked_tilize_src0_vec.size());

    std::transform(
        unpacked_tilize_src0_vec.begin(),
        unpacked_tilize_src0_vec.end(),
        unpacked_src1_vec.begin(),
        result_vec.begin(),
        [&](const bfloat16& lhs, const bfloat16& rhs) { return (static_cast<float>(lhs) + static_cast<float>(rhs)); });

    return tt::test_utils::pack_vector<uint32_t, bfloat16>(result_vec);
}

std::vector<uint32_t> gold_standard_pack_rows(const std::vector<uint32_t>& src_vec, const PackRowsConfig& config) {
    // Each row = 16 datums = 8 uint32_t (bfloat16 pairs)
    size_t num_uint32_to_extract = config.num_rows * 8;
    size_t actual_count = std::min(num_uint32_to_extract, src_vec.size());
    vector<uint32_t> dst_vec(actual_count);
    std::copy_n(src_vec.begin(), actual_count, dst_vec.begin());
    return dst_vec;
}

}  // namespace unit_tests::compute
