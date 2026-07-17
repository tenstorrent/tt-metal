// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
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
    std::vector<uint32_t> dst_vec;

    int num_rows = config.num_tiles_r_dim * config.face_r_dim * (config.num_faces > 2 ? 2 : 1);
    // Number of uint32 words per row: face_c_dim elements × (faces across) × datum_bytes / 4 bytes per uint32
    // BF16 (datum_bytes=2): (nc*16*2)*2/4 = nc*16   FP8 (datum_bytes=1): (nc*16*2)*1/4 = nc*8
    int num_cols = (config.num_tiles_c_dim * config.face_c_dim * (config.num_faces >= 2 ? 2 : 1)) *
                   static_cast<int>(config.datum_bytes) / 4;
    // Half-face width in uint32 words: face_c_dim/2 elements × datum_bytes / 4 bytes per uint32
    // BF16: 16*2/4 = 8   FP8: 16*1/4 = 4
    const int half_face_w = config.face_c_dim * static_cast<int>(config.datum_bytes) / 4;
    // Rows per tile-row: 32 for a full 32x32 tile (2 face-rows), 16 for a 16x32 tiny tile (1 face-row).
    const int tile_r = config.face_r_dim * (config.num_faces > 2 ? 2 : 1);
    for (int x = 0; x < num_rows; x += tile_r) {
        for (int y = 0; y < num_cols; y += 2 * half_face_w) {
            int start = (x * num_cols) + y;

            // Top faces (face_r_dim rows each, not a hardcoded 16, so shortened faces work)
            for (int j = 0; j < 2; j++) {
                int start_ = start + (half_face_w * j);
                for (int k = 0; k < config.face_r_dim; k++) {
                    for (int i = 0; i < half_face_w; i++) {
                        int idx = start_ + (num_cols * k) + i;
                        dst_vec.push_back(src_vec.at(idx));
                    }
                }
            }

            if (config.num_faces > 2) {
                // Bottom faces
                start += config.face_r_dim * num_cols;
                for (int j = 0; j < 2; j++) {
                    int start_ = start + (half_face_w * j);
                    for (int k = 0; k < config.face_r_dim; k++) {
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

// src_vec is expected to be untilized
// result is also untilized
// Templated on the element type: uint16_t holds BF16 bit-patterns,
// uint32_t holds Float32 bit-patterns or Int32.
template <typename T>
std::vector<T> gold_transpose_wh(const std::vector<T>& src_vec, const std::vector<uint32_t>& shape) {
    vector<uint32_t> shapeT{shape[0], shape[1], shape[3], shape[2]};
    TensAddr addr(shape);
    TensAddr addrt(shapeT);

    vector<T> transposed(src_vec.size());
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
}

template std::vector<uint16_t> gold_transpose_wh<uint16_t>(
    const std::vector<uint16_t>& src_vec, const std::vector<uint32_t>& shape);
template std::vector<uint32_t> gold_transpose_wh<uint32_t>(
    const std::vector<uint32_t>& src_vec, const std::vector<uint32_t>& shape);

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

std::vector<uint32_t> gold_standard_tilize_w_reduce_col_max(
    const std::vector<uint32_t>& src0_vec, const std::vector<uint32_t>& src1_vec, const GoldenConfig& config) {
    // Extract scaler from src1_vec (first bfloat16 element)
    float scaler = 1.0f;
    if (!src1_vec.empty()) {
        std::vector<bfloat16> scaler_unpacked = tt::test_utils::unpack_vector<bfloat16, uint32_t>(src1_vec);
        scaler = static_cast<float>(scaler_unpacked[0]);
    }

    // Tilize the row-major input
    std::vector<uint32_t> tilized = gold_standard_tilize(src0_vec, config);
    std::vector<bfloat16> tilized_unpacked = tt::test_utils::unpack_vector<bfloat16, uint32_t>(tilized);

    const int num_tile_rows = config.num_tiles_r_dim;
    const int num_tile_cols = config.num_tiles_c_dim;
    const int face_r_dim = config.face_r_dim;
    const int face_c_dim = config.face_c_dim;
    const int num_faces_c = (config.num_faces >= 2) ? 2 : 1;
    const int num_faces_r = (config.num_faces > 2) ? 2 : 1;
    const int face_elems = face_r_dim * face_c_dim;
    const int tile_c_dim = num_faces_c * face_c_dim;
    const int tile_elems = config.num_faces * face_elems;

    // Reduce col max per tile row: each tile is reduced independently (no accumulation across tile rows)
    std::vector<bfloat16> result(num_tile_rows * num_tile_cols * tile_elems, bfloat16(0.0f));
    std::vector<float> col_max(tile_c_dim, -std::numeric_limits<float>::max());

    for (int tr = 0; tr < num_tile_rows; tr++) {
        for (int tc = 0; tc < num_tile_cols; tc++) {
            std::fill(col_max.begin(), col_max.end(), -std::numeric_limits<float>::max());

            int tile_offset = (tr * num_tile_cols + tc) * tile_elems;
            for (int fr = 0; fr < num_faces_r; fr++) {
                for (int fc = 0; fc < num_faces_c; fc++) {
                    int face_offset = tile_offset + (fr * num_faces_c + fc) * face_elems;
                    int col_base = fc * face_c_dim;
                    for (int r = 0; r < face_r_dim; r++) {
                        for (int c = 0; c < face_c_dim; c++) {
                            float val = static_cast<float>(tilized_unpacked[face_offset + r * face_c_dim + c]);
                            col_max[col_base + c] = fmaxf(col_max[col_base + c], val);
                        }
                    }
                }
            }

            int out_offset = (tr * num_tile_cols + tc) * tile_elems;
            for (int fc = 0; fc < num_faces_c; fc++) {
                int face_start = out_offset + fc * face_elems;
                int col_base = fc * face_c_dim;
                for (int c = 0; c < face_c_dim; c++) {
                    result[face_start + c] = bfloat16(col_max[col_base + c] * scaler);
                }
            }
        }
    }

    return tt::test_utils::pack_vector<uint32_t, bfloat16>(result);
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
