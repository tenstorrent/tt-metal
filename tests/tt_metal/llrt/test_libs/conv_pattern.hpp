// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <map>
#include "tt_metal/test_utils/deprecated/tensor.hpp"
#include <tt_stl/assert.hpp>
#include <tt-metalium/constants.hpp>
using namespace tt::constants;
class ConvParameters {
public:
    uint32_t R = 1;
    uint32_t S = 1;
    uint32_t U = 1;
    uint32_t V = 1;
    uint32_t PadH = 0;
    uint32_t PadW = 0;
    ConvParameters(uint32_t r, uint32_t s, uint32_t u, uint32_t v, uint32_t padH, uint32_t padW) :
        R(r), S(s), U(u), V(v), PadH(padH), PadW(padW) {
        TT_FATAL(U > 0 and V > 0, "Error");
        TT_FATAL(R > 0 and S > 0, "Error");
        TT_FATAL(PadH >= 0 and PadW >= 0, "Error");
    }
    void print() {
        std::cout << "Printing conv params" << std::endl;
        std::cout << "R - " << R << " S - " << S << " U - " << U << " V - " << V << " PadH - " << PadH << " PadW - "
                  << PadW << std::endl;
    }
};

template <typename T>
std::vector<std::vector<T>> move_act_dram_to_l1(tt::deprecated::Tensor<T>& input_nhwc, ConvParameters conv_params) {
    std::vector<std::vector<T>> output;
    auto in_shape = input_nhwc.padded_shape();
    auto input_values = input_nhwc.get_values();

    std::vector<std::pair<int, int>> increments;
    for (auto r = 0; r < conv_params.R; r++) {
        for (auto s = 0; s < conv_params.S; s++) {
            increments.push_back(std::make_pair(s, r));
        }
    }
    for (int w = 0; w < in_shape[0]; w++) {
        for (int y = 0; y <= in_shape[1] - conv_params.R; y = y + conv_params.U) {
            for (int x = 0; x <= in_shape[2] - conv_params.S; x = x + conv_params.V) {
                std::vector<T> row;
                for (auto increment : increments) {
                    auto x_new = x + increment.first;
                    auto y_new = y + increment.second;
                    for (int z = 0; z < in_shape[3]; z++) {
                        auto idx = z + x_new * in_shape[3] + y_new * in_shape[3] * in_shape[2] +
                                   w * in_shape[3] * in_shape[2] * in_shape[1];
                        TT_FATAL(
                            idx >= 0 and idx < input_values.size(),
                            "Index {} out of bounds for input_values of size {}",
                            idx,
                            input_values.size());
                        row.push_back(input_values[idx]);
                    }
                }
                output.push_back(row);
            }
        }
    }

    return output;
}

template <typename T>
std::vector<T> move_act_dram_to_l1_tilized(
    tt::deprecated::Tensor<T>& input_nhwc, uint32_t dram_read_size_bytes, std::vector<uint32_t> address_map) {
    const auto& input_nhwc_values = input_nhwc.get_values();
    std::vector<T> l1_tilized_act;
    TT_FATAL(dram_read_size_bytes % sizeof(T) == 0, "dram_read_size_bytes must be divisible by sizeof(T)");
    uint32_t dram_read_size = dram_read_size_bytes / sizeof(T);
    for (int i = 0; i < address_map.size(); i++) {
        TT_FATAL(address_map[i] % sizeof(T) == 0, "address_map[{}] must be divisible by sizeof(T)", i);
        std::uint32_t dram_address = address_map[i] / sizeof(T);
        for (uint32_t j = 0; j < dram_read_size; j++) {
            l1_tilized_act.push_back(input_nhwc_values[dram_address + j]);
        }
    }
    return l1_tilized_act;
}

template <typename T>
std::vector<T> untilize_act(std::vector<T> tilized_act, std::uint32_t output_rows, std::uint32_t output_cols) {
    TT_FATAL(output_rows % TILE_HEIGHT == 0, "Error");
    TT_FATAL(output_cols % TILE_WIDTH == 0, "Error");
    uint32_t num_tiles_c = output_cols / TILE_WIDTH;
    uint32_t num_tiles_r = output_rows / TILE_HEIGHT;
    std::vector<T> untilized_act;

    for (auto r = 0; r < num_tiles_r; r++) {
        for (auto fi = 0; fi < 2; fi++) {
            for (auto i = 0; i < 16; i++) {
                for (auto c = 0; c < num_tiles_c; c++) {
                    for (auto fj = 0; fj < 2; fj++) {
                        for (auto j = 0; j < 16; j++) {
                            // determin index in tilized data. Tiles are arranged in col major order. But faces are row
                            // major order within the tile
                            uint32_t index = c * TILE_HEIGHT * TILE_WIDTH * num_tiles_r + r * TILE_WIDTH * TILE_HEIGHT +
                                             fi * 16 * 16 * 2 + fj * 16 * 16 + i * 16 + j;
                            TT_FATAL(
                                index < tilized_act.size(),
                                "Index {} out of bounds for tilized_act of size {}",
                                index,
                                tilized_act.size());
                            untilized_act.push_back(tilized_act.at(index));
                        }
                    }
                }
            }
        }
    }
    return untilized_act;
}

template <typename T>
std::vector<std::vector<T>> untilize_weights(
    std::vector<T> tilized_weights,
    std::uint32_t output_rows,
    std::uint32_t output_cols,
    std::uint32_t row_tile_size,
    std::uint32_t col_tile_size) {
    uint32_t num_tiles_rows = std::ceil(output_rows / row_tile_size);
    uint32_t num_tiles_cols = std::ceil(output_cols / col_tile_size);
    std::vector<std::vector<T>> untilized_weights;
    for (uint32_t i = 0; i < output_rows; i++) {
        std::vector<T> row(output_cols, (uint32_t)0);
        untilized_weights.push_back(row);
    }

    // 2d matrix is tilized. Tiles are arranged in row major order. Elements within tiles are in row major order.
    for (int tr = 0; tr < num_tiles_rows; tr++) {
        for (int tc = 0; tc < num_tiles_cols; tc++) {
            for (int r = 0; r < row_tile_size; r++) {
                for (int c = 0; c < col_tile_size; c++) {
                    uint32_t tilized_index = c + r * col_tile_size + tc * row_tile_size * col_tile_size +
                                             tr * num_tiles_cols * row_tile_size * col_tile_size;
                    // determine indices in untilized act
                    uint32_t out_col_index = c + tc * col_tile_size;
                    uint32_t out_row_index = r + tr * row_tile_size;
                    if (out_col_index < output_cols && out_row_index < output_rows) {
                        untilized_weights[out_row_index][out_col_index] = tilized_weights[tilized_index];
                    }
                }
            }
        }
    }
    return untilized_weights;
}

// Given a tilized data (each tile's data is contiguous and row major within the tile)
// transform it back to row major full tensor. (This function inverts the tilize() function)
template <typename T>
std::vector<T> untilize(std::vector<T> data, int rows, int cols) {
    TT_FATAL(rows % TILE_HEIGHT == 0, "Error");
    TT_FATAL(cols % TILE_WIDTH == 0, "Error");
    int num_tiles_r = rows / TILE_HEIGHT;
    int num_tiles_c = cols / TILE_WIDTH;
    std::vector<T> result;
    for (auto r = 0; r < num_tiles_r; r++) {
        for (auto fi = 0; fi < 2; fi++) {
            for (auto i = 0; i < 16; i++) {
                for (auto c = 0; c < num_tiles_c; c++) {
                    for (auto fj = 0; fj < 2; fj++) {
                        for (auto j = 0; j < 16; j++) {
                            uint32_t index = r * TILE_HEIGHT * TILE_WIDTH * num_tiles_c + c * TILE_HEIGHT * TILE_WIDTH +
                                             fi * 16 * 16 * 2 + fj * 16 * 16 + i * 16 + j;
                            TT_FATAL(
                                index < data.size(), "Index {} out of bounds for data of size {}", index, data.size());
                            result.push_back(data.at(index));
                        }
                    }
                }
            }
        }
    }
    return result;
}

template <typename T>
std::vector<T> tilize_2d_matrix(std::vector<std::vector<T>> row_major_matrix) {
    TT_FATAL(row_major_matrix.size() % TILE_HEIGHT == 0, "Error");
    TT_FATAL(row_major_matrix.at(0).size() % TILE_WIDTH == 0, "Error");
    int num_tiles_r = row_major_matrix.size() / TILE_HEIGHT;
    int num_tiles_c = row_major_matrix.at(0).size() / TILE_WIDTH;
    std::vector<T> result;
    for (auto r = 0; r < num_tiles_r; r++) {
        for (auto c = 0; c < num_tiles_c; c++) {
            for (auto j = 0; j < TILE_HEIGHT; j++) {     // tile rows
                for (auto i = 0; i < TILE_WIDTH; i++) {  // tile cols
                    result.push_back(row_major_matrix.at(r * TILE_HEIGHT + j).at(c * TILE_WIDTH + i));
                }
            }
        }
    }
    return result;
}

template <typename T>
std::vector<T> flatten(std::vector<std::vector<T>>& act_matrix) {
    std::vector<T> output;
    for (auto i = 0; i < act_matrix.size(); i++) {
        for (auto j = 0; j < act_matrix[i].size(); j++) {
            output.push_back(act_matrix[i][j]);
        }
    }
    return output;
}

template <typename T>
std::vector<std::vector<T>> move_weights_dram_to_l1(tt::deprecated::Tensor<T>& input_nhwc) {
    std::vector<std::vector<T>> output;
    std::array<uint32_t, 4> in_shape = input_nhwc.padded_shape();
    const auto& input_nhwc_values = input_nhwc.get_values();

    for (auto w = 0; w < in_shape[0]; w++) {
        std::vector<T> row;
        for (auto y = 0; y < in_shape[1]; y++) {
            for (auto x = 0; x < in_shape[2]; x++) {
                for (auto z = 0; z < in_shape[3]; z++) {
                    auto idx = z + x * in_shape[3] + y * in_shape[3] * in_shape[2] +
                               w * in_shape[3] * in_shape[2] * in_shape[1];
                    row.push_back(input_nhwc_values[idx]);
                }
            }
        }
        output.push_back(row);
    }
    return output;
}

template <typename T>
std::vector<std::vector<T>> move_weights_dram_to_l1_mm(tt::deprecated::Tensor<T>& input_nhwc) {
    std::vector<std::vector<T>> output;
    std::array<uint32_t, 4> in_shape = input_nhwc.padded_shape();
    const auto& input_nhwc_values = input_nhwc.get_values();
    uint32_t num_rows = in_shape[1] * in_shape[2] * in_shape[3];
    for (auto i = 0; i < num_rows; i++) {
        std::vector<T> row(in_shape[0], (uint32_t)0);
        output.push_back(row);
    }
    for (auto w = 0; w < in_shape[0]; w++) {
        uint32_t r = 0;
        for (auto y = 0; y < in_shape[1]; y++) {
            for (auto x = 0; x < in_shape[2]; x++) {
                for (auto z = 0; z < in_shape[3]; z++) {
                    auto idx = z + x * in_shape[3] + y * in_shape[3] * in_shape[2] +
                               w * in_shape[3] * in_shape[2] * in_shape[1];
                    output[r][w] = input_nhwc_values[idx];
                    r++;
                }
            }
        }
    }
    return output;
}

namespace test {
// Given a tensor that is row-major datums, make it tilized
// so that its row major within a tile, and each tile's data
// is contiguous
template <typename T>
std::vector<T> tilize(std::vector<T> data, int rows, int cols) {
    TT_FATAL(rows % TILE_HEIGHT == 0, "Error");
    TT_FATAL(cols % TILE_WIDTH == 0, "Error");
    int num_tiles_r = rows / TILE_HEIGHT;
    int num_tiles_c = cols / TILE_WIDTH;
    std::vector<T> result;
    for (auto r = 0; r < num_tiles_r; r++) {
        for (auto c = 0; c < num_tiles_c; c++) {
            for (auto fi = 0; fi < 2; fi++) {            // no. of rows of faces in tile
                for (auto fj = 0; fj < 2; fj++) {        // no. of cols of faces in tile
                    for (auto i = 0; i < 16; i++) {      // face rows
                        for (auto j = 0; j < 16; j++) {  // face cols
                            // each tile has 4 faces of 16x16
                            // each row of tile has 2 faces of 16x16
                            // int index = i + j * 16
                            uint32_t x = i + fi * 16 + r * TILE_HEIGHT;
                            uint32_t y = j + fj * 16 + c * TILE_WIDTH;
                            uint32_t index = y + x * cols;
                            result.push_back(data.at(index));
                        }
                    }
                }
            }
        }
    }
    return result;
}
}  // namespace test

std::tuple<uint32_t, uint32_t, uint32_t, std::vector<uint32_t>> gen_source_addresses_for_conv_act_layout_transform(
    std::array<uint32_t, 4> input_nhwc_shape, ConvParameters conv_params, uint32_t data_type_size_bytes) {
    auto N = input_nhwc_shape[0];
    auto H = input_nhwc_shape[1];
    auto W = input_nhwc_shape[2];
    auto C = input_nhwc_shape[3];
    auto R = conv_params.R;
    auto S = conv_params.S;
    auto U = conv_params.U;
    auto V = conv_params.V;

    std::uint32_t output_rows = (((H - R) / U) + 1) * (((W - S) / V) + 1);
    std::uint32_t output_cols = R * S * C;
    TT_FATAL(
        output_rows % TILE_HEIGHT == 0, "output_rows {} must be divisible by TILE_HEIGHT {}", output_rows, TILE_HEIGHT);
    TT_FATAL(
        output_cols % TILE_WIDTH == 0, "output_cols {} must be divisible by TILE_WIDTH {}", output_cols, TILE_WIDTH);
    std::uint32_t num_tiles_rows = output_rows / TILE_HEIGHT;
    std::uint32_t num_tiles_cols = output_cols / TILE_WIDTH;
    std::map<uint32_t, std::vector<std::vector<uint32_t>>> tiles_address_map;
    std::uint32_t out_col_index = 0;
    std::uint32_t out_row_index = 0;

    for (std::uint32_t n = 0; n < N; n++) {
        for (std::uint32_t h = 0; h < H - (R - 1); h = h + U) {
            for (std::uint32_t w = 0; w < W - (S - 1); w = w + V) {
                for (std::uint32_t r = 0; r < R; r++) {
                    for (std::uint32_t s = 0; s < S; s++) {
                        for (std::uint32_t c = 0; c < C; c++) {
                            // if start of one face within a tile row, push into address map
                            if (out_col_index % 16 == 0) {
                                uint32_t start_dram_address = c + (w + s) * C + (h + r) * W * C + n * H * W * C;
                                uint32_t tile_row_index = out_row_index / TILE_HEIGHT;
                                uint32_t tile_col_index = out_col_index / TILE_WIDTH;
                                // determine tile index. tiles are arranged in col major order
                                uint32_t tile_index = tile_row_index + tile_col_index * num_tiles_rows;
                                // there are 4 16x16 faces in a tile arranged in row major order
                                uint32_t face_row_index = (out_row_index % TILE_HEIGHT) / 16;
                                uint32_t face_col_index = (out_col_index % TILE_WIDTH) / 16;
                                uint32_t face_index = face_col_index + face_row_index * 2;
                                TT_FATAL(face_index < 4, "face_index {} must be less than 4", face_index);
                                if (tiles_address_map.find(tile_index) == tiles_address_map.end()) {
                                    std::vector<std::vector<uint32_t>> faces(4, std::vector<uint32_t>());
                                    tiles_address_map[tile_index] = faces;
                                }
                                tiles_address_map[tile_index][face_index].push_back(
                                    start_dram_address * data_type_size_bytes);
                            }
                            out_col_index++;
                        }
                    }
                }
                out_row_index++;
                out_col_index = 0;
            }
        }
    }
    // Serialize address map values in order of tile indices
    std::vector<uint32_t> address_map;
    uint32_t num_tiles = 0;
    uint32_t num_addresses_per_tile = 64;  // one address per row of face. There are 4 16x16 faces in tile.
    for (auto const& t : tiles_address_map) {
        num_tiles++;
        TT_FATAL(t.second.size() == 4, "Expected 4 faces per tile, got {}", t.second.size());
        for (auto& v : t.second) {
            TT_FATAL(v.size() == 16, "Expected 16 addresses per face, got {}", v.size());
            for (auto& a : v) {
                address_map.push_back(a);
            }
        }
    }
    return make_tuple(num_tiles, num_addresses_per_tile, 16 * data_type_size_bytes, address_map);
}

template <typename T>
std::vector<T> perform_matmul(std::vector<T> input1, std::vector<T> input2, uint32_t K, uint32_t M, uint32_t N) {
    std::vector<std::vector<T>> result;
    for (uint32_t m = 0; m < M; m++) {
        std::vector<T> row(N, (uint32_t)0);
        for (uint32_t n = 0; n < N; n++) {
            for (uint32_t k = 0; k < K; k++) {
                row[n] += (uint32_t)input1[m * K + k] * (uint32_t)input2[k * N + n];
            }
        }
        result.push_back(row);
    }
    return result;
}
