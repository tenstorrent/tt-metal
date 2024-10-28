// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <functional>
#include <limits>
#include <random>
#include <vector>
#include <algorithm>

#include "common/test_tiles.hpp"
#include "tt_metal/common/bfloat16.hpp"
#include "tt_metal/common/bfloat8.hpp"
#include "tt_metal/common/bfloat4.hpp"
#include "tt_metal/host_api.hpp"
#include "tt_metal/detail/tt_metal.hpp"
#include "tests/tt_metal/test_utils/packing.hpp"

//TODO: RT these functions should be templated for different data formats
namespace unit_tests::compute {

//Used if golden function needs tile details
struct GoldenConfig {
    int num_tiles_r_dim;
    int num_tiles_c_dim;
    int face_r_dim = 16;
    int face_c_dim = 16;
    int num_faces = 4;
};

std::vector<uint32_t> gold_standard_untilize(const std::vector<uint32_t> &src_vec, const GoldenConfig &config);

std::vector<uint32_t> gold_standard_tilize(const std::vector<uint32_t> &src_vec,  const GoldenConfig &config);

// input shape.x is assumed to have the full number of elements in bfloat16
// src_vec is expected to be untilized
// result is also untilized
std::vector<uint16_t> gold_transpose_wh(const std::vector<uint16_t> &src_vec, const std::vector<uint32_t> &shape);

// input shape.x is assumed to have the full number of elements in bfloat16
// src_vec is expected to be untilized
// result is also untilized
// red_type : {SUM, AVG, MAX}; i.e. {0, 1, 2};
std::vector<uint16_t> gold_reduce_h(const std::vector<uint16_t> &src_vec, const std::vector<uint32_t> &shape, float scaler, uint8_t red_type = 0, bool zeropad = true);

// input shape.x is assumed to have the full number of elements in bfloat16
// src_vec is expected to be untilized
// result is also untilized
// red_type : {SUM, AVG, MAX}; i.e. {0, 1, 2};
std::vector<uint16_t> gold_reduce_w(const std::vector<uint16_t> &src_vec, const std::vector<uint32_t> &shape, float scaler, uint8_t red_type = 0, bool zeropad = true);

// input shape.x is assumed to have the full number of elements in bfloat16
// src_vec is expected to be untilized
// result is also untilized
// red_type : {SUM, AVG, MAX}; i.e. {0, 1, 2};
std::vector<uint16_t> gold_reduce_hw(const std::vector<uint16_t> &src_vec, const std::vector<uint32_t> &shape, float scaler, uint8_t red_type = 0, bool zeropad = true);

// Takes untilized src0_vec and tilized src1_vec
// returns tilized result of eltwise addition
// Assumes all elements in bfloat16
std::vector<uint32_t> gold_standard_tilize_w_elwadd(const std::vector<uint32_t> &src0_vec, const std::vector<uint32_t> &src1_vec, const GoldenConfig &config);

// Random packed uint32_t vector generator which is data-format agnostic.
// Takes the following parameters:
//
// lower - a lower limit of the input range
// upper - an upper limit of the input range
// num_bytes - number of bytes that the vector will occupy
// data_format - data format of each element, packed to uint32_t, currently supporting Bfloat16, Float32, Bfp8_b and Bfp4_b
// seed - randomization seed
// exclude_zeroes - if true, excludes values around zero, with the limits given by next two parameters
// golden_neg_epsilon - small negative value above which no elements of the vector will take value from
// golden_pos_epsilon - small positive value below which no elements of the vector will take value from
//
// Returns:
//
// a uint32_t vector of packed values depending on the data format and given limits
std::vector<uint32_t> generate_random_vector_generalized(const float lower, const float upper, const size_t num_bytes, const tt::DataFormat data_format, const int seed, bool exclude_zeroes = false, float golden_neg_epsilon = -0.0001f, float golden_pos_epsilon = 0.0001f);

// Unpacking function which is data-format agnostic
// Takes the following parameters:
//
// data_format -  data format in which the vector was packed, currently supporting Bfloat16, Float32, Bfp8_b and Bfp4_b
// packed_input - a uint32_t packed vector
//
// Returns:
// a float vector of unpacked values depending on the data format
std::vector<float> unpack_generalized(const tt::DataFormat data_format, const std::vector<uint32_t>& packed_input);

}   // unit_tests::compute
