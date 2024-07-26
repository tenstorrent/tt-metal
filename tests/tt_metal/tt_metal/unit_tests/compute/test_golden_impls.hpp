// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <functional>
#include <limits>
#include <random>
#include <vector>

namespace unit_tests::compute {

std::vector<uint32_t> gold_standard_untilize(const std::vector<uint32_t> &src_vec, const std::vector<uint32_t> &shape);

std::vector<uint32_t> gold_standard_tilize(const std::vector<uint32_t> &src_vec, const std::vector<uint32_t> &shape);

// input shape.x is assumed to have the full number of elements in bfloat16
// src_vec is expected to be untilized
// result is also untilized
std::vector<uint16_t> gold_transpose_wh(const std::vector<uint16_t> &src_vec, const std::vector<uint32_t> &shape);

// input shape.x is assumed to have the full number of elements in bfloat16
// src_vec is expected to be untilized
// result is also untilized
std::vector<uint16_t> gold_reduce_h(const std::vector<uint16_t> &src_vec, const std::vector<uint32_t> &shape, float scaler, bool red_max = false, bool zeropad = true);

// input shape.x is assumed to have the full number of elements in bfloat16
// src_vec is expected to be untilized
// result is also untilized
std::vector<uint16_t> gold_reduce_w(const std::vector<uint16_t> &src_vec, const std::vector<uint32_t> &shape, float scaler, bool red_max = false, bool zeropad = true);

// input shape.x is assumed to have the full number of elements in bfloat16
// src_vec is expected to be untilized
// result is also untilized
std::vector<uint16_t> gold_reduce_hw(const std::vector<uint16_t> &src_vec, const std::vector<uint32_t> &shape, float scaler, bool red_max = false, bool zeropad = true);

}   // unit_tests::compute
