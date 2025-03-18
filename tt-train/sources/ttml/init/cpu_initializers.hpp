// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>

#include "core/xtensor_utils.hpp"

namespace ttml::init {

struct UniformRange {
    float a = 0;
    float b = 0;
};

struct NormalParams {
    float mean = 0.F;
    float stddev = 1.0F;
};

struct FanParams {
    uint32_t fan_in = 1;
    uint32_t fan_out = 1;
};

xt::xarray<float> uniform_init(const ttnn::Shape& shape, UniformRange range);
xt::xarray<float> normal_init(const ttnn::Shape& shape, NormalParams params);

void uniform_init(std::vector<float>& vec, UniformRange range);

void normal_init(std::vector<float>& vec, NormalParams params);

void constant_init(std::vector<float>& vec, float value);

void xavier_uniform_init(std::vector<float>& vec, FanParams params);

void xavier_normal_init(std::vector<float>& vec, FanParams params);

void kaiming_uniform_init(std::vector<float>& vec, int fan_in);

void kaiming_normal_init(std::vector<float>& vec, int fan_out);

}  // namespace ttml::init
