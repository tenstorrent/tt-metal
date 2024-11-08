// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <vector>

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

void uniform_init(std::vector<float>& vec, UniformRange range);

void normal_init(std::vector<float>& vec, NormalParams params);

void constant_init(std::vector<float>& vec, float value);

void xavier_uniform_init(std::vector<float>& vec, FanParams params);

void xavier_normal_init(std::vector<float>& vec, FanParams params);

void kaiming_uniform_init(std::vector<float>& vec, int fan_in);

void kaiming_normal_init(std::vector<float>& vec, int fan_out);

}  // namespace ttml::init
