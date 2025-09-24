// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "cpu_initializers.hpp"

#include <cmath>
#include <random>

#include "autograd/auto_context.hpp"

namespace ttml::init {

xt::xarray<float> uniform_init(const ttnn::Shape& shape, UniformRange range) {
    std::vector<float> data(shape.volume());
    uniform_init(data, range);
    std::vector<uint32_t> shape_vec(shape.cbegin(), shape.cend());
    return xt::adapt(data, shape_vec);
}

xt::xarray<float> normal_init(const ttnn::Shape& shape, NormalParams params) {
    std::vector<float> data(shape.volume());
    normal_init(data, params);
    std::vector<uint32_t> shape_vec(shape.cbegin(), shape.cend());
    return xt::adapt(data, shape_vec);
}

xt::xarray<float> constant_init(const ttnn::Shape& shape, float value) {
    std::vector<float> data(shape.volume());
    constant_init(data, value);
    std::vector<uint32_t> shape_vec(shape.cbegin(), shape.cend());
    return xt::adapt(data, shape_vec);
}

void uniform_init(std::vector<float>& vec, UniformRange range) {
    auto& gen = autograd::ctx().get_generator();
    uint32_t seed = gen();
    core::parallel_generate(
        std::span{vec.data(), vec.size()},
        [range]() { return std::uniform_real_distribution<float>(range.a, range.b); },
        seed);
}

void normal_init(std::vector<float>& vec, NormalParams params) {
    auto& gen = autograd::ctx().get_generator();
    uint32_t seed = gen();
    core::parallel_generate(
        std::span{vec.data(), vec.size()},
        [params]() { return std::normal_distribution<float>(params.mean, params.stddev); },
        seed);
}

void constant_init(std::vector<float>& vec, float value) {
    std::fill(vec.begin(), vec.end(), value);
}

void xavier_uniform_init(std::vector<float>& vec, FanParams params) {
    auto& gen = autograd::ctx().get_generator();
    uint32_t seed = gen();
    auto& [fan_in, fan_out] = params;
    float limit = std::sqrt(6.0F / (float)(fan_in + fan_out));
    core::parallel_generate(
        std::span{vec.data(), vec.size()},
        [limit]() { return std::uniform_real_distribution<float>(-limit, limit); },
        seed);
}

void xavier_normal_init(std::vector<float>& vec, FanParams params) {
    auto& gen = autograd::ctx().get_generator();
    uint32_t seed = gen();
    auto& [fan_in, fan_out] = params;
    float stddev = std::sqrt(2.0F / (float)(fan_in + fan_out));
    core::parallel_generate(
        std::span{vec.data(), vec.size()}, [stddev]() { return std::normal_distribution<float>(0.0F, stddev); }, seed);
}

void kaiming_uniform_init(std::vector<float>& vec, int fan_in) {
    auto& gen = autograd::ctx().get_generator();
    uint32_t seed = gen();
    float limit = std::sqrt(3.0F / (float)fan_in);
    core::parallel_generate(
        std::span{vec.data(), vec.size()},
        [limit]() { return std::uniform_real_distribution<float>(-limit, limit); },
        seed);
}

void kaiming_normal_init(std::vector<float>& vec, int fan_out) {
    auto& gen = autograd::ctx().get_generator();
    uint32_t seed = gen();
    float stddev = std::sqrt(2.0F / (float)fan_out);
    core::parallel_generate(
        std::span{vec.data(), vec.size()}, [stddev]() { return std::normal_distribution<float>(0.0F, stddev); }, seed);
}

}  // namespace ttml::init
