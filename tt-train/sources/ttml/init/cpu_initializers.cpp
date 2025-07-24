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
    auto& [a, b] = range;
    auto dist_factory = [&]() { return std::uniform_real_distribution<float>(a, b); };
    auto gen = autograd::ctx().get_generator();
    uint32_t seed = gen();
    core::random::sequential_generate(std::span{vec.data(), vec.size()}, dist_factory, seed);
}

void normal_init(std::vector<float>& vec, NormalParams params) {
    auto& [mean, stddev] = params;
    auto dist_factory = [&]() { return std::normal_distribution<float>(mean, stddev); };
    auto gen = autograd::ctx().get_generator();
    uint32_t seed = gen();
    core::random::sequential_generate(std::span{vec.data(), vec.size()}, dist_factory, seed);
}

void constant_init(std::vector<float>& vec, float value) {
    std::fill(vec.begin(), vec.end(), value);
}

void xavier_uniform_init(std::vector<float>& vec, FanParams params) {
    auto& [fan_in, fan_out] = params;
    float limit = std::sqrt(6.0F / (float)(fan_in + fan_out));

    std::uniform_real_distribution<float> dist(-limit, limit);

    // Fill the vector with uniformly distributed random values in the range [-limit, limit]
    std::generate(
        vec.begin(), vec.end(), [&]() { return dist(autograd::AutoContext::get_instance().get_generator()); });
}

void xavier_normal_init(std::vector<float>& vec, FanParams params) {
    auto& [fan_in, fan_out] = params;
    float stddev = std::sqrt(2.0F / (float)(fan_in + fan_out));

    // Random number generator with a seed
    // Mersenne Twister generator
    std::normal_distribution<float> dist(0.0F, stddev);
    std::generate(
        vec.begin(), vec.end(), [&]() { return dist(autograd::AutoContext::get_instance().get_generator()); });
}

void kaiming_uniform_init(std::vector<float>& vec, int fan_in) {
    float limit = std::sqrt(3.0F / (float)fan_in);

    std::uniform_real_distribution<float> dist(-limit, limit);

    // Fill the vector with uniformly distributed random values in the range [-limit, limit]
    std::generate(
        vec.begin(), vec.end(), [&]() { return dist(autograd::AutoContext::get_instance().get_generator()); });
}

void kaiming_normal_init(std::vector<float>& vec, int fan_out) {
    float stddev = std::sqrt(2.0F / (float)fan_out);

    std::normal_distribution<float> dist(0.0F, stddev);

    std::generate(
        vec.begin(), vec.end(), [&]() { return dist(autograd::AutoContext::get_instance().get_generator()); });
}

}  // namespace ttml::init
