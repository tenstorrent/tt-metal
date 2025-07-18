// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <random>
#include <thread>
#include <vector>

#include "autograd/auto_context.hpp"
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

template <typename SeqT, typename DistT>
void sequential_init(SeqT& seq, DistT& d) {
    std::mt19937 rng = autograd::AutoContext::get_instance().get_generator();
    for (auto& it : seq) {
        it = d(rng);
    }
}

template <typename SeqT, typename DistT, typename RngT = std::mt19937>
void parallel_init(SeqT& seq, DistT& d, RngT& rng = autograd::AutoContext::get_instance().get_generator()) {
    constexpr size_t min_size = 1 << 16;  // determined empirically on loudbox that this is about where we start seeing
                                          // gains; need to improve and measure it as a function of the processors too.
    if (seq.size() < min_size) {
        sequential_init(seq, d);
        return;
    }

    size_t num_threads = std::thread::hardware_concurrency();  // FIXME: make it configurable?
    std::vector<std::thread> threads;
    threads.reserve(num_threads);

    std::vector<uint32_t> thread_seeds;  // NOTE: can move these out to autocontext to avoid re-seeding each time
    thread_seeds.reserve(num_threads);

    for (size_t i = 0; i < num_threads; ++i) {
        uint32_t seed = rng();
        thread_seeds.push_back(seed);
    }

    size_t chunk_size = seq.size() / num_threads;
    size_t remainder = seq.size() % num_threads;

    for (size_t i = 0; i < num_threads; ++i) {
        auto adjusted_chunk_size = chunk_size + (i == num_threads - 1 ? remainder : 0);

        threads.emplace_back([&, i]() {
            auto start = seq.begin() + i * chunk_size;
            auto end = start + adjusted_chunk_size;
            auto rng = RngT{thread_seeds[i]};

            for (auto it = start; it < end; ++it) {
                *it = d(rng);
            }
        });
    }

    for (auto& thread : threads) {
        thread.join();
    }
}

}  // namespace ttml::init
