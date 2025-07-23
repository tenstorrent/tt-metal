// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <random>
#include <span>
#include <thread>
#include <vector>

#include "autograd/auto_context.hpp"

namespace ttml::core::random {

template <typename T, typename DistGenFunc>
void sequential_generate(std::span<T> seq, DistGenFunc&& dist_factory, uint32_t seed) {
    auto rng = std::mt19937{seed};
    auto dist = dist_factory();
    for (auto& it : seq) {
        it = dist(rng);
    }
}

template <typename TSeq>
concept Spannable = requires(TSeq seq) {
    typename TSeq::value_type;
    { seq.data() };
    { seq.size() } -> std::convertible_to<std::size_t>;
};

template <Spannable TSeq, typename DistGenFunc>
void parallel_generate(
    TSeq& seq, DistGenFunc dist_factory, uint32_t seed, uint32_t max_threads = std::thread::hardware_concurrency()) {
    using T = typename TSeq::value_type;
    auto rng = std::mt19937{seed};
    constexpr size_t min_size = 1 << 16;  // determined empirically on loudbox that this is about where we start seeing
                                          // gains; need to improve and measure it as a function of the processors too.
    if (seq.size() < min_size) {
        sequential_generate(std::span<T>{seq.data(), seq.size()}, dist_factory, seed);
        return;
    }

    size_t num_threads = std::min(max_threads, std::thread::hardware_concurrency());
    std::vector<std::thread> threads;
    threads.reserve(num_threads);

    std::vector<uint32_t> thread_seeds;  // NOTE: can move these out to autocontext to avoid re-seeding each time
    thread_seeds.reserve(num_threads);

    uint32_t seed_base = rng();
    size_t chunk_size = seq.size() / num_threads;
    size_t remainder = seq.size() % num_threads;

    size_t offset = 0;
    for (size_t i = 0; i < num_threads; ++i) {
        auto adjusted_chunk_size = chunk_size + (i == num_threads - 1 ? remainder : 0);
        threads.emplace_back([=, &dist_factory, &seq]() {
            std::span<T> chunk{seq.data() + offset, adjusted_chunk_size};
            sequential_generate(chunk, dist_factory, seed_base + i);
        });
        offset += adjusted_chunk_size;
    }

    for (auto& thread : threads) {
        thread.join();
    }
}

}  // namespace ttml::core::random
