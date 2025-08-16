// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <random>
#include <span>
#include <thread>
#include <vector>

namespace ttml::core {

template <typename T, typename DistGenFunc>
void sequential_generate(std::span<T> seq, const DistGenFunc& dist_factory, uint32_t seed) {
    auto dist = dist_factory();
    auto rng = std::mt19937{seed};
    std::generate(seq.begin(), seq.end(), [&dist, &rng]() { return dist(rng); });
}

template <typename T, typename DistGenFunc>
void parallel_generate(
    std::span<T> seq,
    DistGenFunc dist_factory,
    uint32_t seed,
    uint32_t max_threads = std::thread::hardware_concurrency()) {
    constexpr size_t min_size = 1 << 12;  // determined empirically that this is where we see an advantage over
                                          // sequential generation even with 2 threads.
    if (seq.size() < min_size) {
        sequential_generate(seq, dist_factory, seed);
        return;
    }

    size_t num_threads = std::min(max_threads, std::thread::hardware_concurrency());
    std::vector<std::thread> threads;
    threads.reserve(num_threads);

    size_t chunk_size = seq.size() / num_threads;
    size_t remainder = seq.size() % num_threads;

    size_t offset = 0;
    for (size_t i = 0; i < num_threads; ++i) {
        auto adjusted_chunk_size = chunk_size + (i == num_threads - 1 ? remainder : 0);
        threads.emplace_back([&dist_factory, &seq, offset, adjusted_chunk_size, seed, i]() {
            std::span<T> chunk{seq.data() + offset, adjusted_chunk_size};
            sequential_generate(chunk, dist_factory, seed + i);
        });
        offset += adjusted_chunk_size;
    }

    for (auto& thread : threads) {
        thread.join();
    }
}

}  // namespace ttml::core
