// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <random>
#include <span>
#include <thread>
#include <vector>

#include "random_parallel.hpp"
#include "random_sse.hpp"

namespace ttml::core {
namespace legacy {

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
    ttml::core::rng::generate_parallel_chunks(
        seq,
        [dist_factory](std::span<T> s, uint32_t s_seed) {
            sequential_generate<T, DistGenFunc>(s, dist_factory, s_seed);
        },
        seed,
        max_threads);
}

}  // namespace legacy

static inline bool use_simd_rng() {
    constexpr auto ENABLE_SIMD_RNG = "TT_TRAIN_ENABLE_SIMD_RNG";
    static bool simd_enabled = (std::getenv(ENABLE_SIMD_RNG) != nullptr);

    return simd_enabled;
}

// use_simd, which TT_TRAIN_ENABLE_SIMD_RNG sets by default, picks the SIMD generator. It has a fast path only for some
// element types and distributions and leaves seq unwritten for the rest, so those, integer distributions included,
// always use the scalar generator.
template <typename T, typename DistGenFunc>
void sequential_generate(
    std::span<T> seq, const DistGenFunc& dist_factory, uint32_t seed, bool use_simd = use_simd_rng()) {
    if constexpr (sse::has_simd_fast_path<T, DistGenFunc>) {
        if (use_simd) {
            return sse::sequential_generate(seq, dist_factory, seed);
        }
    }
    return legacy::sequential_generate(seq, dist_factory, seed);
}

template <typename T, typename DistGenFunc>
void parallel_generate(
    std::span<T> seq,
    DistGenFunc dist_factory,
    uint32_t seed,
    uint32_t max_threads = std::thread::hardware_concurrency(),
    bool use_simd = use_simd_rng()) {
    if constexpr (sse::has_simd_fast_path<T, DistGenFunc>) {
        if (use_simd) {
            return sse::parallel_generate(seq, dist_factory, seed, max_threads);
        }
    }
    return legacy::parallel_generate(seq, dist_factory, seed, max_threads);
}

}  // namespace ttml::core
