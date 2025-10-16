// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <span>
#include <thread>

#include "cpu_features.hpp"
#include "random.hpp"
#include "random_avx.hpp"
#include "random_sse.hpp"

namespace ttml::core::auto_dispatch {

// ============================================================================
// Auto-Dispatching RNG (Runtime CPU Detection)
// ============================================================================

// Check if we have SIMD-optimized implementations for a given type and distribution
template <typename T, typename Dist>
inline constexpr bool has_simd_support =
    // float: uniform and normal distributions
    (std::same_as<T, float> && (std::same_as<Dist, std::uniform_real_distribution<float>> ||
                                std::same_as<Dist, std::normal_distribution<float>>)) ||
    // double: uniform distribution only
    (std::same_as<T, double> && std::same_as<Dist, std::uniform_real_distribution<double>>) ||
    // bfloat16: uniform and normal distributions (via float conversion)
    (std::same_as<T, bfloat16> && (std::same_as<Dist, std::uniform_real_distribution<float>> ||
                                   std::same_as<Dist, std::normal_distribution<float>>));

// Sequential generate - automatically selects best implementation
template <typename T, typename DistGenFunc>
inline void sequential_generate(std::span<T> seq, DistGenFunc dist_factory, uint32_t seed) noexcept {
    using Dist = decltype(dist_factory());

    if constexpr (has_simd_support<T, Dist>) {
        // Runtime dispatch based on CPU features
        static const auto impl = CpuFeatures::get_recommended_rng();

        switch (impl) {
            case CpuFeatures::RngImpl::AVX2: ttml::core::avx::sequential_generate(seq, dist_factory, seed); return;
            case CpuFeatures::RngImpl::SSE: ttml::core::sse::sequential_generate(seq, dist_factory, seed); return;
            default:
                // Fallback to MT19937
                break;
        }
    }

    // Default: use MT19937 for unsupported types/distributions
    core::sequential_generate(seq, dist_factory, seed);
}

// Parallel generate - automatically selects best implementation
template <typename T, typename DistGenFunc>
inline void parallel_generate(
    std::span<T> seq,
    DistGenFunc dist_factory,
    uint32_t seed,
    uint32_t max_threads = std::thread::hardware_concurrency()) noexcept {
    using Dist = decltype(dist_factory());

    if constexpr (has_simd_support<T, Dist>) {
        // Runtime dispatch based on CPU features
        static const auto impl = CpuFeatures::get_recommended_rng();

        switch (impl) {
            case CpuFeatures::RngImpl::AVX2:
                ttml::core::avx::parallel_generate(seq, dist_factory, seed, max_threads);
                return;
            case CpuFeatures::RngImpl::SSE:
                ttml::core::sse::parallel_generate(seq, dist_factory, seed, max_threads);
                return;
            default:
                // Fallback to MT19937
                break;
        }
    }

    // Default: use MT19937
    core::parallel_generate(seq, dist_factory, seed, max_threads);
}

// Get information about which implementation is being used
inline const char* get_active_rng_name() {
    static const auto impl = CpuFeatures::get_recommended_rng();
    return CpuFeatures::rng_impl_name(impl);
}

inline CpuFeatures::RngImpl get_active_rng() {
    static const auto impl = CpuFeatures::get_recommended_rng();
    return impl;
}

}  // namespace ttml::core::auto_dispatch
