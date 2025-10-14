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

// Sequential generate - automatically selects best implementation
template <typename T, typename DistGenFunc>
inline void sequential_generate(std::span<T> seq, DistGenFunc dist_factory, uint32_t seed) noexcept {
    // Only optimize float uniform distributions
    if constexpr (std::same_as<T, float>) {
        auto dist = dist_factory();

        // Check if this is a uniform distribution
        using DistType = decltype(dist);
        constexpr bool is_uniform = requires(DistType d) {
            { d.param().a() } -> std::convertible_to<typename DistType::result_type>;
            { d.param().b() } -> std::convertible_to<typename DistType::result_type>;
        };

        if constexpr (is_uniform) {
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
    }

    // Default: use MT19937
    core::sequential_generate(seq, dist_factory, seed);
}

// Parallel generate - automatically selects best implementation
template <typename T, typename DistGenFunc>
inline void parallel_generate(
    std::span<T> seq,
    DistGenFunc dist_factory,
    uint32_t seed,
    uint32_t max_threads = std::thread::hardware_concurrency()) noexcept {
    // Only optimize float uniform distributions
    if constexpr (std::same_as<T, float>) {
        auto dist = dist_factory();

        // Check if this is a uniform distribution
        using DistType = decltype(dist);
        constexpr bool is_uniform = requires(DistType d) {
            { d.param().a() } -> std::convertible_to<typename DistType::result_type>;
            { d.param().b() } -> std::convertible_to<typename DistType::result_type>;
        };

        if constexpr (is_uniform) {
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
