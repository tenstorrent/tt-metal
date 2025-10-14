// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <wmmintrin.h>  // AES-NI

#include <algorithm>
#include <concepts>
#include <cstdint>
#include <limits>
#include <random>
#include <ranges>
#include <span>
#include <thread>
#include <vector>

#include "random.hpp"  // Fallback to original MT19937 implementation
#include "tt-metalium/bfloat16.hpp"

namespace ttml::core::sse {

// ============================================================================
// Constants
// ============================================================================

inline constexpr size_t simd_float_batch_size = 4;
inline constexpr size_t cache_line_size_bytes = 64;
inline constexpr size_t parallel_min_size = 1 << 12;
inline constexpr size_t thread_seed_shift_bits = 32;
inline constexpr int aes_warmup_rounds = 10;
inline constexpr uint64_t seed_mix_constant_1 = 0x0123456789abcdefULL;
inline constexpr uint64_t seed_mix_constant_2 = 0xfedcba9876543210ULL;
inline constexpr int float_mantissa_shift = 9;
inline constexpr uint32_t float_exponent_bias = 0x3f800000;

// ============================================================================
// AES RNG
// ============================================================================

class AesRng {
private:
    __m128i state_;
    __m128i key_;
    __m128i increment_;

public:
    using result_type = uint32_t;

    static constexpr result_type min() noexcept {
        return std::numeric_limits<result_type>::min();
    }
    static constexpr result_type max() noexcept {
        return std::numeric_limits<result_type>::max();
    }

    __attribute__((target("aes,sse4.2"))) explicit AesRng(uint64_t seed = 0) noexcept :
        state_{_mm_set_epi64x(seed, seed ^ seed_mix_constant_1)},
        key_{_mm_set_epi64x(seed ^ seed_mix_constant_2, seed + seed_mix_constant_1)},
        increment_{_mm_set_epi64x(seed_mix_constant_1, seed_mix_constant_2)} {
        for (int i = 0; i < aes_warmup_rounds; ++i) {
            advance();
        }
    }

    __attribute__((target("aes,sse4.2"))) void advance() noexcept {
        state_ = _mm_aesenc_si128(state_, key_);
        state_ = _mm_aesenc_si128(state_, increment_);
        state_ = _mm_aesenc_si128(state_, key_);
        state_ = _mm_xor_si128(state_, increment_);
        increment_ = _mm_add_epi64(increment_, _mm_set_epi64x(1, 1));
    }

    [[nodiscard]]
    __attribute__((target("aes,sse4.2"))) __m128i generate_128bit() noexcept {
        advance();
        return state_;
    }

    [[nodiscard]]
    __attribute__((target("aes,sse4.2"))) __m128 generate_float_x4() noexcept {
        __m128i rand_int = generate_128bit();
        __m128i mantissa = _mm_srli_epi32(rand_int, float_mantissa_shift);
        __m128i float_bits = _mm_or_si128(mantissa, _mm_set1_epi32(float_exponent_bias));
        __m128 result = _mm_castsi128_ps(float_bits);
        return _mm_sub_ps(result, _mm_set1_ps(1.0f));
    }

    [[nodiscard]]
    __attribute__((target("aes,sse4.2"))) __m128d generate_double_x2() noexcept {
        __m128i rand_int = generate_128bit();
        // For double: 52-bit mantissa, shift right by 12 bits
        __m128i mantissa = _mm_srli_epi64(rand_int, 12);
        // Set exponent to 0x3FF0000000000000 (creates [1.0, 2.0) range)
        __m128i double_bits = _mm_or_si128(mantissa, _mm_set1_epi64x(0x3FF0000000000000LL));
        __m128d result = _mm_castsi128_pd(double_bits);
        return _mm_sub_pd(result, _mm_set1_pd(1.0));
    }

    [[nodiscard]]
    __attribute__((target("aes,sse4.2"))) std::array<bfloat16, 4> generate_bfloat16_x4() noexcept {
        // Generate 4 floats in [0,1) range
        __m128 floats = generate_float_x4();

        // Convert float to bfloat16 by extracting upper 16 bits
        // bfloat16 = upper 16 bits of float32
        __m128i float_bits = _mm_castps_si128(floats);

        // Shift right by 16 to get upper 16 bits of each 32-bit float
        __m128i upper_bits = _mm_srli_epi32(float_bits, 16);

        // Extract and pack into array
        alignas(16) uint32_t packed[4];
        _mm_store_si128((__m128i*)packed, upper_bits);

        return std::array<bfloat16, 4>{
            std::bit_cast<bfloat16>(static_cast<uint16_t>(packed[0])),
            std::bit_cast<bfloat16>(static_cast<uint16_t>(packed[1])),
            std::bit_cast<bfloat16>(static_cast<uint16_t>(packed[2])),
            std::bit_cast<bfloat16>(static_cast<uint16_t>(packed[3]))};
    }

    [[nodiscard]]
    __attribute__((target("aes,sse4.2"))) result_type operator()() noexcept {
        thread_local int idx = simd_float_batch_size;
        thread_local uint32_t buffer[simd_float_batch_size];
        thread_local AesRng* last_rng = nullptr;

        if (idx >= simd_float_batch_size || last_rng != this) {
            __m128i rand = generate_128bit();
            _mm_storeu_si128((__m128i*)buffer, rand);
            idx = 0;
            last_rng = this;
        }
        return buffer[idx++];
    }
};

// ============================================================================
// Internal SIMD Generation (Float)
// ============================================================================

template <std::same_as<float> T = float>
__attribute__((target("aes,sse4.2"))) void generate_uniform_simd(
    std::span<T> output, uint32_t seed, auto dist_factory) {
    AesRng rng{seed};
    auto dist = dist_factory();
    auto params = dist.param();

    const auto min = params.a();
    const auto max = params.b();

    const size_t num_batches = output.size() / simd_float_batch_size;
    const __m128 range_vec = _mm_set1_ps(max - min);
    const __m128 min_vec = _mm_set1_ps(min);

    for (size_t i : std::views::iota(0u, num_batches)) {
        __m128 rand = rng.generate_float_x4();
        __m128 scaled = _mm_add_ps(_mm_mul_ps(rand, range_vec), min_vec);
        _mm_storeu_ps(&output[i * simd_float_batch_size], scaled);
    }

    for (size_t i : std::views::iota(num_batches * simd_float_batch_size, output.size())) {
        T rand_val = _mm_cvtss_f32(rng.generate_float_x4());
        output[i] = min + rand_val * (max - min);
    }
}

// Parallel SIMD generation for uniform_real_distribution<float>
template <typename DistFactory>
__attribute__((target("aes,sse4.2"))) void generate_uniform_simd_parallel(
    std::span<float> output, DistFactory dist_factory, uint32_t seed, size_t num_threads) {
    if (output.size() < parallel_min_size) [[unlikely]] {
        generate_uniform_simd(output, seed, dist_factory);
        return;
    }

    constexpr size_t elems_per_line = cache_line_size_bytes / sizeof(float);
    size_t chunk_size = output.size() / num_threads;
    if constexpr (elems_per_line > 1) {
        chunk_size = ((chunk_size + elems_per_line - 1) / elems_per_line) * elems_per_line;
    }

    auto chunks = std::views::iota(0u, num_threads) | std::views::transform([&](size_t i) {
                      const size_t offset = i * chunk_size;
                      const size_t size = std::min(chunk_size, output.size() - offset);
                      return std::span{output.data() + offset, size};
                  }) |
                  std::views::take_while([](auto chunk) { return !chunk.empty(); });

    std::vector<std::jthread> threads;
    threads.reserve(num_threads);

    size_t thread_id = 0;
    for (auto chunk : chunks) {
        const uint64_t thread_seed =
            static_cast<uint64_t>(seed) + (static_cast<uint64_t>(thread_id++) << thread_seed_shift_bits);
        threads.emplace_back([chunk, thread_seed, dist_factory]() {
            generate_uniform_simd(chunk, static_cast<uint32_t>(thread_seed), dist_factory);
        });
    }
}

// ============================================================================
// Internal SIMD Generation (Double)
// ============================================================================

template <std::same_as<double> T = double>
__attribute__((target("aes,sse4.2"))) void generate_uniform_simd_double(
    std::span<T> output, uint32_t seed, auto dist_factory) {
    constexpr size_t simd_double_batch_size = 2;  // 2 doubles per SSE register

    AesRng rng{seed};
    auto dist = dist_factory();
    auto params = dist.param();

    const auto min = params.a();
    const auto max = params.b();

    const size_t num_batches = output.size() / simd_double_batch_size;
    const size_t remainder = output.size() % simd_double_batch_size;

    const auto range = max - min;
    const __m128d range_vec = _mm_set1_pd(range);
    const __m128d min_vec = _mm_set1_pd(min);

    // Process full batches
    for (size_t i = 0; i < num_batches; ++i) {
        __m128d rand_01 = rng.generate_double_x2();
        __m128d scaled = _mm_add_pd(_mm_mul_pd(rand_01, range_vec), min_vec);
        _mm_storeu_pd(&output[i * simd_double_batch_size], scaled);
    }

    // Process remainder
    if (remainder > 0) {
        std::mt19937 fallback_rng{seed};
        auto fallback_dist = dist_factory();
        for (size_t i = num_batches * simd_double_batch_size; i < output.size(); ++i) {
            output[i] = fallback_dist(fallback_rng);
        }
    }
}

template <std::same_as<double> T = double>
__attribute__((target("aes,sse4.2"))) void generate_uniform_simd_parallel_double(
    std::span<T> output, uint32_t seed, auto dist_factory, size_t num_threads) {
    constexpr size_t simd_double_batch_size = 2;
    constexpr size_t elems_per_cache_line = cache_line_size_bytes / sizeof(T);
    constexpr size_t elems_per_line = (elems_per_cache_line / simd_double_batch_size) * simd_double_batch_size;

    size_t chunk_size = output.size() / num_threads;

    if (chunk_size < elems_per_line) {
        chunk_size = elems_per_line;
    } else {
        chunk_size = ((chunk_size + elems_per_line - 1) / elems_per_line) * elems_per_line;
    }

    auto chunks = std::views::iota(0u, num_threads) | std::views::transform([&](size_t i) {
                      const size_t offset = i * chunk_size;
                      const size_t size = std::min(chunk_size, output.size() - offset);
                      return std::span{output.data() + offset, size};
                  }) |
                  std::views::take_while([](auto chunk) { return !chunk.empty(); });

    std::vector<std::jthread> threads;
    threads.reserve(num_threads);

    size_t thread_id = 0;
    for (auto chunk : chunks) {
        const uint64_t thread_seed =
            static_cast<uint64_t>(seed) + (static_cast<uint64_t>(thread_id++) << thread_seed_shift_bits);
        threads.emplace_back([chunk, thread_seed, dist_factory]() {
            generate_uniform_simd_double(chunk, static_cast<uint32_t>(thread_seed), dist_factory);
        });
    }
}

// ============================================================================
// Internal SIMD Generation (bfloat16)
// ============================================================================

template <std::same_as<bfloat16> T = bfloat16>
__attribute__((target("aes,sse4.2"))) void generate_uniform_simd_bfloat16(
    std::span<T> output, uint32_t seed, auto dist_factory) {
    constexpr size_t simd_bf16_batch_size = 4;  // 4 bfloat16s generated at a time

    AesRng rng{seed};
    auto dist = dist_factory();
    auto params = dist.param();

    const float min = params.a();
    const float max = params.b();
    const float range = max - min;

    const size_t num_batches = output.size() / simd_bf16_batch_size;
    const size_t remainder = output.size() % simd_bf16_batch_size;

    // Process full batches
    for (size_t i = 0; i < num_batches; ++i) {
        auto bf16_values = rng.generate_bfloat16_x4();

        // Scale to [min, max) range
        for (size_t j = 0; j < simd_bf16_batch_size; ++j) {
            float val = static_cast<float>(bf16_values[j]);
            float scaled = val * range + min;
            // Convert scaled float to bfloat16 by truncating to upper 16 bits
            uint32_t float_bits = std::bit_cast<uint32_t>(scaled);
            uint16_t bf16_bits = static_cast<uint16_t>(float_bits >> 16);
            output[i * simd_bf16_batch_size + j] = std::bit_cast<bfloat16>(bf16_bits);
        }
    }

    // Process remainder
    if (remainder > 0) {
        std::mt19937 fallback_rng{seed};
        auto fallback_dist = dist_factory();
        for (size_t i = num_batches * simd_bf16_batch_size; i < output.size(); ++i) {
            float val = fallback_dist(fallback_rng);
            uint32_t float_bits = std::bit_cast<uint32_t>(val);
            uint16_t bf16_bits = static_cast<uint16_t>(float_bits >> 16);
            output[i] = std::bit_cast<bfloat16>(bf16_bits);
        }
    }
}

template <std::same_as<bfloat16> T = bfloat16>
__attribute__((target("aes,sse4.2"))) void generate_uniform_simd_parallel_bfloat16(
    std::span<T> output, uint32_t seed, auto dist_factory, size_t num_threads) {
    constexpr size_t simd_bf16_batch_size = 4;
    constexpr size_t elems_per_cache_line = cache_line_size_bytes / sizeof(T);
    constexpr size_t elems_per_line = (elems_per_cache_line / simd_bf16_batch_size) * simd_bf16_batch_size;

    size_t chunk_size = output.size() / num_threads;

    if (chunk_size < elems_per_line) {
        chunk_size = elems_per_line;
    } else {
        chunk_size = ((chunk_size + elems_per_line - 1) / elems_per_line) * elems_per_line;
    }

    auto chunks = std::views::iota(0u, num_threads) | std::views::transform([&](size_t i) {
                      const size_t offset = i * chunk_size;
                      const size_t size = std::min(chunk_size, output.size() - offset);
                      return std::span{output.data() + offset, size};
                  }) |
                  std::views::take_while([](auto chunk) { return !chunk.empty(); });

    std::vector<std::jthread> threads;
    threads.reserve(num_threads);

    size_t thread_id = 0;
    for (auto chunk : chunks) {
        const uint64_t thread_seed =
            static_cast<uint64_t>(seed) + (static_cast<uint64_t>(thread_id++) << thread_seed_shift_bits);
        threads.emplace_back([chunk, thread_seed, dist_factory]() {
            generate_uniform_simd_bfloat16(chunk, static_cast<uint32_t>(thread_seed), dist_factory);
        });
    }
}

// ============================================================================
// Type Traits for Distribution Detection
// ============================================================================

template <typename Dist>
concept UniformRealDistribution = requires(Dist d) {
    { d.param().a() } -> std::convertible_to<typename Dist::result_type>;
    { d.param().b() } -> std::convertible_to<typename Dist::result_type>;
};

// ============================================================================
// Drop-in Replacement API
// ============================================================================

// Sequential generate - matches original random.hpp API
template <typename T, typename DistGenFunc>
inline void sequential_generate(std::span<T> seq, DistGenFunc dist_factory, uint32_t seed) noexcept {
    // SIMD fast path for uniform_real_distribution<float>
    if constexpr (std::same_as<T, float>) {
        auto dist = dist_factory();
        if constexpr (UniformRealDistribution<decltype(dist)>) {
            generate_uniform_simd(seq, seed, dist_factory);
        } else {
            // Fallback to original MT19937 implementation for non-uniform distributions
            core::sequential_generate(seq, dist_factory, seed);
        }
    }
    // SIMD fast path for uniform_real_distribution<double>
    else if constexpr (std::same_as<T, double>) {
        auto dist = dist_factory();
        if constexpr (UniformRealDistribution<decltype(dist)>) {
            generate_uniform_simd_double(seq, seed, dist_factory);
        } else {
            // Fallback to original MT19937 implementation for non-uniform distributions
            core::sequential_generate(seq, dist_factory, seed);
        }
    }
    // SIMD fast path for uniform_real_distribution<bfloat16>
    else if constexpr (std::same_as<T, bfloat16>) {
        auto dist = dist_factory();
        if constexpr (UniformRealDistribution<decltype(dist)>) {
            generate_uniform_simd_bfloat16(seq, seed, dist_factory);
        } else {
            // Fallback to original MT19937 implementation for non-uniform distributions
            core::sequential_generate(seq, dist_factory, seed);
        }
    } else {
        // Fallback to original MT19937 implementation
        core::sequential_generate(seq, dist_factory, seed);
    }
}

// Parallel generate - matches original random.hpp API
template <typename T, typename DistGenFunc>
inline void parallel_generate(
    std::span<T> seq,
    DistGenFunc dist_factory,
    uint32_t seed,
    uint32_t max_threads = std::thread::hardware_concurrency()) noexcept {
    // SIMD fast path for uniform_real_distribution<float>
    if constexpr (std::same_as<T, float>) {
        auto dist = dist_factory();
        if constexpr (UniformRealDistribution<decltype(dist)>) {
            generate_uniform_simd_parallel(seq, dist_factory, seed, max_threads);
        } else {
            // Fallback to original MT19937 implementation for non-uniform distributions
            core::parallel_generate(seq, dist_factory, seed, max_threads);
        }
    }
    // SIMD fast path for uniform_real_distribution<double>
    else if constexpr (std::same_as<T, double>) {
        auto dist = dist_factory();
        if constexpr (UniformRealDistribution<decltype(dist)>) {
            generate_uniform_simd_parallel_double(seq, dist_factory, seed, max_threads);
        } else {
            // Fallback to original MT19937 implementation for non-uniform distributions
            core::parallel_generate(seq, dist_factory, seed, max_threads);
        }
    }
    // SIMD fast path for uniform_real_distribution<bfloat16>
    else if constexpr (std::same_as<T, bfloat16>) {
        auto dist = dist_factory();
        if constexpr (UniformRealDistribution<decltype(dist)>) {
            generate_uniform_simd_parallel_bfloat16(seq, dist_factory, seed, max_threads);
        } else {
            // Fallback to original MT19937 implementation for non-uniform distributions
            core::parallel_generate(seq, dist_factory, seed, max_threads);
        }
    } else {
        // Fallback to original MT19937 implementation
        core::parallel_generate(seq, dist_factory, seed, max_threads);
    }
}

}  // namespace ttml::core::sse
