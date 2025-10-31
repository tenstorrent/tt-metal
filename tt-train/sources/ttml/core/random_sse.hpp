// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <wmmintrin.h>  // AES-NI

#include <algorithm>
#include <bit>
#include <cmath>
#include <concepts>
#include <cstdint>
#include <iostream>
#include <limits>
#include <numbers>
#include <random>
#include <ranges>
#include <span>
#include <thread>
#include <vector>

#include "tt-metalium/bfloat16.hpp"

namespace ttml::core::sse {

// ============================================================================
// Constants
// ============================================================================

inline constexpr size_t simd_float_batch_size = 4;
inline constexpr size_t simd_bf16_batch_size = 4;  // 4 bfloat16s generated at a time
inline constexpr size_t cache_line_size_bytes = 64;
inline constexpr size_t parallel_min_size = 4096;
inline constexpr size_t thread_seed_shift_bits = 32;
inline constexpr int aes_warmup_rounds = 10;
inline constexpr uint64_t seed_mix_constant_1 = 0x0123456789abcdefULL;
inline constexpr uint64_t seed_mix_constant_2 = 0xfedcba9876543210ULL;
inline constexpr int float_mantissa_shift = 9;
inline constexpr uint32_t float_exponent_bias = 0x3f800000;

// ============================================================================
// Helper Functions
// ============================================================================

// Convert float to bfloat16 by truncating to upper 16 bits
inline bfloat16 float_to_bfloat16(float value) noexcept {
    uint32_t float_bits = std::bit_cast<uint32_t>(value);
    uint16_t bf16_bits = static_cast<uint16_t>(float_bits >> 16);
    return std::bit_cast<bfloat16>(bf16_bits);
}

// Calculate cache-aligned chunk size for parallel processing
template <typename T>
inline size_t calculate_aligned_chunk_size(size_t total_size, size_t num_threads, size_t simd_batch_size) noexcept {
    constexpr size_t elems_per_cache_line = cache_line_size_bytes / sizeof(T);
    const size_t elems_per_line = (elems_per_cache_line / simd_batch_size) * simd_batch_size;

    size_t chunk_size = total_size / num_threads;

    if (chunk_size < elems_per_line) {
        return elems_per_line;
    }
    return ((chunk_size + elems_per_line - 1) / elems_per_line) * elems_per_line;
}

// Create chunks for parallel processing
template <typename T>
inline auto create_chunks(std::span<T> output, size_t num_threads, size_t chunk_size) noexcept {
    return std::views::iota(0u, num_threads) | std::views::transform([&](size_t i) {
               const size_t offset = i * chunk_size;
               const size_t size = std::min(chunk_size, output.size() - offset);
               return std::span{output.data() + offset, size};
           }) |
           std::views::take_while([](auto chunk) { return !chunk.empty(); });
}

// Calculate thread-specific seed
inline uint64_t calculate_thread_seed(uint32_t base_seed, size_t thread_id) noexcept {
    return static_cast<uint64_t>(base_seed) + (static_cast<uint64_t>(thread_id) << thread_seed_shift_bits);
}

// Fill remainder elements using fallback RNG
template <typename T, typename DistFactory>
inline void fill_remainder_fallback(
    std::span<T> output, size_t start_idx, uint32_t seed, DistFactory dist_factory) noexcept {
    if (start_idx >= output.size()) {
        return;
    }
    std::mt19937 fallback_rng{seed + static_cast<uint32_t>(start_idx)};
    auto fallback_dist = dist_factory();
    for (size_t i = start_idx; i < output.size(); ++i) {
        if constexpr (std::same_as<T, bfloat16>) {
            float val = fallback_dist(fallback_rng);
            output[i] = float_to_bfloat16(val);
        } else {
            output[i] = fallback_dist(fallback_rng);
        }
    }
}

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
        thread_local int idx = 0;
        thread_local uint32_t buffer[simd_float_batch_size];
        thread_local AesRng* last_rng = nullptr;
        thread_local bool first_use = true;

        if (first_use || idx >= simd_float_batch_size || last_rng != this) {
            __m128i rand = generate_128bit();
            _mm_storeu_si128((__m128i*)buffer, rand);
            idx = 0;
            last_rng = this;
            first_use = false;
        }
        return buffer[idx++];
    }
};

// ============================================================================
// Internal SIMD Generation (Float)
// ============================================================================

__attribute__((target("aes,sse4.2"))) void generate_uniform_simd(
    std::span<float> output, uint32_t seed, auto dist_factory) {
    using Dist = decltype(dist_factory());
    static_assert(
        std::same_as<Dist, std::uniform_real_distribution<float>>,
        "generate_uniform_simd requires std::uniform_real_distribution<float>");

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
        float rand_val = _mm_cvtss_f32(rng.generate_float_x4());
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

    size_t chunk_size = calculate_aligned_chunk_size<float>(output.size(), num_threads, simd_float_batch_size);
    auto chunks = create_chunks(output, num_threads, chunk_size);

    std::vector<std::jthread> threads;
    threads.reserve(num_threads);

    size_t thread_id = 0;
    for (auto chunk : chunks) {
        uint64_t thread_seed = calculate_thread_seed(seed, thread_id++);
        threads.emplace_back([chunk, thread_seed, dist_factory]() {
            generate_uniform_simd(chunk, static_cast<uint32_t>(thread_seed), dist_factory);
        });
    }
}

// ============================================================================
// Internal SIMD Generation (Normal Distribution - Box-Muller)
// ============================================================================

__attribute__((target("aes,sse4.2"))) void generate_normal_simd(
    std::span<float> output, uint32_t seed, auto dist_factory) {
    using Dist = decltype(dist_factory());
    static_assert(
        std::same_as<Dist, std::normal_distribution<float>>,
        "generate_normal_simd requires std::normal_distribution<float>");

    AesRng rng{seed};
    auto dist = dist_factory();
    auto params = dist.param();

    const float mean = params.mean();
    const float stddev = params.stddev();

    const __m128 mean_vec = _mm_set1_ps(mean);
    const __m128 stddev_vec = _mm_set1_ps(stddev);
    const float two_pi = 2.0f * std::numbers::pi_v<float>;

    // Process pairs of values (Box-Muller generates 2 values per iteration)
    // SSE processes 4 values at a time, so we get 8 normal values per iteration
    // (4 from z1, 4 from z2)
    size_t i = 0;
    for (; i + 8 <= output.size(); i += 8) {
        // Generate 4 uniform random values for U1 and 4 for U2
        __m128 u1 = rng.generate_float_x4();
        __m128 u2 = rng.generate_float_x4();

        // Extract to array for scalar math
        alignas(16) float u1_arr[4], u2_arr[4];
        _mm_store_ps(u1_arr, u1);
        _mm_store_ps(u2_arr, u2);

        // Box-Muller transform using scalar math functions
        alignas(16) float z1_arr[4], z2_arr[4];
        for (int j = 0; j < 4; ++j) {
            float r = std::sqrt(-2.0f * std::log(std::max(u1_arr[j], 1e-10f)));
            float theta = two_pi * u2_arr[j];
            z1_arr[j] = r * std::cos(theta);
            z2_arr[j] = r * std::sin(theta);
        }

        // Load back into SIMD registers
        __m128 z1 = _mm_load_ps(z1_arr);
        __m128 z2 = _mm_load_ps(z2_arr);

        // Scale and shift: z * stddev + mean
        z1 = _mm_add_ps(_mm_mul_ps(z1, stddev_vec), mean_vec);
        z2 = _mm_add_ps(_mm_mul_ps(z2, stddev_vec), mean_vec);

        // Store results
        _mm_storeu_ps(&output[i], z1);
        _mm_storeu_ps(&output[i + 4], z2);
    }

    // Handle remainder (< 8 elements remaining)
    fill_remainder_fallback(output, i, seed, dist_factory);
}

// Parallel SIMD generation for normal_distribution<float>
template <typename DistFactory>
__attribute__((target("aes,sse4.2"))) void generate_normal_simd_parallel(
    std::span<float> output, DistFactory dist_factory, uint32_t seed, size_t num_threads) {
    if (output.size() < parallel_min_size) [[unlikely]] {
        generate_normal_simd(output, seed, dist_factory);
        return;
    }

    size_t chunk_size = calculate_aligned_chunk_size<float>(output.size(), num_threads, simd_float_batch_size);
    auto chunks = create_chunks(output, num_threads, chunk_size);

    std::vector<std::jthread> threads;
    threads.reserve(num_threads);

    size_t thread_id = 0;
    for (auto chunk : chunks) {
        uint64_t thread_seed = calculate_thread_seed(seed, thread_id++);
        threads.emplace_back([chunk, thread_seed, dist_factory]() {
            generate_normal_simd(chunk, static_cast<uint32_t>(thread_seed), dist_factory);
        });
    }
}

// ============================================================================
// Internal SIMD Generation (bfloat16)
// ============================================================================

__attribute__((target("aes,sse4.2"))) void generate_uniform_simd_bfloat16(
    std::span<bfloat16> output, uint32_t seed, auto dist_factory) {
    using Dist = decltype(dist_factory());
    static_assert(
        std::same_as<Dist, std::uniform_real_distribution<float>>,
        "generate_uniform_simd_bfloat16 requires std::uniform_real_distribution<float>");

    AesRng rng{seed};
    auto dist = dist_factory();
    auto params = dist.param();

    const float min = params.a();
    const float max = params.b();
    const float range = max - min;

    const size_t num_batches = output.size() / simd_bf16_batch_size;

    // Process full batches
    for (size_t i = 0; i < num_batches; ++i) {
        auto bf16_values = rng.generate_bfloat16_x4();

        // Scale to [min, max) range
        for (size_t j = 0; j < simd_bf16_batch_size; ++j) {
            float val = static_cast<float>(bf16_values[j]);
            float scaled = val * range + min;
            output[i * simd_bf16_batch_size + j] = float_to_bfloat16(scaled);
        }
    }

    // Process remainder
    fill_remainder_fallback(output, num_batches * simd_bf16_batch_size, seed, dist_factory);
}

__attribute__((target("aes,sse4.2"))) void generate_uniform_simd_parallel_bfloat16(
    std::span<bfloat16> output, uint32_t seed, auto dist_factory, size_t num_threads) {
    if (output.size() < parallel_min_size) [[unlikely]] {
        generate_uniform_simd_bfloat16(output, seed, dist_factory);
        return;
    }

    size_t chunk_size = calculate_aligned_chunk_size<bfloat16>(output.size(), num_threads, simd_bf16_batch_size);
    auto chunks = create_chunks(output, num_threads, chunk_size);

    std::vector<std::jthread> threads;
    threads.reserve(num_threads);

    size_t thread_id = 0;
    for (auto chunk : chunks) {
        uint64_t thread_seed = calculate_thread_seed(seed, thread_id++);
        threads.emplace_back([chunk, thread_seed, dist_factory]() {
            generate_uniform_simd_bfloat16(chunk, static_cast<uint32_t>(thread_seed), dist_factory);
        });
    }
}

// ============================================================================
// Drop-in Replacement API
// ============================================================================

// Sequential generate - matches original random.hpp API
template <typename T, typename DistGenFunc>
inline void sequential_generate(std::span<T> seq, DistGenFunc dist_factory, uint32_t seed) noexcept {
    // SIMD fast path for float distributions
    if constexpr (std::same_as<T, float>) {
        using Dist = decltype(dist_factory());
        if constexpr (std::same_as<Dist, std::uniform_real_distribution<float>>) {
            generate_uniform_simd(seq, seed, dist_factory);
        } else if constexpr (std::same_as<Dist, std::normal_distribution<float>>) {
            generate_normal_simd(seq, seed, dist_factory);
        }
    }
    // SIMD fast path for bfloat16 distributions
    else if constexpr (std::same_as<T, bfloat16>) {
        using Dist = decltype(dist_factory());
        if constexpr (std::same_as<Dist, std::uniform_real_distribution<float>>) {
            generate_uniform_simd_bfloat16(seq, seed, dist_factory);
        } else if constexpr (std::same_as<Dist, std::normal_distribution<float>>) {
            // For normal distribution with bfloat16, generate as float then convert
            std::vector<float> temp(seq.size());
            generate_normal_simd(std::span<float>{temp}, seed, dist_factory);
            for (size_t i = 0; i < seq.size(); ++i) {
                seq[i] = float_to_bfloat16(temp[i]);
            }
        }
    }
}

// Parallel generate - matches original random.hpp API
template <typename T, typename DistGenFunc>
inline void parallel_generate(
    std::span<T> seq,
    DistGenFunc dist_factory,
    uint32_t seed,
    uint32_t max_threads = std::thread::hardware_concurrency()) noexcept {
    // SIMD fast path for float distributions
    if constexpr (std::same_as<T, float>) {
        using Dist = decltype(dist_factory());
        if constexpr (std::same_as<Dist, std::uniform_real_distribution<float>>) {
            generate_uniform_simd_parallel(seq, dist_factory, seed, max_threads);
        } else if constexpr (std::same_as<Dist, std::normal_distribution<float>>) {
            generate_normal_simd_parallel(seq, dist_factory, seed, max_threads);
        }
    }
    // SIMD fast path for bfloat16 distributions
    else if constexpr (std::same_as<T, bfloat16>) {
        using Dist = decltype(dist_factory());
        if constexpr (std::same_as<Dist, std::uniform_real_distribution<float>>) {
            generate_uniform_simd_parallel_bfloat16(seq, seed, dist_factory, max_threads);
        } else if constexpr (std::same_as<Dist, std::normal_distribution<float>>) {
            // For normal distribution with bfloat16, generate as float then convert
            std::vector<float> temp(seq.size());
            generate_normal_simd_parallel(std::span<float>{temp}, dist_factory, seed, max_threads);
            for (size_t i = 0; i < seq.size(); ++i) {
                seq[i] = float_to_bfloat16(temp[i]);
            }
        }
    }
}

}  // namespace ttml::core::sse
