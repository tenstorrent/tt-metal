// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// SIMD Everywhere - portable SIMD across all architectures
// SIMDE provides portable implementations of SIMD intrinsics
// Enables native aliases to use original _mm* intrinsics on any architecture
#define SIMDE_ENABLE_NATIVE_ALIASES
#include <simde/x86/aes.h>
#include <simde/x86/sse.h>
#include <simde/x86/sse2.h>
#include <simde/x86/sse4.1.h>

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

inline constexpr size_t simd_float_batch_size = 4;  // 4 floats per SIMD vector
inline constexpr size_t simd_bf16_batch_size = 4;   // 4 bfloat16s per SIMD vector
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

// ============================================================================
// Portable AES RNG using SIMD Everywhere
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

    explicit AesRng(uint64_t seed = 0) noexcept :
        state_{_mm_set_epi64x(seed, seed ^ seed_mix_constant_1)},
        key_{_mm_set_epi64x(seed ^ seed_mix_constant_2, seed + seed_mix_constant_1)},
        increment_{_mm_set_epi64x(seed_mix_constant_1, seed_mix_constant_2)} {
        for (int i = 0; i < aes_warmup_rounds; ++i) {
            advance();
        }
    }

    // Advance RNG state using AES operations (portable via SIMDE)
    void advance() noexcept {
        state_ = _mm_aesenc_si128(state_, key_);
        state_ = _mm_aesenc_si128(state_, increment_);
        state_ = _mm_aesenc_si128(state_, key_);
        state_ = _mm_xor_si128(state_, increment_);
        increment_ = _mm_add_epi64(increment_, _mm_set_epi64x(1, 1));
    }

    // Generate 128-bit random integer
    [[nodiscard]]
    __m128i generate_128bit() noexcept {
        advance();
        return state_;
    }

    // Generate 4 floats in [0, 1) range using SIMD
    [[nodiscard]]
    __m128 generate_float_x4() noexcept {
        __m128i rand_int = generate_128bit();

        // Extract mantissa by shifting right 9 bits
        __m128i mantissa = _mm_srli_epi32(rand_int, float_mantissa_shift);

        // OR with exponent bias to get float in [1, 2)
        __m128i float_bits = _mm_or_si128(mantissa, _mm_set1_epi32(float_exponent_bias));

        // Cast to float and subtract 1.0 to get [0, 1)
        __m128 result = _mm_castsi128_ps(float_bits);
        return _mm_sub_ps(result, _mm_set1_ps(1.0f));
    }

    // Generate 4 bfloat16 values in [0, 1) range
    [[nodiscard]]
    std::array<bfloat16, 4> generate_bfloat16_x4() noexcept {
        __m128 floats = generate_float_x4();
        __m128i float_bits = _mm_castps_si128(floats);
        __m128i upper_bits = _mm_srli_epi32(float_bits, 16);

        alignas(16) uint32_t packed[4];
        _mm_store_si128((__m128i*)packed, upper_bits);

        return std::array<bfloat16, 4>{
            std::bit_cast<bfloat16>(static_cast<uint16_t>(packed[0])),
            std::bit_cast<bfloat16>(static_cast<uint16_t>(packed[1])),
            std::bit_cast<bfloat16>(static_cast<uint16_t>(packed[2])),
            std::bit_cast<bfloat16>(static_cast<uint16_t>(packed[3]))};
    }

    // Single random uint32 from buffered values
    [[nodiscard]]
    result_type operator()() noexcept {
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
// Portable SIMD Math Helpers using SIMDE
// ============================================================================

// Portable SIMD logarithm - uses standard library std::log for accuracy
inline __m128 _mm_log_ps(__m128 x) noexcept {
    // Extract individual floats and compute logarithms using std::log
    alignas(16) float values[4];
    simde_mm_store_ps(values, x);

    values[0] = std::log(values[0]);
    values[1] = std::log(values[1]);
    values[2] = std::log(values[2]);
    values[3] = std::log(values[3]);

    return simde_mm_load_ps(values);
}

// Portable SIMD square root
inline __m128 _mm_sqrt_ps_fast(__m128 x) noexcept {
    return _mm_sqrt_ps(x);
}

// ============================================================================
// SIMD Remainder Fill - all SIMD, no scalar fallback
// ============================================================================

template <typename T, typename DistFactory>
inline void fill_remainder_simd(
    std::span<T> output, size_t start_idx, uint32_t seed, DistFactory dist_factory) noexcept {
    if (start_idx >= output.size()) {
        return;
    }

    AesRng rng{seed + static_cast<uint32_t>(start_idx)};
    auto dist = dist_factory();
    auto params = dist.param();

    if constexpr (std::same_as<T, float>) {
        using Dist = decltype(dist_factory());
        if constexpr (std::same_as<Dist, std::uniform_real_distribution<float>>) {
            const float min = params.a();
            const float max = params.b();
            const float range = max - min;

            for (size_t i = start_idx; i < output.size(); ++i) {
                __m128 rand = rng.generate_float_x4();
                float val = _mm_cvtss_f32(rand);
                output[i] = min + val * range;
            }
        } else if constexpr (std::same_as<Dist, std::normal_distribution<float>>) {
            const float mean = params.mean();
            const float stddev = params.stddev();
            const float two_pi = 2.0f * std::numbers::pi_v<float>;

            for (size_t i = start_idx; i < output.size(); ++i) {
                __m128 u1 = rng.generate_float_x4();
                __m128 u2 = rng.generate_float_x4();

                float u1_val = _mm_cvtss_f32(u1);
                float u2_val = _mm_cvtss_f32(u2);

                u1_val = std::max(u1_val, 1e-10f);
                float r = std::sqrt(-2.0f * std::log(u1_val));
                float theta = two_pi * u2_val;
                float z = r * std::cos(theta);
                output[i] = z * stddev + mean;
            }
        }
    } else if constexpr (std::same_as<T, bfloat16>) {
        using Dist = decltype(dist_factory());
        if constexpr (std::same_as<Dist, std::uniform_real_distribution<float>>) {
            const float min = params.a();
            const float max = params.b();
            const float range = max - min;

            for (size_t i = start_idx; i < output.size(); ++i) {
                __m128 rand = rng.generate_float_x4();
                float val = _mm_cvtss_f32(rand);
                float scaled = min + val * range;
                output[i] = float_to_bfloat16(scaled);
            }
        } else if constexpr (std::same_as<Dist, std::normal_distribution<float>>) {
            const float mean = params.mean();
            const float stddev = params.stddev();
            const float two_pi = 2.0f * std::numbers::pi_v<float>;

            for (size_t i = start_idx; i < output.size(); ++i) {
                __m128 u1 = rng.generate_float_x4();
                __m128 u2 = rng.generate_float_x4();

                float u1_val = _mm_cvtss_f32(u1);
                float u2_val = _mm_cvtss_f32(u2);

                u1_val = std::max(u1_val, 1e-10f);
                float r = std::sqrt(-2.0f * std::log(u1_val));
                float theta = two_pi * u2_val;
                float z = r * std::cos(theta);
                float result = z * stddev + mean;
                output[i] = float_to_bfloat16(result);
            }
        }
    }
}

// ============================================================================
// Portable SIMD Generation (Float - Uniform)
// ============================================================================

void generate_uniform_simd(std::span<float> output, uint32_t seed, auto dist_factory) {
    using Dist = decltype(dist_factory());
    static_assert(
        std::same_as<Dist, std::uniform_real_distribution<float>>,
        "generate_uniform_simd requires std::uniform_real_distribution<float>");

    AesRng rng{seed};
    auto dist = dist_factory();
    auto params = dist.param();

    const float min = params.a();
    const float max = params.b();

    const size_t num_batches = output.size() / simd_float_batch_size;
    const __m128 range_vec = _mm_set1_ps(max - min);
    const __m128 min_vec = _mm_set1_ps(min);

    // Process full SIMD batches
    for (size_t i : std::views::iota(0u, num_batches)) {
        __m128 rand = rng.generate_float_x4();
        __m128 scaled = _mm_add_ps(_mm_mul_ps(rand, range_vec), min_vec);
        _mm_storeu_ps(&output[i * simd_float_batch_size], scaled);
    }

    // Process remainder with SIMD
    fill_remainder_simd(output, num_batches * simd_float_batch_size, seed, dist_factory);
}

// Parallel SIMD generation for uniform_real_distribution<float>
template <typename DistFactory>
void generate_uniform_simd_parallel(
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
// Portable SIMD Generation (Float - Normal via Box-Muller)
// ============================================================================

void generate_normal_simd(std::span<float> output, uint32_t seed, auto dist_factory) {
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
    const __m128 two_pi_vec = _mm_set1_ps(2.0f * std::numbers::pi_v<float>);
    const __m128 minus_two = _mm_set1_ps(-2.0f);
    const __m128 eps = _mm_set1_ps(1e-10f);

    // Process pairs using Box-Muller with SIMD logarithm
    size_t i = 0;
    for (; i + 8 <= output.size(); i += 8) {
        // Generate 4 uniform random values for U1 and 4 for U2
        __m128 u1 = rng.generate_float_x4();
        __m128 u2 = rng.generate_float_x4();

        // Clamp to avoid log(0)
        u1 = _mm_max_ps(u1, eps);
        u2 = _mm_max_ps(u2, eps);

        // Box-Muller: r = sqrt(-2 * ln(u1)) using portable SIMD functions
        __m128 log_u1 = _mm_log_ps(u1);
        __m128 r = _mm_sqrt_ps_fast(_mm_mul_ps(minus_two, log_u1));

        // theta = 2 * pi * u2
        __m128 theta = _mm_mul_ps(two_pi_vec, u2);

        // Extract for trigonometric functions (portable fallback to scalar)
        alignas(16) float r_arr[4], theta_arr[4];
        _mm_store_ps(r_arr, r);
        _mm_store_ps(theta_arr, theta);

        alignas(16) float z1_arr[4], z2_arr[4];
        for (int j = 0; j < 4; ++j) {
            z1_arr[j] = r_arr[j] * std::cos(theta_arr[j]);
            z2_arr[j] = r_arr[j] * std::sin(theta_arr[j]);
        }

        // Load back into SIMD registers
        __m128 z1 = _mm_load_ps(z1_arr);
        __m128 z2 = _mm_load_ps(z2_arr);

        // Scale and shift: z * stddev + mean (all SIMD)
        z1 = _mm_add_ps(_mm_mul_ps(z1, stddev_vec), mean_vec);
        z2 = _mm_add_ps(_mm_mul_ps(z2, stddev_vec), mean_vec);

        // Store results
        _mm_storeu_ps(&output[i], z1);
        _mm_storeu_ps(&output[i + 4], z2);
    }

    // Process remainder with SIMD
    fill_remainder_simd(output, i, seed, dist_factory);
}

// Parallel SIMD generation for normal_distribution<float>
template <typename DistFactory>
void generate_normal_simd_parallel(
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
// Portable SIMD Generation (bfloat16 - Uniform with Pure SIMD)
// ============================================================================

void generate_uniform_simd_bfloat16(std::span<bfloat16> output, uint32_t seed, auto dist_factory) {
    using Dist = decltype(dist_factory());
    static_assert(
        std::same_as<Dist, std::uniform_real_distribution<float>>,
        "generate_uniform_simd_bfloat16 requires std::uniform_real_distribution<float>");

    AesRng rng{seed};
    auto dist = dist_factory();
    auto params = dist.param();

    const float min = params.a();
    const float max = params.b();

    const size_t num_batches = output.size() / simd_bf16_batch_size;
    const __m128 range_vec = _mm_set1_ps(max - min);
    const __m128 min_vec = _mm_set1_ps(min);

    // Process full batches with pure SIMD operations
    for (size_t i = 0; i < num_batches; ++i) {
        __m128 floats = rng.generate_float_x4();
        __m128 scaled = _mm_add_ps(_mm_mul_ps(floats, range_vec), min_vec);

        // Convert to bfloat16 using SIMD
        __m128i float_bits = _mm_castps_si128(scaled);
        __m128i upper_bits = _mm_srli_epi32(float_bits, 16);

        alignas(16) uint32_t packed[4];
        _mm_store_si128((__m128i*)packed, upper_bits);

        for (size_t j = 0; j < simd_bf16_batch_size; ++j) {
            output[i * simd_bf16_batch_size + j] = std::bit_cast<bfloat16>(static_cast<uint16_t>(packed[j]));
        }
    }

    // Process remainder with SIMD
    fill_remainder_simd(output, num_batches * simd_bf16_batch_size, seed, dist_factory);
}

void generate_uniform_simd_parallel_bfloat16(
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
