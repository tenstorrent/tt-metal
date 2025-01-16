// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <emmintrin.h>
#include "assert.hpp"
#include "aligned_allocator.hpp"
#include "umd/device/device_api_metal.h"

namespace tt::tt_metal {

static constexpr uint32_t MEMCPY_ALIGNMENT = sizeof(__m128i);

template <typename T>
using vector_memcpy_aligned = std::vector<T, tt::stl::aligned_allocator<T, MEMCPY_ALIGNMENT>>;

// Ideally would work by cachelines, but the min size is less than that
// Benchmarked to be approximately 1.4x - 1.8x faster than std::memcpy
// TODO: Revisit this w/ regard to possibly eliminating min sizes and orphan writes at the end
// TODO: ditto alignment isues
template <bool debug_sync = false>
static inline void memcpy_to_device(void* __restrict dst, const void* __restrict src, size_t n) {
    TT_ASSERT((uintptr_t)dst % MEMCPY_ALIGNMENT == 0);

    static constexpr uint32_t inner_loop = 8;
    static constexpr uint32_t inner_blk_size = inner_loop * sizeof(__m256i);

    uint8_t* src8 = (uint8_t*)src;
    uint8_t* dst8 = (uint8_t*)dst;

    if (size_t num_lines = n / inner_blk_size) {
        for (size_t i = 0; i < num_lines; ++i) {
            for (size_t j = 0; j < inner_loop; ++j) {
                __m256i blk = _mm256_loadu_si256((const __m256i*)src8);
                _mm256_stream_si256((__m256i*)dst8, blk);
                src8 += sizeof(__m256i);
                dst8 += sizeof(__m256i);
            }
            n -= inner_blk_size;
        }
    }

    if (n > 0) {
        if (size_t num_lines = n / sizeof(__m256i)) {
            for (size_t i = 0; i < num_lines; ++i) {
                __m256i blk = _mm256_loadu_si256((const __m256i*)src8);
                _mm256_stream_si256((__m256i*)dst8, blk);
                src8 += sizeof(__m256i);
                dst8 += sizeof(__m256i);
            }
            n -= num_lines * sizeof(__m256i);
        }
        if (size_t num_lines = n / sizeof(__m128i)) {
            for (size_t i = 0; i < num_lines; ++i) {
                __m128i blk = _mm_loadu_si128((const __m128i*)src8);
                _mm_stream_si128((__m128i*)dst8, blk);
                src8 += sizeof(__m128i);
                dst8 += sizeof(__m128i);
            }
            n -= n / sizeof(__m128i) * sizeof(__m128i);
        }
        if (n > 0) {
            for (size_t i = 0; i < n / sizeof(int32_t); ++i) {
                _mm_stream_si32((int32_t*)dst8, *(int32_t*)src8);
                src8 += sizeof(int32_t);
                dst8 += sizeof(int32_t);
            }
            n -= n / sizeof(int32_t) * sizeof(int32_t);
            // Copying the last few bytes (n < 4).
            // Overrunning dst buffer is safe, because the actual allocated space for dst is guaranteed to be at least 4
            // byte aligned.
            if (n > 0) {
                int32_t val = 0;
                std::memcpy(&val, src8, n);
                _mm_stream_si32((int32_t*)dst8, val);
            }
        }
    }
    if constexpr (debug_sync) {
        tt_driver_atomics::sfence();
    }
}

}  // namespace tt::tt_metal
