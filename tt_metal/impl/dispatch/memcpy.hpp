// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <tt_stl/assert.hpp>
#include <tt_stl/aligned_allocator.hpp>
#include <umd/device/driver_atomics.hpp>

#include <tt-metalium/vector_aligned.hpp>

#if defined(__x86_64__) || defined(__i386__)
#include <emmintrin.h>

#define LOAD_STREAM_32()                                                               \
    do {                                                                               \
        _mm256_stream_si256((__m256i*)dst8, _mm256_loadu_si256((const __m256i*)src8)); \
        src8 += sizeof(__m256i);                                                       \
        dst8 += sizeof(__m256i);                                                       \
    } while (0)

#define LOAD_STREAM_16()                                                         \
    do {                                                                         \
        _mm_stream_si128((__m128i*)dst8, _mm_loadu_si128((const __m128i*)src8)); \
        src8 += sizeof(__m128i);                                                 \
        dst8 += sizeof(__m128i);                                                 \
    } while (0)

#define LOAD_STREAM_4()                                   \
    do {                                                  \
        _mm_stream_si32((int32_t*)dst8, *(int32_t*)src8); \
        src8 += sizeof(int32_t);                          \
        dst8 += sizeof(int32_t);                          \
    } while (0)

#define LOAD_STREAM_4_UNALIGNED()                 \
    do {                                          \
        int32_t val = 0;                          \
        std::memcpy(&val, src8, sizeof(int32_t)); \
        _mm_stream_si32((int32_t*)dst8, val);     \
        src8 += sizeof(int32_t);                  \
        dst8 += sizeof(int32_t);                  \
    } while (0)

#endif  // x86

namespace tt::tt_metal {

// Ideally would work by cachelines, but the min size is less than that
// Benchmarked to be approximately 1.4x - 1.8x faster than std::memcpy
// TODO: Revisit this w/ regard to possibly eliminating min sizes and orphan writes at the end
// TODO: ditto alignment issues
#if defined(__x86_64__) || defined(__i386__)
template <bool debug_sync = false>
void memcpy_to_device(void* __restrict dst, const void* __restrict src, size_t n) {
    // Ensure destination is properly aligned for optimal SIMD performance
    TT_ASSERT((uintptr_t)dst % MEMCPY_ALIGNMENT == 0);

    // Configuration for bulk processing: inner loop processes 8 x 32-byte operations
    // This creates 256-byte blocks (8 * 32 = 256 bytes) for maximum throughput
    constexpr uint32_t inner_loop = 8;
    constexpr uint32_t inner_blk_size = inner_loop * sizeof(__m256i);  // 256 bytes

    const auto* src8 = static_cast<const uint8_t *>(src);
    auto* dst8 = static_cast<uint8_t*>(dst);

    size_t num_lines = n / inner_blk_size;  // Number of 256-byte blocks to process

    // PHASE 1: Process 256-byte blocks 32 bytes at a time
    // This is the main bulk processing phase for maximum efficiency
    if (num_lines > 0) {
        // Handle potential misalignment by processing a single 16-byte chunk first
        // This ensures subsequent 32-byte operations are properly aligned
        // WARNING: This does not cover the case where dst is not 16-byte aligned
        if ((uintptr_t)dst8 % sizeof(__m256i) != 0) {
            LOAD_STREAM_16();
            n -= sizeof(__m128i);
            num_lines = n / inner_blk_size;  // Recalculate after alignment adjustment
        }

        // Main bulk processing loop: Each iteration processes a 256 byte block. Blocks are processed 32 bytes at a time.
        for (size_t i = 0; i < num_lines; ++i) {
            for (size_t j = 0; j < inner_loop; ++j) {
                LOAD_STREAM_32();
            }
            n -= inner_blk_size;
        }
    }

    // PHASE 2: Process remaining data that doesn't fill a complete 256-byte block
    if (n > 0) {
        // Phase 2.1: Process remaining 32-byte chunks
        num_lines = n / sizeof(__m256i);  // Number of 32-byte blocks to process
        if (num_lines > 0) {
            // Handle alignment for 32-byte operations if needed
            // WARNING: This does not cover the case where dst is not 16-byte aligned
            if ((uintptr_t)dst8 % sizeof(__m256i) != 0) {
                LOAD_STREAM_16();
                n -= sizeof(__m128i);
                num_lines = n / sizeof(__m256i);  // Recalculate after alignment adjustment
            }

            // Process individual 32-byte blocks
            for (size_t i = 0; i < num_lines; ++i) {
                LOAD_STREAM_32();
            }
            n -= num_lines * sizeof(__m256i);
        }

        // PHASE 2.2: Process remaining 16-byte chunks
        num_lines = n / sizeof(__m128i);  // Number of 16-byte blocks to process
        if (num_lines > 0) {
            for (size_t i = 0; i < num_lines; ++i) {
                LOAD_STREAM_16();
            }
            n -= num_lines * sizeof(__m128i);
        }

        // PHASE 2.3: Process remaining 4-byte chunks
        num_lines = n / sizeof(int32_t);  // Number of 4-byte blocks to process
        if (num_lines > 0) {
            if ((uintptr_t)src8 % sizeof(int32_t) != 0) {
                for (size_t i = 0; i < num_lines; ++i) {
                    LOAD_STREAM_4_UNALIGNED();
                }
            } else {
                for (size_t i = 0; i < num_lines; ++i) {
                    LOAD_STREAM_4();
                }
            }
            n -= num_lines * sizeof(int32_t);
        }

        // PHASE 2.4: Handle the final few bytes (< 4 bytes)
        // We are the ones in control of and allocating the dst buffer,
        // so writing a few bytes extra is okay because we are guaranteeing the size is adequate.
        if (n > 0) {
            int32_t val = 0;
            std::memcpy(&val, src8, n);
            _mm_stream_si32((int32_t*)dst8, val);
        }
    }

    // Optional memory fence for debugging/synchronization
    // Ensures all streaming stores complete before function returns
    if constexpr (debug_sync) {
        tt_driver_atomics::sfence();
    }
}

#elif __has_builtin(__builtin_nontemporal_store)

// Generic non-temporal store path for compilers that support the builtin (Clang; GCC does not).
// __builtin_nontemporal_store generates the best non-temporal store the target supports
// (e.g. STNP on AArch64) without requiring arch-specific headers.
// 32-byte vector chunks: on AArch64 the compiler pairs two Q-registers → STNP Q0, Q1, [addr].
typedef uint8_t __attribute__((vector_size(32), aligned(1))) nt_v256;
typedef uint8_t __attribute__((vector_size(16), aligned(1))) nt_v128;

template <bool debug_sync = false>
void memcpy_to_device(void* __restrict dst, const void* __restrict src, size_t n) {
    const auto* src8 = static_cast<const uint8_t*>(src);
    auto* dst8 = static_cast<uint8_t*>(dst);

    constexpr uint32_t inner_loop = 8;
    constexpr uint32_t inner_blk_size = inner_loop * 32;  // 256 bytes

    size_t num_lines = n / inner_blk_size;
    for (size_t i = 0; i < num_lines; ++i) {
        for (size_t j = 0; j < inner_loop; ++j) {
            nt_v256 chunk;
            __builtin_memcpy(&chunk, src8, 32);
            __builtin_nontemporal_store(chunk, (nt_v256*)dst8);
            src8 += 32;
            dst8 += 32;
        }
        n -= inner_blk_size;
    }

    num_lines = n / 32;
    for (size_t i = 0; i < num_lines; ++i) {
        nt_v256 chunk;
        __builtin_memcpy(&chunk, src8, 32);
        __builtin_nontemporal_store(chunk, (nt_v256*)dst8);
        src8 += 32;
        dst8 += 32;
    }
    n -= num_lines * 32;

    num_lines = n / 16;
    for (size_t i = 0; i < num_lines; ++i) {
        nt_v128 chunk;
        __builtin_memcpy(&chunk, src8, 16);
        __builtin_nontemporal_store(chunk, (nt_v128*)dst8);
        src8 += 16;
        dst8 += 16;
    }
    n -= num_lines * 16;

    num_lines = n / 4;
    for (size_t i = 0; i < num_lines; ++i) {
        uint32_t chunk;
        __builtin_memcpy(&chunk, src8, 4);
        __builtin_nontemporal_store(chunk, (uint32_t*)dst8);
        src8 += 4;
        dst8 += 4;
    }
    n -= num_lines * 4;

    if (n > 0) {
        uint32_t val = 0;
        __builtin_memcpy(&val, src8, n);
        __builtin_nontemporal_store(val, (uint32_t*)dst8);
    }

    if constexpr (debug_sync) {
        tt_driver_atomics::sfence();
    }
}

#else
// Fallback for other architectures
template <bool debug_sync = false>
__attribute((nonnull(1, 2))) static inline void memcpy_to_device(
    void* __restrict dst, const void* __restrict src, size_t n) {
    memcpy(dst, src, n);
    if constexpr (debug_sync) {
        tt_driver_atomics::sfence();
    }
}
#endif

}  // namespace tt::tt_metal

#if defined(__x86_64__) || defined(__i386__)
#undef LOAD_STREAM_32
#undef LOAD_STREAM_16
#undef LOAD_STREAM_4
#undef LOAD_STREAM_4_UNALIGNED
#endif
