// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Quasar DM-core cache-write performance kernel.
// Writes `size_bytes` to Tensix L1 and times the region, in one of four modes
// (the "Write path" arg):
//   0 = Uncached (1B stores)             -> uncached port (+4MB alias), byte-at-a-time
//   1 = Uncached (8B stores)             -> uncached port, 64-bit stores
//   2 = Cached + flush (fence per line)  -> cacheable 64-bit write, then flush_l2_cache_range
//                                           (library: 2 fences per 64B line; April's method)
//   3 = Cached + flush (single fence)    -> cacheable 64-bit write, then bare flush-reg writes
//                                           over the touched lines, one fence after all iters
//
// size_bytes is assumed to be a multiple of 8 (the host sweep only uses such sizes),
// so the 8-byte modes write exactly size_bytes/8 words with no sub-8B tail.
//
// Each mode repeats its write(+flush) region `num_iterations` times inside the timed
// DeviceZoneScopedN; the host divides the zone duration by num_iterations (stamped as
// "Number of transactions") to get the amortized per-write cost. Fencing discipline:
//   - Modes 0/1 (uncached) fence EVERY iteration, matching the April DirectSram kernel.
//   - Mode 2 flushes with the library flush_l2_cache_range every iteration (April method).
//   - Mode 3 applies the HW-guided flush optimization: bare flush-reg writes with NO
//     per-line/per-iteration fence, and a SINGLE completion fence after all iterations.
// Stores are volatile so the compiler cannot coalesce/reorder them.

#include <cstdint>
#include "api/dataflow/dataflow_api.h"  // pulls in DeviceZoneScopedN / DeviceTimestampedData
#include "dev_mem_map.h"                // MEM_L1_UNCACHED_BASE
#include "experimental/kernel_args.h"   // get_arg(args::name)
#include "risc_common.h"                // flush_l2_cache_range, L2_FLUSH_ADDR

namespace {
constexpr std::uint64_t kFillWord = 0x5A5A5A5A5A5A5A5AULL;
constexpr std::uint8_t kFillByte = 0x5A;

// Write num_words 64-bit words.
inline __attribute__((always_inline)) void store_words(volatile std::uint64_t* dst64, std::uint32_t num_words) {
    for (std::uint32_t i = 0; i < num_words; i++) {
        dst64[i] = kFillWord;
    }
}

// Mode 0: uncached, byte-at-a-time stores; per-iteration completion fence (April-faithful).
inline __attribute__((always_inline)) void write_uncached_1b(
    volatile std::uint8_t* dst8, std::uint32_t size_bytes, std::uint32_t num_iterations) {
    for (std::uint32_t iter = 0; iter < num_iterations; iter++) {
        for (std::uint32_t i = 0; i < size_bytes; i++) {
            dst8[i] = kFillByte;
        }
        __asm__ __volatile__("fence" ::: "memory");
    }
}

// Mode 1: uncached, 8-byte stores; per-iteration completion fence (April-faithful DirectSRAM).
inline __attribute__((always_inline)) void write_uncached_8b(
    volatile std::uint64_t* dst64, std::uint32_t num_words, std::uint32_t num_iterations) {
    for (std::uint32_t iter = 0; iter < num_iterations; iter++) {
        store_words(dst64, num_words);
        __asm__ __volatile__("fence" ::: "memory");
    }
}

// Mode 2: cached 8-byte stores + library range flush (2 fences per 64B line). April's method.
inline __attribute__((always_inline)) void write_cached_range_flush(
    volatile std::uint64_t* dst64,
    std::uint32_t num_words,
    std::uint32_t base_addr,
    std::uint32_t size_bytes,
    std::uint32_t num_iterations) {
    for (std::uint32_t iter = 0; iter < num_iterations; iter++) {
        store_words(dst64, num_words);
        flush_l2_cache_range(base_addr, size_bytes);
    }
}

// Mode 3: cached 8-byte stores + HW-optimal flush (bare flush-reg writes over the touched
// 64B lines, no per-line/per-iteration fence; single completion fence after all iters).
inline __attribute__((always_inline)) void write_cached_fast_flush(
    volatile std::uint64_t* dst64,
    std::uint32_t num_words,
    std::uint32_t flush_start,
    std::uint32_t flush_end,
    std::uint32_t num_iterations) {
    volatile std::uint64_t* flush_reg = (volatile std::uint64_t*)L2_FLUSH_ADDR;
    for (std::uint32_t iter = 0; iter < num_iterations; iter++) {
        store_words(dst64, num_words);
        for (std::uint32_t a = flush_start; a < flush_end; a += 64) {
            *flush_reg = (std::uint64_t)a;
        }
    }
    __asm__ __volatile__("fence" ::: "memory");
}
}  // namespace

void kernel_main() {
    std::uint32_t base_addr = get_arg(args::base_addr);
    std::uint32_t size_bytes = get_arg(args::size_bytes);
    std::uint32_t mode = get_arg(args::write_path);  // 0=unc 1B, 1=unc 8B, 2=cached range, 3=cached fast
    std::uint32_t num_iterations = get_arg(args::num_iterations);
    std::uint32_t test_id = get_arg(args::test_id);

    bool cached = (mode == 2 || mode == 3);
    std::uint32_t dst_addr = cached ? base_addr : (base_addr + MEM_L1_UNCACHED_BASE);

    volatile std::uint64_t* dst64 = (volatile std::uint64_t*)(uintptr_t)dst_addr;
    volatile std::uint8_t* dst8 = (volatile std::uint8_t*)(uintptr_t)dst_addr;

    std::uint32_t num_words = size_bytes >> 3;  // 8-byte stores (size_bytes is a multiple of 8)
    std::uint32_t flush_start = base_addr & ~(std::uint32_t)63;
    std::uint32_t flush_end = base_addr + size_bytes;

    {
        DeviceZoneScopedN("RISCV1");
        if (mode == 0) {
            write_uncached_1b(dst8, size_bytes, num_iterations);
        } else if (mode == 1) {
            write_uncached_8b(dst64, num_words, num_iterations);
        } else if (mode == 2) {
            write_cached_range_flush(dst64, num_words, base_addr, size_bytes, num_iterations);
        } else {
            write_cached_fast_flush(dst64, num_words, flush_start, flush_end, num_iterations);
        }
    }

    DeviceTimestampedData("Test id", test_id);
    // "Number of transactions" = iteration count; host divides zone duration by this
    // to get the amortized per-write cost, and bandwidth = N * size / duration.
    DeviceTimestampedData("Number of transactions", num_iterations);
    DeviceTimestampedData("Transaction size in bytes", size_bytes);
    DeviceTimestampedData("Write path", mode);
}
