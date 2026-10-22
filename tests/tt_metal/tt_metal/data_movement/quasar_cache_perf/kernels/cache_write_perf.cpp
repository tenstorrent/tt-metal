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
#include "api/debug/device_print.h"
#include "cache_write_probe.hpp"
#include "dev_mem_map.h"                // MEM_L1_UNCACHED_BASE
#include "dev_msgs.h"                   // watcher mailbox
#include "experimental/kernel_args.h"   // get_arg(args::name)
#include "hostdev/debug_ring_buffer_common.h"
#include "risc_common.h"                // flush_l2_cache_range, L2_FLUSH_ADDR

namespace {
constexpr std::uint64_t kFillWord = 0x5A5A5A5A5A5A5A5AULL;
constexpr std::uint8_t kFillByte = 0x5A;

static_assert(sizeof(CacheWriteProbe) <= sizeof(debug_mpsc_ring_buf_msg_quasar_t));

struct CacheWriteProbeIds {
    std::uint32_t probe_id;
    std::uint32_t mode;
    std::uint32_t size_bytes;
    std::uint32_t num_iterations;
};

inline __attribute__((always_inline)) volatile tt_l1_ptr CacheWriteProbe* cache_write_probe() {
    return reinterpret_cast<volatile tt_l1_ptr CacheWriteProbe*>(GET_MAILBOX_ADDRESS_DEV(watcher.debug_ring_buf));
}

// phase is stored last so a host read sees the other fields for that phase.
inline __attribute__((always_inline)) void publish_cache_write_probe(
    const CacheWriteProbeIds& ids, CacheWriteProbePhase phase, std::uint32_t iteration, std::uint32_t flush_addr) {
    volatile tt_l1_ptr CacheWriteProbe* probe = cache_write_probe();
    probe->magic = kCacheWriteProbeMagic;
    probe->probe_id = ids.probe_id;
    probe->mode = ids.mode;
    probe->size_bytes = ids.size_bytes;
    probe->num_iterations = ids.num_iterations;
    probe->iteration = iteration;
    probe->flush_addr = flush_addr;
    probe->phase = static_cast<std::uint32_t>(phase);
}

// Write num_words 64-bit words.
inline __attribute__((always_inline)) void store_words(volatile std::uint64_t* dst64, std::uint32_t num_words) {
    for (std::uint32_t i = 0; i < num_words; i++) {
        dst64[i] = kFillWord;
    }
}

// Mode 0: uncached, byte-at-a-time stores; per-iteration completion fence (April-faithful).
// One print every 32 iterations. A print on every iteration paces the byte-store loop
// enough that the hang no longer reproduces.
inline __attribute__((always_inline)) void write_uncached_1b(
    volatile std::uint8_t* dst8, std::uint32_t size_bytes, std::uint32_t num_iterations) {
    for (std::uint32_t iter = 0; iter < num_iterations; iter++) {
        if ((iter & 31) == 0) {
            DEVICE_PRINT("uncached_1b size={} iter={} before stores\n", size_bytes, iter);
        }
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
    std::uint32_t num_iterations,
    const CacheWriteProbeIds& ids) {
    const std::uint32_t flush_addr = base_addr & ~std::uint32_t{63};
    for (std::uint32_t iter = 0; iter < num_iterations; iter++) {
        publish_cache_write_probe(ids, CacheWriteProbePhase::BeforeStores, iter, flush_addr);
        store_words(dst64, num_words);
        publish_cache_write_probe(ids, CacheWriteProbePhase::BeforeFlush, iter, flush_addr);
        flush_l2_cache_range(base_addr, size_bytes);
        publish_cache_write_probe(ids, CacheWriteProbePhase::AfterFlush, iter, flush_addr);
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
    const CacheWriteProbeIds probe_ids{get_arg(args::probe_id), mode, size_bytes, num_iterations};
    publish_cache_write_probe(probe_ids, CacheWriteProbePhase::Entered, 0, 0);

    DEVICE_PRINT("cache_write_perf start mode={} size={} iters={}\n", mode, size_bytes, num_iterations);

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
            write_cached_range_flush(dst64, num_words, base_addr, size_bytes, num_iterations, probe_ids);
        } else {
            write_cached_fast_flush(dst64, num_words, flush_start, flush_end, num_iterations);
        }
    }

    publish_cache_write_probe(probe_ids, CacheWriteProbePhase::Done, num_iterations, 0);

    DEVICE_PRINT("cache_write_perf done mode={} size={}\n", mode, size_bytes);

    DeviceTimestampedData("Test id", test_id);
    // "Number of transactions" = iteration count; host divides zone duration by this
    // to get the amortized per-write cost, and bandwidth = N * size / duration.
    DeviceTimestampedData("Number of transactions", num_iterations);
    DeviceTimestampedData("Transaction size in bytes", size_bytes);
    DeviceTimestampedData("Write path", mode);
}
