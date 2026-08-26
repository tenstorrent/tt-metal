// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <cstdint>
#include <span>
#include <thread>
#include <vector>

namespace ttml::core::rng {

// ============================================================================
// Constants
// ============================================================================

inline constexpr size_t thread_seed_shift_bits = 32;

inline size_t calculate_num_chunks(size_t total_size, size_t chunk_size) noexcept {
    return (total_size + chunk_size - 1) / chunk_size;
}

inline uint64_t calculate_chunk_seed(uint32_t base_seed, size_t chunk_id, size_t chunk_size) noexcept {
    return static_cast<uint64_t>(base_seed) + static_cast<uint64_t>(chunk_id * chunk_size);
}

// chunk_fn must be callable as chunk_fn(std::span<T>, uint32_t seed).
// Seed is per-chunk so output is identical regardless of thread count.
template <typename T, typename ChunkGenFn>
void generate_parallel_chunks(
    std::span<T> seq, ChunkGenFn chunk_fn, uint32_t seed, uint32_t max_threads = std::thread::hardware_concurrency()) {
    constexpr size_t min_size = 1 << 12;  // determined empirically that this is where we see an advantage over
                                          // sequential generation even with 2 threads.
    if (seq.size() < min_size) {
        chunk_fn(seq, seed);
        return;
    }

    // Fixed chunk size independent of thread count: seed is per-chunk so output
    // is identical regardless of how many threads process those chunks.
    static constexpr size_t CHUNK_SIZE = 512 * 512;
    const size_t num_threads =
        std::min(static_cast<size_t>(max_threads), static_cast<size_t>(std::thread::hardware_concurrency()));
    const size_t num_chunks = calculate_num_chunks(seq.size(), CHUNK_SIZE);
    const size_t actual_threads = std::min(num_threads, num_chunks);
    const size_t chunks_per_thread = num_chunks / actual_threads;
    const size_t leftover = num_chunks % actual_threads;

    std::vector<std::jthread> threads;
    threads.reserve(actual_threads);

    size_t start_chunk = 0;
    for (size_t t = 0; t < actual_threads; ++t) {
        const size_t end_chunk = start_chunk + chunks_per_thread + (t < leftover ? 1 : 0);
        threads.emplace_back([chunk_fn, seq, start_chunk, end_chunk, seed]() {
            for (size_t chunk = start_chunk; chunk < end_chunk; ++chunk) {
                const size_t offset = chunk * CHUNK_SIZE;
                const size_t size = std::min(CHUNK_SIZE, seq.size() - offset);
                uint32_t chunk_seed = calculate_chunk_seed(seed, chunk, CHUNK_SIZE);
                chunk_fn(seq.subspan(offset, size), chunk_seed);
            }
        });
        start_chunk = end_chunk;
    }
}

}  // namespace ttml::core::rng
