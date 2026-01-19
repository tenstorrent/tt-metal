// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/constants.hpp>
#include <optional>
#include <tuple>

inline uint32_t nearest_n(uint32_t x, uint32_t n) { return ((x + n - 1) / n) * n; }

template <uint8_t max>
inline uint8_t nearest_pow_of_2_up_to_8(uint32_t x) {
    if (x == 0) {
        return 1;  // Handle edge case when x is 0
    }

    // Round up to nearest power of 2
    --x;          // Decrease x by 1 to handle exact powers of 2
    x |= x >> 1;  // Propagate the highest set bit
    x |= x >> 2;  // Propagate the highest set bit

    // Uncomment if want to support full range of uin32_t
    // x |= x >> 4;  // Propagate the highest set bit
    // x |= x >> 8;  // Propagate the highest set bit
    // x |= x >> 16;  // Propagate the highest set bit

    uint32_t result = x + 1;  // Add 1 to get the next power of 2

    // Cap the result at max
    return (result > max) ? max : result;
}

inline std::tuple<uint32_t, uint32_t, uint32_t> get_runtime_args(
    int cur_pos, int cur_batch, int core_num, int num_cores_per_batch, uint32_t k_chunk_size) {
    // No sliding window: process from beginning up to cur_pos
    uint32_t valid_seq_len = nearest_n(cur_pos + 1, k_chunk_size);
    uint32_t num_chunks_value = valid_seq_len / k_chunk_size;

    uint32_t k_chunk_start = 0;
    uint32_t k_chunk_end = 0;

    // Strided chunk distribution: core N gets chunks N, N+num_cores, N+2*num_cores, ...
    // This ensures each core reads from the same DRAM bank (round-robin sharding)
    // E.g., with 8 cores and 16 chunks:
    //   core 0 gets chunks 0, 8 (both on bank 1)
    //   core 1 gets chunks 1, 9 (both on bank 3)
    //   etc.
    if (num_cores_per_batch > int(num_chunks_value)) {
        // More cores than chunks: each active core gets 1 chunk
        int chunks_per_core = (core_num < int(num_chunks_value)) ? 1 : 0;
        k_chunk_start = core_num;
        k_chunk_end = k_chunk_start + chunks_per_core;
    } else {
        // More chunks than cores: strided distribution
        // Core N processes chunks: N, N+num_cores, N+2*num_cores, ...
        // We encode this as: k_chunk_start = first chunk for this core
        //                    k_chunk_end = k_chunk_start + num_chunks_for_core * num_cores (stride encoded)
        // But kernel expects contiguous range, so we return stride info differently:
        // k_chunk_start = core_num (first chunk index)
        // k_chunk_end encodes the count: we'll iterate with stride in the kernel
        int chunks_per_core = num_chunks_value / num_cores_per_batch;
        int residuals = num_chunks_value % num_cores_per_batch;
        int num_chunks_for_core = chunks_per_core + (core_num < residuals ? 1 : 0);

        // First chunk for this core
        k_chunk_start = core_num;
        // k_chunk_end = first chunk + (num_chunks - 1) * stride + 1
        // This way the kernel can iterate: for (k = start; k < end; k += stride)
        // where stride = num_cores_per_batch
        k_chunk_end =
            k_chunk_start + (num_chunks_for_core > 0 ? (num_chunks_for_core - 1) * num_cores_per_batch + 1 : 0);
    }

    return {num_chunks_value, k_chunk_start, k_chunk_end};
}

template <uint32_t Sk_chunk_t, uint32_t max_size>
inline uint32_t get_dynamic_Sk_chunk_t(int cur_pos) {
    if constexpr (Sk_chunk_t == 0) {
        // Cur_pos + 1 for position, but -1 for divup, so cancels out
        // ie. divup(a, b) = (a - 1) / b + 1
        uint32_t seq_len_in_tiles = cur_pos / tt::constants::TILE_HEIGHT + 1;

        // Use nearest power of 2 to nicely divide total CB size which is some factor of max_size
        // Technically, should not be an issue but seeing PCC issues when using 3 tiles eg.
        // - Can switch to nearest tile if this is fixed
        return nearest_pow_of_2_up_to_8<max_size>(seq_len_in_tiles);
    }
    return Sk_chunk_t;
}
