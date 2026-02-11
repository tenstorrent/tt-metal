// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tuple>

inline uint32_t nearest_n(uint32_t x, uint32_t n) { return ((x + n - 1) / n) * n; }

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
