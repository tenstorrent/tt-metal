// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// [sdpa_decode_lse] pair split: the K-chunk range of one half of a row, shared by the reader, the compute and the
// writer (included only under SDPA_LSE_PAIR_SPLIT, so the pair_split=False build is the stock code). The host mirror
// is planner.pair_workload_for_core; keep the two identical. The group split (GEMMA4_SDPA_GROUP_SPLIT) is the same
// split over SDPA_LSE_SPLIT_PARTS = G > 2 parts (the define is absent for the pair, G = 2).

#pragma once

#include <cstdint>
#include <tuple>

#include "ttnn/operations/transformer/sdpa_decode/device/kernels/rt_args_common.hpp"

// The finite statistics an empty half writes (bf16 bit patterns, packed twice per uint32): m = -30000 (stored as
// bf16 -29952) and l = 1, so lse = scale * m + ln(l) is hugely negative and the merge weight of the half is 0.
// planner.PAIR_EMPTY_M_PACKED / PAIR_EMPTY_L_PACKED hold the same values.
constexpr uint32_t PAIR_EMPTY_M_PACKED = 0xC6EAC6EA;
constexpr uint32_t PAIR_EMPTY_L_PACKED = 0x3F803F80;

// The number of parts a row's K chunks split into: 2 (the pair) unless the host defines it (the group split).
#ifndef SDPA_LSE_SPLIT_PARTS
#define SDPA_LSE_SPLIT_PARTS 2
#endif

// get_workload_for_core for part ``pair_half`` of ``parts`` of the row (no sliding window: the op refuses it). The
// row's n = ceil((cur_pos + 1) / k_chunk_size) K chunks split as part p = [ceil(p * n / parts), ceil((p + 1) * n /
// parts)) -- for the pair half 0 = [0, ceil(n / 2)), half 1 = [ceil(n / 2), n) -- and the device's num_cores cores
// split their part with the stock reversed rule: member 0 takes the part's last chunks and member c has data iff
// c < the part's chunk count (so a member index is an "active child" test, as in the stock with the row's chunk
// count). Returns the stock tuple -- PSt and num_chunks of the WHOLE row (the causal mask is built from cur_pos
// alone), this core's [chunk_start, chunk_end), window start 0 / chunk 0 -- plus the part's chunk count. A row with
// fewer chunks than parts leaves some parts empty (half 1 of a one-chunk row): every core gets [base, base).
template <uint32_t pair_half, uint32_t parts = SDPA_LSE_SPLIT_PARTS>
inline std::tuple<uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t> get_pair_workload_for_core(
    uint32_t cur_pos, uint32_t core_num, uint32_t num_cores, uint32_t k_chunk_size) {
    static_assert(parts >= 2 && pair_half < parts, "pair_half is a part index below parts");
    const uint32_t valid_seq_len = nearest_n(cur_pos + 1, k_chunk_size);
    const uint32_t num_chunks = valid_seq_len / k_chunk_size;
    const uint32_t half_base = (pair_half * num_chunks + parts - 1) / parts;
    const uint32_t half_chunks = ((pair_half + 1) * num_chunks + parts - 1) / parts - half_base;
    uint32_t chunk_start = half_base;
    uint32_t chunk_end = half_base;
    if (num_cores > half_chunks) {
        if (core_num < half_chunks) {
            chunk_start = half_base + (half_chunks - core_num - 1);
            chunk_end = chunk_start + 1;
        }
    } else {
        const uint32_t chunks_per_core = half_chunks / num_cores;
        const uint32_t residuals = half_chunks % num_cores;
        const uint32_t reversed_core_num = num_cores - core_num - 1;
        chunk_start = half_base + reversed_core_num * chunks_per_core +
                      (residuals < reversed_core_num ? residuals : reversed_core_num);
        chunk_end = chunk_start + chunks_per_core + (reversed_core_num < residuals ? 1 : 0);
    }
    return {valid_seq_len / tt::constants::TILE_HEIGHT, num_chunks, chunk_start, chunk_end, 0, 0, half_chunks};
}
