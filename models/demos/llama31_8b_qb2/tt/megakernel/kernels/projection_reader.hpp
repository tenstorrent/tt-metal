// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "api/dataflow/dataflow_api.h"

// One worker per bank owns consecutive K rows. Two workers per bank own
// alternating half rows and must retain strided source reads. All selected
// tile sizes are multiples of Blackhole's 64-byte DRAM read alignment.
template <uint32_t KBlock, uint32_t N, uint32_t Workers, uint32_t WeightBytes, typename Weight>
void read_projection_weights(const Weight& weight, uint32_t worker, uint32_t k, uint32_t destination) {
    static_assert(WeightBytes % 64 == 0);
    const uint32_t vc = PROJECTION_BANK_VC ? (worker / (Workers / 8)) % 4 : 1;
    if constexpr (Workers == 8 && PROJECTION_READER != 3) {
        noc_async_read<KBlock * N * WeightBytes>(
            weight.get_noc_addr(k * Workers * N + worker * N), destination, KBlock * N * WeightBytes, noc_index, vc);
    } else {
        static_assert(Workers == 8 || Workers == 16);
        for (uint32_t row = 0; row < KBlock; ++row) {
            noc_async_read<N * WeightBytes>(
                weight.get_noc_addr((k + row) * Workers * N + worker * N),
                destination + row * N * WeightBytes, N * WeightBytes, noc_index, vc);
        }
    }
}

template <uint32_t A, uint32_t B, uint32_t KBlock, uint32_t N, uint32_t K, uint32_t Workers,
          uint32_t WeightBytes, typename Input, typename Weight>
void tuned_stream_projection(const Input& input, const Weight& weight, uint32_t worker,
                             uint64_t prefetched_base = 0, uint32_t prefetched_blocks = 0) {
    constexpr uint32_t blocks = K / KBlock;
    static_assert(K % KBlock == 0 && blocks >= 2);
#if PROJECTION_READER >= 2
    // Two or three physical slots; two DMA blocks in flight. Reservations include
    // both the unpublished in-flight block and the next block to issue.
    // TRIDs are reused only after their prior block has completed. This is
    // bounded independently of the number of layers or trace replays.
    cb_reserve_back(A, 2 * KBlock);
    cb_reserve_back(B, 2 * KBlock * N);
    const uint32_t a_start = get_write_ptr(A);
    const uint32_t b_start = get_write_ptr(B);
#endif
    for (uint32_t block = 0; block < blocks; ++block) {
#if PROJECTION_READER >= 2
        const uint32_t slot = block % PROJECTION_BUFFERS;
        const uint32_t trid = slot + 1;
        noc_async_read_set_trid(trid);
        // Each block has fewer than half the available transaction credits,
        // including packetization of a contiguous weight read.
        while (noc_available_transactions(noc_index, trid) < ((NOC_MAX_TRANSACTION_ID_COUNT + 1) / 2)) {}
        const uint32_t a = a_start + slot * KBlock * 2048;
        const uint32_t b = b_start + slot * KBlock * N * WeightBytes;
#else
        cb_reserve_back(A, KBlock);
        cb_reserve_back(B, KBlock * N);
        const uint32_t a = get_write_ptr(A);
        const uint32_t b = get_write_ptr(B);
#endif
#if PROJECTION_COALESCE_INPUT
        // Selected BF16 width shards contain a whole number of K blocks:
        // QKV16/GU8 divide16 tiles, O4 divides4, down7 divides7, head4 divides16.
        // All tiles of this block are contiguous on one source worker.
        noc_async_read<KBlock * 2048>(input.get_noc_addr(block * KBlock), a, KBlock * 2048);
#else
        for (uint32_t row = 0; row < KBlock; ++row) {
            noc_async_read_page(block * KBlock + row, input, a + row * 2048);
        }
#endif
        if (block < prefetched_blocks) {
            noc_async_read<KBlock * N * WeightBytes>(
                prefetched_base + block * KBlock * N * WeightBytes, b, KBlock * N * WeightBytes);
        } else {
            read_projection_weights<KBlock, N, Workers, WeightBytes>(weight, worker, block * KBlock, b);
        }
#if PROJECTION_READER >= 2
        if (block > 0) {
            noc_async_read_barrier_with_trid((block - 1) % PROJECTION_BUFFERS + 1);
            cb_push_back(A, KBlock);
            cb_push_back(B, KBlock * N);
            if (block + 1 < blocks) {
                cb_reserve_back(A, 2 * KBlock);
                cb_reserve_back(B, 2 * KBlock * N);
            }
        }
#else
        noc_async_read_barrier();
        cb_push_back(A, KBlock);
        cb_push_back(B, KBlock * N);
#endif
    }
#if PROJECTION_READER >= 2
    noc_async_read_barrier_with_trid((blocks - 1) % PROJECTION_BUFFERS + 1);
    cb_push_back(A, KBlock);
    cb_push_back(B, KBlock * N);
    noc_async_read_set_trid(0);
#endif
#if PROJECTION_BANK_VC
    // Stateful readers in subsequent phases/programs expect firmware VC1.
    // Configure the register without issuing an extra memory transaction.
    noc_async_read_one_packet_set_state<true>(weight.get_noc_addr(worker * N), WeightBytes, 1);
#endif
}
