// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "api/dataflow/dataflow_api.h"
#if COMPACT_ACTIVATIONS
#include "compact_rows.hpp"
#endif

#if QKV_CUSTOM_MM || CUSTOM_GU || CUSTOM_O
// Custom unpack walks K faces contiguously, independent of CB page stride.
// Compact each unpublished block to512-byte8-row tiles; ring blocks retain
// original2048-byte page spacing. All non-row-zero lanes were zero upstream.
template <uint32_t Tiles>
void compact_custom_input(uint32_t base) {
    for (uint32_t tile = 0; tile < Tiles; ++tile) {
        auto* source = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(base + tile * 2048);
        auto* target = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(base + tile * 512);
        for (uint32_t i = 0; i < 8; ++i) {
            target[i] = source[i]; target[64 + i] = source[128 + i];
        }
    }
}
#endif

// One worker per bank owns consecutive K rows. Two workers per bank own
// alternating half rows and must retain strided source reads. All selected
// tile sizes are multiples of Blackhole's 64-byte DRAM read alignment.
template <uint32_t KBlock, uint32_t N, uint32_t Workers, uint32_t WeightBytes, typename Weight>
void read_projection_weights(const Weight& weight, uint32_t worker, uint32_t k, uint32_t destination) {
    static_assert(WeightBytes % 64 == 0);
    const uint32_t vc = PROJECTION_BANK_VC ? (worker / (Workers / 8)) % 4 : 1;
    if constexpr (GU_BANK_SPLIT && Workers == 16 && WeightBytes == 576) {
        // Raw tile permutation stores each half-bank's128 K rows contiguously.
        // The logical output column rank is still bank*2+half.
        const uint32_t bank = worker / 2, half = worker % 2;
        noc_async_read<KBlock * N * WeightBytes>(
            weight.get_noc_addr((half * 128 + k) * 8 * N + bank * N),
            destination, KBlock * N * WeightBytes, noc_index, vc);
    } else if constexpr (Workers == 8 && PROJECTION_READER != 3) {
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

// Fill an unpublished prefix of the phase's own weight ring while its
// activation is unavailable. No extra copy or DRAM bytes; the normal stream
// publishes each block with its activation and skips this prefix's DRAM read.
template <uint32_t B, uint32_t KBlock, uint32_t N, uint32_t Workers, uint32_t WeightBytes, typename Weight>
void prefetch_local_projection_weights(const Weight& weight, uint32_t worker) {
    static_assert(EARLY_WEIGHT_BLOCKS <= PROJECTION_BUFFERS);
    cb_reserve_back(B, EARLY_WEIGHT_BLOCKS * KBlock * N);
    const uint32_t base = get_write_ptr(B);
    for (uint32_t block = 0; block < EARLY_WEIGHT_BLOCKS; ++block) {
        read_projection_weights<KBlock, N, Workers, WeightBytes>(weight, worker, block * KBlock,
            base + block * KBlock * N * WeightBytes);
    }
    // This join is before the activation wait. It makes prefix readiness
    // explicit and keeps transaction bookkeeping bounded, including TRID0.
    noc_async_read_barrier();
}

template <uint32_t A, uint32_t B, uint32_t KBlock, uint32_t N, uint32_t K, uint32_t Workers,
          uint32_t WeightBytes, typename Input, typename Weight>
void tuned_stream_projection(const Input& input, const Weight& weight, uint32_t worker,
                             uint64_t prefetched_base = 0, uint32_t prefetched_blocks = 0, uint32_t local_prefetched_blocks = 0) {
    constexpr uint32_t blocks = K / KBlock;
#if COMPACT_ACTIVATIONS
    constexpr bool compact = (COMPACT_ACTIVATIONS & 2) || K == 128;
    constexpr uint32_t shard_tiles = K == 32 ? 4 : K == 112 ? 7 : 16;
    const uint32_t compact_base = get_write_ptr(31) + 128;
    static_assert(128 + PROJECTION_BUFFERS * KBlock * 64 <= 4096);
    if constexpr (compact) {
        if (initialize_layer_scratch(1)) {
            // Aliased input rings retain zero padding throughout this token.
            zero_compact_input<KBlock * PROJECTION_BUFFERS * 2048>(get_write_ptr(A));
        }
    }
#endif
    static_assert(K % KBlock == 0 && blocks >= 2);
#if PROJECTION_READER >= 2
    // A bounded window of DMA blocks in flight. Reservations include all
    // unpublished blocks plus the next block to issue.
    // TRIDs are reused only after their prior block has completed. This is
    // bounded independently of the number of layers or trace replays.
    constexpr uint32_t inflight = PROJECTION_LOOKAHEAD;
    static_assert(inflight >= 2 && inflight <= PROJECTION_BUFFERS && blocks >= inflight);
    cb_reserve_back(A, inflight * KBlock);
    cb_reserve_back(B, inflight * KBlock * N);
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
#if COMPACT_ACTIVATIONS
        if constexpr (compact) {
            const uint32_t tile = block * KBlock;
            noc_async_read<KBlock * 64>(input.get_noc_addr(tile / shard_tiles * shard_tiles) + (tile % shard_tiles) * 64,
                compact_base + slot * KBlock * 64, KBlock * 64);
        } else
#endif
        {
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
        }
        if (block < prefetched_blocks) {
            noc_async_read<KBlock * N * WeightBytes>(
                prefetched_base + block * KBlock * N * WeightBytes, b, KBlock * N * WeightBytes);
        } else if (block >= local_prefetched_blocks) {
            read_projection_weights<KBlock, N, Workers, WeightBytes>(weight, worker, block * KBlock, b);
        }
#if PROJECTION_READER >= 2
        if (block + 1 >= inflight) {
            const uint32_t completed = block + 1 - inflight;
            noc_async_read_barrier_with_trid(completed % PROJECTION_BUFFERS + 1);
#if COMPACT_ACTIVATIONS
            if constexpr (compact) {
                expand_bf16_rows<KBlock>(compact_base + (completed % PROJECTION_BUFFERS) * KBlock * 64,
                    a_start + (completed % PROJECTION_BUFFERS) * KBlock * 2048);
            }
#endif
#if QKV_CUSTOM_MM || CUSTOM_GU || CUSTOM_O
            if constexpr ((CUSTOM_O && A == 6 && B == 7 && K == 32) || (QKV_CUSTOM_MM && N == 6 && Workers == 8) || (CUSTOM_GU && (N == 28 || N == 14) && (Workers == 8 || Workers == 16))) {
                compact_custom_input<KBlock>(a_start + (completed % PROJECTION_BUFFERS) * KBlock * 2048);
            }
#endif
            cb_push_back(A, KBlock);
            cb_push_back(B, KBlock * N);
            if (block + 1 < blocks) {
                cb_reserve_back(A, inflight * KBlock);
                cb_reserve_back(B, inflight * KBlock * N);
            }
        }
#else
        noc_async_read_barrier();
        cb_push_back(A, KBlock);
        cb_push_back(B, KBlock * N);
#endif
    }
#if PROJECTION_READER >= 2
    for (uint32_t block = blocks + 1 - inflight; block < blocks; ++block) {
        noc_async_read_barrier_with_trid(block % PROJECTION_BUFFERS + 1);
#if COMPACT_ACTIVATIONS
        if constexpr (compact) {
            expand_bf16_rows<KBlock>(compact_base + (block % PROJECTION_BUFFERS) * KBlock * 64,
                a_start + (block % PROJECTION_BUFFERS) * KBlock * 2048);
        }
#endif
#if QKV_CUSTOM_MM || CUSTOM_GU || CUSTOM_O
        if constexpr ((CUSTOM_O && A == 6 && B == 7 && K == 32) || (QKV_CUSTOM_MM && N == 6 && Workers == 8) || (CUSTOM_GU && (N == 28 || N == 14) && (Workers == 8 || Workers == 16))) {
            compact_custom_input<KBlock>(a_start + (block % PROJECTION_BUFFERS) * KBlock * 2048);
        }
#endif
        cb_push_back(A, KBlock);
        cb_push_back(B, KBlock * N);
    }
    noc_async_read_set_trid(0);
#endif
#if PROJECTION_BANK_VC
    // Stateful readers in subsequent phases/programs expect firmware VC1.
    // Configure the register without issuing an extra memory transaction.
    noc_async_read_one_packet_set_state<true>(weight.get_noc_addr(worker * N), WeightBytes, 1);
#endif
}
