// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
// Dataflow helpers of sdpa_decode_splitfree: the stock paged decode SDPA's page-table walk, the transposed K and
// the V chunk reads, the mask tile fills and the causal / window mask generators (copied from
// ttnn/cpp/ttnn/operations/transformer/sdpa_decode/device/kernels/dataflow/dataflow_common.hpp and the sdpa
// dataflow_common.hpp it includes, trimmed to the non-multicast paged path with a full-tile bf16 mask), plus the
// reads of the cur_pos stick. Every L1 staging address here is a tile or stick start (64-byte aligned).
#pragma once

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/endpoints.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"

namespace sf {

// Given a tile row of the virtual sequence, the physical tile id in the paged cache [blocks, NKV, BLOCK_T*32, D].
template <uint32_t num_heads, uint32_t block_size_t, uint32_t Wt, uint32_t capacity_t>
inline uint32_t virtual_tile_row_to_physical_tile_id(
    uint32_t seq_tile_idx, uint32_t cur_head, const volatile tt_l1_ptr uint32_t* const page_table_ptr) {
    constexpr uint32_t block_stride = num_heads * block_size_t * Wt;
    const uint32_t head_offset = cur_head * block_size_t * Wt;
    if constexpr (capacity_t > 0) {
        seq_tile_idx %= capacity_t;
    }
    const uint32_t virtual_block = seq_tile_idx / block_size_t;
    const uint32_t physical_block = static_cast<uint32_t>(page_table_ptr[virtual_block]);
    const uint32_t block_row_offset = seq_tile_idx % block_size_t;
    return physical_block * block_stride + head_offset + block_row_offset * Wt;
}

// The cur_pos stick (B int32) into the write region of ``cb`` (reserved once, pushed by the caller when a
// consumer waits on it); returns the L1 pointer.
template <uint32_t cb, uint32_t stick_bytes, typename PosArgs>
inline volatile tt_l1_ptr uint32_t* read_cur_pos(Noc& noc, const PosArgs& pos_args, uint32_t pos_addr) {
    CircularBuffer cbo(cb);
    cbo.reserve_back(1);
    const uint32_t ptr = cbo.get_write_ptr();
    const auto reader = TensorAccessor(pos_args, pos_addr, stick_bytes);
    noc.async_read(reader, CoreLocalMem<uint32_t>(ptr), stick_bytes, {.page_id = 0}, {});
    noc.async_read_barrier();
    return reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ptr);
}

// K chunk: tile rows [row0, row0 + SK) of KV head ``head``, read transposed (tile (row, col) at CB index
// col * SK + row: the [DHt, SK] layout the QK matmul takes as K^T with the in-tile transpose).
template <
    uint32_t cb_k,
    uint32_t DHt,
    uint32_t SK,
    uint32_t num_heads,
    uint32_t block_size_t,
    uint32_t capacity_t,
    uint32_t reads_in_flight,
    typename KReader>
inline void read_k_chunk(
    Noc& noc, const KReader& k_reader, uint32_t head, uint32_t row0, const volatile tt_l1_ptr uint32_t* page_table) {
    constexpr uint32_t k_tile_bytes = get_tile_size(cb_k);
    CircularBuffer cb(cb_k);
    cb.reserve_back(SK * DHt);
    const uint32_t base = cb.get_write_ptr();
    uint32_t in_flight = 0;
    for (uint32_t row = 0; row < SK; ++row) {
        uint32_t write_ptr = base + row * k_tile_bytes;
        uint32_t tile_id = virtual_tile_row_to_physical_tile_id<num_heads, block_size_t, DHt, capacity_t>(
            row0 + row, head, page_table);
        for (uint32_t col = 0; col < DHt; ++col) {
            noc.async_read(k_reader, CoreLocalMem<uint32_t>(write_ptr), k_tile_bytes, {.page_id = tile_id}, {});
            tile_id += 1;
            write_ptr += SK * k_tile_bytes;
            if (++in_flight == reads_in_flight) {
                noc.async_read_barrier();
                in_flight = 0;
            }
        }
    }
    noc.async_read_barrier();
    cb.push_back(SK * DHt);
}

// V chunk: tile rows [row0, row0 + SK) of KV head ``head`` in row-major order ([SK, DHt]).
template <
    uint32_t cb_v,
    uint32_t DHt,
    uint32_t SK,
    uint32_t num_heads,
    uint32_t block_size_t,
    uint32_t capacity_t,
    uint32_t reads_in_flight,
    typename VReader>
inline void read_v_chunk(
    Noc& noc, const VReader& v_reader, uint32_t head, uint32_t row0, const volatile tt_l1_ptr uint32_t* page_table) {
    constexpr uint32_t v_tile_bytes = get_tile_size(cb_v);
    CircularBuffer cb(cb_v);
    cb.reserve_back(SK * DHt);
    uint32_t write_ptr = cb.get_write_ptr();
    uint32_t in_flight = 0;
    for (uint32_t row = 0; row < SK; ++row) {
        uint32_t tile_id = virtual_tile_row_to_physical_tile_id<num_heads, block_size_t, DHt, capacity_t>(
            row0 + row, head, page_table);
        for (uint32_t col = 0; col < DHt; ++col) {
            noc.async_read(v_reader, CoreLocalMem<uint32_t>(write_ptr), v_tile_bytes, {.page_id = tile_id}, {});
            tile_id += 1;
            write_ptr += v_tile_bytes;
            if (++in_flight == reads_in_flight) {
                noc.async_read_barrier();
                in_flight = 0;
            }
        }
    }
    noc.async_read_barrier();
    cb.push_back(SK * DHt);
}

// ── mask tiles (bf16 full tiles, written into the reserved block of a one-block CB) ───────────────────────────
constexpr uint32_t NEG_INF_PAIR = 0xFF80FF80u;  // two bf16 -inf

template <uint32_t tile_bytes>
inline void fill_tile(Noc& noc, uint32_t base, uint32_t tile_id, uint32_t val) {
    volatile tt_l1_ptr uint32_t* ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(base + tile_id * tile_bytes);
    constexpr int n = tile_bytes / 4;
    for (int k = 0; k < n; ++k) {
        ptr[k] = val;
    }
}

// Columns [cur_pos_in_tile + 1, 32) of every row -> partial_val; the rest zero (the stock fill_tile_partial).
template <uint32_t tile_bytes>
inline void fill_tile_partial(
    Noc& noc, uint32_t base, uint32_t tile_id, uint32_t cur_pos_in_tile, uint32_t partial_val) {
    constexpr int num_faces = 4;
    fill_tile<tile_bytes>(noc, base, tile_id, 0);
    if (cur_pos_in_tile == 31 || partial_val == 0) {
        return;
    }
    const uint16_t datum_val = partial_val >> 16;
    volatile tt_l1_ptr uint16_t* uint16_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint16_t*>(base + tile_id * tile_bytes);
    volatile tt_l1_ptr uint32_t* uint32_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(base + tile_id * tile_bytes);
    const int face_start = (cur_pos_in_tile < 15) ? 0 : 1;
    const uint32_t fill_pos_in_face = (cur_pos_in_tile + 1) % 16;
    if (face_start == 0) {
        constexpr int num_uint32_datums_tile_face = (16 * 16) / 2;
        for (int k = 1; k < num_faces; k += 2) {
            const uint32_t uint32_face_idx = k << 7;
            for (int j = 0; j < num_uint32_datums_tile_face; ++j) {
                uint32_ptr[uint32_face_idx + j] = partial_val;
            }
        }
    }
    const bool is_odd_pos_filled = fill_pos_in_face % 2 == 1;
    const uint32_t fill_pos_in_uint32_face = (fill_pos_in_face + 1) >> 1;
    constexpr uint32_t num_cols_in_face = 16;
    constexpr uint32_t num_rows_in_face = 16;
    constexpr uint32_t num_cols_in_uint32_face = num_cols_in_face >> 1;
    for (int k = face_start; k < num_faces; k += 2) {
        const uint32_t uint16_face_idx = k << 8;
        const uint32_t uint32_face_idx = k << 7;
        for (uint32_t face_row_idx = 0; face_row_idx < num_rows_in_face; ++face_row_idx) {
            if (is_odd_pos_filled) {
                uint16_ptr[uint16_face_idx + (fill_pos_in_face + num_cols_in_face * face_row_idx)] = datum_val;
            }
            for (uint32_t c = fill_pos_in_uint32_face; c < num_cols_in_uint32_face; ++c) {
                uint32_ptr[uint32_face_idx + (c + num_cols_in_uint32_face * face_row_idx)] = partial_val;
            }
        }
    }
}

// Columns [0, window_start_pos_in_tile) of every row -> partial_val; the rest zero (the stock
// fill_tile_partial_sliding_window).
template <uint32_t tile_bytes>
inline void fill_tile_partial_window(
    Noc& noc, uint32_t base, uint32_t tile_id, uint32_t window_start_pos_in_tile, uint32_t partial_val) {
    constexpr int num_faces = 4;
    fill_tile<tile_bytes>(noc, base, tile_id, 0);
    if (window_start_pos_in_tile == 0 || partial_val == 0) {
        return;
    }
    const uint16_t datum_val = partial_val >> 16;
    volatile tt_l1_ptr uint16_t* uint16_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint16_t*>(base + tile_id * tile_bytes);
    volatile tt_l1_ptr uint32_t* uint32_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(base + tile_id * tile_bytes);
    const int face_start = (window_start_pos_in_tile < 16) ? 0 : 1;
    if (face_start == 1) {
        constexpr int num_uint32_datums_tile_face = (16 * 16) / 2;
        for (int k = 0; k < num_faces; k += 2) {
            const uint32_t uint32_face_idx = k << 7;
            for (int j = 0; j < num_uint32_datums_tile_face; ++j) {
                uint32_ptr[uint32_face_idx + j] = partial_val;
            }
        }
    }
    const uint32_t fill_end_pos_in_face = window_start_pos_in_tile % 16;
    const bool is_odd_end_pos = fill_end_pos_in_face % 2 == 1;
    const uint32_t fill_end_pos_in_uint32_face = fill_end_pos_in_face >> 1;
    constexpr uint32_t num_cols_in_face = 16;
    constexpr uint32_t num_rows_in_face = 16;
    constexpr uint32_t num_cols_in_uint32_face = num_cols_in_face >> 1;
    const int target_face = (window_start_pos_in_tile < 16) ? 0 : 1;
    for (int k = target_face; k < num_faces; k += 2) {
        const uint32_t uint16_face_idx = k << 8;
        const uint32_t uint32_face_idx = k << 7;
        for (uint32_t face_row_idx = 0; face_row_idx < num_rows_in_face; ++face_row_idx) {
            for (uint32_t c = 0; c < fill_end_pos_in_uint32_face; ++c) {
                uint32_ptr[uint32_face_idx + (c + num_cols_in_uint32_face * face_row_idx)] = partial_val;
            }
            if (is_odd_end_pos && fill_end_pos_in_face > 0) {
                uint16_ptr[uint16_face_idx + ((fill_end_pos_in_face - 1) + num_cols_in_face * face_row_idx)] =
                    datum_val;
            }
        }
    }
}

// The causal mask of the chunk holding cur_pos: SK tiles (one Q tile row), tile i zero before the position's
// tile, the partial cut in it, -inf after (the stock generate_mask at PNHt 1, every tile filled in place).
template <uint32_t cb_mask, uint32_t SK>
inline void generate_tail_mask(Noc& noc, uint32_t cur_pos) {
    constexpr uint32_t tile_bytes = get_tile_size(cb_mask);
    const uint32_t cur_pos_in_chunk = cur_pos % (SK * 32);
    const uint32_t cur_pos_in_chunk_t = cur_pos_in_chunk / 32;
    const uint32_t cur_pos_in_tile = cur_pos_in_chunk % 32;
    CircularBuffer cb(cb_mask);
    cb.reserve_back(SK);
    const uint32_t base = cb.get_write_ptr();
    for (uint32_t i = 0; i < SK; ++i) {
        if (i < cur_pos_in_chunk_t) {
            fill_tile<tile_bytes>(noc, base, i, 0);
        } else if (i == cur_pos_in_chunk_t) {
            fill_tile_partial<tile_bytes>(noc, base, i, cur_pos_in_tile, NEG_INF_PAIR);
        } else {
            fill_tile<tile_bytes>(noc, base, i, NEG_INF_PAIR);
        }
    }
    cb.push_back(SK);
}

// The window mask of the first live chunk: -inf before window_start, zero from it on (the stock
// generate_sliding_window_mask at PNHt 1).
template <uint32_t cb_mask, uint32_t SK>
inline void generate_window_mask(Noc& noc, uint32_t window_start) {
    constexpr uint32_t tile_bytes = get_tile_size(cb_mask);
    const uint32_t start_in_chunk = window_start % (SK * 32);
    const uint32_t start_in_chunk_t = start_in_chunk / 32;
    const uint32_t start_in_tile = start_in_chunk % 32;
    CircularBuffer cb(cb_mask);
    cb.reserve_back(SK);
    const uint32_t base = cb.get_write_ptr();
    for (uint32_t i = 0; i < SK; ++i) {
        if (i < start_in_chunk_t) {
            fill_tile<tile_bytes>(noc, base, i, NEG_INF_PAIR);
        } else if (i == start_in_chunk_t) {
            fill_tile_partial_window<tile_bytes>(noc, base, i, start_in_tile, NEG_INF_PAIR);
        } else {
            fill_tile<tile_bytes>(noc, base, i, 0);
        }
    }
    cb.push_back(SK);
}

}  // namespace sf
