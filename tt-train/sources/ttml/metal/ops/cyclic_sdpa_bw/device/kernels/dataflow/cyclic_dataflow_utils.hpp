// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Small dataflow helpers shared by the cyclic backward's readers and writers.
//
// The compute kernel works in the transposed orientation: it forms S^T = K Q^T
// rather than S = Q K^T, so that P^T and dS^T -- the operands dV and dK need
// -- come out of the elementwise chain directly and no score tile is ever
// transposed. Two things on the dataflow side follow from that:
//
//   * the causal mask tile is the transpose of the usual one: S^T[j, i] is
//     live where the key index j is at most the query index i, which is the
//     upper triangle including the diagonal;
//   * the per-row statistics L and D, which arrive one value per row in
//     column 0 of a tile, are needed broadcast along *rows* of S^T -- so the
//     32 values of a tile are gathered into row 0 of a scratch tile, which
//     the compute kernel row-broadcasts. D goes in negated, because the
//     compute kernel seeds its dP^T registers with it and lets the matmul
//     accumulate on top. The RISC does this in a few hundred cycles while it
//     would otherwise wait on the compute kernel.

#pragma once

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

namespace cyclic_dataflow {

constexpr uint32_t kFaceRows = 16;
constexpr uint32_t kFaceElems = 256;
constexpr uint16_t kBf16One = 0x3F80;
constexpr uint32_t kFp32One = 0x3F800000u;

constexpr uint16_t kBf16MinusInf = 0xFF80;

// The additive form of the transposed causal mask: 0 where col >= row (the
// key index is at most the query index), -inf elsewhere. Added to S^T by the
// FPU before the exponential, which turns -inf into an exact 0. Tile layout:
// four 16 x 16 faces, 0 top-left, 1 top-right, 2 bottom-left, 3 bottom-right,
// row-major within a face.
inline void fill_additive_transposed_causal_mask_tile(uint32_t l1_addr) {
    uint16_t* p = reinterpret_cast<uint16_t*>(l1_addr);
    for (uint32_t face = 0; face < 4u; ++face) {
        const uint32_t row0 = (face >= 2u) ? kFaceRows : 0u;
        const uint32_t col0 = (face & 1u) ? kFaceRows : 0u;
        for (uint32_t h = 0; h < kFaceRows; ++h) {
            for (uint32_t w = 0; w < kFaceRows; ++w) {
                *p++ = (col0 + w >= row0 + h) ? uint16_t{0} : kBf16MinusInf;
            }
        }
    }
}

inline void fill_constant_bf16_tile(uint32_t l1_addr, uint16_t value) {
    uint16_t* p = reinterpret_cast<uint16_t*>(l1_addr);
    for (uint32_t i = 0; i < 4u * kFaceElems; ++i) {
        p[i] = value;
    }
}

// The two mask tiles of a diagonal block pair, bfloat16: tile 0 for the
// diagonal score tile (the triangle), tile 1 all -inf for the wholly masked
// tiles above it. Both are added to S^T by the FPU.
inline void generate_causal_mask_tiles(uint32_t cb_id) {
    cb_reserve_back(cb_id, 2);
    const uint32_t base = get_write_ptr(cb_id);
    fill_additive_transposed_causal_mask_tile(base);
    fill_constant_bf16_tile(base + get_tile_size(cb_id), kBf16MinusInf);
    cb_push_back(cb_id, 2);
}

// The Src registers hold 19 bits: a Float32 unpacked into one keeps its sign,
// exponent and the top 10 mantissa bits, and the low 13 bits are dropped
// (truncated, not rounded -- measured). For the softmax statistics that is a
// relative error of up to 2^-10 in L and D, which the exponential turns into
// the same relative error on every P of a query row, and which the
// subtraction dP - D amplifies wherever the two nearly cancel. So each
// statistic travels in two parts that both survive the register:
//
//     hi = x - z          where z = 2^(e-137) is the value of the lowest of
//                         the ten kept mantissa bits; hi has no bits below
//                         it, so the register keeps it exactly,
//     lo = z + rem        where rem is the value of the 13 dropped bits, as a
//                         Float32 whose exponent is z's and whose mantissa is
//                         those 13 bits shifted up -- hence no normalisation,
//                         and hi + lo = x exactly.
//
// The register keeps 10 bits of lo, a bfloat16 7; with hi that is 20 or 17
// significant bits of x. Integer arithmetic only: the data-movement RISCs
// have no floating point unit, and this runs once per row per timestep.
inline void split_statistic(uint32_t x, uint32_t& hi, uint32_t& lo) {
    const uint32_t sign = x & 0x80000000u;
    const uint32_t mag = x & 0x7FFFFFFFu;
    const uint32_t e = mag >> 23;
    if (e < 11u) {
        // z would be subnormal: x is below 2^-116, the whole of it fits.
        hi = x;
        lo = 0u;
        return;
    }
    // Subtracting z from the 19-bit pattern borrows into the exponent when
    // the kept mantissa bits are zero, and below that power of two the kept
    // bits are worth half as much: the result is 2^(e-128) (2 - 2^-9).
    hi = sign | (((mag & 0x7FE000u) != 0u) ? ((mag & 0x7FFFE000u) - 0x2000u) : (((e - 1u) << 23) | 0x7FC000u));
    lo = sign | ((e - 10u) << 23) | ((mag & 0x1FFFu) << 10);
}

// The statistic tiles of one row tile: from column 0 of the Float32 tile at
// src_l1 (one value per row), negated when kNegate is set,
//
//   * hi into row 0 of the Float32 tile at row_l1 (value r at column r), for
//     the compute kernel's row broadcast,
//   * lo, when rem_row_l1 is given, into row 0 of that Float32 tile the same
//     way, for the path that adds it on the SFPU,
//   * lo, when rem_col_l1 is given, as bfloat16 into column 0 of that tile
//     (row r), for the rank-one matmul term against a column of ones.
//
// Only row 0 / column 0 are written; the rest of each tile must be zero.
template <bool kNegate>
inline void gather_statistic(uint32_t src_l1, uint32_t row_l1, uint32_t rem_row_l1, uint32_t rem_col_l1) {
    const uint32_t* src = reinterpret_cast<const uint32_t*>(src_l1);
    uint32_t* row = reinterpret_cast<uint32_t*>(row_l1);
    uint32_t* rem_row = reinterpret_cast<uint32_t*>(rem_row_l1);
    uint16_t* rem_col = reinterpret_cast<uint16_t*>(rem_col_l1);
    constexpr uint32_t sign = kNegate ? 0x80000000u : 0u;
    for (uint32_t r = 0; r < 2u * kFaceRows; ++r) {
        // Source and column destination: face 0 for rows 0..15, face 2 for
        // rows 16..31; column 0. Row destination: row 0 of face 0 for
        // columns 0..15, face 1 for 16..31.
        const uint32_t col_idx = ((r < kFaceRows) ? 0u : 2u) * kFaceElems + (r % kFaceRows) * kFaceRows;
        const uint32_t row_idx = ((r < kFaceRows) ? 0u : 1u) * kFaceElems + (r % kFaceRows);
        uint32_t hi = 0;
        uint32_t lo = 0;
        split_statistic(src[col_idx] ^ sign, hi, lo);
        row[row_idx] = hi;
        if (rem_row_l1 != 0u) {
            rem_row[row_idx] = lo;
        }
        if (rem_col_l1 != 0u) {
            rem_col[col_idx] = static_cast<uint16_t>(lo >> 16);  // bfloat16: the top half, truncated
        }
    }
}

// The statistic tiles the compute kernel takes per row tile of a packet,
// from the L and D tiles at l_l1 / d_l1: -L and -D (their hi parts) in row
// layout, their lo parts as bfloat16 columns, and -L's lo part as a Float32
// row as well where the exponential carries the softmax scale (kFold false:
// the compute kernel then adds it on the SFPU and takes no column). Done by
// the writer RISC, which is otherwise idle while the reader relays packets,
// as soon as the statistics are in L1.
// The same tiles at explicit addresses (one packet slot's worth), for the
// relay: they travel with the packet from the core that loaded the row from
// DRAM, so the reader owns the buffers and this only fills them.
template <bool kFold>
inline void produce_statistic_tiles_at(
    uint32_t l_l1, uint32_t d_l1, uint32_t Bt, uint32_t interm_bytes, uint32_t rem_bytes,
    uint32_t lrow, uint32_t urow, uint32_t lrem_row, uint32_t lrem, uint32_t urem) {
    for (uint32_t k = 0; k < Bt; ++k) {
        if constexpr (kFold) {
            gather_statistic</* negate */ true>(
                l_l1 + k * interm_bytes, lrow + k * interm_bytes, 0u, lrem + k * rem_bytes);
        } else {
            gather_statistic</* negate */ true>(
                l_l1 + k * interm_bytes, lrow + k * interm_bytes, lrem_row + k * interm_bytes, 0u);
        }
        gather_statistic</* negate */ true>(
            d_l1 + k * interm_bytes, urow + k * interm_bytes, 0u, urem + k * rem_bytes);
    }
}

template <bool kFold>
inline void produce_statistic_tiles(
    uint32_t l_l1, uint32_t d_l1, uint32_t Bt, uint32_t interm_bytes,
    uint32_t cb_neg_lse_row, uint32_t cb_neg_u_row, uint32_t cb_neg_lse_rem_row,
    uint32_t cb_neg_lse_rem, uint32_t cb_neg_u_rem) {
    cb_reserve_back(cb_neg_lse_row, Bt);
    cb_reserve_back(cb_neg_u_row, Bt);
    cb_reserve_back(cb_neg_lse_rem_row, Bt);
    cb_reserve_back(cb_neg_lse_rem, Bt);
    cb_reserve_back(cb_neg_u_rem, Bt);
    const uint32_t lrow = get_write_ptr(cb_neg_lse_row);
    const uint32_t urow = get_write_ptr(cb_neg_u_row);
    const uint32_t lrem_row = get_write_ptr(cb_neg_lse_rem_row);
    const uint32_t lrem = get_write_ptr(cb_neg_lse_rem);
    const uint32_t urem = get_write_ptr(cb_neg_u_rem);
    const uint32_t rem_bytes = get_tile_size(cb_neg_u_rem);
    produce_statistic_tiles_at<kFold>(l_l1, d_l1, Bt, interm_bytes, rem_bytes, lrow, urow, lrem_row, lrem, urem);
    cb_push_back(cb_neg_lse_row, Bt);
    cb_push_back(cb_neg_u_row, Bt);
    cb_push_back(cb_neg_lse_rem_row, Bt);
    cb_push_back(cb_neg_lse_rem, Bt);
    cb_push_back(cb_neg_u_rem, Bt);
}

// The statistic block: what a packet carries for one row tile instead of
// the five prepared tiles -- 512 bytes, the 32 values of each part:
//
//   words   0.. 31  -L, hi part (Float32)
//   words  32.. 63  -D, hi part (Float32)
//   halves 128..159 -L, lo part as bfloat16
//   halves 160..191 -D, lo part as bfloat16
//   words  96..127  -L, lo part (Float32; the path with the scale in the
//                   exponential adds it as a row)
//
// Made once where the row's packet enters from DRAM, from the L and D
// tiles; expanded into the tiles the compute kernel takes on every core the
// packet visits, by that core's writer RISC, which has a timestep of lead.
constexpr uint32_t kStatBlockBytes = 512;

inline void gather_statistic_block(uint32_t l_l1, uint32_t d_l1, uint32_t block_l1) {
    const uint32_t* l = reinterpret_cast<const uint32_t*>(l_l1);
    const uint32_t* d = reinterpret_cast<const uint32_t*>(d_l1);
    uint32_t* w = reinterpret_cast<uint32_t*>(block_l1);
    uint16_t* h = reinterpret_cast<uint16_t*>(block_l1);
    for (uint32_t r = 0; r < 2u * kFaceRows; ++r) {
        const uint32_t col_idx = ((r < kFaceRows) ? 0u : 2u) * kFaceElems + (r % kFaceRows) * kFaceRows;
        uint32_t hi = 0;
        uint32_t lo = 0;
        split_statistic(l[col_idx] ^ 0x80000000u, hi, lo);
        w[r] = hi;
        w[96u + r] = lo;
        h[128u + r] = static_cast<uint16_t>(lo >> 16);
        split_statistic(d[col_idx] ^ 0x80000000u, hi, lo);
        w[32u + r] = hi;
        h[160u + r] = static_cast<uint16_t>(lo >> 16);
    }
}

// The block into the tiles: hi parts into row 0 of the row tiles (face 0
// for columns 0..15, face 1 for 16..31), lo parts into column 0 of the
// bfloat16 tiles (face 0 for rows 0..15, face 2 for 16..31), and with
// kFold false -L's lo part into row 0 of its Float32 row tile too. Only
// those rows and columns are written; the tiles are zeroed once at start.
template <bool kFold>
inline void expand_statistic_block(
    uint32_t block_l1, uint32_t lrow, uint32_t urow, uint32_t lrem_row, uint32_t lrem, uint32_t urem) {
    const uint32_t* w = reinterpret_cast<const uint32_t*>(block_l1);
    const uint16_t* h = reinterpret_cast<const uint16_t*>(block_l1);
    uint32_t* lr = reinterpret_cast<uint32_t*>(lrow);
    uint32_t* ur = reinterpret_cast<uint32_t*>(urow);
    uint32_t* lrr = reinterpret_cast<uint32_t*>(lrem_row);
    uint16_t* lc = reinterpret_cast<uint16_t*>(lrem);
    uint16_t* uc = reinterpret_cast<uint16_t*>(urem);
    for (uint32_t r = 0; r < 2u * kFaceRows; ++r) {
        const uint32_t row_idx = ((r < kFaceRows) ? 0u : 1u) * kFaceElems + (r % kFaceRows);
        const uint32_t col_idx = ((r < kFaceRows) ? 0u : 2u) * kFaceElems + (r % kFaceRows) * kFaceRows;
        lr[row_idx] = w[r];
        ur[row_idx] = w[32u + r];
        if constexpr (kFold) {
            lc[col_idx] = h[128u + r];
        } else {
            lrr[row_idx] = w[96u + r];
        }
        uc[col_idx] = h[160u + r];
    }
}

// The reader pushes one page here once a timestep's L and D are in L1, and
// the writer waits on it before producing the statistic tiles. In the relay
// the writer answers on kStatsDoneCb (the L/D scratch buffer's pages, free
// for this since only their memory is used there) once the block is made,
// and the reader forwards it. A buffer
// rather than an L1 word because the host resets buffer state every launch;
// a word in scratch L1 could carry the previous launch's count and pass a
// wait early.
constexpr uint32_t kStatsReadyCb = tt::CBIndex::c_31;
constexpr uint32_t kStatsDoneCb = tt::CBIndex::c_5;

// A bfloat16 tile with 1.0 down column 0 and 0 elsewhere: the left factor of
// the rank-one correction above.
inline void generate_ones_column_tile(uint32_t cb_id) {
    cb_reserve_back(cb_id, 1);
    volatile tt_l1_ptr uint16_t* p = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(get_write_ptr(cb_id));
    for (uint32_t i = 0; i < 4u * kFaceElems; ++i) {
        p[i] = 0u;
    }
    for (uint32_t r = 0; r < 2u * kFaceRows; ++r) {
        const uint32_t face = (r < kFaceRows) ? 0u : 2u;
        p[face * kFaceElems + (r % kFaceRows) * kFaceRows] = kBf16One;
    }
    cb_push_back(cb_id, 1);
}

// The whole tile, once, so the rows the gather never writes hold zeros.
inline void zero_tile(uint32_t l1_addr, uint32_t bytes) {
    volatile tt_l1_ptr uint32_t* p = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l1_addr);
    for (uint32_t i = 0; i < bytes / 4u; ++i) {
        p[i] = 0u;
    }
}

}  // namespace cyclic_dataflow
