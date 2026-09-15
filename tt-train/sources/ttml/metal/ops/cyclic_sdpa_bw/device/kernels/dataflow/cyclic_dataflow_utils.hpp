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

// mask[row, col] = 1 where col >= row (the transposed causal mask), else 0.
// Tile layout: four 16 x 16 faces, 0 top-left, 1 top-right, 2 bottom-left,
// 3 bottom-right, row-major within a face.
template <typename T, T kOne>
inline void fill_transposed_causal_mask_tile(uint32_t l1_addr) {
    T* p = reinterpret_cast<T*>(l1_addr);
    for (uint32_t face = 0; face < 4u; ++face) {
        const uint32_t row0 = (face >= 2u) ? kFaceRows : 0u;
        const uint32_t col0 = (face & 1u) ? kFaceRows : 0u;
        for (uint32_t h = 0; h < kFaceRows; ++h) {
            for (uint32_t w = 0; w < kFaceRows; ++w) {
                *p++ = (col0 + w >= row0 + h) ? kOne : T{0};
            }
        }
    }
}

inline void generate_transposed_causal_mask_tile(uint32_t cb_id) {
    cb_reserve_back(cb_id, 1);
    if (get_dataformat(cb_id) == DataFormat::Float32) {
        fill_transposed_causal_mask_tile<uint32_t, kFp32One>(get_write_ptr(cb_id));
    } else {
        fill_transposed_causal_mask_tile<uint16_t, kBf16One>(get_write_ptr(cb_id));
    }
    cb_push_back(cb_id, 1);
}

// The Src registers hold 19 bits: a Float32 unpacked into one keeps its sign,
// exponent and the top 10 mantissa bits, and the low 13 bits are dropped
// (truncated, not rounded -- measured). For the softmax statistics that is a
// relative error of up to 2^-10 in L and D, which the exponential turns into
// the same relative error on every P of a query row, and which the
// subtraction dP - D amplifies wherever the two nearly cancel. So each
// statistic travels in two parts: the value itself, whose top 19 bits the
// register keeps, and its remainder -- the value of the 13 bits the register
// drops -- as a separate small number that the register keeps 10 bits of.
// Together that is about 20 significant bits.

// The remainder of a Float32 after truncation to 19 bits, as a Float32 with
// the same sign: value(x) - value(x & 0xFFFFE000). Integer arithmetic only;
// the data-movement RISCs have no floating point unit.
inline uint32_t float_remainder_bits(uint32_t x) {
    const uint32_t m = x & 0x1FFFu;
    if (m == 0u) {
        return 0u;
    }
    const uint32_t e = (x >> 23) & 0xFFu;
    // Leading bit of the 13-bit remainder, p in 0..12; the remainder is
    // m * 2^(e - 150) = (m / 2^p) * 2^(e - 150 + p), so the exponent field
    // becomes e - 23 + p.
    uint32_t p = 0u;
    uint32_t v = m;
    if (v >> 8) { p += 8u; v >>= 8; }
    if (v >> 4) { p += 4u; v >>= 4; }
    if (v >> 2) { p += 2u; v >>= 2; }
    if (v >> 1) { p += 1u; }
    const int32_t ep = static_cast<int32_t>(e) - 23 + static_cast<int32_t>(p);
    if (ep <= 0) {
        return 0u;  // would be denormal: negligible against the value itself
    }
    const uint32_t mant = (m << (23u - p)) & 0x7FFFFFu;
    return (x & 0x80000000u) | (static_cast<uint32_t>(ep) << 23) | mant;
}

// Column 0 of a Float32 tile into row 0 of another: value r of the source
// (row r, column 0) lands at (row 0, column r) of the destination, with its
// sign flipped when kNegate is set. Only row 0 of the destination is
// written; the row broadcast reads nothing else. When rem_l1 is given, the
// remainder (see above) of each value goes to row 0 of that tile the same way.
template <bool kNegate = false>
inline void gather_statistic_row(uint32_t src_l1, uint32_t dst_l1, uint32_t rem_l1 = 0u) {
    const uint32_t* src = reinterpret_cast<const uint32_t*>(src_l1);
    uint32_t* dst = reinterpret_cast<uint32_t*>(dst_l1);
    uint32_t* rem = reinterpret_cast<uint32_t*>(rem_l1);
    constexpr uint32_t sign = kNegate ? 0x80000000u : 0u;
    for (uint32_t r = 0; r < 2u * kFaceRows; ++r) {
        // Source: face 0 for rows 0..15, face 2 for rows 16..31; column 0.
        const uint32_t src_face = (r < kFaceRows) ? 0u : 2u;
        const uint32_t src_idx = src_face * kFaceElems + (r % kFaceRows) * kFaceRows;
        // Destination: row 0 of face 0 for columns 0..15, face 1 for 16..31.
        const uint32_t dst_face = (r < kFaceRows) ? 0u : 1u;
        const uint32_t v = src[src_idx] ^ sign;
        dst[dst_face * kFaceElems + (r % kFaceRows)] = v;
        if (rem_l1 != 0u) {
            rem[dst_face * kFaceElems + (r % kFaceRows)] = float_remainder_bits(v);
        }
    }
}

// The remainders of column 0 of a Float32 tile, negated, as bfloat16 in
// column 0 of a bfloat16 tile (same rows). This is the matmul's form of the
// correction: a rank-one product with a column of ones adds -D's remainder
// to every row of dP^T. Only column 0 is written; the rest must be zero.
inline void gather_negated_remainder_column(uint32_t src_l1, uint32_t dst_l1) {
    const uint32_t* src = reinterpret_cast<const uint32_t*>(src_l1);
    uint16_t* dst = reinterpret_cast<uint16_t*>(dst_l1);
    for (uint32_t r = 0; r < 2u * kFaceRows; ++r) {
        const uint32_t face = (r < kFaceRows) ? 0u : 2u;
        const uint32_t idx = face * kFaceElems + (r % kFaceRows) * kFaceRows;
        const uint32_t rem = float_remainder_bits(src[idx] ^ 0x80000000u);
        dst[idx] = static_cast<uint16_t>(rem >> 16);  // bfloat16: the top half, truncated
    }
}

// The four statistic tiles the compute kernel takes per row tile of a
// packet, from the L and D tiles at l_l1 / d_l1: L and -D in row layout, L's
// remainder in row layout, -D's remainder as a bfloat16 column. Done by the
// writer RISC, which is otherwise idle while the reader relays packets, as
// soon as the reader has the statistics in L1.
inline void produce_statistic_tiles(
    uint32_t l_l1, uint32_t d_l1, uint32_t Bt, uint32_t interm_bytes,
    uint32_t cb_lse_row, uint32_t cb_u_row, uint32_t cb_lse_rem, uint32_t cb_u_rem) {
    cb_reserve_back(cb_lse_row, Bt);
    cb_reserve_back(cb_u_row, Bt);
    cb_reserve_back(cb_lse_rem, Bt);
    cb_reserve_back(cb_u_rem, Bt);
    const uint32_t lrow = get_write_ptr(cb_lse_row);
    const uint32_t urow = get_write_ptr(cb_u_row);
    const uint32_t lrem = get_write_ptr(cb_lse_rem);
    const uint32_t urem = get_write_ptr(cb_u_rem);
    const uint32_t rem_bytes = get_tile_size(cb_u_rem);
    for (uint32_t k = 0; k < Bt; ++k) {
        gather_statistic_row(l_l1 + k * interm_bytes, lrow + k * interm_bytes, lrem + k * interm_bytes);
        gather_statistic_row</* negate */ true>(d_l1 + k * interm_bytes, urow + k * interm_bytes);
        gather_negated_remainder_column(d_l1 + k * interm_bytes, urem + k * rem_bytes);
    }
    cb_push_back(cb_lse_row, Bt);
    cb_push_back(cb_u_row, Bt);
    cb_push_back(cb_lse_rem, Bt);
    cb_push_back(cb_u_rem, Bt);
}

// The reader pushes one page here once a timestep's L and D are in L1, and
// the writer waits on it before producing the statistic tiles. A buffer
// rather than an L1 word because the host resets buffer state every launch;
// a word in scratch L1 could carry the previous launch's count and pass a
// wait early.
constexpr uint32_t kStatsReadyCb = tt::CBIndex::c_31;

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
