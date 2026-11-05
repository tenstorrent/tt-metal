// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "internal/risc_attribs.h"
#include "experimental/ckernel_gguf_page.h"
#include "experimental/gguf_level_tables.h"

// RISC-V side of the Blackhole GGUF decoders (tt-llk sfpu/experimental/ckernel_sfpu_gguf_decode.h): the steps a data
// movement core or a TRISC runs on a GGUF page in L1 before it is unpacked to DEST. The page layout and the format ids
// are the LLK's (experimental/ckernel_gguf_page.h).
//   prepare_scales: fp32 copies of the page's fp16 scale fields, so the SFPU needs no fp16 conversion for them;
//   lookup_grid:    the codebook formats' grid indices to level codes, through the format's level table.

namespace experimental::gguf {

using Format = ckernel::gguf::Format;
using ckernel::gguf::page_layout;
using ckernel::gguf::PageLayout;

// Entries of a codebook format's level table (0 for the other formats).
template <Format FMT>
constexpr std::uint32_t table_entries() {
    switch (FMT) {
        case Format::IQ2_XXS:
        case Format::IQ3_XXS: return 256;
        case Format::IQ2_XS:
        case Format::IQ3_S: return 512;
        case Format::IQ2_S: return 1024;
        case Format::IQ1_S:
        case Format::IQ1_M: return 2048;
        default: return 0;
    }
}

template <Format FMT>
constexpr bool is_codebook() {
    return table_entries<FMT>() != 0;
}

// fp16 bits to fp32 bits, exact for every value (subnormals normalized, inf and NaN kept).
inline std::uint32_t fp16_to_fp32(const std::uint32_t h) {
    const std::uint32_t sign = (h & 0x8000u) << 16;
    std::uint32_t mag = h & 0x7FFFu;
    if (mag - 0x0400u < 0x7800u) {
        return sign | ((mag << 13) + 0x38000000u);
    }
    if (mag >= 0x7C00u) {
        return sign | 0x7F800000u | ((mag & 0x3FFu) << 13);
    }
    if (mag == 0) {
        return sign;
    }
    std::uint32_t exp = 113;
    while ((mag & 0x400u) == 0) {
        mag <<= 1;
        exp--;
    }
    return sign | (exp << 23) | ((mag & 0x3FFu) << 13);
}

// The fp32 copies of a page's fp16 scale rows: fp16 row raw_rows + f, half b (column 16p + 2c + b of word 2c + p) goes
// to rows R + 8f + 4b .. R + 8f + 4b + 3 (raw_rows = scale_rows - fp16_rows), R = scale_rows rounded up to 4, the same
// 16 words in each of the four rows, so that one SFPU load gives every lane its column's value. `scales` points at the
// page's scale fields. branch_free: a row of normal values converts with integer ops only; a row with a zero,
// subnormal, inf or NaN converts again through fp16_to_fp32. Both give the same bits.
template <Format FMT, bool branch_free = true>
inline void prepare_scales(volatile tt_l1_ptr std::uint32_t* scales) {
    constexpr PageLayout L = page_layout<FMT>();
    constexpr std::uint32_t raw_rows = L.scale_rows - L.fp16_rows;
    constexpr std::uint32_t R = ckernel::gguf::round_up(L.scale_rows, 4);
#pragma GCC unroll 1
    for (std::uint32_t f = 0; f < L.fp16_rows; f++) {
        volatile tt_l1_ptr std::uint32_t* in = scales + (raw_rows + f) * 16;
        std::uint32_t lo[16], hi[16];
        bool exact = !branch_free;
        if constexpr (branch_free) {
            // Normal halves: the 15 magnitude bits up by 13 and the exponent rebiased by 127 - 15. A half is normal
            // when its exponent is neither 0 (bit 15 of e + 0x7C00 clear) nor 31 (bit 15 of e + 0x0400 set).
            std::uint32_t normal = 0x80008000u;
#pragma GCC unroll 16
            for (std::uint32_t w = 0; w < 16; w++) {
                const std::uint32_t x = in[w];
                const std::uint32_t e = x & 0x7C007C00u;
                normal &= (e + 0x7C007C00u) & ~(e + 0x04000400u);
                lo[w] = (((x << 13) & 0x0FFFE000u) + 0x38000000u) | ((x << 16) & 0x80000000u);
                hi[w] = (((x >> 3) & 0x0FFFE000u) + 0x38000000u) | (x & 0x80000000u);
            }
            exact = normal != 0x80008000u;
        }
        if (exact) {
#pragma GCC unroll 1
            for (std::uint32_t w = 0; w < 16; w++) {
                const std::uint32_t x = in[w];
                lo[w] = fp16_to_fp32(x & 0xFFFFu);
                hi[w] = fp16_to_fp32(x >> 16);
            }
        }
#pragma GCC unroll 4
        for (std::uint32_t k = 0; k < 4; k++) {
#pragma GCC unroll 16
            for (std::uint32_t w = 0; w < 16; w++) {
                scales[(R + 8 * f + k) * 16 + w] = lo[w];
                scales[(R + 8 * f + 4 + k) * 16 + w] = hi[w];
            }
        }
    }
}

// The level table of a codebook format (nullptr for the others).
template <Format FMT>
constexpr const std::uint16_t* level_table() {
    switch (FMT) {
        case Format::IQ2_XXS: return levels::iq2_xxs;
        case Format::IQ2_XS: return levels::iq2_xs;
        case Format::IQ2_S: return levels::iq2_s;
        case Format::IQ3_XXS: return levels::iq3_xxs;
        case Format::IQ3_S: return levels::iq3_s;
        case Format::IQ1_S:
        case Format::IQ1_M: return levels::iq1_s;
        default: return nullptr;
    }
}

// Byte offset of a codebook page's index words: they follow 128 bytes of sign words, except IQ1_S / IQ1_M's.
template <Format FMT>
constexpr std::uint32_t index_offset() {
    return page_layout<FMT>().code_offset + (FMT == Format::IQ1_S || FMT == Format::IQ1_M ? 0 : 128);
}

// Codebook formats: the level codes of a page's grid indices, two 16-bit codes per output word, element i of a grid
// entry at bits w i (level_table). `indices` points at the page's index words, `levels` at the page's first byte.
//   IQ2_XXS, IQ3_XXS: 32 / 64 index words of four 8-bit indices.
//   IQ1_S, IQ1_M, IQ2_S: 32 index words of four low index bytes; 32-bit word j after them holds the high bits of
//     index words 2j (bytes 0..3 at bits 8, 8 + w, 24, 24 + w) and 2j + 1 (at 0, w, 16, 16 + w), w = 3 (IQ1_*) or 2
//     (IQ2_S); its other bits are the decoder's.
//   IQ2_XS: 32 index words of low index bytes; the ninth bit of index word m = 8l + c's byte k is bit 8k + 7 of word
//     16 (l / 2) + 2c + l % 2 of the 32 sign words before them.
//   IQ3_S: 64 index words of low index bytes; word k < 8 of the 12 after them holds the high bits of index words
//     8k + i, bytes 0..3 at bits i, 8 + i, 16 + i, 24 + i; byte 8 (m / 32) + m % 8 of words 8..11 is ORed into bits
//     12..15 of index word m's two codes when (m / 8) % 4 == 0 (low nibble into the first): the columns' scales.
// part / parts: this call looks up part `part` of `parts` equal shares of the index words.
template <Format FMT, typename Out>
inline __attribute__((always_inline)) void lookup_grid(
    const std::uint32_t* indices,
    const std::uint16_t* table,
    Out* levels,
    const std::uint32_t part = 0,
    const std::uint32_t parts = 1) {
    static_assert(is_codebook<FMT>(), "lookup_grid: not a codebook format");
    constexpr std::uint32_t words = FMT == Format::IQ3_S || FMT == Format::IQ3_XXS ? 64 : 32;
    const std::uint32_t m_begin = part * words / parts;
    const std::uint32_t m_end = (part + 1) * words / parts;
    if constexpr (FMT == Format::IQ3_S || FMT == Format::IQ3_XXS) {
        constexpr bool high = FMT == Format::IQ3_S;
        const std::uint32_t* hi = indices + 64;
        const std::uint32_t* sc = hi + 8;
#pragma GCC unroll 1
        for (std::uint32_t m0 = m_begin; m0 < m_end; m0 += 8) {
            // The loads of a batch first: independent L1 loads overlap, dependent ones do not.
            std::uint32_t x[8];
#pragma GCC unroll 8
            for (std::uint32_t b = 0; b < 8; b++) {
                x[b] = indices[m0 + b];
            }
            const std::uint32_t hw = high ? hi[m0 >> 3] : 0;
            std::uint32_t s0 = 0, s1 = 0;
            if (high && ((m0 >> 3) & 3) == 0) {
                s0 = sc[2 * (m0 >> 5)];
                s1 = sc[2 * (m0 >> 5) + 1];
            }
#pragma GCC unroll 8
            for (std::uint32_t b = 0; b < 8; b++) {
                constexpr std::uint32_t lo = 0x00ff00ffu, hi_mask = 0x01000100u;
                const std::uint32_t ac = (x[b] & lo) | ((hw << (8 - b)) & hi_mask);
                const std::uint32_t bd = ((x[b] >> 8) & lo) | ((hw >> b) & hi_mask);
                std::uint32_t g0 = table[ac & 0xffff];
                std::uint32_t g1 = table[bd & 0xffff];
                const std::uint32_t g2 = table[ac >> 16];
                const std::uint32_t g3 = table[bd >> 16];
                if (high && ((m0 >> 3) & 3) == 0) {
                    const std::uint32_t s = ((b < 4 ? s0 : s1) >> (8 * (b & 3))) & 0xff;
                    g0 |= (s & 15) << 12;
                    g1 |= (s >> 4) << 12;
                }
                levels[2 * (m0 + b)] = g0 | (g1 << 16);
                levels[2 * (m0 + b) + 1] = g2 | (g3 << 16);
            }
        }
    } else if constexpr (
        FMT == Format::IQ1_S || FMT == Format::IQ1_M || FMT == Format::IQ2_S || FMT == Format::IQ2_XS) {
        constexpr bool xs = FMT == Format::IQ2_XS;
        constexpr std::uint32_t w_bits = FMT == Format::IQ2_S ? 2 : 3;
        constexpr std::uint32_t lo = 0x00ff00ffu;
        constexpr std::uint32_t hi_mask = FMT == Format::IQ2_S ? 0x03000300u : xs ? 0x01000100u : 0x07000700u;
#pragma GCC unroll 1
        for (std::uint32_t m0 = m_begin; m0 < m_end; m0 += 8) {
            std::uint32_t x[8], hh[8];
#pragma GCC unroll 8
            for (std::uint32_t b = 0; b < 8; b++) {
                x[b] = indices[m0 + b];
            }
            if constexpr (xs) {
                const std::uint32_t* s = indices - 32 + 16 * (m0 >> 4) + ((m0 >> 3) & 1);
#pragma GCC unroll 8
                for (std::uint32_t b = 0; b < 8; b++) {
                    hh[b] = s[2 * b];
                }
            } else {
#pragma GCC unroll 4
                for (std::uint32_t b = 0; b < 4; b++) {
                    hh[b] = indices[32 + m0 / 2 + b];
                }
            }
#pragma GCC unroll 8
            for (std::uint32_t b = 0; b < 8; b++) {
                const std::uint32_t h = hh[xs ? b : b / 2];
                const std::uint32_t h_ac = (xs ? h << 1 : b % 2 ? h << 8 : h) & hi_mask;
                const std::uint32_t h_bd = (xs ? h >> 7 : b % 2 ? h << (8 - w_bits) : h >> w_bits) & hi_mask;
                const std::uint32_t ac = (x[b] & lo) | h_ac;
                const std::uint32_t bd = ((x[b] >> 8) & lo) | h_bd;
                const std::uint32_t g0 = table[ac & 0xffff];
                const std::uint32_t g1 = table[bd & 0xffff];
                const std::uint32_t g2 = table[ac >> 16];
                const std::uint32_t g3 = table[bd >> 16];
                levels[2 * (m0 + b)] = g0 | (g1 << 16);
                levels[2 * (m0 + b) + 1] = g2 | (g3 << 16);
            }
        }
    } else {
#pragma GCC unroll 1
        for (std::uint32_t m0 = m_begin; m0 < m_end; m0 += 8) {
            std::uint32_t x[8];
#pragma GCC unroll 8
            for (std::uint32_t b = 0; b < 8; b++) {
                x[b] = indices[m0 + b];
            }
#pragma GCC unroll 8
            for (std::uint32_t b = 0; b < 8; b++) {
                const std::uint32_t g0 = table[x[b] & 0xff];
                const std::uint32_t g1 = table[(x[b] >> 8) & 0xff];
                const std::uint32_t g2 = table[(x[b] >> 16) & 0xff];
                const std::uint32_t g3 = table[x[b] >> 24];
                levels[2 * (m0 + b)] = g0 | (g1 << 16);
                levels[2 * (m0 + b) + 1] = g2 | (g3 << 16);
            }
        }
    }
}

// Both steps for the page at L1 byte address `page`: what a page needs between its arrival in L1 and its unpack.
template <Format FMT, bool branch_free = true>
inline void prepare_page(const std::uint32_t page) {
    constexpr PageLayout L = page_layout<FMT>();
    if constexpr (L.fp16_rows > 0) {
        prepare_scales<FMT, branch_free>(reinterpret_cast<volatile tt_l1_ptr std::uint32_t*>(page + L.scale_offset));
    }
    if constexpr (is_codebook<FMT>()) {
        lookup_grid<FMT>(
            reinterpret_cast<const std::uint32_t*>(page + index_offset<FMT>()),
            level_table<FMT>(),
            reinterpret_cast<volatile tt_l1_ptr std::uint32_t*>(page));
    }
}

}  // namespace experimental::gguf
