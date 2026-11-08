// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#if __has_include("experimental/ckernel_gguf_page.h")
#include "experimental/ckernel_gguf_page.h"
#else  // host code, which has no tt-llk include directory
#include "tt_metal/tt-llk/tt_llk_blackhole/common/inc/experimental/ckernel_gguf_page.h"
#endif
#include "experimental/gguf_level_tables.h"
#if defined(__riscv)
#include "internal/risc_attribs.h"
#endif
#if defined(KERNEL_BUILD)
#include "api/debug/assert.h"
#endif

// Steps a RISC-V core runs on a GGUF page in L1 before its unpack (page layout: experimental/ckernel_gguf_page.h).
// lookup_grid also runs on the host; prepare_scales and prepare_page are RISC-V only.

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

#if defined(__riscv)
// fp32 copies of the page's fp16 scale rows, four rows per fp16 half-row (ckernel_gguf_page.h); branch_free converts
// normal rows with integer ops and the others through fp16_to_fp32, with the same bits.
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
#endif

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

// The level table for lookups on a RISC-V core: tables of up to 1 KiB are copied to `copy` (512 entries) in the core's
// local data memory, faster to load from than L1.
template <Format FMT>
inline const std::uint16_t* local_level_table(std::uint16_t* copy) {
    static_assert(is_codebook<FMT>(), "local_level_table: not a codebook format");
    const std::uint16_t* table = level_table<FMT>();
    if constexpr (table_entries<FMT>() * 2 <= 1024) {
        for (std::uint32_t j = 0; j < table_entries<FMT>() / 2; j++) {
            reinterpret_cast<std::uint32_t*>(copy)[j] = reinterpret_cast<const std::uint32_t*>(table)[j];
        }
        table = copy;
    }
    return table;
}

// MXFP4 pages in pairs per record: K tile t's words_shift (dest_srca_mm_page_unpack_codes); an odd tile reads the
// scale row of the page before it.
template <Format FMT>
constexpr std::int32_t paired_words_shift(const std::uint32_t t) {
    return FMT == Format::MXFP4 && t % 2 == 1 ? -static_cast<std::int32_t>(page_layout<FMT>().scale_offset + 64) : 0;
}

// Index words of a codebook page, four grid indices each.
template <Format FMT>
constexpr std::uint32_t index_words() {
    return FMT == Format::IQ3_S || FMT == Format::IQ3_XXS ? 64 : 32;
}

// Byte offset of a codebook page's index words: they follow 128 bytes of sign words, except IQ1_S / IQ1_M's.
template <Format FMT>
constexpr std::uint32_t index_offset() {
    return page_layout<FMT>().code_offset + (FMT == Format::IQ1_S || FMT == Format::IQ1_M ? 0 : 128);
}

// Expanded layouts (DataType::GGUF_*_EXPANDED, gguf_host_packing.cpp) hold per K tile the decoder's page up to the
// index words (index_offset bytes: the level codes, then any sign words) and the decoder's bytes after the index words,
// expanded_tail of them: none for IQ3_S, whose high index bits and scales are in its level codes.
template <Format FMT>
constexpr std::uint32_t expanded_tail() {
    constexpr PageLayout L = page_layout<FMT>();
    return FMT == Format::IQ3_S ? 0 : L.code_offset + L.code_bytes - index_offset<FMT>() - 4 * index_words<FMT>();
}

// Whether lookup_grid can split FMT's index words into `parts` equal shares of whole 8-word batches.
template <Format FMT>
constexpr bool lookup_split_ok(const std::uint32_t parts) {
    return parts > 0 && (index_words<FMT>() / 8) % parts == 0;
}

// Codebook formats: a page's grid indices to level codes through level_table, two per output word, for share `part`
// of `parts` of the index words (lookup_split_ok).
template <Format FMT, typename Out>
inline __attribute__((always_inline)) void lookup_grid(
    const std::uint32_t* indices,
    const std::uint16_t* table,
    Out* levels,
    const std::uint32_t part = 0,
    const std::uint32_t parts = 1) {
    static_assert(is_codebook<FMT>(), "lookup_grid: not a codebook format");
    constexpr std::uint32_t batches = index_words<FMT>() / 8;
#if defined(KERNEL_BUILD)
    ASSERT(lookup_split_ok<FMT>(parts) && part < parts);
#endif
    const std::uint32_t m_begin = 8 * (part * batches / parts);
    const std::uint32_t m_end = 8 * ((part + 1) * batches / parts);
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
        constexpr std::uint32_t hi_bits = xs ? 1 : w_bits;
        constexpr std::uint32_t hi_mask = ((1u << hi_bits) - 1) * 0x01000100u;
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
                std::uint32_t h_ac = h;
                std::uint32_t h_bd = h >> w_bits;
                if constexpr (xs) {
                    h_ac = h << 1;
                    h_bd = h >> 7;
                } else if (b % 2) {
                    h_ac = h << 8;
                    h_bd = h << (8 - w_bits);
                }
                const std::uint32_t ac = (x[b] & lo) | (h_ac & hi_mask);
                const std::uint32_t bd = ((x[b] >> 8) & lo) | (h_bd & hi_mask);
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

#if defined(__riscv)
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
#endif

}  // namespace experimental::gguf
