// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// GGUF blocks <-> the pages of a GGUF tensor. Both directions only move bits: a page holds the GGUF bytes of its 32
// columns and 32 * group_tiles rows, regrouped so that each lane of the Blackhole decoder finds its fields in one
// place (tt-llk ckernel_gguf_page.h and ckernel_sfpu_gguf_decode*.h describe the rows). SFPU output o of lane (r, c)
// is element (out_k(o, r), out_n(o, c)) of a 32x32 [K][N] tile; scale word w covers columns lane_col(w) + b, b = 0, 1.

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <numeric>
#include <vector>

#include <tt_stl/assert.hpp>
#include <tt-metalium/tensor/tensor_apis.hpp>

namespace tt::tt_metal {
namespace {

static_assert(std::endian::native == std::endian::little, "GGUF fields and the pages are little-endian words");

using enum DataType;

uint16_t rd16(const uint8_t* p) {
    uint16_t v;
    std::memcpy(&v, p, 2);
    return v;
}

uint32_t rd32(const uint8_t* p) {
    uint32_t v;
    std::memcpy(&v, p, 4);
    return v;
}

uint8_t* put(uint8_t* out, const uint32_t* words, int n) {
    std::memcpy(out, words, 4 * size_t(n));
    return out + 4 * size_t(n);
}

// The GGUF rows of one 32-column N tile: block(n, b) is column n's block b.
struct TileRows {
    const uint8_t* base;
    size_t row_bytes;
    uint32_t block_bytes;

    const uint8_t* block(int n, int64_t b) const { return base + size_t(n) * row_bytes + size_t(b) * block_bytes; }
};

int lane_col(int w) { return 16 * (w % 2) + 2 * (w / 2); }

// Byte k of lane column c's index word is column idx_col(c, k).
int idx_col(int c, int k) { return 16 * (k / 2) + 2 * c + k % 2; }

// Half m % 2 of high-bits word m / 2 carries column nib_col(m)'s fields.
int nib_col(int m) { return 16 * ((m % 4) / 2) + 2 * (m / 4) + m % 2; }

int out_k(int o, int r) { return 16 * (o / 16) + 4 * ((o / 2) % 4) + r; }

int out_n(int o, int c) { return 16 * ((o / 8) % 2) + o % 2 + 2 * c; }

// Word of lane (r, c) in word-vector wv of a codes part.
int lane_word(int wv, int r, int c) { return (4 * (wv / 2) + r) * 16 + 2 * c + wv % 2; }

// A scale row: word w = the 16-bit field at byte off of block b of column lane_col(w), then of the next column.
void pair16(const TileRows& t, int64_t b, int off, uint32_t* row) {
    for (int w = 0; w < 16; ++w) {
        row[w] = rd16(t.block(lane_col(w), b) + off) | uint32_t(rd16(t.block(lane_col(w) + 1, b) + off)) << 16;
    }
}

// A scale row of byte fields: word w = (b0, o0) of column lane_col(w), of the next column, then (b1, o1).
void quad8(const TileRows& t, int64_t b0, int o0, int64_t b1, int o1, uint32_t* row) {
    for (int w = 0; w < 16; ++w) {
        const int n = lane_col(w);
        row[w] = uint32_t(t.block(n, b0)[o0]) | uint32_t(t.block(n + 1, b0)[o0]) << 8 |
                 uint32_t(t.block(n, b1)[o1]) << 16 | uint32_t(t.block(n + 1, b1)[o1]) << 24;
    }
}

// Word n = column n's 32-bit field at byte off of block b.
void col_words(const TileRows& t, int64_t b, int off, uint8_t* out) {
    for (int n = 0; n < 32; ++n) {
        std::memcpy(out + 4 * n, t.block(n, b) + off, 4);
    }
}

// IQ2_XXS / IQ3_XXS sign word: byte l = the 7-bit sign index l (aux bits 7 l ..), the scale bit l (aux bit 28 + l) in
// bit 7.
uint32_t sign_word(uint32_t aux) {
    uint32_t sw = 0;
    for (int l = 0; l < 4; ++l) {
        sw |= ((aux >> 7 * l & 0x7F) | (aux >> (28 + l) & 1) << 7) << 8 * l;
    }
    return sw;
}

// Index words m = 8 l + c: byte k = index byte off + l of column idx_col(c, k), block b.
void idx_words(const TileRows& t, int64_t b, int off, uint8_t* out) {
    for (int l = 0; l < 4; ++l) {
        for (int c = 0; c < 8; ++c) {
            for (int k = 0; k < 4; ++k) {
                out[4 * (8 * l + c) + k] = t.block(idx_col(c, k), b)[off + l];
            }
        }
    }
}

// IQ3 index words m = 8 (4a + g) + c: entries g and g + 4 of columns 16a + 2c and 16a + 2c + 1, in byte order
// (col, g), (col + 1, g), (col, g + 4), (col + 1, g + 4).
void idx3_words(const TileRows& t, int64_t b, int off, uint8_t* out) {
    for (int m = 0; m < 64; ++m) {
        const int a = m / 32, g = m / 8 % 4, c = m % 8;
        for (int k = 0; k < 4; ++k) {
            out[4 * m + k] = t.block(16 * a + 2 * c + k % 2, b)[off + g + 4 * (k / 2)];
        }
    }
}

// High index bits hn (32 index words x 4 bytes, w bits each) for the lookup's 16-bit index lanes: word j holds index
// word 2j's bytes at bits 8, 8 + w, 24, 24 + w and word 2j + 1's at 0, w, 16, 16 + w.
void lane_highs(const uint32_t (&hn)[32][4], int w, uint32_t* hi) {
    const int pos[2][4] = {{8, 8 + w, 24, 24 + w}, {0, w, 16, 16 + w}};
    std::memset(hi, 0, 16 * sizeof(uint32_t));
    for (int m = 0; m < 32; ++m) {
        for (int k = 0; k < 4; ++k) {
            hi[m / 2] |= hn[m][k] << pos[m % 2][k];
        }
    }
}

// ----------------------------------------------------------------------------------------------- plane formats

// Element w of a block is (block[byte[w]] >> shift[w]) & (2^bits - 1).
struct CodePlane {
    int bits = 0;
    std::array<uint8_t, 256> byte{};
    std::array<uint8_t, 256> shift{};

    uint32_t at(const uint8_t* block, int w) const { return block[byte[w]] >> shift[w] & ((1u << bits) - 1); }
};

struct CodePlanes {
    CodePlane lo, hi;
};

CodePlanes planes_of(DataType type, int blk) {
    CodePlanes p;
    const bool two = type == GGUF_Q3_K || type == GGUF_Q2_K || type == GGUF_TQ2_0 || type == GGUF_Q2_0;
    p.lo.bits = type == GGUF_Q8_0 ? 8 : type == GGUF_Q1_0 ? 1 : two ? 2 : 4;
    p.hi.bits = type == GGUF_Q6_K                                                                  ? 2
                : type == GGUF_Q5_K || type == GGUF_Q3_K || type == GGUF_Q5_0 || type == GGUF_Q5_1 ? 1
                                                                                                   : 0;
    for (int w = 0; w < blk; ++w) {
        const int h = w / 128, t = w % 128 / 32, e = w % 32, j = w / 32;
        uint8_t& lb = p.lo.byte[w];
        uint8_t& ls = p.lo.shift[w];
        uint8_t& hb = p.hi.byte[w];
        uint8_t& hs = p.hi.shift[w];
        // clang-format off
        switch (type) {
            case GGUF_Q8_0:   lb = 2 + w; break;
            case GGUF_Q4_0:
            case GGUF_IQ4_NL: lb = 2 + w % 16; ls = 4 * (w / 16); break;
            case GGUF_Q4_1:   lb = 4 + w % 16; ls = 4 * (w / 16); break;
            case GGUF_Q5_0:   lb = 6 + w % 16; ls = 4 * (w / 16); hb = 2 + w / 8; hs = w % 8; break;
            case GGUF_Q5_1:   lb = 8 + w % 16; ls = 4 * (w / 16); hb = 4 + w / 8; hs = w % 8; break;
            case GGUF_MXFP4:  lb = 1 + w % 16; ls = 4 * (w / 16); break;
            case GGUF_NVFP4:  lb = 4 + 8 * (w / 16) + w % 8; ls = 4 * (w % 16 / 8); break;
            case GGUF_Q6_K:   lb = 64 * h + w % 64; ls = 4 * (w % 128 / 64); hb = 128 + 32 * h + e; hs = 2 * t; break;
            case GGUF_Q4_K:   lb = 16 + 32 * (j / 2) + e; ls = 4 * (j % 2); break;
            case GGUF_Q5_K:   lb = 48 + 32 * (j / 2) + e; ls = 4 * (j % 2); hb = 16 + e; hs = j; break;
            case GGUF_Q3_K:   lb = 32 + 32 * h + e; ls = 2 * t; hb = e; hs = j; break;
            case GGUF_Q2_K:   lb = 16 + 32 * h + e; ls = 2 * t; break;
            case GGUF_IQ4_XS: lb = 8 + 16 * j + w % 16; ls = 4 * (w % 32 / 16); break;
            case GGUF_TQ2_0:  lb = 32 * h + e; ls = 2 * t; break;
            case GGUF_Q2_0:   lb = 2 + w / 4; ls = 2 * (w % 4); break;
            case GGUF_Q1_0:   lb = 2 + w / 8; ls = w % 8; break;
            default:          break;
        }
        // clang-format on
    }
    return p;
}

// The codes of K tiles page_tiles * part .. as one codes part: code q = lo | hi << lo bits as one B-bit field, tile
// i's B word-vectors from word-vector B i. Stream position k is row o = 16 (k / 8 % 2) + 8 (k / 16) + k % 8. When B
// divides 32, row o is field o % F of word o / F (F = 32 / B fields a word); otherwise position k < F B is field k % F
// of word k / F, and positions F B .. 31 take the words' top two bits: bit s of that string (word 0 bits 30, 31,
// word 1 bits 30, 31, ..) is bit s % B of position F B + s / B.
void pack_codes(const GgufTraits& f, const CodePlanes& p, const TileRows& t, int64_t part, uint32_t* words) {
    std::memset(words, 0, f.code_bytes);
    const int B = p.lo.bits + p.hi.bits, F = 32 / B, clean = 32 % B ? F * B : 32;
    for (int i = 0; i < int(f.page_tiles); ++i) {
        const int64_t kt = f.page_tiles * part + i, b = 32 * kt / f.block_elems;
        const int w0 = int(32 * kt % f.block_elems);
        for (int k = 0; k < 32; ++k) {
            const int o = 16 * (k / 8 % 2) + 8 * (k / 16) + k % 8;
            for (int r = 0; r < 4; ++r) {
                const int w = w0 + out_k(o, r);
                for (int c = 0; c < 8; ++c) {
                    const uint8_t* blk = t.block(out_n(o, c), b);
                    const uint32_t q = p.lo.at(blk, w) | (p.hi.bits ? p.hi.at(blk, w) << p.lo.bits : 0);
                    if (32 % B == 0) {
                        words[lane_word(B * i + o / F, r, c)] |= q << B * (o % F);
                    } else if (k < clean) {
                        words[lane_word(B * i + k / F, r, c)] |= q << B * (k % F);
                    } else {
                        for (int j = 0; j < B; ++j) {
                            const int bit = B * (k - clean) + j;
                            words[lane_word(B * i + bit / 2, r, c)] |= (q >> j & 1) << (clean + bit % 2);
                        }
                    }
                }
            }
        }
    }
}

// Group g of a plane format: its scale rows and its group_tiles / page_tiles codes parts. A group with one codes part
// stores it first, so that one read fills an L1 page in the decoder's order.
uint8_t* group_planes(
    DataType type, const GgufTraits& f, const CodePlanes& p, const TileRows& t, int64_t g, uint8_t* out) {
    uint32_t S[10][16] = {};
    switch (type) {
        case GGUF_Q8_0:
        case GGUF_Q4_0:
        case GGUF_IQ4_NL:
        case GGUF_Q2_0:
        case GGUF_Q1_0: pair16(t, g, 0, S[0]); break;
        case GGUF_TQ2_0: pair16(t, g, 64, S[0]); break;
        case GGUF_Q4_1:
            pair16(t, g, 0, S[0]);  // d
            pair16(t, g, 2, S[1]);  // m
            break;
        case GGUF_Q5_0:
            pair16(t, 2 * g, 0, S[0]);  // K tile 2g's d
            pair16(t, 2 * g + 1, 0, S[1]);
            break;
        case GGUF_Q5_1:
            for (int i = 0; i < 2; ++i) {  // K tile 2g + i's d, m
                pair16(t, 2 * g + i, 0, S[2 * i]);
                pair16(t, 2 * g + i, 2, S[2 * i + 1]);
            }
            break;
        case GGUF_MXFP4: quad8(t, 2 * g, 0, 2 * g + 1, 0, S[0]); break;
        case GGUF_NVFP4: quad8(t, g / 2, 2 * (g % 2), g / 2, 2 * (g % 2) + 1, S[0]); break;  // K tile g's two scales
        case GGUF_Q6_K:
        case GGUF_Q2_K: {  // rows 0..7: the 16-element scales 2i, 2i + 1; then d (, dmin)
            const int sc = type == GGUF_Q6_K ? 192 : 0;
            for (int i = 0; i < 8; ++i) {
                quad8(t, g, sc + 2 * i, g, sc + 2 * i + 1, S[i]);
            }
            pair16(t, g, type == GGUF_Q6_K ? 208 : 80, S[8]);
            if (type == GGUF_Q2_K) {
                pair16(t, g, 82, S[9]);
            }
            break;
        }
        case GGUF_Q4_K:
        case GGUF_Q5_K:
        case GGUF_Q3_K: {  // rows 0..2: the 12 scale bytes of column lane_col(w), rows 3..5: of the next; d (, dmin)
            const int sc = type == GGUF_Q3_K ? 96 : 4;
            for (int q = 0; q < 6; ++q) {
                for (int w = 0; w < 16; ++w) {
                    S[q][w] = rd32(t.block(lane_col(w) + q / 3, g) + sc + 4 * (q % 3));
                }
            }
            pair16(t, g, type == GGUF_Q3_K ? 108 : 0, S[6]);
            if (type != GGUF_Q3_K) {
                pair16(t, g, 2, S[7]);
            }
            break;
        }
        case GGUF_IQ4_XS:
            // rows 0..2 as one 96-bit field per word: the 6-bit scale j of column lane_col(w) + b at bit 48 b + 6 j; d
            for (int w = 0; w < 16; ++w) {
                uint64_t ls[2] = {};
                for (int b = 0; b < 2; ++b) {
                    const uint8_t* blk = t.block(lane_col(w) + b, g);
                    for (int j = 0; j < 8; ++j) {
                        const uint32_t v = (blk[4 + j / 2] >> 4 * (j % 2) & 15) | (rd16(blk + 2) >> 2 * j & 3) << 4;
                        ls[b] |= uint64_t(v) << 6 * j;
                    }
                }
                S[0][w] = uint32_t(ls[0]);
                S[1][w] = uint32_t(ls[0] >> 32 | ls[1] << 16);
                S[2][w] = uint32_t(ls[1] >> 16);
            }
            pair16(t, g, 0, S[3]);
            break;
        default: TT_THROW("{} is not a plane format", type);
    }
    const int parts = int(f.group_tiles / f.page_tiles);
    std::array<uint32_t, 320> words{};  // the largest codes part, Q5_K's
    if (parts > 1) {
        out = put(out, S[0], 16 * int(f.scale_rows));
    }
    for (int i = 0; i < parts; ++i) {
        pack_codes(f, p, t, parts * g + i, words.data());
        out = put(out, words.data(), int(f.code_bytes / 4));
    }
    if (parts == 1) {
        out = put(out, S[0], 16 * int(f.scale_rows));
    }
    return out;
}

// TQ1_0: one codes part of the base-3 bytes for the group's eight K tiles, then the scale rows.
// Codes, byte j of lane (r, c): word-vector 4a + 2h + b (K tiles 0..4) holds qs[16h + 4j + r] of column 16a + 2c + b,
// word-vector 8 + 2a + b (K tiles 5..7) qs[32 + 4j + r]. Scale rows 0 and 1: word w = qh of column lane_col(w), of the
// next column; row 2 = d.
uint8_t* group_tq1_0(const TileRows& t, int64_t g, uint8_t* out) {
    std::array<uint32_t, 384> words{};
    for (int v = 0; v < 12; ++v) {
        const int a = v < 8 ? v / 4 : (v - 8) / 2, b = v % 2, first = v < 8 ? 16 * (v / 2 % 2) : 32;
        for (int r = 0; r < 4; ++r) {
            for (int c = 0; c < 8; ++c) {
                for (int j = 0; j < 4; ++j) {
                    words[lane_word(v, r, c)] |= uint32_t(t.block(16 * a + 2 * c + b, g)[first + 4 * j + r]) << 8 * j;
                }
            }
        }
    }
    out = put(out, words.data(), 384);
    uint32_t S[3][16];
    for (int w = 0; w < 16; ++w) {
        S[0][w] = rd32(t.block(lane_col(w), g) + 48);
        S[1][w] = rd32(t.block(lane_col(w) + 1, g) + 48);
    }
    pair16(t, g, 52, S[2]);
    return put(out, S[0], 48);
}

// ------------------------------------------------------------------------------------------- codebook formats
// Each: d (one scale row), then per K tile s (sub-block s of the 256-element block) the part the reader's grid lookup
// and the decoder read.

// IQ2_XXS, per K tile: column n's sign word (128 B), index words (128 B).
uint8_t* group_iq2_xxs(const TileRows& t, int64_t g, uint8_t* out) {
    uint32_t d[16];
    pair16(t, g, 0, d);
    out = put(out, d, 16);
    for (int s = 0; s < 8; ++s, out += 256) {
        for (int n = 0; n < 32; ++n) {
            const uint32_t sw = sign_word(rd32(t.block(n, g) + 6 + 8 * s));
            std::memcpy(out + 4 * n, &sw, 4);
        }
        idx_words(t, g, 2 + 8 * s, out + 128);
    }
    return out;
}

// IQ1_S, per K tile: index words (128 B), high bits (64 B; in half b of word j bits 6, 7, 14 the scale and bit 15 the
// delta sign of column nib_col(2j + b)).
uint8_t* group_iq1_s(const TileRows& t, int64_t g, uint8_t* out) {
    uint32_t d[16];
    pair16(t, g, 0, d);
    out = put(out, d, 16);
    for (int s = 0; s < 8; ++s) {
        idx_words(t, g, 2 + 4 * s, out);
        uint32_t hn[32][4], hi[16];
        for (int m = 0; m < 32; ++m) {
            for (int k = 0; k < 4; ++k) {
                hn[m][k] = rd16(t.block(idx_col(m % 8, k), g) + 34 + 2 * s) >> 3 * (m / 8) & 7;
            }
        }
        lane_highs(hn, 3, hi);
        for (int m = 0; m < 32; ++m) {
            const uint32_t nib = rd16(t.block(nib_col(m), g) + 34 + 2 * s) >> 12;
            const int o = 16 * (m % 2);
            hi[m / 2] |= (nib & 3) << (o + 6) | (nib >> 2 & 1) << (o + 14) | (nib >> 3) << (o + 15);
        }
        out = put(out + 128, hi, 16);
    }
    return out;
}

// IQ2_S, per K tile: column n's sign bits (128 B), index low bytes (128 B), high bits (64 B; in half b of word j bits
// 4..7 / 12..15 the scales of column nib_col(2j + b)).
uint8_t* group_iq2_s(const TileRows& t, int64_t g, uint8_t* out) {
    uint32_t d[16];
    pair16(t, g, 0, d);
    out = put(out, d, 16);
    for (int s = 0; s < 8; ++s) {
        col_words(t, g, 34 + 4 * s, out);
        idx_words(t, g, 2 + 4 * s, out + 128);
        uint32_t hn[32][4], hi[16];
        for (int m = 0; m < 32; ++m) {
            for (int k = 0; k < 4; ++k) {
                hn[m][k] = t.block(idx_col(m % 8, k), g)[66 + s] >> 2 * (m / 8) & 3;
            }
        }
        lane_highs(hn, 2, hi);
        for (int m = 0; m < 32; ++m) {
            const uint32_t s8 = t.block(nib_col(m), g)[74 + s];
            const int o = 16 * (m % 2);
            hi[m / 2] |= (s8 & 15) << (o + 4) | (s8 >> 4) << (o + 12);
        }
        out = put(out + 256, hi, 16);
    }
    return out;
}

// IQ3_S, per K tile: column n's sign bits (128 B), IQ3 index words (256 B), 48 B: word i < 8 holds the high index bits
// of index words 8i + j, bytes 0..3 at bits j, 8 + j, 16 + j, 24 + j; byte 8a + c of words 8..11 = the scales of
// columns 16a + 2c (low nibble) and 16a + 2c + 1.
uint8_t* group_iq3_s(const TileRows& t, int64_t g, uint8_t* out) {
    uint32_t d[16];
    pair16(t, g, 0, d);
    out = put(out, d, 16);
    for (int s = 0; s < 8; ++s) {
        col_words(t, g, 74 + 4 * s, out);
        idx3_words(t, g, 2 + 8 * s, out + 128);
        uint32_t W[12] = {};
        for (int m = 0; m < 64; ++m) {
            const int a = m / 32, gg = m / 8 % 4, c = m % 8;
            for (int k = 0; k < 4; ++k) {
                const uint32_t qh = t.block(16 * a + 2 * c + k % 2, g)[66 + s];
                W[m / 8] |= (qh >> (gg + 4 * (k / 2)) & 1) << (8 * k + m % 8);
            }
        }
        uint8_t scb[16];
        const int sh = 4 * (s % 2);  // sub-block s's scale nibble
        for (int i = 0; i < 16; ++i) {
            const int n = 16 * (i / 8) + 2 * (i % 8);
            scb[i] =
                uint8_t((t.block(n, g)[106 + s / 2] >> sh & 15) | (t.block(n + 1, g)[106 + s / 2] >> sh & 15) << 4);
        }
        std::memcpy(W + 8, scb, sizeof(scb));
        out = put(out + 384, W, 12);
    }
    return out;
}

// IQ3_XXS, per K tile: column n's sign word as IQ2_XXS's (128 B), IQ3 index words (256 B).
uint8_t* group_iq3_xxs(const TileRows& t, int64_t g, uint8_t* out) {
    uint32_t d[16];
    pair16(t, g, 0, d);
    out = put(out, d, 16);
    for (int s = 0; s < 8; ++s, out += 384) {
        for (int n = 0; n < 32; ++n) {
            const uint32_t sw = sign_word(rd32(t.block(n, g) + 66 + 4 * s));
            std::memcpy(out + 4 * n, &sw, 4);
        }
        idx3_words(t, g, 2 + 8 * s, out + 128);
    }
    return out;
}

// IQ2_XS: 320 B of scale rows (row u < 4, word w: byte 2t + b = the scale byte of column lane_col(w) + b for K tile
// 2u + t; row 4: d); per K tile: 128 B, word 16 (l / 2) + 2c + l % 2, byte k: entry l of column idx_col(c, k), its
// 7-bit sign index and grid index bit 8 in bit 7; 128 B, word 8 l + c: the same entries' grid index bits 0..7.
uint8_t* group_iq2_xs(const TileRows& t, int64_t g, uint8_t* out) {
    uint32_t S[5][16] = {};
    for (int u = 0; u < 4; ++u) {
        for (int j = 0; j < 4; ++j) {
            for (int w = 0; w < 16; ++w) {
                S[u][w] |= uint32_t(t.block(lane_col(w) + j % 2, g)[66 + 2 * u + j / 2]) << 8 * j;
            }
        }
    }
    pair16(t, g, 0, S[4]);
    out = put(out, S[0], 80);
    for (int s = 0; s < 8; ++s, out += 256) {
        for (int m = 0; m < 32; ++m) {
            const int l = m / 8, c = m % 8, sign = 16 * (l / 2) + l % 2 + 2 * c;
            for (int k = 0; k < 4; ++k) {
                const uint16_t e = rd16(t.block(idx_col(c, k), g) + 2 + 8 * s + 2 * l);
                out[4 * sign + k] = uint8_t(e >> 9 | (e >> 8 & 1) << 7);
                out[128 + 4 * m + k] = uint8_t(e);
            }
        }
    }
    return out;
}

// IQ1_M: 256 B of scale rows (rows 0..2, word w: the 3-bit scale of column lane_col(w) + b, elements 16h .. 16h + 15
// of K tile s as field f = 4s + 2b + h at bit 3 (f % 10) of row f / 10, f < 30; fields 30, 31: bits 0..1 at bits
// 30..31 of rows 0 and 1, bit 2 at bit f of row 2; row 3: d, the fp16 in the scale words' top nibbles); per K tile:
// index words (128 B), high bits as IQ1_S's with column nib_col(2j + b)'s four delta signs at bits 6, 7, 14, 15.
uint8_t* group_iq1_m(const TileRows& t, int64_t g, uint8_t* out) {
    uint32_t S[4][16] = {};
    for (int w = 0; w < 16; ++w) {
        for (int f = 0; f < 32; ++f) {
            const int j = 2 * (f / 4) + f % 2;  // scale j of the block: sub-block j / 2, half j % 2
            const uint32_t v = rd16(t.block(lane_col(w) + f / 2 % 2, g) + 48 + 2 * (j / 4)) >> 3 * (j % 4) & 7;
            if (f < 30) {
                S[f / 10][w] |= v << 3 * (f % 10);
            } else {
                S[f - 30][w] |= (v & 3) << 30;
                S[2][w] |= (v >> 2) << f;
            }
        }
        for (int b = 0; b < 2; ++b) {
            const uint8_t* sc = t.block(lane_col(w) + b, g) + 48;
            const uint32_t d =
                rd16(sc) >> 12 | (rd16(sc + 2) >> 8 & 0xF0) | (rd16(sc + 4) >> 4 & 0xF00) | (rd16(sc + 6) & 0xF000);
            S[3][w] |= d << 16 * b;
        }
    }
    out = put(out, S[0], 64);
    for (int s = 0; s < 8; ++s) {
        idx_words(t, g, 4 * s, out);
        // qh nibble l of sub-block s: 3 high index bits, the delta sign in bit 3
        const auto qh = [&t, g, s](int n, int l) {
            return uint32_t(t.block(n, g)[32 + 2 * s + l / 2] >> 4 * (l % 2) & 15);
        };
        uint32_t hn[32][4], hi[16];
        for (int m = 0; m < 32; ++m) {
            for (int k = 0; k < 4; ++k) {
                hn[m][k] = qh(idx_col(m % 8, k), m / 8) & 7;
            }
        }
        lane_highs(hn, 3, hi);
        const int pos[4] = {6, 7, 14, 15};
        for (int m = 0; m < 32; ++m) {
            for (int e = 0; e < 4; ++e) {
                hi[m / 2] |= (qh(nib_col(m), e) >> 3) << (16 * (m % 2) + pos[e]);
            }
        }
        out = put(out + 128, hi, 16);
    }
    return out;
}

// The pages of an [K, N] tensor from its GGUF blocks (N rows of K / block_elems blocks), page (g, nt) at page index
// g * N / 32 + nt: K group g of N tile nt.
void pack_pages(DataType type, const uint8_t* blocks, int64_t K, int64_t N, uint8_t* pages) {
    const GgufTraits& f = gguf_traits(type);
    const CodePlanes p = planes_of(type, int(f.block_elems));
    const size_t row_bytes = size_t(K / f.block_elems) * f.block_bytes;
    const int64_t groups = K / f.page_height(), n_tiles = N / 32;
    for (int64_t nt = 0; nt < n_tiles; ++nt) {
        const TileRows t{blocks + 32 * nt * row_bytes, row_bytes, f.block_bytes};
        for (int64_t g = 0; g < groups; ++g) {
            uint8_t* page = pages + (g * n_tiles + nt) * f.page_bytes();
            uint8_t* end = nullptr;
            switch (type) {
                case GGUF_IQ2_XXS: end = group_iq2_xxs(t, g, page); break;
                case GGUF_IQ1_S: end = group_iq1_s(t, g, page); break;
                case GGUF_IQ2_S: end = group_iq2_s(t, g, page); break;
                case GGUF_IQ3_S: end = group_iq3_s(t, g, page); break;
                case GGUF_IQ3_XXS: end = group_iq3_xxs(t, g, page); break;
                case GGUF_IQ2_XS: end = group_iq2_xs(t, g, page); break;
                case GGUF_IQ1_M: end = group_iq1_m(t, g, page); break;
                case GGUF_TQ1_0: end = group_tq1_0(t, g, page); break;
                default: end = group_planes(type, f, p, t, g, page); break;
            }
            TT_ASSERT(end == page + f.page_bytes(), "{}: a page must hold exactly its GGUF bytes", type);
        }
    }
}

// pack_pages only moves bits, so to_gguf runs it backwards with a table: for one unit of 32 columns by
// lcm(32 * group_tiles, block_elems) rows of K, source[i] = the GGUF bit that page bit i (of the unit's pages, in K
// order) holds. Read off once per format by packing index-coded probes: in probe p, GGUF bit j is bit p of j + 1.
struct BitTable {
    int64_t unit_k = 0;
    std::vector<uint32_t> source;
};

const BitTable& bit_table(DataType type) {
    static std::array<BitTable, 25> tables;
    static std::array<std::once_flag, 25> once;
    const int index = static_cast<int>(type) - static_cast<int>(GGUF_Q8_0);
    std::call_once(once[index], [type, &table = tables[index]] {
        const GgufTraits& f = gguf_traits(type);
        table.unit_k = std::lcm(int64_t(f.page_height()), int64_t(f.block_elems));
        const size_t bytes = 32 * size_t(table.unit_k / f.block_elems) * f.block_bytes, bits = 8 * bytes;
        std::vector<uint32_t> code(bits, 0);
        std::vector<uint8_t> probe(bytes), pages(bytes);
        for (int p = 0; (bits + 1) >> p; ++p) {
            std::fill(probe.begin(), probe.end(), 0);
            for (size_t j = 0; j < bits; ++j) {
                probe[j / 8] |= uint8_t(((j + 1) >> p & 1) << (j % 8));
            }
            pack_pages(type, probe.data(), table.unit_k, 32, pages.data());
            for (size_t i = 0; i < bits; ++i) {
                code[i] |= uint32_t(pages[i / 8] >> (i % 8) & 1) << p;
            }
        }
        std::vector<bool> seen(bits, false);
        for (uint32_t& c : code) {
            TT_FATAL(c >= 1 && c <= bits && !seen[c - 1], "{}: the page layout must be a permutation of bits", type);
            seen[c - 1] = true;
            c -= 1;
        }
        table.source = std::move(code);
    });
    return tables[index];
}

void unpack_pages(DataType type, const uint8_t* pages, int64_t K, int64_t N, uint8_t* blocks) {
    const GgufTraits& f = gguf_traits(type);
    const BitTable& table = bit_table(type);
    const size_t row_bytes = size_t(K / f.block_elems) * f.block_bytes;
    const size_t unit_row_bytes = size_t(table.unit_k / f.block_elems) * f.block_bytes;
    const int64_t n_tiles = N / 32, unit_pages = table.unit_k / f.page_height();
    std::fill(blocks, blocks + N * row_bytes, 0);
    for (int64_t nt = 0; nt < n_tiles; ++nt) {
        for (int64_t u = 0; u < K / table.unit_k; ++u) {
            uint8_t* rows = blocks + 32 * nt * row_bytes + u * unit_row_bytes;
            for (size_t i = 0; i < table.source.size(); ++i) {
                const int64_t page = u * unit_pages + int64_t(i / (8 * f.page_bytes()));
                const size_t bit = i % (8 * f.page_bytes());
                const uint8_t v = pages[(page * n_tiles + nt) * f.page_bytes() + bit / 8] >> (bit % 8) & 1;
                const uint32_t j = table.source[i];
                rows[(j / 8) / unit_row_bytes * row_bytes + (j / 8) % unit_row_bytes] |= uint8_t(v << (j % 8));
            }
        }
    }
}

// [K, N] of a GGUF tensor spec, checked against what the pages can hold.
std::pair<int64_t, int64_t> gguf_shape(const TensorSpec& spec) {
    const DataType type = spec.data_type();
    TT_FATAL(is_gguf(type), "Expected a GGUF dtype, got {}", type);
    const auto& shape = spec.logical_shape();
    TT_FATAL(shape.rank() == 2, "A GGUF tensor is [K, N], got shape {}", shape);
    const GgufTraits& f = gguf_traits(type);
    const int64_t K = shape[0], N = shape[1], k_unit = std::lcm(int64_t(f.page_height()), int64_t(f.block_elems));
    TT_FATAL(
        K > 0 && N > 0 && K % k_unit == 0 && N % 32 == 0,
        "{} needs K % {} == 0 and N % 32 == 0, got [K, N] = [{}, {}]",
        type,
        k_unit,
        K,
        N);
    TT_FATAL(
        spec.physical_shape() == Shape2D(K, N),
        "{} pages hold no padding, got physical shape {}",
        type,
        spec.physical_shape());
    return {K, N};
}

}  // namespace

HostTensor from_gguf(ttsl::Span<const std::byte> blocks, const TensorSpec& spec) {
    const auto [K, N] = gguf_shape(spec);
    const GgufTraits& f = gguf_traits(spec.data_type());
    const size_t bytes = size_t(N) * size_t(K / f.block_elems) * f.block_bytes;
    TT_FATAL(
        blocks.size() == bytes,
        "{} [K, N] = [{}, {}] takes {} bytes of blocks, got {}",
        spec.data_type(),
        K,
        N,
        bytes,
        blocks.size());
    std::vector<uint32_t> pages(bytes / sizeof(uint32_t));
    pack_pages(
        spec.data_type(),
        reinterpret_cast<const uint8_t*>(blocks.data()),
        K,
        N,
        reinterpret_cast<uint8_t*>(pages.data()));
    return HostTensor::from_buffer(HostBuffer(std::move(pages)), spec);
}

std::vector<std::byte> to_gguf(const HostTensor& tensor) {
    const auto [K, N] = gguf_shape(tensor.tensor_spec());
    const HostBuffer buffer = host_buffer::get_host_buffer(tensor);
    const auto pages = buffer.view_bytes();
    TT_FATAL(
        pages.size() == tensor.tensor_spec().compute_packed_buffer_size_bytes(), "GGUF tensor buffer size mismatch");
    std::vector<std::byte> blocks(pages.size());
    unpack_pages(
        tensor.dtype(),
        reinterpret_cast<const uint8_t*>(pages.data()),
        K,
        N,
        reinterpret_cast<uint8_t*>(blocks.data()));
    return blocks;
}

}  // namespace tt::tt_metal
