// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// GGUF blocks <-> the pages of a GGUF tensor. Both directions only move bits: a page holds the GGUF bytes of its 32
// columns and 32 * group_tiles rows, regrouped so that each lane of the Blackhole decoder finds its fields in one
// place (tt-llk ckernel_gguf_page.h and ckernel_sfpu_gguf_decode*.h describe the rows). SFPU output o of lane (r, c)
// is element (out_k(o, r), out_n(o, c)) of a 32x32 [K][N] tile; scale word w covers columns lane_col(w) + b, b = 0, 1.
// TQ2_0 is the exception: its codes become block-float tiles that the matmul reads without decoding (group_tq2_0).

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <numeric>
#include <type_traits>
#include <vector>

#include <tt_stl/assert.hpp>
#include <tt-metalium/tensor/tensor_apis.hpp>

#include "tt_metal/hw/inc/experimental/gguf_page.h"

namespace tt::tt_metal {
namespace {

static_assert(std::endian::native == std::endian::little, "GGUF fields and the pages are little-endian words");

using enum DataType;
namespace gg = ::experimental::gguf;

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
    const auto format = static_cast<gg::Format>(gguf_traits(type).format_id);
    p.lo.bits = int(ckernel::gguf::code_lo_bits(format));
    p.hi.bits = int(ckernel::gguf::code_hi_bits(format));
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

bool has_bfp_codes(DataType type) {
    return type == GGUF_MXFP4 || type == GGUF_NVFP4 || type == GGUF_IQ4_NL || type == GGUF_IQ4_XS;
}

// MXFP4, NVFP4, IQ4_NL, IQ4_XS: K tile kt's codes as 4-bit block-float datums of the 32x32 [K][N] tile in face order:
// element e in bits 4 (e % 2) .. of byte e / 2. The decoder unpacks them with a fixed exponent (bfp_codes maps them).
void pack_codes_bfp(const GgufTraits& f, const CodePlanes& p, const TileRows& t, int64_t kt, uint8_t* out) {
    std::memset(out, 0, 512);
    const int64_t b = 32 * kt / f.block_elems;
    const int w0 = int(32 * kt % f.block_elems);
    for (int k = 0; k < 32; ++k) {
        for (int n = 0; n < 32; ++n) {
            const int e = 512 * (k / 16) + 256 * (n / 16) + 16 * (k % 16) + n % 16;
            out[e / 2] |= uint8_t(p.lo.at(t.block(n, b), w0 + k) << 4 * (e % 2));
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
        if (has_bfp_codes(type)) {
            pack_codes_bfp(f, p, t, parts * g + i, reinterpret_cast<uint8_t*>(words.data()));
        } else {
            pack_codes(f, p, t, parts * g + i, words.data());
        }
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

// TQ2_0: the group's eight K tiles as BFP2_b tiles without exponents (256 B each: 32x32 [K][N] in face order, element
// i in bits 2 (i % 4) of byte i / 4, a 2-bit sign / magnitude code), then d of column n at byte 2048 + 2 n. The codes
// are the GGUF codes q with tq2_0_codes applied, so that the matmul reads q - 1 with the exponent forced to 1.0.
uint8_t* group_tq2_0(const TileRows& t, int64_t g, uint8_t* out) {
    std::memset(out, 0, 2048);
    for (int i = 0; i < 8; ++i) {
        for (int r = 0; r < 32; ++r) {
            const int w = 32 * i + r;  // element w of the column's block g
            for (int c = 0; c < 32; ++c) {
                const uint32_t q = t.block(c, g)[32 * (w / 128) + w % 32] >> 2 * (w % 128 / 32) & 3;
                const int e = 512 * (r / 16) + 256 * (c / 16) + 16 * (r % 16) + c % 16;
                out[256 * i + e / 4] |= uint8_t(q << 2 * (e % 4));
            }
        }
    }
    for (int c = 0; c < 32; ++c) {
        std::memcpy(out + 2048 + 2 * c, t.block(c, g) + 64, 2);
    }
    return out + 2112;
}

// TQ2_0 codes q (0, 1, 2 for -1, 0, +1) <-> BFP2 codes (q + 3) % 4: sign in bit 1, magnitude in bit 0, so that 0 is +0.
// q = 3, which TQ2_0 quantization never writes, would become sign 1 with magnitude 0, which reads as -inf: from_gguf
// refuses it. Applied to the 64 code bytes of each block by pack_pages and unpack_pages, around the bit
// permutations. row_blocks: blocks per column of N.
void tq2_0_codes(uint8_t* blocks, size_t bytes, int64_t row_blocks, bool to_device) {
    std::array<uint8_t, 256> map{};
    for (int b = 0; b < 256; ++b) {
        for (int f = 0; f < 4; ++f) {
            map[b] |= uint8_t((((b >> 2 * f) + (to_device ? 3 : 1)) & 3) << 2 * f);
        }
    }
    for (size_t block = 0; block < bytes; block += 66) {
        for (size_t j = 0; j < 64; ++j) {
            const uint8_t b = blocks[block + j];
            if (to_device) {
                const int64_t index = int64_t(block / 66);
                TT_FATAL(
                    (b & b >> 1 & 0x55) == 0,
                    "DataType::GGUF_TQ2_0: block {} (column {}, K block {}) holds code 3; TQ2_0 codes are 0, 1 and 2",
                    index,
                    index / row_blocks,
                    index % row_blocks);
            }
            blocks[block + j] = map[b];
        }
    }
}

// GGUF codes <-> the page's block-float codes (x = +-c / 4 for sign and magnitude c, x8 for sign 1 and magnitude 0), a
// permutation of the 16 codes applied to the code bytes (the last block_elems / 2) of each block by pack_pages and
// unpack_pages, around the bit permutations. MXFP4 / NVFP4: the sign bit stays, E2M1 magnitudes 0, 0.5, 1, 1.5, 2, 3,
// 4, 6 (codes 0..7) become codes 0, 1, 4, 5, 6, 7, 2, 3, so that the decoder's table segments hold them in pairs.
// IQ4_NL / IQ4_XS: codes 0..14 (values in increasing order) go to x = 1.75 .. -1.75, code 15 (the largest value) to
// x8, where the decoder's cubic and its saturation give the values (ckernel_sfpu_gguf_decode_iq4.h).
void bfp_codes(DataType type, const GgufTraits& f, uint8_t* blocks, size_t bytes, bool to_device) {
    std::array<uint8_t, 16> to_page{};
    for (int q = 0; q < 16; ++q) {
        if (type == GGUF_IQ4_NL || type == GGUF_IQ4_XS) {
            to_page[q] = uint8_t(q <= 7 ? 7 - q : q <= 14 ? q + 1 : 8);
        } else {
            constexpr uint8_t magnitude[8] = {0, 1, 4, 5, 6, 7, 2, 3};
            to_page[q] = uint8_t((q & 8) | magnitude[q & 7]);
        }
    }
    std::array<uint8_t, 16> code{};
    for (int q = 0; q < 16; ++q) {
        code[to_device ? q : to_page[q]] = to_device ? to_page[q] : uint8_t(q);
    }
    std::array<uint8_t, 256> map{};
    for (int v = 0; v < 256; ++v) {
        map[v] = uint8_t(code[v & 15] | code[v >> 4] << 4);
    }
    for (size_t block = 0; block < bytes; block += f.block_bytes) {
        for (size_t j = f.block_bytes - f.block_elems / 2; j < f.block_bytes; ++j) {
            blocks[block + j] = map[blocks[block + j]];
        }
    }
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
void pack_page_bits(DataType type, const uint8_t* blocks, int64_t K, int64_t N, uint8_t* pages) {
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
                case GGUF_TQ2_0: end = group_tq2_0(t, g, page); break;
                default: end = group_planes(type, f, p, t, g, page); break;
            }
            TT_ASSERT(end == page + f.page_bytes(), "{}: a page must hold exactly its GGUF bytes", type);
        }
    }
}

// pack_page_bits only moves bits, so unpack_page_bits runs it backwards with a table: for one unit of 32 columns by
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
            pack_page_bits(type, probe.data(), table.unit_k, 32, pages.data());
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

void unpack_page_bits(DataType type, const uint8_t* pages, int64_t K, int64_t N, uint8_t* blocks) {
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

// GGUF blocks <-> pages. TQ2_0 and the block-float code formats map their codes first (tq2_0_codes, which refuses code
// 3; bfp_codes); the rest is a bit permutation.
void pack_pages(DataType type, const uint8_t* blocks, int64_t K, int64_t N, uint8_t* pages) {
    if (type != GGUF_TQ2_0 && !has_bfp_codes(type)) {
        pack_page_bits(type, blocks, K, N, pages);
        return;
    }
    const GgufTraits& f = gguf_traits(type);
    const size_t bytes = size_t(N) * size_t(K / f.block_elems) * f.block_bytes;
    std::vector<uint8_t> codes(blocks, blocks + bytes);
    if (type == GGUF_TQ2_0) {
        tq2_0_codes(codes.data(), bytes, K / 256, true);
    } else {
        bfp_codes(type, f, codes.data(), bytes, true);
    }
    pack_page_bits(type, codes.data(), K, N, pages);
}

void unpack_pages(DataType type, const uint8_t* pages, int64_t K, int64_t N, uint8_t* blocks) {
    unpack_page_bits(type, pages, K, N, blocks);
    if (type == GGUF_TQ2_0) {
        tq2_0_codes(blocks, size_t(N) * size_t(K / 256) * 66, K / 256, false);
    } else if (has_bfp_codes(type)) {
        const GgufTraits& f = gguf_traits(type);
        bfp_codes(type, f, blocks, size_t(N) * size_t(K / f.block_elems) * f.block_bytes, false);
    }
}

// ------------------------------------------------------------------------------------------- expanded layouts
// An expanded codebook page is its GGUF page with each K tile's grid indices replaced by the level codes that
// experimental::gguf::lookup_grid (gguf_page.h) makes of them: per K tile, the decoder's L1 page up to the index words
// (the level codes, then the sign words for all but IQ1_S / IQ1_M) and the decoder's bits after them (IQ1_S / IQ1_M /
// IQ2_S: the 64-byte row of scales and high bits). Grid entries are distinct, so the codes give back the indices. An
// expanded TQ1_0 tensor holds TQ2_0's page of the same trits.

struct ExpandedTile {
    const std::uint16_t* table;  // the level table
    uint32_t entries;            // its grid entries
    uint32_t index_offset;       // the index words in a codes part
    uint32_t index_bytes;
    uint32_t head;  // bytes of the decoder's L1 page before the index words
    uint32_t tail;  // bytes of the decoder's bits after them
    // The level codes of a codes part (lookup_grid), code_offset bytes.
    void (*level_codes)(const uint8_t* part, uint8_t* levels);
};

ExpandedTile expanded_tile(DataType source) {
    const auto tile = [](auto format) {
        using F = decltype(format);
        constexpr ckernel::gguf::PageLayout L = gg::page_layout<F::value>();
        return ExpandedTile{
            gg::level_table<F::value>(),
            gg::table_entries<F::value>(),
            gg::index_offset<F::value>() - L.code_offset,
            4 * gg::index_words<F::value>(),
            gg::index_offset<F::value>(),
            gg::expanded_tail<F::value>(),
            [](const uint8_t* part, uint8_t* levels) {
                constexpr ckernel::gguf::PageLayout layout = gg::page_layout<F::value>();
                uint32_t words[108] = {};  // the largest codes part, IQ3_S's
                std::memcpy(words, part, layout.code_bytes);
                uint32_t codes[128];
                gg::lookup_grid<F::value>(
                    words + (gg::index_offset<F::value>() - layout.code_offset) / 4,
                    gg::level_table<F::value>(),
                    codes);
                std::memcpy(levels, codes, layout.code_offset);
            }};
    };
    switch (source) {
        case GGUF_IQ2_XXS: return tile(std::integral_constant<gg::Format, gg::Format::IQ2_XXS>{});
        case GGUF_IQ1_S: return tile(std::integral_constant<gg::Format, gg::Format::IQ1_S>{});
        case GGUF_IQ2_S: return tile(std::integral_constant<gg::Format, gg::Format::IQ2_S>{});
        case GGUF_IQ3_S: return tile(std::integral_constant<gg::Format, gg::Format::IQ3_S>{});
        case GGUF_IQ3_XXS: return tile(std::integral_constant<gg::Format, gg::Format::IQ3_XXS>{});
        case GGUF_IQ2_XS: return tile(std::integral_constant<gg::Format, gg::Format::IQ2_XS>{});
        case GGUF_IQ1_M: return tile(std::integral_constant<gg::Format, gg::Format::IQ1_M>{});
        default: TT_THROW("{} has no expanded codebook layout", source);
    }
}

// Code -> grid index of a level table, -1 for codes of no entry.
const std::vector<int32_t>& grid_of_code(const std::uint16_t* table, uint32_t entries) {
    static std::mutex mutex;
    static std::vector<std::pair<const std::uint16_t*, std::vector<int32_t>>> inverses;
    const std::lock_guard<std::mutex> lock(mutex);
    for (const auto& [t, inverse] : inverses) {
        if (t == table) {
            return inverse;
        }
    }
    std::vector<int32_t> inverse(1 << 16, -1);
    for (uint32_t i = 0; i < entries; ++i) {
        TT_FATAL(inverse[table[i]] < 0, "GGUF level table: grid entries {} and {} share a code", inverse[table[i]], i);
        inverse[table[i]] = int32_t(i);
    }
    inverses.emplace_back(table, std::move(inverse));
    return inverses.back().second;
}

// GGUF pages of `source` <-> its expanded pages; `pages` pages.
void expand_pages(DataType source, const uint8_t* in, size_t pages, uint8_t* out) {
    const GgufTraits& f = gguf_traits(source);
    const ExpandedTile x = expanded_tile(source);
    const uint32_t scale_bytes = 64 * f.scale_rows;
    std::vector<uint8_t> l1(f.code_offset + f.code_bytes);
    for (size_t p = 0; p < pages; ++p) {
        std::memcpy(out, in, scale_bytes);
        in += scale_bytes;
        out += scale_bytes;
        for (uint32_t t = 0; t < f.group_tiles; ++t, in += f.code_bytes) {
            x.level_codes(in, l1.data());
            std::memcpy(l1.data() + f.code_offset, in, f.code_bytes);
            out = std::copy_n(l1.data(), x.head, out);
            out = std::copy_n(l1.data() + x.head + x.index_bytes, x.tail, out);
        }
    }
}

void compact_pages(DataType source, const uint8_t* in, size_t pages, uint8_t* out) {
    const GgufTraits& f = gguf_traits(source);
    const ExpandedTile x = expanded_tile(source);
    const std::vector<int32_t>& grid = grid_of_code(x.table, x.entries);
    const uint32_t scale_bytes = 64 * f.scale_rows;
    std::vector<uint8_t> l1(f.code_offset + f.code_bytes);
    std::vector<uint8_t> check(f.code_offset);
    for (size_t p = 0; p < pages; ++p) {
        std::memcpy(out, in, scale_bytes);
        in += scale_bytes;
        out += scale_bytes;
        for (uint32_t t = 0; t < f.group_tiles; ++t, out += f.code_bytes) {
            std::fill(l1.begin(), l1.end(), 0);
            std::memcpy(l1.data(), in, x.head);
            std::memcpy(l1.data() + x.head + x.index_bytes, in + x.head, x.tail);
            uint8_t* part = l1.data() + f.code_offset;
            uint8_t* idx = part + x.index_offset;
            for (uint32_t m = 0; m < x.index_bytes / 4; ++m) {
                for (uint32_t k = 0; k < 4; ++k) {
                    const uint32_t code = rd16(l1.data() + 2 * (4 * m + k));
                    const int32_t g = grid[source == GGUF_IQ3_S ? code & 0x0FFF : code];
                    TT_FATAL(g >= 0, "{}: level code {:#06x} is no grid entry's", source, code);
                    idx[4 * m + k] = uint8_t(g);
                    if (source == GGUF_IQ3_S) {
                        uint8_t* high = idx + 256;  // high index bits, then the scales
                        high[4 * (m / 8) + k] |= uint8_t((g >> 8 & 1) << (m % 8));
                        if (m / 8 % 4 == 0 && k < 2) {
                            high[32 + 8 * (m / 32) + m % 8] |= uint8_t((code >> 12) << (4 * k));
                        }
                    }
                }
            }
            // The bits kept beside the codes (high index bits) must agree with them.
            x.level_codes(part, check.data());
            TT_FATAL(
                std::memcmp(check.data(), l1.data(), f.code_offset) == 0,
                "{}: an expanded page's level codes disagree with its high index bits",
                source);
            std::memcpy(out, part, f.code_bytes);
            in += x.head + x.tail;
        }
    }
}

// TQ1_0 block (48 bytes of five trits, 4 bytes of four, fp16 d) <-> TQ2_0 block (64 bytes of four 2-bit codes, fp16
// d): element e's trit q (value (q - 1) d) is trit n of a base-3 byte, or the 2-bit field at bit 2 (e % 128 / 32) of
// byte 32 (e / 128) + e % 32.
struct TritPlace {
    int byte, n;
};

TritPlace tq1_place(int e) {
    if (e < 160) {
        return {e % 32, e / 32};
    }
    if (e < 240) {
        return {32 + (e - 160) % 16, (e - 160) / 16};
    }
    return {48 + (e - 240) % 4, (e - 240) / 4};
}

uint32_t tq1_trit(uint8_t byte, int n) {
    static constexpr uint8_t pow3[5] = {1, 3, 9, 27, 81};
    return uint32_t(uint8_t(byte * pow3[n])) * 3 >> 8;
}

void tq1_to_tq2(const uint8_t* in, uint8_t* out) {
    std::memset(out, 0, 64);
    for (int e = 0; e < 256; ++e) {
        const TritPlace t = tq1_place(e);
        out[32 * (e / 128) + e % 32] |= uint8_t(tq1_trit(in[t.byte], t.n) << 2 * (e % 128 / 32));
    }
    std::memcpy(out + 64, in + 52, 2);
}

// TQ1_0 quantization's bytes: trits t0 .. t4 (t4 = 0 for the last 4 bytes) as ceil(256 (81 t0 + .. + t4) / 243).
bool tq2_to_tq1(const uint8_t* in, uint8_t* out) {
    uint32_t value[52] = {};
    for (int e = 0; e < 256; ++e) {
        const uint32_t q = in[32 * (e / 128) + e % 32] >> 2 * (e % 128 / 32) & 3;
        if (q > 2) {
            return false;
        }
        const TritPlace t = tq1_place(e);
        uint32_t weight = 1;
        for (int i = t.n; i < 4; ++i) {
            weight *= 3;
        }
        value[t.byte] += q * weight;
    }
    for (int b = 0; b < 52; ++b) {
        out[b] = uint8_t((value[b] * 256 + 242) / 243);
    }
    std::memcpy(out + 52, in + 64, 2);
    return true;
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
    const DataType type = spec.data_type(), source = gguf_source_dtype(type);
    const GgufTraits& f = gguf_traits(source);
    const size_t bytes = size_t(N) * size_t(K / f.block_elems) * f.block_bytes;
    TT_FATAL(
        blocks.size() == bytes,
        "{} [K, N] = [{}, {}] takes {} bytes of blocks ({}), got {}",
        type,
        K,
        N,
        bytes,
        source,
        blocks.size());
    const auto* in = reinterpret_cast<const uint8_t*>(blocks.data());
    std::vector<uint32_t> pages(spec.compute_packed_buffer_size_bytes() / sizeof(uint32_t));
    auto* out = reinterpret_cast<uint8_t*>(pages.data());
    if (!is_gguf_expanded(type)) {
        pack_pages(type, in, K, N, out);
    } else if (source == GGUF_TQ1_0) {
        const size_t count = bytes / 54;
        std::vector<uint8_t> tq2(count * 66);
        uint8_t back[54];
        for (size_t b = 0; b < count; ++b) {
            tq1_to_tq2(in + 54 * b, tq2.data() + 66 * b);
            TT_FATAL(
                tq2_to_tq1(tq2.data() + 66 * b, back) && std::memcmp(back, in + 54 * b, 54) == 0,
                "{}: TQ1_0 block {} has a byte that TQ1_0 quantization does not write",
                type,
                b);
        }
        pack_pages(GGUF_TQ2_0, tq2.data(), K, N, out);
    } else {
        std::vector<uint8_t> compact(bytes);
        pack_pages(source, in, K, N, compact.data());
        expand_pages(source, compact.data(), pages.size() * sizeof(uint32_t) / gguf_traits(type).page_bytes(), out);
    }
    return HostTensor::from_buffer(HostBuffer(std::move(pages)), spec);
}

std::vector<std::byte> to_gguf(const HostTensor& tensor) {
    const auto [K, N] = gguf_shape(tensor.tensor_spec());
    const HostBuffer buffer = host_buffer::get_host_buffer(tensor);
    const auto pages = buffer.view_bytes();
    TT_FATAL(
        pages.size() == tensor.tensor_spec().compute_packed_buffer_size_bytes(), "GGUF tensor buffer size mismatch");
    const DataType type = tensor.dtype(), source = gguf_source_dtype(type);
    const GgufTraits& f = gguf_traits(source);
    std::vector<std::byte> blocks(size_t(N) * size_t(K / f.block_elems) * f.block_bytes);
    const auto* in = reinterpret_cast<const uint8_t*>(pages.data());
    auto* out = reinterpret_cast<uint8_t*>(blocks.data());
    if (!is_gguf_expanded(type)) {
        unpack_pages(type, in, K, N, out);
    } else if (source == GGUF_TQ1_0) {
        const size_t count = blocks.size() / 54;
        std::vector<uint8_t> tq2(count * 66);
        unpack_pages(GGUF_TQ2_0, in, K, N, tq2.data());
        for (size_t b = 0; b < count; ++b) {
            TT_FATAL(tq2_to_tq1(tq2.data() + 66 * b, out + 54 * b), "{}: block {} holds a code 3", type, b);
        }
    } else {
        std::vector<uint8_t> compact(blocks.size());
        compact_pages(source, in, pages.size() / gguf_traits(type).page_bytes(), compact.data());
        unpack_pages(source, compact.data(), K, N, out);
    }
    return blocks;
}

}  // namespace tt::tt_metal
