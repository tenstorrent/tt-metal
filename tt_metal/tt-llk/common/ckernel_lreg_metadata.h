// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#pragma once

#ifndef TT_LLK_SFPRAWLREG_EFFECT
#if defined(__riscv_xtt_sfprawlreg_effect)
#define TT_LLK_SFPRAWLREG_EFFECT(r, w) __builtin_rvtt_sfprawlreg_effect((r), (w))
#else
#if defined(__has_builtin)
#if __has_builtin(__builtin_rvtt_sfprawlreg_effect)
#define TT_LLK_SFPRAWLREG_EFFECT(r, w) __builtin_rvtt_sfprawlreg_effect((r), (w))
#endif
#endif
#endif
#endif
#ifndef TT_LLK_SFPRAWLREG_EFFECT
#define TT_LLK_SFPRAWLREG_EFFECT(r, w) ((void)0)
#endif

namespace ckernel::raw_lreg_effect {
constexpr unsigned bit(unsigned r) { return r < 8 ? 1u << r : 0u; }
constexpr unsigned op(unsigned x) { return x >> 24; }
constexpr unsigned mod(unsigned x) { return x & 15; }
constexpr unsigned d(unsigned x) { return (x >> 4) & 15; }
constexpr unsigned c(unsigned x) { return (x >> 8) & 15; }
constexpr unsigned b(unsigned x) { return (x >> 12) & 15; }
constexpr unsigned a(unsigned x) { return (x >> 16) & 15; }
constexpr unsigned mr(unsigned x) { return (x >> 20) & 15; }
constexpr unsigned mm(unsigned x) { return (x >> 16) & 15; }

constexpr bool supported(unsigned x) {
    if (op(x) == 0x94) return mod(x) <= 6;
    if (op(x) == 0x95) return mod(x) == 0 || mod(x) == 2 || mod(x) == 3 || mod(x) == 10;
    return op(x) >= 0x70 && op(x) <= 0x99;
}
constexpr unsigned math_r(unsigned x, bool has_c = true) {
    unsigned r = bit(a(x)) | bit(b(x)) | (has_c ? bit(c(x)) : 0u);
    if (mod(x) & 4) r = 0xff;
    if (mod(x) & 8) r |= bit(7);
    return r;
}
constexpr unsigned math_w(unsigned x) { return (mod(x) & 8) ? 0xff : bit(d(x)); }
constexpr unsigned shft2_r(unsigned x) {
    switch (mod(x)) {
        case 0: return 0x0e; case 1: return 0x0f;
        case 2: return 0x0e | bit(c(x));
        case 3: case 4: return bit(c(x));
        case 5: return bit(b(x)) | bit(c(x));
        case 6: return bit(b(x)); default: return 0xff;
    }
}
constexpr unsigned shft2_w(unsigned x) {
    return mod(x) <= 2 ? 0x0f : (mod(x) <= 6 ? bit(d(x)) : 0xff);
}
constexpr unsigned read(unsigned x) {
    const auto o = op(x), m = mod(x), dd = d(x), cc = c(x);
    switch (o) {
        case 0x70: return (mm(x) == 14 || mm(x) == 15) ? bit(mr(x)) : 0;
        case 0x71: return (mm(x) == 8 || mm(x) == 10) ? bit(mr(x)) : 0;
        case 0x72: return bit(mr(x));
        case 0x73: return 0x0f | ((mm(x) & 8) ? bit(7) : 0u);
        case 0x74: case 0x75: return bit(dd) | ((m & 8) ? bit(7) : 0u);
        case 0x76: case 0x77: case 0x78: case 0x7d: case 0x81: case 0x90: return bit(cc);
        case 0x79: return bit(cc) | ((m & 1) ? 0u : bit(dd));
        case 0x7a:
#if defined(TT_LLK_SFPU_ARCH_BH)
            return bit(dd) | (!(m & 1) || (m & 4) ? bit(cc) : 0u);
#else
            return bit(dd) | ((m & 1) ? 0u : bit(cc));
#endif
        case 0x7b: return (m & 8) || (m & 1) ? 0u : bit(cc);
        case 0x7c: return (m & 8) ? 0u : bit(cc);
        case 0x7e: case 0x7f:
#if defined(TT_LLK_SFPU_ARCH_BH)
            return bit(cc) | bit((m & 1) ? b(x) : dd);
#else
            return bit(cc) | bit(dd);
#endif
        case 0x8d: return bit(cc) | bit(dd);
        case 0x80: return bit(cc);
        case 0x82: case 0x83: case 0x89: return bit(cc) | ((m & 1) ? 0u : bit(dd));
        case 0x84: case 0x85: case 0x86: return math_r(x);
        case 0x8c: return 0xff;
        case 0x8e: return bit(cc) | ((m & 8) ? 0u : bit(b(x)));
        case 0x91: return dd < 4 || !(m & 1) ? bit(0) : 0u;
        case 0x92: return bit(cc) | bit(dd) | bit(4 + (cc & 3)) | bit(4 + (dd & 3));
        // The configured LOADMACRO sequence is not encoded in this word.
        // Model every LREG as an input and output rather than hiding effects.
        case 0x93: return 0xff;
        case 0x94: return shft2_r(x);
        case 0x95: return m == 10 ? 0x8f : 0x7f;
        case 0x96: case 0x97: return bit(cc) | bit(dd);
        case 0x98:
#if defined(TT_LLK_SFPU_ARCH_QSR)
            return math_r(x, false);
#else
            return math_r(x);
#endif
        case 0x99:
#if defined(TT_LLK_SFPU_ARCH_BH)
            return bit(cc) | (m == 1 ? bit(b(x)) : 0u);
#else
            return bit(cc);
#endif
        default: return 0;
    }
}
constexpr unsigned write(unsigned x) {
    const auto o = op(x), m = mod(x), dd = d(x), cc = c(x);
    switch (o) {
        // Conservatively include the paired high-half LREG when it is encodable.
        case 0x70: return bit(mr(x)) | (mr(x) < 4 ? bit(mr(x) + 4) : 0u);
        case 0x71: return bit(mr(x));
        case 0x73: return (mm(x) & 8) ? 0xff : bit(mr(x));
        case 0x74: case 0x75: return (m & 8) ? 0xff : bit(dd);
        case 0x76: case 0x77: case 0x78: case 0x79: case 0x7a: case 0x7c:
        case 0x7d: case 0x7e: case 0x7f: case 0x80: case 0x81: case 0x82:
        case 0x83: case 0x89: case 0x8d: case 0x8e: case 0x90: case 0x99: return bit(dd);
        case 0x84: case 0x85: case 0x86: case 0x98: return math_w(x);
        case 0x8c: return 0xff;
        case 0x92: return bit(cc) | bit(dd) | bit(4 + (cc & 3)) | bit(4 + (dd & 3));
        case 0x93: return 0xff;
        case 0x94: return shft2_w(x);
        case 0x95: return m == 10 ? 0xff : bit(dd);
        case 0x96: case 0x97: return (m & 8) ? bit(dd) : 0u;
        default: return 0;
    }
}
// Raw instructions execute under externally managed lane predicates. The
// instruction word alone cannot prove that every destination lane is written.
// Preserve the old destination as an input, including for nominally write-only
// operations. This is deliberately conservative when all lanes are enabled.
constexpr unsigned inputs(unsigned x) { return read(x) | write(x); }
// A template argument requires constant evaluation without introducing local
// constexpr objects (whose initialization can emit stack stores at -O0).
template <unsigned Mask> struct constant_mask { enum : unsigned { value = Mask }; };
} // namespace ckernel::raw_lreg_effect

#define TT_LLK_SFPU_EFFECT(word)                                                                \
    do {                                                                                        \
        static_assert(!__builtin_constant_p(word) || ckernel::raw_lreg_effect::supported(word), \
                      "unsupported raw SFPU effect; use an explicit typed annotation");         \
        TT_LLK_SFPRAWLREG_EFFECT(                                                               \
            (ckernel::raw_lreg_effect::constant_mask<__builtin_constant_p(word)                  \
                ? ckernel::raw_lreg_effect::inputs(word) : 0xffu>::value),                       \
            (ckernel::raw_lreg_effect::constant_mask<__builtin_constant_p(word)                  \
                ? ckernel::raw_lreg_effect::write(word) : 0xffu>::value));                       \
    } while (0)
#define TT_LLK_SFPU_ISSUE_TT(word) do { ckernel::instrn_buffer[0] = word; TT_LLK_SFPU_EFFECT(word); } while (0)
#define TT_LLK_SFPU_ISSUE_TTI(word) do { INSTRUCTION_WORD(word); TT_LLK_SFPU_EFFECT(word); } while (0)
