// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#pragma once

// Internal TT/TTI macro-to-compiler interface, not a user-facing SFPI API.
// The effect marker describes the immediately preceding raw instruction;
// its implementing compiler pass reconstructs hard-register lifetimes.
// The marker is SFPI's lltt::lreg_effect. SFPI headers older than the
// compiler lack it, so the builtin it wraps is called directly instead.
// Existing compilers retain the original raw instruction path below, NOT
// the allocator protection. Callers explicitly carrying values between raw
// operations can instead use SFPI's public l_reg reads and assignments.
// An independent l_reg[R].in_use() at each endpoint is not that lifetime.
// The older __builtin_rvtt_sfprawlreg_access is NOT a fallback: its first
// mask releases a value, whereas the effect builtin's first mask is a read.
// Access-only and older compilers therefore get the explicit no-op below;
// define TT_LLK_REQUIRE_RAW_LREG_METADATA to reject them instead.
#ifndef TT_LLK_SFPRAWLREG_EFFECT
#if defined(__has_builtin)
#if __has_builtin(__builtin_rvtt_sfprawlreg_effect)
#if __has_include("lltt.h")
#include "lltt.h"
#endif
#if defined(LLTT_HAS_LREG_EFFECT)
#define TT_LLK_SFPRAWLREG_EFFECT(r, w) lltt::lreg_effect((r), (w))
#else
#define TT_LLK_SFPRAWLREG_EFFECT(r, w) __builtin_rvtt_sfprawlreg_effect((r), (w))
#endif
#endif
#endif
#endif
#ifndef TT_LLK_SFPRAWLREG_EFFECT
#if defined(TT_LLK_REQUIRE_RAW_LREG_METADATA)
#error "raw-LREG effect metadata required, but this compiler has no effect builtin"
#endif
#define TT_LLK_SFPRAWLREG_EFFECT(r, w) ((void)0)
#endif

// Forced inline so a word that becomes constant only after inlining folds to a
// constant mask instead of a call to the decoder.
#define TT_LLK_RAW_LREG_FN __attribute__((always_inline)) constexpr inline

namespace ckernel::raw_lreg_effect {
constexpr unsigned all_lregs = 0xff;
TT_LLK_RAW_LREG_FN unsigned bit(unsigned r) { return r < 8 ? 1u << r : 0u; }
TT_LLK_RAW_LREG_FN unsigned op(unsigned x) { return x >> 24; }
TT_LLK_RAW_LREG_FN unsigned mod(unsigned x) { return x & 15; }
TT_LLK_RAW_LREG_FN unsigned d(unsigned x) { return (x >> 4) & 15; }
TT_LLK_RAW_LREG_FN unsigned c(unsigned x) { return (x >> 8) & 15; }
TT_LLK_RAW_LREG_FN unsigned b(unsigned x) { return (x >> 12) & 15; }
TT_LLK_RAW_LREG_FN unsigned a(unsigned x) { return (x >> 16) & 15; }
TT_LLK_RAW_LREG_FN unsigned mr(unsigned x) { return (x >> 20) & 15; }
TT_LLK_RAW_LREG_FN unsigned mm(unsigned x) { return (x >> 16) & 15; }

TT_LLK_RAW_LREG_FN bool supported(unsigned x) {
    if (op(x) == 0x94) return mod(x) <= 6;
    // Sign retention (4) and indirect destination (8) are independent flags.
    // The ISA admits 0,2,3,4,6,7,A,E; all other modifier values are reserved.
    if (op(x) == 0x95) return (0x44ddu & (1u << mod(x))) != 0;
    // Each architecture's assembly.yaml defines every SFPU opcode in this
    // range; WH stops at SFPLUTFP32.
#if defined(TT_LLK_SFPU_ARCH_WH)
    return op(x) >= 0x70 && op(x) <= 0x95;
#else
    return op(x) >= 0x70 && op(x) <= 0x99;
#endif
}
TT_LLK_RAW_LREG_FN unsigned math_r(unsigned x, bool has_c = true) {
    unsigned r = bit(a(x)) | bit(b(x)) | (has_c ? bit(c(x)) : 0u);
    if (mod(x) & 4) r = all_lregs;
    if (mod(x) & 8) r |= bit(7);
    return r;
}
TT_LLK_RAW_LREG_FN unsigned math_w(unsigned x) { return (mod(x) & 8) ? all_lregs : bit(d(x)); }
TT_LLK_RAW_LREG_FN unsigned shft2_r(unsigned x) {
    switch (mod(x)) {
        case 0: return 0x0e; case 1: return 0x0f;
        case 2: return 0x0e | bit(c(x));
        case 3: case 4: return bit(c(x));
        // WH/BH shift LReg[Imm12 & 15] into VD (by VC in mode 5, by Imm12 in
        // mode 6); Quasar documents VD as the source. Read both: VD is already
        // an input through the destination write.
        case 5: return bit(b(x)) | bit(c(x)) | bit(d(x));
        case 6: return bit(b(x)) | bit(d(x));
        default: return all_lregs;
    }
}
TT_LLK_RAW_LREG_FN unsigned shft2_w(unsigned x) {
    return mod(x) <= 2 ? 0x0f : (mod(x) <= 6 ? bit(d(x)) : all_lregs);
}
TT_LLK_RAW_LREG_FN unsigned read(unsigned x) {
    const auto o = op(x), m = mod(x), dd = d(x), cc = c(x);
    switch (o) {
        case 0x70: /* SFPLOAD */ return (mm(x) == 14 || mm(x) == 15) ? bit(mr(x)) : 0;
        case 0x71: /* SFPLOADI */ return (mm(x) == 8 || mm(x) == 10) ? bit(mr(x)) : 0;
        case 0x72: /* SFPSTORE */ return bit(mr(x));
        case 0x73: /* SFPLUT */ return 0x0f | ((mm(x) & 8) ? bit(7) : 0u);
        // Modifier bit 2 reads the source from the LREG named by LREG7[3:0],
        // which can be any LREG and is not known from the instruction word.
        case 0x74: case 0x75: /* SFPMULI, SFPADDI */ return (m & 4) ? all_lregs : bit(dd) | ((m & 8) ? bit(7) : 0u);
        // SFPDIVP2, SFPEXEXP, SFPEXMAN, SFPABS, SFPLZ, SFPCAST
        case 0x76: case 0x77: case 0x78: case 0x7d: case 0x81: case 0x90: return bit(cc);
        case 0x79: /* SFPIADD */ return bit(cc) | ((m & 1) ? 0u : bit(dd));
        // lreg_c is the shift amount unless bit 0 selects imm12, and the data
        // input when bit 2 is set (BH/QSR). WH documents no bit 2; honoring it
        // there only adds a read.
        case 0x7a: /* SFPSHFT */ return bit(dd) | (!(m & 1) || (m & 4) ? bit(cc) : 0u);
        case 0x7b: /* SFPSETCC */ return (m & 8) || (m & 1) ? 0u : bit(cc);
        case 0x7c: /* SFPMOV */ return (m & 8) ? 0u : bit(cc);
        case 0x7e: case 0x7f: // SFPAND, SFPOR
#if defined(TT_LLK_SFPU_ARCH_BH)
            return bit(cc) | bit((m & 1) ? b(x) : dd);
#else
            return bit(cc) | bit(dd);
#endif
        case 0x8d: /* SFPXOR */ return bit(cc) | bit(dd);
        case 0x80: /* SFPNOT */ return bit(cc);
        case 0x82: case 0x83: case 0x89: /* SFPSETEXP, SFPSETMAN, SFPSETSGN */ return bit(cc) | ((m & 1) ? 0u : bit(dd));
        case 0x84: case 0x85: case 0x86: /* SFPMAD, SFPADD, SFPMUL */ return math_r(x);
        case 0x8c: /* SFPTRANSP */ return all_lregs;
        case 0x8e: /* SFP_STOCH_RND */ return bit(cc) | ((m & 8) ? 0u : bit(b(x)));
        case 0x91: /* SFPCONFIG */ return dd < 4 || !(m & 1) ? bit(0) : 0u;
        case 0x92: /* SFPSWAP */ return bit(cc) | bit(dd) | bit(4 + (cc & 3)) | bit(4 + (dd & 3));
        // The configured LOADMACRO sequence is not encoded in this word.
        // Model every LREG as an input and output rather than hiding effects.
        // The scheduled sub-ops finish after issue, but this marks the issue
        // point only: typed SFPI code must not sit between a LOADMACRO and its
        // drain. The compiler never moves typed SFPU instructions there, since
        // they are volatile and so ordered against raw words.
        case 0x93: /* SFPLOADMACRO */ return all_lregs;
        case 0x94: /* SFPSHFT2 */ return shft2_r(x);
        case 0x95: /* SFPLUTFP32 */ return ((m & 10) == 10 ? 0x0fu : 0x7fu) | ((m & 8) ? 0x80u : 0u);
        case 0x96: case 0x97: /* SFPLE, SFPGT */ return bit(cc) | bit(dd);
        case 0x98: // SFPMUL24
#if defined(TT_LLK_SFPU_ARCH_QSR)
            return math_r(x, false /*has_c*/);
#else
            return math_r(x);
#endif
        case 0x99: // SFPARECIP (BH), SFPNONLINEAR (QSR)
#if defined(TT_LLK_SFPU_ARCH_BH)
            return bit(cc) | (m == 1 ? bit(b(x)) : 0u);
#else
            return bit(cc);
#endif
        // CC stack, lane-enable and NOP instructions name no LREG.
        case 0x87: case 0x88: case 0x8a: case 0x8b: case 0x8f: return 0; // SFPPUSHC, SFPPOPC, SFPENCC, SFPCOMPC, SFPNOP
        // An opcode without a model must not yield an empty mask.
        default: return all_lregs;
    }
}
TT_LLK_RAW_LREG_FN unsigned write(unsigned x) {
    const auto o = op(x), m = mod(x), dd = d(x), cc = c(x);
    switch (o) {
        // Conservatively include the paired high-half LREG when it is encodable.
        case 0x70: /* SFPLOAD */ return bit(mr(x)) | (mr(x) < 4 ? bit(mr(x) + 4) : 0u);
        case 0x71: /* SFPLOADI */ return bit(mr(x));
        case 0x73: /* SFPLUT */ return (mm(x) & 8) ? all_lregs : bit(mr(x));
        case 0x74: case 0x75: /* SFPMULI, SFPADDI */ return (m & 8) ? all_lregs : bit(dd);
        // Single-destination ALU operations, SFP_STOCH_RND, SFPCAST and SFPARECIP/SFPNONLINEAR.
        case 0x76: case 0x77: case 0x78: case 0x79: case 0x7a: case 0x7c:
        case 0x7d: case 0x7e: case 0x7f: case 0x80: case 0x81: case 0x82:
        case 0x83: case 0x89: case 0x8d: case 0x8e: case 0x90: case 0x99: return bit(dd);
        case 0x84: case 0x85: case 0x86: case 0x98: /* SFPMAD, SFPADD, SFPMUL, SFPMUL24 */ return math_w(x);
        case 0x8c: /* SFPTRANSP */ return all_lregs;
        case 0x92: /* SFPSWAP */ return bit(cc) | bit(dd) | bit(4 + (cc & 3)) | bit(4 + (dd & 3));
        case 0x93: /* SFPLOADMACRO */ return all_lregs;
        case 0x94: /* SFPSHFT2 */ return shft2_w(x);
        case 0x95: /* SFPLUTFP32 */ return (m & 8) ? all_lregs : bit(dd);
        case 0x96: case 0x97: /* SFPLE, SFPGT */ return (m & 8) ? bit(dd) : 0u;
        // Stores, CC updates and SFPCONFIG write no LREG in 0..7.
        case 0x72: case 0x7b: case 0x87: case 0x88: case 0x8a: case 0x8b:
        case 0x8f: case 0x91: return 0;
        default: return all_lregs;
    }
}
// Raw instructions execute under externally managed lane predicates. The
// instruction word alone cannot prove that every destination lane is written.
// A compiler advertising lane merging preserves old destination lanes itself.
// Older effect compilers need the destination explicitly included as an input.
// Keep that compatibility rule shared with caller-described LOADMACRO effects.
#if defined(__riscv_xtt_sfprawlreg_merge)
TT_LLK_RAW_LREG_FN unsigned lane_inputs(unsigned reads, unsigned) { return reads; }
#else
TT_LLK_RAW_LREG_FN unsigned lane_inputs(unsigned reads, unsigned writes) { return reads | writes; }
#endif
TT_LLK_RAW_LREG_FN unsigned inputs(unsigned x) { return lane_inputs(read(x), write(x)); }
// An unsupported word reached only after inlining has no static_assert; keep
// every LREG for it rather than trusting a partial decode.
TT_LLK_RAW_LREG_FN unsigned checked_inputs(unsigned x) { return supported(x) ? inputs(x) : all_lregs; }
TT_LLK_RAW_LREG_FN unsigned checked_write(unsigned x) { return supported(x) ? write(x) : all_lregs; }
// REPLAY with load_mode 0 plays back recorded words whose own effects were
// marked, if at all, where they were recorded; model playback as reading and
// writing every LREG. A load_mode 1 recording executes nothing here, or with
// execute_while_loading executes each word at its own marked site.
// Markers of words recorded without executing are only conservative: every
// claimed write is also an input, so it never ends an earlier value's
// reservation. lltt::replay is a compiler builtin outside these wrappers;
// neither it nor the compiler marks its playback effects.
TT_LLK_RAW_LREG_FN unsigned replay_effect(unsigned x) { return (x & 1) ? 0u : all_lregs; }
// A template argument requires constant evaluation without introducing local
// constexpr objects (whose initialization can emit stack stores at -O0).
template <unsigned Mask> struct constant_mask { enum : unsigned { value = Mask }; };
} // namespace ckernel::raw_lreg_effect

#if defined(__OPTIMIZE__)
// __builtin_constant_p here is resolved after inlining, so LREG operands of an
// inline helper are still decoded exactly. A word that never becomes constant
// selects all_lregs, so no decoder code survives to the generated program.
#define TT_LLK_SFPU_MASK(decode, lreg_word)                                                     \
    (__builtin_constant_p(lreg_word) ? ckernel::raw_lreg_effect::decode(lreg_word)              \
                                     : ckernel::raw_lreg_effect::all_lregs)
#else
// Nothing is inlined without optimization, so only front-end constants can be
// decoded; the template argument keeps that decode out of the generated code.
#define TT_LLK_SFPU_MASK(decode, lreg_word)                                                     \
    (ckernel::raw_lreg_effect::constant_mask<__builtin_constant_p(lreg_word)                     \
        ? ckernel::raw_lreg_effect::decode(lreg_word) : ckernel::raw_lreg_effect::all_lregs>::value)
#endif
// LREG_WORD is the issued instruction with operands that can never name an
// LREG or select how one is accessed (Dest address, address mode, immediate
// data) set to zero. A runtime address or immediate then leaves the mask exact.
#define TT_LLK_SFPU_EFFECT(lreg_word)                                                                     \
    do {                                                                                                  \
        static_assert(!__builtin_constant_p(lreg_word) || ckernel::raw_lreg_effect::supported(lreg_word), \
                      "unsupported raw SFPU effect; use an explicit typed annotation");                   \
        TT_LLK_SFPRAWLREG_EFFECT(TT_LLK_SFPU_MASK(checked_inputs, lreg_word),                             \
                                 TT_LLK_SFPU_MASK(checked_write, lreg_word));                             \
    } while (0)
// Statement expressions keep TT_/TTI_ SFPU macros usable wherever the bare
// TT_INSN/TTI_INSN expressions were, e.g. as comma or conditional operands.
#define TT_LLK_SFPU_ISSUE_TT_LREGS(word, lreg_word)  ({ TT_INSN(word); TT_LLK_SFPU_EFFECT(lreg_word); })
#define TT_LLK_SFPU_ISSUE_TTI_LREGS(word, lreg_word) ({ TTI_INSN(word); TT_LLK_SFPU_EFFECT(lreg_word); })
#define TT_LLK_SFPU_ISSUE_TT(word)                   TT_LLK_SFPU_ISSUE_TT_LREGS(word, word)
#define TT_LLK_SFPU_ISSUE_TTI(word)                  TT_LLK_SFPU_ISSUE_TTI_LREGS(word, word)
// These words program a configured LOADMACRO template (VD 12..15 with the
// instruction-template backdoor enabled); they do not execute its SFPU work.
// Use only while programming such a template, never for an ordinary raw word.
#define TT_LLK_SFPU_ISSUE_TTI_TEMPLATE(word) ({ TTI_INSN(word); TT_LLK_SFPRAWLREG_EFFECT(0u, 0u); })
// A caller that owns the configured LOADMACRO may describe its full LREG effects.
// Unknown macros must continue to use TT_SFPLOADMACRO's all-register fallback.
// The caller must also drain the scheduled work before subsequent typed work.
#define TT_LLK_SFPU_ISSUE_TT_MACRO(word, reads, writes)                                               \
    ({ TT_INSN(word);                                                                               \
       TT_LLK_SFPRAWLREG_EFFECT(                                                                    \
           (ckernel::raw_lreg_effect::constant_mask<                                                 \
               ckernel::raw_lreg_effect::lane_inputs((reads), (writes))>::value),                    \
           (ckernel::raw_lreg_effect::constant_mask<(writes)>::value)); })
// MODE_WORD is the REPLAY word with only its load_mode operand kept.
#define TT_LLK_REPLAY_EFFECT(mode_word) \
    TT_LLK_SFPRAWLREG_EFFECT(TT_LLK_SFPU_MASK(replay_effect, mode_word), TT_LLK_SFPU_MASK(replay_effect, mode_word))
#define TT_LLK_REPLAY_ISSUE_TT(word, mode_word)  ({ TT_INSN(word); TT_LLK_REPLAY_EFFECT(mode_word); })
#define TT_LLK_REPLAY_ISSUE_TTI(word, mode_word) ({ TTI_INSN(word); TT_LLK_REPLAY_EFFECT(mode_word); })
