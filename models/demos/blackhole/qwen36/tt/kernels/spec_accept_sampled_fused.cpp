// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// QWEN36_SPEC_ACCEPT_KERNEL_SAMPLED (tt/spec_fused_kernels.py, SampledAcceptKernel): the stage-S2 SAMPLED accept trace body
// (spec_fused_sampled.FusedSampledDevice.body: union top-M merge, temperature / top-k / top-p softmax, the lossless
// point-mass accept walk with the runner-drawn uniforms, the S1 drafter tail; ~115 ttnn ops) as ONE data-movement kernel.
// Bit-exact host model: tt/spec_sampled_kernel_math.py (its docstring has the arithmetic; the device test compares every
// output word with it).
//
// Inputs:
//   union_vals_in bf16   [1,1,32,TP*M] TILE  per-shard top-M blocks: block d = columns d*M.., each (value desc, local id asc)
//   union_ids_in  uint32 [1,1,32,TP*M] TILE  their local ids (global id = local + d * PER_SHARD)
//   ids_in        uint32 [1,1,1,32]    RM    verify argmax ids (greedy / inactive users commit them, as S1)
//   tok_in        uint32 [1,R]         RM    verify token block (column 0 pending, columns 1..K drafts)
//   aux_in        int32  [7,W]         RM    spec_fused.fused_accept_aux_host
//   samp_in       fp32   [4+B,W]       RM    rows temperature / top_k / top_p / greedy (word u), then one row per user:
//                                            K+2 uniforms (bonus, accepts 0..K-1, correction)
//   pt_rows_in    int32  [R,NB]        RM    the page-table row of each engine row
// Outputs (every word written, zero padding included, as the op graph's ttnn.copy does):
//   pack_out   uint32 [2K+5,W]  drafter input pack   rs_pt_out int32 [R,NB]  reseed page table
//   result_out int32  [3,W]     row 0 committed[R] (-1 past the count), row 1 counts[B], row 2 overflow[B]
// Grid: core c = y * NX + x handles engine row r = c (R cores; extra cores return). User u = r / T, column j = r % T, and
// the user's LEADER is core u*T. Phases:
//   A (every row core): read the small inputs and this row's union face rows (64 B aligned), prefetch its pt row; a
//     SAMPLED user's rows j <= nv compute {accept_j, pick_j, overflow_j}; non-leaders post that record to the leader.
//   B (leader): wait for the T-1 row records, walk (m, committed[m], overflow, real rows), post `real` back to its rows
//     and the user record to core 0.
//   C (every row core): write rs_pt row r: its pt row if row r is real for its user, else scratch words.
//   D (core 0): wait for the B-1 user records, write pack_out and result_out.
// Every post is a NOC write acked before its semaphore increment (spec_argmax_fused's protocol). Every core posts and
// waits a fixed number of times, independent of the data, so the protocol cannot deadlock.
// Float32: libgcc soft-float calls in the reference's operation order (correctly rounded, round to nearest even, gradual
// underflow, canonical NaN); exp and every sum in integer arithmetic. Every float operation is an explicit call, so the
// build's -ffast-math (reciprocal rewrites, contraction) cannot change a result.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

extern "C" float __addsf3(float, float);
extern "C" float __subsf3(float, float);
extern "C" float __mulsf3(float, float);
extern "C" float __divsf3(float, float);

namespace {

constexpr uint32_t B = get_compile_time_arg_val(0);
constexpr uint32_t K = get_compile_time_arg_val(1);
constexpr uint32_t W = get_compile_time_arg_val(2);   // words per aux / samp / pack / result row
constexpr uint32_t NB = get_compile_time_arg_val(3);  // words per page-table row
constexpr uint32_t NX = get_compile_time_arg_val(4);  // worker grid width
constexpr uint32_t M = get_compile_time_arg_val(5);   // candidates per shard block = global candidates
constexpr uint32_t TP = get_compile_time_arg_val(6);  // shard blocks in the union
constexpr uint32_t PER_SHARD = get_compile_time_arg_val(7);
constexpr uint32_t RS_CHUNK = get_compile_time_arg_val(8);   // bytes of one pt / rs_pt row piece (multiple of 64)
constexpr uint32_t SCR_BYTES = get_compile_time_arg_val(9);  // bytes of the scratch-word piece (multiple of 64)
constexpr uint32_t CB_L1 = get_compile_time_arg_val(10);
constexpr uint32_t CB_BYTES = get_compile_time_arg_val(11);
constexpr uint32_t CTA_ACCESSORS = 12;
// Common runtime args: 10 buffer addresses (ids, tok, aux, samp, union vals, union ids, pt rows, pack, rs_pt, result),
// then the (x, y) NOC coordinates of row core c, c = 0..R-1.
constexpr uint32_t ARG_NOC = 10;

constexpr uint32_t T = K + 1;
constexpr uint32_t R = B * T;
constexpr uint32_t TPM = TP * M;
constexpr uint32_t NT = (TPM + 31) / 32;  // union tiles along the width
constexpr uint32_t PACK_ROWS = 2 * K + 5;
constexpr uint32_t ROW_BYTES = NB * 4;
constexpr uint32_t RS_PIECES = (ROW_BYTES + RS_CHUNK - 1) / RS_CHUNK;

// spec_fused.AUX_* / CONST_* rows, spec_fused_sampled samp rows
constexpr uint32_t AUX_NUM_VALID = 0, AUX_ACTIVE = 1, AUX_DECLINED = 2, AUX_KEEP = 3, AUX_P0 = 4, AUX_CONSTS = 5;
constexpr uint32_t AUX_DELTA = 6, AUX_ROWS = 7;
constexpr uint32_t CONST_SCRATCH = 0, CONST_ROPE_ROWS = 2;
constexpr uint32_t SAMP_TEMP = 0, SAMP_TOPK = 1, SAMP_TOPP = 2, SAMP_GREEDY = 3, SAMP_PARAM_ROWS = 4;
constexpr uint32_t SEM_ROW = 0, SEM_BACK = 1, SEM_USER = 2;

constexpr uint32_t F_ONE = 0x3F800000u, F_NAN = 0x7FC00000u, F_INF = 0x7F800000u, F_NEG_INF = 0xFF800000u;
constexpr int32_t E_FRAC = 56, P_FRAC = 62;

static_assert(K >= 1 && B >= 1 && R <= 32, "one 32-row verify bucket");
static_assert(R <= W && B <= W && K + 2 <= W && 5 <= W, "row width");
static_assert(M >= 2 && M <= 64 && TP >= 1 && TP <= 32 && TPM <= 1024, "union of at most 32 blocks / 1024 candidates");
static_assert(RS_CHUNK % 64 == 0 && RS_CHUNK >= 64 && SCR_BYTES % 64 == 0 && SCR_BYTES >= 64, "64 B pieces");
static_assert(sizeof(float) == 4, "float32");

constexpr uint32_t al64(uint32_t x) { return (x + 63u) & ~63u; }

// L1 layout, byte offsets from the 64 B aligned CB base (identical on every core: posts land at the sender's offsets)
constexpr uint32_t ROW = al64(W * 4);
constexpr uint32_t MW = al64(M * 4);
constexpr uint32_t OFF_TOK = 0;                            // R words
constexpr uint32_t OFF_IDS = OFF_TOK + al64(R * 4);        // 32 words
constexpr uint32_t OFF_AUX = OFF_IDS + 128;                // AUX_ROWS rows
constexpr uint32_t OFF_SAMP = OFF_AUX + AUX_ROWS * ROW;    // samp rows 0..3, then this user's uniform row
constexpr uint32_t OFF_UV = OFF_SAMP + 5 * ROW;            // NT tiles x 2 faces x 64 B (bf16 face rows, even|odd)
constexpr uint32_t OFF_UI = OFF_UV + NT * 128;             // NT tiles x 2 faces x 64 B (uint32 face row)
constexpr uint32_t OFF_GID = OFF_UI + NT * 128;            // M words each: merged global ids,
constexpr uint32_t OFF_VB = OFF_GID + MW;                  //   v = vals / temp,
constexpr uint32_t OFF_EB = OFF_VB + MW;                   //   e = exp(v - v0),
constexpr uint32_t OFF_PR = OFF_EB + MW;                   //   prob,
constexpr uint32_t OFF_CDF = OFF_PR + MW;                  //   pick cdf
constexpr uint32_t OFF_HEAD = OFF_CDF + MW;                // (reserved: 8 * TP bytes; the merge heads live in registers)
constexpr uint32_t OFF_ROWREC = OFF_HEAD + al64(8 * TP);   // T row records of 16 B (read on the leader)
constexpr uint32_t OFF_BACK = OFF_ROWREC + al64(16 * T);   // the leader's back record (read on its rows)
constexpr uint32_t OFF_USER = OFF_BACK + 64;               // B user records of 16 B (read on core 0)
constexpr uint32_t OFF_PACK = OFF_USER + al64(16 * B);     // PACK_ROWS rows (core 0)
constexpr uint32_t OFF_RES = OFF_PACK + PACK_ROWS * ROW;   // 3 rows (core 0)
constexpr uint32_t OFF_SCR = OFF_RES + 3 * ROW;            // SCR_BYTES of scratch words
constexpr uint32_t OFF_RS = OFF_SCR + SCR_BYTES;           // RS_CHUNK: this row's pt piece
constexpr uint32_t L1_BYTES = OFF_RS + RS_CHUNK;
static_assert(L1_BYTES + 64 <= CB_BYTES, "scratch CB too small");

using wptr = volatile tt_l1_ptr uint32_t*;
using iptr = volatile tt_l1_ptr int32_t*;
using hptr = volatile tt_l1_ptr uint16_t*;

// constants of the integer exp (tt/spec_sampled_kernel_math.py holds the same literals)
constexpr uint64_t L2E = 0x5C551D94AE0BF85Eull;     // round(log2(e) * 2^62)
constexpr uint64_t LN2_64 = 0xB17217F7D1CF79ACull;  // round(ln(2) * 2^64)
constexpr uint64_t POLY0 = 0x4000000000000000ull, POLY1 = 0x4000000000000000ull, POLY2 = 0x2000000000000000ull;
constexpr uint64_t POLY3 = 0x0AAAAAAAAAAAAAABull, POLY4 = 0x02AAAAAAAAAAAAABull, POLY5 = 0x0088888888888889ull;
constexpr uint64_t POLY6 = 0x0016C16C16C16C17ull;  // round(2^62 / k!), k = 0..6
constexpr uint64_t EXP2_TAB[64] = {  // round(2^(i/64) * 2^62)
    0x4000000000000000ull, 0x40B268F9DE0183BAull, 0x4166C34C5615D0ECull, 0x421D1461D66F2023ull,
    0x42D561B3E6243D8Aull, 0x438FB0CB4F468808ull, 0x444C0740496D4294ull, 0x450A6ABAA4B77ECDull,
    0x45CAE0F1F545EB73ull, 0x468D6FADBF2DD4F3ull, 0x47521CC5A2E6A9E0ull, 0x4818EE218A3358EEull,
    0x48E1E9B9D588E19Bull, 0x49AD159789F37496ull, 0x4A7A77D47F7B84B1ull, 0x4B4A169B900C2D00ull,
    0x4C1BF828C6DC54B8ull, 0x4CF022C9905BFD32ull, 0x4DC69CDCEAA72A9Cull, 0x4E9F6CD3967FDBA8ull,
    0x4F7A993048D088D7ull, 0x50582887DCB8A7E1ull, 0x513821818624B40Cull, 0x521A8AD704F3404Full,
    0x52FF6B54D8A89C75ull, 0x53E6C9DA74B29AB5ull, 0x54D0AD5A753E077Cull, 0x55BD1CDAD49F699Cull,
    0x56AC1F752150A563ull, 0x579DBC56B48521BAull, 0x5891FAC0E95612C8ull, 0x5988E20954889245ull,
    0x5A827999FCEF3242ull, 0x5B7EC8F19468BBC9ull, 0x5C7DD7A3B17DCF75ull, 0x5D7FAD59099F22FEull,
    0x5E8451CFAC061B5Full, 0x5F8BCCDB3D398841ull, 0x6096266533384A2Bull, 0x61A3666D124BB204ull,
    0x62B39508AA836D6Full, 0x63C6BA6455DCD8AEull, 0x64DCDEC3371793D1ull, 0x65F60A7F79393E2Eull,
    0x6712460A8FC24072ull, 0x683199ED779592CAull, 0x69540EC8F895722Dull, 0x6A79AD55E7F6FD10ull,
    0x6BA27E656B4EB57Aull, 0x6CCE8AE13C57EBDBull, 0x6DFDDBCBED791BABull, 0x6F307A412F074892ull,
    0x70666F76154A7089ull, 0x719FC4B95F452D29ull, 0x72DC8373BE41A454ull, 0x741CB5281E25EE34ull,
    0x75606373EE921C97ull, 0x76A7980F6CCA15C2ull, 0x77F25CCDEE6D7AE6ull, 0x7940BB9E2CFFD89Dull,
    0x7A92BE8A92436616ull, 0x7BE86FB985689DDCull, 0x7D41D96DB915019Dull, 0x7E9F06067A4360BAull,
};

// ------------------------------------------------------------------------------------------------ float32 words
FORCE_INLINE float as_f(uint32_t b) {
    float f;
    __builtin_memcpy(&f, &b, 4);
    return f;
}
FORCE_INLINE uint32_t as_b(float f) {
    uint32_t b;
    __builtin_memcpy(&b, &f, 4);
    return b;
}
FORCE_INLINE bool is_nan(uint32_t b) { return (b & 0x7FFFFFFFu) > 0x7F800000u; }
FORCE_INLINE uint32_t canon(uint32_t b) { return is_nan(b) ? F_NAN : b; }
FORCE_INLINE uint32_t fdiv(uint32_t a, uint32_t b) { return canon(as_b(__divsf3(as_f(a), as_f(b)))); }
FORCE_INLINE uint32_t fsub(uint32_t a, uint32_t b) { return canon(as_b(__subsf3(as_f(a), as_f(b)))); }
FORCE_INLINE uint32_t fmul(uint32_t a, uint32_t b) { return canon(as_b(__mulsf3(as_f(a), as_f(b)))); }
FORCE_INLINE uint32_t fadd(uint32_t a, uint32_t b) { return canon(as_b(__addsf3(as_f(a), as_f(b)))); }

// floor(2^62 / significand(b)) of a positive normal word (one 64-bit division per divisor); 0 otherwise
FORCE_INLINE uint64_t rcp62(uint32_t b) {
    const uint32_t ex = (b >> 23) & 0xFFu;
    if ((b & 0x80000000u) || ex == 0 || ex == 0xFFu) {
        return 0;
    }
    return (1ull << 62) / static_cast<uint64_t>((b & 0x7FFFFFu) | 0x800000u);
}

// a / b correctly rounded (round to nearest even, gradual underflow) from r = rcp62(b): Q = floor(ma * 2^26 / mb) from
// one multiply (exact or one short), one exact correction, then the IEEE rounding of Q with its remainder as sticky.
// A zero / denormal / non-finite numerator or a divisor without a reciprocal takes the soft-float division (the model's
// fdiv_r; equal to __divsf3 on every input, host-tested).
uint32_t fdiv_r(uint32_t a, uint32_t b, uint64_t r) {
    const uint32_t ea = (a >> 23) & 0xFFu, eb = (b >> 23) & 0xFFu;
    if (r == 0 || ea == 0 || ea == 0xFFu) {
        return fdiv(a, b);
    }
    const uint32_t sign = a & 0x80000000u;
    const uint32_t ma = (a & 0x7FFFFFu) | 0x800000u, mb = (b & 0x7FFFFFu) | 0x800000u;
    const uint64_t prod = static_cast<uint64_t>(ma) * static_cast<uint32_t>(r) +
                          ((static_cast<uint64_t>(ma) * static_cast<uint32_t>(r >> 32)) << 32);  // ma * r < 2^63
    uint64_t q = prod >> 36;
    uint64_t rem = (static_cast<uint64_t>(ma) << 26) - q * mb;
    if (rem >= mb) {
        ++q;
        rem -= mb;
    }
    int32_t e;
    uint32_t sh;
    if (q >= (1ull << 26)) {
        e = static_cast<int32_t>(ea) - static_cast<int32_t>(eb) + 127;
        sh = 3;
    } else {
        e = static_cast<int32_t>(ea) - static_cast<int32_t>(eb) + 126;
        sh = 2;
    }
    if (e >= 1) {
        uint32_t mant = static_cast<uint32_t>(q >> sh);
        const uint32_t lost = static_cast<uint32_t>(q) & ((1u << sh) - 1), half = 1u << (sh - 1);
        if (lost > half || (lost == half && (rem != 0 || (mant & 1u)))) {
            ++mant;
        }
        if (mant == (1u << 24)) {
            mant = 1u << 23;
            ++e;
        }
        if (e >= 0xFF) {
            return sign | F_INF;
        }
        return sign | (static_cast<uint32_t>(e) << 23) | (mant & 0x7FFFFFu);
    }
    const uint32_t ts = sh + 1 - static_cast<uint32_t>(e);  // a denormal result d * 2^-149
    if (ts > 40) {
        return sign;
    }
    uint64_t d = q >> ts;
    const uint64_t lost = q & ((1ull << ts) - 1), half = 1ull << (ts - 1);
    if (lost > half || (lost == half && (rem != 0 || (d & 1u)))) {
        ++d;
    }
    return sign | static_cast<uint32_t>(d);
}

FORCE_INLINE uint32_t fkey(uint32_t b) {  // total order of non-NaN words, -0 == +0
    if ((b & 0x7FFFFFFFu) == 0) {
        b = 0;
    }
    return (b & 0x80000000u) ? ~b : (b | 0x80000000u);
}
FORCE_INLINE bool fle(uint32_t a, uint32_t b) { return !is_nan(a) && !is_nan(b) && fkey(a) <= fkey(b); }
FORCE_INLINE bool fge(uint32_t a, uint32_t b) { return fle(b, a); }
FORCE_INLINE bool fgt(uint32_t a, uint32_t b) { return !is_nan(a) && !is_nan(b) && fkey(a) > fkey(b); }

FORCE_INLINE int32_t bf16_key(uint32_t h) {  // order of a bf16 value: -0 == +0, NaN above +inf (torch.sort's rank)
    h &= 0xFFFFu;
    if ((h & 0x7FFFu) == 0) {
        h = 0;
    }
    if ((h & 0x7F80u) == 0x7F80u && (h & 0x7Fu)) {
        return 0x10000;
    }
    return static_cast<int32_t>((h & 0x8000u) ? (h ^ 0xFFFFu) : (h | 0x8000u));
}

FORCE_INLINE uint64_t f2fix(uint32_t b, int32_t frac) {  // floor(value * 2^frac), value >= 0 finite
    const uint32_t ex = (b >> 23) & 0xFFu;
    if (ex == 0) {
        return 0;
    }
    const uint64_t m = (b & 0x7FFFFFu) | 0x800000u;
    const int32_t sh = static_cast<int32_t>(ex) - 150 + frac;
    if (sh >= 0) {
        return m << sh;
    }
    return sh <= -24 ? 0 : (m >> (-sh));
}

FORCE_INLINE uint32_t fix2f(uint64_t v, int32_t frac) {  // nearest float32 (ties to even) of v * 2^-frac
    if (v == 0) {
        return 0;
    }
    const uint32_t hi = static_cast<uint32_t>(v >> 32);
    const int32_t nb = hi ? 64 - __builtin_clz(hi) : 32 - __builtin_clz(static_cast<uint32_t>(v));
    int32_t e = nb - 1 - frac;
    uint32_t q;
    if (nb > 24) {
        const int32_t sh = nb - 24;
        q = static_cast<uint32_t>(v >> sh);
        const uint64_t rem = v & ((1ull << sh) - 1), half = 1ull << (sh - 1);
        if (rem > half || (rem == half && (q & 1u))) {
            ++q;
        }
        if (q == (1u << 24)) {
            q = 1u << 23;
            ++e;
        }
    } else {
        q = static_cast<uint32_t>(v << (24 - nb));
    }
    return (static_cast<uint32_t>(e + 127) << 23) | (q & 0x7FFFFFu);
}

FORCE_INLINE uint32_t f2int_trunc(uint32_t b) {  // toward zero; negatives / NaN -> 0; capped at 2^30 (the top_k word)
    if ((b & 0x80000000u) || is_nan(b)) {
        return 0;
    }
    const uint32_t ex = (b >> 23) & 0xFFu;
    if (ex < 127u) {
        return 0;
    }
    if (ex >= 127u + 30u) {
        return 1u << 30;
    }
    const uint32_t m = (b & 0x7FFFFFu) | 0x800000u;
    const int32_t sh = static_cast<int32_t>(ex) - 150;
    return sh >= 0 ? (m << sh) : (m >> (-sh));
}

// (a * b) >> S for 32 < S < 64, the 128-bit product kept exact (every use keeps it below 2^(64 + S))
template <uint32_t S>
FORCE_INLINE uint64_t mulhi(uint64_t a, uint64_t b) {
    const uint32_t a0 = static_cast<uint32_t>(a), a1 = static_cast<uint32_t>(a >> 32);
    const uint32_t b0 = static_cast<uint32_t>(b), b1 = static_cast<uint32_t>(b >> 32);
    const uint64_t p00 = static_cast<uint64_t>(a0) * b0, p01 = static_cast<uint64_t>(a0) * b1;
    const uint64_t p10 = static_cast<uint64_t>(a1) * b0, p11 = static_cast<uint64_t>(a1) * b1;
    const uint64_t mid = (p00 >> 32) + static_cast<uint32_t>(p01) + static_cast<uint32_t>(p10);
    const uint64_t lo = (mid << 32) | static_cast<uint32_t>(p00);
    const uint64_t hi = p11 + (p01 >> 32) + (p10 >> 32) + (mid >> 32);
    return (hi << (64 - S)) | (lo >> S);
}

uint32_t exp_bits(uint32_t xb) {  // the model's exp_f32
    const uint32_t sgn = xb >> 31, ex = (xb >> 23) & 0xFFu, mx = xb & 0x7FFFFFu;
    if (ex == 0xFFu) {
        return mx ? F_NAN : (sgn ? 0u : F_INF);
    }
    if (ex == 0) {
        return F_ONE;  // +-0 and denormals
    }
    if (ex >= 127u + 7u) {
        return sgn ? 0u : F_INF;  // |x| >= 128
    }
    const uint64_t m = mx | 0x800000u;
    const int32_t sh = static_cast<int32_t>(ex) - 127 + 33;  // |x| * 2^56 = m * 2^sh
    const uint64_t X = sh >= 0 ? (m << sh) : (sh <= -24 ? 0 : (m >> (-sh)));
    if (X == 0) {
        return F_ONE;
    }
    const uint64_t Y = mulhi<62>(X, L2E);  // |x| * log2(e) * 2^56
    constexpr uint64_t FM = (1ull << 56) - 1;
    int32_t n;
    uint64_t f;
    if (sgn) {
        const uint64_t ip = Y >> 56, fr = Y & FM;
        if (fr == 0) {
            n = -static_cast<int32_t>(ip);
            f = 0;
        } else {
            n = -static_cast<int32_t>(ip) - 1;
            f = (1ull << 56) - fr;
        }
    } else {
        n = static_cast<int32_t>(Y >> 56);
        f = Y & FM;
    }
    if (n < -126) {
        return 0;
    }
    if (n > 127) {
        return F_INF;
    }
    const uint32_t i = static_cast<uint32_t>(f >> 50);
    const uint64_t g = f & ((1ull << 50) - 1);
    const uint64_t t = mulhi<58>(g, LN2_64);  // g * ln2, Q0.62
    uint64_t acc = POLY6;
    acc = POLY5 + mulhi<62>(t, acc);
    acc = POLY4 + mulhi<62>(t, acc);
    acc = POLY3 + mulhi<62>(t, acc);
    acc = POLY2 + mulhi<62>(t, acc);
    acc = POLY1 + mulhi<62>(t, acc);
    acc = POLY0 + mulhi<62>(t, acc);
    const uint64_t mant = mulhi<62>(EXP2_TAB[i], acc);  // 2^f in [2^62, 2^63)
    uint32_t q = static_cast<uint32_t>(mant >> 39);
    const uint64_t rem = mant & ((1ull << 39) - 1), half = 1ull << 38;
    if (rem > half || (rem == half && (q & 1u))) {
        ++q;
    }
    if (q == (1u << 24)) {
        q = 1u << 23;
        ++n;
        if (n > 127) {
            return F_INF;
        }
    }
    return (static_cast<uint32_t>(n + 127) << 23) | (q & 0x7FFFFFu);
}

FORCE_INLINE uint32_t u32_sat(int32_t v) { return v < 0 ? 0u : static_cast<uint32_t>(v); }

FORCE_INLINE int32_t rope_row(int32_t x, int32_t delta, int32_t rows_m1) {  // min(relu(x + delta), rope_rows - 1)
    int32_t v = x + delta;
    v = v < 0 ? 0 : v;
    return v < rows_m1 ? v : rows_m1;
}

// ------------------------------------------------------------------------------------------------ the union row
// Column c of this row: tile c / 32, face half (c / 16) % 2, lane c % 16; ODD selects the bf16 face row in the 64 B read.
FORCE_INLINE uint32_t uval(uint32_t base, uint32_t c, uint32_t odd) {
    const uint32_t slot = 2 * (c >> 5) + ((c >> 4) & 1u);
    return reinterpret_cast<hptr>(base + OFF_UV + slot * 64 + odd * 32)[c & 15u];
}
FORCE_INLINE uint32_t uid(uint32_t base, uint32_t c) {
    const uint32_t slot = 2 * (c >> 5) + ((c >> 4) & 1u);
    return reinterpret_cast<wptr>(base + OFF_UI + slot * 64)[c & 15u];
}

// The TP-way merge of the shard blocks: the next candidate of (value desc, union position asc). Exhausted blocks key -1.
// The heads and their keys live in the caller's (register / local-memory) arrays.
FORCE_INLINE void merge_init(uint32_t base, uint32_t odd, uint32_t* head, int32_t* key) {
    for (uint32_t d = 0; d < TP; ++d) {
        head[d] = 0;
        key[d] = bf16_key(uval(base, d * M, odd));
    }
}
FORCE_INLINE void merge_next(uint32_t base, uint32_t odd, uint32_t* head, int32_t* key, uint32_t& hw, uint32_t& gid) {
    int32_t best = -1;
    uint32_t bd = 0;
    for (uint32_t d = 0; d < TP; ++d) {
        const int32_t k = key[d];
        if (k > best) {
            best = k;
            bd = d;
        }
    }
    const uint32_t h = head[bd];
    const uint32_t c = bd * M + h;
    head[bd] = h + 1;
    key[bd] = h + 1 < M ? bf16_key(uval(base, c + 1, odd)) : -1;
    hw = uval(base, c, odd);
    gid = uid(base, c) + bd * PER_SHARD;
}

// One sampled row (the model's row_state + row_accept + row_pick + row_pick_gid). DRAFT is the row's draft id when
// CORR (j < nv): its probability is the accept test and the pick removes it; otherwise the pick is the bonus draw.
void row_work(
    uint32_t base,
    uint32_t odd,
    uint32_t temp_b,
    uint32_t k,
    uint32_t omp_b,
    bool corr,
    uint32_t draft,
    uint32_t u_acc,
    uint32_t u_pick,
    wptr rec) {
    const wptr gid = reinterpret_cast<wptr>(base + OFF_GID);
    const wptr vb = reinterpret_cast<wptr>(base + OFF_VB);
    const wptr eb = reinterpret_cast<wptr>(base + OFF_EB);
    const wptr pr = reinterpret_cast<wptr>(base + OFF_PR);
    const wptr cdf = reinterpret_cast<wptr>(base + OFF_CDF);
    // 1-2. merge, v = vals / temp, top-k with ties (the merge stops at the first v below the k-th), overflow.
    // Equal bf16 words give equal quotients: a tie reuses the previous division (and below, its exp / p0 / prob).
    uint32_t head[TP];
    int32_t key[TP];
    merge_init(base, odd, head, key);
    const uint64_t r_t = rcp62(temp_b);
    uint32_t n_merged = 0, n_keep = 0, kth = 0, prev_hw = 0, prev_v = 0;
    while (n_keep < M) {
        uint32_t hw, g;
        merge_next(base, odd, head, key, hw, g);
        const uint32_t vi = (n_merged != 0 && hw == prev_hw) ? prev_v : fdiv_r(hw << 16, temp_b, r_t);
        prev_hw = hw;
        prev_v = vi;
        gid[n_merged++] = g;
        if (n_keep >= k && !fge(vi, kth)) {
            break;
        }
        vb[n_keep++] = vi;
        if (n_keep == k) {
            kth = vi;
        }
    }
    const uint32_t ovf = (n_keep == M && fgt(kth, F_NEG_INF)) ? 1u : 0u;
    // 3. e = exp(v - v0), s = sum(e) (exact, rounded once)
    uint64_t es = 0;
    for (uint32_t i = 0; i < n_keep; ++i) {
        const uint32_t e = (i != 0 && vb[i] == vb[i - 1]) ? eb[i - 1] : exp_bits(fsub(vb[i], vb[0]));
        eb[i] = e;
        es += f2fix(e, E_FRAC);
    }
    const uint32_t s = fix2f(es, E_FRAC);
    // 4. top-p: drop the suffix whose reverse cumsum of p0 = e / s is <= 1 - top_p (never column 0)
    uint32_t n2 = n_keep;
    if (fgt(omp_b, 0)) {
        const uint64_t r_s = rcp62(s);
        uint64_t acc = 0;
        uint32_t pe = 0, pp = 0;  // the previous (from the end) e and its p0
        for (uint32_t i = n_keep - 1; i >= 1; --i) {
            const uint32_t e = eb[i];
            pp = (i + 1 < n_keep && e == pe) ? pp : fdiv_r(e, s, r_s);
            pe = e;
            acc += f2fix(pp, P_FRAC);
            if (!fle(fix2f(acc, P_FRAC), omp_b)) {
                break;
            }
            n2 = i;
        }
    }
    // 5. prob = e / s2 over the kept prefix; the draft's probability and the accept test
    uint32_t s2 = s;
    if (n2 != n_keep) {
        uint64_t a2 = 0;
        for (uint32_t i = 0; i < n2; ++i) {
            a2 += f2fix(eb[i], E_FRAC);
        }
        s2 = fix2f(a2, E_FRAC);
    }
    uint32_t p_d = 0;
    bool first = true;
    const uint64_t r_s2 = rcp62(s2);
    for (uint32_t i = 0; i < n2; ++i) {
        const uint32_t p = (i != 0 && eb[i] == eb[i - 1]) ? pr[i - 1] : fdiv_r(eb[i], s2, r_s2);
        pr[i] = p;
        if (corr && gid[i] == draft) {
            p_d = first ? p : fadd(p_d, p);
            first = false;
        }
    }
    const uint32_t accept = (corr && fgt(p_d, 0) && fge(p_d, u_acc)) ? 1u : 0u;
    // 6. the pick: inverse CDF of the weights (draft removed on a correction row), clamped to the last positive weight
    uint64_t ac = 0;
    uint32_t last = M - 1;
    bool found = false;
    for (uint32_t i = 0; i < n2; ++i) {
        const uint32_t w = (corr && gid[i] == draft) ? 0u : pr[i];
        ac += f2fix(w, P_FRAC);
        cdf[i] = fix2f(ac, P_FRAC);
        if (w) {
            last = i;
            found = true;
        }
    }
    if (!found) {
        last = M - 1;
    }
    const uint32_t total = cdf[n2 - 1];
    const uint32_t thr = fmul(u_pick, total);
    uint32_t pick = 0;
    for (uint32_t i = 0; i < n2; ++i) {
        pick += fle(cdf[i], thr) ? 1u : 0u;
    }
    if (fle(total, thr)) {
        pick += M - n2;
    }
    if (pick > last) {
        pick = last;
    }
    while (n_merged <= pick) {  // a degenerate pick past the kept candidates (all weight removed): merge on to it
        uint32_t hw, g;
        merge_next(base, odd, head, key, hw, g);
        gid[n_merged++] = g;
    }
    rec[0] = accept;
    rec[1] = gid[pick];
    rec[2] = ovf;
    rec[3] = 0;
}

// ------------------------------------------------------------------------------------------------ posts
// Write `bytes` at local `off` to the same offset on core (x, y), ack, then count it there (spec_argmax_fused's post).
FORCE_INLINE void post(uint32_t base, uint32_t off, uint32_t bytes, uint32_t x, uint32_t y, uint32_t sem_id) {
    noc_async_write(base + off, get_noc_addr(x, y, base + off), bytes);
    noc_async_write_barrier();
    noc_semaphore_inc(get_noc_addr(x, y, get_semaphore(sem_id)), 1);
    noc_async_atomic_barrier();
}

FORCE_INLINE void wait_posts(uint32_t sem_id, uint32_t n) {
    volatile tt_l1_ptr uint32_t* sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_id));
    noc_semaphore_wait(sem, n);
    asm volatile("" ::: "memory");
}

FORCE_INLINE uint32_t noc_x(uint32_t c) { return get_common_arg_val<uint32_t>(ARG_NOC + 2 * c); }
FORCE_INLINE uint32_t noc_y(uint32_t c) { return get_common_arg_val<uint32_t>(ARG_NOC + 2 * c + 1); }

}  // namespace

void kernel_main() {
    const uint32_t core = static_cast<uint32_t>(get_absolute_logical_y()) * NX + get_absolute_logical_x();
    if (core >= R) {
        return;
    }
    const uint32_t r = core, u = r / T, j = r % T, r0 = u * T;
    const bool leader = j == 0;

    constexpr auto ids_args = TensorAccessorArgs<CTA_ACCESSORS>();
    constexpr auto tok_args = TensorAccessorArgs<ids_args.next_compile_time_args_offset()>();
    constexpr auto aux_args = TensorAccessorArgs<tok_args.next_compile_time_args_offset()>();
    constexpr auto samp_args = TensorAccessorArgs<aux_args.next_compile_time_args_offset()>();
    constexpr auto uv_args = TensorAccessorArgs<samp_args.next_compile_time_args_offset()>();
    constexpr auto ui_args = TensorAccessorArgs<uv_args.next_compile_time_args_offset()>();
    constexpr auto pt_args = TensorAccessorArgs<ui_args.next_compile_time_args_offset()>();
    constexpr auto pack_args = TensorAccessorArgs<pt_args.next_compile_time_args_offset()>();
    constexpr auto rs_args = TensorAccessorArgs<pack_args.next_compile_time_args_offset()>();
    constexpr auto res_args = TensorAccessorArgs<rs_args.next_compile_time_args_offset()>();
    const auto ids = TensorAccessor(ids_args, get_common_arg_val<uint32_t>(0));
    const auto tok = TensorAccessor(tok_args, get_common_arg_val<uint32_t>(1));
    const auto aux = TensorAccessor(aux_args, get_common_arg_val<uint32_t>(2));
    const auto samp = TensorAccessor(samp_args, get_common_arg_val<uint32_t>(3));
    const auto uv = TensorAccessor(uv_args, get_common_arg_val<uint32_t>(4));
    const auto ui = TensorAccessor(ui_args, get_common_arg_val<uint32_t>(5));
    const auto pt = TensorAccessor(pt_args, get_common_arg_val<uint32_t>(6));
    const auto pack = TensorAccessor(pack_args, get_common_arg_val<uint32_t>(7));
    const auto rs = TensorAccessor(rs_args, get_common_arg_val<uint32_t>(8));
    const auto res = TensorAccessor(res_args, get_common_arg_val<uint32_t>(9));

    const uint32_t base = al64(get_write_ptr(CB_L1));

    // ---- A. inputs: small rows, this row's union face rows (64 B aligned: the bf16 read holds face rows fr & ~1, | 1)
    noc_async_read(tok.get_noc_addr(0), base + OFF_TOK, R * 4);
    noc_async_read(ids.get_noc_addr(0), base + OFF_IDS, 128);
    for (uint32_t i = 0; i < AUX_ROWS; ++i) {
        noc_async_read(aux.get_noc_addr(i), base + OFF_AUX + i * ROW, W * 4);
    }
    for (uint32_t i = 0; i < SAMP_PARAM_ROWS; ++i) {
        noc_async_read(samp.get_noc_addr(i), base + OFF_SAMP + i * ROW, W * 4);
    }
    noc_async_read(samp.get_noc_addr(SAMP_PARAM_ROWS + u), base + OFF_SAMP + SAMP_PARAM_ROWS * ROW, W * 4);
    const uint32_t rb = r >> 4, fr = r & 15u, odd = fr & 1u;
    for (uint32_t t = 0; t < NT; ++t) {
        for (uint32_t h = 0; h < 2; ++h) {
            const uint32_t face = 2 * rb + h, slot = 2 * t + h;
            noc_async_read(uv.get_noc_addr(t, face * 512 + (fr & ~1u) * 32), base + OFF_UV + slot * 64, 64);
            noc_async_read(ui.get_noc_addr(t, face * 1024 + fr * 64), base + OFF_UI + slot * 64, 64);
        }
    }
    noc_async_read_barrier();
    asm volatile("" ::: "memory");
    // the first piece of this row's pt row flies while the row is computed
    noc_async_read(pt.get_noc_addr(r, 0), base + OFF_RS, ROW_BYTES < RS_CHUNK ? ROW_BYTES : RS_CHUNK);

    const wptr tok_w = reinterpret_cast<wptr>(base + OFF_TOK);
    const wptr ids_w = reinterpret_cast<wptr>(base + OFF_IDS);
    auto arow = [&](uint32_t i) { return reinterpret_cast<iptr>(base + OFF_AUX + i * ROW); };
    auto srow = [&](uint32_t i) { return reinterpret_cast<wptr>(base + OFF_SAMP + i * ROW); };
    const int32_t nv_raw = arow(AUX_NUM_VALID)[u];
    const uint32_t nv = nv_raw <= 0 ? 0u : (nv_raw >= static_cast<int32_t>(K) ? K : static_cast<uint32_t>(nv_raw));
    const bool greedy = (srow(SAMP_GREEDY)[u] & 0x7FFFFFFFu) != 0 || arow(AUX_ACTIVE)[u] == 0;
    const int32_t scratch = arow(AUX_CONSTS)[CONST_SCRATCH];
    {
        const wptr scr = reinterpret_cast<wptr>(base + OFF_SCR);
        for (uint32_t i = 0; i < SCR_BYTES / 4; ++i) {
            scr[i] = static_cast<uint32_t>(scratch);
        }
    }

    const wptr my_rec = reinterpret_cast<wptr>(base + OFF_ROWREC + j * 16);
    if (!greedy && j <= nv) {
        const uint32_t t_raw = srow(SAMP_TEMP)[u];
        const bool use_one = !is_nan(t_raw) && ((t_raw & 0x7FFFFFFFu) == 0 || (t_raw & 0x80000000u));
        const uint32_t temp_b = use_one ? F_ONE : t_raw;
        uint32_t k = f2int_trunc(srow(SAMP_TOPK)[u]);
        k = k < 1 ? 1 : (k > M - 1 ? M - 1 : k);
        const uint32_t omp_b = fsub(F_ONE, srow(SAMP_TOPP)[u]);
        const wptr uni = srow(SAMP_PARAM_ROWS);
        const bool corr = j < nv;
        row_work(
            base,
            odd,
            temp_b,
            k,
            omp_b,
            corr,
            corr ? tok_w[r + 1] : 0u,
            corr ? uni[1 + j] : 0u,
            corr ? uni[K + 1] : uni[0],
            my_rec);
    } else {
        my_rec[0] = my_rec[1] = my_rec[2] = my_rec[3] = 0;
    }
    asm volatile("" ::: "memory");
    if (!leader) {
        post(base, OFF_ROWREC + j * 16, 16, noc_x(r0), noc_y(r0), SEM_ROW);
    }

    // ---- B. leader: the walk of user u
    const wptr back = reinterpret_cast<wptr>(base + OFF_BACK);
    uint32_t real_u = 0;
    if (leader) {
        if constexpr (T > 1) {
            wait_posts(SEM_ROW, T - 1);
        }
        uint32_t m = 0, tok_m, ovf = 0;
        if (greedy) {
            while (m < K && static_cast<int32_t>(m) < nv_raw && ids_w[r0 + m] == tok_w[r0 + m + 1]) {
                ++m;
            }
            tok_m = ids_w[r0 + m];
        } else {
            auto rrec = [&](uint32_t jj) { return reinterpret_cast<wptr>(base + OFF_ROWREC + jj * 16); };
            while (m < nv && rrec(m)[0]) {
                ++m;
            }
            tok_m = rrec(m)[1];
            for (uint32_t jj = 0; jj <= m; ++jj) {
                ovf |= rrec(jj)[2];
            }
        }
        const int32_t kv = static_cast<int32_t>(m) + arow(AUX_DECLINED)[u] * arow(AUX_KEEP)[u];
        real_u = kv <= 0 ? 0u : (kv >= static_cast<int32_t>(T) ? T : static_cast<uint32_t>(kv));
        back[0] = real_u;
        back[1] = back[2] = back[3] = 0;
        const wptr urec = reinterpret_cast<wptr>(base + OFF_USER + u * 16);
        urec[0] = m;
        urec[1] = tok_m;
        urec[2] = ovf | (greedy ? 2u : 0u);
        urec[3] = real_u;
        asm volatile("" ::: "memory");
        if constexpr (T > 1) {  // the rows first (they wait to write rs_pt), all writes in flight, one ack, the counts
            for (uint32_t jj = 1; jj < T; ++jj) {
                noc_async_write(base + OFF_BACK, get_noc_addr(noc_x(r0 + jj), noc_y(r0 + jj), base + OFF_BACK), 16);
            }
            noc_async_write_barrier();
            for (uint32_t jj = 1; jj < T; ++jj) {
                noc_semaphore_inc(get_noc_addr(noc_x(r0 + jj), noc_y(r0 + jj), get_semaphore(SEM_BACK)), 1);
            }
            noc_async_atomic_barrier();
        }
        if (u != 0) {
            post(base, OFF_USER + u * 16, 16, noc_x(0), noc_y(0), SEM_USER);
        }
    } else {
        wait_posts(SEM_BACK, 1);
        real_u = back[0];
    }

    // ---- C. rs_pt row r
    noc_async_read_barrier();  // the prefetched pt piece
    if (j < real_u) {
        for (uint32_t p = 0; p < RS_PIECES; ++p) {
            const uint32_t lo = p * RS_CHUNK, n = lo + RS_CHUNK <= ROW_BYTES ? RS_CHUNK : ROW_BYTES - lo;
            if (p != 0) {
                noc_async_writes_flushed();
                noc_async_read(pt.get_noc_addr(r, lo), base + OFF_RS, n);
                noc_async_read_barrier();
            }
            noc_async_write(base + OFF_RS, rs.get_noc_addr(r, lo), n);
        }
    } else {
        for (uint32_t lo = 0; lo < ROW_BYTES; lo += SCR_BYTES) {
            const uint32_t n = lo + SCR_BYTES <= ROW_BYTES ? SCR_BYTES : ROW_BYTES - lo;
            noc_async_write(base + OFF_SCR, rs.get_noc_addr(r, lo), n);
        }
    }

    // ---- D. core 0: pack [2K+5, W] and result [3, W] from the B user records
    if (core == 0) {
        if constexpr (B > 1) {
            wait_posts(SEM_USER, B - 1);
        }
        const wptr pk = reinterpret_cast<wptr>(base + OFF_PACK);
        const iptr rr = reinterpret_cast<iptr>(base + OFF_RES);
        constexpr uint32_t S = ROW / 4;  // words per L1 row
        for (uint32_t i = 0; i < PACK_ROWS * S; ++i) {
            pk[i] = 0;
        }
        for (uint32_t i = 0; i < 3 * S; ++i) {
            rr[i] = 0;
        }
        const iptr dec_r = arow(AUX_DECLINED);
        const iptr p0_r = arow(AUX_P0);
        const iptr delta_r = arow(AUX_DELTA);
        const int32_t rows_m1 = arow(AUX_CONSTS)[CONST_ROPE_ROWS] - 1;
        constexpr uint32_t G2 = K + 1, G3 = K + 3, G4 = K + 4, G5 = K + 5;
        for (uint32_t uu = 0; uu < B; ++uu) {
            const wptr urec = reinterpret_cast<wptr>(base + OFF_USER + uu * 16);
            const uint32_t m = urec[0], tok_m = urec[1], flags = urec[2], real = urec[3];
            const bool g = (flags & 2u) != 0;
            const uint32_t q0 = uu * T;
            const int32_t ndec = 1 - dec_r[uu], p0 = p0_r[uu], delta = delta_r[uu];
            const int32_t p = (p0 + static_cast<int32_t>(m)) * ndec;
            pk[uu] = u32_sat(static_cast<int32_t>(tok_m) * ndec);
            for (uint32_t kk = 0; kk < K; ++kk) {
                const int32_t step = p + static_cast<int32_t>(kk);
                pk[(1 + kk) * S + uu] = u32_sat(step);
                pk[(G5 + kk) * S + uu] = u32_sat(rope_row(step, delta, rows_m1));
            }
            pk[G3 * S + uu] = q0 + m;
            for (uint32_t jj = 0; jj < T; ++jj) {
                const uint32_t rq = q0 + jj;
                const int32_t cm = jj < m ? static_cast<int32_t>(g ? ids_w[rq] : tok_w[rq + 1])
                                          : (jj == m ? static_cast<int32_t>(tok_m) : -1);
                const int32_t rl = jj < real ? 1 : 0;
                const int32_t rpos = (p0 + static_cast<int32_t>(jj)) * rl;
                pk[G2 * S + rq] = u32_sat(cm * rl);
                pk[(G2 + 1) * S + rq] = u32_sat(rpos);
                pk[G4 * S + rq] = u32_sat(rope_row(rpos, delta, rows_m1));
                rr[rq] = cm;
            }
            rr[S + uu] = static_cast<int32_t>(m + 1);
            rr[2 * S + uu] = static_cast<int32_t>(flags & 1u);
        }
        asm volatile("" ::: "memory");
        for (uint32_t i = 0; i < PACK_ROWS; ++i) {
            noc_async_write(base + OFF_PACK + i * ROW, pack.get_noc_addr(i), W * 4);
        }
        for (uint32_t i = 0; i < 3; ++i) {
            noc_async_write(base + OFF_RES + i * ROW, res.get_noc_addr(i), W * 4);
        }
    }
    noc_async_write_barrier();
}
