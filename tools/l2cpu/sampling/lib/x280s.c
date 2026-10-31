// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-FileCopyrightText: © 2026 Kansei Motoe (kinginu)
//
// SPDX-License-Identifier: Apache-2.0

/*
 * x280s: portable, bit-exact token sampling. See x280s.h and SPEC_NOTES.md.
 *
 * Rules that keep the x86-64 and rv64gcv builds bit-identical:
 *   - only IEEE binary32 +, -, *, / and comparisons, integer ops, and fp32<->int conversions of
 *     exactly representable values; no double, no long double, no libm, no FMA
 *     (all builds use -ffp-contract=off -fno-fast-math);
 *   - every floating-point sum is a sequential fp32 running sum in a fixed order;
 *   - the rounding mode is forced to round-to-nearest-even inside every entry point
 *     (and FTZ/DAZ cleared on x86-64), so the caller's FP state cannot change results;
 *   - no global mutable state, no allocation, no libc (no implicit memcpy/memset either:
 *     the firmware flavour adds -fno-tree-loop-distribute-patterns, and `nm` is checked).
 */
#include "x280s.h"

#if defined(X280S_RVV) && !defined(__riscv_vector)
#error "X280S_RVV requires a target with the V extension (-march=rv64gcv)"
#endif
#ifdef X280S_RVV
#include <riscv_vector.h>
#endif

#define X280S_NOINLINE __attribute__((noinline))

#define X280S_INLINE static inline __attribute__((always_inline))

/* ------------------------------------------------------------------------------------------ */
/* bit helpers                                                                                 */

typedef union {
    uint32_t u;
    float f;
} x280s_fu_t;

X280S_INLINE float f_from_bits(uint32_t u) {
    x280s_fu_t x;
    x.u = u;
    return x.f;
}

#define F_NEG_INF f_from_bits(0xff800000u)
#define F_POS_INF f_from_bits(0x7f800000u)
#define F_MAX f_from_bits(0x7f7fffffu)

/* Load element i as fp32 (exact for both dtypes). */
X280S_INLINE float load_logit(const void* base, uint32_t dtype, uint32_t stride, uint32_t i) {
    uint64_t off = (uint64_t)i * stride;
    if (dtype == X280S_DTYPE_F32) {
        return ((const float*)base)[off];
    }
    return f_from_bits((uint32_t)((const uint16_t*)base)[off] << 16);
}

/* ------------------------------------------------------------------------------------------ */
/* floating-point environment                                                                  */

#if defined(__x86_64__)
typedef uint32_t fpenv_t;
X280S_INLINE fpenv_t fpenv_enter(void) {
    uint32_t old, def = 0x1f80u; /* all exceptions masked, RNE, FTZ=0, DAZ=0 */
    __asm__ volatile("stmxcsr %0" : "=m"(old)::"memory");
    __asm__ volatile("ldmxcsr %0" ::"m"(def) : "memory");
    return old;
}
X280S_INLINE void fpenv_leave(fpenv_t old) { __asm__ volatile("ldmxcsr %0" ::"m"(old) : "memory"); }
#elif defined(__riscv) && defined(__riscv_flen)
typedef unsigned long fpenv_t;
X280S_INLINE fpenv_t fpenv_enter(void) {
    unsigned long old;
    __asm__ volatile("csrrw %0, frm, zero" : "=r"(old)::"memory"); /* frm = RNE */
    return old;
}
X280S_INLINE void fpenv_leave(fpenv_t old) { __asm__ volatile("csrw frm, %0" ::"r"(old) : "memory"); }
#else
typedef int fpenv_t;
X280S_INLINE fpenv_t fpenv_enter(void) { return 0; }
X280S_INLINE void fpenv_leave(fpenv_t old) { (void)old; }
#endif

/* ------------------------------------------------------------------------------------------ */
/* expf                                                                                        */
/*
 * x280s_expf, fp32-only and deterministic:
 *   1. NaN -> canonical qNaN 0x7fc00000 (the same bits on both ISAs), x > 88.75 -> +inf (covers +inf), x < -104 -> +0
 * (covers -inf). exp(88.75) and exp(-104) are beyond the fp32 overflow / underflow-to-zero limits, the remaining
 * overflow/underflow happens inside the final multiplications below.
 *   2. n = round_to_nearest_even(x * log2(e)) with the 1.5 * 2^23 shifter (exact for |t| < 2^22).
 *   3. r = (x - n * LN2_HI) - n * LN2_LO (Cody-Waite; LN2_HI has 12 trailing zero bits, so
 *      n * LN2_HI is exact for |n| <= 150), |r| <= ~0.347.
 *   4. exp(r) ~= 1 + r + r^2 * q(r), q = degree-5 Taylor tail (1/2 .. 1/5040) in Horner form.
 *   5. result = p * 2^n, built from exponent bits; for n < -126 as (p * 2^(n+100)) * 2^-100 and for
 *      n > 127 as (p * 2^127) * 2^(n-127), so exactly one rounding happens in the last multiply
 *      (gradual underflow is IEEE on both targets; FTZ/DAZ are cleared on x86-64).
 * Measured accuracy (exhaustive over all fp32 x in [-104, 88.75], see notes/sampling.md).
 */
float x280s_expf(float x) {
    if (x != x) {
        return f_from_bits(0x7fc00000u); /* canonical qNaN: x86 propagates payloads, RISC-V does not */
    }
    if (x > 88.75f) {
        return F_POS_INF;
    }
    if (x < -104.0f) {
        return 0.0f;
    }

    const float shifter = 12582912.0f;              /* 1.5 * 2^23 */
    const float log2e = 1.44269502162933349609375f; /* RN(log2(e)) */
    const float ln2_hi = 0.693145751953125f;        /* 0x3f317200 */
    const float ln2_lo = 1.428606765330187045e-06f; /* RN(ln2 - ln2_hi) */

    float t = x * log2e;
    t = t + shifter;
    float n = t - shifter;
    int32_t ni = (int32_t)n;

    float r = x - n * ln2_hi;
    r = r - n * ln2_lo;

    float q = 1.0f / 5040.0f;
    q = q * r + 1.0f / 720.0f;
    q = q * r + 1.0f / 120.0f;
    q = q * r + 1.0f / 24.0f;
    q = q * r + 1.0f / 6.0f;
    q = q * r + 0.5f;
    float r2 = r * r;
    float p = r2 * q;
    p = p + r;
    p = p + 1.0f;

    if (ni < -126) {
        float s = f_from_bits((uint32_t)(ni + 100 + 127) << 23);
        float y = p * s;
        return y * f_from_bits((uint32_t)(-100 + 127) << 23);
    }
    if (ni > 127) {
        float y = p * f_from_bits((uint32_t)(127 + 127) << 23);
        return y * f_from_bits((uint32_t)(ni - 127 + 127) << 23);
    }
    return p * f_from_bits((uint32_t)(ni + 127) << 23);
}

uint64_t x280s_splitmix64(uint64_t x) {
    uint64_t z = x + 0x9E3779B97F4A7C15ull;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

uint32_t x280s_abi_version(void) { return X280S_ABI_VERSION; }
uint32_t x280s_sizeof_work(void) { return (uint32_t)sizeof(x280s_work_t); }
uint32_t x280s_sizeof_stats(void) { return (uint32_t)sizeof(x280s_stats_t); }
uint32_t x280s_sizeof_params(void) { return (uint32_t)sizeof(x280s_params_t); }
uint32_t x280s_has_rvv(void) {
#ifdef X280S_RVV
    return 1u;
#else
    return 0u;
#endif
}

static void stats_clear(x280s_stats_t* s) {
    volatile uint8_t* b = (volatile uint8_t*)s; /* volatile: never turned into a memset call */
    for (uint32_t i = 0; i < (uint32_t)sizeof(*s); i++) {
        b[i] = 0;
    }
}

/* ------------------------------------------------------------------------------------------ */
/* argmax                                                                                      */

typedef struct {
    float best;
    uint32_t idx;
    uint32_t nan;
} argmax_res_t;

/* Scalar reference: first index of the maximum, NaN -> -inf (counted). */
X280S_INLINE argmax_res_t argmax_scalar(const void* base, uint32_t dtype, uint32_t V, uint32_t stride) {
    argmax_res_t r;
    r.best = F_NEG_INF;
    r.idx = 0;
    r.nan = 0;
    for (uint32_t i = 0; i < V; i++) {
        float v = load_logit(base, dtype, stride, i);
        if (v != v) {
            r.nan++;
            continue;
        }
        if (v > r.best) {
            r.best = v;
            r.idx = i;
        }
    }
    return r;
}

#ifdef X280S_RVV
/*
 * RVV_VL_NOTE: the chunk length is computed as min(remaining, VLMAX) in scalar code instead of using
 * the return value of __riscv_vsetvl_*: riscv64-unknown-elf-gcc 13.2 miscompiled
 * `vl = __riscv_vsetvl_e32m8(V - i); ... i += vl;` in select_scan_rvv into `i += V - i` (the AVL),
 * so only the first chunk was scanned (caught by the QEMU corpus: wrong NaN counts).
 */

/*
 * RVV argmax (exact): per chunk, NaN count via vmfne(v, v); NaN lanes replaced by -inf; chunk max by
 * vfredmax (max is exact and order-independent once NaN is gone); the first lane equal to the chunk
 * max via vmfeq + vfirst; the running best is only replaced on a strictly greater chunk max, so the
 * lowest index wins ties. Contiguous rows (stride 1) only; other strides use the scalar loop.
 */
static argmax_res_t argmax_rvv(const void* base, uint32_t dtype, uint32_t V) {
    argmax_res_t r;
    r.best = F_NEG_INF;
    r.idx = 0;
    r.nan = 0;
    const float ninf = F_NEG_INF;
    const size_t vlmax = __riscv_vsetvlmax_e32m8();
    for (uint32_t i = 0; i < V;) {
        size_t vl = (V - i < vlmax) ? (size_t)(V - i) : vlmax; /* see RVV_VL_NOTE */
        vfloat32m8_t v;
        if (dtype == X280S_DTYPE_F32) {
            v = __riscv_vle32_v_f32m8((const float*)base + i, vl);
        } else {
            vuint16m4_t h = __riscv_vle16_v_u16m4((const uint16_t*)base + i, vl);
            vuint32m8_t w = __riscv_vsll_vx_u32m8(__riscv_vzext_vf2_u32m8(h, vl), 16, vl);
            v = __riscv_vreinterpret_v_u32m8_f32m8(w);
        }
        vbool4_t isnan = __riscv_vmfne_vv_f32m8_b4(v, v, vl);
        r.nan += (uint32_t)__riscv_vcpop_m_b4(isnan, vl);
        v = __riscv_vfmerge_vfm_f32m8(v, ninf, isnan, vl);
        vfloat32m1_t init = __riscv_vfmv_s_f_f32m1(ninf, 1);
        float m = __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredmax_vs_f32m8_f32m1(v, init, vl));
        if (m > r.best) {
            vbool4_t eq = __riscv_vmfeq_vf_f32m8_b4(v, m, vl);
            long first = __riscv_vfirst_m_b4(eq, vl);
            r.idx = i + (uint32_t)first;
            r.best = load_logit(base, dtype, 1, r.idx); /* the element itself (sign of zero as scalar) */
        }
        i += (uint32_t)vl;
    }
    return r;
}
#endif

#if defined(X280S_RVV) && !defined(X280S_RVV_NOFAST)
#define X280S_FAST 1
/*
 * bf16 fast paths. All exact; every condition they cannot handle bit-exactly falls back to the
 * paths above, so the results (token AND every stats byte) are unchanged.
 *
 * Integer key of a bf16 pattern h: key = h ^ ((h >>s 15) | 0x8000). For non-NaN values the key order is
 * the float order, except that -0 (key 0x7fff) sorts below +0 (key 0x8000). key(-inf) = 0x007f,
 * key(+inf) = 0xff80; positive NaNs have keys > 0xff80, negative NaNs keys < 0x007f.
 * One e16/m8 pass (256 lanes) gives the per-lane max and min key: 5 vector ops per 256 elements.
 */
#define KEY_NINF 0x007fu
#define KEY_PINF 0xff80u
#define KEY_NZERO 0x7fffu
#define KEY_PZERO 0x8000u
#define FAST_MIN_V 1024u

X280S_INLINE vuint16m8_t bf16_key(vuint16m8_t h, size_t vl) {
    vuint16m8_t s =
        __riscv_vreinterpret_v_i16m8_u16m8(__riscv_vsra_vx_i16m8(__riscv_vreinterpret_v_u16m8_i16m8(h), 15, vl));
    return __riscv_vxor_vv_u16m8(h, __riscv_vor_vx_u16m8(s, 0x8000u, vl), vl);
}

X280S_INLINE uint16_t key_of(uint16_t h) { return (uint16_t)(h ^ ((h & 0x8000u) ? 0xffffu : 0x8000u)); }
X280S_INLINE uint16_t bf16_of_key(uint16_t k) { return (uint16_t)((k & 0x8000u) ? (k ^ 0x8000u) : (k ^ 0xffffu)); }

/* Pass A: max / min key over [0, V). With lanes != 0 the row is split into nseg segments (each >= one
 * vector) and every segment's per-lane maxima are stored: nseg * vlmax keys of DISTINCT elements. */
static void key_range(const uint16_t* p, uint32_t V, uint16_t* kmax, uint16_t* kmin, uint16_t* lanes, uint32_t nseg) {
    const size_t vlmax = __riscv_vsetvlmax_e16m8();
    vuint16m8_t gmax = __riscv_vmv_v_x_u16m8(0, vlmax);
    vuint16m8_t amin = __riscv_vmv_v_x_u16m8(0xffffu, vlmax);
    if (!lanes) {
        nseg = 1;
    }
    uint32_t seglen = (uint32_t)((V / nseg) / vlmax * vlmax);
    uint32_t i = 0;
    for (uint32_t sg = 0; sg < nseg; sg++) {
        uint32_t end = (sg + 1 == nseg) ? V : i + seglen;
        vuint16m8_t amax = __riscv_vmv_v_x_u16m8(0, vlmax);
        while (i < end) {
            size_t vl = (end - i < vlmax) ? (size_t)(end - i) : vlmax; /* see RVV_VL_NOTE */
            vuint16m8_t k = bf16_key(__riscv_vle16_v_u16m8(p + i, vl), vl);
            amax = __riscv_vmaxu_vv_u16m8_tu(amax, amax, k, vl);
            amin = __riscv_vminu_vv_u16m8_tu(amin, amin, k, vl);
            i += (uint32_t)vl;
        }
        if (lanes) {
            __riscv_vse16_v_u16m8(lanes + (size_t)sg * vlmax, amax, vlmax);
        }
        gmax = __riscv_vmaxu_vv_u16m8(gmax, amax, vlmax);
    }
    vuint16m1_t z = __riscv_vmv_s_x_u16m1(0, 1);
    vuint16m1_t f = __riscv_vmv_s_x_u16m1(0xffffu, 1);
    *kmax = (uint16_t)__riscv_vmv_x_s_u16m1_u16(__riscv_vredmaxu_vs_u16m8_u16m1(gmax, z, vlmax));
    *kmin = (uint16_t)__riscv_vmv_x_s_u16m1_u16(__riscv_vredminu_vs_u16m8_u16m1(amin, f, vlmax));
}

/* Number of keys >= t in a[0 .. nblk * vlmax). */
static uint32_t count_ge(const uint16_t* a, uint32_t nblk, uint16_t t) {
    const size_t vlmax = __riscv_vsetvlmax_e16m8();
    vbool2_t acc = __riscv_vmsgeu_vx_u16m8_b2(__riscv_vle16_v_u16m8(a, vlmax), t, vlmax);
    uint32_t n = (uint32_t)__riscv_vcpop_m_b2(acc, vlmax);
    for (uint32_t b = 1; b < nblk; b++) {
        n += (uint32_t)__riscv_vcpop_m_b2(
            __riscv_vmsgeu_vx_u16m8_b2(__riscv_vle16_v_u16m8(a + (size_t)b * vlmax, vlmax), t, vlmax), vlmax);
    }
    return n;
}

/* First index whose key equals k (k must occur). */
static uint32_t key_first(const uint16_t* p, uint32_t V, uint16_t k) {
    const size_t vlmax = __riscv_vsetvlmax_e16m8();
    for (uint32_t i = 0; i < V;) {
        size_t vl = (V - i < vlmax) ? (size_t)(V - i) : vlmax; /* see RVV_VL_NOTE */
        vuint16m8_t key = bf16_key(__riscv_vle16_v_u16m8(p + i, vl), vl);
        long f = __riscv_vfirst_m_b2(__riscv_vmseq_vx_u16m8_b2(key, k, vl), vl);
        if (f >= 0) {
            return i + (uint32_t)f;
        }
        i += (uint32_t)vl;
    }
    return 0;
}

/* Greedy on a contiguous bf16 row. Returns 0 (not handled: NaN present, or the maximum is -inf or a
 * zero, where -0/+0 key order differs from float order) or 1 with *r filled exactly as argmax_rvv. */
static int argmax_bf16_fast(const uint16_t* p, uint32_t V, argmax_res_t* r) {
    uint16_t kmax, kmin;
    if (V < FAST_MIN_V) {
        return 0;
    }
    key_range(p, V, &kmax, &kmin, 0, 1);
    if (kmin < KEY_NINF || kmax > KEY_PINF) {
        return 0; /* NaN present */
    }
    if (kmax <= KEY_NINF || kmax == KEY_NZERO || kmax == KEY_PZERO) {
        return 0; /* -inf or zero max */
    }
    r->idx = key_first(p, V, kmax);
    r->best = load_logit(p, X280S_DTYPE_BF16, 1, r->idx);
    r->nan = 0;
    return 1;
}

#endif

X280S_INLINE argmax_res_t argmax_any(const void* base, uint32_t dtype, uint32_t V, uint32_t stride) {
#ifdef X280S_FAST
    if (stride == 1 && dtype == X280S_DTYPE_BF16) {
        argmax_res_t r;
        if (argmax_bf16_fast((const uint16_t*)base, V, &r)) {
            return r;
        }
    }
#endif
#ifdef X280S_RVV
    if (stride == 1) {
        return argmax_rvv(base, dtype, V);
    }
#endif
    if (dtype == X280S_DTYPE_F32) {
        return argmax_scalar(base, X280S_DTYPE_F32, V, stride);
    }
    return argmax_scalar(base, X280S_DTYPE_BF16, V, stride);
}

static X280S_NOINLINE int32_t
argmax_impl(const void* logits, uint32_t dtype, uint32_t V, uint32_t stride, x280s_stats_t* st) {
    argmax_res_t r = argmax_any(logits, dtype, V, stride);
    if (st) {
        st->nan_count = r.nan;
        st->greedy = 1;
        st->x_max = r.best;
    }
    return (int32_t)r.idx;
}

static int32_t check_args(const void* logits, uint32_t dtype, uint32_t V, uint32_t stride) {
    if (!logits || V == 0 || stride == 0 || V > 0x7fffffffu) {
        return X280S_ERR_ARG;
    }
    if (dtype != X280S_DTYPE_F32 && dtype != X280S_DTYPE_BF16) {
        return X280S_ERR_DTYPE;
    }
    return 0;
}

int32_t x280s_argmax_row(
    const void* logits, uint32_t dtype, uint32_t vocab, uint32_t stride_elems, x280s_stats_t* stats) {
    if (stats) {
        stats_clear(stats);
    }
    int32_t e = check_args(logits, dtype, vocab, stride_elems);
    if (e) {
        return e;
    }
    fpenv_t env = fpenv_enter();
    int32_t tok = argmax_impl(logits, dtype, vocab, stride_elems, stats);
    fpenv_leave(env);
    return tok;
}

/* ------------------------------------------------------------------------------------------ */
/* top-k selection                                                                             */
/*
 * Total order on candidates ("a ranks above b"): a.value > b.value, or equal values and
 * a.index < b.index. Indices are unique, so the order is strict and the top-k set and the sorted
 * order are fully determined by the specification, independent of the selection algorithm.
 * The workspace holds a min-heap on this order (root = lowest-ranked kept candidate).
 */
X280S_INLINE int ranks_below(x280s_cand_t a, x280s_cand_t b) {
    return a.value < b.value || (a.value == b.value && a.index > b.index);
}

static void sift_down(x280s_cand_t* h, uint32_t pos, uint32_t n) {
    x280s_cand_t x = h[pos];
    for (;;) {
        uint32_t c = 2 * pos + 1;
        if (c >= n) {
            break;
        }
        if (c + 1 < n && ranks_below(h[c + 1], h[c])) {
            c++;
        }
        if (!ranks_below(h[c], x)) {
            break;
        }
        h[pos] = h[c];
        pos = c;
    }
    h[pos] = x;
}

/* Raw (unscaled) value with NaN -> -inf; division by a finite T > 0 is monotone non-decreasing. */
X280S_INLINE float load_clean(const void* base, uint32_t dtype, uint32_t stride, uint32_t i) {
    float v = load_logit(base, dtype, stride, i);
    return (v != v) ? F_NEG_INF : v;
}

/*
 * Scalar reference selection. Elements K..V-1 arrive in increasing index order, so a new element
 * ranks above the root iff its scaled value is strictly greater. Prefilter (exact): if raw v <= raw
 * root then v / T <= root / T, so the element cannot rank above the root; the division is only
 * done for survivors.
 */
X280S_INLINE uint32_t select_scan(
    const void* base,
    uint32_t dtype,
    uint32_t stride,
    uint32_t V,
    uint32_t K,
    float T,
    x280s_cand_t* h,
    uint32_t start,
    float root_raw) {
    uint32_t nan = 0;
    for (uint32_t i = start; i < V; i++) {
        float v = load_logit(base, dtype, stride, i);
        if (v != v) {
            nan++;
            continue;
        }
        if (v <= root_raw) {
            continue;
        }
        float s = v / T;
        if (s > h[0].value) {
            h[0].value = s;
            h[0].index = i;
            sift_down(h, 0, K);
            root_raw = load_clean(base, dtype, stride, h[0].index);
        }
    }
    return nan;
}

#ifdef X280S_RVV
/*
 * RVV prefilter (exact): scans contiguous chunks, counts NaN lanes, and only drops into the scalar
 * update for a chunk that holds a lane with raw value > root_raw (NaN compares false). The scalar
 * update re-walks that chunk from the first such lane in index order with the same compare, so the
 * result is identical to select_scan.
 */
static uint32_t select_scan_rvv(
    const void* base,
    uint32_t dtype,
    uint32_t V,
    uint32_t K,
    float T,
    x280s_cand_t* h,
    uint32_t start,
    float root_raw) {
    uint32_t nan = 0;
    const size_t vlmax = __riscv_vsetvlmax_e32m8();
    for (uint32_t i = start; i < V;) {
        size_t vl = (V - i < vlmax) ? (size_t)(V - i) : vlmax; /* see RVV_VL_NOTE */
        vfloat32m8_t v;
        if (dtype == X280S_DTYPE_F32) {
            v = __riscv_vle32_v_f32m8((const float*)base + i, vl);
        } else {
            vuint16m4_t hv = __riscv_vle16_v_u16m4((const uint16_t*)base + i, vl);
            vuint32m8_t w = __riscv_vsll_vx_u32m8(__riscv_vzext_vf2_u32m8(hv, vl), 16, vl);
            v = __riscv_vreinterpret_v_u32m8_f32m8(w);
        }
        nan += (uint32_t)__riscv_vcpop_m_b4(__riscv_vmfne_vv_f32m8_b4(v, v, vl), vl);
        long first = __riscv_vfirst_m_b4(__riscv_vmfgt_vf_f32m8_b4(v, root_raw, vl), vl);
        if (first >= 0) {
            for (uint32_t j = i + (uint32_t)first; j < i + (uint32_t)vl; j++) {
                float x = load_logit(base, dtype, 1, j);
                if (!(x > root_raw)) {
                    continue; /* NaN and v <= root_raw */
                }
                float s = x / T;
                if (s > h[0].value) {
                    h[0].value = s;
                    h[0].index = j;
                    sift_down(h, 0, K);
                    root_raw = load_clean(base, dtype, 1, h[0].index);
                }
            }
        }
        i += (uint32_t)vl;
    }
    return nan;
}
#endif

#ifdef X280S_FAST
/*
 * Top-K from segment-lane maxima.
 *   - The row is split into nseg = X280S_PREP_LANES / vlmax segments; key_range (or the fused copy) records
 *     each segment's per-lane maximum: X280S_PREP_LANES keys of DISTINCT elements ("groups" (s, l) = the
 *     elements of segment s in lane l, stride vlmax).
 *   - theta = the K-th largest group maximum (binary search with vector counts); vstar = theta stepped down while
 *     value / T == theta / T. Every element below vstar has a scaled value strictly below >= K elements, so it is
 *     not in the top-K whatever the index tie-break.
 *   - Candidates = elements with key >= kcut = key(vstar). They can only sit in groups whose maximum >= kcut, so
 *     only those groups are walked (strided, scalar). The walk is not in index order, so a candidate replaces the
 *     heap root iff it ranks above it in the full order (value, then lower index): the heap ends with the top-K
 *     of the candidates = the top-K of the row, and the heapsort gives the identical sorted order.
 */
#ifndef X280S_WALK_MAX
#define X280S_WALK_MAX 1024u /* elements; tuned on the chip */
#endif
static uint32_t lanes_nseg(void) {
    uint32_t vlmax = (uint32_t)__riscv_vsetvlmax_e16m8();
    return (vlmax <= X280S_PREP_LANES / 4) ? X280S_PREP_LANES / vlmax : 0;
}

X280S_INLINE int lanes_usable(uint32_t V) { return V >= 2u * X280S_PREP_LANES && lanes_nseg() != 0; }

X280S_INLINE int keys_clean(uint16_t kmax, uint16_t kmin) { return !(kmin < KEY_NINF || kmax > KEY_PINF); }

static int cut_from_lanes(
    const uint16_t* lanes, uint32_t nseg, uint16_t kmax, uint32_t K, float T, float* vstar_out, uint16_t* kcut_out) {
    uint32_t lo = KEY_NINF, hi = (uint32_t)kmax + 1u; /* count(>= lo) >= K > count(>= hi) */
    while (hi - lo > 1u) {
        uint32_t mid = (lo + hi) / 2u;
        if (count_ge(lanes, nseg, (uint16_t)mid) >= K) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    uint16_t kth = (uint16_t)lo;
    if (kth <= KEY_NINF) {
        return 0;
    }
    float c = f_from_bits((uint32_t)bf16_of_key(kth) << 16) / T;
    uint16_t k = kth;
    for (uint32_t it = 0;; it++) {
        if (k <= KEY_NINF + 1u || it > 64) {
            return 0;
        }
        float pv = f_from_bits((uint32_t)bf16_of_key((uint16_t)(k - 1u)) << 16);
        if (!(pv / T >= c)) {
            break;
        }
        k--;
    }
    *vstar_out = f_from_bits((uint32_t)bf16_of_key(k) << 16);
    *kcut_out = k;
    return 1;
}

static void topk_walk(
    const uint16_t* p,
    uint32_t V,
    uint32_t K,
    float T,
    x280s_cand_t* h,
    const uint16_t* lanes,
    uint32_t nseg,
    uint16_t kcut,
    float vstar) {
    const uint32_t vlmax = (uint32_t)__riscv_vsetvlmax_e16m8();
    const uint32_t seglen = (V / nseg) / vlmax * vlmax;
    uint32_t n = 0;
    for (uint32_t sg = 0; sg < nseg; sg++) {
        uint32_t S = sg * seglen, E = (sg + 1 == nseg) ? V : S + seglen;
        for (uint32_t l = 0; l < vlmax; l++) {
            if (lanes[sg * vlmax + l] < kcut) {
                continue;
            }
            for (uint32_t j = S + l; j < E; j += vlmax) {
                uint16_t hb = p[j];
                if (key_of(hb) < kcut) {
                    continue;
                }
                float x = f_from_bits((uint32_t)hb << 16);
                if (!(x >= vstar)) {
                    continue;
                }
                x280s_cand_t c;
                c.value = x / T;
                c.index = j;
                if (n < K) {
                    h[n] = c;
                    if (++n == K) {
                        for (uint32_t r = K / 2; r-- > 0;) {
                            sift_down(h, r, K);
                        }
                    }
                } else if (ranks_below(h[0], c)) {
                    h[0] = c;
                    sift_down(h, 0, K);
                }
            }
        }
    }
}

/*
 * Top-K over the candidates (key >= kcut, value >= vstar) in index order: the first K candidates fill the
 * heap (then heapify), later ones replace the root exactly as select_scan does. The result is the top-K of
 * the candidate set, which equals the top-K of the row (topk_cut). Candidate lanes are compacted with
 * vcompress; one scalar check covers 4 chunks.
 */
static void select_scan_cut(
    const uint16_t* p, uint32_t V, uint32_t K, float T, x280s_cand_t* h, uint16_t kcut, float vstar, uint16_t* idxbuf) {
    const size_t vlmax = __riscv_vsetvlmax_e16m8();
    vuint16m8_t lane = __riscv_vid_v_u16m8(vlmax);
    uint32_t n = 0;
    float root_raw = 0.0f;
    for (uint32_t g = 0; g < V; g += 4 * (uint32_t)vlmax) {
        /* group test: any candidate in the next 4 chunks? */
        uint32_t gend = (V - g < 4 * vlmax) ? V : g + 4 * (uint32_t)vlmax;
        vuint16m8_t gm = __riscv_vmv_v_x_u16m8(0, vlmax);
        for (uint32_t i = g; i < gend;) {
            size_t vl = (gend - i < vlmax) ? (size_t)(gend - i) : vlmax; /* see RVV_VL_NOTE */
            gm = __riscv_vmaxu_vv_u16m8_tu(gm, gm, bf16_key(__riscv_vle16_v_u16m8(p + i, vl), vl), vl);
            i += (uint32_t)vl;
        }
        if (__riscv_vfirst_m_b2(__riscv_vmsgeu_vx_u16m8_b2(gm, kcut, vlmax), vlmax) < 0) {
            continue;
        }
        for (uint32_t i = g; i < gend;) {
            size_t vl = (gend - i < vlmax) ? (size_t)(gend - i) : vlmax; /* see RVV_VL_NOTE */
            vuint16m8_t key = bf16_key(__riscv_vle16_v_u16m8(p + i, vl), vl);
            vbool2_t m = __riscv_vmsgeu_vx_u16m8_b2(key, kcut, vl);
            uint32_t c = (uint32_t)__riscv_vcpop_m_b2(m, vl);
            if (c) {
                __riscv_vse16_v_u16m8(idxbuf, __riscv_vcompress_vm_u16m8(lane, m, vl), c);
                for (uint32_t q = 0; q < c; q++) {
                    uint32_t j = i + idxbuf[q];
                    float x = load_logit(p, X280S_DTYPE_BF16, 1, j);
                    if (!(x >= vstar)) {
                        continue;
                    }
                    if (n < K) {
                        h[n].value = x / T;
                        h[n].index = j;
                        if (++n == K) {
                            for (uint32_t r = K / 2; r-- > 0;) {
                                sift_down(h, r, K);
                            }
                            root_raw = load_clean(p, X280S_DTYPE_BF16, 1, h[0].index);
                        }
                        continue;
                    }
                    if (!(x > root_raw)) {
                        continue;
                    }
                    float sc = x / T;
                    if (sc > h[0].value) {
                        h[0].value = sc;
                        h[0].index = j;
                        sift_down(h, 0, K);
                        root_raw = load_clean(p, X280S_DTYPE_BF16, 1, h[0].index);
                    }
                }
            }
            i += (uint32_t)vl;
        }
    }
}

/* Greedy from the group maxima: the lowest index with key == kmax (kmax not a zero / -inf, no NaN). */
static uint32_t greedy_from_lanes(const uint16_t* p, uint32_t V, const uint16_t* lanes, uint32_t nseg, uint16_t kmax) {
    const uint32_t vlmax = (uint32_t)__riscv_vsetvlmax_e16m8();
    const uint32_t seglen = (V / nseg) / vlmax * vlmax;
    uint32_t best = 0xffffffffu;
    for (uint32_t sg = 0; sg < nseg; sg++) {
        uint32_t S = sg * seglen, E = (sg + 1 == nseg) ? V : S + seglen;
        if (S > best) {
            break;
        }
        for (uint32_t l = 0; l < vlmax; l++) {
            if (lanes[sg * vlmax + l] != kmax) {
                continue;
            }
            for (uint32_t j = S + l; j < E && j < best; j += vlmax) {
                if (key_of(p[j]) == kmax) {
                    best = j;
                    break;
                }
            }
        }
    }
    return best;
}

/* Fused copy + segment-lane statistics (one pass over src). */
static void copy_prepare_bf16(uint16_t* dst, const uint16_t* src, uint32_t V, x280s_prep_t* prep) {
    const size_t vlmax = __riscv_vsetvlmax_e16m8();
    uint32_t nseg = lanes_nseg();
    uint32_t seglen = (uint32_t)((V / nseg) / vlmax * vlmax);
    vuint16m8_t gmax = __riscv_vmv_v_x_u16m8(0, vlmax);
    vuint16m8_t amin = __riscv_vmv_v_x_u16m8(0xffffu, vlmax);
    uint32_t i = 0;
    for (uint32_t sg = 0; sg < nseg; sg++) {
        uint32_t end = (sg + 1 == nseg) ? V : i + seglen;
        vuint16m8_t amax = __riscv_vmv_v_x_u16m8(0, vlmax);
        while (i < end) {
            size_t vl = (end - i < vlmax) ? (size_t)(end - i) : vlmax; /* see RVV_VL_NOTE */
            vuint16m8_t h = __riscv_vle16_v_u16m8(src + i, vl);
            __riscv_vse16_v_u16m8(dst + i, h, vl);
            vuint16m8_t k = bf16_key(h, vl);
            amax = __riscv_vmaxu_vv_u16m8_tu(amax, amax, k, vl);
            amin = __riscv_vminu_vv_u16m8_tu(amin, amin, k, vl);
            i += (uint32_t)vl;
        }
        __riscv_vse16_v_u16m8(prep->lanes + (size_t)sg * vlmax, amax, vlmax);
        gmax = __riscv_vmaxu_vv_u16m8(gmax, amax, vlmax);
    }
    vuint16m1_t z = __riscv_vmv_s_x_u16m1(0, 1);
    vuint16m1_t f = __riscv_vmv_s_x_u16m1(0xffffu, 1);
    prep->kmax = (uint16_t)__riscv_vmv_x_s_u16m1_u16(__riscv_vredmaxu_vs_u16m8_u16m1(gmax, z, vlmax));
    prep->kmin = (uint16_t)__riscv_vmv_x_s_u16m1_u16(__riscv_vredminu_vs_u16m8_u16m1(amin, f, vlmax));
    prep->nseg = nseg;
    prep->vocab = V;
    prep->valid = 1;
}
#endif

static X280S_NOINLINE int32_t sample_impl(
    const void* base,
    uint32_t dtype,
    uint32_t V,
    uint32_t stride,
    const x280s_params_t* p,
    uint32_t user,
    uint64_t step,
    x280s_work_t* work,
    x280s_stats_t* st,
    const x280s_prep_t* prep) {
    float T = p->temperature;
    if (T > F_MAX) {
        T = F_MAX; /* +inf temperature: keeps every scaled value well defined (no inf/inf) */
    }

    uint32_t K = p->top_k;
    uint32_t cap_req = 0;
    if (K == 0 || K > X280S_K_MAX) {
        K = X280S_K_MAX;
        cap_req = 1;
    }
    if (K > V) {
        K = V;
    }

    x280s_cand_t* h = work->cand;
    uint32_t nan = 0;
#ifdef X280S_FAST
    float vstar;
    uint16_t kcut, kmax = 0, kmin = 0;
    const uint16_t* lanes = 0;
    uint32_t nseg = 0;
    if (stride == 1 && dtype == X280S_DTYPE_BF16 && K < V && lanes_usable(V)) {
        if (prep && prep->valid && prep->vocab == V) {
            lanes = prep->lanes, nseg = prep->nseg, kmax = prep->kmax, kmin = prep->kmin;
        } else {
            /* the weight array (free until step 3) holds the group maxima */
            uint16_t* w = (uint16_t*)(void*)work->weight;
            nseg = lanes_nseg();
            key_range((const uint16_t*)base, V, &kmax, &kmin, w, nseg);
            lanes = w;
        }
    }
    (void)prep;
    if (lanes && keys_clean(kmax, kmin) && cut_from_lanes(lanes, nseg, kmax, K, T, &vstar, &kcut)) {
        /* strided group walks touch one cache line per element: worth it only for a few groups; otherwise one
         * contiguous vector pass (index order) over the row */
        uint32_t groups = count_ge(lanes, nseg, kcut);
        uint32_t per_group = (V / nseg) / (uint32_t)__riscv_vsetvlmax_e16m8() + 1u;
        if (groups * per_group <= X280S_WALK_MAX) {
            topk_walk((const uint16_t*)base, V, K, T, h, lanes, nseg, kcut, vstar);
        } else {
            uint16_t idxbuf[X280S_PREP_LANES / 4]; /* >= vlmax(e16, m8) for VLEN <= 1024 */
            select_scan_cut((const uint16_t*)base, V, K, T, h, kcut, vstar, idxbuf);
        }
    } else
#else
    (void)prep;
#endif
    {
        /* 1. fill the heap with elements 0..K-1, heapify, then scan the rest. */
        for (uint32_t i = 0; i < K; i++) {
            float v = load_logit(base, dtype, stride, i);
            if (v != v) {
                nan++;
                v = F_NEG_INF;
            }
            h[i].value = v / T;
            h[i].index = i;
        }
        for (uint32_t i = K / 2; i-- > 0;) {
            sift_down(h, i, K);
        }
        float root_raw = load_clean(base, dtype, stride, h[0].index);
#ifdef X280S_RVV
        if (stride == 1) {
            nan += select_scan_rvv(base, dtype, V, K, T, h, K, root_raw);
        } else
#endif
            if (dtype == X280S_DTYPE_F32) {
            nan += select_scan(base, X280S_DTYPE_F32, stride, V, K, T, h, K, root_raw);
        } else {
            nan += select_scan(base, X280S_DTYPE_BF16, stride, V, K, T, h, K, root_raw);
        }
    }
    /* 2. heapsort the min-heap: repeatedly move the lowest-ranked to the end -> descending rank. */
#ifdef X280S_FAST
    /* bottom-up (Floyd) extraction: the hole left by the root walks down along the lower-ranked children to a
     * leaf, then the displaced last element sifts up. Same multiset and strict total order, so the sorted
     * result is identical to the sift_down version, with about half the comparisons. */
    for (uint32_t end = K; end-- > 1;) {
        x280s_cand_t top = h[0];
        x280s_cand_t x = h[end];
        uint32_t pos = 0;
        for (;;) {
            uint32_t c = 2 * pos + 1;
            if (c >= end) {
                break;
            }
            if (c + 1 < end && ranks_below(h[c + 1], h[c])) {
                c++;
            }
            h[pos] = h[c];
            pos = c;
        }
        while (pos > 0) {
            uint32_t par = (pos - 1) / 2;
            if (!ranks_below(x, h[par])) {
                break;
            }
            h[pos] = h[par];
            pos = par;
        }
        h[pos] = x;
        h[end] = top;
    }
#else
    for (uint32_t end = K; end-- > 1;) {
        x280s_cand_t t = h[0];
        h[0] = h[end];
        h[end] = t;
        sift_down(h, 0, end);
    }
#endif

    /* 3. weights, sequential sum S in sorted order. */
    float* w = work->weight;
    float x_max = h[0].value;
    float S = 0.0f;
    for (uint32_t j = 0; j < K; j++) {
        float x = h[j].value;
        float wj = (x == x_max) ? 1.0f : x280s_expf(x - x_max);
        w[j] = wj;
        S = S + wj;
    }

    /* 4. top-p: shortest prefix whose running sum reaches top_p * S. */
    float tp = p->top_p;
    if (!(tp < 1.0f)) {
        tp = 1.0f; /* > 1 and NaN behave as 1 */
    }
    float thr = tp * S;
    uint32_t n_kept = K;
    float run = 0.0f;
    for (uint32_t j = 0; j < K; j++) {
        run = run + w[j];
        if (run >= thr) {
            n_kept = j + 1;
            break;
        }
    }
    float kept_sum = run;

    /* 5. draw. */
    uint64_t r = x280s_splitmix64(p->seed ^ ((uint64_t)user << 32) ^ step);
    float u = (float)(uint32_t)(r >> 40) * 0x1p-24f;
    float target = u * kept_sum;
    uint32_t pick = n_kept - 1;
    uint32_t fallback = 1;
    run = 0.0f;
    for (uint32_t j = 0; j < n_kept; j++) {
        run = run + w[j];
        if (run > target) {
            pick = j;
            fallback = 0;
            break;
        }
    }

    if (st) {
        st->nan_count = nan;
        st->cap_applied = (cap_req && V > X280S_K_MAX) ? 1u : 0u;
        st->k_eff = K;
        st->n_kept = n_kept;
        st->pick = pick;
        st->fallback = fallback;
        st->x_max = x_max;
        st->S = S;
        st->threshold = thr;
        st->kept_sum = kept_sum;
        st->u = u;
        st->target = target;
        st->r = r;
    }
    return (int32_t)h[pick].index;
}

int32_t x280s_sample_row(
    const void* logits,
    uint32_t dtype,
    uint32_t vocab,
    uint32_t stride_elems,
    const x280s_params_t* p,
    uint32_t user,
    uint64_t step,
    x280s_work_t* work,
    x280s_stats_t* stats) {
    if (stats) {
        stats_clear(stats);
    }
    if (!p) {
        return X280S_ERR_ARG;
    }
    int32_t e = check_args(logits, dtype, vocab, stride_elems);
    if (e) {
        return e;
    }
    fpenv_t env = fpenv_enter();
    int32_t tok;
    if (!(p->temperature > 0.0f)) { /* 0, negative, NaN -> greedy */
        tok = argmax_impl(logits, dtype, vocab, stride_elems, stats);
    } else if (!work) {
        tok = X280S_ERR_ARG;
    } else {
        tok = sample_impl(logits, dtype, vocab, stride_elems, p, user, step, work, stats, 0);
    }
    fpenv_leave(env);
    return tok;
}

void x280s_copy_prepare(void* dst, const void* src, uint32_t dtype, uint32_t vocab, x280s_prep_t* prep) {
    prep->valid = 0;
#ifdef X280S_FAST
    if (dtype == X280S_DTYPE_BF16 && lanes_usable(vocab)) {
        copy_prepare_bf16((uint16_t*)dst, (const uint16_t*)src, vocab, prep);
        return;
    }
#endif
    {
        uint64_t n = (uint64_t)vocab * (dtype == X280S_DTYPE_BF16 ? 2u : 4u);
        volatile uint8_t* d = (volatile uint8_t*)dst; /* volatile: never turned into a memcpy call */
        const uint8_t* s8 = (const uint8_t*)src;
        if (((((uintptr_t)dst) | ((uintptr_t)src)) & 7u) == 0) {
            volatile uint64_t* d64 = (volatile uint64_t*)dst;
            const uint64_t* s64 = (const uint64_t*)src;
            uint64_t w = n / 8;
            for (uint64_t i = 0; i < w; i++) {
                d64[i] = s64[i];
            }
            for (uint64_t i = w * 8; i < n; i++) {
                d[i] = s8[i];
            }
        } else {
            for (uint64_t i = 0; i < n; i++) {
                d[i] = s8[i];
            }
        }
    }
}

int32_t x280s_sample_row_prep(
    const void* logits,
    uint32_t dtype,
    uint32_t vocab,
    const x280s_params_t* p,
    uint32_t user,
    uint64_t step,
    x280s_work_t* work,
    const x280s_prep_t* prep,
    x280s_stats_t* stats) {
    if (stats) {
        stats_clear(stats);
    }
    if (!p) {
        return X280S_ERR_ARG;
    }
    int32_t e = check_args(logits, dtype, vocab, 1);
    if (e) {
        return e;
    }
    fpenv_t env = fpenv_enter();
    int32_t tok;
    if (!(p->temperature > 0.0f)) {
#ifdef X280S_FAST
        if (prep && prep->valid && prep->vocab == vocab && dtype == X280S_DTYPE_BF16 &&
            keys_clean(prep->kmax, prep->kmin) && prep->kmax > KEY_NINF && prep->kmax != KEY_NZERO &&
            prep->kmax != KEY_PZERO) {
            uint32_t idx = greedy_from_lanes((const uint16_t*)logits, vocab, prep->lanes, prep->nseg, prep->kmax);
            if (stats) {
                stats->nan_count = 0;
                stats->greedy = 1;
                stats->x_max = load_logit(logits, X280S_DTYPE_BF16, 1, idx);
            }
            tok = (int32_t)idx;
        } else
#endif
            tok = argmax_impl(logits, dtype, vocab, 1, stats);
    } else if (!work) {
        tok = X280S_ERR_ARG;
    } else {
        tok = sample_impl(logits, dtype, vocab, 1, p, user, step, work, stats, prep);
    }
    fpenv_leave(env);
    return tok;
}

/* ---- incremental copy + statistics (streaming) ---- */
int x280s_copy_prepare_begin(void* dst, uint32_t dtype, uint32_t vocab, x280s_prep_t* prep) {
    (void)dst;
    prep->valid = 0;
    prep->done = 0;
    prep->vocab = vocab;
#ifdef X280S_FAST
    if (dtype == X280S_DTYPE_BF16 && lanes_usable(vocab) && __riscv_vsetvlmax_e16m8() <= 512u &&
        X280S_PREP_ALIGN % __riscv_vsetvlmax_e16m8() == 0) {
        uint32_t nseg = lanes_nseg();
        for (uint32_t i = 0; i < nseg * (uint32_t)__riscv_vsetvlmax_e16m8(); i++) {
            prep->lanes[i] = 0;
        }
        for (uint32_t i = 0; i < 512u; i++) {
            prep->lmin[i] = 0xffffu;
        }
        prep->nseg = nseg;
        return 1;
    }
#else
    (void)dtype;
#endif
    return 0;
}

void x280s_copy_prepare_range(void* dst, const void* src, uint32_t to, x280s_prep_t* prep) {
#ifdef X280S_FAST
    const size_t vlmax = __riscv_vsetvlmax_e16m8();
    const uint32_t V = prep->vocab, nseg = prep->nseg;
    const uint32_t seglen = (uint32_t)((V / nseg) / vlmax * vlmax);
    uint16_t* d = (uint16_t*)dst;
    const uint16_t* sp = (const uint16_t*)src;
    uint32_t i = prep->done;
    if (to > V) {
        to = V;
    }
    vuint16m8_t amin = __riscv_vle16_v_u16m8(prep->lmin, vlmax);
    while (i < to) {
        uint32_t sg = seglen ? i / seglen : 0;
        if (sg >= nseg) {
            sg = nseg - 1;
        }
        uint32_t send = (sg + 1 == nseg) ? V : (sg + 1) * seglen;
        uint32_t end = to < send ? to : send;
        uint16_t* acc = prep->lanes + (size_t)sg * vlmax;
        vuint16m8_t amax = __riscv_vle16_v_u16m8(acc, vlmax);
        while (i < end) {
            size_t vl = (end - i < vlmax) ? (size_t)(end - i) : vlmax; /* see RVV_VL_NOTE */
            vuint16m8_t h = __riscv_vle16_v_u16m8(sp + i, vl);
            __riscv_vse16_v_u16m8(d + i, h, vl);
            vuint16m8_t k = bf16_key(h, vl);
            amax = __riscv_vmaxu_vv_u16m8_tu(amax, amax, k, vl);
            amin = __riscv_vminu_vv_u16m8_tu(amin, amin, k, vl);
            i += (uint32_t)vl;
        }
        __riscv_vse16_v_u16m8(acc, amax, vlmax);
    }
    __riscv_vse16_v_u16m8(prep->lmin, amin, vlmax);
    prep->done = i;
#else
    (void)dst;
    (void)src;
    (void)to;
    (void)prep;
#endif
}

void x280s_copy_prepare_end(x280s_prep_t* prep) {
#ifdef X280S_FAST
    const size_t vlmax = __riscv_vsetvlmax_e16m8();
    uint16_t kmax = 0, kmin = 0xffffu;
    for (uint32_t i = 0; i < prep->nseg * (uint32_t)vlmax; i++) {
        if (prep->lanes[i] > kmax) {
            kmax = prep->lanes[i];
        }
    }
    for (uint32_t i = 0; i < (uint32_t)vlmax; i++) {
        if (prep->lmin[i] < kmin) {
            kmin = prep->lmin[i];
        }
    }
    prep->kmax = kmax;
    prep->kmin = kmin;
    prep->valid = (prep->done == prep->vocab);
#else
    (void)prep;
#endif
}
