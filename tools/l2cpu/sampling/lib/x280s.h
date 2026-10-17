// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-FileCopyrightText: © 2026 Kansei Motoe (kinginu)
//
// SPDX-License-Identifier: Apache-2.0

/*
 * x280s: portable, bit-exact token sampling for the x280 on-chip sampling project.
 *
 * The same source is built twice:
 *   - bare-metal rv64gcv firmware (freestanding, no libc/libm, no malloc, M-mode), and
 *   - an x86-64 shared library loaded through ctypes as the bit-exact reference.
 * Both builds use -O2 -ffp-contract=off -fno-fast-math. See SPEC_NOTES.md for the exact
 * semantics, including every corner case decided.
 *
 * Reentrancy: no global mutable state. Each caller (hart) passes its own workspace and stats.
 * Floating-point environment: the functions save the FP control state on entry, force
 * round-to-nearest-even (and on x86-64 also clear FTZ/DAZ), and restore it on exit.
 * The firmware must have enabled the FPU (mstatus.FS != 0) before calling.
 */
#ifndef X280S_H
#define X280S_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define X280S_ABI_VERSION 1u

/* Maximum number of candidates kept by top-k (also the cap when top_k == 0). */
#define X280S_K_MAX 1024u

/* Logit element types. */
#define X280S_DTYPE_F32 0u  /* IEEE binary32, little-endian */
#define X280S_DTYPE_BF16 1u /* bfloat16 = upper 16 bits of the binary32 pattern, little-endian u16 */

/* Negative return codes. Token ids are always >= 0. */
#define X280S_ERR_ARG (-1)   /* NULL pointer, vocab == 0 or stride == 0 */
#define X280S_ERR_DTYPE (-2) /* unknown dtype */

typedef struct {
    float temperature; /* > 0 samples; <= 0 or NaN selects greedy (see SPEC_NOTES.md) */
    uint32_t top_k;    /* 0 = no top-k (capped at X280S_K_MAX); values > X280S_K_MAX are capped too */
    float top_p;       /* (0, 1]; <= 0 keeps one candidate; > 1 or NaN behaves as 1 */
    uint32_t _pad;     /* keeps the layout identical on every ABI; must be ignored */
    uint64_t seed;
} x280s_params_t;

typedef struct {
    float value; /* scaled logit (logit / temperature), NaN replaced by -inf */
    uint32_t index;
} x280s_cand_t;

/* Caller-owned scratch, one per concurrent caller. 12 KiB. */
typedef struct {
    x280s_cand_t cand[X280S_K_MAX];
    float weight[X280S_K_MAX];
} x280s_work_t;

/* Per-call statistics and the intermediate fp32 values (for bit-exactness checks). */
typedef struct {
    uint32_t nan_count;   /* NaN logits seen in [0, vocab) (they were treated as -inf) */
    uint32_t cap_applied; /* 1 when the K_MAX cap removed candidates (top_k == 0 or > K_MAX, and vocab > K_MAX) */
    uint32_t greedy;      /* 1 when the greedy path was taken */
    uint32_t k_eff;       /* number of top-k candidates (0 on the greedy path) */
    uint32_t n_kept;      /* length of the kept top-p prefix (0 on the greedy path) */
    uint32_t pick;        /* position of the returned token inside the sorted candidates */
    uint32_t fallback;    /* 1 when no running sum exceeded the target (last kept candidate returned) */
    uint32_t _pad;
    float x_max;     /* largest scaled value (greedy: the largest logit) */
    float S;         /* sum of all k_eff weights, sequential in sorted order */
    float threshold; /* top_p * S */
    float kept_sum;  /* sum of the kept prefix */
    float u;         /* uniform draw in [0, 1) */
    float target;    /* u * kept_sum */
    uint64_t r;      /* raw SplitMix64 output */
} x280s_stats_t;

/*
 * Sample one row. Returns the token id in [0, vocab) or a negative error code.
 *   logits        pointer to element 0 of the row
 *   dtype         X280S_DTYPE_F32 or X280S_DTYPE_BF16
 *   vocab         real vocabulary size V; elements at index >= V are never read
 *   stride_elems  distance between consecutive logits, in elements (1 = contiguous)
 *   user, step    mixed into the random draw (see SPEC_NOTES.md)
 *   work          caller-owned workspace (may be NULL only on the greedy path)
 *   stats         may be NULL
 */
int32_t x280s_sample_row(
    const void* logits,
    uint32_t dtype,
    uint32_t vocab,
    uint32_t stride_elems,
    const x280s_params_t* p,
    uint32_t user,
    uint64_t step,
    x280s_work_t* work,
    x280s_stats_t* stats);

/*
 * Fused copy + row statistics: for rows that live in memory that is slow to read (the x280's uncached
 * logits zone), copy the row once into a fast buffer and collect the per-lane key statistics the RVV fast paths
 * need in the same pass; then sample the copy with x280s_sample_row_prep. Results (token and every stats byte)
 * are identical to x280s_sample_row on the same row. Builds without the RVV fast paths just copy (valid = 0).
 */
#define X280S_PREP_LANES 2048u
typedef struct {
    uint32_t valid; /* 1: the fields below describe dst[0 .. vocab) */
    uint32_t vocab;
    uint16_t kmax, kmin;
    uint32_t nseg;
    uint16_t lanes[X280S_PREP_LANES];
    uint32_t done; /* incremental API: elements processed so far */
    uint32_t _pad;
    uint16_t lmin[512]; /* incremental API: per-lane minimum keys */
} x280s_prep_t;

/* Copy logits[0 .. vocab) (contiguous, dtype F32 or BF16) from src to dst and fill prep. */
void x280s_copy_prepare(void* dst, const void* src, uint32_t dtype, uint32_t vocab, x280s_prep_t* prep);
/* Incremental form of x280s_copy_prepare for rows that arrive column-prefix by column-prefix (streaming):
 * begin, then copy ranges [prep->done, to) in increasing order (`to` a multiple of X280S_PREP_ALIGN or = vocab),
 * then end; the result equals x280s_copy_prepare. BF16 only; returns 0 if the incremental form is not available
 * (then copy the whole row with x280s_copy_prepare once it has arrived). */
#define X280S_PREP_ALIGN 512u
int x280s_copy_prepare_begin(void* dst, uint32_t dtype, uint32_t vocab, x280s_prep_t* prep);
void x280s_copy_prepare_range(void* dst, const void* src, uint32_t to, x280s_prep_t* prep);
void x280s_copy_prepare_end(x280s_prep_t* prep);
/* x280s_sample_row(logits, dtype, vocab, 1, ...) using prep (which must come from copying into `logits`). */
int32_t x280s_sample_row_prep(
    const void* logits,
    uint32_t dtype,
    uint32_t vocab,
    const x280s_params_t* p,
    uint32_t user,
    uint64_t step,
    x280s_work_t* work,
    const x280s_prep_t* prep,
    x280s_stats_t* stats);

/* Greedy: index of the maximum (NaN treated as -inf), lowest index on ties. stats may be NULL. */
int32_t x280s_argmax_row(
    const void* logits, uint32_t dtype, uint32_t vocab, uint32_t stride_elems, x280s_stats_t* stats);

/* Building blocks, exported for tests. */
float x280s_expf(float x);
uint64_t x280s_splitmix64(uint64_t x);
uint32_t x280s_abi_version(void);
uint32_t x280s_sizeof_work(void);
uint32_t x280s_sizeof_stats(void);
uint32_t x280s_sizeof_params(void);
/* 1 when the library was built with the RVV fast path. */
uint32_t x280s_has_rvv(void);

#ifdef __cplusplus
}
#endif

#endif /* X280S_H */
