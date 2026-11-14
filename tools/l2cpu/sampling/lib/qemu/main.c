// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-FileCopyrightText: © 2026 Kansei Motoe (kinginu)
//
// SPDX-License-Identifier: Apache-2.0

/*
 * QEMU (virt) bit-exactness test of the x280 sampling library.
 * The corpus (rows, params, expected token + raw x280s_stats_t bytes) is produced by the HOST
 * build (tests/gen_corpus.py). Phase 1: hart 0 runs every case and prints PASS/FAIL per case.
 * Phase 2: an x280s_expf hash over every 61st fp32 bit pattern vs the host hash.
 * Phase 3: all 4 harts run the corpus concurrently (own workspace each) to show reentrancy.
 * Exit through the virt test finisher: 0x5555 pass, (code << 16) | 0x3333 fail.
 */
#include <stdint.h>

#include "tests/expf_hash.h"
#include "x280s.h"

#define UART ((volatile uint8_t*)0x10000000UL)
#define FINISHER ((volatile uint32_t*)0x100000UL)
#define NHARTS 4

extern const uint8_t x280s_corpus[];

static void putc_(char c) { *UART = (uint8_t)c; }
static void puts_(const char* s) {
    while (*s) {
        putc_(*s++);
    }
}
static void puthex(uint64_t v, int digits) {
    for (int i = (digits - 1) * 4; i >= 0; i -= 4) {
        putc_("0123456789abcdef"[(v >> i) & 15]);
    }
}
static void putdec(uint64_t v) {
    char b[24];
    int n = 0;
    do {
        b[n++] = (char)('0' + v % 10);
        v /= 10;
    } while (v);
    while (n) {
        putc_(b[--n]);
    }
}

typedef struct {
    char magic[4];
    uint32_t version, n_rows, n_cases, rows_off, cases_off, expf_step, pad;
    uint64_t expf_hash;
} hdr_t;

typedef struct {
    uint64_t off;
    uint32_t dtype, n_elems;
} row_t;

typedef struct {
    uint32_t row, vocab, stride, mode;
    x280s_params_t params;
    uint32_t user, pad;
    uint64_t step;
    int32_t token;
    uint32_t pad2;
    x280s_stats_t stats;
} case_t;

_Static_assert(sizeof(case_t) == 128, "case layout");
_Static_assert(sizeof(x280s_stats_t) == 64, "stats layout");

static x280s_work_t work[NHARTS];
/* prep path (x280s_copy_prepare + x280s_sample_row_prep): copies of contiguous rows, one per hart */
static x280s_prep_t prep[NHARTS];
static uint8_t copybuf[NHARTS][160000 * 4] __attribute__((aligned(64)));
static x280s_stats_t stats[NHARTS];
static volatile uint32_t go;       /* secondaries wait for a non-zero magic, so bss clearing is harmless */
static volatile uint32_t done_cnt; /* bss, cleared by hart 0 before `go` is set */
static volatile uint32_t mt_fail;

/* Run case i on hart h; returns 0 when token and every stats byte match. */
static int run_case(const hdr_t* hd, uint32_t i, int h, int32_t* tok_out) {
    const row_t* rows = (const row_t*)(x280s_corpus + hd->rows_off);
    const case_t* c = (const case_t*)(x280s_corpus + hd->cases_off) + i;
    const void* data = x280s_corpus + rows[c->row].off;
    int32_t tok;
    if (c->mode == 0) {
        tok = x280s_sample_row(
            data, rows[c->row].dtype, c->vocab, c->stride, &c->params, c->user, c->step, &work[h], &stats[h]);
    } else {
        tok = x280s_argmax_row(data, rows[c->row].dtype, c->vocab, c->stride, &stats[h]);
    }
    *tok_out = tok;
    int bad = tok != c->token;
    const volatile uint8_t* a = (const volatile uint8_t*)&stats[h];
    const uint8_t* b = (const uint8_t*)&c->stats;
    for (uint32_t k = 0; k < sizeof(x280s_stats_t); k++) {
        bad |= a[k] != b[k];
    }
    /* the same case through the fused copy + prep path (contiguous rows only) */
    uint32_t es = rows[c->row].dtype == X280S_DTYPE_BF16 ? 2u : 4u;
    if (c->stride == 1 && (uint64_t)c->vocab * es <= sizeof copybuf[0]) {
        x280s_copy_prepare(copybuf[h], data, rows[c->row].dtype, c->vocab, &prep[h]);
        x280s_params_t gp = c->params;
        if (c->mode != 0) {
            gp.temperature = 0.0f; /* argmax case: the greedy path of sample_row_prep */
        }
        int32_t t2 = x280s_sample_row_prep(
            copybuf[h], rows[c->row].dtype, c->vocab, &gp, c->user, c->step, &work[h], &prep[h], &stats[h]);
        bad |= (t2 != c->token) << 1;
        for (uint32_t k = 0; k < sizeof(x280s_stats_t); k++) {
            bad |= (a[k] != b[k]) << 1;
        }
        if (bad & 2) {
            *tok_out = t2;
        }
        /* incremental (streaming) form: ranges of (1 + i % 8) * 512 elements */
        if (x280s_copy_prepare_begin(copybuf[h], rows[c->row].dtype, c->vocab, &prep[h])) {
            uint32_t step512 = (1u + i % 8u) * X280S_PREP_ALIGN;
            for (uint32_t to = step512; prep[h].done < c->vocab; to += step512) {
                x280s_copy_prepare_range(copybuf[h], data, to, &prep[h]);
            }
            x280s_copy_prepare_end(&prep[h]);
            int32_t t3 = x280s_sample_row_prep(
                copybuf[h], rows[c->row].dtype, c->vocab, &gp, c->user, c->step, &work[h], &prep[h], &stats[h]);
            bad |= (t3 != c->token) << 2;
            for (uint32_t k = 0; k < sizeof(x280s_stats_t); k++) {
                bad |= (a[k] != b[k]) << 2;
            }
            if (bad & 4) {
                *tok_out = t3;
            }
        }
    }
    return bad;
}

static void report_fail(const hdr_t* hd, uint32_t i, int32_t tok) {
    const case_t* c = (const case_t*)(x280s_corpus + hd->cases_off) + i;
    puts_("FAIL ");
    putdec(i);
    puts_(" token exp ");
    putdec((uint64_t)(int64_t)c->token);
    puts_(" got ");
    putdec((uint64_t)(int64_t)tok);
    const uint32_t* a = (const uint32_t*)&stats[0];
    const uint32_t* b = (const uint32_t*)&c->stats;
    for (uint32_t k = 0; k < sizeof(x280s_stats_t) / 4; k++) {
        if (a[k] != b[k]) {
            puts_(" w");
            putdec(k);
            puts_(":");
            puthex(b[k], 8);
            puts_("/");
            puthex(a[k], 8);
        }
    }
    puts_("\n");
}

static void finish(uint32_t fails) {
    *FINISHER = fails ? ((fails & 0xffff) << 16) | 0x3333 : 0x5555;
    for (;;) {
        __asm__ volatile("wfi");
    }
}

static void mt_worker(int h) {
    const hdr_t* hd = (const hdr_t*)x280s_corpus;
    uint32_t fails = 0;
    for (uint32_t i = (uint32_t)h; i < hd->n_cases; i += NHARTS) {
        int32_t tok;
        fails += (uint32_t)run_case(hd, i, h, &tok);
    }
    __atomic_fetch_add(&mt_fail, fails, __ATOMIC_SEQ_CST);
    __atomic_fetch_add(&done_cnt, 1, __ATOMIC_SEQ_CST);
}

void secondary_main(uint64_t hart) {
    if (hart >= NHARTS) {
        return;
    }
    while (__atomic_load_n(&go, __ATOMIC_ACQUIRE) != 0x60606060u) {
    }
    mt_worker((int)hart);
}

static uint64_t instret(void) {
    uint64_t v;
    __asm__ volatile("csrr %0, minstret" : "=r"(v));
    return v;
}

/* Retired-instruction counts per row on corpus rows 0 (f32) and 1 (bf16), V = 151936 (QEMU: a vector
 * instruction counts as one). A proxy for x280 cost, not a timing. */
static void instret_report(const hdr_t* hd) {
    const row_t* rows = (const row_t*)(x280s_corpus + hd->rows_off);
    static const struct {
        const char* name;
        float T;
        uint32_t k;
        float p;
    } st[] = {{"greedy", 0.0f, 0, 1.0f}, {"k=50", 0.7f, 50, 0.9f}, {"k=20", 0.6f, 20, 0.95f}, {"k=0", 1.0f, 0, 1.0f}};
    for (uint32_t r = 0; r < 2 && r < hd->n_rows; r++) {
        if (rows[r].n_elems < 151936u) {
            continue;
        }
        for (uint32_t s = 0; s < 4; s++) {
            x280s_params_t p = {st[s].T, st[s].k, st[s].p, 0, 1};
            uint64_t a = instret();
            x280s_sample_row(x280s_corpus + rows[r].off, rows[r].dtype, 151936u, 1, &p, 0, 0, &work[0], &stats[0]);
            uint64_t b = instret();
            puts_("INSTRET ");
            puts_(rows[r].dtype ? "bf16 " : "f32 ");
            puts_(st[s].name);
            puts_(" ");
            putdec(b - a);
            puts_("\n");
        }
    }
}

int main(void) {
    const hdr_t* hd = (const hdr_t*)x280s_corpus;
    puts_("x280s qemu bit-exactness test, rvv=");
    putdec(x280s_has_rvv());
    puts_("\n");
    uint32_t fails = 0;
    if (hd->magic[0] != 'X' || hd->magic[1] != '2' || hd->magic[2] != 'S' || hd->magic[3] != 'C' || hd->version != 1) {
        puts_("bad corpus header\n");
        finish(1);
    }
#ifdef X280S_EXPECT_RVV
    if (!x280s_has_rvv()) {
        puts_("FAIL expected the RVV build\n");
        fails++;
    }
#endif
    for (uint32_t i = 0; i < hd->n_cases; i++) {
        int32_t tok;
        if (run_case(hd, i, 0, &tok)) {
            report_fail(hd, i, tok);
            fails++;
        } else {
            puts_("PASS ");
            putdec(i);
            puts_(" tok ");
            putdec((uint64_t)tok);
            puts_("\n");
        }
    }
    puts_("phase1: ");
    putdec(hd->n_cases - fails);
    puts_("/");
    putdec(hd->n_cases);
    puts_(" cases bit-identical\n");

    uint64_t h = x280s_expf_hash(hd->expf_step);
    puts_(h == hd->expf_hash ? "PASS expf hash " : "FAIL expf hash ");
    puthex(h, 16);
    puts_(" host ");
    puthex(hd->expf_hash, 16);
    puts_("\n");
    fails += h != hd->expf_hash;

    /* phase 3: four harts concurrently */
    done_cnt = 0;
    mt_fail = 0;
    __atomic_store_n(&go, 0x60606060u, __ATOMIC_RELEASE);
    mt_worker(0);
    uint32_t spins = 0;
    while (__atomic_load_n(&done_cnt, __ATOMIC_ACQUIRE) != NHARTS && ++spins < 0x7fffffffu) {
    }
    if (done_cnt != NHARTS) {
        puts_("FAIL multihart: not all harts finished\n");
        fails++;
    } else if (mt_fail) {
        puts_("FAIL multihart: ");
        putdec(mt_fail);
        puts_(" mismatches\n");
        fails += mt_fail;
    } else {
        puts_("PASS multihart: 4 harts x ");
        putdec(hd->n_cases / NHARTS);
        puts_("+ cases each, bit-identical\n");
    }
    instret_report(hd);
    if (fails) {
        puts_("FAILED ");
        putdec(fails);
        puts_("\n");
    } else {
        puts_("ALL PASS ");
        putdec(hd->n_cases);
        puts_(" cases\n");
    }
    finish(fails);
    return 0;
}
