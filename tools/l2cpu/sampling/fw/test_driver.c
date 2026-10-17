// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-FileCopyrightText: © 2026 Kansei Motoe (kinginu)
//
// SPDX-License-Identifier: Apache-2.0

/*
 * test_driver.c (sampling application, QEMU test flavour only). Runs on a fifth hart (mhartid 4) and plays the host
 * and the Tensix producers: fills fake logits (in the region zones or in fake DRAM banks, with its own scatter code),
 * bumps req_seq, rings the emulated doorbell, waits for done_seq and checks next_tokens, the remote token tensor, the
 * output ring and the timing ring against the sampling library run on this hart. Then: ordering stress, the stream
 * protocol (fast / slow / small-block producers), the restart suite (WARM into other slots, a request published
 * while parked, COLD, restart cycles), trap isolation + restart, and the forced bounds of the request path (stalled
 * producer, full reader gate, bad descriptor, stalled worker). Exit code through the virt test finisher.
 */
#include <stdarg.h>

#include "fw.h"
#include "l2cpu_sampling.h"
#include "platform.h"
#include "x280s.h"

#ifndef L2S_TEST_STRESS
#define L2S_TEST_STRESS 20000u
#endif
#ifndef L2S_TEST_STRESS32
#define L2S_TEST_STRESS32 1000u
#endif
#ifndef L2CPU_RESTART_CYCLES
#define L2CPU_RESTART_CYCLES 1000u
#endif
#define DRIVER_SCRATCH 0x88000000ull /* driver-only RAM above the 64 MiB region (QEMU -m 1G) */
#define SLOT_C 0x240000u             /* QEMU-only extra slot: the driver keeps running from slot A */

void console_write(const char* s);
extern char _load_end[];
extern volatile uint32_t l2s_uc_gate;

static uint8_t* R;
static l2cpu_header_t* H;
static l2s_ctrl_t* C;
static uint32_t req, mbreq, ring_wr_expect, timing_expect;
static x280s_work_t dwork;

static void tprintf(const char* fmt, ...) __attribute__((format(printf, 1, 2)));
static void tprintf(const char* fmt, ...) {
    char buf[256];
    va_list ap;
    va_start(ap, fmt);
    fw_vsnprintf(buf, sizeof buf, fmt, ap);
    va_end(ap);
    console_write(buf);
}

static uint64_t ticks_ms(uint64_t ms) { return (uint64_t)L2CPU_QEMU_MTIME_HZ / 1000u * ms; }
static uint64_t ticks_us(uint64_t us) { return (uint64_t)L2CPU_QEMU_MTIME_HZ / 1000000u * us; }

static void dump_log_tail(void) {
    uint64_t wr = rd64(R + L2CPU_OFF_LOG_WR);
    uint64_t n = wr < 3000 ? wr : 3000;
    char buf[2] = {0, 0};
    console_write("---- firmware log tail ----\n");
    for (uint64_t i = wr - n; i < wr; i++) {
        buf[0] = (char)R[L2CPU_OFF_LOG_DATA + i % L2CPU_LOG_DATA_SIZE];
        console_write(buf);
    }
    console_write("---- end of log ----\n");
}

#define FAIL(...)                                    \
    do {                                             \
        tprintf("FAIL %s:%d: ", __FILE__, __LINE__); \
        tprintf(__VA_ARGS__);                        \
        tprintf("\n");                               \
        dump_log_tail();                             \
        plat_finish(1);                              \
    } while (0)
#define CHECK(c, ...)          \
    do {                       \
        if (!(c))              \
            FAIL(__VA_ARGS__); \
    } while (0)

static uint32_t r32(uint32_t off) { return rd32(R + off); }
static void w32(uint32_t off, uint32_t v) { wr32(R + off, v); }

/* ---- PRNG ----------------------------------------------------------------------------------------------------- */
static uint64_t rng_state = 0x9E3779B97F4A7C15ull;
static uint64_t rnd(void) {
    uint64_t x = rng_state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    rng_state = x;
    return x;
}

/* ---- fake NoC (the driver's own copy of the interleaving rule, deliberately not the header's function) -------- */
static uint8_t* noc_ptr(uint8_t x, uint8_t y, uint64_t addr) {
    return (uint8_t*)(uintptr_t)(L2CPU_QEMU_FAKE_NOC_BASE +
                                 (((uint64_t)y * L2CPU_QEMU_FAKE_NOC_MAX_X + x) << L2CPU_QEMU_FAKE_NOC_SLOT_SHIFT) +
                                 addr);
}
static uint8_t* page_ptr(const l2s_tensor_desc_t* d, uint32_t page) {
    uint32_t k = d->first_bank + page;
    const l2s_bank_t* b = &d->bank[k % d->num_banks];
    return noc_ptr(b->x, b->y, b->addr + (uint64_t)(k / d->num_banks) * d->page_stride);
}

static void make_desc(
    l2s_tensor_desc_t* d,
    uint32_t dtype,
    uint32_t layout,
    uint32_t rows,
    uint32_t cols,
    uint32_t page_size,
    uint32_t nb,
    uint32_t first,
    uint64_t base) {
    memset(d, 0, sizeof *d);
    d->dtype = dtype;
    d->layout = layout;
    d->rows = rows;
    d->cols = cols;
    d->page_size = page_size;
    d->page_stride = (page_size + 63) & ~63u;
    d->num_banks = nb;
    d->first_bank = first;
    d->location = L2S_LOC_NOC;
    for (uint32_t i = 0; i < nb; i++) {
        d->bank[i].x = (uint8_t)(i % 8);
        d->bank[i].y = (uint8_t)(i / 8);
        d->bank[i].addr = base + 0x40ull * i; /* per-bank offset exercises bank[i].addr */
    }
    uint32_t es = l2s_dtype_size(dtype);
    uint64_t pages = layout == L2S_LAYOUT_TILE ? (uint64_t)(rows / 32) * (cols / 32)
                                               : (uint64_t)rows * ((cols * es + page_size - 1) / page_size);
    uint64_t per_bank = (pages + first + nb - 1) / nb;
    CHECK(
        base + 0x40ull * nb + per_bank * d->page_stride <= (base < 0x700000 ? 0x700000ull : 0x800000ull),
        "test tensor does not fit a fake bank");
}

/* logical tensor [rows, cols] in driver scratch -> pages in the fake banks */
static void scatter(const l2s_tensor_desc_t* d, const uint8_t* logical, uint32_t nrows) {
    uint32_t es = l2s_dtype_size(d->dtype);
    if (d->layout == L2S_LAYOUT_ROW_MAJOR) {
        uint32_t row_bytes = d->cols * es;
        uint32_t ppr = (row_bytes + d->page_size - 1) / d->page_size;
        for (uint32_t r = 0; r < nrows; r++) {
            for (uint32_t p = 0; p < ppr; p++) {
                uint32_t n = row_bytes - p * d->page_size;
                if (n > d->page_size) {
                    n = d->page_size;
                }
                memcpy(page_ptr(d, r * ppr + p), logical + (uint64_t)r * row_bytes + (uint64_t)p * d->page_size, n);
            }
        }
    } else {
        static uint8_t tile[32 * 32 * 4];
        uint32_t tpr = d->cols / 32;
        for (uint32_t tr = 0; tr * 32 < nrows; tr++) {
            for (uint32_t tc = 0; tc < tpr; tc++) {
                for (uint32_t rr = 0; rr < 32; rr++) {
                    for (uint32_t cc = 0; cc < 32; cc++) {
                        uint32_t idx = ((rr / 16) * 2 + cc / 16) * 256 + (rr % 16) * 16 + (cc % 16);
                        memcpy(tile + idx * es, logical + ((uint64_t)(tr * 32 + rr) * d->cols + tc * 32 + cc) * es, es);
                    }
                }
                memcpy(page_ptr(d, tr * tpr + tc), tile, 32 * 32 * es);
            }
        }
    }
}

static uint32_t* token_ptr(const l2s_tensor_desc_t* d, uint32_t u) {
    uint32_t r = u / d->cols, c = u % d->cols;
    uint32_t ppr = (d->cols * 4 + d->page_size - 1) / d->page_size;
    uint32_t page = r * ppr + (c * 4) / d->page_size;
    return (uint32_t*)(page_ptr(d, page) + (c * 4) % d->page_size);
}

/* ---- logits ----------------------------------------------------------------------------------------------------- */
static inline uint32_t f2u(float f) {
    uint32_t u;
    memcpy(&u, &f, 4);
    return u;
}
static void put(uint8_t* row, uint32_t dtype, uint32_t j, uint32_t bits32) {
    if (dtype == L2S_DTYPE_BF16) {
        ((uint16_t*)row)[j] = (uint16_t)(bits32 >> 16);
    } else {
        ((uint32_t*)row)[j] = bits32;
    }
}
static float get(const uint8_t* row, uint32_t dtype, uint32_t j) {
    uint32_t b = dtype == L2S_DTYPE_BF16 ? (uint32_t)((const uint16_t*)row)[j] << 16 : ((const uint32_t*)row)[j];
    float f;
    memcpy(&f, &b, 4);
    return f;
}

static void gen_row(uint8_t* row, uint32_t dtype, uint32_t vocab, uint32_t cols) {
    for (uint32_t j = 0; j < vocab; j++) {
        uint64_t r = rnd();
        float v = (float)((int32_t)(r % 20000u) - 10000) / 1000.0f;
        uint32_t bits = f2u(v);
        if ((r >> 32) % 997u == 0) {
            bits = 0x7FC00000u; /* NaN */
        }
        put(row, dtype, j, bits);
    }
    uint64_t r = rnd();
    uint32_t kind = r % 4;
    uint32_t a = (uint32_t)((r >> 8) % vocab);
    if (kind == 1) {
        put(row, dtype, a, f2u(40.0f)); /* planted max */
    } else if (kind == 2 && vocab > 1) {
        uint32_t b = (uint32_t)((r >> 32) % vocab);
        put(row, dtype, a, f2u(50.0f)); /* tie: the lowest index wins */
        put(row, dtype, b, f2u(50.0f));
    }
    for (uint32_t j = vocab; j < cols; j++) {
        put(row, dtype, j, f2u(1e30f)); /* padding must be ignored */
    }
}

static int32_t ref_argmax(const uint8_t* row, uint32_t dtype, uint32_t vocab) {
    int32_t best = 0;
    float bv = -__builtin_inff();
    int any = 0;
    for (uint32_t j = 0; j < vocab; j++) {
        float v = get(row, dtype, j);
        if (v != v) {
            continue;
        }
        if (!any || v > bv) {
            bv = v;
            best = (int32_t)j;
            any = 1;
        }
    }
    return best;
}

/* ---- params ---------------------------------------------------------------------------------------------------- */
enum { P_GREEDY, P_MIXED };
static void set_params(int mode, uint32_t batch) {
    for (uint32_t u = 0; u < L2S_MAX_USERS; u++) {
        l2s_user_params_t* p = &C->params[u];
        memset(p, 0, sizeof *p);
        uint32_t k = mode == P_GREEDY ? 0 : (u + batch) % 4;
        if (k == 0) {
            p->temperature = 0.0f;
        } else if (k == 1) {
            p->temperature = 0.7f, p->top_k = 50, p->top_p = 0.9f;
        } else if (k == 2) {
            p->temperature = 1.0f, p->top_k = 0, p->top_p = 1.0f;
        } else {
            p->temperature = 0.6f, p->top_k = 20, p->top_p = 0.95f;
        }
        p->seed = 0x1234567ull * (u + 1) + batch;
    }
}

/* user u of the next request: its (tile-local) params, drawn with the global index user_base + u */
static int32_t ref_sample(const uint8_t* row, uint32_t dtype, uint32_t vocab, uint32_t u) {
    x280s_params_t p;
    memcpy(&p, &C->params[u], sizeof p);
    x280s_stats_t st;
    return x280s_sample_row(
        row,
        dtype,
        vocab,
        1,
        &p,
        rd32(&C->user_base.v) + u,
        (uint64_t)(uint32_t)(req + 1 - C->step_seq_base.v),
        &dwork,
        &st);
}

/* ---- request protocol (what the notify / wait kernels do) ------------------------------------------------------ */
static void issue(void) {
    fence(); /* producer: data, then req_seq */
    req++;
    w32(L2S_OFF_REQ_SEQ, req);
    fence();
    plat_qemu_ring_doorbell(req | 0x80000000u); /* never push 0 */
}

static int wait_done(uint64_t timeout_ms) {
    uint64_t deadline = plat_mtime() + ticks_ms(timeout_ms);
    for (;;) {
        uint32_t d = r32(L2S_OFF_DONE_SEQ);
        if (d == req) {
            break;
        }
        CHECK((int32_t)(d - req) < 0, "done_seq %u ahead of req_seq %u", d, req);
        if (plat_mtime() > deadline) {
            return -1;
        }
    }
    fence(); /* consumer: done_seq, then data */
    return 0;
}

static void set_ctrl(uint32_t batch, uint32_t vocab, uint32_t vpad, uint32_t flags) {
    wr32(&C->batch.v, batch);
    wr32(&C->vocab.v, vocab);
    wr32(&C->vocab_padded.v, vpad);
    wr32(&C->flags.v, flags);
}

/* rows pushed into a region zone (what the Tensix push kernel does) */
static void make_local(l2s_tensor_desc_t* d, uint32_t loc, uint32_t dtype, uint32_t cols) {
    memset(d, 0, sizeof *d);
    d->location = loc;
    d->dtype = dtype;
    d->layout = L2S_LAYOUT_ROW_MAJOR;
    d->rows = 32;
    d->cols = cols;
    d->row_stride = (cols * l2s_dtype_size(dtype) + 63) & ~63u;
    d->local_off = loc == L2S_LOC_REGION_UC ? L2S_OFF_LOGITS_UC : L2S_OFF_LOGITS;
}
static uint8_t* local_row(const l2s_tensor_desc_t* d, uint32_t r) {
    return R + d->local_off + (uint64_t)r * d->row_stride;
}

static void check_ring_and_timing(uint32_t batch, const int32_t* exp, int ring) {
    if (ring) {
        ring_wr_expect++;
        uint32_t rw = rd32(&C->ring_wr.v);
        CHECK(rw == ring_wr_expect, "ring_wr %u expected %u", rw, ring_wr_expect);
        l2s_ring_slot_t* s = (l2s_ring_slot_t*)(R + L2S_OFF_RING) + (rw - 1) % L2S_RING_SLOTS;
        CHECK(s->req_seq == req && s->batch == batch, "ring slot seq %u/%u batch %u", s->req_seq, req, s->batch);
        CHECK(s->step == req - rd32(&C->step_seq_base.v), "ring step %u", s->step);
        for (uint32_t u = 0; u < batch; u++) {
            CHECK(s->tok[u] == (uint32_t)exp[u], "ring tok[%u] = %u expected %d (seq %u)", u, s->tok[u], exp[u], req);
        }
    }
    timing_expect++;
    uint32_t tc = rd32(&C->timing_count.v);
    CHECK(tc == timing_expect, "timing_count %u expected %u", tc, timing_expect);
    l2s_timing_t* t = (l2s_timing_t*)(R + L2S_OFF_TIMING) + (tc - 1) % L2S_TIMING_SLOTS;
    CHECK(
        t->req_seq == req && t->batch == batch && t->mtime_publish >= t->mtime_wake,
        "timing record seq %u",
        t->req_seq);
}

static void resync(void) { /* after a restart that served a pending request */
    ring_wr_expect = rd32(&C->ring_wr.v);
    timing_expect = rd32(&C->timing_count.v);
}

/* ---- functional cases ------------------------------------------------------------------------------------------ */
typedef struct {
    const char* name;
    uint32_t loc; /* L2S_LOC_REGION / REGION_UC: rows pushed into a zone; L2S_LOC_NOC: fake DRAM banks */
    uint32_t batch, vocab, vpad, dtype, layout, page_size, nb, first, steps, mode, flags;
    uint32_t tok_cols, tok_page, tok_nb;
    uint32_t user_base; /* global index of user 0 (a tile's share of a split batch); remote tokens land there */
} case_t;

static void run_case(const case_t* c) {
    static l2s_tensor_desc_t L, T;
    uint8_t* logical = (uint8_t*)(uintptr_t)DRIVER_SCRATCH;
    uint32_t es = l2s_dtype_size(c->dtype);
    if (c->loc != L2S_LOC_NOC) {
        make_local(&L, c->loc, c->dtype, c->vpad);
    } else {
        make_desc(
            &L,
            c->dtype,
            c->layout,
            32,
            c->vpad,
            c->layout == L2S_LAYOUT_TILE ? 1024 * es : c->page_size,
            c->nb,
            c->first,
            0x1000);
    }
    uint32_t tcols = c->tok_cols ? c->tok_cols : 32;
    make_desc(
        &T,
        L2S_DTYPE_UINT32,
        L2S_LAYOUT_ROW_MAJOR,
        32 / tcols,
        tcols,
        c->tok_page ? c->tok_page : 128,
        c->tok_nb ? c->tok_nb : 8,
        0,
        0x700000);
    memcpy(&C->logits, &L, sizeof L);
    memcpy(&C->tokens, &T, sizeof T);
    set_params((int)c->mode, c->batch);
    set_ctrl(c->batch, c->vocab, c->vpad, c->flags);
    wr32(&C->user_base.v, c->user_base);
    wr32(&C->step_seq_base.v, req);
    for (uint32_t u = 0; u < 32; u++) {
        *token_ptr(&T, u) = 0xDEADBEEFu;
        C->next_tokens[u].v = 0xDEADBEEFu;
    }
    uint64_t t0 = plat_mtime(), fw_ticks = 0;
    for (uint32_t s = 0; s < c->steps; s++) {
        int32_t exp[L2S_MAX_USERS];
        uint32_t row_bytes = c->vpad * es;
        for (uint32_t u = 0; u < c->batch; u++) {
            uint8_t* row = logical + (uint64_t)u * row_bytes;
            gen_row(row, c->dtype, c->vocab, c->vpad);
            exp[u] = ref_sample(row, c->dtype, c->vocab, u);
            CHECK(exp[u] >= 0, "reference sampling error %d", exp[u]);
            if (C->params[u].temperature <= 0.0f) {
                int32_t g = ref_argmax(row, c->dtype, c->vocab);
                CHECK(g == exp[u], "library greedy %d != driver argmax %d", exp[u], g);
            }
        }
        if (c->loc != L2S_LOC_NOC) {
            for (uint32_t u = 0; u < c->batch; u++) {
                memcpy(local_row(&L, u), logical + (uint64_t)u * row_bytes, row_bytes);
            }
        } else {
            if (c->layout == L2S_LAYOUT_TILE) { /* rows beyond batch inside the tile row: zeros */
                memset(logical + (uint64_t)c->batch * row_bytes, 0, (uint64_t)(32 - c->batch) * row_bytes);
            }
            scatter(&L, logical, c->layout == L2S_LAYOUT_TILE ? 32 : c->batch);
        }
        uint64_t ts = plat_mtime();
        issue();
        CHECK(
            wait_done(20000) == 0,
            "%s step %u: timeout waiting for done_seq %u (done=%u)",
            c->name,
            s,
            req,
            r32(L2S_OFF_DONE_SEQ));
        fw_ticks += plat_mtime() - ts;
        for (uint32_t u = 0; u < 32; u++) {
            uint32_t got = *token_ptr(&T, u), loc = C->next_tokens[u].v, l = u - c->user_base; /* tensor element u */
            if (u < c->batch) {
                CHECK(
                    loc == (uint32_t)exp[u],
                    "%s step %u user %u: next_tokens %u expected %d",
                    c->name,
                    s,
                    u,
                    loc,
                    exp[u]);
            } else {
                CHECK(loc == 0xDEADBEEFu, "%s: next_tokens[%u] written (0x%x)", c->name, u, loc);
            }
            if (u >= c->user_base && l < c->batch && (c->flags & L2S_FLAG_TOKENS_REMOTE)) {
                CHECK(got == (uint32_t)exp[l], "%s step %u user %u: token %u expected %d", c->name, s, u, got, exp[l]);
            } else {
                CHECK(got == 0xDEADBEEFu, "%s: token slot %u was written (0x%x)", c->name, u, got);
            }
        }
        check_ring_and_timing(c->batch, exp, !(c->flags & L2S_FLAG_NO_RING));
    }
    wr32(&C->user_base.v, 0);
    tprintf(
        "PASS %-40s b%-2u V %6u %s %s %s steps %3u (round trip avg %lu us, total %lu ms)\n",
        c->name,
        c->batch,
        c->vocab,
        c->dtype == L2S_DTYPE_BF16 ? "bf16" : "fp32",
        c->layout == L2S_LAYOUT_TILE ? "TILE" : "RM  ",
        c->loc == L2S_LOC_REGION      ? "coh"
        : c->loc == L2S_LOC_REGION_UC ? "uc "
                                      : "noc",
        c->steps,
        fw_ticks / c->steps / ticks_us(1),
        (plat_mtime() - t0) / ticks_ms(1));
}

/* ---- ordering stress: every request moves the planted maximum; a stale read shows as a wrong token ----------- */
static void stress(uint32_t batch, uint32_t iters) {
    static l2s_tensor_desc_t L, T;
    const uint32_t V = 64;
    make_local(&L, batch > 1 ? L2S_LOC_REGION_UC : L2S_LOC_REGION, L2S_DTYPE_FP32, V);
    make_desc(&T, L2S_DTYPE_UINT32, L2S_LAYOUT_ROW_MAJOR, 1, 32, 128, 8, 0, 0x700000);
    memcpy(&C->logits, &L, sizeof L);
    memcpy(&C->tokens, &T, sizeof T);
    set_params(P_GREEDY, batch);
    set_ctrl(batch, V, V, L2S_FLAG_TOKENS_REMOTE);
    wr32(&C->step_seq_base.v, req);
    float* rowp[L2S_MAX_USERS];
    uint32_t prev[L2S_MAX_USERS];
    for (uint32_t u = 0; u < batch; u++) {
        rowp[u] = (float*)local_row(&L, u);
        for (uint32_t j = 0; j < V; j++) {
            rowp[u][j] = (float)j * 0.01f;
        }
        prev[u] = 0;
    }
    uint64_t t0 = plat_mtime();
    for (uint32_t i = 0; i < iters; i++) {
        int32_t exp[L2S_MAX_USERS];
        for (uint32_t u = 0; u < batch; u++) {
            uint32_t idx = (i * 7919u + u * 17u) % V;
            rowp[u][prev[u]] = (float)prev[u] * 0.01f;
            rowp[u][idx] = 100.0f + (float)(i & 1023);
            prev[u] = idx;
            exp[u] = (int32_t)idx;
        }
        issue();
        CHECK(
            wait_done(5000) == 0, "stress b%u iter %u: timeout (done=%u req=%u)", batch, i, r32(L2S_OFF_DONE_SEQ), req);
        for (uint32_t u = 0; u < batch; u++) {
            CHECK(
                *token_ptr(&T, u) == (uint32_t)exp[u],
                "stress b%u iter %u user %u: STALE token %u expected %d",
                batch,
                i,
                u,
                *token_ptr(&T, u),
                exp[u]);
            CHECK(
                C->next_tokens[u].v == (uint32_t)exp[u],
                "stress b%u iter %u user %u: STALE next_tokens %u",
                batch,
                i,
                u,
                C->next_tokens[u].v);
        }
        check_ring_and_timing(batch, exp, 1);
    }
    tprintf(
        "PASS stress batch %u: %u round trips, no stale read, avg %lu us per round trip (QEMU)\n",
        batch,
        iters,
        (plat_mtime() - t0) / iters / ticks_us(1));
}

/* ---- mailbox ---------------------------------------------------------------------------------------------------- */
static uint32_t mb(uint32_t cmd, uint64_t a0, uint64_t a1, uint64_t a2, uint64_t a3, uint64_t* rep, int ring) {
    l2cpu_mailbox_t* m = (l2cpu_mailbox_t*)(R + L2CPU_OFF_MAILBOX);
    wr32(&m->cmd, cmd);
    wr64(&m->arg[0], a0);
    wr64(&m->arg[1], a1);
    wr64(&m->arg[2], a2);
    wr64(&m->arg[3], a3);
    fence();
    mbreq++;
    wr32(&m->req.v, mbreq);
    fence();
    if (ring) {
        plat_qemu_ring_doorbell(0x4D420000u | (mbreq & 0xFFFF));
    }
    uint64_t deadline = plat_mtime() + ticks_ms(2000);
    while (rd32(&m->ack.v) != mbreq) {
        CHECK(plat_mtime() < deadline, "mailbox cmd %u: no ack", cmd);
    }
    fence();
    for (int i = 0; i < 7 && rep; i++) {
        rep[i] = rd64(&m->reply[i]);
    }
    return rd32(&m->status);
}

/* ---- stream protocol: the driver plays a producer that lands data AFTER req_seq ------------------------------- */
static void spin_us(uint32_t us) {
    uint64_t t = plat_mtime() + ticks_us(us);
    while (plat_mtime() < t) {
    }
}
static void land(uint32_t count) {
    fence(); /* producer: data, then the landed count */
    w32(L2S_OFF_LANDED, L2S_LANDED(req, count));
    fence();
}

static void run_stream_case(
    const char* name, uint32_t loc, uint32_t batch, uint32_t vocab, uint32_t steps, uint32_t pace_us) {
    static l2s_tensor_desc_t L, T;
    uint8_t* logical = (uint8_t*)(uintptr_t)DRIVER_SCRATCH;
    const uint32_t es = 2;
    make_local(&L, loc, L2S_DTYPE_BF16, vocab);
    make_desc(&T, L2S_DTYPE_UINT32, L2S_LAYOUT_ROW_MAJOR, 1, 32, 128, 8, 0, 0x700000);
    memcpy(&C->logits, &L, sizeof L);
    memcpy(&C->tokens, &T, sizeof T);
    set_params(P_MIXED, batch);
    set_ctrl(batch, vocab, vocab, L2S_FLAG_STREAMED | L2S_FLAG_TOKENS_REMOTE);
    C->stream.mode = L2S_STREAM_ROWS;
    C->stream.timeout_us = 0;
    wr32(&C->step_seq_base.v, req);
    uint64_t t0 = plat_mtime();
    for (uint32_t st = 0; st < steps; st++) {
        int32_t exp[L2S_MAX_USERS];
        uint32_t row_bytes = vocab * es;
        for (uint32_t u = 0; u < batch; u++) {
            uint8_t* row = logical + (uint64_t)u * row_bytes;
            gen_row(row, L2S_DTYPE_BF16, vocab, vocab);
            exp[u] = ref_sample(row, L2S_DTYPE_BF16, vocab, u);
        }
        issue(); /* req_seq + doorbell FIRST, data afterwards */
        uint32_t n = 0;
        for (uint32_t pp = 0; pp < L2S_NHARTS * l2s_users_per_hart(batch); pp++) {
            uint32_t r = l2s_stream_row(pp, l2s_users_per_hart(batch));
            if (r >= batch) {
                continue;
            }
            memcpy(local_row(&L, r), logical + (uint64_t)r * row_bytes, row_bytes);
            land(++n);
            if (pace_us) {
                spin_us((uint32_t)(rnd() % pace_us));
            }
        }
        CHECK(wait_done(20000) == 0, "%s step %u: timeout (done %u req %u)", name, st, r32(L2S_OFF_DONE_SEQ), req);
        for (uint32_t u = 0; u < batch; u++) {
            CHECK(
                C->next_tokens[u].v == (uint32_t)exp[u],
                "%s step %u user %u: next_tokens %u expected %d",
                name,
                st,
                u,
                C->next_tokens[u].v,
                exp[u]);
            CHECK(
                *token_ptr(&T, u) == (uint32_t)exp[u],
                "%s step %u user %u: token %u expected %d",
                name,
                st,
                u,
                *token_ptr(&T, u),
                exp[u]);
        }
        check_ring_and_timing(batch, exp, 1);
    }
    tprintf(
        "PASS stream %-36s b%-2u V %6u ROWS pace<=%5u us steps %4u (%lu ms)\n",
        name,
        batch,
        vocab,
        pace_us,
        steps,
        (plat_mtime() - t0) / ticks_ms(1));
}

static void stream_tests(void) {
    CHECK(
        (H->ident.build_flags >> L2CPU_BUILD_APP_SHIFT) & L2S_BUILD_STREAM,
        "image does not advertise L2S_BUILD_STREAM");
    run_stream_case("rows uc fast producer", L2S_LOC_REGION_UC, 32, 151936, 2, 0);
    run_stream_case("rows uc slow producer", L2S_LOC_REGION_UC, 32, 151936, 2, 3000);
    run_stream_case("rows coherent b1", L2S_LOC_REGION, 1, 151936, 3, 2000);
    run_stream_case("rows uc b13", L2S_LOC_REGION_UC, 13, 8192, 10, 200);
    run_stream_case("rows stress", L2S_LOC_REGION_UC, 4, 4096, 2000, 0);
}

/* ---- restart (the driver plays the host's L2cpuCtl) ------------------------------------------------------------- */
static uint8_t* res(uint32_t off) { return R + L2CPU_OFF_RESIDENT + off; }
static uint32_t rec32(uint32_t h, uint32_t off) { return rd32(res(L2CPU_RES_REC + L2CPU_REC_SIZE * h + off)); }
static void copy_image(uint32_t slot) {
    memcpy(R + slot, _image_start, (size_t)(_load_end - _image_start));
    fence();
}
static void wait_parked(uint32_t mask, const char* what) {
    uint64_t deadline = plat_mtime() + ticks_ms(2000);
    for (uint32_t h = 0; h < 4; h++) {
        if (mask & (1u << h)) {
            while (rec32(h, L2CPU_REC_STATE) != L2CPU_STATE_PARKED) {
                CHECK(plat_mtime() < deadline, "%s: hart %u not parked (state %u)", what, h, rec32(h, L2CPU_REC_STATE));
            }
        }
    }
}
/* park every hart: mailbox PARK while hart 0 serves, otherwise park the live workers directly (inject + IPI) */
static void park_all(void) {
    uint64_t r[7];
    if (H->hart_state[0].status != L2CPU_HART_PARKED) {
        CHECK(mb(L2CPU_MB_PARK, 0, 0, 0, 0, r, 1) == L2CPU_MB_OK, "MB_PARK");
    } else {
        for (uint32_t h = 1; h < 4; h++) {
            if (H->hart_state[h].status != L2CPU_HART_PARKED) {
                wr32(&H->inject[h].v, L2CPU_INJECT_PARK);
                fence();
                plat_ipi_send(h);
            }
        }
    }
    wait_parked(0xF, "park");
}
static uint64_t restart_ticks;
static uint32_t restart(uint32_t slot, int warm, int already_parked) {
    uint64_t t0 = plat_mtime();
    if (!already_parked) {
        park_all();
    }
    wait_parked(0xF, "restart");
    uint32_t ep = rd32(res(L2CPU_RES_BOOT_EPOCH)) + 1;
    w32(L2CPU_OFF_FW_STATUS, 0);
    wr64(res(L2CPU_RES_ENTRY), (uint64_t)(uintptr_t)R + slot);
    wr32(res(L2CPU_RES_BOOT_MODE), warm ? L2CPU_BOOT_WARM : L2CPU_BOOT_COLD);
    wr32(res(L2CPU_RES_BOOT_EPOCH), ep);
    fence();
    wr32(res(L2CPU_RES_GO_EPOCH), rd32(res(L2CPU_RES_GO_EPOCH)) + 1); /* last */
    fence();
    uint64_t deadline = plat_mtime() + ticks_ms(3000);
    while (!(r32(L2CPU_OFF_FW_STATUS) == L2CPU_FW_STATUS_READY && r32(L2CPU_OFF_BOOT_EPOCH) == ep)) {
        CHECK(
            plat_mtime() < deadline,
            "restart epoch %u: not READY (status %u epoch %u)",
            ep,
            r32(L2CPU_OFF_FW_STATUS),
            r32(L2CPU_OFF_BOOT_EPOCH));
    }
    fence();
    if (!warm) {
        req = mbreq = ring_wr_expect = timing_expect = 0;
    }
    restart_ticks += plat_mtime() - t0;
    return ep;
}

/* one batch-1 request with a planted maximum (coherent zone, V = 64) */
static void quick_request(uint32_t i) {
    static l2s_tensor_desc_t L;
    const uint32_t V = 64;
    make_local(&L, L2S_LOC_REGION, L2S_DTYPE_FP32, V);
    memcpy(&C->logits, &L, sizeof L);
    set_params(P_GREEDY, 1);
    set_ctrl(1, V, V, 0);
    float* row = (float*)local_row(&L, 0);
    for (uint32_t j = 0; j < V; j++) {
        row[j] = (float)j * 0.01f;
    }
    uint32_t idx = (i * 7919u) % V;
    row[idx] = 100.0f;
    issue();
    CHECK(wait_done(2000) == 0, "quick request %u: timeout", i);
    CHECK(C->next_tokens[0].v == idx, "quick request %u: token %u expected %u", i, C->next_tokens[0].v, idx);
    int32_t exp[1] = {(int32_t)idx};
    check_ring_and_timing(1, exp, 1);
}

static const case_t c32_after = {
    "b32 mixed uc + remote tokens",
    L2S_LOC_REGION_UC,
    32,
    3000,
    3008,
    L2S_DTYPE_BF16,
    0,
    0,
    0,
    0,
    3,
    P_MIXED,
    L2S_FLAG_TOKENS_REMOTE,
    0,
    0,
    0,
    0};

static void restart_suite(void) {
    uint64_t r[7];
    CHECK(H->ident.build_flags & L2CPU_BUILD_RESTART, "image does not advertise L2CPU_BUILD_RESTART");
    CHECK(H->ident.boot_epoch == 1 && H->ident.boot_mode == L2CPU_BOOT_COLD, "first boot epoch/mode");
    copy_image(L2CPU_SLOT_B);
    copy_image(SLOT_C);
    uint32_t done0 = r32(L2S_OFF_DONE_SEQ);
    /* 1. WARM into a different address: sequence words, rings and counters continue */
    uint32_t ep = restart(L2CPU_SLOT_B, 1, 0);
    CHECK(H->ident.image_base == (uint64_t)(uintptr_t)R + L2CPU_SLOT_B, "image base after restart");
    CHECK(r32(L2S_OFF_DONE_SEQ) == done0 && r32(L2S_OFF_REQ_SEQ) == req && H->ident.restart_count == 1, "warm state");
    quick_request(1);
    run_case(&c32_after);
    tprintf("PASS restart: WARM into slot B (different address), epoch %u, sequence words + rings continue\n", ep);
    /* 2. WARM at the same address, image rewritten while parked */
    CHECK(mb(L2CPU_MB_PARK, 0, 0, 0, 0, r, 1) == L2CPU_MB_OK, "MB_PARK");
    wait_parked(0xF, "same address");
    copy_image(L2CPU_SLOT_B);
    ep = restart(L2CPU_SLOT_B, 1, 1);
    quick_request(2);
    tprintf("PASS restart: WARM at the same address (image rewritten while parked), epoch %u\n", ep);
    /* 3. a request published while every hart is parked is served late after the restart */
    park_all();
    {
        static l2s_tensor_desc_t L;
        make_local(&L, L2S_LOC_REGION, L2S_DTYPE_FP32, 64);
        memcpy(&C->logits, &L, sizeof L);
        set_ctrl(1, 64, 64, 0);
        float* row = (float*)local_row(&L, 0);
        for (uint32_t j = 0; j < 64; j++) {
            row[j] = (float)j * 0.01f;
        }
        row[33] = 50.0f;
        issue(); /* the wait side starts polling now */
        CHECK(wait_done(30) != 0, "request served while every hart was parked");
        ep = restart(SLOT_C, 1, 1);
        CHECK(
            wait_done(2000) == 0 && C->next_tokens[0].v == 33,
            "late request after restart: token %u",
            C->next_tokens[0].v);
        int32_t exp[1] = {33};
        check_ring_and_timing(1, exp, 1);
    }
    tprintf("PASS restart: request published while parked, served late after WARM restart into slot C\n");
    /* 4. COLD restart: link block, control and rings start over */
    ep = restart(L2CPU_SLOT_B, 0, 0);
    CHECK(
        r32(L2S_OFF_REQ_SEQ) == 0 && r32(L2S_OFF_DONE_SEQ) == 0 && H->ident.restart_count == 0 &&
            rd32(&C->ring_wr.v) == 0,
        "cold: not reset");
    quick_request(3);
    tprintf("PASS restart: COLD (sequence words, rings, mailbox reset), epoch %u\n", ep);
    /* 5. restart cycles, alternating slots, WARM, one request each */
    restart_ticks = 0;
    for (uint32_t i = 0; i < L2CPU_RESTART_CYCLES; i++) {
        restart((i & 1) ? SLOT_C : L2CPU_SLOT_B, 1, 0);
        quick_request(100 + i);
    }
    tprintf(
        "PASS restart: %u WARM restart cycles (alternating slots B/C), request bit-exact after each, avg restart "
        "%lu us (QEMU)\n",
        L2CPU_RESTART_CYCLES,
        restart_ticks / L2CPU_RESTART_CYCLES / ticks_us(1));
    restart(L2CPU_SLOT_B, 1, 0);
}

/* ---- trap isolation -------------------------------------------------------------------------------------------- */
static uint64_t hb(uint32_t h) { return rd64(&H->heartbeat[h].count); }

static void trap_tests(void) {
    uint64_t r[7], before[4];
    CHECK(mb(L2CPU_MB_INJECT, 2, 0, 0, 0, r, 1) == L2CPU_MB_OK, "inject");
    uint64_t deadline = plat_mtime() + ticks_ms(2000);
    while (H->hart_state[2].status != L2CPU_HART_PARKED) {
        CHECK(plat_mtime() < deadline, "hart 2 not parked");
    }
    while (rec32(2, L2CPU_REC_STATE) != L2CPU_STATE_PARKED) {
        CHECK(plat_mtime() < deadline, "no resident record");
    }
    CHECK(H->hart_state[2].error == L2CPU_ERR_TRAP && H->hart_state[2].mcause == 2, "trap record");
    for (int h = 0; h < 4; h++) {
        before[h] = hb(h);
    }
    spin_us(30000);
    CHECK(hb(0) > before[0] && hb(1) > before[1] && hb(3) > before[3] && hb(2) == before[2], "heartbeats");
    tprintf("PASS trap: hart 2 illegal instruction -> error TRAP, resident error park; harts 0,1,3 alive\n");
    /* batch 1 and batch 2 (harts 0 and 1: one user each) are still served */
    case_t c1 = {
        "after trap: batch 1", L2S_LOC_REGION, 1, 1000, 1024, L2S_DTYPE_BF16, 0, 0, 0, 0, 10, P_MIXED, 0, 0, 0, 0, 0};
    run_case(&c1);
    case_t c9 = {
        "after trap: batch 2 (harts 0,1)",
        L2S_LOC_NOC,
        2,
        1000,
        1024,
        L2S_DTYPE_FP32,
        L2S_LAYOUT_ROW_MAJOR,
        4096,
        8,
        0,
        5,
        P_MIXED,
        0,
        0,
        0,
        0,
        0};
    run_case(&c9);
    /* batch 32 needs hart 2: hart 0 refuses to publish and stays alive */
    set_ctrl(32, 1000, 1024, 0);
    uint32_t done_before = r32(L2S_OFF_DONE_SEQ);
    issue();
    CHECK(wait_done(300) != 0 && r32(L2S_OFF_DONE_SEQ) == done_before, "batch 32 with a parked worker published");
    deadline = plat_mtime() + ticks_ms(1000);
    while (H->hart_state[0].error != L2CPU_ERR_WORKER_DEAD) {
        CHECK(plat_mtime() < deadline, "no WORKER_DEAD");
    }
    CHECK(H->hart_state[0].error_arg == 2, "WORKER_DEAD names hart 2");
    CHECK(mb(L2CPU_MB_PING, 0, 0, 0, 0, r, 1) == L2CPU_MB_OK, "hart 0 alive after WORKER_DEAD");
    tprintf("PASS worker dead: batch 32 not published, hart 0 error WORKER_DEAD(2), mailbox alive\n");
    /* WARM restart revives hart 2; the pending batch-32 request is served */
    uint32_t ep = restart(L2CPU_SLOT_B, 1, 0);
    CHECK(wait_done(5000) == 0, "pending batch-32 request not served after the restart");
    resync();
    for (uint32_t h = 0; h < 4; h++) {
        CHECK(H->hart_state[h].status == L2CPU_HART_IDLE, "hart %u not idle", h);
    }
    CHECK(H->error.code == L2CPU_ERR_TRAP, "error word must survive a WARM restart");
    run_case(&c32_after);
    tprintf("PASS restart after a worker trap: hart 2 back (epoch %u), pending batch-32 request served\n", ep);
}

/* ---- forced bounds of the request path ------------------------------------------------------------------------- */
/* landed wait: batch 16 streamed (4 users per hart); row 15 (hart 3's last) never lands -> hart 3 parks with
 * STREAM_TIMEOUT after the stream timeout, hart 0 does not publish (WORKER_DEAD) and stays alive; the producer
 * finishes late and a WARM restart serves the request. */
static void bound_stalled_producer(void) {
    static l2s_tensor_desc_t L;
    uint64_t r[7];
    make_local(&L, L2S_LOC_REGION_UC, L2S_DTYPE_BF16, 4096);
    memcpy(&C->logits, &L, sizeof L);
    set_params(P_GREEDY, 16);
    set_ctrl(16, 4096, 4096, L2S_FLAG_STREAMED);
    C->stream.mode = L2S_STREAM_ROWS;
    C->stream.timeout_us = 5000;
    uint32_t done_before = r32(L2S_OFF_DONE_SEQ);
    issue();
    uint64_t t0 = plat_mtime();
    uint32_t n = 0;
    for (uint32_t pp = 0; pp < L2S_NHARTS * l2s_users_per_hart(16); pp++) {
        uint32_t row = l2s_stream_row(pp, l2s_users_per_hart(16));
        if (row >= 16 || row == 15) {
            continue;
        }
        land(++n);
    }
    uint64_t deadline = plat_mtime() + ticks_ms(3000);
    const uint32_t sh = 15 / l2s_users_per_hart(16); /* the hart that owns row 15 */
    while (H->hart_state[sh].status != L2CPU_HART_PARKED) {
        CHECK(
            plat_mtime() < deadline,
            "hart %u not parked (states %u %u %u %u, errors %u %u %u %u, landed 0x%x)",
            sh,
            H->hart_state[0].status,
            H->hart_state[1].status,
            H->hart_state[2].status,
            H->hart_state[3].status,
            H->hart_state[0].error,
            H->hart_state[1].error,
            H->hart_state[2].error,
            H->hart_state[3].error,
            r32(L2S_OFF_LANDED));
    }
    uint64_t dt = plat_mtime() - t0;
    CHECK(
        H->hart_state[sh].error == L2S_ERR_STREAM_TIMEOUT && H->hart_state[sh].error_arg == 16,
        "hart %u error %u arg %lu",
        sh,
        H->hart_state[sh].error,
        H->hart_state[sh].error_arg);
    CHECK(wait_done(100) != 0 && r32(L2S_OFF_DONE_SEQ) == done_before, "stalled request published");
    while (H->hart_state[0].error != L2CPU_ERR_WORKER_DEAD) {
        CHECK(plat_mtime() < deadline, "no WORKER_DEAD");
    }
    CHECK(mb(L2CPU_MB_PING, 0, 0, 0, 0, r, 1) == L2CPU_MB_OK, "hart 0 alive after the stall");
    land(++n); /* the producer finishes late */
    restart(L2CPU_SLOT_B, 1, 0);
    CHECK(wait_done(2000) == 0, "late stream request not served after the restart");
    resync();
    C->stream.timeout_us = 0;
    tprintf(
        "PASS bound landed wait: stalled producer -> hart %u STREAM_TIMEOUT(need 16) after %lu ms (bound 5 ms), not "
        "published, hart 0 alive; served after producer + WARM restart\n",
        sh,
        dt / ticks_ms(1));
}

/* reader gate: the gate is forced full -> hart 0 parks with GATE_TIMEOUT after L2S_GATE_TIMEOUT_US, nothing
 * published; recovery = park the workers + WARM restart (the gate is reset), the pending request is served. */
static void bound_reader_gate(void) {
    static l2s_tensor_desc_t L;
    make_local(&L, L2S_LOC_REGION_UC, L2S_DTYPE_FP32, 64);
    memcpy(&C->logits, &L, sizeof L);
    set_params(P_GREEDY, 1);
    set_ctrl(1, 64, 64, 0);
    float* row = (float*)local_row(&L, 0);
    for (uint32_t j = 0; j < 64; j++) {
        row[j] = (float)j * 0.01f;
    }
    row[21] = 70.0f;
    /* the running image's copy of the gate (the firmware runs from another slot than this driver) */
    volatile uint32_t* gate =
        (volatile uint32_t*)(uintptr_t)(H->ident.image_base + ((uintptr_t)&l2s_uc_gate - (uintptr_t)_image_start));
    *gate = L2S_UC_READERS;
    fence();
    uint32_t done_before = r32(L2S_OFF_DONE_SEQ);
    uint64_t t0 = plat_mtime();
    issue();
    uint64_t deadline = plat_mtime() + ticks_ms(3000);
    while (H->hart_state[0].status != L2CPU_HART_PARKED) {
        CHECK(plat_mtime() < deadline, "hart 0 not parked");
    }
    uint64_t dt = plat_mtime() - t0;
    CHECK(H->hart_state[0].error == L2S_ERR_GATE_TIMEOUT, "hart 0 error %u", H->hart_state[0].error);
    CHECK(r32(L2S_OFF_DONE_SEQ) == done_before, "published through a full gate");
    restart(L2CPU_SLOT_B, 1, 0); /* hart 0 is parked: the workers are parked directly */
    CHECK(
        wait_done(2000) == 0 && C->next_tokens[0].v == 21,
        "pending request after the restart: token %u",
        C->next_tokens[0].v);
    resync();
    tprintf(
        "PASS bound reader gate: full gate -> hart 0 GATE_TIMEOUT after %lu ms (bound %u ms), not published; "
        "served after WARM restart\n",
        dt / ticks_ms(1),
        L2S_GATE_TIMEOUT_US / 1000u);
}

/* bad descriptor: hart 0 parks with BAD_DESC, done_seq and the token tensor untouched; the host fixes the
 * descriptor and restarts, and the pending request is served. */
static void bad_desc_test(void) {
    static l2s_tensor_desc_t T, L;
    make_desc(&T, L2S_DTYPE_UINT32, L2S_LAYOUT_ROW_MAJOR, 1, 32, 128, 8, 0, 0x700000);
    memcpy(&C->tokens, &T, sizeof T);
    for (uint32_t u = 0; u < 32; u++) {
        *token_ptr(&T, u) = 0xDEADBEEFu;
    }
    memcpy(&L, &C->logits, sizeof L);
    C->logits.location = L2S_LOC_NOC;
    C->logits.num_banks = 0;
    set_ctrl(1, 64, 64, L2S_FLAG_TOKENS_REMOTE);
    uint32_t done_before = r32(L2S_OFF_DONE_SEQ);
    issue();
    uint64_t deadline = plat_mtime() + ticks_ms(2000);
    while (H->hart_state[0].status != L2CPU_HART_PARKED) {
        CHECK(plat_mtime() < deadline, "hart 0 not parked");
    }
    CHECK(
        H->hart_state[0].error == L2S_ERR_BAD_DESC && H->hart_state[0].error_arg == L2S_BAD_BANKS,
        "bad desc error %u arg %lu",
        H->hart_state[0].error,
        H->hart_state[0].error_arg);
    CHECK(r32(L2S_OFF_DONE_SEQ) == done_before, "done_seq published for a bad descriptor");
    for (uint32_t u = 0; u < 32; u++) {
        CHECK(*token_ptr(&T, u) == 0xDEADBEEFu, "token tensor touched");
    }
    memcpy(&C->logits, &L, sizeof L); /* the host fixes the descriptor */
    restart(L2CPU_SLOT_B, 1, 0);
    CHECK(
        wait_done(2000) == 0 && *token_ptr(&T, 0) == 21, "pending request after the fix: token %u", *token_ptr(&T, 0));
    resync();
    tprintf(
        "PASS bad descriptor: hart 0 parked with BAD_DESC(BAD_BANKS), done_seq + tokens untouched; served after "
        "the fix + WARM restart\n");
}

/* worker wait: hart 1 spins with interrupts off (it never takes the work item) -> hart 0 gives up after
 * ctrl.work_timeout_us with WORK_TIMEOUT(1), does not publish, stays alive. Last: QEMU cannot RNMI hart 1 back. */
static void bound_stalled_worker(void) {
    uint64_t r[7];
    CHECK(mb(L2CPU_MB_INJECT, 1, L2CPU_INJECT_SPIN, 0, 0, r, 1) == L2CPU_MB_OK, "inject spin");
    spin_us(5000);
    wr32(&C->work_timeout_us.v, 50000);
    set_ctrl(9, 64, 64, 0);
    uint32_t done_before = r32(L2S_OFF_DONE_SEQ);
    uint64_t t0 = plat_mtime();
    issue();
    uint64_t deadline = plat_mtime() + ticks_ms(3000);
    while (H->hart_state[0].error != L2CPU_ERR_WORK_TIMEOUT) {
        CHECK(plat_mtime() < deadline, "no WORK_TIMEOUT");
    }
    uint64_t dt = plat_mtime() - t0;
    CHECK(H->hart_state[0].error_arg == 1 && r32(L2S_OFF_DONE_SEQ) == done_before, "WORK_TIMEOUT(1), not published");
    CHECK(mb(L2CPU_MB_PING, 0, 0, 0, 0, r, 1) == L2CPU_MB_OK, "hart 0 alive after WORK_TIMEOUT");
    tprintf(
        "PASS bound worker wait: stalled worker -> hart 0 WORK_TIMEOUT(1) after %lu ms (bound 50 ms), not "
        "published, hart 0 alive (recovery on the chip: L2 RNMI park + restart)\n",
        dt / ticks_ms(1));
}

static void summary(void) {
    for (uint32_t h = 0; h < 4; h++) {
        l2cpu_counters_t* c = &H->counters[h];
        tprintf(
            "  hart %u: requests %lu users %lu nan %lu kmax_caps %lu wakeups %lu spurious %lu mailbox %lu hb %lu\n",
            h,
            c->app[0],
            c->app[1],
            c->app[2],
            c->app[3],
            c->wakeups,
            c->spurious_wakeups,
            c->mailbox_cmds,
            hb(h));
    }
}

void test_driver_main(void) {
    csr_set(mstatus, (1u << 13) | (1u << 9)); /* FS = VS = Initial: the reference sampling runs here too */
    R = g_region;
    H = g_hdr;
    C = (l2s_ctrl_t*)(R + L2S_OFF_CTRL);
    tprintf(
        "l2cpu sampling qemu-test (%s notify), driver on hart %u\n",
        L2CPU_NOTIFY_POLL ? "poll" : "irq",
        (unsigned)csr_read(mhartid));
    uint64_t deadline = plat_mtime() + ticks_ms(3000);
    while (r32(L2CPU_OFF_FW_STATUS) != L2CPU_FW_STATUS_READY) {
        CHECK(plat_mtime() < deadline, "firmware not READY");
    }
    fence();
    CHECK(
        H->ident.magic == L2CPU_MAGIC && H->ident.app_id == L2S_APP_ID &&
            H->ident.app_layout_version == L2S_LAYOUT_VERSION,
        "ident");
    CHECK(rd32(&C->ring.ring_offset) == L2S_OFF_RING && rd32(&C->ring.timing_offset) == L2S_OFF_TIMING, "ring desc");
    for (uint32_t h = 0; h < 4; h++) {
        CHECK(H->hart_state[h].status == L2CPU_HART_IDLE, "hart %u not idle", h);
    }
    req = r32(L2S_OFF_REQ_SEQ);
    mbreq = r32(L2CPU_OFF_MB_ACK);
    resync();
    tprintf(
        "PASS boot: READY, app %u layout %u, link block at +0x%x, control at +0x%x, 4 harts idle\n",
        L2S_APP_ID,
        L2S_LAYOUT_VERSION,
        L2S_OFF_LINK,
        L2S_OFF_CTRL);

    static const case_t cases[] = {
        /* logits pushed into a region zone (Qwen3: bf16, 151936, stride 303872) */
        {"push b1 Qwen greedy",
         L2S_LOC_REGION,
         1,
         151936,
         151936,
         L2S_DTYPE_BF16,
         0,
         0,
         0,
         0,
         3,
         P_GREEDY,
         0,
         0,
         0,
         0,
         0},
        {"push b1 Qwen sampled + remote tokens",
         L2S_LOC_REGION,
         1,
         151936,
         151936,
         L2S_DTYPE_BF16,
         0,
         0,
         0,
         0,
         3,
         P_MIXED,
         L2S_FLAG_TOKENS_REMOTE,
         0,
         0,
         0,
         0},
        {"push b32 Qwen mixed + remote tokens",
         L2S_LOC_REGION,
         32,
         151936,
         151936,
         L2S_DTYPE_BF16,
         0,
         0,
         0,
         0,
         2,
         P_MIXED,
         L2S_FLAG_TOKENS_REMOTE,
         0,
         0,
         0,
         0},
        {"push uc b32 Qwen mixed + remote tokens",
         L2S_LOC_REGION_UC,
         32,
         151936,
         151936,
         L2S_DTYPE_BF16,
         0,
         0,
         0,
         0,
         2,
         P_MIXED,
         L2S_FLAG_TOKENS_REMOTE,
         0,
         0,
         0,
         0},
        {"push uc b32 small vocab mixed",
         L2S_LOC_REGION_UC,
         32,
         3000,
         3008,
         L2S_DTYPE_BF16,
         0,
         0,
         0,
         0,
         30,
         P_MIXED,
         L2S_FLAG_TOKENS_REMOTE,
         0,
         0,
         0,
         0},
        {"push uc b1 fp32 greedy",
         L2S_LOC_REGION_UC,
         1,
         1000,
         1024,
         L2S_DTYPE_FP32,
         0,
         0,
         0,
         0,
         10,
         P_GREEDY,
         0,
         0,
         0,
         0,
         0},
        {"push b32 small vocab mixed",
         L2S_LOC_REGION,
         32,
         3000,
         3008,
         L2S_DTYPE_BF16,
         0,
         0,
         0,
         0,
         30,
         P_MIXED,
         0,
         0,
         0,
         0,
         0},
        {"push b1 fp32 greedy padded",
         L2S_LOC_REGION,
         1,
         1000,
         1024,
         L2S_DTYPE_FP32,
         0,
         0,
         0,
         0,
         50,
         P_GREEDY,
         L2S_FLAG_TOKENS_REMOTE,
         0,
         0,
         0,
         0},
        /* logits read from interleaved DRAM banks through TLB windows */
        {"noc b1 fp32 RM greedy, 7 banks first 3",
         L2S_LOC_NOC,
         1,
         1000,
         1024,
         L2S_DTYPE_FP32,
         L2S_LAYOUT_ROW_MAJOR,
         4096,
         7,
         3,
         20,
         P_GREEDY,
         L2S_FLAG_TOKENS_REMOTE,
         0,
         0,
         0,
         0},
        {"noc b1 bf16 RM Qwen vocab sampled",
         L2S_LOC_NOC,
         1,
         151936,
         152064,
         L2S_DTYPE_BF16,
         L2S_LAYOUT_ROW_MAJOR,
         304128,
         8,
         0,
         2,
         P_MIXED,
         L2S_FLAG_TOKENS_REMOTE,
         0,
         0,
         0,
         0},
        {"noc b32 bf16 RM 1 KiB pages mixed",
         L2S_LOC_NOC,
         32,
         3000,
         3072,
         L2S_DTYPE_BF16,
         L2S_LAYOUT_ROW_MAJOR,
         1024,
         7,
         2,
         10,
         P_MIXED,
         L2S_FLAG_TOKENS_REMOTE,
         0,
         0,
         0,
         0},
        {"noc b32 fp32 RM Qwen vocab mixed",
         L2S_LOC_NOC,
         32,
         151936,
         152064,
         L2S_DTYPE_FP32,
         L2S_LAYOUT_ROW_MAJOR,
         608256,
         8,
         0,
         1,
         P_MIXED,
         L2S_FLAG_TOKENS_REMOTE,
         0,
         0,
         0,
         0},
        {"noc b32 bf16 TILE mixed, tokens 4x8",
         L2S_LOC_NOC,
         32,
         1000,
         1024,
         L2S_DTYPE_BF16,
         L2S_LAYOUT_TILE,
         0,
         5,
         1,
         10,
         P_MIXED,
         L2S_FLAG_TOKENS_REMOTE,
         8,
         32,
         3,
         0},
        {"noc b13 fp32 RM mixed (4 harts)",
         L2S_LOC_NOC,
         13,
         500,
         512,
         L2S_DTYPE_FP32,
         L2S_LAYOUT_ROW_MAJOR,
         2048,
         3,
         0,
         10,
         P_MIXED,
         L2S_FLAG_TOKENS_REMOTE,
         0,
         0,
         0,
         0},
        {"noc b4 region tokens only, no ring",
         L2S_LOC_NOC,
         4,
         700,
         704,
         L2S_DTYPE_BF16,
         L2S_LAYOUT_ROW_MAJOR,
         1408,
         8,
         0,
         3,
         P_GREEDY,
         L2S_FLAG_NO_RING,
         0,
         0,
         0,
         0},
        /* one tile's share of a split batch: global user index user_base + u (draw and remote token) */
        {"split uc b8 Qwen users 24-31 + remote",
         L2S_LOC_REGION_UC,
         8,
         151936,
         151936,
         L2S_DTYPE_BF16,
         0,
         0,
         0,
         0,
         2,
         P_MIXED,
         L2S_FLAG_TOKENS_REMOTE,
         0,
         0,
         0,
         24},
        {"split uc b16 users 16-31 + remote",
         L2S_LOC_REGION_UC,
         16,
         3000,
         3008,
         L2S_DTYPE_BF16,
         0,
         0,
         0,
         0,
         20,
         P_MIXED,
         L2S_FLAG_TOKENS_REMOTE,
         0,
         0,
         0,
         16},
        {"split noc b5 users 7-11 + remote",
         L2S_LOC_NOC,
         5,
         500,
         512,
         L2S_DTYPE_FP32,
         L2S_LAYOUT_ROW_MAJOR,
         2048,
         3,
         0,
         10,
         P_MIXED,
         L2S_FLAG_TOKENS_REMOTE,
         0,
         0,
         0,
         7},
    };
    for (unsigned i = 0; i < sizeof cases / sizeof cases[0]; i++) {
        run_case(&cases[i]);
    }
    stress(1, L2S_TEST_STRESS);
    stress(32, L2S_TEST_STRESS32);
    stream_tests();
    restart_suite();
    trap_tests();
    bound_stalled_producer();
    bound_reader_gate();
    bad_desc_test();
    bound_stalled_worker();
    summary();
    CHECK(H->counters[0].app[0] > 0 && H->counters[3].app[0] > 0, "counters");
    tprintf("ALL PASS (%s notify)\n", L2CPU_NOTIFY_POLL ? "poll" : "irq");
    plat_finish(0);
}
