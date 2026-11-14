// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-FileCopyrightText: © 2026 Kansei Motoe (kinginu)
//
// SPDX-License-Identifier: Apache-2.0

/*
 * app.c: the sampling application of the L2CPU firmware runtime (fw/app.h hooks).
 *
 * One request = sample `batch` user rows of logits, write the tokens, publish done_seq. Hart 0 (app_poll) sees
 * req_seq move, snapshots the control block, validates it (only when it changed), hands users 8h..8h+7 to worker
 * h with fw_dispatch(), samples users 0..7 itself, waits (bounded) for the workers, writes the output ring slot and
 * the timing record, fences and publishes done_seq. Layout: include/l2cpu_sampling.h.
 */
#include "app.h"
#include "fw.h"
#include "l2cpu_sampling.h"
#include "platform.h"
#include "x280s.h"

_Static_assert(L2S_DTYPE_FP32 == X280S_DTYPE_F32, "dtype fp32");
_Static_assert(L2S_DTYPE_BF16 == X280S_DTYPE_BF16, "dtype bf16");
_Static_assert(sizeof(x280s_params_t) == 24, "params prefix");
_Static_assert(offsetof(l2s_user_params_t, temperature) == offsetof(x280s_params_t, temperature), "t");
_Static_assert(offsetof(l2s_user_params_t, top_k) == offsetof(x280s_params_t, top_k), "k");
_Static_assert(offsetof(l2s_user_params_t, top_p) == offsetof(x280s_params_t, top_p), "p");
_Static_assert(offsetof(l2s_user_params_t, seed) == offsetof(x280s_params_t, seed), "seed");

#ifndef L2S_SAMPLING_FAST
#define L2S_SAMPLING_FAST 1 /* bf16 fast paths of the library + fused copy/prepare (Makefile SAMPLING_FAST) */
#endif

/* per-hart heap: x280s_work_t, then x280s_prep_t, then the row buffer */
#define WORK_OFF 0u
#define PREP_OFF 0x4000u
#define ROWBUF_OFF 0x10000u
_Static_assert(sizeof(x280s_work_t) <= PREP_OFF && PREP_OFF + sizeof(x280s_prep_t) <= ROWBUF_OFF, "heap layout");

void copy_from_window(void* dst, const void* src, size_t n); /* copy_rvv.c */

typedef struct {
    uint32_t seq, batch, vocab, vpad, step_base, flags, ring_slot, nharts;
    uint32_t stream_mode, timeout_us, work_timeout_us;
    l2s_tensor_desc_t logits, tokens;
    l2s_user_params_t params[L2S_MAX_USERS];
} job_t;

static job_t job __attribute__((aligned(64)));
static job_t validated __attribute__((aligned(64))); /* last validated configuration */
static int validated_ok;
static volatile uint32_t worker_cycles[L2S_NHARTS];
static uint32_t ring_wr_local, timing_count_local, served_seq;
static uint32_t caps_logged[L2S_NHARTS];
volatile uint32_t l2s_uc_gate; /* harts inside an uncached-zone read (not static: the QEMU test forces the bound) */

typedef struct {
    uint64_t read, sample, write;
} phase_t;

static inline l2s_ctrl_t* ctrl(void) { return (l2s_ctrl_t*)(g_region + L2S_OFF_CTRL); }
static inline uint8_t* link_word(uint32_t off) { return g_region + L2S_OFF_LINK + off; }

/* FP and vector units: Initial (the library uses both; the runtime leaves FS = VS = Off). */
static inline void fpv_on(void) { csr_set(mstatus, (1u << 13) | (1u << 9)); }

static inline const uint8_t* uncached_alias(const uint8_t* p) {
#if L2CPU_PLATFORM_BH
    return (const uint8_t*)((uintptr_t)p - L2CPU_BH_UNCACHED_ALIAS_DELTA);
#else
    return p;
#endif
}

static inline uint64_t us_ticks(uint64_t us) { return fw_mtime_ticks(us); }

static void cnt_add(uint32_t hart, uint32_t off, uint64_t v) {
    uint64_t* p = (uint64_t*)((uint8_t*)&g_hdr->counters[hart] + off); /* off: L2S_CNT_* */
    wr64(p, rd64(p) + v);
}

/* ---- validation ---------------------------------------------------------------------------------------------- */
static uint32_t check_banks(const l2s_tensor_desc_t* d) {
    if (d->num_banks == 0 || d->num_banks > L2S_MAX_BANKS || d->first_bank >= d->num_banks) {
        return L2S_BAD_BANKS;
    }
    if (d->page_size == 0 || d->page_stride < d->page_size) {
        return L2S_BAD_PAGE;
    }
    if (d->layout != L2S_LAYOUT_ROW_MAJOR && d->layout != L2S_LAYOUT_TILE) {
        return L2S_BAD_LAYOUT;
    }
    uint32_t es = l2s_dtype_size(d->dtype);
    if (d->layout == L2S_LAYOUT_TILE) {
        if (d->rows % L2S_TILE_DIM || d->cols % L2S_TILE_DIM) {
            return L2S_BAD_SHAPE;
        }
        if (d->page_size != L2S_TILE_DIM * L2S_TILE_DIM * es) {
            return L2S_BAD_PAGE;
        }
    } else if (d->page_size % es) {
        return L2S_BAD_PAGE;
    }
    if (d->rows == 0 || d->cols == 0) {
        return L2S_BAD_SHAPE;
    }
    return 0;
}

/* Every page the request touches must be mappable. */
static uint32_t check_pages(const l2s_tensor_desc_t* d, uint32_t row, uint32_t col0, uint32_t n) {
    uint32_t prev_page = ~0u;
    uint32_t es = l2s_dtype_size(d->dtype);
    uint32_t step = d->layout == L2S_LAYOUT_TILE ? L2S_TILE_DIM : (d->page_size / es);
    for (uint32_t c = col0; c < col0 + n;) {
        uint32_t page, off, bank;
        uint64_t addr;
        if (l2s_elem_location(d, row, c, &page, &off)) {
            return L2S_BAD_SHAPE;
        }
        if (page != prev_page) {
            l2s_page_location(d, page, &bank, &addr);
            const l2s_bank_t* b = &d->bank[bank];
            uint32_t len = d->page_size < PLAT_MAP_MAX ? d->page_size : PLAT_MAP_MAX;
            if (!plat_noc_valid(b->x, b->y, addr, len)) {
                return L2S_BAD_UNMAPPABLE;
            }
            prev_page = page;
        }
        c = (c / step + 1) * step; /* next page boundary along the row */
    }
    return 0;
}

/* Region-local tensor: rows [0, rows) inside its zone, coherent [LOGITS, LOGITS_UC) or uncached [LOGITS_UC,
 * REGION_SIZE). This enforces "the uncached zone is never touched through the cached alias". */
static uint32_t check_local(const l2s_tensor_desc_t* d) {
    uint32_t lo = d->location == L2S_LOC_REGION_UC ? L2S_OFF_LOGITS_UC : L2S_OFF_LOGITS;
    uint32_t hi = d->location == L2S_LOC_REGION_UC ? L2S_REGION_SIZE : L2S_OFF_LOGITS_UC;
    uint32_t es = l2s_dtype_size(d->dtype);
    if (d->layout != L2S_LAYOUT_ROW_MAJOR) {
        return L2S_BAD_LAYOUT;
    }
    if (d->rows == 0 || d->cols == 0) {
        return L2S_BAD_SHAPE;
    }
    if (d->row_stride < d->cols * es || d->row_stride % es) {
        return L2S_BAD_PAGE;
    }
    if (d->local_off < lo || d->local_off % 64) {
        return L2S_BAD_LOCATION;
    }
    if ((uint64_t)d->local_off + (uint64_t)d->rows * d->row_stride > hi) {
        return L2S_BAD_LOCATION;
    }
    return 0;
}

static uint32_t validate(const job_t* j) {
    if (j->batch == 0 || j->batch > L2S_MAX_USERS) {
        return L2S_BAD_BATCH;
    }
    if (j->vocab == 0 || j->vocab > j->vpad || j->vpad > j->logits.cols) {
        return L2S_BAD_VOCAB;
    }
    const l2s_tensor_desc_t* L = &j->logits;
    uint32_t r;
    if (L->dtype != L2S_DTYPE_FP32 && L->dtype != L2S_DTYPE_BF16) {
        return L2S_BAD_LOGITS_DTYPE;
    }
    if ((j->flags & L2S_FLAG_STREAMED) && j->stream_mode != L2S_STREAM_ROWS) {
        return L2S_BAD_STREAM;
    }
    if (L->location == L2S_LOC_REGION || L->location == L2S_LOC_REGION_UC) {
        r = check_local(L);
        if (r) {
            return r;
        }
        if (L->rows < j->batch) {
            return L2S_BAD_SHAPE;
        }
        if (L->location == L2S_LOC_REGION_UC &&
            ROWBUF_OFF + (uint64_t)j->vocab * l2s_dtype_size(L->dtype) > L2CPU_HEAP_PER_HART) {
            return L2S_BAD_WORKSPACE;
        }
    } else if (L->location == L2S_LOC_NOC) {
        if (j->flags & L2S_FLAG_STREAMED) {
            return L2S_BAD_STREAM; /* streamed rows land in the region */
        }
        r = check_banks(L);
        if (r) {
            return r;
        }
        if (L->rows < j->batch) {
            return L2S_BAD_SHAPE;
        }
        if (ROWBUF_OFF + (uint64_t)j->vocab * l2s_dtype_size(L->dtype) > L2CPU_HEAP_PER_HART) {
            return L2S_BAD_WORKSPACE;
        }
        for (uint32_t u = 0; u < j->batch; u++) {
            if (L->layout == L2S_LAYOUT_TILE && u % L2S_TILE_DIM) {
                continue; /* same tile row as u-1 */
            }
            r = check_pages(L, u, 0, j->vocab);
            if (r) {
                return r;
            }
        }
    } else {
        return L2S_BAD_LOCATION;
    }
    if (j->flags & L2S_FLAG_TOKENS_REMOTE) {
        const l2s_tensor_desc_t* T = &j->tokens;
        if (T->dtype != L2S_DTYPE_UINT32 && T->dtype != L2S_DTYPE_INT32) {
            return L2S_BAD_TOKENS_DTYPE;
        }
        if (T->location != L2S_LOC_NOC) {
            return L2S_BAD_LOCATION;
        }
        r = check_banks(T);
        if (r) {
            return r;
        }
        if ((uint64_t)T->rows * T->cols < j->batch) {
            return L2S_BAD_SHAPE;
        }
        for (uint32_t u = 0; u < j->batch; u++) {
            r = check_pages(T, u / T->cols, u % T->cols, 1);
            if (r) {
                return r;
            }
        }
    }
    return 0;
}

static int same_config(const job_t* a, const job_t* b) {
    return a->batch == b->batch && a->vocab == b->vocab && a->vpad == b->vpad && a->flags == b->flags &&
           a->stream_mode == b->stream_mode && memcmp(&a->logits, &b->logits, sizeof a->logits) == 0 &&
           memcmp(&a->tokens, &b->tokens, sizeof a->tokens) == 0;
}

/* ---- tensor access ------------------------------------------------------------------------------------------- */
/* Elements [0, n) of row `row` of a NoC tensor -> dst, page run by page run. Returns 0 or -1. */
static int read_row(uint32_t hart, const l2s_tensor_desc_t* d, uint32_t row, uint32_t n, uint8_t* dst) {
    uint32_t es = l2s_dtype_size(d->dtype);
    uint32_t slot = plat_slot(hart, PLAT_SLOT_READ);
    for (uint32_t c = 0; c < n;) {
        uint32_t page, off, bank;
        uint64_t addr;
        if (l2s_elem_location(d, row, c, &page, &off)) {
            return -1;
        }
        uint32_t run = l2s_run_length(d, row, c);
        if (run > n - c) {
            run = n - c;
        }
        if (run * es > PLAT_MAP_MAX) {
            run = PLAT_MAP_MAX / es;
        }
        l2s_page_location(d, page, &bank, &addr);
        const l2s_bank_t* b = &d->bank[bank];
        void* src = plat_noc_map(slot, b->x, b->y, b->noc, addr + off, run * es);
        if (!src) {
            return -1;
        }
        plat_noc_read_prepare(slot, src, run * es);
        copy_from_window(dst + (uint64_t)c * es, src, (size_t)run * es);
        c += run;
    }
    return 0;
}

static int write_token(uint32_t hart, const l2s_tensor_desc_t* d, uint32_t u, uint32_t tok) {
    uint32_t page, off, bank;
    uint64_t addr;
    if (l2s_elem_location(d, u / d->cols, u % d->cols, &page, &off)) {
        return -1;
    }
    l2s_page_location(d, page, &bank, &addr);
    const l2s_bank_t* b = &d->bank[bank];
    volatile uint32_t* p = plat_noc_map(plat_slot(hart, PLAT_SLOT_WRITE), b->x, b->y, b->noc, addr + off, 4);
    if (!p) {
        return -1;
    }
    *p = tok;
    return 0;
}

/* Stream protocol: wait until landed carries this request's tag and count >= need. Parks with
 * L2S_ERR_STREAM_TIMEOUT if the count does not advance for the stream timeout. Returns the count. */
static uint32_t stream_wait(uint32_t need) {
    uint32_t tag = job.seq & 0xFFFFu, last = 0xFFFFFFFFu;
    uint64_t tmo = us_ticks(job.timeout_us), t_last = plat_mtime();
    for (;;) {
        uint32_t v = rd32(link_word(L2CPU_LINK_OFF_LANDED));
        uint32_t c = (v >> 16) == tag ? (v & 0xFFFFu) : 0u;
        if ((v >> 16) == tag && c >= need) {
            fence(); /* consumer: landed first, then the data */
            return c;
        }
        if (c != last) {
            last = c;
            t_last = plat_mtime();
        } else if (plat_mtime() - t_last > tmo) {
            fw_log("seq %u: stream stalled at count %u (need %u, landed 0x%x)", job.seq, c, need, v);
            fw_error_park(L2S_ERR_STREAM_TIMEOUT, need, job.seq);
        }
    }
}

/* At most L2S_UC_READERS harts read the uncached zone at once (2 scale, 3+ collapse 20x). Bounded: a holder that
 * trapped mid-copy (it never releases) or a forced full gate parks the waiter with L2S_ERR_GATE_TIMEOUT. */
static void gate_enter(void) {
    uint64_t t0 = plat_mtime(), tmo = us_ticks(L2S_GATE_TIMEOUT_US);
    for (;;) {
        uint32_t v = l2s_uc_gate;
        if (v < L2S_UC_READERS &&
            __atomic_compare_exchange_n(&l2s_uc_gate, &v, v + 1, 0, __ATOMIC_ACQUIRE, __ATOMIC_RELAXED)) {
            return;
        }
        if (plat_mtime() - t0 > tmo) {
            fw_log("seq %u: uncached-reader gate full (%u) for %u us", job.seq, v, L2S_GATE_TIMEOUT_US);
            fw_error_park(L2S_ERR_GATE_TIMEOUT, v, job.seq);
        }
    }
}
static void gate_exit(void) { __atomic_fetch_sub(&l2s_uc_gate, 1u, __ATOMIC_RELEASE); }

/* Row `row` of the logits: in place for the coherent zone, copied into the hart's row buffer otherwise. With prep,
 * the copy also prepares the row statistics for the library's fast paths (same results). */
static const void* logits_row(uint32_t hart, uint32_t row, uint8_t* rowbuf, x280s_prep_t* prep) {
    const l2s_tensor_desc_t* d = &job.logits;
    if (job.flags & L2S_FLAG_STREAMED) {
        stream_wait(l2s_stream_pos(row, job.batch) + 1u); /* ROWS: this row landed */
    }
    if (d->location == L2S_LOC_REGION) {
        return g_region + d->local_off + (uint64_t)row * d->row_stride;
    }
    if (d->location == L2S_LOC_REGION_UC) {
        const void* src = uncached_alias(g_region + d->local_off + (uint64_t)row * d->row_stride);
        gate_enter();
        if (prep) {
            x280s_copy_prepare(rowbuf, src, d->dtype, job.vocab, prep);
        } else {
            copy_from_window(rowbuf, src, (size_t)job.vocab * l2s_dtype_size(d->dtype));
        }
        gate_exit();
        return rowbuf;
    }
    if (read_row(hart, d, row, job.vocab, rowbuf)) {
        return 0;
    }
    return rowbuf;
}

/* ---- per-hart share ------------------------------------------------------------------------------------------- */
static void do_users(uint32_t hart, uint32_t u0, uint32_t u1, phase_t* ph) {
    hart_local_t* me = self();
    x280s_work_t* work = (x280s_work_t*)(me->heap + WORK_OFF);
    x280s_prep_t* prep = L2S_SAMPLING_FAST ? (x280s_prep_t*)(me->heap + PREP_OFF) : 0;
    uint8_t* rowbuf = me->heap + ROWBUF_OFF;
    l2s_ctrl_t* c = ctrl();
    l2s_ring_slot_t* slot = (l2s_ring_slot_t*)(g_region + L2S_OFF_RING) + job.ring_slot;
    uint64_t step = (uint64_t)(uint32_t)(job.seq - job.step_base);
    uint32_t nan = 0, caps = 0;
    for (uint32_t u = u0; u < u1; u++) {
        uint64_t c0 = rdcycle64();
        if (prep) {
            prep->valid = 0;
        }
        const void* row = logits_row(hart, u, rowbuf, prep);
        if (!row) {
            fw_error_park(L2S_ERR_BAD_DESC, L2S_BAD_UNMAPPABLE, job.seq);
        }
        uint64_t c1 = rdcycle64();
        x280s_params_t p;
        p.temperature = job.params[u].temperature;
        p.top_k = job.params[u].top_k;
        p.top_p = job.params[u].top_p;
        p._pad = 0;
        p.seed = job.params[u].seed;
        x280s_stats_t st;
        int32_t tok = prep && prep->valid
                          ? x280s_sample_row_prep(row, job.logits.dtype, job.vocab, &p, u, step, work, prep, &st)
                          : x280s_sample_row(row, job.logits.dtype, job.vocab, 1, &p, u, step, work, &st);
        uint64_t c2 = rdcycle64();
        if (tok < 0) {
            fw_error_park(L2S_ERR_SAMPLE, (uint64_t)(int64_t)tok, job.seq);
        }
        nan += st.nan_count;
        caps += st.cap_applied;
        if (st.cap_applied && caps_logged[hart] < 4) { /* rate-limited; the counter has the total */
            caps_logged[hart]++;
            fw_log("seq %u user %u: K_MAX cap applied (k_eff %u)", job.seq, u, st.k_eff);
        }
        wr32(&c->next_tokens[u].v, (uint32_t)tok);
        if ((job.flags & L2S_FLAG_TOKENS_REMOTE) && write_token(hart, &job.tokens, u, (uint32_t)tok)) {
            fw_error_park(L2S_ERR_BAD_DESC, L2S_BAD_UNMAPPABLE, job.seq);
        }
        if (!(job.flags & L2S_FLAG_NO_RING)) {
            wr32(&slot->tok[u], (uint32_t)tok);
        }
        uint64_t c3 = rdcycle64();
        ph->read += c1 - c0;
        ph->sample += c2 - c1;
        ph->write += c3 - c2;
    }
    uint64_t c4 = rdcycle64();
    if (job.flags & L2S_FLAG_TOKENS_REMOTE) {
        plat_noc_write_barrier(plat_slot(hart, PLAT_SLOT_WRITE));
    }
    fence();
    ph->write += rdcycle64() - c4;
    cnt_add(hart, L2S_CNT_REQUESTS, 1);
    cnt_add(hart, L2S_CNT_USERS, u1 - u0);
    cnt_add(hart, L2S_CNT_NAN, nan);
    cnt_add(hart, L2S_CNT_KMAX_CAPS, caps);
}

static void __attribute__((noinline)) work(uint32_t hart, uint32_t seq) {
    uint64_t c0 = rdcycle64();
    phase_t ph = {0, 0, 0};
    if (seq != job.seq) { /* cannot happen if hart 0 follows the protocol */
        fw_log("work item %u but job.seq %u", seq, job.seq);
        fw_error_park(L2S_ERR_SEQ, seq, job.seq);
    }
    uint32_t u0 = hart * L2S_USERS_PER_HART, u1 = u0 + L2S_USERS_PER_HART;
    if (u1 > job.batch) {
        u1 = job.batch;
    }
    if (u0 < u1) {
        do_users(hart, u0, u1, &ph);
    }
    worker_cycles[hart] = (uint32_t)(rdcycle64() - c0);
}

void app_work(uint32_t hart, uint32_t seq) {
    fpv_on();
    work(hart, seq);
}

/* ---- hart 0: one request -------------------------------------------------------------------------------------- */
static int serve(uint32_t seq, uint64_t mtime_wake, uint64_t cyc_wake) {
    l2s_ctrl_t* c = ctrl();
    phase_t ph = {0, 0, 0};

    /* snapshot (req_seq was read and fenced by the caller) */
    job.seq = seq;
    job.batch = rd32(&c->batch.v);
    job.vocab = rd32(&c->vocab.v);
    job.vpad = rd32(&c->vocab_padded.v);
    job.step_base = rd32(&c->step_seq_base.v);
    job.flags = rd32(&c->flags.v);
    memcpy(&job.logits, &c->logits, sizeof job.logits);
    memcpy(&job.tokens, &c->tokens, sizeof job.tokens);
    memcpy(job.params, c->params, sizeof job.params);
    job.ring_slot = ring_wr_local % L2S_RING_SLOTS;
    job.stream_mode = rd32(&c->stream.mode);
    job.timeout_us = rd32(&c->stream.timeout_us) ? rd32(&c->stream.timeout_us) : L2S_STREAM_TIMEOUT_US;
    job.work_timeout_us = rd32(&c->work_timeout_us.v) ? rd32(&c->work_timeout_us.v) : L2S_WORK_TIMEOUT_US;

    if (!validated_ok || !same_config(&job, &validated)) {
        uint32_t why = validate(&job);
        if (why) {
            fw_log("seq %u: bad descriptor/config, reason %u", seq, why);
            fw_error_park(L2S_ERR_BAD_DESC, why, seq);
        }
        memcpy(&validated, &job, sizeof job);
        validated_ok = 1;
    }
    job.nharts = (job.batch + L2S_USERS_PER_HART - 1) / L2S_USERS_PER_HART;
    uint32_t mask = 0;
    for (uint32_t h = 1; h < job.nharts; h++) {
        if (rd32(&g_hdr->hart_state[h].status) == L2CPU_HART_PARKED) {
            fw_log("seq %u: worker %u is parked, not publishing", seq, h);
            fw_set_error(0, L2CPU_ERR_WORKER_DEAD, h, seq);
            return -1;
        }
        mask |= 1u << h;
        worker_cycles[h] = 0;
    }
    for (uint32_t h = job.nharts; h < L2S_NHARTS; h++) {
        worker_cycles[h] = 0;
    }
    fw_dispatch(mask, seq); /* fence, job data before each worker's sequence word, IPI */

    uint32_t u1 = job.batch < L2S_USERS_PER_HART ? job.batch : L2S_USERS_PER_HART;
    do_users(0, 0, u1, &ph);

    uint64_t cw = rdcycle64();
    uint64_t bound = job.work_timeout_us + ((job.flags & L2S_FLAG_STREAMED) ? job.timeout_us : 0u);
    if (fw_wait_workers_us(mask, seq, bound)) {
        return -1; /* WORKER_DEAD / WORK_TIMEOUT recorded; not published */
    }
    uint64_t cwait = rdcycle64() - cw;

    /* output ring slot header, then the ring index */
    if (!(job.flags & L2S_FLAG_NO_RING)) {
        l2s_ring_slot_t* slot = (l2s_ring_slot_t*)(g_region + L2S_OFF_RING) + job.ring_slot;
        wr32(&slot->req_seq, seq);
        wr32(&slot->batch, job.batch);
        wr32(&slot->step, seq - job.step_base);
        fence();
        ring_wr_local++;
        wr32(&c->ring_wr.v, ring_wr_local);
    }

    /* timing record (before done_seq: a host that sees done_seq also sees the record) */
    l2s_timing_t* t = (l2s_timing_t*)(g_region + L2S_OFF_TIMING) + timing_count_local % L2S_TIMING_SLOTS;
    uint64_t mt_pub = plat_mtime(), cyc_now = rdcycle64();
    wr32(&t->req_seq, seq);
    wr32(&t->batch, job.batch);
    wr64(&t->mtime_wake, mtime_wake);
    wr64(&t->mtime_publish, mt_pub);
    wr32(&t->cyc_read, (uint32_t)ph.read);
    wr32(&t->cyc_sample, (uint32_t)ph.sample);
    wr32(&t->cyc_write, (uint32_t)ph.write);
    wr32(&t->cyc_wait, (uint32_t)cwait);
    wr32(&t->cyc_total, (uint32_t)(cyc_now - cyc_wake));
    for (uint32_t h = 1; h < L2S_NHARTS; h++) {
        wr32(&t->cyc_worker[h - 1], worker_cycles[h]);
    }
    fence();
    timing_count_local++;
    wr32(&c->timing_count.v, timing_count_local);

    fence();
    wr32(link_word(L2CPU_LINK_OFF_DONE_SEQ), seq); /* publish */
    fence();
    return 0;
}

static int __attribute__((noinline)) poll_request(void) {
    uint32_t seq = rd32(link_word(L2CPU_LINK_OFF_REQ_SEQ));
    int32_t d = (int32_t)(seq - served_seq);
    if (d == 0) {
        return 0;
    }
    uint64_t mt = plat_mtime(), cy = rdcycle64();
    if (d < 0) {
        fw_error_park(L2S_ERR_SEQ, seq, served_seq);
    }
    fence(); /* consumer: sequence first, then descriptors and data */
    if (d > 1) {
        wr32(&ctrl()->seq_skips.v, rd32(&ctrl()->seq_skips.v) + (uint32_t)(d - 1));
    }
    wr32(&g_hdr->hart_state[0].status, L2CPU_HART_BUSY);
    (void)serve(seq, mt, cy); /* a failure is recorded; the seq is not retried */
    served_seq = seq;
    wr32(&g_hdr->hart_state[0].last_work, seq);
    wr32(&g_hdr->hart_state[0].status, L2CPU_HART_IDLE);
    return 1;
}

/* ---- hooks ---------------------------------------------------------------------------------------------------- */
uint32_t app_id(void) { return L2S_APP_ID; }
uint32_t app_layout_version(void) { return L2S_LAYOUT_VERSION; }
uint32_t app_build_flags(void) {
    return L2S_BUILD_STREAM | (X280S_RVV_BUILD ? L2S_BUILD_RVV : 0u) | (L2S_SAMPLING_FAST ? L2S_BUILD_FAST : 0u);
}

/* COLD: the runtime zeroed [0, 2 MiB) (link block, control, rings). WARM: the windows are as the last image left
 * them; sequence words and ring positions continue, and a request published while the harts were parked
 * (req_seq != done_seq) is served now. */
void app_init(uint32_t boot_mode) {
    (void)boot_mode;
    fpv_on();
    l2s_ctrl_t* c = ctrl();
    l2s_ring_desc_t* r = &c->ring;
    wr32(&r->ring_offset, L2S_OFF_RING);
    wr32(&r->ring_slots, L2S_RING_SLOTS);
    wr32(&r->slot_size, L2S_RING_SLOT_SIZE);
    wr32(&r->tokens_offset, L2S_RS_TOK);
    wr32(&r->max_users, L2S_MAX_USERS);
    wr32(&r->timing_offset, L2S_OFF_TIMING);
    wr32(&r->timing_slots, L2S_TIMING_SLOTS);
    ring_wr_local = rd32(&c->ring_wr.v);
    timing_count_local = rd32(&c->timing_count.v);
    served_seq = rd32(link_word(L2CPU_LINK_OFF_DONE_SEQ));
    validated_ok = 0;
    l2s_uc_gate = 0;
    fence();
    fw_log(
        "sampling app layout %u: done_seq %u req_seq %u ring_wr %u",
        L2S_LAYOUT_VERSION,
        served_seq,
        rd32(link_word(L2CPU_LINK_OFF_REQ_SEQ)),
        ring_wr_local);
}

int app_poll(void) { return poll_request(); }

uint32_t app_mailbox(uint32_t cmd, const uint64_t* arg, uint64_t* rep) {
    (void)cmd;
    (void)arg;
    (void)rep;
    return L2CPU_MB_ERR_CMD;
}
