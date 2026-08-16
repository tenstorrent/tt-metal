// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// risc_scan_bench — Gate-2 deciding microbenchmark kernel.
//
// Measures the scalar-RISC (BRISC / NCRISC) scan/emit rate "X" over bf16 data
// resident in L1. This one constant prices every RISC-side materialization
// candidate of the top-k selector campaign (dense emission, compressed-stream
// consumer, mask-map gather, tile skipping).
//
// One kernel source, compiled per-variant via compile-time arg VARIANT:
//   v1  256-bin high-byte histogram, counters in core-local data RAM (static array)
//   v2  identical loop, counters in L1 (prices the ~5-cyc non-coalesced L1 store rule)
//   v3  v1 with 4 interleaved sub-histograms + timed 256x4 merge (store-queue-drain
//       mitigation for same-bin/clustered inputs)
//   v4  threshold-compare dense emit: sign-magnitude XOR map, survivor -> store
//       (value_bits, u32 global index) pairs at a bump pointer, cap + overflow flag
//   v5  skip-zero-words consumer over a host-synthesized compressed stream
//       (groups of 32 u32 fused-key datums + 16 B counter block skipped by address
//       arithmetic — the candidate-(e) consumer inner loop)
//   v6  dual-RISC split: BRISC scans the first half, NCRISC the second, both v1,
//       semaphore rendezvous so both inner loops overlap (L1 port contention shot)
//   v7  pure-load floor: high-byte loads (stride 2) summed, no histogram store
//
// Timing: RISCV_DEBUG_REG_WALL_CLOCK_L read before/after the inner loop only
// (data staged and barriered first; one untimed warmup pass precedes the timed
// pass so i-fetch effects are excluded). Cycles are tensix-clock ticks.
//
// Every variant writes {cycles, checksum, count, overflow, magic, elems} to a
// result region plus (histogram variants) the full 256-bin histogram; the host
// verifies bit-exactly against a CPU reference.
//
// Idioms borrowed from:
//   - reader_masked_bincount.cpp (dual-compilation, semaphore choreography,
//     L1-drain-before-signal rule)
//   - device_delay_spin.cpp (wall-clock read on a data-movement RISC)
//   - loopback_dram_copy.cpp (TensorAccessor page staging)

#include <cstdint>

// ---------------------------------------------------------------------------
// Timing helpers
// ---------------------------------------------------------------------------
static inline uint32_t bench_rdclk() {
    // Reading WALL_CLOCK_L samples the 64-bit counter; the low 32 bits are all
    // we need (loops here are << 2^32 cycles).
    return *reinterpret_cast<volatile uint32_t tt_reg_ptr*>(RISCV_DEBUG_REG_WALL_CLOCK_L);
}
// Compiler fence: keep the timed loop's memory ops from drifting across the
// clock reads.
#define BENCH_FENCE() asm volatile("" ::: "memory")

// ---------------------------------------------------------------------------
// Core-local data RAM histogram storage.
// BRISC/NCRISC each have 8 KiB local data RAM (MEM_BRISC_LOCAL_SIZE); statics in
// kernel .bss land there. 4 sub-histograms x 256 u32 = 4 KiB. If the JIT link
// ever reports local-memory overflow, drop NSUB_HIST to 2 (see RUNBOOK).
// ---------------------------------------------------------------------------
constexpr uint32_t NSUB_HIST = 4;
static uint32_t s_hist[NSUB_HIST * 256];

static void zero_u32(uint32_t* h, uint32_t n) {
    for (uint32_t i = 0; i < n; i++) {
        h[i] = 0;
    }
}

// ---------------------------------------------------------------------------
// v1/v2/v6 inner loop: single 256-bin high-byte histogram.
// Input is scanned as u32 words (2 bf16 elements each); high byte of the low
// element is bits [15:8], of the high element bits [31:24]. 8 words (16
// elements) per iteration, loads batched up front so the compiler can schedule
// them into the 8-entry retire queue and hide L0-dcache misses.
// ---------------------------------------------------------------------------
static void hist_scan1(const uint32_t* p, uint32_t nwords, uint32_t* h) {
    for (uint32_t i = 0; i < nwords; i += 8) {
        uint32_t w0 = p[i + 0], w1 = p[i + 1], w2 = p[i + 2], w3 = p[i + 3];
        uint32_t w4 = p[i + 4], w5 = p[i + 5], w6 = p[i + 6], w7 = p[i + 7];
        h[(w0 >> 8) & 0xFFu]++;
        h[w0 >> 24]++;
        h[(w1 >> 8) & 0xFFu]++;
        h[w1 >> 24]++;
        h[(w2 >> 8) & 0xFFu]++;
        h[w2 >> 24]++;
        h[(w3 >> 8) & 0xFFu]++;
        h[w3 >> 24]++;
        h[(w4 >> 8) & 0xFFu]++;
        h[w4 >> 24]++;
        h[(w5 >> 8) & 0xFFu]++;
        h[w5 >> 24]++;
        h[(w6 >> 8) & 0xFFu]++;
        h[w6 >> 24]++;
        h[(w7 >> 8) & 0xFFu]++;
        h[w7 >> 24]++;
    }
}

// ---------------------------------------------------------------------------
// v3 inner loop: 4 interleaved sub-histograms (element (2i+j) -> sub-hist
// (2i+j) % 4). Consecutive updates of the same logical bin are separated by
// three other counter updates, sidestepping the store-queue drain-on-overlap
// rule for same-bin (clustered / all-equal) inputs.
// ---------------------------------------------------------------------------
static void hist_scan4(const uint32_t* p, uint32_t nwords, uint32_t* h) {
    for (uint32_t i = 0; i < nwords; i += 8) {
        uint32_t w0 = p[i + 0], w1 = p[i + 1], w2 = p[i + 2], w3 = p[i + 3];
        uint32_t w4 = p[i + 4], w5 = p[i + 5], w6 = p[i + 6], w7 = p[i + 7];
        // word k holds elements 2k (low half -> sub (2k)%4) and 2k+1 (high half
        // -> sub (2k+1)%4). For even k: subs 0 and 1; odd k: subs 2 and 3.
        h[0 * 256 + ((w0 >> 8) & 0xFFu)]++;
        h[1 * 256 + (w0 >> 24)]++;
        h[2 * 256 + ((w1 >> 8) & 0xFFu)]++;
        h[3 * 256 + (w1 >> 24)]++;
        h[0 * 256 + ((w2 >> 8) & 0xFFu)]++;
        h[1 * 256 + (w2 >> 24)]++;
        h[2 * 256 + ((w3 >> 8) & 0xFFu)]++;
        h[3 * 256 + (w3 >> 24)]++;
        h[0 * 256 + ((w4 >> 8) & 0xFFu)]++;
        h[1 * 256 + (w4 >> 24)]++;
        h[2 * 256 + ((w5 >> 8) & 0xFFu)]++;
        h[3 * 256 + (w5 >> 24)]++;
        h[0 * 256 + ((w6 >> 8) & 0xFFu)]++;
        h[1 * 256 + (w6 >> 24)]++;
        h[2 * 256 + ((w7 >> 8) & 0xFFu)]++;
        h[3 * 256 + (w7 >> 24)]++;
    }
}

// ---------------------------------------------------------------------------
// v4 inner loop: threshold-compare dense emit.
// Sign-magnitude XOR map: key = b ^ (0x8000 + (b>>15)*0x7FFF) — maps bf16 bit
// patterns to a monotone unsigned total order. Survivor iff key > t_mapped.
// Survivors: (raw bits, u32 global index) pairs at a bump pointer, capped at
// `cap` entries; counting continues past the cap (loud overflow, never silent
// truncation). Checksum accumulates over ALL survivors regardless of cap.
// ---------------------------------------------------------------------------
static void dense_emit(
    const uint32_t* p,
    uint32_t nwords,
    uint32_t start_elem,
    uint32_t t_mapped,
    uint32_t* emit_base,
    uint32_t cap,
    uint32_t& count_out,
    uint32_t& csum_out) {
    uint32_t cnt = 0;
    uint32_t cs = 0;
    uint32_t* op = emit_base;
    uint32_t idx = start_elem;
    for (uint32_t i = 0; i < nwords; i += 4) {
        uint32_t w0 = p[i + 0], w1 = p[i + 1], w2 = p[i + 2], w3 = p[i + 3];
#define BENCH_EMIT_HALF(w, shift, off)                    \
    {                                                     \
        uint32_t b = ((w) >> (shift)) & 0xFFFFu;          \
        uint32_t k = b ^ (0x8000u + (b >> 15) * 0x7FFFu); \
        if (k > t_mapped) {                               \
            cs += b + (idx + (off));                      \
            if (cnt < cap) {                              \
                op[0] = b;                                \
                op[1] = idx + (off);                      \
                op += 2;                                  \
            }                                             \
            cnt++;                                        \
        }                                                 \
    }
        BENCH_EMIT_HALF(w0, 0, 0)
        BENCH_EMIT_HALF(w0, 16, 1)
        BENCH_EMIT_HALF(w1, 0, 2)
        BENCH_EMIT_HALF(w1, 16, 3)
        BENCH_EMIT_HALF(w2, 0, 4)
        BENCH_EMIT_HALF(w2, 16, 5)
        BENCH_EMIT_HALF(w3, 0, 6)
        BENCH_EMIT_HALF(w3, 16, 7)
#undef BENCH_EMIT_HALF
        idx += 8;
    }
    count_out = cnt;
    csum_out = cs;
}

// ---------------------------------------------------------------------------
// v5 inner loop: candidate-(e) compressed-stream consumer.
// Stream = groups of 32 u32 datums followed by a 16 B counter block (skipped by
// address arithmetic — fused keys make the metadata unnecessary). Datum:
// (bf16_bits << 16) | (chunk_local_idx + 1); zero word = placeholder. The +1
// offset guarantees no survivor is the zero word.
// ---------------------------------------------------------------------------
static void sparse_consume(
    const uint32_t* base,
    uint32_t ngroups,
    uint32_t idx_base,
    uint32_t** op_io,
    uint32_t cap,
    uint32_t& cnt_io,
    uint32_t& cs_io,
    uint32_t* emit_base) {
    uint32_t cnt = cnt_io;
    uint32_t cs = cs_io;
    uint32_t* op = *op_io;
    const uint32_t* p = base;
    for (uint32_t g = 0; g < ngroups; g++) {
        for (uint32_t j = 0; j < 32; j += 4) {
            uint32_t w0 = p[j + 0], w1 = p[j + 1], w2 = p[j + 2], w3 = p[j + 3];
#define BENCH_CONSUME(w)                              \
    if (w) {                                          \
        uint32_t v = (w) >> 16;                       \
        uint32_t ix = idx_base + ((w) & 0xFFFFu) - 1; \
        cs += v + ix;                                 \
        if (cnt < cap) {                              \
            op[0] = v;                                \
            op[1] = ix;                               \
            op += 2;                                  \
        }                                             \
        cnt++;                                        \
    }
            BENCH_CONSUME(w0)
            BENCH_CONSUME(w1)
            BENCH_CONSUME(w2)
            BENCH_CONSUME(w3)
#undef BENCH_CONSUME
        }
        p += 36;  // 32 datums + 4 counter words (16 B) skipped, no decode
    }
    (void)emit_base;
    cnt_io = cnt;
    cs_io = cs;
    *op_io = op;
}

// ---------------------------------------------------------------------------
// v7 inner loop: pure-load floor. High-byte loads at stride 2, summed into 4
// independent accumulators (distinct dest registers per the retire-queue rule).
// ---------------------------------------------------------------------------
static uint32_t load_sum(const uint8_t* q, uint32_t nelems) {
    uint32_t s0 = 0, s1 = 0, s2 = 0, s3 = 0;
    // q points at the first HIGH byte (input base + 1); element e's high byte
    // is q[2*e]. 16 elements per iteration.
    for (uint32_t i = 0; i < 2 * nelems; i += 32) {
        s0 += q[i + 0];
        s1 += q[i + 2];
        s2 += q[i + 4];
        s3 += q[i + 6];
        s0 += q[i + 8];
        s1 += q[i + 10];
        s2 += q[i + 12];
        s3 += q[i + 14];
        s0 += q[i + 16];
        s1 += q[i + 18];
        s2 += q[i + 20];
        s3 += q[i + 22];
        s0 += q[i + 24];
        s1 += q[i + 26];
        s2 += q[i + 28];
        s3 += q[i + 30];
    }
    return s0 + s1 + s2 + s3;
}

void kernel_main() {
    // ------------------------------ compile-time args ------------------------
    constexpr uint32_t VARIANT = get_compile_time_arg_val(0);     // 1..7
    constexpr uint32_t IS_BRISC = get_compile_time_arg_val(1);    // 1 = BRISC
    constexpr uint32_t START_ELEM = get_compile_time_arg_val(2);  // this RISC's slice start
    constexpr uint32_t NUM_ELEMS = get_compile_time_arg_val(3);   // this RISC's slice length (mult of 16)
    constexpr uint32_t IN_OFF = get_compile_time_arg_val(4);      // L1 region offsets from l1_base
    constexpr uint32_t SPARSE_OFF = get_compile_time_arg_val(5);
    constexpr uint32_t EMIT_OFF = get_compile_time_arg_val(6);
    constexpr uint32_t RESULT_OFF = get_compile_time_arg_val(7);
    constexpr uint32_t HIST_OFF = get_compile_time_arg_val(8);
    constexpr uint32_t EMIT_CAP = get_compile_time_arg_val(9);  // max emitted pairs
    constexpr uint32_t SEM_ID = get_compile_time_arg_val(10);
    constexpr uint32_t STAGE_PAGE = get_compile_time_arg_val(11);  // staging page size (bytes)

    // ------------------------------ runtime args -----------------------------
    const uint32_t l1_base = get_arg_val<uint32_t>(0);
    const uint32_t in_dram = get_arg_val<uint32_t>(1);
    const uint32_t sparse_dram = get_arg_val<uint32_t>(2);
    const uint32_t res_dram = get_arg_val<uint32_t>(3);
    const uint32_t hist_dram = get_arg_val<uint32_t>(4);
    const uint32_t emit_dram = get_arg_val<uint32_t>(5);
    const uint32_t t_mapped = get_arg_val<uint32_t>(6);
    const uint32_t seg0_groups = get_arg_val<uint32_t>(7);
    const uint32_t seg1_groups = get_arg_val<uint32_t>(8);
    const uint32_t in_pages = get_arg_val<uint32_t>(9);
    const uint32_t sparse_pages = get_arg_val<uint32_t>(10);

    constexpr uint32_t ACC_BASE = 12;
    constexpr auto in_args = TensorAccessorArgs<ACC_BASE>();
    const auto in_acc = TensorAccessor(in_args, in_dram);
    constexpr auto sp_args = TensorAccessorArgs<in_args.next_compile_time_args_offset()>();
    const auto sp_acc = TensorAccessor(sp_args, sparse_dram);
    constexpr auto res_args = TensorAccessorArgs<sp_args.next_compile_time_args_offset()>();
    const auto res_acc = TensorAccessor(res_args, res_dram);
    constexpr auto hist_args = TensorAccessorArgs<res_args.next_compile_time_args_offset()>();
    const auto hist_acc = TensorAccessor(hist_args, hist_dram);
    constexpr auto emit_args = TensorAccessorArgs<hist_args.next_compile_time_args_offset()>();
    const auto emit_acc = TensorAccessor(emit_args, emit_dram);

    const uint32_t sem_addr = get_semaphore(SEM_ID);
    // Some of these are unused on the NCRISC compile / in some variants.
    (void)sem_addr;
    (void)in_acc;
    (void)sp_acc;
    (void)res_acc;
    (void)hist_acc;
    (void)emit_acc;
    (void)t_mapped;
    (void)seg0_groups;
    (void)seg1_groups;
    (void)in_pages;
    (void)sparse_pages;

    // ------------------------------ staging (BRISC) --------------------------
    if constexpr (IS_BRISC == 1) {
        // Zero the result region before anyone runs (stale-data guard).
        volatile tt_l1_ptr uint32_t* res_zero = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l1_base + RESULT_OFF);
        for (uint32_t i = 0; i < 64; i++) {
            res_zero[i] = 0;
        }
        if constexpr (VARIANT == 5) {
            for (uint32_t i = 0; i < sparse_pages; i++) {
                noc_async_read_page(i, sp_acc, l1_base + SPARSE_OFF + i * STAGE_PAGE);
            }
        } else {
            for (uint32_t i = 0; i < in_pages; i++) {
                noc_async_read_page(i, in_acc, l1_base + IN_OFF + i * STAGE_PAGE);
            }
        }
        noc_async_read_barrier();
    }

    // ------------------------------ v6 rendezvous ----------------------------
    // Both-arrive barrier so the two inner loops overlap (measures dual-RISC L1
    // read-port contention). BRISC only increments AFTER its staging barrier,
    // so NCRISC can never scan unstaged data.
    if constexpr (VARIANT == 6) {
        noc_semaphore_inc(get_noc_addr(my_x[noc_index], my_y[noc_index], sem_addr), 1);
        noc_async_atomic_barrier();
        noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem_addr), 2);
    }

    // ------------------------------ the timed scan ---------------------------
    uint32_t cycles = 0;
    uint32_t csum = 0;
    uint32_t count = 0;
    uint32_t overflow = 0;
    uint32_t elems = NUM_ELEMS;

    const uint32_t* in_words = reinterpret_cast<const uint32_t*>(l1_base + IN_OFF + START_ELEM * 2);
    constexpr uint32_t nwords = NUM_ELEMS / 2;

    if constexpr (VARIANT == 1 || VARIANT == 6) {
        zero_u32(s_hist, 256);
        hist_scan1(in_words, nwords, s_hist);  // warmup (i-fetch, L0 behavior)
        zero_u32(s_hist, 256);
        BENCH_FENCE();
        uint32_t t0 = bench_rdclk();
        BENCH_FENCE();
        hist_scan1(in_words, nwords, s_hist);
        BENCH_FENCE();
        uint32_t t1 = bench_rdclk();
        BENCH_FENCE();
        cycles = t1 - t0;
        uint32_t* hl1 = reinterpret_cast<uint32_t*>(l1_base + HIST_OFF + (IS_BRISC ? 0 : 1024));
        for (uint32_t b = 0; b < 256; b++) {
            hl1[b] = s_hist[b];
            csum += s_hist[b] * (b + 1);
        }
        count = NUM_ELEMS;
    } else if constexpr (VARIANT == 2) {
        // Histogram lives directly in the L1 hist region — prices the 5-cyc
        // non-coalesced L1 store throughput rule.
        uint32_t* hl1 = reinterpret_cast<uint32_t*>(l1_base + HIST_OFF);
        zero_u32(hl1, 256);
        hist_scan1(in_words, nwords, hl1);  // warmup
        zero_u32(hl1, 256);
        BENCH_FENCE();
        uint32_t t0 = bench_rdclk();
        BENCH_FENCE();
        hist_scan1(in_words, nwords, hl1);
        BENCH_FENCE();
        uint32_t t1 = bench_rdclk();
        BENCH_FENCE();
        cycles = t1 - t0;
        for (uint32_t b = 0; b < 256; b++) {
            csum += hl1[b] * (b + 1);
        }
        count = NUM_ELEMS;
    } else if constexpr (VARIANT == 3) {
        zero_u32(s_hist, NSUB_HIST * 256);
        hist_scan4(in_words, nwords, s_hist);  // warmup
        zero_u32(s_hist, NSUB_HIST * 256);
        BENCH_FENCE();
        uint32_t t0 = bench_rdclk();
        BENCH_FENCE();
        hist_scan4(in_words, nwords, s_hist);
        // The 256x4 merge is part of the mitigation's true cost — timed.
        for (uint32_t b = 0; b < 256; b++) {
            s_hist[b] += s_hist[256 + b] + s_hist[512 + b] + s_hist[768 + b];
        }
        BENCH_FENCE();
        uint32_t t1 = bench_rdclk();
        BENCH_FENCE();
        cycles = t1 - t0;
        uint32_t* hl1 = reinterpret_cast<uint32_t*>(l1_base + HIST_OFF);
        for (uint32_t b = 0; b < 256; b++) {
            hl1[b] = s_hist[b];
            csum += s_hist[b] * (b + 1);
        }
        count = NUM_ELEMS;
    } else if constexpr (VARIANT == 4) {
        uint32_t* emit_base = reinterpret_cast<uint32_t*>(l1_base + EMIT_OFF);
        uint32_t wu_count = 0, wu_csum = 0;
        dense_emit(in_words, nwords, START_ELEM, t_mapped, emit_base, EMIT_CAP, wu_count, wu_csum);  // warmup
        BENCH_FENCE();
        uint32_t t0 = bench_rdclk();
        BENCH_FENCE();
        dense_emit(in_words, nwords, START_ELEM, t_mapped, emit_base, EMIT_CAP, count, csum);
        BENCH_FENCE();
        uint32_t t1 = bench_rdclk();
        BENCH_FENCE();
        cycles = t1 - t0;
        overflow = (count > EMIT_CAP) ? 1 : 0;
    } else if constexpr (VARIANT == 5) {
        const uint32_t* seg0 = reinterpret_cast<const uint32_t*>(l1_base + SPARSE_OFF);
        const uint32_t* seg1 = seg0 + seg0_groups * 36;
        uint32_t* emit_base = reinterpret_cast<uint32_t*>(l1_base + EMIT_OFF);
        elems = (seg0_groups + seg1_groups) * 32;  // datum words scanned
        {
            // warmup
            uint32_t wu_cnt = 0, wu_cs = 0;
            uint32_t* wu_op = emit_base;
            sparse_consume(seg0, seg0_groups, 0, &wu_op, EMIT_CAP, wu_cnt, wu_cs, emit_base);
            sparse_consume(seg1, seg1_groups, 32768, &wu_op, EMIT_CAP, wu_cnt, wu_cs, emit_base);
        }
        uint32_t* op = emit_base;
        BENCH_FENCE();
        uint32_t t0 = bench_rdclk();
        BENCH_FENCE();
        sparse_consume(seg0, seg0_groups, 0, &op, EMIT_CAP, count, csum, emit_base);
        sparse_consume(seg1, seg1_groups, 32768, &op, EMIT_CAP, count, csum, emit_base);
        BENCH_FENCE();
        uint32_t t1 = bench_rdclk();
        BENCH_FENCE();
        cycles = t1 - t0;
        overflow = (count > EMIT_CAP) ? 1 : 0;
    } else if constexpr (VARIANT == 7) {
        const uint8_t* q = reinterpret_cast<const uint8_t*>(l1_base + IN_OFF + START_ELEM * 2) + 1;
        uint32_t wu = load_sum(q, NUM_ELEMS);  // warmup
        BENCH_FENCE();
        uint32_t t0 = bench_rdclk();
        BENCH_FENCE();
        csum = load_sum(q, NUM_ELEMS);
        BENCH_FENCE();
        uint32_t t1 = bench_rdclk();
        BENCH_FENCE();
        cycles = t1 - t0;
        // keep the warmup result live so it cannot be optimized out
        if (wu != csum) {
            csum = 0xDEADBEEFu;  // warmup/timed disagree on identical data: loud failure
        }
        count = NUM_ELEMS;
    }

    // ------------------------------ result write -----------------------------
    volatile tt_l1_ptr uint32_t* res =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l1_base + RESULT_OFF) + (IS_BRISC ? 0 : 16);
    res[0] = cycles;
    res[1] = csum;
    res[2] = count;
    res[3] = overflow;
    res[4] = 0xBE9C0000u | (VARIANT << 8) | IS_BRISC;  // sanity magic
    res[5] = elems;

    if constexpr (VARIANT == 6 && IS_BRISC == 0) {
        // NCRISC: drain L1 stores before the MMIO/NoC semaphore signal (a store
        // can retire before its write lands; the signal must not race ahead —
        // same rule as masked_bincount's load_blocking before gather_sem.up).
        volatile tt_l1_ptr uint32_t* drain =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l1_base + HIST_OFF + 1024 + 255 * 4);
        (void)*drain;
        (void)res[5];
        noc_semaphore_inc(get_noc_addr(my_x[noc_index], my_y[noc_index], sem_addr), 1);
        noc_async_atomic_barrier();
        return;
    }

    if constexpr (IS_BRISC == 1) {
        if constexpr (VARIANT == 6) {
            // Wait for NCRISC's results to be in L1 before shipping to DRAM.
            noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem_addr), 3);
        }
        noc_async_write_page(0, res_acc, l1_base + RESULT_OFF);
        noc_async_write_page(0, hist_acc, l1_base + HIST_OFF);
        noc_async_write_page(0, emit_acc, l1_base + EMIT_OFF);
        noc_async_write_barrier();
    }
}
