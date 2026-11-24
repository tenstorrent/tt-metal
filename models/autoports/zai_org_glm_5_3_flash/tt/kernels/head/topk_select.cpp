// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Stage 14b model head, the select core (BRISC): the lm_head readers' top-K candidates (kernels/mk/topk.hpp: per
// reader one landing page, row r at r * K * 8 bytes: K x (key, vocab id) in descending key order) -> the K best per
// token row (a K-way merge, ties: the lower vocab id) -> row r of the outputs [32, 64] uint32 (ROW_MAJOR, one 256-B
// page per row): words 0..K-1 the vocab ids (best first: word 0 = the greedy token), words 32..32+K-1 their bf16 order
// keys. Two outputs with the same rows: `tokens` (the next step's input, MkEmbed) and `sampled` (the readback copy).
// Compile-time args: K, NREAD, TensorAccessorArgs of the outputs. Runtime args: T, tokens address, landing address,
// sampled address, positions address (0 = none: else the decode positions [1, 32] int32 ROW_MAJOR, one 128-B page,
// get +1 on every active row < T (cur_pos >= 0) after the rows are written: the generator contract's in-trace advance).
// Defines: TOPK_SEM (this core's semaphore, NREAD increments), TOPK_LAND_CB (NREAD pages), TOPK_OUT_CB (one 256-B row),
// TOPK_PAGE. TOPK_FAB (TP mesh): the chip's K best per row are all-gathered over the mesh with Stage 14's in-program
// fabric all-reduce (tt/fabric_ar.py): this core is the phase's only producer and consumer; chip q writes its
// candidates into ITS column slot of a [T, 4096] bf16 partial (every other column 0: the sum over the chips is the
// concatenation, exact), as small integers bf16 holds exactly (6 values per candidate: valid 1, key hi / lo byte, id
// bits 12..17 / 6..11 / 0..5), adds the layers' producer count to both lanes' OWN_READY, waits for both lanes' result
// signal, decodes every chip's candidates from the result and merges them -> identical rows on every chip. Runtime args
// (TOPK_FAB) from index 5: rank, lane 0 / 1 NoC (x, y), own, lane sync, res, packed bytes per tile V, bytes of one
// face's valid rows CH, n_prod. Defines: TOPK_FAB_SEM (the lanes' result signal, 2), TOPK_IMG_CB (2 x 64 x V bytes: the
// packed partial of both lanes), TOPK_NCHIP.
// TOPK_SAMPLE (device sampling, the serving contract's samples_in_decode_trace): runtime args at TOPK_SAMPLE_RT = the
// sampler's params page (32 rows x {k, p (fp32 bits), 1 / temperature (fp32 bits), flags}) and its RNG page (32 rows x
// a xoshiro128** state), one 512-B DRAM page each (TOPK_SAMPLE_CB: 1 KB scratch). A row with flags bit 0 draws its
// token from its merged candidates (the K best: top_k 0 / > K = all K): weights exp((v_i - v_0) / temperature) over
// the first min(k, n), top-p keeps the shortest prefix whose weight reaches p of the total (the first candidate
// always), one uniform draw -> the drawn candidate is swapped into word 0 (the next token, the readback); the row's RNG
// advances by one draw and the RNG page is written back. Rows without the flag (greedy: k = 1 / temperature 0) keep
// word 0 = the argmax (ties: the lower id), bit-exact as without TOPK_SAMPLE. Only ACTIVE rows draw (decode position
// >= 0 in the positions page, when one is given): an inactive row's RNG stream does not move while other rows decode
// or prefill (TOPK_SAMPLE_CB: params, RNG, positions = 512 + 512 + 128 bytes).
#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/mk/mk_common.hpp"
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/mk/probe.hpp"  // GLM53_HEAD_PROBE marks

namespace {

constexpr uint32_t K = get_compile_time_arg_val(0);
constexpr uint32_t NREAD = get_compile_time_arg_val(1);
constexpr uint32_t REC_WORDS = TOPK_PAGE / 4;  // one reader's landing page

// row r's K best over the readers' candidate lists (landing pages in L1) -> ids / keys (n returned). A binary max-heap
// of the non-exhausted readers, ordered by their current heads (larger key, then the lower id: every id is in exactly
// one list, so the order is total): a step takes the root's head, reloads that list's next entry and sifts it down
// (~log2 NREAD levels instead of comparing every reader's head).
FORCE_INLINE bool better(uint32_t ka, uint32_t ia, uint32_t kb, uint32_t ib) {
    return ka > kb || (ka == kb && ia < ib);
}

FORCE_INLINE uint32_t merge_readers(volatile tt_l1_ptr uint32_t* land, uint32_t r, uint32_t* ids, uint32_t* keys) {
    uint32_t pos[NREAD], hkey[NREAD], hid[NREAD], heap[NREAD];
    uint32_t m = 0;
    auto load = [&](uint32_t rd) -> bool {  // reader rd's next candidate -> hkey / hid; false: the list is exhausted
        if (pos[rd] >= K) {
            return false;
        }
        const uint32_t w = rd * REC_WORDS + (r * K + pos[rd]) * 2;
        hkey[rd] = land[w];
        hid[rd] = land[w + 1];
        return hid[rd] != 0xffffffffu;  // an empty slot (0, ~0) ends the list
    };
    auto sift = [&](uint32_t i) {  // heap[i] moves down to where no child's head is better than its own
        const uint32_t x = heap[i], xk = hkey[x], xi = hid[x];
        while (true) {
            uint32_t c = 2 * i + 1;
            if (c >= m) {
                break;
            }
            if (c + 1 < m && better(hkey[heap[c + 1]], hid[heap[c + 1]], hkey[heap[c]], hid[heap[c]])) {
                c = c + 1;
            }
            if (!better(hkey[heap[c]], hid[heap[c]], xk, xi)) {
                break;
            }
            heap[i] = heap[c];
            i = c;
        }
        heap[i] = x;
    };
    for (uint32_t rd = 0; rd < NREAD; ++rd) {
        pos[rd] = 0;
        if (load(rd)) {
            heap[m++] = rd;
        }
    }
    for (uint32_t i = m / 2; i-- > 0;) {
        sift(i);
    }
    uint32_t n = 0;
    while (n < K && m > 0) {
        const uint32_t rd = heap[0];
        ids[n] = hid[rd];
        keys[n] = hkey[rd];
        ++n;
        ++pos[rd];
        if (!load(rd)) {
            heap[0] = heap[--m];
        }
        sift(0);
    }
    return n;
}

#if defined(TOPK_SAMPLE)
// the bf16 value of an order key (topk.hpp bf16_key inverted) as fp32
FORCE_INLINE float key_value(uint32_t key) {
    const uint32_t b = ((key & 0x8000u) ? (key & 0x7fffu) : (~key & 0xffffu)) << 16;
    float f;
    __builtin_memcpy(&f, &b, 4);
    return f;
}

// exp(x) for x <= 0: 2^(x log2 e) = 2^n * 2^f, 2^f on [0, 1) by a degree-6 polynomial (rel. error < 2e-7)
FORCE_INLINE float exp_neg(float x) {
    if (x < -87.0f) {
        return 0.0f;
    }
    const float y = x * 1.44269504088896341f;
    int32_t n = static_cast<int32_t>(y);
    if (static_cast<float>(n) > y) {
        n -= 1;
    }
    const float f = y - static_cast<float>(n);
    float q = 1.0f + f * (0.693147180f +
                          f * (0.240226507f + f * (0.0555041087f +
                                                   f * (0.00961812911f + f * (0.00133335581f + f * 0.000154035304f)))));
    uint32_t b;
    __builtin_memcpy(&b, &q, 4);
    b += static_cast<uint32_t>(n) << 23;
    __builtin_memcpy(&q, &b, 4);
    return q;
}

FORCE_INLINE uint32_t rotl32(uint32_t x, uint32_t k) { return (x << k) | (x >> (32 - k)); }

// xoshiro128** (Blackman / Vigna): one 32-bit draw, the state advanced in place
FORCE_INLINE uint32_t xoshiro_next(volatile tt_l1_ptr uint32_t* s) {
    const uint32_t result = rotl32(s[1] * 5u, 7) * 9u;
    const uint32_t t = s[1] << 9;
    s[2] ^= s[0];
    s[3] ^= s[1];
    s[1] ^= s[2];
    s[0] ^= s[3];
    s[2] ^= t;
    s[3] = rotl32(s[3], 11);
    return result;
}

// row r's sampled candidate index (0 = the argmax: greedy rows and degenerate inputs)
FORCE_INLINE uint32_t sample_row(
    volatile tt_l1_ptr uint32_t* par, volatile tt_l1_ptr uint32_t* rng, uint32_t r, const uint32_t* keys, uint32_t n) {
    volatile tt_l1_ptr uint32_t* pr = par + 4 * r;
    volatile tt_l1_ptr int32_t* pos = reinterpret_cast<volatile tt_l1_ptr int32_t*>(par + 256);
    if ((pr[3] & 1u) == 0 || n < 2 || pos[r] < 0) {
        return 0;
    }
    const uint32_t k = pr[0];
    float p, invt;
    {
        const uint32_t pb = pr[1], tb = pr[2];
        __builtin_memcpy(&p, &pb, 4);
        __builtin_memcpy(&invt, &tb, 4);
    }
    const uint32_t kk = (k == 0 || k > n) ? n : k;
    float w[K];
    float total = 0.0f;
    const float v0 = key_value(keys[0]);
    for (uint32_t i = 0; i < kk; ++i) {
        w[i] = exp_neg((key_value(keys[i]) - v0) * invt);
        total += w[i];
    }
    uint32_t m = kk;
    float kept = total;
    if (p < 1.0f) {  // keep candidate i while the weight before it is below p of the total
        const float thr = p * total;
        float pre = 0.0f;
        m = 0;
        for (uint32_t i = 0; i < kk; ++i) {
            if (i > 0 && pre >= thr) {
                break;
            }
            pre += w[i];
            m = i + 1;
        }
        kept = pre;
    }
    const uint32_t u24 = xoshiro_next(rng + 4 * r) >> 8;  // uniform [0, 1) in 2^-24 steps
    const float target = static_cast<float>(u24) * (1.0f / 16777216.0f) * kept;
    float acc = 0.0f;
    for (uint32_t i = 0; i < m; ++i) {
        acc += w[i];
        if (target < acc) {
            return i;
        }
    }
    return m - 1;
}
#endif

template <typename A>
FORCE_INLINE void write_row(
    uint32_t r, const A& tok, const A& smp, bool has_smp, uint32_t* ids, uint32_t* keys, uint32_t n) {
#if defined(TOPK_SAMPLE)
    {
        volatile tt_l1_ptr uint32_t* sp = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(TOPK_SAMPLE_CB));
        const uint32_t j = sample_row(sp, sp + 128, r, keys, n);
        if (j != 0) {  // the drawn candidate becomes word 0 (the next token)
            const uint32_t ti = ids[0], tk = keys[0];
            ids[0] = ids[j];
            keys[0] = keys[j];
            ids[j] = ti;
            keys[j] = tk;
        }
    }
#endif
    volatile tt_l1_ptr uint32_t* row = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(TOPK_OUT_CB));
    for (uint32_t w = 0; w < 64; ++w) {
        row[w] = 0;
    }
    for (uint32_t i = 0; i < n; ++i) {
        row[i] = ids[i];
        row[32 + i] = keys[i];
    }
    noc_async_write_page(r, tok, get_write_ptr(TOPK_OUT_CB));
    if (has_smp) {
        noc_async_write_page(r, smp, get_write_ptr(TOPK_OUT_CB));
    }
    noc_async_write_barrier();
}

}  // namespace

void kernel_main() {
    MK_MARK();  // 0: start
    constexpr auto out_args = TensorAccessorArgs<2>();
    const uint32_t T = get_arg_val<uint32_t>(0);
    const auto tok = TensorAccessor(out_args, get_arg_val<uint32_t>(1));
    const uint32_t smp_addr = get_arg_val<uint32_t>(3);
    const auto smp = TensorAccessor(out_args, smp_addr);

#if defined(TOPK_SAMPLE)
    // the sampler's params + RNG pages -> L1 (written by the host between steps; the RNG page also by this kernel)
    const InterleavedAddrGen<true> spar{.bank_base_address = get_arg_val<uint32_t>(TOPK_SAMPLE_RT), .page_size = 512};
    const InterleavedAddrGen<true> srng{
        .bank_base_address = get_arg_val<uint32_t>(TOPK_SAMPLE_RT + 1), .page_size = 512};
    noc_async_read_page(0, spar, get_write_ptr(TOPK_SAMPLE_CB));
    noc_async_read_page(0, srng, get_write_ptr(TOPK_SAMPLE_CB) + 512);
    if (get_arg_val<uint32_t>(4) != 0) {  // this step's decode positions (advanced below, after the rows)
        const InterleavedAddrGen<true> spos{.bank_base_address = get_arg_val<uint32_t>(4), .page_size = 128};
        noc_async_read_page(0, spos, get_write_ptr(TOPK_SAMPLE_CB) + 1024);
    } else {  // no positions: every row counts as active
        volatile tt_l1_ptr int32_t* p0 =
            reinterpret_cast<volatile tt_l1_ptr int32_t*>(get_write_ptr(TOPK_SAMPLE_CB) + 1024);
        for (uint32_t r = 0; r < 32; ++r) {
            p0[r] = 0;
        }
    }
#endif
    mk::wait_handoff(TOPK_SEM, NREAD);  // every reader's rows landed in DRAM (write barrier before its increment)
    MK_MARK();                          // 1: the readers are done
    const InterleavedAddrGen<true> pages{.bank_base_address = get_arg_val<uint32_t>(2), .page_size = TOPK_PAGE};
    for (uint32_t rd = 0; rd < NREAD; ++rd) {
        noc_async_read_page(rd, pages, get_write_ptr(TOPK_LAND_CB) + rd * TOPK_PAGE);
    }
    noc_async_read_barrier();
    MK_MARK();  // 2: the landing pages are in L1
    volatile tt_l1_ptr uint32_t* land = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(TOPK_LAND_CB));
    uint32_t ids[K], keys[K];
#if !defined(TOPK_FAB)
    for (uint32_t r = 0; r < T; ++r) {
        const uint32_t n = merge_readers(land, r, ids, keys);
        write_row(r, tok, smp, smp_addr != 0, ids, keys, n);
    }
#else
    // ---- the cross-chip gather: encode -> both lanes' own buffers -> OWN_READY += n_prod -> wait -> decode + merge
    uint32_t a = 5;
    const uint32_t rank = get_arg_val<uint32_t>(a++);
    const uint32_t l0x = get_arg_val<uint32_t>(a++), l0y = get_arg_val<uint32_t>(a++);
    const uint32_t l1x = get_arg_val<uint32_t>(a++), l1y = get_arg_val<uint32_t>(a++);
    const uint32_t own = get_arg_val<uint32_t>(a++), lsync = get_arg_val<uint32_t>(a++);
    const uint32_t lres = get_arg_val<uint32_t>(a++), V = get_arg_val<uint32_t>(a++), CH = get_arg_val<uint32_t>(a++);
    const uint32_t n_prod = get_arg_val<uint32_t>(a++);
    constexpr uint32_t FIELDS = 6;
    const uint32_t img = get_write_ptr(TOPK_IMG_CB);  // lane 0 tiles 0..63 then lane 1 tiles 0..63, V bytes each
    const uint64_t zeros = get_noc_addr(MEM_ZEROS_BASE);
    for (uint32_t off = 0; off < 2 * 64 * V; off += MEM_ZEROS_SIZE) {
        noc_async_read(zeros, img + off, MEM_ZEROS_SIZE < 2 * 64 * V - off ? MEM_ZEROS_SIZE : 2 * 64 * V - off);
    }
    noc_async_read_barrier();
    // bf16 of a small non-negative integer n (< 256): exact. Integer only (BRISC has no FPU: a float conversion is
    // a libgcc call): exponent 127 + e for the leading bit e, the 7 bits below it as the mantissa
    auto enc = [](uint32_t n) -> uint16_t {
        if (n == 0) {
            return 0;
        }
        const uint32_t e = 31u - static_cast<uint32_t>(__builtin_clz(n));
        return static_cast<uint16_t>(((127u + e) << 7) | ((n << (7u - e)) & 0x7fu));
    };
    // element (row r, column c) of the packed partial: tile c / 32 (lane = tile / 64), face (c % 32) / 16. The lanes
    // carry only the layers' BR = CH / 32 valid rows of each tile (the FabricPhase's packed row count, e.g. 4 for the
    // 4-row chunks) while the head has T rows (up to 32): row r goes to packed row r % BR of row group r / BR, and row
    // group g to the columns g * GROUP.. (Stage 14b fix, 2026-10-20: rows >= BR used to land in the next face / tile).
    // Whole tiles (the phase's B > 16: CH = 1024, V = one tile): the 4 faces in the tile's own order, every row.
    const bool whole = CH >= 1024;
    const uint32_t BR = whole ? 32 : CH / 32;
    constexpr uint32_t GROUP = TOPK_NCHIP * K * FIELDS;  // the columns of one row group (every chip's candidates)
    // row r's part of an element address, once per row (no division in the per-field path): the first column of its
    // row group, the offset of its packed row r % BR inside a face
    struct RowAt {
        uint32_t gcol, roff;
    };
    auto row_at = [&](uint32_t r) -> RowAt {
        const uint32_t rr = r % BR;
        return {(r / BR) * GROUP, whole ? ((rr >> 4) << 1) * 512 + (rr & 15) * 32 : rr * 32};
    };
    auto slot = [&](const RowAt& ra, uint32_t c0_) -> volatile tt_l1_ptr uint16_t* {
        const uint32_t c = c0_ + ra.gcol;
        const uint32_t t = c >> 5;
        const uint32_t off = ra.roff + ((c & 31) >> 4) * (whole ? 512u : CH) + (c & 15) * 2;
        return reinterpret_cast<volatile tt_l1_ptr uint16_t*>(img + (t >> 6) * 64 * V + (t & 63) * V + off);
    };
    for (uint32_t r = 0; r < T; ++r) {
        const uint32_t n = merge_readers(land, r, ids, keys);
        const RowAt ra = row_at(r);
        for (uint32_t i = 0; i < n; ++i) {
            const uint32_t c0 = (rank * K + i) * FIELDS, key = keys[i], id = ids[i];
            const uint32_t v[FIELDS] = {1, key >> 8, key & 0xff, (id >> 12) & 63, (id >> 6) & 63, id & 63};
            for (uint32_t j = 0; j < FIELDS; ++j) {
                *slot(ra, c0 + j) = enc(v[j]);
            }
        }
    }
    MK_MARK();  // 3 (gather): readers merged + encoded
    noc_async_write(img, get_noc_addr(l0x, l0y, own), 64 * V);
    noc_async_write(img + 64 * V, get_noc_addr(l1x, l1y, own), 64 * V);
    noc_async_write_barrier();
    noc_semaphore_inc(get_noc_addr(l0x, l0y, lsync), n_prod);  // OWN_READY: the layers' producer count per iteration
    noc_semaphore_inc(get_noc_addr(l1x, l1y, lsync), n_prod);
    noc_async_atomic_barrier();
    MK_MARK();                          // 4: the partial sent
    mk::wait_handoff(TOPK_FAB_SEM, 2);  // both lanes: the reduced partial is in res
    MK_MARK();                          // 5: the lanes are done
    // tiles holding candidates (all in lane 0: MkHead checks (T / BR groups) x GROUP <= 64 x 32 columns)
    const uint32_t USED = (((T + BR - 1) / BR) * GROUP + 31) / 32;
    noc_async_read(get_noc_addr(l0x, l0y, lres), img, USED * V);
    noc_async_read_barrier();
    MK_MARK();  // 6: the gathered candidates are in L1
    // the integer value of a bf16 holding a small non-negative integer (enc's inverse; integer only, truncating like
    // the float -> uint conversion it replaces)
    auto dec = [&](const RowAt& ra, uint32_t c) -> uint32_t {
        const uint32_t b = static_cast<uint32_t>(*slot(ra, c));
        const uint32_t ex = (b >> 7) & 0xffu;
        if ((b & 0x8000u) || ex < 127u) {
            return 0;
        }
        const uint32_t m = (b & 0x7fu) | 0x80u, e = ex - 127u;
        return e <= 7u ? m >> (7u - e) : (e < 31u ? m << (e - 7u) : 0u);
    };
    for (uint32_t r = 0; r < T; ++r) {
        // K-way merge over the chips' (descending) candidate lists; each list's head decoded once into hkey / hid
        // (hid ~0: no more candidates from that chip)
        uint32_t head[TOPK_NCHIP], hkey[TOPK_NCHIP], hid[TOPK_NCHIP];
        const RowAt ra = row_at(r);
        auto load = [&](uint32_t q) {
            const uint32_t c0 = (q * K + head[q]) * FIELDS;
            if (head[q] >= K || dec(ra, c0) != 1) {
                hkey[q] = 0;
                hid[q] = 0xffffffffu;
                return;
            }
            hkey[q] = (dec(ra, c0 + 1) << 8) | dec(ra, c0 + 2);
            hid[q] = (dec(ra, c0 + 3) << 12) | (dec(ra, c0 + 4) << 6) | dec(ra, c0 + 5);
        };
        for (uint32_t q = 0; q < TOPK_NCHIP; ++q) {
            head[q] = 0;
            load(q);
        }
        uint32_t n = 0;
        for (uint32_t i = 0; i < K; ++i) {
            uint32_t best = TOPK_NCHIP, bkey = 0, bid = 0xffffffffu;
            for (uint32_t q = 0; q < TOPK_NCHIP; ++q) {
                if (hid[q] == 0xffffffffu) {
                    continue;
                }
                if (best == TOPK_NCHIP || hkey[q] > bkey || (hkey[q] == bkey && hid[q] < bid)) {
                    best = q;
                    bkey = hkey[q];
                    bid = hid[q];
                }
            }
            if (best == TOPK_NCHIP) {
                break;
            }
            ids[i] = bid;
            keys[i] = bkey;
            ++head[best];
            load(best);
            n = i + 1;
        }
        write_row(r, tok, smp, smp_addr != 0, ids, keys, n);
    }
#endif
    MK_MARK();  // 3 (local) / 7 (gather): every row merged and written
#if defined(TOPK_SAMPLE)
    noc_async_write_page(0, srng, get_write_ptr(TOPK_SAMPLE_CB) + 512);  // the advanced RNG states
    noc_async_write_barrier();
#endif
    const uint32_t pos_addr = get_arg_val<uint32_t>(4);
    if (pos_addr != 0) {  // decode positions: +1 on the active rows (after this step's sampling)
        const InterleavedAddrGen<true> pos{.bank_base_address = pos_addr, .page_size = 128};
        const uint32_t buf = get_write_ptr(TOPK_OUT_CB);
        noc_async_read_page(0, pos, buf);
        noc_async_read_barrier();
        volatile tt_l1_ptr int32_t* p = reinterpret_cast<volatile tt_l1_ptr int32_t*>(buf);
        for (uint32_t r = 0; r < T; ++r) {
            if (p[r] >= 0) {
                p[r] = p[r] + 1;
            }
        }
        noc_async_write_page(0, pos, buf);
#if defined(TOPK_POS_COPIES)
        // per-chunk copies (Stage 14b sparse_dsa: MkMla reads cur_pos rows 0..3 of ITS chunk's tensor): copy c gets
        // rows 4c..4c+3 at its rows 0..3 (the rest of its page: the following rows, never read)
        for (uint32_t c = 0; c < TOPK_POS_COPIES; ++c) {
            const InterleavedAddrGen<true> dst{
                .bank_base_address = get_arg_val<uint32_t>(TOPK_POS_COPY_RT + c), .page_size = 128};
            noc_async_write(buf + 16 * c, get_noc_addr(0, dst), 128);
        }
#endif
        noc_async_write_barrier();
    }
    MK_MARK();  // last: positions advanced
#if defined(MK_PROBE_ARG)
    MK_FLUSH(get_write_ptr(MK_PROBE_CB));
#endif
}
