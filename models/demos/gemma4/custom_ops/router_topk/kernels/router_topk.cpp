// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
// RouterTopk: one data-movement kernel on 32 cores. Core u (< B) reads the logit row tile, ranks ITS row (top-k of
// the bf16 logits as floats), takes the k exponentials against the row max (fp32 scalar code), normalizes by their
// sum, scales by the per-expert scale and sends its local-expert weight vector (E_loc bf16) to slot u of core 0's
// gather buffer, then bumps core 0's semaphore. Core 0 waits for B arrivals, resets the semaphore (trace replay),
// transposes the slots into the [E_loc, 32] weight rows, writes them and the sparsity row.
//
// PACKED mode (compile-time arg after the accessor args, = 1): no gather and no semaphore. Every core u writes row u
// of the packed routing [32, 2K] bf16 ROW_MAJOR (page u = W_PAGE bytes): columns [0, K) = the user's normalized top-k
// weights times the per-expert scale (descending), columns [K, 2K) = the selected GLOBAL expert ids as bf16 (exact
// below 256); rows >= B (no user) get weights 0 and ids 0..K-1 (distinct ids: the dispatch kernel keys its slots by
// id, a row of equal ids hangs it). This is the (values, indices) form the decode MoE kernel path consumes; `base`
// and `sp_out` are unused (read / bound only).
//
// compile-time args: [cb_logits, cb_scratch, cb_gather, cols_t, E, E_loc, K, pes_page, base_page, w_page, sp_page,
//                     slot_bytes, sem_id, leader_noc_x, leader_noc_y]
//                    + TensorAccessorArgs(logits, pes, base, w_out, sp_out)
// runtime args: [logits_addr, pes_addr, base_addr, w_addr, sp_addr, my_user, B]
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/tensor/noc_traits.h"

constexpr uint32_t CB_LOGITS = get_compile_time_arg_val(0);
constexpr uint32_t CB_SCRATCH = get_compile_time_arg_val(1);
constexpr uint32_t CB_GATHER = get_compile_time_arg_val(2);
constexpr uint32_t COLS_T = get_compile_time_arg_val(3);
constexpr uint32_t E = get_compile_time_arg_val(4);
constexpr uint32_t E_LOC = get_compile_time_arg_val(5);
constexpr uint32_t K = get_compile_time_arg_val(6);
constexpr uint32_t PES_PAGE = get_compile_time_arg_val(7);
constexpr uint32_t BASE_PAGE = get_compile_time_arg_val(8);
constexpr uint32_t W_PAGE = get_compile_time_arg_val(9);
constexpr uint32_t SP_PAGE = get_compile_time_arg_val(10);
constexpr uint32_t SLOT = get_compile_time_arg_val(11);
constexpr uint32_t SEM_ID = get_compile_time_arg_val(12);
constexpr uint32_t LEADER_X = get_compile_time_arg_val(13);
constexpr uint32_t LEADER_Y = get_compile_time_arg_val(14);
constexpr auto logits_args = TensorAccessorArgs<15>();
constexpr auto pes_args = TensorAccessorArgs<logits_args.next_compile_time_args_offset()>();
constexpr auto base_args = TensorAccessorArgs<pes_args.next_compile_time_args_offset()>();
constexpr auto w_args = TensorAccessorArgs<base_args.next_compile_time_args_offset()>();
constexpr auto sp_args = TensorAccessorArgs<w_args.next_compile_time_args_offset()>();
constexpr uint32_t PACKED = get_compile_time_arg_val(sp_args.next_compile_time_args_offset());
constexpr uint32_t TILE_BYTES = 2048;
// scratch layout (bytes): pes row | base page | send buffer | staging (E_loc rows of 64 B) | sparsity page
constexpr uint32_t OFF_PES = 0;
constexpr uint32_t OFF_BASE = 512;
constexpr uint32_t OFF_SEND = 640;
constexpr uint32_t OFF_STAGE = 1024;
constexpr uint32_t OFF_SP = 1024 + E_LOC * 64;

union FU32 {
    float f;
    uint32_t u;
};

FORCE_INLINE float bf16_to_f32(uint16_t b) {
    FU32 v;
    v.u = uint32_t(b) << 16;
    return v.f;
}

FORCE_INLINE uint16_t f32_to_bf16(float f) {  // round to nearest even
    FU32 v;
    v.f = f;
    const uint32_t r = v.u + 0x7FFFu + ((v.u >> 16) & 1u);
    return uint16_t(r >> 16);
}

// monotone unsigned key of a float (totally ordered, -0 < +0)
FORCE_INLINE uint32_t fkey(uint32_t bits) { return (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u); }

// exp(x) for x <= 0: 2^n * 2^r with r = x*log2e - n in [0, 1), 2^r by a degree-6 polynomial (rel. err < 2e-7)
FORCE_INLINE float exp_neg(float x) {
    if (x < -87.0f) {
        return 0.0f;
    }
    const float y = x * 1.44269504f;
    int n = int(y);
    if (float(n) > y) {
        --n;  // floor for negative y
    }
    const float r = (y - float(n)) * 0.69314718f;  // ln(2^(y-n)) in [0, ln2)
    float p = 1.0f / 720.0f;
    p = p * r + 1.0f / 120.0f;
    p = p * r + 1.0f / 24.0f;
    p = p * r + 1.0f / 6.0f;
    p = p * r + 0.5f;
    p = p * r + 1.0f;
    p = p * r + 1.0f;
    FU32 s;
    s.u = uint32_t(n + 127) << 23;
    return p * s.f;
}

void kernel_main() {
    const uint32_t logits_addr = get_arg_val<uint32_t>(0);
    const uint32_t pes_addr = get_arg_val<uint32_t>(1);
    const uint32_t base_addr = get_arg_val<uint32_t>(2);
    const uint32_t w_addr = get_arg_val<uint32_t>(3);
    const uint32_t sp_addr = get_arg_val<uint32_t>(4);
    const uint32_t me = get_arg_val<uint32_t>(5);
    const uint32_t B = get_arg_val<uint32_t>(6);

    const uint32_t scratch = get_write_ptr(CB_SCRATCH);
    const uint32_t gather = get_write_ptr(CB_GATHER);
    const uint32_t logits_l1 = get_write_ptr(CB_LOGITS);
    volatile tt_l1_ptr uint32_t* sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SEM_ID));
    const uint64_t leader_sem = get_noc_addr(LEADER_X, LEADER_Y, get_semaphore(SEM_ID));

    if (me < B) {
        const auto s_logits = TensorAccessor(logits_args, logits_addr, TILE_BYTES);
        const auto s_pes = TensorAccessor(pes_args, pes_addr, PES_PAGE);
        const auto s_base = TensorAccessor(base_args, base_addr, BASE_PAGE);
        for (uint32_t c = 0; c < COLS_T; ++c) {
            noc_async_read(s_logits.get_noc_addr(c), logits_l1 + c * TILE_BYTES, TILE_BYTES);
        }
        noc_async_read(s_pes.get_noc_addr(0), scratch + OFF_PES, PES_PAGE);
        noc_async_read(s_base.get_noc_addr(0), scratch + OFF_BASE, BASE_PAGE);
        noc_async_read_barrier();
        volatile tt_l1_ptr uint16_t* lg = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(logits_l1);
        volatile tt_l1_ptr uint16_t* pes = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(scratch + OFF_PES);
        const uint32_t base = *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch + OFF_BASE);
        // my row of the logits: element (me, j) of tile j/32 sits in face (me/16)*2 + (j%32)/16
        uint32_t key[E];
        for (uint32_t j = 0; j < E; ++j) {
            const uint32_t t = j >> 5, cj = j & 31;
            const uint32_t idx = t * 1024 + ((me >> 4) * 2 + (cj >> 4)) * 256 + (me & 15) * 16 + (cj & 15);
            key[j] = fkey(uint32_t(lg[idx]) << 16);
        }
        uint32_t sel[K];
        for (uint32_t s = 0; s < K; ++s) {
            uint32_t best = 0xFFFFFFFFu, bkey = 0;
            for (uint32_t j = 0; j < E; ++j) {
                bool taken = false;
                for (uint32_t q = 0; q < s; ++q) {
                    taken |= (sel[q] == j);
                }
                if (!taken && (best == 0xFFFFFFFFu || key[j] > bkey)) {
                    best = j;
                    bkey = key[j];
                }
            }
            sel[s] = best;
        }
        // softmax over the selected (the max is sel[0]), normalized by their sum, times the per-expert scale
        float p[K];
        float sum = 0.0f;
        const uint32_t t0 = sel[0] >> 5, c0 = sel[0] & 31;
        const float m = bf16_to_f32(lg[t0 * 1024 + ((me >> 4) * 2 + (c0 >> 4)) * 256 + (me & 15) * 16 + (c0 & 15)]);
        for (uint32_t s = 0; s < K; ++s) {
            const uint32_t t = sel[s] >> 5, cj = sel[s] & 31;
            const float l = bf16_to_f32(lg[t * 1024 + ((me >> 4) * 2 + (cj >> 4)) * 256 + (me & 15) * 16 + (cj & 15)]);
            p[s] = exp_neg(l - m);
            sum += p[s];
        }
        volatile tt_l1_ptr uint16_t* send = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(scratch + OFF_SEND);
        if constexpr (PACKED) {
            // my packed row: [w_0 .. w_{K-1} | id_0 .. id_{K-1}] bf16
            for (uint32_t s = 0; s < K; ++s) {
                const uint32_t g = sel[s];
                send[s] = f32_to_bf16(p[s] / sum * bf16_to_f32(pes[g]));
                send[K + s] = f32_to_bf16(float(g));
            }
            const auto s_w = TensorAccessor(w_args, w_addr, W_PAGE);
            noc_async_write(scratch + OFF_SEND, s_w.get_noc_addr(me), W_PAGE);
            noc_async_write_barrier();
            return;
        }
        for (uint32_t e = 0; e < SLOT / 2; ++e) {
            send[e] = 0;
        }
        for (uint32_t s = 0; s < K; ++s) {
            const uint32_t g = sel[s];
            if (g >= base && g < base + E_LOC) {
                send[g - base] = f32_to_bf16(p[s] / sum * bf16_to_f32(pes[g]));
            }
        }
        noc_async_write(scratch + OFF_SEND, get_noc_addr(LEADER_X, LEADER_Y, gather + me * SLOT), SLOT);
        noc_async_write_barrier();
        noc_semaphore_inc(leader_sem, 1);
        noc_async_atomic_barrier();
    } else if constexpr (PACKED) {
        // no user on this row: weight 0, distinct ids 0..K-1 (the frame-padding convention of the kernel path)
        volatile tt_l1_ptr uint16_t* send = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(scratch + OFF_SEND);
        for (uint32_t s = 0; s < K; ++s) {
            send[s] = 0;
            send[K + s] = f32_to_bf16(float(s));
        }
        const auto s_w = TensorAccessor(w_args, w_addr, W_PAGE);
        noc_async_write(scratch + OFF_SEND, s_w.get_noc_addr(me), W_PAGE);
        noc_async_write_barrier();
        return;
    }
    if constexpr (PACKED) {
        return;
    }
    if (me == 0) {
        noc_semaphore_wait_min(sem, B);
        noc_semaphore_set(sem, 0);
        volatile tt_l1_ptr uint16_t* g = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(gather);
        volatile tt_l1_ptr uint16_t* stage = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(scratch + OFF_STAGE);
        volatile tt_l1_ptr uint16_t* sp = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(scratch + OFF_SP);
        const uint32_t slot_elems = SLOT / 2;
        for (uint32_t e = 0; e < E_LOC; ++e) {
            bool any = false;
            for (uint32_t u = 0; u < 32; ++u) {
                const uint16_t v = (u < B) ? g[u * slot_elems + e] : uint16_t(0);
                stage[e * 32 + u] = v;
                any |= (v & 0x7FFFu) != 0;
            }
            sp[e] = any ? uint16_t(0x3F80) : uint16_t(0);
        }
        const auto s_w = TensorAccessor(w_args, w_addr, W_PAGE);
        const auto s_sp = TensorAccessor(sp_args, sp_addr, SP_PAGE);
        for (uint32_t e = 0; e < E_LOC; ++e) {
            noc_async_write(scratch + OFF_STAGE + e * 64, s_w.get_noc_addr(e), W_PAGE);
        }
        noc_async_write(scratch + OFF_SP, s_sp.get_noc_addr(0), SP_PAGE);
        noc_async_write_barrier();
    }
}
