// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Prefill mHC "pre" — sync kernel (NCRISC, NoC1): core-to-core partials, messages and the root's re-layouts
// (mhc_pre_pf_common.hpp). Per row block of the group:
// Round 1  member  CB_PART (M, S) -> root CB_GATHER slot (8 KB) + root SEM_G1
//          root    SEM_G1 == P -> push CB_GATHER (2P); CB_RED (z) -> CB_PLANE (plane layout); CB_COEF -> mailbox
//                  (pre, 4 x 32 fp32) to the P members + their SEM_M1; post / comb tiles of the row block out
//          member  SEM_M1 -> CB_PRE (pre_s in column 0 of tile s, rows = tokens)
// Round 2  member  CB_S2 -> root CB_GATHER page slot + SEM_G2; root SEM_G2 == P -> push CB_GATHER (2P: every round
//          consumes the whole CB, so the next round's pages start at its base again); CB_INVR -> mailbox (1/rms(h),
//          32 fp32) -> SEM_M2 -> member CB_INV.
// Every data write completes (write barrier) before the semaphore increment that publishes it; a core waits on its own
// semaphores only, resets them right after the wait (the next row block's increments can only follow this block's
// messages), and reads its mailbox after invalidating the L1 cache.
// Runtime (member): root (x, y), slot, first row block, row block stride, row blocks.
// Compile-time (root): P, then the TensorAccessorArgs of post, comb. Runtime (root): post, comb addresses, first row
// block, row block stride, row blocks, P x member (x, y).

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "mhc_pre_pf_common.hpp"

using namespace mhcpf;

namespace {

FORCE_INLINE volatile tt_l1_ptr uint32_t* words(uint32_t l1) {
    return reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l1);
}

FORCE_INLINE void zero_l1(uint32_t l1, uint32_t bytes) {
    volatile tt_l1_ptr uint32_t* p = words(l1);
    for (uint32_t i = 0; i < bytes / 4; ++i) {
        p[i] = 0;
    }
}

FORCE_INLINE volatile tt_l1_ptr uint32_t* sem_ptr(uint32_t id) {
    return reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(id));
}

// bytes at local l1 -> (x, y):dst_l1, completed, then +1 on (x, y)'s semaphore `sem`
FORCE_INLINE void send(uint32_t l1, uint32_t bytes, uint32_t x, uint32_t y, uint32_t dst_l1, uint32_t sem) {
    noc_async_write(l1, get_noc_addr(x, y, dst_l1), bytes);
    noc_async_write_barrier();
    noc_semaphore_inc(get_noc_addr(x, y, get_semaphore(sem)), 1);
    noc_async_atomic_barrier();
}

}  // namespace

void kernel_main() {
    const uint32_t gather_l1 = get_write_ptr(CB_GATHER);
    const uint32_t msg_l1 = get_write_ptr(CB_MSG);

#if defined(PF_MEMBER)
    const uint32_t root_x = get_arg_val<uint32_t>(0);
    const uint32_t root_y = get_arg_val<uint32_t>(1);
    const uint32_t slot = get_arg_val<uint32_t>(2);
    const uint32_t rb0 = get_arg_val<uint32_t>(3);
    const uint32_t step = get_arg_val<uint32_t>(4);
    const uint32_t n_rb = get_arg_val<uint32_t>(5);
    bool first = true;
    for (uint32_t rb = rb0; rb < n_rb; rb += step) {
        // the expanded broadcast operands: CB_PRE (4 pages) / CB_INV (1 page) are consumed whole each row block, so
        // every reserve returns the same pages; zeroed once (only column 0 is written afterwards)
        cb_reserve_back(CB_PRE, N_STREAMS);
        cb_reserve_back(CB_INV, 1);
        const uint32_t pre_l1 = get_write_ptr(CB_PRE);
        const uint32_t inv_l1 = get_write_ptr(CB_INV);
        if (first) {
            zero_l1(pre_l1, N_STREAMS * FP32_TILE_BYTES);
            zero_l1(inv_l1, FP32_TILE_BYTES);
            first = false;
        }

        // ---- round 1: (M, S) partial -> root slot
        cb_wait_front(CB_PART, 2);
        send(
            get_read_ptr(CB_PART),
            2 * FP32_TILE_BYTES,
            root_x,
            root_y,
            gather_l1 + slot * 2 * FP32_TILE_BYTES,
            SEM_G1);  // fmt: skip
        cb_pop_front(CB_PART, 2);

        // ---- round-1 message: pre_s(t) at msg[32 s + t]
        noc_semaphore_wait(sem_ptr(SEM_M1), 1);
        noc_semaphore_set(sem_ptr(SEM_M1), 0);
        invalidate_l1_cache();
        {
            volatile tt_l1_ptr uint32_t* msg = words(msg_l1);
            for (uint32_t s = 0; s < N_STREAMS; ++s) {
                volatile tt_l1_ptr uint32_t* tile = words(pre_l1 + s * FP32_TILE_BYTES);
                for (uint32_t t = 0; t < TILE; ++t) {
#if defined(PF_FPU)
                    tile[tile_off(t, t)] = msg[s * TILE + t];  // diag(pre_s): the collapse is D_s @ x_s on the FPU
#else
                    tile[tile_off(t, 0)] = msg[s * TILE + t];
#endif
                }
            }
        }
        cb_push_back(CB_PRE, N_STREAMS);

        // ---- round 2: S2 partial -> root slot (one tile per slot)
        cb_wait_front(CB_S2, 1);
        send(get_read_ptr(CB_S2), FP32_TILE_BYTES, root_x, root_y, gather_l1 + slot * FP32_TILE_BYTES, SEM_G2);
        cb_pop_front(CB_S2, 1);

        // ---- round-2 message: 1/rms(h)(t) at msg[t]
        noc_semaphore_wait(sem_ptr(SEM_M2), 1);
        noc_semaphore_set(sem_ptr(SEM_M2), 0);
        invalidate_l1_cache();
        {
            volatile tt_l1_ptr uint32_t* msg = words(msg_l1);
            volatile tt_l1_ptr uint32_t* tile = words(inv_l1);
            for (uint32_t t = 0; t < TILE; ++t) {
                tile[tile_off(t, 0)] = msg[t];
            }
        }
        cb_push_back(CB_INV, 1);
    }

#elif defined(PF_ROOT)
    constexpr uint32_t P = get_compile_time_arg_val(0);
    constexpr auto post_args = TensorAccessorArgs<1>();
    constexpr auto comb_args = TensorAccessorArgs<post_args.next_compile_time_args_offset()>();
    const auto post = TensorAccessor(post_args, get_arg_val<uint32_t>(0));
    const auto comb = TensorAccessor(comb_args, get_arg_val<uint32_t>(1));
    const uint32_t rb0 = get_arg_val<uint32_t>(2);
    const uint32_t step = get_arg_val<uint32_t>(3);
    const uint32_t n_rb = get_arg_val<uint32_t>(4);
    constexpr uint32_t MEMBER_ARGS = 5;  // P x (x, y)

    // message -> every member's mailbox, completed, then +1 on each member's `sem`
    auto broadcast = [&](uint32_t bytes, uint32_t sem) {
        for (uint32_t m = 0; m < P; ++m) {
            noc_async_write(
                msg_l1,
                get_noc_addr(
                    get_arg_val<uint32_t>(MEMBER_ARGS + 2 * m), get_arg_val<uint32_t>(MEMBER_ARGS + 2 * m + 1), msg_l1),
                bytes);
        }
        noc_async_write_barrier();
        for (uint32_t m = 0; m < P; ++m) {
            noc_semaphore_inc(
                get_noc_addr(
                    get_arg_val<uint32_t>(MEMBER_ARGS + 2 * m),
                    get_arg_val<uint32_t>(MEMBER_ARGS + 2 * m + 1),
                    get_semaphore(sem)),
                1);
        }
        noc_async_atomic_barrier();
    };

    // the plane page (planes >= 24 never written) and the two output assembly tiles (post: columns 0..3, comb: 0..15
    // of every row; the rest stays zero) are 1- / 2-page CBs consumed whole: zeroed once
    cb_reserve_back(CB_SCR, 2);  // root-private scratch: reserved once, never pushed
    const uint32_t scr = get_write_ptr(CB_SCR);
    zero_l1(scr, 2 * FP32_TILE_BYTES);
    bool first = true;
    for (uint32_t rb = rb0; rb < n_rb; rb += step) {
        // ---- round 1: the P partials
        cb_reserve_back(CB_GATHER, 2 * P);
        noc_semaphore_wait(sem_ptr(SEM_G1), P);
        noc_semaphore_set(sem_ptr(SEM_G1), 0);
        cb_push_back(CB_GATHER, 2 * P);

        // z (rows = tokens, cols = the 24 mixes) -> plane layout (plane v, lane t)
        cb_reserve_back(CB_PLANE, 1);
        const uint32_t plane_l1 = get_write_ptr(CB_PLANE);
        if (first) {
            zero_l1(plane_l1, FP32_TILE_BYTES);
            first = false;
        }
        cb_wait_front(CB_RED, 1);
        invalidate_l1_cache();  // packer-written data read by this RISC
        {
            const uint32_t* z = reinterpret_cast<const uint32_t*>(get_read_ptr(CB_RED));
            uint32_t* plane = reinterpret_cast<uint32_t*>(plane_l1);
            for (uint32_t t = 0; t < TILE; ++t) {
#pragma GCC unroll 8
                for (uint32_t v = 0; v < N_MIX; ++v) {
                    plane[plane_off(v, t)] = z[tile_off(t, v)];
                }
            }
            invalidate_l1_cache();  // (Blackhole: a fence) every store lands before the push
        }
        cb_push_back(CB_PLANE, 1);
        cb_pop_front(CB_RED, 1);

        // coefficients -> mailbox (pre) -> members; post / comb tiles of the row block
        cb_wait_front(CB_COEF, 1);
        invalidate_l1_cache();
        {
            volatile tt_l1_ptr uint32_t* coef = words(get_read_ptr(CB_COEF));
            volatile tt_l1_ptr uint32_t* msg = words(msg_l1);
            for (uint32_t s = 0; s < N_STREAMS; ++s) {
                for (uint32_t t = 0; t < TILE; ++t) {
                    msg[s * TILE + t] = coef[plane_off(s, t)];
                }
            }
            broadcast(MSG_FLOATS * 4, SEM_M1);
            // post [T, 4]: row t, column i; comb [T, 16]: row t, column 4j + k (C[j][k])
            volatile tt_l1_ptr uint32_t* po = words(scr);
            volatile tt_l1_ptr uint32_t* co = words(scr + FP32_TILE_BYTES);
            for (uint32_t t = 0; t < TILE; ++t) {
                for (uint32_t i = 0; i < N_STREAMS; ++i) {
                    po[tile_off(t, i)] = coef[plane_off(N_STREAMS + i, t)];
                }
                for (uint32_t e = 0; e < N_STREAMS * N_STREAMS; ++e) {
                    co[tile_off(t, e)] = coef[plane_off(2 * N_STREAMS + e, t)];
                }
            }
            noc_async_write_page(rb, post, scr);
            noc_async_write_page(rb, comb, scr + FP32_TILE_BYTES);
            noc_async_write_barrier();
        }
        cb_pop_front(CB_COEF, 1);

        // ---- round 2: the P S2 partials -> 1/rms(h) -> mailbox -> members
        cb_reserve_back(CB_GATHER, 2 * P);
        noc_semaphore_wait(sem_ptr(SEM_G2), P);
        noc_semaphore_set(sem_ptr(SEM_G2), 0);
        cb_push_back(CB_GATHER, 2 * P);
        cb_wait_front(CB_INVR, 1);
        invalidate_l1_cache();
        {
            volatile tt_l1_ptr uint32_t* inv = words(get_read_ptr(CB_INVR));
            volatile tt_l1_ptr uint32_t* msg = words(msg_l1);
            for (uint32_t t = 0; t < TILE; ++t) {
                msg[t] = inv[tile_off(t, 0)];
            }
        }
        cb_pop_front(CB_INVR, 1);
        broadcast(TILE * 4, SEM_M2);
    }
#endif
}
