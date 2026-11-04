// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// PMK mHC phases — sync kernel (NCRISC, NoC1; mhc_common.hpp). Derived from Stage 7a's kernels/mhc_pre_pf/sync.cpp:
// the same two rounds per (sublayer, row block) (member (M, S) partial -> root -> pre message -> member; member S2
// -> root -> 1/rms(h) -> member), with
//   * PROGRAM-WIDE cumulative counters: the round index of (sublayer, block k) is idx = round0 + sub * NRB + k + 1,
//     round0 = layer index x NSUB x NRB (runtime arg); members wait >= idx, the root >= idx * P; nothing is reset
//     (the launch's initial 0 is the single reset);
//   * the post's coefficients: the root packs K_s | L_s tiles (7a's mhc_post_pf7a reader layout: post_s | C[0][s] |
//     C[1][s] | C[2][s] in columns 0 / 1 / 16 / 17, C[3][s] in column 0 of L_s) in its own CB_PCOEF pages and unicasts
//     the 8 tiles of the block to every member's CB_PCOEF (uniform address) with a write barrier BEFORE its round-2
//     message, so a member's M2 >= idx implies the tiles landed (no coefficient semaphore: 7a review, id 12 is the
//     family's XS_FREE); the member pushes the block's pages right after its M2 wait (live until the post pops them);
//   * H1 / H7 (x-hat): PMK_XH push mode writes the member's 16 hn tiles to the AG-store slice and raises FS_ARRIVE
//     (15) on the holder (F_XH_DRAM: the stream RISC writes them to DRAM instead);
//   * H6 / H10 (y): lane-landed y pages are published to the compute after FS_REDUCED (14) >= idx.
// Parts (PMK_MHC_PART): P1 = [F_INIT zeroing] pre rounds of sublayer 0 (+ H1 push); P10 = y pushes of sublayer 0,
// pre rounds of sublayer 1; P16 = y pushes of sublayer 1; 0 = all three.
// Member runtime (ph_rt0 +): root x, root y, slot j, flags, round0, then per (sublayer, block): AG-store core x, y,
// L1 address of this member's 32 x 512 bf16 slice (3 each).
// Root compile-time (PH_CT0 +): TensorAccessorArgs of post, comb, taps. Root runtime: flags, first row block (= g),
// row block stride, round0, taps address, then per sublayer: post address, comb address (2), then P x member (x, y).
// Defines: PMK_NSUB, PMK_NRB, PMK_SEM_Y, PMK_SEM_XH, PMK_MHC_PART, PMK_PROBE (root: the compute's N_TAPS tiles per
// round -> taps page (rb * NSUB + sub) * N_TAPS + i, when F_PROBE).

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "mhc_common.hpp"

using namespace pmk_mhc;

namespace {

constexpr uint32_t NSUB = PMK_NSUB;
constexpr uint32_t NRB = PMK_NRB;

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

#if defined(PMK_MEMBER)
struct MemberSync {
    uint32_t root_x, root_y, slot, flags, round0;
    static constexpr uint32_t SUB_ARGS = 5;

    void init() const {  // once per program (F_INIT): the expanded broadcast operands are consumed whole per block,
        cb_reserve_back(CB_PRE, N_STREAMS);  // so every reserve returns the same pages: zeroed once (only the
        cb_reserve_back(CB_INV, 1);          // diagonal / column 0 are written afterwards)
        zero_l1(get_write_ptr(CB_PRE), N_STREAMS * FP32_TILE_BYTES);
        zero_l1(get_write_ptr(CB_INV), FP32_TILE_BYTES);
    }

    void pre(uint32_t sub) const {
        const uint32_t gather_l1 = get_write_ptr(CB_GATHER);
        const uint32_t msg_l1 = get_write_ptr(CB_MSG);
        // the root writes this sublayer's NRB x 8 coefficient tiles at the CB base: reserved whole here, pushed per
        // block after M2 (the compute pops per block in the post; the ring is exactly NRB blocks, so the next reserve
        // returns the base again, and the root's next-sublayer writes follow this sublayer's pops causally)
        cb_reserve_back(CB_PCOEF, NRB * 2 * N_STREAMS);
        for (uint32_t k = 0; k < NRB; ++k) {
            const uint32_t idx = round0 + sub * NRB + k + 1;
            cb_reserve_back(CB_PRE, N_STREAMS);
            cb_reserve_back(CB_INV, 1);
            const uint32_t pre_l1 = get_write_ptr(CB_PRE);
            const uint32_t inv_l1 = get_write_ptr(CB_INV);
            // round 1: (M, S) partial -> root slot
            cb_wait_front(CB_PART, 2);
            send(
                get_read_ptr(CB_PART),
                2 * FP32_TILE_BYTES,
                root_x,
                root_y,
                gather_l1 + slot * 2 * FP32_TILE_BYTES,
                SEM_G1);
            cb_pop_front(CB_PART, 2);
            // round-1 message: pre_s(t) at msg[32 s + t] -> diag(pre_s) tiles (the PF_FPU collapse operand)
            noc_semaphore_wait_min(sem_ptr(SEM_M1), idx);
            invalidate_l1_cache();
            {
                volatile tt_l1_ptr uint32_t* msg = words(msg_l1);
                for (uint32_t s = 0; s < N_STREAMS; ++s) {
                    volatile tt_l1_ptr uint32_t* tile = words(pre_l1 + s * FP32_TILE_BYTES);
                    for (uint32_t t = 0; t < TILE; ++t) {
                        tile[tile_off(t, t)] = msg[s * TILE + t];
                    }
                }
            }
            cb_push_back(CB_PRE, N_STREAMS);
            // round 2: S2 partial -> root slot
            cb_wait_front(CB_S2, 1);
            send(get_read_ptr(CB_S2), FP32_TILE_BYTES, root_x, root_y, gather_l1 + slot * FP32_TILE_BYTES, SEM_G2);
            cb_pop_front(CB_S2, 1);
            // round-2 message: 1/rms(h)(t) at msg[t]; the block's coefficient tiles landed before it
            noc_semaphore_wait_min(sem_ptr(SEM_M2), idx);
            invalidate_l1_cache();
            {
                volatile tt_l1_ptr uint32_t* msg = words(msg_l1);
                volatile tt_l1_ptr uint32_t* tile = words(inv_l1);
                for (uint32_t t = 0; t < TILE; ++t) {
                    tile[tile_off(t, 0)] = msg[t];
                }
            }
            cb_push_back(CB_INV, 1);
            cb_push_back(CB_PCOEF, 2 * N_STREAMS);
            // H1 / H7: the hn tiles -> the AG store slice (+ FS_ARRIVE there); DRAM mode: the stream RISC's job
            if (!(flags & F_XH_DRAM)) {
                const uint32_t a = ph_rt0 + SUB_ARGS + 3 * (sub * NRB + k);
                cb_wait_front(CB_OUT, CPC);
                send(
                    get_read_ptr(CB_OUT),
                    CPC * get_tile_size(CB_OUT),
                    get_arg_val<uint32_t>(a),
                    get_arg_val<uint32_t>(a + 1),
                    get_arg_val<uint32_t>(a + 2),
                    PMK_SEM_XH);
                cb_pop_front(CB_OUT, CPC);
            }
        }
    }
    void post(uint32_t sub) const {  // lane-landed y: publish per block after FS_REDUCED >= idx
        if (flags & F_Y_DRAM) {
            return;
        }
        cb_reserve_back(CB_Y, NRB * CPC);
        for (uint32_t k = 0; k < NRB; ++k) {
            noc_semaphore_wait_min(sem_ptr(PMK_SEM_Y), round0 + sub * NRB + k + 1);
            cb_push_back(CB_Y, CPC);
        }
    }
    void part_p1() const {
        if (flags & F_INIT) {
            init();
        }
        pre(0);
    }
    void part_p10() const {
        post(0);
        pre(1);
    }
    void part_p16() const { post(1); }
};
#endif  // PMK_MEMBER

#if defined(PMK_ROOT)
constexpr auto post_args = TensorAccessorArgs<PH_CT0 + 0>();
constexpr auto comb_args = TensorAccessorArgs<post_args.next_compile_time_args_offset()>();
constexpr auto taps_args = TensorAccessorArgs<comb_args.next_compile_time_args_offset()>();

struct RootSync {
    uint32_t flags, rb0, step, round0, taps_addr;
    static constexpr uint32_t SUB_ARGS = 5;
    static constexpr uint32_t MEMBER_ARGS = SUB_ARGS + 2 * NSUB;  // P x (x, y)
    uint32_t mx(uint32_t m) const { return get_arg_val<uint32_t>(ph_rt0 + MEMBER_ARGS + 2 * m); }
    uint32_t my(uint32_t m) const { return get_arg_val<uint32_t>(ph_rt0 + MEMBER_ARGS + 2 * m + 1); }

    // bytes at local l1 -> the same address on every member, completed [then +1 on each member's `sem`]
    void broadcast(uint32_t l1, uint32_t bytes, uint32_t sem, bool signal) const {
        for (uint32_t m = 0; m < P; ++m) {
            noc_async_write(l1, get_noc_addr(mx(m), my(m), l1), bytes);
        }
        noc_async_write_barrier();
        if (signal) {
            for (uint32_t m = 0; m < P; ++m) {
                noc_semaphore_inc(get_noc_addr(mx(m), my(m), get_semaphore(sem)), 1);
            }
            noc_async_atomic_barrier();
        }
    }

    void init() const {  // once per program (F_INIT): root-private assembly pages, zeroed: the plane page (planes >=
        cb_reserve_back(CB_PLANE, 1);  // 24 never written), the post / comb output tiles (DRAM mode), the NRB x 8
        zero_l1(get_write_ptr(CB_PLANE), FP32_TILE_BYTES);  // packed coefficient tiles (every other element must be
        cb_reserve_back(CB_SCR, 2);  // 0: the broadcast masks multiply them by 0 and 0 * NaN would poison the value).
        zero_l1(get_write_ptr(CB_SCR), 2 * FP32_TILE_BYTES);  // SCR / PCOEF are reserved once and never pushed.
        cb_reserve_back(CB_PCOEF, NRB * 2 * N_STREAMS);
        zero_l1(get_write_ptr(CB_PCOEF), NRB * 2 * N_STREAMS * FP32_TILE_BYTES);
    }

    void rounds(uint32_t sub) const {
        const uint32_t msg_l1 = get_write_ptr(CB_MSG);
        const uint32_t scr = get_write_ptr(CB_SCR);
        const uint32_t pc0 = get_write_ptr(CB_PCOEF);
        for (uint32_t k = 0; k < NRB; ++k) {
            const uint32_t idx = round0 + sub * NRB + k + 1;
            const uint32_t rb = rb0 + k * step;
            const uint32_t pc = pc0 + k * 2 * N_STREAMS * FP32_TILE_BYTES;
            // ---- round 1: the P partials
            cb_reserve_back(CB_GATHER, 2 * P);
            noc_semaphore_wait_min(sem_ptr(SEM_G1), idx * P);
            cb_push_back(CB_GATHER, 2 * P);
            // z (rows = tokens, cols = the 24 mixes) -> plane layout (plane v, lane t)
            cb_reserve_back(CB_PLANE, 1);
            const uint32_t plane_l1 = get_write_ptr(CB_PLANE);
            cb_wait_front(CB_RED, 1);
            invalidate_l1_cache();
            {
                const uint32_t* z = reinterpret_cast<const uint32_t*>(get_read_ptr(CB_RED));
                uint32_t* plane = reinterpret_cast<uint32_t*>(plane_l1);
                for (uint32_t t = 0; t < TILE; ++t) {
#pragma GCC unroll 8
                    for (uint32_t v = 0; v < N_MIX; ++v) {
                        plane[plane_off(v, t)] = z[tile_off(t, v)];
                    }
                }
                invalidate_l1_cache();
            }
            cb_push_back(CB_PLANE, 1);
            cb_pop_front(CB_RED, 1);
            // coefficients -> mailbox (pre) -> members; the packed K / L tiles of the block -> members (barriered)
            cb_wait_front(CB_COEF, 1);
            invalidate_l1_cache();
            // the verified pack's end marker (mhc_common.hpp: the compute's round r = this part's round index) and
            // its attempt counts (the consumed CB_RED page): diagnostics only (-> tap 0 with the probe)
            const uint32_t r = (PMK_MHC_PART == 0 ? sub * NRB : 0u) + k;
            const uint32_t marker = marker_final(r);  // the pushed page carries cycle 3's marker
            uint32_t marker_ok = 1, marker_seen = 0, attempts1 = 0, attempts2 = 0;
            {
                volatile tt_l1_ptr uint32_t* coef = words(get_read_ptr(CB_COEF));
                for (uint32_t t = 0; t < TILE; ++t) {
                    const uint32_t v = coef[plane_off(PLANE_MARKER, t)];
                    marker_ok &= (v == marker) ? 1u : 0u;
                    marker_seen = v;
                }
                volatile tt_l1_ptr uint32_t* red = words(get_read_ptr(CB_RED));
                attempts1 = red[plane_off(PLANE_DIAG, 0)];
                attempts2 = red[plane_off(PLANE_DIAG, 1)];
            }
            {
                volatile tt_l1_ptr uint32_t* coef = words(get_read_ptr(CB_COEF));
                volatile tt_l1_ptr uint32_t* msg = words(msg_l1);
                for (uint32_t s = 0; s < N_STREAMS; ++s) {
                    for (uint32_t t = 0; t < TILE; ++t) {
                        msg[s * TILE + t] = coef[plane_off(s, t)];
                    }
                }
                broadcast(msg_l1, MSG_FLOATS * 4, SEM_M1, true);
                {  // rev 6 diagnostics record (the mailbox is free until the round-2 message): -> tap TAP_DIAG
                    msg[0] = DIAG_MAGIC;
                    msg[1] = r;
                    msg[2] = sub;
                    msg[3] = k;
                    msg[4] = marker;
                    msg[5] = coef[plane_off(PLANE_MARKER, 0)];
                    msg[6] = marker_seen;
                    msg[7] = marker_ok;
                    msg[8] = attempts1;
                    msg[9] = attempts2;
                    msg[10] = 6;  // PMK_REV
                    msg[11] = idx;
                }
                for (uint32_t s = 0; s < N_STREAMS; ++s) {
                    volatile tt_l1_ptr uint32_t* k_t = words(pc + 2 * s * FP32_TILE_BYTES);
                    volatile tt_l1_ptr uint32_t* l_t = words(pc + (2 * s + 1) * FP32_TILE_BYTES);
                    for (uint32_t t = 0; t < TILE; ++t) {
                        k_t[tile_off(t, 0)] = coef[plane_off(N_STREAMS + s, t)];
                        k_t[tile_off(t, 1)] = coef[plane_off(2 * N_STREAMS + 0 * N_STREAMS + s, t)];
                        k_t[tile_off(t, 16)] = coef[plane_off(2 * N_STREAMS + 1 * N_STREAMS + s, t)];
                        k_t[tile_off(t, 17)] = coef[plane_off(2 * N_STREAMS + 2 * N_STREAMS + s, t)];
                        l_t[tile_off(t, 0)] = coef[plane_off(2 * N_STREAMS + 3 * N_STREAMS + s, t)];
                    }
                }
                broadcast(pc, 2 * N_STREAMS * FP32_TILE_BYTES, 0, false);  // completed before the M2 message below
                if (flags & F_PC_DRAM) {  // post [T, 4]: row t, column i; comb [T, 16]: row t, column 4 j + k
                    const auto post = TensorAccessor(post_args, get_arg_val<uint32_t>(ph_rt0 + SUB_ARGS + 2 * sub));
                    const auto comb = TensorAccessor(comb_args, get_arg_val<uint32_t>(ph_rt0 + SUB_ARGS + 2 * sub + 1));
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
            }
            cb_pop_front(CB_COEF, 1);
#if defined(PMK_PROBE)
            if (flags & F_PROBE) {  // the compute's taps of this round -> taps pages (rb * NSUB + sub) * N_TAPS + i
                const auto taps = TensorAccessor(taps_args, taps_addr);
                cb_wait_front(CB_DBG, N_TAPS);
                const uint32_t d = get_read_ptr(CB_DBG);
                for (uint32_t i = 0; i < TAP_DIAG; ++i) {
                    noc_async_write_page((rb * NSUB + sub) * N_TAPS + i, taps, d + i * FP32_TILE_BYTES);
                }
                // tap TAP_DIAG: the sync's own record (built in the mailbox after the round-1 message, before the
                // round-2 message overwrites it): the first DIAG_WORDS words of the page
                noc_async_write(msg_l1, taps.get_noc_addr((rb * NSUB + sub) * N_TAPS + TAP_DIAG), DIAG_WORDS * 4);
                noc_async_write_barrier();
                cb_pop_front(CB_DBG, N_TAPS);
            }
#endif
            // ---- round 2: the P S2 partials -> 1/rms(h) -> mailbox -> members
            cb_reserve_back(CB_GATHER, 2 * P);
            noc_semaphore_wait_min(sem_ptr(SEM_G2), idx * P);
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
            broadcast(msg_l1, TILE * 4, SEM_M2, true);
        }
    }
    void part_p1() const {
        if (flags & F_INIT) {
            init();
        }
        rounds(0);
    }
    void part_p10() const { rounds(1); }
    void part_p16() const {}
};
#endif  // PMK_ROOT

}  // namespace

void kernel_main() {
#if defined(PMK_MEMBER)
    const MemberSync m{
        get_arg_val<uint32_t>(ph_rt0 + 0),
        get_arg_val<uint32_t>(ph_rt0 + 1),
        get_arg_val<uint32_t>(ph_rt0 + 2),
        get_arg_val<uint32_t>(ph_rt0 + 3),
        get_arg_val<uint32_t>(ph_rt0 + 4)};
#elif defined(PMK_ROOT)
    const RootSync m{
        get_arg_val<uint32_t>(ph_rt0 + 0),
        get_arg_val<uint32_t>(ph_rt0 + 1),
        get_arg_val<uint32_t>(ph_rt0 + 2),
        get_arg_val<uint32_t>(ph_rt0 + 3),
        get_arg_val<uint32_t>(ph_rt0 + 4)};
#endif
#if PMK_MHC_PART == 1
    m.part_p1();
#elif PMK_MHC_PART == 10
    m.part_p10();
#elif PMK_MHC_PART == 16
    m.part_p16();
#else
    m.part_p1();
    m.part_p10();
    m.part_p16();
#endif
}
