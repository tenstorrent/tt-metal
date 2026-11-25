// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Attention worker compute (see attn_common.hpp). Per attention layer and user:
//   chunk worker: the partial attention of q over this core's KV chunk, block by block with an online
//     softmax (a zero partial without one);
//   tail core: k = rope(rmsnorm_rows(k) * w_k), the new k / v row into the tail tiles, the partial over
//     the tail tile (masked), then the tail tiles as bf8 for the DRAM write-back;
// a group head adds its children's partials to its own.
//
// Runtime args: 0 role (1 chunk worker, 2 tail core), 1 children. The chunk's KV tiles come in cb_ntiles, and
// (rows mode, active_rows.hpp) the active mask in its word kMaskWord: only the active users run.

#include <cstdint>
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_unary/fill.h"
#include "api/compute/binary_max_min.h"
#include "attn_compute_common.hpp"
#include "active_rows.hpp"

using namespace resident_attn;

namespace {

FORCE_INLINE void zero_partial() {
    cb_reserve_back(cb_o, kPart);
    pack_reconfig_data_format(cb_o);
    fill_tile_init();
    for (uint32_t d = 0; d < kPart; d++) {
        tile_regs_acquire();
        fill_tile(0, 0.0f);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_o);
        tile_regs_release();
    }
    cb_push_back(cb_o, kPart);
}

// DEST tile 0 = row max of the block's nb score tiles
FORCE_INLINE void block_max(uint32_t nb) {
    reconfig_data_format(cb_s, cb_one);
    reduce_init<PoolType::MAX, ReduceDim::REDUCE_ROW>(cb_s, cb_one, cb_mrun);
    for (uint32_t j = 0; j < nb; j++) {
        reduce_tile<PoolType::MAX, ReduceDim::REDUCE_ROW>(cb_s, cb_one, j, 0, 0);
    }
    reduce_uninit();
}

// cb_p = exp(s - m) over the block (m: column 0 of m_cb's front tile); consumes cb_s
FORCE_INLINE void block_probs(uint32_t m_cb, uint32_t nb) {
    cb_reserve_back(cb_p, chunk_max);
    reconfig_data_format(cb_s, m_cb);
    pack_reconfig_data_format(cb_p);
    sub_bcast_cols_init(cb_s, m_cb);
    exp_tile_init();
    for (uint32_t j = 0; j < nb; j++) {
        tile_regs_acquire();
        sub_tiles_bcast_cols(cb_s, m_cb, j, 0, 0);
        exp_tile(0, kExpMode);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile<true>(0, cb_p, j);
        tile_regs_release();
    }
    cb_push_back(cb_p, chunk_max);
    cb_pop_front(cb_s, chunk_max);
}

// DEST tile 0 = old * alpha (column broadcast); old is tile i of old_cb
FORCE_INLINE void rescaled(uint32_t old_cb, uint32_t i, uint32_t alpha_cb) {
    reconfig_data_format(old_cb, alpha_cb);
    mul_bcast_cols_init(old_cb, alpha_cb);
    mul_tiles_bcast_cols(old_cb, alpha_cb, i, 0, 0);
}

// O-2: an item queued behind the front one of a two-item FIFO — wrapped past the CB's end when the front one sits in
// its second half; tile indices do not wrap, so the unpacker's read pointer is moved there (on) and back (off)
FORCE_INLINE void at_second(uint32_t cb, uint32_t first_pages, bool on) {
    UNPACK(({
        auto& f = get_local_cb_interface(cb);
        if (on) {
            uint32_t p = f.fifo_rd_ptr + first_pages * f.fifo_page_size;
            f.fifo_rd_ptr = p >= f.fifo_limit ? p - f.fifo_size : p;
        } else {
            uint32_t p = f.fifo_rd_ptr - first_pages * f.fifo_page_size;
            f.fifo_rd_ptr = p < f.fifo_limit - f.fifo_size ? p + f.fifo_size : p;
        }
    }));
}

// a user's running state; O-2 overlaps only users of one block (online_block with first = true reads no old state), so
// the overlapped users' states sit one behind the other in the same FIFOs (cb_orun holds 2 Dt)
struct St {
    uint32_t orun, lrun, mrun, alpha;
};
constexpr St kSt[1] = {{cb_orun, cb_lrun, cb_mrun, cb_alpha}};

// one block of the online softmax: m = max(m, rowmax s), alpha = exp(m_old - m), p = exp(s - m),
// o = o alpha + p v, l = l alpha + rowsum p (the first block starts from o = l = 0, alpha unused)
// (pending, O-2: another user's state still queued ahead in the same CBs — the first block reads its new running max
// as the second item)
FORCE_INLINE void online_block(uint32_t nb, bool first, const St& st, bool pending = false) {
    cb_wait_front(cb_s, chunk_max);
    cb_reserve_back(st.mrun, 1);
    if (!first) {
        cb_reserve_back(st.alpha, 1);
    }
    tile_regs_acquire();
    block_max(nb);
    if (!first) {
        reconfig_data_format_srca(st.mrun);
        copy_init(st.mrun);
        copy_tile(st.mrun, 0, 1);
        binary_max_tile_init();
        binary_max_tile(0, 1, 2);
        sub_binary_tile_init();
        sub_binary_tile(1, 2, 3);
        exp_tile_init();
        exp_tile(3, kExpMode);
    }
    tile_regs_commit();
    tile_regs_wait();
    pack_reconfig_data_format(st.mrun);
    pack_tile(first ? 0 : 2, st.mrun);
    if (!first) {
        pack_reconfig_data_format(st.alpha);
        pack_tile(3, st.alpha);
    }
    tile_regs_release();
    cb_push_back(st.mrun, 1);
    if (!first) {
        cb_push_back(st.alpha, 1);
        cb_pop_front(st.mrun, 1);
        cb_wait_front(st.alpha, 1);
    }
    cb_wait_front(st.mrun, pending ? 2 : 1);
    if (pending) {
        at_second(st.mrun, 1, true);
    }
    block_probs(st.mrun, nb);
    if (pending) {
        at_second(st.mrun, 1, false);
    }

    cb_wait_front(cb_p, chunk_max);
    cb_reserve_back(st.orun, Dt);
    pack_reconfig_data_format(st.orun);
    for (uint32_t d = 0; d < Dt; d++) {
        tile_regs_acquire();
        if (!first) {
            rescaled(st.orun, d, st.alpha);
        }
        reconfig_data_format(cb_v, cb_p);
        matmul_init(cb_p, cb_v, 0);
        for (uint32_t j = 0; j < nb; j++) {
            matmul_tiles(cb_p, cb_v, j, j * Dt + d, 0);
        }
        tile_regs_commit();
        tile_regs_wait();
        pack_tile<true>(0, st.orun, d);
        tile_regs_release();
    }
    cb_push_back(st.orun, Dt);
    cb_reserve_back(st.lrun, 1);
    pack_reconfig_data_format(st.lrun);
    tile_regs_acquire();
    if (!first) {
        rescaled(st.lrun, 0, st.alpha);
    }
    reconfig_data_format(cb_one, cb_p);
    reduce_init<PoolType::SUM, ReduceDim::REDUCE_ROW>(cb_p, cb_one, st.lrun);
    for (uint32_t j = 0; j < nb; j++) {
        reduce_tile<PoolType::SUM, ReduceDim::REDUCE_ROW>(cb_p, cb_one, j, 0, 0);
    }
    reduce_uninit();
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, st.lrun);
    tile_regs_release();
    cb_push_back(st.lrun, 1);
    if (!first) {
        cb_pop_front(st.orun, Dt);
        cb_pop_front(st.lrun, 1);
        cb_pop_front(st.alpha, 1);
    }
    cb_pop_front(cb_p, chunk_max);
}

// sends m (cb_m) — the running row max of the user's state
FORCE_INLINE void send_m(const St& st, bool pending = false) {
    cb_reserve_back(cb_m, 1);
    if (pending) {
        at_second(st.mrun, 1, true);
    }
    tile_regs_acquire();
    reconfig_data_format_srca(st.mrun);
    copy_init(st.mrun);
    copy_tile(st.mrun, 0, 0);
    tile_regs_commit();
    tile_regs_wait();
    pack_reconfig_data_format(cb_m);
    pack_tile(0, cb_m);
    tile_regs_release();
    if (pending) {
        at_second(st.mrun, 1, false);
    }
    cb_push_back(cb_m, 1);
}

// with the global row max M: partial (o, l) * exp(m - M) into cb_o; releases the state
FORCE_INLINE void finish_partial(const St& st) {
    cb_wait_front(cb_M, 1);
    cb_reserve_back(st.alpha, 1);
    tile_regs_acquire();
    // (its own unpacker setup: O-2 can run it right after another user's finish, whose last op left another source
    // configured — the unpacked running max then lost precision; sequentially send_m set it up)
    reconfig_data_format_srca(st.mrun);
    copy_init(st.mrun);
    copy_tile(st.mrun, 0, 0);
    reconfig_data_format_srca(cb_M);
    copy_init(cb_M);
    copy_tile(cb_M, 0, 1);
    sub_binary_tile_init();
    sub_binary_tile(0, 1, 0);
    exp_tile_init();
    exp_tile(0, kExpMode);
    tile_regs_commit();
    tile_regs_wait();
    pack_reconfig_data_format(st.alpha);
    pack_tile(0, st.alpha);
    tile_regs_release();
    cb_push_back(st.alpha, 1);
    cb_wait_front(st.alpha, 1);

    cb_reserve_back(cb_o, kPart);
    pack_reconfig_data_format(cb_o);
    for (uint32_t d = 0; d < kPart; d++) {
        tile_regs_acquire();
        rescaled(d < Dt ? st.orun : st.lrun, d < Dt ? d : 0, st.alpha);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile<true>(0, cb_o, d);
        tile_regs_release();
    }
    cb_push_back(cb_o, kPart);
    cb_pop_front(st.orun, Dt);
    cb_pop_front(st.lrun, 1);
    cb_pop_front(st.mrun, 1);
    cb_pop_front(st.alpha, 1);
    cb_pop_front(cb_M, 1);
}

// sends m (cb_m), then with the global row max M: partial (o, l) * exp(m - M)
FORCE_INLINE void online_finish(const St& st) {
    send_m(st);
    finish_partial(st);
}
}  // namespace

// (S0-8: the rotated k rows go through the tail writer, which rounds them to nearest into cb_knew)
constexpr uint32_t kKnewSrc = kKRtn ? cb_knr : cb_knew;

// the core's role at compile time when the host builds the chunk workers' and the tail cores' binaries apart (each then
// carries only its own role's code: the worker cores' kernels sit near the kernel config buffer's limit)
#if defined(ATTN_CHUNK_CORE)
constexpr uint32_t kCoreRole = 1;
#elif defined(ATTN_TAIL_CORE)
constexpr uint32_t kCoreRole = 2;
#else
constexpr uint32_t kCoreRole = 0;  // (from the runtime arg)
#endif

void kernel_main() {
    const uint32_t role = get_arg_val<uint32_t>(0);
    const uint32_t children = get_arg_val<uint32_t>(1);
    cb_wait_front(cb_one, 1);
    cb_wait_front(cb_ntiles, 1);
    uint32_t n_of[kUsers];
    for (uint32_t u = 0; u < kUsers; u++) {
        n_of[u] = read_tile_value(cb_ntiles, 0, u);
    }
    const ActiveRows<kUsers> act(read_tile_value(cb_ntiles, 0, kMaskWord));
    const uint32_t n_act = act.n;
    if constexpr (kCoreRole != 2) {
        if (kCoreRole == 1 || role == 1) {
            compute_kernel_hw_startup<SrcOrder::Reverse>(cb_q, cb_k, cb_s);
            // (kSkipZeroParts: per user the children with KV tiles, from the reader)
            uint32_t cmask[kUsers];
            for (uint32_t u = 0; u < kUsers; u++) {
                cmask[u] = kSkipZeroParts && children > 0 ? read_tile_value(cb_ntiles, 0, kChildMaskOff / 4 + u) : ~0u;
            }
            if constexpr (kAttnPipe) {
                // O-2: per layer, user k + 1's scores / softmax / PV (state set (i + 1) & 1) between sending m(k) and
                // finishing k with M(k); q is released as soon as a user's scores are done
                auto heavy = [&](uint32_t i, bool pending) {
                    const uint32_t n_tiles = n_of[act.row[i % n_act]];
                    cb_wait_front(cb_q, Dt);
                    for (uint32_t b0 = 0; b0 < n_tiles; b0 += chunk_max) {
                        const uint32_t nb = n_tiles - b0 < chunk_max ? n_tiles - b0 : chunk_max;
                        cb_wait_front(cb_k, chunk_max * Dt);
                        cb_wait_front(cb_v, chunk_max * Dt);
                        scores(cb_q, cb_k, 0, nb, false);
                        online_block(nb, b0 == 0, kSt[0], pending);
                        cb_pop_front(cb_k, chunk_max * Dt);
                        cb_pop_front(cb_v, chunk_max * Dt);
                    }
                    cb_pop_front(cb_q, Dt);
                    if (n_tiles > 0) {
                        send_m(kSt[0], pending);
                    }
                };
                auto finish = [&](uint32_t i) {
                    if (n_of[act.row[i % n_act]] > 0) {
                        finish_partial(kSt[0]);
                    } else {
                        cb_wait_front(cb_M, 1);
                        zero_partial();
                        cb_pop_front(cb_M, 1);
                    }
                    if (children > 0) {
                        sum_partials<true>(children, cmask[act.row[i % n_act]]);
                    }
                };
#ifdef RESIDENT_DBG_CHUNK_SEQ  // (O-2 debug: the chunk workers without the overlap)
                for (uint32_t i = 0; i < layers * n_act; i++) {
                    heavy(i, false);
                    finish(i);
                }
#else
                for (uint32_t l = 0; l < layers && n_act > 0; l++) {
                    for (uint32_t k = 0; k <= n_act; k++) {
                        // (user k overlaps user k - 1's finish only with one block; else k - 1 finishes first — one
                        // call site per stage)
                        const bool first_finish = k > 0 && k < n_act && n_of[act.row[k]] > chunk_max;
                        for (uint32_t step = 0; step < 2; step++) {
                            if (k < n_act && step == (first_finish ? 1u : 0u)) {
                                // (the previous user's state still queued: it had KV tiles and finishes after this)
                                heavy(l * n_act + k, k > 0 && !first_finish && n_of[act.row[k - 1]] > 0);
                            }
                            if (k > 0 && step == (first_finish ? 0u : 1u)) {
                                finish(l * n_act + k - 1);
                            }
                        }
                    }
                }
#endif
                return;
            } else {
                for (uint32_t i = 0; i < layers * n_act; i++) {
                    const uint32_t n_tiles = n_of[act.row[i % n_act]];
                    cb_wait_front(cb_q, Dt);
                    if (n_tiles > 0) {
                        for (uint32_t b0 = 0; b0 < n_tiles; b0 += chunk_max) {
                            const uint32_t nb = n_tiles - b0 < chunk_max ? n_tiles - b0 : chunk_max;
                            cb_wait_front(cb_k, chunk_max * Dt);
                            cb_wait_front(cb_v, chunk_max * Dt);
                            scores(cb_q, cb_k, 0, nb, false);
                            online_block(nb, b0 == 0, kSt[0]);
                            cb_pop_front(cb_k, chunk_max * Dt);
                            cb_pop_front(cb_v, chunk_max * Dt);
                        }
                        online_finish(kSt[0]);
                    } else {
                        cb_wait_front(cb_M, 1);
                        zero_partial();
                        cb_pop_front(cb_M, 1);
                    }
                    if (children > 0) {
                        sum_partials<true>(children, cmask[act.row[i % n_act]]);
                    }
                    cb_pop_front(cb_q, Dt);
                }
            }
            return;
        }
    }
    if constexpr (kCoreRole != 1) {
        compute_kernel_hw_startup(cb_kraw, cb_kraw, cb_sq);
        cb_wait_front(cb_mean, 1);
        if constexpr (kKRot) {
            cb_wait_front(cb_had, 1);
        }
        cb_wait_front(cb_rope, kRopeTail);
        if constexpr (packed) {
            // every row's new k / v into the tail tiles once per layer, then per pass the masked tail tiles (verify
            // rows: per user gr at its first pass, its passes' masks 2 u + j, its placement tiles 4 gr ..; the active
            // passes)
            cb_wait_front(cb_mask, 2 * kUsers);
            cb_wait_front(cb_rowsel, 4 * kVUsers);
            for (uint32_t i = 0; i < layers * n_act; i++) {
                const uint32_t u = act.row[i % n_act], gr = u / kPPU, n_tail = n_of[u];  // (gr: the pass's user)
                if (u % kPPU == 0) {
                    cb_wait_front(cb_kw, Dt);
                    norm_rope(cb_kraw, cb_kw, 0, kKnewSrc, kKnewSrc, kRopeK + 3 * gr);
                    cb_pop_front(cb_kw, Dt);
                    update_tail_rows(n_tail, 4 * gr);
                    cb_wait_front(cb_tailk, tail_pages(n_tail) * Dt);
                    cb_wait_front(cb_tailv, tail_pages(n_tail) * Dt);
                }
                cb_wait_front(cb_q, Dt);
                scores(cb_q, cb_tailk, 0, n_tail, true, 2 * u);
                row_max(n_tail);
                probs_and_partial(cb_tailv, 0, n_tail);
                if (children > 0) {
                    sum_partials<true>(children);
                }
                cb_pop_front(cb_q, Dt);
                if (u % kPPU == kPPU - 1) {
                    pack_tail_bf8(n_tail);
                }
            }
            return;
        }
        cb_wait_front(cb_mask, kRows);
        cb_wait_front(cb_rowsel, 2 * kRows);
        if constexpr (kAttnPipe) {
            // O-2 (the batched single tail): per layer, user k + 1's placement, scores and row max between sending m(k)
            // and finishing k with M(k) — while k's tail tiles and scores still wait, k + 1's sit behind them (pending)
            auto front = [&](uint32_t i, bool pending) {
                const uint32_t ka = i % n_act, u = act.row[ka];
                if (ka == 0) {
                    cb_wait_front(cb_kw, Dt);
                    norm_rope(cb_kraw, cb_kw, 0, kKnewSrc, kKnewSrc, kRopeRows);
                    cb_pop_front(cb_kw, Dt);
                    cb_wait_front(cb_knew, Dt);
                    cb_wait_front(cb_vraw, Dt);
                }
                update_tail_user(2 * u);
                cb_wait_front(cb_q, Dt);
                cb_wait_front(cb_tailk, (pending ? 2 : 1) * Dt);
                if (pending) {
                    at_second(cb_tailk, Dt, true);
                }
                scores(cb_q, cb_tailk, 0, 1, true, u);
                if (pending) {
                    at_second(cb_tailk, Dt, false);
                }
                cb_pop_front(cb_q, Dt);
                if (pending) {
                    cb_wait_front(cb_s, 2 * kSBlk);
                    at_second(cb_s, kSBlk, true);
                }
                row_max(1);
                if (pending) {
                    at_second(cb_s, kSBlk, false);
                }
            };
            auto back = [&](uint32_t i) {
                cb_wait_front(cb_tailv, Dt);
                probs_and_partial(cb_tailv, 0, 1);
                if (children > 0) {
                    sum_partials<true>(children);
                }
                pack_tail_bf8();
                if (i % n_act + 1 == n_act) {
                    cb_pop_front(cb_knew, Dt);
                    cb_pop_front(cb_vraw, Dt);
                }
            };
#ifdef RESIDENT_DBG_TAIL_SEQ  // (O-2 debug: the tail without the overlap — front(k), back(k))
            for (uint32_t i = 0; i < layers * n_act; i++) {
                front(i, false);
                back(i);
            }
#else
            for (uint32_t l = 0; l < layers && n_act > 0; l++) {
                for (uint32_t k = 0; k <= n_act; k++) {  // (one call site per stage)
                    if (k < n_act) {
                        front(l * n_act + k, k > 0);
                    }
                    if (k > 0) {
                        back(l * n_act + k - 1);
                    }
                }
            }
#endif
            return;
        } else {
            for (uint32_t i = 0; i < layers * n_act; i++) {
                const uint32_t ka = i % n_act, u = act.row[ka];
                if constexpr (kTailBatched) {
                    // every active user's new k row of the layer at once (row u = user u, per-row rope), then its
                    // placement
                    if (ka == 0) {
                        cb_wait_front(cb_kw, Dt);
                        norm_rope(cb_kraw, cb_kw, 0, kKnewSrc, kKnewSrc, kRopeRows);
                        cb_pop_front(cb_kw, Dt);
                        cb_wait_front(cb_knew, Dt);
                        cb_wait_front(cb_vraw, Dt);
                    }
                    if (!tail_takes(role, ka)) {
                        // (kTails: the other tail's user — a chunk worker without a chunk)
                        cb_wait_front(cb_q, Dt);
                        cb_wait_front(cb_M, 1);
                        zero_partial();
                        cb_pop_front(cb_M, 1);
                        if (children > 0) {
                            sum_partials<true>(children);
                        }
                        cb_pop_front(cb_q, Dt);
                        if (ka + 1 == n_act) {
                            cb_pop_front(cb_knew, Dt);
                            cb_pop_front(cb_vraw, Dt);
                        }
                        continue;
                    }
                    update_tail_user(2 * u);
                } else {
                    cb_wait_front(cb_kw, Dt);
                    norm_rope(cb_kraw, cb_kw, 0, kKnewSrc, kKnewSrc, 3 * u);
                    cb_pop_front(cb_kw, Dt);
                    update_tail(2 * u);
                }
                cb_wait_front(cb_q, Dt);
                cb_wait_front(cb_tailk, Dt);
                cb_wait_front(cb_tailv, Dt);
                scores(cb_q, cb_tailk, 0, 1, true, u);
                row_max(1);
                probs_and_partial(cb_tailv, 0, 1);
                if (children > 0) {
                    sum_partials<true>(children);
                }
                cb_pop_front(cb_q, Dt);
                pack_tail_bf8();
                if constexpr (kTailBatched) {
                    if (ka + 1 == n_act) {
                        cb_pop_front(cb_knew, Dt);
                        cb_pop_front(cb_vraw, Dt);
                    }
                }
            }
        }
    }
}
