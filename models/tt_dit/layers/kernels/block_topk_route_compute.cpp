// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// block_topk_route compute. Unit = (head h, local Q-block tile row it): the 32 local Q blocks [32 it, 32 it + 32)
// against all NBGT * 32 global blocks. Per unit, all math in fp32 DST (full-sync, 8 tiles):
//
//   scores : S[c] = q_bar @ k_bar[c]^T for every column tile c, kept in cb_s (fp32) for the later phases.
//   top-k  : a running sorted buffer of KT tiles (cb_acc, in the transposed domain: row j of tile p holds position
//            32 p + j of every Q block's descending candidate scores, column r = Q block r), seeded with SENTINEL.
//            Each column tile enters as S[c]^T plus the transposed candidate bias (0 / EXCLUDED) and is inserted
//            level by level with the 64-wide SFPU bitonic sort (topk_local_sort, as ttnn.topk's insertion cascade);
//            the loser of level p moves to level p + 1, the loser of the last level is dropped.
//   k-th   : the sorted tiles go to the writer (cb_fin), which copies position k - 1 of every Q block into a
//            row-broadcast threshold tile (cb_thr_raw); compute adds the per-row bias (0 for a scored Q block,
//            UNSCORED otherwise).
//   mask   : block01[c] = (S[c] + colbias[c] >= thr): candidates compete, exempt columns are INCLUDED, other
//            columns EXCLUDED; typecast to uint32 and packed untilized into row-major pages of MC tiles.
//   coarse : (COARSE) m = row max of S + pad over all columns (elementwise max over the column tiles, then the same
//            sort on the transposed max tile and a row broadcast), p = exp(scale (S + pad - m)), l = p @ ones,
//            out = (p @ v_bar) / l; then for each real block r of the unit, row_sel[r] @ out (every row = row r) is
//            packed bf16 for the writer, which stores it at both 32-row tile rows of that block.
//
// The tiles the selection reads are unpacked straight to DST (UnpackToDestFp32, set by the host), so the sort, the
// threshold and the compare see the fp32 scores unrounded. Unary SFPU ops only touch DST tiles 0..3; the binary
// SFPU ops use 4 and 5 as scratch.
//
// Compile-time args: 0 H, 1 NBPT, 2 NBGT, 3 DT, 4 KT, 5 MC, 6 NX, 7 NY, 8 UNITS, 9 COARSE, 10 SCALE (fp32 bits),
// 11 RT (32-row tile rows per shard).

#include <cstdint>

#include "api/compute/binary_max_min.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/exp.h"
#include "api/compute/eltwise_unary/typecast.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/pack_untilize.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/reg_api.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/topk.h"
#include "api/compute/transpose.h"
#include "api/dataflow/circular_buffer.h"

namespace {
constexpr uint32_t cb_q = 0, cb_k = 1, cb_const = 2, cb_cand_t = 7, cb_colbias = 8, cb_rowbias = 9, cb_pad = 10,
                   cb_v = 11, cb_x = 12, cb_y = 13, cb_mb = 14, cb_l = 15, cb_mask = 16, cb_co = 17, cb_mx = 18,
                   cb_fin = 19, cb_thr_raw = 20, cb_cot = 22, cb_s = 24, cb_acc = 25, cb_loser = 26, cb_thr = 28,
                   cb_p = 29;
// Constant tiles: eye, SENTINEL, ones, then row_sel[r] (column r ones: row_sel[r] @ T copies row r of T to every row).
constexpr uint32_t EYE = 0, SENT = 1, ONES = 2, ROW_SEL = 3, N_CONST = ROW_SEL + 32;
constexpr uint32_t QUAD = 4;
constexpr int SORT_END_PHASE = 5;

ALWI void mm_begin(uint32_t in0, uint32_t in1, uint32_t transpose) {
    reconfig_data_format<SrcOrder::Reverse>(in0, in1);
    matmul_init(in0, in1, transpose);
}

ALWI void copy_begin(uint32_t cb) {
    reconfig_data_format_srca(cb);
    copy_init(cb);
}

// Descending 64-wide sort of DST tiles 0 and 1 (column-wise, in the transposed domain): DST 0 receives the larger 32.
ALWI void sort_pair() {
    ckernel::topk_tile_init();
    ckernel::topk_local_sort(0, 0, SORT_END_PHASE);
}

ALWI void pack_one(uint32_t dst, uint32_t cb) {
    CircularBuffer out(cb);
    out.reserve_back(1);
    pack_tile(dst, cb);
    out.push_back(1);
}
}  // namespace

void kernel_main() {
    constexpr uint32_t NBPT = get_compile_time_arg_val(1);
    constexpr uint32_t NBGT = get_compile_time_arg_val(2);
    constexpr uint32_t DT = get_compile_time_arg_val(3);
    constexpr uint32_t KT = get_compile_time_arg_val(4);
    constexpr uint32_t MC = get_compile_time_arg_val(5);
    constexpr uint32_t NX = get_compile_time_arg_val(6);
    constexpr uint32_t NY = get_compile_time_arg_val(7);
    constexpr uint32_t UNITS = get_compile_time_arg_val(8);
    constexpr bool COARSE = get_compile_time_arg_val(9) == 1;
    constexpr uint32_t SCALE = get_compile_time_arg_val(10);
    constexpr uint32_t RT = get_compile_time_arg_val(11);
    constexpr uint32_t NB = (RT + 1) / 2;
    static_assert(NBGT % 8 == 0 && MC % QUAD == 0 && NBGT % MC == 0, "column tiles go in batches of 8 and 4");
    static_assert(DT < 8, "the coarse output and its row sum share DST");
    static_assert(NBPT > 0);

    constexpr uint32_t NCORES = NX * NY;
    const uint32_t ci = static_cast<uint32_t>(get_absolute_logical_y()) * NX + get_absolute_logical_x();
    const uint32_t unit_start = ci * UNITS / NCORES;
    const uint32_t unit_count = (ci + 1) * UNITS / NCORES - unit_start;
    compute_kernel_hw_startup(cb_q, cb_k, cb_s);
    if (unit_count == 0) {
        return;
    }
    CircularBuffer cq(cb_q), ck(cb_k), cconst(cb_const), ccand(cb_cand_t), ccol(cb_colbias), crow(cb_rowbias),
        cpad(cb_pad), cv(cb_v), cx(cb_x), cy(cb_y), cmb(cb_mb), cl(cb_l), cmask(cb_mask), cco(cb_co), ccot(cb_cot),
        cmx(cb_mx), cthr_raw(cb_thr_raw), cs(cb_s), cacc(cb_acc), closer(cb_loser), cthr(cb_thr), cp(cb_p);
    cconst.wait_front(N_CONST);

    for (uint32_t u = unit_start; u < unit_start + unit_count; ++u) {
        // ---- scores
        cq.wait_front(DT);
        cs.reserve_back(NBGT);
        pack_reconfig_data_format(cb_s);
        for (uint32_t c0 = 0; c0 < NBGT; c0 += 8) {
            tile_regs_acquire();
            mm_begin(cb_q, cb_k, 1);
            for (uint32_t cc = 0; cc < 8; ++cc) {
                ck.wait_front(DT);
                for (uint32_t n = 0; n < DT; ++n) {
                    matmul_tiles(cb_q, cb_k, n, n, cc);
                }
                ck.pop_front(DT);
            }
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t cc = 0; cc < 8; ++cc) {
                pack_tile<true>(cc, cb_s, c0 + cc);
            }
            tile_regs_release();
        }
        cs.push_back(NBGT);
        cs.wait_front(NBGT);
        cq.pop_front(DT);

        // ---- running top-k over the candidate columns
        for (uint32_t p = 0; p < KT; ++p) {
            tile_regs_acquire();
            copy_begin(cb_const);
            copy_tile(cb_const, SENT, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_one(0, cb_acc);
            tile_regs_release();
        }
        for (uint32_t c = 0; c < NBGT; ++c) {
            tile_regs_acquire();
            reconfig_data_format_srca(cb_s);
            transpose_init(cb_s);
            transpose_tile(cb_s, c, 1);
            ccand.wait_front(1);
            copy_begin(cb_cand_t);
            copy_tile(cb_cand_t, 0, 4);
            ccand.pop_front(1);
            add_binary_tile_init();
            add_binary_tile(1, 4, 1);
            cacc.wait_front(1);
            copy_begin(cb_acc);
            copy_tile(cb_acc, 0, 0);
            cacc.pop_front(1);
            sort_pair();
            tile_regs_commit();
            tile_regs_wait();
            pack_one(0, cb_acc);
            if constexpr (KT > 1) {
                pack_one(1, cb_loser);
            }
            tile_regs_release();
            for (uint32_t p = 1; p < KT; ++p) {
                tile_regs_acquire();
                cacc.wait_front(1);
                copy_begin(cb_acc);
                copy_tile(cb_acc, 0, 0);
                cacc.pop_front(1);
                closer.wait_front(1);
                copy_begin(cb_loser);
                copy_tile(cb_loser, 0, 1);
                closer.pop_front(1);
                sort_pair();
                tile_regs_commit();
                tile_regs_wait();
                pack_one(0, cb_acc);
                if (p + 1 < KT) {
                    pack_one(1, cb_loser);
                }
                tile_regs_release();
            }
        }

        // ---- k-th value per Q block (row-broadcast by the writer), plus the unscored-row bias
        for (uint32_t p = 0; p < KT; ++p) {
            tile_regs_acquire();
            cacc.wait_front(1);
            copy_begin(cb_acc);
            copy_tile(cb_acc, 0, 0);
            cacc.pop_front(1);
            tile_regs_commit();
            tile_regs_wait();
            pack_one(0, cb_fin);
            tile_regs_release();
        }
        tile_regs_acquire();
        cthr_raw.wait_front(1);
        copy_begin(cb_thr_raw);
        copy_tile(cb_thr_raw, 0, 0);
        cthr_raw.pop_front(1);
        crow.wait_front(1);
        copy_begin(cb_rowbias);
        copy_tile(cb_rowbias, 0, 1);
        crow.pop_front(1);
        add_binary_tile_init();
        add_binary_tile(0, 1, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_one(0, cb_thr);
        tile_regs_release();

        // ---- block mask rows
        cthr.wait_front(1);
        pack_reconfig_data_format(cb_mask);
        pack_untilize_dest_init<QUAD, MC>(cb_mask);
        for (uint32_t chunk = 0; chunk < NBGT / MC; ++chunk) {
            cmask.reserve_back(1);
            for (uint32_t b = 0; b < MC / QUAD; ++b) {
                const uint32_t c0 = chunk * MC + b * QUAD;
                tile_regs_acquire();
                copy_begin(cb_s);
                for (uint32_t i = 0; i < QUAD; ++i) {
                    copy_tile(cb_s, c0 + i, i);
                }
                copy_begin(cb_thr);
                copy_tile(cb_thr, 0, 5);
                for (uint32_t i = 0; i < QUAD; ++i) {
                    ccol.wait_front(1);
                    copy_begin(cb_colbias);
                    copy_tile(cb_colbias, 0, 4);
                    ccol.pop_front(1);
                    add_binary_tile_init();
                    add_binary_tile(i, 4, i);
                    ge_binary_tile_init();
                    ge_binary_tile(i, 5, i);
                }
                typecast_tile_init<
                    static_cast<uint32_t>(DataFormat::Float32),
                    static_cast<uint32_t>(DataFormat::UInt32)>();
                for (uint32_t i = 0; i < QUAD; ++i) {
                    typecast_tile<
                        static_cast<uint32_t>(DataFormat::Float32),
                        static_cast<uint32_t>(DataFormat::UInt32)>(i);
                }
                tile_regs_commit();
                tile_regs_wait();
                pack_untilize_dest<QUAD, MC>(cb_mask, 1, b);
                tile_regs_release();
            }
            cmask.push_back(1);
        }
        pack_untilize_uninit(cb_mask);
        cthr.pop_front(1);
        pack_reconfig_data_format(cb_s);

        if constexpr (COARSE) {
            // ---- row max over the real columns: elementwise max of S + pad over the column tiles ...
            tile_regs_acquire();
            for (uint32_t c = 0; c < NBGT; ++c) {
                const uint32_t dst = c == 0 ? 0 : 1;
                copy_begin(cb_s);
                copy_tile(cb_s, c, dst);
                cpad.wait_front(1);
                copy_begin(cb_pad);
                copy_tile(cb_pad, 0, 2);
                cpad.pop_front(1);
                add_binary_tile_init();
                add_binary_tile(dst, 2, dst);
                if (c > 0) {
                    binary_max_tile_init();
                    binary_max_tile(0, 1, 0);
                }
            }
            tile_regs_commit();
            tile_regs_wait();
            pack_one(0, cb_mx);
            tile_regs_release();

            // ... then the max of each row of that tile: transposed, sorted against SENTINEL (row 0 = max) ...
            tile_regs_acquire();
            cmx.wait_front(1);
            mm_begin(cb_const, cb_mx, 1);
            matmul_tiles(cb_const, cb_mx, EYE, 0, 0);
            cmx.pop_front(1);
            copy_begin(cb_const);
            copy_tile(cb_const, SENT, 1);
            sort_pair();
            tile_regs_commit();
            tile_regs_wait();
            pack_one(0, cb_x);
            tile_regs_release();

            // ... copied to every row and transposed back: mb[t, :] = m[t].
            tile_regs_acquire();
            cx.wait_front(1);
            mm_begin(cb_const, cb_x, 0);
            matmul_tiles(cb_const, cb_x, ROW_SEL, 0, 0);
            cx.pop_front(1);
            tile_regs_commit();
            tile_regs_wait();
            pack_one(0, cb_y);
            tile_regs_release();

            tile_regs_acquire();
            cy.wait_front(1);
            mm_begin(cb_const, cb_y, 1);
            matmul_tiles(cb_const, cb_y, EYE, 0, 0);
            cy.pop_front(1);
            tile_regs_commit();
            tile_regs_wait();
            pack_one(0, cb_mb);
            tile_regs_release();

            // ---- p = exp(scale (S + pad - m))
            cmb.wait_front(1);
            cp.reserve_back(NBGT);
            for (uint32_t c0 = 0; c0 < NBGT; c0 += QUAD) {
                tile_regs_acquire();
                copy_begin(cb_s);
                for (uint32_t i = 0; i < QUAD; ++i) {
                    copy_tile(cb_s, c0 + i, i);
                }
                copy_begin(cb_mb);
                copy_tile(cb_mb, 0, 5);
                for (uint32_t i = 0; i < QUAD; ++i) {
                    cpad.wait_front(1);
                    copy_begin(cb_pad);
                    copy_tile(cb_pad, 0, 4);
                    cpad.pop_front(1);
                    add_binary_tile_init();
                    add_binary_tile(i, 4, i);
                    sub_binary_tile_init();
                    sub_binary_tile(i, 5, i);
                }
                binop_with_scalar_tile_init();
                for (uint32_t i = 0; i < QUAD; ++i) {
                    mul_unary_tile(i, SCALE);
                }
                exp_tile_init();
                for (uint32_t i = 0; i < QUAD; ++i) {
                    exp_tile(i);
                }
                tile_regs_commit();
                tile_regs_wait();
                for (uint32_t i = 0; i < QUAD; ++i) {
                    pack_tile<true>(i, cb_p, c0 + i);
                }
                tile_regs_release();
            }
            cp.push_back(NBGT);
            cp.wait_front(NBGT);
            cmb.pop_front(1);

            // ---- l = p @ ones (row sums in every column)
            tile_regs_acquire();
            mm_begin(cb_p, cb_const, 0);
            for (uint32_t c = 0; c < NBGT; ++c) {
                matmul_tiles(cb_p, cb_const, c, ONES, 0);
            }
            tile_regs_commit();
            tile_regs_wait();
            pack_one(0, cb_l);
            tile_regs_release();

            // ---- out = (p @ v_bar) / l
            tile_regs_acquire();
            mm_begin(cb_p, cb_v, 0);
            for (uint32_t c = 0; c < NBGT; ++c) {
                cv.wait_front(DT);
                for (uint32_t n = 0; n < DT; ++n) {
                    matmul_tiles(cb_p, cb_v, c, n, n);
                }
                cv.pop_front(DT);
            }
            cl.wait_front(1);
            copy_begin(cb_l);
            copy_tile(cb_l, 0, DT);
            cl.pop_front(1);
            div_binary_tile_init();
            for (uint32_t n = 0; n < DT; ++n) {
                div_binary_tile(n, DT, n);
            }
            tile_regs_commit();
            tile_regs_wait();
            pack_reconfig_data_format(cb_cot);
            ccot.reserve_back(DT);
            for (uint32_t n = 0; n < DT; ++n) {
                pack_tile(n, cb_cot);
            }
            ccot.push_back(DT);
            tile_regs_release();
            cp.pop_front(NBGT);

            // ---- every block's row copied to all rows of its tiles
            const uint32_t it = u % NBPT;
            ccot.wait_front(DT);
            for (uint32_t r = 0; r < 32 && it * 32 + r < NB; ++r) {
                tile_regs_acquire();
                mm_begin(cb_const, cb_cot, 0);
                for (uint32_t n = 0; n < DT; ++n) {
                    matmul_tiles(cb_const, cb_cot, ROW_SEL + r, n, n);
                }
                tile_regs_commit();
                tile_regs_wait();
                cco.reserve_back(DT);
                for (uint32_t n = 0; n < DT; ++n) {
                    pack_tile(n, cb_co);
                }
                cco.push_back(DT);
                tile_regs_release();
            }
            ccot.pop_front(DT);
            pack_reconfig_data_format(cb_s);
        }
        cs.pop_front(NBGT);
    }
}
