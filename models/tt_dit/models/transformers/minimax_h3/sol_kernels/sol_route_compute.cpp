// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Sol-Attn routing pre-pass, compute. Unit = (head h, Q-chunk tile row it): the routing of the 32 local Q chunks
// [32*it, 32*it+32) of head h against every shard's pooled K blocks. See sol_route_reader.cpp for the inputs.
//
// Per unit (all math in fp32 DST, operands bf16 tiles):
//   moments  : part[n] = sum_c kc(c,n) and sum_c kc(c,n)^2 tile-wise (copy/mul + SFPU add), reduced over rows by a
//              ones-row matmul and scaled by 1/n_valid -> mean, E2; var = relu(E2 - mean^2) (row 0 of DT tiles each)
//   threshold: for the unit's Q-block tile rows t in [t0, t1): thr[t] = q_bar[t] @ mean^T + tau * sqrt(q_bar[t]^2 @
//   var^T
//              + eps') (column 0; transposed matmuls with the moment rows), broadcast across columns by thrcol @
//              ones_row
//   per column shard sp:
//     scores : exact[t,c] = gtz(q_bar[t] @ kc[c]^T - thr[t])   (raw units: the base-2 scale is folded into thr)
//     union  : block01[c] = gtz(qsel[it,t0:t1] @ exact[:,c] + const_or[it,c] - 0.5) * full[c]
//     chunk  : chunk01[ct] = gtz(block01 @ ksel[:,ct] - 0.5)          (MODE 0: the mask; MODE 2: the union row)
//              kept[j] = gtz(chunk01 @ ksel[j,:]^T - 0.5) * valid[j]; brow[j] = kept * NEG + len_bias[j]  (MODE 0)
//              brow[j] = block01[j] * NEG + len_bias[j]                                                 (MODE 1/2)
//     bias   : for every real Q chunk i = 32*it + r: erow[r] @ brow[j] (all rows = row r)              (EMIT_BIAS)
// Outputs to the writer per shard: mask tiles (chunk01 in MODE 0, block01 otherwise), union tiles (MODE 2), bias tiles.
//
// Compile-time args: 0 H, 1 S, 2 NBPT, 3 DT, 4 NQCT, 5 NKCLT, 6 NQC, 7 NKC, 8 NKCP, 9 MODE (0 chunk, 1 gather, 2
// compact), 10 EMIT_BIAS, 11 TAU, 12 EPS2, 13 NEG, 14 INV_NV (fp32 bit patterns), 15 TQ (Q-block tile rows per Q-chunk
// tile row = q_chunk / 64). Per-core: none.

#include <cstdint>

#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/comp.h"
#include "api/compute/eltwise_unary/relu.h"
#include "api/compute/eltwise_unary/sqrt.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/reg_api.h"
#include "api/compute/tile_move_copy.h"
#include "api/dataflow/circular_buffer.h"

namespace {
constexpr uint32_t cb_kc = 0, cb_q = 1, cb_ones = 2, cb_qsel = 3, cb_ksel = 4, cb_erow = 5, cb_constor = 6, cb_colc = 7,
                   cb_part = 8, cb_mom = 9, cb_qsq = 10, cb_thrcol = 11, cb_thr = 12, cb_exact = 13, cb_block01 = 14,
                   cb_chunk2 = 15, cb_mask_out = 16, cb_brow = 17, cb_bias_out = 18, cb_var = 20, cb_union_out = 21;
constexpr uint32_t HALF_NEG = 0xBF000000u;  // -0.5f
// Measured on device (tau scan, 2026-10-16): SFPU unary ops (binop-with-scalar, sqrt, relu, gtz) applied to fp32 DST
// tiles >= 4 do not take effect, while matmul/copy/mul and the SFPU binary ops work on tiles 0..7. Every unary SFPU op
// below therefore targets DST tiles 0..3 only: column slabs of SLAB tiles with the scratch tile at index SLAB.
constexpr uint32_t SLAB = 3;        // DST tiles 0..2 hold a column slab, 3 is scratch
constexpr uint32_t QUAD = 4;        // batch size for sections that apply unary SFPU ops
constexpr uint32_t BIAS_BATCH = 8;  // bias tiles per DST batch; always reserved/pushed in full
inline uint32_t min_u(uint32_t a, uint32_t b) { return a < b ? a : b; }
}  // namespace

void kernel_main() {
    constexpr uint32_t S = get_compile_time_arg_val(1);
    constexpr uint32_t NBPT = get_compile_time_arg_val(2);
    constexpr uint32_t DT = get_compile_time_arg_val(3);
    constexpr uint32_t NQCT = get_compile_time_arg_val(4);
    constexpr uint32_t NKCLT = get_compile_time_arg_val(5);
    constexpr uint32_t NQC = get_compile_time_arg_val(6);
    constexpr uint32_t MODE = get_compile_time_arg_val(9);
    constexpr uint32_t EMIT_BIAS = get_compile_time_arg_val(10);
    constexpr uint32_t TAU = get_compile_time_arg_val(11);
    constexpr uint32_t EPS2 = get_compile_time_arg_val(12);
    constexpr uint32_t NEG = get_compile_time_arg_val(13);
    constexpr uint32_t INV_NV = get_compile_time_arg_val(14);
    constexpr uint32_t TQ = get_compile_time_arg_val(15);
    constexpr uint32_t DEBUG = get_compile_time_arg_val(16);  // 1: emit mean/E2/var tiles instead of the bias
    constexpr uint32_t NBGT = S * NBPT;
    constexpr uint32_t KC_T = NBGT * DT;
    constexpr uint32_t Q_T = NBPT * DT;
    constexpr uint32_t MASK_T = MODE == 0 ? NKCLT : NBPT;  // per shard
    // DST tiles 0..7 are used (fp32): the program must run with dst_full_sync_en (half-sync fp32 has only 4 tiles).
    static_assert(DT <= 4, "moments use DST 0..3 (sums) and 4..7 (sums of squares)");
    static_assert(NKCLT <= 8, "chunk tiles are processed in DST batches of QUAD");
    static_assert(TQ <= 8, "Q-block tile rows per Q-chunk tile row must fit one DST batch");

    constexpr uint32_t NX = get_compile_time_arg_val(17);
    constexpr uint32_t NY = get_compile_time_arg_val(18);
    constexpr uint32_t UNITS = get_compile_time_arg_val(19);
    // Unit range of this core from its logical coordinates (no per-core runtime args): units [ci*U/N, (ci+1)*U/N).
    constexpr uint32_t NCORES = NX * NY;
    const uint32_t ci = static_cast<uint32_t>(get_absolute_logical_y()) * NX + get_absolute_logical_x();
    const uint32_t unit_start = ci * UNITS / NCORES;
    const uint32_t unit_count = (ci + 1) * UNITS / NCORES - unit_start;
    compute_kernel_hw_startup(cb_q, cb_kc, cb_bias_out);
    if (unit_count == 0) {
        return;
    }
    CircularBuffer kcb(cb_kc), qb(cb_q), ones(cb_ones), qsel(cb_qsel), ksel(cb_ksel), erow(cb_erow),
        constor(cb_constor), colc(cb_colc), part(cb_part), mom(cb_mom), qsq(cb_qsq), thrcol(cb_thrcol), thr(cb_thr),
        exact(cb_exact), block01(cb_block01), chunk2(cb_chunk2), mask_out(cb_mask_out), brow(cb_brow),
        bias_out(cb_bias_out), var(cb_var), union_out(cb_union_out);

    ones.wait_front(1);
    qsel.wait_front(NQCT * NBPT);
    ksel.wait_front(NBPT * NKCLT);
    erow.wait_front(32);

    for (uint32_t u = unit_start; u < unit_start + unit_count; ++u) {
        const uint32_t it = u % NQCT;
        const uint32_t t0 = min_u(it * TQ, NBPT);
        const uint32_t t1 = min_u(t0 + TQ, NBPT);
        const uint32_t nt = t1 - t0;  // Q-block tile rows of this Q-chunk tile row (0 when it is all padding)
        kcb.wait_front(KC_T);
        qb.wait_front(Q_T);

        // ---- A/B: tile-wise partial sums and sums of squares over the whole gathered pooled K.
        part.reserve_back(8);
        tile_regs_acquire();
        copy_init(cb_kc);
        for (uint32_t n = 0; n < DT; ++n) {
            copy_tile(cb_kc, n * NBGT, n);
        }
        add_binary_tile_init();
        for (uint32_t c = 1; c < NBGT; ++c) {
            for (uint32_t n = 0; n < DT; ++n) {
                copy_tile(cb_kc, n * NBGT + c, 4);
                add_binary_tile(n, 4, n);
            }
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t n = 0; n < DT; ++n) {
            pack_tile<true>(n, cb_part, n);
        }
        tile_regs_release();

        // Squares via copy + SFPU multiply (mul_tiles with one CB as both operands is not a verified pattern).
        tile_regs_acquire();
        copy_init(cb_kc);
        for (uint32_t n = 0; n < DT; ++n) {
            copy_tile(cb_kc, n * NBGT, n);
        }
        mul_binary_tile_init();
        for (uint32_t n = 0; n < DT; ++n) {
            mul_binary_tile(n, n, n);
        }
        for (uint32_t c = 1; c < NBGT; ++c) {
            for (uint32_t n = 0; n < DT; ++n) {
                copy_tile(cb_kc, n * NBGT + c, 4);
                mul_binary_tile_init();
                mul_binary_tile(4, 4, 4);
                add_binary_tile_init();
                add_binary_tile(n, 4, n);
            }
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t n = 0; n < DT; ++n) {
            pack_tile<true>(n, cb_part, 4 + n);
        }
        tile_regs_release();
        part.push_back(8);
        part.wait_front(8);

        // ---- C: reduce the partial tiles over their rows (ones-row matmul), scale by 1/n_valid; two batches of 4.
        mom.reserve_back(8);
        for (uint32_t b0 = 0; b0 < 8; b0 += QUAD) {
            tile_regs_acquire();
            matmul_init(cb_ones, cb_part, 0);
            for (uint32_t i = 0; i < QUAD; ++i) {
                matmul_tiles(cb_ones, cb_part, 0, b0 + i, i);
            }
            binop_with_scalar_tile_init();
            for (uint32_t i = 0; i < QUAD; ++i) {
                mul_unary_tile(i, INV_NV);
            }
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t i = 0; i < QUAD; ++i) {
                pack_tile<true>(i, cb_mom, b0 + i);
            }
            tile_regs_release();
        }
        mom.push_back(8);
        mom.wait_front(8);
        part.pop_front(8);

        // ---- D: var = relu(E2 - mean^2), two dims-tiles per section (DST 0..3).
        var.reserve_back(DT);
        for (uint32_t n0 = 0; n0 < DT; n0 += 2) {
            const uint32_t nn = min_u(2, DT - n0);
            tile_regs_acquire();
            copy_init(cb_mom);
            for (uint32_t k = 0; k < nn; ++k) {
                copy_tile(cb_mom, n0 + k, 2 * k);          // mean
                copy_tile(cb_mom, 4 + n0 + k, 2 * k + 1);  // E2
            }
            mul_binary_tile_init();
            for (uint32_t k = 0; k < nn; ++k) {
                mul_binary_tile(2 * k, 2 * k, 2 * k);
            }
            sub_binary_tile_init();
            for (uint32_t k = 0; k < nn; ++k) {
                sub_binary_tile(2 * k + 1, 2 * k, 2 * k + 1);
            }
            relu_tile_init();
            for (uint32_t k = 0; k < nn; ++k) {
                relu_tile(2 * k + 1);
            }
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t k = 0; k < nn; ++k) {
                pack_tile<true>(2 * k + 1, cb_var, n0 + k);
            }
            tile_regs_release();
        }
        var.push_back(DT);
        var.wait_front(DT);

        if constexpr (DEBUG == 1) {  // moments out: batch 1 = mom tiles 0..7, batch 2 = var tiles 0..DT-1 (+ mom 0..3
                                     // again)
            for (uint32_t b = 0; b < 2; ++b) {
                bias_out.reserve_back(BIAS_BATCH);
                tile_regs_acquire();
                copy_init(b == 0 ? cb_mom : cb_var);
                for (uint32_t i = 0; i < BIAS_BATCH; ++i) {
                    if (b == 0) {
                        copy_tile(cb_mom, i, i);
                    } else if (i < DT) {
                        copy_tile(cb_var, i, i);
                    } else {
                        copy_init(cb_mom);
                        copy_tile(cb_mom, i - DT, i);
                        copy_init(cb_var);
                    }
                }
                tile_regs_commit();
                tile_regs_wait();
                for (uint32_t i = 0; i < BIAS_BATCH; ++i) {
                    pack_tile<true>(i, cb_bias_out, i);
                }
                tile_regs_release();
                bias_out.push_back(BIAS_BATCH);
            }
        }

        // ---- E: q_bar squared for the unit's Q-block tile rows.
        qsq.reserve_back(TQ * DT);
        for (uint32_t i0 = 0; i0 < nt * DT; i0 += 8) {
            const uint32_t ni = min_u(8, nt * DT - i0);
            tile_regs_acquire();
            copy_init(cb_q);
            for (uint32_t i = 0; i < ni; ++i) {
                copy_tile(cb_q, t0 * DT + i0 + i, i);
            }
            mul_binary_tile_init();
            for (uint32_t i = 0; i < ni; ++i) {
                mul_binary_tile(i, i, i);
            }
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t i = 0; i < ni; ++i) {
                pack_tile<true>(i, cb_qsq, i0 + i);
            }
            tile_regs_release();
        }
        qsq.push_back(TQ * DT);
        qsq.wait_front(TQ * DT);

        // ---- F: the threshold column per Q-block tile row.
        thrcol.reserve_back(TQ);
        for (uint32_t tt = 0; tt < nt; ++tt) {
            tile_regs_acquire();
            matmul_init(cb_q, cb_mom, 1);
            for (uint32_t n = 0; n < DT; ++n) {
                matmul_tiles(cb_q, cb_mom, (t0 + tt) * DT + n, n, 0);
            }
            matmul_init(cb_qsq, cb_var, 1);
            for (uint32_t n = 0; n < DT; ++n) {
                matmul_tiles(cb_qsq, cb_var, tt * DT + n, n, 1);
            }
            binop_with_scalar_tile_init();
            add_unary_tile(1, EPS2);
            sqrt_tile_init();
            sqrt_tile(1);
            binop_with_scalar_tile_init();
            mul_unary_tile(1, TAU);
            add_binary_tile_init();
            add_binary_tile(0, 1, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile<true>(0, cb_thrcol, tt);
            tile_regs_release();
        }
        thrcol.push_back(TQ);
        thrcol.wait_front(TQ);

        // ---- G: broadcast the threshold column across the 32 columns.
        thr.reserve_back(TQ);
        for (uint32_t tb = 0; tb < nt; tb += QUAD) {
            const uint32_t ntb = min_u(QUAD, nt - tb);
            tile_regs_acquire();
            matmul_init(cb_thrcol, cb_ones, 0);
            for (uint32_t tt = 0; tt < ntb; ++tt) {
                matmul_tiles(cb_thrcol, cb_ones, tb + tt, 0, tt);
            }
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t tt = 0; tt < ntb; ++tt) {
                pack_tile<true>(tt, cb_thr, tb + tt);
            }
            tile_regs_release();
        }
        thr.push_back(TQ);
        thr.wait_front(TQ);

        for (uint32_t sp = 0; sp < S; ++sp) {
            constor.wait_front(NBPT);
            colc.wait_front(3 * NBPT);

            // ---- H: proxy scores against shard sp's blocks, compared with the threshold.
            exact.reserve_back(TQ * NBPT);
            for (uint32_t tt = 0; tt < nt; ++tt) {
                for (uint32_t c0 = 0; c0 < NBPT; c0 += SLAB) {
                    const uint32_t nc = min_u(SLAB, NBPT - c0);
                    tile_regs_acquire();
                    matmul_init(cb_q, cb_kc, 1);
                    for (uint32_t cc = 0; cc < nc; ++cc) {
                        for (uint32_t n = 0; n < DT; ++n) {
                            matmul_tiles(cb_q, cb_kc, (t0 + tt) * DT + n, n * NBGT + sp * NBPT + c0 + cc, cc);
                        }
                    }
                    copy_init(cb_thr);
                    copy_tile(cb_thr, tt, SLAB);
                    for (uint32_t cc = 0; cc < nc; ++cc) {
                        sub_binary_tile_init();
                        sub_binary_tile(cc, SLAB, cc);
                        gtz_tile_init();
                        gtz_tile(cc);
                    }
                    tile_regs_commit();
                    tile_regs_wait();
                    for (uint32_t cc = 0; cc < nc; ++cc) {
                        pack_tile<true>(cc, cb_exact, tt * NBPT + c0 + cc);
                    }
                    tile_regs_release();
                }
            }
            exact.push_back(TQ * NBPT);
            exact.wait_front(TQ * NBPT);

            // ---- I: OR over the Q blocks of each Q chunk, OR the constant mask, AND full blocks.
            block01.reserve_back(NBPT);
            if constexpr (MODE != 0) {
                mask_out.reserve_back(MASK_T);
            }
            for (uint32_t c0 = 0; c0 < NBPT; c0 += SLAB) {
                const uint32_t nc = min_u(SLAB, NBPT - c0);
                tile_regs_acquire();
                if (nt > 0) {
                    matmul_init(cb_qsel, cb_exact, 0);
                    for (uint32_t cc = 0; cc < nc; ++cc) {
                        for (uint32_t tt = 0; tt < nt; ++tt) {
                            matmul_tiles(cb_qsel, cb_exact, it * NBPT + t0 + tt, tt * NBPT + c0 + cc, cc);
                        }
                    }
                }
                for (uint32_t cc = 0; cc < nc; ++cc) {
                    const uint32_t c = c0 + cc;
                    copy_init(cb_constor);
                    if (nt > 0) {
                        copy_tile(cb_constor, c, SLAB);
                        add_binary_tile_init();
                        add_binary_tile(cc, SLAB, cc);
                    } else {
                        copy_tile(cb_constor, c, cc);
                    }
                    binop_with_scalar_tile_init();
                    add_unary_tile(cc, HALF_NEG);
                    gtz_tile_init();
                    gtz_tile(cc);
                    copy_init(cb_colc);
                    copy_tile(cb_colc, c, SLAB);  // full-block row
                    mul_binary_tile_init();
                    mul_binary_tile(cc, SLAB, cc);
                }
                tile_regs_commit();
                tile_regs_wait();
                for (uint32_t cc = 0; cc < nc; ++cc) {
                    pack_tile<true>(cc, cb_block01, c0 + cc);
                    if constexpr (MODE != 0) {
                        pack_tile<true>(cc, cb_mask_out, c0 + cc);
                    }
                }
                tile_regs_release();
            }
            block01.push_back(NBPT);
            block01.wait_front(NBPT);
            if constexpr (MODE != 0) {
                mask_out.push_back(MASK_T);
            }

            // ---- J: chunk union (MODE 0: the mask, MODE 2: the union row).
            if constexpr (MODE != 1) {
                if constexpr (MODE == 0) {
                    chunk2.reserve_back(NKCLT);
                    mask_out.reserve_back(MASK_T);
                } else {
                    union_out.reserve_back(NKCLT);
                }
                for (uint32_t ct0 = 0; ct0 < NKCLT; ct0 += QUAD) {
                    const uint32_t nct = min_u(QUAD, NKCLT - ct0);
                    tile_regs_acquire();
                    matmul_init(cb_block01, cb_ksel, 0);
                    for (uint32_t cc = 0; cc < nct; ++cc) {
                        for (uint32_t j = 0; j < NBPT; ++j) {
                            matmul_tiles(cb_block01, cb_ksel, j, j * NKCLT + ct0 + cc, cc);
                        }
                    }
                    for (uint32_t cc = 0; cc < nct; ++cc) {
                        binop_with_scalar_tile_init();
                        add_unary_tile(cc, HALF_NEG);
                        gtz_tile_init();
                        gtz_tile(cc);
                    }
                    tile_regs_commit();
                    tile_regs_wait();
                    for (uint32_t cc = 0; cc < nct; ++cc) {
                        if constexpr (MODE == 0) {
                            pack_tile<true>(cc, cb_chunk2, ct0 + cc);
                            pack_tile<true>(cc, cb_mask_out, ct0 + cc);
                        } else {
                            pack_tile<true>(cc, cb_union_out, ct0 + cc);
                        }
                    }
                    tile_regs_release();
                }
                if constexpr (MODE == 0) {
                    chunk2.push_back(NKCLT);
                    mask_out.push_back(MASK_T);
                    chunk2.wait_front(NKCLT);
                } else {
                    union_out.push_back(NKCLT);
                }
            }

            // ---- K: the per-block bias row.
            brow.reserve_back(NBPT);
            for (uint32_t j0 = 0; j0 < NBPT; j0 += SLAB) {
                const uint32_t nj = min_u(SLAB, NBPT - j0);
                tile_regs_acquire();
                if constexpr (MODE == 0) {
                    matmul_init(cb_chunk2, cb_ksel, 1);
                    for (uint32_t jj = 0; jj < nj; ++jj) {
                        for (uint32_t ct = 0; ct < NKCLT; ++ct) {
                            matmul_tiles(cb_chunk2, cb_ksel, ct, (j0 + jj) * NKCLT + ct, jj);
                        }
                    }
                    for (uint32_t jj = 0; jj < nj; ++jj) {
                        binop_with_scalar_tile_init();
                        add_unary_tile(jj, HALF_NEG);
                        gtz_tile_init();
                        gtz_tile(jj);
                        copy_init(cb_colc);
                        copy_tile(cb_colc, NBPT + j0 + jj, SLAB);  // valid-block row
                        mul_binary_tile_init();
                        mul_binary_tile(jj, SLAB, jj);
                    }
                } else {
                    copy_init(cb_block01);
                    for (uint32_t jj = 0; jj < nj; ++jj) {
                        copy_tile(cb_block01, j0 + jj, jj);
                    }
                }
                for (uint32_t jj = 0; jj < nj; ++jj) {
                    binop_with_scalar_tile_init();
                    mul_unary_tile(jj, NEG);
                    copy_init(cb_colc);
                    copy_tile(cb_colc, 2 * NBPT + j0 + jj, SLAB);  // length-bias row
                    add_binary_tile_init();
                    add_binary_tile(jj, SLAB, jj);
                }
                tile_regs_commit();
                tile_regs_wait();
                for (uint32_t jj = 0; jj < nj; ++jj) {
                    pack_tile<true>(jj, cb_brow, j0 + jj);
                }
                tile_regs_release();
            }
            brow.push_back(NBPT);
            brow.wait_front(NBPT);

            // ---- L: expand each real Q chunk's bias row to a full tile (all rows equal), writer order (j, r).
            // Batches always reserve/push BIAS_BATCH pages (the CB is a multiple of it).
            if constexpr (EMIT_BIAS == 1 && DEBUG == 0) {
                for (uint32_t j = 0; j < NBPT; ++j) {
                    for (uint32_t r0 = 0; r0 < 32 && it * 32 + r0 < NQC; r0 += BIAS_BATCH) {
                        const uint32_t nr = min_u(BIAS_BATCH, min_u(32 - r0, NQC - (it * 32 + r0)));
                        bias_out.reserve_back(BIAS_BATCH);
                        tile_regs_acquire();
                        matmul_init(cb_erow, cb_brow, 0);
                        for (uint32_t rr = 0; rr < nr; ++rr) {
                            matmul_tiles(cb_erow, cb_brow, r0 + rr, j, rr);
                        }
                        tile_regs_commit();
                        tile_regs_wait();
                        for (uint32_t rr = 0; rr < nr; ++rr) {
                            pack_tile<true>(rr, cb_bias_out, rr);
                        }
                        tile_regs_release();
                        bias_out.push_back(BIAS_BATCH);
                    }
                }
            }

            exact.pop_front(TQ * NBPT);
            block01.pop_front(NBPT);
            brow.pop_front(NBPT);
            if constexpr (MODE == 0) {
                chunk2.pop_front(NKCLT);
            }
            constor.pop_front(NBPT);
            colc.pop_front(3 * NBPT);
        }

        kcb.pop_front(KC_T);
        qb.pop_front(Q_T);
        mom.pop_front(8);
        var.pop_front(DT);
        qsq.pop_front(TQ * DT);
        thrcol.pop_front(TQ);
        thr.pop_front(TQ);
    }
}
