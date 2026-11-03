// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Sol-Attn pre-pass, route step, compute kernel.
//
// Decides which pooled K blocks each Q chunk keeps. Per unit of work:
//   1. Mean and variance of the pooled K blocks (phases A..D).
//   2. A threshold per Q block from those moments and tau (E..G).
//   3. Score every (Q block, K block) pair and compare with the threshold (H).
//   4. OR the scores over each Q chunk, OR in the constant mask, drop padding blocks (I).
//   5. Chunk mode only: OR over each K chunk to get the final mask (J).
//
// A unit is (head, Q-chunk tile row) when UNITS_PER_SHARD = 0, or (head, K shard) when UNITS_PER_SHARD = 1 (gather
// mode only). With (head, shard) units a core only has its own shard's K, so the moments are built from partial sums
// that the reader exchanges between the head's cores (phases CL and C), and the mask leaves as row-major uint32 rows
// (phase I') instead of tiles.
//
// Compile-time args: 0 H, 1 S, 2 NBPT, 3 DT, 4 NQCT, 5 NKCLT, 6 NQC, 7 NKC, 8 NKCP, 9 MODE (0 chunk, 1 gather,
// 2 compact), 10 TAU, 11 EPS2, 12 NEG, 13 INV_NV (fp32 bits), 14 TQ, 15 NX, 16 NY, 17 UNITS, 18 UNITS_PER_SHARD.

#include <cstdint>

#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/comp.h"
#include "api/compute/eltwise_unary/relu.h"
#include "api/compute/eltwise_unary/sqrt.h"
#include "api/compute/eltwise_unary/typecast.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/pack_untilize.h"
#include "api/compute/reg_api.h"
#include "api/compute/tile_move_copy.h"
#include "api/dataflow/circular_buffer.h"

namespace {
constexpr uint32_t cb_kc = 0, cb_q = 1, cb_ones = 2, cb_qsel = 3, cb_ksel = 4, cb_constor = 6, cb_colc = 7, cb_part = 8,
                   cb_mom = 9, cb_qsq = 10, cb_thrcol = 11, cb_thr = 12, cb_exact = 13, cb_block01 = 14, cb_mask_out = 16,
                   cb_var = 20, cb_union_out = 21, cb_recv = 23, cb_pl = 24, cb_shsel = 29;
constexpr uint32_t PART_T = 8;              // DT sum tiles + DT sum-of-squares tiles
constexpr uint32_t HALF_NEG = 0xBF000000u;  // -0.5f
// Unary SFPU ops only work on DST tiles 0..3, so they run on slabs of SLAB tiles with tile SLAB as scratch.
constexpr uint32_t SLAB = 3;  // DST tiles 0..2: a column slab, 3: scratch
constexpr uint32_t QUAD = 4;  // tiles per DST batch
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
    constexpr uint32_t TAU = get_compile_time_arg_val(10);
    constexpr uint32_t EPS2 = get_compile_time_arg_val(11);
    constexpr uint32_t INV_NV = get_compile_time_arg_val(13);
    constexpr uint32_t TQ = get_compile_time_arg_val(14);
    constexpr uint32_t NBGT = S * NBPT;
    constexpr uint32_t UNITS_PER_SHARD_ = get_compile_time_arg_val(18);
    constexpr uint32_t KCT = UNITS_PER_SHARD_ ? NBPT : NBGT;  // K column tiles per unit
    constexpr uint32_t KC_T = KCT * DT;
    constexpr uint32_t Q_T = NBPT * DT;
    constexpr uint32_t MASK_T = MODE == 0 ? NKCLT : NBPT;  // per shard
    // DST tiles 0..7 are used: the program needs dst_full_sync_en.
    static_assert(DT <= 4, "moments use DST 0..3 (sums) and 4..7 (sums of squares)");
    static_assert(NKCLT <= 8, "chunk tiles are processed in DST batches of QUAD");
    static_assert(TQ <= 8, "Q-block tile rows per Q-chunk tile row must fit one DST batch");

    constexpr uint32_t NX = get_compile_time_arg_val(15);
    constexpr uint32_t NY = get_compile_time_arg_val(16);
    constexpr uint32_t UNITS = get_compile_time_arg_val(17);
    constexpr uint32_t UNITS_PER_SHARD = get_compile_time_arg_val(18);  // 1: unit = (head, shard); 0: (head, it)
    static_assert(UNITS_PER_SHARD == 0 || MODE == 1, "(head, shard) units emit only gather-form rows (MODE 1)");
    static_assert(UNITS == (UNITS_PER_SHARD ? get_compile_time_arg_val(0) * S : get_compile_time_arg_val(0) * NQCT));
    // This core's units, [ci*U/N, (ci+1)*U/N), from its logical coordinates.
    constexpr uint32_t NCORES = NX * NY;
    const uint32_t ci = static_cast<uint32_t>(get_absolute_logical_y()) * NX + get_absolute_logical_x();
    const uint32_t unit_start = ci * UNITS / NCORES;
    const uint32_t unit_count = (ci + 1) * UNITS / NCORES - unit_start;
    // The output CB sets the packer format for the whole kernel: it must be a bf16 tile CB.
    compute_kernel_hw_startup(cb_q, cb_kc, cb_part);
    if (unit_count == 0) {
        return;
    }
    CircularBuffer kcb(cb_kc), qb(cb_q), ones(cb_ones), qsel(cb_qsel), ksel(cb_ksel), constor(cb_constor),
        colc(cb_colc), part(cb_part), mom(cb_mom), qsq(cb_qsq), thrcol(cb_thrcol), thr(cb_thr), exact(cb_exact),
        block01(cb_block01), mask_out(cb_mask_out), var(cb_var), union_out(cb_union_out), recv(cb_recv), pl(cb_pl),
        shsel(cb_shsel);
    constexpr uint32_t GT = (S + 3) / 4;  // gather tiles: 4 shards x 8 rows per tile

    for (uint32_t u = unit_start; u < unit_start + unit_count; ++u) {
        // One of the two ranges is a single value; the loops are the same either way.
        const uint32_t it_lo = UNITS_PER_SHARD ? 0 : u % NQCT;
        const uint32_t it_hi = UNITS_PER_SHARD ? NQCT : it_lo + 1;
        const uint32_t sp_lo = UNITS_PER_SHARD ? u % S : 0;
        const uint32_t sp_hi = UNITS_PER_SHARD ? sp_lo + 1 : S;
        // ---- A/B: per-tile sums and sums of squares of the pooled K.
        part.reserve_back(8);
        tile_regs_acquire();
        copy_init(cb_kc);
        add_binary_tile_init();
        for (uint32_t c = 0; c < KCT; ++c) {
            kcb.wait_front((c + 1) * DT);
            for (uint32_t n = 0; n < DT; ++n) {
                if (c == 0) {
                    copy_tile(cb_kc, n, n);
                } else {
                    copy_tile(cb_kc, c * DT + n, 4);
                    add_binary_tile(n, 4, n);
                }
            }
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t n = 0; n < DT; ++n) {
            pack_tile<true>(n, cb_part, n);
        }
        tile_regs_release();

        // Squares: copy + SFPU multiply.
        tile_regs_acquire();
        copy_init(cb_kc);
        for (uint32_t n = 0; n < DT; ++n) {
            copy_tile(cb_kc, n, n);
        }
        mul_binary_tile_init();
        for (uint32_t n = 0; n < DT; ++n) {
            mul_binary_tile(n, n, n);
        }
        for (uint32_t c = 1; c < KCT; ++c) {
            for (uint32_t n = 0; n < DT; ++n) {
                copy_tile(cb_kc, c * DT + n, 4);
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

        // Per-core constants, read once by the reader.
        ones.wait_front(1);
        qsel.wait_front(NQCT * NBPT);
        ksel.wait_front(NBPT * NKCLT);
        if constexpr (UNITS_PER_SHARD) {
            shsel.wait_front(8);
        }

        // ---- CL: (head, shard) units: column sums of this shard's partial tiles, for the reader to exchange.
        if constexpr (UNITS_PER_SHARD) {
            pl.reserve_back(PART_T);
            for (uint32_t b0 = 0; b0 < PART_T; b0 += QUAD) {
                tile_regs_acquire();
                matmul_init(cb_ones, cb_part, 0);
                for (uint32_t i = 0; i < QUAD; ++i) {
                    matmul_tiles(cb_ones, cb_part, 0, b0 + i, i);
                }
                tile_regs_commit();
                tile_regs_wait();
                for (uint32_t i = 0; i < QUAD; ++i) {
                    pack_tile<true>(i, cb_pl, b0 + i);
                }
                tile_regs_release();
            }
            pl.push_back(PART_T);
            part.pop_front(8);
            recv.wait_front(GT);  // every shard's column sums
        }

        // ---- C: column sums of the partial tiles, scaled by 1/n_valid -> mean and E2.
        mom.reserve_back(8);
        for (uint32_t b0 = 0; b0 < 8; b0 += QUAD) {
            tile_regs_acquire();
            if constexpr (UNITS_PER_SHARD) {  // sum over the gather tiles with the shard selectors
                matmul_init(cb_shsel, cb_recv, 0);
                for (uint32_t i = 0; i < QUAD; ++i) {
                    for (uint32_t t = 0; t < GT; ++t) {
                        matmul_tiles(cb_shsel, cb_recv, b0 + i, t, i);
                    }
                }
            } else {
                matmul_init(cb_ones, cb_part, 0);
                for (uint32_t i = 0; i < QUAD; ++i) {
                    matmul_tiles(cb_ones, cb_part, 0, b0 + i, i);
                }
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
        if constexpr (UNITS_PER_SHARD) {
            recv.pop_front(GT);
        } else {
            part.pop_front(8);
        }

        // ---- D: var = relu(E2 - mean^2).
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
        qb.wait_front(Q_T);  // Q is first needed in E

        for (uint32_t it = it_lo; it < it_hi; ++it) {
            const uint32_t t0 = min_u(it * TQ, NBPT);
            const uint32_t t1 = min_u(t0 + TQ, NBPT);
            const uint32_t nt = t1 - t0;  // Q block tile rows here (0 if all padding)

            // ---- E: q_bar squared.
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

            // ---- F: threshold column per Q block: q.mean + tau * sqrt(q^2.var + eps).
            thrcol.reserve_back(TQ);
            for (uint32_t tt0 = 0; tt0 < nt; tt0 += 2) {
                const uint32_t ntt = min_u(2, nt - tt0);
                tile_regs_acquire();
                matmul_init(cb_q, cb_mom, 1);
                for (uint32_t k = 0; k < ntt; ++k) {
                    for (uint32_t n = 0; n < DT; ++n) {
                        matmul_tiles(cb_q, cb_mom, (t0 + tt0 + k) * DT + n, n, 2 * k);
                    }
                }
                matmul_init(cb_qsq, cb_var, 1);
                for (uint32_t k = 0; k < ntt; ++k) {
                    for (uint32_t n = 0; n < DT; ++n) {
                        matmul_tiles(cb_qsq, cb_var, (tt0 + k) * DT + n, n, 2 * k + 1);
                    }
                }
                binop_with_scalar_tile_init();
                for (uint32_t k = 0; k < ntt; ++k) {
                    add_unary_tile(2 * k + 1, EPS2);
                }
                sqrt_tile_init();
                for (uint32_t k = 0; k < ntt; ++k) {
                    sqrt_tile(2 * k + 1);
                }
                binop_with_scalar_tile_init();
                for (uint32_t k = 0; k < ntt; ++k) {
                    mul_unary_tile(2 * k + 1, TAU);
                }
                add_binary_tile_init();
                for (uint32_t k = 0; k < ntt; ++k) {
                    add_binary_tile(2 * k, 2 * k + 1, 2 * k);
                }
                tile_regs_commit();
                tile_regs_wait();
                for (uint32_t k = 0; k < ntt; ++k) {
                    pack_tile<true>(2 * k, cb_thrcol, tt0 + k);
                }
                tile_regs_release();
            }
            thrcol.push_back(TQ);
            thrcol.wait_front(TQ);

            // ---- G: broadcast the threshold column across all 32 columns.
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

            for (uint32_t sp = sp_lo; sp < sp_hi; ++sp) {
                constor.wait_front(NBPT);
                colc.wait_front(NBPT);

                // ---- H: scores against shard sp's K blocks, 1 where score > threshold.
                exact.reserve_back(TQ * NBPT);
                for (uint32_t tt = 0; tt < nt; ++tt) {
                    for (uint32_t c0 = 0; c0 < NBPT; c0 += SLAB) {
                        const uint32_t nc = min_u(SLAB, NBPT - c0);
                        tile_regs_acquire();
                        matmul_init(cb_q, cb_kc, 1);
                        for (uint32_t cc = 0; cc < nc; ++cc) {
                            for (uint32_t n = 0; n < DT; ++n) {
                                const uint32_t kci = ((UNITS_PER_SHARD ? 0 : sp * NBPT) + c0 + cc) * DT + n;
                                matmul_tiles(cb_q, cb_kc, (t0 + tt) * DT + n, kci, cc);
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

                // ---- I: OR over the Q blocks of each chunk, OR the constant mask, AND the valid-block row.
                block01.reserve_back(NBPT);
                if constexpr (MODE != 0 && !UNITS_PER_SHARD) {
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
                        copy_tile(cb_colc, c, SLAB);  // valid-block row
                        mul_binary_tile_init();
                        mul_binary_tile(cc, SLAB, cc);
                    }
                    tile_regs_commit();
                    tile_regs_wait();
                    for (uint32_t cc = 0; cc < nc; ++cc) {
                        pack_tile<true>(cc, cb_block01, c0 + cc);
                        if constexpr (MODE != 0 && !UNITS_PER_SHARD) {
                            pack_tile<true>(cc, cb_mask_out, c0 + cc);
                        }
                    }
                    tile_regs_release();
                }
                block01.push_back(NBPT);
                block01.wait_front(NBPT);
                if constexpr (MODE != 0 && !UNITS_PER_SHARD) {
                    mask_out.push_back(MASK_T);
                }

                // ---- I': (head, shard) units: typecast the block mask to uint32 and untilize it into one row-major
                // page.
                if constexpr (UNITS_PER_SHARD) {
                    constexpr uint32_t UT = (NBPT <= SLAB) ? NBPT : 1;
                    static_assert(NBPT % UT == 0, "untilize block width must divide the row width");
                    mask_out.reserve_back(1);
                    pack_untilize_dest_init<UT, NBPT>(cb_mask_out);
                    for (uint32_t c0 = 0; c0 < NBPT; c0 += UT) {
                        tile_regs_acquire();
                        copy_init(cb_block01);
                        for (uint32_t cc = 0; cc < UT; ++cc) {
                            copy_tile(cb_block01, c0 + cc, cc);
                        }
                        typecast_tile_init<static_cast<uint32_t>(DataFormat::Float32), static_cast<uint32_t>(DataFormat::UInt32)>();
                        for (uint32_t cc = 0; cc < UT; ++cc) {
                            typecast_tile<static_cast<uint32_t>(DataFormat::Float32), static_cast<uint32_t>(DataFormat::UInt32)>(cc);
                        }
                        tile_regs_commit();
                        tile_regs_wait();
                        pack_untilize_dest<UT, NBPT>(cb_mask_out, 1, c0 / UT);
                        tile_regs_release();
                    }
                    pack_untilize_uninit(cb_block01);  // back to tilized bf16 packing
                    mask_out.push_back(1);
                }

                // ---- J: OR over each K chunk (MODE 0: the mask, MODE 2: the union row).
                if constexpr (MODE != 1) {
                    if constexpr (MODE == 0) {
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
                                pack_tile<true>(cc, cb_mask_out, ct0 + cc);
                            } else {
                                pack_tile<true>(cc, cb_union_out, ct0 + cc);
                            }
                        }
                        tile_regs_release();
                    }
                    if constexpr (MODE == 0) {
                        mask_out.push_back(MASK_T);
                    } else {
                        union_out.push_back(NKCLT);
                    }
                }

                exact.pop_front(TQ * NBPT);
                block01.pop_front(NBPT);
                constor.pop_front(NBPT);
                colc.pop_front(NBPT);
            }

            qsq.pop_front(TQ * DT);
            thrcol.pop_front(TQ);
            thr.pop_front(TQ);
        }

        kcb.pop_front(KC_T);
        qb.pop_front(Q_T);
        mom.pop_front(8);
        var.pop_front(DT);
    }
}
