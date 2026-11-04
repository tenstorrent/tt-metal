// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// attention-front POC, TRISC.  Per core (bank c): mm1 (wqkv columns [7c, 7c+7), blaze streaming matmul, tiny 8x32 in0)
// -> std tiles (BRISC) -> partial sums of squares of the local q / kv tiles -> (BRISC exchanges them) -> rs = rsqrt(sum
// / N + eps) of the q and kv group -> out = y * rs * gamma for the local tiles -> (BRISC distributes the q tiles) ->
// mm2 (wq_b columns [512c, 512c+512) = head c). The norm follows tt/attn_kernels/qkn_compute.cpp (bf16 intermediates,
// reduce_tile over the squares).

#include <cstdint>
#include "models/demos/deepseek_v3_b1/unified_kernels/dram_streaming_matmul.hpp"
#include "api/debug/kernel_profiler.hpp"
#include "api/compute/common.h"
#include "api/compute/reduce.h"
#include "api/compute/bcast.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/dataflow/circular_buffer.h"
#include "af_groups.hpp"

// MT = 2 row tiles (T = 9..16 rows as two 8x32 tiles): the same weight block is multiplied with both in0 tiles (dest
// slots w and SBW + w).
template <
    uint32_t CB_A,
    uint32_t CB_B,
    uint32_t CB_W,
    uint32_t CB_O,
    uint32_t SBK,
    uint32_t PN,
    uint32_t SBW,
    uint32_t NSUBK,
    bool FP32>
void mm_rows2() {
    constexpr bool transpose = false, split_acc = true, dense_packing = false;
    constexpr uint32_t ktiles = SBK * NSUBK;
    if constexpr (FP32 != DST_ACCUM_MODE) {
        custom_mm_block_init<transpose, split_acc, dense_packing, FP32>(CB_A, CB_W, CB_O);
    } else {
        reconfig_full_operand<SrcOrder::Reverse>(CB_A, CB_W);
        pack_reconfig_data_format<true>(CB_O);
        custom_mm_block_init_short<transpose, split_acc, dense_packing>(CB_A, CB_W, CB_O);
    }
    pack_block_contiguous_init(CB_O);
    cb_wait_front(CB_A, ktiles);
    cb_wait_front(CB_B, ktiles);
    for (uint32_t sb_n = 0; sb_n < PN / SBW; ++sb_n) {
        cb_reserve_back(CB_O, 2 * SBW);
        tile_regs_acquire();
        for (uint32_t w = 0; w < SBW; ++w) {
            for (uint32_t sb_k = 0; sb_k < NSUBK; ++sb_k) {
                cb_wait_front(CB_W, SBK);
                if (sb_k + 1 < NSUBK) {
                    custom_mm_block<false>(CB_A, CB_W, sb_k * SBK, 0, w, SBK);
                    custom_mm_block<false>(CB_B, CB_W, sb_k * SBK, 0, SBW + w, SBK);
                } else {
                    custom_mm_block<true>(CB_A, CB_W, sb_k * SBK, 0, w, SBK);
                    custom_mm_block<true>(CB_B, CB_W, sb_k * SBK, 0, SBW + w, SBK);
                }
                cb_pop_front(CB_W, SBK);
            }
        }
        tile_regs_commit();
        tile_regs_wait();
        pack_block_contiguous(0, CB_O, 2 * SBW);
        tile_regs_release();
        cb_push_back(CB_O, 2 * SBW);
    }
    custom_mm_block_uninit<dense_packing>();
    if constexpr (FP32 != DST_ACCUM_MODE) {
        deepseek_compute_kernel_hw_startup<DST_ACCUM_MODE>(CB_A, CB_W, CB_O);
    }
    cb_pop_front(CB_A, ktiles);
    cb_pop_front(CB_B, ktiles);
}

// grouped weight layout: groups of ct columns (af_groups.hpp), blocks of 80 tiles = (80 / ct) k rows of the group; the
// activation tile is reused over the ct columns
template <
    uint32_t MT,
    uint32_t NG,
    uint32_t KT,
    bool FP32,
    bool PROG,
    uint32_t CB_A,
    uint32_t CB_B,
    uint32_t CB_W,
    uint32_t CB_O>
void mm_groups(const uint32_t (&cts)[NG]) {
    constexpr bool transpose = false, split_acc = true, dense_packing = false;
    constexpr uint32_t BLK = 80;
    reconfig_full_operand<SrcOrder::Reverse>(CB_A, CB_W);
    pack_reconfig_data_format<true>(CB_O);
    pack_block_contiguous_init(CB_O);
    if constexpr (!PROG) {
        cb_wait_front(CB_A, KT);
        if constexpr (MT == 2) {
            cb_wait_front(CB_B, KT);
        }
    }
    uint32_t cur_ct = 0;
    for (uint32_t g = 0; g < NG; ++g) {
        const uint32_t ct = cts[g];
        const uint32_t sbk = BLK / ct;
        const uint32_t nblk = KT / sbk;
        if (ct != cur_ct) {
            custom_mm_block_init_short<transpose, split_acc, dense_packing>(CB_A, CB_W, CB_O, ct);
            cur_ct = ct;
        }
        cb_reserve_back(CB_O, ct * MT);
        tile_regs_acquire();
        for (uint32_t kb = 0; kb < nblk; ++kb) {
            cb_wait_front(CB_W, BLK);
            if constexpr (PROG) {  // the activation tiles arrive in K order
                cb_wait_front(CB_A, (kb + 1) * sbk);
                if constexpr (MT == 2) {
                    cb_wait_front(CB_B, (kb + 1) * sbk);
                }
            }
            if (kb + 1 < nblk) {
                custom_mm_block<false>(CB_A, CB_W, kb * sbk, 0, 0, sbk, ct);
                if constexpr (MT == 2) {
                    custom_mm_block<false>(CB_B, CB_W, kb * sbk, 0, ct, sbk, ct);
                }
            } else {
                custom_mm_block<true>(CB_A, CB_W, kb * sbk, 0, 0, sbk, ct);
                if constexpr (MT == 2) {
                    custom_mm_block<true>(CB_B, CB_W, kb * sbk, 0, ct, sbk, ct);
                }
            }
            cb_pop_front(CB_W, BLK);
        }
        tile_regs_commit();
        tile_regs_wait();
        pack_block_contiguous(0, CB_O, ct * MT);
        tile_regs_release();
        cb_push_back(CB_O, ct * MT);
    }
    custom_mm_block_uninit<dense_packing>();
    cb_pop_front(CB_A, KT);
    if constexpr (MT == 2) {
        cb_pop_front(CB_B, KT);
    }
}

void kernel_main() {
    constexpr uint32_t cb_x = get_compile_time_arg_val(0);
    constexpr uint32_t cb_w = get_compile_time_arg_val(1);
    constexpr uint32_t cb_y = get_compile_time_arg_val(2);
    constexpr uint32_t cb_ys = get_compile_time_arg_val(3);
    constexpr uint32_t cb_sq = get_compile_time_arg_val(4);
    constexpr uint32_t cb_pout = get_compile_time_arg_val(5);
    constexpr uint32_t cb_part = get_compile_time_arg_val(6);
    constexpr uint32_t cb_sc = get_compile_time_arg_val(7);
    constexpr uint32_t cb_eps = get_compile_time_arg_val(8);
    constexpr uint32_t cb_g = get_compile_time_arg_val(9);
    constexpr uint32_t cb_ex = get_compile_time_arg_val(10);
    constexpr uint32_t cb_rsq = get_compile_time_arg_val(11);
    constexpr uint32_t cb_rsk = get_compile_time_arg_val(12);
    constexpr uint32_t cb_xr = get_compile_time_arg_val(13);
    constexpr uint32_t cb_n = get_compile_time_arg_val(14);
    constexpr uint32_t cb_in0b = get_compile_time_arg_val(15);
    constexpr uint32_t cb_q = get_compile_time_arg_val(16);
    constexpr uint32_t cb_zero = get_compile_time_arg_val(17);
    constexpr uint32_t NC = get_compile_time_arg_val(18);
    constexpr uint32_t PN1 = get_compile_time_arg_val(19);
    constexpr uint32_t NQT = get_compile_time_arg_val(20);
    constexpr uint32_t p1_sbk = get_compile_time_arg_val(21);
    constexpr uint32_t p1_nsub = get_compile_time_arg_val(22);
    constexpr uint32_t p1_sbw = get_compile_time_arg_val(23);
    constexpr uint32_t p2_sbk = get_compile_time_arg_val(24);
    constexpr uint32_t p2_nsub = get_compile_time_arg_val(25);
    constexpr uint32_t p2_pn = get_compile_time_arg_val(26);
    constexpr uint32_t p2_sbw = get_compile_time_arg_val(27);
    constexpr uint32_t stage = get_compile_time_arg_val(28);
    constexpr uint32_t inv_nq_bits = get_compile_time_arg_val(29);
    constexpr uint32_t inv_nk_bits = get_compile_time_arg_val(30);
    constexpr uint32_t use_ds_init = get_compile_time_arg_val(31);
    constexpr uint32_t eps_bits = get_compile_time_arg_val(32);
    constexpr uint32_t fp32 = get_compile_time_arg_val(33);
    constexpr uint32_t MT = get_compile_time_arg_val(34);
    constexpr uint32_t cb_x1 = get_compile_time_arg_val(35);
    constexpr uint32_t cb_in0b1 = get_compile_time_arg_val(36);
    constexpr uint32_t x_prog = get_compile_time_arg_val(38);
    constexpr uint32_t NKT =
        get_compile_time_arg_val(37);                     // q + kv column tiles of wqkv (56); tiles beyond are padding
    constexpr uint32_t NQ_CORES = (NQT + PN1 - 1) / PN1;  // cores holding q tiles (0 .. NQ_CORES-1): 6
    constexpr uint32_t KV_FIRST = NQT / PN1;              // first core holding kv tiles: 5

    const uint32_t c = get_arg_val<uint32_t>(0);
    const uint32_t gt0 = c * PN1;
    const uint32_t nq = gt0 >= NQT ? 0 : (NQT - gt0 >= PN1 ? PN1 : NQT - gt0);
    const uint32_t rem = gt0 + nq >= NKT ? 0 : NKT - gt0 - nq;
    const uint32_t nkv = rem < PN1 - nq ? rem : PN1 - nq;
    const uint32_t ntile = nq + nkv;
    constexpr uint32_t KV_LAST = (NKT + PN1 - 1) / PN1 - 1;

    if constexpr (use_ds_init) {
        deepseek_compute_kernel_init();
    } else {
        compute_kernel_hw_startup(0, 0, 0);
    }

    {
        DeviceZoneScopedN("T_MM1");
        mm_groups<MT, af::NG1, 160, (fp32 != 0), (x_prog != 0), cb_x, cb_x1, cb_w, cb_y>(af::GRP1);
    }
    if constexpr (stage == 1) {
        return;
    }

    CircularBuffer ys(cb_ys), sq(cb_sq), pout(cb_pout), part(cb_part), sc(cb_sc), eps(cb_eps), g(cb_g), ex(cb_ex),
        rsq(cb_rsq), rsk(cb_rsk), xr(cb_xr), nn(cb_n), zero(cb_zero);

    // the blaze matmul leaves the unpacker / packer configured for the tiny 8x32 tiles (hw startup of cb 0):
    // reconfigure for std tiles
    {
        DeviceZoneScopedN("T_HWS");
        deepseek_compute_kernel_hw_startup<(fp32 != 0)>(cb_ys, cb_ys, cb_sq);
    }

    sc.wait_front(1);
    eps.wait_front(1);
    zero.wait_front(1);
    ys.wait_front(PN1);

    {
        DeviceZoneScopedN("T_SQ");
        reconfig_data_format(cb_ys, cb_ys);
        pack_reconfig_data_format(cb_sq);
        mul_init(cb_ys, cb_ys);
        for (uint32_t i = 0; i < PN1; ++i) {
            tile_regs_acquire();
            mul_tiles(cb_ys, cb_ys, i, i, 0);
            tile_regs_commit();
            tile_regs_wait();
            sq.reserve_back(1);
            pack_tile(0, cb_sq);
            sq.push_back(1);
            tile_regs_release();
        }
        sq.wait_front(PN1);
    }

    {
        DeviceZoneScopedN("T_RED");
        reconfig_data_format(cb_sq, cb_sc);
        for (uint32_t grp = 0; grp < 2; ++grp) {
            const uint32_t t0 = grp == 0 ? 0 : nq;
            const uint32_t cnt = grp == 0 ? nq : nkv;
            if (cnt == 0) {
                continue;
            }
            tile_regs_acquire();
            reduce_init<PoolType::SUM, ReduceDim::REDUCE_ROW>(cb_sq, cb_sc, cb_pout);
            for (uint32_t i = 0; i < cnt; ++i) {
                reduce_tile<PoolType::SUM, ReduceDim::REDUCE_ROW>(cb_sq, cb_sc, t0 + i, 0, 0);
            }
            reduce_uninit();
            tile_regs_commit();
            tile_regs_wait();
            pout.reserve_back(1);
            pack_reconfig_data_format(cb_pout);
            pack_tile(0, cb_pout);
            pout.push_back(1);
            tile_regs_release();
        }
        sq.pop_front(PN1);
    }

    {
        DeviceZoneScopedN("T_COMB");
        part.wait_front(2 * NC);
        for (uint32_t grp = 0; grp < 2; ++grp) {
            if ((grp == 0 && nq == 0) || (grp == 1 && nkv == 0)) {
                continue;
            }
            const uint32_t base = grp == 0 ? 0 : NC;
            const uint32_t first = grp == 0 ? 0 : KV_FIRST;
            const uint32_t last = grp == 0 ? NQ_CORES - 1 : KV_LAST;  // inclusive
            // sum of slots first..last in pairs with accumulation in dest; an odd count pairs the last slot with the
            // zero tile
            reconfig_data_format(cb_part, cb_part);
            tile_regs_acquire();
            bool started = false;
            for (uint32_t s = first; s <= last; s += 2) {
                if (s + 1 <= last) {
                    add_init(cb_part, cb_part, started);
                    add_tiles(cb_part, cb_part, base + s, base + s + 1, 0);
                } else {
                    reconfig_data_format(cb_part, cb_zero);
                    add_init(cb_part, cb_zero, started);
                    add_tiles(cb_part, cb_zero, base + s, 0, 0);
                }
                started = true;
            }
            binop_with_scalar_tile_init();
            mul_unary_tile(0, grp == 0 ? inv_nq_bits : inv_nk_bits);
            add_unary_tile(0, eps_bits);
            rsqrt_tile_init();
            rsqrt_tile(0);
            tile_regs_commit();
            tile_regs_wait();
            if (grp == 0) {
                rsq.reserve_back(1);
                pack_reconfig_data_format(cb_rsq);
                pack_tile(0, cb_rsq);
                rsq.push_back(1);
            } else {
                rsk.reserve_back(1);
                pack_reconfig_data_format(cb_rsk);
                pack_tile(0, cb_rsk);
                rsk.push_back(1);
            }
            tile_regs_release();
        }
        part.pop_front(2 * NC);
    }
    if (nq > 0) {
        rsq.wait_front(1);
    }
    if (nkv > 0) {
        rsk.wait_front(1);
    }

    {
        DeviceZoneScopedN("T_APPLY");
        g.wait_front(PN1);
        for (uint32_t i = 0; i < ntile; ++i) {
            const uint32_t cb_rs = i < nq ? cb_rsq : cb_rsk;
            reconfig_data_format(cb_ys, cb_rs);
            pack_reconfig_data_format(cb_xr);
            mul_bcast_cols_init(cb_ys, cb_rs);
            tile_regs_acquire();
            mul_tiles_bcast_cols(cb_ys, cb_rs, i, 0, 0);
            tile_regs_commit();
            tile_regs_wait();
            xr.reserve_back(1);
            pack_tile(0, cb_xr);
            xr.push_back(1);
            tile_regs_release();
            xr.wait_front(1);
            reconfig_data_format(cb_xr, cb_g);
            pack_reconfig_data_format(cb_n);
            mul_bcast_rows_init(cb_xr, cb_g);
            tile_regs_acquire();
            mul_tiles_bcast_rows(cb_xr, cb_g, 0, i, 0);
            tile_regs_commit();
            tile_regs_wait();
            nn.reserve_back(1);
            pack_tile(0, cb_n);
            nn.push_back(1);
            tile_regs_release();
            xr.pop_front(1);
        }
        ys.pop_front(PN1);
        g.pop_front(PN1);
    }
    if constexpr (stage == 2) {
        return;
    }

    // ---- mm2 on the exchanged qr tiles ----
    deepseek_compute_kernel_hw_startup<(fp32 != 0)>(cb_in0b, cb_w, cb_q);
    {
        DeviceZoneScopedN("T_MM2");
        mm_groups<MT, af::NG2, 40, (fp32 != 0), false, cb_in0b, cb_in0b1, cb_w, cb_q>(af::GRP2);
    }
}
