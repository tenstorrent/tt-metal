// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// Batched decode down + expert sum (Laguna): DST j * Mt + m accumulates x_{e,m} @ Wd_e[:, c0 + j] over every active
// expert e (routing weights are already in x_e; m = 32-row token tile), then packs the CPC * Mt output tiles.

#include <cstdint>
#include "api/compute/reg_api.h"
#include "api/compute/matmul.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/cb_api.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/pack.h"
#include "api/compute/tile_move_copy.h"

void kernel_main() {
    constexpr uint32_t Kt = get_compile_time_arg_val(0);
    constexpr uint32_t CPC = get_compile_time_arg_val(1);
    constexpr uint32_t Kt_sh = get_compile_time_arg_val(2);  // shared expert unit (on one group's cores), cb_w_sh
    constexpr uint32_t KS = get_compile_time_arg_val(3);     // units per routed expert (Kt / KS K tiles each)
    constexpr uint32_t Kq = Kt / KS;
    constexpr uint32_t Mt = get_compile_time_arg_val(4);
    // Mt > 1: one matmul_block per weight tile (rt_dim = Mt, ct_dim = 1): the weight tile unpacked once for every
    // token tile
    constexpr bool block = get_compile_time_arg_val(5) == 1 && Mt > 1;
    constexpr uint32_t cb_x = 0, cb_w = 1, cb_meta = 2, cb_w_sh = 6, cb_out = 16;
    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_x, cb_w, cb_out);
    cb_wait_front(cb_meta, 1);
    const uint32_t na = read_tile_value(cb_meta, 0, 0);
    const uint32_t sh = Kt_sh > 0 ? read_tile_value(cb_meta, 0, 1) : 0;
    cb_pop_front(cb_meta, 1);
    if (na == 0 && sh == 0) {
        return;
    }
    if constexpr (Mt > 1) {
        // more than 32 tokens: per unit and token tile m, DST j = x_m @ W[:, c0 + j] (CPC tiles), packed with L1
        // accumulation into the fp32 cb_acc tile j * Mt + m (the first unit writes, later units add): CPC stays 8
        // at any Mt (DST holds only CPC fp32 tiles), so fewer column groups re-read the activation rows
        constexpr uint32_t cb_acc = 7;
        const uint32_t units = na + sh;
        cb_reserve_back(cb_acc, CPC * Mt);
        pack_reconfig_data_format(cb_acc);
        for (uint32_t a = 0; a < units; ++a) {
            const bool shared_unit = Kt_sh > 0 && a == na;  // the shared expert comes after the routed units
            const uint32_t kq = shared_unit ? Kt_sh : Kq;
            const uint32_t cbw = shared_unit ? cb_w_sh : cb_w;
            if (shared_unit) {
                reconfig_data_format(cb_w_sh, cb_x);
            }
            if constexpr (block) {
                matmul_block_init(cb_x, cbw, false, 1, Mt, kq);
            } else {
                matmul_init(cb_x, cbw);
            }
            pack_reconfig_data_format(cb_acc);  // the matmul init configured the packer for cb_out's format
            cb_wait_front(cb_x, Mt * kq);
            cb_wait_front(cbw, CPC * kq);
            if constexpr (block) {
                // nj columns x Mt token tiles per acquire (4 fp32 DST tiles): DST jj * Mt + m
                constexpr uint32_t nj = Mt >= 4 ? 1 : 4 / Mt;
                for (uint32_t j0 = 0; j0 < CPC; j0 += nj) {
                    const uint32_t n = CPC - j0 < nj ? CPC - j0 : nj;
                    tile_regs_acquire();
                    for (uint32_t jj = 0; jj < n; ++jj) {
                        for (uint32_t k = 0; k < kq; ++k) {
                            matmul_block(cb_x, cbw, k, (j0 + jj) * kq + k, jj * Mt, false, 1, Mt, kq);
                        }
                    }
                    tile_regs_commit();
                    tile_regs_wait();
                    PACK((llk_pack_reconfig_l1_acc(a > 0 ? 1 : 0)));
                    for (uint32_t jj = 0; jj < n; ++jj) {
                        for (uint32_t m = 0; m < Mt; ++m) {
                            pack_tile<true>(jj * Mt + m, cb_acc, (j0 + jj) * Mt + m);
                        }
                    }
                    tile_regs_release();
                }
                cb_pop_front(cb_x, Mt * kq);
                cb_pop_front(cbw, CPC * kq);
                continue;
            }
            for (uint32_t m = 0; m < Mt; ++m) {
                // half-sync DST with fp32 holds 4 tiles per acquire: the CPC columns in groups of 4
                for (uint32_t j0 = 0; j0 < CPC; j0 += 4) {
                    const uint32_t nj = CPC - j0 < 4 ? CPC - j0 : 4;
                    tile_regs_acquire();
                    for (uint32_t j = 0; j < nj; ++j) {
                        for (uint32_t k = 0; k < kq; ++k) {
                            matmul_tiles(cb_x, cbw, m * kq + k, (j0 + j) * kq + k, j);
                        }
                    }
                    tile_regs_commit();
                    tile_regs_wait();
                    PACK((llk_pack_reconfig_l1_acc(a > 0 ? 1 : 0)));
                    for (uint32_t j = 0; j < nj; ++j) {
                        pack_tile<true>(j, cb_acc, (j0 + j) * Mt + m);
                    }
                    tile_regs_release();
                }
            }
            cb_pop_front(cb_x, Mt * kq);
            cb_pop_front(cbw, CPC * kq);
        }
        PACK((llk_pack_reconfig_l1_acc(0)));
        cb_push_back(cb_acc, CPC * Mt);
        // fp32 sums -> the bf16 output tiles (order j * Mt + m)
        cb_wait_front(cb_acc, CPC * Mt);
        copy_init(cb_acc);
        reconfig_data_format_srca(cb_acc);
        pack_reconfig_data_format(cb_out);
        cb_reserve_back(cb_out, CPC * Mt);
        for (uint32_t t0 = 0; t0 < CPC * Mt; t0 += 4) {
            const uint32_t n = CPC * Mt - t0 < 4 ? CPC * Mt - t0 : 4;
            tile_regs_acquire();
            for (uint32_t i = 0; i < n; ++i) {
                copy_tile(cb_acc, t0 + i, i);
            }
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t i = 0; i < n; ++i) {
                pack_tile(i, cb_out);
            }
            tile_regs_release();
        }
        cb_push_back(cb_out, CPC * Mt);
        cb_pop_front(cb_acc, CPC * Mt);
        return;
    }
    matmul_init(cb_x, cb_w);
    tile_regs_acquire();
    for (uint32_t a = 0; a < na; ++a) {
        cb_wait_front(cb_x, Mt * Kq);
        cb_wait_front(cb_w, CPC * Kq);
        for (uint32_t j = 0; j < CPC; ++j) {
            for (uint32_t k = 0; k < Kq; ++k) {
                for (uint32_t m = 0; m < Mt; ++m) {
                    matmul_tiles(cb_x, cb_w, m * Kq + k, j * Kq + k, j * Mt + m);
                }
            }
        }
        cb_pop_front(cb_x, Mt * Kq);
        cb_pop_front(cb_w, CPC * Kq);
    }
    if constexpr (Kt_sh > 0) {
        if (sh) {
            reconfig_data_format(cb_w_sh, cb_x);
            matmul_init(cb_x, cb_w_sh);
            cb_wait_front(cb_x, Mt * Kt_sh);
            cb_wait_front(cb_w_sh, CPC * Kt_sh);
            for (uint32_t j = 0; j < CPC; ++j) {
                for (uint32_t k = 0; k < Kt_sh; ++k) {
                    for (uint32_t m = 0; m < Mt; ++m) {
                        matmul_tiles(cb_x, cb_w_sh, m * Kt_sh + k, j * Kt_sh + k, j * Mt + m);
                    }
                }
            }
            cb_pop_front(cb_x, Mt * Kt_sh);
            cb_pop_front(cb_w_sh, CPC * Kt_sh);
        }
    }
    tile_regs_commit();
    tile_regs_wait();
    cb_reserve_back(cb_out, CPC * Mt);
    for (uint32_t j = 0; j < CPC * Mt; ++j) {
        pack_tile(j, cb_out);
    }
    cb_push_back(cb_out, CPC * Mt);
    tile_regs_release();
}
