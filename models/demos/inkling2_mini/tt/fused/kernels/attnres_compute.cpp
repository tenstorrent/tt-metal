// SPDX-FileCopyrightText: © 2026 Olof Johansson <olof@lixom.net>
// SPDX-License-Identifier: Apache-2.0
//
// Fused block attention residual, compute. Dense vectors = T pages viewed as 32x32 tiles; every scalar sits at
// [0, 0] of its own tile (reduce SCALAR output), which the bcast-scalar ops read.
//   inv  = rsqrt(sum(p * p) / C + eps)
//   n    = p * inv                                   -> stack row k (writer) and state k below
//   l_ij = sum(q_i * s_j), s_j = stack row j (j < k) or n (j = k)     (q pre-scaled by 1/sqrt(C))
//   w_ij = softmax_j(l_ij)                           (SFPU, fp32)
//   out_i = sum_j w_ij * s_j
// FPU ops (mul, bcast, reduce) read fp32 operands as tf32, like the matmul / rms_norm path they replace.
#include <cstdint>

#include "api/compute/bcast.h"
#include "api/compute/binary_max_min.h"
#include "api/compute/common.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/exp.h"
#include "api/compute/eltwise_unary/recip.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/reduce.h"
#include "api/compute/tile_move_copy.h"
#include "api/debug/kernel_profiler.hpp"
#include "scalar_sfpu.hpp"

namespace {
// scalar SFPU ops (scalar_sfpu.hpp): SIT iterations of face 0 (8 = the whole face, as before; 1 = the rows of [0, 0])
constexpr int SIT = get_named_compile_time_arg_val("sit");
constexpr uint32_t cb_p = 0, cb_q = 1, cb_s = 2, cb_scaler = 3, cb_wt = 4, cb_np = 16, cb_out = 17;
constexpr uint32_t cb_sq = 25, cb_inv = 26, cb_prod = 27, cb_w = 28;

FORCE_INLINE void put(uint32_t dst, uint32_t cb) {
    cb_reserve_back(cb, 1);
    pack_tile(dst, cb);
    cb_push_back(cb, 1);
}

// dst[0] = sum of dst[0 .. n-1] (SFPU, fp32)
FORCE_INLINE void sum_dst(uint32_t n) {
    if (n > 1) {
        add_binary_tile_init();
        for (uint32_t i = 1; i < n; ++i) {
            add_binary_tile(0, i, 0);
        }
    }
}

// dst[0..n-1] = reduce_scalar(cb[first + i]); pack them to ocb.
FORCE_INLINE void reduce_scalars(uint32_t cb, uint32_t first, uint32_t n, uint32_t ocb) {
    reconfig_data_format(cb, cb_scaler);
    reduce_init<PoolType::SUM, ReduceDim::REDUCE_SCALAR>(cb, cb_scaler, ocb);
    for (uint32_t i = 0; i < n; ++i) {
        reduce_tile<PoolType::SUM, ReduceDim::REDUCE_SCALAR>(cb, cb_scaler, first + i, 0, i);
    }
    reduce_uninit();
}
}  // namespace

void kernel_main() {
    constexpr uint32_t T = get_named_compile_time_arg_val("pages");
    constexpr uint32_t nq = get_named_compile_time_arg_val("nq");
    constexpr uint32_t k = get_named_compile_time_arg_val("k");
    constexpr uint32_t inv_c = get_named_compile_time_arg_val("inv_c");  // fp32 bits of 1/C
    constexpr uint32_t eps = get_named_compile_time_arg_val("eps");      // fp32 bits
    constexpr uint32_t has_weight = get_named_compile_time_arg_val("has_weight");
    constexpr uint32_t has_norm = get_named_compile_time_arg_val("has_norm");  // + weighted rms of out_0 (MoE input)
    constexpr uint32_t d = k + 1;

    compute_kernel_hw_startup(cb_p, cb_p, cb_sq);
#ifdef ATTNRES_EARLY_PROD
    // one query, sealed rows: their products need no partial, so they run while it is still on its way (pushed to
    // cb_prod in the same order as below: rows 0 .. k-1, then the partial's)
    static_assert(nq == 1 && k > 0);
    cb_wait_front(cb_q, T);
    cb_wait_front(cb_s, k * T);
    {
        DeviceZoneScopedN("C_prod0");
        for (uint32_t j = 0; j < k; ++j) {
            tile_regs_acquire();
            reconfig_data_format(cb_q, cb_s);
            mul_init(cb_q, cb_s);
            for (uint32_t t = 0; t < T; ++t) {
                mul_tiles(cb_q, cb_s, t, j * T + t, t);
            }
            sum_dst(T);
            tile_regs_commit();
            tile_regs_wait();
            pack_reconfig_data_format(cb_prod);
            put(0, cb_prod);
            tile_regs_release();
        }
    }
    reconfig_data_format(cb_p, cb_p);
#endif
    cb_wait_front(cb_scaler, 1);
    cb_wait_front(cb_p, T);

    {  // profiler zone
        DeviceZoneScopedN("C_inv");
        // ---- inv = rsqrt(mean(p^2) + eps)
        tile_regs_acquire();
        reconfig_data_format(cb_p, cb_p);
        mul_init(cb_p, cb_p);
        for (uint32_t t = 0; t < T; ++t) {
            mul_tiles(cb_p, cb_p, t, t, t);
        }
        sum_dst(T);
        tile_regs_commit();
        tile_regs_wait();
        pack_reconfig_data_format(cb_sq);
        put(0, cb_sq);
        tile_regs_release();

        cb_wait_front(cb_sq, 1);
        tile_regs_acquire();
        reduce_scalars(cb_sq, 0, 1, cb_inv);
        binop_with_scalar_tile_init();
        scal::mul_unary<SIT>(0, inv_c);  // scalars sit at [0, 0]: SFPU on face 0 only
        scal::add_unary<SIT>(0, eps);
        rsqrt_tile_init();
        scal::rsqrt<SIT>(0);
        tile_regs_commit();
        tile_regs_wait();
        put(0, cb_inv);
        tile_regs_release();
        cb_pop_front(cb_sq, 1);
    }
    {
        DeviceZoneScopedN("C_n");
        // ---- n = p * inv
        cb_wait_front(cb_inv, 1);
        tile_regs_acquire();
        reconfig_data_format(cb_p, cb_inv);
        mul_bcast_scalar_init(cb_p, cb_inv);
        for (uint32_t t = 0; t < T; ++t) {
            mul_tiles_bcast_scalar(cb_p, cb_inv, t, 0, t);
        }
        if constexpr (has_weight) {  // n *= weight (SFPU, fp32; weight copied unrounded)
            cb_wait_front(cb_wt, T);
            reconfig_data_format_srca(cb_wt);
            copy_init(cb_wt);
            for (uint32_t t = 0; t < T; ++t) {
                copy_tile(cb_wt, t, T + t);
            }
            mul_binary_tile_init();
            for (uint32_t t = 0; t < T; ++t) {
                mul_binary_tile(t, T + t, t);
            }
        }
        tile_regs_commit();
        tile_regs_wait();
        cb_reserve_back(cb_np, T);
        pack_reconfig_data_format(cb_np);  // bf16 for the gate prep (rounds like HF's cast to down_proj dtype)
        for (uint32_t t = 0; t < T; ++t) {
            pack_tile(t, cb_np);
        }
        cb_push_back(cb_np, T);
        tile_regs_release();
        pack_reconfig_data_format(cb_prod);
        cb_pop_front(cb_inv, 1);
        cb_pop_front(cb_p, T);
    }
    if constexpr (nq == 0) {
        return;
    }

    cb_wait_front(cb_np, T);  // also read by the writer, which never pops it
    cb_wait_front(cb_q, nq * T);
    if constexpr (k > 0) {
        cb_wait_front(cb_s, k * T);
    }

    {
        DeviceZoneScopedN("C_prod");
        // ---- elementwise products q_i * s_j, summed over the T pages: one tile per (i, j)
        for (uint32_t i = 0; i < nq; ++i) {
#ifdef ATTNRES_EARLY_PROD
            for (uint32_t j = k; j < d; ++j) {
#else
            for (uint32_t j = 0; j < d; ++j) {
#endif
                const uint32_t scb = j < k ? cb_s : cb_np;
                const uint32_t sbase = j < k ? j * T : 0;
                tile_regs_acquire();
                reconfig_data_format(cb_q, scb);
                mul_init(cb_q, scb);
                for (uint32_t t = 0; t < T; ++t) {
                    mul_tiles(cb_q, scb, i * T + t, sbase + t, t);
                }
                sum_dst(T);
                tile_regs_commit();
                tile_regs_wait();
                put(0, cb_prod);
                tile_regs_release();
            }
        }
    }
    {
        DeviceZoneScopedN("C_soft");
        // ---- logits and softmax over depth, per query
        cb_wait_front(cb_prod, nq * d);
        for (uint32_t i = 0; i < nq; ++i) {
            tile_regs_acquire();
            reduce_scalars(cb_prod, i * d, d, cb_w);
            if constexpr (d > 1) {
                constexpr uint32_t m = 7, s = 6;
                binary_max_tile_init();
                scal::bmax<SIT>(0, 1, m);
                for (uint32_t j = 2; j < d; ++j) {
                    scal::bmax<SIT>(m, j, m);
                }
                sub_binary_tile_init();
                for (uint32_t j = 0; j < d; ++j) {
                    scal::sub<SIT>(j, m, j);
                }
                exp_tile_init();
                for (uint32_t j = 0; j < d; ++j) {
                    scal::exp<SIT>(j);
                }
                add_binary_tile_init();
                scal::add<SIT>(0, 1, s);
                for (uint32_t j = 2; j < d; ++j) {
                    scal::add<SIT>(s, j, s);
                }
                recip_tile_init();
                scal::recip<SIT>(s);
                mul_binary_tile_init();
                for (uint32_t j = 0; j < d; ++j) {
                    scal::mul<SIT>(j, s, j);
                }
            }
            tile_regs_commit();
            tile_regs_wait();
            cb_reserve_back(cb_w, d);
            for (uint32_t j = 0; j < d; ++j) {
                pack_tile(j, cb_w);
            }
            cb_push_back(cb_w, d);
            tile_regs_release();
        }
        cb_pop_front(cb_prod, nq * d);
    }
    {
        DeviceZoneScopedN("C_out");
        // ---- out_i = sum_j w_ij * s_j  (d == 1: w = 1, out = n)
        cb_wait_front(cb_w, nq * d);
        cb_reserve_back(cb_out, nq * T);
        for (uint32_t i = 0; i < nq; ++i) {
            for (uint32_t t = 0; t < T; ++t) {
                tile_regs_acquire();
                if constexpr (d == 1) {
                    reconfig_data_format_srca(cb_np);
                    copy_init(cb_np);
                    copy_tile(cb_np, t, 0);
                } else {
                    // FPU products into dst[j], then SFPU sum (an FPU write may not follow an SFPU read of the same
                    // dst tile inside one acquire: no stall orders them)
                    for (uint32_t j = 0; j < d; ++j) {
                        const uint32_t scb = j < k ? cb_s : cb_np;
                        const uint32_t sidx = j < k ? j * T + t : t;
                        reconfig_data_format(scb, cb_w);
                        mul_bcast_scalar_init(scb, cb_w);
                        mul_tiles_bcast_scalar(scb, cb_w, sidx, i * d + j, j);
                    }
                    sum_dst(d);
                }
                tile_regs_commit();
                tile_regs_wait();
                pack_tile(0, cb_out);
                tile_regs_release();
            }
        }
        cb_push_back(cb_out, nq * T);
    }
    if constexpr (has_norm) {
        // weighted rms of out_0 -> bf16 cb_nn: FusedRowNorm's op sequence (C_inv, C_n with weight) on cb_out, so
        // the result equals a separate FusedRowNorm program reading out_0's dense twin
        DeviceZoneScopedN("C_norm");
        constexpr uint32_t cb_nw = 5, cb_nn = 19;
        cb_wait_front(cb_out, T);  // also read by the writer, which never pops it
        tile_regs_acquire();
        reconfig_data_format(cb_out, cb_out);
        mul_init(cb_out, cb_out);
        for (uint32_t t = 0; t < T; ++t) {
            mul_tiles(cb_out, cb_out, t, t, t);
        }
        sum_dst(T);
        tile_regs_commit();
        tile_regs_wait();
        pack_reconfig_data_format(cb_sq);
        put(0, cb_sq);
        tile_regs_release();

        cb_wait_front(cb_sq, 1);
        tile_regs_acquire();
        reduce_scalars(cb_sq, 0, 1, cb_inv);
        binop_with_scalar_tile_init();
        scal::mul_unary<SIT>(0, inv_c);
        scal::add_unary<SIT>(0, eps);
        rsqrt_tile_init();
        scal::rsqrt<SIT>(0);
        tile_regs_commit();
        tile_regs_wait();
        put(0, cb_inv);
        tile_regs_release();
        cb_pop_front(cb_sq, 1);

        cb_wait_front(cb_inv, 1);
        tile_regs_acquire();
        reconfig_data_format(cb_out, cb_inv);
        mul_bcast_scalar_init(cb_out, cb_inv);
        for (uint32_t t = 0; t < T; ++t) {
            mul_tiles_bcast_scalar(cb_out, cb_inv, t, 0, t);
        }
        cb_wait_front(cb_nw, T);
        reconfig_data_format_srca(cb_nw);
        copy_init(cb_nw);
        for (uint32_t t = 0; t < T; ++t) {
            copy_tile(cb_nw, t, T + t);
        }
        mul_binary_tile_init();
        for (uint32_t t = 0; t < T; ++t) {
            mul_binary_tile(t, T + t, t);
        }
        tile_regs_commit();
        tile_regs_wait();
        cb_reserve_back(cb_nn, T);
        pack_reconfig_data_format(cb_nn);
        for (uint32_t t = 0; t < T; ++t) {
            pack_tile(t, cb_nn);
        }
        cb_push_back(cb_nn, T);
        tile_regs_release();
        cb_pop_front(cb_inv, 1);
    }
}
