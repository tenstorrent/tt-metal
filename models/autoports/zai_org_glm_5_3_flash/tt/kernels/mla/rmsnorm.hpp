// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Stage 7b MLA megakernel: row RMSNorm of KX bf16 tiles (rows = tokens) on the SFPU in exact fp32 — mhc_pre's
// arithmetic (kernels/mhc_pre/compute.cpp: SFPU square + add, sfpu_reduce row sum, * 1/n + eps, rsqrt, column / row
// broadcast multiplies), used by the q_b phase of the reader cores (RMSNorm_1536(q_a)) and by the latent core
// (RMSNorm_512(kv_a)).
//   norm_inv:  CB_X[0 .. KX) -> CB_INV (one fp32 tile, column 0 = rsqrt(sum_j x_j^2 / n + eps) per row)
//   norm_tile: CB_X[i] * inv (column broadcast) * w (row 0 of an fp32 weight tile) -> dst 0
// CB_INV / the weight CB must be unpacked straight to DST (UnpackToDestFp32, set by the host).
#pragma once

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/sfpu_binary_bcast.h"
#include "api/compute/tile_move_copy.h"

namespace mlan {

// dst[d] <- tile i of cb
FORCE_INLINE void load(uint32_t cb, uint32_t i, uint32_t d) {
    reconfig_data_format_srca(cb);
    copy_tile_to_dst_init_short(cb);
    copy_tile(cb, i, d);
}

// CB_X tiles [0, KX) (already waited for) -> one fp32 tile of per-row rsqrt(mean square + eps) packed into cb_inv
template <uint32_t KX>
FORCE_INLINE void norm_inv(uint32_t cb_x, uint32_t cb_inv, uint32_t inv_n_bits, uint32_t eps_bits) {
    cb_reserve_back(cb_inv, 1);
    tile_regs_acquire();
    load(cb_x, 0, 0);
    square_tile_init();
    square_tile(0);
    for (uint32_t j = 1; j < KX; ++j) {
        load(cb_x, j, 1);
        square_tile_init();
        square_tile(1);
        add_binary_tile_init();
        add_binary_tile(0, 1, 0);
    }
    sfpu_reduce_init<PoolType::SUM, DataFormat::Float32>();
    sfpu_reduce<PoolType::SUM, DataFormat::Float32, ReduceDim::REDUCE_ROW>(0);
    binop_with_scalar_tile_init();
    mul_unary_tile(0, inv_n_bits);
    add_unary_tile(0, eps_bits);
    rsqrt_tile_init();
    rsqrt_tile(0);
    tile_regs_commit();
    tile_regs_wait();
    pack_reconfig_data_format(cb_inv);
    pack_tile(0, cb_inv);
    tile_regs_release();
    cb_push_back(cb_inv, 1);
}

// dst 0 <- CB_X[i] * inv * w   (inside the caller's tile_regs_acquire; uses dst 1 as scratch)
FORCE_INLINE void norm_tile(uint32_t cb_x, uint32_t i, uint32_t cb_inv, uint32_t cb_w, uint32_t wi) {
    load(cb_x, i, 0);
    load(cb_inv, 0, 1);
    sfpu_mul_bcast_col_init();
    sfpu_mul_bcast_col(0, 1);
    load(cb_w, wi, 1);
    sfpu_mul_bcast_row_init();
    sfpu_mul_bcast_row(0, 1);
}

}  // namespace mlan
