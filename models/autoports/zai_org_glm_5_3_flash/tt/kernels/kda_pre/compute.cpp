// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Fused KDA "pre" — compute (one source, ROWS / TILES by compile-time arg):
//   gates (4 column tiles n of the core):
//     F = f_a @ w_fb[:, n] (FPU, fp32 DST, HiFi4);  g = sigmoid((F + dt_bias) * rate) * (lower_bound * active)
//     gate_out = g_a @ w_gb[:, n]
//   beta (cores 0, 1): sigmoid(b) * active
//   conv units (ROWS: one packed tile per KDA_PACK user rows, kda_pre_common.hpp; TILES: one column tile):
//     silu(h0 * tap0 + h1 * tap1 + h2 * tap2 + qkv * tap_cur)       (exact fp32 SFPU arithmetic on bf16 operands)
// The per-column constants (dt_bias, rate; the TILES-mode taps) are row-0 tiles applied with the SFPU row broadcast,
// the masks column-0 tiles with the column broadcast; ROWS-mode taps are packed exactly like the data (element-wise).

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/sfpu_binary_bcast.h"
#include "api/compute/tile_move_copy.h"
#include "api/dataflow/circular_buffer.h"

#include "kda_pre_common.hpp"
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/mk/phase_args.hpp"

using namespace kdap;

namespace {

constexpr uint32_t ROWS = get_compile_time_arg_val(PH_CT0 + 0);

// dst[d] <- tile i of cb. The kernel copies CBs of different formats (bf16 data / fp32 constants) and alternates
// with matmuls, and copy_init does not reconfigure data formats: SrcA is set to cb's format on every load.
FORCE_INLINE void load(uint32_t cb, uint32_t i, uint32_t d) {
    reconfig_data_format_srca(cb);
    copy_init(cb);
    copy_tile(cb, i, d);
}

// SrcA / SrcB for the matmul in0 @ in1 (in0 -> SrcB, in1 -> SrcA)
FORCE_INLINE void matmul_formats(uint32_t in0, uint32_t in1) {
    reconfig_data_format<SrcOrder::Reverse>(in0, in1);
    matmul_init(in0, in1);
}

FORCE_INLINE void pack_to(uint32_t d, uint32_t cb) {
    pack_reconfig_data_format(cb);
    pack_tile(d, cb);
}

// dst[d] *= tap (ROWS: element-wise packed tap; TILES: row-0 tap broadcast over the rows)
FORCE_INLINE void mul_tap(uint32_t d, uint32_t tap_dst) {
    if constexpr (ROWS) {
        mul_binary_tile_init();
        mul_binary_tile(d, tap_dst, d);
    } else {
        sfpu_mul_bcast_row_init();
        sfpu_mul_bcast_row(d, tap_dst);
    }
}

}  // namespace

void kernel_main() {
    const uint32_t kk = get_arg_val<uint32_t>(ph_rt0 + 0);
    const uint32_t units = get_arg_val<uint32_t>(ph_rt0 + 1);  // ROWS: user-row pairs; TILES: 12 column tiles
    if (kk >= N_CORES * RSPLIT) {
        return;  // a core past the shard's heads: no kda_pre work
    }
    const uint32_t k = kk % N_CORES, rg = kk / N_CORES;
    uint32_t u0 = 0, u1 = units;
    if constexpr (ROWS) {
        rsplit_range(units, rg, u0, u1);
        u1 = u0 + (u1 - u0 + UP - 1) / UP;  // KDA_PACK: packed tiles of UP pairs
    }

    compute_kernel_hw_startup<SrcOrder::Reverse>(CB_FA, CB_WFB, CB_G);
    CircularBuffer cb_fa(CB_FA), cb_ga(CB_GA), cb_wfb(CB_WFB), cb_wgb(CB_WGB), cb_dt(CB_DT), cb_rate(CB_RATE);
    CircularBuffer cb_act(CB_ACT), cb_g(CB_G), cb_go(CB_GO);
    if (rg == 0) {  // row group 0: g / gate_out / beta
        cb_fa.wait_front(K_TILES);
        cb_ga.wait_front(K_TILES);
        cb_wfb.wait_front(K_TILES * G_PER_CORE);
        cb_wgb.wait_front(K_TILES * G_PER_CORE);
        cb_dt.wait_front(G_PER_CORE);
        cb_rate.wait_front(G_PER_CORE);
        cb_act.wait_front(2);

        // ---- g
        cb_g.reserve_back(G_PER_CORE);
        for (uint32_t j = 0; j < G_PER_CORE; ++j) {
            tile_regs_acquire();
            matmul_formats(CB_FA, CB_WFB);
            for (uint32_t kt = 0; kt < K_TILES; ++kt) {
                matmul_tiles(CB_FA, CB_WFB, kt, kt * G_PER_CORE + j, 0);
            }
            load(CB_DT, j, 1);
            sfpu_add_bcast_row_init();
            sfpu_add_bcast_row(0, 1);
            load(CB_RATE, j, 1);
            sfpu_mul_bcast_row_init();
            sfpu_mul_bcast_row(0, 1);
            sigmoid_tile_init();
            sigmoid_tile(0);
            load(CB_ACT, 1, 1);  // lower_bound * active in column 0
            sfpu_mul_bcast_col_init();
            sfpu_mul_bcast_col(0, 1);
            tile_regs_commit();
            tile_regs_wait();
            pack_to(0, CB_G);
            tile_regs_release();
        }
        cb_g.push_back(G_PER_CORE);
        // ---- gate_out
        cb_go.reserve_back(G_PER_CORE);
        for (uint32_t j = 0; j < G_PER_CORE; ++j) {
            tile_regs_acquire();
            matmul_formats(CB_GA, CB_WGB);
            for (uint32_t kt = 0; kt < K_TILES; ++kt) {
                matmul_tiles(CB_GA, CB_WGB, kt, kt * G_PER_CORE + j, 0);
            }
            tile_regs_commit();
            tile_regs_wait();
            pack_to(0, CB_GO);
            tile_regs_release();
        }
        cb_go.push_back(G_PER_CORE);
        // ---- beta
        if (k < BETA_TILES) {
            CircularBuffer cb_b(CB_B), cb_beta(CB_BETA);
            cb_b.wait_front(1);
            cb_beta.reserve_back(1);
            tile_regs_acquire();
            load(CB_B, 0, 0);
            sigmoid_tile_init();
            sigmoid_tile(0);
            load(CB_ACT, 0, 1);
            sfpu_mul_bcast_col_init();
            sfpu_mul_bcast_col(0, 1);
            tile_regs_commit();
            tile_regs_wait();
            pack_to(0, CB_BETA);
            tile_regs_release();
            cb_beta.push_back(1);
            cb_b.pop_front(1);
        }
        cb_fa.pop_front(K_TILES);
        cb_ga.pop_front(K_TILES);
        cb_wfb.pop_front(K_TILES * G_PER_CORE);
        cb_wgb.pop_front(K_TILES * G_PER_CORE);
        cb_dt.pop_front(G_PER_CORE);
        cb_rate.pop_front(G_PER_CORE);
        cb_act.pop_front(2);
    }  // gate

    // ---- conv
    CircularBuffer cb_h0(CB_H0), cb_h1(CB_H1), cb_h2(CB_H2), cb_q(CB_Q), cb_t(CB_T), cb_conv(CB_CONV);
    if constexpr (ROWS) {
        cb_t.wait_front(4);
    }
    const uint32_t data_cbs[4] = {CB_H0, CB_H1, CB_H2, CB_Q};
    for (uint32_t u = u0; u < u1; ++u) {
        cb_h0.wait_front(1);
        cb_h1.wait_front(1);
        cb_h2.wait_front(1);
        cb_q.wait_front(1);
        if constexpr (!ROWS) {
            cb_t.wait_front(4);
        }
        cb_conv.reserve_back(1);
        tile_regs_acquire();
        load(CB_H0, 0, 0);
        load(CB_T, 0, 2);
        mul_tap(0, 2);
        for (uint32_t j = 1; j < 4; ++j) {
            load(data_cbs[j], 0, 1);
            load(CB_T, j, 2);
            mul_tap(1, 2);
            add_binary_tile_init();
            add_binary_tile(0, 1, 0);
        }
        silu_tile_init();
        silu_tile(0);
        tile_regs_commit();
        tile_regs_wait();
        pack_to(0, CB_CONV);
        tile_regs_release();
        cb_conv.push_back(1);
        cb_h0.pop_front(1);
        cb_h1.pop_front(1);
        cb_h2.pop_front(1);
        cb_q.pop_front(1);
        if constexpr (!ROWS) {
            cb_t.pop_front(4);
        }
    }
    if constexpr (ROWS) {
        cb_t.pop_front(4);
    }
}
