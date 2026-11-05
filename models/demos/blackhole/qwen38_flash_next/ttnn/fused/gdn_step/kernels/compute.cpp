// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// One GDN decode step for one (lane, value head) work item per loop: conv + SiLU, q/k l2 norm, the decay and write
// gates, the fp32 delta-rule state update, the read-out, the gated RMSNorm.  Rounding points follow tt/gdn.py (the
// oracle): fp32 conv sum -> bf16 -> silu -> bf16; the l2 norm's bf16 chain; beta in bf16; the state in fp32 (SFPU
// multiply / fp32 accumulate); matmul operands at HiFi4.  DST holds fp32 (4 tiles per acquire).
// Runtime args: 0 items on this core.

#include <cstdint>

#define REDUCE_OP (PoolType::SUM)
#define REDUCE_DIM (ReduceDim::REDUCE_ROW)

#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/dataflow/circular_buffer.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/bcast.h"
#include "api/compute/matmul.h"
#include "api/compute/reduce.h"
#include "api/compute/transpose.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/exp.h"
#include "../../kernels/source_softplus.h"
#include "api/compute/eltwise_unary/sqrt.h"
#include "../../kernels/source_reciprocal.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"

using namespace ckernel;

namespace {
constexpr uint32_t cb_p = 0, cb_s = 1, cb_t = 2, cb_z = 3, cb_ab = 4, cb_dtna = 5, cb_w = 6, cb_state = 7;
constexpr uint32_t cb_mask = 8, cb_scaler = 9, cb_convsum = 10, cb_qkv = 11, cb_sq = 12, cb_stat = 13, cb_unit = 14;
constexpr uint32_t cb_qrow = 15, cb_krow = 16, cb_kcol = 17, cb_beta = 18, cb_decay = 19, cb_sdec = 20, cb_sdecc = 21;
constexpr uint32_t cb_vread = 22, cb_deltab = 23, cb_snew = 24, cb_outs = 25, cb_outg = 26, cb_rs = 27;
// The gated norm reuses recurrence buffers of the same format and depth.
constexpr uint32_t cb_obf = cb_convsum, cb_nrmw = cb_sq, cb_nrm = cb_unit, cb_sig = cb_vread, cb_sqo = cb_deltab;

constexpr uint32_t head_tiles = 4;  // 128 / 32
constexpr uint32_t qkv_tiles = 12;  // q, k, v x head_tiles
constexpr uint32_t head_state_tiles = 16;
constexpr uint32_t eps_bits = 0x358637BD;       // 1e-6f: l2 norm eps and RMS_NORM_EPS
constexpr uint32_t qk_scale_bits = 0x3DB504F3;  // 128^-0.5
constexpr uint32_t one_bits = 0x3F800000, twenty_bits = 0x41A00000;

// conv[t] = silu(bf16(sum_i slot_i[t] * tap_i[t])) for the 12 q/k/v tiles; taps broadcast row 0.  The reader pushes
// the inputs per tile (cb_s tile-major: tile t's slots at 3t + i; cb_t: its taps at 4t + i), so tile t starts as soon
// as its own eight tiles landed.
ALWI void conv_silu() {
    for (uint32_t t = 0; t < qkv_tiles; ++t) {
        CircularBuffer(cb_p).wait_front(t + 1);
        CircularBuffer(cb_s).wait_front(3 * (t + 1));
        CircularBuffer(cb_t).wait_front(4 * (t + 1));
        tile_regs_acquire();
        reconfig_data_format(cb_s, cb_t);
        mul_bcast_rows_init(cb_s, cb_t);
        for (uint32_t tap = 0; tap < 3; ++tap) {
            mul_tiles_bcast_rows(cb_s, cb_t, 3 * t + tap, 4 * t + tap, 0);
        }
        reconfig_data_format(cb_p, cb_t);
        mul_bcast_rows_init(cb_p, cb_t);
        mul_tiles_bcast_rows(cb_p, cb_t, t, 4 * t + 3, 0);
        tile_regs_commit();
        tile_regs_wait();
        CircularBuffer(cb_convsum).reserve_back(1);
        pack_reconfig_data_format(cb_convsum);
        pack_tile(0, cb_convsum);
        CircularBuffer(cb_convsum).push_back(1);
        tile_regs_release();

        CircularBuffer(cb_convsum).wait_front(1);
        tile_regs_acquire();
        reconfig_data_format_srca(cb_convsum);
        copy_init(cb_convsum);
        copy_tile(cb_convsum, 0, 0);
        silu_tile_init();
        silu_tile(0);
        tile_regs_commit();
        tile_regs_wait();
        CircularBuffer(cb_qkv).reserve_back(1);
        pack_reconfig_data_format(cb_qkv);
        pack_tile(0, cb_qkv);
        CircularBuffer(cb_qkv).push_back(1);
        tile_regs_release();
        CircularBuffer(cb_convsum).pop_front(1);
    }
}

// unit = x * rsqrt(sum(x*x) + eps) in the oracle's bf16 chain; the fp32 copy of unit -> out_cb.
ALWI void l2_norm(uint32_t base, uint32_t out_cb) {
    reconfig_data_format(cb_qkv, cb_qkv);
    mul_init(cb_qkv, cb_qkv);
    pack_reconfig_data_format(cb_sq);
    CircularBuffer(cb_sq).reserve_back(head_tiles);
    for (uint32_t t = 0; t < head_tiles; ++t) {
        tile_regs_acquire();
        mul_tiles(cb_qkv, cb_qkv, base + t, base + t, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_sq);
        tile_regs_release();
    }
    CircularBuffer(cb_sq).push_back(head_tiles);

    CircularBuffer(cb_sq).wait_front(head_tiles);
    reconfig_data_format(cb_scaler, cb_sq);
    reduce_init<PoolType::SUM, ReduceDim::REDUCE_ROW>(cb_sq, cb_scaler, cb_stat);
    pack_reconfig_data_format(cb_stat);
    tile_regs_acquire();
    for (uint32_t t = 0; t < head_tiles; ++t) {
        reduce_tile<PoolType::SUM, ReduceDim::REDUCE_ROW>(cb_sq, cb_scaler, t, 0, 0);
    }
    tile_regs_commit();
    tile_regs_wait();
    CircularBuffer(cb_stat).reserve_back(1);
    pack_tile(0, cb_stat);
    CircularBuffer(cb_stat).push_back(1);
    tile_regs_release();
    reduce_uninit();
    CircularBuffer(cb_sq).pop_front(head_tiles);

    // + eps -> bf16, then rsqrt -> bf16
    for (uint32_t step = 0; step < 2; ++step) {
        CircularBuffer(cb_stat).wait_front(1);
        reconfig_data_format_srca(cb_stat);
        copy_init(cb_stat);
        if (step == 0) {
            binop_with_scalar_tile_init();
        } else {
            sqrt_tile_init();
            qwen38_recip_tile_init();
        }
        tile_regs_acquire();
        copy_tile(cb_stat, 0, 0);
        if (step == 0) {
            add_unary_tile(0, eps_bits);
        } else {
            sqrt_tile(0);
            qwen38_recip_tile(0);
        }
        tile_regs_commit();
        tile_regs_wait();
        CircularBuffer(cb_stat).reserve_back(1);
        pack_tile(0, cb_stat);
        CircularBuffer(cb_stat).push_back(1);
        tile_regs_release();
        CircularBuffer(cb_stat).pop_front(1);
    }

    CircularBuffer(cb_stat).wait_front(1);
    reconfig_data_format(cb_qkv, cb_stat);
    mul_bcast_cols_init(cb_qkv, cb_stat);
    pack_reconfig_data_format(cb_unit);
    CircularBuffer(cb_unit).reserve_back(head_tiles);
    for (uint32_t t = 0; t < head_tiles; ++t) {
        tile_regs_acquire();
        mul_tiles_bcast_cols(cb_qkv, cb_stat, base + t, 0, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_unit);
        tile_regs_release();
    }
    CircularBuffer(cb_unit).push_back(head_tiles);
    CircularBuffer(cb_stat).pop_front(1);

    CircularBuffer(cb_unit).wait_front(head_tiles);
    reconfig_data_format_srca(cb_unit);
    copy_init(cb_unit);
    pack_reconfig_data_format(out_cb);
    CircularBuffer(out_cb).reserve_back(head_tiles);
    for (uint32_t t = 0; t < head_tiles; ++t) {
        tile_regs_acquire();
        copy_tile(cb_unit, t, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, out_cb);
        tile_regs_release();
    }
    CircularBuffer(out_cb).push_back(head_tiles);
    CircularBuffer(cb_unit).pop_front(head_tiles);
}

// beta = bf16(sigmoid(b)) and decay = exp(neg_exp_A * softplus(a + dt_bias)) as full fp32 tiles.
ALWI void gate_scalars() {
    CircularBuffer(cb_ab).wait_front(2);
    CircularBuffer(cb_dtna).wait_front(2);

    tile_regs_acquire();
    reconfig_data_format_srca(cb_ab);
    unary_bcast_init<BroadcastType::SCALAR>(cb_ab);
    unary_bcast<BroadcastType::SCALAR>(cb_ab, 1, 0);
    sigmoid_tile_init<false>();
    sigmoid_tile<VectorMode::RC, false>(0);
    tile_regs_commit();
    tile_regs_wait();
    CircularBuffer(cb_stat).reserve_back(1);
    pack_reconfig_data_format(cb_stat);
    pack_tile(0, cb_stat);
    CircularBuffer(cb_stat).push_back(1);
    tile_regs_release();

    CircularBuffer(cb_stat).wait_front(1);
    tile_regs_acquire();
    reconfig_data_format_srca(cb_stat);
    copy_init(cb_stat);
    copy_tile(cb_stat, 0, 0);
    tile_regs_commit();
    tile_regs_wait();
    CircularBuffer(cb_beta).reserve_back(1);
    pack_reconfig_data_format(cb_beta);
    pack_tile(0, cb_beta);
    CircularBuffer(cb_beta).push_back(1);
    tile_regs_release();
    CircularBuffer(cb_stat).pop_front(1);

    tile_regs_acquire();
    reconfig_data_format_srca(cb_ab);
    unary_bcast_init<BroadcastType::SCALAR>(cb_ab);
    unary_bcast<BroadcastType::SCALAR>(cb_ab, 0, 0);
    reconfig_data_format_srca(cb_dtna);
    copy_init(cb_dtna);
    copy_tile(cb_dtna, 0, 1);
    add_binary_tile_init();
    add_binary_tile(0, 1, 0);
    softplus_tile_init();
    qwen38_softplus_tile(0, one_bits, one_bits, twenty_bits);
    copy_tile(cb_dtna, 1, 1);
    mul_binary_tile_init();
    mul_binary_tile(0, 1, 0);
    exp_tile_init<false>();
    exp_tile<false>(0);
    tile_regs_commit();
    tile_regs_wait();
    CircularBuffer(cb_decay).reserve_back(1);
    pack_reconfig_data_format(cb_decay);
    pack_tile(0, cb_decay);
    CircularBuffer(cb_decay).push_back(1);
    tile_regs_release();
    CircularBuffer(cb_ab).pop_front(2);
    CircularBuffer(cb_dtna).pop_front(2);
}

// S' = S * decay; v_read = k S'; delta_b = ((v - v_read) * mask) * beta; S_new = S' + k^T delta_b; o = (q S_new) *
// 128^-0.5.
ALWI void recurrence() {
    CircularBuffer(cb_state).wait_front(head_state_tiles);
    CircularBuffer(cb_decay).wait_front(1);
    CircularBuffer(cb_beta).wait_front(1);
    CircularBuffer(cb_mask).wait_front(1);
    CircularBuffer(cb_krow).wait_front(head_tiles);
    CircularBuffer(cb_qrow).wait_front(head_tiles);
    CircularBuffer(cb_qkv).wait_front(head_tiles);  // v

    reconfig_data_format_srca(cb_state);
    copy_init(cb_state);
    mul_binary_tile_init();
    pack_reconfig_data_format(cb_sdec);
    CircularBuffer(cb_sdec).reserve_back(head_state_tiles);
    CircularBuffer(cb_sdecc).reserve_back(head_state_tiles);
    for (uint32_t t = 0; t < head_state_tiles; t += 2) {
        tile_regs_acquire();
        copy_tile(cb_state, t, 0);
        copy_tile(cb_decay, 0, 1);
        mul_binary_tile(0, 1, 0);
        copy_tile(cb_state, t + 1, 2);
        copy_tile(cb_decay, 0, 3);
        mul_binary_tile(2, 3, 2);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_sdec);
        pack_tile(2, cb_sdec);
        pack_tile(0, cb_sdecc);
        pack_tile(2, cb_sdecc);
        tile_regs_release();
    }
    CircularBuffer(cb_sdec).push_back(head_state_tiles);
    CircularBuffer(cb_sdecc).push_back(head_state_tiles);
    CircularBuffer(cb_sdec).wait_front(head_state_tiles);
    CircularBuffer(cb_sdecc).wait_front(head_state_tiles);

    reconfig_data_format<SrcOrder::Reverse>(cb_krow, cb_sdec);
    matmul_init(cb_krow, cb_sdec);
    pack_reconfig_data_format(cb_vread);
    tile_regs_acquire();
    for (uint32_t j = 0; j < head_tiles; ++j) {
        for (uint32_t i = 0; i < head_tiles; ++i) {
            matmul_tiles(cb_krow, cb_sdec, i, i * head_tiles + j, j);
        }
    }
    tile_regs_commit();
    tile_regs_wait();
    CircularBuffer(cb_vread).reserve_back(head_tiles);
    for (uint32_t j = 0; j < head_tiles; ++j) {
        pack_tile(j, cb_vread);
    }
    CircularBuffer(cb_vread).push_back(head_tiles);
    tile_regs_release();

    CircularBuffer(cb_vread).wait_front(head_tiles);
    pack_reconfig_data_format(cb_deltab);
    CircularBuffer(cb_deltab).reserve_back(head_tiles);
    for (uint32_t j = 0; j < head_tiles; ++j) {
        tile_regs_acquire();
        reconfig_data_format_srca(cb_qkv);
        copy_init(cb_qkv);
        copy_tile(cb_qkv, j, 0);
        reconfig_data_format_srca(cb_vread);
        copy_init(cb_vread);
        copy_tile(cb_vread, j, 1);
        sub_binary_tile_init();
        sub_binary_tile(0, 1, 0);
        reconfig_data_format_srca(cb_mask);
        copy_init(cb_mask);
        copy_tile(cb_mask, 0, 1);
        mul_binary_tile_init();
        mul_binary_tile(0, 1, 0);
        reconfig_data_format_srca(cb_beta);
        copy_init(cb_beta);
        copy_tile(cb_beta, 0, 1);
        mul_binary_tile(0, 1, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_deltab);
        tile_regs_release();
    }
    CircularBuffer(cb_deltab).push_back(head_tiles);
    CircularBuffer(cb_vread).pop_front(head_tiles);
    CircularBuffer(cb_qkv).pop_front(head_tiles);

    reconfig_data_format_srca(cb_krow);
    transpose_init(cb_krow);
    pack_reconfig_data_format(cb_kcol);
    CircularBuffer(cb_kcol).reserve_back(head_tiles);
    for (uint32_t i = 0; i < head_tiles; ++i) {
        tile_regs_acquire();
        transpose_tile(cb_krow, i, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_kcol);
        tile_regs_release();
    }
    CircularBuffer(cb_kcol).push_back(head_tiles);
    CircularBuffer(cb_krow).pop_front(head_tiles);

    CircularBuffer(cb_kcol).wait_front(head_tiles);
    CircularBuffer(cb_deltab).wait_front(head_tiles);
    CircularBuffer(cb_outs).reserve_back(head_state_tiles);  // the writer has drained the previous item's state
    CircularBuffer(cb_snew).reserve_back(head_state_tiles);
    pack_reconfig_data_format(cb_snew);
    for (uint32_t i = 0; i < head_tiles; ++i) {
        tile_regs_acquire();
        reconfig_data_format_srca(cb_sdecc);
        copy_init(cb_sdecc);
        for (uint32_t j = 0; j < head_tiles; ++j) {
            copy_tile(cb_sdecc, i * head_tiles + j, j);
        }
        reconfig_data_format<SrcOrder::Reverse>(cb_kcol, cb_deltab);
        matmul_init(cb_kcol, cb_deltab);
        for (uint32_t j = 0; j < head_tiles; ++j) {
            matmul_tiles(cb_kcol, cb_deltab, i, j, j);
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t j = 0; j < head_tiles; ++j) {
            pack_tile(j, cb_snew);
        }
        tile_regs_release();
    }
    CircularBuffer(cb_snew).push_back(head_state_tiles);
    CircularBuffer(cb_sdec).pop_front(head_state_tiles);
    CircularBuffer(cb_sdecc).pop_front(head_state_tiles);
    CircularBuffer(cb_kcol).pop_front(head_tiles);
    CircularBuffer(cb_deltab).pop_front(head_tiles);
    CircularBuffer(cb_state).pop_front(head_state_tiles);
    CircularBuffer(cb_decay).pop_front(1);
    CircularBuffer(cb_beta).pop_front(1);
    CircularBuffer(cb_mask).pop_front(1);

    CircularBuffer(cb_snew).wait_front(head_state_tiles);
    reconfig_data_format<SrcOrder::Reverse>(cb_qrow, cb_snew);
    matmul_init(cb_qrow, cb_snew);
    pack_reconfig_data_format(cb_obf);
    tile_regs_acquire();
    for (uint32_t j = 0; j < head_tiles; ++j) {
        for (uint32_t i = 0; i < head_tiles; ++i) {
            matmul_tiles(cb_qrow, cb_snew, i, i * head_tiles + j, j);
        }
    }
    binop_with_scalar_tile_init();
    for (uint32_t j = 0; j < head_tiles; ++j) {
        mul_unary_tile(j, qk_scale_bits);
    }
    tile_regs_commit();
    tile_regs_wait();
    CircularBuffer(cb_obf).reserve_back(head_tiles);
    for (uint32_t j = 0; j < head_tiles; ++j) {
        pack_tile(j, cb_obf);
    }
    CircularBuffer(cb_obf).push_back(head_tiles);
    tile_regs_release();
    CircularBuffer(cb_snew).pop_front(head_state_tiles);
    CircularBuffer(cb_outs).push_back(head_state_tiles);  // same bytes as cb_snew: hands the new state to the writer
    CircularBuffer(cb_qrow).pop_front(head_tiles);
}

// gated = bf16(bf16(w * bf16(o * rsqrt(mean(o^2) + eps))) * sigmoid(z)).
ALWI void gated_norm() {
    CircularBuffer(cb_obf).wait_front(head_tiles);
    CircularBuffer(cb_w).wait_front(head_tiles);
    CircularBuffer(cb_z).wait_front(head_tiles);

    reconfig_data_format(cb_obf, cb_obf);
    mul_init(cb_obf, cb_obf);
    pack_reconfig_data_format(cb_sqo);
    CircularBuffer(cb_sqo).reserve_back(head_tiles);
    for (uint32_t j = 0; j < head_tiles; ++j) {
        tile_regs_acquire();
        mul_tiles(cb_obf, cb_obf, j, j, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_sqo);
        tile_regs_release();
    }
    CircularBuffer(cb_sqo).push_back(head_tiles);

    CircularBuffer(cb_sqo).wait_front(head_tiles);
    reconfig_data_format(cb_scaler, cb_sqo);
    reduce_init<PoolType::SUM, ReduceDim::REDUCE_ROW>(cb_sqo, cb_scaler, cb_rs);
    pack_reconfig_data_format(cb_rs);
    binop_with_scalar_tile_init();
    sqrt_tile_init();
    qwen38_recip_tile_init();
    tile_regs_acquire();
    for (uint32_t j = 0; j < head_tiles; ++j) {
        reduce_tile<PoolType::SUM, ReduceDim::REDUCE_ROW>(cb_sqo, cb_scaler, j, 1, 0);
    }
    add_unary_tile(0, eps_bits);
    sqrt_tile(0);
    qwen38_recip_tile(0);
    tile_regs_commit();
    tile_regs_wait();
    CircularBuffer(cb_rs).reserve_back(1);
    pack_tile(0, cb_rs);
    CircularBuffer(cb_rs).push_back(1);
    tile_regs_release();
    reduce_uninit();
    CircularBuffer(cb_sqo).pop_front(head_tiles);

    CircularBuffer(cb_rs).wait_front(1);
    reconfig_data_format(cb_obf, cb_rs);
    mul_bcast_cols_init(cb_obf, cb_rs);
    pack_reconfig_data_format(cb_nrm);
    CircularBuffer(cb_nrm).reserve_back(head_tiles);
    for (uint32_t j = 0; j < head_tiles; ++j) {
        tile_regs_acquire();
        mul_tiles_bcast_cols(cb_obf, cb_rs, j, 0, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_nrm);
        tile_regs_release();
    }
    CircularBuffer(cb_nrm).push_back(head_tiles);
    CircularBuffer(cb_rs).pop_front(1);
    CircularBuffer(cb_obf).pop_front(head_tiles);

    CircularBuffer(cb_nrm).wait_front(head_tiles);
    reconfig_data_format(cb_nrm, cb_w);
    mul_bcast_rows_init(cb_nrm, cb_w);
    pack_reconfig_data_format(cb_nrmw);
    CircularBuffer(cb_nrmw).reserve_back(head_tiles);
    for (uint32_t j = 0; j < head_tiles; ++j) {
        tile_regs_acquire();
        mul_tiles_bcast_rows(cb_nrm, cb_w, j, j, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_nrmw);
        tile_regs_release();
    }
    CircularBuffer(cb_nrmw).push_back(head_tiles);
    CircularBuffer(cb_nrm).pop_front(head_tiles);

    reconfig_data_format_srca(cb_z);
    copy_init(cb_z);
    sigmoid_tile_init<false>();
    pack_reconfig_data_format(cb_sig);
    CircularBuffer(cb_sig).reserve_back(head_tiles);
    for (uint32_t j = 0; j < head_tiles; ++j) {
        tile_regs_acquire();
        copy_tile(cb_z, j, 0);
        sigmoid_tile<VectorMode::RC, false>(0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_sig);
        tile_regs_release();
    }
    CircularBuffer(cb_sig).push_back(head_tiles);
    CircularBuffer(cb_z).pop_front(head_tiles);

    CircularBuffer(cb_nrmw).wait_front(head_tiles);
    CircularBuffer(cb_sig).wait_front(head_tiles);
    pack_reconfig_data_format(cb_outg);
    CircularBuffer(cb_outg).reserve_back(head_tiles);
    mul_binary_tile_init();
    for (uint32_t j = 0; j < head_tiles; ++j) {
        tile_regs_acquire();
        reconfig_data_format_srca(cb_nrmw);
        copy_init(cb_nrmw);
        copy_tile(cb_nrmw, j, 0);
        reconfig_data_format_srca(cb_sig);
        copy_init(cb_sig);
        copy_tile(cb_sig, j, 1);
        mul_binary_tile(0, 1, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_outg);
        tile_regs_release();
    }
    CircularBuffer(cb_outg).push_back(head_tiles);
    CircularBuffer(cb_nrmw).pop_front(head_tiles);
    CircularBuffer(cb_sig).pop_front(head_tiles);
}
}  // namespace

void kernel_main() {
    const uint32_t items = get_arg_val<uint32_t>(0);
    compute_kernel_hw_startup(cb_s, cb_t, cb_convsum);
    CircularBuffer(cb_scaler).wait_front(2);  // reader-built: tile 0 = 1.0 (sums), tile 1 = 1/128 (means)

    for (uint32_t item = 0; item < items; ++item) {
        {
            CircularBuffer(cb_p).wait_front(1);  // the first tile's group; the conv waits for the rest tile by tile
            CircularBuffer(cb_s).wait_front(3);
            CircularBuffer(cb_t).wait_front(4);
        }
        {
            conv_silu();
        }
        CircularBuffer(cb_p).pop_front(qkv_tiles);
        CircularBuffer(cb_s).pop_front(3 * qkv_tiles);
        CircularBuffer(cb_t).pop_front(4 * qkv_tiles);

        CircularBuffer(cb_qkv).wait_front(2 * head_tiles);
        {
            l2_norm(0, cb_qrow);
            l2_norm(head_tiles, cb_krow);
        }
        CircularBuffer(cb_qkv).pop_front(2 * head_tiles);

        {
            gate_scalars();
        }
        {
            recurrence();
        }
        {
            gated_norm();
        }
    }
}
