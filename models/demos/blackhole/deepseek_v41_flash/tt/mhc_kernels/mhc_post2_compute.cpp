// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// mHC post kernel v2 (one core): sum of the packed partials, RMS normalisation of the mixes (column NCOL holds the sum
// of squares), then the deepseek_prefill mhc_split_sinkhorn parametrisation:
//   comb = Sinkhorn( exp( mixes @ SEL_comb + base_comb ) )   via row/col-sum matmuls RB/CB
//   pre  = sigmoid( mixes @ SEL_pre  + base_pre  ) + eps
//   post = 2 * sigmoid( mixes @ SEL_post + base_post )
// pre and post share one tile (columns 0..3 / 4..7).  Every step is ONE dst section (matmul / SFPU / dest-reuse binary
// ops chained in dst); only the Sinkhorn normalisations round-trip through a CB.  The reciprocal runs on faces 0/1 only
// when T <= 16.

#include <cstdint>
#include "tools/profiler/kernel_profiler.hpp"
#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/matmul.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/exp.h"
#include "api/compute/eltwise_unary/recip.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/sfpu_split_includes.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "mhc_sinkhorn_sfpu.h"
#include "api/dataflow/circular_buffer.h"

namespace {

constexpr uint32_t CB_CONSTS = 0;
constexpr uint32_t CB_Q = 1;      // packed partial pages
constexpr uint32_t CB_SSEL = 2;   // summation tiles
constexpr uint32_t CB_RAW = 3;    // summed raw mixes
constexpr uint32_t CB_MIXES = 4;  // normalised mixes
constexpr uint32_t CB_PP = 5;     // out: pre | post
constexpr uint32_t CB_COMB = 6;   // out: Sinkhorn result in the SFPU layout (see mhc_sinkhorn_sfpu.h)
constexpr uint32_t CB_LOG = 8;    // comb logits, token rows (data mover permutes them into CB_SKIN)
constexpr uint32_t CB_SKIN = 9;   // comb logits in the SFPU layout

// const tile indices within CB_CONSTS
constexpr uint32_t SEL_COMB = 0, BASE_COMB = 1, RB = 2, CB_COL = 3, SEL_PP = 4, BASE_PP = 5, SCALE_PP = 6, OFF_PP = 7,
                   SEL_SS = 8;

constexpr uint32_t NCONST = 9;

}  // namespace

void kernel_main() {
    constexpr uint32_t iters = get_compile_time_arg_val(0);
    constexpr uint32_t eps_bits = get_compile_time_arg_val(1);
    constexpr uint32_t NPG = get_compile_time_arg_val(2);
    constexpr uint32_t ss_eps_bits = get_compile_time_arg_val(3);  // N * norm_eps
    constexpr uint32_t PGPG = get_compile_time_arg_val(4);
    compute_kernel_hw_startup(CB_RAW, CB_CONSTS, CB_COMB);  // must precede any other compute work

    CircularBuffer cb_consts(CB_CONSTS), cb_q(CB_Q), cb_ssel(CB_SSEL), cb_raw(CB_RAW), cb_mixes(CB_MIXES), cb_pp(CB_PP);
    DeviceZoneScopedN("QC_ALL");
    cb_consts.wait_front(NCONST);
    cb_ssel.wait_front((NPG + PGPG - 1) / PGPG);
    cb_q.wait_front(NPG);

    {
        DeviceZoneScopedN("QC_gotQ");
    }
    // ---- raw = sum over pages g, SSEL_g @ Q_page  (rows: token t, columns 0..mix_hc-1 mixes, NCOL sum of squares)
    // ----
    reconfig_data_format(CB_SSEL, CB_Q);
    matmul_init(CB_SSEL, CB_Q);
    tile_regs_acquire();
    for (uint32_t n = 0; n < NPG; ++n) {
        matmul_tiles(CB_SSEL, CB_Q, n / PGPG, n, 0);
    }
    tile_regs_commit();
    tile_regs_wait();
    cb_raw.reserve_back(1);
    pack_tile(0, CB_RAW);
    cb_raw.push_back(1);
    tile_regs_release();

    {
        DeviceZoneScopedN("QC_rawdone");
    }
    // ---- mixes = raw * rsqrt(ss + N*eps)   (sqrt(N) is folded into the SEL tiles) ----
    cb_raw.wait_front(1);
    reconfig_data_format(CB_RAW, CB_CONSTS);
    matmul_init(CB_RAW, CB_CONSTS);
    tile_regs_acquire();
    matmul_tiles(CB_RAW, CB_CONSTS, 0, SEL_SS, 0);  // every column = ss
    binop_with_scalar_tile_init();
    add_unary_tile(0, ss_eps_bits);
    rsqrt_tile_init();
    rsqrt_tile(0);
    mul_reuse_dest_init<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(CB_RAW);
    mul_reuse_dest_tiles<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(CB_RAW, 0, 0);
    tile_regs_commit();
    tile_regs_wait();
    cb_mixes.reserve_back(1);
    pack_tile(0, CB_MIXES);
    cb_mixes.push_back(1);
    tile_regs_release();
    cb_raw.pop_front(1);
    cb_mixes.wait_front(1);

    {
        DeviceZoneScopedN("QC_mixesdone");
    }
    // ---- pre | post = (sigmoid(mixes @ SEL_PP + base_PP) + OFF) * SCALE ----
    reconfig_data_format(CB_MIXES, CB_CONSTS);
    matmul_init(CB_MIXES, CB_CONSTS);
    tile_regs_acquire();
    matmul_tiles(CB_MIXES, CB_CONSTS, 0, SEL_PP, 0);
    add_reuse_dest_init<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(CB_CONSTS);
    add_reuse_dest_tiles<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(CB_CONSTS, BASE_PP, 0);
    sigmoid_tile_init();
    sigmoid_tile(0);
    add_reuse_dest_init<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(CB_CONSTS);
    add_reuse_dest_tiles<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(CB_CONSTS, OFF_PP, 0);
    mul_reuse_dest_init<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(CB_CONSTS);
    mul_reuse_dest_tiles<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(CB_CONSTS, SCALE_PP, 0);
    tile_regs_commit();
    tile_regs_wait();
    cb_pp.reserve_back(1);
    pack_tile(0, CB_PP);
    cb_pp.push_back(1);
    tile_regs_release();

    {
        DeviceZoneScopedN("QC_ppdone");
    }
    // ---- comb logits: mixes @ SEL_comb + base_comb (token rows; the data mover permutes them into the SFPU layout)
    // ----
    reconfig_data_format(CB_MIXES, CB_CONSTS);
    matmul_init(CB_MIXES, CB_CONSTS);
    tile_regs_acquire();
    matmul_tiles(CB_MIXES, CB_CONSTS, 0, SEL_COMB, 0);
    add_reuse_dest_init<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(CB_CONSTS);
    add_reuse_dest_tiles<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(CB_CONSTS, BASE_COMB, 0);
    tile_regs_commit();
    tile_regs_wait();
    CircularBuffer cb_log(CB_LOG);
    cb_log.reserve_back(1);
    pack_tile(0, CB_LOG);
    cb_log.push_back(1);
    tile_regs_release();
    cb_mixes.pop_front(1);

    // ---- Sinkhorn on the SFPU: x = exp(min(logits, 80)); the 4x4 normalisations are lane-wise vector arithmetic ----
    CircularBuffer cb_skin(CB_SKIN), cb_comb(CB_COMB);
    cb_skin.wait_front(1);
    reconfig_data_format_srca(CB_SKIN);
    copy_tile_to_dst_init_short(CB_SKIN);
    tile_regs_acquire();
    copy_tile(CB_SKIN, 0, 0);
    // Cap logits before exp so an out-of-range learned logit cannot overflow to inf (inf/inf = NaN in the softmax
    // divide).
    unary_min_tile_init();
    unary_min_tile(0, 0x42a00000u);  // min(x, 80.0f)
    exp_tile_init();
    exp_tile(0, VectorMode::R);  // faces 0/1 hold the 16 element vectors
    recip_tile_init();           // programmable constants of the Newton reciprocal
    MATH(
        (_llk_math_eltwise_unary_sfpu_params_(ckernel::sfpu::mhc_sinkhorn_sfpu, 0, VectorMode::None, iters, eps_bits)));
    tile_regs_commit();
    tile_regs_wait();
    cb_comb.reserve_back(1);
    pack_tile(0, CB_COMB);
    cb_comb.push_back(1);
    tile_regs_release();
    cb_skin.pop_front(1);
}
