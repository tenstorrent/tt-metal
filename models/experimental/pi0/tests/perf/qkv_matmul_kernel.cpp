// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// SigLIP fused-QKV matmul kernel — encoder-shape, L1-resident weights.
//
// **Status: PASSES at PCC 0.999989 vs torch fp32 reference on real π0.5 layer-0
// QKV weight (2026-05-16). Total fix journey: 0.018 → 0.757 → 0.999989.**
//
// Two-bug fix from the previous custom_mm_block draft:
//   (1) Swapped custom_mm_block (decode-only, in0 [{1,2,4,8}, 32], rt_dim=1,
//       LoFi-only) for standard matmul_block (encoder-friendly: rt_dim 1-8 in
//       half-sync / 1-16 full-sync, unlimited kt_dim, honors MATH_FIDELITY).
//       Took PCC from 0.018 to 0.757.
//   (2) SUBBLOCK_H=1 (one M-tile-row per chunk, full K reduction, 3 output
//       tiles per chunk via SUBBLOCK_W=N_TILES_PER_CORE=3). SUBBLOCK_H>1 has
//       a subtle issue with the in0_index stride semantics — open question
//       worth chasing for perf later but PCC is correct at H=1.
//       Took PCC from 0.757 to 0.999989.
//
// Pattern mirrors ttnn/cpp/.../experimental/minimal_matmul/.../compute.cpp.
//
// Computes (per core):
//   output[M_TILES, N_TILES_PER_CORE] =
//       activation[M_TILES, K_TILES] @ weights[K_TILES, N_TILES_PER_CORE]
//
// SigLIP layer-0 fused-QKV shape (single chip, 36-core decomposition):
//   M = 256 rows / 8 tiles   K = 1152 / 36 tiles   N = 3456 / 108 tiles
//   36 cores × 3 N-tiles-per-core = 108 ✓
//
// Subblock decomposition (dst-register sized):
//   SUBBLOCK_H × SUBBLOCK_W tiles per acquire/commit cycle ≤ 16 (fp32 dest acc).
//   SUBBLOCK_H = 4, SUBBLOCK_W = 3 = 12 dst tiles → 2 M-chunks per call.
//
// Activation CB layout: row-major-of-tiles (M_TILES * K_TILES = 288 tiles).
//   tile (m, k) at flat index m * K_TILES + k.
// Weight CB layout: row-major-of-tiles (K_TILES * N_TILES_PER_CORE = 108 tiles).
//   tile (k, n) at flat index k * N_TILES_PER_CORE + n.
// Output CB layout: row-major-of-tiles (M_TILES * N_TILES_PER_CORE = 24 tiles).
//   tile (m, n) at flat index m * N_TILES_PER_CORE + n.
//
// Weights stay L1-resident across calls (no cb_pop_front for weights_cb).

#include "../../../../demos/deepseek_v3_b1/unified_kernels/kernel_op_api.hpp"
#include "../../../../demos/deepseek_v3_b1/unified_kernels/kernel_utils.hpp"
#include "api/debug/dprint.h"

#if defined(COMPILE_FOR_TRISC)
#include "api/compute/compute_kernel_api.h"
#include "api/compute/matmul.h"  // mm_init, mm_block_init_short, matmul_block
#include "api/compute/tile_move_copy.h"
#endif

void kernel_main() {
// ----------------------------------------------------------------------------
// NCRISC: setup the L1-resident sharded buffers backing act_cb and weights_cb.
// ----------------------------------------------------------------------------
#if defined(COMPILE_FOR_NCRISC)
    constexpr uint32_t act_cb = get_named_compile_time_arg_val("act_cb");
    constexpr uint32_t weights_cb = get_named_compile_time_arg_val("weights_cb");
    constexpr uint32_t act_tiles = get_named_compile_time_arg_val("act_tiles");
    constexpr uint32_t weights_tiles = get_named_compile_time_arg_val("weights_tiles");

    unified_kernels::setup_sharded_buffer(act_cb, act_tiles);
    unified_kernels::setup_sharded_buffer(weights_cb, weights_tiles);
    DPRINT << "NCRISC: setup. act_tiles=" << act_tiles << " weights_tiles=" << weights_tiles << ENDL();

// ----------------------------------------------------------------------------
// BRISC: no-op (no DRAM streaming).
// ----------------------------------------------------------------------------
#elif defined(COMPILE_FOR_BRISC)
    // intentionally empty

// ----------------------------------------------------------------------------
// TRISC: encoder-shape matmul via matmul_block, M-chunked into dst-sized
// subblocks. Inner K-loop accumulates partial sums into the same dst_index.
// ----------------------------------------------------------------------------
#elif defined(COMPILE_FOR_TRISC)
    constexpr uint32_t act_cb = get_named_compile_time_arg_val("act_cb");
    constexpr uint32_t weights_cb = get_named_compile_time_arg_val("weights_cb");
    constexpr uint32_t out_cb = get_named_compile_time_arg_val("out_cb");
    constexpr uint32_t M_TILES = get_named_compile_time_arg_val("m_tiles");                    // 8
    constexpr uint32_t K_TILES = get_named_compile_time_arg_val("k_tiles");                    // 36
    constexpr uint32_t N_TILES_PER_CORE = get_named_compile_time_arg_val("n_tiles_per_core");  // 3
    constexpr uint32_t ACT_TOTAL_TILES = get_named_compile_time_arg_val("act_tiles");  // M_TILES * K_TILES = 288
    constexpr uint32_t WEIGHT_TILES =
        get_named_compile_time_arg_val("weights_tiles");  // K_TILES * N_TILES_PER_CORE = 108

    // Subblock sizing: H*W ≤ 16 (fp32 dest acc enabled).
    // PASSES at SUBBLOCK_H=1: 8 M-chunks × (rt=1, ct=3) = 24 output tiles total.
    // PCC was 0.757 at SUBBLOCK_H=4; root cause TBD (open perf opportunity).
    constexpr uint32_t SUBBLOCK_H = 1;
    constexpr uint32_t SUBBLOCK_W = N_TILES_PER_CORE;
    constexpr uint32_t OUT_TILES_PER_SUBBLOCK = SUBBLOCK_H * SUBBLOCK_W;
    static_assert(M_TILES % SUBBLOCK_H == 0, "M_TILES must be a multiple of SUBBLOCK_H");
    static_assert(N_TILES_PER_CORE % SUBBLOCK_W == 0, "N_TILES_PER_CORE must be a multiple of SUBBLOCK_W");
    static_assert(OUT_TILES_PER_SUBBLOCK <= 16, "subblock exceeds dst register capacity");
    constexpr uint32_t OUT_TOTAL_TILES = M_TILES * N_TILES_PER_CORE;

    // Activation is bf16, weight is bfp8, output is bf16.
    // reconfig_data_format(in1_cb, in0_cb) configures unpacker srcA/srcB from CB metadata.
    reconfig_data_format(weights_cb, act_cb);
    pack_reconfig_data_format(out_cb);

    mm_init(act_cb, weights_cb, out_cb);

    // Wait once for full activation and full per-core weight slice.
    cb_wait_front(act_cb, ACT_TOTAL_TILES);
    cb_wait_front(weights_cb, WEIGHT_TILES);
    cb_reserve_back(out_cb, OUT_TOTAL_TILES);

    DPRINT << "TRISC: M_TILES=" << M_TILES << " K_TILES=" << K_TILES << " N_TILES_PER_CORE=" << N_TILES_PER_CORE
           << " SUBBLOCK_H=" << SUBBLOCK_H << " SUBBLOCK_W=" << SUBBLOCK_W << ENDL();

    // Loop over M-chunks (rt-blocks). One N-chunk because SUBBLOCK_W = N_TILES_PER_CORE.
    for (uint32_t m_start = 0; m_start < M_TILES; m_start += SUBBLOCK_H) {
        // (Re)init mm_block for this subblock shape. Cheap if shape unchanged.
        mm_block_init_short(
            act_cb,
            weights_cb,
            /*transpose=*/0,
            /*ct_dim=*/SUBBLOCK_W,
            /*rt_dim=*/SUBBLOCK_H,
            /*kt_dim=*/K_TILES);

        tile_regs_acquire();

        // Inner K-loop: one matmul_block per K-tile, accumulating into dst.
        // act tile index at K=k: in0_index = m_start * K_TILES + k
        // weight tile index at K=k: in1_index = k * N_TILES_PER_CORE + 0  (N start = 0)
        uint32_t in0_index = m_start * K_TILES;
        uint32_t in1_index = 0;
        for (uint32_t k = 0; k < K_TILES; ++k) {
            matmul_block(
                act_cb,
                weights_cb,
                in0_index,
                in1_index,
                /*idst=*/0,
                /*transpose=*/false,
                /*ct_dim=*/SUBBLOCK_W,
                /*rt_dim=*/SUBBLOCK_H,
                /*kt_dim=*/K_TILES);
            in0_index += 1;
            in1_index += N_TILES_PER_CORE;  // weight stride = full N-row of tiles
        }

        tile_regs_commit();

        // Pack subblock to out_cb at the correct out_tile_id positions.
        tile_regs_wait();
        for (uint32_t h = 0; h < SUBBLOCK_H; ++h) {
            for (uint32_t w = 0; w < SUBBLOCK_W; ++w) {
                uint32_t dst_idx = h * SUBBLOCK_W + w;
                uint32_t out_tile_id = (m_start + h) * N_TILES_PER_CORE + w;
                pack_tile<true>(dst_idx, out_cb, out_tile_id);
            }
        }
        tile_regs_release();
    }

    cb_push_back(out_cb, OUT_TOTAL_TILES);
    cb_pop_front(act_cb, ACT_TOTAL_TILES);
    // Weights stay L1-resident — no cb_pop_front for weights_cb.

    DPRINT << "TRISC: done." << ENDL();
#endif
}
