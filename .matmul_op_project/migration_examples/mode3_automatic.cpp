// Migration Examples: Mode 3 (Automatic)
// Call sites: T1, B9, B10, B15
//
// These call sites use MatmulOp::run() for fully automatic blocked matmul.
// The caller sets up configuration, calls init(), then run() handles the
// entire batch/block/subblock loop with CB management and spill/reload.
//
// SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "api/compute/matmul_op.h"
#include "experimental/circular_buffer.h"

// ============================================================================
// T1: bmm.cpp -- simple tile-mode batched matmul
// Source: ttnn/cpp/ttnn/operations/matmul/device/kernels/compute/bmm.cpp
//
// ORIGINAL CODE (lines 34-61):
//   mm_init(cb_in0, cb_in1, cb_out);
//   for (batch) for (Mt) for (Nt) {
//       acquire_dst();
//       for (Kt) { wait, matmul_tiles(..., 0, 0, 0), pop }
//       pack_tile(0, cb_out); release_dst();
//   }
//
// MIGRATED: The entire nested loop is replaced by a single run() call.
// ============================================================================
namespace t1_bmm {

void kernel_main() {
    uint32_t batch = get_compile_time_arg_val(0);
    uint32_t Mt = get_compile_time_arg_val(1);
    uint32_t Kt = get_compile_time_arg_val(2);
    uint32_t Nt = get_compile_time_arg_val(3);

    constexpr uint32_t cb_in0 = get_named_compile_time_arg_val("cb_in0");
    constexpr uint32_t cb_in1 = get_named_compile_time_arg_val("cb_in1");
    constexpr uint32_t cb_out = get_named_compile_time_arg_val("cb_out");

    // --- NEW: MatmulOp replaces mm_init + nested loops ---
    ckernel::MatmulOpConfig cfg{
        .in0_cb_id = cb_in0,
        .in1_cb_id = cb_in1,
        .out_cb_id = cb_out,
    };
    ckernel::TileMatmulOp mm(cfg);
    mm.init();

    // run() handles: batch * Mt * Nt output tiles, each accumulating Kt inner tiles.
    // For tile mode, in0_num_subblocks, in1_num_subblocks, block tile counts, and
    // in1_block_w are all 1 (single-tile granularity).
    mm.run(
        batch,
        Mt,
        Nt,
        Kt,
        /*in0_num_subblocks=*/1,
        /*in1_num_subblocks=*/1,
        /*in0_block_num_tiles=*/1,
        /*in1_block_num_tiles=*/1,
        /*in1_block_w=*/1);
    // --- END NEW ---
}

}  // namespace t1_bmm

// ============================================================================
// B9: minimal_matmul -- standard block-mode matmul with subblocking
// Source: ttnn/.../experimental/minimal_matmul/device/kernels/compute.cpp
//
// ORIGINAL CODE: Uses a local matmul_blocks() helper function with the standard
//   subblock loop pattern (in0_subblocks * in1_subblocks, each doing K inner dim
//   accumulation via matmul_block, then pack with pack_tile<true> at computed
//   row-major positions).
//
// NOTE: The original uses pack_tile<true> with out-of-order indexing. Mode 3's
//   run() uses sequential pack_tile (not pack_tile<true>). This means B9 can
//   only use run() if the reader/writer layout matches sequential pack order.
//   For the standard minimal_matmul case this works because the matmul output
//   is consumed directly. If out-of-order pack is required, use Mode 1 instead.
//
// MIGRATED: The matmul_blocks() helper call is replaced by run().
// ============================================================================
namespace b9_minimal_matmul {

void kernel_main() {
    constexpr uint32_t K_num_blocks = get_compile_time_arg_val(0);
    constexpr uint32_t M_block_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t N_block_tiles = get_compile_time_arg_val(2);
    constexpr uint32_t K_block_tiles = get_compile_time_arg_val(3);
    constexpr uint32_t subblock_h = get_compile_time_arg_val(4);
    constexpr uint32_t subblock_w = get_compile_time_arg_val(5);

    constexpr uint32_t in0_cb = tt::CBIndex::c_0;
    constexpr uint32_t in1_cb = tt::CBIndex::c_1;
    constexpr uint32_t out_cb = tt::CBIndex::c_2;
    constexpr uint32_t partials_cb = tt::CBIndex::c_24;

    uint32_t in0_num_subblocks = M_block_tiles / subblock_h;
    uint32_t in1_num_subblocks = N_block_tiles / subblock_w;
    uint32_t in0_block_num_tiles = M_block_tiles * K_block_tiles;
    uint32_t in1_block_num_tiles = N_block_tiles * K_block_tiles;

    // --- NEW: MatmulOp replaces the local matmul_blocks() helper ---
    ckernel::MatmulOpConfig cfg{
        .in0_cb_id = in0_cb,
        .in1_cb_id = in1_cb,
        .out_cb_id = out_cb,
        .ct_dim = subblock_w,
        .rt_dim = subblock_h,
        .kt_dim = K_block_tiles,
        .partials_cb_id = (K_num_blocks > 1) ? partials_cb : 0u,
    };
    ckernel::BlockMatmulOp mm(cfg);
    mm.init();

    // run() handles: 1 batch, 1 block_h, 1 block_w, K_num_blocks inner blocks,
    // with subblocking and spill/reload when K_num_blocks > 1.
    mm.run(
        /*batch=*/1,
        /*num_blocks_h=*/1,
        /*num_blocks_w=*/1,
        /*num_blocks_inner=*/K_num_blocks,
        in0_num_subblocks,
        in1_num_subblocks,
        in0_block_num_tiles,
        in1_block_num_tiles,
        /*in1_block_w=*/N_block_tiles);
    // --- END NEW ---

    // NOTE: The original also has a ternary add phase after matmul (bias addition
    // via add_tiles). That phase is UNCHANGED -- it operates on the matmul output
    // CB after run() completes.
}

}  // namespace b9_minimal_matmul

// ============================================================================
// B10: conv3d -- standard block-mode matmul via matmul_blocks helper
// Source: ttnn/.../experimental/conv3d/device/kernels/compute.cpp
//
// ORIGINAL CODE: Uses a local matmul_blocks() function identical to B9's pattern
//   (subblock loop, matmul_block, sequential pack_tile). Called within a
//   K_num_blocks outer loop with spill/reload for K accumulation.
//
// MIGRATED: The outer K loop + matmul_blocks call is replaced by run().
// ============================================================================
namespace b10_conv3d {

void kernel_main() {
    constexpr uint32_t K_num_blocks = get_compile_time_arg_val(0);
    constexpr uint32_t M_block_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t N_block_tiles = get_compile_time_arg_val(2);
    constexpr uint32_t K_block_tiles = get_compile_time_arg_val(3);
    constexpr uint32_t subblock_h = get_compile_time_arg_val(4);
    constexpr uint32_t subblock_w = get_compile_time_arg_val(5);

    constexpr uint32_t in0_cb = tt::CBIndex::c_0;
    constexpr uint32_t in1_cb = tt::CBIndex::c_1;
    constexpr uint32_t out_cb = tt::CBIndex::c_2;
    constexpr uint32_t partials_cb = tt::CBIndex::c_24;

    uint32_t in0_num_subblocks = M_block_tiles / subblock_h;
    uint32_t in1_num_subblocks = N_block_tiles / subblock_w;
    uint32_t in0_block_num_tiles = M_block_tiles * K_block_tiles;
    uint32_t in1_block_num_tiles = N_block_tiles * K_block_tiles;

    // --- NEW: MatmulOp replaces the K-block loop + matmul_blocks helper ---
    ckernel::MatmulOpConfig cfg{
        .in0_cb_id = in0_cb,
        .in1_cb_id = in1_cb,
        .out_cb_id = out_cb,
        .ct_dim = subblock_w,
        .rt_dim = subblock_h,
        .kt_dim = K_block_tiles,
        .partials_cb_id = (K_num_blocks > 1) ? partials_cb : 0u,
    };
    ckernel::BlockMatmulOp mm(cfg);
    mm.init_short();  // conv3d uses init_short (called after prior compute ops)

    mm.run(
        /*batch=*/1,
        /*num_blocks_h=*/1,
        /*num_blocks_w=*/1,
        /*num_blocks_inner=*/K_num_blocks,
        in0_num_subblocks,
        in1_num_subblocks,
        in0_block_num_tiles,
        in1_block_num_tiles,
        /*in1_block_w=*/N_block_tiles);
    // --- END NEW ---

    // Post-matmul bias addition phase is UNCHANGED.
}

}  // namespace b10_conv3d

// ============================================================================
// B15: all_gather_minimal_matmul -- same pattern as B9 with L1_ACC
// Source: ttnn/.../ccl/all_gather_minimal_matmul_async/.../compute.cpp
//
// ORIGINAL CODE: Uses the same matmul_blocks() helper as B9. PACKER_L1_ACC is
//   used for K accumulation (orthogonal to MatmulOp).
//
// MIGRATED: Identical to B9 -- run() handles the matmul, PACKER_L1_ACC config
//   is managed by the caller around the run() call (before init, after run).
// ============================================================================
namespace b15_all_gather_minimal_matmul {

void kernel_main() {
    constexpr uint32_t K_num_blocks = get_compile_time_arg_val(0);
    constexpr uint32_t M_block_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t N_block_tiles = get_compile_time_arg_val(2);
    constexpr uint32_t K_block_tiles = get_compile_time_arg_val(3);
    constexpr uint32_t subblock_h = get_compile_time_arg_val(4);
    constexpr uint32_t subblock_w = get_compile_time_arg_val(5);

    constexpr uint32_t in0_cb = tt::CBIndex::c_0;
    constexpr uint32_t in1_cb = tt::CBIndex::c_1;
    constexpr uint32_t out_cb = tt::CBIndex::c_2;
    constexpr uint32_t partials_cb = tt::CBIndex::c_24;

    uint32_t in0_num_subblocks = M_block_tiles / subblock_h;
    uint32_t in1_num_subblocks = N_block_tiles / subblock_w;
    uint32_t in0_block_num_tiles = M_block_tiles * K_block_tiles;
    uint32_t in1_block_num_tiles = N_block_tiles * K_block_tiles;

    // --- UNCHANGED: PACKER_L1_ACC setup (orthogonal to MatmulOp) ---
#ifdef PACKER_L1_ACC
    PACK((pack_reconfig_l1_acc(1)));
#endif

    // --- NEW: MatmulOp replaces matmul_blocks() helper ---
    ckernel::MatmulOpConfig cfg{
        .in0_cb_id = in0_cb,
        .in1_cb_id = in1_cb,
        .out_cb_id = out_cb,
        .ct_dim = subblock_w,
        .rt_dim = subblock_h,
        .kt_dim = K_block_tiles,
        .partials_cb_id = (K_num_blocks > 1) ? partials_cb : 0u,
    };
    ckernel::BlockMatmulOp mm(cfg);
    mm.init_short();  // called after prior compute configuration

    mm.run(
        /*batch=*/1,
        /*num_blocks_h=*/1,
        /*num_blocks_w=*/1,
        /*num_blocks_inner=*/K_num_blocks,
        in0_num_subblocks,
        in1_num_subblocks,
        in0_block_num_tiles,
        in1_block_num_tiles,
        /*in1_block_w=*/N_block_tiles);
    // --- END NEW ---

#ifdef PACKER_L1_ACC
    PACK((pack_reconfig_l1_acc(0)));
#endif

    // Post-matmul ternary add phase is UNCHANGED.
}

}  // namespace b15_all_gather_minimal_matmul
