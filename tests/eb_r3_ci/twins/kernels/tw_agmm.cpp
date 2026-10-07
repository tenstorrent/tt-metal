// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
// Round 3 eltwise binary twin of all_gather_minimal_matmul_async's fused addcmul epilogue (FUSE_TERNARY, bf16 gate):
// add_bias_and_addcmul_block verbatim (ttnn/cpp/ttnn/operations/experimental/ccl/all_gather_minimal_matmul_async/device/kernels/compute.cpp:136-339), called as kernel_main
// does after each output block; the matmul that fills intermediate_cb is stood in for by one copy per tile from src_cb.
// Compile args: M_block_tiles, N_block_tiles, iterations; common args: scalar bits, broadcast_ternary_b (as the op).

#include "api/compute/compute_kernel_api.h"
#include "api/compute/tilize.h"
#include "api/compute/matmul.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/bcast.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/sfpu_split_includes.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/dataflow/circular_buffer.h"

void add_bias_and_addcmul_block(
    CircularBuffer& intermediate_cb,
    CircularBuffer& bias_cb,
    CircularBuffer& ternary_a_cb,
    CircularBuffer& ternary_b_cb,
    uint32_t scalar_value,
    CircularBuffer& out_cb,
    uint32_t M_block_tiles,
    uint32_t N_block_tiles,
    uint32_t broadcast_ternary_b) {
    // Note: unary_bcast_tile does not work with fp32_acc_to_dest=True.
    // As a workaround, we perform addcmul through multiple LLKs calls (mul_tiles, mul_unary_tile, add_tiles_bcast).

    const uint32_t out_block_num_tiles = M_block_tiles * N_block_tiles;

    constexpr uint32_t DST_ID = 0;
#ifdef FUSE_BIAS
    // ============================================
    // STEP 1: Add bias block
    // Read from intermediate_cb and write back to intermediate_cb
    // ============================================

    add_bcast_rows_init(intermediate_cb.get_cb_id(), bias_cb.get_cb_id());
    reconfig_data_format(intermediate_cb.get_cb_id(), bias_cb.get_cb_id());
    pack_reconfig_data_format(intermediate_cb.get_cb_id());

    // Wait for ALL input data ONCE at the beginning
    bias_cb.wait_front(N_block_tiles);

    // Unpacker waits for intermediate_cb to be ready
    intermediate_cb.wait_front(out_block_num_tiles);

    for (uint32_t m = 0; m < M_block_tiles; m++) {
        for (uint32_t n = 0; n < N_block_tiles; n++) {
            uint32_t tile_id = m * N_block_tiles + n;

            tile_regs_acquire();
            add_tiles_bcast<BroadcastType::ROW>(intermediate_cb.get_cb_id(), bias_cb.get_cb_id(), tile_id, n, DST_ID);

            tile_regs_commit();
            tile_regs_wait();
            pack_tile(DST_ID, intermediate_cb.get_cb_id());
            tile_regs_release();
        }
    }

    // Pop input and push output ONCE at the end
    // intermediate_cb.wait_front(out_block_num_tiles); // Unpacker-Packer sync
    // intermediate_cb.pop_front(out_block_num_tiles);
    bias_cb.pop_front(N_block_tiles);

    intermediate_cb.pop_front(out_block_num_tiles);

    // Restore intermediate_cb to ready (+ sync packer/unpacker)
    intermediate_cb.reserve_back(out_block_num_tiles);
    intermediate_cb.push_back(out_block_num_tiles);
#endif  // FUSE_BIAS

    // ============================================
    // STEP 2: Multiply by ternary_b and scalar
    // Read from intermediate_cb and write back to intermediate_cb
    // broadcast_ternary_b: 1 = single row broadcast, 0 = row-by-row streaming
    // ============================================

    intermediate_cb.wait_front(out_block_num_tiles);

    uint32_t tile_id = 0;

    if (broadcast_ternary_b) {
        // === BROADCAST: single row, wait/pop once ===
        ternary_b_cb.wait_front(N_block_tiles);

#ifndef TERNARY_B_IS_FLOAT32
        mul_bcast_rows_init(intermediate_cb.get_cb_id(), ternary_b_cb.get_cb_id());
#else
        // Full re-arm (hw_configure + pack_dest/math_pack_sync), matching the pre-cleanup
        // 2-arg unary_bcast_init(ternary_b_cb, intermediate_cb); this runs after matmul_blocks
        // regardless of FUSE_BIAS, so a plain reconfig would drop the MATH<->PACK DST re-arm.
        // TODO(#52395): compute_kernel_hw_startup is a call-once API; this mid-kernel re-init (preserving the pre-cleanup full-init behaviour) should become a targeted DST re-arm.
        compute_kernel_hw_startup(ternary_b_cb.get_cb_id(), intermediate_cb.get_cb_id());
        unary_bcast_init<BroadcastType::ROW>(ternary_b_cb.get_cb_id());
#endif  // TERNARY_B_IS_FLOAT32

        binop_with_scalar_tile_init();
        reconfig_data_format(intermediate_cb.get_cb_id(), ternary_b_cb.get_cb_id());
        pack_reconfig_data_format(intermediate_cb.get_cb_id());

        tile_id = 0;
        for (uint32_t m = 0; m < M_block_tiles; m++) {
            for (uint32_t n = 0; n < N_block_tiles; n++) {
                tile_regs_acquire();

#ifndef TERNARY_B_IS_FLOAT32
                // LLK BUG: unary_bcast gives bad values if mixing fp32_acc_to_dest=True and bfloat16 circular buffer
                // (https://github.com/tenstorrent/tt-llk/issues/1338)
                // To avoid the bug, we use:
                // - unary_bcast/mul_binary_tile for fp32 (more accurate)
                // - mul_tiles_bcast for bfloat16 (LLK bug workaround).

                // ternary_b_cb is [1, N], broadcast across M rows
                mul_tiles_bcast<BroadcastType::ROW>(
                    intermediate_cb.get_cb_id(), ternary_b_cb.get_cb_id(), tile_id, n, DST_ID);
#else
                constexpr uint32_t TERNARY_B_DST_ID = 1;
                // TODO(#52395): compute_kernel_hw_startup is a call-once API; this mid-kernel re-init (preserving the pre-cleanup full-init behaviour) should become a targeted DST re-arm.
                compute_kernel_hw_startup(ternary_b_cb.get_cb_id(), intermediate_cb.get_cb_id());
                unary_bcast_init<BroadcastType::ROW>(ternary_b_cb.get_cb_id());
                unary_bcast<BroadcastType::ROW>(ternary_b_cb.get_cb_id(), n, TERNARY_B_DST_ID);

                copy_init(intermediate_cb.get_cb_id());
                copy_tile(intermediate_cb.get_cb_id(), tile_id, DST_ID);

                mul_binary_tile_init();
                mul_binary_tile(DST_ID, TERNARY_B_DST_ID, DST_ID);
#endif  // TERNARY_B_IS_FLOAT32

                mul_unary_tile(DST_ID, scalar_value);

                tile_regs_commit();
                tile_regs_wait();
                pack_tile(DST_ID, intermediate_cb.get_cb_id());
                tile_regs_release();
                tile_id++;
            }
        }

        ternary_b_cb.pop_front(N_block_tiles);
    } else {
        // === NO BROADCAST: row-by-row, wait/pop per M row ===
#ifndef TERNARY_B_IS_FLOAT32
        mul_init(intermediate_cb.get_cb_id(), ternary_b_cb.get_cb_id());
#endif
        binop_with_scalar_tile_init();
        reconfig_data_format(intermediate_cb.get_cb_id(), ternary_b_cb.get_cb_id());
        pack_reconfig_data_format(intermediate_cb.get_cb_id());

        tile_id = 0;
        for (uint32_t m = 0; m < M_block_tiles; m++) {
            ternary_b_cb.wait_front(N_block_tiles);
            for (uint32_t n = 0; n < N_block_tiles; n++) {
                tile_regs_acquire();

#ifndef TERNARY_B_IS_FLOAT32
                mul_tiles(intermediate_cb.get_cb_id(), ternary_b_cb.get_cb_id(), tile_id, n, DST_ID);
#else
                constexpr uint32_t TERNARY_B_DST_ID = 1;
                copy_init(ternary_b_cb.get_cb_id());
                copy_tile(ternary_b_cb.get_cb_id(), n, TERNARY_B_DST_ID);

                copy_init(intermediate_cb.get_cb_id());
                copy_tile(intermediate_cb.get_cb_id(), tile_id, DST_ID);

                mul_binary_tile_init();
                mul_binary_tile(DST_ID, TERNARY_B_DST_ID, DST_ID);
#endif  // TERNARY_B_IS_FLOAT32

                mul_unary_tile(DST_ID, scalar_value);

                tile_regs_commit();
                tile_regs_wait();
                pack_tile(DST_ID, intermediate_cb.get_cb_id());
                tile_regs_release();
                tile_id++;
            }
            ternary_b_cb.pop_front(N_block_tiles);
        }
    }

    intermediate_cb.pop_front(out_block_num_tiles);

    // 'refill' intermediate_cb (also synchronize packer/unpacker)
    intermediate_cb.reserve_back(out_block_num_tiles);
    intermediate_cb.push_back(out_block_num_tiles);

    intermediate_cb.wait_front(out_block_num_tiles);

    add_init(intermediate_cb.get_cb_id(), ternary_a_cb.get_cb_id());
    reconfig_data_format(intermediate_cb.get_cb_id(), ternary_a_cb.get_cb_id());
    pack_reconfig_data_format(out_cb.get_cb_id());

    tile_id = 0;
    for (uint32_t m = 0; m < M_block_tiles; m++) {
        // Wait for one row of ternary_a tiles
        ternary_a_cb.wait_front(N_block_tiles);

        for (uint32_t n = 0; n < N_block_tiles; n++) {
            tile_regs_acquire();

            // ternary_a_cb is pushed one row at a time, so use column index n
            add_tiles(intermediate_cb.get_cb_id(), ternary_a_cb.get_cb_id(), tile_id, n, DST_ID);

            tile_regs_commit();
            tile_regs_wait();
            pack_tile(DST_ID, out_cb.get_cb_id());
            tile_regs_release();
            tile_id++;
        }

        ternary_a_cb.pop_front(N_block_tiles);
        out_cb.push_back(N_block_tiles);
    }

    intermediate_cb.pop_front(out_block_num_tiles);
}

void kernel_main() {
    constexpr uint32_t M_block_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t N_block_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t twin_iters = get_compile_time_arg_val(2);
    const uint32_t fused_ternary_scalar_uint = get_common_arg_val<uint32_t>(0);
    const uint32_t broadcast_ternary_b = get_common_arg_val<uint32_t>(1);

    constexpr uint32_t src_cb_id = tt::CBIndex::c_0;
    constexpr uint32_t out_cb_id = tt::CBIndex::c_2;
    constexpr uint32_t intermediate_cb_id = tt::CBIndex::c_3;
    constexpr uint32_t in2_cb_id = tt::CBIndex::c_4;
    constexpr uint32_t ternary_a_cb_id = tt::CBIndex::c_5;
    constexpr uint32_t ternary_b_cb_id = tt::CBIndex::c_6;

    CircularBuffer src_cb(src_cb_id);
    CircularBuffer out_cb(out_cb_id);
    CircularBuffer intermediate_cb(intermediate_cb_id);
    CircularBuffer in2_cb(in2_cb_id);
    CircularBuffer ternary_a_cb(ternary_a_cb_id);
    CircularBuffer ternary_b_cb(ternary_b_cb_id);

    constexpr uint32_t out_block_num_tiles = M_block_tiles * N_block_tiles;

    compute_kernel_hw_startup<SrcOrder::Reverse>(src_cb_id, ternary_b_cb_id, intermediate_cb_id);

    for (uint32_t it = 0; it < twin_iters; ++it) {
        // Stand-in for matmul_blocks: the block's accumulator lands in intermediate_cb
        reconfig_data_format_srca(src_cb_id);
        pack_reconfig_data_format(intermediate_cb_id);
        copy_init(src_cb_id);
        src_cb.wait_front(out_block_num_tiles);
        intermediate_cb.reserve_back(out_block_num_tiles);
        for (uint32_t t = 0; t < out_block_num_tiles; t++) {
            tile_regs_acquire();
            copy_tile(src_cb_id, t, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, intermediate_cb_id);
            tile_regs_release();
        }
        src_cb.pop_front(out_block_num_tiles);
        intermediate_cb.push_back(out_block_num_tiles);

        out_cb.reserve_back(out_block_num_tiles);
        add_bias_and_addcmul_block(
            intermediate_cb,
            in2_cb,
            ternary_a_cb,
            ternary_b_cb,
            fused_ternary_scalar_uint,
            out_cb,
            M_block_tiles,
            N_block_tiles,
            broadcast_ternary_b);
    }
}
