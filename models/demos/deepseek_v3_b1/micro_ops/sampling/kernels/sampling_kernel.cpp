// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// Top-K sampling unified kernel (k>1 path) wrapper.
// Logic is in unified_kernels/sampling.hpp; this file wires up
// compile-time args and dispatches to the TopKSampling op.
// After the top-K merge, NCRISC packs the global top-K scores into a tile
// for TRISC softmax, then reads back the probabilities.

#include "../../../unified_kernels/kernel_op_api.hpp"
#include "../../../unified_kernels/sampling.hpp"
#if defined(COMPILE_FOR_NCRISC) || defined(COMPILE_FOR_BRISC)
#include "api/dataflow/dataflow_api.h"
#endif

#if defined(COMPILE_FOR_TRISC)
#define REDUCE_OP (PoolType::SUM)
#define REDUCE_DIM (ReduceDim::REDUCE_ROW)
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/exp.h"
#include "api/compute/eltwise_unary/recip.h"
#include "api/compute/reduce.h"
#include "api/compute/bcast.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/pack.h"

// ---------------------------------------------------------------------------
// TRISC softmax helpers (adapted from ttnn sampling compute kernel)
// All operate on tile-granularity CBs with Ht rows and Kt column tiles.
// For our use: Ht=1, Kt=1 (32 values in row 0 of a single tile).
// ---------------------------------------------------------------------------

template <uint32_t in0_cb, uint32_t in1_cb, uint32_t rows, uint32_t cols>
void sub_exp_block_bcast_cols_inplace() {
    sub_bcast_cols_init_short(in0_cb, in1_cb);
    exp_tile_init<true>();
    cb_wait_front(in0_cb, rows * cols);
    cb_wait_front(in1_cb, rows);

    for (uint32_t i = 0; i < rows; ++i) {
        for (uint32_t u = 0; u < cols; ++u) {
            tile_regs_acquire();
            sub_tiles_bcast_cols(in0_cb, in1_cb, 0, i, 0);
            exp_tile<true>(0);
            tile_regs_commit();
            cb_pop_front(in0_cb, 1);
            cb_reserve_back(in0_cb, 1);
            tile_regs_wait();
            pack_tile(0, in0_cb);
            cb_push_back(in0_cb, 1);
            tile_regs_release();
        }
    }
}

template <
    PoolType pool_type,
    ReduceDim reduce_dim,
    uint32_t in0_cb,
    uint32_t scale_cb,
    uint32_t out_cb,
    uint32_t rows,
    uint32_t cols>
void reduce_c() {
    reconfig_data_format(in0_cb, scale_cb);
    reduce_init<pool_type, reduce_dim>(in0_cb, scale_cb, out_cb);

    const uint32_t num_tiles = rows * cols;
    cb_wait_front(scale_cb, 1);
    cb_wait_front(in0_cb, num_tiles);
    cb_reserve_back(out_cb, rows);

    for (uint32_t i = 0; i < rows; i++) {
        acquire_dst();
        for (uint32_t j = 0; j < cols; j++) {
            reduce_tile<pool_type, reduce_dim>(in0_cb, scale_cb, i * cols + j, 0, 0);
        }
        cb_reserve_back(out_cb, 1);
        pack_reconfig_data_format(out_cb);
        pack_tile(0, out_cb);
        cb_push_back(out_cb, 1);
        release_dst();
    }
    reduce_uninit();
    UNPACK(tensix_sync());
}

void recip_block_inplace(uint32_t in_cb, uint32_t num_tiles) {
    copy_tile_to_dst_init_short(in_cb);
    recip_tile_init();
    cb_wait_front(in_cb, num_tiles);
    for (uint32_t i = 0; i < num_tiles; ++i) {
        acquire_dst();
        copy_tile(in_cb, 0, 0);
        cb_pop_front(in_cb, 1);
        recip_tile(0);
        cb_reserve_back(in_cb, 1);
        pack_tile(0, in_cb);
        cb_push_back(in_cb, 1);
        release_dst();
    }
}

void mul_block_bcast_cols(uint32_t in0_cb, uint32_t in1_cb, uint32_t out_cb, uint32_t rows, uint32_t cols) {
    uint32_t num_tiles = rows * cols;
    mul_bcast_cols_init_short(in0_cb, in1_cb);
    cb_wait_front(in0_cb, num_tiles);
    cb_wait_front(in1_cb, rows);
    for (uint32_t i = 0; i < rows; ++i) {
        for (uint32_t j = 0; j < cols; ++j) {
            acquire_dst();
            mul_tiles_bcast_cols(in0_cb, in1_cb, 0, i, 0);
            cb_pop_front(in0_cb, 1);
            cb_reserve_back(out_cb, 1);
            pack_tile(0, out_cb);
            cb_push_back(out_cb, 1);
            release_dst();
        }
    }
    cb_pop_front(in1_cb, rows);
}

#endif  // COMPILE_FOR_TRISC

struct Core {
    static constexpr bool is_active_core = get_named_compile_time_arg_val("sampling_is_active_core") == 1;
    static constexpr bool is_final_core = get_named_compile_time_arg_val("sampling_is_final_core") == 1;
    static constexpr bool is_mesh_sender_core = get_named_compile_time_arg_val("sampling_mesh_sender_core") == 1;
};

void kernel_main() {
#if defined(COMPILE_FOR_NCRISC)
    using SamplingReaderCTArgs = deepseek_b1_ops::TopKSampling::ReaderCTArgs<
        get_named_compile_time_arg_val("sampling_num_values"),
        get_named_compile_time_arg_val("sampling_topk_k"),
        get_named_compile_time_arg_val("sampling_winner_page_bytes"),
        get_named_compile_time_arg_val("sampling_num_senders"),
        get_named_compile_time_arg_val("sampling_expected_remote_incs"),
        get_named_compile_time_arg_val("sampling_receiver_semaphore_id"),
        get_named_compile_time_arg_val("sampling_local_ready_semaphore_id"),
        get_named_compile_time_arg_val("sampling_mesh_mode"),
        get_named_compile_time_arg_val("sampling_stage1_sender"),
        get_named_compile_time_arg_val("sampling_stage1_receiver"),
        get_named_compile_time_arg_val("sampling_stage2_sender"),
        get_named_compile_time_arg_val("sampling_stage2_receiver"),
        get_named_compile_time_arg_val("sampling_stage1_slot_base_offset"),
        get_named_compile_time_arg_val("sampling_stage1_num_slots"),
        get_named_compile_time_arg_val("sampling_stage1_expected_remote_incs"),
        get_named_compile_time_arg_val("sampling_stage1_local_slot_offset"),
        get_named_compile_time_arg_val("sampling_stage2_slot_base_offset"),
        get_named_compile_time_arg_val("sampling_stage2_num_slots"),
        get_named_compile_time_arg_val("sampling_stage2_expected_remote_incs"),
        get_named_compile_time_arg_val("sampling_stage2_local_slot_offset"),
        get_named_compile_time_arg_val("sampling_mesh_local_send_slot_offset"),
        get_named_compile_time_arg_val("sampling_sender_idx"),
        0,
        0,
        0,
        0xFFFFFFFF,
        0,
        get_named_compile_time_arg_val("sampling_gather_cb"),
        get_named_compile_time_arg_val("sampling_winner_cb")>;

    deepseek_b1_ops::TopKSampling::ReaderArgs args{
        .scores_addr = get_common_arg_val<uint32_t>(0),
        .indices_addr = get_common_arg_val<uint32_t>(1),
        .output_addr = get_common_arg_val<uint32_t>(2),
        .final_noc_x = get_common_arg_val<uint32_t>(3),
        .final_noc_y = get_common_arg_val<uint32_t>(4),
        .scratch_addr = get_common_arg_val<uint32_t>(5),
        .global_sem_addr = get_common_arg_val<uint32_t>(6),
        .global_stage2_sem_addr = get_common_arg_val<uint32_t>(7),
        .gather_addr = 0,
    };

    deepseek_b1_ops::TopKSampling::
        Op<SamplingReaderCTArgs, Core::is_active_core, Core::is_final_core, Core::is_mesh_sender_core>
            sampling_op;
    sampling_op(args);

    // ---- Post-Op: softmax tile packing (final core only) ----
    if constexpr (Core::is_final_core) {
        constexpr uint32_t K = SamplingReaderCTArgs::topk_k;
        constexpr uint32_t softmax_in_cb = get_named_compile_time_arg_val("sampling_softmax_in_cb");
        constexpr uint32_t softmax_out_cb = get_named_compile_time_arg_val("sampling_softmax_out_cb");
        constexpr uint32_t scaler_cb = get_named_compile_time_arg_val("sampling_scaler_cb");
        constexpr uint32_t FACE_ELEMS = 256;  // 16*16 bf16 elements per face = 512 bytes / 2
        constexpr uint16_t BF16_ONE = 0x3F80;

        // Winner CB has: [K scores (topk_scores_stride bytes)] [K indices ...]
        const uint32_t winner_addr = get_write_ptr(SamplingReaderCTArgs::winner_cb_id);
        auto global_scores = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(winner_addr);

        // DPRINT top-32 scores before softmax
        DPRINT << "Top-" << K << " scores (before softmax):" << ENDL();
        for (uint32_t i = 0; i < K; ++i) {
            DPRINT << "  [" << i << "] " << BF16(global_scores[i]) << ENDL();
        }

        // Fill scaler CB with bf16 1.0 tile
        cb_reserve_back(scaler_cb, 1);
        auto scaler_ptr = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(get_write_ptr(scaler_cb));
        for (uint32_t i = 0; i < 1024; ++i) {
            scaler_ptr[i] = BF16_ONE;
        }
        cb_push_back(scaler_cb, 1);

        // Pack top-K scores into a bf16 tile in softmax_in_cb.
        // Tile face layout: face0 = rows[0..15] cols[0..15], face1 = rows[0..15] cols[16..31]
        // Row 0 scores[0..15] → face0[0..15], scores[16..31] → face1[0..15]
        // Everything else zero-padded.
        cb_reserve_back(softmax_in_cb, 1);
        auto tile_u32 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(softmax_in_cb));
        for (uint32_t i = 0; i < 512; ++i) {
            tile_u32[i] = 0;
        }
        auto tile_u16 = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(get_write_ptr(softmax_in_cb));
        for (uint32_t i = 0; i < 16 && i < K; ++i) {
            tile_u16[i] = global_scores[i];
        }
        for (uint32_t i = 0; i < 16 && (i + 16) < K; ++i) {
            tile_u16[FACE_ELEMS + i] = global_scores[16 + i];
        }
        cb_push_back(softmax_in_cb, 1);

        // Wait for TRISC softmax to produce output
        cb_wait_front(softmax_out_cb, 1);

        // Read softmax probabilities and DPRINT
        auto out_u16 = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(get_read_ptr(softmax_out_cb));
        DPRINT << "Top-" << K << " probs (after softmax):" << ENDL();
        for (uint32_t i = 0; i < K; ++i) {
            uint16_t prob_bf16 = (i < 16) ? out_u16[i] : out_u16[FACE_ELEMS + (i - 16)];
            DPRINT << "  [" << i << "] " << BF16(prob_bf16) << ENDL();
        }
        cb_pop_front(softmax_out_cb, 1);
    }

#elif defined(COMPILE_FOR_BRISC)
    using SamplingWriterCTArgs = deepseek_b1_ops::TopKSampling::WriterCTArgs<
        get_named_compile_time_arg_val("sampling_winner_page_bytes"),
        get_named_compile_time_arg_val("sampling_local_ready_semaphore_id"),
        0,
        0,
        0>;

    deepseek_b1_ops::TopKSampling::WriterArgs args{
        get_common_arg_val<uint32_t>(0),
        get_common_arg_val<uint32_t>(1),
        get_common_arg_val<uint32_t>(2),
        0,
    };

    deepseek_b1_ops::TopKSampling::
        Op<SamplingWriterCTArgs, Core::is_active_core, Core::is_final_core, Core::is_mesh_sender_core>
            sampling_op;
    sampling_op(args);
#elif defined(COMPILE_FOR_TRISC)
    if constexpr (Core::is_final_core) {
        constexpr uint32_t softmax_in_cb = get_named_compile_time_arg_val("sampling_softmax_in_cb");
        constexpr uint32_t softmax_out_cb = get_named_compile_time_arg_val("sampling_softmax_out_cb");
        constexpr uint32_t max_cb = get_named_compile_time_arg_val("sampling_max_cb");
        constexpr uint32_t sum_cb = get_named_compile_time_arg_val("sampling_sum_cb");
        constexpr uint32_t scaler_cb = get_named_compile_time_arg_val("sampling_scaler_cb");

        // Softmax over a single tile (Ht=1, Kt=1):
        // 1. max-reduce across columns
        reduce_c<PoolType::MAX, ReduceDim::REDUCE_ROW, softmax_in_cb, scaler_cb, max_cb, 1, 1>();
        // 2. subtract max (broadcast) and exp
        sub_exp_block_bcast_cols_inplace<softmax_in_cb, max_cb, 1, 1>();
        // 3. sum-reduce across columns
        reduce_c<PoolType::SUM, ReduceDim::REDUCE_ROW, softmax_in_cb, scaler_cb, sum_cb, 1, 1>();
        // 4. reciprocal of sum
        recip_block_inplace(sum_cb, 1);
        // 5. multiply exp values by 1/sum → probabilities
        mul_block_bcast_cols(softmax_in_cb, sum_cb, softmax_out_cb, 1, 1);
    }
#endif
}
