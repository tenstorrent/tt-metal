// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "dataflow_utils.hpp"
#include "kernel_op_api.hpp"
#include "kernel_utils.hpp"

#if defined(COMPILE_FOR_BRISC) || defined(COMPILE_FOR_NCRISC)
#include "api/dataflow/dataflow_api.h"
#endif

#if defined(COMPILE_FOR_TRISC)
#include "api/compute/tilize.h"
#include "../kernel_includes/tt_metal/include/compute_kernel_api/custom_tilize.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/pack_untilize.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#endif

namespace deepseek_b1_ops {

// ============================================================================
// Distributed Create Q Heads micro-op
//
// Splits tilize work across three 4x2 core sets:
//   - Original SDPA input cores: NOPE phase 1 [8, 256] -> output tiles 0..7
//   - NOPE helper cores: NOPE phase 2 [8, 256] -> output tiles 8..15
//   - ROPE helper cores: ROPE [8, 64] -> output tiles 16..17
//
// Helper cores write their tilized output directly into the original cores'
// FlashMLA Q CB backing storage. Original cores remain the only owners of the
// destination CB metadata and publish the full 18-tile Q shard.
// ============================================================================
struct DistributedCreateQHeads {
    template <uint32_t qnope_data_size_bytes_, uint32_t qrope_head_size_bytes_>
    struct SenderCTArgs {
        static constexpr uint32_t qnope_data_size_bytes = qnope_data_size_bytes_;
        static constexpr uint32_t qrope_head_size_bytes = qrope_head_size_bytes_;
    };
    struct ReceiverCTArgs {};
    struct ComputeCTArgs {};

    // Sender data-movement args (NCRISC on QNOPE/QROPE cores).
    struct SenderArgs {
        uint32_t sender_grid_start_x;
        uint32_t sender_grid_start_y;
        uint32_t qnope_cols;
        uint32_t qnope_cb;
        uint32_t qrope_cb;
        uint32_t src_num_pages;
        // Remote semaphore addresses on receiver/helper cores.
        uint32_t nope_phase1_semaphore_addr;
        uint32_t nope_phase2_semaphore_addr;
        uint32_t rope_semaphore_addr;
        // Per-sender-row destination NOC coords (packed: lo16=x, hi16=y).
        uint32_t original_noc_coords[8];
        uint32_t nope_helper_noc_coords[8];
        uint32_t rope_helper_noc_coords[8];
        uint32_t receiver_data_addr;
    };

    // Receiver data-movement args, shared by original receivers and NOPE/ROPE
    // helpers. Helpers also need ``original_noc_coords`` so they can
    // NOC-write their tilized output back to the original receiver's L1.
    struct ReceiverArgs {
        // Local semaphore addresses (also the remote sem addresses helpers
        // signal back to on the original).
        uint32_t nope_phase1_semaphore_addr;
        uint32_t nope_phase2_semaphore_addr;
        uint32_t rope_semaphore_addr;
        // Per-receiver-row original-receiver NOC coords (helpers write back).
        uint32_t original_noc_coords[8];
        uint32_t receiver_in_cb;
        uint32_t out_cb;
        uint32_t nope_tiles;
        uint32_t rope_tiles;
        uint32_t num_nope_senders;
        uint32_t num_rope_senders;
    };

    struct ComputeArgs {
        uint32_t receiver_in_cb;
        uint32_t out_cb;
        uint32_t nope_tiles;
        uint32_t rope_tiles;
    };

    template <
        typename CTArgs,
        bool IsSenderCore,
        bool IsOriginalReceiverCore,
        bool IsNopeHelperCore,
        bool IsRopeHelperCore,
        bool shared_dm_risc,
        bool setup_sharded_input,
        bool pop_src>
    class Op {
    public:
        void operator()([[maybe_unused]] const SenderArgs& args) {
#if defined(COMPILE_FOR_NCRISC) || defined(COMPILE_FOR_BRISC)
            if constexpr (IsSenderCore) {
                sender_impl<shared_dm_risc && (IsOriginalReceiverCore || IsNopeHelperCore || IsRopeHelperCore)>(args);
            }
#endif
        }

        void operator()([[maybe_unused]] const ReceiverArgs& args) {
#if defined(COMPILE_FOR_NCRISC) || defined(COMPILE_FOR_BRISC)
            if constexpr (IsOriginalReceiverCore) {
                original_receiver_impl<shared_dm_risc && IsSenderCore>(args);
            }
            if constexpr (IsNopeHelperCore) {
                nope_helper_receiver_impl<shared_dm_risc && IsSenderCore>(args);
            }
            if constexpr (IsRopeHelperCore) {
                rope_helper_receiver_impl<shared_dm_risc && IsSenderCore>(args);
            }
#endif
        }

        void operator()([[maybe_unused]] const ComputeArgs& args) {
#if defined(COMPILE_FOR_TRISC)
            // Originals and NOPE helpers tilize the same NOPE half (8 tiles split
            // into 2 chunks); ROPE helpers tilize ROPE tiles one-tile per chunk.
            if constexpr (IsOriginalReceiverCore || IsNopeHelperCore) {
                constexpr uint32_t nope_num_chunks = 2;
                tilize_phase_impl(
                    args.receiver_in_cb,
                    args.out_cb,
                    args.nope_tiles,
                    nope_num_chunks,
                    args.nope_tiles / nope_num_chunks);
            }
            if constexpr (IsRopeHelperCore) {
                tilize_phase_impl(args.receiver_in_cb, args.out_cb, args.rope_tiles, args.rope_tiles, 1);
            }
#endif
        }

    private:
#if defined(COMPILE_FOR_NCRISC) || defined(COMPILE_FOR_BRISC)
        static constexpr uint8_t WRITE_NOC = 0;

        static FORCE_INLINE uint32_t unpack_noc_x(uint32_t packed_coords) { return packed_coords & 0xFFFF; }

        static FORCE_INLINE uint32_t unpack_noc_y(uint32_t packed_coords) { return (packed_coords >> 16) & 0xFFFF; }

        template <bool is_receiver_same_risc>
        FORCE_INLINE void sender_impl(const SenderArgs& args) {
            static_assert(
                WRITE_NOC == noc_index || noc_mode == DM_DYNAMIC_NOC,
                "WRITE_NOC differs from noc_index; must be in dynamic NOC mode");

            uint32_t my_col = my_logical_x_ - args.sender_grid_start_x;
            uint32_t my_row = my_logical_y_ - args.sender_grid_start_y;
            bool is_qnope_core = my_col < args.qnope_cols;
            uint32_t src_cb = is_qnope_core ? args.qnope_cb : args.qrope_cb;

            uint32_t original_packed = args.original_noc_coords[my_row];
            uint32_t nope_helper_packed = args.nope_helper_noc_coords[my_row];
            uint32_t rope_helper_packed = args.rope_helper_noc_coords[my_row];
            const uint64_t original_noc_coord =
                get_noc_addr(unpack_noc_x(original_packed), unpack_noc_y(original_packed), 0, WRITE_NOC);
            const uint64_t nope_helper_noc_coord =
                get_noc_addr(unpack_noc_x(nope_helper_packed), unpack_noc_y(nope_helper_packed), 0, WRITE_NOC);
            const uint64_t rope_helper_noc_coord =
                get_noc_addr(unpack_noc_x(rope_helper_packed), unpack_noc_y(rope_helper_packed), 0, WRITE_NOC);

            constexpr uint32_t half_qnope_data_size_bytes = CTArgs::qnope_data_size_bytes / 2;

            if constexpr (setup_sharded_input) {
                unified_kernels::setup_sharded_buffer(src_cb, args.src_num_pages);
            }

            cb_wait_front(src_cb, args.src_num_pages);
            uint32_t src_addr = get_read_ptr(src_cb);

            if (is_qnope_core) {
                uint32_t dst_offset = my_col * half_qnope_data_size_bytes;
                uint64_t dst_data_noc_addr_0 = original_noc_coord | (uint64_t)(args.receiver_data_addr + dst_offset);
                noc_async_write<half_qnope_data_size_bytes, true, /*posted=*/true>(
                    src_addr, dst_data_noc_addr_0, half_qnope_data_size_bytes, WRITE_NOC);
                noc_semaphore_inc(original_noc_coord | (uint64_t)args.nope_phase1_semaphore_addr, 1, WRITE_NOC);

                uint64_t dst_data_noc_addr_1 = nope_helper_noc_coord | (uint64_t)(args.receiver_data_addr + dst_offset);
                noc_async_write<half_qnope_data_size_bytes, true, /*posted=*/true>(
                    src_addr + half_qnope_data_size_bytes, dst_data_noc_addr_1, half_qnope_data_size_bytes, WRITE_NOC);
                noc_semaphore_inc(nope_helper_noc_coord | (uint64_t)args.nope_phase2_semaphore_addr, 1, WRITE_NOC);
            } else {
                uint32_t qrope_col = my_col - args.qnope_cols;
                uint32_t dst_offset = 2 * qrope_col * CTArgs::qrope_head_size_bytes;
                uint64_t dst_data_noc_addr = rope_helper_noc_coord | (uint64_t)(args.receiver_data_addr + dst_offset);
                constexpr uint32_t double_qrope_head_size_bytes = CTArgs::qrope_head_size_bytes * 2;
                noc_async_write<double_qrope_head_size_bytes, true, /*posted=*/true>(
                    src_addr, dst_data_noc_addr, double_qrope_head_size_bytes, WRITE_NOC);
                noc_semaphore_inc(rope_helper_noc_coord | (uint64_t)args.rope_semaphore_addr, 1, WRITE_NOC);
            }

            if constexpr (!is_receiver_same_risc) {
                noc_async_atomic_barrier(WRITE_NOC);
            }
            if constexpr (pop_src) {
                if constexpr (is_receiver_same_risc) {
                    noc_async_writes_flushed(WRITE_NOC);
                }
                cb_pop_front(src_cb, args.src_num_pages);
            }
        }

        FORCE_INLINE uint32_t nope_helper_receiver_row() { return (my_logical_y_ - 5) * 4 + my_logical_x_; }

        FORCE_INLINE uint32_t rope_helper_receiver_row() { return my_logical_y_; }

        FORCE_INLINE uint64_t original_out_noc_addr(const ReceiverArgs& args, uint32_t row, uint32_t tile_offset) {
            uint32_t original_packed = args.original_noc_coords[row];
            return get_noc_addr(
                unpack_noc_x(original_packed),
                unpack_noc_y(original_packed),
                get_read_ptr(args.out_cb) + tile_offset * get_tile_size(args.out_cb),
                WRITE_NOC);
        }

        FORCE_INLINE uint64_t original_sem_noc_addr(const ReceiverArgs& args, uint32_t row, uint32_t semaphore_addr) {
            uint32_t original_packed = args.original_noc_coords[row];
            return get_noc_addr(
                unpack_noc_x(original_packed), unpack_noc_y(original_packed), semaphore_addr, WRITE_NOC);
        }

        template <bool is_sender_same_risc>
        FORCE_INLINE void original_receiver_impl(const ReceiverArgs& args) {
            volatile tt_l1_ptr uint32_t* phase1_semaphore_ptr =
                (volatile tt_l1_ptr uint32_t*)args.nope_phase1_semaphore_addr;
            volatile tt_l1_ptr uint32_t* phase2_done_semaphore_ptr =
                (volatile tt_l1_ptr uint32_t*)args.nope_phase2_semaphore_addr;
            volatile tt_l1_ptr uint32_t* rope_done_semaphore_ptr =
                (volatile tt_l1_ptr uint32_t*)args.rope_semaphore_addr;

            cb_reserve_back(args.receiver_in_cb, args.nope_tiles);
            noc_semaphore_wait(phase1_semaphore_ptr, args.num_nope_senders);
            noc_semaphore_set(phase1_semaphore_ptr, 0);
            cb_push_back(args.receiver_in_cb, args.nope_tiles);

            cb_wait_front(args.out_cb, args.nope_tiles);
            cb_reserve_back(args.out_cb, args.nope_tiles + args.rope_tiles);
            noc_semaphore_wait(phase2_done_semaphore_ptr, 1);
            noc_semaphore_set(phase2_done_semaphore_ptr, 0);
            noc_semaphore_wait(rope_done_semaphore_ptr, 1);
            noc_semaphore_set(rope_done_semaphore_ptr, 0);

            cb_push_back(args.out_cb, args.nope_tiles + args.rope_tiles);
        }

        FORCE_INLINE void helper_receiver_impl(
            const ReceiverArgs& args,
            uint32_t num_tiles,
            uint32_t num_senders,
            uint32_t local_sem_addr,
            uint32_t row,
            uint32_t dst_tile_offset,
            uint32_t remote_sem_addr) {
            auto* local_sem_ptr = (volatile tt_l1_ptr uint32_t*)local_sem_addr;

            cb_reserve_back(args.receiver_in_cb, num_tiles);
            noc_semaphore_wait(local_sem_ptr, num_senders);
            noc_semaphore_set(local_sem_ptr, 0);
            cb_push_back(args.receiver_in_cb, num_tiles);

            uint32_t src_addr = get_read_ptr(args.out_cb);
            uint64_t dst_data_noc_addr = original_out_noc_addr(args, row, dst_tile_offset);
            uint64_t dst_sem_noc_addr = original_sem_noc_addr(args, row, remote_sem_addr);
            uint32_t write_size_bytes = num_tiles * get_tile_size(args.out_cb);

            unified_kernels::noc_async_write_preprogram_all_state</*posted=*/true>(
                src_addr, dst_data_noc_addr, write_size_bytes, WRITE_NOC);
            unified_kernels::noc_async_atomic_inc_preprogram_all_state</*posted=*/false>(
                dst_sem_noc_addr, 1, 31, WRITE_NOC);

            cb_wait_front(args.out_cb, num_tiles);

            unified_kernels::noc_async_write_issue_txn</*posted=*/true>(WRITE_NOC);
            unified_kernels::noc_async_atomic_inc_issue_txn</*posted=*/false>(WRITE_NOC);

            noc_async_posted_writes_flushed(WRITE_NOC);
            cb_pop_front(args.out_cb, num_tiles);

            static_assert(
                WRITE_NOC == noc_index || noc_mode == DM_DYNAMIC_NOC,
                "WRITE_NOC differs from noc_index; must be in dynamic NOC mode");
            noc_async_atomic_barrier(WRITE_NOC);
        }

        template <bool is_sender_same_risc>
        FORCE_INLINE void nope_helper_receiver_impl(const ReceiverArgs& args) {
            helper_receiver_impl(
                args,
                args.nope_tiles,
                args.num_nope_senders,
                args.nope_phase2_semaphore_addr,
                nope_helper_receiver_row(),
                /*dst_tile_offset=*/args.nope_tiles,
                /*remote_sem_addr=*/args.nope_phase2_semaphore_addr);
        }

        template <bool is_sender_same_risc>
        FORCE_INLINE void rope_helper_receiver_impl(const ReceiverArgs& args) {
            helper_receiver_impl(
                args,
                args.rope_tiles,
                args.num_rope_senders,
                args.rope_semaphore_addr,
                rope_helper_receiver_row(),
                /*dst_tile_offset=*/2 * args.nope_tiles,
                /*remote_sem_addr=*/args.rope_semaphore_addr);
        }
#endif

#if defined(COMPILE_FOR_TRISC)
        FORCE_INLINE static void tilize_phase_impl(
            uint32_t in_cb, uint32_t out_cb, uint32_t num_tiles, uint32_t num_chunks, uint32_t chunk_size) {
            reconfig_data_format_srca<false, true>(in_cb);
            pack_reconfig_data_format<true>(out_cb);
            tilize_init(in_cb, num_tiles, out_cb);
            MATH((t6_semaphore_wait_on_max<p_stall::STALL_MATH>(semaphore::FPU_SFPU)));

            cb_wait_front(in_cb, num_tiles);
            cb_reserve_back(out_cb, num_tiles);
            tilize_block_custom(in_cb, num_chunks, chunk_size, out_cb);
            cb_push_back(out_cb, num_tiles);
            cb_pop_front(in_cb, num_tiles);

            tilize_uninit(in_cb, out_cb);
        }
#endif
    };
};

}  // namespace deepseek_b1_ops
