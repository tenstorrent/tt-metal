// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "dataflow_utils.hpp"
#include "kernel_op_api.hpp"
#include "kernel_utils.hpp"

#if defined(COMPILE_FOR_BRISC) || defined(COMPILE_FOR_NCRISC)
#include "api/dataflow/dataflow_api.h"
#include "api/debug/assert.h"
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

    struct DMArgs {
        uint32_t sender_grid_start_x;
        uint32_t sender_grid_start_y;
        uint32_t receiver_grid_start_x;
        uint32_t receiver_grid_start_y;
        uint32_t receiver_cols;
        uint32_t qnope_cols;
        uint32_t qnope_cb;
        uint32_t qrope_cb;
        uint32_t src_num_pages;
        uint32_t nope_phase1_semaphore_addr;
        uint32_t nope_phase2_semaphore_addr;
        uint32_t rope_semaphore_addr;
        uint32_t original_noc_coords[8];
        uint32_t nope_helper_noc_coords[8];
        uint32_t rope_helper_noc_coords[8];
        uint32_t receiver_data_addr;
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
        void operator()([[maybe_unused]] const DMArgs& args) {
#if defined(COMPILE_FOR_NCRISC) || defined(COMPILE_FOR_BRISC)
            if constexpr (IsSenderCore) {
                sender_impl<shared_dm_risc && (IsOriginalReceiverCore || IsNopeHelperCore || IsRopeHelperCore)>(args);
            }
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
            if constexpr (IsOriginalReceiverCore) {
                original_compute_impl(args);
            }
            if constexpr (IsNopeHelperCore) {
                nope_helper_compute_impl(args);
            }
            if constexpr (IsRopeHelperCore) {
                rope_helper_compute_impl(args);
            }
#endif
        }

    private:
#if defined(COMPILE_FOR_NCRISC) || defined(COMPILE_FOR_BRISC)
        static constexpr uint8_t WRITE_NOC = 0;

        static FORCE_INLINE uint32_t unpack_noc_x(uint32_t packed_coords) { return packed_coords & 0xFFFF; }

        static FORCE_INLINE uint32_t unpack_noc_y(uint32_t packed_coords) { return (packed_coords >> 16) & 0xFFFF; }

        template <bool is_receiver_same_risc>
        FORCE_INLINE void sender_impl(const DMArgs& args) {
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

            uint32_t src_addr = get_read_ptr(src_cb);

            if (is_qnope_core) {
                uint32_t dst_offset = my_col * half_qnope_data_size_bytes;
                uint64_t dst_data_noc_addr_0 = original_noc_coord | (uint64_t)(args.receiver_data_addr + dst_offset);
                uint64_t dst_data_noc_addr_1 = nope_helper_noc_coord | (uint64_t)(args.receiver_data_addr + dst_offset);
                unified_kernels::unicast_write_increment_counters<true>(2, WRITE_NOC);
                unified_kernels::unicast_atomic_inc_increment_counters<false>(2, WRITE_NOC);
                unified_kernels::unicast_write_set_state<true, true, true, true, false, write_cmd_buf>(
                    src_addr, dst_data_noc_addr_0, half_qnope_data_size_bytes, WRITE_NOC);
                unified_kernels::unicast_atomic_inc_set_state<false, true, true, false, write_at_cmd_buf>(
                    original_noc_coord | (uint64_t)args.nope_phase1_semaphore_addr, 1, 31, WRITE_NOC);

                cb_wait_front(src_cb, args.src_num_pages);

                unified_kernels::noc_async_write_issue_txn(WRITE_NOC);
                unified_kernels::noc_async_atomic_inc_issue_txn(WRITE_NOC);

                unified_kernels::unicast_write_set_state<true, true, true, false, false, write_cmd_buf>(
                    src_addr + half_qnope_data_size_bytes, dst_data_noc_addr_1, half_qnope_data_size_bytes, WRITE_NOC);
                unified_kernels::noc_async_write_issue_txn(WRITE_NOC);
                unified_kernels::unicast_atomic_inc_set_state<false, true, false, false, write_at_cmd_buf>(
                    nope_helper_noc_coord | (uint64_t)args.nope_phase2_semaphore_addr, 1, 31, WRITE_NOC);
                unified_kernels::noc_async_atomic_inc_issue_txn(WRITE_NOC);
            } else {
                uint32_t qrope_col = my_col - args.qnope_cols;
                uint32_t dst_offset = 2 * qrope_col * CTArgs::qrope_head_size_bytes;
                uint64_t dst_data_noc_addr = rope_helper_noc_coord | (uint64_t)(args.receiver_data_addr + dst_offset);
                constexpr uint32_t double_qrope_head_size_bytes = CTArgs::qrope_head_size_bytes * 2;
                unified_kernels::noc_async_write_preprogram_all_state<true>(
                    src_addr, dst_data_noc_addr, double_qrope_head_size_bytes, WRITE_NOC);
                unified_kernels::noc_async_atomic_inc_preprogram_all_state<false>(
                    rope_helper_noc_coord | (uint64_t)args.rope_semaphore_addr, 1, 31, WRITE_NOC);

                cb_wait_front(src_cb, args.src_num_pages);

                unified_kernels::noc_async_write_issue_txn(WRITE_NOC);
                unified_kernels::noc_async_atomic_inc_issue_txn(WRITE_NOC);
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

        FORCE_INLINE uint32_t receiver_row(const DMArgs& args) {
            uint32_t rx = (my_logical_x_ - args.receiver_grid_start_x) % args.receiver_cols;
            uint32_t ry = my_logical_y_ - args.receiver_grid_start_y;
            return rx + ry * args.receiver_cols;
        }

        FORCE_INLINE uint32_t nope_helper_receiver_row() { return (my_logical_y_ - 5) * 4 + my_logical_x_; }

        FORCE_INLINE uint32_t rope_helper_receiver_row() { return my_logical_y_; }

        FORCE_INLINE uint64_t original_out_noc_addr(const DMArgs& args, uint32_t row, uint32_t tile_offset) {
            uint32_t original_packed = args.original_noc_coords[row];
            return get_noc_addr(
                unpack_noc_x(original_packed),
                unpack_noc_y(original_packed),
                get_read_ptr(args.out_cb) + tile_offset * get_tile_size(args.out_cb),
                WRITE_NOC);
        }

        FORCE_INLINE uint64_t original_sem_noc_addr(const DMArgs& args, uint32_t row, uint32_t semaphore_addr) {
            uint32_t original_packed = args.original_noc_coords[row];
            return get_noc_addr(
                unpack_noc_x(original_packed), unpack_noc_y(original_packed), semaphore_addr, WRITE_NOC);
        }

        template <bool is_sender_same_risc>
        FORCE_INLINE void original_receiver_impl(const DMArgs& args) {
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

        template <bool is_sender_same_risc>
        FORCE_INLINE void nope_helper_receiver_impl(const DMArgs& args) {
            volatile tt_l1_ptr uint32_t* phase2_semaphore_ptr =
                (volatile tt_l1_ptr uint32_t*)args.nope_phase2_semaphore_addr;

            cb_reserve_back(args.receiver_in_cb, args.nope_tiles);
            noc_semaphore_wait(phase2_semaphore_ptr, args.num_nope_senders);
            noc_semaphore_set(phase2_semaphore_ptr, 0);
            cb_push_back(args.receiver_in_cb, args.nope_tiles);

            uint32_t row = nope_helper_receiver_row();
            uint32_t write_size = args.nope_tiles * get_tile_size(args.out_cb);
            ASSERT(write_size <= NOC_MAX_BURST_SIZE);
            unified_kernels::noc_async_write_preprogram_all_state<true>(
                get_read_ptr(args.out_cb), original_out_noc_addr(args, row, args.nope_tiles), write_size, WRITE_NOC);
            unified_kernels::noc_async_atomic_inc_preprogram_all_state<false>(
                original_sem_noc_addr(args, row, args.nope_phase2_semaphore_addr), 1, 31, WRITE_NOC);
            cb_wait_front(args.out_cb, args.nope_tiles);
            unified_kernels::noc_async_write_issue_txn(WRITE_NOC);
            unified_kernels::noc_async_atomic_inc_issue_txn(WRITE_NOC);
            noc_async_posted_writes_flushed(WRITE_NOC);
            cb_pop_front(args.out_cb, args.nope_tiles);

            static_assert(
                WRITE_NOC == noc_index || noc_mode == DM_DYNAMIC_NOC,
                "WRITE_NOC differs from noc_index; must be in dynamic NOC mode");
            noc_async_atomic_barrier(WRITE_NOC);
        }

        template <bool is_sender_same_risc>
        FORCE_INLINE void rope_helper_receiver_impl(const DMArgs& args) {
            volatile tt_l1_ptr uint32_t* rope_semaphore_ptr = (volatile tt_l1_ptr uint32_t*)args.rope_semaphore_addr;

            cb_reserve_back(args.receiver_in_cb, args.rope_tiles);
            noc_semaphore_wait(rope_semaphore_ptr, args.num_rope_senders);
            noc_semaphore_set(rope_semaphore_ptr, 0);
            cb_push_back(args.receiver_in_cb, args.rope_tiles);

            uint32_t row = rope_helper_receiver_row();
            uint32_t write_size = args.rope_tiles * get_tile_size(args.out_cb);
            ASSERT(write_size <= NOC_MAX_BURST_SIZE);
            unified_kernels::noc_async_write_preprogram_all_state<true>(
                get_read_ptr(args.out_cb),
                original_out_noc_addr(args, row, 2 * args.nope_tiles),
                write_size,
                WRITE_NOC);
            unified_kernels::noc_async_atomic_inc_preprogram_all_state<false>(
                original_sem_noc_addr(args, row, args.rope_semaphore_addr), 1, 31, WRITE_NOC);
            cb_wait_front(args.out_cb, args.rope_tiles);
            unified_kernels::noc_async_write_issue_txn(WRITE_NOC);
            unified_kernels::noc_async_atomic_inc_issue_txn(WRITE_NOC);
            noc_async_posted_writes_flushed(WRITE_NOC);
            cb_pop_front(args.out_cb, args.rope_tiles);

            static_assert(
                WRITE_NOC == noc_index || noc_mode == DM_DYNAMIC_NOC,
                "WRITE_NOC differs from noc_index; must be in dynamic NOC mode");
            noc_async_atomic_barrier(WRITE_NOC);
        }
#endif

#if defined(COMPILE_FOR_TRISC)
        FORCE_INLINE void original_compute_impl(const ComputeArgs& args) {
            constexpr uint32_t nope_num_chunks = 2;
            uint32_t nope_chunk = args.nope_tiles / nope_num_chunks;
            reconfig_data_format_srca<false, true>(args.receiver_in_cb);
            pack_reconfig_data_format<true>(args.out_cb);
            tilize_init(args.receiver_in_cb, args.nope_tiles, args.out_cb);
            MATH((t6_semaphore_wait_on_max<p_stall::STALL_MATH>(semaphore::FPU_SFPU)));

            cb_wait_front(args.receiver_in_cb, args.nope_tiles);
            cb_reserve_back(args.out_cb, args.nope_tiles);
            tilize_block_custom(args.receiver_in_cb, nope_num_chunks, nope_chunk, args.out_cb);
            cb_push_back(args.out_cb, args.nope_tiles);
            cb_pop_front(args.receiver_in_cb, args.nope_tiles);

            tilize_uninit(args.receiver_in_cb, args.out_cb);
        }

        FORCE_INLINE void nope_helper_compute_impl(const ComputeArgs& args) {
            constexpr uint32_t nope_num_chunks = 2;
            uint32_t nope_chunk = args.nope_tiles / nope_num_chunks;
            reconfig_data_format_srca<false, true>(args.receiver_in_cb);
            pack_reconfig_data_format<true>(args.out_cb);
            tilize_init(args.receiver_in_cb, args.nope_tiles, args.out_cb);
            MATH((t6_semaphore_wait_on_max<p_stall::STALL_MATH>(semaphore::FPU_SFPU)));

            cb_wait_front(args.receiver_in_cb, args.nope_tiles);
            cb_reserve_back(args.out_cb, args.nope_tiles);
            tilize_block_custom(args.receiver_in_cb, nope_num_chunks, nope_chunk, args.out_cb);
            cb_push_back(args.out_cb, args.nope_tiles);
            cb_pop_front(args.receiver_in_cb, args.nope_tiles);

            tilize_uninit(args.receiver_in_cb, args.out_cb);
        }

        FORCE_INLINE void rope_helper_compute_impl(const ComputeArgs& args) {
            reconfig_data_format_srca<false, true>(args.receiver_in_cb);
            pack_reconfig_data_format<true>(args.out_cb);
            tilize_init(args.receiver_in_cb, args.rope_tiles, args.out_cb);
            MATH((t6_semaphore_wait_on_max<p_stall::STALL_MATH>(semaphore::FPU_SFPU)));

            cb_wait_front(args.receiver_in_cb, args.rope_tiles);
            cb_reserve_back(args.out_cb, args.rope_tiles);
            tilize_block_custom(args.receiver_in_cb, args.rope_tiles, 1, args.out_cb);
            cb_push_back(args.out_cb, args.rope_tiles);
            cb_pop_front(args.receiver_in_cb, args.rope_tiles);

            tilize_uninit(args.receiver_in_cb, args.out_cb);
        }
#endif
    };
};

}  // namespace deepseek_b1_ops
