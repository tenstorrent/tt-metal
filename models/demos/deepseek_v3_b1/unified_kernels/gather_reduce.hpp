// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "kernel_op_api.hpp"
#include "kernel_utils.hpp"

#if defined(COMPILE_FOR_BRISC) || defined(COMPILE_FOR_NCRISC)
#include "api/dataflow/dataflow_api.h"
#elif defined(COMPILE_FOR_TRISC)
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_binary.h"
#endif

namespace deepseek_b1_ops {

// ============================================================================
// GatherReduce micro-op
//
// Fuses dual-destination gather with pairwise reduction:
// - 96 matmul sender cores are split into two 48-core halves by row-major linear core index.
// - Half0 senders write 64B pages into half0_dst_cb (CB7) on the receiver core.
// - Half1 senders write 64B pages into half1_dst_cb (CB8) at matching sender_idx offsets.
// - Receiver exposes both CBs as 3 x [16,32] tiles each.
// - TRISC reduces in place: half0_dst_cb += half1_dst_cb over num_tiles.
//
// CB States:
//   NCRISC (Sender):
//     - Waits: src_cb (src_num_pages)
//     - Remote writes: dst_cb (dst_num_tiles)
//     - Pops: src_cb (src_num_pages) if pop_src=true
//   BRISC (Receiver):
//     - Reserves: dst_cb (2 * dst_num_tiles)
//     - Pushes: dst_cb (2 * dst_num_tiles)
//   TRISC (Reducer):
//     - Waits: in_cb (2 * num_tiles)
//     - Computes: out_cb = in_cb first half + in_cb second half
//     - Pops: in_cb (2 * num_tiles)
//     - Re-pushes: out_cb (num_tiles)
//
// Semaphore states:
//   Sender: increments receiver semaphore after NOC write completion
//   Receiver: waits on noc0/noc1 sender counts, then resets both semaphores to 0
// ============================================================================

struct GatherReduce {
    // ========================================================================
    // Runtime args structs - different layout per RISC
    // ========================================================================

    struct ReceiverArgs {
        uint32_t noc0_num_senders;
        uint32_t noc1_num_senders;
        uint32_t noc0_receiver_semaphore_addr;
        uint32_t noc1_receiver_semaphore_addr;
        uint32_t dst_cb;
        uint32_t dst_num_tiles;
    };

    struct SenderArgs {
        uint32_t dest_noc_x;
        uint32_t dest_noc_y;
        uint32_t data_size_bytes;
        uint32_t receiver_semaphore_addr;
        uint32_t src_cb;
        uint32_t src_num_pages;
        uint32_t gather_reduce_grid_start_x;
        uint32_t gather_reduce_grid_start_y;
        uint32_t gather_reduce_grid_end_x;
        uint32_t gather_reduce_grid_end_y;
        uint32_t gather_reduce_half_num_cores;
        uint32_t dst_cb_id;
        uint32_t half_size_bytes;
    };

    struct ComputeArgs {
        uint32_t in_cb;
        uint32_t out_cb;
        uint32_t num_tiles;
    };

    // Note: For gather reduce, NCRISC=Sender, BRISC=Receiver, TRISC=Reducer
    using RTArgs = unified_kernels::SelectByRISCV<SenderArgs, ReceiverArgs, ComputeArgs>;

#if defined(COMPILE_FOR_TRISC)
    static inline void add_half_tiles(uint32_t in_cb, uint32_t out_cb, uint32_t num_tiles) {
        reconfig_data_format<false, true>(in_cb, in_cb);
        pack_reconfig_data_format<true>(out_cb);
        add_tiles_init(in_cb, in_cb);
        cb_wait_front(in_cb, 2 * num_tiles);

        // Process tiles - element-wise add
        tile_regs_acquire();
        for (uint32_t i = 0; i < num_tiles; i++) {
            add_tiles(in_cb, in_cb, i, num_tiles + i, i);
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t i = 0; i < num_tiles; i++) {
            pack_tile(i, out_cb);
        }
        cb_push_back(out_cb, num_tiles);
        tile_regs_release();

        cb_pop_front(in_cb, 2 * num_tiles);
    }
#endif

    // ========================================================================
    // Op - the actual operation
    //
    // IsSenderCore: compile-time flag for sender cores
    // IsReceiverCore: compile-time flag for receiver cores
    // IsReduceCore: compile-time flag for reduce cores
    // pop_src: whether to pop the source CB after sending
    // ========================================================================
    template <bool IsSenderCore, bool IsReceiverCore, bool IsReduceCore, bool pop_src>
    class Op {
    public:
        void operator()(const RTArgs& args) { impl(args); }

    private:
        void impl([[maybe_unused]] const RTArgs& args) {
#if defined(COMPILE_FOR_NCRISC)
            // ================================================================
            // NCRISC (Sender) - DataMovementProcessor.RISCV_1
            // ================================================================
            if constexpr (IsSenderCore) {
                const auto half_info = unified_kernels::get_split_half_core_info<true>(
                    args.gather_reduce_grid_start_x,
                    args.gather_reduce_grid_start_y,
                    args.gather_reduce_grid_end_x,
                    args.gather_reduce_grid_end_y,
                    args.gather_reduce_half_num_cores);
                uint32_t dst_base_addr = get_write_ptr(args.dst_cb_id);
                uint32_t dst_offset =
                    half_info.is_half0 ? 0 : args.half_size_bytes + half_info.half_local_idx * args.data_size_bytes;

                const uint64_t dst_noc_coord = get_noc_addr(args.dest_noc_x, args.dest_noc_y, 0);
                uint64_t dst_data_noc_addr = dst_noc_coord | (uint64_t)(dst_base_addr + dst_offset);
                uint64_t dst_sem_noc_addr = dst_noc_coord | (uint64_t)args.receiver_semaphore_addr;

                // Wait for source CB data to be ready
                cb_wait_front(args.src_cb, args.src_num_pages);

                // Get source address from CB
                uint32_t src_addr = get_read_ptr(args.src_cb);

                noc_async_write_one_packet<true, true>(src_addr, dst_data_noc_addr, args.data_size_bytes);
                // BH does not support posted atomics due to a bug
                noc_semaphore_inc(dst_sem_noc_addr, 1);
                noc_async_posted_writes_flushed();

                // Pop the source CB after sending
                if constexpr (pop_src) {
                    cb_pop_front(args.src_cb, args.src_num_pages);
                }
                noc_async_atomic_barrier();
            }
#elif defined(COMPILE_FOR_BRISC)
            // ================================================================
            // BRISC (Receiver) - DataMovementProcessor.RISCV_0
            // ================================================================
            if constexpr (IsReceiverCore) {
                volatile tt_l1_ptr uint32_t* noc0_receiver_semaphore_addr_ptr =
                    (volatile tt_l1_ptr uint32_t*)args.noc0_receiver_semaphore_addr;

                // Reserve space in destination CBs
                cb_reserve_back(args.dst_cb, 2 * args.dst_num_tiles);

                noc_semaphore_wait(noc0_receiver_semaphore_addr_ptr, args.noc0_num_senders);
                noc_semaphore_set(noc0_receiver_semaphore_addr_ptr, 0);

                if (args.noc1_num_senders > 0) {
                    volatile tt_l1_ptr uint32_t* noc1_receiver_semaphore_addr_ptr =
                        (volatile tt_l1_ptr uint32_t*)args.noc1_receiver_semaphore_addr;
                    noc_semaphore_wait(noc1_receiver_semaphore_addr_ptr, args.noc1_num_senders);
                    noc_semaphore_set(noc1_receiver_semaphore_addr_ptr, 0);
                }

                // Push to destination CBs after data arrived
                cb_push_back(args.dst_cb, 2 * args.dst_num_tiles);
            }
#elif defined(COMPILE_FOR_TRISC)
            // ================================================================
            // TRISC - No-op (gather is dataflow only)
            // ================================================================
            if constexpr (IsReduceCore) {
                add_half_tiles(args.in_cb, args.out_cb, args.num_tiles);
            }
#endif
        }
    };
};

}  // namespace deepseek_b1_ops
