// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "kernel_op_api.hpp"
#include "../micro_ops/flash_mla/kernels/rt_args_common.hpp"

#if defined(COMPILE_FOR_NCRISC)
#include "api/dataflow/dataflow_api.h"
#include "mcast.hpp"
#elif defined(COMPILE_FOR_BRISC)
#include "api/dataflow/dataflow_api.h"
#include "mcast.hpp"
#include "api/debug/assert.h"
#elif defined(COMPILE_FOR_TRISC)
#include <cstdint>
#include "api/compute/compute_kernel_api.h"
#include "../kernel_includes/tt_metal/include/compute_kernel_api/sdpa.h"
#include "api/compute/eltwise_unary/exp.h"
#endif

#if defined(COMPILE_FOR_BRISC) || defined(COMPILE_FOR_NCRISC)
static_assert(noc_mode == DM_DYNAMIC_NOC, "Flash MLA Decode kernel only supports DM_DYNAMIC_NOC");
#endif

// ============================================================================
// NCRISC helpers
// ============================================================================
#if defined(COMPILE_FOR_NCRISC)
template <typename Accessor>
FORCE_INLINE uint64_t get_shard_noc_addr_helper(const Accessor& reader, uint32_t shard_id) {
    return reader.get_shard_noc_addr(shard_id);
}

constexpr uint32_t MCAST_INVALID = 0;
constexpr uint32_t MCAST_VALID = 1;
#endif

// ============================================================================
// BRISC helpers
// ============================================================================
#if defined(COMPILE_FOR_BRISC)
template <uint32_t bits_per_step>
FORCE_INLINE constexpr uint32_t step_semaphore_inc(uint32_t step) {
    return 1U << (step * bits_per_step);
}
template <uint32_t bits_per_step>
FORCE_INLINE constexpr uint32_t step_semaphore_shift(uint32_t step) {
    return step * bits_per_step;
}
#endif

namespace deepseek_b1_ops {

// ============================================================================
// Flash MLA Decode micro-op
//
// Implements flash attention decode for MLA where K and V share the same tensor.
//   NCRISC (Reader): Read Q from sharded memory, read K from ND-sharded DRAM
//   BRISC (Writer):  Multicast K data to S block receivers, tree reduction
//   TRISC (Compute): SDPA compute with flash attention chunking and tree reduction
// ============================================================================
struct FlashMLADecode {
    // ========================================================================
    // Args structs - different layout per RISC.
    // Includes both per-core runtime values and compile-time constants.
    // ========================================================================

    template <uint32_t k_page_size_, uint32_t vDHt_, uint32_t cb_out_o_>
    struct WriterCTArgs {
        static constexpr uint32_t k_page_size = k_page_size_;
        static constexpr uint32_t vDHt = vDHt_;
        static constexpr uint32_t cb_out_o = cb_out_o_;
    };

    struct ReaderCTArgs {};

    template <
        uint32_t cb_q_in_,
        uint32_t cb_k_in_,
        uint32_t cb_mask_,
        uint32_t cb_interm_out_,
        uint32_t cb_interm_ms_,
        uint32_t cb_out_in_,
        uint32_t cb_ms_in_,
        uint32_t cb_out_o_,
        uint32_t cb_out_ms_,
        uint32_t cb_out_final_>
    struct ComputeCTArgs {
        static constexpr uint32_t cb_q_in = cb_q_in_;
        static constexpr uint32_t cb_k_in = cb_k_in_;
        static constexpr uint32_t cb_mask = cb_mask_;
        static constexpr uint32_t cb_interm_out = cb_interm_out_;
        static constexpr uint32_t cb_interm_ms = cb_interm_ms_;
        static constexpr uint32_t cb_out_in = cb_out_in_;
        static constexpr uint32_t cb_ms_in = cb_ms_in_;
        static constexpr uint32_t cb_out_o = cb_out_o_;
        static constexpr uint32_t cb_out_ms = cb_out_ms_;
        static constexpr uint32_t cb_out_final = cb_out_final_;
    };

    struct ReaderArgs {
        uint32_t k_addr;
        uint32_t pos_addr;
        uint32_t cur_batch;
        uint32_t core_num_in_reduce;
        uint32_t is_mcast_sender;
        uint32_t mcast_start_x;
        uint32_t mcast_start_y;
        uint32_t vc;
        uint32_t St;
        uint32_t DHt;
        uint32_t Sk_chunk_t;
        uint32_t num_cores_per_head;
        uint32_t k_chunk_size;
        uint32_t mcast_semaphore_addr;
        uint32_t k_page_size;
        uint32_t k_num_pages;
        uint32_t ncrisc_brisc_sync_semaphore_addr;
        uint32_t receiver_ready_semaphore_addr;
        uint32_t kv_cache_cur_pos_ready_semaphore_addr;
        uint32_t kv_cache_cur_pos_ready_value;
        uint32_t cb_k_in;
    };

    struct WriterArgs {
        uint32_t pos_addr;
        uint32_t cur_batch;
        uint32_t core_num_in_reduce;
        uint32_t is_output_core;
        uint32_t is_mcast_sender;
        uint32_t output_core_noc_x;
        uint32_t output_core_noc_y;
        uint32_t mcast_start_x;
        uint32_t mcast_start_y;
        uint32_t mcast_end_x;
        uint32_t mcast_end_y;
        tt_l1_ptr uint32_t* tree_reduction_info;
        uint32_t Sk_chunk_t;
        uint32_t num_cores_per_head;
        uint32_t reducer_semaphore_addr;
        uint32_t k_chunk_size;
        uint32_t q_chunk_size_bytes;
        uint32_t DHt;
        uint32_t num_mcast_dests;
        uint32_t full_grid_mcast_start_x;
        uint32_t full_grid_mcast_start_y;
        uint32_t full_grid_mcast_end_x;
        uint32_t full_grid_mcast_end_y;
        uint32_t full_grid_mcast_num_dests;
        uint32_t q_input_mcast_semaphore_addr;
        uint32_t mcast_semaphore_addr;
        uint32_t ncrisc_brisc_sync_semaphore_addr;
        uint32_t k_num_pages;
        uint32_t num_tree_reduction_steps;
        uint32_t receiver_ready_semaphore_addr;
        uint32_t cb_k_in;
        uint32_t cb_q_in;
        uint32_t cb_mask;
        uint32_t cb_out_in;
        uint32_t cb_ms_in;
        uint32_t cb_out_ms;
    };

    struct ComputeArgs {
        uint32_t pos_addr;
        uint32_t do_reduce;
        uint32_t do_output;
        uint32_t cur_batch;
        uint32_t core_num_in_reduce;
        uint32_t is_sender_after_reduce;
        tt_l1_ptr uint32_t* tree_reduction_info;
        uint32_t k_chunk_size;
        uint32_t num_cores_per_head;
        uint32_t num_tree_reduction_steps;
    };

    using RTArgs = unified_kernels::SelectByRISCV<ReaderArgs, WriterArgs, ComputeArgs>;

    // ========================================================================
    // Op - templated on CTArgs (compile-time args) and IsActiveCore
    // ========================================================================
    template <typename CTArgs, bool IsActiveCore, bool IsKVCacheUpdateCore>
    class Op {
    public:
        void operator()(const RTArgs& args) {
            if constexpr (IsActiveCore) {
                impl(args);
            }
        }

    private:
        void impl([[maybe_unused]] const RTArgs& args) {
// ====================================================================
// NCRISC (Reader)
// ====================================================================
#if defined(COMPILE_FOR_NCRISC)
            constexpr uint8_t READ_NOC_INDEX = 0;
            constexpr uint8_t ATOMIC_NOC_INDEX = 1;
            constexpr auto k_tensor_args = TensorAccessorArgs<0>();

            const bool is_mcast_sender = args.is_mcast_sender == 1;

            volatile tt_l1_ptr uint32_t* pos_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(args.pos_addr);
            uint32_t cur_pos = pos_ptr[0];

            auto [k_num_chunks, k_chunk_start, k_chunk_end] = get_runtime_args(
                cur_pos, args.cur_batch, args.core_num_in_reduce, args.num_cores_per_head, args.k_chunk_size);
            (void)k_num_chunks;

            if (k_chunk_start == k_chunk_end) {
                return;
            }

            const uint32_t k_chunk_tiles = args.Sk_chunk_t * args.DHt;
            const uint32_t k_tile_bytes = get_tile_size(args.cb_k_in);

            const auto k_reader = TensorAccessor(k_tensor_args, args.k_addr, k_tile_bytes);

            const uint32_t num_chunks_per_batch = args.St / args.Sk_chunk_t;

            volatile tt_l1_ptr uint32_t* mcast_semaphore_ptr =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(args.mcast_semaphore_addr);

            volatile tt_l1_ptr uint32_t* ncrisc_brisc_sync_curr_ptr =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(args.ncrisc_brisc_sync_semaphore_addr);
            volatile tt_l1_ptr uint32_t* ncrisc_brisc_sync_next_ptr =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(args.ncrisc_brisc_sync_semaphore_addr + 4);
            volatile tt_l1_ptr uint32_t* k_write_curr_ptr_shared =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(args.ncrisc_brisc_sync_semaphore_addr + 8);
            volatile tt_l1_ptr uint32_t* k_write_next_ptr_shared =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(args.ncrisc_brisc_sync_semaphore_addr + 12);
            // Reset the other semaphores outside the base offset to 0
            *ncrisc_brisc_sync_next_ptr = 0;

            volatile tt_l1_ptr uint32_t* receiver_ready_semaphore_ptr =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(args.receiver_ready_semaphore_addr);
            const uint64_t sender_receiver_ready_noc_addr = get_noc_addr(
                args.mcast_start_x, args.mcast_start_y, args.receiver_ready_semaphore_addr, ATOMIC_NOC_INDEX);

            constexpr uint32_t kv_batch = 0;

            if (is_mcast_sender) {
                const uint32_t shard_id = kv_batch * num_chunks_per_batch + k_chunk_start;
                uint64_t k_src_noc_addr = get_shard_noc_addr_helper(k_reader, shard_id);
                noc_async_read_one_packet_set_state<true>(k_src_noc_addr, args.k_page_size, args.vc, READ_NOC_INDEX);
            }

            volatile tt_l1_ptr uint32_t* kv_cache_cur_pos_ready_semaphore_ptr =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(args.kv_cache_cur_pos_ready_semaphore_addr);

            // Wait for KV cache cur pos ready
            if constexpr (IsKVCacheUpdateCore) {
                noc_semaphore_wait(kv_cache_cur_pos_ready_semaphore_ptr, args.kv_cache_cur_pos_ready_value - 1);
            } else {
                noc_semaphore_wait(kv_cache_cur_pos_ready_semaphore_ptr, args.kv_cache_cur_pos_ready_value);
            }
            for (uint32_t k_chunk = k_chunk_start; k_chunk < k_chunk_end; k_chunk += args.num_cores_per_head) {
                {
                    DeviceZoneScopedN("reader-k-read");

                    cb_reserve_back(args.cb_k_in, k_chunk_tiles);
                    uint32_t k_write_ptr = get_write_ptr(args.cb_k_in);

                    if (is_mcast_sender) {
                        DeviceZoneScopedN("mcast-sender-sharded-read");
                        // Previous multicasts could have put trids into a non-zero state, so reset the barrier counter
                        reset_noc_trid_barrier_counter(NOC_CLEAR_OUTSTANDING_REQ_MASK, READ_NOC_INDEX);
                        const uint32_t shard_id = kv_batch * num_chunks_per_batch + k_chunk;
                        uint64_t k_src_noc_addr = get_shard_noc_addr_helper(k_reader, shard_id);

                        constexpr uint32_t NUM_TRIDS = NOC_MAX_TRANSACTION_ID - 1;
                        uint32_t src_base_addr = (uint32_t)(k_src_noc_addr & 0xFFFFFFFF);
                        uint32_t src_offset = 0;
                        uint32_t dst_addr = k_write_ptr;

                        uint32_t curr_trid = 1;
                        uint32_t wait_trid = 1;
                        uint32_t pages_issued = 0;
                        uint32_t pages_completed = 0;

                        noc_semaphore_wait(ncrisc_brisc_sync_curr_ptr, 0);
                        *k_write_curr_ptr_shared = k_write_ptr;
                        for (uint32_t i = 0; i < NUM_TRIDS && pages_issued < args.k_num_pages; ++i) {
                            noc_async_read_set_trid(curr_trid, READ_NOC_INDEX);
                            noc_async_read_one_packet_with_state_with_trid(
                                src_base_addr, src_offset, dst_addr, curr_trid, READ_NOC_INDEX);
                            src_offset += args.k_page_size;
                            dst_addr += args.k_page_size;
                            curr_trid = (curr_trid % NUM_TRIDS) + 1;
                            pages_issued++;
                        }

                        while (pages_completed < args.k_num_pages) {
                            noc_async_read_barrier_with_trid(wait_trid, READ_NOC_INDEX);
                            *ncrisc_brisc_sync_curr_ptr += 1;
                            pages_completed++;

                            if (pages_issued < args.k_num_pages) {
                                noc_async_read_set_trid(curr_trid, READ_NOC_INDEX);
                                noc_async_read_one_packet_with_state_with_trid(
                                    src_base_addr, src_offset, dst_addr, curr_trid, READ_NOC_INDEX);
                                src_offset += args.k_page_size;
                                dst_addr += args.k_page_size;
                                curr_trid = (curr_trid % NUM_TRIDS) + 1;
                                pages_issued++;
                            }

                            wait_trid = (wait_trid % NUM_TRIDS) + 1;
                        }

                        std::swap(ncrisc_brisc_sync_curr_ptr, ncrisc_brisc_sync_next_ptr);
                        std::swap(k_write_curr_ptr_shared, k_write_next_ptr_shared);
                    } else {
                        DeviceZoneScopedN("mcast-receiver-signal-ready");
                        noc_semaphore_inc(sender_receiver_ready_noc_addr, 1, ATOMIC_NOC_INDEX);

                        noc_semaphore_wait(mcast_semaphore_ptr, MCAST_VALID);
                        noc_semaphore_set(mcast_semaphore_ptr, MCAST_INVALID);
                    }

                    cb_push_back(args.cb_k_in, k_chunk_tiles);
                }
            }
            noc_semaphore_set(kv_cache_cur_pos_ready_semaphore_ptr, 0);

// ====================================================================
// BRISC (Writer)
// ====================================================================
#elif defined(COMPILE_FOR_BRISC)
            constexpr uint8_t MCAST_NOC_INDEX = 0;
            constexpr uint8_t READ_NOC_INDEX = 1;
            constexpr uint8_t WRITE_NOC_INDEX = 1;
            constexpr uint8_t ATOMIC_NOC_INDEX = 1;

            constexpr uint32_t k_page_size = CTArgs::k_page_size;
            constexpr uint32_t vDHt = CTArgs::vDHt;
            constexpr uint32_t cb_out_o = CTArgs::cb_out_o;

            constexpr uint32_t out_chunk_tiles = vDHt;
            constexpr uint32_t tile_bytes_intermed = get_tile_size(cb_out_o);
            constexpr uint32_t o_write_size = out_chunk_tiles * tile_bytes_intermed;
            constexpr uint32_t ms_write_size = tile_bytes_intermed;

            const uint32_t q_chunk_tiles = args.DHt;

            volatile tt_l1_ptr uint32_t* q_input_mcast_semaphore_ptr =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(args.q_input_mcast_semaphore_addr);

            const bool is_mcast_sender = args.is_mcast_sender == 1;
            const bool is_output_core = args.is_output_core == 1;

            {
                DeviceZoneScopedN("reader-q-read");
                if (is_output_core) {
                    cb_wait_front(args.cb_q_in, q_chunk_tiles);
                    if (is_mcast_sender) {
                        noc_semaphore_wait(q_input_mcast_semaphore_ptr, args.num_mcast_dests);
                        uint64_t q_input_mcast_sem_noc_addr = get_noc_multicast_addr<MCAST_NOC_INDEX>(
                            args.full_grid_mcast_start_x,
                            args.full_grid_mcast_start_y,
                            args.full_grid_mcast_end_x,
                            args.full_grid_mcast_end_y,
                            args.q_input_mcast_semaphore_addr);
                        noc_semaphore_inc_multicast(
                            q_input_mcast_sem_noc_addr, 1, args.full_grid_mcast_num_dests, MCAST_NOC_INDEX);
                        // This is needed because we need to wait for all transactions before resetting the trids
                        // Could move it later but don't think it makes much difference
                        noc_async_atomic_barrier(MCAST_NOC_INDEX);
                    } else {
                        const uint64_t sender_receiver_ready_noc_addr = get_noc_addr(
                            args.mcast_start_x,
                            args.mcast_start_y,
                            args.q_input_mcast_semaphore_addr,
                            ATOMIC_NOC_INDEX);
                        noc_semaphore_inc(sender_receiver_ready_noc_addr, 1, ATOMIC_NOC_INDEX);
                        noc_async_atomic_barrier(ATOMIC_NOC_INDEX);
                        noc_semaphore_wait(q_input_mcast_semaphore_ptr, 1);
                    }
                } else {
                    // wait for 8 q heads
                    uint64_t q_noc_addr =
                        get_noc_addr(args.output_core_noc_x, args.output_core_noc_y, get_read_ptr(args.cb_q_in));
                    cb_reserve_back(args.cb_q_in, q_chunk_tiles);
                    noc_semaphore_wait(q_input_mcast_semaphore_ptr, 1);
                    noc_async_read(q_noc_addr, get_write_ptr(args.cb_q_in), args.q_chunk_size_bytes, READ_NOC_INDEX);
                    noc_async_read_barrier(READ_NOC_INDEX);
                    cb_push_back(args.cb_q_in, q_chunk_tiles);
                }
                noc_semaphore_set(q_input_mcast_semaphore_ptr, 0);
            }

            volatile tt_l1_ptr uint32_t* pos_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(args.pos_addr);
            uint32_t cur_pos = pos_ptr[0];

            auto [k_num_chunks, k_chunk_start, k_chunk_end] = get_runtime_args(
                cur_pos, args.cur_batch, args.core_num_in_reduce, args.num_cores_per_head, args.k_chunk_size);

            if (k_chunk_start == k_chunk_end) {
                return;
            }
            // Mask logic could be overlapped with the noc txns above, but currently DRAM reading fully overlaps with
            // the mask logic
            bool mask_last_chunk = k_chunk_end == k_num_chunks && (cur_pos + 1) % args.k_chunk_size != 0;
            if (mask_last_chunk) {
                DeviceZoneScopedN("mask-last-chunk");
                cb_reserve_back(args.cb_mask, 1);
                volatile tt_l1_ptr uint32_t* mask_write_ptr =
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(args.cb_mask));
                uint32_t num_unmasked = cur_pos % args.k_chunk_size + 1;
                uint32_t i = 0;
                for (; i < num_unmasked / 2; i++) {
                    *mask_write_ptr++ = 0x00000000;
                }
                if (num_unmasked % 2 == 1) {
                    *mask_write_ptr++ = 0xFF800000;
                    i++;
                }
                for (; i < args.k_chunk_size / 2; i++) {
                    *mask_write_ptr++ = 0xFF80FF80;
                }
                cb_push_back(args.cb_mask, 1);
            }

            // =================================================================
            // KV Cache Multicast (page-level pipelining)
            // =================================================================
            if (is_mcast_sender) {
                volatile tt_l1_ptr uint32_t* mcast_semaphore_ptr =
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(args.mcast_semaphore_addr);

                volatile tt_l1_ptr uint32_t* ncrisc_brisc_sync_curr_ptr =
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(args.ncrisc_brisc_sync_semaphore_addr);
                volatile tt_l1_ptr uint32_t* ncrisc_brisc_sync_next_ptr =
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(args.ncrisc_brisc_sync_semaphore_addr + 4);
                volatile tt_l1_ptr uint32_t* k_write_curr_ptr_shared =
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(args.ncrisc_brisc_sync_semaphore_addr + 8);
                volatile tt_l1_ptr uint32_t* k_write_next_ptr_shared =
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(args.ncrisc_brisc_sync_semaphore_addr + 12);

                volatile tt_l1_ptr uint32_t* receiver_ready_semaphore_ptr =
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(args.receiver_ready_semaphore_addr);

                const uint64_t mcast_noc_addr = get_noc_multicast_addr<MCAST_NOC_INDEX>(
                    args.mcast_start_x, args.mcast_start_y, args.mcast_end_x, args.mcast_end_y, 0);
                const uint64_t mcast_sem_addr = mcast_noc_addr | args.mcast_semaphore_addr;

                noc_semaphore_set(mcast_semaphore_ptr, 1);

                for (uint32_t k_chunk = k_chunk_start; k_chunk < k_chunk_end; k_chunk += args.num_cores_per_head) {
                    DeviceZoneScopedN("mcast-sender-multicast");

                    noc_semaphore_wait_min(ncrisc_brisc_sync_curr_ptr, 1);
                    invalidate_l1_cache();
                    uint32_t page_addr = *k_write_curr_ptr_shared;

                    uint64_t mcast_dest_addr = mcast_noc_addr | page_addr;

                    noc_semaphore_wait(receiver_ready_semaphore_ptr, args.num_mcast_dests);
                    noc_semaphore_set(receiver_ready_semaphore_ptr, 0);

                    noc_async_write_multicast<k_page_size>(
                        page_addr, mcast_dest_addr, k_page_size, args.num_mcast_dests, false, MCAST_NOC_INDEX);

                    for (uint32_t page = 1; page < args.k_num_pages; ++page) {
                        page_addr += k_page_size;
                        mcast_dest_addr = mcast_noc_addr | page_addr;
                        noc_semaphore_wait_min(ncrisc_brisc_sync_curr_ptr, page + 1);
                        noc_async_write_multicast<k_page_size>(
                            page_addr, mcast_dest_addr, k_page_size, args.num_mcast_dests, false, MCAST_NOC_INDEX);
                    }

                    noc_semaphore_set_multicast(
                        args.mcast_semaphore_addr, mcast_sem_addr, args.num_mcast_dests, false, MCAST_NOC_INDEX);
                    noc_async_writes_flushed(MCAST_NOC_INDEX);
                    *ncrisc_brisc_sync_curr_ptr = 0;
                    std::swap(ncrisc_brisc_sync_curr_ptr, ncrisc_brisc_sync_next_ptr);
                    std::swap(k_write_curr_ptr_shared, k_write_next_ptr_shared);
                }
                noc_async_write_barrier(MCAST_NOC_INDEX);
            }

            // =================================================================
            // Tree Reduction
            // =================================================================
            constexpr uint32_t bits_per_step = 1;
            constexpr uint32_t step_mask = (1U << bits_per_step) - 1;

            volatile tt_l1_ptr uint32_t* in0_receiver_semaphore_addr_ptr =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(args.reducer_semaphore_addr);

            uint32_t num_active_s_blocks =
                (k_num_chunks < args.num_cores_per_head) ? k_num_chunks : args.num_cores_per_head;
            bool needs_reduction = (num_active_s_blocks > 1);
            uint32_t cb_ms_in_base_addr = get_write_ptr(args.cb_ms_in);
            uint32_t cb_out_in_base_addr = get_write_ptr(args.cb_out_in);

            if (needs_reduction) {
                for (uint32_t step = 0; step < args.num_tree_reduction_steps; ++step) {
                    DeviceZoneScopedN("tree-reduction-step");
                    uint32_t role_code = args.tree_reduction_info[step * 4 + 0];
                    uint32_t partner_s_block_idx = args.tree_reduction_info[step * 4 + 1];
                    uint32_t partner_x = args.tree_reduction_info[step * 4 + 2];
                    uint32_t partner_y = args.tree_reduction_info[step * 4 + 3];

                    if (role_code != 0 && partner_s_block_idx >= num_active_s_blocks) {
                        continue;
                    }

                    if (role_code == 1) {
                        DeviceZoneScopedN("tree-reduction-sender");
                        cb_wait_front(cb_out_o, out_chunk_tiles);
                        cb_wait_front(args.cb_out_ms, 1);
                        uint64_t output_write_coord = get_noc_addr(partner_x, partner_y, 0, WRITE_NOC_INDEX);
                        uint64_t output_write_addr = output_write_coord | (cb_ms_in_base_addr + step * ms_write_size);

                        noc_async_write<ms_write_size, false, /*posted=*/true>(
                            get_read_ptr(args.cb_out_ms), output_write_addr, ms_write_size, WRITE_NOC_INDEX);

                        output_write_addr = output_write_coord | (cb_out_in_base_addr + step * o_write_size);
                        noc_async_write<o_write_size, false, /*posted=*/true>(
                            get_read_ptr(cb_out_o), output_write_addr, o_write_size, WRITE_NOC_INDEX);

                        uint64_t partner_semaphore_addr = output_write_coord | args.reducer_semaphore_addr;
                        noc_semaphore_inc(
                            partner_semaphore_addr, step_semaphore_inc<bits_per_step>(step), WRITE_NOC_INDEX);

                        noc_async_posted_writes_flushed(WRITE_NOC_INDEX);
                        cb_pop_front(args.cb_out_ms, 1);
                        cb_pop_front(cb_out_o, out_chunk_tiles);
                        noc_async_atomic_barrier(WRITE_NOC_INDEX);
                        break;

                    } else if (role_code == 2) {
                        DeviceZoneScopedN("tree-reduction-receiver");
                        cb_reserve_back(args.cb_ms_in, 1);
                        cb_reserve_back(args.cb_out_in, out_chunk_tiles);
                        while (true) {
                            invalidate_l1_cache();
                            uint32_t sem_val = *in0_receiver_semaphore_addr_ptr;
                            uint8_t step_sem = (sem_val >> step_semaphore_shift<bits_per_step>(step)) & step_mask;
                            if (step_sem >= 1) {
                                break;
                            }
                        }
                        cb_push_back(args.cb_ms_in, 1);
                        cb_push_back(args.cb_out_in, out_chunk_tiles);
                    }
                }
            }

// ====================================================================
// TRISC (Compute)
// ====================================================================
#elif defined(COMPILE_FOR_TRISC)
            constexpr uint32_t DHt = get_named_compile_time_arg_val("DHt");
            constexpr uint32_t vDHt = get_named_compile_time_arg_val("vDHt");
            constexpr uint32_t Sq_chunk_t = get_named_compile_time_arg_val("PNHt");
            constexpr uint32_t Sk_chunk_t = get_named_compile_time_arg_val("Sk_chunk_t");
            constexpr uint32_t scale_fp32 = get_named_compile_time_arg_val("scale_fp32");
            constexpr uint32_t dst_size = get_named_compile_time_arg_val("dst_size");
            constexpr uint32_t cb_q_in = CTArgs::cb_q_in;
            constexpr uint32_t cb_k_in = CTArgs::cb_k_in;
            constexpr uint32_t cb_mask = CTArgs::cb_mask;
            constexpr uint32_t cb_interm_out = CTArgs::cb_interm_out;
            constexpr uint32_t cb_interm_ms = CTArgs::cb_interm_ms;
            constexpr uint32_t cb_out_in = CTArgs::cb_out_in;
            constexpr uint32_t cb_ms_in = CTArgs::cb_ms_in;
            constexpr uint32_t cb_out_o = CTArgs::cb_out_o;
            constexpr uint32_t cb_out_ms = CTArgs::cb_out_ms;
            constexpr uint32_t cb_out_final = CTArgs::cb_out_final;

            constexpr uint32_t q_chunk_tiles = Sq_chunk_t * DHt;
            constexpr uint32_t out_chunk_tiles = Sq_chunk_t * vDHt;

            static_assert(out_chunk_tiles % 2 == 0, "out_chunk_tiles must be even");

            const bool do_reduce = args.do_reduce == 1;
            const bool do_output = args.do_output == 1;
            const bool is_sender_after_reduce = args.is_sender_after_reduce == 1;

            constexpr uint16_t scale_bf16 = scale_fp32 >> 16;

            constexpr bool transpose_k = true;
            constexpr bool transpose_v = false;

            MATH(ckernel::t6_semaphore_init(ckernel::semaphore::FPU_SFPU, 0, 1));
            PACK(ckernel::t6_semaphore_init(SFPU_FPU, 0, 1));
            PACK((llk_math_sfpu_sdpa_reduce_row_init<false, DST_ACCUM_MODE, DataFormat::Float16_b>()));
            reconfig_data_format<false, true>(cb_k_in, cb_q_in);
            pack_reconfig_data_format<true>(cb_out_o);
            PACK(SFPU_TEMPLATE_INIT_KERNEL(exponential, sfpu::exp_init, true, true, scale_fp32, true));
            sdpa_custom_mm_block_init_short<transpose_k>(cb_q_in, cb_k_in, cb_out_o, Sk_chunk_t);

            volatile tt_l1_ptr uint32_t* pos_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(args.pos_addr);
            uint32_t cur_pos = pos_ptr[0];
            auto [k_num_chunks, k_chunk_start, k_chunk_end] = get_runtime_args(
                cur_pos, args.cur_batch, args.core_num_in_reduce, args.num_cores_per_head, args.k_chunk_size);
            if (k_chunk_start == k_chunk_end) {
                return;
            }

            uint32_t num_active_s_blocks =
                (k_num_chunks < args.num_cores_per_head) ? k_num_chunks : args.num_cores_per_head;

            uint32_t num_cores_to_wait = 0;
            for (uint32_t step = 0; step < args.num_tree_reduction_steps; ++step) {
                uint32_t role_code = args.tree_reduction_info[step * 2 + 0];
                uint32_t partner_s_block_idx = args.tree_reduction_info[step * 2 + 1];
                if (role_code == 2 && partner_s_block_idx < num_active_s_blocks) {
                    num_cores_to_wait++;
                }
            }

            constexpr uint32_t packed_tile_size = 8 * 2;
            constexpr uint32_t mm2_dst_offset = 0;
            constexpr uint32_t mm2_dst_tile_offset = mm2_dst_offset / packed_tile_size;
            constexpr uint32_t max_dst_offset = mm2_dst_offset + packed_tile_size * vDHt;
            constexpr uint32_t max_dst_tile_offset = max_dst_offset / packed_tile_size;
            constexpr uint32_t sum_dst_offset = max_dst_offset + 2;
            constexpr uint32_t corr_exp_dst_offset = max_dst_offset + packed_tile_size;
            constexpr uint32_t mm1_dst_offset = corr_exp_dst_offset + packed_tile_size;
            constexpr uint32_t mm1_dst_tile_offset = mm1_dst_offset / packed_tile_size;

            constexpr bool exp_approx_mode = false;

            bool sdpa_output_is_final = do_output && (!do_reduce || num_cores_to_wait == 0);
            uint32_t sdpa_output_cb = 0;
            uint32_t sdpa_ms_cb = 0;
            if (sdpa_output_is_final) {
                sdpa_output_cb = cb_out_final;
                sdpa_ms_cb = cb_out_ms;
            } else if (num_cores_to_wait > 0) {
                sdpa_output_cb = cb_interm_out;
                sdpa_ms_cb = cb_interm_ms;
            } else {
                sdpa_output_cb = cb_out_o;
                sdpa_ms_cb = cb_out_ms;
            }
            uint32_t num_chunks = (k_chunk_end - k_chunk_start + args.num_cores_per_head - 1) / args.num_cores_per_head;
            bool mask_last_chunk = k_chunk_end == k_num_chunks && (cur_pos + 1) % args.k_chunk_size != 0;
            if (mask_last_chunk) {
                cb_wait_front(cb_mask, 1);
            }
            cb_wait_front(cb_q_in, q_chunk_tiles);
            cb_reserve_back(sdpa_output_cb, vDHt);
            cb_reserve_back(sdpa_ms_cb, Sq_chunk_t);
            tile_regs_acquire();
            for (uint32_t chunk = 0; chunk < num_chunks; chunk++) {
                bool last_chunk = chunk == (num_chunks - 1);
                compute_sdpa_chunk<
                    Sk_chunk_t,
                    q_chunk_tiles,
                    out_chunk_tiles,
                    scale_fp32,
                    scale_bf16,
                    transpose_k,
                    transpose_v,
                    packed_tile_size,
                    exp_approx_mode>(
                    cb_q_in,
                    cb_k_in,
                    cb_mask,
                    sdpa_output_cb,
                    mm1_dst_offset,
                    mm2_dst_offset,
                    max_dst_offset,
                    sum_dst_offset,
                    corr_exp_dst_offset,
                    chunk == 0,
                    !sdpa_output_is_final && last_chunk,
                    mask_last_chunk && last_chunk);
            }
            if (!sdpa_output_is_final) {
                PACK(TTI_STALLWAIT(p_stall::STALL_PACK, p_stall::WAIT_SFPU));
                pack_tile(max_dst_tile_offset, sdpa_ms_cb);
                cb_push_back(sdpa_ms_cb, Sq_chunk_t);
            } else {
                compute_sdpa_recip<out_chunk_tiles, exp_approx_mode, scale_bf16>(
                    cb_q_in, sum_dst_offset, corr_exp_dst_offset, mm2_dst_offset);
            }
            for (uint32_t i = 0; i < out_chunk_tiles; i += 2) {
                PACK(t6_semaphore_wait_on_zero<p_stall::STALL_PACK>(semaphore::FPU_SFPU));
                pack_tile(mm2_dst_tile_offset + i, sdpa_output_cb);
                pack_tile(mm2_dst_tile_offset + i + 1, sdpa_output_cb);
                PACK(t6_semaphore_get<p_stall::PACK>(semaphore::FPU_SFPU));
            }
            cb_push_back(sdpa_output_cb, out_chunk_tiles);
            tile_regs_commit();
            tile_regs_release();
            sdpa_custom_mm_block_uninit();
            MATH(t6_semaphore_wait_on_max<p_stall::STALL_SFPU>(semaphore::FPU_SFPU));

            static_assert(vDHt % dst_size == 0, "vDHt must be divisible by dst_size");
            constexpr uint32_t num_blocks = vDHt / dst_size;
            constexpr uint32_t block_size = vDHt / num_blocks;

            if (do_reduce) {
                exp_tile_init<exp_approx_mode, false, scale_fp32>();
                if (num_cores_to_wait > 0) {
                    reconfig_data_format_srca<false, true>(cb_ms_in);
                    for (uint32_t i = 0; i < num_cores_to_wait - 1; i++) {
                        sdpa_tail<exp_approx_mode, false, block_size, num_blocks, scale_fp32, VectorMode::C>(
                            cb_ms_in, cb_interm_ms, cb_interm_ms, cb_out_in, cb_interm_out, cb_interm_out);
                    }
                    if (is_sender_after_reduce) {
                        sdpa_tail<exp_approx_mode, false, block_size, num_blocks, scale_fp32, VectorMode::C>(
                            cb_ms_in, cb_interm_ms, cb_out_ms, cb_out_in, cb_interm_out, cb_out_o);
                    } else {
                        sdpa_tail<exp_approx_mode, true, block_size, num_blocks, scale_fp32, VectorMode::C>(
                            cb_ms_in, cb_interm_ms, cb_out_ms, cb_out_in, cb_interm_out, cb_out_final);
                    }
                }
            }

            cb_pop_front(cb_q_in, q_chunk_tiles);
            if (mask_last_chunk) {
                cb_pop_front(cb_mask, 1);
            }
#endif
        }
    };
};

}  // namespace deepseek_b1_ops
