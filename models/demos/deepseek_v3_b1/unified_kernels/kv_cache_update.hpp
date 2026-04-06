// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "kernel_op_api.hpp"

#if defined(COMPILE_FOR_BRISC)
#include "api/dataflow/dataflow_api.h"
#elif defined(COMPILE_FOR_NCRISC)
#include "api/dataflow/dataflow_api.h"
#include "mcast.hpp"
#elif defined(COMPILE_FOR_TRISC)
#include "api/compute/compute_kernel_api.h"
#include "api/compute/pack_untilize.h"
#include "api/compute/tilize.h"

#endif

namespace deepseek_b1_ops {

// ============================================================================
// KVCacheUpdate micro-op
//
// Computes: Update existing KV Cache with 1x576 new cache
// assumes one core with 1x512 NOPE cache and 2 cores each with 1x32 ROPE cache
//
// BRISC: streams existing KV cache pages from DRAM into kv_cache_input_cb
// NCRISC: waits on compute (untilize), patches new data, waits on compute
//         (tilize), writes updated pages back to DRAM, signals cache ready
// TRISC: untilize input_cb -> intermed_cb, tilize intermed_cb -> output_cb
// ============================================================================
struct KVCacheUpdate {
    // ========================================================================
    // Compile-time args structs - only what MUST be compile-time
    // (used as template parameters or in constexpr expressions)
    // ========================================================================

    // Reader CTArgs: none needed
    struct ReaderCTArgs {};

    // Writer CTArgs: none needed
    struct WriterCTArgs {};

    // Compute CTArgs: none needed
    struct ComputeCTArgs {};

    // ========================================================================
    // Runtime args structs - different layout per RISC
    // CB indices etc. filled in kernel via get_named_compile_time_arg_val
    // ========================================================================
    struct WriterArgs {
        static constexpr uint32_t MAX_MLA_CORES_PER_HEAD = 8;
        uint32_t kv_cache_buffer_base_addr;
        uint32_t local_cur_pos;
        uint32_t slot_id;
        uint32_t kv_cache_intermed_cb;
        uint32_t kv_cache_intermed_sync_cb;
        uint32_t kv_cache_output_cb;
        uint32_t kv_rmsnorm_output_cb;
        uint32_t krope_output_cb;
        uint32_t grid_start_y;
        uint32_t kv_cache_cur_pos_ready_semaphore_addr;
        uint32_t k_chunk_size;
        uint32_t num_cores_per_head;
        uint32_t mla_sender_noc_x[MAX_MLA_CORES_PER_HEAD];
        uint32_t mla_sender_noc_y[MAX_MLA_CORES_PER_HEAD];
    };
    struct ReaderArgs {
        uint32_t kv_cache_buffer_base_addr;
        uint32_t local_cur_pos;
        uint32_t slot_id;
        uint32_t kv_cache_input_cb;
        uint32_t grid_start_y;
    };
    struct ComputeArgs {
        uint32_t kv_cache_input_cb;
        uint32_t kv_cache_output_cb;
        uint32_t kv_cache_intermed_cb;
        uint32_t kv_cache_intermed_sync_cb;
    };

    using RTArgs = unified_kernels::SelectByRISCV<WriterArgs, ReaderArgs, ComputeArgs>;

    // ========================================================================
    // Op - full KV cache update (owning device)
    // ========================================================================
    template <bool IsNopeCore, bool IsRopeCore>
    class Op {
    public:
        void operator()([[maybe_unused]] const RTArgs& args) { impl(args); }

        void set_pos_and_slot(
            [[maybe_unused]] RTArgs& args, [[maybe_unused]] uint32_t local_cur_pos, [[maybe_unused]] uint32_t slot_id) {
#if defined(COMPILE_FOR_BRISC) || defined(COMPILE_FOR_NCRISC)
            args.local_cur_pos = local_cur_pos;
            args.slot_id = slot_id;
#endif
        }

        void signal_cache_ready([[maybe_unused]] const RTArgs& args) {
#if defined(COMPILE_FOR_NCRISC)
            if constexpr (IsRopeCore || IsNopeCore) {
                static_assert(noc_mode == DM_DYNAMIC_NOC, "KV Cache Update only supports DM_DYNAMIC_NOC");
                constexpr uint8_t WRITE_NOC = 0;
                uint32_t target_core_idx = (args.local_cur_pos / args.k_chunk_size) % args.num_cores_per_head;
                uint64_t sem_noc_addr = get_noc_addr(
                    args.mla_sender_noc_x[target_core_idx],
                    args.mla_sender_noc_y[target_core_idx],
                    args.kv_cache_cur_pos_ready_semaphore_addr,
                    WRITE_NOC);
                noc_semaphore_inc(sem_noc_addr, 1, WRITE_NOC);
                noc_async_atomic_barrier(WRITE_NOC);
            }
#endif
        }

    private:
        void impl([[maybe_unused]] const RTArgs& args) {
#if defined(COMPILE_FOR_BRISC) || defined(COMPILE_FOR_NCRISC)
            if constexpr (IsRopeCore || IsNopeCore) {
                constexpr uint32_t PAGE_SIZE = 1088;
                constexpr uint32_t PAGES_PER_BLOCK = 18;
                constexpr uint32_t CACHES_PER_BLOCK = 32;
                constexpr uint32_t nope_num_pages = 16;
                constexpr uint32_t CHUNK_SIZE = IsRopeCore ? 1 : 8;
                constexpr uint32_t NUM_CHUNKS = IsRopeCore ? 1 : 2;
                constexpr uint32_t pages_per_slot = get_named_compile_time_arg_val("kv_cache_pages_per_slot");

                uint32_t cur_pos = args.local_cur_pos;

                constexpr auto k_args = TensorAccessorArgs<0>();
                auto kv_tensor_accessor = TensorAccessor(k_args, args.kv_cache_buffer_base_addr, PAGE_SIZE);

                uint32_t kv_cache_page_id_start =
                    args.slot_id * pages_per_slot + cur_pos / CACHES_PER_BLOCK * PAGES_PER_BLOCK;
                if constexpr (IsRopeCore) {
                    uint32_t grid_offset_pages = 1 * (get_absolute_logical_y() - args.grid_start_y);
                    kv_cache_page_id_start += grid_offset_pages;
                    kv_cache_page_id_start += nope_num_pages;
                }

#if defined(COMPILE_FOR_BRISC)
                uint32_t kv_cache_input_cb = args.kv_cache_input_cb;

                for (uint32_t chunk = 0; chunk < NUM_CHUNKS; chunk++) {
                    DeviceZoneScopedN("READ_FROM_DRAM");
                    uint32_t tile_offset = chunk * CHUNK_SIZE;
                    cb_reserve_back(kv_cache_input_cb, CHUNK_SIZE);
                    uint32_t cb_addr = get_write_ptr(kv_cache_input_cb);
                    for (uint32_t i = 0; i < CHUNK_SIZE; i++) {
                        noc_async_read_page(kv_cache_page_id_start + tile_offset + i, kv_tensor_accessor, cb_addr);
                        cb_addr += kv_tensor_accessor.get_aligned_page_size();
                    }
                    noc_async_read_barrier();
                    cb_push_back(kv_cache_input_cb, CHUNK_SIZE);
                }
#elif defined(COMPILE_FOR_NCRISC)
                uint32_t kv_cache_intermed_cb = args.kv_cache_intermed_cb;
                uint32_t kv_cache_intermed_sync_cb = args.kv_cache_intermed_sync_cb;
                uint32_t kv_cache_output_cb = args.kv_cache_output_cb;
                uint32_t new_cache_cb = IsRopeCore ? args.krope_output_cb : args.kv_rmsnorm_output_cb;
                uint32_t offset_in_page = cur_pos % CACHES_PER_BLOCK;

                constexpr uint32_t num_bytes_per_chunk = IsRopeCore ? 64 : (CHUNK_SIZE * 32 * 2);

                cb_wait_front(new_cache_cb, 1);
                uint32_t src_addr = get_read_ptr(new_cache_cb);
                uint32_t write_addr_offset = offset_in_page * num_bytes_per_chunk;

                for (uint32_t chunk = 0; chunk < NUM_CHUNKS; chunk++) {
                    {
                        DeviceZoneScopedN("WAIT_UNTILIZE");
                        cb_reserve_back(kv_cache_intermed_sync_cb, CHUNK_SIZE);
                        cb_wait_front(kv_cache_intermed_cb, CHUNK_SIZE);
                    }

                    {
                        DeviceZoneScopedN("UPDATE_NEW_CACHE");
                        uint32_t write_addr = get_read_ptr(kv_cache_intermed_cb) + write_addr_offset;
                        noc_async_write(src_addr, get_noc_addr(write_addr), num_bytes_per_chunk);
                        noc_async_write_barrier();
                    }

                    cb_push_back(kv_cache_intermed_sync_cb, CHUNK_SIZE);
                    cb_pop_front(kv_cache_intermed_cb, CHUNK_SIZE);

                    src_addr += num_bytes_per_chunk;
                }

                cb_pop_front(new_cache_cb, 1);

                // Phase 2: Wait for tilize to complete, write to DRAM
                for (uint32_t chunk = 0; chunk < NUM_CHUNKS; chunk++) {
                    {
                        DeviceZoneScopedN("WAIT_TILIZE");
                        cb_wait_front(kv_cache_output_cb, CHUNK_SIZE);
                    }

                    {
                        DeviceZoneScopedN("WRITE_TO_DRAM");
                        uint32_t tile_offset = chunk * CHUNK_SIZE;
                        uint32_t cb_addr = get_read_ptr(kv_cache_output_cb);
                        for (uint32_t i = 0; i < CHUNK_SIZE; i++) {
                            noc_async_write_page(kv_cache_page_id_start + tile_offset + i, kv_tensor_accessor, cb_addr);
                            cb_addr += kv_tensor_accessor.get_aligned_page_size();
                        }
                        noc_async_write_barrier();
                        cb_pop_front(kv_cache_output_cb, CHUNK_SIZE);
                    }
                }
#endif
            }
#elif defined(COMPILE_FOR_TRISC)
            if constexpr (IsNopeCore || IsRopeCore) {
                uint32_t kv_cache_intermed_cb = args.kv_cache_intermed_cb;
                uint32_t kv_cache_intermed_sync_cb = args.kv_cache_intermed_sync_cb;
                uint32_t kv_cache_input_cb = args.kv_cache_input_cb;
                uint32_t kv_cache_output_cb = args.kv_cache_output_cb;
                constexpr uint32_t kv_cache_num_tiles = IsRopeCore ? 1 : 16;
                constexpr uint32_t CHUNK_SIZE = IsRopeCore ? 1 : 8;
                constexpr uint32_t NUM_CHUNKS = kv_cache_num_tiles / CHUNK_SIZE;

                // Phase 1: Untilize into intermed_cb, pushing each chunk of CHUNK_SIZE
                reconfig_data_format_srca<false, true>(kv_cache_input_cb);
                pack_reconfig_data_format<true>(kv_cache_intermed_cb);
                pack_untilize_init<CHUNK_SIZE, CHUNK_SIZE>(kv_cache_input_cb, kv_cache_intermed_cb);
                for (uint32_t chunk = 0; chunk < NUM_CHUNKS; chunk++) {
                    cb_reserve_back(kv_cache_intermed_cb, CHUNK_SIZE);
                    cb_wait_front(kv_cache_input_cb, CHUNK_SIZE);
                    pack_untilize_block<CHUNK_SIZE, CHUNK_SIZE>(kv_cache_input_cb, 1, kv_cache_intermed_cb, 0);
                    cb_push_back(kv_cache_intermed_cb, CHUNK_SIZE);
                    cb_pop_front(kv_cache_input_cb, CHUNK_SIZE);
                }
                pack_untilize_uninit(kv_cache_intermed_cb);

                // Phase 2: Tilize each chunk after NCRISC signals via sync CB
                reconfig_data_format_srca<false, true>(kv_cache_intermed_sync_cb);
                pack_reconfig_data_format<true>(kv_cache_output_cb);
                tilize_init(kv_cache_intermed_sync_cb, CHUNK_SIZE, kv_cache_output_cb);
                for (uint32_t chunk = 0; chunk < NUM_CHUNKS; chunk++) {
                    cb_reserve_back(kv_cache_output_cb, CHUNK_SIZE);
                    cb_wait_front(kv_cache_intermed_sync_cb, CHUNK_SIZE);
                    tilize_block(kv_cache_intermed_sync_cb, CHUNK_SIZE, kv_cache_output_cb);
                    cb_push_back(kv_cache_output_cb, CHUNK_SIZE);
                    cb_pop_front(kv_cache_intermed_sync_cb, CHUNK_SIZE);
                }
                tilize_uninit(kv_cache_intermed_sync_cb, kv_cache_output_cb);
            }
#endif
        }
    };  // class Op

};  // struct KVCacheUpdate

}  // namespace deepseek_b1_ops
