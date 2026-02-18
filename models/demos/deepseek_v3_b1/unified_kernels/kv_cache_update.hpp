// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "kernel_op_api.hpp"

#if defined(COMPILE_FOR_BRISC)
#include "api/dataflow/dataflow_api.h"
#elif defined(COMPILE_FOR_NCRISC)
#include "api/dataflow/dataflow_api.h"
#elif defined(COMPILE_FOR_TRISC)
#include "api/compute/compute_kernel_api.h"
#include "api/compute/pack_untilize.h"
#include "api/compute/tilize.h"

#endif

namespace deepseek_b1_ops {

// ============================================================================
// RMSNorm micro-op
//
// Computes: Update existing KV Cache with 1x576 new cache
// assumes one core with 1x512 NOPE cache and 2 cores each with 1x32 ROPE cache
//
// ============================================================================
struct KVCacheUpdate {
    // ========================================================================
    // Compile-time args structs - only what MUST be compile-time
    // (used as template parameters or in constexpr expressions)
    // ========================================================================

    // Reader CTArgs:none needed
    struct ReaderCTArgs {};

    // Writer CTArgs: none needed
    struct WriterCTArgs {};

    // Compute CTArgs: none needed
    struct ComputeCTArgs {};

    // ========================================================================
    // Runtime args structs - different layout per RISC
    // CB indices etc. filled in kernel via get_named_compile_time_arg_val
    // ========================================================================
    struct ReaderArgs {};
    struct WriterArgs {
        uint32_t kv_cache_buffer_base_addr;
        uint32_t position_id;
        uint32_t kv_cache_input_cb;
        uint32_t kv_cache_intermed_cb;
        uint32_t kv_cache_output_cb;
        uint32_t kv_rmsnorm_output_cb;
        uint32_t krope_output_cb;
        uint32_t grid_start_y;
    };
    struct ComputeArgs {
        uint32_t kv_cache_input_cb;
        uint32_t kv_cache_output_cb;
        uint32_t kv_cache_intermed_cb;
    };

    using RTArgs = unified_kernels::SelectByRISCV<ReaderArgs, WriterArgs, ComputeArgs>;

    // ========================================================================
    // Op - the actual operation, IsActiveCore
    // ========================================================================
    template <bool IsNopeCore, bool IsRopeCore>
    class Op {
    public:
        void operator()([[maybe_unused]] const RTArgs& args) { impl(args); }

    private:
        void impl([[maybe_unused]] const RTArgs& args) {
#if defined(COMPILE_FOR_BRISC)
            if constexpr (IsRopeCore || IsNopeCore) {
                uint32_t kv_cache_intermed_cb = args.kv_cache_intermed_cb;
                uint32_t kv_cache_input_cb = args.kv_cache_input_cb;
                uint32_t kv_cache_output_cb = args.kv_cache_output_cb;
                uint32_t new_cache_cb = IsRopeCore ? args.krope_output_cb : args.kv_rmsnorm_output_cb;
                constexpr uint32_t PAGE_SIZE = 1088;
                constexpr uint32_t PAGES_PER_BLOCK = 18;
                constexpr uint32_t CACHES_PER_BLOCK = 32;
                constexpr uint32_t nope_num_pages = 16;  // Offset in 1x576 for rope component
                constexpr uint32_t kv_cache_num_tiles = IsRopeCore ? 1 : 16;

                constexpr auto k_args = TensorAccessorArgs<0>();
                auto kv_tensor_accessor = TensorAccessor(k_args, args.kv_cache_buffer_base_addr, PAGE_SIZE);

                // This op needs to update 18 pages starting from kv_cache_page_id_start
                // If args.position_id % CACHES_PER_BLOCK != 0, there is an offset into the rows
                uint32_t kv_cache_page_id_start = args.position_id / CACHES_PER_BLOCK * PAGES_PER_BLOCK;
                uint32_t offset_in_page = args.position_id % CACHES_PER_BLOCK;
                uint32_t num_bytes_per_core = IsRopeCore ? 64 : 1024;

                if (IsRopeCore) {
                    uint32_t grid_offset_pages = 1 * (get_absolute_logical_y() - args.grid_start_y);
                    kv_cache_page_id_start += grid_offset_pages;
                    kv_cache_page_id_start += nope_num_pages;
                }

                cb_reserve_back(kv_cache_input_cb, kv_cache_num_tiles);
                uint32_t cb_addr = get_write_ptr(kv_cache_input_cb);
                for (uint32_t i = 0; i < kv_cache_num_tiles; i++) {
                    noc_async_read_page(kv_cache_page_id_start + i, kv_tensor_accessor, cb_addr);
                    cb_addr += kv_tensor_accessor.page_size;
                }
                    noc_async_read_barrier();

                    cb_push_back(kv_cache_input_cb, kv_cache_num_tiles);

                    // wait for unpacker to untilize
                    cb_wait_front(kv_cache_intermed_cb, kv_cache_num_tiles);

                    // 2. Wait for new cache data and update into kv_cache_intermed_cb
                    cb_wait_front(new_cache_cb, 1);

                    uint32_t write_addr = get_read_ptr(kv_cache_intermed_cb) + offset_in_page * num_bytes_per_core;
                    uint32_t new_cache_addr = get_read_ptr(new_cache_cb);
                    // Local copy data from new cache to intermed, 1x512 for nope, 1x64 for rope (1x32 per core)
                    {
                        uint32_t words_per_face = (num_bytes_per_core >> 2);
                        volatile tt_l1_ptr uint32_t* src =
                            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(new_cache_addr);
                        volatile tt_l1_ptr uint32_t* dst = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(write_addr);
                        for (uint32_t i = 0; i < words_per_face; ++i) {
                            dst[i] = src[i];
                        }
                    }
                    cb_pop_front(new_cache_cb, 1);
                    cb_push_back(kv_cache_intermed_cb, 1);

                    // 3. Wait for TRISC to finish tilize into kv_cache_output_cb and write out to DRAM
                    cb_wait_front(kv_cache_output_cb, kv_cache_num_tiles);

                    cb_addr = get_read_ptr(kv_cache_output_cb);
                    for (uint32_t i = 0; i < kv_cache_num_tiles; i++) {
                        noc_async_write_page(kv_cache_page_id_start + i, kv_tensor_accessor, cb_addr);
                        cb_addr += kv_tensor_accessor.page_size;
                    }
                    noc_async_write_barrier();
                    cb_pop_front(kv_cache_output_cb, kv_cache_num_tiles);
            }
#elif defined(COMPILE_FOR_TRISC)
            if constexpr (IsNopeCore || IsRopeCore) {
                uint32_t kv_cache_intermed_cb = args.kv_cache_intermed_cb;
                uint32_t kv_cache_input_cb = args.kv_cache_input_cb;
                uint32_t kv_cache_output_cb = args.kv_cache_output_cb;
                constexpr uint32_t kv_cache_num_tiles = IsRopeCore ? 1 : 16;
                constexpr uint32_t full_ct_dim = IsRopeCore ? 1 : 16;
                constexpr uint32_t block_ct_dim = IsRopeCore ? 1 : 8;

                reconfig_data_format_srca<false, true>(kv_cache_input_cb);
                pack_reconfig_data_format<true>(kv_cache_intermed_cb);
                cb_wait_front(kv_cache_input_cb, kv_cache_num_tiles);
                cb_reserve_back(kv_cache_intermed_cb, kv_cache_num_tiles + 1);
                pack_untilize_init<block_ct_dim, full_ct_dim>(kv_cache_input_cb, kv_cache_intermed_cb);
                pack_untilize_block<block_ct_dim, full_ct_dim>(kv_cache_input_cb, 1, kv_cache_intermed_cb, 0);
                cb_pop_front(kv_cache_input_cb, block_ct_dim);  // consume first 8 so second block reads tiles 8-15
                pack_untilize_block<block_ct_dim, full_ct_dim>(kv_cache_input_cb, 1, kv_cache_intermed_cb, 1);
                pack_untilize_uninit(kv_cache_intermed_cb);

                cb_push_back(kv_cache_intermed_cb, kv_cache_num_tiles);
                cb_pop_front(kv_cache_input_cb, kv_cache_num_tiles - block_ct_dim);  // pop remaining 8

                cb_wait_front(kv_cache_intermed_cb, kv_cache_num_tiles + 1);
                cb_reserve_back(kv_cache_output_cb, kv_cache_num_tiles);

                reconfig_data_format_srca<false, true>(kv_cache_intermed_cb);
                pack_reconfig_data_format<true>(kv_cache_output_cb);
                tilize_init(kv_cache_intermed_cb, kv_cache_num_tiles, kv_cache_output_cb);
                tilize_block(kv_cache_intermed_cb, kv_cache_num_tiles, kv_cache_output_cb);
                tilize_uninit(kv_cache_intermed_cb, kv_cache_output_cb);
                cb_push_back(kv_cache_output_cb, kv_cache_num_tiles);
                cb_pop_front(kv_cache_intermed_cb, kv_cache_num_tiles);
            }
#endif
        }
    };  // class Op

};  // struct KVCacheUpdate

}  // namespace deepseek_b1_ops
