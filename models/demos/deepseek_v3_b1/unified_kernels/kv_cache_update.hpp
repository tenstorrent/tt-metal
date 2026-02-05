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
    struct ReaderArgs {
        uint32_t kv_cache_buffer_base_addr;
        uint32_t kv_rmsnorm_output_cb;
        uint32_t krope_output_cb;
    };
    struct WriterArgs {
        uint32_t kv_cache_buffer_base_addr;
        uint32_t position_id;
        uint32_t kv_cache_input_cb;
        uint32_t kv_cache_intermed_cb;
        uint32_t kv_cache_output_cb;
        uint32_t kv_rmsnorm_output_cb;
        uint32_t krope_output_cb;
    };
    struct ComputeArgs {
        uint32_t kv_cache_num_tiles;
        uint32_t kv_cache_input_cb;
        uint32_t kv_cache_output_cb;
        uint32_t kv_cache_intermed_cb;
        uint32_t kv_rmsnorm_output_cb;
        uint32_t krope_output_cb;
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
                uint32_t krope_output_cb = args.krope_output_cb;
                uint32_t kv_cache_intermed_cb = args.kv_cache_intermed_cb;
                uint32_t kv_cache_input_cb = args.kv_cache_input_cb;
                uint32_t kv_cache_output_cb = args.kv_cache_output_cb;
                constexpr uint32_t PAGE_SIZE = 1088;
                constexpr uint32_t PAGES_PER_BLOCK = 18;
                constexpr uint32_t CACHES_PER_BLOCK = 32;
                constexpr uint32_t nope_num_pages = 16;

                constexpr auto k_args = TensorAccessorArgs<0>();
                auto kv_tensor_accessor = TensorAccessor(k_args, args.kv_cache_buffer_base_addr, PAGE_SIZE);

                // This op needs to update 18 pages starting from kv_cache_page_id_start
                // If args.position_id % CACHES_PER_BLOCK != 0, there is an offset into the rows
                uint32_t kv_cache_page_id_start = args.position_id / CACHES_PER_BLOCK * PAGES_PER_BLOCK;
                uint32_t offset_in_page = args.position_id % CACHES_PER_BLOCK;
                if constexpr (IsRopeCore) {
                    constexpr uint32_t rope_num_bytes_per_core = 64;  // 1x32 float 16
                    constexpr uint32_t bytes_per_datum = 2;
                    constexpr uint32_t kv_cache_num_tiles = 1;

                    uint32_t grid_offset_pages = 1 * (get_absolute_logical_y() - 8);
                    uint32_t rope_page_id = kv_cache_page_id_start + grid_offset_pages + nope_num_pages;
                    // 1. Read in data from DRAM to kv_cache_input_cb
                    cb_reserve_back(kv_cache_input_cb, kv_cache_num_tiles);
                    uint32_t readback_addr = get_write_ptr(kv_cache_input_cb);
                    for (uint32_t i = 0; i < kv_cache_num_tiles; i++) {
                        uint32_t cb_addr = get_write_ptr(kv_cache_input_cb);
                        noc_async_read_page(rope_page_id, kv_tensor_accessor, cb_addr);
                    }
                    noc_async_read_barrier();
                    cb_push_back(kv_cache_input_cb, kv_cache_num_tiles);

                    // wait for unpacker to untilize
                    cb_wait_front(kv_cache_intermed_cb, kv_cache_num_tiles);

                    // 2. Wait for new cache data and update into kv_cache_intermed_cb
                    cb_wait_front(krope_output_cb, 1);
                    // valid new rope cache from krope_output_cb
                    // calculate offset in tile
                    uint32_t write_addr = get_read_ptr(kv_cache_intermed_cb) + offset_in_page * rope_num_bytes_per_core;
                    uint32_t new_rope_cache_addr = get_read_ptr(krope_output_cb);
                    // Local copy: 64 bytes (1..32 bfloat16) from new_rope_cache to intermed.
                    // Untilized tile layout uses a stride: first 32 bytes at write_addr, next 32 at write_addr+64.
                    {
                        constexpr uint32_t words_per_core = (rope_num_bytes_per_core >> 2);  // 8 uint32_t per 32 bytes
                        volatile tt_l1_ptr uint32_t* src =
                            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(new_rope_cache_addr);
                        volatile tt_l1_ptr uint32_t* dst = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(write_addr);
                        for (uint32_t i = 0; i < words_per_core; ++i) {
                            dst[i] = src[i];
                        }
                    }
                    cb_pop_front(krope_output_cb, 1);
                    cb_push_back(kv_cache_intermed_cb, 1);

                    // 3. Wait for TRISC to finish tilize into kv_cache_output_cb and write out to DRAM
                    cb_wait_front(kv_cache_output_cb, kv_cache_num_tiles);
                    noc_async_write_page(rope_page_id, kv_tensor_accessor, get_read_ptr(kv_cache_output_cb));
                    noc_async_write_barrier();
                    cb_pop_front(kv_cache_output_cb, kv_cache_num_tiles);
                }
                if constexpr (IsNopeCore) {
                    uint32_t kv_rmsnorm_output_cb = args.kv_rmsnorm_output_cb;
                    constexpr uint32_t kv_cache_num_tiles = 16;
                    constexpr uint32_t nope_num_bytes_per_core = 1024;  // 16x32 float 16
                    uint32_t kv_cache_input_cb = args.kv_cache_input_cb;
                    uint32_t kv_cache_intermed_cb = args.kv_cache_intermed_cb;
                    uint32_t kv_cache_output_cb = args.kv_cache_output_cb;
                    cb_reserve_back(kv_cache_input_cb, kv_cache_num_tiles);
                    uint32_t readback_addr = get_write_ptr(kv_cache_input_cb);
                    uint32_t cb_addr = get_write_ptr(kv_cache_input_cb);
                    for (uint32_t i = 0; i < kv_cache_num_tiles; i++) {
                        noc_async_read_page(i, kv_tensor_accessor, cb_addr);
                        cb_addr += kv_tensor_accessor.page_size;
                    }
                    noc_async_read_barrier();
                    cb_push_back(kv_cache_input_cb, kv_cache_num_tiles);

                    // wait for unpacker to untilize
                    cb_wait_front(kv_cache_intermed_cb, kv_cache_num_tiles);

                    // 2. Wait for new cache data and update into kv_cache_intermed_cb
                    cb_wait_front(kv_rmsnorm_output_cb, 1);
                    uint32_t write_addr = get_read_ptr(kv_cache_intermed_cb) + offset_in_page * nope_num_bytes_per_core;
                    uint32_t new_nope_cache_addr = get_read_ptr(kv_rmsnorm_output_cb);
                    uint32_t bytes_per_face = 512;
                    // Local copy: 1024 bytes (1..512 bfloat16) from new_nope_cache to intermed.
                    // Untilized tile layout uses a stride: first 32 bytes at write_addr, next 32 at write_addr+64.
                    {
                        uint32_t words_per_face = (bytes_per_face >> 2);  // 8 uint32_t per 32 bytes
                        volatile tt_l1_ptr uint32_t* src =
                            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(new_nope_cache_addr);
                        volatile tt_l1_ptr uint32_t* dst = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(write_addr);
                        for (uint32_t rows_per_cache = 0; rows_per_cache < 16; rows_per_cache++) {
                            for (uint32_t i = 0; i < 8; i++) {
                                dst[i] = src[i];
                                dst[i + 8] = src[i + 128];
                            }
                            dst += 16;
                            src += 8;
                        }
                    }
                    cb_pop_front(kv_rmsnorm_output_cb, 1);
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
            }
#elif defined(COMPILE_FOR_TRISC)
            if constexpr (IsRopeCore) {
                constexpr uint32_t kv_cache_num_tiles = 1;
                uint32_t krope_output_cb = args.krope_output_cb;
                uint32_t kv_cache_intermed_cb = args.kv_cache_intermed_cb;
                uint32_t kv_cache_input_cb = args.kv_cache_input_cb;
                uint32_t kv_cache_output_cb = args.kv_cache_output_cb;
                // One full 32x32 bfloat8 tile: block_ct_dim=1, full_ct_dim=1
                constexpr uint32_t full_ct_dim = 1;
                constexpr uint32_t block_ct_dim = 1;
                compute_kernel_hw_startup(kv_cache_input_cb, kv_cache_output_cb, kv_cache_intermed_cb);
                cb_wait_front(kv_cache_input_cb, kv_cache_num_tiles);
                cb_reserve_back(
                    kv_cache_intermed_cb, kv_cache_num_tiles + 1);  // one extra for ncrisc to fill in new data

                pack_untilize_init<block_ct_dim, full_ct_dim>(kv_cache_input_cb, kv_cache_intermed_cb);
                pack_untilize_block<block_ct_dim, full_ct_dim>(kv_cache_input_cb, 1, kv_cache_intermed_cb, 0);
                pack_untilize_uninit(kv_cache_intermed_cb);
                cb_pop_front(kv_cache_input_cb, kv_cache_num_tiles);
                cb_push_back(kv_cache_intermed_cb, kv_cache_num_tiles);

                cb_wait_front(kv_cache_intermed_cb, kv_cache_num_tiles + 1);
                cb_reserve_back(kv_cache_output_cb, kv_cache_num_tiles);

                UNPACK(reconfig_data_format_srca(kv_cache_intermed_cb));
                PACK((llk_pack_reconfig_data_format<DST_ACCUM_MODE, true>(kv_cache_output_cb)));
                tilize_init(kv_cache_intermed_cb, kv_cache_num_tiles, kv_cache_output_cb);
                tilize_block(kv_cache_intermed_cb, kv_cache_num_tiles, kv_cache_output_cb);
                // tilize_uninit(kv_cache_intermed_cb, kv_cache_output_cb);
                cb_push_back(kv_cache_output_cb, kv_cache_num_tiles);
                cb_pop_front(kv_cache_intermed_cb, kv_cache_num_tiles);
            }
            if constexpr (IsNopeCore) {
                uint32_t kv_rmsnorm_output_cb = args.kv_rmsnorm_output_cb;
                uint32_t kv_cache_intermed_cb = args.kv_cache_intermed_cb;
                uint32_t kv_cache_input_cb = args.kv_cache_input_cb;
                uint32_t kv_cache_output_cb = args.kv_cache_output_cb;
                uint32_t kv_cache_num_tiles = args.kv_cache_num_tiles;
                constexpr uint32_t full_ct_dim = 16;
                constexpr uint32_t block_ct_dim = 8;
                compute_kernel_hw_startup(kv_cache_input_cb, kv_cache_output_cb, kv_cache_intermed_cb);
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

                UNPACK(reconfig_data_format_srca(kv_cache_intermed_cb));
                PACK((llk_pack_reconfig_data_format<DST_ACCUM_MODE, true>(kv_cache_output_cb)));
                tilize_init(kv_cache_intermed_cb, kv_cache_num_tiles, kv_cache_output_cb);
                tilize_block(kv_cache_intermed_cb, kv_cache_num_tiles, kv_cache_output_cb);
                cb_push_back(kv_cache_output_cb, kv_cache_num_tiles);
                cb_pop_front(kv_cache_intermed_cb, kv_cache_num_tiles);
            }
#endif
        }
    };  // class Op

};  // struct KVCacheUpdate

}  // namespace deepseek_b1_ops
