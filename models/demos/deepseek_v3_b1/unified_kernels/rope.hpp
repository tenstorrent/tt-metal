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
#include "api/compute/matmul.h"
#include "api/compute/bcast.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/reg_api.h"
#endif

namespace deepseek_b1_ops {

// ============================================================================
// RoPE (Rotary Position Embedding) micro-op
//
// Computes: output = (input * cos) + (rotate_half(input) * sin)
// where rotate_half(input) = input @ trans_mat
//
// CB States:
//   NCRISC: Signals sharded input CBs are ready (trans_mat, sin, cos)
//   BRISC: No-op
//   TRISC (Compute): Performs the RoPE computation
// ============================================================================
struct Rope {
    // ========================================================================
    // Compile-time args structs - different layout per RISC
    // ========================================================================

    // Reader CTArgs (NCRISC): Wt/Ht for dimensions, TotalWt/StartTileOffset for DRAM interleaved addressing
    template <
        uint32_t Wt_,
        uint32_t Ht_,
        uint32_t CosSinPageSize_ = 64,
        uint32_t TotalWt_ = 2,
        uint32_t StartTileOffset_ = 0>
    struct ReaderCTArgs {
        static constexpr uint32_t Wt = Wt_;  // head_dim in tiles (per core)
        static constexpr uint32_t Ht = Ht_;  // num_heads per core
        static constexpr uint32_t cos_sin_page_size = CosSinPageSize_;
        static constexpr uint32_t total_Wt = TotalWt_;                   // total width tiles per row in DRAM tensor
        static constexpr uint32_t start_tile_offset = StartTileOffset_;  // this core's first tile in row
    };

    // Writer CTArgs (BRISC): none
    struct WriterCTArgs {};

    // Compute CTArgs (TRISC): Wt and Ht as template parameters
    template <uint32_t Wt_, uint32_t Ht_>
    struct ComputeCTArgs {
        static constexpr uint32_t Wt = Wt_;  // head_dim in tiles
        static constexpr uint32_t Ht = Ht_;  // num_heads per core
    };

    // ========================================================================
    // Runtime args structs - different layout per RISC
    // ========================================================================

    // Reader args (NCRISC): CB indices for sharded input signaling
    struct ReaderArgs {
        uint32_t in_cb;
        uint32_t cos_cb;
        uint32_t sin_cb;
        uint32_t cos_tensor_address;
        uint32_t sin_tensor_address;
        uint32_t position_ids_tensor_address;
        uint32_t trans_mat_cb;
    };

    // Writer args (BRISC): none
    struct WriterArgs {};

    // Compute args (TRISC): CB indices as runtime args
    struct ComputeArgs {
        uint32_t in_cb;
        uint32_t cos_cb;
        uint32_t sin_cb;
        uint32_t trans_mat_cb;
        uint32_t rotated_in_interm_cb;
        uint32_t cos_interm_cb;
        uint32_t sin_interm_cb;
        uint32_t out_cb;
    };

    using RTArgs = unified_kernels::SelectByRISCV<ReaderArgs, WriterArgs, ComputeArgs>;

    // ========================================================================
    // Op - the actual operation, templated on CTArgs, and IsActiveCore
    // ========================================================================
    template <typename CTArgs, bool IsActiveCore>
    class Op {
    public:
        void operator()(const RTArgs& args) {
            if constexpr (IsActiveCore) {
                impl(args);
            }
        }

    private:
        void impl(const RTArgs& args) {
#if defined(COMPILE_FOR_NCRISC)

            volatile tt_l1_ptr uint32_t* position_ids_ptr =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(args.position_ids_tensor_address);
            uint32_t position_id = position_ids_ptr[0];

            constexpr uint32_t Wt = CTArgs::Wt;
            constexpr uint32_t page_size = CTArgs::cos_sin_page_size;
            constexpr uint32_t total_Wt = CTArgs::total_Wt;
            constexpr uint32_t start_tile_offset = CTArgs::start_tile_offset;

            // Cos/sin are INTERLEAVED in DRAM. Each row has total_Wt tiles.
            // This core reads Wt tiles starting at start_tile_offset within the row.
            uint32_t start_page = position_id * total_Wt + start_tile_offset;

            auto cos_accessor = TensorAccessor(
                tensor_accessor::make_interleaved_dspec</*is_dram=*/true>(), args.cos_tensor_address, page_size);
            cb_reserve_back(args.cos_cb, Wt);
            uint32_t l1_write_addr = get_write_ptr(args.cos_cb);
            for (uint32_t i = 0; i < Wt; i++) {
                noc_async_read_page(start_page + i, cos_accessor, l1_write_addr);
                l1_write_addr += page_size;
            }
            noc_async_read_barrier();
            cb_push_back(args.cos_cb, Wt);

            auto sin_accessor = TensorAccessor(
                tensor_accessor::make_interleaved_dspec</*is_dram=*/true>(), args.sin_tensor_address, page_size);
            cb_reserve_back(args.sin_cb, Wt);
            l1_write_addr = get_write_ptr(args.sin_cb);
            for (uint32_t i = 0; i < Wt; i++) {
                noc_async_read_page(start_page + i, sin_accessor, l1_write_addr);
                l1_write_addr += page_size;
            }
            noc_async_read_barrier();
            cb_push_back(args.sin_cb, Wt);

#endif
#if defined(COMPILE_FOR_TRISC)
            constexpr uint32_t Wt = CTArgs::Wt;
            constexpr uint32_t Ht = CTArgs::Ht;
            // Assumes all intermediate and output CBs are configured the same
            reconfig_data_format_srcb<false, true>(args.in_cb);
            pack_reconfig_data_format<true>(args.out_cb);

            // ================================================================
            // Wait for sharded CBs (signaled by NCRISC)
            // ================================================================
            cb_wait_front(args.trans_mat_cb, 1);  // Trans_mat: 1 tile, reused for all heads
            cb_wait_front(args.sin_cb, Wt);       // Sin: Wt tiles (reused for all heads)
            cb_wait_front(args.cos_cb, Wt);       // Cos: Wt tiles (reused for all heads)

            // ================================================================
            // Main loop: process Ht heads, each head consumes Wt tiles
            // ================================================================
            for (uint32_t ht = 0; ht < Ht; ht++) {
                reconfig_data_format_srca<false, true>(args.trans_mat_cb);

                // Reserve intermediate and output buffers
                cb_reserve_back(args.rotated_in_interm_cb, Wt);
                cb_reserve_back(args.sin_interm_cb, Wt);
                cb_reserve_back(args.cos_interm_cb, Wt);
                cb_reserve_back(args.out_cb, Wt);

                cb_wait_front(args.in_cb, Wt);

                // ============================================================
                // Step 1: rotated = input @ trans_mat (matmul for rotate_half)
                // ============================================================
                mm_init_short(args.in_cb, args.trans_mat_cb);
                tile_regs_acquire();
                for (uint32_t j = 0; j < Wt; ++j) {
                    matmul_tiles(args.in_cb, args.trans_mat_cb, j, 0, j);
                }
                tile_regs_commit();
                tile_regs_wait();
                for (uint32_t j = 0; j < Wt; ++j) {
                    pack_tile(j, args.rotated_in_interm_cb, j);
                }
                tile_regs_release();
                cb_push_back(args.rotated_in_interm_cb, Wt);
                cb_wait_front(args.rotated_in_interm_cb, Wt);

                // ============================================================
                // Step 2: sin_interm = rotated * sin (broadcast multiply)
                // ============================================================
                reconfig_data_format_srca<false, true>(args.rotated_in_interm_cb);
                mul_bcast_rows_init_short(args.rotated_in_interm_cb, args.sin_cb);
                tile_regs_acquire();
                for (uint32_t j = 0; j < Wt; ++j) {
                    mul_tiles_bcast<BroadcastType::ROW>(args.rotated_in_interm_cb, args.sin_cb, j, j, j);
                }
                tile_regs_commit();
                tile_regs_wait();
                for (uint32_t j = 0; j < Wt; ++j) {
                    pack_tile(j, args.sin_interm_cb, j);
                }
                tile_regs_release();
                cb_push_back(args.sin_interm_cb, Wt);
                cb_pop_front(args.rotated_in_interm_cb, Wt);

                // ============================================================
                // Step 3: cos_interm = input * cos (broadcast multiply)
                // ============================================================
                tile_regs_acquire();
                for (uint32_t j = 0; j < Wt; ++j) {
                    mul_tiles_bcast<BroadcastType::ROW>(args.in_cb, args.cos_cb, j, j, j);
                }
                tile_regs_commit();
                tile_regs_wait();
                for (uint32_t j = 0; j < Wt; ++j) {
                    pack_tile(j, args.cos_interm_cb, j);
                }
                tile_regs_release();
                cb_push_back(args.cos_interm_cb, Wt);
                cb_pop_front(args.in_cb, Wt);

                // ============================================================
                // Step 4: output = cos_interm + sin_interm (add)
                // ============================================================
                cb_wait_front(args.sin_interm_cb, Wt);
                cb_wait_front(args.cos_interm_cb, Wt);
                add_tiles_init(args.cos_interm_cb, args.sin_interm_cb);
                tile_regs_acquire();
                for (uint32_t j = 0; j < Wt; ++j) {
                    add_tiles(args.cos_interm_cb, args.sin_interm_cb, j, j, j);
                }
                tile_regs_commit();
                tile_regs_wait();
                for (uint32_t j = 0; j < Wt; ++j) {
                    pack_tile(j, args.out_cb, j);
                }
                tile_regs_release();
                cb_push_back(args.out_cb, Wt);
                cb_pop_front(args.sin_interm_cb, Wt);
                cb_pop_front(args.cos_interm_cb, Wt);
            }

            // ================================================================
            // Cleanup: pop sin/cos (trans_mat is reused, not popped)
            // Note: sin/cos are reused for all heads, so only pop once after all heads processed
            // ================================================================
            cb_pop_front(args.sin_cb, Wt);
            cb_pop_front(args.cos_cb, Wt);
#endif
        }
    };
};

}  // namespace deepseek_b1_ops
