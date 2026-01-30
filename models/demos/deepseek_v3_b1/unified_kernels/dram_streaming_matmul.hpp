// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "kernel_op_api.hpp"

#if defined(COMPILE_FOR_BRISC)
#include "api/dataflow/dataflow_api.h"
#elif defined(COMPILE_FOR_NCRISC)
#include "api/dataflow/dataflow_api.h"
#elif defined(COMPILE_FOR_TRISC)
#include "compute_kernel_api.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "../kernel_includes/tt_metal/include/compute_kernel_api/custom_mm.h"
#endif

namespace deepseek_b1_ops {

// ============================================================================
// DRAM Streaming Matmul micro-op
//
// Computes: output[1,per_core_N] = in0[1,K] @ in1[K,per_core_N]
//
// - in0: REPLICATED on compute cores (tensor-backed CB)
// - in1: WIDTH_SHARDED in DRAM, streamed one K subblock at a time
// - out: WIDTH_SHARDED in L1 (tensor-backed CB)
//
// CB States:
//   NCRISC: Pushes in0 (num_tiles_k) to signal tensor-backed CB ready
//   BRISC: Streams in1 from DRAM, waits for output
//   TRISC: Matmul compute with optional SILU
// ============================================================================
struct DRAMStreamingMatmul {
    // ========================================================================
    // Compile-time args structs - different layout per RISC
    // ========================================================================

    // Reader CTArgs (NCRISC): num_tiles_k for signaling tensor-backed CB
    template <uint32_t num_tiles_k_>
    struct ReaderCTArgs {
        static constexpr uint32_t num_tiles_k = num_tiles_k_;
    };

    // Writer CTArgs (BRISC): DRAM streaming parameters
    template <
        uint32_t in1_page_size_,
        uint32_t in1_num_pages_,
        uint32_t subblock_k_,
        uint32_t per_core_N_,
        uint32_t in1_block_size_bytes_,
        uint32_t out_num_tiles_,
        uint32_t num_subblocks_k_>
    struct WriterCTArgs {
        static constexpr uint32_t in1_page_size = in1_page_size_;
        static constexpr uint32_t in1_num_pages = in1_num_pages_;
        static constexpr uint32_t subblock_k = subblock_k_;
        static constexpr uint32_t per_core_N = per_core_N_;
        static constexpr uint32_t in1_block_size_bytes = in1_block_size_bytes_;
        static constexpr uint32_t out_num_tiles = out_num_tiles_;
        static constexpr uint32_t num_subblocks_k = num_subblocks_k_;
    };

    // Compute CTArgs (TRISC): matmul compute parameters
    template <
        uint32_t subblock_k_,
        uint32_t per_core_N_,
        uint32_t subblock_w_,
        uint32_t num_subblocks_k_,
        uint32_t tile_r_dim_>
    struct ComputeCTArgs {
        static constexpr uint32_t subblock_k = subblock_k_;
        static constexpr uint32_t per_core_N = per_core_N_;
        static constexpr uint32_t subblock_w = subblock_w_;
        static constexpr uint32_t num_subblocks_k = num_subblocks_k_;
        static constexpr uint32_t tile_r_dim = tile_r_dim_;
    };

    // ========================================================================
    // Runtime args structs - different layout per RISC
    // ========================================================================

    // Reader args (NCRISC): in0 CB to signal
    struct ReaderArgs {
        uint32_t in0_cb;
    };

    // Writer args (BRISC): DRAM streaming parameters
    struct WriterArgs {
        uint32_t in1_cb;
        uint32_t out_cb;
        uint32_t in1_tensor_addr;
        uint32_t dram_bank_id;
        uint32_t vc;
    };

    // Compute args (TRISC): CB IDs
    struct ComputeArgs {
        uint32_t in0_cb;
        uint32_t in1_cb;
        uint32_t out_cb;
    };

    using RTArgs = unified_kernels::SelectByRISCV<ReaderArgs, WriterArgs, ComputeArgs>;

    // ========================================================================
    // Op - the actual operation, templated on CTArgs and IsActiveCore
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
            // ================================================================
            // NCRISC: Signal tensor-backed CB is ready
            // ================================================================
            cb_push_back(args.in0_cb, CTArgs::num_tiles_k);

#elif defined(COMPILE_FOR_BRISC)
            // ================================================================
            // BRISC: Stream in1 from DRAM with transaction IDs
            // ================================================================
            constexpr uint32_t in1_page_size = CTArgs::in1_page_size;
            constexpr uint32_t in1_num_pages = CTArgs::in1_num_pages;
            constexpr uint32_t subblock_k = CTArgs::subblock_k;
            constexpr uint32_t per_core_N = CTArgs::per_core_N;
            constexpr uint32_t in1_block_size_bytes = CTArgs::in1_block_size_bytes;
            constexpr uint32_t out_num_tiles = CTArgs::out_num_tiles;
            constexpr uint32_t num_subblocks_k = CTArgs::num_subblocks_k;
            constexpr uint32_t num_iterations = num_subblocks_k * per_core_N;

            // Setup DRAM read for in1
            uint64_t in1_base_addr = get_noc_addr_from_bank_id<true>(args.dram_bank_id, args.in1_tensor_addr);
            uint32_t l1_write_addr_in1;
            uint32_t l1_read_addr_in1 = 0;

            noc_async_read_one_packet_set_state<true>(in1_base_addr, in1_page_size, args.vc);

            // Multi-buffering with transaction IDs
            constexpr uint32_t num_buffers = 3 * num_subblocks_k;
            constexpr uint32_t extra_blocks_in_flight = 2;
            uint32_t num_free_blocks_in_buffer = num_buffers;
            uint32_t curr_block_trid = 1;
            uint32_t block_trid_to_wait = 1;

            cb_reserve_back(args.in1_cb, num_free_blocks_in_buffer);
            uint32_t l1_write_addr_in1_offset = 0;
            uint32_t l1_write_addr_in1_start = get_write_ptr(args.in1_cb);
            l1_write_addr_in1 = l1_write_addr_in1_start;

            for (uint32_t n = 0; n < num_iterations; ++n) {
                noc_async_read_set_trid(curr_block_trid);

                for (uint32_t p = 0; p < in1_num_pages; p++) {
                    noc_async_read_one_packet_with_state_with_trid(
                        in1_base_addr, l1_read_addr_in1, l1_write_addr_in1, curr_block_trid);
                    l1_read_addr_in1 += in1_page_size;
                    l1_write_addr_in1 += in1_page_size;
                }

                if (num_free_blocks_in_buffer == num_buffers - extra_blocks_in_flight) {
                    noc_async_read_barrier_with_trid(block_trid_to_wait);
                    cb_push_back(args.in1_cb, subblock_k);
                    block_trid_to_wait = block_trid_to_wait == num_buffers ? 1 : (block_trid_to_wait + 1);
                    cb_reserve_back(args.in1_cb, subblock_k * (extra_blocks_in_flight + 1));
                } else {
                    num_free_blocks_in_buffer -= 1;
                }

                if (curr_block_trid == num_buffers) {
                    l1_write_addr_in1_offset = 0;
                    curr_block_trid = 1;
                } else {
                    l1_write_addr_in1_offset += in1_block_size_bytes;
                    curr_block_trid += 1;
                }
                l1_write_addr_in1 = l1_write_addr_in1_start + l1_write_addr_in1_offset;
            }

            for (uint32_t i = 0; i < extra_blocks_in_flight; ++i) {
                noc_async_read_barrier_with_trid(block_trid_to_wait);
                cb_push_back(args.in1_cb, subblock_k);
                block_trid_to_wait = block_trid_to_wait == num_buffers ? 1 : (block_trid_to_wait + 1);
            }

            cb_wait_front(args.out_cb, out_num_tiles);

#elif defined(COMPILE_FOR_TRISC)
            // ================================================================
            // TRISC: Matmul compute with optional SILU
            // ================================================================
            constexpr uint32_t subblock_k = CTArgs::subblock_k;
            constexpr uint32_t per_core_N = CTArgs::per_core_N;
            constexpr uint32_t subblock_w = CTArgs::subblock_w;
            constexpr uint32_t num_subblocks_k = CTArgs::num_subblocks_k;
            constexpr uint32_t tile_r_dim = CTArgs::tile_r_dim;
            constexpr bool transpose = false;
            constexpr uint32_t num_subblocks_n = per_core_N / subblock_w;
            constexpr uint32_t num_tiles_k = subblock_k * num_subblocks_k;

#ifdef FUSE_SILU
            silu_tile_init();
#endif

            custom_mm_block_init(args.in0_cb, args.in1_cb, args.out_cb, transpose, subblock_k);
            cb_wait_front(args.in0_cb, num_tiles_k);

            for (uint32_t sb_n = 0; sb_n < num_subblocks_n; sb_n++) {
                cb_reserve_back(args.out_cb, subblock_w);

                tile_regs_acquire();
                for (uint32_t w = 0; w < subblock_w; w++) {
                    for (uint32_t sb_k = 0; sb_k < num_subblocks_k - 1; sb_k++) {
                        cb_wait_front(args.in1_cb, subblock_k);
                        custom_mm_block<true>(args.in0_cb, args.in1_cb, sb_k * subblock_k, 0, w, transpose, subblock_k);
                        cb_pop_front(args.in1_cb, subblock_k);
                    }
                    cb_wait_front(args.in1_cb, subblock_k);
                    custom_mm_block<false>(
                        args.in0_cb, args.in1_cb, (num_subblocks_k - 1) * subblock_k, 0, w, transpose, subblock_k);
                    cb_pop_front(args.in1_cb, subblock_k);
                }

#ifdef FUSE_SILU
                for (uint32_t i = 0; i < subblock_w; i++) {
                    if constexpr (tile_r_dim <= 4) {
                        MATH((llk_math_eltwise_unary_sfpu_silu<APPROX, DST_ACCUM_MODE, 2>(i, (int)VectorMode::R)));
                    } else if constexpr (tile_r_dim == 8) {
                        MATH((llk_math_eltwise_unary_sfpu_silu<APPROX, DST_ACCUM_MODE, 4>(i, (int)VectorMode::R)));
                    } else {
                        MATH((llk_math_eltwise_unary_sfpu_silu<APPROX, DST_ACCUM_MODE, 8>(i, (int)VectorMode::R)));
                    }
                }
#endif
                tile_regs_commit();

                tile_regs_wait();
                for (uint32_t w = 0; w < subblock_w; w++) {
                    pack_tile(w, args.out_cb, w);
                }
                tile_regs_release();

                cb_push_back(args.out_cb, subblock_w);
            }

            cb_pop_front(args.in0_cb, num_tiles_k);
#endif
        }
    };  // class Op

};  // struct DRAMStreamingMatmul

}  // namespace deepseek_b1_ops
