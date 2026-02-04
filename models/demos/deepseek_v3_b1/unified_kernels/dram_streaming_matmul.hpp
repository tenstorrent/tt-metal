// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "kernel_op_api.hpp"

#if defined(COMPILE_FOR_BRISC)
#include "api/dataflow/dataflow_api.h"
#elif defined(COMPILE_FOR_NCRISC)
#include "api/dataflow/dataflow_api.h"
#elif defined(COMPILE_FOR_TRISC)
#include <cstdint>
#include "compute_kernel_api/tile_move_copy.h"
#include "../kernel_includes/tt_metal/include/compute_kernel_api/custom_mm.h"
#include "compute_kernel_api.h"
#include "compute_kernel_api/reconfig_data_format.h"
#include "compute_kernel_api/pack.h"
#ifdef TRISC_PACK
#include "ckernel_sfpu_exp.h"
#include "llk_math_eltwise_unary_sfpu_silu.h"
#include "llk_math_eltwise_binary_sfpu_binop.h"
#endif
#endif

namespace deepseek_b1_ops {

// ============================================================================
// DRAMStreamingMatmul micro-op
//
// Computes: output[1,N] = in0[1,K] @ in1[K,N] with in1 streamed from DRAM
//
// CB States:
//   NCRISC: No-op (in0 and index CB setup done externally via setup_sharded_buffer)
//   BRISC: Streams in1 from DRAM with pipelining, waits for output
//   TRISC (Compute):
//     - Waits: in0 (K tiles), in1 (subblock_k tiles streamed)
//     - Reserves/Pushes: out (subblock_w tiles at a time)
//     - Optional fused SiLU activation
// ============================================================================
struct DRAMStreamingMatmul {
    // ========================================================================
    // Compile-time args structs
    // ========================================================================

    // Reader CTArgs (NCRISC) - empty, sharded buffer setup done in kernel file
    struct ReaderCTArgs {};

    // Writer CTArgs (BRISC)
    template <
        uint32_t cb_in1_,
        uint32_t cb_out_,
        uint32_t in1_tensor_addr_,
        uint32_t in1_page_size_,
        uint32_t in1_num_pages_,
        uint32_t subblock_k_,
        uint32_t per_core_n_,
        uint32_t in1_block_size_bytes_,
        uint32_t out_num_tiles_,
        uint32_t num_subblocks_k_,
        uint32_t bank_id_,
        uint32_t vc_,
        uint32_t enable_indexing_ = 0,
        uint32_t cb_index_ = 0,
        uint32_t index_offset_ = 0,
        uint32_t use_hardcoded_expert_index_ = 0>
    struct WriterCTArgs {
        static constexpr uint32_t cb_in1 = cb_in1_;
        static constexpr uint32_t cb_out = cb_out_;
        static constexpr uint32_t in1_tensor_addr = in1_tensor_addr_;
        static constexpr uint32_t in1_page_size = in1_page_size_;
        static constexpr uint32_t in1_num_pages = in1_num_pages_;
        static constexpr uint32_t subblock_k = subblock_k_;
        static constexpr uint32_t per_core_n = per_core_n_;
        static constexpr uint32_t in1_block_size_bytes = in1_block_size_bytes_;
        static constexpr uint32_t out_num_tiles = out_num_tiles_;
        static constexpr uint32_t num_subblocks_k = num_subblocks_k_;
        static constexpr uint32_t bank_id = bank_id_;
        static constexpr uint32_t vc = vc_;
        // Expert indexing support
        static constexpr bool enable_indexing = enable_indexing_ == 1;
        static constexpr uint32_t cb_index = cb_index_;
        static constexpr uint32_t index_offset = index_offset_;  // offset into index tensor
        static constexpr bool use_hardcoded_expert_index = use_hardcoded_expert_index_ == 1;  // For testing
    };

    // Compute CTArgs (TRISC)
    template <
        uint32_t cb_in0_,
        uint32_t cb_in1_,
        uint32_t cb_out_,
        uint32_t subblock_k_,
        uint32_t per_core_n_,
        uint32_t subblock_w_,
        uint32_t num_subblocks_k_,
        uint32_t tile_r_dim_,
        uint32_t fuse_silu_>
    struct ComputeCTArgs {
        static constexpr uint32_t cb_in0 = cb_in0_;
        static constexpr uint32_t cb_in1 = cb_in1_;
        static constexpr uint32_t cb_out = cb_out_;
        static constexpr uint32_t subblock_k = subblock_k_;
        static constexpr uint32_t per_core_n = per_core_n_;
        static constexpr uint32_t subblock_w = subblock_w_;
        static constexpr uint32_t num_subblocks_k = num_subblocks_k_;
        static constexpr uint32_t tile_r_dim = tile_r_dim_;
        static constexpr bool fuse_silu = fuse_silu_ == 1;
    };

    // ========================================================================
    // Op - the actual operation, templated on CTArgs and IsActiveCore
    // PopIn0: If true (default), pops in0 after compute. Set to false to reuse
    //         in0 for multiple matmuls (e.g., gate_proj and up_proj).
    // ========================================================================
    template <typename CTArgs, bool IsActiveCore, bool PopIn0 = true>
    class Op {
    public:
        void operator()() {
            if constexpr (IsActiveCore) {
                impl();
            }
        }

    private:
        void impl() {
#if defined(COMPILE_FOR_NCRISC)
            // ================================================================
            // NCRISC: No-op - sharded buffer setup done in kernel file
            // ================================================================

#elif defined(COMPILE_FOR_BRISC)
            // ================================================================
            // BRISC: Stream in1 from DRAM with pipelining
            // ================================================================
            constexpr uint32_t num_iterations = CTArgs::num_subblocks_k * CTArgs::per_core_n;

            // bank_id and vc are per-core compile-time args
            constexpr uint32_t dram_bank_id = CTArgs::bank_id;
            constexpr uint32_t vc = CTArgs::vc;

            // Expert indexing: compute DRAM offset based on expert index
            uint32_t expert_offset_bytes = 0;
            if constexpr (CTArgs::enable_indexing) {
                // Wait for index tensor to be ready
                cb_wait_front(CTArgs::cb_index, 1);

                // Read expert index from index tensor at specified offset (uint16)
                uint32_t expert_idx;
                if constexpr (CTArgs::use_hardcoded_expert_index) {
                    expert_idx = 0;  // For testing: always use expert 0
                } else {
                    volatile tt_l1_ptr uint16_t* index_ptr =
                        reinterpret_cast<volatile tt_l1_ptr uint16_t*>(get_read_ptr(CTArgs::cb_index));
                    expert_idx = static_cast<uint32_t>(index_ptr[CTArgs::index_offset]);
                }

                // Compute offset: expert_idx * expert_size_per_bank
                // Each expert per bank = k_tiles * per_core_n tiles (column-major shuffled per expert)
                // k_tiles = num_subblocks_k * subblock_k
                // bytes_per_tile = in1_block_size_bytes / subblock_k
                constexpr uint32_t k_tiles = CTArgs::num_subblocks_k * CTArgs::subblock_k;
                constexpr uint32_t bytes_per_tile = CTArgs::in1_block_size_bytes / CTArgs::subblock_k;
                constexpr uint32_t expert_size_bytes = k_tiles * CTArgs::per_core_n * bytes_per_tile;
                expert_offset_bytes = expert_idx * expert_size_bytes;
            }

            // Setup DRAM read for in1
            uint64_t in1_base_addr = get_noc_addr_from_bank_id<true>(dram_bank_id, CTArgs::in1_tensor_addr);
            uint32_t l1_write_addr_in1;
            uint32_t l1_read_addr_in1 = expert_offset_bytes;

            // Set up NOC state for page reads
            noc_async_read_one_packet_set_state<true>(in1_base_addr, CTArgs::in1_page_size, vc);

            // Multi-buffering with transaction IDs for pipelining
            constexpr uint32_t num_buffers = 3 * CTArgs::num_subblocks_k;
            constexpr uint32_t extra_blocks_in_flight = 2;
            uint32_t num_free_blocks_in_buffer = num_buffers;
            uint32_t curr_block_trid = 1;
            uint32_t block_trid_to_wait = 1;

            cb_reserve_back(CTArgs::cb_in1, num_free_blocks_in_buffer);
            uint32_t l1_write_addr_in1_offset = 0;
            uint32_t l1_write_addr_in1_start = get_write_ptr(CTArgs::cb_in1);
            l1_write_addr_in1 = l1_write_addr_in1_start;

            // Read in1: for each N column, read num_subblocks_k K subblocks
            for (uint32_t n = 0; n < num_iterations; ++n) {
                noc_async_read_set_trid(curr_block_trid);

                for (uint32_t p = 0; p < CTArgs::in1_num_pages; p++) {
                    noc_async_read_one_packet_with_state_with_trid(
                        in1_base_addr, l1_read_addr_in1, l1_write_addr_in1, curr_block_trid);
                    l1_read_addr_in1 += CTArgs::in1_page_size;
                    l1_write_addr_in1 += CTArgs::in1_page_size;
                }

                if (num_free_blocks_in_buffer == num_buffers - extra_blocks_in_flight) {
                    noc_async_read_barrier_with_trid(block_trid_to_wait);
                    cb_push_back(CTArgs::cb_in1, CTArgs::subblock_k);
                    block_trid_to_wait = block_trid_to_wait == num_buffers ? 1 : (block_trid_to_wait + 1);
                    cb_reserve_back(CTArgs::cb_in1, CTArgs::subblock_k * (extra_blocks_in_flight + 1));
                } else {
                    num_free_blocks_in_buffer -= 1;
                }

                if (curr_block_trid == num_buffers) {
                    l1_write_addr_in1_offset = 0;
                    curr_block_trid = 1;
                } else {
                    l1_write_addr_in1_offset += CTArgs::in1_block_size_bytes;
                    curr_block_trid += 1;
                }
                l1_write_addr_in1 = l1_write_addr_in1_start + l1_write_addr_in1_offset;
            }

            // Push remaining blocks
            for (uint32_t i = 0; i < extra_blocks_in_flight; ++i) {
                noc_async_read_barrier_with_trid(block_trid_to_wait);
                cb_push_back(CTArgs::cb_in1, CTArgs::subblock_k);
                block_trid_to_wait = block_trid_to_wait == num_buffers ? 1 : (block_trid_to_wait + 1);
            }

            // Wait for compute to finish
            cb_wait_front(CTArgs::cb_out, CTArgs::out_num_tiles);

#elif defined(COMPILE_FOR_TRISC)
            // ================================================================
            // TRISC: Matmul compute with optional fused SiLU
            // ================================================================
            constexpr uint32_t transpose = false;
            constexpr uint32_t num_subblocks_n = CTArgs::per_core_n / CTArgs::subblock_w;
            constexpr uint32_t num_tiles_k = CTArgs::subblock_k * CTArgs::num_subblocks_k;

            if constexpr (CTArgs::fuse_silu) {
                PACK((llk_math_eltwise_unary_sfpu_silu_init<true>()));
            }

            custom_mm_block_init(CTArgs::cb_in0, CTArgs::cb_in1, CTArgs::cb_out, transpose, CTArgs::subblock_k);
            cb_wait_front(CTArgs::cb_in0, num_tiles_k);

            for (uint32_t sb_n = 0; sb_n < num_subblocks_n; sb_n++) {
                cb_reserve_back(CTArgs::cb_out, CTArgs::subblock_w);

                if constexpr (CTArgs::fuse_silu) {
                    // Per-tile pipelining with SFPU overlap
                    for (uint32_t w = 0; w < CTArgs::subblock_w; w++) {
                        tile_regs_acquire();

                        for (uint32_t sb_k = 0; sb_k < CTArgs::num_subblocks_k - 1; sb_k++) {
                            cb_wait_front(CTArgs::cb_in1, CTArgs::subblock_k);
                            custom_mm_block<true>(
                                CTArgs::cb_in0,
                                CTArgs::cb_in1,
                                sb_k * CTArgs::subblock_k,
                                0,
                                0,
                                transpose,
                                CTArgs::subblock_k);
                            cb_pop_front(CTArgs::cb_in1, CTArgs::subblock_k);
                        }
                        cb_wait_front(CTArgs::cb_in1, CTArgs::subblock_k);
                        custom_mm_block<false>(
                            CTArgs::cb_in0,
                            CTArgs::cb_in1,
                            (CTArgs::num_subblocks_k - 1) * CTArgs::subblock_k,
                            0,
                            0,
                            transpose,
                            CTArgs::subblock_k);
                        cb_pop_front(CTArgs::cb_in1, CTArgs::subblock_k);

                        tile_regs_commit();

                        // Run SiLU on PACK thread
                        TTI_SEMWAIT(
                            p_stall::STALL_TDMA | p_stall::STALL_CFG,
                            semaphore::t6_sem(semaphore::MATH_PACK),
                            p_stall::STALL_ON_ZERO);
                        PACK(TT_SETC16(
                            DEST_TARGET_REG_CFG_MATH_Offset_ADDR32, ckernel::packer::get_packer_dest_offset()));

                        if constexpr (CTArgs::tile_r_dim <= 4) {
                            PACK((llk_math_eltwise_unary_sfpu_silu<true, false, 2>(0, (int)VectorMode::R)));
                        } else if constexpr (CTArgs::tile_r_dim == 8) {
                            PACK((llk_math_eltwise_unary_sfpu_silu<true, false, 4>(0, (int)VectorMode::R)));
                        } else {
                            PACK((llk_math_eltwise_unary_sfpu_silu<true, false, 8>(0, (int)VectorMode::R)));
                        }

                        PACK(TTI_STALLWAIT(p_stall::STALL_PACK, p_stall::WAIT_SFPU));
                        pack_tile(0, CTArgs::cb_out, w);
                        tile_regs_release();
                    }
                } else {
                    // Batch processing
                    tile_regs_acquire();

                    for (uint32_t w = 0; w < CTArgs::subblock_w; w++) {
                        for (uint32_t sb_k = 0; sb_k < CTArgs::num_subblocks_k - 1; sb_k++) {
                            cb_wait_front(CTArgs::cb_in1, CTArgs::subblock_k);
                            custom_mm_block<true>(
                                CTArgs::cb_in0,
                                CTArgs::cb_in1,
                                sb_k * CTArgs::subblock_k,
                                0,
                                w,
                                transpose,
                                CTArgs::subblock_k);
                            cb_pop_front(CTArgs::cb_in1, CTArgs::subblock_k);
                        }
                        cb_wait_front(CTArgs::cb_in1, CTArgs::subblock_k);
                        custom_mm_block<false>(
                            CTArgs::cb_in0,
                            CTArgs::cb_in1,
                            (CTArgs::num_subblocks_k - 1) * CTArgs::subblock_k,
                            0,
                            w,
                            transpose,
                            CTArgs::subblock_k);
                        cb_pop_front(CTArgs::cb_in1, CTArgs::subblock_k);
                    }

                    tile_regs_commit();
                    tile_regs_wait();

                    for (uint32_t w = 0; w < CTArgs::subblock_w; w++) {
                        pack_tile(w, CTArgs::cb_out, w);
                    }
                    tile_regs_release();
                }

                cb_push_back(CTArgs::cb_out, CTArgs::subblock_w);
            }

            if constexpr (PopIn0) {
                cb_pop_front(CTArgs::cb_in0, num_tiles_k);
            }
#endif
        }
    };  // class Op

};  // struct DRAMStreamingMatmul

}  // namespace deepseek_b1_ops
