// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "kernel_op_api.hpp"
#include "mcast.hpp"
#include "matmul_silu.hpp"

namespace deepseek_b1_ops {

// ============================================================================
// McastMatmulSiLU - Multi-core fused matmul+SiLU with mcast input distribution
//
// Computes: output[1, N] = SiLU(input[1, K] @ weights[K, N])
//
// Architecture:
//   - Input activations: HEIGHT_SHARDED on single sender core
//   - Weights: WIDTH_SHARDED across multiple matmul cores
//   - Output: WIDTH_SHARDED across same matmul cores
//
// Fusion benefit: SiLU is applied directly to DST registers after matmul,
// avoiding the L1 round-trip that would occur with separate ops.
//
// Data flow:
//   1. Sender (BRISC): Mcast input from src_cb to dst_cb on all matmul cores
//   2. Receivers (NCRISC): Receive mcast data into dst_cb, setup weight buffers
//   3. Compute (TRISC): Fused matmul+SiLU: SiLU(input @ weights) -> output
//
// CB States:
//   BRISC (Sender):
//     - Waits: src_cb (k_num_tiles) on sender core
//     - Pops: src_cb (k_num_tiles) if pop_src=true
//   NCRISC (Receiver):
//     - Reserves: dst_cb (k_num_tiles)
//     - Pushes: dst_cb (k_num_tiles) after mcast complete
//     - Setup: weights_cb (k_num_tiles * out_w_per_core)
//   TRISC (Compute):
//     - Waits: dst_cb (k_num_tiles), weights_cb (k_num_tiles * out_w_per_core)
//     - Reserves: out_cb (out_w_per_core)
//     - Pushes: out_cb (out_w_per_core)
//     - Pops: dst_cb (k_num_tiles) if pop_input=true
// ============================================================================
struct McastMatmulSiLU {
    // ========================================================================
    // Compile-time args structs
    // ========================================================================

    // Reader CTArgs (NCRISC): none needed
    struct ReaderCTArgs {};

    // Writer CTArgs (BRISC): mcast parameters
    template <uint32_t McastNumCores, bool IsPartOfReceiverGrid, bool Loopback>
    struct WriterCTArgs {
        static constexpr uint32_t mcast_num_cores = McastNumCores;
        static constexpr bool is_part_of_receiver_grid = IsPartOfReceiverGrid;
        static constexpr bool loopback = Loopback;
    };

    // Compute CTArgs (TRISC): out_w_per_core
    template <uint32_t out_w_per_core_>
    struct ComputeCTArgs {
        static constexpr uint32_t out_w_per_core = out_w_per_core_;
    };

    // ========================================================================
    // Runtime args structs
    // ========================================================================

    // Reader args (NCRISC): mcast receiver params
    struct ReaderArgs {
        uint32_t data_receiver_semaphore_id;
        uint32_t dst_cb;
        uint32_t dst_num_pages;
        uint32_t weights_cb;
        uint32_t weights_num_pages;
    };

    // Writer args (BRISC): mcast sender params
    struct WriterArgs {
        uint32_t dest_noc_start_x;
        uint32_t dest_noc_start_y;
        uint32_t dest_noc_end_x;
        uint32_t dest_noc_end_y;
        uint32_t data_sender_semaphore_id;
        uint32_t data_receiver_semaphore_id;
        uint32_t data_size_bytes;
        uint32_t src_cb;
        uint32_t src_num_pages;
        uint32_t input_data_addr;
        uint32_t mcast_receiver_data_addr;
    };

    // Compute args (TRISC): fused matmul+SiLU params
    struct ComputeArgs {
        uint32_t in0_cb;  // Mcast destination CB (input activations)
        uint32_t in1_cb;  // Weights CB
        uint32_t out_cb;  // Output CB
        uint32_t k_num_tiles;
    };

    using RTArgs = unified_kernels::SelectByRISCV<ReaderArgs, WriterArgs, ComputeArgs>;

    // ========================================================================
    // Op - the actual operation
    //
    // Template params:
    //   CTArgsT: compile-time args struct
    //   IsSenderCore: this core sends mcast data
    //   IsMatmulCore: this core performs matmul+SiLU compute
    //   pop_input: whether to pop input CB after matmul
    // ========================================================================
    template <typename CTArgsT, bool IsSenderCore, bool IsMatmulCore, bool pop_input>
    class Op {
    public:
        // ====================================================================
        // init - Initialize mcast sender state
        // Must be called before operator() on sender core
        // ====================================================================
        void init([[maybe_unused]] const RTArgs& args) {
#if defined(COMPILE_FOR_BRISC)
            if constexpr (IsSenderCore) {
                // Convert WriterArgs to Mcast::SenderArgs
                Mcast::SenderArgs mcast_args{
                    args.dest_noc_start_x,
                    args.dest_noc_start_y,
                    args.dest_noc_end_x,
                    args.dest_noc_end_y,
                    args.data_sender_semaphore_id,
                    args.data_receiver_semaphore_id,
                    args.data_size_bytes,
                    args.src_cb,
                    args.src_num_pages,
                    args.input_data_addr,
                    args.mcast_receiver_data_addr,
                };
                mcast_.init(mcast_args);
            }
#endif
        }

        // ====================================================================
        // operator() - Execute mcast + fused matmul+SiLU
        // ====================================================================
        void operator()(const RTArgs& args) {
#if defined(COMPILE_FOR_BRISC)
            // ================================================================
            // BRISC: Mcast sender (on sender core), no-op otherwise
            // ================================================================
            if constexpr (IsSenderCore) {
                // Convert WriterArgs to Mcast::SenderArgs
                Mcast::SenderArgs mcast_args{
                    args.dest_noc_start_x,
                    args.dest_noc_start_y,
                    args.dest_noc_end_x,
                    args.dest_noc_end_y,
                    args.data_sender_semaphore_id,
                    args.data_receiver_semaphore_id,
                    args.data_size_bytes,
                    args.src_cb,
                    args.src_num_pages,
                    args.input_data_addr,
                    args.mcast_receiver_data_addr,
                };
                mcast_(mcast_args);
            }

#elif defined(COMPILE_FOR_NCRISC)
            // ================================================================
            // NCRISC: Mcast receiver + buffer setup (on matmul cores)
            // ================================================================
            if constexpr (IsMatmulCore) {
                // Reserve space in destination CB before mcast writes to it
                cb_reserve_back(args.dst_cb, args.dst_num_pages);

                // Wait for mcast data to arrive
                uint32_t data_receiver_semaphore_addr = get_semaphore(args.data_receiver_semaphore_id);
                volatile tt_l1_ptr uint32_t* data_receiver_semaphore_addr_ptr =
                    (volatile tt_l1_ptr uint32_t*)(data_receiver_semaphore_addr);
                noc_semaphore_wait(data_receiver_semaphore_addr_ptr, VALID);
                noc_semaphore_set(data_receiver_semaphore_addr_ptr, INVALID);

                // Push to destination CB after data arrived
                cb_push_back(args.dst_cb, args.dst_num_pages);

                // Setup weight buffer (sharded tensor already in L1)
                unified_kernels::setup_sharded_buffer(args.weights_cb, args.weights_num_pages);
            }

#elif defined(COMPILE_FOR_TRISC)
            // ================================================================
            // TRISC: Fused Matmul+SiLU compute (on matmul cores)
            // ================================================================
            if constexpr (IsMatmulCore) {
                // MatmulSiLU type alias defined here where CTArgsT::out_w_per_core is available
                using MatmulSiLUCTArgs = MatmulSiLU::ComputeCTArgs<CTArgsT::out_w_per_core>;

                MatmulSiLU::ComputeArgs matmul_silu_args{
                    args.in0_cb,
                    args.in1_cb,
                    args.out_cb,
                    args.k_num_tiles,
                };
                // pop_in0 = pop_input, pop_in1 = false (weights are persistent)
                MatmulSiLU::Op<MatmulSiLUCTArgs, true, pop_input, false> matmul_silu;
                matmul_silu(matmul_silu_args);
            }
#endif
        }

        // ====================================================================
        // teardown - Teardown mcast sender state
        // Must be called after all operator() calls on sender core
        // ====================================================================
        void teardown() {
#if defined(COMPILE_FOR_BRISC)
            if constexpr (IsSenderCore) {
                mcast_.teardown();
            }
#endif
        }

    private:
#if defined(COMPILE_FOR_BRISC)
        // Mcast type alias defined here where CTArgsT::mcast_num_cores etc. are available
        using McastCTArgs =
            Mcast::SenderCTArgs<CTArgsT::mcast_num_cores, CTArgsT::is_part_of_receiver_grid, CTArgsT::loopback>;

        // Mcast op instance (used for persistent sender state)
        Mcast::Op<McastCTArgs, IsSenderCore, IsMatmulCore, IsMatmulCore, true> mcast_;
#endif
    };  // class Op

};  // struct McastMatmulSiLU

}  // namespace deepseek_b1_ops
