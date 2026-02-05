// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "kernel_op_api.hpp"
#include "mcast.hpp"
#include "matmul.hpp"
#include "matmul_silu.hpp"
#include "silu.hpp"  // Contains EltwiseMul

namespace deepseek_b1_ops {

// ============================================================================
// McastSwiGLU - Multi-core fused SwiGLU with mcast input distribution
//
// Computes: output[1, N] = SiLU(input[1, K] @ W_gate[K, N]) * (input[1, K] @ W_up[K, N])
//
// Architecture:
//   - Input activations: HEIGHT_SHARDED on single sender core
//   - W_gate: WIDTH_SHARDED across multiple matmul cores
//   - W_up: WIDTH_SHARDED across SAME matmul cores (matching W_gate)
//   - Output: WIDTH_SHARDED across same matmul cores
//
// Fusion benefit: All three operations (gate matmul+SiLU, up matmul, multiply)
// execute on the same core using local CBs, avoiding any cross-core data movement
// between the operations.
//
// Data flow:
//   1. Sender (BRISC): Mcast input from src_cb to dst_cb on all matmul cores
//   2. Receivers (NCRISC): Receive mcast data, setup weight buffers (gate & up)
//   3. Compute (TRISC): Fused SwiGLU:
//      a. gate = SiLU(input @ W_gate) -> gate_intermediate_cb
//      b. up = input @ W_up -> up_intermediate_cb
//      c. output = gate * up -> out_cb
//
// CB Layout (per matmul core):
//   - dst_cb: Mcast destination (input activations)
//   - gate_weights_cb: W_gate shard (backed by tensor)
//   - up_weights_cb: W_up shard (backed by tensor)
//   - gate_intermediate_cb: Gate matmul+SiLU output (dynamically allocated)
//   - up_intermediate_cb: Up matmul output (dynamically allocated)
//   - out_cb: Final output (backed by output tensor)
//
// CB States:
//   BRISC (Sender):
//     - Waits: src_cb (k_num_tiles) on sender core
//     - Pops: src_cb (k_num_tiles) if pop_src=true
//   NCRISC (Receiver):
//     - Reserves: dst_cb (k_num_tiles)
//     - Pushes: dst_cb (k_num_tiles) after mcast complete
//     - Setup: gate_weights_cb, up_weights_cb
//   TRISC (Compute):
//     - Waits: dst_cb, gate_weights_cb, up_weights_cb
//     - Computes: gate -> gate_intermediate_cb, up -> up_intermediate_cb
//     - Computes: gate * up -> out_cb
//     - Pops: dst_cb (k_num_tiles) if pop_input=true, intermediates always popped
// ============================================================================
struct McastSwiGLU {
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

    // Compute CTArgs (TRISC): out_w_per_core (output tiles per core)
    template <uint32_t out_w_per_core_>
    struct ComputeCTArgs {
        static constexpr uint32_t out_w_per_core = out_w_per_core_;
    };

    // ========================================================================
    // Runtime args structs
    // ========================================================================

    // Reader args (NCRISC): mcast receiver + weight buffer setup
    struct ReaderArgs {
        uint32_t data_receiver_semaphore_id;
        uint32_t dst_cb;
        uint32_t dst_num_pages;
        uint32_t gate_weights_cb;
        uint32_t gate_weights_num_pages;
        uint32_t up_weights_cb;
        uint32_t up_weights_num_pages;
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

    // Compute args (TRISC): fused SwiGLU params
    struct ComputeArgs {
        uint32_t in0_cb;                // Mcast destination CB (input activations)
        uint32_t gate_weights_cb;       // W_gate CB
        uint32_t up_weights_cb;         // W_up CB
        uint32_t gate_intermediate_cb;  // Gate output intermediate CB
        uint32_t up_intermediate_cb;    // Up output intermediate CB
        uint32_t out_cb;                // Final output CB
        uint32_t k_num_tiles;
    };

    using RTArgs = unified_kernels::SelectByRISCV<ReaderArgs, WriterArgs, ComputeArgs>;

    // ========================================================================
    // Op - the actual operation
    //
    // Template params:
    //   CTArgsT: compile-time args struct
    //   IsSenderCore: this core sends mcast data
    //   IsMatmulCore: this core performs SwiGLU compute
    //   pop_input: whether to pop input CB after SwiGLU
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
        // operator() - Execute mcast + fused SwiGLU
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

                // Setup weight buffers (sharded tensors already in L1)
                unified_kernels::setup_sharded_buffer(args.gate_weights_cb, args.gate_weights_num_pages);
                unified_kernels::setup_sharded_buffer(args.up_weights_cb, args.up_weights_num_pages);
            }

#elif defined(COMPILE_FOR_TRISC)
            // ================================================================
            // TRISC: Fused SwiGLU compute (on matmul cores)
            // Step 1: gate = SiLU(input @ W_gate) -> gate_intermediate
            // Step 2: up = input @ W_up -> up_intermediate
            // Step 3: output = gate * up -> out
            // ================================================================
            if constexpr (IsMatmulCore) {
                constexpr uint32_t out_w = CTArgsT::out_w_per_core;

                // === Step 1: Gate = MatmulSiLU(input, W_gate) ===
                // pop_in0 = false (need input for up matmul)
                // pop_in1 = false (weights are persistent)
                {
                    using GateCTArgs = MatmulSiLU::ComputeCTArgs<out_w>;
                    MatmulSiLU::ComputeArgs gate_args{
                        args.in0_cb,                // input
                        args.gate_weights_cb,       // W_gate
                        args.gate_intermediate_cb,  // output
                        args.k_num_tiles,
                    };
                    MatmulSiLU::Op<GateCTArgs, true, false, false> gate_matmul_silu;
                    gate_matmul_silu(gate_args);
                }

                // === Step 2: Up = Matmul(input, W_up) ===
                // pop_in0 = pop_input (consume input after up matmul if requested)
                // pop_in1 = false (weights are persistent)
                {
                    using UpCTArgs = Matmul::ComputeCTArgs<out_w>;
                    Matmul::ComputeArgs up_args{
                        args.in0_cb,              // input (same as gate)
                        args.up_weights_cb,       // W_up
                        args.up_intermediate_cb,  // output
                        args.k_num_tiles,
                    };
                    Matmul::Op<UpCTArgs, true, pop_input, false> up_matmul;
                    up_matmul(up_args);
                }

                // === Step 3: Output = EltwiseMul(gate, up) ===
                // pop_in0 = true (gate intermediate consumed)
                // pop_in1 = true (up intermediate consumed)
                {
                    using MulCTArgs = EltwiseMul::ComputeCTArgs<out_w>;
                    EltwiseMul::ComputeArgs mul_args{
                        args.gate_intermediate_cb,  // gate result
                        args.up_intermediate_cb,    // up result
                        args.out_cb,                // final output
                    };
                    EltwiseMul::Op<MulCTArgs, true, true, true> eltwise_mul;
                    eltwise_mul(mul_args);
                }
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

};  // struct McastSwiGLU

}  // namespace deepseek_b1_ops
