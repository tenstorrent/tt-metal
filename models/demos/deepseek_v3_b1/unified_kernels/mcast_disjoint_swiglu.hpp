// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "kernel_op_api.hpp"
#include "mcast.hpp"
#include "matmul.hpp"
#include "matmul_silu.hpp"
#include "silu.hpp"  // Contains EltwiseMul

#if defined(COMPILE_FOR_BRISC) || defined(COMPILE_FOR_NCRISC)
#include "api/dataflow/dataflow_api.h"
#endif

namespace deepseek_b1_ops {

// ============================================================================
// McastDisjointSwiGLU - Multi-core fused SwiGLU with DISJOINT gate/up grids
//
// Computes: output[1, N] = SiLU(input[1, K] @ W_gate[K, N]) * (input[1, K] @ W_up[K, N])
//
// Architecture:
//   - Input activations: HEIGHT_SHARDED on single sender core
//   - W_gate: WIDTH_SHARDED across gate_grid (e.g., 8x9 = 72 cores, 32 elements/core)
//   - W_up: WIDTH_SHARDED across up_grid (e.g., 4x9 = 36 cores, 64 elements/core)
//   - Gate and Up grids are DISJOINT (different cores)
//   - Output: WIDTH_SHARDED on up_grid
//
// Data flow:
//   1. Sender (BRISC): Mcast input to bounding box of gate_grid ∪ up_grid
//   2. Gate cores: Receive mcast, compute SiLU(input @ W_gate), send result to up cores
//   3. Up cores: Receive mcast, receive gate results, compute input @ W_up, multiply
//
// Gate → Up transfer pattern:
//   - Each row: 2 gate cores send to 1 up core
//   - Gate core (gx, gy) → Up core (gx // 2, gy)
//   - Gate output tiles at offset (gx % 2) * out_w_per_gate_core in up's receive buffer
//
// CB Layout:
//   Sender core:
//     - src_cb: Input tensor (HEIGHT_SHARDED)
//   Gate cores:
//     - dst_cb: Mcast destination (input activations)
//     - gate_weights_cb: W_gate shard
//     - gate_output_cb: Gate matmul+SiLU output (sent to up cores)
//   Up cores:
//     - dst_cb: Mcast destination (input activations)
//     - up_weights_cb: W_up shard
//     - gate_recv_cb: Received gate results (from 2 gate cores)
//     - up_output_cb: Up matmul output
//     - out_cb: Final output (gate * up)
// ============================================================================
struct McastDisjointSwiGLU {
    // ========================================================================
    // Compile-time args structs
    // ========================================================================

    // Reader CTArgs (NCRISC): gate_cores_per_up_core for semaphore wait
    template <uint32_t GateCoresPerUpCore>
    struct ReaderCTArgs {
        static constexpr uint32_t gate_cores_per_up_core = GateCoresPerUpCore;
    };

    // Writer CTArgs (BRISC): mcast parameters + gate→up transfer info
    template <
        uint32_t McastNumCores,
        bool IsPartOfReceiverGrid,
        bool Loopback,
        uint32_t OutWPerGateCore,
        uint32_t GateCoresPerUpCore>
    struct WriterCTArgs {
        static constexpr uint32_t mcast_num_cores = McastNumCores;
        static constexpr bool is_part_of_receiver_grid = IsPartOfReceiverGrid;
        static constexpr bool loopback = Loopback;
        static constexpr uint32_t out_w_per_gate_core = OutWPerGateCore;
        static constexpr uint32_t gate_cores_per_up_core = GateCoresPerUpCore;
    };

    // Compute CTArgs (TRISC): output tiles per core (gate vs up)
    template <uint32_t OutWPerGateCore, uint32_t OutWPerUpCore>
    struct ComputeCTArgs {
        static constexpr uint32_t out_w_per_gate_core = OutWPerGateCore;
        static constexpr uint32_t out_w_per_up_core = OutWPerUpCore;
    };

    // ========================================================================
    // Runtime args structs
    // ========================================================================

    // Reader args (NCRISC): mcast receiver + weight buffer setup + gate recv setup
    struct ReaderArgs {
        // Mcast receiver
        uint32_t data_receiver_semaphore_id;
        uint32_t dst_cb;
        uint32_t dst_num_pages;
        // Gate weights setup (for gate cores)
        uint32_t gate_weights_cb;
        uint32_t gate_weights_num_pages;
        // Up weights setup (for up cores)
        uint32_t up_weights_cb;
        uint32_t up_weights_num_pages;
        // Gate recv setup (for up cores)
        uint32_t gate_recv_cb;
        uint32_t gate_recv_num_pages;
        uint32_t gate_recv_semaphore_id;
    };

    // Writer args (BRISC): mcast sender + gate→up transfer params
    struct WriterArgs {
        // Mcast sender params
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
        // Gate→Up transfer params (for gate cores)
        uint32_t gate_output_cb;
        uint32_t gate_output_num_pages;
        uint32_t gate_recv_cb;  // CB index for gate_recv (same on gate and up cores)
        uint32_t gate_recv_semaphore_id;
        // Grid params for computing target up core (all NOC coordinates)
        uint32_t gate_grid_start_noc_x;  // NOC x of gate grid start
        uint32_t gate_grid_start_noc_y;  // NOC y of gate grid start
        uint32_t up_grid_start_noc_x;    // NOC x of up grid start
        uint32_t up_grid_start_noc_y;    // NOC y of up grid start
        uint32_t out_tile_size_bytes;    // For computing transfer offset
    };

    // Compute args (TRISC)
    struct ComputeArgs {
        // Common: input CB
        uint32_t in0_cb;
        // Gate compute args
        uint32_t gate_weights_cb;
        uint32_t gate_output_cb;
        // Up compute args
        uint32_t up_weights_cb;
        uint32_t gate_recv_cb;
        uint32_t up_output_cb;
        uint32_t out_cb;
        uint32_t k_num_tiles;
    };

    using RTArgs = unified_kernels::SelectByRISCV<ReaderArgs, WriterArgs, ComputeArgs>;

    // ========================================================================
    // Op - the actual operation
    //
    // Template params:
    //   CTArgsT: compile-time args struct
    //   IsSenderCore: this core sends mcast data
    //   IsGateCore: this core computes gate matmul+SiLU
    //   IsUpCore: this core computes up matmul + multiply
    //   pop_input: whether to pop input CB after compute
    // ========================================================================
    template <typename CTArgsT, bool IsSenderCore, bool IsGateCore, bool IsUpCore, bool pop_input>
    class Op {
    public:
        // ====================================================================
        // init - Initialize mcast sender state
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
        // operator() - Execute full disjoint SwiGLU pipeline
        // ====================================================================
        void operator()(const RTArgs& args) {
#if defined(COMPILE_FOR_BRISC)
            // ================================================================
            // BRISC Phase 1: Mcast sender (on sender core only)
            // ================================================================
            if constexpr (IsSenderCore) {
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

// ================================================================
// BRISC Phase 2: Gate→Up transfer (on gate cores after compute)
// Wait for gate_output_cb to be filled by TRISC, then NOC write to up core
// ================================================================
// DEBUG: Temporarily disabled to isolate hang
#if 0
            if constexpr (IsGateCore) {
                // Wait for compute to fill gate_output_cb
                cb_wait_front(args.gate_output_cb, args.gate_output_num_pages);

                // Get gate output data address
                uint32_t gate_output_addr = get_read_ptr(args.gate_output_cb);

                // Compute my position in gate grid (all in NOC coordinates)
                uint32_t my_noc_x = static_cast<uint32_t>(my_x[0]);
                uint32_t my_noc_y = static_cast<uint32_t>(my_y[0]);
                uint32_t gate_local_x = my_noc_x - args.gate_grid_start_noc_x;
                uint32_t gate_local_y = my_noc_y - args.gate_grid_start_noc_y;

                // Compute target up core NOC coordinates
                // Gate core (gx, gy) maps to up core (up_start + gx/2, up_start + gy)
                // Up grid is adjacent to gate grid, so we use pre-computed NOC start coordinates
                uint32_t up_local_x = gate_local_x / 2;
                uint32_t target_noc_x = args.up_grid_start_noc_x + up_local_x;
                uint32_t target_noc_y = args.up_grid_start_noc_y + gate_local_y;

                // Compute transfer offset in up core's gate_recv_cb
                // Even gate_local_x -> offset 0, odd gate_local_x -> offset = out_w_per_gate_core tiles
                constexpr uint32_t out_w = CTArgsT::out_w_per_gate_core;
                uint32_t offset_tiles = (gate_local_x % 2) * out_w;
                uint32_t offset_bytes = offset_tiles * args.out_tile_size_bytes;

                // Get gate_recv_cb base address on up core
                // gate_recv_cb is allocated on all matmul cores (gate ∪ up) with same size,
                // so get_write_ptr gives consistent L1 address across cores
                uint32_t gate_recv_base_addr = get_write_ptr(args.gate_recv_cb);
                uint32_t target_l1_addr = gate_recv_base_addr + offset_bytes;

                // Compute full NOC address
                uint64_t target_noc_addr = get_noc_addr(target_noc_x, target_noc_y, target_l1_addr);

                // Calculate transfer size
                uint32_t transfer_size_bytes = out_w * args.out_tile_size_bytes;

                // NOC write gate output to up core
                noc_async_write(gate_output_addr, target_noc_addr, transfer_size_bytes);
                noc_async_write_barrier();

                // Signal up core that gate data has arrived
                uint32_t gate_recv_semaphore_addr = get_semaphore(args.gate_recv_semaphore_id);
                uint64_t up_core_semaphore_noc_addr = get_noc_addr(
                    target_noc_x,
                    target_noc_y,
                    gate_recv_semaphore_addr);
                noc_semaphore_inc(up_core_semaphore_noc_addr, 1);

                // Pop gate output CB
                cb_pop_front(args.gate_output_cb, args.gate_output_num_pages);
            }
#endif

#elif defined(COMPILE_FOR_NCRISC)
            // ================================================================
            // NCRISC: Mcast receiver + buffer setup
            // ================================================================
            if constexpr (IsGateCore || IsUpCore) {
                // Reserve space in destination CB before mcast writes to it
                cb_reserve_back(args.dst_cb, args.dst_num_pages);

                // Wait for mcast data to arrive (will be signaled after data is sent)
                // Note: Do NOT reset semaphore before waiting - the mcast sender's init()
                // sends 0 (current value before being set to VALID), so there's no init flag.
                // Resetting here would race with operator() which sends VALID.
                uint32_t data_receiver_semaphore_addr = get_semaphore(args.data_receiver_semaphore_id);
                volatile tt_l1_ptr uint32_t* data_receiver_semaphore_addr_ptr =
                    (volatile tt_l1_ptr uint32_t*)(data_receiver_semaphore_addr);
                noc_semaphore_wait(data_receiver_semaphore_addr_ptr, VALID);
                noc_semaphore_set(data_receiver_semaphore_addr_ptr, INVALID);

                // Push to destination CB after data arrived
                cb_push_back(args.dst_cb, args.dst_num_pages);
            }

            // Setup weight buffers (sharded tensors already in L1)
            if constexpr (IsGateCore) {
                unified_kernels::setup_sharded_buffer(args.gate_weights_cb, args.gate_weights_num_pages);
            }
            if constexpr (IsUpCore) {
                unified_kernels::setup_sharded_buffer(args.up_weights_cb, args.up_weights_num_pages);

// DEBUG: Disabled gate recv wait to isolate mcast hang
#if 0
                // Reserve gate_recv_cb for incoming gate data from 2 gate cores
                cb_reserve_back(args.gate_recv_cb, args.gate_recv_num_pages);

                // Wait for gate data to arrive (2 gate cores send to this up core)
                uint32_t gate_recv_semaphore_addr = get_semaphore(args.gate_recv_semaphore_id);
                volatile tt_l1_ptr uint32_t* gate_recv_semaphore_addr_ptr =
                    (volatile tt_l1_ptr uint32_t*)(gate_recv_semaphore_addr);
                constexpr uint32_t gate_cores_per_up = CTArgsT::gate_cores_per_up_core;
                noc_semaphore_wait(gate_recv_semaphore_addr_ptr, gate_cores_per_up);
                noc_semaphore_set(gate_recv_semaphore_addr_ptr, 0);

                // Push gate_recv_cb after data arrived
                cb_push_back(args.gate_recv_cb, args.gate_recv_num_pages);
#endif
            }

#elif defined(COMPILE_FOR_TRISC)
// ================================================================
// TRISC: Compute - DEBUG: All disabled to isolate mcast hang
// ================================================================
#if 0
            if constexpr (IsGateCore) {
                // Gate cores: MatmulSiLU(input, W_gate) -> gate_output
                constexpr uint32_t out_w = CTArgsT::out_w_per_gate_core;
                using GateCTArgs = MatmulSiLU::ComputeCTArgs<out_w>;
                MatmulSiLU::ComputeArgs gate_args{
                    args.in0_cb,
                    args.gate_weights_cb,
                    args.gate_output_cb,
                    args.k_num_tiles,
                };
                // pop_in0 = pop_input, pop_in1 = false (weights persistent)
                MatmulSiLU::Op<GateCTArgs, true, pop_input, false> gate_matmul_silu;
                gate_matmul_silu(gate_args);
            }

            if constexpr (IsUpCore) {
                constexpr uint32_t out_w = CTArgsT::out_w_per_up_core;

                // Up cores: Matmul(input, W_up) -> out_cb
                {
                    using UpCTArgs = Matmul::ComputeCTArgs<out_w>;
                    Matmul::ComputeArgs up_args{
                        args.in0_cb,
                        args.up_weights_cb,
                        args.out_cb,
                        args.k_num_tiles,
                    };
                    // pop_in0 = pop_input, pop_in1 = false (weights persistent)
                    Matmul::Op<UpCTArgs, true, pop_input, false> up_matmul;
                    up_matmul(up_args);
                }
            }
#endif
#endif
        }

        // ====================================================================
        // teardown - Teardown mcast sender state
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
        // Mcast type alias
        using McastCTArgs =
            Mcast::SenderCTArgs<CTArgsT::mcast_num_cores, CTArgsT::is_part_of_receiver_grid, CTArgsT::loopback>;

        // Only instantiate on sender core, but need to keep mcast_ member for all BRISC
        static constexpr bool is_sender_and_matmul = IsSenderCore && (IsGateCore || IsUpCore);
        Mcast::Op<McastCTArgs, IsSenderCore, is_sender_and_matmul, (IsGateCore || IsUpCore), true> mcast_;
#endif
    };  // class Op

};  // struct McastDisjointSwiGLU

}  // namespace deepseek_b1_ops
