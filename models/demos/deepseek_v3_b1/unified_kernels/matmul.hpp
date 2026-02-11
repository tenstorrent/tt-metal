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
#include "../kernel_includes/tt_metal/include/compute_kernel_api/custom_mm.h"
#include "api/compute/tile_move_copy.h"
#ifdef TRISC_PACK
#include "ckernel_sfpu_exp.h"
#include "llk_math_eltwise_unary_sfpu_sigmoid.h"
#include "llk_math_eltwise_unary_sfpu_silu.h"
#endif
#endif

// Fused activation types for matmul
enum class FusedActivation : uint32_t {
    NONE = 0,
    SIGMOID = 1,
    SILU = 2,
};

namespace deepseek_b1_ops {

// ============================================================================
// Matmul micro-op with configurable output width (supports large out_w via blocking)
//
// Computes: output[1,out_w] = in0[1,K] @ in1[K,out_w]
//
// CB States:
//   NCRISC: No-op (in1 setup done externally via setup_sharded_buffer)
//   BRISC: No-op (next op waits on output if needed)
//   TRISC (Compute):
//     - Waits: in0 (num_tiles), in1 (num_tiles * out_w)
//     - Reserves: out (out_w tiles)
//     - Pushes: out (out_w tiles)
//     - Pops: in0 (num_tiles) if pop_in0=true, in1 (num_tiles * out_w) if pop_in1=true
// ============================================================================
struct Matmul {
    // ========================================================================
    // Compile-time args structs - different layout per RISC
    // ========================================================================

    // Reader CTArgs (NCRISC): none
    struct ReaderCTArgs {};

    // Writer CTArgs (BRISC): none
    struct WriterCTArgs {};

    // Compute CTArgs (TRISC): out_w (output width in tiles), transpose, fused_activation
    template <uint32_t out_w_, bool transpose_ = false, uint32_t fused_activation_ = 0>
    struct ComputeCTArgs {
        static constexpr uint32_t out_w = out_w_;
        static constexpr bool transpose = transpose_;
        static constexpr FusedActivation fused_activation = static_cast<FusedActivation>(fused_activation_);
        static constexpr bool fuse_sigmoid = fused_activation == FusedActivation::SIGMOID;
        static constexpr bool fuse_silu = fused_activation == FusedActivation::SILU;
    };

    // ========================================================================
    // Runtime args structs - different layout per RISC
    // ========================================================================

    // Reader args (NCRISC): none (NCRISC is no-op, setup done externally)
    struct ReaderArgs {};

    // Writer args (BRISC): none (BRISC is no-op)
    struct WriterArgs {};

    // Compute args (TRISC): [in0, in1, out, num_tiles]
    struct ComputeArgs {
        uint32_t in0;
        uint32_t in1;
        uint32_t out;
        uint32_t k_num_tiles;
    };

    using RTArgs = unified_kernels::SelectByRISCV<ReaderArgs, WriterArgs, ComputeArgs>;

    // ========================================================================
    // Op - the actual operation, templated on CTArgs and IsActiveCore
    // Template args:
    //   CTArgs - compile-time args struct (ReaderCTArgs, WriterCTArgs, or ComputeCTArgs<out_w>)
    //   IsActiveCore - whether this core runs the matmul
    //   pop_in0 - whether to pop in0 after compute (default true)
    //   pop_in1 - whether to pop in1 after compute (default true)
    // ========================================================================
    template <typename CTArgs, bool IsActiveCore, bool pop_in0, bool pop_in1>
    class Op {
    public:
        void operator()(const RTArgs& args) {
            if constexpr (IsActiveCore) {
                impl(args);
            }
        }

    private:
        void impl(const RTArgs& args) {
#if defined(COMPILE_FOR_TRISC)
            // ================================================================
            // TRISC (Compute)
            // ================================================================
            constexpr uint32_t out_subblock_h = 1;
            constexpr uint32_t out_subblock_w = 1;
            constexpr uint32_t in0_block_w = 1;  // Process one K tile at a time
            constexpr uint32_t out_w = CTArgs::out_w;
            constexpr bool transpose = CTArgs::transpose;
            constexpr bool split_acc = true;
            constexpr bool dense_packing = true;
            constexpr bool finalize = split_acc && true;
            constexpr bool read_transposed = transpose && true;

            reconfig_data_format<false, true>(args.in1, args.in0);
            pack_reconfig_data_format<true>(args.out);

            // Wait for all input tiles (both from sharded tensors in L1)
            // in1 has num_tiles * out_w tiles (K tiles for each output column)
            cb_wait_front(args.in0, args.k_num_tiles);
            cb_wait_front(args.in1, args.k_num_tiles * out_w);

            // Reserve output tiles
            cb_reserve_back(args.out, out_w);

            custom_mm_block_init_short<transpose, split_acc, dense_packing>(args.in0, args.in1, args.out, out_w);

            if constexpr (CTArgs::fuse_sigmoid || CTArgs::fuse_silu) {
                // Initialize activation on PACK thread
                if constexpr (CTArgs::fuse_sigmoid) {
                    PACK((ckernel::llk_math_eltwise_unary_sfpu_sigmoid_init<true>()));
                } else {
                    PACK((ckernel::llk_math_eltwise_unary_sfpu_silu_init<true>()));
                }

                // Per-tile: matmul -> activation on PACK -> pack
                for (uint32_t w = 0; w < out_w; w++) {
                    tile_regs_acquire();

                    custom_mm_block<finalize, read_transposed>(args.in0, args.in1, 0, w * args.k_num_tiles, 0, args.k_num_tiles);

                    tile_regs_commit();

                    // Run activation on PACK thread
                    TTI_SEMWAIT(
                        p_stall::STALL_TDMA | p_stall::STALL_CFG,
                        semaphore::t6_sem(semaphore::MATH_PACK),
                        p_stall::STALL_ON_ZERO);
                    PACK(TT_SETC16(DEST_TARGET_REG_CFG_MATH_Offset_ADDR32, ckernel::packer::get_packer_dest_offset()));

                    // Use 2 iterations for 1x32 tiny tiles
                    if constexpr (CTArgs::fuse_sigmoid) {
                        PACK((ckernel::llk_math_eltwise_unary_sfpu_sigmoid<true, false, 2>(0, (int)VectorMode::R)));
                    } else {
                        PACK((ckernel::llk_math_eltwise_unary_sfpu_silu<true, false, 2>(0, (int)VectorMode::R)));
                    }

                    PACK(TTI_STALLWAIT(p_stall::STALL_PACK, p_stall::WAIT_SFPU));

                    pack_tile(0, args.out, w);
                    tile_regs_release();
                }
            } else {
                // Batch processing - all tiles at once
                tile_regs_acquire();

                custom_mm_block<finalize, read_transposed>(args.in0, args.in1, 0, 0, 0, args.k_num_tiles, out_w);

                tile_regs_commit();

                tile_regs_wait();
                for (uint32_t dst_idx = 0; dst_idx < out_w; dst_idx++) {
                    pack_tile(dst_idx, args.out, dst_idx);
                }
                tile_regs_release();
            }

            custom_mm_block_uninit<dense_packing>();

            // Pop inputs
            if constexpr (pop_in0) {
                cb_pop_front(args.in0, args.k_num_tiles);
            }
            if constexpr (pop_in1) {
                cb_pop_front(args.in1, args.k_num_tiles * out_w);
            }

            cb_push_back(args.out, out_w);
            cb_wait_front(args.out, out_w);
            cb_pop_front(args.out, out_w);
#endif
        }
    };  // class Op

};  // struct Matmul

}  // namespace deepseek_b1_ops
