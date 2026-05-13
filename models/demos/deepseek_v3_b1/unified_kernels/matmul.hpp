// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "kernel_op_api.hpp"
#include "kernel_utils.hpp"

#if defined(COMPILE_FOR_BRISC)
#include "api/dataflow/dataflow_api.h"
#elif defined(COMPILE_FOR_NCRISC)
#include "api/dataflow/dataflow_api.h"
#elif defined(COMPILE_FOR_TRISC)
#include "api/compute/compute_kernel_api.h"
#include "api/compute/matmul.h"
#include "../kernel_includes/tt_metal/include/compute_kernel_api/custom_mm.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/experimental/pack_block.h"
#ifdef TRISC_PACK
#include "ckernel_sfpu_exp.h"
#include "llk_math_eltwise_unary_sfpu_sigmoid.h"
#include "llk_math_eltwise_unary_sfpu_silu.h"
// Deepseek-private SFPU op (relative include — file is intentionally not
// surfaced via llk_math_unary_sfpu_api.h; see header for usage rules).
#include "internal/llk_math_eltwise_unary_sfpu_apply_scaler.h"
#endif
#endif

// Fused activation types for matmul
enum class FusedActivation : uint32_t {
    NONE = 0,
    SIGMOID = 1,
    SILU = 2,
    // Bound to the deepseek-private apply_scaler SFPU op (see
    // unified_kernels/internal/llk_math_eltwise_unary_sfpu_apply_scaler.h).
    // Multiplies dst tile face quadrants 0/2/16/18 by the scaler pre-loaded
    // into LReg0 via TT_SFPLOADI in the init block. Requires custom_sfpu_cb_
    // to be set so the init block can wait for and load the scaler tile.
    CUSTOM_SFPU = 3,
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
    // custom_sfpu_cb_ is an optional CB id consumed by the CUSTOM_SFPU activation path
    // (e.g. a scalar/scale tile loaded into the SFPU before the per-tile op). 0 means
    // "no CB" — has_custom_sfpu_cb tells the user-supplied stub whether to read it.
    template <
        uint32_t out_w_,
        bool transpose_ = false,
        uint32_t fused_activation_ = 0,
        bool fused_activation_approx_mode_ = false,
        uint32_t custom_sfpu_cb_ = 0>
    struct ComputeCTArgs {
        static constexpr uint32_t out_w = out_w_;
        static constexpr bool transpose = transpose_;
        static constexpr FusedActivation fused_activation = static_cast<FusedActivation>(fused_activation_);
        static constexpr bool fuse_sigmoid = fused_activation == FusedActivation::SIGMOID;
        static constexpr bool fuse_silu = fused_activation == FusedActivation::SILU;
        static constexpr bool fuse_custom_sfpu = fused_activation == FusedActivation::CUSTOM_SFPU;
        static constexpr bool fused_activation_approx_mode = fused_activation_approx_mode_;
        static constexpr uint32_t custom_sfpu_cb = custom_sfpu_cb_;
        static constexpr bool has_custom_sfpu_cb = custom_sfpu_cb_ > 0;
    };

    // ========================================================================
    // Runtime args structs - different layout per RISC
    // ========================================================================

    // Reader args (NCRISC): none (NCRISC is no-op, setup done externally)
    struct ReaderArgs {};

    // Writer args (BRISC): none (BRISC is no-op)
    struct WriterArgs {};

    // Compute args (TRISC): [in0, in1, out, num_tiles, in1_address_override]
    struct ComputeArgs {
        uint32_t in0;
        uint32_t in1;
        uint32_t out;
        uint32_t k_num_tiles;
        uint32_t in1_address_override = 0;  // byte address; overrides in1 read ptr if > 0
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
    template <typename CTArgs, bool IsActiveCore, bool pop_in0, bool pop_in1, bool skip_reconfig = false>
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
            constexpr uint32_t out_w = CTArgs::out_w;
            constexpr bool transpose = CTArgs::transpose;
            constexpr bool split_acc = true;
            constexpr bool dense_packing = true;
            constexpr bool finalize = split_acc && true;
            constexpr bool read_transposed = transpose && true;
            constexpr bool fuse_activation = CTArgs::fuse_sigmoid || CTArgs::fuse_silu;
            if constexpr (!skip_reconfig) {
                reconfig_data_format<false, true>(args.in1, args.in0);
                pack_reconfig_data_format<true>(args.out);
            }
            custom_mm_block_init_short<transpose, split_acc, dense_packing>(args.in0, args.in1, args.out, out_w);
            if constexpr (!fuse_activation && !skip_reconfig) {
                pack_block_contiguous_init(args.out);
            }

            // Wait for all input tiles (both from sharded tensors in L1)
            // in1 has num_tiles * out_w tiles (K tiles for each output column)
            if (args.in1_address_override > 0) {
                UNPACK(({ unified_kernels::override_cb_rd_ptr(args.in1, args.in1_address_override); }));
            } else {
                cb_wait_front(args.in1, args.k_num_tiles * out_w);
            }
            cb_wait_front(args.in0, args.k_num_tiles);

            // Reserve output tiles
            cb_reserve_back(args.out, out_w);

            if constexpr (fuse_activation) {
                // Initialize activation on PACK thread
                if constexpr (CTArgs::fuse_sigmoid) {
                    PACK((ckernel::llk_math_eltwise_unary_sfpu_sigmoid_init<CTArgs::fused_activation_approx_mode>()));
                } else if constexpr (CTArgs::fuse_silu) {
                    PACK((ckernel::llk_math_eltwise_unary_sfpu_silu_init<CTArgs::fused_activation_approx_mode>()));
                } else if constexpr (CTArgs::fuse_custom_sfpu) {
                    // Bit-copy the first 16-bit element of custom_sfpu_cb into SFPU LReg0
                    // via SFPLOADI on the PACK thread. Mode SFPLOADI_MOD0_FLOATB (0)
                    // treats the 16 bits as BF16 and lands them in LReg[31:16] with
                    // LReg[15:0] zeroed — that's the FP32 representation of the BF16
                    // value, ready to consume as a vFloat in the SFPU. Other useful
                    // modes (sfpi/sfpi_constants.h):
                    //   FLOATA (1)  FP16  → FP32
                    //   USHORT (2)  uint16 → uint32 (zero-extend; bits land low half)
                    //   SHORT  (4)  int16 → int32  (sign-extend)
                    //   UPPER  (8)  raw  → LReg[31:16], LReg[15:0] PRESERVED
                    //   LOWER  (10) raw  → LReg[15:0], LReg[31:16] preserved
                    // See:
                    //   tt-isa-documentation/BlackholeA0/TensixTile/TensixCoprocessor/SFPLOADI.md
                    //
                    // Two layers of UNPACK→PACK sync:
                    //   1. cb_wait_front (cb_api.h) is TRISC_UNPACK-guarded; UNPACK posts
                    //      semaphore::UNPACK_OPERAND_SYNC after the wait returns so PACK
                    //      knows the producer's tile is committed in L1.
                    //   2. get_tile_address (cb_api.h:155-177) does its own mailbox handoff
                    //      of the byte address: UNPACK computes (fifo_rd_ptr + off)<<4 and
                    //      writes to the MathThread/PackThread mailboxes; PACK's invocation
                    //      blocks on mailbox_read until UNPACK posts.
                    if constexpr (CTArgs::has_custom_sfpu_cb) {
                        UNPACK(({ cb_wait_front(CTArgs::custom_sfpu_cb, 1); }));
                        UNPACK((t6_semaphore_post<p_stall::UNPACK0>(semaphore::UNPACK_OPERAND_SYNC)));

                        PACK((t6_semaphore_wait_on_zero<p_stall::STALL_PACK>(semaphore::UNPACK_OPERAND_SYNC)));
                        uint32_t cb_rd_addr = get_tile_address(CTArgs::custom_sfpu_cb, 0);
                        PACK(({
                            volatile tt_l1_ptr uint16_t* l1_ptr =
                                reinterpret_cast<volatile tt_l1_ptr uint16_t*>(cb_rd_addr);
                            uint16_t bits = l1_ptr[0];
                            TT_SFPLOADI(/*lreg=*/0, /*mod=*/0, bits);
                            // SFPTRANSP shuffles out the apply value loaded above so the
                            // calculate_apply_scaler op below can pick it up from the
                            // expected LReg slot (transposes {LReg0..3} and {LReg4..7}
                            // as two independent 4-lane groups).
                            TTI_SFPTRANSP(0, 0, 0, 0);
                        }));
                        PACK((t6_semaphore_get<p_stall::PACK>(semaphore::UNPACK_OPERAND_SYNC)));
                    }
                    PACK((ckernel::llk_math_eltwise_unary_sfpu_apply_scaler_init<
                          CTArgs::fused_activation_approx_mode>()));
                }

                // Per-tile: matmul -> activation on PACK -> pack
                for (uint32_t w = 0; w < out_w; w++) {
                    tile_regs_acquire();

                    custom_mm_block<finalize, read_transposed>(args.in0, args.in1, 0, w * args.k_num_tiles, 0, args.k_num_tiles);

                    tile_regs_commit();

                    // Run activation on PACK thread
                    PACK(TTI_SEMWAIT(
                        p_stall::STALL_TDMA | p_stall::STALL_CFG | p_stall::STALL_SFPU,
                        semaphore::t6_sem(semaphore::MATH_PACK),
                        p_stall::STALL_ON_ZERO));
                    PACK(TT_SETC16(DEST_TARGET_REG_CFG_MATH_Offset_ADDR32, ckernel::packer::get_packer_dest_offset()));

                    // Use 2 iterations for 1x32 tiny tiles
                    if constexpr (CTArgs::fuse_sigmoid) {
                        PACK((ckernel::
                                  llk_math_eltwise_unary_sfpu_sigmoid<CTArgs::fused_activation_approx_mode, false, 2>(
                                      0, (int)VectorMode::R)));
                    } else if constexpr (CTArgs::fuse_silu) {
                        PACK((ckernel::llk_math_eltwise_unary_sfpu_silu<CTArgs::fused_activation_approx_mode, false, 2>(
                            0, (int)VectorMode::R)));
                    } else if constexpr (CTArgs::fuse_custom_sfpu) {
                        PACK((ckernel::llk_math_eltwise_unary_sfpu_apply_scaler<
                              CTArgs::fused_activation_approx_mode,
                              /*is_fp32_dest_acc_en=*/false,
                              /*ITERATIONS=*/2>(0, (int)VectorMode::R)));
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
                pack_block_contiguous(0, args.out, out_w);
                tile_regs_release();
            }
            cb_push_back(args.out, out_w);

            custom_mm_block_uninit<dense_packing>();

            // Pop inputs
            if constexpr (pop_in0) {
                cb_pop_front(args.in0, args.k_num_tiles);
            }
            if constexpr (pop_in1) {
                cb_pop_front(args.in1, args.k_num_tiles * out_w);
            }
            // Pop the per-invocation custom SFPU scalar (UNPACK-side, since cb_pop_front
            // is also TRISC_UNPACK-guarded). One scalar per Op call.
            // TODO(custom-sfpu): if the same scalar is reused across many matmul
            // invocations (e.g. fused MoE expert scaling), hoist push/pop to the caller
            // and remove this block.
            if constexpr (CTArgs::fuse_custom_sfpu && CTArgs::has_custom_sfpu_cb) {
                UNPACK(({ cb_pop_front(CTArgs::custom_sfpu_cb, 1); }));
            }

            cb_push_back(args.out, out_w);
#endif
        }
    };  // class Op

};  // struct Matmul

}  // namespace deepseek_b1_ops
