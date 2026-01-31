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
#include "compute_kernel_api/matmul.h"
#include "../kernel_includes/tt_metal/include/compute_kernel_api/custom_mm.h"
#include "compute_kernel_api/tile_move_copy.h"
#endif
#include "api/debug/dprint.h"

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

    // Compute CTArgs (TRISC): out_w (output width in tiles)
    template <uint32_t out_w_>
    struct ComputeCTArgs {
        static constexpr uint32_t out_w = out_w_;
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
            constexpr bool transpose = false;
            constexpr bool split_acc = true;
            constexpr bool dense_packing = true;
            constexpr bool finalize = split_acc && true;
            constexpr bool read_transposed = transpose && false;
            constexpr const char* version_str = "split";

            // Wait for all input tiles (both from sharded tensors in L1)
            // in1 has num_tiles * out_w tiles (K tiles for each output column)
            cb_wait_front(args.in0, args.k_num_tiles);
            cb_wait_front(args.in1, args.k_num_tiles * out_w);

            // Reserve output tiles
            cb_reserve_back(args.out, out_w);

            if constexpr (out_w <= 16) {
                // Use optimized custom_mm API for single output tile with K-dimension reduction
                custom_mm_block_init<transpose, split_acc, dense_packing>(args.in0, args.in1, args.out, out_w);

                tile_regs_acquire();

                volatile std::uint32_t* base_address = (std::uint32_t*)MEM_LLK_DEBUG_BASE;
                tensix_sync();
                UNPACK((base_address[1] = 1));
                MATH((base_address[2] = 2));
                PACK((base_address[3] = 3));
                while (base_address[1] != 1) {
                    asm("nop");
                }
                while (base_address[2] != 2) {
                    asm("nop");
                }
                while (base_address[3] != 3) {
                    asm("nop");
                }
                UNPACK((base_address[5] = 5));
                MATH((base_address[6] = 6));
                PACK((base_address[7] = 7));
                while (base_address[5] != 5) {
                    asm("nop");
                }
                while (base_address[6] != 6) {
                    asm("nop");
                }
                while (base_address[7] != 7) {
                    asm("nop");
                }
                UNPACK((base_address[1] = 0));
                MATH((base_address[2] = 0));
                PACK((base_address[3] = 0));
                while (base_address[1] != 0) {
                    asm("nop");
                }
                while (base_address[2] != 0) {
                    asm("nop");
                }
                while (base_address[3] != 0) {
                    asm("nop");
                }
                UNPACK((base_address[5] = 0));
                MATH((base_address[6] = 0));
                PACK((base_address[7] = 0));
                uint64_t start = ckernel::read_wall_clock();

                // Single call handles all K tiles internally via MOP replay
                custom_mm_block<finalize, read_transposed>(args.in0, args.in1, 0, 0, 0, args.k_num_tiles, out_w);

                tensix_sync();
                uint64_t end = ckernel::read_wall_clock();
                uint64_t kernel_runtime = (end - start);
                UNPACK(
                    (DPRINT << "version " << version_str << " " << get_operand_face_r_dim(args.in0) << " "
                            << args.k_num_tiles << " " << out_w << " " << kernel_runtime << ENDL()));

                tile_regs_commit();

                // Pack output tile
                tile_regs_wait();
                for (uint32_t dst_idx = 0; dst_idx < out_w; dst_idx++) {
                    pack_tile(dst_idx, args.out, dst_idx);
                }
                tile_regs_release();

                custom_mm_block_uninit<dense_packing>();
            } else {
                // Use standard matmul API for multiple output tiles
                // Process in blocks of up to 256 tiles (max DST size)
                mm_block_init(args.in0, args.in1, args.out, transpose, out_subblock_w, out_subblock_h, in0_block_w);

                constexpr uint32_t max_dst_size = 256;
                constexpr uint32_t num_blocks = (out_w + max_dst_size - 1) / max_dst_size;

                uint32_t block_start = 0;
                for (uint32_t block = 0; block < num_blocks; block++) {
                    uint32_t block_end = (block_start + max_dst_size < out_w) ? block_start + max_dst_size : out_w;
                    uint32_t block_size = block_end - block_start;

                    tile_regs_acquire();

                    uint32_t in1_k_offset = block_start;  // Tracks k * out_w + block_start
                    for (uint32_t k = 0; k < args.k_num_tiles; k++) {
                        uint32_t in1_idx = in1_k_offset;
                        for (uint32_t dst_idx = 0; dst_idx < block_size; dst_idx++) {
                            matmul_tiles(args.in0, args.in1, k, in1_idx, dst_idx);
                            in1_idx++;
                        }
                        in1_k_offset += out_w;
                    }

                    tile_regs_commit();

                    // Pack output tiles for this block
                    tile_regs_wait();
                    uint32_t out_idx = block_start;
                    for (uint32_t dst_idx = 0; dst_idx < block_size; dst_idx++) {
                        pack_tile(dst_idx, args.out, out_idx);
                        out_idx++;
                    }
                    tile_regs_release();

                    block_start += max_dst_size;
                }
            }

            // Pop inputs
            if constexpr (pop_in0) {
                cb_pop_front(args.in0, args.k_num_tiles);
            }
            if constexpr (pop_in1) {
                cb_pop_front(args.in1, args.k_num_tiles * out_w);
            }

            cb_push_back(args.out, out_w);
#endif
        }
    };  // class Op

};  // struct Matmul

}  // namespace deepseek_b1_ops
