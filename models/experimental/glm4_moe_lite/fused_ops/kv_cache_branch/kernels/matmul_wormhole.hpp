// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// Wormhole-compatible Matmul micro-op for GLM KV Cache Branch.
//
// Replaces the Blackhole-only DSv3 matmul.hpp which uses custom_mm_block
// (Blackhole-specific hardware instructions: load_replay_buf,
// TTI_UNPACR_COMMON_EXPLICIT_CONTEXT) with standard Wormhole APIs:
//   - mm_block_init for initialization
//   - matmul_block for the actual block matmul
//
// Uses the standard compute_kernel_api/matmul.h (not custom_mm.h).
//
// Computes: output[1,out_w] = in0[1,K] @ in1[K,out_w]
// This is the DKV down-projection matmul in the KV cache branch.

#pragma once

#include "../../../../../demos/deepseek_v3_b1/unified_kernels/kernel_op_api.hpp"

#if defined(COMPILE_FOR_BRISC)
#include "api/dataflow/dataflow_api.h"
#elif defined(COMPILE_FOR_NCRISC)
#include "api/dataflow/dataflow_api.h"
#elif defined(COMPILE_FOR_TRISC)
#include "api/compute/compute_kernel_api.h"
#include "api/compute/matmul.h"
#include "api/compute/tile_move_copy.h"
#endif

namespace glm4_matmul {

// ============================================================================
// Wormhole Matmul micro-op with configurable output width
//
// Computes: output[1,out_w] = in0[1,K] @ in1[K,out_w]
//
// CB States:
//   NCRISC: No-op (in0/in1 setup done externally via setup_sharded_buffer)
//   BRISC: No-op
//   TRISC (Compute):
//     - Waits: in0 (k_num_tiles), in1 (k_num_tiles * out_w)
//     - Reserves: out (out_w tiles)
//     - Pushes: out (out_w tiles)
//     - Pops: in0 if pop_in0=true, in1 if pop_in1=true
// ============================================================================
struct Matmul {
    // ========================================================================
    // Compile-time args structs
    // ========================================================================
    struct ReaderCTArgs {};
    struct WriterCTArgs {};

    template <uint32_t out_w_, bool transpose_ = false>
    struct ComputeCTArgs {
        static constexpr uint32_t out_w = out_w_;
        static constexpr bool transpose = transpose_;
    };

    // ========================================================================
    // Runtime args structs
    // ========================================================================
    struct ReaderArgs {};
    struct WriterArgs {};

    struct ComputeArgs {
        uint32_t in0;
        uint32_t in1;
        uint32_t out;
        uint32_t k_num_tiles;
    };

    using RTArgs = unified_kernels::SelectByRISCV<ReaderArgs, WriterArgs, ComputeArgs>;

    // ========================================================================
    // Op
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
        void impl([[maybe_unused]] const RTArgs& args) {
#if defined(COMPILE_FOR_TRISC)
            constexpr uint32_t out_w = CTArgs::out_w;
            // Unified kernels execute several micro-ops in one TRISC program;
            // do not inherit unpacker/packer formats from the preceding op.
            reconfig_data_format<false, true>(args.in1, args.in0);
            pack_reconfig_data_format<true>(args.out);

            // Wait for all input tiles
            cb_wait_front(args.in0, args.k_num_tiles);
            cb_wait_front(args.in1, args.k_num_tiles * out_w);

            // Reserve output tiles
            cb_reserve_back(args.out, out_w);

            mm_init_short(args.in0, args.in1);
            tile_regs_acquire();
            MATH(TTI_ZEROACC(p_zeroacc::CLR_ALL, ADDR_MOD_1, 0));
            for (uint32_t out_idx = 0; out_idx < out_w; ++out_idx) {
                for (uint32_t k = 0; k < args.k_num_tiles; ++k) {
                    matmul_tiles(args.in0, args.in1, k, k * out_w + out_idx, out_idx);
                }
            }
            tile_regs_commit();

            tile_regs_wait();
            for (uint32_t dst_idx = 0; dst_idx < out_w; dst_idx++) {
                pack_tile(dst_idx, args.out, dst_idx);
            }
            tile_regs_release();

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

}  // namespace glm4_matmul
