// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// Wormhole-compatible matmul for Pre-SDPA unified kernel.
// Replaces deepseek_v3_b1/unified_kernels/matmul.hpp which uses
// Blackhole-only custom_mm.h (custom_mm_block_init, custom_mm_block,
// custom_mm_block_uninit) from kernel_includes/tt_metal/.
//
// Uses standard compute_kernel_api/matmul.h (mm_init, matmul_tiles)
// which work on both Wormhole and Blackhole architectures.
//
// Computes: output[1, out_w] = in0[1, K] @ in1[K, out_w]
//
// in1 tile layout in CB (row-major tile order):
//   tile(k, c) at CB index: k * out_w + c
// This matches ttnn's standard tile layout for sharded weight tensors.

#pragma once

#include "../../../../deepseek_v3_b1/unified_kernels/kernel_op_api.hpp"

#if defined(COMPILE_FOR_BRISC)
#include "api/dataflow/dataflow_api.h"
#elif defined(COMPILE_FOR_NCRISC)
#include "api/dataflow/dataflow_api.h"
#elif defined(COMPILE_FOR_TRISC)
#include "compute_kernel_api.h"
#include "compute_kernel_api/matmul.h"
#include "compute_kernel_api/tile_move_copy.h"
#endif

namespace deepseek_b1_ops {

// ============================================================================
// Wormhole-compatible Matmul micro-op
//
// Same external API as the Blackhole version (same struct names, same template
// parameters) but uses standard matmul APIs internally.
// ============================================================================
struct Matmul {
    struct ReaderCTArgs {};
    struct WriterCTArgs {};

    template <uint32_t out_w_, bool transpose_ = false>
    struct ComputeCTArgs {
        static constexpr uint32_t out_w = out_w_;
        static constexpr bool transpose = transpose_;
    };

    struct ReaderArgs {};
    struct WriterArgs {};

    struct ComputeArgs {
        uint32_t in0;
        uint32_t in1;
        uint32_t out;
        uint32_t k_num_tiles;
    };

    using RTArgs = unified_kernels::SelectByRISCV<ReaderArgs, WriterArgs, ComputeArgs>;

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
            constexpr uint32_t transpose = CTArgs::transpose ? 1 : 0;

            // Wait for all input tiles
            cb_wait_front(args.in0, args.k_num_tiles);
            cb_wait_front(args.in1, args.k_num_tiles * out_w);

            // Reserve output tiles
            cb_reserve_back(args.out, out_w);

            // Initialize standard matmul
            mm_init(args.in0, args.in1, args.out, transpose);

            tile_regs_acquire();

            // Accumulate over K dimension for each output column
            // in1 tile layout: tile(k, c) = k * out_w + c (row-major)
            for (uint32_t k = 0; k < args.k_num_tiles; k++) {
                for (uint32_t c = 0; c < out_w; c++) {
                    uint32_t in1_idx = k * out_w + c;
                    matmul_tiles(args.in0, args.in1, k, in1_idx, c);
                }
            }

            tile_regs_commit();

            tile_regs_wait();
            for (uint32_t c = 0; c < out_w; c++) {
                pack_tile(c, args.out, c);
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

}  // namespace deepseek_b1_ops
