// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "kernel_op_api.hpp"

#if defined(COMPILE_FOR_BRISC)
#include "api/dataflow/dataflow_api.h"
#include "expert_index_encoding.hpp"
#elif defined(COMPILE_FOR_NCRISC)
#include "api/dataflow/dataflow_api.h"
#elif defined(COMPILE_FOR_TRISC)
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_unary/sfpu_split_includes.h"
#include "api/compute/tile_move_copy.h"
#include "../kernel_includes/tt_metal/include/compute_kernel_api/eltwise_mul_scalar.h"
#endif

namespace deepseek_b1_ops {

// ============================================================================
// GatedReduce micro-op: SiLU(sum(group1)) * sum(group2)
//
// Performs gated local reduction over two groups of input tiles:
//   1. Reduces group1 tiles with pairwise add, applies SiLU
//   2. Reduces group2 tiles with pairwise add (no activation)
//   3. Multiplies the two results
//
// Produces k_num_tiles output tiles (one per K iteration).
// Each iteration consumes tiles_per_k tiles from each group CB.
//
// CB Layout:
//   group1_cb:   Gate partials (tiles_per_k tiles consumed per iteration)
//   group2_cb:   Up partials (tiles_per_k tiles consumed per iteration)
//   intermed_cb: Intermediate buffer (2 tiles, reused each iteration)
//   out_cb:      Output (1 tile produced per iteration)
// ============================================================================
struct GatedReduce {
    struct ReaderCTArgs {};

    // BRISC writer args. Default (all zero) keeps BRISC a no-op for the shared/
    // DRAM path. SRAM path templates these to drive the per-expert scalar copy
    // (gate_output_scores L1 → scalar_cb, filtered by SRAM-flagged TopK indices).
    template <
        uint32_t scalar_cb_ = 0,
        uint32_t scalar_src_l1_addr_ = 0,
        uint32_t indices_l1_addr_ = 0,
        uint32_t num_active_ = 0>
    struct WriterCTArgs {
        static constexpr uint32_t scalar_cb = scalar_cb_;
        static constexpr uint32_t scalar_src_l1_addr = scalar_src_l1_addr_;
        static constexpr uint32_t indices_l1_addr = indices_l1_addr_;
        static constexpr uint32_t num_active = num_active_;
    };

    // tiles_per_k stays compile-time (face inner-reduce count, fixed for both
    // shared and SRAM paths). k_num_tiles moved to runtime (ComputeArgs)
    // so the SRAM path can supply n_sram_active without an extra template flag.
    // EnableScalar: 1 = SRAM path multiplies the silu(g1)*g2 product by a per-K
    // scalar from scalar_cb; 0 = shared path skips the scalar mul. Replaces the
    // old `scalar_cb != 0` runtime check (which would mis-fire if cb_id 0 ever
    // got assigned to scalar_cb).
    template <uint32_t TilesPerK, uint32_t EnableScalar = 0>
    struct ComputeCTArgs {
        static constexpr uint32_t tiles_per_k = TilesPerK;
        static constexpr bool enable_scalar = EnableScalar != 0;
    };

    struct ReaderArgs {};
    struct WriterArgs {};

    struct ComputeArgs {
        uint32_t group1_cb;    // gate partials CB
        uint32_t group2_cb;    // up partials CB
        uint32_t intermed_cb;  // intermediate CB (2 tiles, reused)
        uint32_t out_cb;       // output CB (1 tile per iteration)
        uint32_t scalar_cb;    // SRAM scale CB (1 tile per iter, bf16 at [0,0]); 0 = scale disabled
        uint32_t k_num_tiles;  // outer loop count (1 for shared, n_sram_active for SRAM)
        // Total expected pushes to out_cb per call. After the K loop pushes
        // k_num_tiles real tiles, we pad up to out_cb_total_pushes with empty
        // pushes so the downstream mcast sender's CT src_num_pages always
        // matches GR's per-iter push count. Set to k_num_tiles when no padding
        // is needed (shared path).
        uint32_t out_cb_total_pushes;
    };

    using RTArgs = unified_kernels::SelectByRISCV<ReaderArgs, WriterArgs, ComputeArgs>;

    template <typename CTArgs, bool IsActiveCore>
    class Op {
    public:
        void operator()(const RTArgs& args) {
            if constexpr (IsActiveCore) {
                impl(args);
            }
        }

    private:
        void impl([[maybe_unused]] const RTArgs& args) {
#if defined(COMPILE_FOR_BRISC)
            // SRAM path: copy one bf16 score per SRAM-flagged TopK position into
            // scalar_cb (at byte 0 of each face tile page). TRISC consumes via
            // BroadcastType::SCALAR. No-op when CTArgs::scalar_cb == 0 (shared path).
            if constexpr (CTArgs::scalar_cb != 0) {
                volatile tt_l1_ptr uint16_t* score_ptr =
                    reinterpret_cast<volatile tt_l1_ptr uint16_t*>(CTArgs::scalar_src_l1_addr);
                volatile tt_l1_ptr uint16_t* idx_ptr =
                    reinterpret_cast<volatile tt_l1_ptr uint16_t*>(CTArgs::indices_l1_addr);
                for (uint32_t k = 0; k < CTArgs::num_active; k++) {
                    if (deepseek_b1_ops::is_sram_expert(static_cast<uint32_t>(idx_ptr[k]))) {
                        cb_reserve_back(CTArgs::scalar_cb, 1);
                        volatile tt_l1_ptr uint16_t* dst_ptr =
                            reinterpret_cast<volatile tt_l1_ptr uint16_t*>(get_write_ptr(CTArgs::scalar_cb));
                        dst_ptr[0] = score_ptr[k];
                        cb_push_back(CTArgs::scalar_cb, 1);
                    }
                }
            }
#elif defined(COMPILE_FOR_TRISC)
            constexpr uint32_t tiles_per_k = CTArgs::tiles_per_k;
            const uint32_t k_num_tiles = args.k_num_tiles;
            static_assert(tiles_per_k >= 2 && tiles_per_k % 2 == 0, "tiles_per_k must be even and >= 2");

            // Init once before the loop
            // Assumes all input cbs are configured the same, and the intermediate cb is configured the same as the
            // output cb
            reconfig_data_format<false, true>(args.group1_cb, args.group1_cb);
            pack_reconfig_data_format<true>(args.out_cb);
            silu_tile_init();
            for (uint32_t k = 0; k < k_num_tiles; k++) {
                // Group 1: reduce + SiLU
                add_init(args.group1_cb, args.group1_cb, true /* acc_to_dest */);

                cb_wait_front(args.group1_cb, tiles_per_k);
                cb_reserve_back(args.intermed_cb, 1);

                tile_regs_acquire();
                for (uint32_t i = 0; i < tiles_per_k; i += 2) {
                    add_tiles(args.group1_cb, args.group1_cb, i, i + 1, 0);
                }
                silu_tile(0);
                tile_regs_commit();
                tile_regs_wait();
                pack_tile(0, args.intermed_cb);
                tile_regs_release();

                cb_pop_front(args.group1_cb, tiles_per_k);
                cb_push_back(args.intermed_cb, 1);

                // Group 2: reduce (skip re-init add for different CB assuming they're configured the same)

                cb_wait_front(args.group2_cb, tiles_per_k);
                cb_reserve_back(args.intermed_cb, 1);

                tile_regs_acquire();
                for (uint32_t i = 0; i < tiles_per_k; i += 2) {
                    add_tiles(args.group2_cb, args.group2_cb, i, i + 1, 0);
                }
                tile_regs_commit();
                tile_regs_wait();
                pack_tile(0, args.intermed_cb);
                tile_regs_release();

                cb_pop_front(args.group2_cb, tiles_per_k);
                cb_push_back(args.intermed_cb, 1);

                // Multiply: out = SiLU(g1) * g2  (DRAM path)
                //           out = SiLU(g1) * scale[k] * g2  (SRAM path)
                cb_wait_front(args.intermed_cb, 2);
                cb_reserve_back(args.out_cb, 1);

                if constexpr (CTArgs::enable_scalar) {
                    // Wait for this iteration's scalar (BRISC pushes one bf16 per
                    // active expert into scalar_cb in TopK SRAM-flagged order).
                    cb_wait_front(args.scalar_cb, 1);

                    tile_regs_acquire();
                    // DST[0] = silu(g1) * scale[0]
                    deepseek_mul_tiles_bcast_scalar_init_short(args.intermed_cb, args.scalar_cb);
                    deepseek_mul_tiles_bcast_scalar(args.intermed_cb, args.scalar_cb, 0, 0, 0);
                    // DST[0] *= sum(g2)
                    deepseek_binary_dest_reuse_tiles_init(args.intermed_cb);
                    deepseek_binary_dest_reuse_tiles(args.intermed_cb, 1, 0);
                    tile_regs_commit();
                    tile_regs_wait();
                    pack_tile(0, args.out_cb);
                    tile_regs_release();

                    cb_pop_front(args.scalar_cb, 1);
                } else {
                    mul_init(args.intermed_cb, args.intermed_cb);
                    tile_regs_acquire();
                    mul_tiles(args.intermed_cb, args.intermed_cb, 0, 1, 0);
                    tile_regs_commit();
                    tile_regs_wait();
                    pack_tile(0, args.out_cb);
                    tile_regs_release();
                }

                cb_pop_front(args.intermed_cb, 2);
                cb_push_back(args.out_cb, 1);
            }

            // Pad out_cb pushes to out_cb_total_pushes so the downstream mcast
            // sender's CT src_num_pages always matches GR's per-iter push count.
            // No-op when out_cb_total_pushes == k_num_tiles (shared path).
            const uint32_t padding =
                (args.out_cb_total_pushes > k_num_tiles) ? (args.out_cb_total_pushes - k_num_tiles) : 0;
            if (padding > 0) {
                cb_reserve_back(args.out_cb, padding);
                cb_push_back(args.out_cb, padding);
            }
#endif
        }
    };
};

}  // namespace deepseek_b1_ops
