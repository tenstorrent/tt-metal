// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "kernel_op_api.hpp"

#if defined(COMPILE_FOR_TRISC)
#include "api/compute/eltwise_binary.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/experimental/pack_block.h"
#endif

namespace deepseek_b1_ops {

// ============================================================================
// ResidualAdd micro-op: matmul_out + shard(residual) → add_out
//
// Each matmul core adds its local matmul output tiles with the corresponding
// shard of a broadcast residual CB. The residual CB contains the full [1, N]
// residual (all total_in1_tiles tiles); each core indexes at offset
// core_idx * out_w.
//
// CB Layout:
//   in0_cb:  matmul output (out_w tiles, consumed)
//   in1_cb:  full residual (total_in1_tiles tiles, consumed after add)
//   out_cb:  add output (out_w tiles, produced)
// ============================================================================
struct ResidualAdd {
    struct ReaderCTArgs {};
    struct WriterCTArgs {};

    template <uint32_t OutW>
    struct ComputeCTArgs {
        static constexpr uint32_t out_w = OutW;
    };

    struct ReaderArgs {};
    struct WriterArgs {};

    struct ComputeArgs {
        uint32_t in0_cb;           // matmul output
        uint32_t in1_cb;           // full residual CB (all N tiles)
        uint32_t out_cb;           // add output
        uint32_t total_in1_tiles;  // total tiles in residual CB
        uint32_t core_idx;         // index into residual CB (= core_idx * out_w)
        uint32_t sram_in_cb = 0;   // optional SRAM expert down_proj output (per-core, out_w tiles)
    };

    using RTArgs = unified_kernels::SelectByRISCV<ReaderArgs, WriterArgs, ComputeArgs>;

    // SkipAdd: when true, copies in0→out and pops in1 without adding.
    // Used on non-root devices in multi-device reduce so residual is only
    // counted once after the cross-device sum.
    // Has3Inputs: when true, also adds sram_in_cb (per-core out_w tiles) — used to
    // fold SRAM routed expert contribution into the shared expert output pipeline.
    template <typename CTArgs, bool IsActiveCore, bool SkipAdd = false, bool Has3Inputs = false>
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
            constexpr uint32_t out_w = CTArgs::out_w;

            cb_wait_front(args.in0_cb, out_w);
            cb_wait_front(args.in1_cb, args.total_in1_tiles);
            if constexpr (Has3Inputs) {
                cb_wait_front(args.sram_in_cb, out_w);
            }

            if constexpr (SkipAdd) {
                // Pass-through: copy in0 to out, discard in1 (and sram_in if present)
                reconfig_data_format<false, true>(args.in0_cb, args.in0_cb);
                pack_reconfig_data_format<true>(args.out_cb);
                pack_block_contiguous_init(args.out_cb);
                copy_tile_to_dst_init_short(args.in0_cb);
                cb_reserve_back(args.out_cb, out_w);
                tile_regs_acquire();
                for (uint32_t j = 0; j < out_w; j++) {
                    copy_tile(args.in0_cb, j, j);
                }
                tile_regs_commit();
                tile_regs_wait();
                pack_block_contiguous(0, args.out_cb, out_w);
                tile_regs_release();
            } else if constexpr (Has3Inputs) {
                // 3-way add: dst = sram_in (copy) + in0 + in1 (acc_to_dest)
                reconfig_data_format<false, true>(args.in0_cb, args.in1_cb);
                pack_reconfig_data_format<true>(args.out_cb);
                pack_block_contiguous_init(args.out_cb);
                cb_reserve_back(args.out_cb, out_w);

                copy_tile_to_dst_init_short(args.sram_in_cb);
                tile_regs_acquire();
                for (uint32_t j = 0; j < out_w; j++) {
                    copy_tile(args.sram_in_cb, j, j);
                }
                add_tiles_init(args.in0_cb, args.in1_cb, true /* acc_to_dest */);
                for (uint32_t j = 0; j < out_w; j++) {
                    add_tiles(args.in0_cb, args.in1_cb, j, args.core_idx * out_w + j, j);
                }
                tile_regs_commit();
                tile_regs_wait();
                pack_block_contiguous(0, args.out_cb, out_w);
                tile_regs_release();
            } else {
                // Normal 2-way: matmul_out + shard(residual)
                reconfig_data_format<false, true>(args.in0_cb, args.in1_cb);
                pack_reconfig_data_format<true>(args.out_cb);
                pack_block_contiguous_init(args.out_cb);

                add_tiles_init(args.in0_cb, args.in1_cb);

                cb_reserve_back(args.out_cb, out_w);
                tile_regs_acquire();
                for (uint32_t j = 0; j < out_w; j++) {
                    add_tiles(args.in0_cb, args.in1_cb, j, args.core_idx * out_w + j, j);
                }
                tile_regs_commit();
                tile_regs_wait();
                pack_block_contiguous(0, args.out_cb, out_w);
                tile_regs_release();
            }

            cb_pop_front(args.in0_cb, out_w);
            cb_pop_front(args.in1_cb, args.total_in1_tiles);
            if constexpr (Has3Inputs) {
                cb_pop_front(args.sram_in_cb, out_w);
            }
            cb_push_back(args.out_cb, out_w);
#endif
        }
    };
};

}  // namespace deepseek_b1_ops
