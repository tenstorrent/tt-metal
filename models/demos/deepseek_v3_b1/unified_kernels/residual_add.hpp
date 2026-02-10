// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "kernel_op_api.hpp"

#if defined(COMPILE_FOR_TRISC)
#include "api/compute/eltwise_binary.h"
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
        void impl(const RTArgs& args) {
#if defined(COMPILE_FOR_TRISC)
            constexpr uint32_t out_w = CTArgs::out_w;

            binary_op_init_common(args.in0_cb, args.in1_cb, args.out_cb);
            add_tiles_init(args.in0_cb, args.in1_cb);

            cb_wait_front(args.in0_cb, out_w);
            cb_wait_front(args.in1_cb, args.total_in1_tiles);
            cb_reserve_back(args.out_cb, out_w);
            tile_regs_acquire();
            for (uint32_t j = 0; j < out_w; j++) {
                add_tiles(args.in0_cb, args.in1_cb, j, args.core_idx * out_w + j, j);
            }
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t j = 0; j < out_w; j++) {
                pack_tile(j, args.out_cb, j);
            }
            tile_regs_release();

            cb_pop_front(args.in0_cb, out_w);
            cb_pop_front(args.in1_cb, args.total_in1_tiles);
            cb_push_back(args.out_cb, out_w);
#endif
        }
    };
};

}  // namespace deepseek_b1_ops
