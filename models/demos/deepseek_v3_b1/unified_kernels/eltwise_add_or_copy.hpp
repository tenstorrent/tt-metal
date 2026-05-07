// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "kernel_op_api.hpp"

#if defined(COMPILE_FOR_TRISC)
#include "api/compute/eltwise_binary.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/compute_kernel_api.h"
using namespace ckernel;
#endif

namespace deepseek_b1_ops {

// ============================================================================
// EltwiseAddOrCopy micro-op: per-iter elementwise add OR copy, runtime-selected.
//
//   do_add=1: out = in0 + in1  (cb_in0 + cb_in1, both consumed)
//   do_add=0: out = in1        (cb_in1 only, cb_in0 untouched)
//
// Lets the call site keep one CB-flow path while branching on a runtime flag —
// useful when one of the inputs is conditionally produced (e.g. SRAM down +
// shared down on the 112 mcast receiver cores: do_add=1 when n_sram_active>0,
// else copy shared down through to merged_down_out_cb so residual_add's input
// is uniform across MoE / dense-MLP / no-SRAM cases).
//
// CB layout:
//   in0: num_tiles per core, only consumed when do_add=1.
//   in1: num_tiles per core, always consumed.
//   out: num_tiles per core, always produced.
// ============================================================================
struct EltwiseAddOrCopy {
    struct ReaderCTArgs {};
    struct WriterCTArgs {};

    template <uint32_t Cb_in0, uint32_t Cb_in1, uint32_t Cb_out, uint32_t NumTiles>
    struct ComputeCTArgs {
        static constexpr uint32_t cb_in0 = Cb_in0;
        static constexpr uint32_t cb_in1 = Cb_in1;
        static constexpr uint32_t cb_out = Cb_out;
        static constexpr uint32_t num_tiles = NumTiles;
    };

    struct ReaderArgs {};
    struct WriterArgs {};

    struct ComputeArgs {
        uint32_t do_add;  // 1 = eltwise_add(in0, in1), 0 = copy(in1)
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
#if defined(COMPILE_FOR_TRISC)
            constexpr uint32_t cb_in0 = CTArgs::cb_in0;
            constexpr uint32_t cb_in1 = CTArgs::cb_in1;
            constexpr uint32_t cb_out = CTArgs::cb_out;
            constexpr uint32_t num_tiles = CTArgs::num_tiles;

            cb_wait_front(cb_in1, num_tiles);
            cb_reserve_back(cb_out, num_tiles);

            if (args.do_add) {
                cb_wait_front(cb_in0, num_tiles);
                reconfig_data_format<false, true>(cb_in0, cb_in1);
                pack_reconfig_data_format<true>(cb_out);
                add_tiles_init(cb_in0, cb_in1);
                tile_regs_acquire();
                for (uint32_t i = 0; i < num_tiles; i++) {
                    add_tiles(cb_in0, cb_in1, i, i, i);
                }
                tile_regs_commit();
                tile_regs_wait();
                for (uint32_t i = 0; i < num_tiles; i++) {
                    pack_tile(i, cb_out);
                }
                tile_regs_release();
                cb_pop_front(cb_in0, num_tiles);
            } else {
                reconfig_data_format<false, true>(cb_in1, cb_in1);
                pack_reconfig_data_format<true>(cb_out);
                copy_tile_to_dst_init_short(cb_in1);
                tile_regs_acquire();
                for (uint32_t i = 0; i < num_tiles; i++) {
                    copy_tile(cb_in1, i, i);
                }
                tile_regs_commit();
                tile_regs_wait();
                for (uint32_t i = 0; i < num_tiles; i++) {
                    pack_tile(i, cb_out);
                }
                tile_regs_release();
            }

            cb_pop_front(cb_in1, num_tiles);
            cb_push_back(cb_out, num_tiles);
#endif
        }
    };
};

}  // namespace deepseek_b1_ops
