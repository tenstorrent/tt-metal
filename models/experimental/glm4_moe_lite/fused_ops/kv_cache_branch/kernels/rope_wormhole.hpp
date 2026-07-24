// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "../../../../../demos/deepseek_v3_b1/unified_kernels/kernel_op_api.hpp"

#if defined(COMPILE_FOR_TRISC)
#include "api/compute/bcast.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/matmul.h"
#include "api/compute/tile_move_copy.h"
#endif

namespace glm4_rope {

// Compatibility wrapper for the split cos/sin CB layout used by the GLM fused
// KV branch. DeepSeek's shared Rope micro-op now expects one combined cos/sin
// CB and owns the DRAM reads, while this kernel loads the two CBs itself.
struct Rope {
    struct WriterCTArgs {};

    template <uint32_t Wt_, uint32_t Ht_>
    struct ComputeCTArgs {
        static constexpr uint32_t Wt = Wt_;
        static constexpr uint32_t Ht = Ht_;
    };

    struct WriterArgs {};

    struct ComputeArgs {
        uint32_t in_cb;
        uint32_t cos_cb;
        uint32_t sin_cb;
        uint32_t trans_mat_cb;
        uint32_t rotated_in_interm_cb;
        uint32_t cos_interm_cb;
        uint32_t sin_interm_cb;
        uint32_t out_cb;
    };

    using RTArgs = unified_kernels::SelectByRISCV<WriterArgs, WriterArgs, ComputeArgs>;

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
            constexpr uint32_t Wt = CTArgs::Wt;
            constexpr uint32_t Ht = CTArgs::Ht;

            reconfig_data_format_srcb<false, true>(args.in_cb);
            pack_reconfig_data_format<true>(args.out_cb);

            cb_wait_front(args.trans_mat_cb, 1);
            cb_wait_front(args.cos_cb, Wt);
            cb_wait_front(args.sin_cb, Wt);

            for (uint32_t ht = 0; ht < Ht; ++ht) {
                reconfig_data_format_srca<false, true>(args.trans_mat_cb);
                cb_reserve_back(args.rotated_in_interm_cb, Wt);
                cb_reserve_back(args.sin_interm_cb, Wt);
                cb_reserve_back(args.cos_interm_cb, Wt);
                cb_reserve_back(args.out_cb, Wt);
                cb_wait_front(args.in_cb, Wt);

                mm_init_short(args.in_cb, args.trans_mat_cb);
                tile_regs_acquire();
                for (uint32_t j = 0; j < Wt; ++j) {
                    matmul_tiles(args.in_cb, args.trans_mat_cb, j, 0, j);
                }
                tile_regs_commit();
                tile_regs_wait();
                for (uint32_t j = 0; j < Wt; ++j) {
                    pack_tile(j, args.rotated_in_interm_cb, j);
                }
                tile_regs_release();
                cb_push_back(args.rotated_in_interm_cb, Wt);

                cb_wait_front(args.rotated_in_interm_cb, Wt);
                reconfig_data_format_srca<false, true>(args.rotated_in_interm_cb);
                mul_bcast_rows_init_short(args.rotated_in_interm_cb, args.sin_cb);
                tile_regs_acquire();
                for (uint32_t j = 0; j < Wt; ++j) {
                    mul_tiles_bcast<BroadcastType::ROW>(args.rotated_in_interm_cb, args.sin_cb, j, j, j);
                }
                tile_regs_commit();
                tile_regs_wait();
                for (uint32_t j = 0; j < Wt; ++j) {
                    pack_tile(j, args.sin_interm_cb);
                }
                tile_regs_release();
                cb_push_back(args.sin_interm_cb, Wt);
                cb_pop_front(args.rotated_in_interm_cb, Wt);

                reconfig_data_format_srca<false, true>(args.in_cb);
                mul_bcast_rows_init_short(args.in_cb, args.cos_cb);
                tile_regs_acquire();
                for (uint32_t j = 0; j < Wt; ++j) {
                    mul_tiles_bcast<BroadcastType::ROW>(args.in_cb, args.cos_cb, j, j, j);
                }
                tile_regs_commit();
                tile_regs_wait();
                for (uint32_t j = 0; j < Wt; ++j) {
                    pack_tile(j, args.cos_interm_cb);
                }
                tile_regs_release();
                cb_push_back(args.cos_interm_cb, Wt);
                cb_pop_front(args.in_cb, Wt);

                cb_wait_front(args.cos_interm_cb, Wt);
                cb_wait_front(args.sin_interm_cb, Wt);
                add_tiles_init(args.cos_interm_cb, args.sin_interm_cb);
                tile_regs_acquire();
                for (uint32_t j = 0; j < Wt; ++j) {
                    add_tiles(args.cos_interm_cb, args.sin_interm_cb, j, j, j);
                }
                tile_regs_commit();
                tile_regs_wait();
                for (uint32_t j = 0; j < Wt; ++j) {
                    pack_tile(j, args.out_cb, j);
                }
                tile_regs_release();
                cb_push_back(args.out_cb, Wt);
                cb_pop_front(args.cos_interm_cb, Wt);
                cb_pop_front(args.sin_interm_cb, Wt);
            }

            cb_pop_front(args.cos_cb, Wt);
            cb_pop_front(args.sin_cb, Wt);
#endif
        }
    };
};

}  // namespace glm4_rope
