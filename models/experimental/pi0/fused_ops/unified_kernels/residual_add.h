// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// SigLIP residual-add Op-struct.
//
// Pattern ported from
//   models/demos/deepseek_v3_b1/unified_kernels/residual_add.hpp
// but with our SigLIP shapes: both inputs are full (M, D) HEIGHT_SHARDED
// bf16 tensors with one M-tile per core (8 cores × 32 rows), so each core
// adds its own local (32, D) shard pairwise — no cross-core gather, no
// per-core offset indexing.
//
// CB layout:
//   a_cb:    matmul/residual-output input (in_tiles, consumed)
//   b_cb:    skip-connection input        (in_tiles, consumed)
//   out_cb:  add output                   (in_tiles, produced)
//
// Used in attention sub-block:   out = O-proj(SDPA(LN1(x))) + x
// Used in MLP sub-block:         out = FC2(GELU(FC1(LN2(y)))) + y

#pragma once

#include "../../../../demos/deepseek_v3_b1/unified_kernels/kernel_op_api.hpp"

#if defined(COMPILE_FOR_TRISC)
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_binary.h"
#endif

namespace pi05_siglip_ops {

struct ResidualAdd {
    struct ReaderCTArgs {};
    struct WriterCTArgs {};

    template <uint32_t ACb, uint32_t BCb, uint32_t OutCb, uint32_t InTiles>
    struct ComputeCTArgs {
        static constexpr uint32_t a_cb = ACb;
        static constexpr uint32_t b_cb = BCb;
        static constexpr uint32_t out_cb = OutCb;
        static constexpr uint32_t in_tiles = InTiles;
    };

    struct ReaderArgs {};
    struct WriterArgs {};
    struct ComputeArgs {};

    using RTArgs = unified_kernels::SelectByRISCV<ReaderArgs, WriterArgs, ComputeArgs>;

    // Op is the per-phase entry point. IsActiveCore gates whether this core
    // executes the body, mirroring the deepseek role-flag pattern. A core
    // can carry multiple role flags (e.g. residual-core + LN1-core) so the
    // same binary runs every phase on every core; compile-time gating keeps
    // each phase's body resident only where it's needed.
    template <typename CTArgs, bool IsActiveCore>
    class Op {
    public:
        void operator()(const RTArgs& args) {
            if constexpr (IsActiveCore) {
                impl(args);
            }
        }

    private:
        void impl(const RTArgs& /*args*/) {
#if defined(COMPILE_FOR_TRISC)
            constexpr uint32_t a_cb = CTArgs::a_cb;
            constexpr uint32_t b_cb = CTArgs::b_cb;
            constexpr uint32_t out_cb = CTArgs::out_cb;
            constexpr uint32_t IN_TILES = CTArgs::in_tiles;

            cb_wait_front(a_cb, IN_TILES);
            cb_wait_front(b_cb, IN_TILES);

            // binary_op_init_common is the lesson from pi05-llk-binary-op-init-common:
            // without it TR0/TR1 hang. Each phase that uses binary LLKs must call
            // it once. In the fused kernel main, only the *first* binary phase
            // needs it strictly — but re-init per phase is cheap and safer when
            // composing multiple Op-structs that share dst.
            binary_op_init_common(a_cb, b_cb, out_cb);
            add_tiles_init(a_cb, b_cb);

            cb_reserve_back(out_cb, IN_TILES);
            for (uint32_t i = 0; i < IN_TILES; ++i) {
                tile_regs_acquire();
                add_tiles(a_cb, b_cb, i, i, 0);
                tile_regs_commit();

                tile_regs_wait();
                pack_tile(0, out_cb, i);
                tile_regs_release();
            }
            cb_push_back(out_cb, IN_TILES);
            cb_pop_front(a_cb, IN_TILES);
            cb_pop_front(b_cb, IN_TILES);
#endif
        }
    };
};

}  // namespace pi05_siglip_ops
