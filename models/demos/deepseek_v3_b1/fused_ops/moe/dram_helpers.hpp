// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Per-op DRAM-skip invocation helpers.
//
// When all TopK winners are SRAM-flagged (n_dram_active == 0), the entire
// DRAM gate_proj → up_proj → mul → gather → mcast → down_proj → eltwise_add
// chain is no work — the SRAM path produces the routed contribution and the
// final eltwise_add must pass shared_output through unmodified. These helpers
// give each DRAM op a clean per-iter skip:
//
//   - The matmul/mul/gather/mcast helpers early-return: no CB push, no NOC
//     write, no semaphore inc. Producer/consumer-paired within the DRAM
//     chain so all participating cores skip uniformly (sender_core via the
//     pre-mcast scan, streamer cores via the post-mcast scan, every core
//     derives n_dram_active = num_active_experts - n_sram_active).
//
//   - dram_invoke_eltwise_add always invokes (the eltwise_add output feeds
//     reduce_to_one regardless of routing), switching do_add at runtime:
//       1 → out = down_proj_out + shared_output  (normal DRAM contribution)
//       0 → out = shared_output                  (SRAM-only path)

#pragma once

#include "../../unified_kernels/eltwise_add_or_copy.hpp"
#include "../../unified_kernels/eltwise_mul.hpp"
#include "../../unified_kernels/matmul_expert_compressed_dram.hpp"
#include "../../unified_kernels/mcast.hpp"
#include "../../unified_kernels/moe_gather.hpp"

namespace deepseek_b1_ops {

// DRAM matmul (gate_proj / up_proj / down_proj). The kernel itself iterates
// num_active_experts and filters via is_sram_expert, so the early-return is
// exactly equivalent to running the kernel with num_dram_active=0. Any CBs
// the kernel would have popped (e.g., down_proj's pop_index on cb_index) are
// re-anchored by RECONFIG_MOE_CBS at the top of the next iter — no manual
// pop needed when skipped.
template <typename DramMatmulOp>
FORCE_INLINE void dram_invoke_matmul(DramMatmulOp& op, uint32_t n_dram_active) {
    if (n_dram_active == 0) {
        return;
    }
    op();
}

// EltwiseMul (silu(gate) * scale * up). Skipping is uniform across BRISC
// (scalar copy) and TRISC (multiply). Safe because mcast_expert_scale (the
// only producer of mul_cb_scalar_src) and gate/up DRAM matmul (producers of
// cb_in0/cb_in1) are skipped together — no orphan pushes to drain.
template <typename EltwiseMulOp>
FORCE_INLINE void dram_invoke_eltwise_mul(EltwiseMulOp& op, uint32_t n_dram_active) {
    if (n_dram_active == 0) {
        return;
    }
    op();
}

// MoeGather (down_proj gather: gate_proj cores → sender). The gather sends
// num_active × per_core_n tiles per sender; the DRAM matmul's zero-push
// padding (matmul_expert_compressed_dram.hpp:1087-1095) keeps that count
// deterministic — but only when the matmul ran. When n_dram_active==0 the
// matmul is skipped, so the gather must skip too.
template <typename MoeGatherOp, typename Args>
FORCE_INLINE void dram_invoke_moe_gather(MoeGatherOp& op, const Args& args, uint32_t n_dram_active) {
    if (n_dram_active == 0) {
        return;
    }
    op(args);
}

// Mcast (down_proj mcast / expert_scale_mcast). Skipping uniformly across
// sender + mcast_grid + receivers omits sender's NOC write, sem inc, and
// receiver's cb_push_back, so all three roles stay in lock-step. Sender's
// pop_src=true normally drains the source CB; any unpopped source pages are
// re-anchored by RECONFIG_MOE_CBS at the top of the next iter.
template <typename McastOp, typename Args>
FORCE_INLINE void dram_invoke_mcast(McastOp& op, const Args& args, uint32_t n_dram_active) {
    if (n_dram_active == 0) {
        return;
    }
    op(args);
}

// Final DRAM eltwise_add — ALWAYS runs. do_add=1 fires the down_proj_out +
// shared_output add; do_add=0 copies shared_output through to the output CB
// (which feeds reduce_to_one). Mirrors sram_invoke_eltwise_add_or_copy.
template <typename EltwiseAddOrCopyOp>
FORCE_INLINE void dram_invoke_eltwise_add_or_copy(
    EltwiseAddOrCopyOp& op, EltwiseAddOrCopy::RTArgs& args, [[maybe_unused]] uint32_t n_dram_active) {
#if defined(COMPILE_FOR_TRISC)
    args.do_add = (n_dram_active > 0) ? 1u : 0u;
#endif
    op(args);
}

}  // namespace deepseek_b1_ops
