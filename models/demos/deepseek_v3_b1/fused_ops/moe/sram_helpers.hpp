// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Per-op SRAM-dynamic invocation helpers.
//
// The SRAM expert pipeline runs the same MoeGather / GatedReduce / Mcast
// micro-ops as the DRAM/shared paths, but parameterized by n_sram_active —
// a runtime value (count of SRAM-flagged TopK winners) that's not known
// until after the gate kernel + index mcast. These helpers absorb the
// per-RISC arg patching so the orchestrator can call each SRAM op with a
// single line.
//
// Each helper:
//   - Patches the relevant arg field on the affected RISC (#if-guarded).
//   - Forwards to the underlying op.
//   - Is a no-op patch on RISCs that don't read the field (NCRISC for
//     gather/mcast, BRISC for gated_reduce).

#pragma once

#include "../../unified_kernels/eltwise_add_or_copy.hpp"
#include "../../unified_kernels/gated_reduce.hpp"
#include "../../unified_kernels/mcast.hpp"
#include "../../unified_kernels/moe_gather.hpp"

namespace deepseek_b1_ops {

// All helpers below early-return when n_sram_active == 0 so the orchestrator
// can drop its `if (n_sram_active > 0) { ... }` block-skip. Skipping uniformly
// across all RISCs (sender + receivers) is safe: every helper's underlying op
// is producer/consumer-paired within the SRAM pipeline (matmul → gather → GR
// → mcast → down_proj), and all participating cores compute the same
// n_sram_active from the (mcasted) index CB. With everyone skipping, no CB
// pushes happen — no padding to drain, no NOC writes, no semaphore incs.
//
// Exception: sram_invoke_eltwise_add_or_copy (down-merge) runs on every iter
// — it switches between add (sram + shared) and copy (shared only) based on
// n_sram_active, so it must always invoke the op.

// SRAM matmul (no runtime args). Used by sram_gate_proj, sram_up_proj,
// sram_down_proj — all CT-templated MatmulExpertCompressedSRAM::Op invocations.
template <typename SramMatmulOp>
FORCE_INLINE void sram_invoke_matmul(SramMatmulOp& op, uint32_t n_sram_active) {
    if (n_sram_active == 0) {
        return;
    }
    op();
}

// SRAM dynamic-size MoeGather. BRISC's `num_experts` becomes n_sram_active
// (drives per-core write count). NCRISC's `dst_num_pages` is set in the
// orchestrator's args initializer (compile-time const), unaffected here.
template <typename MoeGatherOp>
FORCE_INLINE void sram_invoke_moe_gather(MoeGatherOp& op, MoeGather::RTArgs& args, uint32_t n_sram_active) {
    if (n_sram_active == 0) {
        return;
    }
#if defined(COMPILE_FOR_BRISC)
    args.num_experts = n_sram_active;
#endif
    op(args);
}

// SRAM dynamic-size GatedReduce. TRISC's outer loop count k_num_tiles
// becomes n_sram_active. BRISC scalar-copy is independent (loops over
// CTArgs::num_active and self-filters).
template <typename GatedReduceOp>
FORCE_INLINE void sram_invoke_gated_reduce(GatedReduceOp& op, GatedReduce::RTArgs& args, uint32_t n_sram_active) {
    if (n_sram_active == 0) {
        return;
    }
#if defined(COMPILE_FOR_TRISC)
    args.k_num_tiles = n_sram_active;
#endif
    op(args);
}

// SRAM dynamic-size down-mcast (sender side). BRISC's data_size_bytes
// becomes n_sram_active * down_proj_mcast_per_expert_bytes (CT). num_pages
// is unchanged so receivers + matmul see consistent advance.
//
// The CT arg name is only registered on BRISC, so we read it inside the
// BRISC #ifdef — using it as a template NTTP would evaluate on TRISC too
// and trigger __builtin_unreachable().
template <typename McastOp>
FORCE_INLINE void sram_invoke_down_mcast(McastOp& op, Mcast::RTArgs& args, uint32_t n_sram_active) {
    if (n_sram_active == 0) {
        return;
    }
#if defined(COMPILE_FOR_BRISC)
    constexpr uint32_t per_expert_bytes = get_named_compile_time_arg_val("down_proj_mcast_per_expert_bytes");
    args.sender.data_size_bytes = n_sram_active * per_expert_bytes;
#endif
    op(args);
}

// SRAM down-merge: TRISC's `do_add` selects between
//   1 → eltwise_add(sram_down, shared_down) → merged
//   0 → copy(shared_down) → merged
// based on whether any SRAM expert fired this iter. BRISC/NCRISC pass through.
// Always invokes the underlying op — no early-return — since the copy path
// must run when n_sram_active==0 to keep the downstream wiring uniform.
template <typename EltwiseAddOrCopyOp>
FORCE_INLINE void sram_invoke_eltwise_add_or_copy(
    EltwiseAddOrCopyOp& op, EltwiseAddOrCopy::RTArgs& args, [[maybe_unused]] uint32_t n_sram_active) {
#if defined(COMPILE_FOR_TRISC)
    args.do_add = (n_sram_active > 0) ? 1u : 0u;
#endif
    op(args);
}

}  // namespace deepseek_b1_ops
