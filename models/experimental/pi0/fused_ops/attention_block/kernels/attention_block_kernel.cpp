// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Fused SigLIP attention sub-block kernel — first increment.
//
// Eventual target: LN1 → QKV → 16× SDPA-head → O-proj → residual in one
// TRISC dispatch on a 128-core BH chip grid (16 × 8 = 128 with 2 spare).
//
// This first increment fuses LN1 + residual only, on a single 8-core row
// (same shard layout the standalone primitives use). Math is contrived
// (out = LN1(x) + x — not a real attention sub-block since QKV/SDPA/O-proj
// are skipped) but validates the multi-Op chaining mechanic on which the
// full composition will be built.
//
// Why this first cut:
//   * Same 8-core grid for both phases → no fan-out/fan-in design yet.
//   * Both Op-structs already PCC- and bit-identical-validated.
//   * Two separate input CBs for x (input_cb for LN, x_residual_cb for
//     residual b-input): trades a small L1 duplicate for not having to
//     change LN1's cb_pop_front discipline.
//
// Next increments will (a) add QKV with a mcast/fan-out from 8→36 cores
// between LN1 and QKV, (b) add the 16-head SDPA loop on a wider grid,
// (c) add O-proj fan-in and final residual.

#include "../../unified_kernels/ln.h"
#include "../../unified_kernels/residual_add.h"
#include "../../../../../demos/deepseek_v3_b1/unified_kernels/kernel_utils.hpp"
// mcast.hpp — included to make deepseek_b1_ops::Mcast::Op visible for the
// upcoming 8→36 fan-out between LN1 and QKV (task #10). At this stage we
// only verify the include compiles in our fused_ops context; we don't yet
// instantiate or invoke it. The actual mcast wiring + QKV phase comes in a
// follow-up commit on this same branch.
#include "../../../../../demos/deepseek_v3_b1/unified_kernels/mcast.hpp"

// Role gating is done at runtime via get_relative_logical_x()/y() — the
// Op-struct's IsActiveCore template is hardcoded to true at every call site,
// and a runtime `if` skips the call on cores that don't participate in that
// phase. For this first increment LN1 and residual share the same 8-core row
// (y=0, x=0..7), so the runtime branches are tautologies; the structural
// pattern is what task #10's later commits need (LN1 row + larger QKV grid
// will require per-phase runtime gating once the grid expands).
//
// Why not compile-time per-range gating (one descriptor per CoreRange with
// different CT args)? Once we add QKV and the 8→36 fan-out, the participating
// sub-ranges overlap and multiply; three+ descriptors with bespoke CT args
// would balloon the host-side wiring. Runtime branching costs a few extra
// instructions and some compiled-out code per core but keeps op.py to a
// single kernel descriptor over the union grid. Compile-time gating can be
// reintroduced as a perf pass once the math is locked in.

void kernel_main() {
#if defined(COMPILE_FOR_NCRISC)
    // LN1 inputs.
    constexpr uint32_t ln_in_cb = get_named_compile_time_arg_val("ln_in_cb");
    constexpr uint32_t gamma_cb = get_named_compile_time_arg_val("gamma_cb");
    constexpr uint32_t beta_cb = get_named_compile_time_arg_val("beta_cb");
    constexpr uint32_t scaler_cb = get_named_compile_time_arg_val("scaler_cb");
    constexpr uint32_t ones_cb = get_named_compile_time_arg_val("ones_cb");
    // Residual second input (separate copy of x).
    constexpr uint32_t x_residual_cb = get_named_compile_time_arg_val("x_residual_cb");
    // Final output.
    constexpr uint32_t final_out_cb = get_named_compile_time_arg_val("final_out_cb");

    constexpr uint32_t in_tiles = get_named_compile_time_arg_val("in_tiles");
    constexpr uint32_t gamma_tiles = get_named_compile_time_arg_val("gamma_tiles");

    unified_kernels::setup_sharded_buffer(ln_in_cb, in_tiles);
    unified_kernels::setup_sharded_buffer(gamma_cb, gamma_tiles);
    unified_kernels::setup_sharded_buffer(beta_cb, gamma_tiles);
    unified_kernels::setup_sharded_buffer(scaler_cb, 1);
    unified_kernels::setup_sharded_buffer(ones_cb, 1);
    unified_kernels::setup_sharded_buffer(x_residual_cb, in_tiles);
    unified_kernels::setup_sharded_buffer(final_out_cb, in_tiles);
#endif

#if defined(COMPILE_FOR_BRISC)
    // no-op: all data is sharded L1; no DRAM streaming in this fused path.
#endif

#if defined(COMPILE_FOR_TRISC)
    constexpr uint32_t ln_in_cb = get_named_compile_time_arg_val("ln_in_cb");
    constexpr uint32_t gamma_cb = get_named_compile_time_arg_val("gamma_cb");
    constexpr uint32_t beta_cb = get_named_compile_time_arg_val("beta_cb");
    constexpr uint32_t scaler_cb = get_named_compile_time_arg_val("scaler_cb");
    constexpr uint32_t ones_cb = get_named_compile_time_arg_val("ones_cb");
    constexpr uint32_t accum_cb = get_named_compile_time_arg_val("accum_cb");
    constexpr uint32_t xmm_cb = get_named_compile_time_arg_val("xmm_cb");
    constexpr uint32_t xmm2_cb = get_named_compile_time_arg_val("xmm2_cb");
    constexpr uint32_t mean_cb = get_named_compile_time_arg_val("mean_cb");
    constexpr uint32_t var_cb = get_named_compile_time_arg_val("var_cb");
    constexpr uint32_t ivar_cb = get_named_compile_time_arg_val("ivar_cb");
    // ln_out_cb is the chaining buffer: LN1 writes it, residual reads it as a_cb.
    constexpr uint32_t ln_out_cb = get_named_compile_time_arg_val("ln_out_cb");
    constexpr uint32_t x_residual_cb = get_named_compile_time_arg_val("x_residual_cb");
    constexpr uint32_t final_out_cb = get_named_compile_time_arg_val("final_out_cb");

    constexpr uint32_t d_tiles = get_named_compile_time_arg_val("d_tiles");
    constexpr uint32_t in_tiles = get_named_compile_time_arg_val("in_tiles");
    constexpr uint32_t eps_bits = get_named_compile_time_arg_val("eps_bits");

    // Runtime role flags. For Commit 1 of task #10 the kernel grid is the
    // 8-core row (y=0, x=0..7) — both phases participate on every core, so
    // these branches are tautologies. They become load-bearing in Commit 2
    // when the grid expands to LN1 ∪ QKV (y=0 x=0..7 plus y=1..5 x=0..5).
    const bool is_ln1_core = (get_relative_logical_y() == 0) && (get_relative_logical_x() < 8);
    const bool is_residual_core = is_ln1_core;

    // =========================================================================
    // PHASE 1: LN1 — y = ((x - mean) / sqrt(var + eps)) * gamma + beta
    //          Reads: ln_in_cb, gamma_cb, beta_cb, scaler_cb, ones_cb
    //          Writes: ln_out_cb (consumed by Phase 2)
    // =========================================================================
    if (is_ln1_core) {
        using LNCTArgs = pi05_siglip_ops::LayerNorm::ComputeCTArgs<
            ln_in_cb,
            gamma_cb,
            beta_cb,
            scaler_cb,
            ones_cb,
            accum_cb,
            xmm_cb,
            xmm2_cb,
            mean_cb,
            var_cb,
            ivar_cb,
            ln_out_cb,
            d_tiles,
            in_tiles,
            eps_bits>;

        pi05_siglip_ops::LayerNorm::Op<LNCTArgs, true> ln1;
        pi05_siglip_ops::LayerNorm::RTArgs ln_args{};
        ln1(ln_args);
    }

    // =========================================================================
    // PHASE 2: residual — out = ln_out + x_residual
    //          Reads: ln_out_cb (from Phase 1), x_residual_cb (separate L1 copy of x)
    //          Writes: final_out_cb
    // =========================================================================
    if (is_residual_core) {
        using ResCTArgs = pi05_siglip_ops::ResidualAdd::ComputeCTArgs<ln_out_cb, x_residual_cb, final_out_cb, in_tiles>;

        pi05_siglip_ops::ResidualAdd::Op<ResCTArgs, true> residual;
        pi05_siglip_ops::ResidualAdd::RTArgs res_args{};
        residual(res_args);
    }
#endif
}
