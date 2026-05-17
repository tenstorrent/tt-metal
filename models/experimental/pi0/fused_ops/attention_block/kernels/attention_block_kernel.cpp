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

// Core role flags — first increment is uniform (all 8 cores in the grid
// run both LN1 and residual). With more phases, this expands into a
// struct with per-phase is_<phase>_core flags gated by compile-time
// args set per core range in op.py.
struct Core {
    static constexpr bool is_ln1_core = true;
    static constexpr bool is_residual_core = true;
};

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

    // =========================================================================
    // PHASE 1: LN1 — y = ((x - mean) / sqrt(var + eps)) * gamma + beta
    //          Reads: ln_in_cb, gamma_cb, beta_cb, scaler_cb, ones_cb
    //          Writes: ln_out_cb (consumed by Phase 2)
    // =========================================================================
    {
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

        pi05_siglip_ops::LayerNorm::Op<LNCTArgs, Core::is_ln1_core> ln1;
        pi05_siglip_ops::LayerNorm::RTArgs ln_args{};
        ln1(ln_args);
    }

    // =========================================================================
    // PHASE 2: residual — out = ln_out + x_residual
    //          Reads: ln_out_cb (from Phase 1), x_residual_cb (separate L1 copy of x)
    //          Writes: final_out_cb
    // =========================================================================
    {
        using ResCTArgs = pi05_siglip_ops::ResidualAdd::ComputeCTArgs<ln_out_cb, x_residual_cb, final_out_cb, in_tiles>;

        pi05_siglip_ops::ResidualAdd::Op<ResCTArgs, Core::is_residual_core> residual;
        pi05_siglip_ops::ResidualAdd::RTArgs res_args{};
        residual(res_args);
    }
#endif
}
