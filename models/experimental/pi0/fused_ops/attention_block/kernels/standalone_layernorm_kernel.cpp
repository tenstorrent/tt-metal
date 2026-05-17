// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Standalone wrapper kernel for pi05_siglip_ops::LayerNorm Op-struct.
//
// Proof-of-concept: same 8-phase LN that siglip_layernorm_kernel.cpp runs,
// but dispatched through the new Op-struct. Validates the pattern carries
// the two-stage reduce + binary_op_init_common reset (LN-specific gotchas)
// before we apply it to matmul + SDPA.

#include "../../unified_kernels/ln.h"
#include "../../../../../demos/deepseek_v3_b1/unified_kernels/kernel_utils.hpp"

void kernel_main() {
#if defined(COMPILE_FOR_NCRISC)
    constexpr uint32_t in_cb = get_named_compile_time_arg_val("in_cb");
    constexpr uint32_t gamma_cb = get_named_compile_time_arg_val("gamma_cb");
    constexpr uint32_t beta_cb = get_named_compile_time_arg_val("beta_cb");
    constexpr uint32_t scaler_cb = get_named_compile_time_arg_val("scaler_cb");
    constexpr uint32_t ones_cb = get_named_compile_time_arg_val("ones_cb");
    constexpr uint32_t in_tiles = get_named_compile_time_arg_val("in_tiles");
    constexpr uint32_t gamma_tiles = get_named_compile_time_arg_val("gamma_tiles");

    unified_kernels::setup_sharded_buffer(in_cb, in_tiles);
    unified_kernels::setup_sharded_buffer(gamma_cb, gamma_tiles);
    unified_kernels::setup_sharded_buffer(beta_cb, gamma_tiles);
    unified_kernels::setup_sharded_buffer(scaler_cb, 1);
    unified_kernels::setup_sharded_buffer(ones_cb, 1);
#endif

#if defined(COMPILE_FOR_BRISC)
    // no-op: LN reads/writes sharded L1 only; no DRAM I/O for BRISC.
#endif

#if defined(COMPILE_FOR_TRISC)
    constexpr uint32_t in_cb = get_named_compile_time_arg_val("in_cb");
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
    constexpr uint32_t out_cb = get_named_compile_time_arg_val("out_cb");
    constexpr uint32_t d_tiles = get_named_compile_time_arg_val("d_tiles");
    constexpr uint32_t in_tiles = get_named_compile_time_arg_val("in_tiles");
    constexpr uint32_t eps_bits = get_named_compile_time_arg_val("eps_bits");

    using LNCTArgs = pi05_siglip_ops::LayerNorm::ComputeCTArgs<
        in_cb,
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
        out_cb,
        d_tiles,
        in_tiles,
        eps_bits>;

    pi05_siglip_ops::LayerNorm::Op<LNCTArgs, true> ln;
    pi05_siglip_ops::LayerNorm::RTArgs args{};
    ln(args);
#endif
}
