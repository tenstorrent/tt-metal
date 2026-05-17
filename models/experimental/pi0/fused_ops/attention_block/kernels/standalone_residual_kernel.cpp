// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Standalone wrapper kernel for pi05_siglip_ops::ResidualAdd Op-struct.
//
// This is the proof-of-concept that validates the Op-struct refactor pattern.
// It does the same work as siglip_residual_kernel.cpp (a + b → out) but
// dispatches through the new Op-struct so we can confirm the pattern compiles
// and PCC-matches before porting LN, matmul, and SDPA the same way.
//
// All CT-arg lookups must live inside per-RISC #if blocks: BRISC has no CT
// args wired (brisc_named_compile_time_args=[]), so a top-level lookup would
// fail brisc compile.

#include "../../unified_kernels/residual_add.h"
#include "../../../../../demos/deepseek_v3_b1/unified_kernels/kernel_utils.hpp"

void kernel_main() {
#if defined(COMPILE_FOR_NCRISC)
    constexpr uint32_t a_cb = get_named_compile_time_arg_val("a_cb");
    constexpr uint32_t b_cb = get_named_compile_time_arg_val("b_cb");
    constexpr uint32_t out_cb = get_named_compile_time_arg_val("out_cb");
    constexpr uint32_t in_tiles = get_named_compile_time_arg_val("in_tiles");

    unified_kernels::setup_sharded_buffer(a_cb, in_tiles);
    unified_kernels::setup_sharded_buffer(b_cb, in_tiles);
    unified_kernels::setup_sharded_buffer(out_cb, in_tiles);
#endif

#if defined(COMPILE_FOR_BRISC)
    // no-op: residual-add reads/writes sharded L1 only; no DRAM I/O for BRISC.
#endif

#if defined(COMPILE_FOR_TRISC)
    constexpr uint32_t a_cb = get_named_compile_time_arg_val("a_cb");
    constexpr uint32_t b_cb = get_named_compile_time_arg_val("b_cb");
    constexpr uint32_t out_cb = get_named_compile_time_arg_val("out_cb");
    constexpr uint32_t in_tiles = get_named_compile_time_arg_val("in_tiles");

    using ResidualAddCTArgs = pi05_siglip_ops::ResidualAdd::ComputeCTArgs<a_cb, b_cb, out_cb, in_tiles>;

    // IsActiveCore=true: every core in the residual grid runs the add.
    // (In the full fused attention_block kernel this becomes a role flag
    // like Core::is_residual_core so non-residual cores skip the body.)
    pi05_siglip_ops::ResidualAdd::Op<ResidualAddCTArgs, true> residual_add;
    pi05_siglip_ops::ResidualAdd::RTArgs args{};
    residual_add(args);
#endif
}
