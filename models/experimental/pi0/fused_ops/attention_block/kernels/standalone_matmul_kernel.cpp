// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Standalone wrapper kernel for pi05_siglip_ops::EncoderMatmul Op-struct.
//
// Proof-of-concept: same encoder-shape matmul that qkv_matmul_kernel.cpp
// runs, but dispatched through the new Op-struct. The Op-struct serves both
// QKV (N_per_core=3) and O-proj (N_per_core=1) shapes via CT-arg templating.

#include "../../unified_kernels/matmul.h"
#include "../../../../../demos/deepseek_v3_b1/unified_kernels/kernel_utils.hpp"

void kernel_main() {
#if defined(COMPILE_FOR_NCRISC)
    constexpr uint32_t act_cb = get_named_compile_time_arg_val("act_cb");
    constexpr uint32_t weights_cb = get_named_compile_time_arg_val("weights_cb");
    constexpr uint32_t act_tiles = get_named_compile_time_arg_val("act_tiles");
    constexpr uint32_t weights_tiles = get_named_compile_time_arg_val("weights_tiles");

    unified_kernels::setup_sharded_buffer(act_cb, act_tiles);
    unified_kernels::setup_sharded_buffer(weights_cb, weights_tiles);
#endif

#if defined(COMPILE_FOR_BRISC)
    // no-op: weights are L1-resident; no DRAM streaming.
#endif

#if defined(COMPILE_FOR_TRISC)
    constexpr uint32_t act_cb = get_named_compile_time_arg_val("act_cb");
    constexpr uint32_t weights_cb = get_named_compile_time_arg_val("weights_cb");
    constexpr uint32_t out_cb = get_named_compile_time_arg_val("out_cb");
    constexpr uint32_t m_tiles = get_named_compile_time_arg_val("m_tiles");
    constexpr uint32_t k_tiles = get_named_compile_time_arg_val("k_tiles");
    constexpr uint32_t n_tiles_per_core = get_named_compile_time_arg_val("n_tiles_per_core");
    constexpr uint32_t act_tiles = get_named_compile_time_arg_val("act_tiles");
    constexpr uint32_t weights_tiles = get_named_compile_time_arg_val("weights_tiles");

    using MatmulCTArgs = pi05_siglip_ops::EncoderMatmul::
        ComputeCTArgs<act_cb, weights_cb, out_cb, m_tiles, k_tiles, n_tiles_per_core, act_tiles, weights_tiles>;

    pi05_siglip_ops::EncoderMatmul::Op<MatmulCTArgs, true> matmul;
    pi05_siglip_ops::EncoderMatmul::RTArgs args{};
    matmul(args);
#endif
}
