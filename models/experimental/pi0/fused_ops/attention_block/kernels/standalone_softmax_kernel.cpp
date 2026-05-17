// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Standalone wrapper for pi05_siglip_ops::Softmax Op-struct.
//
// Used inside SDPA per head between QK^T and Attn@V matmuls. Standalone POC
// here so we can confirm the Op-struct produces bit-identical output to the
// monolithic siglip_softmax_kernel.cpp before composing into attention_block.

#include "../../unified_kernels/softmax.h"
#include "../../../../../demos/deepseek_v3_b1/unified_kernels/kernel_utils.hpp"

void kernel_main() {
#if defined(COMPILE_FOR_NCRISC)
    constexpr uint32_t in_cb = get_named_compile_time_arg_val("in_cb");
    constexpr uint32_t scaler_cb = get_named_compile_time_arg_val("scaler_cb");
    constexpr uint32_t out_cb = get_named_compile_time_arg_val("out_cb");
    constexpr uint32_t in_tiles = get_named_compile_time_arg_val("in_tiles");

    unified_kernels::setup_sharded_buffer(in_cb, in_tiles);
    unified_kernels::setup_sharded_buffer(scaler_cb, 1);
    unified_kernels::setup_sharded_buffer(out_cb, in_tiles);
#endif

#if defined(COMPILE_FOR_BRISC)
    // no-op: softmax reads/writes sharded L1 only.
#endif

#if defined(COMPILE_FOR_TRISC)
    constexpr uint32_t in_cb = get_named_compile_time_arg_val("in_cb");
    constexpr uint32_t scaler_cb = get_named_compile_time_arg_val("scaler_cb");
    constexpr uint32_t max_cb = get_named_compile_time_arg_val("max_cb");
    constexpr uint32_t exp_cb = get_named_compile_time_arg_val("exp_cb");
    constexpr uint32_t sum_cb = get_named_compile_time_arg_val("sum_cb");
    constexpr uint32_t isum_cb = get_named_compile_time_arg_val("isum_cb");
    constexpr uint32_t out_cb = get_named_compile_time_arg_val("out_cb");
    constexpr uint32_t k_tiles = get_named_compile_time_arg_val("k_tiles");
    constexpr uint32_t m_tiles = get_named_compile_time_arg_val("m_tiles");
    constexpr uint32_t in_tiles = get_named_compile_time_arg_val("in_tiles");

    using SMCTArgs = pi05_siglip_ops::Softmax::
        ComputeCTArgs<in_cb, scaler_cb, max_cb, exp_cb, sum_cb, isum_cb, out_cb, k_tiles, m_tiles, in_tiles>;

    pi05_siglip_ops::Softmax::Op<SMCTArgs, true> softmax;
    pi05_siglip_ops::Softmax::RTArgs args{};
    softmax(args);
#endif
}
