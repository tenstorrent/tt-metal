// SPDX-FileCopyrightText: © 2026 Olof Johansson <olof@lixom.net>
// SPDX-License-Identifier: Apache-2.0
//
// Fused decode router, compute. Every core: one fp32 output tile of the K-split router GEMM (matmul_tiles
// accumulating in DEST, as ttnn's bmm). The collector then sums the 9 partials in chunk order on the SFPU (as
// ttnn.sum's accurate fp32 FastReduceNC path), and runs the moe_grouped_topk blocks: sigmoid + bias, and after
// the writer's top-k, normalize / scale. All values are bit-equal to that op chain.
#include "ttnn/operations/experimental/deepseek_prefill/moe_grouped_topk/device/kernels/compute/moe_gate_common_compute.hpp"
#include "api/compute/matmul.h"

void kernel_main() {
    constexpr uint32_t kt = get_named_compile_time_arg_val("kt");
    constexpr uint32_t chunks = get_named_compile_time_arg_val("chunks");
    constexpr uint32_t cb_in_scores = 0, cb_in_bias = 1, cb_out_weights = 2, cb_sigmoid = 4, cb_biased = 5;
    constexpr uint32_t cb_x = 6, cb_w = 7, cb_mm = 8, cb_parts = 9;
    constexpr uint32_t cb_reduce_intermediate = 15, cb_ones = 17, cb_eps = 18, cb_scale = 19, cb_normalized = 20;
    constexpr uint32_t cb_recip = 21, cb_gathered = 22;
    const uint32_t collector = get_arg_val<uint32_t>(0);

    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_x, cb_w, cb_mm);
    matmul_init(cb_x, cb_w);
    cb_wait_front(cb_x, kt);
    cb_wait_front(cb_w, kt);
    tile_regs_acquire();
    for (uint32_t k = 0; k < kt; ++k) {
        matmul_tiles(cb_x, cb_w, k, k, 0);
    }
    tile_regs_commit();
    cb_reserve_back(cb_mm, 1);
    tile_regs_wait();
    pack_tile(0, cb_mm);
    tile_regs_release();
    cb_push_back(cb_mm, 1);
    cb_pop_front(cb_x, kt);
    cb_pop_front(cb_w, kt);
    if (!collector) {
        return;
    }

    // logits = p0 + p1 + ... + p8 (dense pages, expert e at element e)
    cb_wait_front(cb_parts, chunks);
    reconfig_data_format_srca(cb_parts);
    pack_reconfig_data_format(cb_in_scores);
    copy_init(cb_parts);
    add_binary_tile_init();
    tile_regs_acquire();
    copy_tile(cb_parts, 0, 0);
    for (uint32_t c = 1; c < chunks; ++c) {
        copy_tile(cb_parts, c, 1);
        add_binary_tile(0, 1, 0);
    }
    tile_regs_commit();
    cb_reserve_back(cb_in_scores, 1);
    tile_regs_wait();
    pack_tile(0, cb_in_scores);
    tile_regs_release();
    cb_push_back(cb_in_scores, 1);
    cb_pop_front(cb_parts, chunks);

    blocks::apply_score_func_add_bias<blocks::SCORE_FUNC_SIGMOID, false>(
        cb_in_scores, cb_in_bias, cb_sigmoid, cb_biased, 1);
    blocks::normalize_scores(cb_gathered, cb_ones, cb_reduce_intermediate, cb_recip, cb_eps, cb_normalized);
    blocks::scale(cb_normalized, cb_scale, cb_out_weights);
}
