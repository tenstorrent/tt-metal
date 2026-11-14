// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// [sdpa_prefill_splitk] Merge: per output row tile, the S slots' statistics (m_j, l_j) give the weights
// w_j = exp((m_j - M) * scale) * l_j with M = max_j m_j (every exponential <= 1), and the merged row is
// O = sum_j O_j * w_j / sum_j w_j (O_j softmax-normalised over slot j's keys). Weights in fp32 dest; the weighted
// sum accumulates in an fp32 CB (L1 accumulation), then packs to the output format.

#include <cstdint>

#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/exp.h"
#include "api/compute/eltwise_unary/recip.h"
#include "api/compute/eltwise_unary/relu.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/tile_move_copy.h"
#include "ttnn/operations/transformer/sdpa/device/kernels/compute/compute_common.hpp"

template <uint32_t cb>
ALWI void load(uint32_t tile, uint32_t idst) {
    copy_init(cb);
    copy_tile(cb, tile, idst);
}

template <uint32_t cb_out>
ALWI void pack_one(uint32_t idst) {
    tile_regs_commit();
    tile_regs_wait();
    CircularBuffer out(cb_out);
    out.reserve_back(1);
    pack_tile(idst, cb_out);
    out.push_back(1);
    tile_regs_release();
}

void kernel_main() {
    constexpr uint32_t S = get_compile_time_arg_val(0);
    constexpr uint32_t DHt = get_compile_time_arg_val(1);
    constexpr uint32_t scale_fp32 = get_compile_time_arg_val(2);
    constexpr uint32_t cb_stats_in = get_compile_time_arg_val(3);
    constexpr uint32_t cb_o_in = get_compile_time_arg_val(4);
    constexpr uint32_t cb_max = get_compile_time_arg_val(5);
    constexpr uint32_t cb_w = get_compile_time_arg_val(6);
    constexpr uint32_t cb_inv = get_compile_time_arg_val(7);
    constexpr uint32_t cb_wn = get_compile_time_arg_val(8);
    constexpr uint32_t cb_acc = get_compile_time_arg_val(9);
    constexpr uint32_t cb_out = get_compile_time_arg_val(10);
    // merged statistics output (M, then L = sum_j w_j per row tile): the input convention, so a merged result can
    // be merged again
    constexpr bool with_stats = get_compile_time_arg_val(11) == 1;
    constexpr uint32_t cb_sum = get_compile_time_arg_val(12);
    constexpr uint32_t cb_stats_out = get_compile_time_arg_val(13);

    const uint32_t num_rows = get_arg_val<uint32_t>(0);

    compute_kernel_hw_startup(cb_o_in, cb_wn, cb_acc);
    CircularBuffer stats(cb_stats_in);
    CircularBuffer w(cb_w);
    CircularBuffer inv(cb_inv);
    CircularBuffer mx(cb_max);

    for (uint32_t row = 0; row < num_rows; ++row) {
        stats.wait_front(2 * S);
        reconfig_data_format_srca(cb_stats_in);

        // M = max_j m_j as M + relu(m_j - M)
        pack_reconfig_data_format(cb_max);
        tile_regs_acquire();
        load<cb_stats_in>(0, 0);
        for (uint32_t j = 1; j < S; ++j) {
            load<cb_stats_in>(j, 1);
            sub_binary_tile_init();
            sub_binary_tile(1, 0, 1);
            relu_tile_init();
            relu_tile(1);
            add_binary_tile_init();
            add_binary_tile(0, 1, 0);
        }
        pack_one<cb_max>(0);

        // w_j = exp((m_j - M) * scale) * l_j
        mx.wait_front(1);
        if constexpr (with_stats) {
            pack_reconfig_data_format(cb_stats_out);
            tile_regs_acquire();
            load<cb_max>(0, 0);
            pack_one<cb_stats_out>(0);
        }
        pack_reconfig_data_format(cb_w);
        for (uint32_t j = 0; j < S; ++j) {
            tile_regs_acquire();
            load<cb_stats_in>(j, 0);
            load<cb_max>(0, 1);
            sub_binary_tile_init();
            sub_binary_tile(0, 1, 0);
            if constexpr (scale_fp32 != 0x3F800000u) {
                binop_with_scalar_tile_init();
                mul_unary_tile(0, scale_fp32);
            }
            exp_tile_init();
            exp_tile(0);
            load<cb_stats_in>(S + j, 1);
            mul_binary_tile_init();
            mul_binary_tile(0, 1, 0);
            pack_one<cb_w>(0);
        }
        mx.pop_front(1);
        stats.pop_front(2 * S);

        // L = sum_j w_j; inv = 1 / L
        w.wait_front(S);
        reconfig_data_format_srca(cb_w);
        pack_reconfig_data_format(cb_sum);
        tile_regs_acquire();
        load<cb_w>(0, 0);
        for (uint32_t j = 1; j < S; ++j) {
            load<cb_w>(j, 1);
            add_binary_tile_init();
            add_binary_tile(0, 1, 0);
        }
        pack_one<cb_sum>(0);
        CircularBuffer sum(cb_sum);
        sum.wait_front(1);
        if constexpr (with_stats) {
            pack_reconfig_data_format(cb_stats_out);
            tile_regs_acquire();
            load<cb_sum>(0, 0);
            pack_one<cb_stats_out>(0);
        }
        pack_reconfig_data_format(cb_inv);
        tile_regs_acquire();
        load<cb_sum>(0, 0);
        recip_tile_init();
        recip_tile(0);
        pack_one<cb_inv>(0);
        sum.pop_front(1);

        // wn_j = w_j * inv
        inv.wait_front(1);
        pack_reconfig_data_format(cb_wn);
        for (uint32_t j = 0; j < S; ++j) {
            tile_regs_acquire();
            load<cb_w>(j, 0);
            load<cb_inv>(0, 1);
            mul_binary_tile_init();
            mul_binary_tile(0, 1, 0);
            pack_one<cb_wn>(0);
        }
        inv.pop_front(1);
        w.pop_front(S);

        // O = sum_j O_j * wn_j (bcast cols), the first slot written, the rest L1-accumulated onto it
        mul_block_bcast_cols<1, DHt, false, false>(cb_o_in, cb_wn, cb_acc);
        for (uint32_t j = 1; j < S; ++j) {
            mul_block_bcast_cols<1, DHt, false, true>(cb_o_in, cb_wn, cb_acc);
        }
        reconfig_data_format_srca(cb_acc);
        pack_reconfig_data_format(cb_out);
        copy_block(cb_acc, cb_out, DHt);
    }
}
