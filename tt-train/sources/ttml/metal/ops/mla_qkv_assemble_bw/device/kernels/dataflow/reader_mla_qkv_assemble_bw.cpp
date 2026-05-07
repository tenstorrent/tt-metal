// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "tt-train/sources/ttml/metal/common/dataflow_utils.hpp"

void kernel_main() {
    uint32_t runtime_args_counter = 0U;
    uint32_t dQ_addr = get_arg_val<uint32_t>(runtime_args_counter++);
    uint32_t dK_addr = get_arg_val<uint32_t>(runtime_args_counter++);
    uint32_t dV_addr = get_arg_val<uint32_t>(runtime_args_counter++);
    uint32_t num_blocks = get_arg_val<uint32_t>(runtime_args_counter++);
    uint32_t sb = get_arg_val<uint32_t>(runtime_args_counter++);             // s-tile-row in current batch
    uint32_t dQ_block_base = get_arg_val<uint32_t>(runtime_args_counter++);  // head 0 of (b, sb), w=0
    uint32_t dV_block_base = get_arg_val<uint32_t>(runtime_args_counter++);

    constexpr uint32_t cb_dq = tt::CBIndex::c_0;
    constexpr uint32_t cb_dknope = tt::CBIndex::c_1;
    constexpr uint32_t cb_dv = tt::CBIndex::c_2;
    constexpr uint32_t cb_dkpe_in = tt::CBIndex::c_3;

    constexpr uint32_t Th = get_compile_time_arg_val(0);
    constexpr uint32_t Tn = get_compile_time_arg_val(1);
    constexpr uint32_t Tr = get_compile_time_arg_val(2);
    constexpr uint32_t Tv = get_compile_time_arg_val(3);
    constexpr uint32_t n_heads = get_compile_time_arg_val(4);
    constexpr uint32_t kq_HtWt = get_compile_time_arg_val(5);  // S_t * Th
    constexpr uint32_t v_HtWt = get_compile_time_arg_val(6);   // S_t * Tv
    constexpr uint32_t S_t = get_compile_time_arg_val(7);

    constexpr auto dQ_args = TensorAccessorArgs<8>();
    constexpr auto dK_args = TensorAccessorArgs<dQ_args.next_compile_time_args_offset()>();
    constexpr auto dV_args = TensorAccessorArgs<dK_args.next_compile_time_args_offset()>();

    const auto dQ_addr_gen = TensorAccessor(dQ_args, dQ_addr);
    const auto dK_addr_gen = TensorAccessor(dK_args, dK_addr);
    const auto dV_addr_gen = TensorAccessor(dV_args, dV_addr);

    // dK shares dQ's per-head head_dim, so kq_HtWt is the head stride for both.
    // End-of-batch jump to advance dQ_block_base across batch boundary:
    //   from b * H*kq_HtWt + (S_t-1)*Th  →  (b+1) * H*kq_HtWt
    //   = current + ((H-1)*S_t + 1) * Th
    constexpr uint32_t end_of_batch_jump_q = ((n_heads - 1U) * S_t + 1U) * Th;
    constexpr uint32_t end_of_batch_jump_v = ((n_heads - 1U) * S_t + 1U) * Tv;

    for (uint32_t block = 0U; block < num_blocks; ++block) {
        for (uint32_t h = 0U; h < n_heads; ++h) {
            const uint32_t head_dq = dQ_block_base + h * kq_HtWt;
            const uint32_t head_dk = dQ_block_base + h * kq_HtWt;  // dK same layout as dQ
            const uint32_t head_dv = dV_block_base + h * v_HtWt;

            // Th dQ tiles
            for (uint32_t w = 0U; w < Th; ++w) {
                cb_reserve_back(cb_dq, onetile);
                noc_async_read_page(head_dq + w, dQ_addr_gen, get_write_ptr(cb_dq));
                noc_async_read_barrier();
                cb_push_back(cb_dq, onetile);
            }

            // Tn dK_nope tiles
            for (uint32_t w = 0U; w < Tn; ++w) {
                cb_reserve_back(cb_dknope, onetile);
                noc_async_read_page(head_dk + w, dK_addr_gen, get_write_ptr(cb_dknope));
                noc_async_read_barrier();
                cb_push_back(cb_dknope, onetile);
            }

            // Tr dK_pe tiles → compute
            for (uint32_t w = 0U; w < Tr; ++w) {
                cb_reserve_back(cb_dkpe_in, onetile);
                noc_async_read_page(head_dk + Tn + w, dK_addr_gen, get_write_ptr(cb_dkpe_in));
                noc_async_read_barrier();
                cb_push_back(cb_dkpe_in, onetile);
            }

            // Tv dV tiles
            for (uint32_t w = 0U; w < Tv; ++w) {
                cb_reserve_back(cb_dv, onetile);
                noc_async_read_page(head_dv + w, dV_addr_gen, get_write_ptr(cb_dv));
                noc_async_read_barrier();
                cb_push_back(cb_dv, onetile);
            }
        }

        // Advance to next block.
        ++sb;
        if (sb < S_t) {
            dQ_block_base += Th;
            dV_block_base += Tv;
        } else {
            sb = 0U;
            dQ_block_base += end_of_batch_jump_q;
            dV_block_base += end_of_batch_jump_v;
        }
    }
}
