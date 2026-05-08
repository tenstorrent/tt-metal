// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "tt-train/sources/ttml/metal/common/dataflow_utils.hpp"

void kernel_main() {
    uint32_t runtime_args_counter = 0U;
    uint32_t dq_pre_addr = get_arg_val<uint32_t>(runtime_args_counter++);
    uint32_t dkv_up_addr = get_arg_val<uint32_t>(runtime_args_counter++);
    uint32_t dk_pe_addr = get_arg_val<uint32_t>(runtime_args_counter++);
    uint32_t num_blocks = get_arg_val<uint32_t>(runtime_args_counter++);
    uint32_t dq_pre_block_base = get_arg_val<uint32_t>(runtime_args_counter++);
    uint32_t dkv_up_block_base = get_arg_val<uint32_t>(runtime_args_counter++);
    uint32_t dk_pe_block_base = get_arg_val<uint32_t>(runtime_args_counter++);

    constexpr uint32_t cb_dq = tt::CBIndex::c_0;
    constexpr uint32_t cb_dknope = tt::CBIndex::c_1;
    constexpr uint32_t cb_dv = tt::CBIndex::c_2;
    constexpr uint32_t cb_dkpe_out = tt::CBIndex::c_4;

    constexpr uint32_t Th = get_compile_time_arg_val(0);
    constexpr uint32_t Tn = get_compile_time_arg_val(1);
    constexpr uint32_t Tr = get_compile_time_arg_val(2);
    constexpr uint32_t Tv = get_compile_time_arg_val(3);
    constexpr uint32_t n_heads = get_compile_time_arg_val(4);
    constexpr uint32_t kv_per_head = Tn + Tv;

    constexpr auto dq_pre_args = TensorAccessorArgs<5>();
    constexpr auto dkv_up_args = TensorAccessorArgs<dq_pre_args.next_compile_time_args_offset()>();
    constexpr auto dk_pe_args = TensorAccessorArgs<dkv_up_args.next_compile_time_args_offset()>();

    const auto dq_pre_addr_gen = TensorAccessor(dq_pre_args, dq_pre_addr);
    const auto dkv_up_addr_gen = TensorAccessor(dkv_up_args, dkv_up_addr);
    const auto dk_pe_addr_gen = TensorAccessor(dk_pe_args, dk_pe_addr);

    for (uint32_t block = 0U; block < num_blocks; ++block) {
        for (uint32_t h = 0U; h < n_heads; ++h) {
            const uint32_t dq_head_base = dq_pre_block_base + h * Th;
            const uint32_t dkv_head_base = dkv_up_block_base + h * kv_per_head;

            // dq_pre[block_base + h*Th + w] ← cb_dq tile
            for (uint32_t w = 0U; w < Th; ++w) {
                cb_wait_front(cb_dq, onetile);
                noc_async_write_page(dq_head_base + w, dq_pre_addr_gen, get_read_ptr(cb_dq));
                noc_async_write_barrier();
                cb_pop_front(cb_dq, onetile);
            }

            // dkv_up[block_base + h*kv_per_head + 0..Tn) ← cb_dknope tiles
            for (uint32_t w = 0U; w < Tn; ++w) {
                cb_wait_front(cb_dknope, onetile);
                noc_async_write_page(dkv_head_base + w, dkv_up_addr_gen, get_read_ptr(cb_dknope));
                noc_async_write_barrier();
                cb_pop_front(cb_dknope, onetile);
            }

            // dkv_up[block_base + h*kv_per_head + Tn..Tn+Tv) ← cb_dv tiles
            for (uint32_t w = 0U; w < Tv; ++w) {
                cb_wait_front(cb_dv, onetile);
                noc_async_write_page(dkv_head_base + Tn + w, dkv_up_addr_gen, get_read_ptr(cb_dv));
                noc_async_write_barrier();
                cb_pop_front(cb_dv, onetile);
            }
        }

        // dk_pe[block_base + 0..Tr) ← Tr tiles produced by compute (head-axis sum).
        cb_wait_front(cb_dkpe_out, Tr);
        const uint32_t l1_kpe = get_read_ptr(cb_dkpe_out);
        const uint32_t dkpe_tile_bytes = get_tile_size(cb_dkpe_out);
        for (uint32_t w = 0U; w < Tr; ++w) {
            noc_async_write_page(dk_pe_block_base + w, dk_pe_addr_gen, l1_kpe + w * dkpe_tile_bytes);
        }
        noc_async_write_barrier();
        cb_pop_front(cb_dkpe_out, Tr);

        // Outputs are flat across blocks, so advance is constant.
        dq_pre_block_base += n_heads * Th;
        dkv_up_block_base += n_heads * kv_per_head;
        dk_pe_block_base += Tr;
    }
}
