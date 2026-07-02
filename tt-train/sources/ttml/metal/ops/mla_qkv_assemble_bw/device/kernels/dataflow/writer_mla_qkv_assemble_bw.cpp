// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "tt-train/sources/ttml/metal/common/dataflow_utils.hpp"

void kernel_main() {
    uint32_t runtime_args_counter = 0U;
    const uint32_t dq_pre_addr = get_arg_val<uint32_t>(runtime_args_counter++);
    const uint32_t dkv_up_addr = get_arg_val<uint32_t>(runtime_args_counter++);
    const uint32_t dk_pe_addr = get_arg_val<uint32_t>(runtime_args_counter++);
    const uint32_t num_blocks = get_arg_val<uint32_t>(runtime_args_counter++);
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
    constexpr uint32_t block_size = get_compile_time_arg_val(5);
    constexpr uint32_t kv_per_head = Tn + Tv;

    constexpr auto dq_pre_args = TensorAccessorArgs<6>();
    constexpr auto dkv_up_args = TensorAccessorArgs<dq_pre_args.next_compile_time_args_offset()>();
    constexpr auto dk_pe_args = TensorAccessorArgs<dkv_up_args.next_compile_time_args_offset()>();

    const auto dq_pre_addr_gen = TensorAccessor(dq_pre_args, dq_pre_addr);
    const auto dkv_up_addr_gen = TensorAccessor(dkv_up_args, dkv_up_addr);
    const auto dk_pe_addr_gen = TensorAccessor(dk_pe_args, dk_pe_addr);

    const uint32_t tile_bytes = get_tile_size(cb_dkpe_out);

    for (uint32_t block = 0U; block < num_blocks; ++block) {
        for (uint32_t h = 0U; h < n_heads; ++h) {
            const uint32_t dq_head_base = dq_pre_block_base + h * Th;
            const uint32_t dkv_head_base = dkv_up_block_base + h * kv_per_head;

            // dq_pre ← cb_dq; dkv_up = [dK_nope | dV] ← cb_dknope then cb_dv. All in block_size chunks.
            write_full_row_tiles(cb_dq, dq_pre_addr_gen, Th, block_size, tile_bytes, dq_head_base);
            write_full_row_tiles(cb_dknope, dkv_up_addr_gen, Tn, block_size, tile_bytes, dkv_head_base);
            write_full_row_tiles(cb_dv, dkv_up_addr_gen, Tv, block_size, tile_bytes, dkv_head_base + Tn);
        }

        // dk_pe ← Tr tiles produced by the compute kernel (head-axis sum); one burst + one barrier.
        cb_wait_front(cb_dkpe_out, Tr);
        const uint32_t l1_kpe = get_read_ptr(cb_dkpe_out);
        for (uint32_t w = 0U; w < Tr; ++w) {
            noc_async_write_page(dk_pe_block_base + w, dk_pe_addr_gen, l1_kpe + w * tile_bytes);
        }
        noc_async_write_barrier();
        cb_pop_front(cb_dkpe_out, Tr);

        // Outputs are flat across blocks, so the advance is constant.
        dq_pre_block_base += n_heads * Th;
        dkv_up_block_base += n_heads * kv_per_head;
        dk_pe_block_base += Tr;
    }
}
