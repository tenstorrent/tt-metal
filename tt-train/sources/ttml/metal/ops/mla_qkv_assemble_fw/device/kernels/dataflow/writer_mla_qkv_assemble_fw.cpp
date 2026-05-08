// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "tt-train/sources/ttml/metal/common/dataflow_utils.hpp"

void kernel_main() {
    uint32_t runtime_args_counter = 0U;
    uint32_t q_addr = get_arg_val<uint32_t>(runtime_args_counter++);
    uint32_t k_addr = get_arg_val<uint32_t>(runtime_args_counter++);
    uint32_t v_addr = get_arg_val<uint32_t>(runtime_args_counter++);
    uint32_t num_blocks = get_arg_val<uint32_t>(runtime_args_counter++);
    uint32_t sb = get_arg_val<uint32_t>(runtime_args_counter++);         // s-tile-row index within current batch
    uint32_t q_tile_id = get_arg_val<uint32_t>(runtime_args_counter++);  // q[b, 0, sb, 0]
    uint32_t k_tile_id = get_arg_val<uint32_t>(runtime_args_counter++);  // k[b, 0, sb, 0]
    uint32_t v_tile_id = get_arg_val<uint32_t>(runtime_args_counter++);  // v[b, 0, sb, 0]

    constexpr uint32_t cb_q = tt::CBIndex::c_0;
    constexpr uint32_t cb_kv_up = tt::CBIndex::c_1;
    constexpr uint32_t cb_kpe = tt::CBIndex::c_2;

    constexpr uint32_t Tn = get_compile_time_arg_val(0);  // qk_nope_dim / TILE_W
    constexpr uint32_t Tr = get_compile_time_arg_val(1);  // qk_rope_dim / TILE_W
    constexpr uint32_t Tv = get_compile_time_arg_val(2);  // v_dim       / TILE_W
    constexpr uint32_t n_heads = get_compile_time_arg_val(3);
    constexpr uint32_t kq_HtWt = get_compile_time_arg_val(4);  // S_t * (Tn + Tr)  — q and k share head_dim
    constexpr uint32_t v_HtWt = get_compile_time_arg_val(5);   // S_t * Tv
    constexpr uint32_t S_t = get_compile_time_arg_val(6);      // S / TILE_H

    constexpr auto q_args = TensorAccessorArgs<7>();
    constexpr auto k_args = TensorAccessorArgs<q_args.next_compile_time_args_offset()>();
    constexpr auto v_args = TensorAccessorArgs<k_args.next_compile_time_args_offset()>();

    const auto q_addr_gen = TensorAccessor(q_args, q_addr);
    const auto k_addr_gen = TensorAccessor(k_args, k_addr);
    const auto v_addr_gen = TensorAccessor(v_args, v_addr);

    const uint32_t tile_bytes = get_tile_size(cb_kpe);
    constexpr uint32_t Th = Tn + Tr;

    // End-of-batch jump for head-major outputs:
    //   from (b, sb=S_t-1, h=0, w=0)  to  (b+1, sb=0, h=0, w=0)
    //   diff = H*HtWt - (S_t-1)*Wt = ((H-1)*S_t + 1) * Wt
    constexpr uint32_t end_of_batch_jump_q = ((n_heads - 1U) * S_t + 1U) * Th;
    constexpr uint32_t end_of_batch_jump_v = ((n_heads - 1U) * S_t + 1U) * Tv;

    for (uint32_t block = 0U; block < num_blocks; ++block) {
        // Peek-only on cb_kpe: same Tr tiles reused across all heads.
        cb_wait_front(cb_kpe, Tr);
        const uint32_t kpe_l1_base = get_read_ptr(cb_kpe);

        uint32_t q_head_base = q_tile_id;
        uint32_t k_head_base = k_tile_id;
        uint32_t v_head_base = v_tile_id;

        for (uint32_t h = 0U; h < n_heads; ++h) {
            // Q (head-split, no rotation here): drain Th tiles from cb_q
            for (uint32_t w = 0U; w < Th; ++w) {
                cb_wait_front(cb_q, onetile);
                const uint32_t l1_addr = get_read_ptr(cb_q);
                noc_async_write_page(q_head_base + w, q_addr_gen, l1_addr);
                noc_async_write_barrier();
                cb_pop_front(cb_q, onetile);
            }

            // K nope: drain Tn tiles from cb_kv_up
            for (uint32_t w = 0U; w < Tn; ++w) {
                cb_wait_front(cb_kv_up, onetile);
                const uint32_t l1_addr = get_read_ptr(cb_kv_up);
                noc_async_write_page(k_head_base + w, k_addr_gen, l1_addr);
                noc_async_write_barrier();
                cb_pop_front(cb_kv_up, onetile);
            }

            // K pe: broadcast — same peeked tiles for every head
            for (uint32_t w = 0U; w < Tr; ++w) {
                noc_async_write_page(k_head_base + Tn + w, k_addr_gen, kpe_l1_base + w * tile_bytes);
            }
            noc_async_write_barrier();

            // V: drain Tv tiles from cb_kv_up
            for (uint32_t w = 0U; w < Tv; ++w) {
                cb_wait_front(cb_kv_up, onetile);
                const uint32_t l1_addr = get_read_ptr(cb_kv_up);
                noc_async_write_page(v_head_base + w, v_addr_gen, l1_addr);
                noc_async_write_barrier();
                cb_pop_front(cb_kv_up, onetile);
            }

            q_head_base += kq_HtWt;
            k_head_base += kq_HtWt;
            v_head_base += v_HtWt;
        }

        cb_pop_front(cb_kpe, Tr);

        ++sb;
        if (sb < S_t) {
            q_tile_id += Th;
            k_tile_id += Th;
            v_tile_id += Tv;
        } else {
            // q_head_base / k_head_base / v_head_base after the head loop are
            // q_tile_id_at_block_start + n_heads * head_stride — that's the next
            // batch start ONLY when S_t == 1. For S_t > 1 the per-w writes don't
            // get folded into them, so use the explicit jump instead.
            q_tile_id += end_of_batch_jump_q;
            k_tile_id += end_of_batch_jump_q;  // q and k share Th
            v_tile_id += end_of_batch_jump_v;
            sb = 0U;
        }
    }
}
