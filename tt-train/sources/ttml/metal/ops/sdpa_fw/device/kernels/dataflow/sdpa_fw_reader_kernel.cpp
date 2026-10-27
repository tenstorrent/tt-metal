// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <hostdevcommon/kernel_structs.h>

#include <cstdint>
#include <cstring>

#include "api/dataflow/dataflow_api.h"
#include "tt-train/sources/ttml/metal/common/dataflow_utils.hpp"
#include "tt-train/sources/ttml/metal/ops/sdpa_fw/device/kernels/sdpa_fw_work_list.hpp"

void kernel_main() {
    uint32_t runtime_args_counter = 0;
    uint32_t query_address = get_arg_val<uint32_t>(runtime_args_counter++);  // query buffer address
    uint32_t key_address = get_arg_val<uint32_t>(runtime_args_counter++);    // key buffer address
    uint32_t value_address = get_arg_val<uint32_t>(runtime_args_counter++);  // value buffer address
    uint32_t mask_address = get_arg_val<uint32_t>(runtime_args_counter++);   // mask buffer address
    uint32_t gate_address = get_arg_val<uint32_t>(runtime_args_counter++);   // gate buffer address
    const uint32_t work_list_arg_idx = runtime_args_counter;                 // then the work list

    constexpr uint32_t cb_query = tt::CBIndex::c_0;
    constexpr uint32_t cb_key = tt::CBIndex::c_1;
    constexpr uint32_t cb_value = tt::CBIndex::c_2;
#ifdef USE_ATTN_MASK
    constexpr uint32_t cb_attn_mask = tt::CBIndex::c_3;
#endif
#ifdef HAS_GATE
    constexpr uint32_t cb_gate = tt::CBIndex::c_16;
#endif

    constexpr uint32_t qWt = get_compile_time_arg_val(0);      // num tile in inner dim in query/key (d_qk/TILE_W)
    constexpr uint32_t vWt = get_compile_time_arg_val(1);      // num tile in inner dim in value (d_v/TILE_W)
    constexpr uint32_t Ht = get_compile_time_arg_val(2);       // (S / TILE_H)
    constexpr uint32_t q_heads = get_compile_time_arg_val(3);  // num of heads in query
    constexpr uint32_t heads_per_group = get_compile_time_arg_val(4);  // num of heads per group
    constexpr uint32_t Sk_chunk_t = get_compile_time_arg_val(5);       // K/V tiles per inner-loop step
    constexpr uint32_t Sq_chunk_t = get_compile_time_arg_val(6);       // query tile-rows per Q chunk
    constexpr uint32_t heads_per_item = get_compile_time_arg_val(7);   // Q heads stacked in one item
    constexpr uint32_t q_num_chunks = Ht / Sq_chunk_t;
    constexpr auto query_args = TensorAccessorArgs<8>();
    constexpr auto key_args = TensorAccessorArgs<query_args.next_compile_time_args_offset()>();
    constexpr auto value_args = TensorAccessorArgs<key_args.next_compile_time_args_offset()>();
    [[maybe_unused]] constexpr auto mask_args = TensorAccessorArgs<value_args.next_compile_time_args_offset()>();
    [[maybe_unused]] constexpr auto gate_args = TensorAccessorArgs<mask_args.next_compile_time_args_offset()>();

    const uint32_t tile_bytes = get_tile_size(cb_query);

    const auto query_address_generator = TensorAccessor(query_args, query_address);
    const auto key_address_generator = TensorAccessor(key_args, key_address);
    const auto value_address_generator = TensorAccessor(value_args, value_address);

#ifdef USE_ATTN_MASK
    const auto mask_address_generator = TensorAccessor(mask_args, mask_address);
#endif
#ifdef HAS_GATE
    const auto gate_address_generator = TensorAccessor(gate_args, gate_address);
#endif

    // The constant tiles (reduction scaler, matmul reduce, causal mask) come from the writer, so the
    // reader can start its DRAM reads at once.

    const uint32_t num_of_groups = q_heads / heads_per_group;

    // Sq_chunk_t tile-rows of each of the item's heads from a (B, H, S, Wt) tensor, stacked head after
    // head in the CB, pushed as one block.
    auto read_stacked_rows = [&](const uint32_t cb, const auto& addr_gen, const uint32_t Wt, const WorkItemCoords& w) {
        const uint32_t tiles_per_head = Sq_chunk_t * Wt;
        cb_reserve_back(cb, heads_per_item * tiles_per_head);
        uint32_t l1_addr = get_write_ptr(cb);
        for (uint32_t h = 0; h < heads_per_item; ++h) {
            const uint32_t start_idx = ((w.batch * q_heads + w.head0 + h) * Ht + w.first_row_tile) * Wt;
            for (uint32_t t = 0; t < tiles_per_head; ++t) {
                noc_async_read_page(start_idx + t, addr_gen, l1_addr);
                l1_addr += tile_bytes;
            }
        }
        noc_async_read_barrier();
        cb_push_back(cb, heads_per_item * tiles_per_head);
    };

    // Reads one item's Q block and then K/V steps [k_begin, k_end) of Sk_chunk_t tiles each. The push
    // order must match the cb_wait_front order in the compute kernel.
    auto read_q_chunk = [&](const uint32_t item, const uint32_t k_begin, const uint32_t k_end) {
        const auto w = decode_work_item<q_num_chunks, Sq_chunk_t, q_heads, heads_per_item>(item);

        read_stacked_rows(cb_query, query_address_generator, qWt, w);

        const uint32_t kv_group_idx = w.head0 / heads_per_group;
        const uint32_t key_offset = (w.batch * num_of_groups + kv_group_idx) * qWt * Ht;
        const uint32_t value_offset = (w.batch * num_of_groups + kv_group_idx) * vWt * Ht;

        for (uint32_t k_step = k_begin; k_step < k_end; ++k_step) {
            const uint32_t h_start = k_step * Sk_chunk_t;
            const uint32_t key_start_idx = key_offset + h_start * qWt;
            // K lands feature-major in cb_key (all Sk_chunk_t tiles of feature 0, then feature 1, ...),
            // the order matmul_block walks when it multiplies by K^T.
            read_tile_block_transposed(cb_key, key_address_generator, key_start_idx, Sk_chunk_t, qWt, tile_bytes);

#ifdef USE_ATTN_MASK
            // Arbitrary mask: Sq_chunk_t x Sk_chunk_t tiles of this step, row-major, shared by the stacked
            // heads; the mask is (1, 1, S, S), so a row's Sk_chunk_t tiles are contiguous.
            cb_reserve_back(cb_attn_mask, Sq_chunk_t * Sk_chunk_t);
            uint32_t mask_l1_addr = get_write_ptr(cb_attn_mask);
            for (uint32_t r = 0; r < Sq_chunk_t; ++r) {
                const uint32_t mask_start_idx = (w.first_row_tile + r) * Ht + h_start;
                for (uint32_t n = 0; n < Sk_chunk_t; ++n) {
                    noc_async_read_page(mask_start_idx + n, mask_address_generator, mask_l1_addr);
                    mask_l1_addr += tile_bytes;
                }
            }
            noc_async_read_barrier();
            cb_push_back(cb_attn_mask, Sq_chunk_t * Sk_chunk_t);
#endif
            // Note: For CAUSAL_MASK, the two reusable mask tiles are generated once by the writer.
            const uint32_t value_start_idx = value_offset + h_start * vWt;
            read_tiles_by_row(
                cb_value, value_address_generator, value_start_idx, Sk_chunk_t * vWt, tile_bytes, Sk_chunk_t * vWt);
        }
#ifdef HAS_GATE
        read_stacked_rows(cb_gate, gate_address_generator, vWt, w);
#endif
    };

    for_each_work_item(work_list_arg_idx, [&](uint32_t item) {
        const uint32_t first_row_tile = (item % q_num_chunks) * Sq_chunk_t;
        read_q_chunk(item, /* k_begin */ 0, num_kv_steps_of_chunk<Ht, Sq_chunk_t, Sk_chunk_t>(first_row_tile));
    });
}
