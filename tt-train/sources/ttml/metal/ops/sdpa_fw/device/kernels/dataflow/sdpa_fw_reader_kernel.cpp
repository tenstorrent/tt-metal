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
    uint32_t query_address = get_arg_val<uint32_t>(runtime_args_counter++);    // query buffer address
    uint32_t key_address = get_arg_val<uint32_t>(runtime_args_counter++);      // key buffer address
    uint32_t value_address = get_arg_val<uint32_t>(runtime_args_counter++);    // value buffer address
    uint32_t mask_address = get_arg_val<uint32_t>(runtime_args_counter++);     // mask buffer address
    uint32_t gate_address = get_arg_val<uint32_t>(runtime_args_counter++);     // gate buffer address
    uint32_t scratch_address = get_arg_val<uint32_t>(runtime_args_counter++);  // partials scratch (split units)
    const uint32_t work_list_arg_idx = runtime_args_counter;                   // then the work list

    constexpr uint32_t cb_query = tt::CBIndex::c_0;
    constexpr uint32_t cb_key = tt::CBIndex::c_1;
    constexpr uint32_t cb_value = tt::CBIndex::c_2;
#ifdef USE_ATTN_MASK
    constexpr uint32_t cb_attn_mask = tt::CBIndex::c_3;
#endif
#ifdef HAS_GATE
    constexpr uint32_t cb_gate = tt::CBIndex::c_16;
#endif
#ifdef SPLIT_UNITS
    constexpr uint32_t cb_partial_in = tt::CBIndex::c_17;
#endif

    constexpr uint32_t qWt = get_compile_time_arg_val(0);      // num tile in inner dim in query/key (d_qk/TILE_W)
    constexpr uint32_t vWt = get_compile_time_arg_val(1);      // num tile in inner dim in value (d_v/TILE_W)
    constexpr uint32_t Ht = get_compile_time_arg_val(2);       // (S / TILE_H)
    constexpr uint32_t q_heads = get_compile_time_arg_val(3);  // num of heads in query
    constexpr uint32_t heads_per_group = get_compile_time_arg_val(4);  // num of heads per group
    constexpr uint32_t Sk_chunk_t = get_compile_time_arg_val(5);       // K/V tiles per inner-loop step
    constexpr uint32_t Sq_chunk_t = get_compile_time_arg_val(6);       // query tile-rows per Q chunk
    constexpr uint32_t heads_per_item = get_compile_time_arg_val(7);   // Q heads stacked in one item
    // 8..11: the four semaphore ids a merger waits on (sem_idx picks one)
    constexpr uint32_t q_num_chunks = Ht / Sq_chunk_t;
    constexpr uint32_t rows = Sq_chunk_t * heads_per_item;
    constexpr uint32_t k_tiles = Sk_chunk_t * qWt;                             // K tiles of one step
    constexpr uint32_t v_tiles = Sk_chunk_t * vWt;                             // V tiles of one step
    [[maybe_unused]] constexpr uint32_t mask_tiles = Sq_chunk_t * Sk_chunk_t;  // arbitrary mask tiles of one step
    constexpr auto query_args = TensorAccessorArgs<12>();
    constexpr auto key_args = TensorAccessorArgs<query_args.next_compile_time_args_offset()>();
    constexpr auto value_args = TensorAccessorArgs<key_args.next_compile_time_args_offset()>();
    [[maybe_unused]] constexpr auto mask_args = TensorAccessorArgs<value_args.next_compile_time_args_offset()>();
    [[maybe_unused]] constexpr auto gate_args = TensorAccessorArgs<mask_args.next_compile_time_args_offset()>();
    [[maybe_unused]] constexpr auto scratch_args = TensorAccessorArgs<gate_args.next_compile_time_args_offset()>();

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
#ifdef SPLIT_UNITS
    constexpr uint32_t partial_row_tiles = sdpa_fw_items::partial_row_tiles(vWt);
    constexpr uint32_t kPartialRowsPerBatch = sdpa_fw_items::partial_rows_per_batch(rows);
    const uint32_t partial_tile_bytes = get_tile_size(cb_partial_in);
    const auto scratch_address_generator = TensorAccessor(scratch_args, scratch_address, partial_tile_bytes);
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

        // Each step pushes K as soon as it lands, so the compute kernel starts QK^T while the step's mask
        // tiles and V are still in flight; those are issued together and waited for once.
        for (uint32_t k_step = k_begin; k_step < k_end; ++k_step) {
            const uint32_t h_start = k_step * Sk_chunk_t;
            const uint32_t key_start_idx = key_offset + h_start * qWt;
            // K lands feature-major in cb_key (all Sk_chunk_t tiles of feature 0, then feature 1, ...),
            // the order matmul_block walks when it multiplies by K^T.
            read_tile_block_transposed</* UseBarrier */ true>(
                cb_key, key_address_generator, key_start_idx, Sk_chunk_t, qWt, tile_bytes);

#ifdef USE_ATTN_MASK
            // Arbitrary mask: Sq_chunk_t x Sk_chunk_t tiles of this step, row-major, shared by the stacked
            // heads; the mask is (1, 1, S, S), so a row's Sk_chunk_t tiles are contiguous.
            cb_reserve_back(cb_attn_mask, mask_tiles);
            uint32_t mask_l1_addr = get_write_ptr(cb_attn_mask);
            for (uint32_t r = 0; r < Sq_chunk_t; ++r) {
                const uint32_t mask_start_idx = (w.first_row_tile + r) * Ht + h_start;
                for (uint32_t n = 0; n < Sk_chunk_t; ++n) {
                    noc_async_read_page(mask_start_idx + n, mask_address_generator, mask_l1_addr);
                    mask_l1_addr += tile_bytes;
                }
            }
#endif
            // Note: For CAUSAL_MASK, the two reusable mask tiles are generated once by the writer.
            const uint32_t value_start_idx = value_offset + h_start * vWt;
            read_tiles_by_row</* UseBarrier */ false>(
                cb_value, value_address_generator, value_start_idx, v_tiles, tile_bytes, v_tiles);

            noc_async_read_barrier();
#ifdef USE_ATTN_MASK
            cb_push_back(cb_attn_mask, mask_tiles);
#endif
            cb_push_back(cb_value, v_tiles);
        }
    };

    auto read_gate = [&]([[maybe_unused]] const uint32_t item) {
#ifdef HAS_GATE
        const auto w = decode_work_item<q_num_chunks, Sq_chunk_t, q_heads, heads_per_item>(item);
        read_stacked_rows(cb_gate, gate_address_generator, vWt, w);
#endif
    };

    for_each_work_unit(work_list_arg_idx, [&](const WorkUnit& unit) {
        if (unit.role == kUnitWhole) {
            for (uint32_t item = unit.item; item < unit.item + unit.num_items; ++item) {
                const uint32_t first_row_tile = sdpa_fw_items::first_row_tile_of(item, q_num_chunks, Sq_chunk_t);
                read_q_chunk(item, /* k_begin */ 0, num_kv_steps_of_chunk<Ht, Sq_chunk_t, Sk_chunk_t>(first_row_tile));
                read_gate(item);
            }
            return;
        }
        read_q_chunk(unit.item, unit.step_begin, unit.step_end);
#ifdef SPLIT_UNITS
        if (unit.role == kUnitMerger) {
            read_gate(unit.item);  // only needed by finalize, so it goes out before the wait
            // The senders bump this core's semaphore once each after their partial landed in scratch.
            static_assert(sdpa_fw_items::kMaxMergersPerCore == 4U, "the semaphore ids are compile args 8..11");
            const uint32_t sem_ids[sdpa_fw_items::kMaxMergersPerCore] = {
                get_compile_time_arg_val(8),
                get_compile_time_arg_val(9),
                get_compile_time_arg_val(10),
                get_compile_time_arg_val(11)};
            volatile tt_l1_ptr uint32_t* sem =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_ids[unit.sem_idx]));
            noc_semaphore_wait_min(sem, unit.num_partials);
            noc_semaphore_set(sem, 0);
            // Partial rows come in kPartialRowsPerBatch at a time, one barrier each; cb_partial_in holds
            // twice that, so the next batch is read while the merge works on this one.
            for (uint32_t p = 0; p < unit.num_partials; ++p) {
                for (uint32_t r = 0; r < rows; r += kPartialRowsPerBatch) {
                    read_tiles_by_row(
                        cb_partial_in,
                        scratch_address_generator,
                        ((unit.slot + p) * rows + r) * partial_row_tiles,
                        kPartialRowsPerBatch * partial_row_tiles,
                        partial_tile_bytes,
                        kPartialRowsPerBatch * partial_row_tiles);
                }
            }
        }
#endif
    });
}
