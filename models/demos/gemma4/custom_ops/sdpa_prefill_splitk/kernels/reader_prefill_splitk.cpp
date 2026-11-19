// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// [sdpa_prefill_splitk] Reader: the page table once, then per work piece its Q chunk and the K / V chunks of its
// K range from the paged cache (the stock chunked-prefill reads, dataflow_common.hpp helpers).

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/endpoints.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"
#include "ttnn/operations/transformer/sdpa/device/kernels/dataflow/dataflow_common.hpp"

void kernel_main() {
    Noc noc;

    constexpr uint32_t NQH = get_compile_time_arg_val(0);
    constexpr uint32_t NKH = get_compile_time_arg_val(1);  // KV heads of the cache view (the layer's own geometry)
    constexpr uint32_t Sqt = get_compile_time_arg_val(2);  // Q sequence tiles (padded)
    constexpr uint32_t DHt = get_compile_time_arg_val(3);
    constexpr uint32_t vDHt = get_compile_time_arg_val(4);
    constexpr uint32_t Sq_chunk_t = get_compile_time_arg_val(5);
    constexpr uint32_t Sk_chunk_t = get_compile_time_arg_val(6);
    constexpr uint32_t block_size_t = get_compile_time_arg_val(7);
    constexpr uint32_t page_table_stick_size = get_compile_time_arg_val(8);
    constexpr uint32_t num_cores = get_compile_time_arg_val(9);
    constexpr uint32_t cb_q_in = get_compile_time_arg_val(10);
    constexpr uint32_t cb_k_in = get_compile_time_arg_val(11);
    constexpr uint32_t cb_v_in = get_compile_time_arg_val(12);
    constexpr uint32_t cb_page_table = get_compile_time_arg_val(13);

    constexpr auto q_args = TensorAccessorArgs<14>();
    constexpr auto k_args = TensorAccessorArgs<q_args.next_compile_time_args_offset()>();
    constexpr auto v_args = TensorAccessorArgs<k_args.next_compile_time_args_offset()>();
    constexpr auto page_table_args = TensorAccessorArgs<v_args.next_compile_time_args_offset()>();

    uint32_t argidx = 0;
    const uint32_t q_addr = get_arg_val<uint32_t>(argidx++);
    const uint32_t k_addr = get_arg_val<uint32_t>(argidx++);
    const uint32_t v_addr = get_arg_val<uint32_t>(argidx++);
    const uint32_t page_table_addr = get_arg_val<uint32_t>(argidx++);
    const uint32_t num_pieces = get_arg_val<uint32_t>(argidx++);

    constexpr uint32_t q_tile_bytes = get_tile_size(cb_q_in);
    constexpr uint32_t k_tile_bytes = get_tile_size(cb_k_in);
    constexpr uint32_t v_tile_bytes = get_tile_size(cb_v_in);
    constexpr uint32_t barrier_threshold = get_barrier_read_threshold<q_tile_bytes, num_cores>();
    constexpr uint32_t heads_per_kv = NQH / NKH;

    const auto q_reader = TensorAccessor(q_args, q_addr);
    const auto k_reader = TensorAccessor(k_args, k_addr);
    const auto v_reader = TensorAccessor(v_args, v_addr);
    const auto q_tile_shape = TensorTileShape(1, NQH, Sqt, DHt);

    CircularBuffer cb_pt(cb_page_table);
    cb_pt.reserve_back(1);
    volatile tt_l1_ptr uint32_t* page_table_ptr =
        read_page_table_for_batch(noc, cb_page_table, 0, page_table_args, page_table_addr, page_table_stick_size);
    cb_pt.push_back(1);

    for (uint32_t piece = 0; piece < num_pieces; ++piece) {
        const uint32_t head = get_arg_val<uint32_t>(argidx++);
        const uint32_t q_chunk = get_arg_val<uint32_t>(argidx++);
        const uint32_t k_lo = get_arg_val<uint32_t>(argidx++);
        const uint32_t k_hi = get_arg_val<uint32_t>(argidx++);

        const uint32_t q_tile_id = q_tile_shape.id_of(0, head, q_chunk * Sq_chunk_t, 0);
        read_chunk_with_padding<q_tile_bytes>(
            q_reader, cb_q_in, q_tile_id, Sq_chunk_t, DHt, Sq_chunk_t, DHt, barrier_threshold);

        const uint32_t kv_head = head / heads_per_kv;
        for (uint32_t k_chunk = k_lo; k_chunk < k_hi; ++k_chunk) {
            const uint32_t k_row = k_chunk * Sk_chunk_t;
            read_paged_chunk_with_padding<NKH, block_size_t, DHt>(
                k_reader,
                cb_k_in,
                kv_head,
                k_row,
                Sk_chunk_t,
                DHt,
                Sk_chunk_t,
                DHt,
                k_tile_bytes,
                barrier_threshold,
                page_table_ptr,
                true /*transpose*/);
            read_paged_chunk_with_padding<NKH, block_size_t, DHt>(
                v_reader,
                cb_v_in,
                kv_head,
                k_row,
                Sk_chunk_t,
                vDHt,
                Sk_chunk_t,
                vDHt,
                v_tile_bytes,
                barrier_threshold,
                page_table_ptr,
                false);
        }
    }
    cb_pt.pop_front(1);
}
