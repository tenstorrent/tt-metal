// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Router tail, fast sort-core form (RTF_INCR), reader of core B: the 16 logits tiles and the 16 pre-transposed index
// tiles (as today's reader), then, once the compute kernel packed the sorted transposed index tile (cb_idx_t), the tile
// goes by NoC write into core A's cb_idx_t (the same L1 address on every core of the program) and a semaphore increment
// on core A tells A's reader (the unicast + semaphore pattern of gdn_rows_scan/kernels/writer.cpp). Runtime args:
// logits_addr, index_addr, tile_row, a_noc_x, a_noc_y.  Semaphore id: named compile-time arg sem_id.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
#include "../../kernels/zones.h"

#ifndef RTF_SPLIT
#define RTF_SPLIT 0  // the index tiles are read by the other data-movement core (reader_rtf_index.cpp)
#endif
#ifndef RTF_INCR
#define RTF_INCR 0
#endif

void kernel_main() {
    const uint32_t logits_addr = get_arg_val<uint32_t>(0);
    const uint32_t index_addr = get_arg_val<uint32_t>(1);
    const uint32_t tile_row = get_arg_val<uint32_t>(2);
    const uint32_t a_x = get_arg_val<uint32_t>(3);
    const uint32_t a_y = get_arg_val<uint32_t>(4);

    constexpr uint32_t cb_in0 = get_named_compile_time_arg_val("cb_in0");
    constexpr uint32_t cb_index = get_named_compile_time_arg_val("cb_index");
    constexpr uint32_t cb_idx_t = get_named_compile_time_arg_val("cb_idx_t");
    constexpr uint32_t Wt = get_named_compile_time_arg_val("Wt");
    constexpr uint32_t sem_id = get_named_compile_time_arg_val("sem_id");

    constexpr auto logits_args = TensorAccessorArgs<0, 0>();
    constexpr auto index_args = TensorAccessorArgs<
        logits_args.next_compile_time_args_offset(),
        logits_args.next_common_runtime_args_offset()>();
    const auto logits = TensorAccessor(logits_args, logits_addr);
    const auto index_template = TensorAccessor(index_args, index_addr);

    Noc noc;
    DataflowBuffer in0(cb_in0);
    DataflowBuffer index(cb_index);
    DataflowBuffer idx_t(cb_idx_t);

    {
        FUSED_ZONE("fz_rt_r_reads");
        const uint32_t tile_bytes = in0.get_entry_size();
        const uint32_t index_tile_bytes = index.get_entry_size();
        in0.reserve_back(Wt);
#if !RTF_SPLIT
        index.reserve_back(Wt);
#endif
#if RTF_INCR
        // the first group of tiles goes to compute as soon as it lands; the rest is in flight meanwhile
        constexpr uint32_t first = 4;
#else
        constexpr uint32_t first = Wt;
#endif
        for (uint32_t w = 0; w < first; ++w) {
            noc.async_read(logits, in0, tile_bytes, {.page_id = tile_row * Wt + w}, {.offset_bytes = w * tile_bytes});
#if !RTF_SPLIT
            noc.async_read(
                index_template, index, index_tile_bytes, {.page_id = w}, {.offset_bytes = w * index_tile_bytes});
#endif
        }
        noc.async_read_barrier();
        in0.push_back(first);
#if !RTF_SPLIT
        index.push_back(first);
#endif
        if (first < Wt) {  // the write pointer moved past the first group: offsets restart at 0
            for (uint32_t w = first; w < Wt; ++w) {
                noc.async_read(
                    logits,
                    in0,
                    tile_bytes,
                    {.page_id = tile_row * Wt + w},
                    {.offset_bytes = (w - first) * tile_bytes});
#if !RTF_SPLIT
                noc.async_read(
                    index_template,
                    index,
                    index_tile_bytes,
                    {.page_id = w},
                    {.offset_bytes = (w - first) * index_tile_bytes});
#endif
            }
            noc.async_read_barrier();
            in0.push_back(Wt - first);
#if !RTF_SPLIT
            index.push_back(Wt - first);
#endif
        }
    }

    {
        FUSED_ZONE("fz_rt_r_send");
        idx_t.wait_front(1);
        const uint32_t src = idx_t.get_read_ptr();
        noc_async_write(src, get_noc_addr(a_x, a_y, src), 640);  // rows 0..9 of face 0: all the 10 indices of token 0
        noc_async_write_barrier();
        noc_semaphore_inc(get_noc_addr(a_x, a_y, get_semaphore(sem_id)), 1);
        noc_async_atomic_barrier();
        idx_t.pop_front(1);
    }
}
