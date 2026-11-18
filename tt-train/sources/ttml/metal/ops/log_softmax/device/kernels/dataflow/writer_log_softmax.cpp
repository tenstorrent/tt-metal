// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc.h"
#include "api/tensor/tensor_accessor.h"
#include "log_softmax_tile.hpp"

// Writes each unit's output tiles [begin, end) and, from chunk 0 of a row, the forward row state.
// With chunks > 1 it first publishes this core's partial statistics to DRAM and signals the row's
// first core. Backward: overwrites each row's selected element with the reconstructed gradient.
// Cross-entropy forward: chunk 0 writes the row's loss tile. Cross-entropy backward: overwrites each
// row's target element with its gradient s * expm1(y_t), where the target falls in this unit's columns.
void kernel_main() {
    constexpr uint32_t wt = get_compile_time_arg_val(0);
    constexpr uint32_t out_fp32 = get_compile_time_arg_val(2), chunks = get_compile_time_arg_val(3);
    constexpr uint32_t arrive_sem = get_compile_time_arg_val(4);
    const uint32_t out = get_arg_val<uint32_t>(0), state = get_arg_val<uint32_t>(1);
    const uint32_t scratch = get_arg_val<uint32_t>(2), count = get_arg_val<uint32_t>(3);
    const uint32_t start = get_arg_val<uint32_t>(4), begin = get_arg_val<uint32_t>(5), end = get_arg_val<uint32_t>(6);
    const uint32_t chunk = get_arg_val<uint32_t>(7), leader_x = get_arg_val<uint32_t>(8);
    const uint32_t leader_y = get_arg_val<uint32_t>(9), block = get_arg_val<uint32_t>(10);
    constexpr auto oa = TensorAccessorArgs</*cta_offset*/ 5>();
    constexpr auto sa = TensorAccessorArgs<oa.next_compile_time_args_offset()>();
    constexpr auto pa = TensorAccessorArgs<sa.next_compile_time_args_offset()>();
    const auto output = TensorAccessor(oa, out);
    [[maybe_unused]] const auto stats = TensorAccessor(sa, state);
    const auto parts = TensorAccessor(pa, scratch);
#if defined(BACKWARD) && !defined(CROSS_ENTROPY)
    constexpr uint32_t partials = 2;
#else
    constexpr uint32_t partials = 3;
#endif
    constexpr uint32_t th = log_softmax_tile::kTileHeight, tw = log_softmax_tile::kTileWidth;
    const Noc noc;
    DataflowBuffer ob(16), sb(8), part(23), pivot_grad(21), pivots(22);
    const uint32_t bytes = get_tile_size(16), fp32_bytes = get_tile_size(8);
    for (uint32_t row = start; row < start + count; ++row) {
#if defined(BACKWARD) && !defined(CROSS_ENTROPY)
        uint32_t pivot[th];
        pivots.reserve_back(1);
        noc.async_read(stats, pivots, fp32_bytes, {.page_id = row * 3 + 1}, {});
        noc.async_read_barrier();
        log_softmax_tile::read_pivots(pivots.get_write_ptr(), pivot);
#endif
        if constexpr (chunks > 1) {
            // Pack column 0 of each partial statistic tile into one record (pivots are already parsed).
            part.wait_front(partials);
            pivots.reserve_back(1);
            const uint32_t record = pivots.get_write_ptr();
            for (uint32_t i = 0; i < partials; ++i)
                for (uint32_t r = 0; r < th; ++r)
                    log_softmax_tile::copy_fp32(
                        part.get_read_ptr() + i * fp32_bytes, log_softmax_tile::offset(r, 0), record, i * th + r);
            noc.async_write(pivots, parts, log_softmax_tile::kPartialBytes, {}, {.page_id = row * chunks + chunk});
            noc.async_write_barrier();
            part.pop_front(partials);
            noc_semaphore_inc(get_noc_addr(leader_x, leader_y, get_semaphore(arrive_sem)), 1);
        }
#if defined(CROSS_ENTROPY)
#ifndef BACKWARD
        ob.wait_front(1);
        if (chunk == 0) {
            noc.async_write(ob, output, bytes, {}, {.page_id = row});
            noc.async_write_barrier();
        }
        ob.pop_front(1);
        continue;
#else
        // pivot[] holds the targets: the substitution below is the same, with the target gradient.
        uint32_t pivot[th];
        DataflowBuffer target_rows(26);
        target_rows.wait_front(1);
        for (uint32_t r = 0; r < th; ++r) pivot[r] = log_softmax_tile::words(target_rows.get_read_ptr())[r];
        target_rows.pop_front(1);
        uint32_t grad_bits[th];
        pivot_grad.wait_front(1);
        for (uint32_t r = 0; r < th; ++r)
            grad_bits[r] = log_softmax_tile::words(pivot_grad.get_read_ptr())[log_softmax_tile::offset(r, 0)];
        pivot_grad.pop_front(1);
#endif
#elif !defined(BACKWARD)
        sb.wait_front(3);
        if (chunk == 0) {
            for (uint32_t i = 0; i < 3; ++i)
                noc.async_write(sb, stats, fp32_bytes, {.offset_bytes = i * fp32_bytes}, {.page_id = row * 3 + i});
            noc.async_write_barrier();
        }
        sb.pop_front(3);
#else
        uint32_t grad_bits[th];
        pivot_grad.wait_front(1);
        for (uint32_t r = 0; r < th; ++r)
            grad_bits[r] = log_softmax_tile::words(pivot_grad.get_read_ptr())[log_softmax_tile::offset(r, 0)];
        pivot_grad.pop_front(1);
#endif
        for (uint32_t col = begin; col < end; col += block) {
            const uint32_t n = end - col < block ? end - col : block;
            ob.wait_front(n);
            for (uint32_t i = 0; i < n; ++i) {
#ifdef BACKWARD
                // Eq. (7): the selected element's gradient, reconstructed from the other entries.
                const uint32_t address = ob.get_read_ptr() + i * bytes;
                for (uint32_t r = 0; r < th; ++r) {
                    if (pivot[r] / tw == col + i) {
                        log_softmax_tile::store<out_fp32>(
                            address, log_softmax_tile::offset(r, pivot[r] % tw), grad_bits[r]);
                    }
                }
#endif
                noc.async_write(ob, output, bytes, {.offset_bytes = i * bytes}, {.page_id = row * wt + col + i});
            }
            noc.async_write_barrier();
            ob.pop_front(n);
        }
    }
}
