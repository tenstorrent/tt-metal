// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc.h"
#include "api/tensor/tensor_accessor.h"
#include "log_softmax_tile.hpp"

// Streams logits (and upstream gradients) for each unit: tile columns [begin, end) of one tile row.
// Forward: pass 1 (statistics) with tile padding set to -inf, then pass 2 (output).
// Backward: pass 1 reads only the upstream gradient, with the selected element of every row moved
// out into a column vector (cpivotdy) and replaced by zero; pass 2 reads logits and gradient.
// With chunks > 1, between the passes every core of a row waits until all of them have published their
// partial statistics, then gathers them as lanes of one tile per statistic.
// Constant regions are never written element by element: zeros come from noc.async_write_zeros, other constants are
// NoC-copied from the constant tiles the reader keeps behind the column-index tile in buffer 2 (see below).
void kernel_main() {
    constexpr uint32_t wt = get_compile_time_arg_val(0);
    constexpr uint32_t width = get_compile_time_arg_val(2), input_fp32 = get_compile_time_arg_val(3);
    constexpr uint32_t grad_fp32 = get_compile_time_arg_val(4), chunks = get_compile_time_arg_val(5);
    constexpr uint32_t arrive_sem = get_compile_time_arg_val(6), go_sem = get_compile_time_arg_val(7);
    const uint32_t x = get_arg_val<uint32_t>(0), dy = get_arg_val<uint32_t>(1), state = get_arg_val<uint32_t>(2);
    const uint32_t scratch = get_arg_val<uint32_t>(3), count = get_arg_val<uint32_t>(4);
    const uint32_t start = get_arg_val<uint32_t>(5), begin = get_arg_val<uint32_t>(6), end = get_arg_val<uint32_t>(7);
    // Each push is `block` tiles and block divides end - begin, so pushes never straddle the ring's end.
    const uint32_t chunk = get_arg_val<uint32_t>(8), block = get_arg_val<uint32_t>(9);
    constexpr auto xa = TensorAccessorArgs</*cta_offset*/ 8>();
    const auto input = TensorAccessor(xa, x);
#ifdef BACKWARD
    constexpr auto da = TensorAccessorArgs<xa.next_compile_time_args_offset()>();
    constexpr auto sa = TensorAccessorArgs<da.next_compile_time_args_offset()>();
    constexpr auto pa = TensorAccessorArgs<sa.next_compile_time_args_offset()>();
    const auto grad = TensorAccessor(da, dy);
    const auto stats = TensorAccessor(sa, state);
    constexpr uint32_t partials = 2;
#else
    constexpr auto pa = TensorAccessorArgs<xa.next_compile_time_args_offset()>();
    constexpr uint32_t partials = 3;
#endif
    constexpr uint32_t th = log_softmax_tile::kTileHeight, tw = log_softmax_tile::kTileWidth;
    const Noc noc;
    DataflowBuffer xb(0), db(1), constants(2), sb(9), pivot_dy(20), staging(24);
    const uint32_t bytes = get_tile_size(0), grad_bytes = get_tile_size(1), fp32_bytes = get_tile_size(9);
    // Buffer 2 holds five FP32-sized pages. Page 0, the column index of each lane, is pushed to the compute kernel;
    // pages 1-4 stay reserved by this kernel and hold the constant tiles it copies from:
    //   1: -inf (FP32), 2: +inf (FP32), 3: -inf in the input's format, 4: zero in the gradient's format.
    constants.reserve_back(5);
    const uint32_t columns = constants.get_write_ptr();
    [[maybe_unused]] const uint32_t neg_inf_fp32 = columns + fp32_bytes, pos_inf_fp32 = columns + 2 * fp32_bytes;
    [[maybe_unused]] const uint32_t neg_inf_input = columns + 3 * fp32_bytes, zero_grad = columns + 4 * fp32_bytes;
    auto ptr = reinterpret_cast<volatile tt_l1_ptr float *>(columns);
    for (uint32_t r = 0; r < th; ++r)
        for (uint32_t c = 0; c < tw; ++c) ptr[log_softmax_tile::offset(r, c)] = static_cast<float>(c);
#ifndef BACKWARD
    if constexpr (chunks > 1) {
        log_softmax_tile::fill_constant_tile<1>(neg_inf_fp32, fp32_bytes, 0xFF800000U);
        log_softmax_tile::fill_constant_tile<1>(pos_inf_fp32, fp32_bytes, 0x7F800000U);
    }
    if constexpr (width % tw != 0) {
        log_softmax_tile::fill_constant_tile<input_fp32>(neg_inf_input, bytes, 0xFF800000U);
    }
#else
    if constexpr (width % tw != 0) {
        noc.async_write_zeros(constants, grad_bytes, {.offset_bytes = 4 * fp32_bytes});
        noc.write_zeros_l1_barrier();
    }
#endif
    constants.push_back(1);

    // Reads this unit's tiles [begin, end) of `row` into `dfb` in blocks; `patch(col, l1_address)` may edit each
    // tile before the block is pushed.
    const auto read_unit_tiles = [&](const auto &accessor, auto &dfb, uint32_t tile_bytes, uint32_t row, auto patch) {
        for (uint32_t col = begin; col < end; col += block) {
            const uint32_t n = end - col < block ? end - col : block;
            dfb.reserve_back(n);
            for (uint32_t i = 0; i < n; ++i)
                noc.async_read(
                    accessor, dfb, tile_bytes, {.page_id = row * wt + col + i}, {.offset_bytes = i * tile_bytes});
            noc.async_read_barrier();
            for (uint32_t i = 0; i < n; ++i) patch(col + i, dfb.get_write_ptr() + i * tile_bytes);
            dfb.push_back(n);
        }
    };
    const auto no_patch = [](uint32_t, uint32_t) {};

    for (uint32_t row = start; row < start + count; ++row) {
#ifndef BACKWARD
        // Tile padding never wins the maximum and contributes exp(-inf) = 0: copied from the -inf constant tile.
        read_unit_tiles(input, xb, bytes, row, [&](uint32_t col, uint32_t address) {
            if constexpr (width % tw != 0) {
                if (col == wt - 1) {
                    log_softmax_tile::copy_columns<input_fp32 ? 4 : 2>(neg_inf_input, address, width % tw);
                    noc_async_read_barrier();
                }
            }
        });
#else
        // Row state: maximum, selected index and log1p(tail) column vectors.
        sb.reserve_back(3);
        for (uint32_t i = 0; i < 3; ++i)
            noc.async_read(stats, sb, fp32_bytes, {.page_id = row * 3 + i}, {.offset_bytes = i * fp32_bytes});
        noc.async_read_barrier();
        uint32_t pivot[th];
        log_softmax_tile::read_pivots(sb.get_write_ptr() + fp32_bytes, pivot);
        sb.push_back(3);
        pivot_dy.reserve_back(1);
        const uint32_t pivot_address = pivot_dy.get_write_ptr();
        noc.async_write_zeros(pivot_dy, fp32_bytes);
        noc.write_zeros_l1_barrier();
        read_unit_tiles(grad, db, grad_bytes, row, [&](uint32_t col, uint32_t address) {
            for (uint32_t r = 0; r < th; ++r) {
                if (pivot[r] / tw == col) {
                    const uint32_t at = log_softmax_tile::offset(r, pivot[r] % tw);
                    log_softmax_tile::store_fp32(
                        pivot_address, log_softmax_tile::offset(r, 0), log_softmax_tile::load<grad_fp32>(address, at));
                    log_softmax_tile::store<grad_fp32>(address, at, 0U);  // One element: a store, not a copy.
                }
            }
            if constexpr (width % tw != 0) {
                if (col == wt - 1) {
                    log_softmax_tile::copy_columns<grad_fp32 ? 4 : 2>(zero_grad, address, width % tw);
                    noc_async_read_barrier();
                }
            }
        });
        pivot_dy.push_back(1);
#endif
        if constexpr (chunks > 1) {
            const auto parts = TensorAccessor(pa, scratch);
            if (chunk == 0) {
                auto arrive = reinterpret_cast<volatile tt_l1_ptr uint32_t *>(get_semaphore(arrive_sem));
                noc_semaphore_wait(arrive, chunks);
                noc_semaphore_set(arrive, 0);
                for (uint32_t g = 0; g < chunks; ++g) {
                    const uint32_t core_x = get_arg_val<uint32_t>(10 + 2 * g),
                                   core_y = get_arg_val<uint32_t>(11 + 2 * g);
                    noc_semaphore_inc(get_noc_addr(core_x, core_y, get_semaphore(go_sem)), 1);
                }
            }
            auto go = reinterpret_cast<volatile tt_l1_ptr uint32_t *>(get_semaphore(go_sem));
            noc_semaphore_wait(go, 1);
            noc_semaphore_set(go, 0);
            // Fetch every chunk's packed record with one barrier; statistic i of chunk g becomes lane g of tile i.
            staging.reserve_back((chunks * log_softmax_tile::kPartialBytes + fp32_bytes - 1) / fp32_bytes);
            const uint32_t records = staging.get_write_ptr();
            for (uint32_t g = 0; g < chunks; ++g)
                noc.async_read(
                    parts,
                    staging,
                    log_softmax_tile::kPartialBytes,
                    {.page_id = row * chunks + g},
                    {.offset_bytes = g * log_softmax_tile::kPartialBytes});
            noc.async_read_barrier();
            for (uint32_t i = 0; i < partials; ++i) {
                DataflowBuffer lanes(17 + i);
                lanes.reserve_back(1);
                const uint32_t lane_address = lanes.get_write_ptr();
                // Unused lanes: m = -inf, i = +inf, t = 0 (backward: zero sums).
#ifndef BACKWARD
                if (i < 2) {
                    noc_async_read(get_noc_addr(i == 0 ? neg_inf_fp32 : pos_inf_fp32), lane_address, fp32_bytes);
                    noc_async_read_barrier();
                } else {
                    noc.async_write_zeros(lanes, fp32_bytes);
                    noc.write_zeros_l1_barrier();
                }
#else
                noc.async_write_zeros(lanes, fp32_bytes);
                noc.write_zeros_l1_barrier();
#endif
                for (uint32_t g = 0; g < chunks; ++g)
                    for (uint32_t r = 0; r < th; ++r)
                        log_softmax_tile::copy_fp32(
                            records,
                            g * log_softmax_tile::kPartialWords + i * th + r,
                            lane_address,
                            log_softmax_tile::offset(r, g));
                lanes.push_back(1);
            }
        }
#ifndef BACKWARD
        read_unit_tiles(input, xb, bytes, row, no_patch);
#else
        for (uint32_t col = begin; col < end; col += block) {
            const uint32_t n = end - col < block ? end - col : block;
            xb.reserve_back(n);
            db.reserve_back(n);
            for (uint32_t i = 0; i < n; ++i) {
                noc.async_read(input, xb, bytes, {.page_id = row * wt + col + i}, {.offset_bytes = i * bytes});
                noc.async_read(grad, db, grad_bytes, {.page_id = row * wt + col + i}, {.offset_bytes = i * grad_bytes});
            }
            noc.async_read_barrier();
            xb.push_back(n);
            db.push_back(n);
        }
#endif
    }
}
