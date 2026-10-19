// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Routing setup of the DeepSeek-V4.1 decode step (tt/v41/routing.py route_offsets): the dispatch offsets, total
// counts and expert region offsets that ``masked_bincount`` + ``offset_cumsum`` produce for a prefill chunk, from
// the top-k indices of every chip of the dispatch group at once. ``idx`` is the SP all-gather of the chips' index
// tiles: H uint16 tiles (page h = source chip h of the group), rows [0, REAL_ROWS) of each real, columns [0, K).
// ``mask`` is the chip's row of the expert dispatch table (int32, >= 0 for the experts of this group; the table may
// carry a trailing sentinel column); ``pin`` is a per-chip uint32 page whose element 0 is the chip's position in
// the group. One core, as ``offset_cumsum``'s kernel, and the same arithmetic:
//
//   total[e]  = count of (h, t, j) with idx[h][t][j] == e and mask[e] >= 0
//   local[e]  = the same over h < pin
//   region[e] = exclusive prefix over each run of EPC experts of (total rounded up to 32 rows)
//   offset[e] = region[e] + local[e]
//
// Outputs: three uint32 row-major pages of E words (``offsets``, ``totals``, ``region``).
//
// compile_time_args = [cb_scratch, E, K, REAL_ROWS, H, EPC, MASK_BYTES, TensorAccessorArgs(idx)..., (mask)...,
//                      (pin)..., (offsets)..., (totals)..., (region)...]
// runtime args      = [idx_addr, mask_addr, pin_addr, offsets_addr, totals_addr, region_addr]

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"

void kernel_main() {
    const uint32_t idx_addr = get_arg_val<uint32_t>(0);
    const uint32_t mask_addr = get_arg_val<uint32_t>(1);
    const uint32_t pin_addr = get_arg_val<uint32_t>(2);
    const uint32_t offsets_addr = get_arg_val<uint32_t>(3);
    const uint32_t totals_addr = get_arg_val<uint32_t>(4);
    const uint32_t region_addr = get_arg_val<uint32_t>(5);

    constexpr uint32_t cb_scratch = get_compile_time_arg_val(0);
    constexpr uint32_t E = get_compile_time_arg_val(1);
    constexpr uint32_t K = get_compile_time_arg_val(2);
    constexpr uint32_t REAL_ROWS = get_compile_time_arg_val(3);
    constexpr uint32_t H = get_compile_time_arg_val(4);
    constexpr uint32_t EPC = get_compile_time_arg_val(5);
    constexpr uint32_t MASK_BYTES = get_compile_time_arg_val(6);
    constexpr auto idx_args = TensorAccessorArgs<7>();
    constexpr auto mask_args = TensorAccessorArgs<idx_args.next_compile_time_args_offset()>();
    constexpr auto pin_args = TensorAccessorArgs<mask_args.next_compile_time_args_offset()>();
    constexpr auto offsets_args = TensorAccessorArgs<pin_args.next_compile_time_args_offset()>();
    constexpr auto totals_args = TensorAccessorArgs<offsets_args.next_compile_time_args_offset()>();
    constexpr auto region_args = TensorAccessorArgs<totals_args.next_compile_time_args_offset()>();
    constexpr uint32_t U16_TILE = 2048;
    constexpr uint32_t WORDS = E * 4;
    constexpr uint32_t MASK_L1 = (MASK_BYTES + 63) & ~63u;

    const auto idx = TensorAccessor(idx_args, idx_addr);
    const auto mask = TensorAccessor(mask_args, mask_addr);
    const auto pin = TensorAccessor(pin_args, pin_addr);
    const auto offsets = TensorAccessor(offsets_args, offsets_addr);
    const auto totals = TensorAccessor(totals_args, totals_addr);
    const auto region = TensorAccessor(region_args, region_addr);
    DataflowBuffer scratch(cb_scratch);
    const uint32_t base = (scratch.get_write_ptr() + 63) & ~63u;
    const uint32_t tiles_l1 = base;
    const uint32_t mask_l1 = tiles_l1 + H * U16_TILE;
    const uint32_t pin_l1 = mask_l1 + MASK_L1;
    const uint32_t total_l1 = pin_l1 + 64;
    const uint32_t local_l1 = total_l1 + WORDS;
    const uint32_t out_l1 = local_l1 + WORDS;

    for (uint32_t h = 0; h < H; ++h) {
        noc_async_read(idx.get_noc_addr(h, 0), tiles_l1 + h * U16_TILE, U16_TILE);
    }
    noc_async_read(mask.get_noc_addr(0, 0), mask_l1, MASK_BYTES);
    noc_async_read(pin.get_noc_addr(0, 0), pin_l1, 32);
    noc_async_read_barrier();

    volatile tt_l1_ptr uint16_t* tiles = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(tiles_l1);
    volatile tt_l1_ptr int32_t* in_group = reinterpret_cast<volatile tt_l1_ptr int32_t*>(mask_l1);
    const uint32_t my_rank = *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(pin_l1);
    volatile tt_l1_ptr uint32_t* total = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(total_l1);
    volatile tt_l1_ptr uint32_t* local = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(local_l1);
    volatile tt_l1_ptr uint32_t* out = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(out_l1);

    for (uint32_t e = 0; e < E; ++e) {
        total[e] = 0;
        local[e] = 0;
    }
    // uint16 tile: four 16 x 16 faces, (fy, fx) at element fy * 512 + fx * 256; column j < 16 is face column 0
    for (uint32_t h = 0; h < H; ++h) {
        volatile tt_l1_ptr uint16_t* tile = tiles + h * (U16_TILE / 2);
        for (uint32_t t = 0; t < REAL_ROWS; ++t) {
            volatile tt_l1_ptr uint16_t* row = tile + ((t >> 4) << 9) + ((t & 15) << 4);
            for (uint32_t j = 0; j < K; ++j) {
                const uint32_t e = row[j];
                if (e < E && in_group[e] >= 0) {
                    total[e] += 1;
                    if (h < my_rank) {
                        local[e] += 1;
                    }
                }
            }
        }
    }
    for (uint32_t e = 0; e < E; ++e) {
        out[e] = total[e];
    }
    noc_async_write(out_l1, totals.get_noc_addr(0, 0), WORDS);
    noc_async_write_barrier();

    // the region of each expert of a chip starts at a tile boundary after the chip's experts before it
    for (uint32_t g = 0; g < E / EPC; ++g) {
        uint32_t prefix = 0;
        for (uint32_t i = 0; i < EPC; ++i) {
            const uint32_t e = g * EPC + i;
            const uint32_t n = total[e];
            out[e] = prefix;
            prefix += (n + 31) / 32 * 32;
        }
    }
    noc_async_write(out_l1, region.get_noc_addr(0, 0), WORDS);
    noc_async_write_barrier();

    for (uint32_t e = 0; e < E; ++e) {
        out[e] += local[e];
    }
    noc_async_write(out_l1, offsets.get_noc_addr(0, 0), WORDS);
    noc_async_write_barrier();
}
