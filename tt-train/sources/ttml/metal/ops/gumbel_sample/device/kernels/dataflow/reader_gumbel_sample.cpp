// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Streams logits (and, when present, the padding mask) for a contiguous run of TILES.
//
// The work unit is one tile, not one 32-token tile row. That distinction is the whole performance
// story of this op: with a row-based split, decode (tokens == 1 => Ht == 1) yields only B_local work
// units, so a handful of cores carried the entire vocabulary while the rest of the grid idled, and
// the fused kernel measured ~2.9x SLOWER than the six separate ttnn ops it replaces -- each of which
// splits by tile across the full grid. Splitting by tile puts this op on the same footing.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "tt-train/sources/ttml/metal/common/dataflow_utils.hpp"

void kernel_main() {
    uint32_t rt_idx = 0U;
    const uint32_t logits_address = get_arg_val<uint32_t>(rt_idx++);
    const uint32_t mask_address = get_arg_val<uint32_t>(rt_idx++);
    const uint32_t num_tiles = get_arg_val<uint32_t>(rt_idx++);
    const uint32_t start_tile = get_arg_val<uint32_t>(rt_idx++);
    // Ht is a RUNTIME arg, in BOTH modes, and that is a performance decision rather than a stylistic
    // one. Baked as a compile-time arg it put the token dimension into the program-cache key, so
    // every distinct prompt length in a rollout was a fresh cache miss -- and each miss is a fresh
    // JIT build of this kernel, measured at ~6 s against ~3 ms for the dispatch itself. It is read
    // unconditionally so the runtime-arg layout is identical in both modes: the host patches this
    // slot on every dispatch, and in non-position mode the slot would otherwise not exist.
    const uint32_t Ht = get_arg_val<uint32_t>(rt_idx++);
    // Base address of the positions tensor, 0 when absent. Emitted in BOTH modes for the same reason
    // Ht is: the host patches this slot unconditionally on every dispatch.
    const uint32_t positions_address = get_arg_val<uint32_t>(rt_idx++);

    constexpr uint32_t cb_logits_idx = tt::CBIndex::c_0;
    constexpr uint32_t cb_mask_idx = tt::CBIndex::c_1;
    constexpr uint32_t cb_positions_idx = tt::CBIndex::c_5;

    constexpr uint32_t block_size = get_compile_time_arg_val(0);
    constexpr uint32_t Wt = get_compile_time_arg_val(1);
    constexpr uint32_t num_entries = get_compile_time_arg_val(2);

#ifdef DO_LOGITS_MASK
    constexpr bool do_logits_mask = true;
#else
    constexpr bool do_logits_mask = false;
#endif

#ifdef DO_POSITIONS
    constexpr bool do_positions = true;
#else
    constexpr bool do_positions = false;
#endif

    constexpr auto logits_args = TensorAccessorArgs<3>();
    constexpr auto mask_args = TensorAccessorArgs<logits_args.next_compile_time_args_offset()>();
    constexpr auto positions_args = TensorAccessorArgs<mask_args.next_compile_time_args_offset()>();
    const auto logits_address_generator = TensorAccessor(logits_args, logits_address);
    const auto mask_address_generator = TensorAccessor(mask_args, mask_address);

    const uint32_t logits_tile_bytes = get_tile_size(cb_logits_idx);

    // Stage the whole local position list into L1 up front. It cannot be deferred: the very first
    // logits page address depends on it.
    //
    // One ALIGNED page per entry rather than four packed bytes -- a DRAM read moves a whole aligned
    // page, and the NOC requires the L1 destination to be congruent with the DRAM address modulo the
    // DRAM alignment (64 B on Blackhole), not L1's own 16 B. The CB base satisfies that, so
    // base + entry * slot does too.
    uint32_t positions_l1_base = 0U;
    uint32_t positions_slot_bytes = 0U;
    if constexpr (do_positions) {
        const auto positions_address_generator = TensorAccessor(positions_args, positions_address);
        positions_slot_bytes = positions_address_generator.get_aligned_page_size();
        positions_l1_base = get_write_ptr(cb_positions_idx);
        uint32_t l1_addr = positions_l1_base;
        for (uint32_t entry = 0U; entry < num_entries; ++entry) {
            noc_async_read_page(entry, positions_address_generator, l1_addr);
            l1_addr += positions_slot_bytes;
        }
        noc_async_read_barrier();
    }

    // With positions supplied, the indices this loop walks are VIRTUAL: one tile row per batch
    // entry rather than Ht of them. Virtual tile vt covers entry vt / Wt at column vt % Wt, and the
    // real page is found by jumping to the tile row holding that entry's position. Consecutive
    // virtual tiles stay contiguous inside an entry but jump at entry boundaries, so the pages are
    // issued one at a time instead of as a run.
    auto source_page = [&](uint32_t virtual_tile) -> uint32_t {
        if constexpr (do_positions) {
            const uint32_t entry = virtual_tile / Wt;
            const uint32_t column = virtual_tile - entry * Wt;
            // volatile so the load cannot be hoisted above the barrier above.
            const uint32_t position =
                *reinterpret_cast<volatile tt_l1_ptr uint32_t *>(positions_l1_base + entry * positions_slot_bytes);
            // Clamp the TILE ROW, not the position. This kernel consumes position >> 5 and the writer
            // consumes position & 31 -- disjoint bit fields -- so bounding the row here cannot
            // desynchronize the pair, and the writer needs no matching change. Clamping before the
            // multiply also contains the uint32 overflow case.
            //
            // The host can no longer range-check positions (they live in device memory), so without
            // this a bad value reads outside the logits buffer entirely: interleaved accessors do no
            // bounds checking, and watcher validates the whole DRAM window rather than the buffer, so
            // it would not be caught there either. With it, the worst case degrades to the
            // already-documented silent-wrong-row class.
            const uint32_t tile_row = position >> 5U;
            const uint32_t clamped_row = (tile_row < Ht) ? tile_row : (Ht - 1U);
            return (entry * Ht + clamped_row) * Wt + column;
        } else {
            return virtual_tile;
        }
    };

    // A core's tile run is arbitrary in length, so the last block may be partial. All three kernels
    // derive `current` the same way from (num_tiles, block_size) and stay in lockstep.
    for (uint32_t t = 0U; t < num_tiles; t += block_size) {
        const uint32_t remaining = num_tiles - t;
        const uint32_t current = (remaining < block_size) ? remaining : block_size;

        if constexpr (do_positions) {
            cb_reserve_back(cb_logits_idx, current);
            uint32_t l1_addr = get_write_ptr(cb_logits_idx);
            for (uint32_t k = 0U; k < current; ++k) {
                noc_async_read_page(source_page(start_tile + t + k), logits_address_generator, l1_addr);
                l1_addr += logits_tile_bytes;
            }
            noc_async_read_barrier();
            cb_push_back(cb_logits_idx, current);
        } else {
            read_tiles_by_row(
                cb_logits_idx, logits_address_generator, start_tile + t, current, logits_tile_bytes, current);
        }

        if constexpr (do_logits_mask) {
            // The mask is [1, 1, 1, V]: one tile row shared by every token row and every batch
            // entry, so a global tile's mask is selected by its COLUMN position alone.
            cb_reserve_back(cb_mask_idx, current);
            uint32_t l1_addr = get_write_ptr(cb_mask_idx);
            for (uint32_t k = 0U; k < current; ++k) {
                const uint32_t mask_tile = (start_tile + t + k) % Wt;
                noc_async_read_page(mask_tile, mask_address_generator, l1_addr);
                l1_addr += logits_tile_bytes;
            }
            noc_async_read_barrier();
            cb_push_back(cb_mask_idx, current);
        }
    }
}
