// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Streams logits (and, when present, the padding mask) for a contiguous run of TILES.

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
    // Logical token count, for range-clamping positions below. A RUNTIME arg for the same reason Ht
    // is: it derives from the token dimension, which the program hash normalizes away in position
    // mode -- baking it in would make every prompt length a fresh JIT build. Read unconditionally
    // so the runtime-arg layout is identical in both modes.
    const uint32_t logical_tokens = get_arg_val<uint32_t>(rt_idx++);

    constexpr uint32_t cb_logits_idx = tt::CBIndex::c_0;
    constexpr uint32_t cb_mask_idx = tt::CBIndex::c_1;
    constexpr uint32_t cb_positions_idx = tt::CBIndex::c_5;

    constexpr uint32_t block_size = get_compile_time_arg_val(0);
    constexpr uint32_t Wt = get_compile_time_arg_val(1);
    // Unused since positions moved to local-window staging; the slot is kept so the compile-time
    // arg indices (and the TensorAccessorArgs offset chain below) stay stable.
    [[maybe_unused]] constexpr uint32_t num_entries = get_compile_time_arg_val(2);

    constexpr auto logits_args = TensorAccessorArgs<3>();
    constexpr auto mask_args = TensorAccessorArgs<logits_args.next_compile_time_args_offset()>();
    constexpr auto positions_args = TensorAccessorArgs<mask_args.next_compile_time_args_offset()>();
    // Mode flags ride at the END of the compile-time args, past the accessor chain, so the
    // hand-numbered offsets above never move when a flag is added or removed -- the index here is
    // chained, not hard-coded, and the host appends in this same order after its accessor appends.
    constexpr bool do_logits_mask = get_compile_time_arg_val(positions_args.next_compile_time_args_offset()) != 0;
    constexpr bool do_positions = get_compile_time_arg_val(positions_args.next_compile_time_args_offset() + 1) != 0;
    const auto logits_address_generator = TensorAccessor(logits_args, logits_address);
    const auto mask_address_generator = TensorAccessor(mask_args, mask_address);

    const uint32_t logits_tile_bytes = get_tile_size(cb_logits_idx);

    // Stage the entry WINDOW this core's tile run touches -- and only that window; every entry
    // this kernel dereferences is start_tile / Wt ..= (start_tile + num_tiles - 1) / Wt by
    // construction of source_page below. It cannot be deferred: the very first logits page address
    // depends on it. The staging, the slot addressing and the position clamp are single-sourced in
    // PositionWindow (dataflow_utils.hpp); the writer stages the identical window and consumes the
    // complementary bit field of the same clamped value.
    PositionWindow positions{};
    if constexpr (do_positions) {
        const auto positions_address_generator = TensorAccessor(positions_args, positions_address);
        const uint32_t first_entry = start_tile / Wt;
        const uint32_t last_entry = (start_tile + num_tiles - 1U) / Wt;
        positions = stage_position_window(
            cb_positions_idx, positions_address_generator, first_entry, last_entry - first_entry + 1U);
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
            // Clamped BEFORE the split into bit fields: this kernel takes clamped >> 5, the writer
            // takes clamped & 31 of the SAME value -- see PositionWindow::clamped_position for the
            // full rationale (the clamp, the padding band, the out-of-bounds case).
            //
            // No separate Ht clamp is needed: validation pins the padded token dim to
            // round_up(logical_tokens, 32), so clamped >> 5 <= Ht - 1 by construction.
            const uint32_t tile_row = positions.clamped_position(entry, logical_tokens) >> 5U;
            return (entry * Ht + tile_row) * Wt + column;
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
