// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Streams logits (and, when present, the padding mask) for a contiguous run of TILES.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "position_window.hpp"
#include "tt-train/sources/ttml/metal/common/dataflow_utils.hpp"

void kernel_main() {
    uint32_t rt_idx = 0U;
    const uint32_t logits_address = get_arg_val<uint32_t>(rt_idx++);
    const uint32_t mask_address = get_arg_val<uint32_t>(rt_idx++);
    const uint32_t num_tiles = get_arg_val<uint32_t>(rt_idx++);
    const uint32_t start_tile = get_arg_val<uint32_t>(rt_idx++);
    // Ht, positions address, logical token count and mask stride are RUNTIME args so one cached
    // program serves every prompt length and both mask shapes (a compile-time token dimension made
    // every distinct prompt length a fresh ~6 s JIT build). All are read unconditionally so the
    // runtime-arg layout is identical in every mode and the host can patch the slots on every
    // dispatch.
    const uint32_t Ht = get_arg_val<uint32_t>(rt_idx++);
    const uint32_t positions_address = get_arg_val<uint32_t>(rt_idx++);  // 0 when absent
    const uint32_t logical_tokens = get_arg_val<uint32_t>(rt_idx++);
    // 0 for a shared [1, 1, 1, V] mask, Wt for a per-entry [B, 1, 1, V] mask.
    const uint32_t mask_entry_stride = get_arg_val<uint32_t>(rt_idx++);

    constexpr uint32_t cb_logits_idx = tt::CBIndex::c_0;
    constexpr uint32_t cb_mask_idx = tt::CBIndex::c_1;
    constexpr uint32_t cb_positions_idx = tt::CBIndex::c_5;

    constexpr uint32_t block_size = get_compile_time_arg_val(0);
    constexpr uint32_t Wt = get_compile_time_arg_val(1);

    constexpr auto logits_args = TensorAccessorArgs<2>();
    constexpr auto mask_args = TensorAccessorArgs<logits_args.next_compile_time_args_offset()>();
    constexpr auto positions_args = TensorAccessorArgs<mask_args.next_compile_time_args_offset()>();
    // Mode flags ride past the accessor chain (chained offsets, matching the host's append order),
    // so the hand-numbered offsets above never move when a flag is added or removed.
    constexpr bool do_logits_mask = get_compile_time_arg_val(positions_args.next_compile_time_args_offset()) != 0;
    constexpr bool do_positions = get_compile_time_arg_val(positions_args.next_compile_time_args_offset() + 1) != 0;
    const auto logits_address_generator = TensorAccessor(logits_args, logits_address);
    const auto mask_address_generator = TensorAccessor(mask_args, mask_address);

    const uint32_t logits_tile_bytes = get_tile_size(cb_logits_idx);

    // Stage the entry window this core's run touches (see position_window.hpp; the writer stages
    // the identical window). Cannot be deferred: the first logits page address depends on it.
    PositionWindow positions{};
    if constexpr (do_positions) {
        const auto positions_address_generator = TensorAccessor(positions_args, positions_address);
        positions = stage_position_window(cb_positions_idx, positions_address_generator, start_tile, num_tiles, Wt);
    }

    // With positions supplied the loop indices are VIRTUAL -- one tile row per batch entry (entry
    // vt / Wt, column vt % Wt) -- and this maps them to the real page holding the entry's position.
    auto source_page = [&](uint32_t virtual_tile) -> uint32_t {
        if constexpr (do_positions) {
            const uint32_t entry = virtual_tile / Wt;
            const uint32_t column = virtual_tile - entry * Wt;
            // Clamped BEFORE the >> 5 / & 31 bit-field split shared with the writer (see
            // PositionWindow::clamped_position). No separate Ht clamp is needed: validation pins
            // the padded token dim to round_up(logical_tokens, 32).
            const uint32_t tile_row = positions.clamped_position(entry, logical_tokens) >> 5U;
            return (entry * Ht + tile_row) * Wt + column;
        } else {
            return virtual_tile;
        }
    };

    // Reader, compute and writer derive `current` identically and stay in lockstep.
    for (uint32_t t = 0U; t < num_tiles; t += block_size) {
        const uint32_t remaining = num_tiles - t;
        const uint32_t current = (remaining < block_size) ? remaining : block_size;

        if constexpr (do_positions) {
            // Virtual tiles jump at entry boundaries, so pages are issued one at a time.
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
            // Mask page = entry * stride + column: stride 0 shares one tile row across entries,
            // stride Wt gives each entry its own row. Token dim is 1 either way, so the compute
            // kernel's row broadcast is unchanged.
            cb_reserve_back(cb_mask_idx, current);
            uint32_t l1_addr = get_write_ptr(cb_mask_idx);
            for (uint32_t k = 0U; k < current; ++k) {
                const uint32_t global_tile = start_tile + t + k;
                const uint32_t column = global_tile % Wt;
                // In position mode the tile space is one virtual row per entry; otherwise each
                // entry owns Ht consecutive tile rows.
                const uint32_t entry = do_positions ? (global_tile / Wt) : (global_tile / (Ht * Wt));
                noc_async_read_page(entry * mask_entry_stride + column, mask_address_generator, l1_addr);
                l1_addr += logits_tile_bytes;
            }
            noc_async_read_barrier();
            cb_push_back(cb_mask_idx, current);
        }
    }
}
