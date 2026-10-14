// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Fused Gumbel-max sampling, reduction half. The work unit is a TILE, so a token row's vocabulary
// may be split across cores and the argmax becomes a cross-core reduction:
//
//   * A fully local row is reduced and written right here (the common case).
//   * A split row is merged by its OWNER -- the core holding the row's first tile. The split hands
//     each core one contiguous range, so a core sends at most one record (a shard of its FIRST
//     row) and runs at most one merge (its LAST row). Senders NOC-write into a host-assigned slot
//     in the owner's L1 and bump its semaphore; the owner waits for its exact shard count.
//
// Comparison is on raw FP32 bit patterns via float32_greater: data-movement RISCs have no FPU.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/numeric/float32.h"
#include "position_window.hpp"
#include "tt-train/sources/ttml/metal/common/dataflow_utils.hpp"

namespace {

// A boundary record: [valid, row_id, 32 max bit-patterns, 32 indices], NOC-padded. The merge reads
// only the maxima and indices; valid/row_id are watcher breadcrumbs. The records CB holds the
// receive slots, then one staging slot for the outgoing record.
constexpr uint32_t kRecordStrideU32 = 72U;  // 66 used, padded to 288 bytes
constexpr uint32_t kRecordBytes = kRecordStrideU32 * sizeof(uint32_t);

// Each staged output value gets its own NOC-aligned slot.
constexpr uint32_t kOutputSlotBytes = 32U;

}  // namespace

void kernel_main() {
    using namespace tt::constants;  // TILE_HEIGHT / TILE_WIDTH / FACE_HEIGHT / FACE_WIDTH / FACE_HW

    uint32_t rt_idx = 0U;
    const uint32_t output_address = get_arg_val<uint32_t>(rt_idx++);
    const uint32_t num_tiles = get_arg_val<uint32_t>(rt_idx++);
    const uint32_t start_tile = get_arg_val<uint32_t>(rt_idx++);
    const uint32_t positions_address = get_arg_val<uint32_t>(rt_idx++);  // 0 when absent
    // Merge routing, host-derived from the work split: where this core's first-row shard goes, and
    // how many foreign shards of its last row to wait for (0 when that row ends here).
    const uint32_t owner_phys_x = get_arg_val<uint32_t>(rt_idx++);
    const uint32_t owner_phys_y = get_arg_val<uint32_t>(rt_idx++);
    const uint32_t send_slot = get_arg_val<uint32_t>(rt_idx++);
    const uint32_t expected_shards = get_arg_val<uint32_t>(rt_idx++);
    // Runtime (not compile-time) so the program-cache key stays independent of the token dimension.
    const uint32_t logical_tokens = get_arg_val<uint32_t>(rt_idx++);

    constexpr uint32_t cb_scores_idx = tt::CBIndex::c_2;
    constexpr uint32_t cb_output_staging_idx = tt::CBIndex::c_3;
    constexpr uint32_t cb_records_idx = tt::CBIndex::c_4;
    constexpr uint32_t cb_positions_idx = tt::CBIndex::c_6;

    constexpr uint32_t Wt = get_compile_time_arg_val(0);
    constexpr uint32_t logical_vocab = get_compile_time_arg_val(1);
    constexpr uint32_t Ht = get_compile_time_arg_val(2);
    constexpr uint32_t reduction_sem_id = get_compile_time_arg_val(3);
    // Receive-slot count in the records CB (worst-case shard fan-in for one row).
    constexpr uint32_t max_foreign_shards = get_compile_time_arg_val(4);

    constexpr auto output_args = TensorAccessorArgs<5>();
    constexpr auto positions_args = TensorAccessorArgs<output_args.next_compile_time_args_offset()>();
    // Appended past the accessor chain so the hand-numbered offsets above never move.
    constexpr bool do_positions = get_compile_time_arg_val(positions_args.next_compile_time_args_offset()) != 0;
    const auto output_address_generator = TensorAccessor(output_args, output_address);

    const uint32_t staging_address = get_write_ptr(cb_output_staging_idx);
    const uint32_t records_base = get_write_ptr(cb_records_idx);

    // Stage the entry window this core's run touches, exactly as the reader does (single-sourced
    // in position_window.hpp). Effectively free: the kernel next blocks on cb_wait_front(scores).
    PositionWindow positions{};
    if constexpr (do_positions) {
        const auto positions_address_generator = TensorAccessor(positions_args, positions_address);
        positions = stage_position_window(cb_positions_idx, positions_address_generator, start_tile, num_tiles, Wt);
    }

    // Low 5 bits of the clamped position; the reader consumes the high bits of the SAME value.
    auto target_row_of = [&](uint32_t entry) -> uint32_t {
        return positions.clamped_position(entry, logical_tokens) & (TILE_HEIGHT - 1U);
    };

    uint32_t max_values[TILE_HEIGHT];
    uint32_t arg_max[TILE_HEIGHT];

    // Real-token rows in a tile row; bounds the scan as well as the write-out (in decode 31 of 32
    // rows are padding).
    auto valid_rows_of = [&](uint32_t tile_row) -> uint32_t {
        if constexpr (do_positions) {
            return 1U;  // one clamped row per batch entry, never padding
        }
        const uint32_t first_token = (tile_row % Ht) * TILE_HEIGHT;
        if (first_token >= logical_tokens) {
            return 0U;
        }
        const uint32_t remaining = logical_tokens - first_token;
        return (remaining < TILE_HEIGHT) ? remaining : TILE_HEIGHT;
    };

    // Output writes are staged through 32 NOC-aligned slots used as a ring and left in flight: a
    // barrier is paid only when a slot is recycled and once at kernel end, never per row.
    uint32_t staging_cursor = 0U;
    auto stage_and_write = [&](uint32_t page, uint32_t value) {
        if (staging_cursor == TILE_HEIGHT) {
            noc_async_write_barrier();
            staging_cursor = 0U;
        }
        const uint32_t slot = staging_address + staging_cursor * kOutputSlotBytes;
        *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(slot) = value;
        noc_async_write_page(page, output_address_generator, slot);
        ++staging_cursor;
    };

    // Output pages run row-major over [B, 1, tokens], or one page per entry in position mode.
    auto write_row = [&](uint32_t tile_row, uint32_t valid_rows) {
        if constexpr (do_positions) {
            stage_and_write(tile_row, arg_max[target_row_of(tile_row)]);
            return;
        }
        const uint32_t page_base = (tile_row / Ht) * logical_tokens + (tile_row % Ht) * TILE_HEIGHT;
        for (uint32_t h = 0U; h < valid_rows; ++h) {
            stage_and_write(page_base + h, arg_max[h]);
        }
    };

    // Ship this core's shard of its first row to that row's owner. All 32 slots travel verbatim
    // (unscanned rows are still NEG_INF, so the merge ignores them). Fires even for all-padding
    // rows: the owner's expected count comes from the split geometry, so withholding would
    // deadlock it. The records CB sits at the same L1 address on every core.
    auto send_shard = [&](uint32_t tile_row) {
        const uint32_t staging = records_base + max_foreign_shards * kRecordBytes;
        auto* rec = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(staging);
        rec[0] = 1U;  // watcher breadcrumbs; the merge never reads them
        rec[1] = tile_row;
        for (uint32_t h = 0U; h < TILE_HEIGHT; ++h) {
            rec[2U + h] = max_values[h];
            rec[2U + TILE_HEIGHT + h] = arg_max[h];
        }
        // Record first, then the increment: the barrier orders them, so the owner's semaphore
        // never counts a record that has not landed.
        noc_async_write(
            staging,
            get_noc_addr(owner_phys_x, owner_phys_y, records_base + send_slot * kRecordBytes),
            kRecordBytes);
        noc_async_write_barrier();
        noc_semaphore_inc(get_noc_addr(owner_phys_x, owner_phys_y, get_semaphore(reduction_sem_id)), 1U);
    };

    auto finish_row = [&](uint32_t tile_row, uint32_t valid_rows) {
        const uint32_t row_first = tile_row * Wt;
        // Fully local row: write it out here.
        if (row_first >= start_tile && row_first + Wt <= start_tile + num_tiles) {
            if (valid_rows != 0U) {
                write_row(tile_row, valid_rows);
            }
            return;
        }
        if (row_first < start_tile) {
            // Shard of a row that began on an earlier core -- necessarily this core's first row.
            send_shard(tile_row);
            return;
        }
        // Starts here but spills onto later cores -- necessarily the LAST row; the accumulators
        // are left for pass 2's merge.
    };

    auto reset_accumulators = [&]() {
        for (uint32_t h = 0U; h < TILE_HEIGHT; ++h) {
            max_values[h] = NEG_INF_FLOAT32;
            arg_max[h] = 0U;
        }
    };

    // ---- pass 1: reduce the scores as they stream past ----
    uint32_t current_row = start_tile / Wt;
    uint32_t current_valid = valid_rows_of(current_row);
    reset_accumulators();

    for (uint32_t t = 0U; t < num_tiles; ++t) {
        const uint32_t global_tile = start_tile + t;
        const uint32_t tile_row = global_tile / Wt;

        if (tile_row != current_row) {
            finish_row(current_row, current_valid);
            current_row = tile_row;
            current_valid = valid_rows_of(current_row);
            reset_accumulators();
        }

        cb_wait_front(cb_scores_idx, onetile);
        auto* tile_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_read_ptr(cb_scores_idx));
        const uint32_t tile_col_base = (global_tile % Wt) * TILE_WIDTH;

        // With positions only the entry's single row is scanned; its accumulator is indexed by
        // that same row, so write_row and the merge need no extra bookkeeping.
        const uint32_t row_begin = do_positions ? target_row_of(current_row) : 0U;
        const uint32_t row_end = do_positions ? row_begin + 1U : current_valid;

        // Walked face-by-face (not via get_tilized_idx): the face geometry lives in the loop
        // bounds instead of per-element math in the kernel's hottest loop, and the two `continue`s
        // reject a whole face of vocab padding or out-of-window rows with one comparison.
        for (uint32_t face = 0U; face < 4U; ++face) {
            const uint32_t face_row_base = (face >= 2U) ? FACE_HEIGHT : 0U;
            const uint32_t face_col_base = (face & 1U) ? FACE_WIDTH : 0U;
            const uint32_t global_col_base = tile_col_base + face_col_base;

            if (global_col_base >= logical_vocab) {
                continue;
            }
            const uint32_t first_row = (row_begin > face_row_base) ? row_begin : face_row_base;
            const uint32_t face_row_end = face_row_base + FACE_HEIGHT;
            const uint32_t last_row = (row_end < face_row_end) ? row_end : face_row_end;
            if (first_row >= last_row) {
                continue;
            }
            const uint32_t cols_left = logical_vocab - global_col_base;
            const uint32_t cols_to_scan = (cols_left < FACE_WIDTH) ? cols_left : FACE_WIDTH;

            const uint32_t face_offset = face * FACE_HW;
            for (uint32_t row_in_tile = first_row; row_in_tile < last_row; ++row_in_tile) {
                uint32_t running_max = max_values[row_in_tile];
                uint32_t running_arg = arg_max[row_in_tile];

                const uint32_t row_offset = face_offset + (row_in_tile - face_row_base) * FACE_WIDTH;
                for (uint32_t cc = 0U; cc < cols_to_scan; ++cc) {
                    const uint32_t value = tile_ptr[row_offset + cc];
                    // Strict greater, increasing column order: ties keep the lowest index,
                    // matching ttnn::argmax.
                    if (float32_greater(value, running_max)) {
                        running_max = value;
                        running_arg = global_col_base + cc;
                    }
                }
                max_values[row_in_tile] = running_max;
                arg_max[row_in_tile] = running_arg;
            }
        }

        cb_pop_front(cb_scores_idx, onetile);
    }

    finish_row(current_row, current_valid);

    // ---- pass 2: merge the foreign shards of the one row this core owns but did not finish ----
    if (expected_shards > 0U) {
        auto* sem_ptr = get_sem_ptr(reduction_sem_id);
        noc_semaphore_wait(sem_ptr, expected_shards);
        noc_semaphore_set(sem_ptr, 0U);  // re-arm for the next dispatch of this cached program

        // The accumulators still hold the local shard (finish_row deferred exactly this row, the
        // run's last). Every record is a shard of this same row, so no row-id matching is needed.
        for (uint32_t s = 0U; s < expected_shards; ++s) {
            auto* rec = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(records_base + s * kRecordBytes);
            for (uint32_t h = 0U; h < TILE_HEIGHT; ++h) {
                const uint32_t v = rec[2U + h];
                const uint32_t i = rec[2U + TILE_HEIGHT + h];
                // Ties keep the lower index, matching the in-row scan.
                if (float32_greater(v, max_values[h]) || (v == max_values[h] && i < arg_max[h])) {
                    max_values[h] = v;
                    arg_max[h] = i;
                }
            }
        }

        const uint32_t owned_row = (start_tile + num_tiles - 1U) / Wt;
        const uint32_t valid = valid_rows_of(owned_row);
        if (valid != 0U) {
            write_row(owned_row, valid);
        }
    }

    // Drain the write ring; the kernel must not return with NOC writes outbound.
    noc_async_write_barrier();
}
