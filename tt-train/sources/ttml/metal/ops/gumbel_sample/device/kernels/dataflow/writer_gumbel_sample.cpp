// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Fused Gumbel-max sampling, reduction half -- entirely on device, spread across the whole grid.
//
// The work unit is a TILE, so a token row's vocabulary MAY be split across cores. The price of
// tile units is that the argmax then becomes a cross-core reduction, arranged here so the common
// case never pays for it:
//
//   * A FULLY LOCAL row -- every tile inside this core's run -- is reduced and written right here.
//     In prefill nearly every row is like that, so the exchange below is almost never used.
//   * A split row is merged by the row's OWNER, which throughout this file means exactly one
//     thing: the core holding the row's first tile. The split hands each core one contiguous tile
//     range, so a core can hold a foreign shard only of its FIRST row (at most one record to send)
//     and can own a split row only as its LAST (at most one merge to run). Senders NOC-write their
//     record into a host-assigned slot in the owner's L1 and bump the owner's semaphore; the owner
//     waits for exactly its shard count, folds the records into its accumulators, and writes the
//     row. Every split row merges on a different core, in parallel -- there is no global
//     rendezvous core to serialize on.
//
// Comparison is on raw FP32 bit patterns via float32_greater: the data-movement RISCs have no FPU.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/numeric/float32.h"
#include "tt-train/sources/ttml/metal/common/dataflow_utils.hpp"

namespace {

constexpr uint32_t kTileHeight = 32U;
constexpr uint32_t kTileWidth = 32U;
constexpr uint32_t kFaceHeight = 16U;
constexpr uint32_t kFaceWidth = 16U;
constexpr uint32_t kFaceSize = kFaceHeight * kFaceWidth;

// A boundary record: [valid, row_id, 32 max bit-patterns, 32 indices], padded to a NOC-friendly
// multiple of 16 bytes. The merge reads only the maxima and indices; valid and row_id are
// watcher/debug breadcrumbs (the split geometry already fixes which row a record belongs to).
// The records CB holds the receive slots for the one row this core may own, then one staging slot
// for the record it may send.
constexpr uint32_t kRecordStrideU32 = 72U;  // 66 used, padded to 288 bytes
constexpr uint32_t kRecordBytes = kRecordStrideU32 * sizeof(uint32_t);

// Bytes per staged output value: a NOC write needs its L1 source aligned, so each token id gets its
// own slot rather than sitting packed 4 bytes apart.
constexpr uint32_t kOutputSlotBytes = 32U;

}  // namespace

void kernel_main() {
    uint32_t rt_idx = 0U;
    const uint32_t output_address = get_arg_val<uint32_t>(rt_idx++);
    const uint32_t num_tiles = get_arg_val<uint32_t>(rt_idx++);
    const uint32_t start_tile = get_arg_val<uint32_t>(rt_idx++);
    // Base address of the positions tensor, 0 when absent; emitted in both modes so the host can
    // patch the slot unconditionally.
    const uint32_t positions_address = get_arg_val<uint32_t>(rt_idx++);
    // Merge routing, host-derived from the same work split that produced num_tiles/start_tile:
    // where this core's first-row shard goes (meaningful only when that row began on an earlier
    // core), and how many foreign shards of its last row to wait for (0 when that row ends here).
    const uint32_t owner_phys_x = get_arg_val<uint32_t>(rt_idx++);
    const uint32_t owner_phys_y = get_arg_val<uint32_t>(rt_idx++);
    const uint32_t send_slot = get_arg_val<uint32_t>(rt_idx++);
    const uint32_t expected_shards = get_arg_val<uint32_t>(rt_idx++);
    // Logical token count -- a RUNTIME arg in BOTH modes, and the only form it exists in here. In
    // position mode it bounds the clamp in target_row_of and must not enter the program-cache key
    // (the hash normalizes the token dimension away); in non-position mode it bounds the row scan
    // and the output page math, where it is only multiplied and compared, so nothing is lost by
    // not having it constexpr. Ht below stays compile-time instead: it sits in / and %, which fold
    // to shift/multiply only for a constant.
    const uint32_t logical_tokens = get_arg_val<uint32_t>(rt_idx++);

    constexpr uint32_t cb_scores_idx = tt::CBIndex::c_2;
    constexpr uint32_t cb_output_staging_idx = tt::CBIndex::c_3;
    constexpr uint32_t cb_records_idx = tt::CBIndex::c_4;
    constexpr uint32_t cb_positions_idx = tt::CBIndex::c_6;

    constexpr uint32_t Wt = get_compile_time_arg_val(0);
    constexpr uint32_t logical_vocab = get_compile_time_arg_val(1);
    constexpr uint32_t Ht = get_compile_time_arg_val(2);
    constexpr uint32_t reduction_sem_id = get_compile_time_arg_val(3);
    // Receive-slot count in the records CB (the grid-wide worst-case shard fan-in for one row);
    // the outgoing record is staged in the slot just past them.
    constexpr uint32_t max_foreign_shards = get_compile_time_arg_val(4);
    // Unused since positions moved to local-window staging; the slot is kept so the compile-time
    // arg indices (and the TensorAccessorArgs offset chain below) stay stable.
    [[maybe_unused]] constexpr uint32_t num_entries = get_compile_time_arg_val(5);

    constexpr auto output_args = TensorAccessorArgs<6>();
    constexpr auto positions_args = TensorAccessorArgs<output_args.next_compile_time_args_offset()>();
    // Appended past the accessor chain so the hand-numbered offsets above never move; the host
    // appends it in this same position after its accessor appends.
    constexpr bool do_positions = get_compile_time_arg_val(positions_args.next_compile_time_args_offset()) != 0;
    const auto output_address_generator = TensorAccessor(output_args, output_address);

    const uint32_t staging_address = get_write_ptr(cb_output_staging_idx);
    const uint32_t records_base = get_write_ptr(cb_records_idx);

    // Stage the entry WINDOW this core's tile run touches, exactly as the reader does -- the rows
    // this kernel scans (pass 1) and the one row it may merge (pass 2's owned_row, the run's LAST
    // entry) all lie inside [start_tile / Wt, (start_tile + num_tiles - 1) / Wt]. Staging, slot
    // addressing and the position clamp are single-sourced in PositionWindow (dataflow_utils.hpp).
    //
    // The read is free here -- the next thing this kernel does is block on cb_wait_front(scores),
    // which cannot clear until the reader has already fetched logits from DRAM. BRISC issues reads
    // exactly as NCRISC does (brisc.cc runs noc_init + noc_local_state_init), on its own NOC, and
    // noc_async_read_barrier tracks a counter independent of the write/semaphore rendezvous below.
    PositionWindow positions{};
    if constexpr (do_positions) {
        const auto positions_address_generator = TensorAccessor(positions_args, positions_address);
        const uint32_t first_entry = start_tile / Wt;
        const uint32_t last_entry = (start_tile + num_tiles - 1U) / Wt;
        positions = stage_position_window(
            cb_positions_idx, positions_address_generator, first_entry, last_entry - first_entry + 1U);
    }

    // Only the low 5 bits are consumed here; the reader consumes the high bits (clamped >> 5) of
    // the SAME clamped value -- see PositionWindow::clamped_position for the shared clamp and its
    // rationale.
    auto target_row_of = [&](uint32_t entry) -> uint32_t {
        return positions.clamped_position(entry, logical_tokens) & (kTileHeight - 1U);
    };

    uint32_t max_values[kTileHeight];
    uint32_t arg_max[kTileHeight];

    // How many of a tile row's 32 rows are real tokens. This bounds the SCAN, not just the
    // write-out: decode produces one token per step, so 31 of 32 rows are padding and scanning them
    // anyway costs 32x.
    auto valid_rows_of = [&](uint32_t tile_row) -> uint32_t {
        if constexpr (do_positions) {
            // One row per batch entry, always real: target_row_of clamps every position against
            // the logical token count (mirroring the reader), so the selected row is never
            // padding. The host does NOT validate positions -- they live in device memory.
            return 1U;
        }
        const uint32_t first_token = (tile_row % Ht) * kTileHeight;
        if (first_token >= logical_tokens) {
            return 0U;
        }
        const uint32_t remaining = logical_tokens - first_token;
        return (remaining < kTileHeight) ? remaining : kTileHeight;
    };

    // Emit a group's token ids. Output pages run row-major over [B, 1, tokens] normally, and over
    // [B, 1, 1] -- one page per batch entry -- when positions selected a single row each.
    //
    // Writes are staged through the output CB's 32 NOC-aligned slots used as a RING and left in
    // flight: a barrier is paid only when a slot is about to be recycled (its previous write may
    // still be outbound) and once at kernel end -- never per row. This matters exactly when a core
    // owns many one-page rows (position mode, and decode's single valid row): at large batches a
    // core owns ~B/num_cores rows, and a per-row barrier would serialize that many NOC round-trips
    // into pass 1. send_shard's own barrier is global, so it can only OVER-flush ring slots; the
    // ring never under-waits.
    uint32_t staging_cursor = 0U;
    auto stage_and_write = [&](uint32_t page, uint32_t value) {
        if (staging_cursor == kTileHeight) {
            // Every slot may still have a write outbound; drain them all before recycling slot 0.
            noc_async_write_barrier();
            staging_cursor = 0U;
        }
        const uint32_t slot = staging_address + staging_cursor * kOutputSlotBytes;
        *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(slot) = value;
        noc_async_write_page(page, output_address_generator, slot);
        ++staging_cursor;
    };

    auto write_row = [&](uint32_t tile_row, uint32_t valid_rows) {
        if constexpr (do_positions) {
            stage_and_write(tile_row, arg_max[target_row_of(tile_row)]);
            return;
        }
        const uint32_t page_base = (tile_row / Ht) * logical_tokens + (tile_row % Ht) * kTileHeight;
        for (uint32_t h = 0U; h < valid_rows; ++h) {
            stage_and_write(page_base + h, arg_max[h]);
        }
    };

    // Ship this core's shard of its first row to that row's owner. All 32 slots travel verbatim:
    // rows this core never scanned are still NEG_INF from reset_accumulators, so the merge leaves
    // them alone whichever mode is compiled. Fires even for all-padding rows: the owner's expected
    // count is derived from the split geometry alone, so a withheld record would deadlock it.
    //
    // The records CB sits at the same L1 address on every core, so the local base doubles as the
    // remote destination base.
    auto send_shard = [&](uint32_t tile_row) {
        const uint32_t staging = records_base + max_foreign_shards * kRecordBytes;
        auto* rec = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(staging);
        // Watcher/debug breadcrumbs only -- the merge never reads them (the split geometry admits
        // exactly one row per owner, so no row-id matching is needed).
        rec[0] = 1U;
        rec[1] = tile_row;
        for (uint32_t h = 0U; h < kTileHeight; ++h) {
            rec[2U + h] = max_values[h];
            rec[2U + kTileHeight + h] = arg_max[h];
        }
        // Record first, then the increment: the write barrier orders them, so the owner's
        // semaphore never counts a record that has not landed.
        noc_async_write(
            staging,
            get_noc_addr(owner_phys_x, owner_phys_y, records_base + send_slot * kRecordBytes),
            kRecordBytes);
        noc_async_write_barrier();
        noc_semaphore_inc(get_noc_addr(owner_phys_x, owner_phys_y, get_semaphore(reduction_sem_id)), 1U);
    };

    auto finish_row = [&](uint32_t tile_row, uint32_t valid_rows) {
        const uint32_t row_first = tile_row * Wt;
        // Fully local row: every tile was scanned here, so the write-out completes here too.
        if (row_first >= start_tile && row_first + Wt <= start_tile + num_tiles) {
            if (valid_rows != 0U) {
                write_row(tile_row, valid_rows);
            }
            return;
        }
        if (row_first < start_tile) {
            // A shard of a row that began on an earlier core -- necessarily this core's first row.
            send_shard(tile_row);
            return;
        }
        // The row starts here but spills onto later cores -- necessarily this core's LAST row, so
        // leaving the accumulators untouched hands them straight to the merge in pass 2.
    };

    auto reset_accumulators = [&]() {
        for (uint32_t h = 0U; h < kTileHeight; ++h) {
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
        const uint32_t tile_col_base = (global_tile % Wt) * kTileWidth;

        // Rows worth scanning in this tile. Normally that is every real token row; with positions
        // it is the single row the entry asked for, and the accumulator it lands in is indexed by
        // that same row so write_row and the merge need no extra bookkeeping.
        const uint32_t row_begin = do_positions ? target_row_of(current_row) : 0U;
        const uint32_t row_end = do_positions ? row_begin + 1U : current_valid;

        for (uint32_t face = 0U; face < 4U; ++face) {
            const uint32_t face_row_base = (face >= 2U) ? kFaceHeight : 0U;
            const uint32_t face_col_base = (face & 1U) ? kFaceWidth : 0U;
            const uint32_t global_col_base = tile_col_base + face_col_base;

            // Columns past the logical vocab are tile padding; so are rows outside [begin, end).
            if (global_col_base >= logical_vocab) {
                continue;
            }
            const uint32_t first_row = (row_begin > face_row_base) ? row_begin : face_row_base;
            const uint32_t face_row_end = face_row_base + kFaceHeight;
            const uint32_t last_row = (row_end < face_row_end) ? row_end : face_row_end;
            if (first_row >= last_row) {
                continue;
            }
            const uint32_t cols_left = logical_vocab - global_col_base;
            const uint32_t cols_to_scan = (cols_left < kFaceWidth) ? cols_left : kFaceWidth;

            const uint32_t face_offset = face * kFaceSize;
            for (uint32_t row_in_tile = first_row; row_in_tile < last_row; ++row_in_tile) {
                uint32_t running_max = max_values[row_in_tile];
                uint32_t running_arg = arg_max[row_in_tile];

                const uint32_t row_offset = face_offset + (row_in_tile - face_row_base) * kFaceWidth;
                for (uint32_t cc = 0U; cc < cols_to_scan; ++cc) {
                    const uint32_t value = tile_ptr[row_offset + cc];
                    // Strict greater, scanning columns in increasing global order, so ties keep the
                    // lowest index -- matching ttnn::argmax's tie-break.
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

        // The accumulators still hold this core's local shard: finish_row deferred exactly this row,
        // and it is the run's last, so no reset ran after it. Every received record is a shard of
        // this same row -- the split geometry admits no other sender -- so no row-id matching is
        // needed.
        for (uint32_t s = 0U; s < expected_shards; ++s) {
            auto* rec = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(records_base + s * kRecordBytes);
            for (uint32_t h = 0U; h < kTileHeight; ++h) {
                const uint32_t v = rec[2U + h];
                const uint32_t i = rec[2U + kTileHeight + h];
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

    // Drain the output writes still in flight in write_row's slot ring. Unconditional: with
    // nothing outstanding a barrier is a cheap counter check, and the kernel must not return
    // while NOC writes are still outbound.
    noc_async_write_barrier();
}
